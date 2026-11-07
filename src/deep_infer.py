"""
deep_infer.py — run the trained Siamese network inside the app, on held-out patches.

This is the bridge between the two halves of the project:
  * Part 1 (classical): NDVI differencing + threshold, scored region-wide on EE.
  * Part 2 (deep):       the Siamese U-Net trained on 256x256 patches (on Colab).

Here we load the trained checkpoint and run it on a region's TEST patches — the
spatial blocks the model never saw during training — and score it with the SAME
Precision/Recall/F1 used everywhere else. For a fair, pixel-identical comparison
we also run the classical NDVI rule on those *same* patches, so the app can show
both methods on exactly the same held-out pixels.

Everything here is local and CPU-friendly (no GPU, no Earth Engine): it reads the
.npz patches already on disk via src/dataset.py. The only external artefact is the
trained checkpoint (runs/siamese_base.pt), which is produced on Colab and copied
down into runs/.

Band order in every patch is cfg['dataset']['bands'] = [B2, B3, B4, B8], so:
    index 0 = B2 (blue)   index 1 = B3 (green)   index 2 = B4 (red)   index 3 = B8 (NIR)
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import PatchDataset
from src.model import SiameseUNet


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_model(checkpoint_path, in_ch, device="cpu"):
    """
    Load a trained SiameseUNet. Returns the eval-mode model, or None if the
    checkpoint file is not present (so the app can show a friendly message
    instead of crashing when the weights haven't been downloaded yet).
    """
    if not os.path.exists(checkpoint_path):
        return None
    model = SiameseUNet(in_ch=in_ch)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Scoring helpers (identical definition to train.py / evaluate.py)
# --------------------------------------------------------------------------- #
def _prf(tp, fp, fn):
    """Turn confusion counts into precision / recall / F1 (0 when undefined)."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def _ndvi(stack):
    """NDVI = (B8 - B4) / (B8 + B4) for a (C, H, W) patch (B4=idx2, B8=idx3).
    NDVI is invariant to a global reflectance rescale, so raw or /10000 is equal."""
    b4 = stack[2].astype(np.float32)
    b8 = stack[3].astype(np.float32)
    denom = b8 + b4
    denom[denom == 0] = 1e-6
    return (b8 - b4) / denom


def classical_pred(before, after, threshold):
    """Classical rule on ONE patch: flag pixels whose NDVI drops past threshold."""
    diff = _ndvi(after) - _ndvi(before)
    return diff < threshold            # boolean (H, W)


# --------------------------------------------------------------------------- #
# Visualisation helper (true-colour RGB from a patch)
# --------------------------------------------------------------------------- #
def rgb(stack, reflectance_max):
    """(C,H,W) reflectance patch -> (H,W,3) uint8 true-colour image (R=B4,G=B3,B=B2)."""
    r, g, b = stack[2], stack[1], stack[0]           # B4, B3, B2
    img = np.stack([r, g, b], axis=-1).astype(np.float32) / float(reflectance_max)
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Main entry: run both methods over a region's held-out test patches
# --------------------------------------------------------------------------- #
def run_region(region, cfg, checkpoint_path, split="test",
               n_examples=4, device="cpu"):
    """
    Score the deep model and the classical rule on `region`'s `split` patches.

    The deep model (base checkpoint) was trained with normalise=False, i.e.
    inputs scaled by /10000 — so we build the Dataset the SAME way, otherwise the
    inputs wouldn't match what the network learned.

    Returns a plain dict (JSON/numpy only — no torch objects escape) so the app
    can cache it:
        checkpoint_found : bool
        n_patches        : int
        threshold        : float
        deep             : {precision, recall, f1, tp, fp, fn}   (None if no ckpt)
        classical        : {precision, recall, f1, tp, fp, fn}
        examples         : list of dicts with RGB + mask arrays for display
    """
    ds = cfg["dataset"]
    bands = ds["bands"]
    threshold = cfg["threshold"]
    root = ds["out_dir"]
    refl_max = ds.get("preview_reflectance_max", 3000)

    # Same scaling the base model was trained with (normalise=False -> /10000),
    # no augmentation at inference, restricted to this one region + split.
    dataset = PatchDataset(
        os.path.join(root, "manifest.csv"), split, root,
        bands=bands, normalise=False, augment=False, regions=[region],
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    model = load_model(checkpoint_path, in_ch=len(bands), device=device)
    checkpoint_found = model is not None

    # confusion counts, accumulated over valid pixels only
    d_tp = d_fp = d_fn = 0.0        # deep
    c_tp = c_fp = c_fn = 0.0        # classical

    # keep the most informative patches (most truth-loss) for the example gallery
    examples = []

    with torch.no_grad():
        for batch in loader:
            before = batch["before"]                 # (B,C,H,W) scaled /10000
            after = batch["after"]
            label = batch["label"].numpy().astype(bool)     # (B,H,W)
            valid = batch["valid"].numpy().astype(bool)
            # raw reflectance for NDVI + RGB (undo the /10000 the Dataset applied)
            before_raw = before.numpy() * dataset.reflectance_scale
            after_raw = after.numpy() * dataset.reflectance_scale

            # --- deep prediction (if we have the weights) ---
            if checkpoint_found:
                logits = model(before.to(device), after.to(device))
                deep_pred = (torch.sigmoid(logits).squeeze(1) > 0.5).cpu().numpy()  # (B,H,W)
            else:
                deep_pred = np.zeros_like(label)

            # --- per-item accumulation + example harvesting ---
            for i in range(before.shape[0]):
                v = valid[i]
                lab = label[i]
                cl = classical_pred(before_raw[i], after_raw[i], threshold)

                if checkpoint_found:
                    dp = deep_pred[i]
                    d_tp += np.sum(dp & lab & v)
                    d_fp += np.sum(dp & ~lab & v)
                    d_fn += np.sum(~dp & lab & v)

                c_tp += np.sum(cl & lab & v)
                c_fp += np.sum(cl & ~lab & v)
                c_fn += np.sum(~cl & lab & v)

                # candidate example: store its truth-loss count to rank later
                pos = int(np.sum(lab & v))
                examples.append({
                    "pos": pos,
                    "before": rgb(before_raw[i], refl_max),
                    "after": rgb(after_raw[i], refl_max),
                    "classical": (cl & v),
                    "deep": (deep_pred[i] & v) if checkpoint_found else None,
                    "truth": (lab & v),
                })

    # keep the n patches with the most ground-truth loss (clearest to read)
    examples.sort(key=lambda e: e["pos"], reverse=True)
    examples = examples[:n_examples]

    return {
        "checkpoint_found": checkpoint_found,
        "n_patches": len(dataset),
        "threshold": threshold,
        "deep": _prf(d_tp, d_fp, d_fn) if checkpoint_found else None,
        "classical": _prf(c_tp, c_fp, c_fn),
        "examples": examples,
    }
