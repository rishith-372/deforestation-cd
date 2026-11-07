"""
train.py — train/evaluate the Siamese U-Net on the patch dataset.

Runs anywhere (CPU or CUDA). On Colab it auto-uses the GPU.

Examples:
    python train.py --overfit                 # sanity: overfit ONE batch (fast)
    python train.py --normalise --epochs 30   # train WITH seasonal normalisation
    python train.py --epochs 30               # train WITHOUT (baseline)
    python train.py --leave-out Rondonia --epochs 30   # leave-one-region-out

Scoring uses the SAME Precision/Recall/F1 as the classical pipeline, over VALID
pixels only, so the deep model and the NDVI method are directly comparable.
Class imbalance (~5.5% loss pixels) is handled with a pos_weight on BCE.
"""

import os
import json
import argparse

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import PatchDataset
from src.model import SiameseUNet


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def masked_bce(logits, label, valid, pos_weight):
    """BCE-with-logits per pixel, averaged over VALID pixels only."""
    crit = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    loss_map = crit(logits.squeeze(1), label)          # (B,H,W)
    denom = valid.sum().clamp(min=1.0)
    return (loss_map * valid).sum() / denom


@torch.no_grad()
def evaluate(model, loader, device):
    """Accumulate TP/FP/FN over valid pixels -> precision/recall/f1."""
    model.eval()
    tp = fp = fn = 0.0
    for batch in loader:
        before = batch["before"].to(device)
        after = batch["after"].to(device)
        label = batch["label"].to(device)
        valid = batch["valid"].to(device).bool()

        pred = (torch.sigmoid(model(before, after)).squeeze(1) > 0.5)
        lab = label.bool()
        tp += (pred & lab & valid).sum().item()
        fp += (pred & ~lab & valid).sum().item()
        fn += (~pred & lab & valid).sum().item()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def make_loader(cfg, ds, split, normalise, augment, regions, batch_size, shuffle):
    root = ds["out_dir"]
    dataset = PatchDataset(
        os.path.join(root, "manifest.csv"), split, root,
        region_means=os.path.join(root, "region_means.json"),
        bands=ds["bands"], normalise=normalise, augment=augment, regions=regions,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normalise", action="store_true", help="apply seasonal normalisation")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos-weight", type=float, default=15.0, help="BCE weight on the loss class")
    ap.add_argument("--leave-out", default=None, help="hold this region out of train/val (test on it)")
    ap.add_argument("--overfit", action="store_true", help="sanity: overfit ONE batch")
    ap.add_argument("--out", default="runs", help="where to save checkpoint + metrics")
    args = ap.parse_args()

    cfg = load_config()
    ds = cfg["dataset"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  normalise={args.normalise}  leave_out={args.leave_out}")

    model = SiameseUNet(in_ch=len(ds["bands"])).to(device)
    pos_weight = torch.tensor(args.pos_weight, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- leave-one-region-out: train/val on the others, test on the held-out one
    all_regions = None
    train_regions = None
    test_regions = None
    if args.leave_out:
        import pandas as pd
        names = pd.read_csv(os.path.join(ds["out_dir"], "manifest.csv"))["region"].unique().tolist()
        train_regions = [r for r in names if r != args.leave_out]
        test_regions = [args.leave_out]

    # ---------- sanity: overfit one batch ----------
    if args.overfit:
        loader = make_loader(cfg, ds, "train", args.normalise, False, train_regions, args.batch_size, True)
        batch = next(iter(loader))
        before = batch["before"].to(device); after = batch["after"].to(device)
        label = batch["label"].to(device); valid = batch["valid"].to(device)
        model.train()
        print("Overfitting one batch (loss should fall toward 0):")
        for step in range(60):
            opt.zero_grad()
            loss = masked_bce(model(before, after), label, valid, pos_weight)
            loss.backward(); opt.step()
            if step % 10 == 0 or step == 59:
                print(f"  step {step:2d}  loss {loss.item():.4f}")
        return

    # ---------- full training ----------
    train_loader = make_loader(cfg, ds, "train", args.normalise, True, train_regions, args.batch_size, True)
    val_loader = make_loader(cfg, ds, "val", args.normalise, False, train_regions, args.batch_size, False)
    # test: the held-out region (LORO) or the normal test split
    test_loader = make_loader(cfg, ds, "test" if not args.leave_out else "train",
                              args.normalise, False, test_regions, args.batch_size, False) \
        if args.leave_out else make_loader(cfg, ds, "test", args.normalise, False, None, args.batch_size, False)

    os.makedirs(args.out, exist_ok=True)
    tag = ("norm" if args.normalise else "base") + (f"_LO-{args.leave_out}" if args.leave_out else "")
    ckpt = os.path.join(args.out, f"siamese_{tag}.pt")

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            before = batch["before"].to(device); after = batch["after"].to(device)
            label = batch["label"].to(device); valid = batch["valid"].to(device)
            opt.zero_grad()
            loss = masked_bce(model(before, after), label, valid, pos_weight)
            loss.backward(); opt.step()
            running += loss.item()
        val = evaluate(model, val_loader, device)
        print(f"epoch {epoch:2d}  train_loss {running/len(train_loader):.4f}  "
              f"val P={val['precision']:.3f} R={val['recall']:.3f} F1={val['f1']:.3f}")
        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            torch.save(model.state_dict(), ckpt)

    # final test with the best checkpoint
    model.load_state_dict(torch.load(ckpt, map_location=device))
    test = evaluate(model, test_loader, device)
    print(f"\n[{tag}] TEST  P={test['precision']:.3f}  R={test['recall']:.3f}  F1={test['f1']:.3f}")
    with open(os.path.join(args.out, f"metrics_{tag}.json"), "w") as f:
        json.dump({"tag": tag, "normalise": args.normalise, "leave_out": args.leave_out,
                   "best_val_f1": best_f1, "test": test}, f, indent=2)
    print(f"saved {ckpt} and metrics_{tag}.json")


if __name__ == "__main__":
    main()
