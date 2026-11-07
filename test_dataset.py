"""
test_dataset.py — Step 3b: smoke-test the PyTorch Dataset.

Loads ONE batch from each split and prints tensor shapes + the positive-pixel
fraction (over valid pixels). Run AFTER build_manifest.py.

    python test_dataset.py

No Earth Engine needed — reads local .npz patches via the manifest.
"""

import os

import yaml
import torch
from torch.utils.data import DataLoader

from src.dataset import PatchDataset


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    ds = cfg["dataset"]
    root = ds["out_dir"]
    manifest = os.path.join(root, "manifest.csv")
    means = os.path.join(root, "region_means.json")

    print("Loading one batch per split (normalise=ON, augment on train):\n")
    for split in ["train", "val", "test"]:
        dataset = PatchDataset(
            manifest, split, root,
            region_means=means, bands=ds["bands"],
            normalise=True, augment=(split == "train"),
        )
        loader = DataLoader(dataset, batch_size=4, shuffle=(split == "train"))
        batch = next(iter(loader))

        # positive fraction over valid pixels only
        valid_sum = batch["valid"].sum()
        pos = float((batch["label"] * batch["valid"]).sum() / valid_sum) if valid_sum > 0 else 0.0

        print(f"[{split}] n={len(dataset)}")
        print(f"    before {tuple(batch['before'].shape)}  dtype={batch['before'].dtype}")
        print(f"    after  {tuple(batch['after'].shape)}")
        print(f"    label  {tuple(batch['label'].shape)}   valid {tuple(batch['valid'].shape)}")
        print(f"    before value range [{batch['before'].min():.3f}, {batch['before'].max():.3f}]")
        print(f"    positive (loss) fraction over valid pixels: {pos:.4f}")
        print(f"    regions in batch: {list(batch['region'])}\n")

    print("OK — Dataset returns aligned before/after/label/valid tensors per split.")


if __name__ == "__main__":
    main()
