"""
dataset.py — PyTorch Dataset that serves patches to the Siamese network.

Reads data/manifest.csv, loads the .npz patches for one split, and returns
tensors ready for training:
    before : (C, H, W) float32   scaled reflectance (2019)
    after  : (C, H, W) float32   scaled reflectance (2023)
    label  : (H, W)    float32   1 = Hansen forest loss, 0 = no loss
    valid  : (H, W)    float32   1 = usable pixel  (EXCLUDE valid==0 from loss/metrics)

Options:
  * normalise=True  -> divide each band by that region's mean (from
    region_means.json), matching src/normalise.py. This is how the seasonal-
    normalisation contribution is tested on the deep model. normalise=False uses
    a plain reflectance scaling instead.
  * augment=True    -> random horizontal/vertical flips and 90-degree rotations,
    applied IDENTICALLY to before/after/label/valid so they stay aligned.
  * regions=[...]   -> keep only these regions (for leave-one-region-out).

The `valid` mask MUST be used by the training loss and metrics to ignore no-data
pixels — it is returned for exactly that purpose.
"""

import os
import json
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    def __init__(self, manifest, split, data_root,
                 region_means=None, bands=("B2", "B3", "B4", "B8"),
                 normalise=False, augment=False, regions=None,
                 reflectance_scale=10000.0):
        df = pd.read_csv(manifest)
        df = df[df["split"] == split]
        if regions is not None:
            df = df[df["region"].isin(regions)]
        self.rows = df.reset_index(drop=True)

        self.data_root = data_root
        self.bands = list(bands)
        self.normalise = normalise
        self.augment = augment
        self.reflectance_scale = float(reflectance_scale)

        self.means = None
        if normalise:
            if region_means is None:
                raise ValueError("normalise=True requires region_means=<path to region_means.json>")
            with open(region_means, "r") as f:
                self.means = json.load(f)

    def __len__(self):
        return len(self.rows)

    def _scale(self, arr, region, when):
        """arr: (C,H,W) float32 raw reflectance -> scaled per the chosen mode."""
        if self.normalise:
            m = self.means[region][when]                      # {band: mean}
            denom = np.array([m[b] for b in self.bands], dtype=np.float32).reshape(-1, 1, 1)
            return arr / denom                                # match src/normalise.py (band / mean)
        return arr / self.reflectance_scale                   # plain reflectance ~[0, ~0.3]

    def _augment(self, before, after, label, valid):
        """Same flips/rotations applied to every array so they stay registered."""
        if random.random() < 0.5:                             # horizontal flip (width axis)
            before, after = before[:, :, ::-1], after[:, :, ::-1]
            label, valid = label[:, ::-1], valid[:, ::-1]
        if random.random() < 0.5:                             # vertical flip (height axis)
            before, after = before[:, ::-1, :], after[:, ::-1, :]
            label, valid = label[::-1, :], valid[::-1, :]
        k = random.randint(0, 3)                              # 0/90/180/270 rotation
        if k:
            before, after = np.rot90(before, k, (1, 2)), np.rot90(after, k, (1, 2))
            label, valid = np.rot90(label, k, (0, 1)), np.rot90(valid, k, (0, 1))
        return before, after, label, valid

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        path = os.path.join(self.data_root, "patches", r["region"], r["patch_id"] + ".npz")
        d = np.load(path)

        before = d["before"].astype(np.float32)
        after = d["after"].astype(np.float32)
        label = d["label"].astype(np.float32)
        valid = d["valid"].astype(np.float32)

        before = self._scale(before, r["region"], "before")
        after = self._scale(after, r["region"], "after")

        if self.augment:
            before, after, label, valid = self._augment(before, after, label, valid)

        # ascontiguousarray fixes the negative strides left by slicing/rot90.
        return {
            "before": torch.from_numpy(np.ascontiguousarray(before)),
            "after": torch.from_numpy(np.ascontiguousarray(after)),
            "label": torch.from_numpy(np.ascontiguousarray(label)),
            "valid": torch.from_numpy(np.ascontiguousarray(valid)),
            "region": r["region"],
            "patch_id": r["patch_id"],
        }
