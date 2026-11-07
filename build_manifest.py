"""
build_manifest.py — Step 3a: write data/region_means.json and data/manifest.csv.

Run AFTER build_dataset.py has produced the patches.

    python build_manifest.py

What it does:
  1. region_means.json — per-region per-band mean reflectance for the before and
     after composites (same reducer as src/normalise.py). Lets src/dataset.py
     apply the seasonal-normalisation option at training time.  (needs Earth Engine)
  2. manifest.csv — one row per patch on disk: id, region, grid row/col, bounding
     box (UTM + lon/lat), valid fraction, positive (loss) fraction, and a
     train/val/test split assigned by SPATIAL BLOCKS so neighbouring patches never
     leak across splits.  (no Earth Engine — just scans the .npz files)

>>> Earth Engine auth: same project id / credentials as the rest of the project
    (needed only for the region means). <<<
"""

import os
import glob
import json
from collections import defaultdict

import yaml
import ee
import numpy as np
import pandas as pd
from pyproj import Transformer

from src.pipeline import all_regions
from src.dataset_build import utm_epsg, region_grid, region_means


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_row_col(patch_id):
    """'Rondonia_r012_c007' -> (12, 7). Region names have no underscores."""
    parts = patch_id.split("_")
    return int(parts[-2][1:]), int(parts[-1][1:])


def assign_splits(rows, ds):
    """
    Assign each patch a split by WHOLE spatial blocks, independently per region.
    A block is a block_size x block_size group of grid cells; all patches in a
    block get the same split, so adjacent patches can't straddle train/test.
    Reproducible via the fixed seed.
    """
    import random
    rng = random.Random(ds["seed"])
    tr, va, te = ds["split_ratios"]

    # blocks present per region
    blocks = defaultdict(set)
    for r in rows:
        blocks[r["region"]].add((r["block_row"], r["block_col"]))

    split_of = {}
    for region, bset in blocks.items():
        bl = sorted(bset)              # deterministic order before shuffling
        rng.shuffle(bl)
        n = len(bl)
        n_tr = int(round(n * tr))
        n_va = int(round(n * va))
        for i, b in enumerate(bl):
            if i < n_tr:
                s = "train"
            elif i < n_tr + n_va:
                s = "val"
            else:
                s = "test"
            split_of[(region, b[0], b[1])] = s

    for r in rows:
        r["split"] = split_of[(r["region"], r["block_row"], r["block_col"])]


def main():
    cfg = load_config()
    ds = cfg["dataset"]
    out_dir = ds["out_dir"]
    bands = ds["bands"]
    cloud = ds.get("cloud_pct", cfg["cloud_pct"])
    block = ds["block_size"]

    print(f"Initialising Earth Engine (project: {cfg['ee_project']}) ...")
    ee.Initialize(project=cfg["ee_project"])

    regions = all_regions(cfg)

    # --- 1) region means (Earth Engine) ------------------------------------
    print("\nComputing region means (before/after, per band) ...")
    means = {}
    for name, bbox in regions.items():
        means[name] = region_means(bbox, bands, cfg["before"], cfg["after"], cfg, cloud)
        print(f"  {name}: before={ {b: round(means[name]['before'][b],1) for b in bands} }")
    means_path = os.path.join(out_dir, "region_means.json")
    with open(means_path, "w") as f:
        json.dump(means, f, indent=2)
    print(f"  -> wrote {means_path}")

    # --- 2) manifest (scan patches on disk) --------------------------------
    print("\nScanning patches and building manifest ...")
    rows = []
    for name, bbox in regions.items():
        epsg = utm_epsg((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        grid = {(p["row"], p["col"]): p
                for p in region_grid(bbox, epsg, ds["patch_size"], ds["scale_m"])}
        to_lonlat = Transformer.from_crs(epsg, "EPSG:4326", always_xy=True)

        files = sorted(glob.glob(os.path.join(out_dir, "patches", name, "*.npz")))
        for f in files:
            pid = os.path.splitext(os.path.basename(f))[0]
            row, col = parse_row_col(pid)

            d = np.load(f)
            valid = d["valid"]
            label = d["label"]
            vb = valid.astype(bool)
            vf = float(valid.mean())
            pf = float((label[vb] == 1).mean()) if vb.any() else 0.0

            ux0, uy0, ux1, uy1 = grid[(row, col)]["utm_bbox"]
            lon_min, lat_min = to_lonlat.transform(ux0, uy0)
            lon_max, lat_max = to_lonlat.transform(ux1, uy1)

            rows.append({
                "patch_id": pid,
                "region": name,
                "row": row,
                "col": col,
                "block_row": row // block,
                "block_col": col // block,
                "split": None,                      # filled by assign_splits
                "valid_frac": round(vf, 4),
                "pos_frac": round(pf, 6),
                "crs": epsg,
                "utm_xmin": ux0, "utm_ymin": uy0, "utm_xmax": ux1, "utm_ymax": uy1,
                "lon_min": round(lon_min, 6), "lat_min": round(lat_min, 6),
                "lon_max": round(lon_max, 6), "lat_max": round(lat_max, 6),
                "path": os.path.relpath(f, out_dir).replace("\\", "/"),
            })

    assign_splits(rows, ds)

    df = pd.DataFrame(rows)
    manifest_path = os.path.join(out_dir, "manifest.csv")
    df.to_csv(manifest_path, index=False)
    print(f"  -> wrote {manifest_path}  ({len(df)} patches)")

    # --- summary -----------------------------------------------------------
    print("\n================= MANIFEST SUMMARY =================")
    print("Patches per region x split:")
    print(pd.crosstab(df["region"], df["split"]).to_string())
    print("\nPositive (loss) pixel fraction by split (patch-mean):")
    print(df.groupby("split")["pos_frac"].mean().round(5).to_string())
    print("\nBlocks per region (train/val/test):")
    for region in df["region"].unique():
        sub = df[df["region"] == region]
        nb = sub.groupby("split").apply(
            lambda g: g[["block_row", "block_col"]].drop_duplicates().shape[0])
        print(f"  {region:<12} " + "  ".join(f"{k}={int(v)}" for k, v in nb.items()))


if __name__ == "__main__":
    main()
