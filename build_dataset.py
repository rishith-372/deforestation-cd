"""
build_dataset.py — entry point to export the Siamese-network patch dataset.

Usage:
    python build_dataset.py                         # all 5 regions, full build
    python build_dataset.py --regions Rondonia --limit 4   # STEP 1 check
    python build_dataset.py --regions Rondonia,Para        # subset

The build is RESUMABLE: patches whose .npz already exists are kept and skipped,
failed Earth Engine requests are retried with back-off (in fetch_patch) and then
logged instead of crashing, and patches that are mostly no-data are skipped.

>>> EARTH ENGINE AUTH/QUOTA: initialises EE with the project id from config.yaml
    and your `earthengine authenticate` credentials (same as main.py). A full
    build makes many computePixels requests and can be throttled — that is
    expected and handled by the retry/back-off + resume. <<<

Output (under data/):
    patches/<Region>/<patch_id>.npz   before, after, label, valid
    previews/<patch_id>.png           alignment sanity-check images
(manifest.csv, region_means.json and train/val/test splits come in Step 3.)
"""

import os
import glob
import time
import random
import argparse

import yaml
import ee
import numpy as np

from src.pipeline import all_regions
from src.dataset_build import (
    utm_epsg, region_grid, build_stack, fetch_patch, arrays_from_record,
    patch_id, save_npz, valid_fraction, positive_fraction, save_preview,
)


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _centroid(bbox):
    """(lon, lat) centre of a [minLon, minLat, maxLon, maxLat] bbox."""
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def build_region(name, bbox, cfg, ds, limit=None, min_valid=None, min_pos=0.0, max_scan=None):
    """
    Export patches for one region. Returns a stats dict + list of saved paths.

    limit     : stop after this many patches are kept (None = whole grid).
    min_valid : valid-fraction threshold (None => ds['valid_threshold']).
    min_pos   : require at least this loss-pixel fraction (Step-1 check uses a
                small value so previews are guaranteed to contain red; 0 keeps
                no-loss patches, which the full training set needs as negatives).
    max_scan  : cap on patches actually fetched (bounds runtime on limited runs).
    """
    bands = ds["bands"]
    cloud_pct = ds.get("cloud_pct", cfg["cloud_pct"])          # dataset-only cloud filter
    min_valid = min_valid if min_valid is not None else ds["valid_threshold"]

    lon, lat = _centroid(bbox)
    epsg = utm_epsg(lon, lat)

    patches = region_grid(bbox, epsg, ds["patch_size"], ds["scale_m"])
    stack = build_stack(bbox, bands, cfg["before"], cfg["after"], cfg, cloud_pct)  # once per region

    out_dir = os.path.join(ds["out_dir"], "patches", name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== {name} ({epsg}) : {len(patches)} grid patches  "
          f"[cloud_pct={cloud_pct}, min_valid={min_valid}, min_pos={min_pos}"
          f"{', limit ' + str(limit) if limit else ''}"
          f"{', max_scan ' + str(max_scan) if max_scan else ''}] ===")

    saved, existing, skipped, failed, examined = 0, 0, 0, 0, 0
    kept_paths, failures = [], []

    for p in patches:
        if limit is not None and len(kept_paths) >= limit:
            break

        pid = patch_id(name, p["row"], p["col"])
        npz_path = os.path.join(out_dir, pid + ".npz")

        # RESUME: keep anything already on disk.
        if os.path.exists(npz_path):
            existing += 1
            kept_paths.append(npz_path)
            continue

        if max_scan is not None and examined >= max_scan:
            print(f"  (reached max_scan={max_scan}, stopping this region)", flush=True)
            break
        examined += 1

        # Fetch one pixel-aligned patch (retry/back-off is inside fetch_patch).
        # Progress is printed BEFORE the call (and flushed) so a slow Earth
        # Engine request looks like progress, not a freeze.
        print(f"  fetching {pid} (r{p['row']} c{p['col']}) ...", flush=True)
        t0 = time.time()
        try:
            rec = fetch_patch(stack, epsg, p["translateX"], p["translateY"],
                              ds["patch_size"], ds["scale_m"],
                              ds["max_retries"], ds["backoff_s"])
            before, after, label, valid = arrays_from_record(rec, bands)
        except Exception as e:                        # log, don't crash the build
            failed += 1
            failures.append((pid, repr(e)))
            print(f"  FAIL {pid}: {e}", flush=True)
            continue
        dt = time.time() - t0

        vf = valid_fraction(valid)
        pf = positive_fraction(label, valid)
        if vf < min_valid:
            skipped += 1
            print(f"  skip  {pid}  valid={vf:.2f} < {min_valid}  ({dt:.1f}s)", flush=True)
            continue
        if pf < min_pos:
            skipped += 1
            print(f"  skip  {pid}  pos={pf:.4f} < {min_pos} (little/no loss)  ({dt:.1f}s)", flush=True)
            continue

        save_npz(npz_path, before, after, label, valid)
        saved += 1
        kept_paths.append(npz_path)
        print(f"  saved {pid}  valid={vf:.2f}  pos={pf:.4f}  ({dt:.1f}s)", flush=True)

    print(f"  -> saved {saved}, resumed {existing}, skipped {skipped}, failed {failed}, fetched {examined}")
    return {"region": name, "saved": saved, "existing": existing,
            "skipped": skipped, "failed": failed, "failures": failures}, kept_paths


def make_previews(saved_paths, ds, seed):
    """Render up to preview_count random patches as side-by-side PNGs."""
    if not saved_paths:
        return
    prev_dir = os.path.join(ds["out_dir"], "previews")
    os.makedirs(prev_dir, exist_ok=True)

    rng = random.Random(seed)
    chosen = rng.sample(saved_paths, min(ds["preview_count"], len(saved_paths)))
    print(f"\nRendering {len(chosen)} preview PNG(s) -> {prev_dir}")
    for npz_path in chosen:
        data = np.load(npz_path)
        pid = os.path.splitext(os.path.basename(npz_path))[0]
        png = os.path.join(prev_dir, pid + ".png")
        save_preview(png, data["before"], data["after"], data["label"],
                     ds["bands"], ds["preview_reflectance_max"])
        print(f"  {png}")


def disk_stats(ds):
    """Total patch count, size on disk, and overall valid-pixel loss fraction."""
    files = glob.glob(os.path.join(ds["out_dir"], "patches", "*", "*.npz"))
    total_bytes = sum(os.path.getsize(f) for f in files)
    pos_sum, valid_sum = 0, 0
    for f in files:
        d = np.load(f)
        v = d["valid"].astype(bool)
        valid_sum += int(v.sum())
        pos_sum += int((d["label"][v] == 1).sum())
    overall_pos = (pos_sum / valid_sum) if valid_sum else 0.0
    return len(files), total_bytes, overall_pos


def main():
    parser = argparse.ArgumentParser(description="Export Siamese-network patch dataset from Earth Engine.")
    parser.add_argument("--regions", default=None,
                        help="comma-separated region names (default: all in config)")
    parser.add_argument("--limit", type=int, default=None,
                        help="max patches to KEEP per region (default: no limit)")
    parser.add_argument("--min-valid", type=float, default=None,
                        help="valid-fraction threshold (default: config dataset.valid_threshold)")
    parser.add_argument("--min-pos", type=float, default=0.0,
                        help="require >= this loss-pixel fraction (Step-1 check: use a small value "
                             "like 0.002 so previews contain loss; default 0 keeps negatives)")
    parser.add_argument("--max-scan", type=int, default=None,
                        help="cap on patches actually fetched per region (bounds runtime)")
    parser.add_argument("--no-preview", action="store_true", help="skip PNG previews")
    args = parser.parse_args()

    cfg = load_config()
    ds = cfg["dataset"]

    print(f"Initialising Earth Engine (project: {cfg['ee_project']}) ...")
    ee.Initialize(project=cfg["ee_project"])

    regions = all_regions(cfg)
    if args.regions:
        wanted = [r.strip() for r in args.regions.split(",")]
        missing = [r for r in wanted if r not in regions]
        if missing:
            raise SystemExit(f"Unknown region(s): {missing}. Known: {list(regions)}")
        regions = {r: regions[r] for r in wanted}

    all_saved, all_stats = [], []
    for name, bbox in regions.items():
        stats, kept = build_region(name, bbox, cfg, ds, limit=args.limit,
                                   min_valid=args.min_valid, min_pos=args.min_pos,
                                   max_scan=args.max_scan)
        all_stats.append(stats)
        all_saved.extend(kept)

    if not args.no_preview:
        make_previews(all_saved, ds, ds["seed"])

    # ---- summary ----
    n_files, total_bytes, overall_pos = disk_stats(ds)
    print("\n================= DATASET SUMMARY =================")
    for s in all_stats:
        print(f"  {s['region']:<12} saved={s['saved']}  resumed={s['existing']}  "
              f"skipped={s['skipped']}  failed={s['failed']}")
    any_fail = [f for s in all_stats for f in s["failures"]]
    if any_fail:
        print(f"  Failures ({len(any_fail)}): " + ", ".join(pid for pid, _ in any_fail))
    print(f"\n  Patches on disk : {n_files}")
    print(f"  Size on disk    : {total_bytes / 1e6:.1f} MB")
    print(f"  Positive (loss) pixel fraction among valid : {overall_pos:.5f}  "
          f"(small => class imbalance, handled in training)")


if __name__ == "__main__":
    main()
