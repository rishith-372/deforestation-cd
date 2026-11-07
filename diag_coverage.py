"""
diag_coverage.py — one-off diagnostic for the low valid-fraction problem.

It answers two questions fast (a few getInfo calls, ~15s):

1) Over the WHOLE Rondonia region, what fraction is covered by each layer on its
   own — before composite, after composite, Hansen truth — and by all three
   combined (valid)? This isolates WHICH layer is creating the holes.
      - If before ~1, after ~1, truth ~1 but valid is low  => the three don't
        overlap (a reprojection/alignment issue between layers).
      - If one of them is low  => that layer is the culprit (e.g. S2 scene gaps).

2) For a single patch at the CENTRE of the region (where data should be fully
   present), what valid fraction does computePixels return? If region coverage
   is high but this centre patch is low, the patch GRID is the problem, not data.

Run:  python diag_coverage.py
"""

import yaml
import ee
import numpy as np
from pyproj import Transformer

from src.acquire import s2_composite, hansen_truth
from src.dataset_build import utm_epsg, build_stack, fetch_patch, arrays_from_record


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    ds = cfg["dataset"]
    ee.Initialize(project=cfg["ee_project"])

    bbox = cfg["regions"]["dev"]["Rondonia"]
    bands = ds["bands"]
    cloud = ds.get("cloud_pct", cfg["cloud_pct"])
    region = ee.Geometry.Rectangle(bbox)

    before = s2_composite(region, cfg["before"][0], cfg["before"][1], cloud, cfg["s2_collection"]).select(bands)
    after = s2_composite(region, cfg["after"][0], cfg["after"][1], cloud, cfg["s2_collection"]).select(bands)
    truth = hansen_truth(region, cfg["hansen_loss_min"], cfg["hansen_loss_max"], cfg["hansen_asset"])

    bm = before.mask().reduce(ee.Reducer.min()).gt(0).rename("before")
    am = after.mask().reduce(ee.Reducer.min()).gt(0).rename("after")
    tm = truth.mask().gt(0).rename("truth")
    valid = bm.And(am).And(tm).rename("valid")

    # --- (1) whole-region per-layer coverage -------------------------------
    print(f"Region Rondonia {bbox}  cloud_pct={cloud}")
    print("\n[1] Whole-region coverage fraction (scale 30):")
    stats = (bm.addBands(am).addBands(tm).addBands(valid)
             .reduceRegion(ee.Reducer.mean(), region, 30,
                           maxPixels=int(1e9), bestEffort=True).getInfo())
    for k in ["before", "after", "truth", "valid"]:
        print(f"    {k:<7}: {stats.get(k)}")

    # How many S2 scenes actually fed each composite (after cloud filter)?
    coll = cfg["s2_collection"]
    n_before = (ee.ImageCollection(coll).filterBounds(region)
                .filterDate(cfg["before"][0], cfg["before"][1])
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud)).size().getInfo())
    n_after = (ee.ImageCollection(coll).filterBounds(region)
               .filterDate(cfg["after"][0], cfg["after"][1])
               .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud)).size().getInfo())
    print(f"\n    S2 scenes in composite: before={n_before}, after={n_after}")

    # --- (2) single centre patch via computePixels -------------------------
    epsg = utm_epsg((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    tf = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    cx, cy = tf.transform((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    half = (ds["patch_size"] // 2) * ds["scale_m"]
    tx, ty = cx - half, cy + half       # top-left of a patch centred on the region

    print(f"\n[2] Centre patch at UTM ({tx:.0f}, {ty:.0f}) in {epsg}:")
    stack = build_stack(bbox, bands, cfg["before"], cfg["after"], cfg, cloud)
    rec = fetch_patch(stack, epsg, tx, ty, ds["patch_size"], ds["scale_m"],
                      ds["max_retries"], ds["backoff_s"])
    _, _, label, valid_arr = arrays_from_record(rec, bands)
    print(f"    valid fraction      : {valid_arr.mean():.3f}")
    print(f"    loss-pixel fraction : {(label[valid_arr.astype(bool)] == 1).mean() if valid_arr.any() else 0:.4f}")


if __name__ == "__main__":
    main()
