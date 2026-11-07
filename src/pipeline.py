"""
pipeline.py — orchestration: wires acquire -> normalise -> detect -> evaluate.

Two entry points:
  * build_scene(...) -> dict of the INTERMEDIATE ee.Images (before, after,
      detected, truth) plus the region geometry. The Streamlit app uses this to
      display each stage of the method on a map.
  * run_region(...)  -> dict of METRICS for one region/toggle combination.
      main.py uses this for the batch comparison + results.json.

run_region is just build_scene + evaluate, so both paths run identical logic.

NAMING NOTE: the on/off switch is called `normalise_on` (not `normalise`) so it
does not shadow the imported normalise() function.
"""

import ee

from src.acquire import s2_composite, hansen_truth
from src.normalise import normalise
from src.detect import change_map
from src.evaluate import evaluate


def all_regions(cfg):
    """Flatten the config regions dict (dev + test) into {name: bbox}."""
    merged = {}
    merged.update(cfg["regions"].get("dev", {}))
    merged.update(cfg["regions"].get("test", {}))
    return merged


def build_scene(bbox, before, after, cfg, normalise_on):
    """
    Build every image the method produces for one region + toggle setting.

    IMPORTANT (display vs detection): the returned 'before'/'after' images are
    always the RAW composites, so they render correctly as true colour on a map.
    Normalisation only rescales B8/B4 (not B3/B2), which would wreck a true-
    colour view — so we apply normalisation on COPIES used purely for detection,
    and leave the display composites untouched. The detection logic itself is
    unchanged from the validated pipeline.

    Args:
        bbox:         [minLon, minLat, maxLon, maxLat].
        before/after: [start, end] date-string pairs.
        cfg:          parsed config.yaml dict.
        normalise_on: apply seasonal normalisation before detection?

    Returns:
        dict {region, before, after, detected, truth} of ee objects.
    """
    region = ee.Geometry.Rectangle(bbox)

    # Raw composites (used for display, and as the basis for detection).
    before_raw = s2_composite(region, before[0], before[1], cfg["cloud_pct"], cfg["s2_collection"])
    after_raw = s2_composite(region, after[0], after[1], cfg["cloud_pct"], cfg["s2_collection"])

    # Detection inputs: normalised copies if the toggle is on, else the raw ones.
    b_img, a_img = before_raw, after_raw
    if normalise_on:
        b_img = normalise(before_raw, region, scale=cfg["scale"])
        a_img = normalise(after_raw, region, scale=cfg["scale"])

    detected = change_map(b_img, a_img, cfg["threshold"])
    truth = hansen_truth(region, cfg["hansen_loss_min"], cfg["hansen_loss_max"], cfg["hansen_asset"])

    return {
        "region": region,
        "before": before_raw,   # display: true colour
        "after": after_raw,     # display: true colour
        "detected": detected,   # red layer / scoring
        "truth": truth,         # blue layer / scoring
    }


def run_region(name, bbox, normalise_on, cfg):
    """
    Run the full pipeline for one region and return a flat metrics dict.

    Returns:
        dict {region, normalised, tp, fp, fn, precision, recall, f1}
        (shape matches the results.json spec).
    """
    scene = build_scene(bbox, cfg["before"], cfg["after"], cfg, normalise_on)
    metrics = evaluate(scene["detected"], scene["truth"], scene["region"], cfg["scale"])
    return {"region": name, "normalised": normalise_on, **metrics}
