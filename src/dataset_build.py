"""
dataset_build.py — tile each region into fixed 256x256 @10m patches and export
them from Earth Engine as pixel-aligned .npz files (+ PNG previews).

Each patch stores, on one fixed 10 m grid in the region's UTM projection:
  * before : (C, H, W) uint16 raw Sentinel-2 reflectance, 2019 window
  * after  : (C, H, W) uint16 raw Sentinel-2 reflectance, 2023 window
  * label  : (H, W)   uint8  Hansen forest-loss 2020-2023 (1=loss, 0=no loss)
  * valid  : (H, W)   uint8  1 where before AND after AND truth all have data

WHY one request per patch: we stack before+after+label+valid into a SINGLE
ee.Image and fetch it with ee.data.computePixels on ONE grid, so every layer is
pixel-aligned by construction (no resampling drift between before/after/label).
Hansen (30 m) is sampled to the 10 m grid with nearest-neighbour (EE's default).

Raw (un-normalised) reflectance is stored on purpose: seasonal normalisation is
a TRAINING-TIME option (see src/dataset.py), so the Siamese model can be trained
with and without it. Per-region band means for that option are computed here too
(region_means()) using the same reducer as src/normalise.py.

>>> EARTH ENGINE AUTH/QUOTA: ee must already be initialised by the caller
    (build_dataset.py does this). computePixels counts against your quota and can
    throttle/time out — hence the retry + back-off in fetch_patch(). <<<
"""

import os
import json
import math
import time

import numpy as np
import ee
from pyproj import Transformer

from src.acquire import s2_composite, hansen_truth


# --------------------------------------------------------------------------
# Geometry / grid helpers
# --------------------------------------------------------------------------

def utm_epsg(lon, lat):
    """EPSG code string for the UTM zone containing (lon, lat)."""
    zone = int(math.floor((lon + 180.0) / 6.0)) % 60 + 1
    hemisphere = 326 if lat >= 0 else 327        # 326xx = N, 327xx = S
    return f"EPSG:{hemisphere}{zone:02d}"


def region_grid(bbox, epsg, patch_size, scale):
    """
    Tile a lon/lat bounding box into non-overlapping patches on a FIXED grid in
    the region's UTM projection.

    Returns a list of dicts: {row, col, translateX, translateY, utm_bbox}
    where (translateX, translateY) is the patch's top-left corner in UTM metres
    (the affineTransform origin used by computePixels).

    NOTE: we convert lon/lat -> UTM metres LOCALLY with pyproj. Doing it via
    ee (geometry.transform(...).coordinates().getInfo()) does NOT work: EE always
    reports geometry coordinates back in WGS84 degrees regardless of projection,
    which silently yields a bogus grid origin outside the UTM zone.
    """
    patch_m = patch_size * scale                 # patch side length in metres

    # Project the four bbox corners lon/lat -> UTM metres and take the
    # axis-aligned bounds (a lon/lat rectangle is not quite rectangular in UTM).
    min_lon, min_lat, max_lon, max_lat = bbox
    tf = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    corners = [(min_lon, min_lat), (min_lon, max_lat),
               (max_lon, min_lat), (max_lon, max_lat)]
    xy = [tf.transform(lon, lat) for lon, lat in corners]   # always_xy => (easting, northing)
    xs = [p[0] for p in xy]
    ys = [p[1] for p in xy]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)

    # Snap the grid origin to a multiple of the patch size so the grid is fixed
    # and reproducible regardless of which corner we start from.
    x0 = math.floor(xmin / patch_m) * patch_m        # left edge
    y_top = math.ceil(ymax / patch_m) * patch_m      # top edge
    ncols = int(math.ceil((xmax - x0) / patch_m))
    nrows = int(math.ceil((y_top - ymin) / patch_m))

    patches = []
    for r in range(nrows):
        for c in range(ncols):
            tx = x0 + c * patch_m
            ty = y_top - r * patch_m
            patches.append({
                "row": r,
                "col": c,
                "translateX": tx,
                "translateY": ty,
                # UTM bounds of the patch: [xmin, ymin, xmax, ymax]
                "utm_bbox": [tx, ty - patch_m, tx + patch_m, ty],
            })
    return patches


# --------------------------------------------------------------------------
# Image stack (before + after + label + valid) for a region
# --------------------------------------------------------------------------

def build_stack(bbox, bands, before, after, cfg, cloud_pct):
    """
    Build the single ee.Image holding every layer for a region, ready to fetch.

    Band names are prefixed so computePixels returns named fields:
        before_B2 ... before_B8, after_B2 ... after_B8, label, valid

    `cloud_pct` is passed explicitly (the dataset uses its own, looser value than
    the metrics pipeline — see config.yaml dataset.cloud_pct).
    """
    region = ee.Geometry.Rectangle(bbox)

    before_raw = s2_composite(region, before[0], before[1],
                              cloud_pct, cfg["s2_collection"]).select(bands)
    after_raw = s2_composite(region, after[0], after[1],
                             cloud_pct, cfg["s2_collection"]).select(bands)
    truth = hansen_truth(region, cfg["hansen_loss_min"], cfg["hansen_loss_max"],
                         cfg["hansen_asset"])

    # VALID = we have BEFORE and AFTER satellite data to make a prediction.
    #
    # DEVIATION FROM THE ORIGINAL SPEC (which also required "truth unmasked"):
    # Hansen's lossyear band is MASKED on no-loss pixels — it only carries a value
    # where forest loss ever occurred (~33% of Rondonia). Requiring truth-unmasked
    # would keep ONLY loss pixels and discard every no-loss (negative) pixel,
    # destroying the class the network must learn against. Hansen's ABSENCE of loss
    # is a valid "no loss" label, so we base validity on the imagery and unmask the
    # label to 0. (Verified with diag_coverage.py: before=after=1.0, truth=0.33.)
    before_mask = before_raw.mask().reduce(ee.Reducer.min()).gt(0)
    after_mask = after_raw.mask().reduce(ee.Reducer.min()).gt(0)
    valid = before_mask.And(after_mask).rename("valid").toUint8()

    # Fill masked pixels with 0, cast to compact types, prefix band names.
    before_b = before_raw.unmask(0).toUint16().rename([f"before_{b}" for b in bands])
    after_b = after_raw.unmask(0).toUint16().rename([f"after_{b}" for b in bands])
    label = truth.unmask(0).toUint8().rename("label")   # no loss / no Hansen value => 0

    return before_b.addBands(after_b).addBands(label).addBands(valid)


# --------------------------------------------------------------------------
# Fetch a single patch via computePixels (one aligned request)
# --------------------------------------------------------------------------

def fetch_patch(stack, epsg, translate_x, translate_y, patch_size, scale,
                max_retries, backoff_s):
    """
    Fetch one patch as a NUMPY_NDARRAY structured array, with retry + back-off.

    The grid's affineTransform pins the exact UTM origin and 10 m pixel size, so
    the returned array is aligned to the fixed region grid. scaleY is negative
    because image rows go top -> bottom (north -> south).
    """
    request = {
        "expression": stack,
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": patch_size, "height": patch_size},
            "affineTransform": {
                "scaleX": scale, "shearX": 0, "translateX": translate_x,
                "shearY": 0, "scaleY": -scale, "translateY": translate_y,
            },
            "crsCode": epsg,
        },
    }
    last_err = None
    for attempt in range(max_retries):
        try:
            return ee.data.computePixels(request)
        except Exception as e:                       # EE throttle / timeout / transient
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(backoff_s * (2 ** attempt))   # exponential back-off
    raise last_err


def arrays_from_record(rec, bands):
    """Split the structured array into before/after/label/valid numpy arrays."""
    before = np.stack([rec[f"before_{b}"] for b in bands], axis=0).astype(np.uint16)
    after = np.stack([rec[f"after_{b}"] for b in bands], axis=0).astype(np.uint16)
    label = np.asarray(rec["label"]).astype(np.uint8)
    valid = np.asarray(rec["valid"]).astype(np.uint8)
    return before, after, label, valid


# --------------------------------------------------------------------------
# Disk helpers + stats
# --------------------------------------------------------------------------

def patch_id(region, row, col):
    """Stable id => stable filename, which makes the build resumable."""
    return f"{region}_r{row:03d}_c{col:03d}"


def save_npz(path, before, after, label, valid):
    np.savez_compressed(path, before=before, after=after, label=label, valid=valid)


def valid_fraction(valid):
    """Fraction of the patch that has real data in all layers."""
    return float(valid.mean())


def positive_fraction(label, valid):
    """Loss-pixel fraction among VALID pixels (0 if no valid pixels)."""
    v = valid.astype(bool)
    if v.sum() == 0:
        return 0.0
    return float((label[v] == 1).mean())


# --------------------------------------------------------------------------
# Previews
# --------------------------------------------------------------------------

def _rgb(arr, bands, max_reflect):
    """(C,H,W) reflectance -> (H,W,3) float in [0,1] using B4/B3/B2 as R/G/B."""
    r = arr[bands.index("B4")].astype(np.float32)
    g = arr[bands.index("B3")].astype(np.float32)
    b = arr[bands.index("B2")].astype(np.float32)
    rgb = np.stack([r, g, b], axis=-1) / float(max_reflect)
    return np.clip(rgb, 0.0, 1.0)


def save_preview(path, before, after, label, bands, max_reflect):
    """
    Side-by-side PNG: before RGB | after RGB | after RGB with loss label in red.
    This is the alignment sanity check — cleared ground should sit under the red.
    """
    import matplotlib
    matplotlib.use("Agg")                     # headless backend (no display needed)
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    axes[0].imshow(_rgb(before, bands, max_reflect)); axes[0].set_title("Before (2019)")
    axes[1].imshow(_rgb(after, bands, max_reflect)); axes[1].set_title("After (2023)")
    axes[2].imshow(_rgb(after, bands, max_reflect))
    axes[2].imshow(np.ma.masked_where(label == 0, label),
                   cmap=ListedColormap(["red"]), vmin=0, vmax=1, alpha=0.55)
    axes[2].set_title("After + Hansen loss (red)")
    for a in axes:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# --------------------------------------------------------------------------
# Region per-band means (for the training-time normalisation option)
# --------------------------------------------------------------------------

def region_means(bbox, bands, before, after, cfg, cloud_pct=None):
    """
    Per-band mean reflectance over the region, for before and after composites.
    Same reducer/settings as src/normalise.py (mean, scale 30, maxPixels 1e9,
    bestEffort). Returns {"before": {band: mean}, "after": {band: mean}}.

    Uses the dataset cloud_pct (passed in) so the means match the stored patches.
    """
    region = ee.Geometry.Rectangle(bbox)
    scale = cfg.get("means_scale_m", 30)
    cloud_pct = cloud_pct if cloud_pct is not None else cfg["cloud_pct"]

    def _means(start, end):
        img = s2_composite(region, start, end, cloud_pct, cfg["s2_collection"]).select(bands)
        d = img.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region, scale=scale,
            maxPixels=int(1e9), bestEffort=True,
        ).getInfo()
        return {b: d.get(b) for b in bands}

    return {"before": _means(before[0], before[1]),
            "after": _means(after[0], after[1])}
