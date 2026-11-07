"""
acquire.py — pull imagery and ground truth from Google Earth Engine.

Two functions, both faithful ports of the validated GEE Code Editor logic:
  * s2_composite(region, start, end, cloud_pct) -> cloud-filtered median composite
  * hansen_truth(region, y0, y1)                -> binary forest-loss mask (truth)

NOTE on JS -> Python quirks (used throughout this codebase):
  JavaScript .and()/.or()/.not()  become  .And()/.Or()/.Not()  in the Python API.

All of these return server-side ee objects. Nothing is computed until you call
.getInfo() (or similar) somewhere downstream — EE does the work on Google's servers.
"""

import ee


def s2_composite(region, start, end, cloud_pct, collection="COPERNICUS/S2_SR_HARMONIZED"):
    """
    Build a Sentinel-2 Surface Reflectance median composite over `region`.

    Faithful port of:
        COPERNICUS/S2_SR_HARMONIZED
          .filterBounds(region)
          .filterDate(start, end)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_pct))
          .median()
          .clip(region)

    Args:
        region:     ee.Geometry defining the area of interest.
        start, end: date strings "YYYY-MM-DD" (end is exclusive).
        cloud_pct:  drop scenes whose CLOUDY_PIXEL_PERCENTAGE >= this value.
        collection: image collection id (default: harmonised S2 SR).

    Returns:
        ee.Image — the median composite, clipped to the region.
    """
    composite = (
        ee.ImageCollection(collection)
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        .median()
        .clip(region)
    )
    return composite


def scene_count(region, start, end, cloud_pct, collection="COPERNICUS/S2_SR_HARMONIZED"):
    """
    How many Sentinel-2 scenes pass the SAME bounds/date/cloud filters that
    s2_composite() uses. Returns a server-side ee.Number (call .getInfo() to read).

    Used to detect an EMPTY composite before analysis: if this is 0, .median()
    would produce a band-less image and any .select('B4') would fail with
    "Band pattern 'B4' was applied to an Image with no bands".
    """
    return (
        ee.ImageCollection(collection)
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        .size()
    )


def hansen_truth(region, y0, y1, asset="UMD/hansen/global_forest_change_2023_v1_11"):
    """
    Build the binary forest-loss ground-truth mask from Hansen GFC.

    The 'lossyear' band encodes the year of tree-cover loss as an integer:
        0        = no loss
        1..23    = loss in year 2001..2023
    We mark a pixel as 'loss' (truth = 1) when lossyear is in [y0, y1].
    For loss during 2020-2023, pass y0=20, y1=23.

    Faithful port of:
        ee.Image('UMD/hansen/global_forest_change_2023_v1_11')
          .select('lossyear')
          .gte(y0).And(lossyear.lte(y1))

    Args:
        region: ee.Geometry defining the area of interest.
        y0, y1: inclusive lossyear bounds (e.g. 20 and 23).
        asset:  Hansen GFC asset id.

    Returns:
        ee.Image — a 1/0 mask where 1 = forest loss in [y0, y1].
    """
    lossyear = ee.Image(asset).clip(region).select("lossyear")
    # lossyear >= y0  AND  lossyear <= y1   (note .And(), capital A — JS->Python quirk)
    truth = lossyear.gte(y0).And(lossyear.lte(y1))
    return truth
