"""
detect.py — NDVI and the NDVI-difference change map.

NDVI = (B8 - B4) / (B8 + B4)  (vegetation index; high = green/forest).
Deforestation drives NDVI down, so a strongly negative after-minus-before NDVI
difference flags likely loss.

Faithful port of the validated GEE logic:
  ndvi(img)            = img.normalizedDifference(['B8','B4'])
  diff                 = ndvi(after) - ndvi(before)
  detected (binary)    = diff.lt(threshold)      # threshold default -0.20
"""

import ee


def ndvi(img):
    """NDVI as a single-band ee.Image named 'NDVI'."""
    return img.normalizedDifference(["B8", "B4"]).rename("NDVI")


def change_map(before_img, after_img, threshold=-0.20):
    """
    Binary change map: 1 where NDVI dropped by more than |threshold|, else 0.

    Args:
        before_img: ee.Image composite for the "before" window.
        after_img:  ee.Image composite for the "after" window.
        threshold:  NDVI-difference cutoff (default -0.20). More negative diff
                    => clearer loss; diff < threshold is flagged as change.

    Returns:
        ee.Image (1/0) named 'detected'.
    """
    diff = ndvi(after_img).subtract(ndvi(before_img))
    detected = diff.lt(threshold).rename("detected")
    return detected
