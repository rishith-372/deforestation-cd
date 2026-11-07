"""
normalise.py — THE CONTRIBUTION: seasonal / illumination normalisation.

Idea: the 2019 "before" and 2023 "after" composites are from different seasons
and sun angles, so their absolute reflectance levels differ even where nothing
changed on the ground. That offset leaks into the NDVI difference and produces
false "change". We remove the gross per-band offset by rescaling each band to
its own regional mean, so the two dates sit on a comparable footing before we
difference their NDVI.

Faithful port of the validated GEE logic:
  for each band in ['B8','B4']:
      mean = band.reduceRegion(ee.Reducer.mean(), scale=30, maxPixels=1e9,
                               bestEffort=True)
      band_normalised = band / mean
  (B8 and B4 are what NDVI uses, so only those two need rescaling.)

This function always normalises; whether it is applied at all is controlled by
a boolean toggle in pipeline.py (so baseline vs normalised is a one-flag switch).
"""

import ee


def normalise(img, region, bands=("B8", "B4"), scale=30):
    """
    Rescale each of `bands` by its regional mean and return the updated image.

    Args:
        img:    ee.Image composite to normalise.
        region: ee.Geometry over which each band's mean is computed.
        bands:  bands to rescale (default B8, B4 — the NDVI inputs).
        scale:  metres for the mean reduction (default 30).

    Returns:
        ee.Image identical to `img` but with each named band divided by its
        own regional mean (other bands untouched).
    """
    out = img
    for b in bands:
        # Mean of this band over the region (computed on Google's servers).
        mean_dict = img.select(b).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=scale,
            maxPixels=int(1e9),
            bestEffort=True,
        )
        mean = ee.Number(mean_dict.get(b))

        # band / mean, keeping the same band name so NDVI still finds B8/B4.
        rescaled = img.select(b).divide(mean).rename(b)

        # overwrite=True replaces the original band with the rescaled one.
        out = out.addBands(srcImg=rescaled, overwrite=True)
    return out
