"""
visualize.py — turn ee.Image results into something displayable in the app.

Two kinds of helper:
  * vis-param builders for the folium/geemap map layers (interactive path);
  * get_thumb_url(), a robust fallback that fetches a static PNG via
    ee.Image.getThumbURL so the app can st.image(...) side by side if
    streamlit-folium / Leaflet misbehaves in this environment.

Nothing here downloads full rasters — map layers are tiled by Earth Engine and
thumbnails are small PNGs.
"""

import ee


# --- Visualisation parameters ----------------------------------------------

def true_colour_vis():
    """Sentinel-2 true colour (B4=red, B3=green, B2=blue)."""
    return {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000}


def loss_vis():
    """Detected change: solid red (interactive path, apply to detected.selfMask())."""
    return {"palette": ["red"]}


def truth_vis():
    """Hansen ground-truth loss: solid blue (interactive path, apply to truth.selfMask())."""
    return {"palette": ["blue"]}


# Two-colour versions for STATIC thumbnails: the binary mask is rendered
# region-wide (white = 0 / no change, colour = 1 / loss) so a standalone PNG
# reads clearly without transparency tricks.
def loss_vis_thumb():
    """Detected change for st.image thumbnails: white -> red over [0, 1]."""
    return {"min": 0, "max": 1, "palette": ["ffffff", "ff0000"]}


def truth_vis_thumb():
    """Hansen truth for st.image thumbnails: white -> blue over [0, 1]."""
    return {"min": 0, "max": 1, "palette": ["ffffff", "0000ff"]}


# --- Helpers ----------------------------------------------------------------

def bbox_center(bbox):
    """[minLon, minLat, maxLon, maxLat] -> (lat, lon) centre for map centring."""
    min_lon, min_lat, max_lon, max_lat = bbox
    return ((min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0)


def get_thumb_url(img, region, vis_params, dimensions=512):
    """
    Static PNG thumbnail URL for an ee.Image — the st.image fallback path.

    Args:
        img:        ee.Image to render.
        region:     ee.Geometry clip/extent for the thumbnail.
        vis_params: dict like true_colour_vis() (bands/min/max or palette).
        dimensions: max pixel size of the longer edge.

    Returns:
        A URL string you can pass to st.image(url).
    """
    params = dict(vis_params)
    params.update({"region": region, "dimensions": dimensions})
    return img.getThumbURL(params)
