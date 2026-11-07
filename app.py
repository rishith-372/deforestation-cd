"""
app.py — Streamlit demo layer for the deforestation change-detection method.

STEP 2: full interactive controls.
  Sidebar: region dropdown (from config) or a custom bounding box, before/after
  date windows, an "Apply seasonal normalisation" toggle, and a "Run detection"
  button. The main panel shows the four image layers + the three metric cards,
  and — when normalisation is ON and a baseline for the same region/dates has
  already been run — the P/R/F1 delta vs that baseline.

DISPLAY = STATIC THUMBNAILS (robust fallback).
  geemap.foliumap failed to import in this environment (a geemap/basemap
  version clash crashes at import time), so per the spec we use the fallback:
  ee.Image.getThumbURL -> st.image. No geemap / folium / streamlit-folium, so
  there is nothing fragile to break at the viva. Thumbnails are small PNGs
  computed by Earth Engine — no full-raster download.

This file ONLY orchestrates and displays — all pipeline logic lives in src/.

Run:
    cd deforestation-cd
    streamlit run app.py

EARTH ENGINE AUTH: initialised with the project ID from config.yaml and your
local `earthengine authenticate` credentials — same as main.py.
"""

import os
from datetime import date

import yaml
import ee
import numpy as np
import streamlit as st

from src.pipeline import build_scene, all_regions
from src.evaluate import evaluate
from src.visualize import (
    true_colour_vis,
    loss_vis_thumb,
    truth_vis_thumb,
    get_thumb_url,
)
from src.deep_infer import run_region as run_deep_region

# Trained Siamese checkpoint (produced on Colab, copied into runs/).
# base = trained WITHOUT seasonal normalisation (the stronger model, F1 0.591).
DEEP_CHECKPOINT = "runs/siamese_base.pt"


# --- cached setup -----------------------------------------------------------

@st.cache_resource
def init_ee(project):
    """Initialise Earth Engine ONCE per app session (cached across reruns)."""
    ee.Initialize(project=project)
    return True


@st.cache_data
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


cfg = load_config()
init_ee(cfg["ee_project"])   # >>> your authenticated EE + project id from config <<<


def _scene_cfg(cloud_pct, threshold, scale):
    """Assemble the small cfg dict build_scene() needs (static params + overrides)."""
    return {
        "cloud_pct": cloud_pct,
        "threshold": threshold,
        "scale": scale,
        "s2_collection": cfg["s2_collection"],
        "hansen_asset": cfg["hansen_asset"],
        "hansen_loss_min": cfg["hansen_loss_min"],
        "hansen_loss_max": cfg["hansen_loss_max"],
    }


@st.cache_data(show_spinner=False)
def run_pipeline(bbox, before, after, normalise_on, cloud_pct, threshold, scale):
    """
    Run the existing pipeline for one setting and return ONLY plain, cacheable
    data (floats + thumbnail URL strings). ee objects never leave this function,
    which lets @st.cache_data memoise identical runs so re-runs don't recompute.

    Args are all hashable (tuples/strings/numbers) so they form the cache key.
    """
    scene = build_scene(list(bbox), list(before), list(after),
                        _scene_cfg(cloud_pct, threshold, scale), normalise_on)
    metrics = evaluate(scene["detected"], scene["truth"], scene["region"], scale)
    region = scene["region"]
    return {
        "metrics": metrics,
        "before_url": get_thumb_url(scene["before"], region, true_colour_vis()),
        "after_url": get_thumb_url(scene["after"], region, true_colour_vis()),
        "detected_url": get_thumb_url(scene["detected"], region, loss_vis_thumb()),
        "truth_url": get_thumb_url(scene["truth"], region, truth_vis_thumb()),
    }


@st.cache_data(show_spinner=False)
def run_deep(region, ckpt_mtime):
    """
    Score the Siamese model + the classical rule on a region's held-out TEST
    patches (local .npz, CPU). `ckpt_mtime` is only a cache key: it changes when
    the checkpoint file is (re)downloaded, invalidating the cached result.
    Returns the plain dict from src/deep_infer.run_region (numpy only).
    """
    return run_deep_region(region, cfg, DEEP_CHECKPOINT, n_examples=4)


def _ckpt_mtime():
    """Modification time of the checkpoint (0.0 if absent) — used as a cache key."""
    return os.path.getmtime(DEEP_CHECKPOINT) if os.path.exists(DEEP_CHECKPOINT) else 0.0


def _mask_img(mask, color):
    """Boolean (H,W) mask -> (H,W,3) uint8: `color` on a white background."""
    img = np.full(mask.shape + (3,), 255, np.uint8)
    img[mask] = color
    return img


# --- page -------------------------------------------------------------------

st.set_page_config(page_title="Deforestation Change Detection", layout="wide")
st.title("Seasonal-robust deforestation change detection")
st.caption(
    "Pick a region and two time windows, toggle the seasonal-normalisation "
    "stage, and compare the detected forest loss against Hansen ground truth."
)

with st.expander("About this method"):
    st.markdown(
        """
**Goal.** Detect deforestation between a *before* year (2019) and an *after*
year (2023) from satellite imagery, and measure how well it matches an
independent ground truth.

**Pipeline (each stage lives in `src/`):**
1. **Composites** — pull cloud-filtered median Sentinel-2 images for each
   window (`acquire.py`).
2. **Seasonal normalisation** *(the contribution, `normalise.py`)* — the two
   dates come from different seasons / sun angles, so their reflectance levels
   differ even where nothing changed. We rescale the NDVI bands (B8, B4) to
   their regional means so the dates are comparable before differencing.
   Toggle it on/off to see its effect.
3. **Change detection** (`detect.py`) — compute NDVI for each date, take the
   *after − before* difference, and flag pixels where it drops past the
   threshold as likely loss.
4. **Ground truth** (`acquire.py`) — Hansen Global Forest Change `lossyear`,
   masked to loss during 2020–2023.
5. **Scoring** (`evaluate.py`) — pixel-overlap Precision / Recall / F1 at 30 m.

**Why F1, not accuracy?** Loss pixels are a tiny fraction of the scene, so a
"nothing changed" guess would score ~99% accuracy while finding zero
deforestation. F1 focuses on the rare loss class.

**Headline result.** Seasonal normalisation improves F1 across the five study
regions (e.g. Rondônia F1 0.361 → 0.458).
        """
    )

# ============================ SIDEBAR CONTROLS =============================
st.sidebar.header("Controls")

regions = all_regions(cfg)                       # {name: bbox} dev + test
region_choice = st.sidebar.selectbox(
    "Region", list(regions.keys()) + ["Custom bounding box"]
)

if region_choice == "Custom bounding box":
    st.sidebar.caption("Enter the bounding box as four numbers (degrees).")
    default = regions["Rondonia"]                # sensible starting point
    min_lon = st.sidebar.number_input("min lon", value=float(default[0]), format="%.4f")
    min_lat = st.sidebar.number_input("min lat", value=float(default[1]), format="%.4f")
    max_lon = st.sidebar.number_input("max lon", value=float(default[2]), format="%.4f")
    max_lat = st.sidebar.number_input("max lat", value=float(default[3]), format="%.4f")
    bbox = [min_lon, min_lat, max_lon, max_lat]
    region_label = "Custom"
else:
    bbox = regions[region_choice]
    region_label = region_choice

# Date windows, defaulting to the config windows.
b0, b1 = (date.fromisoformat(cfg["before"][0]), date.fromisoformat(cfg["before"][1]))
a0, a1 = (date.fromisoformat(cfg["after"][0]), date.fromisoformat(cfg["after"][1]))
before_range = st.sidebar.date_input("Before window", value=(b0, b1))
after_range = st.sidebar.date_input("After window", value=(a0, a1))

normalise_on = st.sidebar.checkbox("Apply seasonal normalisation", value=True)
run = st.sidebar.button("Run detection", type="primary")

# Threshold / cloud_pct come from config (shown read-only for transparency).
st.sidebar.caption(
    f"threshold={cfg['threshold']}  cloud_pct<{cfg['cloud_pct']}  scale={cfg['scale']}m"
)


# ============================ RUN ON CLICK =================================
def _as_pair(rng):
    """st.date_input may return a single date mid-selection; require a 2-tuple."""
    if isinstance(rng, (tuple, list)) and len(rng) == 2:
        return rng
    return None


if run:
    bpair, apair = _as_pair(before_range), _as_pair(after_range)
    if bpair is None or apair is None:
        st.sidebar.error("Pick a start AND end date for both windows.")
    elif not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        st.sidebar.error(
            "Bounding box is invalid: min lon must be less than max lon, "
            "and min lat less than max lat."
        )
    else:
        before = (bpair[0].isoformat(), bpair[1].isoformat())
        after = (apair[0].isoformat(), apair[1].isoformat())
        with st.spinner("Earth Engine is computing composites and metrics ..."):
            result = run_pipeline(
                tuple(bbox), before, after, normalise_on,
                cfg["cloud_pct"], cfg["threshold"], cfg["scale"],
            )

        # Baseline cache: key a run by everything EXCEPT the normalise flag, so a
        # baseline (normalise OFF) and a normalised run on the same region/dates
        # share a key and we can show the delta between them.
        key = (region_label, before, after, cfg["cloud_pct"], cfg["threshold"], cfg["scale"])
        st.session_state.setdefault("baseline", {})
        if not normalise_on:
            st.session_state["baseline"][key] = result["metrics"]

        # Stash the latest run so the page keeps showing it across Streamlit reruns.
        st.session_state["last"] = {
            "result": result,
            "normalise_on": normalise_on,
            "region_label": region_label,
            "bbox": list(bbox),
            "before": before,
            "after": after,
            "key": key,
        }


# ============================ MAIN PANEL ===================================
if "last" not in st.session_state:
    st.info("Set your options in the sidebar and click **Run detection**.")
    st.stop()

last = st.session_state["last"]
result = last["result"]
m = result["metrics"]

st.subheader(
    f"{last['region_label']} — {'normalised' if last['normalise_on'] else 'baseline'}"
)
st.caption(
    f"before {last['before'][0]}..{last['before'][1]}  |  "
    f"after {last['after'][0]}..{last['after'][1]}  |  bbox {last['bbox']}"
)

# --- imagery + change maps (static thumbnails, side by side) ----------------
def _show(col, url, caption):
    """Full-width image. Newer Streamlit uses width="stretch"; older versions
    only accept use_container_width=True, so fall back to that if needed."""
    try:
        col.image(url, caption=caption, width="stretch")
    except Exception:
        col.image(url, caption=caption, use_container_width=True)


r1c1, r1c2 = st.columns(2)
_show(r1c1, result["before_url"], "Before (true colour B4/B3/B2)")
_show(r1c2, result["after_url"], "After (true colour B4/B3/B2)")
r2c1, r2c2 = st.columns(2)
_show(r2c1, result["detected_url"], "Detected loss (red)")
_show(r2c2, result["truth_url"], "Hansen ground truth (blue)")

# --- metric cards (with delta vs baseline when available) -------------------
st.subheader("Metrics vs Hansen ground truth")

# Show a delta only for a normalised run that has a matching cached baseline.
base = None
if last["normalise_on"]:
    base = st.session_state.get("baseline", {}).get(last["key"])


def _card(col, label, value, base_value):
    if base_value is not None:
        col.metric(label, f"{value:.3f}", delta=f"{value - base_value:+.3f}")
    else:
        col.metric(label, f"{value:.3f}")


c1, c2, c3 = st.columns(3)
_card(c1, "Precision", m["precision"], base["precision"] if base else None)
_card(c2, "Recall", m["recall"], base["recall"] if base else None)
_card(c3, "F1", m["f1"], base["f1"] if base else None)

if last["normalise_on"] and base is None:
    st.caption(
        "Tip: run this region/date with **normalisation OFF** first — the next "
        "normalised run will then show the improvement delta on each card."
    )

st.caption(
    "Accuracy is intentionally not the headline metric — forest-loss pixels are "
    "a tiny fraction of the scene (severe class imbalance), so accuracy would "
    "look ~99% while telling us nothing about loss detection."
)


# ======================= DEEP-LEARNING DETECTOR ===========================
# Part 2 of the project: the Siamese network, trained on 256x256 patches, run
# here on this region's HELD-OUT test patches. For a fair, pixel-identical
# comparison the classical NDVI rule is scored on the SAME patches.
st.divider()
st.header("Deep-learning detector (Siamese network)")

region_label = last["region_label"]

if region_label == "Custom":
    st.info(
        "The deep model is evaluated on the pre-built patch dataset, which only "
        "exists for the five study regions — pick one of those (not a custom "
        "bounding box) to see the Siamese network's results."
    )
else:
    with st.spinner("Running the Siamese network on held-out test patches (CPU) ..."):
        deep = run_deep(region_label, _ckpt_mtime())

    st.caption(
        f"Scored on **{deep['n_patches']} held-out test patches** for "
        f"{region_label} — spatial blocks the model never saw in training. "
        "The classical NDVI rule is run on the *same* patches, so the two methods "
        "are compared on identical pixels. (This differs from the region-wide "
        "classical score above, which is computed live over the whole region.)"
    )

    if not deep["checkpoint_found"]:
        st.warning(
            "**Trained weights not found** at `runs/siamese_base.pt`. The classical "
            "numbers below are still live, but the deep-model column is empty until "
            "you copy the checkpoint down from Colab:\n\n"
            "1. In Google Drive, open `major project-2/deforestation-cd/runs/`.\n"
            "2. Download `siamese_base.pt` (and `metrics_base.json`).\n"
            "3. Put them in this project's `runs/` folder, then re-run detection."
        )

    # --- metric comparison: classical vs deep, on the same held-out pixels ----
    st.subheader("Precision / Recall / F1 on the same held-out patches")
    cls = deep["classical"]
    dp = deep["deep"]
    cc, dc = st.columns(2)
    cc.markdown("**Classical NDVI (on patches)**")
    cc.metric("F1", f"{cls['f1']:.3f}")
    cc.caption(f"Precision {cls['precision']:.3f}  ·  Recall {cls['recall']:.3f}")

    dc.markdown("**Siamese network (deep)**")
    if dp is not None:
        dc.metric("F1", f"{dp['f1']:.3f}", delta=f"{dp['f1'] - cls['f1']:+.3f} vs classical")
        dc.caption(f"Precision {dp['precision']:.3f}  ·  Recall {dp['recall']:.3f}")
    else:
        dc.metric("F1", "—")
        dc.caption("download the checkpoint to populate this")

    # --- example patches: before | after | classical | deep | truth -----------
    st.subheader("Example patches (the clearest loss cases in this region)")
    st.caption("Red = predicted loss · Blue = Hansen ground truth.")
    RED, BLUE = (220, 30, 30), (30, 60, 220)
    head = st.columns(5)
    for col, title in zip(head, ["Before", "After", "Classical", "Deep", "Truth"]):
        col.markdown(f"**{title}**")
    for e in deep["examples"]:
        c = st.columns(5)
        c[0].image(e["before"], use_container_width=True)
        c[1].image(e["after"], use_container_width=True)
        c[2].image(_mask_img(e["classical"], RED), use_container_width=True)
        if e["deep"] is not None:
            c[3].image(_mask_img(e["deep"], RED), use_container_width=True)
        else:
            c[3].info("no weights")
        c[4].image(_mask_img(e["truth"], BLUE), use_container_width=True)

    st.caption(
        "Takeaway for the viva: the deep model is scored only on data it never "
        "trained on, with the identical metric used by the classical method — so "
        "the two numbers are directly comparable. Across the full test set the "
        "Siamese network reaches F1 ≈ 0.59, above the classical method's best."
    )
