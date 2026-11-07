# Deforestation Change Detection

Seasonal-robust deforestation change detection with Google Earth Engine and
Python. For a region, it pulls two Sentinel-2 composites (2019 "before" and
2023 "after"), optionally applies a **seasonal / illumination normalisation**
stage, computes an NDVI difference, thresholds it into a change map, and scores
that map against **Hansen Global Forest Change** ground truth using
Precision / Recall / F1 at 30 m.

**Headline result:** seasonal normalisation improves F1 across all five study
regions (e.g. Rondônia F1 0.361 → 0.458).

The project has **two parts**: (1) the classical NDVI pipeline above, and (2) a
**Siamese neural network** trained on image patches from the same five regions.
Both are wired into one Streamlit app. See
[Two-part results](#two-part-results-classical-method-vs-deep-model) for the full
comparison.

---

## Project layout

```
deforestation-cd/
├── config.yaml        # all parameters: EE project, date windows, cloud_pct,
│                      #   threshold, scale, Hansen asset, regions (dev + test)
├── requirements.txt
├── README.md
├── main.py            # batch runner: all regions x {baseline, normalised},
│                      #   prints a comparison table, writes results.json
├── app.py             # Streamlit demo app (classical panel + deep panel)
│
│   # --- Part 1: classical pipeline ---
├── src/
│   ├── acquire.py     # s2_composite(), hansen_truth()
│   ├── normalise.py   # normalise()  -- the contribution (per-band mean rescale)
│   ├── detect.py      # ndvi(), change_map()
│   ├── evaluate.py    # confusion_counts(), precision_recall_f1(), evaluate()
│   ├── pipeline.py    # build_scene() (images), run_region() (metrics)
│   └── visualize.py   # thumbnail / vis-param helpers for the app
│
│   # --- Part 2: Siamese network (patches + model) ---
├── build_dataset.py   # export 256x256 before/after/label patches (-> data/)
├── build_manifest.py  # region_means.json + manifest.csv (spatial-block splits)
├── train.py           # train/evaluate the Siamese U-Net (base vs normalised)
├── src/
│   ├── dataset_build.py  # patch tiling + Earth Engine export helpers
│   ├── dataset.py        # PatchDataset (serves patches to the model)
│   ├── model.py          # SiameseUNet (shared encoder + feature difference)
│   └── deep_infer.py     # run the trained model in the app, on held-out patches
│
├── data/              # built patches, manifest, region means (not in git)
└── runs/              # trained checkpoints + metrics (siamese_base.pt, ...)
```

`app.py` only orchestrates and displays — all method logic lives in `src/`, and
both `main.py` and the app call the same functions, so they run identical logic.

---

## Setup

Requires **Python 3.11** and a registered Earth Engine noncommercial project.

```bash
cd deforestation-cd
pip install -r requirements.txt
```

### Earth Engine authentication (one-time)

Earth Engine runs the computation on Google's servers; you authenticate once:

```bash
earthengine authenticate
```

This opens a browser, you approve, and credentials are cached locally. The code
then initialises with **your project ID**, which is read from `config.yaml`
(never hardcoded):

```yaml
ee_project: "fswd-project-479113"   # <-- set this to YOUR project ID
```

> If you see an auth or permission error on the first Earth Engine call, this
> project ID and `earthengine authenticate` are the first things to check.

---

## How to run

### 1. Batch comparison (headline numbers)

```bash
python main.py
```

Runs every region twice (baseline = no normalisation, then normalised), prints
a comparison table with a mean row, and writes every run to `results.json`
(a list of `{region, normalised, tp, fp, fn, precision, recall, f1}`).

**Validation target** (from the original Earth Engine prototype), Rondônia:

| run        | P     | R     | F1    |
|------------|-------|-------|-------|
| baseline   | 0.283 | 0.498 | 0.361 |
| normalised | 0.347 | 0.676 | 0.458 |

If Rondônia's numbers drift from these, something was mistranslated — or
`cloud_pct` in `config.yaml` differs from the prototype (it controls which
scenes enter the composite, so it directly affects the result).

### 2. Interactive app (viva demo)

```bash
streamlit run app.py
```

Then in the sidebar: choose a region (or enter a custom bounding box), set the
before/after windows, toggle **Apply seasonal normalisation**, and click
**Run detection**. The main panel shows the before image, after image, detected
loss (red), Hansen truth (blue), and Precision / Recall / F1 cards.

**Baseline-vs-normalised delta:** run a region with normalisation **off** first,
then **on** — the metric cards then show the improvement (Δ) over the baseline
for the same region and dates.

**Deep-learning panel.** Scroll below the classical results to the
**Deep-learning detector** section. For the selected region it scores the trained
Siamese network on that region's **held-out test patches**, and runs the classical
NDVI rule on the *same* patches for a pixel-identical comparison. This needs the
trained checkpoint locally:

> Copy `siamese_base.pt` (and `metrics_base.json`) from your Colab Drive
> `deforestation-cd/runs/` folder into this project's `runs/` folder. Until then
> the panel still shows the classical-on-patches numbers and a reminder to add the
> weights — it does not crash.

---

## Two-part results: classical method vs deep model

The project builds up in two stages on the same five regions and the same Hansen
ground truth, and compares them with the **identical** Precision / Recall / F1.

### Part 1 — classical NDVI pipeline

The hand-built method above: cloud-filtered Sentinel-2 composites → optional
seasonal normalisation → NDVI difference → threshold → score against Hansen.
Seasonal normalisation is the student contribution, and it helps:

| Region (classical) | baseline F1 | normalised F1 |
|--------------------|-------------|---------------|
| Rondônia           | 0.361       | **0.458**     |

Normalising each band to its regional mean lines the two dates up before
differencing, so the single global threshold (`-0.20`) fits better across
seasons — recall in particular climbs.

### Part 2 — Siamese neural network

A learned detector trained on image patches from the same regions.

**Dataset** (`build_dataset.py` + `build_manifest.py`): **2,154** patches of
**256 × 256** pixels at **10 m**, four bands (B2, B3, B4, B8) for both the 2019
*before* and 2023 *after* dates, each with a per-pixel Hansen loss label and a
`valid` mask. Patches are split train / val / test **70 / 15 / 15 by whole
spatial blocks** (seed 42), so neighbouring patches never leak across splits —
every test number is on ground the model never saw. Each region contributes
~60–66 held-out test patches.

**Model** (`src/model.py`): a **Siamese U-Net** (~7.8 M parameters). One shared
encoder runs over the before and after images; at every scale the network takes
the **absolute difference** of the two feature maps (so the decoder sees *what
changed*), and a U-Net decoder upsamples that to a per-pixel loss probability.
Training (`train.py`) uses masked BCE over valid pixels only, with a
`pos_weight` of 15 for the rare loss class; it ran for 30 epochs on a free
Google Colab T4 GPU, keeping the best-by-validation-F1 checkpoint.

The contribution (seasonal normalisation) was tested on the deep model too, by
training once on raw reflectance (`base`) and once on region-mean-normalised
inputs (`norm`). Scored on the full held-out test set:

| Model (deep, full test set) | Precision | Recall | F1        |
|-----------------------------|-----------|--------|-----------|
| Siamese, baseline (`base`)  | 0.478     | 0.772  | **0.591** |
| Siamese, normalised (`norm`)| 0.397     | 0.804  | 0.531     |

### What the comparison shows

| Method                              | best F1   |
|-------------------------------------|-----------|
| Classical NDVI, baseline            | 0.361     |
| Classical NDVI, **normalised**      | 0.458     |
| **Siamese network, baseline**       | **0.591** |
| Siamese network, normalised         | 0.531     |

Two findings, both defensible:

1. **The learned model beats the classical method** — F1 0.591 vs 0.458, a clear
   margin on held-out data.
2. **Seasonal normalisation helps the classical method but *not* the deep model**
   (classical +0.097, deep −0.060). The CNN — with its shared encoder, feature
   differencing and batch normalisation — already learns its own brightness
   invariances, so dividing by regional means removes absolute-reflectance signal
   the network was using. A hand-engineered normalisation that is valuable for a
   fixed threshold becomes redundant, and slightly harmful, for a model that can
   learn the same thing from data.

The app's deep panel lets you see this live, per region, on identical held-out
patches — classical and deep side by side.

> **Caveat (stated honestly):** these are single training runs per setting (one
> seed). The ~0.06 F1 gap is consistent with the precision/recall shift, but to
> make it bullet-proof you would repeat each setting over 2–3 seeds and report
> the mean ± spread.

### Which method to use when

| Situation | Best choice | Why |
|---|---|---|
| Labelled training data available + want maximum accuracy | **Siamese network** (baseline, no normalisation) — F1 **0.591** | Highest F1 on held-out data; learns season/illumination invariance by itself. |
| New region, no training labels, or it must "just work" | **Classical NDVI + normalisation** — F1 **0.458** | Unsupervised: needs only a threshold, no training, works on any region. |
| Transparency / explainability needed (defence, scientific or legal use) | **Classical NDVI** | Every step is interpretable pixel-by-pixel; the deep model is a black box. |
| Huge area, no GPU, instant results | **Classical NDVI** | Runs server-side on Earth Engine over whole regions in seconds. |

**The normalisation toggle (the contribution) cuts both ways:**

| Method | Normalisation | F1 effect |
|---|---|---|
| Classical NDVI | **ON** | 0.361 → **0.458** (+0.097) ✅ |
| Siamese network | **OFF** (raw reflectance) | 0.591, vs 0.531 if ON (−0.060) ❌ |

Rules of thumb: **best overall** = Siamese on raw reflectance (0.591); **best
classical / zero-data / explainable** = NDVI *with* normalisation (0.458); switch
normalisation **ON for NDVI, OFF for the deep model** — the CNN already learns
that invariance from data, so the manual version just removes useful signal.

A per-method flow diagram for each pipeline is in [`docs/`](docs/)
(`classical_ndvi_method.pdf`, `siamese_network_method.pdf`).

---

## Notes

- **Map rendering.** The app shows static thumbnails (`ee.Image.getThumbURL` →
  `st.image`) rather than an interactive Leaflet map, because `geemap.foliumap`
  fails to import in some environments (a geemap/basemap version clash). The
  static path has fewer moving parts and is the more defensible demo. To switch
  back to the interactive map, re-install the optional deps in
  `requirements.txt` and use `geemap.foliumap` + `streamlit-folium` in `app.py`
  (the `src/visualize.py` vis helpers support both paths).
- **Why not accuracy?** Forest-loss pixels are a tiny fraction of each scene
  (severe class imbalance), so accuracy would read ~99% while telling you
  nothing about loss detection. Precision/Recall/F1 target the rare loss class.
- **Ground-truth asset.** We use `UMD/hansen/global_forest_change_2023_v1_11`.
  Earth Engine warns it is superseded by a 2025 version; this is deliberate —
  the 2023 asset is what the validated results were measured against. Switching
  assets would change the ground truth and shift the numbers.
