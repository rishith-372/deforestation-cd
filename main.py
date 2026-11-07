"""
main.py — full batch runner for the deforestation change-detection pipeline.

Loops over every region in config.yaml, runs each twice (baseline = no
normalisation, then normalised), prints a comparison table with a mean row,
and writes every run to results.json.

Run (from the deforestation-cd/ folder, with your authenticated EE env):
    python main.py

VALIDATION TARGET (from your GEE prototype) — Rondonia:
    baseline   F1 ~= 0.361  (P 0.283, R 0.498)
    normalised F1 ~= 0.458  (P 0.347, R 0.676)
If Rondonia's printed numbers are far from these, the port mistranslated
something (or cloud_pct in config.yaml differs from the prototype) — flag it.

>>> Requires config.yaml 'ee_project' to be your real project ID, and
    `earthengine authenticate` to have been run. <<<
"""

import json
import yaml
import ee
import pandas as pd

from src.pipeline import run_region, all_regions


def load_config(path="config.yaml"):
    """Read all parameters from config.yaml (single source of truth)."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    print(f"Initialising Earth Engine (project: {cfg['ee_project']}) ...")
    ee.Initialize(project=cfg["ee_project"])

    regions = all_regions(cfg)   # {name: bbox} across dev + test
    results = []                 # flat list of run dicts -> results.json
    rows = []                    # rows for the printed comparison table

    for name, bbox in regions.items():
        print(f"\n=== {name} {bbox} ===")

        base = run_region(name, bbox, False, cfg)   # baseline (normalise off)
        print(f"  baseline    P={base['precision']:.3f}  R={base['recall']:.3f}  F1={base['f1']:.3f}")

        norm = run_region(name, bbox, True, cfg)    # normalised (normalise on)
        print(f"  normalised  P={norm['precision']:.3f}  R={norm['recall']:.3f}  F1={norm['f1']:.3f}")

        results.extend([base, norm])
        rows.append({
            "region": name,
            "base_P": base["precision"], "base_R": base["recall"], "base_F1": base["f1"],
            "norm_P": norm["precision"], "norm_R": norm["recall"], "norm_F1": norm["f1"],
        })

    # Comparison table (region x baseline P/R/F1 vs normalised P/R/F1) + mean row.
    df = pd.DataFrame(rows).set_index("region")
    df.loc["MEAN"] = df.mean(numeric_only=True)

    print("\n============== COMPARISON: baseline vs normalised ==============")
    print(df.round(3).to_string())
    mean_delta = df.loc["MEAN", "norm_F1"] - df.loc["MEAN", "base_F1"]
    print(f"\nHeadline: normalisation changes mean F1 by {mean_delta:+.3f}.")

    # Persist every run (the spec's results.json shape).
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote results.json ({len(results)} runs).")


if __name__ == "__main__":
    main()
