"""
evaluate.py — score the detected change map against Hansen ground truth.

Pixel-overlap confusion counts at `scale` metres, then precision/recall/F1.

  TP = detected AND truth
  FP = detected AND (NOT truth)
  FN = (NOT detected) AND truth
  precision = TP / (TP + FP)
  recall    = TP / (TP + FN)
  f1        = 2 * P * R / (P + R)

NOTE on JS -> Python: .and()/.or()/.not() become .And()/.Or()/.Not() here.

WHY NOT ACCURACY AS THE HEADLINE METRIC:
  Forest-loss pixels are a tiny fraction of the scene (severe class imbalance).
  A trivial model that predicts "no change everywhere" would score ~99%
  accuracy while finding zero deforestation. Precision/recall/F1 focus on the
  rare positive class, so they actually measure how well we detect loss.
"""

import ee


def confusion_counts(detected, truth, region, scale=30):
    """
    Compute TP/FP/FN pixel counts by summing overlap over the region.

    Each count is a server-side ee.Reducer.sum() (scale, maxPixels=1e9,
    bestEffort=True). We pack the three into one image and do a single
    reduceRegion + .getInfo() to bring the numbers down in one round-trip.

    Returns:
        dict {tp, fp, fn} as floats (pixel counts).
    """
    not_truth = truth.Not()        # capital N — JS->Python quirk

    tp = detected.And(truth).rename("tp")
    fp = detected.And(not_truth).rename("fp")
    fn = detected.Not().And(truth).rename("fn")

    counts_img = tp.addBands(fp).addBands(fn)
    counts = counts_img.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=scale,
        maxPixels=int(1e9),
        bestEffort=True,
    ).getInfo()   # <- brings the numbers down from Google's servers

    # getInfo can return None for a band if it is fully masked; coerce to 0.
    return {
        "tp": float(counts.get("tp") or 0.0),
        "fp": float(counts.get("fp") or 0.0),
        "fn": float(counts.get("fn") or 0.0),
    }


def precision_recall_f1(tp, fp, fn):
    """Standard P/R/F1 with divide-by-zero guards."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate(detected, truth, region, scale=30):
    """
    Full scoring: confusion counts + P/R/F1 in one dict.

    Returns:
        dict {tp, fp, fn, precision, recall, f1}.
    """
    counts = confusion_counts(detected, truth, region, scale)
    metrics = precision_recall_f1(counts["tp"], counts["fp"], counts["fn"])
    return {**counts, **metrics}
