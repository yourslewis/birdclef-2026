#!/usr/bin/env python3
"""Sweep OOF teacher-cache pseudo-label thresholds using embedded y_true.

Unlike birdclef_pseudolabel_threshold_sweep.py, this accepts file-level OOF
teacher caches whose NPZ contains `files`, `labels`, `y_true`, and `teacher_pred`.
It is a no-submit diagnostic for pseudo-label/cache redesign: quantify whether
hard positives/negatives are precise enough, class coverage is broad enough, and
which thresholds are safe before launching another student.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


def parse_float_list(text: str) -> list[float]:
    vals = [float(x) for x in str(text).replace(" ", "").split(",") if x]
    if not vals:
        raise ValueError("empty float list")
    return vals


def safe_div(num: float, den: float) -> float | None:
    return float(num / den) if den else None


def macro_auc(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    vals = []
    per_class = []
    for j in range(y.shape[1]):
        col = y[:, j]
        if col.min() == col.max():
            per_class.append(None)
            continue
        try:
            auc = float(roc_auc_score(col, p[:, j]))
            vals.append(auc)
            per_class.append(auc)
        except Exception:
            per_class.append(None)
    return {"macro_auc": float(np.mean(vals)) if vals else None, "valid_classes": len(vals), "per_class_auc": per_class}


def topk_recall(y: np.ndarray, p: np.ndarray, k: int) -> float | None:
    positives = y.sum(axis=1)
    rows = np.where(positives > 0)[0]
    if rows.size == 0:
        return None
    top = np.argpartition(-p, kth=min(k, p.shape[1] - 1), axis=1)[:, :k]
    recalls = []
    for i in rows:
        recalls.append(float(y[i, top[i]].sum() / max(positives[i], 1.0)))
    return float(np.mean(recalls)) if recalls else None


def threshold_stats(y: np.ndarray, p: np.ndarray, t: float) -> dict[str, Any]:
    pred = p >= t
    true = y > 0
    tp = float((pred & true).sum())
    fp = float((pred & ~true).sum())
    total_true = float(true.sum())
    per_class_pos = pred.sum(axis=0)
    true_class_count = true.sum(axis=0)
    covered_true_classes = int(((per_class_pos > 0) & (true_class_count > 0)).sum())
    return {
        "threshold": float(t),
        "positive_cells": int(pred.sum()),
        "positive_rows": int(pred.any(axis=1).sum()),
        "positive_classes": int((per_class_pos > 0).sum()),
        "covered_true_classes": covered_true_classes,
        "mean_positive_per_row": float(pred.sum(axis=1).mean()),
        "true_positive_cells_captured": int(tp),
        "true_cell_recall": safe_div(tp, total_true),
        "positive_cell_precision_vs_truth": safe_div(tp, tp + fp),
    }


def hard_mask_stats(y: np.ndarray, p: np.ndarray, pos_t: float, neg_t: float) -> dict[str, Any]:
    pos = p >= pos_t
    neg = p <= neg_t
    true = y > 0
    mask = pos | neg
    tp = float((pos & true).sum())
    fp = float((pos & ~true).sum())
    tn = float((neg & ~true).sum())
    fn_neg = float((neg & true).sum())
    total_true = float(true.sum())
    return {
        "positive_threshold": float(pos_t),
        "negative_threshold": float(neg_t),
        "positive_cells": int(pos.sum()),
        "negative_cells": int(neg.sum()),
        "ignored_cells": int((~mask).sum()),
        "masked_fraction": float(mask.mean()),
        "positive_rows": int(pos.any(axis=1).sum()),
        "positive_classes": int(pos.any(axis=0).sum()),
        "true_positive_cells_captured": int(tp),
        "true_cell_recall": safe_div(tp, total_true),
        "positive_cell_precision_vs_truth": safe_div(tp, tp + fp),
        "negative_cell_precision_vs_truth": safe_div(tn, tn + fn_neg),
        "negative_false_negative_cells": int(fn_neg),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--pred-key", default="teacher_pred")
    ap.add_argument("--truth-key", default="y_true")
    ap.add_argument("--powers", default="0.75,0.85,1.0,1.15,1.3")
    ap.add_argument("--positive-thresholds", default="0.5,0.7,0.8,0.9,0.95,0.98")
    ap.add_argument("--negative-thresholds", default="0.001,0.005,0.01,0.02,0.05")
    args = ap.parse_args()

    z = np.load(args.cache, allow_pickle=True)
    labels = [str(x) for x in z["labels"].astype(str).tolist()]
    files = [str(x) for x in z["files"].astype(str).tolist()] if "files" in z.files else [str(i) for i in range(len(z[args.pred_key]))]
    y = z[args.truth_key].astype(np.float32)
    probs = np.clip(z[args.pred_key].astype(np.float32), 0.0, 1.0)
    if y.shape != probs.shape:
        raise ValueError(f"truth/pred shape mismatch: {y.shape} vs {probs.shape}")
    powers = parse_float_list(args.powers)
    pos_thresholds = parse_float_list(args.positive_thresholds)
    neg_thresholds = parse_float_list(args.negative_thresholds)
    true_class_count = y.sum(axis=0)

    base_auc = macro_auc(y, probs)
    result: dict[str, Any] = {
        "status": "oof_threshold_sweep_complete",
        "cache": str(args.cache),
        "pred_key": args.pred_key,
        "truth_key": args.truth_key,
        "n_rows": int(y.shape[0]),
        "n_classes": int(y.shape[1]),
        "truth_positive_cells": int(y.sum()),
        "truth_rows_with_positive": int((y.sum(axis=1) > 0).sum()),
        "truth_classes_with_positive": int((true_class_count > 0).sum()),
        "base_auc_summary": {k: v for k, v in base_auc.items() if k != "per_class_auc"},
        "base_topk_recall": {str(k): topk_recall(y, probs, k) for k in [1, 3, 5, 10]},
        "prob_stats": {
            "min": float(probs.min()),
            "max": float(probs.max()),
            "mean": float(probs.mean()),
            "p90": float(np.quantile(probs, 0.90)),
            "p95": float(np.quantile(probs, 0.95)),
            "p99": float(np.quantile(probs, 0.99)),
            "p999": float(np.quantile(probs, 0.999)),
        },
        "powers": {},
        "hard_masks": [],
        "top_true_classes": [
            {"label": labels[i], "true_count": int(true_class_count[i]), "auc": base_auc["per_class_auc"][i]}
            for i in np.argsort(-true_class_count)[:25]
            if true_class_count[i] > 0
        ],
    }

    for power in powers:
        p = np.clip(probs, 1e-7, 1.0) ** float(power)
        result["powers"][str(power)] = {
            "prob_stats": {
                "mean": float(p.mean()),
                "max": float(p.max()),
                "p90": float(np.quantile(p, 0.90)),
                "p95": float(np.quantile(p, 0.95)),
                "p99": float(np.quantile(p, 0.99)),
                "p999": float(np.quantile(p, 0.999)),
            },
            "thresholds": {str(t): threshold_stats(y, p, t) for t in pos_thresholds},
        }
        for pos_t in pos_thresholds:
            if pos_t <= max(neg_thresholds):
                continue
            for neg_t in neg_thresholds:
                result["hard_masks"].append({"power": float(power), **hard_mask_stats(y, p, pos_t, neg_t)})

    shortlist = [
        m for m in result["hard_masks"]
        if m["positive_classes"] >= 5 and (m["positive_cell_precision_vs_truth"] or 0.0) > 0.0
    ]
    shortlist.sort(
        key=lambda m: (
            (m["positive_cell_precision_vs_truth"] or 0.0),
            (m["true_cell_recall"] or 0.0),
            m["positive_classes"],
            -m["positive_cells"],
        ),
        reverse=True,
    )
    result["conservative_shortlist"] = shortlist[:20]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "cache": result["cache"],
        "n_rows": result["n_rows"],
        "base_auc_summary": result["base_auc_summary"],
        "base_topk_recall": result["base_topk_recall"],
        "prob_stats": result["prob_stats"],
        "shortlist_top5": result["conservative_shortlist"][:5],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
