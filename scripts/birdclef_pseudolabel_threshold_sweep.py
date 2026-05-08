#!/usr/bin/env python3
"""Sweep pseudo-label power/threshold knobs against labeled soundscape truth.

This is a lightweight Spec-B/noisy-student diagnostic.  It does not train a
student; it characterizes a teacher or student probability cache so we can pick
safer hard/soft pseudo-label settings before launching GPU jobs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from birdclef_pseudolabel_cache_summary import build_truth, macro_auc, topk_recall  # noqa: E402


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in str(text).replace(" ", "").split(",") if x]


def load_probs(path: Path, key: str | None) -> tuple[np.ndarray, list[str], np.ndarray, str]:
    z = np.load(path, allow_pickle=True)
    row_ids = z["row_ids"].astype(str)
    labels = [str(x) for x in z["labels"].astype(str).tolist()]
    if key is None:
        for candidate in ["probs", "pred_teacher", "pred_student", "pred_oof"]:
            if candidate in z.files:
                key = candidate
                break
    if key is None or key not in z.files:
        raise KeyError(f"Could not find probability key in {path}; keys={z.files}")
    probs = z[key].astype(np.float32)
    if probs.ndim != 2:
        raise ValueError(f"Expected 2D probability array for {key}, got {probs.shape}")
    if len(row_ids) != probs.shape[0] or len(labels) != probs.shape[1]:
        raise ValueError(
            f"Shape mismatch: row_ids={len(row_ids)}, labels={len(labels)}, probs={probs.shape}"
        )
    return row_ids, labels, np.clip(probs, 0.0, 1.0), key


def safe_div(num: float, den: float) -> float | None:
    return float(num / den) if den else None


def quantiles(arr: np.ndarray) -> dict[str, float]:
    return {str(q): float(np.quantile(arr, q)) for q in [0.5, 0.9, 0.95, 0.99, 0.999]}


def row_topk_positive_hits(y: np.ndarray, p: np.ndarray, k: int) -> dict[str, Any]:
    positives = y.sum(axis=1)
    rows = np.where(positives > 0)[0]
    if len(rows) == 0:
        return {"rows": 0, "mean_recall": None, "any_hit_rate": None}
    top = np.argpartition(-p, kth=min(k, p.shape[1] - 1), axis=1)[:, :k]
    recalls = []
    any_hits = []
    for i in rows:
        hit = float(y[i, top[i]].sum())
        recalls.append(hit / max(float(positives[i]), 1.0))
        any_hits.append(1.0 if hit > 0 else 0.0)
    return {"rows": int(len(rows)), "mean_recall": float(np.mean(recalls)), "any_hit_rate": float(np.mean(any_hits))}


def threshold_stats(y: np.ndarray, p: np.ndarray, thresholds: list[float]) -> dict[str, Any]:
    total_true = float(y.sum())
    out: dict[str, Any] = {}
    for t in thresholds:
        pos = p >= t
        pos_cells = int(pos.sum())
        true_pos = float((pos & (y > 0)).sum())
        out[str(t)] = {
            "positive_cells": pos_cells,
            "positive_rows": int(pos.any(axis=1).sum()),
            "positive_classes": int(pos.any(axis=0).sum()),
            "mean_positive_per_row": float(pos.sum(axis=1).mean()),
            "true_positive_cells_captured": int(true_pos),
            "true_cell_recall": safe_div(true_pos, total_true),
            "positive_cell_precision_vs_truth": safe_div(true_pos, float(pos_cells)),
        }
    return out


def hard_mask_stats(y: np.ndarray, p: np.ndarray, pos_threshold: float, neg_threshold: float) -> dict[str, Any]:
    pos = p >= pos_threshold
    neg = p <= neg_threshold
    mask = pos | neg
    total_true = float(y.sum())
    tp = float((pos & (y > 0)).sum())
    fp = float((pos & (y <= 0)).sum())
    fn_masked = float(((~pos) & (y > 0) & mask).sum())
    return {
        "positive_threshold": float(pos_threshold),
        "negative_threshold": float(neg_threshold),
        "positive_cells": int(pos.sum()),
        "negative_cells": int(neg.sum()),
        "ignored_cells": int((~mask).sum()),
        "masked_fraction": float(mask.mean()),
        "positive_rows": int(pos.any(axis=1).sum()),
        "positive_classes": int(pos.any(axis=0).sum()),
        "true_positive_cells_captured": int(tp),
        "true_cell_recall": safe_div(tp, total_true),
        "positive_cell_precision_vs_truth": safe_div(tp, tp + fp),
        "masked_false_negative_cells": int(fn_masked),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-npz", type=Path, required=True)
    ap.add_argument("--labels-csv", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--prob-key", default=None)
    ap.add_argument("--powers", default="0.75,0.85,1.0,1.15,1.3")
    ap.add_argument("--positive-thresholds", default="0.5,0.9,0.95,0.98")
    ap.add_argument("--negative-thresholds", default="0.01,0.02,0.05")
    args = ap.parse_args()

    row_ids, labels, probs, prob_key = load_probs(args.pred_npz, args.prob_key)
    y = build_truth(pd.read_csv(args.labels_csv), row_ids, labels)
    powers = parse_float_list(args.powers)
    pos_thresholds = parse_float_list(args.positive_thresholds)
    neg_thresholds = parse_float_list(args.negative_thresholds)

    result: dict[str, Any] = {
        "status": "threshold_sweep_complete",
        "pred_npz": str(args.pred_npz),
        "prob_key": prob_key,
        "labels_csv": str(args.labels_csv),
        "n_rows": int(len(row_ids)),
        "n_classes": int(len(labels)),
        "truth_positive_cells": int(y.sum()),
        "truth_rows_with_positive": int((y.sum(axis=1) > 0).sum()),
        "truth_classes_with_positive": int((y.sum(axis=0) > 0).sum()),
        "base_auc_summary": macro_auc(y, probs),
        "base_topk_recall": {str(k): topk_recall(y, probs, k) for k in [1, 3, 5, 10]},
        "base_row_topk_hits": {str(k): row_topk_positive_hits(y, probs, k) for k in [1, 3, 5, 10]},
        "powers": {},
        "hard_masks": [],
    }

    for power in powers:
        p = np.clip(probs, 1e-7, 1.0) ** float(power)
        result["powers"][str(power)] = {
            "prob_quantiles": quantiles(p),
            "prob_mean": float(p.mean()),
            "prob_max": float(p.max()),
            "thresholds": threshold_stats(y, p, pos_thresholds),
        }
        for pos_t in pos_thresholds:
            if pos_t <= max(neg_thresholds):
                continue
            for neg_t in neg_thresholds:
                result["hard_masks"].append({
                    "power": float(power),
                    **hard_mask_stats(y, p, pos_t, neg_t),
                })

    # Conservative shortlist: configs with at least one pseudo-positive class and
    # non-zero truth precision, sorted by precision then true-cell recall.
    shortlist = [m for m in result["hard_masks"] if m["positive_classes"] > 0 and (m["positive_cell_precision_vs_truth"] or 0) > 0]
    shortlist.sort(key=lambda m: ((m["positive_cell_precision_vs_truth"] or 0), (m["true_cell_recall"] or 0), -m["positive_cells"]), reverse=True)
    result["conservative_shortlist"] = shortlist[:12]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "n_rows": result["n_rows"],
        "base_auc_summary": result["base_auc_summary"],
        "base_topk_recall": result["base_topk_recall"],
        "shortlist_top3": result["conservative_shortlist"][:3],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
