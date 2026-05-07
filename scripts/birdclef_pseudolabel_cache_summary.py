#!/usr/bin/env python3
"""Summarize BirdCLEF pseudo-label cache quality and class balance.

Takes a row-level probability NPZ from birdclef_sed_soundscape_infer.py and
compares it to train_soundscapes_labels.csv when available. The output is a
small JSON artifact suitable for AutoResearch logs: macro AUC, top-k recall,
positive/negative threshold counts, and pseudo-label class histograms.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def row_id_from_label_row(row: pd.Series) -> str:
    end = str(row["end"])
    # Expected HH:MM:SS; robustly fall back to numeric seconds if present.
    if ":" in end:
        parts = [int(x) for x in end.split(":")]
        sec = parts[-1] + 60 * parts[-2] + (3600 * parts[-3] if len(parts) >= 3 else 0)
    else:
        sec = int(float(end))
    return f"{Path(str(row['filename'])).stem}_{sec}"


def build_truth(labels_df: pd.DataFrame, row_ids: np.ndarray, labels: list[str]) -> np.ndarray:
    label_to_idx = {label: i for i, label in enumerate(labels)}
    by_row: dict[str, list[str]] = {}
    for _, row in labels_df.iterrows():
        rid = row_id_from_label_row(row)
        vals = [x for x in str(row["primary_label"]).split(";") if x]
        by_row[rid] = vals
    y = np.zeros((len(row_ids), len(labels)), dtype=np.float32)
    missing = 0
    for i, rid in enumerate(row_ids):
        vals = by_row.get(str(rid))
        if vals is None:
            missing += 1
            continue
        for label in vals:
            j = label_to_idx.get(str(label))
            if j is not None:
                y[i, j] = 1.0
    return y


def macro_auc(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    vals = []
    for j in range(y.shape[1]):
        col = y[:, j]
        if col.min() == col.max():
            continue
        try:
            vals.append(float(roc_auc_score(col, p[:, j])))
        except Exception:
            pass
    return {"macro_auc": float(np.mean(vals)) if vals else None, "valid_classes": len(vals)}


def topk_recall(y: np.ndarray, p: np.ndarray, k: int) -> float | None:
    positives = y.sum(axis=1)
    has_pos = positives > 0
    if not np.any(has_pos):
        return None
    top = np.argpartition(-p, kth=min(k, p.shape[1] - 1), axis=1)[:, :k]
    hits = []
    for i in np.where(has_pos)[0]:
        hits.append(float(y[i, top[i]].sum() / max(positives[i], 1.0)))
    return float(np.mean(hits)) if hits else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-npz", type=Path, required=True)
    ap.add_argument("--labels-csv", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--top-preview", type=int, default=5)
    args = ap.parse_args()

    z = np.load(args.pred_npz, allow_pickle=True)
    row_ids = z["row_ids"].astype(str)
    labels = [str(x) for x in z["labels"].astype(str).tolist()]
    probs = z["probs"].astype(np.float32)
    labels_df = pd.read_csv(args.labels_csv)
    y = build_truth(labels_df, row_ids, labels)

    thresholds = [0.90, 0.95, 0.98]
    neg_thresholds = [0.01, 0.02, 0.05]
    positive_counts = {str(t): int((probs >= t).sum()) for t in thresholds}
    positive_rows = {str(t): int(((probs >= t).sum(axis=1) > 0).sum()) for t in thresholds}
    negative_counts = {str(t): int((probs <= t).sum()) for t in neg_thresholds}

    true_class_count = y.sum(axis=0)
    pseudo_class_count = {str(t): (probs >= t).sum(axis=0).astype(int) for t in thresholds}
    rare_true = int((true_class_count > 0).sum())
    pseudo_nonzero = {str(t): int((arr > 0).sum()) for t, arr in pseudo_class_count.items()}

    top_records = []
    for i in range(min(args.top_preview, len(row_ids))):
        idx = np.argsort(-probs[i])[:5]
        true_idx = np.where(y[i] > 0)[0]
        top_records.append({
            "row_id": str(row_ids[i]),
            "true_labels": [labels[j] for j in true_idx],
            "top_labels": [labels[j] for j in idx],
            "top_probs": [round(float(probs[i, j]), 6) for j in idx],
        })

    result: dict[str, Any] = {
        "status": "summary_complete",
        "pred_npz": str(args.pred_npz),
        "labels_csv": str(args.labels_csv),
        "n_rows": int(len(row_ids)),
        "n_classes": int(len(labels)),
        "truth_positive_cells": int(y.sum()),
        "truth_rows_with_positive": int((y.sum(axis=1) > 0).sum()),
        "truth_classes_with_positive": rare_true,
        "auc_summary": macro_auc(y, probs),
        "topk_recall": {str(k): topk_recall(y, probs, k) for k in [1, 3, 5, 10]},
        "positive_counts": positive_counts,
        "positive_rows": positive_rows,
        "pseudo_classes_with_positive": pseudo_nonzero,
        "negative_counts": negative_counts,
        "prob_stats": {
            "min": float(probs.min()),
            "max": float(probs.max()),
            "mean": float(probs.mean()),
            "p95": float(np.quantile(probs, 0.95)),
            "p99": float(np.quantile(probs, 0.99)),
        },
        "top_true_classes": [
            {"label": labels[i], "true_count": int(true_class_count[i])}
            for i in np.argsort(-true_class_count)[:20]
            if true_class_count[i] > 0
        ],
        "top_pseudo_classes_p95": [
            {"label": labels[i], "pseudo_count": int(pseudo_class_count["0.95"][i]), "true_count": int(true_class_count[i])}
            for i in np.argsort(-pseudo_class_count["0.95"])[:20]
            if pseudo_class_count["0.95"][i] > 0
        ],
        "top_records": top_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
