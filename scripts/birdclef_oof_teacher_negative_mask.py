#!/usr/bin/env python3
"""Build broad OOF-teacher negative/no-call masks from an OOF teacher cache.

The existing v13/v15 negative cache only overlaps a small fraction of the
OOF-teacher student rows. This helper derives low-probability negative masks
from the same OOF-only teacher cache used for soft targets, so aux-negative
training can measure whether broad no-call/background suppression helps without
using hidden/test labels or in-sample public dry-run outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def cap_lowest_per_row(mask: np.ndarray, scores: np.ndarray, max_per_row: int) -> np.ndarray:
    if max_per_row <= 0:
        return mask.copy()
    capped = mask.copy()
    for i in range(capped.shape[0]):
        idx = np.flatnonzero(capped[i])
        if len(idx) > max_per_row:
            keep = idx[np.argsort(scores[i, idx])[:max_per_row]]
            capped[i] = False
            capped[i, keep] = True
    return capped


def cap_lowest_per_class(mask: np.ndarray, scores: np.ndarray, max_per_class: int) -> np.ndarray:
    if max_per_class <= 0:
        return mask.copy()
    capped = mask.copy()
    for j in range(capped.shape[1]):
        idx = np.flatnonzero(capped[:, j])
        if len(idx) > max_per_class:
            keep = idx[np.argsort(scores[idx, j])[:max_per_class]]
            capped[:, j] = False
            capped[keep, j] = True
    return capped


def summarize(y_true: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    y = y_true.astype(bool)
    n = int(mask.sum())
    rows = int(mask.any(axis=1).sum())
    classes = int(mask.any(axis=0).sum())
    false_negative_cells = int((mask & y).sum())
    true_negative_cells = int((mask & ~y).sum())
    per_row = mask.sum(axis=1)
    per_class = mask.sum(axis=0)
    return {
        "cells": n,
        "rows": rows,
        "row_coverage_fraction": float(rows / max(mask.shape[0], 1)),
        "classes": classes,
        "class_coverage_fraction": float(classes / max(mask.shape[1], 1)),
        "true_negative_cells": true_negative_cells,
        "false_negative_cells": false_negative_cells,
        "negative_precision": float(true_negative_cells / n) if n else None,
        "false_negative_rate_within_mask": float(false_negative_cells / n) if n else None,
        "mean_cells_per_covered_row": float(n / max(rows, 1)),
        "p50_cells_per_row": float(np.quantile(per_row, 0.50)),
        "p95_cells_per_row": float(np.quantile(per_row, 0.95)),
        "max_cells_per_row": int(per_row.max()) if len(per_row) else 0,
        "max_cells_per_class": int(per_class.max()) if len(per_class) else 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-cache", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.03)
    ap.add_argument("--max-neg-per-row", type=int, default=64)
    ap.add_argument("--max-neg-per-class", type=int, default=0)
    ap.add_argument("--pred-key", default="teacher_pred")
    ap.add_argument("--truth-key", default="y_true")
    args = ap.parse_args()

    z = np.load(args.teacher_cache, allow_pickle=True)
    for key in ["files", "labels", args.pred_key, args.truth_key]:
        if key not in z.files:
            raise RuntimeError(f"Missing key {key!r} in {args.teacher_cache}")
    pred = z[args.pred_key].astype(np.float32)
    y_true = z[args.truth_key].astype(np.float32)
    if pred.shape != y_true.shape:
        raise RuntimeError(f"Prediction/truth shape mismatch: {pred.shape} vs {y_true.shape}")

    raw_negative = pred <= float(args.threshold)
    negative = cap_lowest_per_row(raw_negative, pred, int(args.max_neg_per_row))
    negative = cap_lowest_per_class(negative, pred, int(args.max_neg_per_class))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        files=z["files"].astype(str),
        labels=z["labels"].astype(str),
        y_true=y_true.astype(np.float32),
        teacher_pred=pred.astype(np.float32),
        negative_mask=negative.astype(np.bool_),
        raw_negative_mask=raw_negative.astype(np.bool_),
        source_teacher_cache=np.array([str(args.teacher_cache)]),
        threshold=np.array([float(args.threshold)], dtype=np.float32),
        max_neg_per_row=np.array([int(args.max_neg_per_row)], dtype=np.int32),
        max_neg_per_class=np.array([int(args.max_neg_per_class)], dtype=np.int32),
    )

    summary = {
        "status": "oof_teacher_negative_mask_complete",
        "teacher_cache": str(args.teacher_cache),
        "output_npz": str(args.output),
        "n_files": int(pred.shape[0]),
        "n_classes": int(pred.shape[1]),
        "threshold": float(args.threshold),
        "max_neg_per_row": int(args.max_neg_per_row),
        "max_neg_per_class": int(args.max_neg_per_class),
        "prob_stats": {
            "min": float(pred.min()),
            "max": float(pred.max()),
            "mean": float(pred.mean()),
            "p01": float(np.quantile(pred, 0.01)),
            "p05": float(np.quantile(pred, 0.05)),
            "p10": float(np.quantile(pred, 0.10)),
        },
        "raw_negative": summarize(y_true, raw_negative),
        "capped_negative": summarize(y_true, negative),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
