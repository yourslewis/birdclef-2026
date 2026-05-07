#!/usr/bin/env python3
"""Export clean OOF negative/positive pseudo-label masks.

This turns OOF prediction artifacts into a reusable pseudo-label cache for
regularization. It is deliberately OOF-only: in-sample teacher dry-runs can look
confident but leak validation labels. The main safe use case discovered so far is
low-probability negative/background/no-call regularization; high-confidence
positives are summarized but should be treated cautiously.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


def parse_spec(spec: str) -> tuple[str, Path, float]:
    parts = spec.split(":")
    if len(parts) == 1:
        path = Path(parts[0])
        return path.stem, path, 1.0
    if len(parts) == 2:
        name, path = parts
        return name, Path(path), 1.0
    return parts[0], Path(parts[1]), float(parts[2])


def load_oof(name: str, path: Path) -> dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    return {
        "name": name,
        "path": str(path),
        "files": z["files"].astype(str),
        "labels": z["labels"].astype(str),
        "y": z["y_oof"].astype(np.float32),
        "pred": z["pred_oof"].astype(np.float32),
    }


def align(items: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    labels = items[0]["labels"]
    for item in items[1:]:
        if not np.array_equal(labels, item["labels"]):
            raise RuntimeError(f"Label mismatch: {items[0]['name']} vs {item['name']}")
    common = set(items[0]["files"].tolist())
    for item in items[1:]:
        common &= set(item["files"].tolist())
    files = np.array(sorted(common), dtype=str)
    if len(files) == 0:
        raise RuntimeError("No overlapping files")
    y_ref = None
    preds: list[np.ndarray] = []
    for item in items:
        pos = {f: i for i, f in enumerate(item["files"])}
        idx = np.array([pos[f] for f in files], dtype=int)
        y = item["y"][idx]
        if y_ref is None:
            y_ref = y
        elif not np.array_equal(y_ref, y):
            raise RuntimeError(f"Aligned truth differs for {item['name']}")
        preds.append(item["pred"][idx])
    assert y_ref is not None
    return y_ref, files, labels, preds


def macro_auc(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    aucs: list[float] = []
    for j in range(y.shape[1]):
        col = y[:, j]
        if col.min() == col.max():
            continue
        try:
            aucs.append(float(roc_auc_score(col, pred[:, j])))
        except Exception:
            pass
    return {"macro_auc": float(np.mean(aucs)) if aucs else None, "valid_classes": len(aucs)}


def cap_mask(mask: np.ndarray, scores: np.ndarray, *, max_per_row: int | None, max_per_class: int | None, keep: str) -> np.ndarray:
    """Cap mask density. keep='lowest' for negatives, keep='highest' for positives."""
    capped = mask.copy()
    if max_per_row is not None and max_per_row > 0:
        for i in range(capped.shape[0]):
            idx = np.flatnonzero(capped[i])
            if len(idx) > max_per_row:
                order = np.argsort(scores[i, idx])
                if keep == "highest":
                    order = order[::-1]
                keep_idx = idx[order[:max_per_row]]
                capped[i] = False
                capped[i, keep_idx] = True
    if max_per_class is not None and max_per_class > 0:
        rng = np.random.default_rng(0)
        for j in range(capped.shape[1]):
            idx = np.flatnonzero(capped[:, j])
            if len(idx) > max_per_class:
                # For negatives keep the most confidently low scores; for positives keep highest.
                order = np.argsort(scores[idx, j])
                if keep == "highest":
                    order = order[::-1]
                keep_rows = idx[order[:max_per_class]]
                capped[:, j] = False
                capped[keep_rows, j] = True
    return capped


def summarize_mask(y: np.ndarray, mask: np.ndarray, positive_label: bool) -> dict[str, Any]:
    yb = y.astype(bool)
    n = int(mask.sum())
    rows = int(mask.any(axis=1).sum())
    classes = int(mask.any(axis=0).sum())
    if positive_label:
        tp = int((mask & yb).sum())
        fp = int((mask & ~yb).sum())
        return {
            "cells": n,
            "rows": rows,
            "classes": classes,
            "true_positive_cells": tp,
            "false_positive_cells": fp,
            "precision": float(tp / n) if n else None,
            "recall": float(tp / int(yb.sum())) if yb.sum() else None,
        }
    tn = int((mask & ~yb).sum())
    fn = int((mask & yb).sum())
    return {
        "cells": n,
        "rows": rows,
        "classes": classes,
        "true_negative_cells": tn,
        "false_negative_cells": fn,
        "negative_precision": float(tn / n) if n else None,
        "false_negative_rate_within_mask": float(fn / n) if n else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", action="append", required=True, help="name:path[:weight] (repeatable)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--negative-threshold", type=float, default=0.05)
    ap.add_argument("--positive-threshold", type=float, default=0.95)
    ap.add_argument("--max-neg-per-row", type=int, default=64)
    ap.add_argument("--max-neg-per-class", type=int, default=2000)
    ap.add_argument("--max-pos-per-row", type=int, default=5)
    ap.add_argument("--max-pos-per-class", type=int, default=200)
    args = ap.parse_args()

    specs = [parse_spec(s) for s in args.oof]
    items = [load_oof(name, path) for name, path, _ in specs]
    weights = np.array([w for _, _, w in specs], dtype=np.float64)
    weights = weights / weights.sum()
    y, files, labels, preds = align(items)
    pred = np.zeros_like(preds[0], dtype=np.float64)
    for w, arr in zip(weights, preds):
        pred += w * arr
    pred = pred.astype(np.float32)

    raw_negative = pred <= float(args.negative_threshold)
    raw_positive = pred >= float(args.positive_threshold)
    neg_mask = cap_mask(raw_negative, pred, max_per_row=args.max_neg_per_row, max_per_class=args.max_neg_per_class, keep="lowest")
    pos_mask = cap_mask(raw_positive, pred, max_per_row=args.max_pos_per_row, max_per_class=args.max_pos_per_class, keep="highest")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        files=files,
        labels=labels,
        y_true=y.astype(np.float32),
        teacher_pred=pred.astype(np.float32),
        negative_mask=neg_mask.astype(np.bool_),
        positive_mask=pos_mask.astype(np.bool_),
        raw_negative_mask=raw_negative.astype(np.bool_),
        raw_positive_mask=raw_positive.astype(np.bool_),
        weights=np.array(weights, dtype=np.float32),
        item_names=np.array([name for name, _, _ in specs]),
    )
    summary = {
        "status": "oof_negative_cache_complete",
        "output_npz": str(args.output),
        "items": [{"name": n, "path": str(p), "weight": float(w)} for n, p, w in specs],
        "weights_normalized": {name: float(w) for (name, _, _), w in zip(specs, weights)},
        "n_files": int(len(files)),
        "n_classes": int(len(labels)),
        "truth_positive_cells": int(y.sum()),
        "teacher_auc": macro_auc(y, pred),
        "prob_stats": {
            "min": float(pred.min()),
            "max": float(pred.max()),
            "mean": float(pred.mean()),
            "p95": float(np.quantile(pred, 0.95)),
            "p99": float(np.quantile(pred, 0.99)),
        },
        "thresholds": {
            "negative_threshold": float(args.negative_threshold),
            "positive_threshold": float(args.positive_threshold),
            "max_neg_per_row": int(args.max_neg_per_row),
            "max_neg_per_class": int(args.max_neg_per_class),
            "max_pos_per_row": int(args.max_pos_per_row),
            "max_pos_per_class": int(args.max_pos_per_class),
        },
        "raw_negative": summarize_mask(y, raw_negative, positive_label=False),
        "capped_negative": summarize_mask(y, neg_mask, positive_label=False),
        "raw_positive": summarize_mask(y, raw_positive, positive_label=True),
        "capped_positive": summarize_mask(y, pos_mask, positive_label=True),
    }
    summary_path = args.output.with_suffix(".summary.json")
    text = json.dumps(summary, indent=2)
    summary_path.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
