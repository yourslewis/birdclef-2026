#!/usr/bin/env python3
"""Diagnose clean OOF prediction artifacts as pseudo-label teachers.

Loads one or more SED OOF `.npz` artifacts, aligns overlapping files/labels,
computes macro AUC/correlation/blend diagnostics, and summarizes hard-confidence
pseudo-label thresholds against the OOF ground truth. This is intended to keep
pseudo-label decisions grounded in clean OOF estimates instead of in-sample dry
runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


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
    return {
        "macro_auc": float(np.mean(aucs)) if aucs else None,
        "valid_classes": len(aucs),
    }


def topk_recall(y: np.ndarray, pred: np.ndarray, ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict[str, float]:
    positives = float(y.sum())
    out: dict[str, float] = {}
    if positives <= 0:
        return {str(k): 0.0 for k in ks}
    for k in ks:
        kk = min(k, pred.shape[1])
        idx = np.argpartition(pred, -kk, axis=1)[:, -kk:]
        mask = np.zeros_like(pred, dtype=bool)
        rows = np.arange(pred.shape[0])[:, None]
        mask[rows, idx] = True
        out[str(k)] = float((y.astype(bool) & mask).sum() / positives)
    return out


def threshold_stats(y: np.ndarray, pred: np.ndarray, thresholds: tuple[float, ...]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    yb = y.astype(bool)
    total_true = int(yb.sum())
    for t in thresholds:
        mask = pred >= t
        tp = int((mask & yb).sum())
        fp = int((mask & ~yb).sum())
        n = int(mask.sum())
        rows = int(mask.any(axis=1).sum())
        classes = int(mask.any(axis=0).sum())
        out[f">={t:g}"] = {
            "cells": n,
            "rows": rows,
            "classes": classes,
            "true_positive_cells": tp,
            "false_positive_cells": fp,
            "precision": float(tp / n) if n else None,
            "recall": float(tp / total_true) if total_true else None,
        }
    for t in (0.01, 0.02, 0.05):
        mask = pred <= t
        tn = int((mask & ~yb).sum())
        fn = int((mask & yb).sum())
        n = int(mask.sum())
        out[f"<={t:g}"] = {
            "cells": n,
            "true_negative_cells": tn,
            "false_negative_cells": fn,
            "negative_precision": float(tn / n) if n else None,
        }
    return out


def flat_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1])


def parse_spec(spec: str) -> tuple[str, Path, float]:
    parts = spec.split(":")
    if len(parts) == 1:
        path = Path(parts[0])
        return path.stem, path, 1.0
    if len(parts) == 2:
        name, path = parts
        return name, Path(path), 1.0
    name, path, weight = parts[0], parts[1], parts[2]
    return name, Path(path), float(weight)


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


def align(items: list[dict[str, Any]]) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
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
    preds = []
    y_ref = None
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
    return y_ref, preds, files, labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", action="append", required=True, help="name:path[:weight] (repeatable)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--threshold", type=float, action="append", default=[0.5, 0.7, 0.9, 0.95, 0.98])
    args = ap.parse_args()

    specs = [parse_spec(x) for x in args.oof]
    items = [load_oof(name, path) for name, path, _ in specs]
    weights = np.array([w for _, _, w in specs], dtype=np.float64)
    weights = weights / weights.sum()
    y, preds, files, labels = align(items)
    ensemble = np.zeros_like(preds[0], dtype=np.float64)
    for w, pred in zip(weights, preds):
        ensemble += w * pred
    ensemble = ensemble.astype(np.float32)

    standalone = []
    for item, pred in zip(items, preds):
        standalone.append({
            "name": item["name"],
            "path": item["path"],
            "auc": macro_auc(y, pred),
            "topk_recall": topk_recall(y, pred),
            "thresholds": threshold_stats(y, pred, tuple(args.threshold)),
            "prob_stats": {
                "min": float(pred.min()),
                "max": float(pred.max()),
                "mean": float(pred.mean()),
                "p95": float(np.quantile(pred, 0.95)),
                "p99": float(np.quantile(pred, 0.99)),
            },
        })

    corr = []
    for i, a in enumerate(preds):
        row = []
        for b in preds:
            row.append(flat_corr(a, b))
        corr.append(row)

    blend_grid = []
    if len(preds) == 2:
        for wb in [i / 20 for i in range(21)]:
            pa = (1 - wb) * preds[0] + wb * preds[1]
            blend_grid.append({"weight_second": wb, **macro_auc(y, pa)})
    result = {
        "status": "oof_pseudolabel_diagnostics_complete",
        "n_overlap": int(len(files)),
        "n_classes": int(len(labels)),
        "truth_positive_cells": int(y.sum()),
        "truth_rows_with_positive": int((y.sum(axis=1) > 0).sum()),
        "items": [{"name": n, "path": str(p), "weight": float(w)} for n, p, w in specs],
        "weights_normalized": {name: float(w) for (name, _, _), w in zip(specs, weights)},
        "standalone": standalone,
        "correlation_matrix": corr,
        "ensemble": {
            "auc": macro_auc(y, ensemble),
            "topk_recall": topk_recall(y, ensemble),
            "thresholds": threshold_stats(y, ensemble, tuple(args.threshold)),
            "prob_stats": {
                "min": float(ensemble.min()),
                "max": float(ensemble.max()),
                "mean": float(ensemble.mean()),
                "p95": float(np.quantile(ensemble, 0.95)),
                "p99": float(np.quantile(ensemble, 0.99)),
            },
        },
        "blend_grid": blend_grid,
        "best_blend_grid": max(blend_grid, key=lambda r: r.get("macro_auc") or -1) if blend_grid else None,
    }
    text = json.dumps(result, indent=2)
    print(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
