#!/usr/bin/env python3
"""Blend compatible BirdCLEF teacher NPZ probability caches.

Inputs are NPZ files with row_ids, labels, and probs arrays. The script validates
row/label alignment, writes a compatible teacher NPZ, and optionally evaluates
against train_soundscapes_labels.csv for a lightweight gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from birdclef_pseudolabel_cache_summary import build_truth, topk_recall  # noqa: E402


def parse_input(spec: str) -> tuple[str, Path, float]:
    """Parse name=path:weight while allowing ':' inside absolute paths poorly?"""
    if "=" not in spec:
        raise ValueError(f"Input must be name=path:weight, got {spec!r}")
    name, rest = spec.split("=", 1)
    if ":" not in rest:
        raise ValueError(f"Input must be name=path:weight, got {spec!r}")
    path_text, weight_text = rest.rsplit(":", 1)
    return name, Path(path_text), float(weight_text)


def load_teacher(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    z = np.load(path, allow_pickle=True)
    for key in ["row_ids", "labels", "probs"]:
        if key not in z.files:
            raise KeyError(f"{path} missing {key}; keys={z.files}")
    row_ids = z["row_ids"].astype(str)
    labels = [str(x) for x in z["labels"].astype(str).tolist()]
    probs = z["probs"].astype(np.float32)
    if probs.shape != (len(row_ids), len(labels)):
        raise ValueError(f"Shape mismatch in {path}: probs={probs.shape}, rows={len(row_ids)}, labels={len(labels)}")
    return row_ids, labels, probs


def macro_auc(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    vals = []
    for j in range(y.shape[1]):
        col = y[:, j]
        if col.min() == col.max():
            continue
        vals.append(float(roc_auc_score(col, p[:, j])))
    return {"macro_auc": float(np.mean(vals)) if vals else None, "valid_classes": len(vals)}


def flat_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True, help="name=path:weight; repeatable")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--labels-csv", type=Path)
    ap.add_argument("--normalize", action="store_true", help="Normalize weights to sum to 1")
    args = ap.parse_args()

    specs = [parse_input(x) for x in args.input]
    if not specs:
        raise ValueError("No inputs")
    weights = np.asarray([w for _, _, w in specs], dtype=np.float64)
    if args.normalize:
        total = float(weights.sum())
        if total <= 0:
            raise ValueError(f"Cannot normalize non-positive weights: {weights.tolist()}")
        weights = weights / total

    base_rows: np.ndarray | None = None
    base_labels: list[str] | None = None
    blend: np.ndarray | None = None
    inputs_summary: list[dict[str, Any]] = []
    source_probs: dict[str, np.ndarray] = {}

    for (name, path, _), weight in zip(specs, weights):
        row_ids, labels, probs = load_teacher(path)
        if base_rows is None:
            base_rows = row_ids
            base_labels = labels
            blend = np.zeros_like(probs, dtype=np.float64)
        else:
            if not np.array_equal(base_rows, row_ids):
                raise ValueError(f"row_ids mismatch for {name}: {path}")
            if base_labels != labels:
                raise ValueError(f"labels mismatch for {name}: {path}")
        blend += float(weight) * probs
        source_probs[name] = probs
        inputs_summary.append({
            "name": name,
            "path": str(path),
            "weight": float(weight),
            "prob_stats": {
                "min": float(probs.min()),
                "max": float(probs.max()),
                "mean": float(probs.mean()),
                "p99": float(np.quantile(probs, 0.99)),
            },
        })

    assert base_rows is not None and base_labels is not None and blend is not None
    blended = np.clip(blend, 0.0, 1.0).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, row_ids=base_rows, labels=np.asarray(base_labels, dtype=object), probs=blended)

    summary: dict[str, Any] = {
        "status": "teacher_blend_complete",
        "output": str(args.output),
        "n_rows": int(len(base_rows)),
        "n_classes": int(len(base_labels)),
        "inputs": inputs_summary,
        "blend_prob_stats": {
            "min": float(blended.min()),
            "max": float(blended.max()),
            "mean": float(blended.mean()),
            "p95": float(np.quantile(blended, 0.95)),
            "p99": float(np.quantile(blended, 0.99)),
        },
        "source_correlations": {
            f"blend_vs_{name}": flat_corr(blended, probs) for name, probs in source_probs.items()
        },
    }

    if args.labels_csv and args.labels_csv.exists():
        y = build_truth(pd.read_csv(args.labels_csv), base_rows, base_labels)
        summary["auc_summary"] = macro_auc(y, blended)
        summary["topk_recall"] = {str(k): topk_recall(y, blended, k) for k in [1, 3, 5, 10]}
        summary["input_auc_summary"] = {name: macro_auc(y, probs) for name, probs in source_probs.items()}

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
