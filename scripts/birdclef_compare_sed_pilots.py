#!/usr/bin/env python3
"""Compare BirdCLEF SED pilot holdout prediction artifacts.

Aligns validation files between two pilot `holdout_predictions.npz` files,
reports per-model and blend macro AUC, and measures prediction correlation.
This is intentionally small so it can run on the GPU server or Mac.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def load_npz(path: Path):
    z = np.load(path, allow_pickle=True)
    files = z["files"].astype(str)
    val_idx = z["val_indices"].astype(int)
    val_files = files[val_idx]
    return {
        "path": str(path),
        "val_files": val_files,
        "y": z["y_val"].astype(np.float32),
        "pred": z["pred_val"].astype(np.float32),
        "labels": z["labels"].astype(str),
    }


def macro_auc(y: np.ndarray, pred: np.ndarray) -> dict:
    aucs = []
    for j in range(y.shape[1]):
        col = y[:, j]
        if col.min() == col.max():
            continue
        try:
            aucs.append(float(roc_auc_score(col, pred[:, j])))
        except Exception:
            pass
    return {"macro_auc": float(np.mean(aucs)) if aucs else None, "valid_classes": len(aucs)}


def align(a: dict, b: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    b_pos = {f: i for i, f in enumerate(b["val_files"])}
    pairs = [(i, b_pos[f]) for i, f in enumerate(a["val_files"]) if f in b_pos]
    if not pairs:
        raise RuntimeError("No overlapping validation files")
    ia = np.array([p[0] for p in pairs], dtype=int)
    ib = np.array([p[1] for p in pairs], dtype=int)
    ya = a["y"][ia]
    yb = b["y"][ib]
    if not np.array_equal(ya, yb):
        raise RuntimeError("Aligned labels differ")
    return ya, a["pred"][ia], b["pred"][ib], a["val_files"][ia]


def pearson_flat(x: np.ndarray, y: np.ndarray) -> float:
    xf = x.reshape(-1)
    yf = y.reshape(-1)
    if float(np.std(xf)) == 0.0 or float(np.std(yf)) == 0.0:
        return float("nan")
    return float(np.corrcoef(xf, yf)[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--name-a", default="a")
    parser.add_argument("--name-b", default="b")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    a = load_npz(args.a)
    b = load_npz(args.b)
    y, pa, pb, files = align(a, b)
    blends = []
    for wb in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        pred = (1.0 - wb) * pa + wb * pb
        blends.append({"weight_b": wb, **macro_auc(y, pred)})
    best = max(blends, key=lambda row: row["macro_auc"] if row["macro_auc"] is not None else -1)
    result = {
        "name_a": args.name_a,
        "name_b": args.name_b,
        "path_a": str(args.a),
        "path_b": str(args.b),
        "n_overlap": int(len(files)),
        "n_classes": int(y.shape[1]),
        "a_auc": macro_auc(y, pa),
        "b_auc": macro_auc(y, pb),
        "flat_pearson": pearson_flat(pa, pb),
        "mean_abs_diff": float(np.mean(np.abs(pa - pb))),
        "blend_grid": blends,
        "best_blend": best,
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
