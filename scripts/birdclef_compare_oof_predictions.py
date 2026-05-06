#!/usr/bin/env python3
"""Compare two OOF prediction artifacts and a blend grid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def load(path: Path):
    z = np.load(path, allow_pickle=True)
    return {
        "path": str(path),
        "files": z["files"].astype(str),
        "y": z["y_oof"].astype(np.float32),
        "pred": z["pred_oof"].astype(np.float32),
        "labels": z["labels"].astype(str),
    }


def macro_auc(y, pred):
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


def align(a, b):
    b_pos = {f: i for i, f in enumerate(b["files"])}
    pairs = [(i, b_pos[f]) for i, f in enumerate(a["files"]) if f in b_pos]
    if not pairs:
        raise RuntimeError("No overlapping files")
    ia = np.array([x for x, _ in pairs], dtype=int)
    ib = np.array([y for _, y in pairs], dtype=int)
    if not np.array_equal(a["y"][ia], b["y"][ib]):
        raise RuntimeError("Aligned labels differ")
    return a["y"][ia], a["pred"][ia], b["pred"][ib], a["files"][ia]


def corr(x, y):
    xf, yf = x.reshape(-1), y.reshape(-1)
    return float(np.corrcoef(xf, yf)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--name-a", default="a")
    ap.add_argument("--name-b", default="b")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    a, b = load(args.a), load(args.b)
    y, pa, pb, files = align(a, b)
    blend_grid = []
    for wb in [i / 10 for i in range(11)]:
        blend_grid.append({"weight_b": wb, **macro_auc(y, (1 - wb) * pa + wb * pb)})
    best = max(blend_grid, key=lambda r: r["macro_auc"] if r["macro_auc"] is not None else -1)
    result = {
        "name_a": args.name_a,
        "name_b": args.name_b,
        "n_overlap": int(len(files)),
        "n_classes": int(y.shape[1]),
        "a_auc": macro_auc(y, pa),
        "b_auc": macro_auc(y, pb),
        "flat_pearson": corr(pa, pb),
        "mean_abs_diff": float(np.mean(np.abs(pa - pb))),
        "blend_grid": blend_grid,
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
