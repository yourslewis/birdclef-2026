#!/usr/bin/env python3
"""Sweep the exact Kaggle taxon-max gate on labeled teacher-cache rows.

This complements ``birdclef_taxon_gate_oof_sweep.py``.  The public-winning
v516 gate uses a different multiplier than the older conservative OOF script:

    multiplier = max(floor, group_evidence) ** alpha

where group evidence is the row-wise max (or optional top-k mean) prediction in
a taxonomy class group.  This script loads cached v508-style predictions plus a
matching labeled-soundscape truth artifact and reports which floor/alpha pairs
look best on those labeled rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    aucs: list[float] = []
    valid = 0
    for j in range(y_true.shape[1]):
        col = y_true[:, j]
        if float(col.min()) == float(col.max()):
            continue
        try:
            aucs.append(float(roc_auc_score(col, y_pred[:, j])))
            valid += 1
        except ValueError:
            continue
    return {"macro_auc": float(np.mean(aucs)) if aucs else None, "valid_classes": valid}


def pick_key(z: np.lib.npyio.NpzFile, preferred: str | None, candidates: list[str]) -> str:
    if preferred:
        if preferred not in z.files:
            raise KeyError(f"requested key {preferred!r} not present; available={z.files}")
        return preferred
    for key in candidates:
        if key in z.files:
            return key
    raise KeyError(f"none of {candidates!r} present; available={z.files}")


def load_pred_and_truth(pred_npz: Path, truth_npz: Path, pred_key: str | None, truth_key: str | None) -> dict[str, Any]:
    pred_z = np.load(pred_npz, allow_pickle=True)
    truth_z = np.load(truth_npz, allow_pickle=True)
    pk = pick_key(pred_z, pred_key, ["probs", "pred_teacher", "pred_oof", "pred_student"])
    tk = pick_key(truth_z, truth_key, ["y_true", "y_oof", "target", "targets"])
    labels = pred_z["labels"].astype(str)
    truth_labels = truth_z["labels"].astype(str)
    if list(labels) != list(truth_labels):
        raise ValueError("label mismatch between prediction and truth artifacts")
    pred_rows = pred_z["row_ids"].astype(str) if "row_ids" in pred_z.files else pred_z["files"].astype(str)
    truth_rows = truth_z["row_ids"].astype(str) if "row_ids" in truth_z.files else truth_z["files"].astype(str)
    truth_index = {r: i for i, r in enumerate(truth_rows)}
    common_rows = [r for r in pred_rows if r in truth_index]
    if not common_rows:
        raise ValueError("no overlapping rows between prediction and truth artifacts")
    pred_idx = [i for i, r in enumerate(pred_rows) if r in truth_index]
    truth_idx = [truth_index[r] for r in common_rows]
    return {
        "pred_key": pk,
        "truth_key": tk,
        "rows": np.asarray(common_rows),
        "labels": labels,
        "pred": pred_z[pk][pred_idx].astype(np.float32),
        "y": truth_z[tk][truth_idx].astype(np.float32),
    }


def build_group_indices(labels: np.ndarray, taxonomy_path: Path) -> dict[str, list[int]]:
    taxonomy = pd.read_csv(taxonomy_path, dtype={"primary_label": str})
    label_to_group = dict(zip(taxonomy["primary_label"].astype(str), taxonomy["class_name"].astype(str)))
    groups: dict[str, list[int]] = {}
    for i, label in enumerate(labels.astype(str)):
        groups.setdefault(label_to_group.get(label, "UNKNOWN"), []).append(i)
    return {g: idx for g, idx in sorted(groups.items())}


def group_evidence(pred: np.ndarray, idx: list[int], mode: str, topk: int) -> np.ndarray:
    sub = pred[:, idx]
    if mode == "max":
        return sub.max(axis=1)
    if mode == "topk_mean":
        k = min(topk, sub.shape[1])
        part = np.partition(sub, sub.shape[1] - k, axis=1)[:, -k:]
        return part.mean(axis=1)
    if mode == "mean":
        return sub.mean(axis=1)
    raise ValueError(f"unknown mode {mode!r}")


def apply_kernel_gate(pred: np.ndarray, groups: dict[str, list[int]], floor: float, alpha: float, mode: str, topk: int, exempt_singletons: bool) -> np.ndarray:
    out = pred.copy().astype(np.float64)
    for group, idx in groups.items():
        if group == "UNKNOWN":
            continue
        if exempt_singletons and len(idx) <= 1:
            continue
        evidence = group_evidence(pred, idx, mode=mode, topk=topk)
        mult = np.maximum(float(floor), evidence) ** float(alpha)
        out[:, idx] *= mult[:, None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-npz", type=Path, required=True)
    ap.add_argument("--truth-npz", type=Path, required=True)
    ap.add_argument("--taxonomy", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--pred-key")
    ap.add_argument("--truth-key")
    ap.add_argument("--floors", type=float, nargs="+", default=[0.10, 0.20, 0.30, 0.40, 0.50])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.50, 0.625, 0.75, 0.875, 1.0])
    ap.add_argument("--modes", nargs="+", default=["max", "topk_mean"])
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--include-singletons", action="store_true")
    args = ap.parse_args()

    data = load_pred_and_truth(args.pred_npz, args.truth_npz, args.pred_key, args.truth_key)
    groups = build_group_indices(data["labels"], args.taxonomy)
    y = data["y"]
    pred = data["pred"]
    baseline = macro_auc(y, pred)
    rows = []
    for mode in args.modes:
        for floor in args.floors:
            for alpha in args.alphas:
                gated = apply_kernel_gate(
                    pred,
                    groups,
                    floor=floor,
                    alpha=alpha,
                    mode=mode,
                    topk=args.topk,
                    exempt_singletons=not args.include_singletons,
                )
                auc = macro_auc(y, gated)
                delta = None if auc["macro_auc"] is None or baseline["macro_auc"] is None else auc["macro_auc"] - baseline["macro_auc"]
                rows.append({
                    "mode": mode,
                    "floor": floor,
                    "alpha": alpha,
                    "macro_auc": auc["macro_auc"],
                    "valid_classes": auc["valid_classes"],
                    "delta_vs_baseline": delta,
                })
    rows_sorted = sorted(rows, key=lambda r: (-1e9 if r["macro_auc"] is None else r["macro_auc"]), reverse=True)
    summary = {
        "status": "complete",
        "pred_npz": str(args.pred_npz),
        "truth_npz": str(args.truth_npz),
        "pred_key": data["pred_key"],
        "truth_key": data["truth_key"],
        "n_rows": int(len(data["rows"])),
        "n_classes": int(len(data["labels"])),
        "taxonomy": str(args.taxonomy),
        "groups": {g: len(idx) for g, idx in groups.items()},
        "baseline_auc": baseline,
        "best": rows_sorted[0] if rows_sorted else None,
        "queued_variants": {
            "v517": {"mode": "max", "floor": 0.30, "alpha": 0.50},
            "v516": {"mode": "max", "floor": 0.30, "alpha": 0.75},
            "v523": {"mode": "max", "floor": 0.30, "alpha": 0.875},
            "v524": {"mode": "max", "floor": 0.20, "alpha": 0.75},
            "v525": {"mode": "max", "floor": 0.40, "alpha": 0.75},
        },
        "sweep": rows_sorted,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
