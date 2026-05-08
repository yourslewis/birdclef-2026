#!/usr/bin/env python3
"""Sweep conservative taxonomy-gate postprocessing on OOF predictions.

This is a cheap Spec E simulation: no gate model is trained. It uses existing
OOF probabilities to estimate row-level taxon presence and applies conservative
per-taxon multipliers, then reports macro AUC / blend-safe candidates.
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


def load_npz(path: Path) -> dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    return {
        "path": str(path),
        "y": z["y_oof"].astype(np.float32),
        "pred": z["pred_oof"].astype(np.float32),
        "files": z["files"].astype(str),
        "labels": z["labels"].astype(str),
    }


def align_ensemble(paths: list[Path], weights: list[float] | None) -> dict[str, Any]:
    items = [load_npz(p) for p in paths]
    labels = items[0]["labels"]
    for it in items[1:]:
        if list(it["labels"]) != list(labels):
            raise ValueError(f"label mismatch: {it['path']}")
    if weights is None:
        weights_arr = np.ones(len(items), dtype=np.float64) / len(items)
    else:
        weights_arr = np.array(weights, dtype=np.float64)
        weights_arr = weights_arr / weights_arr.sum()
    common = set(items[0]["files"])
    for it in items[1:]:
        common &= set(it["files"])
    if not common:
        raise ValueError("no overlapping files across OOF artifacts")
    # Stable order follows the first artifact.
    common_order = [f for f in items[0]["files"] if f in common]
    pred = np.zeros((len(common_order), len(labels)), dtype=np.float64)
    y_ref: np.ndarray | None = None
    for w, it in zip(weights_arr, items):
        idx = {f: i for i, f in enumerate(it["files"])}
        order_idx = [idx[f] for f in common_order]
        pred += w * it["pred"][order_idx]
        y_cur = it["y"][order_idx]
        if y_ref is None:
            y_ref = y_cur.copy()
        elif not np.allclose(y_ref, y_cur):
            raise ValueError(f"truth mismatch on common files: {it['path']}")
    assert y_ref is not None
    return {
        "paths": [str(p) for p in paths],
        "weights": weights_arr.tolist(),
        "files": np.array(common_order),
        "labels": labels,
        "y": y_ref.astype(np.float32),
        "pred": pred.astype(np.float32),
    }


def build_group_indices(labels: np.ndarray, taxonomy_path: Path) -> dict[str, list[int]]:
    taxonomy = pd.read_csv(taxonomy_path, dtype={"primary_label": str})
    label_to_group = dict(zip(taxonomy["primary_label"].astype(str), taxonomy["class_name"].astype(str)))
    groups: dict[str, list[int]] = {}
    for i, label in enumerate(labels.astype(str)):
        group = label_to_group.get(label, "UNKNOWN")
        groups.setdefault(group, []).append(i)
    return {g: idx for g, idx in sorted(groups.items())}


def group_presence(pred: np.ndarray, idx: list[int], mode: str, topk: int) -> np.ndarray:
    sub = pred[:, idx]
    if mode == "max":
        return sub.max(axis=1)
    if mode == "mean":
        return sub.mean(axis=1)
    if mode == "topk_mean":
        k = min(topk, sub.shape[1])
        part = np.partition(sub, sub.shape[1] - k, axis=1)[:, -k:]
        return part.mean(axis=1)
    raise ValueError(f"unknown mode {mode}")


def apply_gate(pred: np.ndarray, groups: dict[str, list[int]], floor: float, power: float, mode: str, topk: int, exempt_singletons: bool) -> np.ndarray:
    out = pred.copy().astype(np.float64)
    for group, idx in groups.items():
        if group == "UNKNOWN":
            continue
        if exempt_singletons and len(idx) <= 1:
            continue
        presence = group_presence(pred, idx, mode=mode, topk=topk)
        multiplier = floor + (1.0 - floor) * np.power(np.clip(presence, 0.0, 1.0), power)
        out[:, idx] *= multiplier[:, None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", type=Path, nargs="+", required=True)
    ap.add_argument("--weights", type=float, nargs="*")
    ap.add_argument("--taxonomy", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--floors", type=float, nargs="+", default=[0.50, 0.70, 0.85, 0.95])
    ap.add_argument("--powers", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--modes", nargs="+", default=["max", "topk_mean"])
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--include-singletons", action="store_true")
    args = ap.parse_args()

    if args.weights and len(args.weights) != len(args.oof):
        raise ValueError("--weights length must match --oof length")

    ens = align_ensemble(args.oof, args.weights if args.weights else None)
    groups = build_group_indices(ens["labels"], args.taxonomy)
    y = ens["y"]
    pred = ens["pred"]
    baseline = macro_auc(y, pred)

    rows = []
    for mode in args.modes:
        for floor in args.floors:
            for power in args.powers:
                gated = apply_gate(
                    pred,
                    groups,
                    floor=floor,
                    power=power,
                    mode=mode,
                    topk=args.topk,
                    exempt_singletons=not args.include_singletons,
                )
                auc = macro_auc(y, gated)
                rows.append({
                    "mode": mode,
                    "floor": floor,
                    "power": power,
                    "macro_auc": auc["macro_auc"],
                    "valid_classes": auc["valid_classes"],
                    "delta_vs_baseline": None if auc["macro_auc"] is None or baseline["macro_auc"] is None else auc["macro_auc"] - baseline["macro_auc"],
                })

    rows_sorted = sorted(rows, key=lambda r: (-1e9 if r["macro_auc"] is None else r["macro_auc"]), reverse=True)
    summary = {
        "status": "complete",
        "inputs": ens["paths"],
        "weights": ens["weights"],
        "n_overlap": int(len(ens["files"])),
        "n_classes": int(len(ens["labels"])),
        "taxonomy": str(args.taxonomy),
        "groups": {g: len(idx) for g, idx in groups.items()},
        "baseline_auc": baseline,
        "best": rows_sorted[0] if rows_sorted else None,
        "sweep": rows_sorted,
        "recommendation": "continue" if rows_sorted and rows_sorted[0].get("delta_vs_baseline", 0) > 0.0005 else "reject_or_only_use_with_external_gate",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
