#!/usr/bin/env python3
"""Crossfit a bounded rare/non-bird taxon specialist on teacher-cache rows.

This is a diagnostic, not a submission generator.  It tests whether a learned
multi-output taxon-presence calibration can improve a strong public946 teacher
without relying on in-sample site/hour leakage or blunt hand-written gates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def pick_key(z: np.lib.npyio.NpzFile, preferred: str | None, candidates: list[str]) -> str:
    if preferred:
        if preferred not in z.files:
            raise KeyError(f"requested key {preferred!r} not present; available={z.files}")
        return preferred
    for key in candidates:
        if key in z.files:
            return key
    raise KeyError(f"none of {candidates!r} present; available={z.files}")


def macro_auc(y_true: np.ndarray, y_pred: np.ndarray, idx: list[int] | None = None) -> dict[str, Any]:
    if idx is not None:
        y_true = y_true[:, idx]
        y_pred = y_pred[:, idx]
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


def align_pred_truth(pred_npz: Path, truth_npz: Path, pred_key: str | None, truth_key: str | None) -> dict[str, Any]:
    pred_z = np.load(pred_npz, allow_pickle=True)
    truth_z = np.load(truth_npz, allow_pickle=True)
    pk = pick_key(pred_z, pred_key, ["probs", "pred_teacher", "pred_student", "pred_oof"])
    tk = pick_key(truth_z, truth_key, ["y_true", "y_oof", "target", "targets"])
    labels = pred_z["labels"].astype(str)
    truth_labels = truth_z["labels"].astype(str)
    if list(labels) != list(truth_labels):
        raise ValueError("label mismatch between prediction and truth artifacts")
    pred_rows = pred_z["row_ids"].astype(str) if "row_ids" in pred_z.files else pred_z["files"].astype(str)
    truth_rows = truth_z["row_ids"].astype(str) if "row_ids" in truth_z.files else truth_z["files"].astype(str)
    truth_index = {row: i for i, row in enumerate(truth_rows)}
    pred_idx = [i for i, row in enumerate(pred_rows) if row in truth_index]
    truth_idx = [truth_index[pred_rows[i]] for i in pred_idx]
    if not pred_idx:
        raise ValueError("no overlapping rows between prediction and truth artifacts")
    return {
        "pred_key": pk,
        "truth_key": tk,
        "row_ids": pred_rows[pred_idx],
        "labels": labels,
        "pred": pred_z[pk][pred_idx].astype(np.float32),
        "truth": truth_z[tk][truth_idx].astype(np.float32),
    }


def build_group_indices(labels: np.ndarray, taxonomy_csv: Path) -> dict[str, list[int]]:
    tax = pd.read_csv(taxonomy_csv, dtype={"primary_label": str})
    label_to_group = dict(zip(tax["primary_label"].astype(str), tax["class_name"].astype(str)))
    groups: dict[str, list[int]] = {}
    for i, label in enumerate(labels.astype(str)):
        groups.setdefault(label_to_group.get(str(label), "UNKNOWN"), []).append(i)
    return {k: v for k, v in sorted(groups.items())}


def topk_mean(values: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(int(k), values.shape[1]))
    part = np.partition(values, values.shape[1] - k, axis=1)[:, -k:]
    return part.mean(axis=1)


def file_group(row_ids: np.ndarray) -> np.ndarray:
    return np.array(["_".join(str(r).split("_")[:-1]) for r in row_ids])


def build_features(pred: np.ndarray, group_idx: list[int]) -> np.ndarray:
    sub = pred[:, group_idx]
    all_top5 = topk_mean(pred, 5)
    feats = np.column_stack([
        sub.max(axis=1),
        sub.mean(axis=1),
        topk_mean(sub, min(3, sub.shape[1])),
        sub.std(axis=1),
        pred.max(axis=1),
        pred.mean(axis=1),
        all_top5,
        np.maximum(sub.max(axis=1) - all_top5, 0.0),
    ]).astype(np.float32)
    return feats


def make_cv(y: np.ndarray, groups: np.ndarray, max_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.unique(groups)
    n_splits = min(int(max_folds), len(unique_groups))
    if n_splits >= 2:
        folds = list(GroupKFold(n_splits=n_splits).split(np.zeros_like(y), y, groups))
        usable = []
        for train_idx, val_idx in folds:
            if len(np.unique(y[train_idx])) == 2:
                usable.append((train_idx, val_idx))
        if usable:
            return usable
    n_splits = min(int(max_folds), int(np.bincount(y.astype(int)).min())) if len(np.unique(y)) == 2 else 0
    if n_splits >= 2:
        return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=20260516).split(np.zeros_like(y), y))
    return []


def crossfit_group_model(x: np.ndarray, y: np.ndarray, groups: np.ndarray, max_folds: int) -> dict[str, Any]:
    evidence = x[:, 0].astype(np.float32)
    if len(np.unique(y)) < 2 or int(y.sum()) < 2:
        return {
            "oof_prob": evidence,
            "status": "skipped_single_class_or_too_few_positive",
            "presence_auc": None,
            "n_positive": int(y.sum()),
        }
    oof = np.full(len(y), np.nan, dtype=np.float32)
    folds = make_cv(y, groups, max_folds)
    if not folds:
        return {
            "oof_prob": evidence,
            "status": "skipped_no_usable_cv",
            "presence_auc": None,
            "n_positive": int(y.sum()),
        }
    for train_idx, val_idx in folds:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5, solver="lbfgs"),
        )
        clf.fit(x[train_idx], y[train_idx])
        oof[val_idx] = clf.predict_proba(x[val_idx])[:, 1].astype(np.float32)
    missing = np.isnan(oof)
    if missing.any():
        oof[missing] = evidence[missing]
    auc = None
    try:
        auc = float(roc_auc_score(y, oof))
    except ValueError:
        pass
    return {"oof_prob": oof, "status": "complete", "presence_auc": auc, "n_positive": int(y.sum()), "n_folds": len(folds)}


def apply_specialist(
    pred: np.ndarray,
    groups: dict[str, list[int]],
    group_probs: dict[str, np.ndarray],
    target_groups: list[str],
    alpha: float,
    min_mult: float,
    max_mult: float,
    eps: float = 1e-4,
) -> np.ndarray:
    out = pred.copy().astype(np.float64)
    for group in target_groups:
        idx = groups[group]
        evidence = pred[:, idx].max(axis=1)
        ratio = (group_probs[group] + eps) / (evidence + eps)
        mult = np.clip(np.power(ratio, float(alpha)), float(min_mult), float(max_mult))
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
    ap.add_argument("--target-groups", default="Amphibia,Insecta,Mammalia,Reptilia")
    ap.add_argument("--alphas", default="0,0.125,0.25,0.375,0.5,0.75,1.0")
    ap.add_argument("--min-mults", default="0.50,0.67,0.80")
    ap.add_argument("--max-mults", default="1.25,1.50,2.00")
    ap.add_argument("--max-folds", type=int, default=5)
    args = ap.parse_args()

    data = align_pred_truth(args.pred_npz, args.truth_npz, args.pred_key, args.truth_key)
    pred = data["pred"]
    y_true = data["truth"]
    labels = data["labels"]
    row_ids = data["row_ids"]
    groups = build_group_indices(labels, args.taxonomy)
    target_groups = [g.strip() for g in args.target_groups.split(",") if g.strip() and g.strip() in groups]
    row_groups = file_group(row_ids)

    group_probs: dict[str, np.ndarray] = {}
    group_info: dict[str, Any] = {}
    for group in target_groups:
        idx = groups[group]
        y_group = (y_true[:, idx].max(axis=1) > 0).astype(int)
        x = build_features(pred, idx)
        fit = crossfit_group_model(x, y_group, row_groups, args.max_folds)
        group_probs[group] = fit.pop("oof_prob")
        evidence_auc = None
        if len(np.unique(y_group)) == 2:
            try:
                evidence_auc = float(roc_auc_score(y_group, pred[:, idx].max(axis=1)))
            except ValueError:
                pass
        group_info[group] = {
            **fit,
            "n_labels": len(idx),
            "evidence_presence_auc": evidence_auc,
        }

    baseline = macro_auc(y_true, pred)
    target_idx = [i for g in target_groups for i in groups[g]]
    baseline_target = macro_auc(y_true, pred, target_idx)
    rows = []
    for alpha in [float(x) for x in args.alphas.split(",") if x.strip()]:
        for min_mult in [float(x) for x in args.min_mults.split(",") if x.strip()]:
            for max_mult in [float(x) for x in args.max_mults.split(",") if x.strip()]:
                adjusted = apply_specialist(pred, groups, group_probs, target_groups, alpha, min_mult, max_mult)
                auc = macro_auc(y_true, adjusted)
                target_auc = macro_auc(y_true, adjusted, target_idx)
                rows.append({
                    "alpha": alpha,
                    "min_mult": min_mult,
                    "max_mult": max_mult,
                    "macro_auc": auc["macro_auc"],
                    "valid_classes": auc["valid_classes"],
                    "target_macro_auc": target_auc["macro_auc"],
                    "target_valid_classes": target_auc["valid_classes"],
                    "delta_vs_baseline": None if auc["macro_auc"] is None else float(auc["macro_auc"] - baseline["macro_auc"]),
                    "target_delta_vs_baseline": None if target_auc["macro_auc"] is None else float(target_auc["macro_auc"] - baseline_target["macro_auc"]),
                    "mae_vs_baseline": float(np.mean(np.abs(adjusted - pred))),
                    "max_abs_vs_baseline": float(np.max(np.abs(adjusted - pred))),
                    "corr_vs_baseline": float(np.corrcoef(pred.ravel(), adjusted.ravel())[0, 1]),
                })
    rows_sorted = sorted(rows, key=lambda r: (-1e9 if r["macro_auc"] is None else r["macro_auc"]), reverse=True)
    summary = {
        "status": "complete",
        "pred_npz": str(args.pred_npz),
        "truth_npz": str(args.truth_npz),
        "pred_key": data["pred_key"],
        "truth_key": data["truth_key"],
        "taxonomy": str(args.taxonomy),
        "n_rows": int(len(row_ids)),
        "n_classes": int(len(labels)),
        "target_groups": target_groups,
        "groups": {g: len(groups[g]) for g in sorted(groups)},
        "baseline_auc": baseline,
        "baseline_target_auc": baseline_target,
        "group_models": group_info,
        "best": rows_sorted[0] if rows_sorted else None,
        "sweep": rows_sorted,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in ["status", "baseline_auc", "baseline_target_auc", "group_models", "best"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
