#!/usr/bin/env python3
"""Build a clean OOF ensemble teacher cache from one or more OOF artifacts.

Unlike in-sample teacher dry-runs, this cache is built only from out-of-fold
predictions. It supports union alignment: each file uses the configured weighted
average of whatever members have a prediction for that file, with weights
renormalized per row. This preserves coverage from all OOF artifacts while still
recording availability so downstream training can filter to high-coverage rows.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sklearn.metrics import roc_auc_score as _sklearn_roc_auc_score
except Exception:  # pragma: no cover
    _sklearn_roc_auc_score = None


def roc_auc_score_1d(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if _sklearn_roc_auc_score is not None:
        return float(_sklearn_roc_auc_score(y_true, y_score))
    y = np.asarray(y_true, dtype=np.float64)
    scores = np.asarray(y_score, dtype=np.float64)
    n_pos = int(np.sum(y > 0.5))
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC-AUC requires both positive and negative samples")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks_sorted = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks_sorted[start:end] = (start + 1 + end) / 2.0
        start = end
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = ranks_sorted
    pos_rank_sum = float(np.sum(ranks[y > 0.5]))
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def path_key(raw: str) -> str:
    parts = str(raw).replace("\\", "/").split("/")
    if "train_audio" in parts:
        idx = parts.index("train_audio")
        return "/".join(parts[idx + 1 : idx + 3])
    return "/".join(parts[-2:])


def parse_member(raw: str) -> tuple[str, Path, float]:
    parts = raw.split(":")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError("member must be name:path[:weight]")
    name, path_s = parts[0], parts[1]
    if not name:
        raise argparse.ArgumentTypeError("member name must be non-empty")
    path = Path(path_s)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"member path does not exist: {path}")
    weight = float(parts[2]) if len(parts) == 3 else 1.0
    if weight < 0:
        raise argparse.ArgumentTypeError("member weight must be non-negative")
    return name, path, weight


def load_member(name: str, path: Path, weight: float) -> dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    required = {"files", "labels"}
    missing = sorted(required - set(z.files))
    if missing:
        raise ValueError(f"{path} missing required keys: {missing}")
    if "y_oof" in z.files and "pred_oof" in z.files:
        y = z["y_oof"].astype(np.float32)
        pred = z["pred_oof"].astype(np.float32)
    elif "y_val" in z.files and "pred_val" in z.files:
        y = z["y_val"].astype(np.float32)
        pred = z["pred_val"].astype(np.float32)
    else:
        raise ValueError(f"{path} must contain y_oof/pred_oof or y_val/pred_val")
    files = np.array([path_key(x) for x in z["files"].astype(str)])
    if len(files) != len(pred):
        if "val_indices" in z.files and len(z["val_indices"]) == len(pred):
            files = files[z["val_indices"].astype(int)]
        else:
            raise ValueError(f"{path}: files length {len(files)} does not match predictions {len(pred)}")
    return {
        "name": name,
        "path": str(path),
        "weight": float(weight),
        "files": files,
        "labels": z["labels"].astype(str),
        "y": y,
        "pred": pred,
    }


def macro_auc(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    aucs: list[float] = []
    per_class: list[dict[str, Any]] = []
    for j in range(y.shape[1]):
        col = y[:, j]
        if col.min() == col.max():
            continue
        try:
            value = roc_auc_score_1d(col, pred[:, j])
        except Exception:
            continue
        aucs.append(float(value))
        per_class.append({"class_index": int(j), "auc": float(value), "positives": int(col.sum())})
    return {
        "macro_auc": float(np.mean(aucs)) if aucs else None,
        "valid_classes": len(aucs),
        "per_class": per_class,
    }


def topk_recall(y: np.ndarray, pred: np.ndarray, ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict[str, float]:
    positives = float(y.sum())
    if positives <= 0:
        return {str(k): 0.0 for k in ks}
    out: dict[str, float] = {}
    for k in ks:
        kk = min(k, pred.shape[1])
        idx = np.argpartition(pred, -kk, axis=1)[:, -kk:]
        mask = np.zeros_like(pred, dtype=bool)
        rows = np.arange(pred.shape[0])[:, None]
        mask[rows, idx] = True
        out[str(k)] = float((y.astype(bool) & mask).sum() / positives)
    return out


def threshold_stats(y: np.ndarray, pred: np.ndarray, thresholds: list[float]) -> dict[str, Any]:
    yb = y.astype(bool)
    total_true = int(yb.sum())
    out: dict[str, Any] = {}
    for t in thresholds:
        pos = pred >= t
        n = int(pos.sum())
        tp = int((pos & yb).sum())
        fp = int((pos & ~yb).sum())
        out[f">={t:g}"] = {
            "cells": n,
            "rows": int(pos.any(axis=1).sum()),
            "classes": int(pos.any(axis=0).sum()),
            "true_positive_cells": tp,
            "false_positive_cells": fp,
            "precision": float(tp / n) if n else None,
            "recall": float(tp / total_true) if total_true else None,
        }
    for t in (0.01, 0.02, 0.05, 0.10):
        neg = pred <= t
        n = int(neg.sum())
        tn = int((neg & ~yb).sum())
        fn = int((neg & yb).sum())
        out[f"<={t:g}"] = {
            "cells": n,
            "rows": int(neg.any(axis=1).sum()),
            "classes": int(neg.any(axis=0).sum()),
            "true_negative_cells": tn,
            "false_negative_cells": fn,
            "negative_precision": float(tn / n) if n else None,
        }
    return out


def flat_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--member", action="append", required=True, type=parse_member, help="name:path[:weight]; repeat")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("union", "intersection"), default="union")
    p.add_argument("--threshold", type=float, action="append", default=[0.5, 0.7, 0.9, 0.95, 0.98])
    p.add_argument("--min-available", type=int, default=1, help="minimum available members to keep in output")
    args = p.parse_args()

    if len(args.member) < 1:
        raise SystemExit("Need at least one --member")
    members = [load_member(name, path, weight) for name, path, weight in args.member]
    labels0 = members[0]["labels"]
    for member in members[1:]:
        if not np.array_equal(labels0, member["labels"]):
            raise ValueError(f"label mismatch between {members[0]['name']} and {member['name']}")

    file_sets = [set(member["files"].tolist()) for member in members]
    if args.mode == "intersection":
        selected = set.intersection(*file_sets)
    else:
        selected = set.union(*file_sets)
    if not selected:
        raise RuntimeError("No files selected")
    files_all = np.array(sorted(selected), dtype=str)

    n_files, n_classes, n_members = len(files_all), len(labels0), len(members)
    pred_sum = np.zeros((n_files, n_classes), dtype=np.float64)
    weight_sum = np.zeros((n_files,), dtype=np.float64)
    available = np.zeros((n_files, n_members), dtype=bool)
    y_out = np.full((n_files, n_classes), np.nan, dtype=np.float32)
    file_to_row = {f: i for i, f in enumerate(files_all)}
    truth_conflicts = 0

    for m_idx, member in enumerate(members):
        pos = {f: i for i, f in enumerate(member["files"])}
        for f, src_i in pos.items():
            if f not in file_to_row:
                continue
            dst_i = file_to_row[f]
            y = member["y"][src_i]
            if np.isnan(y_out[dst_i]).all():
                y_out[dst_i] = y
            elif not np.array_equal(y_out[dst_i], y):
                truth_conflicts += 1
            w = float(member["weight"])
            if w > 0:
                pred_sum[dst_i] += w * member["pred"][src_i]
                weight_sum[dst_i] += w
            available[dst_i, m_idx] = True

    avail_count = available.sum(axis=1)
    keep = (avail_count >= int(args.min_available)) & (weight_sum > 0) & ~np.isnan(y_out).any(axis=1)
    files = files_all[keep]
    y_true = y_out[keep].astype(np.float32)
    teacher = (pred_sum[keep] / weight_sum[keep, None]).astype(np.float32)
    available_kept = available[keep]
    avail_count_kept = avail_count[keep]
    weight_sum_kept = weight_sum[keep].astype(np.float32)

    singles = []
    for member in members:
        singles.append({
            "name": member["name"],
            "path": member["path"],
            "weight": member["weight"],
            "n_files": int(len(member["files"])),
            **{k: v for k, v in macro_auc(member["y"], member["pred"]).items() if k != "per_class"},
        })

    pairwise = []
    for i, j in itertools.combinations(range(n_members), 2):
        a, b = members[i], members[j]
        common = sorted(set(a["files"].tolist()) & set(b["files"].tolist()))
        if not common:
            continue
        apos = {f: k for k, f in enumerate(a["files"])}
        bpos = {f: k for k, f in enumerate(b["files"])}
        ia = np.array([apos[f] for f in common], dtype=int)
        ib = np.array([bpos[f] for f in common], dtype=int)
        pairwise.append({
            "a": a["name"],
            "b": b["name"],
            "n_overlap": int(len(common)),
            "flat_pearson": flat_corr(a["pred"][ia], b["pred"][ib]),
            "mean_abs_diff": float(np.mean(np.abs(a["pred"][ia] - b["pred"][ib]))),
        })

    auc = macro_auc(y_true, teacher)
    per_class = auc.pop("per_class")
    summary = {
        "status": "oof_teacher_cache_complete",
        "output_npz": str(args.output),
        "mode": args.mode,
        "min_available": int(args.min_available),
        "n_files": int(len(files)),
        "n_classes": int(n_classes),
        "truth_conflicts": int(truth_conflicts),
        "truth_positive_cells": int(y_true.sum()),
        "members": [{"name": m["name"], "path": m["path"], "weight": m["weight"]} for m in members],
        "single_auc": singles,
        "pairwise": pairwise,
        "availability_histogram": {str(k): int((avail_count_kept == k).sum()) for k in range(1, n_members + 1)},
        "teacher_auc": auc,
        "topk_recall": topk_recall(y_true, teacher),
        "thresholds": threshold_stats(y_true, teacher, args.threshold),
        "prob_stats": {
            "min": float(teacher.min()),
            "max": float(teacher.max()),
            "mean": float(teacher.mean()),
            "p95": float(np.quantile(teacher, 0.95)),
            "p99": float(np.quantile(teacher, 0.99)),
        },
        "per_class_auc": per_class,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        files=files,
        labels=labels0,
        y_true=y_true,
        teacher_pred=teacher,
        available_mask=available_kept.astype(np.bool_),
        available_count=avail_count_kept.astype(np.int16),
        available_weight_sum=weight_sum_kept,
        member_names=np.array([m["name"] for m in members], dtype=str),
        member_weights=np.array([m["weight"] for m in members], dtype=np.float32),
    )
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
