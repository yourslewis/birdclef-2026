#!/usr/bin/env python3
"""Audit a pool of BirdCLEF student prediction NPZs against a teacher cache.

Scans student_predictions.npz files, keeps only row/label-aligned artifacts,
computes standalone AUC/correlation, and sweeps small blend weights into the
teacher. This is for deciding whether an existing student artifact deserves
packaging before spending a Kaggle code-submission slot.
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
from birdclef_pseudolabel_cache_summary import build_truth  # noqa: E402


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in str(text).replace(" ", "").split(",") if x]


def load_teacher(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    z = np.load(path, allow_pickle=True)
    row_ids = z["row_ids"].astype(str)
    labels = [str(x) for x in z["labels"].astype(str).tolist()]
    probs = z["probs"].astype(np.float32)
    if probs.shape != (len(row_ids), len(labels)):
        raise ValueError(f"Bad teacher shape: probs={probs.shape}, rows={len(row_ids)}, labels={len(labels)}")
    return row_ids, labels, probs


def macro_auc(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    vals = []
    for j in range(y.shape[1]):
        col = y[:, j]
        if col.min() == col.max():
            continue
        vals.append(float(roc_auc_score(col, p[:, j])))
    return {"macro_auc": float(np.mean(vals)) if vals else None, "valid_classes": len(vals)}


def flat_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    try:
        value = float(np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1])
    except Exception:
        return None
    return value if np.isfinite(value) else None


def metric_path_for_student(path: Path) -> Path:
    return path.with_name("metrics.json")


def read_metrics_summary(path: Path) -> dict[str, Any] | None:
    mpath = metric_path_for_student(path)
    if not mpath.exists():
        return None
    try:
        data = json.loads(mpath.read_text())
    except Exception:
        return None
    return {
        "experiment_id": data.get("experiment_id"),
        "backbone_actual": data.get("backbone_actual"),
        "target_mode": data.get("target_mode"),
        "loss_name": data.get("loss_name"),
        "auc_loss_weight": data.get("auc_loss_weight"),
        "runtime_sec": data.get("runtime_sec"),
        "export_mb": (data.get("exports") or {}).get("torchscript_size_mb"),
    }


def group_key(row_id: str, mode: str) -> str:
    text = str(row_id)
    if mode == "row":
        return text
    if mode == "file":
        return text.rsplit("_", 1)[0]
    if mode == "site":
        for part in text.replace("-", "_").split("_"):
            if len(part) >= 2 and part[0].upper() == "S" and part[1:].isdigit():
                return part.upper()
        return text.rsplit("_", 1)[0]
    raise ValueError(f"unknown group mode {mode!r}")


def lift_summary(values: list[float] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean_lift": float(arr.mean()),
        "median_lift": float(np.median(arr)),
        "p_lift_gt_0": float(np.mean(arr > 0)),
        "q05_lift": float(np.quantile(arr, 0.05)),
        "q25_lift": float(np.quantile(arr, 0.25)),
        "q75_lift": float(np.quantile(arr, 0.75)),
        "q95_lift": float(np.quantile(arr, 0.95)),
        "min_lift": float(arr.min()),
        "max_lift": float(arr.max()),
    }


def bootstrap_lift_vs_teacher(
    row_ids: np.ndarray,
    y_true: np.ndarray,
    teacher: np.ndarray,
    candidate: np.ndarray,
    *,
    group_mode: str,
    iters: int,
    seed: int,
) -> dict[str, Any]:
    groups = np.array([group_key(x, group_mode) for x in row_ids.astype(str).tolist()])
    unique_groups = np.array(sorted(set(groups.tolist())))
    by_group = {g: np.flatnonzero(groups == g) for g in unique_groups}
    rng = np.random.default_rng(seed)
    lifts: list[float] = []
    for _ in range(int(iters)):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([by_group[g] for g in sampled_groups])
        base_auc = macro_auc(y_true[idx], teacher[idx])["macro_auc"]
        cand_auc = macro_auc(y_true[idx], candidate[idx])["macro_auc"]
        if base_auc is not None and cand_auc is not None:
            lifts.append(float(cand_auc - base_auc))
    out: dict[str, Any] = {"iters": int(iters), "valid_iters": int(len(lifts)), "group_mode": group_mode, "n_groups": int(len(unique_groups))}
    out.update(lift_summary(lifts))
    return out


def leave_one_group_lift_vs_teacher(
    row_ids: np.ndarray,
    y_true: np.ndarray,
    teacher: np.ndarray,
    candidate: np.ndarray,
    *,
    group_mode: str,
    max_groups_detail: int,
) -> dict[str, Any]:
    groups = np.array([group_key(x, group_mode) for x in row_ids.astype(str).tolist()])
    unique_groups = np.array(sorted(set(groups.tolist())))
    rows = []
    for group in unique_groups:
        idx = np.flatnonzero(groups != group)
        if idx.size == 0:
            continue
        base_auc = macro_auc(y_true[idx], teacher[idx])["macro_auc"]
        cand_auc = macro_auc(y_true[idx], candidate[idx])["macro_auc"]
        if base_auc is None or cand_auc is None:
            continue
        rows.append({
            "held_out_group": str(group),
            "train_rows": int(idx.size),
            "teacher_auc": float(base_auc),
            "candidate_auc": float(cand_auc),
            "lift": float(cand_auc - base_auc),
        })
    out: dict[str, Any] = {"group_mode": group_mode, "n_groups": int(len(unique_groups)), "valid_groups": int(len(rows))}
    out.update(lift_summary([r["lift"] for r in rows]))
    out["worst_groups"] = sorted(rows, key=lambda r: r["lift"])[:max_groups_detail]
    out["best_groups"] = sorted(rows, key=lambda r: r["lift"], reverse=True)[:max_groups_detail]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--student-root", type=Path, required=True)
    ap.add_argument("--labels-csv", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--pattern", default="*/student_predictions.npz")
    ap.add_argument("--weights", default="0.0025,0.005,0.01,0.02,0.05,0.075,0.10,0.15,0.20,0.30")
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--bootstrap-iters", type=int, default=0, help="Bootstrap best blend lift by group")
    ap.add_argument("--bootstrap-group", choices=["file", "site", "row"], default="file")
    ap.add_argument("--bootstrap-seed", type=int, default=42)
    ap.add_argument("--leave-one-group", choices=["none", "file", "site", "row"], default="none", help="Leave-one-group-out lift stability for each best blend")
    ap.add_argument("--holdout-detail", type=int, default=30)
    ap.add_argument("--stability-top-n", type=int, default=0, help="Compute expensive stability only after ranking this many top blends")
    args = ap.parse_args()

    row_ids, labels, teacher = load_teacher(args.teacher)
    y = build_truth(pd.read_csv(args.labels_csv), row_ids, labels)
    teacher_auc = macro_auc(y, teacher)
    weights = parse_float_list(args.weights)

    results = []
    skipped = []
    for path in sorted(args.student_root.glob(args.pattern)):
        try:
            z = np.load(path, allow_pickle=True)
            s_rows = z["row_ids"].astype(str)
            s_labels = [str(x) for x in z["labels"].astype(str).tolist()]
            if not np.array_equal(s_rows, row_ids) or s_labels != labels:
                skipped.append({"path": str(path), "reason": "row_or_label_mismatch", "rows": int(len(s_rows)), "classes": int(len(s_labels))})
                continue
            if "pred_student" not in z.files:
                skipped.append({"path": str(path), "reason": f"missing_pred_student keys={z.files}"})
                continue
            student = z["pred_student"].astype(np.float32)
        except Exception as exc:
            skipped.append({"path": str(path), "reason": repr(exc)})
            continue

        standalone = macro_auc(y, student)
        blends = []
        for weight in weights:
            pred = (1.0 - float(weight)) * teacher + float(weight) * student
            bauc = macro_auc(y, pred)
            blends.append({
                "student_weight": float(weight),
                **bauc,
                "lift_vs_teacher": None if bauc["macro_auc"] is None or teacher_auc["macro_auc"] is None else float(bauc["macro_auc"] - teacher_auc["macro_auc"]),
                "corr_vs_teacher": flat_corr(pred, teacher),
            })
        blends.sort(key=lambda row: row["macro_auc"] if row["macro_auc"] is not None else -1, reverse=True)
        best = blends[0] if blends else None
        results.append({
            "path": str(path),
            "name": path.parent.name,
            "metrics_summary": read_metrics_summary(path),
            "standalone": standalone,
            "corr_student_vs_teacher": flat_corr(student, teacher),
            "best_blend": best,
            "best_blend_stability": {},
            "top_blends": blends[:5],
        })

    results.sort(key=lambda row: (row["best_blend"] or {}).get("macro_auc") or -1, reverse=True)
    if args.stability_top_n > 0 and (args.bootstrap_iters > 0 or args.leave_one_group != "none"):
        for row in results[: args.stability_top_n]:
            best = row.get("best_blend")
            if best is None:
                continue
            z = np.load(row["path"], allow_pickle=True)
            student = z["pred_student"].astype(np.float32)
            best_pred = (1.0 - float(best["student_weight"])) * teacher + float(best["student_weight"]) * student
            stability: dict[str, Any] = {}
            if args.bootstrap_iters > 0:
                stability["bootstrap"] = bootstrap_lift_vs_teacher(
                    row_ids,
                    y,
                    teacher,
                    best_pred,
                    group_mode=args.bootstrap_group,
                    iters=args.bootstrap_iters,
                    seed=args.bootstrap_seed,
                )
            if args.leave_one_group != "none":
                stability["leave_one_group"] = leave_one_group_lift_vs_teacher(
                    row_ids,
                    y,
                    teacher,
                    best_pred,
                    group_mode=args.leave_one_group,
                    max_groups_detail=args.holdout_detail,
                )
            row["best_blend_stability"] = stability
    summary = {
        "status": "student_pool_blend_audit_complete",
        "teacher": str(args.teacher),
        "teacher_auc": teacher_auc,
        "student_root": str(args.student_root),
        "pattern": args.pattern,
        "n_scanned": len(results) + len(skipped),
        "n_aligned": len(results),
        "n_skipped": len(skipped),
        "stability_args": {
            "bootstrap_iters": int(args.bootstrap_iters),
            "bootstrap_group": args.bootstrap_group,
            "bootstrap_seed": int(args.bootstrap_seed),
            "leave_one_group": args.leave_one_group,
            "stability_top_n": int(args.stability_top_n),
        },
        "top_by_blend": results[: args.top_k],
        "top_by_standalone": sorted(results, key=lambda row: row["standalone"].get("macro_auc") or -1, reverse=True)[: args.top_k],
        "skipped_head": skipped[: args.top_k],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "teacher_auc": teacher_auc,
        "n_scanned": summary["n_scanned"],
        "n_aligned": summary["n_aligned"],
        "top_by_blend": summary["top_by_blend"][:10],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
