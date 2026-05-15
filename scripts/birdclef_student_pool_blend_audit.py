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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--student-root", type=Path, required=True)
    ap.add_argument("--labels-csv", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--pattern", default="*/student_predictions.npz")
    ap.add_argument("--weights", default="0.0025,0.005,0.01,0.02,0.05,0.075,0.10,0.15,0.20,0.30")
    ap.add_argument("--top-k", type=int, default=30)
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
            "top_blends": blends[:5],
        })

    results.sort(key=lambda row: (row["best_blend"] or {}).get("macro_auc") or -1, reverse=True)
    summary = {
        "status": "student_pool_blend_audit_complete",
        "teacher": str(args.teacher),
        "teacher_auc": teacher_auc,
        "student_root": str(args.student_root),
        "pattern": args.pattern,
        "n_scanned": len(results) + len(skipped),
        "n_aligned": len(results),
        "n_skipped": len(skipped),
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
