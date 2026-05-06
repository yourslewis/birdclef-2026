#!/usr/bin/env python3
"""Run fold OOF SED pilots and aggregate predictions.

This wraps birdclef_sed_pilot_train.py with n_folds/fold_index configs,
then concatenates the held-out fold predictions into one oof_predictions.npz.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


def macro_auc(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
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


def run_fold(repo_root: Path, train_script: Path, base_cfg: dict[str, Any], output_root: Path, fold: int, n_folds: int) -> dict[str, Any]:
    cfg = dict(base_cfg)
    cfg["n_folds"] = n_folds
    cfg["fold_index"] = fold
    cfg["output_dir"] = str(output_root / f"fold{fold}")
    cfg_path = output_root / f"config_fold{fold}.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    cmd = [sys.executable, str(train_script), "--config", str(cfg_path)]
    started = time.time()
    proc = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True)
    fold_dir = Path(cfg["output_dir"])
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "runner_stdout.txt").write_text(proc.stdout)
    (fold_dir / "runner_stderr.txt").write_text(proc.stderr)
    if proc.returncode != 0:
        return {
            "fold_index": fold,
            "status": "failed",
            "returncode": proc.returncode,
            "runtime_sec": round(time.time() - started, 3),
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }
    metrics_path = fold_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["runner_runtime_sec"] = round(time.time() - started, 3)
    return metrics


def aggregate(output_root: Path, fold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    y_parts, pred_parts, file_parts, idx_parts = [], [], [], []
    labels = None
    for m in fold_metrics:
        z = np.load(Path(m["holdout_predictions_path"]), allow_pickle=True)
        all_files = z["files"].astype(str)
        idx = z["val_indices"].astype(int)
        y_parts.append(z["y_val"].astype(np.float32))
        pred_parts.append(z["pred_val"].astype(np.float32))
        file_parts.append(all_files[idx])
        idx_parts.append(idx)
        labels = z["labels"].astype(str)
    y = np.concatenate(y_parts, axis=0)
    pred = np.concatenate(pred_parts, axis=0)
    files = np.concatenate(file_parts, axis=0)
    indices = np.concatenate(idx_parts, axis=0)
    order = np.argsort(indices)
    y = y[order]
    pred = pred[order]
    files = files[order]
    indices = indices[order]
    auc = macro_auc(y, pred)
    out_npz = output_root / "oof_predictions.npz"
    np.savez_compressed(out_npz, y_oof=y, pred_oof=pred, files=files, selected_indices=indices, labels=labels)
    return {
        "status": "oof_complete",
        "n_oof": int(len(files)),
        "n_classes": int(y.shape[1]),
        "oof_predictions_path": str(out_npz),
        "auc_summary": auc,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--experiment-id", type=str)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    train_script = repo_root / "scripts" / "birdclef_sed_pilot_train.py"
    base_cfg = json.loads(args.base_config.read_text())
    if args.experiment_id:
        base_cfg["experiment_id"] = args.experiment_id
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "base_config.json").write_text(json.dumps(base_cfg, indent=2) + "\n")
    fold_metrics = []
    for fold in range(args.n_folds):
        print(f"Running fold {fold}/{args.n_folds} for {base_cfg.get('experiment_id')}", flush=True)
        metrics = run_fold(repo_root, train_script, base_cfg, args.output_root, fold, args.n_folds)
        print(json.dumps({
            "fold_index": fold,
            "status": metrics.get("status"),
            "auc_summary": metrics.get("auc_summary"),
            "last_epoch": metrics.get("epochs", [{}])[-1] if metrics.get("epochs") else None,
        }, indent=2), flush=True)
        if metrics.get("status") != "pilot_complete":
            (args.output_root / "oof_summary.json").write_text(json.dumps({"status": "failed", "folds": fold_metrics + [metrics]}, indent=2) + "\n")
            return 1
        fold_metrics.append(metrics)
    summary = aggregate(args.output_root, fold_metrics)
    summary.update({
        "experiment_id": base_cfg.get("experiment_id"),
        "n_folds": args.n_folds,
        "folds": [
            {
                "fold_index": m["fold_index"],
                "n_train": m["n_train"],
                "n_val": m["n_val"],
                "auc_summary": m["auc_summary"],
                "last_epoch": m["epochs"][-1],
                "exports": m.get("exports", {}),
                "metrics_path": str(args.output_root / f"fold{m['fold_index']}" / "metrics.json"),
            }
            for m in fold_metrics
        ],
    })
    text = json.dumps(summary, indent=2)
    print(text)
    (args.output_root / "oof_summary.json").write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
