#!/usr/bin/env python3
"""Build 72->234 train-soundscape sequence sidecars and audit them vs v616.

This is a no-submit local/proxy audit.  It maps leave-site sequence predictions
for the 72 non-Aves/no-train scope back onto the 240-row v616 train-soundscape
proxy matrix, leaves all other classes equal to the anchor, and runs the
repo-owned ensemble strategy audit against both anchor_only and v616_baseline.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def parse_time_seconds(text: str) -> int:
    parts = [float(x) for x in str(text).split(":")]
    if len(parts) == 3:
        sec = parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        sec = parts[0] * 60 + parts[1]
    else:
        sec = parts[0]
    if abs(sec - round(sec)) < 1e-6:
        return int(round(sec))
    return int(sec)


def soundscape_row_id(filename: str, end_text: str) -> str:
    return f"{Path(filename).stem}_{parse_time_seconds(end_text)}"


def read_csv_matrix(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if not header or header[0] != "row_id":
            raise ValueError(f"{path} missing row_id header")
        row_ids: list[str] = []
        values: list[list[float]] = []
        for row in reader:
            row_ids.append(row[0])
            values.append([float(x) for x in row[1:]])
    return row_ids, header[1:], np.asarray(values, dtype=np.float32)


def write_csv_matrix(path: Path, row_ids: list[str], cols: list[str], values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", *cols])
        for row_id, arr in zip(row_ids, values):
            writer.writerow([row_id, *[f"{float(x):.8g}" for x in arr]])


def load_soundscape_row_ids(labels_csv: Path) -> list[str]:
    with labels_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        return [soundscape_row_id(row["filename"], row["end"]) for row in reader]


def build_sidecar(
    *,
    base_csv: Path,
    sequence_npz: Path,
    labels_csv: Path,
    pred_key: str,
    out_csv: Path,
) -> dict[str, Any]:
    base_rows, cols, base_values = read_csv_matrix(base_csv)
    col_to_idx = {c: i for i, c in enumerate(cols)}
    base_row_to_idx = {r: i for i, r in enumerate(base_rows)}

    z = np.load(sequence_npz, allow_pickle=False)
    val_idx = z["val_idx"].astype(np.int64)
    preds = z[pred_key].astype(np.float32)
    labels = [str(x) for x in z["labels"].tolist()]
    if preds.shape != (len(val_idx), len(labels)):
        raise ValueError(f"prediction shape mismatch: preds={preds.shape} val_idx={len(val_idx)} labels={len(labels)}")

    all_train_row_ids = load_soundscape_row_ids(labels_csv)
    sidecar = base_values.copy()
    matched_pairs = 0
    skipped_unmatched_proxy_rows: list[str] = []
    label_missing = [lab for lab in labels if lab not in col_to_idx]
    if label_missing:
        raise ValueError(f"sequence labels missing from base columns: {label_missing[:8]}")

    matched_base_rows: set[str] = set()
    for k, src_idx in enumerate(val_idx.tolist()):
        if src_idx < 0 or src_idx >= len(all_train_row_ids):
            raise IndexError(f"val_idx {src_idx} outside labels_csv rows {len(all_train_row_ids)}")
        rid = all_train_row_ids[src_idx]
        if rid not in base_row_to_idx:
            continue
        row_i = base_row_to_idx[rid]
        for lab_j, lab in enumerate(labels):
            sidecar[row_i, col_to_idx[lab]] = float(preds[k, lab_j])
        matched_base_rows.add(rid)
        matched_pairs += 1

    for rid in base_rows:
        if rid not in matched_base_rows:
            skipped_unmatched_proxy_rows.append(rid)

    finite = bool(np.isfinite(sidecar).all())
    if not finite:
        raise ValueError("sidecar has non-finite values")
    nonconstant_cols = int(((sidecar.max(axis=0) - sidecar.min(axis=0)) > 0).sum())
    write_csv_matrix(out_csv, base_rows, cols, sidecar)
    return {
        "output_csv": str(out_csv),
        "base_csv": str(base_csv),
        "sequence_npz": str(sequence_npz),
        "pred_key": pred_key,
        "rows": len(base_rows),
        "class_columns": len(cols),
        "sequence_scope_labels": len(labels),
        "matched_proxy_rows": len(matched_base_rows),
        "unmatched_proxy_rows": len(skipped_unmatched_proxy_rows),
        "unmatched_proxy_row_examples": skipped_unmatched_proxy_rows[:12],
        "finite": finite,
        "nonconstant_columns": nonconstant_cols,
        "value_stats": {
            "min": float(sidecar.min()),
            "max": float(sidecar.max()),
            "mean": float(sidecar.mean()),
            "std": float(sidecar.std()),
        },
    }


def make_manifest(args: argparse.Namespace, context_csv: Path, r2_csv: Path, output_dir: Path) -> Path:
    manifest = {
        "name": "birdclef_soundscape_sequence_sidecar_audit_20260526",
        "description": "No-submit audit of train_soundscape sequence/file/site 72-label sidecars wrapped into 234-class v616 proxy matrices.",
        "labels_csv": str(args.labels_csv),
        "anchor": "anchor_v616_raw",
        "baseline": "v616_final",
        "anchor_recipe": "anchor_only",
        "baseline_recipe": "v616_baseline",
        "allow_submit_approval": False,
        "members": {
            "anchor_v616_raw": {
                "role": "anchor",
                "description": "Samejima/v616 visual anchor raw output.",
                "path": str(args.anchor_csv),
                "hidden_safe_status": "private_verifier_output",
            },
            "v616_final": {
                "role": "baseline_tied_recipe",
                "description": "Actual submitted v616 final output; public LB tied 0.949.",
                "path": str(args.v616_csv),
                "hidden_safe_status": "submitted_private_verifier_output",
                "public_lb": 0.949,
            },
            "seq_context_sidecar": {
                "role": "analysis_branch",
                "description": "DyMN10 context MLP leave-site sidecar; 72-label train_soundscape scope only, anchor-filled elsewhere.",
                "path": str(context_csv),
                "hidden_safe_status": "analysis_only_leave_site_oof_proxy_not_submission_package",
            },
            "seq_r2_nofile_reg_sidecar": {
                "role": "analysis_branch",
                "description": "Regularized radius-2 no-file context leave-site sidecar; 72-label train_soundscape scope only, anchor-filled elsewhere.",
                "path": str(r2_csv),
                "hidden_safe_status": "analysis_only_leave_site_oof_proxy_not_submission_package",
            },
        },
        "recipes": [
            {"name": "anchor_only", "type": "rank_blend", "description": "Control: v616 anchor rank reconstruction.", "weights": {"anchor_v616_raw": 1.0}},
            {"name": "v616_baseline", "type": "member", "description": "Control: actual submitted v616 tied recipe.", "member": "v616_final"},
            {"name": "seq_context_w01", "type": "rank_blend", "description": "1% DyMN10 context sequence sidecar.", "weights": {"anchor_v616_raw": 0.99, "seq_context_sidecar": 0.01}},
            {"name": "seq_context_w02", "type": "rank_blend", "description": "2% DyMN10 context sequence sidecar.", "weights": {"anchor_v616_raw": 0.98, "seq_context_sidecar": 0.02}},
            {"name": "seq_context_w04", "type": "rank_blend", "description": "4% DyMN10 context sequence sidecar.", "weights": {"anchor_v616_raw": 0.96, "seq_context_sidecar": 0.04}},
            {"name": "seq_r2_w01", "type": "rank_blend", "description": "1% regularized r2/no-file sidecar.", "weights": {"anchor_v616_raw": 0.99, "seq_r2_nofile_reg_sidecar": 0.01}},
            {"name": "seq_r2_w02", "type": "rank_blend", "description": "2% regularized r2/no-file sidecar.", "weights": {"anchor_v616_raw": 0.98, "seq_r2_nofile_reg_sidecar": 0.02}},
            {"name": "seq_r2_w04", "type": "rank_blend", "description": "4% regularized r2/no-file sidecar.", "weights": {"anchor_v616_raw": 0.96, "seq_r2_nofile_reg_sidecar": 0.04}},
            {"name": "seq_context02_r201", "type": "rank_blend", "description": "2% context + 1% r2 sidecar combo.", "weights": {"anchor_v616_raw": 0.97, "seq_context_sidecar": 0.02, "seq_r2_nofile_reg_sidecar": 0.01}},
            {"name": "seq_context01_r202", "type": "rank_blend", "description": "1% context + 2% r2 sidecar combo.", "weights": {"anchor_v616_raw": 0.97, "seq_context_sidecar": 0.01, "seq_r2_nofile_reg_sidecar": 0.02}},
        ],
        "bootstrap": {"iters": int(args.bootstrap_iters), "groups": ["site", "file"]},
        "promotion_gates": {
            "matched_rows_min": 190,
            "valid_classes_min": 40,
            "lift_vs_anchor_min": 0.006,
            "lift_vs_baseline_min": 0.001,
            "site_bootstrap_q05_min": 0.003,
            "file_bootstrap_q05_min": 0.0015,
            "leave_one_site_min": 0.003,
            "leave_one_file_q05_min": 0.001,
            "leave_one_file_p_gt_0_min": 0.9,
        },
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def summarize_audit(audit_json: Path, summary_path: Path) -> dict[str, Any]:
    data = json.loads(audit_json.read_text())
    rows: list[dict[str, Any]] = []
    for recipe in data.get("recipes", []):
        name = recipe.get("name")
        local = recipe.get("local_metrics", {})
        comps = recipe.get("comparisons", {})
        vs_base = comps.get("v616_baseline", {})
        vs_anchor = comps.get("anchor_only", {})
        rows.append({
            "recipe": name,
            "macro_auc": local.get("macro_auc"),
            "valid_classes": local.get("valid_auc_classes"),
            "lift_vs_anchor": vs_anchor.get("macro_auc_lift"),
            "lift_vs_v616": vs_base.get("macro_auc_lift"),
            "rank_corr_vs_v616": vs_base.get("rank_corr"),
            "mae_vs_v616": vs_base.get("mae"),
            "gate": recipe.get("gate", {}).get("reason"),
            "eligible": recipe.get("gate", {}).get("eligible_for_submission"),
        })
    ranked = sorted(
        [r for r in rows if r["macro_auc"] is not None],
        key=lambda r: (float(r.get("lift_vs_v616") or -999), float(r.get("macro_auc") or -999)),
        reverse=True,
    )
    summary = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "audit_json": str(audit_json),
        "top_by_lift_vs_v616": ranked[:8],
        "all_recipes": rows,
        "submit_approved": bool(data.get("submit_approved", False)),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor-csv", type=Path, default=Path("artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission_anchor_raw.csv"))
    ap.add_argument("--v616-csv", type=Path, default=Path("artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission.csv"))
    ap.add_argument("--labels-csv", type=Path, default=Path("data/train_soundscapes_labels.csv"))
    ap.add_argument("--context-npz", type=Path, default=Path("artifacts/soundscape_sequence_mining/soundscape-sequence-dymn10-context-losite-ep16-20260526/leave_site_predictions.npz"))
    ap.add_argument("--r2-npz", type=Path, default=Path("artifacts/soundscape_sequence_mining/soundscape-sequence-dymn10-r2-nofile-reg-losite-ep20-20260526/leave_site_predictions.npz"))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--bootstrap-iters", type=int, default=500)
    ap.add_argument("--python", default="python")
    args = ap.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    sidecar_dir = out / "sidecars"
    context_csv = sidecar_dir / "seq_context_sidecar_234.csv"
    r2_csv = sidecar_dir / "seq_r2_nofile_reg_sidecar_234.csv"

    build_info = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "context_sidecar": build_sidecar(
            base_csv=args.anchor_csv,
            sequence_npz=args.context_npz,
            labels_csv=args.labels_csv,
            pred_key="context_pred",
            out_csv=context_csv,
        ),
        "r2_sidecar": build_sidecar(
            base_csv=args.anchor_csv,
            sequence_npz=args.r2_npz,
            labels_csv=args.labels_csv,
            pred_key="context_pred",
            out_csv=r2_csv,
        ),
    }
    (out / "sidecar_build_report.json").write_text(json.dumps(build_info, indent=2) + "\n")

    manifest = make_manifest(args, context_csv, r2_csv, out)
    audit_dir = out / "audit"
    cmd = [
        args.python,
        "scripts/birdclef_ensemble_strategy_audit.py",
        "--manifest",
        str(manifest),
        "--output-dir",
        str(audit_dir),
        "--bootstrap-iters",
        str(args.bootstrap_iters),
        "--emit-candidate-csvs",
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    (out / "audit_command.txt").write_text(" ".join(cmd) + "\n")
    (out / "audit_stdout.txt").write_text(result.stdout)
    (out / "audit_stderr.txt").write_text(result.stderr)
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    audit_json = audit_dir / "ensemble_strategy_audit.json"
    summary = summarize_audit(audit_json, out / "audit_summary.json")
    print(json.dumps({"output_dir": str(out), "summary": summary["top_by_lift_vs_v616"][:5]}, indent=2))


if __name__ == "__main__":
    main()
