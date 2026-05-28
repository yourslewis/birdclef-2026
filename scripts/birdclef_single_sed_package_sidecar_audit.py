#!/usr/bin/env python3
"""Package a single TorchScript SED model through soundscape inference and audit as v616 sidecar.

This is a no-submit verifier/audit helper for model data points: create a
birdclef_sed_soundscape_infer manifest, infer train_soundscapes, anchor-fill the
v616 proxy rows, and run birdclef_ensemble_strategy_audit on small rank blends.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_labels(sample_submission: Path, fallback_manifest: Path | None = None) -> list[str]:
    if sample_submission.exists():
        return pd.read_csv(sample_submission, nrows=1).columns[1:].astype(str).tolist()
    if fallback_manifest and fallback_manifest.exists():
        return [str(x) for x in json.loads(fallback_manifest.read_text())["labels"]]
    raise FileNotFoundError(f"No sample submission or fallback manifest for labels: {sample_submission}")


def build_infer_manifest(args: argparse.Namespace, labels: list[str]) -> Path:
    metrics: dict[str, Any] = {}
    if args.metrics_json and args.metrics_json.exists():
        metrics = json.loads(args.metrics_json.read_text())
    manifest = {
        "schema_version": 1,
        "description": args.description,
        "members": [{
            "name": args.member_name,
            "root": str(args.model_path.parent),
            "input_weight": 1.0,
            "normalized_weight": 1.0,
            "summary": metrics.get("summary", {}),
            "experiment_id": args.experiment_id,
        }],
        "models": [{
            "member": args.member_name,
            "fold_index": 0,
            "path": str(args.model_path.resolve()),
            "weight": 1.0,
            "torchscript_size_mb": round(args.model_path.stat().st_size / 1e6, 3),
        }],
        "labels": labels,
        "n_classes": len(labels),
        "audio_config": {
            "sample_rate": args.sample_rate,
            "duration_sec": args.duration_sec,
            "n_fft": args.n_fft,
            "hop_length": args.hop_length,
            "n_mels": args.n_mels,
        },
        "train_config_reference": str(args.train_config) if args.train_config else None,
        "total_size_mb": round(args.model_path.stat().st_size / 1e6, 3),
    }
    path = args.output_dir / f"{args.member_name}_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def build_sidecar(args: argparse.Namespace, infer_csv: Path, out_csv: Path) -> dict[str, Any]:
    anchor = pd.read_csv(args.anchor_csv)
    inf = pd.read_csv(infer_csv)
    cols = anchor.columns[1:].astype(str).tolist()
    for col in cols:
        if col not in inf.columns:
            inf[col] = 0.0
    inf = inf[["row_id", *cols]]
    side = anchor.copy()
    lookup = inf.set_index("row_id")
    matched: list[str] = []
    for i, rid in enumerate(side["row_id"].astype(str).tolist()):
        if rid in lookup.index:
            side.loc[i, cols] = lookup.loc[rid, cols].to_numpy(dtype=float)
            matched.append(rid)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    side.to_csv(out_csv, index=False)
    values = side[cols].to_numpy(dtype=float)
    return {
        "sidecar_csv": str(out_csv),
        "anchor_csv": str(args.anchor_csv),
        "v616_csv": str(args.v616_csv),
        "inference_csv": str(infer_csv),
        "anchor_rows": int(len(anchor)),
        "inference_rows": int(len(inf)),
        "matched_anchor_rows": int(len(matched)),
        "unmatched_anchor_rows": int(len(anchor) - len(matched)),
        "class_columns": int(len(cols)),
        "finite": bool(np.isfinite(values).all()),
        "nonconstant_columns": int(((values.max(axis=0) - values.min(axis=0)) > 0).sum()),
        "value_stats": {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std()),
        },
        "matched_examples": matched[:8],
    }


def audit_manifest(args: argparse.Namespace, sidecar_csv: Path) -> Path:
    recipes: list[dict[str, Any]] = [
        {"name": "anchor_only", "type": "rank_blend", "description": "Control anchor raw", "weights": {"anchor_v616_raw": 1.0}},
        {"name": "v616_baseline", "type": "member", "description": "Submitted v616 tied recipe", "member": "v616_final"},
        {"name": f"{args.recipe_prefix}_member_raw", "type": "member", "description": f"Anchor-filled direct {args.member_name} probabilities", "member": args.member_name},
    ]
    for w in args.weights:
        recipes.append({
            "name": f"{args.recipe_prefix}_w{str(w).replace('.', 'p')}",
            "type": "rank_blend",
            "description": f"{w:.2%} {args.member_name} sidecar rank blend",
            "weights": {"anchor_v616_raw": round(1.0 - w, 6), args.member_name: w},
        })
    manifest = {
        "name": f"birdclef_{args.recipe_prefix}_package_audit",
        "description": args.audit_description,
        "labels_csv": str(args.labels_csv),
        "anchor": "anchor_v616_raw",
        "baseline": "v616_final",
        "anchor_recipe": "anchor_only",
        "baseline_recipe": "v616_baseline",
        "allow_submit_approval": False,
        "members": {
            "anchor_v616_raw": {"role": "anchor", "description": "Samejima/v616 visual anchor raw output", "path": str(args.anchor_csv), "hidden_safe_status": "private_verifier_output"},
            "v616_final": {"role": "baseline_tied_recipe", "description": "Actual submitted v616 final output; public LB tied 0.949", "path": str(args.v616_csv), "hidden_safe_status": "submitted_private_verifier_output", "public_lb": 0.949},
            args.member_name: {"role": "packaged_analysis_branch", "description": args.member_description, "path": str(sidecar_csv), "hidden_safe_status": "train_soundscape_packaging_audit_not_submission_package"},
        },
        "recipes": recipes,
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
    path = args.output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def summarize_audit(audit_json: Path, summary_path: Path) -> dict[str, Any]:
    data = json.loads(audit_json.read_text())
    rows = []
    for recipe in data.get("recipes", []):
        local = recipe.get("local_metrics", {})
        comps = recipe.get("comparisons", {})
        gate = recipe.get("promotion_gate", {})
        rows.append({
            "recipe": recipe.get("name"),
            "macro_auc": local.get("macro_auc"),
            "valid_classes": local.get("valid_auc_classes"),
            "lift_vs_anchor": comps.get("anchor_only", {}).get("macro_auc_lift"),
            "lift_vs_v616": comps.get("v616_baseline", {}).get("macro_auc_lift"),
            "rank_corr_vs_v616": comps.get("v616_baseline", {}).get("rank_corr"),
            "mae_vs_v616": comps.get("v616_baseline", {}).get("mae"),
            "gate": gate.get("reason"),
            "eligible": gate.get("eligible_for_submission"),
        })
    ranked = sorted(
        [r for r in rows if r["macro_auc"] is not None],
        key=lambda r: (float(r.get("lift_vs_v616") if r.get("lift_vs_v616") is not None else -999), float(r.get("macro_auc") or -999)),
        reverse=True,
    )
    summary = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "audit_json": str(audit_json),
        "top_by_lift_vs_v616": ranked[:8],
        "all_recipes": rows,
        "submit_approved": bool(data.get("readiness", {}).get("submit_approved", False)),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--metrics-json", type=Path)
    ap.add_argument("--train-config", type=Path)
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--member-name", required=True)
    ap.add_argument("--recipe-prefix", required=True)
    ap.add_argument("--description", required=True)
    ap.add_argument("--audit-description", required=True)
    ap.add_argument("--member-description", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--base-dir", type=Path, default=Path("data"))
    ap.add_argument("--soundscape-dir", type=Path, default=Path("data/train_soundscapes"))
    ap.add_argument("--soundscape-list", type=Path)
    ap.add_argument("--sample-submission", type=Path, default=Path("data/sample_submission.csv"))
    ap.add_argument("--labels-csv", type=Path, default=Path("data/train_soundscapes_labels.csv"))
    ap.add_argument("--anchor-csv", type=Path, default=Path("artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission_anchor_raw.csv"))
    ap.add_argument("--v616-csv", type=Path, default=Path("artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission.csv"))
    ap.add_argument("--fallback-manifest", type=Path, default=Path("artifacts/sed_soundscape_packaging_audit/20260528T1220Z_sed_soft1279_soundscape_package/sed_soft1279_manifest.json"))
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--torch-threads", type=int, default=2)
    ap.add_argument("--bootstrap-iters", type=int, default=500)
    ap.add_argument("--weights", type=float, nargs="+", default=[0.0025, 0.005, 0.01, 0.02, 0.04, 0.08])
    ap.add_argument("--sample-rate", type=int, default=32000)
    ap.add_argument("--duration-sec", type=float, default=5.0)
    ap.add_argument("--n-fft", type=int, default=1024)
    ap.add_argument("--hop-length", type=int, default=512)
    ap.add_argument("--n-mels", type=int, default=160)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(args.sample_submission, args.fallback_manifest)
    infer_manifest = build_infer_manifest(args, labels)
    infer_csv = args.output_dir / f"train_soundscapes_{args.recipe_prefix}.csv"
    infer_npz = args.output_dir / f"train_soundscapes_{args.recipe_prefix}.npz"
    soundscape_list = args.soundscape_list
    if soundscape_list is None and args.labels_csv.exists():
        labels_df = pd.read_csv(args.labels_csv, usecols=["filename"])
        soundscape_list = args.output_dir / "labeled_soundscape_files.txt"
        soundscape_list.write_text("\n".join(labels_df["filename"].astype(str).drop_duplicates().tolist()) + "\n")
    infer_cmd = [
        sys.executable, "scripts/birdclef_sed_soundscape_infer.py",
        "--manifest", str(infer_manifest), "--base-dir", str(args.base_dir),
        "--sample-submission", str(args.sample_submission), "--output", str(infer_csv), "--npz-output", str(infer_npz),
        "--device", args.device, "--batch-size", str(args.batch_size), "--torch-threads", str(args.torch_threads),
    ]
    if soundscape_list is not None:
        infer_cmd.extend(["--soundscape-list", str(soundscape_list), "--soundscape-dir", str(args.soundscape_dir)])
    else:
        infer_cmd.extend(["--soundscape-dir", str(args.soundscape_dir)])
    proc = subprocess.run(infer_cmd, text=True, capture_output=True)
    (args.output_dir / "inference_command.txt").write_text(" ".join(infer_cmd) + "\n")
    (args.output_dir / "inference_stdout.txt").write_text(proc.stdout)
    (args.output_dir / "inference_stderr.txt").write_text(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    sidecar_csv = args.output_dir / "sidecars" / f"{args.recipe_prefix}_sidecar_234_anchorfill.csv"
    report = build_sidecar(args, infer_csv, sidecar_csv)
    (args.output_dir / "sidecar_build_report.json").write_text(json.dumps(report, indent=2) + "\n")

    manifest = audit_manifest(args, sidecar_csv)
    audit_dir = args.output_dir / "audit"
    audit_cmd = [sys.executable, "scripts/birdclef_ensemble_strategy_audit.py", "--manifest", str(manifest), "--output-dir", str(audit_dir), "--bootstrap-iters", str(args.bootstrap_iters), "--emit-candidate-csvs"]
    proc = subprocess.run(audit_cmd, text=True, capture_output=True)
    (args.output_dir / "audit_command.txt").write_text(" ".join(audit_cmd) + "\n")
    (args.output_dir / "audit_stdout.txt").write_text(proc.stdout)
    (args.output_dir / "audit_stderr.txt").write_text(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    summary = summarize_audit(audit_dir / "ensemble_strategy_audit.json", args.output_dir / "audit_summary.json")
    print(json.dumps({"status": "package_sidecar_audit_complete", "output_dir": str(args.output_dir), "sidecar_report": report, "top": summary["top_by_lift_vs_v616"][:5], "submit_approved": summary["submit_approved"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
