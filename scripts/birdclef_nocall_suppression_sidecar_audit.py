#!/usr/bin/env python3
"""Audit conservative no-call suppression sidecars for BirdCLEF 2026.

This no-slot verifier consumes the aggregate no-call/background gate predictions
from official train_soundscapes, applies bounded multiplicative suppression to a
hidden-safe baseline/proxy matrix, and runs the repo ensemble audit against the
v616 tied baseline.  The negative labels used by the gate are weak/unlabeled
windows, so this artifact is comparison-grade unless a hand-verified no-call
protocol upgrades it.
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


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.replace(",", " ").split() if x.strip()]


def load_scopes(cols: list[str], taxonomy_csv: Path, train_csv: Path, labels_csv: Path) -> dict[str, list[str]]:
    taxonomy = pd.read_csv(taxonomy_csv, dtype={"primary_label": str})
    train = pd.read_csv(train_csv, dtype={"primary_label": str})
    labels_df = pd.read_csv(labels_csv, dtype=str)
    all_cols = set(cols)
    train_labels = set(train["primary_label"].astype(str))
    no_train = {x for x in taxonomy["primary_label"].astype(str) if x not in train_labels}
    nonaves = set(taxonomy.loc[taxonomy["class_name"].astype(str) != "Aves", "primary_label"].astype(str))
    soundscape_positive: set[str] = set()
    for raw in labels_df["primary_label"].fillna("").astype(str):
        soundscape_positive.update(x.strip() for x in raw.split(";") if x.strip())
    scopes = {
        "all": cols,
        "nonaves_notrain": [c for c in cols if c in (nonaves | no_train)],
        "soundscape_positive": [c for c in cols if c in soundscape_positive],
        "notrain_only": [c for c in cols if c in no_train],
    }
    return {k: [c for c in v if c in all_cols] for k, v in scopes.items()}


def matrix_hash(values: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(values.astype(np.float64, copy=False)).tobytes()).hexdigest()


def build_candidate(
    base: pd.DataFrame,
    cols: list[str],
    gate: pd.DataFrame,
    gate_col: str,
    scope_cols: list[str],
    alpha: float,
    power: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = base.copy()
    nc_lookup = gate.set_index("row_id")[gate_col].astype(float)
    nc = base["row_id"].astype(str).map(nc_lookup).fillna(0.0).to_numpy(np.float64)
    suppress = np.clip(alpha * np.power(np.clip(nc, 0.0, 1.0), power), 0.0, 0.95)
    if scope_cols:
        values = out[scope_cols].to_numpy(np.float64)
        values = values * (1.0 - suppress[:, None])
        out.loc[:, scope_cols] = np.clip(values, 0.0, 1.0)
    arr = out[cols].to_numpy(np.float64)
    report = {
        "gate_col": gate_col,
        "scope_columns": int(len(scope_cols)),
        "alpha": float(alpha),
        "power": float(power),
        "matched_gate_rows": int(base["row_id"].astype(str).isin(set(gate["row_id"].astype(str))).sum()),
        "rows": int(len(base)),
        "finite": bool(np.isfinite(arr).all()),
        "nonconstant_columns": int(((arr.max(axis=0) - arr.min(axis=0)) > 0).sum()),
        "matrix_sha256": matrix_hash(arr),
        "suppression_stats": {
            "min": float(suppress.min()),
            "p50": float(np.quantile(suppress, 0.50)),
            "p90": float(np.quantile(suppress, 0.90)),
            "p99": float(np.quantile(suppress, 0.99)),
            "max": float(suppress.max()),
            "mean": float(suppress.mean()),
        },
    }
    return out, report


def audit_manifest(args: argparse.Namespace, members: dict[str, dict[str, Any]]) -> Path:
    recipes: list[dict[str, Any]] = [
        {"name": "anchor_only", "type": "rank_blend", "description": "Control anchor raw", "weights": {"anchor_v616_raw": 1.0}},
        {"name": "v616_baseline", "type": "member", "description": "Submitted v616 tied recipe", "member": "v616_final"},
    ]
    for name in members:
        recipes.append({"name": name, "type": "member", "description": f"No-call suppressed v616 candidate {name}", "member": name})
    manifest = {
        "name": "birdclef_nocall_suppression_sidecar_audit",
        "description": args.description,
        "labels_csv": str(args.labels_csv),
        "anchor": "anchor_v616_raw",
        "baseline": "v616_final",
        "anchor_recipe": "anchor_only",
        "baseline_recipe": "v616_baseline",
        "allow_submit_approval": False,
        "members": {
            "anchor_v616_raw": {
                "role": "anchor",
                "description": "Samejima/v616 visual anchor raw output",
                "path": str(args.anchor_csv),
                "hidden_safe_status": "private_verifier_output",
            },
            "v616_final": {
                "role": "baseline_tied_recipe",
                "description": "Actual submitted v616 final output; public LB tied 0.949",
                "path": str(args.v616_csv),
                "hidden_safe_status": "submitted_private_verifier_output",
                "public_lb": 0.949,
            },
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
    for name, info in members.items():
        manifest["members"][name] = {
            "role": "nocall_suppression_candidate",
            "description": info["description"],
            "path": info["path"],
            "hidden_safe_status": "comparison_grade_weak_no_call_gate_not_submission_package",
        }
    path = args.output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def summarize(audit_json: Path, candidate_reports: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    data = json.loads(audit_json.read_text())
    rows = []
    for recipe in data.get("recipes", []):
        local = recipe.get("local_metrics", {})
        comps = recipe.get("comparisons", {})
        gate = recipe.get("promotion_gate", {})
        name = recipe.get("name")
        rows.append({
            "recipe": name,
            "macro_auc": local.get("macro_auc"),
            "valid_classes": local.get("valid_auc_classes"),
            "matched_rows": local.get("matched_rows"),
            "lift_vs_anchor": comps.get("anchor_only", {}).get("macro_auc_lift"),
            "lift_vs_v616": comps.get("v616_baseline", {}).get("macro_auc_lift"),
            "rank_corr_vs_v616": comps.get("v616_baseline", {}).get("rank_corr"),
            "mae_vs_v616": comps.get("v616_baseline", {}).get("mae"),
            "top3_row_recall": local.get("top3_row_recall"),
            "top5_row_recall": local.get("top5_row_recall"),
            "gate_reason": gate.get("reason"),
            "eligible": gate.get("eligible_for_submission"),
            "candidate_report": candidate_reports.get(name),
        })
    ranked = sorted(
        [r for r in rows if r["macro_auc"] is not None],
        key=lambda r: (float(r["lift_vs_v616"] if r["lift_vs_v616"] is not None else -999), float(r["macro_auc"] or -999)),
        reverse=True,
    )
    summary = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "experiment_id": args.experiment_id,
        "gate_predictions": str(args.gate_predictions),
        "audit_json": str(audit_json),
        "submit_approved": bool(data.get("readiness", {}).get("submit_approved", False)),
        "best_by_lift_vs_v616": ranked[:10],
        "all_recipes": rows,
        "decision": "reject as submission-grade unless site/file gates are positive and weak no-call negatives are upgraded; comparison-grade only",
    }
    (args.output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-id", default="soundscape-nocall-suppression-v616-agg-20260529")
    ap.add_argument("--description", default="Conservative no-call suppression sidecar audit using aggregate soft1279-native gate")
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/soundscape_nocall_suppression/soundscape-nocall-suppression-v616-agg-20260529"))
    ap.add_argument("--gate-predictions", type=Path, default=Path("artifacts/soundscape_nocall_gate/soundscape-nocall-gate-soft1279native-agg-losite-20260528/nocall_gate_predictions.csv"))
    ap.add_argument("--anchor-csv", type=Path, default=Path("artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission_anchor_raw.csv"))
    ap.add_argument("--v616-csv", type=Path, default=Path("artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission.csv"))
    ap.add_argument("--labels-csv", type=Path, default=Path("data/train_soundscapes_labels.csv"))
    ap.add_argument("--taxonomy-csv", type=Path, default=Path("data/taxonomy.csv"))
    ap.add_argument("--train-csv", type=Path, default=Path("data/train.csv"))
    ap.add_argument("--alphas", default="0.01 0.02 0.04 0.06 0.08 0.10 0.12 0.16 0.20")
    ap.add_argument("--powers", default="1.0 2.0")
    ap.add_argument("--gate-cols", default="oof_no_call_prob final_no_call_prob")
    ap.add_argument("--scopes", default="all nonaves_notrain soundscape_positive notrain_only")
    ap.add_argument("--bootstrap-iters", type=int, default=500)
    ap.add_argument("--max-candidates", type=int, default=72)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = args.output_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(args.v616_csv)
    cols = [c for c in base.columns if c != "row_id"]
    gate = pd.read_csv(args.gate_predictions)
    if "row_id" not in gate.columns:
        raise ValueError(f"gate predictions missing row_id: {args.gate_predictions}")
    gate["row_id"] = gate["row_id"].astype(str)
    scopes = load_scopes(cols, args.taxonomy_csv, args.train_csv, args.labels_csv)
    alphas = parse_float_list(args.alphas)
    powers = parse_float_list(args.powers)
    gate_cols = [x for x in args.gate_cols.split() if x.strip()]
    scope_names = [x for x in args.scopes.split() if x.strip()]

    members: dict[str, dict[str, Any]] = {}
    reports: dict[str, Any] = {}
    for gate_col in gate_cols:
        if gate_col not in gate.columns:
            raise ValueError(f"gate column {gate_col!r} not found in {args.gate_predictions}")
        gate_short = gate_col.replace("_no_call_prob", "").replace("_", "")
        for scope in scope_names:
            if scope not in scopes:
                raise ValueError(f"unknown scope {scope!r}; available={sorted(scopes)}")
            for power in powers:
                ptag = str(power).replace(".", "p")
                for alpha in alphas:
                    if len(members) >= int(args.max_candidates):
                        break
                    atag = f"a{int(round(alpha * 1000)):03d}"
                    name = f"nocall_{gate_short}_{scope}_p{ptag}_{atag}"
                    cand, report = build_candidate(base, cols, gate, gate_col, scopes[scope], alpha, power)
                    out_csv = candidate_dir / f"{name}.csv"
                    cand.to_csv(out_csv, index=False)
                    report.update({"scope": scope, "candidate_csv": str(out_csv)})
                    members[name] = {
                        "path": str(out_csv),
                        "description": f"v616 multiplied by (1 - {alpha:g} * {gate_col}^{power:g}) over {scope} classes",
                    }
                    reports[name] = report
                if len(members) >= int(args.max_candidates):
                    break
            if len(members) >= int(args.max_candidates):
                break
        if len(members) >= int(args.max_candidates):
            break

    (args.output_dir / "candidate_reports.json").write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n")
    manifest = audit_manifest(args, members)
    audit_dir = args.output_dir / "audit"
    audit_cmd = [sys.executable, "scripts/birdclef_ensemble_strategy_audit.py", "--manifest", str(manifest), "--output-dir", str(audit_dir), "--bootstrap-iters", str(args.bootstrap_iters), "--emit-candidate-csvs"]
    proc = subprocess.run(audit_cmd, text=True, capture_output=True)
    (args.output_dir / "audit_command.txt").write_text(" ".join(audit_cmd) + "\n")
    (args.output_dir / "audit_stdout.txt").write_text(proc.stdout)
    (args.output_dir / "audit_stderr.txt").write_text(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    summary = summarize(audit_dir / "ensemble_strategy_audit.json", reports, args)
    print(json.dumps({"status": "nocall_suppression_audit_complete", "output_dir": str(args.output_dir), "top": summary["best_by_lift_vs_v616"][:8], "submit_approved": summary["submit_approved"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
