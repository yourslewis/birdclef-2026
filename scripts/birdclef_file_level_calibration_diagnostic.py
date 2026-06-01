#!/usr/bin/env python3
"""File-level calibration/mapping diagnostic for BirdCLEF train_soundscape sidecars.

This no-submit diagnostic uses leave-site OOF train_soundscape predictions from
recent 72-label non-Aves/no-train sequence models and tests whether file-level
MIL evidence can be mapped back to row predictions without destroying the row
ranking that low-weight sidecars need.

It emits:
- OOF row/file metrics for candidate mappings;
- 72->234 anchor-filled sidecar CSVs for the best mappings;
- an ensemble-strategy audit vs the v616 local proxy;
- a compact JSON summary for ledger/table updates.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def parse_time_seconds(text: str) -> int:
    parts = [float(x) for x in str(text).split(":")]
    if len(parts) == 3:
        sec = parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        sec = parts[0] * 60 + parts[1]
    else:
        sec = parts[0]
    return int(round(sec)) if abs(sec - round(sec)) < 1e-6 else int(sec)


def site_from_filename(name: str) -> str:
    m = re.search(r"_(S\d+)_", str(name))
    return m.group(1) if m else "UNKNOWN"


def row_id(filename: str, end_text: str) -> str:
    return f"{Path(filename).stem}_{parse_time_seconds(end_text)}"


def load_rows(labels_csv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with labels_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            filename = str(r["filename"])
            labels = [x.strip() for x in str(r.get("primary_label", "")).split(";") if x.strip()]
            rows.append({
                "src_idx": i,
                "filename": filename,
                "start": str(r["start"]),
                "end": str(r["end"]),
                "row_id": row_id(filename, str(r["end"])),
                "site": site_from_filename(filename),
                "labels": labels,
            })
    return rows


def load_scope_sets(data_root: Path) -> tuple[set[str], set[str]]:
    train_labels: set[str] = set()
    with (data_root / "train.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            train_labels.add(str(r["primary_label"]))
    no_train: set[str] = set()
    nonaves: set[str] = set()
    with (data_root / "taxonomy.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            lab = str(r["primary_label"])
            if lab not in train_labels:
                no_train.add(lab)
            if str(r.get("class_name", "")) != "Aves":
                nonaves.add(lab)
    return no_train, nonaves


def make_targets(rows: list[dict[str, Any]], labels: list[str]) -> np.ndarray:
    lab_to_i = {lab: i for i, lab in enumerate(labels)}
    y = np.zeros((len(rows), len(labels)), dtype=np.float32)
    for i, r in enumerate(rows):
        for lab in r["labels"]:
            j = lab_to_i.get(str(lab))
            if j is not None:
                y[i, j] = 1.0
    return y


def macro_auc(y: np.ndarray, p: np.ndarray, label_mask: np.ndarray | None = None) -> tuple[float | None, int]:
    if label_mask is None:
        cols = range(y.shape[1])
    else:
        cols = np.where(label_mask)[0].tolist()
    vals: list[float] = []
    for j in cols:
        yy = y[:, j]
        if yy.min() == yy.max():
            continue
        vals.append(float(roc_auc_score(yy, p[:, j])))
    if not vals:
        return None, 0
    return float(np.mean(vals)), len(vals)


def file_pool(rows: list[dict[str, Any]], idx: np.ndarray, arr: np.ndarray, mode: str = "max") -> tuple[np.ndarray, np.ndarray, list[str]]:
    files = [rows[int(i)]["filename"] for i in idx.tolist()]
    unique = sorted(set(files))
    out = np.zeros((len(unique), arr.shape[1]), dtype=np.float32)
    for k, fname in enumerate(unique):
        inds = [i for i, f in enumerate(files) if f == fname]
        block = arr[inds]
        if mode == "max":
            out[k] = block.max(axis=0)
        elif mode == "mean":
            out[k] = block.mean(axis=0)
        else:
            raise ValueError(mode)
    return out, np.asarray(unique, dtype=object), files


def file_broadcast(rows: list[dict[str, Any]], idx: np.ndarray, arr: np.ndarray, mode: str = "max") -> np.ndarray:
    pooled, unique, files = file_pool(rows, idx, arr, mode=mode)
    by_file = {str(f): pooled[i] for i, f in enumerate(unique.tolist())}
    return np.vstack([by_file[f] for f in files]).astype(np.float32)


def file_mil_auc(rows: list[dict[str, Any]], idx: np.ndarray, y_all: np.ndarray, p: np.ndarray, label_mask: np.ndarray | None = None) -> tuple[float | None, int]:
    y_file, _, _ = file_pool(rows, idx, y_all[idx], mode="max")
    p_file, _, _ = file_pool(rows, idx, p, mode="max")
    return macro_auc(y_file, p_file, label_mask)


def leave_site_auc_mean(
    rows: list[dict[str, Any]],
    idx: np.ndarray,
    y_all: np.ndarray,
    p: np.ndarray,
    *,
    label_mask: np.ndarray | None = None,
    file_mil: bool = False,
) -> tuple[float | None, int, list[dict[str, Any]]]:
    """Mean of per-held-site macro AUCs, matching sequence-training ledgers.

    A pooled AUC across all held-out rows is misleading here because site priors
    differ strongly; the canonical comparable metric is the mean of fold/site
    macro AUCs emitted by birdclef_soundscape_sequence_mining.py.
    """
    sites = sorted({rows[int(i)]["site"] for i in idx.tolist()})
    fold_rows: list[dict[str, Any]] = []
    vals: list[float] = []
    valid_counts: list[int] = []
    for site in sites:
        pos = np.asarray([k for k, src_i in enumerate(idx.tolist()) if rows[int(src_i)]["site"] == site], dtype=np.int64)
        if len(pos) == 0:
            continue
        sub_idx = idx[pos]
        if file_mil:
            auc, valid = file_mil_auc(rows, sub_idx, y_all, p[pos], label_mask)
        else:
            auc, valid = macro_auc(y_all[sub_idx], p[pos], label_mask)
        fold_rows.append({"site": site, "auc": auc, "valid_classes": valid, "n_rows": int(len(pos))})
        if auc is not None:
            vals.append(float(auc))
            valid_counts.append(int(valid))
    if not vals:
        return None, 0, fold_rows
    return float(np.mean(vals)), int(round(float(np.mean(valid_counts)))) if valid_counts else 0, fold_rows


def read_csv_matrix(path: Path) -> tuple[list[str], list[str], np.ndarray]: 
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
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
        for rid, arr in zip(row_ids, values):
            writer.writerow([rid, *[f"{float(x):.8g}" for x in arr]])


def write_sidecar(*, base_csv: Path, rows: list[dict[str, Any]], idx: np.ndarray, labels: list[str], preds: np.ndarray, out_csv: Path) -> dict[str, Any]:
    base_rows, cols, base_values = read_csv_matrix(base_csv)
    col_to_i = {c: i for i, c in enumerate(cols)}
    row_to_i = {r: i for i, r in enumerate(base_rows)}
    sidecar = base_values.copy()
    matched: set[str] = set()
    for k, src_i in enumerate(idx.tolist()):
        rid = rows[int(src_i)]["row_id"]
        bi = row_to_i.get(rid)
        if bi is None:
            continue
        for j, lab in enumerate(labels):
            sidecar[bi, col_to_i[lab]] = float(preds[k, j])
        matched.add(rid)
    if not np.isfinite(sidecar).all():
        raise ValueError(f"non-finite sidecar for {out_csv}")
    write_csv_matrix(out_csv, base_rows, cols, sidecar)
    return {
        "path": str(out_csv),
        "rows": len(base_rows),
        "class_columns": len(cols),
        "scope_labels": len(labels),
        "matched_proxy_rows": len(matched),
        "unmatched_proxy_rows": len(base_rows) - len(matched),
        "finite": True,
        "nonconstant_columns": int(((sidecar.max(axis=0) - sidecar.min(axis=0)) > 0).sum()),
        "value_stats": {"min": float(sidecar.min()), "max": float(sidecar.max()), "mean": float(sidecar.mean()), "std": float(sidecar.std())},
    }


def safe_member_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")[:80]


def make_manifest(args: argparse.Namespace, sidecar_infos: list[dict[str, Any]], out_dir: Path) -> Path:
    members: dict[str, Any] = {
        "anchor_v616_raw": {"role": "anchor", "description": "v616 anchor raw output", "path": str(args.anchor_csv), "hidden_safe_status": "private_verifier_output"},
        "v616_final": {"role": "baseline_tied_recipe", "description": "submitted v616 final output", "path": str(args.v616_csv), "hidden_safe_status": "submitted_private_verifier_output", "public_lb": 0.949},
    }
    recipes: list[dict[str, Any]] = [
        {"name": "anchor_only", "type": "rank_blend", "description": "Control: raw anchor", "weights": {"anchor_v616_raw": 1.0}},
        {"name": "v616_baseline", "type": "member", "description": "Control: submitted v616", "member": "v616_final"},
    ]
    for info in sidecar_infos:
        m = info["member"]
        members[m] = {"role": "analysis_branch", "description": info["description"], "path": info["path"], "hidden_safe_status": "analysis_only_leave_site_oof_proxy_not_submission_package"}
        for w in (0.005, 0.01, 0.02, 0.04):
            pct = str(w).replace("0.", "w").replace(".", "p")
            recipes.append({
                "name": f"{m}_{pct}",
                "type": "rank_blend",
                "description": f"{w:.1%} file-calibrated sidecar {m}",
                "weights": {"anchor_v616_raw": 1.0 - w, m: w},
            })
    manifest = {
        "name": "birdclef_file_level_calibration_diagnostic_20260601",
        "description": "No-submit audit of file-level calibrated 72-label train_soundscape sidecars wrapped into 234-class v616 proxy matrices.",
        "labels_csv": str(args.labels_csv),
        "anchor": "anchor_v616_raw",
        "baseline": "v616_final",
        "anchor_recipe": "anchor_only",
        "baseline_recipe": "v616_baseline",
        "allow_submit_approval": False,
        "members": members,
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
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def summarize_audit(audit_json: Path, summary_path: Path) -> dict[str, Any]:
    data = json.loads(audit_json.read_text())
    rows: list[dict[str, Any]] = []
    for recipe in data.get("recipes", []):
        local = recipe.get("local_metrics", {})
        comps = recipe.get("comparisons", {})
        vs_base = comps.get("v616_baseline", {})
        vs_anchor = comps.get("anchor_only", {})
        rows.append({
            "recipe": recipe.get("name"),
            "macro_auc": local.get("macro_auc"),
            "valid_classes": local.get("valid_auc_classes"),
            "lift_vs_anchor": vs_anchor.get("macro_auc_lift"),
            "lift_vs_v616": vs_base.get("macro_auc_lift"),
            "rank_corr_vs_v616": vs_base.get("rank_corr"),
            "mae_vs_v616": vs_base.get("mae"),
            "gate": recipe.get("gate", {}).get("reason"),
            "eligible": recipe.get("gate", {}).get("eligible_for_submission"),
        })
    ranked = sorted([r for r in rows if r["macro_auc"] is not None], key=lambda r: (float(r.get("lift_vs_v616") or -999), float(r.get("macro_auc") or -999)), reverse=True)
    summary = {"audit_json": str(audit_json), "top_by_lift_vs_v616": ranked[:10], "all_recipes": rows, "submit_approved": bool(data.get("submit_approved", False))}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def load_npz(path: Path, key: str) -> tuple[np.ndarray, list[str], np.ndarray]:
    z = np.load(path, allow_pickle=False)
    return z["val_idx"].astype(np.int64), [str(x) for x in z["labels"].tolist()], z[key].astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--labels-csv", type=Path, default=Path("data/train_soundscapes_labels.csv"))
    ap.add_argument("--anchor-csv", type=Path, default=Path("artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission_anchor_raw.csv"))
    ap.add_argument("--v616-csv", type=Path, default=Path("artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission.csv"))
    ap.add_argument("--panns-npz", type=Path, default=Path("artifacts/soundscape_sequence_mining/soundscape-sequence-panns-cnn14-nonaves-notrain-rowonly-losite-ep24-20260531/leave_site_predictions.npz"))
    ap.add_argument("--dymn-npz", type=Path, default=Path("artifacts/soundscape_sequence_mining/soundscape-sequence-dymn10-nonaves-notrain-r2-filectx-filemil-losite-ep22-20260531/leave_site_predictions.npz"))
    ap.add_argument("--fused-npz", type=Path, default=Path("artifacts/soundscape_sequence_mining/soundscape-sequence-fused-dymn10-panns-nonaves-notrain-r2-filectx-filemil-losite-ep22-20260601/leave_site_predictions.npz"))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--bootstrap-iters", type=int, default=200)
    ap.add_argument("--python", default="python")
    args = ap.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.labels_csv)
    p_idx, labels, panns_row = load_npz(args.panns_npz, "row_only_pred")
    d_idx, d_labels, dymn_ctx = load_npz(args.dymn_npz, "context_pred")
    f_idx, f_labels, fused_ctx = load_npz(args.fused_npz, "context_pred")
    if not (np.array_equal(p_idx, d_idx) and np.array_equal(p_idx, f_idx) and labels == d_labels == f_labels):
        raise ValueError("OOF prediction idx/label mismatch")
    idx = p_idx
    y_all = make_targets(rows, labels)
    no_train, nonaves = load_scope_sets(args.data_root)
    no_train_mask = np.asarray([lab in no_train for lab in labels], dtype=bool)
    nonaves_mask = np.asarray([lab in nonaves for lab in labels], dtype=bool)

    dymn_filemax = file_broadcast(rows, idx, dymn_ctx, "max")
    dymn_filemean = file_broadcast(rows, idx, dymn_ctx, "mean")
    fused_filemax = file_broadcast(rows, idx, fused_ctx, "max")
    panns_filemax = file_broadcast(rows, idx, panns_row, "max")

    candidates: dict[str, np.ndarray] = {
        "panns_rowonly": panns_row,
        "dymn_context": dymn_ctx,
        "fused_context": fused_ctx,
        "panns_filemax_self": panns_filemax,
        "dymn_filemax_self": dymn_filemax,
        "fused_filemax_self": fused_filemax,
    }
    # File-level calibration/mapping candidates: keep PANNs row rank but inject
    # DyMN10/fused file evidence in logit space.
    for src_name, file_ev in [("dymn_filemax", dymn_filemax), ("dymn_filemean", dymn_filemean), ("fused_filemax", fused_filemax), ("panns_filemax", panns_filemax)]:
        for a in (0.05, 0.10, 0.20, 0.35, 0.50):
            candidates[f"pannsrow__{src_name}__a{int(a*100):02d}"] = sigmoid((1 - a) * logit(panns_row) + a * logit(file_ev)).astype(np.float32)
    # Cross-family row/file blends for comparison.
    candidates["rowblend_panns50_dymn25_fused25"] = sigmoid(0.50 * logit(panns_row) + 0.25 * logit(dymn_ctx) + 0.25 * logit(fused_ctx)).astype(np.float32)
    candidates["rowfile_panns60_dymnfile25_fused15"] = sigmoid(0.60 * logit(panns_row) + 0.25 * logit(dymn_filemax) + 0.15 * logit(fused_ctx)).astype(np.float32)
    candidates["rowfile_panns60_fusedfile25_dymn15"] = sigmoid(0.60 * logit(panns_row) + 0.25 * logit(fused_filemax) + 0.15 * logit(dymn_ctx)).astype(np.float32)

    metrics: list[dict[str, Any]] = []
    for name, pred in candidates.items():
        row_auc, row_valid, row_folds = leave_site_auc_mean(rows, idx, y_all, pred)
        file_auc, file_valid, file_folds = leave_site_auc_mean(rows, idx, y_all, pred, file_mil=True)
        nt_auc, nt_valid, nt_folds = leave_site_auc_mean(rows, idx, y_all, pred, label_mask=no_train_mask)
        na_auc, na_valid, na_folds = leave_site_auc_mean(rows, idx, y_all, pred, label_mask=nonaves_mask)
        pooled_row_auc, pooled_row_valid = macro_auc(y_all[idx], pred)
        metrics.append({
            "candidate": name,
            "row_auc": row_auc,
            "row_valid_classes": row_valid,
            "file_mil_auc": file_auc,
            "file_valid_classes": file_valid,
            "no_train_auc": nt_auc,
            "no_train_valid_classes": nt_valid,
            "nonaves_auc": na_auc,
            "nonaves_valid_classes": na_valid,
            "pooled_row_auc": pooled_row_auc,
            "pooled_row_valid_classes": pooled_row_valid,
            "row_folds": row_folds,
            "file_mil_folds": file_folds,
            "no_train_folds": nt_folds,
            "nonaves_folds": na_folds,
            "mean": float(pred.mean()),
            "std": float(pred.std()),
            "nonconstant_labels": int(((pred.max(axis=0) - pred.min(axis=0)) > 1e-8).sum()),
        })

    # Rank by row AUC first with file-MIL as tie/information term; include the best file-MIL candidate too.
    ranked = sorted(metrics, key=lambda m: (float(m["row_auc"] or -999), float(m["file_mil_auc"] or -999)), reverse=True)
    ranked_file = sorted(metrics, key=lambda m: (float(m["file_mil_auc"] or -999), float(m["row_auc"] or -999)), reverse=True)
    selected_names: list[str] = []
    for m in ranked[:4] + ranked_file[:2]:
        if m["candidate"] not in selected_names and not m["candidate"].endswith("self"):
            selected_names.append(m["candidate"])
    selected_names = selected_names[:5]

    sidecar_infos: list[dict[str, Any]] = []
    sidecar_build: list[dict[str, Any]] = []
    for name in selected_names:
        member = safe_member_name(name)
        out_csv = out / "sidecars" / f"{member}_sidecar_234.csv"
        info = write_sidecar(base_csv=args.anchor_csv, rows=rows, idx=idx, labels=labels, preds=candidates[name], out_csv=out_csv)
        metric = next(m for m in metrics if m["candidate"] == name)
        info["candidate"] = name
        info["member"] = member
        info["metric"] = metric
        sidecar_build.append(info)
        sidecar_infos.append({"member": member, "path": str(out_csv), "description": f"File-level calibration diagnostic candidate {name}", "metric": metric})

    manifest = make_manifest(args, sidecar_infos, out)
    audit_dir = out / "audit"
    cmd = [args.python, "scripts/birdclef_ensemble_strategy_audit.py", "--manifest", str(manifest), "--output-dir", str(audit_dir), "--bootstrap-iters", str(args.bootstrap_iters), "--emit-candidate-csvs"]
    result = subprocess.run(cmd, text=True, capture_output=True)
    (out / "audit_command.txt").write_text(" ".join(cmd) + "\n")
    (out / "audit_stdout.txt").write_text(result.stdout)
    (out / "audit_stderr.txt").write_text(result.stderr)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    audit_summary = summarize_audit(audit_dir / "ensemble_strategy_audit.json", out / "audit_summary.json")

    summary = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "experiment_id": "file-level-calibration-mapping-diagnostic-20260601T0215Z",
        "data": {"windows": len(rows), "val_windows": int(len(idx)), "files": int(len({r['filename'] for r in rows})), "sites": int(len({r['site'] for r in rows})), "labels": len(labels)},
        "selected_sidecars": selected_names,
        "top_by_row_auc": ranked[:10],
        "top_by_file_mil_auc": ranked_file[:10],
        "sidecar_build": sidecar_build,
        "audit_summary": audit_summary,
        "decision": "reject_submission_grade_unless_audit_beats_v616_and_gates",
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output_dir": str(out), "top_by_row_auc": ranked[:5], "top_sidecar_audit": audit_summary["top_by_lift_vs_v616"][:5]}, indent=2))


if __name__ == "__main__":
    main()
