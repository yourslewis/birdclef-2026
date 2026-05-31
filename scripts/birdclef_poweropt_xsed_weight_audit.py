#!/usr/bin/env python3
"""Audit EoS8/PowerOptimization Proto-vs-SED xSED weights on train-soundscape proxy.

Uses the v644/v647 source-winner intermediate train-soundscape streams already
materialized by the public-session audit, then reproduces the lightweight xSED
rank-blend/postprocessing/final taxonomy-smoothing path for configurable
Proto/SED weights. This is a no-submit verifier for v651/v652 style source forks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from birdclef_public946_cache_summary import load_long_labels, topk_row_recall  # noqa: E402

EPS = 1e-5
MIRROR_PAIRS = (
    ("47158son15", "47158son16"),
    ("47158son09", "47158son12"),
    ("47158son02", "47158son14"),
    ("47158son13", "47158son21", "47158son22", "47158son23"),
)


def pred_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "row_id"]


def file_id(row_id: str) -> str:
    return "_".join(str(row_id).split("_")[:-1])


def site_id(row_id: str) -> str:
    parts = str(row_id).split("_")
    return parts[3] if len(parts) > 3 else "UNKNOWN"


def align_like(df: pd.DataFrame, row_ids: pd.Series, cols: list[str], name: str) -> pd.DataFrame:
    if "row_id" not in df.columns:
        raise ValueError(f"{name} missing row_id")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing[:8]}")
    if set(df["row_id"].astype(str)) != set(row_ids.astype(str)):
        raise ValueError(f"{name} row_id set differs from canonical")
    return df.set_index("row_id").loc[row_ids.astype(str).tolist()].reset_index()[["row_id", *cols]]


def rank_cols(values: np.ndarray) -> np.ndarray:
    return pd.DataFrame(np.clip(values, 1e-7, 1.0 - 1e-7)).rank(axis=0, pct=True).to_numpy(np.float32)


def valid_class_indices(y: np.ndarray) -> list[int]:
    return [j for j in range(y.shape[1]) if np.nanmin(y[:, j]) < np.nanmax(y[:, j])]


def macro_auc(y: np.ndarray, score: np.ndarray) -> tuple[float | None, int]:
    valid = valid_class_indices(y)
    if not valid:
        return None, 0
    return float(roc_auc_score(y[:, valid], score[:, valid], average="macro")), len(valid)


def subset_auc(y: np.ndarray, score: np.ndarray, indices: list[int]) -> tuple[float | None, int]:
    valid = [j for j in indices if j < y.shape[1] and np.nanmin(y[:, j]) < np.nanmax(y[:, j])]
    if not valid:
        return None, 0
    return float(roc_auc_score(y[:, valid], score[:, valid], average="macro")), len(valid)


def matrix_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values.astype(np.float32)).tobytes()).hexdigest()[:16]


def group_bootstrap_lift(y: np.ndarray, cand: np.ndarray, base: np.ndarray, groups: np.ndarray, *, n_boot: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(groups.astype(str))))
    lifts: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in sampled])
        auc_c, _ = macro_auc(y[idx], cand[idx])
        auc_b, _ = macro_auc(y[idx], base[idx])
        if auc_c is not None and auc_b is not None and np.isfinite(auc_c) and np.isfinite(auc_b):
            lifts.append(float(auc_c - auc_b))
    if not lifts:
        return {"n": 0}
    arr = np.asarray(lifts, dtype=np.float64)
    return {"n": int(arr.size), "mean": float(arr.mean()), "q05": float(np.quantile(arr, 0.05)), "q50": float(np.quantile(arr, 0.50)), "q95": float(np.quantile(arr, 0.95)), "min": float(arr.min()), "max": float(arr.max()), "p_gt_0": float(np.mean(arr > 0))}


def xsed_poweropt_df(proto: pd.DataFrame, sed: pd.DataFrame, cols: list[str], proto_w: float, taxonomy: pd.DataFrame) -> pd.DataFrame:
    sed = align_like(sed, proto["row_id"], cols, "sed")
    p_proto = np.clip(proto[cols].to_numpy(np.float32), EPS, 1.0 - EPS)
    p_sed = np.clip(sed[cols].to_numpy(np.float32), EPS, 1.0 - EPS)
    rank_proto = rank_cols(p_proto)
    rank_sed = rank_cols(p_sed)
    sed_w = 1.0 - float(proto_w)
    pred = (rank_proto * float(proto_w)) + (rank_sed * sed_w)

    row_ids = proto["row_id"].astype(str).to_numpy()
    file_ids = np.array([file_id(r) for r in row_ids])
    fake_only = (p_proto > 0.50) & (p_sed < 0.05)
    pred = np.where(fake_only, (1.0 - 0.08) * pred + 0.08 * rank_proto, pred)

    offs = np.arange(-3, 4, dtype=np.float32)
    proto_kernel = (1.0 + (offs / 1.20) ** 2 / 2.0) ** (-1.5)
    proto_kernel = (proto_kernel / proto_kernel.sum()).astype(np.float32)
    pa_ctx = p_proto.copy()
    for fid in pd.unique(file_ids):
        m = file_ids == fid
        x = p_proto[m]
        if len(x) > 1:
            xp = np.pad(x, ((3, 3), (0, 0)), mode="edge")
            pa_ctx[m] = sum(proto_kernel[i] * xp[i:i + len(x)] for i in range(7))
    xctx = rank_cols(pa_ctx)
    proto_cont = (xctx > 0.88) & (rank_proto > 0.75) & (p_sed < 0.12) & (~fake_only)
    pred = np.where(proto_cont, (1.0 - 0.15) * pred + 0.15 * np.maximum(rank_proto, xctx), pred)

    sed_only = (rank_sed > 0.95) & (rank_proto < 0.80) & (~fake_only) & (~proto_cont)
    pred = np.where(sed_only, (1.0 - 0.12) * pred + 0.12 * rank_sed, pred)

    sub = proto[["row_id"]].copy()
    for i, c in enumerate(cols):
        sub[c] = pred[:, i].astype(np.float32)

    col_to_idx = {c: i for i, c in enumerate(cols)}
    for group in MIRROR_PAIRS:
        valid_idx = [col_to_idx[s] for s in group if s in col_to_idx]
        if len(valid_idx) >= 2:
            group_max = sub[cols].iloc[:, valid_idx].max(axis=1).to_numpy(np.float32)
            for idx in valid_idx:
                sub.iloc[:, idx + 1] = group_max

    if not taxonomy.empty and "primary_label" in taxonomy and "class_name" in taxonomy:
        tax = taxonomy.set_index("primary_label")
        rare_classes = {"Amphibia", "Mammalia", "Reptilia"}
        for ci, species in enumerate(cols):
            if species in tax.index and str(tax.loc[species, "class_name"]) in rare_classes:
                vals = sub.iloc[:, ci + 1].to_numpy(np.float32)
                thr = vals.mean() + 0.05
                sub.iloc[:, ci + 1] = np.where(vals < thr, vals * 0.9, vals)
    return sub


def tax_smooth(df: pd.DataFrame, taxonomy: pd.DataFrame, genus_alpha: float = 0.15, class_alpha: float = 0.05) -> pd.DataFrame:
    if taxonomy.empty:
        return df
    cols = pred_cols(df)
    tax = taxonomy.copy()
    species_to_genus = {}
    species_to_class = {}
    for _, row in tax.iterrows():
        label = str(row.get("primary_label", ""))
        sci = str(row.get("scientific_name", ""))
        cls = str(row.get("class_name", ""))
        genus = sci.split(" ")[0] if " " in sci else sci
        if label:
            species_to_genus[label] = genus
            species_to_class[label] = cls
    genus_groups: dict[str, list[str]] = {}
    class_groups: dict[str, list[str]] = {}
    for col in cols:
        genus_groups.setdefault(species_to_genus.get(col, col), []).append(col)
        cls = species_to_class.get(col, "")
        if cls:
            class_groups.setdefault(cls, []).append(col)
    probs = df[cols].to_numpy(np.float32, copy=True)
    col_to_idx = {col: i for i, col in enumerate(cols)}
    for members in [v for v in genus_groups.values() if len(v) > 1]:
        idx = [col_to_idx[m] for m in members]
        mean = probs[:, idx].mean(axis=1, keepdims=True)
        probs[:, idx] = (1.0 - genus_alpha) * probs[:, idx] + genus_alpha * mean
    for members in [v for v in class_groups.values() if len(v) > 1]:
        idx = [col_to_idx[m] for m in members]
        mean = probs[:, idx].mean(axis=1, keepdims=True)
        probs[:, idx] = (1.0 - class_alpha) * probs[:, idx] + class_alpha * mean
    out = df[["row_id"]].copy()
    out[cols] = np.clip(probs, 0.0, 1.0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=Path, default=Path("artifacts/source_winner_audit_20260531T0416Z/session_outputs/v644_yaroslav_0950"))
    ap.add_argument("--v616-csv", type=Path, default=Path("artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission.csv"))
    ap.add_argument("--labels-csv", type=Path, default=Path("data/train_soundscapes_labels.csv"))
    ap.add_argument("--taxonomy-csv", type=Path, default=Path("data/taxonomy.csv"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--proto-weights", default="0.6,0.4,0.2")
    ap.add_argument("--bootstrap", type=int, default=200)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cand_dir = args.out_dir / "candidates"; cand_dir.mkdir(exist_ok=True)

    v616 = pd.read_csv(args.v616_csv)
    cols = pred_cols(v616)
    proto = align_like(pd.read_csv(args.source_dir / "submission_protossm.csv"), v616["row_id"], cols, "proto")
    sed = align_like(pd.read_csv(args.source_dir / "submission_sed.csv"), v616["row_id"], cols, "sed")
    taxonomy = pd.read_csv(args.taxonomy_csv) if args.taxonomy_csv.exists() else pd.DataFrame()
    # Public-session final/yukiZ files are sample-session sized, while proto/sed
    # intermediates include train-soundscape proxy rows.  For local comparison we
    # audit the PowerOptimization branch reconstructed from proto/sed only.

    labels_wide = load_long_labels(args.labels_csv, cols)
    matched = v616["row_id"].astype(str).isin(labels_wide["row_id"].astype(str))
    matched_idx = np.flatnonzero(matched.to_numpy())
    y = labels_wide.set_index("row_id").loc[v616["row_id"].astype(str).iloc[matched_idx].tolist(), cols].to_numpy(np.uint8)
    row_ids = v616["row_id"].astype(str)
    sites = row_ids.iloc[matched_idx].map(site_id).to_numpy()
    files = row_ids.iloc[matched_idx].map(file_id).to_numpy()

    non_aves_cols = taxonomy.loc[taxonomy["class_name"].astype(str) != "Aves", "primary_label"].astype(str).tolist() if not taxonomy.empty and "class_name" in taxonomy else []
    non_aves_idx = [cols.index(c) for c in non_aves_cols if c in cols]
    no_train_cols = [c for c in non_aves_cols if str(c).startswith("47158son") or c in {"517063", "1491113", "25073"}]
    no_train_idx = [cols.index(c) for c in no_train_cols if c in cols]

    base_values = v616[cols].to_numpy(np.float32)
    base_auc, valid_n = macro_auc(y, base_values[matched_idx])
    current_auc = None
    results: list[dict[str, Any]] = []
    for proto_w in [float(x) for x in args.proto_weights.split(",") if x.strip()]:
        power = xsed_poweropt_df(proto, sed, cols, proto_w, taxonomy)
        power_values = power[cols].to_numpy(np.float32)
        final = tax_smooth(power, taxonomy, 0.15, 0.05)
        tag = f"proto{int(round(proto_w*100)):03d}_sed{int(round((1-proto_w)*100)):03d}"
        out_csv = cand_dir / f"poweropt_xsed_{tag}_taxonomy.csv"
        final.to_csv(out_csv, index=False)
        values = final[cols].to_numpy(np.float32)
        auc, n = macro_auc(y, values[matched_idx])
        non_auc, non_n = subset_auc(y, values[matched_idx], non_aves_idx)
        nt_auc, nt_n = subset_auc(y, values[matched_idx], no_train_idx)
        result = {
            "candidate": tag,
            "proto_weight": proto_w,
            "sed_weight": 1.0 - proto_w,
            "csv": str(out_csv),
            "local_macro_auc": auc,
            "valid_class_count": n,
            "delta_vs_v616": None if auc is None or base_auc is None else float(auc - base_auc),
            "delta_vs_current_source_final": None if auc is None or current_auc is None else float(auc - current_auc),
            "non_aves_auc": non_auc,
            "non_aves_valid": non_n,
            "no_train_auc": nt_auc,
            "no_train_valid": nt_n,
            "site_boot_lift_vs_v616": group_bootstrap_lift(y, values[matched_idx], base_values[matched_idx], sites, n_boot=args.bootstrap, seed=651 + int(proto_w*1000)),
            "file_boot_lift_vs_v616": group_bootstrap_lift(y, values[matched_idx], base_values[matched_idx], files, n_boot=args.bootstrap, seed=1651 + int(proto_w*1000)),
            "rank_corr_vs_v616": float(np.corrcoef(rank_cols(values).ravel(), rank_cols(base_values).ravel())[0, 1]),
            "mae_vs_v616": float(np.mean(np.abs(values - base_values))),
            "matrix_hash": matrix_hash(values),
            "topk": {f"top{k}_row_recall": topk_row_recall(values[matched_idx], y, k) for k in (1,3,5,10)},
        }
        results.append(result)

    summary = {
        "experiment_id": "eos8-poweropt-xsed-weight-audit-20260531T1816Z",
        "data": {"proxy_rows": int(len(v616)), "label_matched_rows": int(len(matched_idx)), "files": int(len(set(files))), "sites": sorted(set(sites.astype(str))), "class_count": len(cols), "valid_auc_classes": valid_n},
        "baselines": {"v616_local_macro_auc": base_auc, "current_source_final_local_macro_auc": current_auc, "note": "local proxy audits reconstructed PowerOptimization proto/sed branch only because public final/yukiZ files are sample-session sized"},
        "results": results,
        "best_by_local_auc": max(results, key=lambda r: (-1 if r["local_macro_auc"] is None else r["local_macro_auc"])),
    }
    (args.out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    lines = ["# EoS8 PowerOpt xSED weight audit", "", f"Baseline v616 local AUC: {base_auc:.6f} / {valid_n} valid", "Local scope: reconstructed PowerOptimization proto/sed branch only; public final/yukiZ files are sample-session sized.", "", "```text", "candidate          local_auc  delta_v616  site_q05   file_q05   corr_v616  mae_v616"]
    for r in results:
        lines.append(f"{r['candidate']:<18} {r['local_macro_auc']:.6f}  {r['delta_vs_v616']:+.6f}  {r['site_boot_lift_vs_v616'].get('q05', float('nan')):+.6f}  {r['file_boot_lift_vs_v616'].get('q05', float('nan')):+.6f}  {r['rank_corr_vs_v616']:.6f}  {r['mae_vs_v616']:.6f}")
    lines.extend(["```", "", "Decision: comparison-grade local verifier only; hidden-safe source forks still require public-session runtime/schema preflight before any competition submission."])
    (args.out_dir / "audit_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, default=str)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
