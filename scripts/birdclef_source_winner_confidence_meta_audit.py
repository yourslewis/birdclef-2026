#!/usr/bin/env python3
"""Audit/train source-winner Proto/SED confidence sidecars on the v616 proxy.

This is a no-submit BirdCLEF workbench for the v644/v647 EoS8 family.  It uses
source-winner intermediate train-soundscape streams (`submission_protossm.csv`,
`submission_sed.csv`) as hidden-safe-capable signals, evaluates simple rank
blends against the submitted v616 proxy, and trains a tiny leave-site logistic
meta calibrator as a measured data point.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.append(str(Path(__file__).resolve().parent))
from birdclef_public946_cache_summary import load_long_labels, public946_rankblend, topk_row_recall  # noqa: E402


def pred_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "row_id"]


def file_id(row_id: str) -> str:
    return "_".join(str(row_id).split("_")[:-1])


def site_id(row_id: str) -> str:
    parts = str(row_id).split("_")
    return parts[3] if len(parts) > 3 else "UNKNOWN"


def rank_cols(values: np.ndarray) -> np.ndarray:
    return pd.DataFrame(np.clip(values, 1e-7, 1.0 - 1e-7)).rank(axis=0, pct=True).to_numpy(np.float32)


def logit(values: np.ndarray) -> np.ndarray:
    x = np.clip(values, 1e-5, 1.0 - 1e-5)
    return np.log(x / (1.0 - x))


def matrix_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values.astype(np.float32)).tobytes()).hexdigest()[:16]


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


def topk_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in (1, 3, 5, 10):
        out[f"top{k}_row_recall"] = topk_row_recall(score, y, k)
    return out


def group_bootstrap_lift(
    y: np.ndarray,
    cand: np.ndarray,
    base: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
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
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "q05": float(np.quantile(arr, 0.05)),
        "q50": float(np.quantile(arr, 0.50)),
        "q95": float(np.quantile(arr, 0.95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p_gt_0": float(np.mean(arr > 0)),
    }


def align_like(df: pd.DataFrame, row_ids: pd.Series, cols: list[str], name: str) -> pd.DataFrame:
    if "row_id" not in df.columns:
        raise ValueError(f"{name} missing row_id")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing[:8]}")
    if set(df["row_id"].astype(str)) != set(row_ids.astype(str)):
        raise ValueError(f"{name} row_id set differs from canonical")
    return df.set_index("row_id").loc[row_ids.astype(str).tolist()].reset_index()[["row_id", *cols]]


def train_losite_meta(
    y_all: np.ndarray,
    matched_idx: np.ndarray,
    sites_full: np.ndarray,
    streams: dict[str, np.ndarray],
    ranks: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    n_match, n_classes = y_all.shape
    oof = np.zeros((n_match, n_classes), dtype=np.float32)
    sites = sites_full[matched_idx]
    unique_sites = np.array(sorted(set(sites.astype(str))))
    selected_cells = 0
    fallback_cells = 0
    fitted_models = 0
    fallback_classes = set()

    row_max_proto = streams["source_proto"].max(axis=1)
    row_max_sed = streams["source_sed"].max(axis=1)
    row_max_v616 = streams["v616_final"].max(axis=1)

    for j in range(n_classes):
        x_full = np.column_stack([
            logit(streams["v616_final"][:, j]),
            logit(streams["source_proto"][:, j]),
            logit(streams["source_sed"][:, j]),
            logit(streams["source_rankblend"][:, j]),
            ranks["v616_final"][:, j],
            ranks["source_proto"][:, j],
            ranks["source_sed"][:, j],
            ranks["source_rankblend"][:, j],
            np.maximum(ranks["source_proto"][:, j], ranks["source_sed"][:, j]),
            np.abs(ranks["source_proto"][:, j] - ranks["source_sed"][:, j]),
            row_max_proto,
            row_max_sed,
            row_max_v616,
        ]).astype(np.float32)
        y = y_all[:, j]
        for site in unique_sites:
            va = sites == site
            tr = ~va
            pos = int(y[tr].sum())
            neg = int(tr.sum() - pos)
            if y[tr].min() == y[tr].max() or pos < 2 or neg < 8:
                oof[va, j] = streams["source_rankblend"][matched_idx[va], j]
                fallback_cells += int(va.sum())
                fallback_classes.add(j)
                continue
            try:
                clf = LogisticRegression(C=0.15, class_weight="balanced", solver="liblinear", max_iter=200)
                clf.fit(x_full[matched_idx[tr]], y[tr])
                oof[va, j] = clf.predict_proba(x_full[matched_idx[va]])[:, 1]
                selected_cells += int(va.sum())
                fitted_models += 1
            except Exception:
                oof[va, j] = streams["source_rankblend"][matched_idx[va], j]
                fallback_cells += int(va.sum())
                fallback_classes.add(j)
    return oof, {
        "model": "per-class LogisticRegression(C=0.15, class_weight=balanced, liblinear)",
        "features": [
            "logit(v616)", "logit(proto)", "logit(sed)", "logit(rankblend)",
            "rank(v616)", "rank(proto)", "rank(sed)", "rank(rankblend)",
            "max_rank(proto,sed)", "abs_rank_diff(proto,sed)",
            "row_max_proto", "row_max_sed", "row_max_v616",
        ],
        "split": "leave-one-site over matched train-soundscape proxy rows",
        "fitted_site_class_models": fitted_models,
        "selected_cells": selected_cells,
        "fallback_cells": fallback_cells,
        "fallback_classes": len(fallback_classes),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto-csv", type=Path, required=True)
    ap.add_argument("--sed-csv", type=Path, required=True)
    ap.add_argument("--v616-csv", type=Path, required=True)
    ap.add_argument("--anchor-csv", type=Path, required=True)
    ap.add_argument("--labels-csv", type=Path, required=True)
    ap.add_argument("--taxonomy-csv", type=Path, default=Path("data/taxonomy.csv"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--weights", default="0.001,0.0025,0.005,0.01,0.02,0.04,0.08,0.12,0.16,0.20,0.24,0.28,0.32,0.36,0.40,0.50,0.65,0.80,1.00")
    ap.add_argument("--bootstrap", type=int, default=200)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cand_dir = args.out_dir / "candidates"
    cand_dir.mkdir(exist_ok=True)

    v616 = pd.read_csv(args.v616_csv)
    anchor = align_like(pd.read_csv(args.anchor_csv), v616["row_id"], pred_cols(v616), "anchor")
    proto = align_like(pd.read_csv(args.proto_csv), v616["row_id"], pred_cols(v616), "proto")
    sed = align_like(pd.read_csv(args.sed_csv), v616["row_id"], pred_cols(v616), "sed")
    rankblend = public946_rankblend(proto, sed)
    rankblend = align_like(rankblend, v616["row_id"], pred_cols(v616), "rankblend")

    cols = pred_cols(v616)
    row_ids = v616["row_id"].astype(str)
    sites_full = row_ids.map(site_id).to_numpy()
    files_full = row_ids.map(file_id).to_numpy()

    labels_wide = load_long_labels(args.labels_csv, cols)
    matched = v616["row_id"].astype(str).isin(labels_wide["row_id"].astype(str))
    matched_idx = np.flatnonzero(matched.to_numpy())
    matched_row_ids = row_ids.iloc[matched_idx].tolist()
    y = labels_wide.set_index("row_id").loc[matched_row_ids, cols].to_numpy(np.uint8)
    valid = valid_class_indices(y)

    taxonomy = pd.read_csv(args.taxonomy_csv) if args.taxonomy_csv.exists() else pd.DataFrame()
    non_aves_cols: list[str] = []
    if not taxonomy.empty and "primary_label" in taxonomy and "class_name" in taxonomy:
        non_aves_cols = taxonomy.loc[taxonomy["class_name"].astype(str) != "Aves", "primary_label"].astype(str).tolist()
    non_aves_idx = [cols.index(c) for c in non_aves_cols if c in cols]

    streams_df = {
        "anchor_raw": anchor,
        "v616_final": v616,
        "source_proto": proto,
        "source_sed": sed,
        "source_rankblend": rankblend,
    }
    streams = {name: df[cols].to_numpy(np.float32) for name, df in streams_df.items()}
    ranks = {name: rank_cols(values) for name, values in streams.items()}

    baseline_auc, valid_n = macro_auc(y, streams["v616_final"][matched_idx])
    anchor_auc, _ = macro_auc(y, streams["anchor_raw"][matched_idx])

    stream_metrics: dict[str, Any] = {}
    for name, values in streams.items():
        auc, n = macro_auc(y, values[matched_idx])
        non_aves_auc, non_aves_n = subset_auc(y, values[matched_idx], non_aves_idx)
        stream_metrics[name] = {
            "local_macro_auc": auc,
            "valid_class_count": n,
            "delta_vs_v616": None if auc is None or baseline_auc is None else float(auc - baseline_auc),
            "delta_vs_anchor": None if auc is None or anchor_auc is None else float(auc - anchor_auc),
            "non_aves_auc": non_aves_auc,
            "non_aves_valid_class_count": non_aves_n,
            "topk": topk_metrics(y, values[matched_idx]),
            "matrix_hash": matrix_hash(values),
        }

    meta_oof, meta_info = train_losite_meta(y, matched_idx, sites_full, streams, ranks)
    meta_auc, meta_valid = macro_auc(y, meta_oof)
    meta_nonaves_auc, meta_nonaves_valid = subset_auc(y, meta_oof, non_aves_idx)
    meta_full = streams["source_rankblend"].copy()
    meta_full[matched_idx] = meta_oof
    meta_df = v616[["row_id"]].copy()
    meta_df[cols] = meta_full.astype(np.float32)
    meta_path = cand_dir / "source_winner_losite_meta_oof.csv"
    meta_df.to_csv(meta_path, index=False)

    weights = [float(x) for x in args.weights.replace(" ", "").split(",") if x]
    base_rank = ranks["v616_final"]
    sidecar_results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    sidecar_sources = {
        "source_sed": ranks["source_sed"],
        "source_proto": ranks["source_proto"],
        "source_rankblend": ranks["source_rankblend"],
        "losite_meta_oof": rank_cols(meta_full),
    }
    for source_name, source_rank in sidecar_sources.items():
        for w in weights:
            cand = ((1.0 - w) * base_rank + w * source_rank).astype(np.float32)
            auc, n = macro_auc(y, cand[matched_idx])
            lift = None if auc is None or baseline_auc is None else float(auc - baseline_auc)
            rank_corr = float(np.corrcoef(base_rank.ravel(), cand.ravel())[0, 1])
            mae = float(np.mean(np.abs(base_rank - cand)))
            result = {
                "source": source_name,
                "weight": w,
                "local_macro_auc": auc,
                "valid_class_count": n,
                "lift_vs_v616": lift,
                "lift_vs_anchor": None if auc is None or anchor_auc is None else float(auc - anchor_auc),
                "rank_corr_vs_v616": rank_corr,
                "mae_vs_v616": mae,
                "topk": topk_metrics(y, cand[matched_idx]),
                "site_boot_lift": group_bootstrap_lift(y, cand[matched_idx], base_rank[matched_idx], sites_full[matched_idx], n_boot=args.bootstrap, seed=17),
                "file_boot_lift": group_bootstrap_lift(y, cand[matched_idx], base_rank[matched_idx], files_full[matched_idx], n_boot=args.bootstrap, seed=29),
                "matrix_hash": matrix_hash(cand),
            }
            safe_source = source_name.replace("_", "-")
            safe_w = str(w).replace(".", "p")
            cand_path = cand_dir / f"v616_rankblend_{safe_source}_w{safe_w}.csv"
            if source_name == "source_sed" or (best is None and source_name == "losite_meta_oof"):
                out = v616[["row_id"]].copy()
                out[cols] = cand.astype(np.float32)
                out.to_csv(cand_path, index=False)
                result["candidate_csv"] = str(cand_path)
            sidecar_results.append(result)
            if lift is not None and (best is None or lift > best.get("lift_vs_v616", -math.inf)):
                best = result

    # Always materialize the best candidate even when it was not one of the default saved streams.
    if best is not None and "candidate_csv" not in best:
        source_rank = sidecar_sources[best["source"]]
        cand = ((1.0 - best["weight"]) * base_rank + best["weight"] * source_rank).astype(np.float32)
        safe_source = str(best["source"]).replace("_", "-")
        safe_w = str(best["weight"]).replace(".", "p")
        cand_path = cand_dir / f"best_v616_rankblend_{safe_source}_w{safe_w}.csv"
        out = v616[["row_id"]].copy()
        out[cols] = cand.astype(np.float32)
        out.to_csv(cand_path, index=False)
        best["candidate_csv"] = str(cand_path)

    submit_approved = False
    reject_reasons = [
        "no hidden-test package/kernel variant was built in this audit",
        "signal is source-winner/v616-family and local proxy is known to over-promote SED-like blends",
    ]
    if best and best.get("lift_vs_v616") is not None and best["lift_vs_v616"] < 0.001:
        reject_reasons.append("best lift vs v616 below +0.001 promotion threshold")
    if best and best.get("site_boot_lift", {}).get("q05", -1.0) < 0:
        reject_reasons.append("site bootstrap q05 lift is negative")

    summary: dict[str, Any] = {
        "experiment_id": "source-winner-protosed-confidence-meta-audit-20260531",
        "evidence_level": "comparison-grade",
        "submit_approved": submit_approved,
        "reject_reasons": reject_reasons,
        "inputs": {
            "proto_csv": str(args.proto_csv),
            "sed_csv": str(args.sed_csv),
            "v616_csv": str(args.v616_csv),
            "anchor_csv": str(args.anchor_csv),
            "labels_csv": str(args.labels_csv),
            "taxonomy_csv": str(args.taxonomy_csv),
        },
        "data": {
            "proxy_rows": int(len(v616)),
            "label_matched_rows": int(len(matched_idx)),
            "files": int(len(set(files_full))),
            "matched_files": int(len(set(files_full[matched_idx]))),
            "sites": int(len(set(sites_full))),
            "matched_sites": sorted(set(sites_full[matched_idx].astype(str))),
            "class_count": int(len(cols)),
            "valid_auc_classes": int(valid_n),
            "non_aves_class_count": int(len(non_aves_idx)),
        },
        "stream_metrics": stream_metrics,
        "losite_meta": {
            **meta_info,
            "local_macro_auc": meta_auc,
            "valid_class_count": meta_valid,
            "delta_vs_v616": None if meta_auc is None or baseline_auc is None else float(meta_auc - baseline_auc),
            "delta_vs_anchor": None if meta_auc is None or anchor_auc is None else float(meta_auc - anchor_auc),
            "non_aves_auc": meta_nonaves_auc,
            "non_aves_valid_class_count": meta_nonaves_valid,
            "topk": topk_metrics(y, meta_oof),
            "oof_csv": str(meta_path),
        },
        "sidecar_grid": sidecar_results,
        "best_sidecar": best,
    }
    (args.out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    # Compact markdown for reviewers.
    top = sorted(sidecar_results, key=lambda r: (r.get("lift_vs_v616") if r.get("lift_vs_v616") is not None else -999), reverse=True)[:10]
    md = [
        "# Source-winner Proto/SED confidence meta audit — 2026-05-31\n",
        "## Scope\n",
        "Audits v644/v647 EoS8 source-winner intermediate ProtoSSM/SED train-soundscape streams against the v616 local proxy; trains a tiny leave-site logistic meta calibrator; no Kaggle submission.\n",
        "## Data\n",
        f"- Proxy rows: {len(v616)}; label-matched rows: {len(matched_idx)}; matched files/sites: {len(set(files_full[matched_idx]))}/{len(set(sites_full[matched_idx]))}; labels: {len(cols)} with {valid_n} valid local AUC classes.\n",
        "## Stream metrics\n",
        "```text\nstream             local_auc   lift_v616   nonaves_auc  top5_recall\n",
    ]
    for name, m in stream_metrics.items():
        md.append(f"{name[:17]:17s} {m['local_macro_auc'] if m['local_macro_auc'] is not None else float('nan'):.6f}  {m['delta_vs_v616'] if m['delta_vs_v616'] is not None else float('nan'):+.6f}  {m['non_aves_auc'] if m['non_aves_auc'] is not None else float('nan'):.6f}  {m['topk']['top5_row_recall']:.6f}\n")
    md.extend([
        "```\n",
        "## Leave-site meta\n",
        f"- Meta OOF AUC: {meta_auc:.6f} / {meta_valid} valid; lift vs v616: {meta_auc - baseline_auc:+.6f}; fitted site-class models: {meta_info['fitted_site_class_models']}; fallback cells: {meta_info['fallback_cells']}.\n",
        "## Top sidecar grid results\n",
        "```text\nsource            weight   local_auc   lift_v616   site_q05    file_q05    top5\n",
    ])
    for r in top:
        md.append(f"{r['source'][:16]:16s} {r['weight']:6.4f}  {r['local_macro_auc']:.6f}  {r['lift_vs_v616']:+.6f}  {r['site_boot_lift'].get('q05', float('nan')):+.6f}  {r['file_boot_lift'].get('q05', float('nan')):+.6f}  {r['topk']['top5_row_recall']:.6f}\n")
    md.extend([
        "```\n",
        "## Decision\n",
        "Comparison-grade only; submit_approved=false. The raw source SED stream is locally strong, but this audit did not build a hidden-test package and v616-family SED/local-proxy gains are known to over-transfer. Next useful action is a private kernel verifier or a source-code fork that changes EoS8 SED/PowerOpt weights, not a direct competition slot from this CSV.\n",
    ])
    (args.out_dir / "audit_report.md").write_text("".join(md))
    print(json.dumps({
        "summary": str(args.out_dir / "audit_summary.json"),
        "report": str(args.out_dir / "audit_report.md"),
        "best_sidecar": best,
        "meta_auc": meta_auc,
        "submit_approved": submit_approved,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
