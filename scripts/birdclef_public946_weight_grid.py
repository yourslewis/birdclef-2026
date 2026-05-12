#!/usr/bin/env python3
"""Evaluate public946 Proto/SED rank-blend weight variants on dry-run rows.

This is an offline diagnostic for open-solution mining.  It reconstructs the
public946 final rank-blend gates from `submission_protossm.csv` and
`submission_sed.csv`, optionally applies the sonotype mirror and rare-taxon
suppression gates, and reports label-overlap AUC/top-k metrics plus pairwise
correlations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from birdclef_public946_cache_summary import load_long_labels, topk_row_recall


MIRROR_PAIRS = (
    ("47158son15", "47158son16"),
    ("47158son09", "47158son12"),
    ("47158son02", "47158son14"),
    ("47158son13", "47158son21", "47158son22", "47158son23"),
)


def rank_blend_postprocess(
    proto: pd.DataFrame,
    sed: pd.DataFrame,
    proto_weight: float,
    taxonomy_csv: Path | None = None,
    apply_postprocess: bool = True,
) -> pd.DataFrame:
    """Reconstruct a public946-style rank blend for a chosen Proto weight."""
    cols = [c for c in proto.columns if c != "row_id"]
    sed = sed.set_index("row_id").loc[proto["row_id"]].reset_index()
    eps = 1e-5
    p_proto = np.clip(proto[cols].to_numpy(np.float32), eps, 1.0 - eps)
    p_sed = np.clip(sed[cols].to_numpy(np.float32), eps, 1.0 - eps)
    rank_proto = pd.DataFrame(p_proto).rank(axis=0, pct=True).to_numpy(np.float32)
    rank_sed = pd.DataFrame(p_sed).rank(axis=0, pct=True).to_numpy(np.float32)
    pred = rank_proto * float(proto_weight) + rank_sed * (1.0 - float(proto_weight))

    if apply_postprocess:
        row_ids = proto["row_id"].astype(str).to_numpy()
        file_ids = np.array(["_".join(r.split("_")[:-1]) for r in row_ids])
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
                pa_ctx[m] = sum(proto_kernel[i] * xp[i : i + len(x)] for i in range(7))
        xctx = pd.DataFrame(pa_ctx).rank(axis=0, pct=True).to_numpy(np.float32)
        proto_cont = (xctx > 0.88) & (rank_proto > 0.75) & (p_sed < 0.12) & (~fake_only)
        pred = np.where(proto_cont, (1.0 - 0.15) * pred + 0.15 * np.maximum(rank_proto, xctx), pred)

        sed_only = (rank_sed > 0.95) & (rank_proto < 0.80) & (~fake_only) & (~proto_cont)
        pred = np.where(sed_only, (1.0 - 0.12) * pred + 0.12 * rank_sed, pred)

    out = proto[["row_id"]].copy()
    out[cols] = pred.astype(np.float32)

    if apply_postprocess:
        col_to_idx = {label: i for i, label in enumerate(cols)}
        for group in MIRROR_PAIRS:
            valid_idx = [col_to_idx[s] for s in group if s in col_to_idx]
            if len(valid_idx) >= 2:
                group_max = out[cols].iloc[:, valid_idx].max(axis=1).to_numpy(np.float32)
                for idx in valid_idx:
                    out.iloc[:, idx + 1] = group_max

        if taxonomy_csv and taxonomy_csv.exists():
            tax_df = pd.read_csv(taxonomy_csv).set_index("primary_label")
            rare_classes = {"Amphibia", "Mammalia", "Reptilia"}
            for ci, species in enumerate(cols):
                if species in tax_df.index and tax_df.loc[species, "class_name"] in rare_classes:
                    col_idx = ci + 1
                    vals = out.iloc[:, col_idx].to_numpy(np.float32)
                    thr = vals.mean() + 0.05
                    out.iloc[:, col_idx] = np.where(vals < thr, vals * 0.9, vals)
    return out


def summarize(name: str, df: pd.DataFrame, labels_wide: pd.DataFrame) -> dict[str, Any]:
    cols = [c for c in df.columns if c != "row_id"]
    merged = df.merge(labels_wide, on="row_id", suffixes=("_pred", "_true"))
    valid = [c for c in cols if f"{c}_pred" in merged and f"{c}_true" in merged and merged[f"{c}_true"].nunique() > 1]
    info: dict[str, Any] = {
        "name": name,
        "rows": int(len(df)),
        "matched_rows": int(len(merged)),
        "valid_auc_classes": int(len(valid)),
    }
    values = df[cols].to_numpy(np.float32)
    info["prob_stats"] = {"min": float(values.min()), "max": float(values.max()), "mean": float(values.mean())}
    if valid:
        y_true = merged[[f"{c}_true" for c in valid]].to_numpy()
        y_score = merged[[f"{c}_pred" for c in valid]].to_numpy()
        info["macro_auc"] = float(roc_auc_score(y_true, y_score, average="macro"))
        pred_cols = [f"{c}_pred" for c in cols if f"{c}_pred" in merged]
        true_cols = [c.replace("_pred", "_true") for c in pred_cols]
        score_mat = merged[pred_cols].to_numpy()
        true_mat = merged[true_cols].to_numpy()
        for k in (1, 3, 5, 10):
            info[f"top{k}_row_recall"] = topk_row_recall(score_mat, true_mat, k)
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--labels-csv", type=Path, required=True)
    ap.add_argument("--taxonomy-csv", type=Path)
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args()

    proto = pd.read_csv(args.pred_dir / "submission_protossm.csv")
    sed = pd.read_csv(args.pred_dir / "submission_sed.csv")
    cols = [c for c in proto.columns if c != "row_id"]
    labels_wide = load_long_labels(args.labels_csv, cols)

    variants: dict[str, pd.DataFrame] = {}
    for w in (0.80, 0.70, 0.60, 0.54, 0.50, 0.46, 0.40):
        variants[f"proto{w:.2f}_sed{1-w:.2f}"] = rank_blend_postprocess(proto, sed, w, args.taxonomy_csv)
    # Nina Model_61/62 direct ensemble proxy: average of 54/46 and 46/54 after gates.
    variants["nina_m61_m62_direct_50_50_proxy"] = variants["proto0.54_sed0.46"].copy()
    pred_cols = [c for c in variants["nina_m61_m62_direct_50_50_proxy"].columns if c != "row_id"]
    variants["nina_m61_m62_direct_50_50_proxy"][pred_cols] = (
        variants["proto0.54_sed0.46"][pred_cols].to_numpy(np.float32)
        + variants["proto0.46_sed0.54"][pred_cols].to_numpy(np.float32)
    ) / 2.0

    summaries = [summarize(name, df, labels_wide) for name, df in variants.items()]

    # Pairwise correlations among variants on common rows.
    corr: dict[str, dict[str, float]] = {}
    for a, dfa in variants.items():
        corr[a] = {}
        aval = dfa[pred_cols].to_numpy(np.float32).ravel()
        for b, dfb in variants.items():
            bval = dfb[pred_cols].to_numpy(np.float32).ravel()
            corr[a][b] = float(np.corrcoef(aval, bval)[0, 1])

    result = {
        "pred_dir": str(args.pred_dir),
        "labels_csv": str(args.labels_csv),
        "taxonomy_csv": str(args.taxonomy_csv) if args.taxonomy_csv else None,
        "summaries": sorted(summaries, key=lambda x: x.get("macro_auc", float("nan")), reverse=True),
        "correlations": corr,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
