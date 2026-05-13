#!/usr/bin/env python3
"""Gate a public946 + sidecar rank blend before spending a Kaggle slot.

The public946 anchor is now strong enough that a new side stream should clear a
small offline sanity gate before it consumes a daily submission.  This diagnostic
loads an anchor submission and one candidate sidecar submission, row-aligns them,
rank-blends several sidecar weights, and reports label-overlap AUC plus
correlation/MAE versus the anchor.
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


def _parse_weights(text: str) -> list[float]:
    weights: list[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        w = float(part)
        if not 0.0 <= w <= 1.0:
            raise ValueError(f"sidecar weight must be in [0,1], got {w}")
        weights.append(w)
    if not weights:
        raise ValueError("at least one sidecar weight is required")
    return weights


def _prediction_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c != "row_id"]
    if not cols:
        raise ValueError("prediction CSV has no class columns")
    return cols


def _load_and_align(base_csv: Path, sidecar_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    base = pd.read_csv(base_csv)
    sidecar = pd.read_csv(sidecar_csv)
    if "row_id" not in base or "row_id" not in sidecar:
        raise ValueError("both base and sidecar CSVs must contain row_id")
    cols = _prediction_columns(base)
    missing = [c for c in cols if c not in sidecar.columns]
    if missing:
        raise ValueError(f"sidecar is missing {len(missing)} base columns; first={missing[:5]}")
    sidecar = sidecar.set_index("row_id").loc[base["row_id"]].reset_index()
    return base, sidecar[["row_id", *cols]], cols


def _rank_values(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    values = np.clip(df[cols].to_numpy(np.float32), 1e-7, 1.0 - 1e-7)
    return pd.DataFrame(values).rank(axis=0, pct=True).to_numpy(np.float32)


def _blend_df(base: pd.DataFrame, cols: list[str], rank_base: np.ndarray, rank_sidecar: np.ndarray, weight: float) -> pd.DataFrame:
    pred = (1.0 - weight) * rank_base + weight * rank_sidecar
    out = base[["row_id"]].copy()
    out[cols] = pred.astype(np.float32)
    return out


def _summarize(
    name: str,
    df: pd.DataFrame,
    cols: list[str],
    labels_wide: pd.DataFrame | None,
    anchor_values: np.ndarray,
) -> dict[str, Any]:
    values = df[cols].to_numpy(np.float32)
    info: dict[str, Any] = {
        "name": name,
        "rows": int(len(df)),
        "prob_stats": {"min": float(values.min()), "max": float(values.max()), "mean": float(values.mean())},
        "corr_vs_anchor": float(np.corrcoef(anchor_values.ravel(), values.ravel())[0, 1]),
        "mae_vs_anchor": float(np.mean(np.abs(anchor_values - values))),
        "max_abs_vs_anchor": float(np.max(np.abs(anchor_values - values))),
    }
    if labels_wide is not None:
        merged = df.merge(labels_wide, on="row_id", suffixes=("_pred", "_true"))
        valid = [c for c in cols if f"{c}_pred" in merged and f"{c}_true" in merged and merged[f"{c}_true"].nunique() > 1]
        info["matched_rows"] = int(len(merged))
        info["valid_auc_classes"] = int(len(valid))
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
    ap.add_argument("--base-csv", type=Path, required=True, help="Anchor public946 submission.csv")
    ap.add_argument("--sidecar-csv", type=Path, required=True, help="Candidate sidecar submission CSV")
    ap.add_argument("--labels-csv", type=Path, help="Optional train_soundscapes_labels.csv for dry-run overlap metrics")
    ap.add_argument("--weights", default="0,0.01,0.02,0.03,0.05,0.075,0.10", help="Comma-separated sidecar weights")
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args()

    weights = _parse_weights(args.weights)
    base, sidecar, cols = _load_and_align(args.base_csv, args.sidecar_csv)
    rank_base = _rank_values(base, cols)
    rank_sidecar = _rank_values(sidecar, cols)
    anchor_values = rank_base.astype(np.float32)
    labels_wide = load_long_labels(args.labels_csv, cols) if args.labels_csv else None

    variants: dict[str, pd.DataFrame] = {}
    for w in weights:
        variants[f"sidecar_{w:.4f}"] = _blend_df(base, cols, rank_base, rank_sidecar, w)

    summaries = [_summarize(name, df, cols, labels_wide, anchor_values) for name, df in variants.items()]
    sidecar_summary = _summarize("sidecar_rank_standalone", sidecar.assign(**{c: rank_sidecar[:, i] for i, c in enumerate(cols)}), cols, labels_wide, anchor_values)

    result: dict[str, Any] = {
        "base_csv": str(args.base_csv),
        "sidecar_csv": str(args.sidecar_csv),
        "labels_csv": str(args.labels_csv) if args.labels_csv else None,
        "weights": weights,
        "sidecar_standalone": sidecar_summary,
        "summaries": sorted(summaries, key=lambda x: x.get("macro_auc", float("-inf")), reverse=True),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
