#!/usr/bin/env python3
"""Gate a public946 anchor with multiple rank sidecars before submission.

This extends birdclef_public946_sidecar_weight_grid.py from one side stream to a
small Cartesian grid of named sidecars.  It is intended for no-submit dry-run
triage: if a candidate does not improve the public946 train-soundscape overlap
by a meaningful amount, do not spend a daily Kaggle code-submission slot.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from birdclef_public946_cache_summary import load_long_labels, topk_row_recall


def parse_named_path(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"expected NAME=PATH, got {text!r}")
    name, path = text.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"empty sidecar name in {text!r}")
    return name, Path(path.strip())


def parse_named_weights(text: str) -> tuple[str, list[float]]:
    if "=" not in text:
        raise ValueError(f"expected NAME=w1,w2,..., got {text!r}")
    name, raw = text.split("=", 1)
    weights = [float(x) for x in raw.replace(" ", "").split(",") if x]
    if not weights:
        raise ValueError(f"no weights for {name!r}")
    if any(w < 0 or w > 1 for w in weights):
        raise ValueError(f"weights must be in [0,1] for {name!r}: {weights}")
    return name.strip(), weights


def prediction_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c != "row_id"]
    if not cols:
        raise ValueError("prediction CSV has no class columns")
    return cols


def load_sidecar(path: Path, base: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    sidecar = pd.read_csv(path)
    if "row_id" not in sidecar:
        raise ValueError(f"{path} missing row_id")
    missing = [c for c in cols if c not in sidecar.columns]
    if missing:
        raise ValueError(f"{path} missing {len(missing)} base columns; first={missing[:5]}")
    return sidecar.set_index("row_id").loc[base["row_id"]].reset_index()[["row_id", *cols]]


def rank_values(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    values = np.clip(df[cols].to_numpy(np.float32), 1e-7, 1.0 - 1e-7)
    return pd.DataFrame(values).rank(axis=0, pct=True).to_numpy(np.float32)


def summarize(
    name: str,
    row_ids: pd.Series,
    cols: list[str],
    values: np.ndarray,
    labels_wide: pd.DataFrame | None,
    anchor_values: np.ndarray,
    weights: dict[str, float],
) -> dict[str, Any]:
    df = pd.DataFrame(values.astype(np.float32), columns=cols)
    df.insert(0, "row_id", row_ids.to_numpy())
    info: dict[str, Any] = {
        "name": name,
        "weights": weights,
        "total_sidecar_weight": float(sum(weights.values())),
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
    ap.add_argument("--base-csv", type=Path, required=True)
    ap.add_argument("--sidecar", action="append", required=True, help="Repeat NAME=PATH")
    ap.add_argument("--weights", action="append", required=True, help="Repeat NAME=w1,w2,...")
    ap.add_argument("--labels-csv", type=Path)
    ap.add_argument("--max-total-weight", type=float, default=0.16)
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args()

    named_paths = dict(parse_named_path(x) for x in args.sidecar)
    named_weights = dict(parse_named_weights(x) for x in args.weights)
    if set(named_paths) != set(named_weights):
        raise ValueError(f"sidecar names and weight names differ: paths={sorted(named_paths)} weights={sorted(named_weights)}")

    base = pd.read_csv(args.base_csv)
    if "row_id" not in base:
        raise ValueError("base CSV missing row_id")
    cols = prediction_columns(base)
    rank_base = rank_values(base, cols)
    sidecar_ranks = {name: rank_values(load_sidecar(path, base, cols), cols) for name, path in named_paths.items()}
    labels_wide = load_long_labels(args.labels_csv, cols) if args.labels_csv else None

    summaries: list[dict[str, Any]] = [
        summarize("base", base["row_id"], cols, rank_base, labels_wide, rank_base, {name: 0.0 for name in named_paths})
    ]
    names = list(named_paths)
    for combo in itertools.product(*(named_weights[name] for name in names)):
        weights = dict(zip(names, (float(x) for x in combo)))
        total = sum(weights.values())
        if total <= 0 or total > args.max_total_weight:
            continue
        pred = (1.0 - total) * rank_base
        for name, weight in weights.items():
            pred = pred + weight * sidecar_ranks[name]
        label = "_".join(f"{name}_{weight:.4f}" for name, weight in weights.items())
        summaries.append(summarize(label, base["row_id"], cols, pred, labels_wide, rank_base, weights))

    result = {
        "base_csv": str(args.base_csv),
        "sidecars": {name: str(path) for name, path in named_paths.items()},
        "weights": named_weights,
        "max_total_weight": args.max_total_weight,
        "labels_csv": str(args.labels_csv) if args.labels_csv else None,
        "summaries": sorted(summaries, key=lambda x: x.get("macro_auc", float("-inf")), reverse=True),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
