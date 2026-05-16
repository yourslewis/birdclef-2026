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


def macro_auc_matrix(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    vals = []
    for j in range(y_true.shape[1]):
        col = y_true[:, j]
        if col.min() == col.max():
            continue
        vals.append(float(roc_auc_score(col, y_score[:, j])))
    return {"macro_auc": float(np.mean(vals)) if vals else None, "valid_classes": len(vals)}


def group_key(row_id: str, mode: str) -> str:
    text = str(row_id)
    if mode == "row":
        return text
    if mode == "file":
        return text.rsplit("_", 1)[0]
    if mode == "site":
        # Common BirdCLEF train-soundscape ids contain site tokens like S08.
        for part in text.replace("-", "_").split("_"):
            if len(part) >= 2 and part[0].upper() == "S" and part[1:].isdigit():
                return part.upper()
        return text.rsplit("_", 1)[0]
    raise ValueError(f"unknown group mode {mode!r}")


def bootstrap_lift_vs_base(
    row_ids: pd.Series,
    y_true: np.ndarray,
    base_values: np.ndarray,
    candidate_values: np.ndarray,
    *,
    group_mode: str,
    iters: int,
    seed: int,
) -> dict[str, Any]:
    groups = np.array([group_key(x, group_mode) for x in row_ids.astype(str).tolist()])
    unique_groups = np.array(sorted(set(groups.tolist())))
    by_group = {g: np.flatnonzero(groups == g) for g in unique_groups}
    rng = np.random.default_rng(seed)
    lifts = []
    valid_iters = 0
    for _ in range(int(iters)):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([by_group[g] for g in sampled_groups])
        base_auc = macro_auc_matrix(y_true[idx], base_values[idx])["macro_auc"]
        cand_auc = macro_auc_matrix(y_true[idx], candidate_values[idx])["macro_auc"]
        if base_auc is None or cand_auc is None:
            continue
        lifts.append(float(cand_auc - base_auc))
        valid_iters += 1
    arr = np.array(lifts, dtype=np.float64)
    if arr.size == 0:
        return {"iters": int(iters), "valid_iters": 0, "group_mode": group_mode, "n_groups": int(len(unique_groups))}
    return {
        "iters": int(iters),
        "valid_iters": int(valid_iters),
        "group_mode": group_mode,
        "n_groups": int(len(unique_groups)),
        "mean_lift": float(arr.mean()),
        "median_lift": float(np.median(arr)),
        "p_lift_gt_0": float(np.mean(arr > 0)),
        "q05_lift": float(np.quantile(arr, 0.05)),
        "q25_lift": float(np.quantile(arr, 0.25)),
        "q75_lift": float(np.quantile(arr, 0.75)),
        "q95_lift": float(np.quantile(arr, 0.95)),
        "min_lift": float(arr.min()),
        "max_lift": float(arr.max()),
    }


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
            info.update(macro_auc_matrix(y_true, y_score))
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
    ap.add_argument("--bootstrap-iters", type=int, default=0, help="Bootstrap matched labeled rows by group and report candidate lift stability vs base")
    ap.add_argument("--bootstrap-group", choices=["file", "site", "row"], default="file")
    ap.add_argument("--bootstrap-seed", type=int, default=42)
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

    candidate_arrays: dict[str, np.ndarray] = {"base": rank_base}
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
        candidate_arrays[label] = pred
        summaries.append(summarize(label, base["row_id"], cols, pred, labels_wide, rank_base, weights))

    if labels_wide is not None and args.bootstrap_iters > 0:
        label_merge = base[["row_id"]].merge(labels_wide, on="row_id", how="inner")
        matched_row_ids = label_merge["row_id"].astype(str)
        matched_idx = base.index[base["row_id"].astype(str).isin(set(matched_row_ids))].to_numpy()
        y_cols = [c for c in cols if c in label_merge.columns and label_merge[c].nunique() > 1]
        if y_cols and len(matched_idx):
            # Preserve base order for bootstrap arrays.
            labels_by_row = labels_wide.set_index("row_id")
            y_true = labels_by_row.loc[base.loc[matched_idx, "row_id"], y_cols].to_numpy()
            pred_col_idx = np.array([cols.index(c) for c in y_cols], dtype=np.int64)
            for summary in summaries:
                if summary["name"] == "base":
                    continue
                arr = candidate_arrays[summary["name"]]
                summary["bootstrap_vs_base"] = bootstrap_lift_vs_base(
                    base.loc[matched_idx, "row_id"],
                    y_true,
                    rank_base[matched_idx][:, pred_col_idx],
                    arr[matched_idx][:, pred_col_idx],
                    group_mode=args.bootstrap_group,
                    iters=args.bootstrap_iters,
                    seed=args.bootstrap_seed,
                )

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
