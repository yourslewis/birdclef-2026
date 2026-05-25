#!/usr/bin/env python3
"""Per-class capped sidecar selector for BirdCLEF anchor blends.

This is a no-submit rejection/idea screen.  It learns tiny per-class sidecar
weights on labelled train-soundscape overlap and evaluates them by group CV
(file/site).  The goal is to answer whether a class-conditional version of a
bounded global sidecar blend is worth packaging as a private verifier.
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
from birdclef_public946_multi_sidecar_weight_grid import group_key


def parse_named_path(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"expected NAME=PATH, got {text!r}")
    name, path = text.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"empty sidecar name in {text!r}")
    return name, Path(path.strip())


def parse_grid(text: str) -> list[float]:
    values = [float(x) for x in text.replace(" ", "").split(",") if x]
    if not values:
        raise ValueError("empty weight grid")
    if any(w < 0 or w > 1 for w in values):
        raise ValueError(f"weights must be in [0,1]: {values}")
    return sorted(set(values))


def prediction_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c != "row_id"]
    if not cols:
        raise ValueError("prediction CSV has no class columns")
    return cols


def load_prediction(path: Path, row_ids: pd.Series | None = None, cols: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "row_id" not in df.columns:
        raise ValueError(f"{path} missing row_id")
    if cols is not None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{path} missing {len(missing)} columns, first={missing[:5]}")
        df = df[["row_id", *cols]]
    if row_ids is not None:
        df = df.set_index("row_id").loc[row_ids.astype(str)].reset_index()
    return df


def rank_values(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    values = np.clip(df[cols].to_numpy(np.float32), 1e-7, 1.0 - 1e-7)
    return pd.DataFrame(values).rank(axis=0, pct=True).to_numpy(np.float32)


def macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float | None, int, list[float | None]]:
    vals: list[float] = []
    by_class: list[float | None] = []
    for j in range(y_true.shape[1]):
        col = y_true[:, j]
        if col.min() == col.max():
            by_class.append(None)
            continue
        auc = float(roc_auc_score(col, y_score[:, j]))
        vals.append(auc)
        by_class.append(auc)
    return (float(np.mean(vals)) if vals else None, len(vals), by_class)


def candidate_grid(n_sidecars: int, weights: list[float], max_total: float) -> list[tuple[float, ...]]:
    combos = []
    for combo in itertools.product(weights, repeat=n_sidecars):
        if sum(combo) <= max_total + 1e-12:
            combos.append(tuple(float(x) for x in combo))
    return sorted(combos, key=lambda xs: (sum(xs), xs))


def blend_column(base_col: np.ndarray, side_cols: list[np.ndarray], weights: tuple[float, ...]) -> np.ndarray:
    total = float(sum(weights))
    out = (1.0 - total) * base_col.astype(np.float32)
    for w, col in zip(weights, side_cols):
        if w:
            out = out + float(w) * col.astype(np.float32)
    return out.astype(np.float32)


def choose_weights_for_class(
    y: np.ndarray,
    base_col: np.ndarray,
    side_cols: list[np.ndarray],
    train_idx: np.ndarray,
    combos: list[tuple[float, ...]],
    min_train_pos: int,
    min_train_neg: int,
    min_lift: float,
) -> tuple[float, ...]:
    yt = y[train_idx]
    pos = int(yt.sum())
    neg = int(len(yt) - pos)
    zero = tuple(0.0 for _ in side_cols)
    if pos < min_train_pos or neg < min_train_neg or yt.min() == yt.max():
        return zero
    base_auc = float(roc_auc_score(yt, base_col[train_idx]))
    best = (base_auc, 0.0, zero)  # auc, negative complexity, combo
    for combo in combos:
        pred = blend_column(base_col, side_cols, combo)[train_idx]
        auc = float(roc_auc_score(yt, pred))
        complexity = sum(combo) + 1e-3 * sum(w * w for w in combo)
        # Tie-break toward lower total weight / simpler candidate.
        score = (auc, -complexity, combo)
        if score[:2] > best[:2]:
            best = score
    if best[0] - base_auc < min_lift:
        return zero
    return best[2]


def summarize_weights(cols: list[str], sidecar_names: list[str], weights_by_class: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_classes": len(cols),
        "classes_with_any_sidecar": int((weights_by_class.sum(axis=1) > 0).sum()),
        "mean_total_weight": float(weights_by_class.sum(axis=1).mean()),
        "max_total_weight": float(weights_by_class.sum(axis=1).max()),
        "per_sidecar_mean": {},
        "per_sidecar_nonzero_classes": {},
        "top_classes": [],
    }
    for i, name in enumerate(sidecar_names):
        out["per_sidecar_mean"][name] = float(weights_by_class[:, i].mean())
        out["per_sidecar_nonzero_classes"][name] = int((weights_by_class[:, i] > 0).sum())
    totals = weights_by_class.sum(axis=1)
    for idx in np.argsort(-totals)[:20]:
        if totals[idx] <= 0:
            break
        out["top_classes"].append({
            "class": cols[int(idx)],
            "total": float(totals[idx]),
            "weights": {name: float(weights_by_class[int(idx), i]) for i, name in enumerate(sidecar_names)},
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-csv", required=True, type=Path)
    ap.add_argument("--sidecar", action="append", default=[], help="NAME=CSV; can repeat")
    ap.add_argument("--labels-csv", required=True, type=Path)
    ap.add_argument("--weights", default="0,0.01,0.02,0.04,0.06")
    ap.add_argument("--max-total-weight", type=float, default=0.08)
    ap.add_argument("--group", choices=["site", "file", "row"], default="site")
    ap.add_argument("--min-train-pos", type=int, default=2)
    ap.add_argument("--min-train-neg", type=int, default=10)
    ap.add_argument("--min-lift", type=float, default=0.0005)
    ap.add_argument("--output-json", required=True, type=Path)
    args = ap.parse_args()

    sidecars = [parse_named_path(x) for x in args.sidecar]
    if not sidecars:
        raise ValueError("at least one --sidecar required")

    base = load_prediction(args.base_csv)
    cols = prediction_columns(base)
    side_dfs = [(name, load_prediction(path, base["row_id"], cols)) for name, path in sidecars]
    labels = load_long_labels(args.labels_csv, cols)
    labels = labels[["row_id", *[c for c in cols if c in labels.columns]]]

    base_ranks_all = rank_values(base, cols)
    side_ranks_all = [(name, rank_values(df, cols)) for name, df in side_dfs]

    merged = base[["row_id"]].merge(labels, on="row_id", how="inner")
    matched_ids = merged["row_id"].astype(str)
    matched_idx = base.index[base["row_id"].astype(str).isin(set(matched_ids))].to_numpy()
    # Preserve base order on matched rows.
    matched_base = base.iloc[matched_idx].reset_index(drop=True)
    labels_matched = matched_base[["row_id"]].merge(labels, on="row_id", how="inner")
    valid_cols = [c for c in cols if c in labels_matched.columns and labels_matched[c].nunique() > 1]
    valid_idx = np.array([cols.index(c) for c in valid_cols], dtype=int)
    y = labels_matched[valid_cols].to_numpy(np.float32)
    base_values = base_ranks_all[matched_idx][:, valid_idx]
    side_values = [arr[matched_idx][:, valid_idx] for _, arr in side_ranks_all]
    side_names = [name for name, _ in side_ranks_all]

    base_auc, valid_n, base_by_class = macro_auc(y, base_values)
    combos = candidate_grid(len(side_values), parse_grid(args.weights), args.max_total_weight)

    groups = np.array([group_key(x, args.group) for x in labels_matched["row_id"].astype(str)])
    unique_groups = sorted(set(groups.tolist()))
    held_rows = []
    cv_pred = np.zeros_like(base_values, dtype=np.float32)
    cv_weights_records = []
    for group in unique_groups:
        train_idx = np.flatnonzero(groups != group)
        test_idx = np.flatnonzero(groups == group)
        weights_by_class = np.zeros((len(valid_cols), len(side_values)), dtype=np.float32)
        for j in range(len(valid_cols)):
            chosen = choose_weights_for_class(
                y[:, j],
                base_values[:, j],
                [sv[:, j] for sv in side_values],
                train_idx,
                combos,
                args.min_train_pos,
                args.min_train_neg,
                args.min_lift,
            )
            weights_by_class[j, :] = chosen
            cv_pred[test_idx, j] = blend_column(base_values[:, j], [sv[:, j] for sv in side_values], chosen)[test_idx]
        base_auc_g, _, _ = macro_auc(y[test_idx], base_values[test_idx])
        cand_auc_g, _, _ = macro_auc(y[test_idx], cv_pred[test_idx])
        held_rows.append({
            "held_out_group": group,
            "rows": int(len(test_idx)),
            "base_auc": base_auc_g,
            "candidate_auc": cand_auc_g,
            "lift": None if base_auc_g is None or cand_auc_g is None else float(cand_auc_g - base_auc_g),
            "weight_summary": summarize_weights(valid_cols, side_names, weights_by_class),
        })
        cv_weights_records.append(weights_by_class)

    cv_auc, cv_valid_n, _ = macro_auc(y, cv_pred)
    cv_lift = None if base_auc is None or cv_auc is None else float(cv_auc - base_auc)

    # All-row selector for diagnostics / possible fixed-weight recipe, not approval.
    all_idx = np.arange(y.shape[0])
    all_weights = np.zeros((len(valid_cols), len(side_values)), dtype=np.float32)
    all_pred = np.zeros_like(base_values, dtype=np.float32)
    for j in range(len(valid_cols)):
        chosen = choose_weights_for_class(
            y[:, j], base_values[:, j], [sv[:, j] for sv in side_values], all_idx,
            combos, args.min_train_pos, args.min_train_neg, args.min_lift,
        )
        all_weights[j, :] = chosen
        all_pred[:, j] = blend_column(base_values[:, j], [sv[:, j] for sv in side_values], chosen)
    all_auc, _, _ = macro_auc(y, all_pred)

    lifts = np.array([r["lift"] for r in held_rows if r["lift"] is not None], dtype=np.float64)
    out = {
        "base_csv": str(args.base_csv),
        "sidecars": [{"name": n, "path": str(p)} for n, p in sidecars],
        "labels_csv": str(args.labels_csv),
        "matched_rows": int(y.shape[0]),
        "valid_classes": int(len(valid_cols)),
        "group_mode": args.group,
        "n_groups": len(unique_groups),
        "weight_grid": parse_grid(args.weights),
        "max_total_weight": args.max_total_weight,
        "min_train_pos": args.min_train_pos,
        "min_train_neg": args.min_train_neg,
        "min_lift": args.min_lift,
        "base_auc": base_auc,
        "cv_auc": cv_auc,
        "cv_lift": cv_lift,
        "cv_valid_classes": cv_valid_n,
        "leave_group_lifts": {
            "n": int(len(lifts)),
            "mean": float(lifts.mean()) if lifts.size else None,
            "min": float(lifts.min()) if lifts.size else None,
            "q05": float(np.quantile(lifts, 0.05)) if lifts.size else None,
            "p_gt_0": float((lifts > 0).mean()) if lifts.size else None,
        },
        "held_out_groups": held_rows,
        "all_row_auc": all_auc,
        "all_row_lift": None if base_auc is None or all_auc is None else float(all_auc - base_auc),
        "all_row_weight_summary": summarize_weights(valid_cols, side_names, all_weights),
        "topk_recall": {
            "base_top3": topk_row_recall(base_values, y, 3),
            "cv_top3": topk_row_recall(cv_pred, y, 3),
            "all_row_top3": topk_row_recall(all_pred, y, 3),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({
        "base_auc": base_auc,
        "cv_auc": cv_auc,
        "cv_lift": cv_lift,
        "leave_group_lifts": out["leave_group_lifts"],
        "all_row_auc": all_auc,
        "all_row_lift": out["all_row_lift"],
        "all_row_weight_summary": out["all_row_weight_summary"],
        "topk_recall": out["topk_recall"],
    }, indent=2))


if __name__ == "__main__":
    main()
