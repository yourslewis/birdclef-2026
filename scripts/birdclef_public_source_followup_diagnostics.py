#!/usr/bin/env python3
"""Diagnostics for May-16 public-source follow-up ideas.

This intentionally does not submit anything.  It ports two source ideas into a
local, evidence-producing harness:

1. lucataco-style conservative score-desc/rank overlay over existing candidate
   submissions.
2. kruzzcc/Yaroslav-style site+hour prior, evaluated both optimistically and
   with leave-one-group cross-fitting to reduce train-soundscape leakage.

The output is a JSON report with local AUC/displacement/stability summaries that
can be compared against v542/v558/v560 before spending a Kaggle slot.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from birdclef_public946_cache_summary import load_long_labels, topk_row_recall


EPS = 1e-6
ROW_RE = re.compile(r"(?P<prefix>.*?_(?P<site>S\d{2})_(?P<date>\d{8})_(?P<hms>\d{6}))_(?P<end>\d+)$")


def parse_named_path(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"expected NAME=PATH, got {text!r}")
    name, path = text.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"empty name in {text!r}")
    return name, Path(path.strip())


def parse_float_list(text: str) -> list[float]:
    vals = [float(x) for x in text.replace(" ", "").split(",") if x]
    if not vals:
        raise ValueError("empty float list")
    return vals


def prediction_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c != "row_id"]
    if not cols:
        raise ValueError("prediction CSV has no class columns")
    return cols


def load_submission(path: Path, row_ids: pd.Series | None, cols: list[str] | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "row_id" not in df:
        raise ValueError(f"{path} missing row_id")
    if df["row_id"].duplicated().any():
        raise ValueError(f"{path} has duplicate row_id values")
    if cols is not None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{path} missing {len(missing)} columns; first={missing[:5]}")
        df = df[["row_id", *cols]]
    if row_ids is not None:
        df = df.set_index("row_id").loc[row_ids.astype(str)].reset_index()
    return df


def values(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return np.clip(df[cols].to_numpy(np.float32), EPS, 1.0 - EPS)


def rank01(mat: np.ndarray) -> np.ndarray:
    return pd.DataFrame(np.clip(mat, EPS, 1.0 - EPS)).rank(axis=0, pct=True).to_numpy(np.float32)


def macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    vals: list[float] = []
    for j in range(y_true.shape[1]):
        col = y_true[:, j]
        if col.min() == col.max():
            continue
        vals.append(float(roc_auc_score(col, y_score[:, j])))
    return {"macro_auc": float(np.mean(vals)) if vals else None, "valid_classes": len(vals)}


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))).astype(np.float32)


def parse_meta(row_id: str) -> dict[str, Any]:
    m = ROW_RE.match(str(row_id))
    if not m:
        text = str(row_id)
        return {"file_id": text.rsplit("_", 1)[0], "site": "unknown", "hour_utc": -1}
    return {
        "file_id": m.group("prefix"),
        "site": m.group("site"),
        "hour_utc": int(m.group("hms")[:2]),
    }


def meta_frame(row_ids: pd.Series) -> pd.DataFrame:
    rows = [parse_meta(x) for x in row_ids.astype(str).tolist()]
    out = pd.DataFrame(rows)
    out.insert(0, "row_id", row_ids.astype(str).to_numpy())
    return out


def build_prior_tables(meta: pd.DataFrame, y: np.ndarray) -> dict[str, Any]:
    global_p = y.mean(axis=0).astype(np.float32)

    site_keys = sorted(meta["site"].dropna().astype(str).unique())
    site_to_i = {k: i for i, k in enumerate(site_keys)}
    site_p = np.zeros((len(site_keys), y.shape[1]), dtype=np.float32)
    site_n = np.zeros(len(site_keys), dtype=np.float32)
    site_values = meta["site"].astype(str).to_numpy()
    for s in site_keys:
        i = site_to_i[s]
        mask = site_values == s
        site_n[i] = float(mask.sum())
        site_p[i] = y[mask].mean(axis=0)

    hour_keys = sorted(meta["hour_utc"].dropna().astype(int).unique())
    hour_to_i = {int(k): i for i, k in enumerate(hour_keys)}
    hour_p = np.zeros((len(hour_keys), y.shape[1]), dtype=np.float32)
    hour_n = np.zeros(len(hour_keys), dtype=np.float32)
    hour_values = meta["hour_utc"].astype(int).to_numpy()
    for h in hour_keys:
        i = hour_to_i[int(h)]
        mask = hour_values == int(h)
        hour_n[i] = float(mask.sum())
        hour_p[i] = y[mask].mean(axis=0)

    sh_keys = sorted({(str(s), int(h)) for s, h in zip(site_values, hour_values)})
    sh_to_i = {k: i for i, k in enumerate(sh_keys)}
    sh_p = np.zeros((len(sh_keys), y.shape[1]), dtype=np.float32)
    sh_n = np.zeros(len(sh_keys), dtype=np.float32)
    for s, h in sh_keys:
        i = sh_to_i[(s, h)]
        mask = (site_values == s) & (hour_values == h)
        sh_n[i] = float(mask.sum())
        sh_p[i] = y[mask].mean(axis=0)

    return {
        "global_p": global_p,
        "site_to_i": site_to_i,
        "site_p": site_p,
        "site_n": site_n,
        "hour_to_i": hour_to_i,
        "hour_p": hour_p,
        "hour_n": hour_n,
        "sh_to_i": sh_to_i,
        "sh_p": sh_p,
        "sh_n": sh_n,
    }


def prior_matrix(meta: pd.DataFrame, tables: dict[str, Any]) -> np.ndarray:
    n = len(meta)
    p = np.tile(tables["global_p"], (n, 1)).astype(np.float32)
    sites = meta["site"].astype(str).to_numpy()
    hours = meta["hour_utc"].astype(int).to_numpy()

    for i, h in enumerate(hours):
        h = int(h)
        if h in tables["hour_to_i"]:
            j = tables["hour_to_i"][h]
            nh = tables["hour_n"][j]
            w = nh / (nh + 8.0)
            p[i] = w * tables["hour_p"][j] + (1.0 - w) * tables["global_p"]

    for i, s in enumerate(sites):
        if s in tables["site_to_i"]:
            j = tables["site_to_i"][s]
            ns = tables["site_n"][j]
            w = ns / (ns + 8.0)
            p[i] = w * tables["site_p"][j] + (1.0 - w) * p[i]

    for i, (s, h) in enumerate(zip(sites, hours)):
        key = (str(s), int(h))
        if key in tables["sh_to_i"]:
            j = tables["sh_to_i"][key]
            nsh = tables["sh_n"][j]
            w = nsh / (nsh + 1.0)
            p[i] = w * tables["sh_p"][j] + (1.0 - w) * p[i]

    return np.clip(p, 1e-4, 1.0 - 1e-4).astype(np.float32)


def apply_sitehour_prior(base: np.ndarray, meta: pd.DataFrame, tables: dict[str, Any], lam: float) -> np.ndarray:
    if lam == 0:
        return base.copy()
    return sigmoid(logit(base) + float(lam) * logit(prior_matrix(meta, tables)))


def group_keys(meta: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "site":
        return meta["site"].astype(str).to_numpy()
    if mode == "file":
        return meta["file_id"].astype(str).to_numpy()
    if mode == "row":
        return meta["row_id"].astype(str).to_numpy()
    raise ValueError(f"unknown group mode {mode!r}")


def evaluate_values(name: str, row_ids: pd.Series, cols: list[str], pred: np.ndarray, y_wide: pd.DataFrame, anchor: np.ndarray | None = None) -> dict[str, Any]:
    df = pd.DataFrame(pred.astype(np.float32), columns=cols)
    df.insert(0, "row_id", row_ids.astype(str).to_numpy())
    merged = df.merge(y_wide, on="row_id", suffixes=("_pred", "_true"))
    valid = [c for c in cols if f"{c}_pred" in merged and f"{c}_true" in merged and merged[f"{c}_true"].nunique() > 1]
    out: dict[str, Any] = {
        "name": name,
        "rows": int(len(df)),
        "matched_rows": int(len(merged)),
        "valid_auc_classes": int(len(valid)),
        "prob_stats": {"min": float(pred.min()), "max": float(pred.max()), "mean": float(pred.mean())},
    }
    if valid:
        y_true = merged[[f"{c}_true" for c in valid]].to_numpy()
        y_score = merged[[f"{c}_pred" for c in valid]].to_numpy()
        out.update(macro_auc(y_true, y_score))
        pred_cols = [f"{c}_pred" for c in cols if f"{c}_pred" in merged]
        true_cols = [c.replace("_pred", "_true") for c in pred_cols]
        score_mat = merged[pred_cols].to_numpy()
        true_mat = merged[true_cols].to_numpy()
        for k in (1, 3, 5, 10):
            out[f"top{k}_row_recall"] = topk_row_recall(score_mat, true_mat, k)
    if anchor is not None:
        out["corr_vs_anchor"] = float(np.corrcoef(anchor.ravel(), pred.ravel())[0, 1])
        out["mae_vs_anchor"] = float(np.mean(np.abs(anchor - pred)))
        out["max_abs_vs_anchor"] = float(np.max(np.abs(anchor - pred)))
    return out


def conservative_scoredesc_overlay(anchor: np.ndarray, side_values: list[np.ndarray], side_weights: list[float], *, raw_alpha: float, anchor_alpha: float, rank_alpha: float) -> np.ndarray:
    total_side = float(sum(side_weights))
    if total_side > 1.0 + 1e-9:
        raise ValueError("side weights sum above 1")
    mats = [anchor, *side_values]
    weights = [1.0 - total_side, *side_weights]
    raw_total = np.zeros_like(anchor, dtype=np.float32)
    rank_total = np.zeros_like(anchor, dtype=np.float32)
    for mat, w in zip(mats, weights):
        raw_total += float(w) * mat
        rank_total += float(w) * rank01(mat)
    pred = raw_alpha * raw_total + anchor_alpha * anchor + rank_alpha * rank_total
    return np.clip(pred, 0.0, 1.0).astype(np.float32)


def crossfit_sitehour(base: np.ndarray, meta: pd.DataFrame, y: np.ndarray, group_mode: str, lam: float) -> tuple[np.ndarray, dict[str, Any]]:
    groups = group_keys(meta, group_mode)
    unique = np.array(sorted(set(groups.tolist())))
    pred = np.zeros_like(base, dtype=np.float32)
    details: list[dict[str, Any]] = []
    for group in unique:
        va = np.flatnonzero(groups == group)
        tr = np.flatnonzero(groups != group)
        if len(tr) == 0 or len(va) == 0:
            continue
        tables = build_prior_tables(meta.iloc[tr].reset_index(drop=True), y[tr])
        pred[va] = apply_sitehour_prior(base[va], meta.iloc[va].reset_index(drop=True), tables, lam)
        base_auc = macro_auc(y[va], base[va])["macro_auc"]
        cand_auc = macro_auc(y[va], pred[va])["macro_auc"]
        if base_auc is not None and cand_auc is not None:
            details.append({"held_out_group": str(group), "rows": int(len(va)), "base_auc": float(base_auc), "candidate_auc": float(cand_auc), "lift": float(cand_auc - base_auc)})
    return pred, {"group_mode": group_mode, "n_groups": int(len(unique)), "valid_group_details": details}


def lift_summary(lifts: list[float]) -> dict[str, Any]:
    arr = np.array([x for x in lifts if math.isfinite(x)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean_lift": float(arr.mean()),
        "median_lift": float(np.median(arr)),
        "p_lift_gt_0": float(np.mean(arr > 0)),
        "q05_lift": float(np.quantile(arr, 0.05)),
        "min_lift": float(arr.min()),
        "max_lift": float(arr.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-csv", type=Path, required=True)
    ap.add_argument("--labels-csv", type=Path, required=True)
    ap.add_argument("--candidate", action="append", default=[], help="Repeat NAME=PATH; aligned comparison/scoredesc source")
    ap.add_argument("--scoredesc-sources", default="", help="Comma-separated candidate names to use in scoredesc grid")
    ap.add_argument("--scoredesc-weights", default="0,0.005,0.01,0.02,0.03,0.05,0.075,0.10")
    ap.add_argument("--scoredesc-max-total", type=float, default=0.12)
    ap.add_argument("--sitehour-lambdas", default="0,0.005,0.01,0.02,0.03,0.05,0.075,0.10,0.15,0.20,0.30,0.40")
    ap.add_argument("--sitehour-groups", default="file,site")
    ap.add_argument("--raw-alpha", type=float, default=0.70)
    ap.add_argument("--anchor-alpha", type=float, default=0.20)
    ap.add_argument("--rank-alpha", type=float, default=0.10)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args()

    base_df = load_submission(args.base_csv, None, None)
    cols = prediction_columns(base_df)
    row_ids = base_df["row_id"].astype(str)
    base = values(base_df, cols)
    labels_wide = load_long_labels(args.labels_csv, cols)
    # load_long_labels only materializes positive-label rows.  The dry-run
    # train-soundscape output also includes unlabeled/negative windows, so align
    # to the prediction row_ids and fill missing labels with zeros.
    labels_aligned = labels_wide.set_index("row_id").reindex(row_ids).fillna(0).reset_index()
    labels_aligned[cols] = labels_aligned[cols].astype(np.uint8)
    y = labels_aligned[cols].to_numpy(np.uint8)
    meta = meta_frame(row_ids)

    candidates: dict[str, np.ndarray] = {}
    candidate_summaries: list[dict[str, Any]] = []
    for item in args.candidate:
        name, path = parse_named_path(item)
        df = load_submission(path, row_ids, cols)
        mat = values(df, cols)
        candidates[name] = mat
        info = evaluate_values(name, row_ids, cols, mat, labels_wide, base)
        base_info = evaluate_values("base", row_ids, cols, base, labels_wide, None)
        info["lift_vs_base"] = None if info.get("macro_auc") is None or base_info.get("macro_auc") is None else float(info["macro_auc"] - base_info["macro_auc"])
        candidate_summaries.append(info)

    base_summary = evaluate_values("base", row_ids, cols, base, labels_wide)

    # Site-hour prior diagnostics.
    sitehour_rows: list[dict[str, Any]] = []
    all_tables = build_prior_tables(meta, y)
    for lam in parse_float_list(args.sitehour_lambdas):
        pred_full = apply_sitehour_prior(base, meta, all_tables, lam)
        info = evaluate_values(f"sitehour_full_lambda_{lam:g}", row_ids, cols, pred_full, labels_wide, base)
        info["lambda"] = float(lam)
        info["mode"] = "full_in_sample"
        info["lift_vs_base"] = None if info.get("macro_auc") is None or base_summary.get("macro_auc") is None else float(info["macro_auc"] - base_summary["macro_auc"])
        sitehour_rows.append(info)
        if lam == 0:
            continue
        for group_mode in [x for x in args.sitehour_groups.split(",") if x]:
            pred_cf, cf_meta = crossfit_sitehour(base, meta, y, group_mode, lam)
            cf_info = evaluate_values(f"sitehour_crossfit_{group_mode}_lambda_{lam:g}", row_ids, cols, pred_cf, labels_wide, base)
            cf_info["lambda"] = float(lam)
            cf_info["mode"] = f"crossfit_{group_mode}"
            cf_info["crossfit"] = {k: v for k, v in cf_meta.items() if k != "valid_group_details"}
            lifts = [float(d["lift"]) for d in cf_meta["valid_group_details"]]
            cf_info["group_lift_summary"] = lift_summary(lifts)
            cf_info["worst_groups"] = sorted(cf_meta["valid_group_details"], key=lambda d: d["lift"])[:10]
            cf_info["best_groups"] = sorted(cf_meta["valid_group_details"], key=lambda d: d["lift"], reverse=True)[:10]
            cf_info["lift_vs_base"] = None if cf_info.get("macro_auc") is None or base_summary.get("macro_auc") is None else float(cf_info["macro_auc"] - base_summary["macro_auc"])
            sitehour_rows.append(cf_info)

    # Conservative score-desc/rank overlay grid.
    scoredesc_rows: list[dict[str, Any]] = []
    source_names = [x for x in args.scoredesc_sources.split(",") if x]
    weight_values = parse_float_list(args.scoredesc_weights)
    missing = [x for x in source_names if x not in candidates]
    if missing:
        raise ValueError(f"scoredesc source names missing from --candidate: {missing}")
    if source_names:
        mats = [candidates[x] for x in source_names]
        for combo in itertools.product(weight_values, repeat=len(source_names)):
            if sum(combo) > args.scoredesc_max_total + 1e-12:
                continue
            if all(w == 0 for w in combo):
                continue
            pred = conservative_scoredesc_overlay(
                base,
                mats,
                list(combo),
                raw_alpha=args.raw_alpha,
                anchor_alpha=args.anchor_alpha,
                rank_alpha=args.rank_alpha,
            )
            weights = {name: float(w) for name, w in zip(source_names, combo) if w > 0}
            label = "scoredesc_" + "_".join(f"{k}{v:g}" for k, v in weights.items())
            info = evaluate_values(label, row_ids, cols, pred, labels_wide, base)
            info["weights"] = weights
            info["formula"] = {"raw_alpha": args.raw_alpha, "anchor_alpha": args.anchor_alpha, "rank_alpha": args.rank_alpha, "side_total": float(sum(combo))}
            info["lift_vs_base"] = None if info.get("macro_auc") is None or base_summary.get("macro_auc") is None else float(info["macro_auc"] - base_summary["macro_auc"])
            scoredesc_rows.append(info)

    def top(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        usable = [r for r in rows if r.get("macro_auc") is not None]
        return sorted(usable, key=lambda r: (r.get("macro_auc") or -1, -(r.get("mae_vs_anchor") or 0)), reverse=True)[: args.top_k]

    out = {
        "base_csv": str(args.base_csv),
        "labels_csv": str(args.labels_csv),
        "rows": int(len(row_ids)),
        "n_classes": int(len(cols)),
        "base": base_summary,
        "candidate_summaries": candidate_summaries,
        "scoredesc": {
            "source_names": source_names,
            "n_tested": int(len(scoredesc_rows)),
            "top_by_auc": top(scoredesc_rows),
        },
        "sitehour_prior": {
            "lambdas": parse_float_list(args.sitehour_lambdas),
            "n_tested": int(len(sitehour_rows)),
            "top_by_auc": top(sitehour_rows),
            "all": sitehour_rows,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output_json),
        "base_auc": base_summary.get("macro_auc"),
        "top_scoredesc": out["scoredesc"]["top_by_auc"][:5],
        "top_sitehour": out["sitehour_prior"]["top_by_auc"][:5],
    }, indent=2))


if __name__ == "__main__":
    main()
