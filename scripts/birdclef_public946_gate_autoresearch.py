#!/usr/bin/env python3
"""AutoResearch-style sweep for public946 rank-blend gate parameters.

This script is intentionally offline: it uses already-generated dry-run
ProtoSSM/SED outputs plus train-soundscape labels to search small changes to the
public946 post-processing gates.  It is a pre-submit filter, not a substitute for
leaderboard validation.
"""
from __future__ import annotations

import argparse
import itertools
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.metrics import roc_auc_score

warnings.simplefilter("ignore", PerformanceWarning)

from birdclef_public946_cache_summary import load_long_labels, topk_row_recall

MIRROR_PAIRS = (
    ("47158son15", "47158son16"),
    ("47158son09", "47158son12"),
    ("47158son02", "47158son14"),
    ("47158son13", "47158son21", "47158son22", "47158son23"),
)
RARE_CLASSES = {"Amphibia", "Mammalia", "Reptilia"}


@dataclass(frozen=True)
class GateConfig:
    proto_weight: float = 0.60
    fake_boost: float = 0.08
    fake_proto_min: float = 0.50
    fake_sed_max: float = 0.05
    ctx_thr: float = 0.88
    ctx_rank_min: float = 0.75
    ctx_sed_max: float = 0.12
    ctx_boost: float = 0.15
    sed_rank_thr: float = 0.95
    sed_proto_max: float = 0.80
    sed_boost: float = 0.12
    rare_margin: float = 0.05
    rare_scale: float = 0.90


def _parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _rank(values: np.ndarray) -> np.ndarray:
    return pd.DataFrame(values).rank(axis=0, pct=True).to_numpy(np.float32)


def _file_ids(row_ids: Iterable[str]) -> np.ndarray:
    return np.array(["_".join(str(r).split("_")[:-1]) for r in row_ids])


def _proto_context(p_proto: np.ndarray, file_ids: np.ndarray) -> np.ndarray:
    offs = np.arange(-3, 4, dtype=np.float32)
    kernel = (1.0 + (offs / 1.20) ** 2 / 2.0) ** (-1.5)
    kernel = (kernel / kernel.sum()).astype(np.float32)
    pa_ctx = p_proto.copy()
    for fid in pd.unique(file_ids):
        m = file_ids == fid
        x = p_proto[m]
        if len(x) > 1:
            xp = np.pad(x, ((3, 3), (0, 0)), mode="edge")
            pa_ctx[m] = sum(kernel[i] * xp[i : i + len(x)] for i in range(7))
    return pa_ctx.astype(np.float32)


def apply_config(
    proto: pd.DataFrame,
    sed: pd.DataFrame,
    cols: list[str],
    tax_df: pd.DataFrame | None,
    cfg: GateConfig,
    *,
    mirror: bool = True,
) -> pd.DataFrame:
    sed = sed.set_index("row_id").loc[proto["row_id"]].reset_index().copy()
    p_proto = np.clip(proto[cols].to_numpy(np.float32), 1e-7, 1.0 - 1e-7)
    p_sed = np.clip(sed[cols].to_numpy(np.float32), 1e-7, 1.0 - 1e-7)
    rank_proto = _rank(p_proto)
    rank_sed = _rank(p_sed)
    pred = rank_proto * cfg.proto_weight + rank_sed * (1.0 - cfg.proto_weight)

    fake_only = (p_proto > cfg.fake_proto_min) & (p_sed < cfg.fake_sed_max)
    if cfg.fake_boost:
        pred = np.where(fake_only, (1.0 - cfg.fake_boost) * pred + cfg.fake_boost * rank_proto, pred)

    ids = _file_ids(proto["row_id"].astype(str).to_numpy())
    pa_ctx = _proto_context(p_proto, ids)
    rank_ctx = _rank(pa_ctx)
    proto_cont = (
        (rank_ctx > cfg.ctx_thr)
        & (rank_proto > cfg.ctx_rank_min)
        & (p_sed < cfg.ctx_sed_max)
        & (~fake_only)
    )
    if cfg.ctx_boost:
        pred = np.where(proto_cont, (1.0 - cfg.ctx_boost) * pred + cfg.ctx_boost * np.maximum(rank_proto, rank_ctx), pred)

    sed_only = (rank_sed > cfg.sed_rank_thr) & (rank_proto < cfg.sed_proto_max) & (~fake_only) & (~proto_cont)
    if cfg.sed_boost:
        pred = np.where(sed_only, (1.0 - cfg.sed_boost) * pred + cfg.sed_boost * rank_sed, pred)

    out = pd.concat(
        [
            proto[["row_id"]].reset_index(drop=True),
            pd.DataFrame(pred.astype(np.float32), columns=cols),
        ],
        axis=1,
    )

    if mirror:
        col_to_idx = {label: i for i, label in enumerate(cols)}
        for group in MIRROR_PAIRS:
            valid_idx = [col_to_idx[s] for s in group if s in col_to_idx]
            if len(valid_idx) >= 2:
                group_max = out[cols].iloc[:, valid_idx].max(axis=1).to_numpy(np.float32)
                for idx in valid_idx:
                    out.iloc[:, idx + 1] = group_max

    if tax_df is not None:
        for ci, species in enumerate(cols):
            if species in tax_df.index and tax_df.loc[species, "class_name"] in RARE_CLASSES:
                col_idx = ci + 1
                vals = out.iloc[:, col_idx].to_numpy(np.float32)
                thr = vals.mean() + cfg.rare_margin
                out.iloc[:, col_idx] = np.where(vals < thr, vals * cfg.rare_scale, vals)
    return out


def _metric_setup(labels_csv: Path, row_ids: pd.Series, cols: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    labels_wide = load_long_labels(labels_csv, cols).set_index("row_id")
    row_ids_str = row_ids.astype(str).to_numpy()
    present = np.array([rid in labels_wide.index for rid in row_ids_str], dtype=bool)
    aligned = labels_wide.reindex(row_ids_str[present])
    valid = [c for c in cols if c in aligned and aligned[c].nunique() > 1]
    valid_idx = np.array([cols.index(c) for c in valid], dtype=int)
    y_true = aligned[valid].to_numpy(np.float32)
    return present, valid_idx, valid, y_true


def _score_values(values: np.ndarray, present: np.ndarray, valid_idx: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    y_score = values[present][:, valid_idx]
    info: dict[str, float] = {"macro_auc": float(roc_auc_score(y_true, y_score, average="macro"))}
    # Top-k on all classes for matched rows, with true matrix reconstructed over valid columns only.
    true_full = np.zeros((int(present.sum()), values.shape[1]), dtype=np.float32)
    true_full[:, valid_idx] = y_true
    score_full = values[present]
    for k in (1, 3, 5, 10):
        info[f"top{k}_row_recall"] = float(topk_row_recall(score_full, true_full, k))
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--labels-csv", type=Path, required=True)
    ap.add_argument("--taxonomy-csv", type=Path)
    ap.add_argument("--reference-csv", type=Path, help="Optional scored/anchor submission for corr/MAE; defaults to baseline reconstruction")
    ap.add_argument("--proto-weights", default="0.56,0.58,0.60,0.62,0.64")
    ap.add_argument("--fake-boosts", default="0.04,0.08,0.12")
    ap.add_argument("--ctx-thrs", default="0.86,0.88,0.90")
    ap.add_argument("--ctx-boosts", default="0.10,0.15,0.20")
    ap.add_argument("--sed-rank-thrs", default="0.93,0.95,0.97")
    ap.add_argument("--sed-boosts", default="0.08,0.12,0.16")
    ap.add_argument("--rare-scales", default="0.85,0.90,0.95")
    ap.add_argument("--max-trials", type=int, default=800)
    ap.add_argument("--seed", type=int, default=20260513)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--write-top-csv", type=Path)
    args = ap.parse_args()

    proto = pd.read_csv(args.pred_dir / "submission_protossm.csv")
    sed = pd.read_csv(args.pred_dir / "submission_sed.csv")
    cols = [c for c in proto.columns if c != "row_id"]
    tax_df = pd.read_csv(args.taxonomy_csv).set_index("primary_label") if args.taxonomy_csv and args.taxonomy_csv.exists() else None
    present, valid_idx, valid_cols, y_true = _metric_setup(args.labels_csv, proto["row_id"], cols)

    baseline_cfg = GateConfig()
    baseline_df = apply_config(proto, sed, cols, tax_df, baseline_cfg)
    baseline_values = baseline_df[cols].to_numpy(np.float32)
    baseline_metrics = _score_values(baseline_values, present, valid_idx, y_true)

    if args.reference_csv:
        ref = pd.read_csv(args.reference_csv).set_index("row_id").loc[proto["row_id"]].reset_index()
        reference_values = ref[cols].to_numpy(np.float32)
    else:
        reference_values = baseline_values

    axes = [
        _parse_float_list(args.proto_weights),
        _parse_float_list(args.fake_boosts),
        _parse_float_list(args.ctx_thrs),
        _parse_float_list(args.ctx_boosts),
        _parse_float_list(args.sed_rank_thrs),
        _parse_float_list(args.sed_boosts),
        _parse_float_list(args.rare_scales),
    ]
    combos = list(itertools.product(*axes))
    rng = np.random.default_rng(args.seed)
    if len(combos) > args.max_trials:
        keep = rng.choice(len(combos), size=args.max_trials, replace=False)
        combos = [combos[int(i)] for i in keep]
    # Always include the exact baseline.
    combos.insert(0, (0.60, 0.08, 0.88, 0.15, 0.95, 0.12, 0.90))

    rows: list[dict[str, Any]] = []
    top_df: pd.DataFrame | None = None
    top_objective = -1e9
    for proto_w, fake_boost, ctx_thr, ctx_boost, sed_rank_thr, sed_boost, rare_scale in combos:
        cfg = GateConfig(
            proto_weight=proto_w,
            fake_boost=fake_boost,
            ctx_thr=ctx_thr,
            ctx_boost=ctx_boost,
            sed_rank_thr=sed_rank_thr,
            sed_boost=sed_boost,
            rare_scale=rare_scale,
        )
        df = apply_config(proto, sed, cols, tax_df, cfg)
        values = df[cols].to_numpy(np.float32)
        metrics = _score_values(values, present, valid_idx, y_true)
        corr = float(np.corrcoef(reference_values.ravel(), values.ravel())[0, 1])
        mae = float(np.mean(np.abs(reference_values - values)))
        max_abs = float(np.max(np.abs(reference_values - values)))
        # Penalize large anchor displacement; this dry-run label set is tiny/noisy.
        objective = metrics["macro_auc"] - 0.05 * max(0.0, mae - 0.015) - 0.01 * max(0.0, max_abs - 0.10)
        row = {
            "config": asdict(cfg),
            **metrics,
            "delta_auc_vs_baseline": float(metrics["macro_auc"] - baseline_metrics["macro_auc"]),
            "corr_vs_reference": corr,
            "mae_vs_reference": mae,
            "max_abs_vs_reference": max_abs,
            "objective": float(objective),
        }
        rows.append(row)
        if objective > top_objective:
            top_objective = objective
            top_df = df

    rows_sorted = sorted(rows, key=lambda r: r["objective"], reverse=True)
    result = {
        "pred_dir": str(args.pred_dir),
        "labels_csv": str(args.labels_csv),
        "taxonomy_csv": str(args.taxonomy_csv) if args.taxonomy_csv else None,
        "reference_csv": str(args.reference_csv) if args.reference_csv else None,
        "matched_rows": int(present.sum()),
        "valid_auc_classes": int(len(valid_cols)),
        "n_trials": int(len(rows)),
        "baseline": {"config": asdict(baseline_cfg), **baseline_metrics},
        "top": rows_sorted[: args.top_k],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if args.write_top_csv:
        if top_df is None:
            raise RuntimeError("no top candidate was produced")
        args.write_top_csv.parent.mkdir(parents=True, exist_ok=True)
        top_df.to_csv(args.write_top_csv, index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
