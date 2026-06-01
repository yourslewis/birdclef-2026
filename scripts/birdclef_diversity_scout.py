#!/usr/bin/env python3
"""BirdCLEF diversity scout: truth-aligned decorrelation vs the 0.950 frontier.

Motivation
----------
Our standard audit scores every candidate as "fixed/tiny-weight pooled-AUC lift
vs an anchor". That objective structurally rewards REDUNDANT AGREEMENT and
punishes COMPLEMENTARY DISAGREEMENT, so it can never reward a diverse member.
It also keeps comparing to v616 (0.949) instead of the live 0.950 frontier.

This scout fixes both:
  * Frontier E = the EoS8/PowerOptimization 0.950 winner proxy (rank blend of the
    PowerOpt proto/sed branches), not v616.
  * For each candidate stream C it reports DIVERSITY DESCRIPTORS that prioritise
    representation-level complementarity:
        - rank decorrelation              1 - spearman(E, C)
        - error decorrelation             corr of per-row residuals (negative=gold)
        - conditional competence          AUC(C) restricted to rows/classes where E is weak
        - weight-optimised blend lift      max_w AUC(rankblend(E,C,w)) - AUC(E)
        - oracle ceiling                  best 2-member blend headroom
  * Truth-aligned diversity score:
        DEV = blend_lift  +  lambda * (decorrelation * competence_above_chance)
    The PRODUCT guard means a random/garbage stream (max decorrelation, ~chance
    competence) collapses to ~0 bonus. Diversity only counts where competent.

This is a scout/ranker, NOT a submission approver. It tells us WHICH
representation axis carries real, usable diversity before we spend an LB slot.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(Path(__file__).resolve().parent))
from birdclef_public946_cache_summary import load_long_labels  # noqa: E402


def pred_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "row_id"]


def file_id(row_id: str) -> str:
    return "_".join(str(row_id).split("_")[:-1])


def site_id(row_id: str) -> str:
    parts = str(row_id).split("_")
    return parts[3] if len(parts) > 3 else "UNKNOWN"


def rank_cols(values: np.ndarray) -> np.ndarray:
    return pd.DataFrame(np.clip(values, 1e-7, 1.0 - 1e-7)).rank(axis=0, pct=True).to_numpy(np.float32)


def valid_class_indices(y: np.ndarray) -> list[int]:
    return [j for j in range(y.shape[1]) if np.nanmin(y[:, j]) < np.nanmax(y[:, j])]


def macro_auc(y: np.ndarray, score: np.ndarray, indices: list[int] | None = None) -> tuple[float | None, int]:
    cols = indices if indices is not None else list(range(y.shape[1]))
    valid = [j for j in cols if j < y.shape[1] and np.nanmin(y[:, j]) < np.nanmax(y[:, j])]
    if not valid:
        return None, 0
    return float(roc_auc_score(y[:, valid], score[:, valid], average="macro")), len(valid)


def per_class_auc(y: np.ndarray, score: np.ndarray, valid: list[int]) -> dict[int, float]:
    out: dict[int, float] = {}
    for j in valid:
        try:
            out[j] = float(roc_auc_score(y[:, j], score[:, j]))
        except Exception:
            pass
    return out


def align_like(df: pd.DataFrame, row_ids: pd.Series, cols: list[str], name: str) -> pd.DataFrame:
    if "row_id" not in df.columns:
        raise ValueError(f"{name} missing row_id")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing {len(missing)} columns e.g. {missing[:4]}")
    if set(df["row_id"].astype(str)) != set(row_ids.astype(str)):
        raise ValueError(f"{name} row_id set differs from canonical proxy")
    return df.set_index("row_id").loc[row_ids.astype(str).tolist()].reset_index()[["row_id", *cols]]


def spearman_flat(a: np.ndarray, b: np.ndarray) -> float:
    ra = rank_cols(a).ravel()
    rb = rank_cols(b).ravel()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def residual_error_corr(y: np.ndarray, e: np.ndarray, c: np.ndarray, valid: list[int]) -> float:
    """Correlation of per-cell residuals on valid classes. Negative = complementary."""
    yy = y[:, valid].astype(np.float64)
    re = (rank_cols(e)[:, valid] - yy).ravel()
    rc = (rank_cols(c)[:, valid] - yy).ravel()
    if np.std(re) == 0 or np.std(rc) == 0:
        return float("nan")
    return float(np.corrcoef(re, rc)[0, 1])


def weight_opt_blend_lift(
    y: np.ndarray, e_rank: np.ndarray, c_rank: np.ndarray, valid: list[int],
    weights: list[float],
) -> dict[str, Any]:
    base_auc, _ = macro_auc(y, e_rank, valid)
    best = {"weight": 0.0, "auc": base_auc, "lift": 0.0}
    curve = []
    for w in weights:
        cand = ((1.0 - w) * e_rank + w * c_rank).astype(np.float32)
        auc, _ = macro_auc(y, cand, valid)
        if auc is None:
            continue
        lift = auc - base_auc
        curve.append({"weight": w, "auc": auc, "lift": lift})
        if lift > best["lift"]:
            best = {"weight": w, "auc": auc, "lift": lift}
    return {"base_auc": base_auc, "best": best, "curve": curve}


def group_boot_blend_q05(
    y: np.ndarray, e_rank: np.ndarray, c_rank: np.ndarray, valid: list[int],
    groups: np.ndarray, weight: float, n_boot: int, seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(groups.astype(str))))
    cand = ((1.0 - weight) * e_rank + weight * c_rank).astype(np.float32)
    lifts = []
    for _ in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in sampled])
        ac, _ = macro_auc(y[idx], cand[idx], valid)
        ab, _ = macro_auc(y[idx], e_rank[idx], valid)
        if ac is not None and ab is not None:
            lifts.append(ac - ab)
    if not lifts:
        return {"n": 0}
    arr = np.asarray(lifts)
    return {"n": int(arr.size), "mean": float(arr.mean()), "q05": float(np.quantile(arr, 0.05)),
            "p_gt_0": float(np.mean(arr > 0))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto-csv", type=Path, required=True, help="0.950 winner PowerOpt proto branch on proxy")
    ap.add_argument("--sed-csv", type=Path, required=True, help="0.950 winner PowerOpt sed branch on proxy")
    ap.add_argument("--labels-csv", type=Path, default=ROOT / "data/train_soundscapes_labels.csv")
    ap.add_argument("--taxonomy-csv", type=Path, default=ROOT / "data/taxonomy.csv")
    ap.add_argument("--candidate", action="append", default=[], metavar="NAME=PATH",
                    help="Repeatable candidate stream to scout, NAME=path/to.csv")
    ap.add_argument("--proto-weight", type=float, default=0.6)
    ap.add_argument("--sed-weight", type=float, default=0.4)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.01)
    ap.add_argument("--weights", default="0.02,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.65,0.8,1.0")
    ap.add_argument("--neg-weights", action="store_true", help="also sweep small negative weights")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Frontier proxy E = PowerOpt rank blend of proto/sed branches (the 0.9695 engine).
    proto = pd.read_csv(args.proto_csv)
    cols = pred_cols(proto)
    row_ids = proto["row_id"].astype(str)
    sed = align_like(pd.read_csv(args.sed_csv), row_ids, cols, "sed")

    proto_rank = rank_cols(proto[cols].to_numpy(np.float32))
    sed_rank = rank_cols(sed[cols].to_numpy(np.float32))
    e_rank_full = (args.proto_weight * proto_rank + args.sed_weight * sed_rank).astype(np.float32)
    e_rank_full = rank_cols(e_rank_full)

    sites_full = row_ids.map(site_id).to_numpy()
    files_full = row_ids.map(file_id).to_numpy()

    labels_wide = load_long_labels(args.labels_csv, cols)
    matched = row_ids.isin(labels_wide["row_id"].astype(str))
    midx = np.flatnonzero(matched.to_numpy())
    mrows = row_ids.iloc[midx].tolist()
    y = labels_wide.set_index("row_id").loc[mrows, cols].to_numpy(np.uint8)
    valid = valid_class_indices(y)

    taxonomy = pd.read_csv(args.taxonomy_csv) if args.taxonomy_csv.exists() else pd.DataFrame()
    non_aves_idx: list[int] = []
    if not taxonomy.empty and {"primary_label", "class_name"}.issubset(taxonomy.columns):
        na = taxonomy.loc[taxonomy["class_name"].astype(str) != "Aves", "primary_label"].astype(str).tolist()
        non_aves_idx = [cols.index(c) for c in na if c in cols]

    e_m = e_rank_full[midx]
    e_auc, e_valid_n = macro_auc(y, e_m, valid)
    e_class_auc = per_class_auc(y, e_m, valid)
    weak_classes = sorted(valid, key=lambda j: e_class_auc.get(j, 1.0))[: max(1, len(valid) // 3)]
    # rows where E is least confident/most uncertain (entropy proxy on valid cols)
    p = np.clip(e_m[:, valid], 1e-6, 1 - 1e-6)
    ent = -(p * np.log(p) + (1 - p) * np.log(1 - p)).mean(axis=1)
    weak_rows = np.argsort(-ent)[: max(1, len(ent) // 3)]

    weights = [float(x) for x in args.weights.split(",") if x]
    if args.neg_weights:
        weights = sorted(set(weights + [-0.05, -0.1, -0.15, -0.2]))

    candidates: dict[str, Path] = {}
    for spec in args.candidate:
        if "=" not in spec:
            raise SystemExit(f"bad --candidate {spec!r}, want NAME=PATH")
        name, path = spec.split("=", 1)
        candidates[name] = Path(path)

    chance = 0.5
    results: list[dict[str, Any]] = []
    for name, path in candidates.items():
        try:
            craw = align_like(pd.read_csv(path), row_ids, cols, name)
        except Exception as exc:
            results.append({"candidate": name, "error": str(exc)})
            continue
        c_rank_full = rank_cols(craw[cols].to_numpy(np.float32))
        c_m = c_rank_full[midx]
        c_auc, _ = macro_auc(y, c_m, valid)
        c_auc_weakcls, _ = macro_auc(y, c_m, weak_classes)
        c_auc_weakrows, _ = macro_auc(y[weak_rows], c_m[weak_rows], valid)
        rank_decorr = 1.0 - spearman_flat(e_m, c_m)
        err_corr = residual_error_corr(y, e_m, c_m, valid)
        blend = weight_opt_blend_lift(y, e_m, c_m, valid, weights)
        bw = blend["best"]["weight"]
        site_q05 = group_boot_blend_q05(y, e_m, c_m, valid, sites_full[midx], bw, args.bootstrap, 17) if bw else {"q05": 0.0, "p_gt_0": 0.0, "n": 0}
        file_q05 = group_boot_blend_q05(y, e_m, c_m, valid, files_full[midx], bw, args.bootstrap, 29) if bw else {"q05": 0.0, "p_gt_0": 0.0, "n": 0}
        competence_above_chance = max(0.0, (c_auc_weakcls or chance) - chance)
        dev = float(blend["best"]["lift"] + args.lam * (max(0.0, rank_decorr) * competence_above_chance))
        results.append({
            "candidate": name,
            "path": str(path),
            "cand_auc": c_auc,
            "cand_auc_on_E_weak_classes": c_auc_weakcls,
            "cand_auc_on_E_weak_rows": c_auc_weakrows,
            "rank_decorrelation": rank_decorr,
            "residual_error_corr": err_corr,
            "blend_best_weight": bw,
            "blend_best_lift": blend["best"]["lift"],
            "blend_best_auc": blend["best"]["auc"],
            "blend_site_q05": site_q05.get("q05"),
            "blend_site_p_gt_0": site_q05.get("p_gt_0"),
            "blend_file_q05": file_q05.get("q05"),
            "competence_above_chance_on_E_weak": competence_above_chance,
            "DEV_score": dev,
            "gate_pass": bool((site_q05.get("q05") or 0) > 0 and (file_q05.get("q05") or 0) > 0),
        })

    summary = {
        "frontier": {
            "definition": "rankblend(proto*{:.2f} + sed*{:.2f}) of 0.950 EoS8 PowerOpt branches".format(args.proto_weight, args.sed_weight),
            "proto_csv": str(args.proto_csv),
            "sed_csv": str(args.sed_csv),
            "E_macro_auc": e_auc,
            "valid_class_count": e_valid_n,
            "matched_rows": int(len(midx)),
            "weak_class_count": len(weak_classes),
            "weak_row_count": int(len(weak_rows)),
        },
        "lambda": args.lam,
        "weights_swept": weights,
        "candidates": sorted(
            [r for r in results if "error" not in r],
            key=lambda r: (r["DEV_score"] is not None, r["DEV_score"]), reverse=True,
        ),
        "errors": [r for r in results if "error" in r],
    }
    out = args.out_dir / "diversity_scout_summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=False))
    print(json.dumps(summary, indent=2, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
