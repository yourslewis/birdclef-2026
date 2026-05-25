#!/usr/bin/env python3
"""Audit fixed BirdCLEF ensemble recipes from a manifest.

This is a repo-owned, no-submit ensemble workbench.  It loads a small manifest of
anchor/baseline/branch prediction CSVs, validates alignment and values, dedupes
exact matrices, builds fixed class-wise percentile-rank blends, evaluates local
train-soundscape overlap when labels are available, and reports group stability
against both the anchor and the nearest tied baseline recipe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from birdclef_public946_cache_summary import load_long_labels, topk_row_recall
from birdclef_public946_multi_sidecar_weight_grid import group_key


EPS = 1e-7


def repo_path(text: str | None, *, root: Path) -> Path | None:
    if text is None:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    return root / path


def prediction_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "row_id"]


def matrix_hash(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(values.astype(np.float64, copy=False))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def prediction_stats(values: np.ndarray) -> dict[str, Any]:
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p50": float(np.quantile(values, 0.50)),
        "p99": float(np.quantile(values, 0.99)),
    }


def validate_prediction_csv(
    name: str,
    path: Path,
    *,
    canonical_row_ids: pd.Series | None,
    canonical_cols: list[str] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"{name}: missing CSV {path}")
    df = pd.read_csv(path)
    if "row_id" not in df.columns:
        raise ValueError(f"{name}: {path} missing row_id column")
    if df["row_id"].duplicated().any():
        dupes = df.loc[df["row_id"].duplicated(), "row_id"].head(5).tolist()
        raise ValueError(f"{name}: duplicate row_id values, first={dupes}")

    cols = prediction_columns(df)
    if not cols:
        raise ValueError(f"{name}: no prediction columns")
    if canonical_cols is None:
        canonical_cols = cols
    missing = [c for c in canonical_cols if c not in df.columns]
    extra = [c for c in cols if c not in canonical_cols]
    if missing or extra:
        raise ValueError(
            f"{name}: column mismatch vs canonical; missing={missing[:8]} extra={extra[:8]}"
        )

    row_order_equal = True
    if canonical_row_ids is not None:
        cur = df["row_id"].astype(str)
        ref = canonical_row_ids.astype(str)
        if len(cur) != len(ref) or set(cur.tolist()) != set(ref.tolist()):
            cur_only = sorted(set(cur.tolist()) - set(ref.tolist()))[:5]
            ref_only = sorted(set(ref.tolist()) - set(cur.tolist()))[:5]
            raise ValueError(
                f"{name}: row_id set mismatch vs canonical; cur_only={cur_only} ref_only={ref_only}"
            )
        row_order_equal = bool(cur.reset_index(drop=True).equals(ref.reset_index(drop=True)))
        if not row_order_equal:
            df = df.set_index("row_id").loc[ref.tolist()].reset_index()

    df = df[["row_id", *canonical_cols]].copy()
    values = df[canonical_cols].to_numpy(np.float64)
    finite_mask = np.isfinite(values)
    bad_cells = int((~finite_mask).sum())
    if bad_cells:
        raise ValueError(f"{name}: non-finite prediction cells={bad_cells}")
    nonconstant = (np.nanmax(values, axis=0) - np.nanmin(values, axis=0)) > 0
    constant_cols = [canonical_cols[i] for i in np.flatnonzero(~nonconstant)]
    if constant_cols:
        raise ValueError(f"{name}: constant prediction columns={constant_cols[:12]}")

    info = {
        "name": name,
        "path": str(path),
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "rows": int(df.shape[0]),
        "class_columns": int(len(canonical_cols)),
        "row_order_equal_to_anchor": row_order_equal,
        "finite": True,
        "bad_cells": bad_cells,
        "nonconstant_columns": int(nonconstant.sum()),
        "constant_columns": constant_cols,
        "matrix_sha256": matrix_hash(values),
        "prob_stats": prediction_stats(values),
    }
    return df, info


def rank_values(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    values = np.clip(df[cols].to_numpy(np.float32), EPS, 1.0 - EPS)
    return pd.DataFrame(values).rank(axis=0, pct=True).to_numpy(np.float32)


def macro_auc_matrix(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    return_by_class: bool = False,
) -> tuple[float | None, int, list[float | None]]:
    if y_true.size == 0 or y_score.size == 0:
        return None, 0, []
    varying = (np.nanmax(y_true, axis=0) > np.nanmin(y_true, axis=0))
    if not np.any(varying):
        return None, 0, [None for _ in range(y_true.shape[1])] if return_by_class else []
    auc = float(roc_auc_score(y_true[:, varying], y_score[:, varying], average="macro"))
    if not return_by_class:
        return auc, int(varying.sum()), []
    by_class: list[float | None] = []
    for j in range(y_true.shape[1]):
        if not varying[j]:
            by_class.append(None)
        else:
            by_class.append(float(roc_auc_score(y_true[:, j], y_score[:, j])))
    return auc, int(varying.sum()), by_class


def lift_summary(values: list[float] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "q05": float(np.quantile(arr, 0.05)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
        "q95": float(np.quantile(arr, 0.95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p_gt_0": float(np.mean(arr > 0)),
    }


def local_metrics(
    values: np.ndarray,
    *,
    cols: list[str],
    row_ids: pd.Series,
    labels_wide: pd.DataFrame | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if labels_wide is None:
        return {"labels_available": False}, None
    pred = pd.DataFrame(values, columns=cols)
    pred.insert(0, "row_id", row_ids.astype(str).to_numpy())
    merged = pred.merge(labels_wide, on="row_id", suffixes=("_pred", "_true"))
    valid = [
        c for c in cols
        if f"{c}_pred" in merged
        and f"{c}_true" in merged
        and merged[f"{c}_true"].nunique() > 1
    ]
    out: dict[str, Any] = {
        "labels_available": True,
        "matched_rows": int(len(merged)),
        "valid_auc_classes": int(len(valid)),
    }
    if not valid:
        return out, None

    y_true_valid = merged[[f"{c}_true" for c in valid]].to_numpy(np.uint8)
    y_score_valid = merged[[f"{c}_pred" for c in valid]].to_numpy(np.float32)
    auc, valid_n, by_class = macro_auc_matrix(y_true_valid, y_score_valid, return_by_class=True)
    out["macro_auc"] = auc
    out["valid_auc_classes"] = valid_n

    pred_cols = [f"{c}_pred" for c in cols if f"{c}_pred" in merged]
    true_cols = [c.replace("_pred", "_true") for c in pred_cols]
    score_mat = merged[pred_cols].to_numpy(np.float32)
    true_mat = merged[true_cols].to_numpy(np.uint8)
    for k in (1, 3, 5, 10):
        out[f"top{k}_row_recall"] = topk_row_recall(score_mat, true_mat, k)

    per_class = {
        "columns": valid,
        "auc": {c: by_class[i] for i, c in enumerate(valid)},
        "matched_row_ids": merged["row_id"].astype(str).tolist(),
        "y_true": y_true_valid,
        "column_indices": np.array([cols.index(c) for c in valid], dtype=np.int64),
    }
    return out, per_class


def corr_mae(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    flat_a = a.ravel()
    flat_b = b.ravel()
    if float(np.std(flat_a)) == 0.0 or float(np.std(flat_b)) == 0.0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(flat_a, flat_b)[0, 1])
    return {
        "rank_corr": corr,
        "mae": float(np.mean(np.abs(a - b))),
        "max_abs": float(np.max(np.abs(a - b))),
    }


def bootstrap_lift(
    row_ids: pd.Series,
    y_true: np.ndarray,
    reference_values: np.ndarray,
    candidate_values: np.ndarray,
    *,
    group_mode: str,
    iters: int,
    seed: int,
) -> dict[str, Any]:
    if np.array_equal(reference_values, candidate_values):
        out = {"iters": int(iters), "valid_iters": int(iters), "group_mode": group_mode, "n_groups": 0}
        out.update(lift_summary(np.zeros(int(iters), dtype=np.float64)))
        return out
    groups = np.array([group_key(x, group_mode) for x in row_ids.astype(str).tolist()])
    unique_groups = np.array(sorted(set(groups.tolist())))
    by_group = {g: np.flatnonzero(groups == g) for g in unique_groups}
    rng = np.random.default_rng(seed)
    lifts: list[float] = []
    for _ in range(int(iters)):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([by_group[g] for g in sampled])
        ref_auc, _, _ = macro_auc_matrix(y_true[idx], reference_values[idx])
        cand_auc, _, _ = macro_auc_matrix(y_true[idx], candidate_values[idx])
        if ref_auc is None or cand_auc is None:
            continue
        lifts.append(float(cand_auc - ref_auc))
    out = {
        "iters": int(iters),
        "valid_iters": int(len(lifts)),
        "group_mode": group_mode,
        "n_groups": int(len(unique_groups)),
    }
    out.update(lift_summary(lifts))
    return out


def leave_one_group_lift(
    row_ids: pd.Series,
    y_true: np.ndarray,
    reference_values: np.ndarray,
    candidate_values: np.ndarray,
    *,
    group_mode: str,
    max_detail: int,
) -> dict[str, Any]:
    groups = np.array([group_key(x, group_mode) for x in row_ids.astype(str).tolist()])
    unique_groups = np.array(sorted(set(groups.tolist())))
    rows: list[dict[str, Any]] = []
    for group in unique_groups:
        idx = np.flatnonzero(groups != group)
        if idx.size == 0:
            continue
        ref_auc, _, _ = macro_auc_matrix(y_true[idx], reference_values[idx])
        cand_auc, _, _ = macro_auc_matrix(y_true[idx], candidate_values[idx])
        if ref_auc is None or cand_auc is None:
            continue
        rows.append({
            "held_out_group": str(group),
            "kept_rows": int(idx.size),
            "reference_auc": float(ref_auc),
            "candidate_auc": float(cand_auc),
            "lift": float(cand_auc - ref_auc),
        })
    lifts = np.array([r["lift"] for r in rows], dtype=np.float64)
    out = {
        "group_mode": group_mode,
        "n_groups": int(len(unique_groups)),
        "valid_groups": int(len(rows)),
    }
    out.update(lift_summary(lifts))
    out["worst_groups"] = sorted(rows, key=lambda r: r["lift"])[:max_detail]
    out["best_groups"] = sorted(rows, key=lambda r: r["lift"], reverse=True)[:max_detail]
    return out


def build_recipe_values(
    recipe: dict[str, Any],
    *,
    member_dfs: dict[str, pd.DataFrame],
    member_ranks: dict[str, np.ndarray],
    cols: list[str],
) -> tuple[np.ndarray, dict[str, float], list[str]]:
    rtype = recipe.get("type", "rank_blend")
    if rtype == "member":
        member = str(recipe["member"])
        if member not in member_dfs:
            raise ValueError(f"recipe {recipe.get('name')}: unknown member {member!r}")
        values = member_dfs[member][cols].to_numpy(np.float32)
        return values, {member: 1.0}, [member]
    if rtype != "rank_blend":
        raise ValueError(f"recipe {recipe.get('name')}: unsupported type {rtype!r}")

    weights = {str(k): float(v) for k, v in recipe.get("weights", {}).items()}
    if not weights:
        raise ValueError(f"recipe {recipe.get('name')}: rank_blend requires weights")
    unknown = [name for name in weights if name not in member_ranks]
    if unknown:
        raise ValueError(f"recipe {recipe.get('name')}: unknown weighted members {unknown}")
    total = float(sum(weights.values()))
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"recipe {recipe.get('name')}: weights must sum to 1.0, got {total}")
    first = next(iter(member_ranks.values()))
    values = np.zeros_like(first, dtype=np.float32)
    for name, weight in weights.items():
        values += float(weight) * member_ranks[name]
    return values.astype(np.float32), weights, list(weights)


def exact_dedupe(named_values: dict[str, np.ndarray]) -> dict[str, Any]:
    groups: dict[str, list[str]] = {}
    for name, values in named_values.items():
        groups.setdefault(matrix_hash(values), []).append(name)
    return {
        "total_matrices": int(len(named_values)),
        "unique_matrices": int(len(groups)),
        "groups": [
            {"sha256": h, "members": sorted(names), "n": len(names)}
            for h, names in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[1][0]))
        ],
        "duplicate_groups": [
            {"sha256": h, "members": sorted(names), "n": len(names)}
            for h, names in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[1][0]))
            if len(names) > 1
        ],
    }


def gate_recipe(
    recipe_name: str,
    comparisons: dict[str, Any],
    local: dict[str, Any],
    gates: dict[str, Any],
    *,
    allow_submit_approval: bool,
    anchor_name: str,
    baseline_name: str,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    if recipe_name in {anchor_name, baseline_name}:
        return {"eligible_for_submission": False, "checks": {}, "reason": "control recipe"}

    anchor = comparisons.get(anchor_name, {})
    baseline = comparisons.get(baseline_name, {})
    checks["matched_rows_min"] = int(local.get("matched_rows") or 0) >= int(gates.get("matched_rows_min", 190))
    checks["valid_classes_min"] = int(local.get("valid_auc_classes") or 0) >= int(gates.get("valid_classes_min", 60))
    checks["lift_vs_anchor_min"] = float(anchor.get("macro_auc_lift") or 0.0) >= float(gates.get("lift_vs_anchor_min", 0.0060))
    checks["lift_vs_baseline_min"] = float(baseline.get("macro_auc_lift") or 0.0) >= float(gates.get("lift_vs_baseline_min", 0.0010))

    site_boot = anchor.get("bootstrap", {}).get("site", {})
    file_boot = anchor.get("bootstrap", {}).get("file", {})
    site_loo = anchor.get("leave_one", {}).get("site", {})
    file_loo = anchor.get("leave_one", {}).get("file", {})
    checks["site_bootstrap_q05_min"] = float(site_boot.get("q05") or 0.0) >= float(gates.get("site_bootstrap_q05_min", 0.0030))
    checks["file_bootstrap_q05_min"] = float(file_boot.get("q05") or 0.0) >= float(gates.get("file_bootstrap_q05_min", 0.0015))
    checks["leave_one_site_min"] = float(site_loo.get("min") or 0.0) >= float(gates.get("leave_one_site_min", 0.0030))
    checks["leave_one_file_q05_min"] = float(file_loo.get("q05") or 0.0) >= float(gates.get("leave_one_file_q05_min", 0.0010))
    checks["leave_one_file_p_gt_0_min"] = float(file_loo.get("p_gt_0") or 0.0) >= float(gates.get("leave_one_file_p_gt_0_min", 0.90))

    passed = all(checks.values())
    return {
        "eligible_for_submission": bool(passed),
        "submit_approved": bool(allow_submit_approval and passed),
        "checks": checks,
        "reason": "all gates passed" if passed else "one or more promotion gates failed",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--labels-csv", type=Path, help="Override manifest labels_csv")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--bootstrap-iters", type=int, help="Override manifest bootstrap iters")
    ap.add_argument("--bootstrap-seed", type=int, default=42)
    ap.add_argument("--leave-one-detail", type=int, default=999)
    ap.add_argument("--emit-candidate-csvs", action="store_true")
    ap.add_argument("--allow-submit-approval", action="store_true", help="Still requires all manifest gates; default keeps submit_approved=false")
    args = ap.parse_args()

    root = Path.cwd().resolve()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_path = args.labels_csv or repo_path(manifest.get("labels_csv"), root=root)
    bootstrap_cfg = manifest.get("bootstrap", {})
    bootstrap_iters = int(args.bootstrap_iters if args.bootstrap_iters is not None else bootstrap_cfg.get("iters", 1000))
    group_modes = list(bootstrap_cfg.get("groups", ["site", "file"]))

    members_cfg = manifest.get("members", {})
    if not members_cfg:
        raise ValueError("manifest has no members")
    anchor_member = str(manifest.get("anchor"))
    baseline_member = str(manifest.get("baseline"))
    if anchor_member not in members_cfg or baseline_member not in members_cfg:
        raise ValueError("manifest anchor/baseline must refer to member keys")

    member_dfs: dict[str, pd.DataFrame] = {}
    member_infos: dict[str, Any] = {}
    canonical_row_ids: pd.Series | None = None
    canonical_cols: list[str] | None = None
    for name, cfg in members_cfg.items():
        path = repo_path(str(cfg["path"]), root=root)
        assert path is not None
        df, info = validate_prediction_csv(
            name,
            path,
            canonical_row_ids=canonical_row_ids,
            canonical_cols=canonical_cols,
        )
        if canonical_row_ids is None:
            canonical_row_ids = df["row_id"].astype(str)
            canonical_cols = prediction_columns(df)
        member_dfs[name] = df
        info.update({k: v for k, v in cfg.items() if k != "path"})
        member_infos[name] = info

    assert canonical_row_ids is not None and canonical_cols is not None
    cols = canonical_cols
    row_ids = member_dfs[anchor_member]["row_id"].astype(str)
    member_ranks = {name: rank_values(df, cols) for name, df in member_dfs.items()}
    member_values = {name: df[cols].to_numpy(np.float32) for name, df in member_dfs.items()}

    labels_wide: pd.DataFrame | None = None
    if labels_path is not None and labels_path.exists():
        labels_wide = load_long_labels(labels_path, cols)

    recipe_values: dict[str, np.ndarray] = {}
    recipe_infos: list[dict[str, Any]] = []
    recipe_sources: dict[str, Any] = {}
    recipe_local_aux: dict[str, dict[str, Any] | None] = {}
    for recipe in manifest.get("recipes", []):
        name = str(recipe["name"])
        values, weights, source_members = build_recipe_values(
            recipe,
            member_dfs=member_dfs,
            member_ranks=member_ranks,
            cols=cols,
        )
        recipe_values[name] = values
        recipe_sources[name] = {"weights": weights, "source_members": source_members, "type": recipe.get("type", "rank_blend")}
        local, aux = local_metrics(values, cols=cols, row_ids=row_ids, labels_wide=labels_wide)
        recipe_local_aux[name] = aux
        recipe_infos.append({
            "name": name,
            "type": recipe.get("type", "rank_blend"),
            "description": recipe.get("description"),
            "source_members": source_members,
            "weights": weights,
            "rows": int(values.shape[0]),
            "class_columns": int(values.shape[1]),
            "prediction_stats": prediction_stats(values),
            "matrix_sha256": matrix_hash(values),
            "local_metrics": local,
        })

    recipe_names = [r["name"] for r in recipe_infos]
    anchor_recipe = str(manifest.get("anchor_recipe", "anchor_only"))
    baseline_recipe = str(manifest.get("baseline_recipe", "v616_baseline"))
    if anchor_recipe not in recipe_values or baseline_recipe not in recipe_values:
        raise ValueError(f"anchor_recipe/baseline_recipe must exist in recipes; got {anchor_recipe}, {baseline_recipe}")

    matched_context: dict[str, Any] | None = None
    if labels_wide is not None:
        base_df = pd.DataFrame({"row_id": row_ids})
        merged = base_df.merge(labels_wide, on="row_id", how="inner")
        valid_cols = [c for c in cols if c in merged.columns and merged[c].nunique() > 1]
        matched_row_ids = merged["row_id"].astype(str)
        matched_idx = np.array([row_ids.tolist().index(x) for x in matched_row_ids.tolist()], dtype=np.int64)
        valid_idx = np.array([cols.index(c) for c in valid_cols], dtype=np.int64)
        y_true = merged[valid_cols].to_numpy(np.uint8)
        matched_context = {
            "row_ids": matched_row_ids,
            "idx": matched_idx,
            "valid_idx": valid_idx,
            "valid_cols": valid_cols,
            "y_true": y_true,
        }

    local_by_recipe = {r["name"]: r["local_metrics"] for r in recipe_infos}
    for info in recipe_infos:
        name = info["name"]
        comparisons: dict[str, Any] = {}
        for ref_name in (anchor_recipe, baseline_recipe):
            ref_values = recipe_values[ref_name]
            cand_values = recipe_values[name]
            comp: dict[str, Any] = corr_mae(ref_values, cand_values)
            ref_auc = local_by_recipe[ref_name].get("macro_auc")
            cand_auc = local_by_recipe[name].get("macro_auc")
            comp["macro_auc_lift"] = None if ref_auc is None or cand_auc is None else float(cand_auc - ref_auc)

            # Per-class lift summary on the matched valid columns.
            ref_aux = recipe_local_aux.get(ref_name)
            cand_aux = recipe_local_aux.get(name)
            if ref_aux and cand_aux:
                shared = [c for c in cand_aux["columns"] if c in ref_aux["auc"] and ref_aux["auc"].get(c) is not None]
                lifts = [float(cand_aux["auc"][c] - ref_aux["auc"][c]) for c in shared if cand_aux["auc"].get(c) is not None]
                comp["per_class_auc_lift_summary"] = lift_summary(lifts)
                comp["worst_class_lifts"] = [
                    {"class": c, "lift": float(cand_aux["auc"][c] - ref_aux["auc"][c])}
                    for c in sorted(shared, key=lambda c: float(cand_aux["auc"][c] - ref_aux["auc"][c]))[:12]
                    if cand_aux["auc"].get(c) is not None
                ]
                comp["best_class_lifts"] = [
                    {"class": c, "lift": float(cand_aux["auc"][c] - ref_aux["auc"][c])}
                    for c in sorted(shared, key=lambda c: float(cand_aux["auc"][c] - ref_aux["auc"][c]), reverse=True)[:12]
                    if cand_aux["auc"].get(c) is not None
                ]

            if matched_context is not None:
                idx = matched_context["idx"]
                valid_idx = matched_context["valid_idx"]
                y_true = matched_context["y_true"]
                ref_mat = ref_values[idx][:, valid_idx]
                cand_mat = cand_values[idx][:, valid_idx]
                comp["bootstrap"] = {}
                comp["leave_one"] = {}
                for group_mode in group_modes:
                    comp["bootstrap"][group_mode] = bootstrap_lift(
                        matched_context["row_ids"],
                        y_true,
                        ref_mat,
                        cand_mat,
                        group_mode=group_mode,
                        iters=bootstrap_iters,
                        seed=args.bootstrap_seed,
                    )
                    comp["leave_one"][group_mode] = leave_one_group_lift(
                        matched_context["row_ids"],
                        y_true,
                        ref_mat,
                        cand_mat,
                        group_mode=group_mode,
                        max_detail=args.leave_one_detail,
                    )
            comparisons[ref_name] = comp
        info["comparisons"] = comparisons
        info["promotion_gate"] = gate_recipe(
            name,
            comparisons,
            info["local_metrics"],
            manifest.get("promotion_gates", {}),
            allow_submit_approval=args.allow_submit_approval and bool(manifest.get("allow_submit_approval", False)),
            anchor_name=anchor_recipe,
            baseline_name=baseline_recipe,
        )

    if args.emit_candidate_csvs:
        cand_dir = output_dir / "candidate_csvs"
        cand_dir.mkdir(parents=True, exist_ok=True)
        for name, values in recipe_values.items():
            out_df = pd.DataFrame(values.astype(np.float32), columns=cols)
            out_df.insert(0, "row_id", row_ids.to_numpy())
            out_df.to_csv(cand_dir / f"{name}.csv", index=False)

    candidates = [r for r in recipe_infos if r["name"] not in {anchor_recipe, baseline_recipe}]
    best_vs_baseline = sorted(
        candidates,
        key=lambda r: (r.get("comparisons", {}).get(baseline_recipe, {}).get("macro_auc_lift") if r.get("comparisons", {}).get(baseline_recipe, {}).get("macro_auc_lift") is not None else -1e9),
        reverse=True,
    )
    approved = [r for r in recipe_infos if r.get("promotion_gate", {}).get("submit_approved")]
    readiness = {
        "submit_approved": bool(approved),
        "approved_recipes": [r["name"] for r in approved],
        "allow_submit_approval": bool(args.allow_submit_approval and manifest.get("allow_submit_approval", False)),
        "best_recipe_vs_baseline": best_vs_baseline[0]["name"] if best_vs_baseline else None,
        "reason": "no recipe cleared gates or submit approval was disabled" if not approved else "one or more recipes cleared all gates with approval enabled",
    }

    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "manifest_name": manifest.get("name"),
        "labels_csv": str(labels_path) if labels_path else None,
        "labels_loaded": labels_wide is not None,
        "bootstrap_iters": bootstrap_iters,
        "bootstrap_groups": group_modes,
        "anchor_member": anchor_member,
        "baseline_member": baseline_member,
        "anchor_recipe": anchor_recipe,
        "baseline_recipe": baseline_recipe,
        "members": member_infos,
        "member_dedupe": exact_dedupe(member_values),
        "recipe_dedupe": exact_dedupe(recipe_values),
        "recipes": recipe_infos,
        "readiness": readiness,
    }
    out_json = output_dir / "ensemble_strategy_audit.json"
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output_json": str(out_json),
        "labels_loaded": result["labels_loaded"],
        "submit_approved": readiness["submit_approved"],
        "best_recipe_vs_baseline": readiness["best_recipe_vs_baseline"],
        "recipe_summary": [
            {
                "name": r["name"],
                "macro_auc": r["local_metrics"].get("macro_auc"),
                "lift_vs_anchor": r["comparisons"][anchor_recipe].get("macro_auc_lift"),
                "lift_vs_baseline": r["comparisons"][baseline_recipe].get("macro_auc_lift"),
                "submit_approved": r["promotion_gate"].get("submit_approved", False),
            }
            for r in recipe_infos
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
