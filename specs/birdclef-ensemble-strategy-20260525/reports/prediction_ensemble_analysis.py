#!/usr/bin/env python3
"""Report-only prediction artifact comparison for BirdCLEF ensemble strategy."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def parse_seconds(value: Any) -> int:
    text = str(value)
    if ":" in text:
        parts = [int(float(p)) for p in text.split(":" )]
        sec = parts[-1]
        if len(parts) > 1:
            sec += 60 * parts[-2]
        if len(parts) > 2:
            sec += 3600 * parts[-3]
        return sec
    return int(float(text))


def load_long_labels(labels_csv: Path, columns: list[str]) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv)
    labels["row_id"] = (
        labels["filename"].astype(str).str.replace(".ogg", "", regex=False)
        + "_"
        + labels["end"].map(parse_seconds).astype(str)
    )
    unique_rows = labels["row_id"].drop_duplicates().sort_values()
    wide = pd.DataFrame(0, index=unique_rows, columns=columns, dtype=np.uint8)
    for _, row in labels.iterrows():
        row_id = row["row_id"]
        for label in str(row.get("primary_label", "")).split(";"):
            if label in wide.columns:
                wide.loc[row_id, label] = 1
    return wide.reset_index().rename(columns={"index": "row_id"})


def macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float | None, int]:
    aucs: list[float] = []
    for j in range(y_true.shape[1]):
        col = y_true[:, j]
        if col.min() == col.max():
            continue
        aucs.append(float(roc_auc_score(col, y_score[:, j])))
    return (float(np.mean(aucs)) if aucs else None, len(aucs))


def topk_row_recall(score_mat: np.ndarray, true_mat: np.ndarray, k: int) -> float:
    kth = min(k, score_mat.shape[1] - 1)
    top = np.argpartition(-score_mat, kth=kth, axis=1)[:, :k]
    hits = []
    for i in range(len(score_mat)):
        true_idx = np.flatnonzero(true_mat[i] > 0)
        hits.append(bool(len(true_idx) and np.intersect1d(true_idx, top[i]).size))
    return float(np.mean(hits)) if hits else float("nan")


def rank_values(values: np.ndarray) -> np.ndarray:
    return pd.DataFrame(np.clip(values.astype(np.float32), 1e-7, 1 - 1e-7)).rank(axis=0, pct=True).to_numpy(np.float32)


def flat_corr(a: np.ndarray, b: np.ndarray) -> float:
    av = a.reshape(-1)
    bv = b.reshape(-1)
    if np.std(av) == 0 or np.std(bv) == 0:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def sha_values(values: np.ndarray, row_ids: list[str], cols: list[str]) -> str:
    h = hashlib.sha256()
    for rid in row_ids:
        h.update(str(rid).encode())
        h.update(b"\0")
    for col in cols:
        h.update(str(col).encode())
        h.update(b"\0")
    h.update(np.ascontiguousarray(values.astype(np.float32)).tobytes())
    return h.hexdigest()


def classify_path(path: Path) -> str:
    s = str(path)
    name = path.stem
    if "sidecar_grid_inputs" in s:
        return name
    if "v616-anchored-jung21-sed-blend" in s:
        return "v616_" + name
    parent = path.parent.name
    if "source_output_audit" in s:
        # Strip long user/kernel prefix but retain branch file.
        return name.replace("__", ":")
    return name


def load_candidate(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(path), "name": classify_path(path)}
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        out["usable"] = False
        out["reason"] = f"read_error:{type(exc).__name__}:{exc}"
        return out
    out["rows"] = int(df.shape[0])
    out["cols"] = int(df.shape[1])
    if "row_id" not in df.columns:
        out["usable"] = False
        out["reason"] = "missing_row_id"
        return out
    pred_cols = [c for c in df.columns if c != "row_id"]
    if not pred_cols:
        out["usable"] = False
        out["reason"] = "no_prediction_columns"
        return out
    vals = df[pred_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    bad = int((~np.isfinite(vals)).sum())
    out.update({
        "bad_values": bad,
        "unique_rows": int(df["row_id"].nunique()),
        "min": None if bad else float(np.min(vals)),
        "max": None if bad else float(np.max(vals)),
        "mean": None if bad else float(np.mean(vals)),
        "uniq_round_10k": int(len(np.unique(np.round(vals[np.isfinite(vals)] * 10000)))) if vals.size else 0,
        "head_ids": df["row_id"].astype(str).head(3).tolist(),
    })
    if bad:
        out["usable"] = False
        out["reason"] = f"nonfinite_values:{bad}"
        return out
    if df.shape[0] != 240 or df.shape[1] != 235:
        out["usable"] = False
        out["reason"] = f"wrong_shape:{df.shape[0]}x{df.shape[1]}"
        return out
    if out["unique_rows"] != df.shape[0]:
        out["usable"] = False
        out["reason"] = "duplicate_row_id"
        return out
    if out["uniq_round_10k"] <= 5:
        out["usable"] = False
        out["reason"] = "constant_or_near_constant"
        return out
    out["usable"] = True
    out["df"] = df[["row_id", *pred_cols]]
    out["pred_cols"] = pred_cols
    return out


def main() -> int:
    repo = Path.cwd()
    labels_csv = Path("/Volumes/ExternalSSD/data/workspace_don/kaggle_birdclef2026/data/train_soundscapes_labels.csv")
    patterns = [
        "artifacts/anchored_blend_audit/sidecar_grid_inputs_20260525T0200Z/*.csv",
        "artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/*.csv",
        "artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/*.csv",
    ]
    paths: list[Path] = []
    for pat in patterns:
        paths.extend(sorted(repo.glob(pat)))
    paths = sorted(set(paths))

    loaded = [load_candidate(p) for p in paths]
    usable = [x for x in loaded if x.get("usable")]
    anchor_path = repo / "artifacts/anchored_blend_audit/sidecar_grid_inputs_20260525T0200Z/samejima_visual_anchor.csv"
    anchor_rec = next((x for x in usable if Path(x["path"]) == anchor_path), None)
    if anchor_rec is None:
        raise RuntimeError("anchor not found/usable")
    anchor_df = anchor_rec["df"]
    cols = [c for c in anchor_df.columns if c != "row_id"]
    row_ids = anchor_df["row_id"].astype(str).tolist()
    labels_wide = load_long_labels(labels_csv, cols)
    label_by_row = labels_wide.set_index("row_id")
    matched_ids = [rid for rid in row_ids if rid in label_by_row.index]
    matched_idx = np.array([i for i, rid in enumerate(row_ids) if rid in label_by_row.index], dtype=int)
    y_all = label_by_row.loc[matched_ids, cols].to_numpy(np.float32)
    valid_label_cols = np.array([j for j in range(y_all.shape[1]) if y_all[:, j].min() != y_all[:, j].max()], dtype=int)
    y = y_all[:, valid_label_cols]

    anchor_vals = anchor_df[cols].to_numpy(np.float32)
    anchor_rank = rank_values(anchor_vals)
    anchor_auc, anchor_valid = macro_auc(y, anchor_rank[matched_idx][:, valid_label_cols])

    by_hash: dict[str, str] = {}
    rows = []
    arrays: dict[str, np.ndarray] = {}
    ranks: dict[str, np.ndarray] = {}
    for rec in usable:
        df = rec["df"]
        if set(cols) - set(df.columns):
            rec["usable"] = False
            rec["reason"] = "missing_anchor_columns"
            continue
        try:
            aligned = df.set_index("row_id").loc[row_ids].reset_index()
        except Exception:
            rec["usable"] = False
            rec["reason"] = "row_ids_not_alignable_to_anchor"
            continue
        vals = aligned[cols].to_numpy(np.float32)
        rvals = rank_values(vals)
        digest = sha_values(vals, row_ids, cols)
        duplicate_of = by_hash.get(digest)
        if duplicate_of is None:
            by_hash[digest] = rec["name"]
        auc, valid = macro_auc(y, rvals[matched_idx][:, valid_label_cols])
        true_mat = y
        score_mat = rvals[matched_idx][:, valid_label_cols]
        row = {
            "name": rec["name"],
            "path": rec["path"],
            "hash12": digest[:12],
            "duplicate_of": duplicate_of,
            "rows": rec["rows"],
            "cols": rec["cols"],
            "min": rec["min"],
            "max": rec["max"],
            "mean": rec["mean"],
            "uniq_round_10k": rec["uniq_round_10k"],
            "prob_corr_vs_anchor": flat_corr(vals, anchor_vals),
            "rank_corr_vs_anchor": flat_corr(rvals, anchor_rank),
            "prob_mae_vs_anchor": float(np.mean(np.abs(vals - anchor_vals))),
            "rank_mae_vs_anchor": float(np.mean(np.abs(rvals - anchor_rank))),
            "max_abs_vs_anchor": float(np.max(np.abs(vals - anchor_vals))),
            "local_rank_auc": auc,
            "local_auc_lift_vs_anchor": None if auc is None or anchor_auc is None else float(auc - anchor_auc),
            "valid_auc_classes": int(valid),
            "matched_rows": int(len(matched_idx)),
            "top3_row_recall": topk_row_recall(score_mat, true_mat, 3),
        }
        rows.append(row)
        if duplicate_of is None:
            arrays[rec["name"]] = vals
            ranks[rec["name"]] = rvals

    # Select non-duplicate candidates with real train rows and sort by usefulness/diversity.
    unique_rows = [r for r in rows if r["duplicate_of"] is None]
    unique_rows.sort(key=lambda r: (
        -1 if r["local_rank_auc"] is None else -r["local_rank_auc"],
        r["rank_corr_vs_anchor"] if math.isfinite(r["rank_corr_vs_anchor"]) else 9,
    ))

    # Pairwise comparisons for the most relevant candidates: anchor, known sidecars, and top local-AUC unique outputs.
    preferred_names = [
        "samejima_visual_anchor",
        "sakur_visual",
        "jungchan_model21",
        "raunak_sed",
        "raunak_protossm",
        "samejima_sed",
        "samejima_protossm",
        "sakur_protossm",
        "jungchan_protossm",
        "v616_submission",
        "v616_submission_jung21_raw",
        "v616_submission_samejima_sed_raw",
    ]
    selected = []
    for name in preferred_names:
        if name in arrays and name not in selected:
            selected.append(name)
    for r in unique_rows[:20]:
        if r["name"] not in selected:
            selected.append(r["name"])
        if len(selected) >= 24:
            break
    pairwise = []
    for i, a in enumerate(selected):
        for b in selected[i + 1:]:
            pairwise.append({
                "a": a,
                "b": b,
                "prob_corr": flat_corr(arrays[a], arrays[b]),
                "rank_corr": flat_corr(ranks[a], ranks[b]),
                "prob_mae": float(np.mean(np.abs(arrays[a] - arrays[b]))),
                "rank_mae": float(np.mean(np.abs(ranks[a] - ranks[b]))),
            })
    pairwise.sort(key=lambda r: r["rank_corr"] if math.isfinite(r["rank_corr"]) else 9)

    rejected = [
        {k: v for k, v in rec.items() if k not in {"df", "pred_cols"}}
        for rec in loaded if not rec.get("usable")
    ]
    duplicate_rows = [r for r in rows if r["duplicate_of"] is not None]

    out = {
        "labels_csv": str(labels_csv),
        "n_paths_seen": len(paths),
        "n_usable_aligned": len(rows),
        "n_unique_prediction_matrices": len(unique_rows),
        "n_rejected_or_unusable": len(rejected),
        "anchor": {
            "name": "samejima_visual_anchor",
            "path": str(anchor_path),
            "local_rank_auc": anchor_auc,
            "valid_auc_classes": anchor_valid,
            "matched_rows": int(len(matched_idx)),
            "top3_row_recall": topk_row_recall(anchor_rank[matched_idx][:, valid_label_cols], y, 3),
        },
        "unique_candidates": unique_rows,
        "duplicates": duplicate_rows,
        "rejected_or_unusable": rejected,
        "selected_pairwise_diversity": pairwise,
    }
    out_path = repo / "specs/birdclef-ensemble-strategy-20260525/reports/prediction_ensemble_numeric_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")

    md_path = repo / "specs/birdclef-ensemble-strategy-20260525/reports/prediction_ensemble_numeric_tables.md"
    def fmt(x: Any, nd: int = 6) -> str:
        if x is None:
            return ""
        if isinstance(x, float):
            if not math.isfinite(x):
                return "nan"
            return f"{x:.{nd}f}"
        return str(x)
    lines = ["# Prediction ensemble numeric tables", "", f"Source JSON: `{out_path}`", "", "## Unique usable candidates (sorted by local rank AUC)", "", "| name | AUC | lift | rank corr vs anchor | rank MAE | prob MAE | top3 recall | duplicate? |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in unique_rows[:35]:
        lines.append(f"| `{r['name']}` | {fmt(r['local_rank_auc'])} | {fmt(r['local_auc_lift_vs_anchor'])} | {fmt(r['rank_corr_vs_anchor'])} | {fmt(r['rank_mae_vs_anchor'])} | {fmt(r['prob_mae_vs_anchor'])} | {fmt(r['top3_row_recall'])} | {r['duplicate_of'] or ''} |")
    lines += ["", "## Most different selected pairs (lowest rank correlation)", "", "| a | b | rank corr | rank MAE | prob MAE |", "|---|---|---:|---:|---:|"]
    for r in pairwise[:40]:
        lines.append(f"| `{r['a']}` | `{r['b']}` | {fmt(r['rank_corr'])} | {fmt(r['rank_mae'])} | {fmt(r['prob_mae'])} |")
    lines += ["", "## Duplicate matrices", "", "| name | duplicate_of | path |", "|---|---|---|"]
    for r in duplicate_rows[:80]:
        lines.append(f"| `{r['name']}` | `{r['duplicate_of']}` | `{r['path']}` |")
    lines += ["", "## Rejected/unusable CSVs", "", "| name | shape | reason | path |", "|---|---:|---|---|"]
    for r in rejected[:80]:
        lines.append(f"| `{r.get('name')}` | {r.get('rows')}x{r.get('cols')} | {r.get('reason')} | `{r.get('path')}` |")
    md_path.write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "wrote": [str(out_path), str(md_path)],
        "n_paths_seen": len(paths),
        "n_usable_aligned": len(rows),
        "n_unique_prediction_matrices": len(unique_rows),
        "n_rejected_or_unusable": len(rejected),
        "anchor_auc": anchor_auc,
        "top_unique": unique_rows[:8],
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
