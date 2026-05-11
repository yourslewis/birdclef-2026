#!/usr/bin/env python3
"""Apply Kaggle taxon-max gate to a row-level teacher-cache NPZ.

This creates a reusable pseudo-label cache from an existing teacher cache while
preserving row_ids/labels. It intentionally mirrors the kernel implementation:
for every taxonomy class group, compute row-wise max evidence in that group and
multiply all group species by ``max(floor, evidence) ** alpha``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover
    roc_auc_score = None


def macro_auc(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    vals: list[float] = []
    for j in range(y.shape[1]):
        col = y[:, j]
        if float(col.min()) == float(col.max()):
            continue
        if roc_auc_score is None:
            continue
        try:
            vals.append(float(roc_auc_score(col, p[:, j])))
        except Exception:
            pass
    return {"macro_auc": float(np.mean(vals)) if vals else None, "valid_classes": len(vals)}


def row_id_from_label_row(row: pd.Series) -> str:
    end = str(row["end"])
    if ":" in end:
        parts = [int(x) for x in end.split(":")]
        sec = parts[-1] + 60 * parts[-2] + (3600 * parts[-3] if len(parts) >= 3 else 0)
    else:
        sec = int(float(end))
    return f"{Path(str(row['filename'])).stem}_{sec}"


def build_truth(labels_csv: Path, row_ids: np.ndarray, labels: list[str]) -> np.ndarray | None:
    if not labels_csv:
        return None
    if not labels_csv.exists():
        return None
    labels_df = pd.read_csv(labels_csv)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    by_row: dict[str, list[str]] = {}
    for _, row in labels_df.iterrows():
        rid = row_id_from_label_row(row)
        vals = [x for x in str(row["primary_label"]).split(";") if x]
        by_row[rid] = vals
    y = np.zeros((len(row_ids), len(labels)), dtype=np.float32)
    for i, rid in enumerate(row_ids.astype(str)):
        for label in by_row.get(str(rid), []):
            j = label_to_idx.get(str(label))
            if j is not None:
                y[i, j] = 1.0
    return y


def taxon_max_gate(probs: np.ndarray, labels: list[str], taxonomy_csv: Path, floor: float, alpha: float) -> tuple[np.ndarray, dict[str, int]]:
    taxonomy = pd.read_csv(taxonomy_csv, dtype={"primary_label": str})
    class_map = {str(k): str(v) for k, v in taxonomy.set_index("primary_label")["class_name"].to_dict().items()}
    label_groups = np.array([class_map.get(str(label), "Unknown") for label in labels], dtype=object)
    out = probs.copy().astype(np.float32)
    group_sizes: dict[str, int] = {}
    for group in sorted(set(label_groups.tolist())):
        cols = np.where(label_groups == group)[0]
        if len(cols) == 0:
            continue
        group_sizes[str(group)] = int(len(cols))
        evidence = probs[:, cols].max(axis=1, keepdims=True)
        mult = np.maximum(float(floor), evidence) ** float(alpha)
        out[:, cols] = out[:, cols] * mult
    return np.clip(out, 1e-8, 1.0 - 1e-8).astype(np.float32), group_sizes


def threshold_stats(probs: np.ndarray) -> dict[str, Any]:
    pos_thresholds = [0.50, 0.70, 0.80, 0.90, 0.95, 0.98]
    neg_thresholds = [0.01, 0.02, 0.05]
    return {
        "positive_counts": {str(t): int((probs >= t).sum()) for t in pos_thresholds},
        "positive_rows": {str(t): int(((probs >= t).sum(axis=1) > 0).sum()) for t in pos_thresholds},
        "negative_counts": {str(t): int((probs <= t).sum()) for t in neg_thresholds},
        "prob_stats": {
            "min": float(probs.min()),
            "max": float(probs.max()),
            "mean": float(probs.mean()),
            "p95": float(np.quantile(probs, 0.95)),
            "p99": float(np.quantile(probs, 0.99)),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-npz", type=Path, required=True)
    ap.add_argument("--taxonomy", type=Path, required=True)
    ap.add_argument("--labels-csv", type=Path)
    ap.add_argument("--output-npz", type=Path, required=True)
    ap.add_argument("--summary-json", type=Path, required=True)
    ap.add_argument("--floor", type=float, default=0.30)
    ap.add_argument("--alpha", type=float, default=0.50)
    ap.add_argument("--source-name", default="teacher_cache")
    args = ap.parse_args()

    z = np.load(args.pred_npz, allow_pickle=True)
    row_ids = z["row_ids"].astype(str)
    labels = [str(x) for x in z["labels"].astype(str).tolist()]
    probs = z["probs"].astype(np.float32)
    gated, group_sizes = taxon_max_gate(probs, labels, args.taxonomy, args.floor, args.alpha)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        row_ids=row_ids,
        labels=np.array(labels),
        probs=gated,
        source_probs=probs,
        floor=np.array(args.floor, dtype=np.float32),
        alpha=np.array(args.alpha, dtype=np.float32),
    )

    y = build_truth(args.labels_csv, row_ids, labels) if args.labels_csv else None
    summary: dict[str, Any] = {
        "status": "complete",
        "source_name": args.source_name,
        "pred_npz": str(args.pred_npz),
        "output_npz": str(args.output_npz),
        "taxonomy": str(args.taxonomy),
        "labels_csv": str(args.labels_csv) if args.labels_csv else None,
        "floor": float(args.floor),
        "alpha": float(args.alpha),
        "n_rows": int(len(row_ids)),
        "n_classes": int(len(labels)),
        "group_sizes": group_sizes,
        "baseline": threshold_stats(probs),
        "gated": threshold_stats(gated),
        "mean_abs_delta": float(np.mean(np.abs(gated - probs))),
        "flat_corr": float(np.corrcoef(probs.reshape(-1), gated.reshape(-1))[0, 1]),
    }
    if y is not None:
        summary["baseline_auc"] = macro_auc(y, probs)
        summary["gated_auc"] = macro_auc(y, gated)
        if summary["baseline_auc"]["macro_auc"] is not None and summary["gated_auc"]["macro_auc"] is not None:
            summary["delta_auc"] = float(summary["gated_auc"]["macro_auc"] - summary["baseline_auc"]["macro_auc"])
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
