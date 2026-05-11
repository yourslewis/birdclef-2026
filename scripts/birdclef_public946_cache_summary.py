#!/usr/bin/env python3
"""Summarize/cache public946 replay outputs.

Takes Kaggle output CSVs from the repo-owned public946 replay kernel and writes a
compact NPZ + JSON diagnostics.  The dry-run public Kaggle output contains train
soundscape rows for the intermediate Proto/SED streams and sample-submission rows
for the final code-competition CSV; this script preserves all available streams
and evaluates rows that overlap labeled train soundscapes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


STREAM_FILES = {
    "proto": "submission_protossm.csv",
    "sed": "submission_sed.csv",
    "final": "submission.csv",
}


def parse_seconds(value: Any) -> int:
    text = str(value)
    if ":" in text:
        parts = [int(float(p)) for p in text.split(":")]
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


def topk_row_recall(score_mat: np.ndarray, true_mat: np.ndarray, k: int) -> float:
    kth = min(k, score_mat.shape[1] - 1)
    top = np.argpartition(-score_mat, kth=kth, axis=1)[:, :k]
    hits = []
    for i in range(len(score_mat)):
        true_idx = np.flatnonzero(true_mat[i] > 0)
        hits.append(bool(len(true_idx) and np.intersect1d(true_idx, top[i]).size))
    return float(np.mean(hits)) if hits else float("nan")


def summarize_stream(df: pd.DataFrame, labels_wide: pd.DataFrame | None) -> dict[str, Any]:
    columns = [c for c in df.columns if c != "row_id"]
    info: dict[str, Any] = {
        "shape": list(df.shape),
        "row_head": df["row_id"].head(3).tolist() if "row_id" in df.columns else [],
        "col_head": df.columns[:6].tolist(),
    }
    if labels_wide is None or "row_id" not in df.columns:
        return info
    merged = df.merge(labels_wide, on="row_id", suffixes=("_pred", "_true"))
    valid = [
        c
        for c in columns
        if f"{c}_pred" in merged
        and f"{c}_true" in merged
        and merged[f"{c}_true"].nunique() > 1
    ]
    info.update({"matched_rows": int(len(merged)), "valid_auc_classes": int(len(valid))})
    if not valid:
        return info
    y_true = merged[[f"{c}_true" for c in valid]].to_numpy()
    y_score = merged[[f"{c}_pred" for c in valid]].to_numpy()
    info["macro_auc"] = float(roc_auc_score(y_true, y_score, average="macro"))
    pred_cols = [f"{c}_pred" for c in columns if f"{c}_pred" in merged]
    true_cols = [c.replace("_pred", "_true") for c in pred_cols]
    score_mat = merged[pred_cols].to_numpy()
    true_mat = merged[true_cols].to_numpy()
    for k in (1, 3, 5, 10):
        info[f"top{k}_row_recall"] = topk_row_recall(score_mat, true_mat, k)
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--labels-csv", type=Path)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    streams: dict[str, pd.DataFrame] = {}
    for name, filename in STREAM_FILES.items():
        path = args.pred_dir / filename
        if path.exists():
            streams[name] = pd.read_csv(path)

    if not streams:
        raise FileNotFoundError(f"No expected public946 CSVs found in {args.pred_dir}")

    # Use the first intermediate stream as canonical labels/row source.
    first_df = next(iter(streams.values()))
    columns = [c for c in first_df.columns if c != "row_id"]
    labels_wide = None
    if args.labels_csv and args.labels_csv.exists():
        labels_wide = load_long_labels(args.labels_csv, columns)

    arrays: dict[str, Any] = {
        "labels": np.asarray(columns, dtype=object),
    }
    summary: dict[str, Any] = {
        "pred_dir": str(args.pred_dir),
        "labels_csv": str(args.labels_csv) if args.labels_csv else None,
        "streams": {},
    }
    for name, df in streams.items():
        stream_columns = [c for c in df.columns if c != "row_id"]
        arrays[f"{name}_row_ids"] = df["row_id"].astype(str).to_numpy()
        arrays[f"{name}_probs"] = df[stream_columns].to_numpy(np.float32)
        summary["streams"][name] = summarize_stream(df, labels_wide)

    if labels_wide is not None:
        arrays["label_row_ids"] = labels_wide["row_id"].astype(str).to_numpy()
        arrays["label_matrix"] = labels_wide[columns].to_numpy(np.uint8)

    npz_path = args.output_dir / "predictions.npz"
    np.savez_compressed(npz_path, **arrays)
    summary["output_npz"] = str(npz_path)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
