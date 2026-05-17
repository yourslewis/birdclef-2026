#!/usr/bin/env python3
"""Audit BirdCLEF rare/non-bird source coverage and candidate manifests.

This is a data-only diagnostic for deciding whether Amphibia/Mammalia/Insecta/
Reptilia specialist work has enough verified source audio to justify training.
It intentionally does not train or package a submission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

NONBIRD_GROUPS = ["Amphibia", "Insecta", "Mammalia", "Reptilia"]


def read_target_species(sample_submission: Path) -> list[str]:
    cols = pd.read_csv(sample_submission, nrows=1).columns.tolist()
    return [c for c in cols if c != "row_id"]


def quality_tier(rating: float) -> str:
    if rating >= 4.0:
        return "high_4plus"
    if rating >= 3.0:
        return "medium_3plus"
    if rating > 0:
        return "low_positive"
    return "unrated_or_zero"


def candidate_status(class_name: str, total_rows: int, q3_rows: int, existing_q3_rows: int, existing_rows: int) -> str:
    if class_name == "Aves":
        return "not_target_nonbird"
    if existing_q3_rows >= 5:
        return "trainable_verified_q3"
    if q3_rows >= 5 and existing_q3_rows < 5:
        return "source_q3_needs_audio_files"
    if existing_rows >= 5:
        return "trainable_low_quality_only"
    if total_rows > 0:
        return "source_sparse_or_low_quality"
    return "needs_external_discovery"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--min-q3-rows", type=int, default=5)
    args = ap.parse_args()

    data_root = args.data_root
    train_csv = data_root / "train.csv"
    taxonomy_csv = data_root / "taxonomy.csv"
    sample_submission_csv = data_root / "sample_submission.csv"
    train_audio_root = data_root / "train_audio"

    train = pd.read_csv(train_csv, dtype={"primary_label": str})
    taxonomy = pd.read_csv(taxonomy_csv, dtype={"primary_label": str})
    target_species = read_target_species(sample_submission_csv)
    target_set = set(target_species)

    label_to_class = dict(zip(taxonomy["primary_label"].astype(str), taxonomy["class_name"].astype(str)))
    label_to_common = dict(zip(taxonomy["primary_label"].astype(str), taxonomy.get("common_name", taxonomy["primary_label"]).astype(str)))
    label_to_scientific = dict(zip(taxonomy["primary_label"].astype(str), taxonomy.get("scientific_name", taxonomy["primary_label"]).astype(str)))

    target_train = train[train["primary_label"].isin(target_set)].copy()
    target_train["rating"] = pd.to_numeric(target_train.get("rating", 0), errors="coerce").fillna(0.0)
    target_train["class_name"] = target_train["primary_label"].map(label_to_class).fillna(target_train.get("class_name", "UNKNOWN"))
    target_train["quality_tier"] = target_train["rating"].map(quality_tier)
    target_train["relpath"] = target_train["filename"].astype(str).map(lambda x: f"train_audio/{x}")
    target_train["path"] = target_train["relpath"].map(lambda x: str(data_root / x))
    target_train["file_exists"] = target_train["path"].map(lambda x: Path(x).exists())

    species_rows: list[dict[str, Any]] = []
    for label in target_species:
        rows = target_train[target_train["primary_label"] == label]
        class_name = label_to_class.get(label, "UNKNOWN")
        ratings = rows["rating"] if len(rows) else pd.Series([], dtype=float)
        existing = rows[rows["file_exists"]]
        q3 = rows[rows["rating"] >= 3.0]
        q4 = rows[rows["rating"] >= 4.0]
        existing_q3 = existing[existing["rating"] >= 3.0]
        existing_q4 = existing[existing["rating"] >= 4.0]
        species_rows.append({
            "primary_label": label,
            "class_name": class_name,
            "common_name": label_to_common.get(label, ""),
            "scientific_name": label_to_scientific.get(label, ""),
            "total_rows": int(len(rows)),
            "existing_rows": int(len(existing)),
            "q3_rows": int(len(q3)),
            "q4_rows": int(len(q4)),
            "existing_q3_rows": int(len(existing_q3)),
            "existing_q4_rows": int(len(existing_q4)),
            "zero_or_unrated_rows": int((ratings <= 0).sum()) if len(rows) else 0,
            "max_rating": float(ratings.max()) if len(rows) else None,
            "candidate_status": candidate_status(class_name, int(len(rows)), int(len(q3)), int(len(existing_q3)), int(len(existing))),
        })

    species_df = pd.DataFrame(species_rows)
    nonbird_df = species_df[species_df["class_name"].isin(NONBIRD_GROUPS)].copy()

    def by_class_records(df: pd.DataFrame) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for class_name, group in df.groupby("class_name", sort=True):
            out[str(class_name)] = {
                "n_species": int(len(group)),
                "total_rows": int(group["total_rows"].sum()),
                "existing_rows": int(group["existing_rows"].sum()),
                "q3_rows": int(group["q3_rows"].sum()),
                "q4_rows": int(group["q4_rows"].sum()),
                "existing_q3_rows": int(group["existing_q3_rows"].sum()),
                "existing_q4_rows": int(group["existing_q4_rows"].sum()),
                "zero_or_unrated_rows": int(group["zero_or_unrated_rows"].sum()),
                "species_lt_min_q3": int((group["q3_rows"] < args.min_q3_rows).sum()),
                "species_lt_min_existing_q3": int((group["existing_q3_rows"] < args.min_q3_rows).sum()),
                "missing_species": int((group["total_rows"] == 0).sum()),
            }
        return out

    status_counts = {str(k): int(v) for k, v in nonbird_df["candidate_status"].value_counts().sort_index().items()}

    # Candidate manifest: verified q>=3 rows for Amphibia/Mammalia first.
    priority_groups = {"Amphibia", "Mammalia"}
    manifest = target_train[
        target_train["class_name"].isin(priority_groups)
        & target_train["file_exists"]
        & (target_train["rating"] >= 3.0)
    ].copy()
    manifest = manifest.sort_values(["class_name", "primary_label", "rating", "filename"], ascending=[True, True, False, True])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    species_path = args.output_dir / "rare_nonbird_species_coverage.csv"
    nonbird_path = args.output_dir / "rare_nonbird_species_only.csv"
    manifest_path = args.output_dir / "amphibia_mammalia_q3_existing_manifest.csv"
    summary_path = args.output_dir / "rare_nonbird_source_summary.json"

    species_df.to_csv(species_path, index=False)
    nonbird_df.to_csv(nonbird_path, index=False)
    manifest_cols = [
        "path", "relpath", "filename", "primary_label", "class_name", "rating", "quality_tier",
        "collection", "license", "url",
    ]
    for col in manifest_cols:
        if col not in manifest.columns:
            manifest[col] = ""
    manifest[manifest_cols].to_csv(manifest_path, index=False)

    summary: dict[str, Any] = {
        "status": "complete",
        "data_root": str(data_root),
        "n_target_species": int(len(target_species)),
        "n_nonbird_species": int(len(nonbird_df)),
        "rows_by_class": by_class_records(species_df),
        "nonbird_rows_by_class": by_class_records(nonbird_df),
        "nonbird_candidate_status_counts": status_counts,
        "target_species_missing_all_rows": int((species_df["total_rows"] == 0).sum()),
        "target_species_missing_q3_rows": int((species_df["q3_rows"] == 0).sum()),
        "nonbird_species_lt_min_q3": int((nonbird_df["q3_rows"] < args.min_q3_rows).sum()),
        "nonbird_species_lt_min_existing_q3": int((nonbird_df["existing_q3_rows"] < args.min_q3_rows).sum()),
        "amphibia_mammalia_q3_existing_manifest_rows": int(len(manifest)),
        "amphibia_mammalia_q3_existing_species": int(manifest["primary_label"].nunique()) if len(manifest) else 0,
        "species_coverage_csv": str(species_path),
        "nonbird_species_csv": str(nonbird_path),
        "amphibia_mammalia_manifest_csv": str(manifest_path),
        "notes": [
            "Statuses distinguish source rows from locally verified audio files.",
            "Use Amphibia/Mammalia q>=3 verified rows only for a bounded specialist smoke; Insecta/Reptilia require external discovery or abstention.",
            "Do not treat low-quality q0 expansion as slot-worthy without a separate source validation gate.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
