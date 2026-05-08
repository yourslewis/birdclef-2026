#!/usr/bin/env python3
"""Prepare target-species focal-audio manifests for Spec C pretraining.

The script is deliberately data-only: it verifies taxonomy alignment, available
files, collection/rating/class imbalance, and emits deterministic train/val
manifests plus a compact summary for future external/focal pretraining runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


def stable_unit_interval(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return int(digest, 16) / float(16**12 - 1)


def stable_fold(text: str, n_folds: int) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16) % n_folds


def quality_tier(rating: float) -> str:
    if rating >= 4.0:
        return "high_4plus"
    if rating >= 3.0:
        return "medium_3plus"
    if rating > 0.0:
        return "low_positive"
    return "unrated_or_zero"


def read_species(sample_submission: Path) -> list[str]:
    cols = list(pd.read_csv(sample_submission, nrows=1).columns)
    return [c for c in cols if c != "row_id"]


def choose_rows(group: pd.DataFrame, max_per_species: int, prefer_quality: bool) -> pd.DataFrame:
    if len(group) <= max_per_species:
        return group.copy()
    g = group.copy()
    if prefer_quality:
        g = g.sort_values(
            ["rating", "collection_priority", "filename"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        return g.head(max_per_species).copy()
    # Deterministic spread across uploader/order using hash rank.
    g["sample_rank"] = g["filename"].map(lambda x: stable_unit_interval(str(x)))
    return g.sort_values("sample_rank", kind="mergesort").head(max_per_species).drop(columns=["sample_rank"]).copy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-per-species", type=int, default=120)
    ap.add_argument("--min-rating", type=float, default=0.0)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--val-fold", type=int, default=0)
    ap.add_argument("--prefer-quality", action="store_true", help="when capped, pick highest-rating rows before hash spread")
    args = ap.parse_args()

    data_root = args.data_root
    train_path = data_root / "train.csv"
    taxonomy_path = data_root / "taxonomy.csv"
    sample_submission_path = data_root / "sample_submission.csv"
    train_audio_root = data_root / "train_audio"

    train = pd.read_csv(train_path, dtype={"primary_label": str})
    taxonomy = pd.read_csv(taxonomy_path, dtype={"primary_label": str})
    species = read_species(sample_submission_path)
    species_set = set(species)

    label_to_group = dict(zip(taxonomy["primary_label"].astype(str), taxonomy["class_name"].astype(str)))
    train = train[train["primary_label"].isin(species_set)].copy()
    train["rating"] = pd.to_numeric(train["rating"], errors="coerce").fillna(0.0)
    train["class_name"] = train["primary_label"].map(label_to_group).fillna(train["class_name"].astype(str))
    train["relpath"] = train["filename"].astype(str).map(lambda x: f"train_audio/{x}")
    train["path"] = train["relpath"].map(lambda x: str(data_root / x))
    train["file_exists"] = train["path"].map(lambda x: Path(x).exists())
    train["quality_tier"] = train["rating"].map(quality_tier)
    train["collection_priority"] = train["collection"].map({"XC": 0, "iNat": 1}).fillna(9).astype(int)

    before_filter = len(train)
    filtered = train[(train["rating"] >= args.min_rating) & train["file_exists"]].copy()

    selected_parts: list[pd.DataFrame] = []
    for _, group in filtered.groupby("primary_label", sort=True):
        selected_parts.append(choose_rows(group, args.max_per_species, args.prefer_quality))
    manifest = pd.concat(selected_parts, ignore_index=True) if selected_parts else filtered.iloc[:0].copy()

    species_counts = manifest["primary_label"].value_counts().to_dict()
    group_counts = manifest["class_name"].value_counts().to_dict()
    max_count = max(species_counts.values()) if species_counts else 1

    def sample_weight(row: pd.Series) -> float:
        species_count = max(1, int(species_counts.get(row["primary_label"], 1)))
        group_count = max(1, int(group_counts.get(row["class_name"], 1)))
        # Species inverse-sqrt is the main balancing term; group term nudges
        # non-bird classes without exploding rare singleton Reptilia.
        species_term = math.sqrt(max_count / species_count)
        group_term = min(4.0, math.sqrt(len(manifest) / (len(group_counts) * group_count))) if group_counts else 1.0
        rating_term = 1.0 + min(0.5, max(0.0, float(row["rating"]) - 3.0) * 0.125)
        return round(float(species_term * group_term * rating_term), 6)

    manifest["fold"] = manifest.apply(lambda r: stable_fold(f"{r['primary_label']}::{r['filename']}", args.n_folds), axis=1)
    manifest["split"] = manifest["fold"].map(lambda f: "val" if int(f) == args.val_fold else "train")
    manifest["sample_weight"] = manifest.apply(sample_weight, axis=1)

    output_cols = [
        "path", "relpath", "filename", "primary_label", "scientific_name", "common_name", "class_name",
        "collection", "rating", "quality_tier", "fold", "split", "sample_weight", "license", "url",
    ]
    manifest = manifest[output_cols].sort_values(["split", "primary_label", "rating", "filename"], ascending=[True, True, False, True])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "external_pretrain_manifest.csv"
    train_manifest_path = args.output_dir / "external_pretrain_train.csv"
    val_manifest_path = args.output_dir / "external_pretrain_val.csv"
    summary_path = args.output_dir / "external_pretrain_summary.json"
    manifest.to_csv(manifest_path, index=False)
    manifest[manifest["split"] == "train"].to_csv(train_manifest_path, index=False)
    manifest[manifest["split"] == "val"].to_csv(val_manifest_path, index=False)

    available_counts = filtered["primary_label"].value_counts()
    selected_counts = manifest["primary_label"].value_counts()
    rare_species = [s for s in species if int(available_counts.get(s, 0)) < 5]
    capped_species = [s for s in species if int(available_counts.get(s, 0)) > int(selected_counts.get(s, 0))]
    missing_species = [s for s in species if int(available_counts.get(s, 0)) == 0]

    summary: dict[str, Any] = {
        "status": "complete",
        "data_root": str(data_root),
        "n_target_species": len(species),
        "n_train_rows_before_filter": int(before_filter),
        "n_rows_after_rating_and_exists_filter": int(len(filtered)),
        "n_manifest_rows": int(len(manifest)),
        "n_train_split_rows": int((manifest["split"] == "train").sum()),
        "n_val_split_rows": int((manifest["split"] == "val").sum()),
        "min_rating": args.min_rating,
        "max_per_species": args.max_per_species,
        "prefer_quality": args.prefer_quality,
        "n_folds": args.n_folds,
        "val_fold": args.val_fold,
        "rows_by_class": {k: int(v) for k, v in manifest["class_name"].value_counts().sort_index().items()},
        "rows_by_collection": {k: int(v) for k, v in manifest["collection"].value_counts().sort_index().items()},
        "rows_by_quality_tier": {k: int(v) for k, v in manifest["quality_tier"].value_counts().sort_index().items()},
        "available_rows_by_class_before_cap": {k: int(v) for k, v in filtered["class_name"].value_counts().sort_index().items()},
        "available_rows_by_collection_before_cap": {k: int(v) for k, v in filtered["collection"].value_counts().sort_index().items()},
        "missing_file_rows": int((~train["file_exists"]).sum()),
        "rare_species_lt5_available": rare_species,
        "missing_species_after_filter": missing_species,
        "capped_species": capped_species,
        "manifest_path": str(manifest_path),
        "train_manifest_path": str(train_manifest_path),
        "val_manifest_path": str(val_manifest_path),
        "notes": [
            "This manifest is target-taxonomy aligned to sample_submission labels.",
            "Use sample_weight as a first-pass inverse-sqrt species/group balance term; validate OOF before trusting it.",
            "Reptilia has effectively singleton support and should not be overfit without external augmentation.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
