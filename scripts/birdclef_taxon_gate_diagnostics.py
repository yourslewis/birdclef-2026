#!/usr/bin/env python3
"""Build lightweight taxon/no-call diagnostics for BirdCLEF 2026.

This is a Spec E preparation script. It does not train a gate; it validates the
available taxonomy labels and train-soundscape row labels, then writes JSON
artifacts that downstream gate experiments can consume.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


def split_labels(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text == "[]" or text.lower() in {"nan", "none"}:
        return []
    # train_soundscapes_labels uses semicolon-separated label ids.
    if ";" in text:
        return [x.strip() for x in text.split(";") if x.strip()]
    return [text]


def load_species_columns(sample_submission: Path) -> list[str]:
    cols = list(pd.read_csv(sample_submission, nrows=1).columns)
    return [c for c in cols if c != "row_id"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    data_root = args.data_root
    taxonomy_path = data_root / "taxonomy.csv"
    train_path = data_root / "train.csv"
    soundscape_labels_path = data_root / "train_soundscapes_labels.csv"
    sample_submission_path = data_root / "sample_submission.csv"

    taxonomy = pd.read_csv(taxonomy_path, dtype={"primary_label": str})
    train = pd.read_csv(train_path, dtype={"primary_label": str})
    soundscape = pd.read_csv(soundscape_labels_path, dtype={"primary_label": str})
    species = load_species_columns(sample_submission_path)

    label_to_group = dict(zip(taxonomy["primary_label"].astype(str), taxonomy["class_name"].astype(str)))
    species_set = set(map(str, species))

    taxonomy_groups = Counter(taxonomy["class_name"].astype(str))
    submission_groups = Counter(label_to_group.get(s, "UNKNOWN") for s in species)
    train_groups = Counter(train["class_name"].astype(str))

    row_group_counter: Counter[str] = Counter()
    row_label_count_counter: Counter[int] = Counter()
    unknown_labels: Counter[str] = Counter()
    class_pair_counter: Counter[str] = Counter()
    multilabel_rows = 0
    no_call_rows = 0
    target_label_cells = 0

    examples: list[dict[str, Any]] = []
    for row in soundscape.itertuples(index=False):
        labels = [x for x in split_labels(getattr(row, "primary_label")) if x in species_set or x in label_to_group]
        if not labels:
            no_call_rows += 1
            row_group_counter["NO_CALL"] += 1
            row_label_count_counter[0] += 1
            continue
        groups = []
        for lab in labels:
            group = label_to_group.get(lab, "UNKNOWN")
            if group == "UNKNOWN":
                unknown_labels[lab] += 1
            groups.append(group)
        unique_groups = sorted(set(groups))
        row_key = "+".join(unique_groups)
        row_group_counter[row_key] += 1
        row_label_count_counter[len(labels)] += 1
        target_label_cells += len(labels)
        if len(labels) > 1 or len(unique_groups) > 1:
            multilabel_rows += 1
        for i, g1 in enumerate(unique_groups):
            for g2 in unique_groups[i + 1 :]:
                class_pair_counter[f"{g1}+{g2}"] += 1
        if len(examples) < 12:
            examples.append({
                "filename": getattr(row, "filename"),
                "start": getattr(row, "start"),
                "end": getattr(row, "end"),
                "labels": labels,
                "groups": unique_groups,
            })

    group_to_labels: dict[str, list[str]] = defaultdict(list)
    for label in species:
        group_to_labels[label_to_group.get(label, "UNKNOWN")].append(label)

    summary = {
        "status": "complete",
        "data_root": str(data_root),
        "n_submission_species": len(species),
        "n_taxonomy_rows": int(len(taxonomy)),
        "n_train_audio_rows": int(len(train)),
        "n_soundscape_rows": int(len(soundscape)),
        "taxonomy_groups": dict(sorted(taxonomy_groups.items())),
        "submission_species_by_group": dict(sorted(submission_groups.items())),
        "train_audio_rows_by_group": dict(sorted(train_groups.items())),
        "soundscape_rows_by_group_combo": dict(row_group_counter.most_common()),
        "soundscape_rows_by_label_count": {str(k): v for k, v in sorted(row_label_count_counter.items())},
        "soundscape_no_call_rows": no_call_rows,
        "soundscape_multilabel_rows": multilabel_rows,
        "soundscape_target_label_cells": target_label_cells,
        "soundscape_unknown_label_counts": dict(unknown_labels.most_common()),
        "soundscape_cross_group_pair_counts": dict(class_pair_counter.most_common()),
        "group_to_submission_labels": {k: sorted(v) for k, v in sorted(group_to_labels.items())},
        "examples": examples,
        "gate_recommendation": {
            "target_groups": sorted(group_to_labels),
            "candidate_outputs": ["NO_CALL"] + sorted(group_to_labels),
            "notes": [
                "Use taxonomy class_name as first-pass taxon group labels.",
                "Train-soundscape rows are multi-label; taxon gate should be multi-output, not softmax-only.",
                "Use conservative postprocess floors for rare groups; validate macro AUC before suppressing species scores.",
            ],
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
