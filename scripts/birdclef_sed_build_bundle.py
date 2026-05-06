#!/usr/bin/env python3
"""Build a portable TorchScript SED inference bundle from OOF fold artifacts.

The OOF trainer exports one TorchScript model per fold. This utility collects one
or more trained experiment roots into a Kaggle-dataset-friendly directory with a
manifest that the lightweight inference smoke script can load without importing
training code or timm.

Example:
  python scripts/birdclef_sed_build_bundle.py \
    --member v13:0.4:artifacts/sed_oof/sed-nfnet-balanced-oof-v13-10s-160-100cls-lr1e4-ep8 \
    --member v15:0.6:artifacts/sed_oof/sed-nfnet-balanced-oof-v15-10s-160-200cls-lr1e4-ep8 \
    --output-dir artifacts/sed_bundles/sed-nfnet-v13v15-blend-v1 \
    --copy-models
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def parse_member(raw: str) -> tuple[str, float, Path]:
    parts = raw.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("member must be name:weight:experiment_root")
    name, weight_s, root_s = parts
    if not name:
        raise argparse.ArgumentTypeError("member name must be non-empty")
    try:
        weight = float(weight_s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid member weight {weight_s!r}") from exc
    if weight <= 0:
        raise argparse.ArgumentTypeError("member weight must be positive")
    return name, weight, Path(root_s)


def load_labels(root: Path, summary: dict[str, Any]) -> list[str]:
    npz_path = root / Path(summary["oof_predictions_path"]).name
    if not npz_path.exists():
        # oof_predictions_path is usually relative to repo root.
        candidate = Path(summary["oof_predictions_path"])
        if candidate.exists():
            npz_path = candidate
    if not npz_path.exists():
        raise FileNotFoundError(f"Could not find OOF npz for labels: {npz_path}")
    arr = np.load(npz_path, allow_pickle=True)
    return [str(x) for x in arr["labels"].tolist()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--member", action="append", required=True, type=parse_member, help="name:weight:experiment_root; repeat for blends")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--copy-models", action="store_true", help="Copy TorchScript .pt files into output-dir/models")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--description", default="BirdCLEF 2026 NFNet SED TorchScript blend")
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "models"
    if args.copy_models:
        model_dir.mkdir(parents=True, exist_ok=True)

    total_member_weight = sum(weight for _, weight, _ in args.member)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "description": args.description,
        "members": [],
        "models": [],
    }
    labels_ref: list[str] | None = None
    config_ref: dict[str, Any] | None = None

    for member_name, member_weight, root in args.member:
        root = root if root.is_absolute() else args.repo_root / root
        summary_path = root / "oof_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text())
        labels = load_labels(root, summary)
        if labels_ref is None:
            labels_ref = labels
        elif labels_ref != labels:
            raise ValueError(f"Label mismatch for member {member_name}")

        folds = summary.get("folds", [])
        if not folds:
            raise ValueError(f"No folds in {summary_path}")
        member_entry = {
            "name": member_name,
            "root": str(root),
            "input_weight": member_weight,
            "normalized_weight": member_weight / total_member_weight,
            "summary": summary.get("auc_summary"),
            "n_oof": summary.get("n_oof"),
            "experiment_id": summary.get("experiment_id"),
        }
        manifest["members"].append(member_entry)

        per_model_weight = (member_weight / total_member_weight) / len(folds)
        for fold in folds:
            fold_idx = int(fold["fold_index"])
            src = root / f"fold{fold_idx}" / "model_torchscript.pt"
            # Prefer path recorded by metrics when available.
            recorded = fold.get("exports", {}).get("torchscript_path")
            if recorded:
                rec = Path(recorded)
                rec_abs = rec if rec.is_absolute() else args.repo_root / rec
                if rec_abs.exists():
                    src = rec_abs
            if not src.exists():
                raise FileNotFoundError(src)

            if args.copy_models:
                dst = model_dir / f"{member_name}_fold{fold_idx}_model_torchscript.pt"
                shutil.copy2(src, dst)
                model_path = str(dst.relative_to(out_dir))
            else:
                model_path = str(src)

            metrics_path = root / f"fold{fold_idx}" / "metrics.json"
            if metrics_path.exists() and config_ref is None:
                metrics = json.loads(metrics_path.read_text())
                config_ref = metrics.get("config")
            manifest["models"].append({
                "member": member_name,
                "fold_index": fold_idx,
                "path": model_path,
                "weight": per_model_weight,
                "torchscript_size_mb": fold.get("exports", {}).get("torchscript_size_mb"),
                "auc_summary": fold.get("auc_summary"),
            })

    if labels_ref is None:
        raise RuntimeError("No labels loaded")
    manifest["labels"] = labels_ref
    manifest["n_classes"] = len(labels_ref)
    if config_ref:
        manifest["audio_config"] = {
            "sample_rate": config_ref.get("sample_rate", 32000),
            "duration_sec": config_ref.get("duration_sec", 10.0),
            "n_fft": config_ref.get("n_fft", 1024),
            "hop_length": config_ref.get("hop_length", 512),
            "n_mels": config_ref.get("n_mels", 160),
        }
        manifest["train_config_reference"] = config_ref
    else:
        manifest["audio_config"] = {"sample_rate": 32000, "duration_sec": 10.0, "n_fft": 1024, "hop_length": 512, "n_mels": 160}
    manifest["total_size_mb"] = round(sum((out_dir / m["path"]).stat().st_size for m in manifest["models"] if not Path(m["path"]).is_absolute()) / 1e6, 3) if args.copy_models else None

    manifest_path = out_dir / "sed_bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "status": "bundle_ready",
        "manifest_path": str(manifest_path),
        "n_models": len(manifest["models"]),
        "members": manifest["members"],
        "n_classes": manifest["n_classes"],
        "total_size_mb": manifest["total_size_mb"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
