#!/usr/bin/env python3
"""Combine existing portable TorchScript SED bundles with explicit weights.

This is useful after OOF grid search identifies that two already-packaged
Kaggle-ready bundles should be ensembled.  It rewrites manifest model weights,
copies the source TorchScript files into one output directory, and optionally
creates a zip ready for Kaggle dataset upload.

Example:
  python scripts/birdclef_sed_combine_bundles.py \
    --bundle v23:0.40:artifacts/sed_bundles/sed-b0-q3cap80-ep12init-v23-bundle-v1.zip \
    --bundle v26:0.60:artifacts/sed_bundles/sed-b0-q3cap80-ep12init-v26-allfiles-bundle-v1.zip \
    --output-dir artifacts/sed_bundles/sed-b0-v23v26-oofblend040060-v1 \
    --zip
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any


def parse_bundle(raw: str) -> tuple[str, float, Path]:
    parts = raw.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("bundle must be name:weight:path")
    name, weight_s, path_s = parts
    if not name:
        raise argparse.ArgumentTypeError("bundle name must be non-empty")
    try:
        weight = float(weight_s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid bundle weight {weight_s!r}") from exc
    if weight <= 0:
        raise argparse.ArgumentTypeError("bundle weight must be positive")
    path = Path(path_s)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"bundle path does not exist: {path}")
    return name, weight, path


def parse_member_weight(raw: str) -> tuple[str, str, float]:
    parts = raw.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("member weight must be bundle_name:member_name:weight")
    bundle_name, member_name, weight_s = parts
    if not bundle_name or not member_name:
        raise argparse.ArgumentTypeError("bundle and member names must be non-empty")
    try:
        weight = float(weight_s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid member weight {weight_s!r}") from exc
    if weight <= 0:
        raise argparse.ArgumentTypeError("member weight must be positive")
    return bundle_name, member_name, weight


def find_manifest(root: Path) -> Path:
    direct = root / "sed_bundle_manifest.json"
    if direct.exists():
        return direct
    hits = sorted(root.glob("**/sed_bundle_manifest.json"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly one sed_bundle_manifest.json under {root}, found {len(hits)}")
    return hits[0]


def safe_model_name(bundle_name: str, raw_path: str) -> str:
    name = Path(raw_path).name
    return f"{bundle_name}_{name}".replace("/", "_").replace(" ", "_")


def copy_source_model(src_manifest_dir: Path, entry: dict[str, Any], dst: Path) -> None:
    src = Path(str(entry["path"]))
    if not src.is_absolute():
        src = src_manifest_dir / src
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copy2(src, dst)


def model_member_name(entry: dict[str, Any]) -> str:
    return str(entry.get("member") or entry.get("name") or Path(str(entry["path"])).stem)


def zip_dir(src_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src_dir.parent))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", action="append", required=True, type=parse_bundle, help="name:weight:bundle_dir_or_zip; repeat")
    parser.add_argument(
        "--member-weight",
        action="append",
        default=[],
        type=parse_member_weight,
        help="Optional per-source-member override as bundle_name:member_name:weight; repeat. Weights are normalized within that bundle.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--description", default="BirdCLEF 2026 combined TorchScript SED bundle")
    parser.add_argument("--zip", action="store_true", help="Also write output-dir.with_suffix('.zip')")
    parser.add_argument(
        "--allow-mixed-audio-config",
        action="store_true",
        help="Allow source bundles with different audio configs; each model entry records its own audio_config.",
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    if out_dir.exists():
        shutil.rmtree(out_dir)
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    total_input_weight = sum(weight for _, weight, _ in args.bundle)
    member_weight_overrides: dict[tuple[str, str], float] = {}
    for bundle_name, member_name, weight in args.member_weight:
        key = (bundle_name, member_name)
        if key in member_weight_overrides:
            raise ValueError(f"duplicate member-weight override for {bundle_name}:{member_name}")
        member_weight_overrides[key] = weight
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "description": args.description,
        "source_type": "combined_sed_bundles",
        "members": [],
        "models": [],
    }
    labels_ref: list[str] | None = None
    audio_config_ref: dict[str, Any] | None = None
    tempdirs: list[tempfile.TemporaryDirectory[str]] = []

    try:
        for bundle_name, input_weight, bundle_path in args.bundle:
            source_root = bundle_path
            if bundle_path.suffix.lower() == ".zip":
                tmp = tempfile.TemporaryDirectory(prefix=f"birdclef_{bundle_name}_bundle_")
                tempdirs.append(tmp)
                with zipfile.ZipFile(bundle_path) as zf:
                    zf.extractall(tmp.name)
                source_root = Path(tmp.name)
            source_manifest_path = find_manifest(source_root)
            source_manifest = json.loads(source_manifest_path.read_text())
            source_manifest_dir = source_manifest_path.parent

            labels = [str(x) for x in source_manifest.get("labels", [])]
            if labels_ref is None:
                labels_ref = labels
            elif labels_ref != labels:
                raise ValueError(f"label mismatch for bundle {bundle_name}")

            audio_config = source_manifest.get("audio_config", {})
            if audio_config_ref is None:
                audio_config_ref = audio_config
            elif audio_config_ref != audio_config and not args.allow_mixed_audio_config:
                raise ValueError(f"audio_config mismatch for bundle {bundle_name}: {audio_config} vs {audio_config_ref}")

            normalized_bundle_weight = input_weight / total_input_weight
            source_models = list(source_manifest.get("models", []))
            source_model_weight_sum = sum(float(m.get("weight", 0.0)) for m in source_models)
            if source_model_weight_sum <= 0:
                raise ValueError(f"bundle {bundle_name} has no positive model weights")

            bundle_member_overrides = {
                member: weight
                for (override_bundle, member), weight in member_weight_overrides.items()
                if override_bundle == bundle_name
            }
            if bundle_member_overrides:
                members_seen = {model_member_name(entry) for entry in source_models}
                missing = sorted(members_seen - set(bundle_member_overrides))
                extra = sorted(set(bundle_member_overrides) - members_seen)
                if missing or extra:
                    raise ValueError(
                        f"member-weight overrides for {bundle_name} must cover exactly all source members; "
                        f"missing={missing}, extra={extra}"
                    )
                member_source_weight_sum = {
                    member: sum(float(entry.get("weight", 0.0)) for entry in source_models if model_member_name(entry) == member)
                    for member in members_seen
                }
                member_override_sum = sum(bundle_member_overrides.values())
            else:
                member_source_weight_sum = {}
                member_override_sum = 0.0

            manifest["members"].append({
                "name": bundle_name,
                "input_weight": input_weight,
                "normalized_weight": normalized_bundle_weight,
                "source_path": str(bundle_path),
                "source_description": source_manifest.get("description"),
                "source_audio_config": audio_config,
                "source_members": source_manifest.get("members", []),
                "member_weight_overrides": bundle_member_overrides,
            })

            for entry in source_models:
                dst_name = safe_model_name(bundle_name, str(entry["path"]))
                dst = model_dir / dst_name
                copy_source_model(source_manifest_dir, entry, dst)
                new_entry = dict(entry)
                new_entry["source_bundle"] = bundle_name
                new_entry["source_path"] = entry["path"]
                new_entry["path"] = str(dst.relative_to(out_dir))
                if bundle_member_overrides:
                    member = model_member_name(entry)
                    new_entry["weight"] = (
                        normalized_bundle_weight
                        * bundle_member_overrides[member]
                        / member_override_sum
                        * float(entry.get("weight", 0.0))
                        / member_source_weight_sum[member]
                    )
                else:
                    new_entry["weight"] = normalized_bundle_weight * float(entry.get("weight", 0.0)) / source_model_weight_sum
                new_entry["audio_config"] = entry.get("audio_config") or audio_config
                manifest["models"].append(new_entry)

        if labels_ref is None or audio_config_ref is None:
            raise RuntimeError("no bundles loaded")
        manifest["labels"] = labels_ref
        manifest["n_classes"] = len(labels_ref)
        unique_audio_configs = []
        for model in manifest["models"]:
            cfg = model.get("audio_config") or audio_config_ref
            if cfg not in unique_audio_configs:
                unique_audio_configs.append(cfg)
        manifest["audio_config"] = audio_config_ref
        manifest["audio_configs"] = unique_audio_configs
        manifest["mixed_audio_config"] = len(unique_audio_configs) > 1
        manifest["total_size_mb"] = round(sum((out_dir / m["path"]).stat().st_size for m in manifest["models"]) / 1e6, 3)
        manifest["model_weight_sum"] = sum(float(m["weight"]) for m in manifest["models"])

        manifest_path = out_dir / "sed_bundle_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        zip_path = None
        if args.zip:
            zip_path = out_dir.with_suffix(".zip")
            zip_dir(out_dir, zip_path)
        print(json.dumps({
            "status": "combined_bundle_ready",
            "manifest_path": str(manifest_path),
            "zip_path": str(zip_path) if zip_path else None,
            "n_models": len(manifest["models"]),
            "members": manifest["members"],
            "n_classes": manifest["n_classes"],
            "total_size_mb": manifest["total_size_mb"],
            "model_weight_sum": manifest["model_weight_sum"],
        }, indent=2))
        return 0
    finally:
        for tmp in tempdirs:
            tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
