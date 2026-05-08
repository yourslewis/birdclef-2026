#!/usr/bin/env python3
"""BirdCLEF Spec A+G SED GPU pilot trainer.

A small but real weak-label SED pilot that scales the smoke scaffold toward the
spec's EfficientNet-B0/timm direction while keeping a tiny-CNN fallback for
preflight environments. It writes the required AutoResearch artifacts:
config/log/holdout predictions, macro AUC diagnostics, timing, and exports.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover
    roc_auc_score = None


@dataclass
class PilotConfig:
    experiment_id: str = "sed-b0-gpu-pilot-v1-5s-focal15-possqrt"
    track: str = "A+G Real SED frame/event GPU pilot"
    data_root: str = "/mnt/mac_data/workspace_don/kaggle_birdclef2026/data"
    output_dir: str = "artifacts/sed_pilots/sed-b0-gpu-pilot-v1-5s-focal15-possqrt"
    sample_rate: int = 32000
    duration_sec: float = 5.0
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 128
    backbone: str = "efficientnet_b0"
    pretrained: bool = False
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 2
    batch_size: int = 16
    max_files: int = 512
    selection_strategy: str = "default"  # default | balanced_classes | manifest
    manifest_csv: str = ""
    manifest_path_column: str = "path"
    manifest_label_column: str = "primary_label"
    manifest_split: str = ""
    manifest_max_files_per_class: int = 0
    manifest_min_files_per_class: int = 1
    max_classes: int = 30
    files_per_class: int = 10
    min_files_per_class: int = 6
    seed: int = 42
    val_fraction: float = 0.2
    n_folds: int = 1
    fold_index: int = 0
    loss_name: str = "focal_bce"
    focal_gamma: float = 1.5
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0
    class_balancing: str = "pos_weight_sqrt"
    num_workers: int = 0
    export_onnx: bool = True
    oof_negative_cache: str = ""
    aux_negative_weight: float = 0.0
    oof_negative_mask_key: str = "negative_mask"
    restore_best_by_val_loss: bool = False
    initial_checkpoint: str = ""
    initial_load_head: bool = False


def load_config(path: Path | None) -> PilotConfig:
    cfg = PilotConfig()
    if path is None:
        return cfg
    data = json.loads(path.read_text())
    values = asdict(cfg)
    for key, value in data.items():
        if key in values:
            values[key] = value
    return PilotConfig(**values)


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def make_mel_filter(sr: int, n_fft: int, n_mels: int, fmin: float = 20.0, fmax: float | None = None) -> torch.Tensor:
    if fmax is None:
        fmax = sr / 2
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    n_freq = n_fft // 2 + 1
    fb = np.zeros((n_mels, n_freq), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for k in range(left, min(center, n_freq)):
            fb[m - 1, k] = (k - left) / max(center - left, 1)
        for k in range(center, min(right, n_freq)):
            fb[m - 1, k] = (right - k) / max(right - center, 1)
    return torch.from_numpy(fb)


def ffmpeg_binary() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("ffmpeg is required for OGG decode; install ffmpeg or imageio-ffmpeg") from exc


def decode_audio_ffmpeg(path: Path, sr: int, samples: int) -> np.ndarray:
    raw = subprocess.check_output([ffmpeg_binary(), "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"])
    y = np.frombuffer(raw, dtype=np.float32)
    if len(y) < samples:
        y = np.pad(y, (0, samples - len(y)))
    return y[:samples].astype(np.float32, copy=False)


def waveform_to_logmel(wave: torch.Tensor, cfg: PilotConfig, mel_fb: torch.Tensor) -> torch.Tensor:
    window = torch.hann_window(cfg.n_fft)
    spec = torch.stft(wave, n_fft=cfg.n_fft, hop_length=cfg.hop_length, win_length=cfg.n_fft, window=window, center=True, return_complex=True)
    mel = mel_fb @ spec.abs().pow(2.0)
    logmel = torch.log1p(mel)
    return (logmel - logmel.mean()) / (logmel.std() + 1e-6)


class TinySEDSmoke(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.SiLU(inplace=True), nn.MaxPool2d((2, 1)),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.SiLU(inplace=True), nn.MaxPool2d((2, 1)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.SiLU(inplace=True),
        )
        self.frame_head = nn.Conv1d(64, n_classes, kernel_size=1)
        self.backbone_name = "tiny_cnn_sed"

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.features(x.unsqueeze(1)).mean(dim=2)
        frame_logits = self.frame_head(z).transpose(1, 2)
        clip_logits = 0.5 * frame_logits.mean(dim=1) + 0.5 * frame_logits.amax(dim=1)
        return clip_logits, frame_logits


class TimmSEDB0(nn.Module):
    def __init__(self, n_classes: int, backbone: str, pretrained: bool = False):
        super().__init__()
        import timm
        self.encoder = timm.create_model(backbone, pretrained=pretrained, features_only=True, in_chans=1, out_indices=(-1,))
        channels = self.encoder.feature_info.channels()[-1]
        self.frame_head = nn.Conv1d(channels, n_classes, kernel_size=1)
        self.backbone_name = backbone

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(x.unsqueeze(1))[-1]
        z = feat.mean(dim=2)
        frame_logits = self.frame_head(z).transpose(1, 2)
        clip_logits = 0.5 * frame_logits.mean(dim=1) + 0.5 * frame_logits.amax(dim=1)
        return clip_logits, frame_logits


def build_model(cfg: PilotConfig, n_classes: int) -> nn.Module:
    if cfg.backbone != "tiny_cnn":
        try:
            return TimmSEDB0(n_classes, cfg.backbone, cfg.pretrained)
        except Exception as exc:
            print(f"WARNING: timm backbone {cfg.backbone!r} unavailable ({type(exc).__name__}: {exc}); using tiny_cnn fallback", flush=True)
    return TinySEDSmoke(n_classes)


def load_initial_checkpoint(model: nn.Module, cfg: PilotConfig) -> dict[str, Any]:
    if not cfg.initial_checkpoint:
        return {"enabled": False}
    path = Path(cfg.initial_checkpoint)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        obj = torch.jit.load(str(path), map_location="cpu")
        state = obj.state_dict()
        source = "torchscript"
    except Exception:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict) and "state_dict" in obj:
            state = obj["state_dict"]
        elif isinstance(obj, dict) and "model" in obj:
            state = obj["model"]
        elif isinstance(obj, dict):
            state = obj
        else:
            raise TypeError(f"Unsupported checkpoint object from {path}: {type(obj)!r}")
        source = "torch"

    model_state = model.state_dict()
    filtered = {}
    skipped_shape = []
    skipped_head = []
    skipped_missing = []
    for key, value in state.items():
        if not cfg.initial_load_head and key.startswith("frame_head."):
            skipped_head.append(key)
            continue
        if key not in model_state:
            skipped_missing.append(key)
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            skipped_shape.append(key)
            continue
        filtered[key] = value
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return {
        "enabled": True,
        "path": str(path),
        "source": source,
        "load_head": bool(cfg.initial_load_head),
        "loaded_keys": int(len(filtered)),
        "skipped_head_keys": int(len(skipped_head)),
        "skipped_shape_keys": skipped_shape[:20],
        "skipped_missing_keys": skipped_missing[:20],
        "model_missing_after_partial_load": list(missing)[:20],
        "model_unexpected_after_partial_load": list(unexpected)[:20],
    }


def path_key(path: str | Path) -> str:
    parts = Path(str(path)).parts
    if "train_audio" in parts:
        i = parts.index("train_audio")
        return "/".join(parts[i + 1:])
    p = Path(str(path))
    return f"{p.parent.name}/{p.name}"


def resolve_manifest_audio_path(raw_path: str | Path, data_root: Path) -> Path | None:
    """Resolve manifest audio paths across Mac, SMB, and GPU-local mirrors.

    External-pretrain manifests are commonly built on the Mac with absolute
    `/Volumes/ExternalSSD/.../data/train_audio/<label>/<file>.ogg` paths, while
    durable GPU jobs run on `trainer` with data staged under
    `~/birdclef-2026/data/train_audio`.  Keep the manifest portable by falling
    back to the path relative to the `train_audio` anchor.
    """
    raw = Path(str(raw_path))
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([data_root / raw, data_root / "train_audio" / raw])

    key = path_key(raw)
    candidates.extend([
        data_root / "train_audio" / key,
        Path("/mnt/mac_data/workspace_don/kaggle_birdclef2026/data/train_audio") / key,
        Path("/Volumes/ExternalSSD/data/workspace_don/kaggle_birdclef2026/data/train_audio") / key,
    ])
    seen: set[str] = set()
    for cand in candidates:
        marker = str(cand)
        if marker in seen:
            continue
        seen.add(marker)
        if cand.exists():
            return cand
    return None


def select_examples(data_root: Path, labels: list[str], cfg: PilotConfig) -> list[tuple[Path, str]]:
    rng = random.Random(cfg.seed)
    if cfg.selection_strategy == "manifest":
        if not cfg.manifest_csv:
            raise ValueError("selection_strategy=manifest requires manifest_csv")
        manifest_path = Path(cfg.manifest_csv)
        if not manifest_path.is_absolute():
            manifest_path = data_root / cfg.manifest_csv
        manifest = pd.read_csv(manifest_path, dtype={cfg.manifest_label_column: str})
        if cfg.manifest_split:
            if "split" not in manifest.columns:
                raise ValueError("manifest_split was set but manifest has no split column")
            manifest = manifest[manifest["split"].astype(str) == cfg.manifest_split].copy()
        required = {cfg.manifest_path_column, cfg.manifest_label_column}
        missing = sorted(required - set(manifest.columns))
        if missing:
            raise ValueError(f"manifest missing required columns: {missing}")
        if cfg.manifest_max_files_per_class > 0 or cfg.manifest_min_files_per_class > 1:
            balanced_parts = []
            for label, group in manifest.groupby(cfg.manifest_label_column, sort=True):
                if str(label) not in labels or len(group) < cfg.manifest_min_files_per_class:
                    continue
                if cfg.manifest_max_files_per_class > 0 and len(group) > cfg.manifest_max_files_per_class:
                    group = group.sample(n=cfg.manifest_max_files_per_class, random_state=cfg.seed)
                balanced_parts.append(group)
            if balanced_parts:
                manifest = pd.concat(balanced_parts, ignore_index=True).sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
            else:
                manifest = manifest.iloc[0:0].copy()
        selected: list[tuple[Path, str]] = []
        for row in manifest.itertuples(index=False):
            label = str(getattr(row, cfg.manifest_label_column))
            if label not in labels:
                continue
            path = resolve_manifest_audio_path(getattr(row, cfg.manifest_path_column), data_root)
            if path is not None:
                selected.append((path, label))
        rng.shuffle(selected)
        return selected[: cfg.max_files]

    train_audio = data_root / "train_audio"
    if cfg.selection_strategy == "oof_negative_cache":
        if not cfg.oof_negative_cache:
            raise ValueError("selection_strategy=oof_negative_cache requires oof_negative_cache")
        z = np.load(cfg.oof_negative_cache, allow_pickle=True)
        cache_files = [path_key(x) for x in z["files"].astype(str)]
        examples = []
        for key in cache_files:
            label = key.split("/", 1)[0]
            path = train_audio / key
            if label in labels and path.exists():
                examples.append((path, label))
        if len(examples) < 5:
            raise RuntimeError(f"Need at least 5 cache-backed examples, found {len(examples)}")
        rng.shuffle(examples)
        return examples[: cfg.max_files]
    if cfg.selection_strategy == "balanced_classes":
        selected: list[tuple[Path, str]] = []
        eligible = []
        for label_dir in sorted(train_audio.iterdir()):
            if not label_dir.is_dir() or label_dir.name not in labels:
                continue
            files = sorted(label_dir.glob("*.ogg"))
            if len(files) >= cfg.min_files_per_class:
                eligible.append((label_dir.name, files))
        rng.shuffle(eligible)
        for label, files in eligible[: cfg.max_classes]:
            files = list(files)
            rng.shuffle(files)
            for path in files[: cfg.files_per_class]:
                selected.append((path, label))
        selected = selected[: cfg.max_files]
        rng.shuffle(selected)
        return selected

    selected: list[tuple[Path, str]] = []
    for label_dir in sorted(train_audio.iterdir()):
        if not label_dir.is_dir() or label_dir.name not in labels:
            continue
        files = sorted(label_dir.glob("*.ogg"))
        if files:
            selected.append((rng.choice(files), label_dir.name))
        if len(selected) >= cfg.max_files:
            return selected
    all_files = [(p, p.parent.name) for p in train_audio.glob("*/*.ogg") if p.parent.name in labels]
    rng.shuffle(all_files)
    seen = {p for p, _ in selected}
    for path, label in all_files:
        if path not in seen:
            selected.append((path, label))
        if len(selected) >= cfg.max_files:
            break
    return selected


def make_targets(meta: list[dict[str, Any]], labels: list[str]) -> torch.Tensor:
    label_to_idx = {label: i for i, label in enumerate(labels)}
    y = torch.zeros((len(meta), len(labels)), dtype=torch.float32)
    for i, item in enumerate(meta):
        y[i, label_to_idx[item["label"]]] = 1.0
    return y


def load_oof_negative_mask(cfg: PilotConfig, meta: list[dict[str, Any]], labels: list[str]) -> tuple[torch.Tensor | None, dict[str, Any]]:
    if not cfg.oof_negative_cache or cfg.aux_negative_weight <= 0:
        return None, {"enabled": False}
    z = np.load(cfg.oof_negative_cache, allow_pickle=True)
    cache_labels = z["labels"].astype(str)
    if not np.array_equal(cache_labels, np.array(labels, dtype=str)):
        raise RuntimeError("OOF negative cache labels do not match training labels")
    if cfg.oof_negative_mask_key not in z.files:
        raise RuntimeError(f"Mask key {cfg.oof_negative_mask_key!r} not in {cfg.oof_negative_cache}")
    mask_arr = z[cfg.oof_negative_mask_key].astype(bool)
    key_to_idx = {path_key(f): i for i, f in enumerate(z["files"].astype(str))}
    out = np.zeros((len(meta), len(labels)), dtype=np.float32)
    covered_rows = 0
    for i, item in enumerate(meta):
        key = path_key(item["path"])
        idx = key_to_idx.get(key)
        if idx is None:
            continue
        out[i] = mask_arr[idx].astype(np.float32)
        covered_rows += 1
    cells = int(out.sum())
    return torch.from_numpy(out), {
        "enabled": True,
        "cache_path": cfg.oof_negative_cache,
        "mask_key": cfg.oof_negative_mask_key,
        "aux_negative_weight": float(cfg.aux_negative_weight),
        "covered_rows": int(covered_rows),
        "coverage_fraction": float(covered_rows / max(len(meta), 1)),
        "negative_cells": cells,
        "mean_negative_cells_per_covered_row": float(cells / max(covered_rows, 1)),
    }


def masked_negative_loss(logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None or float(mask.sum().detach().cpu()) <= 0:
        return logits.new_tensor(0.0)
    target = torch.zeros_like(logits)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def split_indices(n_items: int, val_fraction: float, seed: int, n_folds: int = 1, fold_index: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    order = torch.randperm(n_items, generator=torch.Generator().manual_seed(seed))
    if n_folds > 1:
        if not 0 <= fold_index < n_folds:
            raise ValueError(f"fold_index must be in [0, {n_folds}), got {fold_index}")
        mask = torch.arange(n_items) % n_folds == fold_index
        val_idx = order[mask]
        train_idx = order[~mask]
        if len(val_idx) == 0 or len(train_idx) == 0:
            raise ValueError(f"Invalid fold split: n_items={n_items} n_folds={n_folds} fold_index={fold_index}")
        return train_idx, val_idx
    n_val = min(max(int(round(n_items * val_fraction)), 1), max(n_items - 1, 1))
    return order[n_val:], order[:n_val]


def smooth_targets(target: torch.Tensor, eps: float) -> torch.Tensor:
    return target if eps <= 0 else target * (1.0 - eps) + 0.5 * eps


def make_pos_weight(n_classes: int, mode: str, device: torch.device) -> torch.Tensor | None:
    if mode == "none":
        return None
    val = max(n_classes - 1, 1)
    if mode == "pos_weight_sqrt":
        val = float(np.sqrt(val))
    elif mode == "pos_weight_linear":
        val = float(val)
    else:
        raise ValueError(f"Unknown class_balancing={mode}")
    return torch.full((n_classes,), val, dtype=torch.float32, device=device)


def compute_loss(logits: torch.Tensor, target: torch.Tensor, cfg: PilotConfig, pos_weight: torch.Tensor | None) -> torch.Tensor:
    target = smooth_targets(target, cfg.label_smoothing)
    if cfg.loss_name == "bce":
        return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    if cfg.loss_name == "focal_bce":
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
        prob = torch.sigmoid(logits)
        pt = prob * target + (1 - prob) * (1 - target)
        return ((1 - pt).clamp_min(1e-6).pow(cfg.focal_gamma) * bce).mean()
    raise ValueError(f"Unknown loss_name={cfg.loss_name}")


def maybe_mixup(x: torch.Tensor, y: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    if alpha <= 0 or len(x) < 2:
        return x, y
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(len(x), device=x.device)
    return lam * x + (1 - lam) * x[perm], lam * y + (1 - lam) * y[perm]


def batch_iter(indices: torch.Tensor, batch_size: int, shuffle: bool = False):
    if shuffle:
        indices = indices[torch.randperm(len(indices))]
    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]


def predict(model: nn.Module, x: torch.Tensor, indices: torch.Tensor, batch_size: int, device: torch.device) -> tuple[np.ndarray, float]:
    model.eval()
    probs = []
    total_loss_time = time.time()
    with torch.no_grad():
        for idx in batch_iter(indices, batch_size):
            logits, _ = model(x[idx].to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs, axis=0), round(time.time() - total_loss_time, 3)


def auc_summary(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    if roc_auc_score is None:
        return {"macro_auc": None, "valid_classes": 0, "reason": "sklearn unavailable"}
    aucs = []
    for j in range(y_true.shape[1]):
        col = y_true[:, j]
        if col.min() == col.max():
            continue
        try:
            aucs.append(float(roc_auc_score(col, y_prob[:, j])))
        except Exception:
            pass
    return {"macro_auc": float(np.mean(aucs)) if aucs else None, "valid_classes": len(aucs)}


def export_model(model: nn.Module, example: torch.Tensor, out_dir: Path, cfg: PilotConfig) -> dict[str, Any]:
    exports: dict[str, Any] = {}
    model_cpu = model.to("cpu").eval()
    traced = torch.jit.trace(model_cpu, example.cpu())
    ts_path = out_dir / "model_torchscript.pt"
    traced.save(str(ts_path))
    exports["torchscript_path"] = str(ts_path)
    exports["torchscript_size_mb"] = round(ts_path.stat().st_size / 1e6, 3)
    if cfg.export_onnx:
        onnx_path = out_dir / "model.onnx"
        try:
            import onnx  # noqa: F401
            torch.onnx.export(
                model_cpu,
                example.cpu(),
                str(onnx_path),
                input_names=["logmel"],
                output_names=["clip_logits", "frame_logits"],
                dynamic_axes={"logmel": {0: "batch", 2: "frames"}, "clip_logits": {0: "batch"}, "frame_logits": {0: "batch", 1: "frames"}},
                opset_version=18,
            )
            exports["onnx_path"] = str(onnx_path)
            exports["onnx_size_mb"] = round(onnx_path.stat().st_size / 1e6, 3)
            exports["onnx_status"] = "exported"
        except Exception as exc:
            exports["onnx_status"] = f"failed: {type(exc).__name__}: {exc}"
    return exports


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-root", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--backbone", type=str)
    args = parser.parse_args()
    cfg = load_config(args.config)
    for key in ["data_root", "output_dir", "max_files", "epochs", "batch_size", "backbone", "selection_strategy", "manifest_csv", "manifest_split"]:
        val = getattr(args, key.replace("-", "_"), None)
        if val is not None:
            setattr(cfg, key, val)

    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    start_all = time.time()
    taxonomy = pd.read_csv(Path(cfg.data_root) / "taxonomy.csv")
    labels = taxonomy["primary_label"].astype(str).tolist()
    examples = select_examples(Path(cfg.data_root), labels, cfg)
    if len(examples) < 5:
        raise RuntimeError(f"Need at least 5 examples, found {len(examples)}")

    mel_fb = make_mel_filter(cfg.sample_rate, cfg.n_fft, cfg.n_mels)
    n_samples = int(cfg.sample_rate * cfg.duration_sec)
    xs, meta = [], []
    decode_start = time.time()
    for path, label in examples:
        wav = decode_audio_ffmpeg(path, cfg.sample_rate, n_samples)
        xs.append(waveform_to_logmel(torch.from_numpy(wav.copy()), cfg, mel_fb))
        meta.append({"path": str(path), "label": label})
    x = torch.stack(xs)
    y = make_targets(meta, labels)
    aux_negative_mask, aux_negative_summary = load_oof_negative_mask(cfg, meta, labels)
    train_idx, val_idx = split_indices(len(x), cfg.val_fraction, cfg.seed, cfg.n_folds, cfg.fold_index)

    model = build_model(cfg, len(labels))
    initial_checkpoint_summary = load_initial_checkpoint(model, cfg)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    pos_weight = make_pos_weight(len(labels), cfg.class_balancing, device)
    epoch_logs = []
    best_state = None
    best_epoch = None
    best_val_loss = None
    for epoch in range(cfg.epochs):
        model.train(); losses = []
        for idx in batch_iter(train_idx, cfg.batch_size, shuffle=True):
            bx, by = x[idx].to(device), y[idx].to(device)
            bx, by = maybe_mixup(bx, by, cfg.mixup_alpha)
            logits, _ = model(bx)
            loss = compute_loss(logits, by, cfg, pos_weight)
            if aux_negative_mask is not None and cfg.aux_negative_weight > 0:
                nm = aux_negative_mask[idx].to(device)
                loss = loss + float(cfg.aux_negative_weight) * masked_negative_loss(logits, nm)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(float(loss.detach().cpu()))
        val_probs, _ = predict(model, x, val_idx, cfg.batch_size, device)
        val_loss = compute_loss(torch.logit(torch.from_numpy(val_probs).clamp(1e-6, 1 - 1e-6)).to(device), y[val_idx].to(device), cfg, pos_weight)
        epoch_log = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": float(val_loss.detach().cpu())}
        epoch_logs.append(epoch_log)
        if cfg.restore_best_by_val_loss and (best_val_loss is None or float(epoch_log["val_loss"]) < float(best_val_loss)):
            best_val_loss = float(epoch_log["val_loss"])
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        print(json.dumps(epoch_logs[-1]), flush=True)

    if cfg.restore_best_by_val_loss and best_state is not None:
        model.load_state_dict(best_state)
        (out_dir / "best_checkpoint_info.json").write_text(json.dumps({"best_epoch": best_epoch, "best_val_loss": best_val_loss, "criterion": "val_loss"}, indent=2) + "\n")

    val_probs, pred_time = predict(model, x, val_idx, cfg.batch_size, device)
    train_probs, _ = predict(model, x, train_idx, cfg.batch_size, device)
    auc = auc_summary(y[val_idx].numpy(), val_probs)
    npz_path = out_dir / "holdout_predictions.npz"
    np.savez_compressed(npz_path, val_indices=val_idx.numpy(), train_indices=train_idx.numpy(), y_val=y[val_idx].numpy(), pred_val=val_probs, pred_train=train_probs, labels=np.array(labels), files=np.array([m["path"] for m in meta]))
    exports = export_model(model, x[:1], out_dir, cfg)
    metrics = {
        "experiment_id": cfg.experiment_id,
        "track": cfg.track,
        "status": "pilot_complete",
        "device": str(device),
        "backbone_actual": getattr(model, "backbone_name", cfg.backbone),
        "n_examples": len(examples),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_classes": len(labels),
        "n_folds": int(cfg.n_folds),
        "fold_index": int(cfg.fold_index),
        "input_shape": list(x.shape),
        "epochs": epoch_logs,
        "auc_summary": auc,
        "restore_best_by_val_loss": bool(cfg.restore_best_by_val_loss),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "initial_checkpoint_summary": initial_checkpoint_summary,
        "aux_negative_summary": aux_negative_summary,
        "prediction_time_sec": pred_time,
        "holdout_predictions_path": str(npz_path),
        "decode_feature_sec": round(time.time() - decode_start, 3),
        "runtime_sec": round(time.time() - start_all, 3),
        "exports": exports,
        "config": asdict(cfg),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out_dir / "config.resolved.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    (out_dir / "training_log.jsonl").write_text("\n".join(json.dumps(row) for row in epoch_logs) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
