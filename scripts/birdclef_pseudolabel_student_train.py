#!/usr/bin/env python3
"""Train a small SED student from row-level pseudo-label probabilities.

This is a Spec-B/noisy-student pilot: it consumes row-level pseudo labels from
birdclef_sed_soundscape_infer.py, reconstructs the 5-second endpoint context
windows from train_soundscapes, trains a timm SED student with soft-label BCE,
and evaluates against train_soundscapes_labels.csv when present.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from birdclef_sed_pilot_train import (  # noqa: E402
    auc_summary,
    batch_iter,
    build_model,
    decode_audio_ffmpeg,
    export_model,
    ffmpeg_binary,
    make_mel_filter,
    maybe_mixup,
    resolve_manifest_audio_path,
    waveform_to_logmel,
)
from birdclef_sed_soundscape_infer import decode_soundscape, extract_context  # noqa: E402
from birdclef_pseudolabel_cache_summary import build_truth  # noqa: E402


@dataclass
class StudentConfig:
    experiment_id: str = "pl-r1-b0-soft-power085-labeled-soundscapes"
    pred_npz: str = "artifacts/pseudolabels/sed-v13v15-2m-teacher-r0/labeled_train_soundscape_probs.npz"
    labels_csv: str = "/mnt/mac_data/workspace_don/kaggle_birdclef2026/data/train_soundscapes_labels.csv"
    soundscape_dir: str = "/mnt/mac_data/workspace_don/kaggle_birdclef2026/data/train_soundscapes"
    output_dir: str = "artifacts/pseudolabels/students/pl-r1-b0-soft-power085-labeled-soundscapes"
    sample_rate: int = 32000
    duration_sec: float = 10.0
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 160
    backbone: str = "efficientnet_b0"
    pretrained: bool = False
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 4
    batch_size: int = 16
    max_rows: int | None = None
    val_fraction: float = 0.2
    seed: int = 42
    teacher_power: float = 0.85
    mixup_alpha: float = 0.2
    num_workers: int = 0
    export_onnx: bool = False
    initial_checkpoint: str = ""
    initial_load_head: bool = False
    target_mode: str = "soft"  # soft, hard_conf, or soft_anchor
    positive_threshold: float = 0.90
    negative_threshold: float = 0.05
    max_positive_per_row: int = 0
    max_negative_per_row: int = 0
    max_positive_per_class: int = 0
    max_negative_per_class: int = 0
    soft_label_weight: float = 1.0
    anchor_positive_weight: float = 2.0
    anchor_negative_weight: float = 1.0
    restore_best_by_val_auc: bool = False
    supervised_csv: str = ""
    supervised_data_root: str = ""
    supervised_path_column: str = "filename"
    supervised_label_column: str = "primary_label"
    supervised_secondary_column: str = "secondary_labels"
    supervised_max_files: int = 0
    supervised_max_files_per_class: int = 0
    supervised_min_files_per_class: int = 1
    supervised_weight: float = 1.0
    supervised_label_smoothing: float = 0.0
    supervised_crop_start_sec_max: float = 0.0
    loss_name: str = "bce"  # bce | bce_soft_auc
    auc_loss_weight: float = 0.0
    soft_auc_scale: float = 8.0


def load_config(path: Path | None) -> StudentConfig:
    cfg = StudentConfig()
    if path is None:
        return cfg
    data = json.loads(path.read_text())
    values = asdict(cfg)
    for key, value in data.items():
        if key in values:
            values[key] = value
    return StudentConfig(**values)


def cap_confidence_mask(
    mask: np.ndarray,
    scores: np.ndarray,
    *,
    max_per_row: int = 0,
    max_per_class: int = 0,
    keep: str = "highest",
) -> np.ndarray:
    """Deterministically cap hard pseudo-label masks by confidence.

    keep="highest" preserves the largest probabilities for positives;
    keep="lowest" preserves the smallest probabilities for mined negatives.
    Row caps run first, then class caps to prevent common classes from
    dominating sparse hard-confidence training.
    """
    capped = mask.astype(bool).copy()
    if max_per_row and max_per_row > 0:
        for i in range(capped.shape[0]):
            idx = np.flatnonzero(capped[i])
            if len(idx) > max_per_row:
                order = np.argsort(scores[i, idx], kind="mergesort")
                if keep == "highest":
                    order = order[::-1]
                keep_idx = idx[order[:max_per_row]]
                capped[i] = False
                capped[i, keep_idx] = True
    if max_per_class and max_per_class > 0:
        for j in range(capped.shape[1]):
            idx = np.flatnonzero(capped[:, j])
            if len(idx) > max_per_class:
                order = np.argsort(scores[idx, j], kind="mergesort")
                if keep == "highest":
                    order = order[::-1]
                keep_rows = idx[order[:max_per_class]]
                capped[:, j] = False
                capped[keep_rows, j] = True
    return capped


def row_end_sec(row_id: str) -> int:
    return int(str(row_id).rsplit("_", 1)[1])


def row_stem(row_id: str) -> str:
    return str(row_id).rsplit("_", 1)[0]


def resolve_data_path(path_like: str | Path) -> Path:
    """Resolve Mac/server mirror paths without changing committed configs."""
    path = Path(path_like)
    if path.exists():
        return path
    text = str(path)
    mirrors = [
        ("/mnt/mac_data", "/Volumes/ExternalSSD/data"),
        ("/Volumes/ExternalSSD/data", "/mnt/mac_data"),
    ]
    for src, dst in mirrors:
        if text.startswith(src):
            alt = Path(dst + text[len(src):])
            if alt.exists():
                return alt
    return path


def split_indices(n_items: int, val_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    order = torch.randperm(n_items, generator=torch.Generator().manual_seed(seed))
    n_val = min(max(int(round(n_items * val_fraction)), 1), max(n_items - 1, 1))
    return order[n_val:], order[:n_val]


def load_pseudo_data(cfg: StudentConfig) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(cfg.pred_npz, allow_pickle=True)
    row_ids = z["row_ids"].astype(str)
    labels = [str(x) for x in z["labels"].astype(str).tolist()]
    probs = z["probs"].astype(np.float32)
    if cfg.max_rows is not None:
        row_ids = row_ids[: cfg.max_rows]
        probs = probs[: cfg.max_rows]
    mode = str(cfg.target_mode).lower()
    if mode == "soft":
        # Power scaling for soft labels, clipped to preserve valid BCE targets.
        targets = np.clip(probs, 1e-5, 1 - 1e-5) ** float(cfg.teacher_power)
        targets = np.clip(targets, 1e-5, 1 - 1e-5).astype(np.float32)
        mask = np.ones_like(targets, dtype=np.float32)
    elif mode == "hard_conf":
        pos = probs >= float(cfg.positive_threshold)
        neg = probs <= float(cfg.negative_threshold)
        pos = cap_confidence_mask(
            pos,
            probs,
            max_per_row=cfg.max_positive_per_row,
            max_per_class=cfg.max_positive_per_class,
            keep="highest",
        )
        neg = cap_confidence_mask(
            neg,
            probs,
            max_per_row=cfg.max_negative_per_row,
            max_per_class=cfg.max_negative_per_class,
            keep="lowest",
        )
        mask = (pos | neg).astype(np.float32)
        targets = pos.astype(np.float32)
        if not np.any(pos):
            raise RuntimeError(f"hard_conf found no positives at threshold {cfg.positive_threshold}")
        if not np.any(neg):
            raise RuntimeError(f"hard_conf found no negatives at threshold {cfg.negative_threshold}")
    elif mode == "soft_anchor":
        targets = np.clip(probs, 1e-5, 1 - 1e-5) ** float(cfg.teacher_power)
        targets = np.clip(targets, 1e-5, 1 - 1e-5).astype(np.float32)
        mask = np.full_like(targets, float(cfg.soft_label_weight), dtype=np.float32)
        pos = probs >= float(cfg.positive_threshold)
        neg = probs <= float(cfg.negative_threshold)
        pos = cap_confidence_mask(
            pos,
            probs,
            max_per_row=cfg.max_positive_per_row,
            max_per_class=cfg.max_positive_per_class,
            keep="highest",
        )
        neg = cap_confidence_mask(
            neg,
            probs,
            max_per_row=cfg.max_negative_per_row,
            max_per_class=cfg.max_negative_per_class,
            keep="lowest",
        )
        targets[pos] = 1.0
        targets[neg] = 0.0
        mask[pos] = float(cfg.anchor_positive_weight)
        mask[neg] = float(cfg.anchor_negative_weight)
        if not np.any(pos):
            raise RuntimeError(f"soft_anchor found no positives at threshold {cfg.positive_threshold}")
        if not np.any(neg):
            raise RuntimeError(f"soft_anchor found no negatives at threshold {cfg.negative_threshold}")
    else:
        raise ValueError(f"Unsupported target_mode={cfg.target_mode!r}; expected soft, hard_conf, or soft_anchor")
    return row_ids, labels, probs, targets.astype(np.float32), mask.astype(np.float32)


def build_windows(cfg: StudentConfig, row_ids: np.ndarray) -> torch.Tensor:
    soundscape_dir = resolve_data_path(cfg.soundscape_dir)
    mel_fb = make_mel_filter(cfg.sample_rate, cfg.n_fft, cfg.n_mels)
    file_samples = int(round(cfg.sample_rate * 60.0))
    audio_cache: dict[str, np.ndarray] = {}
    xs = []
    for rid in row_ids:
        stem = row_stem(str(rid))
        audio = audio_cache.get(stem)
        if audio is None:
            path = soundscape_dir / f"{stem}.ogg"
            if not path.exists():
                raise FileNotFoundError(path)
            audio = decode_soundscape(path, cfg.sample_rate, file_samples)
            audio_cache[stem] = audio
        wave = extract_context(audio, cfg.sample_rate, row_end_sec(str(rid)), cfg.duration_sec)
        xs.append(waveform_to_logmel(torch.from_numpy(wave.copy()), cfg, mel_fb))
    return torch.stack(xs)


def parse_secondary_labels(value: object) -> list[str]:
    text = str(value)
    if not text or text == "nan" or text == "[]":
        return []
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].replace("'", "").replace('"', "")
        return [x.strip() for x in text.split(",") if x.strip()]
    return [x.strip() for x in text.split(";") if x.strip()]


def stable_crop_start_sec(path: Path, seed: int, max_start_sec: float) -> float:
    if max_start_sec <= 0:
        return 0.0
    # Deterministic per file without depending on Python's randomized hash().
    key = f"{seed}:{path}".encode("utf-8")
    value = 0
    for byte in key:
        value = (value * 131 + byte) % 1_000_003
    return float(max_start_sec) * (value / 1_000_003.0)


def decode_audio_ffmpeg_segment(path: Path, sr: int, samples: int, start_sec: float) -> np.ndarray:
    cmd = [ffmpeg_binary(), "-v", "error"]
    if start_sec > 0:
        cmd.extend(["-ss", f"{start_sec:.3f}"])
    cmd.extend(["-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"])
    raw = subprocess.check_output(cmd)
    y = np.frombuffer(raw, dtype=np.float32)
    if len(y) < samples:
        y = np.pad(y, (0, samples - len(y)))
    return y[:samples].astype(np.float32, copy=False)


def build_supervised_clip_data(cfg: StudentConfig, labels: list[str]) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, Any]]:
    if not cfg.supervised_csv:
        return None, None, {"enabled": False}
    csv_path = Path(cfg.supervised_csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    data_root = Path(cfg.supervised_data_root or Path(cfg.soundscape_dir).parent)
    df = pd.read_csv(csv_path, dtype={cfg.supervised_label_column: str})
    required = {cfg.supervised_path_column, cfg.supervised_label_column}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"supervised_csv missing required columns: {missing}")
    df = df[df[cfg.supervised_label_column].astype(str).isin(labels)].copy()
    if cfg.supervised_max_files_per_class > 0 or cfg.supervised_min_files_per_class > 1:
        parts = []
        for label, group in df.groupby(cfg.supervised_label_column, sort=True):
            if len(group) < cfg.supervised_min_files_per_class:
                continue
            if cfg.supervised_max_files_per_class > 0 and len(group) > cfg.supervised_max_files_per_class:
                group = group.sample(n=cfg.supervised_max_files_per_class, random_state=cfg.seed)
            parts.append(group)
        df = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()
    if cfg.supervised_max_files > 0 and len(df) > cfg.supervised_max_files:
        df = df.sample(n=cfg.supervised_max_files, random_state=cfg.seed)
    df = df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    mel_fb = make_mel_filter(cfg.sample_rate, cfg.n_fft, cfg.n_mels)
    samples = int(round(cfg.sample_rate * cfg.duration_sec))
    label_to_idx = {label: i for i, label in enumerate(labels)}
    xs = []
    ys = []
    used_rows = 0
    missing_paths = 0
    for row in df.itertuples(index=False):
        label = str(getattr(row, cfg.supervised_label_column))
        raw_path = getattr(row, cfg.supervised_path_column)
        path = resolve_manifest_audio_path(raw_path, data_root)
        if path is None:
            path = resolve_manifest_audio_path(Path("train_audio") / str(raw_path), data_root)
        if path is None:
            missing_paths += 1
            continue
        try:
            start_sec = stable_crop_start_sec(path, cfg.seed, float(cfg.supervised_crop_start_sec_max))
            wave = decode_audio_ffmpeg_segment(path, cfg.sample_rate, samples, start_sec)
        except Exception:
            missing_paths += 1
            continue
        target = np.full(len(labels), float(cfg.supervised_label_smoothing), dtype=np.float32)
        target[label_to_idx[label]] = 1.0
        if cfg.supervised_secondary_column and cfg.supervised_secondary_column in df.columns:
            for sec in parse_secondary_labels(getattr(row, cfg.supervised_secondary_column)):
                if sec in label_to_idx:
                    target[label_to_idx[sec]] = max(target[label_to_idx[sec]], 1.0)
        xs.append(waveform_to_logmel(torch.from_numpy(wave.copy()), cfg, mel_fb))
        ys.append(target)
        used_rows += 1
    if not xs:
        return None, None, {"enabled": True, "requested_rows": int(len(df)), "used_rows": 0, "missing_paths": int(missing_paths)}
    y = torch.from_numpy(np.stack(ys).astype(np.float32))
    return torch.stack(xs), y, {
        "enabled": True,
        "csv": str(csv_path),
        "data_root": str(data_root),
        "requested_rows": int(len(df)),
        "used_rows": int(used_rows),
        "missing_paths": int(missing_paths),
        "max_files": int(cfg.supervised_max_files),
        "max_files_per_class": int(cfg.supervised_max_files_per_class),
        "min_files_per_class": int(cfg.supervised_min_files_per_class),
        "weight": float(cfg.supervised_weight),
        "label_smoothing": float(cfg.supervised_label_smoothing),
        "crop_start_sec_max": float(cfg.supervised_crop_start_sec_max),
    }


def bce_soft_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weight = torch.ones_like(loss)
    if mask is not None:
        weight = weight * mask
    if sample_weight is not None:
        weight = weight * sample_weight.view(-1, 1)
    denom = weight.sum().clamp_min(1.0)
    return (loss * weight).sum() / denom


def soft_auc_pairwise_loss(logits: torch.Tensor, target: torch.Tensor, scale: float = 8.0) -> torch.Tensor:
    """Differentiable soft-label macro AUC surrogate.

    For each class, treat target values as positive weights and (1-target) as
    negative weights, then penalize pos logits not ranking above neg logits. This
    follows the BirdCLEF writeup hint to optimize AUC directly while still
    supporting soft pseudo-labels.
    """
    losses = []
    for j in range(logits.shape[1]):
        y = target[:, j].float().clamp(0.0, 1.0)
        pos_w = y
        neg_w = 1.0 - y
        denom = pos_w.sum() * neg_w.sum()
        if float(denom.detach().cpu()) <= 1e-6:
            continue
        diff = logits[:, j][:, None] - logits[:, j][None, :]
        pair_w = pos_w[:, None] * neg_w[None, :]
        # softplus(-scale * diff) is small when positive rows rank above negatives.
        losses.append((F.softplus(-float(scale) * diff) * pair_w).sum() / denom.clamp_min(1e-6))
    if not losses:
        return logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def student_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None, sample_weight: torch.Tensor | None, cfg: StudentConfig) -> torch.Tensor:
    base = bce_soft_loss(logits, target, mask, sample_weight)
    if str(cfg.loss_name).lower() == "bce_soft_auc" and float(cfg.auc_loss_weight) > 0:
        return base + float(cfg.auc_loss_weight) * soft_auc_pairwise_loss(logits, target, cfg.soft_auc_scale)
    return base


def predict_probs(model, x: torch.Tensor, indices: torch.Tensor, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for idx in batch_iter(indices, batch_size):
            logits, _ = model(x[idx].to(device))
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out, axis=0)


def flat_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1])


def load_initial_checkpoint(model: torch.nn.Module, cfg: StudentConfig) -> dict[str, Any]:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    row_ids, labels, teacher_probs, train_targets, target_mask_np = load_pseudo_data(cfg)
    decode_start = time.time()
    x_pseudo = build_windows(cfg, row_ids)
    y_target_pseudo = torch.from_numpy(train_targets)
    target_mask_pseudo = torch.from_numpy(target_mask_np)
    x_sup, y_sup, supervised_summary = build_supervised_clip_data(cfg, labels)
    if x_sup is not None and y_sup is not None:
        x = torch.cat([x_pseudo, x_sup], dim=0)
        y_target = torch.cat([y_target_pseudo, y_sup], dim=0)
        target_mask = torch.cat([target_mask_pseudo, torch.ones_like(y_sup)], dim=0)
        sample_weight = torch.cat([torch.ones(len(row_ids)), torch.full((len(y_sup),), float(cfg.supervised_weight))])
    else:
        x = x_pseudo
        y_target = y_target_pseudo
        target_mask = target_mask_pseudo
        sample_weight = torch.ones(len(row_ids))
    y_teacher = teacher_probs
    y_true = build_truth(pd.read_csv(resolve_data_path(cfg.labels_csv)), row_ids, labels)
    pseudo_train_idx, val_idx = split_indices(len(row_ids), cfg.val_fraction, cfg.seed)
    if x_sup is not None and y_sup is not None:
        sup_idx = torch.arange(len(row_ids), len(row_ids) + len(y_sup))
        train_idx = torch.cat([pseudo_train_idx, sup_idx])
    else:
        train_idx = pseudo_train_idx

    model = build_model(cfg, len(labels)).to(device)
    initial_checkpoint_summary = load_initial_checkpoint(model, cfg)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    epoch_logs = []
    best_state = None
    best_epoch = None
    best_val_auc = None
    for epoch in range(cfg.epochs):
        model.train(); losses = []
        for idx in batch_iter(train_idx, cfg.batch_size, shuffle=True):
            bx = x[idx].to(device)
            by = y_target[idx].to(device)
            bm = target_mask[idx].to(device)
            bw = sample_weight[idx].to(device)
            if str(cfg.target_mode).lower() == "soft":
                # Mixup only for all-pseudo batches. Mixing supervised one-hot clip labels
                # with pseudo-label rows made attribution hard during early pilots.
                if cfg.mixup_alpha > 0 and torch.all(bw == 1.0):
                    bx, by = maybe_mixup(bx, by, cfg.mixup_alpha)
                bm = None
            logits, _ = model(bx)
            loss = student_loss(logits, by, bm, bw, cfg)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(float(loss.detach().cpu()))
        val_pred = predict_probs(model, x, val_idx, cfg.batch_size, device)
        val_teacher = y_teacher[val_idx.numpy()]
        val_true = y_true[val_idx.numpy()]
        val_student_auc = auc_summary(val_true, val_pred)
        epoch_logs.append({
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "val_student_vs_truth": val_student_auc,
            "val_teacher_vs_truth": auc_summary(val_true, val_teacher),
            "val_student_teacher_corr": flat_corr(val_pred, val_teacher),
            "val_student_teacher_mae": float(np.mean(np.abs(val_pred - val_teacher))),
        })
        metric = val_student_auc.get("macro_auc")
        if cfg.restore_best_by_val_auc and metric is not None and (best_val_auc is None or float(metric) > float(best_val_auc)):
            best_val_auc = float(metric)
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        print(json.dumps(epoch_logs[-1]), flush=True)

    if cfg.restore_best_by_val_auc and best_state is not None:
        model.load_state_dict(best_state)
        (out_dir / "best_checkpoint_info.json").write_text(json.dumps({
            "selection": "max_val_student_vs_truth_macro_auc",
            "best_epoch": int(best_epoch),
            "best_val_auc": float(best_val_auc),
        }, indent=2) + "\n")

    all_idx = torch.arange(len(row_ids))
    student_probs = predict_probs(model, x, all_idx, cfg.batch_size, device)
    npz_path = out_dir / "student_predictions.npz"
    np.savez_compressed(
        npz_path,
        row_ids=row_ids,
        labels=np.array(labels),
        pred_student=student_probs.astype(np.float32),
        pred_teacher=y_teacher.astype(np.float32),
        y_true=y_true.astype(np.float32),
        train_indices=train_idx.numpy(),
        val_indices=val_idx.numpy(),
    )
    exports = export_model(model, x[:1], out_dir, cfg)
    metrics = {
        "experiment_id": cfg.experiment_id,
        "track": "B Pseudo-label/noisy-student soft-label pilot",
        "status": "student_complete",
        "device": str(device),
        "backbone_actual": getattr(model, "backbone_name", cfg.backbone),
        "n_rows": int(len(row_ids)),
        "n_pseudo_train": int(len(pseudo_train_idx)),
        "n_supervised_train": int(len(y_sup)) if y_sup is not None else 0,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "supervised_summary": supervised_summary,
        "n_classes": int(len(labels)),
        "teacher_power": float(cfg.teacher_power),
        "target_mode": str(cfg.target_mode),
        "loss_name": str(cfg.loss_name),
        "auc_loss_weight": float(cfg.auc_loss_weight),
        "soft_auc_scale": float(cfg.soft_auc_scale),
        "target_mask_fraction": float(target_mask_np.mean()),
        "target_positive_cells": int((train_targets * target_mask_np).sum()),
        "target_negative_cells": int(((1.0 - train_targets) * target_mask_np).sum()),
        "restore_best_by_val_auc": bool(cfg.restore_best_by_val_auc),
        "initial_checkpoint_summary": initial_checkpoint_summary,
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "best_val_auc": float(best_val_auc) if best_val_auc is not None else None,
        "input_shape": list(x.shape),
        "epochs": epoch_logs,
        "final_all_student_vs_truth": auc_summary(y_true, student_probs),
        "final_all_teacher_vs_truth": auc_summary(y_true, y_teacher),
        "final_student_teacher_corr": flat_corr(student_probs, y_teacher),
        "final_student_teacher_mae": float(np.mean(np.abs(student_probs - y_teacher))),
        "student_predictions_path": str(npz_path),
        "decode_feature_sec": round(time.time() - decode_start, 3),
        "runtime_sec": round(time.time() - started, 3),
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
