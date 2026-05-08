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
    export_model,
    make_mel_filter,
    maybe_mixup,
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
    target_mode: str = "soft"  # soft, hard_conf, or soft_anchor
    positive_threshold: float = 0.90
    negative_threshold: float = 0.05
    soft_label_weight: float = 1.0
    anchor_positive_weight: float = 2.0
    anchor_negative_weight: float = 1.0
    restore_best_by_val_auc: bool = False


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


def bce_soft_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if mask is None:
        return loss.mean()
    denom = mask.sum().clamp_min(1.0)
    return (loss * mask).sum() / denom


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
    x = build_windows(cfg, row_ids)
    y_target = torch.from_numpy(train_targets)
    target_mask = torch.from_numpy(target_mask_np)
    y_teacher = teacher_probs
    y_true = build_truth(pd.read_csv(resolve_data_path(cfg.labels_csv)), row_ids, labels)
    train_idx, val_idx = split_indices(len(row_ids), cfg.val_fraction, cfg.seed)

    model = build_model(cfg, len(labels)).to(device)
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
            if str(cfg.target_mode).lower() == "soft":
                bx, by = maybe_mixup(bx, by, cfg.mixup_alpha)
                bm = None
            logits, _ = model(bx)
            loss = bce_soft_loss(logits, by, bm)
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
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_classes": int(len(labels)),
        "teacher_power": float(cfg.teacher_power),
        "target_mode": str(cfg.target_mode),
        "target_mask_fraction": float(target_mask_np.mean()),
        "target_positive_cells": int((train_targets * target_mask_np).sum()),
        "target_negative_cells": int(((1.0 - train_targets) * target_mask_np).sum()),
        "restore_best_by_val_auc": bool(cfg.restore_best_by_val_auc),
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
