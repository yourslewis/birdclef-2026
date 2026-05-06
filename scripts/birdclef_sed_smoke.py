#!/usr/bin/env python3
"""BirdCLEF Spec A+G SED smoke training/export scaffold.

This is intentionally small and dependency-light. It validates the real-audio
SED pipeline before we spend GPU time on EfficientNet/timm models:
- decode 3-5 real OGG clips via ffmpeg
- build log-mel features with torch.stft + an internal mel filterbank
- train a weak-label frame/event model for a smoke epoch
- export TorchScript and ONNX when the onnx package is available
- write artifacts + metrics in the AutoResearch log format
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SmokeConfig:
    experiment_id: str = "sed-b0-5s-attn-v1-smoke"
    data_root: str = "/Volumes/ExternalSSD/data/workspace_don/kaggle_birdclef2026/data"
    output_dir: str = "artifacts/sed_smoke/sed-b0-5s-attn-v1-smoke"
    sample_rate: int = 32000
    duration_sec: float = 5.0
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 128
    learning_rate: float = 3e-4
    epochs: int = 1
    batch_size: int = 2
    max_files: int = 5
    seed: int = 42
    export_onnx: bool = True


def load_config(path: Path | None) -> SmokeConfig:
    cfg = SmokeConfig()
    if path is None:
        return cfg
    data = json.loads(path.read_text())
    allowed = set(SmokeConfig.__annotations__)
    values = asdict(cfg)
    for key, value in data.items():
        if key in allowed:
            values[key] = value
    return SmokeConfig(**values)


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def make_mel_filter(sr: int, n_fft: int, n_mels: int, fmin: float = 20.0, fmax: float | None = None) -> torch.Tensor:
    if fmax is None:
        fmax = sr / 2
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    n_freq = n_fft // 2 + 1
    fb = np.zeros((n_mels, n_freq), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for k in range(left, min(center, n_freq)):
            fb[m - 1, k] = (k - left) / max(center - left, 1)
        for k in range(center, min(right, n_freq)):
            fb[m - 1, k] = (right - k) / max(right - center, 1)
    return torch.from_numpy(fb)


def decode_audio_ffmpeg(path: Path, sr: int, samples: int) -> np.ndarray:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for OGG decode but was not found on PATH")
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-",
    ]
    raw = subprocess.check_output(cmd)
    y = np.frombuffer(raw, dtype=np.float32)
    if len(y) < samples:
        y = np.pad(y, (0, samples - len(y)))
    return y[:samples].astype(np.float32, copy=False)


def waveform_to_logmel(wave: torch.Tensor, cfg: SmokeConfig, mel_fb: torch.Tensor) -> torch.Tensor:
    window = torch.hann_window(cfg.n_fft, device=wave.device)
    spec = torch.stft(
        wave,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.n_fft,
        window=window,
        center=True,
        return_complex=True,
    )
    power = spec.abs().pow(2.0)
    mel = mel_fb.to(wave.device) @ power
    logmel = torch.log1p(mel)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
    return logmel


class TinySEDSmoke(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1)),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.frame_head = nn.Conv1d(64, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: B, n_mels, frames
        z = self.features(x.unsqueeze(1))
        z = z.mean(dim=2)  # B, C, frames
        frame_logits = self.frame_head(z).transpose(1, 2)  # B, frames, classes
        clip_logits = 0.5 * frame_logits.mean(dim=1) + 0.5 * frame_logits.amax(dim=1)
        return clip_logits, frame_logits


def select_examples(data_root: Path, labels: list[str], max_files: int, seed: int) -> list[tuple[Path, str]]:
    rng = random.Random(seed)
    train_audio = data_root / "train_audio"
    examples: list[tuple[Path, str]] = []
    for label_dir in sorted(train_audio.iterdir()):
        if not label_dir.is_dir() or label_dir.name not in labels:
            continue
        files = sorted(label_dir.glob("*.ogg"))
        if files:
            examples.append((rng.choice(files), label_dir.name))
        if len(examples) >= max_files:
            break
    if len(examples) < max_files:
        all_files = [(p, p.parent.name) for p in train_audio.glob("*/*.ogg") if p.parent.name in labels]
        rng.shuffle(all_files)
        seen = {p for p, _ in examples}
        for p, label in all_files:
            if p not in seen:
                examples.append((p, label))
            if len(examples) >= max_files:
                break
    return examples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_files is not None:
        cfg.max_files = args.max_files
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.output_dir is not None:
        cfg.output_dir = str(args.output_dir)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    data_root = Path(cfg.data_root)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    taxonomy = pd.read_csv(data_root / "taxonomy.csv")
    labels = taxonomy["primary_label"].astype(str).tolist()
    label_to_idx = {label: i for i, label in enumerate(labels)}
    examples = select_examples(data_root, labels, cfg.max_files, cfg.seed)
    if not examples:
        raise RuntimeError(f"No examples found under {data_root / 'train_audio'}")

    n_samples = int(cfg.sample_rate * cfg.duration_sec)
    mel_fb = make_mel_filter(cfg.sample_rate, cfg.n_fft, cfg.n_mels)
    xs, ys, meta = [], [], []
    for path, label in examples:
        y = decode_audio_ffmpeg(path, cfg.sample_rate, n_samples)
        logmel = waveform_to_logmel(torch.from_numpy(y.copy()), cfg, mel_fb)
        target = torch.zeros(len(labels), dtype=torch.float32)
        target[label_to_idx[label]] = 1.0
        xs.append(logmel)
        ys.append(target)
        meta.append({"path": str(path), "label": label, "frames": int(logmel.shape[-1])})

    x = torch.stack(xs)
    y = torch.stack(ys)
    model = TinySEDSmoke(n_classes=len(labels))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    losses = []
    start = time.time()
    model.train()
    for epoch in range(cfg.epochs):
        order = torch.randperm(len(x))
        for i in range(0, len(order), cfg.batch_size):
            idx = order[i:i + cfg.batch_size]
            logits, frame_logits = model(x[idx])
            loss = F.binary_cross_entropy_with_logits(logits, y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))

    model.eval()
    with torch.no_grad():
        clip_logits, frame_logits = model(x)
        probs = torch.sigmoid(clip_logits)
        top_prob, top_idx = probs.max(dim=1)

    example_input = x[:1]
    traced = torch.jit.trace(model, example_input)
    torchscript_path = out_dir / "tiny_sed_smoke_torchscript.pt"
    traced.save(str(torchscript_path))

    onnx_path = out_dir / "tiny_sed_smoke.onnx"
    onnx_status = "skipped_onnx_package_not_available"
    if cfg.export_onnx:
        try:
            import onnx  # noqa: F401
            torch.onnx.export(
                model,
                example_input,
                str(onnx_path),
                input_names=["logmel"],
                output_names=["clip_logits", "frame_logits"],
                dynamic_axes={"logmel": {0: "batch", 2: "frames"}, "clip_logits": {0: "batch"}, "frame_logits": {0: "batch", 1: "frames"}},
                opset_version=17,
            )
            onnx_status = "exported"
        except Exception as exc:  # pragma: no cover - artifact logging path
            onnx_status = f"failed: {type(exc).__name__}: {exc}"

    metrics = {
        "experiment_id": cfg.experiment_id,
        "track": "A+G Real SED frame/event smoke + export packaging",
        "status": "smoke_passed",
        "n_examples": len(examples),
        "n_classes": len(labels),
        "input_shape": list(x.shape),
        "frame_logits_shape": list(frame_logits.shape),
        "loss_start": losses[0] if losses else None,
        "loss_end": losses[-1] if losses else None,
        "runtime_sec": round(time.time() - start, 3),
        "torchscript_path": str(torchscript_path),
        "onnx_path": str(onnx_path) if onnx_status == "exported" else None,
        "onnx_status": onnx_status,
        "examples": meta,
        "top_predictions": [
            {"file": Path(item["path"]).name, "true_label": item["label"], "top_label": labels[int(idx)], "top_prob": float(prob)}
            for item, idx, prob in zip(meta, top_idx, top_prob)
        ],
        "config": asdict(cfg),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out_dir / "config.resolved.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
