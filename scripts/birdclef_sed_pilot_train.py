#!/usr/bin/env python3
"""BirdCLEF Spec A+G SED GPU pilot trainer.

A small but real weak-label SED pilot that scales the smoke scaffold toward the
spec's EfficientNet-B0/timm direction while keeping a tiny-CNN fallback for
preflight environments. It writes the required AutoResearch artifacts:
config/log/holdout predictions, macro AUC diagnostics, timing, and exports.
"""
from __future__ import annotations

import argparse
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
    seed: int = 42
    val_fraction: float = 0.2
    loss_name: str = "focal_bce"
    focal_gamma: float = 1.5
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0
    class_balancing: str = "pos_weight_sqrt"
    num_workers: int = 0
    export_onnx: bool = True


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


def select_examples(data_root: Path, labels: list[str], max_files: int, seed: int) -> list[tuple[Path, str]]:
    rng = random.Random(seed)
    train_audio = data_root / "train_audio"
    selected: list[tuple[Path, str]] = []
    for label_dir in sorted(train_audio.iterdir()):
        if not label_dir.is_dir() or label_dir.name not in labels:
            continue
        files = sorted(label_dir.glob("*.ogg"))
        if files:
            selected.append((rng.choice(files), label_dir.name))
        if len(selected) >= max_files:
            return selected
    all_files = [(p, p.parent.name) for p in train_audio.glob("*/*.ogg") if p.parent.name in labels]
    rng.shuffle(all_files)
    seen = {p for p, _ in selected}
    for path, label in all_files:
        if path not in seen:
            selected.append((path, label))
        if len(selected) >= max_files:
            break
    return selected


def make_targets(meta: list[dict[str, Any]], labels: list[str]) -> torch.Tensor:
    label_to_idx = {label: i for i, label in enumerate(labels)}
    y = torch.zeros((len(meta), len(labels)), dtype=torch.float32)
    for i, item in enumerate(meta):
        y[i, label_to_idx[item["label"]]] = 1.0
    return y


def split_indices(n_items: int, val_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    order = torch.randperm(n_items, generator=torch.Generator().manual_seed(seed))
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
    for key in ["data_root", "output_dir", "max_files", "epochs", "batch_size", "backbone"]:
        val = getattr(args, key.replace("-", "_"), None)
        if val is not None:
            setattr(cfg, key, val)

    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    start_all = time.time()
    taxonomy = pd.read_csv(Path(cfg.data_root) / "taxonomy.csv")
    labels = taxonomy["primary_label"].astype(str).tolist()
    examples = select_examples(Path(cfg.data_root), labels, cfg.max_files, cfg.seed)
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
    train_idx, val_idx = split_indices(len(x), cfg.val_fraction, cfg.seed)

    model = build_model(cfg, len(labels)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    pos_weight = make_pos_weight(len(labels), cfg.class_balancing, device)
    epoch_logs = []
    for epoch in range(cfg.epochs):
        model.train(); losses = []
        for idx in batch_iter(train_idx, cfg.batch_size, shuffle=True):
            bx, by = x[idx].to(device), y[idx].to(device)
            bx, by = maybe_mixup(bx, by, cfg.mixup_alpha)
            logits, _ = model(bx)
            loss = compute_loss(logits, by, cfg, pos_weight)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(float(loss.detach().cpu()))
        val_probs, _ = predict(model, x, val_idx, cfg.batch_size, device)
        val_loss = compute_loss(torch.logit(torch.from_numpy(val_probs).clamp(1e-6, 1 - 1e-6)).to(device), y[val_idx].to(device), cfg, pos_weight)
        epoch_logs.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": float(val_loss.detach().cpu())})
        print(json.dumps(epoch_logs[-1]), flush=True)

    val_probs, pred_time = predict(model, x, val_idx, cfg.batch_size, device)
    train_probs, _ = predict(model, x, train_idx, cfg.batch_size, device)
    auc = auc_summary(y[val_idx].numpy(), val_probs)
    npz_path = out_dir / "holdout_predictions.npz"
    np.savez_compressed(npz_path, val_indices=val_idx.numpy(), train_indices=train_idx.numpy(), y_val=y[val_idx].numpy(), pred_val=val_probs, pred_train=train_probs, labels=np.array(labels), files=np.array([m["path"] for m in meta]))
    exports = export_model(model, x[:1], out_dir, cfg)
    metrics = {
        "experiment_id": cfg.experiment_id,
        "track": "A+G Real SED frame/event GPU pilot",
        "status": "pilot_complete",
        "device": str(device),
        "backbone_actual": getattr(model, "backbone_name", cfg.backbone),
        "n_examples": len(examples),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_classes": len(labels),
        "input_shape": list(x.shape),
        "epochs": epoch_logs,
        "auc_summary": auc,
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
