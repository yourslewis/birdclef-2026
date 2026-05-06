#!/usr/bin/env python3
"""Lightweight TorchScript SED inference smoke runner.

This intentionally avoids importing the training module or timm. It validates that
exported fold TorchScript files can be loaded from a bundle manifest, decode OGG
audio with ffmpeg, reproduce the log-mel preprocessing, and emit averaged class
probabilities for a few files. It is a packaging bridge toward a Kaggle inference
kernel/dataset.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


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


def decode_audio_ffmpeg(path: Path, sr: int, samples: int, start_sec: float = 0.0) -> np.ndarray:
    cmd = [ffmpeg_binary(), "-v", "error"]
    if start_sec > 0:
        cmd += ["-ss", str(start_sec)]
    cmd += ["-i", str(path), "-t", str(samples / sr), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
    raw = subprocess.check_output(cmd)
    y = np.frombuffer(raw, dtype=np.float32)
    if len(y) < samples:
        y = np.pad(y, (0, samples - len(y)))
    return y[:samples].astype(np.float32, copy=False)


def waveform_to_logmel(wave: torch.Tensor, audio_cfg: dict[str, Any], mel_fb: torch.Tensor) -> torch.Tensor:
    n_fft = int(audio_cfg["n_fft"])
    hop_length = int(audio_cfg["hop_length"])
    window = torch.hann_window(n_fft, device=wave.device)
    spec = torch.stft(wave, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window, center=True, return_complex=True)
    mel = mel_fb.to(wave.device) @ spec.abs().pow(2.0)
    logmel = torch.log1p(mel)
    return (logmel - logmel.mean()) / (logmel.std() + 1e-6)


def collect_audio(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for raw in args.audio or []:
        paths.append(Path(raw))
    if args.audio_dir:
        paths.extend(sorted(Path(args.audio_dir).glob("*/*.ogg")))
        paths.extend(sorted(Path(args.audio_dir).glob("*.ogg")))
    # Stable de-duplication.
    seen = set()
    unique = []
    for p in paths:
        key = str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    if args.max_files is not None:
        unique = unique[: args.max_files]
    if not unique:
        raise ValueError("No audio files provided")
    return unique


def load_models(manifest: dict[str, Any], manifest_dir: Path, device: torch.device):
    loaded = []
    for entry in manifest["models"]:
        path = Path(entry["path"])
        if not path.is_absolute():
            path = manifest_dir / path
        model = torch.jit.load(str(path), map_location=device).eval()
        loaded.append((entry, model))
    return loaded


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio", action="append", help="Audio file; repeatable")
    parser.add_argument("--audio-dir", type=Path, help="Directory containing OGGs, optionally class subdirectories")
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--output", type=Path, required=True, help="Wide CSV probabilities")
    parser.add_argument("--npz-output", type=Path, help="Optional compressed NPZ probabilities")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.torch_threads))
    manifest = json.loads(args.manifest.read_text())
    manifest_dir = args.manifest.parent
    labels = [str(x) for x in manifest["labels"]]
    audio_cfg = manifest["audio_config"]
    sr = int(audio_cfg["sample_rate"])
    samples = int(round(sr * float(audio_cfg["duration_sec"])))
    device = torch.device(args.device)
    mel_fb = make_mel_filter(sr, int(audio_cfg["n_fft"]), int(audio_cfg["n_mels"]))
    models = load_models(manifest, manifest_dir, device)
    weight_sum = float(sum(float(entry.get("weight", 0.0)) for entry, _ in models))
    if weight_sum <= 0:
        raise ValueError("Bundle model weights sum to zero")

    files = collect_audio(args)
    rows = []
    probs_all = []
    start = time.time()
    with torch.no_grad():
        for path in files:
            wave = torch.from_numpy(decode_audio_ffmpeg(path, sr, samples).copy())
            x = waveform_to_logmel(wave, audio_cfg, mel_fb).unsqueeze(0).to(device)
            probs = np.zeros((len(labels),), dtype=np.float32)
            for entry, model in models:
                clip_logits, _frame_logits = model(x)
                p = torch.sigmoid(clip_logits).detach().cpu().numpy()[0].astype(np.float32)
                probs += (float(entry.get("weight", 0.0)) / weight_sum) * p
            probs_all.append(probs)
            top_idx = np.argsort(-probs)[: args.top_k]
            row = {"file": str(path), "top_labels": ";".join(labels[i] for i in top_idx), "top_probs": ";".join(f"{probs[i]:.6f}" for i in top_idx)}
            row.update({label: float(probs[i]) for i, label in enumerate(labels)})
            rows.append(row)
    probs_arr = np.stack(probs_all)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    if args.npz_output:
        args.npz_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.npz_output, files=np.array([str(p) for p in files]), labels=np.array(labels), probs=probs_arr)
    print(json.dumps({
        "status": "inference_complete",
        "n_files": len(files),
        "n_models": len(models),
        "n_classes": len(labels),
        "output": str(args.output),
        "npz_output": str(args.npz_output) if args.npz_output else None,
        "runtime_sec": round(time.time() - start, 3),
        "sec_per_file": round((time.time() - start) / max(len(files), 1), 3),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
