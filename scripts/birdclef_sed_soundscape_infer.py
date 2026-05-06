#!/usr/bin/env python3
"""Kaggle-style 5s soundscape inference for TorchScript SED bundles.

This converts the manifest bundle created by birdclef_sed_build_bundle.py into
row-level BirdCLEF predictions. It is intentionally dependency-light and does not
import timm/training code:
- load one or more TorchScript fold models from a manifest
- decode 60s OGG soundscapes with ffmpeg
- create one prediction row per 5s endpoint: <stem>_5, ..., <stem>_60
- feed each row a fixed-duration context window (default: model duration, 10s)
  ending at the row endpoint, with zero-padding at file boundaries
- average fold/member probabilities using manifest weights
- align to sample_submission columns when provided

This script is a bridge toward a Kaggle kernel. It can dry-run on train
soundscapes locally/server-side and can write a real submission.csv when test
soundscapes are available.
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


ROW_END_SECONDS = tuple(range(5, 65, 5))


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


def decode_soundscape(path: Path, sr: int, target_samples: int) -> np.ndarray:
    raw = subprocess.check_output([ffmpeg_binary(), "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"])
    y = np.frombuffer(raw, dtype=np.float32).copy()
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)))
    return y[:target_samples].astype(np.float32, copy=False)


def extract_context(audio: np.ndarray, sr: int, end_sec: int, context_sec: float) -> np.ndarray:
    n = int(round(sr * context_sec))
    end = int(round(sr * end_sec))
    start = end - n
    out = np.zeros(n, dtype=np.float32)
    src_start = max(start, 0)
    src_end = min(end, len(audio))
    if src_end > src_start:
        dst_start = src_start - start
        out[dst_start:dst_start + (src_end - src_start)] = audio[src_start:src_end]
    return out


def waveform_to_logmel(wave: torch.Tensor, audio_cfg: dict[str, Any], mel_fb: torch.Tensor) -> torch.Tensor:
    n_fft = int(audio_cfg["n_fft"])
    hop_length = int(audio_cfg["hop_length"])
    window = torch.hann_window(n_fft, device=wave.device)
    spec = torch.stft(wave, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window, center=True, return_complex=True)
    mel = mel_fb.to(wave.device) @ spec.abs().pow(2.0)
    logmel = torch.log1p(mel)
    return (logmel - logmel.mean()) / (logmel.std() + 1e-6)


def load_models(manifest: dict[str, Any], manifest_dir: Path, device: torch.device):
    loaded = []
    for entry in manifest["models"]:
        path = Path(entry["path"])
        if not path.is_absolute():
            path = manifest_dir / path
        model = torch.jit.load(str(path), map_location=device).eval()
        loaded.append((entry, model))
    return loaded


def find_soundscapes(args: argparse.Namespace, base: Path) -> tuple[list[Path], bool]:
    if args.soundscape:
        return [Path(p) for p in args.soundscape], False
    if args.soundscape_dir:
        return sorted(Path(args.soundscape_dir).glob("*.ogg")), False
    test_dir = base / "test_soundscapes"
    if test_dir.exists():
        test_paths = sorted(test_dir.glob("*.ogg"))
        if test_paths:
            return test_paths, True
    train_paths = sorted((base / "train_soundscapes").glob("*.ogg"))
    return train_paths, False


def infer_one_file(path: Path, models, labels: list[str], audio_cfg: dict[str, Any], mel_fb: torch.Tensor, device: torch.device, batch_size: int) -> tuple[list[str], np.ndarray]:
    sr = int(audio_cfg["sample_rate"])
    context_sec = float(audio_cfg["duration_sec"])
    file_samples = int(round(sr * 60.0))
    audio = decode_soundscape(path, sr, file_samples)
    windows = [waveform_to_logmel(torch.from_numpy(extract_context(audio, sr, end_sec, context_sec).copy()), audio_cfg, mel_fb) for end_sec in ROW_END_SECONDS]
    x_all = torch.stack(windows)
    row_probs = np.zeros((len(ROW_END_SECONDS), len(labels)), dtype=np.float32)
    weight_sum = float(sum(float(entry.get("weight", 0.0)) for entry, _ in models))
    with torch.no_grad():
        for entry, model in models:
            model_probs = []
            for start in range(0, len(x_all), batch_size):
                xb = x_all[start:start + batch_size].to(device)
                clip_logits, _frame_logits = model(xb)
                model_probs.append(torch.sigmoid(clip_logits).cpu().numpy().astype(np.float32))
            row_probs += (float(entry.get("weight", 0.0)) / weight_sum) * np.concatenate(model_probs, axis=0)
    row_ids = [f"{path.stem}_{end_sec}" for end_sec in ROW_END_SECONDS]
    return row_ids, row_probs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=Path("/kaggle/input/birdclef-2026"))
    parser.add_argument("--soundscape", action="append", help="Specific soundscape OGG; repeatable")
    parser.add_argument("--soundscape-dir", type=Path)
    parser.add_argument("--sample-submission", type=Path)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--npz-output", type=Path)
    parser.add_argument("--max-files", type=int, help="Dry-run limit")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    start_all = time.time()
    torch.set_num_threads(max(1, args.torch_threads))
    manifest = json.loads(args.manifest.read_text())
    manifest_dir = args.manifest.parent
    labels = [str(x) for x in manifest["labels"]]
    audio_cfg = manifest["audio_config"]
    device = torch.device(args.device)
    mel_fb = make_mel_filter(int(audio_cfg["sample_rate"]), int(audio_cfg["n_fft"]), int(audio_cfg["n_mels"]))
    models = load_models(manifest, manifest_dir, device)

    paths, is_test = find_soundscapes(args, args.base_dir)
    if args.max_files is not None:
        paths = paths[: args.max_files]
    if not paths:
        raise ValueError("No soundscape OGG files found")

    all_row_ids: list[str] = []
    all_probs: list[np.ndarray] = []
    for path in paths:
        row_ids, probs = infer_one_file(path, models, labels, audio_cfg, mel_fb, device, args.batch_size)
        all_row_ids.extend(row_ids)
        all_probs.append(probs)
    probs_arr = np.vstack(all_probs)
    probs_arr = np.clip(probs_arr, 0.0, 1.0)

    sample_path = args.sample_submission
    if sample_path is None:
        candidate = args.base_dir / "sample_submission.csv"
        sample_path = candidate if candidate.exists() else None
    primary_labels = labels
    if sample_path and sample_path.exists():
        sample_sub = pd.read_csv(sample_path)
        primary_labels = sample_sub.columns[1:].astype(str).tolist()
        # Align class columns and optionally row IDs for real test mode.
        pred = pd.DataFrame(probs_arr, columns=labels)
        pred.insert(0, "row_id", all_row_ids)
        for col in primary_labels:
            if col not in pred.columns:
                pred[col] = 0.0
        pred = pred[["row_id"] + primary_labels]
        if is_test:
            pred = pred.set_index("row_id").reindex(sample_sub["row_id"].values, fill_value=0.0).reset_index().rename(columns={"index": "row_id"})
    else:
        pred = pd.DataFrame(probs_arr, columns=primary_labels)
        pred.insert(0, "row_id", all_row_ids)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.output, index=False)
    if args.npz_output:
        args.npz_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.npz_output, row_ids=np.array(pred["row_id"].astype(str)), labels=np.array(primary_labels), probs=pred[primary_labels].to_numpy(np.float32))

    top_records = []
    for i in range(min(3, len(pred))):
        vals = pred.iloc[i][primary_labels].to_numpy(dtype=np.float32)
        idx = np.argsort(-vals)[: args.top_k]
        top_records.append({"row_id": pred.iloc[i]["row_id"], "top_labels": [primary_labels[j] for j in idx], "top_probs": [round(float(vals[j]), 6) for j in idx]})
    elapsed = time.time() - start_all
    print(json.dumps({
        "status": "soundscape_inference_complete",
        "is_test": is_test,
        "n_files": len(paths),
        "n_rows": len(pred),
        "n_models": len(models),
        "n_classes": len(primary_labels),
        "output": str(args.output),
        "npz_output": str(args.npz_output) if args.npz_output else None,
        "runtime_sec": round(elapsed, 3),
        "sec_per_file": round(elapsed / max(len(paths), 1), 3),
        "top_records": top_records,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
