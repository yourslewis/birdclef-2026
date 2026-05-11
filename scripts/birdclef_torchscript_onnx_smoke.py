#!/usr/bin/env python3
"""Export a BirdCLEF TorchScript SED bundle member to ONNX and run a smoke benchmark.

This is intentionally small and dependency-light: it loads a bundle manifest,
selects one TorchScript model, wraps the model to export clip logits only, and
records PyTorch timing plus ONNX checker metadata. If onnxruntime is installed,
it also compares ONNX Runtime output against PyTorch.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


class ClipOnly(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 - torch export wrapper
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


def infer_time_frames(audio_config: dict[str, Any]) -> int:
    sr = int(audio_config["sample_rate"])
    duration = float(audio_config["duration_sec"])
    hop = int(audio_config["hop_length"])
    # torch.stft(center=True) produces roughly floor(n_samples / hop) + 1 frames.
    return int(round(sr * duration)) // hop + 1


def bench(fn, x: torch.Tensor, repeats: int, warmup: int = 1) -> tuple[np.ndarray, float]:
    with torch.no_grad():
        for _ in range(warmup):
            y = fn(x)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            y = fn(x)
            times.append(time.perf_counter() - t0)
    return y.detach().cpu().numpy(), float(np.mean(times))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--model-index", type=int, default=0)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--dynamic-batch", action="store_true")
    p.add_argument("--dynamo", action="store_true", help="Use the new torch.export-based ONNX exporter; default uses legacy exporter for TorchScript compatibility")
    args = p.parse_args()

    manifest = load_manifest(args.manifest)
    entries = manifest["models"]
    entry = entries[args.model_index]
    audio_config = dict(entry.get("audio_config") or manifest.get("audio_config") or {})
    n_mels = int(audio_config["n_mels"])
    n_frames = infer_time_frames(audio_config)

    model_path = Path(entry["path"])
    if not model_path.is_absolute():
        model_path = args.manifest.parent / model_path
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.output_dir / f"{model_path.stem}.clip_logits.onnx"
    metrics_path = args.output_dir / "onnx_smoke_metrics.json"

    torch.set_num_threads(2)
    device = torch.device("cpu")
    raw = torch.jit.load(str(model_path), map_location=device).eval()
    model = ClipOnly(raw).eval()
    x = torch.randn(args.batch_size, n_mels, n_frames, dtype=torch.float32, device=device)

    y_torch, torch_mean = bench(model, x, repeats=args.repeats)

    # Legacy ONNX export cannot always see a nested ScriptModule through a plain
    # Python wrapper. Tracing the clip-only wrapper first materializes a single
    # registered ScriptModule and worked for NFNet/TimmSEDB0 exports.
    export_model: torch.nn.Module = model
    traced_wrapper = False
    if not args.dynamo:
        export_model = torch.jit.trace(model, x)
        traced_wrapper = True

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {"logmel": {0: "batch"}, "clip_logits": {0: "batch"}}

    t0 = time.perf_counter()
    try:
        torch.onnx.export(
            export_model,
            x,
            str(onnx_path),
            input_names=["logmel"],
            output_names=["clip_logits"],
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=bool(args.dynamo),
        )
        export_sec = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001 - record export failure as a smoke result
        metrics = {
            "status": "export_failed",
            "manifest": str(args.manifest),
            "model_index": args.model_index,
            "member": entry.get("member"),
            "fold_index": entry.get("fold_index"),
            "model_path": str(model_path),
            "audio_config": audio_config,
            "input_shape": list(x.shape),
            "opset": args.opset,
            "dynamic_batch": bool(args.dynamic_batch),
            "dynamo": bool(args.dynamo),
            "traced_wrapper": traced_wrapper,
            "torch_mean_sec_per_batch": torch_mean,
            "torch_sec_per_clip": torch_mean / max(args.batch_size, 1),
            "export_error_type": type(exc).__name__,
            "export_error": str(exc)[:4000],
        }
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics, indent=2))
        return 2

    import onnx  # imported after export so missing checker is reported clearly

    checked = False
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    checked = True

    ort = None
    ort_mean = None
    max_abs_diff = None
    cosine = None
    try:
        import onnxruntime as ort_mod  # type: ignore

        sess = ort_mod.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        for _ in range(1):
            y_ort = sess.run(None, {"logmel": x.cpu().numpy()})[0]
        times = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            y_ort = sess.run(None, {"logmel": x.cpu().numpy()})[0]
            times.append(time.perf_counter() - t0)
        ort = True
        ort_mean = float(np.mean(times))
        max_abs_diff = float(np.max(np.abs(y_torch - y_ort)))
        cosine = float(np.dot(y_torch.reshape(-1), y_ort.reshape(-1)) / (np.linalg.norm(y_torch.reshape(-1)) * np.linalg.norm(y_ort.reshape(-1)) + 1e-12))
    except Exception as exc:  # noqa: BLE001 - smoke should record missing ORT/export mismatch, not crash
        ort = f"unavailable_or_failed: {type(exc).__name__}: {exc}"

    metrics = {
        "status": "complete",
        "manifest": str(args.manifest),
        "model_index": args.model_index,
        "member": entry.get("member"),
        "fold_index": entry.get("fold_index"),
        "model_path": str(model_path),
        "audio_config": audio_config,
        "input_shape": list(x.shape),
        "opset": args.opset,
        "dynamic_batch": bool(args.dynamic_batch),
        "dynamo": bool(args.dynamo),
        "traced_wrapper": traced_wrapper,
        "torch_mean_sec_per_batch": torch_mean,
        "torch_sec_per_clip": torch_mean / max(args.batch_size, 1),
        "export_sec": export_sec,
        "onnx_path": str(onnx_path),
        "onnx_size_mb": round(onnx_path.stat().st_size / 1024 / 1024, 3),
        "onnx_checked": checked,
        "onnxruntime": ort,
        "onnxruntime_mean_sec_per_batch": ort_mean,
        "onnxruntime_max_abs_diff": max_abs_diff,
        "onnxruntime_cosine": cosine,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
