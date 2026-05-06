#!/usr/bin/env python3
"""Run a small AutoResearch-style SED smoke hyperparameter sweep.

This intentionally stays CPU/lightweight. It is the bridge between the first
3-5 file smoke and a real GPU EfficientNet/timm pilot: each variant still uses
real BirdCLEF audio, writes resolved configs/metrics, and checks export.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


BASE_CONFIG = {
    "data_root": "/Volumes/ExternalSSD/data/workspace_don/kaggle_birdclef2026/data",
    "sample_rate": 32000,
    "n_fft": 1024,
    "hop_length": 512,
    "learning_rate": 3e-4,
    "epochs": 2,
    "batch_size": 4,
    "max_files": 24,
    "seed": 42,
    "val_fraction": 0.2,
    "export_onnx": True,
}

VARIANTS = [
    {
        "experiment_id": "sed-smoke-sweep-v2-5s-bce-m128",
        "duration_sec": 5.0,
        "n_mels": 128,
        "loss_name": "bce",
        "label_smoothing": 0.0,
        "mixup_alpha": 0.0,
        "class_balancing": "none",
    },
    {
        "experiment_id": "sed-smoke-sweep-v2-5s-focal15-possqrt",
        "duration_sec": 5.0,
        "n_mels": 128,
        "loss_name": "focal_bce",
        "focal_gamma": 1.5,
        "label_smoothing": 0.0,
        "mixup_alpha": 0.0,
        "class_balancing": "pos_weight_sqrt",
    },
    {
        "experiment_id": "sed-smoke-sweep-v2-5s-bce-smooth001-mixup02",
        "duration_sec": 5.0,
        "n_mels": 128,
        "loss_name": "bce",
        "label_smoothing": 0.01,
        "mixup_alpha": 0.2,
        "class_balancing": "none",
    },
    {
        "experiment_id": "sed-smoke-sweep-v2-10s-bce-m160",
        "duration_sec": 10.0,
        "n_mels": 160,
        "loss_name": "bce",
        "label_smoothing": 0.0,
        "mixup_alpha": 0.0,
        "class_balancing": "none",
    },
]


def run_variant(repo_root: Path, output_root: Path, variant: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(BASE_CONFIG)
    cfg.update(variant)
    out_dir = output_root / cfg["experiment_id"]
    cfg["output_dir"] = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    cmd = [sys.executable, str(repo_root / "scripts" / "birdclef_sed_smoke.py"), "--config", str(cfg_path)]
    start = time.time()
    proc = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True)
    (out_dir / "stdout.txt").write_text(proc.stdout)
    (out_dir / "stderr.txt").write_text(proc.stderr)
    if proc.returncode != 0:
        return {
            "experiment_id": cfg["experiment_id"],
            "status": "failed",
            "returncode": proc.returncode,
            "runtime_sec": round(time.time() - start, 3),
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-4000:],
            "config": cfg,
        }
    metrics = json.loads((out_dir / "metrics.json").read_text())
    metrics["status"] = metrics.get("status", "smoke_passed")
    metrics["sweep_runtime_sec"] = round(time.time() - start, 3)
    return metrics


def write_summary(output_root: Path, results: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "sweep_results.json").write_text(json.dumps(results, indent=2) + "\n")
    passed = [r for r in results if r.get("status") == "smoke_passed"]
    ranked = sorted(passed, key=lambda r: (r.get("val_loss_final") is None, r.get("val_loss_final", 1e9)))
    lines = [
        "# SED Smoke Sweep v2 Results",
        "",
        "Tiny CPU smoke sweep over real BirdCLEF audio. Ranking is operational only; it is not a true model-selection signal yet.",
        "",
        "| Rank | Experiment | Status | n | Input | Loss | Train loss | Val loss | ONNX |",
        "|---:|---|---|---:|---|---|---:|---:|---|",
    ]
    for rank, item in enumerate(ranked, start=1):
        cfg = item.get("config", {})
        input_shape = item.get("input_shape")
        input_text = "x".join(map(str, input_shape)) if input_shape else "?"
        lines.append(
            f"| {rank} | `{item.get('experiment_id')}` | {item.get('status')} | {item.get('n_examples')} | "
            f"{input_text} | {cfg.get('loss_name')} | {item.get('train_loss_final'):.5f} | "
            f"{item.get('val_loss_final'):.5f} | {item.get('onnx_status')} |"
        )
    failed = [r for r in results if r.get("status") != "smoke_passed"]
    if failed:
        lines.extend(["", "## Failed variants", ""])
        for item in failed:
            lines.append(f"- `{item.get('experiment_id')}`: returncode={item.get('returncode')} stderr={item.get('stderr_tail', '')[-400:]}")
    (output_root / "sweep_summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/sed_smoke/sweep-v2"))
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_root = args.output_root
    results = []
    for variant in VARIANTS:
        print(f"Running {variant['experiment_id']}...", flush=True)
        result = run_variant(repo_root, output_root, variant)
        print(json.dumps({
            "experiment_id": result.get("experiment_id"),
            "status": result.get("status"),
            "train_loss_final": result.get("train_loss_final"),
            "val_loss_final": result.get("val_loss_final"),
            "onnx_status": result.get("onnx_status"),
        }, indent=2), flush=True)
        results.append(result)
    write_summary(output_root, results)
    failed = [r for r in results if r.get("status") != "smoke_passed"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
