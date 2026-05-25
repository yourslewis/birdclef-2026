#!/usr/bin/env python3
"""Train a bounded BirdCLEF train-soundscape specialist smoke.

This is intentionally a small data-point trainer for non-Aves / no-train
soundscape windows.  It uses the official train_soundscapes_labels.csv 5s
windows, trains a timm SED backbone or tiny fallback, and writes enough metrics
and artifacts to decide whether the branch deserves a stronger implementation.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover
    roc_auc_score = None

from birdclef_sed_pilot_train import (
    TimmSEDB0,
    TinySEDSmoke,
    compute_loss,
    ffmpeg_binary,
    load_initial_checkpoint,
    make_mel_filter,
    maybe_mixup,
    waveform_to_logmel,
)


@dataclass
class SoundscapeSpecialistConfig:
    experiment_id: str = "soundscape-nonaves-notrain-b0-5s160-siteS08-ep3-20260525"
    track: str = "Non-Aves/no-train soundscape specialist smoke"
    data_root: str = "/home/yourslewis/birdclef-2026/data"
    output_dir: str = "artifacts/soundscape_specialists/soundscape-nonaves-notrain-b0-5s160-siteS08-ep3-20260525"
    sample_rate: int = 32000
    duration_sec: float = 5.0
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 160
    backbone: str = "efficientnet_b0"
    pretrained: bool = False
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 3
    batch_size: int = 12
    max_windows: int = 0
    class_scope: str = "nonaves_or_no_train"  # nonaves_or_no_train | soundscape_positive | all
    val_site: str = "S08"
    min_val_windows: int = 50
    seed: int = 42
    loss_name: str = "bce"
    focal_gamma: float = 1.5
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0
    class_balancing: str = "none"
    num_workers: int = 0
    export_onnx: bool = True
    initial_checkpoint: str = "artifacts/external_pretrain/xc-b0-q3-cap80-external-pretrain-balanced-ep12/model_torchscript.pt"
    initial_load_head: bool = False
    restore_best_by_val_loss: bool = True


def load_config(path: Path | None) -> SoundscapeSpecialistConfig:
    cfg = SoundscapeSpecialistConfig()
    if path is None:
        return cfg
    data = json.loads(path.read_text())
    values = asdict(cfg)
    for key, value in data.items():
        if key in values:
            values[key] = value
    return SoundscapeSpecialistConfig(**values)


def parse_time_seconds(text: str) -> float:
    parts = [float(x) for x in str(text).split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def decode_window(path: Path, start_sec: float, cfg: SoundscapeSpecialistConfig) -> np.ndarray:
    samples = int(cfg.sample_rate * cfg.duration_sec)
    raw = subprocess.check_output([
        ffmpeg_binary(),
        "-v", "error",
        "-ss", f"{start_sec:.3f}",
        "-i", str(path),
        "-t", f"{cfg.duration_sec:.3f}",
        "-f", "f32le",
        "-ac", "1",
        "-ar", str(cfg.sample_rate),
        "-",
    ])
    y = np.frombuffer(raw, dtype=np.float32)
    if len(y) < samples:
        y = np.pad(y, (0, samples - len(y)))
    return y[:samples].astype(np.float32, copy=False)


def site_from_filename(name: str) -> str:
    m = re.search(r"_(S\d+)_", str(name))
    return m.group(1) if m else "UNKNOWN"


def choose_labels(data_root: Path, cfg: SoundscapeSpecialistConfig, soundscape_df: pd.DataFrame) -> tuple[list[str], dict[str, Any]]:
    taxonomy = pd.read_csv(data_root / "taxonomy.csv", dtype={"primary_label": str})
    train = pd.read_csv(data_root / "train.csv", dtype={"primary_label": str})
    all_labels = taxonomy["primary_label"].astype(str).tolist()
    train_labels = set(train["primary_label"].astype(str))
    no_train = {x for x in all_labels if x not in train_labels}
    nonaves = set(taxonomy.loc[taxonomy["class_name"].astype(str) != "Aves", "primary_label"].astype(str))
    positive = set()
    for raw in soundscape_df["primary_label"].fillna("").astype(str):
        positive.update(x.strip() for x in raw.split(";") if x.strip())

    if cfg.class_scope == "all":
        labels = all_labels
    elif cfg.class_scope == "soundscape_positive":
        labels = [x for x in all_labels if x in positive]
    elif cfg.class_scope == "nonaves_or_no_train":
        labels = [x for x in all_labels if x in nonaves or x in no_train]
    else:
        raise ValueError(f"Unknown class_scope={cfg.class_scope!r}")
    return labels, {
        "n_taxonomy_labels": len(all_labels),
        "n_train_primary_labels": len(train_labels),
        "n_no_train_labels": len(no_train),
        "n_nonaves_labels": len(nonaves),
        "n_soundscape_positive_labels": len(positive),
        "class_scope": cfg.class_scope,
        "n_training_labels": len(labels),
        "no_train_labels_in_scope": sorted(no_train & set(labels)),
        "soundscape_positive_labels_in_scope": sorted(positive & set(labels)),
    }


def make_dataset(cfg: SoundscapeSpecialistConfig) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], list[str], dict[str, Any]]:
    data_root = Path(cfg.data_root)
    soundscape_df = pd.read_csv(data_root / "train_soundscapes_labels.csv", dtype=str)
    labels, label_info = choose_labels(data_root, cfg, soundscape_df)
    label_to_idx = {label: i for i, label in enumerate(labels)}

    rows: list[dict[str, Any]] = []
    for r in soundscape_df.itertuples(index=False):
        filename = str(getattr(r, "filename"))
        path = data_root / "train_soundscapes" / filename
        if not path.exists():
            continue
        present = [x.strip() for x in str(getattr(r, "primary_label")).split(";") if x.strip()]
        target_indices = [label_to_idx[x] for x in present if x in label_to_idx]
        # Keep all rows: absence across scoped labels is useful negative evidence for this specialist.
        rows.append({
            "filename": filename,
            "path": str(path),
            "start": str(getattr(r, "start")),
            "start_sec": parse_time_seconds(str(getattr(r, "start"))),
            "end": str(getattr(r, "end")),
            "labels_raw": present,
            "target_indices": target_indices,
            "site": site_from_filename(filename),
        })
    if cfg.max_windows and cfg.max_windows > 0:
        rng = np.random.default_rng(cfg.seed)
        idx = rng.permutation(len(rows))[: cfg.max_windows]
        rows = [rows[int(i)] for i in idx]

    mel_fb = make_mel_filter(cfg.sample_rate, cfg.n_fft, cfg.n_mels)
    x_items = []
    y = torch.zeros((len(rows), len(labels)), dtype=torch.float32)
    decode_t0 = time.time()
    for i, item in enumerate(rows):
        wav = torch.from_numpy(decode_window(Path(item["path"]), float(item["start_sec"]), cfg))
        x_items.append(waveform_to_logmel(wav, cfg, mel_fb).to(torch.float32))
        if item["target_indices"]:
            y[i, torch.tensor(item["target_indices"], dtype=torch.long)] = 1.0
    x = torch.stack(x_items, dim=0)
    info = {
        **label_info,
        "n_windows": len(rows),
        "input_shape": list(x.shape),
        "target_positive_cells": int(y.sum().item()),
        "target_density": float(y.mean().item()),
        "decode_feature_seconds": float(time.time() - decode_t0),
        "site_counts": {k: int(v) for k, v in pd.Series([r["site"] for r in rows]).value_counts().sort_index().items()},
    }
    return x, y, rows, labels, info


def split_indices(rows: list[dict[str, Any]], cfg: SoundscapeSpecialistConfig) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    val_rows = [i for i, r in enumerate(rows) if r["site"] == cfg.val_site]
    if len(val_rows) < cfg.min_val_windows:
        rng = torch.Generator().manual_seed(cfg.seed)
        order = torch.randperm(len(rows), generator=rng)
        n_val = max(cfg.min_val_windows, int(round(0.2 * len(rows))))
        val_idx = order[:n_val]
        train_idx = order[n_val:]
        strategy = "random_fallback"
    else:
        val_mask = torch.zeros(len(rows), dtype=torch.bool)
        val_mask[torch.tensor(val_rows, dtype=torch.long)] = True
        val_idx = torch.arange(len(rows), dtype=torch.long)[val_mask]
        train_idx = torch.arange(len(rows), dtype=torch.long)[~val_mask]
        strategy = "site_holdout"
    return train_idx, val_idx, {
        "strategy": strategy,
        "val_site": cfg.val_site,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
    }


def macro_auc(y: np.ndarray, p: np.ndarray, labels: list[str], subset: set[str] | None = None) -> dict[str, Any]:
    if roc_auc_score is None:
        return {"macro_auc": None, "valid_classes": 0, "error": "sklearn unavailable"}
    aucs = []
    label_aucs: dict[str, float] = {}
    for j, label in enumerate(labels):
        if subset is not None and label not in subset:
            continue
        col = y[:, j]
        if col.min() == col.max():
            continue
        try:
            val = float(roc_auc_score(col, p[:, j]))
        except Exception:
            continue
        aucs.append(val)
        label_aucs[label] = val
    return {
        "macro_auc": float(np.mean(aucs)) if aucs else None,
        "valid_classes": int(len(aucs)),
        "per_label_auc_top10": dict(sorted(label_aucs.items(), key=lambda kv: kv[1], reverse=True)[:10]),
        "per_label_auc_bottom10": dict(sorted(label_aucs.items(), key=lambda kv: kv[1])[:10]),
    }


def predict_probs(model: torch.nn.Module, x: torch.Tensor, indices: torch.Tensor, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start + batch_size]
            logits, _ = model(x[idx].to(device))
            out.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 0), dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_dir / "config.input.json")
    (output_dir / "config.resolved.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")

    t0 = time.time()
    x, y, rows, labels, data_info = make_dataset(cfg)
    train_idx, val_idx, split_info = split_indices(rows, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimmSEDB0(len(labels), cfg.backbone, cfg.pretrained) if cfg.backbone != "tiny_cnn" else TinySEDSmoke(len(labels))
    init_info = load_initial_checkpoint(model, cfg)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    pos_weight = None
    if cfg.class_balancing == "pos_weight_sqrt":
        pos_weight = torch.full((len(labels),), float(np.sqrt(max(len(labels) - 1, 1))), device=device)
    elif cfg.class_balancing not in ("none", ""):
        raise ValueError(f"Unsupported class_balancing={cfg.class_balancing}")

    log_path = output_dir / "training_log.jsonl"
    best_val_loss = float("inf")
    best_state = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = train_idx[torch.randperm(len(train_idx))]
        losses = []
        for start in range(0, len(order), cfg.batch_size):
            idx = order[start:start + cfg.batch_size]
            xb = x[idx].to(device)
            yb = y[idx].to(device)
            xb, yb = maybe_mixup(xb, yb, cfg.mixup_alpha)
            logits, _ = model(xb)
            loss = compute_loss(logits, yb, cfg, pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for start in range(0, len(val_idx), cfg.batch_size):
                idx = val_idx[start:start + cfg.batch_size]
                logits, _ = model(x[idx].to(device))
                val_losses.append(float(F.binary_cross_entropy_with_logits(logits, y[idx].to(device)).detach().cpu()))
        train_loss = float(np.mean(losses)) if losses else None
        val_loss = float(np.mean(val_losses)) if val_losses else None
        rec = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(rec)
        with log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if cfg.restore_best_by_val_loss and best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    val_probs = predict_probs(model, x, val_idx, cfg.batch_size, device)
    val_y = y[val_idx].numpy()
    taxonomy = pd.read_csv(Path(cfg.data_root) / "taxonomy.csv", dtype={"primary_label": str})
    train = pd.read_csv(Path(cfg.data_root) / "train.csv", dtype={"primary_label": str})
    no_train = {l for l in taxonomy["primary_label"].astype(str) if l not in set(train["primary_label"].astype(str))}
    nonaves = set(taxonomy.loc[taxonomy["class_name"].astype(str) != "Aves", "primary_label"].astype(str))

    metrics = {
        "status": "complete",
        "experiment_id": cfg.experiment_id,
        "track": cfg.track,
        "runtime_seconds": float(time.time() - t0),
        "device": str(device),
        "data": data_info,
        "split": split_info,
        "initial_checkpoint": init_info,
        "history": history,
        "best_val_loss": float(best_val_loss),
        "val_macro_auc_all_scope": macro_auc(val_y, val_probs, labels),
        "val_macro_auc_no_train": macro_auc(val_y, val_probs, labels, no_train),
        "val_macro_auc_nonaves": macro_auc(val_y, val_probs, labels, nonaves),
        "prediction_stats": {
            "min": float(val_probs.min()) if val_probs.size else None,
            "max": float(val_probs.max()) if val_probs.size else None,
            "mean": float(val_probs.mean()) if val_probs.size else None,
            "std": float(val_probs.std()) if val_probs.size else None,
        },
        "evidence_level": "comparison-grade training data point; not submission-grade",
        "decision_hint": "Use as measured non-Aves/no-train landscape point; needs branch emission + v616 audit before any submit consideration.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(
        output_dir / "holdout_predictions.npz",
        files=np.array([rows[int(i)]["filename"] for i in val_idx], dtype=str),
        starts=np.array([rows[int(i)]["start"] for i in val_idx], dtype=str),
        sites=np.array([rows[int(i)]["site"] for i in val_idx], dtype=str),
        labels=np.array(labels, dtype=str),
        y_true=val_y.astype(np.float32),
        pred=val_probs.astype(np.float32),
    )

    model_cpu = model.to("cpu").eval()
    sample = x[: min(2, len(x))].to("cpu")
    traced = torch.jit.trace(model_cpu, sample, strict=False)
    traced.save(str(output_dir / "model_torchscript.pt"))
    if cfg.export_onnx:
        try:
            torch.onnx.export(model_cpu, sample, str(output_dir / "model.onnx"), input_names=["logmel"], output_names=["clip_logits", "frame_logits"], opset_version=17)
            metrics["onnx_export"] = {"ok": True, "path": str(output_dir / "model.onnx")}
        except Exception as exc:  # pragma: no cover
            metrics["onnx_export"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
