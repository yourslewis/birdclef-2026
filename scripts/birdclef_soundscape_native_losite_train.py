#!/usr/bin/env python3
"""Compact soundscape-native CNN/SED leave-site trainer for BirdCLEF 2026.

This is a no-slot ClawTeam model data-point script.  Unlike the frozen
embedding heads, it fine-tunes a compact SED backbone directly on official
train_soundscapes 5s windows and evaluates with leave-site/file gates.

It intentionally emits comparison-grade artifacts only; it is not a Kaggle
submission package.
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
class NativeLoSiteConfig:
    experiment_id: str = "soundscape-native-b0-losite-nonaves-notrain-ep4-20260526"
    track: str = "Deeper soundscape-native compact CNN/SED leave-site data point"
    data_root: str = "/home/yourslewis/birdclef-2026/data"
    output_dir: str = "artifacts/soundscape_native_losite/soundscape-native-b0-losite-nonaves-notrain-ep4-20260526"
    sample_rate: int = 32000
    duration_sec: float = 5.0
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 160
    backbone: str = "efficientnet_b0"
    pretrained: bool = False
    learning_rate: float = 2e-4
    weight_decay: float = 3e-4
    epochs: int = 4
    final_train_epochs: int = 4
    batch_size: int = 12
    max_windows: int = 0
    class_scope: str = "nonaves_or_no_train"  # nonaves_or_no_train | soundscape_positive | all
    min_val_windows: int = 40
    min_valid_classes: int = 4
    seed: int = 43
    loss_name: str = "bce"
    focal_gamma: float = 1.5
    label_smoothing: float = 0.01
    mixup_alpha: float = 0.1
    class_balancing: str = "pos_weight_sqrt"  # none | pos_weight_sqrt | observed_sqrt
    pos_weight_clip: float = 12.0
    num_workers: int = 0
    export_onnx: bool = True
    initial_checkpoint: str = "artifacts/external_pretrain/xc-b0-q3-cap80-external-pretrain-balanced-ep12/model_torchscript.pt"
    initial_load_head: bool = False
    freeze_encoder: bool = False
    restore_best_by_val_loss: bool = True
    fold_sites: list[str] | None = None
    train_sampling: str = "random"  # random | site_balanced
    soft_teacher_npz: str = ""  # path to teacher leave_site_predictions.npz (aligned by filename+start)
    soft_teacher_weight: float = 0.0  # 0=hard only; blend train target = (1-w)*hard + w*teacher_soft


def load_config(path: Path | None) -> NativeLoSiteConfig:
    cfg = NativeLoSiteConfig()
    if path is None:
        return cfg
    data = json.loads(path.read_text())
    values = asdict(cfg)
    for key, value in data.items():
        if key in values:
            values[key] = value
    return NativeLoSiteConfig(**values)


def parse_time_seconds(text: str) -> float:
    parts = [float(x) for x in str(text).split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def decode_window(path: Path, start_sec: float, cfg: NativeLoSiteConfig) -> np.ndarray:
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


def choose_labels(data_root: Path, cfg: NativeLoSiteConfig, soundscape_df: pd.DataFrame) -> tuple[list[str], dict[str, Any], set[str], set[str]]:
    taxonomy = pd.read_csv(data_root / "taxonomy.csv", dtype={"primary_label": str})
    train = pd.read_csv(data_root / "train.csv", dtype={"primary_label": str})
    all_labels = taxonomy["primary_label"].astype(str).tolist()
    train_labels = set(train["primary_label"].astype(str))
    no_train = {x for x in all_labels if x not in train_labels}
    nonaves = set(taxonomy.loc[taxonomy["class_name"].astype(str) != "Aves", "primary_label"].astype(str))
    positive: set[str] = set()
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
    label_info = {
        "n_taxonomy_labels": len(all_labels),
        "n_train_primary_labels": len(train_labels),
        "n_no_train_labels": len(no_train),
        "n_nonaves_labels": len(nonaves),
        "n_soundscape_positive_labels": len(positive),
        "class_scope": cfg.class_scope,
        "n_training_labels": len(labels),
        "no_train_labels_in_scope": sorted(no_train & set(labels)),
        "nonaves_labels_in_scope": sorted(nonaves & set(labels)),
        "soundscape_positive_labels_in_scope": sorted(positive & set(labels)),
    }
    return labels, label_info, no_train, nonaves


def data_profile(rows: list[dict[str, Any]], labels: list[str], y: np.ndarray, label_info: dict[str, Any]) -> dict[str, Any]:
    site_series = pd.Series([r["site"] for r in rows], dtype=str)
    file_series = pd.Series([r["filename"] for r in rows], dtype=str)
    label_counts = {labels[i]: int(y[:, i].sum()) for i in range(len(labels)) if int(y[:, i].sum()) > 0}
    site_label_density: dict[str, Any] = {}
    for site in sorted(site_series.unique()):
        idx = np.where(site_series.values == site)[0]
        yy = y[idx]
        site_label_density[site] = {
            "windows": int(len(idx)),
            "files": int(file_series.iloc[idx].nunique()),
            "positive_cells": int(yy.sum()),
            "density": float(yy.mean()) if yy.size else 0.0,
            "valid_auc_classes": int(sum((yy[:, j].min() != yy[:, j].max()) for j in range(yy.shape[1]))),
        }
    return {
        **label_info,
        "n_windows": int(len(rows)),
        "n_files": int(file_series.nunique()),
        "n_sites": int(site_series.nunique()),
        "site_counts": {k: int(v) for k, v in site_series.value_counts().sort_index().items()},
        "file_window_count_top10": {k: int(v) for k, v in file_series.value_counts().head(10).items()},
        "target_positive_cells": int(y.sum()),
        "target_density": float(y.mean()) if y.size else 0.0,
        "positive_label_count_top20": dict(sorted(label_counts.items(), key=lambda kv: kv[1], reverse=True)[:20]),
        "positive_label_count_bottom20": dict(sorted(label_counts.items(), key=lambda kv: kv[1])[:20]),
        "site_label_density": site_label_density,
    }


def make_dataset(cfg: NativeLoSiteConfig) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], list[str], dict[str, Any], set[str], set[str]]:
    data_root = Path(cfg.data_root)
    soundscape_df = pd.read_csv(data_root / "train_soundscapes_labels.csv", dtype=str)
    labels, label_info, no_train, nonaves = choose_labels(data_root, cfg, soundscape_df)
    label_to_idx = {label: i for i, label in enumerate(labels)}

    rows: list[dict[str, Any]] = []
    for src_idx, r in enumerate(soundscape_df.itertuples(index=False)):
        filename = str(getattr(r, "filename"))
        path = data_root / "train_soundscapes" / filename
        if not path.exists():
            continue
        present = [x.strip() for x in str(getattr(r, "primary_label")).split(";") if x.strip()]
        rows.append({
            "src_idx": int(src_idx),
            "filename": filename,
            "path": str(path),
            "start": str(getattr(r, "start")),
            "start_sec": parse_time_seconds(str(getattr(r, "start"))),
            "end": str(getattr(r, "end")),
            "labels_raw": present,
            "target_indices": [label_to_idx[x] for x in present if x in label_to_idx],
            "site": site_from_filename(filename),
        })
    if cfg.max_windows and cfg.max_windows > 0:
        rng = np.random.default_rng(cfg.seed)
        idx = rng.permutation(len(rows))[: cfg.max_windows]
        rows = [rows[int(i)] for i in idx]

    y = torch.zeros((len(rows), len(labels)), dtype=torch.float32)
    for i, row in enumerate(rows):
        if row["target_indices"]:
            y[i, torch.tensor(row["target_indices"], dtype=torch.long)] = 1.0

    y_soft = y.clone()
    soft_info: dict[str, Any] = {"enabled": False}
    if cfg.soft_teacher_npz and cfg.soft_teacher_weight > 0.0:
        tz = np.load(cfg.soft_teacher_npz, allow_pickle=True)
        t_files = [str(v) for v in tz["files"]]
        t_starts = [str(v) for v in tz["starts"]]
        t_labels = [str(v) for v in tz["labels"]]
        t_pred = tz["pred"].astype(np.float32)
        tkey = {(f, s): k for k, (f, s) in enumerate(zip(t_files, t_starts))}
        tcol = {lb: j for j, lb in enumerate(t_labels)}
        col_map = np.array([tcol.get(lb, -1) for lb in labels], dtype=np.int64)
        w = float(cfg.soft_teacher_weight)
        matched = 0
        for i, row in enumerate(rows):
            k = tkey.get((str(row["filename"]), str(row["start"])))
            if k is None:
                continue
            matched += 1
            tp = t_pred[k]
            soft_row = np.zeros(len(labels), dtype=np.float32)
            valid = col_map >= 0
            soft_row[valid] = tp[col_map[valid]]
            blended = (1.0 - w) * y[i].numpy() + w * soft_row
            y_soft[i] = torch.from_numpy(blended.astype(np.float32))
        soft_info = {
            "enabled": True,
            "npz": cfg.soft_teacher_npz,
            "weight": w,
            "matched_rows": int(matched),
            "total_rows": int(len(rows)),
            "mapped_cols": int(np.sum(col_map >= 0)),
        }

    mel_fb = make_mel_filter(cfg.sample_rate, cfg.n_fft, cfg.n_mels)
    x_items = []
    decode_t0 = time.time()
    for item in rows:
        wav = torch.from_numpy(decode_window(Path(item["path"]), float(item["start_sec"]), cfg))
        x_items.append(waveform_to_logmel(wav, cfg, mel_fb).to(torch.float32))
    x = torch.stack(x_items, dim=0)
    profile = data_profile(rows, labels, y.numpy(), label_info)
    profile["input_shape"] = list(x.shape)
    profile["decode_feature_seconds"] = float(time.time() - decode_t0)
    profile["soft_teacher"] = soft_info
    return x, y, y_soft, rows, labels, profile, no_train, nonaves


def build_model(n_labels: int, cfg: NativeLoSiteConfig) -> torch.nn.Module:
    if cfg.backbone == "tiny_cnn":
        return TinySEDSmoke(n_labels)
    return TimmSEDB0(n_labels, cfg.backbone, cfg.pretrained)


def make_pos_weight(y_train: torch.Tensor, cfg: NativeLoSiteConfig, device: torch.device) -> torch.Tensor | None:
    if cfg.class_balancing in ("none", ""):
        return None
    if cfg.class_balancing == "pos_weight_sqrt":
        value = float(np.sqrt(max(y_train.shape[1] - 1, 1)))
        return torch.full((y_train.shape[1],), min(value, cfg.pos_weight_clip), device=device)
    if cfg.class_balancing == "observed_sqrt":
        pos = y_train.sum(dim=0).clamp_min(1.0)
        neg = (len(y_train) - pos).clamp_min(1.0)
        w = torch.sqrt(neg / pos).clamp(max=cfg.pos_weight_clip)
        return w.to(device)
    raise ValueError(f"Unsupported class_balancing={cfg.class_balancing}")


def batch_iter(indices: torch.Tensor, batch_size: int, shuffle: bool, seed: int):
    if shuffle:
        gen = torch.Generator().manual_seed(seed)
        indices = indices[torch.randperm(len(indices), generator=gen)]
    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]


def make_epoch_order(train_idx: torch.Tensor, rows: list[dict[str, Any]], cfg: NativeLoSiteConfig) -> torch.Tensor:
    if cfg.train_sampling in ("", "random"):
        return train_idx[torch.randperm(len(train_idx))]
    if cfg.train_sampling == "site_balanced":
        sites = [str(rows[int(i)]["site"]) for i in train_idx]
        counts = pd.Series(sites, dtype=str).value_counts().to_dict()
        weights = torch.tensor([1.0 / float(counts[s]) for s in sites], dtype=torch.float32)
        sampled_pos = torch.multinomial(weights, num_samples=len(train_idx), replacement=True)
        return train_idx[sampled_pos]
    raise ValueError(f"Unsupported train_sampling={cfg.train_sampling!r}")


def train_one_fold(x: torch.Tensor, y: torch.Tensor, rows: list[dict[str, Any]], train_idx: torch.Tensor, val_idx: torch.Tensor, labels: list[str], cfg: NativeLoSiteConfig, fold_seed: int, y_target: torch.Tensor | None = None) -> tuple[torch.nn.Module, list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    if y_target is None:
        y_target = y
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(labels), cfg)
    init_info = load_initial_checkpoint(model, cfg)
    trainable_params = list(model.parameters())
    if cfg.freeze_encoder and hasattr(model, "encoder"):
        for param in model.encoder.parameters():
            param.requires_grad = False
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        init_info = {**init_info, "freeze_encoder": True, "trainable_parameters": int(sum(p.numel() for p in trainable_params))}
    else:
        init_info = {**init_info, "freeze_encoder": False, "trainable_parameters": int(sum(p.numel() for p in trainable_params))}
    if not trainable_params:
        raise ValueError("No trainable parameters remain after freeze settings")
    model.to(device)
    opt = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    pos_weight = make_pos_weight(y[train_idx], cfg, device)
    best_val_loss = float("inf")
    best_state = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        order = make_epoch_order(train_idx, rows, cfg)
        for start in range(0, len(order), cfg.batch_size):
            idx = order[start:start + cfg.batch_size]
            xb = x[idx].to(device)
            yb = y_target[idx].to(device)
            xb, yb = maybe_mixup(xb, yb, cfg.mixup_alpha)
            logits, _ = model(xb)
            loss = compute_loss(logits, yb, cfg, pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
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
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if cfg.restore_best_by_val_loss and best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    pred = predict_probs(model, x, val_idx, cfg.batch_size, device)
    return model, history, pred, {"initial_checkpoint": init_info, "best_val_loss": float(best_val_loss)}


def predict_probs(model: torch.nn.Module, x: torch.Tensor, indices: torch.Tensor, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start + batch_size]
            logits, _ = model(x[idx].to(device))
            out.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 0), dtype=np.float32)


def macro_auc(y_true: np.ndarray, pred: np.ndarray, labels: list[str], subset: set[str] | None = None) -> dict[str, Any]:
    if roc_auc_score is None:
        return {"macro_auc": None, "valid_classes": 0, "error": "sklearn unavailable"}
    aucs: list[float] = []
    by_label: dict[str, float] = {}
    for j, label in enumerate(labels):
        if subset is not None and label not in subset:
            continue
        col = y_true[:, j]
        if col.min() == col.max():
            continue
        try:
            val = float(roc_auc_score(col, pred[:, j]))
        except Exception:
            continue
        aucs.append(val)
        by_label[label] = val
    return {
        "macro_auc": float(np.mean(aucs)) if aucs else None,
        "valid_classes": int(len(aucs)),
        "per_label_auc_top10": dict(sorted(by_label.items(), key=lambda kv: kv[1], reverse=True)[:10]),
        "per_label_auc_bottom10": dict(sorted(by_label.items(), key=lambda kv: kv[1])[:10]),
    }


def file_mil_auc(rows: list[dict[str, Any]], val_idx: torch.Tensor, y_true: np.ndarray, pred: np.ndarray, labels: list[str], subset: set[str] | None = None) -> dict[str, Any]:
    file_names = [rows[int(i)]["filename"] for i in val_idx]
    files = sorted(set(file_names))
    if not files:
        return {"macro_auc": None, "valid_classes": 0, "n_files": 0}
    y_file = []
    p_file = []
    for f in files:
        ii = [k for k, name in enumerate(file_names) if name == f]
        y_file.append(y_true[ii].max(axis=0))
        p_file.append(pred[ii].max(axis=0))
    return {**macro_auc(np.vstack(y_file), np.vstack(p_file), labels, subset), "n_files": int(len(files))}


def export_final_model(model: torch.nn.Module, x: torch.Tensor, output_dir: Path, cfg: NativeLoSiteConfig) -> dict[str, Any]:
    exports: dict[str, Any] = {}
    sample = x[: min(2, len(x))].to("cpu")
    model_cpu = model.to("cpu").eval()
    traced = torch.jit.trace(model_cpu, sample, strict=False)
    ts_path = output_dir / "model_torchscript.pt"
    traced.save(str(ts_path))
    exports["torchscript_path"] = str(ts_path)
    exports["torchscript_size_mb"] = round(ts_path.stat().st_size / 1e6, 3)
    if cfg.export_onnx:
        onnx_path = output_dir / "model.onnx"
        try:
            import onnx  # noqa: F401
            torch.onnx.export(
                model_cpu,
                sample,
                str(onnx_path),
                input_names=["logmel"],
                output_names=["clip_logits", "frame_logits"],
                dynamic_axes={"logmel": {0: "batch", 2: "frames"}, "clip_logits": {0: "batch"}, "frame_logits": {0: "batch", 1: "frames"}},
                opset_version=18,
            )
            import onnx as _onnx
            _onnx.checker.check_model(str(onnx_path))
            exports["onnx_path"] = str(onnx_path)
            exports["onnx_size_mb"] = round(onnx_path.stat().st_size / 1e6, 3)
            exports["onnx_status"] = "exported_checked"
        except Exception as exc:  # pragma: no cover
            exports["onnx_status"] = f"failed: {type(exc).__name__}: {exc}"
    with torch.no_grad():
        logits, frame_logits = model_cpu(sample)
    exports["torchscript_smoke"] = {
        "sample_shape": list(sample.shape),
        "clip_logits_shape": list(logits.shape),
        "frame_logits_shape": list(frame_logits.shape),
        "finite": bool(torch.isfinite(logits).all().item() and torch.isfinite(frame_logits).all().item()),
    }
    return exports


def train_final_model(x: torch.Tensor, y: torch.Tensor, rows: list[dict[str, Any]], labels: list[str], cfg: NativeLoSiteConfig, y_target: torch.Tensor | None = None) -> tuple[torch.nn.Module, list[dict[str, Any]], dict[str, Any]]:
    old_epochs = cfg.epochs
    cfg.epochs = cfg.final_train_epochs
    idx = torch.arange(len(x), dtype=torch.long)
    # Use a tiny pseudo-val slice only for best-state bookkeeping; final model is a packaging smoke, not a metric source.
    train_idx = idx
    val_idx = idx[: min(len(idx), max(8, cfg.batch_size))]
    model, history, _pred, info = train_one_fold(x, y, rows, train_idx, val_idx, labels, cfg, cfg.seed + 999, y_target=y_target)
    cfg.epochs = old_epochs
    return model, history, info


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
    x, y, y_soft, rows, labels, data_info, no_train, nonaves = make_dataset(cfg)
    sites = sorted({r["site"] for r in rows})
    if cfg.fold_sites:
        sites = [s for s in cfg.fold_sites if s in sites]

    folds: list[dict[str, Any]] = []
    all_idx: list[int] = []
    all_pred: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    for fold_num, site in enumerate(sites, start=1):
        val_idx = torch.tensor([i for i, r in enumerate(rows) if r["site"] == site], dtype=torch.long)
        if len(val_idx) < cfg.min_val_windows:
            folds.append({"site": site, "status": "skipped", "reason": "too_few_val_windows", "n_val": int(len(val_idx))})
            continue
        train_idx = torch.tensor([i for i, r in enumerate(rows) if r["site"] != site], dtype=torch.long)
        valid_classes = int(sum((y[val_idx, j].min() != y[val_idx, j].max()) for j in range(y.shape[1])))
        if valid_classes < cfg.min_valid_classes:
            folds.append({"site": site, "status": "skipped", "reason": "too_few_valid_classes", "n_val": int(len(val_idx)), "valid_classes": valid_classes})
            continue
        fold_t0 = time.time()
        _model, history, pred, train_info = train_one_fold(x, y, rows, train_idx, val_idx, labels, cfg, cfg.seed + fold_num, y_target=y_soft)
        yy = y[val_idx].numpy()
        fold = {
            "site": site,
            "status": "complete",
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_val_files": int(len({rows[int(i)]["filename"] for i in val_idx})),
            "valid_classes": valid_classes,
            "history": history,
            "best_val_loss": train_info["best_val_loss"],
            "initial_checkpoint": train_info["initial_checkpoint"],
            "row_auc_all_scope": macro_auc(yy, pred, labels),
            "row_auc_no_train": macro_auc(yy, pred, labels, no_train),
            "row_auc_nonaves": macro_auc(yy, pred, labels, nonaves),
            "file_mil_auc_all_scope": file_mil_auc(rows, val_idx, yy, pred, labels),
            "file_mil_auc_no_train": file_mil_auc(rows, val_idx, yy, pred, labels, no_train),
            "file_mil_auc_nonaves": file_mil_auc(rows, val_idx, yy, pred, labels, nonaves),
            "prediction_stats": {
                "min": float(pred.min()) if pred.size else None,
                "max": float(pred.max()) if pred.size else None,
                "mean": float(pred.mean()) if pred.size else None,
                "std": float(pred.std()) if pred.size else None,
                "nonconstant_columns": int(sum(float(pred[:, j].std()) > 1e-8 for j in range(pred.shape[1]))),
            },
            "runtime_seconds": float(time.time() - fold_t0),
        }
        folds.append(fold)
        all_idx.extend([int(i) for i in val_idx])
        all_pred.append(pred.astype(np.float32))
        all_y.append(yy.astype(np.float32))
        (output_dir / "fold_metrics_live.json").write_text(json.dumps(folds, indent=2) + "\n")

    final_model, final_history, final_info = train_final_model(x, y, rows, labels, cfg, y_target=y_soft)
    exports = export_final_model(final_model, x, output_dir, cfg)

    complete = [f for f in folds if f.get("status") == "complete"]
    def mean_metric(path: tuple[str, str]) -> float | None:
        vals = []
        for f in complete:
            obj: Any = f
            for key in path:
                obj = obj.get(key, {}) if isinstance(obj, dict) else {}
            if isinstance(obj, (float, int)):
                vals.append(float(obj))
        return float(np.mean(vals)) if vals else None

    if all_pred:
        pred_all = np.concatenate(all_pred, axis=0)
        y_all = np.concatenate(all_y, axis=0)
        idx_all = np.array(all_idx, dtype=np.int64)
        order = np.argsort(idx_all)
        np.savez_compressed(
            output_dir / "leave_site_predictions.npz",
            row_indices=idx_all[order],
            files=np.array([rows[int(i)]["filename"] for i in idx_all[order]], dtype=str),
            starts=np.array([rows[int(i)]["start"] for i in idx_all[order]], dtype=str),
            sites=np.array([rows[int(i)]["site"] for i in idx_all[order]], dtype=str),
            labels=np.array(labels, dtype=str),
            y_true=y_all[order].astype(np.float32),
            pred=pred_all[order].astype(np.float32),
        )
        pred_stats = {
            "shape": list(pred_all.shape),
            "min": float(pred_all.min()),
            "max": float(pred_all.max()),
            "mean": float(pred_all.mean()),
            "std": float(pred_all.std()),
            "nonconstant_columns": int(sum(float(pred_all[:, j].std()) > 1e-8 for j in range(pred_all.shape[1]))),
        }
        pooled_row_auc = macro_auc(y_all, pred_all, labels)
        pooled_no_train_auc = macro_auc(y_all, pred_all, labels, no_train)
    else:
        pred_stats = {"shape": [0, len(labels)], "nonconstant_columns": 0}
        pooled_row_auc = {"macro_auc": None, "valid_classes": 0}
        pooled_no_train_auc = {"macro_auc": None, "valid_classes": 0}

    metrics = {
        "status": "complete",
        "experiment_id": cfg.experiment_id,
        "track": cfg.track,
        "runtime_seconds": float(time.time() - t0),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data": data_info,
        "folds": folds,
        "summary": {
            "completed_folds": int(len(complete)),
            "skipped_folds": int(len(folds) - len(complete)),
            "row_auc_mean": mean_metric(("row_auc_all_scope", "macro_auc")),
            "row_auc_no_train_mean": mean_metric(("row_auc_no_train", "macro_auc")),
            "row_auc_nonaves_mean": mean_metric(("row_auc_nonaves", "macro_auc")),
            "file_mil_auc_mean": mean_metric(("file_mil_auc_all_scope", "macro_auc")),
            "pooled_row_auc": pooled_row_auc,
            "pooled_no_train_auc": pooled_no_train_auc,
            "prediction_stats": pred_stats,
        },
        "final_train": {"history": final_history, **final_info},
        "exports": exports,
        "evidence_level": "comparison-grade model data point; not submission-grade",
        "decision_hint": "Use for search landscape and possible future sidecar only after 234-class wrapper plus v616 audit; not a leaderboard candidate by itself.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
