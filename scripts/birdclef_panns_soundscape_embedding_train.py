#!/usr/bin/env python3
"""Train a bounded PANNs/AudioSet embedding-head BirdCLEF data point.

This is a no-slot landscape probe for the ClawTeam hill-climb loop.  It uses
AudioSet-pretrained PANNs/Cnn14 embeddings on official train_soundscape windows,
then trains a small multilabel head for non-Aves/no-train classes plus a no-call
auxiliary target.  The artifact is intentionally diagnostic first; it is not a
competition submission generator.
"""
from __future__ import annotations

import argparse
import json
import re
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

from birdclef_sed_pilot_train import ffmpeg_binary


@dataclass
class PannsSoundscapeConfig:
    experiment_id: str = "panns-cnn14-audioset-soundscape-nonaves-notrain-nocall-siteS08-ep12-20260526"
    track: str = "PANNs/Cnn14 AudioSet embedding non-Aves/no-train/no-call data point"
    data_root: str = "/home/yourslewis/birdclef-2026/data"
    output_dir: str = "artifacts/panns_soundscape_embeddings/panns-cnn14-audioset-soundscape-nonaves-notrain-nocall-siteS08-ep12-20260526"
    sample_rate: int = 32000
    duration_sec: float = 5.0
    embedding_model: str = "panns_cnn14_audioset"
    checkpoint_path: str = ""  # panns_inference default if empty
    class_scope: str = "nonaves_or_no_train"
    val_site: str = "S08"
    min_val_windows: int = 50
    max_windows: int = 0
    seed: int = 42
    batch_size_extract: int = 16
    batch_size_train: int = 64
    epochs: int = 12
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    hidden_dim: int = 256
    dropout: float = 0.15
    no_call_aux_weight: float = 0.2
    cache_embeddings: bool = True
    restore_best_by_val_loss: bool = True


def load_config(path: Path | None) -> PannsSoundscapeConfig:
    cfg = PannsSoundscapeConfig()
    if path is None:
        return cfg
    data = json.loads(path.read_text())
    values = asdict(cfg)
    for key, value in data.items():
        if key in values:
            values[key] = value
    return PannsSoundscapeConfig(**values)


def parse_time_seconds(text: str) -> float:
    parts = [float(x) for x in str(text).split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def site_from_filename(name: str) -> str:
    m = re.search(r"_(S\d+)_", str(name))
    return m.group(1) if m else "UNKNOWN"


def decode_window(path: Path, start_sec: float, cfg: PannsSoundscapeConfig) -> np.ndarray:
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


def choose_labels(data_root: Path, cfg: PannsSoundscapeConfig, soundscape_df: pd.DataFrame) -> tuple[list[str], dict[str, Any]]:
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


def make_rows_and_targets(cfg: PannsSoundscapeConfig) -> tuple[list[dict[str, Any]], list[str], torch.Tensor, torch.Tensor, dict[str, Any]]:
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

    y = torch.zeros((len(rows), len(labels)), dtype=torch.float32)
    for i, item in enumerate(rows):
        if item["target_indices"]:
            y[i, torch.tensor(item["target_indices"], dtype=torch.long)] = 1.0
    no_call = (y.sum(dim=1) == 0).float().unsqueeze(1)
    info = {
        **label_info,
        "n_windows": len(rows),
        "target_positive_cells": int(y.sum().item()),
        "target_density": float(y.mean().item()) if y.numel() else 0.0,
        "no_call_aux_positive_rows": int(no_call.sum().item()),
        "no_call_aux_positive_rate": float(no_call.mean().item()) if len(rows) else 0.0,
        "site_counts": {k: int(v) for k, v in pd.Series([r["site"] for r in rows]).value_counts().sort_index().items()},
    }
    return rows, labels, y, no_call, info


def split_indices(rows: list[dict[str, Any]], cfg: PannsSoundscapeConfig) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
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


def extract_panns_embeddings(rows: list[dict[str, Any]], cfg: PannsSoundscapeConfig, output_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cache_path = output_dir / "panns_embeddings.npz"
    if cfg.cache_embeddings and cache_path.exists():
        obj = np.load(cache_path, allow_pickle=False)
        return obj["clipwise_output"], obj["embedding"], {"loaded_from_cache": True, "cache_path": str(cache_path)}

    t0 = time.time()
    from panns_inference import AudioTagging  # type: ignore

    checkpoint = cfg.checkpoint_path or None
    at = AudioTagging(checkpoint_path=checkpoint, device="cuda")
    clip_batches: list[np.ndarray] = []
    emb_batches: list[np.ndarray] = []
    n = len(rows)
    for start in range(0, n, cfg.batch_size_extract):
        batch_rows = rows[start : start + cfg.batch_size_extract]
        audio = np.stack([decode_window(Path(r["path"]), float(r["start_sec"]), cfg) for r in batch_rows], axis=0)
        clipwise, embedding = at.inference(audio)
        clip_batches.append(np.asarray(clipwise, dtype=np.float32))
        emb_batches.append(np.asarray(embedding, dtype=np.float32))
        print(f"[extract] {min(start + len(batch_rows), n)}/{n}", flush=True)
    clipwise_output = np.concatenate(clip_batches, axis=0)
    embedding = np.concatenate(emb_batches, axis=0)
    meta = {
        "loaded_from_cache": False,
        "cache_path": str(cache_path),
        "extract_seconds": float(time.time() - t0),
        "clipwise_shape": list(clipwise_output.shape),
        "embedding_shape": list(embedding.shape),
    }
    if cfg.cache_embeddings:
        np.savez_compressed(cache_path, clipwise_output=clipwise_output, embedding=embedding)
    return clipwise_output, embedding, meta


class EmbeddingHead(nn.Module):
    def __init__(self, in_dim: int, n_labels: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.label_head = nn.Linear(hidden_dim, n_labels)
        self.no_call_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.net(x)
        return self.label_head(z), self.no_call_head(z)


def macro_auc(y: np.ndarray, p: np.ndarray, labels: list[str], subset: set[str] | None = None) -> dict[str, Any]:
    if roc_auc_score is None:
        return {"macro_auc": None, "valid_classes": 0, "error": "sklearn unavailable"}
    aucs: list[float] = []
    label_aucs: dict[str, float] = {}
    for j, label in enumerate(labels):
        if subset is not None and label not in subset:
            continue
        col = y[:, j]
        if col.min() == col.max():
            continue
        try:
            auc = float(roc_auc_score(col, p[:, j]))
        except Exception:
            continue
        aucs.append(auc)
        label_aucs[label] = auc
    return {
        "macro_auc": float(np.mean(aucs)) if aucs else None,
        "valid_classes": len(aucs),
        "label_auc": dict(sorted(label_aucs.items(), key=lambda kv: kv[1])),
    }


def binary_auc(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    if roc_auc_score is None:
        return {"auc": None, "error": "sklearn unavailable"}
    yy = np.asarray(y).reshape(-1)
    pp = np.asarray(p).reshape(-1)
    if yy.min() == yy.max():
        return {"auc": None, "valid": False}
    return {"auc": float(roc_auc_score(yy, pp)), "valid": True}


def train_head(cfg: PannsSoundscapeConfig, embedding: np.ndarray, y: torch.Tensor, no_call: torch.Tensor, train_idx: torch.Tensor, val_idx: torch.Tensor, output_dir: Path) -> tuple[nn.Module, dict[str, Any], np.ndarray, np.ndarray]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    x = torch.from_numpy(embedding.astype(np.float32))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EmbeddingHead(x.shape[1], y.shape[1], cfg.hidden_dim, cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    history: list[dict[str, Any]] = []
    best_state = None
    best_val_loss = float("inf")
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = train_idx[torch.randperm(len(train_idx))]
        train_losses: list[float] = []
        for start in range(0, len(order), cfg.batch_size_train):
            idx = order[start : start + cfg.batch_size_train]
            xb = x[idx].to(device)
            yb = y[idx].to(device)
            nb = no_call[idx].to(device)
            logits, no_logits = model(xb)
            loss_main = F.binary_cross_entropy_with_logits(logits, yb)
            loss_aux = F.binary_cross_entropy_with_logits(no_logits, nb)
            loss = loss_main + cfg.no_call_aux_weight * loss_aux
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for start in range(0, len(val_idx), cfg.batch_size_train):
                idx = val_idx[start : start + cfg.batch_size_train]
                xb = x[idx].to(device)
                yb = y[idx].to(device)
                nb = no_call[idx].to(device)
                logits, no_logits = model(xb)
                loss_main = F.binary_cross_entropy_with_logits(logits, yb)
                loss_aux = F.binary_cross_entropy_with_logits(no_logits, nb)
                val_losses.append(float((loss_main + cfg.no_call_aux_weight * loss_aux).detach().cpu()))
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(json.dumps(history[-1]), flush=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if cfg.restore_best_by_val_loss and best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, no_logits = model(x.to(device))
        probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        no_call_probs = torch.sigmoid(no_logits).cpu().numpy().astype(np.float32)
    torch.save({"model_state": model.state_dict(), "config": asdict(cfg)}, output_dir / "embedding_head.pt")
    ts = torch.jit.trace(model.cpu().eval(), torch.randn(2, x.shape[1]))
    ts.save(str(output_dir / "embedding_head_torchscript.pt"))
    model.to(device)
    return model, {"history": history, "train_seconds": float(time.time() - t0), "best_val_loss": float(best_val_loss)}, probs, no_call_probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.input.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")

    rows, labels, y, no_call, data_info = make_rows_and_targets(cfg)
    train_idx, val_idx, split_info = split_indices(rows, cfg)
    clipwise_output, embedding, extract_info = extract_panns_embeddings(rows, cfg, output_dir)
    model, train_info, probs, no_call_probs = train_head(cfg, embedding, y, no_call, train_idx, val_idx, output_dir)

    data_root = Path(cfg.data_root)
    taxonomy = pd.read_csv(data_root / "taxonomy.csv", dtype={"primary_label": str})
    train = pd.read_csv(data_root / "train.csv", dtype={"primary_label": str})
    train_labels = set(train["primary_label"].astype(str))
    no_train = {x for x in taxonomy["primary_label"].astype(str) if x not in train_labels}
    nonaves = set(taxonomy.loc[taxonomy["class_name"].astype(str) != "Aves", "primary_label"].astype(str))

    val_np = val_idx.numpy()
    metrics = {
        "experiment_id": cfg.experiment_id,
        "track": cfg.track,
        "config": asdict(cfg),
        "data": data_info,
        "split": split_info,
        "embedding": extract_info,
        "training": train_info,
        "macro_auc_all_scoped": macro_auc(y[val_idx].numpy(), probs[val_np], labels),
        "macro_auc_no_train": macro_auc(y[val_idx].numpy(), probs[val_np], labels, subset=no_train),
        "macro_auc_nonaves": macro_auc(y[val_idx].numpy(), probs[val_np], labels, subset=nonaves),
        "no_call_aux_auc": binary_auc(no_call[val_idx].numpy(), no_call_probs[val_np]),
        "panns_clipwise_mean": float(np.mean(clipwise_output)),
        "panns_clipwise_std": float(np.std(clipwise_output)),
        "prediction_stats": {
            "label_min": float(np.min(probs)),
            "label_max": float(np.max(probs)),
            "label_mean": float(np.mean(probs)),
            "no_call_min": float(np.min(no_call_probs)),
            "no_call_max": float(np.max(no_call_probs)),
            "no_call_mean": float(np.mean(no_call_probs)),
        },
        "labels": labels,
        "rows_preview": rows[:5],
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(
        output_dir / "holdout_predictions.npz",
        val_idx=val_np,
        y_true=y[val_idx].numpy().astype(np.float32),
        pred=probs[val_np].astype(np.float32),
        no_call_true=no_call[val_idx].numpy().astype(np.float32),
        no_call_pred=no_call_probs[val_np].astype(np.float32),
        labels=np.array(labels),
        filenames=np.array([rows[i]["filename"] for i in val_np]),
        sites=np.array([rows[i]["site"] for i in val_np]),
    )
    (output_dir / "config.resolved.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    print(json.dumps({
        "output_dir": str(output_dir),
        "macro_auc": metrics["macro_auc_all_scoped"]["macro_auc"],
        "valid_classes": metrics["macro_auc_all_scoped"]["valid_classes"],
        "no_call_auc": metrics["no_call_aux_auc"].get("auc"),
        "best_val_loss": train_info["best_val_loss"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
