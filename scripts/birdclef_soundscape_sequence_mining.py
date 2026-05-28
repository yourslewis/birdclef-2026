#!/usr/bin/env python3
"""Sequence/file/site-aware train_soundscapes mining probe for BirdCLEF 2026.

This is a no-slot ClawTeam data-point trainer.  It treats the official
train_soundscapes labels as ordered files/sites rather than independent rows:

- reconstructs full 5s-window sequences per soundscape file;
- builds temporal context features from EfficientAT/PANNs-style embeddings;
- evaluates leave-one-site folds plus file-level MIL max pooling;
- compares row-only vs temporal/file-context heads under the same protocol.

Class scopes include the original broad non-Aves/no-train target and a
focused no-train-only target for the 28 classes that have soundscape labels but
no train-audio primary supervision.

The output is intentionally a measured landscape artifact, not a Kaggle
submission package.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
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
class SequenceMiningConfig:
    experiment_id: str = "soundscape-sequence-dymn10-context-losite-ep16-20260526"
    track: str = "train_soundscapes sequence/file/site mining data point"
    data_root: str = "/home/yourslewis/birdclef-2026/data"
    output_dir: str = "artifacts/soundscape_sequence_mining/soundscape-sequence-dymn10-context-losite-ep16-20260526"
    embedding_npz: str = "artifacts/efficientat_soundscape_embeddings/efficientat-dymn10-audioset-soundscape-nonaves-notrain-nocall-siteS08-ep12-20260526/efficientat_embeddings.npz"
    embedding_key: str = "embedding"
    class_scope: str = "nonaves_or_no_train"  # no_train_only | nonaves_or_no_train | soundscape_positive | all
    context_radius: int = 1
    include_prev_next: bool = True
    include_local_mean: bool = True
    include_local_max: bool = True
    include_file_mean: bool = True
    include_file_max: bool = False
    include_time_features: bool = True
    include_site_onehot: bool = False
    hidden_dim: int = 384
    dropout: float = 0.20
    epochs: int = 16
    batch_size: int = 128
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    seed: int = 42
    pos_weight: bool = True
    pos_weight_power: float = 0.5
    pos_weight_clip: float = 20.0
    file_mil_loss_weight: float = 0.0
    file_mil_pos_weight: bool = True
    site_balanced_sampling: bool = True
    min_val_rows: int = 40
    min_valid_classes: int = 4
    final_train_epochs: int = 16


def load_config(path: Path | None) -> SequenceMiningConfig:
    cfg = SequenceMiningConfig()
    if path is None:
        return cfg
    data = json.loads(path.read_text())
    values = asdict(cfg)
    for key, value in data.items():
        if key in values:
            values[key] = value
    return SequenceMiningConfig(**values)


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


def choose_labels(data_root: Path, cfg: SequenceMiningConfig, soundscape_df: pd.DataFrame) -> tuple[list[str], dict[str, Any], set[str], set[str]]:
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
    elif cfg.class_scope == "no_train_only":
        labels = [x for x in all_labels if x in no_train]
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


def make_rows_targets(cfg: SequenceMiningConfig) -> tuple[list[dict[str, Any]], list[str], torch.Tensor, dict[str, Any], set[str], set[str]]:
    data_root = Path(cfg.data_root)
    soundscape_df = pd.read_csv(data_root / "train_soundscapes_labels.csv", dtype=str)
    labels, label_info, no_train, nonaves = choose_labels(data_root, cfg, soundscape_df)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    rows: list[dict[str, Any]] = []
    for src_idx, r in enumerate(soundscape_df.itertuples(index=False)):
        filename = str(getattr(r, "filename"))
        start = parse_time_seconds(str(getattr(r, "start")))
        present = [x.strip() for x in str(getattr(r, "primary_label")).split(";") if x.strip()]
        rows.append({
            "src_idx": int(src_idx),
            "filename": filename,
            "start": str(getattr(r, "start")),
            "start_sec": float(start),
            "end": str(getattr(r, "end")),
            "labels_raw": present,
            "target_indices": [label_to_idx[x] for x in present if x in label_to_idx],
            "site": site_from_filename(filename),
        })
    y = torch.zeros((len(rows), len(labels)), dtype=torch.float32)
    for i, row in enumerate(rows):
        if row["target_indices"]:
            y[i, torch.tensor(row["target_indices"], dtype=torch.long)] = 1.0
    profile = data_profile(rows, labels, y.numpy(), label_info)
    return rows, labels, y, profile, no_train, nonaves


def data_profile(rows: list[dict[str, Any]], labels: list[str], y: np.ndarray, label_info: dict[str, Any]) -> dict[str, Any]:
    sites = pd.Series([r["site"] for r in rows], dtype=str)
    files = pd.Series([r["filename"] for r in rows], dtype=str)
    labels_per_row = y.sum(axis=1)
    per_label_counts = {labels[i]: int(v) for i, v in enumerate(y.sum(axis=0)) if v > 0}
    top_pairs: list[dict[str, Any]] = []
    label_indices = np.where(y.sum(axis=0) > 0)[0]
    for a, b in itertools.combinations(label_indices.tolist(), 2):
        c = int(np.logical_and(y[:, a] > 0, y[:, b] > 0).sum())
        if c > 0:
            top_pairs.append({"a": labels[a], "b": labels[b], "count": c})
    top_pairs = sorted(top_pairs, key=lambda d: d["count"], reverse=True)[:40]
    file_density = []
    for fname, idx in files.groupby(files).groups.items():
        arr = y[list(idx)]
        file_density.append({
            "filename": str(fname),
            "site": rows[int(next(iter(idx)))]["site"],
            "n_windows": int(len(idx)),
            "positive_cells": int(arr.sum()),
            "unique_labels": int((arr.sum(axis=0) > 0).sum()),
        })
    return {
        **label_info,
        "n_windows": len(rows),
        "n_files": int(files.nunique()),
        "n_sites": int(sites.nunique()),
        "site_counts": {k: int(v) for k, v in sites.value_counts().sort_index().items()},
        "file_counts_top20": sorted(file_density, key=lambda d: d["n_windows"], reverse=True)[:20],
        "labels_per_row": {
            "min": float(labels_per_row.min()) if len(labels_per_row) else 0.0,
            "median": float(np.median(labels_per_row)) if len(labels_per_row) else 0.0,
            "mean": float(labels_per_row.mean()) if len(labels_per_row) else 0.0,
            "max": float(labels_per_row.max()) if len(labels_per_row) else 0.0,
        },
        "target_positive_cells": int(y.sum()),
        "target_density": float(y.mean()) if y.size else 0.0,
        "no_scoped_label_rows": int((labels_per_row == 0).sum()),
        "positive_label_counts_top40": dict(sorted(per_label_counts.items(), key=lambda kv: kv[1], reverse=True)[:40]),
        "cooccurrence_pairs_top40": top_pairs,
    }


def load_embedding(cfg: SequenceMiningConfig, n_rows: int) -> np.ndarray:
    path = Path(cfg.embedding_npz)
    if not path.exists():
        raise FileNotFoundError(path)
    obj = np.load(path, allow_pickle=False)
    x = obj[cfg.embedding_key].astype(np.float32)
    if x.shape[0] != n_rows:
        raise ValueError(f"Embedding row mismatch: {x.shape[0]} vs {n_rows}")
    return x


def build_context_features(rows: list[dict[str, Any]], emb: np.ndarray, cfg: SequenceMiningConfig) -> tuple[np.ndarray, dict[str, Any]]:
    features: list[np.ndarray] = []
    dim = emb.shape[1]
    by_file: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_file.setdefault(row["filename"], []).append(i)
    for inds in by_file.values():
        inds.sort(key=lambda i: rows[i]["start_sec"])
    file_mean = {f: emb[idxs].mean(axis=0) for f, idxs in by_file.items()}
    file_max = {f: emb[idxs].max(axis=0) for f, idxs in by_file.items()}
    site_values = sorted({r["site"] for r in rows})
    site_to_idx = {s: i for i, s in enumerate(site_values)}

    pos_in_file: dict[int, tuple[list[int], int]] = {}
    for fname, idxs in by_file.items():
        for pos, idx in enumerate(idxs):
            pos_in_file[idx] = (idxs, pos)

    max_start_by_file = {fname: max(rows[i]["start_sec"] for i in idxs) for fname, idxs in by_file.items()}
    for i, row in enumerate(rows):
        idxs, pos = pos_in_file[i]
        parts = [emb[i]]
        if cfg.include_prev_next:
            prev_i = idxs[max(0, pos - 1)]
            next_i = idxs[min(len(idxs) - 1, pos + 1)]
            parts.extend([emb[prev_i], emb[next_i]])
        lo = max(0, pos - cfg.context_radius)
        hi = min(len(idxs), pos + cfg.context_radius + 1)
        local = emb[idxs[lo:hi]]
        if cfg.include_local_mean:
            parts.append(local.mean(axis=0))
        if cfg.include_local_max:
            parts.append(local.max(axis=0))
        if cfg.include_file_mean:
            parts.append(file_mean[row["filename"]])
        if cfg.include_file_max:
            parts.append(file_max[row["filename"]])
        if cfg.include_time_features:
            denom = max(max_start_by_file[row["filename"]], 1.0)
            frac = row["start_sec"] / denom
            parts.append(np.array([frac, math.sin(2 * math.pi * frac), math.cos(2 * math.pi * frac), len(idxs) / 12.0], dtype=np.float32))
        if cfg.include_site_onehot:
            v = np.zeros(len(site_values), dtype=np.float32)
            v[site_to_idx[row["site"]]] = 1.0
            parts.append(v)
        features.append(np.concatenate(parts).astype(np.float32, copy=False))
    x = np.stack(features, axis=0)
    info = {
        "row_embedding_dim": int(dim),
        "context_feature_dim": int(x.shape[1]),
        "n_files": len(by_file),
        "context_radius": cfg.context_radius,
        "parts": {
            "current": True,
            "prev_next": cfg.include_prev_next,
            "local_mean": cfg.include_local_mean,
            "local_max": cfg.include_local_max,
            "file_mean": cfg.include_file_mean,
            "file_max": cfg.include_file_max,
            "time_features": cfg.include_time_features,
            "site_onehot": cfg.include_site_onehot,
        },
    }
    return x, info


class ContextHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
            val = float(roc_auc_score(col, p[:, j]))
        except Exception:
            continue
        aucs.append(val)
        label_aucs[label] = val
    return {
        "macro_auc": float(np.mean(aucs)) if aucs else None,
        "valid_classes": int(len(aucs)),
        "label_auc_bottom10": dict(sorted(label_aucs.items(), key=lambda kv: kv[1])[:10]),
        "label_auc_top10": dict(sorted(label_aucs.items(), key=lambda kv: kv[1], reverse=True)[:10]),
    }


def file_mil_auc(rows: list[dict[str, Any]], y: np.ndarray, p: np.ndarray, labels: list[str], subset: set[str] | None = None) -> dict[str, Any]:
    files = sorted({r["filename"] for r in rows})
    if len(files) < 2:
        return {"macro_auc": None, "valid_classes": 0, "n_files": len(files)}
    y_file = []
    p_file = []
    for fname in files:
        idx = [i for i, r in enumerate(rows) if r["filename"] == fname]
        y_file.append(y[idx].max(axis=0))
        p_file.append(p[idx].max(axis=0))
    out = macro_auc(np.stack(y_file), np.stack(p_file), labels, subset=subset)
    out["n_files"] = len(files)
    return out


def make_site_balanced_order(rows: list[dict[str, Any]], train_idx: np.ndarray, rng: np.random.Generator, enabled: bool) -> np.ndarray:
    if not enabled:
        return rng.permutation(train_idx)
    by_site: dict[str, list[int]] = {}
    for idx in train_idx.tolist():
        by_site.setdefault(rows[int(idx)]["site"], []).append(int(idx))
    max_n = max(len(v) for v in by_site.values())
    sampled: list[int] = []
    for vals in by_site.values():
        sampled.extend(rng.choice(vals, size=max_n, replace=len(vals) < max_n).tolist())
    return rng.permutation(np.array(sampled, dtype=np.int64))


def pos_weight_tensor(y_train: torch.Tensor, cfg: SequenceMiningConfig) -> torch.Tensor | None:
    if not cfg.pos_weight:
        return None
    pos = y_train.sum(dim=0)
    neg = y_train.shape[0] - pos
    pw = torch.ones_like(pos)
    mask = pos > 0
    pw[mask] = torch.pow(neg[mask] / torch.clamp(pos[mask], min=1.0), cfg.pos_weight_power)
    pw = torch.clamp(pw, 1.0, cfg.pos_weight_clip)
    return pw


def train_one(name: str, x_np: np.ndarray, y: torch.Tensor, rows: list[dict[str, Any]], train_idx_np: np.ndarray, val_idx_np: np.ndarray, cfg: SequenceMiningConfig) -> tuple[ContextHead, dict[str, Any], np.ndarray]:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(x_np.astype(np.float32))
    train_idx = torch.from_numpy(train_idx_np.astype(np.int64))
    val_idx = torch.from_numpy(val_idx_np.astype(np.int64))
    model = ContextHead(x.shape[1], y.shape[1], cfg.hidden_dim, cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    pw = pos_weight_tensor(y[train_idx], cfg)
    if pw is not None:
        pw = pw.to(device)

    train_file_groups: list[torch.Tensor] = []
    train_file_targets: torch.Tensor | None = None
    file_pw: torch.Tensor | None = None
    if cfg.file_mil_loss_weight > 0:
        pos_by_global = {int(g): p for p, g in enumerate(train_idx_np.tolist())}
        by_file: dict[str, list[int]] = {}
        for g in train_idx_np.tolist():
            by_file.setdefault(rows[int(g)]["filename"], []).append(int(g))
        target_rows = []
        for globals_for_file in by_file.values():
            positions = [pos_by_global[g] for g in globals_for_file]
            train_file_groups.append(torch.tensor(positions, dtype=torch.long, device=device))
            target_rows.append(y[torch.tensor(globals_for_file, dtype=torch.long)].max(dim=0).values)
        train_file_targets = torch.stack(target_rows).to(device) if target_rows else None
        if cfg.file_mil_pos_weight and train_file_targets is not None:
            file_cfg = SequenceMiningConfig(**{**asdict(cfg), "pos_weight": True})
            file_pw = pos_weight_tensor(train_file_targets.cpu(), file_cfg)
            if file_pw is not None:
                file_pw = file_pw.to(device)

    history: list[dict[str, Any]] = []
    best = None
    best_val = float("inf")
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order_np = make_site_balanced_order(rows, train_idx_np, rng, cfg.site_balanced_sampling)
        train_losses: list[float] = []
        for start in range(0, len(order_np), cfg.batch_size):
            idx = torch.from_numpy(order_np[start:start + cfg.batch_size]).long()
            xb = x[idx].to(device)
            yb = y[idx].to(device)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
        mil_loss_value: float | None = None
        if cfg.file_mil_loss_weight > 0 and train_file_targets is not None and train_file_groups:
            opt.zero_grad(set_to_none=True)
            logits_all = model(x[train_idx].to(device))
            mil_logits = torch.stack([logits_all[group].max(dim=0).values for group in train_file_groups])
            mil_loss = F.binary_cross_entropy_with_logits(mil_logits, train_file_targets, pos_weight=file_pw)
            (cfg.file_mil_loss_weight * mil_loss).backward()
            opt.step()
            mil_loss_value = float(mil_loss.detach().cpu())
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for start in range(0, len(val_idx), cfg.batch_size):
                idx = val_idx[start:start + cfg.batch_size]
                logits = model(x[idx].to(device))
                loss = F.binary_cross_entropy_with_logits(logits, y[idx].to(device), pos_weight=pw)
                val_losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        history.append({"epoch": epoch, "train_loss": train_loss, "file_mil_loss": mil_loss_value, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best is not None:
        model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        pred = []
        for start in range(0, x.shape[0], cfg.batch_size):
            logits = model(x[start:start + cfg.batch_size].to(device))
            pred.append(torch.sigmoid(logits).cpu().numpy().astype(np.float32))
    info = {"name": name, "history": history, "best_val_loss": float(best_val), "train_seconds": float(time.time() - t0)}
    return model, info, np.concatenate(pred, axis=0)


def summarize_folds(folds: list[dict[str, Any]], key: str) -> dict[str, Any]:
    vals = [f[key]["macro_auc_all_scoped"].get("macro_auc") for f in folds]
    vals = [float(v) for v in vals if v is not None]
    file_vals = [f[key]["file_mil_macro_auc_all_scoped"].get("macro_auc") for f in folds]
    file_vals = [float(v) for v in file_vals if v is not None]
    return {
        "n_folds": len(folds),
        "row_macro_auc_mean": float(np.mean(vals)) if vals else None,
        "row_macro_auc_min": float(np.min(vals)) if vals else None,
        "row_macro_auc_max": float(np.max(vals)) if vals else None,
        "file_mil_macro_auc_mean": float(np.mean(file_vals)) if file_vals else None,
        "file_mil_macro_auc_min": float(np.min(file_vals)) if file_vals else None,
        "file_mil_macro_auc_max": float(np.max(file_vals)) if file_vals else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.input.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")

    rows, labels, y, profile, no_train, nonaves = make_rows_targets(cfg)
    emb = load_embedding(cfg, len(rows))
    x_context, context_info = build_context_features(rows, emb, cfg)
    x_row = emb.astype(np.float32, copy=False)
    y_np = y.numpy().astype(np.float32)

    sites = sorted(profile["site_counts"])
    folds: list[dict[str, Any]] = []
    all_val_predictions: dict[str, list[np.ndarray]] = {"row_only": [], "context": []}
    all_val_indices: list[np.ndarray] = []
    for site in sites:
        val_idx = np.array([i for i, r in enumerate(rows) if r["site"] == site], dtype=np.int64)
        train_idx = np.array([i for i, r in enumerate(rows) if r["site"] != site], dtype=np.int64)
        if len(val_idx) < cfg.min_val_rows:
            continue
        # Need enough label variation to make the fold meaningful.
        valid = int(((y_np[val_idx].min(axis=0) != y_np[val_idx].max(axis=0))).sum())
        if valid < cfg.min_valid_classes:
            continue
        print(json.dumps({"fold": site, "n_train": len(train_idx), "n_val": len(val_idx), "valid_classes": valid}), flush=True)
        row_model, row_train, row_pred = train_one(f"row_only_{site}", x_row, y, rows, train_idx, val_idx, cfg)
        ctx_model, ctx_train, ctx_pred = train_one(f"context_{site}", x_context, y, rows, train_idx, val_idx, cfg)
        val_rows = [rows[int(i)] for i in val_idx]
        fold = {
            "site": site,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_val_files": int(len({r["filename"] for r in val_rows})),
            "valid_classes_raw": valid,
            "row_only_training": row_train,
            "context_training": ctx_train,
            "row_only": {
                "macro_auc_all_scoped": macro_auc(y_np[val_idx], row_pred[val_idx], labels),
                "macro_auc_no_train": macro_auc(y_np[val_idx], row_pred[val_idx], labels, subset=no_train),
                "macro_auc_nonaves": macro_auc(y_np[val_idx], row_pred[val_idx], labels, subset=nonaves),
                "file_mil_macro_auc_all_scoped": file_mil_auc(val_rows, y_np[val_idx], row_pred[val_idx], labels),
            },
            "context": {
                "macro_auc_all_scoped": macro_auc(y_np[val_idx], ctx_pred[val_idx], labels),
                "macro_auc_no_train": macro_auc(y_np[val_idx], ctx_pred[val_idx], labels, subset=no_train),
                "macro_auc_nonaves": macro_auc(y_np[val_idx], ctx_pred[val_idx], labels, subset=nonaves),
                "file_mil_macro_auc_all_scoped": file_mil_auc(val_rows, y_np[val_idx], ctx_pred[val_idx], labels),
            },
        }
        rauc = fold["row_only"]["macro_auc_all_scoped"].get("macro_auc")
        cauc = fold["context"]["macro_auc_all_scoped"].get("macro_auc")
        fold["context_minus_row_auc"] = None if rauc is None or cauc is None else float(cauc - rauc)
        folds.append(fold)
        all_val_indices.append(val_idx)
        all_val_predictions["row_only"].append(row_pred[val_idx])
        all_val_predictions["context"].append(ctx_pred[val_idx])
        print(json.dumps({"fold": site, "row_auc": rauc, "context_auc": cauc, "delta": fold["context_minus_row_auc"]}), flush=True)

    summary_row = summarize_folds(folds, "row_only")
    summary_ctx = summarize_folds(folds, "context")
    if summary_row["row_macro_auc_mean"] is not None and summary_ctx["row_macro_auc_mean"] is not None:
        delta = float(summary_ctx["row_macro_auc_mean"] - summary_row["row_macro_auc_mean"])
    else:
        delta = None

    # Train one final context head on all rows for export/smoke only.
    full_idx = np.arange(len(rows), dtype=np.int64)
    final_model, final_train, final_pred = train_one("context_final_all_rows", x_context, y, rows, full_idx, full_idx, cfg)
    final_model_cpu = final_model.cpu().eval()
    traced = torch.jit.trace(final_model_cpu, torch.randn(2, x_context.shape[1]))
    traced.save(str(out_dir / "context_head_torchscript.pt"))
    torch.save({"model_state": final_model_cpu.state_dict(), "config": asdict(cfg), "labels": labels, "context_info": context_info}, out_dir / "context_head.pt")

    metrics = {
        "experiment_id": cfg.experiment_id,
        "track": cfg.track,
        "config": asdict(cfg),
        "data_profile": profile,
        "context_features": context_info,
        "folds": folds,
        "summary": {
            "row_only": summary_row,
            "context": summary_ctx,
            "context_minus_row_macro_auc_mean": delta,
            "fold_deltas": [{"site": f["site"], "delta": f["context_minus_row_auc"]} for f in folds],
        },
        "final_all_rows_training": final_train,
        "final_prediction_stats": {
            "min": float(final_pred.min()),
            "max": float(final_pred.max()),
            "mean": float(final_pred.mean()),
            "std": float(final_pred.std()),
            "nonconstant_columns": int((final_pred.std(axis=0) > 1e-7).sum()),
        },
        "labels": labels,
        "rows_preview": rows[:8],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out_dir / "data_profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    (out_dir / "config.resolved.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    if all_val_indices:
        np.savez_compressed(
            out_dir / "leave_site_predictions.npz",
            val_idx=np.concatenate(all_val_indices),
            row_only_pred=np.concatenate(all_val_predictions["row_only"], axis=0),
            context_pred=np.concatenate(all_val_predictions["context"], axis=0),
            labels=np.array(labels),
        )
    print(json.dumps({
        "output_dir": str(out_dir),
        "n_folds": len(folds),
        "row_mean_auc": summary_row.get("row_macro_auc_mean"),
        "context_mean_auc": summary_ctx.get("row_macro_auc_mean"),
        "delta": delta,
        "context_dim": context_info["context_feature_dim"],
        "final_nonconstant_columns": metrics["final_prediction_stats"]["nonconstant_columns"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
