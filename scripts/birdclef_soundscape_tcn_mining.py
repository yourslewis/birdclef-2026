#!/usr/bin/env python3
"""Compact per-file TCN probe for BirdCLEF train_soundscapes.

This no-slot data-point trainer treats train_soundscapes as ordered files/sites.
It consumes cached acoustic embeddings (default EfficientAT DyMN10), trains a small
per-file temporal convolutional model, evaluates leave-site folds, and compares
against the previous context-MLP sequence-mining artifact when provided.

It intentionally emits landscape evidence, not a Kaggle submission.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from birdclef_soundscape_sequence_mining import (
    SequenceMiningConfig,
    file_mil_auc,
    load_embedding,
    macro_auc,
    make_rows_targets,
)


@dataclass
class TcnMiningConfig:
    experiment_id: str = "soundscape-tcn-dymn10-losite-ep20-20260526"
    track: str = "train_soundscapes compact per-file TCN data point"
    data_root: str = "/home/yourslewis/birdclef-2026/data"
    output_dir: str = "artifacts/soundscape_sequence_mining/soundscape-tcn-dymn10-losite-ep20-20260526"
    embedding_npz: str = "artifacts/efficientat_soundscape_embeddings/efficientat-dymn10-audioset-soundscape-nonaves-notrain-nocall-siteS08-ep12-20260526/efficientat_embeddings.npz"
    embedding_key: str = "embedding"
    class_scope: str = "nonaves_or_no_train"
    hidden_dim: int = 256
    n_layers: int = 3
    kernel_size: int = 3
    dropout: float = 0.30
    input_dropout: float = 0.05
    epochs: int = 20
    batch_files: int = 8
    learning_rate: float = 6e-4
    weight_decay: float = 2e-4
    seed: int = 42
    pos_weight: bool = True
    pos_weight_power: float = 0.5
    pos_weight_clip: float = 20.0
    site_balanced_file_sampling: bool = True
    min_val_rows: int = 40
    min_valid_classes: int = 4
    include_time_features: bool = True
    final_train_epochs: int = 20
    reference_metrics_json: str = "artifacts/soundscape_sequence_mining/soundscape-sequence-dymn10-context-losite-ep16-20260526/metrics.json"


def load_config(path: Path | None) -> TcnMiningConfig:
    cfg = TcnMiningConfig()
    if path is None:
        return cfg
    data = json.loads(path.read_text())
    values = asdict(cfg)
    for key, value in data.items():
        if key in values:
            values[key] = value
    return TcnMiningConfig(**values)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_base_sequence_cfg(cfg: TcnMiningConfig) -> SequenceMiningConfig:
    return SequenceMiningConfig(
        experiment_id=cfg.experiment_id,
        track=cfg.track,
        data_root=cfg.data_root,
        output_dir=cfg.output_dir,
        embedding_npz=cfg.embedding_npz,
        embedding_key=cfg.embedding_key,
        class_scope=cfg.class_scope,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        epochs=cfg.epochs,
        batch_size=128,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        seed=cfg.seed,
        pos_weight=cfg.pos_weight,
        pos_weight_power=cfg.pos_weight_power,
        pos_weight_clip=cfg.pos_weight_clip,
        site_balanced_sampling=True,
        min_val_rows=cfg.min_val_rows,
        min_valid_classes=cfg.min_valid_classes,
        final_train_epochs=cfg.final_train_epochs,
    )


def add_time_features(rows: list[dict[str, Any]], emb: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return emb.astype(np.float32, copy=False)
    by_file: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_file.setdefault(str(row["filename"]), []).append(i)
    max_start_by_file = {fname: max(float(rows[i]["start_sec"]) for i in idxs) for fname, idxs in by_file.items()}
    feats = []
    for row in rows:
        denom = max(float(max_start_by_file[str(row["filename"])]), 1.0)
        frac = float(row["start_sec"]) / denom
        feats.append([frac, math.sin(2 * math.pi * frac), math.cos(2 * math.pi * frac), len(by_file[str(row["filename"])]) / 12.0])
    return np.concatenate([emb.astype(np.float32, copy=False), np.asarray(feats, dtype=np.float32)], axis=1)


def make_file_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_file: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_file.setdefault(str(row["filename"]), []).append(i)
    groups = []
    for fname, idxs in by_file.items():
        idxs = sorted(idxs, key=lambda i: float(rows[i]["start_sec"]))
        groups.append({"filename": fname, "site": rows[idxs[0]]["site"], "idx": np.asarray(idxs, dtype=np.int64)})
    return sorted(groups, key=lambda g: (str(g["site"]), str(g["filename"])))


class ResidualTcnBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.norm = nn.LayerNorm(dim)
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,T,C
        residual = x
        z = self.norm(x).transpose(1, 2)
        z = self.conv1(z).transpose(1, 2)
        z = F.silu(z)
        z = self.dropout(z)
        z = self.conv2(z.transpose(1, 2)).transpose(1, 2)
        z = self.dropout(z)
        return residual + z


class FileTcn(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_layers: int, kernel_size: int, dropout: float, input_dropout: float):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_dim)
        self.in_drop = nn.Dropout(input_dropout)
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualTcnBlock(hidden_dim, kernel_size=kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        z = self.proj(self.in_drop(self.in_norm(x)))
        for block in self.blocks:
            z = block(z)
        z = self.out_norm(z)
        return self.head(z)


def make_pos_weight(y_train: torch.Tensor, cfg: TcnMiningConfig) -> torch.Tensor | None:
    if not cfg.pos_weight:
        return None
    pos = y_train.sum(dim=0)
    neg = y_train.shape[0] - pos
    pw = torch.ones_like(pos)
    mask = pos > 0
    pw[mask] = torch.pow(neg[mask] / torch.clamp(pos[mask], min=1.0), cfg.pos_weight_power)
    return torch.clamp(pw, 1.0, cfg.pos_weight_clip)


def file_order(groups: list[dict[str, Any]], train_files: list[int], cfg: TcnMiningConfig, rng: np.random.Generator) -> list[int]:
    if not cfg.site_balanced_file_sampling:
        return rng.permutation(np.asarray(train_files, dtype=np.int64)).tolist()
    by_site: dict[str, list[int]] = {}
    for gi in train_files:
        by_site.setdefault(str(groups[gi]["site"]), []).append(int(gi))
    max_n = max(len(v) for v in by_site.values())
    out: list[int] = []
    for vals in by_site.values():
        out.extend(rng.choice(vals, size=max_n, replace=len(vals) < max_n).tolist())
    return rng.permutation(np.asarray(out, dtype=np.int64)).tolist()


def batch_from_groups(groups: list[dict[str, Any]], batch_gi: list[int], x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    idxs = [groups[gi]["idx"] for gi in batch_gi]
    max_len = max(len(idx) for idx in idxs)
    b = len(idxs)
    xb = torch.zeros((b, max_len, x.shape[1]), dtype=x.dtype)
    yb = torch.zeros((b, max_len, y.shape[1]), dtype=y.dtype)
    mask = torch.zeros((b, max_len), dtype=torch.bool)
    flat_indices = []
    for bi, idx_np in enumerate(idxs):
        idx = torch.from_numpy(idx_np).long()
        t = len(idx_np)
        xb[bi, :t] = x[idx]
        yb[bi, :t] = y[idx]
        mask[bi, :t] = True
        flat_indices.extend(idx_np.tolist())
    return xb, yb, mask, np.asarray(flat_indices, dtype=np.int64)


def masked_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, pos_weight: torch.Tensor | None) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
    return loss[mask].mean()


def predict(model: FileTcn, groups: list[dict[str, Any]], group_indices: list[int], x: torch.Tensor, y: torch.Tensor, cfg: TcnMiningConfig, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    pred = []
    idx_out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(group_indices), cfg.batch_files):
            batch_gi = group_indices[start:start + cfg.batch_files]
            xb, _, mask, flat_idx = batch_from_groups(groups, batch_gi, x, y)
            logits = model(xb.to(device), mask.to(device)).cpu()
            probs = torch.sigmoid(logits)[mask].numpy().astype(np.float32)
            pred.append(probs)
            idx_out.append(flat_idx)
    return np.concatenate(idx_out), np.concatenate(pred, axis=0)


def train_fold(name: str, groups: list[dict[str, Any]], train_files: list[int], val_files: list[int], x_np: np.ndarray, y: torch.Tensor, cfg: TcnMiningConfig, epochs: int | None = None) -> tuple[FileTcn, dict[str, Any], np.ndarray, np.ndarray]:
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(x_np.astype(np.float32, copy=False))
    train_row_idx = np.concatenate([groups[gi]["idx"] for gi in train_files])
    pw = make_pos_weight(y[torch.from_numpy(train_row_idx).long()], cfg)
    if pw is not None:
        pw = pw.to(device)
    model = FileTcn(x.shape[1], y.shape[1], cfg.hidden_dim, cfg.n_layers, cfg.kernel_size, cfg.dropout, cfg.input_dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    hist = []
    best_state = None
    best_val = float("inf")
    t0 = time.time()
    n_epochs = int(epochs if epochs is not None else cfg.epochs)
    for epoch in range(1, n_epochs + 1):
        model.train()
        order = file_order(groups, train_files, cfg, rng)
        train_losses = []
        for start in range(0, len(order), cfg.batch_files):
            batch_gi = order[start:start + cfg.batch_files]
            xb, yb, mask, _ = batch_from_groups(groups, batch_gi, x, y)
            logits = model(xb.to(device), mask.to(device))
            loss = masked_bce(logits, yb.to(device), mask.to(device), pw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
        val_losses = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(val_files), cfg.batch_files):
                batch_gi = val_files[start:start + cfg.batch_files]
                xb, yb, mask, _ = batch_from_groups(groups, batch_gi, x, y)
                logits = model(xb.to(device), mask.to(device))
                val_losses.append(float(masked_bce(logits, yb.to(device), mask.to(device), pw).detach().cpu()))
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        hist.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    idx, pred = predict(model, groups, val_files, x, y, cfg, device)
    return model, {"name": name, "history": hist, "best_val_loss": best_val, "train_seconds": float(time.time() - t0)}, idx, pred


def load_reference(path: str) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return {}
    m = json.loads(Path(path).read_text())
    out = {"path": path, "folds": {}, "summary": m.get("summary", {})}
    for fold in m.get("folds", []):
        site = fold.get("site")
        if site:
            out["folds"][site] = {
                "row_only_auc": fold.get("row_only", {}).get("macro_auc_all_scoped", {}).get("macro_auc"),
                "context_auc": fold.get("context", {}).get("macro_auc_all_scoped", {}).get("macro_auc"),
                "context_file_mil_auc": fold.get("context", {}).get("file_mil_macro_auc_all_scoped", {}).get("macro_auc"),
                "context_minus_row_auc": fold.get("context_minus_row_auc"),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.input.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")

    base_cfg = build_base_sequence_cfg(cfg)
    rows, labels, y, profile, no_train, nonaves = make_rows_targets(base_cfg)
    emb = load_embedding(base_cfg, len(rows))
    x_np = add_time_features(rows, emb, cfg.include_time_features)
    y_np = y.numpy().astype(np.float32)
    groups = make_file_groups(rows)
    sites = sorted(profile["site_counts"])
    reference = load_reference(cfg.reference_metrics_json)

    folds: list[dict[str, Any]] = []
    val_idx_all = []
    val_pred_all = []
    for site in sites:
        val_files = [gi for gi, g in enumerate(groups) if g["site"] == site]
        train_files = [gi for gi, g in enumerate(groups) if g["site"] != site]
        val_idx = np.concatenate([groups[gi]["idx"] for gi in val_files]) if val_files else np.asarray([], dtype=np.int64)
        if len(val_idx) < cfg.min_val_rows:
            continue
        valid = int(((y_np[val_idx].min(axis=0) != y_np[val_idx].max(axis=0))).sum())
        if valid < cfg.min_valid_classes:
            continue
        print(json.dumps({"fold": site, "n_train_files": len(train_files), "n_val_files": len(val_files), "n_val_rows": int(len(val_idx)), "valid_classes": valid}), flush=True)
        model, train_info, pred_idx, pred = train_fold(f"tcn_{site}", groups, train_files, val_files, x_np, y, cfg)
        # align by prediction order
        order = np.argsort(pred_idx)
        pred_idx_sorted = pred_idx[order]
        pred_sorted = pred[order]
        val_rows = [rows[int(i)] for i in pred_idx_sorted]
        row_auc = macro_auc(y_np[pred_idx_sorted], pred_sorted, labels)
        nt_auc = macro_auc(y_np[pred_idx_sorted], pred_sorted, labels, subset=no_train)
        na_auc = macro_auc(y_np[pred_idx_sorted], pred_sorted, labels, subset=nonaves)
        fmil = file_mil_auc(val_rows, y_np[pred_idx_sorted], pred_sorted, labels)
        ref = reference.get("folds", {}).get(site, {})
        fold = {
            "site": site,
            "n_train_files": int(len(train_files)),
            "n_val_files": int(len(val_files)),
            "n_val_rows": int(len(pred_idx_sorted)),
            "valid_classes_raw": valid,
            "training": train_info,
            "tcn": {
                "macro_auc_all_scoped": row_auc,
                "macro_auc_no_train": nt_auc,
                "macro_auc_nonaves": na_auc,
                "file_mil_macro_auc_all_scoped": fmil,
            },
            "reference_context": ref,
            "tcn_minus_reference_context_auc": None if ref.get("context_auc") is None or row_auc.get("macro_auc") is None else float(row_auc["macro_auc"] - ref["context_auc"]),
        }
        folds.append(fold)
        val_idx_all.append(pred_idx_sorted)
        val_pred_all.append(pred_sorted)
        print(json.dumps({"fold": site, "tcn_auc": row_auc.get("macro_auc"), "tcn_file_mil_auc": fmil.get("macro_auc"), "minus_context": fold["tcn_minus_reference_context_auc"]}), flush=True)

    auc_vals = [f["tcn"]["macro_auc_all_scoped"].get("macro_auc") for f in folds]
    auc_vals = [float(v) for v in auc_vals if v is not None]
    fmil_vals = [f["tcn"]["file_mil_macro_auc_all_scoped"].get("macro_auc") for f in folds]
    fmil_vals = [float(v) for v in fmil_vals if v is not None]
    ref_ctx_mean = reference.get("summary", {}).get("context", {}).get("row_macro_auc_mean")
    ref_file_mean = reference.get("summary", {}).get("context", {}).get("file_mil_macro_auc_mean")

    # Final model on all rows: use all files as both train and validation only for export/runtime smoke.
    all_files = list(range(len(groups)))
    final_model, final_train, final_idx, final_pred = train_fold("tcn_final_all_rows", groups, all_files, all_files, x_np, y, cfg, epochs=cfg.final_train_epochs)
    final_model_cpu = final_model.cpu().eval()
    example = torch.randn(2, max(len(g["idx"]) for g in groups), x_np.shape[1])
    traced = torch.jit.trace(final_model_cpu, (example, torch.ones(example.shape[:2], dtype=torch.bool)))
    traced.save(str(out_dir / "file_tcn_torchscript.pt"))
    torch.save({"model_state": final_model_cpu.state_dict(), "config": asdict(cfg), "labels": labels, "input_dim": int(x_np.shape[1])}, out_dir / "file_tcn.pt")

    metrics = {
        "experiment_id": cfg.experiment_id,
        "track": cfg.track,
        "config": asdict(cfg),
        "data_profile": profile,
        "input_dim": int(x_np.shape[1]),
        "n_files": int(len(groups)),
        "folds": folds,
        "reference": reference,
        "summary": {
            "n_folds": int(len(folds)),
            "tcn_row_macro_auc_mean": float(np.mean(auc_vals)) if auc_vals else None,
            "tcn_row_macro_auc_min": float(np.min(auc_vals)) if auc_vals else None,
            "tcn_row_macro_auc_max": float(np.max(auc_vals)) if auc_vals else None,
            "tcn_file_mil_macro_auc_mean": float(np.mean(fmil_vals)) if fmil_vals else None,
            "tcn_file_mil_macro_auc_min": float(np.min(fmil_vals)) if fmil_vals else None,
            "tcn_file_mil_macro_auc_max": float(np.max(fmil_vals)) if fmil_vals else None,
            "reference_context_row_macro_auc_mean": ref_ctx_mean,
            "reference_context_file_mil_macro_auc_mean": ref_file_mean,
            "tcn_minus_reference_context_row_mean": None if ref_ctx_mean is None or not auc_vals else float(np.mean(auc_vals) - ref_ctx_mean),
            "tcn_minus_reference_context_file_mil_mean": None if ref_file_mean is None or not fmil_vals else float(np.mean(fmil_vals) - ref_file_mean),
            "fold_deltas_vs_context": [{"site": f["site"], "delta": f["tcn_minus_reference_context_auc"]} for f in folds],
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
    if val_idx_all:
        np.savez_compressed(out_dir / "leave_site_tcn_predictions.npz", val_idx=np.concatenate(val_idx_all), tcn_pred=np.concatenate(val_pred_all, axis=0), labels=np.array(labels))
    print(json.dumps({
        "output_dir": str(out_dir),
        "n_folds": len(folds),
        "tcn_mean_auc": metrics["summary"]["tcn_row_macro_auc_mean"],
        "tcn_file_mil_mean_auc": metrics["summary"]["tcn_file_mil_macro_auc_mean"],
        "delta_vs_context": metrics["summary"]["tcn_minus_reference_context_row_mean"],
        "delta_file_mil_vs_context": metrics["summary"]["tcn_minus_reference_context_file_mil_mean"],
        "final_nonconstant_columns": metrics["final_prediction_stats"]["nonconstant_columns"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
