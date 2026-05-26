#!/usr/bin/env python3
"""Guarded residual/gated sequence smoother for BirdCLEF train_soundscapes.

This no-slot data-point trainer follows the train_soundscapes-as-sequences
strategy.  It anchors on the stronger context-MLP features and gives a compact
per-file TCN only a bounded residual/gated correction.  The goal is to test
whether the TCN's useful S03 behavior can be recovered without the broad
S08/S19/S23 regressions seen in the unconstrained TCN probe.

It emits landscape evidence only; it is not a competition submission package.
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
    build_context_features,
    file_mil_auc,
    load_embedding,
    macro_auc,
    make_rows_targets,
)
from birdclef_soundscape_tcn_mining import (
    ResidualTcnBlock,
    add_time_features,
    load_reference,
    make_file_groups,
)


@dataclass
class GatedSequenceMiningConfig:
    experiment_id: str = "soundscape-gated-sequence-dymn10-context-tcn-losite-ep18-20260526"
    track: str = "train_soundscapes residual/gated sequence smoother data point"
    data_root: str = "/home/yourslewis/birdclef-2026/data"
    output_dir: str = "artifacts/soundscape_sequence_mining/soundscape-gated-sequence-dymn10-context-tcn-losite-ep18-20260526"
    embedding_npz: str = "artifacts/efficientat_soundscape_embeddings/efficientat-dymn10-audioset-soundscape-nonaves-notrain-nocall-siteS08-ep12-20260526/efficientat_embeddings.npz"
    embedding_key: str = "embedding"
    class_scope: str = "nonaves_or_no_train"
    context_radius: int = 1
    include_prev_next: bool = True
    include_local_mean: bool = True
    include_local_max: bool = True
    include_file_mean: bool = True
    include_file_max: bool = False
    include_time_features: bool = True
    include_site_onehot: bool = False
    row_hidden_dim: int = 256
    seq_hidden_dim: int = 128
    gate_hidden_dim: int = 96
    n_tcn_layers: int = 2
    kernel_size: int = 3
    dropout: float = 0.28
    input_dropout: float = 0.05
    max_residual_logit: float = 1.25
    gate_l1: float = 0.002
    residual_l2: float = 0.003
    epochs: int = 18
    batch_files: int = 8
    learning_rate: float = 5e-4
    weight_decay: float = 2e-4
    seed: int = 42
    pos_weight: bool = True
    pos_weight_power: float = 0.5
    pos_weight_clip: float = 20.0
    site_balanced_file_sampling: bool = True
    min_val_rows: int = 40
    min_valid_classes: int = 4
    final_train_epochs: int = 18
    reference_metrics_json: str = "artifacts/soundscape_sequence_mining/soundscape-sequence-dymn10-context-losite-ep16-20260526/metrics.json"


def load_config(path: Path | None) -> GatedSequenceMiningConfig:
    cfg = GatedSequenceMiningConfig()
    if path is None:
        return cfg
    data = json.loads(path.read_text())
    values = asdict(cfg)
    for key, value in data.items():
        if key in values:
            values[key] = value
    return GatedSequenceMiningConfig(**values)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def base_sequence_cfg(cfg: GatedSequenceMiningConfig) -> SequenceMiningConfig:
    return SequenceMiningConfig(
        experiment_id=cfg.experiment_id,
        track=cfg.track,
        data_root=cfg.data_root,
        output_dir=cfg.output_dir,
        embedding_npz=cfg.embedding_npz,
        embedding_key=cfg.embedding_key,
        class_scope=cfg.class_scope,
        context_radius=cfg.context_radius,
        include_prev_next=cfg.include_prev_next,
        include_local_mean=cfg.include_local_mean,
        include_local_max=cfg.include_local_max,
        include_file_mean=cfg.include_file_mean,
        include_file_max=cfg.include_file_max,
        include_time_features=cfg.include_time_features,
        include_site_onehot=cfg.include_site_onehot,
        hidden_dim=cfg.row_hidden_dim,
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


def file_order(groups: list[dict[str, Any]], train_files: list[int], cfg: GatedSequenceMiningConfig, rng: np.random.Generator) -> list[int]:
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


def make_pos_weight(y_train: torch.Tensor, cfg: GatedSequenceMiningConfig) -> torch.Tensor | None:
    if not cfg.pos_weight:
        return None
    pos = y_train.sum(dim=0)
    neg = y_train.shape[0] - pos
    pw = torch.ones_like(pos)
    mask = pos > 0
    pw[mask] = torch.pow(neg[mask] / torch.clamp(pos[mask], min=1.0), cfg.pos_weight_power)
    return torch.clamp(pw, 1.0, cfg.pos_weight_clip)


def batch_from_groups(
    groups: list[dict[str, Any]],
    batch_gi: list[int],
    x_ctx: torch.Tensor,
    x_seq: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    idxs = [groups[gi]["idx"] for gi in batch_gi]
    max_len = max(len(idx) for idx in idxs)
    b = len(idxs)
    xb_ctx = torch.zeros((b, max_len, x_ctx.shape[1]), dtype=x_ctx.dtype)
    xb_seq = torch.zeros((b, max_len, x_seq.shape[1]), dtype=x_seq.dtype)
    yb = torch.zeros((b, max_len, y.shape[1]), dtype=y.dtype)
    mask = torch.zeros((b, max_len), dtype=torch.bool)
    flat_indices: list[int] = []
    for bi, idx_np in enumerate(idxs):
        idx = torch.from_numpy(idx_np).long()
        t = len(idx_np)
        xb_ctx[bi, :t] = x_ctx[idx]
        xb_seq[bi, :t] = x_seq[idx]
        yb[bi, :t] = y[idx]
        mask[bi, :t] = True
        flat_indices.extend(idx_np.tolist())
    return xb_ctx, xb_seq, yb, mask, np.asarray(flat_indices, dtype=np.int64)


class RowHead(nn.Module):
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


class GatedResidualSequenceModel(nn.Module):
    def __init__(self, ctx_dim: int, seq_dim: int, out_dim: int, cfg: GatedSequenceMiningConfig):
        super().__init__()
        self.max_residual_logit = float(cfg.max_residual_logit)
        self.row_head = RowHead(ctx_dim, out_dim, cfg.row_hidden_dim, cfg.dropout)
        self.seq_norm = nn.LayerNorm(seq_dim)
        self.seq_drop = nn.Dropout(cfg.input_dropout)
        self.seq_proj = nn.Linear(seq_dim, cfg.seq_hidden_dim)
        self.seq_blocks = nn.ModuleList([
            ResidualTcnBlock(cfg.seq_hidden_dim, cfg.kernel_size, dilation=2 ** i, dropout=cfg.dropout)
            for i in range(cfg.n_tcn_layers)
        ])
        self.seq_out_norm = nn.LayerNorm(cfg.seq_hidden_dim)
        self.residual_head = nn.Linear(cfg.seq_hidden_dim, out_dim)
        self.gate = nn.Sequential(
            nn.LayerNorm(ctx_dim),
            nn.Linear(ctx_dim, cfg.gate_hidden_dim),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.gate_hidden_dim, 1),
        )

    def forward(self, x_ctx: torch.Tensor, x_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row_logits = self.row_head(x_ctx)
        z = self.seq_proj(self.seq_drop(self.seq_norm(x_seq)))
        for block in self.seq_blocks:
            z = block(z)
        z = self.seq_out_norm(z)
        residual = torch.tanh(self.residual_head(z))
        gate = torch.sigmoid(self.gate(x_ctx))
        logits = row_logits + self.max_residual_logit * gate * residual
        return logits, row_logits, gate, residual


def masked_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor | None,
    gate: torch.Tensor,
    residual: torch.Tensor,
    cfg: GatedSequenceMiningConfig,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")[mask].mean()
    gated_residual = gate * residual
    reg = cfg.residual_l2 * torch.square(gated_residual[mask]).mean() + cfg.gate_l1 * gate[mask].mean()
    return bce + reg


def predict(
    model: GatedResidualSequenceModel,
    groups: list[dict[str, Any]],
    group_indices: list[int],
    x_ctx: torch.Tensor,
    x_seq: torch.Tensor,
    y: torch.Tensor,
    cfg: GatedSequenceMiningConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    pred: list[np.ndarray] = []
    row_pred: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    idx_out: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(group_indices), cfg.batch_files):
            batch_gi = group_indices[start:start + cfg.batch_files]
            xb_ctx, xb_seq, _, mask, flat_idx = batch_from_groups(groups, batch_gi, x_ctx, x_seq, y)
            logits, row_logits, gate, _ = model(xb_ctx.to(device), xb_seq.to(device))
            mask_dev = mask.to(device)
            pred.append(torch.sigmoid(logits)[mask_dev].cpu().numpy().astype(np.float32))
            row_pred.append(torch.sigmoid(row_logits)[mask_dev].cpu().numpy().astype(np.float32))
            gates.append(gate[mask_dev].cpu().numpy().astype(np.float32))
            idx_out.append(flat_idx)
    gate_arr = np.concatenate(gates, axis=0) if gates else np.zeros((0, 1), dtype=np.float32)
    stats = {
        "gate_mean": float(gate_arr.mean()) if gate_arr.size else 0.0,
        "gate_min": float(gate_arr.min()) if gate_arr.size else 0.0,
        "gate_max": float(gate_arr.max()) if gate_arr.size else 0.0,
        "gate_std": float(gate_arr.std()) if gate_arr.size else 0.0,
    }
    return np.concatenate(idx_out), np.concatenate(pred, axis=0), {**stats, "row_pred_mean": float(np.concatenate(row_pred, axis=0).mean()) if row_pred else 0.0}


def train_fold(
    name: str,
    groups: list[dict[str, Any]],
    train_files: list[int],
    val_files: list[int],
    x_ctx_np: np.ndarray,
    x_seq_np: np.ndarray,
    y: torch.Tensor,
    cfg: GatedSequenceMiningConfig,
    epochs: int | None = None,
) -> tuple[GatedResidualSequenceModel, dict[str, Any], np.ndarray, np.ndarray, dict[str, float]]:
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_ctx = torch.from_numpy(x_ctx_np.astype(np.float32, copy=False))
    x_seq = torch.from_numpy(x_seq_np.astype(np.float32, copy=False))
    train_row_idx = np.concatenate([groups[gi]["idx"] for gi in train_files])
    pw = make_pos_weight(y[torch.from_numpy(train_row_idx).long()], cfg)
    if pw is not None:
        pw = pw.to(device)
    model = GatedResidualSequenceModel(x_ctx.shape[1], x_seq.shape[1], y.shape[1], cfg).to(device)
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
            xb_ctx, xb_seq, yb, mask, _ = batch_from_groups(groups, batch_gi, x_ctx, x_seq, y)
            logits, _, gate, residual = model(xb_ctx.to(device), xb_seq.to(device))
            loss = masked_loss(logits, yb.to(device), mask.to(device), pw, gate, residual, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
        val_losses = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(val_files), cfg.batch_files):
                batch_gi = val_files[start:start + cfg.batch_files]
                xb_ctx, xb_seq, yb, mask, _ = batch_from_groups(groups, batch_gi, x_ctx, x_seq, y)
                logits, _, gate, residual = model(xb_ctx.to(device), xb_seq.to(device))
                val_losses.append(float(masked_loss(logits, yb.to(device), mask.to(device), pw, gate, residual, cfg).detach().cpu()))
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        hist.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    idx, pred, gate_stats = predict(model, groups, val_files, x_ctx, x_seq, y, cfg, device)
    return model, {"name": name, "history": hist, "best_val_loss": best_val, "train_seconds": float(time.time() - t0)}, idx, pred, gate_stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.input.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")

    seq_cfg = base_sequence_cfg(cfg)
    rows, labels, y, profile, no_train, nonaves = make_rows_targets(seq_cfg)
    emb = load_embedding(seq_cfg, len(rows))
    x_ctx_np, context_info = build_context_features(rows, emb, seq_cfg)
    x_seq_np = add_time_features(rows, emb, cfg.include_time_features)
    y_np = y.numpy().astype(np.float32)
    groups = make_file_groups(rows)
    sites = sorted(profile["site_counts"])
    reference = load_reference(cfg.reference_metrics_json)

    folds: list[dict[str, Any]] = []
    val_idx_all: list[np.ndarray] = []
    val_pred_all: list[np.ndarray] = []
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
        model, train_info, pred_idx, pred, gate_stats = train_fold(f"gated_{site}", groups, train_files, val_files, x_ctx_np, x_seq_np, y, cfg)
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
            "gate_stats": gate_stats,
            "gated": {
                "macro_auc_all_scoped": row_auc,
                "macro_auc_no_train": nt_auc,
                "macro_auc_nonaves": na_auc,
                "file_mil_macro_auc_all_scoped": fmil,
            },
            "reference_context": ref,
            "gated_minus_reference_context_auc": None if ref.get("context_auc") is None or row_auc.get("macro_auc") is None else float(row_auc["macro_auc"] - ref["context_auc"]),
            "gated_minus_reference_context_file_mil_auc": None if ref.get("context_file_mil_auc") is None or fmil.get("macro_auc") is None else float(fmil["macro_auc"] - ref["context_file_mil_auc"]),
        }
        folds.append(fold)
        val_idx_all.append(pred_idx_sorted)
        val_pred_all.append(pred_sorted)
        print(json.dumps({"fold": site, "gated_auc": row_auc.get("macro_auc"), "gated_file_mil_auc": fmil.get("macro_auc"), "minus_context": fold["gated_minus_reference_context_auc"], "gate_mean": gate_stats["gate_mean"]}), flush=True)

    auc_vals = [f["gated"]["macro_auc_all_scoped"].get("macro_auc") for f in folds]
    auc_vals = [float(v) for v in auc_vals if v is not None]
    fmil_vals = [f["gated"]["file_mil_macro_auc_all_scoped"].get("macro_auc") for f in folds]
    fmil_vals = [float(v) for v in fmil_vals if v is not None]
    ref_ctx_mean = reference.get("summary", {}).get("context", {}).get("row_macro_auc_mean")
    ref_file_mean = reference.get("summary", {}).get("context", {}).get("file_mil_macro_auc_mean")
    fold_deltas = [{"site": f["site"], "delta": f["gated_minus_reference_context_auc"], "file_mil_delta": f["gated_minus_reference_context_file_mil_auc"]} for f in folds]
    guard_sites = {"S03", "S22"}
    guard = {
        "beats_context_row_mean": bool(ref_ctx_mean is not None and auc_vals and float(np.mean(auc_vals)) > float(ref_ctx_mean)),
        "beats_context_file_mil_mean": bool(ref_file_mean is not None and fmil_vals and float(np.mean(fmil_vals)) > float(ref_file_mean)),
        "s03_s22_not_regressed": all((d["delta"] is not None and d["delta"] >= 0.0) for d in fold_deltas if d["site"] in guard_sites),
        "max_negative_fold_delta": float(min([d["delta"] for d in fold_deltas if d["delta"] is not None], default=0.0)),
        "decision": "PROMOTE_TO_WRAPPER_AUDIT" if False else "NO_SUBMISSION_DECISION_PENDING",
    }
    guard["decision"] = "PROMOTE_TO_WRAPPER_AUDIT" if (guard["beats_context_row_mean"] and guard["s03_s22_not_regressed"] and guard["max_negative_fold_delta"] >= -0.03) else "REJECT_AS_SUBMISSION__KEEP_AS_DATA_POINT"

    all_files = list(range(len(groups)))
    final_model, final_train, final_idx, final_pred, final_gate = train_fold("gated_final_all_rows", groups, all_files, all_files, x_ctx_np, x_seq_np, y, cfg, epochs=cfg.final_train_epochs)
    final_model_cpu = final_model.cpu().eval()
    max_len = max(len(g["idx"]) for g in groups)
    example_ctx = torch.randn(2, max_len, x_ctx_np.shape[1])
    example_seq = torch.randn(2, max_len, x_seq_np.shape[1])
    traced = torch.jit.trace(final_model_cpu, (example_ctx, example_seq), strict=False)
    traced.save(str(out_dir / "gated_sequence_torchscript.pt"))
    torch.save({
        "model_state": final_model_cpu.state_dict(),
        "config": asdict(cfg),
        "labels": labels,
        "context_info": context_info,
        "ctx_dim": int(x_ctx_np.shape[1]),
        "seq_dim": int(x_seq_np.shape[1]),
    }, out_dir / "gated_sequence.pt")

    metrics = {
        "experiment_id": cfg.experiment_id,
        "track": cfg.track,
        "config": asdict(cfg),
        "data_profile": profile,
        "context_features": context_info,
        "seq_input_dim": int(x_seq_np.shape[1]),
        "n_files": int(len(groups)),
        "folds": folds,
        "reference": reference,
        "summary": {
            "n_folds": int(len(folds)),
            "gated_row_macro_auc_mean": float(np.mean(auc_vals)) if auc_vals else None,
            "gated_row_macro_auc_min": float(np.min(auc_vals)) if auc_vals else None,
            "gated_row_macro_auc_max": float(np.max(auc_vals)) if auc_vals else None,
            "gated_file_mil_macro_auc_mean": float(np.mean(fmil_vals)) if fmil_vals else None,
            "gated_file_mil_macro_auc_min": float(np.min(fmil_vals)) if fmil_vals else None,
            "gated_file_mil_macro_auc_max": float(np.max(fmil_vals)) if fmil_vals else None,
            "reference_context_row_macro_auc_mean": ref_ctx_mean,
            "reference_context_file_mil_macro_auc_mean": ref_file_mean,
            "gated_minus_reference_context_row_mean": None if ref_ctx_mean is None or not auc_vals else float(np.mean(auc_vals) - ref_ctx_mean),
            "gated_minus_reference_context_file_mil_mean": None if ref_file_mean is None or not fmil_vals else float(np.mean(fmil_vals) - ref_file_mean),
            "fold_deltas_vs_context": fold_deltas,
            "guard": guard,
        },
        "final_all_rows_training": final_train,
        "final_prediction_stats": {
            "min": float(final_pred.min()),
            "max": float(final_pred.max()),
            "mean": float(final_pred.mean()),
            "std": float(final_pred.std()),
            "nonconstant_columns": int((final_pred.std(axis=0) > 1e-7).sum()),
            **final_gate,
        },
        "labels": labels,
        "rows_preview": rows[:8],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out_dir / "data_profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    (out_dir / "config.resolved.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    if val_idx_all:
        np.savez_compressed(out_dir / "leave_site_gated_predictions.npz", val_idx=np.concatenate(val_idx_all), gated_pred=np.concatenate(val_pred_all, axis=0), labels=np.array(labels))
    print(json.dumps({
        "output_dir": str(out_dir),
        "n_folds": len(folds),
        "gated_mean_auc": metrics["summary"]["gated_row_macro_auc_mean"],
        "gated_file_mil_mean_auc": metrics["summary"]["gated_file_mil_macro_auc_mean"],
        "delta_vs_context": metrics["summary"]["gated_minus_reference_context_row_mean"],
        "delta_file_mil_vs_context": metrics["summary"]["gated_minus_reference_context_file_mil_mean"],
        "guard": guard,
        "final_nonconstant_columns": metrics["final_prediction_stats"]["nonconstant_columns"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
