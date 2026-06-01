#!/usr/bin/env python3
"""Train a hidden-safe train_soundscape student from the file-calibration teacher.

This is a no-submit ClawTeam data-point trainer.  It distills the strongest
file-level calibration diagnostic (PANNs row head + DyMN10 file evidence) into a
single fused-embedding context MLP that can be exported and, in principle,
re-run on hidden soundscapes without using leave-site OOF matrices as outputs.

Outputs:
- leave-site hard-label validation metrics;
- a TorchScript context head smoke/export;
- a 72->234 anchor-filled v616 proxy sidecar audit for the OOF predictions.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def parse_time_seconds(text: str) -> int:
    parts = [float(x) for x in str(text).split(":")]
    if len(parts) == 3:
        sec = parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        sec = parts[0] * 60 + parts[1]
    else:
        sec = parts[0]
    return int(round(sec)) if abs(sec - round(sec)) < 1e-6 else int(sec)


def row_id(filename: str, end_text: str) -> str:
    return f"{Path(filename).stem}_{parse_time_seconds(end_text)}"


def site_from_filename(name: str) -> str:
    m = re.search(r"_(S\d+)_", str(name))
    return m.group(1) if m else "UNKNOWN"


@dataclass
class StudentConfig:
    experiment_id: str = "soundscape-filecal-teacher-student-fused-r2-losite-20260601"
    data_root: str = "data"
    output_dir: str = "artifacts/filecal_teacher_student/soundscape-filecal-teacher-student-fused-r2-losite-20260601"
    embedding_npz: str = "artifacts/fused_soundscape_embeddings/dymn10_panns_cnn14_train_soundscapes_20260527/fused_embeddings.npz"
    embedding_key: str = "embedding"
    panns_oof_npz: str = "artifacts/soundscape_sequence_mining/soundscape-sequence-panns-cnn14-nonaves-notrain-rowonly-losite-ep24-20260531/leave_site_predictions.npz"
    dymn_oof_npz: str = "artifacts/soundscape_sequence_mining/soundscape-sequence-dymn10-nonaves-notrain-r2-filectx-filemil-losite-ep22-20260531/leave_site_predictions.npz"
    panns_pred_key: str = "row_only_pred"
    dymn_pred_key: str = "context_pred"
    teacher_alpha: float = 0.35
    teacher_file_mode: str = "mean"  # mean|max
    context_radius: int = 2
    include_prev_next: bool = True
    include_local_mean: bool = True
    include_local_max: bool = True
    include_file_mean: bool = True
    include_file_max: bool = True
    include_time_features: bool = True
    hidden_dim: int = 384
    dropout: float = 0.35
    epochs: int = 24
    batch_size: int = 80
    learning_rate: float = 3.5e-4
    weight_decay: float = 1.5e-3
    seed: int = 611
    hard_loss_weight: float = 0.55
    soft_loss_weight: float = 0.45
    pos_weight_power: float = 0.40
    pos_weight_clip: float = 12.0
    site_balanced_sampling: bool = True
    min_val_rows: int = 40
    min_valid_classes: int = 2
    anchor_csv: str = "artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission_anchor_raw.csv"
    v616_csv: str = "artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission.csv"
    bootstrap_iters: int = 200


def load_config(path: Path | None) -> StudentConfig:
    cfg = StudentConfig()
    if path is None:
        return cfg
    vals = asdict(cfg)
    vals.update(json.loads(path.read_text()))
    return StudentConfig(**vals)


def choose_labels(data_root: Path, sound_df: pd.DataFrame) -> tuple[list[str], set[str], set[str], dict[str, Any]]:
    taxonomy = pd.read_csv(data_root / "taxonomy.csv", dtype={"primary_label": str})
    train = pd.read_csv(data_root / "train.csv", dtype={"primary_label": str})
    all_labels = taxonomy["primary_label"].astype(str).tolist()
    train_labels = set(train["primary_label"].astype(str))
    no_train = {x for x in all_labels if x not in train_labels}
    nonaves = set(taxonomy.loc[taxonomy["class_name"].astype(str) != "Aves", "primary_label"].astype(str))
    labels = [x for x in all_labels if x in no_train or x in nonaves]
    positives: set[str] = set()
    for raw in sound_df["primary_label"].fillna("").astype(str):
        positives.update(x.strip() for x in raw.split(";") if x.strip())
    info = {
        "n_taxonomy_labels": len(all_labels),
        "n_training_labels": len(labels),
        "n_no_train_labels": len(no_train),
        "n_nonaves_labels": len(nonaves),
        "n_soundscape_positive_labels": len(positives),
        "scope": "nonaves_or_no_train",
    }
    return labels, no_train, nonaves, info


def load_rows_targets(cfg: StudentConfig) -> tuple[list[dict[str, Any]], list[str], np.ndarray, set[str], set[str], dict[str, Any]]:
    root = Path(cfg.data_root)
    sound_df = pd.read_csv(root / "train_soundscapes_labels.csv", dtype=str)
    labels, no_train, nonaves, info = choose_labels(root, sound_df)
    lab_to_i = {lab: i for i, lab in enumerate(labels)}
    rows: list[dict[str, Any]] = []
    y = np.zeros((len(sound_df), len(labels)), dtype=np.float32)
    for i, r in enumerate(sound_df.itertuples(index=False)):
        filename = str(getattr(r, "filename"))
        labs = [x.strip() for x in str(getattr(r, "primary_label")).split(";") if x.strip()]
        rows.append({
            "src_idx": i,
            "filename": filename,
            "start": str(getattr(r, "start")),
            "end": str(getattr(r, "end")),
            "start_sec": float(parse_time_seconds(str(getattr(r, "start")))),
            "row_id": row_id(filename, str(getattr(r, "end"))),
            "site": site_from_filename(filename),
            "labels_raw": labs,
        })
        for lab in labs:
            j = lab_to_i.get(lab)
            if j is not None:
                y[i, j] = 1.0
    sites = pd.Series([r["site"] for r in rows])
    files = pd.Series([r["filename"] for r in rows])
    profile = {
        **info,
        "n_windows": len(rows),
        "n_files": int(files.nunique()),
        "n_sites": int(sites.nunique()),
        "site_counts": {str(k): int(v) for k, v in sites.value_counts().sort_index().items()},
        "positive_cells": int(y.sum()),
        "density": float(y.mean()),
    }
    return rows, labels, y, no_train, nonaves, profile


def build_features(rows: list[dict[str, Any]], emb: np.ndarray, cfg: StudentConfig) -> tuple[np.ndarray, dict[str, Any]]:
    by_file: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_file.setdefault(r["filename"], []).append(i)
    for idxs in by_file.values():
        idxs.sort(key=lambda i: rows[i]["start_sec"])
    file_mean = {f: emb[idxs].mean(axis=0) for f, idxs in by_file.items()}
    file_max = {f: emb[idxs].max(axis=0) for f, idxs in by_file.items()}
    max_start = {f: max(rows[i]["start_sec"] for i in idxs) for f, idxs in by_file.items()}
    pos: dict[int, tuple[list[int], int]] = {}
    for f, idxs in by_file.items():
        for k, i in enumerate(idxs):
            pos[i] = (idxs, k)
    out = []
    for i, r in enumerate(rows):
        idxs, k = pos[i]
        parts = [emb[i]]
        if cfg.include_prev_next:
            parts += [emb[idxs[max(0, k - 1)]], emb[idxs[min(len(idxs) - 1, k + 1)]]]
        lo, hi = max(0, k - cfg.context_radius), min(len(idxs), k + cfg.context_radius + 1)
        local = emb[idxs[lo:hi]]
        if cfg.include_local_mean:
            parts.append(local.mean(axis=0))
        if cfg.include_local_max:
            parts.append(local.max(axis=0))
        if cfg.include_file_mean:
            parts.append(file_mean[r["filename"]])
        if cfg.include_file_max:
            parts.append(file_max[r["filename"]])
        if cfg.include_time_features:
            frac = r["start_sec"] / max(max_start[r["filename"]], 1.0)
            parts.append(np.array([frac, math.sin(2 * math.pi * frac), math.cos(2 * math.pi * frac), len(idxs) / 12.0], dtype=np.float32))
        out.append(np.concatenate(parts).astype(np.float32))
    x = np.stack(out)
    return x, {"row_embedding_dim": int(emb.shape[1]), "context_feature_dim": int(x.shape[1]), "n_files": len(by_file), "context_radius": cfg.context_radius}


def load_oof(path: Path, key: str, n_rows: int, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    idx = z["val_idx"].astype(np.int64)
    pred = z[key].astype(np.float32)
    zlabels = [str(x) for x in z["labels"].tolist()]
    if zlabels != labels:
        raise ValueError(f"label mismatch for {path}: {len(zlabels)} vs {len(labels)}")
    all_pred = np.full((n_rows, len(labels)), np.nan, dtype=np.float32)
    all_pred[idx] = pred
    return idx, all_pred


def file_broadcast(rows: list[dict[str, Any]], arr: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    out = np.full_like(arr, np.nan)
    by_file: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        if mask[i]:
            by_file.setdefault(r["filename"], []).append(i)
    for idxs in by_file.values():
        block = arr[idxs]
        pooled = np.nanmean(block, axis=0) if mode == "mean" else np.nanmax(block, axis=0)
        for i in idxs:
            out[i] = pooled
    return out


def make_teacher(rows: list[dict[str, Any]], cfg: StudentConfig, labels: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    n = len(rows)
    p_idx, panns = load_oof(Path(cfg.panns_oof_npz), cfg.panns_pred_key, n, labels)
    d_idx, dymn = load_oof(Path(cfg.dymn_oof_npz), cfg.dymn_pred_key, n, labels)
    mask = np.isfinite(panns).all(axis=1) & np.isfinite(dymn).all(axis=1)
    d_file = file_broadcast(rows, dymn, mask, cfg.teacher_file_mode)
    teacher = np.full_like(panns, np.nan)
    teacher[mask] = sigmoid(logit(panns[mask]) + cfg.teacher_alpha * logit(d_file[mask])).astype(np.float32)
    info = {
        "panns_oof_rows": int(len(p_idx)),
        "dymn_oof_rows": int(len(d_idx)),
        "teacher_rows": int(mask.sum()),
        "teacher_alpha": cfg.teacher_alpha,
        "teacher_file_mode": cfg.teacher_file_mode,
        "teacher_stats": {"min": float(np.nanmin(teacher)), "max": float(np.nanmax(teacher)), "mean": float(np.nanmean(teacher)), "std": float(np.nanstd(teacher))},
    }
    return teacher, mask, info


class Head(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def pos_weight(y: torch.Tensor, cfg: StudentConfig) -> torch.Tensor:
    pos = y.sum(dim=0)
    neg = y.shape[0] - pos
    pw = torch.ones_like(pos)
    m = pos > 0
    pw[m] = torch.pow(neg[m] / torch.clamp(pos[m], min=1.0), cfg.pos_weight_power)
    return torch.clamp(pw, 1.0, cfg.pos_weight_clip)


def balanced_order(rows: list[dict[str, Any]], idx: np.ndarray, rng: np.random.Generator, enabled: bool) -> np.ndarray:
    if not enabled:
        return rng.permutation(idx)
    by_site: dict[str, list[int]] = {}
    for i in idx.tolist():
        by_site.setdefault(rows[int(i)]["site"], []).append(int(i))
    max_n = max(len(v) for v in by_site.values())
    sampled: list[int] = []
    for vals in by_site.values():
        sampled.extend(rng.choice(vals, size=max_n, replace=len(vals) < max_n).tolist())
    return rng.permutation(np.asarray(sampled, dtype=np.int64))


def train_fold(x_np: np.ndarray, y_np: np.ndarray, teacher_np: np.ndarray, rows: list[dict[str, Any]], train_idx: np.ndarray, cfg: StudentConfig) -> tuple[Head, dict[str, Any]]:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device("cpu")
    x = torch.from_numpy(x_np.astype(np.float32)).to(device)
    y = torch.from_numpy(y_np.astype(np.float32)).to(device)
    t = torch.from_numpy(teacher_np.astype(np.float32)).to(device)
    model = Head(x.shape[1], y.shape[1], cfg.hidden_dim, cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    train_t = torch.from_numpy(train_idx.astype(np.int64)).to(device)
    pw = pos_weight(y[train_t], cfg).to(device)
    best_state = None
    best_loss = float("inf")
    hist = []
    t0 = time.time()
    for ep in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        order = balanced_order(rows, train_idx, rng, cfg.site_balanced_sampling)
        for st in range(0, len(order), cfg.batch_size):
            bi = torch.from_numpy(order[st:st + cfg.batch_size].astype(np.int64)).to(device)
            logits = model(x[bi])
            hard = F.binary_cross_entropy_with_logits(logits, y[bi], pos_weight=pw)
            soft = F.binary_cross_entropy_with_logits(logits, t[bi])
            loss = cfg.hard_loss_weight * hard + cfg.soft_loss_weight * soft
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        hist.append({"epoch": ep, "train_loss": mean_loss})
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval(), {"best_train_loss": best_loss, "history": hist, "seconds": time.time() - t0}


def macro_auc(y: np.ndarray, p: np.ndarray, labels: list[str], subset: set[str] | None = None) -> dict[str, Any]:
    vals = []
    for j, lab in enumerate(labels):
        if subset is not None and lab not in subset:
            continue
        yy = y[:, j]
        if yy.min() == yy.max():
            continue
        vals.append(float(roc_auc_score(yy, p[:, j])))
    return {"macro_auc": float(np.mean(vals)) if vals else None, "valid_classes": len(vals)}


def file_mil_auc(rows: list[dict[str, Any]], y: np.ndarray, p: np.ndarray, labels: list[str], subset: set[str] | None = None) -> dict[str, Any]:
    files = sorted({r["filename"] for r in rows})
    yf, pf = [], []
    for f in files:
        idx = [i for i, r in enumerate(rows) if r["filename"] == f]
        yf.append(y[idx].max(axis=0))
        pf.append(p[idx].max(axis=0))
    out = macro_auc(np.stack(yf), np.stack(pf), labels, subset)
    out["n_files"] = len(files)
    return out


def summarize_folds(folds: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(path: tuple[str, ...]) -> float | None:
        vals = []
        for f in folds:
            cur: Any = f
            for k in path:
                cur = cur[k]
            if cur is not None:
                vals.append(float(cur))
        return float(np.mean(vals)) if vals else None
    return {
        "n_folds": len(folds),
        "row_macro_auc_mean": avg(("student", "macro_auc_all", "macro_auc")),
        "file_mil_macro_auc_mean": avg(("student", "file_mil_all", "macro_auc")),
        "no_train_macro_auc_mean": avg(("student", "macro_auc_no_train", "macro_auc")),
        "nonaves_macro_auc_mean": avg(("student", "macro_auc_nonaves", "macro_auc")),
        "teacher_row_macro_auc_mean": avg(("teacher", "macro_auc_all", "macro_auc")),
        "teacher_file_mil_macro_auc_mean": avg(("teacher", "file_mil_all", "macro_auc")),
    }


def read_csv_matrix(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows, vals = [], []
        for row in reader:
            rows.append(row[0])
            vals.append([float(x) for x in row[1:]])
    return rows, header[1:], np.asarray(vals, dtype=np.float32)


def write_csv_matrix(path: Path, row_ids: list[str], cols: list[str], vals: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", *cols])
        for rid, arr in zip(row_ids, vals):
            w.writerow([rid, *[f"{float(x):.8g}" for x in arr]])


def build_sidecar(rows: list[dict[str, Any]], labels: list[str], val_idx: np.ndarray, pred: np.ndarray, cfg: StudentConfig, out_csv: Path) -> dict[str, Any]:
    base_rows, cols, base = read_csv_matrix(Path(cfg.anchor_csv))
    col_i = {c: i for i, c in enumerate(cols)}
    row_i = {r: i for i, r in enumerate(base_rows)}
    side = base.copy()
    matched = set()
    for k, src in enumerate(val_idx.tolist()):
        rid = rows[int(src)]["row_id"]
        bi = row_i.get(rid)
        if bi is None:
            continue
        for j, lab in enumerate(labels):
            side[bi, col_i[lab]] = float(pred[k, j])
        matched.add(rid)
    if not np.isfinite(side).all():
        raise ValueError("non-finite sidecar")
    write_csv_matrix(out_csv, base_rows, cols, side)
    return {"path": str(out_csv), "rows": len(base_rows), "class_columns": len(cols), "matched_proxy_rows": len(matched), "unmatched_proxy_rows": len(base_rows) - len(matched), "finite": True, "nonconstant_columns": int(((side.max(axis=0)-side.min(axis=0))>0).sum()), "value_stats": {"min": float(side.min()), "max": float(side.max()), "mean": float(side.mean()), "std": float(side.std())}}


def make_manifest(cfg: StudentConfig, sidecar_csv: Path, out_dir: Path) -> Path:
    manifest = {
        "name": "birdclef_filecal_teacher_student_sidecar_audit",
        "description": "No-submit audit of hidden-safe file-calibration teacher student OOF sidecar wrapped into 234-class v616 proxy matrix.",
        "labels_csv": "data/train_soundscapes_labels.csv",
        "anchor": "anchor_v616_raw",
        "baseline": "v616_final",
        "anchor_recipe": "anchor_only",
        "baseline_recipe": "v616_baseline",
        "allow_submit_approval": False,
        "members": {
            "anchor_v616_raw": {"role": "anchor", "description": "v616 anchor raw output", "path": cfg.anchor_csv, "hidden_safe_status": "private_verifier_output"},
            "v616_final": {"role": "baseline_tied_recipe", "description": "Actual submitted v616 final output", "path": cfg.v616_csv, "hidden_safe_status": "submitted_private_verifier_output", "public_lb": 0.949},
            "filecal_student_sidecar": {"role": "analysis_branch", "description": "Fused-embedding student distilled from PANNs-row + DyMN10-filemean calibration teacher; 72-label scope anchor-filled elsewhere.", "path": str(sidecar_csv), "hidden_safe_status": "student_model_oof_proxy_not_submission_package"},
        },
        "recipes": [
            {"name": "anchor_only", "type": "rank_blend", "weights": {"anchor_v616_raw": 1.0}},
            {"name": "v616_baseline", "type": "member", "member": "v616_final"},
        ] + [
            {"name": f"filecal_student_w{str(w).replace('0.','').replace('.','p')}", "type": "rank_blend", "description": f"{w:.1%} filecal student sidecar", "weights": {"anchor_v616_raw": 1.0 - w, "filecal_student_sidecar": w}}
            for w in (0.005, 0.01, 0.02, 0.04, 0.08)
        ],
        "bootstrap": {"iters": cfg.bootstrap_iters, "groups": ["site", "file"]},
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def summarize_audit(audit_json: Path, out_json: Path) -> dict[str, Any]:
    data = json.loads(audit_json.read_text())
    rows = []
    for r in data.get("recipes", []):
        comps = r.get("comparisons", {})
        rows.append({
            "recipe": r.get("name"),
            "macro_auc": r.get("local_metrics", {}).get("macro_auc"),
            "valid_classes": r.get("local_metrics", {}).get("valid_auc_classes"),
            "lift_vs_anchor": comps.get("anchor_only", {}).get("macro_auc_lift"),
            "lift_vs_v616": comps.get("v616_baseline", {}).get("macro_auc_lift"),
            "rank_corr_vs_v616": comps.get("v616_baseline", {}).get("rank_corr"),
            "mae_vs_v616": comps.get("v616_baseline", {}).get("mae"),
            "eligible": r.get("gate", {}).get("eligible_for_submission"),
            "gate": r.get("gate", {}).get("reason"),
        })
    ranked = sorted([x for x in rows if x["macro_auc"] is not None], key=lambda x: (float(x.get("lift_vs_v616") or -999), float(x.get("macro_auc") or -999)), reverse=True)
    summary = {"created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "audit_json": str(audit_json), "top_by_lift_vs_v616": ranked[:10], "all_recipes": rows, "submit_approved": bool(data.get("submit_approved", False))}
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--python", default="python")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.resolved.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")

    rows, labels, y, no_train, nonaves, profile = load_rows_targets(cfg)
    emb = np.load(cfg.embedding_npz, allow_pickle=False)[cfg.embedding_key].astype(np.float32)
    x, feature_info = build_features(rows, emb, cfg)
    teacher, teacher_mask, teacher_info = make_teacher(rows, cfg, labels)

    sites = sorted(profile["site_counts"])
    folds = []
    all_idx, all_pred = [], []
    for site in sites:
        val_idx = np.asarray([i for i, r in enumerate(rows) if r["site"] == site and teacher_mask[i]], dtype=np.int64)
        train_idx = np.asarray([i for i, r in enumerate(rows) if r["site"] != site and teacher_mask[i]], dtype=np.int64)
        if len(val_idx) < cfg.min_val_rows:
            continue
        valid = int((y[val_idx].min(axis=0) != y[val_idx].max(axis=0)).sum())
        if valid < cfg.min_valid_classes:
            continue
        print(json.dumps({"fold": site, "n_train": len(train_idx), "n_val": len(val_idx), "valid_classes": valid}), flush=True)
        model, tr = train_fold(x, y, teacher, rows, train_idx, cfg)
        with torch.no_grad():
            pred = torch.sigmoid(model(torch.from_numpy(x[val_idx].astype(np.float32)))).numpy().astype(np.float32)
        val_rows = [rows[int(i)] for i in val_idx]
        fold = {
            "site": site,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_val_files": int(len({r["filename"] for r in val_rows})),
            "valid_classes_raw": valid,
            "training": tr,
            "student": {
                "macro_auc_all": macro_auc(y[val_idx], pred, labels),
                "macro_auc_no_train": macro_auc(y[val_idx], pred, labels, no_train),
                "macro_auc_nonaves": macro_auc(y[val_idx], pred, labels, nonaves),
                "file_mil_all": file_mil_auc(val_rows, y[val_idx], pred, labels),
            },
            "teacher": {
                "macro_auc_all": macro_auc(y[val_idx], teacher[val_idx], labels),
                "macro_auc_no_train": macro_auc(y[val_idx], teacher[val_idx], labels, no_train),
                "macro_auc_nonaves": macro_auc(y[val_idx], teacher[val_idx], labels, nonaves),
                "file_mil_all": file_mil_auc(val_rows, y[val_idx], teacher[val_idx], labels),
            },
        }
        folds.append(fold)
        all_idx.append(val_idx)
        all_pred.append(pred)
        print(json.dumps({"fold": site, "student_auc": fold["student"]["macro_auc_all"]["macro_auc"], "teacher_auc": fold["teacher"]["macro_auc_all"]["macro_auc"]}), flush=True)

    val_idx_all = np.concatenate(all_idx) if all_idx else np.asarray([], dtype=np.int64)
    pred_all = np.concatenate(all_pred, axis=0) if all_pred else np.zeros((0, len(labels)), dtype=np.float32)
    summary = summarize_folds(folds)

    # final export/smoke: train on all teacher rows, export context head
    final_model, final_train = train_fold(x, y, teacher, rows, np.where(teacher_mask)[0].astype(np.int64), cfg)
    final_model_cpu = final_model.cpu().eval()
    traced = torch.jit.trace(final_model_cpu, torch.randn(2, x.shape[1]))
    traced.save(str(out_dir / "filecal_teacher_student_context_head.ts.pt"))
    torch.save({"model_state": final_model_cpu.state_dict(), "config": asdict(cfg), "labels": labels, "feature_info": feature_info}, out_dir / "filecal_teacher_student_context_head.pt")

    np.savez_compressed(out_dir / "leave_site_predictions.npz", val_idx=val_idx_all, context_pred=pred_all, labels=np.asarray(labels))
    metrics = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "experiment_id": cfg.experiment_id,
        "branch_family": "train_soundscapes hidden-safe file-calibration teacher student",
        "config": asdict(cfg),
        "data_profile": profile,
        "teacher_info": teacher_info,
        "feature_info": feature_info,
        "folds": folds,
        "summary": summary,
        "final_train": final_train,
        "prediction_stats": {"min": float(pred_all.min()) if pred_all.size else None, "max": float(pred_all.max()) if pred_all.size else None, "mean": float(pred_all.mean()) if pred_all.size else None, "std": float(pred_all.std()) if pred_all.size else None, "nonconstant_labels": int((pred_all.std(axis=0) > 1e-7).sum()) if pred_all.size else 0},
        "decision_hint": "comparison-grade hidden-safe student data point; require v616 sidecar/private verifier lift before submission",
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    sidecar_dir = out_dir / "sidecar_audit"
    sidecar_csv = sidecar_dir / "sidecars" / "filecal_teacher_student_72to234.csv"
    sidecar_report = build_sidecar(rows, labels, val_idx_all, pred_all, cfg, sidecar_csv)
    (sidecar_dir / "sidecar_report.json").parent.mkdir(parents=True, exist_ok=True)
    (sidecar_dir / "sidecar_report.json").write_text(json.dumps(sidecar_report, indent=2) + "\n")
    manifest = make_manifest(cfg, sidecar_csv, sidecar_dir)
    audit_dir = sidecar_dir / "audit"
    cmd = [args.python, "scripts/birdclef_ensemble_strategy_audit.py", "--manifest", str(manifest), "--output-dir", str(audit_dir), "--bootstrap-iters", str(cfg.bootstrap_iters), "--emit-candidate-csvs"]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    (sidecar_dir / "audit_command.txt").write_text(" ".join(cmd) + "\n")
    (sidecar_dir / "audit_stdout.txt").write_text(proc.stdout)
    (sidecar_dir / "audit_stderr.txt").write_text(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"audit failed code {proc.returncode}: {proc.stderr[-1000:]}")
    audit_summary = summarize_audit(audit_dir / "ensemble_strategy_audit.json", sidecar_dir / "audit_summary.json")
    print(json.dumps({"output_dir": str(out_dir), "summary": summary, "sidecar_report": sidecar_report, "top_sidecar": audit_summary["top_by_lift_vs_v616"][:5], "submit_approved": audit_summary["submit_approved"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
