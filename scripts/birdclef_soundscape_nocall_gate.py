#!/usr/bin/env python3
"""Trusted no-call/background gate data point for BirdCLEF 2026.

This no-slot ClawTeam probe evaluates whether existing soundscape-native package
predictions contain a usable signal for distinguishing labeled target-call 5s
windows from unlabeled/background 5s windows inside official train_soundscapes.

Important caveat: the negative class is "unlabeled in train_soundscapes_labels",
not a hand-verified no-call label.  The output is comparison-grade only unless a
separate negative-label audit upgrades the protocol.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"scikit-learn is required for this probe: {exc}")


@dataclass
class NoCallGateConfig:
    experiment_id: str = "soundscape-nocall-gate-soft1279pair-losite-20260528"
    track: str = "Trusted no-call/background gate over soundscape package predictions"
    data_root: str = "data"
    output_dir: str = "artifacts/soundscape_nocall_gate/soundscape-nocall-gate-soft1279pair-losite-20260528"
    feature_npz: list[str] | None = None
    feature_names: list[str] | None = None
    c_value: float = 0.20
    max_iter: int = 2000
    seed: int = 83
    include_class_probs: bool = True
    # Optional stricter weak-negative protocol.  Unlabeled soundscape windows
    # closer than this many seconds to a labeled positive in the same file are
    # excluded from the binary gate fit/eval rather than treated as no-call.
    # Default 0 preserves the original all-unlabeled protocol.
    negative_min_distance_sec: int = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--experiment-id", default=None)
    return p.parse_args()


def load_config(path: Path | None) -> NoCallGateConfig:
    cfg = NoCallGateConfig()
    if path is None:
        return cfg
    data = json.loads(path.read_text())
    values = asdict(cfg)
    for k, v in data.items():
        if k in values:
            values[k] = v
    return NoCallGateConfig(**values)


def parse_row_id(row_id: str) -> tuple[str, int, int, str]:
    m = re.match(r"(.+)_(\d+)$", str(row_id))
    if not m:
        raise ValueError(f"Cannot parse row_id={row_id!r}")
    stem, end_s = m.group(1), int(m.group(2))
    filename = f"{stem}.ogg"
    start_s = end_s - 5
    site_m = re.search(r"_(S\d+)_", filename)
    site = site_m.group(1) if site_m else "UNKNOWN"
    return filename, start_s, end_s, site


def parse_time_seconds(text: str) -> int:
    parts = [int(float(x)) for x in str(text).split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def load_positive_windows(data_root: Path) -> tuple[dict[tuple[str, int], set[str]], dict[str, Any]]:
    labels_df = pd.read_csv(data_root / "train_soundscapes_labels.csv", dtype=str)
    pos: dict[tuple[str, int], set[str]] = {}
    duplicate_rows = 0
    for r in labels_df.itertuples(index=False):
        filename = str(getattr(r, "filename"))
        start_s = parse_time_seconds(str(getattr(r, "start")))
        labs = {x.strip() for x in str(getattr(r, "primary_label")).split(";") if x.strip()}
        key = (filename, start_s)
        if key in pos:
            duplicate_rows += 1
            pos[key].update(labs)
        else:
            pos[key] = set(labs)
    profile = {
        "raw_label_rows": int(len(labels_df)),
        "unique_labeled_windows": int(len(pos)),
        "duplicate_label_window_rows": int(duplicate_rows),
        "n_files_with_labels": int(labels_df["filename"].nunique()),
    }
    return pos, profile


def load_feature_npz(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    z = np.load(path, allow_pickle=True)
    row_ids = z["row_ids"].astype(str)
    labels = z["labels"].astype(str).tolist()
    probs = z["probs"].astype(np.float32)
    if probs.shape[0] != len(row_ids):
        raise ValueError(f"row/prob mismatch in {path}")
    return row_ids, probs, labels


def aggregate_features(probs: np.ndarray, prefix: str, include_class_probs: bool = True) -> tuple[np.ndarray, list[str]]:
    eps = 1e-7
    p = np.clip(probs, eps, 1 - eps)
    # Sort descending once for top-k and confidence/mass summaries.
    sorted_p = np.sort(p, axis=1)[:, ::-1]
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p)).mean(axis=1, keepdims=True)
    feats = []
    names: list[str] = []
    if include_class_probs:
        feats.append(probs)
        names.extend(f"{prefix}_class_{i:03d}" for i in range(probs.shape[1]))
    feats.extend([
        sorted_p[:, :1].mean(axis=1, keepdims=True),
        sorted_p[:, :3].mean(axis=1, keepdims=True),
        sorted_p[:, :5].mean(axis=1, keepdims=True),
        sorted_p[:, :10].mean(axis=1, keepdims=True),
        sorted_p[:, :25].mean(axis=1, keepdims=True),
        p.sum(axis=1, keepdims=True),
        p.mean(axis=1, keepdims=True),
        p.std(axis=1, keepdims=True),
        entropy,
        (p > 0.50).sum(axis=1, keepdims=True).astype(np.float32),
        (p > 0.20).sum(axis=1, keepdims=True).astype(np.float32),
        (p > 0.10).sum(axis=1, keepdims=True).astype(np.float32),
        (p > 0.05).sum(axis=1, keepdims=True).astype(np.float32),
    ])
    names += [
        f"{prefix}_top1", f"{prefix}_top3", f"{prefix}_top5", f"{prefix}_top10", f"{prefix}_top25",
        f"{prefix}_sum", f"{prefix}_mean", f"{prefix}_std", f"{prefix}_entropy",
        f"{prefix}_n_gt_050", f"{prefix}_n_gt_020", f"{prefix}_n_gt_010", f"{prefix}_n_gt_005",
    ]
    return np.concatenate(feats, axis=1).astype(np.float32), names


def auc_safe(y: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, score))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.output_dir is not None:
        cfg.output_dir = str(args.output_dir)
    if args.experiment_id:
        cfg.experiment_id = args.experiment_id
    if not cfg.feature_npz:
        cfg.feature_npz = [
            "artifacts/sed_soundscape_packaging_audit/20260528T1424Z_soft1279init_native_allcls_package/train_soundscapes_soft1279init_native_allcls.npz",
            "artifacts/sed_soundscape_packaging_audit/20260528T1628Z_soft1279enc_native_allcls_package/train_soundscapes_soft1279enc_native_allcls.npz",
            "artifacts/sed_soundscape_packaging_audit/20260528T1819Z_soft1279init_obspos_native_allcls_package/train_soundscapes_soft1279init_obspos_native_allcls.npz",
        ]
    if not cfg.feature_names:
        cfg.feature_names = [f"member{i}" for i in range(len(cfg.feature_npz))]
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    data_root = Path(cfg.data_root)
    pos_windows, label_profile = load_positive_windows(data_root)

    feature_blocks: list[np.ndarray] = []
    feature_cols: list[str] = []
    row_ids_ref: np.ndarray | None = None
    labels_ref: list[str] | None = None
    raw_members: dict[str, np.ndarray] = {}
    for path_s, name in zip(cfg.feature_npz, cfg.feature_names):
        row_ids, probs, labels = load_feature_npz(Path(path_s))
        if row_ids_ref is None:
            row_ids_ref = row_ids
            labels_ref = labels
        else:
            if labels != labels_ref:
                raise ValueError(f"labels do not align for {path_s}")
            # Package inference scripts may emit the same 792 file windows in a
            # different file order.  Align by stable row_id before concatenating
            # features; missing/extra rows are a hard failure.
            pos = {rid: i for i, rid in enumerate(row_ids.tolist())}
            missing = [rid for rid in row_ids_ref.tolist() if rid not in pos]
            extra = [rid for rid in row_ids.tolist() if rid not in set(row_ids_ref.tolist())]
            if missing or extra:
                raise ValueError(f"row_ids set mismatch for {path_s}: missing={len(missing)} extra={len(extra)}")
            probs = probs[[pos[rid] for rid in row_ids_ref.tolist()]]
        block, cols = aggregate_features(probs, name, include_class_probs=bool(cfg.include_class_probs))
        feature_blocks.append(block)
        feature_cols.extend(cols)
        raw_members[name] = probs

    assert row_ids_ref is not None and labels_ref is not None
    positives_by_file: dict[str, list[int]] = {}
    for (filename, start_s) in pos_windows.keys():
        positives_by_file.setdefault(filename, []).append(int(start_s))

    meta_rows: list[dict[str, Any]] = []
    y_any = np.zeros(len(row_ids_ref), dtype=np.int64)
    labels_joined: list[str] = []
    for i, row_id in enumerate(row_ids_ref):
        filename, start_s, end_s, site = parse_row_id(str(row_id))
        labs = sorted(pos_windows.get((filename, start_s), set()))
        y_any[i] = 1 if labs else 0
        labels_joined.append(";".join(labs))
        nearest_pos_distance_sec = min((abs(int(start_s) - ps) for ps in positives_by_file.get(filename, [])), default=999999)
        meta_rows.append({
            "row_id": str(row_id),
            "filename": filename,
            "start_sec": int(start_s),
            "end_sec": int(end_s),
            "site": site,
            "any_call_label": int(y_any[i]),
            "labels": ";".join(labs),
            "nearest_positive_distance_sec": int(nearest_pos_distance_sec),
        })
    meta = pd.DataFrame(meta_rows)

    X = np.concatenate(feature_blocks, axis=1).astype(np.float32)
    original_rows = int(len(meta))
    original_negative_rows = int((1 - y_any).sum())
    if int(cfg.negative_min_distance_sec) > 0:
        keep_mask = (y_any == 1) | ((y_any == 0) & (meta["nearest_positive_distance_sec"].to_numpy() > int(cfg.negative_min_distance_sec)))
        X = X[keep_mask]
        y_any = y_any[keep_mask]
        meta = meta.loc[keep_mask].reset_index(drop=True)
        raw_members = {name: probs[keep_mask] for name, probs in raw_members.items()}
    sites = sorted(meta["site"].unique())
    oof = np.full(len(meta), np.nan, dtype=np.float32)
    site_metrics: list[dict[str, Any]] = []
    for site in sites:
        val = (meta["site"].values == site)
        train = ~val
        # If train or val is single-class, still fit only when train has both classes;
        # val AUC will be marked invalid when applicable.
        if len(np.unique(y_any[train])) < 2:
            site_metrics.append({"site": site, "n_val": int(val.sum()), "valid_auc": False, "auc": None, "reason": "single-class train"})
            continue
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(cfg.c_value),
                max_iter=int(cfg.max_iter),
                class_weight="balanced",
                random_state=int(cfg.seed),
                solver="liblinear",
            ),
        )
        clf.fit(X[train], y_any[train])
        score = clf.predict_proba(X[val])[:, 1]
        oof[val] = score.astype(np.float32)
        auc = auc_safe(y_any[val], score)
        site_metrics.append({
            "site": site,
            "n_val": int(val.sum()),
            "positives": int(y_any[val].sum()),
            "negatives": int((1 - y_any[val]).sum()),
            "valid_auc": auc is not None,
            "auc": auc,
            "mean_pred_pos": float(score[y_any[val] == 1].mean()) if (y_any[val] == 1).any() else None,
            "mean_pred_neg": float(score[y_any[val] == 0].mean()) if (y_any[val] == 0).any() else None,
        })

    valid_oof = np.isfinite(oof)
    primary_auc = auc_safe(y_any[valid_oof], oof[valid_oof])
    # Baselines from raw package confidence summaries.
    baseline_metrics: dict[str, Any] = {}
    for name, probs in raw_members.items():
        maxp = probs.max(axis=1)
        top5 = np.sort(probs, axis=1)[:, -5:].mean(axis=1)
        sump = probs.sum(axis=1)
        baseline_metrics[f"{name}_max_auc"] = auc_safe(y_any, maxp)
        baseline_metrics[f"{name}_top5_auc"] = auc_safe(y_any, top5)
        baseline_metrics[f"{name}_sum_auc"] = auc_safe(y_any, sump)
    best_baseline_name = None
    best_baseline_auc = None
    for k, v in baseline_metrics.items():
        if v is not None and (best_baseline_auc is None or float(v) > best_baseline_auc):
            best_baseline_name = k
            best_baseline_auc = float(v)

    # Final model fit on all rows for reusable comparison artifacts only.
    final_clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=float(cfg.c_value), max_iter=int(cfg.max_iter), class_weight="balanced", random_state=int(cfg.seed), solver="liblinear"),
    )
    final_clf.fit(X, y_any)
    final_pred = final_clf.predict_proba(X)[:, 1].astype(np.float32)

    pred_df = meta.copy()
    pred_df["oof_any_call_prob"] = oof
    pred_df["final_any_call_prob"] = final_pred
    pred_df["oof_no_call_prob"] = 1.0 - pred_df["oof_any_call_prob"]
    pred_df["final_no_call_prob"] = 1.0 - pred_df["final_any_call_prob"]
    pred_path = out / "nocall_gate_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    site_valid_aucs = [m["auc"] for m in site_metrics if m.get("valid_auc") and m.get("auc") is not None]
    metrics = {
        "experiment_id": cfg.experiment_id,
        "track": cfg.track,
        "config": asdict(cfg),
        "data_profile": {
            **label_profile,
            "rows_in_feature_npz": int(original_rows),
            "rows_after_negative_protocol": int(len(meta)),
            "files": int(meta["filename"].nunique()),
            "sites": int(meta["site"].nunique()),
            "positive_any_call_windows": int(y_any.sum()),
            "unlabeled_background_windows_original": int(original_negative_rows),
            "unlabeled_background_windows_used": int((1 - y_any).sum()),
            "negative_min_distance_sec": int(cfg.negative_min_distance_sec),
            "negative_label_caveat": "unlabeled train_soundscape windows are weak no-call/background labels, not hand-verified negatives; optional distance guard only removes adjacent ambiguous negatives",
            "site_counts": {k: int(v) for k, v in meta["site"].value_counts().sort_index().items()},
            "site_positive_counts": {str(s): int(y_any[meta["site"].values == s].sum()) for s in sites},
            "site_negative_counts": {str(s): int((1 - y_any[meta["site"].values == s]).sum()) for s in sites},
        },
        "primary_metric": {
            "name": "leave-site OOF any-call ROC-AUC",
            "value": primary_auc,
            "valid_rows": int(valid_oof.sum()),
            "valid_sites": int(sum(1 for m in site_metrics if m.get("valid_auc"))),
        },
        "secondary_metrics": {
            "no_call_auc": primary_auc,
            "site_auc_mean": float(np.mean(site_valid_aucs)) if site_valid_aucs else None,
            "site_auc_min": float(np.min(site_valid_aucs)) if site_valid_aucs else None,
            "site_auc_q05": float(np.quantile(site_valid_aucs, 0.05)) if site_valid_aucs else None,
            "mean_oof_pred_positive": float(np.nanmean(oof[y_any == 1])) if np.isfinite(oof[y_any == 1]).any() else None,
            "mean_oof_pred_background": float(np.nanmean(oof[y_any == 0])) if np.isfinite(oof[y_any == 0]).any() else None,
            "baseline_metrics": baseline_metrics,
            "best_baseline": best_baseline_name,
            "best_baseline_auc": best_baseline_auc,
            "delta_vs_best_baseline": (float(primary_auc) - best_baseline_auc) if primary_auc is not None and best_baseline_auc is not None else None,
        },
        "site_metrics": site_metrics,
        "artifacts": {
            "predictions_csv": str(pred_path),
            "metrics_json": str(out / "metrics.json"),
        },
        "decision": "comparison-grade no-call/background data point; do not submit without hand-verified negative audit and sidecar suppression verifier",
        "runtime_sec": float(time.time() - started),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    (out / "config.resolved.json").write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
