#!/usr/bin/env python3
"""Grid-search OOF blend weights across multiple BirdCLEF prediction artifacts.

The existing two-artifact comparator is useful for quick checks, but once new
SED/zoo/pseudo-label artifacts exist we need a reusable multi-member grid.  This
script aligns OOF rows by a stable train_audio-relative key, validates labels,
then evaluates simplex weights at a configurable step size.

Example:
  python scripts/birdclef_oof_blend_grid.py \
    --member v13:artifacts/sed_oof/v13/oof_predictions.npz \
    --member v15:artifacts/sed_oof/v15/oof_predictions.npz \
    --member v29:artifacts/sed_oof/v29/oof_predictions.npz \
    --step 0.1 --top-k 20 --output artifacts/blend_grids/v13_v15_v29.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


def path_key(raw: str) -> str:
    parts = str(raw).replace("\\", "/").split("/")
    if "train_audio" in parts:
        idx = parts.index("train_audio")
        return "/".join(parts[idx + 1 : idx + 3])
    return "/".join(parts[-2:])


def parse_member(raw: str) -> tuple[str, Path]:
    if ":" not in raw:
        raise argparse.ArgumentTypeError("member must be name:path")
    name, path_s = raw.split(":", 1)
    if not name:
        raise argparse.ArgumentTypeError("member name must be non-empty")
    path = Path(path_s)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"member path does not exist: {path}")
    return name, path


def load_member(name: str, path: Path) -> dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    required = {"files", "labels"}
    missing = sorted(required - set(z.files))
    if missing:
        raise ValueError(f"{path} missing required keys: {missing}")
    if "y_oof" in z.files and "pred_oof" in z.files:
        y = z["y_oof"].astype(np.float32)
        pred = z["pred_oof"].astype(np.float32)
    elif "y_val" in z.files and "pred_val" in z.files:
        y = z["y_val"].astype(np.float32)
        pred = z["pred_val"].astype(np.float32)
    else:
        raise ValueError(f"{path} must contain y_oof/pred_oof or y_val/pred_val")
    files = np.array([path_key(x) for x in z["files"].astype(str)])
    if len(files) != len(pred):
        # holdout_predictions.npz stores all selected files plus val_indices.
        if "val_indices" in z.files and len(z["val_indices"]) == len(pred):
            files = files[z["val_indices"].astype(int)]
        else:
            raise ValueError(f"{path}: files length {len(files)} does not match predictions {len(pred)}")
    return {
        "name": name,
        "path": str(path),
        "files": files,
        "labels": z["labels"].astype(str),
        "y": y,
        "pred": pred,
    }


def macro_auc(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    aucs: list[float] = []
    for j in range(y.shape[1]):
        col = y[:, j]
        if col.min() == col.max():
            continue
        try:
            aucs.append(float(roc_auc_score(col, pred[:, j])))
        except Exception:
            pass
    return {"macro_auc": float(np.mean(aucs)) if aucs else None, "valid_classes": len(aucs)}


def align_members(members: list[dict[str, Any]]) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    labels0 = members[0]["labels"]
    for member in members[1:]:
        if not np.array_equal(labels0, member["labels"]):
            raise ValueError(f"label mismatch between {members[0]['name']} and {member['name']}")

    common = set(members[0]["files"].tolist())
    for member in members[1:]:
        common &= set(member["files"].tolist())
    if not common:
        raise RuntimeError("No overlapping OOF rows across members")
    common_sorted = np.array(sorted(common))

    aligned_y: np.ndarray | None = None
    preds: list[np.ndarray] = []
    for member in members:
        pos = {f: i for i, f in enumerate(member["files"])}
        idx = np.array([pos[f] for f in common_sorted], dtype=int)
        y = member["y"][idx]
        if aligned_y is None:
            aligned_y = y
        elif not np.array_equal(aligned_y, y):
            raise ValueError(f"aligned labels differ for member {member['name']}")
        preds.append(member["pred"][idx])
    assert aligned_y is not None
    return aligned_y, preds, common_sorted, labels0


def simplex_weights(n: int, step: float) -> list[tuple[float, ...]]:
    if n < 2:
        return [(1.0,)]
    scale = round(1.0 / step)
    if abs(scale * step - 1.0) > 1e-9:
        raise ValueError("step must evenly divide 1.0, e.g. 0.5, 0.25, 0.2, 0.1, 0.05")
    scale_i = int(scale)
    out: list[tuple[float, ...]] = []

    def rec(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            out.append(tuple(v / scale_i for v in prefix + [remaining]))
            return
        for value in range(remaining + 1):
            rec(prefix + [value], remaining - value, slots - 1)

    rec([], scale_i, n)
    return out


def flat_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--member", action="append", required=True, type=parse_member, help="name:path to OOF npz; repeat")
    p.add_argument("--step", type=float, default=0.1, help="simplex grid step; must divide 1.0")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--output", type=Path)
    args = p.parse_args()

    if len(args.member) < 2:
        raise SystemExit("Need at least two --member entries")
    members = [load_member(name, path) for name, path in args.member]
    y, preds, files, labels = align_members(members)

    singles = []
    for member, pred in zip(members, preds):
        singles.append({
            "name": member["name"],
            "path": member["path"],
            **macro_auc(y, pred),
        })

    corr = []
    for i, j in itertools.combinations(range(len(members)), 2):
        corr.append({
            "a": members[i]["name"],
            "b": members[j]["name"],
            "flat_pearson": flat_corr(preds[i], preds[j]),
            "mean_abs_diff": float(np.mean(np.abs(preds[i] - preds[j]))),
        })

    rows = []
    for weights in simplex_weights(len(members), args.step):
        blend = np.zeros_like(preds[0], dtype=np.float32)
        for weight, pred in zip(weights, preds):
            blend += float(weight) * pred
        auc = macro_auc(y, blend)
        rows.append({
            "weights": {members[i]["name"]: float(weights[i]) for i in range(len(members))},
            **auc,
        })
    rows.sort(key=lambda r: r["macro_auc"] if r["macro_auc"] is not None else -1.0, reverse=True)

    result = {
        "n_members": len(members),
        "members": [{"name": m["name"], "path": m["path"]} for m in members],
        "n_overlap": int(len(files)),
        "n_classes": int(len(labels)),
        "step": args.step,
        "single_auc": singles,
        "pairwise": corr,
        "best": rows[0],
        "top": rows[: max(args.top_k, 1)],
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
