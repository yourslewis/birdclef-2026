#!/usr/bin/env python3
"""Download and audit selected public-source BirdCLEF 2026 winner session outputs.

Uses Kaggle Bearer SDK session-output listing, not legacy /kernels/output URLs.
Writes compact JSON summaries and local CSV/JSON copies for verifier/source audits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from kagglesdk.kaggle_http_client import KaggleHttpClient
from kagglesdk.kernels.services.kernels_api_service import KernelsApiClient
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

REFS = {
    "v644_yaroslav_0950": ("yaroslavkholmirzayev", "0950-replay"),
    "v647_ryuto_eos8_sidecar": ("ryutoyoda", "birdclef-2026-exp013-eos8-sidecar"),
}


def file_name(file_obj: Any) -> str:
    return getattr(file_obj, "file_name", getattr(file_obj, "fileName", ""))


def file_url(file_obj: Any) -> str:
    return getattr(file_obj, "url", "")


def csv_summary(path: Path) -> dict[str, Any]:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return {"status": "read_error", "error": repr(exc), "path": str(path)}
    out: dict[str, Any] = {"status": "ok", "path": str(path), "shape": list(df.shape)}
    if "row_id" in df.columns:
        out["unique_row_ids"] = int(df["row_id"].nunique())
        out["head_row_ids"] = df["row_id"].head(3).astype(str).tolist()
        vals = df.drop(columns=["row_id"])
    else:
        vals = df
    num = vals.apply(pd.to_numeric, errors="coerce")
    arr = num.to_numpy(dtype=np.float64, copy=True)
    finite = np.isfinite(arr)
    out.update({
        "numeric_cells": int(arr.size),
        "finite_cells": int(finite.sum()),
        "nonfinite_cells": int(arr.size - finite.sum()),
        "min": float(np.nanmin(arr)) if finite.any() else None,
        "max": float(np.nanmax(arr)) if finite.any() else None,
        "mean": float(np.nanmean(arr)) if finite.any() else None,
        "std": float(np.nanstd(arr)) if finite.any() else None,
        "sha256_16": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
    })
    if finite.any():
        flat = arr[finite]
        out["unique_rounded_8"] = int(len(set(np.round(flat[: min(len(flat), 10000)], 8))))
    return out


def compare_csv(a: Path, b: Path) -> dict[str, Any]:
    da = pd.read_csv(a); db = pd.read_csv(b)
    key_ok = "row_id" in da.columns and "row_id" in db.columns and da["row_id"].astype(str).tolist() == db["row_id"].astype(str).tolist()
    common = [c for c in da.columns if c != "row_id" and c in db.columns]
    xa = da[common].to_numpy(np.float64); xb = db[common].to_numpy(np.float64)
    diff = xb - xa
    return {
        "a": str(a), "b": str(b), "row_order_equal": bool(key_ok), "n_common_cols": len(common),
        "max_abs_delta": float(np.max(np.abs(diff))), "mean_abs_delta": float(np.mean(np.abs(diff))),
        "changed_cells_gt_1e_9": int(np.sum(np.abs(diff) > 1e-9)),
        "corr_flat": float(np.corrcoef(xa.ravel(), xb.ravel())[0,1]) if xa.size and np.std(xa) and np.std(xb) else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("artifacts/source_winner_audit_20260531T0416Z/session_outputs"))
    args = ap.parse_args()
    token = json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))["key"]
    kernels = KernelsApiClient(KaggleHttpClient(api_token=token))
    args.out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"refs": {}, "comparisons": []}
    local_submission_paths: dict[str, Path] = {}
    for label, (owner, slug) in REFS.items():
        req = ApiListKernelSessionOutputRequest(); req.user_name = owner; req.kernel_slug = slug; req.page_size = 100
        resp = kernels.list_kernel_session_output(req)
        files = list(getattr(resp, "files", []) or [])
        ref_dir = args.out / label
        ref_dir.mkdir(parents=True, exist_ok=True)
        item: dict[str, Any] = {"owner": owner, "slug": slug, "file_names": [], "downloaded": {}, "csv_summaries": {}}
        for f in files:
            name = file_name(f)
            item["file_names"].append(name)
            if not (name.endswith(".csv") or name.endswith(".json")):
                continue
            url = file_url(f)
            if not url:
                continue
            dest = ref_dir / name.replace("/", "__")
            r = requests.get(url, timeout=180)
            r.raise_for_status()
            dest.write_bytes(r.content)
            item["downloaded"][name] = {"path": str(dest), "bytes": len(r.content), "sha256_16": hashlib.sha256(r.content).hexdigest()[:16]}
            if name.endswith(".csv"):
                item["csv_summaries"][name] = csv_summary(dest)
                if name == "submission.csv":
                    local_submission_paths[label] = dest
        # within-ref deltas from pre-sidecar anchors to final
        final = ref_dir / "submission.csv"
        if final.exists():
            for before in [p for p in ref_dir.glob("submission_before*.csv") if p.exists()]:
                try:
                    item.setdefault("internal_deltas", []).append(compare_csv(before, final))
                except Exception as exc:
                    item.setdefault("internal_deltas", []).append({"a": str(before), "b": str(final), "error": repr(exc)})
        report["refs"][label] = item
    labels = list(local_submission_paths)
    for i, a in enumerate(labels):
        for b in labels[i+1:]:
            report["comparisons"].append(compare_csv(local_submission_paths[a], local_submission_paths[b]) | {"label_a": a, "label_b": b})
    (args.out / "audit_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    compact = {
        "out": str(args.out),
        "refs": {k: {"n_files": len(v["file_names"]), "downloaded": list(v["downloaded"].keys()), "submission": v["csv_summaries"].get("submission.csv"), "internal_deltas": v.get("internal_deltas", [])} for k, v in report["refs"].items()},
        "comparisons": report["comparisons"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
