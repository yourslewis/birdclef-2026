#!/usr/bin/env python3
"""Monitor v554 public946 gate-retune dry-run without submitting.

Waits for the private Kaggle kernel to finish, downloads output CSVs, validates
shape/NaNs, and compares the v554 final submission against a reconstructed v542
baseline from the same ProtoSSM/SED dry-run outputs.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError:
    fallback = Path("/Users/yourslewis/.openclaw/workspace-don/kaggle/playground-series-s6e3/.venv/bin/python")
    if fallback.exists() and Path(sys.executable) != fallback:
        os.execv(str(fallback), [str(fallback), *sys.argv])
    raise

from kagglesdk.kaggle_http_client import KaggleHttpClient
from kagglesdk.kernels.services.kernels_api_service import KernelsApiClient
from kagglesdk.kernels.types.kernels_api_service import (
    ApiDownloadKernelOutputRequest,
    ApiGetKernelSessionStatusRequest,
    ApiListKernelFilesRequest,
)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from birdclef_public946_gate_autoresearch import GateConfig, _metric_setup, _score_values, apply_config  # noqa: E402

OWNER = os.environ.get("KERNEL_OWNER", "yourslewis")
SLUG = os.environ.get("KERNEL_SLUG", "bc26-v554-public946-gate-retune-pw056")
VERSION = int(os.environ.get("KERNEL_VERSION", "1"))
OUTPUT_NAME = os.environ.get("OUTPUT_NAME", "v554-public946-gate-retune-pw056")
OUT_DIR = REPO / "artifacts" / "kaggle_outputs" / OUTPUT_NAME
GRID_DIR = REPO / "artifacts" / "blend_grids"
LABELS_CSV = Path(os.environ.get("LABELS_CSV", "/Volumes/ExternalSSD/data/workspace_don/kaggle_birdclef2026/data/train_soundscapes_labels.csv"))
TAXONOMY_CSV = Path(os.environ.get("TAXONOMY_CSV", "/Volumes/ExternalSSD/data/workspace_don/kaggle_birdclef2026/data/taxonomy.csv"))
FILES_TO_DOWNLOAD = ("submission.csv", "submission_protossm.csv", "submission_sed.csv")


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def make_client() -> KernelsApiClient:
    with open(os.path.expanduser("~/.kaggle/kaggle.json")) as f:
        token = json.load(f)["key"]
    return KernelsApiClient(KaggleHttpClient(api_token=token))


def call_with_retries(label: str, fn, attempts: int = 4, sleep_s: int = 30):
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt == attempts:
                break
            log(f"{label} failed attempt {attempt}/{attempts}: {exc!r}; retrying in {sleep_s}s")
            time.sleep(sleep_s)
    raise RuntimeError(f"{label} failed after {attempts} attempts") from last_exc


def kernel_status(client: KernelsApiClient) -> str:
    req = ApiGetKernelSessionStatusRequest()
    req.user_name = OWNER
    req.kernel_slug = SLUG
    status = call_with_retries("get_kernel_session_status", lambda: client.get_kernel_session_status(req)).to_dict()
    log(f"status={status}")
    if status.get("failureMessage"):
        raise RuntimeError(f"kernel failureMessage: {status['failureMessage']}")
    return str(status.get("status") or "UNKNOWN").upper()


def list_files(client: KernelsApiClient) -> list[str]:
    req = ApiListKernelFilesRequest()
    req.user_name = OWNER
    req.kernel_slug = SLUG
    req.page_size = 100
    resp = call_with_retries("list_kernel_files", lambda: client.list_kernel_files(req))
    files = [getattr(x, "name", "") for x in (resp.files or [])]
    log(f"files={files}")
    return files


def download_file(client: KernelsApiClient, file_path: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    req = ApiDownloadKernelOutputRequest()
    req.owner_slug = OWNER
    req.kernel_slug = SLUG
    req.version_number = VERSION
    req.file_path = file_path
    resp = call_with_retries(f"download_kernel_output:{file_path}", lambda: client.download_kernel_output(req))
    resp.raise_for_status()
    out = OUT_DIR / Path(file_path).name
    out.write_bytes(resp.content)
    log(f"downloaded {file_path} -> {out} ({out.stat().st_size} bytes)")
    return out


def validate_csv(path: Path, expected_rows: int | None = None, expected_cols: int | None = None) -> tuple[int, int]:
    df = pd.read_csv(path)
    if "row_id" not in df.columns:
        raise ValueError(f"{path} missing row_id")
    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError(f"{path} row mismatch: {len(df)} != {expected_rows}")
    if expected_cols is not None and len(df.columns) != expected_cols:
        raise ValueError(f"{path} column mismatch: {len(df.columns)} != {expected_cols}")
    if df.drop(columns=["row_id"]).isna().any().any():
        raise ValueError(f"{path} contains NaNs")
    log(f"validated {path.name}: shape={df.shape}")
    return int(len(df)), int(len(df.columns))


def run_gate() -> Path:
    proto = pd.read_csv(OUT_DIR / "submission_protossm.csv")
    sed = pd.read_csv(OUT_DIR / "submission_sed.csv")
    final = pd.read_csv(OUT_DIR / "submission.csv")
    cols = [c for c in proto.columns if c != "row_id"]
    final = final.set_index("row_id").loc[proto["row_id"]].reset_index()
    tax_df = pd.read_csv(TAXONOMY_CSV).set_index("primary_label") if TAXONOMY_CSV.exists() else None
    baseline = apply_config(proto, sed, cols, tax_df, GateConfig())
    present, valid_idx, valid_cols, y_true = _metric_setup(LABELS_CSV, proto["row_id"], cols)
    base_values = baseline[cols].to_numpy(np.float32)
    final_values = final[cols].to_numpy(np.float32)
    base_metrics = _score_values(base_values, present, valid_idx, y_true)
    final_metrics = _score_values(final_values, present, valid_idx, y_true)
    result = {
        "kernel": f"{OWNER}/{SLUG}",
        "version": VERSION,
        "output_dir": str(OUT_DIR),
        "labels_csv": str(LABELS_CSV),
        "taxonomy_csv": str(TAXONOMY_CSV),
        "matched_rows": int(present.sum()),
        "valid_auc_classes": int(len(valid_cols)),
        "baseline": base_metrics,
        "candidate_final": final_metrics,
        "delta_auc_vs_baseline": float(final_metrics["macro_auc"] - base_metrics["macro_auc"]),
        "corr_vs_baseline": float(np.corrcoef(base_values.ravel(), final_values.ravel())[0, 1]),
        "mae_vs_baseline": float(np.mean(np.abs(base_values - final_values))),
        "max_abs_vs_baseline": float(np.max(np.abs(base_values - final_values))),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = OUTPUT_NAME.replace("/", "_").replace("-", "_")
    out = GRID_DIR / f"{safe_name}_final_vs_baseline_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    log(f"gate written: {out}")
    log(json.dumps(result, indent=2))
    return out


def main() -> None:
    interval = int(os.environ.get("V554_MONITOR_INTERVAL_SEC", "600"))
    max_polls = int(os.environ.get("V554_MONITOR_MAX_POLLS", "144"))
    client = make_client()
    log(f"monitoring {OWNER}/{SLUG} v{VERSION}; submit action is disabled")
    for _ in range(max_polls):
        status = kernel_status(client)
        if status in {"COMPLETE", "SUCCEEDED", "SUCCESS"}:
            files = list_files(client)
            missing = [f for f in FILES_TO_DOWNLOAD if f not in files]
            if missing:
                raise RuntimeError(f"kernel complete but expected files missing: {missing}")
            for file_path in FILES_TO_DOWNLOAD:
                download_file(client, file_path)
            rows, cols = validate_csv(OUT_DIR / "submission.csv")
            validate_csv(OUT_DIR / "submission_protossm.csv", rows, cols)
            validate_csv(OUT_DIR / "submission_sed.csv", rows, cols)
            run_gate()
            log(f"{OUTPUT_NAME} dry-run gate complete; no competition submission made")
            return
        if status in {"ERROR", "FAILED", "CANCELED", "CANCELLED"}:
            raise RuntimeError(f"kernel ended unsuccessfully: {status}")
        log(f"kernel still {status}; sleeping {interval}s")
        time.sleep(interval)
    raise TimeoutError(f"max polls exceeded waiting for {SLUG}")


if __name__ == "__main__":
    main()
