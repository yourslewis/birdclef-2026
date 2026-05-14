#!/usr/bin/env python3
"""Monitor v550 Snowflake SED dry-run and gate sidecar before submission.

This script intentionally does **not** submit to BirdCLEF.  It waits for the
private Kaggle kernel to finish, downloads its output CSVs, and runs the local
public946 sidecar rank-blend gate so the next UTC submission slot is spent only
if the sidecar looks sane.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from kagglesdk.kaggle_http_client import KaggleHttpClient
from kagglesdk.kernels.services.kernels_api_service import KernelsApiClient
from kagglesdk.kernels.types.kernels_api_service import (
    ApiDownloadKernelOutputRequest,
    ApiGetKernelSessionStatusRequest,
    ApiListKernelFilesRequest,
)

OWNER = "yourslewis"
SLUG = "bc26-v550-public946-snowflake-sed-w001"
VERSION = 1
REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "artifacts" / "kaggle_outputs" / "v550-public946-snowflake-sed-w001"
BASE_CSV = Path("/Users/yourslewis/Documents/birdclef-2026/artifacts/kaggle_outputs/v542-afr1ste-updated-public946/submission.csv")
LABELS_CSV = Path("/Volumes/ExternalSSD/data/workspace_don/kaggle_birdclef2026/data/train_soundscapes_labels.csv")
GRID_DIR = REPO / "artifacts" / "blend_grids"
FILES_TO_DOWNLOAD = (
    "submission.csv",
    "submission_snowflake_sed.csv",
    "submission_sed.csv",
    "submission_protossm.csv",
)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def make_client() -> KernelsApiClient:
    with open(os.path.expanduser("~/.kaggle/kaggle.json")) as f:
        token = json.load(f)["key"]
    return KernelsApiClient(KaggleHttpClient(api_token=token))


def kernel_status(client: KernelsApiClient) -> str:
    req = ApiGetKernelSessionStatusRequest()
    req.user_name = OWNER
    req.kernel_slug = SLUG
    status = client.get_kernel_session_status(req).to_dict()
    log(f"status={status}")
    failure = status.get("failureMessage")
    if failure:
        raise RuntimeError(f"kernel failureMessage: {failure}")
    return str(status.get("status") or "UNKNOWN").upper()


def list_files(client: KernelsApiClient) -> list[str]:
    req = ApiListKernelFilesRequest()
    req.user_name = OWNER
    req.kernel_slug = SLUG
    req.page_size = 100
    resp = client.list_kernel_files(req)
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
    resp = client.download_kernel_output(req)
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
    if not BASE_CSV.exists():
        raise FileNotFoundError(f"base v542 CSV missing: {BASE_CSV}")
    sidecar_csv = OUT_DIR / "submission_snowflake_sed.csv"
    if not sidecar_csv.exists():
        raise FileNotFoundError(f"sidecar CSV missing: {sidecar_csv}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_json = GRID_DIR / f"v550_snowflake_sidecar_weight_grid_{stamp}.json"
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "birdclef_public946_sidecar_weight_grid.py"),
        "--base-csv",
        str(BASE_CSV),
        "--sidecar-csv",
        str(sidecar_csv),
        "--weights",
        "0,0.0025,0.005,0.01,0.02,0.03,0.05",
        "--output-json",
        str(out_json),
    ]
    if LABELS_CSV.exists():
        cmd.extend(["--labels-csv", str(LABELS_CSV)])
    log("running gate: " + " ".join(cmd))
    subprocess.run(cmd, cwd=REPO, check=True)
    log(f"gate written: {out_json}")
    return out_json


def main() -> None:
    interval = int(os.environ.get("V550_MONITOR_INTERVAL_SEC", "600"))
    max_polls = int(os.environ.get("V550_MONITOR_MAX_POLLS", "144"))
    client = make_client()
    log(f"monitoring {OWNER}/{SLUG} v{VERSION}; submit action is disabled")
    for poll in range(max_polls):
        status = kernel_status(client)
        if status in {"COMPLETE", "SUCCEEDED", "SUCCESS"}:
            files = list_files(client)
            missing = [f for f in FILES_TO_DOWNLOAD if f not in files]
            if missing:
                raise RuntimeError(f"kernel complete but expected files missing: {missing}")
            for file_path in FILES_TO_DOWNLOAD:
                download_file(client, file_path)
            rows, cols = validate_csv(OUT_DIR / "submission.csv")
            validate_csv(OUT_DIR / "submission_snowflake_sed.csv", rows, cols)
            validate_csv(OUT_DIR / "submission_sed.csv", rows, cols)
            validate_csv(OUT_DIR / "submission_protossm.csv", rows, cols)
            run_gate()
            log("v550 dry-run gate complete; no competition submission made")
            return
        if status in {"ERROR", "FAILED", "CANCELED", "CANCELLED"}:
            raise RuntimeError(f"kernel ended unsuccessfully: {status}")
        log(f"kernel still {status}; sleeping {interval}s")
        time.sleep(interval)
    raise TimeoutError(f"max polls exceeded waiting for {SLUG}")


if __name__ == "__main__":
    main()
