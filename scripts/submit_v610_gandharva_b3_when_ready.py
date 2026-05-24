"""Guarded submitter for v610 Gandharva EfficientNet-B3 checkpoint inference verifier."""
from __future__ import annotations

import csv
import json
import math
import os
import re
import time
from typing import Callable, TypeVar

import requests
from kagglesdk.kaggle_http_client import KaggleHttpClient
from kagglesdk.competitions.services.competition_api_service import CompetitionApiClient
from kagglesdk.competitions.types.competition_api_service import ApiCreateCodeSubmissionRequest, ApiListSubmissionsRequest
from kagglesdk.kernels.services.kernels_api_service import KernelsApiClient
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest, ApiListKernelSessionOutputRequest

COMPETITION = "birdclef-2026"
KERNEL_OWNER = "yourslewis"
KERNEL_SLUG = "bc26-v610-gandharva-b3-checkpoint-inference"
KERNEL_VERSION = 1
DESCRIPTION = "v610: Repo-owned Gandharva B3 checkpoint inference"
TODAY_UTC = "2026-05-23"
REQUIRED_OUTPUTS = {"submission.csv", "submission_gandharva_b3_raw.csv"}
T = TypeVar("T")


def make_clients():
    with open(os.path.expanduser("~/.kaggle/kaggle.json")) as f:
        token = json.load(f)["key"]
    http = KaggleHttpClient(api_token=token)
    return CompetitionApiClient(http), KernelsApiClient(http)


def call(label: str, fn: Callable[[], T], attempts: int = 4, sleep_s: int = 120) -> T:
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last = exc
            if attempt == attempts:
                break
            print(f"Transient {label} error {type(exc).__name__}: {exc}; sleeping {sleep_s}s", flush=True)
            time.sleep(sleep_s)
    raise RuntimeError(f"{label} failed after {attempts} attempts") from last


def recent_submissions(competitions):
    req = ApiListSubmissionsRequest()
    req.competition_name = COMPETITION
    req.page_size = 200
    return call("list_submissions", lambda: competitions.list_submissions(req)).submissions or []


def csv_stats(text: str):
    rows = list(csv.reader(text.splitlines()))
    vals = []
    row_ids = []
    for row in rows[1:]:
        if row:
            row_ids.append(row[0])
        for x in row[1:]:
            try:
                vals.append(float(x))
            except Exception:
                vals.append(float("nan"))
    bad = sum(1 for x in vals if math.isnan(x) or math.isinf(x))
    clean = [x for x in vals if not (math.isnan(x) or math.isinf(x))]
    return {
        "n_rows": len(rows) - 1,
        "n_cols": len(rows[0]) if rows else 0,
        "unique_row_ids": len(set(row_ids)),
        "bad_values": bad,
        "min": min(clean) if clean else None,
        "max": max(clean) if clean else None,
        "mean": sum(clean) / len(clean) if clean else None,
        "unique_rounded_first_10000": len(set(round(x, 6) for x in clean[:10000])),
    }


def output_preflight(kernels) -> bool:
    sreq = ApiGetKernelSessionStatusRequest()
    sreq.user_name = KERNEL_OWNER
    sreq.kernel_slug = KERNEL_SLUG
    status = call("kernel_status", lambda: kernels.get_kernel_session_status(sreq))
    status_text = str(getattr(status, "status", status)).upper()
    failure = getattr(status, "failure_message", getattr(status, "failureMessage", ""))
    print("kernel status", status, flush=True)
    if failure or "ERROR" in status_text or "FAILED" in status_text or "COMPLETE" not in status_text:
        print("kernel not complete/safe", failure, flush=True)
        return False

    req = ApiListKernelSessionOutputRequest()
    req.user_name = KERNEL_OWNER
    req.kernel_slug = KERNEL_SLUG
    req.page_size = 100
    out = call("kernel_output", lambda: kernels.list_kernel_session_output(req))
    files = {getattr(f, "file_name", getattr(f, "fileName", "")): f for f in (getattr(out, "files", []) or [])}
    print("output files", sorted(files), flush=True)
    missing = sorted(REQUIRED_OUTPUTS - set(files))
    if missing:
        print("missing outputs", missing, flush=True)
        return False
    log = getattr(out, "log", "") or ""
    if "Traceback" in log or "NotebookThrewException" in log:
        print("traceback/error marker in log", flush=True)
        return False
    for marker in ["loaded fold0", "submission.csv", "uniq6"]:
        if marker not in log:
            print("missing log marker", marker, flush=True)
            return False

    stats = {}
    for name in ["submission.csv", "submission_gandharva_b3_raw.csv"]:
        f = files[name]
        url = getattr(f, "url", None) or getattr(f, "_url", None)
        rr = call(f"download_{name}", lambda: requests.get(url, timeout=120))
        rr.raise_for_status()
        stats[name] = csv_stats(rr.text)
    print("csv stats", json.dumps(stats, indent=2), flush=True)
    final = stats["submission.csv"]
    raw = stats["submission_gandharva_b3_raw.csv"]
    if final["n_cols"] != 235 or final["bad_values"] or final["unique_row_ids"] != final["n_rows"]:
        return False
    if final["unique_rounded_first_10000"] <= 100:
        return False
    if raw["n_cols"] != 235 or raw["bad_values"] or raw["unique_row_ids"] != raw["n_rows"]:
        return False
    if raw["unique_rounded_first_10000"] <= 1000:
        return False
    os.makedirs("logs", exist_ok=True)
    with open("logs/v610_gandharva_b3_preflight.txt", "w") as f:
        f.write("stats=" + repr(stats) + "\n")
        f.write("log_tail=" + log[-8000:] + "\n")
    return True


def submit(competitions):
    req = ApiCreateCodeSubmissionRequest()
    req.competition_name = COMPETITION
    req.kernel_owner = KERNEL_OWNER
    req.kernel_slug = KERNEL_SLUG
    req.kernel_version = KERNEL_VERSION
    req.file_name = "submission.csv"
    req.submission_description = DESCRIPTION
    return competitions.create_code_submission(req)


def main():
    competitions, kernels = make_clients()
    submissions = recent_submissions(competitions)
    today_count = sum(1 for s in submissions if str(getattr(s, "date", "")).startswith(TODAY_UTC))
    print(f"visible UTC submissions today: {today_count}", flush=True)
    if any(str(s.description) == DESCRIPTION for s in submissions):
        print("v610 already submitted; exiting", flush=True)
        return
    if today_count >= 5:
        raise SystemExit("daily cap appears exhausted; not submitting")
    if not output_preflight(kernels):
        raise SystemExit("output preflight failed; not submitting")
    print(f"Submitting {KERNEL_OWNER}/{KERNEL_SLUG} v{KERNEL_VERSION} as {DESCRIPTION}", flush=True)
    print("Submission result:", submit(competitions), flush=True)


if __name__ == "__main__":
    main()
