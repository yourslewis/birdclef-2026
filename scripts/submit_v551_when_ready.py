"""Poll and submit v551 public946+tiny CLAP INT8 kernel once ready.

Guardrails:
- exits if an identical submission description is already present
- requires COMPLETE kernel status and submission.csv
- backs off on daily cap and transient Kaggle API failures
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Callable, TypeVar

import requests
from kagglesdk.kaggle_http_client import KaggleHttpClient
from kagglesdk.competitions.services.competition_api_service import CompetitionApiClient
from kagglesdk.competitions.types.competition_api_service import (
    ApiCreateCodeSubmissionRequest,
    ApiListSubmissionsRequest,
)
from kagglesdk.kernels.services.kernels_api_service import KernelsApiClient
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest, ApiListKernelFilesRequest

COMPETITION = "birdclef-2026"
KERNEL_OWNER = "yourslewis"
KERNEL_SLUG = "bc26-v551-public946-clap-int8-w0005"
KERNEL_VERSION = 1
DESCRIPTION = "v551: Public946 v542 plus source-clean CLAP INT8 ONNX tiny rank sidecar 0.5%"

T = TypeVar("T")


def make_clients():
    with open(os.path.expanduser("~/.kaggle/kaggle.json")) as f:
        token = json.load(f)["key"]
    http = KaggleHttpClient(api_token=token)
    return CompetitionApiClient(http), KernelsApiClient(http)


def call_with_retries(label: str, fn: Callable[[], T], attempts: int = 4, sleep_s: int = 600) -> T:
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_exc = exc
            if attempt == attempts:
                break
            print(f"Transient {label} error {type(exc).__name__}: {exc}; sleeping {sleep_s}s", flush=True)
            time.sleep(sleep_s)
    raise RuntimeError(f"{label} failed after {attempts} attempts") from last_exc


def recent_descriptions(competitions):
    req = ApiListSubmissionsRequest()
    req.competition_name = COMPETITION
    req.page_size = 200
    return {
        str(s.description)
        for s in call_with_retries("list_submissions", lambda: competitions.list_submissions(req)).submissions
    }


def kernel_status(kernels):
    req = ApiGetKernelSessionStatusRequest()
    req.user_name = KERNEL_OWNER
    req.kernel_slug = KERNEL_SLUG
    return call_with_retries("kernel_status", lambda: kernels.get_kernel_session_status(req))


def has_submission_csv(kernels):
    req = ApiListKernelFilesRequest()
    req.user_name = KERNEL_OWNER
    req.kernel_slug = KERNEL_SLUG
    req.page_size = 100
    files = call_with_retries("list_kernel_files", lambda: kernels.list_kernel_files(req)).files or []
    names = {getattr(f, "name", "") for f in files}
    print(f"Kernel files: {sorted(names)}", flush=True)
    return "submission.csv" in names


def quota_sleep_seconds(text):
    m = re.search(r"(\d+(?:\.\d+)?)\s+hours?\s+from now", text)
    if m:
        return max(300, int(float(m.group(1)) * 3600) + 120)
    m = re.search(r"(\d+)\s+minutes?\s+from now", text)
    if m:
        return max(300, int(m.group(1)) * 60 + 120)
    return 3600


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
    while True:
        if DESCRIPTION in recent_descriptions(competitions):
            print("v551 already submitted; exiting.", flush=True)
            return
        status = kernel_status(kernels)
        status_text = str(getattr(status, "status", status)).upper()
        print(f"Kernel status: {status}", flush=True)
        if "ERROR" in status_text or "FAILED" in status_text:
            raise SystemExit("v551 kernel failed; not submitting.")
        if "COMPLETE" not in status_text:
            print("v551 not complete yet; sleeping 10 minutes.", flush=True)
            time.sleep(600)
            continue
        if not has_submission_csv(kernels):
            raise SystemExit("v551 COMPLETE but submission.csv missing; not submitting.")
        try:
            print(f"Submitting v551 kernel version {KERNEL_VERSION}...", flush=True)
            print("Submission result:", submit(competitions), flush=True)
            return
        except requests.exceptions.HTTPError as exc:
            response = getattr(exc, "response", None)
            text = getattr(response, "text", "") if response is not None else str(exc)
            print(f"Submission attempt failed: {type(exc).__name__}: {exc}", flush=True)
            if text:
                print(text[:2000], flush=True)
            if "daily Submission allowance" in text or ("daily" in text.lower() and "allowance" in text.lower()):
                sleep_s = quota_sleep_seconds(text)
                print(f"Daily cap exhausted; sleeping {sleep_s}s before retry.", flush=True)
                time.sleep(sleep_s)
                continue
            raise
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            print(f"Transient submit API error {type(exc).__name__}: {exc}; sleeping 10 minutes.", flush=True)
            time.sleep(600)


if __name__ == "__main__":
    main()
