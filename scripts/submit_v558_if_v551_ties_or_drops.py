"""Submit v558 only after v551 scores and fails to improve the public946 anchor.

Guardrails:
- exits while/if v551 is already pending unless polling in the main loop
- exits without submitting if v551 improves above 0.946
- exits if v558 was already submitted
- requires v558 COMPLETE and submission.csv present
- submits one code-competition kernel submission under daily cap
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
V551_DESC = "v551: Public946 v542 plus source-clean CLAP INT8 ONNX tiny rank sidecar 0.5%"
V551_IMPROVE_THRESHOLD = 0.946
V558_KERNEL_SLUG = "bc26-v558-gateretune-a010-clip002"
V558_KERNEL_VERSION = 1
V558_DESCRIPTION = "v558: Public946 v542 plus exact-base clipped gate retune alpha0.10 maxabs0.02"

T = TypeVar("T")


def make_clients():
    with open(os.path.expanduser("~/.kaggle/kaggle.json")) as f:
        token = json.load(f)["key"]
    http = KaggleHttpClient(api_token=token)
    return CompetitionApiClient(http), KernelsApiClient(http)


def call_with_retries(label: str, fn: Callable[[], T], attempts: int = 4, sleep_s: int = 120) -> T:
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


def list_submissions(competitions):
    req = ApiListSubmissionsRequest()
    req.competition_name = COMPETITION
    req.page_size = 200
    return call_with_retries("list_submissions", lambda: competitions.list_submissions(req)).submissions


def recent_descriptions(competitions) -> set[str]:
    return {str(s.description) for s in list_submissions(competitions)}


def find_v551(competitions):
    for sub in list_submissions(competitions):
        if str(getattr(sub, "description", "")) == V551_DESC:
            return sub
    return None


def parse_score(score) -> float | None:
    text = str(score or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def v558_status(kernels):
    req = ApiGetKernelSessionStatusRequest()
    req.user_name = KERNEL_OWNER
    req.kernel_slug = V558_KERNEL_SLUG
    return call_with_retries("kernel_status", lambda: kernels.get_kernel_session_status(req))


def v558_has_submission_csv(kernels) -> bool:
    req = ApiListKernelFilesRequest()
    req.user_name = KERNEL_OWNER
    req.kernel_slug = V558_KERNEL_SLUG
    req.page_size = 100
    files = call_with_retries("list_kernel_files", lambda: kernels.list_kernel_files(req)).files or []
    names = {getattr(f, "name", "") for f in files}
    print(f"v558 kernel files: {sorted(names)}", flush=True)
    return "submission.csv" in names


def quota_sleep_seconds(text: str) -> int:
    m = re.search(r"(\d+(?:\.\d+)?)\s+hours?\s+from now", text)
    if m:
        return max(300, int(float(m.group(1)) * 3600) + 120)
    m = re.search(r"(\d+)\s+minutes?\s+from now", text)
    if m:
        return max(300, int(m.group(1)) * 60 + 120)
    return 3600


def submit_v558(competitions):
    req = ApiCreateCodeSubmissionRequest()
    req.competition_name = COMPETITION
    req.kernel_owner = KERNEL_OWNER
    req.kernel_slug = V558_KERNEL_SLUG
    req.kernel_version = V558_KERNEL_VERSION
    req.file_name = "submission.csv"
    req.submission_description = V558_DESCRIPTION
    return competitions.create_code_submission(req)


def should_submit_v558(v551) -> bool | None:
    status_text = str(getattr(v551, "status", "")).upper()
    error_text = str(getattr(v551, "error_description", "") or "")
    score = parse_score(getattr(v551, "public_score", None))
    print(
        "v551 state:",
        json.dumps(
            {
                "date": str(getattr(v551, "date", None)),
                "status": status_text,
                "public_score": getattr(v551, "public_score", None),
                "error": error_text,
            },
            default=str,
        ),
        flush=True,
    )
    if "PENDING" in status_text or ("COMPLETE" not in status_text and not error_text):
        return None
    if score is None:
        if error_text:
            print("v551 completed with error/no score; v558 fallback is allowed.", flush=True)
            return True
        return None
    if score > V551_IMPROVE_THRESHOLD:
        print(f"v551 improved to {score}; not submitting v558.", flush=True)
        return False
    print(f"v551 score {score} did not improve above {V551_IMPROVE_THRESHOLD}; v558 fallback is allowed.", flush=True)
    return True


def main():
    interval = int(os.environ.get("V558_SUBMIT_MONITOR_INTERVAL_SEC", "600"))
    competitions, kernels = make_clients()
    while True:
        descs = recent_descriptions(competitions)
        if V558_DESCRIPTION in descs:
            print("v558 already submitted; exiting.", flush=True)
            return
        v551 = find_v551(competitions)
        if v551 is None:
            print("v551 submission not visible yet; sleeping.", flush=True)
            time.sleep(interval)
            continue
        decision = should_submit_v558(v551)
        if decision is None:
            print(f"v551 not scored yet; sleeping {interval}s.", flush=True)
            time.sleep(interval)
            continue
        if decision is False:
            return
        status = v558_status(kernels)
        status_text = str(getattr(status, "status", status)).upper()
        print(f"v558 kernel status: {status}", flush=True)
        if "ERROR" in status_text or "FAILED" in status_text:
            raise SystemExit("v558 kernel failed; not submitting.")
        if "COMPLETE" not in status_text:
            print(f"v558 not complete yet; sleeping {interval}s.", flush=True)
            time.sleep(interval)
            continue
        if not v558_has_submission_csv(kernels):
            raise SystemExit("v558 COMPLETE but submission.csv missing; not submitting.")
        try:
            print(f"Submitting v558 kernel version {V558_KERNEL_VERSION}...", flush=True)
            print("Submission result:", submit_v558(competitions), flush=True)
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
            print(f"Transient submit API error {type(exc).__name__}: {exc}; sleeping {interval}s.", flush=True)
            time.sleep(interval)


if __name__ == "__main__":
    main()
