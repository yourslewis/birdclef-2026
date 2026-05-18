"""Guarded direct replay submitter for Chaney v37 0.96-frontier candidate.

This intentionally replaces the low-upside v577/v578 EoS5 scalar lane.  It only
submits a public-code replay after source/output preflight passes and a daily
slot is available.
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
from kagglesdk.competitions.types.competition_api_service import ApiCreateCodeSubmissionRequest, ApiListSubmissionsRequest
from kagglesdk.kernels.services.kernels_api_service import KernelsApiClient
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest, ApiListKernelSessionOutputRequest

COMPETITION = "birdclef-2026"
KERNEL_OWNER = "chaneyma"
KERNEL_SLUG = "bc26-gate-v37-ninastyle-branch"
KERNEL_VERSION = 1
DESCRIPTION = "v580: Guarded direct Chaney v37 Nina-style gate frontier replay"

REQUIRED_OUTPUTS = {
    "submission.csv",
    "v37_ninastyle_branch_shared_blend_summary.json",
    "submission_imaad0946.csv",
    "submission_sed.csv",
}
REQUIRED_SOURCE_MARKERS = [
    "v37 final: reproduce Nina Model_4 final blend logic",
    "Default submission.csv =",
    "test_soundscapes",
    "IS_DRY_RUN",
    "sample_submission.csv",
    "row_id",
]
FORBIDDEN_SOURCE_MARKERS = [
    "submission.csv passthrough from Imaad dry-run\"",  # allowed only inside guarded fallback, not final path
]
T = TypeVar("T")


def make_clients():
    with open(os.path.expanduser("~/.kaggle/kaggle.json")) as f:
        token = json.load(f)["key"]
    http = KaggleHttpClient(api_token=token)
    return token, CompetitionApiClient(http), KernelsApiClient(http)


def call(label: str, fn: Callable[[], T], attempts: int = 4, sleep_s: int = 300) -> T:
    last: Exception | None = None
    for i in range(1, attempts + 1):
        try:
            return fn()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last = exc
            if i == attempts:
                break
            print(f"Transient {label} error {type(exc).__name__}: {exc}; sleeping {sleep_s}s", flush=True)
            time.sleep(sleep_s)
    raise RuntimeError(f"{label} failed after {attempts} attempts") from last


def recent_submissions(competitions):
    req = ApiListSubmissionsRequest()
    req.competition_name = COMPETITION
    req.page_size = 200
    return call("list_submissions", lambda: competitions.list_submissions(req)).submissions or []


def source_preflight(token: str) -> bool:
    url = f"https://www.kaggle.com/api/v1/kernels/pull/{KERNEL_OWNER}/{KERNEL_SLUG}"
    r = call("kernel_pull", lambda: requests.get(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"}, timeout=90))
    print("source pull status", r.status_code, flush=True)
    if r.status_code != 200:
        print(r.text[:1000], flush=True)
        return False
    data = r.json()
    meta = data.get("metadata", {})
    version = meta.get("currentVersionNumberNullable") or meta.get("currentVersionNumber")
    if int(version) != KERNEL_VERSION:
        print(f"Version drift: expected {KERNEL_VERSION}, got {version}", flush=True)
        return False
    blob = data.get("blob") or {}
    source = blob.get("source") or data.get("source") or data.get("text") or ""
    if not source:
        for value in blob.values():
            if isinstance(value, str) and len(value) > 1000:
                source = value
                break
    missing = [m for m in REQUIRED_SOURCE_MARKERS if m not in source]
    if missing:
        print("Missing source markers:", missing, flush=True)
        return False
    # The dry-run fallback marker exists in source but is guarded by overlap==0 and SystemExit.
    # Make that guard explicit so future edits do not silently submit a dry-run passthrough.
    guarded = "dryrun_fallback" in source and "raise SystemExit(0)" in source
    if not guarded:
        print("Dry-run fallback guard not found", flush=True)
        return False
    print("source preflight ok", {"version": version, "source_len": len(source)}, flush=True)
    return True


def kernel_status(kernels):
    req = ApiGetKernelSessionStatusRequest()
    req.user_name = KERNEL_OWNER
    req.kernel_slug = KERNEL_SLUG
    return call("kernel_status", lambda: kernels.get_kernel_session_status(req))


def output_preflight(kernels) -> bool:
    req = ApiListKernelSessionOutputRequest()
    req.user_name = KERNEL_OWNER
    req.kernel_slug = KERNEL_SLUG
    req.page_size = 100
    out = call("kernel_output", lambda: kernels.list_kernel_session_output(req))
    files = {getattr(f, "file_name", getattr(f, "fileName", "")) for f in (getattr(out, "files", []) or [])}
    missing = sorted(REQUIRED_OUTPUTS - files)
    print("output files", sorted(files), flush=True)
    if missing:
        print("Missing required output files:", missing, flush=True)
        return False
    os.makedirs("logs", exist_ok=True)
    with open("logs/v580_chaney_v37_output_preflight.txt", "w") as f:
        f.write("files=" + repr(sorted(files)) + "\n")
        f.write("missing=" + repr(missing) + "\n")
    return True


def quota_sleep_seconds(text: str) -> int:
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
    token, competitions, kernels = make_clients()
    while True:
        submissions = recent_submissions(competitions)
        if any(str(s.description) == DESCRIPTION for s in submissions):
            print("v580 already submitted; exiting.", flush=True)
            return
        # Explicitly avoid duplicate stale scalar submission descriptions.
        if any("v577: Repo-owned EoS5 Model5-only rank-aware power 0.55" in str(s.description) for s in submissions):
            print("Note: stale v577 scalar diagnostic is already visible; continuing v580 only if not duplicate.", flush=True)
        if not source_preflight(token):
            raise SystemExit("source preflight failed; not submitting")
        status = kernel_status(kernels)
        status_text = str(getattr(status, "status", status)).upper()
        failure = getattr(status, "failure_message", getattr(status, "failureMessage", ""))
        print(f"kernel status: {status}", flush=True)
        if failure or "ERROR" in status_text or "FAILED" in status_text:
            raise SystemExit(f"kernel failed; not submitting. failure={failure!r}")
        if "COMPLETE" not in status_text:
            print("kernel not complete yet; sleeping 10 minutes", flush=True)
            time.sleep(600)
            continue
        if not output_preflight(kernels):
            raise SystemExit("output preflight failed; not submitting")
        try:
            print(f"Submitting {KERNEL_OWNER}/{KERNEL_SLUG} v{KERNEL_VERSION} as {DESCRIPTION}", flush=True)
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
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            print(f"Transient submit API error {type(exc).__name__}: {exc}; sleeping 10 minutes", flush=True)
            time.sleep(600)


if __name__ == "__main__":
    main()
