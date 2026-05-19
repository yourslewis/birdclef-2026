"""Guarded fallback submitter for Lucataco A2Prime/NFNet public-source replay.

Waits for v580 Chaney v37 to score first. If v580 improves beyond the current
0.949 best, this exits so the next action can be repo-owned confirmation rather
than spending another exploratory slot. If v580 ties/drops/no-scores, this can
use a later slot for a distinct A2Prime/NFNet frontier hypothesis.
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
KERNEL_OWNER = "lucataco"
KERNEL_SLUG = "bc26-claude-a2prime-nfnet-fix"
KERNEL_VERSION = 2
DESCRIPTION = "v581: Guarded direct Lucataco A2Prime NFNet frontier replay"
V580_DESCRIPTION = "v580: Guarded direct Chaney v37 Nina-style gate frontier replay"
CURRENT_BEST = 0.949

REQUIRED_SOURCE_MARKERS = [
    # Pull returns notebook JSON, so exact quote escaping may vary. Keep these
    # markers semantic and verify the concrete output CSV via output_preflight.
    "default_name",
    "a2_nfnet_w03",
    "A2NF blend complete",
    "Final submission diagnostics",
    "test_soundscapes",
    "IS_DRY_RUN",
    "row_id",
]
REQUIRED_OUTPUTS = {
    "submission.csv",
    "submission_a2_nfnet_w03.csv",
    "a2nfnet_blend_summary.csv",
    "nfnet_branch_summary.csv",
    "nfnet_sanity_file_summary.csv",
    "submission_nfnet.csv",
    "submission_base_3way.csv",
}
T = TypeVar("T")


def make_clients():
    with open(os.path.expanduser("~/.kaggle/kaggle.json")) as f:
        token = json.load(f)["key"]
    http = KaggleHttpClient(api_token=token)
    return token, CompetitionApiClient(http), KernelsApiClient(http)


def call(label: str, fn: Callable[[], T], attempts: int = 4, sleep_s: int = 300) -> T:
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


def score_float(sub) -> float | None:
    raw = str(getattr(sub, "public_score", getattr(sub, "publicScore", "")) or "")
    try:
        return float(raw)
    except ValueError:
        return None


def v580_allows_fallback(submissions) -> bool:
    for sub in submissions:
        if str(sub.description) == V580_DESCRIPTION:
            status = str(getattr(sub, "status", "")).lower()
            score = score_float(sub)
            print(f"v580 visible: status={status} score={score}", flush=True)
            if "complete" not in status:
                return False
            if score is not None and score > CURRENT_BEST:
                raise SystemExit(f"v580 improved to {score}; stop v581 and port/confirm v580 instead")
            return True
    print("v580 not visible yet; preserving slots", flush=True)
    return False


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
    with open("logs/v581_a2prime_nfnet_output_preflight.txt", "w") as f:
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
            print("v581 already submitted; exiting", flush=True)
            return
        if not v580_allows_fallback(submissions):
            print("waiting for v580 result before v581 fallback; sleeping 15 minutes", flush=True)
            time.sleep(900)
            continue
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
                print(f"Daily cap exhausted; sleeping {sleep_s}s before retry", flush=True)
                time.sleep(sleep_s)
                continue
            raise
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            print(f"Transient submit API error {type(exc).__name__}: {exc}; sleeping 10 minutes", flush=True)
            time.sleep(600)


if __name__ == "__main__":
    main()
