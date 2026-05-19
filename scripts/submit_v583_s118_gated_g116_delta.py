"""Submit guarded direct replay for Itshyao S118 gated G116 delta launcher.

Chosen only after v582 scored below the 0.949 frontier. This is a direct
public-source replay: the visible notebook is a small launcher that executes an
attached `s118_source.ipynb`, so preflight verifies the launcher semantics plus
concrete output files and schema, not repo-owned portability.
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
KERNEL_OWNER = "itshyao"
KERNEL_SLUG = "birdclef-2026-s118-gated-g116-delta-launcher"
KERNEL_VERSION = 2
DESCRIPTION = "v583: Guarded direct S118 gated G116 delta launcher replay"

REQUIRED_SOURCE_MARKERS = [
    "s118_source.ipynb",
    "Executing source notebook",
    "run source cell",
    "S118 launcher complete",
]
REQUIRED_OUTPUTS = {
    "submission.csv",
    "submission_g116_hgnet_b1_all5_s118.csv",
    "subm_5.csv",
    "submission_protossm.csv",
    "submission_sed.csv",
    "v17_logs.json",
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
    source = (data.get("blob") or {}).get("source") or data.get("source") or data.get("text") or ""
    if not source:
        for value in (data.get("blob") or {}).values():
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
    with open("logs/v583_s118_gated_g116_output_preflight.txt", "w") as f:
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
    submissions = recent_submissions(competitions)
    if any(str(s.description) == DESCRIPTION for s in submissions):
        print("v583 already submitted; exiting", flush=True)
        return
    if not source_preflight(token):
        raise SystemExit("source preflight failed; not submitting")
    status = kernel_status(kernels)
    status_text = str(getattr(status, "status", status)).upper()
    failure = getattr(status, "failure_message", getattr(status, "failureMessage", ""))
    print(f"kernel status: {status}", flush=True)
    if failure or "ERROR" in status_text or "FAILED" in status_text:
        raise SystemExit(f"kernel failed; not submitting. failure={failure!r}")
    if "COMPLETE" not in status_text:
        raise SystemExit(f"kernel not complete; not submitting. status={status}")
    if not output_preflight(kernels):
        raise SystemExit("output preflight failed; not submitting")
    try:
        print(f"Submitting {KERNEL_OWNER}/{KERNEL_SLUG} v{KERNEL_VERSION} as {DESCRIPTION}", flush=True)
        print("Submission result:", submit(competitions), flush=True)
    except requests.exceptions.HTTPError as exc:
        response = getattr(exc, "response", None)
        text = getattr(response, "text", "") if response is not None else str(exc)
        print(f"Submission attempt failed: {type(exc).__name__}: {exc}", flush=True)
        if text:
            print(text[:2000], flush=True)
        if "daily Submission allowance" in text or ("daily" in text.lower() and "allowance" in text.lower()):
            sleep_s = quota_sleep_seconds(text)
            raise SystemExit(f"Daily cap exhausted; retry after about {sleep_s}s")
        raise


if __name__ == "__main__":
    main()
