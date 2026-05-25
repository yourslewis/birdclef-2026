"""Guarded submitter for v616 anchored Jung21 + Samejima SED private verifier."""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import time
from typing import Callable, TypeVar

import requests
from kagglesdk.kaggle_http_client import KaggleHttpClient
from kagglesdk.competitions.services.competition_api_service import CompetitionApiClient
from kagglesdk.competitions.types.competition_api_service import ApiCreateCodeSubmissionRequest, ApiListSubmissionsRequest
from kagglesdk.kernels.services.kernels_api_service import KernelsApiClient
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest, ApiListKernelSessionOutputRequest

COMPETITION = "birdclef-2026"
OWNER = "yourslewis"
SLUG = "bc26-v616-anchored-jung21-sed-blend"
VERSION = 1
DESCRIPTION = "v616: Repo-owned Samejima anchor plus Jung21 and SED rank blend"
REQUIRED_OUTPUTS = {
    "submission.csv",
    "submission_anchor_raw.csv",
    "submission_samejima_sed_raw.csv",
    "submission_jung21_raw.csv",
    "submission_before_alignment.csv",
}
REQUIRED_LOG_MARKERS = [
    "v616 preserved submission_anchor_raw.csv",
    "v616 preserved submission_samejima_sed_raw.csv",
    "v616 wrote submission_jung21_raw.csv",
    "v616 anchored rank blend",
    "v616 wrote submission.csv",
]
T = TypeVar("T")


def call(label: str, fn: Callable[[], T], attempts: int = 3, sleep_s: int = 20) -> T:
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


def clients():
    with open(os.path.expanduser("~/.kaggle/kaggle.json")) as f:
        token = json.load(f)["key"]
    http = KaggleHttpClient(api_token=token)
    return CompetitionApiClient(http), KernelsApiClient(http)


def recent_submissions(competitions):
    req = ApiListSubmissionsRequest(); req.competition_name = COMPETITION; req.page_size = 200
    return call("list_submissions", lambda: competitions.list_submissions(req)).submissions or []


def count_today(submissions) -> int:
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    return sum(str(getattr(s, "date", getattr(s, "date_nullable", ""))).startswith(today) for s in submissions)


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
    clean = [x for x in vals if not (math.isnan(x) or math.isinf(x))]
    return {
        "rows": len(rows) - 1,
        "cols": len(rows[0]) if rows else 0,
        "unique_rows": len(set(row_ids)),
        "bad": len(vals) - len(clean),
        "min": min(clean) if clean else None,
        "max": max(clean) if clean else None,
        "unique_rounded_first_10000": len(set(round(x, 6) for x in clean[:10000])),
    }


def output_preflight(kernels) -> bool:
    sreq = ApiGetKernelSessionStatusRequest(); sreq.user_name = OWNER; sreq.kernel_slug = SLUG
    status = call("kernel_status", lambda: kernels.get_kernel_session_status(sreq))
    st = str(getattr(status, "status", status)).upper()
    failure = getattr(status, "failure_message", getattr(status, "failureMessage", ""))
    print("kernel status", status, flush=True)
    if failure or "ERROR" in st or "FAILED" in st or "COMPLETE" not in st:
        print("bad kernel status/failure", failure, flush=True)
        return False
    oreq = ApiListKernelSessionOutputRequest(); oreq.user_name = OWNER; oreq.kernel_slug = SLUG; oreq.page_size = 100
    out = call("kernel_output", lambda: kernels.list_kernel_session_output(oreq))
    files = {getattr(f, "file_name", getattr(f, "fileName", "")): f for f in (getattr(out, "files", []) or [])}
    print("output files", sorted(files), flush=True)
    missing = sorted(REQUIRED_OUTPUTS - set(files))
    if missing:
        print("missing required outputs", missing, flush=True)
        return False
    log = getattr(out, "log", "") or ""
    if "Traceback" in log or "NotebookThrewException" in log:
        print("error marker in log", flush=True)
        return False
    missing_markers = [m for m in REQUIRED_LOG_MARKERS if m not in log]
    if missing_markers:
        print("missing log markers", missing_markers, flush=True)
        return False
    stats = {}
    for name in REQUIRED_OUTPUTS:
        f = files[name]
        url = getattr(f, "url", None) or getattr(f, "_url", None)
        txt = call(f"download_{name}", lambda: requests.get(url, timeout=120).text)
        stats[name] = csv_stats(txt)
    print("csv stats", json.dumps(stats, indent=2), flush=True)
    for name, stt in stats.items():
        if stt["rows"] <= 0 or stt["cols"] != 235 or stt["bad"] or stt["unique_rows"] != stt["rows"]:
            print("schema/numeric guard failed", name, stt, flush=True)
            return False
        if stt["unique_rounded_first_10000"] <= 100:
            print("constant-ish output guard failed", name, stt, flush=True)
            return False
    return True


def submit(competitions):
    req = ApiCreateCodeSubmissionRequest()
    req.competition_name = COMPETITION
    req.kernel_owner = OWNER
    req.kernel_slug = SLUG
    req.kernel_version = VERSION
    req.file_name = "submission.csv"
    req.submission_description = DESCRIPTION
    return competitions.create_code_submission(req)


def main() -> int:
    competitions, kernels = clients()
    submissions = recent_submissions(competitions)
    today_count = count_today(submissions)
    print("visible UTC submissions today", today_count, flush=True)
    if any(str(getattr(s, "description", "")) == DESCRIPTION for s in submissions):
        print("v616 already submitted; exiting", flush=True)
        return 0
    if today_count >= 5:
        raise SystemExit("daily cap appears exhausted")
    if not output_preflight(kernels):
        raise SystemExit("output preflight failed")
    print("Submitting", DESCRIPTION, flush=True)
    print("Submission result:", submit(competitions), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
