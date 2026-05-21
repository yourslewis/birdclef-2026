"""Guarded direct submitter for Cheny exp070 public0952 Perch-prior probe.

This is a distinct 0.96-frontier source candidate found in the 2026-05-21 04:01Z
scan. Unlike the saturated EoS/S124 scalar/fork family, it fits classwise
logistic probes on Perch embeddings plus site/hour prior meta-features, then
applies the frozen probe to hidden test embeddings. Public dry-run completed
with a valid 240x235 submission and no NaNs.
"""
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
KERNEL_OWNER = "chenyfdws"
KERNEL_SLUG = "bc26-exp070-public0952-s124-g124-repro"
KERNEL_VERSION = 1
DESCRIPTION = "v595: Guarded direct Cheny public0952 Perch prior probe replay"

REQUIRED_SOURCE_MARKERS = [
    'MODE = "submit"',
    'require_full_cache_in_submit',
    'frozen_best_probe',
    'fit_prior_tables',
    'run_oof_embedding_probe',
    'Training final classwise probes',
    'infer_perch_with_embeddings',
    'Hidden test files',
    'Dry-run on first',
    'assert not submission.isna().any().any()',
    'submission.to_csv("submission.csv", index=False)',
    'EXP070_SAFE_ALIGN',
    'test_soundscapes',
    'sample_submission.csv',
    'row_id',
]
REQUIRED_OUTPUTS = {"submission.csv", "perch_cache/full_oof_meta_features.npz"}
T = TypeVar("T")


def make_clients():
    with open(os.path.expanduser("~/.kaggle/kaggle.json")) as f:
        token = json.load(f)["key"]
    http = KaggleHttpClient(api_token=token)
    return token, CompetitionApiClient(http), KernelsApiClient(http)


def call(label: str, fn: Callable[[], T], attempts: int = 4, sleep_s: int = 180) -> T:
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
    try:
        nb = json.loads(source)
        decoded_source = "\n".join(
            "".join(c.get("source", "") if isinstance(c.get("source", ""), list) else str(c.get("source", "")))
            for c in nb.get("cells", [])
        )
    except Exception:
        decoded_source = source
    missing = [m for m in REQUIRED_SOURCE_MARKERS if m not in decoded_source and m not in source]
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
    print("output files", sorted(files), flush=True)
    missing = sorted(REQUIRED_OUTPUTS - files)
    if missing:
        print("Missing required output files:", missing, flush=True)
        return False

    log = getattr(out, "log", "") or ""
    if "EXP070_SAFE_ALIGN: submission writer completed" not in log:
        print("Missing EXP070 final writer log marker", flush=True)
        return False

    stats = None
    for f in (getattr(out, "files", []) or []):
        name = getattr(f, "file_name", getattr(f, "fileName", ""))
        url = getattr(f, "url", None)
        if name == "submission.csv" and url:
            rr = call("download_submission", lambda: requests.get(url, timeout=120))
            rr.raise_for_status()
            rows = list(csv.reader(rr.text.splitlines()))
            vals = []
            for row in rows[1:]:
                for x in row[1:]:
                    try:
                        vals.append(float(x))
                    except Exception:
                        vals.append(float("nan"))
            bad = sum(1 for x in vals if math.isnan(x) or math.isinf(x))
            clean = [x for x in vals if not (math.isnan(x) or math.isinf(x))]
            stats = {
                "n_rows": len(rows) - 1,
                "n_cols": len(rows[0]) if rows else 0,
                "bad_values": bad,
                "min": min(clean) if clean else None,
                "max": max(clean) if clean else None,
                "zeros": sum(1 for x in clean if x == 0.0),
            }
            break
    print("submission.csv stats", stats, flush=True)
    if not stats or stats["n_rows"] != 240 or stats["n_cols"] != 235 or stats["bad_values"]:
        return False
    os.makedirs("logs", exist_ok=True)
    with open("logs/v595_cheny_public0952_output_preflight.txt", "w") as f:
        f.write("files=" + repr(sorted(files)) + "\n")
        f.write("stats=" + repr(stats) + "\n")
        f.write("log_tail=" + log[-6000:] + "\n")
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
    today_count = sum(1 for s in submissions if str(getattr(s, "date", "")).startswith("2026-05-21"))
    print(f"visible UTC submissions today: {today_count}", flush=True)
    if any(str(s.description) == DESCRIPTION for s in submissions):
        print("v595 already submitted; exiting", flush=True)
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
        raise SystemExit("kernel not complete; not submitting")
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
            print(f"Daily cap exhausted; sleep {sleep_s}s before manual/cron retry", flush=True)
        raise


if __name__ == "__main__":
    main()
