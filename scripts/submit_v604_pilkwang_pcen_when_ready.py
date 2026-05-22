"""Guarded submitter for v604 Pilkwang PCEN sidecar repo-owned verifier.

This script is intentionally conservative: it only submits after the private
verification kernel is COMPLETE and its output contains a valid, non-constant
competition-format submission.csv plus the expected PCEN sidecar diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
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
KERNEL_SLUG = "bc26-v604-pilkwang-pcen-sidecar-verify"
KERNEL_VERSION = 1
DESCRIPTION = "v604: Repo-owned Pilkwang PCEN sidecar verify"

REQUIRED_SOURCE_MARKERS = [
    "Acoustic Prior-Field Fusion + PCEN Sidecar",
    "birdclef26-sidecar-exp001",
    "RUN_EXP001_SIDECAR = True",
    "SIDECAR_EXP001_REQUIRE = True",
    "SIDECAR_EXP001_FORCE_INFER = True",
    "SIDECAR_EXP001_WEIGHT_CAP = 0.020",
    "SIDECAR_EXP001_D_BUDGET = 0.003",
    "sidecar_exp001_diagnostics.csv",
    "submission_before_exp001_sidecar.csv",
    "submission.csv",
    "sample_submission.csv",
    "test_soundscapes",
]
REQUIRED_OUTPUTS = {
    "submission.csv",
    "sidecar_exp001_diagnostics.csv",
    "submission_before_exp001_sidecar.csv",
    "submission_before_all_sidecars.csv",
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


def count_today_utc(submissions) -> int:
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    return sum(str(getattr(s, "date", getattr(s, "date_nullable", ""))).startswith(today) for s in submissions)


def source_text_from_pull(data: dict) -> str:
    source = (data.get("blob") or {}).get("source") or data.get("source") or data.get("text") or ""
    try:
        nb = json.loads(source)
        if isinstance(nb, dict) and isinstance(nb.get("cells"), list):
            parts: list[str] = []
            for cell in nb.get("cells", []):
                src = cell.get("source", "")
                if isinstance(src, list):
                    src = "".join(src)
                parts.append(str(src))
            return "\n".join(parts)
    except Exception:
        pass
    return source


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
    source = source_text_from_pull(data)
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


def _file_name(file_obj) -> str:
    return getattr(file_obj, "file_name", getattr(file_obj, "fileName", ""))


def _file_url(file_obj) -> str:
    return getattr(file_obj, "url", "")


def validate_submission_csv(text: str) -> tuple[bool, dict]:
    rows = list(csv.reader(text.splitlines()))
    if not rows:
        return False, {"error": "empty csv"}
    n_rows = len(rows) - 1
    n_cols = len(rows[0])
    vals = []
    bad = 0
    for row in rows[1:]:
        if len(row) != n_cols:
            return False, {"error": "ragged row", "n_rows": n_rows, "n_cols": n_cols}
        for x in row[1:]:
            try:
                v = float(x)
            except Exception:
                bad += 1
                continue
            if not math.isfinite(v):
                bad += 1
            vals.append(v)
    clean = [v for v in vals if math.isfinite(v)]
    stats = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "bad_values": bad,
        "min": min(clean) if clean else None,
        "max": max(clean) if clean else None,
        "zeros": sum(1 for v in clean if v == 0.0),
        "uniq100": len(set(clean[:100])) if clean else 0,
    }
    ok = n_rows > 0 and n_cols == 235 and bad == 0 and bool(clean) and stats["max"] > stats["min"] and stats["uniq100"] > 10
    return ok, stats


def output_preflight(kernels) -> bool:
    req = ApiListKernelSessionOutputRequest()
    req.user_name = KERNEL_OWNER
    req.kernel_slug = KERNEL_SLUG
    req.page_size = 100
    out = call("kernel_output", lambda: kernels.list_kernel_session_output(req))
    files = {_file_name(f) for f in (getattr(out, "files", []) or [])}
    missing = sorted(REQUIRED_OUTPUTS - files)
    print("output files", sorted(files), flush=True)
    if missing:
        print("Missing required output files:", missing, flush=True)
        return False
    sub_url = ""
    diag_url = ""
    for f in getattr(out, "files", []) or []:
        if _file_name(f) == "submission.csv":
            sub_url = _file_url(f)
        if _file_name(f) == "sidecar_exp001_diagnostics.csv":
            diag_url = _file_url(f)
    if not sub_url:
        print("No signed URL for submission.csv", flush=True)
        return False
    resp = call("download_submission", lambda: requests.get(sub_url, timeout=120))
    resp.raise_for_status()
    ok, stats = validate_submission_csv(resp.text)
    print("submission.csv stats", stats, flush=True)
    diag_head = ""
    if diag_url:
        diag_resp = call("download_diagnostics", lambda: requests.get(diag_url, timeout=120))
        diag_resp.raise_for_status()
        diag_head = diag_resp.text[:2000]
        print("diagnostics head", diag_head.replace("\n", " ")[:500], flush=True)
    log = getattr(out, "log", "") or ""
    if "Traceback" in log:
        print("Traceback marker in log; not submitting", flush=True)
        ok = False
    os.makedirs("logs", exist_ok=True)
    with open("logs/v604_pilkwang_pcen_preflight.txt", "w") as f:
        f.write("files=" + repr(sorted(files)) + "\n")
        f.write("missing=" + repr(missing) + "\n")
        f.write("submission_stats=" + repr(stats) + "\n")
        f.write("diagnostics_head=" + diag_head + "\n")
        f.write("log_tail=" + log[-5000:] + "\n")
    return ok


def seconds_until_next_utc_reset() -> int:
    now = dt.datetime.now(dt.timezone.utc)
    tomorrow = now.date() + dt.timedelta(days=1)
    reset = dt.datetime.combine(tomorrow, dt.time.min, tzinfo=dt.timezone.utc)
    return max(300, int((reset - now).total_seconds()) + 180)


def quota_sleep_seconds(text: str) -> int:
    m = re.search(r"(\d+(?:\.\d+)?)\s+hours?\s+from now", text)
    if m:
        return max(300, int(float(m.group(1)) * 3600) + 180)
    m = re.search(r"(\d+)\s+minutes?\s+from now", text)
    if m:
        return max(300, int(m.group(1)) * 60 + 180)
    return seconds_until_next_utc_reset()


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--wait-for-slot", action="store_true")
    args = parser.parse_args()

    token, competitions, kernels = make_clients()
    while True:
        submissions = recent_submissions(competitions)
        if any(str(s.description) == DESCRIPTION for s in submissions):
            print("v604 already submitted; exiting", flush=True)
            return
        today_count = count_today_utc(submissions)
        print(f"visible UTC submissions today: {today_count}", flush=True)
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
        if args.preflight_only:
            print("preflight-only requested; not submitting", flush=True)
            return
        if today_count >= 5:
            if not args.wait_for_slot:
                raise SystemExit("daily cap reached; rerun with --wait-for-slot to sleep")
            sleep_s = seconds_until_next_utc_reset()
            print(f"daily cap reached; sleeping {sleep_s}s", flush=True)
            time.sleep(sleep_s)
            continue
        try:
            resp = submit(competitions)
            print("submitted", resp, flush=True)
            return
        except Exception as exc:
            text = str(exc)
            print("submit exception", repr(text), flush=True)
            if "daily" in text.lower() or "limit" in text.lower() or "quota" in text.lower():
                if not args.wait_for_slot:
                    raise
                sleep_s = quota_sleep_seconds(text)
                print(f"quota hit; sleeping {sleep_s}s", flush=True)
                time.sleep(sleep_s)
                continue
            raise


if __name__ == "__main__":
    main()
