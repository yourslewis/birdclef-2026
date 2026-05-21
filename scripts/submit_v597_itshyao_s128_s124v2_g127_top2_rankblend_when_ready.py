"""Guarded submitter for v597 Itshyao S128/S124v2/G127 top-2 rankblend.

Prepared after the 2026-05-21 08:00 UTC source scan found a fresh Itshyao
S128/S124v2/G127 notebook. This spends the remaining UTC slot only after
source/output preflight confirms the new G127 two-fold sidecar path and final
submission verifier are present.
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
KERNEL_OWNER = "itshyao"
KERNEL_SLUG = "birdclef-2026-s128-s124v2-g127-top2-rankblend"
KERNEL_VERSION = 1
DESCRIPTION = "v597: Guarded direct Itshyao S128 S124v2 plus G127 top2 rankblend"

REQUIRED_SOURCE_MARKERS = [
    "S128: S124/S125 anchor + G127 EfficientNet-B0 NS softCE pseudo top2 rank blend",
    "S128_RANK_WEIGHT = 0.075",
    "S128_ANCHOR_TOPK = 48",
    "S128_SIDE_TOPK = 30",
    "S128_MIN_TOP3_OVERLAP = 0.76",
    "S128_MIN_TOP10_OVERLAP = 0.82",
    "g127_fold1_fp16.pt",
    "g127_fold2_fp16.pt",
    "submission_g127_effb0ns_softce_top2_s128.csv",
    "S128 final submission",
    "test_soundscapes",
    "sample_submission.csv",
    "submission.csv",
    "row_id",
]
REQUIRED_OUTPUTS = {
    "submission.csv",
    "submission_g124_effv2s_fold1_s124.csv",
    "submission_g127_effb0ns_softce_top2_s128.csv",
    "submission_protossm.csv",
    "submission_sed.csv",
    "subm_karnakbayev_power_optimization.csv",
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


def count_today_utc(submissions) -> int:
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    return sum(str(getattr(s, "date", getattr(s, "date_nullable", ""))).startswith(today) for s in submissions)


def decode_notebook_source(source: str) -> str:
    try:
        nb = json.loads(source)
        if isinstance(nb, dict) and isinstance(nb.get("cells"), list):
            parts: list[str] = []
            for cell in nb.get("cells", []):
                cell_source = cell.get("source", "")
                if isinstance(cell_source, list):
                    cell_source = "".join(cell_source)
                parts.append(str(cell_source))
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
    source = (data.get("blob") or {}).get("source") or data.get("source") or data.get("text") or ""
    source_for_search = decode_notebook_source(source)
    missing = [m for m in REQUIRED_SOURCE_MARKERS if m not in source_for_search]
    if missing:
        print("Missing source markers:", missing, flush=True)
        return False
    print("source preflight ok", {"version": version, "source_len": len(source), "decoded_len": len(source_for_search)}, flush=True)
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
    }
    # Direct public kernels often expose only 3 dry-run rows publicly; hidden scoring reruns on full test.
    ok = n_rows > 0 and n_cols == 235 and bad == 0 and bool(clean) and stats["max"] > stats["min"]
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
    for f in getattr(out, "files", []) or []:
        if _file_name(f) == "submission.csv":
            sub_url = _file_url(f)
            break
    if not sub_url:
        print("No signed URL for submission.csv", flush=True)
        return False
    resp = call("download_submission", lambda: requests.get(sub_url, timeout=120))
    resp.raise_for_status()
    ok, stats = validate_submission_csv(resp.text)
    print("submission.csv stats", stats, flush=True)
    if not ok:
        return False
    log = getattr(out, "log", "") or ""
    if "S128 final submission" not in log:
        print("Missing S128 final submission log marker", flush=True)
        return False
    if "S128 G127 top2 rank sidecar failed" in log:
        print("S128 rank sidecar failed in public run; not submitting", flush=True)
        return False
    os.makedirs("logs", exist_ok=True)
    with open("logs/v597_itshyao_s128_s124v2_g127_top2_preflight.txt", "w") as f:
        f.write("files=" + repr(sorted(files)) + "\n")
        f.write("missing=" + repr(missing) + "\n")
        f.write("submission_stats=" + repr(stats) + "\n")
        f.write("log_tail=" + log[-5000:] + "\n")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true", help="Validate source/status/output and exit before submitting.")
    args = parser.parse_args()

    token, competitions, kernels = make_clients()
    submissions = recent_submissions(competitions)
    if any(str(s.description) == DESCRIPTION for s in submissions):
        print("v597 already submitted; exiting", flush=True)
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
        raise SystemExit("daily cap appears exhausted; not submitting")
    print(f"Submitting {KERNEL_OWNER}/{KERNEL_SLUG} v{KERNEL_VERSION} as {DESCRIPTION}", flush=True)
    print("Submission result:", submit(competitions), flush=True)


if __name__ == "__main__":
    main()
