"""Guarded direct submitter for Svanikkolli PerchFusion v951 source.

Fresh 2026-05-23 18UTC candidate. It is a structurally distinct Perch/ProtoSSM/SED
source: 3x Perch audio-shift TTA, 3x SED TTA, larger in-notebook ProtoSSM,
ResidualSSM correction, raw Perch as a third rank-blend member, and conservative
safety gates. Public dry-run aligns final submission.csv to sample_submission, so
preflight validates source hidden-test handling plus intermediate 240-row ProtoSSM
and SED outputs instead of requiring public final submission.csv to be 240 rows.
"""
from __future__ import annotations

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
KERNEL_OWNER = "svanikkolli"
KERNEL_SLUG = "perchfusion-engine"
KERNEL_VERSION = 8
DESCRIPTION = "v609: Guarded direct PerchFusion v951 TTA source"
TODAY_UTC = "2026-05-23"

REQUIRED_SOURCE_MARKERS = [
    "BirdCLEF+ 2026 - v951 Target",
    "PERCH_TTA_SHIFTS_S",
    "SED_TTA_SHIFTS_S",
    "PROTO_D_MODEL",
    "BLEND_W_PERCH",
    "test_soundscapes",
    "No hidden test - dry-run",
    "submission_protossm.csv",
    "submission_sed.csv",
    "3-way rank blend",
    "Dry-run: aligning with sample_submission.csv",
    "Diagnostics OK",
    "row_id",
]
REQUIRED_OUTPUTS = {
    "submission.csv",
    "submission_protossm.csv",
    "submission_sed.csv",
    "cache/perch_arrays_tta3x.npz",
    "cache/perch_meta_tta3x.parquet",
}
T = TypeVar("T")


def make_clients():
    with open(os.path.expanduser("~/.kaggle/kaggle.json")) as f:
        token = json.load(f)["key"]
    http = KaggleHttpClient(api_token=token)
    return token, CompetitionApiClient(http), KernelsApiClient(http)


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
    req = ApiListKernelSessionOutputRequest()
    req.user_name = KERNEL_OWNER
    req.kernel_slug = KERNEL_SLUG
    req.page_size = 100
    out = call("kernel_output", lambda: kernels.list_kernel_session_output(req))
    files = {getattr(f, "file_name", getattr(f, "fileName", "")): f for f in (getattr(out, "files", []) or [])}
    print("output files", sorted(files), flush=True)
    missing = sorted(REQUIRED_OUTPUTS - set(files))
    if missing:
        print("Missing required output files:", missing, flush=True)
        return False

    log = getattr(out, "log", "") or ""
    for marker in ["Training complete", "ProtoSSM branch done", "SED branch done", "3-way rank blend", "Diagnostics OK"]:
        if marker not in log:
            print(f"Missing log marker: {marker}", flush=True)
            return False
    if "Traceback" in log or "NotebookThrewException" in log:
        print("Traceback/error marker found in log", flush=True)
        return False

    stats_by_name = {}
    for name in ["submission.csv", "submission_protossm.csv", "submission_sed.csv"]:
        f = files.get(name)
        url = getattr(f, "url", None) or getattr(f, "_url", None)
        rr = call(f"download_{name}", lambda: requests.get(url, timeout=120))
        rr.raise_for_status()
        stats_by_name[name] = csv_stats(rr.text)
    print("csv stats", json.dumps(stats_by_name, indent=2), flush=True)

    # Public final is expected to be sample-shaped because source deliberately aligns dry-runs.
    final = stats_by_name["submission.csv"]
    if final["n_cols"] != 235 or final["bad_values"] or final["unique_row_ids"] != final["n_rows"]:
        return False
    if final["unique_rounded_first_10000"] <= 10:
        print("Final dry-run output too constant; not submitting", flush=True)
        return False

    # Intermediate hidden-path branches should be full train dry-run rows and non-constant.
    for name in ["submission_protossm.csv", "submission_sed.csv"]:
        st = stats_by_name[name]
        if st["n_rows"] != 240 or st["n_cols"] != 235 or st["bad_values"]:
            print(f"Intermediate {name} failed shape/numeric guard", flush=True)
            return False
        if st["unique_rounded_first_10000"] <= 100:
            print(f"Intermediate {name} too constant", flush=True)
            return False

    os.makedirs("logs", exist_ok=True)
    with open("logs/v609_svanikkolli_perchfusion_output_preflight.txt", "w") as f:
        f.write("files=" + repr(sorted(files)) + "\n")
        f.write("stats=" + repr(stats_by_name) + "\n")
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
    token, competitions, kernels = make_clients()
    submissions = recent_submissions(competitions)
    today_count = sum(1 for s in submissions if str(getattr(s, "date", "")).startswith(TODAY_UTC))
    print(f"visible UTC submissions today: {today_count}", flush=True)
    if any(str(s.description) == DESCRIPTION for s in submissions):
        print("v609 already submitted; exiting", flush=True)
        return
    if today_count >= 5:
        raise SystemExit("daily slot cap appears exhausted; not submitting")
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
    print(f"Submitting {KERNEL_OWNER}/{KERNEL_SLUG} v{KERNEL_VERSION} as {DESCRIPTION}", flush=True)
    print("Submission result:", submit(competitions), flush=True)


if __name__ == "__main__":
    main()
