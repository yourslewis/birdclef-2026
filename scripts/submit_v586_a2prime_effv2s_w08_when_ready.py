"""Cap-aware submitter for repo-owned v586 A2Prime/EffV2S w08 extraction.

v585 FrankSunP scored 0.922, so this fallback is now the next distinct
0.96-relevant hypothesis. The pushed Kaggle kernel emits multiple CSVs; version 2 makes the EffV2S w08 blend the notebook's submission.csv because BirdCLEF only accepts that filename for code submissions.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Callable, TypeVar

import requests
from kagglesdk.competitions.services.competition_api_service import CompetitionApiClient
from kagglesdk.competitions.types.competition_api_service import ApiCreateCodeSubmissionRequest, ApiListSubmissionsRequest
from kagglesdk.kaggle_http_client import KaggleHttpClient
from kagglesdk.kernels.services.kernels_api_service import KernelsApiClient
from kagglesdk.kernels.types.kernels_api_service import (
    ApiDownloadKernelOutputRequest,
    ApiGetKernelSessionStatusRequest,
    ApiListKernelSessionOutputRequest,
)

COMPETITION = "birdclef-2026"
KERNEL_OWNER = "yourslewis"
KERNEL_SLUG = "bc26-v586-a2prime-effv2s-extraction-r2"
KERNEL_VERSION = 2
FILE_NAME = "submission.csv"
DESCRIPTION = "v586: Repo-owned A2Prime EffV2S rank blend w08 after v585 drop"
LOG_PATH = Path("logs/v586_a2prime_effv2s_w08_preflight.txt")

REQUIRED_SOURCE_MARKERS = [
    "BaiyuEffV2S",
    "submission_effv2s.csv",
    "a2_effv2s_w08",
    "a2prime_blend_summary.csv",
    "test_soundscapes",
    "sample_submission.csv",
    "row_id",
]
REQUIRED_OUTPUTS = {
    FILE_NAME,
    "submission_effv2s.csv",
    "submission_protossm.csv",
    "submission_sed.csv",
    "submission_birdnet.csv",
    "a2prime_blend_summary.csv",
    "effv2s_branch_summary.csv",
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
    r = call(
        "kernel_pull",
        lambda: requests.get(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"}, timeout=90),
    )
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


def output_files(kernels) -> set[str]:
    req = ApiListKernelSessionOutputRequest()
    req.user_name = KERNEL_OWNER
    req.kernel_slug = KERNEL_SLUG
    req.page_size = 100
    out = call("kernel_output", lambda: kernels.list_kernel_session_output(req))
    return {getattr(f, "file_name", getattr(f, "fileName", "")) for f in (getattr(out, "files", []) or [])}


def download_output_text(kernels, file_path: str) -> str:
    req = ApiDownloadKernelOutputRequest()
    req.owner_slug = KERNEL_OWNER
    req.kernel_slug = KERNEL_SLUG
    req.file_path = file_path
    req.version_number = KERNEL_VERSION
    redirect = call(f"download {file_path}", lambda: kernels.download_kernel_output(req))
    url = getattr(redirect, "url", "")
    if not url:
        raise RuntimeError(f"No download URL for {file_path}: {redirect}")
    r = call(f"fetch {file_path}", lambda: requests.get(url, timeout=120))
    r.raise_for_status()
    return r.text


def validate_blend_summary(text: str) -> tuple[bool, str]:
    rows = list(csv.DictReader(io.StringIO(text)))
    target = None
    for row in rows:
        if row.get("candidate") == "a2_effv2s_w08":
            target = row
            break
    if target is None:
        return False, "a2_effv2s_w08 missing from a2prime_blend_summary.csv"
    if str(target.get("effv2s_active", "")).lower() != "true":
        return False, f"effv2s_active not true: {target}"
    if abs(float(target.get("w_effv2s", "nan")) - 0.08) > 1e-9:
        return False, f"w_effv2s is not 0.08: {target}"
    corr = float(target.get("proto_effv2s_rank_corr", "nan"))
    if not math.isfinite(corr):
        return False, f"invalid proto_effv2s_rank_corr: {target}"
    return True, f"a2_effv2s_w08 ok; proto_effv2s_rank_corr={corr:.6f}; mean_score={target.get('mean_score')}"


def validate_submission_csv(text: str) -> tuple[bool, str]:
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        return False, "empty CSV"
    if not header or header[0] != "row_id" or len(header) < 10:
        return False, f"bad header: first={header[:3]} columns={len(header)}"
    rows = 0
    min_v = float("inf")
    max_v = float("-inf")
    first_values: list[float] = []
    seen = set()
    dupes = 0
    for row in reader:
        if not row:
            continue
        rows += 1
        if row[0] in seen:
            dupes += 1
        seen.add(row[0])
        if len(row) != len(header):
            return False, f"row {rows} has {len(row)} columns, expected {len(header)}"
        # Sample all rows but only store a few values; output is competition-sized.
        for raw in row[1:]:
            try:
                v = float(raw)
            except ValueError:
                return False, f"non-numeric probability at row {rows}: {raw!r}"
            if not math.isfinite(v) or v < 0.0 or v > 1.0:
                return False, f"invalid probability {v} at row {rows}"
            min_v = min(min_v, v)
            max_v = max(max_v, v)
            if len(first_values) < 1000:
                first_values.append(v)
    if rows <= 0:
        return False, "no data rows"
    if dupes:
        return False, f"duplicate row_id count={dupes}"
    if max_v <= min_v:
        return False, f"constant predictions min=max={min_v}"
    sample_unique = len({round(v, 8) for v in first_values})
    if sample_unique < 3:
        return False, f"too few unique sampled probabilities={sample_unique}"
    return True, f"csv ok rows={rows} cols={len(header)} min={min_v:.6g} max={max_v:.6g} sampled_unique={sample_unique}"


def output_preflight(kernels) -> bool:
    files = output_files(kernels)
    missing = sorted(REQUIRED_OUTPUTS - files)
    print("output files", sorted(files), flush=True)
    LOG_PATH.parent.mkdir(exist_ok=True)
    notes = ["files=" + repr(sorted(files)), "missing=" + repr(missing)]
    if missing:
        LOG_PATH.write_text("\n".join(notes) + "\n")
        print("Missing required output files:", missing, flush=True)
        return False

    blend_ok, blend_msg = validate_blend_summary(download_output_text(kernels, "a2prime_blend_summary.csv"))
    csv_ok, csv_msg = validate_submission_csv(download_output_text(kernels, FILE_NAME))
    notes.extend(["blend=" + blend_msg, "csv=" + csv_msg])
    LOG_PATH.write_text("\n".join(notes) + "\n")
    print(blend_msg, flush=True)
    print(csv_msg, flush=True)
    return blend_ok and csv_ok


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
    req.file_name = FILE_NAME
    req.submission_description = DESCRIPTION
    return competitions.create_code_submission(req)


def main():
    token, competitions, kernels = make_clients()
    while True:
        submissions = recent_submissions(competitions)
        if any(str(s.description) == DESCRIPTION for s in submissions):
            print("v586 already submitted; exiting", flush=True)
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
            print("kernel not complete yet; sleeping 10 minutes", flush=True)
            time.sleep(600)
            continue
        if not output_preflight(kernels):
            raise SystemExit("output preflight failed; not submitting")
        try:
            print(f"Submitting {KERNEL_OWNER}/{KERNEL_SLUG} v{KERNEL_VERSION} file={FILE_NAME} as {DESCRIPTION}", flush=True)
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
