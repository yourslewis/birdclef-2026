#!/usr/bin/env python3
"""Guarded late-window submitter for v653-v655 EoS8 proto/SED frontier forks.

Each candidate must have a COMPLETE Kaggle session, expected source markers,
and a valid nonconstant submission.csv before a code submission is created.
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
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, TypeVar

import requests
from kagglesdk.kaggle_http_client import KaggleHttpClient
from kagglesdk.competitions.services.competition_api_service import CompetitionApiClient
from kagglesdk.competitions.types.competition_api_service import ApiCreateCodeSubmissionRequest, ApiListSubmissionsRequest
from kagglesdk.kernels.services.kernels_api_service import KernelsApiClient
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest, ApiListKernelSessionOutputRequest

COMPETITION = "birdclef-2026"
OWNER = "yourslewis"
MAX_DAILY = 5
REPORT_PATH = Path("artifacts/source_winner_private_verifier_20260531T2225Z/submit_v653_v655_report.json")

@dataclass(frozen=True)
class Candidate:
    label: str
    slug: str
    version: int
    description: str
    expected_tag: str
    expected_proto: str
    expected_family: str

CANDIDATES = [
    Candidate("v653", "bc26-v653-eos8-proto080-sed020-verifier", 1, "v653: EoS8 PowerOpt proto080 sed020 frontier verifier", "v653_proto080_sed020_20260531T2225Z", "PROTO_RANK_WEIGHT = 0.800", "EoS8/PowerOptimization proto-heavy frontier source fork"),
    Candidate("v654", "bc26-v654-eos8-proto070-sed030-verifier", 1, "v654: EoS8 PowerOpt proto070 sed030 frontier verifier", "v654_proto070_sed030_20260531T2225Z", "PROTO_RANK_WEIGHT = 0.700", "EoS8/PowerOptimization near-original frontier source fork"),
    Candidate("v655", "bc26-v655-eos8-proto050-sed050-verifier", 1, "v655: EoS8 PowerOpt proto050 sed050 frontier verifier", "v655_proto050_sed050_20260531T2225Z", "PROTO_RANK_WEIGHT = 0.500", "EoS8/PowerOptimization balanced xSED frontier source fork"),
]

T = TypeVar("T")


def make_clients():
    with open(os.path.expanduser("~/.kaggle/kaggle.json")) as f:
        token = json.load(f)["key"]
    http = KaggleHttpClient(api_token=token)
    return token, CompetitionApiClient(http), KernelsApiClient(http)


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


def recent_submissions(competitions):
    req = ApiListSubmissionsRequest()
    req.competition_name = COMPETITION
    req.page_size = 200
    return call("list_submissions", lambda: competitions.list_submissions(req), attempts=4, sleep_s=30).submissions or []


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


def source_preflight(token: str, cand: Candidate) -> dict:
    url = f"https://www.kaggle.com/api/v1/kernels/pull/{OWNER}/{cand.slug}"
    r = call("kernel_pull", lambda: requests.get(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"}, timeout=90), attempts=4, sleep_s=30)
    info = {"status_code": r.status_code}
    if r.status_code != 200:
        info["ok"] = False
        info["error"] = r.text[:1000]
        return info
    data = r.json()
    meta = data.get("metadata", {})
    version = meta.get("currentVersionNumberNullable") or meta.get("currentVersionNumber")
    source = source_text_from_pull(data)
    markers = [cand.expected_tag, cand.expected_proto, "submission.csv", "test_soundscapes", "sample_submission.csv", "EXPERIMENT_TAG"]
    missing = [m for m in markers if m not in source]
    info.update({"version": version, "source_len": len(source), "missing_markers": missing, "ok": int(version) == cand.version and not missing})
    return info


def kernel_status(kernels, cand: Candidate):
    req = ApiGetKernelSessionStatusRequest()
    req.user_name = OWNER
    req.kernel_slug = cand.slug
    return call("kernel_status", lambda: kernels.get_kernel_session_status(req), attempts=4, sleep_s=30)


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
    row_ids = []
    ragged = 0
    for row in rows[1:]:
        if len(row) != n_cols:
            ragged += 1
            continue
        row_ids.append(row[0])
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
        "rows": n_rows,
        "cols": n_cols,
        "unique_rows": len(set(row_ids)),
        "bad_values": bad,
        "ragged_rows": ragged,
        "min": min(clean) if clean else None,
        "max": max(clean) if clean else None,
        "mean": sum(clean) / len(clean) if clean else None,
        "uniq_first100": len(set(clean[:100])) if clean else 0,
        "head_ids": row_ids[:3],
    }
    ok = n_rows > 0 and n_cols == 235 and stats["unique_rows"] == n_rows and bad == 0 and ragged == 0 and bool(clean) and stats["max"] > stats["min"] and stats["uniq_first100"] > 10
    return ok, stats


def output_preflight(kernels, cand: Candidate) -> dict:
    req = ApiListKernelSessionOutputRequest()
    req.user_name = OWNER
    req.kernel_slug = cand.slug
    req.page_size = 100
    out = call("kernel_output", lambda: kernels.list_kernel_session_output(req), attempts=4, sleep_s=30)
    files = {_file_name(f) for f in (getattr(out, "files", []) or [])}
    info = {"files": sorted(files), "ok": False}
    if "submission.csv" not in files:
        info["error"] = "missing submission.csv"
        return info
    sub_url = ""
    for f in getattr(out, "files", []) or []:
        if _file_name(f) == "submission.csv":
            sub_url = _file_url(f)
            break
    resp = call("download_submission", lambda: requests.get(sub_url, timeout=120), attempts=4, sleep_s=30)
    resp.raise_for_status()
    ok, stats = validate_submission_csv(resp.text)
    info.update({"submission_stats": stats, "ok": bool(ok)})
    log = getattr(out, "log", "") or ""
    if "Traceback" in log:
        info["ok"] = False
        info["error"] = "Traceback marker in kernel log"
    return info


def submit(competitions, cand: Candidate):
    req = ApiCreateCodeSubmissionRequest()
    req.competition_name = COMPETITION
    req.kernel_owner = OWNER
    req.kernel_slug = cand.slug
    req.kernel_version = cand.version
    req.file_name = "submission.csv"
    req.submission_description = cand.description
    return competitions.create_code_submission(req)


def write_report(report: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true", help="wait for running kernels and available slots")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    token, competitions, kernels = make_clients()
    report = {"utc_start": dt.datetime.now(dt.timezone.utc).isoformat(), "candidates": [], "submitted": []}

    pending = list(CANDIDATES)
    while pending:
        submissions = recent_submissions(competitions)
        today_count = count_today_utc(submissions)
        existing_desc = {str(getattr(s, "description", "")) for s in submissions}
        print(f"UTC submissions today: {today_count}/{MAX_DAILY}", flush=True)
        if today_count >= MAX_DAILY:
            print("daily slot cap reached", flush=True)
            break
        next_pending = []
        made_progress = False
        for cand in pending:
            cinfo = {"candidate": asdict(cand), "utc": dt.datetime.now(dt.timezone.utc).isoformat()}
            if cand.description in existing_desc:
                cinfo["decision"] = "skip_already_submitted"
                report["candidates"].append(cinfo)
                continue
            src = source_preflight(token, cand)
            cinfo["source_preflight"] = src
            if not src.get("ok"):
                cinfo["decision"] = "reject_source_preflight"
                report["candidates"].append(cinfo)
                continue
            st = kernel_status(kernels, cand)
            cinfo["status"] = json.loads(str(st)) if str(st).startswith("{") else str(st)
            status_text = str(getattr(st, "status", st)).upper()
            failure = getattr(st, "failure_message", getattr(st, "failureMessage", ""))
            if failure or "ERROR" in status_text or "FAILED" in status_text:
                cinfo["decision"] = "reject_kernel_failed"
                cinfo["failure"] = failure
                report["candidates"].append(cinfo)
                continue
            if "COMPLETE" not in status_text:
                cinfo["decision"] = "pending_kernel"
                report["candidates"].append(cinfo)
                next_pending.append(cand)
                continue
            out = output_preflight(kernels, cand)
            cinfo["output_preflight"] = out
            if not out.get("ok"):
                cinfo["decision"] = "reject_output_preflight"
                report["candidates"].append(cinfo)
                continue
            if args.preflight_only:
                cinfo["decision"] = "preflight_ok_no_submit"
                report["candidates"].append(cinfo)
                continue
            if today_count >= MAX_DAILY:
                cinfo["decision"] = "slot_cap_before_submit"
                report["candidates"].append(cinfo)
                next_pending.append(cand)
                continue
            resp = submit(competitions, cand)
            cinfo["decision"] = "submitted"
            cinfo["submit_result"] = str(resp)
            report["submitted"].append(cinfo)
            report["candidates"].append(cinfo)
            today_count += 1
            made_progress = True
            print("submitted", cand.label, resp, flush=True)
            if today_count >= MAX_DAILY:
                break
        write_report(report)
        if not next_pending or not args.wait:
            break
        pending = next_pending
        print(f"waiting {args.poll_seconds}s for pending kernels: {[c.label for c in pending]}", flush=True)
        time.sleep(max(30, args.poll_seconds))
    report["utc_end"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_report(report)
    print("report", REPORT_PATH, flush=True)

if __name__ == "__main__":
    main()
