#!/usr/bin/env python3
"""Late-UTC guarded exploratory BirdCLEF slot fill for 2026-05-30.

Candidate pool is intentionally limited to fresh/high-score public source kernels that
were not already submitted in v621-v643.  This script uses source-code Kaggle
submissions only; it rejects malformed/static-looking public-session outputs and
exact duplicate dry-run hashes seen in recent late fills.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Callable, TypeVar

import requests
from kagglesdk.kaggle_http_client import KaggleHttpClient
from kagglesdk.competitions.services.competition_api_service import CompetitionApiClient
from kagglesdk.competitions.types.competition_api_service import ApiCreateCodeSubmissionRequest, ApiListSubmissionsRequest
from kagglesdk.kernels.services.kernels_api_service import KernelsApiClient
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest, ApiGetKernelSessionStatusRequest, ApiListKernelSessionOutputRequest

COMPETITION = "birdclef-2026"
MAX_DAILY = 5
T = TypeVar("T")

# Dry-run hashes already consumed or rejected as exact duplicates in recent slot fills.
RECENT_DRYRUN_HASHES = {
    "2cd2be250a4020a4", "62274b98d6a4f39c", "09ef02cb55ff66b7", "97cd802bb60f6b83", "89438737d0b97271",
    "7439ae3b15a3f6c6", "6b5910239e37bd4b", "aa41ccbbf2a84046", "3224ced9a582e251", "e293021c399fa925",
    "9fd71fb24d94ca92", "c6867f1d294b8ee5", "30cc8796efbbaee8", "48e8eb7f8409ac11", "4ea48f143ed5b877",
    "e5c937e6d87cb4fc", "f0947cc50457ccdd", "d0545f2c89ce36b8", "6dcda7328bb22532", "c166a6c8e22078c1",
    "07f51c964cdf249e", "fb94c3843d36f980", "21d11c1a70aad873",
    "f61e71b9368cc673", "0ee04c918f807616", "5afa1de99305ffd1",
}

@dataclass(frozen=True)
class Candidate:
    label: str
    owner: str
    slug: str
    description: str
    reason: str
    expected_family: str
    min_unique_first100: int = 2

CANDIDATES: list[Candidate] = [
    Candidate(
        "v644", "yaroslavkholmirzayev", "0950-replay",
        "v644: Late-fill Yaroslav 0950 replay source",
        "Fresh high-score public source listed at the frontier on 2026-05-30; likely distinct Yaroslav/taxonomy family and useful late-day information.",
        "Yaroslav 0950 replay public source",
    ),
    Candidate(
        "v645", "nina2025", "birdclef-2026-eos-9",
        "v645: Late-fill Nina EoS9 source",
        "Top score-sorted public EoS.9 source; distinct from submitted EoS1/EoS4/EoS7 fills and plausible high-LB late-day candidate.",
        "Nina EoS9 public source",
    ),
    Candidate(
        "v646", "anthonytherrien", "birdclef-2026-ensemble-0-950",
        "v646: Late-fill Anthony ensemble 0.950 source",
        "High-score public ensemble source; fallback if earlier candidates fail preflight or duplicate guards.",
        "Anthony ensemble 0.950 public source",
    ),
    Candidate(
        "v647", "ryutoyoda", "birdclef-2026-exp013-eos8-sidecar",
        "v647: Late-fill Ryuto EoS8 sidecar source",
        "EoS8 sidecar variant with nontrivial votes; fallback source-code datapoint under late-day policy.",
        "Ryuto EoS8 sidecar public source",
    ),
    Candidate(
        "v648", "fleongg", "birdclef-2026-hier-tax-claude-fork",
        "v648: Late-fill Fleong hier-tax fork source",
        "Hierarchical-taxonomy fork from score/frontier list; fallback if source and output guards pass.",
        "Fleong hierarchical taxonomy public source",
    ),
    Candidate(
        "v649", "fleongg", "birdclef-2026-eos-9-claude-fork",
        "v649: Late-fill Fleong EoS9 fork source",
        "EoS9 fork with separate author/source; fallback if nonduplicate and hidden-safe.",
        "Fleong EoS9 fork public source",
    ),
    Candidate(
        "v650", "ahmedkhudair121", "bc2026-claude-nina-eos-8-fork",
        "v650: Late-fill Ahmed Nina EoS8 fork source",
        "EoS8 fork in high-score list; fallback late-day public-source datapoint.",
        "Ahmed Nina EoS8 fork public source",
    ),
]

def make_clients():
    with open(os.path.expanduser("~/.kaggle/kaggle.json")) as f:
        token = json.load(f)["key"]
    http = KaggleHttpClient(api_token=token)
    return CompetitionApiClient(http), KernelsApiClient(http)

def call(label: str, fn: Callable[[], T]) -> T:
    try:
        return fn()
    except requests.exceptions.HTTPError as exc:
        text = getattr(getattr(exc, "response", None), "text", "")
        raise RuntimeError(f"{label} HTTPError: {exc} {text[:1200]}") from exc

def recent_submissions(competitions: CompetitionApiClient):
    req = ApiListSubmissionsRequest(); req.competition_name = COMPETITION; req.page_size = 200
    return call("list_submissions", lambda: competitions.list_submissions(req)).submissions or []

def sub_date_str(s) -> str:
    return str(getattr(s, "date", getattr(s, "date_nullable", "")))

def sub_score_str(s) -> str:
    return str(getattr(s, "public_score", getattr(s, "publicScore", "")))

def today_count(submissions) -> int:
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    return sum(sub_date_str(s).startswith(today) for s in submissions)

def get_kernel(kernels: KernelsApiClient, cand: Candidate):
    req = ApiGetKernelRequest(); req.user_name = cand.owner; req.kernel_slug = cand.slug
    return call(f"get_kernel {cand.owner}/{cand.slug}", lambda: kernels.get_kernel(req))

def kernel_status(kernels: KernelsApiClient, cand: Candidate):
    req = ApiGetKernelSessionStatusRequest(); req.user_name = cand.owner; req.kernel_slug = cand.slug
    return call(f"status {cand.owner}/{cand.slug}", lambda: kernels.get_kernel_session_status(req))

def kernel_output(kernels: KernelsApiClient, cand: Candidate):
    req = ApiListKernelSessionOutputRequest(); req.user_name = cand.owner; req.kernel_slug = cand.slug; req.page_size = 100
    return call(f"output {cand.owner}/{cand.slug}", lambda: kernels.list_kernel_session_output(req))

def file_name(file_obj) -> str:
    return getattr(file_obj, "file_name", getattr(file_obj, "fileName", ""))

def file_url(file_obj) -> str:
    return getattr(file_obj, "url", "")

def validate_submission_csv(text: str, cand: Candidate) -> tuple[bool, dict]:
    rows = list(csv.reader(text.splitlines()))
    if not rows:
        return False, {"error": "empty csv"}
    n_rows, n_cols = len(rows) - 1, len(rows[0])
    vals: list[float] = []; ids: list[str] = []; bad = ragged = 0
    for row in rows[1:]:
        if len(row) != n_cols:
            ragged += 1; continue
        ids.append(row[0])
        for x in row[1:]:
            try: v = float(x)
            except Exception: bad += 1; continue
            if not math.isfinite(v): bad += 1
            else: vals.append(v)
    sha = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
    uniq_first100 = len(set(round(v, 8) for v in vals[:min(100, len(vals))])) if vals else 0
    stats = {
        "n_rows": n_rows, "n_cols": n_cols, "unique_rows": len(set(ids)), "bad_values": bad, "ragged_rows": ragged,
        "min": min(vals) if vals else None, "max": max(vals) if vals else None, "mean": sum(vals)/len(vals) if vals else None,
        "uniq_first100": uniq_first100, "head_ids": ids[:3], "sha256_16": sha,
    }
    ok = n_rows > 0 and n_cols == 235 and ragged == 0 and bad == 0 and bool(vals) and stats["max"] is not None and stats["max"] > stats["min"] and uniq_first100 >= cand.min_unique_first100
    return ok, stats

def source_markers(source: str) -> dict:
    low = source.lower()
    keys = ["test_soundscapes", "sample_submission", "submission.csv", "/kaggle/input", "read_csv", "onnx", "torch", "birdnet", "protossm", "sed", "perch", "no hidden test", "dry-run", "fallback"]
    return {k: (k in low) for k in keys}

def preflight(kernels: KernelsApiClient, cand: Candidate, seen_hashes: set[str]) -> tuple[bool, int | None, dict]:
    kernel = get_kernel(kernels, cand)
    meta = getattr(kernel, "metadata", None); blob = getattr(kernel, "blob", None); source = getattr(blob, "source", "") if blob else ""
    version = None
    if meta is not None:
        try: md = meta.to_dict()
        except Exception: md = {}
        version = md.get("currentVersionNumber") or md.get("current_version_number") or md.get("currentVersionNumberNullable") or md.get("current_version_number_nullable")
    st = kernel_status(kernels, cand); status_text = str(getattr(st, "status", st)).upper(); failure = getattr(st, "failure_message", getattr(st, "failureMessage", ""))
    markers = source_markers(source)
    info = {"ref": f"{cand.owner}/{cand.slug}", "version": version, "source_len": len(source), "source_markers": markers, "status": str(st), "failure": failure, "reason": cand.reason, "expected_family": cand.expected_family}
    if failure or "ERROR" in status_text or "FAILED" in status_text or "COMPLETE" not in status_text:
        info["reject"] = "kernel not COMPLETE or has failure"; return False, version, info
    if not markers.get("test_soundscapes") or not markers.get("sample_submission") or not markers.get("submission.csv"):
        info["reject"] = "source lacks required hidden-test/sample/submission markers"; return False, version, info
    out = kernel_output(kernels, cand); files = list(getattr(out, "files", []) or []); names = sorted(file_name(f) for f in files); info["files"] = names
    sub_url = next((file_url(f) for f in files if file_name(f) == "submission.csv"), "")
    if not sub_url:
        info["reject"] = "submission.csv missing from public session output"; return False, version, info
    resp = requests.get(sub_url, timeout=120); resp.raise_for_status()
    ok, stats = validate_submission_csv(resp.text, cand); info["submission_stats"] = stats
    if not ok:
        info["reject"] = "submission.csv malformed/nonfinite/constant/wrong columns"; return False, version, info
    if stats["sha256_16"] in seen_hashes or stats["sha256_16"] in RECENT_DRYRUN_HASHES:
        info["reject"] = f"duplicate public dry-run hash {stats['sha256_16']}"; return False, version, info
    return True, int(version) if version is not None else None, info

def submit(competitions: CompetitionApiClient, cand: Candidate, version: int):
    req = ApiCreateCodeSubmissionRequest(); req.competition_name = COMPETITION; req.kernel_owner = cand.owner; req.kernel_slug = cand.slug; req.kernel_version = version; req.file_name = "submission.csv"; req.submission_description = cand.description
    return competitions.create_code_submission(req)

def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--submit", action="store_true"); ap.add_argument("--max-submissions", type=int, default=2); ap.add_argument("--artifact", default="artifacts/public_kernels_20260530_late_scout/submit_v644_v645_20260530.json")
    args = ap.parse_args()
    competitions, kernels = make_clients(); submissions = recent_submissions(competitions); existing_desc = {str(getattr(s, "description", "")) for s in submissions}; count = today_count(submissions)
    now = dt.datetime.now(dt.timezone.utc); reset = now.replace(hour=0, minute=0, second=0, microsecond=0) + dt.timedelta(days=1)
    seen_hashes: set[str] = set(); submitted = 0
    report = {"utc_now": now.isoformat(), "utc_reset": reset.isoformat(), "hours_to_reset": (reset-now).total_seconds()/3600, "today_count_start": count, "latest_submissions": [{"date": sub_date_str(s), "description": str(getattr(s,"description", "")), "score": sub_score_str(s), "status": str(getattr(s,"status", "")), "ref": getattr(s,"ref", None)} for s in submissions[:20]], "candidates": [], "submitted": []}
    print(json.dumps({"utc_now": report["utc_now"], "today_count_start": count, "hours_to_reset": report["hours_to_reset"], "mode": "submit" if args.submit else "dry-run"}, indent=2), flush=True)
    for cand in CANDIDATES:
        item = {"label": cand.label, "ref": f"{cand.owner}/{cand.slug}", "description": cand.description, "expected_family": cand.expected_family}
        if count >= MAX_DAILY:
            item["decision"] = "skip_cap_reached"; report["candidates"].append(item); break
        if submitted >= args.max_submissions:
            item["decision"] = "skip_run_max_reached"; report["candidates"].append(item); break
        if cand.description in existing_desc:
            item["decision"] = "skip_duplicate_description"; report["candidates"].append(item); continue
        try: ok, version, info = preflight(kernels, cand, seen_hashes)
        except Exception as exc:
            item.update({"decision": "preflight_exception", "error": repr(exc)}); print(json.dumps(item, default=str)[:3000], flush=True); report["candidates"].append(item); continue
        item.update({"preflight_ok": ok, "version": version, "info": info})
        if not ok or version is None:
            item["decision"] = "reject_preflight"; print(json.dumps(item, default=str)[:3000], flush=True); report["candidates"].append(item); continue
        seen_hashes.add(info["submission_stats"]["sha256_16"])
        if not args.submit:
            item["decision"] = "dry_run_would_submit"; print(json.dumps(item, default=str)[:3000], flush=True); report["candidates"].append(item); continue
        try:
            result = submit(competitions, cand, version)
            item["decision"] = "submitted"; item["submit_result"] = str(result); submitted += 1; count += 1; existing_desc.add(cand.description); report["submitted"].append(item); print(json.dumps({"submitted": cand.label, "description": cand.description, "result": str(result)}, default=str), flush=True)
        except requests.exceptions.HTTPError as exc:
            text = getattr(getattr(exc, "response", None), "text", "") or str(exc); item.update({"decision": "submit_http_error", "error": text[:2000]}); print(json.dumps(item, default=str)[:3000], flush=True); report["candidates"].append(item)
            if "daily" in text.lower() and ("allowance" in text.lower() or "limit" in text.lower() or "quota" in text.lower()): break
            raise
        report["candidates"].append(item)
    report["today_count_end_est"] = count; report["submitted_count"] = submitted
    os.makedirs(os.path.dirname(args.artifact), exist_ok=True)
    with open(args.artifact, "w") as f: json.dump(report, f, indent=2, default=str)
    print(json.dumps({"submitted_count": submitted, "today_count_end_est": count, "artifact": args.artifact}, indent=2), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
