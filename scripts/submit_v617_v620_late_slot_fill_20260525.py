#!/usr/bin/env python3
"""Late-UTC guarded exploratory BirdCLEF slot fill for 2026-05-25.

User policy for the hill-climb cron explicitly says to use available slots late in
UTC day if valid exploratory candidates exist. This submitter is intentionally
narrow and guarded:
- lists current submissions and UTC daily count;
- duplicate-guards by description;
- requires public source kernel COMPLETE with no failure;
- requires submission.csv to exist and be finite/nonconstant in public dry-run;
- rejects malformed, constant, missing-output, or duplicate-description finals;
- records source/data/kernel metadata before creating code submissions.

These are exploratory source-code submissions, not static public-output uploads:
Kaggle reruns each source kernel on the competition environment.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
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


@dataclass(frozen=True)
class Candidate:
    label: str
    owner: str
    slug: str
    description: str
    reason: str
    min_unique_first100: int = 2


CANDIDATES: list[Candidate] = [
    Candidate(
        "v617",
        "nina2025",
        "birdclef-2026-eos-7-sz",
        "v617: Exploratory direct Nina EoS7 sz sidecar source",
        "Fresh EoS7/sidecar source; COMPLETE, nonconstant final, raw ProtoSSM/SED/BirdNET sidecars present; late-day high-info plateau-family fill.",
    ),
    Candidate(
        "v618",
        "kruzzcc",
        "bc26-nina-eos4-fixed",
        "v618: Exploratory direct Kruzzcc Nina EoS4 BirdNET source",
        "BirdNET+EoS4 variant with valid nonconstant public final and nonconstant raw branches; source reruns hidden test path.",
    ),
    Candidate(
        "v619",
        "kruzzcc",
        "bc26-mtoshi-umap-bn-a",
        "v619: Exploratory direct Kruzzcc Mtoshi UMAP BirdNET source",
        "Mtoshi/UMAP/BirdNET branch variant; valid public final and raw branch outputs; not previously submitted under this source/description.",
    ),
    Candidate(
        "v620",
        "kazuhirokuriyama",
        "birdclef2026-karnak-rank-fusion",
        "v620: Exploratory direct Kazuhiro Karnak rank fusion source",
        "Karnak rank-fusion source with valid nonconstant final and COMPLETE hidden-test capable source; best remaining late-day filler after rejecting malformed P949/Kijiang/teacher-only kernels.",
    ),
]


def make_clients():
    with open(os.path.expanduser("~/.kaggle/kaggle.json")) as f:
        token = json.load(f)["key"]
    http = KaggleHttpClient(api_token=token)
    return token, CompetitionApiClient(http), KernelsApiClient(http)


def call(label: str, fn: Callable[[], T]) -> T:
    try:
        return fn()
    except requests.exceptions.HTTPError as exc:
        text = getattr(getattr(exc, "response", None), "text", "")
        raise RuntimeError(f"{label} HTTPError: {exc} {text[:1200]}") from exc


def recent_submissions(competitions: CompetitionApiClient):
    req = ApiListSubmissionsRequest()
    req.competition_name = COMPETITION
    req.page_size = 200
    return call("list_submissions", lambda: competitions.list_submissions(req)).submissions or []


def today_count(submissions) -> int:
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    return sum(str(getattr(s, "date", getattr(s, "date_nullable", ""))).startswith(today) for s in submissions)


def get_kernel(kernels: KernelsApiClient, cand: Candidate):
    req = ApiGetKernelRequest()
    req.user_name = cand.owner
    req.kernel_slug = cand.slug
    return call(f"get_kernel {cand.owner}/{cand.slug}", lambda: kernels.get_kernel(req))


def kernel_status(kernels: KernelsApiClient, cand: Candidate):
    req = ApiGetKernelSessionStatusRequest()
    req.user_name = cand.owner
    req.kernel_slug = cand.slug
    return call(f"status {cand.owner}/{cand.slug}", lambda: kernels.get_kernel_session_status(req))


def kernel_output(kernels: KernelsApiClient, cand: Candidate):
    req = ApiListKernelSessionOutputRequest()
    req.user_name = cand.owner
    req.kernel_slug = cand.slug
    req.page_size = 100
    return call(f"output {cand.owner}/{cand.slug}", lambda: kernels.list_kernel_session_output(req))


def file_name(file_obj) -> str:
    return getattr(file_obj, "file_name", getattr(file_obj, "fileName", ""))


def file_url(file_obj) -> str:
    return getattr(file_obj, "url", "")


def validate_submission_csv(text: str, cand: Candidate) -> tuple[bool, dict]:
    rows = list(csv.reader(text.splitlines()))
    if not rows:
        return False, {"error": "empty csv"}
    n_rows = len(rows) - 1
    n_cols = len(rows[0])
    vals: list[float] = []
    bad = 0
    ragged = 0
    ids: list[str] = []
    for row in rows[1:]:
        if len(row) != n_cols:
            ragged += 1
            continue
        ids.append(row[0])
        for x in row[1:]:
            try:
                v = float(x)
            except Exception:
                bad += 1
                continue
            if not math.isfinite(v):
                bad += 1
            else:
                vals.append(v)
    clean = vals
    uniq_first100 = len(set(round(v, 8) for v in clean[: min(100, len(clean))])) if clean else 0
    stats = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "unique_rows": len(set(ids)),
        "bad_values": bad,
        "ragged_rows": ragged,
        "min": min(clean) if clean else None,
        "max": max(clean) if clean else None,
        "mean": sum(clean) / len(clean) if clean else None,
        "uniq_first100": uniq_first100,
        "head_ids": ids[:3],
    }
    ok = (
        n_rows > 0
        and n_cols == 235
        and ragged == 0
        and bad == 0
        and bool(clean)
        and stats["max"] is not None
        and stats["max"] > stats["min"]
        and uniq_first100 >= cand.min_unique_first100
    )
    return ok, stats


def source_markers(source: str) -> dict:
    low = source.lower()
    keys = [
        "test_soundscapes",
        "sample_submission",
        "submission.csv",
        "/kaggle/input",
        "read_csv",
        "onnx",
        "birdnet",
        "protossm",
        "sed",
        "no hidden test",
        "dry-run",
        "fallback",
    ]
    return {k: (k in low) for k in keys}


def preflight(kernels: KernelsApiClient, cand: Candidate) -> tuple[bool, int | None, dict]:
    kernel = get_kernel(kernels, cand)
    meta = getattr(kernel, "metadata", None)
    blob = getattr(kernel, "blob", None)
    source = getattr(blob, "source", "") if blob else ""
    version = None
    meta_dict = None
    if meta is not None:
        try:
            meta_dict = meta.to_dict()
        except Exception:
            meta_dict = {"ref": getattr(meta, "ref", None), "currentVersionNumber": getattr(meta, "current_version_number", None)}
        version = meta_dict.get("currentVersionNumber") or meta_dict.get("current_version_number")
    st = kernel_status(kernels, cand)
    status_text = str(getattr(st, "status", st)).upper()
    failure = getattr(st, "failure_message", getattr(st, "failureMessage", ""))
    info = {
        "ref": f"{cand.owner}/{cand.slug}",
        "version": version,
        "metadata": meta_dict,
        "source_len": len(source),
        "source_markers": source_markers(source),
        "status": str(st),
        "failure": failure,
        "reason": cand.reason,
    }
    if failure or "ERROR" in status_text or "FAILED" in status_text or "COMPLETE" not in status_text:
        info["reject"] = "kernel not COMPLETE or has failure"
        return False, version, info
    out = kernel_output(kernels, cand)
    files = list(getattr(out, "files", []) or [])
    names = sorted(file_name(f) for f in files)
    info["files"] = names
    sub_url = ""
    for f in files:
        if file_name(f) == "submission.csv":
            sub_url = file_url(f)
            break
    if not sub_url:
        info["reject"] = "submission.csv missing"
        return False, version, info
    resp = requests.get(sub_url, timeout=120)
    resp.raise_for_status()
    ok, stats = validate_submission_csv(resp.text, cand)
    info["submission_stats"] = stats
    if not ok:
        info["reject"] = "submission.csv malformed, nonfinite, constant, or wrong columns"
        return False, version, info
    if not info["source_markers"].get("test_soundscapes"):
        info["reject"] = "source lacks test_soundscapes marker"
        return False, version, info
    return True, int(version) if version is not None else None, info


def submit(competitions: CompetitionApiClient, cand: Candidate, version: int):
    req = ApiCreateCodeSubmissionRequest()
    req.competition_name = COMPETITION
    req.kernel_owner = cand.owner
    req.kernel_slug = cand.slug
    req.kernel_version = version
    req.file_name = "submission.csv"
    req.submission_description = cand.description
    return competitions.create_code_submission(req)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--max-submissions", type=int, default=4)
    ap.add_argument("--artifact", default="artifacts/public_kernels_20260525_late_scout/submit_v617_v620_late_slot_fill_20260525.json")
    args = ap.parse_args()

    _, competitions, kernels = make_clients()
    submissions = recent_submissions(competitions)
    existing_desc = {str(s.description) for s in submissions}
    count = today_count(submissions)
    now = dt.datetime.now(dt.timezone.utc)
    reset = now.replace(hour=0, minute=0, second=0, microsecond=0) + dt.timedelta(days=1)
    report = {
        "utc_now": now.isoformat(),
        "utc_reset": reset.isoformat(),
        "hours_to_reset": (reset - now).total_seconds() / 3600,
        "today_count_start": count,
        "latest_submissions": [
            {
                "date": str(getattr(s, "date", "")),
                "description": str(getattr(s, "description", "")),
                "score": str(getattr(s, "public_score", getattr(s, "publicScore", ""))),
                "status": str(getattr(s, "status", "")),
                "ref": getattr(s, "ref", None),
            }
            for s in submissions[:15]
        ],
        "candidates": [],
        "submitted": [],
    }
    print(json.dumps({"utc_now": report["utc_now"], "today_count_start": count, "mode": "submit" if args.submit else "dry-run"}, indent=2), flush=True)

    submitted = 0
    for cand in CANDIDATES:
        item = {"label": cand.label, "ref": f"{cand.owner}/{cand.slug}", "description": cand.description}
        if count >= MAX_DAILY:
            item["decision"] = "skip_cap_reached"
            report["candidates"].append(item)
            break
        if submitted >= args.max_submissions:
            item["decision"] = "skip_run_max_reached"
            report["candidates"].append(item)
            break
        if cand.description in existing_desc:
            item["decision"] = "skip_duplicate_description"
            report["candidates"].append(item)
            continue
        try:
            ok, version, info = preflight(kernels, cand)
        except Exception as exc:
            item.update({"decision": "preflight_exception", "error": repr(exc)})
            print(json.dumps(item, default=str), flush=True)
            report["candidates"].append(item)
            continue
        item.update({"preflight_ok": ok, "version": version, "info": info})
        if not ok or version is None:
            item["decision"] = "reject_preflight"
            print(json.dumps(item, default=str)[:3000], flush=True)
            report["candidates"].append(item)
            continue
        if not args.submit:
            item["decision"] = "dry_run_would_submit"
            print(json.dumps(item, default=str)[:3000], flush=True)
            report["candidates"].append(item)
            continue
        try:
            result = submit(competitions, cand, version)
            item["decision"] = "submitted"
            item["submit_result"] = str(result)
            existing_desc.add(cand.description)
            submitted += 1
            count += 1
            report["submitted"].append(item)
            print(json.dumps({"submitted": cand.label, "description": cand.description, "result": str(result)}, default=str), flush=True)
        except requests.exceptions.HTTPError as exc:
            text = getattr(getattr(exc, "response", None), "text", "") or str(exc)
            item.update({"decision": "submit_http_error", "error": text[:2000]})
            report["candidates"].append(item)
            print(json.dumps(item, default=str)[:3000], flush=True)
            if "daily" in text.lower() and ("allowance" in text.lower() or "limit" in text.lower() or "quota" in text.lower()):
                break
            raise
        report["candidates"].append(item)

    report["today_count_end_est"] = count
    report["submitted_count"] = submitted
    os.makedirs(os.path.dirname(args.artifact), exist_ok=True)
    with open(args.artifact, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps({"submitted_count": submitted, "today_count_end_est": count, "artifact": args.artifact}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
