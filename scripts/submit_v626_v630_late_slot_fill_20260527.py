#!/usr/bin/env python3
"""Late-UTC guarded exploratory BirdCLEF slot fill for 2026-05-27.

Uses the standing hill-climb cron authorization and late-day policy:
- list current submissions and UTC daily count;
- only submit if <5 daily slots used;
- source-code submissions only (Kaggle reruns the kernel), not static public CSV uploads;
- require COMPLETE public source kernels with submission.csv output;
- require hidden-test path marker, finite/nonconstant public dry-run output, 235 columns;
- reject duplicate descriptions and exact duplicate public dry-run hashes within this run / recent submitted public dry-runs.
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

# Public dry-run output hashes for v621-v625 and other exact duplicates observed in the 2026-05-26 scout.
# These are only a guard against obvious duplicate matrices; hidden reruns can still differ in row count.
RECENT_SUBMITTED_DRYRUN_HASHES = {
    "2cd2be250a4020a4",  # v621 Pilkwang EoS7 OOF-gated PCEN
    "62274b98d6a4f39c",  # v622 Beicicc EoS6 P090
    "09ef02cb55ff66b7",  # v623 Anthony M5-only
    "97cd802bb60f6b83",  # v624 Haru public top2 P125
    "89438737d0b97271",  # v625 Safar 0948
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
        "v626",
        "shahadaljayzani",
        "bc2026-claude-jaejohn-perch-starter-fork",
        "v626: Late-fill Jaejohn Perch meta-probe source",
        "Clean source-risk profile in 2026-05-26 scout; Perch metadata/probe branch with hidden-test path and nonconstant public dry-run; distinct from v621-v625 hashes.",
        "Perch meta/probe soundscape source",
    ),
    Candidate(
        "v627",
        "abdulrahmansu10",
        "bc2026-claude-hideyukizushi-protossm-src-fork",
        "v627: Late-fill Hideyukizushi ProtoSSM source",
        "Clean source-risk profile; ProtoSSM/source branch with hidden-test path and nonconstant 240-row public dry-run; high information value vs plateau sources.",
        "ProtoSSM source/fork",
    ),
    Candidate(
        "v628",
        "sultanalgizani",
        "bc2026-claude-cliff-gate-combo-fork",
        "v628: Late-fill Cliff gate combo source",
        "Gate-combo public source; notebook explicitly uses hidden test when mounted and falls back only for public dry-run; nonconstant output and unique dry-run hash.",
        "Gate-combo ensemble",
    ),
    Candidate(
        "v629",
        "hassanalgizani",
        "bc2026-claude-yaroslav-birdnet-3rd-fork",
        "v629: Late-fill Yaroslav BirdNET third source",
        "BirdNET/Yaroslav-family branch with hidden-test path, finite nonconstant dry-run, and unique hash; useful distinct public-family slot fill late in UTC day.",
        "BirdNET/Yaroslav public source",
    ),
    Candidate(
        "v630",
        "hassan1417",
        "bc2026-claude-tucker-distilled-sed-fork",
        "v630: Late-fill Tucker distilled SED source",
        "Distilled SED/perch branch with hidden-test path and 60-row finite nonconstant public dry-run; decorrelated enough for late exploratory slot.",
        "Distilled SED public source",
        min_unique_first100=10,
    ),
    # Fallbacks if one of the primary five fails live preflight.
    Candidate(
        "v631",
        "joriahmed",
        "bc2026-claude-maryna-twopass-ssm-fork",
        "v631: Late-fill Maryna two-pass SSM source",
        "Two-pass SSM source fallback; hidden-test path, nonconstant public dry-run, unique hash.",
        "Two-pass SSM public source",
    ),
    Candidate(
        "v632",
        "abdulrahmansu10",
        "bc2026-claude-vyanktesh-fork",
        "v632: Late-fill Vyanktesh source",
        "Vyanktesh source fallback; hidden-test path, nonconstant public dry-run, unique hash.",
        "Vyanktesh public source",
    ),
    Candidate(
        "v633",
        "ahmedkhudair121",
        "bc2026-claude-raunak-multi-model-fork",
        "v633: Late-fill Raunak multi-model source",
        "Raunak multi-model source fallback; hidden-test path, nonconstant public dry-run, unique hash.",
        "Raunak multi-model public source",
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


def sub_date_str(s) -> str:
    return str(getattr(s, "date", getattr(s, "date_nullable", "")))


def sub_score_str(s) -> str:
    return str(getattr(s, "public_score", getattr(s, "publicScore", "")))


def today_count(submissions) -> int:
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    return sum(sub_date_str(s).startswith(today) for s in submissions)


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
    uniq_first100 = len(set(round(v, 8) for v in vals[: min(100, len(vals))])) if vals else 0
    sha = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
    stats = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "unique_rows": len(set(ids)),
        "bad_values": bad,
        "ragged_rows": ragged,
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
        "mean": sum(vals) / len(vals) if vals else None,
        "uniq_first100": uniq_first100,
        "head_ids": ids[:3],
        "sha256_16": sha,
    }
    ok = (
        n_rows > 0
        and n_cols == 235
        and ragged == 0
        and bad == 0
        and bool(vals)
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
        "torch",
        "birdnet",
        "protossm",
        "sed",
        "perch",
        "no hidden test",
        "dry-run",
        "fallback",
    ]
    return {k: (k in low) for k in keys}


def preflight(kernels: KernelsApiClient, cand: Candidate, seen_hashes: set[str]) -> tuple[bool, int | None, dict]:
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
            meta_dict = {
                "ref": getattr(meta, "ref", None),
                "currentVersionNumber": getattr(meta, "current_version_number", None),
                "currentVersionNumberNullable": getattr(meta, "current_version_number_nullable", None),
            }
        version = (
            meta_dict.get("currentVersionNumber")
            or meta_dict.get("current_version_number")
            or meta_dict.get("currentVersionNumberNullable")
            or meta_dict.get("current_version_number_nullable")
        )
    st = kernel_status(kernels, cand)
    status_text = str(getattr(st, "status", st)).upper()
    failure = getattr(st, "failure_message", getattr(st, "failureMessage", ""))
    markers = source_markers(source)
    info = {
        "ref": f"{cand.owner}/{cand.slug}",
        "version": version,
        "source_len": len(source),
        "source_markers": markers,
        "status": str(st),
        "failure": failure,
        "reason": cand.reason,
        "expected_family": cand.expected_family,
    }
    if failure or "ERROR" in status_text or "FAILED" in status_text or "COMPLETE" not in status_text:
        info["reject"] = "kernel not COMPLETE or has failure"
        return False, version, info
    if not markers.get("test_soundscapes") or not markers.get("sample_submission") or not markers.get("submission.csv"):
        info["reject"] = "source lacks required hidden-test/sample/submission markers"
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
        info["reject"] = "submission.csv missing from public session output"
        return False, version, info
    resp = requests.get(sub_url, timeout=120)
    resp.raise_for_status()
    ok, stats = validate_submission_csv(resp.text, cand)
    info["submission_stats"] = stats
    if not ok:
        info["reject"] = "submission.csv malformed, nonfinite, constant, or wrong columns"
        return False, version, info
    if stats["sha256_16"] in seen_hashes or stats["sha256_16"] in RECENT_SUBMITTED_DRYRUN_HASHES:
        info["reject"] = f"duplicate public dry-run hash {stats['sha256_16']}"
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
    ap.add_argument("--max-submissions", type=int, default=5)
    ap.add_argument("--artifact", default="artifacts/public_kernels_20260527_late_scout/submit_v626_v630_late_fill_20260527.json")
    args = ap.parse_args()

    _, competitions, kernels = make_clients()
    submissions = recent_submissions(competitions)
    existing_desc = {str(getattr(s, "description", "")) for s in submissions}
    count = today_count(submissions)
    now = dt.datetime.now(dt.timezone.utc)
    reset = now.replace(hour=0, minute=0, second=0, microsecond=0) + dt.timedelta(days=1)
    seen_hashes: set[str] = set()
    report = {
        "utc_now": now.isoformat(),
        "utc_reset": reset.isoformat(),
        "hours_to_reset": (reset - now).total_seconds() / 3600,
        "today_count_start": count,
        "latest_submissions": [
            {
                "date": sub_date_str(s),
                "description": str(getattr(s, "description", "")),
                "score": sub_score_str(s),
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
        item = {"label": cand.label, "ref": f"{cand.owner}/{cand.slug}", "description": cand.description, "expected_family": cand.expected_family}
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
            ok, version, info = preflight(kernels, cand, seen_hashes)
        except Exception as exc:
            item.update({"decision": "preflight_exception", "error": repr(exc)})
            print(json.dumps(item, default=str)[:3000], flush=True)
            report["candidates"].append(item)
            continue
        item.update({"preflight_ok": ok, "version": version, "info": info})
        if not ok or version is None:
            item["decision"] = "reject_preflight"
            print(json.dumps(item, default=str)[:3000], flush=True)
            report["candidates"].append(item)
            continue
        seen_hashes.add(info["submission_stats"]["sha256_16"])
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
