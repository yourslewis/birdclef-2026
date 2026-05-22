"""Submit the remaining most-promising BirdCLEF 0.96-frontier public candidates.

Created after v598 scored 0.860 and the user explicitly asked to test the
promising remaining experiments. The batch is still guarded: each candidate must
be COMPLETE, expose a finite/non-constant submission.csv, and not duplicate an
existing submission description. It submits up to remaining daily slots.
"""
from __future__ import annotations

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
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest, ApiListKernelSessionOutputRequest

COMPETITION = "birdclef-2026"
MAX_DAILY = 5
T = TypeVar("T")


@dataclass(frozen=True)
class Candidate:
    label: str
    owner: str
    slug: str
    description: str
    min_unique_first100: int = 2


CANDIDATES: list[Candidate] = [
    Candidate(
        "v599",
        "claudedevore",
        "birdclef-2026-r0952-run2-sidecar-submit",
        "v599: Guarded direct Claudedevore R0952 run2 sidecar",
    ),
    Candidate(
        "v600",
        "gendaijin",
        "birdclef2026-day0522-pilkwang-new",
        "v600: Guarded direct Gendaijin Pilkwang prior-field fusion",
    ),
    Candidate(
        "v601",
        "gendaijin",
        "birdclef2026-day0522-meenal-new",
        "v601: Guarded direct Gendaijin Meenal new visual prior",
    ),
    Candidate(
        "v602",
        "nicolasschuldt",
        "nfnet-aves-lprior075",
        "v602: Guarded direct Nicolas NFNet Aves lprior075",
    ),
    # Fallback if any of the above fail preflight before slots are filled.
    Candidate(
        "v603",
        "gendaijin",
        "birdclef2026-day0522-anthony-s124",
        "v603: Guarded direct Gendaijin Anthony S124 blend",
    ),
    Candidate(
        "v604",
        "meenalsinha",
        "birdclef-2026-improved",
        "v604: Guarded direct Meenal improved v25 visual prior",
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
        raise RuntimeError(f"{label} HTTPError: {exc} {text[:1000]}") from exc


def recent_submissions(competitions: CompetitionApiClient):
    req = ApiListSubmissionsRequest()
    req.competition_name = COMPETITION
    req.page_size = 200
    return call("list_submissions", lambda: competitions.list_submissions(req)).submissions or []


def today_count(submissions) -> int:
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    return sum(str(getattr(s, "date", getattr(s, "date_nullable", ""))).startswith(today) for s in submissions)


def pull_version(token: str, cand: Candidate) -> tuple[int, int]:
    url = f"https://www.kaggle.com/api/v1/kernels/pull/{cand.owner}/{cand.slug}"
    r = requests.get(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"}, timeout=120)
    r.raise_for_status()
    data = r.json()
    meta = data.get("metadata", {})
    version = meta.get("currentVersionNumberNullable") or meta.get("currentVersionNumber")
    source = (data.get("blob") or {}).get("source") or ""
    return int(version), len(source)


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


def validate_submission_csv(text: str) -> tuple[bool, dict]:
    rows = list(csv.reader(text.splitlines()))
    if not rows:
        return False, {"error": "empty csv"}
    n_rows = len(rows) - 1
    n_cols = len(rows[0])
    vals: list[float] = []
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
    uniq_first100 = len(set(round(v, 8) for v in clean[: min(100, len(clean))]))
    stats = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "bad_values": bad,
        "min": min(clean) if clean else None,
        "max": max(clean) if clean else None,
        "zeros": sum(1 for v in clean if v == 0.0),
        "uniq_first100": uniq_first100,
    }
    ok = n_rows > 0 and n_cols == 235 and bad == 0 and bool(clean) and stats["max"] > stats["min"] and uniq_first100 >= 2
    return ok, stats


def preflight(token: str, kernels: KernelsApiClient, cand: Candidate) -> tuple[bool, int | None, dict]:
    version, source_len = pull_version(token, cand)
    status = kernel_status(kernels, cand)
    status_text = str(getattr(status, "status", status)).upper()
    failure = getattr(status, "failure_message", getattr(status, "failureMessage", ""))
    if failure or "ERROR" in status_text or "FAILED" in status_text:
        return False, version, {"source_len": source_len, "status": str(status), "failure": failure}
    if "COMPLETE" not in status_text:
        return False, version, {"source_len": source_len, "status": str(status), "failure": "not complete"}
    out = kernel_output(kernels, cand)
    files = sorted({file_name(f) for f in (getattr(out, "files", []) or [])})
    sub_url = ""
    for f in getattr(out, "files", []) or []:
        if file_name(f) == "submission.csv":
            sub_url = file_url(f)
            break
    if not sub_url:
        return False, version, {"source_len": source_len, "files": files, "failure": "no submission.csv url"}
    resp = requests.get(sub_url, timeout=120)
    resp.raise_for_status()
    ok, stats = validate_submission_csv(resp.text)
    return ok, version, {"source_len": source_len, "files": files, "stats": stats}


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
    token, competitions, kernels = make_clients()
    submissions = recent_submissions(competitions)
    existing_desc = {str(s.description) for s in submissions}
    count = today_count(submissions)
    print("utc_now", dt.datetime.now(dt.timezone.utc).isoformat(), flush=True)
    print("today_count_start", count, flush=True)
    submitted = 0
    for cand in CANDIDATES:
        if count >= MAX_DAILY:
            print("daily cap reached; stopping", flush=True)
            break
        print(f"\n=== {cand.label} {cand.owner}/{cand.slug}", flush=True)
        print("description", cand.description, flush=True)
        if cand.description in existing_desc:
            print("skip duplicate", flush=True)
            continue
        try:
            ok, version, info = preflight(token, kernels, cand)
        except Exception as exc:
            print("preflight exception", repr(exc), flush=True)
            continue
        print("preflight_ok", ok, "version", version, "info", json.dumps(info, default=str)[:3000], flush=True)
        if not ok or version is None:
            continue
        try:
            result = submit(competitions, cand, version)
            print("submitted", result, flush=True)
            submitted += 1
            count += 1
            existing_desc.add(cand.description)
        except requests.exceptions.HTTPError as exc:
            text = getattr(getattr(exc, "response", None), "text", "") or str(exc)
            print("submit failed", text[:2000], flush=True)
            if "daily" in text.lower() and ("allowance" in text.lower() or "limit" in text.lower() or "quota" in text.lower()):
                break
            raise
    print("submitted_count", submitted, "today_count_end_est", count, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
