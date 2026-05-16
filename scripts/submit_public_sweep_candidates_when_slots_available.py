#!/usr/bin/env python3
"""Submit prioritized public-sweep kernels when daily BirdCLEF slots are available.

This is intentionally conservative about *which* public kernels it will submit:
- it excludes known output-only/invalid-format public ensemble kernels such as
  Lucataco score-desc;
- it requires the source kernel to be COMPLETE and expose submission.csv;
- it duplicate-guards by submission description;
- it stops on daily-cap errors and defaults to dry-run unless --submit is set.

The candidate queue is for exploratory LB datapoints after v561-v565.  It should
only be used after the UTC daily reset, or with --dry-run for planning.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Callable, TypeVar

import requests
from kagglesdk.kaggle_http_client import KaggleHttpClient
from kagglesdk.competitions.services.competition_api_service import CompetitionApiClient
from kagglesdk.competitions.types.competition_api_service import ApiCreateCodeSubmissionRequest, ApiListSubmissionsRequest
from kagglesdk.kernels.services.kernels_api_service import KernelsApiClient
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest, ApiListKernelFilesRequest

COMPETITION = "birdclef-2026"
T = TypeVar("T")


@dataclass(frozen=True)
class Candidate:
    label: str
    owner: str
    slug: str
    version: int
    hypothesis: str

    @property
    def description(self) -> str:
        return f"{self.label}: {self.hypothesis}"


# Ordered for next UTC reset.  These are all public kernels whose source appears
# to run full inference (Perch/SED/BirdNET), unlike v561's output-only rank blend.
CANDIDATES: list[Candidate] = [
    Candidate(
        "v566",
        "kruzzcc",
        "bc26-nina-eos4-fixed",
        2,
        "Sweep Nina EoS4 fixed plus BirdNET public kernel direct",
    ),
    Candidate(
        "v567",
        "kruzzcc",
        "bc26-mtoshi-umap-bn-a",
        1,
        "Sweep Mtoshi UMAP plus BirdNET public kernel direct",
    ),
    Candidate(
        "v568",
        "meenalsinha",
        "birdclef-2026-improved",
        9,
        "Sweep Meenal improved BirdNET public kernel direct",
    ),
    Candidate(
        "v569",
        "pilkwang",
        "birdclef-2026-safe-ensemble",
        4,
        "Sweep safe ensemble public kernel direct",
    ),
    Candidate(
        "v570",
        "mtoshidesu",
        "lb-improved",
        5,
        "Sweep Mtoshi improved public kernel direct",
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
        raise RuntimeError(f"{label} HTTPError: {exc} {text[:1000]}") from exc


def recent_submissions(competitions: CompetitionApiClient):
    req = ApiListSubmissionsRequest()
    req.competition_name = COMPETITION
    req.page_size = 200
    return call("list_submissions", lambda: competitions.list_submissions(req)).submissions or []


def candidate_complete_with_submission(kernels: KernelsApiClient, cand: Candidate) -> tuple[bool, str]:
    sr = ApiGetKernelSessionStatusRequest()
    sr.user_name = cand.owner
    sr.kernel_slug = cand.slug
    status = call(f"kernel_status {cand.owner}/{cand.slug}", lambda: kernels.get_kernel_session_status(sr))
    status_text = str(getattr(status, "status", status)).upper()
    if "COMPLETE" not in status_text:
        return False, f"not COMPLETE: {status}"
    fr = ApiListKernelFilesRequest()
    fr.user_name = cand.owner
    fr.kernel_slug = cand.slug
    fr.page_size = 100
    files = call(f"list_files {cand.owner}/{cand.slug}", lambda: kernels.list_kernel_files(fr)).files or []
    names = sorted({getattr(f, "name", "") for f in files})
    if "submission.csv" not in names:
        return False, f"submission.csv missing; files={names}"
    return True, f"COMPLETE files={names}"


def create_submission(competitions: CompetitionApiClient, cand: Candidate):
    req = ApiCreateCodeSubmissionRequest()
    req.competition_name = COMPETITION
    req.kernel_owner = cand.owner
    req.kernel_slug = cand.slug
    req.kernel_version = cand.version
    req.file_name = "submission.csv"
    req.submission_description = cand.description
    return competitions.create_code_submission(req)


def is_daily_cap_error(text: str) -> bool:
    low = text.lower()
    return "daily" in low and ("allowance" in low or "quota" in low or "limit" in low)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true", help="Actually submit. Default is dry-run.")
    ap.add_argument("--max-submissions", type=int, default=3, help="Maximum candidates to submit this run")
    args = ap.parse_args()

    competitions, kernels = make_clients()
    submissions = recent_submissions(competitions)
    existing_desc = {str(s.description) for s in submissions}
    print("Latest submissions:")
    for s in submissions[:10]:
        print(json.dumps({
            "date": str(getattr(s, "date", "")),
            "description": str(getattr(s, "description", "")),
            "score": str(getattr(s, "public_score", getattr(s, "publicScore", ""))),
            "status": str(getattr(s, "status", "")),
            "ref": getattr(s, "ref", None),
        }, default=str))

    submitted = 0
    for cand in CANDIDATES:
        print(f"\n=== {cand.label} {cand.owner}/{cand.slug} v{cand.version}")
        print("description:", cand.description)
        if cand.description in existing_desc:
            print("skip: already submitted")
            continue
        ok, reason = candidate_complete_with_submission(kernels, cand)
        print("preflight:", reason)
        if not ok:
            continue
        if not args.submit:
            print("dry-run: would submit")
            continue
        if submitted >= args.max_submissions:
            print(f"stop: reached --max-submissions={args.max_submissions}")
            break
        try:
            result = create_submission(competitions, cand)
            print("submitted:", result)
            existing_desc.add(cand.description)
            submitted += 1
        except requests.exceptions.HTTPError as exc:
            text = getattr(getattr(exc, "response", None), "text", "") or str(exc)
            print("submit failed:", text[:2000])
            if is_daily_cap_error(text):
                print("daily cap hit; stop and retry after reset")
                break
            raise
    print(f"\nsubmitted_count={submitted} mode={'submit' if args.submit else 'dry-run'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
