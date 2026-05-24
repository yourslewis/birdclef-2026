"""Use expiring 2026-05-24 BirdCLEF slots for legitimate exploratory candidates.

These are not leaderboard probes. Each candidate is a real public/model/source
hypothesis with hidden-test handling markers and COMPLETE public runs. They were
previously below the 0.960 promotion bar, but Wenhao asked to use otherwise
expiring slots properly for modeling signal.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

import requests
from kagglesdk.kaggle_http_client import KaggleHttpClient
from kagglesdk.competitions.services.competition_api_service import CompetitionApiClient
from kagglesdk.competitions.types.competition_api_service import ApiCreateCodeSubmissionRequest, ApiListSubmissionsRequest
from kagglesdk.kernels.services.kernels_api_service import KernelsApiClient
from kagglesdk.kernels.types.kernels_api_service import (
    ApiGetKernelRequest,
    ApiGetKernelSessionStatusRequest,
    ApiListKernelSessionOutputRequest,
)

COMPETITION = "birdclef-2026"
TODAY_UTC = dt.datetime.now(dt.timezone.utc).date().isoformat()
MAX_DAILY = 5
T = TypeVar("T")


@dataclass(frozen=True)
class Candidate:
    tag: str
    owner: str
    slug: str
    version: int
    description: str
    rationale: str
    required_files: tuple[str, ...]
    marker_files: tuple[str, ...]
    source_markers: tuple[str, ...]
    allow_sample_shaped_final: bool = True


CANDIDATES = [
    Candidate(
        tag="v613",
        owner="alexycactus",
        slug="birdclef-2026-ns1-ensemble",
        version=1,
        description="v613: Exploratory direct Alexy NS1 CNN noisy-student source",
        rationale="Distinct CNN/noisy-student sidecar family; public direct source is below frontier but can reveal whether this family transfers at all.",
        required_files=("submission.csv", "submission_no_postproc.csv", "diagnostics_nb21.json"),
        marker_files=("submission.csv",),
        source_markers=("test_soundscapes", "staging fallback", "submission_no_postproc.csv", "TOP_K", "onnxruntime"),
        allow_sample_shaped_final=True,
    ),
    Candidate(
        tag="v614",
        owner="raunakdey07",
        slug="birdclef-2026-v9",
        version=4,
        description="v614: Exploratory direct Raunak v9 ProtoSSM SED source",
        rationale="Legitimate Model_7/ProtoSSM/SED public family with hidden-test handling; useful final-slot baseline even if likely plateau.",
        required_files=("submission.csv", "submission_protossm.csv", "submission_sed.csv"),
        marker_files=("submission_protossm.csv", "submission_sed.csv"),
        source_markers=("test_soundscapes", "sample_submission", "Dry-run detected", "submission_protossm.csv", "submission_sed.csv"),
        allow_sample_shaped_final=True,
    ),
    Candidate(
        tag="v615",
        owner="jungchanryu",
        slug="birdclef-first",
        version=19,
        description="v615: Exploratory direct Jungchan CT-MoBE branch source",
        rationale="CT-MoBE/Model21 plus ProtoSSM/SED branches; saturated lineage risk, but has branch diversity and hidden-test handling.",
        required_files=("submission.csv", "subm_21.csv", "submission_protossm.csv", "submission_sed.csv"),
        marker_files=("subm_21.csv", "submission_protossm.csv", "submission_sed.csv"),
        source_markers=("test_soundscapes", "sample_submission", "Dry-run detected", "subm_21.csv", "submission_protossm.csv", "submission_sed.csv"),
        allow_sample_shaped_final=True,
    ),
]


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
    req = ApiListSubmissionsRequest(); req.competition_name = COMPETITION; req.page_size = 200
    return call("list_submissions", lambda: competitions.list_submissions(req)).submissions or []


def count_today(submissions) -> int:
    return sum(str(getattr(s, "date", getattr(s, "date_nullable", ""))).startswith(TODAY_UTC) for s in submissions)


def decode_source(src: str) -> str:
    try:
        nb = json.loads(src)
        if isinstance(nb, dict) and isinstance(nb.get("cells"), list):
            return "\n".join(
                "".join(c.get("source", "")) if isinstance(c.get("source", ""), list) else str(c.get("source", ""))
                for c in nb["cells"]
            )
    except Exception:
        pass
    return src


def csv_stats(text: str):
    rows = list(csv.reader(text.splitlines()))
    if not rows:
        return {"empty": True}
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
    clean = [x for x in vals if not (math.isnan(x) or math.isinf(x))]
    bad = len(vals) - len(clean)
    return {
        "rows": len(rows) - 1,
        "cols": len(rows[0]),
        "unique_rows": len(set(row_ids)),
        "bad": bad,
        "min": min(clean) if clean else None,
        "max": max(clean) if clean else None,
        "mean": sum(clean) / len(clean) if clean else None,
        "uniq_round_10k": len(set(round(x, 6) for x in clean[:10000])),
    }


def source_preflight(kernels, c: Candidate) -> bool:
    req = ApiGetKernelRequest(); req.user_name = c.owner; req.kernel_slug = c.slug
    ker = call(f"get_kernel_{c.tag}", lambda: kernels.get_kernel(req))
    d = ker.to_dict(); md = d.get("metadata", {})
    got_version = int(md.get("currentVersionNumber") or md.get("currentVersionNumberNullable") or 0)
    print(c.tag, "metadata", {"ref": md.get("ref"), "version": got_version, "lastRun": md.get("lastRunTime"), "private": md.get("isPrivate")}, flush=True)
    if got_version != c.version:
        print(c.tag, f"version drift expected {c.version} got {got_version}", flush=True)
        return False
    source = decode_source((d.get("blob") or {}).get("source") or "")
    missing = [m for m in c.source_markers if m not in source]
    if missing:
        print(c.tag, "missing source markers", missing, flush=True)
        return False
    return True


def output_preflight(kernels, c: Candidate) -> bool:
    sreq = ApiGetKernelSessionStatusRequest(); sreq.user_name = c.owner; sreq.kernel_slug = c.slug
    status = call(f"status_{c.tag}", lambda: kernels.get_kernel_session_status(sreq))
    st = str(getattr(status, "status", status)).upper()
    failure = getattr(status, "failure_message", getattr(status, "failureMessage", ""))
    print(c.tag, "status", status, flush=True)
    if failure or "ERROR" in st or "FAILED" in st or "COMPLETE" not in st:
        print(c.tag, "bad status/failure", failure, flush=True)
        return False
    oreq = ApiListKernelSessionOutputRequest(); oreq.user_name = c.owner; oreq.kernel_slug = c.slug; oreq.page_size = 100
    out = call(f"outputs_{c.tag}", lambda: kernels.list_kernel_session_output(oreq))
    files = {getattr(f, "file_name", getattr(f, "fileName", "")): f for f in (getattr(out, "files", []) or [])}
    print(c.tag, "files", sorted(files), flush=True)
    missing = [x for x in c.required_files if x not in files]
    if missing:
        print(c.tag, "missing files", missing, flush=True)
        return False
    log = getattr(out, "log", "") or ""
    if "Traceback" in log or "NotebookThrewException" in log:
        print(c.tag, "traceback marker in log", flush=True)
        return False
    stats = {}
    for name in set(("submission.csv",) + c.marker_files):
        if name not in files:
            continue
        url = getattr(files[name], "url", None) or getattr(files[name], "_url", None)
        text = call(f"download_{c.tag}_{name}", lambda: requests.get(url, timeout=60).text)
        stats[name] = csv_stats(text)
    print(c.tag, "stats", json.dumps(stats, indent=2), flush=True)
    final = stats.get("submission.csv") or {}
    if final.get("cols") != 235 or final.get("bad", 1) or final.get("unique_rows") != final.get("rows"):
        print(c.tag, "final failed numeric/schema guard", flush=True)
        return False
    if final.get("uniq_round_10k", 0) <= 10:
        # Jungchan final public dry-run is sample-shaped constant, but its marker branches are full/nonconstant.
        print(c.tag, "final public dry-run is low-diversity; requiring marker branches", flush=True)
    for name in c.marker_files:
        stt = stats.get(name)
        if not stt:
            continue
        if stt.get("cols") != 235 or stt.get("bad", 1) or stt.get("unique_rows") != stt.get("rows"):
            print(c.tag, name, "failed marker schema/numeric guard", flush=True)
            return False
        if stt.get("uniq_round_10k", 0) <= 100:
            print(c.tag, name, "too constant", flush=True)
            return False
    os.makedirs("logs", exist_ok=True)
    with open(f"logs/{c.tag}_exploratory_preflight_20260524.json", "w") as f:
        json.dump({"candidate": c.__dict__, "stats": stats, "files": sorted(files), "log_tail": log[-8000:]}, f, indent=2)
    return True


def submit(competitions, c: Candidate):
    req = ApiCreateCodeSubmissionRequest()
    req.competition_name = COMPETITION
    req.kernel_owner = c.owner
    req.kernel_slug = c.slug
    req.kernel_version = c.version
    req.file_name = "submission.csv"
    req.submission_description = c.description
    return competitions.create_code_submission(req)


def main():
    _token, competitions, kernels = make_clients()
    submitted = []
    for c in CANDIDATES:
        submissions = recent_submissions(competitions)
        today = count_today(submissions)
        print("\n===", c.tag, c.owner + "/" + c.slug, "today", today, "/", MAX_DAILY, "===", flush=True)
        if today >= MAX_DAILY:
            print("daily cap reached; stop", flush=True)
            break
        if any(str(getattr(s, "description", "")) == c.description for s in submissions):
            print(c.tag, "already submitted; skip", flush=True)
            continue
        if not source_preflight(kernels, c):
            print(c.tag, "source preflight failed; skip", flush=True)
            continue
        if not output_preflight(kernels, c):
            print(c.tag, "output preflight failed; skip", flush=True)
            continue
        print(c.tag, "SUBMIT", c.description, "reason:", c.rationale, flush=True)
        res = submit(competitions, c)
        print(c.tag, "submission result", res, flush=True)
        submitted.append(c.tag)
    print("submitted", submitted, flush=True)


if __name__ == "__main__":
    main()
