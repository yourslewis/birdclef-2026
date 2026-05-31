#!/usr/bin/env python3
"""Submit v651/v652 EoS8 SED-heavy source-fork verifiers when runtime guards pass."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from kagglesdk.kaggle_http_client import KaggleHttpClient
from kagglesdk.competitions.services.competition_api_service import CompetitionApiClient
from kagglesdk.competitions.types.competition_api_service import ApiCreateCodeSubmissionRequest, ApiListSubmissionsRequest
from kagglesdk.kernels.services.kernels_api_service import KernelsApiClient
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest, ApiGetKernelSessionStatusRequest, ApiListKernelSessionOutputRequest

COMPETITION = "birdclef-2026"
MAX_DAILY = 5

@dataclass(frozen=True)
class Candidate:
    label: str
    slug: str
    description: str
    expected_family: str
    local_auc: float
    delta_vs_v616: float
    local_note: str

CANDIDATES = [
    Candidate(
        "v652", "bc26-v652-eos8-sed-heavy-proto040-verifier",
        "v652: EoS8 PowerOpt proto040 sed060 source verifier",
        "EoS8/PowerOptimization SED-heavy xSED moderate source fork",
        0.9942674902508776, 0.0007868226767386854,
        "Local reconstructed PowerOpt xSED proto040/sed060 improves v616 proxy +0.000787 with site q05 +0.000324; file q05 -0.000226.",
    ),
    Candidate(
        "v651", "bc26-v651-eos8-sed-heavy-proto020-verifier",
        "v651: EoS8 PowerOpt proto020 sed080 source verifier",
        "EoS8/PowerOptimization SED-heavy xSED aggressive source fork",
        0.9952101520329955, 0.0017294844588565672,
        "Local reconstructed PowerOpt xSED proto020/sed080 improves v616 proxy +0.001729 with site q05 +0.000487; file q05 -0.000105 but high movement.",
    ),
]

def make_clients():
    with open(os.path.expanduser("~/.kaggle/kaggle.json")) as f:
        token = json.load(f)["key"]
    http = KaggleHttpClient(api_token=token)
    return CompetitionApiClient(http), KernelsApiClient(http)

def sub_date(s) -> str:
    return str(getattr(s, "date", getattr(s, "date_nullable", "")))

def recent_submissions(comp: CompetitionApiClient):
    req = ApiListSubmissionsRequest(); req.competition_name = COMPETITION; req.page_size = 200
    return comp.list_submissions(req).submissions or []

def today_count(submissions) -> int:
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    return sum(sub_date(s).startswith(today) for s in submissions)

def output_files(kernels: KernelsApiClient, slug: str):
    req = ApiListKernelSessionOutputRequest(); req.user_name = "yourslewis"; req.kernel_slug = slug; req.page_size = 100
    return kernels.list_kernel_session_output(req).files or []

def fname(f) -> str:
    return getattr(f, "file_name", getattr(f, "fileName", ""))

def furl(f) -> str:
    return getattr(f, "url", "")

def get_status(kernels: KernelsApiClient, slug: str) -> dict[str, Any]:
    req = ApiGetKernelSessionStatusRequest(); req.user_name = "yourslewis"; req.kernel_slug = slug
    st = kernels.get_kernel_session_status(req)
    try:
        return st.to_dict()
    except Exception:
        try:
            return json.loads(str(st))
        except Exception:
            return {"raw": str(st)}

def get_version(kernels: KernelsApiClient, slug: str) -> int | None:
    req = ApiGetKernelRequest(); req.user_name = "yourslewis"; req.kernel_slug = slug
    k = kernels.get_kernel(req)
    md = k.metadata.to_dict()
    v = md.get("currentVersionNumber") or md.get("current_version_number")
    return int(v) if v is not None else None

def validate_csv(text: str) -> dict[str, Any]:
    rows = list(csv.reader(text.splitlines()))
    if not rows:
        return {"ok": False, "error": "empty"}
    ncols = len(rows[0]); vals=[]; ids=[]; bad=ragged=0
    for row in rows[1:]:
        if len(row) != ncols:
            ragged += 1; continue
        ids.append(row[0])
        for x in row[1:]:
            try: v=float(x)
            except Exception: bad += 1; continue
            if not math.isfinite(v): bad += 1
            else: vals.append(v)
    stats={
        "rows": len(rows)-1, "cols": ncols, "unique_rows": len(set(ids)), "bad_values": bad, "ragged_rows": ragged,
        "min": min(vals) if vals else None, "max": max(vals) if vals else None, "mean": sum(vals)/len(vals) if vals else None,
        "uniq_first100": len(set(round(v,8) for v in vals[:min(100,len(vals))])) if vals else 0,
        "hash": hashlib.sha256(text.encode()).hexdigest()[:16], "head_ids": ids[:3],
    }
    stats["ok"] = stats["rows"] > 0 and stats["cols"] == 235 and stats["unique_rows"] == stats["rows"] and bad == 0 and ragged == 0 and vals and stats["max"] > stats["min"]
    return stats

def preflight(kernels: KernelsApiClient, cand: Candidate) -> dict[str, Any]:
    st = get_status(kernels, cand.slug)
    status = str(st.get("status", "")).upper()
    item = {"status": st, "local_auc": cand.local_auc, "delta_vs_v616": cand.delta_vs_v616, "local_note": cand.local_note}
    if status != "COMPLETE" or st.get("failureMessage"):
        item["ok"] = False; item["reject"] = "kernel session not COMPLETE or has failure"; return item
    version = get_version(kernels, cand.slug); item["version"] = version
    files = output_files(kernels, cand.slug); item["files"] = sorted(fname(f) for f in files)
    sub_file = next((f for f in files if fname(f) == "submission.csv"), None)
    if sub_file is None:
        item["ok"] = False; item["reject"] = "submission.csv missing"; return item
    txt = requests.get(furl(sub_file), timeout=120).text
    stats = validate_csv(txt); item["submission_stats"] = stats
    item["ok"] = bool(stats.get("ok")) and version is not None
    if not item["ok"]: item["reject"] = "submission.csv failed schema/nonconstant verifier or missing version"
    return item

def submit(comp: CompetitionApiClient, cand: Candidate, version: int):
    req = ApiCreateCodeSubmissionRequest(); req.competition_name = COMPETITION
    req.kernel_owner = "yourslewis"; req.kernel_slug = cand.slug; req.kernel_version = int(version); req.file_name = "submission.csv"; req.submission_description = cand.description
    return comp.create_code_submission(req)

def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--submit", action="store_true"); ap.add_argument("--artifact", default="artifacts/source_winner_private_verifier_20260531T1816Z/submit_v651_v652_report.json")
    args = ap.parse_args()
    comp, kernels = make_clients(); submissions = recent_submissions(comp); count = today_count(submissions); existing = {str(getattr(s, "description", "")) for s in submissions}
    now = dt.datetime.now(dt.timezone.utc); reset = now.replace(hour=0, minute=0, second=0, microsecond=0) + dt.timedelta(days=1)
    report={"utc_now": now.isoformat(), "hours_to_reset": (reset-now).total_seconds()/3600, "today_count_start": count, "mode": "submit" if args.submit else "dry-run", "latest_submissions": [{"ref": getattr(s,"ref",None), "date": sub_date(s), "score": str(getattr(s,"public_score", "")), "status": str(getattr(s,"status", "")), "description": str(getattr(s,"description", ""))} for s in submissions[:15]], "candidates": [], "submitted": []}
    for cand in CANDIDATES:
        item={"label": cand.label, "slug": cand.slug, "description": cand.description, "expected_family": cand.expected_family}
        if count >= MAX_DAILY:
            item["decision"] = "skip_cap_reached"; report["candidates"].append(item); continue
        if cand.description in existing:
            item["decision"] = "skip_duplicate_description"; report["candidates"].append(item); continue
        try:
            pf = preflight(kernels, cand)
        except Exception as e:
            item.update({"decision":"preflight_exception", "error":repr(e)}); report["candidates"].append(item); continue
        item["preflight"] = pf
        if not pf.get("ok"):
            item["decision"] = "reject_preflight"; report["candidates"].append(item); continue
        if not args.submit:
            item["decision"] = "dry_run_would_submit"; report["candidates"].append(item); continue
        try:
            res = submit(comp, cand, int(pf["version"]))
            item["decision"] = "submitted"; item["submit_result"] = str(res); count += 1; existing.add(cand.description); report["submitted"].append(item)
        except requests.exceptions.HTTPError as exc:
            text = getattr(getattr(exc, "response", None), "text", "") or str(exc)
            item.update({"decision":"submit_http_error", "error": text[:2000]})
            report["candidates"].append(item)
            if "daily" in text.lower() and ("allowance" in text.lower() or "limit" in text.lower() or "quota" in text.lower()):
                break
            raise
        report["candidates"].append(item)
    report["today_count_end_est"] = count
    Path(args.artifact).parent.mkdir(parents=True, exist_ok=True)
    Path(args.artifact).write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str)[:8000])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
