#!/usr/bin/env python3
"""Verify completed Kaggle kernel output files via Bearer-backed Kaggle SDK.

This is intentionally read-only. It checks the latest session status, lists output
files, and optionally verifies log markers such as "Applied real SED bundle blend".
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from kagglesdk.kaggle_http_client import KaggleHttpClient
from kagglesdk.kernels.services.kernels_api_service import KernelsApiClient
from kagglesdk.kernels.types.kernels_api_service import (
    ApiGetKernelSessionStatusRequest,
    ApiListKernelSessionOutputRequest,
)


PRESETS: dict[str, dict[str, list[str] | str]] = {
    "v510-real-sed": {
        "slug": "bc26-v510-real-sed-bundle-blend-005",
        "required_files": ["submission.csv"],
        "required_log_markers": ["Applied real SED bundle blend", "submission.csv saved"],
    },
    "v560-direct-v2s": {
        "slug": "bc26-v560-public946-direct-v2s-r003",
        "required_files": [
            "submission.csv",
            "submission_direct_v2s_student.csv",
            "submission_sed.csv",
            "submission_protossm.csv",
        ],
        "required_log_markers": [
            "Direct V2S student sidecar complete",
            "Applied Direct V2S student rank sidecar blend",
        ],
    },
}


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def _load_token(path: str) -> str:
    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        data = json.load(f)
    token = data.get("key")
    if not token:
        raise SystemExit(f"No Kaggle API key found in {path}")
    return token


def verify_kernel(
    owner: str,
    slug: str,
    required_files: list[str],
    required_log_markers: list[str],
    kaggle_json: str,
    page_size: int,
) -> dict[str, Any]:
    client = KernelsApiClient(KaggleHttpClient(api_token=_load_token(kaggle_json)))

    status_req = ApiGetKernelSessionStatusRequest()
    status_req.user_name = owner
    status_req.kernel_slug = slug
    status = client.get_kernel_session_status(status_req)

    output_req = ApiListKernelSessionOutputRequest()
    output_req.user_name = owner
    output_req.kernel_slug = slug
    output_req.page_size = page_size
    output = client.list_kernel_session_output(output_req)

    files = []
    for item in getattr(output, "files", []) or []:
        files.append(_get(item, "file_name", "fileName", default=str(item)))
    log = _get(output, "log", default="") or ""

    missing_files = [name for name in required_files if name not in files]
    missing_markers = [marker for marker in required_log_markers if marker not in log]
    status_text = str(_get(status, "status", default=status))
    failure_message = _get(status, "failure_message", "failureMessage", default=None)

    ok = not missing_files and not missing_markers and "COMPLETE" in status_text.upper() and not failure_message
    return {
        "owner": owner,
        "slug": slug,
        "ok": ok,
        "status": status_text,
        "failure_message": failure_message,
        "files": files,
        "missing_files": missing_files,
        "required_log_markers": required_log_markers,
        "missing_log_markers": missing_markers,
        "log_chars": len(log),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slug", nargs="?", help="Kernel slug, e.g. bc26-v510-real-sed-bundle-blend-005")
    parser.add_argument("--preset", choices=sorted(PRESETS), help="Use known required files/log markers for a tracked kernel")
    parser.add_argument("--all-presets", action="store_true", help="Verify every known preset and return nonzero if any fail")
    parser.add_argument("--owner", default="yourslewis")
    parser.add_argument("--require", action="append", default=[], help="Required output file; repeatable")
    parser.add_argument("--log-contains", action="append", default=[], help="Required substring in kernel log; repeatable")
    parser.add_argument("--kaggle-json", default="~/.kaggle/kaggle.json")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    if args.all_presets:
        if args.slug or args.preset:
            parser.error("--all-presets cannot be combined with slug or --preset")
        results = []
        for name, preset in PRESETS.items():
            required_files = _dedupe(list(preset.get("required_files", ["submission.csv"])) + args.require)
            required_markers = _dedupe(list(preset.get("required_log_markers", [])) + args.log_contains)
            result = verify_kernel(args.owner, str(preset["slug"]), required_files, required_markers, args.kaggle_json, args.page_size)
            result["preset"] = name
            results.append(result)
        payload = {"ok": all(item["ok"] for item in results), "results": results}
        print(json.dumps(payload, indent=2 if args.pretty else None, sort_keys=True))
        return 0 if payload["ok"] else 1

    preset = PRESETS.get(args.preset or "", {})
    slug = args.slug or preset.get("slug")
    if not slug:
        parser.error("slug is required unless --preset supplies one")

    required_files = _dedupe(list(preset.get("required_files", ["submission.csv"])) + args.require)
    required_markers = _dedupe(list(preset.get("required_log_markers", [])) + args.log_contains)

    result = verify_kernel(args.owner, str(slug), required_files, required_markers, args.kaggle_json, args.page_size)
    if args.preset:
        result["preset"] = args.preset
    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
