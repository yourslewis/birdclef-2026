"""Poll v275 and submit completed kernel version to BirdCLEF 2026.

Handles Kaggle daily allowance errors by retrying near the reset time.
"""
import json
import os
import re
import time
from kaggle.api.kaggle_api_extended import KaggleApi

with open(os.path.expanduser("~/.kaggle/kaggle.json"), "r") as f:
    os.environ["KAGGLE_API_TOKEN"] = json.load(f)["key"]

api = KaggleApi()
api.authenticate()
slug = "yourslewis/birdclef-2026-v275-immediate-quantile0475"
kernel_version = int(os.environ.get("KAGGLE_KERNEL_VERSION", "1"))
message = "v275: immediate-only temporal smoothing + quantile mix alpha 0.475"

def quota_sleep_seconds(text: str) -> int:
    match = re.search(r"(\d+(?:\.\d+)?)\s+hours?\s+from now", text)
    if match:
        return max(300, int(float(match.group(1)) * 3600) + 120)
    match = re.search(r"(\d+)\s+minutes?\s+from now", text)
    if match:
        return max(300, int(match.group(1)) * 60 + 120)
    return 3600

print(f"Polling {slug} version {kernel_version}...")
while True:
    status = api.kernels_status(slug)
    status_text = str(getattr(status, "status", status))
    print("Status:", status, flush=True)
    if "COMPLETE" in status_text or "complete" in status_text.lower():
        print(f"Kernel complete; submitting version {kernel_version} to code competition...", flush=True)
        try:
            res = api.competition_submit_code(
                file_name="submission.csv",
                message=message,
                competition="birdclef-2026",
                kernel=slug,
                kernel_version=kernel_version,
            )
            print("Submission result:", res, flush=True)
            break
        except Exception as exc:
            response = getattr(exc, "response", None)
            text = getattr(response, "text", "") if response is not None else ""
            print(f"Submission attempt failed: {type(exc).__name__}: {exc}", flush=True)
            if text:
                print(text[:2000], flush=True)
            if "daily Submission allowance" in text or ("daily" in text.lower() and "allowance" in text.lower()):
                sleep_s = quota_sleep_seconds(text)
                print(f"Daily submission allowance exhausted; sleeping {sleep_s} seconds before retry.", flush=True)
                time.sleep(sleep_s)
                continue
            raise
    if "ERROR" in status_text or "error" in status_text.lower():
        print("Kernel failed; download logs before retrying.", flush=True)
        raise SystemExit(1)
    time.sleep(30)
