"""Submit pending completed BirdCLEF kernels in order, with quota retry.

This avoids losing completed kernels when Kaggle's daily code-submission cap is hit.
It skips any message already visible in recent submissions.
"""
import json
import os
import time
from kaggle.api.kaggle_api_extended import KaggleApi

PENDING = [
    {
        "name": "v247",
        "kernel": "yourslewis/birdclef-2026-v247-minimal-temporal-smoothing",
        "version": 1,
        "message": "v247: v245/v246 axis + minimal temporal smoothing center 0.90",
    },
    {
        "name": "v248",
        "kernel": "yourslewis/birdclef-2026-v248-v245-gamma080",
        "version": 1,
        "message": "v248: v245 smoothing + power gamma 0.80",
    },
    {
        "name": "v249",
        "kernel": "yourslewis/birdclef-2026-v249-v245-gamma090",
        "version": 1,
        "message": "v249: v245 smoothing + power gamma 0.90",
    },
    {
        "name": "v250",
        "kernel": "yourslewis/birdclef-2026-v250-immediate-temporal-smoothing",
        "version": 1,
        "message": "v250: v245 center mass + immediate-only temporal smoothing",
    },
    {
        "name": "v251",
        "kernel": "yourslewis/birdclef-2026-v251-v245-context015",
        "version": int(os.environ.get("KAGGLE_KERNEL_VERSION", "1")),
        "message": "v251: v245 smoothing + gentler file context alpha 0.15",
    },
]

with open(os.path.expanduser("~/.kaggle/kaggle.json"), "r") as f:
    os.environ["KAGGLE_API_TOKEN"] = json.load(f)["key"]
api = KaggleApi(); api.authenticate()

def recent_messages():
    return {str(getattr(s, "description", "")) for s in api.competition_submissions("birdclef-2026")[:50]}

def is_complete(kernel):
    status = api.kernels_status(kernel)
    print(f"Kernel status {kernel}: {status}", flush=True)
    return "COMPLETE" in str(getattr(status, "status", status)).upper()

while True:
    messages = recent_messages()
    progressed = False
    all_done = True
    for item in PENDING:
        if item["message"] in messages:
            print(f"{item['name']} already submitted; skipping.", flush=True)
            continue
        all_done = False
        if not is_complete(item["kernel"]):
            print(f"{item['name']} not complete yet; sleeping 10 minutes.", flush=True)
            time.sleep(600)
            progressed = True
            break
        print(f"Submitting {item['name']} kernel version {item['version']}...", flush=True)
        try:
            res = api.competition_submit_code(
                file_name="submission.csv",
                message=item["message"],
                competition="birdclef-2026",
                kernel=item["kernel"],
                kernel_version=item["version"],
            )
            print("Submission result:", res, flush=True)
            progressed = True
            time.sleep(30)
            break
        except Exception as exc:
            response = getattr(exc, "response", None)
            text = getattr(response, "text", "") if response is not None else ""
            print(f"Submission attempt failed for {item['name']}: {type(exc).__name__}: {exc}", flush=True)
            if text:
                print(text[:2000], flush=True)
            if "daily Submission allowance" in text or ("daily" in text.lower() and "allowance" in text.lower()):
                print("Daily submission allowance exhausted; sleeping 1 hour before retry.", flush=True)
                time.sleep(3600)
                progressed = True
                break
            raise
    if all_done:
        print("All pending kernels are already submitted.", flush=True)
        break
    if not progressed:
        time.sleep(600)
