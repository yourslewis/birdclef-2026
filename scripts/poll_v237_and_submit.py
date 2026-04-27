"""Poll v237 and submit completed kernel version to BirdCLEF 2026."""
import json
import os
import time
from kaggle.api.kaggle_api_extended import KaggleApi

with open(os.path.expanduser("~/.kaggle/kaggle.json"), "r") as f:
    os.environ["KAGGLE_API_TOKEN"] = json.load(f)["key"]

api = KaggleApi()
api.authenticate()
slug = "yourslewis/birdclef-2026-v237-weighted-ensemble"
# Version returned by the latest successful kernels_push during this fix.
kernel_version = 3
print(f"Polling {slug} version {kernel_version}...")
while True:
    status = api.kernels_status(slug)
    status_text = str(getattr(status, "status", status))
    print("Status:", status)
    if "COMPLETE" in status_text or "complete" in status_text.lower():
        print("Kernel complete; submitting version 1 to code competition...")
        res = api.competition_submit_code(
            file_name="submission.csv",
            message="v237: weighted ensemble ew0.6 + widened Gaussian smoothing; fixed data/model mounts",
            competition="birdclef-2026",
            kernel=slug,
            kernel_version=kernel_version,
        )
        print("Submission result:", res)
        break
    if "ERROR" in status_text or "error" in status_text.lower():
        print("Kernel failed; download logs before retrying.")
        raise SystemExit(1)
    time.sleep(30)
