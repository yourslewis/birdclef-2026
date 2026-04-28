"""Poll v246 and submit completed kernel version to BirdCLEF 2026."""
import json
import os
import time
from kaggle.api.kaggle_api_extended import KaggleApi

with open(os.path.expanduser("~/.kaggle/kaggle.json"), "r") as f:
    os.environ["KAGGLE_API_TOKEN"] = json.load(f)["key"]

api = KaggleApi()
api.authenticate()
slug = "yourslewis/birdclef-2026-v246-ultra-sharp-temporal-smoothing"
kernel_version = int(os.environ.get("KAGGLE_KERNEL_VERSION", "1"))
print(f"Polling {slug} version {kernel_version}...")
while True:
    status = api.kernels_status(slug)
    status_text = str(getattr(status, "status", status))
    print("Status:", status, flush=True)
    if "COMPLETE" in status_text or "complete" in status_text.lower():
        print(f"Kernel complete; submitting version {kernel_version} to code competition...", flush=True)
        res = api.competition_submit_code(
            file_name="submission.csv",
            message="v246: v245 idea + ultra-sharp Gaussian temporal smoothing",
            competition="birdclef-2026",
            kernel=slug,
            kernel_version=kernel_version,
        )
        print("Submission result:", res, flush=True)
        break
    if "ERROR" in status_text or "error" in status_text.lower():
        print("Kernel failed; download logs before retrying.", flush=True)
        raise SystemExit(1)
    time.sleep(30)
