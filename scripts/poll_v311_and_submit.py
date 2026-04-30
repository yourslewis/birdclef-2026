"""Poll v311 and submit completed kernel version to BirdCLEF 2026."""
import json, os, re, time
from kaggle.api.kaggle_api_extended import KaggleApi
with open(os.path.expanduser("~/.kaggle/kaggle.json")) as f:
    os.environ["KAGGLE_API_TOKEN"] = json.load(f)["key"]
api=KaggleApi(); api.authenticate()
slug="yourslewis/bc26-v311-immediate-topk-alpha045"
kernel_version=int(os.environ.get("KAGGLE_KERNEL_VERSION","1"))
message="v311: immediate top-k contrast + rank-heavy quantile alpha 0.45"
def quota_sleep_seconds(text):
    m=re.search(r"(\d+(?:\.\d+)?)\s+hours?\s+from now", text)
    if m: return max(300, int(float(m.group(1))*3600)+120)
    m=re.search(r"(\d+)\s+minutes?\s+from now", text)
    if m: return max(300, int(m.group(1))*60+120)
    return 3600
while True:
    status=api.kernels_status(slug); status_text=str(getattr(status,"status",status)); print("Status:", status, flush=True)
    if "COMPLETE" in status_text.upper():
        try:
            print("Submission result:", api.competition_submit_code(file_name="submission.csv", message=message, competition="birdclef-2026", kernel=slug, kernel_version=kernel_version), flush=True); break
        except Exception as exc:
            response=getattr(exc,"response",None); text=getattr(response,"text","") if response is not None else ""
            print(f"Submission attempt failed: {type(exc).__name__}: {exc}", flush=True)
            if text: print(text[:2000], flush=True)
            if "daily Submission allowance" in text or ("daily" in text.lower() and "allowance" in text.lower()):
                sleep_s=quota_sleep_seconds(text); print(f"Daily submission allowance exhausted; sleeping {sleep_s} seconds before retry.", flush=True); time.sleep(sleep_s); continue
            raise
    if "ERROR" in status_text.upper(): raise SystemExit(1)
    time.sleep(30)
