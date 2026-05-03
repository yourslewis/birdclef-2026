"""Poll v374 and submit completed kernel version to BirdCLEF 2026."""
import json, os, re, time
import requests
from kagglesdk.kaggle_http_client import KaggleHttpClient
from kagglesdk.competitions.services.competition_api_service import CompetitionApiClient
from kagglesdk.competitions.types.competition_api_service import ApiCreateCodeSubmissionRequest
from kagglesdk.kernels.services.kernels_api_service import KernelsApiClient
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest

with open(os.path.expanduser("~/.kaggle/kaggle.json")) as f:
    token=json.load(f)["key"]
competitions=CompetitionApiClient(KaggleHttpClient(api_token=token))
kernels=KernelsApiClient(KaggleHttpClient(api_token=token))
slug="yourslewis/bc26-v374-ew0575-g0825-ctx025-a0525"
kernel_version=int(os.environ.get("KAGGLE_KERNEL_VERSION","1"))
message="v374: immediate top-k + ProtoSSM EW0.575 + gamma 0.825 + context 0.25 + quantile alpha 0.525"

def quota_sleep_seconds(text):
    m=re.search(r"(\d+(?:\.\d+)?)\s+hours?\s+from now", text)
    if m: return max(300, int(float(m.group(1))*3600)+120)
    m=re.search(r"(\d+)\s+minutes?\s+from now", text)
    if m: return max(300, int(m.group(1))*60+120)
    return 3600

while True:
    owner, kslug = slug.split('/', 1)
    sreq=ApiGetKernelSessionStatusRequest(); sreq.user_name=owner; sreq.kernel_slug=kslug
    status=kernels.get_kernel_session_status(sreq); print("Status:", status, flush=True)
    if "COMPLETE" in str(status.status).upper():
        try:
            req=ApiCreateCodeSubmissionRequest(); req.competition_name="birdclef-2026"; req.kernel_owner=owner; req.kernel_slug=kslug; req.kernel_version=kernel_version; req.file_name="submission.csv"; req.submission_description=message
            print("Submission result:", competitions.create_code_submission(req), flush=True); break
        except requests.exceptions.HTTPError as exc:
            response=getattr(exc,"response",None); text=getattr(response,"text","") if response is not None else str(exc)
            print(f"Submission attempt failed: {type(exc).__name__}: {exc}", flush=True)
            if text: print(text[:2000], flush=True)
            if "daily Submission allowance" in text or ("daily" in text.lower() and "allowance" in text.lower()):
                sleep_s=quota_sleep_seconds(text); print(f"Daily submission allowance exhausted; sleeping {sleep_s} seconds before retry.", flush=True); time.sleep(sleep_s); continue
            raise
    if "ERROR" in str(status.status).upper(): raise SystemExit(1)
    time.sleep(30)
