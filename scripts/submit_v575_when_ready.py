"""Poll and submit v575 repo-owned EoS5 confirmation kernel once ready."""
from __future__ import annotations

import json
import os
import re
import time
from typing import Callable, TypeVar

import requests
from kagglesdk.kaggle_http_client import KaggleHttpClient
from kagglesdk.competitions.services.competition_api_service import CompetitionApiClient
from kagglesdk.competitions.types.competition_api_service import ApiCreateCodeSubmissionRequest, ApiListSubmissionsRequest
from kagglesdk.kernels.services.kernels_api_service import KernelsApiClient
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest, ApiListKernelSessionOutputRequest

COMPETITION = 'birdclef-2026'
KERNEL_OWNER = 'yourslewis'
KERNEL_SLUG = 'bc26-v575-repo-owned-eos5-confirmation'
KERNEL_VERSION = 1
DESCRIPTION = 'v575: Repo-owned EoS5 confirmation of v574 public949 path'
REQUIRED_FILES = {
    'submission.csv',
    'subm_2.csv',
    'subm_5.csv',
    'subm_karnakbayev_power_optimization.csv',
    'submission_protossm.csv',
    'submission_sed.csv',
}
REQUIRED_LOG_MARKERS = [
    'Karnakbayev_PowerOptimization_LB0948',
    'Wrote submission.csv:',
    'Blend and post-processing complete',
    "Ensemble: ['Model_2', 'Model_5']",
]
T = TypeVar('T')


def make_clients():
    with open(os.path.expanduser('~/.kaggle/kaggle.json')) as f:
        token = json.load(f)['key']
    http = KaggleHttpClient(api_token=token)
    return CompetitionApiClient(http), KernelsApiClient(http)


def call_with_retries(label: str, fn: Callable[[], T], attempts: int = 4, sleep_s: int = 300) -> T:
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_exc = exc
            if attempt == attempts:
                break
            print(f'Transient {label} error {type(exc).__name__}: {exc}; sleeping {sleep_s}s', flush=True)
            time.sleep(sleep_s)
    raise RuntimeError(f'{label} failed after {attempts} attempts') from last_exc


def recent_submissions(competitions):
    req = ApiListSubmissionsRequest()
    req.competition_name = COMPETITION
    req.page_size = 200
    return call_with_retries('list_submissions', lambda: competitions.list_submissions(req)).submissions or []


def kernel_status(kernels):
    req = ApiGetKernelSessionStatusRequest()
    req.user_name = KERNEL_OWNER
    req.kernel_slug = KERNEL_SLUG
    return call_with_retries('kernel_status', lambda: kernels.get_kernel_session_status(req))


def output_ok(kernels) -> bool:
    req = ApiListKernelSessionOutputRequest()
    req.user_name = KERNEL_OWNER
    req.kernel_slug = KERNEL_SLUG
    req.page_size = 100
    output = call_with_retries('kernel_output', lambda: kernels.list_kernel_session_output(req))
    files = {getattr(f, 'file_name', getattr(f, 'fileName', '')) for f in (getattr(output, 'files', []) or [])}
    log = getattr(output, 'log', '') or ''
    print(f'Kernel output files: {sorted(files)}', flush=True)
    missing_files = sorted(REQUIRED_FILES - files)
    missing_markers = [m for m in REQUIRED_LOG_MARKERS if m not in log]
    if missing_files:
        print(f'Missing required files: {missing_files}', flush=True)
    if missing_markers:
        print(f'Missing required log markers: {missing_markers}', flush=True)
    # Persist a short verification log for review without committing Kaggle outputs.
    os.makedirs('logs', exist_ok=True)
    with open('logs/v575_output_verification_summary.txt', 'w') as f:
        f.write('files=' + repr(sorted(files)) + '\n')
        f.write('missing_files=' + repr(missing_files) + '\n')
        f.write('missing_markers=' + repr(missing_markers) + '\n')
        f.write('log_tail=' + log[-4000:] + '\n')
    return not missing_files and not missing_markers


def quota_sleep_seconds(text: str) -> int:
    m = re.search(r'(\d+(?:\.\d+)?)\s+hours?\s+from now', text)
    if m:
        return max(300, int(float(m.group(1)) * 3600) + 120)
    m = re.search(r'(\d+)\s+minutes?\s+from now', text)
    if m:
        return max(300, int(m.group(1)) * 60 + 120)
    return 3600


def submit(competitions):
    req = ApiCreateCodeSubmissionRequest()
    req.competition_name = COMPETITION
    req.kernel_owner = KERNEL_OWNER
    req.kernel_slug = KERNEL_SLUG
    req.kernel_version = KERNEL_VERSION
    req.file_name = 'submission.csv'
    req.submission_description = DESCRIPTION
    return competitions.create_code_submission(req)


def main():
    competitions, kernels = make_clients()
    while True:
        submissions = recent_submissions(competitions)
        if any(str(s.description) == DESCRIPTION for s in submissions):
            print('v575 already submitted; exiting.', flush=True)
            return
        status = kernel_status(kernels)
        status_text = str(getattr(status, 'status', status)).upper()
        failure_message = getattr(status, 'failure_message', getattr(status, 'failureMessage', ''))
        print(f'Kernel status: {status}', flush=True)
        if failure_message or 'ERROR' in status_text or 'FAILED' in status_text:
            raise SystemExit(f'v575 kernel failed; not submitting. failure={failure_message!r}')
        if 'COMPLETE' not in status_text:
            print('v575 not complete yet; sleeping 10 minutes.', flush=True)
            time.sleep(600)
            continue
        if not output_ok(kernels):
            raise SystemExit('v575 COMPLETE but output verification failed; not submitting.')
        try:
            print(f'Submitting v575 kernel version {KERNEL_VERSION}...', flush=True)
            print('Submission result:', submit(competitions), flush=True)
            return
        except requests.exceptions.HTTPError as exc:
            response = getattr(exc, 'response', None)
            text = getattr(response, 'text', '') if response is not None else str(exc)
            print(f'Submission attempt failed: {type(exc).__name__}: {exc}', flush=True)
            if text:
                print(text[:2000], flush=True)
            if 'daily Submission allowance' in text or ('daily' in text.lower() and 'allowance' in text.lower()):
                sleep_s = quota_sleep_seconds(text)
                print(f'Daily cap exhausted; sleeping {sleep_s}s before retry.', flush=True)
                time.sleep(sleep_s)
                continue
            raise
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            print(f'Transient submit API error {type(exc).__name__}: {exc}; sleeping 10 minutes.', flush=True)
            time.sleep(600)


if __name__ == '__main__':
    main()
