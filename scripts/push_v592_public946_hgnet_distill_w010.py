"""Push v592 public946 + Qiuzi HGNet distill 10% sidecar Kaggle kernel via Bearer API v1."""
import json
import os
from pathlib import Path
import requests

FOLDER = Path(os.path.expanduser('~/Documents/birdclef-2026-v545/kaggle-kernels/v592-public946-hgnet-distill-w010'))
META = json.loads((FOLDER / 'kernel-metadata.json').read_text())
SCRIPT = (FOLDER / META['code_file']).read_text()
with open(os.path.expanduser('~/.kaggle/kaggle.json')) as f:
    token = json.load(f)['key']
payload = {
    'id': META.get('id_no'),
    'slug': META['id'],
    'newTitle': META.get('title'),
    'text': SCRIPT,
    'language': META['language'],
    'kernelType': META['kernel_type'],
    'isPrivate': META.get('is_private'),
    'enableGpu': META.get('enable_gpu'),
    'enableTpu': META.get('enable_tpu'),
    'enableInternet': META.get('enable_internet'),
    'datasetDataSources': META.get('dataset_sources', []),
    'competitionDataSources': META.get('competition_sources', []),
    'kernelDataSources': META.get('kernel_sources', []),
    'modelDataSources': META.get('model_sources', []),
    'categoryIds': META.get('keywords', []),
    'dockerImagePinningType': META.get('docker_image_pinning_type'),
}
print('Pushing v592 public946 HGNet distill W010 kernel...')
resp = requests.post(
    'https://www.kaggle.com/api/v1/kernels/push',
    headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json', 'Accept': 'application/json'},
    json=payload,
    timeout=120,
)
print('Kernel push status:', resp.status_code)
print('Kernel push result:', resp.text)
resp.raise_for_status()
