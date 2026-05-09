"""Push v525 taxon-gate floor0.40 follow-up kernel with KGAT Bearer auth."""
import json
import os
from pathlib import Path
import requests
folder = Path('kaggle-kernels/v525-taxon-max-gate-floor040')
meta = json.loads((folder / 'kernel-metadata.json').read_text())
script = (folder / meta['code_file']).read_text()
with open(os.path.expanduser('~/.kaggle/kaggle.json')) as f:
    token = json.load(f)['key']
payload = {
    'id': meta.get('id_no'),
    'slug': meta['id'],
    'newTitle': meta.get('title'),
    'text': script,
    'language': meta['language'],
    'kernelType': meta['kernel_type'],
    'isPrivate': meta.get('is_private'),
    'enableGpu': meta.get('enable_gpu'),
    'enableTpu': meta.get('enable_tpu'),
    'enableInternet': meta.get('enable_internet'),
    'datasetDataSources': meta.get('dataset_sources', []),
    'competitionDataSources': meta.get('competition_sources', []),
    'kernelDataSources': meta.get('kernel_sources', []),
    'modelDataSources': meta.get('model_sources', []),
    'categoryIds': meta.get('keywords', []),
    'dockerImagePinningType': meta.get('docker_image_pinning_type'),
}
print(f"Pushing {meta['id']}...")
resp = requests.post(
    'https://www.kaggle.com/api/v1/kernels/push',
    headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json', 'Accept': 'application/json'},
    json=payload,
    timeout=120,
)
print('status:', resp.status_code)
print(resp.text)
resp.raise_for_status()
