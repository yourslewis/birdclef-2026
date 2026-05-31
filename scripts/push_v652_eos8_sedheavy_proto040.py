#!/usr/bin/env python3
import json, os
from pathlib import Path
import requests
FOLDER = Path(__file__).resolve().parents[1] / 'kaggle-kernels' / 'v652-eos8-sedheavy-proto040'
META=json.loads((FOLDER/'kernel-metadata.json').read_text())
TEXT=(FOLDER/META['code_file']).read_text()
with open(os.path.expanduser('~/.kaggle/kaggle.json')) as f: token=json.load(f)['key']
payload={'id': META.get('id_no'), 'slug': META['id'], 'newTitle': META.get('title'), 'text': TEXT, 'language': META['language'], 'kernelType': META['kernel_type'], 'isPrivate': META.get('is_private', True), 'enableGpu': META.get('enable_gpu', False), 'enableTpu': META.get('enable_tpu', False), 'enableInternet': META.get('enable_internet', False), 'datasetDataSources': META.get('dataset_sources', []), 'competitionDataSources': META.get('competition_sources', []), 'kernelDataSources': META.get('kernel_sources', []), 'modelDataSources': META.get('model_sources', []), 'categoryIds': META.get('keywords', [])}
print(json.dumps({'pushing':META['id'],'source_chars':len(TEXT)},indent=2),flush=True)
r=requests.post('https://www.kaggle.com/api/v1/kernels/push',headers={'Authorization':f'Bearer {token}','Content-Type':'application/json','Accept':'application/json'},json=payload,timeout=180)
print('status',r.status_code); print(r.text[:4000]); r.raise_for_status()
