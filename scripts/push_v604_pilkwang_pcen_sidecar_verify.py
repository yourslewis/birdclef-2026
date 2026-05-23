"""Push v604 Pilkwang PCEN sidecar private verification kernel via Kaggle Bearer API."""
import json, os
from pathlib import Path
import requests
FOLDER=Path(os.path.expanduser('~/Documents/birdclef-2026-v545/kaggle-kernels/v604-pilkwang-pcen-sidecar-verify'))
META=json.loads((FOLDER/'kernel-metadata.json').read_text())
TEXT=(FOLDER/META['code_file']).read_text()
with open(os.path.expanduser('~/.kaggle/kaggle.json')) as f: token=json.load(f)['key']
payload={
  'id': META.get('id_no'),
  'slug': META['id'],
  'newTitle': META.get('title'),
  'text': TEXT,
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
}
print('Pushing', META['id'])
r=requests.post('https://www.kaggle.com/api/v1/kernels/push',headers={'Authorization':f'Bearer {token}','Content-Type':'application/json','Accept':'application/json'},json=payload,timeout=120)
print('status',r.status_code); print(r.text[:4000]); r.raise_for_status()
try:
    js=r.json(); kid=js.get('kernelId')
    if kid:
        META['id_no']=kid
        (FOLDER/'kernel-metadata.json').write_text(json.dumps(META,indent=2))
        print('updated id_no', kid)
except Exception as e:
    print('id update warning', e)
