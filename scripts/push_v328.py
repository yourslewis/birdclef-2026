"""Push v328 Kaggle kernel with KGAT Bearer auth via Kaggle API 2.x."""
import json, os
from kaggle.api.kaggle_api_extended import KaggleApi
with open(os.path.expanduser('~/.kaggle/kaggle.json')) as f:
    os.environ['KAGGLE_API_TOKEN'] = json.load(f)['key']
api=KaggleApi(); api.authenticate()
folder=os.path.expanduser('~/Documents/birdclef-2026/kaggle-kernels/v328-immediate-topk-context030')
print('Pushing kernel v328-immediate-topk-context030...')
print('Kernel push result:', api.kernels_push(folder))
