"""Push v246 Kaggle kernel with KGAT Bearer auth via Kaggle API 2.x."""
import json
import os
from kaggle.api.kaggle_api_extended import KaggleApi

with open(os.path.expanduser("~/.kaggle/kaggle.json"), "r") as f:
    os.environ["KAGGLE_API_TOKEN"] = json.load(f)["key"]

api = KaggleApi()
api.authenticate()
folder = os.path.expanduser("~/Documents/birdclef-2026/kaggle-kernels/v246-ultrasharp-temporal-smoothing")
print("Pushing kernel v246-ultrasharp-temporal-smoothing...")
result = api.kernels_push(folder)
print("Kernel push result:", result)
