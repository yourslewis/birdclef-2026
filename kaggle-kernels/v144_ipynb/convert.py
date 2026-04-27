import json
import os

with open('/Users/yourslewis/.openclaw/workspace-don/kaggle/birdclef-2026/v111/script.py') as f:
    script = f.read()

cells = [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + '\n' for line in script.split('\n')]}]
notebook = {
    "cells": cells, 
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, 
        "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.10.12"}
    }, 
    "nbformat": 4, 
    "nbformat_minor": 4
}

with open('/Users/yourslewis/.openclaw/workspace-don/kaggle/birdclef-2026/v144_ipynb/v144.ipynb', 'w') as f:
    json.dump(notebook, f)

meta = {
    "id": "yourslewis/bc26-v144-ipynb-wrapper",
    "title": "bc26-v144-ipynb-wrapper",
    "code_file": "v144.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "false",
    "dataset_sources": [],
    "competition_sources": ["birdclef-2026"],
    "kernel_sources": [],
    "model_sources": []
}

with open('/Users/yourslewis/.openclaw/workspace-don/kaggle/birdclef-2026/v144_ipynb/kernel-metadata.json', 'w') as f:
    json.dump(meta, f)
