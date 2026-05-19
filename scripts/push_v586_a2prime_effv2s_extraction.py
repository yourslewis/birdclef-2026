"""Push repo-owned v586 A2Prime EffV2S extraction kernel with Bearer API v1.

This helper only pushes the private Kaggle kernel. It does not submit to the
competition. Run only after v585 has scored/dropped/no-scored and this fallback
is still the chosen next action.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import requests

FOLDER = Path(os.path.expanduser("~/Documents/birdclef-2026-v545/kaggle-kernels/v586-a2prime-effv2s-extraction"))


def main() -> None:
    meta = json.loads((FOLDER / "kernel-metadata.json").read_text())
    text = (FOLDER / meta["code_file"]).read_text()
    with open(os.path.expanduser("~/.kaggle/kaggle.json")) as f:
        token = json.load(f)["key"]
    payload = {
        "id": meta.get("id_no"),
        "slug": meta["id"],
        "newTitle": meta.get("title"),
        "text": text,
        "language": meta["language"],
        "kernelType": meta["kernel_type"],
        "isPrivate": meta.get("is_private"),
        "enableGpu": meta.get("enable_gpu"),
        "enableTpu": meta.get("enable_tpu"),
        "enableInternet": meta.get("enable_internet"),
        "datasetDataSources": meta.get("dataset_sources", []),
        "competitionDataSources": meta.get("competition_sources", []),
        "kernelDataSources": meta.get("kernel_sources", []),
        "modelDataSources": meta.get("model_sources", []),
        "categoryIds": meta.get("keywords", []),
        "dockerImagePinningType": meta.get("docker_image_pinning_type"),
    }
    print("Pushing v586 A2Prime EffV2S extraction notebook...")
    resp = requests.post(
        "https://www.kaggle.com/api/v1/kernels/push",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json=payload,
        timeout=120,
    )
    print("Kernel push status:", resp.status_code)
    print("Kernel push result:", resp.text)
    resp.raise_for_status()


if __name__ == "__main__":
    main()
