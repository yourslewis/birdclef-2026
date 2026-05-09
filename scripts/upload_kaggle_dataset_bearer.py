#!/usr/bin/env python3
"""Upload/create a Kaggle dataset using KGAT Bearer auth.

This is a small repo-local helper because the legacy `kaggle datasets` CLI can
401 under the current KGAT token while the kagglesdk service clients work.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import time
from pathlib import Path

import requests
from kagglesdk.blobs.services.blob_api_service import BlobApiClient
from kagglesdk.blobs.types.blob_api_service import ApiBlobType, ApiStartBlobUploadRequest
from kagglesdk.datasets.services.dataset_api_service import DatasetApiClient
from kagglesdk.datasets.types.dataset_api_service import ApiCreateDatasetRequest, ApiDatasetNewFile
from kagglesdk.kaggle_http_client import KaggleHttpClient


def load_token() -> str:
    with open(os.path.expanduser("~/.kaggle/kaggle.json")) as f:
        data = json.load(f)
    return data["key"]


def upload_blob(path: Path, token: str) -> str:
    size = path.stat().st_size
    req = ApiStartBlobUploadRequest()
    req.type = ApiBlobType.DATASET
    req.name = path.name
    req.content_length = size
    req.content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    req.last_modified_epoch_seconds = int(path.stat().st_mtime)

    print(f"Starting blob upload: {path.name} ({size / 1024 / 1024:.1f} MB)", flush=True)
    blob_client = BlobApiClient(KaggleHttpClient(api_token=token))
    start = blob_client.start_blob_upload(req)
    print(f"Blob token received; uploading to signed URL...", flush=True)

    with path.open("rb") as f:
        resp = requests.put(start.create_url, data=f, timeout=1800)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"blob PUT failed: {resp.status_code} {resp.text[:500]}")
    print("Blob upload complete", flush=True)
    return start.token


def create_dataset(
    owner: str,
    slug: str,
    title: str,
    description: str,
    file_token: str,
    private: bool,
    token: str,
    file_description: str,
):
    file = ApiDatasetNewFile()
    file.token = file_token
    file.description = file_description

    req = ApiCreateDatasetRequest()
    req.owner_slug = owner
    req.slug = slug
    req.title = title
    req.license_name = "CC0-1.0"
    req.is_private = private
    req.files = [file]
    req.description = description
    req.category_ids = []

    print(f"Creating dataset {owner}/{slug}...", flush=True)
    ds_client = DatasetApiClient(KaggleHttpClient(api_token=token))
    result = ds_client.create_dataset(req)
    print(result.to_json(), flush=True)
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--owner", default="yourslewis")
    p.add_argument("--slug", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--description", default="BirdCLEF 2026 TorchScript SED bundle")
    p.add_argument("--file-description", default="TorchScript SED bundle zip")
    p.add_argument("--public", action="store_true")
    args = p.parse_args()

    token = load_token()
    blob_token = upload_blob(args.file, token)
    create_dataset(
        args.owner,
        args.slug,
        args.title,
        args.description,
        blob_token,
        private=not args.public,
        token=token,
        file_description=args.file_description,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
