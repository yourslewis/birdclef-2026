#!/usr/bin/env python3
"""Extract Jungchan Model21 source slice for v616 implementation planning.

The unified sidecar grid found a strong local candidate using Jungchan's
`subm_21.csv` branch. This helper turns the pulled public source audit artifact
into a bounded source slice so the next implementation step can create a
hidden-safe repo-owned verifier instead of relying on static public outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("artifacts/source_audits/v616_hidden_safe_branch_extraction_20260525T0400Z/jungchan.py.txt"))
    ap.add_argument("--output", type=Path, default=Path("artifacts/source_audits/v616_hidden_safe_branch_extraction_20260525T0400Z/jungchan_model21_block.py.txt"))
    ap.add_argument("--summary", type=Path, default=Path("artifacts/source_audits/v616_hidden_safe_branch_extraction_20260525T0400Z/jungchan_model21_block_summary.json"))
    args = ap.parse_args()

    text = args.source.read_text()
    lines = text.splitlines()
    start = None
    end = None
    for i, line in enumerate(lines):
        if "## Model_21" in line:
            start = i
            break
    if start is None:
        raise SystemExit("Could not find Model_21 start marker")
    for j in range(start + 1, len(lines)):
        if "## Model_52" in lines[j]:
            end = j
            break
    if end is None:
        raise SystemExit("Could not find Model_52 end marker")

    block = "\n".join(lines[start:end]) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(block)

    markers = [
        "subm_21.csv",
        "test_soundscapes",
        "sample_submission.csv",
        "perch_v2.onnx",
        "ProtoSSM",
        "ResidualSSM",
        "write_final_submission",
        "submission.to_csv",
    ]
    summary = {
        "source": str(args.source),
        "output": str(args.output),
        "start_line_1based": start + 1,
        "end_line_1based_exclusive": end + 1,
        "line_count": end - start,
        "char_count": len(block),
        "markers": {m: (m in block) for m in markers},
        "notes": [
            "Slice starts at Jungchan public source Model_21 block and stops before Model_52.",
            "This is an implementation aid only; hidden-safe v616 must rerun this logic on test_soundscapes and must not read static public output CSVs.",
            "Samejima SED and Raunak SED public dry-run outputs are identical, so v616 can use the Samejima/v612 SED branch instead of importing Raunak source for the SED member.",
        ],
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
