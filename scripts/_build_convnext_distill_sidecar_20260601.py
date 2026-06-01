#!/usr/bin/env python3
"""Build a 234-class proxy sidecar for the ConvNeXt-nano soundscape-native stream.

Maps leave-site OOF preds (files+starts) onto the canonical proxy rows
(row_id = <file_stem>_<end_seconds>), anchor-filling unmatched rows with the
0.950 frontier E rankblend so the candidate is a pure diversity probe.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "artifacts/diversity_scout/convnext_distill_20260601/leave_site_predictions.npz"
PROTO = ROOT / "artifacts/source_winner_audit_20260531T0416Z/session_outputs/v644_yaroslav_0950/submission_protossm.csv"
SED = ROOT / "artifacts/source_winner_audit_20260531T0416Z/session_outputs/v644_yaroslav_0950/submission_sed.csv"
OUT = ROOT / "artifacts/diversity_scout/convnext_distill_20260601/E_convnext_distill.csv"


def read_matrix(p: Path):
    with open(p) as f:
        r = csv.reader(f)
        header = next(r)
        rows = []
        vals = []
        for line in r:
            rows.append(line[0])
            vals.append([float(x) for x in line[1:]])
    return header[1:], rows, np.array(vals, dtype=np.float64)


def rankblend(a, b, wa=0.6, wb=0.4):
    def rank(m):
        out = np.empty_like(m)
        for j in range(m.shape[1]):
            order = m[:, j].argsort()
            ranks = np.empty(len(order))
            ranks[order] = np.arange(len(order))
            out[:, j] = ranks / max(1, len(order) - 1)
        return out
    return wa * rank(a) + wb * rank(b)


def main():
    cols, rows, proto = read_matrix(PROTO)
    cols2, rows2, sed = read_matrix(SED)
    assert cols == cols2 and rows == rows2
    E = rankblend(proto, sed)
    row_to_i = {r: i for i, r in enumerate(rows)}
    col_to_j = {c: j for j, c in enumerate(cols)}

    z = np.load(NPZ, allow_pickle=True)
    files = [str(x) for x in z["files"]]
    starts = [str(x) for x in z["starts"]]
    labels = [str(x) for x in z["labels"]]
    pred = z["pred"].astype(np.float64)

    def end_sec(s):
        parts = [int(x) for x in s.split(":")]
        sec = parts[0] * 3600 + parts[1] * 60 + parts[2]
        return sec + 5

    sidecar = E.copy()
    matched = 0
    for k in range(len(files)):
        stem = files[k][:-4] if files[k].endswith(".ogg") else files[k]
        rid = f"{stem}_{end_sec(starts[k])}"
        if rid not in row_to_i:
            continue
        ri = row_to_i[rid]
        for lj, lab in enumerate(labels):
            if lab in col_to_j:
                sidecar[ri, col_to_j[lab]] = pred[k, lj]
        matched += 1

    assert np.isfinite(sidecar).all()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id"] + cols)
        for i, r in enumerate(rows):
            w.writerow([r] + [f"{x:.7g}" for x in sidecar[i]])
    print(f"matched {matched}/{len(files)} npz rows onto {len(rows)} proxy rows")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
