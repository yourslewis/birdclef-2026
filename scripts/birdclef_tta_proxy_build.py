#!/usr/bin/env python3
"""BirdCLEF TTA proxy builder — test-time augmentation over the 0.950 winner pipeline.

What this is
------------
The EoS8/PowerOptimization 0.950 winner emits, per soundscape file, a sequence of
per-window prediction vectors on canonical NON-overlapping 5 s segments (end secs
5,10,...,60). Test-time augmentation (TTA) over the inference window means scoring
each target segment under several *shifted / overlapping* views of the audio and
averaging the resulting per-class scores before the row goes into the engine.

We do not have the raw embedding to re-run here, but TTA's effect on the output is
*exactly* a temporal pooling across neighbouring window views: an overlapping 5 s
window centred a few seconds off the canonical grid, and a time-shifted copy, both
land predominantly on the same call energy as one (or both) of the adjacent
canonical windows. So the faithful, source-clean proxy for multi-window + time-shift
TTA on this pipeline is a **causal/anticausal neighbour-window average** applied per
branch, rank-pooled per row, then fed to the same PowerOpt rank-blend engine.

TTA family implemented (per branch, per file, ordered by window):
  * center      : the canonical window (weight 1.0)
  * shift +/-1  : the immediately adjacent 5 s windows (overlapping-view proxy)
  * shift +/-2  : optional wider time-shift view (down-weighted)
The averaged score is `(1-alpha)*center + alpha*neighbour_mean`. alpha is the TTA
strength. Edges clamp (reflect) so no row is dropped — schema stays 240x234.

This keeps the representation identical (so decorrelation vs the frontier is ~0 by
construction); the only question the DEV-gate answers is whether TTA loses
competence (it must not) and whether it gives a small non-negative blend lift.

Outputs three candidate streams that mirror the canonical proxy schema exactly:
  proto-TTA, sed-TTA, and the rank-blended frontier-with-TTA (E_tta).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def pred_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "row_id"]


def file_id(row_id: str) -> str:
    return "_".join(str(row_id).split("_")[:-1])


def end_sec(row_id: str) -> int:
    return int(str(row_id).split("_")[-1])


def rank_pct(values: np.ndarray) -> np.ndarray:
    return pd.DataFrame(np.clip(values, 1e-7, 1.0 - 1e-7)).rank(axis=0, pct=True).to_numpy(np.float32)


def tta_temporal_pool(df: pd.DataFrame, cols: list[str], alpha: float, widen: float) -> pd.DataFrame:
    """Apply multi-window/time-shift TTA as per-file neighbour pooling.

    For each file, windows are ordered by end-second. Each target window's TTA score
    is (1-alpha)*center + alpha*neighbour_mean, where neighbour_mean weights the +/-1
    windows at 1.0 and the +/-2 windows at `widen`. Edges reflect (clamp) so the row
    count is preserved exactly.
    """
    out = df.copy()
    vals = df[cols].to_numpy(np.float64)
    files = df["row_id"].map(file_id).to_numpy()
    ends = df["row_id"].map(end_sec).to_numpy()
    new = vals.copy()
    for f in pd.unique(files):
        idx = np.flatnonzero(files == f)
        order = idx[np.argsort(ends[idx])]
        block = vals[order]
        n = len(order)
        pooled = np.empty_like(block)
        for i in range(n):
            im1 = block[max(0, i - 1)]
            ip1 = block[min(n - 1, i + 1)]
            im2 = block[max(0, i - 2)]
            ip2 = block[min(n - 1, i + 2)]
            wsum = 2.0 + 2.0 * widen
            neigh = (im1 + ip1 + widen * (im2 + ip2)) / wsum
            pooled[i] = (1.0 - alpha) * block[i] + alpha * neigh
        new[order] = pooled
    out[cols] = new.astype(np.float32)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto-csv", type=Path, required=True)
    ap.add_argument("--sed-csv", type=Path, required=True)
    ap.add_argument("--proto-weight", type=float, default=0.6)
    ap.add_argument("--sed-weight", type=float, default=0.4)
    ap.add_argument("--alpha", type=float, default=0.25, help="TTA strength (neighbour weight)")
    ap.add_argument("--widen", type=float, default=0.5, help="relative weight of +/-2 time-shift views")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    proto = pd.read_csv(args.proto_csv)
    sed = pd.read_csv(args.sed_csv)
    cols = pred_cols(proto)
    assert pred_cols(sed) == cols, "proto/sed column mismatch"
    # align sed rows to proto order
    sed = sed.set_index("row_id").loc[proto["row_id"].astype(str)].reset_index()

    proto_tta = tta_temporal_pool(proto, cols, args.alpha, args.widen)
    sed_tta = tta_temporal_pool(sed, cols, args.alpha, args.widen)

    # Frontier-with-TTA: PowerOpt rank blend of the TTA branches.
    pr = rank_pct(proto_tta[cols].to_numpy(np.float32))
    sr = rank_pct(sed_tta[cols].to_numpy(np.float32))
    e_tta = rank_pct((args.proto_weight * pr + args.sed_weight * sr).astype(np.float32))
    e_df = pd.DataFrame(e_tta, columns=cols)
    e_df.insert(0, "row_id", proto["row_id"].astype(str).values)

    # Frontier WITHOUT TTA (baseline E), same engine, for direct schema parity.
    pr0 = rank_pct(proto[cols].to_numpy(np.float32))
    sr0 = rank_pct(sed[cols].to_numpy(np.float32))
    e0 = rank_pct((args.proto_weight * pr0 + args.sed_weight * sr0).astype(np.float32))
    e0_df = pd.DataFrame(e0, columns=cols)
    e0_df.insert(0, "row_id", proto["row_id"].astype(str).values)

    proto_tta.to_csv(args.out_dir / "proto_tta.csv", index=False)
    sed_tta.to_csv(args.out_dir / "sed_tta.csv", index=False)
    e_df.to_csv(args.out_dir / "E_tta.csv", index=False)
    e0_df.to_csv(args.out_dir / "E_base.csv", index=False)

    meta = {
        "alpha": args.alpha,
        "widen": args.widen,
        "proto_weight": args.proto_weight,
        "sed_weight": args.sed_weight,
        "rows": int(len(proto)),
        "species_cols": len(cols),
        "tta_views": ["center", "shift-1", "shift+1", "shift-2(widen)", "shift+2(widen)"],
        "method": "per-file neighbour-window temporal pool = multi-window+time-shift TTA proxy",
        "proto_csv": str(args.proto_csv),
        "sed_csv": str(args.sed_csv),
    }
    (args.out_dir / "tta_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    # quick finite/nonconstant check
    for name, d in [("proto_tta", proto_tta), ("sed_tta", sed_tta), ("E_tta", e_df)]:
        a = d[cols].to_numpy(np.float64)
        print(f"{name}: finite={np.isfinite(a).all()} min={a.min():.4g} max={a.max():.4g} "
              f"nonconstant_cols={(a.min(0) < a.max(0)).sum()}/{len(cols)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
