# BirdCLEF NFNet-L0 20s public946 pseudo-label pivot — 2026-05-18

Status: queued after v573 cw-style B0 sidecar dropped to public `0.945`.

## Motivation

`v573` proved that stronger local lift from a cw-style B0 sidecar still does not reliably transfer to public LB:

- local public946 teacher-cache lift: `+0.000023632`
- site bootstrap p(lift>0): `0.8033`
- public LB: `0.945`, below the `0.946` plateau

Therefore this branch stops cw-style B0 sidecar work and pivots to a distinct model family from the 2025 recipe/spec pack: `eca_nfnet_l0`.

## Config

`configs/birdclef/pl_public946_sed85_rankblend15_nfnet_20s_m160_lr1e4_ep20_center_20260518.json`

Design choices:

- teacher: public946 `teacher_sed85_rankblend15.npz`
- backbone: `eca_nfnet_l0`
- context: `20s`, `160` mel bins
- target: center-only soft pseudo labels, **not** cw/local-window
- LR: `1e-4`
- epochs: `20`
- batch size: `4`
- restore best by validation AUC

## Gate

No Kaggle submission from config alone. Before packaging, require:

1. completed train-soundscape student metrics;
2. aligned public946 blend/stability audit;
3. evidence materially stronger or more distinct than v573, not just tiny local lift;
4. runtime/package feasibility, since NFNet is much larger than B0.

## Launch status — 2026-05-18 12:26 UTC

Launched on trainer GPU1:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/birdclef_pseudolabel_student_train.py \
  --config configs/birdclef/pl_public946_sed85_rankblend15_nfnet_20s_m160_lr1e4_ep20_center_20260518.json
```

Runtime paths:

- log: `logs/pl_public946_sed85_rankblend15_nfnet_20s_m160_lr1e4_ep20_center_20260518.log`
- output: `artifacts/pseudolabels/students/pl-public946-sed85-rankblend15-nfnet-20s-m160-lr1e4-ep20-center-20260518/`

Early signal at epoch 9: validation AUC `0.981810` over `59` valid classes, teacher correlation `0.900515`, MAE `0.026362`. Job still running; wait for full metrics and blend/stability audit before making any packaging decision.
