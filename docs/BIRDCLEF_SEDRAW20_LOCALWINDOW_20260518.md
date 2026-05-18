# BirdCLEF raw-SED 20s local-window pseudo-label diagnostic — 2026-05-18

Status: prepared after v573 confirmed that final-rankblend/cw-style same-teacher sidecars do not transfer reliably.

## Motivation

The current public best remains `0.946`. Recent same-teacher sidecars have under-transferred despite positive local gates:

- `v560` V2S sidecar: public `0.945`.
- `v573` cw0.75 20s B0 sidecar: public `0.945`.
- NFNet20 and B3XC20 public946 center-only students were stable but too low-lift to submit.

This candidate tests a more SED-specific target: train a compact B0 SED-head student on the raw public946 SED teacher stream, not the final sed85/rankblend teacher, while preserving the gentler local-window target transform that previously looked operationally viable.

## Config

`configs/birdclef/pl_public946_sedraw_b0_centerlocalmax_r1_cw075_20s_m160_lr3e4_ep20_20260518.json`

Design:

- teacher: `teacher_sed.npz` from public946-v540 cache
- backbone: EfficientNet-B0 SED-head student
- init: refreshed q3/cap80 B0 external-pretrain checkpoint
- context: `20s`
- mel bins: `160`
- target transform: center/local-max mix, radius `1`, center weight `0.75`
- LR: `3e-4`
- epochs: `20`
- batch size: `4`

## Gate

Do not package or submit from config alone. Require:

1. completed training metrics;
2. aligned blend/stability audit versus the public946 sed85/rankblend teacher;
3. evidence materially stronger than raw-SED 10s (`negative lift`) and post-v573 low-weight sidecar bar;
4. package/runtime sanity if the audit clears.

## Result and audit — 2026-05-18 13:50 UTC

Training completed on trainer GPU1:

- rows/classes: `792` rows, `234` classes
- best validation AUC: `0.991514` over `61` valid classes at epoch `19`
- final-all student AUC vs truth: `0.991099` over `75` valid classes
- raw SED teacher AUC vs truth: `0.996475`
- student/raw-SED-teacher corr: `0.968311`
- student/raw-SED-teacher MAE: `0.004448`
- runtime: `65.212s`
- TorchScript size: `15.391 MB`

Aligned blend/stability audit against the public946 sed85/rankblend teacher:

- audit path on trainer: `artifacts/pseudolabels/audits/public946_sedraw20_localwindow_blend_audit_20260518T1348Z.json`
- standalone AUC: `0.991099`
- corr vs sed85/rankblend teacher: `0.895538`
- best tested blend weight: `0.001`
- lift vs sed85/rankblend teacher: `-0.000000961`
- site-bootstrap p(lift>0): `0.3267`, mean lift `-0.000005362`
- leave-one-site p(lift>0): `0.1111`, min lift `-0.000012070`

Decision: **kill this exact raw-SED 20s local-window candidate**. It learned the raw SED stream well and stayed compact/fast, but it is not additive to the public946 sed85/rankblend teacher. Do not package or submit.
