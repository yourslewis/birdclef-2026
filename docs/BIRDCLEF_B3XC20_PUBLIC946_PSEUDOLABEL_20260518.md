# BirdCLEF B3 XC-init 20s public946 pseudo-label candidate — 2026-05-18

Status: queued after v573 dropped and NFNet20 produced only a tiny stable local lift.

## Motivation

After `v573` scored `0.945`, cw-style B0 sidecars are stopped. The next candidates must be genuinely distinct model/source signals.

NFNet20 center-only was distinct but not strong enough to package immediately:

- local lift: `+0.000007693`
- TorchScript size: `89.872 MB`

The refreshed B3 external-pretrained student is a better next context experiment because the prior 5s/128mel B3 XC-init artifact had a stronger local signal and lower teacher correlation:

- prior config: `pl_public946_sed85_rankblend15_b3_xc_q3_manifest20260517_extinit_5s_m128_lr1e4_ep20_20260517.json`
- standalone AUC: `0.968505`
- corr vs public946 teacher: `0.936244`
- best local lift: `+0.000045896` at weight `0.05`
- TorchScript size: `41.995 MB`

It was not submitted earlier because local sidecar transfer was uncertain. This PR does not submit it; it tests whether a 20s/160mel B3 context variant materially improves the offline evidence.

## Config

`configs/birdclef/pl_public946_sed85_rankblend15_b3_xc_q3_extinit_20s_m160_lr1e4_ep20_center_20260518.json`

Design:

- teacher: public946 `teacher_sed85_rankblend15.npz`
- backbone: EfficientNet-B3
- initialization: refreshed q3/cap80 external-pretrained B3 checkpoint
- context: `20s`
- mel bins: `160`
- targets: center-only soft pseudo labels, no cw/local-window
- LR: `1e-4`
- epochs: `20`
- batch size: `4`

## Gate

No Kaggle packaging/submission from config alone. Require:

1. completed metrics;
2. aligned public946 blend/stability audit;
3. evidence materially stronger than both NFNet20 and failed v573-style B0 sidecars;
4. runtime/package feasibility.
