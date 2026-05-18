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

## Result and audit — 2026-05-18 12:43 UTC

Training completed on trainer GPU1:

- rows/classes: `792` rows, `234` classes
- best validation AUC: `0.973045` over `59` valid classes at epoch `19`
- final-all student AUC: `0.972055` over `75` valid classes
- final-all teacher AUC: `0.997018`
- student/teacher corr: `0.932813`
- student/teacher MAE: `0.021694`
- runtime: `111.318s`
- TorchScript size: `41.995 MB`

Aligned public946 blend/stability audit:

- audit path on trainer: `artifacts/pseudolabels/audits/public946_b3_xc_q3_20s_m160_blend_audit_20260518T1242Z.json`
- standalone AUC: `0.972055`
- corr vs teacher: `0.932813`
- best student rank weight: `0.005`
- local lift: `+0.000017820`
- site-bootstrap p(lift>0): `0.7667`, mean lift `+0.000021348`
- leave-one-site p(lift>0): `1.0`, min lift `+0.000000145`

Decision: do **not** package/submit immediately. The result is more stable than the old 5s B3 audit but has less local lift, and v573 showed even larger local lift can drop public LB. Keep as evidence that B3 external-pretrained/context variants are viable but need a stronger offline bar before another slot.
