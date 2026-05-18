# BirdCLEF public946 20s pseudo-label student — 2026-05-18

Status: queued after supervised frame-head 20s train-audio transfer audit failed.

## Motivation

The supervised frame-head B0 20s lane improved random train-audio holdout substantially:

- 20s / 2048-file / 8-epoch: `0.902068` holdout AUC over `144` valid classes.
- 20s / 4096-file / 12-epoch: `0.922414` holdout AUC over `179` valid classes.

But its required public946/labeled-soundscape audit failed:

- standalone soundscape AUC `0.467988` over `75` valid classes;
- best tiny blend into public946 had lift `-0.000001105`;
- site bootstrap mean lift was negative.

So the next actionable hypothesis is not to submit the supervised train-audio model, but to adapt the same 20s/context B0 sidecar directly on public946 train-soundscape pseudo labels.

## Config

`configs/birdclef/pl_public946_sed85_rankblend15_b0_centerlocalmax_r1_cw075_20s_m160_lr3e4_ep20_20260518.json`

This is the 20s sibling of the prior 10s cw0.75 local-window public946 student:

- teacher: `public946-v540-teacher-cache66-v1/teacher_sed85_rankblend15.npz`
- context: `20s`
- mel bins: `160`
- backbone: EfficientNet-B0 frame-head SED
- init: q3/cap80 external B0 TorchScript encoder, no head load
- temporal targets: `center_localmax_mix`, radius `1`, center weight `0.75`
- training: BCE, lr `3e-4`, 20 epochs, restore best by validation AUC

## Gate

Before any Kaggle packaging/submission:

1. run the train-soundscape blend audit versus public946 teacher cache;
2. require positive or at least clearly safer lift than the 10s cw0.75 sibling;
3. package only at a tiny audited weight if the site/bootstrap stability is acceptable.

## Result, audit, and package — 2026-05-18 10:47 UTC

Training completed on trainer GPU1:

- rows: `792` (`634` train / `158` val)
- classes: `234`
- best epoch: `16`
- best validation AUC: `0.990481` over `61` valid classes
- final-all student AUC: `0.991183` over `75` valid classes
- final-all teacher AUC: `0.996798`
- student/teacher corr: `0.965250`
- student/teacher MAE: `0.017386`
- runtime: `64.705s`
- TorchScript size: `15.391 MB`

Blend/stability audit versus the public946 teacher cache passed the exploratory packaging gate:

- audit path on trainer: `artifacts/pseudolabels/audits/public946_cw075_20s_b0_blend_audit_20260518T1055Z.json`
- standalone AUC: `0.991183`
- best tested student rank weight: `0.015`
- local lift vs teacher: `+0.000023632`
- site-bootstrap p(lift>0): `0.8033`
- site-bootstrap mean lift: `+0.000027692`
- leave-one-site p(lift>0): `0.8889`
- leave-one-site min lift: `-0.000000172` on `S09`

Packaged exploratory candidate:

- private dataset: `yourslewis/bc26-public946-cw075-20s-b0-v1`
- zip SHA256: `3ab570390d1ee8cccdd154b83d66a70e6e68770488dd23d8ae94638e408fbf86`
- kernel: `yourslewis/bc26-v573-public946-cw075-20s-b0-w0015`
- kernel URL: https://www.kaggle.com/code/yourslewis/bc26-v573-public946-cw075-20s-b0-w0015
- sidecar CSV: `submission_cw075_20s_b0_student.csv`
- final sidecar rank weight: `0.015`

A guarded submit monitor is running and will submit only after the Kaggle kernel is `COMPLETE` and output verification sees `submission.csv`, `submission_cw075_20s_b0_student.csv`, `submission_sed.csv`, `submission_protossm.csv`, plus the required log markers.

## Submission + power0.85 follow-up — 2026-05-18 11:40 UTC

The guarded monitor submitted v573 after output verification:

- submission: `v573: Public946 v542 plus cw0.75 20s B0 rank sidecar 1.5%`
- ref: `52773142`
- status at recheck: `pending`, no score/error yet
- required files verified: `submission.csv`, `submission_cw075_20s_b0_student.csv`, `submission_sed.csv`, `submission_protossm.csv`
- kernel log markers verified for sidecar completion and final rank blend
- public dry-run sidecar inference: `45.0s` for 20 files, shape `(240,235)`, prob range `0.013601` to `0.983358`, mean `0.094878`

A held power-scaled follow-up was run without spending another Kaggle slot:

- config: `configs/birdclef/pl_public946_sed85_rankblend15_b0_centerlocalmax_r1_cw075_20s_m160_lr3e4_ep20_power085_20260518.json`
- change vs v573 training: `teacher_power=0.85`, seed `96`
- best epoch: `20`
- best val AUC: `0.993466` over `58` valid classes
- final-all student AUC: `0.991986` over `75`
- teacher AUC: `0.996798`
- student/teacher corr: `0.957827`
- runtime: `64.201s`
- TorchScript: `15.391 MB`

Power0.85 audit:

- audit path on trainer: `artifacts/pseudolabels/audits/public946_cw075_20s_b0_power085_blend_audit_20260518T1140Z.json`
- best student rank weight: `0.0075`
- local lift: `+0.000018286`
- site-bootstrap p(lift>0): `0.7067`
- leave-one-site p(lift>0): `0.8889`
- leave-one-site min lift: `-0.000005326` on `S09`

Decision: hold power0.85 as a diagnostic only. It is weaker than v573/power1.0 on both local lift and bootstrap stability, so it should not be packaged/submitted before v573 scores.

## Public result and stop rule — 2026-05-18 12:25 UTC

`v573` completed and scored below the plateau:

- submission: `v573: Public946 v542 plus cw0.75 20s B0 rank sidecar 1.5%`
- ref: `52773142`
- public score: `0.945`
- status: complete, no error

Decision:

- Stop spending Kaggle slots on cw-style B0 sidecar variants.
- Do not package/submit the power0.85 follow-up.
- Treat local train-soundscape cw/B0 sidecar audits as rejection filters only; positive local lift is not sufficient for public submission.
- Pivot to a genuinely distinct model/source signal. The immediate follow-up lane is NFNet-L0 public946 pseudo-label distillation with 20s context and center-only targets, evaluated offline before any packaging.
