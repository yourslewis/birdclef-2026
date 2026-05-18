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
