# BirdCLEF 2026 v546 Decision Matrix — 2026-05-13

Status: prepared while `v545` is complete but waiting behind the Kaggle daily cap. **Do not push `v546` until `v545` scores.**

Current scored anchor: `v541`/`v542`/`v543`/`v544` = **0.946 public LB**.

## Candidate summary

| Candidate | Readiness | Upside | Risk | Required next gate |
|---|---:|---:|---:|---|
| Lower CLAP `0.01`/`0.02` | Conditional | Medium if `v545` improves | Medium; CLAP local overlap is weak | Only consider if `v545 > 0.946`; run lower-weight local grid from downloaded v545 outputs |
| Public946 + CV9245 `0.02`/`0.05` | High prep | Medium-high | Medium runtime/model-source complexity | Implement dry-run, write `submission_cv9245_cnnonly_sharedperch.csv`, then sidecar grid |
| Public946 + train-audio-head `0.03`/`0.05` | High prep | Medium | Low runtime, medium hidden-fit uncertainty | Implement dry-run, write `submission_train_audio_head.csv`, then sidecar grid |
| Tuned public946 gates | High prep | Low-medium | Overfit risk to tiny dry-run label overlap | Use full-sweep config only if no distinct sidecar passes gates |
| BirdNET-only continuation | Done/stop | Low | Slot waste | Do not submit; 10% and 5% both tied 0.946 |

## Decision by v545 result

### If `v545 > 0.946`

1. Treat CLAP as live signal, but do **not** widen it.
2. Compare lower CLAP weights (`0.01`, `0.02`) against CV9245 and train-audio-head dry-run gates.
3. Prefer the candidate with the best risk-adjusted sidecar metrics and bounded displacement from v542/v545.

### If `v545 = 0.946`

1. Treat CLAP 5% as a safe tie but not enough to continue CLAP tuning immediately.
2. Lead with a distinct non-CLAP sidecar:
   - CV9245 `0.02` if we want the safest first probe.
   - Train-audio-head `0.05` if we want the lower-runtime hidden-tie-break bet.
3. Use `scripts/birdclef_public946_sidecar_weight_grid.py` on the candidate dry-run sidecar output before submission.

### If `v545 < 0.946`

1. Stop CLAP for public slots.
2. Choose between CV9245 and train-audio-head using local sidecar gate metrics.
3. If both sidecars look harmful or cannot be gated, submit the tuned public946-gate candidate only as a fallback.

## Risk-adjusted gate metrics

For each sidecar, record:

- local macro AUC and delta vs public946 baseline
- top-k row recall
- corr vs v542/v541 final
- MAE and max absolute displacement vs anchor
- runtime and log evidence for no silent fallback
- row/column alignment and no NaNs

Submit only one candidate per cap window unless the first candidate scores quickly and clearly improves/ties with low risk.

## Current concrete recommendation

Before `v545` scores: no new Kaggle push.

After `v545` scores:

1. If `v545` improves: prepare lower CLAP `0.01`/`0.02` and compare to CV9245/train-head.
2. If `v545` ties/drops: implement train-audio-head or CV9245 dry-run first, because BirdNET is exhausted and CLAP is no longer the best next slot.
3. If runtime allows, train-audio-head is the fastest dry-run candidate; CV9245 is the stronger model-diversity candidate.
