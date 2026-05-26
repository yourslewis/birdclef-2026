# Model Data Point — PANNs/Cnn14 AudioSet Soundscape Non-Aves / No-Train / No-Call

Timestamp: 2026-05-26 00:22 UTC

See canonical ledger: `artifacts/model_data_point_ledger/20260526T0022Z_panns_cnn14_audioset_soundscape.md`.

## One-line decision

PANNs/Cnn14 AudioSet embeddings are now packaged and measured; they slightly improve the harsh S08 soundscape-specialist proxy over the prior B0 branch (`0.517` vs `0.489` macro AUC), but remain comparison-grade only and are not submission-format.

## Key metrics

- Rows/labels: 1,478 official train-soundscape windows; 72 non-Aves/no-train labels; 30 no-call auxiliary positives.
- Runtime: Cnn14 checkpoint downloaded; embedding extraction 49.84s CUDA; 12-epoch MLP head best val loss `0.45604`.
- Validation: S08 holdout macro AUC `0.517333` / no-train AUC `0.520824`; no-call AUC invalid for this holdout.
- Verification: holdout predictions finite/nonconstant and TorchScript embedding head smoke passed.

## Decision

No submission/no scale. Keep as a useful external-model data point; next work should either build broader no-call masks, run 20s temporal/localmax, or test PANNs with leave-one-site/no-call-valid validation before any 234-class sidecar wrapper.
