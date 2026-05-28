# BirdCLEF evaluation data point — PANNs localmax all-class sidecar audit

UTC: 2026-05-28 00:22

## Audit
- **Input model:** `soundscape-sequence-panns-cnn14-allcls-r2-localmaxonly-losite-ep20-20260528`
- **Branch family:** all-class train_soundscapes sequence sidecar audit
- **Proxy:** v616 local proxy matrix, 240 rows / 234 columns; 156 matched sequence rows, 84 anchor-filled rows
- **Validation:** local v616 proxy macro AUC over 42 valid classes; no-submit audit

## Result
Best recipe: `allcls_seq_w0p005`.

| Metric | Value |
|---|---:|
| Local macro AUC | 0.991753 |
| Valid classes | 42 |
| Lift vs anchor | +0.001362 |
| Lift vs v616 | -0.001728 |
| Rank corr vs v616 | 0.999681 |
| MAE vs v616 | 0.005869 |

## Decision
Reject as a slot candidate. The sidecar is finite/nonconstant and beats the raw anchor locally, but remains below the tied v616 baseline by `-0.001728` and fails promotion gates. No Kaggle submission.

Artifacts: `artifacts/soundscape_sequence_sidecar_audit/20260528T0018Z_panns_localmax/`.
