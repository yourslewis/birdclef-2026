# 20260529T1220Z — PANNs no-train localmax-only sequence data point

## Status
- **Experiment id:** `soundscape-sequence-panns-cnn14-notrain-r2-localmaxonly-losite-ep24-20260529`
- **Branch family:** train_soundscapes sequence/file/site AudioSet temporal-localmax mining
- **Evidence level:** comparison-grade no-slot data point
- **Submission decision:** **REJECT / no submission** — below the comparable PANNs no-train context baseline and far below v616 in proxy sidecar audit.

## Live context at run start
- Kaggle Bearer API live check: public best remains `0.949`; v616 is still the tied repo baseline to beat; v634 is the only latest v631-v635 tie.
- Latest completed submissions: v631 `0.926`, v632 `0.940`, v633 `0.946`, v634 `0.949`, v635 `0.941`.
- 2026-05-29 UTC slots used at start: `0/5`; early-day policy active, so no slot without verifier-grade/high-information candidate.
- Active jobs: no local BirdCLEF jobs; trainer GPUs free.

## Model / data
- **Training data:** official `train_soundscapes`, reconstructed as 5s windows/sequences.
- **Rows/files/sites:** 1,478 windows / 66 files / 9 sites.
- **Target scope:** 28 no-train primary labels (`1491113`, `25073`, `517063`, `47158son01`–`47158son25`).
- **Input/features:** frozen PANNs/Cnn14 embeddings (`2048d`) + radius-2 local-max context + time features; no prev/next, no local mean, no file context.
- **Model:** small MLP context head; hidden dim 192; dropout 0.45; site-balanced sampling; positive weights power 0.35 clipped at 10; 24 epochs.
- **Validation:** leave-one-site; 6 complete folds (`S03`, `S08`, `S13`, `S19`, `S22`, `S23`), low-valid sites skipped by protocol.

## Comparable performance

| Metric | Value | Comparator / delta |
|---|---:|---|
| Context row macro AUC | `0.582799` / 6 folds | vs row-only `-0.021661`; vs prior PANNs no-train context `-0.018506` |
| Context file-MIL AUC | `0.615630` | vs row-only `-0.012717`; vs prior PANNs no-train `-0.000519` |
| Row-only internal AUC | `0.604460` | sanity comparator; only localmax context branch was final-trained |
| No-train / non-Aves AUC | `0.582799` | identical to scoped metric because all 28 labels are no-train non-Aves |
| Fold deltas context-row | S03 `-0.067857`, S08 `+0.027738`, S13 `-0.062500`, S19 `-0.006207`, S22 `-0.012251`, S23 `-0.008887` | only S08 improves; sonotype/site inversion persists |
| Final prediction stats | 28/28 nonconstant; min ~`1.33e-28`, max `1.0`, mean `0.049957` | finite final artifact |

## v616 proxy sidecar audit
- Wrapped the 28-label leave-site predictions into v616 proxy rows with anchor-fill for all non-emitted labels.
- Audit used `scripts/birdclef_soundscape_sequence_sidecar_audit.py` with both sidecar slots pointed at this scoped NPZ as a compact reuse smoke.
- Matched proxy rows: 156/240 sequence-overlap rows; sidecar CSV finite/nonconstant 240x234.
- Best non-control recipe: `seq_context_w01` (same result as duplicated `seq_r2_w01`) local AUC `0.990398` / 42 valid.
- Lift vs anchor: `+0.000008`; lift vs v616: `-0.003082`; rank corr vs v616 `0.999689`; MAE vs v616 `0.006251`.
- Verdict: no slot; all non-control recipes remain below v616.

## ClawTeam decisions
- **Coordinator:** chose a genuinely distinct train_soundscape sequence/file/site data point after no submit-ready candidate existed.
- **Data/Feature:** this isolates the 28 no-train frog/sonotype labels and tests whether localmax temporal context helps site transfer.
- **Validation/Metrics:** comparison-grade only; leave-site shows context hurts 5/6 valid folds despite S08 gain.
- **Prediction/Ensemble:** sidecar movement is near-anchor and below v616; no hidden-safe promotion evidence.
- **Critic:** localmax-only no-train context does not fix the sonotype/site-pair inversion; the useful row-only signal suggests regularization/selector diagnostics rather than submission.
- **Verifier:** artifacts finite, nonconstant, aligned, but not submission-grade; no Kaggle submission.

## Artifacts
- Config: `configs/birdclef/soundscape_sequence_panns_cnn14_notrain_r2_localmaxonly_losite_ep24_20260529.json`
- Metrics: `artifacts/soundscape_sequence_mining/soundscape-sequence-panns-cnn14-notrain-r2-localmaxonly-losite-ep24-20260529/metrics.json`
- OOF predictions: `artifacts/soundscape_sequence_mining/soundscape-sequence-panns-cnn14-notrain-r2-localmaxonly-losite-ep24-20260529/leave_site_predictions.npz`
- Sidecar audit: `artifacts/soundscape_sequence_sidecar_audit/20260529T1220Z_panns_notrain_localmax_sidecar/audit_summary.json`

## Next exact action
Diagnose no-train row-only/localmax class-site movement and/or run a stricter sonotype site-pair selector; do not submit this sidecar.
