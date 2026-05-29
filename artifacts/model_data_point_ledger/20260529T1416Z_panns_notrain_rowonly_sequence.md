# 2026-05-29 14:16 UTC — PANNs no-train row-only sequence data point

## Status
- Live Kaggle state before run: public best `0.949`; latest v631-v635 `0.926/0.940/0.946/0.949/0.941`; 2026-05-29 UTC slots `0/5`; early-day policy active, no submission without verifier-grade/high-info candidate.
- No active local/trainer BirdCLEF jobs; trainer GPUs free.

## Model / data point
- Experiment: `soundscape-sequence-panns-cnn14-notrain-rowonly-losite-ep24-20260529`.
- Family: train_soundscapes sequence/file/site AudioSet row-only no-train mining.
- Data: official `train_soundscapes`, 1,478 5s windows / 66 files / 9 sites.
- Target scope: 28 no-train labels, all also non-Aves in current taxonomy/soundscape-positive scope.
- Features/model: frozen PANNs/Cnn14 AudioSet embeddings only (`context_radius=0`, no time/site/file/neighbor features), hidden MLP 192, dropout 0.45, site-balanced sampling, 24 epochs.
- Validation: leave-one-site; 6 completed folds / 3 skipped low-count or low-valid sites.

## Metrics
- Row macro AUC: `0.573836` / 6 folds.
- File-MIL macro AUC: `0.567138`.
- No-train AUC: `0.573836`; non-Aves AUC: `0.573836` (same 28-label scope).
- Fold row AUCs: S03 `0.619481`, S08 `0.498115`, S13 `0.695747`, S19 `0.584389`, S22 `0.536536`, S23 `0.508749`.

## Comparison
- vs previous internal row-only baseline from localmax run: row `-0.030624`, file-MIL `-0.061209`.
- vs latest localmax-only context model: row `-0.008963`, file-MIL `-0.048492`.
- vs prior PANNs no-train context model: row `-0.027469`, file-MIL `-0.049011`.
- Interpretation: the explicit exportable row-only no-train model is weaker than both the prior no-train context branch and the localmax run's internal row-only baseline; likely seed/feature/dropout/site-balanced instability, not a robust sonotype sidecar.

## Sidecar audit
- Wrapped OOF predictions into an anchor-preserved 28→234 v616 proxy matrix.
- Best non-control recipe: `seq_context_w01` local macro AUC `0.990405` / 42 valid classes.
- Lift vs v616: `-0.003076`; lift vs anchor `+0.000014`; rank corr vs v616 `0.999688`; MAE `0.006257`.
- Output finite/nonconstant; audit script completed; `submit_approved=false` by metrics/gates.

## Decision
Reject as slot candidate; keep as a negative comparison-grade data point. This strengthens the conclusion that current no-train PANNs sonotype branches need class/site movement diagnostics or target redesign, not more low-context wrappers.

## Artifacts
- Config: `configs/birdclef/soundscape_sequence_panns_cnn14_notrain_rowonly_losite_ep24_20260529.json`
- Metrics: `artifacts/soundscape_sequence_mining/soundscape-sequence-panns-cnn14-notrain-rowonly-losite-ep24-20260529/metrics.json`
- OOF predictions: `artifacts/soundscape_sequence_mining/soundscape-sequence-panns-cnn14-notrain-rowonly-losite-ep24-20260529/leave_site_predictions.npz`
- Sidecar audit: `artifacts/soundscape_sequence_sidecar_audit/20260529T1416Z_panns_notrain_rowonly_sidecar/audit_summary.json`
- Trainer log: `logs/soundscape_sequence_panns_notrain_rowonly_20260529T1416Z.log`
