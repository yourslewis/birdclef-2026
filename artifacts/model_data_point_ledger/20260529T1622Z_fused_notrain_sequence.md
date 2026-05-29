# 2026-05-29 16:22 UTC — Fused DyMN10+PANNs no-train sequence data point

## Status
- Live Kaggle state before run: public best `0.949`; latest v631-v635 `0.926/0.940/0.946/0.949/0.941`; 2026-05-29 UTC slots `0/5`; early/mid-day policy active with ~7.7h to reset.
- No active local/trainer BirdCLEF jobs; trainer GPUs were free. Ran this embedding-head training locally after syncing the small fused train_soundscapes embedding cache from trainer.

## Model / data point
- Experiment: `soundscape-sequence-fused-dymn10-panns-notrain-r2-nofile-losite-ep24-20260529`.
- Family: train_soundscapes sequence/file/site fused AudioSet embedding no-train mining.
- Data: official `train_soundscapes`, 1,478 5s windows / 66 files / 9 sites.
- Target scope: 28 no-train labels, all non-Aves in current taxonomy/soundscape-positive scope.
- Features/model: fused EfficientAT DyMN10 + PANNs/Cnn14 embeddings, radius-2 prev/next + local mean/max + time features, no file/site one-hot, hidden MLP 256, dropout 0.45, site-balanced sampling, 24 epochs.
- Validation: leave-one-site; 6 completed folds / 3 skipped low-count or low-valid sites.

## Metrics
- Context row macro AUC: `0.554429` / 6 folds.
- Row-only baseline inside same run: `0.487616`; context lift `+0.066813`.
- File-MIL macro AUC: `0.660711`; row-only file-MIL `0.615956`; context file lift `+0.044756`.
- No-train AUC: `0.554429`; non-Aves AUC: `0.554429` (same 28-label scope).
- Fold context-minus-row deltas: S03 `+0.146591`, S08 `-0.006062`, S13 `+0.208008`, S19 `+0.010252`, S22 `-0.016998`, S23 `+0.059088`.

## Comparison
- vs prior PANNs no-train context: row `-0.046876`, file-MIL `+0.044562`.
- vs PANNs no-train localmax-only: row `-0.028370`, file-MIL `+0.045081`.
- vs PANNs no-train row-only export: row `-0.019407`, file-MIL `+0.093573`.
- vs DyMN10 no-train r2 no-file context: row `+0.000784`, file-MIL `+0.022433`.
- Interpretation: fused DyMN10+PANNs underfits row discrimination but is the best current 28-label no-train file-MIL point; useful for file-level/sonotype movement diagnosis, not direct sidecar submission.

## Sidecar audit
- Wrapped OOF predictions into an anchor-preserved 28→234 v616 proxy matrix using both sequence-sidecar slots pointed at the fused NPZ.
- Best non-control recipe: `seq_context_w01` local macro AUC `0.990398` / 42 valid classes.
- Lift vs v616: `-0.003083`; lift vs anchor `+0.000007`; rank corr vs v616 `0.999687`; MAE `0.006268`.
- Output finite/nonconstant; audit script completed; `submit_approved=false` by metrics/gates.

## Critic / verifier decision
- Critic: file-MIL improvement is real but the row metric and v616 proxy sidecar remain weak; spending an early-day slot would be leaderboard probing, not a verifier-grade move.
- Verifier: row/column alignment and finite/nonconstant checks passed through the audit; candidate is below v616 locally and is not submission-grade.

## Artifacts
- Config: `configs/birdclef/soundscape_sequence_fused_dymn10_panns_notrain_r2_nofile_losite_ep24_20260529.json`
- Training metrics: `artifacts/soundscape_sequence_mining/soundscape-sequence-fused-dymn10-panns-notrain-r2-nofile-losite-ep24-20260529/metrics.json`
- OOF predictions: `artifacts/soundscape_sequence_mining/soundscape-sequence-fused-dymn10-panns-notrain-r2-nofile-losite-ep24-20260529/leave_site_predictions.npz`
- Sidecar audit: `artifacts/soundscape_sequence_sidecar_audit/20260529T1618Z_fused_notrain_sidecar/audit_summary.json`
- Logs: `logs/soundscape_sequence_fused_notrain_20260529T1618Z.log`, `logs/soundscape_sequence_fused_notrain_sidecar_20260529T1618Z.log`
