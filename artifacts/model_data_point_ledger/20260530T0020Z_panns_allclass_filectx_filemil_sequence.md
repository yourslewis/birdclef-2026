# 2026-05-30T00:20Z — PANNs all-class file-context + file-MIL sequence data point

## Status / phase
- Phase: no-slot model data-point training + wrapper audit.
- Live state before training: Kaggle best public LB remained `0.949`; v636-v640 completed at `0.944/0.943/0.939/0.944/0.945`; 2026-05-30 UTC slots `0/5` with a fresh daily cap.
- No active BirdCLEF local/trainer jobs were found before launch; trainer venv `~/kaggle_envs/s6e3` used.

## Model / data
- Experiment id: `soundscape-sequence-panns-cnn14-allcls-r2-filectx-filemil-losite-ep20-20260530`.
- Branch family: train_soundscapes sequence/file/site AudioSet mining.
- Data: official `train_soundscapes`, 1,478 5s windows / 66 files / 9 sites.
- Targets: all 234 taxonomy/submission labels.
- Model/init: frozen PANNs/Cnn14 AudioSet embeddings; radius-2 prev/next + local mean/max + file mean/max + time features; MLP context head, `hidden_dim=384`, dropout `0.42`, file-MIL BCE weight `0.35`, site-balanced sampling, seed `71`, 20 epochs.

## Validation metrics
- Split: leave-one-site; 7 valid folds.
- Primary metric: context row macro ROC-AUC `0.644272`.
- Internal row-only baseline: `0.624443`; context delta `+0.019830`.
- File-MIL macro ROC-AUC: context `0.678888` vs row-only `0.653546` (delta `+0.025342`).
- Secondary slice metrics: no-train `0.631044` mean across valid context folds; non-Aves `0.701883` mean across valid context folds.
- Fold deltas (context minus row): S03 +0.117934, S08 -0.012156, S13 -0.008577, S15 -0.030688, S19 +0.020274, S22 +0.013855, S23 +0.038165.

## Sidecar audit vs v616
- Wrapped leave-site OOF context predictions into the 240-row v616 local proxy with 156 matched sequence rows and all 234 labels; unmatched rows anchor-filled.
- Best non-control recipe: `allcls_seq_w0p0025`.
- Local macro AUC `0.990951` / 42 valid classes.
- Lift vs v616 `-0.002529`; lift vs anchor `+0.000561`; rank corr vs v616 `0.999692`; MAE `0.005987`.
- Promotion gate: `one or more promotion gates failed`; eligible `False`; `submit_approved=false`.

## Decision
- **Reject as slot candidate / keep as comparison data point.** PANNs full file-context + file-MIL improves the same-run row and file-MIL metrics and roughly matches the best prior PANNs local file-MIL lane, but the v616 proxy sidecar remains below the tied baseline by `-0.002529`.
- Submission decision: no Kaggle submission. Early UTC-day slots should wait for a verifier-grade candidate; this branch is not one.

## Artifacts
- Config: `configs/birdclef/soundscape_sequence_panns_cnn14_allcls_r2_filectx_filemil_losite_ep20_20260530.json`
- Training output: `artifacts/soundscape_sequence_mining/soundscape-sequence-panns-cnn14-allcls-r2-filectx-filemil-losite-ep20-20260530/`
- Sidecar audit: `artifacts/soundscape_sequence_sidecar_audit/panns_allcls_filectx_filemil_20260530/`
