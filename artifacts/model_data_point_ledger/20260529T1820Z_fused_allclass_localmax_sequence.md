# Fused DyMN10+PANNs all-class localmax-only sequence data point — 2026-05-29 18:20 UTC

## Live state
- Public LB best: `0.949`; v616 remains tied repo-owned baseline; latest v631-v635 completed `0.926/0.940/0.946/0.949/0.941`.
- 2026-05-29 UTC slots used at live check: `0/5`; ~5.7h to reset, so mid-day policy applies.
- Active jobs before training: no local/trainer BirdCLEF jobs; trainer GPUs free.

## Experiment
- Experiment id: `soundscape-sequence-fused-dymn10-panns-allcls-r2-localmaxonly-losite-ep20-20260529`.
- Branch family: train_soundscapes sequence/file/site AudioSet fusion mining.
- Data: official `train_soundscapes`, 1,478 5s windows / 66 files / 9 sites.
- Target scope: all 234 taxonomy labels.
- Model/init: fused frozen EfficientAT DyMN10 + PANNs/Cnn14 embeddings, localmax-only radius-2 temporal feature, time features, MLP hidden 384, dropout 0.42, site-balanced sampling, 20 epochs.
- Validation split: leave-one-site; 7 completed folds.

## Metrics
- Primary: context row macro ROC-AUC `0.572762`.
- Internal row-only baseline: `0.547572`; localmax context delta `+0.025190`.
- File-MIL AUC: `0.669996` vs row-only `0.645143`; delta `+0.024853`.
- No-train AUC: `0.550756` across 7 folds / 41 valid fold-classes.
- Non-Aves AUC: `0.632578` across 7 folds / 68 valid fold-classes.
- Per-site deltas context-row: S03 -0.008681, S08 -0.049307, S13 +0.115924, S15 -0.035523, S19 +0.098610, S22 +0.021850, S23 +0.033454.

## Comparisons
- Vs fused all-class r2 no-file context: row `-0.023880`, file-MIL `-0.005986`.
- Vs PANNs all-class localmax-only: row `-0.068739`, file-MIL `-0.011757`.
- Interpretation: fused localmax-only helps its own row-only baseline but underperforms both previous fused all-class context and PANNs all-class localmax; it does not explain the no-train file-MIL clue well enough to promote.

## Sidecar audit
- Wrapped all-class leave-site OOF predictions into the v616 proxy matrix and ran `birdclef_soundscape_allclass_sidecar_audit.py` with 200 bootstrap iterations.
- Best non-control recipe: `allcls_seq_w0p005`.
- Local macro AUC `0.991500` / 42 valid classes.
- Lift vs v616 `-0.001981`; lift vs anchor `+0.001109`; rank corr vs v616 `0.999671`; MAE `0.005939`.
- Submit approved: `False`.

## Decision
- Reject as slot candidate; no submission. This is a useful negative data point: fused localmax-only all-class training gives moderate file-MIL but remains below comparable sequence baselines and below v616 in sidecar audit.
- Next: soft1279 head-loaded per-class/site movement diagnosis remains the best positive local lead; no-train file-MIL/sonotype diagnostics remain useful for explaining file-level clues.

## Artifacts
- Config: `configs/birdclef/soundscape_sequence_fused_dymn10_panns_allcls_r2_localmaxonly_losite_ep20_20260529.json`
- Training metrics: `artifacts/soundscape_sequence_mining/soundscape-sequence-fused-dymn10-panns-allcls-r2-localmaxonly-losite-ep20-20260529/metrics.json`
- OOF predictions: `artifacts/soundscape_sequence_mining/soundscape-sequence-fused-dymn10-panns-allcls-r2-localmaxonly-losite-ep20-20260529/leave_site_predictions.npz`
- TorchScript head: `artifacts/soundscape_sequence_mining/soundscape-sequence-fused-dymn10-panns-allcls-r2-localmaxonly-losite-ep20-20260529/context_head_torchscript.pt`
- Sidecar audit: `artifacts/soundscape_allclass_sidecar_audit/20260529T1815Z_fused_allclass_localmax/audit_summary.json`
