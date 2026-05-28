# 20260528T0624Z — PANNs/Cnn14 soundscape-positive localmax LOSO data point

## Summary
- **Experiment:** `soundscape-sequence-panns-cnn14-soundpos-localmax-losite-ep20-20260528`
- **Branch family:** Sequence/file/site AudioSet temporal/localmax target redesign
- **Data:** official `train_soundscapes`, 1,478 windows / 66 files / 9 sites
- **Target scope:** soundscape-positive labels, 75 classes, 6,244 positive cells
- **Model/init:** frozen PANNs/Cnn14 AudioSet embeddings, radius-2 localmax-only temporal context MLP, time features, site-balanced sampling
- **Validation:** leave-one-site; 7 complete folds, 2 skipped low-window/low-valid folds

## Comparable metrics
- Row macro AUC: **0.642375** / 7 folds
- Row-only baseline: **0.632793**; context delta **+0.009582**
- File-MIL macro AUC: **0.662504**
- No-train row AUC: **0.592102**
- Non-Aves row AUC: **0.667663**
- Prediction guard: finite, final all-row predictions `1478x75`, `75/75` nonconstant columns

## Baseline deltas
- Versus PANNs all-class localmax (`20260528T0022Z`, 234-class): row **+0.000874**, file-MIL **-0.019249**; target scope differs.
- Versus native B0 soundscape-positive (`20260528T0422Z`): row **-0.015790**, file-MIL **-0.013879**.
- Versus PANNs row-only in this run: row **+0.009582**, file-MIL **-0.024978**.

## Export/runtime
- TorchScript context head: `artifacts/soundscape_sequence_mining/soundscape-sequence-panns-cnn14-soundpos-localmax-losite-ep20-20260528/context_head_torchscript.pt`
- OOF predictions: `artifacts/soundscape_sequence_mining/soundscape-sequence-panns-cnn14-soundpos-localmax-losite-ep20-20260528/leave_site_predictions.npz`
- Trainer: `CUDA_VISIBLE_DEVICES=1`; run log `logs/soundscape_sequence_panns_cnn14_soundpos_localmax_losite_ep20_20260528.log`

## Decision
**Keep as comparison data point; no early-day submission.** The soundscape-positive PANNs localmax target slightly improves row AUC over the PANNs all-class localmax row metric, but loses file-MIL and loses to native soundscape-positive. Sidecar-v616 audit is below promotion, so this is not slot-worthy.

## Artifacts
- Metrics: `artifacts/soundscape_sequence_mining/soundscape-sequence-panns-cnn14-soundpos-localmax-losite-ep20-20260528/metrics.json`
- Config: `configs/birdclef/soundscape_sequence_panns_cnn14_soundpos_localmax_losite_ep20_20260528.json`
- Sidecar audit: `artifacts/soundscape_sequence_sidecar_audit/20260528T0624Z_panns_soundpos_localmax/audit_summary.json`
