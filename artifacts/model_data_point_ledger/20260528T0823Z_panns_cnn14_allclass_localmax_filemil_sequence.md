# 20260528T0823Z — PANNs/Cnn14 all-class localmax + file-MIL LOSO data point

## Summary
- **Experiment:** `soundscape-sequence-panns-cnn14-allcls-r2-localmax-filemil-losite-ep20-20260528`
- **Branch family:** Sequence/file/site AudioSet temporal/localmax file-MIL mining
- **Data:** official `train_soundscapes`, 1,478 windows / 66 files / 9 sites
- **Target scope:** all taxonomy labels, 234 classes, 6,244 positive cells
- **Model/init:** frozen PANNs/Cnn14 AudioSet embeddings, radius-2 localmax-only temporal context MLP, time features, site-balanced sampling, extra file-MIL max-logit BCE pass (`file_mil_loss_weight=0.35`)
- **Validation:** leave-one-site; 7 complete folds, 2 skipped low-window/low-valid folds

## Comparable metrics
- Row macro AUC: **0.644053** / 7 folds
- Row-only baseline: **0.617555**; context delta **+0.026498**
- File-MIL macro AUC: **0.665302** vs row-only **0.675630**; delta **-0.010328**
- No-train row AUC: **0.613032**
- Non-Aves row AUC: **0.670490**
- Prediction guard: finite, OOF predictions `1410x234`, final all-row predictions `1478x234`, `234/234` nonconstant columns

## Baseline deltas
- Versus PANNs all-class localmax (`20260528T0022Z`): row **+0.002552**, file-MIL **-0.016451**, no-train **+0.006502**, non-Aves **+0.000712**.
- Versus row-only in this run: row **+0.026498**, file-MIL **-0.010328**.
- Sidecar audit vs v616: best lift **-0.002117**, which is **-0.000389** worse than the prior localmax sidecar despite slightly better row AUC.

## Fold notes
- Improved 6/7 leave-site folds vs row-only; S08 regressed **-0.042654**.
- File-MIL objective did not achieve its intended metric: file-MIL fell from the prior localmax baseline **0.681753** to **0.665302**.

## Export/runtime
- TorchScript context head: `artifacts/soundscape_sequence_mining/soundscape-sequence-panns-cnn14-allcls-r2-localmax-filemil-losite-ep20-20260528/context_head_torchscript.pt`
- OOF predictions: `artifacts/soundscape_sequence_mining/soundscape-sequence-panns-cnn14-allcls-r2-localmax-filemil-losite-ep20-20260528/leave_site_predictions.npz`
- Trainer: `CUDA_VISIBLE_DEVICES=1`; run log `logs/soundscape_sequence_panns_cnn14_allcls_localmax_filemil_losite_ep20_20260528.log`

## Decision
**Reject as a direct/package candidate; keep as comparison-grade data point.** The file-MIL regularizer slightly improves row/no-train AUC over the earlier PANNs localmax branch, but it reduces file-MIL and worsens the v616 sidecar audit. Do not spend an early-day slot on it.

## Artifacts
- Metrics: `artifacts/soundscape_sequence_mining/soundscape-sequence-panns-cnn14-allcls-r2-localmax-filemil-losite-ep20-20260528/metrics.json`
- Config: `configs/birdclef/soundscape_sequence_panns_cnn14_allcls_r2_localmax_filemil_losite_ep20_20260528.json`
- Sidecar audit: `artifacts/soundscape_sequence_sidecar_audit/20260528T0820Z_panns_localmax_filemil/audit_summary.json`
