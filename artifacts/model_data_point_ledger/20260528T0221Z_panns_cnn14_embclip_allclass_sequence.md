# 2026-05-28 02:21 UTC — PANNs/Cnn14 embedding+AudioSet-tag all-class sequence data point

## Experiment
- **ID:** `soundscape-sequence-panns-cnn14-embclip-allcls-r2-nofile-reg-losite-ep18-20260528`
- **Family:** train_soundscapes sequence/file/site AudioSet semantic mining
- **Data:** official `train_soundscapes`; 1,478 windows / 66 files / 9 sites
- **Targets:** all 234 taxonomy labels
- **Model/init:** frozen PANNs/Cnn14 AudioSet features, concatenating z-scored 2048-d embedding with z-scored 527-d AudioSet clipwise tag logits, then radius-2 no-file context MLP (local mean/max + time features).
- **Validation:** leave-one-site; 7 completed folds.

## Metrics
- **Primary:** context row macro AUC `0.609194` over fold-valid classes, versus row-only `0.614447` (Δ `-0.005253`).
- **Secondary:** file-MIL AUC `0.668715` versus row-only `0.671901` (Δ `-0.003186`); no-train AUC `0.550128`; non-Aves AUC `0.642714`.
- **Export/runtime:** TorchScript exported; final all-row predictions finite/nonconstant `234/234`; trainer ran on `CUDA_VISIBLE_DEVICES=1`.

## Fold deltas
- S03: row 0.886201 → context 0.915625 (Δ +0.029424)
- S08: row 0.502087 → context 0.510639 (Δ +0.008552)
- S13: row 0.796354 → context 0.738351 (Δ -0.058003)
- S15: row 0.572846 → context 0.534147 (Δ -0.038698)
- S19: row 0.635800 → context 0.636220 (Δ +0.000420)
- S22: row 0.380777 → context 0.384368 (Δ +0.003591)
- S23: row 0.527061 → context 0.545008 (Δ +0.017947)

## Comparison
- Versus PANNs/Cnn14 all-class no-file context baseline: row Δ `-0.038622`, file-MIL Δ `-0.002008`.
- Versus PANNs localmax-only: row Δ `-0.032307`, file-MIL Δ `-0.013038`.

## Decision
**Reject unchanged / keep data point.** Adding AudioSet tag logits diluted the embedding-only PANNs context signal: row and file-MIL both regressed versus the best PANNs no-file/localmax variants. It is useful negative evidence for the broad acoustic-tag lane, but not a package/submission candidate.

## Artifacts
- Metrics: `artifacts/soundscape_sequence_mining/soundscape-sequence-panns-cnn14-embclip-allcls-r2-nofile-reg-losite-ep18-20260528/metrics.json`
- Config: `configs/birdclef/soundscape_sequence_panns_cnn14_embclip_allcls_r2_nofile_reg_losite_ep18_20260528.json`
- Embedding cache: `artifacts/panns_soundscape_embeddings/panns-cnn14-audioset-embclip-combined-20260528/panns_embclip_embeddings.npz`
- Train log: `logs/soundscape_sequence_panns_embclip_allcls_20260528T0218Z.log`
