# Fused DyMN10+PANNs all-class file-context + file-MIL sequence data point — 2026-05-29 20:25 UTC

## Live state
- Public LB best: `0.949`; v616 remains tied repo-owned baseline to beat.
- Latest completed submissions at live check: v631 `0.926`, v632 `0.940`, v633 `0.946`, v634 `0.949`, v635 `0.941`.
- 2026-05-29 UTC slots used: `0/5`; ~3.7h to reset at run start, so mid-day policy still applied (late fill begins under 3h).
- Active jobs before training: no local/trainer BirdCLEF jobs; trainer GPUs free.

## Experiment
- Experiment id: `soundscape-sequence-fused-dymn10-panns-allcls-r2-filectx-filemil-losite-ep20-20260529`.
- Branch family: train_soundscapes sequence/file/site AudioSet fusion mining.
- Data: official `train_soundscapes`, 1,478 5s windows / 66 files / 9 sites.
- Target scope: all 234 taxonomy labels.
- Model/init: fused frozen EfficientAT DyMN10 + PANNs/Cnn14 embeddings, radius-2 previous/next + local mean/max + file mean/max + time features, MLP hidden 384, dropout 0.45, site-balanced sampling, file-MIL BCE weight 0.35, 20 epochs.
- Validation split: leave-one-site; 7 completed folds.

## Metrics
- Primary: context row macro ROC-AUC `0.594204` / 7 folds / 100 fold-class evaluations.
- Internal row-only baseline: `0.572355`; context delta `+0.021849`.
- File-MIL AUC: `0.678623` vs row-only `0.645253`; delta `+0.033370` / 92 fold-class evaluations.
- No-train AUC: `0.574279` / 41 fold-class evaluations.
- Non-Aves AUC: `0.645232` / 68 fold-class evaluations.
- Per-site deltas context-row: S03 -0.098704, S08 -0.016971, S13 +0.026075, S15 +0.012660, S19 +0.120052, S22 +0.062913, S23 +0.046916.
- Final all-row export smoke: TorchScript head saved; final predictions finite/nonconstant, 234/234 nonconstant columns, context dimension 21060.

## Comparisons
- Vs fused all-class r2 no-file context: row `-0.002438`, file-MIL `+0.002641`.
- Vs fused all-class localmax-only: row `+0.021442`, file-MIL `+0.008627`.
- Vs PANNs all-class no-file: row `-0.053612`, file-MIL `+0.007900`.
- Vs PANNs all-class localmax-only: row `-0.047297`, file-MIL `-0.003130`.
- Interpretation: file context + MIL recovered file-level strength and improved no-train/non-Aves over the fused localmax-only data point, but row AUC stayed slightly below the fused no-file baseline and well below PANNs all-class sequence leaders.

## Sidecar audit
- Wrapped all-class leave-site OOF predictions into the v616 proxy matrix and ran `birdclef_soundscape_allclass_sidecar_audit.py` with 200 bootstrap iterations.
- Best non-control recipe: `allcls_seq_w0p0025`.
- Local macro AUC `0.990981` / 42 valid classes.
- Lift vs v616 `-0.002499`; lift vs anchor `+0.000591`; rank corr vs v616 `0.999689`; MAE `0.006019`.
- Submit approved: `False`.

## Decision
- Reject as slot candidate; no submission. This is a comparison-grade data point only.
- The experiment supports the hypothesis that file context/file-MIL helps file-level aggregation, but its v616-proxy sidecar is weaker than prior fused localmax and far below promotion gates.
- Next: prioritize soft1279 head-loaded class/site movement diagnosis and prepare late-day valid public/source fill if no verifier-grade candidate appears under 3h to reset.

## Artifacts
- Config: `configs/birdclef/soundscape_sequence_fused_dymn10_panns_allcls_r2_filectx_filemil_losite_ep20_20260529.json`
- Training metrics: `artifacts/soundscape_sequence_mining/soundscape-sequence-fused-dymn10-panns-allcls-r2-filectx-filemil-losite-ep20-20260529/metrics.json`
- OOF predictions: `artifacts/soundscape_sequence_mining/soundscape-sequence-fused-dymn10-panns-allcls-r2-filectx-filemil-losite-ep20-20260529/leave_site_predictions.npz`
- TorchScript head: `artifacts/soundscape_sequence_mining/soundscape-sequence-fused-dymn10-panns-allcls-r2-filectx-filemil-losite-ep20-20260529/context_head_torchscript.pt`
- Sidecar audit: `artifacts/soundscape_allclass_sidecar_audit/20260529T2025Z_fused_allclass_filectx_filemil/audit_summary.json`
