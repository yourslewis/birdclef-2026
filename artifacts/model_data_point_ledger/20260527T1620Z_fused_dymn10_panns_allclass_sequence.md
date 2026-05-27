# Fused DyMN10 + PANNs all-class train_soundscapes sequence data point — 2026-05-27 16:20 UTC

## Status
- Evidence level: comparison-grade model/data point plus no-submit v616 proxy sidecar audit.
- Slot decision: no Kaggle submission. The fused head is finite/nonconstant and mildly improves file-MIL over PANNs, but row AUC is worse than PANNs-only and the v616 proxy sidecar still loses to v616.

## Live competition state
- Latest Kaggle Bearer check before training: best public LB remains `0.949`.
- Latest scored submissions: v621/v622/v623 tied `0.949`, v625 `0.948`, v624 `0.943`; v616 remains a tied baseline to beat.
- 2026-05-27 UTC slots used at start: `0/5`; ~7.7h to reset.
- No active BirdCLEF local/trainer jobs before training; trainer GPU1 was available.

## Experiment
- Experiment id: `soundscape-sequence-fused-dymn10-panns-allcls-r2-nofile-reg-losite-ep18-20260527`.
- Branch family: train_soundscapes sequence/file/site AudioSet fusion mining.
- Training data: official `train_soundscapes` only, `1,478` 5s windows / `66` files / `9` sites.
- Target scope: all `234` taxonomy labels from soundscape supervision.
- Model/init: z-scored concat of EfficientAT DyMN10 embeddings (`960` dim) + PANNs/Cnn14 AudioSet embeddings (`2048` dim), radius-2 no-file context MLP (`15,044` context features, hidden `512`, dropout `0.45`, AdamW, site-balanced BCE, 18 epochs).
- Validation split: leave-one-site, 7 completed folds (`S03`, `S08`, `S13`, `S15`, `S19`, `S22`, `S23`).

## Results

| Metric | Row-only | Context | Delta |
|---|---:|---:|---:|
| row macro AUC mean | 0.553997 | 0.596642 | +0.042645 |
| file-MIL macro AUC mean | 0.654035 | 0.675982 | +0.021947 |
| no-train row AUC mean | 0.506297 | 0.548856 | +0.042559 |
| non-Aves row AUC mean | 0.606875 | 0.636103 | +0.029229 |

Fold context deltas vs row-only: S03 `+0.070252`, S08 `-0.031358`, S13 `+0.090603`, S15 `-0.005824`, S19 `+0.089412`, S22 `+0.041470`, S23 `+0.043961`.

## Comparison to nearest baselines
- Versus PANNs/Cnn14 all-class r2 context (`0.647816` row / `0.670723` file-MIL): fused is row `-0.051174`, file-MIL `+0.005259`.
- Versus DyMN10 all-class r2 context (`0.597633` row / `0.635285` file-MIL): fused is row `-0.000991`, file-MIL `+0.040697`.
- Interpretation: fusion did not improve row ranking over PANNs-only, but it slightly improved file-MIL and fixed several non-S08 folds; S08 and S15 still regress, so this is not a promotion-grade branch.

## Sidecar audit vs v616
- Audit artifact: `artifacts/soundscape_allclass_sidecar_audit/20260527T1620Z_fused_dymn10_panns_allclass_sequence/`.
- Wrapper: 240 proxy rows / 234 columns; 156 matched sequence rows, 84 anchor-filled rows; finite/nonconstant `240x234` CSV on trainer.
- Best recipe: `allcls_seq_w0p005` local macro AUC `0.991215` / `42` valid classes.
- Lift vs anchor: `+0.000824`.
- Lift vs v616: `-0.002266`.
- Rank corr vs v616: `0.999674`; MAE vs v616 `0.005927`.
- Promotion gate: failed; not submission-grade.

## Verifier / critic decision
- Verifier: fused embedding rows matched (`1478`); train/eval outputs finite; final all-row head nonconstant `234/234`; TorchScript export/smoke completed on trainer; sidecar audit schema/finite checks passed.
- Critic: high-dimensional fusion on only 66 files has overfit risk. It does not beat PANNs-only row AUC and still loses to v616 in the proxy audit. Do not submit; treat as a useful file-MIL datapoint and pivot to true hidden-test packaging/no-call protocol rather than more direct OOF proxy sidecars.
- Decision: keep comparison-grade data point; no submit; next action is either package PANNs/fused hidden-safe inference for audit or build the no-call/acoustic-background protocol.

## Artifact paths
- Config: `configs/birdclef/soundscape_sequence_fused_dymn10_panns_allcls_r2_nofile_reg_losite_ep18_20260527.json`
- Model artifact root: `artifacts/soundscape_sequence_mining/soundscape-sequence-fused-dymn10-panns-allcls-r2-nofile-reg-losite-ep18-20260527/` (full trainer artifact includes TorchScript/head/OOF predictions)
- Fused embedding artifact on trainer: `artifacts/fused_soundscape_embeddings/dymn10_panns_cnn14_train_soundscapes_20260527/fused_embeddings.npz`
- Sidecar audit root: `artifacts/soundscape_allclass_sidecar_audit/20260527T1620Z_fused_dymn10_panns_allclass_sequence/`
- Canonical table: `artifacts/model_data_point_ledger/performance_table.md`
