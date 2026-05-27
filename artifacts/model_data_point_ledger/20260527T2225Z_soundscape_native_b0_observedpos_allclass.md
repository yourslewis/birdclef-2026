# BirdCLEF model data point — soundscape-native B0 observed-positive all-class LOSO — 2026-05-27 22:25 UTC

## Summary
- **Experiment id:** `soundscape-native-b0-losite-allcls-observedpos-ep4-20260527`
- **Branch family:** deeper soundscape-native compact CNN/SED; observed positive-rate class weighting ablation
- **Evidence level:** comparison-grade model data point; not submission-grade
- **Decision:** **reject unchanged / do not package or submit.** Observed-sqrt class weights reduced the prior native B0 row and file-MIL metrics, especially file-MIL, so the previous native-B0 `pos_weight_sqrt` data point remains the better native-CNN reference.

## Live competition / slot state
- Best known public LB at check: `0.949` (v616/v621/v622/v623 tied before this UTC day; current v626/v628/v629 still pending at the 22:25 UTC recount).
- 2026-05-27 UTC slots: `5/5`; reset in ~`1.6h`.
- Newly scored from the late-fill set: v627 `0.928`, v630 `0.917`; v626/v628/v629 still pending.

## Data / targets
- Source: official `train_soundscapes` only
- Windows/files/sites: 1,478 windows / 66 files / 9 sites
- Target scope: all 234 taxonomy labels from soundscape-label supervision
- Split: leave-one-site validation, 7 completed folds / 1 skipped fold (`S18`, too few validation windows)

## Model / training
- Model/init: EfficientNet-B0 SED native soundscape fine-tune from `xc-b0-q3-cap80-external-pretrain-balanced-ep12`, head reinitialized for 234 labels
- Input: 5s windows, 32 kHz, 160-mel logmel
- Change vs previous native B0 all-class: `class_balancing=observed_sqrt` instead of uniform `pos_weight_sqrt`.
- Other key settings: BCE, label smoothing `0.01`, mixup `0.1`, 4 LOSO epochs + 4-epoch final train
- Runtime/device: `98.8s` on trainer CUDA with `CUDA_VISIBLE_DEVICES=1`

## Comparable performance
| Metric | Value | Notes |
|---|---:|---|
| row macro AUC mean | 0.624340 | leave-one-site fold mean, 7 folds |
| no-train row AUC | 0.615515 | 28 no-train labels where valid |
| non-Aves row AUC | 0.585486 | non-Aves subset where valid |
| file-MIL AUC mean | 0.582914 | max-pooled by file |
| pooled row AUC | 0.380753 | pooled OOF, 71 valid classes |
| pooled no-train AUC | 0.260271 | pooled OOF no-train, 28 valid classes |

## Baseline comparison
- Versus **prior native B0 all-class pos-weight baseline** (`20260527T2020Z`): row `-0.011821`, file-MIL `-0.090842`, no-train `-0.010569`, non-Aves `-0.032551`.
- Versus **PANNs/Cnn14 all-class r2 no-file context** (`20260527T1220Z`): row `-0.023476`, file-MIL `-0.087809`.
- Versus **DyMN10 all-class r2 no-file context** (`20260527T0618Z`): row `+0.026707`, file-MIL `-0.052371`.

## Export / verifier notes
- TorchScript export: `artifacts/soundscape_native_losite/soundscape-native-b0-losite-allcls-observedpos-ep4-20260527/model_torchscript.pt` (15.392 MB on trainer)
- ONNX export: `exported_checked`; path `artifacts/soundscape_native_losite/soundscape-native-b0-losite-allcls-observedpos-ep4-20260527/model.onnx` (0.568 MB plus external data on trainer)
- TorchScript smoke: finite `[2, 234]` clip logits and `[2, 10, 234]` frame logits
- OOF predictions: finite/nonconstant `1410x234`, 234 nonconstant columns
- Heavy binary exports retained on trainer; local repo stores metrics/config/prediction manifest only.

## Critic / verifier decision
- **Critic:** this is a useful one-variable regularization data point, but the observed class weighting degrades the strongest reason to keep native B0 (file-MIL), so do not continue this exact variant.
- **Verifier:** comparison artifacts are finite/nonconstant and exports smoke-tested. No Kaggle submission approved because slots are capped and this branch is weaker than the nearest native/PANNs baselines.

## Artifacts
- Metrics: `artifacts/soundscape_native_losite/soundscape-native-b0-losite-allcls-observedpos-ep4-20260527/metrics.json`
- Lightweight local artifact root: `artifacts/soundscape_native_losite/soundscape-native-b0-losite-allcls-observedpos-ep4-20260527/`
- Config: `configs/birdclef/soundscape_native_b0_losite_allcls_observedpos_ep4_20260527.json`
- Remote heavy exports: `/home/yourslewis/birdclef-2026/artifacts/soundscape_native_losite/soundscape-native-b0-losite-allcls-observedpos-ep4-20260527/`
