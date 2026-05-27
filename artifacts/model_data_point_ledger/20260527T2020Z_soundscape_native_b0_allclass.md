# BirdCLEF model data point — soundscape-native B0 all-class LOSO — 2026-05-27 20:20 UTC

## Summary
- **Experiment id:** `soundscape-native-b0-losite-allcls-ep4-20260527`
- **Branch family:** deeper soundscape-native compact CNN/SED
- **Evidence level:** comparison-grade model data point; not submission-grade
- **Decision:** keep as a useful all-class native-CNN data point, but **do not submit/package directly**. It is close to PANNs embedding heads on leave-site metrics, but pooled OOF behavior is poor and no hidden-test package/v616 audit exists.

## Data / targets
- Source: official `train_soundscapes` only
- Windows/files/sites: 1,478 windows / 66 files / 9 sites
- Target scope: all 234 taxonomy labels from soundscape-label supervision
- Split: leave-one-site validation, 7 completed folds / 1 skipped fold (`S18`, too few validation windows)

## Model / training
- Model/init: EfficientNet-B0 SED native soundscape fine-tune from `xc-b0-q3-cap80-external-pretrain-balanced-ep12`, head reinitialized for 234 labels
- Input: 5s windows, 32 kHz, 160-mel logmel
- Loss/training: BCE/focal-style config with label smoothing/mixup, pos-weight sqrt balancing, 4 epochs + final 4-epoch train
- Runtime/device: 97.1s on trainer CUDA (`CUDA_VISIBLE_DEVICES=1`)

## Comparable performance
| Metric | Value | Notes |
|---|---:|---|
| row macro AUC mean | 0.636161 | leave-one-site, fold mean |
| no-train row AUC | 0.626084 | 28 no-train labels where valid |
| non-Aves row AUC | 0.618037 | non-Aves subset where valid |
| file-MIL AUC mean | 0.673756 | max-pooled by file |
| pooled row AUC | 0.328850 | poor pooled OOF transfer signal |
| pooled no-train AUC | 0.180350 | poor pooled OOF no-train signal |

## Baseline comparison
- Versus **PANNs/Cnn14 all-class r2 no-file context** (`20260527T1220Z`): row `-0.011655`, file-MIL `+0.003033`.
- Versus **PANNs/Cnn14 all-class file-context** (`20260527T1818Z`): row `-0.006041`, file-MIL `+0.021105`.
- Versus **DyMN10 all-class r2 no-file context** (`20260527T0618Z`): row `+0.038528`, file-MIL `+0.038471`.

## Export / verifier notes
- TorchScript export: `artifacts/soundscape_native_losite/soundscape-native-b0-losite-allcls-ep4-20260527/model_torchscript.pt` (15.392 MB on trainer)
- ONNX export: `exported_checked`; path `artifacts/soundscape_native_losite/soundscape-native-b0-losite-allcls-ep4-20260527/model.onnx` (0.568 MB plus external data on trainer)
- TorchScript smoke: finite `[2, 234]` clip logits and `[2, 10, 234]` frame logits
- OOF predictions: finite/nonconstant `1410x234`, 234 nonconstant columns
- Heavy binary exports retained on trainer; local repo stores metrics/config/prediction manifest only.

## Artifacts
- Metrics: `artifacts/soundscape_native_losite/soundscape-native-b0-losite-allcls-ep4-20260527/metrics.json`
- Lightweight local artifact root: `artifacts/soundscape_native_losite/soundscape-native-b0-losite-allcls-ep4-20260527/`
- Remote heavy exports: `/home/yourslewis/birdclef-2026/artifacts/soundscape_native_losite/soundscape-native-b0-losite-allcls-ep4-20260527/`
