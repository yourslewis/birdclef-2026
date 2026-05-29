# 2026-05-29 04:35 UTC — Native B0 soft1279-init calibration-none all-class LOSO

## Summary
- Experiment: `soundscape-native-b0-soft1279init-calibnone-losite-allcls-ep4-20260529`.
- Branch family: soundscape-native calibration/domain-adaptation ablation.
- Objective: test whether preserving the soft1279 OOF-teacher calibration by removing class weights, mixup, and label smoothing improves transfer versus the prior head-loaded soft1279 native model.
- Data: official `train_soundscapes`, 1,478 windows / 66 files / 9 sites / 234 labels.
- Split: leave-one-site; 7 completed folds, 2 skipped low-information sites.

## Performance
| Metric | Value |
|---|---:|
| Row macro ROC-AUC mean | 0.585879 |
| No-train ROC-AUC mean | 0.561323 |
| Non-Aves ROC-AUC mean | 0.534311 |
| File-MIL ROC-AUC mean | 0.526705 |
| Pooled row ROC-AUC | 0.347981 |
| OOF prediction shape | 1410 x 234 |
| Nonconstant columns | 234 |

## Baseline deltas
- Versus head-loaded soft1279 native all-class baseline: row `-0.014481`, file-MIL `-0.079100`.
- Versus site-balanced soft1279 native ablation: row `+0.016474`, file-MIL `+0.012926`.

## Export/runtime
- Runtime: 84.595s on trainer GPU1.
- TorchScript: `artifacts/soundscape_native_losite/soundscape-native-b0-soft1279init-calibnone-losite-allcls-ep4-20260529/model_torchscript.pt`; smoke finite: `True`.
- ONNX: `exported_checked`.

## Decision
Reject unchanged. Removing class weights/mixup/smoothing recovered a little versus the site-balanced ablation but remained below the prior head-loaded soft1279 native model and transferred poorly in the sidecar package audit. Keep as a negative calibration data point; do not submit.
