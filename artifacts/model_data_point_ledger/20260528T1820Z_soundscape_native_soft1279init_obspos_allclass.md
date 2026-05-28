# Model Data Point — Soft1279-init observed-positive native all-class LOSO

Timestamp: 2026-05-28 18:20 UTC

## Summary

Trained a one-variable class-weighting ablation of the head-loaded soft1279-initialized soundscape-native all-class B0 model: `class_balancing=observed_sqrt` instead of the prior constant `pos_weight_sqrt`. This tested whether sparse observed-positive weighting would improve soundscape LOSO transfer or preserve the promising soft1279 head-loaded sidecar signal.

It regressed sharply versus the head-loaded soft1279 baseline and is rejected unchanged.

## Ledger

- **Model family:** soundscape-native B0 calibration/domain adaptation, soft1279 head-loaded observed-positive weighting.
- **Training data:** official `train_soundscapes`, 1,478 5s windows / 66 files / 9 sites.
- **Targets:** all 234 taxonomy labels.
- **Model/init:** EfficientNet-B0 SED initialized from `sed-b0-oofteacher-b0v26-nfnetv29-soft-1279-ep4-20260528/model_torchscript.pt`, `initial_load_head=true`.
- **Training:** 4 LOSO epochs + 4 final-train epochs, LR `1e-4`, BCE, label smoothing `0.01`, mixup `0.05`, observed-sqrt positive class weighting, seed `71`.
- **Validation split:** leave-one-site; 7 completed folds, 2 skipped for too few validation windows/classes.
- **Primary metric:** row macro ROC-AUC mean `0.569148`.
- **Secondary metrics:** no-train AUC `0.506363`; non-Aves AUC `0.532133`; file-MIL AUC `0.474353`; pooled row AUC `0.508009`; pooled no-train AUC `0.537156`; OOF predictions finite/nonconstant `1410x234` with `234/234` nonconstant columns.
- **Baseline/delta:** vs head-loaded soft1279 native all-class row `-0.031212` / file-MIL `-0.131452`.
- **Export/runtime status:** TorchScript and ONNX exported/checked on trainer GPU1; runtime `75.942s`; smoke finite `(2,160,313)->(2,234)`.
- **Decision:** **reject unchanged; no submission.** Observed-positive weighting damaged LOSO and package proxy transfer; keep as negative evidence against this weighting path.

## Artifacts

- Config: `configs/birdclef/soundscape_native_b0_soft1279init_obspos_losite_allcls_ep4_20260528.json`
- Metrics: `artifacts/soundscape_native_losite/soundscape-native-b0-soft1279init-obspos-losite-allcls-ep4-20260528/metrics.json`
- OOF predictions: `artifacts/soundscape_native_losite/soundscape-native-b0-soft1279init-obspos-losite-allcls-ep4-20260528/leave_site_predictions.npz`
- Log: `logs/soundscape_native_b0_soft1279init_obspos_losite_allcls_ep4_20260528.log`
