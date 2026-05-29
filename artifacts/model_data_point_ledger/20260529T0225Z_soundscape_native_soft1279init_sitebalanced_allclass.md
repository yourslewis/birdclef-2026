# Model Data Point — Soft1279-init native all-class with site-balanced sampling

Timestamp: 2026-05-29 02:25 UTC

## Summary

Trained a one-variable site-balanced sampling ablation of the head-loaded soft1279-initialized soundscape-native EfficientNet-B0 all-class model. This tested whether rebalancing the under-mined `train_soundscapes` sites could improve leave-site robustness or convert the prior soft1279 sidecar lift into a safer candidate.

## Ledger

- **Model family:** soundscape-native B0 calibration/domain adaptation, soft1279 head-loaded init, site-balanced epoch sampling.
- **Training data:** official `train_soundscapes`, 1,478 5s windows / 66 files / 9 sites.
- **Targets:** all 234 taxonomy labels.
- **Model/init:** EfficientNet-B0 SED, initialized from `sed-b0-oofteacher-b0v26-nfnetv29-soft-1279-ep4-20260528/model_torchscript.pt`, `initial_load_head=true`.
- **Training:** 4 LOSO epochs + 4 final-train epochs, LR `1e-4`, BCE, label smoothing `0.01`, mixup `0.05`, sqrt positive class weighting, `train_sampling=site_balanced`, seed `71`.
- **Validation split:** leave-one-site; 7 completed folds, 2 skipped.
- **Primary metric:** row macro ROC-AUC mean `0.569405`.
- **Secondary metrics:** no-train AUC `0.559505`; non-Aves AUC `0.545574`; file-MIL AUC `0.513779`; pooled row AUC `0.359269`; OOF predictions finite/nonconstant `1410x234` with `234/234` nonconstant columns.
- **Baseline/delta:** vs prior soft1279 head-loaded native all-class row `-0.030955` / file-MIL `-0.092026`; vs q3 native B0 all-class row `-0.066756` / file-MIL `-0.159977`; vs PANNs all-class no-file row `-0.078411` / file-MIL `-0.156944`.
- **Export/runtime status:** TorchScript and ONNX exported/checked on trainer GPU1; runtime `74.532s`; smoke finite `(2,160,313)->(2,234)`.
- **Decision:** **reject unchanged; no direct submission.** Site-balanced resampling harmed both row and file-MIL metrics and did not fix soft1279 site robustness.

## Artifacts

- Config: `configs/birdclef/soundscape_native_b0_soft1279init_sitebalanced_losite_allcls_ep4_20260529.json`
- Metrics: `artifacts/soundscape_native_losite/soundscape-native-b0-soft1279init-sitebalanced-losite-allcls-ep4-20260529/metrics.json`
- OOF predictions: `artifacts/soundscape_native_losite/soundscape-native-b0-soft1279init-sitebalanced-losite-allcls-ep4-20260529/leave_site_predictions.npz`
- Log: `logs/soundscape_native_b0_soft1279init_sitebalanced_losite_allcls_ep4_20260529.log`
