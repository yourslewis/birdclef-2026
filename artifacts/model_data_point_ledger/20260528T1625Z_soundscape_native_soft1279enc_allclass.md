# Model Data Point — Soft1279 encoder-only native all-class LOSO

Timestamp: 2026-05-28 16:25 UTC

## Summary

Trained a one-variable ablation of the soft1279-initialized soundscape-native all-class B0 model with the soft1279 encoder loaded but the 234-class head re-initialized (`initial_load_head=false`). This tested whether the prior run's weak LOSO metrics were caused by importing the train-audio OOF-teacher head. The encoder-only ablation performed much worse, indicating that the soft1279 head was not the cause of the LOSO regression and may be necessary for the local sidecar lift seen in the head-loaded package.

## Ledger

- **Model family:** soundscape-native B0 calibration/domain adaptation, soft1279 encoder-only init.
- **Training data:** official `train_soundscapes`, 1,478 5s windows / 66 files / 9 sites.
- **Targets:** all 234 taxonomy labels.
- **Model/init:** EfficientNet-B0 SED, initialized from `sed-b0-oofteacher-b0v26-nfnetv29-soft-1279-ep4-20260528/model_torchscript.pt`; `initial_load_head=false`, 352 encoder keys loaded, 2 head keys skipped.
- **Training:** 4 LOSO epochs + 4 final-train epochs, LR `1e-4`, BCE, label smoothing `0.01`, mixup `0.05`, sqrt positive class weighting, seed `73`.
- **Validation split:** leave-one-site; 7 completed folds, 2 skipped for too few validation windows.
- **Primary metric:** row macro ROC-AUC mean `0.506642`.
- **Secondary metrics:** no-train AUC `0.552063`; non-Aves AUC `0.467678`; file-MIL AUC `0.460169`; pooled row AUC `0.311630`; pooled no-train AUC `0.156827`; OOF predictions finite/nonconstant `1410x234` with `234/234` nonconstant columns.
- **Baseline/delta:** vs soft1279-head-loaded native all-class row `-0.093718` / file-MIL `-0.145636`; vs native B0 q3/cap80 all-class row `-0.129519` / file-MIL `-0.213587`.
- **Export/runtime status:** TorchScript and ONNX exported/checked on trainer GPU1; summed fold runtime about `91.759s`; smoke finite `(2,160,313)->(2,234)`.
- **Decision:** **reject unchanged; no direct package as candidate.** Keep as evidence that encoder-only soft1279 init is harmful for soundscape LOSO and that the head-loaded package's local proxy lift is not explained by encoder transfer alone.

## Artifacts

- Config: `configs/birdclef/soundscape_native_b0_soft1279enc_losite_allcls_ep4_20260528.json`
- Metrics: `artifacts/soundscape_native_losite/soundscape-native-b0-soft1279enc-losite-allcls-ep4-20260528/metrics.json`
- OOF predictions: `artifacts/soundscape_native_losite/soundscape-native-b0-soft1279enc-losite-allcls-ep4-20260528/leave_site_predictions.npz`
- Log: `logs/soundscape_native_b0_soft1279enc_losite_allcls_ep4_20260528.log`
