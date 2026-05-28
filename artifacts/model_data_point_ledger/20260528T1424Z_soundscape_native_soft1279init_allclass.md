# Model Data Point — Soft1279-initialized soundscape-native B0 all-class LOSO

Timestamp: 2026-05-28 14:24 UTC

## Summary

Trained a soundscape-native EfficientNet-B0 SED model initialized from the strong `sed-b0-oofteacher-b0v26-nfnetv29-soft-1279-ep4-20260528` TorchScript checkpoint, loading both encoder and 234-class head, then adapted on official `train_soundscapes` all-class labels with leave-site validation. This tests whether the high OOF-teacher train-audio score becomes useful after explicit soundscape-domain adaptation.

## Ledger

- **Model family:** soundscape-native B0 calibration/domain adaptation from OOF-teacher soft1279 init.
- **Training data:** official `train_soundscapes`, 1,478 5s windows / 66 files / 9 sites.
- **Targets:** all 234 taxonomy labels from `train_soundscapes_labels.csv`.
- **Model/init:** EfficientNet-B0 SED, 5s/160-mel; initialized from `artifacts/sed_oof_teacher_students/sed-b0-oofteacher-b0v26-nfnetv29-soft-1279-ep4-20260528/model_torchscript.pt`, `initial_load_head=true`, 354 keys loaded.
- **Training:** 4 LOSO epochs + 4 final-train epochs, LR `1e-4`, BCE, label smoothing `0.01`, mixup `0.05`, sqrt positive class weighting, seed `71`.
- **Validation split:** leave-one-site; 7 completed folds, 2 skipped for too few validation windows.
- **Primary metric:** row macro ROC-AUC mean `0.600360`.
- **Secondary metrics:** no-train AUC `0.568181`; non-Aves AUC `0.604565`; file-MIL AUC `0.605805`; pooled row AUC `0.377836`; pooled no-train AUC `0.320312`; OOF predictions finite/nonconstant `1410x234` with `234/234` nonconstant columns.
- **Baseline/delta:** vs prior native B0 all-class q3/cap80 init, row `-0.035801` and file-MIL `-0.067951`; vs PANNs all-class no-file, row `-0.047456` and file-MIL `-0.064918`.
- **Export/runtime status:** TorchScript and ONNX exported/checked on trainer GPU1; runtime `75.694s`; smoke finite `(2,160,313)->(2,234)`.
- **Decision:** **reject as a direct training data point**; soft1279 head initialization hurt LOSO row/file metrics. Package/audit separately because the adapted model may still be useful as a calibrated sidecar.

## Artifacts

- Config: `configs/birdclef/soundscape_native_b0_soft1279init_losite_allcls_ep4_20260528.json`
- Metrics: `artifacts/soundscape_native_losite/soundscape-native-b0-soft1279init-losite-allcls-ep4-20260528/metrics.json`
- OOF predictions: `artifacts/soundscape_native_losite/soundscape-native-b0-soft1279init-losite-allcls-ep4-20260528/leave_site_predictions.npz`
- Trainer model/export dir: `~/birdclef-2026/artifacts/soundscape_native_losite/soundscape-native-b0-soft1279init-losite-allcls-ep4-20260528/`
- Log: `logs/soundscape_native_b0_soft1279init_losite_allcls_ep4_20260528.log`
