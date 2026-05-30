# Model Data Point — Native B0 soft1279-init focal2 all-class LOSO

Timestamp: 2026-05-30 18:20 UTC

## Summary

Trained a new soundscape-native EfficientNet-B0 data point from the soft1279 head-loaded checkpoint using focal BCE (`gamma=2.0`) instead of the prior BCE objective. This tested whether the original head-loaded soft1279 sidecar signal could be retained while improving rare/hard positives under leave-site validation.

Result: focal training did **not** improve the lane. LOSO row AUC is almost flat vs the original soft1279-head-loaded all-class model, but file-MIL drops sharply, and the packaged sidecar falls below v616 locally. This closes the focal-objective variant for now.

## Training ledger

- **Branch family:** soundscape-native calibration/domain adaptation / focal-objective ablation.
- **Training data:** official train_soundscapes, `1478` windows / `66` files / `9` sites.
- **Target scope:** all `234` taxonomy labels.
- **Model/init:** EfficientNet-B0 SED, soft1279 OOF-teacher TorchScript checkpoint, head loaded, full model trainable; focal BCE gamma `2.0`.
- **Validation split:** leave-one-site; `7` completed folds / `2` skipped.
- **Primary metric:** LOSO row macro AUC `0.599447`.
- **Secondary metrics:** no-train AUC `0.550339`; non-Aves AUC `0.583959`; file-MIL AUC `0.540916`; pooled row AUC `0.445561` / `71` valid; pooled no-train AUC `0.491300` / `28` valid.
- **Baseline/delta:** vs original head-loaded soft1279 all-class LOSO row `0.600360` = `-0.000913`; file-MIL `0.605805` = `-0.064889`.
- **Export/runtime status:** TorchScript and ONNX exported/checked; runtime `72.443s`; finite nonconstant OOF predictions (`1410x234`, `234` nonconstant columns).
- **Decision:** **reject unchanged; no submission.** Keep as a negative data point; focal objective hurts the transferable sidecar.

## Package/audit ledger

- **Wrapper/audit data:** train_soundscapes inference produced `792` rows; v616 proxy wrapper matched `240/240` rows; finite, `234` nonconstant class columns.
- **Primary sidecar metric:** best non-control recipe `soft1279init_focal2_allcls_w0p08` local AUC `0.992675` / `42` valid.
- **Secondary sidecar metrics:** lift vs v616 `-0.000805`; lift vs anchor `+0.002285`; rank corr vs v616 `0.996452`; MAE `0.020961`; submit_approved `False`.
- **Baseline/delta:** v616 local `0.993481`; best focal sidecar delta `-0.000805`. Prior head-loaded soft1279 w0.16 local was `0.995545`, so focal is `-0.002870` lower.
- **Decision:** **reject slot candidate.** Package is valid but below v616 and promotion gates.

## Artifacts

- Config: `configs/birdclef/soundscape_native_b0_soft1279init_focal2_losite_allcls_ep4_20260530.json`
- Training dir: `artifacts/soundscape_native_losite/soundscape-native-b0-soft1279init-focal2-losite-allcls-ep4-20260530`
- Training metrics: `artifacts/soundscape_native_losite/soundscape-native-b0-soft1279init-focal2-losite-allcls-ep4-20260530/metrics.json`
- Package/audit dir: `artifacts/sed_soundscape_packaging_audit/20260530T1820Z_soft1279init_focal2_allcls_package`
- Sidecar build report: `artifacts/sed_soundscape_packaging_audit/20260530T1820Z_soft1279init_focal2_allcls_package/sidecar_build_report.json`
- Audit summary: `artifacts/sed_soundscape_packaging_audit/20260530T1820Z_soft1279init_focal2_allcls_package/audit_summary.json`
