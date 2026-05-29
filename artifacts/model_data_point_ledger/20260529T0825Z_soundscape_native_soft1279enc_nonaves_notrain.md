# Model Data Point — Soft1279 encoder-init native non-Aves/no-train specialist

Timestamp: 2026-05-29 08:25 UTC

## Summary

Trained and packaged a scoped soundscape-native EfficientNet-B0 specialist over the 72 non-Aves/no-train labels. This is a measured train-soundscape data point for the under-mined non-Aves/no-train lane, not a submission candidate.

## Training ledger

- **Experiment id:** `soundscape-native-b0-soft1279enc-nonaves-notrain-losite-ep5-20260529`
- **Branch family:** deeper soundscape-native non-Aves/no-train specialist.
- **Data:** official `train_soundscapes`; 1,478 5s windows / 66 files / 9 sites.
- **Targets:** 72 labels (`nonaves_or_no_train`); includes 28 no-train-primary labels.
- **Model/init:** EfficientNet-B0 SED head over 5s 32kHz/160-mel windows; soft1279 OOF-teacher TorchScript loaded as encoder init (352 keys loaded; head skipped due 72-label scope).
- **Validation split:** leave-one-site; completed 6 folds, skipped 3 low-count/low-valid folds.
- **Primary metric:** row macro ROC-AUC mean `0.609793`.
- **Secondary metrics:** no-train AUC `0.613437`; non-Aves AUC `0.609793`; file-MIL AUC `0.551016`; pooled row AUC `0.293648` / 45 valid; pooled no-train AUC `0.151414` / 27 valid.
- **Export/runtime:** TorchScript and ONNX exported; runtime `78.752s`; TorchScript smoke finite, output `[2,72]`.

## Package / sidecar audit

- **Inference:** 240 / 240 v616 proxy rows matched; 72 scoped columns updated; all 234 competition columns nonconstant/finite after preserving anchor values for missing labels.
- **Best audit row:** `soft1279enc_nonaves_notrain_member_raw` local macro AUC `0.993828` / 42 valid classes; lift vs v616 `+0.000347`; lift vs anchor `+0.003437`; MAE vs v616 `0.146383`.
- **Low-weight blends:** all tested rank blends (`0.25%`–`12%`) remained below v616; the raw anchor-preserved member is the only local lift, with high displacement and failed promotion gates.
- **Implementation note:** fixed `scripts/birdclef_single_sed_package_sidecar_audit.py` so scoped/specialist models preserve anchor values for labels they do not emit instead of zero-filling missing columns.

## Decision

Reject as a slot candidate. Keep as a comparison-grade landscape point: soft1279 encoder init recovers some site-mean row AUC for frogs/non-Aves, but pooled/site-mixed no-train sonotype behavior is poor and package lift vs v616 is only `+0.000347`, below the `+0.001` gate. Next action should be a stronger site-pair/sonotype-specific validation or a hand-verified background/no-call audit, not a leaderboard slot.

## Artifact paths

- Config: `configs/birdclef/soundscape_native_b0_soft1279enc_nonaves_notrain_losite_ep5_20260529.json`
- Metrics: `artifacts/soundscape_native_losite/soundscape-native-b0-soft1279enc-nonaves-notrain-losite-ep5-20260529/metrics.json`
- Model export: `artifacts/soundscape_native_losite/soundscape-native-b0-soft1279enc-nonaves-notrain-losite-ep5-20260529/model_torchscript.pt`
- Package audit: `artifacts/sed_soundscape_packaging_audit/20260529T0825Z_soft1279enc_nonaves_notrain_package/audit_anchorpreserve_summary.json`
- Audit JSON: `artifacts/sed_soundscape_packaging_audit/20260529T0825Z_soft1279enc_nonaves_notrain_package/audit_anchorpreserve/ensemble_strategy_audit.json`
