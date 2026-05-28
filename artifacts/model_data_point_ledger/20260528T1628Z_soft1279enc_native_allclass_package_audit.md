# Model Data Point — Soft1279 encoder-only native all-class package/audit

Timestamp: 2026-05-28 16:28 UTC

## Summary

Packaged the soft1279 encoder-only native all-class model through the TorchScript soundscape inference path and audited it as a v616-compatible sidecar. Unlike the head-loaded model, the encoder-only package did not beat v616 locally.

## Ledger

- **Model/eval family:** soundscape-native encoder-only package audit.
- **Source model:** `soundscape-native-b0-soft1279enc-losite-allcls-ep4-20260528`; all 234 labels.
- **Packaging/inference:** `scripts/birdclef_single_sed_package_sidecar_audit.py` generated a TorchScript manifest and ran soundscape inference over 66 labeled train_soundscapes; 792 inference rows mapped to 240/240 proxy rows.
- **Primary metric:** best non-control recipe `soft1279enc_native_allcls_w0p08` local macro AUC `0.993144` / 42 valid classes.
- **Secondary metrics:** lift vs v616 `-0.000337`; lift vs anchor `+0.002753`; rank corr vs v616 `0.996203`; MAE `0.022953`.
- **Baseline/delta:** vs v616 local `-0.000337`; vs head-loaded soft1279-init `w0p08` sidecar `-0.001669` local AUC and vs expanded `w0p16` `-0.002401` local AUC.
- **Export/runtime status:** TorchScript inference/audit OK after rerunning with visible GPU `cuda:0`; finite/nonconstant `240x234` sidecar; `submit_approved=false`.
- **Decision:** **reject slot candidate.** This ablation supports keeping/inspecting the head-loaded calibrated sidecar rather than encoder-only transfer.

## Artifacts

- Package/audit dir: `artifacts/sed_soundscape_packaging_audit/20260528T1628Z_soft1279enc_native_allcls_package/`
- Manifest: `artifacts/sed_soundscape_packaging_audit/20260528T1628Z_soft1279enc_native_allcls_package/manifest.json`
- Audit JSON: `artifacts/sed_soundscape_packaging_audit/20260528T1628Z_soft1279enc_native_allcls_package/audit/ensemble_strategy_audit.json`
- Audit summary: `artifacts/sed_soundscape_packaging_audit/20260528T1628Z_soft1279enc_native_allcls_package/audit_summary.json`
- Sidecar CSV: `artifacts/sed_soundscape_packaging_audit/20260528T1628Z_soft1279enc_native_allcls_package/sidecars/soft1279enc_native_allcls_sidecar_234_anchorfill.csv`
- Log: `logs/soft1279enc_native_allcls_package_audit_20260528T1628Z.log`
