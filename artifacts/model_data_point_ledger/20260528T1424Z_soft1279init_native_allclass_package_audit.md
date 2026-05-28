# Model Data Point — Soft1279-initialized native all-class package/audit

Timestamp: 2026-05-28 14:24 UTC

## Summary

Packaged the soft1279-initialized soundscape-native all-class model through the generic TorchScript soundscape inference path and audited it as a v616-compatible 234-class sidecar. Unlike the unadapted soft1279 package, the adapted native model produced a strong local proxy signal, but it still failed promotion gates and was not submitted.

## Ledger

- **Model family:** soundscape-native B0 calibration/domain-adapted hidden-safe package audit.
- **Source model:** `soundscape-native-b0-soft1279init-losite-allcls-ep4-20260528`; all 234 taxonomy labels.
- **Packaging/inference:** `scripts/birdclef_single_sed_package_sidecar_audit.py` generated a single-model TorchScript manifest and ran `scripts/birdclef_sed_soundscape_infer.py` over the 66 labeled train_soundscapes only. Runtime output: 792 row predictions / 234 labels.
- **Wrapper/audit data:** v616 proxy matrix, 240 rows x 234 classes; matched 240/240 proxy rows; finite and 234/234 nonconstant columns.
- **Primary metric:** best rank-blend recipe `soft1279init_native_allcls_w0p08` local macro AUC `0.994813` over 42 valid classes, lift vs v616 `+0.001332`.
- **Secondary metrics:** raw member AUC `0.994941` but high displacement (`rank_corr_vs_v616=0.204537`, `MAE=0.426876`); anchor local AUC `0.990391`; v616 local AUC `0.993481`; `w0p08` lift vs anchor `+0.004422`, rank corr vs v616 `0.996241`, MAE `0.023480`.
- **Bootstrap/gates:** `w0p08` passed matched-row, valid-class, lift-vs-v616, file-bootstrap, leave-one-site, and leave-one-file gates, but failed the stricter lift-vs-anchor and site-bootstrap q05 gates. `submit_approved=false` and `allow_submit_approval=false`.
- **Baseline/delta:** vs v616 local proxy `+0.001332` for best rank blend; vs unadapted soft1279 package best `+0.004169` local AUC and `+0.004169` lift-vs-v616 improvement.
- **Decision:** **revise/hold; no submission in early/mid UTC day.** This is the first current-run sidecar to beat v616 locally, but promotion gates fail and the required evidence remains comparison-grade, not submission-grade.

## Artifacts

- Package/audit dir: `artifacts/sed_soundscape_packaging_audit/20260528T1424Z_soft1279init_native_allcls_package/`
- Manifest: `artifacts/sed_soundscape_packaging_audit/20260528T1424Z_soft1279init_native_allcls_package/soft1279init_native_allcls_manifest.json`
- Soundscape predictions: `artifacts/sed_soundscape_packaging_audit/20260528T1424Z_soft1279init_native_allcls_package/train_soundscapes_soft1279init_native_allcls.csv`
- Sidecar CSV: `artifacts/sed_soundscape_packaging_audit/20260528T1424Z_soft1279init_native_allcls_package/sidecars/soft1279init_native_allcls_sidecar_234_anchorfill.csv`
- Audit JSON: `artifacts/sed_soundscape_packaging_audit/20260528T1424Z_soft1279init_native_allcls_package/audit/ensemble_strategy_audit.json`
- Audit summary: `artifacts/sed_soundscape_packaging_audit/20260528T1424Z_soft1279init_native_allcls_package/audit_summary.json`
- Build report: `artifacts/sed_soundscape_packaging_audit/20260528T1424Z_soft1279init_native_allcls_package/sidecar_build_report.json`
- Generic helper added: `scripts/birdclef_single_sed_package_sidecar_audit.py`
- Log: `logs/soft1279init_native_allcls_package_audit_20260528T1424Z.log`
