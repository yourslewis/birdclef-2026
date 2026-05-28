# Model Data Point — Soft1279-init observed-positive native all-class package/audit

Timestamp: 2026-05-28 18:21 UTC

## Summary

Packaged the observed-positive soft1279-init native all-class model through the TorchScript soundscape inference path and audited it as a bounded v616/anchor sidecar. The best recipe (`soft1279init_obspos_native_allcls_w0p16`) was positive versus v616 but much weaker than the prior head-loaded `w0.16` grid and failed promotion gates.

## Ledger

- **Model/eval family:** soundscape-native calibrated package audit, observed-positive weighting ablation.
- **Source model:** `soundscape-native-b0-soft1279init-obspos-losite-allcls-ep4-20260528`; all 234 labels, soft1279 checkpoint with head loaded, observed-sqrt class weights.
- **Evaluation data:** v616 local proxy matrix, 240 rows x 234 classes, 42 valid labeled classes; official train-soundscape labels for proxy AUC/bootstrap.
- **Primary metric:** best recipe `soft1279init_obspos_native_allcls_w0p16` local macro AUC `0.993906` / 42 valid classes.
- **Secondary metrics:** lift vs v616 `+0.000425`; lift vs anchor `+0.003516`; rank corr vs v616 `0.983392`; MAE vs v616 `0.046623`.
- **Baseline/delta:** vs v616 local proxy AUC `0.993481` / `+0.000425`; vs prior head-loaded `w0.16` AUC `0.995545` / `-0.001639`.
- **Export/runtime status:** TorchScript soundscape inference OK; finite/nonconstant sidecar `240x234`; audit script OK.
- **Gate result:** not eligible (`one or more promotion gates failed`); submit_approved false.
- **Decision:** **reject slot candidate.** Observed-positive weighting does not improve the current best calibrated sidecar.

## Artifacts

- Audit summary: `artifacts/sed_soundscape_packaging_audit/20260528T1819Z_soft1279init_obspos_native_allcls_package/audit_summary.json`
- Audit JSON: `artifacts/sed_soundscape_packaging_audit/20260528T1819Z_soft1279init_obspos_native_allcls_package/audit/ensemble_strategy_audit.json`
- Sidecar CSV: `artifacts/sed_soundscape_packaging_audit/20260528T1819Z_soft1279init_obspos_native_allcls_package/sidecars/soft1279init_obspos_native_allcls_sidecar_234_anchorfill.csv`
