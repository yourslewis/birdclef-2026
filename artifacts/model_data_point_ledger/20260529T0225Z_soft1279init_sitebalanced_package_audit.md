# Package Audit — Soft1279-init site-balanced native all-class sidecar

Timestamp: 2026-05-29 02:25 UTC

## Summary

Packaged the site-balanced soft1279-init native all-class model through the TorchScript soundscape inference path and audited low-weight rank blends against the v616 local proxy. The ablation failed: every non-control blend scored below the v616 local baseline, so the prior soft1279 local lift was not improved by site-balanced training.

## Ledger

- **Model family:** soundscape-native calibrated package audit / v616 sidecar.
- **Training data lineage:** official `train_soundscapes`, 1,478 windows / 66 files / 9 sites / 234 labels.
- **Inference/audit data:** 66 labeled train_soundscape files -> 792 inference rows; sidecar matched 240/240 v616 proxy rows.
- **Validation split:** v616 local proxy + 200 bootstrap iterations.
- **Primary metric:** best non-control `soft1279init_sitebalanced_native_allcls_w0p16` local macro AUC `0.993104` / 42 valid classes.
- **Secondary metrics:** lift vs v616 `-0.000377`; lift vs anchor `+0.002713`; rank corr vs v616 `0.984163`; MAE vs v616 `0.044431`.
- **Baseline/delta:** v616 local `0.993481`; vs prior head-loaded soft1279 w0.16 sidecar `-0.002441` AUC and `-0.002441` lift-vs-v616.
- **Export/runtime status:** TorchScript inference/audit OK; finite/nonconstant 240x234 sidecar; `submit_approved=false`.
- **Decision:** **reject slot candidate.** No submission: below v616 local and failed promotion gates.

## Artifacts

- Audit summary: `artifacts/sed_soundscape_packaging_audit/20260529T0225Z_soft1279init_sitebalanced_package/audit_summary.json`
- Full audit: `artifacts/sed_soundscape_packaging_audit/20260529T0225Z_soft1279init_sitebalanced_package/audit/ensemble_strategy_audit.json`
- Sidecar CSV: `artifacts/sed_soundscape_packaging_audit/20260529T0225Z_soft1279init_sitebalanced_package/sidecars/soft1279init_sitebalanced_native_allcls_sidecar_234_anchorfill.csv`
- Log: `logs/soft1279init_sitebalanced_package_audit_20260529T0225Z.log`
