# 2026-05-29 04:35 UTC — Soft1279-init calibration-none package/audit

## Summary
- Packaged `soundscape-native-b0-soft1279init-calibnone-losite-allcls-ep4-20260529` through TorchScript soundscape inference and v616 proxy sidecar audit.
- Inference rows: 792 labeled train-soundscape windows; matched proxy rows: 240/240; class columns: 234.
- Sidecar finite: `True`; nonconstant columns: 234.

## Best non-control recipe
| Metric | Value |
|---|---:|
| Recipe | `soft1279init_calibnone_native_allcls_w0p12` |
| Local macro ROC-AUC | 0.992844 |
| Valid classes | 42 |
| Lift vs v616 | -0.000637 |
| Lift vs anchor | +0.002453 |
| Rank corr vs v616 | 0.991502 |
| MAE vs v616 | 0.032907 |
| Raw member AUC | 0.954153 |

## Comparison
- Best recipe was below the v616 local proxy by `-0.000637`.
- It was below the prior head-loaded soft1279 `w0.16` stability-grid best by `-0.002701` local AUC.
- `submit_approved=false`; promotion gates failed.

## Decision
Reject slot candidate and do not submit. This calibration-focused ablation reduced the useful head-loaded sidecar signal rather than stabilizing it.

Artifacts:
- Audit dir: `artifacts/sed_soundscape_packaging_audit/20260529T0425Z_soft1279init_calibnone_package/`
- Sidecar CSV: `artifacts/sed_soundscape_packaging_audit/20260529T0425Z_soft1279init_calibnone_package/sidecars/soft1279init_calibnone_native_allcls_sidecar_234_anchorfill.csv`
- Audit summary: `artifacts/sed_soundscape_packaging_audit/20260529T0425Z_soft1279init_calibnone_package/audit_summary.json`
