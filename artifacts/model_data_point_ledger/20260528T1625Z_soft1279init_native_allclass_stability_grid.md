# Model Data Point — Soft1279-init native all-class stability/calibration grid

Timestamp: 2026-05-28 16:25 UTC

## Summary

Re-audited the previously packaged soft1279-head-loaded native all-class sidecar with an expanded rank-blend grid around `w0.015-0.16` and 150 bootstrap iterations. The proxy AUC kept increasing with weight; the best local recipe moved from `w0p08` to `w0p16`, but the recipe still did not clear the strict promotion gates, so no submission was made.

## Ledger

- **Model/eval family:** soundscape-native calibrated package audit / expanded stability grid.
- **Source model:** `soundscape-native-b0-soft1279init-losite-allcls-ep4-20260528`; all 234 labels, soft1279 checkpoint with head loaded.
- **Evaluation data:** v616 local proxy matrix, 240 rows x 234 classes, 42 valid labeled classes; official train-soundscape labels for proxy AUC/bootstrap.
- **Primary metric:** best recipe `soft1279init_native_allcls_w0p16` local macro AUC `0.995545` / 42 valid classes.
- **Secondary metrics:** lift vs v616 `+0.002064`; lift vs anchor `+0.005155`; rank corr vs v616 `0.982923`; MAE vs v616 `0.048867`. Against anchor, site bootstrap q05 `0.003139`, file bootstrap q05 `0.002484`, leave-one-site min `0.004835`. Against v616, site bootstrap q05 `0.000448`, file bootstrap q05 `0.000364`.
- **Gate result:** `submit_approved=false`; `w0p16` passed matched rows, valid classes, lift-vs-v616, site/file bootstrap, and leave-one gates, but failed the strict lift-vs-anchor gate (`0.005155` < `0.006`).
- **Decision:** **hold / no early-day submission.** This is the strongest local sidecar signal so far, but evidence remains comparison-grade and may be proxy/calibration-specific.

## Artifacts

- Manifest: `artifacts/sed_soundscape_packaging_audit/20260528T1618Z_soft1279init_native_allcls_stability_grid/manifest.json`
- Audit JSON: `artifacts/sed_soundscape_packaging_audit/20260528T1618Z_soft1279init_native_allcls_stability_grid/audit/ensemble_strategy_audit.json`
- Summary JSON: `artifacts/sed_soundscape_packaging_audit/20260528T1618Z_soft1279init_native_allcls_stability_grid/stability_grid_summary.json`
- Candidate CSVs: `artifacts/sed_soundscape_packaging_audit/20260528T1618Z_soft1279init_native_allcls_stability_grid/audit/candidate_csvs/`
