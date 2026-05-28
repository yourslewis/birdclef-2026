# Model Data Point — Soft1279-init head-loaded per-class sidecar selector diagnostic

Timestamp: 2026-05-28 18:21 UTC

## Summary

Ran a no-submit per-class capped sidecar selector on the stronger head-loaded soft1279-init raw sidecar, using v616 as the base and tiny class-specific rank-blend weights. This directly tested whether the global `w0.16` lift is concentrated in reusable classes or robust across held-out sites/files.

The all-row selector can overfit to a large lift, and file-CV is positive, but site-CV is not robust: one held-out site has a negative lift and only 1/6 held-out site groups improve. This keeps the candidate comparison-grade, not submission-grade in mid-day policy.

## Ledger

- **Eval family:** per-class sidecar selector / calibration diagnostic.
- **Base:** submitted v616 final proxy CSV.
- **Sidecar:** head-loaded `soft1279init_native_allcls_member_raw` sidecar from `20260528T1618Z` package audit.
- **Evaluation data:** 240 proxy rows x 234 classes with train-soundscape labels; 42 valid AUC classes.
- **Site-CV primary metric:** AUC `0.993761` / 42 valid, lift vs v616 `+0.000280`; leave-site lift min `-0.005000`, q05 `-0.003768`, p>0 `0.167`.
- **File-CV secondary metric:** AUC `0.995051` / 42 valid, lift vs v616 `+0.001571`; leave-file q05 `+0.000000`, p>0 `0.062`.
- **All-row diagnostic:** AUC `0.995949`, lift `+0.002468`; selector uses 13 classes, mean total weight `0.041905`, max `0.160000`.
- **Top selected classes:** `116570, chacha1, 22973, 555146, 47144, trsowl, 47158son17, 47158son10`.
- **Decision:** **hold/reject as submission-grade.** Useful diagnosis; not enough site robustness for an early/mid-day slot.

## Artifacts

- Site-CV JSON: `artifacts/per_class_sidecar_selector/20260528T1821Z_soft1279init_head_loaded/site_cv.json`
- File-CV JSON: `artifacts/per_class_sidecar_selector/20260528T1821Z_soft1279init_head_loaded/file_cv.json`
