# Model Data Point — Soft1279 capped per-class selector retry

Timestamp: 2026-05-30 04:18 UTC

## Summary

Re-tested the strongest current repo-owned clue, the head-loaded soft1279 native all-class sidecar, with a stricter capped per-class selector and a small multi-soft1279 variant selector. This was a no-submit verifier/diagnostic pass meant to see whether the previous global `w0.16` local lift (`+0.002064` vs v616) could be converted into a robust low-displacement class-capped recipe.

Result: tighter caps removed the unstable negative site movement, but also collapsed the useful signal. Site-CV lift is only `+0.000077` for the head-loaded-only selector and `+0.000051` for the multi-sidecar selector; file-CV lift is only `+0.000218` / `+0.000193`. The all-row diagnostic lift is only `+0.000348`, using four classes, with no top-3 recall gain. This is below the promotion threshold and not worth an early UTC slot.

## Ledger

- **Eval family:** per-class capped sidecar selector / soft1279 movement diagnostic.
- **Base:** submitted v616 final proxy CSV (`artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission.csv`).
- **Sidecars tested:**
  - Head-loaded raw: `soft1279init_native_allcls_member_raw.csv`.
  - Multi low-cap: head-loaded + observed-positive + encoder-only + calibration-none soft1279 variants.
- **Evaluation data:** 240 v616 proxy rows x 234 classes using train-soundscape labels; 42 valid AUC classes.
- **Selector constraints:** rank blend, weights `0/0.005/0.01/0.02/0.04`, max total weight `0.04`, min train positives `3`, min train negatives `10`, min lift `0.001`.
- **Primary metric:** head-loaded low-cap site-CV AUC `0.993558` / 42 valid, lift vs v616 `+0.000077`.
- **Secondary metrics:**
  - Head-loaded file-CV AUC `0.993699`, lift `+0.000218`.
  - Multi-soft1279 site-CV AUC `0.993532`, lift `+0.000051`.
  - Multi-soft1279 file-CV AUC `0.993673`, lift `+0.000193`.
  - All-row diagnostic lift `+0.000348`; selected classes `4` (`116570`, `chacha1`, `555146`, `undtin1`).
  - Top-3 recall unchanged at `0.942105`.
- **Baseline/delta:** v616 local proxy AUC `0.993481`; current best site-CV delta `+0.000077`. Compared with the prior uncapped per-class selector site-CV lift `+0.000280`, low-cap reduced lift by `-0.000203`; compared with global `w0.16` stability-grid lift `+0.002064`, low-cap reduced lift by `-0.001987`.
- **Export/runtime status:** no submission CSV generated; selector JSON artifacts complete; finite input matrices; no external submission.
- **Decision:** **reject/no submission.** Low-cap selectors are too close to v616, and multi-sidecar does not recover the lost lift. The original head-loaded signal is real locally but depends on heavier movement that fails robustness gates.

## Artifacts

- Selector directory: `artifacts/per_class_sidecar_selector/20260530T0418Z_soft1279_capped_retry/`
- Head site JSON: `artifacts/per_class_sidecar_selector/20260530T0418Z_soft1279_capped_retry/head_lowcap_site.json`
- Head file JSON: `artifacts/per_class_sidecar_selector/20260530T0418Z_soft1279_capped_retry/head_lowcap_file.json`
- Multi site JSON: `artifacts/per_class_sidecar_selector/20260530T0418Z_soft1279_capped_retry/multi_lowcap_site.json`
- Multi file JSON: `artifacts/per_class_sidecar_selector/20260530T0418Z_soft1279_capped_retry/multi_lowcap_file.json`
