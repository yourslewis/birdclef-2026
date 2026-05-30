# Model Data Point — Soft1279 stable-winner capped selector

Timestamp: 2026-05-30 18:16 UTC

## Summary

Re-tested the head-loaded soft1279 native sidecar with a stricter “stable winner” selector: higher allowed class weights (`0.12`) but only classes with at least 5 positives, 20 negatives, and >=0.002 train lift can move. This was a no-submit diagnostic to see whether the concentrated global `w0.16` soft1279 signal can be converted into a robust class-capped recipe without using a slot.

Result: the selector still does not clear promotion gates. File-CV lift is positive (`+0.000651`) and top-3 recall improves (`0.942105` -> `0.947368`), but site-CV lift is only `+0.000148` with leave-site q05 `-0.000054`. The all-row recipe selects only four classes (`116570`, `chacha1`, `555146`, `undtin1`) and remains comparison-grade.

## Ledger

- **Branch family:** per-class sidecar calibration diagnostic / robust soft1279 class caps.
- **Evaluation data:** v616 proxy rows with train-soundscape labels; `190` matched rows / `42` valid AUC classes; 20 files / 6 labeled proxy sites.
- **Target scope:** 234 competition labels, evaluated on 42 valid local AUC classes.
- **Model/init:** no new model; rank-blend selector over head-loaded soft1279 native raw sidecar.
- **Validation split:** leave-site and leave-file CV.
- **Primary metric:** site-CV AUC `0.993629` / `42` valid, lift vs v616 `+0.000148`.
- **Secondary metrics:** file-CV AUC `0.994132`, lift `+0.000651`; all-row lift `+0.000986`; site q05 `-0.000054`; selected all-row classes `4`; top3 recall site-CV `0.942105`, file-CV `0.947368`.
- **Baseline/delta:** v616 local proxy AUC `0.993481`; site delta `+0.000148`; file delta `+0.000651`. Below prior global soft1279 w0.16 local lift `+0.002064` and below promotion threshold.
- **Export/runtime status:** selector JSONs written; no submission CSV; no external submission.
- **Decision:** **reject/no submission.** Useful diagnostic: stable caps recover some file-CV signal, but site robustness is too weak for an early/mid-day slot.

## Artifacts

- Selector directory: `artifacts/per_class_sidecar_selector/20260530T1816Z_soft1279_stable_winner_caps`
- Site-CV JSON: `artifacts/per_class_sidecar_selector/20260530T1816Z_soft1279_stable_winner_caps/head_strict_site.json`
- File-CV JSON: `artifacts/per_class_sidecar_selector/20260530T1816Z_soft1279_stable_winner_caps/head_strict_file.json`
