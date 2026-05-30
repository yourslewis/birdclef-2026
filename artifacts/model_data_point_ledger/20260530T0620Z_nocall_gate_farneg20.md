# Model Data Point — No-call gate far-negative 20s protocol

Timestamp: 2026-05-30 06:20 UTC

## Summary

Evaluated a stricter no-call/background protocol by keeping only unlabeled train_soundscape windows more than 20 seconds from any labeled positive in the same file, using aggregate soft1279-native package prediction features. This was intended to upgrade the weak negative protocol beyond the prior farneg10 run.

Result: the threshold is too strict for leave-site validation. It leaves only `13` weak background windows, all effectively concentrated on `S09`, so there are `0` valid leave-site AUC folds and no reliable no-call AUC. The raw-confidence baseline remains measurable (`soft1279enc_native_max_auc=0.981576` on the retained rows), but the trained OOF gate itself is invalid as a validation signal.

## Ledger

- **Branch family:** no-call/background weak-negative gate protocol.
- **Training/eval data:** soft1279-native package predictions over official train_soundscapes; `792` package rows, `752` retained rows after the >20s protocol; `739` positives / `13` weak negatives; 66 files / 9 sites.
- **Targets:** binary any-call/no-call target; unlabeled windows are weak negatives, not hand-verified no-call labels.
- **Model/init:** logistic regression over aggregate soft1279-native confidence features (`include_class_probs=false`, `C=0.2`).
- **Validation split:** leave-one-site.
- **Primary metric:** leave-site OOF any-call ROC-AUC `null`; valid sites `0`.
- **Secondary metrics:** no-call AUC `null`; best raw confidence baseline `soft1279enc_native_max_auc=0.981576`; mean OOF prediction on positives `0.917578`; mean background OOF prediction `null`.
- **Baseline/delta:** farneg20 invalidates group validation; compared with farneg10 (`0.963346` AUC / 3 valid sites), this protocol loses cross-site evaluability.
- **Export/runtime status:** metrics JSON and prediction CSV written; no suppression sidecar run because OOF no-call predictions are not validation-grade; no external submission.
- **Decision:** **reject/no submission.** Do not tighten weak negatives past 10–12s without adding hand-verified negatives from multiple sites; otherwise the gate collapses to an S09-only artifact.

## Artifacts

- Gate artifact: `artifacts/soundscape_nocall_gate/soundscape-nocall-gate-soft1279native-agg-farneg20-losite-20260530/`
- Config: `configs/birdclef/soundscape_nocall_gate_soft1279native_agg_farneg20_losite_20260530.json`
