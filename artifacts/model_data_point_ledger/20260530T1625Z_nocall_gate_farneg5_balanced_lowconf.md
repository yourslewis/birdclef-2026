# Model Data Point — No-call gate farneg5 site-balanced low-confidence protocol

Timestamp: 2026-05-30 16:25 UTC

## Summary

Evaluated a site-balanced weak no-call/background gate after the farneg20 protocol collapsed to S09-only negatives. The new protocol keeps unlabeled train_soundscape windows more than 5 seconds from a labeled positive, then caps weak negatives per site using the lowest mean top-1 confidence across the three soft1279-native package members. This deliberately trades negative count for less S09 dominance.

Result: the logistic gate is comparison-grade and slightly higher than farneg10 on aggregate (`0.964624` vs `0.963346` no-call AUC) with three valid negative-bearing sites, but it remains below the raw confidence baseline (`0.986468`) and does not create a slot-worthy suppression sidecar. Best suppression recipe lifts v616 by only `+0.000029`, below promotion gates and below the prior farneg10 suppression lift (`+0.000084`).

## Ledger

- **Branch family:** no-call/background weak-negative gate protocol + suppression sidecar verifier.
- **Training/eval data:** soft1279-native package predictions over official train_soundscapes; `753` retained rows after protocol / `66` files / `9` sites; `739` labeled any-call positives / `14` weak background negatives.
- **Negative protocol:** distance guard `>5s`; capped negatives per site from lowest-confidence unlabeled windows: `{'S09': 6, 'S18': 6, 'S22': 2}`.
- **Targets:** binary any-call/no-call target; unlabeled windows are weak negatives, not hand-verified no-call labels.
- **Model/init:** logistic regression over aggregate soft1279-native confidence features (`include_class_probs=false`, `C=0.2`).
- **Validation split:** leave-one-site.
- **Primary metric:** leave-site OOF any-call/no-call ROC-AUC `0.964624` / `3` valid sites.
- **Secondary metrics:** site mean/min/q05 `0.939902` / `0.833333` / `0.848637`; best raw confidence baseline `soft1279enc_native_max_auc=0.986468`; delta vs best baseline `-0.021844`.
- **Suppression sidecar audit:** best `nocall_final_nonaves_notrain_p1p0_a020` local AUC `0.993510` / `42` valid; lift vs v616 `+0.000029`; lift vs anchor `+0.003120`; top5 recall `0.631579`; eligible `False`.
- **Baseline/delta:** vs farneg10 gate `+0.001278` no-call AUC, but vs farneg10 suppression `-0.000055` v616 lift; vs raw soft1279enc max baseline `-0.021844`.
- **Export/runtime status:** metrics JSON/prediction CSV/candidate CSVs written; finite/nonconstant suppression candidates; no external submission.
- **Decision:** **reject/no submission.** Keep as a cleaner comparison-grade no-call protocol, but it is still weak-negative/site-skewed and below promotion gates.

## Artifacts

- Gate metrics: `artifacts/soundscape_nocall_gate/soundscape-nocall-gate-soft1279native-agg-farneg5-balanced-lowconf-losite-20260530/metrics.json`
- Gate predictions: `artifacts/soundscape_nocall_gate/soundscape-nocall-gate-soft1279native-agg-farneg5-balanced-lowconf-losite-20260530/nocall_gate_predictions.csv`
- Suppression audit: `artifacts/soundscape_nocall_suppression/soundscape-nocall-suppression-v616-agg-farneg5-balanced-lowconf-20260530T1625Z/audit_summary.json`
- Best candidate CSV: `artifacts/soundscape_nocall_suppression/soundscape-nocall-suppression-v616-agg-farneg5-balanced-lowconf-20260530T1625Z/candidates/nocall_final_nonaves_notrain_p1p0_a020.csv`
