# 2026-05-31 02:24 UTC — farneg5 farthest no-call gate + suppression audit

## Context
- Live Kaggle Bearer check before training: best public LB is now `0.950` from v644 Yaroslav 0950 replay source and v647 Ryuto EoS8 sidecar source; current UTC day slots `0/5` with ~21.7h to reset.
- No active local/trainer BirdCLEF jobs; trainer GPUs idle.
- Early UTC-day slot policy: no submission unless verifier-grade. This run produced a comparison-grade no-call/background data point only.

## Model/data point
- Experiment: `soundscape-nocall-gate-soft1279native-agg-farneg5-balanced-farthest-losite-20260531`.
- Branch family: no-call/background gate protocol over existing soft1279 soundscape-native package predictions.
- Data: official `train_soundscapes` package predictions, 753 retained rows after protocol, 66 files, 9 sites; 739 labeled any-call positives and 14 weak unlabeled negatives.
- Negative protocol: keep unlabeled windows >5s from labeled positives, cap to 6 negatives/site by farthest-in-file distance. Selected negatives remain only on S09=6, S18=6, S22=2, so evidence is comparison-grade and site-limited.
- Model: logistic regression over aggregate confidence features from soft1279init-native, soft1279enc-native, and soft1279init-observed-positive packages; leave-one-site evaluation.

## Performance
- Gate primary metric: leave-site any-call/no-call AUC `0.965397` over `3` valid sites.
- Site AUCs: S09 `0.754386`, S18 `0.877778`, S22 `0.984277`; q05 `0.766725`.
- Best raw confidence baseline: `soft1279enc_native_max_auc` AUC `0.981345`; gate delta `-0.015948`.
- Suppression audit best recipe: `nocall_final_nonaves_notrain_p1p0_a010` with local macro AUC `0.993543` / `42` valid classes.
- Best suppression lift vs v616: `+0.000063`; lift vs anchor `+0.003153`; rank corr vs v616 `0.999992792`; submit approved `false`.

## Decision
Reject as submission-grade. The farthest weak-negative gate slightly improves gate AUC over low-confidence farneg5 (`0.965397` vs `0.964624`) but has worse site q05 (`0.766725` vs `0.848637`) and lower suppression lift than farneg10 (`+0.000063` vs `+0.000084`). It does not clear the +0.001 lift-vs-v616 gate or site/file robustness gates.

## Artifacts
- Gate metrics: `artifacts/soundscape_nocall_gate/soundscape-nocall-gate-soft1279native-agg-farneg5-balanced-farthest-losite-20260531/metrics.json`
- Gate predictions: `artifacts/soundscape_nocall_gate/soundscape-nocall-gate-soft1279native-agg-farneg5-balanced-farthest-losite-20260531/nocall_gate_predictions.csv`
- Suppression audit summary: `artifacts/soundscape_nocall_suppression/soundscape-nocall-suppression-v616-agg-farneg5-farthest-smoke-20260531T0228Z/audit_summary.json`
- Best candidate CSV: `artifacts/soundscape_nocall_suppression/soundscape-nocall-suppression-v616-agg-farneg5-farthest-smoke-20260531T0228Z/candidates/nocall_final_nonaves_notrain_p1p0_a010.csv`
