# soundscape-nocall-gate-soft1279native-agg-losite-20260528 — no-call/background gate data point — 2026-05-28 20:23 UTC

## Role / family
- Family: trusted no-call/background gate over existing soundscape-native package predictions.
- Evidence level: comparison-grade only. The negative class is weak: unlabeled train_soundscape windows, not hand-verified empty/no-call audio.

## Data
- Feature rows: 792 full 5s windows from the 66 official train_soundscape files.
- Sites/files: 9 sites / 66 files.
- Target: binary any-call/no-call proxy, 739 labeled any-call windows vs 53 unlabeled/background windows.
- Weak-negative distribution: {'S03': 0, 'S08': 0, 'S09': 41, 'S13': 0, 'S15': 0, 'S18': 9, 'S19': 0, 'S22': 3, 'S23': 0}.

## Model / validation
- Model: balanced logistic over aggregate confidence summaries from 3 soft1279-native packages (no per-class probs).
- Split: leave-one-site OOF; only sites with both positive and weak-negative windows have valid per-site AUC.
- Artifact metrics: `artifacts/soundscape_nocall_gate/soundscape-nocall-gate-soft1279native-agg-losite-20260528/metrics.json`.
- Prediction artifact: `artifacts/soundscape_nocall_gate/soundscape-nocall-gate-soft1279native-agg-losite-20260528/nocall_gate_predictions.csv`.

## Metrics
- Primary: leave-site OOF any-call/no-call ROC-AUC `0.950469` over 792 rows / 3 valid sites.
- Site AUC mean/min/q05: `0.853572` / `0.700899` / `0.718957`.
- Mean OOF predicted any-call: positives `0.882076`, weak-background `0.299017`.
- Best simple raw confidence baseline: `soft1279enc_native_max_auc` = `0.977098`; delta `-0.026630`.

## Decision
- continue as comparison-grade no-call protocol; no submit before hand-verified negatives + suppression sidecar verifier.
- No Kaggle submission: no competition-format suppression sidecar was approved, weak negatives are not sufficient for leaderboard action, and current UTC timing is not yet late-day slot-fill.

## Critic / verifier notes
- This protocol is useful for measuring whether package confidence can detect sparse/background windows, but not enough to alter hidden-test predictions without a sidecar audit against v616 and a hand-checked/teacher-consensus negative set.
- Promotion would require a finite/nonconstant suppression sidecar, positive lift vs v616 under site/file gates, and no recall damage on labeled positives.
