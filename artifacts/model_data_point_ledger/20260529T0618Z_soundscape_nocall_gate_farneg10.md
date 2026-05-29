# Soundscape no-call/background gate — far-negative protocol — 2026-05-29 06:18 UTC

## Summary
- Experiment: `soundscape-nocall-gate-soft1279native-agg-farneg10-losite-20260529`
- Branch family: no-call/background gate / stricter weak-negative protocol.
- Data: official train_soundscapes package predictions from three soft1279-native soundscape models; 792 full 5s windows before filtering, 762 rows after excluding unlabeled negatives within 10s of a labeled positive.
- Labels: binary any-call/no-call proxy; positives are 739 labeled target-call windows; weak background negatives used are 23 unlabeled windows across S09/S18/S22.
- Features/model: aggregate confidence summaries only (`include_class_probs=false`) from soft1279init, soft1279 encoder-only, and observed-positive native package NPZs; logistic regression, C=0.2, class_weight balanced.
- Validation: leave-one-site; 3 sites had both classes for AUC.

## Metrics
- Primary: leave-site OOF any-call/no-call ROC-AUC `0.963346` / 3 valid sites.
- Site AUC mean/min/q05: `0.869894` / `0.703601` / `0.726574`.
- Mean OOF predicted any-call: positives `0.894400`, background `0.257840`.
- Best raw confidence baseline: `soft1279enc_native_max_auc` = `0.985645`; gate delta `-0.022298`.

## Verifier / decision
- Evidence level: comparison-grade only; negatives are stricter than the prior all-unlabeled protocol but still not hand verified and remain site-skewed.
- Decision: keep as a measured no-call protocol point; do not submit directly. Use only through bounded suppression sidecar audits.

## Artifacts
- Metrics: `artifacts/soundscape_nocall_gate/soundscape-nocall-gate-soft1279native-agg-farneg10-losite-20260529/metrics.json`
- Predictions: `artifacts/soundscape_nocall_gate/soundscape-nocall-gate-soft1279native-agg-farneg10-losite-20260529/nocall_gate_predictions.csv`
- Config: `configs/birdclef/soundscape_nocall_gate_soft1279native_agg_farneg10_losite_20260529.json`
