# No-call suppression sidecar smoke — far-negative gate — 2026-05-29 06:30 UTC

## Summary
- Experiment: `soundscape-nocall-suppression-v616-agg-farneg10-smoke-20260529T0630Z`
- Branch family: no-call/background suppression sidecar verifier.
- Input gate: `artifacts/soundscape_nocall_gate/soundscape-nocall-gate-soft1279native-agg-farneg10-losite-20260529/nocall_gate_predictions.csv` from the far-negative (>10s) aggregate gate.
- Baseline: v616 tied final (`0.949` public LB), local proxy AUC `0.993481` / 42 valid classes.
- Candidate grid: compact smoke, final no-call probability, scopes `nonaves_notrain` and `all`, alphas 0.005/0.01/0.02/0.04, power 1.0, 20 bootstrap iterations.

## Best candidate
- Recipe: `nocall_final_nonaves_notrain_p1p0_a010`.
- Data/shape: 240 proxy rows, 234 columns; 212 rows matched to gate predictions; 72 scoped non-Aves/no-train columns suppressed.
- Local macro AUC: `0.993564` / 42 valid classes.
- Lift vs v616: `+0.000084`; lift vs anchor: `+0.003174`.
- Rank corr vs v616: `0.999990`; MAE `0.000389`.
- Top-5 row recall: `0.626316` vs v616 `0.636842` (regressed).
- Suppression mean/p90/max: `0.002502` / `0.008957` / `0.010000`.

## Verifier / decision
- Finite/nonconstant candidate matrix; `submit_approved=false`.
- Reject as slot candidate: lift vs v616 is only `+0.000084` (< +0.001 gate), top-5 recall regressed, and the far-negative gate still uses weak site-skewed negatives.
- Keep as comparison-grade clue: final non-Aves/no-train alpha 0.01 remains the only positive suppression pattern; all-class suppression was worse.

## Artifacts
- Audit summary: `artifacts/soundscape_nocall_suppression/soundscape-nocall-suppression-v616-agg-farneg10-smoke-20260529T0630Z/audit_summary.json`
- Candidate CSV: `artifacts/soundscape_nocall_suppression/soundscape-nocall-suppression-v616-agg-farneg10-smoke-20260529T0630Z/candidates/nocall_final_nonaves_notrain_p1p0_a010.csv`
- Audit JSON: `artifacts/soundscape_nocall_suppression/soundscape-nocall-suppression-v616-agg-farneg10-smoke-20260529T0630Z/audit/ensemble_strategy_audit.json`
