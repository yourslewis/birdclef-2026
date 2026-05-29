# Ranked queue — 2026-05-29 06:30 UTC

## Live status
- Public LB best: `0.949`; v616 remains the tied baseline to beat; v634 also tied `0.949` from the latest late-fill source batch.
- Latest completed submissions: v631 `0.926`, v632 `0.940`, v633 `0.946`, v634 `0.949`, v635 `0.941`.
- 2026-05-29 UTC slots used: `0/5`; about `17.7h` to reset at status check, so early-day policy active.
- Active jobs before run: no local/trainer BirdCLEF jobs; trainer GPUs free.

## Current run result
- Implemented a stricter no-call/background weak-negative protocol in `scripts/birdclef_soundscape_nocall_gate.py`: unlabeled windows within `10s` of a labeled positive in the same file are excluded from no-call training/eval.
- Evaluated `soundscape-nocall-gate-soft1279native-agg-farneg10-losite-20260529` using aggregate confidence features from the soft1279-native trio.
  - Data: 792 full windows before filtering; 762 after protocol; 739 labeled any-call positives; 23 weak far-background negatives across S09/S18/S22.
  - Primary no-call gate AUC: `0.963346` / 3 valid sites; site mean/min/q05 `0.869894/0.703601/0.726574`.
  - Best raw confidence baseline remains stronger: soft1279enc max AUC `0.985645`; gate delta `-0.022298` vs that raw baseline.
- Ran compact suppression-sidecar smoke over v616 proxy using farneg10 gate.
  - Best: `nocall_final_nonaves_notrain_p1p0_a010`, local AUC `0.993564` / 42 valid, lift vs v616 `+0.000084`, lift vs anchor `+0.003174`, rank corr `0.999990`, top5 recall `0.626316` vs v616 `0.636842`.
  - `submit_approved=false`; no submission.

## Ranked next actions
1. **Hand/stricter no-call negative audit** — highest information value. Farneg10 improves the comparison metric vs all-unlabeled but uses only 23 weak negatives and remains site-skewed; verify or broaden negatives before any suppression slot. Expected LB potential: medium; evidence value: high.
2. **Soft1279 head-loaded sidecar stability/class diagnosis** — still the strongest current positive local sidecar (`w0.16` lift vs v616 `+0.002064`), but prior selectors/gates failed; diagnose per-class/site failures and candidate packaging before late-day exploratory slot. Expected LB potential: medium; submission grade: no.
3. **Train-soundscape sequence/file/site mining with a distinct encoder or stronger file objective** — PANNs localmax remains the best sequence/file clue; direct sidecars are below v616. Expected LB potential: medium-low; data-point value: high.
4. **Non-Aves/no-train specialist with strict leave-site/site-pair gates** — aligned with under-mined soundscapes; avoid direct OOF sidecars unless wrapper passes v616 audit. Expected LB potential: medium-low; data-point value: high.
5. **Late-day public/source slot fill** — only after `<3h` to reset if no verifier-grade candidate exists; source must pass schema/runtime/dedup guards and cap recount.

## Critic / verifier decision
- Critic: the farneg10 protocol is directionally cleaner than all-unlabeled, but it shrinks the negative set from 53 to 23 and makes S22/S18 validation very low-count. Treat the higher gate AUC as comparison-grade, not promotion evidence.
- Verifier: suppression smoke is finite/nonconstant but fails the `+0.001` lift-vs-v616 gate and regresses top-5 recall. No slot.

## Artifacts
- Gate ledger: `artifacts/model_data_point_ledger/20260529T0618Z_soundscape_nocall_gate_farneg10.md`
- Suppression ledger: `artifacts/model_data_point_ledger/20260529T0630Z_nocall_suppression_farneg10_smoke.md`
- Gate metrics: `artifacts/soundscape_nocall_gate/soundscape-nocall-gate-soft1279native-agg-farneg10-losite-20260529/metrics.json`
- Suppression audit: `artifacts/soundscape_nocall_suppression/soundscape-nocall-suppression-v616-agg-farneg10-smoke-20260529T0630Z/audit_summary.json`
