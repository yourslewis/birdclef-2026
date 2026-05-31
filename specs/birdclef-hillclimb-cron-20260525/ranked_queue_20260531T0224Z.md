# Ranked queue — 2026-05-31 02:24 UTC

## Live status
- Best public LB: `0.950` (v644 Yaroslav 0950 replay source, v647 Ryuto EoS8 sidecar source). Old v616 `0.949` is now the comparison floor, not the best.
- Latest submissions: v647 `0.950`, v644 `0.950`, v643 `0.946`, v642 `0.948`, v641 `0.947`.
- UTC slots used today: `0/5`; reset in ~21.7h at status check.
- Active jobs: none local/trainer; trainer GPUs idle.

## Coordinator ranking

1. **Exploit/audit v644/v647 source winners** — highest expected LB potential and information value.
   - Evidence: both public source hidden reruns reached `0.950`, +0.001 over v616; likely contains the useful hidden-behavior clue for the next move.
   - Next action: pull/inspect source components and emitted public-session outputs if available; identify distinct sidecar components, row-rank movement, and packageable repo-owned branch artifacts.
   - Gate: no static public-output-only submissions; require hidden-safe source rerun, finite/nonconstant output, nonduplicate description/hash, and comparison to the new 0.950 score.

2. **Source-winner ensemble/stacking workbench** — high EV after source audit.
   - Evidence: public slots show source candidates can escape plateau; need local workbench to test v644/v647 compatibility with v616/anchor families.
   - Next action: reconstruct or fetch raw component outputs from v644/v647, dedupe vs v616/known public outputs, run site/file bootstrap and correlation movement reports.
   - Gate: compare against 0.950 public baseline, not just v616 local; no early-day submission unless verifier-grade.

3. **Hand/teacher-audited multi-site no-call negatives** — medium EV data lane, but current weak-negative protocols are capped.
   - This run's farneg5-farthest data point: no-call AUC `0.965397`, but only 14 weak negatives across S09/S18/S22; suppression lift only `+0.000063` vs v616.
   - Next action: stop threshold-only weak-negative variants; upgrade negative labels with manual/teacher agreement or a new source-winner confidence signal before another gate.
   - Gate: >3 valid negative sites or explicit human/teacher audit; suppression lift vs v616 >= +0.001 and positive site/file q05.

4. **Site-risk-constrained soft1279 class recipe** — medium/low EV, useful diagnostic but not early-slot-worthy.
   - Evidence: global w0.16 local lift `+0.002064` concentrated in S03/S22 with S18 regression; stable caps only `+0.000148` site-CV.
   - Next action: only revisit if combined with v644/v647 movement or explicit site-risk constraints; avoid more blind B0 objective knobs.

5. **Train_soundscape sequence/file/site mining refresh** — lower current EV unless source-winner features are added.
   - Evidence: PANNs/DyMN10/fused sequence variants produced useful data points but negative v616 sidecar lifts; latest fused r4 20s sidecar lift `-0.002366`.
   - Next action: do not keep blind PANNs/fused temporal variants; use source-winner features or new negative labels if training another sequence model.

## Submission decision
No submission this run. Early UTC day, 5 slots open, but no candidate reached verifier-grade vs the new 0.950 public best; the trained/evaluated no-call sidecar is comparison-grade and explicitly rejected.
