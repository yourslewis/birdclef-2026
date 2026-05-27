# Ranked queue — 2026-05-27 16:27 UTC

## Live status
- Best public LB: `0.949`; latest tied submissions v621/v622/v623, plus v616/v617/v620.
- 2026-05-27 UTC slots used: `0/5`; ~7.7h to reset at start of run.
- Current run trained/evaluated fused DyMN10+PANNs all-class context. Result: row AUC `0.596642`, file-MIL `0.675982`, sidecar lift vs v616 `-0.002266`. No submission.

## Ranked next queue
1. **True hidden-test package for best AudioSet sequence signal** — package PANNs all-class first; fused only as file-MIL sidecar if packaging/audit supports it. Rationale: PANNs row AUC remains best (`0.647816`), fused improves file-MIL slightly but fails v616 proxy.
2. **No-call / acoustic-background protocol** — build trusted background/no-call labels and calibrated suppressor; highest remaining decorrelated behavior branch.
3. **S08/S15 guarded AudioSet refinement** — only a narrow guard if it fixes the fused/PANNs regressions without hurting S03/S13/S19/S22/S23; no broad weight sweep.
4. **Broader OOF negative/no-call SED student** — useful fallback data point if no package/no-call protocol is ready.
5. **Late-day public/source slot fill** — only inside `<3h` to reset and only after schema/runtime/dedup checks; current time is not yet late-fill.

## Critic / verifier decision
- Do not submit fused direct sidecar: local proxy loses to v616 and OOF wrapper is not hidden-test safe.
- Stop direct OOF proxy sidecar sweeps unless integration changes; they consistently lift vs anchor but lose to v616.
