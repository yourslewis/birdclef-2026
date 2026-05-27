# Ranked queue — 2026-05-27 18:18 UTC

## Live status
- Best public LB: `0.949`; latest tied submissions v621/v622/v623 plus v616/v617/v620.
- 2026-05-27 UTC slots used: `0/5`; ~5.7h to reset at run start.
- Current run trained/evaluated PANNs/Cnn14 all-class r2 file-context sequence model. Result: row AUC `0.642202`, file-MIL `0.652651`, sidecar lift vs v616 `-0.002314`. No submission.

## Compact model-performance comparison

| Rank | Experiment | Row AUC | File-MIL | Decision |
|---:|---|---:|---:|---|
| 1 | PANNs all-class r2 no-file | 0.647816 | 0.670723 | best row; package first |
| 2 | PANNs all-class r2 file-context | 0.642202 | 0.652651 | useful but weaker than no-file |
| 3 | DyMN10 all-class r2 no-file | 0.597633 | 0.635285 | backup context package |
| 4 | Fused DyMN10+PANNs all-class r2 | 0.596642 | 0.675982 | best file-MIL clue; sidecar still fails v616 |
| 5 | DyMN10 all-class r3 robust | 0.501812 | 0.532188 | reject unchanged |

## Ranked next queue
1. **True hidden-test package for PANNs all-class no-file signal** — still best all-class row AUC and better than file-context on file-MIL; package before more OOF-wrapper sweeps.
2. **No-call / acoustic-background protocol** — define robust no-call/background labels and suppression/calibration audit; highest remaining decorrelated behavior branch.
3. **Fused sequence package/audit only if it uses file-MIL signal differently** — fused has best file-MIL but loses row and v616 proxy; use as diagnostic sidecar, not direct OOF wrapper.
4. **Broader OOF negative/no-call SED student** — fallback distinct model data point if package/no-call protocol is blocked.
5. **Late-day public/source slot fill** — only inside `<3h` to reset and only after schema/runtime/dedup checks.

## Critic / verifier decision
- PANNs file-context is a reasonable data point but **not** a submission candidate: it loses to the PANNs no-file baseline on row and file-MIL, and sidecar lift vs v616 is negative.
- Stop direct OOF proxy sidecar sweeps unless integration changes; they continue to lift vs anchor but lose to v616.
- No verifier-grade submission is ready at 18:18 UTC; keep slots unused until late-fill window or a true hidden-test package passes guards.
