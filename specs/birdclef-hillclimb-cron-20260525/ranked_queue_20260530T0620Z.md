# Ranked Queue — 2026-05-30 06:20 UTC

## Live state

- Best public LB remains `0.949`; v616/v621-v623/v634 are still the tied references to beat.
- Latest scored submissions: v636 `0.944`, v637 `0.943`, v638 `0.939`, v639 `0.944`, v640 `0.945`.
- 2026-05-30 UTC slots used: `0/5`; early UTC day, so no low-confidence slot fill.
- Active jobs: no local/trainer BirdCLEF jobs; trainer GPUs free.

## Run findings

1. **PANNs all-class file-context no-MIL ablation — REJECT.** Row AUC `0.626315`, file-MIL `0.649487`; removing file-MIL loss is worse than the prior PANNs filectx+fileMIL run (`0.644272` row / `0.678888` file-MIL). Sidecar lift vs v616 `-0.002698`.
2. **No-call farneg20 protocol — REJECT.** The stricter `>20s` weak-negative rule leaves only 13 negatives and 0 valid leave-site AUC folds; it is S09-only and not promotion-grade.

## Ranked next actions

1. **Soft1279 head-loaded movement diagnosis / class-site attribution** — highest info value. The global w0.16 sidecar is still the only local artifact with meaningful lift vs v616, but class/site robustness failed; diagnose which classes/sites drive movement before more training.
2. **Hand-verified or semi-curated no-call/background negatives** — farneg20 proved distance-only filtering collapses coverage. Need multi-site negatives, not a stricter scalar distance.
3. **G124/V2S target-shape hard-confidence/power ablation** — lower priority but distinct from exhausted PANNs/soft1279 knobs; run only if it answers target-shape, not another clone.
4. **Late-day source slot fill** — if still <3h to reset and no verifier-grade repo candidate exists, use guarded public-source candidates only after schema/runtime/dedupe checks.
5. **Stop condition:** avoid more blind PANNs file-context/file-MIL variants; they are good landscape points but sidecar audits are consistently below v616.

## Submission decision

No submission. Early-day slots remain available; no candidate passed verifier gates or exceeded v616 on a robust proxy.
