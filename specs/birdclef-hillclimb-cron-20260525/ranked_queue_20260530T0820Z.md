# Ranked Queue — 2026-05-30 08:20 UTC

## Live state

- Best public LB remains `0.949`; v616/v621-v623/v634 are still the tied references to beat.
- Latest scored submissions: v636 `0.944`, v637 `0.943`, v638 `0.939`, v639 `0.944`, v640 `0.945`.
- 2026-05-30 UTC slots used: `0/5`; early UTC day with ~15.7h to reset at live check.
- Active jobs before run: no local/trainer BirdCLEF jobs; trainer GPUs free.

## Run findings

1. **G124/V2S hard-confidence localmax target-shape ablation — REJECT.** Trained `g124-effv2s-public946-pseudo-pilot-20260530-v2sinit-hardconf90-localmax-ep6` on 792 train_soundscape teacher rows / 234 labels. Hard target mask was only `0.995%` of cells (`618` positive, `1,226` negative), and validation collapsed: best val AUC `0.622851` / 67 valid; all-row student AUC `0.623120` / 75 valid. Teacher cache AUC on same rows remained `0.995541`; student/teacher corr only `0.141234`.
2. **Teacher-cache blend audit — REJECT.** Best tiny student blend w0.005 scored `0.997018` / 75 valid, lift vs teacher `-0.000000443`; site bootstrap q05 `-0.00000704`, leave-site q05 `-0.00000176`, positive leave-site fraction `0.333`. Not package-worthy and not a slot candidate.
3. **Critic decision.** Hard-confidence-only G124 target shape is too sparse and underfits relative to the prior soft localmax G124 run (`0.960094` val / `0.944720` all-row). This closes the hard-confidence branch unless reframed as soft-anchor rather than hard-only.

## Compact performance table for this run

```text
experiment                         primary                 secondary / delta                       decision
g124 hardconf90 localmax ep6       val AUC 0.622851/67     all-row 0.623120/75; corr 0.141; vs soft G124 val -0.337243; teacher-blend lift -0.000000443  reject
```

## Ranked next actions

1. **Soft1279 head-loaded movement diagnosis / class-site attribution** — still highest information value. The global w0.16 sidecar is the only local artifact with meaningful lift vs v616, but low-cap selector retry failed; diagnose classes/sites and whether movement is proxy-only before any more soft1279 knobs.
2. **Hand-verified or semi-curated no-call/background negatives** — farneg20 proved distance-only filtering collapses coverage. Need multi-site negatives, not a stricter scalar distance.
3. **G124 soft-anchor target-shape ablation** — only if revisiting G124; hard-confidence-only is rejected. A soft-anchor mask could answer whether high-confidence anchors help without starving the BCE objective.
4. **Late-day source slot fill** — if still <3h to reset and no verifier-grade repo candidate exists, use guarded public-source candidates only after schema/runtime/dedupe checks.
5. **Stop condition:** avoid more blind PANNs file-context/file-MIL variants and hard-only G124 variants.

## Submission decision

No submission. Early-day slots remain available; no candidate passed verifier gates or exceeded v616 on a robust proxy.
