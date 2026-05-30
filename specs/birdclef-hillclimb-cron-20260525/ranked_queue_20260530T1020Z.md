# Ranked Queue — 2026-05-30 10:20 UTC

## Live state

- Best public LB remains `0.949`; v616/v621-v623/v634 are still tied references to beat.
- Latest scored submissions remain v636 `0.944`, v637 `0.943`, v638 `0.939`, v639 `0.944`, v640 `0.945`.
- 2026-05-30 UTC slots used: `0/5`; early UTC day with ~13.7h to reset at live check.
- Active jobs before run: no local/trainer BirdCLEF jobs; trainer GPUs free.

## Run findings

1. **G124/V2S soft-anchor90 localmax — CONTINUE AS DIAGNOSTIC, NO SUBMIT.** Trained `g124-effv2s-public946-pseudo-pilot-20260530-v2sinit-softanchor90-localmax-ep6` on 792 train_soundscape teacher rows / 66 files / 9 sites / 234 labels. Soft-anchor keeps soft labels plus high-confidence anchors and avoids the hardconf sparse-mask failure.
2. **Training metric improved vs G124 controls.** Best val AUC `0.961641` / 67 valid and all-row AUC `0.965053` / 75 valid, versus prior soft localmax `0.960094` / `0.944720` and hardconf90 `0.622851` / `0.623120`.
3. **Promotion gate still fails.** Teacher-cache blend best is w`0.02` with AUC `0.997042`, lift vs teacher `+0.00002330`. Site bootstrap q05 is `-0.00016009` and p(lift>0) only `0.54`. This is too small/unstable for an early-day slot and is not a v616 sidecar audit.

## Compact performance table for this run

```text
experiment                      primary               secondary / delta                                      decision
g124 softanchor90 localmax      val AUC 0.961641/67   all-row 0.965053/75; corr 0.8569; blend lift +0.0000233; site boot q05 -0.000160; vs soft G124 val +0.001547  no submit
```

## Comparable top-5/context table

```text
model / eval                         metric                         decision
G124 softanchor90 localmax            val 0.961641; all-row 0.965053 diagnostic continue; no submit
G124 soft localmax 20260526           val 0.960094; all-row 0.944720 promising but microscopic v616 lift
G124 hardconf90 localmax              val 0.622851; all-row 0.623120 reject sparse target starvation
Soft1279 head-loaded stability grid   local proxy 0.995545; lift +0.002064 hold; failed strict gates
Soft1279 capped selector retry        site-CV 0.993558; lift +0.000077 reject/no submit
```

## Ranked next actions

1. **Soft1279 head-loaded movement diagnosis / class-site attribution** — highest info value; global w0.16 is still the only sizable local proxy lift, but low-cap selector failed.
2. **G124 soft-anchor v616-sidecar/package audit** — now reasonable only as a cheap verifier, because soft-anchor fixed training but teacher-cache lift is tiny.
3. **Curated multi-site no-call/background negatives** — farneg20 showed distance-only strict negatives collapse; need hand/semi-curated negatives before another suppression branch.
4. **Late-day source slot fill** — if still <3h to reset and no verifier-grade repo candidate exists, use guarded public-source candidates after schema/runtime/dedupe checks.
5. **Stop condition:** no more hard-confidence-only G124 or blind PANNs file-context variants.

## Submission decision

No submission. Early-day slots remain available; this candidate is comparison-grade only and lacks a v616 sidecar/package audit with robust promotion gates.
