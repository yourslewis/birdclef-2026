# Ranked Queue — BirdCLEF ClawTeam Hill-Climb — 2026-05-28 20:23 UTC

## Live state

- Public best: `0.949` (v616/v621-v623 tied best; v626-v630 below best at `0.899/0.928/0.940/0.946/0.917`).
- UTC slots used at run start: `0/5`; ~3.6h to reset.
- Active jobs before run: no local/trainer BirdCLEF jobs.
- Slot policy decision: mid/late UTC day, but still >3h to reset. No submission without a verifier-grade/high-info nonduplicate candidate.

## Result of this run

1. **New no-call/background model data point:** trained/evaluated `soundscape-nocall-gate-soft1279native-trio-losite-20260528` on 792 full train_soundscape windows (739 labeled any-call / 53 weak unlabeled background).
   - High-dimensional class-prob logistic OOF no-call AUC `0.530166` / 3 valid sites; site min `0.185185`.
   - Decision: reject. It overfit per-class/package idiosyncrasies and inverted on S18.
2. **Revised aggregate no-call/background gate:** trained/evaluated `soundscape-nocall-gate-soft1279native-agg-losite-20260528` using only confidence summaries from three soft1279-native package outputs.
   - Leave-site OOF any-call/no-call AUC `0.950469` over 792 rows / 3 valid sites; site mean `0.853572`, min `0.700899`, q05 `0.718957`.
   - Best raw confidence baseline was `soft1279enc_native_max_auc=0.977098`; aggregate gate trails by `-0.026630` but is more structured than the failed class-prob gate.
   - Decision: continue as comparison-grade no-call/background protocol; no submission until weak negatives are audited and a suppression sidecar passes v616/site/file gates.

## Compact performance table for this run

```text
experiment                                      primary AUC  valid sites  site min  baseline delta  decision
nocall soft1279native trio full-prob gate        0.530166     3            0.185185  -0.446932       reject/overfit
nocall soft1279native aggregate gate             0.950469     3            0.700899  -0.026630       continue protocol; no submit
```

## Top comparable no-call/background evidence

```text
candidate / evidence                         metric                         decision
soft1279enc raw max confidence               weak no-call AUC 0.977098       promising signal; not OOF-trained sidecar
aggregate soft1279-native no-call gate        OOF no-call AUC 0.950469       continue protocol
full class-prob no-call gate                  OOF no-call AUC 0.530166       reject/overfit
strict-neg0010 OOF-teacher aux                train-audio AUC 0.930294       reject unchanged vs soft1279 control
soft1279 OOF-teacher soft control             train-audio AUC 0.935542       packaged direct sidecar failed transfer
```

## Ranked next actions

1. **Build suppression-sidecar verifier for the aggregate no-call gate** — apply a conservative no-call cap to v616/proxy rows only where gate confidence is low; require no recall damage on labeled positives, finite/nonconstant 234-col output, and lift vs v616 under site/file gates. Evidence level remains comparison-grade until this exists.
2. **Hand/teacher-audit weak negative windows** — the 53 unlabeled windows are concentrated in S09/S18/S22 and are not guaranteed empty. Upgrade them with manual spot checks or multi-teacher agreement before any hidden-test use.
3. **Late-day slot decision on head-loaded soft1279-init `w0.16` if no stronger candidate appears** — still best current local sidecar (AUC `0.995545`, lift vs v616 `+0.002064`) but fails strict anchor/site robustness gates. Consider only inside <3h if slots would otherwise expire and verifier accepts exploratory risk.
4. **Public/source slot fill (<3h only)** — use source-clean, nonduplicate candidates with dry-run/schema/dedup guards if no repo-owned candidate clears.
5. **Site-specific failure diagnosis** — focus S09/S18/S22 no-call rows and classes/sites that broke soft1279 per-class selector (`116570`, `chacha1`, `22973`, `555146`, `47144`, `trsowl`, `47158son17`, `47158son10`).

## Critic / verifier decision

- Critic: the no-call idea is strategically right, but weak negatives are too sparse/site-skewed to justify direct leaderboard action. Full per-class gate is rejected; aggregate confidence gate is worth a suppression-sidecar audit.
- Verifier: artifacts are finite and row-aligned; no Kaggle submission was made because no candidate is submission-grade and reset is still >3h away.
