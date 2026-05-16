# BirdCLEF Public946 Stop-Retune Decision — 2026-05-15

Status: active and strengthened after `v560` scored `0.945` despite positive local/dry-run gates.

## Current public state

- Public best remains **0.946**.
- `v551` tiny CLAP sidecar 0.5% scored `0.946`.
- `v558` exact-base clipped gate retune alpha0.10/maxabs0.02 scored `0.946`.
- `v560` direct blended-teacher V2S student rank sidecar 3% scored `0.945`, despite a positive strict v542 dry-run gate (`+0.000081879` AUC on train-soundscape overlap).
- Earlier same-family/diversity probes also tied or dropped:
  - `v546` train-audio-head 5%: `0.946`.
  - `v547/v548/v549` CV9245 2%/0.5%/1%: all `0.946`.
  - `v543/v544` BirdNET 10%/5%: both `0.946`.
  - `v545` CLAP 5%: `0.944`.

## Interpretation

The public946 anchor is extremely robust, and low-displacement sidecars are not moving the displayed public score. The available local train-soundscape overlap gates are now mostly useful for **rejecting unsafe candidates**, not for selecting further leaderboard slots. `v560` is the clearest warning case: it had clean runtime/output validation and the strongest recent local sidecar gate, but dropped publicly to `0.945`. Repeated public946 postprocess/sidecar micro-variants are likely to consume daily quota without improving display score.

## Stop rule

Do **not** submit more public946-only postprocess retunes, tiny trained-student sidecars, or single-family low-weight brackets unless one of these changes:

1. A new public/private score breaks the 0.946 plateau.
2. A genuinely new prediction source appears with strong source-clean provenance and a clean low-displacement gate.
3. A trained model produces an OOF/test prediction artifact with competitive AUC and lower correlation to public946.

Specifically, do not submit `v554`, `v555`, `v556`, or `v557`; `v558` has already tested the safest clipped-retune formulation and tied. Do not submit additional V2S/public946 sidecars after `v560=0.945` unless the candidate is backed by a stronger out-of-sample/OOF gate rather than the 190-row train-soundscape overlap.

## Next actionable tracks

1. **New source mining / source-clean model family audit**
   - Search for public notebooks/datasets newer than the Nina/Afr1ste public946 stack.
   - Candidate must provide a distinct model family or features, not another tiny rank perturbation.

2. **Pseudo-label / noisy-student cache from public946**
   - Use `v541/v542` public946 outputs as teacher.
   - Train/evaluate a student with OOF artifacts before any Kaggle slot.
   - Prefer a true OOF/holdout gate over train-soundscape overlap.

3. **Training/infrastructure lane**
   - Resume real SED/student work only if it produces a source-clean inference artifact with runtime headroom.
   - Do not spend competition slots on training artifacts without a clear gate.

## Submission policy

Hold remaining UTC daily submissions until the next candidate has fresh evidence. If forced to pick from existing candidates, prefer waiting over submitting another tied-family retune.

## 2026-05-16 bootstrap gate addendum

For any future public946 sidecar candidate, the local gate should include grouped bootstrap lift stability, not only one aggregate train-soundscape AUC. The v559 V2S+B0 strict dry-run gate looked positive on mean AUC (`+0.000035240`), but 200 file-group bootstrap iterations had a negative 5th percentile lift (`-0.000071763`) and only `p_lift_gt_0=0.84`. Treat that as insufficient for a slot after `v560=0.945`. A future candidate should have materially larger mean lift and a positive grouped-bootstrap lower tail before packaging/submission.
