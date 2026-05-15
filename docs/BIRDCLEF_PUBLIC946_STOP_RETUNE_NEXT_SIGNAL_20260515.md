# BirdCLEF Public946 Stop-Retune Decision — 2026-05-15

Status: active after `v551` and `v558` both tied public LB `0.946`.

## Current public state

- Public best remains **0.946**.
- `v551` tiny CLAP sidecar 0.5% scored `0.946`.
- `v558` exact-base clipped gate retune alpha0.10/maxabs0.02 scored `0.946`.
- Earlier same-family/diversity probes also tied or dropped:
  - `v546` train-audio-head 5%: `0.946`.
  - `v547/v548/v549` CV9245 2%/0.5%/1%: all `0.946`.
  - `v543/v544` BirdNET 10%/5%: both `0.946`.
  - `v545` CLAP 5%: `0.944`.

## Interpretation

The public946 anchor is extremely robust, and low-displacement sidecars are not moving the displayed public score. The available local train-soundscape overlap gates are now mostly useful for **rejecting unsafe candidates**, not for selecting further leaderboard slots. Repeated public946 postprocess/sidecar micro-variants are likely to consume daily quota without improving display score.

## Stop rule

Do **not** submit more public946-only postprocess retunes or single-family low-weight brackets unless one of these changes:

1. A new public/private score breaks the 0.946 plateau.
2. A genuinely new prediction source appears with strong source-clean provenance and a clean low-displacement gate.
3. A trained model produces an OOF/test prediction artifact with competitive AUC and lower correlation to public946.

Specifically, do not submit `v554`, `v555`, `v556`, or `v557`; `v558` has already tested the safest clipped-retune formulation and tied.

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
