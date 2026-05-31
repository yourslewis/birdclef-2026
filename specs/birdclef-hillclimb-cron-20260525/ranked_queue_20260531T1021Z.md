# BirdCLEF hill-climb ranked queue — 2026-05-31 10:21 UTC

## Live state
- Kaggle Bearer API live check: current public LB best remains `0.950`, tied by v644 Yaroslav 0950 replay and v647 Ryuto EoS8 sidecar source from the previous UTC day.
- 2026-05-31 UTC slot use: `0/5`; ~13.7h to reset at run start.
- Active jobs: no local BirdCLEF processes; trainer GPU0 occupied by unrelated HSTU job, GPU1 available and used for this run.

## This-run result
Trained/evaluated `soundscape-sequence-panns-cnn14-notrain-rowonly-h384-losite-ep24-20260531`: a PANNs/Cnn14 28-label no-train-only row-only capacity/isolation data point.

- Data: `1,478` windows / `66` files / `9` sites / `28` no-train labels.
- Leave-site metrics: row AUC `0.590497`; file-MIL `0.640872`; no-train `0.590497`; non-Aves `0.590497`; 6 valid site folds.
- Delta vs old 28-label PANNs row-only: row `+0.016661`, file-MIL `+0.073734`.
- Delta vs 08:20 72-label PANNs row-only no-train slice: row `-0.009984`, file-MIL `-0.050284`, so the broader 72-label multitask target is better than no-train-only isolation.
- Sidecar audit: best 1% scoped sidecar local AUC `0.990402` / 42 valid; lift vs v616 `-0.003079`, lift vs anchor `+0.000011`; `submit_approved=false`.
- Per-class selector diagnostics: 72-label row-only sidecar has tiny site/file CV lift (`+0.000079`/`+0.000095`) but no robust held-group support; new 28-label h384 selector is weaker and file-risky (`file q05 -0.022727`).

## Ranked queue

1. **EoS8/PowerOptimization/taxonomy source exploitation — ACCEPTED / top priority**
   - v644/v647 are the current `0.950` public-best family and essentially one EoS8/PowerOptimization/taxonomy-smoothed anchor.
   - Next: build a repo-owned verifier/parameter audit around the PowerOptimization path or sidecar assets (`exp001/exp002b`) that actually runs on audio; avoid exact/near-duplicate early-day reruns.

2. **Class/site-constrained selector over best soundscape sidecars — NEEDS REVISION**
   - The 72-label PANNs row-only sidecar has small selector lift for `517063`, `555146`, `47144`, but robustness is insufficient and the new 28-label h384 selector is weaker.
   - Next: only continue if selector can be constrained by positive leave-file/site movement; otherwise retire PANNs row-only selector as comparison-only.

3. **Train_soundscapes targeted sequence/file/site mining — CONTINUE ONLY WITH NEW SIGNAL**
   - h384 no-train-only isolation underperformed the 72-label multitask row-only no-train slice, arguing against more no-train-only PANNs capacity tweaks.
   - Next model data point should use a genuinely different signal/encoder/objective, not another blind PANNs wrapper.

4. **No-call/background gate — COMPARISON-GRADE, not slot-ready**
   - Farneg weak-negative variants produce tiny sidecar lifts and remain site-limited.
   - Continue only with hand/teacher-audited multi-site negatives or source-winner confidence signals.

5. **G124/V2S pseudo-label student — DIAGNOSTIC only**
   - Soft-anchor fixed target starvation but no robust v616/v950 sidecar lift.
   - Revisit only if tied to v950 PowerOptimization anchor or a clearly decorrelated new package.

## Critic / verifier decision
- Critic: no-train-only capacity isolation was worth one measured data point after the 72-label row-only result, but the result closes this micro-lane; multitask non-Aves auxiliary labels are likely helpful.
- Verifier: training and sidecar artifacts are finite/nonconstant/aligned; sidecar and selectors fail promotion. No early-day submission.

## Slot decision
No submission: `0/5` slots remain because no nonduplicate v950-family or newly trained candidate is verifier-grade. Fill slots later in the UTC day only with highest-ranked valid exploratory/source-clean candidates if no stronger verifier candidate emerges.
