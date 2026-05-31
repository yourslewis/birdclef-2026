# BirdCLEF hill-climb ranked queue — 2026-05-31 08:20 UTC

## Live state
- Kaggle Bearer API live check: current public LB best remains `0.950`, tied by v644 Yaroslav 0950 replay and v647 Ryuto EoS8 sidecar source from the previous UTC day.
- 2026-05-31 UTC slot use: `0/5`; ~15.7h to reset at run start.
- Active jobs: no local BirdCLEF processes; trainer GPU0 occupied by unrelated HSTU Python job, GPU1 available and used for this run.

## This-run result
Trained/evaluated `soundscape-sequence-panns-cnn14-nonaves-notrain-rowonly-losite-ep24-20260531`: PANNs/Cnn14 targeted 72-label non-Aves/no-train row-only model on official `train_soundscapes`.

- Data: `1,478` windows / `66` files / `9` sites / `72` labels.
- Leave-site metrics: row AUC `0.674485`; file-MIL `0.691156`; no-train `0.600481`; non-Aves `0.674485`; 6 valid site folds.
- Delta vs the 06:23 PANNs 72-label r2 filectx+fileMIL data point: row `+0.042893`, file-MIL `+0.001151`, no-train `+0.058851`.
- Sidecar audit: best 1% row-only scoped sidecar local AUC `0.990950` / 42 valid; lift vs v616 `-0.002530`, lift vs anchor `+0.000560`; `submit_approved=false`.

## Ranked queue

1. **EoS8/PowerOptimization/taxonomy source exploitation — ACCEPTED / no duplicate slot yet**
   - v644/v647 are the current `0.950` public-best family but near-identical; exploit only nonduplicate, source-clean variants or sidecars with real hidden-test applicability.
   - Next: inspect/adapt `exp001/exp002b` sidecars into a repo-owned verifier or derive a small parameter variant from the PowerOptimization family; do not submit a near-duplicate early-day.

2. **Site-risk-constrained soft1279/v950-anchor calibration — NEEDS REVISION**
   - Soft1279 global w0.16 beat v616 locally but gains were concentrated on S03/S22 and regressed S18; stable caps were too weak.
   - New v950 floor raises the bar: require positive site-risk constraints and no early-day slot without nonduplicate hidden-safe packaging.

3. **Train_soundscapes targeted sequence/file/site mining — CONTINUE ONLY WITH DIAGNOSTICS**
   - New row-only 72-label PANNs point is much stronger than the filectx/MIL variant locally, confirming context/MIL was hurting targeted signal.
   - However, sidecar remains below v616; next step should be class/site movement diagnostics and maybe calibrated class subsets, not another blind context wrapper.

4. **No-call/background gate — COMPARISON-GRADE, not slot-ready**
   - Farneg weak-negative variants produce tiny local sidecar lifts and remain site-limited.
   - Next: hand/teacher-audited multi-site negatives; stop threshold-only negative variants.

5. **G124/V2S pseudo-label student — DIAGNOSTIC only**
   - Soft-anchor fixed target starvation but no robust v616/v950 sidecar lift.
   - Continue only if tied to the v950 PowerOptimization anchor or a clearer decorrelation hypothesis.

## Critic / verifier decision
- Critic: row-only targeted PANNs was the right diagnostic after file-context/MIL underperformed; it should not be promoted because sidecar lift vs v616 is negative.
- Verifier: sidecar is finite/nonconstant and row/column aligned, but it is analysis-only OOF proxy, not a hidden-test package; no submission.

## Slot decision
No early-day submission: `0/5` slots remain because no nonduplicate v950-family or newly trained candidate is verifier-grade. Use slots later in the UTC day only for highest-ranked valid exploratory/source-clean candidates if no stronger verifier candidate emerges.
