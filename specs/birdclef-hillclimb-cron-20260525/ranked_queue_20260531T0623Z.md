# BirdCLEF hill-climb ranked queue — 2026-05-31 06:23 UTC

## Live state
- Public LB best remains `0.950`, tied by v644 Yaroslav 0950 replay and v647 Ryuto EoS8 sidecar source from the previous UTC day.
- 2026-05-31 UTC slot use: `0/5`; ~17.7h to reset at run start.
- Trainer: no BirdCLEF job was active; GPU1 used for this run while a non-BirdCLEF HSTU job occupied GPU0.

## This-run result
Trained/evaluated `soundscape-sequence-panns-cnn14-nonaves-notrain-r2-filectx-filemil-losite-ep22-20260531`: PANNs/Cnn14 targeted non-Aves/no-train sequence/file/site model with radius-2 file context and file-MIL.

- Data: 1,478 windows / 66 files / 9 sites / 72 labels.
- Context row AUC `0.631592`; file-MIL `0.690005`; no-train `0.541630`; non-Aves `0.631592`.
- Own row-only baseline was stronger: row `0.669160`, file-MIL `0.720051`.
- Sidecar audit best local AUC `0.990695` / 42 valid, lift vs v616 `-0.002786`; `submit_approved=false`.

## Ranked queue

1. **EoS8/PowerOptimization/taxonomy source exploitation — ACCEPTED / no duplicate slot yet**
   - v644/v647 are the current `0.950` public-best family but near-identical; exploit parameters/sidecars only if nonduplicate and verifier-clean.
   - Next: repo-owned verifier for source-sidecar applicability or parameter variant dry-run; do not submit a near-duplicate early-day.

2. **Site-risk-constrained soft1279/v950-anchor calibration — NEEDS REVISION**
   - Soft1279 global w0.16 beat v616 locally but concentrated gains on S03/S22 and regressed S18; stable caps were too weak.
   - Next: use v644/v647 PowerOptimization family as the new floor and require positive site-risk constraints.

3. **Train_soundscapes targeted sequence/file/site mining — CONTINUE SELECTIVELY**
   - New PANNs 72-label targeted filectx/fileMIL point improved over old DyMN10 72-label context but failed own row-only and v616 sidecar gates.
   - Next: class/site movement diagnostics or co-occurrence/calibration; avoid more broad PANNs wrappers without a new hypothesis.

4. **No-call/background gate — COMPARISON-GRADE, not slot-ready**
   - farneg weak-negative variants remain tiny local sidecar lifts and site-limited.
   - Next: hand/teacher-audited multi-site negatives; no threshold-only variants.

5. **G124/V2S pseudo-label student — DIAGNOSTIC only**
   - Soft-anchor fixed target starvation but has no robust v616/v950 sidecar lift.
   - Continue only if tied to v950 anchor or a clearer decorrelation hypothesis.

## Slot decision
No early-day submission: the trained candidate is below v616 locally and no new nonduplicate v950-family candidate is verifier-grade. Slots remain available for late-day valid exploration if nothing stronger appears.
