# BirdCLEF hill-climb ranked queue — 2026-05-31 04:35 UTC

## Live state
- Current public LB best from user account: `0.950`, tied by `v644` and `v647` from 2026-05-30 UTC.
- 2026-05-31 UTC slot use at start: `0/5`.
- Time to reset at check: ~`19.65h`.
- Trainer status: no BirdCLEF job running before this run; GPU1 free. One non-BirdCLEF HSTU job was resident on GPU0.

## This-run critic update
`v644` and `v647` are not independent directions. Source and output audit shows both use the same dominant EoS8 taxonomy-smoothed PowerOptimization anchor (`0.9695`) plus small YukiZ/Perch/ProtoSSM/ResidualSSM component (`0.0305`), and public dry-run sidecars are no-ops. Their public outputs differ in only 3 cells above `1e-9`, max abs delta `1e-6`.

## Ranked queue

1. **EoS8/PowerOptimization/taxonomy source exploitation — ACCEPTED / no duplicate slot yet**
   - Evidence: v644/v647 reached public LB `0.950`, +0.001 over v616.
   - Caveat: current audited public outputs are near-identical; v647 sidecar and v644 exp098 sidecar did not apply in public dry-run.
   - Next action: inspect/adapt sidecar branches into a repo-owned private verifier or search for nonduplicate PowerOptimization parameter variants; do not submit static/near-duplicate reruns.

2. **Train_soundscapes sequence/file/site mining — ACCEPTED / keep training data points**
   - Latest data point: PANNs r4 local + file-context/file-MIL row AUC `0.640758`, file-MIL `0.667273`; sidecar audit below v616 by `-0.002024`.
   - Best comparable sequence branch remains PANNs r2 filectx+fileMIL: row `0.644272`, file-MIL `0.678888`.
   - Next action: move from broad PANNs wrappers to class/site-aware calibration or co-occurrence/no-call protocols; avoid slot until v616 lift is positive and robust.

3. **Soundscape-native calibrated sidecar / per-class stable caps — NEEDS REVISION**
   - Evidence: Soft1279 head-loaded diagnostic produced local AUC `0.995545`, lift vs v616 `+0.002064`, but concentrated site/class movements failed anchor gate; stable caps lift only `+0.000148`.
   - Next action: build stricter class/site stability gate using v644/v647 anchor as new floor, not v616 only.

4. **No-call/background gate — COMPARISON-GRADE, not slot-ready**
   - Evidence: farneg5/farthest no-call AUC `0.965397`, suppression audit lift vs v616 `+0.000063`; farneg10 still best tiny lift `+0.000084` but weak-negative/site-limited.
   - Next action: hand-label or construct multi-site background negatives; current weak-negative protocol should not submit.

5. **G124/V2S pseudo-label student target-shape lane — DIAGNOSTIC only**
   - Evidence: soft-anchor90 localmax val `0.961641`, teacher-blend lift +`0.000023` vs teacher but no v616-positive sidecar.
   - Next action: only continue if using v644/v647 source anchor as teacher or for decorrelation analysis.

6. **Fresh public frontier source candidates — BLOCKED / rejected by preflight**
   - v645/v646/v648/v649/v650 dry-run preflight found malformed public `submission.csv` outputs with nonnumeric/nonfinite parse under existing validator and duplicate dry-run hash `0ee04c918f807616`; no submission.

## Slot decision
No submission made at 04:35 UTC because this is early UTC day and no verifier-grade nonduplicate candidate passed. Slots remain available for later-day use if a valid nonduplicate source variant or verifier-positive sidecar is produced.
