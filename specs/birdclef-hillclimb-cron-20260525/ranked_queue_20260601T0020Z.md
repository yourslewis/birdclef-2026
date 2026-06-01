# BirdCLEF hill-climb ranked queue — 2026-06-01 00:20 UTC

## Live status
- Public best is now confirmed at `0.950`, tied by `v644` Yaroslav 0950 replay source and `v647` Ryuto EoS8 sidecar source.
- Latest late-window frontier verifiers completed: `v653` proto080/sed020 scored `0.947`, `v654` proto070/sed030 scored `0.949`, and `v655` proto050/sed050 scored `0.949`.
- UTC daily slots after reset: `0/5` used, ~23.7h to reset.
- No active local BirdCLEF jobs; trainer GPU1 is free while GPU0 has unrelated load.

## Ranked queue for this run

1. **Train-soundscape fused DyMN10+PANNs non-Aves/no-train file-context/file-MIL data point — selected.**
   - Rationale: PANNs 72-label row-only has the best row AUC (`0.674485`) while DyMN10 72-label file-context/file-MIL has the strongest file-MIL clue (`0.745704`). A fused file-context/file-MIL run directly tests whether the two complementary signals combine at sequence/file level.
   - Evidence level: comparison-grade model data point; no slot unless the 72→234 proxy sidecar beats v616 and verifier gates.

2. **File-level calibration/mapping from file-MIL gains.**
   - Rationale: recent gains are file-MIL-heavy but raw row sidecars remain below v616. Needs a calibrated row/file mapping rather than another blind low-weight row-rank blend.
   - Status: next if fused file-context confirms a stronger file-MIL signal.

3. **Train-soundscape native deeper variant only with new supervision/gates.**
   - Rationale: soft1279/head-loaded variants are mostly exhausted; revisit only with stronger teacher targets, multi-site no-call negatives, or a new architecture/regularizer.

4. **Public source scout for distinct source-clean kernels.**
   - Rationale: current EoS8 scalar frontier did not exceed the `0.950` public best; avoid more scalar/source-family probes early in the UTC day.

## Critic / verifier pre-decision
- Early UTC day: do not spend a Kaggle slot on v653-v655 neighbors or malformed public fills.
- Training a bounded new data point is the highest-EV action because it uses no slots and addresses the under-mined `train_soundscapes` sequence/file/site lane.
- Candidate config: `configs/birdclef/soundscape_sequence_fused_dymn10_panns_nonaves_notrain_r2_filectx_filemil_losite_ep22_20260601.json`.

## Outcome update — 2026-06-01 00:40 UTC
- Trained selected fused DyMN10+PANNs non-Aves/no-train file-context/file-MIL model.
- Comparable metrics: context row AUC `0.652377`, file-MIL `0.722866`, no-train `0.567523`; same-run row-only row AUC `0.620622`, file-MIL `0.695848`.
- The fused context model improved fused row-only by `+0.036211` row AUC but did not beat PANNs row-only row AUC (`0.674485`) or DyMN10 filectx file-MIL (`0.745704`).
- 72→234 sidecar audit best local AUC `0.990914`, lift vs v616 `-0.002567`, `submit_approved=false`; no Kaggle submission.

## Next ranked actions
1. Build a file-level calibration/mapping diagnostic from the best row/file candidates (PANNs row-only, DyMN10 filectx, fused context) to test whether file-MIL signal can transfer without raw row-rank degradation.
2. If no calibration gain, pivot to a genuinely deeper/native soundscape variant with new supervision/gates, not another fused low-weight sidecar.
3. Keep public-source scouting passive until a source-clean candidate is materially distinct from the EoS8 scalar frontier.
