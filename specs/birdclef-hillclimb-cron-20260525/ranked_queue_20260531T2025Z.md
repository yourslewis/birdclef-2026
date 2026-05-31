# BirdCLEF hill-climb ranked queue — 2026-05-31 20:25 UTC

## Live status
- Public best now: `0.950`, tied by `v644` Yaroslav 0950 replay source and `v647` Ryuto EoS8 sidecar source.
- Latest score readout: `v652` EoS8 PowerOpt proto040/sed060 scored `0.948`; `v651` proto020/sed080 scored `0.941`. Both underperformed despite positive local source-winner proxy deltas.
- UTC slots used today: `2/5`; ~3.6h to reset at status check. Remaining slots: `3`.
- No active BirdCLEF local/trainer jobs after this run; trainer GPU1 was used for the DyMN10 sequence data point.

## Actions this run
1. Read Kaggle status via Bearer/SDK report and updated v651/v652 score interpretation.
2. Trained `soundscape-sequence-dymn10-nonaves-notrain-r2-filectx-filemil-losite-ep22-20260531` on official train_soundscapes.
3. Ran 72→234 scoped sidecar audit for the new DyMN10 branch.
4. Updated canonical performance table and JSONL plus ledgers.

## Ranked queue after this run

1. **Late-day slot fill if `<3h` to reset and no verifier-grade candidate appears.**
   - Slots remaining: `3`; do not submit before the late window unless a candidate is hidden-safe and verifier-grade.
   - Eligible source fills must pass code rerun, schema, finite/nonconstant, nonduplicate description/hash, and source-clean checks.
   - Previously rejected malformed v645/v646/v648/v649/v650 remain ineligible unless repaired and reverified.

2. **Train-soundscape sequence/file/site mining — file-MIL focused DyMN10/PANNs follow-up.**
   - New DyMN10 filectx/fileMIL point improved DyMN10 72-label context (`+0.040447` row, `+0.113577` file-MIL) and beat PANNs row-only on file-MIL by `+0.054548`, but lost row-wise and sidecar was below v616.
   - Next useful no-slot work: diagnose why file-MIL gains fail in 72→234 sidecar; consider file-level calibration/mapping rather than more raw rank sidecars.

3. **Source-winner family audit, not SED-heavy fork continuation.**
   - v651/v652 show SED-heavy xSED local proxy overfit. Demote proto-low/sed-high variants.
   - If exploring v950 source further, prefer orthogonal diagnostics around the original proto060/sed040 frontier, public-source dedupe, or hidden-safe reproduction stability—not another SED-heavy neighbor.

4. **Deeper soundscape-native / no-call branch with new evidence only.**
   - Continue only if new supervision appears (hand/teacher-audited multi-site no-call negatives or source-winner confidence features). Threshold-only farneg variants are exhausted.

5. **20s temporal/localmax / G124 target-shape ablations.**
   - Keep as diagnostics; not current slot candidates.

## Critic / verifier notes
- The SED-heavy fork result is an important negative: local +0.0017 vs v616 became public `0.941`, so the v616/source-sed local proxy is not approval-grade.
- The new DyMN10 file-context branch is a real data point but not submission-grade: best sidecar lift vs v616 is `-0.002275`.
- No submission this run: not yet inside the configured late-fill window and no valid verifier-grade candidate is ready.
