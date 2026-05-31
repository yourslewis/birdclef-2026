# BirdCLEF hill-climb ranked queue — 2026-05-31 22:40 UTC

## Live status
- Public best before this run remains `0.950`, tied by `v644` Yaroslav 0950 replay source and `v647` Ryuto EoS8 sidecar source.
- UTC slots used today after this run: `5/5`.
- New submissions this run are pending public scoring: `v653` ref `53232636`, `v655` ref `53232648`, `v654` ref `53232692`.
- Earlier same-day verifier scores: `v652` proto040/sed060 scored `0.948`; `v651` proto020/sed080 scored `0.941`.
- No active local/trainer BirdCLEF jobs were found at start; trainer GPU0 was occupied by non-BirdCLEF work and GPU1 was free.

## Actions this run
1. Rechecked Kaggle submissions via Bearer API: `2/5` slots used at start, `<2h` to UTC reset.
2. Refreshed the late-window queue: no newly verifier-grade trained branch existed; repaired/previously rejected malformed public EoS9-style fills remained ineligible.
3. Built and pushed three hidden-safe private source-fork verifiers from the v647 EoS8/PowerOptimization lineage:
   - `v653`: proto080/sed020
   - `v654`: proto070/sed030
   - `v655`: proto050/sed050
4. Ran a local xSED proxy audit for proto080/proto070/proto060/proto050.
5. Submitted all three after Kaggle public-session COMPLETE + schema/finite/nonconstant output preflight.
6. Updated canonical performance table/jsonl and per-model ledgers.

## Compact performance table for this run

```text
id    family                    local AUC   Δv616 local  site q05   file q05   public/ref/status
v653  EoS8 proto080/sed020      0.990031    -0.003450    -0.011526  -0.007532  ref 53232636 pending
v654  EoS8 proto070/sed030      0.991227    -0.002253    -0.008898  -0.006307  ref 53232692 pending
v655  EoS8 proto050/sed050      0.993290    -0.000191    -0.001668  -0.001458  ref 53232648 pending
```

Top comparable local xSED/source-winner rows:

```text
candidate          local AUC   Δv616 local  public LB/status
source_sed raw     0.995976    +0.002495    diagnostic only; not packaged
proto020/sed080    0.995210    +0.001729    v651 public 0.941
proto040/sed060    0.994267    +0.000787    v652 public 0.948
proto050/sed050    0.993290    -0.000191    v655 pending
v616 baseline      0.993481     0.000000    historical 0.949
```

## Ranked queue after this run

1. **Monitor v653/v654/v655 scores after Kaggle scoring completes.**
   - If any reach/tie `0.950`, keep the source lineage but treat local xSED proxy as weak: v651/v652 already showed local-positive can go public-negative.
   - If all underperform, demote EoS8 Proto/SED scalar frontier and stop spending slots on this line.

2. **Return to train-soundscape sequence/file/site mining.**
   - Best useful clue from today remains file-MIL improvements in DyMN10/PANNs targeted branches, not direct 72→234 sidecars.
   - Next action: file-level calibration/mapping from file-MIL gains rather than raw row-rank sidecars.

3. **Train-soundscape native deeper variant only with new supervision/gates.**
   - No-call farneg variants and shallow head/softinit variants are exhausted without multi-site verified negatives or stronger teacher targets.

4. **Public-source scout only for truly distinct, source-clean kernels.**
   - EoS9/fork candidates with malformed/nonfinite public outputs remain rejected.
   - Avoid duplicate descriptions/matrices and static/sample fallback outputs.

## Critic / verifier decision
- Submitting v653/v654/v655 is justified only by late-window slot policy and source-clean hidden-safe execution, not by local model-quality evidence.
- Local audit is unfavorable for proto-heavy variants; v655 is the least bad local proxy point, while v653/v654 mainly sample the hidden frontier around yesterday’s public `0.950` source winner.
- All three passed runtime/schema/finiteness/nonconstant checks before submission.
