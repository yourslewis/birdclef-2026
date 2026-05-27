# Ranked queue — BirdCLEF hill-climb cron — 2026-05-27 22:25 UTC

## Live state
- Best known public LB: `0.949` (v616 and multiple public source fills tied before this run).
- 2026-05-27 UTC slots: `5/5`; reset in ~`1.6h`.
- Latest v626-v630 status: v627 complete `0.928`, v630 complete `0.917`, v626/v628/v629 still pending at recount.
- Active jobs: no BirdCLEF local/trainer jobs after training; trainer has an unrelated GPU0 LRM job, so this run used GPU1.

## Actions this run
1. **Slot monitor — COMPLETE**
   - Recounted Kaggle submissions via Bearer API. Slots already capped from the late-fill run; no additional submission possible.
2. **Model data point — COMPLETE / REJECT UNCHANGED**
   - Trained `soundscape-native-b0-losite-allcls-observedpos-ep4-20260527` on official train_soundscapes.
   - Purpose: test whether observed positive-rate class weighting fixes native B0 calibration/file-MIL behavior.
   - Result: row AUC `0.624340`, file-MIL `0.582914`, no-train `0.615515`, non-Aves `0.585486`.
   - Delta vs prior native B0 all-class: row `-0.011821`, file-MIL `-0.090842`; reject this weighting.
3. **Verifier — PASS FOR ARTIFACT / NO SUBMISSION**
   - Predictions finite/nonconstant `1410x234`; TS/ONNX export smoke passed.
   - No leaderboard action: slots capped and candidate weaker than nearest baselines.

## Updated ranked queue

1. **Monitor remaining v626/v628/v629 scores — TOP IMMEDIATE**
   - These are the only remaining live slot outcomes for the day. v627/v630 already scored below best.
2. **PANNs/Cnn14 all-class no-file hidden-test package — TOP repo-owned action**
   - Best row metric among train_soundscape sequence heads: `0.647816`; file-MIL `0.670723`; no-train `0.641399`; non-Aves `0.679851`.
   - Direct proxy sidecars failed vs v616, so next move must be true hidden-safe packaging/eval, not another low-weight OOF wrapper.
3. **Prior native B0 all-class pos-weight baseline — KEEP / potential calibration study**
   - Row `0.636161`; file-MIL `0.673756`; observed-positive ablation regressed strongly, so revert to uniform `pos_weight_sqrt` if continuing native B0.
4. **Fused DyMN10+PANNs file-MIL clue — REVISE**
   - Best file-MIL (`0.675982`) but weaker row (`0.596642`) and negative v616 sidecar lift.
5. **No-call/acoustic-background protocol — NEXT distinct branch after score monitor**
   - Still under-mined; should start with a trustworthy no-call/background target audit and leave-site validation before any package.
6. **Observed-positive native B0 all-class — REJECT UNCHANGED**
   - Useful negative: observed class weighting worsened row and file-MIL vs previous native B0.

## Top comparable model table

| Rank | Experiment | Row AUC | File-MIL | No-train | Non-Aves | Decision |
|---:|---|---:|---:|---:|---:|---|
| 1 | PANNs/Cnn14 all-class r2 no-file | 0.647816 | 0.670723 | 0.641399 | 0.679851 | package/eval next |
| 2 | PANNs/Cnn14 all-class r2 filectx | 0.642202 | 0.652651 | 0.667300 | 0.707500 | keep data point |
| 3 | Native B0 all-class pos-weight | 0.636161 | 0.673756 | 0.626084 | 0.618037 | keep; no direct package |
| 4 | Native B0 observed-positive | 0.624340 | 0.582914 | 0.615515 | 0.585486 | reject unchanged |
| 5 | DyMN10 all-class r2 no-file | 0.597633 | 0.635285 | 0.545890 | 0.614342 | superseded by PANNs |

## Next exact action
- Recheck pending v626/v628/v629. If none beat `0.949`, start PANNs all-class no-file hidden-test package/eval or the no-call/acoustic-background target audit after UTC reset.
