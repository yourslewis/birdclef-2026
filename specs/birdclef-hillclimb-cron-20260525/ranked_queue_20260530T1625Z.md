# Ranked queue — 2026-05-30 16:25 UTC

## Live status

- Public LB best verified by Bearer API: `0.949` (v616/v621/v622/v623/v634 tied lineage; latest v636-v640 scored `0.944/0.943/0.939/0.944/0.945`).
- 2026-05-30 UTC slots used: `0/5`; time to reset at run start: ~7.7h.
- Active jobs: no local/trainer BirdCLEF jobs; trainer GPUs idle (both 4090s showed 0 MiB used by ML jobs).
- Slot policy: mid/early UTC-day, no comparison-grade-only slot fill. No candidate cleared verifier/submission-grade gates this run.

## Evaluated data points this run

Evaluated `soundscape-nocall-gate-soft1279native-agg-farneg5-balanced-lowconf-losite-20260530` and its v616 suppression sidecar audit.

- Branch: no-call/background weak-negative gate over soft1279-native package predictions, with `>5s` distance guard and per-site cap using lowest-confidence weak negatives.
- Data: 753 retained train_soundscape package rows / 66 files / 9 sites; 739 labeled any-call positives / 14 weak background negatives. Selected negatives by site: S09=6, S18=6, S22=2.
- Gate metric: leave-site any-call/no-call AUC `0.964624` / 3 valid negative-bearing sites; site mean/min/q05 `0.939902/0.833333/0.848637`.
- Baseline: raw `soft1279enc_native_max_auc=0.986468`; gate trails by `-0.021844`. Compared with farneg10, gate AUC is `+0.001278`, but this is still weak-negative comparison-grade.
- Suppression audit best: `nocall_final_nonaves_notrain_p1p0_a020` local AUC `0.993510` / 42 valid; lift vs v616 `+0.000029`; lift vs anchor `+0.003120`; rank corr vs v616 `0.999978`; top5 recall `0.631579` vs v616 `0.636842`.
- Decision: reject/no submission. Cleaner than farneg20 and marginally better gate AUC than farneg10, but suppression lift is smaller than farneg10 and far below promotion gates.

## Comparable no-call/background top comparison

1. Raw confidence baseline on balanced farneg5 rows: `0.986468` AUC (`soft1279enc_native_max`), no sidecar.
2. Balanced farneg5 gate: `0.964624` AUC / 3 valid sites; sidecar lift `+0.000029` vs v616.
3. Farneg10 gate: `0.963346` AUC / 3 valid sites; sidecar lift `+0.000084` vs v616.
4. Original aggregate all-unlabeled gate: `0.950469` AUC / 3 valid sites; sidecar lift `+0.000066` vs v616.
5. Farneg20 gate: invalid (`0` valid sites; 13 negatives S09-only), no sidecar.

## Ranked next actions

1. **Hand/teacher-audited multi-site no-call negatives** — Expected LB potential: medium; information value: high. Distance/low-confidence curation still has only S09/S18/S22 weak negatives; next upgrade needs actual multi-site negative evidence or a teacher-assisted manual audit, not another threshold.
2. **Robust soft1279 class/site caps from stable winners only** — Expected LB potential: low-medium; information value: high. Use movement diagnosis to avoid S18 regressors and over-concentrated global weight; only classes with repeatable file/site lift should move.
3. **New encoder/objective train_soundscape data point** — Expected LB potential: low-medium; information value: medium. Avoid duplicate PANNs/fused file-context variants; use a genuinely different acoustic encoder or loss if training more sequence/file/site models.
4. **Late-day guarded source fill** — Expected LB potential: low; information value: low-medium. Activate only under `<3h` to reset if no verifier-grade candidate exists and source/runtime/schema guards pass.
5. **G124/teacher-cache lane** — Expected LB potential: low; information value: low. Soft-anchor trains well but v616 transfer failed; only revisit with a new transfer hypothesis.

## Critic / verifier decision

- Critic: balanced farneg5 improves the protocol shape but not the strategic flaw: weak negatives are too sparse and not hand-verified, and raw confidence still dominates the trained gate.
- Verifier: candidate matrices are finite/nonconstant and aligned; `submit_approved=false`; best lift vs v616 is only `+0.000029` and top5 recall regresses.
- Submission decision: no submission this run.
