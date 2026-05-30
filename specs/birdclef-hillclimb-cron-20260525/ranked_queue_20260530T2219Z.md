# Ranked queue — 2026-05-30 22:19 UTC

## Live status

- Public LB best before pending submissions: `0.949` (v616/v621-v623/v634 lineage tied; v636-v640 scored `0.944/0.943/0.939/0.944/0.945`).
- 2026-05-30 UTC slots: started `0/5`; late-day policy active at ~1.68h to reset; ended estimated `5/5` after v641-v644/v647 source-code submissions.
- Active jobs: no local BirdCLEF jobs; trainer process scan clean; GPU1 idle, GPU0 had non-BirdCLEF memory only.

## Actions this run

1. Submitted v641 Nina EoS1 source, v642 Nina EoS4 source, v643 Raunak v7 source from existing fallback queue after guarded preflight.
2. Scouted score/date-run public kernels for two more fills; submitted v644 Yaroslav 0950 replay and v647 Ryuto EoS8 sidecar.
3. Rejected v645/v646/v648/v649/v650 candidates with malformed/nonfinite 243-row public `submission.csv` output.
4. Updated performance table artifacts and late-fill ledger.

## Compact performance table for this run

| Exp | Family | Metric | Delta vs v616 | Status |
|---|---|---:|---:|---|
| v641 | Nina EoS1 public source | pending public LB | pending vs 0.949 | submitted ref 53197129 |
| v642 | Nina EoS4 public source | pending public LB | pending vs 0.949 | submitted ref 53197131 |
| v643 | Raunak v7 public source | pending public LB | pending vs 0.949 | submitted ref 53197133 |
| v644 | Yaroslav 0950 replay public source | pending public LB | pending vs 0.949 | submitted ref 53197162 |
| v647 | Ryuto EoS8 sidecar public source | pending public LB | pending vs 0.949 | submitted ref 53197164 |


## Ranked next actions

1. **Monitor v641-v644/v647 scores and update table rows** — highest urgency because all five slots are pending.
2. **After UTC reset, resume model data-point lane** — hand/teacher-audited multi-site no-call negatives or explicit site-risk-constrained soft1279 recipe.
3. **Avoid malformed EoS9/Anthony public outputs unless repaired as hidden-safe source** — current public `submission.csv` output is invalid for direct submission.
4. **Do not spend more PANNs/B0 objective knobs without a new transfer hypothesis** — recent sequence and focal2 data points were locally negative vs v616 sidecar gates.

## Critic / verifier decision

- Submission decision: **approved and executed under late-day policy** for v641-v644/v647; all source-clean guards passed.
- Evidence level: exploratory/submission-executed; not verifier-grade local improvement.
- Blocker: scores pending; update canonical rows once Kaggle completes.
