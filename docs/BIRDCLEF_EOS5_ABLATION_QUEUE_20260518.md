# BirdCLEF EoS5 Ablation Queue — 2026-05-18

Status: prepared while `v575` repo-owned EoS5 confirmation is pending.
Current public best: **0.949** from `v574`.

## Live state

- `v574` scored **0.949** from the guarded direct Nina EoS5 source replay.
- `v575` is the repo-owned, source-equivalent confirmation and is pending.
- 2026-05-18 UTC has four visible submissions after `v575`, so likely one slot remains.
- Do not spend the last slot until either:
  - `v575` confirms near `0.949`, or
  - a clearly safer/non-EoS candidate appears with stronger evidence.

## Source facts from EoS5

Pulled source: `artifacts/public_kernels_20260518/eos5/birdclef-2026-eos-5.py`.

Top-level blend:

```python
solutions = {
 'type_add' : 'direct',
 'Models'   : [
  {'Model':'Model_2','subm':'subm_2.csv','weight':0.0327,'xSED':[         ],'LB':'0.928'},
  {'Model':'Model_5','subm':'subm_5.csv','weight':0.9673,'xSED':[0.60,0.40],'LB':'0.949'}
 ]
}
```

`Model_5` source comments describe the winning path as:

- EoS.4 `0.948` anchor;
- `lambda_prior=0.4 -> 0.5` at both train/threshold and test prior applications;
- rank-aware scaling `power=0.5 -> 0.6`;
- independent postprocess dials: `file_confidence_scale`, `rank_aware_scaling`, `adaptive_delta_smooth`.

Observed active `Model_5` postprocess values:

- `lambda_prior=0.5` in prior application;
- `file_confidence_scale(... top_k=2, power=0.4)`;
- `rank_aware_scaling(... power=0.6)`;
- `adaptive_delta_smooth(... base_alpha=0.20)`;
- `xSED=[0.60, 0.40]` for the internal Proto/SED blend.

## Candidate order after `v575` scores

### v576 — Model5-only confirmation / remove weak complement

Hypothesis: the `0.949` lift is driven by `Model_5`, while the `3.27%` `Model_2` complement (`LB=0.928`) may be neutral or slightly harmful.

Change only:

```python
solutions = {
 'type_add' : 'direct',
 'Models'   : [
  {'Model':'Model_5','subm':'subm_5.csv','weight':1.0,'xSED':[0.60,0.40],'LB':'0.949'}
 ]
}
```

Guard:

- same repo-owned notebook and sources as `v575`;
- expected files can drop `subm_2.csv` because `Model_2` is not executed;
- require `submission.csv`, `subm_5.csv`, `subm_karnakbayev_power_optimization.csv`, `submission_protossm.csv`, `submission_sed.csv`;
- submit only if `v575` confirms the repo-owned path.

### v577 — Slightly lower rank-aware power (`0.55`)

Hypothesis: EoS5 pushed rank-aware power from `0.5` to `0.6`; if `0.6` improved EoS4 to `0.949`, midpoint `0.55` may retain lift while reducing over-suppression.

Change only:

```python
probs = rank_aware_scaling(probs, n_windows=N_WINDOWS, power=0.55)
```

Guard: only after `v576` or `v575` confirms; one scalar change only.

### v578 — Slightly higher prior (`lambda_prior=0.55`)

Hypothesis: comments say if lambda `0.5` improves over EoS4, bisect upward. Test a conservative midpoint rather than a jump to `0.6`.

Change both prior applications from `lambda_prior=0.5` to `0.55`.

Guard: no rank-power change in the same candidate.

### v579 — Model5-only plus tiny top-level complement grid is **not** first

Do not immediately sweep top-level blend weights. If `v576` drops, the `Model_2` complement is useful; if it ties/improves, eliminate it. Only then consider a single midpoint like `Model_2=0.015`, not a grid.

## Stop rules

- If `v575` fails/no-scores: diagnose repo-owned notebook drift first; do not run v576-v579.
- If `v575` scores below `0.949`: still diagnose before tuning, because the direct public source and repo-owned confirmation diverged.
- If `v575` confirms `0.949`, use at most one same-day slot for the most interpretable ablation (`v576` Model5-only).
- Avoid broad direct EoS public sibling submissions. Future EoS work must stay repo-owned.

## Execution update — v576 pushed and guarded monitor started

`v575` is still pending, so the last likely same-day slot is preserved. To avoid idle time, v576 was prepared and pushed but guarded behind the v575 confirmation condition.

v576 implementation:

- Repo directory: `kaggle-kernels/v576-eos5-model5-only/`
- Kernel URL: `https://www.kaggle.com/code/yourslewis/bc26-v576-eos5-model5-only-ablation`
- Kernel id: `119735856`, version `1`
- Change: top-level `solutions` contains only `Model_5` with weight `1.0`.
- Extra safety fix: final direct combiner now handles the one-model case by reading the generated `subm_5.csv` and writing `submission.csv` with `row_id` index.
- Submit guard: `scripts/submit_v576_when_ready.py` requires v576 COMPLETE/output verification and `v575` complete with `0.949+` before submitting.

This is queued as an execution-ready candidate, not yet a competition submission.
