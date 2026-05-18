# BirdCLEF EoS5 Port Plan — 2026-05-18

Status: waiting on guarded direct datapoint `v574`
Anchor: current public best remains **0.946** from the repo-owned public946 family and tied variants.

## Why this plan exists

`v574` submitted `nina2025/birdclef-2026-eos-5` as a **single guarded exception** because it is newer and materially different from the exhausted public946 sidecar lanes.  The submission is still pending at plan time, so this document prepares the repo-owned follow-up path without spending another slot blindly.

## Live submission context at plan time

- `v574`: pending, ref `52780102`, guarded direct Nina EoS5 replay.
- `v573`: `0.945`, raw-SED 20s local-window B0 sidecar; killed.
- `v572`: `0.946`, cw0.75 local-window B0 sidecar; tied but did not improve.
- `v571`: `0.946`, safe xSED rank-ratio repackage; tied but did not improve.
- `v570`: no-score RAM failure.
- `v568`: no-score hidden rerun failure.
- `v567`: `0.944`.
- `v566`: `0.946`.
- `v565`: `0.943`.
- `v563`: `0.946`.

## Source preflight summary

Pulled source: `artifacts/public_kernels_20260518/eos5/birdclef-2026-eos-5.py` (ignored artifact, not committed)

- Source size: `295478` chars / `6533` lines.
- Source SHA256: `601ff2cb291cd26f007a64cbf01468cdb5ad3a8ebf232ab74c49c77c24714e8f`.
- Kernel metadata: CPU, no internet, competition source `birdclef-2026`, public Perch/SED datasets, Perch model, and two public notebook sources.
- Current version used for `v574`: `9`.

Top-level solution block:

```python
solutions = {
 'type_add' : 'direct',
 'Models'   : [
  {'Model':'Model_2','subm':'subm_2.csv','weight':0.0327,'xSED':[         ],'LB':'0.928'},
  {'Model':'Model_5','subm':'subm_5.csv','weight':0.9673,'xSED':[0.60,0.40],'LB':'0.949'}
 ]
}
```

Key hidden-safety markers present in source:

- `if 'Model_2' in _ensemble_models`
- `if 'Model_5' in _ensemble_models`
- `Karnakbayev_PowerOptimization_LB0948`
- `test_soundscapes/*.ogg`
- `IS_DRY_RUN = len(test_paths) == 0`
- dry-run sample alignment only under `if IS_DRY_RUN`
- final `write_final_submission(..., "submission.csv")` verifier
- `row_id` uniqueness and finite `[0,1]` checks before final write

## What appears new versus v542/v571

The promising signal is not another tiny `0.60/0.40` Proto/SED rank-ratio tweak.  EoS5 wraps a stronger `Model_5` component described in source comments as an EoS.4/PowerOptimization path:

- comments reference `EoS.4 0.948` and a `0.949` component line;
- explicit scalar changes include prior/threshold and rank-aware postprocess controls such as `apply_prior`, `file_confidence_scale`, `rank_aware_scaling`, and `adaptive_delta_smooth`;
- the final top-level blend is mostly `Model_5` plus a small `Model_2` complement (`3.27%`).

This is structurally different from:

- `v571`, which only changed the final Proto/SED rank ratio;
- `v572`/`v573`, which blended small CNN/SED student sidecars into public946;
- BirdNET/CLAP/V2S/ConvNeXt low-weight sidecar lanes that have tied or dropped.

## Decision tree after `v574` scores

### If `v574 > 0.946`

1. Immediately stop broad direct-public submission.
2. Port the EoS5 structural recipe into a repo-owned kernel:
   - start from the hidden-safe public946 port infrastructure where possible;
   - preserve exact input sources and CPU/no-internet runtime;
   - reproduce `Model_5` first as a standalone repo-owned component;
   - then add the `0.0327/0.9673` top-level blend only after standalone verifier passes.
3. Run Kaggle private dry-run and verify `submission.csv` row/column schema.
4. Submit at most one repo-owned confirmation candidate.

### If `v574 = 0.946`

- Treat EoS5 as useful but not urgent.  Do not spend another same-family slot immediately.
- Extract only the structural deltas into a smaller repo-owned diagnostic if there is idle implementation time:
  - `lambda_prior`/prior calibration;
  - `rank_aware_scaling` power;
  - `adaptive_delta_smooth`;
  - small `Model_2` complement.
- Use a single repo-owned candidate only if the extracted delta is clearly distinct from the already-tied public946 variants.

### If `v574 < 0.946` or no-scores

- Stop EoS5 direct public route.
- Do not submit EoS3/EoS4/EoS5 siblings directly.
- Only continue if a minimal repo-owned patch can be isolated and verified without notebook bloat.

## Port hygiene requirements

A repo-owned EoS5 port must not be a blind 6.5k-line notebook dump unless there is no practical alternative.  Preferred path:

1. Identify the minimal functions/deltas relative to `kaggle-kernels/v542-afr1ste-updated-public946/script.py`.
2. Keep source diff reviewable: input discovery, model inference, final schema verifier.
3. Avoid committing pulled public artifacts, outputs, logs, notebooks, or generated caches.
4. Run `py_compile` plus a Kaggle private dry-run.
5. Verify final `submission.csv`:
   - exact sample/hidden column order;
   - one row per hidden test 5s window;
   - finite numeric probabilities in `[0,1]`;
   - no dry-run/sample-only fallback in submit mode.

## Next action

Wait for `v574` to score.  If it is still pending at the next loop, preserve at least one slot for a possible EoS5 follow-up and use implementation time for source minimization or new-model training diagnostics, not more direct public submissions.
