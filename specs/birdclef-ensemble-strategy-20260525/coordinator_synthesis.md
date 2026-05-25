# Coordinator Synthesis — BirdCLEF Ensemble Strategy

Date: 2026-05-25

## Coordinator decision

Phase 1 is **ACCEPTED** as analysis. It does **not** approve a Kaggle submission.

The team converged on the same conclusion from four angles: the current `0.949` plateau is a saturated cluster, not a pile of independent top models. The useful near-term move is to build a stricter ensemble workbench/strategy that can ingest all good/different candidates, dedupe them, compare against both anchor and nearest tied recipe, and enforce group-robust gates before any future private verifier or slot.

## What we learned

- There are many `0.949` variants, but after exact dedupe and lineage grouping they collapse to a small number of branch families.
- Prediction analysis scanned `121` CSV artifacts: `58` aligned usable outputs, only `10` unique matrices after exact dedupe.
- The best current local recipe is rank-space `0.90 anchor + 0.02 Sakur visual + 0.04 Jung21 + 0.04 SED`, but it is only a microscopic local improvement over v616, which already tied public LB.
- Data proxy is narrow: `190` matched rows / `20` files / `6` sites / `42` valid classes. This is useful for rejection and comparison, not approval.
- Validation veto is strict: v616-like local lifts are no longer approval evidence.

## Immediate ensemble strategy

Build a reusable workbench around a dominant hidden-safe anchor plus small capped branch sidecars.

### Primary anchor/control
- Samejima/v616 visual anchor:
  - `artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission_anchor_raw.csv`

### Baseline tied recipe
- v616 final:
  - `artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission.csv`
- Purpose: every future candidate must beat this, not merely the anchor.

### Candidate branches to include now
- SED raw:
  - `artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission_samejima_sed_raw.csv`
- Jung21 raw:
  - `artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission_jung21_raw.csv`
- Sakur visual, as tiny stabilizer only:
  - `artifacts/anchored_blend_audit/sidecar_grid_inputs_20260525T0200Z/sakur_visual.csv`

### Candidate branches to hold
- HGNet raw sidecars: diagnostic only, prior hidden-safe final blends tied.
- S14: only if source/output can be rerun hidden-safely.
- Alexy NS1: high diversity but blocked/source inaccessible and direct score weak.
- G124/exportable SED students: future new-signal research, not current ensemble members.

### Rejected for Phase 2
- SYD52p / p949 clone increments.
- Per-class adaptive selector from v616 branches.
- Scalar/rank-power/temperature/taxon tweaks on tied branches.
- Additional EoS/PCEN/visual/HGNet clone submissions.

## Phase 2 implementation target

Create a repo-owned ensemble strategy workbench that:

1. Loads a candidate manifest of anchor, baseline tied recipe, and branch CSVs.
2. Validates row/column alignment, finiteness, nonconstant columns, and duplicate matrices.
3. Creates class-wise rank blends from fixed candidate recipes.
4. Computes metrics vs local train-soundscape truth where available.
5. Compares every candidate against both anchor and v616 baseline.
6. Runs site and file bootstrap / leave-one-group checks.
7. Emits a JSON readiness report and a candidate CSV only for no-slot inspection.
8. Marks `submit_approved=false` unless strict gates are met.

## Candidate recipes to audit first

1. `v616_baseline`: existing `0.92 anchor + 0.04 Jung21 + 0.04 SED` via actual v616 final.
2. `sakur_restored`: `0.90 anchor + 0.02 Sakur visual + 0.04 Jung21 + 0.04 SED`.
3. `sed_only_capped`: `0.94 anchor + 0.06 SED`.
4. `sed_jung_tighter`: `0.94 anchor + 0.03 SED + 0.03 Jung21`.
5. `anchor_only`: reconstruction/control.

## Promotion rules for now

No competition submission unless a future candidate clears all of:

- hidden-safe branch rerun/private verifier;
- aggregate local lift vs anchor `>= +0.0060` or stronger independent evidence;
- lift vs v616 baseline `>= +0.0010`;
- site bootstrap q05 `>= +0.0030`;
- file bootstrap q05 `>= +0.0015`;
- leave-one-site all positive with min `>= +0.0030`;
- leave-one-file at least 90% positive and q05 `>= +0.0010`;
- final rank corr/displacement within validation bounds;
- independent Verifier approval.

## Coordinator next action

Spawn Phase 2 Experiment Engineer to implement the workbench and run it on the five fixed recipes above. Then spawn Verifier to review implementation/results. No Kaggle submission in Phase 2.
