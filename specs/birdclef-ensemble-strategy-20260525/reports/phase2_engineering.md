# Phase 2 Engineering Report — Ensemble Strategy Audit Workbench

Date: 2026-05-25  
Role: Phase 2 Experiment Engineer  
Decision: **ACCEPTED as no-submit audit tooling; no recipe clears submission gates.**

## Summary

Implemented a repo-owned BirdCLEF ensemble strategy/audit workbench and ran it on the Phase 1 coordinator manifest. The workbench validates prediction CSVs, exact-dedupes matrices, computes fixed class-wise percentile-rank recipes, evaluates local train-soundscape metrics where labels are available, compares every recipe against both the v616 anchor and the v616 tied baseline, runs site/file bootstrap plus leave-one-site/file summaries, and emits JSON plus candidate CSVs for inspection.

This is an audit workbench only. It does **not** submit to Kaggle and the manifest has `allow_submit_approval=false`.

## Changed files

- `scripts/birdclef_ensemble_strategy_audit.py`
  - New no-submit audit CLI.
  - Loads JSON manifest members/recipes.
  - Validates schema, row IDs, finite values, and nonconstant class columns.
  - Reorders aligned rows when needed, but hard-fails row/column set mismatch.
  - Exact-dedupes member and recipe prediction matrices by SHA-256 over aligned numeric matrices.
  - Builds fixed class-wise percentile-rank blends.
  - Computes macro AUC and top-k row recall on matched train-soundscape labels.
  - Compares every recipe vs `anchor_only` and `v616_baseline` using AUC lift, rank correlation, MAE, max displacement, per-class lift summaries, bootstrap, and leave-one-group checks.
  - Writes optional candidate CSVs.
- `configs/birdclef/ensemble_strategy_20260525.json`
  - Manifest with the requested anchor, baseline, branches, and five fixed recipes.
- `specs/birdclef-ensemble-strategy-20260525/reports/phase2_engineering.md`
  - This report.

Generated ignored artifacts:

- `artifacts/ensemble_strategy_20260525/ensemble_strategy_audit.json`
- `artifacts/ensemble_strategy_20260525/candidate_csvs/anchor_only.csv`
- `artifacts/ensemble_strategy_20260525/candidate_csvs/v616_baseline.csv`
- `artifacts/ensemble_strategy_20260525/candidate_csvs/sakur_restored.csv`
- `artifacts/ensemble_strategy_20260525/candidate_csvs/sed_only_capped.csv`
- `artifacts/ensemble_strategy_20260525/candidate_csvs/sed_jung_tighter.csv`

## Manifest contents

Members:

- Anchor: `anchor_v616_raw` → `artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission_anchor_raw.csv`
- Baseline: `v616_final` → `artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission.csv`
- Branch: `sed_raw` → `artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission_samejima_sed_raw.csv`
- Branch: `jung21_raw` → `artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission_jung21_raw.csv`
- Branch: `sakur_visual` → `artifacts/anchored_blend_audit/sidecar_grid_inputs_20260525T0200Z/sakur_visual.csv`

Recipes:

- `anchor_only`: `1.00 anchor_v616_raw` in class-wise rank space.
- `v616_baseline`: actual submitted v616 final output.
- `sakur_restored`: `0.90 anchor + 0.02 sakur_visual + 0.04 jung21_raw + 0.04 sed_raw`.
- `sed_only_capped`: `0.94 anchor + 0.06 sed_raw`.
- `sed_jung_tighter`: `0.94 anchor + 0.03 sed_raw + 0.03 jung21_raw`.

## Commands run

From `/Users/yourslewis/.openclaw/repos/birdclef-2026`:

```bash
PY=/Users/yourslewis/.openclaw/workspace-don/kaggle/playground-series-s6e3/.venv/bin/python

$PY -m py_compile scripts/birdclef_ensemble_strategy_audit.py
$PY -m json.tool configs/birdclef/ensemble_strategy_20260525.json >/dev/null

PYTHONPATH=scripts $PY scripts/birdclef_ensemble_strategy_audit.py \
  --manifest configs/birdclef/ensemble_strategy_20260525.json \
  --output-dir artifacts/ensemble_strategy_20260525 \
  --bootstrap-iters 200 \
  --emit-candidate-csvs

$PY -m py_compile \
  scripts/birdclef_ensemble_strategy_audit.py \
  scripts/birdclef_public946_multi_sidecar_weight_grid.py \
  scripts/birdclef_per_class_sidecar_selector.py
$PY -m json.tool configs/birdclef/ensemble_strategy_20260525.json >/dev/null
$PY -m json.tool artifacts/ensemble_strategy_20260525/ensemble_strategy_audit.json >/dev/null
```

Note: the manifest default is 1000 bootstrap iterations, but this Phase 2 run used `--bootstrap-iters 200` to keep the engineering audit bounded. The emitted JSON records `bootstrap_iters: 200`.

## Validation results

All loaded member CSVs validated:

- Shape: `240 x 235` for all members.
- Required `row_id` present.
- Row IDs aligned to anchor.
- Class columns aligned to anchor.
- All prediction cells finite.
- All 234 class columns nonconstant.
- Exact member dedupe: 5 loaded matrices, 5 unique matrices.
- Exact recipe dedupe: 5 recipe matrices, 5 unique matrices.

Labels:

- Used `/Volumes/ExternalSSD/data/workspace_don/kaggle_birdclef2026/data/train_soundscapes_labels.csv`.
- Matched rows: `190`.
- Valid local AUC classes: `42`.

## Key numeric result

Local metrics are still rejection/comparison evidence only. `sakur_restored` is best against the baseline, but the incremental lift over v616 is microscopic and all promotion gates remain failed.

| recipe | macro AUC | lift vs anchor | lift vs v616 baseline | top3 recall | submit approved |
|---|---:|---:|---:|---:|---|
| `anchor_only` | `0.9903905` | `+0.0000000` | `-0.0030902` | `0.4526` | false |
| `v616_baseline` | `0.9934807` | `+0.0030902` | `+0.0000000` | `0.4842` | false |
| `sakur_restored` | `0.9935362` | `+0.0031457` | `+0.0000556` | `0.4895` | false |
| `sed_only_capped` | `0.9932675` | `+0.0028770` | `-0.0002131` | `0.4789` | false |
| `sed_jung_tighter` | `0.9934097` | `+0.0030192` | `-0.0000710` | `0.4789` | false |

Selected group-stability checks vs anchor:

| recipe | site bootstrap q05 | site leave-one min | file bootstrap q05 | file leave-one q05 |
|---|---:|---:|---:|---:|
| `v616_baseline` | `+0.0017065` | `+0.0026750` | `+0.0008489` | `+0.0025107` |
| `sakur_restored` | `+0.0017675` | `+0.0027233` | `+0.0010081` | `+0.0025754` |
| `sed_only_capped` | `+0.0015324` | `+0.0024932` | `+0.0012047` | `+0.0023049` |
| `sed_jung_tighter` | `+0.0016230` | `+0.0026505` | `+0.0008087` | `+0.0024369` |

## Gate decision

`submit_approved=false` globally and for every recipe.

Reasons:

- The manifest intentionally has `allow_submit_approval=false`.
- Only `42` valid local AUC classes are available, below the configured `60` preferred threshold.
- No candidate reaches `+0.0060` lift vs anchor.
- No candidate reaches `+0.0010` lift vs the v616 tied baseline.
- Site bootstrap q05 remains below the configured `+0.0030` threshold.
- File bootstrap q05 remains below the configured `+0.0015` threshold.
- Leave-one-site min remains below the configured `+0.0030` threshold.
- `sakur_restored` is only `+0.0000556` local AUC over v616, and v616 already tied public LB.

## Conclusion

Phase 2 produced a reusable, repo-owned ensemble audit workbench and confirmed the coordinator’s caution: the fixed recipes are useful controls/baselines, not a submission candidate. The best immediate use of this script is as the standard no-slot audit gate for any future genuinely new branch before building a private verifier.
