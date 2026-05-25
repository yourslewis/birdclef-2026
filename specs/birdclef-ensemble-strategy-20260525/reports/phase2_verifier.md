# Phase 2 Verifier Report — Ensemble Strategy Audit Workbench

Date: 2026-05-25  
Role: Independent Verifier  
Decision: **ACCEPTED as no-submit audit tooling; no Kaggle submission approved**

## Executive decision

I accept the Phase 2 implementation and results for the intended scope: a local/no-submit ensemble audit workbench. The script validates the fixed manifest members, constructs the requested rank-blend recipes, compares each recipe against both the anchor and the v616 tied baseline, runs local AUC/top-k plus site/file group stability checks, and keeps `submit_approved=false`.

No recipe clears the promotion gates. The best local recipe, `sakur_restored`, improves only `+0.0000556` macro AUC over the v616 tied baseline, so it is not actionable after v616 itself tied public LB.

## Files reviewed

- `specs/birdclef-ensemble-strategy-20260525/spec.md`
- `specs/birdclef-ensemble-strategy-20260525/coordinator_synthesis.md`
- `specs/birdclef-ensemble-strategy-20260525/reports/validation_metrics.md`
- `specs/birdclef-ensemble-strategy-20260525/reports/phase2_engineering.md`
- `scripts/birdclef_ensemble_strategy_audit.py`
- `configs/birdclef/ensemble_strategy_20260525.json`
- `artifacts/ensemble_strategy_20260525/ensemble_strategy_audit.json`

## Implementation findings

### CSV validation

Accepted.

The audit script checks:

- `row_id` exists;
- duplicate `row_id` values fail;
- prediction columns exist;
- member column sets match the canonical anchor columns;
- member row-id sets match the canonical anchor rows;
- misordered but matching rows are reordered to anchor order;
- nonfinite cells fail;
- constant prediction columns fail.

The generated audit confirms all five manifest members are `240 x 235`, row-aligned, finite, and have all `234` class columns nonconstant.

### Rank blend construction

Accepted.

For `rank_blend` recipes, the script computes class-wise percentile ranks per member and requires weights to sum to `1.0`. The fixed recipes match the coordinator target:

- `anchor_only`: `1.00 anchor_v616_raw`
- `v616_baseline`: actual submitted v616 final member
- `sakur_restored`: `0.90 anchor + 0.02 sakur_visual + 0.04 jung21_raw + 0.04 sed_raw`
- `sed_only_capped`: `0.94 anchor + 0.06 sed_raw`
- `sed_jung_tighter`: `0.94 anchor + 0.03 sed_raw + 0.03 jung21_raw`

### AUC and top-k metrics

Accepted.

Local AUC uses matched train-soundscape labels, filters to classes with both positive and negative examples, and reports macro AUC. Top-k row recall uses the shared BirdCLEF helper over the matched rows. The reported validation context is narrow but correctly surfaced: `190` matched rows and `42` valid AUC classes.

### Bootstrap and leave-group checks

Accepted with caveat.

The script runs both site and file bootstrap plus leave-one-site/file comparisons. It computes these comparisons against both:

1. `anchor_only`, and
2. `v616_baseline`.

The Phase 2 artifact used `200` bootstrap iterations, which is smoke-grade and matches the engineering report. This is enough for the current no-submit audit, but not enough for any future submission-grade signoff; the validation spec prefers `>=5000` iterations.

### Comparison vs anchor and v616 baseline

Accepted.

Every recipe includes comparisons vs both configured controls. This is the critical fix over earlier sidecar gates: `sakur_restored` looks mildly positive vs anchor, but only microscopic vs the already-tied v616 baseline.

Key audited metrics:

| Recipe | Macro AUC | Lift vs anchor | Lift vs v616 baseline | Submit approved |
|---|---:|---:|---:|---|
| `anchor_only` | `0.9903905` | `+0.0000000` | `-0.0030902` | false/control |
| `v616_baseline` | `0.9934807` | `+0.0030902` | `+0.0000000` | false/control |
| `sakur_restored` | `0.9935362` | `+0.0031457` | `+0.0000556` | false |
| `sed_only_capped` | `0.9932675` | `+0.0028770` | `-0.0002131` | false |
| `sed_jung_tighter` | `0.9934097` | `+0.0030192` | `-0.0000710` | false |

### Gate logic and `submit_approved`

Accepted.

`submit_approved` is safely blocked by two layers:

- manifest: `allow_submit_approval=false`;
- script: `submit_approved = allow_submit_approval and all gates passed`.

I also reran the audit with `--allow-submit-approval`; readiness still reported `allow_submit_approval=false` and `submit_approved=false` because the manifest disables approval. In addition, no non-control recipe passes the gates anyway:

- valid classes are `42`, below the configured `60` threshold;
- lift vs anchor is below `+0.0060`;
- lift vs v616 baseline is below `+0.0010`;
- site bootstrap q05 is below `+0.0030`;
- file bootstrap q05 is below `+0.0015` for the best candidate;
- leave-one-site min is below `+0.0030`.

## Verification commands run

From `/Users/yourslewis/.openclaw/repos/birdclef-2026` using the Kaggle venv:

```bash
PY=/Users/yourslewis/.openclaw/workspace-don/kaggle/playground-series-s6e3/.venv/bin/python

$PY -m py_compile scripts/birdclef_ensemble_strategy_audit.py
$PY -m json.tool configs/birdclef/ensemble_strategy_20260525.json >/dev/null
$PY -m json.tool artifacts/ensemble_strategy_20260525/ensemble_strategy_audit.json >/dev/null

PYTHONPATH=scripts $PY scripts/birdclef_ensemble_strategy_audit.py \
  --manifest configs/birdclef/ensemble_strategy_20260525.json \
  --output-dir /tmp/birdclef_ensemble_strategy_verifier_49329 \
  --bootstrap-iters 5

$PY -m json.tool /tmp/birdclef_ensemble_strategy_verifier_49329/ensemble_strategy_audit.json >/dev/null

PYTHONPATH=scripts $PY scripts/birdclef_ensemble_strategy_audit.py \
  --manifest configs/birdclef/ensemble_strategy_20260525.json \
  --output-dir /tmp/birdclef_ensemble_strategy_verifier_allow_\$\$ \
  --bootstrap-iters 1 \
  --allow-submit-approval
```

Results:

- `py_compile` passed.
- Manifest JSON parsed.
- Existing audit JSON parsed.
- Minimal audit invocation completed successfully with labels loaded.
- Minimal audit output reported `submit_approved=false`, `approved_recipes=[]`, and best recipe vs baseline `sakur_restored`.
- `--allow-submit-approval` still produced `submit_approved=false` because the manifest has `allow_submit_approval=false` and recipes fail gates.

## No Kaggle side effects

Confirmed for this Phase 2 verification/implementation scope:

- `scripts/birdclef_ensemble_strategy_audit.py` contains no Kaggle API calls, no `kaggle` CLI invocation, no `requests`, no `subprocess`, no kernel push, and no competition submit path.
- My verification commands were local-only: Python compile, JSON parsing, local audit runs, and file inspection.
- No Kaggle submission was made by this verifier.
- No Kaggle kernel/private verifier was launched by this verifier.

## Candidate CSV artifact check

Generated candidate CSVs are local/no-submit artifacts:

- `artifacts/ensemble_strategy_20260525/candidate_csvs/anchor_only.csv`
- `artifacts/ensemble_strategy_20260525/candidate_csvs/v616_baseline.csv`
- `artifacts/ensemble_strategy_20260525/candidate_csvs/sakur_restored.csv`
- `artifacts/ensemble_strategy_20260525/candidate_csvs/sed_only_capped.csv`
- `artifacts/ensemble_strategy_20260525/candidate_csvs/sed_jung_tighter.csv`

They are under an audit artifact directory, not at repo root, and none is named `submission.csv`. The manifest and readiness report both keep submission approval disabled. These should remain inspection artifacts only.

## Caveats

- The local validation proxy remains narrow: `190` matched rows, `20` files, `6` sites, and `42` valid AUC classes.
- Bootstrap iterations in the Phase 2 artifact are `200`; this is acceptable for no-submit audit but not for submission approval.
- `sakur_visual` is explicitly `analysis_only_public_dry_run_branch`, so `sakur_restored` is not hidden-test-safe as-is.
- Candidate CSVs are valid BirdCLEF-shaped CSVs, so process discipline still matters: do not rename or submit them manually.

## Final decision

**ACCEPTED** for Phase 2 as a reusable no-submit audit workbench.

**Submission decision:** **not approved**. No recipe clears the stated gates, and `submit_approved=false` globally and per recipe.

## Required follow-ups

1. Keep this script as the standard no-slot audit gate for future genuinely new branches.
2. For any future submission candidate, rerun with submission-grade stability (`>=5000` site/file bootstrap iterations) and hidden-safe branch generation.
3. Do not promote `sakur_restored`, `sed_only_capped`, or `sed_jung_tighter` to Kaggle; they are controls/sensitivity checks only.
