# BirdCLEF public-source follow-up diagnostics — 2026-05-16

## Purpose

After the fresh public-source sweep, two ideas looked worth implementing as **non-submitting diagnostics** before spending another Kaggle slot:

1. `lucataco/bc26-scoredesc-conservative-ensemble` — conservative raw/rank overlay of strong public outputs.
2. `kruzzcc/bc26-yaroslav-sitehour-bn` — Yaroslav-style site/hour prior plus BirdNET-family final blending.

The post-v560 rule still applies: train-soundscape local gates can reject candidates, but cannot approve a 0.95 slot alone. Any candidate must be materially stronger than failed `v560` or backed by independent source/OOF evidence.

## Implementation

Added `scripts/birdclef_public_source_followup_diagnostics.py`.

The script ports and tests:

- **Conservative score-desc/rank overlay** inspired by Lucataco:
  - align candidate CSVs to an anchor;
  - form a small raw weighted mix;
  - add a small column-wise percentile-rank signal;
  - keep a fixed anchor term to avoid blunt public-output shifts.
- **Site+hour prior** inspired by Yaroslav/Kruzzcc:
  - parse `site` and UTC hour from train-soundscape row ids;
  - build global, hour, site, and site-hour label-frequency priors;
  - add prior logit to prediction logits;
  - evaluate both optimistic in-sample prior and leave-one-file/site crossfit prior.

## Validation commands

```bash
python -m py_compile scripts/birdclef_public_source_followup_diagnostics.py

python scripts/birdclef_public_source_followup_diagnostics.py \
  --base-csv artifacts/kaggle_outputs/v542-afr1ste-updated-public946/submission.csv \
  --labels-csv /Volumes/ExternalSSD/data/workspace_don/kaggle_birdclef2026/data/train_soundscapes_labels.csv \
  --candidate v558=artifacts/kaggle_outputs/v558-gateretune-a010-clip002/submission.csv \
  --candidate v551_clap_tiny=artifacts/kaggle_outputs/v551-public946-clap-int8-w0005/submission.csv \
  --candidate v550_snowflake=artifacts/kaggle_outputs/v550-public946-snowflake-sed-w001/submission.csv \
  --candidate v560_direct_v2s=artifacts/kaggle_outputs/v560-public946-direct-v2s-r003/submission.csv \
  --candidate v545_clap5=artifacts/kaggle_outputs/v545-public946-clap-int8/submission.csv \
  --scoredesc-sources v558,v551_clap_tiny,v550_snowflake,v560_direct_v2s \
  --scoredesc-weights 0,0.0025,0.005,0.01,0.02,0.03,0.05 \
  --scoredesc-max-total 0.08 \
  --sitehour-lambdas 0,0.0025,0.005,0.01,0.02,0.03,0.05,0.075,0.10,0.15,0.20,0.30 \
  --sitehour-groups file,site \
  --output-json artifacts/public_sweeps/followup_diagnostics_20260516T0800Z.json

python scripts/birdclef_public_source_followup_diagnostics.py \
  --base-csv artifacts/kaggle_outputs/v558-gateretune-a010-clip002/submission.csv \
  --labels-csv /Volumes/ExternalSSD/data/workspace_don/kaggle_birdclef2026/data/train_soundscapes_labels.csv \
  --candidate v542=artifacts/kaggle_outputs/v542-afr1ste-updated-public946/submission.csv \
  --candidate v551_clap_tiny=artifacts/kaggle_outputs/v551-public946-clap-int8-w0005/submission.csv \
  --candidate v550_snowflake=artifacts/kaggle_outputs/v550-public946-snowflake-sed-w001/submission.csv \
  --candidate v560_direct_v2s=artifacts/kaggle_outputs/v560-public946-direct-v2s-r003/submission.csv \
  --scoredesc-sources v542,v551_clap_tiny,v550_snowflake,v560_direct_v2s \
  --scoredesc-weights 0,0.0025,0.005,0.01,0.02,0.03,0.05 \
  --scoredesc-max-total 0.08 \
  --sitehour-lambdas 0,0.0025,0.005,0.01,0.02,0.03,0.05,0.075,0.10 \
  --sitehour-groups file,site \
  --output-json artifacts/public_sweeps/followup_diagnostics_v558_anchor_20260516T0810Z.json
```

## Results

### Anchor and known candidates

Using `v542` as anchor:

| candidate | local macro AUC | lift vs v542 | public result context |
|---|---:|---:|---|
| `v542` | 0.992524901 | — | 0.946 |
| `v558` | 0.992630437 | +0.000105536 | 0.946 |
| `v551_clap_tiny` | 0.992411639 | -0.000113262 | 0.946 |
| `v550_snowflake` | 0.992328830 | -0.000196071 | tied/dropped lane, not stronger |
| `v560_direct_v2s` | 0.992747137 | +0.000222236 | **0.945 failed despite local lift** |
| `v545_clap5` | 0.990108256 | -0.002416645 | 0.944 |

This reinforces the v560 lesson: even a `+0.000222` local train-soundscape lift was not approval-quality.

### Conservative score-desc overlay

Best v542-anchored Lucataco-style local overlay:

- weights: `v551_clap_tiny=0.05`, `v550_snowflake=0.01`, `v560_direct_v2s=0.02`
- macro AUC: `0.992563907`
- lift vs v542: `+0.000039006`
- MAE vs v542: `0.00434`

Best v558-anchored score-desc overlays produced **no lift over v558** (`0.992630437`, lift `0.0`).

Interpretation: the conservative rank overlay is safe/bounded, but the best lift is less than half of v558's local lift and far below the failed v560 local lift. It is a rejection, not a slot candidate.

### Site/hour prior

Optimistic in-sample prior looks huge:

- v542 anchor, best full in-sample lambda `0.20`:
  - macro AUC `0.993371874`
  - lift `+0.000846973`
  - but MAE vs anchor `0.2866` / corr `0.8795` — very large distribution shift.
- v558 anchor, best full in-sample lambda `0.10`:
  - macro AUC `0.993365394`
  - lift `+0.000734957`

Crossfit kills it:

- v542 anchor, leave-one-file best lambda `0.0025`: macro AUC `0.992455657`, lift `-0.000069244`.
- v542 anchor, leave-one-site best lambda `0.0025/0.005`: macro AUC `0.992528107`, lift only `+0.000003207`.
- v558 anchor, leave-one-file best lambda `0.0025`: macro AUC `0.992514573`, lift `-0.000115864`.
- v558 anchor, leave-one-site best lambda `0.005`: macro AUC `0.992630437`, lift `0.0`.

Interpretation: site/hour prior is mostly label leakage on the train-soundscape overlap. It may be useful inside a full source notebook where test site/hour distribution is known and prior tables are competition-legal, but our crossfit evidence says it is not safe as a standalone slot driver.

## Recommendation / combination plan

Do **not** submit either approach directly.

Best practical combination if we continue:

1. Keep `v558`/public946 as the anchor family because it tied 0.946 and is locally stronger than v542.
2. Use conservative score-desc only as a **small stabilizer** in future package generation, not as a primary signal. Local result says it can move predictions by MAE ~0.004 without breaking the dry-run, but it does not improve enough.
3. Treat site/hour prior as a gated micro-feature only:
   - lambda must be tiny (`<=0.005`) unless independent validation appears;
   - require leave-one-file and leave-one-site non-negative before considering a kernel;
   - never use the in-sample lift as approval evidence.
4. If we want one more public-source implementation pass, port `kruzzcc/bc26-yaroslav-sitehour-bn` only as a source-read/package experiment that emits separate `submission_sitehour_prior.csv` and `submission_birdnet.csv`, then run the same script plus public-output displacement checks. Do not queue it automatically.

Current best route to a real improvement remains **new independent OOF/source signal**, not another public-output overlay. The follow-up diagnostics make the public sweep useful as guardrails and code, but they do not clear the 0.95 slot bar.
