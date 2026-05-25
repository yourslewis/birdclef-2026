# Phase 1B — Prediction & Ensemble Report

## Executive conclusion

Current candidate artifacts are good enough to define a stronger **analysis baseline**, but not good enough to justify another competition slot from the same sidecar family.

The best numerical recipe from the available aligned outputs is still a bounded **rank-space anchored blend**:

```text
0.90 * Samejima visual anchor
0.02 * Sakur visual
0.04 * Jungchan Model21
0.04 * Samejima/Raunak SED
```

It is the top train-soundscape overlap candidate I found (`AUC 0.9935362`, `+0.0031457` over anchor; site-bootstrap q05 `+0.001810`; leave-one-site all positive). However, v616 already submitted the near-neighbor `0.92 anchor + 0.04 Jung21 + 0.04 SED`, scored `0.949`, and tied the plateau. The Sakur-restored recipe is only `+0.0000556` local AUC over v616, so I recommend **no second submission from this family** unless Phase 2 is explicitly a no-slot/private verifier or a genuinely new hidden-safe branch is added.

Primary actionable outcome: use the current blend as a **baseline harness**, not a slot candidate.

## Analysis artifacts produced

Report-only helper and derived tables:

- `specs/birdclef-ensemble-strategy-20260525/reports/prediction_ensemble_analysis.py`
- `specs/birdclef-ensemble-strategy-20260525/reports/prediction_ensemble_numeric_analysis.json`
- `specs/birdclef-ensemble-strategy-20260525/reports/prediction_ensemble_numeric_tables.md`
- `specs/birdclef-ensemble-strategy-20260525/reports/sidecar_grid_top_stability_local_20260525.json`
- `specs/birdclef-ensemble-strategy-20260525/reports/v616_syd52p_top_stability_local_20260525.json`
- `specs/birdclef-ensemble-strategy-20260525/reports/per_class_selector_local_20260525.json`

Existing artifacts read/cited:

- `artifacts/anchored_blend_audit/sidecar_grid_20260525T0200Z_fast.json`
- `artifacts/anchored_blend_audit/sidecar_grid_20260525T0200Z_top_stability.json`
- `artifacts/anchored_blend_audit/v616_per_class_selector_20260525T0810Z.json`
- `artifacts/anchored_blend_audit/v616_per_class_selector_minlift0_20260525T0810Z.json`
- `artifacts/anchored_blend_audit/v616_syd52p_grid_fast_20260525T1000Z.json`
- `artifacts/anchored_blend_audit/v616_syd52p_top_stability_20260525T1000Z.json`
- `artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/*`
- `artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/*`

Label source for local train-soundscape overlap:

- `/Volumes/ExternalSSD/data/workspace_don/kaggle_birdclef2026/data/train_soundscapes_labels.csv`

The repo-relative `data/train_soundscapes_labels.csv` referenced by prior JSON artifacts is not present in this clone, so I used the mounted canonical data path above.

## 1. Usable artifacts and why

### Summary counts

From `prediction_ensemble_numeric_analysis.json`:

- CSV paths scanned: `121`
- usable/aligned `240x235` prediction CSVs: `58`
- unique prediction matrices after exact hash dedupe: `10`
- rejected/unusable CSVs: `63`
- matched labeled train-soundscape rows: `190`
- valid AUC classes on those rows: `42`

### Usable groups

1. **Sidecar grid inputs** — usable.
   - Directory: `artifacts/anchored_blend_audit/sidecar_grid_inputs_20260525T0200Z/`
   - These are aligned `240x235`, finite, nonconstant dry-run train-soundscape branch outputs.
   - Meaningful unique members: `samejima_visual_anchor`, `sakur_visual`, `jungchan_model21`, `jungchan_protossm`, `raunak_protossm`, `raunak_sed`, `samejima_protossm`, `sakur_protossm`.
   - Exact duplicate found: `samejima_sed == raunak_sed` on audited rows.

2. **v616 private verifier outputs** — usable for actual repo-owned hidden-safe validation, but not for new-slot optimism.
   - Directory: `artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/`
   - Final and raw branches are finite, `240x235`, row-aligned, and nonconstant.
   - Dedupe results:
     - `submission_anchor_raw.csv == samejima_visual_anchor`
     - `submission_jung21_raw.csv == jungchan_model21`
     - `submission_samejima_sed_raw.csv == samejima_sed == raunak_sed`
     - `submission.csv == submission_before_alignment.csv`
   - Actual v616 score was `0.949`; this is the key caution against treating local lift as approval.

3. **Fresh-scout public kernel outputs** — mostly duplicate branch outputs; a few usable branches.
   - Directory: `artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/`
   - Usable recurring branches: `subm_21`, `subm_52p`, `submission_protossm`, `submission_sed`.
   - Most p949/SYD variants are duplicates of already-known branch matrices:
     - many `subm_21` files duplicate `jungchan_model21`;
     - many `submission_sed` files duplicate `raunak_sed` / Samejima SED;
     - many `submission_protossm` files duplicate `jungchan_protossm`;
     - `subm_52p` is a distinct ProtoSSM-like branch, but weak as a new sidecar.

### Rejected/unusable groups

The rejected CSVs were not suitable for ensemble design because they were one of:

- `3x235` sample-submission rows, not train-soundscape dry-run rows;
- `243x235` outputs with many nonfinite cells / malformed alignment;
- constant or near-constant fallback outputs;
- probe-only runs without prediction CSVs;
- failed/missing-model branches.

Detailed rejected paths and reasons are in `prediction_ensemble_numeric_tables.md` under “Rejected/unusable CSVs”.

## 2. Numeric diversity / comparison table

Metrics below are computed in rank space against the Samejima visual anchor unless noted. Local AUC is against matched train-soundscape labels and must be treated as rejection/comparison evidence only.

| unique artifact | role | local rank AUC | lift vs anchor | rank corr vs anchor | rank MAE vs anchor | prob MAE vs anchor | note |
|---|---:|---:|---:|---:|---:|---:|---|
| `samejima_visual_anchor` | anchor | `0.990391` | `0.000000` | `1.000000` | `0.000000` | `0.000000` | plateau anchor |
| `raunak_sed` / `samejima_sed` | SED branch | `0.995976` | `+0.005585` | `0.778911` | `0.144987` | `0.488670` | most locally complementary; standalone hidden evidence is not enough |
| `jungchan_model21` | Model21 branch | `0.987426` | `-0.002964` | `0.821531` | `0.126921` | `0.299041` | locally worse alone, useful in low-weight blend |
| `sakur_visual` | visual sidecar | `0.984775` | `-0.005615` | `0.954954` | `0.061103` | `0.059673` | high-corr small stabilizer; best recipe uses only 2% |
| `jungchan_protossm` | ProtoSSM branch | `0.986253` | `-0.004137` | `0.900163` | `0.096253` | `0.437583` | not selected by best grid |
| `raunak_protossm` | ProtoSSM branch | `0.984640` | `-0.005750` | `0.898378` | `0.096941` | `0.403804` | rejected in best grid (`0` weight) |
| `syd52p` (`subm_52p`) | SYD branch | `0.983729` | `-0.006661` | `0.899258` | `0.096649` | `0.403721` | distinct but only microscopic blend gain |
| `samejima_protossm` | ProtoSSM branch | `0.982155` | `-0.008236` | `0.899268` | `0.096823` | `0.403491` | duplicate-family / weak alone |
| `sakur_protossm` | ProtoSSM branch | `0.978405` | `-0.011986` | `0.866397` | `0.110622` | `0.347998` | weak alone; no grid role |
| `v616_submission` | already submitted final | `0.993481` | `+0.003090` | `0.999805` | `0.003161` | `0.039663` | hidden score tied `0.949` |

Most informative pairwise diversity results:

- `raunak_sed` is the only strongly different high-local-AUC branch:
  - rank corr vs anchor `0.778911`;
  - rank corr vs `jungchan_model21` `0.466481`;
  - rank corr vs ProtoSSM-like branches around `0.456–0.483`.
- `jungchan_model21` is moderately different from anchor (`0.821531`) and from SED (`0.466481`), which explains why low weight helps the grid even though standalone local AUC is below anchor.
- `sakur_visual` is highly correlated with anchor (`0.954954`) and should only be a tiny stabilizer, not a major member.
- `syd52p`, `raunak_protossm`, `samejima_protossm`, and `jungchan_protossm` are mutually close ProtoSSM-like variants; they add little after `Jung21 + SED`.

## 3. Best ensemble recipe(s)

### Recipe A — best numerical global rank blend; **no-submit / private verifier only**

```text
Transform: class-wise percentile rank
Final: 0.90 * Samejima visual anchor
     + 0.02 * Sakur visual
     + 0.04 * Jungchan Model21
     + 0.04 * Samejima/Raunak SED
Total sidecar weight: 0.10
```

Evidence:

- Source: `sidecar_grid_top_stability_local_20260525.json`
- Matched rows/classes: `190` rows / `42` classes
- Anchor local AUC: `0.9903905`
- Candidate local AUC: `0.9935362`
- Local lift vs anchor: `+0.0031457`
- Corr vs anchor: `0.9996411`
- MAE vs anchor: `0.0067127`
- Top3 row recall in existing script metric: `0.4895` vs base `0.4526`
- Site bootstrap (`1000` iters):
  - mean lift `+0.0049773`
  - q05 lift `+0.0018103`
  - p(lift > 0) `0.999`
- Leave-one-site:
  - all `6/6` held-out sites positive;
  - min lift `+0.0027233`;
  - q05 lift `+0.0027353`.

Expected benefit:

- Locally best among available branch blends.
- Uses all meaningfully different useful branches: anchor + SED + Jung21 + tiny Sakur visual.

Risk:

- It is only a near-neighbor of v616. v616 used `0.92 anchor + 0.04 Jung21 + 0.04 SED`, passed the same style of local/site gates, and tied hidden public LB at `0.949`.
- Incremental local lift over v616 is only about `+0.0000556` (`0.9935362 - 0.9934807`).
- Sakur visual is high-correlation to anchor, so it is unlikely to unlock hidden ordering by itself.

Recommendation:

- Use as a Phase 2 **baseline/private verifier** only if the coordinator wants a rerunnable Sakur-inclusive kernel.
- Do **not** spend a competition slot on this recipe without a new independent signal or a stricter validation result that clears the validation team’s bar.

### Recipe B — submitted v616 baseline; **keep as reference, not candidate**

```text
Transform: class-wise percentile rank
Final: 0.92 * Samejima visual anchor
     + 0.04 * Jungchan Model21
     + 0.04 * Samejima SED
```

Evidence:

- Actual v616 outputs: `artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/`
- Validation stats: final `240x235`, finite, nonconstant, row-aligned.
- Local AUC: `0.9934807`, lift `+0.0030902`.
- Corr/MAE from grid in rank space: corr `0.999696`, MAE `0.006244`.
- Hidden public LB: `0.949` tie.

Recommendation:

- Use as the no-submit baseline for future branch comparisons.
- Any new branch must beat this by more than microscopic local movement and should not be just another branch from the same public946/SYD/ProtoSSM/SED family.

### Recipe C — SYD52p near-duplicate; **rejected**

```text
Transform: class-wise percentile rank
Final: 0.90 * Samejima visual anchor
     + 0.04 * Jungchan Model21
     + 0.04 * Samejima SED
     + 0.02 * SYD52p
```

Evidence:

- Source: `v616_syd52p_top_stability_local_20260525.json`
- Local AUC: `0.9935006`, lift `+0.0031101` vs anchor.
- Site bootstrap q05 `+0.0017685`, p(lift > 0) `0.999`.
- Leave-one-site all `6/6` positive, min `+0.0027280`.

Risk/rejection reason:

- Only `+0.0000200` local AUC over v616.
- v616 just tied hidden LB, so this is not enough evidence for a second slot.
- `SYD52p` is ProtoSSM-like and not meaningfully new after Jung21/SED.

## 4. Rejected recipes and why

### Per-class adaptive sidecar selector — rejected

Source: `per_class_selector_local_20260525.json` and existing `v616_per_class_selector_20260525T0810Z.json`.

Result:

- Base AUC: `0.9903905`
- Leave-site CV AUC: `0.9903940`
- CV lift: `+0.0000035`
- Leave-group lift summary: min `0.0`, q05 `0.0`, p_gt_0 `0.0`
- All-row/in-sample lift: `+0.0028787`

Why rejected:

- It overfits the labeled overlap. The in-sample selector finds weights for `10/42` classes, but leave-site transfer is effectively zero.
- This should not become a Phase 2 kernel unless a future selector shows real site-held-out lift.

### Raw sidecar/direct branch submission — rejected

`raunak_sed` has excellent local rank AUC (`0.995976`, `+0.005585`) and strong diversity, but direct or branch-family public results have already plateaued:

- v614 direct Raunak v9 scored `0.949`.
- Samejima/Raunak SED is a branch member, not a robust final hidden-test solution by itself.
- v616 used the SED branch in a bounded hidden-safe blend and still tied.

Conclusion: keep SED as a **low-weight sidecar**, never standalone.

### ProtoSSM-heavy grids — rejected

Rejected members/weights:

- `raunak_protossm`
- `samejima_protossm`
- `jungchan_protossm`
- `sakur_protossm`
- `syd52p` beyond tiny exploratory weights

Why rejected:

- They are mutually high-correlation ProtoSSM-like variants.
- Standalone local AUC is below anchor.
- Best grids assign zero or tiny weights after `Jung21 + SED` are present.

### Scalar/power variants of v616 — rejected

Any scalar tweak around v616 is low-EV because:

- v616 already had clean local lift and site stability yet tied hidden public LB;
- SYD52p only adds `+0.000020` local AUC over v616;
- per-class adaptation shows no leave-site transfer.

### Static public-output blending — rejected for Phase 2 implementation

Public dry-run CSVs are valid for analysis, but Phase 2 must not read static public-output CSVs as hidden-test inputs. Any candidate must rerun/generate branches on mounted `test_soundscapes` and write raw branch outputs for validation.

## 5. Concrete commands / script changes needed for Phase 2

### Commands used for this analysis

Numeric artifact scan/dedupe/diversity table:

```bash
PY=/Users/yourslewis/.openclaw/workspace-don/kaggle/playground-series-s6e3/.venv/bin/python
$PY specs/birdclef-ensemble-strategy-20260525/reports/prediction_ensemble_analysis.py
```

Best Sakur-inclusive rank blend stability:

```bash
PY=/Users/yourslewis/.openclaw/workspace-don/kaggle/playground-series-s6e3/.venv/bin/python
LABEL=/Volumes/ExternalSSD/data/workspace_don/kaggle_birdclef2026/data/train_soundscapes_labels.csv
PYTHONPATH=scripts $PY scripts/birdclef_public946_multi_sidecar_weight_grid.py \
  --base-csv artifacts/anchored_blend_audit/sidecar_grid_inputs_20260525T0200Z/samejima_visual_anchor.csv \
  --sidecar sakur_visual=artifacts/anchored_blend_audit/sidecar_grid_inputs_20260525T0200Z/sakur_visual.csv \
  --sidecar jung21=artifacts/anchored_blend_audit/sidecar_grid_inputs_20260525T0200Z/jungchan_model21.csv \
  --sidecar sameji_sed=artifacts/anchored_blend_audit/sidecar_grid_inputs_20260525T0200Z/samejima_sed.csv \
  --weights sakur_visual=0.02 \
  --weights jung21=0.04 \
  --weights sameji_sed=0.04 \
  --labels-csv $LABEL \
  --max-total-weight 0.14 \
  --bootstrap-iters 1000 --bootstrap-group site --leave-one-group site \
  --output-json specs/birdclef-ensemble-strategy-20260525/reports/sidecar_grid_top_stability_local_20260525.json
```

SYD52p near-duplicate stability:

```bash
PYTHONPATH=scripts $PY scripts/birdclef_public946_multi_sidecar_weight_grid.py \
  --base-csv artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission_anchor_raw.csv \
  --sidecar jung21=artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission_jung21_raw.csv \
  --sidecar sameji_sed=artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission_samejima_sed_raw.csv \
  --sidecar syd52p=artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/joriahmed__bc2026-p949-syd-ort-effv2-a08__subm_52p.csv \
  --weights jung21=0.04 \
  --weights sameji_sed=0.04 \
  --weights syd52p=0.02 \
  --labels-csv $LABEL \
  --max-total-weight 0.10 \
  --bootstrap-iters 1000 --bootstrap-group site --leave-one-group site \
  --output-json specs/birdclef-ensemble-strategy-20260525/reports/v616_syd52p_top_stability_local_20260525.json
```

Per-class selector rejection check:

```bash
PYTHONPATH=scripts $PY scripts/birdclef_per_class_sidecar_selector.py \
  --base-csv artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission_anchor_raw.csv \
  --sidecar jung21=artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission_jung21_raw.csv \
  --sidecar sameji_sed=artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission_samejima_sed_raw.csv \
  --labels-csv $LABEL \
  --weights 0,0.005,0.01,0.02,0.04,0.06 \
  --max-total-weight 0.08 \
  --group site --min-train-pos 2 --min-train-neg 10 --min-lift 0.0005 \
  --output-json specs/birdclef-ensemble-strategy-20260525/reports/per_class_selector_local_20260525.json
```

### Phase 2 implementation options

#### Recommended: no new Kaggle slot; keep harness and wait for new signal

No production script change is required. Keep using:

- `scripts/birdclef_public946_multi_sidecar_weight_grid.py`
- `scripts/birdclef_per_class_sidecar_selector.py`
- `specs/birdclef-ensemble-strategy-20260525/reports/prediction_ensemble_analysis.py`

as the candidate rejection harness for future artifacts.

#### If Coordinator wants a no-slot/private verifier anyway

Create a new private-verifier kernel by forking the v616 scaffold:

- source base: `kaggle-kernels/v616-anchored-jung21-sed-blend/script.py`
- target: `kaggle-kernels/v617-anchored-sakur-jung21-sed-blend/script.py` or equivalent
- add hidden-safe generation of a Sakur visual branch; do not read the static `sakur_visual.csv`
- write raw outputs:
  - `submission_anchor_raw.csv`
  - `submission_sakur_visual_raw.csv`
  - `submission_jung21_raw.csv`
  - `submission_samejima_sed_raw.csv`
  - `submission_before_alignment.csv`
  - final `submission.csv`
- final blend:

```python
# class-wise rank values, not probability or logit average
final = (
    0.90 * rank(anchor)
    + 0.02 * rank(sakur_visual)
    + 0.04 * rank(jung21)
    + 0.04 * rank(samejima_sed)
)
```

Required guards:

- exact row and class alignment across raw branches;
- no nonfinite cells;
- all `234` class columns nonconstant;
- final row order matches sample submission / generated soundscape rows;
- final rank corr vs anchor target `>= 0.9994` and rank MAE around `0.006–0.008`;
- runtime margin better than v616 if possible (`~1088s` public dry-run including nbconvert was high).

Promotion rule:

- Private/no-slot verifier can be built for knowledge, but a competition submission should be blocked unless it introduces genuinely new hidden-safe signal beyond v616/SYD/ProtoSSM/SED or clears stricter validation thresholds from the Validation & Metrics report.
