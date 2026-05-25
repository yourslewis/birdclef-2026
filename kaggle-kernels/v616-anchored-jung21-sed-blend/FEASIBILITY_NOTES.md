# v616 anchored Jung21 + SED blend feasibility notes — 2026-05-25 04UTC

## Candidate from 02UTC grid

Best local rejection-screen blend:

```text
rank_final = 0.90 * rank(Samejima visual anchor)
           + 0.02 * rank(Sakur visual)
           + 0.04 * rank(Jungchan Model21 / subm_21)
           + 0.04 * rank(Raunak SED)
```

Local train-soundscape matched-row result:

- Base Samejima visual anchor AUC: `0.9903905` on `190` matched rows / `42` valid classes.
- Best blend AUC: `0.9935362`, lift `+0.0031457`.
- Corr vs anchor: `0.999641`; MAE: `0.006713`.
- Site bootstrap q05: `+0.00181`; leave-one-site min: `+0.002723` across 6 held-out sites.

This is strong rejection-screen evidence, **not approval evidence**. v611/v612 showed local sidecar gates can tie publicly.

## Hidden-safe reduction

Do not package a kernel that reads static public dry-run CSVs from other kernels. That would not rerun on hidden test rows.

The candidate can be simplified for hidden-safe implementation:

1. Samejima visual anchor and Samejima SED are already implemented/rerunnable in the repo-owned v612 scaffold.
2. Raunak SED public dry-run output is exactly identical to Samejima SED on the audited `240x235` rows (`maxabs=0`, corr≈1), so no Raunak source extraction is needed for the SED branch.
3. Sakur visual is optional; the 02UTC grid without Sakur still had strong local movement:

```text
rank_final = 0.92 * rank(Samejima visual anchor)
           + 0.04 * rank(Jungchan Model21 / subm_21)
           + 0.04 * rank(Samejima SED)
```

This Sakur-free version had local AUC around `0.9934807` (`+0.00309`) and avoids one full visual-family source import.

Therefore proposed v616 should target:

```text
rank_final = 0.92 * rank(anchor)
           + 0.04 * rank(jungchan_model21)
           + 0.04 * rank(samejima_sed)
```

## Source extraction performed

Artifacts:

- `artifacts/source_audits/v616_hidden_safe_branch_extraction_20260525T0400Z/summary.json`
- `artifacts/source_audits/v616_hidden_safe_branch_extraction_20260525T0400Z/jungchan.py.txt`
- `artifacts/source_audits/v616_hidden_safe_branch_extraction_20260525T0400Z/jungchan_model21_block.py.txt`
- `artifacts/source_audits/v616_hidden_safe_branch_extraction_20260525T0400Z/jungchan_model21_block_summary.json`

Jungchan Model21 slice:

- Start line: `1356` in decoded Jungchan source.
- End before Model52: line `9737`.
- Size: `8381` lines / `367451` chars.
- Contains required hidden-safe markers: `test_soundscapes`, `sample_submission.csv`, `perch_v2.onnx`, `ProtoSSM`, `ResidualSSM`, `write_final_submission`, `subm_21.csv`.

## Implementation blocker

The Model21 branch is large and self-contained, but it assumes the surrounding Jungchan notebook globals (`_ensemble_models`, `_runSED_once`, install/setup state, display helpers, shared imports, and potentially duplicated helper names). Blindly pasting it into the v612 scaffold risks silent conflicts and runtime failure.

The next safe implementation step is to create a minimal private verifier script that:

1. Starts from `kaggle-kernels/v612-anchored-sameji-hgnet57-pt/script.py` only for Samejima visual anchor + SED generation.
2. Adds a cleaned Jungchan Model21 function extracted from `jungchan_model21_block.py.txt`.
3. Forces `_ensemble_models = ['Model_21']` and `_runSED_once = False` for the Jungchan slice, since SED will come from the Samejima/v612 branch.
4. Writes diagnostics:
   - `submission_anchor_raw.csv`
   - `submission_samejima_sed_raw.csv`
   - `submission_jung21_raw.csv`
   - `submission_before_alignment.csv`
   - final `submission.csv`
5. Hard-fails if `submission_jung21_raw.csv` is missing, row-misaligned, non-finite, or constant.
6. Runs as private Kaggle verifier first; competition submission only after COMPLETE/no failure and schema/runtime validation.

## Slot decision

No competition slot should be used until the private v616 verifier completes. The local sidecar grid is strong enough to justify verifier packaging, but not direct submission.
