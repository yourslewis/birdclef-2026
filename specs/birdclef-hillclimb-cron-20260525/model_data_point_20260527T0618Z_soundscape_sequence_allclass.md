# 2026-05-27 06:18 UTC — All-class train_soundscapes sequence/file/site DyMN10 context data point

## Status
- **Experiment:** `soundscape-sequence-dymn10-allcls-r2-nofile-reg-losite-ep18-20260527`
- **Family:** train_soundscapes sequence/file/site mining, all 234 taxonomy classes
- **Evidence level:** comparison-grade no-slot model data point
- **Submission decision:** no Kaggle submission; 234-class head/export exists, but hidden-test inference package and v616 sidecar audit are not yet complete.

## Data / target contract
- Official `train_soundscapes` only: **1478** 5s windows / **66** files / **9** sites.
- Target scope: **234** taxonomy labels; **6244** positive target cells; **0** rows without scoped labels.
- Features/model: cached EfficientAT `dymn10_as` embeddings; radius-2 prev/next/local mean/local max/time context; no file mean/max; no site one-hot; regularized MLP head.
- Validation: leave-one-site; **7** folds completed with at least 2 valid classes. Valid classes per fold: S03=4, S08=19, S13=6, S15=12, S19=17, S22=29, S23=13.

## Results
| Split | Row-only AUC | Context AUC | Delta | File-MIL row-only | File-MIL context | File-MIL delta |
|---|---:|---:|---:|---:|---:|---:|
| LOSO mean | 0.504940 | 0.597633 | 0.092693 | 0.487558 | 0.635285 | 0.147728 |

### Fold and slice metrics
| Site | Val rows | Valid classes | Row-only AUC | Context AUC | Delta | Context no-train AUC | Context non-Aves AUC | Context file-MIL AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S03 | 48 | 4 | 0.491653 | 0.635766 | 0.144113 | 0.408685 | 0.635766 | 1.000000 |
| S08 | 120 | 19 | 0.491299 | 0.536993 | 0.045694 | 0.504102 | 0.517540 | 0.532407 |
| S13 | 48 | 6 | 0.692824 | 0.756583 | 0.063759 | 0.758464 | 0.756583 | 0.800000 |
| S15 | 96 | 12 | 0.508880 | 0.529369 | 0.020488 | 0.489535 | 0.489535 | 0.450000 |
| S19 | 72 | 17 | 0.290312 | 0.574903 | 0.284590 | 0.476574 | 0.550566 | 0.529412 |
| S22 | 954 | 29 | 0.476513 | 0.558599 | 0.082085 | 0.588969 | 0.684869 | 0.589724 |
| S23 | 72 | 13 | 0.583100 | 0.591222 | 0.008122 | 0.594901 | 0.665537 | 0.545455 |

## Baseline comparison
- Versus the same all-class row-only DyMN10 head: context row AUC **+0.092693** and file-MIL **+0.147728**.
- Versus 2026-05-26 10:20 broad 72-label context sequence baseline (`0.601355` row / `0.632127` file-MIL): all-class context is **-0.003722** row and **+0.003158** file-MIL. This comparison is scope-shifted because the new head covers all 234 classes.
- Versus 2026-05-27 04:18 focused no-train context (`0.553645` row; no-train-only scope): all-class full-scope row AUC is **+0.043988**, while all-class no-train slice mean is **0.545890**.

## Verifier notes
- `leave_site_predictions.npz` finite/nonconstant: context predictions shape `1410 x 234`; **234/234** nonconstant columns.
- Final all-row TorchScript export smoke on trainer passed: `context_head_torchscript.pt` maps `(2, 4804) -> (2, 234)` with finite output.
- Risk: output is a model head, not yet an end-to-end Kaggle hidden-test package. Need EfficientAT hidden extraction/inference wrapper and v616 sidecar audit before any slot.

## Decision
**Continue/package next; no submit this run.** This is the strongest sequence-family clue since the original 72-label context MLP because every completed fold improved over row-only and the final head covers all 234 classes. Next exact action: build the hidden-safe EfficientAT DyMN10 feature extraction + 234-class context-head inference package, then audit low-weight sidecar recipes against v616 before considering a slot.

## Artifacts
- Artifact root: `artifacts/soundscape_sequence_mining/soundscape-sequence-dymn10-allcls-r2-nofile-reg-losite-ep18-20260527`
- Config: `configs/birdclef/soundscape_sequence_dymn10_allcls_r2_nofile_reg_losite_ep18_20260527.json`
- Log: `logs/soundscape_sequence_dymn10_allcls_r2_nofile_reg_losite_ep18_20260527.log`
