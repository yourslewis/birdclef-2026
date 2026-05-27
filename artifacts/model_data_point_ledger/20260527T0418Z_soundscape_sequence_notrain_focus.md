# 2026-05-27 04:18 UTC — Focused no-train train_soundscapes sequence/file/site DyMN10 context data point

## Status
- **Experiment:** `soundscape-sequence-dymn10-notrain-r2-nofile-reg-losite-ep24-20260527`
- **Family:** train_soundscapes sequence/file/site mining, focused no-train target scope
- **Evidence level:** comparison-grade no-slot model data point
- **Submission decision:** no Kaggle submission; not 234-class hidden-test package and S03 guard failed.

## Data / target contract
- Official `train_soundscapes` only: **1478** 5s windows / **66** files / **9** sites.
- Target scope: **28** no-train-primary labels; **1944** positive target cells; **440** scoped-negative rows.
- Features/model: cached EfficientAT `dymn10_as` embeddings; radius-2 prev/next/local mean/local max/time context; no file mean/max; no site one-hot; regularized MLP head.
- Validation: leave-one-site; 6 folds completed with at least 2 valid classes.

## Results
| Split | Row-only AUC | Context AUC | Delta | File-MIL row-only | File-MIL context | File-MIL delta |
|---|---:|---:|---:|---:|---:|---:|
| LOSO mean | 0.545666 | 0.553645 | 0.007980 | 0.602732 | 0.638278 | 0.035546 |

### Fold deltas
| Site | Val rows | Valid classes | Row-only AUC | Context AUC | Delta |
|---|---:|---:|---:|---:|---:|
| S03 | 48 | 2 | 0.457468 | 0.232305 | -0.225162 |
| S08 | 120 | 17 | 0.493533 | 0.530208 | 0.036675 |
| S13 | 48 | 2 | 0.772569 | 0.821289 | 0.048720 |
| S19 | 72 | 11 | 0.396326 | 0.440974 | 0.044648 |
| S22 | 954 | 2 | 0.629873 | 0.651271 | 0.021398 |
| S23 | 72 | 6 | 0.524225 | 0.645825 | 0.121600 |

## Baseline comparison
- Versus previous broad 72-label r2/no-file sequence branch no-train AUC (`0.489591`): focused no-train context AUC is **+0.064054**.
- Versus its own row-only DyMN10 no-train head: row AUC **+0.007980**, file-MIL **+0.035546**.
- Not directly comparable to the 72-label broad context MLP row AUC (`0.601355`) because this target scope is 28 no-train labels only.

## Verifier notes
- `leave_site_predictions.npz` finite/nonconstant: context predictions shape `1314 x 28`.
- Final all-row export smoke passed: TorchScript `context_head_torchscript.pt`; final predictions nonconstant columns **28/28**.
- Risk: S03 regressed badly (`-0.225162`), so this is not a wrapper/submission candidate without a site guard or scoped cap.

## Decision
**Revise / continue as targeted no-train clue.** The focused no-train objective beats the prior broad branch on no-train AUC and improves file-MIL, but fold instability blocks packaging. Next exact action: either build a S03-guarded no-train wrapper audit at very low capped weights, or pivot to a true 234-class DyMN10/AudioSet hidden-safe package with no-train caps.

## Artifacts
- Artifact root: `artifacts/soundscape_sequence_mining/soundscape-sequence-dymn10-notrain-r2-nofile-reg-losite-ep24-20260527`
- Config: `configs/birdclef/soundscape_sequence_dymn10_notrain_r2_nofile_reg_losite_ep24_20260527.json`
- Log: `logs/soundscape_sequence_dymn10_notrain_r2_nofile_reg_losite_ep24_20260527.log`
