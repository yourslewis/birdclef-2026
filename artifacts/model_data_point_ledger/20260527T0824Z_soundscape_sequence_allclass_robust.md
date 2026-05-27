# 2026-05-27 08:24 UTC — All-class DyMN10 robust r3 sequence/file/site ablation

## Status
- **Experiment:** `soundscape-sequence-dymn10-allcls-r3-robust-losite-ep24-20260527`
- **Family:** train_soundscapes sequence/file/site mining, all 234 taxonomy classes
- **Evidence level:** comparison-grade no-slot model data point
- **Submission decision:** no Kaggle submission; reject unchanged because it is much weaker than the 06:18 all-class r2 context baseline and unstable on S03/S08.

## Data / target contract
- Official `train_soundscapes` only: **1,478** 5s windows / **66** files / **9** sites.
- Target scope: **234** taxonomy labels; **6,244** positive target cells.
- Features/model: cached EfficientAT `dymn10_as` embeddings; radius-3 prev/next/local mean/local max/time context; no file mean/max; no site one-hot; smaller/stronger-regularized MLP (`hidden_dim=192`, dropout `0.45`, weight decay `0.002`, pos-weight power `0.25`).
- Validation: leave-one-site; **7** folds completed.

## Results
| Split | Row-only AUC | Context AUC | Delta | Row-only file-MIL | Context file-MIL | File-MIL delta |
|---|---:|---:|---:|---:|---:|---:|
| LOSO mean | 0.493697 | 0.501812 | +0.008115 | 0.523772 | 0.532188 | +0.008416 |

### Fold deltas
- S03: **-0.162908**
- S08: **-0.096429**
- S13: **+0.055647**
- S15: **+0.105244**
- S19: **+0.068256**
- S22: **+0.050721**
- S23: **+0.036274**

### Slice metrics
- Context no-train AUC mean: **0.509978**.
- Context non-Aves AUC mean: **0.540250**.
- Final all-row prediction export/smoke: finite, **234/234** nonconstant columns.

## Baseline comparison
- Versus same-config row-only: context is only **+0.008115** row AUC and **+0.008416** file-MIL.
- Versus 06:18 all-class r2 context baseline (`0.597633` row / `0.635285` file-MIL): robust r3 is **-0.095821** row and **-0.103097** file-MIL.
- Intended worst-site robustness did not hold: S03 and S08 regressed badly.

## Decision
**Reject unchanged; no submit.** This ablation confirms that stronger regularization/radius-3 context is too conservative/unstable. The useful all-class sequence baseline remains the 06:18 r2 no-file model, but its v616 sidecar audit also failed.

## Artifacts
- Artifact root: `artifacts/soundscape_sequence_mining/soundscape-sequence-dymn10-allcls-r3-robust-losite-ep24-20260527/`
- Config: `configs/birdclef/soundscape_sequence_dymn10_allcls_r3_robust_losite_ep24_20260527.json`
- Log: `logs/soundscape_sequence_dymn10_allcls_r3_robust_losite_ep24_20260527.log`
