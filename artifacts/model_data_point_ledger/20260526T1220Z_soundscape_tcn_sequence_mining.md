# Model Data Point — Soundscape Per-File TCN Sequence Mining — 2026-05-26 12:20 UTC

## Identity
- Experiment ID: `soundscape-tcn-dymn10-losite-ep20-20260526`
- Branch family: train_soundscapes sequence/file/site mining; compact per-file temporal convolutional network.
- Script: `scripts/birdclef_soundscape_tcn_mining.py`
- Config: `configs/birdclef/soundscape_tcn_dymn10_losite_ep20_20260526.json`
- Artifact root: `artifacts/soundscape_sequence_mining/soundscape-tcn-dymn10-losite-ep20-20260526/`

## Data / targets
- Source rows: official `train_soundscapes_labels.csv` only.
- Rows/files/sites: 1,478 5s windows, 66 files, 9 sites.
- Input features: cached EfficientAT DyMN10 AudioSet embeddings plus time-position features.
- Target labels: 72 non-Aves/no-train labels.
- Validation: leave-one-site folds with minimum 40 validation rows and 4 variable classes.

## Model / training
- Architecture: per-file residual TCN over ordered 5s windows.
- Hidden dim: 256.
- Layers: 3 dilated residual Conv1d blocks, kernel size 3.
- Dropout/input dropout: 0.30 / 0.05.
- Loss: BCE with clipped positive weights (`pos_weight_power=0.5`, clip 20).
- Sampling: site-balanced file sampling.
- Epochs: 20 per fold + 20 final all-row export/smoke model.
- Runtime: ~14.4s summed folds + final training on trainer CUDA.

## Metrics
Compared against previous context-MLP sequence branch `soundscape-sequence-dymn10-context-losite-ep16-20260526`.

- TCN leave-site row macro AUC mean: `0.547582`.
- Previous context MLP row macro AUC mean: `0.601355`.
- Delta: `-0.053773`.
- TCN file-MIL macro AUC mean: `0.606240`.
- Previous context MLP file-MIL macro AUC mean: `0.632127`.
- Delta: `-0.025887`.

Fold deltas vs context MLP row AUC:
- `S03`: `+0.195896` — useful diagnostic; TCN fixes the prior S03 context regression.
- `S08`: `-0.076063`.
- `S13`: `-0.053791`.
- `S19`: `-0.085756`.
- `S22`: `-0.021799`.
- `S23`: `-0.281125`.

## Export / runtime status
- Final predictions: finite and nonconstant, 72/72 nonconstant columns.
- Prediction stats: min `1.228e-6`, max `0.999726`, mean `0.056753`, std `0.210297`.
- TorchScript saved: `file_tcn_torchscript.pt`.
- TorchScript smoke passed on trainer: `(2,12,input_dim)->(2,12,72)`.

## Anchor / v616 audit
- Not run. This artifact is not competition-format and has no 234-class wrapper.
- No Kaggle submission was made.

## Diversity value
- High diagnostic value for train_soundscapes sequence modeling: it is a distinct per-file temporal model rather than independent-row shallow head.
- Negative as a standalone branch; positive as evidence that S03 may benefit from a residual/gated temporal smoother while S08/S19/S23 need stronger regularization or fallback to context MLP.

## Critic / verifier decision
- Critic: **REVISE**. Do not submit or wrap unchanged; use the S03 gain to motivate a guarded residual/gated sequence model, or move to deeper soundscape-native training.
- Verifier: **ACCEPTED as no-slot data point; REJECTED as submission**. Output is finite/export-smoked, but 72-label only and not group-stable enough.

## Next exact action
Train/audit a residual or gated sequence branch that combines row/context features with the TCN smoother under explicit S03/S22 guard, or pivot to compact soundscape-native CNN/SED if the gated smoother cannot beat the context-MLP leave-site mean.
