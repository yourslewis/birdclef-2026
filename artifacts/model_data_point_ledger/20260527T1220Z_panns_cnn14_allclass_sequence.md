# PANNs/Cnn14 all-class train_soundscapes sequence data point — 2026-05-27 12:20 UTC

## Status
- Evidence level: comparison-grade model/data point plus no-submit local v616 proxy sidecar audit.
- Slot decision: no early-day Kaggle submission. The trained branch is finite/nonconstant and informative, but the proxy sidecar loses to v616 and is not a hidden-test package.

## Live competition state
- Latest Kaggle Bearer check before training: best public LB remains `0.949`.
- Latest scored submissions: v621/v622/v623 tied `0.949`, v625 `0.948`, v624 `0.943`.
- 2026-05-27 UTC slots used at start: `0/5`; ~11.7h to reset.
- No active local/trainer BirdCLEF jobs; trainer GPUs idle.

## Experiment
- Experiment id: `soundscape-sequence-panns-cnn14-allcls-r2-nofile-reg-losite-ep18-20260527`.
- Branch family: train_soundscapes sequence/file/site mining with a different AudioSet encoder.
- Training data: official `train_soundscapes` only, `1,478` 5s windows / `66` files / `9` sites.
- Target scope: all `234` taxonomy labels from soundscape supervision.
- Model/init: frozen PANNs/Cnn14 AudioSet embeddings (`2048` dim) + radius-2 no-file context MLP (`10,244` context features, hidden `384`, dropout `0.40`, AdamW, site-balanced BCE, 18 epochs).
- Validation split: leave-one-site, 7 completed folds (`S03`, `S08`, `S13`, `S15`, `S19`, `S22`, `S23`).

## Results

| Metric | Row-only | Context | Delta |
|---|---:|---:|---:|
| row macro AUC mean | 0.588246 | 0.647816 | +0.059571 |
| file-MIL macro AUC mean | 0.651697 | 0.670723 | +0.019026 |
| no-train row AUC mean | 0.521794 | 0.641399 | +0.119606 |
| non-Aves row AUC mean | 0.610005 | 0.679851 | +0.069846 |

Fold context deltas vs row-only: S03 `+0.197908`, S08 `-0.008732`, S13 `+0.071340`, S15 `+0.078324`, S19 `+0.056543`, S22 `+0.018585`, S23 `+0.003027`.

## Sidecar audit vs v616
- Audit artifact: `artifacts/soundscape_allclass_sidecar_audit/20260527T1220Z_panns_allclass_sequence/`.
- Wrapper: 240 proxy rows / 234 columns; 156 matched sequence rows, 84 anchor-filled rows; finite/nonconstant `240x234` CSV.
- Best recipe: `allcls_seq_w0p0025` local macro AUC `0.990943` / 42 valid classes.
- Lift vs anchor: `+0.000553`.
- Lift vs v616: `-0.002538`.
- Rank corr vs v616: `0.999693`; MAE vs v616 `0.005974`.
- Promotion gate: failed; not submission-grade.

## Comparison
- PANNs/Cnn14 all-class context row AUC `0.647816` beats DyMN10 all-class r2 row AUC `0.597633` by `+0.050183` and file-MIL `0.670723` beats `0.635285` by `+0.035438`.
- The local v616 proxy sidecar is slightly worse than DyMN10 all-class sidecar (`-0.002538` vs `-0.002372` lift vs v616), so direct OOF proxy wrapping still does not promote.

## Verifier / critic decision
- Verifier: training artifacts finite/nonconstant; final all-row head nonconstant `234/234`; TorchScript export exists; sidecar finite/nonconstant and schema-valid.
- Critic: strong leave-site all-class AudioSet landscape signal, but S08/S23 deltas are weak and v616-proxy lift is negative. Do not spend an early-day slot on this wrapper.
- Decision: keep as the best sequence-family local data point; reject direct proxy sidecar submission; next action should be true hidden-test packaging or no-call/acoustic-context integration rather than another OOF proxy weight sweep.

## Artifact paths
- Config: `configs/birdclef/soundscape_sequence_panns_cnn14_allcls_r2_nofile_reg_losite_ep18_20260527.json`
- Model artifact root: `artifacts/soundscape_sequence_mining/soundscape-sequence-panns-cnn14-allcls-r2-nofile-reg-losite-ep18-20260527/`
- Sidecar audit root: `artifacts/soundscape_allclass_sidecar_audit/20260527T1220Z_panns_allclass_sequence/`
- Canonical table: `artifacts/model_data_point_ledger/performance_table.md`
