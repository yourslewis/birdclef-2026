# PANNs/Cnn14 focused no-train train_soundscapes sequence data point — 2026-05-27 14:21 UTC

## Status
- Evidence level: comparison-grade model/data point; no Kaggle submission.
- Slot decision: no mid-day slot. There are 0/5 UTC submissions used and ~9.7h to reset; no verifier-grade nonduplicate candidate was ready.

## Live competition state
- Latest Kaggle Bearer check: best public LB remains `0.949`.
- Latest scored submissions: v621/v622/v623 tied `0.949`, v625 `0.948`, v624 `0.943`.
- Active jobs before training: no local/trainer BirdCLEF jobs; trainer GPUs idle.

## Experiment
- Experiment id: `soundscape-sequence-panns-cnn14-notrain-r2-nofile-reg-losite-ep24-20260527`.
- Branch family: train_soundscapes sequence/file/site mining with PANNs/Cnn14 AudioSet embeddings, focused on classes missing train-audio primary supervision.
- Training data: official `train_soundscapes` only, `1,478` 5s windows / `66` files / `9` sites.
- Target scope: `28` no-train-primary labels.
- Model/init: frozen PANNs/Cnn14 AudioSet embeddings (`2048` dim) + radius-2 no-file context MLP (`10,244` context features, hidden `192`, dropout `0.40`, AdamW, site-balanced BCE, 24 epochs).
- Validation split: leave-one-site, 6 completed folds (`S03`, `S08`, `S13`, `S19`, `S22`, `S23`).

## Results

| Metric | Row-only | Context | Delta |
|---|---:|---:|---:|
| row macro AUC mean | 0.563916 | 0.601305 | +0.037389 |
| file-MIL macro AUC mean | 0.638104 | 0.616149 | -0.021956 |

Fold context deltas vs row-only: S03 `+0.198539`, S08 `-0.029646`, S13 `-0.023220`, S19 `+0.005432`, S22 `+0.046688`, S23 `+0.026544`.

## Comparison
- Compared with DyMN10 focused no-train r2 (`0.553645` row / `0.638278` file-MIL), PANNs no-train is `+0.047660` row AUC but `-0.022129` file-MIL.
- Compared with PANNs all-class context (`0.647816` row / `0.670723` file-MIL), the focused target is weaker globally but specifically measures the no-train slice.
- Context improves row AUC over PANNs row-only, but file-MIL regresses; this is useful signal for row-level no-train ranking, not a package/submission candidate.

## Verifier / critic decision
- Verifier: leave-site predictions finite/nonconstant `1314x28`; final all-row head nonconstant `28/28`; TorchScript smoke on trainer produced finite `(2,28)` output.
- Critic: row-level no-train lift is real and beats DyMN10, but S08/S13 regress and file-MIL worsens. Do not spend an early/mid-day slot; use it as an AudioSet no-train data point and only package if integrated with a true hidden-test path plus site/file guards.
- Decision: continue/revise; no submission.

## Artifact paths
- Config: `configs/birdclef/soundscape_sequence_panns_cnn14_notrain_r2_nofile_reg_losite_ep24_20260527.json`
- Model artifact root: `artifacts/soundscape_sequence_mining/soundscape-sequence-panns-cnn14-notrain-r2-nofile-reg-losite-ep24-20260527/`
- Canonical table: `artifacts/model_data_point_ledger/performance_table.md`
