# v612 anchored Samejima HGNet-v57 PT feasibility notes

Created after `samejimatink0/birdclef-2026-hgnetv2-b0-baseline-training` v57 produced usable PT checkpoints despite kernel ERROR during OOF aggregation.

## What is implemented

- Repo-owned private validation scaffold: `kaggle-kernels/v612-anchored-sameji-hgnet57-pt/`.
- Anchor source: Samejima visual CPU inference, preserved as `submission_anchor_raw.csv`.
- Sidecar source: Samejima HGNetV2-B0 v57 `best_model_fold*.pt` outputs mounted from the public training kernel source.
- Final: conservative rank blend `0.94 anchor + 0.06 Samejima HGNet-v57`.

## Why no competition slot yet

- v611 proved a different HGNet sidecar can validate cleanly and still only tie `0.949`.
- Samejima v57 training has strong fold CV (`~0.958/0.960/0.969/0.966`) but prior clean/train-audio CV was not approval evidence.
- This kernel must first prove mounted asset availability, runtime, nonconstant output, row alignment, and local movement.
