# Model data point — regularized DyMN10 context sequence ablation — 2026-05-26 18:20 UTC

## Summary
- **Experiment id:** `soundscape-sequence-dymn10-r2-nofile-reg-losite-ep20-20260526`
- **Branch family:** train_soundscapes sequence/file/site mining; DyMN10 frozen AudioSet embeddings + regularized context MLP.
- **Purpose:** one-variable-ish robustness ablation of the best 10:20 context-MLP control: wider local context radius, no file-mean shortcut, smaller hidden layer, heavier dropout/weight decay.
- **Evidence level:** comparison-grade no-slot model data point.

## Data / target contract
- Official `train_soundscapes` only.
- `1,478` 5s windows / `66` files / `9` sites.
- Target scope: `72` non-Aves/no-train labels; `5,420` scoped positive target cells.
- Validation: leave-one-site folds with min-row/min-valid-class gates; completed `6` folds (`S03`, `S08`, `S13`, `S19`, `S22`, `S23`).

## Model / config
- Script: `scripts/birdclef_soundscape_sequence_mining.py`
- Config: `configs/birdclef/soundscape_sequence_dymn10_r2_nofile_reg_losite_ep20_20260526.json`
- Embeddings: EfficientAT `dymn10_as` cached soundscape embeddings (`960d`).
- Context features: current + prev/next + radius-2 local mean/max + time features; **no file mean/max, no site one-hot**.
- Head: MLP, hidden `256`, dropout `0.35`, AdamW lr `6e-4`, weight decay `8e-4`, `20` epochs, site-balanced sampling, sqrt pos weights.

## Metrics
- Row-only mean AUC: `0.567307`
- Context mean AUC: `0.587753` over the same six leave-site folds
- Context delta vs row-only: `+0.020445`
- File-MIL context AUC: `0.664545`
- No-train context AUC: `0.489591`
- Non-Aves context AUC: `0.587753`

Fold deltas vs row-only:
- `S03`: `+0.075345` (fixed the prior context-regression site directionally)
- `S08`: `+0.007517`
- `S13`: `+0.006725`
- `S19`: `+0.020595`
- `S22`: `-0.063999` (still fails the S22 guard)
- `S23`: `+0.076490`

Comparison to current best sequence control (`20260526T1020Z` DyMN10 context MLP):
- Row AUC: `0.587753` vs `0.601355` = `-0.013602`
- File-MIL AUC: `0.664545` vs `0.632127` = `+0.032418`

## Verifier checks
- Leave-site prediction artifact: `1314 x 72` context predictions.
- Finite/nonconstant: all finite; `72/72` nonconstant columns; min `0.0`, max `0.963420`, std `0.113015`.
- TorchScript export exists and smokes by construction: `context_head_torchscript.pt` (`5.31 MB`).
- Not 234-class competition format; no v616 sidecar audit; no submission approved.

## Decision
**Continue as a file-MIL/robustness clue; reject as direct submission.**

This ablation improves file-level MIL and fixes S03 directionally, but lowers row AUC and still regresses S22. The best next action is a cautious 72→234 wrapper/audit only if it can cap S22/no-train risk, otherwise pivot to a multi-site DyMN10/AudioSet 234-class sidecar formulation or late-day slot-fill review.

## Artifacts
- Root: `artifacts/soundscape_sequence_mining/soundscape-sequence-dymn10-r2-nofile-reg-losite-ep20-20260526/`
- Metrics: `artifacts/soundscape_sequence_mining/soundscape-sequence-dymn10-r2-nofile-reg-losite-ep20-20260526/metrics.json`
- Predictions: `artifacts/soundscape_sequence_mining/soundscape-sequence-dymn10-r2-nofile-reg-losite-ep20-20260526/leave_site_predictions.npz`
- Log: `logs/soundscape_sequence_dymn10_r2_nofile_reg_losite_ep20_20260526.log`
