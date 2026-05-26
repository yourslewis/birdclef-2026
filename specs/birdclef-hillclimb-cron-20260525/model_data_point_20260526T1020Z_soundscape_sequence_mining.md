# Model data point — train_soundscapes sequence/file/site mining — 20260526T1020Z

## Coordinator decision
**Comparison-grade data point accepted; no submission.** This run finally mined `train_soundscapes` as ordered files/sites rather than isolated rows. It produced a real sequence-aware signal: temporal/file-context DyMN10 features improved leave-site mean row AUC from `0.578422` to `0.601355` (`+0.022933`), and file-level MIL max-pooling AUC from `0.563852` to `0.632127`.

The branch is **not competition-format**: it is a 72-label non-Aves/no-train specialist head over cached DyMN10 embeddings. No Kaggle slot approved.

## Model / data contract
- Model family: sequence/file/site-aware shallow MLP over EfficientAT `dymn10_as` AudioSet embeddings.
- Source/init: cached public EfficientAT DyMN10 embeddings from `artifacts/efficientat_soundscape_embeddings/efficientat-dymn10-audioset-soundscape-nonaves-notrain-nocall-siteS08-ep12-20260526/efficientat_embeddings.npz`.
- Train rows: `1478` official `train_soundscapes` 5s windows, grouped as `66` files / `9` sites.
- Labels/targets: `72` non-Aves-or-no-train labels; `5420` positive scoped cells; `30` rows with no scoped label treated as in-scope negatives.
- Input features: current embedding + previous/next window + local mean/max over radius 1 + file mean + time-position features; no site one-hot shortcut. Feature dim `5764`.
- Sampling/loss: site-balanced row sampling, BCE with clipped sqrt pos-weight, 16 epochs, AdamW, dropout `0.20`.
- Validation: leave-one-site folds with >=40 rows and >=4 valid classes; row-level macro AUC plus file-level MIL max-pool AUC.

## Results
- Row-only LOSO mean AUC: `0.578422` (min `0.422777`, max `0.749633`).
- Context LOSO mean AUC: `0.601355` (min `0.428268`, max `0.805100`).
- Context delta: `+0.022933`.
- Row-only file-MIL mean AUC: `0.563852`.
- Context file-MIL mean AUC: `0.632127`.

### Fold deltas
- `S03`: 48 rows / 2 files, row-only AUC `0.479536`, context AUC `0.428268`, delta `-0.051268`.
- `S08`: 120 rows / 5 files, row-only AUC `0.562242`, context AUC `0.570197`, delta `+0.007956`.
- `S13`: 48 rows / 2 files, row-only AUC `0.749633`, context AUC `0.805100`, delta `+0.055466`.
- `S19`: 72 rows / 3 files, row-only AUC `0.422777`, context AUC `0.520003`, delta `+0.097226`.
- `S22`: 954 rows / 40 files, row-only AUC `0.592750`, context AUC `0.545639`, delta `-0.047111`.
- `S23`: 72 rows / 3 files, row-only AUC `0.663594`, context AUC `0.738923`, delta `+0.075329`.

## Diagnostics / artifacts
- Script: `scripts/birdclef_soundscape_sequence_mining.py`.
- Config: `configs/birdclef/soundscape_sequence_dymn10_context_losite_ep16_20260526.json`.
- Artifact root: `artifacts/soundscape_sequence_mining/soundscape-sequence-dymn10-context-losite-ep16-20260526`.
- Metrics: `artifacts/soundscape_sequence_mining/soundscape-sequence-dymn10-context-losite-ep16-20260526/metrics.json`.
- Data profile / co-occurrence diagnostics: `artifacts/soundscape_sequence_mining/soundscape-sequence-dymn10-context-losite-ep16-20260526/data_profile.json`.
- Leave-site predictions: `artifacts/soundscape_sequence_mining/soundscape-sequence-dymn10-context-losite-ep16-20260526/leave_site_predictions.npz`.
- TorchScript context head: `artifacts/soundscape_sequence_mining/soundscape-sequence-dymn10-context-losite-ep16-20260526/context_head_torchscript.pt`.
- Trainer log: `logs/soundscape_sequence_dymn10_context_losite_ep16_20260526.log`.

## Critic review
- Positive: this is the first run in the corrected top queue that genuinely treats `train_soundscapes` as files/sites/sequences. The mean lift is nontrivial for this small data point, especially on S13/S19/S23 and file-MIL.
- Negative: S22 is huge and context hurts there (`-0.047111`), while S03 also drops. This is not a monotonic promotion-grade win.
- Validation caveat: labels are sparse and site-correlated; leave-site AUC is comparison-grade, not hidden-safe approval. No-call remains unsupported by these positive-labeled soundscape rows.
- Opportunity-cost decision: continue this lane, but move from concatenated context features to a true compact sequence model or calibrated per-file smoother only if it preserves S22/S03.

## Verifier decision
- No-slot artifact checks passed on trainer: finite/nonconstant leave-site predictions `(1314, 72)` and TorchScript smoke `(2, 5764) -> (2, 72)`.
- Rule safety: official train soundscape labels plus public/pre-existing DyMN10 embeddings only.
- Submission decision: **REJECTED for Kaggle slot**. It is not a 234-column hidden-test package and lacks a v616 sidecar audit.

## Next exact action
Design a second sequence-mining data point that fixes the S22/S03 regressions: either (a) residual context head initialized/regularized against the row-only DyMN10 head, or (b) true temporal smoother/TCN over per-file predictions with leave-site loss. In parallel, keep a late-day fallback queue for slot-fill only if no packageable verifier-grade candidate emerges.
