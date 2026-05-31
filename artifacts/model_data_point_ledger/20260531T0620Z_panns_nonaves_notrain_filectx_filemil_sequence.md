# PANNs/Cnn14 non-Aves/no-train file-context + file-MIL sequence data point — 2026-05-31 06:23 UTC

## Summary
- Experiment: `soundscape-sequence-panns-cnn14-nonaves-notrain-r2-filectx-filemil-losite-ep22-20260531`
- Branch family: train_soundscapes sequence/file/site AudioSet mining, targeted to non-Aves/no-train labels.
- Data: official `train_soundscapes`, 1,478 windows / 66 files / 9 sites, 72 labels, 5,420 scoped positive cells.
- Model/init: frozen PANNs/Cnn14 AudioSet embeddings plus MLP context head; radius-2 prev/next + local mean/max + file mean/max + time features; file-MIL BCE weight 0.40; site-balanced sampling.
- Validation: leave-one-site, 6 valid folds; v616 local proxy sidecar audit with 200 bootstraps.

## Performance
- Row-only baseline: row AUC `0.669160`, file-MIL AUC `0.720051`.
- Context/file-MIL head: row AUC `0.631592`, file-MIL AUC `0.690005`.
- Context delta vs same-run row-only: `-0.037568` row / `-0.030046` file-MIL.
- Slice metrics: no-train AUC `0.541630`, non-Aves AUC `0.631592`.
- Sidecar audit best recipe: `seq_context_w02`, local AUC `0.990695` / 42 valid classes, lift vs v616 `-0.002786`, lift vs anchor `+0.000304`, rank corr vs v616 `0.999615`, MAE `0.006435`.

## Decision
Reject as a slot candidate. The targeted 72-label PANNs scope improved over the older DyMN10 72-label context baseline (`+0.030237` row / `+0.057878` file-MIL), but it underperformed its own row-only head and remained clearly below v616 in the sidecar audit. Keep as a measured landscape point; do not submit.

## Artifacts
- Training output: `artifacts/soundscape_sequence_mining/soundscape-sequence-panns-cnn14-nonaves-notrain-r2-filectx-filemil-losite-ep22-20260531`
- Sidecar audit: `artifacts/model_data_point_ledger/20260531T0620Z_panns_nonaves_notrain_filectx_filemil_sequence`
- Config: `configs/birdclef/soundscape_sequence_panns_cnn14_nonaves_notrain_r2_filectx_filemil_losite_ep22_20260531.json`
- Logs: `logs/soundscape_sequence_panns_nonaves_notrain_r2_filectx_filemil_20260531.log`, `logs/panns_nonaves_notrain_filectx_filemil_sidecar_audit_20260531.log`
