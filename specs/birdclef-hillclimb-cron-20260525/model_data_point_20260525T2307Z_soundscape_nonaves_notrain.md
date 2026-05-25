# Model Data Point — Soundscape Non-Aves / No-Train Specialist

Timestamp: 2026-05-25 23:07 UTC

## Summary

Trained a bounded non-Aves/no-train soundscape specialist as a measured search-landscape data point, not a submission candidate.

## Ledger

- **Model family:** EfficientNet-B0 SED-style CNN over logmel windows; specialist label head.
- **Init/source:** repo-owned q3/cap80 external-pretrain TorchScript encoder (`xc-b0-q3-cap80-external-pretrain-balanced-ep12`); 352 keys loaded, 2 head keys skipped.
- **Train rows:** 1,478 official `train_soundscapes` 5s windows.
- **Labels/targets:** 72 `nonaves_or_no_train` labels from taxonomy; multilabel targets from `train_soundscapes_labels.csv`; 5,420 positive cells, density 5.09%.
- **Input window/features:** 5s, 32 kHz, 160 mel bins, hop 512; one labeled soundscape window per row.
- **Augmentations:** none; `mixup_alpha=0.0`, label smoothing 0.
- **Loss:** BCE.
- **Epochs/runtime:** 3 epochs; 19.46s CUDA runtime; best val loss at epoch 2 (`0.26949`).
- **Validation/proxy:** site-holdout `S08` (120 windows); macro AUC `0.48865` over 18 valid scoped classes; no-train macro AUC `0.47610` over 17 valid classes; non-Aves macro AUC `0.48865` over 18 classes.
- **Prediction artifacts:** `artifacts/soundscape_specialists/soundscape-nonaves-notrain-b0-5s160-siteS08-ep3-20260525/holdout_predictions.npz`.
- **Export/runtime:** TorchScript and ONNX exported; `runtime_check.json` reports ONNX checker OK and CPU TorchScript smoke 0.093s for 2 logmel samples with output shapes `[2,72]` and `[2,10,72]`.
- **Correlation/blend vs anchor/v616:** not run; branch has no hidden/test row-aligned 234-class output yet. It must be wrapped into a 234-class branch and audited before any slot.
- **Diversity value:** high data-slice diversity (official soundscape windows; all no-train sonotypes + non-Aves labels), but first site-holdout metric is weak/inconsistent.
- **Critic decision:** keep as a useful negative/landscape data point; do not scale unchanged. Next use AudioSet embeddings or site-balanced training.
- **Verifier decision:** no-slot training is rule-safe; output is not competition format; no submission approved.

## Artifact paths

- Config: `configs/birdclef/soundscape_nonaves_notrain_b0_5s160_siteS08_ep3_20260525.json`
- Script: `scripts/birdclef_soundscape_specialist_train.py`
- Metrics: `artifacts/soundscape_specialists/soundscape-nonaves-notrain-b0-5s160-siteS08-ep3-20260525/metrics.json`
- Runtime check: `artifacts/soundscape_specialists/soundscape-nonaves-notrain-b0-5s160-siteS08-ep3-20260525/runtime_check.json`
- Remote/local log: `logs/soundscape_nonaves_notrain_b0_5s160_siteS08_ep3_20260525.log`
