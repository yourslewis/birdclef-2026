# Model Data Point — soundscape-native-b0-losite-nonaves-notrain-ep4-20260526

## Summary
- Timestamp: 2026-05-26 16:20 UTC
- Branch type: deeper soundscape-native compact CNN/SED data point.
- Evidence level: comparison-grade no-slot artifact.
- Decision: **keep as landscape data point; reject for submission/wrapper now**.

## Contract
- Model family: EfficientNet-B0 SED-style compact CNN, trained on logmel windows.
- Init/source: repo q3/cap80 train-audio TorchScript checkpoint `artifacts/external_pretrain/xc-b0-q3-cap80-external-pretrain-balanced-ep12/model_torchscript.pt`; head not loaded.
- Train rows: 1,478 official `train_soundscapes` 5s windows / 66 files / 9 sites.
- Labels/targets: 72 non-Aves or no-train labels; 5,420 positive target cells; class density 0.05093.
- Input window/features: 5.0s mono audio -> 160-mel logmel, hop 512, FFT 1024.
- Augmentations/regularization: mixup 0.1, label smoothing 0.01, AdamW weight decay 3e-4, observed-sqrt pos weights capped at 12, grad clip 5.
- Loss: BCE with logits.
- Epochs/runtime: 4 epochs per leave-site fold + 4-epoch all-row export smoke; total runtime `52.59s` on trainer CUDA.

## Metrics
- Completed leave-site folds: `6`; skipped `2` (`S15` too few valid classes, `S18` too few windows).
- Row macro AUC mean: `0.558044`.
- No-train row AUC mean: `0.573554`.
- Non-Aves row AUC mean: `0.558044`.
- File-MIL macro AUC mean: `0.429828`.
- Pooled row AUC: `0.396540` over `45` valid classes.
- Pooled no-train AUC: `0.305887` over `27` valid classes.

## Comparison / diversity value
- Stronger than the original single-site S08 B0 soundscape specialist in broad setup, but weaker than the best DyMN10 context-MLP sequence artifact (`0.601355` row / `0.632127` file-MIL).
- Useful as a negative/diagnostic point: direct compact logmel fine-tuning alone does not rescue train_soundscapes beyond frozen acoustic embeddings plus engineered file/context features.
- Some fold/class signal remains (notably S22 no-train file-MIL), but pooled no-train AUC is weak and fold stability is insufficient.

## Artifacts
- Script: `scripts/birdclef_soundscape_native_losite_train.py`
- Config: `configs/birdclef/soundscape_native_b0_losite_nonaves_notrain_ep4_20260526.json`
- Artifact root: `artifacts/soundscape_native_losite/soundscape-native-b0-losite-nonaves-notrain-ep4-20260526/`
- Metrics: `artifacts/soundscape_native_losite/soundscape-native-b0-losite-nonaves-notrain-ep4-20260526/metrics.json`
- Predictions: `artifacts/soundscape_native_losite/soundscape-native-b0-losite-nonaves-notrain-ep4-20260526/leave_site_predictions.npz`
- TorchScript: `artifacts/soundscape_native_losite/soundscape-native-b0-losite-nonaves-notrain-ep4-20260526/model_torchscript.pt`
- ONNX: `artifacts/soundscape_native_losite/soundscape-native-b0-losite-nonaves-notrain-ep4-20260526/model.onnx`

## Verification
- Prediction matrix shape: `1314 x 72`.
- Finite/nonconstant predictions: `72/72` nonconstant columns; min `0.00140194`, max `0.998872`, std `0.146312`.
- TorchScript smoke passed: `2x160x313 -> 2x72`, finite.
- ONNX status: `exported_checked`.
- Not submission-format; no 234-class wrapper; no v616 audit.

## Critic / verifier decision
- Critic: **do not continue direct native B0 as-is**; use context-MLP as control and pivot to wrapper/robustness or cleaner external feature branch.
- Verifier: **accepted no-slot artifact, rejected Kaggle submission**.
