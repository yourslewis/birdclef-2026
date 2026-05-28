# 20260528T0422Z — Soundscape-native B0 soundscape-positive LOSO data point

## Summary
- **Experiment:** `soundscape-native-b0-losite-soundpos-ep5-20260528`
- **Branch family:** Deeper soundscape-native compact CNN/SED target redesign
- **Data:** official `train_soundscapes`, 1,478 windows / 66 files / 9 sites
- **Target scope:** soundscape-positive labels, 75 classes, 6,244 positive cells
- **Model/init:** EfficientNet-B0 SED, q3/cap80 external pretrain init, head reset, BCE + sqrt positive weighting, label smoothing 0.01, mixup 0.1
- **Validation:** leave-one-site; 7 complete folds, 2 skipped low-window folds

## Comparable metrics
- Row macro AUC: **0.658165**
- File-MIL macro AUC: **0.676383**
- No-train row AUC: **0.610377**
- Non-Aves row AUC: **0.603244**
- Pooled row AUC: **0.341538** / pooled no-train **0.227970** (diagnostic only; pooled site confounding is severe)
- Prediction guard: finite, nonconstant `1410x75` OOF preds, `75` nonconstant columns

## Baseline deltas
- Versus native B0 all-class LOSO (`20260527T2020Z`, 234-class): row **+0.022004**, file-MIL **+0.002627**; target scope differs (75 vs 234 classes).
- Versus DyMN10 soundscape-positive sequence head (`20260527T1018Z`): row **+0.140044**, file-MIL **+0.164219**.
- Versus PANNs/Cnn14 all-class no-file (`20260527T1220Z`): row **+0.010349**, file-MIL **+0.005660**; target scope differs.

## Export/runtime
- TorchScript: `artifacts/soundscape_native_losite/soundscape-native-b0-losite-soundpos-ep5-20260528/model_torchscript.pt` (15.188 MB)
- ONNX: `artifacts/soundscape_native_losite/soundscape-native-b0-losite-soundpos-ep5-20260528/model.onnx` (0.568 MB), status `exported_checked`
- TorchScript smoke finite: `True`, sample `[2, 160, 313]` -> clip `[2, 75]`

## Decision
**Revise / keep data point; no submission.** This is the strongest soundscape-positive/native data point so far by leave-site row and file-MIL, but the v616 proxy sidecar remains below v616 and the model only covers 75 classes. Useful next as a stronger target-design clue, not direct slot material.

## Artifacts
- Metrics: `artifacts/soundscape_native_losite/soundscape-native-b0-losite-soundpos-ep5-20260528/metrics.json`
- OOF predictions: `artifacts/soundscape_native_losite/soundscape-native-b0-losite-soundpos-ep5-20260528/leave_site_predictions.npz`
- Sequence-compatible audit npz: `artifacts/soundscape_native_losite/soundscape-native-b0-losite-soundpos-ep5-20260528/leave_site_predictions_sequence_compat.npz`
- Config: `configs/birdclef/soundscape_native_b0_losite_soundpos_ep5_20260528.json`
