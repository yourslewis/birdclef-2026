# 2026-05-29 10:20 UTC — Soft1279 head-only constrained native adaptation

## Status
- Evidence level: comparison-grade model data point + package audit.
- Kaggle action: **no submission**. Early UTC day, 0/5 slots used, and the candidate failed v616-local promotion gates.

## Model / data
- Experiment: `soundscape-native-b0-soft1279init-headonly-losite-allcls-ep4-20260529`
- Branch family: soundscape-native constrained head-only calibration/domain adaptation.
- Data: official `train_soundscapes`; 1,478 5s windows / 66 files / 9 sites.
- Targets: all 234 taxonomy labels.
- Init/model: EfficientNet-B0 SED initialized from `sed-b0-oofteacher-b0v26-nfnetv29-soft-1279-ep4-20260528`, `initial_load_head=true`, `freeze_encoder=true`; only 75,114 head parameters trainable.
- Split: leave-one-site; 7 complete folds / 2 skipped low-count/low-valid sites.

## Training metrics
- Row macro AUC mean: `0.537303`
- No-train AUC mean: `0.557939`
- Non-Aves AUC mean: `0.512910`
- File-MIL AUC mean: `0.465716`
- Pooled row/no-train AUC: `0.280354` / `0.224432`
- Export/runtime: TorchScript + ONNX exported; ONNX status `exported_checked`; runtime `44.631s`.

## Package / sidecar audit
- Package path: `artifacts/sed_soundscape_packaging_audit/20260529T1020Z_soft1279init_headonly_package/`
- Rows: 240/240 v616 proxy rows matched; finite `True`; nonconstant columns `234`/234.
- Best non-control recipe: `soft1279init_headonly_native_allcls_w0p01`
  - Local macro AUC: `0.990502` / 42 valid classes
  - Lift vs v616: `-0.002979`
  - Lift vs anchor: `+0.000111`
  - Rank corr / MAE vs v616: `0.999633` / `0.005907`
- Submit approved: `False`

## Comparison / decision
- Versus the original soft1279 head-loaded sidecar grid (`w0.16`, local AUC `0.995545`, lift vs v616 `+0.002064`), head-only adaptation is much worse: best local AUC delta `-0.005043` and lift-vs-v616 delta `-0.005043`.
- Versus the head-loaded LOSO training row metric (`0.600360`), head-only row AUC regressed by `-0.063057` and file-MIL by `-0.140089`.
- Decision: **reject unchanged / no slot**. Freezing the encoder over-constrained adaptation and collapses local sidecar value; the positive head-loaded clue appears to require encoder movement, not just head recalibration.

## Artifacts
- Metrics: `artifacts/soundscape_native_losite/soundscape-native-b0-soft1279init-headonly-losite-allcls-ep4-20260529/metrics.json`
- Package audit summary: `artifacts/sed_soundscape_packaging_audit/20260529T1020Z_soft1279init_headonly_package/audit_summary.json`
- Candidate CSVs: `artifacts/sed_soundscape_packaging_audit/20260529T1020Z_soft1279init_headonly_package/audit/candidate_csvs/`
