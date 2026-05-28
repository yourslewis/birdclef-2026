# BirdCLEF model data point — PANNs/Cnn14 all-class localmax-only sequence

UTC: 2026-05-28 00:22

## Experiment
- **ID:** `soundscape-sequence-panns-cnn14-allcls-r2-localmaxonly-losite-ep20-20260528`
- **Branch family:** train_soundscapes sequence/file/site AudioSet mining — localmax-only temporal ablation
- **Data:** official train_soundscapes, 1,478 5s windows / 66 files / 9 sites
- **Target scope:** all 234 taxonomy classes
- **Model/init:** frozen PANNs/Cnn14 AudioSet embeddings + context MLP using current embedding, radius-2 local-max embedding, and time features; no prev/next concat, no local mean, no file mean/max
- **Validation:** leave-one-site; 7 completed folds

## Comparable performance

| Metric | Row-only | Localmax context | Delta |
|---|---:|---:|---:|
| Row macro AUC | 0.618042 | 0.641501 | +0.023459 |
| File-MIL macro AUC | 0.679564 | 0.681753 | +0.002189 |
| No-train AUC | n/a | 0.606530 | n/a |
| Non-Aves AUC | n/a | 0.669778 | n/a |

Fold deltas: S03 +0.037365, S08 +0.004626, S13 +0.083401, S15 +0.083700, S19 -0.048526, S22 +0.001861, S23 +0.001789.

## Baseline comparison
- Versus PANNs all-class r2 no-file context (`20260527T1220Z`): row `-0.006315`, file-MIL `+0.011030`.
- Versus PANNs all-class file-context (`20260527T1818Z`): row `-0.000701`, file-MIL `+0.029102`.
- Versus native B0 all-class (`20260527T2020Z`): row `+0.005340`, file-MIL `+0.007997`.

## Export/runtime/verifier
- Trainer: `CUDA_VISIBLE_DEVICES=1`; no active BirdCLEF trainer jobs at launch; GPU 1 free.
- TorchScript context head exported; final all-row predictions finite/nonconstant (`234/234` columns).
- Leave-site prediction artifact: `artifacts/soundscape_sequence_mining/soundscape-sequence-panns-cnn14-allcls-r2-localmaxonly-losite-ep20-20260528/leave_site_predictions.npz`.

## Decision
Keep as a comparison-grade data point and integration clue, but do **not** package/submit directly. It improved every fold except S19 and produced the strongest all-class sequence sidecar proxy lift versus anchor so far, but it still loses to v616 in the proxy audit.
