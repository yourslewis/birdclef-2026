# Model data point — 20s temporal/localmax B0 OOF-teacher SED — 2026-05-26 04:19 UTC

## Summary
- **Experiment id:** `sed-b0-oofteacher-b0v26-nfnetv29-soft-20s-localmax-512-ep3-20260526`
- **Model family:** EfficientNet-B0 SED-style frame model with clip output `0.5*mean + 0.5*amax` over temporal frames.
- **Purpose:** Test whether 20s temporal context/localmax behavior creates a decorrelated all-class raw branch relative to the 5s B0 OOF-teacher student family.
- **Evidence level:** comparison-grade landscape data point only.

## Training setup
- **Source/init:** repo-owned SED trainer, q3/cap80 external-pretrain TorchScript init `artifacts/external_pretrain/xc-b0-q3-cap80-external-pretrain-balanced-ep12/model_torchscript.pt`; 352 keys loaded, head skipped.
- **Training rows:** 512 OOF-teacher-cache-backed official train-audio files; split 410 train / 102 validation, random seed 43.
- **Labels/targets:** all 234 taxonomy labels; soft OOF teacher target from `b0v26_nfnetv29_w090010_intersection_cache.npz`; min available count 2; covered 512/512 rows.
- **Input window:** 20.0 seconds, 32 kHz, 160 mel bins, hop 512; resulting input shape `[512, 160, 1251]`.
- **Augmentations:** none; no mixup; BCE loss; no class balancing; no negative aux.
- **Epochs/runtime:** 3 epochs, batch size 4, CUDA runtime `20.778s`; decode/feature `20.762s`.

## Metrics
- Epoch losses:
  - epoch 1: train `0.493047`, val `0.376257`
  - epoch 2: train `0.337137`, val `0.325512`
  - epoch 3: train `0.325893`, val `0.322308`
- Best epoch by val loss: 3.
- Holdout macro AUC: **0.672996** over 72 valid classes.
- Correlation versus 5s soft-only 1024/ep4 B0 student on 407 overlapping files:
  - global Pearson correlation: **0.599986**
  - MAE: **0.036360**
  - 20s mean/std: `0.104971 / 0.043476`
  - 5s mean/std: `0.108007 / 0.067022`

## Export/runtime/verifier checks
- TorchScript export: `15.389 MB`.
- ONNX export: `0.56 MB` plus external data; ONNX checker passed.
- CPU TorchScript inference smoke: 4 files, 234 classes, runtime `0.301s`, `0.075s/file`; finite/nonconstant CSV/NPZ generated.
- Rule safety: official train audio + OOF teacher cache only; no hidden/test labels; no public-output-only final.

## Artifacts
- Config: `configs/birdclef/sed_b0_oofteacher_b0v26_nfnetv29_soft_20s_localmax_512_ep3_20260526.json`
- Artifact root: `artifacts/sed_oof_teacher_students/sed-b0-oofteacher-b0v26-nfnetv29-soft-20s-localmax-512-ep3-20260526/`
- Metrics: `artifacts/sed_oof_teacher_students/sed-b0-oofteacher-b0v26-nfnetv29-soft-20s-localmax-512-ep3-20260526/metrics.json`
- Holdout predictions: `.../holdout_predictions.npz`
- Correlation audit: `.../correlation_vs_5s_soft1024.json`
- Bundle manifest: `.../sed_bundle_manifest.json`
- Inference smoke output: `.../infer_smoke_probs.csv`, `.../infer_smoke_probs.npz`
- Trainer log: `logs/sed_b0_oofteacher_b0v26_nfnetv29_soft_20s_localmax_512_ep3_20260526.log`

## Critic decision
**Do not package or submit unchanged.** The branch is meaningfully decorrelated from the 5s B0 control, but the random-split proxy is too weak (`0.673`) to justify a slot or immediate scale. Treat it as a measured negative data point: naive 20s context with soft OOF teacher targets hurts strongly versus the 5s 1024-row soft-only control (`0.911`). If revisited, change target construction to true local-window/offset pseudo-labeling or use multiple 5s crops with localmax aggregation rather than a single padded 20s prefix.

## Verifier decision
**Accepted as a no-slot training artifact; rejected as submission-grade.** Export/runtime passed, but output is not a competition final and the proxy score is not candidate quality.
