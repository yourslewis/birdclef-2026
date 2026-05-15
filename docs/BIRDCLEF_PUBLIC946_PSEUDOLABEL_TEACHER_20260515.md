# Public946 Teacher Cache / Noisy-Student Pivot — 2026-05-15

Status: active diagnostic after `v551` and `v558` both tied public LB `0.946`.

## Motivation

The public946 retune/tiny-sidecar lane is saturated. The next useful work should create a reusable training artifact rather than spend another Kaggle slot. This diagnostic turns the repo-owned `v542` public946 replay outputs into teacher caches and gates them for Spec B pseudo-label/noisy-student work.

## Inputs

- Prediction source: `artifacts/kaggle_outputs/v542-afr1ste-updated-public946/`
- Labeled soundscape truth: `/Volumes/ExternalSSD/data/workspace_don/kaggle_birdclef2026/data/train_soundscapes_labels.csv`
- Cache script: `scripts/birdclef_public946_cache_summary.py`
- Threshold script: `scripts/birdclef_pseudolabel_threshold_sweep.py`
- Smoke config: `configs/birdclef/pl_public946_sed_hardconf_smoke8_20260515.json`

## Generated artifacts

Ignored artifacts/logs are intentionally not committed:

- `artifacts/public946_teacher_cache_v542_20260515T0355Z/predictions.npz`
- `artifacts/public946_teacher_cache_v542_20260515T0355Z/teacher_sed.npz`
- `artifacts/public946_teacher_cache_v542_20260515T0355Z/teacher_rankblend.npz`
- `artifacts/public946_teacher_cache_v542_20260515T0355Z/summary.json`
- `artifacts/pseudolabel_thresholds/public946_v542_sed_threshold_sweep_20260515T0355Z.json`
- `artifacts/pseudolabel_thresholds/public946_v542_rankblend_threshold_sweep_20260515T0355Z.json`
- `artifacts/pseudolabels/students/pl-public946-sed-hardconf-smoke8-20260515/metrics.json`
- `logs/public946_teacher_cache_v542_20260515T0355Z.log`
- `logs/public946_teacher_sed_threshold_sweep_20260515T0355Z.log`
- `logs/public946_teacher_threshold_sweep_20260515T0355Z.log`
- `logs/pl_public946_sed_hardconf_smoke8_20260515.log`

## Teacher cache findings

`v542` intermediate SED is the cleanest pseudo-label seed on the labeled soundscape overlap:

| Stream | Macro AUC | Top1 row recall | Top3 row recall | Top5 row recall | Notes |
|---|---:|---:|---:|---:|---|
| `sed` | `0.995976` | `0.978947` | `0.989474` | `0.994737` | sparse, high-precision teacher |
| `rankblend` | `0.992525` | `0.384211` | `0.626316` | `0.747369` | good AUC but too dense for hard pseudo-label positives |

Threshold sweep on `teacher_sed.npz` recommends hard-confidence positives at `power=1.0`, `positive_threshold=0.8`. On the labeled overlap this yields:

- `97` positive cells across `66` rows and `8` classes
- `97/97` true-positive cells on overlap (`precision_vs_truth = 1.0`)
- `true_cell_recall = 0.1448`
- Good negative-mask options:
  - conservative: `negative_threshold=0.001`, mask fraction `0.346`
  - practical smoke/default: `negative_threshold=0.005`, mask fraction `0.779`

The rank-blend teacher is not a good hard-positive seed: its best shortlist has only ~`0.237` positive precision at very high threshold, because rank-space scores are dense across many classes.

## Smoke result

Ran a 3-5-sample-style pipeline smoke with `max_rows=8`, `target_mode=hard_conf`, `positive_threshold=0.8`, `negative_threshold=0.005`, one epoch, batch size 2.

Result:

- status: `student_complete`
- device: CPU
- actual backbone: `tiny_cnn_sed` fallback
- runtime: `4.777s`
- target mask fraction: `0.2740`
- target positive cells: `1`
- teacher-vs-truth macro AUC on tiny split: `0.9375` over 2 valid classes
- student-vs-truth macro AUC on tiny split: `0.6473` over 2 valid classes
- TorchScript export: `0.184 MB`

This validates the decode/train/export plumbing, not model quality. The fallback backbone means the next meaningful training run should use the GPU/server environment or install/verify `timm` in the local environment before expecting EfficientNet/ConvNeXt quality.

## Recommendation

Continue Spec B, but do not submit anything yet. Next step should be a full 240-row diagnostic or GPU-backed student run using `teacher_sed.npz` with:

- `target_mode=hard_conf`
- `positive_threshold=0.8`
- `negative_threshold=0.005` initially
- max positives per row capped at `3`
- negative cap kept moderate to avoid all-negative domination

Only consider a Kaggle candidate after a trained student artifact produces competitive OOF/holdout diagnostics and lower correlation to public946.

## 2026-05-15 04:55 UTC full 240-row GPU diagnostic

Added and ran `configs/birdclef/pl_public946_v542_sed_hardconf_b0_5s_ep20_20260515.json` on GPU server `192.168.0.10` with CUDA/timm available.

Config highlights:

- Teacher: `artifacts/public946_teacher_cache_v542_20260515T0355Z/teacher_sed.npz`
- Backbone: `efficientnet_b0`
- Initial checkpoint: `artifacts/external_pretrain/xc-b0-q3-cap80-external-pretrain-balanced-ep6/model_torchscript.pt`
- Rows: all `240` v542 train-soundscape dry-run rows
- Target mode: `hard_conf`
- Positive threshold: `0.8`
- Negative threshold: `0.005`
- Caps: max positives per row `3`, max negatives per row `64`
- Epochs: `20`, restore best by val AUC

Result artifact paths:

- Remote/local metrics: `artifacts/pseudolabels/students/pl-public946-v542-sed-hardconf-b0-5s-ep20-20260515/metrics.json`
- Remote/local predictions: `artifacts/pseudolabels/students/pl-public946-v542-sed-hardconf-b0-5s-ep20-20260515/student_predictions.npz`
- Blend gate: `artifacts/pseudolabels/students/pl-public946-v542-sed-hardconf-b0-5s-ep20-20260515/blend_gate.json`
- Log: `logs/pl_public946_v542_sed_hardconf_b0_5s_ep20_20260515.log`

Metrics:

- status: `student_complete`
- device: `cuda`
- actual backbone: `efficientnet_b0`
- rows/train/val: `240 / 192 / 48`
- target mask fraction: `0.2752`
- target positive/negative cells: `97 / 15360`
- best val AUC: `0.81217` over 30 valid classes
- final student macro AUC: `0.75003` over 42 valid classes
- final teacher macro AUC: `0.99532` over 42 valid classes
- student/teacher correlation: `0.17263`
- student/teacher MAE: `0.38474`
- TorchScript export: `15.391 MB`
- runtime: `8.866s`

Blend gate on labeled-overlap rows:

- Student is too weak standalone for packaging.
- Tiny blend into SED teacher has only microscopic local lift: best observed `student_weight=0.01` gives macro AUC `0.995331` vs SED teacher `0.995316`.
- Blend into rankblend does not improve local AUC.

Decision: do **not** submit or package this hard-confidence student. Keep it as a low-correlation diagnostic. Next Spec B step should change the learning target (e.g. soft-anchor/high-confidence positives with supervised clip mix, or larger 792-row teacher cache) rather than scaling this exact hard-conf recipe.
