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
