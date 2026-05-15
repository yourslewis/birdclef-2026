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


## 2026-05-15 05:55 UTC soft-anchor supervised target-design pivot

After the hard-confidence B0 run proved low-correlation but too weak, tested a softer target design instead of scaling the same recipe.

### Partial remote mirror run

Config: `configs/birdclef/pl_public946_v542_sed_softanchor_supervised_b0_5s_ep12_20260515.json`

- Target mode: `soft_anchor`
- Teacher: v542 `teacher_sed.npz`
- Soft label weight: `0.5`
- Anchors: positives `p>=0.8`, negatives `p<=0.005`
- Supervised clip mix: intended `160` files from `data/train.csv`, max `1` per class, weight `0.75`, smoothing `0.01`
- Result: only `38/160` supervised clips were usable because the remote train-audio mirror is partial and the original CSV sampling happened before path-existence filtering.
- Final student AUC: `0.911316` over 42 classes
- Best val AUC: `0.903733` over 34 classes
- Student/teacher corr: `0.869918`, MAE `0.097020`
- Blend gate: no useful lift into SED or rankblend teacher.

### Existing-audio manifest run

Built `artifacts/pseudolabels/manifests/train_existing_audio_manifest_20260515.csv` from the remote mirror: `3388` existing audio files across `206` classes. Then reran with `configs/birdclef/pl_public946_v542_sed_softanchor_supervised_existing160_b0_5s_ep12_20260515.json`.

- Supervised clip mix: `160/160` usable, zero missing paths
- Final student AUC: `0.936198` over 42 classes
- Best val AUC: `0.926750` over 34 classes
- Student/teacher corr: `0.908199`, MAE `0.046395`
- TorchScript: `15.391 MB`
- Runtime: `12.469s`
- Blend gate:
  - SED teacher baseline `0.995316`; student blend did not improve (`w=0.005` tie, `w>=0.01` drops)
  - rankblend baseline `0.990665`; all student blends drop

Interpretation: soft-anchor + supervised mix is a real improvement over hard-conf (`0.936` vs `0.750` all-row AUC), but still not strong or complementary enough to package. The path-filtered supervised manifest is important and should be reused. Next useful step is a larger/stronger teacher-cache or model-family run, not a Kaggle submission.


## 2026-05-15 06:55 UTC 792-row student/teacher ensemble gate

Audited the stronger existing 792-row public946 students before doing any more packaging.

Inputs on trainer:

- `pl-public946-sed-b0-5s-lr3e4-ep20-bestval`: B0, SED teacher, final AUC `0.976669`, teacher `0.996743`, corr `0.97746`.
- `pl-public946-rankblend-convnext-tiny-5s-lr3e4-ep20-bestval`: ConvNeXt-tiny, rankblend teacher, final AUC `0.987875`, teacher `0.994567`, corr `0.94308`.
- `pl-public946-rankblend-nfnet-5s-lr1e4-ep20-bestval`: NFNet-L0, rankblend teacher, final AUC `0.984806`, teacher `0.994567`, corr `0.92441`.

Artifact: `artifacts/pseudolabels/audits/public946_792_student_ensemble_gate_20260515T0655Z.json`.

Findings:

- Student-to-teacher blends do not beat the better teacher strongly enough to justify packaging.
- The useful signal was teacher-level, not student-level: blending `teacher_sed` with `teacher_rankblend` improved the labeled-overlap gate.
- Best checked mixture: `0.85 * teacher_sed + 0.15 * teacher_rankblend` = macro AUC `0.997018` over 75 classes, versus SED `0.996743` and rankblend `0.994567`.

Implemented utility `scripts/birdclef_blend_teacher_npz.py` and created reusable blended teacher cache on trainer:

- `artifacts/pseudolabels/public946-v540-teacher-cache66-v1/teacher_sed85_rankblend15.npz`
- `artifacts/pseudolabels/public946-v540-teacher-cache66-v1/teacher_sed85_rankblend15_summary.json`

The blended teacher has correlation `0.9248` vs SED and `0.5834` vs rankblend, and top-k recall `0.3128 / 0.7095 / 0.9223 / 0.9934` for k=1/3/5/10.

### Blended-teacher ConvNeXt smoke

Config: `configs/birdclef/pl_public946_sed85_rankblend15_convnext_tiny_5s_smoke_20260515.json`.

- ConvNeXt-tiny, 256 rows, 3 epochs, soft target from `teacher_sed85_rankblend15.npz`.
- Completed on CUDA in `10.04s`.
- Final student AUC: `0.799819` over 42 classes.
- Teacher AUC on same rows: `0.995304`.
- Corr/MAE: `0.42485` / `0.04334`.

Decision: kill this blended-teacher ConvNeXt smoke. It is far worse than the earlier public946 rankblend ConvNeXt smoke (`0.882870`) and should not be scaled. The blended teacher cache remains useful as a target artifact, but it needs a different learner/initialization or more careful curriculum, not direct ConvNeXt scaling.


## 2026-05-15 07:55 UTC blended-teacher B0 smoke + scale

After the direct blended-teacher ConvNeXt smoke failed, ran an EfficientNet-B0 control against the same `teacher_sed85_rankblend15.npz` target.

### B0 smoke

Config: `configs/birdclef/pl_public946_sed85_rankblend15_b0_5s_smoke_20260515.json`.

- B0 + external-pretrain init, 256 rows, 3 epochs, soft target from `teacher_sed85_rankblend15.npz`.
- Completed on CUDA in `4.845s`.
- Final student AUC: `0.900997` over 42 classes.
- Teacher AUC: `0.995304`.
- Corr/MAE: `0.56112` / `0.30058`.
- Scale gate: passed because it beat both the old B0 SED smoke (`0.818694`) and old ConvNeXt rankblend smoke (`0.882870`).

### B0 full 792-row scale

Config: `configs/birdclef/pl_public946_sed85_rankblend15_b0_5s_ep20_20260515.json`.

- 792 rows, 20 epochs, best-val restore.
- Completed on CUDA in `21.326s`.
- Best val AUC: `0.992890` over 61 classes.
- Final student AUC: `0.992137` over 75 classes.
- Teacher AUC: `0.997018` over 75 classes.
- Corr/MAE: `0.96336` / `0.01921`.
- TorchScript: `15.391 MB`.

Blend gate (`blend_gate.json`):

- Blending the B0 student into the blended teacher gives a tiny local lift:
  - teacher baseline: `0.997018`
  - best observed: `student_weight=0.01` -> `0.997046`
- Blending into SED teacher also improves SED (`0.996870` at `w=0.10` vs `0.996743`), but remains below blended-teacher+student.
- Blending into rankblend improves rankblend but remains far below the blended teacher.

Decision: do not submit/package yet. This is the best student artifact so far, but the local lift over the blended teacher is only `+0.000028`, too small for a fresh Kaggle slot after repeated public946 ties. Next step should be either a second seed/fold robustness check or a different learner/curriculum using the same blended teacher.
