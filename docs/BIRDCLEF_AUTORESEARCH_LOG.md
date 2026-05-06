# BirdCLEF AutoResearch Log

This log tracks spec-driven implementation/tuning work from `docs/BIRDCLEF_NEW_DIRECTIONS_SPECS.md`.

## 2026-05-06 06:50 UTC — `sed-b0-5s-attn-v1-smoke`

- **Track:** A+G Real SED frame/event smoke + export/inference packaging
- **Hypothesis:** Before scaling to EfficientNet-B0/V2/NFNet on GPU, validate that the repo has a real-audio weak-label SED pipeline that can decode BirdCLEF OGG files, build mel features, train frame/event logits, and export an artifact.
- **Branch:** `feature/sed-smoke-export-scaffold`
- **Config:** `configs/birdclef/sed_b0_5s_attn_smoke.json`
- **Script:** `scripts/birdclef_sed_smoke.py`
- **Data used:** local BirdCLEF data at `/Volumes/ExternalSSD/data/workspace_don/kaggle_birdclef2026/data`, smoke-limited to 5 real `train_audio/*/*.ogg` files.
- **Hyperparameters:** tiny CNN SED smoke backbone, 5s crop, 128 mel bins, hop 512, BCEWithLogits, AdamW lr 3e-4, batch size 2, 1 epoch, no mixup/label smoothing/class balancing.
- **Export:** TorchScript required; ONNX attempted only if local `onnx` package is available.
- **GPU status:** `yourslewis@192.168.0.23` (stale address) SSH timed out in this run, so no long GPU job was launched. Next run should retry GPU and scale this scaffold there.
- **Next step:** If smoke passes, add EfficientNet-B0/timm backbone on GPU or Kaggle image, then run 1-fold 2-epoch pilot and export ONNX/OpenVINO.

### Smoke result

- **Command:** `~/.openclaw/workspace-don/kaggle/playground-series-s6e3/.venv/bin/python scripts/birdclef_sed_smoke.py --config configs/birdclef/sed_b0_5s_attn_smoke.json --max-files 5 --epochs 1`
- **Status:** passed on 5 real OGG clips.
- **Input shape:** `[5, 128, 313]`
- **Frame logits shape:** `[5, 313, 234]`
- **Loss:** `0.79476 -> 0.74537`
- **Runtime:** `0.21s` for the tiny smoke model after decode/feature setup.
- **Artifact:** `artifacts/sed_smoke/sed-b0-5s-attn-v1-smoke/tiny_sed_smoke_torchscript.pt`
- **Metrics:** `artifacts/sed_smoke/sed-b0-5s-attn-v1-smoke/metrics.json`
- **ONNX:** blocked locally by missing `onnx` package (`ModuleNotFoundError`). This is now a clear dependency/setup task for the next A+G run.
- **Interpretation:** This is not a meaningful classifier yet; it proves the real-audio SED scaffold, weak-label frame output shape, training loop, and TorchScript export path work.

## 2026-05-06 07:50 UTC — `sed-smoke-sweep-v2`

- **Track:** A+G Real SED frame/event smoke + export/inference packaging.
- **Hypothesis:** After the first 5-file SED smoke passed, add a small AutoResearch-style knob sweep so later GPU pilots can choose between BCE, focal/class-balanced BCE, light label smoothing/mixup, and larger 10s/160-mel inputs instead of jumping blindly to EfficientNet.
- **Branch/PR:** `feature/sed-smoke-export-scaffold`, PR #204.
- **Scripts:**
  - `scripts/birdclef_sed_smoke.py` now supports `loss_name`, `focal_gamma`, `label_smoothing`, `mixup_alpha`, `class_balancing`, `val_fraction`, `duration_sec`, and `n_mels` overrides.
  - `scripts/birdclef_sed_smoke_sweep.py` runs a small CPU sweep and writes per-variant metrics under ignored `artifacts/sed_smoke/sweep-v2/`.
- **Data used:** 24 real BirdCLEF train OGG clips from `/Volumes/ExternalSSD/data/workspace_don/kaggle_birdclef2026/data`, with an 80/20 smoke split.
- **Command launched:** `~/.openclaw/workspace-don/kaggle/playground-series-s6e3/.venv/bin/python scripts/birdclef_sed_smoke_sweep.py --output-root artifacts/sed_smoke/sweep-v2`
- **Results:** all four variants passed smoke training/export checks:
  1. `sed-smoke-sweep-v2-5s-focal15-possqrt`: input `24x128x313`, focal BCE gamma 1.5 + sqrt positive weight, train loss `0.26659`, val loss `0.26442`.
  2. `sed-smoke-sweep-v2-10s-bce-m160`: input `24x160x626`, BCE, train loss `0.70210`, val loss `0.69681`.
  3. `sed-smoke-sweep-v2-5s-bce-m128`: input `24x128x313`, BCE, train loss `0.70040`, val loss `0.69746`.
  4. `sed-smoke-sweep-v2-5s-bce-smooth001-mixup02`: input `24x128x313`, BCE + label smoothing 0.01 + mixup 0.2, train loss `0.70066`, val loss `0.69772`.
- **Export:** TorchScript artifacts produced for every variant. ONNX remains blocked locally by missing `onnx` (`ModuleNotFoundError`) and should be resolved before the real export pilot.
- **Interpretation:** This is still operational smoke, not model selection. Focal + sqrt positive weighting clearly changes the loss scale and is the best next first GPU pilot candidate because it handles BirdCLEF's sparse multi-label imbalance more explicitly. The 10s/160-mel variant also validated memory/shape for a context pilot.
- **GPU status:** Retried stale address `ssh -o BatchMode=yes -o ConnectTimeout=8 yourslewis@192.168.0.23`; still timed out with exit 255. No remote durable GPU job launched.
- **Queue monitor:** Previous `mild-kelp` session was gone, so a new durable `nohup` monitor was started: `logs/submit_pending_birdclef_queue_20260506T075308Z.log`, pid `46665`. It verified v505 is COMPLETE, attempted submission, hit the daily cap with ~16h remaining, and is sleeping until retry.
- **Next step:** When GPU SSH is reachable, launch `sed-smoke-sweep-v2-5s-focal15-possqrt` as the first EfficientNet-B0/timm 1-fold 2-epoch pilot, plus a 10s/160-mel sibling if resources permit. Also install/enable ONNX export in the training/export environment.


## 2026-05-06 09:35 UTC — `sed-b0-gpu-pilot-v1-5s-focal15-possqrt`

- **Track:** A+G Real SED frame/event GPU pilot + export/inference packaging.
- **Hypothesis:** The best operational smoke variant (focal BCE gamma 1.5 + sqrt positive class weighting) should scale to a real EfficientNet-B0/timm weak-label SED pilot on the corrected GPU host and produce holdout prediction artifacts for later OOF/blend work.
- **Branch/PR:** `feature/sed-smoke-export-scaffold`, PR #204.
- **Infrastructure correction:** GPU host is `yourslewis@192.168.0.10` (`trainer`), not stale `192.168.0.23`. Verified 2x RTX 4090 and CUDA in `~/kaggle_envs/s6e3`.
- **Config:** `configs/birdclef/sed_b0_gpu_pilot_v1_focal_possqrt.json`.
- **Script:** `scripts/birdclef_sed_pilot_train.py`.
- **Hyperparameters:** EfficientNet-B0 via timm, 5s crops, 128 mels, hop 512, sample rate 32k, focal BCE gamma 1.5, sqrt positive class weighting, label smoothing 0, mixup 0, AdamW lr 3e-4 wd 1e-4, batch size 16, max files 512, 80/20 holdout, 2 epochs.
- **Setup:** Installed/enabled `timm`, `onnx`, `onnxscript`, and `imageio-ffmpeg` in the GPU venv as needed. `imageio-ffmpeg` is used as a portable fallback because system `ffmpeg` is not installed on the server.
- **Smoke/preflight:** Local tiny-CNN preflight passed on 6-8 real clips. Remote CUDA EfficientNet-B0 preflight passed on 8 real clips: device `cuda`, input `[8,128,313]`, TorchScript export size `15.388 MB`; ONNX initially failed on missing `onnxscript`, then `onnxscript` was installed before the full pilot launch.
- **Command launched:** on `192.168.0.10`, from `~/birdclef-2026`:
  `nohup env CUDA_VISIBLE_DEVICES=0 python scripts/birdclef_sed_pilot_train.py --config configs/birdclef/sed_b0_gpu_pilot_v1_focal_possqrt.json > logs/sed_b0_gpu_pilot_v1_20260506T094113Z.log 2>&1 &`
- **Remote PID/log:** pid `2524246`, log `~/birdclef-2026/logs/sed_b0_gpu_pilot_v1_20260506T094113Z.log`.
- **Expected artifacts:** `~/birdclef-2026/artifacts/sed_pilots/sed-b0-gpu-pilot-v1-5s-focal15-possqrt/metrics.json`, `holdout_predictions.npz`, `model_torchscript.pt`, optional `model.onnx`, config snapshot, and training log.
- **Final status:** complete.
- **Result:** 512 real clips, train/val `410/102`, input `[512,128,313]`, 2 epochs. Train loss `0.31540 -> 0.27830`; val loss `0.29509 -> 0.27717`; holdout macro AUC `0.51354` across 76 valid classes.
- **Artifacts:** `~/birdclef-2026/artifacts/sed_pilots/sed-b0-gpu-pilot-v1-5s-focal15-possqrt/metrics.json`, `holdout_predictions.npz`, `model_torchscript.pt` (`15.388 MB`), and `model.onnx` + external data (`0.56 MB` + `14.647 MB`). ONNX exported after installing `onnxscript`; PyTorch emitted opset conversion warnings but produced an ONNX artifact.
- **Interpretation:** First real EfficientNet-B0 SED prediction artifact exists. AUC is only smoke-holdout quality, but it is a less-correlated frame/SED model family and satisfies the artifact path needed for later blend/OOF work.


## 2026-05-06 09:42 UTC — `sed-b0-gpu-pilot-v2-10s-m160-focal15-possqrt`

- **Track:** A+G Real SED frame/event GPU pilot + crop/mel resolution sibling.
- **Hypothesis:** The 10s/160-mel context variant should improve weak-label SED discrimination over the 5s/128-mel pilot by adding more temporal context, while keeping the best smoke loss/class-balance settings fixed.
- **Branch/PR:** `feature/sed-smoke-export-scaffold`, PR #204.
- **Config:** `configs/birdclef/sed_b0_gpu_pilot_v2_10s_m160_focal_possqrt.json`.
- **Command launched:** on `192.168.0.10`, from `~/birdclef-2026`:
  `nohup env CUDA_VISIBLE_DEVICES=1 bash -lc "source ~/kaggle_envs/s6e3/bin/activate; python scripts/birdclef_sed_pilot_train.py --config configs/birdclef/sed_b0_gpu_pilot_v2_10s_m160_focal_possqrt.json" > logs/sed_b0_gpu_pilot_v2_20260506T094221Z.log 2>&1 &`
- **Hyperparameters:** EfficientNet-B0 via timm, 10s crops, 160 mels, hop 512, focal BCE gamma 1.5, sqrt positive class weighting, AdamW lr 3e-4 wd 1e-4, batch size 8, max files 512, seed 43, 80/20 holdout, 2 epochs.
- **Result:** complete. 512 real clips, train/val `410/102`, input `[512,160,626]`. Train loss `0.30597 -> 0.26143`; val loss `0.30555 -> 0.33323`; holdout macro AUC `0.57967` across 78 valid classes.
- **Artifacts:** `~/birdclef-2026/artifacts/sed_pilots/sed-b0-gpu-pilot-v2-10s-m160-focal15-possqrt/metrics.json`, `holdout_predictions.npz`, `model_torchscript.pt` (`15.388 MB`), and `model.onnx` + external data (`0.56 MB` + `14.647 MB`).
- **Interpretation:** 10s/160-mel sibling has better tiny holdout macro AUC than the 5s/128 pilot (`0.57967` vs `0.51354`) but worse final val loss, suggesting useful context signal with possible overfit/calibration drift. Next A+G move should add proper fold split/OOF and compare blend correlation with the v504/v508 prediction family before expanding epochs.

## 2026-05-06 09:46 UTC — `sed-b0-gpu-pilot-v3-10s-m160-seed42-focal15-possqrt` + matched split blend check

- **Track:** A+G Real SED frame/event GPU pilot + same-split comparison.
- **Hypothesis:** The prior 10s/160-mel v2 pilot used seed 43, so it could not be directly correlated/blended with the 5s/128-mel v1 holdout. Rerun the 10s/160-mel variant with seed 42 to match v1's file/holdout split and test whether crop/mel diversity produces complementary predictions.
- **Branch/PR:** `feature/sed-smoke-export-scaffold`, PR #204.
- **Config:** `configs/birdclef/sed_b0_gpu_pilot_v3_10s_m160_seed42_focal_possqrt.json`.
- **Comparison script:** `scripts/birdclef_compare_sed_pilots.py` aligns holdout files and computes per-model AUC, flat prediction correlation, and a simple blend grid.
- **Command launched:** on `192.168.0.10`, from `~/birdclef-2026`:
  `nohup env CUDA_VISIBLE_DEVICES=1 bash -lc "source ~/kaggle_envs/s6e3/bin/activate; python scripts/birdclef_sed_pilot_train.py --config configs/birdclef/sed_b0_gpu_pilot_v3_10s_m160_seed42_focal_possqrt.json" > logs/sed_b0_gpu_pilot_v3_20260506T094611Z.log 2>&1 &`
- **Result:** complete. 512 real clips, train/val `410/102`, input `[512,160,626]`. Train loss `0.31030 -> 0.26279`; val loss `0.31319 -> 0.27319`; holdout macro AUC `0.51991` across 76 valid classes. TorchScript and ONNX exported.
- **Matched split comparison:** v1 5s/128 seed42 vs v3 10s/160 seed42 aligned on 102 holdout files and 234 classes.
  - v1 macro AUC: `0.513541`
  - v3 macro AUC: `0.519907`
  - flat Pearson correlation: `0.164600`
  - mean absolute prediction difference: `0.030940`
  - best simple blend in grid: 50% v3 / 50% v1, macro AUC `0.573316`
  - blend grid AUCs by v3 weight: 0.0=`0.513541`, 0.1=`0.547246`, 0.2=`0.563622`, 0.3=`0.570250`, 0.4=`0.572934`, 0.5=`0.573316`, 0.6=`0.560046`, 0.7=`0.550302`, 0.8=`0.539320`, 0.9=`0.529698`, 1.0=`0.519907`.
- **Interpretation:** This is the first strong evidence that the SED crop/mel variants are complementary: individual tiny-holdout AUCs are modest, but same-split blend improves by about +0.060 over v1 and correlation is low. Next step is to convert this from tiny holdout into proper fold/OOF artifacts, then compare/blend against the v504/v508 teacher family if raw prediction artifacts can be located or regenerated.

## 2026-05-06 10:35 UTC — SED OOF runner + balanced-class OOF check

- **Track:** A+G Real SED frame/event OOF artifacts.
- **Hypothesis:** The prior 512-file OOF attempt was structurally weak because the default selector included many classes with too few examples, so folds often validated species with no training positives. Add explicit fold support plus a balanced-class selector to produce a more meaningful small OOF benchmark before scaling.
- **Branch/PR:** `feature/sed-smoke-export-scaffold`, PR #204.
- **Code/configs added:**
  - `scripts/birdclef_sed_oof_runner.py` — runs `birdclef_sed_pilot_train.py` across folds and aggregates `oof_predictions.npz`.
  - `scripts/birdclef_compare_oof_predictions.py` — aligns OOF files and computes AUC/correlation/blend grid.
  - `birdclef_sed_pilot_train.py` now supports `n_folds`, `fold_index`, and `selection_strategy=balanced_classes` with `max_classes`, `files_per_class`, `min_files_per_class`.
  - `configs/birdclef/sed_b0_balanced_oof_v1_5s_128.json`.
  - `configs/birdclef/sed_b0_balanced_oof_v3_10s_160.json`.
- **Smoke validation:** local tiny-CNN 2-fold OOF smoke passed on 6 real files.
- **Teacher artifact search:** no raw v504/v508 OOF/test prediction artifact was found locally; only kernel push/poll scripts and Perch cache files were visible. Comparison to v504/v508 will require regenerating/locating raw teacher predictions.

### Unbalanced 512-file OOF baseline

- Ran 3-fold OOF for the previous default 512-file selector.
- `sed-b0-oof-v1-5s-128-focal15-possqrt`: OOF macro AUC `0.499948` over 206 valid classes.
- `sed-b0-oof-v3-10s-160-focal15-possqrt`: OOF macro AUC `0.455467` over 206 valid classes.
- OOF blend comparison: flat Pearson `0.152588`; best blend was all v1 (`weight_b=0.0`, AUC `0.499948`).
- **Interpretation:** this OOF is not trustworthy for model selection because class coverage per fold is poor.

### Balanced-class 300-file OOF benchmark

- Selection: `balanced_classes`, 30 classes, 10 files/class, 300 files total, 3 folds; this ensures each fold has positives in train/validation for the chosen classes.
- `sed-b0-balanced-oof-v1-5s-128-focal15-possqrt`: OOF macro AUC `0.476316` over 30 valid classes. Fold AUCs were roughly low/mid `0.4` to `0.557793`.
- `sed-b0-balanced-oof-v3-10s-160-focal15-possqrt`: OOF macro AUC `0.534575` over 30 valid classes. Fold AUCs: `0.544453`, `0.594332`, `0.605642`.
- Balanced OOF comparison: aligned 300 files; flat Pearson `0.092054`; mean absolute prediction diff `0.024793`.
- Blend grid by v3 weight: 0.0=`0.476316`, 0.1=`0.486609`, 0.2=`0.498391`, 0.3=`0.508402`, 0.4=`0.514299`, 0.5=`0.521264`, 0.6=`0.526822`, 0.7=`0.531931`, 0.8=`0.532828`, 0.9=`0.533563`, 1.0=`0.534575`.
- **Interpretation:** On a more meaningful balanced-class OOF subset, 10s/160-mel is better than 5s/128, but the simple blend does not beat 10s alone. Low correlation still suggests diversity, but the 5s model is too weak at this setting. Next A+G move should improve the 10s model (more classes/files, more epochs, label smoothing/mixup or LR sweep), not push a Kaggle kernel yet.

## 2026-05-06 11:35 UTC — SED 10s/160 larger balanced OOF + smoothing/mixup A/B

- **Track:** A+G Real SED frame/event OOF hyperparameter tuning.
- **Hypothesis:** Since the balanced 30-class OOF showed 10s/160-mel is stronger than 5s/128, scale the 10s/160 setting to more balanced classes/files and test one regularization bundle (label smoothing 0.01 + mixup 0.2) while keeping backbone, crop, mel bins, loss, and class balancing fixed.
- **Branch/PR:** `feature/sed-smoke-export-scaffold`, PR #204.
- **Configs:**
  - `configs/birdclef/sed_b0_balanced_oof_v4_10s_160_moredata.json`
  - `configs/birdclef/sed_b0_balanced_oof_v5_10s_160_smooth_mixup.json`
- **Common setup:** EfficientNet-B0, 10s crops, 160 mels, focal BCE gamma 1.5, sqrt positive class weighting, 50 classes × 10 files/class = 500 files, 3 folds, 5 epochs, ONNX/TorchScript export.
- **v4 command:** on `192.168.0.10`, `CUDA_VISIBLE_DEVICES=0`, `scripts/birdclef_sed_oof_runner.py --base-config configs/birdclef/sed_b0_balanced_oof_v4_10s_160_moredata.json --output-root artifacts/sed_oof/sed-b0-balanced-oof-v4-10s-160-moredata --n-folds 3`.
- **v5 command:** on `192.168.0.10`, `CUDA_VISIBLE_DEVICES=1`, `scripts/birdclef_sed_oof_runner.py --base-config configs/birdclef/sed_b0_balanced_oof_v5_10s_160_smooth_mixup.json --output-root artifacts/sed_oof/sed-b0-balanced-oof-v5-10s-160-smooth-mixup --n-folds 3`.
- **v4 result (more data, no regularization):** OOF macro AUC `0.506684` over 50 valid classes. Fold AUCs: `0.635036`, `0.562333`, `0.602611`; final fold val losses around `0.1631`-`0.1705`. Artifacts under `~/birdclef-2026/artifacts/sed_oof/sed-b0-balanced-oof-v4-10s-160-moredata/`.
- **v5 result (label smoothing 0.01 + mixup 0.2):** OOF macro AUC `0.533127` over 50 valid classes. Fold AUCs approximately `0.607204`, `0.579449`, plus a high-0.5/low-0.6 first fold; final fold val losses around `0.1790`-`0.1983`. Artifacts under `~/birdclef-2026/artifacts/sed_oof/sed-b0-balanced-oof-v5-10s-160-smooth-mixup/`.
- **v4/v5 comparison:** aligned 500 files; v4 AUC `0.506684`, v5 AUC `0.533127`, flat Pearson `0.666455`, mean absolute diff `0.013875`. Blend grid barely beats v5 at v5 weight 0.9: AUC `0.533167`; v5 alone is effectively best.
- **Interpretation:** Scaling from 30 to 50 classes made the benchmark harder, but smoothing+mixup recovered most of the previous 30-class 10s/160 performance and clearly beat the unregularized larger run. Regularization helps, but the high v4/v5 correlation means this is not a new diversity axis. Next A+G action should tune 10s/160 regularized model learning rate/gamma or increase epochs carefully; alternatively move to a stronger backbone (EfficientNetV2-S/NFNet) on the same balanced OOF harness.

## 2026-05-06 12:35 UTC — SED stronger-backbone balanced OOF check

- **Track:** A+G Real SED frame/event model-zoo/backbone sweep on the balanced OOF harness.
- **Hypothesis:** Since EfficientNet-B0 10s/160 with smoothing+mixup is the current best SED configuration, test whether stronger Spec-A backbones add quality/diversity on the same 50-class balanced OOF benchmark. Smoke first, then scale only if safe.
- **Branch/PR:** `feature/sed-smoke-export-scaffold`, PR #204.
- **Configs:**
  - `configs/birdclef/sed_v2s_balanced_oof_v6_10s_160_smooth_mixup.json`
  - `configs/birdclef/sed_nfnet_balanced_oof_v7_10s_160_smooth_mixup.json`
- **Common setup:** 10s crops, 160 mels, 50 classes × 10 files/class = 500 files, 3-fold OOF, 5 epochs, focal BCE gamma 1.5, sqrt positive class weighting, label smoothing 0.01, mixup 0.2.
- **Preflight:** both backbones passed tiny CUDA preflight on `192.168.0.10`: `tf_efficientnetv2_s` on 12 files and `eca_nfnet_l0` on 8 files. ONNX export for V2-S was too slow/hung during first preflight, so full OOF runs used TorchScript export only (`export_onnx=false`) for these larger backbones.
- **v6 EfficientNetV2-S result:** OOF macro AUC `0.538471` over 50 valid classes. Fold AUCs: `0.605863`, `0.589326`, `0.594774`. TorchScript size about `81.443 MB` per fold.
- **v7 eca_nfnet_l0 result:** OOF macro AUC `0.565955` over 50 valid classes. Fold AUCs: `0.615302`, `0.634777`, `0.652672`. TorchScript size about `89.870 MB` per fold.
- **Backbone comparison against v5 B0 regularized baseline:**
  - v5 B0 AUC `0.533127`; v6 V2-S AUC `0.538471`; correlation `0.273131`; best simple blend at 50% V2-S = `0.547722`.
  - v5 B0 AUC `0.533127`; v7 NFNet AUC `0.565955`; correlation `0.352376`; best simple blend at 50% NFNet = `0.578510`.
  - v6 V2-S AUC `0.538471`; v7 NFNet AUC `0.565955`; correlation `0.588825`; best simple blend at 70% NFNet = `0.572567`.
- **Interpretation:** NFNet is the best SED backbone so far on the balanced OOF harness and also blends well with B0, giving the best observed SED OOF blend (`0.578510`) on this benchmark. This is a real model-family improvement, not a postprocess micro-sweep. Next step should either (a) launch a larger NFNet/B0 OOF with more classes/files or more epochs, or (b) start packaging an inference path for the NFNet+B0 SED ensemble once teacher/raw prediction artifacts are available for blend calibration.
