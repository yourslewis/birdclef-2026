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

## 2026-05-06 13:35 UTC — SED B0/NFNet 100-class balanced OOF scale-up

- **Track:** A+G Real SED frame/event scaled balanced OOF.
- **Hypothesis:** The 50-class balanced OOF showed NFNet is the best SED backbone so far and NFNet+B0 gives the best SED blend. Scale the exact B0/NFNet 10s/160 regularized pair from 50 to 100 balanced classes to see whether the signal survives broader class coverage.
- **Branch/PR:** `feature/sed-smoke-export-scaffold`, PR #204.
- **Configs:**
  - `configs/birdclef/sed_b0_balanced_oof_v8_10s_160_100cls.json`
  - `configs/birdclef/sed_nfnet_balanced_oof_v9_10s_160_100cls.json`
- **Common setup:** 10s crops, 160 mels, 100 classes × 10 files/class = 1000 files, 3-fold OOF, 5 epochs, focal BCE gamma 1.5, sqrt positive class weighting, label smoothing 0.01, mixup 0.2, TorchScript export only (`export_onnx=false`).
- **v8 B0 command:** launched on `192.168.0.10`, `CUDA_VISIBLE_DEVICES=0`, output root `artifacts/sed_oof/sed-b0-balanced-oof-v8-10s-160-100cls`, log `logs/sed_oof_v8_b0_100cls_20260506T133739Z.log`.
- **v9 NFNet command:** launched on `192.168.0.10`, `CUDA_VISIBLE_DEVICES=1`, output root `artifacts/sed_oof/sed-nfnet-balanced-oof-v9-10s-160-100cls`, log `logs/sed_oof_v9_nfnet_100cls_20260506T133739Z.log`.
- **v8 B0 status/result:** complete. OOF macro AUC `0.485820` over 100 valid classes, 1000 OOF files. Fold 0 AUC `0.558764`; fold 2 AUC `0.555638`; overall AUC dropped materially vs 50-class v5 (`0.533127`), so B0 does not scale cleanly to broader class coverage in this setup.
- **v9 NFNet status at log time:** still running fold 2. Fold 0 AUC `0.618094` over 100 classes, fold 1 AUC `0.633719` over 98 classes; fold 2 child process active (`birdclef_sed_pilot_train.py --config ...config_fold2.json`) on GPU. Next run should collect `artifacts/sed_oof/sed-nfnet-balanced-oof-v9-10s-160-100cls/oof_summary.json`, compare v8/v9 if complete, and decide whether to scale NFNet further or tune it.
- **Interpretation so far:** B0 weakens badly at 100 classes, while NFNet fold 0/1 remain strong (>0.61 fold AUC). This supports continuing NFNet as the primary SED backbone and deprioritizing B0 except as a diversity/blend component if its correlation remains useful.

### v9 NFNet 100-class completion + v8/v9 comparison

- **v9 NFNet final result:** complete. OOF macro AUC `0.587033` over 100 valid classes, 1000 OOF files. Fold AUCs: `0.618094`, `0.633719`, `0.648795`. This is a strong scale-up from 50-class v7 (`0.565955`) despite doubling class count.
- **v8/v9 comparison:** aligned 1000 files. B0 v8 AUC `0.485820`; NFNet v9 AUC `0.587033`; flat Pearson `0.620113`; mean absolute diff `0.126029`.
- **Blend grid:** B0->NFNet weight 0.0=`0.485820`, 0.1=`0.524593`, 0.2=`0.539884`, 0.3=`0.552519`, 0.4=`0.562065`, 0.5=`0.570213`, 0.6=`0.577273`, 0.7=`0.582413`, 0.8=`0.586091`, 0.9=`0.587667`, 1.0=`0.587033`.
- **Interpretation update:** NFNet clearly scales; B0 mainly contributes a tiny complementary bump at ~10% weight. Best SED-only balanced OOF result so far is B0 10% + NFNet 90% = `0.587667`. Next actionable step should be NFNet-focused: either tune NFNet lr/gamma/epochs on 100-class OOF, or extend NFNet to more classes/files before building inference packaging.

## 2026-05-06 14:35 UTC — NFNet 100-class LR/gamma sweep launch

- **Track:** A+G Real SED frame/event NFNet-focused hyperparameter tuning.
- **Hypothesis:** NFNet is the strongest SED backbone so far on the 100-class balanced OOF harness. Test two single-knob variants against v9 baseline (`lr=3e-4`, focal gamma `1.5`, AUC `0.587033`): lower focal gamma to `1.0`, and lower learning rate to `1e-4`.
- **Branch/PR:** `feature/sed-smoke-export-scaffold`, PR #204.
- **Baseline:** `sed-nfnet-balanced-oof-v9-10s-160-100cls-smooth001-mixup02`, OOF AUC `0.587033`; best B0/NFNet blend `0.587667`.
- **Configs:**
  - `configs/birdclef/sed_nfnet_balanced_oof_v10_10s_160_100cls_gamma10.json` — same as v9, but focal gamma `1.0`.
  - `configs/birdclef/sed_nfnet_balanced_oof_v11_10s_160_100cls_lr1e4.json` — same as v9, but learning rate `1e-4`.
- **Common setup:** eca_nfnet_l0, 10s crops, 160 mels, 100 classes × 10 files/class = 1000 files, 3-fold OOF, 5 epochs, sqrt positive class weighting, label smoothing 0.01, mixup 0.2, TorchScript export only.
- **Commands launched:**
  - v10 on `192.168.0.10`, `CUDA_VISIBLE_DEVICES=0`, log `logs/sed_oof_v10_nfnet_gamma10_20260506T143709Z.log`, pid `2979001`.
  - v11 on `192.168.0.10`, `CUDA_VISIBLE_DEVICES=1`, log `logs/sed_oof_v11_nfnet_lr1e4_20260506T143709Z.log`, pid `2979003`.
- **Status at report time:** both OOF runners are still active. v10 fold 0 completed with AUC `0.599032` over 100 classes (below v9 fold 0 `0.618094`), then started fold 1. v11 had started fold 0 and was still running. Next run should collect both `oof_summary.json` files, compare v10/v11/v9, and decide whether focal gamma/lr tuning improves NFNet or whether to scale v9 directly.

## 2026-05-06 15:40 UTC — NFNet LR=1e-4 win + scale probes launched

- **Track:** A+G Real SED frame/event NFNet-focused scaling/tuning.
- **Hypothesis:** The prior NFNet 100-class sweep showed optimizer step size is a major knob. Lower LR may stabilize NFNet on weak-label SED training; if it holds under wider class coverage or longer training, this becomes the primary SED candidate for inference packaging.
- **Branch/PR:** `feature/sed-smoke-export-scaffold`, PR #204.
- **Completed sweep results:**
  - `v9` baseline NFNet, LR `3e-4`, focal gamma `1.5`: OOF macro AUC `0.587033` over 100 valid classes / 1000 files.
  - `v10` NFNet, LR `3e-4`, focal gamma `1.0`: OOF macro AUC `0.524482`; fold 2 collapsed (`val_loss=0.716058`), so lower gamma is rejected.
  - `v11` NFNet, LR `1e-4`, focal gamma `1.5`: OOF macro AUC `0.622721` over 100 valid classes / 1000 files. Fold AUCs `0.686372`, `0.654855`, `0.681563`.
- **OOF comparison artifacts:** on GPU server under `artifacts/sed_oof/comparisons/`.
  - `v9_vs_v11.json`: Pearson `0.873790`, mean abs diff `0.043709`; best blend uses v11 weight `0.7` for OOF AUC `0.628163`, better than v11 alone.
  - `v8_vs_v11.json`: Pearson `0.589312`, mean abs diff `0.141561`; best blend is v11 alone (`0.622721`), so B0 no longer adds useful signal after the LR fix.
  - `v10_vs_v11.json`: Pearson `0.659454`; best blend is v11 alone.
- **Interpretation:** LR `1e-4` is the clearest SED OOF improvement so far. v11 is both a stronger standalone SED model and a useful complement to the older LR `3e-4` NFNet, but B0 can be deprioritized.
- **New configs launched:**
  - `configs/birdclef/sed_nfnet_balanced_oof_v12_10s_160_150cls_lr1e4.json`: scale winning LR to 150 balanced classes × 10 files/class = 1500 files, 3-fold, 5 epochs.
  - `configs/birdclef/sed_nfnet_balanced_oof_v13_10s_160_100cls_lr1e4_ep8.json`: same 100-class benchmark as v11 but train 8 epochs to test whether longer low-LR training improves or overfits.
- **Commands launched on `192.168.0.10`:**
  - v12: `CUDA_VISIBLE_DEVICES=0`, pid `3114781`, log `logs/sed_oof_v12_nfnet_150cls_lr1e4_20260506T154031Z.log`, output `artifacts/sed_oof/sed-nfnet-balanced-oof-v12-10s-160-150cls-lr1e4/`.
  - v13: `CUDA_VISIBLE_DEVICES=1`, pid `3114783`, log `logs/sed_oof_v13_nfnet_100cls_lr1e4_ep8_20260506T154031Z.log`, output `artifacts/sed_oof/sed-nfnet-balanced-oof-v13-10s-160-100cls-lr1e4-ep8/`.
- **Status at log time:** both new OOF runners started fold 0 cleanly. Next run should collect v12/v13 summaries, compare against v11, and if v11/v13 remain best, start inference/kernel packaging for NFNet TorchScript folds and/or build a v9+v11 SED blend candidate.

## 2026-05-06 16:37 UTC — NFNet 8-epoch SED win + broader ep8 scale launch

- **Track:** A+G Real SED frame/event NFNet tuning/scaling.
- **Hypothesis:** The low-LR NFNet recipe improves with longer training on the same 100-class OOF harness. Test whether that 8-epoch recipe remains stable when class coverage expands beyond 100 classes.
- **Branch/PR:** `feature/sed-smoke-export-scaffold`, PR #204.
- **Completed results collected:**
  - `v11` NFNet 100-class, LR `1e-4`, 5 epochs: OOF macro AUC `0.622721` over 100 valid classes / 1000 files.
  - `v12` NFNet 150-class, LR `1e-4`, 5 epochs: OOF macro AUC `0.620044` over 150 valid classes / 1500 files. Fold AUCs `0.680537`, `0.663893`, `0.668911`.
  - `v13` NFNet 100-class, LR `1e-4`, 8 epochs: OOF macro AUC `0.636878` over 100 valid classes / 1000 files. Fold AUCs `0.695652`, `0.661716`, `0.681349`.
- **OOF comparison artifacts:** on GPU server under `artifacts/sed_oof/comparisons/`.
  - `v11_vs_v13.json`: Pearson `0.783948`, mean abs diff `0.042331`; best blend uses v13 weight `0.6` for OOF AUC `0.644676`, better than either model alone.
  - `v9_vs_v13.json`: Pearson `0.724891`, mean abs diff `0.069515`; best blend uses v13 weight `0.8` for OOF AUC `0.638740`.
- **Interpretation:** 8 epochs at LR `1e-4` is the strongest same-benchmark SED model so far, and blending 5-epoch + 8-epoch low-LR NFNet snapshots gives a large OOF gain. The 150-class 5-epoch run stayed stable and near the 100-class v11 score despite broader class coverage.
- **New configs launched:**
  - `configs/birdclef/sed_nfnet_balanced_oof_v14_10s_160_150cls_lr1e4_ep8.json`: 150 classes × 10 files/class, 8 epochs, 3-fold.
  - `configs/birdclef/sed_nfnet_balanced_oof_v15_10s_160_200cls_lr1e4_ep8.json`: 200 classes × 10 files/class, 8 epochs, 3-fold.
- **Commands launched on `192.168.0.10`:**
  - v14: `CUDA_VISIBLE_DEVICES=0`, pid `3280334`, log `logs/sed_oof_v14_nfnet_150cls_lr1e4_ep8_20260506T163713Z.log`, output `artifacts/sed_oof/sed-nfnet-balanced-oof-v14-10s-160-150cls-lr1e4-ep8/`.
  - v15: `CUDA_VISIBLE_DEVICES=1`, pid `3280336`, log `logs/sed_oof_v15_nfnet_200cls_lr1e4_ep8_20260506T163713Z.log`, output `artifacts/sed_oof/sed-nfnet-balanced-oof-v15-10s-160-200cls-lr1e4-ep8/`.
- **Status at log time:** both new runners started fold 0 cleanly. Next run should collect v14/v15 summaries, compare v14 against v12 and v13, then start NFNet TorchScript inference/kernel packaging around the best low-LR/8-epoch SED folds.

## 2026-05-06 17:35 UTC — NFNet v15 broad SED result + TorchScript bundle smoke

- **Track:** A+G Real SED frame/event inference packaging prep.
- **Hypothesis:** If the low-LR 8-epoch NFNet signal survives broad class coverage, package the best complementary TorchScript folds into a portable bundle that can become a Kaggle dataset/kernel input.
- **Branch/PR:** `feature/sed-smoke-export-scaffold`, PR #204.
- **Status checks:** Kaggle LB unchanged: latest scored v504/v503/v502/v501 at `0.927`, v500 at `0.926`; v505-v509 kernels COMPLETE/no failure messages; queue monitor pid `52652` is alive and sleeping on daily submission cap after v505 attempt.
- **Completed results collected:**
  - `v14` NFNet 150-class LR `1e-4`, 8 epochs: OOF macro AUC `0.620580` over 150 valid classes / 1500 files. This is essentially tied with v12 150-class 5-epoch (`0.620044`) and not a useful standalone upgrade.
  - `v15` NFNet 200-class LR `1e-4`, 8 epochs: OOF macro AUC `0.640274` over 181 valid classes / 1810 files. Fold AUCs `0.678960`, `0.692266`, `0.687347`; only 181 eligible classes reached the min-file threshold under the balanced selector.
- **OOF comparison artifacts:** on GPU server under `artifacts/sed_oof/comparisons/`.
  - `v12_vs_v14.json`: Pearson `0.658493`, mean abs diff `0.065642`; best blend uses v14 weight `0.6` for OOF AUC `0.633421` over 150 classes.
  - `v13_vs_v15.json`: on the 100-class overlap, v13 AUC `0.636878`, v15 overlap AUC `0.633091`, Pearson `0.294601`, mean abs diff `0.069209`; best blend uses v15 weight `0.6` for OOF AUC `0.657329`.
  - `v14_vs_v15.json`: on the 150-class overlap, v14 AUC `0.620580`, v15 overlap AUC `0.635854`, Pearson `0.359808`, mean abs diff `0.057227`; best blend uses v15 weight `0.7` for OOF AUC `0.652849`.
- **Interpretation:** v15 is the best broad-coverage SED candidate and has unusually low correlation with v13/v14, so a v13+v15 or v14+v15 fold blend is the right inference packaging target.
- **Packaging code added:**
  - `scripts/birdclef_sed_build_bundle.py` builds a manifest-based TorchScript SED bundle from one or more OOF experiment roots, with per-member blend weights and optional model copying.
  - `scripts/birdclef_sed_infer_torchscript.py` loads that manifest without timm/training code, decodes OGG via ffmpeg, recreates log-mel features, averages TorchScript fold probabilities, and writes wide CSV/NPZ predictions.
- **Bundle smoke validation on `192.168.0.10`:** built `artifacts/sed_bundles/sed-nfnet-v13v15-blend-v1/` from 6 TorchScript fold models with weights v13=0.4 and v15=0.6. Manifest has 6 models, 234 classes, copied model size `539.223 MB`. CPU smoke inference on 3 real train OGGs passed: 3 files × 234 classes, about `0.67 sec/file` with 2 torch threads after model load. Next step is a Kaggle-style inference script that maps soundscape 5s rows and blends SED probabilities into the current v504/v508 axis.

## 2026-05-06 18:35 UTC — Kaggle-style SED soundscape row inference smoke

- **Track:** A+G Real SED frame/event inference packaging prep.
- **Hypothesis:** The v13/v15 TorchScript SED bundle is only useful if it can produce BirdCLEF 5-second soundscape rows (`<soundscape_stem>_5` ... `_60`) in the exact `sample_submission` column shape. Validate that bridge before attempting a Kaggle kernel/dataset push.
- **Branch/PR:** `feature/sed-smoke-export-scaffold`, PR #204.
- **Status checks:** Kaggle LB unchanged: latest scored v504/v503/v502/v501 at `0.927`, v500 at `0.926`; v505-v509 kernels COMPLETE/no failure messages; queue monitor pid `52652` is alive and sleeping on daily submission cap after v505 attempt.
- **Code added:** `scripts/birdclef_sed_soundscape_infer.py`.
  - Loads the manifest bundle without timm/training code.
  - Decodes 60s OGG soundscapes with ffmpeg.
  - Emits one prediction row per 5s endpoint with Kaggle row ids.
  - Uses the trained model context length (10s) ending at each 5s row endpoint, zero-padded at file boundaries.
  - Aligns columns to `sample_submission.csv` when provided and can write CSV + compressed NPZ.
- **Smoke validation on `192.168.0.10`:** ran the v13/v15 bundle on one real train soundscape (`BC2026_Train_0001_S08_20250606_030007.ogg`) using CPU, 6 TorchScript folds, batch size 4, 2 torch threads. Output `artifacts/sed_bundles/sed-nfnet-v13v15-blend-v1/soundscape_smoke_submission.csv` has shape `12 x 235` (`row_id` + 234 labels), no NaNs, probability range `0.001024` to `0.422720`, and row ids `_5` through `_60`. Runtime was `6.409 sec/file` for a 60s soundscape.
- **Interpretation:** SED packaging now reaches real Kaggle row shape. The next implementation step is to embed this script into a Kaggle kernel candidate with the model bundle as an input dataset, then blend SED probabilities into the existing v504/v508 inference axis rather than submitting SED-only.

## 2026-05-06 19:50 UTC — v510 real SED bundle Kaggle kernel push

- **Track:** A+G Real SED frame/event inference packaging → Kaggle kernel candidate.
- **Hypothesis:** The strong low-correlation NFNet SED v13/v15 OOF bundle can add real temporal/model-family signal to the current v508 Perch/ProtoSSM axis if blended conservatively after existing probability shaping.
- **Branch/PR:** `feature/v510-real-sed-bundle-kernel`, PR #205. PR #204 was already merged, so this v510 work was moved to a fresh review branch.
- **Status checks:** Latest scored LB remains v504/v503/v502/v501 at `0.927` and v500 at `0.926`. v505-v509 kernels are `COMPLETE` with no failure messages, still waiting behind the daily submission cap.
- **Dataset packaging:** Created private Kaggle dataset `yourslewis/bc26-sed-nfnet-v13v15-bundle-v1` from `sed-nfnet-v13v15-blend-v1.zip` (manifest + 6 TorchScript NFNet folds; about 514 MB zipped / 539 MB unzipped). Upload used the repo helper `scripts/upload_kaggle_dataset_bearer.py` because legacy `kaggle datasets` CLI returned 401 under current KGAT auth.
- **Kernel candidate:** Added and pushed real Kaggle kernel `yourslewis/bc26-v510-real-sed-bundle-blend-005`, version 1.
- **Config:** Base is v508 (`ProtoSSM EW=0.625`, gamma `0.825`, context alpha `0.275`, top3 local-logit event propagation), plus `REAL_SED_BLEND_WEIGHT=0.05` after the v508 final probability post-processing.
- **Runtime guard:** v510 loads the zipped SED bundle from `/kaggle/input`, extracts it to `/kaggle/working`, selects TorchScript models with a time-budget guard (round-robin across v13/v15 if capped), emits 5-second soundscape row predictions aligned to `sample_submission.csv`, and falls back to pure v508 probabilities if the bundle is missing or inference fails.
- **Validation:** `py_compile` passed for the v510 kernel script, the dataset-upload helper, and the queue monitor script. Kaggle push returned version `1`, no invalid data/competition/kernel/model sources. Kernel status immediately after push: `RUNNING`, no failure message.
- **Queue monitor:** Refreshed monitor with v510 inserted after v509 and before old v376+ candidates. New monitor pid `68226`, log `logs/submit_pending_birdclef_queue_20260506T194754Z.log`; it retried v505 and is sleeping on the daily cap (~4.2h at launch). It will submit v505-v510 in order once quota returns, provided v510 completes.
- **Next step:** Monitor v510 kernel completion/failure. If it completes without timeout/model-mount issues, let the refreshed queue submit it after v505-v509. If v510 times out, reduce `REAL_SED_MAX_MODELS`/blend path to a 2-model v13+v15 representative bundle or precompute a lighter exported ONNX/OpenVINO SED path.

## 2026-05-06 20:45 UTC — v510 v1 fallback diagnosis + v2 mount-search fix

- **Track:** A+G Real SED frame/event Kaggle inference packaging and monitoring.
- **Hypothesis:** v510 version 1 completed but did not actually use the real SED signal because the SED dataset mounted as extracted files under a Kaggle-normalized directory rather than as `sed-nfnet-v13v15-blend-v1.zip` under the exact slug path.
- **Status checks:** Latest scored LB unchanged: v504/v503/v502/v501 at `0.927`, v500 at `0.926`. v505-v509 are `COMPLETE`. v510 v1 is `COMPLETE` with `submission.csv`, but its log shows `WARNING: real SED bundle dataset not found; using v508 probabilities only`, so it must not be submitted as the real SED candidate.
- **Root cause evidence:** Dataset API reports private dataset `yourslewis/bc26-sed-nfnet-v13v15-bundle-v1` is `ready` and lists extracted files (`sed_bundle_manifest.json` + six `models/*.pt`) rather than the zip archive. The v510 v1 finder only checked exact slug manifest paths plus recursive zip names, not recursive manifest paths.
- **Fix:** Updated `_sed_find_manifest()` to recursively search `/kaggle/input/**/sed_bundle_manifest.json`, print manifest candidates / input roots for debugging, and only then fall back to zip extraction. Updated the queue monitor so v510 submits kernel version `2` instead of bad/fallback version `1`.
- **Validation:** `py_compile` passed for v510 script and queue monitor. Next: push v510 version 2 via Bearer API, verify the log contains `Real SED manifest candidates` and `Applied real SED bundle blend`, then keep the queue monitor on v505-v510 with v510 version 2.

## 2026-05-06 21:45 UTC — v510 v2 verified + v511 blend weight 0.10 follow-up

- **Track:** A+G Real SED frame/event Kaggle inference packaging and lightweight blend-weight tuning.
- **Status checks:** Latest scored LB still unchanged: v504/v503/v502/v501 at `0.927`, v500 at `0.926`. v505-v509 are `COMPLETE`, and v510 v2 is now `COMPLETE` with `submission.csv`.
- **v510 v2 verification:** Output log confirms the real SED path actually ran: `Real SED manifest candidates: /kaggle/input/datasets/yourslewis/bc26-sed-nfnet-v13v15-bundle-v1/sed_bundle_manifest.json`, `Loading 6/6 real SED TorchScript models`, `Real SED prob range: 0.000003 to 0.624691, mean: 0.0617; runtime 214.4s`, and `Applied real SED bundle blend: weight=0.05`. Dry-run output shape was `240 x 235`, wall time `370.6s`; this is safely within Kaggle CPU budget on the public dry-run workload.
- **Follow-up hypothesis:** Since v510 v2 successfully uses all six SED models and runtime is acceptable, test a single stronger SED blend weight before pivoting tracks. v511 changes only `REAL_SED_BLEND_WEIGHT=0.05 -> 0.10` on the same v508 + real SED bundle path.
- **Kernel candidate:** Added and pushed real Kaggle kernel `yourslewis/bc26-v511-real-sed-bundle-blend-010`, version 1, with no invalid data/competition/kernel/model sources.
- **Queue monitor:** Updated monitor queue to submit v510 version 2, then v511 version 1, before old v376+ variants. Next step is to restart/verify the monitor with this updated queue and monitor v511 completion/logs.

## 2026-05-06 22:45 UTC — v511 verified + v512 ultra-conservative SED blend

- **Track:** A+G Real SED frame/event Kaggle inference packaging and lightweight blend-weight tuning.
- **Status checks:** Latest scored LB remains unchanged: v504/v503/v502/v501 at `0.927`, v500 at `0.926`; v505-v512 kernels are `COMPLETE` or running as noted below. Existing queue monitor was alive and sleeping on daily cap after v505 retry.
- **v511 verification:** v511 version 1 completed with `submission.csv` and confirmed real SED usage: found the SED manifest under `/kaggle/input/datasets/yourslewis/bc26-sed-nfnet-v13v15-bundle-v1/sed_bundle_manifest.json`, loaded `6/6` TorchScript models, `Real SED prob range: 0.000003 to 0.624691, mean: 0.0617; runtime 222.2s`, and applied `REAL_SED_BLEND_WEIGHT=0.10`. Dry-run output shape was `240 x 235`, wall time `354.1s`; final prob range `0.017495` to `0.914253`, mean `0.4115`.
- **Follow-up hypothesis:** Complete the small planned SED blend-weight bracket (`0.02`, `0.05`, `0.10`) with a safer low-weight variant in case the real SED model improves rank diversity but is undercalibrated versus the v508 axis.
- **Kernel candidate:** Added and pushed real Kaggle kernel `yourslewis/bc26-v512-real-sed-bundle-blend-002`, version 1, changing only `REAL_SED_BLEND_WEIGHT=0.02` from the same v508 + SED bundle path. Kaggle push returned version `1` with no invalid sources.
- **Queue monitor:** Updated queue to submit v510 version 2, then v511 version 1, then v512 version 1 after v505-v509 and before old v376+ variants. Next step: restart/verify the monitor with v512 included and monitor v512 completion/logs for the same SED markers.

### Monitor refresh after v512 push

- Restarted consolidated queue monitor with v512 included: pid `72673`, log `logs/submit_pending_birdclef_queue_20260506T223724Z.log`.
- It retried v505 and hit the daily submission cap again, with about `82 minutes` remaining until UTC reset at restart time.
- Final kernel status in this run: v510 `COMPLETE`, v511 `COMPLETE`, v512 `RUNNING` with no failure message and no output log yet. Next run should verify v512 logs for `Real SED manifest candidates`, `Loading 6/6 real SED TorchScript models`, `Applied real SED bundle blend: weight=0.02`, and `submission.csv saved`.

## 2026-05-06 23:45 UTC — v512 verified + prioritize real SED submissions at reset

- **Track:** A+G Real SED frame/event Kaggle inference packaging and submission monitoring.
- **Status checks:** Latest scored LB still unchanged: v504/v503/v502/v501 at `0.927`, v500 at `0.926`. v505-v512 kernels are all `COMPLETE` with no failure messages.
- **v512 verification:** v512 version 1 completed with `submission.csv` and confirmed real SED usage: found the SED manifest under `/kaggle/input/datasets/yourslewis/bc26-sed-nfnet-v13v15-bundle-v1/sed_bundle_manifest.json`, loaded `6/6` TorchScript models, `Real SED prob range: 0.000003 to 0.624691, mean: 0.0617; runtime 233.3s`, and applied `REAL_SED_BLEND_WEIGHT=0.02`. Dry-run output shape was `240 x 235`, wall time `386.3s`; final prob range `0.019048` to `0.977209`, mean `0.4426`.
- **Queue decision:** Reordered the submission monitor to prioritize the genuinely new real SED candidates at the UTC reset. New order after already-scored v500-v504: v510 version 2 (`0.05`), v511 (`0.10`), v512 (`0.02`), then older v505-v509 postprocess candidates, then old v376+ variants. This avoids spending the next daily cap entirely on older micro-sweeps while real SED candidates wait another day.
- **Validation:** `py_compile` passed for the reordered queue monitor. Next step: restart the monitor before UTC reset and verify it submits v510/v511/v512 first when quota returns.
