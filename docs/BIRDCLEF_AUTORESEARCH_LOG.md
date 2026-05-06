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
- **Status at launch:** process running; no final metrics yet. Next run should tail the log, collect metrics, rsync artifacts if needed, and decide whether to launch the 10s/160-mel sibling.
