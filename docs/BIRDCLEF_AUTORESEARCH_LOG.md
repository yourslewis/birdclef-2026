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
- **GPU status:** `yourslewis@192.168.0.23` SSH timed out in this run, so no long GPU job was launched. Next run should retry GPU and scale this scaffold there.
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
