# Model Data Point — OOF-teacher B0 1024 ep4 + Broad Negative/No-Call Control

Timestamp: 2026-05-26 02:20 UTC

## Summary

Trained the next repo-owned SED data point after the PANNs/Cnn14 branch: a B0 soft OOF-teacher student with a newly broadened negative/no-call mask. Also trained a matched soft-only 1,024-row / 4-epoch control so the critic could tell whether the lift came from the negative auxiliary term or from more rows/epochs.

## Ledger

- **Model family:** EfficientNet-B0 SED-style clip/frame model, q3/cap80 external-pretrain encoder init, 5s/160-mel input.
- **Init/source:** `artifacts/external_pretrain/xc-b0-q3-cap80-external-pretrain-balanced-ep12/model_torchscript.pt`; loaded 352 encoder keys, skipped 2 head keys.
- **Train rows:** 1,024 OOF-teacher-backed official train-audio files; 819 train / 205 validation random split.
- **Labels/targets:** all 234 submission classes; soft targets from `artifacts/pseudolabels/oof-teacher-cache/b0v26_nfnetv29_w090010_intersection_cache.npz`; truth from same OOF cache for validation metrics.
- **Input window/features:** 5-second waveform, 32 kHz, 160 mel bins, hop 512.
- **Augmentations:** none; BCE soft-label training.
- **Loss:** soft-only BCE control; broad-neg branch uses BCE plus `0.01` masked-negative auxiliary BCE-to-zero loss.
- **Epochs/runtime:** 4 epochs. Soft-only runtime 26.059s CUDA; broad-neg runtime 32.685s CUDA.
- **CV/proxy metric:** soft-only macro AUC `0.911067` over 122 valid classes, best val loss `0.318399`; broad-neg macro AUC `0.908278` over 122 valid classes, best val loss `0.318376`.
- **Negative/no-call mask:** built with `scripts/birdclef_oof_teacher_negative_mask.py` from the OOF teacher cache using threshold `0.03` and row cap `64`. Capped mask has 47,343 negative cells, 1,259/1,279 row coverage, 230/234 class coverage, and 0 false-negative cells. In the selected 1,024 training rows it covers all rows with 37,993 negative cells.
- **Prediction artifacts:**
  - Soft-only: `artifacts/sed_oof_teacher_students/sed-b0-oofteacher-b0v26-nfnetv29-soft-1024-ep4-20260526/holdout_predictions.npz`
  - Broad-neg: `artifacts/sed_oof_teacher_students/sed-b0-oofteacher-b0v26-nfnetv29-soft-broadneg003-w001-1024-ep4-20260526/holdout_predictions.npz`
- **Export/runtime status:** both branches exported TorchScript (~15.389 MB) and ONNX (0.56 MB); ONNX checker passed; CPU TorchScript inference smoke on 4 files passed with all 234 probability columns nonconstant. Soft-only CPU smoke `0.199s` total / `0.050s` per file; broad-neg `0.185s` total / `0.046s` per file.
- **Correlation/blend vs anchor/v616:** not run yet. These are not row-aligned hidden/test sidecar outputs yet.
- **Diversity value:** medium/high. Still B0 SED, but the soft OOF-teacher 1024 control is materially stronger than the prior 512-row soft smoke (`0.819`), and the broad mask resolves the prior sparse-negative blocker.
- **Critic decision:** keep the soft-only control as the better promotion candidate. Do not scale broad-neg aux unchanged because it slightly underperforms the matched control.
- **Verifier decision:** no-slot training is rule-safe; exports pass; not competition-format; no submission approved.

## Artifact paths

- Script: `scripts/birdclef_oof_teacher_negative_mask.py`
- Soft-only config: `configs/birdclef/sed_b0_oofteacher_b0v26_nfnetv29_soft_1024_ep4_20260526.json`
- Broad-neg config: `configs/birdclef/sed_b0_oofteacher_b0v26_nfnetv29_soft_broadneg003_w001_1024_ep4_20260526.json`
- Mask summary: `artifacts/pseudolabels/oof-negative-cache/b0v26_nfnetv29_teacher_neg003_cap64_20260526.summary.json`
- Soft-only metrics: `artifacts/sed_oof_teacher_students/sed-b0-oofteacher-b0v26-nfnetv29-soft-1024-ep4-20260526/metrics.json`
- Broad-neg metrics: `artifacts/sed_oof_teacher_students/sed-b0-oofteacher-b0v26-nfnetv29-soft-broadneg003-w001-1024-ep4-20260526/metrics.json`
- Logs: `logs/sed_b0_oofteacher_b0v26_nfnetv29_soft_1024_ep4_20260526.log`, `logs/sed_b0_oofteacher_b0v26_nfnetv29_soft_broadneg003_w001_1024_ep4_20260526.log`
