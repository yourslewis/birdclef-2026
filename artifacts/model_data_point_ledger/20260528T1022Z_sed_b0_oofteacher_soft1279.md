# Model Data Point — Full OOF-teacher B0 SED Control (1,279 rows)

Timestamp: 2026-05-28 10:22 UTC

## Summary

Trained a full-row control for the no-call/background lane using the existing B0v26/NFNetv29 OOF-teacher cache. This is the matched control needed before judging stricter OOF-negative/no-call regularization on all 1,279 teacher-backed files.

## Ledger

- **Model family:** EfficientNet-B0 SED-style clip/frame model, q3/cap80 external-pretrain encoder init, 5s/160-mel input.
- **Init/source:** `artifacts/external_pretrain/xc-b0-q3-cap80-external-pretrain-balanced-ep12/model_torchscript.pt`; 352 encoder keys loaded, 2 head keys skipped.
- **Training data:** 1,279 OOF-teacher-backed official train-audio files; 1,023 train / 256 validation random split, seed 42.
- **Labels/targets:** all 234 taxonomy labels; soft targets from `artifacts/pseudolabels/oof-teacher-cache/b0v26_nfnetv29_w090010_intersection_cache.npz`; validation truth from the same OOF cache.
- **Loss/epochs:** soft-label BCE, 4 epochs, no aux-negative loss.
- **Primary metric:** macro ROC-AUC `0.935542` over 128 valid classes.
- **Secondary metrics:** non-Aves AUC `0.974510` over 4 valid classes; no-train AUC N/A (`0` valid classes in train-audio OOF split); no-call AUC N/A (no trusted no-call labels); negative-cell mean prediction `0.099097`; positive-cell mean prediction `0.281666`; best val loss `0.313012`.
- **Baseline/delta:** vs prior 1,024-row soft control `0.911067`, delta `+0.024475` macro AUC and val loss `-0.005387`; vs prior 1,024-row broad-neg branch `0.908278`, delta `+0.027264`.
- **Export/runtime status:** TorchScript and ONNX exported on trainer GPU1; runtime `59.773s`; `metrics.json`, training log, and best-checkpoint info synced locally; model binaries retained on trainer.
- **Decision:** **continue/package candidate only after hidden-safe inference/audit path exists**. Strongest OOF-teacher SED validation point so far, but not a competition-format hidden-test branch yet.

## Artifacts

- Config: `configs/birdclef/sed_b0_oofteacher_b0v26_nfnetv29_soft_1279_ep4_20260528.json`
- Metrics: `artifacts/sed_oof_teacher_students/sed-b0-oofteacher-b0v26-nfnetv29-soft-1279-ep4-20260528/metrics.json`
- Trainer model/export dir: `~/birdclef-2026/artifacts/sed_oof_teacher_students/sed-b0-oofteacher-b0v26-nfnetv29-soft-1279-ep4-20260528/`
- Log: `logs/sed_b0_oofteacher_b0v26_nfnetv29_soft_1279_ep4_20260528.log`
