# Model Data Point — Strict OOF-Negative No-Call B0 SED Student

Timestamp: 2026-05-28 10:22 UTC

## Summary

Built a stricter OOF-only negative/no-call mask and trained a matched full-row B0 OOF-teacher SED student with aux-negative regularization. This tests whether the earlier broad negative mask failed because it was too broad/ambiguous.

## Negative/no-call mask

- **Source:** `artifacts/pseudolabels/oof-teacher-cache/b0v26_nfnetv29_w090010_intersection_cache.npz`; OOF-only teacher predictions, no hidden/test labels.
- **Threshold/cap:** teacher probability `<=0.010`, `max_neg_per_row=64`.
- **Coverage:** 8,787 negative cells, 668/1,279 rows (`52.23%`), 79/234 classes (`33.76%`).
- **Precision against OOF truth:** 8,787 true-negative cells, `0` false-negative cells; mask false-negative rate `0.0`.

## Ledger

- **Model family:** EfficientNet-B0 SED-style clip/frame model, q3/cap80 external-pretrain encoder init, 5s/160-mel input.
- **Training data:** 1,279 OOF-teacher-backed official train-audio files; 1,023 train / 256 validation random split, seed 42.
- **Labels/targets:** all 234 taxonomy labels; soft OOF-teacher targets plus strict OOF-negative auxiliary BCE-to-zero mask.
- **Loss/epochs:** BCE + `0.05` masked negative auxiliary loss, 4 epochs.
- **Primary metric:** macro ROC-AUC `0.930294` over 128 valid classes.
- **Secondary metrics:** non-Aves AUC `0.956863` over 4 valid classes; no-train AUC N/A (`0` valid classes in train-audio OOF split); no-call AUC N/A (mask precision available, but no true no-call AUC labels); negative-cell mean prediction `0.099358`; positive-cell mean prediction `0.285513`; best val loss `0.313245`.
- **Baseline/delta:** vs matched 1,279-row soft control `-0.005248` macro AUC and `-0.017647` non-Aves AUC; vs prior broad-neg 1,024-row branch `+0.022016` macro AUC, mostly from full-row data/control effect.
- **Export/runtime status:** TorchScript and ONNX exported on trainer GPU1; runtime `37.442s`; model binaries retained on trainer.
- **Decision:** **reject unchanged as no-call regularizer**. The stricter mask is clean and useful as a protocol artifact, but aux-negative training still underperforms the matched soft control.

## Artifacts

- Config: `configs/birdclef/sed_b0_oofteacher_b0v26_nfnetv29_soft_strictneg0010_w005_1279_ep4_20260528.json`
- Mask summary: `artifacts/pseudolabels/oof-negative-cache/b0v26_nfnetv29_teacher_neg0010_cap64_20260528.summary.json`
- Metrics: `artifacts/sed_oof_teacher_students/sed-b0-oofteacher-b0v26-nfnetv29-soft-strictneg0010-w005-1279-ep4-20260528/metrics.json`
- Trainer model/export dir: `~/birdclef-2026/artifacts/sed_oof_teacher_students/sed-b0-oofteacher-b0v26-nfnetv29-soft-strictneg0010-w005-1279-ep4-20260528/`
- Logs: `logs/oof_teacher_neg0010_cap64_20260528.log`, `logs/sed_b0_oofteacher_b0v26_nfnetv29_soft_strictneg0010_w005_1279_ep4_20260528.log`
