# BirdCLEF Hill-Climb Ranked Queue — 20260528T1022Z

## Live status
- Best public LB remains **0.949** (v616/v621/v622/v623 tied best; latest v626-v630 scored `0.899/0.928/0.940/0.946/0.917`).
- 2026-05-28 UTC slots: **0/5 used** at live check; ~13.7h to reset, so early/mid-day verifier policy applies.
- Active BirdCLEF jobs before launch: none local/trainer. GPU0 was occupied by an unrelated LRM job; GPU1 was free and used via `CUDA_VISIBLE_DEVICES=1`.

## New result summary
- Built strict OOF-negative/no-call mask `<=0.010`: 8,787 clean negative cells, 668/1,279 rows, 79/234 classes, `0` false-negative cells against OOF truth.
- Trained matched full-row soft control `sed-b0-oofteacher-b0v26-nfnetv29-soft-1279-ep4-20260528`: macro AUC **0.935542** / 128 valid, non-Aves **0.974510**, best val loss **0.313012**.
- Trained strict-negative branch `sed-b0-oofteacher-b0v26-nfnetv29-soft-strictneg0010-w005-1279-ep4-20260528`: macro AUC **0.930294** / 128 valid, non-Aves **0.956863**, best val loss **0.313245**.
- Delta: full-row soft control improved over prior 1,024-row soft control by **+0.024475** macro AUC; strict negative regularization regressed vs matched soft by **-0.005248** macro AUC and **-0.017647** non-Aves AUC.

## Comparable top-5 for OOF-teacher SED validation
1. `B0 OOF-teacher soft 1279` — macro **0.935542**, non-Aves **0.974510**; new best control.
2. `B0 OOF-teacher strict-neg0010 w0.05 1279` — macro **0.930294**, non-Aves **0.956863**; negative aux still worse than matched soft.
3. `B0 OOF-teacher soft 1024` — macro **0.911067**, non-Aves **0.877451**.
4. `B0 OOF-teacher broadneg003 w0.01 1024` — macro **0.908278**, non-Aves **0.872549**.
5. `B0 20s localmax 512` — macro **0.672996** / 72 valid; decorrelation point, not comparable strength.

## Ranked queue after this run
1. **Package/evaluate the 1,279-row soft OOF-teacher SED as a hidden-safe raw branch** — highest new OOF-teacher data point; needs competition-format inference, row/schema guard, and v616/proxy audit before any slot.
2. **No-call/background protocol audit v2** — keep the strict mask artifact, but do not train unchanged aux-negative; next audit should define actual no-call labels or suppression gates, not only masked BCE.
3. **Hidden-safe wrapper for PANNs/localmax or native soundpos clues** — direct OOF sidecars remain below v616, but packageable raw hidden-test inference could still provide diversity.
4. **Site-robust native/PANNs hybrid target revision** — only if it changes site robustness/packageability, not another minor localmax/file-MIL tweak.
5. **Late-day source slot fill** — only inside <3h to reset after source/runtime/dedup guards; yesterday's public fills were weak, so keep last.

## Critic / red-team decision
- **Proceed with soft 1,279-row package/audit, reject strict negative unchanged.** The no-call mask is clean, but the auxiliary loss moved validation in the wrong direction against a matched control. The useful discovery is that more OOF-teacher rows materially help; negative regularization still lacks a trusted no-call target.

## Verifier decision
- Training used official train-audio + OOF teacher cache only; no hidden/test labels or disallowed data.
- Both models exported TorchScript/ONNX and produced finite validation probabilities; no competition submission artifact was produced.
- Submission gate: **failed/not applicable** — neither branch is yet a hidden-safe competition-format output; no slots used.

## Artifacts
- Model ledgers: `artifacts/model_data_point_ledger/20260528T1022Z_sed_b0_oofteacher_soft1279.md`, `artifacts/model_data_point_ledger/20260528T1022Z_sed_b0_oofteacher_strictneg0010.md`
- Configs: `configs/birdclef/sed_b0_oofteacher_b0v26_nfnetv29_soft_1279_ep4_20260528.json`, `configs/birdclef/sed_b0_oofteacher_b0v26_nfnetv29_soft_strictneg0010_w005_1279_ep4_20260528.json`
- Metrics: `artifacts/sed_oof_teacher_students/sed-b0-oofteacher-b0v26-nfnetv29-soft-1279-ep4-20260528/metrics.json`, `artifacts/sed_oof_teacher_students/sed-b0-oofteacher-b0v26-nfnetv29-soft-strictneg0010-w005-1279-ep4-20260528/metrics.json`
- Mask summary: `artifacts/pseudolabels/oof-negative-cache/b0v26_nfnetv29_teacher_neg0010_cap64_20260528.summary.json`
- Logs: `logs/sed_b0_oofteacher_b0v26_nfnetv29_soft_1279_ep4_20260528.log`, `logs/sed_b0_oofteacher_b0v26_nfnetv29_soft_strictneg0010_w005_1279_ep4_20260528.log`
