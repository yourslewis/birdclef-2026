# BirdCLEF Hill-Climb Ranked Queue — 20260528T0422Z

## Live status
- Best public LB remains **0.949** (v616/v621/v622/v623 tied baseline before private).
- 2026-05-28 UTC slots: **0/5 used** at run start; early UTC-day policy applies.
- Active jobs before launch: none local/trainer BirdCLEF.
- New work this run: trained/evaluated native B0 soundscape-positive target redesign and v616 proxy sidecar audit.

## Ranked queue after this run
1. **PANNs/native hidden-test package path** — highest information value. Current proxy sidecars still lose to v616, but PANNs all-class/localmax and native soundpos are the strongest measured train_soundscape clues. Next exact action: build hidden-safe inference package or wrapper that operates on test rows, not only OOF proxy rows.
2. **No-call/background protocol audit** — still under-mined. Before training more suppressors, build a trusted negative/background set and report coverage/calibration, because unmatched proxy rows are not reliable negatives.
3. **Site-robust soundscape-positive/native revision** — native soundpos row/file metrics are strong, but pooled AUC and sidecar-v616 are weak. Try site-adversarial/site-balanced revision or 234-class wrapper only if it changes integration, not another direct sidecar replay.
4. **Fused/PANNs localmax sequence package or 20s local-context branch** — localmax improved file-MIL and anchor lift but not v616; useful as package/eval candidate if hidden-safe integration differs.
5. **Late-day public/source slot fill** — only inside <3h to reset and only after source/runtime/dedup/schema guards; yesterday's v626-v630 all scored below best.

## Critic / red-team decision
- Do **not** spend an early-day slot: best native-soundpos sidecar lift vs v616 is -0.001930, below promotion.
- The model is a real data point, not submission-grade: 75-class target scope, proxy row mismatch, and poor pooled diagnostics mean local LOSO AUC is comparison-grade only.

## Verifier decision
- Model OOF/export guards passed: finite/nonconstant predictions; TorchScript/ONNX smoke checked.
- Sidecar guard passed shape/finite/nonconstant, but failed v616 delta. **Submission rejected.**

## Artifacts
- Model ledger: `artifacts/model_data_point_ledger/20260528T0422Z_soundscape_native_b0_soundpos.md`
- Sidecar ledger: `artifacts/model_data_point_ledger/20260528T0422Z_native_soundpos_sidecar_audit.md`
- Metrics: `artifacts/soundscape_native_losite/soundscape-native-b0-losite-soundpos-ep5-20260528/metrics.json`
- Audit: `artifacts/soundscape_sequence_sidecar_audit/20260528T0415Z_native_soundpos/audit_summary.json`
