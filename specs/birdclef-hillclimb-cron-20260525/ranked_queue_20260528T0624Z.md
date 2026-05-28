# BirdCLEF Hill-Climb Ranked Queue — 20260528T0624Z

## Live status
- Best public LB remains **0.949** (v616/v621/v622/v623 tied best; v629 is latest near-best at 0.946, v626-v630 all below best).
- 2026-05-28 UTC slots: **0/5 used** at run start and post-run check pending; early UTC-day policy applies (~17.7h to reset at start).
- Active BirdCLEF jobs before launch: none local/trainer; trainer GPU 1 was free, unrelated LRM job present on GPU 0.
- New work this run: trained/evaluated `PANNs/Cnn14 soundscape-positive localmax` sequence data point and ran 75→234 v616 proxy sidecar audit.

## New result summary
- Model row AUC: **0.642375** LOSO / 7 folds; file-MIL **0.662504**; no-train **0.592102**; non-Aves **0.667663**.
- Deltas: vs PANNs all-class localmax row **+0.000874** but file-MIL **-0.019249**; vs native B0 soundscape-positive row **-0.015790**, file-MIL **-0.013879**.
- Best sidecar recipe: `seq_context_w01`, local proxy AUC **0.991188** / 42 valid; lift vs v616 **-0.002292**; lift vs anchor **+0.000798**.
- Decision: **no submission**. Comparison-grade data point only; sidecar fails v616 promotion.

## Ranked queue after this run
1. **Hidden-safe package/wrapper for PANNs/native soundscape clues** — highest information value, but avoid replaying proxy-only sidecars. Need a test-row-operable wrapper that changes hidden behavior, using PANNs localmax all-class and native soundpos as clues.
2. **No-call/background protocol audit** — under-mined and likely more important than another soundscape-positive target ablation. Build trusted background/no-call coverage and calibration diagnostics before training suppressors.
3. **Site-robust native/PANNs hybrid target revision** — native soundpos remains strongest 75-class LOSO, PANNs all-class localmax remains best sidecar lift; try only if it changes site robustness or packageability.
4. **File-MIL optimized PANNs/localmax sequence branch** — PANNs all-class localmax had best file-MIL/sidecar among PANNs variants; a true file-MIL objective may be more useful than soundpos target filtering.
5. **Late-day public/source slot fill** — only inside <3h to reset, after schema/runtime/dedup/source guards. Yesterday's five public fills scored 0.899/0.928/0.940/0.946/0.917, so this is last resort.

## Critic / red-team decision
- The new model does **not** justify a slot: row AUC is not best-in-family, file-MIL regressed, and the v616 proxy sidecar is worse than both PANNs all-class localmax and native soundpos sidecars.
- The target-scope ablation is useful as a measured data point: soundscape-positive filtering helps PANNs row AUC only trivially and hurts file-MIL/sidecar, so future PANNs work should prefer all-class/localmax or file-MIL objectives.

## Verifier decision
- Model guards passed: finite/nonconstant final predictions, TorchScript context head exported, OOF predictions saved.
- Sidecar guards passed: finite `240x234`, 156/240 proxy rows matched, 234 nonconstant columns.
- Submission gate failed: best lift vs v616 is **-0.002292**, so **reject slot candidate**.

## Artifacts
- Model ledger: `artifacts/model_data_point_ledger/20260528T0624Z_panns_cnn14_soundpos_localmax_sequence.md`
- Sidecar ledger: `artifacts/model_data_point_ledger/20260528T0624Z_panns_soundpos_localmax_sidecar_audit.md`
- Metrics: `artifacts/soundscape_sequence_mining/soundscape-sequence-panns-cnn14-soundpos-localmax-losite-ep20-20260528/metrics.json`
- Audit: `artifacts/soundscape_sequence_sidecar_audit/20260528T0624Z_panns_soundpos_localmax/audit_summary.json`
- Logs: `logs/soundscape_sequence_panns_cnn14_soundpos_localmax_losite_ep20_20260528.log`, `logs/soundscape_sequence_panns_cnn14_soundpos_localmax_sidecar_audit_20260528T0624Z.log`
