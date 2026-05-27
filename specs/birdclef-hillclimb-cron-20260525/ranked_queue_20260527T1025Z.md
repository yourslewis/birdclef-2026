# Ranked queue refresh — 2026-05-27 10:25 UTC

## Live state
- Best public LB remains **0.949**. Latest scored submissions: v621/v622/v623 tied `0.949`, v625 `0.948`, v624 `0.943`; v616 remains a tied baseline to beat.
- 2026-05-27 UTC slots used: **0/5**; ~13.6h to reset. Early-day policy applies.
- No active local/trainer BirdCLEF jobs at start; trainer was idle after this bounded run.

## Work completed this run
1. **Soundscape-positive target-redesign sequence model**
   - Trained `soundscape-sequence-dymn10-soundpos-r2-filectx-losite-ep20-20260527` on official train_soundscapes.
   - Data/model: 1,478 windows / 66 files / 9 sites; 75 soundscape-positive labels; cached EfficientAT DyMN10 embeddings; radius-2 context MLP with file-mean context.
   - Leave-site result: context row AUC `0.518121` vs row-only `0.514539` (`+0.003582`); no-train AUC `0.528556`; non-Aves AUC `0.537355`; file-MIL `0.512164` vs row-only `0.609112` (`-0.096949`).
   - Fold deltas: S03 `-0.035092`, S08 `-0.097247`, S13 `-0.078824`, S15 `+0.090237`, S19 `+0.036365`, S22 `+0.078737`, S23 `+0.030896`.
2. **75→234 v616 proxy sidecar audit**
   - Wrapped the 75-label leave-site predictions into a 234-class v616 proxy sidecar; 156/240 rows matched, 84 anchor-filled; finite/nonconstant `240x234`.
   - Best recipe `seq_context_w01`: local AUC `0.990665` / 42 valid classes, lift vs anchor `+0.000274`, but lift vs v616 `-0.002816`; corr vs v616 `0.999675`.
   - Critic/verifier decision: **reject as slot candidate**; comparison-grade data point only.

## Queue after this run
1. **True hidden-test package/inference path for DyMN10/all-class only if integration changes** — OOF/proxy wrappers have failed three times vs v616. Continue only if building a real hidden-safe feature extraction package or a class-gated integration not equivalent to proxy sidecar replacement.
2. **Reformulated AudioSet/DyMN10 acoustic-context/no-call wrapper** — next distinct high-info branch: broad acoustic context/no-call features, multi-site validation, not direct 72/75/234 OOF replacement.
3. **S03/S08-aware objective change** — target redesign helped S15/S22/S23 but still regressed S03/S08/S13 and file-MIL; if continued, use worst-site/DRO-style loss or explicit per-site calibration rather than radius/file-context tweaks.
4. **Deeper soundscape-native adapter/compact SED variant** — only bounded adapter/last-block or no-call branch with leave-site/file gates; previous full B0 fine-tune was weak.
5. **Late UTC slot-fill scout** — if `<3h` to reset and no verifier-grade package exists, use highest-ranked clean public/source candidates that are nonduplicate and pass schema/runtime/dedup guards.

## Submission decision
**No submission now.** Early-day slots remain available, but the trained/evaluated candidate loses clearly to v616 in proxy audit and is not a hidden-test package. Submitting would be leaderboard probing.

## Updated artifacts
- Ledger: `artifacts/model_data_point_ledger/20260527T1018Z_soundscape_sequence_soundpos_target_redesign.md`
- Canonical table: `artifacts/model_data_point_ledger/performance_table.md`
- JSONL table: `artifacts/model_data_point_ledger/performance_table.jsonl`
- Model root: `artifacts/soundscape_sequence_mining/soundscape-sequence-dymn10-soundpos-r2-filectx-losite-ep20-20260527/`
- Sidecar audit: `artifacts/soundscape_sequence_sidecar_audit/20260527T1018Z_soundpos/`
- Config: `configs/birdclef/soundscape_sequence_dymn10_soundpos_r2_filectx_losite_ep20_20260527.json`
