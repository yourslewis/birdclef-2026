# Ranked queue — 2026-05-29 10:20 UTC

## Live status
- Public LB best: `0.949`; v616 remains the tied baseline to beat; v634 is the only latest late-fill tie.
- Latest completed submissions: v631 `0.926`, v632 `0.940`, v633 `0.946`, v634 `0.949`, v635 `0.941`.
- 2026-05-29 UTC slots used: `0/5`; about `13.6h` to reset at status check, so early-day policy is active.
- Active jobs before run: no local/trainer BirdCLEF jobs; trainer GPUs free.

## Current run result
- Added `freeze_encoder` support to `scripts/birdclef_soundscape_native_losite_train.py` and trained `soundscape-native-b0-soft1279init-headonly-losite-allcls-ep4-20260529`.
  - Data: official train_soundscapes, 1,478 windows / 66 files / 9 sites / 234 labels.
  - Init: soft1279 OOF-teacher TorchScript checkpoint with head loaded; encoder frozen; 75,114 trainable head parameters.
  - Leave-site result: row AUC `0.537303`, no-train `0.557939`, non-Aves `0.512910`, file-MIL `0.465716`; pooled row/no-train `0.280354/0.224432`.
  - Delta vs original head-loaded soft1279 native all-class: row `-0.063057`, file-MIL `-0.140089`.
- Packaged/evaluated through train_soundscape inference as v616 proxy sidecar.
  - Matched 240/240 proxy rows; finite and nonconstant 240x234.
  - Best non-control recipe `w0.01`: local AUC `0.990502` / 42 valid, lift vs v616 `-0.002979`, lift vs anchor `+0.000111`.
  - This is `-0.005043` AUC worse than the original head-loaded `w0.16` sidecar grid (`0.995545`).
  - `submit_approved=false`; no submission.

## Ranked next actions
1. **Soft1279 head-loaded constrained selector / movement diagnosis** — head-only failed, so the useful clue likely comes from encoder movement. Diagnose per-class/site displacement and consider capped class movement from the original head-loaded sidecar, not more frozen-head training. Expected LB potential: medium; evidence value: high.
2. **Hand/stricter no-call negative audit** — farneg10 gate remains directional but too tiny; verify/broaden true background negatives before any suppression slot. Expected LB potential: medium; evidence value: high.
3. **Sonotype/site-pair specialist validation** — scoped non-Aves/no-train model exposed site/sonotype inversion; isolate problematic site pairs or anti-site shortcuts. Expected LB potential: medium-low; data-point value: high.
4. **Train-soundscape sequence/file/site mining with distinct encoder/objective** — PANNs localmax remains the best sequence clue; direct OOF sidecars below v616. Try only if it adds a genuinely different hidden-safe wrapper or file objective. Expected LB potential: medium-low; data-point value: medium.
5. **Late-day public/source slot fill** — only after `<3h` to reset if no verifier-grade candidate exists; source must pass schema/runtime/dedup guards and cap recount.

## Critic / verifier decision
- Critic: freezing the encoder answered the constrained-movement question but is not promising; it harmed both LOSO and v616-local sidecar transfer.
- Verifier: package is finite, nonconstant, aligned, and reproducible, but every non-control recipe is below v616. No slot.

## Artifacts
- Ledger: `artifacts/model_data_point_ledger/20260529T1020Z_soundscape_native_soft1279init_headonly.md`
- Training metrics: `artifacts/soundscape_native_losite/soundscape-native-b0-soft1279init-headonly-losite-allcls-ep4-20260529/metrics.json`
- Package audit summary: `artifacts/sed_soundscape_packaging_audit/20260529T1020Z_soft1279init_headonly_package/audit_summary.json`
- Canonical performance table updated: `artifacts/model_data_point_ledger/performance_table.md` and `.jsonl`
