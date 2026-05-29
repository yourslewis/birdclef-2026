# Ranked queue — 2026-05-29 08:25 UTC

## Live status
- Public LB best: `0.949`; v616 remains the tied baseline to beat; v634 also tied `0.949` from the latest late-fill source batch.
- Latest completed submissions: v631 `0.926`, v632 `0.940`, v633 `0.946`, v634 `0.949`, v635 `0.941`.
- 2026-05-29 UTC slots used: `0/5`; about `15.6h` to reset at status check, so early-day policy active.
- Active jobs before run: no local/trainer BirdCLEF jobs; trainer GPUs free.

## Current run result
- Trained `soundscape-native-b0-soft1279enc-nonaves-notrain-losite-ep5-20260529`, a scoped native EfficientNet-B0 specialist over 72 non-Aves/no-train labels.
  - Data: official train_soundscapes, 1,478 windows / 66 files / 9 sites / 72 labels.
  - Init: soft1279 OOF-teacher TorchScript encoder init; head skipped/reinitialized for the scoped 72-label target.
  - Leave-site result: row AUC `0.609793`, no-train `0.613437`, non-Aves `0.609793`, file-MIL `0.551016`; 6 complete folds, 3 skipped low-count/low-valid folds.
  - Pooled/site-mixed diagnostics are weak: pooled row AUC `0.293648`; pooled no-train AUC `0.151414`, indicating site/sonotype inversion risk.
- Packaged/evaluated as a 72-label anchor-preserved sidecar on the v616 local proxy.
  - Matched 240/240 proxy rows; finite/nonconstant 240x234 after preserving anchor values for labels outside the model scope.
  - Best non-control: raw anchor-preserved member AUC `0.993828` / 42 valid, lift vs v616 `+0.000347`, lift vs anchor `+0.003437`; low-weight rank blends stayed below v616.
  - `submit_approved=false`; no submission.
- Fixed `scripts/birdclef_single_sed_package_sidecar_audit.py` so scoped models preserve anchor columns for missing labels instead of zero-filling them.

## Ranked next actions
1. **Soft1279 head-loaded sidecar diagnosis / constrained class movement** — still the strongest current positive local clue (`w0.16` lift vs v616 `+0.002064`), but selectors failed site gates. Need diagnose which classes/sites drive the raw local lift and whether a stricter per-site capped movement exists. Expected LB potential: medium; evidence value: high.
2. **Hand/stricter no-call negative audit** — farneg10 gate and suppression clues are tiny but directionally positive; verify/broaden negatives before any no-call slot. Expected LB potential: medium; evidence value: high.
3. **Sonotype/site-pair specialist validation** — the new non-Aves/no-train scoped model shows mean leave-site row AUC but pooled no-train inversion; next should isolate sonotypes by site pairs (`S08/S15/S19/S23`) or apply group-DRO/anti-site shortcuts, not submit the raw sidecar. Expected LB potential: medium-low; data-point value: high.
4. **Train-soundscape sequence/file/site mining with stronger file objective or distinct encoder** — PANNs localmax remains the best sequence/file clue; direct sidecars are below v616. Expected LB potential: medium-low; data-point value: medium.
5. **Late-day public/source slot fill** — only after `<3h` to reset if no verifier-grade candidate exists; source must pass schema/runtime/dedup guards and cap recount.

## Critic / verifier decision
- Critic: the scoped native model is not a slot candidate. Mean LOSO looks acceptable only because strong frog sites offset sonotype inversions; pooled no-train AUC is poor and S23 sonotypes invert badly.
- Verifier: package sidecar is finite/nonconstant and auditable after anchor-preserve fix, but best lift vs v616 is only `+0.000347`, below the `+0.001` promotion gate; low-weight blends are below v616. No slot.

## Artifacts
- Training ledger: `artifacts/model_data_point_ledger/20260529T0825Z_soundscape_native_soft1279enc_nonaves_notrain.md`
- Training metrics: `artifacts/soundscape_native_losite/soundscape-native-b0-soft1279enc-nonaves-notrain-losite-ep5-20260529/metrics.json`
- Package audit summary: `artifacts/sed_soundscape_packaging_audit/20260529T0825Z_soft1279enc_nonaves_notrain_package/audit_anchorpreserve_summary.json`
- Canonical performance table updated: `artifacts/model_data_point_ledger/performance_table.md` and `.jsonl`
