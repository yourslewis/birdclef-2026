# Ranked Queue — BirdCLEF hill-climb cron — 2026-05-27 12:25 UTC

## Live state
- Best public LB: `0.949` (v616/v617/v620/v621/v622/v623 tied; v625 `0.948`, v624 `0.943`).
- UTC slots: `0/5` used at 12:16 UTC; early/mid-day, ~11.7h to reset.
- Active jobs: no local/trainer BirdCLEF jobs; trainer GPUs idle before this run.
- Submission decision this run: **no submission**. PANNs all-class sequence is a strong local data point but the proxy sidecar loses to v616 (`-0.002538`) and is not a hidden-test package.

## Run result summary
- Trained `soundscape-sequence-panns-cnn14-allcls-r2-nofile-reg-losite-ep18-20260527`.
- Context row AUC `0.647816` vs row-only `0.588246` (`+0.059571`); file-MIL `0.670723` vs `0.651697` (`+0.019026`).
- Secondary: no-train `0.641399`, non-Aves `0.679851`.
- PANNs beats DyMN10 all-class r2 on leave-site row/file-MIL, but the low-weight v616 proxy sidecar still fails promotion.

## Ranked next queue

1. **True hidden-test package for best AudioSet sequence signal (PANNs/Cnn14, optionally DyMN10 fallback)** — ACCEPTED / highest info value
   - Why: PANNs all-class sequence is the best local train_soundscape sequence data point so far (`0.647816` row / `0.670723` file-MIL), but OOF proxy wrapping is structurally weak. Need a real hidden-safe inference path over test soundscapes before judging LB value.
   - Gate: package encoder/head/context features, schema/runtime/dedup, finite/nonconstant competition CSV, compare with v616 and previous tied sources before any slot.

2. **No-call/acoustic-context branch using AudioSet logits/embeddings** — ACCEPTED / high diversity
   - Why: current local data is positive-label heavy; no-call/background behavior remains under-measured and could differ from rank-space v616.
   - Gate: trusted negative protocol or calibrated any-call/no-target target; no submission until suppression does not destroy positive rows.

3. **S08/S23 guarded PANNs all-class refinement** — NEEDS_REVISION
   - Why: PANNs improved 6/7 folds but S08 regressed slightly and S23 barely improved. It may need site/worst-fold weighting before packaging.
   - Gate: one controlled ablation only; do not weight-sweep OOF sidecars.

4. **Broader OOF negative/no-call SED student** — ACCEPTED fallback
   - Why: repo-owned fast path for no-call signal. Earlier negative cache was too narrow; use only after target protocol improves.

5. **Late-day clean public/source slot fill** — CONDITIONAL
   - Why: daily slots should not expire unused, but only inside `<3h` to reset if no verifier-grade package exists.
   - Gate: source-clean, nonduplicate, schema/runtime-safe, not public-output-only/static/fallback.

## Critic / verifier notes
- Critic: direct leave-site OOF proxy wrappers are now rejection evidence, not promotion evidence. New useful action must change integration/packaging or target protocol.
- Verifier: PANNs artifacts are finite/nonconstant and auditable, but not submission-grade. No rules issue observed; no external slot used.

## Artifacts
- Ledger: `artifacts/model_data_point_ledger/20260527T1220Z_panns_cnn14_allclass_sequence.md`
- Model root: `artifacts/soundscape_sequence_mining/soundscape-sequence-panns-cnn14-allcls-r2-nofile-reg-losite-ep18-20260527/`
- Sidecar audit: `artifacts/soundscape_allclass_sidecar_audit/20260527T1220Z_panns_allclass_sequence/`
