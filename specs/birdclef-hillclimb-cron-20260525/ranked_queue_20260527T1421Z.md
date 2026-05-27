# Ranked Queue — BirdCLEF hill-climb cron — 2026-05-27 14:21 UTC

## Live state
- Best public LB: `0.949` (v616/v617/v620/v621/v622/v623 tied; v625 `0.948`, v624 `0.943`).
- UTC slots: `0/5` used at 14:17 UTC; mid-day, ~9.7h to reset.
- Active jobs: no local/trainer BirdCLEF jobs at start; trainer GPUs idle.
- Submission decision this run: **no submission**. No verifier-grade nonduplicate candidate is ready, and the new PANNs no-train data point is not a hidden-test package.

## Run result summary
- Trained `soundscape-sequence-panns-cnn14-notrain-r2-nofile-reg-losite-ep24-20260527`.
- Context row AUC `0.601305` vs row-only `0.563916` (`+0.037389`).
- File-MIL `0.616149` vs row-only `0.638104` (`-0.021956`).
- Versus DyMN10 focused no-train: row `+0.047660`, file-MIL `-0.022129`.

## Ranked next queue

1. **True hidden-test package for best AudioSet sequence signal (PANNs/Cnn14 all-class, optionally PANNs no-train as capped slice)** — ACCEPTED / highest information value
   - Why: PANNs all-class is the strongest local sequence signal so far (`0.647816` row / `0.670723` file-MIL); PANNs no-train adds row-level no-train lift but weaker file-MIL. OOF proxy wrappers are not promotion evidence.
   - Gate: package encoder/head/context inference over hidden rows, schema/runtime/dedup, finite/nonconstant competition CSV, compare with v616 and tied sources before any slot.

2. **Real no-call/acoustic-context protocol** — ACCEPTED / high diversity
   - Why: current soundscape labels are positive-heavy; no-call/background behavior remains under-measured.
   - Gate: trusted negative/any-call target first; no suppression package until positive-row recall survives.

3. **S08/S13 guarded AudioSet refinement** — NEEDS_REVISION
   - Why: PANNs no-train regresses S08/S13; PANNs all-class slightly regresses S08 and barely helps S23.
   - Gate: one controlled site-guard ablation only; no weight sweep of failed OOF sidecars.

4. **Broader OOF negative/no-call SED student** — ACCEPTED fallback
   - Why: repo-owned fast path for no-call signal after target protocol is improved.

5. **Late-day clean public/source slot fill** — CONDITIONAL
   - Why: daily slots should not expire unused, but only inside `<3h` to reset if no verifier-grade package exists.
   - Gate: source-clean, nonduplicate, schema/runtime-safe, not public-output-only/static/fallback.

## Critic / verifier notes
- Critic: this was the right distinct data point after PANNs all-class; row-level no-train lift is promising, but file-level behavior is worse and S08/S13 are red flags.
- Verifier: artifacts finite/nonconstant; TorchScript smoke passed; not competition-format; no rules issue observed; no external slot used.

## Artifacts
- Ledger: `artifacts/model_data_point_ledger/20260527T1421Z_panns_cnn14_notrain_sequence.md`
- Model root: `artifacts/soundscape_sequence_mining/soundscape-sequence-panns-cnn14-notrain-r2-nofile-reg-losite-ep24-20260527/`
- Config: `configs/birdclef/soundscape_sequence_panns_cnn14_notrain_r2_nofile_reg_losite_ep24_20260527.json`
