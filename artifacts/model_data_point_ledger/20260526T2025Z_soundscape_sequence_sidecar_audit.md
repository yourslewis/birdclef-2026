# Model evaluation — train_soundscape sequence 72→234 sidecar audit — 2026-05-26 20:25 UTC

## Summary
- **Experiment id:** `soundscape-sequence-sidecar-audit-20260526T2025Z`
- **Branch family:** train_soundscapes sequence/file/site mining; 72-label non-Aves/no-train sequence heads wrapped into 234-class v616 proxy sidecars.
- **Purpose:** test whether the two best sequence artifacts (`10:20` context MLP and `18:20` r2/no-file regularized context) are useful as low-weight 234-class sidecars against the tied v616 baseline.
- **Evidence level:** comparison-grade local/proxy audit only; not a hidden-test package.

## Data / target contract
- Base matrix: v616 train-soundscape proxy anchor/final, `240` rows × `234` classes.
- Sequence sidecar source: official `train_soundscapes` leave-site OOF predictions, `72` non-Aves/no-train labels.
- Wrapper: replace only the 72 scoped class columns on matched proxy rows; all other classes/rows remain anchor-filled.
- Matched proxy rows: `156 / 240`; unmatched proxy rows: `84 / 240` (folds without usable leave-site prediction, anchor-filled).
- Local labels available for audit: `190` matched rows / `42` valid AUC classes.

## Models / recipes evaluated
- Context MLP sidecar: `soundscape-sequence-dymn10-context-losite-ep16-20260526`, recipes `1%`, `2%`, `4%`.
- R2 no-file regularized sidecar: `soundscape-sequence-dymn10-r2-nofile-reg-losite-ep20-20260526`, recipes `1%`, `2%`, `4%`.
- Combo recipes: `2% context + 1% r2` and `1% context + 2% r2`.
- Audit script: `scripts/birdclef_soundscape_sequence_sidecar_audit.py`, which calls `scripts/birdclef_ensemble_strategy_audit.py` with `20` bootstrap iterations for this cron smoke.

## Metrics
Controls:
- `anchor_only`: local macro AUC `0.990390507` over `42` classes.
- `v616_baseline`: local macro AUC `0.993480668` over `42` classes; public LB `0.949` tied best.

Best audited recipe:
- `seq_context02_r201` (`97%` anchor + `2%` context + `1%` r2): local macro AUC `0.991293583`.
- Lift vs anchor: `+0.000903076`.
- Lift vs v616 baseline: `-0.002187085`.
- Rank correlation vs v616: `0.999514417`; MAE vs v616 `0.006936131`.

Best single-model sequence sidecars:
- `seq_context_w01`: local macro AUC `0.991279099`; lift vs v616 `-0.002201568`; rank corr vs v616 `0.999676811`.
- `seq_r2_w01`: local macro AUC `0.991031704`; lift vs v616 `-0.002448964`; rank corr vs v616 `0.999676840`.

## Critic / verifier decision
**Reject as slot candidate; keep as a useful wrapper data point.**

The sequence sidecars beat the raw anchor slightly, but they are clearly worse than the already-submitted v616 recipe on the same local proxy. They also only cover `156/240` proxy rows, are not packaged for hidden-test inference, and are analysis-only leave-site OOF artifacts. No Kaggle submission is approved.

Strategic implication: sequence/file/site mining remains useful for understanding the non-Aves/no-train landscape, but the current 72-label wrapper should not spend a leaderboard slot. Next useful work is either a true hidden-safe 234-class DyMN10/AudioSet sidecar package with multi-site validation, or a late-day guarded slot-fill review if no package is ready near reset.

## Artifacts
- Root: `artifacts/soundscape_sequence_sidecar_audit/20260526T2025Z/`
- Build report: `artifacts/soundscape_sequence_sidecar_audit/20260526T2025Z/sidecar_build_report.json`
- Audit summary: `artifacts/soundscape_sequence_sidecar_audit/20260526T2025Z/audit_summary.json`
- Full audit: `artifacts/soundscape_sequence_sidecar_audit/20260526T2025Z/audit/ensemble_strategy_audit.json`
- Sidecars: `artifacts/soundscape_sequence_sidecar_audit/20260526T2025Z/sidecars/`
