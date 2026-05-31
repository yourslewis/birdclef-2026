# BirdCLEF hill-climb ranked queue — 2026-05-31 14:35 UTC

## Live state
- Public best: `0.950`, tied by v644 (`yaroslavkholmirzayev/0950-replay`) and v647 (`ryutoyoda/birdclef-2026-exp013-eos8-sidecar`). Treat as one EoS8/PowerOptimization family, not two independent wins.
- UTC slots used today at live check: `0/5`; ~9.7h to reset. This is still early/mid-day, so verifier-grade/high-info only.
- Active BirdCLEF jobs: none local; none on trainer. Unrelated trainer/user jobs present only.
- This run trained/evaluated a source-winner confidence/meta data point and did **not** submit: no hidden-test package/source fork was built, and early-day slot gate rejects proxy-only CSVs.

## New evidence from this run
- v644/v647 source SED intermediate stream is a strong local signal on the v616 train-soundscape proxy: local macro AUC `0.995976` / 42 valid, lift vs v616 `+0.002495`, non-Aves AUC `0.996866`, top-5 recall `0.994737`.
- Best no-slot grid: `0.20*v616 + 0.80*source_sed` rank blend, local AUC `0.996059`, lift vs v616 `+0.002578`, site/file bootstrap q05 `+0.000450/+0.000083` (20-boot smoke), rank corr vs v616 `0.857333`.
- Trainable leave-site logistic meta over v616/proto/sed/rank features failed: OOF AUC `0.990463`, delta vs v616 `-0.003018`; reject this meta formulation.
- Critic: source SED local lift is the best new clue today, but SED/v616-family local lifts have over-transferred before; high displacement and missing hidden package block a slot.

## Ranked queue
1. **Build private verifier / source fork for EoS8 SED-vs-PowerOpt weights** — `ACCEPTED`, highest expected information value. Start from v644/v647 source; preserve hidden-test audio path; expose a small config grid over outer SED/PowerOpt rank weights (e.g. current EoS8, source_sed-heavy 0.24/0.40/0.65/0.80 equivalents) and verify runtime/output. Promotion requires non-static hidden-safe source and no duplicate final.
2. **PowerOptimization component introspection** — `ACCEPTED`. Extract/replicate the Karnakbayev PowerOptimization/TaxRank branch on train rows if possible, then test SED/Proto/PowerOpt blend movement under leave-site/file gates. Goal: determine whether public-best `0.950` came from hidden PowerOpt behavior while local proxy prefers SED.
3. **Source-winner confidence/no-call sidecar using SED confidence** — `NEEDS_REVISION`. The source SED top-5 recall is strong; convert into bounded class/site/no-call confidence controls rather than a high-displacement full-rank replacement.
4. **Hand/teacher-audited no-call negatives** — `NEEDS_REVISION`. Only revisit after upgrading weak negatives beyond farneg threshold protocols; prior no-call lifts are tiny (`+0.000063` to `+0.000084` vs v616).
5. **Train-soundscape sequence/file/site branches** — `CONTINUE LOW PRIORITY`. Keep training distinct data points when no source-verifier work is unblocked, but stop blind PANNs/B0 threshold/file-context tweaks unless they introduce a new source-winner feature or audited negative protocol.
6. **Late-day guarded public source fill** — `CONDITIONAL`. If <3h to reset and no verifier-grade repo candidate exists, fill slots with best remaining clean, hidden-rerunnable public-source candidates only after fresh preflight/dedup. Do not rerun v644/v647 or malformed v645/v646/v648-v650.

## Submission decision
- `submit_approved=false` for the source SED rank-grid CSV. It is comparison-grade and promising, but lacks a hidden-test source package and is too close to known over-optimistic SED/local-proxy behavior to spend an early-day slot.

## Artifacts
- Ledger: `artifacts/model_data_point_ledger/20260531T1428Z_source_winner_protosed_confidence_meta.md`
- Audit: `artifacts/source_winner_confidence_meta_audit/20260531T1428Z/audit_summary.json`
- Report: `artifacts/source_winner_confidence_meta_audit/20260531T1428Z/audit_report.md`
- Script: `scripts/birdclef_source_winner_confidence_meta_audit.py`

## Next exact action
Fork/private-verify the v644/v647 EoS8 source with configurable SED/PowerOpt outer weights. First smoke should produce hidden-safe public-session `submission.csv` plus intermediate `submission_sed.csv`, validate schema/nonconstant/dedup, and only then consider a competition slot.
