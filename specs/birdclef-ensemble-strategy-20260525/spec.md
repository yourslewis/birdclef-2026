# BirdCLEF 2026 Ensemble Strategy Spec — 2026-05-25

## Goal
Build a better ensemble strategy from all good and meaningfully different BirdCLEF candidates available in the canonical repo, targeting a public LB improvement over the current 0.949 plateau while reducing slot waste.

## Scope
- Analyze available top/tied candidate outputs, sidecars, private verifier outputs, public-output artifacts, OOF/training-soundscape labels, and docs.
- Separate true diversity from near-duplicate variants.
- Design a hidden-test-safe ensemble strategy that can be implemented as a repo-owned Kaggle kernel/verifier.
- Prefer no-slot validation first. Competition submission remains blocked until Coordinator + Verifier approve.

## Non-goals
- No cron score-push loop.
- No direct submission during Phase 1.
- No near-duplicate scalar/power tweak just because it locally lifts.
- No merge to main without Wenhao approval.

## Canonical repo
`/Users/yourslewis/.openclaw/repos/birdclef-2026`

## Current facts
- Current confirmed public best: 0.949.
- Many 0.949 variants exist, but many are correlated plateau families.
- v611/v612/v616 showed strong local sidecar lifts can tie, not improve.
- Local train-soundscape/sidecar gates are rejection filters, not approval filters.

## Acceptance criteria for Phase 1
- Candidate lineage/diversity matrix exists.
- Prediction/ensemble analysis exists with actual artifact paths and numeric comparisons where possible.
- Validation gate proposal exists.
- Data/slice analysis exists.
- Coordinator synthesizes one implementable ensemble strategy, one backup, and explicit no-submit/reject rules.
