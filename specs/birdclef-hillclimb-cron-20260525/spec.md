# BirdCLEF 2026 ClawTeam Hill-Climb Cron Spec — 2026-05-25

## User directive
Use ClawTeam for hill climbing this competition. Explore all possible solutions, rank them by potential public LB score, use up the slots, and run by cron.

## Competition state at setup
- Current best public LB: 0.949.
- Latest live check: v616 completed and scored 0.949.
- 2026-05-25 UTC submissions used at setup: 1/5.
- Deadline is near; slots should be treated as useful exploration budget, not something to preserve indefinitely.

## Operating principle
Aggressive hill-climbing is authorized, including competition submissions, but every candidate must still pass basic competition-integrity guards:
- no disallowed/private data;
- no hidden/test labels or leakage;
- no malformed, constant, nonfinite, misaligned, sample-fallback, or static public-output-only final submissions;
- no duplicate submission descriptions or exact duplicate matrices;
- source/runtime/kernel status must be checked before submission when applicable.

## ClawTeam roles per run
Each cron run should internally use or emulate:
- Coordinator / Research Lead
- Public Solution & Model Scout
- Data & Feature Scientist, when data/branch insight is needed
- Prediction & Ensemble Analyst, when predictions/artifacts exist
- Experiment Engineer
- Critic / Red Team
- Verifier / Skeptic

## Slot policy
- Check UTC daily slot count at start and before each submission.
- Target: use all available daily slots by reset if there are valid candidates.
- Early in the day: prefer verifier-grade or strong exploratory candidates.
- Late in the day (<3h to UTC reset): fill remaining slots with the highest-ranked valid exploratory candidates, even if evidence is exploratory/comparison-grade, as long as they are source-clean and pass guards.
- Never submit a candidate known to be invalid, duplicate, fallback/static-only, or rule-risky.

## Candidate ranking
Rank candidates by expected public LB potential and information value:
1. genuinely new hidden-behavior branches: EfficientAT/PANNs AudioSet event/no-call, non-Aves/no-train soundscape specialist, broadened OOF negative/no-call SED, 20s temporal branch;
2. source-clean public candidates with distinct model families and valid hidden-safe output path;
3. repo-owned private verifiers that complete and pass output guards;
4. ensemble candidates that beat v616 in audit or offer high information value;
5. near-duplicates only if late-day slots would otherwise expire and candidate passes all guards.

## Required reporting
Each run must report:
- current best score and last submissions;
- UTC slot usage and time to reset;
- ranked candidate queue with reasons;
- actions taken;
- submissions made or why none;
- artifact paths / kernel refs / commits;
- next exact action.
