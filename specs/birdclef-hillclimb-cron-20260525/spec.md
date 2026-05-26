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

## Data-point model training policy — added 2026-05-25
The hill-climb loop should train distinct new model families anyway, even when the immediate smoke is unlikely to beat v616 or qualify for submission. The goal is to create a measured search landscape, not just a binary submit/no-submit gate.

Rules:
- Prefer many bounded, diverse data points over repeatedly polishing the same plateau family.
- For each new model/training branch, record: model family, init/source, train rows, labels/targets, input window, augmentations, loss, epochs, runtime, CV/proxy metric, branch/anchor/v616 correlation when possible, export/runtime status, and whether it created useful hidden-behavior diversity.
- Weak local score does not automatically kill a branch if it is decorrelated, covers rare/non-Aves/no-call slices, or can become an ensemble sidecar.
- Still kill branches that are malformed, rule-risky, impossible to package, all-zero/constant, or exact duplicates.
- Maintain an experiment ledger so the critic can rank future choices from evidence rather than vibes.

Default new-model data-point queue:
1. Train-soundscape sequence/file/site mining branch: sequence-aware MIL or temporal pooling over 5s windows, leave-site/file evaluation, site-balanced sampling, and per-file/context features. This is now the top data-driven lane because train_soundscapes is the most useful under-mined dataset.
2. Deeper soundscape-native training variant: fine-tune more than a shallow head (last blocks/adapters or compact CNN/SED) on task-aligned soundscape/OOF-teacher targets, with strong regularization and leave-site/file gates. Do not full-fine-tune large AudioSet encoders on sparse labels blindly.
3. EfficientAT/PANNs AudioSet event/no-call branch, but only after reformulating from frozen 72-label heads into broad acoustic context/no-call features or a 234-class sidecar wrapper with multi-site validation.
4. Broader OOF negative/no-call SED student.
5. Non-Aves / no-train-soundscape specialist.
6. 20s temporal context/localmax branch.
7. G124/V2S-init larger/all-row pilot.
8. Alexy/sidecar-derived model only if source/checkpoint access becomes clean.

Soundscape sequence/file/site branch requirements:
- Treat `train_soundscapes` as sequences/files/sites, not isolated rows.
- Use group-aware validation: leave-site, leave-file, and site-balanced bootstrap.
- Include temporal context features: neighbor windows, local max/mean pooling, file-level MIL pooling, label persistence, and time-bin effects.
- Include data diagnostics: per-site label distribution, per-file label density, class co-occurrence, no-train/non-Aves coverage, and no-call/background protocol.
- Promotion requires a raw branch or wrapper audited against v616, not just row-level ROC.
