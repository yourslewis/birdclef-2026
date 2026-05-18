
## 2026-05-18 20:55 UTC execution update — SafeAlign/Pilkwang triage, Chaney v37 queued

Live status at loop start:

- Latest confirmed public best remains **0.949** from `v574`/`v575`/`v576`.
- 2026-05-18 UTC visible submission count is `5`, so the day is capped.
- No stale `submit_v577_when_ready.py` / v578 scalar monitor was alive.
- PR #245 was already merged to `main`; new work moved to fresh branch `feature/birdclef-096-frontier-chaney-v37-20260518`.

### SafeAlign/S106 result

Compared `itshyao/birdclef-2026-s106-eos5-0949-safealign2` and `beicicc/bc26-s106-eos5-sa2-may18` against EoS5.

Finding: **not a 0.96 candidate by itself**.

- Source is EoS5-like with top-level weight `Model_2=0.04`, `Model_5=0.96`.
- Diff versus EoS5 is small: markdown/table changes, weight change from `0.0327/0.9673` to `0.04/0.96`, and a robust final row-id/blend guard.
- Since `v576` already proved Model5-only scores `0.949`, this top-level complement/row-alignment hardening is unlikely to produce a 0.960 jump.

Decision: do not spend a slot on SafeAlign/S106 unless later evidence says it scores materially above EoS5.

### Pilkwang acoustic time-window rank fusion result

Inspected `pilkwang/birdclef-26-acoustic-time-window-rank-fusion`.

Finding: **idea-mining only, no direct slot**.

- It is essentially a clean single-branch replay of `Karnakbayev_PowerOptimization_LB0948` with `xSED=[0.600,0.400]`.
- It uses older EoS4-style literals: `lambda_prior=0.4` and `rank_aware_scaling(... power=0.5)`.
- EoS5's improvement already moved those to `lambda_prior=0.5` and rank-aware power `0.6`, reaching `0.949`.

Decision: skip direct submission. Extract only if later useful as documentation of the PowerOptimization lineage.

### New candidate selected: Chaney v37 Nina-style gate frontier

Candidate:

- Public kernel: `chaneyma/bc26-gate-v37-ninastyle-branch`
- Public version: `1`
- Guarded submission description: `v580: Guarded direct Chaney v37 Nina-style gate frontier replay`

Why it is more 0.96-relevant than EoS5 scalar sweeps:

- Structurally different from EoS5 scalar tuning: combines an Imaad-style public946 base, replicated gate baseline artifacts, multiple ProtoSSM artifact branches, Nina-style final gate logic, temporal LSE/file-confidence postprocess, and tiny direct/rank diversity blends.
- Source includes local OOF/CV references around `0.967` for the gate/postprocess stack; not a leaderboard guarantee, but materially stronger evidence than `v577` scalar tuning.
- Kernel is COMPLETE/no failure and emits competition-format `submission.csv` plus many branch outputs.

Preflight performed:

- Source pull via Kaggle Bearer API v1 succeeded.
- Current public kernel version matched expected `1`.
- Required source markers found: v37 final blend cell, `submission_v37_direct_i974_n026.csv`, `Default submission.csv =`, `test_soundscapes`, `IS_DRY_RUN`, `sample_submission.csv`, `row_id`.
- Dry-run fallback is explicitly guarded by `dryrun_fallback` plus `raise SystemExit(0)`.
- Kernel status COMPLETE/no failure.
- Output preflight passed after requiring durable outputs: `submission.csv`, `v37_ninastyle_branch_shared_blend_summary.json`, `submission_imaad0946.csv`, `submission_sed.csv`.

Action:

- Added `scripts/submit_v580_chaney_v37_when_slot.py`.
- Started guarded monitor pid `43469`; log `logs/submit_v580_chaney_v37_when_slot_20260518T2055Z.log`.
- It attempted submission, hit daily cap (`3.3 hours from now`), and is sleeping `12000s` before retry.

Caveat:

- Some `datasetDataSources` are blank in Kaggle metadata, so repo-owned porting may not be immediately possible. This is a guarded direct public-code replay first. If it scores high, the next task is to identify/attach the underlying artifact datasets or reproduce the branch artifacts in repo-owned form.
