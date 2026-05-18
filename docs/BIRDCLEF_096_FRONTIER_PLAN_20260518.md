
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

## 2026-05-18 21:45 UTC execution update — A2Prime/NFNet fallback prepared

Live status:

- Current best remains **0.949** from `v574`/`v575`/`v576`.
- UTC day still capped with `5` visible 2026-05-18 submissions.
- `v580` monitor is alive and sleeping after daily-cap response; it has not submitted yet.

Additional source scan:

- `claudedevore/birdclef-2026-r0946-a2prime-nfnet-submit`: source is useful but public kernel status is `ERROR` and lacks `submission.csv`; skip direct replay.
- `claudedevore/birdclef-2026-r0946-a2prime-effv2s-submit`: COMPLETE, but `submission.csv` defaults to `base_3way` rather than an EffV2S candidate. Direct replay would not test the new EffV2S branch unless submitting an alternate output file; skip for now.
- `lucataco/bc26-claude-a2prime-nfnet-fix`: COMPLETE and explicitly fixes the NFNet candidate so hidden runs default to `a2_nfnet_w03` for `submission.csv`. Outputs include `a2nfnet_blend_summary.csv`, `nfnet_branch_summary.csv`, `nfnet_sanity_file_summary.csv`, `submission_nfnet.csv`, `submission_a2_nfnet_w03.csv`, and `submission.csv`.

Action:

- Added guarded fallback submitter `scripts/submit_v581_a2prime_nfnet_when_ready.py`.
- Started monitor pid `68275`, log `logs/submit_v581_a2prime_nfnet_when_ready_20260518T2145Z.log`.
- The monitor intentionally waits for `v580` to become visible and complete before doing anything.
- If `v580` improves above current best (`>0.949`), v581 exits to preserve slots for repo-owned v580 confirmation instead.
- If `v580` ties/drops/no-scores, v581 preflights and submits the distinct Lucataco A2Prime/NFNet candidate as the next 0.96-relevant fallback.

Rationale:

- v581 is structurally distinct from the Chaney gate family and from EoS5 scalar tuning.
- It uses a public A2Prime/NFNet branch with new `brendancarlin/birdclef2026-models` model source plus Perch/SED/BirdNET rank blending.
- It is not auto-competing with v580 for the first reset slot; it is result-gated.

## 2026-05-18 22:50 UTC execution update — monitor restart + v580 port dependency map

Live status:

- Current best remains **0.949** from `v574`/`v575`/`v576`.
- 2026-05-18 UTC remains capped with `5` visible submissions.
- No v580/v581 submission is visible yet.

Monitor health finding:

- The earlier nohup PID files for v580 (`43469`) and v581 (`68275`) were stale; both processes were no longer alive and had not appended any new error to their logs.
- Restarted them as OpenClaw-managed background sessions:
  - v580 session `tender-ridge`, pid `88792`.
  - v581 session `brisk-kelp`, pid `88794`.
- v580 preflight re-passed and attempted submission again, then hit daily cap with `78 minutes from now`; it is sleeping `4800s` before retry.
- v581 is alive and waiting for v580 to become visible before any fallback action.

### v580 repo-owned confirmation dependency map

If v580 improves, the next task is repo-owned confirmation. The Chaney source reveals the likely artifact dependencies that were blank in Kaggle metadata:

Core public/common sources:

- competition: `birdclef-2026`
- model: `google/bird-vocalization-classifier/TensorFlow2/perch_v2_cpu/1`
- dataset/model paths:
  - `jaejohn/perch-meta`
  - `rishikeshjani/perch-onnx-for-birdclef-2026`
  - `tuckerarrants/perch-v2-no-dft-onnx`
  - `lixin73/birdclef2026-v27-onnx-perch-meta-forum-v1-lb872`
- notebook source / wheel source:
  - `ashok205/tf-wheels`
  - `vyankteshdwivedi/notebook1b25083f0d`

Chaney artifact datasets referenced by source:

- `chaneyma/birdclef2026-edits-protossm-sed-onnx-infer-artifacts`
- `chaneyma/bc26-gate-fake008-head0015-baseline-onnx`
- `chaneyma/bc26-edits-protossm-sed-v7-all66-40x20`
- `chaneyma/bc26-edits-protossm-sed-v8-all66-synth-p010-40x20`
- `chaneyma/bc26-probe-middle-pca128-raw085-logreg015`

Likely confirmation plan if `v580 > 0.949`:

1. Pull public source v1 and normalize to a repo-owned private kernel.
2. Attach the above explicit sources instead of relying on blank dataset entries.
3. Keep default output equivalent to public v37: `submission_v37_direct_i974_n026.csv -> submission.csv`.
4. Preserve dry-run fallback guard (`dryrun_fallback` + `raise SystemExit(0)`) and final row-id checks.
5. Push repo-owned confirmation and submit only after COMPLETE/output verification.

If any Chaney artifact dataset is not attachable, fallback confirmation path is to reproduce those branch artifacts from public source where possible, but that is larger and should only happen if the direct v580 score justifies it.
