
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

## 2026-05-18 23:30 UTC execution update — reset queue healthy, repo-owned v580 dependency blocker found

Live status:

- Current best remains **0.949** from `v574`/`v575`/`v576`.
- 2026-05-18 UTC still has `5` visible submissions; 2026-05-19 UTC has `0` visible submissions at check time.
- Managed monitors are alive:
  - v580: session `tender-ridge`, pid `88792`; sleeping after daily-cap response and ready to retry near/after reset.
  - v581: session `brisk-kelp`, pid `88794`; polling every 15 minutes and preserving slots until v580 is visible/scored.

### Repo-owned v580 confirmation blocker

Queried the explicit Chaney artifact dataset candidates via Kaggle Bearer API:

- `chaneyma/birdclef2026-edits-protossm-sed-onnx-infer-artifacts` — `403 datasets.get denied`
- `chaneyma/bc26-gate-fake008-head0015-baseline-onnx` — `403 datasets.get denied`
- `chaneyma/bc26-edits-protossm-sed-v7-all66-40x20` — `403 datasets.get denied`
- `chaneyma/bc26-edits-protossm-sed-v8-all66-synth-p010-40x20` — `403 datasets.get denied`
- `chaneyma/bc26-probe-middle-pca128-raw085-logreg015` — `403 datasets.get denied`

Public/common dependencies are attachable:

- `lixin73/birdclef2026-v27-onnx-perch-meta-forum-v1-lb872` — OK
- `jaejohn/perch-meta` — OK
- `rishikeshjani/perch-onnx-for-birdclef-2026` — OK
- `tuckerarrants/perch-v2-no-dft-onnx` — OK

Implication:

- If `v580` improves, an immediate repo-owned confirmation may be blocked by private/non-attachable Chaney artifacts.
- Best follow-up becomes either:
  1. reproduce the needed Chaney artifact branches in our repo-owned kernel/datasets, or
  2. mine the public source for the minimal logic that can be applied to attachable/public artifacts.
- Do not assume a simple v582 repo-owned replay can attach all sources.

### Other frontier attachability notes

Potentially attachable follow-up lanes if v580/v581 do not improve:

- `lucataco/bc26-claude-a2prime-nfnet-fix` uses public/attachable `brendancarlin/birdclef2026-models` and common Perch/SED/BirdNET sources.
- `kamongi/pantanal-distill-birdclef2026` uses attachable `konbu17/bird26-train-audio-head-v1` plus common sources, but source comments suggest public LB around `0.944`; treat as idea-mining rather than a priority slot.

Action this run: no new submitters added; preserve the queue order v580 → v581.

## Heartbeat update — v580 submitted after UTC reset — 2026-05-19 00:05 UTC

- Managed v580 session `tender-ridge` woke after the UTC reset, re-ran source/output preflight, and successfully submitted.
- Submission: `v580: Guarded direct Chaney v37 Nina-style gate frontier replay`
- Kaggle ref: `52790976`
- Initial status: `pending`, no score yet.
- 2026-05-19 UTC visible submission count is now `1`.
- Managed v581 session `brisk-kelp` remains alive and is correctly waiting for v580 to become complete before taking any fallback action.

Next decision gate:

- If `v580 > 0.949`: stop v581 and work on Chaney artifact reproduction / portable logic extraction, because direct repo-owned replay is blocked by private Chaney artifacts.
- If `v580 <= 0.949` or no-scores: allow v581 A2Prime/NFNet fallback to proceed.

## 2026-05-19 00:45 UTC execution update — v580 pending, v582 scan while waiting

Live status:

- `v580` is visible and pending: ref `52790976`, description `v580: Guarded direct Chaney v37 Nina-style gate frontier replay`.
- Current confirmed best remains **0.949** until v580 scores.
- 2026-05-19 UTC visible submission count is `1`.
- `v581` fallback monitor remains alive and correctly waits for v580 to complete before acting.
- v580 submit process exited after successful submission, so `tender-ridge` is gone by design.

Additional source scan for a possible v582 queue item:

- `cocoaai/bc26-stars-v129-exp019-eos4-birdnet`: EoS4/Model7 clone around `0.948`; not distinct enough after EoS5/v580.
- `cocoaai/bc26-stars-v130-nina-eos3-birdnet`: EoS3-like `Model_3/Model_10` clone; not a high-upside new lineage.
- `cocoaai/bc26-karnak-advance-ensemble-patched`: EoS5-like `Model_2/Model_5` blend; already covered by v574-v576.
- `adarsh5harma/birdclef-2026-v63-nina-eos5-fork` and `itshyao/birdclef-2026-s103-public-eos5-0949`: EoS5-like `0.04/0.96` top-level blend; same conclusion as SafeAlign/S106.
- `amulopapa67/bc26-full-yous-gate-rb035-nb-20260517`: COMPLETE with `submission.csv`; public/attachable sources include `konbu17/bird26-train-audio-head-v1`; final blend is `0.65 * Youssef rank + 0.35 * gate rank`. It is a plausible idea-mining fallback but probably overlaps public946/gate-sidecar lanes; do not queue before v580/v581 results.
- `karnakbaevarthur/optimized-dual-architecture-ensemble`: COMPLETE with `submission.csv`; attachable sources; appears to be pc010/gate + taxonomy/mirror/rare postprocess lineage. Candidate for later source extraction, but not stronger than current v580/v581 queue.
- `cocoaai/bc26-alexy-ensemble-perch-cnn`: COMPLETE with `submission.csv`, uses `alexycactus/birdclef-2026-cnn-fold-checkpoints`; source reports OOF AUC diagnostics and Perch+CNN ensemble. Interesting model-zoo idea, but no visible strong LB claim; hold for later.
- `kospintr/birdclef-efficientnet-perch-distill-mixup`: COMPLETE but sample/empty-output risk when no hidden test (`submission_df` starts from sample columns and only fills if `test_audio_full_paths`); do not submit blindly.

Decision:

- Do not start a v582 submitter now. Preserve queue order: v580 result gate, then v581 fallback if needed.
- Best next action remains: wait for v580 score. If v580 improves, focus on reproducing/extracting Chaney artifacts; if not, v581 proceeds.

## 2026-05-19 01:50 UTC execution update — v580 dropped, v581 submitted

Live result:

- `v580: Guarded direct Chaney v37 Nina-style gate frontier replay` completed at **0.944**, below the current `0.949` best.
- Lesson: Chaney v37's OOF/CV/gate stack did not transfer to public LB; kill the Chaney direct-replay lane for slots. Its artifact access blocker remains relevant only for idea-mining, not immediate confirmation.

Fallback action:

- The v581 guard initially exited because source marker matching was too strict for raw notebook JSON (`submission_a2_nfnet_w03.csv` is produced as an output but not present literally in source).
- Relaxed the source markers to semantic notebook markers (`default_name`, `a2_nfnet_w03`, `A2NF blend complete`, diagnostics, hidden-test markers) while keeping concrete output-file verification strict.
- Re-ran v581 preflight successfully:
  - source pull OK, version `2`, source length `99456`
  - kernel COMPLETE/no failure
  - required output files present: `submission.csv`, `submission_a2_nfnet_w03.csv`, `a2nfnet_blend_summary.csv`, `nfnet_branch_summary.csv`, `nfnet_sanity_file_summary.csv`, `submission_nfnet.csv`, `submission_base_3way.csv`
- Submitted `v581: Guarded direct Lucataco A2Prime NFNet frontier replay`, ref `52793377`.
- Initial status: pending. 2026-05-19 UTC visible submission count is now `2`.

Next gate:

- If v581 improves above `0.949`, pursue repo-owned confirmation; this path appears more attachable than Chaney because `brendancarlin/birdclef2026-models` is public/attachable.
- If v581 ties/drops, continue source frontier scan; current possible idea-mining candidates are Amulopapa Youssef+gate, Karnak optimized-dual, and Alexy Perch+CNN, but none should be queued before v581 scores.

## 2026-05-19 02:45 UTC execution update — v581 pending; v582 fallback staged

Live status:

- `v581: Guarded direct Lucataco A2Prime NFNet frontier replay` remains pending, ref `52793377`.
- Current best remains `0.949`; 2026-05-19 UTC visible count is `2` (`v580`, `v581`).
- No stale v577/v578 scalar submitters are alive.

Result-gated fallback prepared:

- Added `scripts/submit_v582_amulopapa_yous_gate_when_ready.py` for `amulopapa67/bc26-full-yous-gate-rb035-nb-20260517` v4.
- The monitor is running as managed OpenClaw session `lucky-zephyr`, pid `35622`.
- It does **not** submit while v581 is pending.
- If v581 scores above `0.949`, it exits so we can port/confirm v581.
- If v581 ties/drops/no-scores, it preflights and submits v582 as a distinct Youssef+gate rank-blend source replay.

Independent preflight findings:

- Source pull OK, version `4`, source length `201355`.
- Key source markers present: `submission_youssef.csv`, `submission_gate.csv`, `0.65 * yr`, `0.35 * gr`, `test_soundscapes`, `sample_submission.csv`, `submission.csv`, `row_id`.
- Kernel status COMPLETE/no failure.
- Output files present: `submission.csv`, `submission_youssef.csv`, `submission_gate.csv`, `submission_protossm.csv`, `submission_sed.csv`, `submission_birdnet.csv`, plus cache files.
- Public output schema check: 3 dry-run/sample rows, 235 columns, `row_id` present/unique, finite numeric values in `[0.0, 1.0]`.

Rationale:

- This is more distinct than another EoS/Karnak clone: final output is a rank blend of Youssef branch and gate branch (`0.65/0.35`).
- It is still guarded behind v581 because spending a slot before the A2Prime/NFNet result would be premature.

## 2026-05-19 03:50 UTC execution update — v581 timeout, v582 submitted, v583 scan

Live result/state:

- `v581: Guarded direct Lucataco A2Prime NFNet frontier replay`, ref `52793377`, completed with **no public score**.
- Kaggle error: `Your submission notebook exceeded the allowed runtime.`
- Root cause classification: hidden runtime timeout, not source/output schema failure. Public output schema from the source run was valid sample-sized output (`3 x 235`, unique `row_id`, finite values), but hidden execution exceeded the competition runtime limit.
- `v582: Guarded direct Amulopapa Youssef gate rb035 frontier replay`, ref `52796003`, was submitted by the gated monitor after v581 no-scored. Current status: pending.
- 2026-05-19 UTC visible count is now `3`.

v582 monitor behavior:

- OpenClaw session `lucky-zephyr` exited successfully after submission.
- Re-preflight before submission passed: source pull v4, kernel COMPLETE/no failure, required output files present.

Next-candidate scan while v582 is pending:

- `karnakbaevarthur/optimized-dual-architecture-ensemble` v3:
  - Pull OK, source length `124390`.
  - Sources: distilled SED public, train-audio-head, perch-meta, perch ONNX, tf-wheels, Perch model.
  - Kernel COMPLETE/no failure.
  - Outputs: `submission.csv`, `submission_protossm.csv`, `submission_sed.csv`, cache files.
  - Public output schema valid sample-sized output: `3 x 235`, unique `row_id`, finite values in `[0.226, 0.361]`.
  - Possible v583 fallback if v582 ties/drops/no-scores, but lower confidence because it looks closer to known Perch/SED/taxon/gate lineage than a true 0.96 successor.
- `cocoaai/bc26-alexy-ensemble-perch-cnn` v1:
  - Pull OK, source length `37597`; kernel COMPLETE/no failure.
  - Distinct CNN/Perch idea, but direct output is unsafe for competition replay: `submission.csv` has `192` rows with `BC2026_Train_*` row IDs, not sample/hidden-test-shaped rows.
  - Do not submit direct. Only idea-mine/port if later needed.

Decision:

- Do not queue v583 while v582 is pending. Preserve the remaining two 2026-05-19 slots for v582 result-driven action.

## 2026-05-19 04:45 UTC execution update — v582 still pending, v583 source scan held

Live state:

- `v582: Guarded direct Amulopapa Youssef gate rb035 frontier replay`, ref `52796003`, remains pending.
- Current confirmed best remains `0.949`; 2026-05-19 UTC visible count remains `3`.
- No active v577/v578/v58x submitter processes are alive; v582 monitor exited successfully after submission.
- PR #246 remains open, mergeable, and blocked. Working tree was clean at loop start.

Kaggle API source search:

- Searched competition kernels via `list_kernels` for recent/date-run candidates around `0.95`, `0.96`, `EoS`, `SafeAlign`, `PowerOptimization`, `gate rank fusion`, `Perch CNN`, `NFNet`, and score-claim strings `0.951`–`0.955`.
- No explicit `0.951`–`0.955` kernel claims surfaced.
- New/recent May 19 cluster is mostly Beicicc/Karnak safe/gated variants and Rabeya V4 pipeline.

Deep scan findings:

- `beicicc/bc26-v65-karnak-safe-may19` v1:
  - COMPLETE/no failure, valid sample-shaped output (`3 x 235`, unique row IDs, finite values).
  - Top-level solution is still Model2/Model5 direct blend: `Model_2=0.03`, `Model_5=0.97`, `LB 0.949` on Model5.
  - This is an EoS5-family safety/weight tweak, not a distinct 0.96 hypothesis. Hold; do not submit before v582 result.
- `beicicc/bc26-karnak-gated-safe-may19` v1:
  - COMPLETE/no failure, valid sample-shaped output (`3 x 235`).
  - Top-level solution is `Model_2=0.0321`, `Model_5=0.9679`, with markdown claiming Model5 `0.949+` and 5 gates.
  - Also EoS5-family; hold.
- `beicicc/bc26-v65-karnak-gated-may19` v1:
  - COMPLETE but direct output unsafe: `243` rows with `240` train row IDs and non-finite parse in downloaded `submission.csv`.
  - Do not direct-submit.
- `mtoshidesu/testbirdclef-2026-v6` v3:
  - Status ERROR/no outputs; skip direct.
- `mtoshidesu/test-s106-eos5-sa2-may18` v1:
  - COMPLETE/valid sample-shaped output, but it is S106/EoS5-like `0.04/0.96`; already covered by earlier SafeAlign/S106 triage and v576 lesson.
- `cocoaai/bc26-mtoshi-visual-birdnet` v1:
  - COMPLETE/valid sample-shaped output; uses `mtoshidesu/birdclef-flow-diagram` plus BirdNET/Perch sources.
  - Interesting visual/BirdNET idea-mining lane, but no strong score claim and output distribution resembles conservative sample-run blending. Hold; possible idea extraction only.
- `rabeya100x/birdclef-2026-v4-pipeline` v5:
  - Source pull OK and structurally resembles the visual/BirdNET/Perch family, but kernel was RUNNING with no outputs at scan time. Recheck later; do not submit yet.
- `anthonytherrien/birdclef-2026-ensemble` v2:
  - COMPLETE but direct output unsafe: `243` rows with `240` train row IDs and non-finite parse; do not direct-submit.

Decision:

- Do not queue v583 while v582 is pending. The best preflighted candidates are either EoS5-family 0.949 variants or direct-unsafe.
- If v582 ties/drops/no-scores, the least-bad direct fallback is currently `beicicc/bc26-karnak-gated-safe-may19` or `beicicc/bc26-v65-karnak-safe-may19`, but their expected upside is low. Prefer another broad source scan / Rabeya completion check before spending a slot unless the day is near expiry.

## 2026-05-19 05:50 UTC execution update — v582 dropped, v583 S118 submitted

Live result/state:

- `v582: Guarded direct Amulopapa Youssef gate rb035 frontier replay`, ref `52796003`, scored **0.947**.
- Current best remains `0.949` from v574/v575/v576.
- 2026-05-19 UTC visible count before v583 was `3`; after v583 submission it is `4`.
- No stale v577/v578 scalar or v58x submitter process was alive.

Lesson from v582:

- Amulopapa Youssef+gate `0.65/0.35` rank blend is valid and scored above older public946 lines, but it did not beat EoS5/Model5 `0.949`.
- Treat Youssef+gate rank blend as useful idea-mining, not a confirmation lane.

Broader scan after v582:

- Recent Kaggle `DATE_RUN` pages show the newest cluster: Mtoshi/Zhaorong visual BirdNET, Beicicc Anthony/safe/gated EoS5-family variants, Itshyao S118/S120 launchers, JGuevara TTA, CocoaAI Youssef D2/E1, Apachikoff V6, Karnak advance.
- Web search did not surface external explicit `0.95+` claims.
- Rabeya V4 became inaccessible via pull/session/output (`403`) despite appearing in list search, so it is not usable now.
- Beicicc safe/gated kernels are valid but EoS5-family Model2/Model5 variants; expected upside is low.
- Zhaorong/Mtoshi Visual BirdNET and CocoaAI Youssef D2/E1 are valid and idea-mining-worthy, but lack strong public-score evidence; hold for later.
- JGuevara TTA emits a zero-valued fallback `submission.csv`; do not direct-submit.

New selected candidate:

- `itshyao/birdclef-2026-s118-gated-g116-delta-launcher` v2.
- Reason: more structurally distinct than the Beicicc EoS5-weight variants; outputs include `submission_g116_hgnet_b1_all5_s118.csv`, suggesting a G116/HGNet delta branch blended into the EoS/PowerOptimization family.
- Caveat: source is a public launcher that executes attached `s118_source.ipynb`; repo-owned confirmation may require recovering/porting the attached source dataset. This is a guarded direct replay, not a portable port.

Preflight:

- Source pull OK, version `2`, launcher source length `1165`.
- Required launcher source markers present: `s118_source.ipynb`, `Executing source notebook`, `run source cell`, `S118 launcher complete`.
- Kernel COMPLETE/no failure.
- Output files present: `submission.csv`, `submission_g116_hgnet_b1_all5_s118.csv`, `subm_5.csv`, `submission_protossm.csv`, `submission_sed.csv`, `v17_logs.json`.
- Prior schema scan showed valid sample-shaped output: `3 x 235`, unique row IDs, finite values in `[0.473, 0.546]`.

Action:

- Added `scripts/submit_v583_s118_gated_g116_delta.py`.
- Submitted `v583: Guarded direct S118 gated G116 delta launcher replay`, ref `52799220`, initial status pending.
- Decision: preserve the final remaining 2026-05-19 slot until v583 scores or a stronger source appears.

## 2026-05-19 06:50 UTC execution update — v583 no-score, v584 final slot submitted

Live result/state:

- `v583: Guarded direct S118 gated G116 delta launcher replay`, ref `52799220`, completed with **no public score**.
- Kaggle error: hidden rerun hit an unhandled error. Root cause classification: launcher/attached-source hidden rerun failure; avoid S118/S120-style launchers for direct replay unless the real attached source is recovered and made portable.
- Current best remains `0.949` from v574/v575/v576.
- 2026-05-19 UTC visible count became `5` after v584; day is now capped.

Final-slot scan:

- Deep-scanned full-source candidates after v583 failed:
  - `cocoaai/bc26-youssef-d2-sonomirror-birdnet` v1: COMPLETE/valid sample output; full visible source; Youssef/BirdNET sonomirror tweak.
  - `cocoaai/bc26-youssef-e1-rare-tail-birdnet` v1: COMPLETE/valid sample output; rare-tail tweak; likely same family.
  - `zhaorongdai/bc26-cocoa-mtoshi-visual-birdnet` v1: COMPLETE/valid sample output; full visible source; Visual/BirdNET/Mtoshi lineage with `0.949-style` prior, TTA Proto, per-class ensemble weights, BirdNET source.
  - `mtoshidesu/birdclef-2026-visual-cpu-inference` v17: ERROR/no outputs; skip direct.
  - `kotata0306/birdclef-2026-youssef-c2-birdnet-aves-spike` v1: COMPLETE/valid, but same Youssef/BirdNET family.
  - `kotata0306/birdclef-2026-youssef-a1-rankblend-sedup` v1: COMPLETE/valid, but same Youssef/BirdNET family.

Decision:

- Use final slot on `zhaorongdai/bc26-cocoa-mtoshi-visual-birdnet` rather than another Youssef/BirdNET micro-tweak because v582 already showed Youssef+gate tops out below `0.949`, while Visual/BirdNET is the most distinct remaining full-source, schema-safe candidate.

Preflight/action:

- Added `scripts/submit_v584_zhaorong_visual_birdnet.py`.
- Source pull OK, version `1`, source length `101392`.
- Required markers present: `run_tta_proto`, `0.949-style tweak`, `lambda_prior=0.5`, `ENSEMBLE_W_PER_CLASS`, `BirdNET`, `test_soundscapes`, `sample_submission.csv`, `row_id`.
- Kernel COMPLETE/no failure.
- Required outputs present: `submission.csv`, `submission_birdnet.csv`, `submission_protossm.csv`, `submission_sed.csv`, `cache/perch_arrays.npz`, `cache/perch_meta.parquet`.
- Prior schema scan: valid sample-shaped output (`3 x 235`, unique row IDs, finite values in `[0.475, 0.556]`).
- Submitted `v584: Guarded direct Zhaorong Mtoshi Visual BirdNET replay`, ref `52800792`, initial status pending.

## 2026-05-19 07:45 UTC execution update — v584 pending, day capped, next-reset queue ranked

Live state:

- `v584: Guarded direct Zhaorong Mtoshi Visual BirdNET replay`, ref `52800792`, remains pending.
- Current best remains `0.949` from v574/v575/v576.
- 2026-05-19 UTC visible count remains `5`; day is capped.
- No stale v577/v578/v58x submitter process is alive.
- PR #246 is open, mergeable, and blocked; working tree was clean before this log update.

Fresh source search while capped:

- Re-ran Kaggle DATE_RUN list over recent BirdCLEF kernels and saved `artifacts/public_kernels_20260519_frontier_candidates/date_run_all_20260519T0740Z.json`.
- Web search found no external explicit `0.950+` / `0.951+` claims.
- Deep-scanned promising next-reset candidates and saved `artifacts/public_kernels_20260519_frontier_candidates/deep_scan4_nextreset_20260519T0740Z.json`.

Next-reset candidate ranking if v584 does not improve:

1. `franksunp/birdclef-2026-5-branch-v4-tta-fix` v1 — **best distinct next-reset candidate so far**
   - COMPLETE/no failure; full visible source; output schema valid (`3 x 235`, unique row IDs, finite values).
   - Structurally distinct 5-stream rank-percentile ensemble: ProtoSSM, Tucker SED, Snowflake SED, CLAP, plus BirdNET branch / graceful degradation.
   - Sources include public CLAP INT8 and Snowflake SED bundles plus common Perch/SED sources.
   - Caveat: prior tiny CLAP/Snowflake-ish sidecars did not improve, and sample output distribution is compressed (`~0.27-0.40`), so submit only if v584 fails and no stronger 0.950+ source appears before reset.
2. `meenalsinha/birdclef-2026-improved` v20 — valid but likely close to v584/Visual-BirdNET family
   - COMPLETE/no failure; valid sample output.
   - Has TTA Proto including temporal flip, residual SSM, MLP probe improvements.
   - Similar visual/BirdNET/0.949-style lineage; hold until v584 score because v584 already covers much of this family.
3. `kojimar/0-949-lb-birdclef-2026-prior-axis-rank-fusion` v1 — valid but low-upside
   - COMPLETE/no failure; valid sample output.
   - Explicit `[0.949 LB]` title but source shows a clean Karnakbayev PowerOptimization LB0.948 direct branch, not an obvious 0.960 structural jump.
4. `beicicc/bc26-cocoa-karnak-safe-may19` v1 — valid but EoS5-family low-upside
   - COMPLETE/no failure; valid sample output.
   - Model2/Model5 EoS-family blend; likely already covered by v574-v576.

Rejected / not next-reset direct candidates:

- `rabeya100x/birdclef-2026-0-947-lb`: status ERROR/no outputs.
- `aiaiaiooo/birdclef2026`: status ERROR/no `submission.csv`; training artifacts only.
- `muhammadsaadalvi/birdclef-2026-wildsound-v8`: status ERROR/no outputs.
- `mtoshidesu/birdclef-2026-visual-cpu-inference`: still ERROR/no outputs despite latest run.
- S118/S120 launchers: avoid due v583 hidden unhandled error unless attached source can be recovered/ported.

Decision:

- No new submission can be made while capped, and no monitor was started.
- If v584 scores above `0.949`, next work is repo-owned Visual/BirdNET confirmation/extraction.
- If v584 ties/drops/no-scores and no stronger source appears before next reset, first next-reset preflight should be FrankSunP 5-branch V4 TTA Fix; Meenal v20 is the backup if FrankSunP is disqualified or appears too risky.

## 2026-05-19 08:50 UTC execution update — v584 dropped, v585 next-reset monitor queued

Live result/state:

- `v584: Guarded direct Zhaorong Mtoshi Visual BirdNET replay`, ref `52800792`, scored **0.942**.
- Current best remains `0.949` from v574/v575/v576.
- 2026-05-19 UTC visible count remains `5`; day is capped.
- No stale v577/v578 scalar submitter was alive.

Lesson from v584:

- The Zhaorong/Mtoshi Visual BirdNET lane is full-source and schema-safe, but the direct replay underperformed materially (`0.942`).
- Visual/BirdNET/Mtoshi 0.949-style tweaks should now be treated as idea-mining only, not a confirmation lane.

Next-reset action:

- Added `scripts/submit_v585_franksunp_5branch_tta_fix_when_slot.py` for `franksunp/birdclef-2026-5-branch-v4-tta-fix` v1.
- Rationale: it remains the best distinct preflighted candidate after v584 dropped: full visible source, COMPLETE/no failure, valid schema, and a 5-stream rank-percentile ensemble using ProtoSSM, Tucker SED, Snowflake SED, CLAP, and BirdNET/graceful degradation.
- Caveat remains: previous tiny CLAP/Snowflake sidecars did not improve and the public sample output is compressed, so this is a high-diversity source probe, not a high-confidence 0.960 candidate.

Preflight before queueing:

- Source pull OK, version `1`, source length `103548`.
- Required markers present: `5-Branch Multistream Ensemble`, `ProtoSSM`, `Snowflake SED`, `CLAP`, `Graceful Degradation`, `test_soundscapes`, `sample_submission.csv`, `row_id`.
- Kernel COMPLETE/no failure.
- Required outputs present: `submission.csv`, `submission_birdnet.csv`, `submission_protossm.csv`, `submission_sed.csv`, `clap_preds.npy`, `snowflake_preds.npy`, `site_hour_prior_table.pkl`, `cache/perch_arrays.npz`, `cache/perch_meta.parquet`.
- Prior schema scan: valid sample-shaped output (`3 x 235`, unique row IDs, finite values around `[0.2707, 0.3998]`).

Monitor state:

- Started OpenClaw-managed background session `quiet-basil`, pid `18696`.
- The monitor attempted submission, hit the expected Kaggle daily cap (`15 hours from now`), and is sleeping `54120s` before retry.
- Expected behavior: submit v585 after next UTC reset unless it is already visible by then.

## 2026-05-19 09:45 UTC execution update — v585 monitor restarted

Live state:

- Latest submissions unchanged: v584 scored `0.942`; current best remains `0.949`; 2026-05-19 UTC count remains `5`/capped.
- The previously reported `quiet-basil` v585 monitor was no longer visible/alive at this check, and no `submit_v585` process was running.

Action:

- Restarted the v585 FrankSunP monitor as OpenClaw-managed session `mild-harbor`, pid `38214`.
- It re-ran full preflight successfully: source v1, kernel COMPLETE/no failure, all required output files present.
- It attempted submission and hit the expected daily cap (`14 hours from now`), then slept `50520s` before retry.

Decision:

- Keep `mild-harbor` as the active reset monitor. No additional submitters should be started while it is sleeping.

## 2026-05-19 11:45 UTC execution update — v585 moved to durable tmux monitor

Live state:

- Latest submissions unchanged: v584 scored `0.942`; current best remains `0.949`; 2026-05-19 UTC count is `5`/capped.
- No v585 submission is visible yet.
- The previous managed/nohup v585 monitors were not durable across turns (`mild-harbor` not found; PID-file process not alive), despite successful preflight and daily-cap sleep logs.

Durability fix:

- Started v585 in a detached tmux session: `birdclef-v585-reset`.
- The tmux monitor re-ran preflight successfully and attempted submission.
- It hit the expected Kaggle daily cap (`12 hours from now`) and is sleeping `43320s` before retry.
- Current tmux capture shows the process is alive and sleeping on cap.
- Use `tmux capture-pane -t birdclef-v585-reset -p | tail -80` to inspect it, and avoid starting duplicate v585 monitors while this session exists.

Fresh capped-source scan:

- Saved recent-date-run scan to `artifacts/public_kernels_20260519_frontier_candidates/date_run_all_20260519T1040Z.json`.
- Web search still found no explicit external `0.950+` public-claim source.
- New/high-signal recent entries observed: `meenalsinha/birdclef-2026-improved`, `beicicc/bc26-cocoa-v129-safe-may19`, `kojimar/0-949-lb-birdclef-2026-prior-axis-rank-fusion`, `solokop/birdclef-2026-perch-onnx`, and training-only HGNet kernels.
- No candidate from this scan clearly outranks the already queued FrankSunP 5-branch v585 before reset.

Decision:

- Keep only `birdclef-v585-reset` active for reset submission.
- Continue source mining while capped, but do not launch additional submitters unless a clearly stronger full-source 0.950+ candidate appears.

## 2026-05-19 12:52 UTC capped source audit — no duplicate submitter

Live state:

- Latest submissions unchanged: v580 `0.944`, v581 hidden timeout/no score, v582 `0.947`, v583 hidden unhandled error/no score, v584 `0.942`.
- Current confirmed best remains `0.949` from v574/v575/v576; target remains `0.960`.
- 2026-05-19 UTC visible count is `5`; slots capped.
- PR #245 is merged; active frontier PR is #246 (`MERGEABLE`, `BLOCKED`, open).
- No v577/v578 scalar submitter is active.
- Active v585 reset monitor remains detached tmux session `birdclef-v585-reset`, process alive, sleeping on daily cap after successful FrankSunP source/output preflight.

Fresh scan/audit artifacts:

- DATE_RUN scan saved: `artifacts/public_kernels_20260519_frontier_candidates/date_run_all_20260519T1240Z.json` (100 recent kernels).
- Source/output audit saved locally: `artifacts/public_kernels_20260519_frontier_candidates/source_audit_20260519T1245Z/summary.json`.
- Downloaded and schema-checked sample `submission.csv` outputs for Pilkwang 949, Aditya Exp019, and Shinak 260519; all are sample-shaped `3 x 235`, finite, unique row IDs.

Candidate findings:

- `pilkwang/949-birdclef-2026-rank-power-soundscape-fusion` v5: full source, COMPLETE/no failure, schema-safe output. It explicitly packages an `eos5_locked` blend: small yukiZ residual (`YUKIZ_BLEND_WEIGHT=0.0264`) + dominant rank-power branch (`PROTO_RANK_WEIGHT=0.600`, prior lambda `0.5`, rank-aware power `0.6`). Good explanation/source-extraction reference, but it claims/ties `0.949` and is close to the EoS5/PowerOptimization family, so it does not outrank v585 for the next slot.
- `adityaraghuvanshi999/birdclef-2026-exp019-rank-power-safe-validation` v1: full source, COMPLETE, schema-safe. It is mostly a transparent scalar explanation of Exp019 (`lambda_prior=0.5`, rank-aware power `0.6`) and explicitly says future work needs less-correlated branches. Treat as documentation/evidence, not a new slot.
- `yaroslavkholmirzayev/v6-0949-replay` v8: full source but currently RUNNING/no outputs. It is a `lambda_prior=0.65` microblend against an existing `0.949` axis, so monitor later but do not queue ahead of v585.
- `shinak0502/birdclef-260519` v2: full source, COMPLETE, schema-safe, no score claim; interesting upgraded joint site-hour/circular-hour prior plus TTA path, but it lacks FrankSunP's CLAP/Snowflake/BirdNET extra branches and has weaker score evidence.
- `mtoshidesu/notebookc6e90ae327` v1: full source and COMPLETE, but internet-enabled and same PowerOptimization-family sources; use for idea mining only unless repackaged and verified offline.
- `huydo170302/dsai1-internship-birdclef-2026`: training/no output, not submission-safe.
- `solokop/birdclef-2026-perch-onnx`: valid but baseline/low-upside.

Decision:

- Keep only v585 active for reset; do not start duplicate submitters while `birdclef-v585-reset` exists.
- If v585 drops/no-scores, next useful work is not another broad 0.949 replay; extract either (a) Pilkwang's residual-diversity packaging as a repo-owned confirmation/reference, or (b) Shinak's joint/circular site-hour prior as a small reviewable repo-owned patch after correlation/schema checks.
- Continue source monitoring for a true `0.950+` or structurally new full-source lane before spending another public slot.

## 2026-05-19 13:55 UTC capped scan — fresh top-feed candidates audited

Live state:

- Latest Kaggle submissions unchanged: v580 `0.944`, v581 hidden timeout/no score, v582 `0.947`, v583 hidden unhandled error/no score, v584 `0.942`.
- Current confirmed best remains `0.949` from v574/v575/v576; target remains `0.960`.
- 2026-05-19 UTC visible count remains `5`; slots capped.
- PR #245 is merged; active frontier PR #246 remains open, mergeable, and blocked.
- No v577/v578 scalar submitter is active.
- Active v585 reset monitor remains detached tmux session `birdclef-v585-reset`; process alive and sleeping on daily cap after successful FrankSunP source/output preflight.

Fresh scan artifacts:

- DATE_RUN scan saved: `artifacts/public_kernels_20260519_frontier_candidates/date_run_all_20260519T1345Z.json`.
- Fresh top-feed source/output audit saved locally: `artifacts/public_kernels_20260519_frontier_candidates/source_audit_20260519T1345Z/summary.json`.
- External web search found no explicit `0.950`/`0.951`/`0.96` BirdCLEF source claim.

Fresh candidate findings:

- `rajnish1419kumar/birdclef-2026-rankpower-safe-candidate` v1: full source, COMPLETE, schema-safe sample output; source length and outputs match Pilkwang 949 Rank-Power Soundscape Fusion. Treat as duplicate/clone of the 0.949-family reference, not a new slot.
- `claudedevore/birdclef-2026-r0946-a2prime-nfnet-submit` v6: full source, hidden-path markers present, but RUNNING/no outputs. It is an A2Prime/NFNet R0946 lineage and v581 already timed out on a similar A2Prime/NFNet replay, so monitor only; do not queue ahead of v585.
- `yaroslavkholmirzayev/v6-0949-replay` v8: now COMPLETE, but `submission.csv` is invalid: 243 rows including train rows, empty numeric cells (`finite_bad=56862`). Reject for direct submission until fixed at source.
- `adkasd/birdclef-2026-exp019-fast` v2: COMPLETE/schema-safe, but another Exp019 rank-power scalar path; no new signal beyond prior 0.949-family audit.
- `chaneyma/bc26-gate-v67-eos5-postprocess` v1: COMPLETE with many useful intermediate outputs, but primary `submission.csv` sample is constant `0.66666675` across all classes. Reject direct replay; mine only if a specific intermediate output can be source-extracted and schema-verified.

Decision:

- Keep v585 as the sole active reset submitter.
- Do not spend a slot on Rajnish/Pilkwang clones, Exp019-fast clones, Yaroslav invalid output, or Chaney constant-output primary file.
- Monitor Claude A2Prime/NFNet only for completion/output, but it needs stronger evidence and timeout mitigation before any slot.
- If v585 fails/drops, next best work remains repo-owned extraction rather than another direct 0.949-family replay.

## 2026-05-19 14:58 UTC capped scan — A2Prime/EffV2S vs NFNet fallback triage

Live state:

- Latest Kaggle submissions unchanged: v580 `0.944`, v581 hidden timeout/no score, v582 `0.947`, v583 hidden unhandled error/no score, v584 `0.942`.
- Current confirmed best remains `0.949` from v574/v575/v576; target remains `0.960`.
- 2026-05-19 UTC visible count remains `5`; slots capped.
- PR #245 is merged; active frontier PR #246 remains open, mergeable, and blocked.
- No v577/v578 scalar submitter is active.
- Active v585 reset monitor remains detached tmux session `birdclef-v585-reset`; process alive and sleeping on daily cap after successful FrankSunP source/output preflight.

Fresh scan artifacts:

- DATE_RUN scan saved: `artifacts/public_kernels_20260519_frontier_candidates/date_run_all_20260519T1445Z.json`.
- A2Prime/source-output audit saved locally: `artifacts/public_kernels_20260519_frontier_candidates/source_audit_20260519T1445Z/summary.json`.
- External web search again found no explicit `0.950`/`0.951`/`0.96` BirdCLEF source claim.

Candidate findings:

- `claudedevore/birdclef-2026-r0946-a2prime-nfnet-submit` v6 has now completed with outputs. Primary `submission.csv` is sample-shaped `3 x 235`, finite, unique row IDs. It also emits full train-shaped branch files and summaries. NFNet branch is active with 5 folds CPU, but sanity top-5 hit rate is only `0.30`; Proto/NFNet rank correlation is `0.169`. Treat as source-valid but weaker than EffV2S and timeout-risk because v581 already no-scored on A2Prime/NFNet.
- `claudedevore/birdclef-2026-r0946-a2prime-effv2s-submit` v5 is COMPLETE/schema-safe. EffV2S branch has 4 folds CPU, sanity top-5 hit rate `0.55`, and very low Proto/EffV2S rank correlation `0.053`, making it a more plausible diversity extraction candidate than NFNet despite its R0946 title. It is still a public946-family candidate, so do not queue ahead of v585, but if v585 drops/no-scores and no stronger 0.950+ source appears, this is the best concrete repo-owned extraction target from the A2Prime family.
- `rajnish1419kumar/birdclef-2026-rankpower-nfnet-selective` v1 is COMPLETE/schema-safe and combines Pilkwang 949 RankPower with a selective NFNet sidecar. Primary `submission.csv` is sample-shaped and finite, but its source is mostly Pilkwang 949 plus NFNet graft; intermediate `submission_nfnet_selective.csv` is only 36 rows on public run and starts with train IDs. Treat as idea-mining only unless source is ported and forced to hidden/test rows.
- `aiaiaiooo/birdclef2026` v8 is RUNNING/no outputs, no attached datasets in metadata despite hidden-path markers. Monitor only; not slot-ready.

Concrete fallback ranking behind v585:

1. If v585 improves: port/confirm FrankSunP 5-branch repo-owned immediately.
2. If v585 drops/no-scores and a true `0.950+` source has not appeared: extract Claude A2Prime EffV2S branch into a repo-owned, timeout-controlled variant before any direct replay. Rationale: low rank correlation (`0.053`) and better sanity hit rate (`0.55`) than NFNet.
3. Mine Rajnish RankPower+NFNet selective only after fixing hidden/test-row selection, because its intermediate selective branch produced train-row outputs on public run.
4. Avoid direct Yaroslav/Chaney/Rajnish-clone replays already rejected by schema or duplication checks.

Decision:

- Keep v585 as the sole active reset submitter.
- No additional submission monitor launched.
- Next meaningful non-slot work should be a repo-owned EffV2S extraction scaffold if the cap persists and no better source appears.

## 2026-05-19 15:58 UTC repo-owned EffV2S fallback scaffold prepared

Live state:

- Latest Kaggle submissions unchanged: v580 `0.944`, v581 hidden timeout/no score, v582 `0.947`, v583 hidden unhandled error/no score, v584 `0.942`.
- Current confirmed best remains `0.949` from v574/v575/v576; target remains `0.960`.
- 2026-05-19 UTC visible count remains `5`; slots capped.
- PR #245 is merged; active frontier PR #246 remains open, mergeable, and blocked.
- No v577/v578 scalar submitter is active.
- Active v585 reset monitor remains detached tmux session `birdclef-v585-reset`; process alive and sleeping on daily cap after successful FrankSunP source/output preflight.

Action taken while capped:

- Prepared repo-owned fallback scaffold `kaggle-kernels/v586-a2prime-effv2s-extraction/` from `claudedevore/birdclef-2026-r0946-a2prime-effv2s-submit` v5.
- Added metadata for private repo-owned Kaggle kernel `yourslewis/bc26-v586-a2prime-effv2s-extraction` with internet disabled and the same required dataset/kernel/model sources.
- Inserted a leading notebook policy cell documenting that this is fallback-only after v585 result, and why EffV2S outranks NFNet as the A2Prime extraction candidate.
- Added push helper `scripts/push_v586_a2prime_effv2s_extraction.py`. This only pushes the private kernel; it does not submit. Do not run while `birdclef-v585-reset` exists and owns the next reset slot.

Validation:

- Notebook JSON parses; generated notebook has 39 cells.
- Metadata validation passed; expected EffV2S dataset source `baiyuby/birdclef2026-distill-models` and model sources are present.
- Push helper compiles.
- v585 submitter still compiles.
- `git diff --check` and `git_maint.py hygiene` passed.

Decision:

- Keep v585 as the sole active reset submitter.
- If v585 improves, abandon v586 and port/confirm FrankSunP.
- If v585 drops/no-scores and no true `0.950+` source appears, next safe step is to run the v586 push helper, verify COMPLETE/no failure and `submission.csv`, then decide whether to submit a repo-owned EffV2S extraction candidate.

## 2026-05-19 16:50 UTC capped re-scan — no source outranks v585/v586 queue

Live state:

- Latest Kaggle submissions remain v580 `0.944`, v581 timeout/no score, v582 `0.947`, v583 hidden unhandled error/no score, v584 `0.942`.
- Current confirmed best remains `0.949`; target remains `0.960`.
- 2026-05-19 UTC is capped with `5` visible submissions.
- No v577/v578 scalar submitter is active.
- `birdclef-v585-reset` remains the only active submitter and is sleeping on the reset slot after source/output preflight.

Fresh artifacts:

- DATE_RUN scan: `artifacts/public_kernels_20260519_frontier_candidates/date_run_all_20260519T1648Z.json`.
- Source/output audit: `artifacts/public_kernels_20260519_frontier_candidates/source_audit_20260519T1649Z/summary.json`.

Candidate updates:

- `yaroslavkholmirzayev/v6-0949-replay`: rerun did not fix direct-submission safety. Primary `submission.csv` has 243 rows, train rows, and empty numeric cells. Sample-shaped side outputs are Karnakbayev/EoS5-family and low-upside.
- `meenalsinha/birdclef-2026-improved`: rerun still emits train-row dry-run primary output; prior direct replay already failed hidden behavior. Reject for direct slot.
- `samejimatink0/birdclef-2026-visual-cpu-inference`: COMPLETE/finite but primary output is train-row dry-run output; title says visual but available markers look like ProtoSSM/SED/BirdNET. Mine only if source reveals a portable distinct branch.
- `aiaiaiooo/birdclef2026`: source has hidden-path markers but no outputs; not preflight-safe.

Queue decision:

1. Preserve v585 FrankSunP 5-branch as the sole reset submitter.
2. If v585 improves, immediately port/confirm FrankSunP repo-owned.
3. If v585 drops/no-scores and no true `0.950+` source appears, use prepared repo-owned v586 EffV2S extraction path rather than another direct 0.949-family replay.
4. Continue rejecting train-row/constant/empty-cell primary outputs even if side outputs look sample-shaped.

## 2026-05-19 17:49 UTC queue update — EoS.6 running, recheck before reset

Live state:

- Latest submissions unchanged: v580 `0.944`, v581 timeout/no score, v582 `0.947`, v583 hidden error/no score, v584 `0.942`; best remains `0.949`; target remains `0.960`.
- 2026-05-19 UTC remains capped at `5` submissions.
- `birdclef-v585-reset` remains alive and is the only active submitter.

Fresh artifacts:

- DATE_RUN scan: `artifacts/public_kernels_20260519_frontier_candidates/date_run_all_20260519T1747Z.json`.
- EoS.6 audit: `artifacts/public_kernels_20260519_frontier_candidates/source_audit_20260519T1747Z_eos6/summary.json`.
- Parsed EoS.6 source cells: `artifacts/public_kernels_20260519_frontier_candidates/source_audit_20260519T1747Z_eos6/nina2025__birdclef-2026-eos-6-silver-zone.source.cells.py`.

Candidate queue:

1. `nina2025/birdclef-2026-eos-6-silver-zone` v9 — **recheck before reset**. Source is a direct EoS successor and therefore high-signal, but the public session is currently `RUNNING` with no outputs. Active config blends `Model_21`/`Model_73`/`Model_74` at `0.032/0.967/0.001`; markdown says v7 timed out and v9 tries to run SED once, but source has a possible `task1`/`task` mismatch around the early Model_1 guard. Do not submit unless COMPLETE/no failure and primary `submission.csv` is schema-safe.
2. v585 FrankSunP 5-branch remains the current reset-slot owner unless EoS.6 completes safely first and clearly outranks it.
3. v586 repo-owned EffV2S remains fallback only if v585 drops/no-scores and no stronger source appears.

Rejected/low-priority this scan:

- `damianleandrotamburi/20260329-birdclef` v70 — no outputs and no 0.949+/EoS/PowerOptimization evidence.
- Web search found no explicit `0.950`/`0.951`/`0.96` BirdCLEF code claim.

Decision:

- No Kaggle push/submission while capped.
- Next run should first check whether EoS.6 v9 completed. If complete and output-safe before v585 submits, consider pausing/killing v585 and queueing guarded EoS.6 replay; otherwise preserve v585.

## 2026-05-19 17:52 UTC heartbeat recheck — EoS.6 currently unavailable

- Live LB/submission state unchanged and capped.
- `nina2025/birdclef-2026-eos-6-silver-zone` now fails source pull/get (`404 Not Found`) and has no outputs despite stale session-status `RUNNING` response. It should not displace v585 unless it becomes pullable again and completes with valid primary `submission.csv`.
- v585 FrankSunP remains the reset-slot owner.

## 2026-05-19 18:50 UTC queue update — keep v585, reject EoS.6 primary

Live state:

- Latest submissions unchanged: v580 `0.944`, v581 timeout/no score, v582 `0.947`, v583 hidden error/no score, v584 `0.942`; best remains `0.949`; target remains `0.960`.
- 2026-05-19 UTC remains capped at `5` submissions.
- `birdclef-v585-reset` remains alive and is the only active submitter.

New artifacts:

- Fresh DATE_RUN scan: `artifacts/public_kernels_20260519_frontier_candidates/date_run_all_20260519T1847Z.json`.
- EoS.6 output schema audit: `artifacts/public_kernels_20260519_frontier_candidates/eos6_outputs_20260519T1847Z/summary.json`.
- NFNet/lprior source-output audit: `artifacts/public_kernels_20260519_frontier_candidates/source_audit_20260519T1847Z_new/summary.json`.

Candidate decisions:

1. `nina2025/birdclef-2026-eos-6-sz` v9 — reject direct replay. It is COMPLETE/no failure and source-pullable now, but primary `submission.csv` is invalid (`243` rows, train rows, empty numeric cells). Side outputs are sample-shaped but not the configured competition file.
2. `nicolasschuldt/nfnet-lprior075` v1 — schema-safe primary output, but mostly EoS5/RankPower family with `lambda_prior=0.75` and tiny NFNet selective graft. The NFNet selective output is train-row-only on public sample. Keep as idea-mining/fallback; it does not outrank v585 or repo-owned v586 EffV2S under the 0.96 target.
3. v585 FrankSunP remains reset-slot owner.
4. v586 EffV2S remains the preferred prepared fallback if v585 drops/no-scores and no stronger 0.950+ source appears.

Decision:

- No new Kaggle push/submission while capped.
- Do not start duplicate submitters.
- Next run should check whether v585 submitted/scored; if still capped, continue source scan and keep v586 ready.

## 2026-05-19 19:50 UTC queue update — v585 still best reset-slot owner

Live state:

- Latest submissions unchanged: v580 `0.944`, v581 timeout/no score, v582 `0.947`, v583 hidden error/no score, v584 `0.942`; best remains `0.949`; target remains `0.960`.
- 2026-05-19 UTC remains capped at `5` submissions.
- PR #245 and PR #246 are merged; new work continues on branch `feature/birdclef-096-frontier-v585-hold-20260519`.
- `birdclef-v585-reset` remains alive and is the only active submitter.

New artifacts:

- Fresh DATE_RUN scan: `artifacts/public_kernels_20260519_frontier_candidates/date_run_all_20260519T1947Z.json`.
- Fresh top-feed audit: `artifacts/public_kernels_20260519_frontier_candidates/source_audit_20260519T1947Z_top/summary.json`.

Candidate decisions:

1. `meenalsinha/birdclef-2026-improved` v22 — reject direct replay. Fresh rerun still outputs train rows for primary `submission.csv`; source is a lambda-prior scalar (`exp_067 v6_prior065`) rather than a new 0.96 structure.
2. `nina2025/birdclef-2026-eos-6-sz` v10 — reject direct replay. Markdown now says v8 scored `0.949`, but v10 primary `submission.csv` is still invalid (`243` rows / train rows / empty cells). Side output `subm_73.csv` is sample-shaped but same 0.949-family branch.
3. `evgendvorkin/birdclef-baseline` v33 — reject; train-row baseline output and no high-score/source evidence.
4. `nicolasschuldt/nfnet-lprior075` v1 — keep as idea-mining/fallback only; schema-safe primary but mostly RankPower/EoS5 scalar with train-row NFNet selective intermediate.
5. v585 FrankSunP remains reset-slot owner.
6. v586 repo-owned EffV2S remains preferred prepared fallback if v585 drops/no-scores and no stronger 0.950+ source appears.

Decision:

- No new Kaggle push/submission while capped.
- Do not start duplicate submitters.
- Next run should check whether v585 submitted/scored; if still capped, continue source scan and keep v586 ready.

## 2026-05-19 20:50 UTC queue update — no safer source found before reset

Live state:

- Latest submissions unchanged: v580 `0.944`, v581 timeout/no score, v582 `0.947`, v583 hidden error/no score, v584 `0.942`; best remains `0.949`; target remains `0.960`.
- 2026-05-19 UTC remains capped at `5` submissions.
- PR #247 is open/blocked; `birdclef-v585-reset` remains alive and is the only active submitter.

New artifacts:

- Fresh DATE_RUN scan: `artifacts/public_kernels_20260519_frontier_candidates/date_run_all_20260519T2047Z.json`.
- Fresh top-feed audit: `artifacts/public_kernels_20260519_frontier_candidates/source_audit_20260519T2047Z_top/summary.json`.

Candidate decisions:

1. `evgendvorkin/birdclef-baseline` v34 — reject; train-row primary output and no high-score/source evidence.
2. `meenalsinha/birdclef-2026-improved` v22 — still reject; primary and branch outputs are train-row-only; scalar lambda-prior lane.
3. `nina2025/birdclef-2026-eos-6-sz` v10 — still reject; primary `submission.csv` invalid despite sample-shaped side output.
4. No fresh 0.950+/0.96 source with safe primary output found.
5. v585 FrankSunP remains reset-slot owner.
6. v586 repo-owned EffV2S remains preferred prepared fallback if v585 drops/no-scores.

Decision:

- No new Kaggle push/submission while capped.
- Do not start duplicate submitters.
- Next run should check whether v585 submitted/scored after UTC reset; if still pending/capped, continue source scan.

## 2026-05-19 21:50 UTC queue update — still no source to displace v585

Live state:

- Latest submissions unchanged: v580 `0.944`, v581 timeout/no score, v582 `0.947`, v583 hidden error/no score, v584 `0.942`; best remains `0.949`; target remains `0.960`.
- 2026-05-19 UTC remains capped at `5` submissions; v585 is not visible yet.
- PR #247 is open/blocked; `birdclef-v585-reset` remains alive and is the only active submitter.

New artifacts:

- Fresh DATE_RUN scan: `artifacts/public_kernels_20260519_frontier_candidates/date_run_all_20260519T2147Z.json`.
- Fresh top-feed audit: `artifacts/public_kernels_20260519_frontier_candidates/source_audit_20260519T2147Z_top/summary.json`.

Candidate decisions:

1. `huydo170302/dsai1-internship-birdclef-2026` v8 — reject; EDA/training baseline, no outputs, no submission path, no high-score evidence.
2. `evgendvorkin/birdclef-baseline` v34 — reject; train-row primary output and no high-score/source evidence.
3. `meenalsinha/birdclef-2026-improved` v22 — reject; train-row-only outputs and scalar lambda-prior lane.
4. `nina2025/birdclef-2026-eos-6-sz` v10 — reject; invalid primary `submission.csv`; side output not acceptable as configured competition file.
5. v585 FrankSunP remains reset-slot owner.
6. v586 repo-owned EffV2S remains preferred prepared fallback if v585 drops/no-scores.

Decision:

- No new Kaggle push/submission while capped.
- Do not start duplicate submitters.
- Next run should check whether v585 submitted/scored after UTC reset; if still pending/capped, continue source scan.

## 2026-05-19 22:50 UTC queue update — Pilkwang prior-field safe but not enough to replace v585

Live state:

- Latest submissions unchanged: v580 `0.944`, v581 timeout/no score, v582 `0.947`, v583 hidden error/no score, v584 `0.942`; best remains `0.949`; target remains `0.960`.
- 2026-05-19 UTC remains capped at `5` submissions; v585 is not visible yet.
- PR #247 is open/blocked; `birdclef-v585-reset` remains alive and is the only active submitter.

New artifacts:

- Fresh DATE_RUN scan: `artifacts/public_kernels_20260519_frontier_candidates/date_run_all_20260519T2247Z.json`.
- Fresh top-feed audit: `artifacts/public_kernels_20260519_frontier_candidates/source_audit_20260519T2247Z_top/summary.json`.

Candidate decisions:

1. `meenalsinha/birdclef-2026-improved` v23 — running/no outputs; source remains lambda-prior/RankPower scalar tuning, so do not replace v585.
2. `jguevarag/07-optimal-sed-training` v4 — training notebook, no outputs; not a submission candidate.
3. `pilkwang/949-birdclef-2026-acoustic-prior-field-fusion` v6 — schema-safe primary and useful idea-mining, but explicitly 0.949-family prior-field/RankPower; keep as fallback/analysis, not reset-slot owner under 0.960 target.
4. `adarsh5harma/birdclef-2026-v66-phase1-integrated` v1 — invalid primary `submission.csv`; reject direct replay.
5. `muhammadsaadalvi/birdclef-2026-wildsound-v8` v68 — no outputs; not slot-ready.
6. v585 FrankSunP remains reset-slot owner.
7. v586 repo-owned EffV2S remains preferred prepared fallback if v585 drops/no-scores.

Decision:

- No new Kaggle push/submission while capped.
- Do not start duplicate submitters.
- Next run should check whether v585 submitted/scored after UTC reset; if still pending/capped, continue source scan.

## 2026-05-19 23:30 UTC broad-score 0.96 audit — no reset-slot replacement

User requested another research round targeting `0.960`. This pass broadened beyond fresh DATE_RUN feed into Kaggle score/vote/search surfaces and older high-vote candidates.

Artifacts:

- Broad search: `artifacts/public_kernels_20260519_frontier_candidates/broad_score_search_20260519T2323Z.json`.
- Broad-score audit: `artifacts/public_kernels_20260519_frontier_candidates/source_audit_20260519T2323Z_broadscore/summary.json`.
- Legacy/diverse audit: `artifacts/public_kernels_20260519_frontier_candidates/source_audit_20260519T2335Z_legacydiverse/summary.json`.

Current queue remains:

1. v585 FrankSunP 5-branch V4 TTA Fix — sole reset-slot owner, monitor alive.
2. v586 repo-owned A2Prime/EffV2S extraction — prepared fallback only if v585 drops/no-scores and no stronger true `0.950+` source appears.

Candidate decisions:

- `ulyanovantonamaranta/birdclef-2026-gate-fake008-head0015` — schema-safe, but mostly public ProtoSSM/SED gate family plus net ~1.5% train-audio-head rank contribution; useful idea-mining only.
- `cliff376/bc26-public-gate-combo-pc010-v2` — schema-safe, public-gate Proto/SED variant; no evidence it clears the 0.949 plateau.
- `raunakdey07/birdclef-2026-multi-model-ensemble` — schema-safe and has sonotype mirroring / rare-class threshold ideas, but no direct 0.950+ evidence.
- `marynaborovska/birdclef-26-two-pass-ssm-advanced-pp` — promising source architecture, but no outputs; not direct-submit-safe.
- `aminmahmoudalifayed/birdclef-2026` — invalid/empty primary submission; reject.
- EoS/Karnak/RankPower legacy candidates (`anthonytherrien`, `beicicc`, `kijiang`, `karnakbaevarthur`, `nicolasschuldt/eos5-meta`, `apachikoff/eos-5`, `starsdaisuki`, `adityaraghuvanshi999`) are saturated 0.949-family scalar/blend variants and should not consume the next slot.

Next action:

- First check v585 visibility/score after UTC reset.
- If v585 improves, port/confirm FrankSunP.
- If v585 drops/no-scores, push/verify v586 EffV2S before considering any direct EoS/Karnak/RankPower replay.
- Separately mine Raunak/Maryna ideas for future repo-owned structural work, not immediate submissions.

## 2026-05-20 00:06 UTC — v585 submitted, wait for score before next slot

- v585 FrankSunP 5-branch V4 TTA Fix submitted after reset: ref `52831360`, date `2026-05-20T00:01:09.75Z`, status `pending`.
- 2026-05-20 UTC has `1` visible submission so far.
- `birdclef-v585-reset` exited after the successful submission.
- Current confirmed best remains `0.949` until v585 scores.
- Fresh pre-reset scan: `artifacts/public_kernels_20260519_frontier_candidates/date_run_all_20260519T2347Z.json`; no newly surfaced candidate outranks the existing v585/v586 queue.

Queue policy now:

1. Do not submit more until v585 result is known, unless explicitly instructed.
2. If v585 improves, port/confirm FrankSunP repo-owned.
3. If v585 drops/no-scores, use prepared repo-owned v586 A2Prime/EffV2S extraction before EoS/Karnak/RankPower direct clones.

## 2026-05-20 00:48 UTC — v585 pending; Zeyad schema-safe but 0.949-family

Live state:

- v585 ref `52831360` remains pending/no score.
- Current confirmed best remains `0.949`.
- 2026-05-20 UTC count is `1`; no active submitter processes.

Artifacts:

- Fresh scan: `artifacts/public_kernels_20260520_frontier_candidates/scan_20260520T0047Z.json`.
- Fresh audit: `artifacts/public_kernels_20260520_frontier_candidates/source_audit_20260520T0047Z_top/summary.json`.

Candidate decisions:

1. `mtoshidesu/notebookc6e90ae327` v3 — pullable but outputless; not direct-submit-safe.
2. `zeyadmohamadezzat/birdclef-2026-proto-fusion-and-temporal-flip` v3 — schema-safe primary output, but source is explicitly consolidated `0.949` exp019/EoS4/Karnakbayev PowerOptimization with small BirdNET/public-CNN branches (`0.04/0.04`). Keep as idea-mining only.
3. `meenalsinha/birdclef-2026-improved` v23 — outputs exist but primary `submission.csv` is train-row dry-run output; reject direct replay.

Queue remains unchanged:

- Wait for v585 result.
- If v585 improves, port/confirm FrankSunP.
- If v585 drops/no-scores, push/verify prepared repo-owned v586 A2Prime/EffV2S before direct EoS/Karnak/RankPower-family clones.

## 2026-05-20 02:05 UTC update — v585 rejected, v586 EffV2S active fallback

- v585 FrankSunP 5-branch V4 TTA Fix replay scored `0.922`; reject this lane and do not port/confirm.
- Active slot candidate is repo-owned `yourslewis/bc26-v586-a2prime-effv2s-extraction-r2` v1, specifically alternate output `submission_a2_effv2s_w08.csv` (not default `submission.csv`, which is `base_3way`).
- Guarded monitor: `scripts/submit_v586_a2prime_effv2s_w08_when_ready.py`, log `logs/v586_a2prime_effv2s_w08_submitter.log`, PID `96513` at launch. It waits for COMPLETE and validates output/summary before submit.
- Fresh source queue notes from `source_audit_20260520T0158Z_top/summary.json`: Samejima Visual CPU remains visual/BirdNET idea-mining; Pilkwang/Adarsh prior-field forks are `0.949` RankPower-family; Mtoshi V6 is ERROR. None outrank v586.

## 2026-05-20 02:55 UTC update — v586 filename fix, v587 S121 pending

- v586 r2 v1 completed but alternate-output submission failed because BirdCLEF requires `submission.csv`. Notebook patched so `a2_effv2s_w08` becomes `submission.csv`; r2 version 2 pushed and monitor PID `12539` is waiting for COMPLETE/output preflight.
- v587 direct replay submitted: `itshyao/birdclef-2026-s121-s114-g116-f1-delta` v1, ref `52835586`, pending. Rationale: protected HGNet/G116 delta sidecar on S114 anchor is a more structural candidate than fresh RankPower/PriorField clones; public dry-run has row-id fallback and S121 final diagnostics.
- Fresh 02:47 scan rejects/queues: Mtoshi v4 now COMPLETE but still Karnak/RankPower/visual family; Koushik Pantanal ERROR; Qiuzi HGNet training artifacts only; Rikuter v6/prior-field reproductions are 0.949 family. Continue scanning for true non-saturated structural sources.

## 2026-05-20 03:50 UTC update — v586/v587/v588 pending, reserve final slot

- v586 repo-owned A2Prime/EffV2S w08 submitted as ref `52835975` after r2 v2 completed and made w08 the normal `submission.csv`.
- v587 Itshyao S121 S114+G116 protected delta remains pending, ref `52835586`.
- v588 Itshyao S122 S114+G123 EfficientNetV2-B0 protected delta submitted, ref `52836864`.
- 2026-05-20 slots used: `4/5`; reserve final slot pending scores or a genuinely stronger source. Fresh 03:47 scan did not surface a stronger non-saturated candidate besides the S122 sibling; EoS6/Kijiang/Pilkwang/Rikuter remain 0.949-family, Samejima remains Visual/BirdNET family.

## 2026-05-20 04:50 UTC update — capped after v589 final-slot S124 rankblend

- Results landed: v586 A2Prime/EffV2S w08 `0.941` (reject lane), v587 S121/G116 `0.949` tie, v588 S122/G123 `0.949` tie. Best remains `0.949`.
- Final 2026-05-20 slot used on v589 Itshyao S124 S114+G124 rankblend, ref `52838266`, pending. This is the last S114+G-sidecar sibling worth trying today because it changes from delta to protected rank blending with a G124 EfficientNetV2-S 2025-pretrained pseudo sidecar.
- Daily cap now `5/5`. If v589 ties/drops, deprioritize S114+G116/G123/G124 sidecar siblings and resume 0.96 source search / training-lane prep for next reset.

## 2026-05-20 05:55 UTC update — capped; next lead is HGNet training artifact

- v589 remains pending and daily slots are `5/5`; no active submit monitors.
- Fresh scan/audit: `scan_20260520T0547Z.json`, `source_audit_20260520T0547Z_top/summary.json`.
- Best next non-saturated lead: Qiuzi `hgnetv2-b0-training-e2c7fc`, which completed fold artifacts and logged rank-pred validation AUC `0.9586928494392578`. It is not direct-submit-safe because it is training-only, but it may support a repo-owned HGNet sidecar/inference candidate next reset.
- Deprioritize further S114+G116/G123/G124 siblings unless v589 improves; v587/v588 tied and S123/S124-style variants are likely plateau-safe rather than lift-producing.

### v589 result — 2026-05-20 05:50 UTC

- v589 S124 G124 rankblend scored `0.949`, tie only. Daily 5/5 set is complete: `0.922`, `0.941`, `0.949`, `0.949`, `0.949`.
- Stop S114+G-sidecar sibling submissions for now. Next reset should focus on new non-saturated training/source directions, especially Qiuzi HGNet artifacts or another independently validated model-zoo sidecar.

## 2026-05-20 06:47 UTC update — HGNet downgraded to source recipe, no ready slot candidate

- Slots remain capped `5/5`; v589 scored `0.949`. Confirmed current best `0.949`.
- Fresh scan/audit paths: `scan_20260520T0647Z.json`, `source_audit_20260520T0647Z_top/summary.json`, `hgnet_lead_20260520T0647Z/`.
- Qiuzi HGNet is promising as a **recipe** (fold0 best validation AUC `0.96378`, HGNetV2-B0, EMA, mixup, train-audio + train-soundscape labels), but not a ready submission because the session is `CANCEL_ACKNOWLEDGED` and only fold0 output URL is exposed.
- Samejima HGNet inference is sample-invalid/all-NaN in dry-run and has unresolved weights-dataset availability. Patch/resolution required before any direct or repo-owned submission.
- OmModi dual-resolution EffV2/temporal-transformer notebook rejected: epoch-1 checkpoint AUC `0.5124`, all-zero fallback output.
- Next reset candidate queue: (1) continue source frontier scan for a genuinely new >0.949 public lineage; (2) resolve HGNet weights/data availability and build a sample-safe repo-owned inference only if complete artifacts are available; (3) only then consider a guarded HGNet/EoS blend as a structural diagnostic.

## 2026-05-20 07:47 UTC update — new branch after PR #248 merge

- PR #248 merged; continuing on `feature/birdclef-096-hgnet-nfnet-triage-20260520`.
- Slots remain capped `5/5`; current best unchanged at `0.949`.
- Fresh artifacts: `scan_20260520T0747Z.json`, `source_audit_20260520T0747Z_top/summary.json`.
- Qiuzi `hgnetv2-b0-training-distill` is not a candidate: cancelled after epoch 0 with val_score `0.54157`, no submission artifact.
- Henry `bc2026-rankpower-nfnet-v80` is schema-valid and hidden-path plausible, with a complete `submission.csv` and NFNet safety check. However, source self-identifies as EoS5/Sunderekkiz/Pilkwang rank-power 0.949-family (`YUKIZ_BLEND_WEIGHT=0.0264`, `PROTO_RANK_WEIGHT=0.600`), so it is a backup/diagnostic next-reset candidate only.
- Candidate queue: (1) continue fresh source scan for non-saturated 0.95/0.96 claims; (2) resolve HGNet complete artifact/inference path; (3) if no stronger lane exists near reset, consider Henry v80 guarded direct replay as one low-upside structural diagnostic rather than more S114+G siblings.

## 2026-05-20 08:48 UTC update — v590 backup prepared, still capped

- Slots remain capped `5/5`; best unchanged at `0.949`; PR #249 open/BLOCKED.
- Fresh artifacts: `scan_20260520T0848Z.json`, `source_audit_20260520T0848Z_top/summary.json`.
- Top-feed triage: aiaiaiooo all-zero fallback, Mtoshi notebook ERROR, Qiuzi distill COMPLETE/no outputs, Mtoshi S124 duplicate running, PriorField/EoS6/Kijiang safe writers saturated; Claude V6 replay invalid 243-row/all-NaN output.
- Prepared next-reset backup script `scripts/submit_v590_rajnish_zeyad_proto_temporal_safe_when_ready.py` for Rajnish/Zeyad Proto Temporal Safe. Preflight-only passed (source v1, COMPLETE, required outputs, finite 3x235 sample-shaped `submission.csv`).
- Candidate queue now: (1) continue source scan for true non-saturated 0.95/0.96 claim; (2) resolve HGNet complete artifacts/inference; (3) if no stronger lane appears by reset, v590 Zeyad Proto Temporal Safe is the best backup diagnostic; (4) Henry NFNet v80 is second backup; avoid more S114+G siblings.

## 2026-05-20 09:48 UTC update — v590 still leads backup queue

- Slots capped `5/5`; best unchanged `0.949`; PR #249 open/BLOCKED.
- Fresh artifacts: `scan_20260520T0948Z.json`, `source_audit_20260520T0948Z_top/summary.json`.
- New audit did not find a stronger 0.96 lane: Bugra HDMR has no outputs; Henry v81/v82 are valid but low-upside NFNet parameter siblings; Nina EoS6 v12 primary output invalid (243 rows/all NaN); Mtoshi S124 is a duplicate tied lane; Pilkwang PriorField remains saturated.
- Original Zeyad v4 matches the prepared Rajnish safe-writer v590 branch set and validates the candidate lineage. Keep `scripts/submit_v590_rajnish_zeyad_proto_temporal_safe_when_ready.py` as first backup at reset only if no stronger source appears.
- Backup order: v590 Zeyad/Rajnish Proto Temporal Safe; Henry v82/v81/v80 NFNet diagnostics; then no-submit/continue scanning rather than more S114+G or PriorField/EoS6 replays.

## 2026-05-20 10:48 UTC update — no queue change

- Slots still capped `5/5`; best remains `0.949`; no active submitter.
- Fresh artifacts: `scan_20260520T1048Z.json`, `source_audit_20260520T1048Z_top/summary.json`.
- Re-ran v590 preflight-only: still passes and correctly does not submit while capped.
- Pilkwang PriorField v11 fixed sample output validity (`3x235`, finite, min `0.46079`, max `0.53817`) and includes BirdNET v24 side output, but it is still saturated 0.949 PriorField/EoS6/Karnak-family. It does not displace v590.
- Nina EoS6 v12 remains invalid primary output (`243x235`, all NaN).
- Queue unchanged: (1) true new 0.95/0.96 source if found, (2) v590 Zeyad/Rajnish Proto Temporal Safe as first backup diagnostic, (3) Henry NFNet v82/v81/v80, (4) Pilkwang v11/PriorField only as last-resort valid saturated fallback.

## 2026-05-20 11:48 UTC update — no new candidate above v590

- Slots still capped `5/5`; best remains `0.949`; no active submitter.
- Fresh artifacts: `scan_20260520T1148Z.json`, `source_audit_20260520T1148Z_top/summary.json`.
- Fresh top-feed triage: Haridoss custom model is running/no submission artifact and no high-LB evidence; Meenal v23 and Samejima Visual v7 produce train/fallback-shaped `240x235` primary outputs; Jacques minimal is constant-probability baseline; Evgendvorkin baseline is `240x235` with many zeros; Qiuzi distill still running/no outputs.
- Queue unchanged: true new 0.95/0.96 source if found; otherwise v590 Zeyad/Rajnish; then Henry NFNet v82/v81/v80; then Pilkwang v11 last-resort saturated fallback.

## 2026-05-20 12:48 UTC update — Qiuzi HGNet distill promoted to v591 validation run

- Slots still capped `5/5`; best remains `0.949`; no v577/v578/v590 submitter active; PR #249 open/BLOCKED at check time.
- Fresh artifacts: `scan_20260520T1248Z.json`, `source_audit_20260520T1248Z_top/summary.json`, and downloaded Qiuzi result CSVs under `source_audit_20260520T1248Z_top/qiuzilang_distill_outputs/`.
- Fresh top-feed triage:
  - `haridoss31/birdclef-my-model` v42 is ERROR; markers suggest custom/Perch/BirdNET/distill code but no usable `submission.csv` output.
  - `mtoshidesu/notebookc6e90ae327` v6 is ERROR; saturated Karnakbayev/PowerOptimization lineage, no usable direct output.
  - `henryszy/bc2026-rankpower-nfnet-v83` is schema-safe and complete, but still 0.949 RankPower/NFNet-family; keep behind new-model lanes.
  - `qiuzilang/hgnetv2-b0-training-distill` is now COMPLETE with all four fold weights and validation outputs. Best fold val scores: fold0 `0.9651087`, fold1 `0.9701546`, fold2 `0.9669707`, fold3 `0.9729050`; final logged OOF AUC: raw `0.9583789`, rank `0.9672701`.
- Decision: promote Qiuzi HGNet from recipe-only to the first real non-saturated model-zoo lead. It is not direct-submit-safe by itself because it is training-only, but complete fold artifacts can be consumed as a kernel source.
- Implementation: prepared `kaggle-kernels/v591-public946-hgnet-distill-w0025/` by forking the public946 v542 anchor, attaching `qiuzilang/hgnetv2-b0-training-distill`, adding a guarded 4-fold HGNetV2-B0 inference sidecar that writes `submission_hgnet.csv`, and blending it conservatively as a `2.5%` rank sidecar (`Proto=0.585`, `SED=0.390`, `HGNet=0.025`).
- Pushed private Kaggle validation kernel `yourslewis/bc26-v591-public946-hgnet-distill-w0025` version 1 via Bearer API. Push returned no invalid competition/data/kernel/model sources. Kernel is RUNNING at log time; no submission attempted while capped.
- Queue update: if v591 completes with valid `submission.csv`/`submission_hgnet.csv`, it should displace v590 as the next-reset structural diagnostic. If v591 fails (mount/timm/runtime), preserve v590 Zeyad/Rajnish as backup and fix v591 only if the failure is straightforward.

## 2026-05-20 13:48 UTC update — v591 fixed, v592 HGNet 10% promoted

- Slots remain capped `5/5`; latest 2026-05-20 submissions are v585 `0.922`, v586 `0.941`, v587 `0.949`, v588 `0.949`, v589 `0.949`; best remains `0.949`; no stale v577/v578/v590 submitter active; PR #249 remains open/BLOCKED.
- Fresh scan/audit artifacts: `scan_20260520T1348Z.json` and `source_audit_20260520T1348Z_top/summary.json`.
- Fresh source decisions:
  - `meenalsinha/birdclef-2026-improved` v23 is now schema-safe (`240x235`) but remains BirdNET/Prior/RankPower-family; not enough to displace HGNet.
  - `samejimatink0/birdclef-2026-hgnetv2-b0-baseline-training` exposes fold/OpenVINO-looking HGNet artifacts in file listing but current session outputs were not downloadable at audit time; watch later as a possible faster HGNet inference/export lead.
  - `sclim2022080004/iter7-protossm-mlp` is schema-safe but explicitly SED-free 0.949/ProtoSSM+MLP lineage; hold as idea-mining only.
  - `mtoshidesu/notebookc6e90ae327` v7 is schema-safe but saturated Karnak/PowerOptimization family.
  - Nina EoS6 v14 primary `submission.csv` remains invalid (`243x235`, all NaN numeric parse).
- v591 validation/fixes:
  - v591 v1 failed on Kaggle CUDA during torchaudio STFT (`cudaErrorNoKernelImageForDevice`).
  - v591 v2 moved preprocessing to CPU but HGNet model itself still failed on Kaggle CUDA with the same error.
  - v591 v3 forced HGNet CPU-only and completed, but local validation caught a bug: `submission_hgnet.csv` was constant across rows because row-id parsing used the second-to-last token as end second and silently zero-filled missing dry-run audio.
  - v591 v4 fixed row-id parsing (`end_sec = final token`) and removed the zero fallback. It completed successfully in ~524s, wrote valid `submission.csv`, `submission_hgnet.csv`, `submission_protossm.csv`, and `submission_sed.csv`; final CSV shape `240x235`, no NaNs, min/max `0.0053125/1.0`; HGNet min/max `3.39e-07/0.9477211`.
- Local sidecar gate for fixed v591/v4: `artifacts/blend_grids/v591_hgnet_sidecar_weight_grid_20260520T1348Z_v4.json`. HGNet standalone rank sidecar has macro AUC `0.9956425` on the train-soundscape overlap with corr vs anchor `0.4808`. Blend grid: base `0.9925249`; HGNet `0.025` -> `0.9927187`; HGNet `0.10` -> `0.9932913`, top3 recall `0.6211` vs base `0.5211`.
- Decision: promote a higher-upside 10% HGNet variant rather than spending the next slot on conservative 2.5% HGNet or saturated v590.
- Added and pushed private validation kernel `yourslewis/bc26-v592-public946-hgnet-distill-w010` v1. It completed successfully in ~562s with final `submission.csv` shape `240x235`, no NaNs, min/max `0.0065000006/1.0`; `submission_hgnet.csv` valid, min/max `3.39e-07/0.9477211`.
- Added guarded submitter `scripts/submit_v592_public946_hgnet_w010_when_ready.py`; preflight-only passed (source version 1, COMPLETE/no failure, required outputs, valid final CSV). It did not submit because the day is capped.
- Queue update: v592 HGNet 10% is now the preferred next-reset candidate. v591/v4 is conservative fallback; v590 Zeyad/Rajnish is demoted behind HGNet unless later evidence invalidates HGNet transfer/runtime.

## 2026-05-20 14:48 UTC update — v592 remains reset-slot owner; submitter parked

- Live state unchanged: v585 `0.922`, v586 `0.941`, v587/v588/v589 `0.949`; current best remains `0.949`; 2026-05-20 daily submissions remain `5/5`; no stale v577/v578/v590/v591/v592 submitter was active at check start.
- PR #249 remains open/BLOCKED or UNKNOWN/BLOCKED depending on GitHub merge-state fetch; branch clean before this log update.
- Fresh scan artifact: `artifacts/public_kernels_20260520_frontier_candidates/scan_20260520T1448Z.json`.
- Fresh source audit artifact: `artifacts/public_kernels_20260520_frontier_candidates/source_audit_20260520T1448Z_top/summary.json`.
- Fresh top-feed decisions:
  - `mtoshidesu/notebookc6e90ae327` v8: schema-safe sample output (`3x235`) but still saturated Karnak/PowerOptimization family.
  - `jacqueszhelinzhang/birdclef26-perch-minimal` v22: constant probability baseline (`3x235`, all `0.0042735`); reject.
  - `samejimatink0/birdclef-2026-hgnetv2-b0-baseline-training` v23: still RUNNING; file list exposes PyTorch and OpenVINO fold artifacts, but downloadable session outputs/log unavailable in this audit. Track as future HGNet acceleration/export lead, not a current slot displacer.
  - `sclim2022080004/iter7-protossm-mlp` v5: schema-safe (`240x235`) but source self-identifies as 0.949 community/ProtoSSM+MLP SED-free subset; idea-mining only.
  - `abhiiiish/birdclef-26-nb-training` v2: training artifact notebook only, no submission output; possible future model-artifact idea but not slot-ready.
  - `thbdh5765/birdclef-2026-s124-s114-g124-f1-rankblend-fork` v1: S124/G124 fork duplicate/sibling; v589 already tied only, so reject.
- Decision: no source found to displace v592. Keep v592 HGNet 10% as next-reset owner.
- Action: parked a guarded reset submitter for v592. PID `13173`, log `logs/v592_hgnet_w010_reset_submitter_20260520.log`, nohup log `logs/v592_hgnet_w010_reset_nohup_20260520.out`; it sleeps until ~`2026-05-21T00:05:00Z` and then runs `scripts/submit_v592_public946_hgnet_w010_when_ready.py` once. The submitter itself rechecks duplicate submissions, source, COMPLETE status, outputs, final CSV, and daily cap before submitting.

## 2026-05-20 15:48 UTC update — capped, v592 submitter healthy, Samejima OpenVINO lead not ready

- Live state unchanged: latest 2026-05-20 submissions remain v585 `0.922`, v586 `0.941`, v587/v588/v589 `0.949`; current best remains `0.949`; daily cap remains `5/5`.
- Guarded v592 reset submitter remains alive: PID `13173`, sleeping until about `2026-05-21T00:05:00Z`; log target `logs/v592_hgnet_w010_reset_submitter_20260520.log`. It has not attempted submission yet.
- Branch clean at start; PR #249 still open with GitHub merge state `UNKNOWN` in this fetch.
- Fresh scan artifact: `artifacts/public_kernels_20260520_frontier_candidates/scan_20260520T1548Z.json`.
- Fresh source audit artifact: `artifacts/public_kernels_20260520_frontier_candidates/source_audit_20260520T1548Z_top/summary.json`.
- Fresh top-feed decisions:
  - `samejimatink0/birdclef-2026-hgnetv2-b0-baseline-training` v25: now ERROR, but file list exposes fold `.pt` and OpenVINO `.xml/.bin` artifacts. Useful future acceleration/export lead for our HGNet lane, but not slot-ready.
  - `samejimatink0/birdclef-2026-hgnetv2-b0-baseline-inference` v3: COMPLETE but primary `submission.csv` is invalid/all-NaN (`3x235`, 702 NaNs); do not direct-submit.
  - `deepanshus167/bird-claasifier-comp` v14: constant probability baseline; reject.
  - `nina2025/birdclef-2026-eos-6-sz` v15: primary output remains invalid (`243x235`, all NaN numeric parse); reject direct replay.
  - `mtoshidesu/notebookc6e90ae327` v9: schema-safe sample output but saturated Karnak/PowerOptimization family; does not displace v592.
  - `anthonytherrien/birdclef-2026-s124-s114-g124-f1-blend` v1: S124/G124 sibling/duplicate after v589 tied only; reject.
  - `scenerysunfireink/birdclef-2026-v6-fork-model-7-single` v1: Model_7/Karnak-style branch, schema-safe sample output but saturated family; no slot before HGNet.
- Decision: no candidate found above v592. Keep v592 as reset-slot owner. Samejima OpenVINO is worth mining only after v592 result or if hidden runtime becomes the blocker.
- Reminder/lesson: positive local train-soundscape sidecar gates remain rejection filters, not approval filters; v560/v573 proved locally positive sidecars can public-drop. v592 is allowed because it is a structurally distinct complete HGNet model-zoo artifact, not another micro scalar/sidecar sweep.

## 2026-05-20 16:48 UTC update — capped; v592 reset submitter still healthy

- Live state unchanged: latest 2026-05-20 submissions remain v585 `0.922`, v586 `0.941`, v587/v588/v589 `0.949`; current best remains `0.949`; daily cap remains `5/5`.
- Guarded v592 reset submitter remains alive: PID `13173`, sleeping until about `2026-05-21T00:05:00Z`; no duplicate submitter started and no submission attempted while capped.
- PR #249 remains open with GitHub merge state `UNKNOWN` in this fetch; branch was clean before this log update.
- Fresh scan artifact: `artifacts/public_kernels_20260520_frontier_candidates/scan_20260520T1648Z.json`.
- Fresh top-feed/source conclusions from scan and prior audit:
  - `samejimatink0/birdclef-2026-hgnetv2-b0-baseline-training` reran again, but latest checked public state remains not slot-ready: training artifact/OpenVINO lead only; not a validated competition submission.
  - `samejimatink0/birdclef-2026-hgnetv2-b0-baseline-inference` remains invalid/all-NaN on primary output from the 15:48 audit.
  - New/recent top-feed items are mostly EoS6/Karnak/RankPower/S124 siblings or constant/baseline notebooks; none exceeds the validated repo-owned v592 HGNet candidate.
  - No explicit new `0.950+`/`0.96` source claim surfaced in the search buckets.
- Decision: keep v592 as the sole reset-slot owner. Do not start extra submitters. Samejima OpenVINO remains future acceleration work, especially if v592 hidden runtime becomes the blocker.

## 2026-05-20 17:48 UTC update — capped; fresh two-pass SSM rejected by local gate

- Live state unchanged: latest 2026-05-20 submissions remain v585 `0.922`, v586 `0.941`, v587/v588/v589 `0.949`; current best remains `0.949`; daily cap remains `5/5`.
- Guarded v592 reset submitter remains alive: PID `13173`, sleeping toward about `2026-05-21T00:05:00Z`; no duplicate submitter started and no submission attempted while capped.
- PR #249 remains open/BLOCKED; branch was clean before this log update.
- Fresh scan artifact: `artifacts/public_kernels_20260520_frontier_candidates/scan_20260520T1748Z.json`.
- Fresh source audit artifact: `artifacts/public_kernels_20260520_frontier_candidates/source_audit_20260520T1748Z_top/summary.json`.
- Fresh top-feed decisions:
  - `tulayppppp/my-efficientnet-b0-weights` v20: ERROR/no outputs; weight/training notebook, not slot-ready.
  - `scenerysunfireink/birdclef-2026-eos-6` v1: primary `submission.csv` invalid (`243x235`, all NaN numeric parse); reject direct replay.
  - `deepanshus167/bird-claasifier-comp` v17: constant baseline (`3x235`, all same probability); reject.
  - `scenerysunfireink/birdclef-2026-two-pass-ssm` v1: schema-safe `240x235` and structurally named two-pass SSM, but local gate is weak. Artifact `artifacts/blend_grids/scenery_two_pass_sidecar_weight_grid_20260520T1748Z.json`: standalone rank AUC `0.97745`, corr vs v542 anchor `0.8884`; every tested blend weight reduced macro AUC vs base `0.9925249` (0.025 -> `0.9924122`, 0.10 -> `0.9917289`). Reject as reset-slot displacer.
  - `karnakbaevarthur/s124-g124-reverse-engineered` v2: no outputs; S124/G124 duplicate family; reject.
  - `kospintr/birdclef-efficientnet-perch-distill-mixup` v26: CANCEL_ACKNOWLEDGED/partial, no reliable submission output; not slot-ready.
  - `scenerysunfireink/birdclef-2026-perch-v2-full-v2` v1: invalid output shape (`119988x235`) and score range includes `-1000`; reject direct.
- Decision: no candidate found above v592. Keep v592 as sole reset-slot owner. The two-pass SSM audit reinforces that schema-safe/public-looking outputs still need local rejection gates before slot ownership.

## 2026-05-20 17:54 UTC verification

- Rechecked status after the 17:48 two-pass audit: still capped `5/5`, best `0.949`, v592 reset submitter PID `13173` alive.
- Additional lightweight queries for `0.95`, `0.96`, EoS6, SafeAlign, RankPower/NFNet, HGNet, and two-pass SSM did not reveal a candidate beyond the already-audited Scenery/Nina/S124/RankPower/Samejima families.
- Keep v592 as the only reset-slot owner.

## 2026-05-20 18:48 UTC update — capped; Tulay EfficientNet rerun rejected by source audit

- Live state unchanged: latest 2026-05-20 submissions remain v585 `0.922`, v586 `0.941`, v587/v588/v589 `0.949`; current best remains `0.949`; daily cap remains `5/5`.
- Guarded v592 reset submitter remains alive as PID `13173`; log is still empty because the process is sleeping until the reset window. No v577/v578 scalar submitter and no duplicate v59x submitter found.
- PR #249 remains open with merge-state fetch returning `UNKNOWN`; branch was clean before this log update.
- Fresh scan artifact: `artifacts/public_kernels_20260520_frontier_candidates/scan_20260520T1848Z.json`.
- New source audit artifact: `artifacts/public_kernels_20260520_frontier_candidates/source_audit_20260520T1848Z_new/summary.json`.
- Only genuinely new dateRun item since the 17:48 audit was `tulayppppp/my-efficientnet-b0-weights` rerun at 18:45Z. Audit result: version 23 still RUNNING/no outputs, and source is not competition-safe: it writes generic `bird_i` dynamic columns, uses dummy/fallback row handling, can emit an empty `submission.csv`, and loads a generic EfficientNet baseline rather than a verified 235-class BirdCLEF submission pipeline. Reject as a slot displacer.
- Other fresh scan rows are the already-audited Scenery EoS6/two-pass SSM, Nina EoS6, S124/G124, RankPower/NFNet, Samejima/TY0912 HGNet, SafeAlign, and acoustic-prior/EoS families; no new credible 0.96 candidate surfaced.
- Decision: keep v592 as sole reset-slot owner; do not spend a slot on Tulay EfficientNet or scalar EoS5 tweaks.

## 2026-05-20 19:04 UTC direct user lead — Itshyao S124/S114/G124 rankblend checked

- User flagged `https://www.kaggle.com/code/itshyao/birdclef-2026-s124-s114-g124-f1-rankblend` as a new post.
- Pulled current Kaggle source via API: metadata reports current version `2`, title `BirdCLEF 2026 S124 S114 G124 F1 RankBlend`; artifact saved at `artifacts/public_kernels_20260520_frontier_candidates/itshyao_s124_s114_g124_rankblend_latest/`.
- Compared decoded current source against the previously audited/source-preflighted artifact from `source_audit_20260520T0547Z_top/itshyao__birdclef-2026-s124-s114-g124-f1-rankblend.source.txt`: exact decoded SHA match `c5aed8358ce6ba4b8772c1649bed9475151adff011d07617a8ba2b6f223a62f9` and same `6819` decoded lines. No source change despite v2 metadata.
- Current session output COMPLETE/no failure with outputs including `submission.csv`, `submission_g124_effv2s_fold1_s124.csv`, Proto/SED/Karnak branch files, and `v17_logs.json`; primary `submission.csv` is sample/dry-run shaped `3x235`, finite, min/max `0.47687027/0.5553993`, no NaNs/zeros. Log says `S124 dry-run/mismatch: keeping S114 anchor submission.csv`.
- Existing guarded direct replay v589 already submitted this source and scored `0.949` (`52838266`). Decision: do not let this displace v592; do not resubmit unless the author posts a genuinely changed version or independent >0.949 evidence appears.

## 2026-05-20 19:08 UTC update — user reports Itshyao v2 is 0.952; reset slot moved to v593

- User reported the Itshyao S124/S114/G124 rankblend post is `0.952`. This is independent >0.949 evidence and overrides the previous saturated-family rejection.
- Key correction: our existing v589 submitter was hard-pinned to Kaggle kernel version `1` and scored `0.949`. Current metadata exposes version `2`. Source text appears decoded-identical, but version-specific Kaggle scoring/output can still differ; with a reported `0.952`, v2 deserves the next slot.
- Prepared guarded submitter `scripts/submit_v593_itshyao_s124_g124_rankblend_v2_when_ready.py` for `itshyao/birdclef-2026-s124-s114-g124-f1-rankblend` version `2`, description `v593: Guarded direct Itshyao S124 S114 plus G124 F1 rankblend v2 0952 lead`.
- Preflight-only passed while capped: visible UTC submissions today `5`; source pull version `2`, decoded length `309836`; kernel COMPLETE/no failure; required outputs present; primary `submission.csv` finite `3x235`, min/max `0.47687027/0.5553993`, no zeros/NaNs; required log marker present.
- Killed old v592 reset submitter PID `13173` to preserve the reset slot. Started v593 reset submitter PID `96527`, log `logs/v593_itshyao_s124_g124_v2_reset_submitter_20260520.log`, sleeping toward `2026-05-21T00:05Z`.
- New queue: v593 Itshyao v2 first at reset; v592 HGNet 10% demoted to next candidate if v593 ties/drops/no-scores.

## 2026-05-20 20:00 UTC update — capped; v593 reset owner healthy, 0.952/Karnak lead audited

- Live submission state unchanged: v585 `0.922`, v586 `0.941`, v587/v588/v589 `0.949`; current repo-confirmed best remains `0.949`; 2026-05-20 UTC cap remains `5/5`.
- External/user-reported best lead is now Itshyao S124/S114/G124 v2 at `0.952`, but our own direct replay score is not confirmed yet; v593 remains queued for the reset slot.
- v593 reset submitter remains alive: PID `96527`, log `logs/v593_itshyao_s124_g124_v2_reset_submitter_20260520.log`, sleeping toward `2026-05-21T00:05Z`. Old v592 PID `13173` is not running. No stale v577/v578 scalar submitter found.
- PR #249 remains open with merge-state fetch `UNKNOWN`; PR #245 is merged.
- Fresh scan artifact: `artifacts/public_kernels_20260520_frontier_candidates/scan_20260520T2000Z.json`.
- Fresh audit artifact: `artifacts/public_kernels_20260520_frontier_candidates/source_audit_20260520T2000Z_top/summary.json`.
- Audited new/changed 20:00 leads:
  - `karnakbaevarthur/s124-g124-reverse-engineered`: appears in `0.952` search and explicitly describes reverse-engineering Itshyao S124/G124 LB `0.952`, but it is a training/recipe notebook with no output files/session submission artifact. Not slot-ready; useful as code-mining only after v593.
  - `haivan11/birdclef-2026-prior-field-fusion-vi`: COMPLETE with finite `3x235` output and BirdNET/Yaroslav/yukiZ branch files, but it is a clean fork of Pilkwang/Yaroslav prior-field 0.949-family; dry-run BirdNET row-id mismatch keeps anchor. Backup only, below v593 and v592.
  - `tulayppppp/my-efficientnet-b0-weights`: v27 ERROR/no outputs; source still generic/dynamic/fallback EfficientNet, not competition-safe.
- Decision: keep v593 Itshyao v2 as reset owner; v592 HGNet remains backup if v593 ties/drops/no-scores.

## 2026-05-20 22:00 UTC update — capped; v593 still reset owner, fresh top-feed rejected

- Live submission state unchanged: v585 `0.922`, v586 `0.941`, v587/v588/v589 `0.949`; repo-confirmed best remains `0.949`; 2026-05-20 cap remains `5/5`; no 2026-05-21 submissions visible yet.
- v593 Itshyao v2 reset submitter remains healthy as PID `96527`, sleeping toward `2026-05-21T00:05Z`. Old v592 PID `13173` is not running. No v577/v578 scalar submitter found.
- PR #249 remains open/BLOCKED; PR #245 remains merged.
- Fresh scan artifact: `artifacts/public_kernels_20260520_frontier_candidates/scan_20260520T2200Z.json`.
- Fresh audit artifact: `artifacts/public_kernels_20260520_frontier_candidates/source_audit_20260520T2200Z_top/summary.json`.
- Fresh top-feed audit:
  - `muhammadsaadalvi/birdclef-2026-wildsound-v8`: v69 ERROR/no outputs; training/preprocessing path fails on missing `/kaggle/input/birdclef-2026/train_metadata.csv`; not slot-ready despite distinct WildSound-ish direction.
  - `juanpp11/birdclef2026-seguda-parte`: v4 ERROR/no outputs; local/offline EfficientNet weight path notebook, no validated `submission.csv`; not slot-ready.
  - `tulayppppp/my-efficientnet-b0-weights`: v28 ERROR/no outputs; still generic/dynamic/fallback EfficientNet and not competition-safe.
- Decision: no candidate displaces v593. Keep v593 first at reset; v592 HGNet remains backup if v593 ties/drops/no-scores.

## 2026-05-21 00:01 UTC reset update — v593 submitted, pending

- At reset, 2026-05-21 visible submissions were initially `0`; 2026-05-20 final remained v585 `0.922`, v586 `0.941`, v587/v588/v589 `0.949`.
- The parked v593 submitter PID `96527` woke but failed before submission because it used system `python3` and hit a local Kaggle SDK signature mismatch: `KaggleHttpClient.__init__() got an unexpected keyword argument 'api_token'`. No submission was created by that failed wrapper.
- Immediately reran the same guarded submitter with the repo/Kaggle venv: `/Users/yourslewis/.openclaw/workspace-don/kaggle/playground-series-s6e3/.venv/bin/python scripts/submit_v593_itshyao_s124_g124_rankblend_v2_when_ready.py`.
- v593 manual guarded run passed all checks and submitted successfully: ref `52866246`, description `v593: Guarded direct Itshyao S124 S114 plus G124 F1 rankblend v2 0952 lead`, date `2026-05-21T00:06:26.767Z`, status `pending` at `00:08Z`; 2026-05-21 count is now `1/5`.
- Pre-submit scan artifact: `artifacts/public_kernels_20260521_frontier_candidates/scan_20260521T0001Z.json`.
- Pre-submit audit artifact: `artifacts/public_kernels_20260521_frontier_candidates/source_audit_20260521T0001Z_top/summary.json`.
- Fresh 00:01 audit decisions: `rauffauzanrambe/birdclef-2026-s124-s114-g124-f1-rankblend` is a fork of the S124/S114/G124 family but lacks the G124 asset and logs `S124 G124 fold1 rank sidecar failed`; not better than v593. `mtoshidesu/notebookc6e90ae327` v11 is COMPLETE/schema-valid but only a Karnakbayev PowerOptimization `0.948`/0.949-family output; not a slot displacer.
- Hold remaining 4 slots until v593 scores or a clearly stronger fresh source appears. If v593 confirms >=0.952, port/confirm repo-owned; if it ties/drops/no-scores, fall back to v592 HGNet or S124/G124 reconstruction mining.

## 2026-05-21 02:00 UTC update — v593 tied; v594 HGNet CPU submitted

- v593 Itshyao S124/S114/G124 v2 scored `0.949`, tying the plateau and failing to confirm the external `0.952` report. 2026-05-21 now had `1/5` used before the v594 action.
- Contingency executed: v592/v594 HGNet 10% sidecar was next structural backup. First direct v592 submit attempt passed source/status/output preflight but Kaggle rejected it because the repo-owned kernel metadata was marked GPU: `Submission not allowed: Your Notebook's runtime of 9 minutes exceeds this competition's GPU max of 0 minutes.`
- Fixed the private kernel metadata for `yourslewis/bc26-v592-public946-hgnet-distill-w010`: set `enable_gpu=false`, corrected metadata slug to the actual `...hgnet-distill-w010`, added `id_no=119970462`, and repushed as kernel version `2`. CPU run completed.
- Added `scripts/submit_v594_public946_hgnet_w010_cpu_when_ready.py`, pinned to kernel version `2`, description `v594: Repo-owned public946 plus Qiuzi HGNet rank sidecar 10pct CPU v2`.
- v594 preflight passed: source version `2`, COMPLETE/no failure, required outputs present, final `submission.csv` valid `240x235`, no NaNs, min/max `0.0065000006/1.0`.
- Submitted v594 ref `52869105` at `2026-05-21T02:14:22.29Z`; status `pending`; 2026-05-21 count now `2/5`.
- Fresh 02:00 scan artifact: `artifacts/public_kernels_20260521_frontier_candidates/scan_20260521T0200Z.json`. New feed was mostly pulled/forked S124/EoS/PriorField or already-audited error/no-output notebooks; no source displaced v594 before submission.
- Hold remaining `3/5` slots until v594 scores or a clearly stronger non-saturated source appears.
