
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

## 2026-05-18 23:45 UTC execution update — reset queue healthy, repo-owned v580 dependency blocker found

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
