# BirdCLEF AutoResearch Log

This log tracks spec-driven implementation/tuning work from `docs/BIRDCLEF_NEW_DIRECTIONS_SPECS.md`.

## 2026-05-13 07:55 UTC — `v545-public946-clap-int8`

- **Track:** P2 public946 AutoResearch distinct-signal layer; CLAP INT8 side stream on top of the confirmed 0.946 public946 anchor.
- **Hypothesis:** `v543` and `v544` showed BirdNET is safe but did not beat 0.946, so the next useful experiment should add a different acoustic representation. A small CLAP INT8 rank stream may add complementary audio-semantic signal while preserving the `v542` public946 floor.
- **Branch/PR:** `feature/v545-public946-clap-int8` (fresh worktree from `origin/main`; PR pending at implementation time).
- **Base:** forked `kaggle-kernels/v542-afr1ste-updated-public946/`, which is the confirmed 0.946 Afr1ste updated public946 V8 replay.
- **New source:** public dataset `habedi/birdclef-2026-clap-int8-bundle`, previously verified attachable by Bearer Dataset API. It provides `clap_audio_int8.onnx`, `probe_weights.npz`, `mel_filters_slaney.npy`, and `probe_config.json`.
- **Config/hyperparameters:** final rank blend changed from Proto/SED `0.60/0.40` to Proto/SED/CLAP `0.57/0.38/0.05`; CLAP side stream has hard no-fallback gates, writes `submission_clap_onnx.csv`, requires finite nonzero predictions, and uses a 45-minute hidden-test budget.
- **Implementation files:** `kaggle-kernels/v545-public946-clap-int8/`, `scripts/push_v545.py`.
- **Validation:** static Python compile passed for the push script and Kaggle script. Required Kaggle dry-run gates after push: CLAP ONNX session loads, `submission_clap_onnx.csv` exists and row-aligns with Proto/SED, final log says explicit v545 3-way CLAP blend, final `submission.csv` has no NaNs and shape `(240,235)`, wall time remains safe.
- **Next step:** push real Kaggle kernel `yourslewis/bc26-v545-public946-clap-int8`, monitor to COMPLETE/ERROR, then submit only after dry-run gates pass and daily cap allows.

### v545 v1 failure + v2 mount-search fix — 2026-05-13 08:52 UTC

- **v1 result:** Kaggle kernel `yourslewis/bc26-v545-public946-clap-int8`, version 1, failed before final blend. Failure was intentional hard-stop: `FileNotFoundError: CLAP ONNX model not found: /kaggle/input/birdclef-2026-clap-int8-bundle/clap_audio_int8.onnx`. Partial outputs contained `submission_protossm.csv` and `submission_sed.csv`, confirming the public946 base ran before the CLAP mount check.
- **Root cause:** dataset source attached, but Kaggle mounted it at a non-flat path rather than the public notebook's `/kaggle/input/birdclef-2026-clap-int8-bundle` path. This mirrors the earlier v510 real-SED mount-path class of failures.
- **Fix:** updated v545 to check both flat and `/kaggle/input/datasets/habedi/...` paths, then recursively search `/kaggle/input/**/clap_audio_int8.onnx` and print candidates. Still hard-fails if no CLAP model is found; no silent public946 fallback.
- **Validation:** `python3 -m py_compile` passed after the fix. Pushed v545 version 2 via Bearer API; push returned no invalid data/competition/kernel/model sources. v545 v2 is COMPLETE/no failure. Kaggle log confirms CLAP bundle resolved at `/kaggle/input/datasets/habedi/birdclef-2026-clap-int8-bundle`, CLAP processed `20/20` dry-run files in `55.6s`, wrote `submission_clap_onnx.csv (240,235)`, and executed explicit `57/38/5` Proto/SED/CLAP blend with sonotype mirroring and rare thresholding. Downloaded outputs to ignored artifact `artifacts/kaggle_outputs/v545-public946-clap-int8/`; final `submission.csv` shape `(240,235)`, no NaNs; CLAP CSV no NaNs; v545 vs v542 final corr `0.998122`, MAE `0.01546`, max abs `0.08383`. Guarded submit monitor `logs/submit_v545_when_ready_20260513T085043Z.log` (pid `68912`) attempted submission and hit daily cap with ~14h remaining; it is sleeping and will retry.

### v545 CLAP weight-grid diagnostic while capped — 2026-05-13 10:45 UTC

- **Command/artifact:** ran local CLAP weight grid from downloaded v545 Proto/SED/CLAP outputs plus v542 final output; JSON artifact `artifacts/blend_grids/public946_clap_int8_weight_grid_20260513.json` (ignored).
- **Grid:** kept the public946 Proto/SED ratio fixed at 60/40 among the non-CLAP mass and swept CLAP rank weight `0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.075, 0.10, 0.125, 0.15`; applied the same fake-only/proto-continuity/SED-spike/mirror/rare gates as v545.
- **Result:** on the dry-run labeled overlap (`240` rows, `42` valid classes), pure public946 `w=0.00` had local AUC `0.990665`; submitted v545 `w=0.05` had `0.988749`, corr vs v542 `0.998122`, MAE `0.01546`; CLAP standalone AUC was weak (`0.448223`) but very low-correlation (`corr=0.0098`).
- **Decision:** this diagnostic argues against widening CLAP before leaderboard feedback. Keep v545 as the single queued CLAP probe. If it drops below 0.946, kill CLAP INT8 for public slots; if it ties, do not spend a slot on higher CLAP weights; if it unexpectedly improves, consider a smaller `0.01`-`0.02` CLAP follow-up rather than increasing weight.

### Public946 next-sidecar source audit while v545 capped — 2026-05-13 11:45 UTC

- **Track:** P2/D next distinct-signal preparation while v545 is complete but blocked by daily cap. No new Kaggle kernel/submission was added.
- **Command/artifact:** Bearer API source audit for `zeyadmohamadezzat/birdclef-2026-two-branch-perch-sed-sidecar`, `meenalsinha/birdclef-2026-improved`, and `henryszy/bc2026-raunak0946-direct-v44`; ignored JSON artifact `artifacts/public946_sidecar_source_audit_20260513.json`.
- **Findings:** `chaneyma/birdclef-2026-cv9245-moe-artifacts` is public/attachable and contains four MoE fold weights plus `pantanal_infer_only_submission.py`, `student_cnn...pt`, and `student_crnn...pt` (~69 MB total). `tsubasatech/birdclef-2026-snowflake-sed` is public/attachable and contains ConvNeXt-Tiny and EfficientNetV2-M SED ONNX files (~328 MB). Zeyad's two-branch public fork uses CV9245 with `CV9245_RANK_WEIGHT=0.05` and optional BirdNET `0.025`, but also attaches CLAP/Snowflake sources that must be source-cleaned before any repo candidate. Henry's train-audio-head fork adds a public train-audio-head rank voter at 5% and claims hidden tie-break improvement while displaying 0.946. Meenal's improved fork is essentially a heavier BirdNET branch (`20%`) and is deprioritized because our BirdNET 10%/5% probes only tied.
- **Decision:** Do not push a v546 while v545 is unscored/capped. If v545 drops/ties and no better LB signal appears, the next source-clean AutoResearch candidate should be a minimal public946 + CV9245 sidecar (likely `0.02`-`0.05` rank weight) or the train-audio-head 5% fork after local output/correlation gates. Avoid Meenal/BirdNET widening.

### v545 cap hold recheck + next-candidate gate — 2026-05-13 12:45 UTC

- **Status check:** latest scored submissions remain `v544=0.946`, `v543=0.946`, `v538=0.930`, `v542=0.946`, `v541=0.946`; current best remains **0.946 public LB**. `v545` is COMPLETE/no failure with `submission.csv`, `submission_clap_onnx.csv`, `submission_protossm.csv`, and `submission_sed.csv`. `v510` is still COMPLETE/no failure with `submission.csv`; log confirms real SED manifest found, `6/6` TorchScript models loaded, `REAL_SED_BLEND_WEIGHT=0.05` applied, and wall time `370.6s`.
- **Queue/monitor:** guarded `v545` submit monitor pid `68912` remains alive. It attempted `bc26-v545-public946-clap-int8` version 2 and hit the daily 5-submission cap; no duplicate submission or queue restart was made.
- **Spec read:** active spec now treats the old 0.927 language as stale and prioritizes distinct signal over public946 copies. Because `v545` is unscored and capped, the safe action remains monitoring + preparation rather than pushing `v546`.
- **Next gate:** hold all new Kaggle pushes until v545 scores. If v545 drops/ties, run a local output/correlation gate for source-clean public946+CV9245 (`0.02`/`0.05`) or train-audio-head 5% before spending the next daily slot; if v545 unexpectedly improves, consider only a lower CLAP weight (`0.01`/`0.02`) follow-up.

### Public946 sidecar gate utility smoke — 2026-05-13 13:45 UTC

- **Track:** P2/P3 preparation while `v545` is capped and unscored; no new Kaggle kernel or submission was pushed.
- **Implementation:** added reusable pre-submit diagnostic `scripts/birdclef_public946_sidecar_weight_grid.py`. It row-aligns a public946 anchor CSV with any candidate sidecar CSV, rank-blends a short sidecar-weight grid, and reports label-overlap AUC plus correlation/MAE/max-abs versus the anchor. This is the intended local gate before a future source-clean CV9245 or train-audio-head candidate consumes a daily slot.
- **Smoke command/artifact:** ran the utility on the known v542 anchor plus the v545 CLAP side stream as a validation case; ignored output `artifacts/blend_grids/public946_sidecar_gate_clap_smoke_20260513.json`.
- **Smoke result:** anchor-only `sidecar_0.0000` was best on the available dry-run label overlap (`190` matched rows / `42` valid classes) with macro AUC `0.992525`; CLAP sidecar standalone macro AUC `0.455042`, corr vs anchor `-0.02796`. Increasing CLAP weight monotonically reduced local AUC through the tested `0.10` weight, consistent with the earlier v545 CLAP-specific diagnostic.
- **Validation:** `python3 -m py_compile scripts/birdclef_public946_sidecar_weight_grid.py` passed and the smoke run completed.
- **Next:** keep `v545` as the only queued CLAP probe. After v545 scores, use this utility for `public946+CV9245` (`0.02`/`0.05`) or train-audio-head 5% local gates before any `v546` Kaggle push.

### Public946 gate AutoResearch scaffold + v545 monitor restart — 2026-05-13 14:45 UTC

- **Status check:** latest scored submissions remain `v544=0.946`, `v543=0.946`, `v538=0.930`, `v542=0.946`, `v541=0.946`; current best remains **0.946 public LB**. `v545` is COMPLETE/no failure with CLAP side output and final `submission.csv`. `v510` remains COMPLETE/no failure; log still confirms real SED manifest found, `6/6` TorchScript models loaded, blend `0.05` applied, and wall time `370.6s`.
- **Queue fix:** stale `v545` monitor pid `68912` had exited after the cap sleep. Rechecked recent submissions (`v545` absent), then restarted guarded monitor as pid `22153`, log `logs/submit_v545_when_ready_20260513T144650Z_restart.log`. It immediately re-hit the daily cap with `9.2h` remaining and is sleeping; no duplicate submission was created.
- **Track:** Spec F/P2 AutoResearch parameter tuning around the locked public946 anchor, while avoiding a new Kaggle push until `v545` scores.
- **Implementation:** added `scripts/birdclef_public946_gate_autoresearch.py`, a reusable random/grid search over public946 rank-blend gate parameters: Proto/SED weight, fake-only boost, Proto continuity threshold/boost, SED-only spike threshold/boost, and rare-taxon scale. It reports label-overlap macro AUC/top-k, delta vs baseline, correlation/MAE/max-abs vs a reference submission, and writes the top candidate CSV if requested.
- **Smoke command/artifacts:** ran 601 deterministic trials on v542 dry-run Proto/SED outputs with train-soundscape labels and v542 final as reference. Outputs: `artifacts/blend_grids/public946_gate_autoresearch_v542_smoke_20260513.json` and top CSV `artifacts/blend_grids/public946_gate_autoresearch_v542_top_20260513.csv` (ignored).
- **Smoke result:** baseline reconstruction macro AUC `0.992525` on `190` matched rows / `42` valid classes. Best sampled config improved local AUC to `0.993314` (`+0.000789`), with `proto_weight=0.56`, lighter fake/proto/sed boosts (`fake_boost=0.04`, `ctx_boost=0.10`, `sed_boost=0.08`), corr vs v542 final `0.99465`, MAE `0.01242`. This is only a local gate signal, not enough to submit while `v545` is pending, but it supports the owner's point that AutoResearch-style tuning can find a few `0.00x` candidates.
- **Validation:** `python3 -m py_compile scripts/birdclef_public946_gate_autoresearch.py` passed; quickcheck rerun completed with warnings suppressed.
- **Next:** after v545 scores, use this harness plus the sidecar-weight gate to rank one candidate from: tuned public946 gates, public946+CV9245, and public946+train-audio-head. Submit only the best distinct/safe candidate rather than several adjacent parameter variants.

## 2026-05-13 06:55 UTC — `public946-live-status-and-clap-int8-audit`

- **Track:** P2 public946 diversity-stream triage while `v544` score is pending; no new submission slot consumed.
- **Live LB/submission state:** latest Bearer API poll shows `v541=0.946`, `v542=0.946`, `v543=0.946`, `v538=0.930`, and `v544` ref `52603058` still pending/no publicScore field. The five 2026-05-13 UTC submissions are `v541`, `v542`, `v538`, `v543`, and `v544`, so the daily code-submission allowance is effectively consumed.
- **Queue/monitor state:** no persistent `submit_pending_birdclef_queue.py` process is active. Latest focus queue log ended with `All pending kernels are already submitted.` The queue script still lists older pending definitions, but focus priority stops at already-submitted v538; do not restart it until a new unsubmitted completed kernel is intentionally added.
- **Failure diagnosis:** `v510` remains COMPLETE with `submission.csv`. Kaggle log reconfirmed real SED manifest found, 6/6 TorchScript models loaded, real SED runtime `214.4s`, blend weight `0.05` applied, output `(240,235)`, wall time `370.6s`; no dataset-mount, zip, TorchScript, timeout, row/column, memory, ffmpeg, TF/XLA-fatal, onnxruntime, invalid-source, or silent-skip failure was found. `v544` is COMPLETE with `submission.csv`, `submission_birdnet.csv`, `submission_protossm.csv`, and `submission_sed.csv`; log confirms BirdNET mapped `157/234`, BirdNET runtime `16.6s`, and explicit `56/39/5` 3-way blend.
- **Spec decision:** do not queue another BirdNET weight before `v544` scores. Since `needless090` V5/CLAP remains source-blocked, audited a separate source-resolved CLAP INT8 route from `xiyuetong/birdclef2026-ensemble-v2`.
- **CLAP INT8 audit:** Bearer Dataset API confirms public dataset `habedi/birdclef-2026-clap-int8-bundle` with `clap_audio_int8.onnx`, `probe_weights.npz`, `mel_filters_slaney.npy`, and `probe_config.json`. Kernel metadata for `xiyuetong/birdclef2026-ensemble-v2` attaches this dataset plus the standard public946 inputs. Extracted source cached locally at ignored artifact `artifacts/public_kernels_20260513/birdclef2026-ensemble-v2.py` and contains a fast `submission_clap_onnx.csv` path.
- **Next step:** after `v544` score lands, if no improvement, prepare `v545` as a source-clean public946 + CLAP-ONNX minority stream from the `v542` anchor with conservative CLAP rank weight `0.03`-`0.05`; require CLAP session load, row alignment, no NaNs, non-fallback final blend log, safe wall time, and bounded correlation/MAE vs `v542` before using a submission slot.

## 2026-05-13 05:56 UTC — `v544-public946-birdnet05`

- **Track:** P2 BirdNET minority-stream follow-up after `v543` tied the 0.946 anchor.
- **Hypothesis:** Since `v543` (10% BirdNET) tied `v541/v542` at 0.946 but local grid preferred a smaller BirdNET perturbation, a 5% BirdNET stream may keep the 0.946 floor while adding slightly safer distinct signal.
- **Branch/PR:** `feature/v543-public946-birdnet3`, PR #226.
- **Kernel:** pushed real Kaggle kernel `yourslewis/bc26-v544-public946-birdnet-5pct`, version 1; push returned no invalid data/kernel/model sources.
- **Config/hyperparameters:** forked `v543`; changed final rank blend from Proto/SED/BirdNET `0.52/0.38/0.10` to `0.56/0.39/0.05`; kept BirdNET model source `shadiakiki1/birdnet-analyzer/TfLite/birdnet_global_6k_v2.4_model_fp32-1/3`, sonotype mirroring, and rare-taxon adaptive thresholding.
- **Kaggle validation:** kernel COMPLETE/no failure. Output files include `submission.csv`, `submission_birdnet.csv`, `submission_protossm.csv`, and `submission_sed.csv`; log confirms BirdNET runtime `16.6s`, 5% 3-way rank blend executed, sonotype mirroring applied to 10 columns, rare thresholding applied to 44 species, and notebook wall time about `556s`.
- **Dry-run gates:** downloaded outputs to `artifacts/kaggle_outputs/v544-public946-birdnet05/`; final `submission.csv` shape `(240,235)`, no NaNs. v544 vs v542 final corr `0.999884`, MAE `0.01083`; v544 vs v543 final corr `0.999864`, MAE `0.01082`. Validation summary: `artifacts/kaggle_outputs/v544-public946-birdnet05/validation_summary.json`.
- **Submission:** submitted code competition kernel version 1 as ref `52603058` with description `v544: Public946 v542 plus source-clean BirdNET 6K 3-way rank blend 56/39/5`. Score was pending at the first post-submit poll.
- **Next step:** Monitor ref `52603058`. If it ties/improves 0.946, stop BirdNET widening and pivot to V5/CLAP source resolution or public946 teacher/student work; if it drops, stop BirdNET and do not queue further weights.

## 2026-05-13 04:44 UTC — `public946-birdnet-weight-grid-hold`

- **Track:** P2 BirdNET minority-stream hyperparameter gate while `v543` score is pending.
- **Hypothesis:** If `v543` 10% BirdNET ties/improves the 0.946 anchor, the safest follow-up is likely a smaller BirdNET minority weight that preserves more of the validated `v542` public946 signal.
- **Branch/PR:** `feature/v543-public946-birdnet3`, PR #226.
- **Data/artifacts used:** downloaded Kaggle outputs under `artifacts/kaggle_outputs/v543-public946-birdnet3/` plus `v542` final dry-run output; local train-soundscape labels for overlap scoring.
- **Command:** inline Python grid over BirdNET weights using the exact v542/v543 final postprocess sequence; output JSON `artifacts/blend_grids/public946_birdnet_weight_grid_20260513.json`.
- **Grid:** keeping the v543 schedule `proto=0.60-0.80*w_bn`, `sed=0.40-0.20*w_bn`, best local overlap AUC was at `w_bn=0.05` (`proto=0.56`, `sed=0.39`, AUC `0.992617`, corr vs v542 `0.999884`, MAE `0.01083`). The submitted `w_bn=0.10` had AUC `0.992540`, corr `0.999558`, MAE `0.02166`.
- **Decision:** do not submit another BirdNET variant while ref `52600158` is score-pending. If `v543` ties/improves 0.946, package `v544` as the smaller 5% BirdNET candidate; if `v543` drops, stop BirdNET and pivot to V5/CLAP source resolution or public946 teacher/student work.

## 2026-05-13 04:10 UTC — `v543-public946-birdnet3`

- **Track:** P2 distinct public946 diversity stream: source-clean BirdNET 6K 3-way rank blend after `v541`/`v542` anchor lock.
- **Hypothesis:** A conservative BirdNET-only stream from the public BirdNET 4-way family may add acoustic diversity to the 0.946 Perch/ProtoSSM + distilled SED anchor without relying on the blocked custom EffNet notebook-output source.
- **Branch:** `feature/v543-public946-birdnet3`.
- **Kernel:** pushed real Kaggle kernel `yourslewis/bc26-v543-public946-birdnet-3way`, version 1, ref URL from push `https://www.kaggle.com/code/yourslewis/bc26-v543-public946-birdnet-3way`; push returned no invalid data/kernel/model sources.
- **Config/hyperparameters:** forked `v542` and inserted BirdNET TFLite stream from model source `shadiakiki1/birdnet-analyzer/TfLite/birdnet_global_6k_v2.4_model_fp32-1/3`; final rank blend is Proto `0.52` + SED `0.38` + BirdNET `0.10`; kept sonotype mirroring and rare-taxon adaptive thresholding. Custom EffNet remains intentionally skipped because its notebook-output source is not available.
- **Kaggle validation:** kernel status COMPLETE/no failure. Output files include `submission.csv`, `submission_birdnet.csv`, `submission_protossm.csv`, and `submission_sed.csv`; log confirms BirdNET model resolved, `157/234` labels mapped, BirdNET runtime `15.6s`, 3-way rank blend executed, sonotype mirroring applied to 10 columns, rare thresholding applied to 44 species, and total notebook wall time about `546s`.
- **Dry-run gates:** downloaded outputs to `artifacts/kaggle_outputs/v543-public946-birdnet3/`; final `submission.csv` shape `(240,235)`, no NaNs, train-overlap `190` rows / `42` valid AUC classes. Local overlap AUC: v543 final `0.992540` vs v542 final `0.992525`; BirdNET standalone is weak locally (`0.501832`) but low-correlation with v542 (`corr=0.0196`). v543 final vs v542 final corr `0.999558`, MAE `0.02166`. Validation summary: `artifacts/kaggle_outputs/v543-public946-birdnet3/validation_summary.json`.
- **Submission:** submitted code competition kernel version 1 as ref `52600158` with description `v543: Public946 v542 plus source-clean BirdNET 6K 3-way rank blend 52/38/10`. Status was still pending/no public score at the last poll.
- **Next step:** Monitor ref `52600158`. If it ties/improves 0.946, keep BirdNET as a minority public946 diversity stream and optionally test a smaller `0.05` BirdNET weight; if it drops, stop BirdNET and pivot to V5/CLAP source resolution or public946 teacher/student work.

### Public946 gate AutoResearch full sweep + cap monitor refresh — 2026-05-13 15:45 UTC

- **Status check:** latest scored submissions remain `v544=0.946`, `v543=0.946`, `v538=0.930`, `v542=0.946`, `v541=0.946`; current best remains **0.946 public LB**. `v545` remains COMPLETE/no failure but unsubmitted behind the daily cap. `v510` remains COMPLETE/no failure with `submission.csv`; log still confirms real SED manifest found, `6/6` TorchScript models loaded, blend `0.05` applied, and wall time `370.6s`.
- **Queue fix:** the restarted `v545` monitor pid `22153` had already exited/staled. Rechecked recent 200 submissions (`v545` absent), then restarted guarded monitor as pid `35141`, log `logs/submit_v545_when_ready_20260513T154642Z_restart2.log`. It immediately re-hit the daily cap with `8.2h` remaining and is sleeping; no duplicate submission was created.
- **Full AutoResearch result:** the background public946 gate sweep completed successfully. Artifact `artifacts/blend_grids/public946_gate_autoresearch_v542_full_20260513.json` contains `3646` trials on v542 dry-run Proto/SED outputs with `190` matched rows / `42` valid classes. Baseline macro AUC `0.992525`. Best configs reached macro AUC `0.993325` (`+0.000800`) with the same stable pattern: `proto_weight=0.56`, lighter fake-only boost `0.04`, lighter Proto-continuity boost `0.10`, lighter SED-spike boost `0.08`, and `ctx_thr=0.90`. Corr vs v542 final was about `0.9946`-`0.9951`, MAE `0.0118`-`0.0127`, max abs `0.3530`.
- **Interpretation:** the full sweep reinforces the earlier smoke signal that a tuned public946-gate candidate may be worth one slot after v545 scores, but only as one of the ranked candidates against CV9245/train-audio-head. Do not submit a pure gate-tuned v546 while `v545` is pending.

### v545 cap hold + public946 spec refresh — 2026-05-13 16:45 UTC

- **Status check:** latest scored submissions remain `v544=0.946`, `v543=0.946`, `v538=0.930`, `v542=0.946`, `v541=0.946`; current best remains **0.946 public LB**. `v545` remains COMPLETE/no failure with CLAP side output and final `submission.csv`, but is still unsubmitted behind the daily cap. `v510` remains COMPLETE/no failure with real SED manifest found, `6/6` TorchScript models loaded, blend `0.05` applied, and wall time `370.6s`.
- **Queue/monitor:** guarded `v545` submit monitor pid `35141` is alive and sleeping after a cap response with ~`8.2h` remaining from the 15:46 UTC restart. No duplicate submission exists in recent submissions.
- **Spec maintenance:** refreshed `docs/BIRDCLEF_PUBLIC946_PRIORITIZED_SPEC_20260512.md` so `v544` is no longer marked pending. Both BirdNET weights (`v543` 10% and `v544` 5%) tied 0.946; BirdNET alone is now a safe-tie diversity stream, not the next breakthrough lane.
- **Decision:** continue holding new Kaggle pushes until `v545` scores. Next v546 decision should rank the full-sweep tuned public946-gate candidate against CV9245 and train-audio-head sidecars, not another BirdNET-only variant.

### CV9245 sidecar port preflight while v545 capped — 2026-05-13 17:45 UTC

- **Status check:** latest scored submissions remain `v544=0.946`, `v543=0.946`, `v538=0.930`, `v542=0.946`, `v541=0.946`; current best remains **0.946 public LB**. `v545` remains COMPLETE/no failure but unsubmitted behind the daily cap. `v510` remains COMPLETE/no failure with real SED manifest found, `6/6` TorchScript models loaded, blend `0.05` applied, and wall time `370.6s`.
- **Queue/monitor:** guarded `v545` submit monitor pid `35141` is alive and sleeping after the cap response; no duplicate `v545` submission exists.
- **Track:** P2 next distinct-signal preparation while waiting for v545 score.
- **CV9245 preflight:** downloaded and statically inspected public dataset files `README.md` and `pantanal_infer_only_submission.py` from `chaneyma/birdclef-2026-cv9245-moe-artifacts`; ignored local copies are under `artifacts/public946_cv9245_audit_20260513/`. The script exposes `ProtoSSM`, `StudentCNN`, `StudentCRNN`, `build_training_priors`, `prior_logits_from_tables`, and `postprocess_probs_filewise`, matching Zeyad's shared-Perch integration pattern.
- **Plan artifact:** added `docs/BIRDCLEF_PUBLIC946_CV9245_PORT_PLAN_20260513.md` with source audit, integration pattern, candidate rank weights (`0.02`/`0.05`), runtime/failure gates, and the post-v545 decision rule.
- **Decision:** no Kaggle push until v545 scores. If v545 ties/drops, implement source-clean public946+CV9245 as the leading v546 candidate and gate it against the tuned public946-gate candidate and train-audio-head before spending a slot.

### Train-audio-head sidecar preflight while v545 capped — 2026-05-13 19:45 UTC

- **Status check:** latest scored submissions remain `v544=0.946`, `v543=0.946`, `v538=0.930`, `v542=0.946`, `v541=0.946`; current best remains **0.946 public LB**. `v545` remains COMPLETE/no failure but unsubmitted behind the daily cap. `v510` remains COMPLETE/no failure with real SED manifest found, `6/6` TorchScript models loaded, blend `0.05` applied, and wall time `370.6s`.
- **Queue fix:** guarded `v545` monitor pid `35141` had staled/exited. Rechecked recent 200 submissions (`v545` absent), then restarted guarded monitor as pid `86320`, log `logs/submit_v545_when_ready_20260513T194622Z_restart3.log`; it immediately re-hit daily cap with about `4.2h` remaining and is sleeping. No duplicate submission was created.
- **Track:** P2 next distinct-signal preparation while waiting for v545 score.
- **Train-audio-head preflight:** audited Henry's `bc2026-raunak0946-direct-v44` train-audio-head branch and public dataset `konbu17/bird26-train-audio-head-v1`. Dataset is public/attachable and contains `head_weights_train_audio.npz` (~1.44 MB). Ignored local copy: `artifacts/public946_train_audio_head_audit_20260513/head_weights_train_audio.npz`. NPZ keys/shapes: `W (234,1536) float32`, `b (234,) float32`, `trained_mask (234,) bool`, `feature_dim=1536`, `notes`.
- **Plan artifact:** added `docs/BIRDCLEF_PUBLIC946_TRAIN_AUDIO_HEAD_PLAN_20260513.md` with source audit, integration pattern, candidate weights (`0.03`/`0.05`), runtime/failure gates, and post-v545 decision rule.
- **Decision:** no Kaggle push until v545 scores. If v545 ties/drops, rank train-audio-head against CV9245 and tuned public946 gates before spending the v546 slot.

### Train-audio-head local gate prerequisite check — 2026-05-13 20:45 UTC

- **Status check:** latest scored submissions remain `v544=0.946`, `v543=0.946`, `v538=0.930`, `v542=0.946`, `v541=0.946`; current best remains **0.946 public LB**. `v545` remains COMPLETE/no failure but unsubmitted behind the daily cap. `v510` remains COMPLETE/no failure with real SED manifest found, `6/6` TorchScript models loaded, blend `0.05` applied, and wall time `370.6s`.
- **Queue/monitor:** guarded `v545` submit monitor pid `86320` is alive and sleeping after a cap response with about `4.2h` remaining from the 19:46 UTC restart. No duplicate `v545` submission exists.
- **Train-audio-head gate check:** inspected available local public946 artifacts. `artifacts/public946_train_audio_head_audit_20260513/head_weights_train_audio.npz` is present and valid, but the downloaded public946 artifacts only preserve the train-cache `perch_arrays.npz`, not the dry-run `emb_te` matrix that produced `submission.csv`. Therefore an exact local train-audio-head sidecar CSV cannot be reconstructed from current artifacts alone.
- **Plan update:** updated `docs/BIRDCLEF_PUBLIC946_TRAIN_AUDIO_HEAD_PLAN_20260513.md` to require the v546 train-head implementation to write `submission_train_audio_head.csv` during Kaggle dry-run, download it, then run `scripts/birdclef_public946_sidecar_weight_grid.py` before submission.
- **Decision:** no Kaggle push until v545 scores. This keeps train-audio-head ready but prevents a false local gate based on mismatched train-cache embeddings.

### v546 decision matrix while v545 capped — 2026-05-13 21:45 UTC

- **Status check:** latest scored submissions remain `v544=0.946`, `v543=0.946`, `v538=0.930`, `v542=0.946`, `v541=0.946`; current best remains **0.946 public LB**. `v545` remains COMPLETE/no failure and unsubmitted behind the daily cap; no `v545` submission is visible in recent submissions. `v510` remains COMPLETE/no failure with real SED manifest found, `6/6` TorchScript models loaded, blend `0.05` applied, and wall time `370.6s`.
- **Queue/monitor:** guarded `v545` submit monitor pid `86320` is alive and sleeping after the 19:46 UTC cap response; no duplicate submission exists.
- **Track:** P2/F pre-submit planning while waiting for v545 score.
- **Plan artifact:** added `docs/BIRDCLEF_PUBLIC946_V546_DECISION_MATRIX_20260513.md`, ranking lower-CLAP, CV9245, train-audio-head, tuned gates, and BirdNET stop conditions by v545 outcome.
- **Decision:** no Kaggle push before v545 scores. If v545 ties/drops, the leading next slot should be a source-clean train-audio-head or CV9245 dry-run with sidecar-grid evidence; if v545 improves, compare smaller CLAP (`0.01`/`0.02`) against those sidecars before choosing v546.

### v545 CLAP sidecar lower-weight gate — 2026-05-13 23:45 UTC

- **Status check:** latest scored submissions remain `v544=0.946`, `v543=0.946`, `v538=0.930`, `v542=0.946`, `v541=0.946`; no `v545` submission is visible yet. `v545` kernel remains COMPLETE/no failure, output files are present, and guarded submit monitor pid `86320` is alive/sleeping after daily cap. `v510` remains COMPLETE/no failure with real SED manifest, `6/6` TorchScript models loaded, blend `0.05`, and wall time `370.6s`.
- **Track:** P2/F public946 + CLAP sidecar gate while waiting for the cap reset; no Kaggle push.
- **Command:** ran `scripts/birdclef_public946_sidecar_weight_grid.py` with v542 `submission.csv` as anchor and v545 `submission_clap_onnx.csv` as the sidecar, labels `/Volumes/ExternalSSD/data/workspace_don/kaggle_birdclef2026/data/train_soundscapes_labels.csv`, weights `0,0.005,0.01,0.02,0.03,0.05,0.075,0.10`.
- **Artifact:** ignored local JSON `artifacts/blend_grids/v545_clap_sidecar_weight_grid_20260513T2345Z.json`.
- **Result:** CLAP standalone is anti/near-uncorrelated with the anchor (`corr=-0.028`, `macro_auc=0.455`, `top3=0.121`). The only dry-run AUC improvement is tiny at `0.005` (`0.992549` vs anchor `0.992525`, top3 lower `0.511` vs `0.521`); `0.01+` drops AUC and `0.05` drops to `0.989306`.
- **Decision:** if `v545` ties/drops, do not spend v546 on another CLAP-only weight. If `v545` unexpectedly improves, only consider a very small `0.005`/`0.01` follow-up after comparing against CV9245/train-audio-head gates.

### v545 submitted after UTC reset; pending score — 2026-05-14 00:45 UTC

- **Status check:** the cap-reset monitor submitted `v545` at `2026-05-14T00:00:25.983Z`, ref `52630458`, description `v545: Public946 v542 plus source-clean CLAP INT8 ONNX 3-way rank blend 57/38/5`. It is visible in the submissions list but has no public score/status yet. Latest scored submissions remain `v544=0.946`, `v543=0.946`, `v538=0.930`, `v542=0.946`, `v541=0.946`; current best remains **0.946 public LB**.
- **Monitor:** `logs/submit_v545_when_ready_20260513T194622Z_restart3.log` shows the retry succeeded with `Submission result: {"message": "", "ref": 52630458}`. The monitor process has exited after the successful submission; no duplicate `v545` submission is visible.
- **v510 check:** still COMPLETE/no failure; `submission.csv` exists and logs confirm real SED manifest found, `6/6` TorchScript models loaded, blend `0.05` applied, and wall time `370.6s`.
- **Decision:** keep the no-new-Kaggle-push gate until `v545` receives a public score. If `v545` ties/drops, v546 should pivot away from CLAP-only to source-clean train-audio-head or CV9245 dry-run plus sidecar-grid evidence. If it improves, compare tiny CLAP `0.005`/`0.01` with those candidates before spending another slot.

### v545 dropped; v546 train-audio-head dry-run and submission — 2026-05-14 02:45 UTC

- **Status check:** `v545` scored `0.944`, below the locked `0.946` public946 anchor, so CLAP-only follow-ups are stopped for public slots. Latest scored: `v545=0.944`, `v544=0.946`, `v543=0.946`, `v538=0.930`, `v542=0.946`, `v541=0.946`; current best remains **0.946 public LB**.
- **v510 check:** remains COMPLETE/no failure; `submission.csv` exists and logs confirm real SED manifest found, `6/6` TorchScript models loaded, blend `0.05`, wall time `370.6s`.
- **Track:** P2 public946 + train-audio-head sidecar after CLAP drop.
- **Implementation:** added `kaggle-kernels/v546-public946-train-audio-head/` and `scripts/push_v546.py` on PR #229. The kernel forks v542, attaches `konbu17/bird26-train-audio-head-v1`, locates `head_weights_train_audio.npz`, hard-fails on missing/misaligned `emb_te`, applies a 5% class-masked head rank blend to `202/234` trained classes, writes `submission_train_audio_head.csv`, then writes final `submission.csv`.
- **Kernel:** pushed real Kaggle kernel `yourslewis/bc26-v546-public946-train-audio-head`, version 1; push returned no invalid dataset/competition/kernel/model sources. Kernel COMPLETE/no failure and outputs include `submission.csv`, `submission_train_audio_head.csv`, `submission_protossm.csv`, `submission_sed.csv`, `perch_arrays.npz`, and `perch_meta.parquet`. Log confirms `trained_classes=202/234`, `head_prob_range=(0.000001,0.999939)`, `head_prob_mean=0.094336`, and `submission_train_audio_head.csv (240, 235)`.
- **Gate artifacts:** downloaded CSV outputs to ignored `artifacts/kaggle_outputs/v546-public946-train-audio-head/`. Ran `scripts/birdclef_public946_sidecar_weight_grid.py`; JSON artifacts `artifacts/blend_grids/v546_train_audio_head_sidecar_weight_grid_20260514T0245Z.json` and `artifacts/blend_grids/v546_train_audio_head_final_vs_v542_20260514T0245Z.json`. The final-vs-v542 check shows high correlation (`corr=0.9984`), low displacement (`MAE=0.0112`, `max_abs=0.0833`), no NaNs, but lower dry-run macro AUC (`0.991756` vs anchor `0.992525`); sidecar reblend at 0.25 gives tiny AUC +0.0000006 and top3 +1.05pp. Because train-soundscape dry-run is leakage-prone and the audited source claims a hidden tie-break gain while public display stays 0.946, this is a bounded-risk slot after CLAP failed.
- **Submission:** submitted v546 code submission ref `52633928`, description `v546: Public946 v542 plus source-clean train-audio-head rank blend 5% trained classes`; score pending.

### v547 CV9245 AutoResearch sidecar, fixes, gate, and submission — 2026-05-14 06:45 UTC

- **Status check:** current best remains **0.946 public LB**. `v546` train-audio-head scored `0.946` (safe tie), `v545` CLAP scored `0.944` (drop), and `v544`/`v543`/`v542`/`v541` remain `0.946`. UTC daily submissions used before v547: `2/5`, so three slots remained.
- **v510 check:** still COMPLETE/no failure; `submission.csv` exists and logs confirm real SED manifest found, `6/6` TorchScript models loaded, blend `0.05`, wall time `370.6s`.
- **Track:** P2/F AutoResearch ensemble tuning around public946 after CLAP dropped and train-audio-head tied.
- **Implementation:** added `kaggle-kernels/v547-public946-cv9245/` and `scripts/push_v547.py` on PR #229. Kernel forks v542, attaches `chaneyma/birdclef-2026-cv9245-moe-artifacts`, imports `pantanal_infer_only_submission.py`, reuses public946 `sc_te`/`emb_te`, loads four CV9245 MoE folds plus student CNN, uses `CV9245_PRIOR_SCALE=0.45`, `CV9245_CNN_WEIGHT=0.20`, and final `CV9245_RANK_BLEND=0.02`. It writes `submission_cv9245_cnnonly_sharedperch.csv` and final `submission.csv`; missing/misaligned sources hard-fail.
- **Fixes:** version 1 failed on `site_emb.weight` shape mismatch because the exported folds expect `n_sites=10`; fixed by using the exact CV9245 training site count. Version 2 completed the sidecar but failed final blend because `p_cv9245` was not loaded/aligned; fixed in version 3 by reading `submission_cv9245_cnnonly_sharedperch.csv` before ranking.
- **Kernel:** pushed real Kaggle kernel `yourslewis/bc26-v547-public946-cv9245-sidecar`, version 3; COMPLETE/no failure. Outputs include `submission.csv`, `submission_cv9245_cnnonly_sharedperch.csv`, `submission_protossm.csv`, `submission_sed.csv`, `perch_arrays.npz`, `perch_meta.parquet`. Log confirms CV9245 artifacts found, `n_sites=10`, sidecar `(240,235)`, prob range `(0.000001,0.963308)`, mean `0.028908`, sidecar runtime `13.7s`, applied final CV9245 rank blend `0.02`, mirroring 10 columns, rare thresholding 44 species.
- **Gate artifacts:** downloaded outputs to ignored `artifacts/kaggle_outputs/v547-public946-cv9245-sidecar/`. Sidecar grid JSON `artifacts/blend_grids/v547_cv9245_sidecar_weight_grid_20260514T0645Z.json`: CV9245 standalone is low-corr/different (`corr=0.6699`, `macro_auc=0.9657`), while 2% blend has `corr=0.999888`, `MAE=0.00362`, `max_abs=0.01917`, top3 `0.5421` vs anchor `0.5211`, top5 `0.6579` vs anchor `0.6316`, and only tiny AUC drop (`0.992509` vs `0.992525`). Final-vs-v542 JSON `artifacts/blend_grids/v547_cv9245_final_vs_v542_20260514T0645Z.json`: final `submission.csv` has `corr=0.999727`, `MAE=0.00436`, `max_abs=0.0500`, `macro_auc=0.992546` vs anchor `0.992525`, no NaNs.
- **Submission:** submitted v547 code submission ref `52639661`, description `v547: Public946 v542 plus source-clean CV9245 sidecar rank blend 2%`; score pending.

### v548 CV9245 0.5% low-displacement follow-up — 2026-05-14 08:20 UTC

- **Status check:** `v547` CV9245 2% scored `0.946`, tying the public946 anchor. Current best remains **0.946 public LB**. `v546` train-audio-head also tied `0.946`; `v545` CLAP dropped to `0.944`. UTC daily submissions before v548: `3/5`, so two slots remained.
- **v510 check:** still COMPLETE/no failure; `submission.csv` exists and logs confirm real SED manifest found, `6/6` TorchScript models loaded, blend `0.05`, wall time `370.6s`.
- **Track:** P2/F AutoResearch ensemble weight optimization after v547 tied.
- **Implementation:** added `kaggle-kernels/v548-public946-cv9245-w0005/` and `scripts/push_v548.py` on PR #229. It reuses the source-clean v547 CV9245 sidecar but lowers `CV9245_RANK_BLEND` from `0.02` to `0.005` to reduce anchor displacement.
- **Kernel:** pushed real Kaggle kernel `yourslewis/bc26-v548-public946-cv9245-w0005`, version 1; COMPLETE/no failure. Outputs include `submission.csv`, `submission_cv9245_cnnonly_sharedperch.csv`, `submission_protossm.csv`, `submission_sed.csv`, `perch_arrays.npz`, and `perch_meta.parquet`. Log confirms CV9245 artifacts found, `n_sites=10`, sidecar `(240,235)`, prob range `(0.000001,0.963308)`, mean `0.028908`, sidecar runtime `13.1s`, final CV9245 rank blend `0.005`, mirroring 10 columns, rare thresholding 44 species.
- **Gate basis:** prior combo/grid artifact `artifacts/blend_grids/v548_head_cv9245_combo_grid_20260514T0700Z.json` and v547 sidecar grid show the 0.5% CV9245 blend is the lowest-displacement safe follow-up (`corr=0.999993`, `MAE=0.000906`, `max_abs=0.00479`, dry-run macro AUC `0.992505` vs anchor `0.992525`, top3 `0.5368` vs anchor `0.5211`).
- **Submission:** submitted v548 code submission ref `52642350`, description `v548: Public946 v542 plus source-clean CV9245 sidecar rank blend 0.5%`; score pending.

### v549 CV9245 1% bracket candidate and final slot — 2026-05-14 09:25 UTC

- **Status check:** `v548` CV9245 0.5% scored `0.946`, tying the public946 anchor. Current best remains **0.946 public LB**. UTC daily submissions before v549: `4/5`, so one slot remained.
- **Track:** P2/F AutoResearch ensemble weight bracket after both CV9245 2% (`v547`) and 0.5% (`v548`) tied.
- **Implementation:** added `kaggle-kernels/v549-public946-cv9245-w001/` and `scripts/push_v549.py` on PR #229. It reuses the v547/v548 source-clean CV9245 sidecar and sets final `CV9245_RANK_BLEND=0.01`, exactly between the tied 0.5% and 2% runs.
- **Kernel:** pushed real Kaggle kernel `yourslewis/bc26-v549-public946-cv9245-w001`, version 1; COMPLETE/no failure. Outputs include `submission.csv`, `submission_cv9245_cnnonly_sharedperch.csv`, `submission_protossm.csv`, `submission_sed.csv`, `perch_arrays.npz`, `perch_meta.parquet`. Log confirms CV9245 artifacts found, `n_sites=10`, sidecar `(240,235)`, prob range `(0.000001,0.963308)`, mean `0.028908`, sidecar runtime `14.2s`, final CV9245 rank blend `0.01`, mirroring 10 columns, rare thresholding 44 species.
- **Submission:** submitted v549 code submission ref `52644106`, description `v549: Public946 v542 plus source-clean CV9245 sidecar rank blend 1%`; score pending. This used the final known UTC submission slot (`5/5`).

### v549 pending and daily quota exhausted; next-after-CV9245 plan — 2026-05-14 09:45 UTC

- **Status check:** latest submissions show `v549` pending, `v548=0.946`, `v547=0.946`, `v546=0.946`, `v545=0.944`; current best remains **0.946 public LB**. UTC daily submissions are now `5/5`, so no further competition submission is possible this UTC day without hitting cap.
- **v510 check:** still COMPLETE/no failure; `submission.csv` exists and logs confirm real SED manifest found, `6/6` TorchScript models loaded, blend `0.05`, wall time `370.6s`.
- **Decision:** hold all further submissions until next UTC reset and v549 score. Do not spend more effort on CV9245-only weights unless v549 improves; if v549 ties/drops, move to a new sidecar family.
- **Plan artifact:** added `docs/BIRDCLEF_PUBLIC946_NEXT_AFTER_CV9245_20260514.md`. Priority after v549: (1) wait for score, (2) build source-clean Snowflake SED dry-run and gate `submission_snowflake_sed.csv`, (3) test tiny combined hidden-diversity blends only after component outputs exist, (4) use public946 gate retune only as fallback.

### v550 Snowflake SED sidecar pushed for dry-run gate — 2026-05-14 10:55 UTC

- **Status check:** `v549` scored `0.946`, tying v548/v547/v546; current best remains **0.946 public LB**. UTC submissions are `5/5`, so no new competition submission was attempted.
- **v510 check:** still COMPLETE/no failure; `submission.csv` exists and logs confirm real SED manifest found, `6/6` TorchScript models loaded, blend `0.05`, wall time `370.6s`.
- **Track:** public946 new sidecar signal after CV9245-only bracket exhausted. Implemented `v550` from `v542` with Tsubasa Kanno's public Snowflake SED dataset (`tsubasatech/birdclef-2026-snowflake-sed`), loading `sed_convnext-tiny_fold0.onnx` and `sed_tf-efficientnetv2-m_fold0.onnx`, writing `submission_snowflake_sed.csv`, and applying conservative `SNOWFLAKE_RANK_BLEND=0.01` to final `submission.csv`.
- **Validation before push:** `python3 -m py_compile kaggle-kernels/v550-public946-snowflake-sed-w001/script.py scripts/push_v550.py` passed. Kaggle push succeeded as private kernel `yourslewis/bc26-v550-public946-snowflake-sed-w001`, version 1, with no invalid data/kernel/model sources.
- **Monitor state:** v550 is RUNNING/no failure at log time. Do not submit it under current UTC cap; after dry-run completion, download outputs and run `scripts/birdclef_public946_sidecar_weight_grid.py` on `submission_snowflake_sed.csv` vs v542 before choosing any next-day slot.

### v550 Snowflake SED dry-run gate complete — 2026-05-14 11:10 UTC

- `v550` Kaggle kernel completed successfully with no failure message. Downloaded outputs to `artifacts/kaggle_outputs/v550-public946-snowflake-sed-w001/`: `submission.csv`, `submission_snowflake_sed.csv`, `submission_sed.csv`, and `submission_protossm.csv`, all validated as `(240, 235)` with no NaNs.
- Ran the no-submit gate monitor `scripts/monitor_v550_snowflake_gate.py`; output grid: `artifacts/blend_grids/v550_snowflake_sidecar_weight_grid_20260514T110730Z.json`.
- Gate result vs v542 public946 anchor on train-soundscape overlap: anchor/weight 0 macro AUC `0.992525`; Snowflake standalone rank macro AUC `0.735086` and low correlation vs anchor `0.3728`. Tiny blends did **not** improve macro AUC: 0.25%/0.5% `0.992502`, 1% `0.992476`, 2% `0.992103`, 5% `0.990878`.
- Decision: hold v550 for now; do **not** spend a capped competition slot on Snowflake standalone/1% unless a later ensemble policy values its top-k behavior despite AUC degradation. Next better action is a different source-clean sidecar or teacher/student route, not submitting v550 automatically.

### v551 tiny CLAP sidecar candidate prepared — 2026-05-14 11:20 UTC

- Ran a local multi-sidecar gate over v542 anchor plus CLAP, CV9245, train-audio-head, and Snowflake sidecars: `artifacts/blend_grids/v551_multisidecar_snowflake_combo_grid_20260514T1115Z.json` (ignored artifact, local only).
- Best overlap macro AUC was a tiny CLAP sidecar: v542 + `0.5%` CLAP rank (`0.992549`) vs v542 anchor `0.992525`. CV9245/head did not improve macro AUC further in the tested small-weight grid; Snowflake was not selected.
- Prepared `v551` as a conservative follow-up to failed v545 5% CLAP: it runs the same source-clean CLAP INT8 side stream but keeps the public946 Proto/SED gates intact and applies `CLAP_RANK_BLEND=0.005` after the gates.
- Validation: `python3 -m py_compile kaggle-kernels/v551-public946-clap-int8-w0005/script.py scripts/push_v551.py` passed. Kaggle push succeeded as private kernel `yourslewis/bc26-v551-public946-clap-int8-w0005`, version 1, with no invalid data/kernel/model sources. v551 is RUNNING/no failure; no-submit gate monitor started via generalized `scripts/monitor_v550_snowflake_gate.py` (`logs/monitor_v551_clap_gate_*.log`). Do not submit under current UTC cap.

### v551 complete + gated; guarded submit monitor started — 2026-05-14 11:50 UTC

- `v551` completed successfully with no failure message. Downloaded outputs to `artifacts/kaggle_outputs/v551-public946-clap-int8-w0005/`: `submission.csv`, `submission_clap_onnx.csv`, `submission_sed.csv`, and `submission_protossm.csv`, all validated as `(240, 235)` with no NaNs.
- The first v551 no-submit monitor died on a transient Kaggle `RemoteDisconnected`; hardened `scripts/monitor_v550_snowflake_gate.py` with retry wrappers for status/list/download and reran the gate successfully.
- Gate output: `artifacts/blend_grids/v551_clap_sidecar_weight_grid_20260514T114751Z.json`. v542 anchor macro AUC `0.992525`; CLAP standalone rank is bad (`0.455042`, corr `-0.028`), but tiny blends are slightly positive at 0.25%-0.5%: `0.0025 -> 0.992538`, `0.005 -> 0.992549`; 1%+ degrades.
- Decision: `v551` is the single next-slot candidate, not a widened CLAP lane. Started guarded submit script `scripts/submit_v551_when_ready.py`; it exits on duplicate descriptions, requires `submission.csv`, and backs off on daily cap. Current UTC cap is still 5/5, so it should sleep until reset if the submit attempt hits allowance.

### public946 ConvNeXt-tiny rankblend student smoke + scale — 2026-05-14 12:55 UTC

- Status before training: public LB best remains **0.946**; v551 is COMPLETE/gated and guarded submit monitor pid `53054` is sleeping after daily cap. v510 remains COMPLETE with `submission.csv`.
- Track: **B/D** public946 pseudo-label/noisy-student + model-zoo diversity. Hypothesis: ConvNeXt-tiny trained on the public946 rankblend teacher may provide a less-correlated student sidecar than the prior NFNet student while keeping enough AUC for a tiny teacher blend.
- Added configs: `configs/birdclef/pl_public946_rankblend_convnext_tiny_5s_lr3e4_smoke.json` and `configs/birdclef/pl_public946_rankblend_convnext_tiny_5s_lr3e4_ep20_bestval.json`.
- Smoke run on GPU server `192.168.0.10`, GPU1, config `pl_public946_rankblend_convnext_tiny_5s_lr3e4_smoke`: max_rows 256, 5s/160mel, ConvNeXt-tiny, lr 3e-4, ep3, soft public946 rankblend teacher. It completed in `9.7s`: final AUC `0.882870` over 42 valid classes, teacher AUC `0.990095`, corr `0.737764`, MAE `0.134967`. This beat the earlier public946 rankblend NFNet smoke (`0.853951`) and V2S smoke (`0.835216`), so it passed the scale gate.
- Scaled run `pl-public946-rankblend-convnext-tiny-5s-lr3e4-ep20-bestval` completed on GPU1 in `55.5s`: best val AUC `0.985183` at epoch 20 over 61 classes; final all-row student AUC `0.987875` over 75 classes vs teacher `0.994567`; student-teacher corr `0.943076`, MAE `0.061591`; TorchScript size `112.355 MB`.
- Blend diagnostic on server artifact `artifacts/pseudolabels/students/pl-public946-rankblend-convnext-tiny-5s-lr3e4-ep20-bestval/blend_grid.json`: best rank blend teacher+student is student weight `0.075`, AUC `0.994618` (+`0.000051` vs teacher); best probability blend is student weight `0.02`, AUC `0.994609` (+`0.000042`). This is a viable private-robustness sidecar but smaller than v551's immediate next-slot priority; do not package/submit until v551 score lands.

### v552 ConvNeXt student Kaggle candidate pushed — 2026-05-14 13:55 UTC

- Status before push: best remains **0.946**; v551 guarded submit monitor pid `53054` remains alive/sleeping after daily cap; v510 remains COMPLETE with `submission.csv`.
- Packaged the scaled public946 ConvNeXt-tiny student into private Kaggle dataset `yourslewis/bc26-public946-convnext-tiny-student-v1` (version 1, READY). Dataset contains `model_torchscript.pt` and `sed_bundle_manifest.json`; zip size `99.4 MB`, SHA256 `f68e56748adf9fae818b265cadb7adb28085dfa7aa773ab0e9e1c5bc1ceca45b`.
- Added/pushed kernel `yourslewis/bc26-v552-public946-convnext-student-r075`, version 1. It forks v542 public946, runs the ConvNeXt student TorchScript sidecar to write `submission_convnext_student.csv`, then applies a conservative post-gate per-class rank blend `STUDENT_RANK_BLEND=0.075`. Kaggle push succeeded with no invalid data/kernel/model sources.
- Started a no-submit gate monitor via generalized `scripts/monitor_v550_snowflake_gate.py` (`logs/monitor_v552_convnext_gate_*.log`). Do **not** submit v552 while v551 is queued for next reset; gate it offline first and wait for v551 score.

### v552 ConvNeXt student completed + no-submit gate — 2026-05-14 14:55 UTC

- `v552` completed successfully with no failure message. Downloaded outputs to `artifacts/kaggle_outputs/v552-public946-convnext-student-rank075/`: `submission.csv`, `submission_convnext_student.csv`, `submission_sed.csv`, and `submission_protossm.csv`, all validated as `(240,235)` with no NaNs.
- Gate output: `artifacts/blend_grids/v552_convnext_student_sidecar_weight_grid_20260514T140456Z.json`. ConvNeXt student standalone rank is reasonably competitive but weaker than anchor on the train-soundscape overlap: standalone macro AUC `0.979342`, corr vs anchor `0.904834`.
- Blends vs v542 anchor macro AUC `0.992525`: 2% `0.992502`, 5% `0.992430`, 7.5% `0.992354`, 10% `0.992341`, 15% `0.992014`. Top-k row recall increases at larger weights, but macro AUC consistently degrades.
- Decision: **hold/no-submit v552**. Keep the packaged ConvNeXt student as an artifact/private-robustness option only; do not spend a competition slot before v551 scores. The live submit monitor remains only `v551` pid `53054`, sleeping after daily cap.

### v553 taxon max gate option added for public946/ConvNeXt testing — 2026-05-14 15:30 UTC

- Added a reusable v517-style taxon max gate option to `scripts/birdclef_public946_weight_grid.py` via `--taxon-gate`, `--taxon-floors`, and `--taxon-alphas`, so public946 Proto/SED rank-blend variants can be swept with the same taxon evidence gate that previously lifted the v508 axis.
- Local dry-run grid: `artifacts/blend_grids/public946_taxon_gate_option_grid_20260514.json` over floors `0.20,0.30,0.40` and alphas `0.375,0.50,0.75`. On train-soundscape overlap the taxon gate is not additive to public946: for the v542-weight `0.60/0.40` blend, best taxon variant was `0.992376` vs base `0.992525` (`-0.000149`); all tested Proto weights were slightly negative. This is only a public/train-overlap gate, not final LB proof.
- Added optional constants to the v552 final blend (`APPLY_TAXON_MAX_GATE`, `TAXON_MAX_GATE_FLOOR=0.30`, `TAXON_MAX_GATE_ALPHA=0.50`) and created a live test fork `v553` with the gate enabled.
- Pushed private Kaggle kernel `yourslewis/bc26-v553-public946-convnext-r075-taxon-a050`, version 1, with no invalid data/kernel/model sources. Started a no-submit monitor `logs/monitor_v553_taxon_gate_*.log`; v553 is for output/gate validation only and should not displace the guarded v551 submit candidate.


### Spec B 792-row public946 student ensemble audit + blended-teacher smoke — 2026-05-15 06:55 UTC

- **Status check:** current public best remains **0.946**; latest submissions `v558/v551/v549/v548/v547` all scored `0.946`. No Kaggle submission used. `v510` remains COMPLETE/no failure.
- **Audit:** compared existing 792-row public946 students on trainer. B0 SED student AUC `0.976669` vs teacher `0.996743`; ConvNeXt rankblend student `0.987875` vs teacher `0.994567`; NFNet rankblend student `0.984806` vs teacher `0.994567`. Student blends did not provide enough lift to package.
- **Teacher blend finding:** teacher-level blend was stronger: `0.85 * teacher_sed + 0.15 * teacher_rankblend` reached macro AUC `0.997018` over 75 classes, beating SED teacher `0.996743` and rankblend teacher `0.994567`. Added `scripts/birdclef_blend_teacher_npz.py` and created trainer artifact `artifacts/pseudolabels/public946-v540-teacher-cache66-v1/teacher_sed85_rankblend15.npz` plus summary.
- **Smoke:** added `configs/birdclef/pl_public946_sed85_rankblend15_convnext_tiny_5s_smoke_20260515.json` and ran ConvNeXt-tiny 256-row/3-epoch smoke on blended teacher. It completed on CUDA in `10.04s`, but final student AUC was only `0.799819` over 42 classes vs teacher `0.995304`, corr `0.42485`, MAE `0.04334`.
- **Decision:** kill direct blended-teacher ConvNeXt scaling. The blended teacher cache is useful for future target design, but this learner/curriculum is worse than the prior public946 rankblend ConvNeXt smoke (`0.882870`) and should not be packaged or submitted.


### Spec B public946 SED soft-anchor supervised pivot — 2026-05-15 05:55 UTC

- **Status check:** current public best remains **0.946**; latest submissions `v558/v551/v549/v548/v547` all scored `0.946`. No Kaggle submission used. `v510` remains COMPLETE/no failure.
- **Chosen track:** Spec B target-design pivot after hard-conf B0 was low-correlation but too weak.
- **Partial-supervision run:** added/reran `configs/birdclef/pl_public946_v542_sed_softanchor_supervised_b0_5s_ep12_20260515.json` on GPU. Soft-anchor target (`p>=0.8`, `p<=0.005`, soft weight 0.5) + intended 160 supervised clips from `data/train.csv`. Only `38/160` supervised clips loaded because the remote train-audio mirror is partial and sampling happened before path filtering. Final AUC `0.911316`, best val AUC `0.903733`, corr `0.869918`, no blend lift.
- **Existing-audio manifest fix:** built `artifacts/pseudolabels/manifests/train_existing_audio_manifest_20260515.csv` on trainer from `3388` existing audio files across `206` classes; reran `configs/birdclef/pl_public946_v542_sed_softanchor_supervised_existing160_b0_5s_ep12_20260515.json`. This used `160/160` supervised clips, zero missing paths.
- **Existing160 result:** final student AUC `0.936198` over 42 classes, best val AUC `0.926750` over 34 classes, student-teacher corr `0.908199`, MAE `0.046395`, TorchScript `15.391 MB`, runtime `12.469s`.
- **Blend gate:** `blend_gate.json` shows no useful local lift: SED teacher baseline `0.995316`; student blend ties only at `w=0.005` and drops at `w>=0.01`; rankblend blends all drop. Decision: no packaging/submission. Reuse the path-filtered manifest, but next work needs a larger/stronger teacher cache or model-family change.


### Spec B public946 SED hard-confidence B0 full diagnostic — 2026-05-15 04:55 UTC

- **Status check:** current public best remains **0.946**; latest submissions `v558/v551/v549/v548/v547` all scored `0.946`. No Kaggle submission used. `v510` remains COMPLETE/no failure.
- **Environment check:** local Mac venv has Torch CPU but no `timm`; GPU server `192.168.0.10` has CUDA + `timm`. Synced updated training scripts, `teacher_sed.npz`, and new config to `~/birdclef-2026` on the GPU server.
- **Config:** added `configs/birdclef/pl_public946_v542_sed_hardconf_b0_5s_ep20_20260515.json`: EfficientNet-B0, external-pretrain init, all 240 v542 teacher rows, hard-conf `positive_threshold=0.8`, `negative_threshold=0.005`, pos cap 3/row, neg cap 64/row, 20 epochs, best-val restore.
- **Run:** `CUDA_VISIBLE_DEVICES=1 python scripts/birdclef_pseudolabel_student_train.py --config configs/birdclef/pl_public946_v542_sed_hardconf_b0_5s_ep20_20260515.json`; log `logs/pl_public946_v542_sed_hardconf_b0_5s_ep20_20260515.log`. Completed on CUDA in `8.866s`, exported TorchScript `15.391 MB`.
- **Metrics:** final student AUC `0.750030` over 42 valid classes vs teacher `0.995316`; best val AUC `0.812170`; student-teacher corr `0.172628`, MAE `0.384736`; target mask fraction `0.2752`, positives/negatives `97/15360`.
- **Blend gate:** copied predictions back and wrote `blend_gate.json`. Tiny blend into SED teacher has only microscopic local lift (`w=0.01`: `0.995331` vs teacher `0.995316`); blend into rankblend does not improve. Decision: do not package/submit this hard-conf student. Next Spec B step should alter target design (soft-anchor + supervised mix or larger 792-row teacher cache) rather than scaling this exact recipe.


### Spec B public946 SED teacher cache + hard-confidence smoke — 2026-05-15 04:05 UTC

- **Status check:** current public best remains **0.946**. Latest scored submissions remain `v558=0.946`, `v551=0.946`, `v549/v548/v547/v546=0.946`; no Kaggle slot used this run. `v510` remains COMPLETE/no failure.
- **Chosen track:** Spec B pseudo-label/noisy-student, because public946 retunes are stopped after repeated ties/drops.
- **Teacher cache:** generated cache from `artifacts/kaggle_outputs/v542-afr1ste-updated-public946` into `artifacts/public946_teacher_cache_v542_20260515T0355Z/`. `teacher_sed.npz` is the cleanest seed: labeled-overlap macro AUC `0.995976`, row top1/top3/top5 recall `0.978947/0.989474/0.994737`. `teacher_rankblend.npz` has macro AUC `0.992525` but very dense hard positives and much worse top-k row recall.
- **Threshold sweep:** `artifacts/pseudolabel_thresholds/public946_v542_sed_threshold_sweep_20260515T0355Z.json` recommends SED hard-confidence positives at `power=1.0`, `positive_threshold=0.8`; overlap precision `97/97 = 1.0`, `66` positive rows, `8` positive classes. Use `negative_threshold=0.005` as the practical smoke/default (mask fraction `0.779`) or `0.001` as more conservative.
- **Smoke train/export:** added config `configs/birdclef/pl_public946_sed_hardconf_smoke8_20260515.json` and ran `scripts/birdclef_pseudolabel_student_train.py` on `max_rows=8`, one epoch, hard-conf p0.8/n0.005. It completed on CPU in `4.777s`, exported TorchScript `0.184 MB`, and wrote `artifacts/pseudolabels/students/pl-public946-sed-hardconf-smoke8-20260515/metrics.json`. Actual backbone was `tiny_cnn_sed` fallback, so this validates plumbing only; next quality run should use GPU/server/timm environment.
- **Plan doc:** added `docs/BIRDCLEF_PUBLIC946_PSEUDOLABEL_TEACHER_20260515.md`. Do not submit a student kernel until a full/OOF artifact shows competitive diagnostics and lower correlation to public946.


### v558 tied; stop public946 retune lane — 2026-05-15 02:55 UTC

- **Status check:** `v558` completed with public score **0.946**, tying but not improving the public946 anchor. `v551` also tied **0.946**. Current best remains **0.946 public LB**. `v510` remains COMPLETE/no failure with `submission.csv`.
- **Interpretation:** Low-displacement public946 retunes/sidecars have repeatedly tied (`v543/v544/v546/v547/v548/v549/v551/v558`) or dropped (`v545`). The train-soundscape overlap gates are now mostly useful as rejection filters, not as enough evidence to spend more slots on same-family variants.
- **Research check:** a fresh web search surfaced the same Nina public946 ONNX+Perch+Proto+SED notebook as the high-signal result, not a new distinct source. Local artifact audit shows no remaining source-clean sidecar with both local lift and new-signal evidence strong enough to justify another immediate slot.
- **Plan artifact:** added `docs/BIRDCLEF_PUBLIC946_STOP_RETUNE_NEXT_SIGNAL_20260515.md` documenting the stop rule: no more public946-only postprocess retunes or single-family low-weight brackets unless a new source/model/OFF artifact changes the evidence. Do not submit `v554`-`v557`; `v558` already tested the safest clipped formulation and tied.
- **Decision:** hold remaining UTC daily submissions. Next work should pivot to genuinely new signal: source-clean model-family audit, public946 teacher/noisy-student OOF artifacts, or real SED/student training artifacts with runtime headroom.


### v551 tied; conditional monitor submitted v558 — 2026-05-15 01:55 UTC

- **Status check:** `v551` completed with public score **0.946**, tying but not improving the public946 anchor. Current best remains **0.946 public LB**. `v510` remains COMPLETE/no failure with `submission.csv`. `v558` remains COMPLETE/no failure with clean actual-v542 gate evidence.
- **Conditional monitor:** `scripts/submit_v558_if_v551_ties_or_drops.py` saw v551 score `0.946` (`<=0.946` threshold), verified v558 COMPLETE + `submission.csv`, and submitted exactly once. Log `logs/submit_v558_if_v551_ties_or_drops_20260515T005238Z.log` reports `Submission result: {"message": "", "ref": 52665049}`.
- **Visible submission:** `v558: Public946 v542 plus exact-base clipped gate retune alpha0.10 maxabs0.02` is visible at `2026-05-15 01:12:41.840 UTC`, status PENDING, no public score yet.
- **Decision:** hold all further submissions until v558 scores. Do not submit v554-v557. If v558 ties/drops, the public946 postprocess-retune lane should pause and the next slot should return to genuinely new signal or a fresh evidence source.


### v558 conditional submit monitor prepared while v551 pending — 2026-05-15 00:55 UTC

- **Status check:** `v551` is visible at `2026-05-15 00:00:33.863 UTC`, status PENDING, no public score yet. Current best remains **0.946 public LB** from `v541/v542/v543/v544/v546-v549`. `v510` remains COMPLETE/no failure with `submission.csv`. `v558` remains COMPLETE/no failure with clean actual-v542 gate evidence.
- **Track:** guarded fallback orchestration while waiting for v551 score; no extra competition submission was made in this run.
- **Implementation:** added `scripts/submit_v558_if_v551_ties_or_drops.py`. It polls submissions, exits without submitting if `v551` improves above `0.946`, and submits `v558` only if `v551` completes with score `<=0.946` or error/no-score. It also guards against duplicate v558 descriptions and requires the v558 kernel to be COMPLETE with `submission.csv`.
- **Validation:** `python3 -m py_compile scripts/submit_v558_if_v551_ties_or_drops.py` passed.
- **Monitor:** started pid `84410`, log `logs/submit_v558_if_v551_ties_or_drops_20260515T005238Z.log`. Initial check saw `v551` PENDING and slept; no v558 submission attempted.
- **Decision:** continue holding until v551 scores. This monitor is the only conditional fallback path; do not manually submit v558 unless the monitor fails.


### v551 submitted at UTC reset; hold for score — 2026-05-15 00:05 UTC

- **Status check:** immediately before reset, latest scored submissions remained `v549=0.946`, `v548=0.946`, `v547=0.946`, `v546=0.946`, `v545=0.944`; current best remains **0.946 public LB**. `v510` remains COMPLETE/no failure with `submission.csv`. `v558` remains the clean prepared fallback, but no fallback submission was attempted.
- **Submit monitor:** guarded `scripts/submit_v551_when_ready.py` pid `96890` woke after UTC reset, rechecked `v551` COMPLETE + `submission.csv`, and submitted exactly once. Log `logs/submit_v551_when_ready_restart_20260514T155229Z.log` reports `Submission result: {"message": "", "ref": 52663601}`.
- **Visible submission:** `v551: Public946 v542 plus source-clean CLAP INT8 ONNX tiny rank sidecar 0.5%` is now visible at `2026-05-15 00:00:33.863 UTC`, status PENDING, no public score yet.
- **Decision:** wait for v551 public score before spending any more daily slots. If v551 improves, bracket tiny CLAP only after comparing against v558. If v551 ties/drops, prefer prepared v558 over v554-v557 for a retune fallback, but do not submit automatically until the score is known.


### v558 confirmed clean against actual v542 baseline — 2026-05-14 22:55 UTC

- **Status check:** latest Bearer API submissions remain `v549=0.946`, `v548=0.946`, `v547=0.946`, `v546=0.946`, `v545=0.944`; current best remains **0.946 public LB**. `v510` is COMPLETE/no failure with `submission.csv`. `v551` remains COMPLETE and guarded submit monitor pid `96890` is alive/sleeping after daily cap. No duplicate submissions were created.
- **v558 result:** `yourslewis/bc26-v558-gateretune-a010-clip002` version 1 completed and validated `(240,235)` outputs. The first generic gate artifact using reconstructed baseline still showed the same misleading high displacement as v556/v557, so the actual v542 output was downloaded from Kaggle to `artifacts/kaggle_outputs/v542-afr1ste-updated-public946/` and used as the true comparison baseline.
- **Monitor fix:** updated `scripts/monitor_v554_gate_retune.py` to prefer `BASE_CSV` / actual downloaded v542 `submission.csv` when available, falling back to reconstructed baseline only if the actual CSV is missing. Re-ran the monitor for v558 with the actual baseline.
- **Actual-baseline gate:** `artifacts/blend_grids/v558_gateretune_a010_clip002_final_vs_baseline_20260514T225302Z.json` shows baseline macro AUC `0.992525`; v558 macro AUC `0.992630` (`+0.000106`); top3 `0.637` vs baseline `0.626`; top5 unchanged `0.747`; displacement is low (`corr=0.999982`, `MAE=0.001067`, `max_abs=0.016018`). This confirms v558 is the cleanest prepared gate-retune fallback.
- **Decision:** keep `v551` as the only next-reset submit monitor. If v551 ties/drops and a postprocess fallback is needed, v558 is safer than v554/v555/v556/v557. Do not submit v554-v557.


### v557 completed but baseline mismatch persists; v558 exact-base clip launched — 2026-05-14 21:55 UTC

- **Status check:** latest Bearer API submissions remain `v549=0.946`, `v548=0.946`, `v547=0.946`, `v546=0.946`, `v545=0.944`; current best remains **0.946 public LB**. `v510` is COMPLETE/no failure with `submission.csv`. `v551` remains COMPLETE and guarded submit monitor pid `96890` is alive/sleeping after daily cap. No duplicate submissions were created.
- **v557 result:** `yourslewis/bc26-v557-public946-gate-retune-alpha010-clip002` version 1 completed and validated `(240,235)` outputs. Gate artifact: `artifacts/blend_grids/v557_public946_gate_retune_alpha010_clip002_final_vs_baseline_20260514T210406Z.json`. It preserved AUC/top3 lift (`0.992630` vs baseline `0.992525`, top3 `0.637` vs `0.621`) but still showed high displacement (`corr=0.99595`, `MAE=0.00349`, `max_abs=0.38941`).
- **Diagnosis:** v557 final is effectively identical to v556; clipping applied relative to the kernel's internal reconstructed baseline, but that baseline diverges from the monitor's v542 reconstruction on a small number of cells. Therefore v557 is **not** a clean fallback.
- **Implementation:** added `kaggle-kernels/v558-public946-gateretune-a010clip002-exactbase/` and `scripts/push_v558.py`. v558 starts from the exact standard v542 `sub` generated by the kernel, stores those values, computes the retuned branch separately, then adds `clip(0.10 * (retune - exact_base), -0.02, 0.02)` to that exact base. This should make the max delta cap meaningful relative to the final submitted baseline.
- **Validation/push:** `python3 -m py_compile scripts/push_v558.py kaggle-kernels/v558-public946-gateretune-a010clip002-exactbase/script.py` passed. First push failed only because Kaggle title exceeded 50 chars; shortened title and retried successfully. Real private Kaggle kernel is `yourslewis/bc26-v558-gateretune-a010-clip002`, version 1, no invalid sources.
- **Monitor:** an initial monitor used the pre-normalization slug and got 403; killed it and restarted with the actual slug. Active no-submit monitor pid `57008`, log `logs/monitor_v558_gate_retune_20260514T215319Z.log`, initial status RUNNING/no failure.
- **Decision:** keep v551 as the only next-reset submit monitor. v558 is only a prepared fallback if it completes and its gate confirms low displacement; do not submit v556/v557.


### v557 clipped low-displacement gate-retune candidate launched — 2026-05-14 20:55 UTC

- **Status check:** latest Bearer API submissions remain `v549=0.946`, `v548=0.946`, `v547=0.946`, `v546=0.946`, `v545=0.944`; current best remains **0.946 public LB**. `v510` is COMPLETE/no failure with `submission.csv`. `v551` remains COMPLETE and guarded submit monitor pid `96890` is alive/sleeping after daily cap. No duplicate submissions were created.
- **v556 result:** `yourslewis/bc26-v556-public946-gate-retune-alpha010` version 1 completed successfully and validated `(240,235)` outputs. Gate artifact: `artifacts/blend_grids/v556_public946_gate_retune_alpha010_final_vs_baseline_20260514T200307Z.json`. It matched the small AUC/top3 lift (`0.992630` vs baseline `0.992525`, top3 `0.637` vs `0.621`) but a few cells still had large movement (`corr=0.99595`, `MAE=0.00349`, `max_abs=0.38941`), so v556 is not a clean fallback as-is.
- **Post-gate diagnosis:** v556 alpha is correctly ~0.10 for ~98.3% of moved cells; the problem is outlier cells where small baseline/retune denominator differences create large max movement. Clipping observed v556 deltas to `±0.02` preserves the same dry-run AUC/top3 lift while reducing displacement to `corr=0.99993`, `MAE=0.00136`, `max_abs=0.0200`.
- **Implementation:** added `kaggle-kernels/v557-public946-gateretune-a010clip002/` and `scripts/push_v557.py`. v557 uses the same retuned branch and alpha `0.10` as v556, but applies `np.clip(retune_delta, -0.02, 0.02)` before adding it to the reconstructed v542 baseline.
- **Validation:** `python3 -m py_compile scripts/push_v557.py kaggle-kernels/v557-public946-gateretune-a010clip002/script.py` passed. Pushed real private Kaggle kernel `yourslewis/bc26-v557-public946-gate-retune-alpha010-clip002`, version 1; no invalid data/competition/kernel/model sources.
- **Monitor:** started no-submit monitor pid `48210`, log `logs/monitor_v557_gate_retune_20260514T205329Z.log`, with `KERNEL_SLUG=bc26-v557-public946-gate-retune-alpha010-clip002` and `OUTPUT_NAME=v557-public946-gate-retune-alpha010-clip002`. Initial status RUNNING/no failure.
- **Decision:** keep v551 as the only next-reset submit monitor. If v551 ties/drops and v557 dry-run gate confirms the clipped displacement, v557 is safer than v554/v555/v556 for a later slot.


### v556 low-displacement gate-retune candidate launched — 2026-05-14 19:55 UTC

- **Status check:** latest Bearer API submissions remain `v549=0.946`, `v548=0.946`, `v547=0.946`, `v546=0.946`, `v545=0.944`; current best remains **0.946 public LB**. `v510` is COMPLETE/no failure with `submission.csv`. `v551` remains COMPLETE and guarded submit monitor pid `96890` is still alive/sleeping after daily cap. No duplicate submissions were created.
- **Track:** F/P2 public946 gate-retune preparation while capped, using the v555 post-gate diagnosis.
- **Hypothesis:** A lower interpolation alpha should capture a small but real local top-k/AUC lift from the v554/v555 retuned gates while keeping anchor displacement much lower. Observed-delta backoff from v555 estimated alpha `0.10` at macro AUC `0.992630` (`+0.000106`), top3 `0.637` vs baseline `0.621`, `corr=0.99961`, `MAE=0.00177`, `max_abs=0.12012`.
- **Implementation:** added `kaggle-kernels/v556-public946-gateretune-a010/` and `scripts/push_v556.py`. v556 forks v555 but sets `V556_FULL_RETUNE_ALPHA=0.10`, i.e. `0.90 * reconstructed v542 baseline + 0.10 * retuned final` after both full postprocess paths.
- **Validation:** `python3 -m py_compile scripts/push_v556.py scripts/monitor_v554_gate_retune.py kaggle-kernels/v556-public946-gateretune-a010/script.py` passed. Also updated `scripts/monitor_v554_gate_retune.py` to write `candidate_final` in future JSON artifacts instead of the stale `v554_final` key.
- **Kaggle push/monitor:** pushed real private Kaggle kernel `yourslewis/bc26-v556-public946-gate-retune-alpha010`, version 1; no invalid data/competition/kernel/model sources. Initial status RUNNING/no failure. Started no-submit monitor pid `38739`, log `logs/monitor_v556_gate_retune_20260514T195228Z.log`, with `KERNEL_SLUG=bc26-v556-public946-gate-retune-alpha010` and `OUTPUT_NAME=v556-public946-gate-retune-alpha010`.
- **Decision:** keep v551 as the only next-reset submit monitor. v556 is a prepared fallback candidate only if v551 ties/drops and v556 dry-run gates pass; no competition submission was attempted.


### v555 completed; hold as research-only, prefer lower-alpha if needed — 2026-05-14 18:55 UTC

- **Status check:** latest Bearer API submissions remain `v549=0.946`, `v548=0.946`, `v547=0.946`, `v546=0.946`, `v545=0.944`; current best remains **0.946 public LB**. `v510` is COMPLETE/no failure with `submission.csv`. `v551` remains COMPLETE and guarded submit monitor pid `96890` is still alive/sleeping after daily cap.
- **v555 result:** `yourslewis/bc26-v555-public946-gate-retune-alpha033` version 1 completed successfully. No-submit monitor downloaded `submission.csv`, `submission_protossm.csv`, and `submission_sed.csv`; all validate `(240,235)` with no NaNs. Gate artifact from the generic monitor: `artifacts/blend_grids/v554_gate_retune_final_vs_baseline_20260514T180412Z.json`.
- **v555 gate:** baseline macro AUC `0.992525`; v555 final macro AUC `0.992915` (`+0.000391`), top3 `0.637` vs `0.621`, top5 unchanged `0.747`. However real-run displacement is higher than the prior offline interpolation estimate: `corr=0.99577`, `MAE=0.00585`, `max_abs=0.39640`. This means v555 still moves a few cells too aggressively and should **not** displace v551 or be auto-submitted.
- **Post-gate diagnosis:** Scaling the observed v555 delta back to alpha-equivalents shows `alpha≈0.10` would keep most of the useful local lift with lower displacement: macro AUC `0.992630` (`+0.000106`), top3 `0.637`, `corr=0.99961`, `MAE=0.00177`, `max_abs=0.12012`. `alpha=0.05` is even safer (`corr=0.99990`, `max_abs=0.0601`) but only a tiny AUC lift.
- **Code hygiene:** fixed `scripts/monitor_v554_gate_retune.py` so future gate artifacts use the `OUTPUT_NAME` prefix instead of always `v554_*`.
- **Decision:** keep `v551` as the only next-reset submission. If `v551` ties/drops and we still want a gate-retune slot, prepare a lower-alpha candidate (likely `v556 alpha=0.10`) rather than submitting v554/v555 as-is. No new Kaggle submission monitor was started.


### v554 completed; v555 conservative gate-retune interpolation launched — 2026-05-14 17:55 UTC

- **Status check:** latest Bearer API submissions remain `v549=0.946`, `v548=0.946`, `v547=0.946`, `v546=0.946`, `v545=0.944`; current best remains **0.946 public LB**. `v510` is COMPLETE/no failure with `submission.csv`. `v551` remains COMPLETE and guarded submit monitor pid `96890` is still alive/sleeping after daily cap.
- **v554 result:** `yourslewis/bc26-v554-public946-gate-retune-pw056` version 1 completed successfully. No-submit monitor downloaded `submission.csv`, `submission_protossm.csv`, and `submission_sed.csv`; all validate as `(240,235)` with no NaNs. Gate artifact: `artifacts/blend_grids/v554_gate_retune_final_vs_baseline_20260514T170417Z.json`.
- **v554 gate:** v542 baseline reconstruction macro AUC `0.992525`; v554 final macro AUC `0.993325` (`+0.000800`), top3 `0.647` vs `0.621`, top5 `0.753` vs `0.747`. Risk: displacement is large (`corr=0.99418`, `MAE=0.01273`, `max_abs=0.41676`), so v554 should not be auto-submitted immediately after v551.
- **Risk-control sweep:** interpolating fully postprocessed v542 baseline with v554 final showed alpha `0.33` gives macro AUC `0.992915` (`+0.000391`) with much lower displacement (`corr=0.99937`, `MAE=0.00420`, `max_abs=0.1375`), preserving the top-k lift without the full retune jump.
- **Implementation:** added conservative no-submit candidate `kaggle-kernels/v555-public946-gateretune-a033/` plus `scripts/push_v555.py`. v555 reconstructs the standard v542 final blend, applies the v554 retune, then writes `submission.csv` as `0.67 * baseline + 0.33 * retune` after both branches' mirror/rare postprocess.
- **Kaggle push/monitor:** pushed real private Kaggle kernel `yourslewis/bc26-v555-public946-gate-retune-alpha033`, version 1; no invalid sources. Initial status RUNNING/no failure. Started no-submit monitor pid `17827`, log `logs/monitor_v555_gate_retune_20260514T175337Z.log`, using the generic gate-retune monitor with `KERNEL_SLUG=bc26-v555-public946-gate-retune-alpha033` and `OUTPUT_NAME=v555-public946-gate-retune-alpha033`.
- **Decision:** keep v551 as the only next-reset submission. v555 is the preferred *prepared* gate-retune candidate over v554 if v551 ties/drops and the v555 real dry-run gate matches the offline interpolation result.


### v554 public946 gate-retune dry-run candidate — 2026-05-14 16:55 UTC

- **Status check:** latest Bearer API submissions remain `v549=0.946`, `v548=0.946`, `v547=0.946`, `v546=0.946`, `v545=0.944`; current best remains **0.946 public LB**. `v510` is still COMPLETE/no failure with `submission.csv`. `v551` monitor pid `96890` remains alive and sleeping after daily cap; no duplicate v551 submission is visible.
- **Track:** F/P2 public946 AutoResearch gate retuning after multiple distinct sidecars tied but did not move displayed 0.946. This is explicitly a **no-submit dry-run candidate** while v551 is queued for the next reset.
- **Hypothesis:** The full public946 gate sweep found a stronger local configuration (`proto_weight=0.56`, lighter fake/proto/SED boosts, rare scale `0.85`) with overlap AUC `0.993325` vs baseline `0.992525`, but it needs a real Kaggle dry-run to verify runtime/output and avoid editing only local CSVs.
- **Implementation:** added `kaggle-kernels/v554-public946-gateretune-pw056/` from v542 with final blend constants `V554_PROTO_WEIGHT=0.56`, `V554_FAKE_BOOST=0.04`, `V554_CTX_THR=0.90`, `V554_CTX_BOOST=0.10`, `V554_SED_RANK_THR=0.93`, `V554_SED_BOOST=0.08`, `V554_RARE_SCALE=0.85`. Added `scripts/push_v554.py` and no-submit monitor `scripts/monitor_v554_gate_retune.py`.
- **Validation:** `python3 -m py_compile scripts/push_v554.py scripts/monitor_v554_gate_retune.py kaggle-kernels/v554-public946-gateretune-pw056/script.py` passed. Pushed real private Kaggle kernel via Bearer API; Kaggle created `yourslewis/bc26-v554-public946-gate-retune-pw056`, version 1, with no invalid data/competition/kernel/model sources.
- **Monitor:** started no-submit monitor pid `8418`, log `logs/monitor_v554_gate_retune_20260514T165341Z.log`. Initial status is RUNNING/no failure. The monitor will download `submission.csv`, `submission_protossm.csv`, and `submission_sed.csv`, then compare v554 final against reconstructed v542 baseline on train-soundscape overlap without submitting.
- **Decision:** keep v551 as the only live next-reset submission. v554 is preparation for the *following* slot only if its Kaggle dry-run completes and v551 fails to improve.


### v553 completed + Task239 Snowflake agreement-gate research pass — 2026-05-14 16:05 UTC

- **Status check:** latest Bearer API submissions show `v549=0.946`, `v548=0.946`, `v547=0.946`, `v546=0.946`, and `v545=0.944`; current best remains **0.946 public LB**. Kernels `v551`, `v552`, and `v553` are all COMPLETE/no failure with `submission.csv` present. UTC daily cap remains exhausted; guarded `v551` submit monitor pid `96890` is alive and sleeping after Kaggle returned `5/5` allowance used with about `8.1h` until reset. No duplicate v551 submission is visible.
- **v553 gate result:** downloaded/monitored v553 outputs via `logs/monitor_v553_taxon_gate_20260514T152721Z.log`; no-submit gate `artifacts/blend_grids/v553_convnext_taxon_sidecar_gate_20260514T153755Z.json` shows taxon-gated ConvNeXt sidecar is still negative vs public946 anchor on overlap: anchor `0.992525`; 0.25%-2% blends `0.992502`; 5% `0.992430`. Decision: **hold/no-submit v553**.
- **Research source:** audited cached public kernels from the 2026-05-14 scan, especially `ttyn4519__perch-task239-task233-snowflake-agree-w003`, which adds a Snowflake SED sidecar only where base V8, filemax-scaled SED, and Snowflake ranks all agree (`min_rank≈0.70`, `max_gap≈0.20`) instead of naive global Snowflake blending.
- **Offline grid:** ran an inline Task239-style agreement-gate sweep using v550 dry-run outputs (`submission_protossm.csv`, `submission_sed.csv`, `submission_snowflake_sed.csv`) and local train-soundscape labels. Artifact: `artifacts/blend_grids/v554_task239_snowflake_agree_grid_20260514T1605Z.json`.
- **Grid result:** reconstructed V8 anchor macro AUC `0.992525`. Best agreement-gated Snowflake variant was `p125=0.000`, `snow=0.030`, `min_rank=0.70`, `gap=0.20`, macro AUC `0.992567` (`+0.000043`), but displacement is large (`corr=0.9882`, `MAE=0.0439`) and top-k recall drops sharply vs the reconstructed anchor (`top5 0.632` vs `0.747`). This is not strong enough to supersede `v551` or justify a blind next-day slot.
- **Decision:** next reset candidate remains **v551 tiny CLAP 0.5%** because it has the best bounded local lift (`0.992549`) among completed candidates and a live guarded submit monitor. The next research idea to implement, if v551 ties/drops, is not another single sidecar sweep; it should be either (a) a source-clean **agreement-gated Snowflake** candidate with much tighter displacement gates and exact mapped/proxy mask, or (b) a **public946 gate-retune** candidate from the full sweep (`proto_weight≈0.56`, lighter fake/proto/sed boosts) compared directly against v551/v552/v553. Do not submit v552/v553 automatically.


## 2026-05-13 02:45 UTC — `public946-anchor-student-sidecar-gate`

- **Track:** P2/P3 public946 distinct-signal gate after anchor lock.
- **Hypothesis:** After `v541`/`v542` locked the 0.946 public946 anchor, the existing `rankblend->NFNet 5s power1.0 ep20` student should only consume a submission slot if it adds enough local diversity over the public946 teacher.
- **Branch/PR:** `feature/v541-v542-public946-anchor-lock`, PR #225.
- **Data/artifacts used:**
  - Teacher cache: `artifacts/pseudolabels/public946-v540-teacher-cache66-v1/teacher_rankblend.npz`.
  - Student predictions: `artifacts/pseudolabels/students/pl-public946-rankblend-nfnet-5s-lr1e4-ep20-bestval/student_predictions.npz`.
  - Anchor dry-run caches: `artifacts/pseudolabels/public946-v541-dryrun-cache-v1/` and `artifacts/pseudolabels/public946-v542-dryrun-cache-v1/`.
- **Command:** inline Python diagnostic using the Mac Kaggle venv; output JSON `artifacts/blend_grids/public946_anchor_student_gate_20260513.json`.
- **Result:** on 792 teacher rows / 75 valid classes, public946 teacher AUC `0.994567`, NFNet student AUC `0.984806`, student-teacher corr `0.924409`, MAE `0.07138`. Best linear blend was teacher `0.95` + student `0.05`, AUC `0.994637`, delta `+0.0000699` vs teacher.
- **Anchor comparison:** v541 rankblend vs v542 rankblend on 240 common rows has corr `0.999277`, MAE `0.00332`; v542 final vs v542 rankblend corr `0.997630`, MAE `0.00485`. The 0.946 anchors are essentially the same signal family.
- **Decision:** do **not** make the NFNet sidecar the default next Kaggle slot. It remains a 2-5% public946+student minority fallback with tiny local lift. Prefer source-clean BirdNET or resolved V5/CLAP for the next true diversity slot; if those source gates fail, package the 95/5 NFNet sidecar as the safer fallback.

## 2026-05-13 01:45 UTC — `public946-anchor-lock-v541-v542`

- **Track:** P0 public946 anchor reproduction and lock.
- **Hypothesis:** Restoring the missing public 0.946 postprocess details and independently replaying the updated Afr1ste public946 V8 stack should lift the repo-owned public946 baseline above `v539` (0.943) and lock a new canonical anchor.
- **Branch/PR:** `feature/v541-v542-public946-anchor-lock` (documentation/update PR for the score lock).
- **Kaggle candidates and refs:**
  - `v541` / `yourslewis/bc26-v541-public946-mirror-rare`, ref `52594869`.
  - `v542` / `yourslewis/bc26-v542-afr1ste-updated-public946`, ref `52594882`.
  - trailing diagnostic `v538` / `yourslewis/bc26-v538-v517-plus-oofteacher-b0-blend-005`, ref `52594896`.
- **Config/hyperparameters:**
  - `v541`: public946 Perch/ProtoSSM + distilled SED rank blend with sonotype mirroring and rare-taxon adaptive thresholding.
  - `v542`: Afr1ste updated public946 V8 replay, standard 60/40 rank blend, sonotype mirroring on 10 columns, rare thresholding on 44 species.
  - `v538`: old v517 taxon gate + fold-aware OOF-teacher B0 sidecar blend weight 0.05 diagnostic.
- **Validation before submission:** v541/v542 kernels COMPLETE/no failure; v542 output verification showed SED folds loaded, full dry-run `submission.csv` shape `(240,235)`, no NaNs, and runtime about 528s.
- **LB result:**
  - `v541`: **0.946 public LB**.
  - `v542`: **0.946 public LB**.
  - `v538`: `0.930 public LB`.
- **Interpretation:** `v541` is now the canonical repo-owned public946 anchor; `v542` independently confirms the 0.946 public frontier. `v539` (0.943) is superseded, and old internal 0.930-axis sidecars should remain diagnostics only.
- **Next step:** Do not queue a plain public weight clone next. Prefer distinct-signal work gated by local diagnostics: public946 teacher cache/student sidecar, source-clean BirdNET-only 3-way rank blend, or V5/CLAP only after source refs are resolved.

## 2026-05-06 06:50 UTC — `sed-b0-5s-attn-v1-smoke`

- **Track:** A+G Real SED frame/event smoke + export/inference packaging
- **Hypothesis:** Before scaling to EfficientNet-B0/V2/NFNet on GPU, validate that the repo has a real-audio weak-label SED pipeline that can decode BirdCLEF OGG files, build mel features, train frame/event logits, and export an artifact.
- **Branch:** `feature/sed-smoke-export-scaffold`
- **Config:** `configs/birdclef/sed_b0_5s_attn_smoke.json`
- **Script:** `scripts/birdclef_sed_smoke.py`
- **Data used:** local BirdCLEF data at `/Volumes/ExternalSSD/data/workspace_don/kaggle_birdclef2026/data`, smoke-limited to 5 real `train_audio/*/*.ogg` files.
- **Hyperparameters:** tiny CNN SED smoke backbone, 5s crop, 128 mel bins, hop 512, BCEWithLogits, AdamW lr 3e-4, batch size 2, 1 epoch, no mixup/label smoothing/class balancing.
- **Export:** TorchScript required; ONNX attempted only if local `onnx` package is available.
- **GPU status:** `yourslewis@192.168.0.23` (stale address) SSH timed out in this run, so no long GPU job was launched. Next run should retry GPU and scale this scaffold there.
- **Next step:** If smoke passes, add EfficientNet-B0/timm backbone on GPU or Kaggle image, then run 1-fold 2-epoch pilot and export ONNX/OpenVINO.

### Smoke result

- **Command:** `~/.openclaw/workspace-don/kaggle/playground-series-s6e3/.venv/bin/python scripts/birdclef_sed_smoke.py --config configs/birdclef/sed_b0_5s_attn_smoke.json --max-files 5 --epochs 1`
- **Status:** passed on 5 real OGG clips.
- **Input shape:** `[5, 128, 313]`
- **Frame logits shape:** `[5, 313, 234]`
- **Loss:** `0.79476 -> 0.74537`
- **Runtime:** `0.21s` for the tiny smoke model after decode/feature setup.
- **Artifact:** `artifacts/sed_smoke/sed-b0-5s-attn-v1-smoke/tiny_sed_smoke_torchscript.pt`
- **Metrics:** `artifacts/sed_smoke/sed-b0-5s-attn-v1-smoke/metrics.json`
- **ONNX:** blocked locally by missing `onnx` package (`ModuleNotFoundError`). This is now a clear dependency/setup task for the next A+G run.
- **Interpretation:** This is not a meaningful classifier yet; it proves the real-audio SED scaffold, weak-label frame output shape, training loop, and TorchScript export path work.

## 2026-05-06 07:50 UTC — `sed-smoke-sweep-v2`

- **Track:** A+G Real SED frame/event smoke + export/inference packaging.
- **Hypothesis:** After the first 5-file SED smoke passed, add a small AutoResearch-style knob sweep so later GPU pilots can choose between BCE, focal/class-balanced BCE, light label smoothing/mixup, and larger 10s/160-mel inputs instead of jumping blindly to EfficientNet.
- **Branch/PR:** `feature/sed-smoke-export-scaffold`, PR #204.
- **Scripts:**
  - `scripts/birdclef_sed_smoke.py` now supports `loss_name`, `focal_gamma`, `label_smoothing`, `mixup_alpha`, `class_balancing`, `val_fraction`, `duration_sec`, and `n_mels` overrides.
  - `scripts/birdclef_sed_smoke_sweep.py` runs a small CPU sweep and writes per-variant metrics under ignored `artifacts/sed_smoke/sweep-v2/`.
- **Data used:** 24 real BirdCLEF train OGG clips from `/Volumes/ExternalSSD/data/workspace_don/kaggle_birdclef2026/data`, with an 80/20 smoke split.
- **Command launched:** `~/.openclaw/workspace-don/kaggle/playground-series-s6e3/.venv/bin/python scripts/birdclef_sed_smoke_sweep.py --output-root artifacts/sed_smoke/sweep-v2`
- **Results:** all four variants passed smoke training/export checks:
  1. `sed-smoke-sweep-v2-5s-focal15-possqrt`: input `24x128x313`, focal BCE gamma 1.5 + sqrt positive weight, train loss `0.26659`, val loss `0.26442`.
  2. `sed-smoke-sweep-v2-10s-bce-m160`: input `24x160x626`, BCE, train loss `0.70210`, val loss `0.69681`.
  3. `sed-smoke-sweep-v2-5s-bce-m128`: input `24x128x313`, BCE, train loss `0.70040`, val loss `0.69746`.
  4. `sed-smoke-sweep-v2-5s-bce-smooth001-mixup02`: input `24x128x313`, BCE + label smoothing 0.01 + mixup 0.2, train loss `0.70066`, val loss `0.69772`.
- **Export:** TorchScript artifacts produced for every variant. ONNX remains blocked locally by missing `onnx` (`ModuleNotFoundError`) and should be resolved before the real export pilot.
- **Interpretation:** This is still operational smoke, not model selection. Focal + sqrt positive weighting clearly changes the loss scale and is the best next first GPU pilot candidate because it handles BirdCLEF's sparse multi-label imbalance more explicitly. The 10s/160-mel variant also validated memory/shape for a context pilot.
- **GPU status:** Retried stale address `ssh -o BatchMode=yes -o ConnectTimeout=8 yourslewis@192.168.0.23`; still timed out with exit 255. No remote durable GPU job launched.
- **Queue monitor:** Previous `mild-kelp` session was gone, so a new durable `nohup` monitor was started: `logs/submit_pending_birdclef_queue_20260506T075308Z.log`, pid `46665`. It verified v505 is COMPLETE, attempted submission, hit the daily cap with ~16h remaining, and is sleeping until retry.
- **Next step:** When GPU SSH is reachable, launch `sed-smoke-sweep-v2-5s-focal15-possqrt` as the first EfficientNet-B0/timm 1-fold 2-epoch pilot, plus a 10s/160-mel sibling if resources permit. Also install/enable ONNX export in the training/export environment.


## 2026-05-06 09:35 UTC — `sed-b0-gpu-pilot-v1-5s-focal15-possqrt`

- **Track:** A+G Real SED frame/event GPU pilot + export/inference packaging.
- **Hypothesis:** The best operational smoke variant (focal BCE gamma 1.5 + sqrt positive class weighting) should scale to a real EfficientNet-B0/timm weak-label SED pilot on the corrected GPU host and produce holdout prediction artifacts for later OOF/blend work.
- **Branch/PR:** `feature/sed-smoke-export-scaffold`, PR #204.
- **Infrastructure correction:** GPU host is `yourslewis@192.168.0.10` (`trainer`), not stale `192.168.0.23`. Verified 2x RTX 4090 and CUDA in `~/kaggle_envs/s6e3`.
- **Config:** `configs/birdclef/sed_b0_gpu_pilot_v1_focal_possqrt.json`.
- **Script:** `scripts/birdclef_sed_pilot_train.py`.
- **Hyperparameters:** EfficientNet-B0 via timm, 5s crops, 128 mels, hop 512, sample rate 32k, focal BCE gamma 1.5, sqrt positive class weighting, label smoothing 0, mixup 0, AdamW lr 3e-4 wd 1e-4, batch size 16, max files 512, 80/20 holdout, 2 epochs.
- **Setup:** Installed/enabled `timm`, `onnx`, `onnxscript`, and `imageio-ffmpeg` in the GPU venv as needed. `imageio-ffmpeg` is used as a portable fallback because system `ffmpeg` is not installed on the server.
- **Smoke/preflight:** Local tiny-CNN preflight passed on 6-8 real clips. Remote CUDA EfficientNet-B0 preflight passed on 8 real clips: device `cuda`, input `[8,128,313]`, TorchScript export size `15.388 MB`; ONNX initially failed on missing `onnxscript`, then `onnxscript` was installed before the full pilot launch.
- **Command launched:** on `192.168.0.10`, from `~/birdclef-2026`:
  `nohup env CUDA_VISIBLE_DEVICES=0 python scripts/birdclef_sed_pilot_train.py --config configs/birdclef/sed_b0_gpu_pilot_v1_focal_possqrt.json > logs/sed_b0_gpu_pilot_v1_20260506T094113Z.log 2>&1 &`
- **Remote PID/log:** pid `2524246`, log `~/birdclef-2026/logs/sed_b0_gpu_pilot_v1_20260506T094113Z.log`.
- **Expected artifacts:** `~/birdclef-2026/artifacts/sed_pilots/sed-b0-gpu-pilot-v1-5s-focal15-possqrt/metrics.json`, `holdout_predictions.npz`, `model_torchscript.pt`, optional `model.onnx`, config snapshot, and training log.
- **Final status:** complete.
- **Result:** 512 real clips, train/val `410/102`, input `[512,128,313]`, 2 epochs. Train loss `0.31540 -> 0.27830`; val loss `0.29509 -> 0.27717`; holdout macro AUC `0.51354` across 76 valid classes.
- **Artifacts:** `~/birdclef-2026/artifacts/sed_pilots/sed-b0-gpu-pilot-v1-5s-focal15-possqrt/metrics.json`, `holdout_predictions.npz`, `model_torchscript.pt` (`15.388 MB`), and `model.onnx` + external data (`0.56 MB` + `14.647 MB`). ONNX exported after installing `onnxscript`; PyTorch emitted opset conversion warnings but produced an ONNX artifact.
- **Interpretation:** First real EfficientNet-B0 SED prediction artifact exists. AUC is only smoke-holdout quality, but it is a less-correlated frame/SED model family and satisfies the artifact path needed for later blend/OOF work.


## 2026-05-06 09:42 UTC — `sed-b0-gpu-pilot-v2-10s-m160-focal15-possqrt`

- **Track:** A+G Real SED frame/event GPU pilot + crop/mel resolution sibling.
- **Hypothesis:** The 10s/160-mel context variant should improve weak-label SED discrimination over the 5s/128-mel pilot by adding more temporal context, while keeping the best smoke loss/class-balance settings fixed.
- **Branch/PR:** `feature/sed-smoke-export-scaffold`, PR #204.
- **Config:** `configs/birdclef/sed_b0_gpu_pilot_v2_10s_m160_focal_possqrt.json`.
- **Command launched:** on `192.168.0.10`, from `~/birdclef-2026`:
  `nohup env CUDA_VISIBLE_DEVICES=1 bash -lc "source ~/kaggle_envs/s6e3/bin/activate; python scripts/birdclef_sed_pilot_train.py --config configs/birdclef/sed_b0_gpu_pilot_v2_10s_m160_focal_possqrt.json" > logs/sed_b0_gpu_pilot_v2_20260506T094221Z.log 2>&1 &`
- **Hyperparameters:** EfficientNet-B0 via timm, 10s crops, 160 mels, hop 512, focal BCE gamma 1.5, sqrt positive class weighting, AdamW lr 3e-4 wd 1e-4, batch size 8, max files 512, seed 43, 80/20 holdout, 2 epochs.
- **Result:** complete. 512 real clips, train/val `410/102`, input `[512,160,626]`. Train loss `0.30597 -> 0.26143`; val loss `0.30555 -> 0.33323`; holdout macro AUC `0.57967` across 78 valid classes.
- **Artifacts:** `~/birdclef-2026/artifacts/sed_pilots/sed-b0-gpu-pilot-v2-10s-m160-focal15-possqrt/metrics.json`, `holdout_predictions.npz`, `model_torchscript.pt` (`15.388 MB`), and `model.onnx` + external data (`0.56 MB` + `14.647 MB`).
- **Interpretation:** 10s/160-mel sibling has better tiny holdout macro AUC than the 5s/128 pilot (`0.57967` vs `0.51354`) but worse final val loss, suggesting useful context signal with possible overfit/calibration drift. Next A+G move should add proper fold split/OOF and compare blend correlation with the v504/v508 prediction family before expanding epochs.

## 2026-05-06 09:46 UTC — `sed-b0-gpu-pilot-v3-10s-m160-seed42-focal15-possqrt` + matched split blend check

- **Track:** A+G Real SED frame/event GPU pilot + same-split comparison.
- **Hypothesis:** The prior 10s/160-mel v2 pilot used seed 43, so it could not be directly correlated/blended with the 5s/128-mel v1 holdout. Rerun the 10s/160-mel variant with seed 42 to match v1's file/holdout split and test whether crop/mel diversity produces complementary predictions.
- **Branch/PR:** `feature/sed-smoke-export-scaffold`, PR #204.
- **Config:** `configs/birdclef/sed_b0_gpu_pilot_v3_10s_m160_seed42_focal_possqrt.json`.
- **Comparison script:** `scripts/birdclef_compare_sed_pilots.py` aligns holdout files and computes per-model AUC, flat prediction correlation, and a simple blend grid.
- **Command launched:** on `192.168.0.10`, from `~/birdclef-2026`:
  `nohup env CUDA_VISIBLE_DEVICES=1 bash -lc "source ~/kaggle_envs/s6e3/bin/activate; python scripts/birdclef_sed_pilot_train.py --config configs/birdclef/sed_b0_gpu_pilot_v3_10s_m160_seed42_focal_possqrt.json" > logs/sed_b0_gpu_pilot_v3_20260506T094611Z.log 2>&1 &`
- **Result:** complete. 512 real clips, train/val `410/102`, input `[512,160,626]`. Train loss `0.31030 -> 0.26279`; val loss `0.31319 -> 0.27319`; holdout macro AUC `0.51991` across 76 valid classes. TorchScript and ONNX exported.
- **Matched split comparison:** v1 5s/128 seed42 vs v3 10s/160 seed42 aligned on 102 holdout files and 234 classes.
  - v1 macro AUC: `0.513541`
  - v3 macro AUC: `0.519907`
  - flat Pearson correlation: `0.164600`
  - mean absolute prediction difference: `0.030940`
  - best simple blend in grid: 50% v3 / 50% v1, macro AUC `0.573316`
  - blend grid AUCs by v3 weight: 0.0=`0.513541`, 0.1=`0.547246`, 0.2=`0.563622`, 0.3=`0.570250`, 0.4=`0.572934`, 0.5=`0.573316`, 0.6=`0.560046`, 0.7=`0.550302`, 0.8=`0.539320`, 0.9=`0.529698`, 1.0=`0.519907`.
- **Interpretation:** This is the first strong evidence that the SED crop/mel variants are complementary: individual tiny-holdout AUCs are modest, but same-split blend improves by about +0.060 over v1 and correlation is low. Next step is to convert this from tiny holdout into proper fold/OOF artifacts, then compare/blend against the v504/v508 teacher family if raw prediction artifacts can be located or regenerated.

## 2026-05-06 10:35 UTC — SED OOF runner + balanced-class OOF check

- **Track:** A+G Real SED frame/event OOF artifacts.
- **Hypothesis:** The prior 512-file OOF attempt was structurally weak because the default selector included many classes with too few examples, so folds often validated species with no training positives. Add explicit fold support plus a balanced-class selector to produce a more meaningful small OOF benchmark before scaling.
- **Branch/PR:** `feature/sed-smoke-export-scaffold`, PR #204.
- **Code/configs added:**
  - `scripts/birdclef_sed_oof_runner.py` — runs `birdclef_sed_pilot_train.py` across folds and aggregates `oof_predictions.npz`.
  - `scripts/birdclef_compare_oof_predictions.py` — aligns OOF files and computes AUC/correlation/blend grid.
  - `birdclef_sed_pilot_train.py` now supports `n_folds`, `fold_index`, and `selection_strategy=balanced_classes` with `max_classes`, `files_per_class`, `min_files_per_class`.
  - `configs/birdclef/sed_b0_balanced_oof_v1_5s_128.json`.
  - `configs/birdclef/sed_b0_balanced_oof_v3_10s_160.json`.
- **Smoke validation:** local tiny-CNN 2-fold OOF smoke passed on 6 real files.
- **Teacher artifact search:** no raw v504/v508 OOF/test prediction artifact was found locally; only kernel push/poll scripts and Perch cache files were visible. Comparison to v504/v508 will require regenerating/locating raw teacher predictions.

### Unbalanced 512-file OOF baseline

- Ran 3-fold OOF for the previous default 512-file selector.
- `sed-b0-oof-v1-5s-128-focal15-possqrt`: OOF macro AUC `0.499948` over 206 valid classes.
- `sed-b0-oof-v3-10s-160-focal15-possqrt`: OOF macro AUC `0.455467` over 206 valid classes.
- OOF blend comparison: flat Pearson `0.152588`; best blend was all v1 (`weight_b=0.0`, AUC `0.499948`).
- **Interpretation:** this OOF is not trustworthy for model selection because class coverage per fold is poor.

### Balanced-class 300-file OOF benchmark

- Selection: `balanced_classes`, 30 classes, 10 files/class, 300 files total, 3 folds; this ensures each fold has positives in train/validation for the chosen classes.
- `sed-b0-balanced-oof-v1-5s-128-focal15-possqrt`: OOF macro AUC `0.476316` over 30 valid classes. Fold AUCs were roughly low/mid `0.4` to `0.557793`.
- `sed-b0-balanced-oof-v3-10s-160-focal15-possqrt`: OOF macro AUC `0.534575` over 30 valid classes. Fold AUCs: `0.544453`, `0.594332`, `0.605642`.
- Balanced OOF comparison: aligned 300 files; flat Pearson `0.092054`; mean absolute prediction diff `0.024793`.
- Blend grid by v3 weight: 0.0=`0.476316`, 0.1=`0.486609`, 0.2=`0.498391`, 0.3=`0.508402`, 0.4=`0.514299`, 0.5=`0.521264`, 0.6=`0.526822`, 0.7=`0.531931`, 0.8=`0.532828`, 0.9=`0.533563`, 1.0=`0.534575`.
- **Interpretation:** On a more meaningful balanced-class OOF subset, 10s/160-mel is better than 5s/128, but the simple blend does not beat 10s alone. Low correlation still suggests diversity, but the 5s model is too weak at this setting. Next A+G move should improve the 10s model (more classes/files, more epochs, label smoothing/mixup or LR sweep), not push a Kaggle kernel yet.

## 2026-05-06 11:35 UTC — SED 10s/160 larger balanced OOF + smoothing/mixup A/B

- **Track:** A+G Real SED frame/event OOF hyperparameter tuning.
- **Hypothesis:** Since the balanced 30-class OOF showed 10s/160-mel is stronger than 5s/128, scale the 10s/160 setting to more balanced classes/files and test one regularization bundle (label smoothing 0.01 + mixup 0.2) while keeping backbone, crop, mel bins, loss, and class balancing fixed.
- **Branch/PR:** `feature/sed-smoke-export-scaffold`, PR #204.
- **Configs:**
  - `configs/birdclef/sed_b0_balanced_oof_v4_10s_160_moredata.json`
  - `configs/birdclef/sed_b0_balanced_oof_v5_10s_160_smooth_mixup.json`
- **Common setup:** EfficientNet-B0, 10s crops, 160 mels, focal BCE gamma 1.5, sqrt positive class weighting, 50 classes × 10 files/class = 500 files, 3 folds, 5 epochs, ONNX/TorchScript export.
- **v4 command:** on `192.168.0.10`, `CUDA_VISIBLE_DEVICES=0`, `scripts/birdclef_sed_oof_runner.py --base-config configs/birdclef/sed_b0_balanced_oof_v4_10s_160_moredata.json --output-root artifacts/sed_oof/sed-b0-balanced-oof-v4-10s-160-moredata --n-folds 3`.
- **v5 command:** on `192.168.0.10`, `CUDA_VISIBLE_DEVICES=1`, `scripts/birdclef_sed_oof_runner.py --base-config configs/birdclef/sed_b0_balanced_oof_v5_10s_160_smooth_mixup.json --output-root artifacts/sed_oof/sed-b0-balanced-oof-v5-10s-160-smooth-mixup --n-folds 3`.
- **v4 result (more data, no regularization):** OOF macro AUC `0.506684` over 50 valid classes. Fold AUCs: `0.635036`, `0.562333`, `0.602611`; final fold val losses around `0.1631`-`0.1705`. Artifacts under `~/birdclef-2026/artifacts/sed_oof/sed-b0-balanced-oof-v4-10s-160-moredata/`.
- **v5 result (label smoothing 0.01 + mixup 0.2):** OOF macro AUC `0.533127` over 50 valid classes. Fold AUCs approximately `0.607204`, `0.579449`, plus a high-0.5/low-0.6 first fold; final fold val losses around `0.1790`-`0.1983`. Artifacts under `~/birdclef-2026/artifacts/sed_oof/sed-b0-balanced-oof-v5-10s-160-smooth-mixup/`.
- **v4/v5 comparison:** aligned 500 files; v4 AUC `0.506684`, v5 AUC `0.533127`, flat Pearson `0.666455`, mean absolute diff `0.013875`. Blend grid barely beats v5 at v5 weight 0.9: AUC `0.533167`; v5 alone is effectively best.
- **Interpretation:** Scaling from 30 to 50 classes made the benchmark harder, but smoothing+mixup recovered most of the previous 30-class 10s/160 performance and clearly beat the unregularized larger run. Regularization helps, but the high v4/v5 correlation means this is not a new diversity axis. Next A+G action should tune 10s/160 regularized model learning rate/gamma or increase epochs carefully; alternatively move to a stronger backbone (EfficientNetV2-S/NFNet) on the same balanced OOF harness.

## 2026-05-06 12:35 UTC — SED stronger-backbone balanced OOF check

- **Track:** A+G Real SED frame/event model-zoo/backbone sweep on the balanced OOF harness.
- **Hypothesis:** Since EfficientNet-B0 10s/160 with smoothing+mixup is the current best SED configuration, test whether stronger Spec-A backbones add quality/diversity on the same 50-class balanced OOF benchmark. Smoke first, then scale only if safe.
- **Branch/PR:** `feature/sed-smoke-export-scaffold`, PR #204.
- **Configs:**
  - `configs/birdclef/sed_v2s_balanced_oof_v6_10s_160_smooth_mixup.json`
  - `configs/birdclef/sed_nfnet_balanced_oof_v7_10s_160_smooth_mixup.json`
- **Common setup:** 10s crops, 160 mels, 50 classes × 10 files/class = 500 files, 3-fold OOF, 5 epochs, focal BCE gamma 1.5, sqrt positive class weighting, label smoothing 0.01, mixup 0.2.
- **Preflight:** both backbones passed tiny CUDA preflight on `192.168.0.10`: `tf_efficientnetv2_s` on 12 files and `eca_nfnet_l0` on 8 files. ONNX export for V2-S was too slow/hung during first preflight, so full OOF runs used TorchScript export only (`export_onnx=false`) for these larger backbones.
- **v6 EfficientNetV2-S result:** OOF macro AUC `0.538471` over 50 valid classes. Fold AUCs: `0.605863`, `0.589326`, `0.594774`. TorchScript size about `81.443 MB` per fold.
- **v7 eca_nfnet_l0 result:** OOF macro AUC `0.565955` over 50 valid classes. Fold AUCs: `0.615302`, `0.634777`, `0.652672`. TorchScript size about `89.870 MB` per fold.
- **Backbone comparison against v5 B0 regularized baseline:**
  - v5 B0 AUC `0.533127`; v6 V2-S AUC `0.538471`; correlation `0.273131`; best simple blend at 50% V2-S = `0.547722`.
  - v5 B0 AUC `0.533127`; v7 NFNet AUC `0.565955`; correlation `0.352376`; best simple blend at 50% NFNet = `0.578510`.
  - v6 V2-S AUC `0.538471`; v7 NFNet AUC `0.565955`; correlation `0.588825`; best simple blend at 70% NFNet = `0.572567`.
- **Interpretation:** NFNet is the best SED backbone so far on the balanced OOF harness and also blends well with B0, giving the best observed SED OOF blend (`0.578510`) on this benchmark. This is a real model-family improvement, not a postprocess micro-sweep. Next step should either (a) launch a larger NFNet/B0 OOF with more classes/files or more epochs, or (b) start packaging an inference path for the NFNet+B0 SED ensemble once teacher/raw prediction artifacts are available for blend calibration.

## 2026-05-06 13:35 UTC — SED B0/NFNet 100-class balanced OOF scale-up

- **Track:** A+G Real SED frame/event scaled balanced OOF.
- **Hypothesis:** The 50-class balanced OOF showed NFNet is the best SED backbone so far and NFNet+B0 gives the best SED blend. Scale the exact B0/NFNet 10s/160 regularized pair from 50 to 100 balanced classes to see whether the signal survives broader class coverage.
- **Branch/PR:** `feature/sed-smoke-export-scaffold`, PR #204.
- **Configs:**
  - `configs/birdclef/sed_b0_balanced_oof_v8_10s_160_100cls.json`
  - `configs/birdclef/sed_nfnet_balanced_oof_v9_10s_160_100cls.json`
- **Common setup:** 10s crops, 160 mels, 100 classes × 10 files/class = 1000 files, 3-fold OOF, 5 epochs, focal BCE gamma 1.5, sqrt positive class weighting, label smoothing 0.01, mixup 0.2, TorchScript export only (`export_onnx=false`).
- **v8 B0 command:** launched on `192.168.0.10`, `CUDA_VISIBLE_DEVICES=0`, output root `artifacts/sed_oof/sed-b0-balanced-oof-v8-10s-160-100cls`, log `logs/sed_oof_v8_b0_100cls_20260506T133739Z.log`.
- **v9 NFNet command:** launched on `192.168.0.10`, `CUDA_VISIBLE_DEVICES=1`, output root `artifacts/sed_oof/sed-nfnet-balanced-oof-v9-10s-160-100cls`, log `logs/sed_oof_v9_nfnet_100cls_20260506T133739Z.log`.
- **v8 B0 status/result:** complete. OOF macro AUC `0.485820` over 100 valid classes, 1000 OOF files. Fold 0 AUC `0.558764`; fold 2 AUC `0.555638`; overall AUC dropped materially vs 50-class v5 (`0.533127`), so B0 does not scale cleanly to broader class coverage in this setup.
- **v9 NFNet status at log time:** still running fold 2. Fold 0 AUC `0.618094` over 100 classes, fold 1 AUC `0.633719` over 98 classes; fold 2 child process active (`birdclef_sed_pilot_train.py --config ...config_fold2.json`) on GPU. Next run should collect `artifacts/sed_oof/sed-nfnet-balanced-oof-v9-10s-160-100cls/oof_summary.json`, compare v8/v9 if complete, and decide whether to scale NFNet further or tune it.
- **Interpretation so far:** B0 weakens badly at 100 classes, while NFNet fold 0/1 remain strong (>0.61 fold AUC). This supports continuing NFNet as the primary SED backbone and deprioritizing B0 except as a diversity/blend component if its correlation remains useful.

### v9 NFNet 100-class completion + v8/v9 comparison

- **v9 NFNet final result:** complete. OOF macro AUC `0.587033` over 100 valid classes, 1000 OOF files. Fold AUCs: `0.618094`, `0.633719`, `0.648795`. This is a strong scale-up from 50-class v7 (`0.565955`) despite doubling class count.
- **v8/v9 comparison:** aligned 1000 files. B0 v8 AUC `0.485820`; NFNet v9 AUC `0.587033`; flat Pearson `0.620113`; mean absolute diff `0.126029`.
- **Blend grid:** B0->NFNet weight 0.0=`0.485820`, 0.1=`0.524593`, 0.2=`0.539884`, 0.3=`0.552519`, 0.4=`0.562065`, 0.5=`0.570213`, 0.6=`0.577273`, 0.7=`0.582413`, 0.8=`0.586091`, 0.9=`0.587667`, 1.0=`0.587033`.
- **Interpretation update:** NFNet clearly scales; B0 mainly contributes a tiny complementary bump at ~10% weight. Best SED-only balanced OOF result so far is B0 10% + NFNet 90% = `0.587667`. Next actionable step should be NFNet-focused: either tune NFNet lr/gamma/epochs on 100-class OOF, or extend NFNet to more classes/files before building inference packaging.

## 2026-05-06 14:35 UTC — NFNet 100-class LR/gamma sweep launch

- **Track:** A+G Real SED frame/event NFNet-focused hyperparameter tuning.
- **Hypothesis:** NFNet is the strongest SED backbone so far on the 100-class balanced OOF harness. Test two single-knob variants against v9 baseline (`lr=3e-4`, focal gamma `1.5`, AUC `0.587033`): lower focal gamma to `1.0`, and lower learning rate to `1e-4`.
- **Branch/PR:** `feature/sed-smoke-export-scaffold`, PR #204.
- **Baseline:** `sed-nfnet-balanced-oof-v9-10s-160-100cls-smooth001-mixup02`, OOF AUC `0.587033`; best B0/NFNet blend `0.587667`.
- **Configs:**
  - `configs/birdclef/sed_nfnet_balanced_oof_v10_10s_160_100cls_gamma10.json` — same as v9, but focal gamma `1.0`.
  - `configs/birdclef/sed_nfnet_balanced_oof_v11_10s_160_100cls_lr1e4.json` — same as v9, but learning rate `1e-4`.
- **Common setup:** eca_nfnet_l0, 10s crops, 160 mels, 100 classes × 10 files/class = 1000 files, 3-fold OOF, 5 epochs, sqrt positive class weighting, label smoothing 0.01, mixup 0.2, TorchScript export only.
- **Commands launched:**
  - v10 on `192.168.0.10`, `CUDA_VISIBLE_DEVICES=0`, log `logs/sed_oof_v10_nfnet_gamma10_20260506T143709Z.log`, pid `2979001`.
  - v11 on `192.168.0.10`, `CUDA_VISIBLE_DEVICES=1`, log `logs/sed_oof_v11_nfnet_lr1e4_20260506T143709Z.log`, pid `2979003`.
- **Status at report time:** both OOF runners are still active. v10 fold 0 completed with AUC `0.599032` over 100 classes (below v9 fold 0 `0.618094`), then started fold 1. v11 had started fold 0 and was still running. Next run should collect both `oof_summary.json` files, compare v10/v11/v9, and decide whether focal gamma/lr tuning improves NFNet or whether to scale v9 directly.

## 2026-05-06 15:40 UTC — NFNet LR=1e-4 win + scale probes launched

- **Track:** A+G Real SED frame/event NFNet-focused scaling/tuning.
- **Hypothesis:** The prior NFNet 100-class sweep showed optimizer step size is a major knob. Lower LR may stabilize NFNet on weak-label SED training; if it holds under wider class coverage or longer training, this becomes the primary SED candidate for inference packaging.
- **Branch/PR:** `feature/sed-smoke-export-scaffold`, PR #204.
- **Completed sweep results:**
  - `v9` baseline NFNet, LR `3e-4`, focal gamma `1.5`: OOF macro AUC `0.587033` over 100 valid classes / 1000 files.
  - `v10` NFNet, LR `3e-4`, focal gamma `1.0`: OOF macro AUC `0.524482`; fold 2 collapsed (`val_loss=0.716058`), so lower gamma is rejected.
  - `v11` NFNet, LR `1e-4`, focal gamma `1.5`: OOF macro AUC `0.622721` over 100 valid classes / 1000 files. Fold AUCs `0.686372`, `0.654855`, `0.681563`.
- **OOF comparison artifacts:** on GPU server under `artifacts/sed_oof/comparisons/`.
  - `v9_vs_v11.json`: Pearson `0.873790`, mean abs diff `0.043709`; best blend uses v11 weight `0.7` for OOF AUC `0.628163`, better than v11 alone.
  - `v8_vs_v11.json`: Pearson `0.589312`, mean abs diff `0.141561`; best blend is v11 alone (`0.622721`), so B0 no longer adds useful signal after the LR fix.
  - `v10_vs_v11.json`: Pearson `0.659454`; best blend is v11 alone.
- **Interpretation:** LR `1e-4` is the clearest SED OOF improvement so far. v11 is both a stronger standalone SED model and a useful complement to the older LR `3e-4` NFNet, but B0 can be deprioritized.
- **New configs launched:**
  - `configs/birdclef/sed_nfnet_balanced_oof_v12_10s_160_150cls_lr1e4.json`: scale winning LR to 150 balanced classes × 10 files/class = 1500 files, 3-fold, 5 epochs.
  - `configs/birdclef/sed_nfnet_balanced_oof_v13_10s_160_100cls_lr1e4_ep8.json`: same 100-class benchmark as v11 but train 8 epochs to test whether longer low-LR training improves or overfits.
- **Commands launched on `192.168.0.10`:**
  - v12: `CUDA_VISIBLE_DEVICES=0`, pid `3114781`, log `logs/sed_oof_v12_nfnet_150cls_lr1e4_20260506T154031Z.log`, output `artifacts/sed_oof/sed-nfnet-balanced-oof-v12-10s-160-150cls-lr1e4/`.
  - v13: `CUDA_VISIBLE_DEVICES=1`, pid `3114783`, log `logs/sed_oof_v13_nfnet_100cls_lr1e4_ep8_20260506T154031Z.log`, output `artifacts/sed_oof/sed-nfnet-balanced-oof-v13-10s-160-100cls-lr1e4-ep8/`.
- **Status at log time:** both new OOF runners started fold 0 cleanly. Next run should collect v12/v13 summaries, compare against v11, and if v11/v13 remain best, start inference/kernel packaging for NFNet TorchScript folds and/or build a v9+v11 SED blend candidate.

## 2026-05-06 16:37 UTC — NFNet 8-epoch SED win + broader ep8 scale launch

- **Track:** A+G Real SED frame/event NFNet tuning/scaling.
- **Hypothesis:** The low-LR NFNet recipe improves with longer training on the same 100-class OOF harness. Test whether that 8-epoch recipe remains stable when class coverage expands beyond 100 classes.
- **Branch/PR:** `feature/sed-smoke-export-scaffold`, PR #204.
- **Completed results collected:**
  - `v11` NFNet 100-class, LR `1e-4`, 5 epochs: OOF macro AUC `0.622721` over 100 valid classes / 1000 files.
  - `v12` NFNet 150-class, LR `1e-4`, 5 epochs: OOF macro AUC `0.620044` over 150 valid classes / 1500 files. Fold AUCs `0.680537`, `0.663893`, `0.668911`.
  - `v13` NFNet 100-class, LR `1e-4`, 8 epochs: OOF macro AUC `0.636878` over 100 valid classes / 1000 files. Fold AUCs `0.695652`, `0.661716`, `0.681349`.
- **OOF comparison artifacts:** on GPU server under `artifacts/sed_oof/comparisons/`.
  - `v11_vs_v13.json`: Pearson `0.783948`, mean abs diff `0.042331`; best blend uses v13 weight `0.6` for OOF AUC `0.644676`, better than either model alone.
  - `v9_vs_v13.json`: Pearson `0.724891`, mean abs diff `0.069515`; best blend uses v13 weight `0.8` for OOF AUC `0.638740`.
- **Interpretation:** 8 epochs at LR `1e-4` is the strongest same-benchmark SED model so far, and blending 5-epoch + 8-epoch low-LR NFNet snapshots gives a large OOF gain. The 150-class 5-epoch run stayed stable and near the 100-class v11 score despite broader class coverage.
- **New configs launched:**
  - `configs/birdclef/sed_nfnet_balanced_oof_v14_10s_160_150cls_lr1e4_ep8.json`: 150 classes × 10 files/class, 8 epochs, 3-fold.
  - `configs/birdclef/sed_nfnet_balanced_oof_v15_10s_160_200cls_lr1e4_ep8.json`: 200 classes × 10 files/class, 8 epochs, 3-fold.
- **Commands launched on `192.168.0.10`:**
  - v14: `CUDA_VISIBLE_DEVICES=0`, pid `3280334`, log `logs/sed_oof_v14_nfnet_150cls_lr1e4_ep8_20260506T163713Z.log`, output `artifacts/sed_oof/sed-nfnet-balanced-oof-v14-10s-160-150cls-lr1e4-ep8/`.
  - v15: `CUDA_VISIBLE_DEVICES=1`, pid `3280336`, log `logs/sed_oof_v15_nfnet_200cls_lr1e4_ep8_20260506T163713Z.log`, output `artifacts/sed_oof/sed-nfnet-balanced-oof-v15-10s-160-200cls-lr1e4-ep8/`.
- **Status at log time:** both new runners started fold 0 cleanly. Next run should collect v14/v15 summaries, compare v14 against v12 and v13, then start NFNet TorchScript inference/kernel packaging around the best low-LR/8-epoch SED folds.

## 2026-05-06 17:35 UTC — NFNet v15 broad SED result + TorchScript bundle smoke

- **Track:** A+G Real SED frame/event inference packaging prep.
- **Hypothesis:** If the low-LR 8-epoch NFNet signal survives broad class coverage, package the best complementary TorchScript folds into a portable bundle that can become a Kaggle dataset/kernel input.
- **Branch/PR:** `feature/sed-smoke-export-scaffold`, PR #204.
- **Status checks:** Kaggle LB unchanged: latest scored v504/v503/v502/v501 at `0.927`, v500 at `0.926`; v505-v509 kernels COMPLETE/no failure messages; queue monitor pid `52652` is alive and sleeping on daily submission cap after v505 attempt.
- **Completed results collected:**
  - `v14` NFNet 150-class LR `1e-4`, 8 epochs: OOF macro AUC `0.620580` over 150 valid classes / 1500 files. This is essentially tied with v12 150-class 5-epoch (`0.620044`) and not a useful standalone upgrade.
  - `v15` NFNet 200-class LR `1e-4`, 8 epochs: OOF macro AUC `0.640274` over 181 valid classes / 1810 files. Fold AUCs `0.678960`, `0.692266`, `0.687347`; only 181 eligible classes reached the min-file threshold under the balanced selector.
- **OOF comparison artifacts:** on GPU server under `artifacts/sed_oof/comparisons/`.
  - `v12_vs_v14.json`: Pearson `0.658493`, mean abs diff `0.065642`; best blend uses v14 weight `0.6` for OOF AUC `0.633421` over 150 classes.
  - `v13_vs_v15.json`: on the 100-class overlap, v13 AUC `0.636878`, v15 overlap AUC `0.633091`, Pearson `0.294601`, mean abs diff `0.069209`; best blend uses v15 weight `0.6` for OOF AUC `0.657329`.
  - `v14_vs_v15.json`: on the 150-class overlap, v14 AUC `0.620580`, v15 overlap AUC `0.635854`, Pearson `0.359808`, mean abs diff `0.057227`; best blend uses v15 weight `0.7` for OOF AUC `0.652849`.
- **Interpretation:** v15 is the best broad-coverage SED candidate and has unusually low correlation with v13/v14, so a v13+v15 or v14+v15 fold blend is the right inference packaging target.
- **Packaging code added:**
  - `scripts/birdclef_sed_build_bundle.py` builds a manifest-based TorchScript SED bundle from one or more OOF experiment roots, with per-member blend weights and optional model copying.
  - `scripts/birdclef_sed_infer_torchscript.py` loads that manifest without timm/training code, decodes OGG via ffmpeg, recreates log-mel features, averages TorchScript fold probabilities, and writes wide CSV/NPZ predictions.
- **Bundle smoke validation on `192.168.0.10`:** built `artifacts/sed_bundles/sed-nfnet-v13v15-blend-v1/` from 6 TorchScript fold models with weights v13=0.4 and v15=0.6. Manifest has 6 models, 234 classes, copied model size `539.223 MB`. CPU smoke inference on 3 real train OGGs passed: 3 files × 234 classes, about `0.67 sec/file` with 2 torch threads after model load. Next step is a Kaggle-style inference script that maps soundscape 5s rows and blends SED probabilities into the current v504/v508 axis.

## 2026-05-06 18:35 UTC — Kaggle-style SED soundscape row inference smoke

- **Track:** A+G Real SED frame/event inference packaging prep.
- **Hypothesis:** The v13/v15 TorchScript SED bundle is only useful if it can produce BirdCLEF 5-second soundscape rows (`<soundscape_stem>_5` ... `_60`) in the exact `sample_submission` column shape. Validate that bridge before attempting a Kaggle kernel/dataset push.
- **Branch/PR:** `feature/sed-smoke-export-scaffold`, PR #204.
- **Status checks:** Kaggle LB unchanged: latest scored v504/v503/v502/v501 at `0.927`, v500 at `0.926`; v505-v509 kernels COMPLETE/no failure messages; queue monitor pid `52652` is alive and sleeping on daily submission cap after v505 attempt.
- **Code added:** `scripts/birdclef_sed_soundscape_infer.py`.
  - Loads the manifest bundle without timm/training code.
  - Decodes 60s OGG soundscapes with ffmpeg.
  - Emits one prediction row per 5s endpoint with Kaggle row ids.
  - Uses the trained model context length (10s) ending at each 5s row endpoint, zero-padded at file boundaries.
  - Aligns columns to `sample_submission.csv` when provided and can write CSV + compressed NPZ.
- **Smoke validation on `192.168.0.10`:** ran the v13/v15 bundle on one real train soundscape (`BC2026_Train_0001_S08_20250606_030007.ogg`) using CPU, 6 TorchScript folds, batch size 4, 2 torch threads. Output `artifacts/sed_bundles/sed-nfnet-v13v15-blend-v1/soundscape_smoke_submission.csv` has shape `12 x 235` (`row_id` + 234 labels), no NaNs, probability range `0.001024` to `0.422720`, and row ids `_5` through `_60`. Runtime was `6.409 sec/file` for a 60s soundscape.
- **Interpretation:** SED packaging now reaches real Kaggle row shape. The next implementation step is to embed this script into a Kaggle kernel candidate with the model bundle as an input dataset, then blend SED probabilities into the existing v504/v508 inference axis rather than submitting SED-only.

## 2026-05-06 19:50 UTC — v510 real SED bundle Kaggle kernel push

- **Track:** A+G Real SED frame/event inference packaging → Kaggle kernel candidate.
- **Hypothesis:** The strong low-correlation NFNet SED v13/v15 OOF bundle can add real temporal/model-family signal to the current v508 Perch/ProtoSSM axis if blended conservatively after existing probability shaping.
- **Branch/PR:** `feature/v510-real-sed-bundle-kernel`, PR #205. PR #204 was already merged, so this v510 work was moved to a fresh review branch.
- **Status checks:** Latest scored LB remains v504/v503/v502/v501 at `0.927` and v500 at `0.926`. v505-v509 kernels are `COMPLETE` with no failure messages, still waiting behind the daily submission cap.
- **Dataset packaging:** Created private Kaggle dataset `yourslewis/bc26-sed-nfnet-v13v15-bundle-v1` from `sed-nfnet-v13v15-blend-v1.zip` (manifest + 6 TorchScript NFNet folds; about 514 MB zipped / 539 MB unzipped). Upload used the repo helper `scripts/upload_kaggle_dataset_bearer.py` because legacy `kaggle datasets` CLI returned 401 under current KGAT auth.
- **Kernel candidate:** Added and pushed real Kaggle kernel `yourslewis/bc26-v510-real-sed-bundle-blend-005`, version 1.
- **Config:** Base is v508 (`ProtoSSM EW=0.625`, gamma `0.825`, context alpha `0.275`, top3 local-logit event propagation), plus `REAL_SED_BLEND_WEIGHT=0.05` after the v508 final probability post-processing.
- **Runtime guard:** v510 loads the zipped SED bundle from `/kaggle/input`, extracts it to `/kaggle/working`, selects TorchScript models with a time-budget guard (round-robin across v13/v15 if capped), emits 5-second soundscape row predictions aligned to `sample_submission.csv`, and falls back to pure v508 probabilities if the bundle is missing or inference fails.
- **Validation:** `py_compile` passed for the v510 kernel script, the dataset-upload helper, and the queue monitor script. Kaggle push returned version `1`, no invalid data/competition/kernel/model sources. Kernel status immediately after push: `RUNNING`, no failure message.
- **Queue monitor:** Refreshed monitor with v510 inserted after v509 and before old v376+ candidates. New monitor pid `68226`, log `logs/submit_pending_birdclef_queue_20260506T194754Z.log`; it retried v505 and is sleeping on the daily cap (~4.2h at launch). It will submit v505-v510 in order once quota returns, provided v510 completes.
- **Next step:** Monitor v510 kernel completion/failure. If it completes without timeout/model-mount issues, let the refreshed queue submit it after v505-v509. If v510 times out, reduce `REAL_SED_MAX_MODELS`/blend path to a 2-model v13+v15 representative bundle or precompute a lighter exported ONNX/OpenVINO SED path.

## 2026-05-06 20:45 UTC — v510 v1 fallback diagnosis + v2 mount-search fix

- **Track:** A+G Real SED frame/event Kaggle inference packaging and monitoring.
- **Hypothesis:** v510 version 1 completed but did not actually use the real SED signal because the SED dataset mounted as extracted files under a Kaggle-normalized directory rather than as `sed-nfnet-v13v15-blend-v1.zip` under the exact slug path.
- **Status checks:** Latest scored LB unchanged: v504/v503/v502/v501 at `0.927`, v500 at `0.926`. v505-v509 are `COMPLETE`. v510 v1 is `COMPLETE` with `submission.csv`, but its log shows `WARNING: real SED bundle dataset not found; using v508 probabilities only`, so it must not be submitted as the real SED candidate.
- **Root cause evidence:** Dataset API reports private dataset `yourslewis/bc26-sed-nfnet-v13v15-bundle-v1` is `ready` and lists extracted files (`sed_bundle_manifest.json` + six `models/*.pt`) rather than the zip archive. The v510 v1 finder only checked exact slug manifest paths plus recursive zip names, not recursive manifest paths.
- **Fix:** Updated `_sed_find_manifest()` to recursively search `/kaggle/input/**/sed_bundle_manifest.json`, print manifest candidates / input roots for debugging, and only then fall back to zip extraction. Updated the queue monitor so v510 submits kernel version `2` instead of bad/fallback version `1`.
- **Validation:** `py_compile` passed for v510 script and queue monitor. Next: push v510 version 2 via Bearer API, verify the log contains `Real SED manifest candidates` and `Applied real SED bundle blend`, then keep the queue monitor on v505-v510 with v510 version 2.

## 2026-05-06 21:45 UTC — v510 v2 verified + v511 blend weight 0.10 follow-up

- **Track:** A+G Real SED frame/event Kaggle inference packaging and lightweight blend-weight tuning.
- **Status checks:** Latest scored LB still unchanged: v504/v503/v502/v501 at `0.927`, v500 at `0.926`. v505-v509 are `COMPLETE`, and v510 v2 is now `COMPLETE` with `submission.csv`.
- **v510 v2 verification:** Output log confirms the real SED path actually ran: `Real SED manifest candidates: /kaggle/input/datasets/yourslewis/bc26-sed-nfnet-v13v15-bundle-v1/sed_bundle_manifest.json`, `Loading 6/6 real SED TorchScript models`, `Real SED prob range: 0.000003 to 0.624691, mean: 0.0617; runtime 214.4s`, and `Applied real SED bundle blend: weight=0.05`. Dry-run output shape was `240 x 235`, wall time `370.6s`; this is safely within Kaggle CPU budget on the public dry-run workload.
- **Follow-up hypothesis:** Since v510 v2 successfully uses all six SED models and runtime is acceptable, test a single stronger SED blend weight before pivoting tracks. v511 changes only `REAL_SED_BLEND_WEIGHT=0.05 -> 0.10` on the same v508 + real SED bundle path.
- **Kernel candidate:** Added and pushed real Kaggle kernel `yourslewis/bc26-v511-real-sed-bundle-blend-010`, version 1, with no invalid data/competition/kernel/model sources.
- **Queue monitor:** Updated monitor queue to submit v510 version 2, then v511 version 1, before old v376+ variants. Next step is to restart/verify the monitor with this updated queue and monitor v511 completion/logs.

## 2026-05-06 22:45 UTC — v511 verified + v512 ultra-conservative SED blend

- **Track:** A+G Real SED frame/event Kaggle inference packaging and lightweight blend-weight tuning.
- **Status checks:** Latest scored LB remains unchanged: v504/v503/v502/v501 at `0.927`, v500 at `0.926`; v505-v512 kernels are `COMPLETE` or running as noted below. Existing queue monitor was alive and sleeping on daily cap after v505 retry.
- **v511 verification:** v511 version 1 completed with `submission.csv` and confirmed real SED usage: found the SED manifest under `/kaggle/input/datasets/yourslewis/bc26-sed-nfnet-v13v15-bundle-v1/sed_bundle_manifest.json`, loaded `6/6` TorchScript models, `Real SED prob range: 0.000003 to 0.624691, mean: 0.0617; runtime 222.2s`, and applied `REAL_SED_BLEND_WEIGHT=0.10`. Dry-run output shape was `240 x 235`, wall time `354.1s`; final prob range `0.017495` to `0.914253`, mean `0.4115`.
- **Follow-up hypothesis:** Complete the small planned SED blend-weight bracket (`0.02`, `0.05`, `0.10`) with a safer low-weight variant in case the real SED model improves rank diversity but is undercalibrated versus the v508 axis.
- **Kernel candidate:** Added and pushed real Kaggle kernel `yourslewis/bc26-v512-real-sed-bundle-blend-002`, version 1, changing only `REAL_SED_BLEND_WEIGHT=0.02` from the same v508 + SED bundle path. Kaggle push returned version `1` with no invalid sources.
- **Queue monitor:** Updated queue to submit v510 version 2, then v511 version 1, then v512 version 1 after v505-v509 and before old v376+ variants. Next step: restart/verify the monitor with v512 included and monitor v512 completion/logs for the same SED markers.

### Monitor refresh after v512 push

- Restarted consolidated queue monitor with v512 included: pid `72673`, log `logs/submit_pending_birdclef_queue_20260506T223724Z.log`.
- It retried v505 and hit the daily submission cap again, with about `82 minutes` remaining until UTC reset at restart time.
- Final kernel status in this run: v510 `COMPLETE`, v511 `COMPLETE`, v512 `RUNNING` with no failure message and no output log yet. Next run should verify v512 logs for `Real SED manifest candidates`, `Loading 6/6 real SED TorchScript models`, `Applied real SED bundle blend: weight=0.02`, and `submission.csv saved`.

## 2026-05-06 23:45 UTC — v512 verified + prioritize real SED submissions at reset

- **Track:** A+G Real SED frame/event Kaggle inference packaging and submission monitoring.
- **Status checks:** Latest scored LB still unchanged: v504/v503/v502/v501 at `0.927`, v500 at `0.926`. v505-v512 kernels are all `COMPLETE` with no failure messages.
- **v512 verification:** v512 version 1 completed with `submission.csv` and confirmed real SED usage: found the SED manifest under `/kaggle/input/datasets/yourslewis/bc26-sed-nfnet-v13v15-bundle-v1/sed_bundle_manifest.json`, loaded `6/6` TorchScript models, `Real SED prob range: 0.000003 to 0.624691, mean: 0.0617; runtime 233.3s`, and applied `REAL_SED_BLEND_WEIGHT=0.02`. Dry-run output shape was `240 x 235`, wall time `386.3s`; final prob range `0.019048` to `0.977209`, mean `0.4426`.
- **Queue decision:** Reordered the submission monitor to prioritize the genuinely new real SED candidates at the UTC reset. New order after already-scored v500-v504: v510 version 2 (`0.05`), v511 (`0.10`), v512 (`0.02`), then older v505-v509 postprocess candidates, then old v376+ variants. This avoids spending the next daily cap entirely on older micro-sweeps while real SED candidates wait another day.
- **Validation:** `py_compile` passed for the reordered queue monitor. Next step: restart the monitor before UTC reset and verify it submits v510/v511/v512 first when quota returns.

## 2026-05-07 00:35 UTC — real SED submissions queued after UTC reset

- **Track:** A+G Real SED frame/event Kaggle submission monitoring.
- **Status checks:** Latest scored LB remains unchanged: v504/v503/v502/v501 at `0.927`, v500 at `0.926`. After UTC reset, the queue monitor submitted five kernels before hitting the new daily cap: v510, v511, v512, v505, and v506. All five are currently `PENDING` score.
- **Submitted real SED candidates:**
  - v510 ref `52403401`: real NFNet SED v13/v15 bundle blend weight `0.05` + v508 axis.
  - v511 ref `52403421`: real NFNet SED v13/v15 bundle blend weight `0.10` + v508 axis.
  - v512 ref `52403456`: real NFNet SED v13/v15 bundle blend weight `0.02` + v508 axis.
- **Additional submitted older candidates:** v505 ref `52403474`, v506 ref `52403489`.
- **Queue monitor:** pid `74432`, log `logs/submit_pending_birdclef_queue_20260506T233656Z.log`. It attempted v507 after the five submissions and hit the daily cap with `23 hours` remaining, then slept `82920s`.
- **Interpretation:** The real SED blend bracket is now finally in Kaggle scoring. No further Kaggle submissions can be made today; next step is to monitor pending scores and then decide whether to continue real SED blend/runtime variants, pivot to pseudo-label/noisy-student, or prune if LB drops.

## 2026-05-07 01:35 UTC — v510 tied, v511 dropped, v513 rank-blend candidate

- **Track:** A+G Real SED frame/event Kaggle monitoring + calibrated follow-up.
- **Status checks:** New scores arrived: v510 real SED probability blend `0.05` scored `0.927` (safe tie), v511 stronger probability blend `0.10` scored `0.926` (drop), v505 and v506 older postprocess candidates scored `0.927`; v512 low probability blend `0.02` remains `PENDING`. Current best remains `0.927`.
- **Interpretation:** Real SED signal can tie the plateau, but probability blending appears calibration-sensitive; increasing the probability weight to 0.10 hurt LB. Since AUC only cares per-class rank ordering, the next safer follow-up is to keep the tied-safe 0.05 weight but blend in per-class rank space rather than probability space.
- **Kernel candidate:** Added and pushed real Kaggle kernel `yourslewis/bc26-v513-real-sed-rankblend-005`, version 1. It uses the same v508 + six-model NFNet SED bundle path but replaces `probs = (1-w)*v508_probs + w*sed_probs` with `rank_average_ensemble([v508_probs, sed_probs], weights=[0.95, 0.05])`, then clips to valid probability range.
- **Queue monitor:** Added v513 after v512 and before older v505-v509 in `scripts/submit_pending_birdclef_queue.py`. Daily cap is already consumed for today, so v513 is prepared for the next available submission window after it completes.

### v513 monitor status

- Restarted queue monitor with v513 included: pid `77351`, log `logs/submit_pending_birdclef_queue_20260507T013726Z.log`.
- Monitor sees v510/v511/v512 already submitted, then checks v513 and reports `RUNNING` with no failure message; it is sleeping in 10-minute intervals until v513 completes. No output log/files were available at final check.
- Current scoring at this point: v510 `0.927` (safe tie), v511 `0.926` (drop), v512 still `PENDING`, v505/v506 `0.927`; current best remains `0.927`.

### v513 completed + monitor hardening

- v513 completed and produced `submission.csv`. Output log confirms the intended rank-blend path: SED manifest found, `6/6` TorchScript models loaded, `Real SED prob range: 0.000003 to 0.624691, mean: 0.0617; runtime 336.8s`, `Applied real SED rank blend: weight=0.05`, output shape `240 x 235`, wall time `517.2s`.
- v512 completed but did not receive a public score: Kaggle returned runtime exceeded for the hidden submission, so v512 is not a useful scored candidate despite public dry-run success.
- Queue monitor crashed on a transient Kaggle `RemoteDisconnected`/`ConnectionError` while listing submissions after v513 completed. Hardened `scripts/submit_pending_birdclef_queue.py` to catch `ConnectionError`/`Timeout`, sleep 10 minutes, and retry instead of dying. Next: restart the monitor; it should wait on the daily cap and submit v513 at the next reset.

## 2026-05-07 03:45 UTC — v514 runtime-safe 2-model real SED rank-blend candidate

- **Track:** A+G Real SED frame/event Kaggle inference packaging and monitoring.
- **Status checks:** Current best remains `0.927`. Latest scored real SED results are v510 probability blend `0.05` = `0.927`, v511 probability blend `0.10` = `0.926`, v505/v506 = `0.927`; v512 completed but hidden submission exceeded allowed runtime and has no public score. v513 full six-model rank blend completed and is waiting behind the daily submission cap.
- **Hypothesis:** Since v512 hit hidden runtime limits and v513 full rank-blend uses all six TorchScript SED folds, keep the safer per-class rank-blend calibration from v513 but cap inference to one v13 + one v15 representative model (`REAL_SED_MAX_MODELS=2`). This should materially reduce CPU/runtime risk while preserving both low-correlation NFNet SED members.
- **Kernel candidate:** Added `kaggle-kernels/v514-real-sed-rankblend005-2model/` copied from v513 with only the runtime/model-count cap and metadata/message changed. Constants: `REAL_SED_BLEND_WEIGHT=0.05`, `REAL_SED_MAX_MODELS=2`, `REAL_SED_MIN_MODELS=2`, v508 base axis unchanged.
- **Validation:** Local syntax check passed with `python3 -m py_compile`. Pushed real Kaggle kernel `yourslewis/bc26-v514-real-sed-rankblend-005-2m`, version 1; push returned no invalid dataset/kernel/model sources.
- **Queue:** Added v514 immediately after v513 in `scripts/submit_pending_birdclef_queue.py`. Restarted monitor at `logs/submit_pending_birdclef_queue_20260507T034053Z.log` (pid `80441`); it confirmed v513 is complete, attempted submission, hit the daily cap with ~20h remaining, and is sleeping until the next UTC allowance. v514 is therefore queued but not yet reachable until v513 submits or the cap resets.
- **Next step:** Collect v514 completion log/output; at the next reset, let the monitor submit v513 first, then v514 if daily quota remains. If v513 hidden-times out or underperforms, v514 is the runtime-safe fallback real SED rank-blend candidate.

### v514 completion verification

- v514 completed successfully. Kaggle session output contains `submission.csv` and confirms intended runtime-safe path: manifest found under `/kaggle/input/datasets/yourslewis/bc26-sed-nfnet-v13v15-bundle-v1/sed_bundle_manifest.json`, loaded `2/6` TorchScript models, `Real SED prob range: 0.000000 to 0.825038, mean: 0.0523; runtime 125.8s`, applied `real SED rank blend: weight=0.05`, output shape `240 x 235`, wall time `331.1s` / `5.5 min`.
- Interpretation: v514 is now verified as the runtime-safe fallback to v513. It uses about 37% of v513's real-SED runtime on the public dry run (125.8s vs 336.8s) while keeping both v13/v15 members represented. It remains queued behind v513 due to daily cap.

## 2026-05-07 04:55 UTC — Spec D EfficientNet-B3 SED diversity OOF baseline

- **Track:** D Model zoo diversity baseline/backbone sweep, while A+G real SED submissions are blocked by daily cap.
- **Hypothesis:** The NFNet v13/v15 real SED bundle is the strongest packaged SED signal, but an EfficientNet-B3 SED may add a lower-correlation convolutional backbone component. Test it on the same 100-class balanced OOF harness before considering any bundle expansion.
- **Config:** Added `configs/birdclef/sed_b3_preflight_v16_10s_160.json` and `configs/birdclef/sed_b3_balanced_oof_v16_10s_160_100cls_lr1e4_ep8.json`. Main run uses backbone `efficientnet_b3`, 10s clips, 160 mels, hop 512, focal BCE gamma 1.5, label smoothing 0.01, mixup 0.2, sqrt positive class weights, LR `1e-4`, batch size 6, 8 epochs, 100 balanced classes x 10 files/class, 3 folds, TorchScript export only.
- **Smoke gate:** Remote CUDA preflight on `192.168.0.10` passed: 16 files, `efficientnet_b3`, input `[16,160,626]`, 1 epoch, macro AUC `0.6667` over 3 valid toy classes, TorchScript size `41.99 MB`, runtime `4.8s` after decode.
- **Command launched/run:** `CUDA_VISIBLE_DEVICES=0 python scripts/birdclef_sed_oof_runner.py --base-config configs/birdclef/sed_b3_balanced_oof_v16_10s_160_100cls_lr1e4_ep8.json --output-root artifacts/sed_oof/sed-b3-balanced-oof-v16-10s-160-100cls-lr1e4-ep8 --n-folds 3`, log `logs/sed_oof_v16_b3_100cls_20260507T043753Z.log`.
- **OOF result:** v16 B3 completed: OOF macro AUC `0.506158` over 100 valid classes. Fold AUCs: `0.569995`, `0.595988`, `0.559534`. Artifacts: `artifacts/sed_oof/sed-b3-balanced-oof-v16-10s-160-100cls-lr1e4-ep8/oof_predictions.npz` plus three `41.99 MB` TorchScript folds.
- **Comparisons:** vs v13 NFNet-100 ep8, B3 standalone is much weaker (`0.5062` vs `0.6369`) but low-correlation (`r=0.2723`) and gives a small blend bump at 10% B3: `0.638184` vs `0.636878`. vs v15 NFNet-200, B3 has very low correlation (`r=0.0627`) but no two-way blend gain. Three-way grid over v13/v15/v16 finds best `w13=0.35, w15=0.60, w16=0.05` with AUC `0.657364`, essentially tied with the old best v13/v15-only grid (`w13=0.40, w15=0.60`, AUC `0.657329`).
- **Interpretation:** EfficientNet-B3 is diverse but too weak to justify immediate Kaggle packaging. Keep v16 as a possible 5% auxiliary member if future bundle work needs extra diversity, but do not spend a daily submission slot on B3 until v513/v514 LB results are known. Next non-Kaggle research action should be pseudo-label/noisy-student cache or a stronger model-zoo candidate, not B3 packaging.

## 2026-05-07 05:55 UTC — Spec B pseudo-label cache smoke + labeled soundscape teacher cache

- **Track:** B Pseudo-label/noisy-student cache, while A+G v513/v514 are complete but blocked by daily submission cap.
- **Status checks:** Current best remains `0.927`. Latest scored real SED submissions are unchanged: v510 `0.927`, v511 `0.926`, v512 hidden runtime exceeded/no score, v505/v506 `0.927`. v513 and v514 kernels are both `COMPLETE` and queue monitor `80441` is sleeping on the daily cap before submitting v513.
- **Hypothesis:** Before training noisy-student models, build a durable row-level pseudo-label cache and measure whether the packaged NFNet v13/v15 teacher is calibrated enough for the spec's hard thresholds (`0.90/0.95/0.98`) or should be used only as soft labels / low-threshold features.
- **Implementation:** Added `scripts/birdclef_pseudolabel_cache_summary.py` to summarize row-level pseudo-label NPZ files against `train_soundscapes_labels.csv` with macro AUC, top-k recall, positive/negative threshold counts, class histograms, and previews. Extended `scripts/birdclef_sed_soundscape_infer.py` with `--soundscape-list` so cache generation can target the 66 labeled train soundscapes instead of accidentally scanning all `10,658` train soundscape OGGs.
- **Teacher:** Created remote manifest `artifacts/pseudolabels/sed-v13v15-2m-teacher-r0/sed_bundle_manifest_2m.json` with one v13 fold and one v15 fold, weights `0.5/0.5`, using absolute model paths from the existing SED bundle. This mirrors v514's runtime-safe two-model teacher.
- **Smoke gate:** 5 labeled soundscapes / 60 rows on GPU passed. Runtime `2.75s`; summary macro AUC `0.4781` over 19 valid classes, top-k recall `0.0`, and no probabilities above `0.90`. This showed the train soundscape labels include many `47158son*`/soundscape-heavy targets and that hard high-threshold pseudo positives are likely too strict for this teacher.
- **Full labeled cache:** Generated cache for the 66 labeled train soundscapes / 792 rows: `artifacts/pseudolabels/sed-v13v15-2m-teacher-r0/labeled_train_soundscape_probs.npz` and `.csv`; summary at `artifacts/pseudolabels/sed-v13v15-2m-teacher-r0/labeled_summary.json`. Runtime `7.67s` on GPU, `0.116s/file`, 2 models, 234 classes.
- **Cache metrics:** Macro AUC `0.555240` over 75 valid classes. Top-k recall remains low (`top1=0.0008`, `top3=0.0171`, `top5=0.0289`, `top10=0.0499`). Prob stats: max `0.8688`, mean `0.0509`, p95 `0.2162`, p99 `0.3782`; hard positives at `0.90/0.95/0.98` are all `0`. Negative counts are plentiful (`<=0.01`: `82,845`, `<=0.02`: `98,600`, `<=0.05`: `127,122`).
- **Interpretation:** The current NFNet SED bundle is not calibrated for high-confidence hard pseudo-labeling on labeled train soundscapes; the spec's `p095/p098` hard-positive variants should be rejected for this teacher unless logits are recalibrated or a stronger v501-v504 teacher cache is regenerated. The cache is still useful for soft-label/noisy-student experiments and for negative mining / low-confidence background signals. Next Spec B action should be a soft-label student or power-scaled teacher (`power=0.75/0.85`) rather than hard-positive thresholding.

## 2026-05-07 06:55 UTC — Spec B soft-label noisy-student B0 pilot

- **Track:** B Pseudo-label/noisy-student student training, while A+G v513/v514 remain complete but blocked by daily cap.
- **Status checks:** Current best remains `0.927`; latest scored submissions unchanged (`v510=0.927`, `v511=0.926`, `v512` hidden runtime exceeded/no score, `v505/v506=0.927`). v513/v514 are `COMPLETE`; monitor pid `80441` is still sleeping after daily cap.
- **Hypothesis:** Since the v13/v15 2-model teacher produced no hard positives at `0.90+`, test the softer route: train a small EfficientNet-B0 SED student on the row-level train-soundscape teacher probabilities with power scaling `0.85`. This checks whether a student can distill/denoise the teacher and become a usable new prediction artifact.
- **Implementation:** Added `scripts/birdclef_pseudolabel_student_train.py`, which reconstructs 10s endpoint windows from pseudo-label row IDs, trains a timm SED student with soft-label BCE, exports TorchScript, and evaluates against `train_soundscapes_labels.csv`. Added configs `configs/birdclef/pl_r1_b0_soft_power085_smoke.json` and `configs/birdclef/pl_r1_b0_soft_power085_labeled.json`.
- **Smoke gate:** 96 rows, 1 epoch, EfficientNet-B0, 10s/160-mel, batch 16, power `0.85`, mixup `0.2` passed on GPU. It exported a `15.391 MB` TorchScript model. Smoke student remained weak after 1 epoch: all-row student AUC `0.4457` vs teacher AUC `0.6643` over 23 valid classes.
- **Full labeled pilot:** Ran all 792 labeled rows, 4 epochs, same config. Artifacts under `artifacts/pseudolabels/students/pl-r1-b0-soft-power085-labeled-soundscapes/`: `student_predictions.npz`, `metrics.json`, `model_torchscript.pt`, and `teacher_student_blend.json`. Runtime `10.94s` after feature decode on GPU; TorchScript size `15.391 MB`.
- **Results:** Validation student AUC peaked at epoch 2 (`0.5443`) but ended lower at epoch 4 (`0.5209`), while the teacher validation AUC was `0.5508`. All-row student AUC `0.5052` vs teacher `0.5552` over 75 valid classes. Student-teacher correlation `0.3668`, MAE `0.3411`. Teacher+student blend grid shows only a tiny gain at 10% student (`0.55545` vs teacher `0.55524`).
- **Interpretation:** The soft-label B0 student is operational but not yet useful as a standalone or Kaggle-package candidate. Early stopping around epoch 2 is better than epoch 4, but the tiny blend gain is too small to justify submission. Next Spec B action should either (a) improve the teacher with a v501-v504/v508 cache before student training, or (b) test a stronger pseudo student/backbone (`NFNet` or `V2-S`) with early stopping and/or lower teacher power, rather than spending a submission slot on this B0 student.

## 2026-05-07 07:55 UTC — Spec B NFNet soft-label student follow-up

- **Track:** B Pseudo-label/noisy-student stronger-student follow-up while A+G v513/v514 remain complete but blocked by daily cap.
- **Status checks:** Current best remains `0.927`; latest scored submissions unchanged (`v510=0.927`, `v511=0.926`, `v512` hidden runtime exceeded/no score, `v505/v506=0.927`). v513/v514 are `COMPLETE`; monitor pid `80441` is still sleeping after the daily cap.
- **Hypothesis:** The B0 soft-label student was operational but weak and low-correlation. Try the stronger `eca_nfnet_l0` student with early stopping at 2 epochs using the same v13/v15 2-model soft-label cache and power `0.85`, to see if a more capable architecture can distill the teacher without overfitting.
- **Config:** Added `configs/birdclef/pl_r1_nfnet_soft_power085_smoke.json` and `configs/birdclef/pl_r1_nfnet_soft_power085_labeled_ep2.json`. Main run: `eca_nfnet_l0`, 10s/160-mel, LR `1e-4`, batch 8, 2 epochs, teacher power `0.85`, mixup `0.2`, 792 labeled rows, TorchScript export only.
- **Smoke gate:** 96 rows, 1 epoch passed on GPU and exported an `89.872 MB` TorchScript model. Smoke validation student AUC `0.6331` vs teacher `0.6755` over 21 valid classes; all-row teacher AUC `0.6643`. This was much healthier than the B0 smoke.
- **Full pilot:** All 792 rows completed in `13.09s` after feature decode. Artifacts under `artifacts/pseudolabels/students/pl-r1-nfnet-soft-power085-labeled-soundscapes-ep2/`: `student_predictions.npz`, `metrics.json`, `model_torchscript.pt`, and `teacher_student_blend.json`.
- **Results:** Final all-row student AUC `0.53213` vs teacher `0.55524` over 75 valid classes. Student-teacher correlation is very high (`0.9136`) with low MAE (`0.0243`), meaning the NFNet student mostly copied a slightly worse/smoothed teacher. Blend grid shows no gain: best is `0%` student, AUC `0.55524`; any positive student weight reduces AUC.
- **Interpretation:** Stronger NFNet distills the teacher much more faithfully than B0, but it does not improve or diversify the teacher. The bottleneck is teacher quality/calibration, not student capacity. Stop training more students on the current 2-model SED teacher cache. Next Spec B action should regenerate a stronger pseudo-label teacher from the v501-v504/v508 Kaggle axis or use the current SED cache only for negatives/background, not standalone positive-label student training.

## 2026-05-07 08:35 UTC — Spec B v508 teacher cache + longer B0 distillation sweep

- **Track:** B Pseudo-label/noisy-student cache and student tuning, while A+G v513/v514 remain complete but blocked by daily submission cap.
- **Status checks:** Current best remains `0.927`. Latest scored submissions: v510 real SED probability blend `0.05` = `0.927`, v511 `0.10` = `0.926`, v512 hidden runtime exceeded/no public score, v505/v506 = `0.927`. v513/v514 are complete and queue monitor pid `80441` is still sleeping on the daily cap before submitting v513.
- **Hypothesis:** The prior 2-model NFNet SED teacher cache was too weak/calibration-poor for pseudo-label positives. Regenerate the 66 labeled-soundscape cache using the stronger v508 Kaggle inference stack itself (`ProtoSSM EW=0.625`, gamma `0.825`, context alpha `0.275`, top3 local-logit event propagation), then test whether a tiny B0 student can distill that stronger teacher.
- **Kaggle teacher-cache kernel:** Added `kaggle-kernels/v515-v508-teacher-cache66/`, copied from v508 with `DRYRUN_N_FILES=66` and metadata `yourslewis/bc26-v515-v508-teacher-cache66`. Pushed real Kaggle kernel version 1; Kaggle canonical slug returned `yourslewis/bc26-v515-v508-teacher-cache-66`. Kernel completed with `submission.csv`.
- **Teacher cache artifacts:** Downloaded v515 `submission.csv` and converted it to `artifacts/pseudolabels/v508-teacher-cache66/predictions.npz` locally; generated summary `artifacts/pseudolabels/v508-teacher-cache66/summary.json`.
- **Teacher cache metrics:** 792 rows x 234 classes. Macro AUC vs labeled train soundscapes `0.991149` over 75 valid classes, top-k recall `top1=0.2266`, `top3=0.5272`, `top5=0.6437`, `top10=0.7689`. Prob stats: min `0.01535`, max `0.99718`, mean `0.44222`, p95 `0.72743`, p99 `0.89057`. High-confidence positives now exist: `1668` entries >= `0.90`, `880` >= `0.95`, `372` >= `0.98`.
- **Caution:** This v508 teacher-cache score is a labeled-train soundscape sanity check, not a clean OOF estimate; v508 trains/fits components on the same labeled soundscape pool before dry-running the 66 files, so the very high AUC may include in-sample leakage. Use it for operational distillation experiments, not as proof of generalization.
- **Configs added:**
  - `configs/birdclef/pl_r1_b0_v508_soft_p100_smoke.json`
  - `configs/birdclef/pl_r1_b0_v508_soft_p100_labeled.json`
  - `configs/birdclef/pl_r1_b0_v508_soft_p100_lr1e3_nomix_ep20.json`
  - `configs/birdclef/pl_r1_b0_v508_soft_p100_lr3e4_mix02_ep20.json`
- **Smoke gate:** B0, 96 rows, 1 epoch, teacher power `1.0`, mixup `0.2` passed on GPU and exported a `15.391 MB` TorchScript model. Smoke all-row student AUC `0.46125` vs teacher `0.98727` over 23 valid classes, so the first epoch underfit badly but the pipeline was valid.
- **Full B0 soft-label baseline:** 792 rows, 4 epochs, LR `3e-4`, mixup `0.2`. Final all-row student AUC `0.78116` vs teacher `0.99115`, student-teacher corr `0.6189`, MAE `0.1247`; blend grid did not beat teacher.
- **Longer B0 sweep 1:** 792 rows, 20 epochs, LR `1e-3`, no mixup. Runtime `27.3s`, TorchScript `15.391 MB`. Validation student AUC climbed to a best of `0.98572` at epoch 19; final all-row student AUC `0.98340` vs teacher `0.99115`, corr `0.9798`, MAE `0.0289`. Blend grid best on all rows remains `0%` student; validation had only a negligible `1%` student gain (`0.994095` vs teacher `0.994091`).
- **Longer B0 sweep 2:** 792 rows, 20 epochs, LR `3e-4`, mixup `0.2`. Runtime `27.3s`, TorchScript `15.391 MB`. Final all-row student AUC `0.95997`, corr `0.9317`, MAE `0.0489`; blend grid again did not beat teacher.
- **Interpretation:** The stronger v508 teacher cache fixes the pseudo-label confidence problem and produces meaningful hard-positive candidates, but a tiny B0 student mostly learns to imitate the teacher and does not provide useful blend diversity. The best distilled model is lightweight and close to the teacher, but too correlated to justify a Kaggle submission slot yet. Next Spec B step should add a clean OOF/holdout teacher-cache path or use hard high-confidence v508 positives/negatives for regularizing a genuinely independent real-audio SED model, rather than submitting this B0 student directly.

## 2026-05-07 09:35 UTC — Spec B v508 hard-confidence student/regularizer pilot

- **Track:** B Pseudo-label/noisy-student hard-positive/negative regularization, while A+G v513/v514 remain complete but blocked by daily submission cap.
- **Status checks:** Current best remains `0.927`. Latest scored submissions unchanged: v510 `0.927`, v511 `0.926`, v512 hidden runtime/no public score, v505/v506 `0.927`. v513 and v514 are both `COMPLETE` with `submission.csv`; queue monitor pid `80441` remains alive and sleeping on the daily cap before submitting v513.
- **Hypothesis:** The v508 teacher cache created useful high-confidence positive cells, but soft-label B0 distillation was too correlated with the teacher. Train a masked hard-confidence student using only `p>=positive_threshold` as positives and `p<=0.05` as negatives, ignoring the ambiguous middle, to see whether it creates lower-correlation signal that can regularize/blend with the v508 teacher.
- **Implementation:** Extended `scripts/birdclef_pseudolabel_student_train.py` with `target_mode="hard_conf"`, `positive_threshold`, `negative_threshold`, target-mask accounting, and masked BCE loss. Soft-label behavior remains the default. Added configs:
  - `configs/birdclef/pl_r1_b0_v508_hard_p90n05_smoke.json`
  - `configs/birdclef/pl_r1_b0_v508_hard_p90n05_lr1e3_ep20.json`
  - `configs/birdclef/pl_r1_b0_v508_hard_p95n05_lr1e3_ep20.json`
- **Smoke gate:** B0, 96 rows, 1 epoch, `p>=0.90` positives / `p<=0.05` negatives passed on GPU. Mask fraction `0.0176`, positive cells `178`, negative cells `217`; smoke all-row student AUC `0.5357` vs teacher `0.9873` over 23 classes. This validated the masked-loss path.
- **P90/N05 full run:** 792 rows, 20 epochs, LR `1e-3`, no mixup. Mask fraction `0.02441`, positives `1668`, negatives `2856`, runtime `27.8s`, TorchScript `15.391 MB`. Final student AUC `0.63155` vs teacher `0.99115` over 75 classes, corr `0.2381`, MAE `0.2338`. Validation AUC peaked at epoch 2 (`0.6608`) then overfit. Despite weak standalone AUC, blend grid showed a small all-row gain at `10%` student: `0.991539` vs teacher `0.991149`; validation best `20%` student: `0.994126` vs teacher `0.994091`.
- **P95/N05 full run:** 792 rows, 20 epochs, LR `1e-3`, no mixup. Mask fraction `0.02016`, positives `880`, negatives `2856`, runtime `27.1s`, TorchScript `15.391 MB`. Final student AUC `0.56397`, corr `0.2670`, MAE `0.1891`. Blend grid showed a smaller all-row gain at `10%` student: `0.991237`; validation best `10%` student: `0.994411`.
- **Interpretation:** Hard-confidence training gives genuinely lower-correlation signal than soft distillation, and tiny teacher+hard-student blend gains appear on the labeled-train sanity set. However, the standalone students are weak and the validation split is still not clean OOF because the v508 teacher cache is in-sample/leaky. Do not submit this directly. The useful next step is to add early-best checkpointing and/or a clean OOF teacher-cache/hard-label path before packaging any hard-confidence student blend.

## 2026-05-07 10:35 UTC — Spec B hard-confidence early-best checkpointing

- **Track:** B Pseudo-label/noisy-student hard-confidence regularizer refinement, while A+G v513/v514 remain complete but blocked by daily submission cap.
- **Status checks:** Current best remains `0.927`. Latest scored submissions unchanged: v510 `0.927`, v511 `0.926`, v512 hidden runtime/no public score, v505/v506 `0.927`. v513/v514 remain `COMPLETE` with `submission.csv`; queue monitor pid `80441` is alive and sleeping on the daily cap before submitting v513.
- **Hypothesis:** The previous hard-confidence p90/n05 student peaked early and then overfit. Restoring the best validation-AUC checkpoint should preserve the lower-correlation signal while improving standalone sanity-set AUC and export the actually selected checkpoint rather than the final overfit epoch.
- **Implementation:** Added `restore_best_by_val_auc` to `scripts/birdclef_pseudolabel_student_train.py`. When enabled, it tracks max `val_student_vs_truth.macro_auc`, stores the best state dict, restores it before final all-row prediction/export, and writes `best_checkpoint_info.json`. Default remains `false` to preserve existing config behavior.
- **Configs added:**
  - `configs/birdclef/pl_r1_b0_v508_hard_p90n05_lr1e3_ep20_bestval.json`
  - `configs/birdclef/pl_r1_b0_v508_hard_p95n05_lr1e3_ep20_bestval.json`
- **P90/N05 best-val run:** 792 rows, 20 epochs, LR `1e-3`, no mixup, `restore_best_by_val_auc=true`. Selected epoch `2` with validation AUC `0.658354`. Final restored all-row student AUC `0.658888` vs teacher `0.991149`, corr `0.2185`, MAE `0.2046`. Blend grid best all-row is only a tiny `2%` student gain (`0.991165` vs `0.991149`); validation best is also `2%` student (`0.994245` vs teacher `0.994091`). Artifact path: `artifacts/pseudolabels/students/pl-r1-b0-v508-hard-p90n05-lr1e3-ep20-bestval/`.
- **P95/N05 best-val run:** attempted after p90, but GPU 0 OOMed because a separate non-BirdCLEF LRM job was occupying ~22.6GB on GPU 0 and another LRM process occupied GPU 1. I did not kill those jobs. The p95 best-val config is committed for later rerun when GPU memory is free.
- **Interpretation:** Early-best checkpointing fixes the overfit-final-export issue and improves p90 hard-confidence student standalone AUC (`0.6589` vs previous final `0.6315`), but the teacher+student blend gain remains negligible on the leaky sanity set. This is useful infrastructure, not a Kaggle candidate yet. Next step remains a clean OOF teacher-cache/hard-label path or applying hard labels as a regularizer inside an independent SED OOF harness.

## 2026-05-07 11:35 UTC — Spec B clean OOF pseudo-label diagnostics

- **Track:** B Pseudo-label/noisy-student clean teacher-cache diagnostics, while A+G v513/v514 remain complete but blocked by daily submission cap.
- **Status checks:** Current best remains `0.927`. Latest scored submissions unchanged: v510 `0.927`, v511 `0.926`, v512 hidden runtime/no public score, v505/v506 `0.927`. v513 and v514 remain `COMPLETE` with `submission.csv`. At final check, old queue monitor pid `80441` was gone, so I restarted it as pid `99291`, log `logs/submit_pending_birdclef_queue_20260507T114053Z.log`; it confirmed v513 is complete, attempted submission, hit the daily cap with ~12h remaining, and is sleeping until retry.
- **Hypothesis:** The v508 teacher-cache hard positives looked strong only on an in-sample/leaky labeled-train sanity set. Before training more students, inspect clean OOF SED predictions to see whether high-confidence pseudo positives are actually precise under out-of-fold evaluation.
- **Implementation:** Added `scripts/birdclef_oof_pseudolabel_diagnostics.py`. It loads one or more `oof_predictions.npz` artifacts, aligns overlapping files/labels, computes macro AUC, top-k recall, correlation, ensemble/blend diagnostics, and threshold precision/recall for pseudo-label cutoffs. This gives a reusable clean gate for pseudo-label thresholds.
- **Artifacts analyzed:** pulled remote OOF artifacts for v13 NFNet-100 ep8, v15 NFNet-200 ep8, and v16 B3-100 ep8 into local ignored `artifacts/sed_oof/...` and wrote summaries under `artifacts/pseudolabels/oof-teacher-diagnostics/`.
- **v13/v15 clean OOF ensemble:** On the 1000-file overlap, v13/v15 weights `0.4/0.6` gives macro AUC `0.657329` over 100 classes, top-k recall top1 `0.087`, top3 `0.146`, top5 `0.184`, top10 `0.259`. High thresholds are extremely sparse: `p>=0.90` gives 11 cells with precision `0.636` and recall `0.007`; `p>=0.95` gives 5 cells with precision `0.800`; `p>=0.98` gives 2 cells with precision `1.000`. Negatives at `p<=0.05` are abundant and clean: 125,015 cells, negative precision `0.99934`.
- **v13/v15/v16 clean OOF ensemble:** On the same 1000-file overlap, weights `0.35/0.60/0.05` gives macro AUC `0.657364`, confirming the prior tiny B3 blend bump. Thresholds remain too sparse for positive mining: `p>=0.90` gives 10 cells, precision `0.700`, recall `0.007`; `p>=0.95` gives 2 cells, precision `1.000`; no `p>=0.98` cells. `p<=0.05` negatives remain abundant/clean with negative precision `0.99954`.
- **v15 full clean OOF:** On all 1810 v15 files, macro AUC `0.640274` over 181 classes. High thresholds are not reliable positives at wider class coverage: `p>=0.90` gives 202 cells but precision only `0.119`; `p>=0.95` gives 97 cells with precision `0.113`; `p>=0.98` gives 42 cells with precision `0.119`. `p<=0.05` negatives remain high precision (`0.99801`) but include 465 false negatives.
- **Interpretation:** Clean OOF diagnostics strongly contradict the leaky v508-cache hard-positive story. High-confidence positives are either too sparse (v13/v15 overlap) or low precision (v15 full coverage), while low-probability negatives are consistently plentiful and very clean. Stop pursuing hard-positive pseudo-label students until a stronger clean OOF teacher exists. Near-term Spec B value is negative/background/no-call regularization or cleaner OOF teacher generation, not hard positive mining from the current SED OOF artifacts.

## 2026-05-07 12:35 UTC — Spec B clean OOF negative-cache export

- **Track:** B Pseudo-label/noisy-student negative/background/no-call regularization preparation, while A+G v513/v514 remain complete but blocked by daily submission cap.
- **Status checks:** Current best remains `0.927`. Latest scored submissions unchanged: v510 `0.927`, v511 `0.926`, v512 hidden runtime/no public score, v505/v506 `0.927`. v513/v514 remain `COMPLETE` with `submission.csv`. Queue monitor pid `99291` remains alive and sleeping after hitting the daily cap on v513 with ~12h remaining. PR #205 is now merged, so this run started fresh branch `feature/oof-negative-pseudolabel-cache` from `origin/main`.
- **Hypothesis:** Clean OOF diagnostics showed high-confidence positives are sparse/unreliable, but `p<=0.05` negatives are abundant and precise. Export a capped OOF negative-cache artifact that future SED/no-call experiments can consume for negative/background regularization without trusting leaky in-sample v508 positives.
- **Implementation:** Added `scripts/birdclef_oof_negative_cache.py`. It loads one or more OOF artifacts, aligns common files/labels, builds a weighted teacher ensemble, exports `negative_mask` / `positive_mask` / raw masks / teacher predictions / truth to compressed NPZ, and writes a `.summary.json` with precision/recall and mask-density accounting. It supports per-row/per-class caps to avoid over-weighting common easy negatives.
- **v13/v15 negative cache:** weights `0.4/0.6`, negative threshold `0.05`, positive threshold `0.95`, max `64` negatives/row. Output `artifacts/pseudolabels/oof-negative-cache/v13v15_neg005_pos095_cache.npz` (ignored artifact). Raw negatives: `125,015` cells, negative precision `0.999336`, false negatives `83`. Capped negatives: `64,000` cells across all `1000` rows and `176` classes, **negative precision `1.000000`** with zero false negatives on this OOF overlap. Raw positives remain tiny: 5 cells, precision `0.800`.
- **v13/v15/v16 negative cache:** weights `0.35/0.60/0.05`, same thresholds/caps. Output `artifacts/pseudolabels/oof-negative-cache/v13v15v16_neg005_pos095_cache.npz`. Raw negatives: `109,713` cells, negative precision `0.999544`, false negatives `50`. Capped negatives: `63,715` cells across all `1000` rows and `211` classes, negative precision `0.999984` with only one false negative. Raw positives: 2 cells, precision `1.000`.
- **Validation:** `python3 -m py_compile scripts/birdclef_oof_negative_cache.py` passed; both cache-generation commands completed. The clean capped negative masks are much more reliable than any current positive pseudo-label set.
- **Interpretation:** This creates a concrete OOF-derived negative regularization artifact and confirms the correct pivot: use current SED OOF signal for negatives/background/no-call suppression, not positive pseudo-label mining. Next step is to wire this cache into a SED training loss as an auxiliary masked-negative penalty or no-call/taxon gate smoke, ideally on a small B0/V2S OOF run when GPU memory is available.

## 2026-05-07 13:41 UTC — Spec B negative-cache auxiliary loss smoke

- **Track:** B Pseudo-label/noisy-student negative/background regularization wiring, while A+G v513/v514 remain complete but blocked by daily submission cap.
- **Status checks:** Current best remains `0.927`. Latest scored submissions unchanged: v510 `0.927`, v511 `0.926`, v512 hidden runtime/no public score, v505/v506 `0.927`. v513/v514 remain `COMPLETE` with `submission.csv`. Queue monitor pid `99291` remains alive and sleeping after daily cap on v513. PR #206 is open/mergeable.
- **Hypothesis:** The clean OOF negative cache can be used as a safe auxiliary no-call/background regularizer in SED training. First, wire the cache into the trainer and smoke it on cache-backed examples before launching any GPU OOF run.
- **Implementation:** Extended `scripts/birdclef_sed_pilot_train.py` with `oof_negative_cache`, `aux_negative_weight`, and `oof_negative_mask_key`. Added path-key alignment between cache files and local/remote train-audio paths, a new `selection_strategy="oof_negative_cache"`, masked negative BCE (`target=0` only where the OOF cache says negative), and `aux_negative_summary` in metrics. Existing configs keep old behavior because the new weight defaults to `0.0`.
- **Configs added:**
  - `configs/birdclef/sed_tiny_oof_negcache_smoke.json` — CPU/tiny smoke, 24 cache-backed files, aux negative weight `0.1`.
  - `configs/birdclef/sed_b0_oof_negcache_v17_10s_160_100cls_lr3e4_ep4.json` — prepared B0 1000-file 3-fold style candidate, 10s/160-mel, aux negative weight `0.1`, not launched yet because GPUs are occupied by non-BirdCLEF LRM jobs.
- **Smoke gate:** Ran on GPU server CPU path with `CUDA_VISIBLE_DEVICES=""` to avoid interfering with occupied GPUs. Command: `python scripts/birdclef_sed_pilot_train.py --config configs/birdclef/sed_tiny_oof_negcache_smoke.json`. Result: passed in `8.727s`, 24 examples, input `[24,128,313]`, TorchScript export `0.183 MB`, macro AUC `0.5` over 6 toy-valid classes. Crucially, cache alignment worked: `covered_rows=24`, coverage `1.0`, negative cells `1536`, mean `64` negatives per covered row.
- **GPU status:** Two non-BirdCLEF LRM jobs are still using the GPUs (~16.6GB on GPU0, ~11.5GB on GPU1), so I did not launch the prepared B0 OOF run.
- **Interpretation:** Negative-cache auxiliary loss is now wired and smoke-tested. Next safe step, once GPU memory is available, is to launch the prepared B0 OOF neg-cache candidate and compare against the comparable B0 baseline before considering any packaging.

## 2026-05-07 14:41 UTC — Spec B B0 negative-cache OOF candidate + lower-weight attempt

- **Track:** B Pseudo-label/noisy-student negative/background regularization OOF test.
- **Status checks:** Current best remains `0.927`. Latest scored submissions unchanged: v510 `0.927`, v511 `0.926`, v512 hidden runtime/no public score, v505/v506 `0.927`. v513/v514 remain `COMPLETE` with `submission.csv`; queue monitor pid `99291` remains alive and sleeping on the daily cap before v513 retry. PR #206 is open and mergeable.
- **GPU status:** At start, GPU1 had ~20GB free while GPU0 was mostly occupied, so I used `CUDA_VISIBLE_DEVICES=1`. During the later lower-weight attempt, the non-BirdCLEF LRM job on physical GPU1 expanded to ~21.8GB, causing an OOM; I did not kill it.
- **Main run:** Launched and completed prepared `configs/birdclef/sed_b0_oof_negcache_v17_10s_160_100cls_lr3e4_ep4.json`: B0, 1000 cache-backed OOF files, 3 folds, 10s/160-mel, focal BCE, aux negative weight `0.1`, max 64 OOF negatives/row. Log `logs/sed_oof_negcache_v17_20260507T144221Z.log`.
- **v17 OOF result:** complete, `1000` OOF rows, macro AUC `0.474693` over 100 classes. Fold AUCs: `0.527719`, `0.556185`, `0.538529`. Each fold exported a `15.389 MB` TorchScript model and had full cache coverage (`1000` covered rows, `64,000` negative cells total). Compared with B0 v8 baseline on the same 1000 files, v17 is worse: baseline `0.485820` vs v17 `0.474693`, flat correlation `0.5106`, and blend grid best remains `0%` v17.
- **Lower-weight attempt:** Added `configs/birdclef/sed_b0_oof_negcache_v18_10s_160_100cls_lr3e4_w002_ep4.json` with aux negative weight `0.02`. Fold 0 completed with AUC `0.547484`, but fold 1 failed with CUDA OOM after the non-BirdCLEF LRM process grew on the selected GPU. Partial result is encouraging only directionally, not a valid OOF comparison. Rerun later with free GPU and/or lower batch size if this lane is continued.
- **Interpretation:** The first full negative-cache B0 candidate did not beat the comparable B0 baseline, so aux negative weight `0.1` is too strong or the training recipe needs retuning. Current evidence does not justify Kaggle packaging. If continuing, retry a lower-weight run (`0.02`) under clean GPU conditions and compare complete OOF before doing anything more expensive.

## 2026-05-07 15:41 UTC — Spec B lower-weight negative-cache B0 rerun

- **Track:** B Pseudo-label/noisy-student negative/background regularization retest.
- **Status checks:** Current best remains `0.927`. Latest scored submissions unchanged: v510 `0.927`, v511 `0.926`, v512 hidden runtime/no public score, v505/v506 `0.927`. v513/v514 remain `COMPLETE` with `submission.csv`; queue monitor pid `99291` remains alive and sleeping on the daily cap before v513 retry. PR #206 remains open/mergeable.
- **Hypothesis:** v17 with aux negative weight `0.1` hurt OOF, while v18's partial fold0 looked better before OOM. Retry the lower aux-negative weight `0.02` with smaller batch size (`6`) on the now-more-available GPU0 to get a full OOF comparison without killing unrelated LRM jobs.
- **Config/run:** Added `configs/birdclef/sed_b0_oof_negcache_v19_10s_160_100cls_lr3e4_w002_bs6_ep4.json`. It uses B0, 1000 cache-backed files, 3 folds, 10s/160-mel, focal BCE, aux negative weight `0.02`, batch size `6`, max 64 OOF negatives/row. Launched with `CUDA_VISIBLE_DEVICES=0`; log `logs/sed_oof_negcache_v19_20260507T154210Z.log`.
- **v19 OOF result:** complete, `1000` OOF rows, macro AUC `0.475945` over 100 classes. Fold AUCs: `0.603197`, `0.574165`, `0.567725`; each fold exported `15.389 MB` TorchScript. Compared with B0 v8 baseline on the same files, v19 remains worse: baseline `0.485820` vs v19 `0.475945`, flat correlation `0.6328`, mean absolute diff `0.0344`, and blend grid best remains `0%` v19.
- **Interpretation:** Lowering aux-negative weight and batch size did not recover baseline performance. This lane has now failed two full OOF tests (`0.1` and `0.02` aux weights), so stop spending cycles on B0 negative-cache regularization unless trying a substantially different recipe (e.g. a dedicated no-call/taxon gate or different backbone). Do not package/submit these negative-cache B0 models.

## 2026-05-07 15:41 UTC — Spec D model-zoo RegNet/ConvNeXt SED sweep

- **Track:** D Model zoo diversity sweep after B0 negative-cache regularization failed at aux weights `0.1` and `0.02`.
- **Status checks:** Current best remains `0.927`. Latest scored submissions unchanged: v510 `0.927`, v511 `0.926`, v512 hidden runtime/no public score, v505/v506 `0.927`. v513/v514 remain `COMPLETE` with `submission.csv`; queue monitor pid `99291` remains alive and sleeping on the daily cap before v513 retry.
- **Branch/PR:** Created branch `feature/zoo-regnet-convnext-sed` from `origin/main` for this new lane. Existing PR #206 remains separate for negative-cache artifacts.
- **Hypothesis:** RegNetY-008 or ConvNeXt-Tiny may add a lower-correlation SED prediction family beyond EfficientNet-B0/B3 and NFNet. Use the same 100-class balanced OOF benchmark as B0 v8 / NFNet v11/v13 before considering packaging.
- **Smoke configs:** Added `zoo_regnety008_smoke.json` and `zoo_convnext_tiny_smoke.json` (48 files, 12 classes, 5s/128-mel, 1 epoch, CUDA GPU1). Both used real timm backbones, no fallback. RegNet smoke: AUC `0.548776` over 6 toy-valid classes, TorchScript `23.414 MB`, runtime `7.149s`. ConvNeXt smoke: AUC `0.460483`, TorchScript `112.350 MB`, runtime `10.0s`.
- **Scaled config/run:** Added and ran `zoo_regnety008_balanced_oof_v20_10s_160_100cls_lr1e4_ep5.json`: `regnety_008`, 1000 files / 100 classes, 10s/160-mel, 3-fold OOF, 5 epochs, LR `1e-4`, focal BCE gamma `1.5`, sqrt positive class weighting, label smoothing `0.01`, mixup `0.2`, batch size `8`. Log: `logs/sed_oof_v20_regnety008_20260507T155216Z.log`.
- **v20 RegNet result:** complete OOF AUC `0.506402` over 100 classes. Fold AUCs: `0.572570`, `0.560841`, `0.573995`; each fold TorchScript `23.414 MB`. Compared with B0 v8: RegNet is better (`0.506402` vs `0.485820`), Pearson `0.6497`, best blend `90%` RegNet gives `0.508803`. Compared with NFNet v11: RegNet is worse (`0.506402` vs `0.622721`), Pearson `0.8238`, best blend remains `0%` RegNet. Compared with NFNet v13: RegNet is worse (`0.506402` vs `0.636878`), Pearson `0.7013`, best blend remains `0%` RegNet.
- **Interpretation:** RegNetY-008 is a valid model-zoo baseline and modestly improves over B0, but it is not competitive with the existing NFNet SED family and adds no blend value to NFNet on this benchmark. Keep artifacts as a diversity baseline; do not package for Kaggle. ConvNeXt smoke passed but has much larger TorchScript size and weaker smoke AUC, so defer full ConvNeXt until stronger lanes are exhausted or a low-correlation need remains.

## 2026-05-07 16:41 UTC — Spec A NFNet 20s context sweep launched

- **Track:** A Real SED frame/event model hyperparameter tuning after B0 negative-cache (`v17`/`v19`) and RegNet zoo (`v20`) failed to add value over NFNet. This returns to the strongest current OOF family, NFNet-L0, and tests the spec's crop-length knob rather than another postprocess micro-sweep.
- **Status checks:** Current best remains `0.927`. Latest scored submissions unchanged: v510 `0.927`, v511 `0.926`, v512 hidden runtime/no public score, v505/v506 `0.927`. v513/v514 remain `COMPLETE` with `submission.csv`; queue monitor pid `99291` remains alive and sleeping on the daily cap before v513 retry.
- **Branch/PR:** Created branch `feature/nfnet-sed-context-sweep` from `origin/main` for this lane.
- **Hypothesis:** The best current SED bundle uses NFNet-L0 10s/160-mel models (`v13`/`v15`). A 20s context window may improve localization/context for long soundscape rows and provide complementary signal, if it fits GPU/Kaggle runtime constraints.
- **Smoke:** Added `configs/birdclef/sed_nfnet_20s_smoke.json` and ran 48 files / 12 classes, 20s/160-mel, NFNet-L0, batch size 1, 1 epoch on GPU0. Smoke passed with input shape `[48,160,1251]`, macro AUC `0.481481` over 6 toy-valid classes, TorchScript `89.871 MB`, runtime `7.419s`, prediction time `0.089s`.
- **Full config:** Added `configs/birdclef/sed_nfnet_balanced_oof_v22_20s_160_100cls_lr1e4_ep5.json`: NFNet-L0, 100 classes / 1000 files, 20s/160-mel, 3-fold OOF, 5 epochs, LR `1e-4`, focal BCE gamma `1.5`, sqrt positive class weighting, label smoothing `0.01`, mixup `0.2`, batch size `1`.
- **First full attempt:** Launched v22 on GPU0; fold0 failed immediately with CUDA OOM because unrelated process `3768035` expanded to `22.54 GiB`, leaving only `6.50 MiB` free. This is a resource-collision failure, not a model/root-cause failure. I did not kill the unrelated LRM process.
- **Retry:** Added `configs/birdclef/sed_nfnet_balanced_oof_v22b_20s_160_100cls_lr1e4_ep5_gpu1retry.json` with a distinct output dir and launched on GPU1, where free memory recovered. Durable process `4190799` is running under wrapper `4190796`; log `logs/sed_oof_v22b_nfnet_20s_gpu1retry_20260507T164420Z.log`; current state at check time: fold0 running, no stderr/output yet. Next run should collect fold/OFF summary and compare v22b against NFNet v13/v15 before deciding whether 20s context is worth packaging.

## 2026-05-07 17:41 UTC — Spec A v22b monitor blocked, v23 fallback prepared

- **Track:** A Real SED NFNet crop/context tuning, continuing PR #208 (`feature/nfnet-sed-context-sweep`).
- **Status checks:** Current best remains `0.927`. Latest scored submissions unchanged: v510 `0.927`, v511 `0.926`, v512 hidden runtime/no public score, v505/v506 `0.927`; v513/v514 remain complete with `submission.csv`. Queue monitor pid `99291` is alive and still sleeping after daily cap on v513.
- **v22b monitoring:** Attempted to collect `sed-nfnet-balanced-oof-v22b-20s-160-100cls-lr1e4-ep5-gpu1retry` from GPU server, but SSH repeatedly timed out during banner exchange after TCP connect. Ping succeeds, so the host is reachable but SSH is not currently accepting/completing sessions. Last known v22b state from prior run: pid `4190799` fold0 running on GPU1 with no stderr.
- **Prepared fallback:** Added `configs/birdclef/sed_nfnet_balanced_oof_v23_20s_128_100cls_lr1e4_ep5_fallback.json`: same NFNet-L0 20s context / 100-class / 5-epoch OOF recipe as v22b, but lowers mel bins from `160` to `128` to reduce memory and runtime if v22b fails or proves too slow. Do not launch v23 until v22b outcome is known or SSH access recovers.
- **Next step:** Retry SSH later, collect v22b OOF summary/logs, compare against NFNet v13/v15. If v22b failed due memory/runtime, launch v23 fallback; if v22b completes and does not improve OOF/blend, kill the 20s context lane.

## 2026-05-07 18:41 UTC — Spec A v22b still unreachable via SSH

- **Track:** A Real SED NFNet crop/context tuning, PR #208 (`feature/nfnet-sed-context-sweep`).
- **Status checks:** Current best remains `0.927`. Latest scored submissions unchanged: v510 `0.927`, v511 `0.926`, v512 hidden runtime/no public score, v505/v506 `0.927`; v513/v514 remain complete with `submission.csv`. Queue monitor pid `99291` is alive and still sleeping after daily cap on v513.
- **v22b monitoring:** Retried GPU server collection for `sed-nfnet-balanced-oof-v22b-20s-160-100cls-lr1e4-ep5-gpu1retry`. Network is degraded: ping to `192.168.0.10` had packet loss (`1/3` replies) and SSH again reached TCP connect but timed out during banner/key exchange. No reliable read of v22b artifacts was possible this turn.
- **Action taken:** Did not launch `v23` fallback because v22b may still be running/completed and duplicating 20s NFNet OOF would waste scarce GPU. Kept PR #208 state as-is and documented the blocker. Next run should retry SSH first; if still blocked, consider a user-visible infrastructure note rather than stacking more GPU jobs blindly.

## 2026-05-07 19:41 UTC — Spec A v22b monitor blocked by GPU host outage

- **Track:** A Real SED NFNet crop/context tuning, PR #208 (`feature/nfnet-sed-context-sweep`).
- **Status checks:** Current best remains `0.927`. Latest scored submissions unchanged: v510 `0.927`, v511 `0.926`, v512 hidden runtime/no public score, v505/v506 `0.927`; v513/v514 remain complete with `submission.csv`. Queue monitor pid `99291` is alive and still sleeping after daily cap on v513.
- **Infrastructure status:** GPU server `192.168.0.10` is now worse than previous SSH banner timeouts: ping had `0/5` replies (`100%` packet loss), and SSH returned `Host is down`. No reliable remote read of v22b logs/artifacts was possible.
- **Decision:** Did not launch `v23` fallback or stack any new GPU jobs because the server is unreachable and v22b may already be running/completed. Launching blind would risk duplicate work once the host returns.
- **Next step:** When `192.168.0.10` is reachable again, first collect `artifacts/sed_oof/sed-nfnet-balanced-oof-v22b-20s-160-100cls-lr1e4-ep5-gpu1retry/oof_summary.json` and logs, compare against NFNet v13/v15, then decide whether to launch the prepared v23 20s/128-mel fallback.

## 2026-05-07 20:41 UTC — Spec E taxon-gate diagnostics while GPU host is down

- **Track:** E Non-bird/background/taxon gate preparation, pivoted because PR #208 v22b NFNet 20s OOF is blocked by GPU host outage.
- **Status checks:** Current best remains `0.927`. Latest scored submissions unchanged: v510 `0.927`, v511 `0.926`, v512 hidden runtime/no public score, v505/v506 `0.927`; v513/v514 remain complete with `submission.csv`. Queue monitor pid `99291` is alive and sleeping after daily cap on v513. Kaggle kernel status confirms v513/v514 are `COMPLETE` and both expose `submission.csv`.
- **Infrastructure:** GPU server `192.168.0.10` remains unreachable (`0/5` ping replies; SSH says `Host is down`), so no remote v22b collection or new GPU job was attempted.
- **Branch/PR:** Created branch `feature/taxon-gate-diagnostics` from `origin/main` for non-GPU progress.
- **Implementation:** Added `scripts/birdclef_taxon_gate_diagnostics.py`, a Spec E preparation script that reads `taxonomy.csv`, `train.csv`, `train_soundscapes_labels.csv`, and `sample_submission.csv`, maps species to taxonomy `class_name`, summarizes submission/train/soundscape taxon distributions, detects multi-label cross-group rows, and writes JSON artifacts for future gate training/postprocess experiments.
- **Validation:** `py_compile` passed. Ran locally against `/Volumes/ExternalSSD/data/workspace_don/kaggle_birdclef2026/data` and wrote ignored artifact `artifacts/gate/taxon_gate_diagnostics.json`.
- **Key diagnostics:** submission species by group: Amphibia `35`, Aves `162`, Insecta `28`, Mammalia `8`, Reptilia `1`. Train-audio rows are heavily bird-skewed: Aves `34,799`, Amphibia `451`, Insecta `199`, Mammalia `99`, Reptilia `1`. Train-soundscape rows are multi-label and often cross-taxon: top combos include Amphibia `866`, Amphibia+Aves `188`, Aves+Insecta `162`; cross-pairs include Amphibia+Aves `216`, Aves+Insecta `176`, Aves+Reptilia `12`. Label-count distribution is dense multi-label (no no-call rows in the label file; most rows have 3-6 labels).
- **Interpretation:** A taxon gate should be multi-output (not a single softmax) and conservative, with floors for rare groups. Reptilia has only one target species and essentially no training mass, so any gate multiplier there needs a high floor. Next useful Spec E step is a postprocess-only OOF simulation using existing v13/v15/v508 row predictions, not a fresh GPU model, once suitable OOF/test-row artifacts are available.

## 2026-05-07 21:41 UTC — Spec E taxon-gate OOF simulation rejects self-gating

- **Track:** E Non-bird/background/taxon gate, continued PR #209 (`feature/taxon-gate-diagnostics`) while GPU host remains down.
- **Status checks:** Current best remains `0.927`. Latest scored submissions unchanged: v510 `0.927`, v511 `0.926`, v512 hidden runtime/no public score, v505/v506 `0.927`; v513/v514 remain complete with `submission.csv`. Queue monitor pid `99291` remains alive and sleeping after daily cap on v513. v374/v375 also appear as complete `0.926` in the submission list; no new best.
- **Infrastructure:** GPU server `192.168.0.10` remains unreachable (`0/3` ping replies; SSH timeout/host-down), so no v22b collection or GPU launch was attempted.
- **Implementation:** Added `scripts/birdclef_taxon_gate_oof_sweep.py`, a postprocess-only OOF simulator that aligns one or more OOF NPZ files, builds taxonomy group indices from `taxonomy.csv`, derives row-level group presence from the predictions, applies conservative group multipliers (`floor + (1-floor)*presence^power`) with `max` or `topk_mean` group aggregation, and reports macro AUC deltas.
- **Validation:** `py_compile` passed. Ran sweeps locally on existing OOF artifacts: v13 NFNet, v15 NFNet, and the v13/v15 0.4/0.6 ensemble on their 1000-file overlap.
- **Results:** v13/v15 baseline OOF `0.657329` over 100 classes; best taxon self-gate was slightly worse: `0.657264` (`topk_mean`, floor `0.95`, power `2.0`, delta `-0.000066`). v13 baseline `0.636878`; best gate `0.636596` (delta `-0.000282`). v15 baseline `0.640274` over 181 classes; best gate `0.640145` (delta `-0.000130`).
- **Interpretation:** Self-gating from the same species predictions does not improve OOF; even very conservative taxon multipliers slightly hurt. Spec E remains plausible only with an independent gate signal (external/no-call/background model or metadata), not with a derived self-gate. Do not package taxon self-gating for Kaggle.

## 2026-05-07 22:41 UTC — Spec C external/focal pretrain manifest prep while GPU remains down

- **Track:** C External-data / target-species focal-audio pretraining preparation. Pivoted from Spec E self-gating because taxon self-gate OOF slightly hurt, and GPU host remains unreachable for v22b/NFNet work.
- **Status checks:** Current best remains `0.927`. Latest scored submissions unchanged: v510 `0.927`, v511 `0.926`, v512 hidden runtime/no public score, v505/v506 `0.927`; v374/v375 are complete `0.926`. v513/v514 remain complete with `submission.csv`. Queue monitor pid `99291` remains alive and sleeping after daily cap on v513.
- **Infrastructure:** GPU server `192.168.0.10` is still unreachable (`0/3` ping replies; SSH timed out), so no v22b collection or GPU launch was attempted.
- **Branch/PR:** Created branch `feature/external-pretrain-manifest` from `origin/main`.
- **Implementation:** Added `scripts/birdclef_external_pretrain_manifest.py`. It reads `train.csv`, `taxonomy.csv`, and `sample_submission.csv`, verifies target-taxonomy alignment and file existence, computes collection/class/rating imbalance, caps rows per species deterministically, assigns stable train/val folds, and emits manifest CSVs plus summary JSON for future Spec C pretraining/fine-tuning runs.
- **Validation:** `py_compile` passed. Ran two local manifest builds against `/Volumes/ExternalSSD/data/workspace_don/kaggle_birdclef2026/data`: `artifacts/external_pretrain/manifest_qall_cap120/` and `artifacts/external_pretrain/manifest_q3_cap80/`.
- **qall/cap120 manifest:** `17,709` rows (`14,215` train / `3,494` val), with class mix Amphibia `451`, Aves `17,020`, Insecta `138`, Mammalia `99`, Reptilia `1`; collection mix XC `12,152`, iNat `5,557`; no missing files. This is the broadest target-taxonomy aligned manifest and keeps non-bird/iNat coverage.
- **q3/cap80 quality manifest:** `11,565` rows (`9,271` train / `2,294` val), all XC, class mix Aves `11,487`, Amphibia `57`, Mammalia `21`, no Insecta/Reptilia. This is cleaner audio but badly drops non-bird coverage; use only as a bird-heavy pretraining baseline, not as the sole final fine-tune mix.
- **Interpretation:** Spec C should not naively filter to high-rating audio only; it erases most non-bird target diversity. A robust pretrain plan should use the broad qall manifest or a two-stage schedule: quality XC pretrain followed by broad target-taxonomy fine-tune/rebalancing with iNat/non-bird rows retained.

## 2026-05-07 23:41 UTC — Spec C manifest-backed SED smoke wiring

- **Track:** C External/focal pretraining prep, continued PR #210 (`feature/external-pretrain-manifest`) while GPU server remains unreachable.
- **Status checks:** Current best remains `0.927`. Latest scored submissions unchanged: v510 `0.927`, v511 `0.926`, v512 hidden runtime/no public score, v505/v506 `0.927`; v374/v375 complete `0.926`. Queue monitor pid `99291` remains alive and sleeping after daily cap on v513 at check time.
- **Infrastructure:** GPU server `192.168.0.10` still unreachable (`0/3` ping replies; SSH timeout), so no v22b collection or GPU launch was attempted.
- **Implementation:** Extended `scripts/birdclef_sed_pilot_train.py` with `selection_strategy="manifest"` and config fields `manifest_csv`, `manifest_path_column`, `manifest_label_column`, and `manifest_split`. The selector reads a manifest CSV, filters by split/labels, verifies file paths, shuffles deterministically, and caps to `max_files`.
- **Smoke config:** Added `configs/birdclef/sed_tiny_external_manifest_smoke.json`, pointing at the local broad manifest `artifacts/external_pretrain/manifest_qall_cap120/external_pretrain_manifest.csv` with `manifest_split="train"`, tiny CNN, 64 examples, 1 epoch.
- **Validation:** `py_compile` passed for the manifest builder and SED trainer. JSON config parsed. Local CPU smoke completed: `64` examples (`51` train / `13` val), input from manifest-backed train paths, tiny CNN TorchScript `0.184 MB`, runtime `6.618s`, macro AUC `0.5227` over 12 toy-valid classes. Artifact path: `artifacts/external_pretrain/sed-tiny-external-manifest-smoke/`.
- **Interpretation:** The external/focal manifest is now wired into the SED training harness and smoke-tested. Once GPU host returns, the next real Spec C step is a small B0/NFNet manifest-backed pilot using the broad manifest, likely with a quality-XC pretrain stage followed by broad target-taxonomy fine-tune.

## 2026-05-08 00:41 UTC — Spec B pseudo-label threshold/power sweep diagnostic

- **Track:** B Pseudo-label/noisy-student cache prep while GPU server remains unreachable.
- **Status checks:** Current best remains `0.927`. After UTC reset, queue monitor pid `99291` submitted the five queued candidates and is alive/sleeping after cap on v376. Pending submissions at check time: v513 real NFNet SED rank blend, v514 2-model rank blend, v507 SED-logit top3 EW0.60, v508 SED-logit top3 EW0.625/context0.275, v509 no-call gate. Completed references remain v510 `0.927`, v511 `0.926`, v512 complete/hidden, v505/v506 `0.927`.
- **Infrastructure:** GPU server `192.168.0.10` still blocked (`0/3` ping replies; SSH says `Host is down`), so no v22b collection or GPU training launch was attempted.
- **Repo state:** Prior external-manifest PR #210 was already merged upstream as combined PR `Combine PRs #206-#210`; created new branch `feature/pseudolabel-threshold-sweep` from `origin/main`.
- **Implementation:** Added `scripts/birdclef_pseudolabel_threshold_sweep.py`, a lightweight diagnostic for row-level teacher/student probability NPZs. It loads `row_ids`/`labels` and a probability key (`probs`, `pred_teacher`, `pred_student`, or `pred_oof`), reconstructs labeled soundscape truth, sweeps teacher power values and positive/negative hard-label thresholds, and reports macro AUC, top-k recall, threshold coverage, truth-cell recall, precision-vs-truth, masked fraction, and a conservative shortlist.
- **Validation commands:** `python -m py_compile scripts/birdclef_pseudolabel_threshold_sweep.py scripts/birdclef_pseudolabel_cache_summary.py` passed. Ran sweeps into ignored local artifacts under `artifacts/pseudolabels/threshold-sweep/`.
- **Key diagnostic results:** v508 teacher cache66 remains very strong on labeled soundscapes (`macro_auc=0.9911`, top5 recall `0.6437`). Conservative hard positives at `power=1.0`, `positive_threshold=0.98` produce 372 positive cells across 282 rows / 11 classes with apparent precision `1.0` but only `0.119` truth-cell recall. Soft student `pl-r1-b0-v508-soft-p100-lr3e4-mix02-ep20` is weaker but usable (`macro_auc=0.9600`, top5 recall `0.4745`); hard-conf best-val student is poor/over-broad (`macro_auc=0.6589`, top5 recall `0.3986`, shortlist precision only `0.15` despite high recall). Interpretation: next pseudo-label round should prefer the original v508 teacher or soft student distillation, not the hard-conf student as teacher; p0.98 positives are very precise but too sparse, so use them as hard anchors alongside soft labels rather than standalone training labels.
- **Next step:** Push PR for the diagnostic. When GPU returns, launch a small R1 student using teacher soft labels plus hard-anchor positives (`p>=0.98`) and broader negatives (`<=0.05`) rather than trusting the hard-conf student predictions.

## 2026-05-08 01:41 UTC — Spec B soft-anchor pseudo-label student launched on recovered GPU

- **Track:** B Pseudo-label/noisy-student training, following the threshold sweep from PR #211.
- **Status checks:** Current best remains `0.927`. New submissions after UTC reset scored/finished: v507 `0.927`, v508 `0.927`, v509 `0.927`; v514 2-model SED rank blend dropped to `0.924`; v513 completed but hidden/no public score. v510 remains `0.927`, v511 `0.926`, v512 hidden/no score, v505/v506 `0.927`. Queue monitor pid `99291` remains alive and sleeping after hitting daily cap on old v376.
- **Infrastructure:** GPU host `192.168.0.10` recovered: ping `3/3`, SSH OK, hostname `trainer`, both RTX 4090s idle. The SMB mount at `/mnt/mac_data` is stale/empty after recovery, so I staged train-soundscape data to the server-local runner copy instead of blocking on remount: `~/birdclef-2026/data/train_soundscapes` (`5.1G`) and `data/train_soundscapes_labels.csv`.
- **v22b collection:** Collected the previously blocked NFNet 20s v22b result from the GPU host. It completed with 3 folds / 1000 OOF rows: macro AUC `0.607845` over 100 valid classes; fold AUCs `0.654546`, `0.661929`, `0.674741`; TorchScript exports ~`89.871 MB` each. This is weaker than v13/v15-style 10s NFNet and should not be packaged without an OOF blend win.
- **Implementation:** Extended `scripts/birdclef_pseudolabel_student_train.py` with `target_mode="soft_anchor"`. It keeps full soft teacher targets (`probs ** teacher_power`) but overrides high-confidence positives to `1.0` and low-confidence negatives to `0.0`, with configurable weights (`soft_label_weight`, `anchor_positive_weight`, `anchor_negative_weight`). Also added mirror-path resolution so `/mnt/mac_data` configs run on Mac via `/Volumes/ExternalSSD/data`.
- **Configs:** Added `configs/birdclef/pl_r1_b0_v508_soft_anchor_p98n05_smoke.json` and `configs/birdclef/pl_r1_b0_v508_soft_anchor_p98n05_lr3e4_ep12.json`. Full run uses v508 teacher cache, EfficientNet-B0, 10s/160 mel, `teacher_power=1.0`, `positive_threshold=0.98`, `negative_threshold=0.05`, soft weight `1.0`, positive anchor weight `2.0`, negative anchor weight `1.0`, lr `3e-4`, 12 epochs, restore best by val AUC.
- **Validation:** Local CPU smoke passed with 256 rows / 1 epoch; timm unavailable locally so tiny-CNN fallback ran, wrote TorchScript `0.184 MB`, runtime `12.947s`, final all-student macro AUC `0.58288` on 42 valid classes; teacher on same subset `0.98474`. The smoke primarily validates data path fallback, target construction, and training loop.
- **GPU launch:** Copied updated scripts/configs to `trainer` and launched durable job pid `5630`, log `~/birdclef-2026/logs/pl_r1_b0_soft_anchor_p98n05_lr3e4_ep12_20260508T014753Z.log`, command `CUDA_VISIBLE_DEVICES=0 nohup ~/kaggle_envs/s6e3/bin/python scripts/birdclef_pseudolabel_student_train.py --config configs/birdclef/pl_r1_b0_v508_soft_anchor_p98n05_lr3e4_ep12.json`. Initial process check alive; GPU memory only warming at first 5s check. Next run should monitor metrics/log and compare against prior soft student (`macro_auc=0.9600`) and hard-conf student (`0.6589`) on labeled soundscapes.

### Completion update — soft-anchor pseudo-label student

- GPU job pid `5630` completed successfully in `32.561s` on CUDA / EfficientNet-B0. Artifact copied back locally under `artifacts/pseudolabels/students/pl-r1-b0-v508-soft-anchor-p98n05-lr3e4-ep12/`.
- Metrics: `n_rows=792`, train/val `634/158`, best epoch `12`, best val macro AUC `0.951178` over 61 valid val classes, final all-row macro AUC `0.950241` over 75 valid classes, teacher all-row macro AUC `0.991149`, student-teacher corr `0.934538`, MAE `0.049890`, TorchScript `15.391 MB`.
- Epoch trajectory improved steadily from val AUC `0.6743` at epoch 1 to `0.9512` at epoch 12; no overfit turn was visible by epoch 12.
- Follow-up threshold sweep on the new student (`artifacts/pseudolabels/threshold-sweep/soft_anchor_p98n05_ep12_student_sweep.json`) shows lower standalone utility than the prior soft student: macro AUC `0.950241`, top5 recall `0.432987`, conservative p1.3/p>=0.98 positives have precision `0.9716` but only `0.0439` truth-cell recall. Interpretation: soft-anchor training is stable and exportable but slightly underperforms the earlier soft-only B0 student (`0.9600` all-row macro AUC). Keep it as an ablation; next useful Spec B knob is likely longer training/EMA or mixing soft-only + anchor loss with lower anchor weight, not hard-conf teacher replacement.

## 2026-05-08 02:41 UTC — Spec B soft-label B0 power/epoch sweep

- **Track:** B Pseudo-label/noisy-student hyperparameter tuning after soft-anchor underperformed soft-only.
- **Status checks:** Current best remains `0.927`. Latest completed: v507 `0.927`, v508 `0.927`, v509 `0.927`, v514 `0.924`, v513 hidden/no public score, v510 `0.927`, v511 `0.926`, v512 hidden/no score, v505/v506 `0.927`. Queue monitor pid `99291` remains alive and sleeping after daily cap on old v376.
- **Infrastructure:** GPU host `192.168.0.10` is reachable and idle at start/end. Server-local soundscape copy from prior run remains available because `/mnt/mac_data` was stale after host recovery.
- **Hypothesis:** Since hard labels and soft-anchor hurt, the best pseudo-label path may simply be high-LR soft distillation from v508 with best-val checkpointing. Test whether (a) longer training with `teacher_power=1.0` improves the existing 20-epoch soft-only run, and (b) power-softened labels (`teacher_power=0.85`) trade a small AUC loss for better high-confidence coverage.
- **Configs added:** `configs/birdclef/pl_r1_b0_v508_soft_p100_lr1e3_nomix_ep30_bestval.json` and `configs/birdclef/pl_r1_b0_v508_soft_power085_lr1e3_nomix_ep20_bestval.json`. Both use EfficientNet-B0, v508 teacher cache, 10s/160 mel, lr `1e-3`, no mixup, soft BCE, restore-best-by-val-AUC. The p100 run trains 30 epochs; p085 trains 20 epochs.
- **Validation/launch:** JSON parsed and `py_compile` passed. Copied configs/scripts to `trainer` and launched two concurrent durable GPU jobs: p100/ep30 on GPU0 pid `22996`; p085/ep20 on GPU1 pid `22998`. Both completed successfully; artifacts copied back locally under `artifacts/pseudolabels/students/` and threshold sweeps written under `artifacts/pseudolabels/threshold-sweep/`.
- **p100/ep30 result:** best epoch `28`, best val macro AUC `0.987435` over 61 val classes, final all-row macro AUC `0.985348` over 75 classes, student-teacher corr `0.984719`, runtime `65.897s`, TorchScript `15.391 MB`. This improves the prior p100/ep20 final all-row macro AUC `0.983398` and is now the best B0 pseudo-label student artifact.
- **p085/ep20 result:** best epoch `18`, best val macro AUC `0.985011`, final all-row macro AUC `0.983310`, corr `0.979779`, runtime `59.855s`, TorchScript `15.391 MB`. Slightly weaker than p100/ep30, but its p>=0.98 hard-positive coverage is larger: 422 positive cells / 313 rows / 14 classes, precision `0.9976`, truth-cell recall `0.13485` vs p100/ep30's 275 cells / 209 rows / 14 classes, precision `1.0`, recall `0.08808`.
- **Interpretation:** For clean leaderboard-facing student signal, p100/ep30 best-val is the strongest pseudo-label student so far. For pseudo-label cache generation, p085 may be useful as a broader hard-anchor source, but not as the standalone student. Next useful step is package/evaluate p100/ep30 as a lightweight TorchScript sidecar or use it in an OOF/test blend grid; avoid more hard-conf replacements.

## 2026-05-08 03:41 UTC — Spec B pretrained B0 full fine-tune check

- **Track:** B/D pseudo-label student + model-zoo training. Followed up Wenhao's question about whether retraining is head-only over a pretrained Google model: current student training updates all `model.parameters()` (backbone + SED head), and recent configs were `pretrained=false`. This run explicitly tested a timm ImageNet-pretrained EfficientNet-B0 with full fine-tuning.
- **Status checks:** Current best remains `0.927`. Latest visible results unchanged: v507 `0.927`, v508 `0.927`, v509 `0.927`, v514 `0.924`, v513 hidden/no public score, v510 `0.927`, v511 `0.926`, v512 hidden/no public score, v505/v506 `0.927`. Queue monitor pid `99291` remains alive/sleeping on daily cap after old v376. GPU host `192.168.0.10` reachable and idle before/after.
- **Safety diagnostic before packaging:** On labeled soundscapes, blending the best p100/ep30 student with v508 teacher produced only tiny in-sample gain: linear blend best was weight `0.005`, macro AUC `0.9911698` vs teacher `0.9911494` (+`0.0000205`); rank blend best was weight `0.01`, macro AUC `0.9911577` (+`0.0000083`). This is too small/high-correlation to justify spending a submission slot yet.
- **Config added:** `configs/birdclef/pl_r1_b0_v508_soft_p100_pretrained_lr3e4_nomix_ep20_bestval.json`: EfficientNet-B0, `pretrained=true`, v508 teacher cache, 10s/160 mel, lr `3e-4`, no mixup, soft BCE, 20 epochs, restore-best-by-val-AUC. The training script already fine-tunes full backbone + head; no encoder freeze is applied.
- **Validation/launch:** JSON parsed and `py_compile` passed. Launched on `trainer` GPU0 pid `31747`; timm loaded pretrained weights with expected classifier/head key mismatches due feature-only adaptation. Full run completed successfully and artifacts copied back locally.
- **Result:** best epoch `19`, best val macro AUC `0.987221` over 61 val classes, final all-row macro AUC `0.984845` over 75 classes, student-teacher corr `0.982466`, MAE `0.027189`, runtime `27.432s`, TorchScript `15.391 MB`. Threshold sweep: top1 recall `0.23754`, top5 recall `0.63299`; conservative p0.75/p>=0.98 positives have 202 cells / 185 rows / 11 classes with precision `1.0` and truth-cell recall `0.0647`.
- **Interpretation:** ImageNet-pretrained full fine-tune is stable and learns faster, but it does not beat scratch p100/ep30 (`0.985348` all-row, best val `0.987435`). Keep the pretrained result as an ablation, but the strongest B0 pseudo-label artifact remains scratch p100/ep30. Do not push a student sidecar Kaggle kernel yet; the labeled-soundscape blend gain is too small and v514 showed that adding weak/highly-correlated SED sidecars can hurt LB.

## 2026-05-08 04:41 UTC — Spec D pseudo-label model-zoo ConvNeXt/V2S sweep

- **Track:** D Model zoo diversity + B pseudo-label/noisy-student training. Goal was to find a stronger/lower-correlation student than EfficientNet-B0 before spending Kaggle sidecar slots.
- **Status checks:** Current best remains `0.927`. Latest visible results unchanged: v507 `0.927`, v508 `0.927`, v509 `0.927`, v514 `0.924`, v513 hidden/no public score, v510 `0.927`, v511 `0.926`, v512 hidden/no public score, v505/v506 `0.927`. Queue monitor pid `99291` remains alive/sleeping on daily cap after old v376. GPU host `192.168.0.10` reachable and idle before/after.
- **Configs added:** `pl_r1_v2s_v508_soft_p100_lr3e4_nomix_smoke.json`, `pl_r1_v2s_v508_soft_p100_lr3e4_nomix_ep20_bestval.json`, `pl_r1_convnext_tiny_v508_soft_p100_lr3e4_nomix_smoke.json`, and `pl_r1_convnext_tiny_v508_soft_p100_lr3e4_nomix_ep20_bestval.json`. All use v508 teacher cache, 10s/160 mel, soft BCE, `teacher_power=1.0`, lr `3e-4`, no mixup, restore best by val AUC.
- **Smoke gate:** EfficientNetV2-RW-S smoke passed technically but looked bad after 1 epoch: final all-row macro AUC `0.565865`, teacher corr `-0.3933`, TorchScript `88.740 MB`; full V2S run was not launched. ConvNeXt-Tiny smoke passed and looked promising: final all-row macro AUC `0.820733` on 23 valid classes, corr `0.8889`, TorchScript `112.355 MB`, runtime `5.223s`.
- **Full ConvNeXt-Tiny run:** Launched durable GPU job on trainer pid `37087`, completed successfully. Final metrics: best epoch `19`, best val macro AUC `0.987823` over 61 val classes, final all-row macro AUC `0.986614` over 75 classes, student-teacher corr `0.987393`, runtime `71.432s`, TorchScript `112.355 MB`.
- **Threshold/blend diagnostics:** ConvNeXt threshold sweep: top5 recall `0.644230`, conservative p1.15/p>=0.95 positives 579 cells / 406 rows / 12 classes with precision `1.0` and truth-cell recall `0.18546`. However labeled-soundscape blend with v508 teacher still showed essentially no useful gain: linear blend best stayed at weight `0` (teacher only); rank blend best weight `0.005`, macro AUC `0.9911578` vs teacher `0.9911494` (+`0.0000084`).
- **Interpretation:** ConvNeXt-Tiny is now the strongest standalone pseudo-label student (`0.986614`, better than B0 p100/ep30 `0.985348`) and gives much broader precise hard-positive coverage, but it is highly correlated with v508 and not worth a Kaggle sidecar submission yet. Treat it as useful for pseudo-label cache expansion / future training, not immediate LB packaging. V2S scratch at lr3e-4 failed the smoke signal; revisit only with pretrained or different LR if needed.

## 2026-05-08 05:41 UTC — Spec C external-data B0 manifest pretrain pilot

- **Track:** C External-data pretraining on target species, with a small D/backbone-reuse component. Hypothesis: q>=3 Xeno-Canto target-species clips can initialize a lightweight B0 SED/clip student for later BirdCLEF 2026 fine-tuning, but the manifest selection needs class-aware balancing before larger GPU time.
- **Status checks:** Current best remains `0.927`. Latest visible submissions: v509 `0.927`, v508 `0.927`, v507 `0.927`, v514 `0.924`, v513 complete hidden/no public score, v506/v505 `0.927`, v512 hidden, v511 `0.926`, v510 `0.927`. Queue monitor pid `99291` is alive and sleeping on daily cap after attempting old v376. v510 is complete/submitted/scored, so no v510 failure fix was needed. GPU host `192.168.0.10` reachable; both RTX 4090s idle before/after.
- **Branch/PR:** Created branch `feature/external-pretrain-b0-pilot` from updated `origin/main` (after PR #210 merge). PR pending creation in this run.
- **Implementation:** Made `scripts/birdclef_sed_pilot_train.py` manifest paths portable across Mac/SMB/GPU-local mirrors via `resolve_manifest_audio_path()`, added configurable `track`, and added manifest class-selection knobs `manifest_max_files_per_class` and `manifest_min_files_per_class`.
- **Configs added:** `configs/birdclef/xc_b0_q3_cap80_external_pretrain_smoke.json` (B0 ImageNet-pretrained, q>=3 XC manifest, 128 files, 1 epoch), `configs/birdclef/xc_b0_q3_cap80_external_pretrain_pilot_ep4.json` (same, 768 random-staged files, 4 epochs), and `configs/birdclef/xc_b0_q3_cap80_external_pretrain_balanced_ep6.json` (manifest per-class cap 6, min 4, max 1024, 6 epochs). All use 5s/32k/128 mel, focal BCE gamma 1.5, label smoothing 0.005, pos_weight_sqrt, no mixup.
- **Data staging:** `/mnt/mac_data` is still effectively unavailable after GPU host recovery, so staged a deterministic subset of q3 manifest audio into trainer-local `~/birdclef-2026/data/train_audio`: first random staging set 900 rows / 163 labels, then balanced staging set 976 rows / 163 labels. Also staged `taxonomy.csv` and manifest artifacts under `artifacts/external_pretrain/manifest_q3_cap80/`.
- **Validation/smoke:** py_compile and JSON parse passed on Mac and trainer. B0 smoke completed on GPU: 128 examples (102 train / 26 val), 234 target classes, valid-class macro AUC `0.643700` over 25 classes, runtime `4.343s`, TorchScript `15.389 MB`.
- **Pilot 1 (random staged):** 768 examples (614 train / 154 val), macro AUC `0.516500` over 97 valid classes after 4 epochs, runtime `16.297s`, TorchScript `15.389 MB`. Loss decreased (`0.64`→`0.25`) but class support was too sparse and validation AUC was weak.
- **Pilot 2 (class-balanced cap/min):** 976 examples (781 train / 195 val), manifest cap 6 / min 4, macro AUC `0.588165` over 122 valid classes after 6 epochs, runtime `20.588s`, TorchScript `15.387 MB`. This recovered some signal vs random staging but is still not a useful standalone model.
- **Interpretation:** The Spec C data path is now operational and portable, and the new class-aware manifest knobs are necessary. However, tiny external-only B0 pretraining underfits/over-fragments across 234 classes at this scale. Next useful step is not Kaggle packaging; it is to add a two-stage fine-tune path (load external checkpoint then fine-tune on 2026 train/soundscape labels or pseudo-label cache) and/or run a larger class-balanced external pretrain with fewer target classes per fold/longer epochs. Do not spend LB submissions on these artifacts yet.

## 2026-05-08 06:41 UTC — Spec C→B external-init pseudo-label fine-tune

- **Track:** C External-data pretraining + B pseudo-label/noisy-student fine-tune. Hypothesis: the q>=3 XC class-balanced external B0 checkpoint can initialize a 2026 pseudo-label student and produce a less-correlated CNN signal than ImageNet-only/scratch B0.
- **Status checks:** Current best remains `0.927`. Latest visible submissions unchanged: v509/v508/v507 `0.927`, v514 `0.924`, v513 hidden complete, v506/v505 `0.927`, v512 hidden complete, v511 `0.926`, v510 `0.927`. Queue monitor pid `99291` alive and sleeping on daily cap after old v376; no duplicate submissions made. v510 already complete/submitted/scored, so no failure diagnosis required.
- **Branch/PR:** Continued PR #212 `feature/external-pretrain-b0-pilot`.
- **Implementation:** Extended `scripts/birdclef_pseudolabel_student_train.py` with optional `initial_checkpoint` loading from TorchScript/torch state dict. It supports encoder-only initialization via `initial_load_head=false`, skips mismatched/missing keys, and records a load summary in metrics. This lets external-pretrained SED checkpoints seed pseudo-label students without forcing the weak external classifier head.
- **Configs added:** `pl_r1_b0_v508_soft_p100_xc_extinit_lr1e4_nomix_smoke.json`, `pl_r1_b0_v508_soft_p100_xc_extinit_lr1e4_nomix_ep20_bestval.json`, `pl_r1_b0_v508_soft_p100_xc_extinit_lr3e4_nomix_ep20_bestval.json`, and `pl_r1_b0_v508_soft_p100_xc_extinit_lr3e4_nomix_ep30_bestval.json`. All use v508 teacher cache, EfficientNet-B0, external balanced ep6 TorchScript encoder init (`initial_load_head=false`), soft labels p100, no mixup, restore best by val AUC, 10s/160 mel.
- **Validation:** py_compile passed locally and on trainer. Smoke with lr1e-4 / 256 rows / 3 epochs loaded 352 encoder keys and skipped 2 head keys; final all-row macro AUC `0.907135` over 42 valid classes, corr `0.6972`, runtime `4.344s`.
- **Full lr1e-4:** 20 epochs, best epoch 19, best val macro AUC `0.987008`, final all-row macro AUC `0.983614`, corr `0.980269`, runtime `35.361s`. Labeled-soundscape blend with v508: linear best teacher-only; rank best weight `0.01`, AUC `0.9911637` vs teacher `0.9911494` (+`0.0000144`). Too slow/underfit at lr1e-4.
- **Full lr3e-4:** 20 epochs, best epoch 20, best val macro AUC `0.990383`, final all-row macro AUC `0.986924`, corr `0.985633`, runtime `39.680s`. This beats previous B0 p100/ep30 (`0.985348`) and slightly beats ConvNeXt-Tiny pseudo-label student (`0.986614`) as the strongest standalone student so far, but remains high-correlation.
- **Full lr3e-4 ep30:** best epoch remained 20, final all-row macro AUC `0.986927`, corr `0.985687`, runtime `60.597s`. Blend with v508: linear best teacher-only; rank best weight `0.03`, AUC `0.9911725` vs teacher `0.9911494` (+`0.0000231`). This is the best local rank-blend gain from the student family so far, but still too small to justify a Kaggle submission by itself.
- **Interpretation:** External-pretrained encoder init is genuinely helpful for the pseudo-label student (new best standalone B0/student), but direct v508 blend remains only microscopic. Next step should be a stronger low-correlation use: package/evaluate a student ensemble only after OOF-safe blend improves more, or train a different architecture (e.g. ConvNeXt/V2S) initialized/regularized with external data rather than submitting this B0 sidecar directly.

## 2026-05-08 07:41 UTC — Spec B noisy-student real-clip mixing probe from 2024/2025 writeups

- **Track:** B Pseudo-label/noisy-student + C external-init. Hypothesis from BirdCLEF 2024/2025 writeups: second-level students should train on original labeled clips plus pseudo-labeled soundscape windows, using larger batches and controlled pseudo/real ratios. This run tests whether adding real train-audio clips improves the current external-init B0 pseudo-label student.
- **Status checks:** Current best remains `0.927`. Latest visible submissions unchanged: v509/v508/v507 `0.927`, v514 `0.924`, v513 hidden complete, v506/v505 `0.927`, v512 hidden complete, v511 `0.926`, v510 `0.927`. Queue monitor pid `99291` alive and sleeping on daily cap; no duplicate submissions. v510 already complete/submitted/scored.
- **Branch/PR:** Continued PR #212 `feature/external-pretrain-b0-pilot`.
- **Implementation:** Extended `scripts/birdclef_pseudolabel_student_train.py` with optional supervised clip mixing from `train.csv`: `supervised_csv`, `supervised_data_root`, path/label/secondary columns, max files, per-class cap/min, `supervised_weight`, and `supervised_label_smoothing`. Supervised clips are decoded to the same 10s/160-mel tensor shape and appended to the training pool while evaluation remains on held-out pseudo-labeled soundscape rows. Added weighted BCE so supervised clips can be downweighted.
- **Configs added:** `pl_r2_b0_v508_xc_extinit_soft_plus_realclip_w05_smoke.json`, `pl_r2_b0_v508_xc_extinit_soft_plus_realclip_w01_smoke.json`, `pl_r2_b0_v508_xc_extinit_soft_plus_realclip_w05_ep20_bestval.json`, and `pl_r2_b0_v508_xc_extinit_soft_plus_realclip_w01_ep20_bestval.json`. All use external-init B0, v508 soft pseudo labels, batch size 64, supervised per-class cap 4, label smoothing 0.001; w05/w01 set supervised loss weights 0.5/0.1.
- **Data staging:** Staged deterministic balanced train-audio subset from `train.csv` on trainer: 768 rows / 206 labels, class mix Aves 620, Amphibia 111, Mammalia 24, Insecta 12, Reptilia 1. Trainer now has `data/train.csv` and 2507 staged `data/train_audio` files total.
- **Validation:** py_compile passed locally/on trainer. Smoke w0.5 used 120/128 requested supervised clips, final all-row macro AUC `0.805337` over 42 valid classes, corr `0.8400`, runtime `6.915s` — clear degradation vs no-realclip smoke (`0.907135`). Smoke w0.1 used 120 supervised clips, final all-row macro AUC `0.882399`, corr `0.8728`, runtime `6.707s` — better than w0.5 but still below no-realclip smoke.
- **Full w0.1:** 792 pseudo rows + 739/768 supervised clips, 20 epochs, best epoch 19, best val macro AUC `0.983588`, final all-row macro AUC `0.980850`, corr `0.975976`, runtime `77.578s`. Blend with v508: linear and rank both best at weight 0 (teacher only). This is much worse than external-init B0 without real clips (`0.986927`, rank blend +`0.0000231`).
- **Interpretation:** Naively mixing full-recording train-audio primary-label clips into the soundscape pseudo-label student hurts badly despite the blog-derived recipe. Likely causes: train_audio domain/label noise, long/full-recording weak labels conflicting with 10s endpoint soundscape targets, and no proper crop policy/large-batch ratio scheduling. Kill this naive realclip-mixing variant. If revisited, use 5s random crops, larger true batch/gradient accumulation, stricter XC quality filtering, or pseudo-label the train_audio clips with the v508 teacher instead of trusting primary labels directly.

## 2026-05-08 08:41 UTC — Spec E taxon max gate candidate v516

- **Track:** E Background/non-bird/taxon gate, motivated by 2025 winner's separate insect/amphibian handling and by the previous run's real-clip mixing failure. Hypothesis: row-level evidence for broad taxon groups can suppress classes from unsupported taxa without changing the core v508 ensemble.
- **Status checks:** Current best remains `0.927`. Latest visible submissions unchanged: v509/v508/v507 `0.927`, v514 `0.924`, v513 hidden complete, v506/v505 `0.927`, v512 hidden complete, v511 `0.926`, v510 `0.927`. v510 complete/submitted/scored; no failure fix needed. GPU host reachable and idle.
- **Local diagnostic:** On v508 teacher cache (`792 x 234` labeled train-soundscape rows), swept row-wise taxon gates using taxonomy class groups. Best local result used group max evidence with `floor=0.30`, `alpha=0.75`: macro AUC `0.992162` vs v508 teacher `0.991149` (+`0.001013`). Top-5 mean evidence also helped (+`0.000569`), while sum evidence did not. This is much larger than the recent student blend gains (~+0.00002), so worth a real Kaggle candidate despite overfit risk.
- **Implementation:** Created Kaggle kernel folder `kaggle-kernels/v516-taxon-max-gate`, copied from v508 axis, and added `taxon_max_gate()`. It computes max probability per taxon group (`Aves`, `Amphibia`, `Insecta`, `Mammalia`, `Reptilia`) from the final shaped probabilities, then multiplies each label by `max(floor, group_evidence)^alpha`, clipped to valid probability bounds. Constants: `TAXON_MAX_GATE_FLOOR=0.30`, `TAXON_MAX_GATE_ALPHA=0.75`.
- **Push/monitor:** Added `scripts/push_v516.py`, pushed real Kaggle kernel `yourslewis/bc26-v516-taxon-max-gate`, version 1; push response had no invalid sources and kernel status after push was RUNNING with no failure message. Inserted v516 into `scripts/submit_pending_birdclef_queue.py` before old v376. Refreshed queue monitor: killed old pid `99291`, started pid `94182`, log `logs/submit_pending_birdclef_queue_20260508T084638Z.log`; monitor skips already-submitted v505-v514/v509, sees v516 RUNNING, and sleeps 10 minutes without duplicating submissions.
- **Validation:** `python3 -m py_compile kaggle-kernels/v516-taxon-max-gate/script.py` passed. Local taxon gate sweep used the same taxonomy-driven class grouping and confirmed the selected constants. Await Kaggle kernel completion and daily submission slot.
- **Interpretation:** This is the first recent non-training/non-student candidate with a meaningful local labeled-soundscape gain. It may overfit local train soundscapes, but the effect size is large enough to spend one Kaggle candidate slot once complete. If v516 scores below 0.927, next taxon-gate variants should be less aggressive (`alpha=0.5` or mean_top5/floor0.3) rather than more realclip mixing.

## 2026-05-08 09:41 UTC — Spec E softer taxon gate backup v517

- **Track:** E Background/non-bird/taxon gate. v516 completed successfully but is blocked behind the daily submission cap, so this run prepared a softer backup candidate rather than spending more time on real-clip primary-label mixing, which previous smokes killed.
- **Status checks:** Current best remains `0.927`. Latest visible submissions unchanged: v509/v508/v507 `0.927`, v514 `0.924`, v513 hidden complete, v506/v505 `0.927`, v512 hidden complete, v511 `0.926`, v510 `0.927`. Queue monitor was alive. v516 kernel status is COMPLETE with no failure message, but submission attempt hit daily cap (`14 hours from now`). GPU host reachable; active GPU work belongs to unrelated LRM jobs, no BirdCLEF GPU jobs.
- **Local diagnostic:** Rechecked v508 teacher-cache taxon gate variants on labeled train-soundscape rows. v516 (`max`, floor `0.30`, alpha `0.75`) remains best local: `0.992162` vs baseline `0.991149` (+`0.001013`). Softer max gate floor `0.30`, alpha `0.50` scored `0.992008` (+`0.000859`). mean_top5 floor `0.30`, alpha `0.50` scored `0.991718` (+`0.000569`).
- **Implementation:** Created `kaggle-kernels/v517-taxon-max-gate-alpha050` by copying v516 and changing only `TAXON_MAX_GATE_ALPHA=0.50` while keeping `TAXON_MAX_GATE_FLOOR=0.30`. Added `scripts/push_v517.py`.
- **Push/monitor:** Pushed real Kaggle kernel `yourslewis/bc26-v517-taxon-max-gate-a050`, version 1; response had no invalid sources. Status after push: RUNNING, no failure message. Inserted v517 into `scripts/submit_pending_birdclef_queue.py` immediately after v516 and before old v376. Refreshed monitor: killed old pid `94182`, started pid `3604`, log `logs/submit_pending_birdclef_queue_20260508T094350Z.log`. It sees v516 COMPLETE and is sleeping on daily cap; after reset it should submit v516 first, then v517 when complete.
- **Validation:** `python3 -m py_compile` passed for v517 script and queue script. v516 had already completed on Kaggle, verifying the base taxon-gate kernel path is runnable.
- **Interpretation:** v517 gives a safer/less aggressive alternative if v516 over-suppresses taxa on public LB. Do not add more taxon variants until v516/v517 score; next high-upside work should return to true multi-round pseudo-labeling or taxon-specific model branches.

## 2026-05-08 10:41 UTC — Spec B SoftAUC pseudo-label student smoke

- **Track:** B Pseudo-label/noisy-student loss tuning. Hypothesis from BirdCLEF 2025 4th-place notes: directly optimizing an AUC surrogate may improve ranking over pure BCE for pseudo-label students.
- **Status checks:** Current best remains `0.927`. Latest visible submissions unchanged: v509/v508/v507 `0.927`, v514 `0.924`, v513 hidden complete, v506/v505 `0.927`, v512 hidden complete, v511 `0.926`, v510 `0.927`. v516 and v517 are both COMPLETE with no failure messages; monitor pid `3604` is sleeping on daily cap before v516 submission. GPU host reachable; GPU1 is occupied by unrelated LRM work, GPU0 available and used for smokes.
- **Implementation:** Added `loss_name`, `auc_loss_weight`, and `soft_auc_scale` to `scripts/birdclef_pseudolabel_student_train.py`. Implemented `soft_auc_pairwise_loss()`, a differentiable soft-label macro AUC surrogate using target values as positive weights and `(1-target)` as negative weights, combined as `BCE + auc_loss_weight * SoftAUC` when `loss_name="bce_soft_auc"`.
- **Configs added:** `pl_r2_b0_v508_xc_extinit_softauc_w005_smoke.json`, `pl_r2_b0_v508_xc_extinit_softauc_w005_ep20_bestval.json`, and `pl_r2_b0_v508_xc_extinit_softauc_w0005_smoke.json`. All use external-init B0, v508 soft pseudo labels, no real clips, 10s/160 mel, batch size 64, restore best by val AUC.
- **Validation:** py_compile passed locally and on trainer. Smoke w0.05 / 256 rows / 3 epochs: best val AUC `0.882004`, final all-row macro AUC `0.854890`, corr `0.5477`, runtime `4.715s`. Smoke w0.005: best val AUC `0.901445`, final all-row macro AUC `0.878482`, corr `0.5792`, runtime `4.867s`.
- **Interpretation:** SoftAUC surrogate in this form hurts badly versus the external-init BCE smoke (`0.907135` final all-row / `0.917617` val at 3 epochs) and much worse than full BCE external-init (`0.986927`). Kill SoftAUC as implemented; do not launch full w0.05/w0.005. If revisited, use hard/clean labels or an OOF teacher, not soft in-sample pseudo probabilities.

## 2026-05-08 11:01 UTC — Spec B 5s external-init pseudo-label student + v518 Kaggle bridge

- **Track:** B Pseudo-label/noisy-student crop-length retuning after killing SoftAUC. Hypothesis: the external-pretrained B0 student may train better when pseudo-label windows match the competition 5s rows instead of using 10s context, while still providing a lightweight new model artifact for v508 blending.
- **Status checks:** Current best remains `0.927`. Latest visible submissions unchanged: v509/v508/v507 `0.927`, v514 `0.924`, v513 hidden complete, v506/v505 `0.927`, v512 hidden complete, v511 `0.926`, v510 `0.927`. v510/v516/v517 are COMPLETE with no failure messages. Monitor was refreshed to pid `13479`, log `logs/submit_pending_birdclef_queue_20260508T105813Z.log`; it stops at v516 because the daily cap has ~13h remaining, so v516 -> v517 -> v518 are queued without duplicate submissions.
- **Smoke sweep:** Added four external-init B0 smoke configs using v508 soft labels, lr3e-4, 256 rows, 3 epochs, best-val checkpointing: baseline 10s/no-mix (`final_all_auc=0.927522`, val `0.932025`), 5s/no-mix (`final_all_auc=0.939150`, val `0.951928`), power0.85 (`final_all_auc=0.934179`, val `0.937250`), mixup0.1 (`final_all_auc=0.921826`, val `0.924711`). 5s context is the clear smoke winner.
- **Scaled run:** Launched and completed full `pl-r2-b0-v508-xc-extinit-5s-lr3e4-ep20-bestval` on trainer GPU0. Config: EfficientNet-B0, external XC pretrain init, v508 teacher cache, 5s/160 mel, lr3e-4, batch 16, 20 epochs, no mixup, teacher_power=1.0, restore best by val AUC. Runtime `21.653s`; best epoch 19; best val AUC `0.989095`; final all-row macro AUC `0.987695` over 75 valid classes; teacher corr `0.98549`. This beats the previous best pseudo-label student (`0.986927` from 10s external-init ep30) by about `+0.000768` standalone.
- **Blend sweep on labeled soundscapes:** v508 teacher baseline `0.991149`; 5s student standalone `0.987695`; correlation `0.98549`. Best linear blend was tiny: weight `0.02` -> `0.991156` (`+0.0000069`). Best rank blend was weight `0.10` -> `0.991172` (`+0.0000226`), but recent real-SED rankblend underperformed publicly, so the safer Kaggle bridge uses conservative linear `0.02`.
- **Kaggle bridge:** Created private Kaggle dataset `yourslewis/bc26-student-b0-5s-v1` containing `bc26-student-b0-5s-v1.zip` with `model_torchscript.pt` and `sed_bundle_manifest.json` (~13.4MB zip). Created and pushed real kernel `yourslewis/bc26-v518-student-b0-5s-blend-002`, version 1; push returned no invalid data/kernel/model sources. v518 copies the v510 TorchScript-bundle loading path but points at the lightweight B0 5s dataset, uses one model, and blends after v508 probability shaping with `REAL_SED_BLEND_WEIGHT=0.02`. Kernel status after push: RUNNING, no failure message.
- **Interpretation / next:** 5s external-init student is a real improvement in the pseudo-label-student family and cheap enough for Kaggle CPU. Await v518 kernel completion, then daily-cap submission after v516/v517. If v518 ties/helps LB, consider a rankblend or 5s+10s student ensemble; if it falls, keep 5s student as local artifact but do not spend more daily slots on high-correlation B0 blend variants.

## 2026-05-08 12:10 UTC — Spec B/D 5s student mel/hop + ConvNeXt smoke

- **Track:** B pseudo-label/noisy-student hyperparameter tuning around the new 5s external-init B0 student, plus a small D model-zoo smoke. Hypothesis: after 5s beat 10s, mel resolution or hop size might further improve the student; ConvNeXt-Tiny 5s might provide a less-correlated model-zoo alternative.
- **Status checks:** Current best remains `0.927`. Latest visible submissions unchanged: v509/v508/v507 `0.927`, v514 `0.924`, v513 hidden complete, v506/v505 `0.927`, v512 hidden complete, v511 `0.926`, v510 `0.927`. v510/v516/v517/v518 are all COMPLETE with no failure messages. Queue monitor pid `13479` is alive and sleeping on daily cap at v516, then should submit v516 -> v517 -> v518; no duplicate submissions added.
- **B0 5s smoke sweep:** Added and ran three 256-row/3-epoch smoke configs on trainer GPU0, all external-init B0, v508 soft labels, lr3e-4, no mixup, best-val restore. Baseline from previous run was 5s/160mel/hop512 final `0.939150`, val `0.951928`. New results: 5s/128mel/hop512 final `0.936330`, val `0.946321`; 5s/224mel/hop512 final `0.941508`, val `0.952209`; 5s/160mel/hop320 final `0.937033`, val `0.948189`. Smoke winner was 224 mel.
- **Scaled 224-mel run:** Added and ran `pl-r2-b0-v508-xc-extinit-5s-m224-lr3e4-ep20-bestval`. Config: EfficientNet-B0, external XC pretrain init, v508 teacher cache, 5s/224 mel/hop512, lr3e-4, batch 16, 20 epochs, no mixup, teacher_power=1.0, best-val checkpoint. Runtime `24.667s`; best epoch 19; best val AUC `0.989453`; final all-row macro AUC `0.986159`; teacher corr `0.98578`. Despite the better smoke, full 224-mel underperformed the 160-mel full run (`0.987695`).
- **Blend check:** m224 full has no useful linear blend with v508 (best linear weight `0`, delta `0`); best rank blend weight `0.01` only gives `+0.0000018`. Combining m160+m224 students still picks m160 only (linear `w160=0.02,w224=0`; rank `w160=0.10,w224=0`). Do not package a 224-mel Kaggle variant.
- **ConvNeXt smoke:** Added and ran `pl-r2-convnext-tiny-v508-soft-p100-5s-lr3e4-smoke`: ConvNeXt-Tiny, no external init, v508 soft labels, 5s/160 mel, batch 8, 256 rows, 3 epochs. Result: best val `0.923619`, final all-row `0.919453`, corr `0.93692`, runtime `8.728s`, TorchScript `112.354MB`. This is far below B0 5s smokes and not worth scaling/packaging in this form.
- **Interpretation / next:** Keep v518 as the only queued 5s-student Kaggle bridge. Kill m128, m224, hop320, and ConvNeXt-5s variants unless new evidence appears. Next meaningful pseudo-label direction should be a less-correlated teacher/source (OOF teacher or external/test-style audio), not more high-correlation B0 spectrogram geometry tweaks.

## 2026-05-08 13:05 UTC — Spec C+B ep12 external pretrain init + v519 Kaggle bridge

- **Track:** C External-data pretraining + B pseudo-label/noisy-student. Hypothesis: the 5s B0 pseudo-label student was limited by a shallow 6-epoch XC initialization; a stronger target-species external pretrain could improve the student and create a better low-cost Kaggle bridge.
- **Status checks:** Current best remains `0.927`. Latest visible submissions unchanged: v509/v508/v507 `0.927`, v514 `0.924`, v513 hidden complete, v506/v505 `0.927`, v512 hidden complete, v511 `0.926`, v510 `0.927`. v510/v516/v517/v518 are COMPLETE with no failure messages. Refreshed queue monitor to pid `29481`, log `logs/submit_pending_birdclef_queue_20260508T130022Z.log`; it is sleeping on daily cap before v516, then should submit v516 -> v517 -> v518 -> v519.
- **Pretrain smoke:** Added/reran `xc-b0-q3-cap80-external-pretrain-m160-smoke` to test matching downstream 160 mel. It passed technically but underperformed prior 128-mel smoke: AUC `0.519200` over 23 valid classes vs old 128-mel smoke `0.643700`; killed m160 external pretrain.
- **Stronger external init:** Added/reran `xc-b0-q3-cap80-external-pretrain-balanced-ep12`: EfficientNet-B0, ImageNet/timm init, XC q>=3 cap80 manifest, 5s/128mel/hop512, focal BCE gamma1.5, label_smoothing 0.005, pos_weight_sqrt, 1024 max files, 12 epochs. Runtime `32.525s`; validation macro AUC `0.717673` over 122 valid classes, much better than the previous ep6 init `0.588165`. Exported TorchScript `15.387MB`.
- **Downstream smoke:** Added `pl-r2-b0-v508-xc-ep12init-5s-lr3e4-smoke`: ep12 external checkpoint, v508 soft labels, 5s/160mel, lr3e-4, 256 rows, 3 epochs. Result best val `0.959443`, final all-row `0.948186`, corr `0.90531`, runtime `3.776s`. This beat the previous ep6-init 5s smoke (`0.939150` final, `0.951928` val), so scaled.
- **Scaled student:** Added/reran `pl-r2-b0-v508-xc-ep12init-5s-lr3e4-ep20-bestval`: same but all 792 rows / 20 epochs. Runtime `21.699s`; best epoch 20; best val AUC `0.988928`; final all-row macro AUC `0.988494` over 75 valid classes; teacher corr `0.98677`. This beats previous best ep6-init 5s student `0.987695` by `+0.000799` standalone.
- **Blend sweep:** v508 teacher baseline `0.991149`. ep12-init 5s student: best linear blend weight `0.15` -> `0.991374` (`+0.000224`), best rank blend weight `0.075` -> `0.991215` (`+0.000066`). ep6+ep12 ensemble was weaker than ep12 alone (best linear w6=0.02/w12=0.10 -> `0.991285`). Use ep12-only linear blend 0.15.
- **Kaggle bridge:** Created private dataset `yourslewis/bc26-student-b0-5s-ep12init-v1` containing `bc26-student-b0-5s-ep12init-v1.zip` with TorchScript model + manifest (~13.4MB). Pushed real kernel `yourslewis/bc26-v519-student-b0-5s-ep12-init-blend-015`, version 1; no invalid sources. v519 copies v518 bundle path, points at ep12-init student dataset, uses one model, and blends after v508 probability shaping with `REAL_SED_BLEND_WEIGHT=0.15`. Kernel status after push: RUNNING, no failure message.
- **Interpretation / next:** v519 is the strongest local labeled-soundscape student bridge so far and worth a daily slot after v516-v518. If v519 ties/improves LB, consider ep12-init 5s student as the new student baseline. If it falls, the local blend gain is likely leakage from v508 teacher/probe labels; next direction should be clean OOF teacher generation rather than more student blend variants.

## 2026-05-08 14:05 UTC — Spec C+B ep18 external pretrain overfit check

- **Track:** C external pretraining depth + B pseudo-label/noisy-student. Hypothesis: since ep12 XC pretrain improved the 5s student and v519 local blend, pushing the same external pretrain longer might further improve downstream student initialization.
- **Status checks:** Current best remains `0.927`. Latest visible submissions unchanged: v509/v508/v507 `0.927`, v514 `0.924`, v513 hidden complete, v506/v505 `0.927`, v512 hidden complete, v511 `0.926`, v510 `0.927`. v510/v516/v517/v518/v519 are COMPLETE with no failure messages. Queue monitor pid `29481` alive, log `logs/submit_pending_birdclef_queue_20260508T130022Z.log`, sleeping on daily cap before v516; no duplicate submissions added.
- **External pretrain:** Added and ran `xc-b0-q3-cap80-external-pretrain-balanced-ep18`: same as ep12 (EfficientNet-B0, ImageNet/timm init, XC q>=3 cap80 manifest, 5s/128mel/hop512, focal BCE gamma1.5, label_smoothing 0.005, pos_weight_sqrt, 1024 max files) but 18 epochs. Runtime `36.579s`; validation macro AUC `0.743267` over 122 valid classes vs ep12 `0.717673`. Val loss bottomed around epoch 13 then degraded by epoch 18, so the final checkpoint likely overfits despite higher AUC.
- **Downstream smoke:** Added `pl-r2-b0-v508-xc-ep18init-5s-lr3e4-smoke`: ep18 checkpoint, v508 soft labels, 5s/160mel, lr3e-4, 256 rows, 3 epochs. Result best val `0.959678`, final all-row `0.949731`, corr `0.91751`, runtime `3.920s`. This slightly beat ep12 smoke (`0.948186` final, `0.959443` val), so scaled.
- **Scaled student:** Added and ran `pl-r2-b0-v508-xc-ep18init-5s-lr3e4-ep20-bestval`: all 792 rows / 20 epochs. Runtime `21.471s`; best epoch 19; best val AUC `0.989365`; final all-row macro AUC `0.987920` over 75 valid classes; teacher corr `0.98747`. This underperformed ep12-init full student (`0.988494`) and only slightly beat ep6-init (`0.987695`).
- **Blend sweep:** v508 teacher baseline `0.991149`. ep18-init best linear blend weight `0.10` -> `0.991190` (`+0.000041`); best rank blend weight `0.03` -> `0.991182` (`+0.000032`). Combining ep12+ep18 still picked ep12 alone (linear w12=0.15,w18=0; rank w12=0.075,w18=0). Do **not** package ep18 or create v520.
- **Interpretation / next:** ep18 external pretrain improves external holdout AUC but hurts downstream pseudo-label distillation versus ep12, likely due to overfitting / mismatched final checkpoint. Keep v519/ep12 as the student bridge. If revisiting external depth, implement best-checkpoint export in `birdclef_sed_pilot_train.py` and test epoch-13/val-loss restore rather than training longer final checkpoints.

## 2026-05-08 15:05 UTC — Spec C best-val-loss checkpoint export test

- **Track:** C external pretraining checkpoint selection + B downstream pseudo-label smoke. Hypothesis from previous run: ep18 final checkpoint overfit; restoring the best external-pretrain val-loss checkpoint around epoch 13 might improve downstream student initialization without spending a Kaggle slot.
- **Status checks:** Current best remains `0.927`. Latest visible submissions unchanged: v509/v508/v507 `0.927`, v514 `0.924`, v513 hidden complete, v506/v505 `0.927`, v512 hidden complete, v511 `0.926`, v510 `0.927`. v510/v516/v517/v518/v519 are COMPLETE with no failure messages. Queue monitor pid `29481` alive/sleeping on daily cap before v516; no duplicate submissions added.
- **Implementation:** Added `restore_best_by_val_loss` support to `scripts/birdclef_sed_pilot_train.py`: deep-copies model state whenever val loss improves, restores it before final predictions/export, writes `best_checkpoint_info.json`, and records `best_epoch` / `best_val_loss` in metrics. Py_compile passed locally and on trainer.
- **External pretrain rerun:** Added `xc_b0_q3_cap80_external_pretrain_balanced_ep18_bestloss.json` and ran `xc-b0-q3-cap80-external-pretrain-balanced-ep18-bestloss`: same ep18 config but `restore_best_by_val_loss=true`. Best val loss was epoch 13 (`0.085548`); restored/exported epoch-13 checkpoint. Holdout macro AUC `0.747224` over 122 valid classes, slightly above final ep18 `0.743267`; runtime `37.354s`; TorchScript `15.387MB`.
- **Downstream smoke:** Added `pl-r2-b0-v508-xc-ep18bestlossinit-5s-lr3e4-smoke`: best-loss ep18 checkpoint, v508 soft labels, 5s/160mel, lr3e-4, 256 rows, 3 epochs. Result best val `0.959082`, final all-row `0.947536`, corr `0.91029`, runtime `3.757s`.
- **Interpretation:** Best-loss restored ep18 checkpoint improved external holdout AUC but hurt downstream distillation versus both ep18-final smoke (`0.949731`) and ep12 smoke (`0.948186`). Kill this branch; do not scale or package. Keep v519/ep12 as active bridge. `restore_best_by_val_loss` remains useful infrastructure for future external-pretrain experiments, but val-loss is not aligned enough with downstream pseudo-label transfer here.

## 2026-05-08 16:05 UTC — Spec D pretrained model-zoo pseudo-label smoke

- **Track:** D model-zoo diversity + B pseudo-label student training. Hypothesis: a non-B0 ImageNet-pretrained backbone may be weaker standalone but lower-correlated enough to improve the v508/ep12-student blend.
- **Status checks:** Current best remains `0.927`. Latest visible submissions unchanged: v509/v508/v507 `0.927`, v514 `0.924`, v513 hidden complete, v506/v505 `0.927`, v512 hidden complete, v511 `0.926`, v510 `0.927`. v510/v516/v517/v518/v519 are COMPLETE with no failure messages. Queue monitor pid `29481` remains alive/sleeping on daily cap before v516; no duplicate submissions added.
- **Smoke sweep:** Added and ran 256-row / 3-epoch / 5s/160mel pretrained model-zoo pseudo-label smokes with v508 soft labels, no external init, lr1e-4, batch 8, best-val restore. EfficientNet-B3: best val `0.894639`, final all-row `0.878512`, corr `0.55088`, TorchScript `41.995MB`. EfficientNetV2-RW-S: best val `0.875823`, final `0.869060`, corr `0.67006`, TorchScript `88.739MB`. RegNetY-008: best val `0.929520`, final `0.934671`, corr `0.93849`, TorchScript `23.419MB`.
- **Scale decision:** B3/V2S were far below B0 smokes, so killed. RegNetY-008 had the best smoke plus lower corr than B0, so scaled.
- **Scaled RegNetY:** Added and ran `pl-r2-regnety008-v508-soft-p100-5s-pretrained-lr1e4-ep20-bestval`: all 792 rows / 20 epochs. Runtime `31.963s`; best epoch 20; best val AUC `0.985620`; final all-row macro AUC `0.986882` over 75 valid classes; teacher corr `0.98257`; TorchScript `23.419MB`.
- **Blend sweep:** v508 teacher baseline `0.991149`. RegNetY standalone is lower than ep12 B0 (`0.986882` vs `0.988494`). Best linear blend with v508 was weight `0` (no improvement); best rank blend weight `0.02` -> `0.991174` (`+0.0000247`). Combining RegNetY with ep12 B0 still picks ep12 B0 alone (linear w12=0.15,wr=0; rank w12=0.075,wr=0). Do **not** package RegNetY or create v520.
- **Interpretation / next:** RegNetY provides some lower-correlation signal but not enough local blend gain to justify a Kaggle slot, especially after recent rankblend public underperformance. Keep v519/ep12 B0 as active bridge. Next useful model-zoo direction would require external pretraining per backbone or clean OOF teacher labels, not raw v508-distilled zoo students.

## 2026-05-08 17:05 UTC — Spec C/D external-pretrain model-zoo smokes

- **Track:** C external-data pretraining + D model-zoo diversity. Hypothesis from the previous RegNetY raw pseudo-label result: model-zoo backbones may need target-species external pretraining before pseudo-label distillation can become useful.
- **Status checks:** Current best remains `0.927`. Latest visible submissions unchanged: v509/v508/v507 `0.927`, v514 `0.924`, v513 hidden complete, v506/v505 `0.927`, v512 hidden complete, v511 `0.926`, v510 `0.927`. v510/v516/v517/v518/v519 are COMPLETE with no failure messages. Queue monitor pid `29481` remains alive/sleeping on daily cap before v516; no duplicate submissions added.
- **RegNetY external smoke:** Added `xc-regnety008-q3-cap80-external-pretrain-smoke` (lr1e-4) and `xc-regnety008-q3-cap80-external-pretrain-lr3e4-smoke`, both ImageNet-pretrained RegNetY-008, XC q>=3 cap80 manifest, 5s/128mel/hop512, 128 files, 1 epoch. lr1e-4 AUC `0.533341` over 23 valid classes, runtime `5.617s`; lr3e-4 AUC `0.546421`, runtime `5.303s`. Both are far below B0 128-mel smoke `0.643700`.
- **B3/V2S external smoke:** Added `xc-b3-q3-cap80-external-pretrain-lr1e4-smoke` and `xc-v2s-q3-cap80-external-pretrain-lr1e4-smoke`. EfficientNet-B3 AUC `0.538324`, runtime `5.665s`, TorchScript `41.988MB`; EfficientNetV2-RW-S AUC `0.544671`, runtime `7.444s`, TorchScript `88.730MB`. Both are also far below B0 smoke and not better than RegNetY lr3e-4.
- **Decision:** Kill external-pretrained model-zoo branch for now; do not run ep6 full or package any zoo model. The best model-zoo smoke (RegNetY lr3e-4 `0.546421`) is not close enough to justify scaling, especially after raw RegNetY full did not blend linearly with v508/ep12-B0.
- **Interpretation / next:** The only validated active bridge remains v519 / ep12 external-init B0. Further model-zoo work should wait for clean OOF teacher labels or a stronger external manifest/data-cleaning pass, not just same q3/cap80 manifest with larger/different timm backbones.

## 2026-05-08 17:42 UTC — Spec C manifest quality/coverage smoke sweep

- **Track:** C external-data pretraining/taxonomy mapping, after model-zoo external-pretrain smokes failed. Goal was to test whether the active B0 external bridge is sensitive to manifest cleanliness/coverage before spending another full pretrain or Kaggle slot.
- **Status checks:** Current best remains `0.927`. Latest visible submissions unchanged: v509/v508/v507 `0.927`, v514 `0.924`, v513 hidden complete, v506/v505 `0.927`, v512 hidden complete, v511 `0.926`, v510 `0.927`. v510/v516/v517/v518/v519 are COMPLETE with no failure messages. Queue monitor pid `29481` is alive and sleeping on daily cap before v516; no duplicate submissions added.
- **Infrastructure note:** Staged missing `sample_submission.csv` onto trainer so the manifest builder can run remotely. Built `manifest_q4_cap80` (XC only, min rating 4, cap80, quality-preferring): 1,926 rows, 1,867 train / 59 val, class mix Amphibia 27 / Aves 1,889 / Mammalia 10, no iNat rows.
- **Smoke configs/results:**
  - `xc-b0-q4-cap80-external-pretrain-smoke`: B0, ImageNet/timm init, q4/cap80 manifest, 5s/128mel, lr3e-4, 128 files, 1 epoch. AUC `0.478367` over 25 valid classes, runtime `4.586s`.
  - `xc-b0-qall-cap120-external-pretrain-smoke`: B0, qall/cap120 manifest with iNat/unrated coverage, same recipe. AUC `0.459203` over 23 valid classes, runtime `4.885s`.
  - `xc-b0-q3-cap80-external-pretrain-lr1e3-smoke`: original q3/cap80 manifest but higher spec-grid LR `1e-3`. AUC `0.455688` over 23 valid classes, runtime `5.090s`.
  - Built `manifest_q3_cap40` (XC only, min rating 3, cap40, quality-preferring): 2,151 rows, 2,082 train / 69 val, class mix Amphibia 31 / Aves 2,109 / Mammalia 11. `xc-b0-q3-cap40-external-pretrain-smoke` AUC `0.529792` over 24 valid classes, runtime `4.456s`.
- **Comparison:** The prior q3/cap80 B0 smoke remains much stronger at `0.643700` over 25 valid classes. All four new manifest/LR smokes are substantially worse, so none pass the scale gate.
- **Decision:** Do not scale q4, qall/iNat, cap40, or lr1e-3 variants. Keep the original q3/cap80 + lr3e-4 B0 path as the only useful external-pretrain recipe, with v519/ep12 still the active bridge. Next C work should be a genuinely stronger data-cleaning or teacher-pseudo-labeling pass, not simpler rating/cap changes.

## 2026-05-08 18:42 UTC — Spec C+A external-init clean SED OOF v22/v23 + v520 Kaggle candidate

- **Track:** C external-data pretraining -> A real SED clean OOF teacher. After simple external manifest tweaks failed, pivoted to using the strong q3/cap80 ep12 B0 external checkpoint as initialization for real train-audio SED OOF, rather than more v508-derived pseudo-label students.
- **Status checks:** Current best remains `0.927`. Latest visible submissions unchanged: v509/v508/v507 `0.927`, v514 `0.924`, v513 hidden complete, v506/v505 `0.927`, v512 hidden complete, v511 `0.926`, v510 `0.927`. v510/v516/v517/v518/v519 are COMPLETE with no failure messages. Queue was daily-cap blocked before v516.
- **Implementation:** Extended `scripts/birdclef_sed_pilot_train.py` with `initial_checkpoint` / `initial_load_head`, loading TorchScript or torch checkpoints before training and skipping `frame_head.*` by default. This reuses external-pretrain encoder weights for clean SED OOF training while reinitializing the classification head.
- **Paired smoke:** Added paired configs `sed_b0_balanced_oof_v8_10s_160_100cls_paired_smoke.json` and `sed_b0_q3cap80_ep12init_oof_10s_160_100cls_paired_smoke.json` (B0, 10s/160mel, 300 files, 2 folds, 3 epochs). Baseline smoke AUC `0.511638` over 30 valid classes; ep12 external-init smoke AUC `0.709000` over 30 valid classes. Initial checkpoint loaded `352` encoder keys and skipped the two head keys.
- **Scaled v22:** Added `sed_b0_q3cap80_ep12init_oof_v22_10s_160_100cls_ep5.json` and ran 1000-file / 100-class / 3-fold / 5-epoch clean OOF. Result: macro AUC `0.860578` over 100 valid classes, fold AUCs `0.892919`, `0.877087`, `0.884974`, three `15.388MB` TorchScript folds. Comparable old B0 v8 was only `0.485820`, so external init is a large clean-OOF improvement.
- **Scaled v23 all-class coverage:** Added `sed_b0_q3cap80_ep12init_oof_v23_10s_160_allcls_ep5.json` with max_classes `234`, files_per_class `12`, min_files_per_class `1`, max_files `4000`. Completed 2053-row / 206-valid-class / 3-fold OOF: macro AUC `0.859564`, fold AUCs `0.891531`, `0.902704`, `0.898693`, three `15.388MB` TorchScript folds. This gives broad class coverage without losing the v22 signal.
- **Packaging:** Built bundle `artifacts/sed_bundles/sed-b0-q3cap80-ep12init-v23-bundle-v1` (3 TorchScript B0 folds, `46.164MB` uncompressed / `42.051MB` zip) and uploaded private Kaggle dataset `yourslewis/bc26-sed-b0-ep12init-v23-bundle-v1` with status OK.
- **Kaggle candidate:** Added and pushed real Kaggle kernel `yourslewis/bc26-v520-sed-b0-ep12init-v23-blend-005`, version 1. It copies the v510 real-SED v508-axis path but points to the new B0 ep12-init v23 bundle, uses `REAL_SED_BLEND_WEIGHT=0.05`, `REAL_SED_MAX_MODELS=3`, `REAL_SED_MIN_MODELS=1`, and a lighter time estimate. Push returned no invalid data/competition/kernel/model sources; immediate status was RUNNING with no failure message.
- **Queue:** Updated and restarted queue monitor with v520 inserted after v519. New monitor pid `73148`, log `logs/submit_pending_birdclef_queue_20260508T1842Z.log`; it skipped already-submitted v500-v514/v505-v509, retried v516, hit daily cap with about 5.1h remaining, and is sleeping. It will submit v516-v520 in order after cap clears, assuming v520 completes.
- **Next step:** Monitor v520 completion logs. Required success signal: manifest found under the new dataset, `Loading 3/3 real SED TorchScript models`, real SED probabilities applied, and `submission.csv` produced. If v520 completes, let the queue submit it after v516-v519; if it times out or falls back, reduce max models or create a 1-model representative bundle.

## 2026-05-08 19:42 UTC — v520 verification + v521 stronger B0 SED blend follow-up

- **Track:** A+G inference packaging / Spec F one-knob blend retune after a genuinely new SED artifact. No adjacent postprocess-only sweep; this is tied to the new B0 ep12-init v23 SED bundle.
- **Status checks:** Current best remains `0.927`. Latest visible submissions unchanged: v509/v508/v507 `0.927`, v514 `0.924`, v513 hidden complete, v506/v505 `0.927`, v512 hidden complete, v511 `0.926`, v510 `0.927`. v510/v516/v517/v518/v519 are COMPLETE with no failure messages. Queue monitor is alive but daily-cap blocked before v516.
- **v520 verification:** v520 version 1 completed with `submission.csv`. Output log confirms the intended new SED path ran: manifest found under `/kaggle/input/datasets/yourslewis/bc26-sed-b0-ep12init-v23-bundle-v1/sed-b0-q3cap80-ep12init-v23-bundle-v1/sed_bundle_manifest.json`, loaded `3/3` TorchScript models, real SED prob range `0.012542` to `0.731011`, mean `0.1211`, runtime `51.9s`, applied real SED bundle blend `weight=0.05`, final prob range `0.020101` to `0.955785`, output shape `240 x 235`, wall time `240.7s` / `4.0 min`.
- **Follow-up hypothesis:** Since v520 is much faster than the NFNet six-model path and uses a clean OOF model with v23 AUC `0.859564` over 206 classes, test one stronger blend bracket before waiting for LB: B0 SED blend weight `0.10`. This is the only follow-up variant added this run.
- **Kernel candidate:** Added and pushed real Kaggle kernel `yourslewis/bc26-v521-sed-b0-ep12init-v23-blend-010`, version 1. It is the v520 path with `REAL_SED_BLEND_WEIGHT=0.10`; same dataset `yourslewis/bc26-sed-b0-ep12init-v23-bundle-v1`, `REAL_SED_MAX_MODELS=3`, `REAL_SED_MIN_MODELS=1`. Push returned no invalid data/competition/kernel/model sources. Status after push: RUNNING, no failure message.
- **Queue:** Updated and restarted queue monitor with v521 after v520. New monitor pid `80397`, log `logs/submit_pending_birdclef_queue_20260508T1942Z.log`; it skipped already-submitted v500-v514/v505-v509, retried v516, hit daily cap with about 4.3h remaining, and is sleeping. Current order after cap: v516, v517, v518, v519, v520, then v521.
- **Next step:** Monitor v521 completion logs. If it completes, required success signs mirror v520: manifest found, `Loading 3/3`, real SED probabilities applied, and `submission.csv`. Do not add more B0 SED blend variants until v520/v521 LB scores arrive.

## 2026-05-08 20:42 UTC — v521 verification + Spec A 5s external-init SED OOF crop test

- **Track:** A+G monitoring plus Spec A crop-length OOF test. Per prior note, no additional B0 SED Kaggle blend variants were created; v520/v521 must score before more public candidates.
- **Status checks:** Current best remains `0.927`. Latest visible submissions unchanged: v509/v508/v507 `0.927`, v514 `0.924`, v513 hidden complete, v506/v505 `0.927`, v512 hidden complete, v511 `0.926`, v510 `0.927`. v510/v516/v517/v518/v519/v520/v521 are COMPLETE with no failure messages. Queue monitor pid `80397` remains alive and sleeping on daily cap before v516; current order after cap is v516, v517, v518, v519, v520, v521.
- **v521 verification:** v521 version 1 completed with `submission.csv`. Output log confirms intended path: manifest under `/kaggle/input/datasets/yourslewis/bc26-sed-b0-ep12init-v23-bundle-v1/sed-b0-q3cap80-ep12init-v23-bundle-v1/sed_bundle_manifest.json`, loaded `3/3` TorchScript models, real SED prob range `0.012542` to `0.731011`, mean `0.1211`, runtime `47.1s`, applied blend `weight=0.10`, final prob range `0.020765` to `0.924375`, output shape `240 x 235`, wall time `242.3s` / `4.0 min`.
- **5s smoke:** Tested whether direct 5s SED crops beat the packaged 10s v23 recipe, because 5s aligned better for earlier pseudo-label students. Added `sed_b0_q3cap80_ep12init_oof_5s_160_100cls_paired_smoke.json`: B0, q3/cap80 ep12 init, 5s/160mel, 300 files, 2 folds, 3 epochs, batch 12. Result: OOF AUC `0.733805` over 30 valid classes, fold AUCs `0.792276` and `0.811661`. This beats the earlier paired 10s smoke (`0.709000`) and passed the scale gate.
- **Scaled v24 5s all-class:** Added `sed_b0_q3cap80_ep12init_oof_v24_5s_160_allcls_ep5.json`: same all-class coverage as v23 (`max_classes=234`, files/class 12, min_files 1, max_files 4000), but 5s crops and batch 12. Completed 2053-row / 206-valid-class / 3-fold OOF: macro AUC `0.848133`, fold AUCs `0.880633`, `0.875337`, `0.892135`, three `15.388MB` folds.
- **Decision:** Kill v24 5s for packaging despite the smoke win. Full all-class v24 underperforms v23 10s (`0.848133` vs `0.859564` on the same 2053-row / 206-class coverage), so keep v23 as the best packaged B0 external-init SED bundle. Do not upload/package v24.
- **Next step:** Wait for LB scores for queued v516-v521. If v520/v521 tie/improve, v23 becomes the current B0 SED packaging baseline; if they fall, do not submit further blend variants and instead investigate OOF/test mismatch or alternate non-B0 backbones with external initialization.

## 2026-05-08 21:42 UTC — Spec A external-init SED 20s/mel224 crop-resolution probes

- **Track:** A+G monitoring plus Spec A crop/mel resolution probes for the new B0 q3/cap80 ep12-init SED family. No additional public Kaggle blend variants were created; v520/v521 still need LB scores before more public candidates.
- **Status checks:** Current best remains `0.927`. Latest visible submissions unchanged: v509/v508/v507 `0.927`, v514 `0.924`, v513 hidden complete, v506/v505 `0.927`, v512 hidden complete, v511 `0.926`, v510 `0.927`. v510/v516/v517/v518/v519/v520/v521 are COMPLETE with no failure messages. Queue monitor pid `80397` remains alive and sleeping on daily cap before v516; current order after cap is v516, v517, v518, v519, v520, v521.
- **20s crop smoke:** Added `sed_b0_q3cap80_ep12init_oof_20s_160_100cls_paired_smoke.json`: B0 q3/cap80 ep12 init, 20s/160mel, 300 files, 2 folds, 3 epochs, batch 6. Result: OOF AUC `0.677264` over 30 valid classes, fold AUCs `0.783197` and `0.762203`. This underperformed both 10s smoke `0.709000` and 5s smoke `0.733805`.
- **224-mel smoke:** Added `sed_b0_q3cap80_ep12init_oof_10s_224_100cls_paired_smoke.json`: B0 q3/cap80 ep12 init, 10s/224mel, 300 files, 2 folds, 3 epochs, batch 6. Result: OOF AUC `0.650149` over 30 valid classes, fold AUCs `0.737570` and `0.749079`. This is worse than 10s/160 and 5s/160.
- **Decision:** Kill 20s/160 and 10s/224 for scaling/packaging. Current best clean SED recipe remains v23: 10s/160mel all-class, OOF `0.859564` over 206 valid classes, already packaged as v520/v521. Keep waiting for v520/v521 LB before any more public B0 SED candidates.
- **Next step:** If continuing before LB scores arrive, the next internal-only Spec A knob should be loss/regularization on v23 (e.g. focal gamma or label smoothing) rather than crop length/mel resolution, because 5s/20s/224 did not beat the 10s/160 all-class recipe at scale.

## 2026-05-08 22:42 UTC — Spec A v23 loss/regularization smoke + no-mix full check

- **Track:** A+G monitoring plus internal-only Spec A loss/regularization probes on the best B0 q3/cap80 ep12-init SED v23 recipe. No new public Kaggle candidates were created; v520/v521 still need LB scores before any further B0 SED submissions.
- **Status checks:** Current best remains `0.927`. Latest visible submissions unchanged: v509/v508/v507 `0.927`, v514 `0.924`, v513 hidden complete, v506/v505 `0.927`, v512 hidden complete, v511 `0.926`, v510 `0.927`. v510/v516/v517/v518/v519/v520/v521 are COMPLETE with no failure messages. Queue monitor pid `80397` remains alive and sleeping on daily cap before v516; order after cap remains v516, v517, v518, v519, v520, v521.
- **Smoke baseline reference:** v23 paired smoke (10s/160mel, 300 files, 2 folds, 3 epochs, focal gamma1.5, label smoothing0.01, mixup0.2) was `0.709000` over 30 valid classes.
- **Gamma smoke:** `sed_b0_q3cap80_ep12init_oof_10s_160_100cls_gamma10_paired_smoke`: focal gamma `1.0`, otherwise v23 smoke. OOF AUC `0.709161` over 30 valid classes, fold AUCs `0.781037`, `0.802206`. Essentially tied baseline; not worth scaling.
- **No-mix smoke:** `sed_b0_q3cap80_ep12init_oof_10s_160_100cls_nomix_paired_smoke`: mixup `0.0`, otherwise v23 smoke. OOF AUC `0.725897` over 30 valid classes, fold AUCs `0.786903`, `0.808087`. Passed the smoke gate, so scaled.
- **No-smoothing smoke:** `sed_b0_q3cap80_ep12init_oof_10s_160_100cls_smooth000_paired_smoke`: label smoothing `0.0`, otherwise v23 smoke. OOF AUC `0.710793` over 30 valid classes, fold AUCs `0.778656`, `0.807260`. Small smoke lift only; not worth scaling after no-mix was stronger.
- **Scaled v25 no-mix:** `sed_b0_q3cap80_ep12init_oof_v25_10s_160_allcls_nomix_ep5`: same all-class coverage as v23 (`2053` rows / `206` valid classes), 10s/160mel, mixup `0.0`, 3 folds, 5 epochs. Result: OOF AUC `0.857140` over 206 classes, fold AUCs `0.904430`, `0.892359`, `0.892338`.
- **Decision:** Kill v25 no-mix for packaging. Despite a stronger 300-row smoke, full all-class no-mix underperforms packaged v23 (`0.857140` vs `0.859564`). Keep v23 (10s/160, gamma1.5, smoothing0.01, mixup0.2) as the best B0 external-init SED recipe. Do not package/upload v25.
- **Next step:** Wait for v516-v521 LB scores. If continuing internal work before scores, avoid more B0 SED micro-regularization unless testing a meaningfully different axis (e.g. non-B0 external-init SED with clean OOF), because crop/mel/mixup/gamma/smoothing probes have not beaten v23 at full all-class scale.

## 2026-05-08 23:42 UTC — Spec A v26 all-files external-init SED + v522 package

- **Track:** A+G monitoring plus Spec A data-coverage scaling for the best B0 q3/cap80 ep12-init SED family. This is not another blend micro-sweep: it changes the underlying SED artifact from v23's cap12 coverage to all available train-audio files.
- **Status checks:** Current best remains `0.927`. Latest visible submissions unchanged: v509/v508/v507 `0.927`, v514 `0.924`, v513 hidden complete, v506/v505 `0.927`, v512 hidden complete, v511 `0.926`, v510 `0.927`. v510/v516/v517/v518/v519/v520/v521 are COMPLETE with no failure messages. Queue monitor was alive but daily-cap blocked before v516; after refresh it was sleeping with about 11 minutes remaining.
- **Coverage check:** Train-audio target coverage has 206 class dirs / 2507 files. v23's `files_per_class=12` selected 2053 rows; increasing cap to 24 includes all 2507 rows (`cap24=2507`, same as cap40). This is a meaningful data-axis test, not a regularization tweak.
- **Scaled v26 all-files:** Added `sed_b0_q3cap80_ep12init_oof_v26_10s_160_allfiles_ep5.json`: same v23 recipe (10s/160mel, focal gamma1.5, label smoothing0.01, mixup0.2, B0 q3/cap80 ep12 init, 3 folds, 5 epochs) but `files_per_class=24`, `max_files=3000`, selecting all 2507 available training files. Result: OOF AUC `0.883617` over 206 valid classes, fold AUCs `0.926670`, `0.906451`, `0.917304`. This is a large improvement over packaged v23 (`0.859564` on 2053 rows / 206 classes).
- **Packaging:** Built bundle `artifacts/sed_bundles/sed-b0-q3cap80-ep12init-v26-allfiles-bundle-v1` with 3 TorchScript B0 folds (`46.164MB` uncompressed, `42.051MB` zip). Uploaded private Kaggle dataset `yourslewis/bc26-sed-b0-ep12init-v26-allfiles-bundle-v1` successfully.
- **Kaggle candidate:** Added and pushed real Kaggle kernel `yourslewis/bc26-v522-sed-b0-ep12init-v26-allfiles-blend-005`, version 1. It copies the v520 path but points to the v26 all-files dataset and keeps conservative `REAL_SED_BLEND_WEIGHT=0.05`, `REAL_SED_MAX_MODELS=3`, `REAL_SED_MIN_MODELS=1`. Push returned no invalid data/competition/kernel/model sources. Current status at logging time: RUNNING, no failure message, output not available yet.
- **Queue:** Updated and restarted queue monitor with v522 after v521. New monitor pid `13826`, log `logs/submit_pending_birdclef_queue_20260508T2342Z.log`. Current order after cap: v516, v517, v518, v519, v520, v521, v522.
- **Next step:** Monitor v522 completion logs for manifest found, `Loading 3/3`, applied real SED blend, and `submission.csv`. If v522 completes, let queue submit it after v516-v521. Do not add more B0 SED public candidates until queued LB scores arrive.

## 2026-05-09 00:42 UTC — v522 verified + queue priority hotfix

- **Track:** A+G monitoring/packaging plus queue reliability. No new public kernel was created because daily slots were already consumed; the priority was preventing another day of quota from going to stale legacy variants.
- **Status:** Current best remains `0.927`. Latest submission list after UTC reset showed five pending submissions: old `v250`, `v249`, `v248`, `v247`, and new `v516`. Recent scored rows remain `v509/v508/v507/v506/v505/v510=0.927`, `v511=0.926`, `v514=0.924`; `v512/v513` hidden complete. All active kernels `v516`-`v522` are COMPLETE with no failure messages.
- **v522 verification:** `yourslewis/bc26-v522-sed-b0-ep12init-v26-allfiles-blend-005` completed with `submission.csv`. Output log confirmed the intended real B0 SED path: manifest found under `bc26-sed-b0-ep12init-v26-allfiles-bundle-v1`, loaded `3/3` TorchScript models, real SED prob range `0.011945` to `0.803207`, mean `0.1217`, runtime `50.6s`, applied bundle blend weight `0.05`, final prob range `0.019895` to `0.961477`, output shape `240 x 235`, wall time `233.9s` / `3.9 min`.
- **Queue issue diagnosed:** The refreshed monitor correctly submitted `v516` after reset, but then used the historical `PENDING` order and submitted old `v247`-`v250`, consuming today's remaining four code-submission slots before reaching `v517`-`v522`. This contradicts the active spec-driven priority.
- **Fix:** Updated `scripts/submit_pending_birdclef_queue.py` to sort active focus candidates `v516`-`v522` ahead of the long legacy backlog while preserving the old backlog after focus candidates. Also increased recent-submission page size from 50 to 200 to reduce duplicate risk from older descriptions falling out of the first page.
- **Monitor:** Killed old pid `13826` and restarted pid `22189`, log `logs/submit_pending_birdclef_queue_20260509T0042Z.log`. Sanity check: it skipped already-submitted `v516`, attempted `v517`, hit daily cap, and is sleeping until next UTC reset. Next intended submissions are `v517`, `v518`, `v519`, `v520`, `v521`, `v522` in that order.
- **Next step:** Wait for pending `v516`/legacy `v247`-`v250` scores, then let the fixed monitor submit `v517`-`v522` at the next reset. Do not add more B0 SED public variants until v520-v522 LB scores arrive.

## 2026-05-09 01:42 UTC — v516 broke plateau; Spec E taxon-gate follow-ups v523/v524

- **Track:** Spec E Background/no-call/taxon gate, triggered by live LB evidence. This is follow-up tuning on a newly successful signal, not a generic postprocess micro-sweep.
- **Status:** `v516` scored `0.929`, breaking the previous `0.927` plateau. Legacy submissions from the accidental queue burn scored/are scoring safely below the new best: `v247=0.925`, `v248=0.926`, `v249=0.926`, `v250=0.926`. Recent real SED rows remain `v510=0.927`, `v511=0.926`, `v514=0.924`, with `v512/v513` hidden complete. Active queued kernels `v517`-`v522` remain COMPLETE/no failure.
- **v510 verification:** Rechecked required real SED v510 path. Kernel status COMPLETE/no failure; output has `submission.csv`; log confirms manifest under `bc26-sed-nfnet-v13v15-bundle-v1`, loaded `6/6` TorchScript models, real SED prob range `0.000003` to `0.624691`, runtime `214.4s`, applied blend weight `0.05`, output shape `240 x 235`, wall time `370.6s` / `6.2 min`.
- **Hypothesis:** v516's family/taxon max gate is suppressing cross-family false positives enough to move public LB. Existing v517 tests softer alpha `0.50`; add two focused neighbors around the winning v516 floor/alpha to test whether the gain prefers stronger evidence sharpening or lower no-evidence floor.
- **New candidates pushed:**
  - `v523`: `yourslewis/bc26-v523-taxon-max-gate-alpha0875`, copies v516 and changes only `TAXON_MAX_GATE_ALPHA=0.75 -> 0.875` with floor fixed at `0.30`.
  - `v524`: `yourslewis/bc26-v524-taxon-max-gate-floor020`, copies v516 and changes only `TAXON_MAX_GATE_FLOOR=0.30 -> 0.20` with alpha fixed at `0.75`.
  Both pushed via Bearer API v1; push responses had no invalid data/competition/kernel/model sources. Initial status: RUNNING/no failure, no output yet.
- **Queue:** Updated `scripts/submit_pending_birdclef_queue.py` so the active priority is `v516`, `v517`, `v523`, `v524`, `v518`, `v519`, `v520`, `v521`, `v522`, then legacy backlog. Restarted monitor pid `30519`, log `logs/submit_pending_birdclef_queue_20260509T0142Z.log`. Sanity check skipped submitted `v516`, attempted `v517`, hit daily cap, and slept until next UTC reset.
- **Next step:** Monitor v523/v524 completion logs for `submission.csv` and then let the queue submit v517/v523/v524 first at the next reset. If v517 underperforms but v523/v524 are still queued, keep them because v516's 0.929 indicates the gate axis has real public signal.

## 2026-05-09 03:42 UTC — v523/v524 verified + v525 high-floor taxon gate

- **Track:** Spec E taxon-gate follow-up around the new public best. This run keeps exploiting the only axis that has broken the plateau so far (`v516=0.929`) while the queue is capped.
- **Status:** Current best remains `0.929` from `v516` (`taxon max gate floor0.30 alpha0.75 + v508 axis`). Latest visible submissions: `v250=0.926`, `v249=0.926`, `v248=0.926`, `v247=0.925`, `v516=0.929`, then `v509/v508/v507=0.927`. Active kernels `v517`, `v523`, `v524`, `v518`, `v519`, `v520`, `v521`, `v522` are COMPLETE/no failure.
- **v523/v524 verification:**
  - `v523` (`yourslewis/bc26-v523-taxon-max-gate-alpha0875`) completed with `submission.csv`; log confirmed taxon max gate `floor=0.3`, `alpha=0.875`, final prob range `0.009520` to `0.990947`, mean `0.3910`, wall time `146.9s` / `2.4 min`.
  - `v524` (`yourslewis/bc26-v524-taxon-max-gate-floor020`) completed with `submission.csv`; log confirmed taxon max gate `floor=0.2`, `alpha=0.75`, final prob range `0.010599` to `0.991548`, mean `0.3984`, wall time `172.8s` / `2.9 min`.
- **Hypothesis for new candidate:** v517/v523 bracket alpha around winning v516; v524 tests a more aggressive lower floor. Add a single complementary high-floor variant to test whether v516's gain benefits from preserving more rare/low-evidence probabilities while retaining family-level suppression.
- **New candidate:** Added and pushed `v525` (`yourslewis/bc26-v525-taxon-max-gate-floor040`), copying v516 and changing only `TAXON_MAX_GATE_FLOOR=0.30 -> 0.40`, with `TAXON_MAX_GATE_ALPHA=0.75` unchanged. Pushed via Bearer API v1; no invalid data/competition/kernel/model sources. Initial status: RUNNING/no failure, output not available yet.
- **Queue:** Existing monitor had died while capped. Restarted monitor pid `50272`, log `logs/submit_pending_birdclef_queue_20260509T0342Z.log`. Updated active priority to `v516`, `v517`, `v523`, `v524`, `v525`, `v518`, `v519`, `v520`, `v521`, `v522`, then legacy backlog. Sanity check skipped submitted `v516`, attempted `v517`, hit daily cap, and slept until next UTC reset.
- **Next step:** Monitor v525 completion. At next UTC reset, submit `v517`, `v523`, `v524`, `v525`, and then `v518` if quota allows. If any of the taxon-gate neighbors beats/ties `0.929`, continue this axis; otherwise return to queued new-model SED/pseudo candidates.

## 2026-05-09 04:42 UTC — v525 verified; taxon-gate queue ready

- **Track:** Spec E taxon-gate monitoring/validation. No new public kernel was added this run because the next reset is already filled with focused v516-neighbor candidates, and adding another correlated gate variant before `v517/v523/v524/v525` LB would be wasteful.
- **Status:** Current best remains `0.929` from `v516` (`taxon max gate floor0.30 alpha0.75 + v508 axis`). Latest visible rows: `v250=0.926`, `v249=0.926`, `v248=0.926`, `v247=0.925`, `v516=0.929`, `v509/v508/v507=0.927`, `v514=0.924`, `v510=0.927`. Active queued kernels `v517`, `v523`, `v524`, `v525`, `v518`, `v519`, `v520`, `v521`, `v522` are all COMPLETE/no failure.
- **v525 verification:** `yourslewis/bc26-v525-taxon-max-gate-floor040` completed with `submission.csv`. Log confirmed taxon max gate `floor=0.4`, `alpha=0.75`, final prob range `0.010599` to `0.991548`, mean `0.3986`, output shape `240 x 235`, wall time `159.8s` / `2.7 min`. No Traceback; Kaggle log contains the usual TensorFlow `ERROR (303)` noise before successful ONNX/Perch load and submission write.
- **Queue:** Monitor pid `50272`, log `logs/submit_pending_birdclef_queue_20260509T0342Z.log`, is alive. It skipped already-submitted `v516`, attempted `v517`, hit the daily cap, and is sleeping until next UTC reset. Active priority remains `v516`, `v517`, `v523`, `v524`, `v525`, `v518`, `v519`, `v520`, `v521`, `v522`, then legacy backlog.
- **Next step:** Let the monitor submit the bracketed taxon-gate set at reset: `v517` (softer alpha), `v523` (stronger alpha), `v524` (lower floor), `v525` (higher floor), then `v518` if quota allows. Evaluate against `v516=0.929` before adding further gate variants.

## 2026-05-09 05:42 UTC — taxon-gate teacher-cache sweep; no extra public variant

- **Track:** Spec E taxon-gate diagnostics while waiting for the next UTC reset. Public LB best remains `0.929` from `v516`; queued focus kernels `v517/v523/v524/v525/v518/v519/v520/v521/v522` are all COMPLETE/no failure, and monitor pid `50272` remains alive/sleeping on daily cap before `v517`.
- **Status:** Latest visible submissions unchanged: `v250=0.926`, `v249=0.926`, `v248=0.926`, `v247=0.925`, `v516=0.929`, `v509/v508/v507=0.927`, `v514=0.924`, `v510=0.927`. v510 was already verified with real SED blend and `submission.csv`; no new v510 issue.
- **Implementation:** Added `scripts/birdclef_taxon_gate_teacher_cache_sweep.py`, a reproducible diagnostic for the exact Kaggle taxon gate formula used by v516+ (`multiplier = max(floor, group_evidence) ** alpha`). It aligns cached v508 teacher predictions with labeled train-soundscape truth, maps labels to taxonomy `class_name`, sweeps `floor/alpha/mode`, and writes JSON output.
- **Validation command:** `python3 -m py_compile scripts/birdclef_taxon_gate_teacher_cache_sweep.py`, then ran against `artifacts/pseudolabels/v508-teacher-cache66/predictions.npz` + `artifacts/pseudolabels/students/pl-r2-regnety008-v508-soft-p100-5s-pretrained-lr1e4-ep20-bestval/student_predictions.npz`, taxonomy `/Volumes/ExternalSSD/data/workspace_don/kaggle_birdclef2026/data/taxonomy.csv`. Output artifact: `artifacts/gate/taxon_gate_teacher_cache_v508_kernel_formula.json`.
- **Results:** Baseline v508 teacher-cache macro AUC `0.991149` over 75 valid classes. Best max-gate result is alpha `0.75` with macro AUC `0.992162` (+`0.001013`). On these labeled rows, floor is saturated/invariant for `0.10`-`0.50` at alpha `0.75`; queued v516/v524/v525 all match the local best. Alpha bracket: `0.50` -> `0.992008` (+`0.000859`), `0.625` -> `0.992065` (+`0.000915`), `0.75` -> `0.992162` (+`0.001013`), `0.875` -> `0.992124` (+`0.000975`), `1.0` -> `0.992043` (+`0.000894`). Top-k-mean mode is weaker and drops as alpha rises.
- **Decision:** Do not add another public gate variant before v517/v523/v524/v525 scores. The local diagnostic says v516's alpha `0.75` is the peak and floor variants are locally indistinguishable, so the next useful information must come from LB. Queue order remains `v517`, `v523`, `v524`, `v525`, then `v518` if quota allows.

## 2026-05-09 06:42 UTC — queue/candidate verification while capped

- **Track:** Monitoring/validation for Spec E taxon-gate queue plus required v510 real-SED check. No new public candidate was added: the next reset is already fully allocated to the focused v516-neighbor bracket, and the latest teacher-cache sweep showed no reason to add another correlated gate variant before LB feedback.
- **Status:** Public best remains `0.929` from `v516` (`taxon max gate floor0.30 alpha0.75 + v508 axis`). Latest visible submissions unchanged: `v250=0.926`, `v249=0.926`, `v248=0.926`, `v247=0.925`, `v516=0.929`, `v509/v508/v507=0.927`, `v514=0.924`, `v510=0.927`. All focus kernels `v517`, `v523`, `v524`, `v525`, `v518`, `v519`, `v520`, `v521`, `v522` are COMPLETE/no failure.
- **Required v510 check:** `yourslewis/bc26-v510-real-sed-bundle-blend-005` remains COMPLETE with `submission.csv`; log confirms manifest under `bc26-sed-nfnet-v13v15-bundle-v1`, loaded `6/6` real SED TorchScript models, real SED runtime `214.4s`, blend weight `0.05`, and wall time `370.6s` / `6.2 min`. No Traceback.
- **Taxon-gate candidate verification:** Confirmed all queued taxon-gate neighbors have `submission.csv` and no Traceback: `v517` floor `0.3` alpha `0.5`, wall `157.1s`; `v523` floor `0.3` alpha `0.875`, wall `146.9s`; `v524` floor `0.2` alpha `0.75`, wall `172.8s`; `v525` floor `0.4` alpha `0.75`, wall `159.8s`.
- **Queue:** Monitor pid `50272`, log `logs/submit_pending_birdclef_queue_20260509T0342Z.log`, is alive. It skipped already-submitted `v516`, attempted `v517`, hit cap, and is sleeping. Intended next reset submissions remain `v517`, `v523`, `v524`, `v525`, then `v518` if quota allows.
- **Next step:** Do not touch queue order or add variants unless monitor dies. After reset/scoring, compare `v517/v523/v524/v525` to `v516=0.929`; if none improve/tie safely, pivot slots to queued new-model signals (`v518`/`v519` pseudo-student and `v520`-`v522` B0 SED) rather than further gate micro-sweeps.

## 2026-05-09 07:42 UTC — capped queue audit; hold variants until LB feedback

- **Track:** Monitoring/decision discipline for Spec E + queued new-model signals. No new candidate was added this run because the next reset already has five complete, non-duplicated focus kernels ready, and the latest teacher-cache sweep says additional taxon-gate variants before public LB feedback would be over-correlated.
- **Status:** Public best remains `0.929` from `v516` (`taxon max gate floor0.30 alpha0.75 + v508 axis`). Latest visible submissions unchanged: `v250=0.926`, `v249=0.926`, `v248=0.926`, `v247=0.925`, `v516=0.929`, `v509/v508/v507=0.927`, `v514=0.924`, `v510=0.927`. All focus kernels are COMPLETE/no failure: `v517`, `v523`, `v524`, `v525`, `v518`, `v519`, `v520`, `v521`, `v522`.
- **v510/failure check:** Reconfirmed `v510` remains COMPLETE with `submission.csv` and no Traceback; previous log had verified real SED manifest, `6/6` TorchScript load, blend `0.05`, and wall time `6.2 min`. No dataset mount/zip/TorchScript/CPU-time/missing-output issue exists.
- **Queue check:** Monitor pid `50272`, log `logs/submit_pending_birdclef_queue_20260509T0342Z.log`, is still alive and sleeping on daily cap before `v517`. Script focus priority is correct: `v516`, `v517`, `v523`, `v524`, `v525`, `v518`, `v519`, `v520`, `v521`, `v522`, then legacy backlog. `recent_messages()` page size is `200`, reducing duplicate risk.
- **Decision:** Do not restart the monitor or touch queue order while it is sleeping correctly. Do not add another taxon-gate variant before the bracket scores. After reset, expected submissions are `v517`, `v523`, `v524`, `v525`, then `v518` if quota allows. If the gate bracket does not beat/tie `0.929`, pivot future slots to new-model signals (`v518/v519` pseudo-student and `v520`-`v522` B0 SED), not more taxon micro-sweeps.

## 2026-05-09 08:42 UTC — no-op hold: queue healthy, variants complete

- **Track:** Monitoring / queue integrity while daily cap blocks submissions. No implementation or public candidate was added because all prepared focus kernels are complete and the monitor is correctly sleeping; disturbing the queue would increase duplicate/ordering risk without adding signal.
- **Status:** Public best remains `0.929` from `v516`. Latest visible submissions unchanged: `v250=0.926`, `v249=0.926`, `v248=0.926`, `v247=0.925`, `v516=0.929`, `v509/v508/v507=0.927`, `v514=0.924`, `v510=0.927`. All focus kernels remain COMPLETE/no failure: `v517`, `v523`, `v524`, `v525`, `v518`, `v519`, `v520`, `v521`, `v522`.
- **Queue:** Monitor pid `50272` is alive, log `logs/submit_pending_birdclef_queue_20260509T0342Z.log`. It is sleeping on the daily cap before `v517`. Focus priority remains correct in `scripts/submit_pending_birdclef_queue.py`: `v516`, `v517`, `v523`, `v524`, `v525`, `v518`, `v519`, `v520`, `v521`, `v522`, then legacy backlog. `recent_messages()` still pages 200 submissions.
- **Candidate readiness:** `v517/v523/v524/v525` taxon-gate bracket and `v518/v519/v520/v521/v522` new-model signals are all already COMPLETE/no failure. Required v510 real-SED path remains previously verified and has no open issue.
- **Decision:** Hold. Do not add more taxon variants and do not restart the monitor unless it dies. Next meaningful event is UTC reset: submit `v517`, `v523`, `v524`, `v525`, then `v518` if quota allows. After scores land, compare against `v516=0.929`; if the bracket misses, pivot daily slots toward `v518/v519` pseudo-students and `v520`-`v522` B0 SED rather than gate micro-sweeps.

## 2026-05-09 09:42 UTC — queued new-model kernel log verification

- **Track:** Spec A/G + B packaging validation while daily cap blocks submissions. No new candidate was added; instead, verified every queued new-model kernel behind the taxon bracket for silent-skip/output issues before it spends future quota.
- **Status:** Public best remains `0.929` from `v516`. Latest visible submissions unchanged: `v250=0.926`, `v249=0.926`, `v248=0.926`, `v247=0.925`, `v516=0.929`, `v509/v508/v507=0.927`, `v514=0.924`, `v510=0.927`. Focus kernels remain COMPLETE/no failure: `v517/v523/v524/v525/v518/v519/v520/v521/v522`.
- **Queue:** Monitor pid `50272`, log `logs/submit_pending_birdclef_queue_20260509T0342Z.log`, remains alive and sleeping on daily cap before `v517`. Queue order remains `v517`, `v523`, `v524`, `v525`, then `v518` if quota permits, followed by `v519/v520/v521/v522` on later resets.
- **v518 validation:** `yourslewis/bc26-v518-student-b0-5s-blend-002` has `submission.csv`; manifest under `bc26-student-b0-5s-v1`, loaded `1/1` TorchScript model, student prob range `0.021846` to `0.988939`, runtime `15.2s`, applied blend weight `0.02`, wall time `215.0s` / `3.6 min`, no Traceback.
- **v519 validation:** `yourslewis/bc26-v519-student-b0-5s-ep12-init-blend-015` has `submission.csv`; manifest under `bc26-student-b0-5s-ep12init-v1`, loaded `1/1` TorchScript model, prob range `0.019439` to `0.991308`, runtime `12.1s`, applied blend weight `0.15`, wall time `183.9s` / `3.1 min`, no Traceback.
- **v520/v521/v522 validation:** B0 SED kernels all have `submission.csv`, loaded `3/3` TorchScript models, applied intended blend weights, and have no Traceback. v520 v23 weight `0.05`: prob range `0.012542` to `0.731011`, runtime `51.9s`, wall `240.7s`. v521 v23 weight `0.10`: same prob range, runtime `47.1s`, wall `242.3s`. v522 v26 all-files weight `0.05`: prob range `0.011945` to `0.803207`, runtime `50.6s`, wall `233.9s`.
- **Decision:** All queued new-model candidates are packaging-safe. Hold queue and wait for reset. If taxon bracket misses `0.929`, use the validated next slots on `v518/v519/v520-v522` rather than creating more gate variants.

## 2026-05-09 10:42 UTC — capped wait audit; no duplicate submissions

- **Track:** Queue monitoring / anti-duplication while daily cap blocks submissions. No new implementation or public kernel was added; the complete focus queue already covers the immediate Spec E bracket plus queued Spec B/A new-model candidates.
- **Status:** Public best remains `0.929` from `v516`. Latest visible submissions unchanged: `v250=0.926`, `v249=0.926`, `v248=0.926`, `v247=0.925`, `v516=0.929`, `v509/v508/v507=0.927`, `v514=0.924`, `v510=0.927`. All focus kernels remain COMPLETE/no failure: `v517`, `v523`, `v524`, `v525`, `v518`, `v519`, `v520`, `v521`, `v522`.
- **Duplicate check:** Confirmed the next focus descriptions are not already present in the visible submission list: `v517`, `v523`, `v524`, `v525`, and `v518` are all not submitted yet. This matches the monitor state and prevents accidental duplicate work.
- **Queue:** Monitor pid `50272`, log `logs/submit_pending_birdclef_queue_20260509T0342Z.log`, remains alive and sleeping on daily cap before `v517`. Do not restart it unless it dies.
- **Decision:** Continue holding until UTC reset. Expected next submissions remain `v517`, `v523`, `v524`, `v525`, then `v518` if quota allows. If taxon bracket does not beat/tie `0.929`, use validated new-model kernels (`v518/v519/v520-v522`) rather than more taxon gate micro-sweeps.

## 2026-05-09 11:42 UTC — capped queue plus GPU-host blocker audit

- **Track:** A+G/B/E monitoring and decision discipline while daily cap blocks public submissions. No new kernel/candidate was added because the next reset is already filled by complete focus candidates, and adding another correlated variant would not improve information value before LB feedback.
- **Status:** Public best remains `0.929` from `v516`. Latest visible submissions unchanged: `v250=0.926`, `v249=0.926`, `v248=0.926`, `v247=0.925`, `v516=0.929`, `v509/v508/v507=0.927`, `v514=0.924`, `v510=0.927`. All focus kernels remain COMPLETE/no failure: `v517`, `v523`, `v524`, `v525`, `v518`, `v519`, `v520`, `v521`, `v522`.
- **Queue:** Monitor pid `50272`, log `logs/submit_pending_birdclef_queue_20260509T0342Z.log`, is alive and sleeping on daily cap before `v517`. It skipped submitted `v516`, attempted `v517`, received Kaggle's daily allowance error, and is sleeping. Do not restart unless it dies.
- **Infrastructure check:** Tried to reach GPU server `yourslewis@192.168.0.10` to collect the old NFNet 20s v22b outcome / inspect whether any BirdCLEF GPU job is active, but SSH timed out on port 22. Local repo still has no v22b `oof_summary.json`, so v22b remains unverified. Do not stack a v23 fallback launch blindly while the host is unreachable.
- **Decision:** Hold queue and avoid duplicate submissions. Expected reset order remains `v517`, `v523`, `v524`, `v525`, then `v518` if quota allows. If the gate bracket does not beat/tie `0.929`, pivot available slots to already-validated new-model signals `v518/v519/v520-v522`; if GPU host returns, first collect v22b before launching any additional SED training.

## 2026-05-09 12:42 UTC — focus-only queue guard activated

- **Track:** Queue safety / Spec A+G/B/E monitoring while Kaggle daily cap blocks submissions and GPU host remains unreachable. This is a concrete maintenance action to preserve future submission quota, not a new model candidate.
- **Status:** Public best remains `0.929` from `v516`. Latest visible unchanged: `v250=0.926`, `v249=0.926`, `v248=0.926`, `v247=0.925`, `v516=0.929`, `v509/v508/v507=0.927`, `v514=0.924`, `v510=0.927`. Focus kernels remain COMPLETE/no failure: `v517`, `v523`, `v524`, `v525`, `v518`, `v519`, `v520`, `v521`, `v522`.
- **Required v510 check:** `bc26-v510-real-sed-bundle-blend-005` remains COMPLETE/no failure and already scored `0.927`; no current v510 fix is needed.
- **Implementation:** Updated `scripts/submit_pending_birdclef_queue.py` so `BIRDCLEF_QUEUE_STOP_AFTER_FOCUS` defaults to `1`. The monitor now filters `PENDING` to the focus queue (`v516`, `v517`, `v523`, `v524`, `v525`, `v518`, `v519`, `v520`, `v521`, `v522`) and will not fall through into stale legacy micro-sweep variants unless explicitly started with `BIRDCLEF_QUEUE_STOP_AFTER_FOCUS=0`.
- **Validation/monitor:** `python3 -m py_compile scripts/submit_pending_birdclef_queue.py` passed. Restarted the sleeping monitor once to activate the guard: old pid `50272` -> new pid `32468`, log `logs/submit_pending_birdclef_queue_20260509T124415Z_focusonly.log`. It skipped submitted `v516`, attempted `v517`, hit Kaggle daily cap with ~11h remaining, and is sleeping correctly.
- **Infrastructure:** GPU server `192.168.0.10` still times out on SSH, so old NFNet 20s `v22b` remains unverified and no blind `v23` fallback was launched.
- **Decision:** Hold focus-only queue until reset. After focus queue completes, the monitor should exit instead of spending a spare quota slot on legacy variants. If gate bracket misses `0.929`, next information-rich slots are already validated `v518/v519/v520-v522`.

## 2026-05-09 13:42 UTC — focus-only guard preserved after PR consolidation

- **Track:** Queue safety and monitoring while capped. PR #212 was closed unmerged after its prior commits were consolidated through PR #213 into `main`, but the latest focus-only queue guard commit was not in `origin/main`; this run preserved that safety change on a fresh active branch.
- **Status:** Public best remains `0.929` from `v516`. Latest visible unchanged: `v250=0.926`, `v249=0.926`, `v248=0.926`, `v247=0.925`, `v516=0.929`, `v509/v508/v507=0.927`, `v514=0.924`, `v510=0.927`. Focus kernels remain COMPLETE/no failure: `v517`, `v523`, `v524`, `v525`, `v518`, `v519`, `v520`, `v521`, `v522`.
- **Required v510 check:** `bc26-v510-real-sed-bundle-blend-005` remains COMPLETE/no failure and already scored `0.927`; no SED packaging failure is open.
- **Queue/monitor:** Running focus-only monitor pid `32468`, log `logs/submit_pending_birdclef_queue_20260509T124415Z_focusonly.log`, is sleeping on daily cap before `v517` after correctly skipping submitted `v516`. The script defaults `BIRDCLEF_QUEUE_STOP_AFTER_FOCUS=1`, so it will stop after `v516/v517/v523/v524/v525/v518/v519/v520/v521/v522` instead of falling into legacy backlog.
- **Infrastructure:** GPU server `192.168.0.10` still times out on SSH, so NFNet 20s `v22b` remains unverified and no blind `v23` fallback was launched.
- **Decision:** Open a new PR for the focus-only guard because PR #212 is closed and no open PR remains. Continue holding until UTC reset; after scores land, compare taxon bracket to `v516=0.929`, then pivot slots to `v518/v519/v520-v522` if needed.

## 2026-05-09 14:42 UTC — focus queue hold; GPU host still down

- **Track:** Queue monitoring / infrastructure triage. No new model/kernel was added because all focus candidates are complete, the daily cap still blocks `v517`, and the GPU host is unreachable for collecting or launching training work.
- **Status:** Public best remains `0.929` from `v516`. Latest visible unchanged: `v250=0.926`, `v249=0.926`, `v248=0.926`, `v247=0.925`, `v516=0.929`, `v509/v508/v507=0.927`, `v514=0.924`, `v510=0.927`. Focus kernels remain COMPLETE/no failure: `v517`, `v523`, `v524`, `v525`, `v518`, `v519`, `v520`, `v521`, `v522`.
- **Required v510 check:** `bc26-v510-real-sed-bundle-blend-005` remains COMPLETE/no failure and is already scored/submitted at `0.927`; no dataset-mount, TorchScript, timeout, missing-output, or silent-skip issue is currently indicated.
- **Queue/PR:** Focus-only monitor pid `32468`, log `logs/submit_pending_birdclef_queue_20260509T124415Z_focusonly.log`, remains alive and sleeping on daily cap before `v517`. PR #214 is open/mergeable but branch-policy BLOCKED, with no status checks reported. It is now the only open PR and preserves the focus-only guard after PR #212 was closed/consolidated.
- **Infrastructure:** GPU server `192.168.0.10` had `100%` ping packet loss (`0/3`) and SSH timed out. NFNet 20s `v22b` remains unverified; do not launch `v23` fallback or any new GPU job until the host returns and v22b artifacts are checked.
- **Decision:** Hold. Let the guarded monitor submit after UTC reset. If `v517/v523/v524/v525` miss `0.929`, use queued new-model signals `v518/v519/v520-v522` rather than more gate micro-sweeps.

## 2026-05-09 15:42 UTC — v22b collected; v23b 20s/128 NFNet launched

- **Track:** A Real SED crop/context tuning + queue monitoring. GPU host recovered during this run, so I collected the previously blocked NFNet 20s `v22b` result and launched the prepared lower-mel follow-up instead of only waiting on the Kaggle cap.
- **Status:** Public best remains `0.929` from `v516`. Latest visible unchanged: `v250=0.926`, `v249=0.926`, `v248=0.926`, `v247=0.925`, `v516=0.929`, `v509/v508/v507=0.927`, `v514=0.924`, `v510=0.927`. Focus kernels remain COMPLETE/no failure, and monitor pid `32468` remains sleeping on cap before `v517`.
- **Required v510 check:** `bc26-v510-real-sed-bundle-blend-005` remains COMPLETE/no failure and already scored/submitted at `0.927`.
- **v22b outcome:** GPU server `192.168.0.10` recovered (`0%` ping loss; SSH ok). Collected `/home/yourslewis/birdclef-2026/artifacts/sed_oof/sed-nfnet-balanced-oof-v22b-20s-160-100cls-lr1e4-ep5-gpu1retry/oof_summary.json`. Result: OOF complete, `1000` rows, macro AUC `0.607845` over `100` valid classes. Fold AUCs: `0.654546`, `0.661929`, `0.674741`; each TorchScript export is `89.871 MB`.
- **v22b comparison:** Against v13 10s/160 ep8 on the same 1000-row overlap, v22b is weaker standalone (`0.607845` vs `0.636878`) but complementary: Pearson `0.7417`, best blend `40%` v22b gives `0.646332` vs `0.636878`. Against the v13/v15 `0.4/0.6` OOF base, v22b has lower correlation (`0.5835`) and improves a low-weight blend: base `0.657329`, best `20%` v22b gives `0.661518`.
- **v23 attempt/fix:** First launch of original `v23` 20s/128 failed in `18.8s` because the old config pointed to missing `/mnt/mac_data/workspace_don/kaggle_birdclef2026/data/taxonomy.csv` after the host returned without the SMB mount. Root cause verified; not a model failure.
- **v23b launch:** Added `configs/birdclef/sed_nfnet_balanced_oof_v23b_20s_128_100cls_lr1e4_ep5_datahome.json` with `data_root=/home/yourslewis/birdclef-2026/data`, experiment/output `sed-nfnet-balanced-oof-v23b-20s-128-100cls-lr1e4-ep5-datahome`. Launched durable run on GPU server with `CUDA_VISIBLE_DEVICES=0`: parent pid `4715`, fold0 child pid `4785`, log `logs/sed_oof_v23b_nfnet_20s128_datahome_20260509T154755Z.log`. Initial check: fold0 running, no immediate traceback.
- **Decision:** Do not package v22b directly yet; it is useful as low-correlation OOF signal but weaker standalone and heavy. Monitor v23b next; if 20s/128 retains the blend lift with lower runtime/size pressure, consider a low-weight SED bundle variant after current Kaggle focus queue scores.

## 2026-05-09 16:42 UTC — v23b/v23c NFNet 20s 128-mel resolved

- **Track:** A Real SED crop/context tuning + queue monitoring. Continued from the recovered GPU host state and completed the 20s/128 NFNet follow-up analysis.
- **Status:** Public best remains `0.929` from `v516`. Latest visible unchanged: `v250=0.926`, `v249=0.926`, `v248=0.926`, `v247=0.925`, `v516=0.929`, `v509/v508/v507=0.927`, `v514=0.924`, `v510=0.927`. All focus queue kernels remain COMPLETE/no failure; focus-only monitor pid `32468` remains sleeping on cap before `v517`. PR #214 is open/mergeable but branch-policy BLOCKED.
- **Required v510 check:** `bc26-v510-real-sed-bundle-blend-005` remains COMPLETE/no failure and already scored/submitted at `0.927`; no current real-SED packaging fix needed.
- **v23b result:** The data-root-fixed 20s/128 NFNet run completed: `sed-nfnet-balanced-oof-v23b-20s-128-100cls-lr1e4-ep5-datahome`, `1000` OOF rows, macro AUC `0.623477` over `100` valid classes. Fold AUCs: `0.709531`, `0.678290`, `0.674094`; TorchScript size `89.87 MB`/fold. It is better standalone than v22b (`0.607845`) but uses a different 1000-row selection, so it is not directly comparable to the v13/v15 overlap grid.
- **Comparable manifest retry:** Added manifest `configs/birdclef/manifests/nfnet_v22b_1000_manifest.csv` from v22b's original 1000 rows and config `configs/birdclef/sed_nfnet_balanced_oof_v23c_20s_128_100cls_lr1e4_ep5_v22bmanifest.json` to force same-file comparison with data root `/home/yourslewis/birdclef-2026/data`. The run completed quickly but only `119/1000` manifest files exist under the home data copy, producing an invalid low-coverage OOF (`n_oof=119`, macro AUC `0.303789`, fold valid classes only `28/30/31`). This is a data-coverage failure, not evidence against 20s/128 modeling.
- **Decision:** Do not package v23b/v23c yet. The only clean comparable signal remains v22b: weaker standalone but low-correlated and OOF-blend-positive with the v13/v15 base. A future packaging candidate should use v22b-style file/model artifacts only after the current focus Kaggle queue scores, or after restoring the full old training data mount for a fair v23c rerun.

## 2026-05-09 17:42 UTC — restored v22b manifest data; v23d fair 20s/128 retry launched

- **Track:** A Real SED crop/context tuning + queue monitoring. The previous v23c same-file attempt was invalid because the recovered GPU host's home data copy only had `119/1000` v22b manifest files. This run repaired that data coverage and launched a fair 20s/128 comparison on the v22b file set.
- **Status:** Public best remains `0.929` from `v516`. Latest visible unchanged: `v250=0.926`, `v249=0.926`, `v248=0.926`, `v247=0.925`, `v516=0.929`, `v509/v508/v507=0.927`, `v514=0.924`, `v510=0.927`. All focus queue kernels remain COMPLETE/no failure; focus-only monitor pid `32468` remains sleeping on daily cap before `v517`. PR #214 remains open; mergeability was temporarily UNKNOWN after recent pushes.
- **Required v510 check:** `bc26-v510-real-sed-bundle-blend-005` remains COMPLETE/no failure and already scored/submitted at `0.927`.
- **Data repair:** Verified local `/Volumes/ExternalSSD/data/workspace_don/kaggle_birdclef2026/data/train_audio` has all `1000/1000` v22b manifest files (`279.5 MB`). Used `rsync --files-from` to copy only missing manifest audio into `/home/yourslewis/birdclef-2026/data/` on `192.168.0.10`; remote coverage is now `1000/1000` files, `279.5 MB`.
- **v23d launch:** Added `configs/birdclef/sed_nfnet_balanced_oof_v23d_20s_128_100cls_lr1e4_ep5_v22bmanifest_fullcopy.json`, same as v23c but new experiment/output `sed-nfnet-balanced-oof-v23d-20s-128-100cls-lr1e4-ep5-v22bmanifest-fullcopy`. Launched durable fair-comparison run on GPU0 with `CUDA_VISIBLE_DEVICES=0`: parent pid `81841`, fold0 child pid `81911`, log `logs/sed_oof_v23d_nfnet_20s128_v22bmanifest_fullcopy_20260509T174431Z.log`. Initial check: fold0 running, GPU0 active (`~4.9 GB` used, ~67% util), no immediate traceback.
- **Decision:** Monitor v23d next. If v23d beats v22b and preserves v22b's low-correlation blend lift on the exact v13/v15/v22b overlap, then it becomes the cleaner candidate for a future low-weight 20s NFNet packaging variant after the current Kaggle focus queue scores.

## 2026-05-09 18:42 UTC — v23d fair-grid comparison; packaging deferred until focus scores

- **Track:** A Real SED crop/context tuning + queue monitoring. v23d completed, so this run compared it on the exact v13/v15/v22b 1000-row overlap and decided whether it is worth packaging immediately.
- **Status:** Public best remains `0.929` from `v516`. Latest visible unchanged: `v250=0.926`, `v249=0.926`, `v248=0.926`, `v247=0.925`, `v516=0.929`, `v509/v508/v507=0.927`, `v514=0.924`, `v510=0.927`. All focus queue kernels remain COMPLETE/no failure; focus-only monitor pid `32468` remains sleeping on daily cap before `v517`. PR #214 remains open; mergeability temporarily UNKNOWN after recent pushes.
- **Required v510 check:** `bc26-v510-real-sed-bundle-blend-005` remains COMPLETE/no failure and already scored/submitted at `0.927`.
- **v23d result:** `sed-nfnet-balanced-oof-v23d-20s-128-100cls-lr1e4-ep5-v22bmanifest-fullcopy` completed with `1000` OOF rows, macro AUC `0.623527` over `100` valid classes. Fold AUCs: `0.677646`, `0.693629`, `0.690017`; TorchScript size `89.87 MB`/fold.
- **Fair overlap grid:** On the exact v13/v15/v22b/v23d 1000-row overlap, baseline `0.4*v13 + 0.6*v15` is `0.657329`. Standalone AUCs: v13 `0.636878`, v15 `0.633091`, v22b `0.607845`, v23d `0.623527`. Correlation to base: v22b `0.5835`, v23d `0.5496`. Best one-model add: `25%` v23d gives `0.668613`, much stronger than best v22b-only `20%` -> `0.661518`. Best tested combo: `10%` v22b + `20%` v23d gives `0.670135`.
- **Interpretation:** v23d is a real new low-correlation SED signal and clearly better than v22b on the same file set. The result supports a future low-weight 20s/128 NFNet sidecar/bundle candidate, but immediate public packaging is deferred because the guarded Kaggle queue is already full of COMPLETE candidates (`v517/v523/v524/v525/v518` first) and `v520-v522` are the already-packaged B0 SED transfer checks waiting behind them.
- **Decision:** Do not add another public kernel before focus scores land. If the taxon bracket and queued B0/student candidates fail to improve beyond `0.929`, the next high-signal candidate should be a v23d 20s/128 NFNet bundle at conservative blend weight, probably starting around `REAL_SED_BLEND_WEIGHT=0.05`-`0.10` or a rank/OOF-informed low-weight blend rather than a standalone replacement.

## 2026-05-09 19:42 UTC — v23d bundle staged for future packaging

- **Track:** A+G SED packaging preparation + queue monitoring. No public Kaggle kernel was added because the guarded focus queue is still full/capped, but I prepared the v23d low-correlation SED artifact so it can become the next candidate quickly if queued focus/B0/student kernels miss `0.929`.
- **Status:** Public best remains `0.929` from `v516`. Latest visible unchanged: `v250=0.926`, `v249=0.926`, `v248=0.926`, `v247=0.925`, `v516=0.929`, `v509/v508/v507=0.927`, `v514=0.924`, `v510=0.927`. All focus queue kernels remain COMPLETE/no failure; focus-only monitor pid `32468` remains sleeping on daily cap before `v517`.
- **Required v510 check:** `bc26-v510-real-sed-bundle-blend-005` remains COMPLETE/no failure and already scored/submitted at `0.927`.
- **v23d bundle prep:** Built internal bundle on trainer: `artifacts/sed_bundles/sed-nfnet-v23d-20s128-lowcorr-v1/`, zip `artifacts/sed_bundles/sed-nfnet-v23d-20s128-lowcorr-v1.zip`. It contains 3 TorchScript fold models from `sed-nfnet-balanced-oof-v23d-20s-128-100cls-lr1e4-ep5-v22bmanifest-fullcopy`, manifest `sed_bundle_manifest.json`, labels, and audio config. Bundle directory size `258 MB`, zip size `238 MB`; manifest reports model total `269.611 MB` and OOF macro AUC `0.623527` over 100 valid classes.
- **Rationale:** Exact-overlap blend grid showed v23d is stronger and less correlated than v22b (`corr=0.5496` to v13/v15 base; best v23d-only add `25% -> 0.668613`; best v22b+v23d combo `0.670135`). This is the best prepared new SED sidecar signal, but public packaging remains deferred until current queued kernels score.
- **Decision:** Hold public queue. If `v517/v523/v524/v525/v518/v519/v520-v522` do not improve beyond `0.929`, next implementation should create a v23d Kaggle dataset/kernel from this bundle, starting with conservative blend weights around `0.05` and `0.10` against the v508/v516 axis.

## 2026-05-09 20:42 UTC — v23d private Kaggle dataset uploaded, kernel still held

- **Track:** A+G SED packaging preparation + queue monitoring. This run avoided adding a new public kernel/submission while the focus queue is capped, but moved the v23d low-correlation SED bundle from trainer-only artifact to a reusable private Kaggle dataset.
- **Status:** Public best remains `0.929` from `v516`. Latest visible unchanged: `v250=0.926`, `v249=0.926`, `v248=0.926`, `v247=0.925`, `v516=0.929`, `v509/v508/v507=0.927`, `v514=0.924`, `v510=0.927`. `v510` and all focus queue kernels (`v517`, `v523`, `v524`, `v525`, `v518`, `v519`, `v520`, `v521`, `v522`) are COMPLETE/no failure. Focus-only monitor pid `32468` is still sleeping on Kaggle daily cap before `v517`.
- **Implementation:** Extended `scripts/upload_kaggle_dataset_bearer.py` with `--file-description` so new SED bundle uploads no longer hard-code the old v13/v15 file description. Validation: `python3 -m py_compile scripts/upload_kaggle_dataset_bearer.py` passed.
- **Dataset upload:** Copied `sed-nfnet-v23d-20s128-lowcorr-v1.zip` from trainer to `artifacts/sed_dataset_upload/bc26-sed-nfnet-v23d-20s128-lowcorr-v1/` on the Mac. SHA256 verified as `ea11185f6b3cf03f3e2a1ef7c3c3fbd5f4d22c21abab0eaeb7bbf3f2608a47aa`; local zip size `238 MB`.
- **Kaggle artifact:** Created private dataset `yourslewis/bc26-sed-nfnet-v23d-20s128-lowcorr-v1` (URL `https://www.kaggle.com/datasets/yourslewis/bc26-sed-nfnet-v23d-20s128-lowcorr-v1`). Dataset API verification reports version `1`, status `Ready`, total bytes `269,618,235`, private `true`, last updated `2026-05-09T20:48:37.21Z`.
- **Decision:** Still hold the public kernel/queue. If the current focus queue misses `0.929`, the next implementation can create a real runnable v23d kernel using this dataset, with first candidates `REAL_SED_BLEND_WEIGHT=0.05` and/or `0.10` against the v516/v508 axis. This removes the slow dataset-upload step from the next run without burning a submission slot now.

## 2026-05-09 21:42 UTC — v526 v516+v23d SED bridge pushed and queued after focus block

- **Track:** A+G real SED packaging + Spec E taxon gate integration. Since the focus queue was still capped but the v23d dataset was ready, this run created the first runnable v23d Kaggle kernel bridge instead of another pure postprocess micro-sweep.
- **Status:** Public best remains `0.929` from `v516`. Latest visible unchanged: `v250=0.926`, `v249=0.926`, `v248=0.926`, `v247=0.925`, `v516=0.929`, `v509/v508/v507=0.927`, `v514=0.924`, `v510=0.927`. Required v510 check remains COMPLETE/no failure and already scored `0.927`. Existing focus kernels (`v517`, `v523`, `v524`, `v525`, `v518`, `v519`, `v520`, `v521`, `v522`) remain COMPLETE/no failure.
- **Implementation:** Added Kaggle kernel folder `kaggle-kernels/v526-v516-v23d-blend005/` and push helper `scripts/push_v526.py`. v526 uses the v508/v510 real-SED loading path, points to private dataset `yourslewis/bc26-sed-nfnet-v23d-20s128-lowcorr-v1`, sets `REAL_SED_BLEND_WEIGHT=0.05`, `REAL_SED_MAX_MODELS=3`, `REAL_SED_MIN_MODELS=1`, and applies the v516 taxon max gate (`floor=0.30`, `alpha=0.75`) after blending. This is effectively conservative v23d sidecar + v516 taxon-gated axis.
- **Validation:** `python3 -m py_compile` passed for the v526 kernel, push helper, and queue monitor. Pushed real Kaggle kernel `yourslewis/bc26-v526-v516-plus-v23d-sed-blend-005`, version 1. Push response had no invalid dataset/competition/kernel/model sources. Kernel completed with `submission.csv`; Kaggle output log confirms manifest found under the v23d dataset, loaded `3/3` TorchScript models, real SED prob range `0.000098` to `0.738340` (mean `0.0920`), real SED runtime `275.9s`, applied blend weight `0.05`, applied taxon gate floor `0.3` alpha `0.75`, output shape `240 x 235`, and wall time `466.0s` / `7.8 min`.
- **Queue:** Added v526 to `scripts/submit_pending_birdclef_queue.py` after `v522` and extended `FOCUS_PRIORITY_NAMES` to include it last, preserving order `v517/v523/v524/v525/v518/v519/v520/v521/v522/v526` after already-submitted v516. Restarted focus-only monitor: old pid `32468` -> new pid `11243`, log `logs/submit_pending_birdclef_queue_20260509T2142Z_focus_v526.log`. It skipped v516, attempted v517, hit daily cap with `2.0 hours` remaining, and is sleeping `7320s`.
- **Decision:** v526 is now ready but will not jump ahead of the already-complete focus/B0 candidates. If the taxon bracket/B0/student queue misses `0.929`, v526 provides the next high-signal v23d NFNet sidecar slot without additional setup.

## 2026-05-09 22:42 UTC — capped queue hold; artifact triage confirms no safe extra package

- **Track:** Queue monitoring + Spec B/D artifact triage while Kaggle daily cap blocks `v517`. No new public kernel was added because v526 was just verified/queued, the focus queue is already long, and existing pseudo/model-zoo artifacts do not justify another submission ahead of queued candidates.
- **Status:** Public best remains `0.929` from `v516`. Latest visible unchanged: `v250=0.926`, `v249=0.926`, `v248=0.926`, `v247=0.925`, `v516=0.929`, `v509/v508/v507=0.927`, `v514=0.924`, `v510=0.927`. Required v510 check remains COMPLETE/no failure. All focus kernels remain COMPLETE/no failure, including new `v526` (`yourslewis/bc26-v526-v516-plus-v23d-sed-blend-005`) with verified `submission.csv`.
- **Queue:** Focus-only monitor pid `11243`, log `logs/submit_pending_birdclef_queue_20260509T2142Z_focus_v526.log`, is alive and sleeping on cap before `v517`. It skipped submitted `v516`, attempted `v517`, received the daily allowance error (`2.0 hours from now`), and is sleeping `7320s`. Queue order remains `v517`, `v523`, `v524`, `v525`, `v518`, `v519`, `v520`, `v521`, `v522`, `v526`.
- **GPU/infrastructure:** Trainer `192.168.0.10` is reachable, but both GPUs are busy with non-BirdCLEF LRM/HSTU jobs (`~92%` and `~88%` util, high memory use). I did not stack a new BirdCLEF GPU training run.
- **Artifact triage:** Rechecked completed pseudo/model-zoo artifacts already on trainer. RegNetY pseudo-label student `pl-r2-regnety008-v508-soft-p100-5s-pretrained-lr1e4-ep20-bestval` exports a small `23.419 MB` TorchScript and has final student-teacher corr `0.98257`, but prior blend sweep did not improve linear blend over v508 and only gave a tiny rank gain; do not package. B0 ep12/ep18 5s students remain highly teacher-correlated (`~0.9868`/`0.9875`), and ep18 underperformed ep12; keep queued `v519` as the active ep12 bridge. No new RegNet/B0 sidecar is justified before queued `v518-v522` score.
- **Decision:** Hold and let monitor submit after cap reset. If `v517/v523/v524/v525` miss `0.929`, next information-rich submissions are already queued new-model candidates `v518-v522`, followed by v23d bridge `v526`; do not add more adjacent blend/gate variants until those scores land.

## 2026-05-09 23:42 UTC / 2026-05-10 00:03 UTC — UTC reset submissions landed

- **Track:** Queue monitoring / submission execution. With cap reset close, this run waited through the UTC reset and verified the focus-only monitor submitted the next five COMPLETE kernels rather than adding new variants.
- **Pre-reset status:** Public best remained `0.929` from `v516`; latest visible unchanged. Required `v510` remained COMPLETE/no failure and already scored `0.927`. Focus kernels `v517`, `v523`, `v524`, `v525`, `v518`, `v519`, `v520`, `v521`, `v522`, `v526` all remained COMPLETE/no failure.
- **Submissions after reset:** Monitor pid `11243` / log `logs/submit_pending_birdclef_queue_20260509T2142Z_focus_v526.log` successfully submitted five rows after reset: `v517` ref `52497570`, `v523` ref `52497585`, `v524` ref `52497605`, `v525` ref `52497631`, and `v518` ref `52497648`. Kaggle submissions list shows these five as `pending`.
- **Cap behavior:** The monitor then attempted `v519`, hit the daily allowance error (`23 hours from now`), and is sleeping `82920s`. Remaining focus queue for the next reset is `v519`, `v520`, `v521`, `v522`, `v526`.
- **Infrastructure:** Trainer `192.168.0.10` was reachable, but GPUs were occupied by non-BirdCLEF LRM/HSTU work, so no new BirdCLEF GPU job was launched.
- **Decision:** Hold. Wait for public scores on `v517/v523/v524/v525/v518`. If none beat `0.929`, the next queued day already covers B0/student and v23d SED signal via `v519-v522/v526` without adding more variants.

## 2026-05-10 00:42 UTC — v517 reaches 0.930; v527 alpha0.375 follow-up queued

- **Track:** Queue monitoring + targeted Spec E follow-up from a real LB improvement. This run did not add broad postprocess sweeps; it added one midpoint around the newly validated taxon-gate optimum while preserving the already queued new-model block.
- **Status:** Public best improved to `0.930` from `v517` (`taxon max gate floor0.30 alpha0.50 + v508 axis`). Other scored reset rows: `v523=0.928`, `v524=0.929`, `v525=0.929`; `v518` remains pending. Previous best `v516` remains `0.929`; `v510` remains COMPLETE/no failure and scored `0.927`.
- **Interpretation:** The taxon-gate alpha curve now shows softer gating is better than the original `alpha=0.75`: `alpha0.875 -> 0.928`, `alpha0.75 -> 0.929`, `alpha0.50 -> 0.930`. Floor changes at alpha0.75 tied but did not improve (`floor0.20=0.929`, `floor0.40=0.929`). The next informative single point is a lower-alpha midpoint, not another floor change.
- **Implementation:** Added `kaggle-kernels/v527-taxon-max-gate-alpha0375/` copied from v517 and changed only `TAXON_MAX_GATE_ALPHA=0.50 -> 0.375` (floor remains `0.30`, v508 axis unchanged). Added `scripts/push_v527.py`.
- **Validation:** `python3 -m py_compile` passed for v527 script, push helper, and queue monitor. Pushed real Kaggle kernel `yourslewis/bc26-v527-taxon-max-gate-a0375`, version 1; push returned no invalid sources. Kernel status COMPLETE/no failure with `submission.csv`; output log confirms taxon gate floor `0.3`, alpha `0.375`, output shape `240 x 235`, wall time `160.9s` / `2.7 min`.
- **Queue:** Added v527 after v526 in `scripts/submit_pending_birdclef_queue.py` and extended focus priority to `v516/v517/v523/v524/v525/v518/v519/v520/v521/v522/v526/v527`. Restarted monitor old pid `11243` -> new pid `32518`, log `logs/submit_pending_birdclef_queue_20260510T0042Z_focus_v527.log`. It skipped already-submitted v516/v517/v523/v524/v525/v518, attempted v519, hit daily cap (`23 hours from now`), and is sleeping `82920s`.
- **Decision:** Hold. Remaining next-reset queue is `v519`, `v520`, `v521`, `v522`, `v526`; v527 is queued after those and will not displace the pending new-model/v23d sidecar block.

## 2026-05-10 01:42 UTC — v518 scored; v28 fuller observed-class NFNet SED launched

- **Track:** A Real SED frame/event model signal. The public queue is capped until the next UTC reset, so this run used the open training window for a higher-upside SED coverage experiment rather than adding another adjacent postprocess/gate submission.
- **Status:** Public best is now `0.930` from `v517` (`taxon max gate floor0.30 alpha0.50 + v508 axis`). Latest reset scores: `v523=0.928`, `v524=0.929`, `v525=0.929`, and `v518=0.927`. Previous best `v516=0.929`; required `v510` remains COMPLETE/no failure and scored `0.927`.
- **Queue/monitor:** Focus monitor pid `32518`, log `logs/submit_pending_birdclef_queue_20260510T0042Z_focus_v527.log`, is alive. It skipped already-submitted `v516/v517/v523/v524/v525/v518`, attempted `v519`, hit Kaggle daily cap, and is sleeping. Remaining next-reset queue is `v519`, `v520`, `v521`, `v522`, `v526`, then `v527`; no duplicate submissions were added.
- **Hypothesis:** v23d proved that 20s/128 NFNet is a low-correlation real SED signal on the v13/v15/v22b 1000-row overlap, but it only trained `100` classes. A fuller observed-primary-label version with similar compute (`206` observed train labels, `5` files/class, ~`1030` files) may improve sidecar coverage and become a stronger bundle candidate for the 0.95 path.
- **Configs added:** `configs/birdclef/sed_nfnet_allobserved_v28_smoke_20s_128_12cls_ep1.json` and `configs/birdclef/sed_nfnet_allobserved_v28_20s_128_206cls_5per_ep5.json`.
- **Smoke validation:** Copied configs to trainer `192.168.0.10` and ran a 2-fold smoke on GPU0: `python scripts/birdclef_sed_oof_runner.py --base-config configs/birdclef/sed_nfnet_allobserved_v28_smoke_20s_128_12cls_ep1.json --output-root artifacts/sed_oof/sed-nfnet-allobserved-v28-smoke-20s-128-12cls-ep1 --n-folds 2`. Smoke completed with `60` OOF rows, `12` valid classes, fold AUCs `0.570749` and `0.515997`, aggregate macro AUC `0.480000`, and TorchScript exports `89.87 MB`/fold. This validates decode/training/export on the fuller-class config family.
- **Scaled launch:** Started durable 3-fold OOF run on trainer GPU0 with `CUDA_VISIBLE_DEVICES=0`: `python scripts/birdclef_sed_oof_runner.py --base-config configs/birdclef/sed_nfnet_allobserved_v28_20s_128_206cls_5per_ep5.json --output-root artifacts/sed_oof/sed-nfnet-allobserved-v28-20s-128-206cls-5per-ep5 --n-folds 3`. Log: `logs/sed_oof_v28_nfnet_allobserved_20s128_206cls5per_ep5_20260510T014737Z.log`; pid file: `logs/sed_oof_v28_nfnet_allobserved_20s128_206cls5per_ep5.pid`. Follow-up check: fold0 completed with macro AUC `0.622521` over `154` valid classes after 5 epochs, fold1 is running, no traceback, GPU0 active.
- **Next step:** Monitor v28 to completion. If OOF macro AUC/correlation is competitive with v23d, build a v28 SED bundle and test a conservative low-weight Kaggle bridge against the v517/v516 taxon-gated axis after the queued `v519-v522/v526/v527` slots score.

## 2026-05-10 02:42 UTC — v28 completed; v29 broader 10/class NFNet SED launched

- **Track:** A Real SED frame/event model signal + OOF model-selection. Continued the higher-upside SED path while Kaggle submission quota remains capped.
- **Status:** Public best remains `0.930` from `v517` (`taxon max gate floor0.30 alpha0.50 + v508 axis`). Latest scored reset row `v518=0.927`; `v523=0.928`, `v524=0.929`, `v525=0.929`, previous `v516=0.929`. Required `v510` remains COMPLETE/no failure and scored `0.927`.
- **Queue/monitor:** Focus monitor pid `32518`, log `logs/submit_pending_birdclef_queue_20260510T0042Z_focus_v527.log`, remains alive and sleeping after hitting daily cap on `v519`. Focus candidates `v519/v520/v521/v522/v526/v527` remain COMPLETE/no failure and queued for future resets; no duplicate submissions added.
- **v28 result:** `sed-nfnet-allobserved-v28-20s-128-206cls-5per-ep5` completed 3-fold OOF on trainer. Aggregate: `875` OOF rows, macro AUC `0.552257` over `175` valid classes. Fold AUCs were stronger individually (`0.622521`, `0.637635`, `0.639385`) but cross-fold aggregate is weak, likely from too few examples per class and sparse held-out positives.
- **v28 OOF comparison:** Key-normalized overlap comparisons show v28 is low-correlation but not a good bundle candidate. On overlap with v13 (`238` rows / `96` valid classes), v13 AUC `0.658013`, v28 `0.559528`, best blend is only `10%` v28 for `0.659112`. Against v15 (`267` rows / `115` valid classes), v15 `0.658214`, v28 `0.546511`, best `10%` v28 gives `0.658525`. Against v23d (`238` rows / `96` classes), v23d `0.629344`, v28 `0.559528`, best is `0%` v28. Decision: do not package/submit v28.
- **Hypothesis for v29:** v28's broad-class intent is still right, but `5` files/class undertrained the classes. Launch a cleaner 20s/128 NFNet with `10` files/class and `min_files_per_class=10`, matching v15's data density while testing the 20s/128 context that v23d suggested may be complementary.
- **Config added:** `configs/birdclef/sed_nfnet_allobserved_v29_20s_128_181cls_10per_ep5.json` with NFNet-L0, 20s crops, 128 mels, focal BCE gamma `1.5`, label smoothing `0.01`, mixup `0.2`, sqrt positive weighting, `files_per_class=10`, `min_files_per_class=10`, `max_files=1810`, `epochs=5`, `batch_size=1`.
- **Scaled launch:** Copied config to trainer and launched durable 3-fold OOF on GPU0: `python scripts/birdclef_sed_oof_runner.py --base-config configs/birdclef/sed_nfnet_allobserved_v29_20s_128_181cls_10per_ep5.json --output-root artifacts/sed_oof/sed-nfnet-allobserved-v29-20s-128-181cls-10per-ep5 --n-folds 3`. Log: `logs/sed_oof_v29_nfnet_allobserved_20s128_181cls10per_ep5_20260510T024625Z.log`; runner pid `370557`. Initial check: fold0 started, no traceback.
- **Next step:** Monitor v29. If it beats v28 and is competitive with v13/v15/v23d on OOF/overlap blend, build a v29 or v15+v29 SED bundle candidate; otherwise keep v23d/v13/v15 as the only package-worthy NFNet sidecars and wait for queued Kaggle slots to score.

## 2026-05-10 03:42 UTC — v29 fold2 retry, v528 v517+v29 SED bridge pushed

- **Track:** A+G real SED OOF validation + Kaggle packaging. v29 had completed folds 0/1 strongly but fold2 had failed with return code `-11` and empty stdout/stderr, so this run first diagnosed and salvaged the OOF before deciding whether to package.
- **Status:** Public best remains `0.930` from `v517` (`taxon max gate floor0.30 alpha0.50 + v508 axis`). Latest scored rows unchanged: `v518=0.927`, `v523=0.928`, `v524=0.929`, `v525=0.929`, previous `v516=0.929`. Required `v510` remains COMPLETE/no failure and scored `0.927`.
- **Queue/monitor before changes:** Focus monitor pid `32518`, log `logs/submit_pending_birdclef_queue_20260510T0042Z_focus_v527.log`, was alive and sleeping after cap on `v519`. Existing queued kernels `v519/v520/v521/v522/v526/v527` remained COMPLETE/no failure.
- **v29 failure diagnosis:** `sed-nfnet-allobserved-v29-20s-128-181cls-10per-ep5` had fold0 AUC `0.714199` over `167` valid classes and fold1 AUC `0.710685` over `169`, but fold2 failed with `returncode=-11`, no stdout/stderr, and no fold2 artifacts. Trainer GPUs were heavily occupied by non-BirdCLEF HSTU/LRM processes, suggesting a transient CUDA/runtime crash rather than data/config failure.
- **v29 fix:** Retried only fold2 directly on `CUDA_VISIBLE_DEVICES=1`: `python scripts/birdclef_sed_pilot_train.py --config artifacts/sed_oof/sed-nfnet-allobserved-v29-20s-128-181cls-10per-ep5/config_fold2.json`. Retry succeeded with fold2 macro AUC `0.696939` over `166` valid classes, TorchScript `89.87 MB`, and `holdout_predictions.npz`.
- **v29 aggregate:** Manually aggregated fold0/fold1/fold2 into `artifacts/sed_oof/sed-nfnet-allobserved-v29-20s-128-181cls-10per-ep5/oof_predictions.npz` and repaired `oof_summary.json`. Aggregate OOF: `1700` rows, macro AUC `0.649309` over `170` valid classes.
- **v29 overlap comparisons:** Key-normalized OOF comparisons show v29 is package-worthy. Vs v13 on `497` overlap rows / `100` classes: v13 `0.655017`, v29 `0.640396`, flat Pearson `0.3322`, best blend `50%` v29 -> `0.681094`. Vs v15 on `539` rows / `132` classes: v15 `0.667781`, v29 `0.663040`, Pearson `0.3499`, best blend `40%` v29 -> `0.703605`. Vs v23d on `497` rows / `100` classes: v29 is complementary and best blend is `50%` v29 -> `0.674447`. Decision: v29 is much stronger than v28 and worth staging as a public-kernel sidecar after the existing queue.
- **Dataset packaging:** Built trainer bundle `artifacts/sed_bundles/sed-nfnet-v29-20s128-broad181-v1/` and zip `artifacts/sed_bundles/sed-nfnet-v29-20s128-broad181-v1.zip`. Bundle has 3 TorchScript models, manifest, labels, audio config; manifest total model size `269.611 MB`, zip size `238 MB`, SHA256 `6ef5621398b8c54025be995e0759c401c26b557ad1ddc576248b9f9d641d40c4`.
- **Private Kaggle dataset:** Uploaded and verified private dataset `yourslewis/bc26-sed-nfnet-v29-20s128-broad181-v1`, version `1`, status `Ready`, total bytes `269,617,910`, last updated `2026-05-10T03:52:47.51Z`.
- **v528 implementation:** Added Kaggle kernel `kaggle-kernels/v528-v517-v29-blend005/` and push helper `scripts/push_v528.py`. v528 uses the v508/v526 real-SED loading path, points to dataset `yourslewis/bc26-sed-nfnet-v29-20s128-broad181-v1`, loads `3/3` TorchScript models, uses `REAL_SED_BLEND_WEIGHT=0.05`, and applies the current-best v517 taxon gate (`floor=0.30`, `alpha=0.50`) after blending.
- **v528 validation:** `py_compile` passed. Pushed real Kaggle kernel `yourslewis/bc26-v528-v517-plus-v29-sed-blend-005`, version 1; push had no invalid sources. Kernel COMPLETE/no failure with `submission.csv`; output log confirms manifest found under v29 dataset, loaded `3/3` models, real SED prob range `0.000068` to `0.682857` (mean `0.1010`), real SED runtime `244.1s`, blend weight `0.05`, taxon gate floor `0.3` alpha `0.5`, output `240 x 235`, wall time `410.7s` / `6.8 min`.
- **Queue update:** Added v528 to `scripts/submit_pending_birdclef_queue.py` after v527 and extended focus priority to `v516/v517/v523/v524/v525/v518/v519/v520/v521/v522/v526/v527/v528`. Restarted monitor old pid `32518` -> new pid `59796`, log `logs/submit_pending_birdclef_queue_20260510T0409Z_focus_v528.log`; it skipped submitted `v516/v517/v523/v524/v525/v518`, attempted `v519`, hit cap with `19 hours from now`, and sleeps `68520s`.
- **Next step:** Let queued `v519-v522/v526/v527/v528` score in order. If v526/v528 show gains, use v29/v23d OOF grids to test a slightly larger SED blend or v15+v29 combined bundle; otherwise keep v29 as the strongest offline SED signal and shift to OOF-calibrated blending rather than more raw sidecar submissions.


## 2026-05-10 04:42 UTC - capped queue hold; multi-member OOF blend grid tool added

- Track: F post-new-signal OOF retuning support plus A/G queue monitoring. No additional public kernel was added because the focus queue is already full/capped and v528 is complete/queued behind v519-v522/v526/v527.
- Status: Public best remains 0.930 from v517 (taxon max gate floor0.30 alpha0.50 + v508 axis). Latest visible scored rows unchanged: v518=0.927, v525=0.929, v524=0.929, v523=0.928, v516=0.929, v510=0.927. Required v510 remains COMPLETE/no failure.
- Queue/monitor: Focus monitor log logs/submit_pending_birdclef_queue_20260510T0409Z_focus_v528.log shows submitted/skipped v516/v517/v523/v524/v525/v518, then attempted v519 and hit Kaggle daily cap (19 hours from now). A sleeping monitor process is present. Focus kernels v519/v520/v521/v522/v526/v527/v528 are COMPLETE/no failure. No duplicate submissions were added.
- Infrastructure: Tried to inspect trainer 192.168.0.10, but SSH/SCP timed out or closed during banner exchange in this run. This appears transient while the host is busy; no new GPU training was launched.
- Implementation: Added scripts/birdclef_oof_blend_grid.py, a reusable multi-member OOF blend grid tool. It accepts repeated --member name:path, aligns rows by stable train_audio/<label>/<file> key, validates label consistency, reports single-model macro AUC, pairwise flat Pearson / mean absolute difference, and simplex blend-grid top weights. This replaces ad-hoc two-model comparison for v13/v15/v23d/v29 and future SED/zoo/pseudo artifacts.
- Validation: python3 -m py_compile scripts/birdclef_oof_blend_grid.py passed. A synthetic two-member NPZ smoke test using the project venv completed, wrote /tmp/birdclef_blend_grid_test/out.json, and verified overlap/top-k output. Attempting to copy/run the tool on trainer for the real v13/v15/v23d/v29 grid failed only because SSH closed; rerun when trainer is reachable.
- Next step: When trainer SSH returns, run scripts/birdclef_oof_blend_grid.py on v13/v15/v23d/v29 with --step 0.1 or 0.05, then decide whether the next package-worthy candidate should be a combined v15+v29 or v23d+v29 bundle. Meanwhile let the capped public queue submit v519-v522/v526/v527/v528 in order.

## 2026-05-10 05:55 UTC — monitor recovered; trainer SSH still banner-times out

- **Track:** Queue monitoring / Spec F+G support after new SED signals. I did not add another public Kaggle kernel because the focus queue is already capped/full and v528 is complete/queued behind v519-v522/v526/v527.
- **Submission state:** Latest five BirdCLEF submissions via Bearer API are all complete: `v518=0.927`, `v525=0.929`, `v524=0.929`, `v523=0.928`, `v517=0.930`. Current public best remains `0.930` from `v517` (taxon max gate floor0.30 alpha0.50 + v508 axis).
- **Required v510 check:** `yourslewis/bc26-v510-real-sed-bundle-blend-005` remains `COMPLETE` with no failure message and has already scored `0.927`, so no SED bundle mount/TorchScript/output failure is open.
- **Queued kernels:** `v519`, `v520`, `v521`, `v522`, `v526`, `v527`, and `v528` all report `COMPLETE` with no failure message. Queue order remains `v519 -> v520 -> v521 -> v522 -> v526 -> v527 -> v528` after already-submitted focus candidates.
- **Monitor fix:** The prior focus monitor log `logs/submit_pending_birdclef_queue_20260510T0409Z_focus_v528.log` showed it sleeping after the v519 daily-cap error, but no matching process was alive. Restarted the focus-only monitor with `BIRDCLEF_QUEUE_STOP_AFTER_FOCUS=1`: pid `75964`, log `logs/submit_pending_birdclef_queue_20260510T055002Z_focus_restart_v528.log`. It skipped already-submitted `v516/v517/v523/v524/v525/v518`, attempted `v519`, hit the expected daily cap (`18 hours from now`), and is now sleeping `64920s` before retry. This avoids duplicate submissions and keeps the focus queue guarded against legacy fallthrough.
- **Validation:** `python3 -m py_compile scripts/submit_pending_birdclef_queue.py scripts/birdclef_oof_blend_grid.py` passed. Monitor process recheck shows pid `75964` alive under `ppid=1`.
- **Infrastructure:** Trainer `192.168.0.10` is pingable (`0%` packet loss), but SSH still times out during banner exchange even with a 30s connect timeout. Because of that I could not run the new multi-member OOF blend grid on remote v13/v15/v23d/v29 artifacts in this run.
- **Decision:** Hold public queue. Next useful action once SSH recovers is to run `scripts/birdclef_oof_blend_grid.py` on v13/v15/v23d/v29 with `--step 0.1` or `0.05` and decide whether a combined v15+v29 or v23d+v29 bundle is worth packaging after the queued submissions score.


## 2026-05-10 06:55 UTC - capped queue hold; local OOF blend grids while trainer SSH is blocked

- Track: Spec F post-new-signal OOF retuning support plus A/G queue monitoring. I did not add or push a new public Kaggle kernel because the daily quota is exhausted and the focus queue already has diverse complete candidates v519-v522/v526-v527/v528 waiting.
- Submission state: Latest five BirdCLEF submissions via Bearer API are unchanged and complete: v518=0.927, v525=0.929, v524=0.929, v523=0.928, v517=0.930. Current public best remains 0.930 from v517.
- Required v510 check: yourslewis/bc26-v510-real-sed-bundle-blend-005 remains COMPLETE with no failure message; already scored 0.927.
- Queued kernels: v519, v520, v521, v522, v526, v527, and v528 all report COMPLETE with no failure message. Focus monitor pid 75964, log logs/submit_pending_birdclef_queue_20260510T055002Z_focus_restart_v528.log, is alive and sleeping after the expected v519 daily-cap response.
- Infrastructure: 192.168.0.10 is still pingable but SSH times out during banner exchange, so remote v23d/v29 OOF grid remains blocked. No blind GPU job was launched.
- Local grid work: Ran scripts/birdclef_oof_blend_grid.py with /Users/yourslewis/.openclaw/workspace-don/kaggle/playground-series-s6e3/.venv/bin/python on locally available OOF artifacts:
  - artifacts/blend_grids/v13_v15_step005.json: overlap 1000, valid classes 100; singles v13=0.636878, v15=0.633091; pairwise flat Pearson 0.2946; best grid blend v13=0.40, v15=0.60, macro AUC 0.657329.
  - artifacts/blend_grids/v13_b3v16_regv20_step01.json: overlap 1000, valid classes 100; singles v13=0.636878, b3v16=0.506158, regv20=0.506402; best grid blend v13=0.90, b3v16=0.10, regv20=0.00, macro AUC 0.638184. B3 has low correlation to v13 (0.2723) but only a tiny blend lift, so it is not package-worthy yet.
  - artifacts/blend_grids/v23_v26_step005.json: overlap 2053, valid classes 206; singles v23=0.859564, v26=0.883556; pairwise flat Pearson 0.5912; best grid blend v23=0.40, v26=0.60, macro AUC 0.903667. This is the strongest local signal this run and supports preparing a combined v23+v26 package if queued v520/v521/v522 scores justify another slot.
- Validation: The grid script executed successfully on three real OOF combinations. The JSON outputs are under ignored artifacts/blend_grids/; key metrics are copied here for durable tracking.
- Next step: Let v519-v522/v526-v528 score. If v520/v521/v522 show any public lift or safe tie, prepare a combined v23+v26 B0 SED bundle/kernel using OOF-informed weights near v23=0.40, v26=0.60; if v23/v26 underperform publicly, wait for trainer SSH and prioritize v23d/v29 OOF grid before adding more kernels.


## 2026-05-10 07:55 UTC - OOF grid tool portability fix during capped queue

- Track: Spec F OOF retuning infrastructure while A/G public queue is capped. No new public kernel was added; v519-v522/v526-v528 remain the active complete queue.
- Status: Latest submissions are unchanged: v518=0.927, v525=0.929, v524=0.929, v523=0.928, v517=0.930. Current public best remains 0.930 from v517. Required v510 remains COMPLETE/no failure and already scored 0.927.
- Queue/monitor: v519, v520, v521, v522, v526, v527, and v528 all report COMPLETE/no failure. Focus monitor pid 75964 is alive and sleeping after the expected v519 daily-cap response.
- Infrastructure: 192.168.0.10 remains pingable but SSH still times out during banner exchange, so no remote GPU/OOF work was launched.
- Implementation: Hardened scripts/birdclef_oof_blend_grid.py so it no longer requires scikit-learn. It now uses sklearn roc_auc_score when available and falls back to a numpy rank-sum ROC-AUC implementation when sklearn is absent. This makes the blend-grid tool runnable under the repo's default system python as well as the Kaggle project venv.
- Validation: python3 -m py_compile scripts/birdclef_oof_blend_grid.py passed. Then system python3, which previously failed with ModuleNotFoundError: sklearn, successfully ran the real v23+v26 grid and reproduced the same result: singles v23=0.859564 and v26=0.883556, pairwise Pearson 0.5912, best v23=0.40/v26=0.60 with macro AUC 0.903667 over 2053 rows / 206 valid classes.
- Next step: Keep holding queue until reset. If v520/v521/v522 validate the B0 SED axis publicly, package a combined v23+v26 bundle/kernel with OOF-informed 0.40/0.60 weighting; otherwise wait for trainer SSH and prioritize v23d/v29 grid before spending more slots.


## 2026-05-10 08:55 UTC - staged v23+v26 combined SED bundle smoke while queue is capped

- Track: Spec A+G packaging prep plus Spec F OOF-informed retuning. I did not push a new Kaggle kernel/dataset because v519-v522/v526-v528 remain complete and queued behind the daily cap, and v520/v521/v522 public scores should decide whether another B0 SED slot is justified.
- Status: Latest submissions remain v518=0.927, v525=0.929, v524=0.929, v523=0.928, v517=0.930. Current public best remains 0.930 from v517. Required v510 remains COMPLETE/no failure and already scored 0.927.
- Queue/monitor: v519, v520, v521, v522, v526, v527, and v528 all report COMPLETE/no failure. Focus monitor pid 75964 is alive and sleeping after the expected v519 daily-cap response.
- Infrastructure: 192.168.0.10 is still pingable but SSH times out during banner exchange, so remote v23d/v29 OOF grid remains blocked.
- Implementation: Added scripts/birdclef_sed_combine_bundles.py, a reusable utility that combines existing portable SED bundle dirs or zip files with explicit weights, rewrites per-model manifest weights, copies TorchScript files into a single output bundle, and optionally emits a Kaggle-uploadable zip.
- Artifact prep: Used the new combiner to stage artifacts/sed_bundles/sed-b0-v23v26-oofblend040060-v1 from existing packaged B0 SED bundles v23 and v26 using OOF-grid weights v23=0.40 and v26=0.60. The combined manifest has 6 TorchScript models, 234 classes, total model size 92.328 MB, model weight sum 1.0, and zip artifacts/sed_bundles/sed-b0-v23v26-oofblend040060-v1.zip is 84.099 MB with sha256 abd60fbf85023406f50c684ffa7ac7d4703e38fe2f243f401a2570523a9bf7b9.
- Smoke validation: python3 -m py_compile scripts/birdclef_sed_combine_bundles.py passed. Ran scripts/birdclef_sed_soundscape_infer.py on one real train soundscape with the combined bundle using CPU, batch size 4, and 2 torch threads. It loaded 6/6 models and wrote soundscape_smoke_submission.csv with shape 12 x 235, no NaNs, probability range 0.021154 to 0.405027, mean 0.120202, and runtime 4.308 sec/file.
- Next step: If v520/v521/v522 produce a public safe tie or lift, upload the staged v23+v26 zip as a private Kaggle dataset and push a real kernel candidate blending it into the v517/v508 axis. If those B0 SED submissions underperform, keep the bundle staged but wait for trainer SSH and v23d/v29 OOF grid before using another public slot.


## 2026-05-10 09:55 UTC - 5-file v23+v26 combined bundle smoke; queue still capped

- Track: Spec A+G inference packaging validation while public queue is capped. No new Kaggle dataset/kernel was pushed because v519-v522/v526-v528 are already complete and queued, and v520/v521/v522 scores should decide whether the staged B0 SED combined bundle deserves the next slot.
- Status: Latest submissions remain unchanged: v518=0.927, v525=0.929, v524=0.929, v523=0.928, v517=0.930. Current public best remains 0.930 from v517. Required v510 remains COMPLETE/no failure and already scored 0.927.
- Queue/monitor: v519, v520, v521, v522, v526, v527, and v528 all report COMPLETE/no failure. Focus monitor pid 75964 is alive and sleeping after the expected v519 daily-cap response.
- Infrastructure: 192.168.0.10 is still pingable but SSH times out during banner exchange; no remote job was launched.
- Smoke validation: Expanded the combined v23+v26 bundle smoke from 1 to 5 real train soundscapes using artifacts/sed_bundles/sed-b0-v23v26-oofblend040060-v1/sed_bundle_manifest.json, CPU, batch size 4, and 2 torch threads. It loaded 6 models and wrote soundscape_smoke5_submission.csv plus soundscape_smoke5_probs.npz.
- Smoke result: 5 files, 60 rows, output shape 60 x 235, no NaNs, probability range 0.016877 to 0.486025, mean 0.126975, runtime 16.430 sec total / 3.286 sec per file. First row BC2026_Train_0001_S08_20250606_030007_5, last row BC2026_Train_0005_S08_20250607_070007_60.
- Interpretation: The staged v23+v26 0.40/0.60 bundle is operationally safe and fast enough locally; keep it ready but do not spend a public slot until the queued v520/v521/v522 public scores validate the B0 SED axis.
- Next step: Let the focus monitor submit v519 after UTC reset, then v520-v522/v526-v528 in order. If v520/v521/v522 tie/lift, upload the v23+v26 zip and push a real v529-style kernel; otherwise wait for trainer SSH and v23d/v29 OOF grid.


## 2026-05-10 10:55 UTC - v23+v26 combined SED private dataset uploaded; queue still capped

- Track: Spec A+G inference packaging prep while public queue is capped. No new public Kaggle kernel was pushed because v519-v522/v526-v528 remain complete and queued; v520/v521/v522 public scores should still decide whether to spend the next submission slot on this B0 SED combined artifact.
- Status: Latest submissions remain unchanged: v518=0.927, v525=0.929, v524=0.929, v523=0.928, v517=0.930. Current public best remains 0.930 from v517. Required v510 remains COMPLETE/no failure and already scored 0.927.
- Queue/monitor: v519, v520, v521, v522, v526, v527, and v528 all report COMPLETE/no failure. Focus monitor pid 75964 is alive and sleeping after the expected v519 daily-cap response.
- Dataset upload: Created private Kaggle dataset yourslewis/bc26-sed-b0-v23v26-oofblend040060-v1 from artifacts/sed_bundles/sed-b0-v23v26-oofblend040060-v1.zip. Upload response status was Ok, URL https://www.kaggle.com/datasets/yourslewis/bc26-sed-b0-v23v26-oofblend040060-v1.
- Dataset verification: Kaggle dataset status is READY. File listing shows 7 extracted files: six TorchScript models under sed-b0-v23v26-oofblend040060-v1/models/ at 15,387,928 bytes each plus sed_bundle_manifest.json at 7,858 bytes. This confirms the dataset is ready for a future kernel mount.
- Infrastructure: 192.168.0.10 remains pingable but SSH still times out during banner exchange, so remote v23d/v29 OOF grid remains blocked.
- Decision: Keep the dataset staged but do not push a v529 kernel yet. If v520/v521/v522 tie or lift publicly, the next run can immediately create/push a v529-style kernel pointing at dataset slug bc26-sed-b0-v23v26-oofblend040060-v1; otherwise wait for trainer SSH and v23d/v29 grid before spending another slot.


## 2026-05-10 11:55 UTC - v529 v517 plus v23/v26 SED kernel pushed and queued after v528

- Track: Spec A+G real SED packaging and queue management. I promoted the already-staged v23+v26 B0 SED combined dataset into a real Kaggle kernel candidate, but placed it after the existing focus queue so it cannot jump ahead of v519-v522/v526-v528.
- Status: Latest submissions remain unchanged: v518=0.927, v525=0.929, v524=0.929, v523=0.928, v517=0.930. Current public best remains 0.930 from v517. Required v510 remains COMPLETE/no failure and already scored 0.927.
- v529 implementation: Added kaggle-kernels/v529-v517-v23v26-blend005/ and scripts/push_v529.py. v529 copies the v528/v517 taxon-gated real-SED path, points to private dataset yourslewis/bc26-sed-b0-v23v26-oofblend040060-v1, uses REAL_SED_BLEND_WEIGHT=0.05, REAL_SED_MAX_MODELS=3, REAL_SED_MIN_MODELS=1, and applies the v517 taxon max gate after blending.
- Push: Pushed real Kaggle kernel yourslewis/bc26-v529-v517-plus-v23v26-sed-blend-005, version 1. Push returned no invalid dataset/competition/kernel/model sources. Initial status is RUNNING with no failure message; no completion/output yet at final check.
- Queue update: Added v529 to scripts/submit_pending_birdclef_queue.py after v528 and extended focus priority to include v529. Restarted focus-only monitor old pid 75964 -> new pid 31553, log logs/submit_pending_birdclef_queue_20260510T114618Z_focus_v529.log. It skipped already-submitted v516/v517/v523/v524/v525/v518, attempted v519, hit the expected daily cap with 12 hours remaining, and sleeps 43320 seconds.
- Validation: python3 -m py_compile passed for kaggle-kernels/v529-v517-v23v26-blend005/script.py, scripts/push_v529.py, and scripts/submit_pending_birdclef_queue.py. The v23+v26 dataset was already verified READY in the previous run.
- Next step: Monitor v529 completion/output log for manifest found, loaded models, applied real SED blend, v517 taxon gate, and submission.csv. Let the focus monitor submit v519 first after reset; v529 is intentionally queued behind v520-v522/v526-v528.


## 2026-05-10 12:55 UTC - v529 completed and real SED path verified

- Track: Spec A+G real SED packaging verification plus queue monitoring. No new candidate was added; this run focused on verifying the v529 kernel pushed in the previous run and keeping the capped queue healthy.
- Submission state: Latest submissions remain unchanged: v518=0.927, v525=0.929, v524=0.929, v523=0.928, v517=0.930. Current public best remains 0.930 from v517. Required v510 remains COMPLETE/no failure and already scored 0.927.
- v529 status: Kaggle kernel yourslewis/bc26-v529-v517-plus-v23v26-sed-blend-005 version 1 is COMPLETE with no failure message and output includes submission.csv.
- v529 output verification: log confirms manifest found under dataset yourslewis/bc26-sed-b0-v23v26-oofblend040060-v1, loaded 3/6 real SED TorchScript models, real SED prob range 0.009794 to 0.895201 with mean 0.1124, real SED runtime 30.3s, applied real SED blend weight 0.05, applied v517 taxon max gate floor 0.3 alpha 0.5, final prob range 0.013733 to 0.937133 with mean 0.3911, wrote submission.csv shape 240 x 235, wall time 154.3s / 2.6 min. The usual TensorFlow CUDA 303 stderr appears but does not affect successful ONNX/CPU execution.
- Queue/monitor: Focus monitor pid 31553, log logs/submit_pending_birdclef_queue_20260510T114618Z_focus_v529.log, remains alive and sleeping after the expected v519 daily-cap response. Queue order keeps v529 after v528, so it will not jump ahead of v519-v522/v526-v528.
- Infrastructure: 192.168.0.10 was not reachable in this check (ping 100 percent packet loss; SSH operation timed out), so remote v23d/v29 OOF grid remains blocked.
- Next step: Let the monitor submit v519 first after reset, then v520-v522/v526-v528/v529. If v529 eventually scores safely or improves, consider a higher B0 v23/v26 blend weight only after v520-v522/v529 public scores are known; otherwise prioritize v23d/v29 OOF grid when trainer returns.


## 2026-05-10 13:55 UTC - remote v13/v15/v23d/v29 OOF blend grid completed

- Track: Spec F post-new-signal OOF retuning plus A/G queue monitoring. No new public kernel was added because v519-v522/v526-v528/v529 are already complete and queued behind the daily cap.
- Status: Latest submissions remain unchanged: v518=0.927, v525=0.929, v524=0.929, v523=0.928, v517=0.930. Current public best remains 0.930 from v517. Required v510 remains COMPLETE/no failure; v529 also remains COMPLETE/no failure.
- Queue/monitor: Focus monitor pid 31553, log logs/submit_pending_birdclef_queue_20260510T114618Z_focus_v529.log, is alive and sleeping after the expected v519 daily-cap response. Queue order keeps v529 after v528.
- Infrastructure: Trainer 192.168.0.10 became reachable again. Copied the current sklearn-optional scripts/birdclef_oof_blend_grid.py to /tmp on trainer and ran it with ~/kaggle_envs/s6e3/bin/python.
- OOF grid: Ran a four-member NFNet SED grid over v13, v15, v23d, and v29 on 497 overlapping rows / 100 valid classes. Step 0.10 best was v13=0.20, v15=0.30, v23d=0.20, v29=0.30 with macro AUC 0.709900. Step 0.05 improved slightly to v13=0.15, v15=0.30, v23d=0.25, v29=0.30 with macro AUC 0.710048.
- Singles on the 497-row overlap: v13=0.655017, v15=0.662568, v23d=0.641578, v29=0.640396. Pairwise correlations show useful diversity: v13-v15 0.2962, v15-v23d 0.2449, v15-v29 0.3487, v23d-v29 0.3879; v13-v23d is higher at 0.6969.
- Interpretation: The strongest NFNet blend is a balanced multi-member stack, not a two-member pair. If current public queue misses, next package-worthy candidate should be a combined v13/v15/v23d/v29 NFNet bundle with weights near 0.15/0.30/0.25/0.30, likely blended conservatively into the v517 axis.
- Next step: Let queued public scores land first. If v526/v528/v529 do not improve but tie safely, consider packaging the four-member NFNet blend as the next real-Sed sidecar; if any queued SED sidecar drops, hold and use the OOF grid only for offline calibration.


## 2026-05-10 14:55 UTC - queue monitor recovered; fine NFNet grid aborted as too slow

- Track: Queue monitoring plus Spec F OOF retuning. No new public Kaggle kernel was added because v519-v522/v526-v528/v529 remain complete and queued behind the daily cap.
- Status: Latest submissions remain unchanged: v518=0.927, v525=0.929, v524=0.929, v523=0.928, v517=0.930. Current public best remains 0.930 from v517. Required v510 and v529 both remain COMPLETE/no failure.
- Queue/monitor: The previous focus monitor log logs/submit_pending_birdclef_queue_20260510T114618Z_focus_v529.log showed sleeping after the v519 cap response, but no matching process was alive. Restarted the focus-only monitor with v529 included: pid 71760, log logs/submit_pending_birdclef_queue_20260510T144421Z_focus_v529_restart.log. It skipped already-submitted v516/v517/v523/v524/v525/v518, attempted v519, hit the expected daily cap with 9.3 hours remaining, and sleeps 33600 seconds.
- OOF grid follow-up: Tried refining the remote four-member NFNet grid from step 0.05 to step 0.025 for v13/v15/v23d/v29. The run was still active after an extended wait and was killed to avoid tying up the cron/trainer. No result file was produced. The current actionable NFNet weights remain the prior completed step-0.05 result: v13=0.15, v15=0.30, v23d=0.25, v29=0.30 with macro AUC 0.710048 on 497 overlap rows / 100 valid classes.
- Packaging caveat: v13/v15 use 10s/160-mel audio config, while v23d/v29 use 20s/128-mel. A single-manifest combined bundle would be unsafe with the current one-audio-config kernel loader; a true four-member NFNet package needs either grouped multi-config inference support or separate bundle passes before blending.
- Next step: Let the queue submit v519 after reset. If queued SED sidecars score safely, implement grouped multi-config SED inference before packaging the four-member NFNet blend; otherwise keep the OOF result as offline calibration only.


## 2026-05-10 15:55 UTC - mixed-audio-config NFNet bundle support and smoke

- Track: Spec A+G packaging/inference support plus queue monitoring. No new public Kaggle kernel was pushed because v519-v522/v526-v528/v529 are already complete and queued behind the daily cap.
- Status: Latest submissions remain unchanged: v518=0.927, v525=0.929, v524=0.929, v523=0.928, v517=0.930. Current public best remains 0.930 from v517. Required v510 and v529 remain COMPLETE/no failure.
- Queue/monitor: Focus monitor pid 71760, log logs/submit_pending_birdclef_queue_20260510T144421Z_focus_v529_restart.log, is alive and sleeping after the expected v519 daily-cap response with about 9.3h remaining at launch.
- Implementation: Extended scripts/birdclef_sed_combine_bundles.py with --allow-mixed-audio-config. The combiner now records source_audio_config per source bundle, audio_config per model entry, audio_configs, and mixed_audio_config in the combined manifest. Extended scripts/birdclef_sed_soundscape_infer.py so inference groups models by per-entry audio_config and reuses decoded audio/windows per config. This resolves the packaging caveat where v13/v15 use 10s/160-mel and v23d/v29 use 20s/128-mel.
- Artifact smoke: Built local mixed-config bundle artifacts/sed_bundles/sed-nfnet-v13v15-v23d-v29-mixedcfg-v1.zip from existing zips with source bundle weights v13v15=0.45, v23d=0.25, v29=0.30. Because the v13/v15 source zip internally uses v13=0.40 and v15=0.60, this approximates the OOF grid as effective weights v13=0.18, v15=0.27, v23d=0.25, v29=0.30. Bundle has 12 TorchScript models, two audio configs, total model size 1078.446 MB, zip size 998.822 MB, sha256 4b9bd06e1f06380573efe58bff3206e2e42c1ed50915ac599dae87dd7414b0fb.
- Validation: python3 -m py_compile passed for scripts/birdclef_sed_combine_bundles.py and scripts/birdclef_sed_soundscape_infer.py. Ran one real train soundscape through the mixed bundle using CPU, batch size 4, 2 torch threads: loaded 12 models, output 12 x 235, no NaNs, probability range 0.003634 to 0.342064, mean 0.097957, runtime 9.047s/file.
- Decision: Mixed-config inference is now operational locally, but do not upload/push the four-member NFNet package until queued public SED sidecar scores land. The bundle is large (~1GB zip), and the v13/v15 effective weights are approximate unless we split/reweight their source models directly.
- Next step: Let v519 submit after reset. If queued SED sidecars tie/improve, either upload this mixed-config bundle or build a precise v13/v15 split-weight bundle before a public kernel; otherwise keep the mixed-config work as packaging infrastructure.


## 2026-05-10 16:58 UTC - precise mixed-config NFNet OOF-grid bundle prepared

- Track: Spec A+G packaging/inference support while daily submissions remain capped. Public best remains 0.930 from v517; latest visible submissions are v518=0.927, v525=0.929, v524=0.929, v523=0.928, v517=0.930. v510/v519/v520/v521/v522/v526/v527/v528/v529 kernel statuses are COMPLETE/no failure. Focus queue monitor pid 71760 is alive and sleeping after expected v519 daily-cap response.
- Hypothesis: If queued SED sidecar kernels tie/improve, the next higher-upside candidate should use the four-member NFNet OOF-grid blend rather than the approximate v13/v15 source-bundle ratio. Prior OOF grid best was 0.710048 over 497 overlap rows / 234 classes with weights v13=0.15, v15=0.30, v23d=0.25, v29=0.30.
- Implementation: Extended scripts/birdclef_sed_combine_bundles.py with repeated --member-weight bundle:member:weight overrides. When a source bundle contains multiple members, overrides now redistribute that source bundle's normalized weight across source members while preserving each member's internal fold weights. This allows the packaged v13/v15 bundle to reproduce the OOF-grid v13/v15 split exactly instead of being constrained to its original 0.40/0.60 internal ratio.
- Artifact prepared locally: artifacts/sed_bundles/sed-nfnet-v13v15-v23d-v29-oofgrid071005-v1.zip. Inputs: v13v15 source bundle weight 0.45 with member overrides v13_ep8_100=0.15 and v15_ep8_200=0.30, v23d=0.25, v29=0.30. Effective member weights verified exactly: v13_ep8_100=0.15, v15_ep8_200=0.30, v23d_20s128=0.25, v29=0.30. Bundle has 12 TorchScript models, two audio configs (10s/160-mel and 20s/128-mel), model_weight_sum=1.0, zip size 998.822 MB, sha256 7f50069bb7d79ae3b44b461c3af5e533e8a77245e00c0837bf520bac1a9b8534.
- Validation: python3 -m py_compile passed for scripts/birdclef_sed_combine_bundles.py and scripts/birdclef_sed_soundscape_infer.py. CPU smoke on one real train soundscape with batch size 4 and 2 torch threads loaded all 12 models and produced 12 x 235 submission output with no NaNs; probability range 0.003606 to 0.344208, mean 0.098336, runtime 8.979s/file.
- Decision: Do not upload/push this ~1GB mixed NFNet package yet. It is now ready as the precise next candidate, but daily submission slots are capped and v519-v522/v526-v529 public scores are still pending. Upload only if queued SED/NFNet sidecars justify spending a dataset/kernel/submission slot.
- Branch/PR: feature/focus-only-queue-guard / PR #216.


## 2026-05-10 18:03 UTC - B0 v26 + NFNet OOF blend diagnostics while capped

- Track: Spec F retune after new SED prediction artifacts, gated by Spec A/G queue monitoring. No new public Kaggle kernel was pushed because the focus queue is capped and v519-v522/v526-v529 public scores are still pending.
- Status: Latest visible submissions remain v518=0.927, v525=0.929, v524=0.929, v523=0.928, v517=0.930. Current public best remains v517=0.930. v510/v519/v520/v521/v522/v526/v527/v528/v529 kernel statuses are COMPLETE/no failure. Focus monitor pid 71760 is alive and sleeping after the expected daily-cap response on v519.
- Diagnostic inputs: copied missing local OOF npz files for v23d and v29 from trainer so local blend grids can include the current NFNet sidecars. Used scripts/birdclef_oof_blend_grid.py with current local artifacts.
- Grid 1: b0v26 + v13 + v15 + v23d + v29, step 0.10, artifact artifacts/blend_grids/b0v26_v13_v15_v23d_v29_step01.json. Overlap is only 76 rows / 42 valid classes because all five artifacts have different balanced selections, so treat this as directional only. Best coarse weights: b0v26=0.50, v13=0.20, v15=0.10, v23d=0.10, v29=0.10, macro AUC 0.932648 vs b0v26 single 0.922453 on that tiny overlap.
- Grid 2: b0v26 + v13 + v15 + v23d, step 0.05, artifact artifacts/blend_grids/b0v26_v13_v15_v23d_step005.json. Overlap 119 rows / 61 valid classes. b0v26 single AUC 0.938555; best blend b0v26=0.75, v13=0.15, v15=0.00, v23d=0.10 reaches 0.943334. Because overlap is small, this supports low-weight NFNet sidecars but is not strong enough alone to spend a public slot before queued scores.
- Grid 3: b0v26 + v13 + v15, step 0.05, artifact artifacts/blend_grids/b0v26_v13_v15_step005.json. Same 119-row overlap; best b0v26=0.80, v13=0.15, v15=0.05 reaches 0.941903, so v23d adds a small incremental lift over v13/v15 on the same overlap.
- Grid 4: b0v26 + v29, step 0.05, artifact artifacts/blend_grids/b0v26_v29_step005.json. This is the most stable check: overlap 1279 rows / 170 valid classes. b0v26 single AUC 0.910015; best b0v26=0.90, v29=0.10 reaches 0.911282. v29 is low-correlation to b0v26 (flat Pearson 0.3108) and a 5-10% weight is locally positive.
- Decision: Keep the precise mixed-config NFNet bundle and these B0+NFNet OOF grids as next-candidate preparation only. Do not upload/push another ~1GB package until v519-v522/v526-v529 score. If v522 B0 all-files SED and/or v528 v29 sidecar tie/improve, the most justified follow-up is a conservative B0-v26 + v29 or B0-v26 + v13/v23d low-weight blend, not more taxon/gamma micro-sweeps.
- Branch/PR: feature/focus-only-queue-guard / PR #216. Artifacts are under ignored artifacts/blend_grids and are summarized here for durability.


## 2026-05-10 18:58 UTC - B0 v26 + NFNet v29 mixed bundle prepared

- Track: Spec A/G packaging plus Spec F retune after new SED artifacts. No public Kaggle dataset/kernel was uploaded because daily code-submission slots are still capped and v519-v522/v526-v529 public scores are pending.
- Status: Latest visible submissions remain v518=0.927, v525=0.929, v524=0.929, v523=0.928, v517=0.930. Current public best remains v517=0.930. v510/v519/v520/v521/v522/v526/v527/v528/v529 are COMPLETE/no failure. Focus monitor pid 71760 is alive and sleeping after expected daily-cap response on v519.
- Hypothesis: The stable OOF blend check from the previous run showed b0v26+v29 has useful low-correlation signal on a much larger overlap than the all-NFNet grids (1279 rows / 170 classes): b0v26 alone 0.910015, best b0v26=0.90 + v29=0.10 reaches 0.911282, v29 flat correlation to b0v26 0.3108. Prepare the exact mixed-config artifact so it is ready if v522 and/or v528 public scores justify a follow-up.
- Artifact prepared locally: artifacts/sed_bundles/sed-b0v26-plus-nfnet-v29-oofblend090010-v1.zip. Inputs: b0v26 bundle weight 0.90 from sed-b0-q3cap80-ep12init-v26-allfiles-bundle-v1.zip; NFNet v29 bundle weight 0.10 from sed-nfnet-v29-20s128-broad181-v1.zip. Bundle has 6 TorchScript models, two audio configs (10s/160-mel B0 and 20s/128-mel NFNet), model_weight_sum=1.0, total model size 315.775 MB, zip size 291.764 MB, sha256 21771572e0d54f388ccd0cd0bde9480d9ca0b5aeddbad7eb12958e918bed3fc2.
- Validation: python3 -m py_compile passed for scripts/birdclef_sed_combine_bundles.py and scripts/birdclef_sed_soundscape_infer.py. CPU smoke on one real train soundscape with batch size 4 and two torch threads loaded all 6 models and produced 12 x 235 output with no NaNs; probability range 0.019460 to 0.496435, mean 0.124421, runtime 5.020s/file.
- Decision: This is now the smallest precise mixed B0+NFNet package candidate (~292 MB instead of the ~999 MB four-member NFNet package). Hold upload/push until queued LB scores land. If v522 (B0 v26) and/or v528 (v29 sidecar) tie/improve, this is the preferred next package candidate at conservative blend weight; otherwise keep as infrastructure and do not spend a slot.
- Branch/PR: feature/focus-only-queue-guard / PR #216. Artifact is ignored under artifacts/; log entry preserves enough metadata to reproduce it.


## 2026-05-10 19:58 UTC - held v530 mixed B0v26+v29 kernel scaffold

- Track: Spec A/G packaging plus Spec F retune after new SED artifacts. No public Kaggle push or dataset upload this run because daily submission slots are still capped and v519-v522/v526-v529 public scores are pending.
- Status: Latest visible submissions remain v518=0.927, v525=0.929, v524=0.929, v523=0.928, v517=0.930. Current public best remains v517=0.930. v510/v519/v520/v521/v522/v526/v527/v528/v529 are COMPLETE/no failure. Focus monitor pid 71760 is alive and sleeping after expected daily-cap response on v519.
- Implementation: Added held local Kaggle scaffold kaggle-kernels/v530-b0v26-v29-mixed-hold/. It copies the v522 B0-v26 all-files path but points metadata at future private dataset slug yourslewis/bc26-sed-b0v26-nfnet-v29-oofblend090010-v1 and kernel id yourslewis/bc26-v530-b0v26-v29-mixed-blend-005. This is intentionally not pushed to Kaggle yet.
- Mixed-config fix: The held v530 script now supports per-model audio_config in the in-kernel real SED loader. It caches decoded audio by sample rate and 12 row windows by full audio_config, so one bundle can safely run B0 10s/160-mel folds and NFNet v29 20s/128-mel folds in the same code competition kernel. Constants: REAL_SED_BLEND_WEIGHT=0.05, REAL_SED_MAX_MODELS=6, REAL_SED_MIN_MODELS=3, REAL_SED_EST_SEC_PER_FILE_PER_MODEL=0.95, REAL_SED_ZIP_NAME=sed-b0v26-plus-nfnet-v29-oofblend090010-v1.zip.
- Validation: python3 -m py_compile passed for kaggle-kernels/v530-b0v26-v29-mixed-hold/script.py. The underlying bundle smoke from previous run loaded all 6 models on one real train soundscape and produced 12 x 235 output with no NaNs in 5.020s/file.
- Decision: Keep v530 as a held scaffold only. If v522 and/or v528 public scores tie/improve after the monitor submits them, upload the prepared 291.764 MB B0v26+v29 dataset and push this v530 kernel. If queued SED sidecars fall, do not spend a public slot.
- Branch/PR: feature/focus-only-queue-guard / PR #216.


## 2026-05-10 20:58 UTC - v530 scaffold corrected to v517 taxon-gated base

- Track: Spec A/G packaging plus Spec F retune after new SED artifacts. No public Kaggle push or dataset upload this run; daily slots remain capped and queued v519-v522/v526-v529 scores are still pending.
- Status: Latest visible submissions remain v518=0.927, v525=0.929, v524=0.929, v523=0.928, v517=0.930. Current public best remains v517=0.930. Focus monitor pid 71760 is alive and sleeping after expected daily-cap response on v519.
- Fix: Rebased held kaggle-kernels/v530-b0v26-v29-mixed-hold/script.py from v528 (the v517 softer taxon-gated axis + v29 sidecar) instead of v522 (older v508+B0v26 axis). This matters because v517 is current public best and v530 should test new mixed SED signal on top of the strongest base, not regress to the older non-taxoned base.
- Preserved mixed-config support: v530 still points at future dataset slug bc26-sed-b0v26-nfnet-v29-oofblend090010-v1, keeps REAL_SED_BLEND_WEIGHT=0.05, REAL_SED_MAX_MODELS=6, REAL_SED_MIN_MODELS=3, and supports per-model audio_config for B0 10s/160-mel + NFNet 20s/128-mel. The v517 taxon max gate remains active with floor=0.30, alpha=0.50.
- Added kaggle-kernels/v530-b0v26-v29-mixed-hold/HOLD_NOTES.md with the exact future dataset upload command and reminder not to push until queued v522/v528 evidence justifies a slot.
- Validation: python3 -m py_compile passed for the corrected v530 script. No Kaggle push performed.
- Branch/PR: feature/focus-only-queue-guard / PR #216.


## 2026-05-10 21:47 UTC - pre-reset monitor refresh, no new variant

- Track: A/G queue monitoring and guarded packaging hold. No public Kaggle dataset/kernel upload this run because slots remain capped and queued v519-v522/v526-v529 LB scores are still pending.
- Status: Latest visible submissions remain v518=0.927, v525=0.929, v524=0.929, v523=0.928, v517=0.930. Current public best remains v517=0.930. Required status check: v510/v519/v520/v521/v522/v526/v527/v528/v529 are all COMPLETE/no failure.
- Queue action: Old focus monitor pid 71760 was still sleeping from the earlier 9.3h cap response and had been started before the latest monitor-safety script state. Killed it and restarted current scripts/submit_pending_birdclef_queue.py so the active process has STOP_AFTER_FOCUS semantics and the latest focus priority through v529. New monitor pid 31151, log logs/submit_pending_birdclef_queue_20260510T214426Z_focus_v529_current.log.
- Monitor sanity: New monitor skipped already-submitted v516/v517/v523/v524/v525/v518, verified v519 COMPLETE/no failure, attempted v519 submission, hit the expected daily cap with 2.3h remaining, and is sleeping 8400s. This should wake just after UTC reset and submit v519 first, then continue focus order v520/v521/v522/v526/v527/v528/v529 under the 5/day cap.
- Decision: Continue holding v530 and the prepared B0v26+v29 dataset until v522 and/or v528 public scores arrive. No extra variants or micro-sweeps added.
- Branch/PR: feature/focus-only-queue-guard / PR #216.


## 2026-05-10 22:58 UTC - capped queue hold + v530 static preflight

- Track: A/G monitoring plus held-package preflight. No public Kaggle dataset/kernel upload this run because daily slots remain capped and v519-v522/v526-v529 scores are still pending.
- Status: Latest visible submissions remain v518=0.927, v525=0.929, v524=0.929, v523=0.928, v517=0.930. Current public best remains v517=0.930. Required status check: v510/v519/v520/v521/v522/v526/v527/v528/v529 are all COMPLETE/no failure.
- Queue: Current monitor pid 31151, log logs/submit_pending_birdclef_queue_20260510T214426Z_focus_v529_current.log, remains alive. It already skipped submitted v516/v517/v523/v524/v525/v518, verified v519 complete, hit expected daily cap with 2.3h remaining, and should wake shortly after UTC reset to submit v519 first.
- Static preflight: Re-ran py_compile for kaggle-kernels/v530-b0v26-v29-mixed-hold/script.py and scripts/submit_pending_birdclef_queue.py. Added a local static assertion pass for v530: metadata id is yourslewis/bc26-v530-b0v26-v29-mixed-blend-005; metadata includes future dataset yourslewis/bc26-sed-b0v26-nfnet-v29-oofblend090010-v1; script has v517 taxon constants floor=0.30/alpha=0.50; mixed-audio helper def _sed_audio_config_key exists; per-entry audio_config is used; REAL_SED_MAX_MODELS=6 and REAL_SED_MIN_MODELS=3. All assertions passed.
- Decision: Keep holding v530 until v522 and/or v528 public scores justify upload. No extra variants or micro-sweeps added.
- Branch/PR: feature/focus-only-queue-guard / PR #216.


## 2026-05-11 00:47 UTC - UTC reset focus submissions v519-v522/v526 landed

- Track: A/G queue monitoring and guarded packaging hold. No new Kaggle dataset/kernel was uploaded; this run waited through UTC reset to verify the focus monitor actually submitted the next completed kernels.
- Status before new scores: latest scored best remains v517=0.930. Latest visible scored rows before reset were v518=0.927, v525=0.929, v524=0.929, v523=0.928, v517=0.930. New reset submissions are PENDING score, not scored yet.
- Queue result: Active monitor pid 31151/log logs/submit_pending_birdclef_queue_20260510T214426Z_focus_v529_current.log woke after cap reset and submitted five focus kernels in order: v519 ref 52528303 at 00:04:31Z, v520 ref 52528318 at 00:05:03Z, v521 ref 52528328 at 00:05:36Z, v522 ref 52528352 at 00:06:07Z, and v526 ref 52528369 at 00:06:39Z. It then attempted v527, hit daily cap with 23h remaining, and is sleeping 82920s.
- Pending-score set: v519 (B0 ep12-init pseudo-label student blend 0.15 + v508), v520/v521 (B0 v23 SED blend 0.05/0.10 + v508), v522 (B0 v26 all-files SED blend 0.05 + v508), v526 (v516 taxon gate + v23d NFNet blend 0.05). Remaining focus queue for next reset: v527, v528, v529.
- Decision: Keep holding v530 and the prepared B0v26+v29 dataset until scores for v522 and/or v528 arrive. v522 is now submitted/pending; v528 is still queued behind v527 for the next reset. Do not add variants or upload v530 yet.
- Branch/PR: feature/focus-only-queue-guard / PR #216.


## 2026-05-11 01:50 UTC - reset scores arrived; B0 SED sidecars below v517

- Track: A/G monitoring and Spec F decision gate after new SED artifacts scored. No new Kaggle dataset/kernel was uploaded.
- Score update: Current public best remains v517=0.930. Newly scored reset submissions: v519=0.929 (B0 ep12-init pseudo-label student blend 0.15 + v508), v520=0.928 (B0 v23 SED blend 0.05 + v508), v521=0.928 (B0 v23 SED blend 0.10 + v508), v522=0.927 (B0 v26 all-files SED blend 0.05 + v508). v526 remains pending at this check. Previous scored rows: v518=0.927, v525=0.929, v524=0.929, v523=0.928, v517=0.930.
- Interpretation: v519 confirms the ep12-init pseudo-label student is safer than v518 but still below current best; do not add more B0 student slots. B0 SED public transfer is weaker than local OOF suggested: v23/v26 all failed to beat v517 and v22 dropped to 0.927. This weakens the held B0v26+v29 v530 case unless v528 independently shows v29 helps on top of v517.
- Queue: Monitor pid 31151 remains alive and sleeping after attempting v527 and hitting cap. Remaining focus queue: v527, v528, v529. v528 is now the key evidence for the held B0v26+v29 dataset because v522 did not justify upload by itself.
- Hold update: Tightened kaggle-kernels/v530-b0v26-v29-mixed-hold/HOLD_NOTES.md: because v522 scored only 0.927, require v528 to tie/improve the 0.930 best before reconsidering upload/push of v530. If v528 scores below 0.930, keep v530 as infrastructure only.
- Branch/PR: feature/focus-only-queue-guard / PR #216.


## 2026-05-11 02:50 UTC - v526 hidden timeout; add timeout-safe v531 v29 probe

- Track: Spec A/G inference packaging failure diagnosis and runtime-safe follow-up. Current public best remains v517=0.930. Latest scored rows: v519=0.929, v520=0.928, v521=0.928, v522=0.927; v526 code submission completed without score because Kaggle reported hidden notebook runtime exceeded.
- Failure diagnosis: v526 kernel itself was COMPLETE/no failure, but the competition code submission record for ref 52528369 has `errorDescription`: hidden submission notebook exceeded allowed runtime. This means the real NFNet SED path is operational but too slow for hidden code rerun at the existing 3-model / 0.05 blend settings. v527/v528/v529 kernels remain COMPLETE/no failure, but full SED variants are now timeout-risky.
- Implemented v531 as a minimal A/G runtime probe based on v528/v517 axis: `kaggle-kernels/v531-v517-v29-fast1-blend002`. It keeps the v517 taxon gate (`floor=0.30`, `alpha=0.50`) and v29 20s/128 NFNet dataset, but changes `REAL_SED_BLEND_WEIGHT=0.02`, `REAL_SED_MAX_MODELS=1`, `REAL_SED_TIME_BUFFER_SEC=55*60`, and `REAL_SED_EST_SEC_PER_FILE_PER_MODEL=6.0`. If hidden file count/time budget is too large, it skips real SED and falls back to the safe v517 axis instead of timing out.
- Pushed real Kaggle kernel `yourslewis/bc26-v531-v517-plus-v29-fast1-sed-blend-002`, version 1. Initial status after push: RUNNING/no failure.
- Queue safety: Updated `scripts/submit_pending_birdclef_queue.py` so focus queue is now already-submitted v516/v517/v523/v524/v525/v518/v519/v520/v521/v522/v526, then v527, then v531. Removed timeout-prone v528/v529 from focus while `STOP_AFTER_FOCUS=1`, so the monitor will not burn future slots on full 3-model SED submissions unless explicitly re-enabled.
- Restarted monitor after replacing old pid 31151: new pid 71865, log `logs/submit_pending_birdclef_queue_20260511T024936Z_focus_v531_fast.log`. It hit daily cap at v527 and is sleeping until the next UTC reset.
- Branch/PR: `feature/focus-only-queue-guard`, PR #217 (`https://github.com/yourslewis/birdclef-2026/pull/217`). Do not merge without Wenhao approval.
- Next step: wait for v531 kernel completion; after reset submit v527 first, then v531 if complete. If v531 scores/ties and does not timeout, consider an ONNX/OpenVINO or one-model weight sweep; if it skips SED or scores below v517, stop public SED packaging and pivot to cleaner OOF-teacher pseudo-label/model-zoo training.


## 2026-05-11 03:48 UTC - v531 complete; launch ConvNeXt model-zoo OOF while capped

- Status check: Current public best remains v517=0.930. Latest submissions: v526 completed with hidden runtime timeout/no score; v522=0.927; v521=0.928; v520=0.928; v519=0.929. No duplicate submissions added.
- Kernel status: v510 COMPLETE/no failure; v527 COMPLETE/no failure; v531 COMPLETE/no failure.
- v531 output verification: Kaggle session output lists `submission.csv`. Log confirms v531 found the v29 manifest, loaded 1/3 TorchScript models, applied real SED blend weight 0.02, applied taxon max gate floor0.30 alpha0.50, wrote `submission.csv` shape `(240,235)`, and finished in 289.9s / 4.8 min on the public dry-run. This validates the runtime-safe path and real SED application on public sample.
- Queue: monitor pid 71865 remains alive, log `logs/submit_pending_birdclef_queue_20260511T024936Z_focus_v531_fast.log`, sleeping on daily cap before v527. Current focus queue remains v527 then v531; timeout-prone v528/v529 stay held.
- Track pivot while capped: Spec D model-zoo diversity baseline. Because submissions are capped and SED full bundles timeout, launched a durable ConvNeXt-Tiny 3-fold OOF run on trainer GPU0 to collect a low-correlation prediction artifact rather than waiting.
- Launch: synced runner/training script/config to `trainer` and started pid `4538`, log `~/birdclef-2026/logs/sed_oof_v21_convnext_tiny_20260511T034518Z.log`, command `CUDA_VISIBLE_DEVICES=0 nohup ~/kaggle_envs/s6e3/bin/python scripts/birdclef_sed_oof_runner.py --base-config configs/birdclef/zoo_convnext_tiny_balanced_oof_v21_10s_160_100cls_lr1e4_ep5.json --output-root artifacts/sed_oof/zoo-convnext-tiny-balanced-oof-v21-10s-160-100cls-lr1e4-ep5 --n-folds 3 --experiment-id zoo-convnext-tiny-balanced-oof-v21-10s-160-100cls-lr1e4-ep5`. Initial monitor: process alive, fold 0 started, GPUs otherwise idle.
- Branch/PR: `feature/focus-only-queue-guard`, PR #217 (`https://github.com/yourslewis/birdclef-2026/pull/217`). Do not merge without Wenhao approval.
- Next step: monitor ConvNeXt OOF summary and v527/v531 submissions after UTC reset. If ConvNeXt produces competitive/low-correlation OOF, run a blend grid with v517/v508 teacher-cache predictions before any public packaging.


## 2026-05-11 03:50 UTC - ConvNeXt v21 data-root fix and fold0 result

- Follow-up on Spec D ConvNeXt OOF launch: initial trainer job pid 4538 failed quickly because the config still pointed at stale SMB path `/mnt/mac_data/workspace_don/kaggle_birdclef2026/data/taxonomy.csv`; on trainer the valid data root is `/home/yourslewis/birdclef-2026/data`. This was a config/environment issue, not a model/training failure.
- Fixed `configs/birdclef/zoo_convnext_tiny_balanced_oof_v21_10s_160_100cls_lr1e4_ep5.json` to use `/home/yourslewis/birdclef-2026/data`, synced it to trainer, removed the failed fold0 output, and relaunched durable retry pid `5006`, log `~/birdclef-2026/logs/sed_oof_v21_convnext_tiny_retry_20260511T034634Z.log`.
- Early validation: fold0 completed successfully with macro AUC `0.605954` over 99 valid classes after 5 epochs (`train_loss=0.06477`, `val_loss=0.08419`); fold1 started and GPU0 was active (~2.3GB, high utilization). This is a real model-zoo OOF artifact in progress, not a public micro-sweep.
- Next step: let pid 5006 finish folds 1-2 and aggregate `artifacts/sed_oof/zoo-convnext-tiny-balanced-oof-v21-10s-160-100cls-lr1e4-ep5/oof_summary.json`; then compare with prior RegNet/NFNet/B0 OOF and only consider packaging/blending if correlation/diversity justifies it.


## 2026-05-11 04:48 UTC - ConvNeXt OOF weak; scale B3 pseudo-label student

- Status check: public best remains v517=0.930. Latest submissions unchanged: v526 completed with hidden runtime timeout/no score; v522=0.927, v521=0.928, v520=0.928, v519=0.929. v510/v527/v531 kernels are all COMPLETE/no failure. Queue monitor pid 71865 remains alive and sleeping on daily cap before v527, then v531.
- ConvNeXt Spec D result: trainer retry pid 5006 finished all 3 OOF folds for `zoo-convnext-tiny-balanced-oof-v21-10s-160-100cls-lr1e4-ep5`. Fold AUCs were 0.605954 / 0.609649 / 0.575837, but aggregated OOF macro AUC was only `0.537610` over 100 valid classes. Artifact: `artifacts/sed_oof/zoo-convnext-tiny-balanced-oof-v21-10s-160-100cls-lr1e4-ep5/oof_predictions.npz`. Interpretation: no-pseudo ConvNeXt is too weak; do not package or submit this lane.
- Follow-up track: Spec B/D pseudo-label model-zoo scale from existing smoke. Created `configs/birdclef/pl_r2_b3_v508_soft_p100_5s_pretrained_lr1e4_ep20_bestval.json` by scaling the B3 smoke from 256 rows / 3 epochs to all 792 rows / 20 epochs while keeping 5s/160mel, ImageNet-pretrained EfficientNet-B3, LR 1e-4, soft v508 teacher targets, and best-val-AUC restore.
- Launched on trainer GPU1, pid 4079, log `~/birdclef-2026/logs/pl_r2_b3_v508_soft_p100_5s_pretrained_lr1e4_ep20_bestval_20260511T044449Z.log`; completed in 81.491s. Result: best val AUC `0.983658`, final all-row macro AUC `0.981388` over 75 valid classes, teacher AUC `0.991149`, final teacher corr `0.972191`, TorchScript `41.995 MB`.
- Blend recheck against teacher/student artifacts: `artifacts/blend_grids/student_teacher_blend_recheck_20260511.json`. B3 did not improve the teacher even at tiny weights; best B3 blend was weight 0.01 with delta `-0.00000167`. The only local positive remains B0 ep12 at weight 0.15 (`+0.000224`), already represented publicly by v519 which scored only 0.929. Interpretation: do not create a B3 public sidecar; current pseudo-student family is too teacher-correlated/weak for LB gain.
- Next step: keep queue focused on v527 then v531. For training, pivot away from teacher-distilled students unless using a genuinely different teacher/source; candidate next work is OOF-teacher pseudo-label generation or an export/runtime optimization path, not another B0/B3/ConvNeXt sidecar.


## 2026-05-11 05:55 UTC - ONNX export path for v29 SED and v532 kernel

- Status check: public best remains v517=0.930. Latest submissions unchanged: v526 completed with hidden runtime timeout/no score; v522=0.927, v521=0.928, v520=0.928, v519=0.929. v510/v527/v531 are COMPLETE/no failure. Queue was still capped before v527.
- Track: Spec A/G inference export/runtime optimization after v526 hidden timeout. Hypothesis: TorchScript NFNet v29 is too slow for hidden code rerun, but ONNX Runtime can make the full 3-fold v29 sidecar feasible.
- Implemented reusable export smoke `scripts/birdclef_torchscript_onnx_smoke.py`. It loads a bundle manifest, selects one TorchScript model, traces a clip-logits-only wrapper for legacy ONNX export, checks the ONNX graph, and benchmarks PyTorch vs ONNX Runtime when available. Py_compile passed locally and on trainer.
- Smoke result on trainer for v29 fold0: direct new dynamo exporter failed on TorchScript/NFNet (`Cannot call numel() on tensor with symbolic sizes/strides` / nested ScriptModule trace issue), but legacy export after tracing the clip-only wrapper succeeded. ONNX graph checked cleanly; max abs diff ~`1e-5`, cosine ~`1.0`. ONNX Runtime was ~0.14s per 4-window batch versus TorchScript ~0.14s per clip, about 4x faster for clip logits.
- Exported all three v29 folds to ONNX clip-logit models and packaged `artifacts/sed_bundles/sed-nfnet-v29-20s128-broad181-onnx-v1.zip` (468.4 MB). Each ONNX is ~168.753 MB with ORT cosine ~1.0. Uploaded private Kaggle dataset `yourslewis/bc26-sed-nfnet-v29-20s128-onnx-v1` (`https://www.kaggle.com/datasets/yourslewis/bc26-sed-nfnet-v29-20s128-onnx-v1`).
- Created/pushed real Kaggle kernel `yourslewis/bc26-v532-v517-plus-v29-onnx3-blend-005`, version 1. It copies the v517/v528 axis but uses the ONNX v29 dataset, `REAL_SED_BLEND_WEIGHT=0.05`, `REAL_SED_MAX_MODELS=3`, `REAL_SED_TIME_BUFFER_SEC=30*60`, `REAL_SED_EST_SEC_PER_FILE_PER_MODEL=1.00`, and SED ONNX inference via the existing Kaggle onnxruntime wheel. Push returned no invalid sources. Status after push/poll: RUNNING/no failure.
- Queue update: added v532 after v531 in focus priority and restarted monitor old pid 71865 -> new pid 3175, log `logs/submit_pending_birdclef_queue_20260511T055130Z_focus_v532_onnx.log`. It hit daily cap at v527 and sleeps until next UTC reset. Queue order: v527, v531, v532. Timeout-prone TorchScript v528/v529 remain held.
- Branch/PR: `feature/focus-only-queue-guard`, PR #217. Do not merge without Wenhao approval.
- Next step: recheck v532 completion/output log. At reset, submit v527, v531, then v532 if complete. If v532 completes and scores/ties without hidden timeout, use ONNX for any future real SED sidecar; if it times out or scores below v517, stop public SED packaging until a new low-correlation model signal exists.


## 2026-05-11 06:45 UTC - v532 ONNX3 kernel verified complete

- Status check: public best remains v517=0.930. Latest submissions unchanged: v526 completed with hidden runtime timeout/no score; v522=0.927, v521=0.928, v520=0.928, v519=0.929. No duplicate submissions added.
- Kernel status: v510/v527/v531/v532 are all COMPLETE/no failure. v532 output contains `submission.csv`.
- v532 output verification: Kaggle log found the ONNX v29 manifest under `yourslewis/bc26-sed-nfnet-v29-20s128-onnx-v1`, loaded 3/3 real SED ONNX models, inferred all 20 public dry-run files, applied real SED blend weight 0.05, applied taxon max gate floor0.30 alpha0.50, wrote submission shape `(240,235)`, and finished in 453.6s / 7.6 min. SED runtime was 276.1s for 20 files; this is much slower than the isolated ORT synthetic benchmark but still far faster/safer than the TorchScript hidden-timeout path and leaves ~82 min remaining on public run.
- Queue: monitor pid 3175 is alive, log `logs/submit_pending_birdclef_queue_20260511T055130Z_focus_v532_onnx.log`, sleeping after daily-cap response before v527. Current reset order remains v527 -> v531 -> v532. Full TorchScript v528/v529 stay held.
- Decision: no more public SED variants before v531/v532 scores. If v532 scores/ties and avoids hidden timeout, ONNX becomes the only allowed packaging path for future real SED sidecars; if v532 times out or scores below v517, stop public SED packaging until a new low-correlation model signal exists.
- Branch/PR: `feature/focus-only-queue-guard`, PR #217. Do not merge without Wenhao approval.

## 2026-05-11 07:58 UTC - V2S pseudo-label student scaled; no public sidecar

- Status check: Current public best remains v517=0.930. Latest submissions: v526 completed with hidden runtime timeout/no score; v522=0.927, v521=0.928, v520=0.928, v519=0.929. Required kernel checks: v510/v527/v531/v532 are all COMPLETE/no failure. Active monitor pid 3175, log `logs/submit_pending_birdclef_queue_20260511T055130Z_focus_v532_onnx.log`, is sleeping on daily cap before v527; reset order remains v527 -> v531 -> v532. No duplicate submissions added.
- Track: Spec B/D pseudo-label model-zoo diversity while public slots are capped. Hypothesis: the V2S smoke had a healthier learning curve than no-pseudo ConvNeXt and might add a different low-correlation student signal to the v508/v517 teacher axis.
- Implementation: Added `configs/birdclef/pl_r2_v2s_v508_soft_p100_5s_pretrained_lr1e4_ep20_bestval.json`, scaling the successful V2S 5s pretrained smoke from 256 rows / 3 epochs to all 792 pseudo-labeled rows / 20 epochs. Config: EfficientNetV2-S (`efficientnetv2_rw_s`), pretrained=true, 5s / 160 mels, LR 1e-4, BCE soft v508 teacher targets, no mixup, best-val-AUC restore.
- Launch/validation: Synced the config to trainer and ran `CUDA_VISIBLE_DEVICES=1 nohup ~/kaggle_envs/s6e3/bin/python scripts/birdclef_pseudolabel_student_train.py --config configs/birdclef/pl_r2_v2s_v508_soft_p100_5s_pretrained_lr1e4_ep20_bestval.json`. Log: `~/birdclef-2026/logs/pl_r2_v2s_v508_soft_p100_5s_pretrained_lr1e4_ep20_bestval_20260511T075053Z.log`. Completed in 56.346s.
- Result: best epoch 20, best val AUC `0.982115` over 61 valid classes. Final all-row student AUC `0.983987` over 75 valid classes versus teacher AUC `0.991149`; student/teacher corr `0.967937`, MAE `0.037628`. Artifacts: `artifacts/pseudolabels/students/pl-r2-v2s-v508-soft-p100-5s-pretrained-lr1e4-ep20-bestval/metrics.json`, `student_predictions.npz`, TorchScript `model_torchscript.pt` (88.739 MB).
- Blend check: Wrote `artifacts/blend_grids/v2s_student_teacher_blend_20260511.json`. Tiny teacher blend weights barely improve local AUC: best checked weight 0.005 gives `+0.00000885`; 0.01 gives `+0.00000699`; 0.02 and above are negative. This is far weaker than the B0 ep12 local gain that still scored only 0.929 publicly.
- Decision: Do not package or submit a V2S sidecar. It is a useful artifact but not enough public-LB evidence to spend a code slot while v531/v532 are queued. Next step remains queue monitoring for v527/v531/v532; if v532 scores/ties without timeout, ONNX remains the only SED packaging path. For training, pivot to a genuinely different teacher/source or OOF-teacher cache rather than more v508-distilled students.
- Branch/PR: `feature/v533-v2s-pseudolabel-student` for this config/log update. No merge without Wenhao approval.

## 2026-05-11 08:51 UTC - V2S external pretrain + pseudo-label init trial

- Status check: Current public best remains v517=0.930. Latest submissions unchanged: v526 hidden runtime timeout/no score; v522=0.927, v521=0.928, v520=0.928, v519=0.929. Required kernel checks: v510/v527/v531/v532 all COMPLETE/no failure. Active monitor pid 3175/log `logs/submit_pending_birdclef_queue_20260511T055130Z_focus_v532_onnx.log` is alive and sleeping on daily cap before v527; reset order remains v527 -> v531 -> v532. No duplicate submissions added.
- Track: Spec C + B, external-data pretraining followed by pseudo-label/noisy-student fine-tuning. Hypothesis: a V2S encoder pretrained on cleaned target-species Xeno-Canto audio could add less teacher-cloned structure than pure ImageNet-pretrained V2S.
- Implemented configs: `configs/birdclef/xc_v2s_q3_cap80_external_pretrain_balanced_ep12_bestloss.json` and `configs/birdclef/pl_r2_v2s_v508_xc_extinit_5s_lr1e4_ep20_bestval.json`.
- External pretrain run: synced config to trainer and ran `CUDA_VISIBLE_DEVICES=1 nohup ~/kaggle_envs/s6e3/bin/python scripts/birdclef_sed_pilot_train.py --config configs/birdclef/xc_v2s_q3_cap80_external_pretrain_balanced_ep12_bestloss.json`. Log: `~/birdclef-2026/logs/xc_v2s_q3_cap80_external_pretrain_balanced_ep12_bestloss_20260511T084532Z.log`. Config: EfficientNetV2-S, pretrained=true, XC q>=3 manifest, 976 examples, 5s/128-mel, focal BCE gamma1.5, sqrt pos weights, LR 1e-4, 12 epochs, restore best by val loss.
- External pretrain result: completed in 64.534s; best epoch 2 by val loss 0.160222; holdout macro AUC `0.587925` over 122 valid classes; TorchScript size 88.732 MB. Artifact: `artifacts/external_pretrain/xc-v2s-q3-cap80-external-pretrain-balanced-ep12-bestloss/model_torchscript.pt`.
- Pseudo-label fine-tune run: ran `CUDA_VISIBLE_DEVICES=1 nohup ~/kaggle_envs/s6e3/bin/python scripts/birdclef_pseudolabel_student_train.py --config configs/birdclef/pl_r2_v2s_v508_xc_extinit_5s_lr1e4_ep20_bestval.json`. Log: `~/birdclef-2026/logs/pl_r2_v2s_v508_xc_extinit_5s_lr1e4_ep20_bestval_20260511T084658Z.log`. It loaded 786 encoder keys from the external TorchScript checkpoint, skipped the frame head, trained all 792 v508 teacher-cache rows for 20 epochs.
- Pseudo-label result: completed in 52.397s; best val AUC `0.982867` at epoch 19 over 61 valid classes. Final all-row student AUC `0.982950` over 75 valid classes vs teacher `0.991149`; student/teacher corr `0.969834`, MAE `0.036394`; TorchScript size 88.739 MB.
- Blend check: Wrote `artifacts/blend_grids/v2s_xc_extinit_student_teacher_blend_20260511.json`. External-init V2S best tiny blend was weight 0.01 with delta `+0.00000544`, weaker than the pure pretrained V2S tiny delta `+0.00000885`, and far weaker than the already-submitted B0 ep12 student that only scored 0.929 publicly.
- Decision: Do not package or submit V2S external-init. External V2S pretraining is usable infrastructure, but this exact 5s/128-mel pseudo-label fine-tune is not a public-slot candidate. Next training pivot should be an OOF/ensemble teacher cache or a different data source/objective, not more v508-distilled V2S/B0 clones.
- Branch/PR: `feature/v533-v2s-pseudolabel-student`, PR #218. No merge without Wenhao approval.

## 2026-05-11 10:01 UTC - OOF model-zoo blend audit after V2S kill

- Status check: Current public best remains v517=0.930. Latest submissions unchanged: v526 hidden runtime timeout/no score; v522=0.927, v521=0.928, v520=0.928, v519=0.929. Required kernel checks: v510/v527/v531/v532 all COMPLETE/no failure. Active monitor pid 3175/log `logs/submit_pending_birdclef_queue_20260511T055130Z_focus_v532_onnx.log` is alive and sleeping on daily cap before v527; reset order remains v527 -> v531 -> v532. No duplicate submissions added.
- Track: Spec D + F model-zoo diversity / post-new-artifact OOF blend tuning. Hypothesis: weak low-correlation zoo models such as ConvNeXt-Tiny or NFNet v29 might still improve a B0 SED OOF blend enough to justify a future packaged sidecar or new training target.
- Full audit command: `python3 scripts/birdclef_oof_blend_grid.py --member b0v26:artifacts/sed_oof/sed-b0-q3cap80-ep12init-oof-v26-10s-160-allfiles-ep5/oof_predictions.npz --member nfnet_v29:artifacts/sed_oof/sed-nfnet-allobserved-v29-20s-128-181cls-10per-ep5/oof_predictions.npz --member nfnet_v23d:artifacts/sed_oof/sed-nfnet-balanced-oof-v23d-20s-128-100cls-lr1e4-ep5-v22bmanifest-fullcopy/oof_predictions.npz --member b3v16:artifacts/sed_oof/sed-b3-balanced-oof-v16-10s-160-100cls-lr1e4-ep8/oof_predictions.npz --member regv20:artifacts/sed_oof/zoo-regnety008-balanced-oof-v20-10s-160-100cls-lr1e4-ep5/oof_predictions.npz --member convnext_v21:artifacts/sed_oof/zoo-convnext-tiny-balanced-oof-v21-10s-160-100cls-lr1e4-ep5/oof_predictions.npz --step 0.05 --top-k 20 --output artifacts/blend_grids/model_zoo_b0v26_v29_v23d_b3_reg_convnext_step005_20260511.json`.
- Full audit result: six-way common intersection was only 35 rows / 24 valid classes, so treat as directional only. Single AUCs on that tiny intersection: b0v26 `0.931298`, nfnet_v29 `0.762126`, nfnet_v23d `0.594858`, b3v16 `0.514877`, regv20 `0.501297`, convnext_v21 `0.518562`. Best grid blend was b0v26 0.75 + nfnet_v29 0.10 + convnext_v21 0.15 = `0.935012`.
- Reliability check command: reran a broader three-way grid on b0v26/nfnet_v29/convnext_v21 with `--step 0.025`, output `artifacts/blend_grids/b0v26_v29_convnext_step0025_20260511.json`. This had 734 overlapping rows / 100 valid classes. Single AUCs: b0v26 `0.914969`, nfnet_v29 `0.641420`, convnext_v21 `0.542206`. Pairwise corr: b0v26-v29 `0.310343`, b0v26-convnext `0.183061`, v29-convnext `0.377142`.
- Reliable-grid result: best was b0v26 0.925 + nfnet_v29 0.075 + convnext 0.0 = `0.915309`, only `+0.000340` over b0v26 alone. The best blends with ConvNeXt included tiny 0.025-0.05 ConvNeXt weights and were weaker than the v29-only blend.
- Decision: ConvNeXt has low correlation but too little AUC to package or train forward from the current recipe. NFNet v29 remains the only model-zoo sidecar with consistent small OOF lift, but public submission should still wait for queued v531/v532 because v526 proved hidden-timeout risk and B0/NFNet SED sidecars have underperformed local OOF. No new Kaggle kernel pushed.
- Branch/PR: `feature/v533-v2s-pseudolabel-student`, PR #218. No merge without Wenhao approval.

## 2026-05-11 10:51 UTC - v517 taxon-gated teacher cache + hard-confidence student

- Status check: Current public best remains v517=0.930. Latest submissions unchanged: v526 hidden runtime timeout/no score; v522=0.927, v521=0.928, v520=0.928, v519=0.929. Required kernel checks: v510/v527/v531/v532 all COMPLETE/no failure. Active monitor pid 3175/log `logs/submit_pending_birdclef_queue_20260511T055130Z_focus_v532_onnx.log` is alive and sleeping on daily cap before v527; reset order remains v527 -> v531 -> v532. No duplicate submissions added.
- Track: Spec B + E pseudo-label/noisy-student cache, pivoting away from v508-clone soft-label students toward current-best v517 taxon-gated labels and hard-confidence supervision.
- Implemented helper: `scripts/birdclef_apply_taxon_gate_teacher_cache.py`. It applies the exact Kaggle taxon-max gate (`group_evidence=max`, multiplier `max(floor,evidence)^alpha`) to a row-level teacher-cache NPZ while preserving `row_ids` and labels, then writes summary diagnostics.
- Generated cache: `artifacts/pseudolabels/v517-taxon-gated-teacher-cache66/predictions.npz` from `artifacts/pseudolabels/v508-teacher-cache66/predictions.npz` with `floor=0.30`, `alpha=0.50`. Summary artifact: `artifacts/pseudolabels/v517-taxon-gated-teacher-cache66/summary.json`. Gated cache AUC on 792 labeled soundscape rows is `0.992008` vs v508 baseline `0.991149` (`+0.000859`), matching the v517 local diagnostic. Gated teacher/old teacher corr `0.991849`, MAE `0.033818`. Hard-label counts after gate: p95 positives 681 cells / 431 rows, negatives <=0.05 4736 cells.
- Added config: `configs/birdclef/pl_r2_b0_v517_hard_p95n05_lr1e3_ep20_bestval.json`. It trains EfficientNet-B0 10s/160-mel with hard-conf targets from the v517 cache (`positive_threshold=0.95`, `negative_threshold=0.05`, lr=1e-3, 20 epochs, best-val-AUC restore).
- Trainer run: `CUDA_VISIBLE_DEVICES=1 nohup ~/kaggle_envs/s6e3/bin/python scripts/birdclef_pseudolabel_student_train.py --config configs/birdclef/pl_r2_b0_v517_hard_p95n05_lr1e3_ep20_bestval.json`; log `~/birdclef-2026/logs/pl_r2_b0_v517_hard_p95n05_lr1e3_ep20_bestval_20260511T104656Z.log`. Completed in 36.205s.
- Student result: target mask fraction `0.029229`, positives 681, negatives 4736. Best val AUC `0.707491` at epoch 19; final all-row student AUC `0.663124` vs v517 teacher `0.992008`. Student/teacher corr only `0.335521`, MAE `0.195208`, TorchScript size `15.391 MB`. Artifact: `artifacts/pseudolabels/students/pl-r2-b0-v517-hard-p95n05-lr1e3-ep20-bestval/metrics.json`.
- Blend check: `artifacts/blend_grids/v517_hard_student_blend_20260511.json`. Best linear blend is tiny: 5% hard student gives `0.992099`, `+0.000091` over v517 teacher. Larger weights rapidly degrade. This is not enough to justify a public kernel, but confirms hard-conf labels create genuinely low-correlation signal.
- Decision: Do not package/submit this hard-conf student. Keep the v517-gated cache helper as reusable infrastructure; next hard-label attempt, if any, should add real supervised examples or class-balanced negative mining rather than pure sparse hard-conf distillation.
- Branch/PR: `feature/v533-v2s-pseudolabel-student`, PR #218. No merge without Wenhao approval.

## 2026-05-11 11:50 UTC - v517 hard-conf real-clip mix smoke killed

- Status check: Current public best remains v517=0.930; cron prompt's 0.927 plateau is stale. Latest submissions remain v526 hidden runtime timeout/no score, v522=0.927, v521=0.928, v520=0.928, v519=0.929. Required kernel checks via Bearer API: v510/v527/v531/v532 are all COMPLETE/no failure. Active monitor pid 3175/log `logs/submit_pending_birdclef_queue_20260511T055130Z_focus_v532_onnx.log` is alive and sleeping on daily cap before v527; reset order remains v527 -> v531 -> v532. No duplicate submissions added.
- Track: Spec B pseudo-label/noisy-student follow-up after the v517 taxon-gated hard-confidence cache. Hypothesis: the pure sparse hard-conf B0 student was low-correlation but too weak; adding a small amount of real supervised `train.csv` clip examples might anchor the student without collapsing it into the teacher.
- Implemented smoke configs:
  - `configs/birdclef/pl_r2_b0_v517_hard_p95n05_realclip_w01_smoke.json`: v517 gated cache, hard_conf p95/n05, EfficientNet-B0 10s/160-mel, lr=1e-3, 256 pseudo rows, 128 real clips capped at 4/class, real clip weight 0.10, 3 epochs.
  - `configs/birdclef/pl_r2_b0_v517_hard_p95n05_realclip_w002_smoke.json`: same but real clip weight 0.02.
  - `configs/birdclef/pl_r2_b0_v517_hard_p95n05_realclip_w002_lr3e4_smoke.json`: same as w0.02 but lr=3e-4.
- Commands launched on trainer (`192.168.0.10`, GPU1) with `scripts/birdclef_pseudolabel_student_train.py`; logs under `~/birdclef-2026/logs/pl_r2_b0_v517_hard_p95n05_realclip_*_smoke_20260511T114*.log`.
- Smoke results:
  - w0.10/lr1e-3: used 121/128 real clips; best val AUC `0.527373`; final all-row student AUC `0.529729`; final corr to teacher `0.007961`; MAE `0.489442`.
  - w0.02/lr1e-3: used 121/128 real clips; best val AUC `0.604746`; final all-row student AUC `0.585193`; final corr to teacher about `0.279`; MAE `0.520911`.
  - w0.02/lr3e-4: used 121/128 real clips; best val AUC `0.521979`; final all-row student AUC `0.511207`; final corr to teacher `0.342117`; MAE `0.188411`.
- Interpretation/kill decision: Real-clip anchoring in this naive form makes the already-weak hard-conf student worse than the previous full pure hard-conf run (`0.663124` final all-row AUC and tiny +0.000091 teacher blend). Do not scale to all rows or package a public sidecar. The issue is likely objective mismatch/class imbalance: random real clips with one-hot labels do not align with the row-level teacher-cache soundscape task.
- Validation: `python3 -m json.tool` passed for all three configs; `python3 -m py_compile` passed for `scripts/birdclef_pseudolabel_student_train.py` and `scripts/birdclef_apply_taxon_gate_teacher_cache.py`.
- Branch/PR: `feature/v534-v517-hard-supervised-mix`. Next step is to push this as a negative-result PR/log update, then pivot away from naive real-clip mixing. Better next candidates are OOF-teacher cache generation, class-balanced negative mining, or SED ONNX queue scoring for v531/v532 rather than more v508/v517-distilled sidecars.

## 2026-05-11 12:58 UTC - v517 hard-conf class-balanced negative mining

- Status check: Current public best remains v517=0.930; cron prompt's 0.927 plateau is stale. Latest submissions remain v526 hidden runtime timeout/no score, v522=0.927, v521=0.928, v520=0.928, v519=0.929. Active queue monitor pid 3175/log `logs/submit_pending_birdclef_queue_20260511T055130Z_focus_v532_onnx.log` is alive and sleeping on daily cap before v527; reset order remains v527 -> v531 -> v532. No duplicate submissions added.
- Track: Spec B pseudo-label/noisy-student follow-up after naive real-clip mixing failed. Hypothesis: the low-correlation v517 hard-confidence student may benefit from more mined negatives (`p<=0.10`) if they are capped by row/class to avoid common-class/background domination, instead of mixing mismatched one-hot real clips.
- Implementation: Extended `scripts/birdclef_pseudolabel_student_train.py` with deterministic hard-mask caps for `hard_conf` and `soft_anchor`: `max_positive_per_row`, `max_negative_per_row`, `max_positive_per_class`, and `max_negative_per_class`. Caps keep highest-probability positives and lowest-probability negatives, row caps first then class caps.
- Smoke configs added and run on trainer GPU1, 256 v517-gated teacher-cache rows, 3 epochs:
  - `configs/birdclef/pl_r2_b0_v517_hard_p95n10_capneg16_smoke.json`: p95/n10, max 16 negatives/row, max 64 negatives/class. Log `~/birdclef-2026/logs/pl_r2_b0_v517_hard_p95n10_capneg16_smoke_20260511T124958Z.log`. Mask: 82 positive cells, 2406 negative cells, fraction 0.04153. Result: best val AUC 0.601025; final all-row student AUC 0.626134 over 42 classes; corr to teacher 0.172945; MAE 0.178503.
  - `configs/birdclef/pl_r2_b0_v517_hard_p95n10_capneg32_smoke.json`: max 32 negatives/row, max 64/class. Log `~/birdclef-2026/logs/pl_r2_b0_v517_hard_p95n10_capneg32_smoke_20260511T125004Z.log`. Mask: 82 positive cells, 2957 negative cells, fraction 0.05073. Result: best val AUC 0.577879; final all-row student AUC 0.620543; corr 0.199793; MAE 0.183150.
  - `configs/birdclef/pl_r2_b0_v517_hard_p95n10_capneg16_lr3e4_smoke.json`: same cap16 but lr 3e-4. Log `~/birdclef-2026/logs/pl_r2_b0_v517_hard_p95n10_capneg16_lr3e4_smoke_20260511T125010Z.log`. Result: best val AUC 0.483650; final all-row AUC 0.463872; corr -0.059831; MAE 0.171172.
- Scale-up: Because cap16/lr1e-3 was the only smoke above the killed real-clip variants and had lower MAE than cap32, added `configs/birdclef/pl_r2_b0_v517_hard_p95n10_capneg16_ep20_bestval.json` and ran all 792 rows / 20 epochs. Log `~/birdclef-2026/logs/pl_r2_b0_v517_hard_p95n10_capneg16_ep20_bestval_20260511T125102Z.log`. Result: best epoch 10; best val AUC 0.708542; final all-row student AUC 0.689983 over 75 classes vs teacher 0.992008; corr 0.326682; MAE 0.196088. Mask after full caps: fraction 0.018945, 681 positive cells, 2830 negative cells.
- Blend check: `artifacts/blend_grids/v517_hard_capneg16_student_blend_20260511.json` on trainer. Best checked linear blend is 2% student + 98% v517 teacher: macro AUC 0.992136, delta +0.000127 over the v517 teacher cache. This is a small improvement over the previous pure p95/n05 hard-conf student's +0.000091 local blend, but still too tiny to spend a public Kaggle slot.
- Decision: Keep the class-balanced hard-mask caps as useful training infrastructure, but do not package/submit a public sidecar from this student. The lane improved local hard-conf AUC versus pure p95/n05 but remains far below teacher quality and only supports tiny blend weights. Next good training pivot is OOF/ensemble-teacher cache generation or a genuinely different source/objective; public queue remains focused on v527/v531/v532 scoring.

## 2026-05-11 13:51 UTC - clean OOF ensemble teacher cache export

- Status check: Current public best remains v517=0.930; cron prompt's 0.927 plateau is stale. Latest submissions remain v526 hidden runtime timeout/no score, v522=0.927, v521=0.928, v520=0.928, v519=0.929. Active queue monitor pid 3175/log `logs/submit_pending_birdclef_queue_20260511T055130Z_focus_v532_onnx.log` is alive and sleeping on the daily cap before v527; order remains v527 -> v531 -> v532. No duplicate submissions added.
- Track: Spec B pseudo-label/noisy-student cache generation, pivoting away from v517-cloned students. Hypothesis: a clean OOF-only teacher cache is safer than in-sample teacher dry-runs and can support future noisy-student/negative-mining experiments with availability-aware filtering.
- Implementation: Added `scripts/birdclef_oof_teacher_cache.py`. It builds an OOF-only ensemble teacher cache from one or more `oof_predictions.npz` artifacts, validates matching labels/truth, supports union or intersection alignment, renormalizes configured weights per row according to available members, and writes NPZ fields `files`, `labels`, `y_true`, `teacher_pred`, `available_mask`, `available_count`, `available_weight_sum`, `member_names`, and `member_weights`. Summary JSON includes single-member AUCs, pairwise overlap/correlation, availability histogram, macro AUC, top-k recall, thresholds, probability stats, and per-class AUC.
- Validation/cache runs:
  - Stable B0v26+NFNet v29 union cache: `artifacts/pseudolabels/oof-teacher-cache/b0v26_nfnetv29_w090010_union_cache.npz` with weights b0v26=0.90, nfnet_v29=0.10. Coverage 2928 files; availability histogram `{1:1649, 2:1279}`; truth conflicts 0; teacher macro AUC 0.871018 over 206 classes; top-k recall top1 0.257514 / top3 0.422131 / top5 0.501025 / top10 0.595628.
  - Same B0v26+NFNet v29 intersection cache: `artifacts/pseudolabels/oof-teacher-cache/b0v26_nfnetv29_w090010_intersection_cache.npz`. Coverage 1279 files; all rows have both members; truth conflicts 0; teacher macro AUC 0.911282 over 170 classes; top-k recall top1 0.279124 / top3 0.479281 / top5 0.570758 / top10 0.666145. This matches the prior b0v26+v29 OOF blend optimum and is the safest cache candidate for future student training.
  - Conservative model-zoo union cache: `artifacts/pseudolabels/oof-teacher-cache/modelzoo_conservative_union_cache.npz` with b0v26=0.75, nfnet_v29=0.10, nfnet_v23d=0.10, b3v16=0.025, convnext_v21=0.025. Coverage 3388 files; availability histogram `{1:1185, 2:964, 3:897, 4:307, 5:35}`; truth conflicts 0; macro AUC 0.870158 over 206 classes; top-k recall top1 0.236718 / top3 0.389906 / top5 0.469599 / top10 0.560213.
- Interpretation: The stable b0v26+v29 intersection cache is the best clean OOF teacher target now: it has stronger AUC/top-k than the broader union caches, and v29 remains the clearest low-correlation additive signal. The full model-zoo union cache adds coverage but dilutes teacher quality with weak zoo members, so it should not be the first training target.
- Decision: No public Kaggle kernel/submission from this infrastructure run. Next step is to train a student or hard/negative cache against the `b0v26_nfnetv29_w090010_intersection_cache` artifact, or wait for v531/v532 LB scores to decide whether v29 deserves more packaging slots.

## 2026-05-11 15:00 UTC - OOF teacher-cache soft-label student smoke killed

- Status check: Current public best remains v517=0.930; cron prompt's 0.927 plateau is stale. Latest submissions remain v526 hidden runtime timeout/no score, v522=0.927, v521=0.928, v520=0.928, v519=0.929. Active queue monitor pid 3175/log `logs/submit_pending_birdclef_queue_20260511T055130Z_focus_v532_onnx.log` is alive and sleeping on the daily cap before v527; order remains v527 -> v531 -> v532. No duplicate submissions added.
- Track: Spec B pseudo-label/noisy-student training from the clean OOF teacher cache created in the previous run. Hypothesis: a B0 SED student trained from the non-leaky b0v26+NFNet-v29 intersection teacher cache might distill v29's low-correlation signal into a compact sidecar.
- Implementation: Extended `scripts/birdclef_sed_pilot_train.py` with `selection_strategy=oof_teacher_cache` and `oof_teacher_cache` soft-label targets. It aligns selected train-audio files to OOF cache rows, supports `oof_teacher_power`, `oof_teacher_min_available`, uses teacher probabilities as training targets, keeps OOF truth for AUC evaluation, and records `oof_teacher_summary` in metrics.
- Smoke configs/run on trainer GPU1:
  - `configs/birdclef/sed_b0_oofteacher_b0v26_v29_intersection_power100_smoke.json`: 128 rows, 2 epochs, power 1.0. AUC 0.491701 over 24 valid classes, teacher target coverage 128/128, runtime 7.047s.
  - `configs/birdclef/sed_b0_oofteacher_b0v26_v29_intersection_power085_smoke.json`: 128 rows, 2 epochs, power 0.85. AUC 0.504271 over 24 valid classes, runtime 5.423s. This only barely passed operationally.
- Scale/pilot runs:
  - `configs/birdclef/sed_b0_oofteacher_b0v26_v29_intersection_power085_ep4.json`: 1279 intersection-cache rows, 4 epochs, random holdout AUC 0.519602 over 128 classes vs teacher-target AUC 0.907241 on the same holdout; student-teacher corr 0.341924, MAE 0.085552. OOF runner artifact `artifacts/sed_oof_teacher_students/sed-b0-oofteacher-b0v26-v29-intersection-power085-oof3/oof_predictions.npz` reached only macro AUC 0.518421 over 170 classes (fold AUCs 0.529310, 0.529344, 0.554285).
  - B0v26-initialized checks (`power100_ep3_b0v26init`, `power100_ep4_b0v26encinit`) produced very high random-holdout AUCs 0.990720/0.987829, but these are not trustworthy for this purpose because the initializer is a B0v26 OOF fold model and the random holdout does not preserve fold isolation. Treat them as checkpoint-loading smoke only, not valid evidence for packaging.
- Decision/kill: From-scratch OOF teacher-cache distillation is too weak after smoke/full OOF (0.518 OOF AUC) and should not be packaged or submitted. The high B0v26-init random-holdout results are leakage-prone and should not be used for public decisions unless a fold-aware initializer is implemented. Keep the teacher-cache training plumbing, but pivot away from compact B0 distillation; wait for v531/v532 public results or implement a fold-aware OOF student if continuing this lane.

## 2026-05-11 15:58 UTC - fold-aware OOF-teacher initialized student becomes held bundle candidate

- Status check: Current public best remains v517=0.930; cron prompt's 0.927 plateau is stale. Latest submissions remain v526 hidden runtime timeout/no score, v522=0.927, v521=0.928, v520=0.928, v519=0.929. Active queue monitor pid 3175/log `logs/submit_pending_birdclef_queue_20260511T055130Z_focus_v532_onnx.log` is alive and sleeping on the daily cap before v527; order remains v527 -> v531 -> v532. No duplicate submissions added.
- Track: Spec B OOF teacher-cache student training, fixing the leakage concern from the previous random-holdout initialized checks. Hypothesis: if each validation fold uses the matching B0v26 source-fold initializer, the B0v26 checkpoint has not seen that fold's validation rows and we can trust the OOF signal more than random holdout.
- Implementation: Extended `scripts/birdclef_sed_pilot_train.py` with `split_strategy="source_oof_fold"`, `split_source_oof`, `split_source_n_folds`, and `split_source_seed`. It reconstructs original source-OOF fold membership from `selected_indices` and uses the matching source fold as validation. Extended `scripts/birdclef_sed_oof_runner.py` with `initial_checkpoint_template`, formatting `{fold}` into each fold config. Added config `configs/birdclef/sed_b0_oofteacher_b0v26_v29_sourcefold_power100_ep2_b0v26init.json`.
- Validation/preflight: 64-row source-fold preflight passed on trainer GPU1 and verified the split/source-cache plumbing. Full source-fold 3-fold OOF run: `artifacts/sed_oof_teacher_students/sed-b0-oofteacher-b0v26-v29-sourcefold-power100-ep2-b0v26init-oof3/oof_predictions.npz`. It initializes fold k from `artifacts/sed_oof/sed-b0-q3cap80-ep12init-oof-v26-10s-160-allfiles-ep5/fold{k}/model_torchscript.pt`, trains 2 epochs on the non-source-fold cache rows, and validates on source fold k.
- OOF result: macro AUC 0.943801 over 1279 rows / 170 valid classes. Fold AUCs: fold0 0.954803, fold1 0.948695, fold2 0.944378. Same-row comparison artifact `artifacts/blend_grids/oofteacher_sourcefold_student_vs_b0v26_v29_20260511.json`: student 0.943801 vs B0v26 0.910015 vs NFNet v29 0.653025 vs teacher blend b0v26/v29 0.911282. Pair/blend check shows the student is far stronger on this OOF overlap, though still likely correlated with B0v26 because it is B0v26-initialized.
- Held bundle: built but did not upload/push public Kaggle candidate `artifacts/sed_bundles/sed-b0-oofteacher-sourcefold-b0v26init-v1.zip` on trainer. Bundle has 3 TorchScript B0 models, total model size 46.167 MB, zip size 41 MB, sha256 `e88120e57594fda464764b3114855c49bb5a546912399b5df6b4ab10358ec4b1`. CPU soundscape smoke with `scripts/birdclef_sed_soundscape_infer.py` loaded all 3 models and produced 12x234 rows for one train soundscape in 1.488s/file, no NaNs/failures.
- Decision: This is the first OOF-teacher student lane with a genuinely positive clean OOF signal. Do not insert it ahead of the active capped queue yet; v527/v531/v532 should score first. If v531/v532 confirm that v29/B0-derived SED signal is useful or if a slot opens for a compact sidecar, the next action is to create a v537 held/public kernel blending this 41 MB bundle conservatively into the v517 taxon-gated axis (start weight 0.02 or 0.05) and submit under cap.

## 2026-05-11 16:55 UTC - v537 source-fold OOF-teacher sidecar Kaggle candidate pushed

- Status check: Current public best remains v517=0.930; cron prompt's 0.927 plateau is stale. Latest submissions remain v526 hidden runtime timeout/no score, v522=0.927, v521=0.928, v520=0.928, v519=0.929. Active queue monitor pid 3175/log `logs/submit_pending_birdclef_queue_20260511T055130Z_focus_v532_onnx.log` is alive and sleeping on the daily cap before v527; order remains v527 -> v531 -> v532. No duplicate submissions added.
- Track: Spec A+G/B Kaggle packaging for the new fold-aware OOF-teacher B0 sidecar. Hypothesis: the compact source-fold student (OOF 0.943801 on the B0v26+v29 intersection) may add safe signal to current public-best v517 if blended conservatively.
- Dataset: Uploaded private Kaggle dataset `yourslewis/bc26-sed-b0-oofteacher-sourcefold-v1` containing `sed-b0-oofteacher-sourcefold-b0v26init-v1.zip`. Zip size 40.1 MiB local / 41 MB trainer, sha256 `e88120e57594fda464764b3114855c49bb5a546912399b5df6b4ab10358ec4b1`. URL: https://www.kaggle.com/datasets/yourslewis/bc26-sed-b0-oofteacher-sourcefold-v1
- Kernel candidate: Added/pushed `kaggle-kernels/v537-v517-oofteacher-sourcefold-blend002/`, copied from the v532 v517+SED bridge but pointed to the new TorchScript bundle. Constants: `REAL_SED_BLEND_WEIGHT=0.02`, `REAL_SED_MAX_MODELS=3`, `REAL_SED_MIN_MODELS=1`, `REAL_SED_TIME_BUFFER_SEC=10*60`, `REAL_SED_EST_SEC_PER_FILE_PER_MODEL=0.25`, dataset slug `bc26-sed-b0-oofteacher-sourcefold-v1`, zip `sed-b0-oofteacher-sourcefold-b0v26init-v1.zip`. It blends before the v517 taxon max gate (floor 0.30, alpha 0.50).
- Kaggle push: real kernel `yourslewis/bc26-v537-v517-plus-oofteacher-b0-blend-002`, version 1, pushed successfully with no invalid data/kernel/model sources. Push URL: https://www.kaggle.com/code/yourslewis/bc26-v537-v517-plus-oofteacher-b0-blend-002. Immediate status polling still showed `RUNNING` after several minutes, no failure message available yet.
- Queue decision: Do not add v537 to the active submission monitor yet; current focus queue remains v527 -> v531 -> v532. Once v537 completes and v527/v531/v532 score, add v537 to the queue only if slots/evidence justify spending a submission.

## 2026-05-11 18:00 UTC - v537 verified complete, v538 moderate OOF-teacher blend pushed

- Status check: Current public best remains v517=0.930; cron prompt's 0.927 plateau is stale. Latest submissions remain v526 hidden runtime timeout/no score, v522=0.927, v521=0.928, v520=0.928, v519=0.929. v510/v527/v531/v532/v537 are COMPLETE. Active queue monitor was refreshed to pid 8011, log `logs/submit_pending_birdclef_queue_20260511T174619Z_focus_v537.log`, and is sleeping on daily cap before v527 with reset order v527 -> v531 -> v532 -> v537. No duplicate submissions added.
- Track: Spec A+G/B real Kaggle packaging for the new fold-aware OOF-teacher B0 sidecar, now using the compact source-fold bundle as a new prediction artifact rather than another postprocess-only tweak.
- v537 verification: Kaggle output contains `submission.csv`; log confirms the OOF-teacher bundle path loaded and ran, real SED prob range 0.014843-0.682891 mean 0.1209, SED runtime 54.6s, applied OOF-teacher B0 blend weight 0.02, applied v517 taxon max gate, wrote `(240,235)`, wall time 255.3s / 4.3 min. This clears the public runtime/mount smoke gate and should be hidden-runtime safer than the earlier large TorchScript SED candidates.
- Queue update: Added v537 to `scripts/submit_pending_birdclef_queue.py` focus priority after v532 and restarted the monitor old pid 3175 -> pid 8011. The monitor hit daily cap at v527 and will retry after UTC reset. v538 is not yet in the active submission queue because its Kaggle run is still RUNNING at this log write.
- Follow-up candidate: Added/pushed `kaggle-kernels/v538-v517-oofteacher-sourcefold-blend005/`, identical to v537 but with `REAL_SED_BLEND_WEIGHT=0.05` as the moderate companion to v537's conservative 0.02. Real Kaggle kernel `yourslewis/bc26-v538-v517-plus-oofteacher-b0-blend-005`, version 1, pushed successfully with no invalid data/kernel/model sources. Initial status checks show RUNNING/no output yet.
- Decision: Spend the next reset first on v527/v531/v532/v537. Add v538 to the queue only after it reaches COMPLETE and only if there is still an unused daily slot or v537/SED evidence justifies the moderate-weight companion. If v538 remains stuck or errors, inspect logs before any retry.

## 2026-05-11 18:47 UTC - v538 verified complete and queued for remaining reset slot

- Status check: Current public best remains v517=0.930. Latest submissions remain v526 hidden runtime timeout/no score, v522=0.927, v521=0.928, v520=0.928, v519=0.929. v510/v527/v531/v532/v537/v538 are all COMPLETE. No duplicate submissions added.
- Track: Spec A+G/B OOF-teacher B0 sidecar packaging. Since v538 completed cleanly and the reset queue has exactly five focus candidates, it is now safe to use the remaining daily slot after v527/v531/v532/v537.
- v538 verification: Kaggle output contains `submission.csv`; log confirms the OOF-teacher bundle path loaded and ran, real SED prob range 0.014843-0.682891 mean 0.1209, SED runtime 50.2s, applied OOF-teacher B0 blend weight 0.05, applied v517 taxon max gate, wrote `(240,235)`, wall time 328.7s / 5.5 min. This passes the mount/runtime/output smoke gate.
- Queue update: Added v538 to `scripts/submit_pending_birdclef_queue.py` after v537 and restarted the monitor old pid 8011 -> pid 15457, log `logs/submit_pending_birdclef_queue_20260511T184542Z_focus_v538.log`. It hit daily cap before v527 with 5.2h remaining. Reset order is now v527 -> v531 -> v532 -> v537 -> v538.
- Decision: Wait for the capped reset rather than pushing more public kernels. The next actionable work is monitoring scores/hidden-runtime outcomes for the five queued candidates; only continue OOF-teacher blend weighting if v537/v538 tie or improve v517, otherwise pivot back to a genuinely different teacher/source.

## 2026-05-11 19:40 UTC - v539 public946 replay baseline pushed

- Track: Public approach reprioritization P0. Hypothesis: the public distilled SED + Perch/ProtoSSM rank-blend stack around 0.946 should replace v517=0.930 as the immediate baseline/teacher if it reproduces in our controlled workflow.
- Branch: `feature/v539-public946-replay`.
- Implementation: Created `kaggle-kernels/v539-public946-replay/` as a repo-owned controlled port of public robust source `yaroslavkholmirzayev/0-946-replay-with-robust-inputs`. The script includes a header documenting the source and objective. Metadata attaches `tuckerarrants/bc2026-distilled-sed-public`, `jaejohn/perch-meta`, `rishikeshjani/perch-onnx-for-birdclef-2026`, `ashok205/tf-wheels`, and the Perch v2 Kaggle model.
- Validation before push: `python3 -m py_compile kaggle-kernels/v539-public946-replay/script.py` passed. Public source was previously inspected and uses ONNX Perch + distilled SED rank blending with Proto rescue, temporal continuity rescue, SED local-spike rescue, sonotype mirroring, and rare taxon suppression.
- Kaggle push: real private kernel `yourslewis/bc26-v539-public946-replay-baseline`, version 1, pushed successfully via Bearer API v1 with no invalid data/kernel/model sources. URL: https://www.kaggle.com/code/yourslewis/bc26-v539-public946-replay-baseline
- Status after initial polling: RUNNING, no output files/log yet. Do not queue for submission until it reaches COMPLETE and `submission.csv` is verified. Current capped queue remains v527 -> v531 -> v532 -> v537 -> v538.
- Next step: poll v539 completion/output. If complete and runtime-safe, add v539 ahead of further experimental sidecars at the next available submission opportunity and use its output as `public946` teacher-cache seed.

## 2026-05-11 19:50 UTC - v539 public946 replay verified complete and prioritized in queue

- Status check: Current public best remains v517=0.930. Latest submissions unchanged: v526 hidden runtime timeout/no score, v522=0.927, v521=0.928, v520=0.928, v519=0.929. v510/v527/v531/v532/v537/v538/v539 are COMPLETE. No duplicate submissions added.
- Track: Public approach reprioritization P0. v539 is now the highest-priority candidate because it reproduces the public ~0.946 Perch/ProtoSSM + distilled SED rank-blend stack in our repo-owned Kaggle workflow.
- v539 verification: Kaggle output files include `submission.csv`, `submission_protossm.csv`, and `submission_sed.csv`. Log confirms SED dir `/kaggle/input/datasets/tuckerarrants/bc2026-distilled-sed-public`, loaded `sed_fold0.onnx` through `sed_fold4.onnx`, SED inference completed for 20 public files in 68.9s, saved `submission_sed.csv` `(240,235)`, executed standard 2-way rank blend (60% Proto / 40% SED), and completed successfully. Public dry-run final `submission.csv` is aligned to `sample_submission.csv` `(3,235)`, which matches the public replay behavior; hidden code submission should rerun on real test soundscapes.
- Queue update: Inserted v539 into `scripts/submit_pending_birdclef_queue.py` immediately after already-submitted v526 and before v527/v531/v532/v537/v538. Restarted monitor old pid 15457 -> pid 27404, log `logs/submit_pending_birdclef_queue_20260511T194723Z_focus_v539_public946.log`. It attempted v539, hit daily cap with 4.2h remaining, and will retry after UTC reset. Effective reset order: v539 -> v527 -> v531 -> v532 -> v537 -> v538 (v538 likely spills to next day if all earlier submissions succeed).
- Branch/PR: `feature/v539-public946-replay`, PR #223. Next step after score: if v539 >=0.940, make it the new teacher/cache anchor; if it lands far below public claim, inspect hidden/public behavior, data-source versions, and row alignment before continuing public946 derivatives.

## 2026-05-11 21:50 UTC - v540 public946 teacher-cache66 launched

- Status check: Current public best remains v517=0.930. Latest submissions unchanged; v539 is COMPLETE and active monitor pid 27404/log `logs/submit_pending_birdclef_queue_20260511T194723Z_focus_v539_public946.log` is sleeping on daily cap after attempting v539. Effective reset order remains v539 -> v527 -> v531 -> v532 -> v537 -> v538. No duplicate submissions added.
- Track: Public approach reprioritization P1 teacher-cache preparation while waiting for submission cap reset. Hypothesis: if v539 reproduces the public ~0.946 stack, a larger dry-run teacher cache should replace the v508 teacher cache as the next pseudo-label/student anchor.
- Downloaded v539 public dry-run outputs to `artifacts/public946/v539_outputs/`: `submission.csv`, `submission_protossm.csv`, `submission_sed.csv`.
- Added `scripts/birdclef_public946_cache_summary.py`, which converts public946 output CSVs into a compact NPZ and JSON diagnostics, preserving proto/SED/final streams and evaluating overlaps with labeled train soundscape rows.
- Ran the summary on v539 outputs with labels `/Volumes/ExternalSSD/data/workspace_don/kaggle_birdclef2026/data/train_soundscapes_labels.csv`; artifact: `artifacts/pseudolabels/public946-v539-dryrun-cache-v1/predictions.npz` and `summary.json`. v539 intermediate streams are 240x235; 190 rows overlap labels, 42 valid AUC classes. SED stream macro AUC is 0.995976 on this small overlap, much stronger than Proto 0.978480, confirming the public distilled SED artifact is highly informative on dry-run train rows. Final `submission.csv` is sample-aligned 3x235 by design and not useful as a teacher cache.
- Created and pushed real Kaggle kernel `yourslewis/bc26-v540-public946-teacher-cache66`, version 1. v540 is copied from v539 but forces `dryrun_n_files=66` in submit/dry-run mode so intermediate `submission_protossm.csv` and `submission_sed.csv` should cover 792 rows, matching the existing v508 teacher-cache66 scale. v540 is **not** queued for code submission; it is an artifact-generation kernel only.
- Initial v540 status after push: RUNNING/no failure. Next step: poll v540, download intermediate outputs when complete, build the full 792-row public946 cache, then train the first public946 student only after v539 LB score confirms the public stack transfers.

## 2026-05-11 22:50 UTC - v540 full public946 teacher cache built

- Status check: Current public best remains v517=0.930. Latest submissions unchanged; v539 public946 replay is COMPLETE and the active monitor pid 27404/log `logs/submit_pending_birdclef_queue_20260511T194723Z_focus_v539_public946.log` is sleeping on daily cap after attempting v539. No duplicate submissions added.
- v540 artifact kernel completed successfully. Kaggle outputs include `submission.csv`, `submission_protossm.csv`, and `submission_sed.csv`. Log confirms `submission_protossm.csv` `(792,235)`, SED dataset `tuckerarrants/bc2026-distilled-sed-public`, 5 ONNX SED folds loaded, SED inference for 66 train files in 161.2s, `submission_sed.csv` `(792,235)`, and standard 60% Proto / 40% SED rank blend executed. Final `submission.csv` remains sample-aligned `(3,235)` by dry-run design and is not the teacher artifact.
- Downloaded v540 outputs to `artifacts/public946/v540_outputs/` and built full cache `artifacts/pseudolabels/public946-v540-teacher-cache66-v1/` with `scripts/birdclef_public946_cache_summary.py`.
- Enhanced the summary script to reconstruct the pre-sample-align public946 rank blend for train rows and to emit student-compatible teacher NPZs: `teacher_proto.npz`, `teacher_sed.npz`, `teacher_rankblend.npz` with keys `row_ids`, `labels`, `probs`.
- v540 cache diagnostics versus local train soundscape labels (`739` matched positive rows, `75` valid classes): Proto stream macro AUC `0.987485`, top1/top3/top5/top10 row recall `0.754/0.858/0.897/0.939`; public SED stream macro AUC `0.997067`, top1/top3/top5/top10 `0.982/0.995/0.999/1.000`; reconstructed rankblend macro AUC `0.994834`, top1/top3/top5/top10 `0.419/0.687/0.797/0.917`. Interpretation: the public distilled SED stream is an exceptionally strong direct teacher on labeled train soundscapes; rankblend optimizes per-class ranking but is less useful as a BCE-style top-k target because it is rank-uniform around mean 0.5.
- Decision: hold student training until v539 LB score arrives. If v539 scores near public946, first student should likely use `teacher_sed.npz` as the soft/event-local teacher, with `teacher_rankblend.npz` reserved for rank/blend diagnostics rather than direct BCE training.

## 2026-05-11 23:50 UTC - public946 SED teacher B0 smoke completed

- Status check: Current public best remains v517=0.930. Latest submissions unchanged; active monitor pid 27404/log `logs/submit_pending_birdclef_queue_20260511T194723Z_focus_v539_public946.log` is still sleeping on daily cap after attempting v539. v539/v540/v527/v531/v532/v537/v538 are COMPLETE. No duplicate submissions added.
- Track: Public approach P1 smoke while waiting for v539 score. Hypothesis: the v540 public distilled SED teacher is a much stronger direct pseudo-label source than v508/v517 and should be tested with a small B0 sanity student before any full training.
- Added config `configs/birdclef/pl_public946_sed_b0_5s_lr3e4_smoke.json`. It trains EfficientNet-B0, 5s/160mel, external XC B0 encoder init, teacher `artifacts/pseudolabels/public946-v540-teacher-cache66-v1/teacher_sed.npz`, soft BCE, teacher power 1.0, no mixup, LR 3e-4, 256 rows, 3 epochs, restore best val AUC.
- Launched on trainer GPU0 with `CUDA_VISIBLE_DEVICES=0 ~/kaggle_envs/s6e3/bin/python scripts/birdclef_pseudolabel_student_train.py --config configs/birdclef/pl_public946_sed_b0_5s_lr3e4_smoke.json`; log `~/birdclef-2026/logs/pl_public946_sed_b0_5s_lr3e4_smoke_20260511T234552Z.log`. Runtime ~4.2s after feature decode; copied artifacts back locally under `artifacts/pseudolabels/students/pl-public946-sed-b0-5s-lr3e4-smoke/`.
- Result: best epoch 3, val student macro AUC `0.891356` over 35 classes vs teacher `0.994754`; final all-row student macro AUC `0.818694` over 42 classes vs teacher `0.995378`; student-teacher corr `0.4796`, MAE `0.3790`; TorchScript size 15.391 MB. The curve improved each epoch, but 3-epoch/256-row student is far below teacher and not a public packaging candidate.
- Decision: Do not submit/package this smoke. If v539 scores near public946, next step should be a full 792-row longer B0 public946-SED student or a stronger NFNet/V2S student, but only after v539 LB confirms transfer. The smoke shows the teacher is learnable but needs more capacity/epochs or a harder target formulation.

## 2026-05-12 00:05 UTC - reset submissions: v539 public946 first, v538 capped

- Queue monitor pid 27404/log `logs/submit_pending_birdclef_queue_20260511T194723Z_focus_v539_public946.log` woke after UTC reset and submitted five kernels under the daily cap: v539 public946 replay baseline (ref 52559109), v527 taxon gate alpha 0.375 (ref 52559133), v531 timeout-safe single-model v29 blend 0.02 (ref 52559147), v532 ONNX3 v29 blend 0.05 (ref 52559170), and v537 OOF-teacher B0 blend 0.02 (ref 52559190). All are PENDING score at log time.
- v538 OOF-teacher B0 blend 0.05 was next in queue but hit the daily submission cap with 23h remaining and the monitor is sleeping for the next reset. No duplicate submissions were added.
- Current best before these pending scores remains v517=0.930. The first score to watch is v539; if v539 lands >=0.940, make public946 the new teacher/cache anchor and continue with the public946 SED teacher student lane. If v539 fails/times out or scores near 0.930, inspect hidden behavior and row alignment before launching larger public946-derived students.

## 2026-05-12 00:55 UTC - full public946 SED B0 student scaled while v539 pending

- Status check: v539/v527/v531/v532/v537 remain PENDING score after reset submissions; current scored best remains v517=0.930. Monitor pid 27404 is sleeping on cap before v538 for the next UTC reset. No duplicate submissions added.
- Track: Public approach P1 student preparation. The small 256-row smoke was not packageable but improved each epoch, so scaled the same setup to the full 792-row v540 public SED teacher cache while waiting for v539 LB.
- Added `configs/birdclef/pl_public946_sed_b0_5s_lr3e4_ep20_bestval.json`: EfficientNet-B0, 5s/160mel, external XC B0 encoder init, teacher `teacher_sed.npz`, soft BCE, power 1.0, no mixup, LR 3e-4, all rows, 20 epochs, restore best val AUC.
- Launched on trainer GPU1: `CUDA_VISIBLE_DEVICES=1 ~/kaggle_envs/s6e3/bin/python scripts/birdclef_pseudolabel_student_train.py --config configs/birdclef/pl_public946_sed_b0_5s_lr3e4_ep20_bestval.json`; log `~/birdclef-2026/logs/pl_public946_sed_b0_5s_lr3e4_ep20_bestval_20260512T004530Z.log`. Copied artifacts back locally under `artifacts/pseudolabels/students/pl-public946-sed-b0-5s-lr3e4-ep20-bestval/`.
- Result: best epoch 20, best val student macro AUC `0.988504` over 61 classes vs teacher `0.996585`; final all-row student macro AUC `0.976669` over 75 classes vs teacher `0.996743`; student-teacher corr `0.977457`, MAE `0.005771`; TorchScript size 15.391 MB, runtime 29.5s.
- Interpretation: the full B0 student is technically strong and much better than the smoke, but still materially below the public SED teacher. It is not a public packaging candidate until v539 score lands and a blend check shows it adds diversity; likely next better route is stronger/capacity-diverse student (NFNet/V2S/ConvNeXt) or use SED teacher as distillation target for a compact sidecar only if it improves a public946 blend.

## 2026-05-12 01:50 UTC - v539 scores 0.943; v541 mirror/rare public946 follow-up queued

- Status: v539 public946 replay scored `0.943`, a large jump over the old v517/v527/v531/v537 `0.930` plateau and the earlier 0.927 family. v527/v531/v537 also tied `0.930`; v532 timed out. Daily cap is exhausted after v539/v527/v531/v532/v537, and v538 remains unsubmitted.
- Track: P0 public946 baseline reproduction / focused follow-up. Since v539 was below the public 0.946 target but clearly valid, inspected the stronger public source `lb-score-0-946.py`. Difference from v539 lane: it keeps post-blend sonotype mirroring and adaptive rare-taxon thresholding.
- Implemented `kaggle-kernels/v541-public946-mirror-rare/` by wrapping the public 0.946 stack in a repo-owned kernel with the same validated data sources as v539 plus the mirror/rare postprocess enabled.
- Validation before push: local `python3 -m py_compile kaggle-kernels/v541-public946-mirror-rare/script.py` passed; source contains SED/Proto rank blend plus `Sonotype mirroring` and `Adaptive thresholding` code paths.
- Pushed real Kaggle kernel `yourslewis/bc26-v541-public946-mirror-rare`, version 1, via Bearer API v1; push returned kernel id `118866025`, no invalid data/kernel/model sources.
- Queue: updated `scripts/submit_pending_birdclef_queue.py` to include v541 immediately after v539 and before old/internal sidecars. Killed stale monitor pid `27404` and started refreshed monitor pid `78746`, log `logs/submit_pending_birdclef_queue_20260512T014825Z_focus_v541_public946.log`. It skips submitted focus kernels, sees v541 RUNNING, and sleeps in 10-min checks. Once v541 completes it will submit v541 first after the daily cap resets, then v538.
- Next step: monitor v541 completion/runtime/log path; if it scores >=0.943 and approaches 0.946, make it the new anchor and run one public946+internal minority blend. If it fails/times out, inspect whether ONNX/TF path, mirroring/rare taxon metadata, or extra source/version mismatch is responsible.

## 2026-05-12 02:55 UTC - v541 completed and verified; queued on cap

- Status: current best remains `v539 = 0.943`. Latest scored submissions: v539 `0.943`; v527/v531/v537 `0.930`; v532 hidden-timeout/no score. v510 remains COMPLETE/scored `0.927` from prior runs, with no open mount/TorchScript/output issue.
- v541 kernel status: `yourslewis/bc26-v541-public946-mirror-rare` version 1 is COMPLETE with no failure message. Monitor pid `78746` attempted to submit it but hit the daily 5-submission cap and is sleeping ~22h; v541 remains first pending submission, before v538.
- v541 output verification: downloaded session output metadata/CSVs to ignored artifact dir `artifacts/kaggle_outputs/v541-public946-mirror-rare/`. Log confirms `USE_ONNX=True`, SED dir `tuckerarrants/bc2026-distilled-sed-public`, folds `sed_fold0.onnx`..`sed_fold4.onnx` loaded, SED output `(240,235)`, executed 2-way rank blend `60% Proto / 40% SED`, `Sonotype mirroring applied to 10 columns`, `Adaptive thresholding applied to 44 rare species`, and dry-run final `submission.csv` shape `(3,235)`. Total Kaggle session wall time to ready was ~423s / 7.1 min.
- CSV validation: `submission_protossm.csv` `(240,235)`, `submission_sed.csv` `(240,235)`, final `submission.csv` `(3,235)` sample-aligned by code-competition dry-run behavior.
- P1 cache/diagnostics: ran `scripts/birdclef_public946_cache_summary.py` on v541 dry-run outputs. Artifact dir `artifacts/pseudolabels/public946-v541-dryrun-cache-v1/` includes `teacher_proto.npz`, `teacher_sed.npz`, and `teacher_rankblend.npz`. On 190 label-overlap rows / 42 valid AUC classes: proto AUC `0.983987`, SED AUC `0.995976`, reconstructed rankblend AUC `0.992734`; top3 row recall proto `0.6263`, SED `0.9895`, rankblend `0.6421`.
- Interpretation: v541 is runtime-safe and faithfully executes the extra public 0.946 mirror/rare paths that v539 omitted. It should consume the next daily slot. The dry-run diagnostics again show the public SED stream is the dominant train-row teacher, while rankblend is more leaderboard-oriented than local label-AUC optimized.
- Next step: leave monitor running until v541 submits/scores. If v541 >= v539, make v541 the public946 anchor. If v541 falls, keep v539 as anchor and avoid more public postprocess forks until an internal minority blend or stronger student has validation.

## 2026-05-12 03:55 UTC - prioritized public946 spec written; v542 pushed/running

- Status: current scored best remains `v539 = 0.943`; `v541` is complete/verified and first pending behind daily cap; `v542` is RUNNING with no failure message and no output yet. Refreshed monitor pid `95675`, log `logs/submit_pending_birdclef_queue_20260512T034801Z_focus_v542_public946.log`, attempted `v541` and hit daily cap, so queue order remains `v541 -> v542 -> v538` once reset/complete allows.
- Open-solution mining: pulled and converted public kernels `afr1ste/birdclef-2026-0-946-updated-perch-sed` and `nina2025/birdclef-2026-ensemble-of-solutions-3` into `artifacts/public_kernels_20260512/` for inspection. Afr1ste v3 is a clean updated public946 V8 source; Nina documents 0.946 via Model_61/62 direct ensemble but is a large kitchen-sink notebook.
- Implemented/pushed `v542` as a controlled repo-owned port of Afr1ste updated public946: `yourslewis/bc26-v542-afr1ste-updated-public946`, version 1, kernel id `118877746`; push had no invalid sources. Added it to queue immediately after v541.
- Wrote planning spec `docs/BIRDCLEF_PUBLIC946_PRIORITIZED_SPEC_20260512.md` and linked it from `docs/BIRDCLEF_NEW_DIRECTIONS_SPECS.md`. The spec prioritizes scoring v541/v542, mining Nina narrowly, public-public ensembles, public946+V5/CLAP/BirdNET only after anchor score, and public946-teacher NFNet/EfficientNetV2-S AutoResearch.
- Next step: poll v542 completion/output. If valid, leave it queued after v541. Do not add another public fork until v541/v542 scores unless v542 fails.

## 2026-05-12 05:00 UTC - v542 verified; public946 NFNet/V2S AutoResearch smokes and full NFNet scale

- Status: current scored best remains `v539 = 0.943`. `v541` COMPLETE/verified and first pending after daily cap; monitor pid `95675` sleeping after cap. `v542` (`yourslewis/bc26-v542-afr1ste-updated-public946`, v1) completed with no failure message and should stay queued after v541.
- v542 verification: downloaded session output to `artifacts/kaggle_outputs/v542-afr1ste-updated-public946/` (ignored artifact dir). CSV shapes: `submission.csv` `(240,235)`, `submission_protossm.csv` `(240,235)`, `submission_sed.csv` `(240,235)`. Log confirms ONNX Perch (`USE_ONNX=True`), distilled SED processing complete, 60/40 rank blend executed, sonotype mirroring applied to 10 cols, adaptive thresholding applied to 44 rare species, and full dry-run train row_ids kept. Runtime to final output ~519s / 8.6 min.
- v542 dry-run cache diagnostics (`artifacts/pseudolabels/public946-v542-dryrun-cache-v1/`): on 190 overlap rows / 42 valid classes, proto AUC `0.984068`, SED AUC `0.995976`, final AUC `0.992525`, reconstructed rankblend AUC `0.992525`. This matches the expected public946 behavior and provides a cleaner full-train-row final dry-run output than v541.
- Added AutoResearch configs:
  - `pl_public946_sed_nfnet_5s_lr1e4_smoke.json` — eca_nfnet_l0, SED teacher, 5s/160mel, BCE, power1.0, max_rows256, ep3.
  - `pl_public946_sed_v2s_5s_lr1e4_smoke.json` — efficientnetv2_rw_s, SED teacher, pretrained, 5s/160mel, max_rows256, ep3.
  - `pl_public946_rankblend_nfnet_5s_lr1e4_smoke.json` — eca_nfnet_l0, rankblend teacher, 5s/160mel, max_rows256, ep3.
  - `pl_public946_rankblend_nfnet_5s_lr1e4_ep20_bestval.json` — scaled winner candidate, all 792 rows, ep20, restore best val AUC.
- Execution: initial NFNet smoke accidentally targeted busy GPU0 and OOMed because unrelated process pid `512484` held ~15.7GB. Relaunched smokes on free GPU1 in sequential wrapper `logs/pl_public946_gpu1_smokes_20260512T044736Z.log`; all completed.
- Smoke results:
  - SED->NFNet smoke: best val AUC `0.707440`, final all-row AUC `0.720263`, teacher AUC `0.995378`, corr `0.632919`, MAE `0.01272`, runtime 8.8s. SED target is too sparse/difficult for short NFNet smoke.
  - SED->V2S smoke: best val AUC `0.792906`, final all-row AUC `0.773891`, teacher AUC `0.995378`, corr `0.289202`, MAE `0.06159`, runtime 7.1s. Low correlation but weak standalone; possible diversity later, not first scale.
  - Rankblend->NFNet smoke: best val AUC `0.852589`, final all-row AUC `0.853951`, teacher AUC `0.990095`, corr `0.711918`, MAE `0.14460`, runtime 8.2s. Best smoke, selected for scale.
- Full rankblend->NFNet scale: launched `pl_public946_rankblend_nfnet_5s_lr1e4_ep20_bestval` on GPU1, log `logs/pl_public946_rankblend_nfnet_ep20_20260512T044954Z.log`. Completed in 100.4s; best epoch 20, best val AUC `0.981990` over 61 classes; final all-row student AUC `0.984806` over 75 classes vs teacher `0.994567`; student-teacher corr `0.924409`, MAE `0.07138`; TorchScript size 89.872MB.
- Blend diagnostic on 739 matched labeled rows / 75 valid classes: rankblend teacher AUC `0.994834`, full NFNet student AUC `0.985062`, corr `0.95677`. Best linear blend is teacher `0.95` + student `0.05` -> `0.994903` (+0.000069); best rank blend also student weight `0.05` -> `0.994887` (+0.000053). This is a small but real local blend lift; do not spend a submission slot until v541/v542 scores, but it is a viable private-robustness sidecar candidate at 2-5% weight.
- Next step: keep queue untouched until v541/v542 scores. For training, the next meaningful smoke is either full-scale V2S only if diversity is desired, or a rankblend->NFNet variant with 10s crop / power0.85 after anchor scoring.

## 2026-05-12 05:55 UTC - public946 rankblend student follow-up smokes; no new scale

- Status: current scored best remains `v539 = 0.943`. `v541` and `v542` are both COMPLETE/verified and queued behind daily cap (`v541 -> v542 -> v538`). Monitor pid `95675` remains alive/sleeping after cap. v510 remains complete/scored 0.927 with no active failure.
- Ran additional public946 rankblend student smokes on free GPU1, respecting the rule not to add submission candidates before v541/v542 score. Added configs:
  - `pl_public946_rankblend_v2s_5s_lr1e4_smoke.json`
  - `pl_public946_rankblend_nfnet_10s_lr1e4_smoke.json`
  - `pl_public946_rankblend_nfnet_5s_power085_lr1e4_smoke.json`
  - `pl_public946_rankblend_nfnet_5s_power115_lr1e4_smoke.json`
- Results vs previous best smoke (`rankblend->NFNet 5s power1.0`: final AUC `0.853951`, corr `0.711918`):
  - rankblend->V2S 5s power1.0: best val `0.838204`, final `0.835216`, corr `0.585914`, MAE `0.18511`; lower AUC despite lower corr, do not scale now.
  - rankblend->NFNet 10s power1.0: best val `0.844647`, final `0.847622`, corr `0.712828`, MAE `0.14399`; worse than 5s, so 10s crop not worth scaling.
  - rankblend->NFNet 5s power0.85: best val `0.844561`, final `0.851504`, corr `0.715902`, MAE `0.15129`; worse than power1.0.
  - rankblend->NFNet 5s power1.15: best val `0.853424`, final `0.851591`, corr `0.702949`, MAE `0.14768`; slightly lower final AUC than power1.0 despite lower corr.
- Decision: keep the already-scaled `rankblend->NFNet 5s power1.0 ep20` as the only current student sidecar candidate. Do not scale V2S, 10s, or power variants before anchor scores. Next best training action after v541/v542 score is either (a) package the 95/5 public946+NFNet private-robustness sidecar if anchor results justify, or (b) try a genuinely different V5/CLAP/BirdNET public stream.

## 2026-05-12 06:55 UTC - Nina Model_61/62 public ensemble mining; no v543 before scores

- Status: current scored best remains `v539 = 0.943`. `v541` and `v542` remain COMPLETE/verified and queued behind daily cap, with monitor pid `95675` sleeping after attempting v541. No duplicate submissions added; v510 remains complete/scored 0.927 and healthy.
- Track: P1 open-solution mining without adding a new submission before v541/v542 scores.
- Added `scripts/birdclef_public946_weight_grid.py`, an offline diagnostic that reconstructs public946 Proto/SED rank-blend variants with public gates (fake-only rescue, proto continuity, SED-only rescue, sonotype mirroring, rare-taxon suppression), evaluates label-overlap AUC/top-k, and reports correlations.
- Ran it on v542 dry-run outputs (`artifacts/kaggle_outputs/v542-afr1ste-updated-public946/`) with train labels/taxonomy. Output: `artifacts/public946/v542_weight_grid/summary.json`.
- Results on 190 overlap rows / 42 valid classes:
  - `proto0.40_sed0.60`: AUC `0.994484`, top3 `0.6526`.
  - `proto0.46_sed0.54`: AUC `0.993964`, top3 `0.6579`.
  - Nina `Model_61/62` direct proxy (`0.54/0.46` + `0.46/0.54` average): AUC `0.993627`, top3 `0.6474`.
  - exact `50/50`: AUC `0.993616`, top3 `0.6474`.
  - v542-style `60/40`: AUC `0.992525`, top3 `0.6263`.
  - `70/30`: AUC `0.991446`; `80/20`: AUC `0.989805`.
- Correlation vs v542 60/40 is extremely high for the Nina/50-50 family: `0.9974` for 54/46, `0.9930` for Nina direct proxy, `0.9930` for exact 50/50, `0.9863` for 46/54.
- Interpretation: Nina Model_61/62's clean extractable idea is effectively a 50/50-ish public Proto/SED rank-blend, not a genuinely new model stream. Local dry-run labels favor more SED-heavy blends, but this likely reflects train-label leakage because Afr1ste's public ablations note 50/50 tied 0.946 while Proto-heavy 70/30 and 80/20 fell to 0.944/0.942. Decision: do not create v543 before v541/v542 scores. If both miss 0.946, the only clean v543 worth considering is a 50/50 or 40/60 weight test, not a full Nina kitchen-sink port.

## 2026-05-12 07:55 UTC - V5/CLAP and BirdNET diversity-stream triage; queue hold

- Status: current scored best remains `v539 = 0.943`. `v541` and `v542` are still COMPLETE/verified and queued behind daily cap. Monitor pid `95675` remains alive/sleeping after attempting v541; no duplicate submissions were added.
- Track: P2 diversity-stream preparation without creating a new submission candidate before v541/v542 scores.
- Wrote `docs/BIRDCLEF_PUBLIC946_DIVERSITY_STREAM_TRIAGE_20260512.md` and linked it from the prioritized spec. It triages the two genuinely distinct public-stream families:
  - `needless090/birdclef-2026-perch-sed-lb-0-946-clap`: V5 ONNX trio/quintet plus optional CLAP probe. Observed expected V5 files (`v5_cluster_aware.onnx`, `v5_focal.onnx`, `v5_pseudo2.onnx`, `v5_pseudo.onnx`, `v5_external.onnx`), CLAP probe files (`clap_probe_W.npy`, `clap_probe_b.npy`, `clap_probe_fitmask.npy`), weights `V5_W=0.15`, `CLAP_W=0.10`, CLAP dynamic budget `45*60` sec. Risk: downloaded metadata has blank dataset sources for V5/CLAP, so a repo port must explicitly attach/validate `needless090/birdclef2026-sed-v5-trio` and `needless090/birdclef2026-clap-probe`; otherwise it silently degrades to plain public946.
  - `raunakdey07/birdclef-2026-birdnet-4-way-rank-blend`: adds BirdNET 6K TFLite and custom EffNet ONNX with intended 4-way weights Proto `0.40`, SED `0.30`, BirdNET `0.15`, EffNet `0.15`. Risk: local metadata does not list the BirdNET model or custom EffNet notebook/model source despite code paths under `/kaggle/input/models/...` and `/kaggle/input/notebooks/...`; exact attachable refs must be resolved before any port.
- Decision: no new kernel push/queue this run. After v541/v542 scores, prefer V5/CLAP source validation as the first distinct diversity-stream candidate; BirdNET/EffNet remains second due source/runtime fragility. Nina remains held as a high-correlation weight perturbation, not a new stream.

## 2026-05-12 09:15 UTC - Source audit for public946 diversity streams; no new queue

- Status check: current scored best remains `v539 = 0.943`. Latest visible submissions are still v539/v527/v531/v532/v537 from the 2026-05-12 UTC cap; v541 and v542 kernels are COMPLETE/no failure but not yet submitted due daily cap. Monitor pid `95675` is alive; latest log shows it attempted v541 and slept after `Submission allowance (5)` with ~20h remaining. No duplicate submissions added.
- Track: P2 source-clean diversity prep while holding new submissions until v541/v542 score.
- V5/CLAP source audit:
  - `GetKernel` for `needless090/birdclef-2026-perch-sed-lb-0-946-clap` returns two blank dataset refs plus public946 refs. Embedded notebook JSON exposes numeric datasetVersion IDs `16013757`/`16003884` (datasetIds `10267502`/`10025194`) for the V5/CLAP extras.
  - Bearer Dataset API lookups for likely slugs `needless090/birdclef2026-sed-v5-trio` and `needless090/birdclef2026-clap-probe` returned 403; public dataset search returned no rows. Conclusion: V5/CLAP remains source-blocked unless we can resolve numeric sources or recreate equivalent datasets. Do not queue a V5/CLAP candidate yet.
- BirdNET/EffNet source audit:
  - BirdNET model is source-clean and attachable: `shadiakiki1/birdnet-analyzer/TfLite/birdnet_global_6k_v2.4_model_fp32-1/3`; model API confirms TFLite instance v3 (~52MB).
  - Raunak 4-way embedded JSON includes BirdNET model source and a custom EffNet notebook-output `kernelVersion` source `317846744`, but `GetKernel` for `raunakdey07/offline-training-efficientnet-b0-focal-recording` returns 403. Full EffNet 4-way remains blocked.
  - Found `claudedevore/birdclef-2026-r0946-birdnet-3way-submit` as a safer public reference: metadata includes BirdNET + normal public946 sources and skips unavailable EffNet. Saved ignored local audit artifact under `artifacts/public_kernels_20260511/birdclef-2026-r0946-birdnet-3way-submit.*`. Caveat: fetched snapshot has repeated EffNet-skip cells and no clean final blend cell in extracted source, so a repo port should extract only the BirdNET inference block and write our own explicit 3-way rank blend.
- Updated `docs/BIRDCLEF_PUBLIC946_DIVERSITY_STREAM_TRIAGE_20260512.md` and prioritized spec. New source-clean fallback after v541/v542: if V5/CLAP remains blocked, prefer a minimal BirdNET-only 3-way candidate over the 4-way EffNet path.

## 2026-05-12 09:58 UTC - Queue hold + v542 output revalidation

- Status check: current scored best remains `v539 = 0.943`. Latest visible submissions are unchanged from UTC cap: v539 scored 0.943; v527/v531/v537 scored 0.930; v532 timed out. v541/v542/v510/v538 kernels are all COMPLETE/no failure.
- Queue: monitor pid `95675` alive. Latest log `logs/submit_pending_birdclef_queue_20260512T034801Z_focus_v542_public946.log` shows v541 submit attempt hit daily allowance with ~20h remaining; no duplicate submissions added.
- Track: P0 queue/verification hold; no new candidate because spec says not to add submissions before v541/v542 scores.
- v542 revalidation:
  - output files present under `artifacts/kaggle_outputs/v542-afr1ste-updated-public946/`: `submission.csv`, `submission_protossm.csv`, `submission_sed.csv`, `perch_meta.parquet`, `session_output_response.txt`.
  - log confirms ONNX Runtime install, ONNX Perch, SED folds loaded, SED processing complete, standard 2-way 60/40 rank blend, sonotype mirroring on 10 columns, adaptive rare thresholding on 44 species, full train-row dry-run output, and total runtime about 528s.
  - CSV sanity via pandas venv: `submission.csv` `(240,235)`, no NaNs, min `0.00375`, max `1.0`, mean `0.501526`; `submission_protossm.csv` `(240,235)`, no NaNs; `submission_sed.csv` `(240,235)`, no NaNs. Proto/SED dry-run corr `0.405286`; v542 final corr vs Proto `0.475426`, vs SED `0.236754`, confirming the final output is not a trivial copy of either stream.
- v541 sanity retained: final dry-run aligned sample shape `(3,235)` and proto/SED full diagnostic files `(240,235)`, no NaNs. This is expected because v541 aligns public dry-run `submission.csv` to sample submission while v542 preserves full train rows for validation.
- Updated prioritized spec checklist: both v541/v542 complete+verified; hold queue; if both miss, choose one clean public weight test vs source-clean BirdNET-only 3-way; V5/CLAP remains blocked until source refs resolve.

## 2026-05-12 10:58 UTC - Cap hold; prepared BirdNET-only 3-way port plan

- Status check: current scored best remains `v539 = 0.943`; latest visible submissions unchanged. v541/v542/v510/v538 kernels are COMPLETE/no failure. Monitor pid `95675` is alive and sleeping on daily cap after attempting v541; log still shows `Submission allowance (5)` and a 72120s sleep from 03:48 UTC. No duplicate submissions added.
- Spec read: current addendum deprecates 0.927 plateau and says to hold new candidates until v541/v542 score unless a candidate fails.
- Track: P0 hold + P2 preparation only; no Kaggle push/queue.
- Prepared `docs/BIRDCLEF_PUBLIC946_BIRDNET3_PORT_PLAN_20260512.md` as the next source-clean diversity fallback if v541/v542 miss and V5/CLAP remains blocked.
  - Base: fork v542 into a later `v543-public946-birdnet3` candidate only after scores land.
  - Metadata addition: `shadiakiki1/birdnet-analyzer/TfLite/birdnet_global_6k_v2.4_model_fp32-1/3` alongside Perch v2.
  - Insert BirdNET inference after `submission_sed.csv`, before final rank blend.
  - Use central 3s of each 5s/48k BirdNET window; write `submission_birdnet.csv`; fail loudly if model/labels missing in final candidate.
  - First blend recommendation: conservative rank blend `52% Proto / 38% SED / 10% BirdNET`, keeping v542 post-blend gates unchanged. Rationale: BirdNET is true acoustic diversity but label mapping is sparse/brittle; public reference used 15% BirdNET with custom EffNet, so 10% is safer for first source-clean slot.
  - Validation gates: BirdNET mapped class count, TFLite interpreter init, row alignment across proto/sed/birdnet/final, `(240,235)` dry-run no NaNs, explicit 3-way log, runtime safe.
- Linked the BirdNET plan from the prioritized spec. This is preparation only; no candidate will be queued before v541/v542 scores.

## 2026-05-12 11:50 UTC - Queue hold recheck; monitor ETA validated

- Status check: public LB best remains `v539 = 0.943`; no new scored rows since the 2026-05-12 UTC cap set. Latest visible: v539 0.943, v527/v531/v537 0.930, v532 timeout, v526 timeout, v522 0.927, v521 0.928.
- Kernel status: v541, v542, v510, and v538 are all COMPLETE/no failure.
- Queue: monitor pid `95675` remains alive (`STAT SN`, elapsed ~7h58m). Latest log is still `logs/submit_pending_birdclef_queue_20260512T034801Z_focus_v542_public946.log`; it skipped already-submitted rows through v539, attempted v541, hit daily cap, and slept `72120s`.
- ETA check: log mtime/sleep imply wake around `2026-05-12T23:50:02Z`, roughly 10 minutes before UTC reset. This should let it retry v541 promptly without restart. No action needed and no duplicate submission added.
- Decision: continue strict hold. Do not push/queue BirdNET3 or public-weight v543 until v541/v542 scores land; BirdNET3 plan remains prepared only.

## 2026-05-12 12:55 UTC - Queue hold; v510 output reverified via Kaggle API

- Status check: public LB best remains `v539 = 0.943`; no new scored submissions since the 2026-05-12 UTC cap. Latest visible remains v539 0.943; v527/v531/v537 0.930; v532 and v526 timed out; v522 0.927; v521 0.928.
- Kernel status: v541, v542, v510, and v538 are all COMPLETE/no failure.
- Queue: monitor pid `95675` alive and still sleeping on the daily-cap backoff before v541; latest log remains `logs/submit_pending_birdclef_queue_20260512T034801Z_focus_v542_public946.log` with cap sleep `72120s`. No restart and no duplicate submissions.
- v510 recheck (requested legacy A+G diagnostic): `ApiListKernelSessionOutput` for `yourslewis/bc26-v510-real-sed-bundle-blend-005` confirms `submission.csv` exists. Log confirms real SED bundle path was active: manifest found at `yourslewis/bc26-sed-nfnet-v13v15-bundle-v1`, loaded `6/6` TorchScript models, real SED runtime `214.4s`, applied blend weight `0.05`, saved `submission.csv (240,235)`, wall time `370.6s`, and v510 already scored a safe 0.927 tie. No v510 fix needed.
- Decision: strict hold continues. Do not push/queue BirdNET3 or public-weight v543 before v541/v542 scores land.

## 2026-05-12 13:50 UTC - Queue script order guard revalidated

- Status check: public LB best remains `v539 = 0.943`; latest visible submissions unchanged. v541/v542/v510/v538 kernels are all COMPLETE/no failure.
- Queue monitor: pid `95675` alive (`STAT SN`, elapsed ~9h57m) and still sleeping on cap before v541. Latest log unchanged: skipped through v539, attempted v541, hit daily allowance, slept `72120s`.
- Queue script validation: `python3 -m py_compile scripts/submit_pending_birdclef_queue.py` passed. Relevant pending entries are ordered `v541 -> v542 -> ... -> v538`, with focus priority `... v539, v541, v542, v527, v531, v532, v537, v538`.
- Duplicate guard: Bearer submission list with page size 200 shows zero submitted descriptions for v541/v542/v538, so the monitor has not submitted them yet and adding another monitor would risk duplication. No restart performed.
- Decision: strict hold continues. Next real action is monitor retry near UTC reset; do not queue BirdNET3/public-weight v543 before v541/v542 scores.
