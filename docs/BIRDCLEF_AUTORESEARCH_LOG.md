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

### v545 CLAP sidecar lower-weight gate — 2026-05-13 23:30 UTC

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


### Spec B existing student-pool blend audit — 2026-05-15 12:10 UTC

- **Status check:** current public best remains **0.946**. Latest submissions `v558/v551/v549/v548/v547` all scored `0.946`; `v510` kernel COMPLETE/no failure. No Kaggle submission used.
- **Hypothesis:** before training another adjacent blended-teacher student, search existing aligned student artifacts for lower-correlation blend signal against `teacher_sed85_rankblend15.npz`.
- **Implementation:** added `scripts/birdclef_student_pool_blend_audit.py`, which scans `student_predictions.npz` artifacts, verifies row/label alignment, computes standalone AUC/correlation, and sweeps small teacher+student blend weights.
- **Audit:** trainer artifact `artifacts/pseudolabels/audits/public946_sed85_rankblend15_student_pool_audit_20260515T1155Z.json`; scanned `82` student files, `33` aligned. Teacher baseline `0.997018454` over 75 valid classes.
- **Best single sidecar:** `pl-r2-v2s-v508-soft-p100-5s-pretrained-lr1e4-ep20-bestval` (`efficientnetv2_rw_s`, TorchScript `88.739 MB`) standalone AUC `0.983987`, corr vs teacher `0.3752`; best blend teacher `0.95` + V2S `0.05` = `0.997187110`, lift `+0.000168656`.
- **Pair sweep:** trainer artifact `artifacts/pseudolabels/audits/public946_sed85_rankblend15_student_pool_pair_sweep_20260515T1205Z.json`. Best blend: teacher `0.90` + V2S v508 `0.06` + B0 soft-anchor v508 `0.04` = `0.997228528`, lift `+0.000210074`, corr `0.98262`.
- **Decision:** best local sidecar signal so far, but still tiny and only validated on 792 labeled train-soundscape rows. Do not spend an immediate slot without packaging/runtime verification. Next candidate if a slot is available: source-clean two-student sidecar using V2S `0.06` + B0 soft-anchor `0.04` into the public946 anchor.


### Spec B blended-teacher B0 Soft-AUC curriculum — 2026-05-15 10:55 UTC

- **Status check:** current public best remains **0.946**; latest submissions `v558/v551/v549/v548/v547` all scored `0.946`. No Kaggle submission used. `v510` remains COMPLETE/no failure.
- **Smoke:** added `configs/birdclef/pl_public946_sed85_rankblend15_b0_5s_softauc_w0005_smoke_20260515.json` using B0 + external-pretrain init, 256 rows / 3 epochs, `loss_name=bce_soft_auc`, `auc_loss_weight=0.005`. Completed CUDA in `6.161s`; final AUC `0.916208` over 42 classes vs teacher `0.995304`; best val `0.931595`, corr `0.56944`. It beat B0+BCE smoke `0.900997`, so scaled.
- **Scale:** added `configs/birdclef/pl_public946_sed85_rankblend15_b0_5s_softauc_w0005_ep20_20260515.json` and ran all 792 rows / 20 epochs. Completed CUDA in `47.366s`; best val AUC `0.992015`; final student AUC `0.989343` over 75 classes vs blended teacher `0.997018`; corr `0.96012`, MAE `0.02010`, TorchScript `15.391 MB`.
- **Blend gate:** no lift. Best checked weight `0.0025` gives AUC `0.9970182`, essentially equal/slightly below teacher `0.99701845`; larger weights drop.
- **Decision:** kill Soft-AUC curriculum for this target. It passed smoke but underperformed B0+BCE at scale (`0.989343` vs `0.992137/0.991832`) and gives no blend lift. Do not package/submit.


### Spec B blended-teacher RegNetY learner pivot — 2026-05-15 09:55 UTC

- **Status check:** current public best remains **0.946**; latest submissions `v558/v551/v549/v548/v547` all scored `0.946`. No Kaggle submission used. `v510` remains COMPLETE/no failure.
- **Run:** added `configs/birdclef/pl_public946_sed85_rankblend15_regnety008_5s_smoke_20260515.json` and ran RegNetY-008 (`pretrained=true`, lr `1e-4`) against `teacher_sed85_rankblend15.npz`, 256 rows / 3 epochs. Completed on CUDA in `6.152s`, TorchScript `23.42 MB`.
- **Metrics:** final student AUC `0.891280` over 42 classes vs teacher `0.995304`; best val AUC `0.906245`; corr `0.71449`, MAE `0.04140`.
- **Decision:** kill direct RegNetY scaling for this target. It did not beat the blended-teacher B0 smoke (`0.900997`) and is below the B0 scaled path; do not package/submit.


### Spec B blended-teacher B0 second-seed robustness — 2026-05-15 08:55 UTC

- **Status check:** current public best remains **0.946**; latest submissions `v558/v551/v549/v548/v547` all scored `0.946`. No Kaggle submission used. `v510` remains COMPLETE/no failure.
- **Run:** added `configs/birdclef/pl_public946_sed85_rankblend15_b0_5s_ep20_seed43_20260515.json` and ran same B0 + external-pretrain setup against `teacher_sed85_rankblend15.npz`, 792 rows / 20 epochs, seed `43`. Completed on CUDA in `21.506s`.
- **Metrics:** seed43 final student AUC `0.991832` over 75 classes vs blended teacher `0.997018`; best val AUC `0.994676`; corr `0.97008`, MAE `0.01643`, TorchScript `15.391 MB`.
- **Robust blend gate:** seed42 standalone `0.992137`, seed43 `0.991832`, two-seed ensemble `0.993027`, teacher `0.997018`. Best blend is still seed42 at `w=0.01`: `0.997046`; two-seed ensemble best `w=0.05`: `0.997041`; seed43 best `w=0.005`: `0.997038`.
- **Decision:** robust signal but too small and too correlated after blending for a Kaggle slot. Keep artifacts; do not package/submit unless later independent validation/private proxy strengthens the case.


### Spec B blended-teacher B0 smoke + scale — 2026-05-15 07:55 UTC

- **Status check:** current public best remains **0.946**; latest submissions `v558/v551/v549/v548/v547` all scored `0.946`. No Kaggle submission used. `v510` remains COMPLETE/no failure.
- **B0 smoke:** added `configs/birdclef/pl_public946_sed85_rankblend15_b0_5s_smoke_20260515.json` and ran B0 + external-pretrain init on 256 rows / 3 epochs against `teacher_sed85_rankblend15.npz`. It completed on CUDA in `4.845s`, final AUC `0.900997` over 42 classes vs teacher `0.995304`, corr `0.56112`. This beat old B0 SED smoke `0.818694` and old ConvNeXt rankblend smoke `0.882870`, so it passed scale gate.
- **B0 scale:** added `configs/birdclef/pl_public946_sed85_rankblend15_b0_5s_ep20_20260515.json` and ran all 792 rows / 20 epochs. Completed on CUDA in `21.326s`; best val AUC `0.992890`; final student AUC `0.992137` over 75 classes vs blended teacher `0.997018`; corr `0.96336`, MAE `0.01921`, TorchScript `15.391 MB`.
- **Blend gate:** copied metrics and `blend_gate.json`. Best local blend into blended teacher is `student_weight=0.01`: AUC `0.997046` vs teacher baseline `0.997018` (+`0.000028`). This also improves SED teacher at `w=0.10` (`0.996870` vs `0.996743`), but lift is tiny.
- **Decision:** no packaging/submission yet. This is the best student artifact so far, but local gain is too small after repeated public946 ties. Next useful action is robustness (second seed/fold) or a different learner/curriculum against the blended teacher, not Kaggle quota.


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


### v558 conditional submit monitor prepared while v551 pending — 2026-05-15 00:48 UTC

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

## 2026-05-06 23:30 UTC — v512 verified + prioritize real SED submissions at reset

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

## 2026-05-12 00:48 UTC - full public946 SED B0 student scaled while v539 pending

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

### Spec B v559 multi-sidecar dry-run gate — 2026-05-15 13:20 UTC

- **Status check:** current public best remains **0.946**. Latest visible submissions `v558/v551/v549/v548/v547` all scored `0.946`; `v510` kernel remains COMPLETE/no failure. PR #229 was merged, so this work started a fresh branch `feature/v559-public946-v2s-b0sidecar-prep-20260515` from updated `main`. No Kaggle submission used.
- **Hypothesis:** the 792-row blended-teacher audit found a larger local sidecar signal from V2S + B0 soft-anchor, but v552 showed teacher-cache lift can disappear after the actual public946 final rank/gate mechanics. Gate the candidate against the real v542 Kaggle dry-run output before packaging.
- **Implementation:** added `scripts/birdclef_public946_multi_sidecar_weight_grid.py`, a reusable no-submit dry-run gate for multiple named rank sidecars. It row-aligns sidecar CSVs to a base public946 `submission.csv`, Cartesian-sweeps named weights with a max total sidecar cap, and reports macro AUC, top-k row recall, correlation, MAE, and max displacement on train-soundscape overlap.
- **Inputs:** v542 dry-run `artifacts/kaggle_outputs/v542-afr1ste-updated-public946/submission.csv`; sidecar CSVs materialized from existing student NPZs: V2S `pl-r2-v2s-v508-soft-p100-5s-pretrained-lr1e4-ep20-bestval` and B0 soft-anchor `pl-r1-b0-v508-soft-anchor-p98n05-lr3e4-ep12`.
- **Gate artifact:** `artifacts/blend_grids/v559_v2s_b0_multi_sidecar_gate_20260515T1315Z.json`.
- **Result:** base v542 dry-run rank anchor macro AUC `0.992524901` over 42 valid classes / 190 matched rows. Best grid row was V2S `0.005` + B0 soft-anchor `0.010`, macro AUC `0.992560140` (`+0.000035240`), top5 recall `0.6526` vs base `0.6316`, MAE `0.00297`. B0-only `0.020` was close at `0.992553682`; the originally tempting 792-row pair weights (`0.06/0.04`) were not favored by this stricter gate.
- **Decision:** do **not** package/push v559 yet. The candidate is directionally positive and runtime-light, but the stricter dry-run lift is too small for a blind Kaggle slot after repeated public946 sidecars tied. Keep the script and gate output as the next-slot prep; if a slot must be filled, safest candidate is a very small rank sidecar around V2S `0.005` + B0 soft-anchor `0.010`, not the earlier 6%/4% pair.

### Spec B v560 direct blended-teacher V2S student + no-submit Kaggle runtime gate — 2026-05-15 14:00 UTC

- **Status check:** current public best remains **0.946**. Latest submissions `v558/v551/v549/v548/v547` all scored `0.946`; `v510` kernel remains COMPLETE/no failure. PR #230 is open/mergeable. No competition submission used.
- **Hypothesis:** older V2S trained on stale v508 was the lowest-correlation sidecar found in the pool audit. Train V2S directly against the stronger public946 `teacher_sed85_rankblend15.npz` target to see if it gives a more packageable diversity stream.
- **Smoke configs:** added `configs/birdclef/pl_public946_sed85_rankblend15_v2s_5s_lr1e4_smoke_20260515.json` (256 rows / 3 epochs) and `configs/birdclef/pl_public946_sed85_rankblend15_v2s_5s_lr1e4_ep8_smoke_20260515.json` (256 rows / 8 epochs). Both use EfficientNetV2-RW-S, pretrained=true, 5s/160 mel, lr `1e-4`, soft BCE, teacher power `1.0`, no mixup, target `teacher_sed85_rankblend15.npz`.
- **Smoke result:** 3 epochs underfit (`final_all_student_vs_truth=0.789590`, corr `0.2969`), but 8 epochs recovered to `final_all_student_vs_truth=0.928674`, best val `0.930854`, corr `0.6613`; it was still improving, so it passed a slow-starter scale gate.
- **Scale config:** added `configs/birdclef/pl_public946_sed85_rankblend15_v2s_5s_lr1e4_ep20_20260515.json`; 792 rows / 20 epochs. Completed on CUDA in `52.827s`, TorchScript `88.74 MB`. Final all-row student AUC `0.990667` over 75 classes vs teacher `0.997018`; best val AUC `0.986623`, corr `0.956984`, MAE `0.019636`.
- **Blend gate:** trainer artifact `artifacts/pseudolabels/students/pl-public946-sed85-rankblend15-v2s-5s-lr1e4-ep20-20260515/blend_gate.json`. Best teacher+student local blend was V2S weight `0.075`, AUC `0.997058764`, lift `+0.000040310`; pair blends with B0 seeds did not materially improve (`+0.000040688` max).
- **Strict v542 dry-run gate:** materialized `submission_direct_v2s.csv` from the student NPZ and ran `birdclef_public946_sidecar_weight_grid.py` against v542 dry-run output. Artifact `artifacts/blend_grids/v560_direct_v2s_official_gate_20260515T1410Z.json`: direct-V2S standalone rank AUC `0.975338`, corr vs anchor `0.7719`; best sidecar weight `0.03`, macro AUC `0.992606780`, lift `+0.000081879` vs v542 dry-run anchor `0.992524901`, top5 recall `0.6632` vs base `0.6316`, MAE `0.00441`.
- **Kaggle packaging:** because the strict dry-run lift was stronger than v559, packaged a private dataset `yourslewis/bc26-public946-direct-v2s-student-v1` (zip `78 MB`, SHA256 `3f536693807a0b239cb63d5a0879833f0dcf033345f3130c1a9ffd17e124b104`) with `model_torchscript.pt` and `sed_bundle_manifest.json`.
- **Kernel:** added/pushed no-submit candidate `yourslewis/bc26-v560-public946-direct-v2s-r003`, version 1. It forks v552/public946, attaches the direct-V2S dataset, writes `submission_direct_v2s_student.csv`, and applies `STUDENT_RANK_BLEND=0.03` after public946 gates. Push returned no invalid sources. Kernel status after push: RUNNING/no failure. Started no-submit monitor `logs/monitor_v560_direct_v2s_gate_20260515T135758Z.log`; it downloads outputs and gates them but does **not** submit to the competition.
- **Decision:** do not submit unless v560 completes cleanly and the no-submit gate remains sane. Even then, treat as optional next-slot candidate; repeated public946 sidecars have tied, so this is runtime/prep work, not an automatic slot spend.

### v560 complete, no-submit gate passed, submitted — 2026-05-15 14:55 UTC

- `v560` (`yourslewis/bc26-v560-public946-direct-v2s-r003`) completed successfully with no failure message. Monitor `logs/monitor_v560_direct_v2s_gate_20260515T135758Z.log` downloaded and validated `submission.csv`, `submission_direct_v2s_student.csv`, `submission_sed.csv`, and `submission_protossm.csv`, all `(240,235)` with no NaNs.
- Gate artifact: `artifacts/blend_grids/v560_direct_v2s_sidecar_weight_grid_20260515T141004Z.json`. Direct-V2S sidecar standalone rank AUC `0.975319`, corr vs v542 anchor `0.771902`. Best blend is `sidecar_0.0300`, macro AUC `0.992606780` over 42 valid classes / 190 matched rows, lift `+0.000081879` vs v542 dry-run anchor `0.992524901`, top5 row recall `0.663158` vs anchor `0.631579`, corr `0.999816`, MAE `0.004414`.
- Added `scripts/submit_v560_when_ready.py` and used it after confirming no duplicate v560 submission was visible. It verified COMPLETE status and `submission.csv`, then submitted kernel version 1 with description `v560: Public946 v542 plus direct blended-teacher V2S student rank sidecar 3%`.
- Submission ref `52683717` is currently `PENDING` score. This used one Kaggle competition submission slot; no duplicate submission was made.

### v560 scored 0.945 + XC-initialized V2S blended-teacher branch killed — 2026-05-15 15:55 UTC

- **Status check:** `v560` scored **0.945**, below the 0.946 public946 plateau. Latest scored submissions are `v560=0.945`, `v558=0.946`, `v551=0.946`, `v549=0.946`, `v548=0.946`. `v510` remains COMPLETE/no failure. This confirms that tiny local/dry-run public946 sidecar lifts are not translating to public LB. Stop spending slots on micro-sidecars unless a substantially stronger, genuinely new artifact appears.
- **Pivot hypothesis:** the direct ImageNet-pretrained V2S sidecar dropped, but the previous pool audit suggested V2S can provide lower-correlation signal. Test whether the existing XC external-pretrain V2S checkpoint changes the blended-teacher student enough to matter.
- **Smoke config:** `configs/birdclef/pl_public946_sed85_rankblend15_v2s_xc_extinit_5s_lr1e4_ep8_smoke_20260515.json`; V2S, XC external-pretrain checkpoint `xc-v2s-q3-cap80-external-pretrain-balanced-ep12-bestloss`, 5s/128 mel, lr `1e-4`, 256 rows / 8 epochs, soft BCE against `teacher_sed85_rankblend15.npz`. Result: final all-row AUC `0.949310` over 42 classes, best val `0.931553`, corr `0.809742`, runtime `12.619s`. This beat the direct/ImageNet V2S 8-epoch smoke (`0.928674`), so scaled.
- **Scale config:** `configs/birdclef/pl_public946_sed85_rankblend15_v2s_xc_extinit_5s_lr1e4_ep20_20260515.json`; 792 rows / 20 epochs. Completed on CUDA in `60.308s`; final student AUC `0.989274` over 75 classes, best val AUC `0.988724`, corr `0.957270`, MAE `0.019614`, TorchScript `88.74 MB`.
- **Blend gate:** artifact `artifacts/pseudolabels/students/pl-public946-sed85-rankblend15-v2s-xc-extinit-5s-lr1e4-ep20-20260515/blend_gate.json`. Best teacher+student weight was only `0.0025`, AUC `0.997040669`, lift `+0.000022215`; larger weights quickly flatten/drop.
- **Decision:** kill XC-initialized V2S for packaging/submission. It improves the small smoke but loses to the direct V2S full run (`0.989274` vs `0.990667`) and has weaker blend lift (`+0.000022` vs `+0.000040`). Combined with v560's 0.945 score, do not package another V2S sidecar. Next work should leave public946 micro-sidecars and move to a materially different training/evaluation direction.

### v560 confirms micro-sidecar stop + NFNet blended-teacher smoke blocked by trainer SSH — 2026-05-15 17:05 UTC

- **Status check:** current public best remains **0.946**. `v560` scored **0.945**, while `v558/v551/v549/v548/v547` remain `0.946`; `v510` still reports COMPLETE/no failure. This makes the prior v560/v559 local-gate evidence insufficient and confirms the stop rule: do not spend more slots on tiny public946 sidecars.
- **Track:** Spec B/D model-zoo diversity after V2S/public946 micro-sidecars failed live LB. Chose NFNet because previous public946-rankblend NFNet existed but no direct `teacher_sed85_rankblend15.npz` NFNet check had been run.
- **Config added:** `configs/birdclef/pl_public946_sed85_rankblend15_nfnet_5s_lr1e4_smoke_20260515.json`; ECA-NFNet-L0, 5s/160 mel, pretrained=false, lr `1e-4`, 256 rows / 3 epochs, soft BCE against `teacher_sed85_rankblend15.npz`, seed `45`.
- **Run attempt:** foreground SSH/CUDA smoke launch to trainer (`192.168.0.10`) started but produced no first-epoch output within the cron window. Subsequent independent SSH checks timed out during banner exchange, suggesting trainer SSH/runtime congestion rather than a validated model result. The blocking local session was killed to avoid hanging the cron.
- **Decision:** mark NFNet blended-teacher smoke as **blocked/incomplete**, not failed as a model. Do not scale or package. Next run should first verify trainer reachability/process state; if clean, rerun as a durable `nohup` job with log tail rather than foreground SSH, or skip NFNet if the host remains unstable.

### Trainer SSH still blocked after v560 drop — 2026-05-15 18:00 UTC

- Follow-up diagnostics after the NFNet foreground attempt: ICMP ping to `192.168.0.10` is healthy (`3/3`, ~1ms) and TCP port 22 accepts connections, but SSH does not complete banner/auth (`Connection timed out during banner exchange`, then `Connection closed by 192.168.0.10 port 22`).
- Found and killed a stale local `rsync` process from the earlier interrupted NFNet config sync (`rsync -az ... pl_public946_sed85_rankblend15_nfnet_5s_lr1e4_smoke_20260515.json ...`), but SSH still timed out on three retries afterward.
- Validated locally that the NFNet smoke config parses and `scripts/birdclef_pseudolabel_student_train.py` compiles. No model result exists yet; trainer-side process state cannot be verified until SSH recovers.
- Decision: leave NFNet blended-teacher smoke blocked. Next safe action is host/service recovery or waiting for SSH to drain; do not launch additional remote work through hanging SSH sessions.

### Public946 micro-sidecar stop rule codified — 2026-05-15 19:00 UTC

- Trainer remains network-reachable but SSH-unusable (`ping` and TCP 22 work; SSH closes/times out during banner/auth), so no remote GPU NFNet rerun was safe.
- Codified the v560 lesson in `docs/BIRDCLEF_PUBLIC946_STOP_RETUNE_NEXT_SIGNAL_20260515.md` and `docs/BIRDCLEF_PUBLIC946_PRIORITIZED_SPEC_20260512.md`: `v560=0.945` despite clean runtime and positive local gate means public946 micro-sidecars/low-weight trained-student perturbations should not consume more slots without materially stronger out-of-sample/OOF evidence or a genuinely new model/source.

### Guarded NFNet blended-teacher smoke launcher — 2026-05-15 19:55 UTC

- Trainer SSH remains unhealthy on direct preflight (`Connection timed out during banner exchange`), so no remote GPU job was launched.
- Added `scripts/launch_nfnet_pseudolabel_smoke_if_trainer_ready.sh` to make the next NFNet attempt safe and non-blocking. It checks SSH with BatchMode/ConnectTimeout first, exits `75` without launching if SSH is unhealthy, syncs the NFNet smoke config only after preflight, then launches `scripts/birdclef_pseudolabel_student_train.py --config configs/birdclef/pl_public946_sed85_rankblend15_nfnet_5s_lr1e4_smoke_20260515.json` via remote `nohup` and prints pid/log.
- Local validation: `bash -n` passed, config JSON parses, and the script was exercised against the current blocked trainer state; it exited `75` and did not launch remote work.

### Status/source-mining pass while trainer SSH blocked — 2026-05-15 20:55 UTC

- Current public best remains **0.946**. Latest submissions: `v560=0.945`, `v558=0.946`, `v551=0.946`, `v549=0.946`, `v548=0.946`. `v510` and `v560` Kaggle kernels are COMPLETE with no failure message.
- Queue/process check found no active pending-submission monitor requiring restart; no duplicate submissions were added.
- Exercised `scripts/launch_nfnet_pseudolabel_smoke_if_trainer_ready.sh` again. Trainer SSH failed fast (`Connection closed by 192.168.0.10 port 22`), launcher exited `75`, and no remote GPU job was launched.
- Non-GPU pivot: searched for newer public BirdCLEF 2026 Perch/ProtoSSM/SED/CV9245 sources. Search only surfaced the already-known Nina `birdclef-2026-onnx-perch-proto-sed` notebook and no distinct new Kaggle dataset/artifact source. Keep public946 micro-sidecars stopped; next useful work still depends on trainer recovery for NFNet/student OOF or a genuinely new source appearing.

### Kaggle kernel output verifier while trainer SSH blocked — 2026-05-15 22:00 UTC

- Trainer SSH remains blocked; guarded NFNet launcher failed fast with banner timeout and exited `75`, so no GPU job was launched.
- Added read-only verifier `scripts/birdclef_kernel_output_verify.py` for Bearer-backed Kaggle SDK output checks. It checks latest kernel status, output file names, and required log markers without downloading large outputs.
- Validation: `python3 -m py_compile scripts/birdclef_kernel_output_verify.py` passed. Verifier confirmed `bc26-v510-real-sed-bundle-blend-005` is COMPLETE, has `submission.csv`, and log markers `Applied real SED bundle blend` / `submission.csv saved`. It also confirmed `bc26-v560-public946-direct-v2s-r003` is COMPLETE with `submission.csv`, `submission_direct_v2s_student.csv`, `submission_sed.csv`, and `submission_protossm.csv`, plus direct-V2S sidecar log markers.
- Current public score state is unchanged: best `0.946`; latest scored `v560=0.945`, `v558/v551/v549/v548=0.946`. No new submissions or queue changes.

### Kernel verifier presets while trainer SSH blocked — 2026-05-15 23:00 UTC

- Trainer SSH is still blocked: guarded NFNet launcher timed out during banner exchange and exited `75`, so no remote GPU job was launched.
- Improved `scripts/birdclef_kernel_output_verify.py` with `--preset v510-real-sed` and `--preset v560-direct-v2s` so future cron passes can verify known kernel files/log markers without restating fragile marker strings.
- Validation: `python3 -m py_compile scripts/birdclef_kernel_output_verify.py` passed; both preset checks returned `ok=true`. Current public score state unchanged: best `0.946`, latest `v560=0.945`, `v558/v551/v549/v548=0.946`; no monitor process needed restart and no new submission was made.

### All-preset kernel verifier mode while trainer SSH blocked — 2026-05-16 00:00 UTC

- Trainer SSH remains blocked: guarded NFNet launcher timed out during banner exchange and exited `75`; no remote GPU job was launched.
- Current public score state is unchanged: best `0.946`, latest `v560=0.945`, `v558/v551/v549/v548=0.946`; v510/v560 kernels remain COMPLETE with no failure; no monitor process needed restart and no submission was made.
- Improved `scripts/birdclef_kernel_output_verify.py` with `--all-presets`, which verifies every known preset in one read-only Kaggle SDK call sequence and exits nonzero if any preset fails. Validation: `python3 -m py_compile` passed and `scripts/birdclef_kernel_output_verify.py --all-presets --pretty` returned top-level `ok=true` for `v510-real-sed` and `v560-direct-v2s`.

### Post-reset no-slot status pass while trainer SSH blocked — 2026-05-16 00:48 UTC

- Current public score state is unchanged after UTC reset: best `0.946`, latest `v560=0.945`, `v558/v551/v549/v548=0.946`. No new Kaggle submission was made because the v560 result strengthened the stop rule against public946 micro-sidecars, and no new OOF/source-clean artifact exists this pass.
- `v510` and `v560` kernels remain COMPLETE with no failure messages. `scripts/birdclef_kernel_output_verify.py --all-presets --pretty` returned top-level `ok=true` for `v510-real-sed` and `v560-direct-v2s`.
- Trainer SSH remains blocked: guarded NFNet launcher timed out during banner exchange and exited `75`, so no remote GPU job was launched. No active pending-submission monitor process needed restart.
- Decision: keep waiting for trainer SSH recovery before NFNet/student OOF work; do not use the fresh daily slot unless a materially stronger out-of-sample/OOF signal or genuinely new source appears.

### NFNet blended-teacher smoke recovered, scaled, and killed for submission — 2026-05-16 02:10 UTC

- Trainer SSH recovered enough for the guarded launcher to pass preflight. First launch exposed a launcher quoting bug (`$!` expanded locally under `set -u`) before remote work started. Fixed `scripts/launch_nfnet_pseudolabel_smoke_if_trainer_ready.sh` by building a remote command string with escaped remote PID capture, then validated `bash -n` and relaunched successfully.
- Ran the original 3-epoch NFNet smoke on trainer: `configs/birdclef/pl_public946_sed85_rankblend15_nfnet_5s_lr1e4_smoke_20260515.json`, log `logs/pl_public946_sed85_rankblend15_nfnet_5s_lr1e4_smoke_20260516T015252Z.log`, output `artifacts/pseudolabels/students/pl-public946-sed85-rankblend15-nfnet-5s-lr1e4-smoke-20260515/`. Result: final all-row student AUC `0.768877` vs teacher `0.995304`, corr `0.622429`, runtime `16.8s`; too weak to scale directly, but validation AUC was still climbing.
- Added and ran focused ep8 smoke `configs/birdclef/pl_public946_sed85_rankblend15_nfnet_5s_lr1e4_ep8_smoke_20260516.json`, log `logs/pl_public946_sed85_rankblend15_nfnet_5s_lr1e4_ep8_smoke_20260516T0158Z.log`, output `artifacts/pseudolabels/students/pl-public946-sed85-rankblend15-nfnet-5s-lr1e4-ep8-smoke-20260516/`. Result: final all-row AUC `0.933012`, teacher `0.995304`, corr `0.799621`, runtime `9.0s`. This passed the minimal scale gate.
- Added and ran full 792-row ep20 diagnostic `configs/birdclef/pl_public946_sed85_rankblend15_nfnet_5s_lr1e4_ep20_20260516.json`, log `logs/pl_public946_sed85_rankblend15_nfnet_5s_lr1e4_ep20_20260516T0200Z.log`, output `artifacts/pseudolabels/students/pl-public946-sed85-rankblend15-nfnet-5s-lr1e4-ep20-20260516/`. Result: final all-row AUC `0.988538` over 75 classes vs teacher `0.997018`, corr `0.956990`, MAE `0.020259`, runtime `43.4s`, TorchScript `89.872 MB`.
- Blend audit artifact on trainer: `artifacts/pseudolabels/audits/public946_sed85_rankblend15_nfnet_ep20_audit_20260516T0205Z.json`. Best teacher+NFNet blend is student weight `0.075`, macro AUC `0.997055577`, lift `+0.000037123` vs teacher, blended corr `0.999763`. This is a real but tiny local lift, smaller than v560's failed local gate, so do **not** package or submit this NFNet sidecar without stronger OOF/new-source evidence.

### B3 blended-teacher model-zoo diagnostic killed by blend gate — 2026-05-16 03:15 UTC

- Current public state unchanged: best `0.946`, latest `v560=0.945`, `v558/v551/v549/v548=0.946`; no Kaggle slot used and no queue/monitor restart needed. `v510`/`v560` verifier presets remain `ok=true`; trainer healthy again with both 4090s idle.
- RegNetY against this teacher had already been killed earlier (`0.891280` smoke AUC), so the next uncovered model-zoo diagnostic was EfficientNet-B3 against `teacher_sed85_rankblend15.npz`.
- Added and ran B3 ep3 smoke `configs/birdclef/pl_public946_sed85_rankblend15_b3_5s_pretrained_lr1e4_smoke_20260516.json`, log `logs/pl_public946_sed85_rankblend15_b3_5s_pretrained_lr1e4_smoke_20260516T0255Z.log`, output `artifacts/pseudolabels/students/pl-public946-sed85-rankblend15-b3-5s-pretrained-lr1e4-smoke-20260516/`. Result: final all-row AUC `0.767158` over 42 classes vs teacher `0.995304`, corr `0.191475`, runtime `7.3s`. Validation AUC was still climbing, so ran ep8.
- Added and ran B3 ep8 smoke `configs/birdclef/pl_public946_sed85_rankblend15_b3_5s_pretrained_lr1e4_ep8_smoke_20260516.json`, log `logs/pl_public946_sed85_rankblend15_b3_5s_pretrained_lr1e4_ep8_smoke_20260516T0300Z.log`, output `artifacts/pseudolabels/students/pl-public946-sed85-rankblend15-b3-5s-pretrained-lr1e4-ep8-smoke-20260516/`. Result: final AUC `0.923309` vs teacher `0.995304`, corr `0.494312`, runtime `8.2s`. Low correlation justified one full diagnostic.
- Added and ran full 792-row B3 ep20 diagnostic `configs/birdclef/pl_public946_sed85_rankblend15_b3_5s_pretrained_lr1e4_ep20_20260516.json`, log `logs/pl_public946_sed85_rankblend15_b3_5s_pretrained_lr1e4_ep20_20260516T0305Z.log`, output `artifacts/pseudolabels/students/pl-public946-sed85-rankblend15-b3-5s-pretrained-lr1e4-ep20-20260516/`. Result: final all-row AUC `0.976570` over 75 classes vs teacher `0.997018`, corr `0.936009`, MAE `0.022659`, runtime `33.9s`, TorchScript `41.995 MB`.
- Blend audit artifact on trainer: `artifacts/pseudolabels/audits/public946_sed85_rankblend15_b3_ep20_audit_20260516T0310Z.json`. Best tested teacher+B3 blend was still negative: student weight `0.01`, macro AUC `0.997014564`, lift `-0.000003890` vs teacher, corr `0.999994`. Decision: kill B3 packaging/submission for this target; it adds no local blend lift.

### Bootstrap stability gate for public946 sidecars — 2026-05-16 04:05 UTC

- Current public state unchanged: best `0.946`, latest `v560=0.945`, `v558/v551/v549/v548=0.946`; no Kaggle slot used and no queue/monitor restart needed. `v510`/`v560` verifier presets remain `ok=true`; trainer healthy with both 4090s idle.
- Implemented a stronger validation-proxy extension in `scripts/birdclef_public946_multi_sidecar_weight_grid.py`: optional grouped bootstrap lift stability via `--bootstrap-iters`, `--bootstrap-group {file,site,row}`, and `--bootstrap-seed`. It reports mean/median lift, quantiles, and `p_lift_gt_0` for each candidate vs the public946 base instead of trusting one fragile mean AUC.
- Validation: `py_compile` passed, synthetic smoke passed, and the real v559 V2S+B0 strict dry-run gate was rerun with 200 file-group bootstrap iterations. Artifact: `artifacts/blend_grids/v559_v2s_b0_multi_sidecar_gate_bootstrap_20260516T0400Z.json`.
- Real v559 bootstrap result: best mean row remains `V2S=0.005 + B0=0.010`, macro AUC `0.992560140`, lift `+0.000035240` vs base `0.992524901`, but bootstrap stability is weak: `p_lift_gt_0=0.84`, median lift `+0.000034959`, and 5th percentile lift `-0.000071763`. This supports the post-v560 rule: tiny positive local sidecar lifts are not slot-worthy unless their grouped bootstrap lower tail is positive and the evidence is materially stronger.

### v560 bootstrap backtest shows local gates still insufficient — 2026-05-16 05:10 UTC

- Current public state unchanged: best `0.946`, latest `v560=0.945`, `v558/v551/v549/v548=0.946`; no Kaggle slot used and no queue/monitor restart needed. `v510`/`v560` verifier presets remain `ok=true`; trainer healthy, though GPU1 was busy so no new trainer job was launched.
- Backtested the new grouped-bootstrap gate on the known failed `v560` direct-V2S sidecar. Artifacts: `artifacts/blend_grids/v560_direct_v2s_gate_bootstrap_20260516T0500Z.json` (file-group) and `artifacts/blend_grids/v560_direct_v2s_gate_site_bootstrap_20260516T0505Z.json` (site-group).
- Result: the submitted `direct_v2s=0.03` candidate still looked robust locally: aggregate lift `+0.000081879` vs v542, file-bootstrap `p_lift_gt_0=0.995` with q05 `+0.000017475`, and site-bootstrap `p_lift_gt_0=1.0` with q05 `+0.000023012`. Yet the real public score was `0.945`.
- Interpretation: grouped bootstrap is useful for rejecting fragile cases like v559, but it is **not sufficient** for 0.95 decisions. Future slot candidates must clear a higher bar than v560: materially larger local lift than `+0.000081879`, genuinely new source/OOF evidence, and preferably validation outside the train-soundscape overlap. Do not rely on bootstrap-positive micro-sidecars alone.

### Leave-one-group gate backtest confirms train-soundscape optimism — 2026-05-16 06:05 UTC

- Current public state unchanged: best `0.946`, latest `v560=0.945`, `v558/v551/v549/v548=0.946`; no Kaggle slot used and no queue/monitor restart needed. `v510`/`v560` verifier presets remain `ok=true`; trainer healthy but this pass stayed CPU-local.
- Extended `scripts/birdclef_public946_multi_sidecar_weight_grid.py` again with optional leave-one-group-out diagnostics via `--leave-one-group {file,site,row}` and `--holdout-detail`. This reports per-held-out-group lift stability in addition to bootstrap quantiles.
- Validation: `py_compile` passed and a synthetic smoke covered both bootstrap and leave-one-file-out output.
- Backtested v560 and v559 with file bootstrap plus leave-one-site-out. Artifacts: `artifacts/blend_grids/v560_direct_v2s_gate_holdout_20260516T0600Z.json` and `artifacts/blend_grids/v559_v2s_b0_multi_sidecar_gate_holdout_20260516T0600Z.json`.
- Result: leave-one-site-out still did **not** catch the failed v560. The submitted `direct_v2s=0.03` had aggregate lift `+0.000081879`, bootstrap q05 `+0.000014389`, and leave-one-site min lift `+0.000034296` across 6 sites, yet public LB was `0.945`. v559 remained weaker under bootstrap, but some site holdouts stayed positive.
- Interpretation: grouped bootstrap/holdout are useful rejection filters but not sufficient approval filters. The train-soundscape overlap is too optimistic for 0.95 slot decisions; future submissions need independent OOF/new-source evidence or a much larger lift than v560, not just local-group stability.

### Leave-one-file gate also misses failed v560 — 2026-05-16 06:15 UTC

- Added file-level holdout backtest artifacts: `artifacts/blend_grids/v560_direct_v2s_gate_leave_one_file_20260516T0610Z.json` and `artifacts/blend_grids/v559_v2s_b0_gate_leave_one_file_20260516T0610Z.json`.
- Result: the failed v560 `direct_v2s=0.03` remained positive under leave-one-file-out too: 20 valid held-out files, min lift `+0.000061769`, q05 lift `+0.000063445`, and `p_lift_gt_0=1.0`, despite real public LB `0.945`.
- v559 best row also remained file-holdout positive (`min_lift=+0.000010589`, `p_lift_gt_0=1.0`) even though bootstrap had already shown a negative lower tail. Therefore file/site holdout over the labeled train-soundscape overlap cannot be used as an approval gate either.
- Updated decision rule: local train-soundscape gates may reject clearly fragile candidates, but cannot approve public946 sidecars. Next 0.95 slot candidate should come from independent OOF/new-source validation or a qualitatively different model/source, not further low-weight public946 sidecar tuning.

### Spec B/D 10s ConvNeXt public946 student smoke and full run — 2026-05-16 07:15 UTC

- Current public state unchanged: best `0.946`, latest `v560=0.945`, `v558/v551/v549/v548=0.946`; no Kaggle slot used. `v510`/`v560` verifier presets remain `ok=true`; trainer healthy/idle at start.
- Chosen track: pivot away from public946 micro-sidecars to Spec B/D pseudo-label/model-zoo diversity with changed temporal context. Rationale: train-soundscape gates cannot approve sidecars; test whether 10s context can create a stronger independent ConvNeXt student signal than the prior 5s ConvNeXt.
- Added configs: `configs/birdclef/pl_public946_sed85_rankblend15_v2s_xc_extinit_10s_m160_lr1e4_smoke_20260516.json`, `configs/birdclef/pl_public946_sed85_rankblend15_convnext_tiny_10s_m160_lr3e4_smoke_20260516.json`, and `configs/birdclef/pl_public946_sed85_rankblend15_convnext_tiny_10s_m160_lr3e4_ep20_20260516.json`.
- Smoke 1, V2S external-XC init 10s/160mel: launched durable trainer job pid `22436`, log `logs/pl_public946_sed85_rankblend15_v2s_xc_extinit_10s_m160_lr1e4_smoke_20260516T0653Z.log`; result best val AUC `0.851115`, final-all AUC `0.840604`, teacher corr `0.420416`, runtime `7.8s`, TorchScript `88.74 MB`. Decision: too weak, do not scale.
- Smoke 2, ConvNeXt-Tiny 10s/160mel: launched pid `24144`, log `logs/pl_public946_sed85_rankblend15_convnext_tiny_10s_m160_lr3e4_smoke_20260516T0654Z.log`; result best val AUC `0.877010`, final-all AUC `0.876685`, teacher corr `0.640900`, runtime `12.0s`, TorchScript `112.36 MB`. This beat the earlier 5s ConvNeXt ep3 smoke, so scaled.
- Full ConvNeXt-Tiny 10s/160mel ep20: launched pid `25126`, log `logs/pl_public946_sed85_rankblend15_convnext_tiny_10s_m160_lr3e4_ep20_20260516T0656Z.log`; result final-all AUC `0.993316844` over 75 classes, teacher AUC `0.997018454`, teacher corr `0.972329`, runtime `72.1s`, TorchScript `112.36 MB`. Audit artifact `artifacts/pseudolabels/audits/public946_sed85_rankblend15_convnext10_ep20_audit_20260516T0710Z.json`.
- Blend audit: best teacher+ConvNeXt10 blend was weight `0.01`, macro AUC `0.997041778`, lift `+0.000023323`, corr vs teacher `0.9999973`. This is far below v560's failed local lift and too correlated; do not package or submit. Interpretation: 10s ConvNeXt improves standalone student AUC but collapses toward teacher, so it is not a 0.95 slot candidate.

### Fresh public-source sweep after v560/ConvNeXt rejection — 2026-05-16 07:45 UTC

- Reran a broad Kaggle public notebook sweep for `birdclef 2026 0.947/0.948`, BirdNET, Snowflake, CLAP, score-desc, EoS, and ensemble terms. Artifact: `artifacts/public_sweeps/kaggle_public_list_20260516T0725Z.json` with 77 unique refs; detailed get-kernel cache: `artifacts/public_sweeps/kaggle_public_getkernel_20260516T0728Z.json` plus per-ref JSON under `artifacts/public_sweeps/getkernel_20260516T0728Z/`.
- Downloaded outputs/logs for the promising May-16 cluster under `artifacts/public_sweeps/outputs_20260516T0730Z/` and triaged with `artifacts/public_sweeps/output_triage_20260516T0735Z.json`. No Kaggle submission slot was used.
- Main new cluster: public notebooks around BirdNET/EoS4/score-desc/site-hour and Franksunp 5-branch CLAP/Snowflake/site-hour variants. Several are only public test dry-run outputs with 3 rows, so output correlation is a weak screen only.
- BirdNET-active public forks (`kruzzcc/bc26-safar-0947-birdnet`, `kruzzcc/bc26-nina-eos4-fixed`, `kruzzcc/bc26-raunak-deep-dyn-bn`, `claudedevore/birdclef-2026-r0946-clap-sidecar-v1`) run the resolved BirdNET model and map about `157 direct + 4 genus-proxy` labels, with final sample output only lightly displaced from `mtoshidesu` (`MAE≈0.0032`, `corr≈0.978`). This is source-clean but not new enough by itself: our `v543/v544` BirdNET 10%/5% already tied `0.946` and did not improve.
- `kruzzcc/bc26-yaroslav-sitehour-bn` is the most interesting BirdNET-family variant: it combines BirdNET with a Yaroslav/site-hour-style prior and logs coverage `64/234` classes. Sample displacement remains small (`MAE≈0.0038`, `corr≈0.960`). Worth source-reading if we need one new public-source candidate, but not auto-submit-worthy.
- `lucataco/bc26-scoredesc-conservative-ensemble` is a clean public-public rank ensemble over EoS/Raunak/Mtoshi/Safar refs; sample output has bounded displacement vs mtoshi (`MAE≈0.0194`, `corr≈0.948`). The older non-conservative version had a much larger global shift (`MAE≈0.138`) and is likely too blunt. Conservative score-desc may be worth porting as a no-submit local/gate candidate, but it is still a public-output ensemble rather than independent signal.
- Franksunp 5-branch CLAP/Snowflake/site-hour notebooks attach public CLAP and Snowflake datasets, but their logs show BirdNET unavailable/fallback in these runs; CLAP/Snowflake/site-hour shift the 3-row sample heavily (`MAE≈0.13-0.21`, sometimes low/negative correlation). Given our source-clean CLAP/Snowflake attempts already dropped/tied and these branches are poorly anchored on the sample, do not submit directly. At most mine the site-hour prior implementation separately.
- `xiyuetong/birdclef2026-ensemble-v3-topn-pseudo-clap` uses pseudo-cache + CLAP and writes `submission_clap_onnx.csv`; sample displacement is small (`MAE≈0.0031`, `corr≈0.945`). Since v545/v551 CLAP already failed/tied, this is not enough for a slot without source-level evidence that the top-N pseudo logic is materially different and non-fallback.
- Decision: no immediate Kaggle push/submission from the sweep. Best follow-up, if we continue public-source mining, is source-read/port a **non-submitting candidate** for (1) conservative score-desc rank ensemble or (2) Yaroslav/site-hour + BirdNET gating, then compare against v542/v558/v560 evidence. Avoid another BirdNET/CLAP/Snowflake global sidecar unless it has independent OOF/new-source validation or materially stronger evidence than failed v560.

### 2025 top-team recipe port prep — 2026-05-16 18:05 UTC

- Status check unchanged: current best remains `0.946`. Latest scored rows are `v563=0.946`, `v565=0.943`, `v564=0.942`, `v562=0.945`, and `v561` invalid format. Daily slots remain used; no new submission was possible.
- Read current spec and article evidence. The strongest outside evidence points away from more tiny random-init sidecars and toward 2025-style SED/noisy-student recipes: eca_nfnet_l0 / EfficientNetV2-S, Focal+BCE, sqrt balancing, power-scaled pseudo-label rounds, external/pretrained data, and taxon-specialist handling.
- Added new active prep doc `docs/BIRDCLEF_2025_RECIPE_PORT_SPEC_20260516.md` and linked it from `docs/BIRDCLEF_NEW_DIRECTIONS_SPECS.md`.
- Extended `scripts/birdclef_pseudolabel_student_train.py` to support `loss_name="focal_bce"`, `focal_gamma`, `focal_loss_weight`, and `class_weight_mode="sqrt_inv_prevalence"` / `"inv_prevalence"` with clipping and mean-normalized class weights.
- Prepared NFNetL0 and EfficientNetV2-S focal/BCE sqrt-class-weight smoke configs for the next free GPU window.
- Validation: tensor unit check for focal+BCE + sqrt class weights produced finite loss/backprop; `py_compile` and `git diff --check` passed. Remote timm check confirmed `eca_nfnet_l0` and `tf_efficientnetv2_s` are available in `~/kaggle_envs/s6e3`.

### Promising candidate slate spec — 2026-05-16

- Added a ranked candidate slate to `docs/BIRDCLEF_2025_RECIPE_PORT_SPEC_20260516.md` based on the article scan and recent failed student-sidecar lessons.
- Candidates:
  1. NFNetL0 focal/BCE noisy student from public946 teacher.
  2. EfficientNetV2-S focal/BCE noisy student from public946 teacher.
  3. External/pretrained 2025-style CNN init followed by public946 distillation.
  4. Non-bird / rare-taxon specialist correction model.
  5. Real SED/MIL frame-local student rather than clip-only sidecar.
  6. Quantile-Mix / rank+mean ensemble refresh only after a genuinely new prediction artifact exists.
- Each candidate has hypothesis, recipe, smoke gate, scale/submission bar, and kill rule. Recommended execution order prioritizes NFNetL0/EffV2-S when GPU is free, then external/pretraining and taxon specialist work rather than further tiny random-init retreads.

### PR #231 opened + candidate smoke tests — 2026-05-16 19:15 UTC

- Created clean PR branch `feature/2025-recipe-candidates-pr` from fresh `origin/main`; cherry-picked only the intended recipe/trainer/spec commits rather than PR'ing the long experiment branch. Opened PR #231: https://github.com/yourslewis/birdclef-2026/pull/231. GitHub reports mergeable but blocked by normal branch policy/review.
- Candidate A NFNetL0 focal/BCE sqrt-class-weight smoke launched on trainer GPU0, pid `132168`, log `logs/pl_public946_sed85_rankblend15_nfnetl0_focalbce_sqrtcw_5s_m160_lr1e4_ep8_smoke_20260516T1852Z.log`. Result: final all-row AUC `0.885977541`, teacher AUC `0.995303584`, corr `0.707264239`, runtime `9.7s`, TorchScript `89.872 MB`. Decision: fail smoke; do not scale/submit.
- Candidate B EfficientNetV2-S focal/BCE sqrt-class-weight smoke launched on trainer GPU0, pid `132996`, log `logs/pl_public946_sed85_rankblend15_effv2s_focalbce_sqrtcw_5s_m160_lr3e4_ep8_smoke_20260516T1900Z.log`. Result: final all-row AUC `0.714379835`, teacher AUC `0.995303584`, corr `0.245783760`, runtime `10.5s`, TorchScript `81.451 MB`. Decision: fail smoke; very low corr is failure-to-learn.
- Candidate C existing external-init B0 sanity recheck launched pid `134986`, log `logs/pl_public946_sed85_rankblend15_b0_extinit_5s_smoke_recheck_20260516T1905Z.log`, config `pl_public946_sed85_rankblend15_b0_5s_smoke_20260515.json`. Result: AUC `0.901693090`, teacher AUC `0.995303584`, corr `0.561511425`, runtime `4.9s`, TS `15.391 MB`. Not submission-ready; val trajectory was still rising, so a better external/pretraining full diagnostic may still be worthwhile.
- Conclusion: PR is useful as infrastructure/spec, but none of the tested candidates are ready for Kaggle submission. Next submission should use already verified repo-owned public-kernel candidates after reset or a future artifact that clears full-row blend gates.

### Candidate C/D follow-up diagnostics — 2026-05-16 19:20 UTC

- Ran aligned blend audit for existing full-row external-init B0 public946 student `pl-public946-sed85-rankblend15-b0-5s-ep20-20260515`: standalone AUC `0.992137465` over 75 valid classes vs teacher `0.997018454`, corr `0.963364380`, TS `15.391 MB`. Best blend was student weight `0.01`, AUC `0.997046430`, lift `+0.000027976`, corr vs teacher `0.999996273`. Decision: not slot-worthy; below failed-v560 local-lift bar.
- Ran taxon diagnostics: submission labels are Amphibia 35 / Aves 162 / Insecta 28 / Mammalia 8 / Reptilia 1. Train audio rows are heavily Aves-skewed (34,799 Aves vs 451 Amphibia / 199 Insecta / 99 Mammalia / 1 Reptilia), while train soundscape rows are mostly multi-label (`1322/1478`) and often non-bird/mixed. This supports Candidate D as a multi-output rare/non-bird specialist, not a softmax-only taxon classifier.
- Ran public946 taxon-gate sweep on labeled cache rows. Baseline AUC `0.997018454`; best was `mode=max`, `floor=0.30`, `alpha=0.25`, AUC `0.997043408`, lift `+0.000024954`. Stronger queued-style gates generally drop. Decision: no standalone taxon-gate submission; need learned/source-backed specialist or bounded correction with stronger evidence.

### Rare/non-bird specialist crossfit diagnostic — 2026-05-16 20:55 UTC

- Status check: current public best remains `0.946`; latest batch unchanged (`v563=0.946`, `v565=0.943`, `v564=0.942`, `v562=0.945`, `v561` invalid). Daily slots remain used; no Kaggle submission attempted. No active BirdCLEF processes; GPUs were occupied by unrelated work, so chose CPU-safe diagnostics.
- Chosen track: Candidate D rare/non-bird specialist. Implemented `scripts/birdclef_rare_taxon_specialist_diagnostics.py`, a diagnostic that crossfits per-taxon logistic presence models from teacher-cache row features using grouped file folds, then sweeps bounded multiplicative corrections for Amphibia/Insecta/Mammalia/Reptilia columns. This avoids in-sample site/hour leakage and tests whether learned group presence is stronger than the base group evidence.
- Ran on trainer against `artifacts/pseudolabels/public946-v540-teacher-cache66-v1/teacher_sed85_rankblend15.npz` with truth from `pl-public946-sed85-rankblend15-b0-5s-ep20-20260515/student_predictions.npz`. Output: `artifacts/pseudolabels/audits/public946_rare_taxon_specialist_diag_20260516T2055Z.json`.
- Baseline public946 teacher AUC: `0.997018454` over 75 valid classes; target rare/non-bird macro AUC: `0.998161252` over 47 valid classes.
- Crossfit group presence AUCs: Amphibia `0.998194` vs evidence `0.998538`; Insecta `0.999981` vs evidence `1.000000`; Mammalia `0.992206` vs evidence `0.995973`; Reptilia `0.987756` vs evidence `0.990520`. The learned specialist did not beat simple max-evidence group detection.
- Best bounded correction: alpha `0.5`, min mult `0.8`, max mult `1.25`, macro AUC `0.997032292`, lift `+0.000013838`, target lift `+0.000022082`, MAE `0.00405`, max abs `0.15984`, corr `0.99521`.
- Decision: not submission-ready and below failed-v560 local evidence. Candidate D remains plausible only with a genuinely new source/specialist model, not a learned calibrator over the same public946 predictions.

### Candidate E SED/MIL external-init B0 smoke — 2026-05-16 22:05 UTC

- Status check: current public best remains `0.946`. Latest scored submissions unchanged: `v563=0.946`, `v565=0.943`, `v564=0.942`, `v562=0.945`, `v561` invalid format. UTC daily slots are still fully used, so no Kaggle submission was attempted.
- Repo/process state: started from clean `feature/rare-taxon-specialist-diagnostics`, then created clean follow-up branch `feature/sed-mil-smoke-v566-prep` from updated `origin/main`. No active BirdCLEF queue/trainer jobs were running; GPU server was free.
- Chosen track: Candidate E real SED/MIL frame-local student. Rationale: A/B focal noisy-student smokes failed, Candidate D same-prediction specialist failed, and direct public/source sidecars have tied or dropped. This tests whether the existing frame-head/MIL training path is operational enough to become the next medium-term architecture lane.
- Smoke launched on `trainer` GPU0 with `configs/birdclef/sed_b0_q3cap80_ep12init_oof_10s_160_100cls_paired_smoke.json`; log `logs/sed_b0_q3cap80_ep12init_oof_10s_160_100cls_paired_smoke_20260516T215814Z.log`; output dir `artifacts/sed_oof/sed-b0-q3cap80-ep12init-oof-10s-160-100cls-paired-smoke/`.
- Result: completed in `15.686s` on 300 balanced files (`n_train=240`, `n_val=60`, `n_classes=234`). External checkpoint loaded encoder weights (`352` keys, head skipped). Validation loss improved each epoch: `0.1500 -> 0.0938 -> 0.0696`. Holdout macro AUC was only `0.810206872` over 27 valid classes. TorchScript export succeeded at `15.389 MB`.
- Decision: operational smoke **passes** (decode/train/export works), but modeling smoke **fails** for submission/scale. Do not spend a slot or scale this exact 3-epoch B0 SED/MIL recipe. Candidate E remains alive only as a medium-term lane requiring stronger frame/local targets, longer/full diagnostics, or a different teacher target before packaging.

### Next-reset public sweep submitter restored/preflighted — 2026-05-16 23:05 UTC

- Status check: current public best remains `0.946`. Latest scored submissions unchanged: `v563=0.946`, `v565=0.943`, `v564=0.942`, `v562=0.945`, and `v561` invalid format. There are already five `2026-05-16` UTC submissions, so no new submission was attempted.
- Chosen track: B/C next-reset public-source datapoints. Since all same-teacher student/taxon/SED diagnostics are below the failed-v560 evidence bar and slots are capped, prepared the controlled post-reset submitter rather than launching another weak sidecar.
- Restored `scripts/submit_public_sweep_candidates_when_slots_available.py` from the prior experiment branch onto a fresh `origin/main` branch. The script is dry-run by default, duplicate-guards by exact description, verifies source kernels are `COMPLETE` and expose `submission.csv`, excludes known invalid output-only Lucataco score-desc, and queues at most three next-reset exploratory public kernels so two slots remain for follow-up.
- Candidate order for reset remains: `v566` Nina EoS4 fixed + BirdNET, `v567` Mtoshi UMAP + BirdNET, `v568` Meenal improved BirdNET, `v569` safe ensemble, `v570` Mtoshi improved. Use only after UTC reset and prefer `--max-submissions 1` or `2` if manual review wants extra slot conservation.
- Validation: `py_compile` passed, script `--help` passed, dry-run `--max-submissions 3` passed and would submit `v566`/`v567`/`v568` only, and `git diff --check` passed. An initial dry-run exposed that later public refs can 404 on `ListKernelFiles`; the script now stops before extra preflight once the requested dry-run/submit limit is reached and treats per-candidate preflight failures as skips. Did not run `--submit`; no Kaggle slot used.

### v566/v567 submitted after UTC reset — 2026-05-17 00:05 UTC

- Waited for the 2026-05-17 UTC reset, then used the guarded submitter instead of spending all five slots. Submitted exactly two candidates with `scripts/submit_public_sweep_candidates_when_slots_available.py --submit --max-submissions 2`; log `logs/submit_public_sweep_after_reset_20260516T235256Z.log`.
- Submitted `v566`: `kruzzcc/bc26-nina-eos4-fixed` version 2, description `v566: Sweep Nina EoS4 fixed plus BirdNET public kernel direct`, Kaggle ref `52723318`. Preflight saw COMPLETE plus `submission.csv`, `submission_birdnet.csv`, `submission_protossm.csv`, `submission_sed.csv`, `subm_3.csv`, `subm_4.csv`, and cache files.
- Submitted `v567`: `kruzzcc/bc26-mtoshi-umap-bn-a` version 1, description `v567: Sweep Mtoshi UMAP plus BirdNET public kernel direct`, Kaggle ref `52723321`. Preflight saw COMPLETE plus `submission.csv`, `submission_birdnet.csv`, `submission_protossm.csv`, `submission_sed.csv`, and cache files.
- Post-submit Bearer API check showed both `v566` and `v567` as `pending` with no errors. Preserved three UTC-day slots for follow-up. Current scored best remains `0.946` until these score.

### v567 scored 0.944; hold remaining public-sweep slots pending v566 — 2026-05-17 00:48 UTC

- Status check after reset submissions: `v567` completed at `0.944`, below the `0.946` plateau; `v566` remains `pending` with no error. Current scored best remains `0.946` from v541/v542/v558/v563 and earlier tied public946 variants.
- Lesson: the Mtoshi UMAP + BirdNET public-kernel direct path is not enough and is worse than the repo-owned public946 anchor. This joins v562 (`0.945`) as another drop in the direct BirdNET-family public-source lane.
- Decision: do **not** submit `v568`/`v569`/`v570` while `v566` is still pending. Preserve the remaining three 2026-05-17 UTC slots until `v566` scores. If `v566` also drops, kill the remaining direct public BirdNET-family queue and pivot back to repo-owned/new-signal work. If `v566` ties/improves, consider one targeted follow-up (`v568`) rather than spending all remaining slots.

### v566 tied 0.946; v568 submitted as single targeted follow-up — 2026-05-17 01:55 UTC

- Status check: `v566` completed at `0.946`, tying the plateau; `v567` completed at `0.944`, confirming the broader direct BirdNET-family queue is mixed/risky. Current scored best remains `0.946`.
- Because `v566` tied, used exactly one preserved slot for a targeted follow-up and continued preserving the other two slots. Ran `scripts/submit_public_sweep_candidates_when_slots_available.py --submit --max-submissions 1`; log `logs/submit_public_sweep_v568_followup_20260517T015255Z.log`.
- Submitted `v568`: `meenalsinha/birdclef-2026-improved` version 9, description `v568: Sweep Meenal improved BirdNET public kernel direct`, Kaggle ref `52725667`. Preflight saw COMPLETE plus `submission.csv`, `submission_birdnet.csv`, `submission_protossm.csv`, and `submission_sed.csv`.
- Post-submit Bearer API check showed `v568` as `pending` with no error. Decision: do not submit `v569`/`v570` unless `v568` ties/improves or new evidence appears; two slots remain preserved.

### v568 hidden rerun failed; v570 pending; submitter made target-safe — 2026-05-17 02:55 UTC

- Status check: current scored best remains `0.946`. `v566` tied `0.946`, `v567` scored `0.944`, `v568` completed with no score and Kaggle error `Your notebook hit an unhandled error while rerunning your code`, and `v570` is now pending with no error. Four 2026-05-17 UTC submissions are visible, so likely one slot remains.
- Diagnostics: `get_submission` for `v568` confirms totalBytes `0`, status COMPLETE, no public score, and only the generic hidden-rerun unhandled-error message. This is another reason to avoid further direct public-kernel submissions unless we repackage/own the kernel and can preflight hidden-format/runtime behavior.
- A heartbeat follow-up attempted one-slot `v569`, but `v569` failed preflight because `ListKernelFiles` returned 404. Because the submitter previously skipped preflight failures, it continued and submitted lower-priority `v570` (`mtoshidesu/lb-improved`, ref `52726497`). This consumed an exploratory slot unintentionally; wait for `v570` score but treat it as low-confidence because prior triage showed it was nearly identical to the Mtoshi baseline on sample outputs.
- Fixed `scripts/submit_public_sweep_candidates_when_slots_available.py` to support targeted `--labels` and to stop conservatively on preflight failure by default. Continuing to lower-priority candidates now requires explicit `--skip-preflight-failures`. Validation: `py_compile`, `--help`, targeted dry-run `--labels v569 --max-submissions 1` (stops at v569 preflight 404 without fallback), and `git diff --check` passed.
- Decision: make no further Kaggle submissions now. Preserve the last slot until `v570` scores or a repo-owned/new-signal candidate appears. Direct public BirdNET-family results so far are tie/drop/error/pending (`v566=0.946`, `v567=0.944`, `v568` hidden error, `v570` pending), so the lane should be considered exhausted unless `v570` unexpectedly improves.

### v570 RAM failure; direct public-kernel lane stopped — 2026-05-17 03:55 UTC

- Status check: current scored best remains `0.946`. `v570` completed with no score and Kaggle error `Your notebook requested more memory (RAM) than is available.` `v568` remains no-score from hidden-rerun unhandled error; `v567=0.944`; `v566=0.946` tie. Four 2026-05-17 UTC submissions are visible, so likely one slot remains.
- Added public-source follow-up stop-rule doc: `docs/BIRDCLEF_PUBLIC_SOURCE_FOLLOWUP_DIAGNOSTICS_20260517.md`. Summary: direct public-kernel lane produced one tie (`v566`), one drop (`v567`), two no-score failures (`v568` hidden error, `v570` RAM), and one preflight 404 (`v569`).
- Decision: stop direct public-kernel submissions for this lane. Preserve the last daily slot unless a repo-owned/repackaged hidden-rerun-safe candidate or genuinely new-source candidate appears. Future public-source work should port/repackage the idea, not direct-submit public notebooks with uncertain hidden behavior.

### Repo-owned repackage plan for next public-source candidate — 2026-05-17 04:55 UTC

- Status check: current scored best remains `0.946`; v570 remains complete/no-score from RAM error, v568 no-score hidden error, v567 `0.944`, v566 `0.946`. Four 2026-05-17 UTC submissions are visible, so likely one slot remains. No new submission made.
- GPUs remain busy with non-BirdCLEF work, so this pass stayed CPU/doc/prep-only and preserved the last slot.
- Added `docs/BIRDCLEF_REPACKAGE_NEXT_CANDIDATE_PLAN_20260517.md`. It identifies the only remaining public-source idea worth further work as a repo-owned `v571` safe xSED / stacker-inspired rank blend based on the `pilkwang` safe-ensemble idea, not a direct public submission.
- Plan: start from hidden-safe `kaggle-kernels/v542-afr1ste-updated-public946/script.py`, keep input discovery/schema/verifier/ProtoSSM/SED branches, change only the final rank-blend formula toward source-backed xSED-style weights, avoid BirdNET and output-only public predictions, and require runtime/memory/schema validation before spending the last slot.

### v571 repo-owned safe xSED repackage submitted — 2026-05-17 06:10 UTC

- Status check before submission: current scored best remained `0.946`; `v570` complete/no-score RAM error; `v568` complete/no-score hidden-rerun error; `v567=0.944`; `v566=0.946`. Four 2026-05-17 UTC submissions were visible, leaving likely one slot.
- Implemented repo-owned `kaggle-kernels/v571-public946-safe-xsed-rankblend/` by copying hidden-safe `v542` and changing only the final ProtoSSM/SED rank blend from `0.60/0.40` to source-backed xSED-inspired `0.5964/0.4036`. No BirdNET branch, output-only public prediction, GPU, internet, or new data source was added.
- Added a hard verifier before writing `submission.csv`: exact `sample_submission.csv` columns, unique/nonempty rows, finite `[0,1]` values, dry-run row count equals ProtoSSM rows, hidden row count equals `len(test_soundscapes) * 12`, and formal mode must not collapse to sample-output shape.
- Validation: local `python3 -m py_compile` passed; metadata parse/static checks passed; diff against v542 was limited to metadata slug/title plus final blend/verifier; Kaggle private dry-run version 1 completed. Downloaded dry-run outputs from Kaggle: `submission.csv` rows `240`, cols `235`, finite values, min `0.0037500006`, max `1.0`; `submission_sed.csv` and `submission_protossm.csv` also had `240 x 235` finite outputs.
- Submitted v571 using the last likely 2026-05-17 slot: description `v571: Public946 safe xSED-inspired Proto/SED rank blend 0.5964/0.4036`, ref `52731029`. Immediate Bearer API check showed status `pending`, no score/error yet.
- Next step: monitor v571. If it ties/drops, treat xSED-weight-only as no better than v542 and pivot to real new-signal/noisy-student/model-zoo work. If it improves, explore source-backed xSED/stacker variants in repo-owned kernels.

### v571 tied; student-pool audit refreshed — 2026-05-17 06:58 UTC

- Status check: `v571` completed and scored `0.946` (ref `52731029`, `totalBytes=17835535`), tying but not beating the current public946 plateau. `v570` remains complete/no-score RAM error; `v568` remains complete/no-score hidden-rerun error; `v567=0.944`; `v566=0.946`. The 2026-05-17 UTC submission slots appear fully used.
- Lesson: the source-backed xSED ratio-only repackage (`0.5964 Proto / 0.4036 SED`) is hidden-safe and valid, but not an improvement over v542's `0.60/0.40`. Do not spend another slot on microscopic Proto/SED rank-weight changes unless a genuinely different source/stacker signal is added.
- With slots capped, refreshed the public946 student-pool blend audit on the GPU server: `/home/yourslewis/birdclef-2026/artifacts/pseudolabels/audits/public946_sed85_rankblend15_student_pool_audit_20260517T0655Z.json`. It scanned `108` student artifacts, found `40` row/label-aligned, and used teacher macro AUC `0.997018454` over `75` valid train-soundscape classes.
- Top local blend from the audit was old `pl-r2-v2s-v508-soft-p100-5s-pretrained-lr1e4-ep20-bestval`: standalone AUC `0.983987`, student/teacher corr `0.3752`, best weight `0.05`, local lift `+0.000168656`. This is interesting for offline study, but not slot-worthy by itself because a V2S sidecar already produced `v560=0.945` despite positive local gates.
- Checked the prepared 2025-style focal/BCE smoke candidates. `nfnetl0_focalbce_sqrtcw` ep8 completed but underperformed (`best_val_auc=0.898793`, final student/teacher corr `0.7073`), worse than the prior non-focal NFNet smoke (`0.940256`) and far below stronger ep20 students. `effv2s_focalbce_sqrtcw` ep8 also failed (`best_val_auc=0.707770`, corr `0.2458`). Demote these exact focal/sqrt smoke recipes; continue only with substantially changed initialization/pretraining or robust blend evidence.
- Next step: wait for 2026-05-18 UTC slots, but prioritize real new-signal work (external/pretrained/noisy-student or robust multi-student packaging with cross-site stability) over further direct public kernels or one-parameter rank-weight tweaks.

### External manifest audit and q0 B0 smoke after v571 — 2026-05-17 08:00 UTC

- Status check: current public best remains `0.946`; latest visible submission `v571` is complete/scored `0.946`. `v570` and `v568` remain complete/no-score failures; `v567=0.944`; `v566=0.946`. All 2026-05-17 UTC slots appear used, so no Kaggle submission was attempted.
- Followed `docs/BIRDCLEF_NEXT_SIGNAL_AFTER_V571_20260517.md` and ran the external manifest audit on the GPU server. High-quality q>=3 cap80 manifest (`artifacts/external_pretrain/manifest_q3_cap80_20260517/`) has `2659` rows (`2470` train / `189` val) but is almost entirely birds: class rows `Aves=2605`, `Amphibia=38`, `Mammalia=16`; `Insecta=0`, `Reptilia=0`. It still has `72` species with <5 available rows and `54` missing target species after filter.
- Ran a broader q>=0 cap80 manifest audit (`artifacts/external_pretrain/manifest_q0_cap80_20260517/`) to test whether lower-quality/iNat rows improve coverage. It has `3388` rows (`3050` train / `338` val), classes `Aves=3138`, `Amphibia=182`, `Mammalia=49`, `Insecta=18`, `Reptilia=1`, and `28` missing target species. Coverage is better but includes `652` unrated/zero-quality rows and remains very sparse for non-bird taxa.
- Added config `configs/birdclef/xc_b0_q0_cap80_external_pretrain_smoke_20260517.json` and ran a B0 q0/cap80 external-pretrain smoke on GPU0 (`CUDA_VISIBLE_DEVICES=0`) using 512 manifest files, 2 epochs, 5s/128 mel, pretrained EfficientNet-B0, focal BCE, pos_weight_sqrt. Output: `artifacts/external_pretrain/xc-b0-q0-cap80-external-pretrain-smoke-20260517/`; `n_train=410`, `n_val=102`, TorchScript `15.389 MB`, runtime `18.2s`, holdout macro AUC only `0.482335` over `88` valid classes.
- Decision: broad q0/all-quality external pretraining is not a good next candidate as-is; the added low-quality/non-bird coverage hurts signal and is still too sparse for Reptilia/Insecta. Keep q3/bestloss external checkpoints as possible high-quality bird pretrain, but do not package or submit q0. Next external step should be either a targeted non-bird/rare manifest acquisition/audit or a robust distillation/stability gate using existing q3 bestloss init, not more blind q0 scaling.

### Student blend site-stability audit helper and results — 2026-05-17 09:00 UTC

- Status check: current public best remains `0.946`; `v571=0.946`; `v570`/`v568` no-score failures; `v567=0.944`; `v566=0.946`. No Kaggle submission attempted because 2026-05-17 UTC slots appear used.
- Extended `scripts/birdclef_student_pool_blend_audit.py` with optional best-blend stability gates: site/file/row bootstrap lift and leave-one-group lift. Added `--stability-top-n` so expensive stability is computed only after ranking the top candidates.
- Ran site-stability audit on the GPU server: `artifacts/pseudolabels/audits/public946_sed85_rankblend15_student_pool_site_stability_20260517T0855Z.json`. It scanned `108` students, found `40` aligned, and evaluated stability for the top `8` blends with `50` site-bootstrap iterations plus leave-one-site.
- Strongest local-stable candidate remains `pl-r2-v2s-v508-soft-p100-5s-pretrained-lr1e4-ep20-bestval` at student weight `0.05`: local lift `+0.000168656`, student/teacher corr `0.3752`, leave-one-site `p_lift_gt_0=1.0` with min lift `+0.000063181`, site bootstrap `p_lift_gt_0=0.94` but q05 slightly negative (`-0.000020597`). The next robust candidate was `pl-r1-convnext-tiny-v508-soft-p100-lr3e4-nomix-ep20-bestval` at `0.05`: lift `+0.000100392`, leave-one-site `p=1.0`, min `+0.000039548`, bootstrap q05 `+0.000007196`.
- Interpretation: the helper is useful, but these are not immediate slot candidates because both V2S/ConvNeXt families have already under-transferred on public LB (`v560=0.945`, `v564=0.942`, `v565=0.943`). Next packaging must combine stability with a source/hidden-safety difference, not just train-soundscape robustness.

### Reproducible rare/non-bird source audit script — 2026-05-17 10:00 UTC

- Status check: current public best remains `0.946`; `v571=0.946`; `v570`/`v568` no-score failures; `v567=0.944`; `v566=0.946`. No Kaggle submission attempted because 2026-05-17 UTC slots appear used.
- Added `scripts/birdclef_rare_nonbird_source_audit.py`, a deterministic data-only audit that separates source rows from locally verified audio, summarizes per-target species/taxon q>=3/q>=4 coverage, assigns non-bird candidate statuses, and writes an Amphibia/Mammalia q>=3 verified manifest if available.
- Ran the script on trainer with output `artifacts/external_pretrain/rare_nonbird_source_audit_20260517T0955Z/`. Summary: `234` target species, `72` non-bird species, `28` target species with no source rows, `49` with no q>=3 rows, `69/72` non-bird species with fewer than five q>=3 source rows and fewer than five q>=3 verified local audio rows.
- Non-bird status counts: `needs_external_discovery=28`, `source_sparse_or_low_quality=30`, `trainable_low_quality_only=11`, `trainable_verified_q3=3`. Amphibia/Mammalia q>=3 verified manifest has only `54` rows across `21` species, too small and skewed for a slot-bound specialist model by itself.
- Decision: do not train/package an Amphibia/Mammalia specialist from current local source alone. The next non-bird route requires external discovery/acquisition or a source-backed pseudo-label target; current data is enough for an audit/abstention plan, not a Kaggle candidate.

### Raw-SED 10s B0 student smoke and full diagnostic — 2026-05-17 11:05 UTC

- Status check: current public best remains `0.946`; `v571=0.946`; `v570`/`v568` no-score failures; `v567=0.944`; `v566=0.946`; no new Kaggle submission because slots appear used.
- Pivoted to the alternate v573-style real SED/MIL-ish lane: train a B0 SED-head student against raw public946 SED teacher output (`teacher_sed.npz`) with 10s context, rather than final rankblend targets or another rank-weight tweak.
- Added smoke config `configs/birdclef/pl_public946_sedraw_b0_10s_m160_lr3e4_ep8_smoke_20260517.json` and ran it on trainer GPU0. Result: best/last val AUC `0.943722` over `30` valid classes, final-all AUC `0.904111` over `42`, corr `0.8385`, runtime `6.274s`; trajectory improved but small-row final-all remained weak.
- Added full-row config `configs/birdclef/pl_public946_sedraw_b0_10s_m160_lr3e4_ep20_20260517.json` and ran it on trainer GPU0. Result: val AUC peaked around `0.993693` over `60` classes, final-all AUC `0.988647` over `75`, corr vs teacher `0.977247`, runtime `27.642s`, TorchScript `15.391 MB`.
- Blend/stability audit versus the public946 sed85/rankblend teacher (`artifacts/pseudolabels/audits/public946_sedraw_b0_10s_ep20_blend_audit_20260517T1102Z.json`) rejected it: best tested student weight was only `0.0025` and still had lift `-0.000003042`; leave-one-site p_lift_gt_0 `0.0` with all site-held-out lifts negative. Decision: do not package/submit raw-SED 10s B0; it is a good teacher mimic but not additive to public946.

### Refreshed q3 external B3 pretrain + public946 distill diagnostic — 2026-05-17 11:30 UTC

- Status check: current public best remains `0.946`. Latest scored candidates: `v571=0.946`, `v570` RAM no-score, `v568` hidden-rerun no-score, `v567=0.944`, `v566=0.946`, `v565=0.943`, `v563=0.946`. Five 2026-05-17 UTC submission slots are already used, so no Kaggle submission was made.
- Found a manifest hygiene issue while starting the B3 external-pretrain diagnostic: older `artifacts/external_pretrain/manifest_q3_cap80/external_pretrain_manifest.csv` has `976` balanced q>=3 candidate rows but only `295` resolve to local files on trainer. The refreshed `manifest_q3_cap80_20260517` resolves all `976` balanced rows. Treat old-manifest external-pretrain metrics as stale/path-limited unless rerun on the refreshed manifest.
- Ran B3 q>=3/cap80 external pretrain on the refreshed manifest with `configs/birdclef/xc_b3_q3_cap80_manifest20260517_external_pretrain_balanced_ep6_20260517.json`: `976` examples, val macro AUC `0.650746` over `117` valid classes, best epoch `6`, runtime `37.4s`, TorchScript `41.991 MB`.
- Scaled to `configs/birdclef/xc_b3_q3_cap80_manifest20260517_external_pretrain_balanced_ep18_20260517.json`: val macro AUC `0.722691` over `117` classes, best val loss at epoch `10`, runtime `48.109s`, TorchScript `41.991 MB`. This improves B3 substantially over the old 128-row smoke but still trails the existing B0 q3 ep18-bestloss external checkpoint (`0.747224` over `122`).
- Distilled public946 teacher from the B3 q3 refreshed external checkpoint with `configs/birdclef/pl_public946_sed85_rankblend15_b3_xc_q3_manifest20260517_extinit_5s_m128_lr1e4_ep20_20260517.json`: final-all student AUC `0.968505` over `75`, student/teacher corr `0.936244`, runtime `31.243s`, TorchScript `41.995 MB`.
- Blend/stability audit `artifacts/pseudolabels/audits/public946_b3_xc_q3_manifest20260517_extinit_blend_audit_20260517T1125Z.json`: best tested student weight `0.05`, local lift `+0.000045896`; site-bootstrap p_lift_gt_0 `0.80` with q05 `-0.000060676`; leave-one-site p_lift_gt_0 `0.8889`, worst site `S09` lift `-0.000011376`. Decision: no submission/package yet. The signal is mildly additive locally but not robust enough to spend a slot, especially after V2S/ConvNeXt sidecars under-transferred on LB.

### Refreshed q3 B0 external-pretrain apples-to-apples diagnostic — 2026-05-17 12:10 UTC

- Status check: public best remains `0.946`; `v571=0.946`, `v570` RAM no-score, `v568` hidden-rerun no-score, `v567=0.944`, `v566=0.946`, `v565=0.943`, `v563=0.946`. Five 2026-05-17 UTC slots remain used, so no Kaggle submission was made.
- Ran refreshed-manifest B0 q>=3/cap80 external pretrain with `configs/birdclef/xc_b0_q3_cap80_manifest20260517_external_pretrain_balanced_ep18_20260517.json`: `976` examples, val macro AUC `0.717722` over `117` valid classes, best val loss at epoch `13`, runtime `28.725s`, TorchScript `15.389 MB`. Same refreshed manifest / seed as B3 shows B0 is slightly behind B3 (`0.722691`) in this diagnostic, and both are below the older seed42 B0 metric (`0.747224`) on a non-identical split.
- Distilled public946 from the refreshed B0 checkpoint with `configs/birdclef/pl_public946_sed85_rankblend15_b0_xc_q3_manifest20260517_extinit_5s_m128_lr3e4_ep20_20260517.json`: final-all student AUC `0.992896` over `75`, student/teacher corr `0.970497`, runtime `14.896s`, TorchScript `15.391 MB`.
- Blend/stability audit `artifacts/pseudolabels/audits/public946_b0_xc_q3_manifest20260517_extinit_blend_audit_20260517T1208Z.json`: best tested weight `0.01`, local lift `+0.000028672`; leave-one-site p_lift_gt_0 `1.0` with min `+0.000001733`, but site-bootstrap p_lift_gt_0 only `0.78` with q05 `-0.000050795`. Decision: B0 refreshed-q3 extinit is safer than B3 on leave-one-site but the lift is very small and bootstrap remains fragile; do not submit while slots are capped. It is a possible low-weight exploratory fallback for next reset only if no stronger new-source candidate is ready.

### Local-window SED/MIL target diagnostic — 2026-05-17 13:15 UTC

- Status check: public best remains `0.946`; latest submissions unchanged with `v571=0.946`, `v570` RAM no-score, `v568` hidden no-score, `v567=0.944`, `v566=0.946`, `v565=0.943`, `v563=0.946`. Five 2026-05-17 UTC slots remain used, so no Kaggle submission was made.
- Implemented a lightweight frame/local target hook in `scripts/birdclef_pseudolabel_student_train.py`: `temporal_target_mode` (`center`, `local_max`, `local_mean`, `center_localmax_mix`), `temporal_neighbor_radius`, and `temporal_center_weight`. This preserves historical center-row behavior by default but allows a 10s context window to learn from neighboring 5s teacher rows as weak SED/MIL-style supervision.
- Ran smoke config `configs/birdclef/pl_public946_sed85_rankblend15_b0_centerlocalmax_r1_10s_m160_lr3e4_ep8_smoke_20260517.json`: B0, refreshed q3 B0 init, 10s/160mel, center/localmax mix radius 1, center weight 0.5, 256 rows, 8 epochs. Result: val AUC `0.953302` over `28`, final-all AUC `0.947316` over `42`, corr `0.871117`, runtime `7.431s`. This passed scale gate relative to the raw-SED 10s smoke (`0.904111` final-all over `42`).
- Scaled to full-row config `configs/birdclef/pl_public946_sed85_rankblend15_b0_centerlocalmax_r1_10s_m160_lr3e4_ep20_20260517.json`: final-all student AUC `0.990436` over `75`, corr `0.967644`, runtime `26.702s`, TorchScript `15.391 MB`.
- Blend/stability audit `artifacts/pseudolabels/audits/public946_centerlocalmax_r1_10s_b0_blend_audit_20260517T1310Z.json`: best tested weight `0.01`, local lift `+0.000020186`; site-bootstrap p_lift_gt_0 `0.66`, q05 `-0.000073194`; leave-one-site p_lift_gt_0 `0.7778`, worst sites `S22=-0.000016085` and `S09=-0.000004629`. Decision: local-window target support works and is worth keeping, but this exact center/localmax r1 B0 is not robust enough to package/submit.

### Local-window variant prep under GPU contention — 2026-05-17 15:05 UTC

- Status check: public best remains `0.946`; latest submissions unchanged (`v571=0.946`, `v570` RAM no-score, `v568` hidden no-score, `v567=0.944`, `v566=0.946`, `v565=0.943`, `v563=0.946`). No Kaggle submission; 2026-05-17 UTC slots remain used.
- Prepared two next local-window SED/MIL smoke configs based on the previous center/localmax-r1 result: `configs/birdclef/pl_public946_sed85_rankblend15_b0_centerlocalmax_r1_cw075_10s_m160_lr3e4_ep8_smoke_20260517.json` (weaker neighbor influence, center weight `0.75`) and `configs/birdclef/pl_public946_sed85_rankblend15_b0_localmean_r1_10s_m160_lr3e4_ep8_smoke_20260517.json` (local mean target, radius `1`). Both use B0, refreshed-q3 B0 init, 10s/160mel, 256 rows, 8 epochs.
- GPU check showed both GPUs occupied by unrelated LRM P30 work, so the smokes were attempted with `CUDA_VISIBLE_DEVICES=""`. CPU execution was too slow/heavy and was killed before completion to avoid contention. Partial cw075 log only reached epoch 4 and is not treated as evidence; localmean produced no usable metric. Decision: keep configs as ready-to-run, but do not evaluate or package these variants until a GPU is available.

### Local-window target transform audit while GPU monitor waits — 2026-05-17 15:55 UTC

- Status check: public best remains `0.946`; latest submissions unchanged (`v571=0.946`, `v570` RAM no-score, `v568` hidden no-score, `v567=0.944`, `v566=0.946`, `v565=0.943`, `v563=0.946`). No Kaggle submission; 2026-05-17 UTC slots remain used.
- GPU monitor `/tmp/run_birdclef_localwindow_when_gpu_free_20260517.sh` is still active and waiting for a GPU with <2GB used; both GPUs remain occupied by unrelated LRM P30 work. No active BirdCLEF training process.
- Ran a CPU-light target-distribution audit for the local-window transforms: `artifacts/pseudolabels/audits/local_window_target_transform_summary_20260517T1555Z.json`. Compared with center targets, cw0.50 increases mean target probability `0.08917 -> 0.09396` and `>=0.95` cells `280 -> 317`; cw0.75 is gentler (`mean=0.09156`, `>=0.95=296`, mean abs delta `0.00239`); local_mean preserves mean (`0.08918`) while smoothing confidence (`>=0.95=255`, row-top mean `0.79646`). This supports testing cw0.75 as a less aggressive neighbor variant and local_mean as a smoothing/control variant once GPU is free.

### Local-window cw0.75 GPU smoke/full scale and audit — 2026-05-17 22:00 UTC

- Status check: public best remains `0.946`. Kaggle Bearer API shows latest scored/complete submissions: `v571=0.946`, `v570` complete/no-score RAM error, `v568` complete/no-score hidden-rerun error, `v567=0.944`, `v566=0.946`, `v565=0.943`, `v564=0.942`, `v563=0.946`. The five 2026-05-17 UTC slots are already used, so no Kaggle submission was attempted.
- Repo hygiene: branch `feature/next-reset-public-sweep-submit` was clean before this run; no active BirdCLEF trainer process was running. The prior local-window GPU monitor had already completed and removed its marker.
- Collected the queued GPU smokes from trainer. `center_localmax_mix` with weaker neighbor influence (`temporal_center_weight=0.75`) completed at `logs/pl_centerlocalmax_r1_cw075_gpu_20260517T1500Z.log`: best val AUC `0.926609` over `29`, final-all AUC `0.948212` over `42`, corr `0.863944`, runtime `5.709s`. The `local_mean` control completed at `logs/pl_localmean_r1_gpu_20260517T1500Z.log`: best val AUC `0.951101` over `35`, but final-all AUC only `0.924528` over `42`, corr `0.809669`; decision: kill/local-mean control, do not scale it.
- Added and ran full-row config `configs/birdclef/pl_public946_sed85_rankblend15_b0_centerlocalmax_r1_cw075_10s_m160_lr3e4_ep20_20260517.json` on trainer GPU0. Output: `artifacts/pseudolabels/students/pl-public946-sed85-rankblend15-b0-centerlocalmax-r1-cw075-10s-m160-lr3e4-ep20-20260517/`; `792` rows, best epoch `15`, best val AUC `0.992308` over `61`, final-all student AUC `0.991336` over `75`, teacher corr `0.964665`, runtime `30.460s`, TorchScript `15.391 MB`.
- Blend/stability audit `artifacts/pseudolabels/audits/public946_centerlocalmax_r1_cw075_10s_b0_blend_audit_20260517T2158Z.json`: standalone AUC `0.991336`; best tested student weight `0.0025`, local lift `+0.000015339`; site-bootstrap p_lift_gt_0 `0.78`, q05 `-0.000028233`; leave-one-site p_lift_gt_0 `0.8889`, min lift `-0.000002217` on `S09`.
- Interpretation: cw0.75 is slightly more stable than the earlier cw0.50 center/localmax full run (`leave-one-site p=0.8889` vs `0.7778`) but its lift is smaller (`+0.0000153` vs `+0.0000202`) and best weight shrank to `0.0025`. Keep it as evidence that gentler local-window targets are less fragile, but it is not strong enough to spend a Kaggle slot unless no better candidate is ready after UTC reset.

### v572 cw0.75 local-window B0 package + guarded submit monitor — 2026-05-18 04:00 UTC

- Status check after UTC reset: current public best remains `0.946`; latest visible submissions are unchanged (`v571=0.946`, `v570` RAM no-score, `v568` hidden no-score, `v567=0.944`, `v566=0.946`, `v565=0.943`, `v564=0.942`, `v563=0.946`). No 2026-05-18 UTC competition submissions were visible, so slots are available.
- Track choice: despite the public946 micro-sidecar stop rule, Wenhao asked to keep producing controlled exploratory datapoints. Chose a repo-owned, verified-format **local-window/weak-SED-MIL** sidecar rather than a direct public notebook. This is distinct from the failed V2S/ConvNeXt family: B0, 10s/160mel, center/localmax r1 target with `temporal_center_weight=0.75`, and an audited tiny rank weight `0.0025`.
- Packaged private dataset `yourslewis/bc26-public946-cw075-localwindow-b0-v1` from trainer artifact `pl-public946-sed85-rankblend15-b0-centerlocalmax-r1-cw075-10s-m160-lr3e4-ep20-20260517/model_torchscript.pt`. Bundle zip SHA256 `cad40dd2b6731c46116ab9827c8ffd3cfa689e64775ec2fca31f59ad73cfdf12`, size about `13.4 MB`, containing `sed_bundle_manifest.json` and one TorchScript model (`15.391 MB`). Dataset creation returned `Ok` at `https://www.kaggle.com/datasets/yourslewis/bc26-public946-cw075-localwindow-b0-v1`.
- Added repo-owned kernel `kaggle-kernels/v572-public946-cw075-localwindow-b0-w00025/`, forked from the hidden-safe v560/v542 public946 sidecar path. It mounts the cw0.75 dataset, writes `submission_cw075_localwindow_b0_student.csv`, and applies `STUDENT_RANK_BLEND=0.0025` after the standard public946 gates. Metadata slug: `yourslewis/bc26-v572-public946-cw075-b0-w00025`.
- Validation before push: `python3 -m py_compile` on the kernel script and `python3 -m json.tool` on metadata passed. Kaggle kernel push returned version `1`, no invalid dataset/competition/kernel/model sources, URL `https://www.kaggle.com/code/yourslewis/bc26-v572-public946-cw075-b0-w00025`.
- Started guarded submit monitor `scripts/submit_v572_when_ready.py` with log `logs/submit_v572_when_ready_20260518T0355Z.log`. It exits if the v572 description already exists, waits for kernel `COMPLETE`, requires `submission.csv`, `submission_cw075_localwindow_b0_student.csv`, `submission_sed.csv`, `submission_protossm.csv`, and requires log markers for the cw0.75 sidecar before submitting. Initial status: kernel `RUNNING`; no competition submission made yet.

### v572 verified and submitted — 2026-05-18 04:08 UTC

- v572 kernel version `1` completed successfully. Read-only verifier passed: output files included `submission.csv`, `submission_cw075_localwindow_b0_student.csv`, `submission_sed.csv`, `submission_protossm.csv`, and Perch cache files; required log markers for cw0.75 sidecar completion and final rank-sidecar blend were present.
- Submitted to BirdCLEF 2026 with description `v572: Public946 v542 plus cw0.75 local-window B0 rank sidecar 0.25%`. Submission ref `52762124`; immediate status `pending`, no score/error yet, `totalBytes=0` while pending.
- Guardrail note: the background monitor `logs/submit_v572_when_ready_20260518T0355Z.log` has a duplicate-description guard and should exit on its next wake now that the submission exists.

### v572 tied and cadence accelerated — 2026-05-18 08:00 UTC

- Status check: `v572` completed and scored `0.946`, tying the current public best. It was hidden-safe and competition-format valid (`totalBytes=17836470`) but did not improve beyond the public946 plateau. Latest context remains `v571=0.946`, `v570` RAM no-score, `v568` hidden no-score, `v567=0.944`, `v566=0.946`.
- Lesson: the cw0.75 local-window B0 sidecar at `0.25%` is safer than V2S/ConvNeXt drops but still not enough. Extend the micro-sidecar stop rule to this low-weight local-window B0 packaging path.
- Wenhao requested more urgency / more PRs. Restored the BirdCLEF autonomous cron from 6-hour cadence back to hourly and changed operating style toward multiple small, reviewable PRs for distinct hypotheses while preserving output-verification before Kaggle submissions.
### Frame-head SED pilot configs queued for faster PR cadence — 2026-05-18 08:05 UTC

- In response to Wenhao's urgency request, split the next true-new-signal work into its own small PR rather than bundling it with v572 docs.
- Added two real SED/frame-head EfficientNet-B0 pilot configs using refreshed q3 B0 external init, balanced class sampling, focal BCE gamma `1.5`, sqrt positive weights, label smoothing `0.005`, and restore-best-by-val-loss:
  - `configs/birdclef/sed_b0_framehead_10s_m160_q3init_ep4_20260518.json`
  - `configs/birdclef/sed_b0_framehead_20s_m160_q3init_ep4_20260518.json`
- Purpose: move away from public946 micro-sidecars toward actual frame/event SED signal; smoke/scale runs should report holdout AUC, TorchScript export size, and whether either model is low-correlation enough to become a future sidecar package.

### Frame-head 10s q3-init pilot result — 2026-05-18 08:10 UTC

- Launched the 10s/160mel B0 frame-head pilot on trainer GPU0: `configs/birdclef/sed_b0_framehead_10s_m160_q3init_ep4_20260518.json`.
- Result: completed on CUDA with `1024` examples (`819` train / `205` val), input shape `[1024,160,626]`, best epoch `4`, best val loss `0.082988`, holdout macro AUC `0.723326` over `94` valid classes, runtime `31.409s`, TorchScript size `15.389 MB`.
- Interpretation: this is a real frame-head SED signal and much healthier than the earlier q0 external-pretrain smoke, but still below the stronger q3 external-pretrain/OOF baselines. Continue by launching the 20s sibling and only consider packaging if a later OOF/blend audit shows low-correlation additive value.

### Frame-head 20s q3-init pilot result — 2026-05-18 08:15 UTC

- Launched the 20s/160mel B0 frame-head sibling on trainer GPU0: `configs/birdclef/sed_b0_framehead_20s_m160_q3init_ep4_20260518.json`.
- Result: completed on CUDA with `1024` examples (`819` train / `205` val), input shape `[1024,160,1251]`, best epoch `4`, best val loss `0.068910`, holdout macro AUC `0.806310` over `92` valid classes, runtime `36.574s`, TorchScript size `15.389 MB`.
- Interpretation: 20s context is clearly stronger than 10s in this pilot (`0.8063` vs `0.7233`). Next PR should scale the 20s frame-head recipe to more files/epochs or fold-aware OOF before any packaging decision.
### Frame-head 20s scale config queued — 2026-05-18 08:20 UTC

- Added scale config `configs/birdclef/sed_b0_framehead_20s_m160_q3init_ep8_2048_20260518.json` after the 20s/1024-file pilot beat the 10s pilot (`0.806310` vs `0.723326`).
- This scale check uses 20s/160mel, refreshed q3 B0 init, focal BCE gamma `1.5`, sqrt positive weights, label smoothing `0.005`, `2048` max files, `160` max classes, `8` epochs, and restore-best-by-val-loss.
- Purpose: test whether true frame-head SED signal scales before any package/submit work.

### Frame-head 20s scale result — 2026-05-18 09:40 UTC

- Collected `configs/birdclef/sed_b0_framehead_20s_m160_q3init_ep8_2048_20260518.json` from trainer GPU0.
- Result: completed on CUDA with `2005` examples (`1604` train / `401` val), input shape `[2005,160,1251]`, best epoch `7`, best val loss `0.055458`, holdout macro AUC `0.902068` over `144` valid classes, runtime `88.874s`, TorchScript size `15.389 MB`.
- Interpretation: this is the first strong frame-head SED scale signal in the current run and a clear improvement over the 1024-file 20s pilot (`0.806310`). Next action is to scale breadth/epochs again and then run a blend/correlation audit before packaging.
### Frame-head 20s ep12/4096 result and public946 audit — 2026-05-18 10:48 UTC

- Collected `configs/birdclef/sed_b0_framehead_20s_m160_q3init_ep12_4096_20260518.json` from trainer GPU1.
- Result: completed on CUDA with `3051` examples (`2441` train / `610` val), input shape `[3051,160,1251]`, best epoch `5`, best val loss `0.052794`, holdout macro AUC `0.922414` over `179` valid classes, runtime `189.555s`, TorchScript size `15.389 MB`.
- Built temporary single-model TorchScript bundle `artifacts/sed_bundles/framehead-20s-q3init-ep12-4096-v1` and ran teacher66 train-soundscape inference: `792` rows, `234` classes, `6.56s` total (`0.099s/file`).
- Blend audit versus public946 teacher cache (`artifacts/pseudolabels/audits/public946_framehead20s_ep12_blend_audit_20260518T1045Z.json` on trainer): standalone train-soundscape macro AUC `0.467988` over `75` valid classes, flat corr vs teacher `0.053539`, best tiny blend weight `0.0025` with lift `-0.000001105`. Site bootstrap mean lift `-0.00000391`, p(lift>0)=`0.275`; leave-one-site mean lift `-0.000001515`.
- Interpretation: random train-audio holdout keeps improving, but it does not transfer to labeled train soundscapes yet. Do **not** package/submit this supervised frame-head model directly. Pivot to soundscape/pseudo-label-adapted 20s training before spending Kaggle slots.
### v573 public946 cw0.75 20s B0 sidecar package + guarded monitor — 2026-05-18 10:47 UTC

- After the supervised frame-head 20s train-audio model failed public946 transfer, trained the soundscape/pseudo-label-adapted 20s sibling `configs/birdclef/pl_public946_sed85_rankblend15_b0_centerlocalmax_r1_cw075_20s_m160_lr3e4_ep20_20260518.json` on trainer GPU1.
- Training result: `792` rows (`634` train / `158` val), `234` classes, best epoch `16`, best val AUC `0.990481` over `61` valid classes, final-all student AUC `0.991183` over `75` valid classes vs teacher `0.996798`, corr vs teacher `0.965250`, MAE `0.017386`, runtime `64.705s`, TorchScript `15.391 MB`.
- Blend/stability audit `artifacts/pseudolabels/audits/public946_cw075_20s_b0_blend_audit_20260518T1055Z.json`: best student rank weight `0.015`, local AUC `0.997042086`, lift `+0.000023632` vs teacher, standalone AUC `0.991183`, corr `0.963024`. Site bootstrap p(lift>0)=`0.8033`, mean lift `+0.000027692`; leave-one-site p(lift>0)=`0.8889`, min lift `-0.000000172` on S09.
- Packaged private dataset `yourslewis/bc26-public946-cw075-20s-b0-v1` from trainer artifact `pl-public946-sed85-rankblend15-b0-centerlocalmax-r1-cw075-20s-m160-lr3e4-ep20-20260518/model_torchscript.pt`. Bundle zip SHA256 `3ab570390d1ee8cccdd154b83d66a70e6e68770488dd23d8ae94638e408fbf86`, size about `13.4 MB`, containing `sed_bundle_manifest.json` and one TorchScript model.
- Added repo-owned kernel `kaggle-kernels/v573-public946-cw075-20s-b0-w0015/`, forked from v572, mounting the new 20s sidecar dataset, writing `submission_cw075_20s_b0_student.csv`, and applying `STUDENT_RANK_BLEND=0.015` after standard public946 gates. Metadata slug: `yourslewis/bc26-v573-public946-cw075-20s-b0-w0015`.
- Validation before push: `python3 -m json.tool` on kernel metadata; `python3 -m py_compile` on kernel script, `scripts/submit_v573_when_ready.py`, and `scripts/push_v573.py`; `git_maint.py hygiene` clean. Bearer kernel push returned version `1`, URL `https://www.kaggle.com/code/yourslewis/bc26-v573-public946-cw075-20s-b0-w0015`, no invalid sources.
- Started guarded submit monitor `scripts/submit_v573_when_ready.py` with pid `10368`, log `logs/submit_v573_when_ready_20260518T1105Z.log`. Initial status: kernel `RUNNING`; no v573 competition submission made yet.

### v573 submitted + 20s power0.85 follow-up audit — 2026-05-18 11:40 UTC

- Guarded v573 monitor completed kernel verification and submitted `v573: Public946 v542 plus cw0.75 20s B0 rank sidecar 1.5%` to BirdCLEF. Submission ref `52773142`; status at recheck: `pending`, no score/error yet, `totalBytes=0` while pending.
- Verified v573 Kaggle output before submit: kernel COMPLETE/no failure; output files included `submission.csv`, `submission_cw075_20s_b0_student.csv`, `submission_sed.csv`, `submission_protossm.csv`; required log markers for 20s B0 sidecar completion and final rank-sidecar blend were present. Kernel log showed student inference on 20 public dry-run files in `45.0s`, output shape `(240,235)`, prob range `0.013601` to `0.983358`, mean `0.094878`.
- Prepared and ran follow-up config `configs/birdclef/pl_public946_sed85_rankblend15_b0_centerlocalmax_r1_cw075_20s_m160_lr3e4_ep20_power085_20260518.json`, which keeps the v573 20s/cw0.75 setup but changes `teacher_power=1.0 -> 0.85` and seed `95 -> 96`.
- Power0.85 result on trainer GPU1: `792` rows, best epoch `20`, best val AUC `0.993466` over `58` valid classes, final-all student AUC `0.991986` over `75`, teacher AUC `0.996798`, corr `0.957827`, MAE `0.043671`, runtime `64.201s`, TorchScript `15.391 MB`.
- Blend/stability audit `artifacts/pseudolabels/audits/public946_cw075_20s_b0_power085_blend_audit_20260518T1140Z.json`: best student rank weight `0.0075`, local lift `+0.000018286`, standalone AUC `0.991986`, corr `0.955097`. Site bootstrap p(lift>0)=`0.7067`, mean lift `+0.000019431`; leave-one-site p(lift>0)=`0.8889`, min lift `-0.000005326` on S09.
- Decision: power0.85 is a useful held diagnostic but **weaker than v573/power1.0** (`+0.000023632` lift, bootstrap p=0.8033, min site lift nearly zero). Do not package or submit power0.85 before v573 score lands.

### v573 scored 0.945 — cw-style B0 sidecar stop rule triggered — 2026-05-18 12:25 UTC

- Live Kaggle API check: `v573: Public946 v542 plus cw0.75 20s B0 rank sidecar 1.5%` completed with public score `0.945` (`ref=52773142`, bytes `17834452`, no error). Current public best remains `0.946` from v541/v542/v558/v566/v571/v572 family.
- Diagnosis: the stronger local 20s B0 sidecar audit (`+0.000023632` local lift, site bootstrap p(lift>0)=`0.8033`) still failed to transfer to public LB, matching the earlier v560 lesson that local train-soundscape sidecar gates are rejection filters, not approval filters.
- Stop rule: **do not spend more Kaggle slots on cw-style B0 sidecar variants**, including the queued power0.85 diagnostic. Treat power0.85 (`+0.000018286` local lift) as explicitly killed for public submission.
- Pivot: next work should be a genuinely distinct model/source signal. Chosen next lane is an NFNet-L0 public946 pseudo-label student with 20s context and center-only targets (no cw/local-window B0 sidecar), evaluated offline first.
### NFNet20 public946 pseudo-label result — 2026-05-18 12:38 UTC

- Collected `configs/birdclef/pl_public946_sed85_rankblend15_nfnet_20s_m160_lr1e4_ep20_center_20260518.json` from trainer GPU1. It is a distinct NFNet-L0, 20s/160mel, center-only soft target pivot after v573 dropped to `0.945`.
- Training result: `792` rows, best epoch `16`, best validation AUC `0.988233` over `59` valid classes, final-all student AUC `0.990618` over `75`, teacher AUC `0.997018`, corr `0.954731`, MAE `0.019269`, runtime `238.801s`, TorchScript `89.872 MB`.
- Blend/stability audit `artifacts/pseudolabels/audits/public946_nfnet20_center_blend_audit_20260518T1238Z.json`: best student rank weight `0.01`, local lift `+0.000007693`, standalone AUC `0.990618`, corr `0.954731`. Site bootstrap p(lift>0)=`0.9367`, mean lift `+0.000011651`; leave-one-site p(lift>0)=`1.0`, min lift `+0.000006406`.
- Decision: stable but too small/heavy for packaging after v573. Do **not** submit. Continue distinct-model/source search; next candidate should improve evidence materially, not just produce a tiny stable local lift.
### B3 XC-init 20s public946 pseudo-label result — 2026-05-18 12:43 UTC

- After v573 scored `0.945` and NFNet20 center produced only tiny stable lift, opened PR #243 and ran `configs/birdclef/pl_public946_sed85_rankblend15_b3_xc_q3_extinit_20s_m160_lr1e4_ep20_center_20260518.json` as a distinct external-pretrained EfficientNet-B3 20s/160mel center-only public946 pseudo-label candidate.
- Training result on trainer GPU1: `792` rows, best validation AUC `0.973045` over `59` valid classes at epoch `19`, final-all student AUC `0.972055` over `75`, teacher AUC `0.997018`, corr `0.932813`, MAE `0.021694`, runtime `111.318s`, TorchScript `41.995 MB`.
- Blend/stability audit `artifacts/pseudolabels/audits/public946_b3_xc_q3_20s_m160_blend_audit_20260518T1242Z.json`: best student rank weight `0.005`, local lift `+0.000017820`, standalone AUC `0.972055`, corr `0.932813`. Site bootstrap p(lift>0)=`0.7667`, mean lift `+0.000021348`; leave-one-site p(lift>0)=`1.0`, min lift `+0.000000145`.
- Decision: do **not** package/submit immediately. It is more stable than the older 5s B3 audit but lower-lift, and v573 showed a stronger local sidecar can still drop public LB. Continue searching for materially stronger distinct signals before spending another slot.

### Student-pool re-audit after NFNet20/B3XC20 — 2026-05-18 12:55 UTC

- Refreshed aligned public946 student-pool audit on trainer after adding the NFNet20 and B3 XC 20s artifacts: `artifacts/pseudolabels/audits/public946_sed85_rankblend15_student_pool_audit_20260518T1250Z.json`.
- Scan summary: `122` student prediction files scanned, `50` row/label-aligned against `teacher_sed85_rankblend15.npz`; teacher baseline remains `0.997018454` over `75` valid classes.
- Top local sidecar remains old V2S-v508 (`pl-r2-v2s-v508-soft-p100-5s-pretrained-lr1e4-ep20-bestval`): best weight `0.05`, local lift `+0.000168656`, site-bootstrap p(lift>0)=`0.9700`, leave-one-site p(lift>0)=`1.0`, min lift `+0.000063181`.
- This does **not** authorize another V2S slot because the already-submitted V2S/public946 sidecar family dropped (`v560=0.945`). Treat this refreshed pool audit as a ranking/rejection tool only.
- New NFNet20/B3XC20 artifacts did not displace the older top local candidates. Combined with v573, current decision remains: no more low-weight same-teacher sidecar submissions unless a candidate clears a much stronger out-of-family/offline bar or produces a genuinely new repo-owned inference artifact.
### Raw-SED 20s local-window diagnostic — 2026-05-18 13:50 UTC

- Status check before work: current public best remained `0.946`; latest visible submissions unchanged with `v573=0.945`, `v572=0.946`, `v571=0.946`, `v570` RAM no-score, `v568` hidden-rerun no-score, `v567=0.944`, `v566=0.946`, `v565=0.943`, `v563=0.946`. GPU1 idle; GPU0 occupied by unrelated LRM job. Repo hygiene clean.
- Opened PR #244: https://github.com/yourslewis/birdclef-2026/pull/244 (`Add BirdCLEF raw SED 20s local-window diagnostic`). Added `configs/birdclef/pl_public946_sedraw_b0_centerlocalmax_r1_cw075_20s_m160_lr3e4_ep20_20260518.json` and `docs/BIRDCLEF_SEDRAW20_LOCALWINDOW_20260518.md`.
- Hypothesis: train a compact B0 SED-head student on raw public946 SED teacher targets (`teacher_sed.npz`) with 20s context plus gentler center/localmax r1 cw0.75 target transform, instead of another final-rankblend/cw-style sidecar.
- Trainer result on GPU1: 792 rows, best val AUC `0.991514` over 61 at epoch 19, final-all student AUC `0.991099` over 75, raw SED teacher AUC `0.996475`, corr vs raw SED teacher `0.968311`, MAE `0.004448`, runtime `65.212s`, TorchScript `15.391 MB`.
- Audit vs sed85/rankblend public946 teacher `artifacts/pseudolabels/audits/public946_sedraw20_localwindow_blend_audit_20260518T1348Z.json`: standalone AUC `0.991099`, corr vs teacher `0.895538`, best tested weight `0.001`, lift `-0.000000961`, site-bootstrap p(lift>0)=`0.3267`, leave-one-site p(lift>0)=`0.1111`, min lift `-0.000012070`.
- Decision: kill this exact raw-SED 20s local-window candidate; it is compact and learns raw SED well but is not additive to the public946 sed85/rankblend teacher. No Kaggle submission.

### v574 guarded Nina EoS5 public-source replay submitted — 2026-05-18 14:50 UTC

- Status check: current public best remains `0.946`. Latest visible submissions: `v573=0.945`, `v572=0.946`, `v571=0.946`, `v570` RAM no-score, `v568` hidden-rerun no-score, `v567=0.944`, `v566=0.946`, `v565=0.943`, `v563=0.946`. 2026-05-18 UTC had two visible submissions before this run (`v572`, `v573`), leaving slots available.
- Track choice: public-source mining / high-upside guarded replay rather than another public946 sidecar. Pulled and inspected `nina2025/birdclef-2026-eos-5`, a newer EoS public stack with top-level direct blend weights `Model_2=0.0327`, `Model_5=0.9673`, and source comments around a `0.949` LB component. This is distinct from the exhausted cw/V2S/ConvNeXt sidecar lanes.
- Preflight: `nina2025/birdclef-2026-eos-5` status is `COMPLETE` with `failureMessage=null`; public output files include `submission.csv`, `submission_protossm.csv`, `submission_sed.csv`, `subm_5.csv`, and `subm_2.csv`. Public output files are dry-run-sized, so I did not treat them as proof by themselves.
- Hidden-path source guard: reviewed the pulled source under `artifacts/public_kernels_20260518/eos5/`. Required markers passed: hidden/test path checks `test_soundscapes/*.ogg`, `IS_DRY_RUN = len(test_paths) == 0`, sample alignment only under `if IS_DRY_RUN`, model branches for `Model_2` and `Model_5`, `Karnakbayev_PowerOptimization_LB0948`, and final `write_final_submission(..., "submission.csv")` verifier with unique `row_id` assertion. This makes the submission a guarded code-rerun datapoint, not a blind public-output CSV submission.
- Submission mechanics: initial latest-version attempt with `kernel_version=0` failed `403`; probe with `kernel_version=1` failed safely (`Did not find provided Notebook Output File`), revealing that the exact version was required. `ApiGetKernel` reported `currentVersionNumber=9`; submission with version `9` succeeded.
- Submitted `v574: Guarded direct Nina EoS5 public source replay after hidden-path source preflight`, Kaggle ref `52780102`. Immediate Bearer API check shows `pending`, no score/error yet.
- Caveat: despite source preflight, this is still a direct public notebook rerun rather than a fully repo-owned port. Do not queue additional direct public notebooks from the same family unless `v574` improves or yields a clear actionable lesson; if it ties/drops/fails, port only the reproducible structural idea into repo-owned code.

### EoS5 repo-owned port plan prepared while v574 pending — 2026-05-18 15:45 UTC

- Status check: `v574` remains pending (ref `52780102`); current public best remains `0.946`. Latest scored submissions remain `v573=0.945`, `v572=0.946`, `v571=0.946`, `v570` RAM no-score, `v568` hidden-rerun no-score, `v567=0.944`, `v566=0.946`, `v565=0.943`, `v563=0.946`. Three 2026-05-18 UTC slots are visible (`v572`, `v573`, `v574`), so likely two remain, but I preserved them while `v574` is pending.
- Repo/process state: branch `feature/birdclef-sedraw20-localwindow-20260518` was clean at start; no active BirdCLEF trainer or submit monitor. GPU0 is occupied by non-BirdCLEF work; GPU1 had low memory use but nonzero utilization, so I did not start another training job in this loop.
- Source inspection: summarized pulled EoS5 source (`295478` chars / `6533` lines, SHA256 `601ff2cb291cd26f007a64cbf01468cdb5ad3a8ebf232ab74c49c77c24714e8f`). Key markers are present for `Model_2`, `Model_5`, `Karnakbayev_PowerOptimization_LB0948`, hidden `test_soundscapes/*.ogg`, dry-run-only sample alignment, and final `write_final_submission` verifier.
- Prepared `docs/BIRDCLEF_EOS5_PORT_PLAN_20260518.md` with the post-score decision tree. If `v574` improves, port the EoS5 structural recipe into a repo-owned kernel rather than submitting siblings directly. If it ties/drops/fails, stop direct EoS-family submissions and only continue via minimal repo-owned structural deltas or a new model/source signal.

### v574 scored 0.949; v575 repo-owned EoS5 confirmation pushed — 2026-05-18 15:55 UTC

- Live Kaggle API check confirmed `v574: Guarded direct Nina EoS5 public source replay after hidden-path source preflight` completed at **0.949** (`ref=52780102`). This raises the current public best from `0.946` to **0.949**.
- Latest submission context before v575: `v574=0.949`, `v573=0.945`, `v572=0.946`, `v571=0.946`, `v570` RAM no-score, `v568` hidden-rerun no-score, `v567=0.944`, `v566=0.946`, `v565=0.943`, `v563=0.946`. Three 2026-05-18 UTC slots were visible, leaving likely two slots.
- Followed the EoS5 port-plan trigger: prepared repo-owned confirmation kernel `kaggle-kernels/v575-eos5-repo-confirm/`, using the pulled EoS5 notebook source as a private repo-owned Kaggle notebook rather than another direct public-kernel submission. The copied notebook has 19 cells and zero stored outputs; metadata is private, CPU, no internet, and keeps the same addable competition/dataset/kernel/model sources used by the successful public EoS5 path.
- Added guarded push/submit scripts `scripts/push_v575.py` and `scripts/submit_v575_when_ready.py`. Validation before push: kernel metadata parses as JSON, notebook JSON parses and has no outputs, and both scripts pass `py_compile`.
- Pushed Kaggle kernel with Bearer API v1. Kaggle returned version `1`, kernel id `119729759`, URL `https://www.kaggle.com/code/yourslewis/bc26-v575-repo-owned-eos5-confirmation`, with no invalid data/competition/kernel/model sources (only tag strings were rejected as tags).
- Started guarded submit monitor pid `95949`, log `logs/submit_v575_when_ready_20260518T1545Z.log`. Initial status: repo-owned v575 kernel `RUNNING`, no failure message. The monitor will submit only after COMPLETE status plus required output files/log markers.

### v575 repo-owned EoS5 confirmation submitted — 2026-05-18 16:35 UTC

- The repo-owned v575 Kaggle kernel completed successfully: `bc26-v575-repo-owned-eos5-confirmation` version `1`, COMPLETE/no failure. Output verification passed with required files: `submission.csv`, `subm_2.csv`, `subm_5.csv`, `subm_karnakbayev_power_optimization.csv`, `submission_protossm.csv`, and `submission_sed.csv`; required log markers were present.
- Submitted `v575: Repo-owned EoS5 confirmation of v574 public949 path`, Kaggle ref `52783235`. Immediate Bearer API check shows `pending`, no score/error yet. Current public best remains **0.949** from `v574` while v575 is pending.
- Validation caveat/fix: Kaggle nbconvert emitted a notebook-schema warning because the copied repo notebook had empty `execution_count`/`outputs` keys on markdown cells after output stripping. The kernel still ran and produced valid outputs, but I fixed the repo notebook to remove code-only fields from markdown cells for future pushes.

### EoS5 ablation queue prepared while v575 pending — 2026-05-18 16:45 UTC

- Status check: current best is **0.949** from `v574`; `v575` repo-owned confirmation remains pending (ref `52783235`). Latest visible submissions: `v575` pending, `v574=0.949`, `v573=0.945`, `v572=0.946`, `v571=0.946`, `v570` RAM no-score, `v568` hidden-rerun no-score, `v567=0.944`, `v566=0.946`. Four 2026-05-18 UTC submissions are visible, so likely one slot remains.
- Repo/process state: branch clean at start, no active BirdCLEF monitor/trainer processes; GPU1 is free and GPU0 is occupied by unrelated LRM work.
- Chosen action: preserve the last likely slot while `v575` is pending and prepare the next repo-owned EoS5 ablation queue. Added `docs/BIRDCLEF_EOS5_ABLATION_QUEUE_20260518.md`.
- Source inspection confirms EoS5's active `Model_5` path uses `lambda_prior=0.5`, `file_confidence_scale(top_k=2,power=0.4)`, `rank_aware_scaling(power=0.6)`, `adaptive_delta_smooth(base_alpha=0.20)`, and internal `xSED=[0.60,0.40]`. The top-level blend is `Model_2=0.0327`, `Model_5=0.9673`.
- Next candidate order if `v575` confirms: first `v576` Model5-only (remove weak `Model_2` complement), then one-scalar candidates such as rank-aware power `0.55` or `lambda_prior=0.55`. Do not run these if `v575` fails or diverges from `v574`.

### v576 Model5-only repo-owned ablation pushed while v575 pending — 2026-05-18 17:45 UTC

- Status check: current best remains **0.949** from `v574`; `v575` repo-owned EoS5 confirmation remains pending (ref `52783235`). Four 2026-05-18 UTC submissions are visible (`v572`-`v575`), so likely one slot remains.
- Decision: preserve the last likely competition slot until `v575` confirms, but prepare and run the next repo-owned kernel now. This avoids idle time without violating the v575 confirmation gate.
- Implemented `v576` as the first planned EoS5 ablation: Model5-only / remove the weak `Model_2` complement. It copies the repo-owned v575 notebook and changes only the top-level `solutions` block to `Model_5` weight `1.0`; added a one-model fallback in the final direct combiner so the notebook can emit `submission.csv` from `subm_5.csv` without executing `Model_2`.
- Added `kaggle-kernels/v576-eos5-model5-only/`, `scripts/push_v576.py`, and guarded `scripts/submit_v576_when_ready.py`. Validation: metadata JSON parses, notebook JSON parses with no stored outputs, notebook contains `Model_5` weight `1.0` and no `Model_2` in the top-level solutions cell, and scripts pass `py_compile`.
- Pushed Kaggle kernel with Bearer API v1. Kaggle returned version `1`, kernel id `119735856`, URL `https://www.kaggle.com/code/yourslewis/bc26-v576-eos5-model5-only-ablation`, with no invalid data/competition/kernel/model sources (only tag strings rejected).
- Started guarded submit monitor pid `9580`, log `logs/submit_v576_when_ready_20260518T1745Z.log`. Initial status: v576 kernel `RUNNING`, no failure message. The monitor will verify outputs and then submit only if `v575` is complete with `0.949+`; otherwise it preserves the slot.

### v575/v576 confirmed 0.949; v577 rank-power 0.55 pushed for next slot/reset — 2026-05-18 18:45 UTC

- Live Kaggle API check: `v575` repo-owned EoS5 confirmation completed at `0.949`, confirming the repo-owned path. The guarded `v576` Model5-only monitor then submitted and `v576` also completed at `0.949`. Current public best remains **0.949** from `v574`/`v575`/`v576`.
- 2026-05-18 UTC now has five visible submissions (`v572`-`v576`), so the daily competition slots are full. No further competition submission was made manually this loop.
- Lesson: removing the weak `Model_2` top-level complement did not hurt (`v576=0.949`), so the lift is indeed carried by the `Model_5` EoS5/PowerOptimization path. Future EoS tuning can use Model5-only as a cleaner base.
- Prepared the next repo-owned ablation `v577`: copy of v576/Model5-only with exactly one active scalar change, `rank_aware_scaling(... power=0.6) -> power=0.55`, testing whether the EoS5 rank-aware scaling bump is over-aggressive.
- Added `kaggle-kernels/v577-eos5-model5-rankp055/`, `scripts/push_v577.py`, and guarded `scripts/submit_v577_when_ready.py`. Validation: metadata JSON parses, notebook JSON parses/no stored outputs, top-level remains Model5-only, exactly one active `power=0.55` line is present, and scripts pass `py_compile`.
- Pushed Kaggle kernel with Bearer API v1. Kaggle returned version `1`, kernel id `119739708`, URL `https://www.kaggle.com/code/yourslewis/bc26-v577-eos5-model5-rank-power-0-55`, with no invalid data/competition/kernel/model sources (only tag strings rejected).
- Started guarded submit monitor pid `18325`, log `logs/submit_v577_when_ready_20260518T1845Z.log`. Initial status: v577 kernel `RUNNING`, no failure message. The monitor requires v577 output verification and `v576=0.949+`; if the daily cap is hit, it will sleep until the next slot window.

### 0.96 frontier source work: SafeAlign/Pilkwang triage, Chaney v37 queued — 2026-05-18 20:55 UTC

- Live Kaggle: current best remains `0.949` from `v574`/`v575`/`v576`; 2026-05-18 UTC visible count is `5`, so submissions are capped. No stale v577/v578 scalar monitor was alive.
- PR #245 was already merged, so new work moved to fresh branch `feature/birdclef-096-frontier-chaney-v37-20260518` from updated `origin/main`.
- SafeAlign/S106 diff result: `itshyao/birdclef-2026-s106-eos5-0949-safealign2` / `beicicc/bc26-s106-eos5-sa2-may18` are EoS5-like with `Model_2=0.04`, `Model_5=0.96` plus a robust final blend/row-id guard. Not slot-worthy for 0.960 because `v576` Model5-only already tied `0.949`.
- Pilkwang result: `pilkwang/birdclef-26-acoustic-time-window-rank-fusion` is a clean older `Karnakbayev_PowerOptimization_LB0948` single branch using EoS4-style `lambda_prior=0.4` and rank-aware power `0.5`; EoS5 already improved this to `0.5/0.6`. Skip direct slot.
- Selected higher-upside frontier candidate: `chaneyma/bc26-gate-v37-ninastyle-branch` v1. Rationale: structurally distinct Nina-style gate/branch stack, source comments/logs cite OOF/CV around `0.967`, COMPLETE/no failure, competition-format outputs. It is more 0.96-relevant than v577 scalar tuning.
- Added guarded submitter `scripts/submit_v580_chaney_v37_when_slot.py`. Preflight passed: Kaggle source pull v1, required source markers, dry-run fallback guard, kernel COMPLETE, output files include `submission.csv`, `v37_ninastyle_branch_shared_blend_summary.json`, `submission_imaad0946.csv`, `submission_sed.csv`.
- Started monitor pid `43469`, log `logs/submit_v580_chaney_v37_when_slot_20260518T2055Z.log`. It attempted submit as `v580: Guarded direct Chaney v37 Nina-style gate frontier replay`, hit daily cap (`3.3 hours from now`), and is sleeping `12000s` before retry.
- Caveat: Kaggle metadata includes blank datasetDataSources for some Chaney artifact inputs; repo-owned confirmation may require identifying/attaching the underlying artifact datasets or reproducing the branches if v580 scores high.

### A2Prime/NFNet fallback prepared behind v580 result — 2026-05-18 21:45 UTC

- Rechecked live status: best remains `0.949`; 2026-05-18 UTC still has 5 visible submissions; `v580` monitor pid `43469` remains alive and sleeping after cap, no submission visible yet.
- Scanned A2Prime/NFNet/EffV2S source cluster. `claudedevore/birdclef-2026-r0946-a2prime-nfnet-submit` is ERROR and lacks `submission.csv`; skip. `claudedevore/birdclef-2026-r0946-a2prime-effv2s-submit` is COMPLETE, but default `submission.csv` is `base_3way`, not EffV2S; skip for direct replay unless alternate output-file submission is intentionally chosen.
- Selected `lucataco/bc26-claude-a2prime-nfnet-fix` v2 as distinct fallback: COMPLETE/no failure; hidden default is `a2_nfnet_w03`; outputs include `submission.csv`, `submission_a2_nfnet_w03.csv`, `a2nfnet_blend_summary.csv`, `nfnet_branch_summary.csv`, `nfnet_sanity_file_summary.csv`, `submission_nfnet.csv`, and `submission_base_3way.csv`.
- Added `scripts/submit_v581_a2prime_nfnet_when_ready.py` and started guarded monitor pid `68275`, log `logs/submit_v581_a2prime_nfnet_when_ready_20260518T2145Z.log`. It waits for v580 to complete first; if v580 improves above `0.949`, it exits so the next step can be repo-owned v580 confirmation; if v580 ties/drops/no-scores, it preflights and submits v581.

### v580/v581 monitor restart and Chaney dependency map — 2026-05-18 22:50 UTC

- Rechecked live Kaggle: best remains `0.949`; UTC day still capped with five visible submissions; no v580/v581 submission visible yet.
- Found stale monitor PIDs: previous nohup pids `43469` (v580) and `68275` (v581) were no longer alive without additional log errors. Restarted as OpenClaw-managed background sessions to keep the reset queue live.
- v580 managed session `tender-ridge` / pid `88792`: source/output preflight re-passed, submit attempt hit daily cap with `78 minutes from now`, sleeping `4800s` before retry.
- v581 managed session `brisk-kelp` / pid `88794`: alive, waiting for v580 visibility/result before fallback action.
- Mapped Chaney v37 likely repo-owned confirmation dependencies from source paths: `chaneyma/birdclef2026-edits-protossm-sed-onnx-infer-artifacts`, `chaneyma/bc26-gate-fake008-head0015-baseline-onnx`, `chaneyma/bc26-edits-protossm-sed-v7-all66-40x20`, `chaneyma/bc26-edits-protossm-sed-v8-all66-synth-p010-40x20`, `chaneyma/bc26-probe-middle-pca128-raw085-logreg015`, plus common Perch/SED sources. If v580 improves, first follow-up is repo-owned confirmation with explicit source attachments.

### Reset queue healthy; v580 repo-owned dependency blocker found — 2026-05-18 23:30 UTC

- Live Kaggle unchanged: best `0.949`; 2026-05-18 UTC visible submissions `5`, 2026-05-19 visible submissions `0` at check time.
- Managed monitors are alive: v580 `tender-ridge` pid `88792` sleeping after cap and ready to retry; v581 `brisk-kelp` pid `88794` waiting for v580 visibility/result.
- Checked Chaney artifact dataset attachability for possible repo-owned v580 confirmation. All key Chaney artifact datasets returned `403 datasets.get denied`: `birdclef2026-edits-protossm-sed-onnx-infer-artifacts`, `bc26-gate-fake008-head0015-baseline-onnx`, `bc26-edits-protossm-sed-v7-all66-40x20`, `bc26-edits-protossm-sed-v8-all66-synth-p010-40x20`, `bc26-probe-middle-pca128-raw085-logreg015`.
- Public/common dependencies are attachable (`lixin73`, `jaejohn`, `rishikeshjani`, `tuckerarrants`). If v580 improves, simple repo-owned replay may be blocked; follow-up likely requires reproducing Chaney artifacts or extracting portable logic. Do not promise immediate repo-owned confirmation until artifact access is solved.
- Checked other attachability: Lucataco A2Prime/NFNet uses public `brendancarlin/birdclef2026-models`; Kamongi uses public `konbu17/bird26-train-audio-head-v1` but appears closer to `0.944`/idea-mining. No new submitter added; preserve v580 → v581 queue.

### v580 submitted after UTC reset — 2026-05-19 00:05 UTC

- Heartbeat check: managed v580 session `tender-ridge` woke after cap, re-ran source/output preflight, and successfully submitted `v580: Guarded direct Chaney v37 Nina-style gate frontier replay`, ref `52790976`.
- Live Kaggle: v580 is visible with status `pending`; current best remains `0.949` until it scores. 2026-05-19 UTC visible count is `1`.
- Managed v581 session `brisk-kelp` remains alive and is waiting for v580 to complete before fallback action. If v580 improves, stop v581 and solve Chaney artifact reproduction/portable extraction; if v580 ties/drops/no-scores, let v581 proceed.

### v580 pending; v582 source scan while waiting — 2026-05-19 00:45 UTC

- Live Kaggle: v580 is visible/pending, ref `52790976`; current confirmed best remains `0.949`; 2026-05-19 UTC visible count is `1`. v580 submit process exited after success, so `tender-ridge` is gone by design. v581 `brisk-kelp` remains alive and waits for v580 completion.
- Ran another source scan for possible v582 candidates. Most new/recent hits were clones or lower-evidence: CocoaAI stars v129/v130 are EoS4/EoS3 clones; CocoaAI Karnak/Adarsh/Itshyao S103 are EoS5-like; Kospintr EfficientNet has sample/empty-output risk and should not be submitted blindly.
- Potential later idea-mining candidates: `amulopapa67/bc26-full-yous-gate-rb035-nb-20260517` (Youssef rank + gate rank, attachable sources) and `karnakbaevarthur/optimized-dual-architecture-ensemble` (pc010/gate + taxonomy/mirror/rare lineage). Neither is strong enough to queue ahead of v580/v581.
- Decision: no v582 submitter added. Preserve v580 -> v581 queue and wait for v580 score.

### v580 dropped to 0.944; v581 submitted — 2026-05-19 01:50 UTC

- Live Kaggle: v580 completed at `0.944`, below current best `0.949`. Kill Chaney v37 direct-replay lane for slots; OOF/CV/gate evidence did not transfer. Artifact access blocker remains an idea-mining issue only.
- v581 fallback initially exited because source preflight markers were too brittle for raw notebook JSON. Relaxed source markers to semantic markers (`default_name`, `a2_nfnet_w03`, `A2NF blend complete`, diagnostics, hidden-test markers) while preserving strict output-file verification.
- Re-ran v581 preflight: source pull OK v2, kernel COMPLETE/no failure, required outputs present. Submitted `v581: Guarded direct Lucataco A2Prime NFNet frontier replay`, ref `52793377`, pending. 2026-05-19 UTC visible count is now `2`.

### v581 pending; v582 result-gated fallback staged — 2026-05-19 02:45 UTC

- Live Kaggle: v581 still pending; v580 scored `0.944`; current best remains `0.949`; 2026-05-19 UTC visible count is `2`.
- Added `scripts/submit_v582_amulopapa_yous_gate_when_ready.py`, a result-gated monitor for `amulopapa67/bc26-full-yous-gate-rb035-nb-20260517` v4. It waits while v581 is pending, exits if v581 improves above `0.949`, and only submits if v581 ties/drops/no-scores.
- Started v582 monitor as OpenClaw session `lucky-zephyr`, pid `35622`; first poll confirms it is sleeping on pending v581.
- Independent preflight: source pull OK v4/source length `201355`; required source markers present (`submission_youssef.csv`, `submission_gate.csv`, rank blend `0.65 * yr + 0.35 * gr`, hidden/test markers); kernel COMPLETE/no failure; required output files present. Public output schema is dry-run/sample-sized (3 rows, 235 cols), `row_id` unique, finite values in `[0,1]`.

### v581 hidden timeout; v582 submitted; v583 scan — 2026-05-19 03:50 UTC

- Live Kaggle: v581 completed with no public score. Error description: hidden submission notebook exceeded allowed runtime. Public source-run output schema was valid (`3 x 235`, unique `row_id`, finite), so root cause is runtime timeout, not format.
- v582 gated monitor `lucky-zephyr` observed v581 no-score, re-preflighted Amulopapa Youssef+gate v4 successfully, and submitted `v582: Guarded direct Amulopapa Youssef gate rb035 frontier replay`, ref `52796003`, pending. 2026-05-19 UTC visible count now `3`.
- Next scan while v582 pending: Karnak optimized-dual v3 is a possible but lower-confidence v583 fallback (COMPLETE, valid sample-sized output, known Perch/SED/taxon/gate-ish lineage). Alexy Perch+CNN is direct-unsafe because `submission.csv` has 192 `BC2026_Train_*` rows; do not direct-submit.

### v582 pending; v583 source scan held — 2026-05-19 04:45 UTC

- Live Kaggle: v582 remains pending; current best stays `0.949`; 2026-05-19 visible count remains `3`. No active submitter processes are alive.
- Ran Kaggle `list_kernels` searches across recent/date-run BirdCLEF kernels and score claims `0.951`–`0.955`; no explicit >0.949 public claim surfaced.
- Deep-scanned May 19 candidates. Beicicc `bc26-v65-karnak-safe-may19` and `bc26-karnak-gated-safe-may19` are COMPLETE with valid sample-shaped outputs, but they are EoS5-family Model2/Model5 blends (`0.03/0.97`, `0.0321/0.9679`) with Model5 `0.949`, so low expected upside. Beicicc ungated, Anthony ensemble are direct-unsafe (train-row outputs). Mtoshi V6 is ERROR/no outputs. Mtoshi S106 is EoS5/SafeAlign-like. CocoaAI Mtoshi Visual BirdNET is valid and idea-mining-worthy, but lacks strong score evidence. Rabeya V4 was RUNNING/no outputs.
- Decision: do not queue v583 while v582 is pending; preserve remaining slots. If v582 fails, recheck Rabeya and broader source search before falling back to Beicicc safe/gated EoS5-family variants.

### v582 scored 0.947; v583 S118 submitted — 2026-05-19 05:50 UTC

- Live Kaggle: v582 scored `0.947`, below current best `0.949`; current best remains v574/v575/v576. Youssef+gate rank blend is not a confirmation lane.
- Broadened recent source scan. Rabeya V4 is now inaccessible (`403`). Beicicc safe/gated are valid but EoS5-family low-upside. Zhaorong/Mtoshi Visual BirdNET and CocoaAI Youssef D2/E1 are valid idea-mining candidates but lack strong score evidence. JGuevara TTA outputs a zero fallback; skip.
- Selected distinct Itshyao S118 gated G116 delta launcher as v583 because it includes `submission_g116_hgnet_b1_all5_s118.csv` and is more structurally different than Beicicc EoS5-weight variants. Caveat: visible source is a launcher around attached `s118_source.ipynb`, so direct replay is not immediately repo-portable.
- Added `scripts/submit_v583_s118_gated_g116_delta.py`; preflight passed (source v2, COMPLETE/no failure, required outputs present, prior schema valid). Submitted `v583: Guarded direct S118 gated G116 delta launcher replay`, ref `52799220`, pending. 2026-05-19 visible count now `4`; preserve final slot.

### v583 no-scored; v584 final slot submitted — 2026-05-19 06:50 UTC

- Live Kaggle: v583 completed with no score due hidden unhandled error. Classify S118 as launcher/attached-source hidden failure; avoid S118/S120-style launchers unless source is recovered/ported.
- Final-slot scan found full-source valid candidates: CocoaAI Youssef D2 sonomirror, CocoaAI Youssef E1 rare-tail, Kotata Youssef C2/A1 variants, and Zhaorong Mtoshi Visual BirdNET. Mtoshi Visual CPU source itself is ERROR/no outputs.
- Selected `zhaorongdai/bc26-cocoa-mtoshi-visual-birdnet` v1 for v584 as the most distinct remaining full-source, schema-safe candidate: Visual/BirdNET/Mtoshi lineage with TTA Proto, `0.949-style` prior, per-class ensemble weights, and BirdNET branch. Added `scripts/submit_v584_zhaorong_visual_birdnet.py`; preflight passed; submitted `v584: Guarded direct Zhaorong Mtoshi Visual BirdNET replay`, ref `52800792`, pending. 2026-05-19 UTC visible count is now `5`; day capped.

### v584 pending; capped next-reset scan — 2026-05-19 07:45 UTC

- Live Kaggle: v584 remains pending; current best remains `0.949`; 2026-05-19 UTC visible count is `5`/capped. No active v577/v578/v58x submitters.
- Fresh Kaggle DATE_RUN scan saved `date_run_all_20260519T0740Z.json`; web search found no external explicit `0.950+`/`0.951+` claims.
- Deep-scanned next-reset candidates (`deep_scan4_nextreset_20260519T0740Z.json`). Best distinct fallback if v584 fails is `franksunp/birdclef-2026-5-branch-v4-tta-fix` v1: COMPLETE/no failure, valid sample output, full source, 5-stream rank ensemble with ProtoSSM/Tucker SED/Snowflake/CLAP/BirdNET. Caveat: output is compressed and prior small CLAP/Snowflake sidecars did not improve.
- Backup: `meenalsinha/birdclef-2026-improved` v20 is valid but overlaps the Visual/BirdNET family already being tested by v584. Kojimar `[0.949 LB]` and Beicicc Cocoa Karnak are valid but low-upside EoS/Karnak-family. Rabeya 0.947, aiaiaiooo, WildSound V8, Mtoshi Visual CPU are ERROR/no output. No monitor started while capped.

### v584 scored 0.942; v585 next-reset monitor queued — 2026-05-19 08:50 UTC

- Live Kaggle: v584 scored `0.942`, below current best `0.949`. Visual/BirdNET/Mtoshi direct replay is idea-mining only, not confirmation.
- Added `scripts/submit_v585_franksunp_5branch_tta_fix_when_slot.py` for `franksunp/birdclef-2026-5-branch-v4-tta-fix` v1. Preflight passed: source v1 length `103548`, COMPLETE/no failure, required outputs present. It remains the best distinct next-reset candidate: 5-stream rank ensemble (ProtoSSM/Tucker SED/Snowflake/CLAP/BirdNET), but with caveat that previous CLAP/Snowflake sidecars did not improve.
- Started monitor `quiet-basil` pid `18696`; it attempted submission, hit daily cap (`15 hours from now`), and is sleeping `54120s` before retry. 2026-05-19 UTC count remains `5`/capped.

### v585 monitor restarted — 2026-05-19 09:45 UTC

- Live Kaggle unchanged: v584 `0.942`, best `0.949`, 2026-05-19 count `5`/capped. The earlier `quiet-basil` v585 monitor was no longer visible/alive and no `submit_v585` process was running.
- Restarted v585 monitor as OpenClaw session `mild-harbor` pid `38214`; preflight re-passed and submit attempt hit expected daily cap (`14 hours from now`), then slept `50520s` before retry. No other submitters active.

### v585 moved to durable tmux monitor — 2026-05-19 11:45 UTC

- Live Kaggle unchanged: v584 `0.942`, best `0.949`, 2026-05-19 count `5`/capped. No v585 submission visible.
- Prior managed/nohup v585 monitors were not durable across turns (`mild-harbor` not found; PID-file process absent). Switched to detached tmux session `birdclef-v585-reset`.
- tmux monitor re-preflighted v585 successfully, attempted submit, hit expected daily cap (`12 hours from now`), and is sleeping `43320s` before retry. Inspect with `tmux capture-pane -t birdclef-v585-reset -p | tail -80`.
- Fresh DATE_RUN scan saved `date_run_all_20260519T1040Z.json`; no explicit 0.950+ source found, and no candidate clearly outranks FrankSunP v585 before reset.

### Capped 0.96 frontier source audit — 2026-05-19 12:52 UTC

- Live Kaggle unchanged: v580 `0.944`, v581 timeout/no score, v582 `0.947`, v583 hidden error/no score, v584 `0.942`; best remains `0.949`; 2026-05-19 count `5`/capped. PR #245 is merged; active PR #246 is open/mergeable/blocked. No v577/v578 scalar submitter is active.
- Verified `birdclef-v585-reset` tmux monitor remains alive and sleeping on cap after successful FrankSunP source/output preflight; no duplicate submitter started.
- Saved DATE_RUN scan `date_run_all_20260519T1240Z.json` and source/output audit `source_audit_20260519T1245Z/summary.json`.
- Audited candidates: Pilkwang 949 Rank-Power Soundscape Fusion (COMPLETE/schema-safe, explicit `YUKIZ_BLEND_WEIGHT=0.0264`, `PROTO_RANK_WEIGHT=0.600`, lambda prior `0.5`, rank power `0.6`, but 0.949-family); Aditya Exp019 (COMPLETE/schema-safe, scalar explanation only); Yaroslav v6_0949 (RUNNING/no outputs, lambda prior `0.65` microblend); Shinak 260519 (COMPLETE/schema-safe, interesting joint/circular site-hour prior + TTA but no score claim); Mtoshi notebook (COMPLETE but internet-enabled/same family); HuyDo training/no output; Solokop baseline/low-upside.
- Decision: keep only v585 active for reset. If v585 fails/drops, prefer repo-owned extraction from Pilkwang residual-diversity packaging or Shinak joint/circular prior over another broad 0.949 replay.

### Fresh capped top-feed source audit — 2026-05-19 13:55 UTC

- Live Kaggle unchanged: v580 `0.944`, v581 timeout/no score, v582 `0.947`, v583 hidden error/no score, v584 `0.942`; best remains `0.949`; 2026-05-19 count `5`/capped. PR #245 is merged; PR #246 open/mergeable/blocked. No v577/v578 scalar submitter active.
- Verified `birdclef-v585-reset` tmux monitor remains alive and sleeping on cap after successful FrankSunP preflight; no duplicate submitter started.
- Saved scan `date_run_all_20260519T1345Z.json` and audit `source_audit_20260519T1345Z/summary.json`. Web search found no explicit 0.950/0.951/0.96 source claim.
- Audited new candidates: Rajnish RankPower Safe Candidate is a schema-safe Pilkwang 949 clone; Claude R0946 A2Prime/NFNet is RUNNING/no outputs and low priority given v581 timeout; Yaroslav v6_0949 is COMPLETE but invalid `submission.csv` (243 rows/train rows/empty numeric cells); Adkasd Exp019 Fast is schema-safe but duplicate scalar 0.949-family path; Chaney v67 has useful intermediates but primary `submission.csv` is constant `0.66666675`, so reject direct replay.
- Decision: keep v585 as sole reset submitter and continue mining for genuinely new source/structure; no extra slots queued.

### A2Prime/EffV2S vs NFNet fallback triage — 2026-05-19 14:58 UTC

- Live Kaggle unchanged: v580 `0.944`, v581 timeout/no score, v582 `0.947`, v583 hidden error/no score, v584 `0.942`; best remains `0.949`; 2026-05-19 count `5`/capped. PR #245 merged; PR #246 open/mergeable/blocked. No v577/v578 scalar submitter active.
- Verified `birdclef-v585-reset` tmux monitor remains alive and sleeping on cap after successful FrankSunP preflight; no duplicate submitter started.
- Saved scan `date_run_all_20260519T1445Z.json` and audit `source_audit_20260519T1445Z/summary.json`. Web search again found no explicit 0.950/0.951/0.96 source claim.
- Claude A2Prime/NFNet v6 completed and is schema-safe, but NFNet sanity top-5 hit rate is `0.30` and Proto/NFNet rank correlation is `0.169`; still timeout-risk because v581 no-scored on similar lineage.
- Claude A2Prime/EffV2S v5 is schema-safe and has stronger diversity evidence: sanity top-5 hit rate `0.55`, Proto/EffV2S rank correlation `0.053`. If v585 fails and no 0.950+ source appears, this is the best concrete repo-owned extraction target from the A2Prime family.
- Rajnish RankPower+NFNet selective is schema-safe on primary sample `submission.csv`, but selective intermediate output has train rows / 36 rows on public run; idea-mining only unless row-selection is fixed in a repo-owned port. Aiaiaiooo is RUNNING/no outputs.
- Decision: keep v585 as sole reset submitter; prepare EffV2S extraction next if cap persists and no stronger source appears.

### Repo-owned EffV2S fallback scaffold prepared — 2026-05-19 15:58 UTC

- Live Kaggle unchanged: v580 `0.944`, v581 timeout/no score, v582 `0.947`, v583 hidden error/no score, v584 `0.942`; best remains `0.949`; 2026-05-19 count `5`/capped. PR #245 merged; PR #246 open/mergeable/blocked. No v577/v578 scalar submitter active.
- Verified `birdclef-v585-reset` tmux monitor remains alive and sleeping on cap after successful FrankSunP preflight; no duplicate submitter started.
- Prepared repo-owned fallback scaffold `kaggle-kernels/v586-a2prime-effv2s-extraction/` from `claudedevore/birdclef-2026-r0946-a2prime-effv2s-submit` v5, with private kernel metadata `yourslewis/bc26-v586-a2prime-effv2s-extraction`, internet disabled, and matching dataset/kernel/model sources.
- Added push helper `scripts/push_v586_a2prime_effv2s_extraction.py`; it only pushes the private kernel and does not submit. Do not run while v585 owns the reset slot.
- Validation passed: notebook JSON parses (39 cells); metadata includes `baiyuby/birdclef2026-distill-models` and model sources; push helper and v585 submitter compile; `git diff --check` and hygiene pass.
- Decision: keep v585 as sole reset submitter. Use v586 only if v585 drops/no-scores and no stronger 0.950+ source appears.

### Capped 0.96 frontier re-scan + Yaroslav/visual audit — 2026-05-19 16:50 UTC

- Live Kaggle unchanged: v580 `0.944`, v581 timeout/no score, v582 `0.947`, v583 hidden unhandled error/no score, v584 `0.942`; current confirmed best remains `0.949` from v574/v575/v576; target remains `0.960`.
- 2026-05-19 UTC visible submission count remains `5`/capped. No v577/v578 scalar submitter is active.
- Verified durable tmux monitor `birdclef-v585-reset` remains alive and sleeping on the daily cap after successful FrankSunP v585 source/output preflight; no duplicate submitter was started.
- Saved fresh DATE_RUN scan `artifacts/public_kernels_20260519_frontier_candidates/date_run_all_20260519T1648Z.json` and source/output audit `artifacts/public_kernels_20260519_frontier_candidates/source_audit_20260519T1649Z/summary.json`.
- `yaroslavkholmirzayev/v6-0949-replay` reran at 16:37 UTC but is still unsafe for direct replay: primary `submission.csv` has 243 rows, includes 240 train rows, and has empty numeric cells (`finite_bad=56862`). Its sample-shaped `subm_5.csv`/`subm_karnakbayev_power_optimization.csv` are EoS5/Karnakbayev-family outputs, not a new 0.96 structure.
- `meenalsinha/birdclef-2026-improved` reran at 16:32 UTC but primary `submission.csv` is train-row dry-run output (240 train rows), matching the prior hidden failure class; do not direct-submit.
- `samejimatink0/birdclef-2026-visual-cpu-inference` is COMPLETE with finite outputs, but primary `submission.csv` is train-row dry-run output and the source/output markers are essentially ProtoSSM/SED/BirdNET rather than a confirmed new visual path; idea-mining only.
- `aiaiaiooo/birdclef2026` has hidden-path markers in source but no session outputs; not slot-ready.
- Decision: keep v585 as the sole active next-reset submitter. Keep v586 EffV2S as the prepared repo-owned fallback only if v585 drops/no-scores and no genuine `0.950+` source appears. No new Kaggle push/submission was made.

### Capped source scan — Nina EoS.6 appears but is still running — 2026-05-19 17:49 UTC

- Live Kaggle unchanged: v580 `0.944`, v581 timeout/no score, v582 `0.947`, v583 hidden unhandled error/no score, v584 `0.942`; current confirmed best remains `0.949`; 2026-05-19 UTC count remains `5`/capped.
- PR #245 is merged; PR #246 remains open/blocked. No v577/v578 scalar submitter is active.
- `birdclef-v585-reset` is still the only active submitter and is sleeping on daily cap after successful FrankSunP v585 source/output preflight.
- Fresh scan saved `artifacts/public_kernels_20260519_frontier_candidates/date_run_all_20260519T1747Z.json`.
- New high-signal candidate: `nina2025/birdclef-2026-eos-6-silver-zone` v9, found at the top of DATE_RUN. Pulled source through Bearer API and saved audit `artifacts/public_kernels_20260519_frontier_candidates/source_audit_20260519T1747Z_eos6/summary.json` plus parsed cells file.
- EoS.6 v9 source is a direct EoS successor with public/attachable sources only. Its active config blends `Model_21`/`Model_73`/`Model_74` with weights `0.032/0.967/0.001`; `Model_73` and `Model_74` are Yaroslav/Karnakbayev 0.949-family branches with xSED `[0.60,0.40]` and `[0.605,0.395]`. The markdown says v7 timed out and v9 is intended to run SED once, but the active code uses `task1='run SED once'` while the early Model_1 guard checks `task`, so this still needs actual output/status validation rather than blind trust.
- EoS.6 v9 live status at audit time was `RUNNING` with no outputs yet; therefore it is not direct-submit-safe and not ready to displace v585. If it completes before reset with valid sample-shaped `submission.csv` and no failure, it becomes a strong candidate to consider ahead of v585; if it times out/no-outputs, keep v585.
- Also checked `damianleandrotamburi/20260329-birdclef` v70; it has no outputs and no 0.949/0.95/0.96/source-family evidence, so it is not slot-ready.
- Web searches for explicit `0.950`/`0.951`/`0.96` BirdCLEF code claims returned no results.
- Decision: do not submit or push anything while capped. Preserve v585 monitor for the reset slot, but put EoS.6 v9 at the top of the recheck queue for the next cron before reset.

### Public946 sidecar lesson reminder

- Keep using train-soundscape/local gates only as rejection filters, not approval filters: v560 and v573 had positive local signals but dropped publicly. This is why EoS.6 needs real public-kernel completion/output schema and, ideally, direct LB evidence before it replaces the existing queue.

### Heartbeat EoS.6 availability recheck — 2026-05-19 17:52 UTC

- Heartbeat rechecked live submissions: state unchanged (`0.949` best; v580 `0.944`, v581 timeout, v582 `0.947`, v583 hidden error, v584 `0.942`; 2026-05-19 UTC capped at 5).
- Rechecked `nina2025/birdclef-2026-eos-6-silver-zone`: session status API still reports `RUNNING`, but output list is empty and both `/api/v1/kernels/pull/nina2025/birdclef-2026-eos-6-silver-zone` and SDK `GetKernel` now return `404 Not Found`; fresh list search no longer finds it. Treat this as unavailable/not direct-submit-safe until it reappears with pullable source and outputs.
- Briefly prepared an EoS.6 takeover watcher, but discarded it after the fresh `pull/get` 404 made automated takeover unsafe. v585 FrankSunP remains the only active reset-slot monitor.
- No new submission, no push, no PR/merge action.

### Cron recheck — EoS.6 invalid primary, NFNet lprior075 triage — 2026-05-19 18:50 UTC

- Live Kaggle unchanged: v580 `0.944`, v581 timeout/no score, v582 `0.947`, v583 hidden unhandled error/no score, v584 `0.942`; current confirmed best remains `0.949` from v574/v575/v576; 2026-05-19 UTC count remains `5`/capped.
- PR #245 is merged; PR #246 remains open/blocked. No v577/v578 scalar submitter is active.
- `birdclef-v585-reset` remains the only active submitter. FrankSunP v585 preflight is still valid: public kernel COMPLETE/no failure and required outputs include `submission.csv`, ProtoSSM/SED/BirdNET branch CSVs, CLAP/Snowflake arrays, and site-hour prior/cache files. It is sleeping on the daily cap and has not submitted yet.
- Rechecked Nina EoS.6 under its current visible slug `nina2025/birdclef-2026-eos-6-sz`. It is now COMPLETE/no failure and pullable as version 9, but primary `submission.csv` is invalid: `243` rows, `240` train rows, and `56862` empty/non-finite numeric cells. Sample-shaped side outputs `subm_73.csv`/`subm_74.csv` are valid, but the competition submission target is the invalid primary file. Reject direct replay and do not displace v585.
- Saved EoS.6 output schema audit: `artifacts/public_kernels_20260519_frontier_candidates/eos6_outputs_20260519T1847Z/summary.json`.
- Fresh DATE_RUN scan saved `artifacts/public_kernels_20260519_frontier_candidates/date_run_all_20260519T1847Z.json`.
- Audited new top candidate `nicolasschuldt/nfnet-lprior075` v1; saved `artifacts/public_kernels_20260519_frontier_candidates/source_audit_20260519T1847Z_new/summary.json`. It is COMPLETE/no failure, pullable, and primary `submission.csv` is sample-shaped/finite. However, source is still EoS5/RankPower-family with `RUN_MODE="eos5_locked"`, `YUKIZ_BLEND_WEIGHT=0.0264`, `PROTO_RANK_WEIGHT=0.600`, `lambda_prior=0.75`, plus small NFNet selective graft (`NFNET_BLEND_W=0.035`, `NFNET_SPIKE_W=0.080`). The selective NFNet intermediate output is train-row-only on public sample. Treat as idea-mining/fallback, not high-upside enough to replace v585 or the prepared repo-owned v586 EffV2S fallback.
- Public946 sidecar lesson remains active: train-soundscape/local gates are rejection filters only. v560/v573 had positive local gates but dropped publicly, so no source gets a slot unless primary output is hidden/test safe and the hypothesis is distinct enough.
- Decision: keep v585 as reset-slot owner; do not launch duplicate submitters. If v585 drops/no-scores, prefer prepared repo-owned v586 EffV2S extraction before spending a direct slot on another RankPower/NFNet scalar-family clone.

### Cron recheck — v585 holds reset slot, fresh top-feed rejects — 2026-05-19 19:50 UTC

- Live Kaggle unchanged: v580 `0.944`, v581 timeout/no score, v582 `0.947`, v583 hidden unhandled error/no score, v584 `0.942`; current confirmed best remains `0.949` from v574/v575/v576; 2026-05-19 UTC count remains `5`/capped.
- PR #245 and PR #246 are now merged. Continued new logging on fresh branch `feature/birdclef-096-frontier-v585-hold-20260519` from updated `origin/main`.
- No v577/v578 scalar submitter is active. `birdclef-v585-reset` remains the only active submitter; FrankSunP v585 is still COMPLETE/no failure with required outputs and is sleeping on the daily cap after its prior submit attempt failed only due quota.
- Fresh DATE_RUN scan saved `artifacts/public_kernels_20260519_frontier_candidates/date_run_all_20260519T1947Z.json`.
- Audited fresh/re-run top-feed candidates; saved `artifacts/public_kernels_20260519_frontier_candidates/source_audit_20260519T1947Z_top/summary.json`.
- `meenalsinha/birdclef-2026-improved` v22 reran at 19:40 UTC. It is pullable and outputs CSVs, but every relevant output including primary `submission.csv` is train-row dry-run (`240` train rows); reject direct replay. Source comments call it `exp_067: v6_prior065`, a lambda-prior scalar direction rather than a distinct 0.96 path.
- `nina2025/birdclef-2026-eos-6-sz` v10 reran at 19:06 UTC. It still has invalid primary `submission.csv` (`243` rows, `240` train rows, `56862` empty/non-finite cells). New markdown says v8 scored `0.949`, but v10 primary remains unsafe; do not submit. Side output `subm_73.csv` is sample-shaped but only reproduces the same 0.949-family branch.
- `evgendvorkin/birdclef-baseline` v33 is a low-signal baseline; primary output is train-row dry-run (`120` train rows), no 0.949/0.95/0.96/EoS/RankPower evidence.
- `nicolasschuldt/nfnet-lprior075` remains idea-mining/fallback only: primary is schema-safe, but it is mostly EoS5/RankPower with `lambda_prior=0.75` and tiny NFNet selective graft; NFNet selective output is train-row-only.
- Public946 sidecar lesson remains active: local train-soundscape gates reject bad candidates but do not approve submissions. v560/v573 had positive local gates and dropped publicly.
- Decision: keep v585 FrankSunP as reset-slot owner. No new Kaggle push/submission and no duplicate submitter. If v585 drops/no-scores, prefer the prepared repo-owned v586 EffV2S extraction path before direct RankPower/NFNet clones.

### Cron recheck — v585 still capped, 20:47 top-feed audit — 2026-05-19 20:50 UTC

- Live Kaggle unchanged: v580 `0.944`, v581 timeout/no score, v582 `0.947`, v583 hidden unhandled error/no score, v584 `0.942`; current confirmed best remains `0.949` from v574/v575/v576; 2026-05-19 UTC count remains `5`/capped.
- PR #247 remains open/blocked on branch `feature/birdclef-096-frontier-v585-hold-20260519`. No v577/v578 scalar submitter is active.
- `birdclef-v585-reset` remains the only active submitter. FrankSunP v585 is still COMPLETE/no failure with required outputs and is sleeping on daily cap after a quota-only failed submit attempt.
- Fresh DATE_RUN scan saved `artifacts/public_kernels_20260519_frontier_candidates/date_run_all_20260519T2047Z.json`.
- Fresh top-feed audit saved `artifacts/public_kernels_20260519_frontier_candidates/source_audit_20260519T2047Z_top/summary.json`.
- `evgendvorkin/birdclef-baseline` v34 reran at 20:14 UTC. Primary `submission.csv` is still train-row dry-run (`240` train rows), and source has no 0.949/0.95/0.96/EoS/RankPower evidence. Reject.
- `meenalsinha/birdclef-2026-improved` v22 remains train-row-only on all relevant outputs, including primary `submission.csv`; source is lambda-prior scalar, not 0.96 frontier. Reject.
- `nina2025/birdclef-2026-eos-6-sz` v10 remains invalid on primary `submission.csv` (`243` rows, train rows, empty/non-finite cells); side output `subm_73.csv` is sample-shaped but not the configured competition file and only represents the 0.949-family branch. Reject direct replay.
- No fresh 0.950+/0.96 source with safe primary output appeared. Public946 sidecar/local gate lesson remains active: train-soundscape gates reject candidates but do not approve slots.
- Decision: keep v585 FrankSunP as reset-slot owner; no duplicate submitter and no new push/submission. If v585 drops/no-scores, use prepared repo-owned v586 EffV2S path before direct RankPower/NFNet clones.

### Cron recheck — v585 still queued, 21:47 top-feed audit — 2026-05-19 21:50 UTC

- Live Kaggle unchanged: v580 `0.944`, v581 timeout/no score, v582 `0.947`, v583 hidden unhandled error/no score, v584 `0.942`; current confirmed best remains `0.949` from v574/v575/v576; 2026-05-19 UTC count remains `5`/capped. v585 is not visible yet.
- PR #247 remains open/blocked on branch `feature/birdclef-096-frontier-v585-hold-20260519`. No v577/v578 scalar submitter is active.
- `birdclef-v585-reset` remains the only active submitter. FrankSunP v585 is still COMPLETE/no failure with required outputs and is sleeping on daily cap after a quota-only failed submit attempt.
- Fresh DATE_RUN scan saved `artifacts/public_kernels_20260519_frontier_candidates/date_run_all_20260519T2147Z.json`.
- Fresh top-feed audit saved `artifacts/public_kernels_20260519_frontier_candidates/source_audit_20260519T2147Z_top/summary.json`.
- `huydo170302/dsai1-internship-birdclef-2026` v8 is the newest DATE_RUN item but is an EDA/training baseline notebook: no `test_soundscapes`, no `sample_submission.csv`, no `submission.csv`, no outputs, and no 0.949+/0.95/0.96/EoS/RankPower evidence. Reject.
- `evgendvorkin/birdclef-baseline` v34 remains train-row primary output only (`240` train rows) and low-signal. Reject.
- `meenalsinha/birdclef-2026-improved` v22 remains train-row-only across primary and branch outputs, with source describing `lambda_prior=0.60 -> 0.65` scalar tuning. Reject.
- `nina2025/birdclef-2026-eos-6-sz` v10 remains invalid on primary `submission.csv` (`243` rows, train rows, empty/non-finite cells). Side `subm_73.csv` is sample-shaped but not the competition file and only captures a 0.949-family branch. Reject direct replay.
- Public946 sidecar lesson remains explicit: local/train-soundscape gates are rejection filters, not approval filters; v560/v573 had positive local gates and still dropped.
- Decision: keep v585 FrankSunP as reset-slot owner; no duplicate submitter and no new push/submission. If v585 drops/no-scores, use prepared repo-owned v586 EffV2S path before direct RankPower/NFNet clones.

### Cron recheck — v585 still queued, 22:47 top-feed audit — 2026-05-19 22:50 UTC

- Live Kaggle unchanged: v580 `0.944`, v581 timeout/no score, v582 `0.947`, v583 hidden unhandled error/no score, v584 `0.942`; current confirmed best remains `0.949` from v574/v575/v576; 2026-05-19 UTC count remains `5`/capped. v585 is not visible yet.
- PR #247 remains open/blocked on branch `feature/birdclef-096-frontier-v585-hold-20260519`. No v577/v578 scalar submitter is active.
- `birdclef-v585-reset` remains the only active submitter. FrankSunP v585 is still COMPLETE/no failure with required outputs and is sleeping on daily cap after a quota-only failed submit attempt; expected retry remains near reset, so no duplicate submitter was started.
- Fresh DATE_RUN scan saved `artifacts/public_kernels_20260519_frontier_candidates/date_run_all_20260519T2247Z.json`.
- Fresh top-feed audit saved `artifacts/public_kernels_20260519_frontier_candidates/source_audit_20260519T2247Z_top/summary.json`.
- `meenalsinha/birdclef-2026-improved` v23 is newly running/no outputs at audit time. Source is still scalar RankPower/lambda-prior tuning (`lambda_prior=0.60 -> 0.65`, `Model_4` 0.10 + `Model_7` 0.90 with xSED `[0.650,0.350]`), not a high-upside 0.96 structure. Do not displace v585.
- `jguevarag/07-optimal-sed-training` v4 has no outputs and is a training notebook, not a submission candidate.
- `pilkwang/949-birdclef-2026-acoustic-prior-field-fusion` v6 is COMPLETE/no failure and primary `submission.csv` is schema-safe/sample-shaped. Source is a well-documented EoS6/Yaroslav/Pilkwang prior-field branch with yukiZ low-weight diversity and v6 prior/scalar knobs. It is useful idea-mining/fallback, but still explicitly 0.949-family and does not outrank v585's more structurally distinct 5-branch FrankSunP hypothesis under the 0.960 target.
- `adarsh5harma/birdclef-2026-v66-phase1-integrated` v1 has invalid primary `submission.csv` (`243` rows / train rows / empty cells) despite sample-shaped side output; reject direct replay.
- `muhammadsaadalvi/birdclef-2026-wildsound-v8` v68 has no outputs and no high-score/source-family evidence; not slot-ready.
- Public946/local-gate lesson remains active: train-soundscape gates are rejection filters only, not slot approvals.
- Decision: keep v585 FrankSunP as reset-slot owner; no new push/submission and no duplicate submitter. If v585 drops/no-scores, first push/verify the prepared repo-owned v586 EffV2S extraction before direct RankPower/PriorField/NFNet clones.

### Broad 0.96 source-frontier audit — 2026-05-19 23:30 UTC

- User requested another research round targeting `0.960` public LB. I treated this as a discovery/audit pass, not a slot burn.
- Live state at start of pass: latest visible submissions still v580 `0.944`, v581 timeout/no score, v582 `0.947`, v583 hidden error/no score, v584 `0.942`; current confirmed best remains `0.949`; 2026-05-19 UTC is capped at 5. `birdclef-v585-reset` remains alive and is the sole active reset-slot submitter.
- PR #247 had merged into `main`; created new branch `feature/birdclef-096-broadscore-audit-20260519` for this round's notes.
- Fresh broad Kaggle kernel search saved `artifacts/public_kernels_20260519_frontier_candidates/broad_score_search_20260519T2323Z.json`, using DATE_RUN, SCORE_DESCENDING, and VOTE_COUNT over `birdclef 2026`, `0.95`, `0.950`, `0.951`, `0.96`, `0.960`, `silver`, `bronze`, `949`, and `eos 6` queries.
- Web search for explicit `BirdCLEF 2026 0.950/0.96 public LB notebook` found no stronger public source claim beyond Kaggle EoS.3-style pages.
- Broad-score source audit saved `artifacts/public_kernels_20260519_frontier_candidates/source_audit_20260519T2323Z_broadscore/summary.json`.
- Legacy/diverse source audit saved `artifacts/public_kernels_20260519_frontier_candidates/source_audit_20260519T2335Z_legacydiverse/summary.json`.

Key audited candidates:

1. `ulyanovantonamaranta/birdclef-2026-gate-fake008-head0015` v5 — COMPLETE/pullable/schema-safe primary. It is a Vyanktesh/ProtoSSM+SED family notebook with final `pc010=0.70` + `rank1_pc015_head005=0.30`, where `HEAD_RANK_BLEND=0.0500` inside the rank1 branch. Net train-audio-head contribution is about `1.5%`, similar in spirit to already-tested head-sidecar work (v573 scored `0.945`). Useful idea-mining but not enough to displace v585/v586.
2. `cliff376/bc26-public-gate-combo-pc010-v2` v2 — COMPLETE/pullable/schema-safe primary. Same public-gate/Proto fat-tail continuity family without the dual head final average. Mostly a `0.941`/`0.945` public-gate branch; lower-upside than v585 and prepared v586.
3. `raunakdey07/birdclef-2026-multi-model-ensemble` v9 — COMPLETE/pullable/schema-safe primary. Adds sonotype mirroring and rare-class adaptive thresholding on top of Proto/SED rank blend. Interesting postprocess ideas, but still public ProtoSSM/SED lineage and not a direct 0.950+ source.
4. `marynaborovska/birdclef-26-two-pass-ssm-advanced-pp` v3 — source is architecturally interesting (LightProtoSSM + MLP probes + ResidualSSM + adaptive smoothing + isotonic thresholds), but current audit has no outputs, so it is not direct-submit-safe. Keep as idea-mining for future repo-owned work, not a reset-slot candidate.
5. `aminmahmoudalifayed/birdclef-2026` v11 — not submission-safe; primary `submission.csv` is empty/invalid. Reject direct replay.
6. `anthonytherrien/birdclef-2026-ensemble`, `beicicc/bc26-anthony-ens-safe-may19`, `kijiang/birdclef2026-v337`, `karnakbaevarthur/gated-rank-fusion-pipeline`, `nicolasschuldt/eos5-meta`, `apachikoff/birdclef-2026-eos-5`, `starsdaisuki/birdclef-2026-v130-nina-eos3`, `beicicc/bc26-v63-nina-eos5-may18`, and `adityaraghuvanshi999/birdclef-2026-safe-eos5-rank-blend-validation` are all EoS/EoS5/Karnak/RankPower-family variants around Model_2 + Model_5 / Model_10 blends. Several are schema-safe, but the source evidence explicitly documents saturation at `0.949` and weight-sweep deltas (`0.04/0.96`, `0.035/0.965`, etc.), so they should not consume a slot while chasing `0.960`.
7. `apachikoff/birdclef-2026-v6` is schema-safe but is a `0.948`/V6/BirdNET branch already represented in the EoS/Karnak family, not a new frontier.

Decision:

- Keep v585 FrankSunP as reset-slot owner.
- Do not start a duplicate submitter and do not submit/push a new Kaggle candidate while capped.
- The best prepared fallback remains repo-owned v586 A2Prime/EffV2S extraction if v585 drops/no-scores and no stronger `0.950+` source appears.
- New idea-mining queue from this pass: (a) Ulyanov dual gate/head blend only as a low-risk postprocess idea, (b) Raunak sonotype mirroring / rare-class thresholding as class-specific postprocess research, (c) Maryna two-pass SSM architecture as a heavier repo-owned architecture experiment. None outrank v585/v586 as next slot owner.

### v585 submitted after UTC reset — 2026-05-20 00:06 UTC

- Held the run through the reset window because `birdclef-v585-reset` retried at 2026-05-19 23:47 UTC and Kaggle reported the cap would clear in 14 minutes.
- v585 submitted successfully at reset: ref `52831360`, description `v585: Guarded direct FrankSunP 5-branch V4 TTA Fix replay`, date `2026-05-20T00:01:09.75Z`, status `pending`, file `submission.csv`.
- Current confirmed best remains `0.949` from v574/v575/v576 until v585 scores. Do not submit v586 or any other 2026-05-20 slot before v585 result unless Wenhao explicitly asks; v585 is the active high-upside result gate.
- The `birdclef-v585-reset` tmux session exited after submission; no stale v577/v578 scalar submitters were visible.
- Fresh pre-reset DATE_RUN scan saved `artifacts/public_kernels_20260519_frontier_candidates/date_run_all_20260519T2347Z.json`; it found no new candidate beyond already-audited Meenal/Pilkwang/JGuevara/Adarsh/WildSound/Nina/NFNet family.
- PR #248 remains open/blocked with the broad 0.96 source audit notes; no merge performed.

Decision:

- Wait for v585 score/status.
- If v585 improves materially, immediately port/confirm FrankSunP as repo-owned.
- If v585 drops/no-scores, first push/verify the prepared repo-owned v586 A2Prime/EffV2S extraction, then decide whether to submit it.

### v585 pending, fresh post-reset source scan — 2026-05-20 00:48 UTC

- v585 remains visible and pending: ref `52831360`, description `v585: Guarded direct FrankSunP 5-branch V4 TTA Fix replay`, date `2026-05-20T00:01:09.75Z`, status `pending`, no score/error yet. Current confirmed best remains `0.949`; 2026-05-20 UTC count is `1`.
- No v577/v578/v585 submitter processes remain active. PR #248 remains open/blocked; no merge performed.
- Fresh scan saved `artifacts/public_kernels_20260520_frontier_candidates/scan_20260520T0047Z.json`.
- Fresh top-source audit saved `artifacts/public_kernels_20260520_frontier_candidates/source_audit_20260520T0047Z_top/summary.json`.
- `mtoshidesu/notebookc6e90ae327` v3: pullable, but no outputs yet. Source is documentation/executable scaffold around `Model_7` final power-optimized path (`LB 0.948`) and true-OOF validation warnings. Not direct-submit-safe while outputless.
- `zeyadmohamadezzat/birdclef-2026-proto-fusion-and-temporal-flip` v3: COMPLETE/pullable/schema-safe primary `submission.csv`. Source explicitly says it is a consolidated `0.949` inference script from exp019/EoS4/Karnakbayev PowerOptimization, with final preset `proto=0.56`, `sed=0.36`, `birdnet=0.04`, `cnn=0.04`; side outputs are train-row/constant dry-run artifacts. Useful as idea-mining for BirdNET/public-CNN fail-closed branch structure, but still 0.949-family and not enough to displace v585 pending or prepared v586 EffV2S.
- `meenalsinha/birdclef-2026-improved` v23: now has outputs, but primary `submission.csv` is train-row dry-run output (`240` train rows), not competition-safe. Source remains v6_prior065 / scalar lambda-prior tuning (`lambda_prior=0.60 -> 0.65`, Model_4 0.10 + Model_7 0.90). Reject direct replay.

Decision:

- Do not spend another 2026-05-20 slot while v585 is pending.
- If v585 improves, port/confirm FrankSunP.
- If v585 drops/no-scores, prepared repo-owned v586 A2Prime/EffV2S still outranks the newly audited Zeyad/Mtoshi/Meenal paths because it is a more structurally distinct extraction target.

## 2026-05-20 02:05 UTC / 2026-05-19 PDT — v585 drop, v586 EffV2S push + guarded submitter

- **Live LB/submission state:** v585 (`52831360`, `v585: Guarded direct FrankSunP 5-branch V4 TTA Fix replay`) scored `0.922`, far below the current confirmed best `0.949` from v574/v575/v576. Treat FrankSunP 5-branch replay as dead for confirmation/porting. 2026-05-20 UTC has 1 visible submission used; no v577/v578 scalar submitter or duplicate v585/v586 submitter was active at the initial check.
- **Chosen track:** moved to the prepared repo-owned A2Prime/EffV2S fallback rather than spending a slot on direct EoS/Karnak/RankPower clones.
- **Push details:** initial v586 push to `yourslewis/bc26-v586-a2prime-effv2s-extraction` failed with `Invalid DockerImagePinningType` because Kaggle's current API only accepts `original/latest`-style pinning, not `PIN_CURRENT_IMAGE`. Removed the stale pinning field and added explicit API error handling. The original slug then returned `Notebook not found` (poisoned no-version kernel record), so pushed a clean private r2 kernel: `yourslewis/bc26-v586-a2prime-effv2s-extraction-r2`, version `1`, kernel id `119902655`, URL `https://www.kaggle.com/code/yourslewis/bc26-v586-a2prime-effv2s-extraction-r2`.
- **Important file choice:** the source notebook's `submission.csv` defaults to conservative `base_3way`; that would not test EffV2S. Added guarded submitter `scripts/submit_v586_a2prime_effv2s_w08_when_ready.py` to submit alternate output `submission_a2_effv2s_w08.csv` as the actual v586 hypothesis. Description: `v586: Repo-owned A2Prime EffV2S rank blend w08 after v585 drop`.
- **Validation:** `python3 -m py_compile scripts/push_v586_a2prime_effv2s_extraction.py scripts/submit_v586_a2prime_effv2s_w08_when_ready.py` and `git diff --check` passed. Submitter source preflight passed against r2 (`version=1`, source length `97070`) and is waiting for Kaggle status to become COMPLETE. Current status at launch: RUNNING/no failure/no outputs yet.
- **Fresh frontier scan while v586 runs:** saved `scan_20260520T0158Z.json` and `source_audit_20260520T0158Z_top/summary.json`. Samejima Visual CPU is COMPLETE/schema-output but same Visual/BirdNET family that already dropped via v584. Pilkwang `(949) Acoustic Prior-Field Fusion` and Adarsh v67 are COMPLETE/schema-output but explicit `0.949` prior-field/RankPower family. Mtoshi `testBirdCLEF+ 2026 V6` is ERROR. No new source displaced v586.
- **Next:** let the guarded v586 submitter finish. It will require COMPLETE/no failure, required output files, `a2_effv2s_w08` active in `a2prime_blend_summary.csv`, and finite/nonconstant `submission_a2_effv2s_w08.csv` before submitting. If v586 errors or its output preflight fails, continue source frontier scanning rather than falling back to scalar EoS5 tuning.

## 2026-05-20 02:55 UTC / 2026-05-19 PDT — v586 v2 filename fix + v587 S121 submitted

- **Status check:** v585 remains a confirmed drop at `0.922`; best remains `0.949` from v574/v575/v576. At `2026-05-20T02:51Z`, visible 2026-05-20 submissions are `2/5` after v587 submission; v587 is pending and v586 is not visible yet.
- **v586 issue/fix:** v586 r2 version 1 completed cleanly and produced EffV2S outputs, including `submission_a2_effv2s_w08.csv`, but Kaggle rejected code submission because BirdCLEF only permits `fileName=submission.csv` (`Submission files must be named "submission.csv" for this Competition.`). Updated the notebook so `default_name` prefers `a2_effv2s_w08` when present, making the EffV2S w08 candidate the notebook's `submission.csv`. Pushed `yourslewis/bc26-v586-a2prime-effv2s-extraction-r2` version `2`; submitter `scripts/submit_v586_a2prime_effv2s_w08_when_ready.py` now targets version 2 and normal `submission.csv`.
- **v586 validation:** source preflight passed for r2 v2 (`source_len=97128`), and the guarded monitor is alive (`logs/v586_a2prime_effv2s_w08_submitter_v2.log`, PID `12539`) waiting for Kaggle RUNNING -> COMPLETE before output preflight/submission.
- **Fresh frontier scan:** saved `scan_20260520T0247Z.json` and `source_audit_20260520T0247Z_top/summary.json`. New top candidates included Mtoshi notebookc6e90ae327 v4 (now COMPLETE but same Karnak/RankPower/visual family), Koushik Pantanal (ERROR), Qiuzi HGNet training (RUNNING/training artifacts only), Rikuter v6 replay/prior-field reproductions (0.949 family), and Itshyao S121.
- **v587 selected/submitted:** Itshyao `birdclef-2026-s121-s114-g116-f1-delta` v1 is a distinct source replay: S114/0.949-family anchor plus `G116 HGNetV2-B1 raw-pseudo fold1` protected delta with `S121_DELTA_WEIGHT=0.035`, top-k overlap guards, and row-id mismatch fallback. Public run is COMPLETE/no failure, outputs include `submission.csv`, `submission_g116_hgnet_b1_fold1_s121.csv`, `submission_protossm.csv`, `submission_sed.csv`, `subm_karnakbayev_power_optimization.csv`, and `v17_logs.json`; log contains `S121 final submission`. Added `scripts/submit_v587_itshyao_s121_g116_delta_when_ready.py`; source/output preflight passed and submitted successfully as ref `52835586`, description `v587: Guarded direct Itshyao S121 S114 plus G116 F1 delta replay`, status pending.
- **Next:** monitor v587 score and v586 v2 completion/submission. If v587 improves, port/confirm the S121 protected G116 delta path repo-owned. If v586 completes and submits, compare both against `0.949` and continue source frontier scanning for non-RankPower structural lines.

## 2026-05-20 03:50 UTC / 2026-05-19 PDT — v586 submitted, v588 S122 submitted, last slot reserved

- **Status check:** v586, v587, and v588 are now all visible and pending. Current confirmed best remains `0.949`; v585 remains a drop at `0.922`. 2026-05-20 UTC visible submissions used: `4/5`. No v577/v578 scalar submitters or v586/v587/v588 monitors remain active.
- **v586 submitted:** r2 version 2 completed successfully. Output preflight passed: required outputs present; `a2prime_blend_summary.csv` contains `a2_effv2s_w08` with `proto_effv2s_rank_corr=0.052920`; `submission.csv` is now the w08 candidate and passed finite/nonconstant shape diagnostics (`rows=3`, `cols=235`, `min=0.470359`, `max=0.540824`, sampled unique `227`). Submitted ref `52835975`, description `v586: Repo-owned A2Prime EffV2S rank blend w08 after v585 drop`, status pending.
- **Fresh 03:47 source scan:** saved `scan_20260520T0347Z.json` and `source_audit_20260520T0347Z_top/summary.json`. New/updated findings: Samejima Visual CPU is RUNNING and same weak Visual/BirdNET family; Nina EoS6 v11 is COMPLETE but saturated EoS/Karnak/RankPower family; Kijiang v338 is COMPLETE 0.949/EoS-family; Koushik acoustic species ID has no useful output listing/high-score path; Itshyao S122 is a credible sibling to v587.
- **v588 selected/submitted:** Itshyao `birdclef-2026-s122-s114-g123-f1-delta` v1 is S114/0.949-family anchor plus `G123 EfficientNetV2-B0 pseudo fold1` protected delta with `S122_DELTA_WEIGHT=0.045`, top-k overlap guards (`top3>=0.80`, `top10>=0.86`), and row-id mismatch fallback. Public run is COMPLETE/no failure, outputs include `submission.csv`, `submission_g123_effv2b0_fold1_s122.csv`, `submission_protossm.csv`, `submission_sed.csv`, `subm_karnakbayev_power_optimization.csv`, and `v17_logs.json`; log contains `S122 final submission` and no sidecar-failed marker. Added `scripts/submit_v588_itshyao_s122_g123_delta_when_ready.py`; source/output preflight passed and submitted ref `52836864`, description `v588: Guarded direct Itshyao S122 S114 plus G123 F1 delta replay`, status pending.
- **Decision:** reserve the final 2026-05-20 slot until at least one of v586/v587/v588 scores or an unusually strong non-saturated source appears. Do not spend it on RankPower/PriorField/EoS clones.

## 2026-05-20 04:50 UTC / 2026-05-19 PDT — v586 dropped, v587/v588 tied, v589 final slot submitted

- **Status check:** v586 scored `0.941`, below the `0.949` plateau; reject A2Prime/EffV2S w08 as a slot lane. v587 S121/G116 and v588 S122/G123 both scored `0.949`, tying the current best but not improving. Current confirmed best remains `0.949`. No v577/v578 scalar submitters or stale monitors are active.
- **Lesson:** protected G116/G123 sidecar deltas are safe enough to tie but have not lifted above the Karnak/PowerOptimization plateau. Treat them as idea-mining/porting only if a later variant clears `0.949`. A2Prime/EffV2S w08 is actively bad (`0.941`) despite low Proto correlation, reinforcing that low local/source correlation is not an approval signal.
- **Fresh final-slot scan:** saved `scan_20260520T0447Z.json` and `source_audit_20260520T0447Z_top/summary.json`. New source findings: Itshyao S123/S124 G124 EffV2S siblings, Kijiang v339/EoS-style 0.949 family, Qiuzi HGNet training-only, Mtoshi/Samejima visual runs ERROR/weak family, Nina EoS6 v11 complete but saturated EoS/Karnak family.
- **v589 selected/submitted:** Used the final slot on Itshyao `birdclef-2026-s124-s114-g124-f1-rankblend` v1, a sibling to v587/v588 but with G124 EfficientNetV2-S 2025-pretrained pseudo fold1 protected **rank blend** (`S124_RANK_WEIGHT=0.115`, top3/top10 guards `0.56/0.68`) rather than a delta. Public run COMPLETE/no failure; required outputs present (`submission.csv`, `submission_g124_effv2s_fold1_s124.csv`, `submission_protossm.csv`, `submission_sed.csv`, `subm_karnakbayev_power_optimization.csv`, `v17_logs.json`); log contains `S124 final submission` and no rank-sidecar failure marker. Added `scripts/submit_v589_itshyao_s124_g124_rankblend_when_ready.py`; source/output preflight passed and submitted ref `52838266`, description `v589: Guarded direct Itshyao S124 S114 plus G124 F1 rankblend replay`, status pending.
- **Cap state:** 2026-05-20 UTC is now capped at `5/5` visible submissions: v585 `0.922`, v586 `0.941`, v587 `0.949`, v588 `0.949`, v589 pending.
- **Next:** wait for v589 score. If it improves above `0.949`, port/confirm S124 G124 rankblend repo-owned. If it ties/drops, stop spending slots on S114+G-sidecar siblings and pivot to new non-saturated source/training directions for the next UTC day.

## 2026-05-20 05:55 UTC / 2026-05-19 PDT — capped score hold + HGNet training lead

- **Status check:** v589 remains pending; v586 is `0.941`, v587 is `0.949`, v588 is `0.949`, v585 is `0.922`. Current confirmed best remains `0.949`. 2026-05-20 UTC is capped at `5/5`; no v577/v578 or v586-v589 submitters/monitors are active. PR #248 remains open/blocked.
- **Fresh capped scan:** saved `scan_20260520T0547Z.json` and `source_audit_20260520T0547Z_top/summary.json` while no more slots are available.
- **Findings:**
  - Qiuzi `hgnetv2-b0-training-e2c7fc` completed as a training artifact kernel with fold weights and validation arrays. Log reports `auc for raw pred : 0.9554015527165799` and `auc for rank pred: 0.9586928494392578`. This is not a direct submission candidate (no `submission.csv`), but it is the best fresh non-EoS training-lane lead and should be investigated for a repo-owned inference/sidecar package after the cap resets.
  - Samejima `birdclef-2026-hgnetv2-b0-baseline-inference` is COMPLETE with `submission.csv`, but it has blank dataset metadata, no score claim, and appears to be a generic HGNet baseline inference. Keep as idea-mining only until source/data dependencies and public/hidden behavior are better understood.
  - Kijiang v340/v341, Yash/Pilkwang prior-field, Nina EoS6, and related forks are saturated EoS/Karnak/PowerOptimization/PriorField variants, not worth displacing new structural work.
  - Samejima Visual CPU is still the weak Visual/BirdNET family; prior direct visual submissions dropped below the plateau.
- **Decision:** no more submissions today. If v589 ties/drops, pivot next reset away from S114+G-sidecar siblings and toward a repo-owned HGNet training/inference sidecar investigation using Qiuzi artifacts or other non-saturated model-zoo diversity sources.

### v589 result — 2026-05-20 05:50 UTC

- v589 (`52838266`, Itshyao S124 S114+G124 F1 rankblend) scored `0.949`, tying the plateau but not improving. Current best remains `0.949`.
- Final 2026-05-20 result set: v585 `0.922`, v586 `0.941`, v587 `0.949`, v588 `0.949`, v589 `0.949`.
- Decision confirmed: stop spending slots on S114+G-sidecar siblings (G116/G123/G124 delta/rankblend) unless a future variant has independent evidence above `0.949`; pivot next reset to new non-saturated sources/training lanes, with Qiuzi HGNet training artifacts as the most interesting fresh lead.

## 2026-05-20 06:47 UTC / 2026-05-19 PDT — capped scan; HGNet lead downgraded to prep-only

- **Status:** latest Kaggle submissions confirmed complete: v585 `0.922`, v586 `0.941`, v587 `0.949`, v588 `0.949`, v589 `0.949`. Current best remains `0.949`; 2026-05-20 UTC slots are `5/5` used. PR #248 remains open and BLOCKED. No v577/v578 or other submitter process is active.
- **Fresh scan:** saved `artifacts/public_kernels_20260520_frontier_candidates/scan_20260520T0647Z.json` and audited top/new leads in `source_audit_20260520T0647Z_top/summary.json`; pulled HGNet details into `hgnet_lead_20260520T0647Z/`.
- **Qiuzi HGNet training lead:** `qiuzilang/hgnetv2-b0-training-e2c7fc` has structurally useful code (HGNetV2-B0, 5s log-mel, EMA, mixup, full train-audio + train-soundscape training) and fold0 validation best `0.9637819061584422`, but the live session status is now `CANCEL_ACKNOWLEDGED`; session output URLs expose only fold0, while file listing names all folds with suspicious tiny metadata sizes. Treat as recipe/source lead, not a ready artifact package.
- **Samejima HGNet inference:** `samejimatink0/birdclef-2026-hgnetv2-b0-baseline-inference` is hidden-path aware and uses 10s/shifted HGNet inference, but its dry-run `submission.csv` is all NaN because sample rows are reindexed after train-soundscape fallback. Also its required weights dataset is not discoverable through dataset search. Treat as code-mining only unless the data source can be resolved and sample-safe output patched.
- **New date-run candidate:** `ommodi07/birdclef2026` is a dual-resolution EfficientNetV2/temporal-transformer training notebook, but the completed output loaded checkpoint epoch 1 AUC `0.5124` and produced all-zero fallback `submission.csv` due no test files. Reject for submission.
- **Other 06:47 search hits:** Aditya Safe EoS5/RankPower and Pilkwang/Yash/Rikuter PriorField are saturated EoS/Karnak/PriorField 0.949-family variants; do not spend slots unless all new-signal paths are blocked and a diagnostic slot would otherwise expire.
- **Decision:** no submission while capped. For next reset, do not submit raw HGNet sources yet. First prepare a repo-owned HGNet recipe/sidecar only if we can obtain complete weights or run controlled training/inference and compare against the 0.949 EoS5 anchor; otherwise continue source frontier scan.

## 2026-05-20 07:47 UTC / 2026-05-20 PDT — post-merge capped scan; Qiuzi distill cancelled, Henry NFNet v80 valid but low-upside

- **Status:** latest submissions unchanged: v585 `0.922`, v586 `0.941`, v587 `0.949`, v588 `0.949`, v589 `0.949`. Current confirmed best remains `0.949`; 2026-05-20 UTC slots remain `5/5` used. No v577/v578 scalar submitter or other active submitter process. PR #248 is now MERGED, so new work moved to branch `feature/birdclef-096-hgnet-nfnet-triage-20260520`.
- **Fresh scan:** saved ignored local scan artifact `artifacts/public_kernels_20260520_frontier_candidates/scan_20260520T0747Z.json`; audited new top-feed/source items in `source_audit_20260520T0747Z_top/summary.json`.
- **Qiuzi `hgnetv2-b0-training-distill`:** new top date-run item. It is training-only, status `CANCEL_ACKNOWLEDGED`, and log tail only reaches epoch 0 (`val_score=0.54157`). No submission artifact. This does not supersede the earlier Qiuzi non-distill recipe lead; treat distill as not useful until a complete run appears.
- **Henry `bc2026-rankpower-nfnet-v80`:** COMPLETE with schema-valid non-empty `submission.csv` and outputs `subm_rankpower06_prior05_consensus.csv`, `submission_nfnet_selective.csv`, ProtoSSM/SED artifacts, and logs showing `[NFNet] final submission.csv safety check passed (3, 235)`. Source says it is an EoS5/Sunderekkiz/Pilkwang rank-power branch with a low-weight YukiZ residual blend (`YUKIZ_BLEND_WEIGHT=0.0264`, `PROTO_RANK_WEIGHT=0.600`, `RANK_BLEND_WEIGHT=0.9736`) and marks the rank-power LB as `0.949`. Hidden-test path/output behavior looks safer than the earlier timed-out NFNet attempt, but it is still largely saturated EoS5/Karnak/Pilkwang family, so it is a **backup next-reset candidate**, not a 0.96-first candidate.
- **Koushik acoustic species notebook:** COMPLETE but no output files; skip for submission/source port.
- **Decision:** while capped, do not submit anything. Next reset priority remains: search for a genuinely new >0.949 source lineage first; if none appears and a slot would otherwise idle, Henry NFNet v80 is the cleanest guarded direct backup, but label it a diagnostic/backup because expected upside is modest.

## 2026-05-20 08:48 UTC / 2026-05-20 PDT — capped scan; v590 Zeyad Proto Temporal backup preflighted

- **Status:** latest submissions unchanged and complete: v585 `0.922`, v586 `0.941`, v587 `0.949`, v588 `0.949`, v589 `0.949`; best remains `0.949`; 2026-05-20 UTC slots remain `5/5`. PR #249 remains open/BLOCKED; no active submitter processes.
- **Fresh scan:** saved ignored local scan `artifacts/public_kernels_20260520_frontier_candidates/scan_20260520T0848Z.json` and audited top-feed items in `source_audit_20260520T0848Z_top/summary.json`.
- **Rejected/low-signal fresh items:**
  - `aiaiaiooo/birdclef2026` COMPLETE but dry-run `submission.csv` is all zeros; reject for slots.
  - `mtoshidesu/notebookc6e90ae327` status ERROR; skip.
  - `mtoshidesu/testbirdclef-2026-s124-s114-g124-f1-rankblend` is a running duplicate of already-submitted/tied S124 lane; skip unless it later proves a distinct >0.949 source.
  - Qiuzi `hgnetv2-b0-training-distill` now reports COMPLETE but has no output files/no submission artifact; still not useful.
  - Claude/Rajnish PriorField/EoS6/Kijiang safe writers are complete and schema-valid except Claude V6 replay has bad 243-row/all-NaN output; all remain saturated 0.949-family replays.
- **Best backup discovered:** `rajnish1419kumar/birdclef-2026-zeyad-proto-temporal-safe` is COMPLETE with finite sample-shaped `submission.csv` plus branch artifacts `submission_birdnet.csv`, `submission_centroid.csv`, `submission_public_cnn.csv`, `submission_protossm.csv`, and `submission_sed.csv`. It is an exp019/Pilkwang 0.948→0.949 Proto+SED path plus optional BirdNET/centroid/public-CNN branches and temporal flip/shift TTA. Public-CNN is zero on the dry run, but BirdNET/centroid branches have independent nonzero signals; branch correlations vs ProtoSSM are low/moderate (BirdNET/Proto ~0.25, Centroid/Proto ~0.31, SED/Proto ~0.45), making it more structurally distinct than additional S114+G siblings.
- **Prepared v590:** added `scripts/submit_v590_rajnish_zeyad_proto_temporal_safe_when_ready.py` with source/version/output/submission-shape preflights and `--preflight-only`. Preflight passed: source v1 length `106762`, kernel COMPLETE/no failure, required outputs present, `submission.csv` stats rows=3 cols=235 finite min=`0.47687027` max=`0.556084` zeros=0. Script does not submit while capped.
- **Decision:** keep v590 as the cleanest next-reset backup/diagnostic if no genuinely new >0.949 source appears. It is not a 0.96-first lane, but it is more diverse than Henry NFNet v80 or more EoS/S114/PriorField replays.

## 2026-05-20 09:48 UTC / 2026-05-20 PDT — capped scan; v590 remains best backup, no new 0.96 lane

- **Status:** latest submissions unchanged: v585 `0.922`, v586 `0.941`, v587/v588/v589 `0.949`; current confirmed best remains `0.949`; 2026-05-20 UTC slots remain `5/5`. PR #249 is open/BLOCKED; no active submitter or v577/v578 process.
- **Fresh scan:** saved ignored local scan `artifacts/public_kernels_20260520_frontier_candidates/scan_20260520T0948Z.json`; audited top/new items in `source_audit_20260520T0948Z_top/summary.json`.
- **New top-feed findings:**
  - `bugraeyidogan/hdmrlib-as-an-interpretable-spectrogram` is COMPLETE but has no output files / no submission; idea-only, no slot.
  - Henry `bc2026-rankpower-nfnet-v81` and `v82` are COMPLETE, schema-valid, and output the same branch set as v80. v81 uses NFNet blend/spike `0.06/0.080`; v82 uses `0.035/0.13`. Both retain `YUKIZ_BLEND_WEIGHT=0.0264`, `PROTO_RANK_WEIGHT=0.600`, and rank-power LB marker `0.949`, so they remain saturated EoS5/Sunderekkiz/Pilkwang-family backups behind v590.
  - Pilkwang PriorField v10 was RUNNING at audit time and remains same PriorField/EoS6 lane; not a new distinct candidate.
  - Nina EoS6 v12 COMPLETE but primary `submission.csv` is invalid for sample submission: 243 rows and all NaN values. Continue rejecting direct EoS6 primary-output submissions unless a safe writer fixes the output and shows evidence beyond 0.949.
  - Mtoshi S124 test COMPLETE with valid 3x235 output but is a duplicate of already-submitted/tied S124/G124 rankblend lane.
  - Original Zeyad `birdclef-2026-proto-fusion-and-temporal-flip` v4 is equivalent to the Rajnish safe writer behind prepared v590: COMPLETE, finite 3x235 output, same BirdNET/centroid/public-CNN/Proto/SED branch set.
- **Decision:** v590 Rajnish/Zeyad Proto Temporal Safe remains the best prepared next-reset backup diagnostic. Do not replace it with Henry v81/v82 or S124/PriorField/EoS6 reruns. Still prioritize true non-saturated 0.95/0.96 source discovery if one appears before reset.

## 2026-05-20 10:48 UTC / 2026-05-20 PDT — capped scan; Pilkwang v11 fixed output but still saturated

- **Status:** latest submissions unchanged; current best remains `0.949`; 2026-05-20 UTC slots are still `5/5`. PR #249 remains open, merge state currently UNKNOWN/BLOCKED-flaky; no active submitter or v577/v578 process.
- **Fresh scan:** saved ignored local scan `artifacts/public_kernels_20260520_frontier_candidates/scan_20260520T1048Z.json`; audited refreshed high-rank items in `source_audit_20260520T1048Z_top/summary.json`.
- **v590 preflight recheck:** `scripts/submit_v590_rajnish_zeyad_proto_temporal_safe_when_ready.py --preflight-only` still passes. Source v1 length `106762`, kernel COMPLETE/no failure, required outputs present, finite `3x235` sample submission, min `0.47687027`, max `0.556084`, zeros `0`; script correctly refuses to submit while the visible UTC count is `5`.
- **Pilkwang PriorField refresh:** `pilkwang/949-birdclef-2026-acoustic-prior-field-fusion` v11 is now COMPLETE with finite sample-shaped `submission.csv` (`3x235`, min `0.460793537150703`, max `0.5381690938702316`) and adds `subm_birdnet_v24.csv`. This fixes output validity vs some prior unsafe wrappers, but it remains the known 0.949 PriorField/EoS6/Karnak lane, not a new 0.96-first source. Keep below v590 in backup order.
- **Nina EoS6:** v12 still invalid primary output (`243x235`, all NaN values). Continue rejecting direct EoS6 primary-output submissions.
- **Other scan items:** no new non-saturated 0.95/0.96 claim; Henry v80/v81/v82 and S124/PriorField/EoS6 remain backup/diagnostic families.
- **Decision:** no candidate queue change. v590 Rajnish/Zeyad Proto Temporal Safe remains first reset backup; Pilkwang v11 becomes a later valid-but-saturated fallback behind v590 and Henry NFNet only if a slot would otherwise idle.

## 2026-05-20 11:48 UTC / 2026-05-20 PDT — capped scan; fresh top-feed mostly invalid/debug outputs

- **Status:** latest submissions unchanged; current best remains `0.949`; 2026-05-20 UTC count remains `5/5`. PR #249 open with flaky UNKNOWN/BLOCKED merge state; no active submitter or v577/v578 process.
- **Fresh scan:** saved ignored local scan `artifacts/public_kernels_20260520_frontier_candidates/scan_20260520T1148Z.json`; audited new top-feed notebooks in `source_audit_20260520T1148Z_top/summary.json`.
- **New top-feed findings:**
  - `haridoss31/birdclef-my-model` v37 was RUNNING at audit time with `native_resnet_fold_0.pth` and `submission.csv` listed but no downloadable submission output yet. Source is a custom EfficientNet/OpenVINO training/inference notebook with target comments around `0.85+` macro ROC-AUC and no high-LB evidence. Monitor only; not slot-ready.
  - `meenalsinha/birdclef-2026-improved` v23 COMPLETE but primary `submission.csv` is `240x235` train/fallback-shaped, not sample-shaped. Reject for direct submit; same visual/BirdNET family has already failed/dropped in previous attempts.
  - `jacqueszhelinzhang/birdclef26-perch-minimal` v21 COMPLETE with valid `3x235` shape but constant probabilities (`min=max=0.0042735`); baseline/debug only.
  - `samejimatink0/birdclef-2026-visual-cpu-inference` v7 COMPLETE but primary output is `240x235` train/fallback-shaped. Reject for direct submit; visual/BirdNET lane remains weak/idea-mining.
  - Qiuzi `hgnetv2-b0-training-distill` v15 was RUNNING with no outputs at audit time. Continue to wait for complete artifacts before considering HGNet distill.
  - `evgendvorkin/birdclef-baseline` v34 COMPLETE but primary output is `240x235` train/fallback-shaped with many zeros; reject.
- **Decision:** no queue change. v590 Rajnish/Zeyad Proto Temporal Safe remains first prepared next-reset backup diagnostic. Continue scanning for true new non-saturated 0.95/0.96 source; do not submit Meenal/Samejima/Visual/BirdNET/debug baseline outputs.

## 2026-05-20 12:48 UTC — v591 Qiuzi HGNet distill sidecar validation

- **Track:** 0.96 frontier source scan + repo-owned high-upside extraction while daily submissions are capped.
- **Live status:** latest 2026-05-20 submissions are capped at `5/5`: v585 `0.922`, v586 `0.941`, v587 `0.949`, v588 `0.949`, v589 `0.949`. Current best remains `0.949`. No stale v577/v578/v590 submitter is active.
- **Source scan artifacts:** `artifacts/public_kernels_20260520_frontier_candidates/scan_20260520T1248Z.json`; `artifacts/public_kernels_20260520_frontier_candidates/source_audit_20260520T1248Z_top/summary.json`.
- **Key finding:** `qiuzilang/hgnetv2-b0-training-distill` completed all four HGNetV2-B0 folds and exposes `best_model_fold0.pt` ... `best_model_fold3.pt`, validation prediction arrays, and `result_df_fold*.csv`. Downloaded fold result CSVs to `artifacts/public_kernels_20260520_frontier_candidates/source_audit_20260520T1248Z_top/qiuzilang_distill_outputs/`.
- **Validation evidence:** fold best val scores are `0.9651087`, `0.9701546`, `0.9669707`, `0.9729050`; final kernel log reports `auc for raw pred : 0.9583789191588714` and `auc for rank pred: 0.9672700848733766`. This is stronger evidence than the previous cancelled/partial HGNet lead and structurally distinct from EoS/RankPower/S114 sidecars.
- **Implementation:** added `kaggle-kernels/v591-public946-hgnet-distill-w0025/` and `scripts/push_v591_public946_hgnet_distill.py`. The kernel forks v542 public946, attaches Qiuzi's training output as a kernel source, runs guarded 4-fold HGNetV2-B0 inference, writes `submission_hgnet.csv`, and blends HGNet at `0.025` rank weight with the existing Proto/SED anchor.
- **Kaggle push:** private kernel `yourslewis/bc26-v591-public946-hgnet-distill-w0025`, version 1, pushed successfully with no invalid sources. Status is RUNNING at log time; no leaderboard submission was attempted because slots are capped.
- **Next gate:** monitor v591. If COMPLETE with finite `submission.csv` and `submission_hgnet.csv`, make it the next-reset candidate ahead of v590. If it fails due mount/dependency/runtime, diagnose and either patch v591 or fall back to v590 Zeyad/Rajnish.

## 2026-05-20 13:48 UTC — v592 HGNet 10% sidecar prepared

- **Track:** repo-owned high-upside extraction from Qiuzi HGNet distill artifacts while daily slots are capped.
- **Live status:** best remains `0.949`; 2026-05-20 submissions are capped at `5/5` with v585 `0.922`, v586 `0.941`, v587/v588/v589 `0.949`; no v577/v578/v590 submitter active.
- **v591 failure/fix chain:** v1 failed on Kaggle CUDA during torchaudio STFT; v2 kept preprocessing on CPU but failed on HGNet CUDA ReLU; v3 completed CPU-only but produced constant HGNet predictions because row-id parsing was wrong and dry-run audio missing silently zero-filled; v4 fixed row-id parsing to use the final row-id token as end second and removed the zero fallback.
- **v591 v4 validation:** private kernel `yourslewis/bc26-v591-public946-hgnet-distill-w0025` version 4 COMPLETE/no failure, runtime ~524s. Outputs downloaded under `artifacts/kaggle_outputs/v591-public946-hgnet-distill-w0025-v4/`. Final `submission.csv` is `240x235`, no NaNs, min/max `0.0053125/1.0`; `submission_hgnet.csv` is nonconstant and valid, min/max `3.390359e-07/0.9477211`.
- **Local sidecar grid:** `artifacts/blend_grids/v591_hgnet_sidecar_weight_grid_20260520T1348Z_v4.json`. HGNet standalone rank sidecar: macro AUC `0.9956425`, corr vs public946 anchor `0.4808`. Weight grid improved dry-run overlap from base `0.9925249` to `0.9927187` at `0.025` and `0.9932913` at `0.10`; top3 row recall improved from `0.5211` base to `0.6211` at `0.10`.
- **Decision/implementation:** added `kaggle-kernels/v592-public946-hgnet-distill-w010/`, `scripts/push_v592_public946_hgnet_distill_w010.py`, and `scripts/submit_v592_public946_hgnet_w010_when_ready.py`. v592 uses the same guarded HGNet CPU-only inference but sets `HGNET_RANK_WEIGHT=0.10` (`Proto=0.54`, `SED=0.36`, `HGNet=0.10`).
- **v592 validation:** pushed private Kaggle kernel `yourslewis/bc26-v592-public946-hgnet-distill-w010` version 1; push returned no invalid sources. Kernel COMPLETE/no failure, runtime ~562s. Outputs downloaded under `artifacts/kaggle_outputs/v592-public946-hgnet-distill-w010/`. Final `submission.csv` is `240x235`, no NaNs, min/max `0.0065000006/1.0`; `submission_hgnet.csv` valid, min/max `3.390359e-07/0.9477211`.
- **Submitter preflight:** `python3 scripts/submit_v592_public946_hgnet_w010_when_ready.py --preflight-only` passed: source v1, COMPLETE/no failure, required outputs, final CSV stats `240x235`, finite nonconstant. It did not submit because visible UTC submissions today are `5`.
- **Next gate:** v592 is first next-reset candidate; v591/v4 is conservative fallback; v590 Zeyad/Rajnish is now backup only if HGNet is invalidated or Wenhao prefers the lower-risk saturated lane.

## 2026-05-20 14:48 UTC — v592 reset submitter parked after fresh source audit

- **Live status:** best remains `0.949`; 2026-05-20 remains capped `5/5`; latest scores unchanged (`v585=0.922`, `v586=0.941`, `v587/v588/v589=0.949`). No stale v577/v578/v590/v591/v592 submitter was active at run start.
- **Fresh scan/audit:** saved `artifacts/public_kernels_20260520_frontier_candidates/scan_20260520T1448Z.json` and `artifacts/public_kernels_20260520_frontier_candidates/source_audit_20260520T1448Z_top/summary.json`.
- **Candidate decisions:** Mtoshi v8 remains Karnak/PowerOptimization family; Jacques minimal is constant baseline; Samejima HGNet/OpenVINO artifacts are interesting but still RUNNING/no downloadable outputs at audit time; Sclim iter7 ProtoSSM+MLP is schema-safe but 0.949-family; Abhiiiish is training-artifact only; THBDH S124/G124 fork is duplicate of the tied v589 lane.
- **Decision:** no stronger 0.96-relevant source displaced v592. v592 remains first reset-slot candidate.
- **Action:** started a reset submitter: PID `13173`, log `logs/v592_hgnet_w010_reset_submitter_20260520.log`, nohup log `logs/v592_hgnet_w010_reset_nohup_20260520.out`. It sleeps until about `2026-05-21T00:05:00Z`, then runs `scripts/submit_v592_public946_hgnet_w010_when_ready.py`; the script revalidates source/version, kernel COMPLETE/no failure, required outputs, CSV stats, duplicate description, and daily cap before submitting.
- **Next gate:** if v592 submits and improves, immediately preserve/confirm the HGNet path and consider OpenVINO acceleration via Samejima artifacts. If it ties/drops, keep HGNet as a structural diagnostic but continue frontier scan before using further slots.

## 2026-05-20 15:48 UTC — capped source scan; v592 still first reset candidate

- **Live status:** best remains `0.949`; 2026-05-20 daily cap remains `5/5`; v585 `0.922`, v586 `0.941`, v587/v588/v589 `0.949`.
- **Submitter health:** v592 reset submitter PID `13173` is alive and sleeping toward ~`2026-05-21T00:05:00Z`. No duplicate v592 submitter started.
- **Fresh scan/audit:** saved `artifacts/public_kernels_20260520_frontier_candidates/scan_20260520T1548Z.json` and `artifacts/public_kernels_20260520_frontier_candidates/source_audit_20260520T1548Z_top/summary.json`.
- **Candidate decisions:** Samejima HGNet training v25 exposes OpenVINO artifacts but is ERROR; Samejima HGNet inference v3 outputs all-NaN primary `submission.csv`; Deepanshu is constant baseline; Nina EoS6 v15 primary remains invalid/all-NaN; Mtoshi v9 and Scenery Model_7 are saturated Karnak/PowerOptimization; Anthony S124/G124 is a duplicate of tied v589 family.
- **Decision:** keep v592 as the reset-slot owner. No new submission or kernel push. Samejima OpenVINO is logged as a possible future acceleration path if v592 hidden runtime is problematic or if HGNet improves.
- **Transfer lesson maintained:** local train-soundscape sidecar gates are rejection filters, not approval filters. The v592 exception is justified by complete, structurally distinct HGNet artifacts and low-correlation sidecar evidence, not just a tiny local gain.

## 2026-05-20 16:48 UTC — capped re-scan; no source displaces v592

- **Live status:** best remains `0.949`; 2026-05-20 daily cap remains `5/5`; latest scored rows unchanged (`v585=0.922`, `v586=0.941`, `v587/v588/v589=0.949`).
- **Submitter health:** v592 reset submitter PID `13173` is still alive, sleeping toward ~`2026-05-21T00:05:00Z`; no duplicate submitter started.
- **Fresh scan:** saved `artifacts/public_kernels_20260520_frontier_candidates/scan_20260520T1648Z.json`.
- **Candidate decisions:** the newest DATE_RUN feed did not reveal a stronger candidate than v592. Samejima HGNet remains an OpenVINO/artifact lead but not a validated submission; current inference primary output was invalid in the prior audit. EoS6/Karnak/RankPower/S124-family reruns are saturated or invalid; constant/baseline notebooks remain rejected.
- **Decision:** preserve v592 as the only reset-slot owner. No new Kaggle push/submission. Continue scanning until reset; if no stronger source appears, allow the parked submitter to fire.

## 2026-05-20 17:48 UTC — two-pass SSM audited, v592 still owner

- **Live status:** best remains `0.949`; 2026-05-20 daily cap remains `5/5`; v592 reset submitter PID `13173` remains alive and sleeping toward reset.
- **Fresh scan/audit:** saved `artifacts/public_kernels_20260520_frontier_candidates/scan_20260520T1748Z.json` and `artifacts/public_kernels_20260520_frontier_candidates/source_audit_20260520T1748Z_top/summary.json`.
- **Candidate audit:** Tulay EfficientNet weights ERROR/no outputs; Scenery EoS6 invalid primary; Deepanshu constant baseline; Karnak S124/G124 reverse-engineered no outputs; Kospintr EfficientNet+Perch Distill+MixUp cancelled/partial; Scenery Perch V2 Full invalid `119988x235` output with `-1000` values.
- **Two-pass SSM gate:** `scenerysunfireink/birdclef-2026-two-pass-ssm` is schema-safe (`240x235`) and distinct-sounding, so ran local sidecar gate. Artifact: `artifacts/blend_grids/scenery_two_pass_sidecar_weight_grid_20260520T1748Z.json`. Standalone rank AUC `0.97745`, corr vs v542 anchor `0.8884`; every blend weight reduced macro AUC vs base `0.9925249` (0.025 `0.9924122`, 0.05 `0.9922645`, 0.10 `0.9917289`). Do not let it displace v592.
- **Decision:** keep v592 as sole reset-slot owner; no new push/submission; no duplicate submitter.

## 2026-05-20 17:54 UTC — post-audit verification, still capped

- Rechecked Kaggle submissions: v585 `0.922`, v586 `0.941`, v587/v588/v589 `0.949`; current best remains `0.949`; 2026-05-20 UTC remains capped at `5/5`.
- Rechecked repo/PR/process state: branch clean before this log update; PR #249 open/BLOCKED; v592 reset submitter PID `13173` remains the only visible v59x submitter and is sleeping toward reset.
- Ran an additional lightweight fresh source query across `0.95`, `0.96`, EoS6, SafeAlign, RankPower/NFNet, HGNet, and two-pass SSM terms. It returned the same already-audited Scenery two-pass/EoS6, Nina EoS6, S124/G124, RankPower/NFNet, Samejima/TY0912 HGNet, and saturated EoS/SafeAlign family; no new credible slot displacer.
- Decision unchanged: preserve v592 as sole reset-slot owner; no new submission while capped.

## 2026-05-20 18:48 UTC — capped source scan, Tulay EfficientNet rejected

- **Live status:** best `0.949`; 2026-05-20 submissions remain v585 `0.922`, v586 `0.941`, v587/v588/v589 `0.949`; cap `5/5`.
- **Process/PR:** v592 reset submitter PID `13173` alive and sleeping toward reset; no v577/v578 scalar submitter and no duplicate v59x submitter visible. PR #249 open; merge-state fetch returned `UNKNOWN`.
- **Scan artifact:** `artifacts/public_kernels_20260520_frontier_candidates/scan_20260520T1848Z.json`.
- **New audit artifact:** `artifacts/public_kernels_20260520_frontier_candidates/source_audit_20260520T1848Z_new/summary.json`.
- **Tulay EfficientNet:** `tulayppppp/my-efficientnet-b0-weights` reran at 18:45Z, but audit shows v23 RUNNING/no outputs and source is unsafe/non-competitive: generic dynamic `bird_i` columns, dummy/fallback rows, potential empty `submission.csv`, and generic EfficientNet baseline load. Reject as reset-slot displacer.
- **Other scan rows:** Scenery/Nina EoS6/two-pass, S124/G124, RankPower/NFNet, Samejima HGNet, SafeAlign, and acoustic-prior rows are previously audited/saturated; no new credible source above v592.
- **Decision:** preserve v592 as only reset-slot owner.

## 2026-05-20 19:04 UTC — user lead Itshyao S124/S114/G124 rankblend audited

- User flagged `itshyao/birdclef-2026-s124-s114-g124-f1-rankblend` as new.
- Pulled current source via Kaggle API and saved under `artifacts/public_kernels_20260520_frontier_candidates/itshyao_s124_s114_g124_rankblend_latest/`.
- Current metadata says version `2`, but decoded source exactly matches the artifact audited before v589 (`6819` decoded lines, SHA `c5aed8358ce6ba4b8772c1649bed9475151adff011d07617a8ba2b6f223a62f9`).
- Session output is COMPLETE/no failure; primary `submission.csv` is finite `3x235` dry-run/sample shape, min/max `0.47687027/0.5553993`, no NaNs/zeros. Log notes `S124 dry-run/mismatch: keeping S114 anchor submission.csv`.
- Existing v589 guarded direct replay of this source scored `0.949`, tying but not improving. Decision: no resubmission; v592 remains sole reset-slot owner.

## 2026-05-20 19:08 UTC — v593 Itshyao v2 0.952 lead takes reset slot

- User clarified/reported score `0.952` for `itshyao/birdclef-2026-s124-s114-g124-f1-rankblend`.
- Important correction: v589 submitted kernel version `1` and scored `0.949`; current Kaggle metadata is version `2`. Although decoded source matches previous source, the reported `0.952` is independent evidence and justifies one guarded v2 replay.
- Added `scripts/submit_v593_itshyao_s124_g124_rankblend_v2_when_ready.py` pinned to version `2` with duplicate, source, status, output, and cap checks.
- Preflight-only result: cap `5/5`, source version `2` OK, COMPLETE/no failure, required outputs present, `submission.csv` finite `3x235` min/max `0.47687027/0.5553993`, no NaNs/zeros.
- Killed v592 reset submitter PID `13173`; started v593 reset submitter PID `96527` with log `logs/v593_itshyao_s124_g124_v2_reset_submitter_20260520.log`, sleeping toward reset.
- Queue: v593 first; v592 HGNet sidecar demoted to backup if v593 does not improve.

## 2026-05-20 20:00 UTC — capped scan, v593 healthy

- **Live status:** v585 `0.922`, v586 `0.941`, v587/v588/v589 `0.949`; repo-confirmed best remains `0.949`; daily cap `5/5`.
- **Reset owner:** v593 Itshyao S124/S114/G124 v2 remains alive as PID `96527`, sleeping toward reset; v592 PID `13173` is stopped. No v577/v578 scalar submitter visible.
- **Scan artifact:** `artifacts/public_kernels_20260520_frontier_candidates/scan_20260520T2000Z.json`.
- **Audit artifact:** `artifacts/public_kernels_20260520_frontier_candidates/source_audit_20260520T2000Z_top/summary.json`.
- **New lead audit:** `karnakbaevarthur/s124-g124-reverse-engineered` explicitly cites Itshyao S124/G124 LB `0.952`, but has no output files; code-mining only. `haivan11/birdclef-2026-prior-field-fusion-vi` is output-valid but saturated PriorField/Yaroslav/BirdNET family; not above v593. Tulay EfficientNet v27 ERROR/no outputs and still unsafe.
- **Decision:** keep v593 first; v592 HGNet backup.

## 2026-05-20 22:00 UTC — capped scan, v593 still healthy

- **Live status:** v585 `0.922`, v586 `0.941`, v587/v588/v589 `0.949`; repo-confirmed best remains `0.949`; 2026-05-20 cap `5/5`; no 2026-05-21 submissions visible yet.
- **Reset owner:** v593 Itshyao S124/S114/G124 v2 PID `96527` still alive and sleeping toward reset; v592 PID `13173` stopped; no stale v577/v578 scalar submitter visible.
- **Scan artifact:** `artifacts/public_kernels_20260520_frontier_candidates/scan_20260520T2200Z.json`.
- **Audit artifact:** `artifacts/public_kernels_20260520_frontier_candidates/source_audit_20260520T2200Z_top/summary.json`.
- **Fresh audit:** Muhammad WildSound v8 is ERROR/no outputs due missing train metadata path; Juanpp segunda parte is ERROR/no outputs and local/offline-weight style; Tulay EfficientNet v28 remains ERROR/no outputs and unsafe/generic. None displaces v593.
- **Decision:** keep v593 first; v592 HGNet backup.

## 2026-05-21 00:01 UTC — v593 submitted and pending

- **Reset status:** 2026-05-21 initially `0/5`; 2026-05-20 final remained v585 `0.922`, v586 `0.941`, v587/v588/v589 `0.949`.
- **Wrapper issue:** parked v593 PID `96527` failed before submission due system Python using an incompatible Kaggle SDK (`KaggleHttpClient.__init__() got an unexpected keyword argument 'api_token'`). No submission was created by the wrapper.
- **Recovery:** reran guarded v593 with venv Python. Checks passed: source v2, COMPLETE/no failure, required outputs present, finite `3x235` submission. Submitted ref `52866246`; visible as `pending` at `2026-05-21T00:06:26.767Z`; count now `1/5`.
- **Scan/audit:** saved `scan_20260521T0001Z.json` and `source_audit_20260521T0001Z_top/summary.json`. Rauffauzan S124 fork fails G124 sidecar (`S124 G124 fold1 rank sidecar failed`) and is not above v593; Mtoshi v11 is schema-valid but Karnak/0.948-family only.
- **Decision:** hold remaining four slots until v593 score lands or a clearly stronger fresh source appears.

## 2026-05-21 02:00 UTC — v593 tied, v594 HGNet submitted

- **Score update:** v593 ref `52866246` scored `0.949`, tying but not improving. The external `0.952` Itshyao v2 lead did not reproduce via our direct replay.
- **Fallback action:** attempted the repo-owned HGNet 10% sidecar (v592 path). Preflight passed, but Kaggle rejected the version-1 kernel submission because metadata had GPU enabled (`GPU max of 0 minutes`).
- **Fix:** patched `kaggle-kernels/v592-public946-hgnet-distill-w010/kernel-metadata.json` to `enable_gpu=false`, actual slug `yourslewis/bc26-v592-public946-hgnet-distill-w010`, `id_no=119970462`; repushed as kernel v2. Added `scripts/submit_v594_public946_hgnet_w010_cpu_when_ready.py` with `KERNEL_VERSION=2` and v594 description.
- **v594:** CPU run COMPLETE; preflight valid (`240x235`, no NaNs, min/max `0.0065000006/1.0`); submitted ref `52869105`, status pending at `2026-05-21T02:14:22Z`. 2026-05-21 count `2/5`.
- **Scan:** `scan_20260521T0200Z.json` showed no stronger source than v594; mostly pulled/forked saturated S124/EoS/PriorField or already-rejected error/no-output lanes.
- **Decision:** hold remaining three slots pending v594 score.

## 2026-05-21 04:01 UTC — v594 RAM fail; v595 submitted

- **v594 result:** no public score. Kaggle hidden run failed with `Your notebook requested more memory (RAM) than is available.` Lesson: HGNet 10% sidecar must be memory-refactored/streamed before another slot; public dry-run success was insufficient.
- **Frontier scan:** wrote `scan_20260521T0401Z.json`; focused fresh audit wrote `source_audit_20260521T0410Z_fresh/summary.json`.
- **Candidate decisions:**
  - Samejima HGNet inference: COMPLETE but invalid dry-run `submission.csv` (702 NaNs/bad values), skip.
  - Cheny exp071 BirdNET-unmapped: ERROR/no final `submission.csv`, skip.
  - Kijiang EoS forks: malformed/NaN outputs or saturated EoS scalar variants, skip.
  - Karnak S124/G124 reverse-engineered: no submission output yet, keep as training/porting research lane.
  - Cheny exp070 public0952: source-safe and output-valid; distinct Perch embedding + site/hour prior + classwise logistic-probe structure.
- **v595:** added `scripts/submit_v595_cheny_public0952_perch_prior_probe.py`; preflight passed source version `1`, COMPLETE/no failure, outputs `submission.csv` + `perch_cache/full_oof_meta_features.npz`, stats `240x235`, no bad values, min/max `9.252071e-12/1.0`; submitted ref `52871700`, pending. UTC count now `3/5`.
- **Decision:** hold last two slots until v595 score or a stronger verified candidate appears.

## 2026-05-21 05:55 UTC — v595 scored 0.899; >0.949 notes reviewed

- v595 ref `52871700` scored `0.899`, a major drop. The Cheny exp070 `public0952`/Perch-prior-probe source is therefore not a viable direct replay despite source/output safety.
- Review of >0.949 notes: only the user-reported Itshyao `0.952` and derivative references to that same S124/G124 lead were found as explicit public-LB claims above `0.949`; our v593 direct v2 replay reproduced only `0.949`. Cheny `public0952` did not reproduce (`0.899`). Karnak `s124-g124-reverse-engineered` cites the Itshyao 0.952 but has no submission artifact; use only for code-mining/training-port ideas.
- Working hypothesis for Itshyao 0.952 non-repro: exact decoded v1/v2 source matched and public dry-run log kept the S114 anchor due row mismatch; if the public post actually showed 0.952, the lift likely depended on a private/non-current artifact/version, author-side submission state, leaderboard/reporting mismatch, or non-replayable attached asset/version rather than visible source changes.

## 2026-05-21 18:00 UTC — capped scan; v598 Samejima HGNet artifact port prepared

- **Live state:** best remains `0.949`; 2026-05-21 UTC cap remains `5/5` with v593 `0.949`, v594 RAM/no-score, v595 `0.899`, v596 `0.946`, v597 `0.949`. PR #250 is open/BLOCKED; no stale v577/v578 scalar submitter visible.
- **Fresh scan artifact:** `artifacts/public_kernels_20260521_frontier_candidates/scan_20260521T1800Z.json`.
- **Fresh audit artifacts:** `artifacts/public_kernels_20260521_frontier_candidates/source_audit_20260521T1800Z_fresh/summary.json`, `samejima_hgnet_v41_files_1800.json`, and `samejima_v41_artifact_probe/summary.json`.
- **Fresh feed decisions:**
  - `claudedevore/birdclef-2026-r0952-run2-sidecar-submit` is COMPLETE/schema-valid (`3x235`) but source is a Model2/Model5 EoS/Karnak sidecar/blend family, not enough to justify first reset slot after v574-v576/v597 already tied `0.949`.
  - `vicmcorrea/birdclef-2026-v6-prior-field` is COMPLETE/schema-valid but logs BirdNET unavailable and falls back to the PriorField/Visual/BirdNET family; idea-mining only after v584/v595 drops.
  - `deepanshus167` and similar generic notebooks remain no-output/non-submit-ready.
- **Samejima artifact mining:** Samejima HGNet training v41 errored only in final metric calculation (`continuous-multioutput format is not supported`) but produced real fold artifacts. Downloaded/inspected `best_model_fold0-3.pt` (about 34.6 MB each), val prediction arrays, and result CSVs. Fold best validation scores are `0.9646`, `0.9659`, `0.9684`, `0.9707` (mean best `0.9674`), making it a stronger structural lead than repeated public946 micro-sidecars.
- **v598 implementation:** created branch `feature/birdclef-v598-samejima-hgnet-20260521`; added private repo-owned notebook kernel `kaggle-kernels/v598-samejima-hgnet-openvino-artifact/` plus push/submitter scripts. The kernel is built from Samejima inference source, attaches `samejimatink0/birdclef-2026-hgnetv2-b0-baseline-training` as a kernel source, fixes the public dry-run row-id/NaN merge path, and uses OpenVINO `best_model_fold*_256x512.xml` artifacts when present.
- **Kaggle push/preflight:** pushed `yourslewis/bc26-v598-samejima-hgnet-openvino-artifact` v1, kernel id `120124341`; run COMPLETE with `submission.csv` only. Dry-run output valid `120x235`, no bad values, min/max `0.00097276596/0.86856884`; log shows OpenVINO fold inference completed (`~2.7s/fold` for 120 segments) and no merge NaNs.
- **Submission scheduling:** started guarded wait-for-slot monitor PID recorded in `logs/v598_samejima_hgnet_submit_20260522.pid`; log `logs/v598_samejima_hgnet_submit_20260522.log`. It completed source/status/output preflight, saw 2026-05-21 cap `5/5`, and is sleeping until just after 2026-05-22 UTC reset. It will submit at most one v598 code submission with description `v598: Repo-owned Samejima HGNet OpenVINO artifact inference`, unless already visible or preflight fails.


## 2026-05-21 20:00 UTC — capped scan; v598 monitor remains reset owner

- **Live state:** best remains `0.949`; 2026-05-21 UTC cap remains `5/5` with v593 `0.949`, v594 RAM/no-score, v595 `0.899`, v596 `0.946`, v597 `0.949`. PR #251 (v598) and PR #250 (v596) are open/BLOCKED. No v577/v578 scalar submitter visible.
- **v598 monitor:** PID in `logs/v598_samejima_hgnet_submit_20260522.pid` is alive and still sleeping after successful source/status/output preflight; log `logs/v598_samejima_hgnet_submit_20260522.log` shows cap `5/5` and sleep until after 2026-05-22 UTC reset.
- **Fresh scan artifact:** `artifacts/public_kernels_20260521_frontier_candidates/scan_20260521T2000Z.json`.
- **Fresh focused audit:** `artifacts/public_kernels_20260521_frontier_candidates/source_audit_20260521T2000Z_fresh/summary.json`.
- **New/changed feed:** only `deepanshus167/bird-claasifier-comp` reran after the 18UTC scan (`2026-05-21T18:11Z`). Audit showed it is COMPLETE but still has no output files/submission and is an exploratory training/visualization notebook, not a competition-submit candidate.
- **External web check:** search for fresh `0.960/0.96` BirdCLEF Kaggle notebook claims returned no results.
- **Decision:** keep v598 Samejima HGNet OpenVINO artifact as the first reset-slot owner. Do not start duplicate submitters or spend a slot on repeated EoS/Karnak/R0952/PriorField/Visual/NFNet clones while v598 is queued.


## 2026-05-21 22:00 UTC — capped scan; WildSound rerun not slot-ready

- **Live state:** best remains `0.949`; 2026-05-21 UTC cap remains `5/5` with v593 `0.949`, v594 RAM/no-score, v595 `0.899`, v596 `0.946`, v597 `0.949`. PR #251 (v598) is open/BLOCKED. No v577/v578 scalar submitter visible.
- **v598 monitor:** PID in `logs/v598_samejima_hgnet_submit_20260522.pid` remains alive and sleeping after successful source/status/output preflight; it is still the first reset-slot owner.
- **Fresh scan artifact:** `artifacts/public_kernels_20260521_frontier_candidates/scan_20260521T2200Z.json`.
- **Fresh focused audit:** `artifacts/public_kernels_20260521_frontier_candidates/source_audit_20260521T2200Z_fresh/summary.json`.
- **New/changed feed:** `muhammadsaadalvi/birdclef-2026-wildsound-v8` reran at `2026-05-21T21:57Z` and was audited because it is the only fresh top-feed change. Current status is RUNNING with no output files/no `submission.csv`. Source is a full training pipeline using external BirdCLEF/Xeno-Canto-style data and Google BVC model source; it is potentially interesting as a training/data-diversity lane but not hidden-safe or slot-ready until it completes with a valid competition-format output.
- **External web check:** searches for fresh public `0.960` and `0.952` BirdCLEF Kaggle notebook claims returned no results.
- **Decision:** keep v598 Samejima HGNet OpenVINO artifact as next reset-slot owner. Watch WildSound only after it completes; do not displace v598 with a running/no-output training notebook.


## 2026-05-22 00:05 UTC — reset opened; v598 submitted and pending

- **Live state after UTC reset:** 2026-05-22 count is `1/5`; v598 was submitted by the guarded wait-for-slot monitor at `2026-05-22T00:03:02Z`, ref `52905096`, status `pending`. Current confirmed best remains `0.949` until v598 scores.
- **Monitor result:** `logs/v598_samejima_hgnet_submit_20260522.log` shows the monitor woke after reset, observed `visible UTC submissions today: 0`, reran source/status/output preflight, and submitted `yourslewis/bc26-v598-samejima-hgnet-openvino-artifact` v1 with description `v598: Repo-owned Samejima HGNet OpenVINO artifact inference`. No duplicate submitter remains visible.
- **Fresh scan artifact:** `artifacts/public_kernels_20260522_frontier_candidates/scan_20260522T0000Z.json`.
- **Fresh focused audit:** `artifacts/public_kernels_20260522_frontier_candidates/source_audit_20260522T0000Z_fresh/summary.json`.
- **Reset scan decisions:**
  - `aiaiaiooo/birdclef2026` reran and is COMPLETE with `submission.csv`, but dry-run output is all-zero (`3x235`, min=max=0, 702 zeros); reject.
  - `scenerysunfireink/eos-6-v7-power-0-90-extreme` is COMPLETE with outputs, but `submission.csv` is malformed/invalid (`243x235`, 56862 bad/NaN values). It is also a saturated EoS6/Karnak/Power blend family; reject direct submit.
  - `muhammadsaadalvi/birdclef-2026-wildsound-v8` moved from RUNNING to ERROR with no outputs; keep only as future training/data-diversity idea if repaired.
- **Decision:** hold remaining 4 slots while v598 is pending; no second reset-slot candidate is currently source/output-safe and distinct enough to spend immediately.


## 2026-05-22 00:30 UTC — v598 scored 0.860; standalone Samejima HGNet rejected

- **v598 result:** ref `52905096` completed with public score `0.860`, far below current best `0.949`.
- **Lesson:** the Samejima HGNet OpenVINO artifact is hidden-test format-safe but not leaderboard-competitive as a standalone submission. Strong local/training fold metrics (`~0.967` validation) did not transfer to the competition public LB, likely due to objective/domain/task mismatch rather than output-format failure.
- **Decision:** do not spend another slot on standalone Samejima/HGNet artifact submissions. If HGNet is revisited, require a guarded, tiny-weight anchored blend with stronger public/offline evidence and preferably class/order/correlation diagnostics first.
- **Current slot state:** 2026-05-22 UTC count is `1/5`; 4 slots remain, but no currently audited reset-feed candidate is source/output-safe and high-upside enough for immediate submission.


## 2026-05-22 00:31 UTC heartbeat — post-v598 source scan, no second slot yet

- **Live state:** v598 scored `0.860`, best remains `0.949`; 2026-05-22 UTC count is `1/5`, leaving 4 slots. PR #252 remains open for reset logging. No duplicate/stale submitter visible.
- **Heartbeat scan artifact:** `artifacts/public_kernels_20260522_frontier_candidates/heartbeat_scan_20260522T0030Z.json`.
- **Focused audit artifact:** `artifacts/public_kernels_20260522_frontier_candidates/source_audit_20260522T0030Z_heartbeat/summary.json`.
- **New Gendaijin day0522 candidates:**
  - `gendaijin/birdclef2026-day0522-nina-eos6-bz`: COMPLETE but `submission.csv` is malformed/invalid (`243x235`, 56862 bad/NaN values). Reject.
  - `gendaijin/birdclef2026-day0522-anthony-s124`: COMPLETE and valid dry-run `3x235`, but it is Anthony/S124/Sunderek/Karnak Model2+Model5 blend-family already saturated by v593/v597 at `0.949`. Hold; not a 0.96 reset-slot owner.
  - `gendaijin/birdclef2026-day0522-pilkwang-new`: COMPLETE and valid dry-run `3x235`; source is Acoustic Prior-Field Fusion with low-weight yukiZ branch + dominant v6 prior-field + optional BirdNET sidecar. Dry-run log says `BirdNET row_id mismatch; keeping anchor submission`, so hidden behavior may differ, but it remains PriorField/BirdNET/Pilkwang lineage already broadly saturated/weak in recent replays. Hold for now; do not spend a slot before stronger evidence.
- **Decision:** after v598 failure, continue scanning rather than immediately spending remaining slots. Next slot should wait for either a genuinely new source-safe candidate or a better grounded repo-owned extraction.


## 2026-05-22 02:00 UTC — post-v598 scan; no second slot spent

- **Live state:** v598 scored `0.860`; best remains `0.949`; 2026-05-22 UTC count remains `1/5` with 4 slots unused. PR #252 remains open. No stale v577/v578 scalar submitter or duplicate v598 submitter visible.
- **Fresh scan artifact:** `artifacts/public_kernels_20260522_frontier_candidates/scan_20260522T0200Z.json`.
- **Fresh focused audit:** `artifacts/public_kernels_20260522_frontier_candidates/source_audit_20260522T0200Z_fresh/summary.json`.
- **Audit decisions:**
  - `koushikkumardinda/birdclef-2026-acoustic-species-identification`: COMPLETE but no output files/no `submission.csv`; educational/training notebook only, not submit-ready.
  - `meenalsinha/birdclef-2026-improved` v25: COMPLETE/schema-valid (`240x235`) and high-vote, but it is still the Visual/BirdNET/Prior/Karnak family; previous Visual/BirdNET replay v584 scored `0.942`, so no immediate slot.
  - `samejimatink0/birdclef-2026-hgnetv2-b0-baseline-training` v43: RUNNING with the same artifact family; after standalone v598 scored `0.860`, do not pursue standalone HGNet. Watch only for evidence useful to anchored diagnostics.
  - `jguevarag/07-optimal-sed-training`: COMPLETE but no outputs/submission; training idea only.
  - `gendaijin/birdclef2026-day0522-meenal-new`: COMPLETE/schema-valid (`240x235`) but effectively same Meenal v6_prior065 / Visual-BirdNET-prior family; hold.
- **Decision:** preserve remaining slots. None of the fresh candidates is both distinct and source/output-safe enough to justify a second 2026-05-22 submission under the 0.96 target.


## 2026-05-22 04:01 UTC — post-v598 scan, reject constant/invalid fresh outputs

- **Live state:** v598 scored `0.860`; best remains `0.949`; 2026-05-22 UTC count remains `1/5` with 4 slots unused. PR #252 is open/BLOCKED. No stale v577/v578 scalar submitter or duplicate submission process visible.
- **Fresh scan artifact:** `artifacts/public_kernels_20260522_frontier_candidates/scan_20260522T0401Z.json`.
- **Fresh focused audit:** `artifacts/public_kernels_20260522_frontier_candidates/source_audit_20260522T0401Z_fresh/summary.json`.
- **Audit decisions:**
  - `samejimatink0/birdclef-2026-hgnetv2-b0-baseline-inference` v7: COMPLETE but output invalid (`3x235`, 702 bad/NaN values). Reject; standalone Samejima/HGNet already failed via v598 `0.860`.
  - `koushikkumardinda/birdclef-2026-acoustic-species-identification` v8: COMPLETE with `submission.csv`, but output is constant `1/234` (`3x235`, min=max `0.0042735`, `uniq100=1`); reject as fallback/baseline output.
  - `mlclsumit/notebook2e815ef354`: no Kaggle session/status/output; generic training notebook, not submit-ready.
  - `jguevarag/07-optimal-sed-training` v7: RUNNING with no outputs; training idea only.
  - Recent Meenal/Gendaijin visual/BirdNET/prior variants remain valid but saturated families after v584/v595/v598 failures; not worth immediate slot.
- **External web check:** no fresh public `0.960/0.96/0.952/0.951/0.950` Kaggle notebook claims found.
- **Decision:** preserve all remaining slots. No second 2026-05-22 submission should be made from the current queue.


## 2026-05-22 05:22 UTC — user-approved broad promising batch submitted

- **User direction:** Wenhao asked to test all the most promising remaining experiments. With v598 failed and 4 daily slots still open, I loosened the prior hold rule while preserving output/source preflight guards.
- **Live state before batch:** best remained `0.949`; 2026-05-22 count was `1/5` from v598 `0.860`; no duplicate submitter process visible.
- **Batch submitter:** added `scripts/submit_v599_v602_promising_public_batch.py`. It pulls each public kernel via Bearer API, requires COMPLETE status and finite/non-constant `submission.csv`, duplicate-guards descriptions, and submits until daily cap.
- **Submitted candidates:**
  - v599 ref `52913376`: `claudedevore/birdclef-2026-r0952-run2-sidecar-submit` v1, valid dry-run `3x235`, min/max `0.47687027/0.5553993`, `uniq_first100=91`.
  - v600 ref `52913377`: `gendaijin/birdclef2026-day0522-pilkwang-new` v1, valid dry-run `3x235`, min/max `0.460793537150703/0.5381690938702316`, `uniq_first100=96`.
  - v601 ref `52913379`: `gendaijin/birdclef2026-day0522-meenal-new` v1, valid dry-run `240x235`, min/max `0.0037499997/1.0`, `uniq_first100=94`.
  - v602 ref `52913380`: `nicolasschuldt/nfnet-aves-lprior075` v2, valid dry-run `3x235`, min/max `0.4642808950427329/0.5409460535973569`, `uniq_first100=95`.
- **Slot state after batch:** 2026-05-22 count is now `5/5`; v599-v602 are all pending immediately after submit.
- **Decision:** wait for v599-v602 scores. If one improves/ties high, port/confirm; if all drop, treat the public 0.95-ish visual/prior/NFNet/R0952 families as saturated or misleading under the 0.96 target.


## 2026-05-22 06:01 UTC — broad batch pending, fresh scan no new slot path

- **Live state:** v599-v602 are still pending; v598 remains `0.860`; best confirmed score remains `0.949`. 2026-05-22 UTC count is `5/5`, so no more submissions today.
- **Submitted batch still pending:** v599 `52913376` Claudedevore R0952 run2 sidecar; v600 `52913377` Gendaijin Pilkwang prior-field fusion; v601 `52913379` Gendaijin Meenal new visual prior; v602 `52913380` Nicolas NFNet Aves lprior075.
- **Fresh scan artifact:** `artifacts/public_kernels_20260522_frontier_candidates/scan_20260522T0601Z.json`.
- **Fresh feed:** no new source-safe candidate supersedes the pending batch. The newest visible items are Samejima HGNet inference/training reruns, JGuevara/Koushik training notebooks, Meenal visual-prior rerun, and the already-submitted/held Gendaijin day0522 family.
- **Decision:** wait for v599-v602 scores. If all drop or tie below 0.949, mark public R0952/Pilkwang/Meenal/NFNet direct lanes as exhausted for 0.96 purposes and pivot back to source discovery/training artifact evidence.


## 2026-05-22 08:00 UTC — v599-v602 scored tied-best, capped-source audit

- **Live state:** v599-v602 all completed at `0.949`; v598 remains `0.860`; best confirmed public LB remains `0.949` and target remains `0.960`. 2026-05-22 UTC submissions are `5/5`, so no more submissions can be made today.
- **Scores landed:**
  - v599 `52913376` Claudedevore R0952 run2 sidecar: `0.949`.
  - v600 `52913377` Gendaijin Pilkwang prior-field fusion: `0.949`.
  - v601 `52913379` Gendaijin Meenal new visual prior: `0.949`.
  - v602 `52913380` Nicolas NFNet Aves lprior075: `0.949`.
- **Lesson:** the broad 0.95-ish public families are real high-plateau signals but not 0.96 breakthroughs. Treat R0952, Pilkwang/new prior-field, Meenal/visual-prior, and NFNet/Aves-lprior direct replay as saturated at the current `0.949` plateau until a genuinely new artifact/source appears.
- **Fresh scan artifact:** `artifacts/public_kernels_20260522_frontier_candidates/scan_20260522T0800Z.json`.
- **Source audit artifact:** `artifacts/public_kernels_20260522_frontier_candidates/source_audit_20260522T0800Z_newleads/summary.json` plus pulled source/full JSON files.
- **08UTC lead triage:**
  - EoS6 family (`dhyuk54/birdclef-2026-eos-6-bz`, `gendaijin/birdclef2026-day0522-nina-eos6-bz`, `scenerysunfireink/eos-6-v7-power-0-90-extreme`) is source-pullable and hidden-path aware. The live config is effectively EoS6 Version.15: direct blend of Model_21 `0.014`, Model_52 PSSM `0.021`, and Model_74 `0.965`; source table shows prior EoS6 variants mostly `0.948`/`0.949`, so this is a high-plateau candidate, not currently strong enough to spend scarce slots without a better score clue.
  - Ykuroka pseudo/OOF notebooks are **not direct-submit safe**: they append train soundscape OOF paths to hidden test paths (`test_paths = test_paths + _OOF_PATHS`), which can create extra non-sample rows. Use only for idea mining around OOF validation / rank gates.
  - Anatoly Iter5 SED ensemble is structurally distinct (`seresnext26t_32x4d`, 5 fold ckpts, `anatoly7m/bc2026-iter5-ckpts`) and sample-schema aware, but direct output/status is unavailable and its no-test branch emits zeros; keep as a repo-owned extraction candidate only after deeper preflight.
  - WildSound v8 is a training notebook using internet/external XC-style data and a 60-epoch ConvNeXtBase path; not a safe/fast direct code-submission candidate.
- **Decision:** no new submission attempt while capped. Next useful slot should require either (a) a new public source with evidence above plateau, or (b) a repo-owned extraction from Anatoly/EoS-style source with stronger validation than direct replay.


## 2026-05-22 10:00 UTC — capped 10UTC scan + v603 Anatoly preflight push

- **Live state:** best remains `0.949`; v599-v602 are all complete at `0.949`; v598 remains `0.860`; 2026-05-22 UTC submissions remain capped at `5/5`. No stale v577/v578 scalar submitter is active.
- **Fresh scan artifact:** `artifacts/public_kernels_20260522_frontier_candidates/scan_20260522T1000Z.json`.
- **Focused source audit artifact:** `artifacts/public_kernels_20260522_frontier_candidates/source_audit_20260522T1000Z_newleads/summary.json`.
- **10UTC lead triage:**
  - Anatoly `bc2026-iter-5-sed-ensemble-submit-v3` updated to version 24. Source now targets `tf_efficientnet_b0.ns_jft_in1k` with `TARGET_FOLDS=[1]` (`b0_fold1 val 0.9579` comment) for speed instead of the earlier 5-SRX version. Dataset `anatoly7m/bc2026-iter5-ckpts` is public, CC0, ~0.86GB, current v4, with notes saying it contains 15 `best.pth` checkpoints in `iter5/{backbone}_fold{N}/`.
  - Ykuroka's new `949-birdclef-2026-acoustic-prior-field-fusion-oof`, `perch-v2-protossm-0-925-oof`, and updated `iter-pseudo-oof` remain **direct-submit unsafe** because source appends OOF/train paths to test paths (`test_paths = test_paths + _OOF_PATHS`). Use for idea-mining only.
  - Suncrest `eval-birdnet-1000-soundscapes` is an evaluation notebook with BirdNET TFLite source only; not a competition submit candidate.
- **Repo-owned preparation:** created private repo-owned verification kernel `kaggle-kernels/v603-anatoly-iter5-b0f1-verify/` and push script `scripts/push_v603_anatoly_iter5_b0f1_verify.py`. Pushed to Kaggle as `yourslewis/bc26-v603-anatoly-iter5-b0f1-verify`, kernel id `120204403`, version 1. This is **not** a competition submission.
- **Verification status:** Kaggle push succeeded with no invalid data/competition/model sources. `kernels/pull` can retrieve the private kernel, but `kernels/status` and `kernels/output` currently return 404/HTML for this private pushed kernel, so runtime/output verification is not yet available. Do not submit v603 to competition until a later check confirms COMPLETE and a valid non-constant `submission.csv`, or until the verifier path is repaired.


## 2026-05-22 12:00 UTC — capped scan, v603 verifier still blocked, fresh lead audit

- **Live state:** best remains `0.949`; v599-v602 all `0.949`; v598 `0.860`; 2026-05-22 UTC count remains `5/5`. No stale v577/v578 scalar submitter is active.
- **Fresh scan artifact:** `artifacts/public_kernels_20260522_frontier_candidates/scan_20260522T1200Z.json`.
- **Focused source audit artifact:** `artifacts/public_kernels_20260522_frontier_candidates/source_audit_20260522T1200Z_newleads/summary.json`.
- **v603 verification:** tried Bearer API variants and local Kaggle CLI. Bearer `kernels/pull` still retrieves private v603 (`yourslewis/bc26-v603-anatoly-iter5-b0f1-verify`), but `kernels/status`/`kernels/output` still return 404/HTML; Kaggle CLI with the available venv is unauthorized/forbidden. v603 remains **not competition-submit-ready** until `submission.csv` output can be verified.
- **Fresh 12UTC lead triage:**
  - `jguevarag/08-winning-tta-submission-pipeline`: source-pullable but unsafe/weak as direct candidate. It creates sample-submission fallback immediately, then relies on a model from JGuevara 07; no attached output/status via API, and fallback/zero-output risk is present.
  - `mtoshidesu/testbirdclef-2026-eos-6-bz`: EoS6 derivative; large source with train fallback/debug path and zero/fallback output risk. It remains high-plateau EoS family, not distinct 0.96 evidence.
  - `samejimatink0/birdclef-2026-hgnetv2-b0-baseline-training`: training/internet path only; previous v598 inference artifact scored `0.860`, so do not spend slots on standalone HGNet without stronger anchored diagnostics.
  - `scenerysunfireink/birdclef-2026-perch-v2-full-v3`: Perch V2 model can only predict a subset of 234 species and returns `-1000` rows when runtime threshold/no-prediction triggers; it is evaluation/idea-mining, not direct-submit safe.
  - `deepanshus167/bird-claasifier-comp`: training/EDA notebook, no credible competition-submit path.
- **Decision:** still no submission while capped. Next reset should not use v603 unless verifier output is available; otherwise continue source discovery or build a local/repo-owned validation harness for Anatoly B0 fold1.


## 2026-05-22 14:00 UTC — capped scan, PCEN sidecar lead held

- **Live state:** best remains `0.949`; v599-v602 all `0.949`; v598 `0.860`; 2026-05-22 UTC count remains `5/5`. No durable v577/v578/v6 submitter is active.
- **v603 verification:** unchanged. Bearer `kernels/pull` can retrieve private `yourslewis/bc26-v603-anatoly-iter5-b0f1-verify`, but `kernels/status` and `kernels/output` still return 404/HTML. Do not competition-submit v603 until output can be verified.
- **Fresh scan artifact:** `artifacts/public_kernels_20260522_frontier_candidates/scan_20260522T1400Z.json`.
- **Focused source audit artifact:** `artifacts/public_kernels_20260522_frontier_candidates/source_audit_20260522T1400Z_newleads/summary.json`.
- **Strongest fresh lead:** Pilkwang `birdclef-2026-pcen-sidecar-package` / dataset `pilkwang/birdclef26-sidecar-exp001`.
  - Source describes **Acoustic Prior-Field Fusion + PCEN Sidecar**: low-weight yukiZ Perch/SSM + dominant v6 prior-field branch; BirdNET and custom PCEN/ConvNeXt sidecar are constrained rank corrections with top-k masks, overlap guards, and perturbation budgets.
  - Dataset `pilkwang/birdclef26-sidecar-exp001` is public, v1, ~0.56GB, license unknown, tagged pre-trained model.
  - Current notebook settings: `RUN_EXP001_SIDECAR=True`, `SIDECAR_EXP001_REQUIRE=True`, `SIDECAR_EXP001_DEVICE="cpu"`, `SIDECAR_EXP001_BATCH_SIZE=8`, `SIDECAR_EXP001_FOLDS=[0]`, `SIDECAR_EXP001_FORCE_INFER=True`, timeout `600s`, weight cap `0.020`, D budget `0.003`, anchor top-k `48`, side top-k `32`, tau `0.55`, max active fraction `0.25`.
  - **Decision:** high-upside idea-mining / repo-owned extraction candidate, but not direct-submit-ready because API status/output are unavailable and source has fallback/debug/constant-risk paths. It needs output verification or a repo-owned harness before a slot.
- **Other fresh lead triage:**
  - Jungchan `birdclef-first`: Big Mods / class-aware blend + targeted BirdNET + sonotype mirroring; but train fallback, zero/fallback, and constant/ones risks are present.
  - Samejima visual CPU update: still Perch/visual lineage with fallback/constant risks; same family as saturated visual candidates.
  - Scenery Perch V2 Full v4: only Perch subset/species path; train fallback and constant/ones risk.
  - Lamido/Deepanshu: EDA/training or no clear competition writer.
- **Decision:** no submission while capped. Next useful work: build a local/repo-owned PCEN sidecar extraction/verification plan, or verify public output if Kaggle status/output becomes available.


## 2026-05-22 16:02 UTC — v604 PCEN verifier packaged, still capped

- **Live state:** best remains `0.949`; v599-v602 all `0.949`; v598 `0.860`; 2026-05-22 UTC submissions remain `5/5` capped. No durable v577/v578/v6 submitter is active.
- **v603 verification:** unchanged; private pull works but status/output remain 404/HTML.
- **Fresh scan artifact:** `artifacts/public_kernels_20260522_frontier_candidates/scan_20260522T1602Z.json`.
- **Focused audit artifact:** `artifacts/public_kernels_20260522_frontier_candidates/source_audit_20260522T1602Z_newleads/summary.json`.
- **New/updated lead triage:**
  - `studyexchange/birdclef-2026-infer-s14`: pulls, but no clear competition submission writer; references S14/SED assets only, hold/reject as direct candidate.
  - `anatoly7m/bc2026-iter-5-sed-ensemble-submit-v3`: updated to v26, still zero/fallback output risk and no API output. Keep v603/vAnatoly held pending verifier.
  - `tuannm3812/birdclef-2026-perch-v2`: Perch v2 path with sample/test writer markers, but no output/status and Perch-only/subset lineage; idea-mining only.
  - `junseonglee11/birdclef2026-eos5-scoreblend-g004-w05`: EoS5 scoreblend / G004 weight 0.5, but EoS plateau derivative with fallback/constant risks. Do not spend scarce slot unless a real score appears.
  - `samejimatink0/birdclef-2026-hgnetv2-b0-baseline-inference`: updated v8 but same standalone HGNet lane; v598 already scored `0.860`.
  - `pilkwang/birdclef-2026-pcen-sidecar-package`: still top watch item; source unchanged at v7 and remains unverified but structurally distinct.
- **Repo-owned preparation:** created private repo-owned verification kernel `kaggle-kernels/v604-pilkwang-pcen-sidecar-verify/` and push script `scripts/push_v604_pilkwang_pcen_sidecar_verify.py`. Pushed to Kaggle as `yourslewis/bc26-v604-pilkwang-pcen-sidecar-verify`, kernel id `120230356`, version 1. This is **not** a competition submission.
- **Validation:** v604 push succeeded with no invalid data/competition/kernel/model sources; metadata and notebook JSON parse; push script py_compile passes. Like v603, `kernels/pull` works but `kernels/status` and `kernels/output` return 404/HTML, so v604 is **not competition-submit-ready** until output can be verified.


## 2026-05-22 18:01 UTC — v604 guarded submit monitor queued for reset

- **Live state:** best remains `0.949`; v599-v602 all `0.949`; v598 `0.860`; 2026-05-22 UTC submissions remain capped at `5/5`.
- **Fresh scan artifact:** `artifacts/public_kernels_20260522_frontier_candidates/scan_20260522T1801Z.json`.
- **Focused audit artifact:** `artifacts/public_kernels_20260522_frontier_candidates/source_audit_20260522T1801Z_newleads/summary.json`.
- **Private verifier status:** SDK verification now works for v604/v603 even though raw REST status/output endpoints return 404. v604 is COMPLETE with files `submission.csv`, `sidecar_exp001_diagnostics.csv`, `submission_before_all_sidecars.csv`, `submission_before_exp001_sidecar.csv`, `submission_protossm.csv`, `submission_sed.csv`, and `v17_logs.json`. v604 sample/public output has shape `3 x 235`, finite/non-constant values, range `0.4607935..0.5381691`, `0` zeros, `98` unique first-100 values. v603 is COMPLETE but its output is all-zero sample fallback (`0.0..0.0`, 702 zeros), so it remains rejected/held.
- **v604 caveat:** the public/sample run diagnostics show `effective_weight=0.0` and `skip_reason=No test_soundscapes .ogg files found; public/dry-run anchor rows cannot be matched by exp001 inference.` This means schema/output is verified, but the PCEN sidecar correction itself is only exercised on hidden test. It is still the best distinct reset slot because the code path is source-safe and guarded, but this is a real uncertainty.
- **Guarded submitter:** added `scripts/submit_v604_pilkwang_pcen_when_ready.py`. Preflight-only passed with source markers, COMPLETE status, required outputs, and non-constant `submission.csv`. Started reset monitor pid `73273` with `--wait-for-slot`; log `logs/v604_pilkwang_pcen_submit_monitor_20260522T1801Z.log`; it is sleeping until UTC reset after daily cap.
- **Fresh lead triage:** Kalyan Blend 2 is another EoS/Blend plateau derivative with fallback/constant risks; StudyExchange S14 still lacks a clear writer; Anatoly v26 remains unverified/zero-fallback; Junseong scoreblend is EoS plateau derivative; Lamido has model weights but zero/fallback risk; Pilkwang PCEN remains selected.


## 2026-05-22 20:00 UTC — v604 monitor alive, fresh PCEN/scoreblend scan

- **Live state:** best remains `0.949`; v599-v602 all `0.949`; v598 `0.860`; 2026-05-22 UTC submissions remain capped at `5/5`.
- **v604 reset monitor:** pid `73273` is still alive and sleeping after successful preflight; log `logs/v604_pilkwang_pcen_submit_monitor_20260522T1801Z.log` still shows `daily cap reached; sleeping 21506s`. No duplicate submitter started.
- **Fresh scan artifact:** `artifacts/public_kernels_20260522_frontier_candidates/scan_20260522T2000Z.json`.
- **Focused audit artifact:** `artifacts/public_kernels_20260522_frontier_candidates/source_audit_20260522T2000Z_newleads/summary.json`.
- **Fresh lead triage:**
  - `beicicc/bc26-pcen-sidecar-may22` is a fork/copy of the Pilkwang PCEN package: same `pilkwang/birdclef26-sidecar-exp001` dataset, same yukiZ/v6/BirdNET/PCEN lineage, same fallback/zero/constant risks. It does **not** supersede queued v604.
  - `junseonglee11/birdclef2026-eos6-scoreblend-g004-w05` is an EoS6 scoreblend derivative with G004-style score blend markers, but has fallback/zero/constant risks and no output; hold as plateau-family idea-mining only.
  - `meenalsinha/birdclef-2026-improved` v28 remains saturated Visual/BirdNET/Perch lineage; prior direct family tied `0.949` and did not break out.
  - `kalyankkr/birdclef-2026-blend-2` remains EoS/Blend derivative with fallback/zero/constant risks.
  - `studyexchange/birdclef-2026-infer-s14` still lacks a clear competition writer; `lamidoahmad/birdclef-2026` pull is now 403/inaccessible.
- **Decision:** keep v604 as the single queued reset slot. Do not add another monitor or spend slots on Beicicc/Junseong/Meenal/Kalyan siblings without verified output and score evidence above the `0.949` plateau.


## 2026-05-22 22:00 UTC — v604 monitor still sole reset owner

- **Live state:** best remains `0.949`; v599-v602 all `0.949`; v598 `0.860`; 2026-05-22 UTC submissions remain capped at `5/5`. 2026-05-23 count is still `0` at this check.
- **v604 reset monitor:** pid `73273` remains alive after ~4h and is still sleeping from the successful preflight/cap response. Do not start a duplicate monitor. It should wake around UTC reset and submit `v604: Repo-owned Pilkwang PCEN sidecar verify` once a slot exists.
- **Fresh scan artifact:** `artifacts/public_kernels_20260522_frontier_candidates/scan_20260522T2200Z.json`.
- **Focused audit artifact:** `artifacts/public_kernels_20260522_frontier_candidates/source_audit_20260522T2200Z_newleads/summary.json`.
- **Fresh lead triage:**
  - Beicicc `bc26-pcen-sidecar-may22` remains a direct fork/copy of Pilkwang PCEN package, same dataset and risks; it does not supersede queued v604.
  - Junseong `eos6-scoreblend-g004-w05` is an EoS6 scoreblend derivative with fallback/zero/constant risks; no verified output or non-plateau evidence.
  - Meenal v28 / Kalyan Blend2 remain saturated visual/EoS/Blend family with fallback/constant risks.
  - StudyExchange S14 still lacks a clear competition writer; Anatoly v27 still has zero/fallback risk and no usable output; v603 remains held.
- **Decision:** keep v604 as the single reset-slot owner. No new kernels or submitters were created this turn.


## 2026-05-23 00:10 UTC — v604 submitted; preserve remaining reset slots pending score

- **Live state:** best remains `0.949` until v604 scores. 2026-05-23 UTC now has `1/5` submissions used.
- **Submitted:** v604 `Repo-owned Pilkwang PCEN sidecar verify`, ref `52937418`, at `2026-05-23T00:03:03Z`; status is currently `pending` with no error.
- **Monitor:** pid `73273` exited after successful submit. Log confirms reset preflight repeated after cap cleared, kernel COMPLETE/no failure, `submission.csv` valid/non-constant sample shape, then `submitted {"ref": 52937418}`.
- **Fresh scan artifact:** `artifacts/public_kernels_20260523_frontier_candidates/scan_20260523T0000Z.json`.
- **Focused audit artifact:** `artifacts/public_kernels_20260523_frontier_candidates/source_audit_20260523T0008Z_newleads/summary.json`.
- **Post-reset lead triage:**
  - Pilkwang PCEN package v8 is the same PCEN sidecar lineage already represented by v604; do not duplicate before v604 score.
  - Junseong EoS5/EoS6 scoreblend variants are source-pullable but output-private and have train/fallback/zero/constant risk markers; no direct submit without repo-owned verification.
  - Gendaijin/Kalyan/Meenal/Nina/EoS derivatives remain plateau-family; no evidence they beat 0.949.
  - Alrickh/Starsdaisuki public0952 copies are source-identical (`eeb13d1a4a130e5a`) and similar to the already-bad public0952 lane; hold.
  - Cheny exp080 Karnak dual-arch safe is source-pullable but fallback/constant-risk and overlaps Karnak/EoS plateau; hold.
- **Decision:** preserve remaining four 2026-05-23 slots until v604 scores or a clearly distinct source-safe candidate appears. No duplicate submitter started.


## 2026-05-23 02:10 UTC — v604 tied; v605 Eslam v26C verifier launched

- **Live state:** v604 `Repo-owned Pilkwang PCEN sidecar verify` scored `0.949`, tying the current best but not improving toward `0.960`. 2026-05-23 UTC slots used: `1/5`; confirmed best remains `0.949`.
- **Lesson:** the Pilkwang PCEN sidecar is safe but still plateau-bound. Do not duplicate PCEN forks (`pilkwang`, `beicicc`, `gendaijin day0523-pcen`) without a new artifact or stronger evidence.
- **Fresh scan artifact:** `artifacts/public_kernels_20260523_frontier_candidates/scan_20260523T0200Z.json`.
- **Focused audit artifact:** `artifacts/public_kernels_20260523_frontier_candidates/source_audit_20260523T0200Z_newleads/summary.json`.
- **New verifier:** pushed repo-owned private kernel v605 `yourslewis/bc26-v605-eslam-v26c-source-verify`, version 1 / kernel id `120276728`, from `eslamelokpy/birdclef2026-v26c` source. Push returned HTTP 200 and the kernel was observed RUNNING via SDK immediately after push.
- **v605 caveat:** Kaggle rejected `eslamelokpy/birdclef2026-student-onnx` as an invalid dataset source. Source treats missing student ONNX folds as optional, so this verifier still tests Eslam's main Perch/SED/prior path. Do **not** submit v605 until kernel COMPLETE output is verified with non-constant `submission.csv` and no traceback.
- **Other fresh leads:** Koushik/Perch-v2/JGuevara are structurally simpler baseline/training lanes; StudyExchange S14 still lacks writer; Gendaijin PCEN/Junseong are copies of already-held PCEN/scoreblend lineages; WildSound/CKPT-chain require internet or lack competition writer.
- **Decision:** preserve remaining 4 slots while v605 verifier runs. If v605 completes cleanly, inspect output before any submit; if it fails or falls back, do not spend a slot.


## 2026-05-23 04:15 UTC — v605 failed guard; v606 ProtoSSM repair verifier running

- **Live state:** confirmed best remains `0.949`; v604 tied `0.949`; 2026-05-23 slots used remain `1/5`.
- **v605 result:** verifier `yourslewis/bc26-v605-eslam-v26c-source-verify` reached `ERROR` before `submission.csv`. Guard correctly prevented submission. Root cause: `NameError: name 'proto_model' is not defined` at the ProtoSSM inference cell; only cache outputs were produced.
- **Repair action:** created and pushed v606 `yourslewis/bc26-v606-eslam-v26c-proto-repair`, version 1 / kernel id `120285231`. Patch trains `LightProtoSSM` in submit mode and materializes `emb_te_f`, `sc_te_f`, `test_site_ids`, and `test_hour_ids` before the ProtoSSM inference cell.
- **v606 status:** SDK reports `RUNNING`, no failure message. Output listing currently has no files/log yet, so no submission is allowed until COMPLETE + valid non-constant `submission.csv` + no traceback.
- **Fresh scan artifact:** `artifacts/public_kernels_20260523_frontier_candidates/scan_20260523T0400Z.json`.
- **Focused audit artifact:** `artifacts/public_kernels_20260523_frontier_candidates/source_audit_20260523T0400Z_newleads/summary.json`.
- **Fresh lead triage:** Koushik source writes random uniform placeholder predictions despite safe schema; do not submit. Tuannm Perch v2 needs a private/unclear probe artifact. Safe EoS5 / Ykuroka / Cheny variants are fallback-risk or plateau-family. CKPT-chain/WildSound require internet or lack a competition writer. v606 remains the only active candidate.


## 2026-05-23 06:15 UTC — v606 failed guard; v607 ProtoSSM-save repair running

- **Live state:** confirmed best remains `0.949`; v604 tied `0.949`; 2026-05-23 slots used remain `1/5`.
- **v606 result:** verifier `yourslewis/bc26-v606-eslam-v26c-proto-repair` reached `ERROR` before final `submission.csv`. Root cause: downstream blend cell attempted to read missing `submission_protossm.csv`; only cache outputs were produced. No submission was spent.
- **Repair action:** created and pushed v607 `yourslewis/bc26-v607-eslam-v26c-proto-save`, version 1 / kernel id `120293556`. Patch persists `submission_protossm.csv` from ProtoSSM sigmoid probabilities before the SED/student/final rank-blend cell.
- **v607 status:** SDK reports `RUNNING`, no failure message. Output listing currently has no files/log yet; no submission allowed until COMPLETE + valid non-constant `submission.csv` + no traceback.
- **Fresh scan artifact:** `artifacts/public_kernels_20260523_frontier_candidates/scan_20260523T0600Z.json`.
- **Focused audit artifact:** `artifacts/public_kernels_20260523_frontier_candidates/source_audit_20260523T0600Z_newleads/summary.json`.
- **Fresh lead triage:** new Beicicc/Gendaijin/karansinghbisht public forks are copies/near-copies of already plateaued PCEN/EoS/NFNet/Junseong lineages and have fallback/zero/constant risk markers; no direct submit supersedes v607.


## 2026-05-23 08:05 UTC — v607 submitted; promising recipe autoresearch specs created

- **Live state:** confirmed best remains `0.949`; 2026-05-23 slots used are now `2/5`.
- **Submitted:** v607 `Repo-owned Eslam v26C ProtoSSM save repair`, ref `52947220`, status `pending` at check time.
- **v607 preflight:** private verifier COMPLETE/no failure; outputs include `submission.csv`, `submission_protossm.csv`, `submission_sed.csv`, and cache files; no traceback; `submission.csv` stats valid `240x235`, finite, non-constant, unique row IDs.
- **Specs created:**
  - `AUTORESEARCH.md` root pointer.
  - `experiments/autoresearch/2026-05-23-birdclef-096-frontier-recipes/protocol.md` active protocol.
  - `docs/BIRDCLEF_096_PROMISING_RECIPES_AUTORESEARCH_20260523.md` recipe plan.
- **Priority:** P0 is G124/S124 reconstruction or artifact discovery. Missing private asset remains `itshyao/birdclef2026-g124-effv2s-2025pre-pseudo-assets` with expected `g124_fold1_fp16.pt`, `_best.pt`, and `submission_g124_effv2s_fold1_s124.csv`-like outputs. P1 is result-gated Eslam v607; P2 is source/artifact scout.
- **Non-goal:** LLM labeling path is skipped because no audio-capable labeler is available.


## 2026-05-23 08:15 UTC — fresh source audit + G124 reconstruction configs

- Audited new 08UTC frontier candidates from `scan_20260523T0800Z.json`.
- `studyexchange/birdclef-2026-infer-s14`: COMPLETE with valid `240x235` output and fresh-Perch/S14 package, but source score history is `0.932`/`0.943`, below current `0.949`; hold for idea-mining, not a slot before v607 scores.
- `henryszy/bc2026-g124-protectdelta-v84`: source contains useful G124 protected-delta logic and expects `g124_fold1_fp16.pt` under the private G124 asset dataset, but if unavailable it catches the error and keeps NFNet/anchor `submission.csv`; likely plateau unless the real G124 artifact is attached.
- New PCEN/EoS/NFNet forks remain plateau-like after v599-v604; `anatoly7m`/`jguevarag` fresh TTA/SED public outputs were constant 3-row dry-run outputs, so no blind direct submission.
- Added concrete G124 reconstruction configs:
  - `configs/birdclef/g124_effv2s_public946_pseudo_smoke_20260523.json`
  - `configs/birdclef/g124_effv2s_public946_pseudo_pilot_20260523.json`
- Commit pushed: `9165dc4` (`Add G124 EffV2-S reconstruction configs`) on PR #254.


## 2026-05-23 10:15 UTC — v607 dropped; v608 submitted; G124 smoke rejected

- **Status:** v607 `Repo-owned Eslam v26C ProtoSSM save repair` scored `0.934`, below the `0.949` plateau. Kill the repaired Eslam 2-way fallback lane unless the missing student ONNX artifact becomes available; the repair was mechanically valid but not leaderboard-competitive.
- **Slots:** 2026-05-23 used `2/5` before new action; best remains `0.949`.
- **Fresh scan:** saved `artifacts/public_kernels_20260523_frontier_candidates/scan_20260523T1000Z.json`; audit saved `artifacts/public_kernels_20260523_frontier_candidates/source_audit_20260523T1000Z_newleads/summary.json`.
- **Source audit lessons:**
  - `scenerysunfireink/birdclef-2026-improved-fork` has valid `240x235` dry-run output, hidden-test source markers, and structurally different inner blend / BirdNET / distill model sources. Dry-run output correlation vs v607 was only `0.860`, so it is distinct enough for one guarded slot despite no >0.949 evidence.
  - `scenerysunfireink/bc26-inner-ensemble-v1/v2` are dependent-output kernels over `birdclef-2026-improved-fork`, not preferred for direct submission because notebook-output dependency may not recompute hidden rows.
  - `pilkwang/birdclef-2026-eos6-bz-pcen-rank-sidecar` is a PCEN/EoS6-bz sidecar but public output is only 3-row dry-run; after v604 PCEN tied `0.949`, hold unless it shows a real scored improvement.
  - `ommodi07`, `adarsh5harma`, and `jacqueszhelinzhang/deepcnn` outputs were invalid/constant/ragged for direct guarded use.
- **Submitted:** v608 `Guarded direct Scenery improved inner-blend source`, ref `52950601`, using source version 1 after COMPLETE/no-failure/output preflight. This uses the base improved source, not the dependent inner-v1/v2 wrappers.
- **G124 smoke:** copied G124 configs to trainer and ran `g124_effv2s_public946_pseudo_smoke_20260523.json` on GPU1. Training reached best val AUC `0.726` at epoch 3 with low student/teacher corr `0.192`; torchscript and student predictions were produced, but ONNX export hung and was killed. Criterion `>=0.93` failed, so do not run the pilot config from this initialization. This reinforces that simple from-scratch EffV2-S pseudo smoke is not enough to recreate the missing G124 artifact.


## 2026-05-23 12:10 UTC — v608 tied; 12UTC scan no safe high-upside slot

- **Status:** v608 `Guarded direct Scenery improved inner-blend source` completed at `0.949`, tied current best. Lesson: Scenery/Ykuroka/KingKong inner-blend/BirdNET-family source is distinct from v607 but still public plateau; do not tune dependent inner-v1/v2 wrappers without new evidence.
- **Slots:** 2026-05-23 used `3/5`; best remains `0.949`.
- **Fresh scan saved:** `artifacts/public_kernels_20260523_frontier_candidates/scan_20260523T1200Z.json`.
- **Fresh audit saved:** `artifacts/public_kernels_20260523_frontier_candidates/source_audit_20260523T1200Z_newleads/summary.json`.
- **Artifact search saved:** `artifacts/public_kernels_20260523_frontier_candidates/artifact_search_20260523T1200Z.json`.
- **No new submission this run:**
  - `anthonytherrien/gated-rank-fusion-pipeline` is source-interesting but public `submission.csv` is invalid/ragged (`243x235`, nonnumeric cells); do not submit blindly.
  - `raunakdey07/birdclef-2026-v9` is COMPLETE with valid 3-row dry-run output, but source records `Model_7` as LB `0.948`; below plateau and not 0.96-upside enough.
  - `pilkwang/birdclef-2026-eos6-bz-pcen-rank-sidecar` remains PCEN/EoS6-bz family; after repo-owned v604 PCEN tied `0.949`, hold unless a real scored improvement appears.
  - `anatoly7m/bc2026-iter-5-sed-ensemble-submit-v3` output is constant/sample-like 3-row dry-run; no direct slot.
  - `mlclsumit` / `gandharvakhedekar` were still RUNNING/no `submission.csv`; `deepanshus167` no output; `neslihannuryilmaz` ERROR.
- **G124 artifact search:** exact `g124_fold1_fp16.pt` / private G124 slug still only surfaces derivative kernels, no public dataset. Dataset search returns no public artifact. `fold0_ep12_auc0.9643` and `efficientnet_b3_pretrained.pt` also no dataset/kernel artifact hit.
- **Decision:** preserve remaining 2 slots for a genuinely source-safe distinct candidate or a new artifact; next loop should recheck running candidates and any new >0.949 claims.


## 2026-05-23 14:10 UTC — 14UTC scan, no slot spent

- **Status:** best remains `0.949`; 2026-05-23 slots used `3/5`; remaining `2/5`. Latest scored: v608 `0.949`, v607 `0.934`, v604 `0.949`.
- **Fresh scan saved:** `artifacts/public_kernels_20260523_frontier_candidates/scan_20260523T1400Z.json`.
- **Fresh audit saved:** `artifacts/public_kernels_20260523_frontier_candidates/source_audit_20260523T1400Z_newleads/summary.json`.
- **No new submission:** no candidate cleared the distinct/high-upside/source-safe bar.
- Candidate notes:
  - `samejimatink0/birdclef-2026-visual-cpu-inference`: COMPLETE with valid `240x235` output and train8 artifacts, but source labels visual branch around `0.948`; dry-run output is highly correlated with v608 (`corr≈0.993`, MAE≈0.018). Since v608 already tied `0.949`, this is likely another plateau/visual-source slot, not 0.96-upside. Hold.
  - `samejimatink0/birdclef-2026-hgnetv2-b0-baseline-inference`: public output invalid/non-numeric 3-row dry-run; earlier repo-owned Samejima/HGNet attempts did not improve (`v598=0.860`, v596 HGNet sidecar `0.946`). Hold unless a new artifact/score appears.
  - `samejimatink0/birdclef-2026-hgnetv2-b0-baseline-training`: still RUNNING/no submission output; training/source artifact only.
  - `adarsh5harma/v62`: invalid/ragged public output; `v66`: valid 3-row dry-run but EoS/phase1 family; `v68` was already invalid at 10UTC. Hold.
  - `pilkwang/birdclef-2026-eos6-pcen-rank-sidecar`: PCEN/EoS6 family, v604 already tied `0.949`; hold.
  - `thbdh5765/bc26-y948-w010-lanec-v2-fold0-art-v1`: valid 3-row dry-run but source is visual/BirdNET-like `0.948` lane; hold.
  - `gandharvakhedekar` / `mlclsumit`: still RUNNING/no `submission.csv` at audit time.
- GPU server check: no BirdCLEF student train process; GPU1 free. GPU0 occupied by unrelated/other workload. No training launched because G124 from-scratch smoke already failed gate and no stronger initialization/artifact found.



## 2026-05-23 16:20 UTC — 16UTC frontier scan, no slot spent

- **Status:** latest Kaggle submissions unchanged: v608 `0.949`, v607 `0.934`, v604 `0.949`; current confirmed best remains `0.949` vs target `0.960`. 2026-05-23 UTC slots used `3/5`, remaining `2/5`.
- **Repo/process:** branch `feature/birdclef-20260522-v599-v602-pending`, latest commit `71d846c`; PR #254 remains OPEN / REVIEW_REQUIRED / BLOCKED, PR #245 is merged. No active v577/v578 scalar submitter or BirdCLEF monitor process was found.
- **Fresh scan saved:** `artifacts/public_kernels_20260523_frontier_candidates/scan_20260523T1600Z.json`.
- **Fresh audit saved:** `artifacts/public_kernels_20260523_frontier_candidates/source_audit_20260523T1600Z_newleads/summary.json`.
- **Output audits saved:** `artifacts/public_kernels_20260523_frontier_candidates/output_audit_20260523T1600Z/summary.json` and `artifacts/public_kernels_20260523_frontier_candidates/output_audit_20260523T1600Z_more/summary.json`.
- **No new submission:** no 16UTC candidate cleared the distinct/high-upside/source-safe bar.
- Candidate notes:
  - `chenyfdws/bc26-exp070-public0952-s124-g124-repro`: COMPLETE with valid `240x235` output and safe hidden-test writer, but this is the already-tested public0952/Perch-prior-probe lane (`v595=0.899`). Do not duplicate.
  - `samejimatink0/birdclef-2026-visual-cpu-inference`: now RUNNING/no output in the 16UTC output API check. The earlier valid visual output was already judged plateau-like (`0.948` branch, high corr to v608), so hold.
  - `samejimatink0/birdclef-2026-hgnetv2-b0-baseline-inference`: COMPLETE but public `submission.csv` is invalid/non-numeric (`3x235`, 702 bad numeric cells). Hold unless a fixed output/artifact appears.
  - `ykuroka`/`beicicc`/`gendaijin`/`karansinghbisht` Nina EoS6-bz and PCEN forks: mostly invalid `243x235` public outputs with nonnumeric cells or 3-row dry-runs; overlaps v604/v608 plateau families. Hold.
  - `jungchanryu/birdclef-first`, `beicicc/bc26-gendaijin-junseong-eos6-may23`, `junseonglee11/birdclef2026-eos6-scoreblend-g004-w05`, `starsdaisuki/birdclef-2026-v131-nina-eos6-sz`, and `beicicc/bc26-nfnet-aves-lp075-may23`: valid only as 3-row dry-run/sample outputs, not enough for a guarded direct slot after similar families tied/dropped.
  - `itshyao/birdclef-2026-s116-g116-hgnet-b1-rawpseudo-all5`: source-safe-looking pure G116/HGNet all5 sidecar with finite 12-row dry-run output, but G116/G123/G124 siblings have already tied rather than lifted (`v587/v588/v589/v597=0.949`). Treat as idea-mining only unless a scored improvement appears.
  - `karnakbaevarthur/s124-g124-reverse-engineered`: still COMPLETE but no output artifacts; useful for code-mining/reconstruction, not direct submit.
- **Decision:** preserve the remaining 2 slots for a real new artifact/high-claim source. Next loop should recheck Samejima HGNet training / running notebooks and keep hunting for public G124 artifacts or a non-EoS/non-PCEN structural source.


## 2026-05-23 18:10 UTC — v609 PerchFusion submitted; Gandharva training artifact noted

- **Status before action:** best remained `0.949`; latest scored submissions unchanged (v608 `0.949`, v607 `0.934`, v604 `0.949`). 2026-05-23 UTC slots used `3/5`; no v577/v578 scalar submitter or BirdCLEF monitor process was active. PR #254 remained open/review-required; branch was clean at `cd79e42`.
- **Fresh scan saved:** `artifacts/public_kernels_20260523_frontier_candidates/scan_20260523T1800Z.json`.
- **Fresh audit saved:** `artifacts/public_kernels_20260523_frontier_candidates/source_audit_20260523T1800Z_newleads/summary.json`.
- **Output audit saved:** `artifacts/public_kernels_20260523_frontier_candidates/output_audit_20260523T1800Z/summary.json`.
- **Selected candidate:** `svanikkolli/perchfusion-engine` v8 as v609. Rationale: structurally distinct enough for one guarded slot after the 0.949 plateau — source labels itself `v951 Target` and adds 3x Perch audio-shift TTA, 3x SED TTA, larger in-notebook ProtoSSM (`d_model=192`, 3 SSM layers), ResidualSSM correction, raw Perch logits as a third rank-blend member, and safety gates. It is not a dependent-output wrapper.
- **v609 preflight:** source pull version `8`; kernel COMPLETE/no failure; outputs include `submission.csv`, `submission_protossm.csv`, `submission_sed.csv`, `cache/perch_arrays_tta3x.npz`, `cache/perch_meta_tta3x.parquet`; log includes `Training complete`, `ProtoSSM branch done`, `SED branch done`, `3-way rank blend`, and `Diagnostics OK`. Public final `submission.csv` is intentionally sample-shaped (`3x235`) because the source aligns dry-runs to `sample_submission.csv`; hidden source path uses `/test_soundscapes`. Intermediate dry-run branches validated as full train rows: ProtoSSM `240x235`, finite/non-constant, range `0.00038196085..0.95813334`; SED `240x235`, finite/non-constant, range `2.2149461e-05..0.97304124`.
- **Submitted:** v609 `Guarded direct PerchFusion v951 TTA source`, ref `52962837`, pending at check time. 2026-05-23 slots used now `4/5`; preserve final slot unless a stronger source-safe candidate appears or v609 scores high and needs immediate confirmation/port.
- **Other 18UTC findings:**
  - `gandharvakhedekar/birdclef2026-new` completed as a training artifact with EfficientNet-B3 checkpoints and fold AUC filenames around `0.965`-`0.969`, but no inference/submission writer. High-upside for repo-owned inference-port work, not direct submit-ready.
  - Samejima/Cheny visual outputs remain valid but highly correlated with v608 (`corr≈0.993`) and visual branch evidence is below plateau; do not spend the final slot there.
  - HGNet/Kosuke/Henry/Praxel/Chaney variants mostly provide valid 240-row raw branch files, but final submissions are often 3-row dry-runs or highly correlated with the v608/PCEN plateau. Standalone HGNet raw branches are diverse but previous HGNet slots dropped/tied; require anchored blend/score evidence before slot use.
  - PCEN/EoS6 forks remain saturated after v604/v608; Cheny public0952 remains rejected by v595 `0.899`.


## 2026-05-23 20:15 UTC — v609 timeout; v610 Gandharva B3 verifier running

- **Status:** v609 `Guarded direct PerchFusion v951 TTA source` completed with no public score due Kaggle runtime timeout. Best remains `0.949`; target remains `0.960`. 2026-05-23 slots used `4/5`, with one slot remaining. No active v577/v578 scalar submitter or BirdCLEF monitor process was found.
- **Repo state at start:** branch `feature/birdclef-20260522-v599-v602-pending`, latest pushed commit `dfbac4b`; PR #254 open/review-required; PR #245 merged.
- **Fresh scan saved:** `artifacts/public_kernels_20260523_frontier_candidates/scan_20260523T2000Z.json`.
- **Fresh audit saved:** `artifacts/public_kernels_20260523_frontier_candidates/source_audit_20260523T2000Z_newleads/summary.json`.
- **v609 lesson:** PerchFusion is structurally interesting but public dry-run wall time was already ~11.8 min and hidden rerun exceeded the competition limit. Do not resubmit TTA-heavy in-notebook training/TTA variants; any PerchFusion follow-up must be repo-owned and much faster (precomputed artifacts or no in-kernel training/TTA).
- **20UTC source triage:** Sakur/Samejima/Cheny visual forks remain the same visual plateau family highly correlated with v608; EoS6/PCEN forks remain saturated after v604; Samejima/TTAhara HGNet inference public output remains invalid/non-numeric; Itshyao S116/G116 remains G-sidecar plateau-family. No direct public source deserved the final slot.
- **Repo-owned high-upside action:** built and pushed v610 verifier `yourslewis/bc26-v610-gandharva-b3-checkpoint-inference`, kernel id `120358093`, version 1. It attaches `gandharvakhedekar/birdclef2026-new` as a kernel source and implements a minimal CPU inference writer for the EfficientNet-B3 SED checkpoints (`fold0_ep12_auc0.9679.pth`, `fold1_ep15_auc0.9658.pth`, `fold2_ep12_auc0.9688.pth`, `fold3_ep13_auc0.9692.pth`, plus available fold4 artifact). It writes `submission_gandharva_b3_raw.csv` and final `submission.csv`, with shape/finite/range guards.
- **Gandharva artifact audit:** saved `artifacts/public_kernels_20260523_frontier_candidates/gandharva_b3_artifact_audit_20260523T2000Z/`; downloaded `fold_results.json`, showing fold AUCs `0.96794`, `0.96581`, `0.96880`, `0.96925` for listed folds. This is promising structurally but not yet slot-ready until verifier completes with valid output.
- **v610 status:** SDK status is `RUNNING`, no failure message, no output/log yet. Do not submit until COMPLETE + valid finite non-constant `submission.csv`/raw output. If verifier completes before reset, v610 is the best candidate for the final slot.


## 2026-05-23 22:15 UTC — v610 submitted; day capped

- **Status at start:** v609 completed with no score due runtime timeout; best remains `0.949`; 2026-05-23 slots used `4/5`; no v577/v578 scalar submitter or stale BirdCLEF monitor process was active. Branch `feature/birdclef-20260522-v599-v602-pending`, latest commit `6f7ecdd`; PR #254 still open/review-required/BLOCKED.
- **v610 verifier result:** `yourslewis/bc26-v610-gandharva-b3-checkpoint-inference` reached COMPLETE/no failure. Outputs: `submission.csv` and `submission_gandharva_b3_raw.csv`. Log loaded all five Gandharva B3 checkpoints with zero missing/unexpected keys and wrote guarded outputs.
- **v610 preflight:** final public dry-run `submission.csv` valid `3x235`, finite/non-constant, range `0.0003119007..0.9673849`, unique-rounded first 10k `234`; raw soundscape output `submission_gandharva_b3_raw.csv` valid `12x235`, finite/non-constant, range `5.9441132e-05..0.99251634`, unique-rounded first 10k `2537`; no traceback. Preflight log: `logs/v610_gandharva_b3_preflight.txt`.
- **Submitted final slot:** v610 `Repo-owned Gandharva B3 checkpoint inference`, ref `52967355`, pending. 2026-05-23 is now capped at `5/5` visible submissions.
- **22UTC scan while capped:** saved `artifacts/public_kernels_20260523_frontier_candidates/scan_20260523T2200Z.json` and `source_audit_20260523T2200Z_newleads/summary.json`. New WildSound v8 lead (`muhammadsaadalvi/birdclef-2026-wildsound-v8`) is ERROR/no outputs; visual forks remain v608-correlated plateau family; Samejima/TTAhara HGNet training artifacts are available but prior standalone HGNet inference dropped badly (`v598=0.860`) and need a careful repo-owned anchored blend before any future slot; EoS6/PCEN and G-sidecar forks remain saturated.
- **Next:** monitor v610 score. If high, port/confirm/tune; if it drops/no-scores, diagnose hidden behavior and continue tomorrow with faster artifact-based inference/blend lanes, not direct slow training notebooks.


## 2026-05-24 00:20 UTC — v610 dropped; reset scan, no v611 slot spent

- **Status:** v610 `Repo-owned Gandharva B3 checkpoint inference` scored `0.852`, a severe drop. Best remains `0.949` vs `0.960` target. v609 remains no-score/runtime-timeout. 2026-05-24 UTC slots used `0/5` after reset; no active v577/v578 scalar submitter or BirdCLEF monitor process was found.
- **Lesson from v610:** high local fold AUC from clean/train-audio EfficientNet-B3 checkpoints is not approval evidence for hidden soundscape LB. The simple artifact inference is valid mechanically but domain-mismatched and far below plateau. Do not spend more slots on clean-audio-only checkpoint inference unless anchored/blended with strong hidden-test-aligned soundscape predictions and validated by much stronger evidence.
- **Fresh reset scan saved:** `artifacts/public_kernels_20260523_frontier_candidates/scan_20260524T0000Z.json`.
- **Fresh reset audit saved:** `artifacts/public_kernels_20260523_frontier_candidates/source_audit_20260524T0000Z_newleads/summary.json`.
- **No v611 submission:** the only fresh post-reset public movement was mostly saturated or invalid. `pilkwang/birdclef-2026-eos6-pcen-rank-sidecar` v22 was QUEUED/no output and is a PCEN/EoS6 family already tied by v604; no blind slot. `muhammadsaadalvi/birdclef-2026-wildsound-v8` remains ERROR/no outputs; root cause is `FileNotFoundError: /kaggle/input/birdclef-2026/train_metadata.csv` and source would train ConvNeXtBase in-kernel, so not direct-slot-safe. Sakur/Samejima visual forks remain v608-correlated plateau family. Samejima/TTAhara HGNet artifacts need careful anchored repo-owned blend work before another slot because v598 standalone HGNet dropped to `0.860`.
- **Decision:** preserve all five 2026-05-24 slots until a genuinely source-safe higher-upside candidate appears or a repo-owned artifact blend verifier completes with evidence. Next work should focus on fast artifact-based blends with plateau anchors, not clean-only standalone models or slow in-kernel training/TTA.

## 2026-05-24 02:20 UTC — 02UTC frontier scan, G124 manual audit, no slot spent

- **Status:** latest Kaggle submissions unchanged after reset: v610 `0.852`, v609 timeout/no score, v608 `0.949`, v607 `0.934`, v604 `0.949`. Current confirmed best remains `0.949` vs target `0.960`. 2026-05-24 UTC slots used `0/5`.
- **Repo/process:** started fresh branch `feature/birdclef-20260524-reset-frontier` from merged `origin/main` (`e07cf99`). PR #245 and PR #254 are merged. No active v577/v578 scalar submitter or BirdCLEF queue/monitor process was found.
- **Fresh scan saved:** `artifacts/public_kernels_20260523_frontier_candidates/scan_20260524T0200Z.json`.
- **Fresh source audit saved:** `artifacts/public_kernels_20260523_frontier_candidates/source_audit_20260524T0200Z_newleads/summary.json`.
- **Fresh output audit saved:** `artifacts/public_kernels_20260523_frontier_candidates/output_audit_20260524T0200Z/summary.json`.
- **G124 manual search/audit saved:** `artifacts/public_kernels_20260523_frontier_candidates/search_20260524T0200Z_g124_096.txt` and `source_audit_20260524T0200Z_g124_manual/summary.json`.
- **No v611 submission:** no candidate cleared the distinct 0.96-relevant slot bar.
- Candidate notes:
  - `karnakbaevarthur/s124-g124-reverse-engineered`: source confirms the exact missing artifacts (`g124_fold1_fp16.pt`, `submission_g124_effv2s_fold1_s124.csv`) and the private-ish path `/kaggle/input/birdclef2026-g124-effv2s-2025pre-pseudo-assets`, but kernel has no output artifacts. It is code-mining/reconstruction evidence, not direct submit-ready.
  - `anthonytherrien/birdclef-2026-s124-s114-g124-f1-blend`, `gendaijin/birdclef2026-day0522-anthony-s124`, and `karansinghbisht/bc26-pulled-rauf-s124-s114-g124-f1-rankblend`: source/output audits show the G124 fold1 sidecar fails because the assets are missing, then falls back to the S114/Model5 anchor; these are duplicate plateau/fallback slots.
  - `henryszy/bc2026-g124-protectdelta-v84`: useful protected-delta logic and NFNet sidecar code, but G124 sidecar also fails on missing fold1 assets and final dry-run keeps NFNet/anchor output. Prior NFNet/PCEN/EoS siblings tied plateau; hold unless a real artifact appears.
  - `anatoly7m/bc2026-iter-5-sed-ensemble-submit-v3`: structurally different SED ensemble but public run kept a constant `0.5` safety baseline because it found `0` test soundscapes; no direct slot.
  - `koushikkumardinda/birdclef-2026-pantanal-wetlands`: COMPLETE but only sample-shaped constant output after `0` test soundscapes; source also references a placeholder `/kaggle/input/your-trained-head/model_weights.h5`; no slot.
  - `pilkwang`/`ykuroka`/`gendaijin` EoS6+PCEN v23/v2/v1 outputs are complete but dry-run final is unchanged anchor (`final_D_vs_base_anchor: 0.0`) and v604 already tied `0.949`; no duplicate.
  - `sakur7a/birdclef-2026-visual-cpu-fork`: valid `240x235`, but final output is highly correlated with v608 (`corr≈0.993`, MAE≈0.018) and visual branch has already plateaued; hold.
- **Decision:** preserve all five 2026-05-24 slots. The best next work is not a direct public replay; it is either finding the actual G124 pseudo-assets or building a fast repo-owned anchored blend that can use diverse branch artifacts without leaving the 0.949 anchor unprotected.

## 2026-05-24 04:20 UTC — 04UTC frontier scan, Jungchan CT-MoBE held, no slot spent

- **Status:** latest submissions unchanged: v610 `0.852`, v609 timeout/no score, v608 `0.949`, v607 `0.934`, v604 `0.949`. Best remains `0.949` vs target `0.960`. 2026-05-24 UTC slots used `0/5`.
- **Memory note:** `memory_search` failed with provider error `Unknown system error -11`; direct `memory_get` from `memory/2026-05-24.md` was used to recover the 00UTC/02UTC context.
- **Repo/process:** local worktree git metadata became unhealthy after the 02UTC PR: `.git/HEAD`/refs report macOS `Resource deadlock avoided` / dataless state, and `git status` now fails. Remote PR #255 remains OPEN / REVIEW_REQUIRED / BLOCKED at `58e735b`. No v577/v578 scalar submitter or BirdCLEF monitor process was found. For any commit, use a fresh clone or repair the local worktree first.
- **Fresh scan saved:** `artifacts/public_kernels_20260523_frontier_candidates/scan_20260524T0400Z.json`.
- **Fresh source audit saved:** `artifacts/public_kernels_20260523_frontier_candidates/source_audit_20260524T0400Z_newleads/summary.json`.
- **Fresh output audit saved:** `artifacts/public_kernels_20260523_frontier_candidates/output_audit_20260524T0400Z/summary.json`.
- **No v611 submission:** no candidate cleared the distinct 0.96-relevant slot bar.
- Candidate notes:
  - `jungchanryu/birdclef-first` v18 is the freshest structurally interesting candidate. It emits a CT-MoBE/Model_21+52p+74 final dry-run `submission.csv` (`3x235`, finite) plus full `240x235` branch artifacts: `submission_custom_sed.csv`, `submission_protossm.csv`, `submission_sed.csv`, `teacher_pseudo_train_soundscapes.csv`, and `subm_21.csv`. The source is still EoS6/v6 lineage, explicitly lists model LBs `0.928/0.949/0.949`, weights `0.014/0.021/0.965`, and says v7 timed out. Public run wall time is about 948s. Hold for code-mining/anchored blend, not direct slot.
  - `pilkwang/birdclef-2026-eos6-pcen-rank-sidecar` v25 now has added `subm_karnakbayev_power_optimization*.csv` outputs, but final public dry-run remains EoS6/PCEN plateau-family and v604 already tied `0.949`; no duplicate.
  - `samejimatink0/birdclef-2026-visual-cpu-inference` remains valid but highly correlated with the v608/visual reference (`corr≈0.993`, MAE≈0.018). Hold.
  - `samejimatink0/birdclef-2026-hgnetv2-b0-baseline-inference` still emits invalid/non-numeric public `submission.csv` (`3x235`, 702 bad cells). Hold.
  - `gandharvakhedekar/birdclef2026-new` updated artifacts, but v610 already proved standalone Gandharva B3 hidden LB is poor (`0.852`); use only as a heavily anchored sidecar if ever revisited.
- **Decision:** preserve all five 2026-05-24 slots. The best next step is a repo-owned anchored blend/movement audit using Jungchan's diverse branch artifacts or continued search for actual G124 assets, not a slow direct EoS6-family replay.

## 2026-05-24 06:15 UTC — 06UTC frontier scan, G124 asset search empty, no slot spent

- **Status:** latest Bearer API submissions are unchanged: v610 `Repo-owned Gandharva B3 checkpoint inference` scored `0.852`, v609 `Guarded direct PerchFusion v951 TTA source` no-scored on runtime timeout, v608 `0.949`, v607 `0.934`, v604 `0.949`. Current confirmed best remains `0.949` vs the `0.960` target. 2026-05-24 UTC slots used: `0/5`.
- **Memory note:** `memory_search` is still unavailable with provider error `Unknown system error -11`; used the daily memory file directly for 00/02/04UTC context.
- **Repo/process:** `/Users/yourslewis/Documents/birdclef-2026-v545` git metadata still fails with `fatal: not a git repository: .../.git/worktrees/birdclef-2026-v545` and `Resource deadlock avoided` on HEAD/refs. PR #255 is OPEN / REVIEW_REQUIRED / BLOCKED; remote head already contains `f6d343e` (`Log BirdCLEF 04UTC frontier scan`). No active v577/v578 scalar submitter or BirdCLEF queue/monitor process was found.
- **Fresh scan saved:** `artifacts/public_kernels_20260523_frontier_candidates/scan_20260524T0600Z.json`.
- **Fresh source audit saved:** `artifacts/public_kernels_20260523_frontier_candidates/source_audit_20260524T0600Z_newleads/summary.json`.
- **Fresh output audit saved:** `artifacts/public_kernels_20260523_frontier_candidates/output_audit_20260524T0600Z/summary.json`.
- **G124 asset search:** Bearer dataset searches for `birdclef2026 g124`, `g124 effv2s 2025pre pseudo`, `submission_g124_effv2s`, `g124_fold1_fp16`, and `birdclef 2026 s124 g124` returned no public datasets. The actual fold1 asset path remains missing, so G124 wrappers are still reconstruction/code-mining only.
- **No v611 submission:** no candidate cleared the distinct/source-safe 0.96 bar.
- Candidate notes:
  - `minyam/birdclef-eos6-fork` is COMPLETE but public `submission.csv` is invalid (`243x235` with 56,862 bad/non-numeric cells); no slot.
  - `mins00/birdclef-2026-pcen-sidecar-fork`, `pilkwang/birdclef-2026-eos6-pcen-rank-sidecar`, `ykuroka`, and `gendaijin` PCEN/EoS6 forks are saturated plateau-family; final public submissions are sample-shaped `3x235` and v604/v608 already cover this family.
  - `sakur7a/bc2026-distilled-sed-fork` is standalone SED only (`60x235`, corr vs visual/plateau ref about `0.233`, mean near `0.015`) and not hidden-format/direct-safe enough for a slot.
  - `sakur7a/birdclef-2026-visual-cpu-fork` and `samejimatink0/birdclef-2026-visual-cpu-inference` remain valid but plateau-correlated; Samejima visual remains `corr≈0.993`, MAE≈`0.018` vs the v608/visual reference.
  - `itshyao/birdclef-2026-s116-g116-hgnet-b1-rawpseudo-all5` only produced `12x235` public output and is a G-sidecar/row-count mismatch, not direct-safe.
  - `chaneyma/bc26-gate-v22-hgnet-sidecar-rank50-base50-hg50` has interesting branch artifacts and a diverse HGNet sidecar, but final public output is `120x235` and not competition-hidden shaped; keep for repo-owned anchored blend/movement audit, not direct replay.
  - `praxel/birdclef-2026-kosuke-v15-hgnet` has full `240x235` raw blend/HGNet branch artifacts; the raw blend is still highly anchor-correlated (`corr≈0.957`, MAE≈`0.059`) and final public `submission.csv` is sample-shaped/constant. Code-mine only.
  - `chenyfdws/bc26-exp070-public0952-s124-g124-repro` does not actually contain `g124`, `submission_g124`, or the missing G124 path in source; it is an older Perch embedding-probe notebook despite the title and should not be treated as recovered G124.
  - `ttahara/birdclef-2026-hgnetv2-b0-baseline-inference` still emits invalid/non-numeric public output; `mlclsumit/notebook2e815ef354` errors on missing `best_model.pth`.
- **Decision:** preserve all five 2026-05-24 slots. Next useful work is a fast repo-owned anchored blend/movement audit over Jungchan/Chaney/Praxel raw branch artifacts, or continued discovery of real G124 assets; do not spend slots on PCEN/EoS6 duplicates, visual duplicates, clean-audio-only checkpoints, or broken HGNet direct notebooks.

## 2026-05-24 08:20 UTC — 08UTC scan + anchored movement audit, no slot spent

- **Status:** latest Bearer API submissions unchanged: v610 `0.852`, v609 timeout/no score, v608 `0.949`, v607 `0.934`, v604 `0.949`. Current confirmed best remains `0.949` vs target `0.960`. 2026-05-24 UTC slots used: `0/5`.
- **Memory/repo note:** `memory_search` is still unavailable with `Unknown system error -11`; daily memory was read directly. The main worktree `/Users/yourslewis/Documents/birdclef-2026-v545` still has broken git metadata (`fatal: not a git repository: .../.git/worktrees/birdclef-2026-v545`), so logging continues from fresh clone `/tmp/birdclef-pr255-0600`. PR #255 remote head before this log was `1a41d81` and remained OPEN / REVIEW_REQUIRED / BLOCKED. No active v577/v578 scalar submitter or BirdCLEF queue/monitor process was found.
- **Fresh scan saved:** `artifacts/public_kernels_20260523_frontier_candidates/scan_20260524T0800Z.json`.
- **Fresh source audit saved:** `artifacts/public_kernels_20260523_frontier_candidates/source_audit_20260524T0800Z_newleads/summary.json`.
- **Fresh output audit saved:** `artifacts/public_kernels_20260523_frontier_candidates/output_audit_20260524T0800Z/summary.json` plus `studyexchange_s14_summary.json`.
- **Anchored movement audit saved:** `artifacts/public_kernels_20260523_frontier_candidates/anchored_blend_audit_20260524T0800Z/`.
- **StudyExchange S14:** `studyexchange/birdclef-2026-infer-s14` v26 completed after the scan. Source says it restores `fusion_alpha`, uses BidirProtoSSM + MLP probes + ResidualSSM + Tucker/Snowflake SED, and expects `~0.943 -> 0.946+`, not 0.96. Public output is valid `240x235`, finite/non-constant, corr vs Samejima visual anchor `0.9345`, MAE `0.0677`, runtime about `669s`, dry-run OOF AUC `0.991722`. It is structurally interesting but below the current `0.949` bar by its own stated expectation, so no direct slot.
- **Beicicc/Jungchan/EoS family scan:** `beicicc/bc26-junseong-eos5-g004-inkernel-may23` is Model_7/Karnakbayev PowerOptimization LB `0.948` / single-model EoS lineage; output is sample-shaped final plus 240-row branches. `beicicc/bc26-nfnet-aves-lp075-may23` has invalid final (`243x235`, many bad cells). `anatoly7m` still writes constant `0.5`; Beicicc Anatoly/PCEN outputs are sample-shaped plateau-family. `jungchanryu/birdclef-first` v19 completed but final remains sample-shaped/constant; full 240-row branch artifacts exist (`subm_21`, `subm_52p`, `submission_protossm`, `submission_sed`), but no direct replay slot.
- **Anchored blend/movement results:** using Samejima visual/plateau output as a 0.949-family anchor and train-soundscape labels as rejection-only gate:
  - Base local macro AUC `0.9903905`, top3 row recall `0.4526` on `190` matched rows / `42` valid classes.
  - Praxel-only best: `prax_hgnet=0.06`, `prax_blend=0.02`, `prax_pc010=0.02` -> local AUC `0.9935637` (`+0.00317`), top3 `0.6263`, corr `0.99844`, MAE `0.0140`.
  - Jungchan+Praxel best: `prax_hgnet=0.06`, `jung21=0.04` -> local AUC `0.9936394` (`+0.00325`), top3 `0.6368`, corr `0.99834`, MAE `0.0150`.
  - S14 as sidecar best: `s14=0.20` -> local AUC `0.9931105` (`+0.00272`), top3 `0.4789`, corr `0.99743`, MAE `0.0141`; S14 standalone local AUC `0.9915122`.
- **Decision:** no v611 submission. The movement audit is useful, but it is still a train-soundscape/local gate and previous v560/v573/v610 showed local positives are rejection filters, not approval filters. The result supports a future repo-owned anchored blend candidate only if we can implement hidden-test inference without relying on public-output artifacts; do not submit direct S14, Beicicc, PCEN/EoS, visual duplicate, or branch-output wrappers.
- **Next:** keep watching for a source-safe 0.96 candidate or real G124 assets. If no better public source appears and slots remain idle, the next preparation task is a repo-owned hidden-safe implementation plan for the Samejima/Praxel/Jungchan anchored blend, including which models/artifacts can actually be attached and rerun on hidden test.

## 2026-05-24 10:20 UTC — 10UTC scan + hidden-safe anchored blend plan, no slot spent

- **Status:** latest Bearer API submissions unchanged: v610 `0.852`, v609 timeout/no score, v608 `0.949`, v607 `0.934`, v604 `0.949`. Current confirmed best remains `0.949` vs target `0.960`. 2026-05-24 UTC slots used: `0/5`.
- **Memory/repo/process:** `memory_search` still fails with `Unknown system error -11`; direct daily memory was used. Main worktree git remains unhealthy (`fatal: not a git repository: .../.git/worktrees/birdclef-2026-v545`). PR #255 was OPEN / REVIEW_REQUIRED / BLOCKED at remote head `b52e4d4`. No active v577/v578 scalar submitter or BirdCLEF queue/monitor process was found.
- **Fresh scan saved:** `artifacts/public_kernels_20260523_frontier_candidates/scan_20260524T1000Z.json`.
- **Fresh source audit saved:** `artifacts/public_kernels_20260523_frontier_candidates/source_audit_20260524T1000Z_newleads/summary.json`.
- **Fresh output audit saved:** `artifacts/public_kernels_20260523_frontier_candidates/output_audit_20260524T1000Z/summary.json`.
- **New/fresh candidates:**
  - `deepanshus167/bird-claasifier-comp`: COMPLETE but no output files; source contains fallback/constant markers. No slot.
  - `scottfyy/birdclef-2026-code`: COMPLETE with `best_bird_model.pth`, but public `submission.csv` is `1x235` all `0.5`; not competition-format or useful. No slot.
  - `anatoly7m/bc2026-iter-5-sed-ensemble-submit-v3`: v54 still writes sample-shaped constant `0.5`; no slot.
  - `mlclsumit/notebook2e815ef354`: still ERROR/no outputs; no slot.
  - Remaining visible leads are already-known EoS/PCEN/visual/HGNet plateau or invalid families.
- **Prepared repo-owned plan:** added `docs/BIRDCLEF_096_ANCHORED_BLEND_IMPLEMENTATION_PLAN_20260524.md`. It maps Samejima visual anchor, Praxel HGNet/raw sidecar, Jungchan Model21 sidecar, and S14 sidecar; defines candidate low-weight rank blends; and lists hidden-safe implementation/pre-submit gates.
- **Plan decision:** preferred implementation direction is not direct replay. Candidate P1 is Samejima/v608-family hidden-safe anchor + Praxel HGNet raw low-weight sidecar (`0.06`) plus optional Praxel blend/pc010 (`0.02/0.02`). Candidate P2 adds Jungchan `subm_21` (`0.04`) but is more EoS-overlap-heavy. Candidate P3 S14 is lower priority because source expectation is below current best.
- **No v611 submission:** 10UTC scan produced no source-safe direct 0.96 candidate, and the anchored blend still needs hidden-safe repo-owned implementation. Preserve all five 2026-05-24 slots.

## 2026-05-24 12:25 UTC — v611 anchored HGNet scaffold pushed for private validation; no submission

- **Status:** latest Bearer API submissions unchanged: v610 `0.852`, v609 timeout/no score, v608 `0.949`, v607 `0.934`, v604 `0.949`. Current confirmed best remains `0.949` vs target `0.960`. 2026-05-24 UTC competition slots used: `0/5`.
- **Memory/repo/process:** `memory_search` still fails with `Unknown system error -11`; direct daily memory was used. Main worktree git remains unhealthy (`fatal: not a git repository: .../.git/worktrees/birdclef-2026-v545`). PR #255 was OPEN / REVIEW_REQUIRED / BLOCKED at remote head `8a4bff1`. No active v577/v578 scalar submitter or BirdCLEF submit monitor process was found.
- **Fresh scan saved:** `artifacts/public_kernels_20260523_frontier_candidates/scan_20260524T1200Z.json`.
- **Fresh source audit saved:** `artifacts/public_kernels_20260523_frontier_candidates/source_audit_20260524T1200Z_newleads/summary.json`.
- **12UTC source triage:** no new direct 0.96/source-safe candidate appeared. `scottfyy/birdclef-2026-code` is now ERROR and only exposes `best_bird_model.pth`; its prior output was `1x235` all `0.5`. `neslihannuryilmaz/neslihan-nur-yilmaz` completed with no outputs and no useful BirdCLEF submission evidence. `mlclsumit` still errors/no outputs. Gandharva remains rejected by v610 `0.852`. Remaining EoS/PCEN/visual/HGNet leads are known plateau/invalid families.
- **Repo-owned implementation:** added `kaggle-kernels/v611-anchored-hgnet-sidecar/` and `scripts/push_v611_anchored_hgnet_sidecar.py`.
  - `script.py` preserves Samejima visual CPU inference as anchor and writes `submission_anchor_raw.csv`.
  - It reimplements a streaming Praxel/Kosuke/TTAhara OpenVINO HGNet sidecar that searches `/kaggle/input/**/best_model_fold0.xml`, requires all four `best_model_fold*.xml/.bin`, reruns on the same anchor row IDs, and writes `submission_prax_hgnet_raw.csv`.
  - Final candidate blend is intentionally conservative: `0.94 * rank(anchor) + 0.06 * rank(prax_hgnet_raw)`.
  - Diagnostics: `submission_before_alignment.csv` and final `submission.csv`; no public-output CSV artifacts are used.
- **Local validation:** `python3 -m py_compile` passed for `script.py` and the push script; AST parse passed; `kernel-metadata.json` JSON parse passed.
- **Kaggle private validation push:** pushed private kernel `yourslewis/bc26-v611-anchored-hgnet-sidecar`, version 1, kernel id `120423812`. Push returned no invalid dataset/competition/kernel/model sources. This is **not** a competition submission and spent no daily submission slot.
- **Validation status at handoff:** v611 is still `RUNNING`, with no output files or log exposed yet. Do not submit until it reaches COMPLETE, branch outputs exist, final `submission.csv` passes shape/finite/nonconstant checks, and runtime/logs confirm the hidden-safe path.
- **Decision:** no v611 competition submission yet. Preserve all five slots. Next loop should inspect v611 completion; if COMPLETE and valid, decide whether the hidden-safe anchored HGNet sidecar is worth one distinct daily slot or needs further runtime/branch-weight adjustment.

## 2026-05-24 14:20 UTC — v611 submitted after validation; 14UTC scan no better direct source

- **Status:** before submission, latest Bearer API submissions were unchanged: v610 `0.852`, v609 timeout/no score, v608 `0.949`, v607 `0.934`, v604 `0.949`; 2026-05-24 UTC slots used `0/5`.
- **v611 private validation result:** `yourslewis/bc26-v611-anchored-hgnet-sidecar` v1 reached COMPLETE/no failure. Outputs include `submission.csv`, `submission_anchor_raw.csv`, `submission_before_alignment.csv`, `submission_prax_hgnet_raw.csv`, `submission_protossm.csv`, and `submission_sed.csv`.
- **v611 log/runtime:** public dry-run completed in about `715s`. Samejima anchor completed and was preserved as `submission_anchor_raw.csv (240,235)`; HGNet OpenVINO artifacts resolved at `/kaggle/input/datasets/skidive/no-wav-use`; HGNet sidecar processed 20 dry-run train audio files / 240 rows across 4 folds in about `82s` after the anchor. Final log: `v611 anchored rank blend: anchor=0.940, prax_hgnet=0.060`, `submission.csv shape=(240,235)`, `nonconstant_cols=234/234`.
- **v611 output validation:** downloaded outputs to `artifacts/kaggle_outputs/v611-anchored-hgnet-sidecar/`. Final `submission.csv`: `240x235`, 240 unique row IDs, no bad values, range `0.004166667..1.0`, unique-rounded first 10k `7204`. Praxel sidecar: `240x235`, no bad values, range `1.94e-07..0.9313675`, unique-rounded first 10k `2759`. Rank-space final vs anchor corr `0.99852`, MAE `0.01369`; sidecar rank vs anchor rank corr `0.47472`, MAE `0.22811`. Local rejection-gate metric: anchor raw AUC `0.9903905`; v611 final AUC `0.9935681`; HGNet raw AUC `0.9935998` on 190 matched rows / 42 valid classes.
- **Submitted:** guarded submitter `scripts/submit_v611_anchored_hgnet_when_ready.py` passed preflight and submitted v611 as `v611: Repo-owned Samejima anchor plus Praxel HGNet sidecar`, ref `52988938`. The submission is currently `pending` with no public score yet. 2026-05-24 UTC slots used now `1/5`.
- **Fresh 14UTC scan saved:** `artifacts/public_kernels_20260523_frontier_candidates/scan_20260524T1400Z.json`.
- **Fresh 14UTC source audit saved:** `artifacts/public_kernels_20260523_frontier_candidates/source_audit_20260524T1400Z_newleads/summary.json`.
- **14UTC source triage:** no better direct source-safe 0.96 candidate appeared. Samejima HGNet training v57 is RUNNING/no outputs; Scott is complete again but prior output was all-0.5 and current source still has fallback markers; Deepanshu has no outputs; Mins00/PCEN, Beicicc/Jungchan/EoS, Pilkwang, visual, and gated/HGNet forks remain known plateau/invalid families.
- **Decision:** do not submit anything else until v611 scores or a materially stronger source-safe candidate appears. Preserve remaining `4/5` slots.

## 2026-05-24 16:25 UTC — v611 scored tied best; 16UTC source/output audit favors no new slot

- **Status:** Bearer API submissions now show v611 `Repo-owned Samejima anchor plus Praxel HGNet sidecar` complete with public LB `0.949`. Best remains `0.949`; target remains `0.960`. v610 remains `0.852`; v609 remains runtime/no-score; v608/v604 remain `0.949`. 2026-05-24 UTC slots used `1/5`.
- **Lesson:** v611 proves another locally-plausible, diverse HGNet OpenVINO sidecar can preserve the 0.949 anchor but not lift it. This reinforces the public946/sidecar lesson: train-soundscape/local gates are rejection filters, not approval filters. Do not spend slots on low-weight sidecar additions unless they introduce genuinely new hidden-test-safe signal or a public source implies a higher LB lineage.
- **Process/repo:** main clone at `/Users/yourslewis/Documents/birdclef-2026-v545` still has unreadable `.git`; work continues from `/tmp/birdclef-pr255-0600`. PR #255 is open/review-required/blocked, head `b4db095`. No v577/v578 scalar submitter or active BirdCLEF submit monitor found.
- **Fresh 16UTC scan saved:** `artifacts/public_kernels_20260523_frontier_candidates/scan_20260524T1600Z.json`.
- **Fresh 16UTC source audit saved:** `artifacts/public_kernels_20260523_frontier_candidates/source_audit_20260524T1600Z_newleads/summary.json`.
- **Fresh 16UTC output audit saved:** `artifacts/public_kernels_20260523_frontier_candidates/output_audit_20260524T1600Z/summary.json`.
- **New/source findings:**
  - `alexycactus/birdclef-2026-ns1-ensemble`: structurally different Noisy Student CNN ensemble plus Perch/MLP; source comment claims nb21 LB `0.922` and mean OOF `0.9745`. Dry-run output is 192 rows, valid/nonconstant; correlation with v611 anchor is low (`~0.263`), but standalone public clue is far below 0.949, so it is candidate material for a guarded repo-owned sidecar only, not direct submission.
  - `raunakdey07/birdclef-2026-v9`: final `submission.csv` is only 3-row power-optimization fallback in dry-run; branch artifacts `submission_protossm.csv`/`submission_sed.csv` are valid 240-row dry-run outputs but are known ProtoSSM/SED lineage, not a direct source-safe 0.96 candidate.
  - `samejimatink0/birdclef-2026-hgnetv2-b0-baseline-inference`: public `submission.csv` has only 3 rows with nonnumeric/bad values; `ghnet_test.csv` is 120-row HGNet artifact. Samejima HGNet training v57 remains RUNNING/no outputs.
  - `scottfyy/birdclef-2026-code`: output `submission.csv` is all `0.5`; reject as fallback.
  - `mins00/birdclef-2026-pcen-sidecar-fork`/PCEN/EoS6 family: complete but dry-run final is 3 rows and lineage is already plateau-covered by v604/v608-style tied-best results.
- **Decision:** do not spend a second 2026-05-24 slot from this scan. Preserve `4/5` remaining slots. Next plausible implementation lane is a private, no-slot repo-owned v612 feasibility scaffold using the Alexy NS1 CNN as a low/medium-weight anchored sidecar, but only if it can pass runtime/output validation and show stronger evidence than previous sidecar ties.

## 2026-05-24 18:25 UTC — Samejima HGNet-v57 PT candidate discovered; v612 private validation started

- **Status:** latest submissions unchanged: v611 `0.949`, v610 `0.852`, v609 runtime/no-score, v608/v604 `0.949`. Best remains `0.949`; target `0.960`; 2026-05-24 UTC slots used `1/5`.
- **Repo/process:** main clone `.git` still unreadable; continued in `/tmp/birdclef-pr255-0600`. PR #255 open/review-required/blocked. No v577/v578 scalar submitter or active BirdCLEF submit monitor found.
- **Fresh 18UTC scan saved:** `artifacts/public_kernels_20260523_frontier_candidates/scan_20260524T1800Z.json`.
- **Fresh 18UTC source audit saved:** `artifacts/public_kernels_20260523_frontier_candidates/source_audit_20260524T1800Z_newleads/summary.json`.
- **New actionable finding:** `samejimatink0/birdclef-2026-hgnetv2-b0-baseline-training` v57 moved from RUNNING to ERROR, but preserved usable outputs: `best_model_fold0..3.pt`, `best_val_pred_fold0..3.npy`, and `result_df_fold0..3.csv`.
  - Fold best val scores: fold0 `0.958302`, fold1 `0.960042`, fold2 `0.968769`, fold3 `0.965945`.
  - Root cause of public kernel ERROR: OOF aggregation shape mismatch (`best_val_pred_fold0.npy` shape `(9075,234)` vs `val_idxs` shape `(9084,234)`), after the PT checkpoints were already saved. No OpenVINO artifacts were produced.
  - Treat as new structural asset, but not approval evidence by itself because v610 and v611 showed local/CV sidecar evidence can fail to lift hidden LB.
- **Implemented no-slot repo-owned verifier:** `kaggle-kernels/v612-anchored-sameji-hgnet57-pt/`.
  - Preserves Samejima visual anchor as `submission_anchor_raw.csv`.
  - Discovers mounted Samejima v57 `best_model_fold*.pt` checkpoints under `/kaggle/input`.
  - Rebuilds the HGNetV2-B0 LSE model from the public training source, reruns 4 folds on the same anchor row IDs, writes `submission_sameji_hgnet57_raw.csv`, then final rank blend `0.94 anchor + 0.06 Samejima HGNet-v57`.
  - Hard-fails on missing assets, schema drift, non-finite/constant output, or row misalignment.
- **Validation:** py_compile + AST + metadata JSON passed. Pushed private Kaggle validation kernel `yourslewis/bc26-v612-anchored-sameji-hgnet57-pt` v1, kernel id `120456720`; no invalid data/kernel/model sources. It is currently RUNNING with no outputs/log yet. No competition submission was made.
- **Other 18UTC leads:** Tulay EfficientNet weights still RUNNING/no outputs; Claudedevore R0952 teacher/hybrid train produced pseudo sidecar checkpoints but no hidden-test inference path; Deepanshu still no outputs; EoS6/PCEN/HGNet forks remain known plateau or fallback families.

## 2026-05-24 20:25 UTC — v612 validated/submitted; 20UTC scan no stronger direct source

- **Status before submission:** latest submissions showed v611 `0.949`, v610 `0.852`, v609 runtime/no-score, v608/v604 `0.949`; best remained `0.949`; target `0.960`; 2026-05-24 UTC slots used `1/5`.
- **PR/repo:** PR #255 was merged by the time of this loop; new work continued from `origin/main` on branch `feature/birdclef-20260524-20utc-v612-submit`. Main local clone `.git` remains unreadable, so `/tmp/birdclef-pr255-0600` remains the working clone.
- **v612 private validation result:** `yourslewis/bc26-v612-anchored-sameji-hgnet57-pt` v1 reached COMPLETE/no failure. Outputs include `submission.csv`, `submission_anchor_raw.csv`, `submission_before_alignment.csv`, `submission_sameji_hgnet57_raw.csv`, `submission_protossm.csv`, `submission_sed.csv`, and `submission_birdnet.csv`.
- **v612 runtime/log:** public dry-run completed in about `813s`. Samejima visual anchor completed by about `685s`; v612 sidecar found the v57 PT checkpoints under `/kaggle/input/notebooks/samejimatink0/birdclef-2026-hgnetv2-b0-baseline-training`, processed 20 dry-run train audio files / 240 rows across 4 folds in about `106s`, and wrote final `0.94/0.06` rank blend. Log markers: `v612 wrote submission_sameji_hgnet57_raw.csv`, `corr=0.502131`, `mae=0.220846`, final `submission.csv shape=(240,235)`, `nonconstant_cols=234/234`.
- **v612 output validation:** downloaded outputs to `artifacts/kaggle_outputs/v612-anchored-sameji-hgnet57-pt/`. Final `submission.csv`: `240x235`, 240 unique row IDs, no bad values, range `0.004166667..1.0`, unique-rounded first 10k `7186`, all 234 class columns nonconstant. Samejima HGNet-v57 raw: `240x235`, no bad values, range `1.97e-06..0.7989003`, unique-rounded first 10k `4108`. Final vs anchor raw corr `0.98707`, MAE `0.03915`; final rank vs anchor rank corr `0.99879`, MAE `0.01109`; sidecar raw rank vs anchor rank corr `0.50213`, MAE `0.22085`.
- **v612 local rejection-gate comparison:** on train-soundscape dry-run rows with local primary labels (190 matched rows / 11 valid classes), Samejima visual anchor AUC `0.93315`, v611 final AUC `0.93895`, v611 Praxel HGNet raw AUC `0.95096`; v612 final AUC `0.94089`, v612 Samejima HGNet-v57 raw AUC `0.96489`. This is stronger than v611 locally, but still only rejection-filter evidence.
- **Submitted:** guarded submitter `scripts/submit_v612_anchored_sameji_hgnet57_when_ready.py` passed preflight and submitted v612 as `v612: Repo-owned Samejima anchor plus HGNet-v57 PT sidecar`, ref `52998418`. Status currently `pending`; 2026-05-24 UTC slots used now `2/5`.
- **Fresh 20UTC scan saved:** `artifacts/public_kernels_20260523_frontier_candidates/scan_20260524T2000Z.json`.
- **Fresh 20UTC source audit saved:** `artifacts/public_kernels_20260523_frontier_candidates/source_audit_20260524T2000Z_newleads/summary.json`.
- **Fresh 20UTC focused output audit saved:** `artifacts/public_kernels_20260523_frontier_candidates/output_audit_20260524T2000Z_focus/summary.json`.
- **20UTC source triage:** no better direct source-safe candidate appeared. `tulayppppp/my-efficientnet-b0-weights` completed with `2x207` all-zero/mock output and no hidden-test path; reject. `claudedevore` R0952 teacher/hybrid train outputs are training checkpoints/pseudo manifests only, no hidden-test inference path yet; hold for code-mining. `raunak` remains 3-row fallback final; EoS6/PCEN/visual/HGNet forks are known plateau/fallback families.
- **Decision:** after v612 submission, hold remaining `3/5` slots until v612 scores or a materially stronger source-safe candidate appears.

### 2026-05-24 22:30 UTC — v612 score + 22UTC frontier scan, no slot spent

- **Live LB/submission state:** `v612` (`Repo-owned Samejima anchor plus HGNet-v57 PT sidecar`, ref `52998418`) scored `0.949`, tying the current best but giving no lift. `v611` also remains `0.949`; `v610` is rejected at `0.852`; `v609` timed out/no score. Current confirmed best remains **0.949** vs target **0.960**. 2026-05-24 UTC submissions used: `2/5`, so `3` estimated slots remain.
- **Repo/process state:** original Documents clone is still unreadable (`fatal: error reading .../.git`), so canonical active work moved to `/Users/yourslewis/.openclaw/repos/birdclef-2026` on branch `feature/birdclef-20260524-20utc-v612-submit`. PR #245 is merged; PR #256 is open/blocked for v612. No stale v577/v578 scalar submitter was active; no BirdCLEF submit monitor was running.
- **Lesson:** v612 reinforces the v611/public946-sidecar lesson: train/public dry-run gates and sidecar local AUC are useful rejection filters, not approval filters. A strong HGNet-v57 sidecar improved local overlap versus v611 but only tied hidden public LB.
- **Chosen track:** A — public/source frontier scan and preflight, preserving remaining slots for distinct 0.96-relevant hypotheses.
- **Artifacts:** saved ignored scan/audit artifacts under `artifacts/public_kernels_20260524_frontier_candidates/scan_20260524T2200Z.json` and `source_audit_20260524T2200Z_newleads/summary.json`.
- **New/fresh candidate triage:**
  - `muhammadsaadalvi/birdclef-2026-wildsound-v8`: latest run still ERROR; log tail shows `FileNotFoundError: /kaggle/input/birdclef-2026/train_metadata.csv`. Not hidden/test safe; no slot.
  - `tulayppppp/my-efficientnet-b0-weights`: ERROR/mock path, output is tiny/fallback (`submission.csv` around 852 bytes) and log says `MOCK MODU`; no slot.
  - `samejimatink0/birdclef-2026-hgnetv2-b0-baseline-inference`: COMPLETE but output/log indicate train-row/dry-run behavior (`submission.csv` printed `(3,235)`, `ghnet_test.csv` `(120,235)`); not a competition-format direct candidate. Existing repo-owned v612/v598 already tested Samejima HGNet ideas.
  - `alexycactus/birdclef-2026-ns1-ensemble`: structurally different CNN+Perch/noisy-student source, but source header says prior `LB 0.922`, public output writes `(192,235)` rather than the expected 240-row dry-run shape, and OOF/Perch AUC evidence is weak (`Perch 0.7478`, logit on `0.4913`). Idea-mining only; no direct slot.
  - `raunakdey07/birdclef-2026-v9`, `deepanshus167/bird-claasifier-comp`, Claudedevore R0952 train notebooks, and known PCEN/EoS6/rank-fusion families remain rejected/held from earlier audits (fallback/train-only/plateau lineage).
- **Decision:** no v613 submission at 22UTC. Spending one of the 3 remaining slots on these would be inconsistent with the 0.960 target; continue source discovery and repo-owned extraction only when a candidate passes hidden-safety/output preflight.

### 2026-05-24 22:55 UTC — Good-ideas backlog restored

- Corrected the post-v612 framing: public-source replay was exhausted for 22UTC, but the research queue is **not** exhausted. If no source-safe public kernel is slot-worthy, the loop must pivot to no-slot repo-owned validation lanes rather than waiting.
- Added `docs/BIRDCLEF_GOOD_IDEAS_BACKLOG_20260524.md` with a concrete backlog and promotion gates.
- Top immediate no-slot ideas:
  1. Alexy NS1 CNN/noisy-student sidecar extraction as hidden-safe repo-owned branch, not direct replay.
  2. G124 EffV2-S reconstruction with external/V2S init, not scratch-only retry.
  3. Unified anchored sidecar validation harness across Praxel/Samejima/S14/Jungchan/Alexy outputs with file/site bootstrap.
- Verified local repo scripts compile for key lanes (`birdclef_sed_smoke.py`, `birdclef_sed_pilot_train.py`, `birdclef_pseudolabel_student_train.py`, `birdclef_public946_multi_sidecar_weight_grid.py`), and key configs parse (`g124_effv2s_public946_pseudo_smoke_20260523.json`, `sed_b0_5s_attn_smoke.json`, `pl_public946_sed_b0_5s_lr3e4_smoke.json`). Local Mac python lacks pandas for execution, but trainer venv `~/kaggle_envs/s6e3` has pandas/torch/timm/sklearn/numpy. Trainer `~/birdclef-2026` exists but is not a git repo, so sync intentionally before long runs.

### 2026-05-24 23:10 UTC — Two-day good-ideas sprint spec

- Added `docs/BIRDCLEF_TWO_DAY_EXPERIMENT_SPEC_20260524.md` to prevent the loop from idling after public replay exhaustion.
- Spec confirms enough good ideas for today/tomorrow and defines slot policy, no-slot validation gates, and concrete work items.
- Today focus: Alexy NS1 sidecar skeleton/private verifier, unified anchored sidecar manifest, and G124/V2S-init config prep.
- Tomorrow focus: finish Alexy verifier and grid; run G124 EffV2-S V2S-init smoke on trainer; run SED/export smoke if needed; start pseudo-label threshold redesign or per-class residual selector as fallback lanes.
- Slot rule: preserve remaining 2026-05-24 slots unless a private/no-slot candidate passes promotion gates; on 2026-05-25 use at most two promoted submissions and keep reserve slots.

### 2026-05-24 22:35 UTC — used remaining UTC slots as legitimate exploratory candidates

- **User direction:** use the remaining three 2026-05-24 UTC slots properly, but do not do test-set probing. Interpreted as real candidate submissions that can plausibly teach model/lineage signal, not artificial diagnostic/probing submissions.
- **Live before action:** v612 and v611 both scored `0.949`; current best remained `0.949`; 2026-05-24 UTC slots used `2/5`, remaining `3/5` with about 1.5h before reset.
- **Added guarded batch submitter:** `scripts/submit_v613_v615_exploratory_slots_20260524.py`. It checks duplicate submissions, live UTC cap, source markers, COMPLETE/no-failure status, required outputs, numeric/row guards, and branch output guards before submitting.
- **Submitted v613:** `v613: Exploratory direct Alexy NS1 CNN noisy-student source`, ref `53000944`, source `alexycactus/birdclef-2026-ns1-ensemble` v1. Rationale: distinct CNN/noisy-student family. Public dry-run output is valid/nonconstant `192x235`, but source handles `test_soundscapes`; risk is below-frontier source header (`LB 0.922`) and row-count behavior. Use result as family signal, not as probe.
- **Submitted v614:** `v614: Exploratory direct Raunak v9 ProtoSSM SED source`, ref `53000945`, source `raunakdey07/birdclef-2026-v9` v4. Rationale: legitimate Model_7/ProtoSSM/SED public family with hidden-test handling; public final is sample-shaped but branch outputs `submission_protossm.csv`/`submission_sed.csv` are valid full `240x235` and nonconstant. Likely plateau but useful baseline.
- **Submitted v615:** `v615: Exploratory direct Jungchan CT-MoBE branch source`, ref `53000949`, source `jungchanryu/birdclef-first` v19. Rationale: CT-MoBE/Model21 plus ProtoSSM/SED branch diversity. Public final is sample-shaped/low-diversity, so preflight required full nonconstant branch outputs (`subm_21.csv`, `submission_protossm.csv`, `submission_sed.csv`) and hidden-test markers. Saturated lineage risk, but legitimate exploratory slot.
- **After action:** 2026-05-24 UTC cap is now `5/5`. v613-v615 are PENDING; v611/v612 remain complete at `0.949`. These were not artificial leaderboard probes; they were real candidate/model-family tests with guarded preflight.
- **Next:** monitor v613-v615 scores. If any improve/tie meaningfully, port/confirm repo-owned or add to the anchored sidecar harness. If they drop/no-score, update the lesson and continue tomorrow with Alexy sidecar extraction, G124 V2S-init smoke, and unified sidecar validation.

### 2026-05-25 00:15 UTC — v613-v615 scored; G124 V2S-init smoke completed

- **Live LB/submission state:** 2026-05-25 UTC reset is active with `0/5` submissions used. v613-v615 from the final 2026-05-24 slots have scored: v613 Alexy NS1 direct `0.923` (reject direct), v614 Raunak v9 ProtoSSM/SED `0.949` (plateau tie), v615 Jungchan CT-MoBE branch source `0.949` (plateau tie). Current confirmed best remains **0.949** vs target **0.960**.
- **PR/repo/process state:** active branch `feature/birdclef-20260524-20utc-v612-submit`; PR #256 remains open / review-required / blocked. No local BirdCLEF monitors were active; trainer had no BirdCLEF job before this run.
- **Chosen track:** C — G124 EffV2-S reconstruction with external/V2S init. Reason: v613-v615 direct public exploration did not improve; G124/V2S-init remains one of the few high-upside no-slot lanes and scratch G124 had previously failed.
- **Config created:** `configs/birdclef/g124_effv2s_public946_pseudo_smoke_20260525_v2sinit.json`. It copies the prior G124 smoke but switches to `backbone=efficientnetv2_rw_s`, sets `initial_checkpoint=artifacts/external_pretrain/xc-v2s-q3-cap80-external-pretrain-balanced-ep12-bestloss/model_torchscript.pt`, keeps `max_rows=384`, and writes to `artifacts/pseudolabels/students/g124-effv2s-public946-pseudo-smoke-20260525-v2sinit`.
- **Trainer run:** synced the config to trainer and ran under `~/kaggle_envs/s6e3` with log `logs/g124_v2sinit_smoke_20260525T0005Z.log`. The run completed in about `29s`, exported TorchScript and ONNX, and loaded `786` keys from the V2S external-pretrain TorchScript checkpoint (head skipped intentionally).
- **G124 V2S-init smoke metrics:** best epoch `4`; val student-vs-truth macro AUC improved from epoch1 `0.8271` to epoch4 `0.956867` over `44` valid classes. Final-all student macro AUC `0.962116` over `48` valid classes; teacher macro AUC `0.995714`; student/teacher corr `0.830097`; MAE `0.033398`. This clears the smoke continuation gate (`>=0.90–0.93`) and is dramatically better than the scratch G124 smoke (`0.726`), but still below teacher.
- **Blend audit:** standard `birdclef_student_pool_blend_audit.py` skipped the new artifact because the smoke used `384` rows while the teacher cache has more rows. A custom intersection audit was saved on trainer at `artifacts/pseudolabels/audits/g124_v2sinit_smoke_intersection_blend_20260525T0005Z.json`: on 384 intersecting rows / 48 valid classes, teacher AUC `0.995714`, student standalone `0.962116`, best tiny blend was weight `0.01` with lift only `+0.0000021`; weights `>=0.02` hurt. Treat this as a good training/export smoke but **not** a slot candidate yet.
- **Sidecar manifest:** created ignored manifest `artifacts/anchored_blend_audit/sidecar_manifest_20260525T0000Z.json` summarizing v613-v615 lessons and available sidecar branches for the unified anchored grid.
- **Next:** do not submit G124 smoke. Next exact action is either (1) run a larger/all-row V2S-init G124 pilot only if we want to see whether the tiny +0.01 blend survives more rows, or (2) prioritize unified anchored sidecar grid/Alexy sidecar extraction because direct Alexy failed but sidecar mining remains possible.

### 2026-05-25 00:30 UTC — all-row G124 V2S-init pilot completed

- **User direction:** chose option `1` from the prior decision point: run a larger/all-row G124 V2S-init pilot.
- **Config created:** `configs/birdclef/g124_effv2s_public946_pseudo_pilot_20260525_v2sinit_allrows_ep8.json`. It uses the full `792`-row public946 teacher cache (`max_rows=null`), `backbone=efficientnetv2_rw_s`, V2S external-pretrain TorchScript init, `epochs=8`, `lr=1e-4`, and exports TorchScript+ONNX. Output dir: `artifacts/pseudolabels/students/g124-effv2s-public946-pseudo-pilot-20260525-v2sinit-allrows-ep8`.
- **Trainer run:** synced config to trainer and ran under `~/kaggle_envs/s6e3`; log `logs/g124_v2sinit_allrows_ep8_20260525T0025Z.log`. Runtime `29.448s`. Outputs: `student_predictions.npz`, `model_torchscript.pt`, `model.onnx`, `model.onnx.data`, `metrics.json`, `training_log.jsonl`.
- **Training metrics:** best epoch `7`, best val macro AUC `0.947911` over `59` valid classes. Final-all student macro AUC `0.947190` over `75` valid classes; teacher macro AUC `0.997018`; student/teacher corr `0.878257`; MAE `0.031810`. This is much better than scratch G124 but materially worse than the teacher and worse than the smaller smoke's subset AUC.
- **Blend audit:** saved `artifacts/pseudolabels/audits/g124_v2sinit_allrows_ep8_blend_audit_20260525T0025Z.json` on trainer. The artifact row/label aligned with the teacher cache (`n_aligned=1`). Best blend is tiny: student weight `0.0025`, macro AUC `0.99701911`, lift only `+0.00000066`, corr `0.99999937`. Stability is weak/rejection-only: site bootstrap q05 lift `-0.00000282`, leave-one-site q05 `-0.00000027`, and 2/9 leave-one-site groups are negative. Weights `>=0.005` mostly hurt, and `0.01` is already negative.
- **Decision:** all-row G124 V2S-init pilot is a successful training/export experiment but **not** a Kaggle submission candidate. It does not justify spending a slot; the useful lesson is that V2S init fixes scratch training but still does not add enough independent signal to beat the public946 teacher. Next action should pivot to unified anchored sidecar grid/Alexy sidecar extraction or a different data/architecture change rather than scaling this exact setup.

### 2026-05-25 02:05 UTC — unified anchored sidecar grid found a strong no-slot candidate

- **Live LB/submission state:** 2026-05-25 UTC has `0/5` submissions used. Latest scored results remain v613 Alexy direct `0.923`, v614 Raunak v9 `0.949`, v615 Jungchan CT-MoBE `0.949`, v611/v612 `0.949`; confirmed best remains **0.949**. No active BirdCLEF monitors/training/private-verifier processes were found locally or on trainer.
- **Chosen track:** B — unified anchored sidecar validation harness. Reason: G124 V2S-init all-row pilot was not a slot candidate; the spec's next high-value no-slot work is to compare the available v614/v615 branch outputs and plateau-family sidecars under one protocol before any 2026-05-25 slot.
- **Inputs downloaded:** saved public dry-run branch CSVs under `artifacts/anchored_blend_audit/sidecar_grid_inputs_20260525T0200Z/` and copied them to trainer: Samejima visual anchor, Samejima ProtoSSM/SED, Sakur visual/ProtoSSM, Jungchan Model21/ProtoSSM, Raunak ProtoSSM/SED. Each downloaded CSV was `240x235`.
- **Grid run:** copied current `scripts/birdclef_public946_multi_sidecar_weight_grid.py` plus dependency to trainer and ran a no-submit grid against `data/train_soundscapes_labels.csv`. Fast grid artifact: `artifacts/anchored_blend_audit/sidecar_grid_20260525T0200Z_fast.json`; stability artifact: `artifacts/anchored_blend_audit/sidecar_grid_20260525T0200Z_top_stability.json`. Results were copied back to the neutral repo artifact directory (ignored by git).
- **Base:** Samejima visual anchor local matched-row AUC `0.9903905` on `190` matched rows / `42` valid classes.
- **Best fast-grid candidate:** rank blend `0.90*Samejima visual anchor + 0.02*Sakur visual + 0.04*Jungchan Model21 + 0.04*Raunak SED` (no Raunak ProtoSSM). Local AUC `0.9935362`, lift `+0.0031457`; corr vs anchor `0.999641`; MAE `0.006713`; top3 row recall improved from anchor's ~`0.4526` to `0.4895`.
- **Stability for best candidate:** site bootstrap `1000` iters: mean lift `+0.00498`, q05 lift `+0.00181`, p(lift>0)=`0.999`, min lift `0.0`. Leave-one-site: all `6/6` held-out sites positive, q05 `+0.002735`, min `+0.002723`, max `+0.007064`. This is much stronger as a rejection screen than the G124 V2S-init pilot.
- **Decision:** still **no competition submission this run**. This is train-soundscape/local evidence and local gates have failed before. But the candidate is strong enough to justify the next no-slot step: package a repo-owned private verifier candidate (proposed v616) that reruns Samejima anchor plus branch logic/available public sources hidden-safely, writes raw branch outputs, and validates runtime/schema before considering a 2026-05-25 slot.

### 2026-05-25 04:10 UTC — v616 hidden-safe branch extraction / feasibility audit

- **Live state:** 2026-05-25 UTC still has `0/5` submissions used. Latest scored remain v613 `0.923`, v614/v615/v611/v612 `0.949`; confirmed best remains **0.949**. PR #256 remains open/review-required/blocked. No active BirdCLEF jobs were running locally or on trainer.
- **Chosen track:** continue B — turn the strong 02UTC unified sidecar grid into a hidden-safe v616 private verifier path, while avoiding static public-output blending.
- **Important guardrail:** no v616 competition submission this run. A kernel that reads public dry-run CSV outputs from other kernels would not rerun on hidden `test_soundscapes`, so it is not promotion-safe.
- **Source/branch audit:** pulled decoded sources for Samejima visual, Jungchan, Raunak v9, and Sakur visual into `artifacts/source_audits/v616_hidden_safe_branch_extraction_20260525T0400Z/` (ignored artifact). Wrote summary/excerpts for branch extraction.
- **Key simplification:** Samejima SED and Raunak SED dry-run branch outputs are exactly identical on the audited `240x235` rows (`maxabs=0`, corr≈1). Therefore the v616 candidate does **not** need to import Raunak source; it can reuse the Samejima/v612 SED branch.
- **Sakur-free candidate remains strong:** the 02UTC grid without Sakur visual, using `0.92*Samejima visual + 0.04*Jungchan Model21 + 0.04*Samejima/Raunak SED`, still had local AUC about `0.9934807` (`+0.00309`) and avoids another full visual-family source import.
- **Jungchan extraction:** added helper `scripts/extract_v616_jungchan_model21_block.py`. It extracts the Jungchan Model21 block from decoded source lines `1356..9736` into `artifacts/source_audits/v616_hidden_safe_branch_extraction_20260525T0400Z/jungchan_model21_block.py.txt`. The slice is `8381` lines / `367451` chars and contains `subm_21.csv`, `test_soundscapes`, `sample_submission.csv`, `perch_v2.onnx`, `ProtoSSM`, `ResidualSSM`, and writer/diagnostic markers.
- **Feasibility notes:** added `kaggle-kernels/v616-anchored-jung21-sed-blend/FEASIBILITY_NOTES.md`. Proposed hidden-safe v616 target is `0.92*Samejima visual anchor + 0.04*Jungchan Model21 + 0.04*Samejima SED`, starting from the v612 scaffold for Samejima anchor/SED and adding a cleaned Model21 function. The private verifier must write `submission_anchor_raw.csv`, `submission_samejima_sed_raw.csv`, `submission_jung21_raw.csv`, `submission_before_alignment.csv`, and final `submission.csv`; hard-fail on missing/misaligned/constant/nonfinite outputs.
- **Blocker:** the Jungchan Model21 slice is large and depends on notebook globals/setup. Blindly pasting it into v612 risks silent conflicts. Next exact step is to clean that slice into a function and push a private v616 verifier; only then consider a 2026-05-25 slot.

### 2026-05-25 06:10 UTC — v616 private verifier implemented and pushed

- **Live state:** 2026-05-25 UTC still has `0/5` submissions used. Latest scores unchanged: v613 `0.923`, v614/v615/v611/v612 `0.949`; confirmed best remains **0.949**. PR #256 remains open/review-required; no active BirdCLEF jobs were running locally or on trainer at start.
- **Chosen track:** implement the v616 private verifier from the 04UTC feasibility plan. No competition submission was attempted.
- **Implementation:** created `kaggle-kernels/v616-anchored-jung21-sed-blend/script.py` by taking the Samejima/v612 scaffold through the Samejima anchor/SED writer, preserving `submission_anchor_raw.csv` and `submission_samejima_sed_raw.csv`, then appending a cleaned Jungchan Model21-only source slice (stops before Model22), and finally writing a fixed rank blend `0.92*Samejima visual anchor + 0.04*Jungchan Model21 + 0.04*Samejima SED`.
- **Guards in script:** v616 does not read public output CSVs. It reruns Samejima anchor/SED and Jungchan Model21 on the current Kaggle mount. It writes `submission_anchor_raw.csv`, `submission_samejima_sed_raw.csv`, `submission_jung21_raw.csv`, `submission_before_alignment.csv`, and final `submission.csv`; hard-fails on missing row_id/classes, row misalignment, nonfinite values, constant branches, or constant final columns.
- **Validation before push:** `python3 -m py_compile` passed for `kaggle-kernels/v616-anchored-jung21-sed-blend/script.py`; metadata JSON validation passed. Script size is about `6021` lines / `232893` chars.
- **Pushed private verifier:** `yourslewis/bc26-v616-anchored-jung21-sed-blend`, version `1`, kernel id `120505284`; Kaggle push returned no invalid dataset/competition/kernel/model sources. Initial status check after push: RUNNING, no failure message, no outputs yet.
- **Decision:** no competition slot. Wait for v616 private verifier to COMPLETE and then validate outputs/log/runtime before any submit decision. If v616 fails due Jungchan slice/global conflicts, next action is to inspect its log and either repair the slice or fall back to a simpler Samejima anchor + SED-only/Sakur-free branch candidate.

### 2026-05-25 08:05 UTC — v616 validated and submitted

- **Live state before action:** 2026-05-25 UTC had `0/5` submissions used. Latest scored remained v613 `0.923`, v614/v615/v611/v612 `0.949`; confirmed best remained **0.949**. PR #256 is now MERGED/approved; active branch still `feature/birdclef-20260524-20utc-v612-submit`. No active BirdCLEF jobs were running locally or on trainer at start.
- **v616 private verifier result:** `yourslewis/bc26-v616-anchored-jung21-sed-blend` v1 completed with no failure. Outputs included `submission.csv`, `submission_anchor_raw.csv`, `submission_samejima_sed_raw.csv`, `submission_jung21_raw.csv`, `submission_before_alignment.csv`, `submission_sed.csv`, `subm_21.csv`, and expected cache/model outputs.
- **Output validation:** downloaded v616 outputs to `artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/`. Final `submission.csv`: `240x235`, 240 unique row IDs, no bad values, all 234 class columns nonconstant, range `0.004166667..1.0`. Anchor, Samejima SED, and Jung21 raw branches were finite, nonconstant, and row-aligned. Final vs anchor corr `0.98851`, MAE `0.03966`; branch raw corr vs anchor: Jung21 `0.40117`, SED `0.23945` (as expected, final is rank-blended/bounded).
- **Runtime/log validation:** private verifier completed in the Kaggle run; log markers confirmed Samejima anchor/SED preservation, Jungchan Model21 branch write, fixed v616 rank blend, and final output write. Public dry-run wall-clock log tail reached about `1088s` including nbconvert; high but completed.
- **Local matched-row gate on actual v616 outputs:** copied v616 outputs to trainer and ran `birdclef_public946_multi_sidecar_weight_grid.py` with actual `submission_anchor_raw.csv`, `submission_jung21_raw.csv`, and `submission_samejima_sed_raw.csv`. Base AUC `0.9903905` on 190 rows / 42 classes. v616 blend AUC `0.9934807`, lift `+0.0030902`, corr vs anchor `0.999696`, MAE `0.006244`, top3 recall `0.4842`. Stability: site bootstrap 1000 iters mean lift `+0.004888`, q05 `+0.0017568`, p(lift>0)=`0.999`; leave-one-site all 6 held-out sites positive, min lift `+0.002675`, q05 `+0.002682`.
- **Submitted:** added guarded submitter `scripts/submit_v616_anchored_jung21_sed_when_ready.py`; it rechecked cap, duplicate description, COMPLETE/no-failure status, required outputs/log markers, schema, numeric validity, and nonconstant guards. Submitted v616 as `v616: Repo-owned Samejima anchor plus Jung21 and SED rank blend`, ref `53012761`. Post-submit state: 2026-05-25 UTC slots used `1/5`; v616 is PENDING.
- **Next:** monitor v616 score. If it improves above 0.949, immediately build/confirm a repo-owned variant and inspect branch contributions; if it ties, treat Jung21/SED as plateau but keep the unified grid harness for future branches; if it drops, log the failure and pivot to the next non-public-replay lane (per-class residual selector or fresh source/artifact scout).

### 2026-05-25 08:15 UTC — v616 pending; Alexy blocked; per-class selector rejects adaptive variant

- **Live state:** v616 (`53012761`) is still PENDING. 2026-05-25 UTC slots used `1/5`; estimated remaining `4`. Latest scored results unchanged: v613 Alexy direct `0.923`, v614/v615/v611/v612 `0.949`; confirmed best remains **0.949**. No open PRs; PR #256 is merged. No active BirdCLEF jobs were found locally or on trainer.
- **Chosen track:** A first, then F. Tried to advance Alexy NS1 sidecar extraction because it is highest priority after v616, but current source/output access is blocked; pivoted to per-class capped residual selector as concrete no-slot progress.
- **Alexy blocker:** added `kaggle-kernels/v613-alexy-ns1-sidecar/FEASIBILITY_NOTES.md`. Current Kaggle Bearer API returns 403 for `GetKernel`, `GetKernelSessionStatus`, `ListKernelSessionOutput`, and `kaggle kernels pull` for `alexycactus/birdclef-2026-ns1-ensemble`; web fetch hits reCAPTCHA. Existing preflight still shows valid/nonconstant `192x235` output and logs with five NS1 CNN checkpoints + Perch/MLP, but direct v613 already scored `0.923`, so no further slot is justified without source access.
- **Implemented no-slot selector:** added `scripts/birdclef_per_class_sidecar_selector.py`. It loads an anchor plus named sidecars, ranks columns, chooses tiny per-class weights under a total cap, and evaluates by leave-group CV (`site`/`file`/`row`) against train-soundscape labels. This is explicitly a rejection/idea screen, not an approval gate.
- **v616 per-class selector result:** ran on trainer using actual v616 raw outputs (`submission_anchor_raw.csv`, `submission_jung21_raw.csv`, `submission_samejima_sed_raw.csv`) with grid `0,0.005,0.01,0.02,0.04,0.06`, cap `0.08`, group `site`.
  - Regularized `min_lift=0.0005`: base AUC `0.9903905`; leave-site CV AUC `0.9903940`, lift only `+0.0000035`; leave-group lift summary all `0`; all-row in-sample AUC `0.9932692` (`+0.0028787`) but this does not transfer across sites. Artifact: `artifacts/anchored_blend_audit/v616_per_class_selector_20260525T0810Z.json`; log: `logs/v616_per_class_selector_20260525T0810Z.log`.
  - Unregularized `min_lift=0`: CV lift only `+0.0000019`; all-row in-sample lift `+0.0029182`. Artifact: `artifacts/anchored_blend_audit/v616_per_class_selector_minlift0_20260525T0810Z.json`; log: `logs/v616_per_class_selector_minlift0_20260525T0810Z.log`.
- **Decision:** no second submission. The per-class selector exposes classic in-sample overfit: good all-row lift but essentially zero leave-site transfer. Do not package a v617 per-class variant unless a future selector shows real leave-site lift. Next exact action is to monitor v616 score; if still pending next run, pivot to fresh source/artifact scout or SED/export smoke rather than scalar/per-class tweaks of v616.

### 2026-05-25 10:10 UTC — v616 tied; fresh scout + SYD52p sidecar screen

- **Live state:** v616 ref `53012761` completed at `0.949`, tying the current best but not improving. 2026-05-25 UTC slots used `1/5`; estimated remaining `4`. Latest scored remain v616/v615/v614/v612/v611 all `0.949`, v613 Alexy direct `0.923`, v610 `0.852`. No open PRs; active branch `feature/birdclef-20260524-20utc-v612-submit`. No active BirdCLEF jobs were found locally or on trainer.
- **Decision from v616:** the strong train-soundscape sidecar gate was again only rejection-screen evidence. Do not spend slots on scalar variants or per-class variants of v616; the previous per-class selector already showed no leave-site transfer.
- **Chosen track:** G fresh source/artifact scout, because Alexy is blocked and v616 tied. Ran Kaggle API kernel scan for BirdCLEF 2026 / 0.95 / 0.949 / final / v10 queries. Artifact: `artifacts/public_kernels_20260525_fresh_scout/scan_20260525T1000Z.json`.
- **Audited fresh/recent candidates:** wrote source/output audit artifacts under `artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/` for 15 refs including `kijiang/birdclef2026-v354`, `kijiang/birdclef2026-v353`, the new `BC2026 P949 SYD ...` cluster, `ykuroka/*nosidecar`, `probe-atfujita2023-jit`, `probe-babych2025-openvino`, and `jacqueszhelinzhang/birdclef26-deepcnn`.
- **Scout findings:**
  - The fresh `BC2026 P949 SYD ORT/ECA/EFFV2` cluster exposes valid raw branch CSVs (`subm_21`, `subm_52p`, `submission_protossm`, `submission_sed`) but public final `submission.csv` is malformed (`243x235` with 56,862 bad values). Outputs appear branch-identical across clones and anchored in the already-tested Jungchan/Samejima/Raunak lineage, not a direct slot candidate.
  - `kijiang` v353/v354 similarly have malformed public finals and familiar ProtoSSM/SED branch outputs.
  - `ykuroka` no-sidecar kernels produce valid 3-row dry-run finals but are EoS6/Yukiz/Karnak/Yaroslav branch-family variants, not a new 0.960 clue.
  - `atfujita2023_jit` and `babych2025_openvino` kernels are probe/report-only; no `submission.csv` output.
  - `jacqueszhelinzhang/birdclef26-deepcnn` outputs a constant 3-row `submission.csv`; rejected.
- **No-slot branch screen:** because SYD exposes a previously unsubmitted `subm_52p` branch, ran an anchored grid against actual v616 raw outputs plus `syd52p` from `joriahmed/bc2026-p949-syd-ort-effv2-a08`. Fast artifact: `artifacts/anchored_blend_audit/v616_syd52p_grid_fast_20260525T1000Z.json`; log: `logs/v616_syd52p_grid_fast_20260525T1000Z.log`.
- **Best SYD52p local candidate:** `0.90*anchor + 0.04*Jung21 + 0.04*Samejima SED + 0.02*SYD52p`, AUC `0.9935006`, lift `+0.0031101`, corr vs anchor `0.999717`, MAE `0.006366`, top3 `0.4842`. This is only `+0.000020` local AUC over v616's already-submitted/tied recipe.
- **Stability for best SYD52p:** artifact `artifacts/anchored_blend_audit/v616_syd52p_top_stability_20260525T1000Z.json`; log `logs/v616_syd52p_top_stability_20260525T1000Z.log`. Site bootstrap q05 `+0.0017685`, p(lift>0)=`0.999`; leave-one-site all 6 positive, min lift `+0.002728`. Good rejection-screen stability, but not approval evidence because v616 with nearly same gate just tied public LB.
- **Decision:** no submission. The SYD52p branch is too near v616 and only microscopic local gain over a freshly tied public result. Preserve remaining slots for genuinely new source/artifact-backed lines, not near-duplicate branch increments.

### 2026-05-25 12:10 UTC — real SED/export smoke after v616 plateau

- **Live state:** v616 remains a completed `0.949` tie. 2026-05-25 UTC slots used `1/5`, estimated remaining `4`; confirmed best remains **0.949**. Latest scored: v616/v615/v614/v612/v611 all `0.949`, v613 `0.923`, v610 `0.852`. No open PRs and no active BirdCLEF jobs locally or on trainer.
- **Chosen track:** D — real SED/export smoke. Reason: v616, per-class v616 variants, and SYD52p near-duplicate branch tweaks are exhausted as approval evidence; the spec's next useful no-slot work is an exportable hidden-safe SED signal check.
- **Config added:** `configs/birdclef/sed_b0_q3cap80_ep12init_exportsmoke_5s_160_allcls_20260525.json`. It runs EfficientNet-B0 SED on 512 balanced real-audio files / all 234 classes, 5s/160-mel, 2 epochs, q3/cap80 external-pretrain TorchScript init, focal BCE, mixup 0.2, sqrt pos-weight, TorchScript + ONNX export.
- **Run command:** on trainer, `CUDA_VISIBLE_DEVICES=0 python scripts/birdclef_sed_pilot_train.py --config configs/birdclef/sed_b0_q3cap80_ep12init_exportsmoke_5s_160_allcls_20260525.json`; log `logs/sed_b0_q3cap80_ep12init_exportsmoke_5s_160_allcls_20260525.log`.
- **Training/export result:** completed on CUDA in `39.153s`; 512 examples, 410 train / 102 val, all 234 classes. Loss improved epoch1 `0.2041/0.1497` to epoch2 `0.1161/0.1084`; best epoch `2`. Holdout macro AUC `0.754065` over 79 valid classes, so modeling signal is weak and **not** a scale/submit candidate. External init loaded 352 keys and skipped only the 2 head keys.
- **Artifacts:** trainer/local ignored artifact root `artifacts/sed_oof/sed-b0-q3cap80-ep12init-exportsmoke-5s-160-allcls-20260525/` with `metrics.json`, `config.resolved.json`, `training_log.jsonl`, holdout predictions, `model_torchscript.pt` (15.389 MB), `model.onnx` (0.56 MB), and generated `sed_bundle_manifest.json`.
- **Runtime/export validation:** ONNX checker passed. Lightweight TorchScript inference smoke ran on CPU for 4 real audio files via `scripts/birdclef_sed_infer_torchscript.py`; runtime `0.326s` total / `0.082s` per file, output `4x237` CSV with all 234 probability columns nonconstant, min `0.1654`, max `0.4749`, mean `0.2792`. Log `logs/sed_b0_q3cap80_ep12init_exportsmoke_infer_smoke_20260525.log`; output `artifacts/sed_oof/sed-b0-q3cap80-ep12init-exportsmoke-5s-160-allcls-20260525/infer_smoke_probs.csv`.
- **Decision:** no submission and do not scale this exact config: operational/export gate passed, but model gate failed (AUC too low). The useful carry-forward is the validated export/inference path. Next SED work should change the learning target/data distribution (OOF-teacher cache or hard-negative/no-call residual), not simply scale this weak supervised balanced smoke.

### 2026-05-25 14:15 UTC — pseudo-label threshold/cache redesign + OOF-teacher soft smoke

- **Live state:** current best remains **0.949**; v616/v615/v614/v612/v611 are all `0.949`; v613 `0.923`; v610 `0.852`. 2026-05-25 UTC slots used `1/5`, estimated remaining `4`. No active BirdCLEF jobs locally or on trainer; no open PRs.
- **Chosen track:** E — pseudo-label threshold/cache redesign. Reason: the 12UTC SED export smoke passed operationally but failed the model gate (`0.754` AUC), so the next useful step is target/cache quality rather than another v616 sidecar or scaling the weak supervised SED config.
- **Tooling added:** `scripts/birdclef_oof_teacher_threshold_sweep.py`, a file-level OOF cache threshold diagnostic for NPZs with `files`, `labels`, `y_true`, and `teacher_pred`. It evaluates AUC/top-k, positive/negative threshold masks, precision/recall, class coverage, and conservative hard-mask shortlist.
- **Threshold sweeps run on trainer:** artifacts copied locally under `artifacts/pseudolabels/threshold_sweeps/`; logs under `logs/*threshold_sweep_20260525T1400Z.log`.
  - Public train-soundscape `teacher_sed85_rankblend15`: macro AUC `0.997018` over 75 classes, top10 recall `0.9934`. Best hard positives can be 100% precise but cover only `9` classes / `456` cells (`14.6%` true-cell recall). This is too narrow/leaky for all-class student scaling.
  - Public train-soundscape `teacher_sed`: macro AUC `0.996743` over 75 classes, top10 recall `0.9972`; best hard positives 100% precise but again only `9` classes / `556` cells (`17.8%` true-cell recall). Also too narrow for all-class training.
  - OOF teacher `b0v26_nfnetv29_w090010_intersection_cache`: macro AUC `0.911282` over 170 classes, top10 recall `0.6661`; broader but less calibrated. Conservative hard-positive shortlist has only `12` positive cells / `6` classes at 91.7% precision, so hard positives are too sparse. Soft-label use is more plausible than hard-threshold use.
  - OOF negative cache `v13v15_neg005_pos095`: weak as a positive teacher, macro AUC `0.657329` over 100 classes, top10 recall `0.259`; hard positives only `9` cells / `5` classes at 77.8% precision. Keep as negative-mask idea only, not positive pseudo-label source.
- **Soft OOF-teacher smoke config added:** `configs/birdclef/sed_b0_oofteacher_b0v26_nfnetv29_soft_512_ep3_20260525.json`. It trains EfficientNet-B0 SED on 512 OOF-teacher-backed files, all 234 classes, 5s/160-mel, q3/cap80 external init, soft `teacher_pred` targets, BCE, no mixup/class-balancing, 3 epochs, TorchScript+ONNX export.
- **Soft OOF-teacher smoke result:** trainer CUDA run completed in `23.381s`. Loss improved `0.5788/0.4831` -> `0.4263/0.3798` -> `0.3592/0.3427`; best epoch `3`. Holdout macro AUC `0.819021` over 80 valid classes, better than the supervised balanced export smoke (`0.754`) but still far below the useful gate (`~0.90–0.93`). External init loaded 352 keys and skipped only 2 head keys. OOF teacher coverage on selected rows was 100%, all rows had both members available.
- **Export/runtime validation:** TorchScript 15.389 MB and ONNX 0.56 MB exported; ONNX checker passed. CPU TorchScript inference smoke on 4 real files completed in `0.193s` total / `0.048s` per file with all 234 probability columns nonconstant; output `artifacts/sed_oof_teacher_students/sed-b0-oofteacher-b0v26-nfnetv29-soft-512-ep3-20260525/infer_smoke_probs.csv`; log `logs/sed_b0_oofteacher_b0v26_nfnetv29_soft_512_ep3_infer_smoke_20260525.log`.
- **Decision:** no submission. The target redesign improved the SED model gate from `0.754` to `0.819`, but it is not enough to scale/submit. Do not use hard positives from any current cache; public train-soundscape caches are too narrow (9-class leakage), and OOF hard positives are too sparse. Next exact target work should test a curriculum/longer OOF-teacher soft student or a negative-mask auxiliary loss, but only if it can clear a small-smoke AUC closer to `0.90` before scale.

### 2026-05-25 16:15 UTC — OOF-teacher soft + negative-mask auxiliary smoke

- **Live state:** confirmed best remains **0.949**. Latest scored remain v616/v615/v614/v612/v611 at `0.949`; v613 `0.923`; v610 `0.852`; v609 no public score. 2026-05-25 UTC slots used `1/5`, estimated remaining `4`. No active BirdCLEF jobs locally or on trainer; no open PRs.
- **Chosen track:** E — pseudo-label threshold/cache redesign. Reason: the 14UTC soft OOF-teacher smoke improved the SED target-design gate from `0.754` to `0.819` but remained below scale/submit quality; the next distinct target-design test was to add a small negative-mask auxiliary loss from the OOF negative cache rather than hard pseudo-positive thresholds.
- **Config added:** `configs/birdclef/sed_b0_oofteacher_b0v26_nfnetv29_soft_negaux002_512_ep3_20260525.json`.
- **Smoke result:** trainer CUDA run completed in `16.726s` on 512 OOF-teacher-backed files / 410 train / 102 val, all 234 classes, q3/cap80 init, 5s/160-mel, 3 epochs. Loss improved `0.5843/0.4827` -> `0.4302/0.3816` -> `0.3617/0.3432`; best epoch `3`. Holdout macro AUC `0.819410` over 80 valid classes.
- **Aux negative coverage:** the OOF negative cache covered only `26/512` rows (`5.08%`) with `1664` masked negative cells / 64 per covered row. The tiny AUC lift over soft-only (`0.819410` vs `0.819021`) is noise-sized and not enough for scale or submission.
- **Export/runtime validation:** TorchScript 15.389 MB and ONNX 0.56 MB exported; ONNX checker passed. CPU TorchScript inference smoke on 4 real audio files completed in `0.193s` total / `0.048s` per file, all 234 probability columns nonconstant. Trainer artifacts: `artifacts/sed_oof_teacher_students/sed-b0-oofteacher-b0v26-nfnetv29-soft-negaux002-512-ep3-20260525/`; logs `logs/sed_b0_oofteacher_b0v26_nfnetv29_soft_negaux002_512_ep3_20260525.log` and `logs/sed_b0_oofteacher_b0v26_nfnetv29_soft_negaux002_512_ep3_infer_smoke_20260525.log`.
- **Decision:** no submission/no scale. Negative-mask auxiliary loss is operationally safe but currently too sparse to matter. Next exact target-design work should either (a) build a broader negative/no-call mask with much higher row coverage before retrying aux loss, or (b) test a curriculum/longer soft OOF-teacher student only if the small-smoke AUC can approach `0.90`.

### 2026-05-25 22:43 UTC — ClawTeam late-day slot fill v617-v620

- **Live state:** best remained **0.949**; `v616` completed at `0.949`; 2026-05-25 UTC slots were `1/5`; about `1.28h` remained before reset. No active local/trainer BirdCLEF jobs were found.
- **Slot-policy decision:** under the hill-climb cron's new policy, late-day unused slots should be filled with highest-ranked valid exploratory candidates if no verifier-grade candidate exists. The high-priority new-branch lanes were not submission-ready inside the final window, so preserving 4 slots was not acceptable.
- **Scouting/audit:** refreshed public scan to `artifacts/public_kernels_20260525_late_scout/scan_20260525T2238Z.json` and used guarded source/output preflight in `scripts/submit_v617_v620_late_slot_fill_20260525.py`; submit report is `artifacts/public_kernels_20260525_late_scout/submit_v617_v620_late_slot_fill_20260525.json`.
- **Rejected by verifier:** WildSound v8 `ERROR`; Udaken cancelled/no final; P952 Exp070 teacher/cache kernels wrote 7992 train rows and lacked a competition-final path; Kijiang/P949/Gendaijin direct finals were malformed; Samejima HGNetV2 and Viktoriia finals had bad values; Om Modi was all-zero; Ykuroka wrote zero rows; Tulay was mock/wrong-shape.
- **Submissions made:** filled remaining slots with guarded code submissions:
  - `v617: Exploratory direct Nina EoS7 sz sidecar source`, ref `53032516`.
  - `v618: Exploratory direct Kruzzcc Nina EoS4 BirdNET source`, ref `53032520`.
  - `v619: Exploratory direct Kruzzcc Mtoshi UMAP BirdNET source`, ref `53032523`.
  - `v620: Exploratory direct Kazuhiro Karnak rank fusion source`, ref `53032524`.
- **Post-submit state:** 2026-05-25 UTC slots are now `5/5`; v617-v620 are pending. Current scored LB still `0.949` pending those results.
- **Next:** monitor v617-v620. If any improves, build a repo-owned confirmer from that source family. If all tie/drop, resume genuinely new-signal work rather than more EoS/ProtoSSM/SED repeats.

### 2026-05-25 23:07 UTC — capped slots + soundscape non-Aves/no-train data point

- **Live state:** best remains **0.949**. `v616` tied at `0.949`; `v617`-`v620` are still pending; 2026-05-25 UTC slots are **5/5** with ~52 minutes to reset. No active BirdCLEF jobs were found locally or on trainer before this run.
- **Slot decision:** no additional competition submission possible because cap is full. Per the new data-point policy, continued by training a distinct branch rather than idling.
- **Scout refresh:** EfficientAT and PANNs/Cnn14 remain the strongest AudioSet event/no-call leads, but the current trainer venv lacks `panns_inference`, TensorFlow/TF-Hub, and PaSST packages. A real AudioSet embedding branch needs an explicit packaging step.
- **Implementation:** added `scripts/birdclef_soundscape_specialist_train.py` and config `configs/birdclef/soundscape_nonaves_notrain_b0_5s160_siteS08_ep3_20260525.json`. The script trains on official `train_soundscapes_labels.csv` 5s windows, with a 72-class non-Aves/no-train specialist head.
- **Training result:** 1,478 windows, 5,420 positive target cells, site-holdout `S08` with 120 validation rows. EfficientNet-B0 SED-style model used q3/cap80 external-pretrain encoder init; 352 keys loaded, head skipped. Runtime `19.46s` on CUDA. Best val loss at epoch 2 (`0.26949`).
- **Metrics:** site-holdout macro AUC `0.48865` over 18 valid scoped classes; no-train macro AUC `0.47610` over 17 valid classes. Some sonotypes were learnable (`47158son22=0.988`, `son13=0.944`, `son11=0.910`), but several inverted badly (`son18=0.057`, `son25=0.092`, `son10=0.106`). This is a useful landscape data point, not a slot candidate.
- **Export/runtime:** TorchScript and ONNX exported under `artifacts/soundscape_specialists/soundscape-nonaves-notrain-b0-5s160-siteS08-ep3-20260525/`; ONNX checker OK; CPU TorchScript smoke `0.093s` for 2 logmel samples with `[2,72]` clip output.
- **Decision:** no submission/no scale. The branch is rule-safe and diverse, but not competition-format and not submission-grade. Next after reset: score-check `v617`-`v620`; if none improve, package EfficientAT/PANNs AudioSet embeddings for this same non-Aves/no-call target space or test site-balanced/group-DRO training.

### 2026-05-26 00:22 UTC — reset-day score check + PANNs/Cnn14 AudioSet soundscape data point

- **Live state:** current best remains **0.949**. New UTC day slots were **0/5 used** with ~23.7h to reset. The late-day exploratory submissions all completed: `v617=0.949`, `v618=0.946`, `v619=0.944`, `v620=0.949`; no improvement over `v616=0.949`. No active BirdCLEF jobs were found locally or on trainer.
- **Slot decision:** no Kaggle submission this early UTC run. `v617`/`v620` are ties and exact replays/duplicates are not valid; `v618`/`v619` dropped; no verifier-grade or high-info submit-ready candidate was available.
- **Scout refresh:** saved Kaggle public scan to `artifacts/public_kernels_20260526_scout/scan_20260526T0020Z.json`. Recent/query results were known/rejected plateau-family or malformed/error sources (WildSound, Viktoriia, Tulay, Pilkwang/Nina/Jungchan/EoS/Prior Field); no fresh >0.949 public clue surfaced. PANNs/EfficientAT-specific search returned no direct BirdCLEF notebook lead.
- **Chosen track:** package/train the highest-ranked distinct model data point: PANNs/Cnn14 AudioSet embeddings for non-Aves/no-train/no-call soundscape windows. Installed `panns-inference==0.1.1` in the trainer venv and downloaded the public Cnn14 AudioSet checkpoint to `/home/yourslewis/panns_data/Cnn14_mAP=0.431.pth`.
- **Implementation:** added `scripts/birdclef_panns_soundscape_embedding_train.py` and config `configs/birdclef/panns_cnn14_audioset_soundscape_nonaves_notrain_nocall_siteS08_ep12_20260526.json`.
- **Training result:** extracted PANNs 2048-d embeddings for 1,478 official `train_soundscapes` 5s windows, then trained a 12-epoch MLP head for 72 non-Aves/no-train labels plus no-call aux. Embedding extraction took 49.84s CUDA; best val loss `0.45604` at epoch 5.
- **Metrics:** site-holdout `S08` macro AUC `0.517333` over 18 valid scoped classes; no-train macro AUC `0.520824` over 17 valid classes; no-call aux AUC invalid on this split because the validation target lacked both classes. This is a mild improvement over the B0 soundscape specialist's `0.48865`, but still weak.
- **Artifacts:** `artifacts/panns_soundscape_embeddings/panns-cnn14-audioset-soundscape-nonaves-notrain-nocall-siteS08-ep12-20260526/` with metrics, holdout predictions, `embedding_head.pt`, and `embedding_head_torchscript.pt`; ledger `artifacts/model_data_point_ledger/20260526T0022Z_panns_cnn14_audioset_soundscape.md`; queue report `specs/birdclef-hillclimb-cron-20260525/ranked_queue_20260526T0022Z.md`.
- **Verifier:** no-slot training is rule-safe; uses official train soundscapes plus public AudioSet checkpoint. Holdout predictions are finite/nonconstant and TorchScript head smoke passed. Output is 72-label specialist only, not a 234-class competition submission; **no submission approved**.
- **Next:** do not scale this unchanged. Next exact action should be broader no-call/negative mask coverage, 20s temporal/localmax branch, or a PANNs leave-one-site/no-call-valid split before any capped 234-class sidecar wrapper.

### 2026-05-26 02:20 UTC — broad OOF negative/no-call mask + 1024-row control

- **Live state:** current best remains **0.949**. The 2026-05-25 late-day exploratory submissions completed as `v617=0.949`, `v618=0.946`, `v619=0.944`, `v620=0.949`; `v616` remains the tied baseline at `0.949`. 2026-05-26 UTC slots used `0/5`. No active BirdCLEF jobs were found locally or on trainer before this run.
- **Slot decision:** no early-day competition submission. No verifier-grade/high-info hidden-safe candidate is ready, and duplicate replays of v616/v617/v620 are forbidden.
- **Chosen track:** broader OOF negative/no-call SED student. The prior v13/v15 negative aux cache covered only `26/512` selected rows, so this run created a broad OOF-teacher-derived negative mask from `artifacts/pseudolabels/oof-teacher-cache/b0v26_nfnetv29_w090010_intersection_cache.npz`.
- **Mask builder added:** `scripts/birdclef_oof_teacher_negative_mask.py`. With threshold `0.03` and cap `64` negatives/row, the capped mask has 47,343 cells, 1,259/1,279 row coverage (98.4%), 230/234 class coverage (98.3%), and 0 false-negative cells. Summary: `artifacts/pseudolabels/oof-negative-cache/b0v26_nfnetv29_teacher_neg003_cap64_20260526.summary.json`.
- **Broad-neg branch:** config `configs/birdclef/sed_b0_oofteacher_b0v26_nfnetv29_soft_broadneg003_w001_1024_ep4_20260526.json`; 1,024 OOF-teacher-backed files, all 234 classes, 5s/160-mel, q3/cap80 init, 4 epochs, aux negative weight `0.01`. CUDA runtime `32.685s`; macro AUC `0.908278` over 122 valid classes; selected-row aux coverage 1,024/1,024 rows and 37,993 negative cells.
- **Matched control:** config `configs/birdclef/sed_b0_oofteacher_b0v26_nfnetv29_soft_1024_ep4_20260526.json`; same rows/epochs without aux loss. CUDA runtime `26.059s`; macro AUC `0.911067` over 122 valid classes, slightly better than broad-neg.
- **Export/runtime validation:** both branches exported TorchScript (~15.389 MB) and ONNX (0.56 MB); ONNX checker passed; CPU TorchScript inference smoke on 4 real audio files passed with all 234 probability columns nonconstant. Soft-only smoke: `0.199s` total / `0.050s` per file. Broad-neg smoke: `0.185s` total / `0.046s` per file.
- **Decision:** no submission. The data point is useful: 1,024-row soft OOF-teacher training reaches the first recent B0 smoke in the useful `0.90–0.93` proxy band. But the broad negative auxiliary at weight `0.01` does not help versus the matched control, and both are still random-split comparison-grade only. Next: package the soft-only B0 as a raw 234-class sidecar for no-slot v616 audit, or run the distinct 20s temporal/localmax branch before packaging.

### 2026-05-26 04:19 UTC — 20s temporal/localmax B0 data point

- **Live state:** current best remains **0.949**. `v616` is still the tied baseline; `v617=0.949`, `v618=0.946`, `v619=0.944`, `v620=0.949`. 2026-05-26 UTC slots used **0/5**. No active BirdCLEF jobs were found locally or on trainer before this run.
- **Slot decision:** no early-day submission. No verifier-grade/high-info non-duplicate candidate was ready, and duplicate replays of v616/v617/v620 remain forbidden.
- **Scout refresh:** web/search scan surfaced no fresh clean >0.949 public lead; recent visible leads are still Nina/EoS/plateau families or discussion/model leads already reflected in the queue.
- **Chosen track:** 20s temporal/localmax branch from the default data-point queue. Added config `configs/birdclef/sed_b0_oofteacher_b0v26_nfnetv29_soft_20s_localmax_512_ep3_20260526.json`.
- **Training result:** `sed-b0-oofteacher-b0v26-nfnetv29-soft-20s-localmax-512-ep3-20260526` trained on 512 OOF-teacher-backed official train-audio files, all 234 classes, 20s/160-mel input, q3/cap80 init, BCE, 3 epochs. CUDA runtime `20.778s`; best val loss `0.322308`; macro AUC `0.672996` over 72 valid classes.
- **Diversity check:** compared full predictions against the 5s soft-only 1024/ep4 B0 control on 407 overlapping files. Global Pearson correlation `0.599986`, MAE `0.036360`, so it is decorrelated but weak.
- **Export/runtime:** TorchScript `15.389 MB`; ONNX checker passed; CPU TorchScript smoke on 4 files completed in `0.301s` total / `0.075s` per file with finite 234-class output.
- **Decision:** no submission/no package unchanged. This is a useful negative/decorrelation data point: naive 20s context with soft OOF targets hurts badly versus the 5s 1024-row soft-only control (`0.911067`). If revisited, use true local-window/offset pseudo-labels or multi-crop localmax aggregation.
- **Reports/artifacts:** `specs/birdclef-hillclimb-cron-20260525/ranked_queue_20260526T0419Z.md`, `specs/birdclef-hillclimb-cron-20260525/model_data_point_20260526T0419Z_20s_localmax.md`, ledger `artifacts/model_data_point_ledger/20260526T0419Z_20s_localmax.md`, artifact root `artifacts/sed_oof_teacher_students/sed-b0-oofteacher-b0v26-nfnetv29-soft-20s-localmax-512-ep3-20260526/`.
- **Next:** package/audit the stronger 1024-row soft-only B0 as a raw 234-class sidecar against v616, or move to G124/V2S if B0 audit fails.

### 2026-05-26 06:34 UTC — G124/V2S target-design localmax data point + B0/G124 sidecar audit

- **Live state:** best remains **0.949**; `v616` is still the tied baseline. Latest scored submissions remain `v617=0.949`, `v618=0.946`, `v619=0.944`, `v620=0.949`. 2026-05-26 UTC slots used **0/5** with ~17.4h to reset. No active jobs remained after cleanup.
- **Scout refresh:** web/Kaggle-search queries for EfficientAT/PANNs/AudioSet and fresh 0.949+ leads found no clean new public lead; results were generic EDA/baseline or irrelevant.
- **Duplicate prevention:** prior G124 all-row V2S-init center pilot already existed on trainer, so it was not rerun.
- **Trained data point:** `g124-effv2s-public946-pseudo-pilot-20260526-v2sinit-power085-localmax-ep6` using EfficientNetV2-RW-S, external V2S init, 792 teacher train-soundscape rows, `teacher_power=0.85`, local-max radius 1 targets, focal BCE, 6 epochs. Best val AUC `0.960094` over 62 valid classes; all-row student AUC `0.944720`; student/teacher corr `0.847478`; TorchScript+ONNX export passed on trainer.
- **Sidecar audit:** generated raw train-soundscape predictions for the soft-only B0 `1024_ep4` student, filtered to the 240 v616 proxy rows, converted G124 center/localmax predictions to sidecar CSVs, and ran `audit_vs_v616_fast.json`. Best tiny G124-only recipe lifted local proxy from `0.993480668` to `0.993484059` (`+0.00000339`) with corr `0.999986`; soft-B0 weights did not help.
- **Decision:** no submission. The G124 signal is interesting but the local lift is too small and teacher/proxy-derived for an early-day Kaggle slot. Next: EfficientAT AudioSet embedding branch if assets are clean, otherwise a bounded G124 hard-confidence/power ablation.


### 2026-05-26 06:59 UTC — EfficientAT MN10 AudioSet soundscape embedding branch

- **User request:** train the EfficientAT embedding branch as a new-model data point.
- **Implementation:** added `scripts/birdclef_efficientat_soundscape_embedding_train.py` and config `configs/birdclef/efficientat_mn10_audioset_soundscape_nonaves_notrain_nocall_siteS08_ep12_20260526.json`. The script uses EfficientAT `mn10_as` AudioSet-pretrained embeddings on official `train_soundscapes` 5s windows and trains a small MLP head for 72 non-Aves/no-train labels plus a no-call auxiliary target.
- **Trainer setup:** cloned EfficientAT to `/home/yourslewis/external_models/EfficientAT`, installed missing `wget`, and used the public `mn10_as_mAP_471.pt` checkpoint through the EfficientAT loader.
- **Training result:** 1,478 windows, site-holdout `S08` with 120 validation rows, 960-d embeddings and 527 AudioSet logits extracted in `13.30s` CUDA. 12-epoch MLP head best validation loss `0.487352` at epoch 5.
- **Metrics:** site-holdout macro AUC `0.488240` over 18 valid scoped classes; no-train macro AUC `0.472842` over 17 valid classes; non-Aves macro AUC `0.488240`; no-call AUC invalid because this validation split lacked both no-call classes. Predictions were finite and nonconstant.
- **Verification:** TorchScript head smoke passed on trainer (`2x960 -> 2x72` label logits and `2x1` no-call logits). Artifacts under `artifacts/efficientat_soundscape_embeddings/efficientat-mn10-audioset-soundscape-nonaves-notrain-nocall-siteS08-ep12-20260526/`; log `logs/efficientat_mn10_audioset_soundscape_nonaves_notrain_nocall_siteS08_ep12_20260526.log`; ledger `artifacts/model_data_point_ledger/20260526T0659Z_efficientat_mn10_soundscape.md`.
- **Decision:** no submission/no scale unchanged. EfficientAT MN10 is a useful branch data point but weaker than the prior PANNs/Cnn14 AudioSet branch on the same soundscape target (`0.488240` vs `0.517333`). If continuing EfficientAT, try `dymn10_as` or a site-balanced/leave-one-site head rather than repeating `mn10_as` unchanged.

### 2026-05-26 08:20 UTC — EfficientAT DyMN10 AudioSet soundscape embedding branch

- **Live state:** best remains **0.949**. Bearer API listing shows no 2026-05-26 UTC submissions yet; latest scored are `v617=0.949`, `v618=0.946`, `v619=0.944`, `v620=0.949`, with `v616` still the tied repo-owned baseline. Slots used: **0/5** with ~15.7h to reset. No active BirdCLEF jobs were found locally/on trainer before the run.
- **Scout/critic:** role report `specs/birdclef-hillclimb-cron-20260525/reports/scout_critic_20260526T0815Z.md` recommended EfficientAT `dymn10_as` as the next bounded no-slot data point and rejected early-day submission.
- **Training:** added config `configs/birdclef/efficientat_dymn10_audioset_soundscape_nonaves_notrain_nocall_siteS08_ep12_20260526.json` and trained with existing `scripts/birdclef_efficientat_soundscape_embedding_train.py` on trainer. Used public EfficientAT `dymn10_as.pt`, official train-soundscape 5s windows, 72 non-Aves/no-train labels, no-call aux, site holdout `S08`, 12 epochs.
- **Result:** extracted `1478 x 960` embeddings in `36.23s` CUDA; best val loss `0.428341`; S08 macro AUC `0.568586` over 18 valid scoped classes; no-train AUC `0.553327`; no-call AUC invalid on S08.
- **Comparison:** DyMN10 beat EfficientAT MN10 (`0.488240`) and PANNs/Cnn14 (`0.517333`) on this same target contract, so AudioSet remains alive as a rare-slice sidecar lane.
- **Verifier:** finite/nonconstant holdout predictions shape `120 x 72`; TorchScript head smoke passed `(2,960)->(2,72)+(2,1)`. Not submission-format; no Kaggle slot approved.
- **Artifacts:** `artifacts/efficientat_soundscape_embeddings/efficientat-dymn10-audioset-soundscape-nonaves-notrain-nocall-siteS08-ep12-20260526/`, log `logs/efficientat_dymn10_audioset_soundscape_nonaves_notrain_nocall_siteS08_ep12_20260526.log`, ledger `artifacts/model_data_point_ledger/20260526T0820Z_efficientat_dymn10_soundscape.md`, queue `specs/birdclef-hillclimb-cron-20260525/ranked_queue_20260526T0820Z.md`.
- **Next:** run multi-site/leave-one-site evaluation for AudioSet heads and decide whether DyMN10 deserves a 234-class sidecar wrapper; otherwise pivot to G124 hard-confidence/power ablation.

### 2026-05-26 10:20 UTC — train_soundscapes sequence/file/site mining data point

- **Live state:** best remains **0.949**; `v616` remains tied baseline. Latest scored submissions are `v617=0.949`, `v618=0.946`, `v619=0.944`, `v620=0.949`; no 2026-05-26 UTC submissions yet, slots `0/5` at live check. No active local/trainer BirdCLEF jobs before this run.
- **Scout/slot decision:** fresh web search did not surface a clean new >0.949 public code lead; early-day submission rejected because no verifier-grade, competition-format, nonduplicate candidate exists.
- **Chosen track:** corrected top queue item — mine official `train_soundscapes` as sequences/files/sites, not isolated rows.
- **Implementation:** added `scripts/birdclef_soundscape_sequence_mining.py` and config `configs/birdclef/soundscape_sequence_dymn10_context_losite_ep16_20260526.json`. It consumes cached EfficientAT `dymn10_as` embeddings, reconstructs 5s windows by file/site, builds context features (current, prev/next, local mean/max, file mean, time features; no site one-hot), trains row-only and context heads under site-balanced sampling, and evaluates leave-one-site plus file-level MIL max pooling.
- **Data:** 1,478 official train-soundscape windows, 60 files, 9 sites, 72 non-Aves/no-train labels, 5,420 scoped positive target cells.
- **Result:** 6 meaningful leave-site folds. Row-only mean AUC `0.578422`; context mean AUC `0.601355`, delta `+0.022933`. File-MIL mean AUC improved `0.563852` -> `0.632127`. Best fold deltas: S19 `+0.097226`, S23 `+0.075329`, S13 `+0.055466`; regressions: S03 `-0.051268`, S22 `-0.047111`.
- **Verifier:** no-slot artifact checks passed on trainer: finite/nonconstant leave-site predictions `(1314, 72)` and TorchScript smoke `(2, 5764) -> (2, 72)`. Not submission-format and not v616-audited; no Kaggle slot approved.
- **Artifacts:** `artifacts/soundscape_sequence_mining/soundscape-sequence-dymn10-context-losite-ep16-20260526/`, log `logs/soundscape_sequence_dymn10_context_losite_ep16_20260526.log`, ledger `artifacts/model_data_point_ledger/20260526T1020Z_soundscape_sequence_mining.md`, queue `specs/birdclef-hillclimb-cron-20260525/ranked_queue_20260526T1020Z.md`.
- **Decision:** comparison-grade signal; continue sequence lane but fix S22/S03 regressions before any wrapper. Next exact action: residual/regularized context head or compact per-file TCN/smoother with explicit S22/S03 guard; otherwise G124 hard-confidence/power ablation.

## 2026-05-26 12:20 UTC — BirdCLEF compact per-file TCN soundscape sequence data point
- Live status via Kaggle Bearer API: best remains `0.949`; latest scored `v616=0.949`, `v617=0.949`, `v618=0.946`, `v619=0.944`, `v620=0.949`; 2026-05-26 UTC slots `0/5`; no active local/trainer BirdCLEF jobs and trainer GPUs idle.
- Early UTC-day slot decision: no submission. No verifier-grade/high-info non-duplicate candidate was ready; exact/near replays of tied public families remain forbidden.
- Added `scripts/birdclef_soundscape_tcn_mining.py` and config `configs/birdclef/soundscape_tcn_dymn10_losite_ep20_20260526.json`.
- Trained `soundscape-tcn-dymn10-losite-ep20-20260526`: compact residual per-file TCN over cached EfficientAT DyMN10 train-soundscape embeddings + time features; official train_soundscapes only; 1,478 windows / 66 files / 9 sites; 72 non-Aves/no-train labels; leave-site validation; 20 epochs; ~14.4s summed CUDA runtime.
- Result: leave-site row macro AUC mean `0.547582` vs previous context-MLP `0.601355` (`-0.053773`); file-MIL mean `0.606240` vs `0.632127` (`-0.025887`). Fold delta vs context: S03 `+0.195896`, S08 `-0.076063`, S13 `-0.053791`, S19 `-0.085756`, S22 `-0.021799`, S23 `-0.281125`.
- Verifier: final predictions finite/nonconstant (72/72 columns), TorchScript smoke passed `(2,12,input_dim)->(2,12,72)`. No 234-class wrapper/v616 audit; no Kaggle submission.
- Reports/artifacts: `specs/birdclef-hillclimb-cron-20260525/ranked_queue_20260526T1220Z.md`, `artifacts/model_data_point_ledger/20260526T1220Z_soundscape_tcn_sequence_mining.md`, artifact root `artifacts/soundscape_sequence_mining/soundscape-tcn-dymn10-losite-ep20-20260526/`.
- Decision: useful negative/diagnostic data point. Naive per-file TCN is weaker overall but fixes S03; next exact action is a residual/gated sequence smoother with S03/S22 guard, or pivot to compact deeper soundscape-native CNN/SED.
