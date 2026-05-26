# Scout + Critic Report — BirdCLEF hill-climb cron — 2026-05-26 18:15 UTC

## Scope / evidence read
- Current spec: `specs/birdclef-hillclimb-cron-20260525/spec.md`.
- Latest official queue reviewed: `specs/birdclef-hillclimb-cron-20260525/ranked_queue_20260526T1620Z.md`.
- Model ledger/performance table reviewed through `artifacts/model_data_point_ledger/performance_table.md` and `20260526T1620Z_soundscape_native_losite.md`.
- Relevant recent reports/ledgers: `reports/scout_critic_20260526T1615Z.md`, `reports/scout_critic_20260526T1020Z.md`, `reports/scout_critic_20260526T0815Z.md`, plus sequence/native ledgers.
- Additional local artifact observed but not yet in the canonical performance table: `artifacts/soundscape_sequence_mining/soundscape-sequence-dymn10-r2-nofile-reg-losite-ep20-20260526/` with matching untracked config `configs/birdclef/soundscape_sequence_dymn10_r2_nofile_reg_losite_ep20_20260526.json`.

Evidence level: **live-independent comparison-grade scout/critic synthesis**. I did not call Kaggle, submit, train, or modify code. This report uses the last recorded live state from 16:20Z plus local repo/artifact evidence.

## Live-independent status assumption
- Last recorded best public LB remains **0.949**.
- Last recorded scored submissions: `v616=0.949`, `v617=0.949`, `v618=0.946`, `v619=0.944`, `v620=0.949`.
- Last recorded 2026-05-26 UTC slot usage: **0/5**.
- No verifier-grade, competition-format, nonduplicate candidate is documented after the 16:20Z soundscape-native B0 run.
- Because this is intentionally no-API/live-independent, re-check Kaggle state only if the coordinator is about to submit or report live status externally.

## Scout finding
No fresh clean public/source lead should displace the internal no-slot queue right now.

- Repeated public scans and scout reports have not found a clean >0.949 public code path. The known public frontier is mostly plateau families, malformed finals, static/cache-only risk, or already-submitted tied/dropped kernels.
- The 16:20Z compact soundscape-native B0 result answered an important question: direct compact logmel fine-tuning on the sparse train-soundscape target did **not** beat frozen DyMN10 context features.
- The strongest measured soundscape-specific signal remains the DyMN10 sequence/context lane, not native B0, TCN, or gated residual variants.

## Submission decision
**Do not submit now.**

Reasoning:
- `0/5` slots are available, but availability alone is not a submission reason while still outside the late-day forced-fill window.
- Current train-soundscape artifacts are 72-label landscape models, not 234-column competition submissions.
- No artifact has passed a v616/v617/v620 nonduplicate sidecar audit with meaningful lift.
- The best public/source candidates are either duplicates/plateau, malformed/static-risk, or already known to tie/drop.
- Submitting now would be low-information leaderboard probing, not verifier-grade hill climbing.

Slot policy interpretation:
- Keep slots unused for now.
- If still no verifier-grade package near the <3h-to-reset window, activate only the guarded late-day slot-fill queue with strict source/output/duplication checks.

## Critique of the 16:20Z result
The soundscape-native B0 branch should be **kept as a negative data point and rejected as a submission path**.

Key evidence:
- Native B0 LOSO row AUC: `0.558044`.
- Native B0 file-MIL AUC: `0.429828`.
- Best prior DyMN10 context MLP: `0.601355` row / `0.632127` file-MIL.
- Native B0 verifier passed finite/nonconstant and TS/ONNX export checks, but it is not competition-format and has no 234-class wrapper or v616 audit.

Critic read:
- The issue is not “we need one more direct B0 fine-tune.” Direct native training underfit/overfit the sparse site-correlated target relative to frozen acoustic embeddings plus engineered file/context features.
- Do not continue native B0 as-is.
- Do not run another generic TCN/gated smoother as the immediate next action; two such attempts already underperformed the context baseline.

## Best next no-slot model data point
**Primary recommendation: a one-variable controlled DyMN10 context-MLP robustness ablation focused on file/site generalization.**

Exact candidate:
- `soundscape-sequence-dymn10-r2-nofile-reg-losite-ep20-20260526`
- Contract: official `train_soundscapes`, 1,478 windows / 66 files / 9 sites, 72 non-Aves/no-train labels, cached EfficientAT DyMN10 embeddings, leave-site evaluation.
- Change vs original context MLP: radius 2 context, no file-mean/file-max features, stronger dropout/weight decay, same no-site-onehot discipline.

Local artifact status observed:
- Artifact root exists: `artifacts/soundscape_sequence_mining/soundscape-sequence-dymn10-r2-nofile-reg-losite-ep20-20260526/`.
- Verifier-style checks from artifact inspection: leave-site predictions shape `1314 x 72`; finite; `72/72` nonconstant columns; context prediction min `1.68e-07`, max `0.9634`, std `0.1130`; TorchScript head file exists.
- Metrics: row context AUC mean `0.587753`, file-MIL context AUC mean `0.664545`, context-minus-row mean `+0.020445`.
- Fold deltas vs its row-only control: S03 `+0.075345`, S08 `+0.007517`, S13 `+0.006725`, S19 `+0.020595`, S22 `-0.063999`, S23 `+0.076490`.

Critic interpretation of this candidate:
- It is the right *type* of next data point after native B0 because it tests whether the context signal survives without a file-mean shortcut and with stronger regularization.
- It does **not** beat the original context MLP on row AUC (`0.587753` vs `0.601355`), so it is not a raw promotion-grade replacement.
- It **does** improve file-MIL mean (`0.664545` vs `0.632127`), which makes it useful for wrapper/sidecar design if file-level aggregation is the intended hidden-test behavior.
- It still regresses S22 more than the original context run, so do not package blindly.

Decision for this data point:
- **ACCEPT as no-slot artifact / landscape evidence.**
- **Reject as Kaggle submission.**
- If this artifact is already considered produced by the current run, the next action is to ledger it and compare original-context vs r2/no-file in a no-submit 72→234 sidecar audit. If it is not official yet, run/record exactly this ablation rather than starting a new branch.

## Live-independent ranked queue
1. **DyMN10 context robustness / wrapper audit — next control lane.**
   - Best immediate value: decide whether original context or r2/no-file file-MIL signal can become a cautious 234-class sidecar.
   - Gate: no Kaggle submission until schema, alignment, finite/nonconstant output, duplicate check, and v616-sidecar audit pass with non-noise lift.

2. **AudioSet/DyMN10 multi-site 234-class sidecar reformulation.**
   - DyMN10 remains the best frozen AudioSet source and powers the best context artifacts.
   - Do not run another single-site 72-label head; reformulate into multi-site features or a wrapper/audit.

3. **Fresh pretrained asset preflight: YAMNet / SurfPerch / Bioacoustics Model Zoo / PaSST / HTS-AT / BEATs / CLAP.**
   - High diversity potential, but only after license/source/runtime preflight.
   - Use the same leave-site soundscape protocol before any wrapper.

4. **G124/V2S hard-confidence or target-power ablation.**
   - Keep as fallback; prior proxy AUC was strong but sidecar lift vs v616 was only `+0.00000339`, so unchanged G124 is low EV.

5. **Calibrated no-call/background detector.**
   - Distinct but currently blocked by trusted negative/no-call validation.
   - Broad negative coverage improved, but the aux branch hurt matched soft-only control.

6. **Late-day guarded slot-fill queue.**
   - Activate only near reset if no verifier-grade candidate appears.
   - Avoid exact v616/v617/v620 replays, duplicate descriptions/matrices, malformed finals, static public-output-only finals, sample fallbacks, and known bad-value outputs.

## Bottom line
- **Submit now? No.** `0/5` slots are available, but there is no verifier-grade candidate.
- **Best next no-slot model data point after 16:20Z:** controlled DyMN10 context robustness (`r2/no-file/regularized`) and, if already produced, a no-submit ledger + sidecar-audit comparison against the original context MLP.
- **Strategic direction:** stop native B0 as-is; stop generic sequence postprocessor tweaks; use the context/DyMN10 evidence to decide whether a careful sidecar wrapper is worth building before reset.
