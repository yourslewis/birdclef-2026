# BirdCLEF Hill-Climb Ranked Queue — 2026-05-26 16:20 UTC

## Live status
- Coordinator status: early/mid UTC-day no-slot training pass; no leaderboard submission approved.
- Kaggle Bearer API check: best known public LB remains **0.949**. Latest scored submissions: `v616=0.949`, `v617=0.949`, `v618=0.946`, `v619=0.944`, `v620=0.949`.
- UTC slot usage at check: **0/5** used on 2026-05-26; ~7.66h until reset.
- Active jobs: no active local BirdCLEF/Kaggle job found; no active BirdCLEF trainer process found after this run.
- Git branch: `feature/birdclef-20260524-20utc-v612-submit`.

## Public scout / critic refresh
- Report-only scout/critic artifact: `specs/birdclef-hillclimb-cron-20260525/reports/scout_critic_20260526T1615Z.md`.
- Scout found no fresh clean Kaggle/public model lead above the tied `0.949` plateau; current public candidates remain plateau, duplicate, malformed/static-risk, or asset/license preflight lanes.
- Critic decision: **PROCEED WITH REVISION** on a bounded compact soundscape-native CNN/SED data point; reject another shallow TCN/gating postprocessor as the immediate next action; reject early-day Kaggle submission.

Evidence level: **comparison-grade scout/critic synthesis** plus verified no-slot training artifact.

## Action taken this run — model data point
Trained a compact deeper soundscape-native CNN/SED branch directly on official `train_soundscapes` logmels instead of frozen embeddings.

- Script: `scripts/birdclef_soundscape_native_losite_train.py`
- Config: `configs/birdclef/soundscape_native_b0_losite_nonaves_notrain_ep4_20260526.json`
- Artifact root: `artifacts/soundscape_native_losite/soundscape-native-b0-losite-nonaves-notrain-ep4-20260526/`
- Model family: EfficientNet-B0 SED-style compact CNN, full compact-backbone fine-tune from repo q3/cap80 train-audio checkpoint, fresh 72-label head.
- Data: 1,478 official train-soundscape 5s windows, 66 files, 9 sites, 72 non-Aves/no-train labels, 5,420 scoped positive cells.
- Training: leave-site folds over S03/S08/S13/S15/S18/S19/S22/S23 with min-row/min-valid-class gates, site/file held out by construction; BCE, observed sqrt pos weights capped at 12, label smoothing 0.01, mixup 0.1, 4 epochs, then final all-row export smoke.

Results:
- Completed folds: `6`; skipped `2` (`S15` too few valid classes, `S18` too few windows).
- Leave-site row macro AUC mean: `0.558044` over completed folds.
- Leave-site no-train row AUC mean: `0.573554`.
- Leave-site file-MIL macro AUC mean: `0.429828`.
- Pooled leave-site row AUC: `0.396540`; pooled no-train AUC: `0.305887`.
- Comparison: underperforms the current best context-MLP sequence artifact (`0.601355` row / `0.632127` file-MIL). It is a useful data point but not a wrapper/submission candidate.

Verifier checks:
- Leave-site prediction artifact shape: `1314 x 72`.
- Predictions finite/nonconstant: `72/72` nonconstant columns, min `0.00140194`, max `0.998872`, std `0.146312`.
- TorchScript export: `15.184 MB`; smoke shapes `2x160x313 -> 2x72` and finite.
- ONNX export/check: `exported_checked`, `0.568 MB`.
- Not competition-format; no 234-class wrapper; no v616 audit. **No submission approved.**

## Ranked queue after this run

1. **Context-MLP sequence artifact robustness / wrapper decision (ACCEPTED next control)**
   - Expected LB potential: medium if converted to a cautious 234-class sidecar, but only after S03/S22 regression guard.
   - Evidence: still best train-soundscape sequence artifact (`0.601355` row, `0.632127` file-MIL). Native B0 and TCN/gated variants did not beat it.
   - Next exact experiment: one-variable regularized context-MLP ablation or worst-site/objective guard; then decide whether a 72->234 wrapper/audit is worth building.

2. **AudioSet reformulation into multi-site features / 234-class sidecar wrapper**
   - Expected LB potential: medium as a broad acoustic/no-call/rare-slice sidecar.
   - Evidence: DyMN10 remains the strongest frozen AudioSet source and powers the best context artifact; avoid more single-site shallow heads.
   - Next action if chosen: multi-site feature/audit wrapper, not another S08 MLP.

3. **Fresh pretrained asset preflight (YAMNet / SurfPerch / PaSST / HTS-AT / BEATs / CLAP)**
   - Expected LB potential: medium/high if clean and decorrelated, but setup/license/runtime risk is the blocker.
   - Action: asset/license/runtime preflight only, then same leave-site soundscape protocol.

4. **G124/V2S hard-confidence / target-power ablation**
   - Expected LB potential: medium but lower after tiny prior v616-sidecar lift (`+0.00000339`).
   - Only proceed if target contract changes materially.

5. **Calibrated no-call/background detector**
   - Expected LB potential: low/medium but distinct.
   - Needs trusted negative protocol; broad negative aux coverage improved but hurt matched soft-only control.

6. **Late-day guarded slot-fill queue**
   - Activate near reset if no verifier-grade package appears. Avoid exact duplicates of `v616`/`v617`/`v620`, malformed/static/public-output-only candidates, and known bad finals.

## Critic decision
**REVISE, do not submit.** The deeper native branch answered the representation question: compact direct logmel fine-tuning does not beat the frozen-DyMN10 context-MLP sequence artifact and has weak pooled no-train behavior. Do not spend an early-day slot.

## Verifier decision
**ACCEPTED as no-slot artifact; REJECTED as submission.** Training completed, artifacts are finite/nonconstant, and TS/ONNX exports pass. The branch is 72-label only, weaker than the sequence context baseline, not v616-audited, and not competition schema.
