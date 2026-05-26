# Ranked queue — BirdCLEF hill-climb cron — 20260526T1020Z

## Live state verified
- Best public LB remains **0.949**; `v616` remains the repo-owned tied baseline to beat.
- Latest Bearer API listing checked this run: `v617=0.949`, `v618=0.946`, `v619=0.944`, `v620=0.949`; no 2026-05-26 UTC submissions yet.
- UTC daily slots used: **0/5** at the live check. Early-day policy still requires verifier-grade/high-info competition-format candidates.
- Active job checks: no local BirdCLEF/Kaggle jobs; trainer only showed unrelated processes before training.

## Public scout refresh
- Fresh web search did not surface a clean new BirdCLEF 2026 public code lead above the `0.949` plateau.
- Search did resurface already-known plateau/Perch/Proto/SED/Nina/EoS families and general Bioacoustics/SurfPerch/Perch-style leads. Treat these as research/model-family leads, not direct slot candidates.

## Work completed this run
1. Prevented duplicate work by checking local/trainer processes, recent ranked queues, and model ledger.
2. Verified Kaggle submission state via Bearer API v1; daily slots remain `0/5`.
3. Implemented and trained the top corrected data-driven branch: `train_soundscapes` sequence/file/site mining with DyMN10 embeddings.
4. Wrote data diagnostics, leave-site fold metrics, file-MIL metrics, and verifier smoke artifacts.

## Current ranked queue

1. **Sequence/file/site mining v2 — residual temporal smoother / compact TCN** — **ACCEPTED next data point**. Current context branch gave mean LOSO lift `+0.022933` and file-MIL lift `+0.068275`, but S22/S03 regressions must be fixed before wrapper/audit.
2. **234-class DyMN10/sequence sidecar wrapper + v616 audit** — **NEEDS REVISION**. Only after multi-site sequence signal is stable; current output is 72-label only.
3. **G124/V2S hard-confidence / target-power ablation** — **ACCEPTED bounded data point**. Still valuable because prior G124 had strong proxy AUC but near-zero v616 lift.
4. **Fresh pretrained model-family scout/assets** — **ACCEPTED research lane**. BirdNET embeddings, Bioacoustics Model Zoo/Perch2/SurfPerch, PaSST/HTS-AT/BEATs remain possible sources if license/runtime/package checks pass.
5. **Late-day guarded exploratory fallback queue** — **DEFER until <3h reset**. Only source-clean, nonduplicate, nonmalformed candidates; no v616/v617/v620 replay.
6. **B0 soft-only raw sidecar** — **LOWER PRIORITY**. Strong B0 smoke but did not help v616 audit.

## Slot decision
No Kaggle submission. The new sequence branch is high-information and rule-safe, but not submission-format and not v616-audited. Early-day leaderboard probing remains rejected.

## Critic / verifier summary
- Critic: sequence context is the right corrected lane and shows signal, but fold heterogeneity prevents promotion. S22 dominates row count and got worse, so a naive context wrapper could hurt hidden.
- Verifier: finite/nonconstant leave-site predictions and TorchScript head smoke passed; no hidden-test output path exists; no slot approved.

## Next exact action
Run sequence-mining v2 with residual/regularized context (or a true per-file TCN/smoother) and explicit S22/S03 regression guard. If no packageable candidate exists near reset, construct late-day guarded fallback only then.
