# Scout + Critic Report — BirdCLEF hill-climb cron — 20260526T1020Z

## Fresh scout
- Web searches for `BirdCLEF 2026 EfficientAT/PANNs/DyMN10`, `0.949/0.950 public solution`, and Bioacoustics/SurfPerch/Perch-style model leads did not reveal a clean new direct public notebook above the current `0.949` plateau.
- Returned public-code leads are already-known or adjacent to Perch/ProtoSSM/SED/Nina/EoS families; these are saturated as direct submissions.
- Bioacoustics/SurfPerch/Perch2-style resources remain model-family research leads only; require license/source/runtime checks and no static-output dependency.

## Critic decision
**Proceed with train_soundscapes sequence/file/site mining, not another isolated-row AudioSet head.** The user correction is strategically right: the under-mined asset is sequence/file/site structure in official train soundscapes.

## Result of accepted action
- New context branch improved leave-site mean row AUC `0.578422` -> `0.601355`.
- File-level MIL max-pool improved `0.563852` -> `0.632127`.
- Fold heterogeneity remains the blocker: S13/S19/S23 improved, S03/S22 worsened.

## Decision
- No submission now.
- Rank sequence-mining v2 above generic AudioSet repetition.
- Keep G124 hard-confidence/power ablation as the next non-sequence fallback.
