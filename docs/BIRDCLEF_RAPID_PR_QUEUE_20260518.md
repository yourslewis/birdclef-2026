# BirdCLEF rapid PR queue — 2026-05-18

Status: active after Wenhao urgency request.

## Goal

Increase review velocity without spending Kaggle slots blindly. Keep multiple small PRs open, each tied to one distinct hypothesis and one clear validation gate.

## Current scoreboard anchor

- Public best remains `0.946`.
- `v572` cw0.75 local-window B0 sidecar tied `0.946`.
- Micro-sidecars are hidden-safe in some cases but have not broken the plateau.

## PR lanes to keep moving

1. **Stop-rule / score accounting PRs**
   - Purpose: immediately record LB outcomes and kill/keep decisions.
   - Gate: live Kaggle Bearer API evidence.

2. **Frame-head SED pilot PRs**
   - Purpose: move beyond rank perturbations into actual frame/event signal.
   - First configs: 10s/160mel and 20s/160mel B0 frame-head pilots with refreshed q3 B0 initialization.
   - Gate: holdout macro AUC, TorchScript size, runtime, and low-correlation audit before packaging.

3. **Packaging PRs**
   - Purpose: repo-owned kernels only, no direct public notebooks unless output preflight proves hidden/code format.
   - Gate: `submission.csv` plus sidecar/proto/sed outputs and required log markers before competition submit.

4. **External/source audit PRs**
   - Purpose: find genuinely new sources or rare/non-bird coverage rather than recycling public946 micro-sidecars.
   - Gate: verified local audio coverage by species/taxon, not just manifest row counts.

5. **OOF/stability helper PRs**
   - Purpose: improve rejection filters for local optimism.
   - Gate: grouped bootstrap + leave-one-group results, with explicit note that positive local stability is not an approval signal after v560/v572.

## Cadence rule

- Keep the autonomous loop hourly while this urgency mode is active.
- Prefer one focused branch per hypothesis.
- Do not wait to bundle unrelated docs/configs into one giant PR.
- Never stage generated logs, checkpoints, model artifacts, or Kaggle output CSVs.

## Current next best actions

1. Collect the 10s frame-head SED pilot result.
2. If it passes smoke, launch the 20s sibling.
3. If either is promising, open a packaging-prep PR only after a blend/correlation audit.
4. Use remaining Kaggle slots only for repo-owned kernels with verified output shape/log markers.
