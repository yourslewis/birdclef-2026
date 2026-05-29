# Ranked Queue — BirdCLEF ClawTeam Hill-Climb — 2026-05-29 02:25 UTC

## Live state

- Public best: `0.949` (v616/v621-v623/v634 tied; no new better score).
- Latest scored submissions: v631 `0.926`, v632 `0.940`, v633 `0.946`, v634 `0.949`, v635 `0.941`.
- UTC slots used today: `0/5`; ~21.6h to reset at post-run check.
- Active jobs: no local/trainer BirdCLEF jobs after run; trainer GPUs free.
- Git branch: `feature/birdclef-20260524-20utc-v612-submit`.

## Actions this run

1. Added `train_sampling=site_balanced` to `scripts/birdclef_soundscape_native_losite_train.py`.
2. Trained `soundscape-native-b0-soft1279init-sitebalanced-losite-allcls-ep4-20260529` as a one-variable site-balanced ablation of the head-loaded soft1279 native all-class model.
3. Packaged/audited the final TorchScript model through the train_soundscapes inference path and v616 local proxy sidecar grid.
4. Updated canonical performance table artifacts and new model/audit ledgers.
5. No submission: early UTC day and the package audit fell below v616 local baseline.

## Ranked queue after this run

1. **Do not continue site-balanced soft1279 native sampling unchanged** — row AUC `0.569405` and file-MIL `0.513779`; sidecar best `0.993104`, lift vs v616 `-0.000377`. Site-balanced sampling degraded both training and sidecar transfer.
2. **Upgrade no-call/background labels before any suppression submit** — current suppression is the only positive current-day direction vs v616, but lift is only `+0.000066`; needs stricter/hand-verified negatives or less site-skewed background labels.
3. **Soft1279 head-loaded w0.16 remains best repo-owned local sidecar but held** — local AUC `0.995545`, lift vs v616 `+0.002064`; still failed strict lift-vs-anchor/site gates and should not spend early-day slots.
4. **Next distinct model data point: calibration not sampling** — if training instead of no-call audit, test a conservative calibration/temperature or smaller LR/shorter adaptation branch, not site-balanced resampling.
5. **Public/source-code refresh only for late-day slots** — recent public fills mostly underperformed; save early slots unless a clearly stronger source appears.

## Compact performance table for this run

```text
experiment / candidate                         metric                         vs baseline           decision
soft1279 site-balanced native LOSO             row AUC 0.569405 / 7 folds     -0.030955 vs soft1279 reject
                                                file-MIL 0.513779              -0.092026 vs soft1279
soft1279 site-balanced sidecar w0.16            local AUC 0.993104 / 42 valid  -0.000377 vs v616    reject
                                                lift vs anchor +0.002713       -0.002441 vs old w0.16
```

## Top comparable local sidecar/package audits

```text
candidate / recipe                    local AUC  lift vs v616   status
soft1279init native w0.16             0.995545   +0.002064      hold; strict gates failed
soft1279init native w0.08             0.994813   +0.001332      hold; strict gates failed
soft1279 obspos package w0.16         0.993906   +0.000425      reject
nocall suppression nonaves α0.01      0.993546   +0.000066      reject; tiny lift
v616 baseline                         0.993481   +0.000000      tied public best
soft1279 site-balanced w0.16          0.993104   -0.000377      reject
```

## Critic / verifier

- Critic: site balancing was a valid one-variable test because prior soft1279 gains looked site-skewed, but the result shows the gain depended on the original adaptation distribution; resampling reduced both LOSO and proxy transfer.
- Verifier: package sidecar matched 240/240 proxy rows, finite/nonconstant 240x234, and audit mechanics passed. `submit_approved=false`; best non-control blend is below v616 and failed promotion gates.
- Submission decision: **not approved**. Early-day slots remain unused.

## Artifacts

- Training ledger: `artifacts/model_data_point_ledger/20260529T0225Z_soundscape_native_soft1279init_sitebalanced_allclass.md`
- Package ledger: `artifacts/model_data_point_ledger/20260529T0225Z_soft1279init_sitebalanced_package_audit.md`
- Metrics: `artifacts/soundscape_native_losite/soundscape-native-b0-soft1279init-sitebalanced-losite-allcls-ep4-20260529/metrics.json`
- Audit summary: `artifacts/sed_soundscape_packaging_audit/20260529T0225Z_soft1279init_sitebalanced_package/audit_summary.json`
- Full audit: `artifacts/sed_soundscape_packaging_audit/20260529T0225Z_soft1279init_sitebalanced_package/audit/ensemble_strategy_audit.json`
- Config: `configs/birdclef/soundscape_native_b0_soft1279init_sitebalanced_losite_allcls_ep4_20260529.json`

## Next exact action

Either hand-audit/upgrade no-call negatives for a stronger suppression verifier, or train a calibration-focused soft1279 branch with smaller adaptation movement instead of more site-balanced resampling.
