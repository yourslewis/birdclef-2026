# BirdCLEF no-call suppression sidecar audit — 2026-05-29 00:26 UTC

## Experiment

- **Experiment id:** `soundscape-nocall-suppression-v616-agg-20260529T0035Z`
- **Branch family:** No-call/background suppression sidecar verifier
- **Input signal:** aggregate no-call gate from `soundscape-nocall-gate-soft1279native-agg-losite-20260528`
- **Gate data:** 792 official train_soundscape 5s windows / 66 files / 9 sites; 739 labeled any-call windows and 53 weak unlabeled/background windows.
- **Suppression target:** v616 tied baseline proxy matrix, 240 rows x 234 labels; evaluated on the local v616 train_soundscape proxy with 190 matched label rows and 42 valid AUC classes.
- **Candidate grid:** 16 bounded multiplicative sidecars: OOF/final no-call probability × {all labels, non-Aves/no-train labels} × alpha {0.01, 0.02, 0.04, 0.08}. Best recipe was `nocall_final_nonaves_notrain_p1p0_a010`.

## Results

```text
recipe                               local AUC  lift v616  lift anchor  top5 recall  decision
nocall_final_nonaves_notrain_a010    0.993546   +0.000066  +0.003156   0.626316     reject gates
nocall_oof_nonaves_notrain_a010      0.993520   +0.000040  +0.003130   0.626316     reject gates
v616_baseline                        0.993481   +0.000000  +0.003090   0.636842     control
nocall_final_all_a010                0.992741   -0.000739  +0.002351   0.636842     reject
```

Best candidate (`final` no-call, non-Aves/no-train scope, alpha 0.01):

- Local macro AUC: `0.9935463889` / 42 valid classes.
- Lift vs v616 local proxy: `+0.0000657213`.
- Lift vs anchor: `+0.0031558824`.
- Rank corr vs v616: `0.9999842022`; MAE vs v616 `0.0005655196`.
- Site bootstrap q05 vs v616: `+0.0000542411`; file bootstrap q05 vs v616: `+0.0000046898`.
- Leave-one-site min vs v616: `+0.0000705982`; leave-one-file min vs v616: `+0.0000419720`.
- Top-5 row recall regressed vs v616 (`0.626316` vs `0.636842`).

## Critic / verifier decision

- **Decision:** reject as submission-grade; keep comparison-grade data point.
- The best suppression recipe is finite/nonconstant and directionally positive vs v616, but the lift is microscopic and far below promotion gates (`+0.001` vs v616, site/file q05 thresholds, and lift-vs-anchor threshold).
- Suppressing all classes is harmful; the only useful signal is a very conservative non-Aves/no-train cap.
- Weak background negatives are still site-skewed (mostly S09/S18/S22) and not hand-verified, so this cannot justify an early-day Kaggle slot.

## Artifacts

- Script: `scripts/birdclef_nocall_suppression_sidecar_audit.py`
- Audit summary: `artifacts/soundscape_nocall_suppression/soundscape-nocall-suppression-v616-agg-20260529T0035Z/audit_summary.json`
- Full audit JSON: `artifacts/soundscape_nocall_suppression/soundscape-nocall-suppression-v616-agg-20260529T0035Z/audit/ensemble_strategy_audit.json`
- Candidate CSVs: `artifacts/soundscape_nocall_suppression/soundscape-nocall-suppression-v616-agg-20260529T0035Z/candidates/`

## Next exact action

Do not submit no-call suppression. Next useful branch is either (a) hand-audit/upgrade the no-call negative protocol, or (b) train a distinct soundscape-native/domain-adaptation data point with stronger signal than a tiny v616 suppression cap.
