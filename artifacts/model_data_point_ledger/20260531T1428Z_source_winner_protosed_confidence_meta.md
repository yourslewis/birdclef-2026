# Source-winner Proto/SED confidence meta audit — 2026-05-31 14:28 UTC

## Scope
Audited the v644/v647 EoS8 source-winner intermediate train-soundscape streams (`submission_protossm.csv`, `submission_sed.csv`) against the submitted v616 local proxy. This run also trained a tiny per-class leave-site logistic meta calibrator to test whether Proto/SED/source-rank confidence features add robust held-site signal.

## Data and validation
- Data: 240 v616/source-winner proxy rows, 190 label-matched train-soundscape rows, 20 files, 6 sites (`S03`, `S08`, `S09`, `S13`, `S18`, `S22`).
- Target scope: 234 BirdCLEF labels; 42 valid local AUC classes; 72 non-Aves taxonomy classes available for slice metrics.
- Split: local v616 proxy plus leave-one-site OOF for the meta calibrator. Sidecar grid uses local proxy metric with 20-iteration site/file bootstrap smoke.
- Baselines: v616 final local macro AUC `0.993481`; anchor raw local macro AUC `0.990391`.

## Performance
```text
model / stream             metric                         value       delta vs v616
source_sed raw             local macro AUC / 42 valid     0.995976    +0.002495
source_proto raw           local macro AUC / 42 valid     0.986299    -0.007182
source_rankblend           local macro AUC / 42 valid     0.992723    -0.000758
leave-site meta OOF        local macro AUC / 42 valid     0.990463    -0.003018
best v616+SED rank blend   local macro AUC / 42 valid     0.996059    +0.002578
```

Secondary metrics:
- Source SED non-Aves AUC `0.996866`; top-5 row recall `0.994737`.
- Best sidecar: `0.20*v616 + 0.80*source_sed` in rank space, candidate `artifacts/source_winner_confidence_meta_audit/20260531T1428Z/candidates/v616_rankblend_source-sed_w0p8.csv`.
- Best sidecar rank corr vs v616 `0.857333`, MAE `0.114691`, top-5 recall `0.742105`.
- Site bootstrap lift smoke q05 `+0.000450`; file bootstrap lift q05 `+0.000083` (20 iterations only).
- Leave-site meta trained `206` site-class logistic models but fell back on `38,504` cells and underperformed v616, so the trainable confidence calibrator is rejected as currently formulated.

## Critic / verifier decision
`submit_approved=false`.

The source SED stream is the strongest local clue in the current v950-family audit and deserves a private kernel/source-fork verifier. However, this run did not build a hidden-test package, and the local v616 proxy is already known to over-promote SED-like sidecars. The high-weight SED blend is also a large displacement from v616 (`rank_corr=0.857`), so it should not consume an early-day competition slot as a static/proxy-only CSV. Next action: fork/private-verify the EoS8 source with an explicit SED/PowerOpt weight ablation, or build a hidden-safe verifier that emits the same source SED stream on held/public/hidden audio.

## Artifacts
- Audit report: `artifacts/source_winner_confidence_meta_audit/20260531T1428Z/audit_report.md`
- Summary JSON: `artifacts/source_winner_confidence_meta_audit/20260531T1428Z/audit_summary.json`
- Candidate CSVs: `artifacts/source_winner_confidence_meta_audit/20260531T1428Z/candidates/`
- Script: `scripts/birdclef_source_winner_confidence_meta_audit.py`
