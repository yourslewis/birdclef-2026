# v654 EoS8 PowerOpt proto070/sed030 frontier verifier

UTC: 2026-05-31 22:35Z  
Family: EoS8/PowerOptimization source-winner frontier fork  
Source: `ryutoyoda/birdclef-2026-exp013-eos8-sidecar` lineage, private repo fork `bc26-v654-eos8-proto070-sed030-verifier` v1.

## Candidate
- Description: `v654: EoS8 PowerOpt proto070 sed030 frontier verifier`
- Weight: `PROTO_RANK_WEIGHT=0.700`, `SED_RANK_WEIGHT=0.300`.
- Scope: 234 submission labels; hidden-safe code rerun from public source with official competition data only.

## Local verifier
- Data: 240 v616 proxy rows, 190 label-matched rows, 20 files, 6 sites, 234 labels / 42 valid AUC classes.
- Primary local metric: `0.991227` macro AUC / 42 valid.
- Delta vs v616 local proxy: `-0.002253`.
- Non-Aves AUC: `0.993664` / 31 valid.
- No-train AUC: `0.996156` / 19 valid.
- Site/file q05 lift vs v616: `-0.008898` / `-0.006307`.
- Rank corr / MAE vs v616: `0.983168` / `0.054012`.

## Runtime / submission verifier
- Kaggle public-session status: COMPLETE before submission.
- Output preflight: rows `3`, cols `235`, unique rows `3`, bad values `0`, uniq_first100 `96`, min/max `0.45959768` / `0.5250699`.
- Competition submission ref: `53232692`.
- Public LB score: `0.949` (scored by 2026-06-01 00:16 UTC).
- Public delta: -0.001 vs live public best 0.950 / +0.000 vs v616-era 0.949.

## Decision
Tied old v616-era baseline but below live 0.950; no further scalar frontier slot early-day.
