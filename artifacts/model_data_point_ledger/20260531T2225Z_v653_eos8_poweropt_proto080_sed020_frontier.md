# v653 EoS8 PowerOpt proto080/sed020 frontier verifier

UTC: 2026-05-31 22:30Z  
Family: EoS8/PowerOptimization source-winner frontier fork  
Source: `ryutoyoda/birdclef-2026-exp013-eos8-sidecar` lineage, private repo fork `bc26-v653-eos8-proto080-sed020-verifier` v1.

## Candidate
- Description: `v653: EoS8 PowerOpt proto080 sed020 frontier verifier`
- Weight: `PROTO_RANK_WEIGHT=0.800`, `SED_RANK_WEIGHT=0.200`.
- Scope: 234 submission labels; hidden-safe code rerun from public source with official competition data only.

## Local verifier
- Data: 240 v616 proxy rows, 190 label-matched rows, 20 files, 6 sites, 234 labels / 42 valid AUC classes.
- Primary local metric: `0.990031` macro AUC / 42 valid.
- Delta vs v616 local proxy: `-0.003450`.
- Non-Aves AUC: `0.992810` / 31 valid.
- No-train AUC: `0.995513` / 19 valid.
- Site/file q05 lift vs v616: `-0.011526` / `-0.007532`.
- Rank corr / MAE vs v616: `0.963329` / `0.063863`.

## Runtime / submission verifier
- Kaggle public-session status: COMPLETE before submission.
- Output preflight: rows `3`, cols `235`, unique rows `3`, bad values `0`, uniq_first100 `96`, min/max `0.4544277` / `0.5185959`.
- Competition submission ref: `53232636`.
- Public LB score: `0.947` (scored by 2026-06-01 00:16 UTC).
- Public delta: -0.003 vs live public best 0.950 / -0.002 vs v616-era 0.949.

## Decision
Rejected hidden fork; proto-heavy frontier underperformed public best.
