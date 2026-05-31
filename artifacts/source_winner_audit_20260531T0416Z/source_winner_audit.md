# BirdCLEF 2026 source-winner audit — v644/v647 — 2026-05-31 04:30 UTC

## Scope
Audited the two newly scored public-LB `0.950` source-code submissions from 2026-05-30:

- `v644`: `yaroslavkholmirzayev/0950-replay`, submitted as `v644: Late-fill Yaroslav 0950 replay source`, public LB `0.950`.
- `v647`: `ryutoyoda/birdclef-2026-exp013-eos8-sidecar`, submitted as `v647: Late-fill Ryuto EoS8 sidecar source`, public LB `0.950`.

Kaggle kernel output retrieval used SDK `list_kernel_session_output`; direct legacy `/api/v1/kernels/output/...` endpoints returned `404` before this audit.

## Static source findings
Both notebooks are dominated by the same EoS8 taxonomy-smoothed anchor:

- `RUN_MODE = "eos8_tax"`.
- Active outer models:
  - `yukiZ_Perch_ProtoSSM_ResSSM`, weight `0.0305`.
  - `Karnakbayev_PowerOptimization_TaxRank`, weight `0.9695`.
- Internal PowerOptimization branch:
  - `PROTO_RANK_WEIGHT = 0.600`; `SED_RANK_WEIGHT = 0.400`.
  - `POWEROPT_PRIOR_LAMBDA = 0.65`.
  - `POWEROPT_FILE_CONFIDENCE_POWER = 0.40`.
  - `POWEROPT_RANK_AWARE_POWER = 0.65`.
  - `POWEROPT_DELTA_SMOOTH_ALPHA = 0.20`.
  - correction grid `[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]`.
  - taxonomy smoothing: genus alpha `0.15`, class alpha `0.05`.
- Source/runtime is hidden-test capable: reads `test_soundscapes`, writes `submission.csv`, and uses attached model/data assets rather than a static final CSV as the only path.

## Sidecar findings
- `v644` has an optional `exp098`/`exp001` sidecar probe enabled in source, but public dry-run diagnostics show it did **not** apply: `public_dryrun_no_audio=true`, `test_files_count=0`, skip reason: no `test_soundscapes` `.ogg` files found. Public dry-run `submission_before_exp098_exp001_sidecar.csv` is byte-identical to final `submission.csv`.
- `v647` has BirdNET and `exp002b_5s` sidecar controls, but public dry-run final is still unchanged after sidecar stages. `sidecar_exp002b_diagnostics.csv` reports `applicable=False`, `effective_weight=0.0`, `D=0.0`, and skip reason: no public dry-run audio rows could be matched by exp002b inference.
- Therefore, the public visible output difference between v644 and v647 is not meaningful evidence of a productive sidecar; it is essentially the same anchor path with tiny formatting/numeric differences.

## Output audit summary
Downloaded output files and audit summary:

- `artifacts/source_winner_audit_20260531T0416Z/session_outputs/audit_summary.json`
- `artifacts/source_winner_audit_20260531T0416Z/session_outputs/v644_yaroslav_0950/`
- `artifacts/source_winner_audit_20260531T0416Z/session_outputs/v647_ryuto_eos8_sidecar/`

Public session output comparison:

- Both final public dry-run submissions are valid finite/nonconstant `3 x 235` matrices.
- v644 final hash: `87e9ac1be51a0b15`.
- v647 final hash: `3db34d13e789ae73`.
- Cross-final comparison: row order equal; 234 common species columns; only `3` cells differ above `1e-9`; max absolute delta `1.0e-6`; flat correlation `0.999999999987`.
- Internal sidecar deltas for both: `0` changed cells from `submission_before_all_sidecars.csv` to final public `submission.csv`.

## Decision
- Treat `v644`/`v647` as a new public-best/tied family (`0.950`) but not as two independent directions.
- Do **not** spend a new slot on exact/near-duplicate reruns or static blends of these outputs.
- The exploitable signal is the EoS8/PowerOptimization/taxonomy-smoothed anchor; sidecar components need a private/verifier audio run before they count as distinct.
- Next source-audit action: inspect whether `exp001/exp002b` sidecars can be adapted into a repo-owned hidden-safe verifier or whether their public no-op behavior makes them low priority compared with train_soundscape sequence/file/site mining.
