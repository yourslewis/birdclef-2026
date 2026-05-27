# 2026-05-27 08:20 UTC — All-class DyMN10 sequence 234-class sidecar audit

## Status
- **Experiment:** `soundscape-sequence-allclass-sidecar-audit-20260527T0820Z`
- **Family:** train_soundscapes sequence/file/site 234-class sidecar wrapper audit
- **Evidence level:** comparison-grade no-slot local proxy audit
- **Submission decision:** reject as a slot candidate; no Kaggle submission.

## Data / wrapper contract
- Source model: `soundscape-sequence-dymn10-allcls-r2-nofile-reg-losite-ep18-20260527`.
- Proxy matrix: v616 anchor/v616 final dry-run outputs, 240 rows x 234 classes.
- Sequence OOF wrapper: 234-class leave-site context predictions from official train_soundscapes.
- Matched proxy rows: **156**; anchor-filled unmatched proxy rows: **84**.
- Sidecar validation: finite values, **234/234** nonconstant columns.

## Audit results
Best sidecar recipe by lift vs v616 was the tiniest tested rank blend:

- `allcls_seq_w0p0025`: local proxy macro AUC **0.991108** over **42** valid classes.
- Lift vs anchor reconstruction: **+0.000718**.
- Lift vs v616 final: **-0.002372**.
- Rank correlation vs v616: **0.999689**; MAE **0.006032**.
- Promotion gates failed; submit approval remained disabled.

All stronger tested weights (0.5%, 1%, 2%, 4%, 8%) were worse vs v616. This confirms the all-class sequence head is not currently useful as a direct low-weight v616 sidecar despite improving leave-site row/file-MIL metrics over row-only.

## Decision
**Reject slot / keep as comparison-grade data point.** The model remains useful for train_soundscapes sequence/file/site landscape learning, but local proxy audit says it should not consume an early-day Kaggle slot without a new hidden-safe formulation.

## Artifacts
- Audit root: `artifacts/soundscape_sequence_sidecar_audit/20260527T0820Z_allclass/`
- Sidecar CSV: `artifacts/soundscape_sequence_sidecar_audit/20260527T0820Z_allclass/sidecars/allclass_sequence_sidecar_234.csv`
- Audit summary: `artifacts/soundscape_sequence_sidecar_audit/20260527T0820Z_allclass/audit_summary.json`
- Audit JSON: `artifacts/soundscape_sequence_sidecar_audit/20260527T0820Z_allclass/audit/ensemble_strategy_audit.json`
