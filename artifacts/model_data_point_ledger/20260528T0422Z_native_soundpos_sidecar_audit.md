# 20260528T0422Z — Native soundscape-positive sidecar audit vs v616

## Summary
- **Input model:** `soundscape-native-b0-losite-soundpos-ep5-20260528` leave-site OOF predictions, mapped as a 75-label sidecar into the v616 234-class proxy matrix.
- **Proxy rows:** 156/240 matched; 84 anchor-filled.
- **Output guard:** finite `240x234`, 234 nonconstant columns.

## Best recipe
- Recipe: `seq_context_w04`
- Local proxy macro AUC: **0.991551** / 42 valid classes
- Lift vs anchor: **+0.001160**
- Lift vs v616: **-0.001930**
- Rank correlation vs v616: **0.999389**

## Decision
**Reject as slot candidate; keep as integration clue.** It improves over the raw anchor but remains below the tied v616 baseline, so it fails the promotion gate for early UTC-day submission.

## Artifacts
- Audit root: `artifacts/soundscape_sequence_sidecar_audit/20260528T0415Z_native_soundpos/`
- Summary: `artifacts/soundscape_sequence_sidecar_audit/20260528T0415Z_native_soundpos/audit_summary.json`
- Sidecar CSV: `artifacts/soundscape_sequence_sidecar_audit/20260528T0415Z_native_soundpos/sidecars/seq_context_sidecar_234.csv`
