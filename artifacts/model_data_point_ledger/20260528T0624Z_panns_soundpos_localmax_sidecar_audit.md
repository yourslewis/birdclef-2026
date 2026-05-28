# 20260528T0624Z — PANNs soundscape-positive localmax 75→234 sidecar audit vs v616

## Summary
- **Input model:** `soundscape-sequence-panns-cnn14-soundpos-localmax-losite-ep20-20260528` leave-site OOF predictions, mapped as a 75-label sidecar into the v616 234-class proxy matrix.
- **Proxy rows:** 156/240 matched; 84 anchor-filled.
- **Output guard:** finite `240x234`, 234 nonconstant columns.

## Best recipe
- Recipe: `seq_context_w01`
- Local proxy macro AUC: **0.991188** / 42 valid classes
- Lift vs anchor: **+0.000798**
- Lift vs v616: **-0.002292**
- Rank correlation vs v616: **0.999676**

## Decision
**Reject as slot candidate; keep as comparison/integration clue.** The best sidecar blend improves the raw anchor but is below the tied v616 proxy by -0.002292, so it fails the early UTC-day promotion gate.

## Artifacts
- Audit root: `artifacts/soundscape_sequence_sidecar_audit/20260528T0624Z_panns_soundpos_localmax/`
- Summary: `artifacts/soundscape_sequence_sidecar_audit/20260528T0624Z_panns_soundpos_localmax/audit_summary.json`
- Sidecar CSV: `artifacts/soundscape_sequence_sidecar_audit/20260528T0624Z_panns_soundpos_localmax/sidecars/seq_context_sidecar_234.csv`
