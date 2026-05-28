# 20260528T0823Z — PANNs all-class localmax file-MIL 234-class sidecar audit vs v616

## Summary
- **Input model:** `soundscape-sequence-panns-cnn14-allcls-r2-localmax-filemil-losite-ep20-20260528` leave-site OOF predictions, wrapped directly as a 234-label all-class sidecar into the v616 proxy matrix.
- **Proxy rows:** 156/240 matched; 84 anchor-filled.
- **Output guard:** finite `240x234`, 234 nonconstant columns.

## Best recipe
- Recipe: `allcls_seq_w0p0025`
- Local proxy macro AUC: **0.991363** / 42 valid classes
- Lift vs anchor: **+0.000973**
- Lift vs v616: **-0.002117**
- Rank correlation vs v616: **0.999693**
- MAE vs v616: **0.005968**

## Decision
**Reject as slot candidate; keep as integration evidence.** The sidecar improves the raw anchor but remains below the tied v616 local proxy by -0.002117; early-day promotion gates fail.

## Artifacts
- Audit root: `artifacts/soundscape_sequence_sidecar_audit/20260528T0820Z_panns_localmax_filemil/`
- Summary: `artifacts/soundscape_sequence_sidecar_audit/20260528T0820Z_panns_localmax_filemil/audit_summary.json`
- Sidecar CSV: `artifacts/soundscape_sequence_sidecar_audit/20260528T0820Z_panns_localmax_filemil/sidecars/allclass_sequence_sidecar_234.csv`
