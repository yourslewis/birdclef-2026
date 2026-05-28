# 2026-05-28 02:21 UTC — PANNs embedding+AudioSet-tag sidecar audit

## Evaluated candidate
- **Source model:** `soundscape-sequence-panns-cnn14-embclip-allcls-r2-nofile-reg-losite-ep18-20260528`
- **Wrapper:** 234-class all-class leave-site OOF predictions mapped to v616 proxy rows with anchor fill.
- **Proxy coverage:** 240 proxy rows, 156 matched sequence rows, 42 valid AUC classes.

## Best sidecar recipe
- **Recipe:** `allcls_seq_w0p0025`
- **Local macro AUC:** `0.990515`
- **Lift vs anchor:** `+0.000124`
- **Lift vs v616:** `-0.002966`
- **Rank corr vs v616:** `0.999693`
- **MAE vs v616:** `0.005983`
- **Gate:** one or more promotion gates failed; eligible `False`.

## Decision
**Reject slot candidate.** The sidecar is finite/nonconstant but loses to v616 by `-0.002966` and adds less anchor lift than the PANNs localmax-only sidecar. No Kaggle submission.

## Artifacts
- Audit summary: `artifacts/soundscape_allclass_sidecar_audit/20260528T0220Z_panns_embclip_allclass_sequence/audit_summary.json`
- Candidate/audit dir: `artifacts/soundscape_allclass_sidecar_audit/20260528T0220Z_panns_embclip_allclass_sequence/`
- Audit log: `logs/soundscape_allclass_sidecar_panns_embclip_20260528T0220Z.log`
