# Model Data Point — OOF-teacher B0 1024 ep4 + Broad Negative/No-Call Control

Timestamp: 2026-05-26 02:20 UTC

See canonical ledger: `artifacts/model_data_point_ledger/20260526T0220Z_oofteacher_broadneg.md`.

## One-line decision

The broader negative/no-call mask is now measured and operational, but the matched soft-only 1024-row control is better (`0.9111` vs `0.9083` macro AUC), so do not scale the aux branch unchanged; consider packaging the soft-only student for a no-slot v616 sidecar audit.

## Key metrics

- Broad mask: 1,259/1,279 rows, 230/234 classes, 47,343 capped negative cells, 0 false-negative cells.
- Soft-only control: 1,024 rows, 4 epochs, macro AUC `0.911067` over 122 classes; TorchScript/ONNX + CPU smoke passed.
- Broad-neg aux: 1,024 rows, 4 epochs, aux weight `0.01`, macro AUC `0.908278` over 122 classes; TorchScript/ONNX + CPU smoke passed.

## Decision

No submission. Evidence is comparison-grade and random-split only. The next promotion step is a hidden-safe raw sidecar/v616 audit for the soft-only model, or a distinct 20s temporal/localmax data point.
