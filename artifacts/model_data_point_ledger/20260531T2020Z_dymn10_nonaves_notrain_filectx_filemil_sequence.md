# DyMN10 non-Aves/no-train file-context/file-MIL sequence data point — 2026-05-31 20:20 UTC

## Purpose
After v651/v652 SED-heavy EoS8 source forks scored below the `0.950` public frontier, return to the data-driven lane: official `train_soundscapes` sequence/file/site mining. This run tests whether EfficientAT DyMN10 embeddings benefit from the same radius-2 temporal context + file mean/max + file-MIL protocol previously tried with PANNs.

## Configuration
- Experiment id: `soundscape-sequence-dymn10-nonaves-notrain-r2-filectx-filemil-losite-ep22-20260531`
- Data: official `train_soundscapes`; `1,478` windows, `66` files, `9` sites.
- Target scope: `72` non-Aves or no-train labels.
- Features/model: frozen EfficientAT DyMN10 embeddings; radius-2 temporal context, local mean/max, file mean/max, time features; MLP h384, dropout `0.40`.
- Training: leave-one-site; `22` epochs; batch `80`; LR `3.5e-4`; weight decay `0.0015`; file-MIL loss weight `0.40`; site-balanced sampling.
- Runtime/export: trainer GPU1; TorchScript head smoke OK; final 72/72 columns nonconstant; context feature dim `6,724`.

## Comparable performance

| Metric | Context/file model | Row-only same run | Delta |
|---|---:|---:|---:|
| Row macro AUC mean | 0.641802 | 0.559198 | +0.082604 |
| File-MIL macro AUC mean | 0.745704 | 0.529143 | +0.216561 |
| No-train AUC mean | 0.514582 | 0.516859 | -0.002277 |
| Non-Aves AUC mean | 0.641802 | 0.559198 | +0.082604 |

Fold deltas were positive on all six valid held-out sites: S03 `+0.111183`, S08 `+0.000792`, S13 `+0.114829`, S19 `+0.115934`, S22 `+0.005815`, S23 `+0.147074`.

## Baseline comparison
- Vs prior DyMN10 72-label context baseline (`2026-05-26 10:20`): row `+0.040447`, file-MIL `+0.113577`.
- Vs PANNs 72-label row-only (`2026-05-31 08:20`): row `-0.032683`, file-MIL `+0.054548`.
- Interpretation: this is a strong file-context/MIL signal for DyMN10, but still weaker row-wise than the best targeted PANNs row-only branch. It is useful as a landscape point and possible file-MIL component, not a direct submission.

## 72→234 sidecar audit
Audit artifact: `artifacts/model_data_point_ledger/20260531T2019Z_dymn10_nonaves_notrain_filectx_filemil_sidecar_audit/audit_summary.json`.

Best scoped sidecar recipe:
- `seq_context_w01`: local macro AUC `0.991206` / `42` valid classes.
- Lift vs v616 local proxy: `-0.002275`.
- Lift vs anchor: `+0.000816`.
- Rank corr vs v616: `0.999676`.
- `submit_approved=false`.

## Decision
Reject as a slot candidate. Keep as a measured sequence/file/site data point: DyMN10 benefits substantially from file context and MIL in held-site validation, but its anchor-filled 72→234 proxy sidecar remains below v616 and should not be submitted.
