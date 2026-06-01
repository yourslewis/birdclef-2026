# Fused DyMN10+PANNs non-Aves/no-train file-context/file-MIL sequence data point — 2026-06-01 00:29 UTC

## Purpose
Early UTC-day no-slot data point after v653/v654/v655 completed below the live `0.950` public best. This run tests whether the complementary recent train-soundscape signals combine: PANNs row-only was strongest row-wise, while DyMN10 file-context/file-MIL was strongest at file-MIL.

## Configuration
- Experiment id: `soundscape-sequence-fused-dymn10-panns-nonaves-notrain-r2-filectx-filemil-losite-ep22-20260601`
- Data: official `train_soundscapes`; `1,478` windows, `66` files, `9` sites.
- Target scope: `72` non-Aves or no-train labels.
- Features/model: fused frozen DyMN10 + PANNs/Cnn14 embeddings; radius-2 temporal context, local mean/max, file mean/max, time features; MLP h384, dropout `0.40`.
- Training: leave-one-site; `22` epochs; batch `80`; LR `3.5e-4`; weight decay `0.0015`; file-MIL loss weight `0.40`; site-balanced sampling.
- Runtime/export: trainer GPU1; TorchScript head smoke OK; final 72/72 columns nonconstant; context feature dim `21,060`.

## Comparable performance

| Metric | Context/file model | Row-only same run | Delta |
|---|---:|---:|---:|
| Row macro AUC mean | 0.652377 | 0.620622 | +0.031755 |
| File-MIL macro AUC mean | 0.722866 | 0.695848 | +0.027018 |
| No-train AUC mean | 0.567523 | 0.522501 | +0.045022 |
| Non-Aves AUC mean | 0.652377 | 0.620622 | +0.031755 |

Fold row-AUC deltas were positive on five of six valid held-out sites: S03 `+0.056029`, S08 `+0.006144`, S13 `-0.013518`, S19 `+0.087306`, S22 `+0.018043`, S23 `+0.036525`.

## Baseline comparison
- Vs fused 72-label row-only (`2026-05-31 12:16`): row `+0.036211`, file-MIL `-0.001051`.
- Vs DyMN10 72-label file-context/file-MIL (`2026-05-31 20:20`): row `+0.010575`, file-MIL `-0.022838`.
- Vs PANNs 72-label row-only (`2026-05-31 08:20`): row `-0.022108`, file-MIL `+0.031710`.
- Interpretation: fusion + file context improves the fused row-only model and substantially improves no-train AUC, but it does not beat the best single-family row (PANNs) or file-MIL (DyMN10) clue.

## 72→234 sidecar audit
Audit artifact: `artifacts/model_data_point_ledger/20260601T0029Z_fused_nonaves_notrain_filectx_filemil_sidecar_audit/audit_summary.json`.

Best scoped sidecar recipe:
- `seq_context_w02`: local macro AUC `0.990914` / `42` valid classes.
- Lift vs v616 local proxy: `-0.002567`.
- Lift vs anchor: `+0.000523`.
- Rank corr / MAE vs v616: `0.999611` / `0.006475`.
- `submit_approved=false`.

## Decision
Keep as a measured sequence/file/site data point; reject as direct or sidecar slot candidate. Next exact action: build a file-level calibration/mapping diagnostic from the best row/file candidates rather than another raw low-weight 72→234 sidecar.
