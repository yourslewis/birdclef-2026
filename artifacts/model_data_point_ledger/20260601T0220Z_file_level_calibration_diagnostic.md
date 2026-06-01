# File-level calibration/mapping diagnostic — 2026-06-01 02:20 UTC

## Purpose
Early UTC-day no-slot ClawTeam data point after the fused DyMN10+PANNs file-context run showed useful file-MIL signal but raw 72→234 sidecars remained below v616. This diagnostic tests whether file-level MIL evidence can be mapped back onto the best row-wise PANNs head instead of submitting another blind low-weight row sidecar.

## Configuration
- **Experiment id:** `file-level-calibration-mapping-diagnostic-20260601T0215Z`
- **Branch family:** train_soundscapes sequence/file/site file-level calibration/mapping diagnostic.
- **Data:** official `train_soundscapes`; `1,478` windows, `66` files, `9` sites. OOF validation rows: `1,314` windows from six valid held-out sites.
- **Target scope:** `72` non-Aves/no-train labels.
- **Inputs/models:** existing leave-site OOF predictions from:
  - PANNs/Cnn14 72-label row-only (`20260531T0820Z`).
  - DyMN10 72-label radius-2 file-context/file-MIL (`20260531T2020Z`).
  - Fused DyMN10+PANNs 72-label file-context/file-MIL (`20260601T0029Z`).
- **Mapping:** logit-space combinations of PANNs row predictions plus per-file mean/max evidence from DyMN10/fused/PANNs. No new neural training; this is a calibration/evaluation data point.
- **Validation split:** leave-one-site fold metrics, matching the sequence-training ledgers. Also audited selected mappings as 72→234 anchor-filled proxy sidecars against v616.
- **Runtime/export:** local CPU diagnostic and audit completed; finite/nonconstant 240x234 sidecar CSVs emitted; no hidden/Kaggle package.

## Comparable performance

| Metric | Best file-cal mapping | Baseline / comparator | Delta |
|---|---:|---:|---:|
| Row macro AUC mean | `0.687350` (`pannsrow + 35% DyMN10 file-mean`) | PANNs row-only `0.674485` | `+0.012865` |
| File-MIL AUC for same best-row mapping | `0.757145` | DyMN10 filectx/fileMIL `0.745704` | `+0.011441` |
| Best file-MIL in grid | `0.784044` (`pannsrow + 50% DyMN10 file-max`) | DyMN10 filectx/fileMIL `0.745704` | `+0.038340` |
| No-train AUC for best-row mapping | `0.599442` | PANNs row-only `0.600481` | `-0.001039` |
| Non-Aves AUC for best-row mapping | `0.687350` | PANNs row-only `0.674485` | `+0.012865` |

## Top comparable 72-label sequence/file/site row-AUC points

| Rank | Experiment | Row AUC | File-MIL AUC | Notes |
|---:|---|---:|---:|---|
| 1 | File-cal mapping: PANNs row + DyMN10 file mean 35% | `0.687350` | `0.757145` | New best row/file combined diagnostic; no hidden package. |
| 2 | PANNs/Cnn14 72-label row-only | `0.674485` | `0.691156` | Previous best row-wise targeted model. |
| 3 | Fused DyMN10+PANNs 72-label filectx/fileMIL | `0.652377` | `0.722866` | Improved fused row-only but not PANNs row. |
| 4 | DyMN10 72-label filectx/fileMIL | `0.641802` | `0.745704` | Previous best file-MIL clue. |
| 5 | PANNs/Cnn14 72-label filectx/fileMIL | `0.631592` | `0.690005` | Context hurt row vs PANNs row-only. |

## 72→234 sidecar audit vs v616

Selected file-calibrated mappings were wrapped into 234-class anchor-filled proxy sidecars and rank-blended against the v616 anchor.

- Best audited recipe: `pannsrow__dymn_filemax__a20_w02`.
- Local macro AUC: `0.991112` / `42` valid local classes.
- Lift vs anchor: `+0.000722`.
- Lift vs v616: `-0.002368`.
- Rank corr vs v616: `0.999613`; MAE `0.006451`.
- `submit_approved=false`: below v616 local proxy and not a hidden-safe package.

## Critic / verifier decision
- **Critic:** This is strategically useful because it confirms the file-MIL signal is not an illusion; mapping DyMN10 file evidence onto PANNs row predictions improves comparable row/file validation. But the 72→234 proxy sidecar still degrades vs v616, so the signal is not yet slot-worthy.
- **Verifier:** No external submission. Sidecar CSVs are finite/nonconstant and schema-audited, but they are leave-site OOF proxy artifacts, not hidden-test inference packages. Reject as submission-grade.
- **Decision:** keep as a measured landscape point; no Kaggle slot. Next exact action is to convert this into a hidden-safe inference/package path or use it as a local teacher for a compact train_soundscape-native student with stronger class/file calibration gates.

## Artifacts
- Diagnostic script: `scripts/birdclef_file_level_calibration_diagnostic.py`
- Metrics: `artifacts/model_data_point_ledger/20260601T0220Z_file_level_calibration_diagnostic/metrics.json`
- Sidecar audit: `artifacts/model_data_point_ledger/20260601T0220Z_file_level_calibration_diagnostic/audit_summary.json`
- Manifest and sidecars: `artifacts/model_data_point_ledger/20260601T0220Z_file_level_calibration_diagnostic/manifest.json`, `.../sidecars/`
