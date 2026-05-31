# Fused DyMN10+PANNs all-class r4 20s localmeanmax — 2026-05-31 00:20 UTC

## Status
- Evidence level: comparison-grade no-slot data point.
- Decision: reject as direct slot/sidecar candidate; keep as a measured 20s-fusion temporal diagnostic.
- Submission: none. Early UTC day, 0/5 slots used, and sidecar is below v616/local proxy and new 0.950 public source baseline.

## Training / data
- Experiment: `soundscape-sequence-fused-dymn10-panns-allcls-r4-20s-localmeanmax-losite-ep20-20260531`
- Branch family: sequence/file/site AudioSet fusion 20s temporal-context mining.
- Data: official `train_soundscapes`, 1,478 windows / 66 files / 9 sites.
- Targets: all 234 taxonomy labels.
- Model/init: frozen fused EfficientAT DyMN10 + PANNs/Cnn14 embeddings; MLP context head over current embedding + ±20s local mean/max + time features.
- Split: leave-one-site, site-balanced sampling.

## Metrics
- Context row macro ROC-AUC: `0.581429` / 7 folds.
- Row-only baseline in same run: `0.559001`; delta `+0.022428`.
- File-MIL AUC: `0.659437`.
- No-train AUC: `0.524384`.
- Non-Aves AUC: `0.613439`.
- Per-site deltas: S03: -0.014373, S08: +0.019467, S13: +0.053499, S15: -0.006013, S19: +0.062766, S22: +0.074623, S23: -0.032974.

## Sidecar audit vs v616 proxy
- Audit output: `artifacts/soundscape_sequence_sidecar_audit/20260531T0020Z_fused_allclass_r4_20s_localmeanmax/`.
- Best recipe: `allcls_seq_w0p005`.
- Local macro AUC: `0.991115` / 42 valid classes.
- Lift vs v616: `-0.002366`.
- Lift vs anchor: `+0.000724`.
- Rank corr vs v616: `0.999672`; MAE `0.005939`.
- Promotion gate: failed; `submit_approved=false`.

## Interpretation
The 20s fused context improves its internal row-only control and helps S13/S19/S22, but it remains below the stronger PANNs r4/file-context sequence rows and below v616 in the sidecar audit. The model is useful as a negative data point: fusion plus wider local temporal context does not recover hidden-safe lift when wrapped as a low-weight all-class sidecar.

## Artifacts
- Training metrics: `artifacts/soundscape_sequence_mining/soundscape-sequence-fused-dymn10-panns-allcls-r4-20s-localmeanmax-losite-ep20-20260531/metrics.json`
- OOF predictions: `artifacts/soundscape_sequence_mining/soundscape-sequence-fused-dymn10-panns-allcls-r4-20s-localmeanmax-losite-ep20-20260531/leave_site_predictions.npz`
- Audit summary: `artifacts/soundscape_sequence_sidecar_audit/20260531T0020Z_fused_allclass_r4_20s_localmeanmax/audit_summary.json`
