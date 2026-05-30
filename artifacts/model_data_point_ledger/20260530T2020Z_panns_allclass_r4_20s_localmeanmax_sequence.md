# Model Data Point — PANNs all-class r4 20s local mean+max sequence

Timestamp: 2026-05-30 20:20 UTC

## Summary

Trained a distinct 20s-radius temporal-context data point using frozen PANNs/Cnn14 embeddings on official `train_soundscapes`. This tests whether a wider local window (±20s / radius 4) with local mean+max pooling can improve over the prior radius-2 localmax/file-context variants without using file-level context.

Result: the wider 20s local context is slightly better than its row-only baseline but weaker than the best prior PANNs all-class temporal variants. Context row AUC is `0.627559` versus row-only `0.626128` (`+0.001430`); no-train `0.604764`, non-Aves `0.665293`, file-MIL `0.673926`. Gains are site-skewed (`S03`, `S08`, `S23`) and regress `S13`, `S15`, `S19`, and slightly `S22`.

The v616 sidecar audit is negative: best `allcls_seq_w0p0025` local AUC `0.991131` / 42 valid classes, lift vs anchor `+0.000741` but lift vs v616 `-0.002349`. No submission.

## Ledger

- **Branch family:** sequence/file/site AudioSet temporal context mining / 20s local context branch.
- **Training data:** official `train_soundscapes`, `1,478` windows / `66` files / `9` sites.
- **Target scope:** all `234` competition labels.
- **Model/init:** frozen PANNs/Cnn14 AudioSet embeddings; context MLP head with current embedding + local mean + local max over radius-4 windows + time features; hidden dim 384; dropout 0.40; 20 epochs; site-balanced sampling.
- **Validation split:** leave-one-site over 7 valid sites.
- **Primary metric:** context row macro ROC-AUC `0.627559` over 7 folds.
- **Secondary metrics:** no-train AUC `0.604764`; non-Aves AUC `0.665293`; file-MIL AUC `0.673926`; row-only baseline row `0.626128`, file-MIL `0.671220`; context-row delta `+0.001430`; sidecar lift vs v616 `-0.002349`.
- **Baseline/delta:** vs prior best PANNs all-class filectx+fileMIL row `0.644272`: `-0.016713`; vs prior PANNs localmax-only row `0.641501`: `-0.013942`; vs row-only internal baseline: `+0.001430` row / `+0.002706` file-MIL.
- **Export/runtime status:** TorchScript context head exported; final predictions finite/nonconstant across 234 columns; sidecar candidate CSV finite/nonconstant; audit complete.
- **Decision:** **reject/no submission.** Useful negative/diagnostic data point: wider 20s PANNs context helps weakly over row-only but does not transfer to v616 and is below prior PANNs variants.

## Artifacts

- Config: `configs/birdclef/soundscape_sequence_panns_cnn14_allcls_r4_20s_localmeanmax_losite_ep20_20260530.json`
- Training output: `artifacts/soundscape_sequence_mining/soundscape-sequence-panns-cnn14-allcls-r4-20s-localmeanmax-losite-ep20-20260530/`
- Metrics: `artifacts/soundscape_sequence_mining/soundscape-sequence-panns-cnn14-allcls-r4-20s-localmeanmax-losite-ep20-20260530/metrics.json`
- Sidecar audit: `artifacts/model_data_point_ledger/20260530T2020Z_panns_allclass_r4_20s_localmeanmax_sidecar_audit/audit_summary.json`
