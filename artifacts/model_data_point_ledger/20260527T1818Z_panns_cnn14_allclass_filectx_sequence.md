# Model data point — PANNs/Cnn14 all-class r2 file-context sequence — 2026-05-27 18:18 UTC

## Status
- **Experiment id:** `soundscape-sequence-panns-cnn14-allcls-r2-filectx-reg-losite-ep18-20260527`
- **Branch family:** train_soundscapes sequence/file/site AudioSet mining
- **Evidence level:** comparison-grade no-slot validation + local v616 proxy rejection
- **Decision:** keep as measured data point; **reject direct sidecar submission**; do not prioritize over PANNs no-file for true hidden-test package.

## Training / validation contract
- **Data:** official `train_soundscapes`; 1,478 5s windows / 66 files / 9 sites.
- **Target scope:** all taxonomy labels; 234 classes.
- **Model/init:** frozen PANNs/Cnn14 AudioSet embeddings + radius-2 context MLP with prev/next, local mean/max, **file mean/max**, time features; hidden_dim=384, dropout=0.45, seed=59.
- **Validation split:** leave-one-site; 7 completed folds; min valid classes gate `2`.
- **Export/runtime:** TorchScript smoke produced `context_head_torchscript.pt`; final predictions finite/nonconstant 234/234 columns.

## Comparable performance table

| Metric | Row-only baseline | File-context model | Delta |
|---|---:|---:|---:|
| Row macro AUC | 0.616097 | 0.642202 | 0.026105 |
| File-MIL macro AUC | 0.677553 | 0.652651 | -0.024902 |
| No-train AUC | — | 0.667300 | — |
| Non-Aves AUC | — | 0.707500 | — |
| v616 proxy sidecar lift | — | -0.002314 | vs v616 |

## Baseline comparisons
- Versus **PANNs/Cnn14 all-class r2 no-file**: row -0.005614, file-MIL -0.018072.
- Versus **fused DyMN10+PANNs no-file**: row 0.045560, file-MIL -0.023331.
- Fold deltas vs row-only: S03 0.102151, S08 -0.001386, S13 -0.090476, S15 0.049713, S19 0.042204, S22 0.042678, S23 0.037849.

## Sidecar audit vs v616
- Wrapped leave-site OOF context predictions into a 234-class v616 proxy sidecar.
- Proxy rows: 240; matched sequence rows: 156; anchor-filled rows: 84; finite=True; nonconstant columns=234.
- Best recipe `allcls_seq_w0p0025`: local macro AUC 0.991167 / 42 valid classes; lift vs anchor 0.000776; lift vs v616 -0.002314; rank corr vs v616 0.999693.
- Promotion gate: failed. This is not a submission-grade candidate.

## Critic / verifier notes
- File context improved S03/S19/S22/S23 and almost held S08, but regressed S13 heavily and degraded file-MIL versus row-only and the prior no-file PANNs all-class model.
- The sidecar remains an OOF proxy wrapper, not a hidden-test package; direct slot use would probe the LB with a known local loss vs v616.
- Next useful action is not another OOF sidecar weight sweep. Build a true hidden-test PANNs/no-file or fused package, or pivot to no-call/acoustic-background protocol.

## Artifacts
- Training artifact root: `artifacts/soundscape_sequence_mining/soundscape-sequence-panns-cnn14-allcls-r2-filectx-reg-losite-ep18-20260527/`
- Sidecar audit root: `artifacts/soundscape_sequence_sidecar_audit/20260527T1818Z_panns_filectx/`
- Config: `configs/birdclef/soundscape_sequence_panns_cnn14_allcls_r2_filectx_reg_losite_ep18_20260527.json`
