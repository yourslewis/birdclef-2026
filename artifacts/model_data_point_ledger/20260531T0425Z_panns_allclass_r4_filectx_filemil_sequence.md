# PANNs all-class r4 20s local + file-context/file-MIL sequence data point — 2026-05-31 04:25 UTC

## Experiment
- **Experiment id:** `soundscape-sequence-panns-cnn14-allcls-r4-filectx-filemil-losite-ep20-20260531`
- **Branch family:** train_soundscapes sequence/file/site AudioSet mining
- **Goal:** combine the best prior PANNs all-class file-context/file-MIL lane with 20s-radius local mean/max temporal context, instead of choosing local context or file context separately.
- **Config:** `configs/birdclef/soundscape_sequence_panns_cnn14_allcls_r4_filectx_filemil_losite_ep20_20260531.json`
- **Model artifact dir:** `artifacts/soundscape_sequence_mining/soundscape-sequence-panns-cnn14-allcls-r4-filectx-filemil-losite-ep20-20260531`
- **Sidecar audit dir:** `artifacts/soundscape_sequence_sidecar_audits/panns_allclass_r4_filectx_filemil_20260531T0425Z`

## Data / targets / model
- **Training data:** official `train_soundscapes` only, `1,478` 5s windows, `66` files, `9` sites.
- **Target scope:** all `234` submission labels.
- **Embedding/init:** cached PANNs/Cnn14 AudioSet soundscape embeddings; MLP context head.
- **Context:** radius `4` (20s local window), prev/next, local mean, local max, file mean, file max, time features; no site one-hot.
- **Training:** leave-one-site validation, site-balanced sampling, BCE with pos-weight power `0.35`, file-MIL loss weight `0.40`, 20 epochs/fold + 20 final epochs.
- **Runtime:** ~`138.0s` summed trainer training time; final all-row head `19.38s`.

## Metrics
| Metric | Value |
|---|---:|
| Context row macro AUC mean | `0.640758` |
| Row-only macro AUC mean | `0.614224` |
| Context minus row-only | `+0.026534` |
| Context file-MIL macro AUC mean | `0.667273` |
| Context no-train AUC mean | `0.622672` |
| Context non-Aves AUC mean | `0.678785` |
| Final nonconstant columns | `234/234` |

Fold deltas (context minus row): S03 `+0.044174`, S08 `-0.012385`, S13 `+0.030298`, S15 `+0.036531`, S19 `+0.044126`, S22 `+0.011088`, S23 `+0.031908`.

## Comparisons
- Versus prior `PANNs/Cnn14 all-class filectx+fileMIL` (2026-05-30 00:20): row AUC `0.640758` vs `0.644272` (`-0.003514`); file-MIL `0.667273` vs `0.678888` (`-0.011615`).
- Versus prior `PANNs/Cnn14 all-class r4 20s localmeanmax` (2026-05-30 20:20): row AUC `0.640758` vs `0.627559` (`+0.013199`); file-MIL `0.667273` vs `0.673926` (`-0.006653`).
- Interpretation: adding file context/MIL to the 20s-radius branch rescued row AUC substantially vs local-only r4 but still did not beat the shorter r2 file-context/file-MIL baseline on file-MIL, the most relevant sequence aggregate.

## Sidecar audit vs v616 proxy
Best sidecar recipe from `birdclef_soundscape_allclass_sidecar_audit.py`:

- `allcls_seq_w0p0025`: local macro AUC `0.991456` / `42` valid classes.
- Lift vs anchor: `+0.001066`.
- Lift vs v616: `-0.002024`.
- Rank corr vs v616: `0.999691`; MAE vs v616: `0.005994`.
- Promotion gate: failed; `submit_approved=false`.

## Decision
Reject as a direct slot candidate. Keep as a useful landscape point: 20s temporal context improves over local-only r4 on row AUC, but the strongest measured PANNs sequence branch remains r2 file-context/file-MIL, and all PANNs sequence sidecar audits remain below v616 on the local proxy.
