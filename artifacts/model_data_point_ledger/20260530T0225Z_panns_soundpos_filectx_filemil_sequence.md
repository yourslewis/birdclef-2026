# PANNs soundscape-positive file-context + file-MIL sequence data point — 20260530T0225Z

## Status
- Evidence level: **comparison-grade** no-slot train/eval + v616 proxy sidecar audit.
- Live state at start: public best `0.949`; v636-v640 scored `0.944/0.943/0.939/0.944/0.945`; 2026-05-30 UTC slots `0/5`; no active local/trainer BirdCLEF jobs.
- Decision: **reject/no submission**. The branch is a useful negative data point: soundscape-positive scope plus heavy file context underperformed its own row-only baseline and did not improve v616 proxy.

## Experiment
- Experiment id: `soundscape-sequence-panns-cnn14-soundpos-r2-filectx-filemil-losite-ep22-20260530`
- Branch family: train_soundscapes sequence/file/site AudioSet soundscape-positive target mining.
- Training data: official train_soundscapes `1,478` 5s windows / `66` files / `9` sites.
- Target scope: soundscape-positive labels, `75` classes.
- Model/init: frozen PANNs/Cnn14 AudioSet embeddings + radius-2 prev/next/local/file mean+max/time context MLP; file-MIL BCE weight `0.35`; site-balanced sampling; 22 epochs; seed 73.
- Validation split: leave-one-site, `7` valid folds.

## Metrics

| Item | Value |
|---|---:|
| Context row macro AUC | `0.610622` |
| Row-only row macro AUC | `0.632460` |
| Context - row-only | `-0.021838` |
| Context file-MIL AUC | `0.646776` |
| Row-only file-MIL AUC | `0.678655` |
| Context no-train AUC | `0.574690` |
| Context non-Aves AUC | `0.639635` |
| Final nonconstant columns | `75/75` |

Fold deltas context-row: S03 -0.151290, S08 -0.028133, S13 -0.056737, S15 -0.005836, S19 +0.064709, S22 +0.013778, S23 +0.010644.

## Sidecar audit vs v616
- Audit path: `artifacts/soundscape_sequence_sidecar_audits/panns-soundpos-r2-filectx-filemil-losite-ep22-20260530`
- Proxy rows: `240` total, `156` matched train_soundscape proxy rows, `234` columns; finite/nonconstant columns `234`.
- Best non-control recipe: `seq_context_w01` local macro AUC `0.990561` / `42` valid classes.
- Lift vs v616: `-0.002919`; lift vs anchor `+0.000171`; rank corr vs v616 `0.999679`; MAE `0.006186`.
- Submit approved: `false` (fails lift-vs-v616 and bootstrap promotion gates).

## Top comparable soundscape-positive / PANNs-context rows

| Rank | Model | Row AUC | File-MIL AUC | Sidecar lift vs v616 | Note |
|---:|---|---:|---:|---:|---|
| 1 | Native B0 soundpos LOSO | 0.658165 | 0.676383 | -0.001930 | Best soundpos row; still sidecar-negative vs v616 |
| 2 | PANNs soundpos localmax | 0.642375 | 0.662504 | -0.002292 | Earlier PANNs soundpos temporal branch |
| 3 | PANNs soundpos filectx+fileMIL (this) | 0.610622 | 0.646776 | -0.002919 | File context hurt row/file vs row-only |
| 4 | DyMN10 soundpos filectx | 0.518121 | 0.512164 | -0.002816 | Weak sequence target redesign |
| 5 | PANNs all-class filectx+fileMIL | 0.644272 | 0.678888 | -0.002529 | Comparable PANNs context but all 234 labels |

## Critic / verifier
- Critic: this variant answered a useful question but was probably over-contextualized; S03/S13/S08 regressed enough that the file context/MIL objective is not a good soundscape-positive wrapper.
- Verifier: no external submission. Sidecar is finite/nonconstant and hidden-safe as analysis artifact, but it is proxy-negative vs v616 and ineligible for slot use.

## Artifacts
- Config: `configs/birdclef/soundscape_sequence_panns_cnn14_soundpos_r2_filectx_filemil_losite_ep22_20260530.json`
- Metrics: `artifacts/soundscape_sequence_mining/soundscape-sequence-panns-cnn14-soundpos-r2-filectx-filemil-losite-ep22-20260530/metrics.json`
- OOF predictions: `artifacts/soundscape_sequence_mining/soundscape-sequence-panns-cnn14-soundpos-r2-filectx-filemil-losite-ep22-20260530/leave_site_predictions.npz`
- Sidecar audit: `artifacts/soundscape_sequence_sidecar_audits/panns-soundpos-r2-filectx-filemil-losite-ep22-20260530/audit_summary.json`
