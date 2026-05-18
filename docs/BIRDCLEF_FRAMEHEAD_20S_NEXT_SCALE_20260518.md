# BirdCLEF frame-head 20s next scale — 2026-05-18

Status: queued after 20s/2048-file scale result.

## Evidence

The 20s frame-head B0 pilot is currently the strongest non-public946-micro-sidecar signal in this run:

- 10s / 1024-file pilot: macro AUC `0.723326` over `94` valid classes.
- 20s / 1024-file pilot: macro AUC `0.806310` over `92` valid classes.
- 20s / 2048-file / 8-epoch scale: macro AUC `0.902068` over `144` valid classes.

## Next config

`configs/birdclef/sed_b0_framehead_20s_m160_q3init_ep12_4096_20260518.json`

Changes from the 2048-file scale:

- `max_files`: `2048 -> 4096`
- `max_classes`: `160 -> 220`
- `files_per_class`: `14 -> 20`
- `epochs`: `8 -> 12`
- seed `94`

Everything else stays on the winning path: 20s context, 160 mel bins, refreshed q3 B0 external init, focal BCE gamma `1.5`, sqrt positive weights, label smoothing `0.005`, restore-best-by-val-loss.

## Gate

Do not package this model directly after training. First require:

1. holdout AUC improves or remains competitive as class coverage expands;
2. TorchScript remains small enough for Kaggle CPU sidecar use;
3. a blend/correlation audit against public946 shows additive value beyond local optimism.
