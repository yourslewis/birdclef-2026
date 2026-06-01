# BirdNET baseline source provenance

- `baseline_code.py`: code cells extracted from the public Kaggle kernel
  `ahmadzulfiqar001/birdclef-2026-birdnet-baseline` (BirdNET-Analyzer v2.4 via birdnetlib),
  pulled live via the Kaggle REST `kernels/pull` endpoint using `Authorization: Bearer`
  with the new `KGAT_` token (the installed kaggle CLI cannot auth that token format).
- Wiring/label-mapping reference (not re-hosted here):
  `yaroslavkholmirzayev/birdnet-third-branch-site-hour-prior-restore`.
- Our forked/extended inference: `scripts/birdnet_proxy_infer.py` (dense per-segment
  probability vectors + canonical proxy schema alignment).
