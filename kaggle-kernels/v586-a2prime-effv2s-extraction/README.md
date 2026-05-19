# v586 A2Prime EffV2S extraction scaffold

Prepared as a repo-owned fallback behind v585 FrankSunP.

Source: `claudedevore/birdclef-2026-r0946-a2prime-effv2s-submit` v5.

Audit summary from 2026-05-19 14:58 UTC:

- Kernel COMPLETE/no failure.
- Primary `submission.csv` sample output: `3 x 235`, finite, unique row IDs.
- EffV2S branch: 4 CPU folds, sanity top-5 hit rate 0.55.
- Proto/EffV2S rank correlation: 0.053.
- NFNet fallback was deprioritized because v581 no-scored on hidden timeout and NFNet sanity/correlation were weaker.

Do not push/submit this while `birdclef-v585-reset` owns the next reset slot.
If v585 improves, abandon this fallback and port/confirm FrankSunP instead.
