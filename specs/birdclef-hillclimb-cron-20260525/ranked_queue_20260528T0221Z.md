# BirdCLEF hill-climb ranked queue — 2026-05-28 02:21 UTC

## Live state
- Best public LB remains **0.949** (v616/v617/v621/v622/v623 tied lineage).
- Latest scored submissions: v626 `0.899`, v627 `0.928`, v628 `0.940`, v629 `0.946`, v630 `0.917`; none beat/tie best.
- 2026-05-28 UTC slots used: **0/5** at start; ~21.6h to reset.
- Active jobs before launch: no BirdCLEF jobs locally/trainer; unrelated LRM job on GPU0, GPU1 free.

## Role synthesis
- **Coordinator:** early-day slot policy says preserve slots unless verifier-grade/high-info candidate appears; train/evaluate next data point instead.
- **Data/Feature:** concatenating AudioSet tag logits with embeddings tested whether broad acoustic event semantics add signal beyond PANNs embeddings.
- **Validation:** emb+tag context regressed row AUC (`0.609194` vs PANNs no-file `0.647816`) and file-MIL (`0.668715` vs localmax `0.681753`). Evidence is rejection/comparison-grade.
- **Prediction/Ensemble:** sidecar best lift vs v616 `-0.002966`; weaker than localmax sidecar and below v616.
- **Critic:** this result argues against blind tag-logit concatenation. If broad acoustic/no-call remains a target, design a trusted no-call/background protocol rather than more feature concatenation.
- **Verifier:** finite/nonconstant artifacts and TorchScript export passed; sidecar audit not eligible; no submission.

## Comparable top-5 all-class sequence row-AUC data points
1. PANNs/Cnn14 no-file context — `0.647816` row / `0.670723` file-MIL.
2. PANNs/Cnn14 file-context — `0.642202` row / `0.652651` file-MIL.
3. PANNs/Cnn14 localmax-only — `0.641501` row / `0.681753` file-MIL.
4. Soundscape-native B0 all-class — `0.636161` row / `0.673756` file-MIL.
5. Soundscape-native B0 observed-pos — `0.624340` row / `0.582914` file-MIL.
- New emb+tag run: `0.609194` row / `0.668715` file-MIL, outside top 5 by row and below top file-MIL options.

## Ranked queue
1. **PANNs all-class hidden-test package/eval** — still highest immediate value; do actual test-soundscape inference rather than OOF proxy replacement.
2. **No-call/acoustic-background protocol audit** — define trusted negatives/any-call target first; high diversity, but do not assume unmatched rows are negatives.
3. **PANNs localmax integration redesign** — best file-MIL and best sequence-family sidecar lift so far; diagnose S19 and cap class/taxon movement.
4. **Fused/PANNs calibrated file-level pooling** — file-MIL remains competitive; package only with hidden-safe path.
5. **Deeper soundscape-native pooled/objective redesign** — native B0 is competitive but pooled metrics are poor; observed-pos weighting regressed.
6. **Late-day public/source slot-fill** — only inside <3h to reset after fresh source-clean audit.

## Submission decision
No submission. The trained/evaluated candidate failed both local CV and v616 sidecar gates; slots remain **0/5 used** for 2026-05-28 UTC.
