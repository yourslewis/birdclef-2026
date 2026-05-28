# BirdCLEF hill-climb ranked queue — 2026-05-28 00:22 UTC

## Live state
- Best public LB remains **0.949** (v616/v617/v621/v622/v623 tied lineage; v629 latest best late fill at 0.946).
- 2026-05-28 UTC slots used: **0/5** at start of run; ~23.7h to reset.
- 2026-05-27 late-fill scores now complete: v626 0.899, v627 0.928, v628 0.940, v629 0.946, v630 0.917. None beat/tie best.
- Active jobs: no BirdCLEF local/trainer jobs at launch; trainer GPU 1 was free; unrelated LRM job occupied GPU 0.

## Role synthesis
- **Coordinator:** early-day slot policy says do not burn fresh slots without verifier-grade/high-info candidate; train/evaluate next data point instead.
- **Data/Feature:** train_soundscapes sequence/file/site remains the richest under-mined official data; localmax-only ablation isolates whether temporal max pooling carries useful signal.
- **Validation:** localmax context improves average row/file metrics but one held-out site (S19) regresses; sidecar proxy still below v616, so evidence is comparison-grade only.
- **Prediction/Ensemble:** localmax sidecar has the best sequence-family lift vs anchor so far (+0.00136) but still loses vs v616 (-0.00173); not a slot candidate.
- **Critic:** direct OOF proxy sidecars repeatedly fail because they cover only 156/240 proxy rows and optimize a narrow local proxy. Next improvement must be hidden-test packaging or a real no-call/background protocol, not more tiny sidecar weights.
- **Verifier:** finite/nonconstant artifacts passed; no hidden-test package/submission approval.

## Ranked queue

1. **PANNs all-class no-file hidden-test package/eval** — highest immediate value. Build inference path that computes PANNs embeddings/context on test soundscapes, not OOF proxy row replacement. Gate with schema/runtime/dedup and compare candidate displacement vs v616.
2. **No-call/acoustic-background protocol** — high diversity. Start with trusted negative/background audit and any-call/no-target target design; avoid assuming unmatched rows are negatives.
3. **PANNs localmax integration redesign** — use today’s localmax-only result as a signal clue; investigate S19 regression and whether localmax can be class/taxon/site-capped without v616 loss.
4. **Fused/PANNs file-MIL branch packaging or calibrated file-level pooling** — file-MIL remains competitive (fused 0.675982; localmax 0.681753) but direct row proxy fails.
5. **Deeper soundscape-native regularized variant** — revisit only with better pooled/objective design; observed-positive weighting regressed.
6. **Late-day public/source slot-fill** — only inside <3h to reset and only after fresh source-clean audit; latest v626-v630 all below best.

## Submission decision
No submission this run. Slots are fresh (0/5) and the evaluated candidate failed v616 promotion (`-0.001728` local lift vs v616).
