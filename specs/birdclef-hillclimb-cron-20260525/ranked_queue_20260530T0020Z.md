# Ranked queue — 2026-05-30T00:20Z

## Live status
- Best public LB: `0.949` (v616 tied baseline; v634 latest tie). v636-v640 completed below best: `0.944/0.943/0.939/0.944/0.945`.
- UTC slots used today: `0/5`; early UTC-day policy active, so preserve slots for verifier-grade/high-info candidates.
- Active jobs checked before run: no local/trainer BirdCLEF jobs.

## Current run result
- Trained/evaluated `soundscape-sequence-panns-cnn14-allcls-r2-filectx-filemil-losite-ep20-20260530`.
- LOSO context row AUC `0.644272`; file-MIL `0.678888`; sidecar lift vs v616 `-0.002529`.
- Decision: no submission; below v616 proxy and fails promotion gates.

## Ranked next actions
1. **Soft1279 head-loaded sidecar class/site movement diagnosis + constrained selector verifier** — highest immediate value because the original head-loaded native sidecar remains the only recent local proxy lift > v616 (`+0.002064`) but failed strict site/anchor gates. Need class/site attribution, negative controls, and a safer class-capped recipe before slot use.
2. **No-call/background negative audit upgrade** — hand/stricter far-negative protocol or teacher-agreement negatives before another suppression model; current no-call gates are comparison-grade but weak-negative-risky.
3. **PANNs/DyMN10 sequence file-MIL diagnostics by class/site** — mine why file-MIL improves local folds but wrapper sidecars regress vs v616; target S08/S15/S22/S23 and non-Aves/no-train inversion.
4. **Deeper soundscape-native compact variant with stronger regularization but not another soft1279 knob** — only if it changes objective/data (e.g., mix of native all-class + no-call calibrated targets), not more head-only/site-balanced repeats.
5. **Late-day guarded source fill** — only if <3h to reset and no verifier-grade repo candidate exists; avoid v636-v640 families unless new source evidence appears.

## Critic/verifier stance
- Critic: the PANNs file-context/file-MIL experiment was worth measuring, but not worth a slot; proxy sidecar is below v616 and highly correlated.
- Verifier: all artifacts finite/nonconstant; no hidden/test labels; no static/fallback submission; `submit_approved=false`.
