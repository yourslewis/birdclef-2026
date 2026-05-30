# Ranked Queue — BirdCLEF Hill-Climb — 2026-05-30 02:25 UTC

## Live state
- Public best: `0.949` (v616/v621-v623/v634 tied lineage; v616 remains baseline to beat).
- Latest completed submissions: v636-v640 scored `0.944 / 0.943 / 0.939 / 0.944 / 0.945`.
- 2026-05-30 UTC slots: `0/5` used at run start; early-day policy applies.
- Active jobs: none after this run; trainer GPUs free before training.

## This-run data point
- Trained `soundscape-sequence-panns-cnn14-soundpos-r2-filectx-filemil-losite-ep22-20260530`.
- Result: row AUC `0.610622`, file-MIL `0.646776`, no-train `0.574690`, non-Aves `0.639635`.
- Sidecar best `seq_context_w01`: local AUC `0.990561`, lift vs v616 `-0.002919`.
- Decision: **reject/no submission**; file/context-MIL soundscape-positive scope is worse than row-only and prior soundpos branches.

## Ranked next queue
1. **Soft1279 head-loaded sidecar movement diagnosis / capped selector retry** — highest actionable clue remains the 2026-05-28 head-loaded soft1279 native package audit (`+0.002064` local lift vs v616) but failed site/lift-vs-anchor gates; diagnose per-class/site/file movement, cap unstable classes, and only package if robustness improves.
2. **No-call negative audit upgrade** — hand/stricter background negatives or multi-source no-call labels; current farneg10 suppression lift is only `+0.000084` vs v616 and top5 recall regresses.
3. **No-train sonotype/file-MIL diagnostic** — fused/PANNs no-train branches show file-MIL clues but sidecars are anchor-flat; inspect class-site inversion and file pooling before more wrappers.
4. **Broader acoustic-context feature branch** — use AudioSet outputs as no-call/context features rather than scoped class sidecars; require multi-site validation.
5. **Late UTC slot fill** — if <3h to reset and no verifier-grade repo candidate exists, use guarded nonduplicate public source candidates only after schema/runtime/dedup preflight.

## Critic / verifier decision
- Critic: stop adding blind PANNs soundpos context variants; evidence now says file context/MIL hurts this target scope.
- Verifier: no slot. Candidate is finite/nonconstant but proxy-negative vs v616 and not submission-grade.
