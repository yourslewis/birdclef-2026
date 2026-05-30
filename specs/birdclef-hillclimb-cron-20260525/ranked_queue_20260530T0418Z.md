# Ranked Queue — BirdCLEF Hill-Climb — 2026-05-30 04:18 UTC

## Live state
- Public best: `0.949`; v616 remains the tied baseline to beat.
- Latest completed submissions: v636-v640 scored `0.944 / 0.943 / 0.939 / 0.944 / 0.945`.
- 2026-05-30 UTC slots: `0/5` used; ~19.7h to reset at live check, so early-day policy applies.
- Active jobs: none local/trainer; trainer GPUs free.

## This-run data point / evaluation
- Evaluated `soft1279-capped-perclass-selector-retry-20260530T0418Z` on 240 v616 proxy rows / 42 valid classes.
- Head-loaded low-cap selector: site-CV AUC `0.993558`, lift vs v616 `+0.000077`; file-CV AUC `0.993699`, lift `+0.000218`.
- Multi-soft1279 low-cap selector: site/file AUC `0.993532` / `0.993673`; lifts `+0.000051` / `+0.000193`.
- All-row diagnostic lift only `+0.000348` using four classes; top-3 recall unchanged.
- Decision: **reject/no submission**. Capping fixes instability by mostly zeroing the sidecar, not by making the soft1279 movement robust.

## Ranked next queue
1. **No-call negative audit upgrade / hand-stricter background protocol** — current farneg10 suppression lift is only `+0.000084`; needs better negatives or no-call sources before slot use.
2. **Soft1279 head-loaded movement diagnosis, class/site attribution only** — do not retry low-cap selectors; instead identify which classes/sites create the global `w0.16` lift and why strict gates fail.
3. **No-train sonotype/file-MIL diagnostic** — fused no-train file-MIL remains a clue (`0.660711`) but sidecars are anchor-flat; inspect class-site inversion before more wrappers.
4. **Broader acoustic-context/no-call feature branch** — use AudioSet outputs as contextual suppression/no-call features, not direct scoped class sidecars.
5. **Late UTC guarded source fill** — if <3h to reset and no verifier-grade repo candidate exists, use only nonduplicate public-source candidates that pass schema/runtime/dedup guards.

## Critic / verifier decision
- Critic: the capped selector answered the main question negatively; small per-class caps cannot preserve the useful soft1279 lift. Further selector grid search would be low-value.
- Verifier: no slot. The evaluated recipes are finite and reproducible, but the best site-CV lift is only `+0.000077` vs v616, well below promotion gates and not submission-grade.
