# Ranked Queue — BirdCLEF ClawTeam Hill-Climb — 2026-05-28 18:21 UTC

## Live state

- Public best: `0.949` (v616/v621-v623 family still tied best; v626-v630 below best at `0.899/0.928/0.940/0.946/0.917`).
- UTC slots used at run start: `0/5`; ~5.7h to reset.
- Active jobs before run: no local/trainer BirdCLEF jobs.
- Slot policy decision: mid UTC day. No submission unless a verifier-grade/high-info nonduplicate candidate clears gates.

## Result of this run

1. **New model data point:** trained `soundscape-native-b0-soft1279init-obspos-losite-allcls-ep4-20260528`.
   - LOSO row `0.569148`, file-MIL `0.474353`, no-train `0.506363`, non-Aves `0.532133`.
   - Versus head-loaded soft1279 baseline: row `-0.031212`, file-MIL `-0.131452`. Decision: reject unchanged.
2. **Package/audit:** observed-positive package best `soft1279init_obspos_native_allcls_w0p16` local AUC `0.993906`, lift vs v616 `+0.000425`; weaker than prior head-loaded `w0.16` by `-0.001639`. Decision: reject slot candidate.
3. **Per-class selector diagnostic:** head-loaded raw sidecar site-CV AUC `0.993761` (lift `+0.000280`), file-CV AUC `0.995051` (lift `+0.001571`), all-row lift `+0.002468`. Site robustness is weak: q05 `-0.003768`, p>0 `0.167`.

## Ranked next actions

1. **Late-day slot decision on head-loaded soft1279-init `w0.16` only if no stronger candidate appears** — still the best current local sidecar (AUC `0.995545`, lift vs v616 `+0.002064`) but not verifier-grade; per-class site-CV weakens the case.
2. **Trusted no-call/background protocol** — build/validate any-call/no-call/background labels from soundscape file/site evidence and evaluate as a gate, not another broad train-audio negative aux.
3. **Site-specific failure diagnosis for selected sidecar classes** — investigate sites/classes causing negative site-CV (`116570`, `chacha1`, `22973`, `555146`, `47144`, `trsowl`, `47158son17`, `47158son10`).
4. **PANNs/localmax hidden-test-operable wrapper** — lower priority; prior wrappers below v616, but may be late-day high-info if source-clean and nonduplicate.
5. **Public/source slot fill (<3h only)** — if no verifier-grade candidate emerges, use remaining slots only with source-clean candidates that pass schema/runtime/dedup guards.

## Critic / verifier decision

- Critic: observed-positive weighting is a clear regression; do not spend more runs on this weighting family. The head-loaded sidecar lift is real in all-row/local grids but not stable enough across sites.
- Verifier: artifacts are finite/nonconstant and row/column aligned; no submission this run because all new candidates fail promotion/robustness gates and reset is still >3h away.
