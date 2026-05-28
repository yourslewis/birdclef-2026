# Ranked Queue — BirdCLEF ClawTeam Hill-Climb — 2026-05-28 16:25 UTC

## Live state

- Public best: `0.949` (v616/v621-v623 family still tied best; v626-v630 below best at `0.899/0.928/0.940/0.946/0.917`).
- UTC slots used at run start: `0/5`; ~7.7h to reset.
- Active jobs before run: no local BirdCLEF jobs; trainer GPU1 free, GPU0 had unrelated activity.
- Slot policy decision: mid UTC day. No submission unless a verifier-grade/high-info nonduplicate candidate clears gates.

## Result of this run

1. **Stability/calibration audit:** Expanded the head-loaded soft1279-init native all-class sidecar grid from `w0.08` to `w0.16`.
   - Best local recipe: `soft1279init_native_allcls_w0p16` AUC `0.995545` / 42 valid, lift vs v616 `+0.002064`, lift vs anchor `+0.005155`.
   - Gate: no submit. `w0.16` still failed strict lift-vs-anchor (`0.005155` < `0.006`) despite passing the other manifest gates.
2. **New model data point:** Trained `soundscape-native-b0-soft1279enc-losite-allcls-ep4-20260528` (soft1279 encoder-only, head reinit).
   - LOSO row `0.506642`, file-MIL `0.460169`, no-train `0.552063`, non-Aves `0.467678`.
   - It regressed strongly vs head-loaded soft1279 native all-class (`row -0.093718`, file `-0.145636`), so reject unchanged.
3. **Package audit for encoder-only:** best sidecar `soft1279enc_native_allcls_w0p08` AUC `0.993144`, lift vs v616 `-0.000337`; rejected.

## Ranked next actions

1. **Late-day consideration of head-loaded soft1279-init `w0.16` only if no stronger candidate appears** — Highest local AUC/lift so far, but still comparison-grade and below strict anchor-lift threshold. Re-evaluate near <3h reset with slot policy and duplicate guards.
2. **Per-site/per-class failure diagnosis for `w0.16`** — Identify whether the lift is concentrated in a few scarce classes (`strher2`, `555146`, etc.) or robust enough for hidden behavior. This is the needed verifier upgrade before early-day use.
3. **Trusted no-call/background label protocol** — Build explicit any-call/no-call/background labels from soundscape/site/file evidence; avoid more random train-audio negative aux until this is clear.
4. **PANNs/localmax hidden-test-operable wrapper** — Lower priority; prior direct localmax sidecar still below v616, but may be a decorrelated late-day exploratory candidate if packaged safely.
5. **Public/source slot fill (<3h only)** — If no verifier-grade candidate emerges, compare `w0.16` against source-clean public candidates and use remaining slots only after verifier accepts residual risk.

## Critic / verifier decision

- Critic: the monotonic local gain with increasing sidecar weight is encouraging but could be proxy/calibration overfit; encoder-only ablation failing suggests the head-loaded train-audio teacher head is part of the signal, not generic transfer.
- Verifier: all evaluated artifacts are finite/nonconstant and row/column aligned; no submission this run because `submit_approved=false` and mid-day policy requires stronger gates.
