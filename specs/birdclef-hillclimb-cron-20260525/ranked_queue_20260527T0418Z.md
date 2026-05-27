# Ranked Queue — 2026-05-27 04:18 UTC

## Live status
- Best public LB remains **0.949**. Newly scored late-fill submissions: v621 `0.949`, v622 `0.949`, v623 `0.949`, v624 `0.943`, v625 `0.948`.
- 2026-05-27 UTC slots used: **0/5** at live check; ~19.7h to reset.
- Active jobs: no BirdCLEF local/trainer jobs; trainer has unrelated LRM process but GPUs were effectively idle for this bounded run.

## Slot decision
Early UTC day. No submission made: the newly scored public candidates did not beat v616, and the model trained here is not a hidden-safe 234-class package. Preserve early slots for verifier-grade/high-information candidates, then revisit late-fill if needed near reset.

## Ranked next queue
1. **True hidden-safe 234-class DyMN10/AudioSet package** — build from train_soundscapes/AudioSet features without leave-site OOF wrapper leakage; include no-train caps and S03 guard. Highest information value after focused no-train branch improved file-MIL but failed one site.
2. **S03-guarded no-train sidecar audit** — very low capped 28-class replacement on proxy rows; require no S03/S22 deterioration and compare vs v616, not anchor.
3. **Broader OOF negative/no-call SED student** — still valuable for no-call/background behavior; only after negative set coverage audit.
4. **G124/V2S hard-confidence/power ablation** — technically working lane; only if sidecar lift exceeds microscopic prior.
5. **Late-day public source fills** — only inside <3h to reset, after source/schema/dedup checks; v624/Haru and v625/Safar are now rejected/below best; v621-v623 tied only.

## Critic / verifier decision
- Critic: focused no-train model is worth keeping as a data point, but the S03 fold collapse means no packaging without guardrails.
- Verifier: finite/nonconstant OOF predictions and TorchScript smoke passed; no competition submission approved.
