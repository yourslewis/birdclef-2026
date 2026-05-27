# Ranked queue refresh — 2026-05-27 06:18 UTC

## Live state
- Best public LB remains **0.949**. Tied/highest latest: v616, v617, v620, v621, v622, v623 at 0.949; v625 0.948; v618 0.946; v619 0.944; v624 0.943.
- 2026-05-27 UTC slots used: **0/5** at start; ~17.7h to reset. Early-day policy applies.
- No active local/trainer BirdCLEF jobs before this run; trainer GPUs were idle.

## Queue after this run
1. **Package/audit all-class DyMN10 sequence head** — highest information value. New 234-class context head: leave-site row AUC `0.597633`, file-MIL `0.635285`, all folds positive vs row-only, TS smoke OK. Needs hidden EfficientAT extraction/inference wrapper and v616 sidecar audit before any slot.
2. **S03/S15/S23 guarded wrapper for no-train/non-Aves slices** — focused no-train branch helped most sites but failed S03; all-class branch fixes S03 at full scope. Build caps/guards rather than broad replacement.
3. **Group-balanced / worst-site all-class sequence ablation** — improve low folds S15/S23 without sacrificing S03/S19/S22; use this only if package audit shows lift potential.
4. **Deeper soundscape-native adapter/compact SED variant** — current B0 full fine-tune underperformed; try adapter/last-block or smaller regularized native model only after sequence package audit.
5. **Late UTC slot-fill scout** — if <3h to reset and no verifier-grade package exists, use highest-ranked clean public/source candidates that are nonduplicate and pass schema/runtime/dedup guards.

## Critic / verifier decision
- **No submission now.** The new model is a valid data point and package candidate, but not yet an end-to-end hidden-safe submission artifact.
- **Next exact action:** implement hidden-test feature extraction/inference for the 234-class DyMN10 context head, generate a raw sidecar CSV, then run v616 audit and dedup/schema checks.
