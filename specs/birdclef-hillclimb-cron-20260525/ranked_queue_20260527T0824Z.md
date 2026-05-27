# Ranked queue refresh — 2026-05-27 08:24 UTC

## Live state
- Best public LB remains **0.949**. Latest scored submissions: v621/v622/v623 tied 0.949, v625 0.948, v624 0.943; v616 remains a tied baseline to beat.
- 2026-05-27 UTC slots used: **0/5** at start; ~15.7h to reset. Early-day policy applies.
- No active local/trainer BirdCLEF jobs before this run; trainer GPU was used for one bounded sequence ablation and is idle after completion.

## Work completed this run
1. **All-class DyMN10 sequence sidecar audit**
   - Wrapped the 06:18 234-class leave-site context predictions into the v616 240-row proxy matrix.
   - Matched 156 proxy rows; anchor-filled 84 rows; output finite and 234/234 nonconstant.
   - Best tested blend was `allcls_seq_w0p0025`: local AUC `0.991108` / 42 valid classes, lift vs anchor `+0.000718`, but lift vs v616 `-0.002372`.
   - Critic/verifier decision: **reject as slot candidate**; useful comparison-grade audit only.
2. **All-class robust r3 context ablation**
   - Trained `soundscape-sequence-dymn10-allcls-r3-robust-losite-ep24-20260527` on official train_soundscapes: 1,478 windows / 66 files / 9 sites / 234 labels.
   - Leave-site result: row context AUC `0.501812` vs row-only `0.493697` (`+0.008115`); file-MIL `0.532188` vs `0.523772` (`+0.008416`).
   - Versus the 06:18 all-class r2 context baseline, the robust r3 ablation is `-0.095821` row and `-0.103097` file-MIL. S03 and S08 regressed.
   - Critic/verifier decision: **reject unchanged**; no package/no slot.

## Queue after this run
1. **True hidden-safe 234-class DyMN10 package only if formulation changes** — the all-class head remains the best row/file-MIL sequence clue, but direct v616 sidecar failed. A future package needs a different integration path: class-gated/capped use, only hidden rows/sites not anchor-covered, or an actual hidden feature extraction package with verifier before slot.
2. **S03/S08-aware sequence target redesign** — robust r3 failed; next sequence model should change the objective rather than only regularization (e.g., site-adversarial weighting, calibrated row-to-file pooling, or per-site caps learned from leave-site errors).
3. **Reformulated AudioSet/DyMN10 234-class acoustic-context wrapper** — use AudioSet embeddings as broad/no-call/context features, not a direct 72-label replacement; require multi-site validation.
4. **Deeper soundscape-native adapter/compact SED variant** — current B0 full fine-tune and robust sequence ablation are weak; try only a bounded adapter/last-block variant with leave-site/file gates.
5. **Late UTC slot-fill scout** — if <3h to reset and no verifier-grade package exists, use highest-ranked clean public/source candidates that are nonduplicate and pass schema/runtime/dedup guards.

## Submission decision
**No submission now.** Early-day slots remain available, but both evaluated candidates failed verifier/critic gates versus v616. Spending an early slot would be leaderboard probing rather than a verifier-grade action.
