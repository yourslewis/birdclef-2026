# BirdCLEF hill-climb ranked queue — 2026-05-31 12:20 UTC

## Live state
- Kaggle Bearer API live check at start: current public LB best is `0.950`, tied by v644 Yaroslav 0950 replay and v647 Ryuto EoS8 sidecar source from 2026-05-30 UTC.
- 2026-05-31 UTC slot use at run start: `0/5`; ~11.7h to reset.
- Active jobs: no local/trainer BirdCLEF jobs at start; trainer GPU0 occupied by unrelated HSTU, GPU1 used for this run.

## This-run result
Trained/evaluated `soundscape-sequence-fused-dymn10-panns-nonaves-notrain-rowonly-losite-ep24-20260531`: fused DyMN10+PANNs row-only 72-label non-Aves/no-train data point.

- Data: `1,478` windows / `66` files / `9` sites / `72` labels.
- Leave-site metrics: row AUC `0.616166`; file-MIL `0.723917`; no-train `0.491181`; non-Aves `0.616166`; `6` valid site folds.
- Delta vs PANNs 72-label row-only: row `-0.058319`, file-MIL `+0.032761`.
- Delta vs PANNs 72-label filectx+fileMIL: row `-0.015425`, file-MIL `+0.033912`.
- Sidecar audit: best non-anchor recipe `seq_context_w02` local AUC `0.990059` / 42 valid; lift vs v616 `-0.003422`, lift vs anchor `-0.000332`; `submit_approved=false`.

## Ranked queue

1. **EoS8/PowerOptimization/taxonomy source exploitation — ACCEPTED / top priority**
   - v644/v647 are the current `0.950` public-best family and effectively one PowerOptimization/taxonomy-smoothed anchor.
   - Next: build a repo-owned verifier/parameter audit for the PowerOptimization path or use source-winner confidence signals; avoid exact/near-duplicate reruns early day.

2. **Source-winner-informed no-call/confidence sidecar — NEEDS REVISION**
   - Threshold-only farneg no-call variants are saturated. The next no-call work needs hand/teacher-audited multi-site negatives or winner-anchor confidence features.

3. **Train_soundscapes targeted sequence/file/site mining — CONTINUE ONLY WITH NEW SIGNAL**
   - Fused 72-label row-only improved file-MIL but hurt row AUC and failed sidecar transfer; PANNs-only row-level remains the better local row signal.
   - Continue only with a genuinely new transfer mechanism (site-risk-constrained class selector, v950 anchoring, or calibrated file-presence use), not more blind PANNs/fusion wrappers.

4. **Class/site-constrained selector over best soundscape sidecars — COMPARISON-GRADE**
   - Tiny local selector lifts exist but lack robust held-group support; no submission until positive leave-site/file movement is demonstrated.

5. **No-train-only PANNs capacity tweaks — REJECTED for now**
   - 28-label h384 no-train isolation underperformed the broader 72-label multitask scope; auxiliary non-Aves labels help no-train generalization.

## Critic / verifier decision
- Critic: fused 72-label row-only was a reasonable encoder-fusion isolation point, but row underperformance plus sidecar regression closes this micro-lane unless file-MIL can be explicitly and safely transferred.
- Verifier: training artifacts and sidecar CSVs are finite/nonconstant/aligned; sidecar fails promotion. No early-day submission.

## Slot decision
No submission: `0/5` slots remain because no newly trained candidate is verifier-grade and it is still early/mid UTC day. Fill remaining slots later only with highest-ranked valid exploratory/source-clean candidates if no stronger verifier candidate emerges.
