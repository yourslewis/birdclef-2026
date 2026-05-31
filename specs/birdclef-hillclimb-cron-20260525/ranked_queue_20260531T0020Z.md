# Ranked queue — 2026-05-31 00:20 UTC

## Live status

- Public LB best is now `0.950`: v644 (`53197162`, Yaroslav 0950 replay source) and v647 (`53197164`, Ryuto EoS8 sidecar source) scored `0.950`, +0.001 over the old v616/tied 0.949 plateau.
- Latest submitted scores: v641 `0.947`, v642 `0.948`, v643 `0.946`, v644 `0.950`, v647 `0.950`.
- UTC daily slots after reset: `0/5` used with ~23.7h to reset at live check.
- Active jobs: no local/trainer BirdCLEF jobs after the sequence training/audit completed.

## Actions this run

1. Monitored and resolved v641-v644/v647; updated canonical performance table rows from pending to scored.
2. Trained a new data-point model: `soundscape-sequence-fused-dymn10-panns-allcls-r4-20s-localmeanmax-losite-ep20-20260531`.
3. Ran all-class sidecar audit vs the v616 local proxy: `artifacts/soundscape_sequence_sidecar_audit/20260531T0020Z_fused_allclass_r4_20s_localmeanmax/`.
4. Updated ledger artifacts, canonical table MD/JSONL, and autoresearch log.

## Compact performance table for this run

| Exp | Family | Metric | Delta | Status |
|---|---|---:|---:|---|
| v641 | Nina EoS1 public source | 0.947 public LB | -0.002 vs v616 | reject |
| v642 | Nina EoS4 public source | 0.948 public LB | -0.001 vs v616 | reject |
| v643 | Raunak v7 public source | 0.946 public LB | -0.003 vs v616 | reject |
| v644 | Yaroslav 0950 replay | 0.950 public LB | +0.001 vs v616 | new best tie |
| v647 | Ryuto EoS8 sidecar | 0.950 public LB | +0.001 vs v616 | new best tie |
| fused-r4-20s | sequence/file/site fusion | 0.581429 row AUC | -0.046129 vs PANNs r4 | reject |
| fused-r4-20s sidecar | all-class wrapper audit | 0.991115 local AUC | -0.002366 vs v616 | reject |

## Ranked next actions

1. **Exploit/understand new 0.950 public-source winners** — audit v644/v647 source lineage, package/runtime, output diversity, and whether either can be a stable anchor/sidecar without relying on static public output.
2. **Resume train-soundscape model data-point lane with a new hypothesis** — hand/teacher-audited multi-site no-call negatives or explicit site-risk-constrained soft1279 recipe; avoid blind PANNs/B0 knob cycling.
3. **Public source scout** — search for hidden-safe variants adjacent to v644/v647 that are not duplicates/malformed; early UTC day requires verifier-grade or high-info candidates before spending slots.
4. **Do not submit the fused 20s sequence sidecar** — it is finite/aligned but below v616 in proxy and below the new 0.950 public best.

## Critic / verifier decision

- v644/v647: submission already executed under late-day policy; now accepted as new public best ties, monitor private only.
- Fused 20s model: **rejected as slot candidate**. It improves its own row-only baseline but fails opportunity cost and v616 sidecar gates.
- Slot decision for current UTC day: **no submission now**; early day with 0/5 used, no verifier-grade candidate ready.
