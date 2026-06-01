# BirdCLEF Hill-Climb Ranked Queue — 2026-06-01 12:30 UTC

## Live status
- UTC slots: **0/5 used** (reset ~11.7h away). Public best **0.950** (v644/v647). Latest v653/v654/v655 = 0.947/0.949/0.949.
- Trainer GPUs **FREE** this run (both idle) — used for the ConvNeXt-distill representation experiment.
- Foundation lane still exhausted externally: BirdNET fully closed (geo/scope/no-location all DEMOTE); SurfPerch/AudioMAE absent; Perch-v2 redundant. No new accessible non-Perch foundation kernel/dataset.

## This run's result — BREAKTHROUGH (still no-slot)
- **ConvNeXt-nano distilled from competent soft1279 native-B0 teacher (soft targets w=0.7, ep6).** Row AUC **0.7279** (vanilla ConvNeXt 0.5566), file-MIL **0.8034** (0.6023), pooled macro **0.7126** (0.3472).
- DEV gate: weak-class AUC **0.7997 (>chance; vanilla was 0.388)** AND rank_decorr **0.7679 (still high)**. **FIRST stream ever simultaneously orthogonal AND competent** — breaks the two-cluster mutual-exclusivity law that has held across PANNs/DyMN10/BirdNET/ConvNeXt-vanilla.
- BUT blend_best_weight **0.0**, site/file q05 **0.0**, DEV 0.0023 → **gate_pass=false**. The 42-valid-class / 240-row proxy is now the binding limiter (frontier E already saturates it); a competent-orthogonal member still earns 0 marginal weight there. Consistent with the framework's "two-way liar" caveat.

## Refreshed ranked queue (by DEV potential)
1. **Distillation strength sweep toward gate-pass (NEXT).** Sweep teacher_weight {0.5,0.85,1.0} x ep {6,10} on the ConvNeXt-distill front-end; measure whether more competence earns positive blend weight OR whether proxy sparsity caps it regardless. Cheap (≈2-4 min each on free GPU). Highest-info follow-through on the breakthrough.
2. **Per-class / Quantile-Mix blend of distilled stream vs E on E-weak columns only.** The global optimizer weight=0 may mask a per-column gain on the rare classes where the distilled stream is now competent (weak-AUC 0.80). If a per-column blend lifts E-weak without harming E-strong → genuine slot candidate.
3. **Guarded late-window LB probe of the distilled stream** (or its E-blend). A record orthogonal+competent stream that fails ONLY on the sparse proxy is exactly the guarded-exploratory pick the slot policy allows late-window. Needs a hidden-safe 234-class kernel package first.
4. **Output-pool TTA (alpha=0.25) on 0.950 pipeline** — READY/DEV-passing but partially duplicates the winner's existing circular `tta_shifts`; lower-info. Hold unless late slot pressure with nothing better.
5. **BirdNET embedding-feature head** (carry, BLOCKED: birdnetlib not in .venv_scout; trainer build).

## Decision this run
- **No slot spent.** Early UTC (~11.7h to reset). The distillation breakthrough is the strongest diversity lead to date; the correct move is to push it (sweep + per-class blend), not spend a slot before it clears the gate or is packaged for a guarded probe.
- STOP rule intact: this was a representation-level change (distinct backbone + distilled competence), NOT a shared-embedding head knob.
- Next concrete action: distillation strength sweep (#1) → per-class E-weak blend (#2); package for a guarded late-window LB probe if competence+decorr hold but the proxy keeps capping blend weight.

## Artifacts
- DEV gate: `artifacts/diversity_scout/convnext_distill_20260601/diversity_scout_summary.json`
- Ledger: `artifacts/model_data_point_ledger/20260601T1230Z_convnext_distill_soft1279teacher_dev_gate.md`
- Performance table rows appended (md + jsonl).
