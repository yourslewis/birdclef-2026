# BirdCLEF Hill-Climb Ranked Queue — 2026-06-01 14:40 UTC

## Live status
- UTC slots: **0/5 used** (reset ~9.3h away, ~23:59 UTC). Public best **0.950** (v644/v647).
  Latest scored: v653 0.947 / v654 0.949 / v655 0.949.
- Trainer GPUs **FREE** (both idle before+after this run; one RegNetY datapoint trained, 79s).
- Foundation lane exhausted externally: BirdNET fully closed (geo/scope/no-location all DEMOTE);
  SurfPerch/AudioMAE absent; Perch-v2 redundant. No new accessible non-Perch foundation kernel.

## This run's results (3 corrective/negative datapoints — NO slot spent)
1. **Queue#2 per-class E-weak blend — DECISIVE NEGATIVE.** The ConvNeXt-distill "breakthrough"
   (12:30Z) beats frontier E on **0 of 14** E-weak columns individually. Its aggregate weak-AUC
   0.80 was a macro-AVERAGING artifact, not per-column complementarity. Best targeted blend
   +0.000115 macro but **file_q05 -0.000173 (negative)** → fails. Proxy saturation is GENUINE,
   not a global-scalar artifact. *Corrects last run's over-optimistic "breakthrough" framing.*
2. **TTA alpha=0.25 re-audit — DOWNGRADE READY→redundant.** The 0.950 winner notebook already
   runs `tta_shifts=[0,1,-1,2,-2]` internally on proto. Our output-pool TTA proxy is
   **TTA-on-TTA**; the +0.000364 proxy lift is double-smoothing that won't transfer. Not a slot
   candidate (near-STOP-rule, same representation, lever already applied upstream).
3. **RegNetY-008 distill — GENERALIZATION CONFIRMED, still DEMOTE.** Third distinct backbone
   reproduces orthogonal (decorr 0.685) + competent (weak-AUC 0.792); row 0.7336 / file-MIL
   0.8153 / pooled 0.7500 (all ≥ ConvNeXt-distill). Blend weight ~0.02, **both q05 negative** →
   gate_pass false. The orthogonal+competent distill property is backbone-AGNOSTIC, but frontier
   E is per-column stronger on every measurable proxy class. Binding constraint = the
   **42-valid-class / 240-row proxy ceiling** (192/234 classes unmeasurable), not the model.

## Hardened conclusion
The diversity lane has now produced 3 backbone families (B0/ConvNeXt/RegNetY) of
orthogonal+competent distilled streams, and NONE earns positive blend q05 on the canonical proxy.
The per-class analysis proves frontier E dominates them column-by-column on all 42 measurable
classes. The only honest path to value these streams is a **live LB probe of a 234-class
hidden-safe package** — but with zero positive proxy q05, that does NOT justify an early-UTC slot.

## Refreshed ranked queue (by DEV potential + info value)
1. **Package a distilled student (RegNetY-008 or ConvNeXt) as a 234-class hidden-safe kernel for
   a GUARDED LATE-WINDOW LB probe.** This is the ONLY remaining way to test whether proxy-saturated
   orthogonal+competent streams carry value on the 192 unmeasurable classes. Needs: 234-class
   inference wrapper + v616-audit + COMPLETE Kaggle dry-run. Late-window guarded-exploratory pick
   if no DEV-passer appears and a slot would otherwise expire. *Build cost = real (kernel wiring).*
2. **Stronger / multi-teacher distillation** (e.g. distill from frontier-E SED posterior, or an
   ensemble teacher, into RegNetY) — try to push a distilled student to actually beat E on ≥1 weak
   column. Cheap on free GPU; only promote if it flips a per-column win.
3. **Train a 4th-backbone datapoint only if it tests a NEW hypothesis** (e.g. a non-CNN/attention
   front-end like a small AST/MaxViT) — diminishing returns; the 3-backbone generalization is now
   well-established, so this is low-info unless paired with a teacher change.
4. ~~Output-pool TTA~~ — DOWNGRADED to redundant (TTA-on-TTA); do not slot.
5. ~~BirdNET~~ — closed (all three DEMOTE).

## Decision this run
- **No slot spent.** Early UTC (~9.3h to reset). No DEV-passing candidate exists; all 3 datapoints
  DEMOTE. Per slot policy, advanced the diversity plan with a generalization datapoint instead of
  burning an early slot on a redundant/STOP-rule lever. STOP rule intact (all representation-level).
- Next concrete action: if a slot would otherwise expire late-window, build the #1 hidden-safe
  234-class student package for a guarded LB probe; otherwise try #2 (stronger teacher) for a
  per-column win first.

## Artifacts
- `artifacts/diversity_scout/convnext_distill_perclass_20260601/perclass_blend_summary.json`
- `artifacts/diversity_scout/regnety008_distill_20260601/scout/diversity_scout_summary.json`
- ledgers: `artifacts/model_data_point_ledger/20260601T14{30,35,40}Z_*.md`
- perf table rows appended (md + jsonl)
