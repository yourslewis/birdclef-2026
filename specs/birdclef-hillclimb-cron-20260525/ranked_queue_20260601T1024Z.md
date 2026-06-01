# BirdCLEF Hill-Climb Ranked Queue — 2026-06-01 10:24 UTC

## Live status
- UTC slots: **0/5 used** (reset ~13.5h away). Public best **0.950** (v644/v647). Latest v653/v654/v655 = 0.947/0.949/0.949.
- Trainer GPUs **FREE** this run (both idle) — used for the ConvNeXt-nano representation data point.
- Foundation lane still exhausted: BirdNET fully closed (geo + scope + no-location all DEMOTE); SurfPerch/AudioMAE absent; Perch-v2 redundant. No new accessible non-Perch foundation kernel/dataset.

## This run's result
- **ConvNeXt-nano soundscape-native (allcls, ImageNet-pretrained)** — first structurally-distinct front-end backbone trained directly on train_soundscapes (STOP-rule-safe representation change, not a head knob). Row AUC 0.5566, file-MIL 0.6023. DEV: rank_decorr **0.9759 (record for a native stream)** but weak-class AUC **0.3877 (< chance)**, competence_above_chance 0.0, all q05 0.0, DEV 0.0 → **gate_pass=false, DEMOTE**.
- LESSON (new, important): the orthogonality↔competence two-cluster law is **not** a quirk of PANNs/DyMN10/BirdNET embeddings — even a fresh, structurally distinct CNN front-end reproduces it. Backbone family is NOT the binding constraint; **rare-class competence on train_soundscapes is**. Representation novelty alone does not pass DEV.

## Refreshed ranked queue (by DEV potential)
1. **Genuine audio-overlap Perch TTA kernel (NEW build).** Re-extract Perch embeddings on raw windows centered off the canonical grid (±2.5s overlap) and average per segment, then feed the existing PowerOpt engine. Distinct from the winner's circular `tta_shifts` → true front-end lever on the COMPETENT lineage (the missing ingredient ConvNeXt lacked). Risk: ~2x Perch wall-time on hidden test → must verify wall-time safety. Top slot candidate once packaged + COMPLETE-verified.
2. **Competence-first representation variant.** Repeat the ConvNeXt/distinct-backbone idea BUT initialize from a competent OOF-teacher (soft1279) and/or longer schedule + soundscape-positive scope, so the distinct front-end inherits rare-class competence before being judged for diversity. The only path that could move a distinct backbone from cluster A (orthogonal-incompetent) toward truth-aligned diversity. No slot until DEV-positive.
3. **BirdNET embedding-feature head** (carry). Penultimate embeddings (not thresholded detections) as features for a learned head on E-weak classes. BLOCKED: birdnetlib not in .venv_scout; trainer-side build. No slot.
4. **Output-pool TTA (alpha=0.25) on 0.950 pipeline** — READY/DEV-passing but partially duplicates the winner's existing circular `tta_shifts`; lower-info than #1. Hold unless late-window slot pressure with nothing better.
5. **Late UTC slot fill** (near reset): highest DEV-ranked valid candidate; audio-overlap TTA preferred. Never a STOP-RULE shared-embedding head variant.

## Decision this run
- **No slot spent.** Early UTC (~13.5h to reset). ConvNeXt-nano was a high-info representation-level data point (record decorrelation) but DEMOTE on competence — exactly the kind of landscape point the policy wants while slots are conserved for a DEV-passing candidate.
- STOP rule intact: no shared-embedding head variants. This run advanced the diversity plan with a genuine representation change.
- Next concrete action: build the audio-overlap Perch TTA kernel (verifier-grade, wall-time-checked) OR a competence-seeded distinct-backbone variant (#2) for a future slot.

## Artifacts
- DEV gate: `artifacts/diversity_scout/convnextnano_20260601/diversity_scout_summary.json`
- Ledger: `artifacts/model_data_point_ledger/20260601T1024Z_convnextnano_soundscape_native_dev_gate.md`
- Performance table rows appended (md + jsonl).
