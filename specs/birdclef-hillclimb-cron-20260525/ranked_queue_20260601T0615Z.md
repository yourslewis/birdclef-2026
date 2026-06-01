# BirdCLEF hill-climb ranked queue — 2026-06-01 06:15 UTC

## Live status
- Bearer API submission check at 06:15 UTC: latest scored `v654=0.949`, `v655=0.949`, `v653=0.947`, `v652=0.948`, `v651=0.941`; public best remains **0.950** from `v644`/`v647`.
- UTC daily slots: **0/5 used** for 2026-06-01; ~17.75h to reset. Early UTC day.
- Active work: no BirdCLEF trainer jobs (GPUs on unrelated HSTU/LRM). Local CPU/`.venv_birdnet` used this run.

## Role synthesis
- **Coordinator:** Early UTC day, no DEV-passing slot candidate packaged for a live LB shot. Per pivot, advance the diversity plan (representation-level / open-question resolution), do NOT spend a slot on a STOP-RULE head variant.
- **Public Solution & Model Scout:** EoS8 scalar frontier exhausted (v651-v655 all ≤0.949). BirdNET was the last structurally-different lever; this run closed its remaining open question.
- **Data & Feature Scientist:** Tested whether BirdNET's Pantanal geo-prior caused the orthogonality↔competence tradeoff — a representation/front-end conditioning hypothesis (in scope).
- **Validation & Metrics Scientist:** Location-agnostic re-score changed only 6/56160 cells; DEV identical (0.0010543 vs 0.0010548). Hypothesis REFUTED.
- **Prediction & Ensemble Analyst:** blend_best_lift +0.000421 but file_q05 -0.000482; gate fails both (a) and (b). No slot.
- **Critic / Red Team:** Geo-prior was a plausible confound; now eliminated. BirdNET incompetence on E-weak classes is a genuine representational gap, not maskable.
- **Verifier / Skeptic:** Finite/nonconstant 240x234 proxy; gate_pass=false; no submission queued — correct.

## Current run result
- Added `--no-location` to `scripts/birdnet_proxy_infer.py`; copied 20 proxy `.ogg` files from trainer; re-scored BirdNET location-agnostic; DEV-gated vs 0.950 frontier E.
- **Verdict:** geo-prior REFUTED as cause of the tradeoff; closes the BirdNET domain-shift open question. DEMOTE, no slot.

## Refreshed ranked queue (by DEV potential)

1. **TTA real-kernel LB shot (READY candidate, needs packaging).**
   - Status: only DEV-gate-passing candidate we own (site_q05 +0.000193, file_q05 +0.000134, gate_pass=true), but it is a proxy construction. A live LB test requires re-running the 0.950 winner source kernel with overlapping-5s + time-shift TTA windows (alpha≈0.25). Highest-value slot use once packaged.
   - Gate: 3x235 schema OK; finite; non-degrading vs 0.950.

2. **New self-supervised audio foundation hunt (different backbone than EoS8 Perch-ProtoSSM).**
   - Rationale: BirdNET lane now exhausted (both geo and scope tradeoffs closed). Need a genuinely new accessible public birdclef-2026 backbone (SurfPerch/AudioMAE/Google bird-vocalization variants with REAL kernels). No fabricated slugs.
   - Gate: DEV gate vs E.

3. **Representation-changing soundscape-native variant (fine-tune blocks/adapter, not head).**
   - Rationale: changes the embedding (in scope). File-cal teacher signal (row 0.712 / file-MIL 0.783) is the best 72-label data point; distill into a deeper student that consumes file-level teacher targets.
   - Gate: beat row 0.712 or improve sidecar lift with nonnegative q05.

4. **BirdNET embedding-feature scoring (not detection probabilities).**
   - Rationale: the only remaining BirdNET angle — use BirdNET's penultimate embedding as features for a learned head on E-weak classes, rather than thresholded detections. Could decouple orthogonality from detection competence.
   - Gate: DEV gate vs E.

5. **Late UTC slot fill.**
   - Rationale: near reset, fill remaining slots with highest DEV-ranked valid candidate (TTA real kernel preferred). Never a STOP-RULE head variant.

## Artifacts
- Ledger: `artifacts/model_data_point_ledger/20260601T0615Z_birdnet_nolocation_dev_gate.md`
- DEV gate: `artifacts/diversity_scout/birdnet_nolocation_20260601T0615Z/diversity_scout_summary.json`
- Proxy: `artifacts/diversity_scout/birdnet_proxy/nolocation_20260601T0615Z/birdnet_nolocation_proxy.csv`
- Script change: `scripts/birdnet_proxy_infer.py` (`--no-location`)
