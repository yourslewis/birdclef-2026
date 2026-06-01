# BirdCLEF hill-climb ranked queue — 2026-06-01 04:25 UTC

## Live status
- Bearer API submission check at 04:15 UTC: latest scored submissions are `v653=0.947`, `v654=0.949`, `v655=0.949`, `v651=0.941`, `v652=0.948`; public best remains `0.950` from `v644`/`v647`.
- UTC daily slots: `0/5` used for 2026-06-01; ~19.7h to reset at status check.
- Active work check: no BirdCLEF local/trainer jobs found. Trainer GPUs are occupied by unrelated HSTU/LRM jobs, so this run used local CPU for the compact student/audit.

## Role synthesis
- **Coordinator:** Early UTC day; do not spend slots on below-v616/v950 sidecars. Train a distinct high-information student data point.
- **Public Solution & Model Scout:** v651-v655 EoS8 scalar frontier failed (`0.941`-`0.949`), so public-source perturbations are demoted until a materially distinct source appears.
- **Data & Feature Scientist:** The under-mined train_soundscapes sequence/file/site lane remains useful; file-calibration teacher signal transfers into a hidden-safe MLP student.
- **Validation & Metrics Scientist:** Student beats prior row/file train_soundscape diagnostics on leave-site metrics, but sidecar proxy remains below v616.
- **Prediction & Ensemble Analyst:** 72→234 rank-blend sidecar best is `0.990896`, lift vs v616 `-0.002585`; no slot.
- **Experiment Engineer:** Added `scripts/birdclef_filecal_teacher_student.py`, exported TorchScript/state, emitted OOF predictions and audit.
- **Critic / Red Team:** Strong row/file AUC can still overfit local soundscape labels; v616 proxy degradation blocks submission.
- **Verifier / Skeptic:** Finite/nonconstant/schema-valid OOF sidecar; `submit_approved=false`; no malformed/static/duplicate output.

## Current run result
- Trained `soundscape-filecal-teacher-student-fused-r2-losite-20260601` on official train_soundscapes (`1,478` windows / `66` files / `9` sites / `72` labels) using fused DyMN10+PANNs context features and a PANNs-row + DyMN10-file teacher.
- Metrics: row AUC `0.711995`, file-MIL `0.783017`, no-train `0.638918`, non-Aves `0.711995`.
- Delta vs 02:20 file-cal teacher mapping: row `+0.027133`, file-MIL `+0.038952`.
- Sidecar verifier: best `filecal_student_w02` local AUC `0.990896` / `42` valid, lift vs v616 `-0.002585`, rank corr `0.999610`; no submit.

## Refreshed ranked queue

1. **Class/site movement selector for file-cal student — selected next.**
   - Rationale: student is now best 72-label row/file data point, but broad 72→234 sidecar is below v616. Need identify classes/files/sites where movement is robustly positive and cap the rest.
   - Gate: positive site/file CV lift vs v616 and nonnegative q05; finite 234-class candidate; no static/fallback output.

2. **Hidden-test package path for file-cal student inference.**
   - Rationale: TorchScript head exists; next packaging must reproduce fused embeddings/context on hidden soundscapes without OOF proxy leakage.
   - Gate: private verifier schema/runtime OK and v616 proxy does not degrade.

3. **Deeper native/student with file-cal teacher targets.**
   - Rationale: the MLP student transferred teacher signal; try compact CNN/SED or adapter only if compute is free and it can consume file-level teacher targets.
   - Gate: beat row `0.711995` or improve sidecar lift.

4. **Public source scout only for materially distinct source-clean candidates.**
   - Rationale: EoS8 scalar frontier neighbors and BirdNET/TTA lanes are informative but not stronger than v644/v647 public `0.950` today.

5. **Late UTC slot fills.**
   - Rationale: Use remaining slots near reset only after verifier/schema/dedup checks, and never submit known below-best/malformed candidates.

## Artifacts
- Ledger: `artifacts/model_data_point_ledger/20260601T0421Z_filecal_teacher_student.md`
- Metrics: `artifacts/filecal_teacher_student/soundscape-filecal-teacher-student-fused-r2-losite-20260601/metrics.json`
- Sidecar audit: `artifacts/filecal_teacher_student/soundscape-filecal-teacher-student-fused-r2-losite-20260601/sidecar_audit/audit_summary.json`
- Script/export: `scripts/birdclef_filecal_teacher_student.py`, `artifacts/filecal_teacher_student/soundscape-filecal-teacher-student-fused-r2-losite-20260601/filecal_teacher_student_context_head.ts.pt`
