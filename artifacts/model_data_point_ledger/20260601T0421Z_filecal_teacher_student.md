# File-calibration teacher student — 2026-06-01 04:21 UTC

## Purpose
Convert the strongest 02:20 UTC file-level calibration diagnostic into a more hidden-safe, rerunnable student model instead of using leave-site OOF matrices as a submission artifact.  The teacher is the best local mapping signal: PANNs/Cnn14 72-label row-only predictions plus DyMN10 file-mean evidence at alpha `0.35`.

## Configuration
- **Experiment id:** `soundscape-filecal-teacher-student-fused-r2-losite-20260601`
- **Branch family:** train_soundscapes hidden-safe file-calibration teacher student.
- **Data:** official `train_soundscapes`; `1,478` windows, `66` files, `9` sites; teacher-covered OOF rows `1,314`.
- **Target scope:** `72` non-Aves/no-train labels.
- **Model/init:** fused DyMN10+PANNs embedding context MLP, feature dim `21060`, hidden `384`, dropout `0.35`; trained with hard BCE + soft teacher BCE (`0.55` / `0.45`).
- **Validation split:** leave-one-site, 6 valid folds; sidecar audit is 72→234 anchor-filled local v616 proxy + `200` bootstrap iterations.
- **Runtime/export:** local CPU train/audit completed; TorchScript and state dict exported; OOF sidecar finite/nonconstant (`156` matched proxy rows, `234` classes); no Kaggle submission.

## Comparable performance

| Metric | File-cal student | Baseline / comparator | Delta |
|---|---:|---:|---:|
| Row macro AUC mean | `0.711995` | file-cal teacher mapping `0.684862` | `+0.027133` |
| File-MIL AUC mean | `0.783017` | file-cal teacher mapping `0.744065` | `+0.038952` |
| No-train AUC | `0.638918` | 02:20 mapping `0.599442` | `+0.039476` |
| Non-Aves AUC | `0.711995` | 02:20 mapping `0.687350` | `+0.024644` |
| 72→234 sidecar local AUC | `0.990896` (`filecal_student_w02`) | v616 local proxy `0.993481` | `-0.002585` |

## Top comparable 72-label sequence/file/site row-AUC points

| Rank | Experiment | Row AUC | File-MIL AUC | Notes |
|---:|---|---:|---:|---|
| 1 | File-cal teacher student fused-r2 | `0.711995` | `0.783017` | New best local row/file validation point; hidden-safe model form, but sidecar still below v616. |
| 2 | File-cal mapping: PANNs row + DyMN10 file mean 35% | `0.687350` | `0.757145` | Prior best diagnostic; no hidden package. |
| 3 | PANNs/Cnn14 72-label row-only | `0.674485` | `0.691156` | Previous best row-wise targeted model. |
| 4 | Fused DyMN10+PANNs 72-label filectx/fileMIL | `0.652377` | `0.722866` | File-context fusion landscape point. |
| 5 | DyMN10 72-label filectx/fileMIL | `0.641802` | `0.745704` | Previous best file-MIL clue. |

## 72→234 sidecar audit vs v616
- Best audited recipe: `filecal_student_w02`.
- Local macro AUC: `0.990896` / `42` valid local classes.
- Lift vs anchor: `+0.000505`.
- Lift vs v616: `-0.002585`.
- Rank corr vs v616: `0.999610`; MAE `0.006471`.
- `submit_approved=false`: below v616 proxy and still an OOF sidecar audit, not a submitted hidden package.

## Critic / verifier decision
- **Critic:** The student is a strong landscape point: row AUC rose above the teacher diagnostic and all prior 72-label sequence variants, and file-MIL also improved. But the v616 proxy sidecar remained negative vs v616, so the signal still does not justify an early UTC slot.
- **Verifier:** Export/smoke OK, sidecar finite/nonconstant/schema-valid, no duplicate/static/sample output. Reject for submission because local proxy lift vs v616 is `-0.002585` and `submit_approved=false`.
- **Decision:** keep and use as a teacher/package component; no Kaggle slot. Next action: class/site movement selector or private-verifier-safe package that only activates classes/sites with robust positive movement.

## Artifacts
- Script: `scripts/birdclef_filecal_teacher_student.py`
- Metrics: `artifacts/filecal_teacher_student/soundscape-filecal-teacher-student-fused-r2-losite-20260601/metrics.json`
- Export: `artifacts/filecal_teacher_student/soundscape-filecal-teacher-student-fused-r2-losite-20260601/filecal_teacher_student_context_head.ts.pt`
- OOF predictions: `artifacts/filecal_teacher_student/soundscape-filecal-teacher-student-fused-r2-losite-20260601/leave_site_predictions.npz`
- Sidecar audit: `artifacts/filecal_teacher_student/soundscape-filecal-teacher-student-fused-r2-losite-20260601/sidecar_audit/audit_summary.json`
