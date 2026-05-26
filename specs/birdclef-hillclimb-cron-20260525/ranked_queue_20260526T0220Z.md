# BirdCLEF Hill-Climb Queue — 2026-05-26 02:20 UTC

## Live status

- Public LB best remains **0.949**; `v616` remains the tied baseline to beat.
- Latest completed submissions: `v617=0.949`, `v618=0.946`, `v619=0.944`, `v620=0.949`, `v616=0.949`.
- 2026-05-26 UTC slots used: **0/5**.
- No active BirdCLEF jobs were found locally or on trainer before this run.
- Slot decision: **no submission this early UTC run**. There is no verifier-grade hidden-safe candidate yet; duplicate replays of v616/v617/v620 are invalid by policy.

## Actions this run

- Built a broad OOF-teacher-derived negative/no-call mask from `artifacts/pseudolabels/oof-teacher-cache/b0v26_nfnetv29_w090010_intersection_cache.npz` using threshold `0.03`, cap `64` negatives per row.
- Trained the next ranked repo-owned model data point: B0 soft OOF-teacher student with broad negative auxiliary loss, 1,024 rows, 4 epochs.
- Ran the matched 1,024-row / 4-epoch soft-only control to separate row/epoch lift from aux-mask lift.
- Verified TorchScript/ONNX export and CPU inference smoke for both new models.

## Ranked queue after this run

| Rank | Candidate | Expected LB potential | Info/data-point value | Current decision |
|---:|---|---:|---:|---|
| 1 | OOF-teacher soft B0 1024 ep4 control | Medium if converted into hidden-safe row-aligned sidecar and audit beats v616 | High | **DATA POINT TRAINED**; strongest SED proxy so far (`0.9111`), not submit-ready |
| 2 | Broader OOF negative/no-call aux B0 | Medium; tests broad no-call/background suppression | High | **DATA POINT TRAINED**; mask works but aux slightly hurts AUC vs soft-only (`0.9083` vs `0.9111`) |
| 3 | 20s temporal/localmax branch | Medium; tests temporal context not covered by v616 | Medium/high | Next distinct repo-owned smoke |
| 4 | PANNs/Cnn14 AudioSet sidecar wrapper / leave-site split | Medium; external AudioSet diversity but weak S08 proxy | High | Continue only with better split or tiny capped wrapper audit |
| 5 | G124/V2S target-design mini-grid | Medium; prior all-row pilot technically good but low blend utility | Medium | Only with changed targets/localmax/power, not unchanged rerun |
| 6 | Fresh source-clean public candidates | Unknown; no fresh >0.949 clue from 00:22 scan | Medium | Rescan later; do not duplicate v617/v620/v616 |
| 7 | Alexy sidecar | Low until source/checkpoint access is clean | Medium | Blocked; direct replay already scored `0.923` |

## New model/control data points

### Broad negative/no-call mask

- Script: `scripts/birdclef_oof_teacher_negative_mask.py`
- Output summary: `artifacts/pseudolabels/oof-negative-cache/b0v26_nfnetv29_teacher_neg003_cap64_20260526.summary.json`
- Raw threshold coverage: 48,650 cells, 1,259/1,279 rows, 233/234 classes, 1 false-negative cell.
- Capped mask coverage: 47,343 cells, 1,259/1,279 rows (98.4%), 230/234 classes (98.3%), **0 false-negative cells**, mean 37.6 negatives per covered row.

### Soft OOF-teacher 1024 ep4 control

- Config: `configs/birdclef/sed_b0_oofteacher_b0v26_nfnetv29_soft_1024_ep4_20260526.json`
- Artifact root: `artifacts/sed_oof_teacher_students/sed-b0-oofteacher-b0v26-nfnetv29-soft-1024-ep4-20260526/`
- Rows/classes: 1,024 examples, 819 train / 205 validation, 234 classes.
- Metric: macro AUC `0.911067` over 122 valid classes; best val loss `0.318399`.
- Runtime/export: 26.059s CUDA; TorchScript 15.389 MB; ONNX exported and checker passed; CPU inference smoke 4 files in 0.199s total / 0.050s per file; all 234 columns nonconstant.

### Soft OOF-teacher + broad negative aux 1024 ep4

- Config: `configs/birdclef/sed_b0_oofteacher_b0v26_nfnetv29_soft_broadneg003_w001_1024_ep4_20260526.json`
- Artifact root: `artifacts/sed_oof_teacher_students/sed-b0-oofteacher-b0v26-nfnetv29-soft-broadneg003-w001-1024-ep4-20260526/`
- Rows/classes: 1,024 examples, 819 train / 205 validation, 234 classes.
- Aux coverage on selected rows: 1,024/1,024 rows, 37,993 negative cells, 37.1 cells per row.
- Metric: macro AUC `0.908278` over 122 valid classes; best val loss `0.318376`.
- Runtime/export: 32.685s CUDA; TorchScript 15.389 MB; ONNX exported and checker passed; CPU inference smoke 4 files in 0.185s total / 0.046s per file; all 234 columns nonconstant.

## Critic review

- The broad negative cache solved the prior coverage blocker (5.08% row overlap -> 100% selected-row coverage), so the experiment was worth running.
- The control is essential: the improvement from `0.819` to `~0.91` is mostly due to using 1,024 rows / 4 epochs rather than the negative auxiliary term.
- The broad negative auxiliary is not currently beneficial at weight `0.01`: it slightly lowers AUC versus soft-only while improving val loss only marginally. Do not scale the aux branch unchanged.
- The soft-only control is the first recent B0 OOF-teacher branch near the useful `0.90–0.93` smoke band, but evidence is still random-split comparison-grade, not hidden-safe approval.

## Verifier decision

- Competition integrity: **ACCEPTED** for no-slot training. Uses official train audio plus OOF caches; no hidden/test labels, no public-output-only final, no Kaggle submission.
- Output/schema: **not submit-capable** yet. These are model artifacts and holdout predictions, not a hidden-safe 240-row/234-class competition output or v616-audited sidecar.
- Export/runtime: TorchScript/ONNX + CPU smoke passed for both models.
- Submission decision: **not approved** this run.

## Next exact action

Package the soft `1024×ep4` B0 student as a raw 234-class sidecar for no-slot dry-run/v616 audit, or run the distinct 20s temporal/localmax smoke if we want another model-family data point before packaging.
