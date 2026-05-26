# Ranked queue — BirdCLEF hill-climb cron — 2026-05-26 06:34 UTC

## Live state verified
- Best public LB remains **0.949**; `v616` is still the tied baseline to beat.
- Latest scored submissions: `v617=0.949`, `v618=0.946`, `v619=0.944`, `v620=0.949`; none improved over `v616`.
- UTC submission slots used for 2026-05-26: **0/5** at this run; about **17.4h** to reset.
- Active local/trainer jobs: checked before and after; no relevant active jobs after killing a redundant bootstrap audit left by a broken SSH session.
- Public scout refresh: web/Kaggle-search queries for BirdCLEF 2026 EfficientAT/PANNs/AudioSet and fresh 0.949+ leads surfaced only generic EDA/baseline pages or irrelevant results; no new clean >0.949 public lead.

## Slot decision
No Kaggle submission this early UTC run. The only positive audit signal was a microscopic local-proxy lift from tiny G124 sidecar weights (`+0.00000339` macro AUC vs v616 on the 190-row labeled overlap), which is comparison-grade noise and not high-information enough for an early-day slot. Duplicate/tied `v616`/`v617`/`v620` replays remain disallowed.

## Work completed this run
1. Prevented duplicate work: found no active BirdCLEF train jobs; discovered the prior G124 all-row V2S-init pilot already existed on trainer, so did **not** rerun it.
2. Trained a distinct G124/V2S target-design data point anyway:
   - Config: `configs/birdclef/g124_effv2s_public946_pseudo_pilot_20260526_v2sinit_power085_localmax_ep6.json`.
   - Model: EfficientNetV2-RW-S SED student, V2S external-pretrain TorchScript init, all 792 train-soundscape teacher rows, `teacher_power=0.85`, `temporal_target_mode=local_max`, radius 1, focal BCE, 6 epochs.
   - Result: best val AUC `0.960094` over 62 valid classes; all-row truth AUC `0.944720` over 75 classes; final student/teacher corr `0.847478`, MAE `0.037473`; TorchScript and ONNX exported on trainer.
3. Packaged/audited sidecar candidates vs `v616_baseline`:
   - Generated train-soundscape raw output for the stronger soft-only B0 `1024_ep4` student and filtered it to the 240 v616 proxy rows.
   - Converted prior G124 center/all-row and new G124 localmax student predictions into sidecar CSVs.
   - Ran fast rank-blend grid: `artifacts/anchored_blend_audit/20260526T0615Z_b0_g124_sidecar/audit_vs_v616_fast.json`.

## Ranked queue after this run

| Rank | Candidate | Evidence / value | Decision |
|---:|---|---|---|
| 1 | **G124/V2S sidecar confirmer or private-verifier package** | New localmax G124 data point is strong on proxy (`0.9601` val AUC) and tiny G124 weights slightly improve v616 local AUC, but lift is only `+0.00000339` and soft-B0 weights hurt. | **NEEDS REVISION / no slot yet** |
| 2 | EfficientAT AudioSet embedding branch | Still the strongest untried AudioSet family after PANNs/Cnn14 weak-but-decorrelated result. Needs clean package/weights. | **ACCEPTED next data-point candidate** |
| 3 | G124 target mini-grid hard-confidence / power-only ablation | The localmax target is not obviously worse than prior center; a hard-conf or `teacher_power=0.85` center ablation would separate target-shape vs power effects. | **ACCEPTED bounded data point** |
| 4 | B0 soft-only raw sidecar packaging | Best recent B0 train-audio smoke (`0.911067`) produced valid raw rows, but in this audit any nonzero soft-B0 weight did not beat the top G124-only recipes. | **LOWER PRIORITY / no slot** |
| 5 | 20s temporal/localmax follow-up with true local-window targets | Prior 20s branch was decorrelated (`corr=0.600`) but weak (`0.673`). | **NEEDS REVISION** |
| 6 | Alexy/sidecar-derived model | Direct v613 scored `0.923`; source/checkpoint access still not clean. | **BLOCKED** |

## Critic / Red Team
- The G124 local proxy remains narrow and teacher-derived; `0.960` val AUC is not submission evidence because the teacher itself is `0.9955` on this overlap.
- The best blend lift vs v616 is effectively zero (`0.99348406` vs `0.99348067`) despite better top-k row recall, so using an early slot would be leaderboard probing.
- Soft-B0 raw sidecar inference was operationally heavier than expected and did not appear in the best recipes; do not spend more time packaging it unless a stronger validation slice says it helps.

## Verifier decision
- Training is rule-safe: official train soundscapes + public/owned pseudo-label cache + public/existing V2S init; no hidden/test labels.
- Outputs are finite/nonconstant; G124 sidecar CSVs have 792×234 predictions, B0 filtered sidecar has 240×234 rows aligned to the v616 proxy.
- Export status passed on trainer for G124 TorchScript/ONNX. The audit is no-slot/comparison-grade only; **submission not approved**.

## Next exact action
Train or package **EfficientAT AudioSet embeddings** as the next distinct model family if assets are clean; otherwise run a bounded G124 hard-confidence/power ablation to turn this promising-but-too-small G124 signal into a clearer data point.
