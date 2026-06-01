# Late-window public representation/source submissions — 2026-06-01 21:05 UTC
## Context
- Live best before submissions: 0.950 public LB (v644/v647); v616 secondary baseline ~0.949.
- v656 distinct-backbone distill standalone scored **0.529**, refuting the standalone distill-foundation lane despite proxy weak-AUC 0.8319 / decorr 0.737.
- Slot state after this action: **5/5 used**, reset in 2.89h; four late-window source-clean exploratory submissions were made after preflight.
- Rationale: no DEV-passing owned candidate remained; STOP-rule head knobs stayed frozen. Remaining slots were spent only on finite/nonconstant public code sources with representation/data-source variation (BirdNET/PCEN/RegNetY/ConvNeXt), not on shared-embedding head ablations.

## Submitted candidates
| label | ref | public source | publicScore | status | preflight | decision |
|---|---:|---|---:|---|---|---|
| v657 | 53267235 | `jungchanryu/birdclef-submission2` | pending | pending | v36; 3x235; uniq100=97; hash=01c39b92aa147bb4 | submitted; compare vs E 0.950 and v616 0.949 when scored |
| v658 | 53267236 | `pilkwang/birdclef-2026-eos-oof-gated-pcen` | pending | pending | v9; 3x235; uniq100=97; hash=abad620908ef0d89 | submitted; compare vs E 0.950 and v616 0.949 when scored |
| v659 | 53267249 | `alexycactus/birdclef-2026-cnn-inference-regnety` | pending | pending | v3; 192x235; uniq100=99; hash=3ea9cb640d1476c4 | submitted; compare vs E 0.950 and v616 0.949 when scored |
| v660 | 53267251 | `kruzzcc/bc26-convnext-v3r3-active-a03` | pending | pending | v1; 240x235; uniq100=90; hash=1f50509ddc13b13e | submitted; compare vs E 0.950 and v616 0.949 when scored |

## Critic / verifier decision
- All submitted candidates had Kaggle session status COMPLETE, `test_soundscapes` + `sample_submission` + `submission.csv` source markers, finite/nonconstant public outputs, 235 columns, and nonduplicate dry-run hashes versus the local recent-fill guard.
- Rejected during this run: malformed/nonfinite public outputs for Nina/Anthony/Fleong/Ahmed EoS forks and Liuyanfeng JAX Perch; duplicate/known-lower sources were not submitted.
- These are guarded late-window data points, not promotions over the 0.950 frontier. Scores pending at write time.
