# BirdCLEF hill-climb ranked queue — 2026-06-01 21:05 UTC

## Live status
- UTC slots: **5/5 used** (~2.89h to reset).
- Public best remains **0.950** from v644/v647 unless pending v657-v660 beat it. v616 secondary baseline ~0.949.
- v656 distinct-backbone distill standalone scored **0.529** (delta −0.421 vs E), refuting standalone distill as a competitive foundation despite proxy DEV.

## Pending submissions
- **v657** ref `53267235` `jungchanryu/birdclef-submission2` — status `pending`, score `pending`, dry-run hash `01c39b92aa147bb4`.
- **v658** ref `53267236` `pilkwang/birdclef-2026-eos-oof-gated-pcen` — status `pending`, score `pending`, dry-run hash `abad620908ef0d89`.
- **v659** ref `53267249` `alexycactus/birdclef-2026-cnn-inference-regnety` — status `pending`, score `pending`, dry-run hash `3ea9cb640d1476c4`.
- **v660** ref `53267251` `kruzzcc/bc26-convnext-v3r3-active-a03` — status `pending`, score `pending`, dry-run hash `1f50509ddc13b13e`.

## Ranked next actions
1. Poll v657-v660 scores and immediately ledger deltas vs 0.950 E and v616. Promote only if a source ties/improves; otherwise demote as late-window public-source datapoints.
2. Stop standalone distinct-distill backbone lane; v656 hidden LB 0.529 is lane-closing. Only revisit as a tiny in-kernel blend with E if a credible package can run E and distill together, but EV is low because proxy blend weight was 0.0.
3. Continue scouting genuinely new accessible foundations; current live rescout found no SurfPerch/AudioMAE and rejected malformed/nonfinite public outputs for several 0.950-like forks.
4. No PANNs/B0/DyMN/no-call/power/head-knob variants; STOP rule remains active.
