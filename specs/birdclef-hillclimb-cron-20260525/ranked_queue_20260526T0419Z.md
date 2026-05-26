# Ranked queue — BirdCLEF hill-climb cron — 2026-05-26 04:19 UTC

## Live state verified
- Best public LB remains **0.949**; `v616` is still the tied baseline to beat.
- Latest scored submissions: `v617=0.949`, `v618=0.946`, `v619=0.944`, `v620=0.949`; none improved over `v616`.
- UTC submission slots used for 2026-05-26: **0/5** at run start.
- Time to UTC reset at run start: about **19.7h**.
- Active local/trainer jobs: none relevant; trainer GPUs idle before training.
- Scout refresh: web/search scan surfaced no fresh clean >0.949 BirdCLEF lead; recent visible leads remain Nina/EoS/PANNs discussions and already-tested plateau families.

## Slot decision
No Kaggle submission this early UTC run. There is no verifier-grade or high-information, non-duplicate candidate ready for the current 0.949 plateau. Submitting `v616`/`v617`/`v620` replays or malformed/public-output-only variants is disallowed by the spec.

## Ranked queue after this run

| Rank | Candidate | Evidence / value | Decision |
|---:|---|---|---|
| 1 | Package `sed-b0-oofteacher-b0v26-nfnetv29-soft-1024-ep4` as a raw 234-class sidecar and audit vs v616 | Best recent B0 data point: macro AUC `0.911067` over 122 classes; export/runtime passed; repo-owned and 234-class. Needs row-aligned raw test/soundscape output + ensemble audit. | **ACCEPTED next no-slot action** |
| 2 | G124/V2S-init larger/all-row pilot | Distinct from B0/PANNs/non-Aves; still in default queue; prior V2S utility was tiny, but new all-row/target-design data point is useful for landscape measurement. | **ACCEPTED data-point candidate** |
| 3 | EfficientAT AudioSet embedding branch | PANNs/Cnn14 was weak but slightly better than B0 soundscape specialist; EfficientAT remains the strongest untried AudioSet family. Needs clean package/weights. | **ACCEPTED if assets clean** |
| 4 | 20s temporal/localmax B0 follow-up with better target construction | This run's 20s simple OOF-teacher branch was weak (`0.673` macro AUC), but decorrelated from 5s (`corr=0.600`). Only revisit with center/offset crops or true local-window pseudo-labels. | **NEEDS REVISION** |
| 5 | Broader negative/no-call aux variants | Broad mask coverage is solved, but aux weight `0.01` hurt vs soft-only control. Try only if paired with no-call-valid split/curriculum. | **LOWER PRIORITY** |
| 6 | Alexy/sidecar-derived model | Still source/checkpoint blocked; direct v613 scored `0.923`. | **BLOCKED** |

## Critic / Red Team
- The 20s branch is decorrelated, but the proxy score is too weak to package unchanged; a low-correlation weak model can add noise rather than useful sidecar diversity.
- Repeating B0 target variants risks over-exploring the same family; next B0 action should be packaging/audit, not another random-split training score.
- Daily slots are available, but early-day slot use without a valid high-information candidate would be leaderboard probing.

## Verifier decision
- Training artifact is rule-safe: official train audio + OOF teacher cache only, no hidden/test labels, no public-output-only final.
- Export/runtime checks passed for the 20s branch; output is **not** competition-format and not submission-grade.
- Submission: **not approved**.

## Next exact action
Build/audit a raw 234-class sidecar from the `1024_ep4` soft-only B0 student against the v616 anchor/baseline, then decide whether it deserves a private verifier or whether to move to G124/V2S.
