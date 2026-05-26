# Ranked queue — BirdCLEF ClawTeam hill-climb — 2026-05-26 22:20 UTC

## Live status
- Public LB best before pending submissions remains `0.949`; v616/v617/v620 are tied baselines; v618=0.946, v619=0.944.
- UTC slots at start: `0/5`; time to reset ~`1.66h`; active local/trainer BirdCLEF jobs: none.
- After this run: submitted v621-v625; estimated UTC slots `5/5`; all pending at immediate post-submit check.

## Ranked late-fill decisions
1. **v621 — `pilkwang/birdclef-2026-eos7-sz-oof-gated-pcen-sidecar`** — Highest-ranked fresh/hot public EoS7+PCEN sidecar; source has hidden test path and nonconstant public dry-run; distinct from prior repo PCEN wrapper. Submitted ref `53063922`; public dry-run `3x235`, hash `2cd2be250a4020a4`.
2. **v622 — `beicicc/bc26-eos6-p090-may23`** — High-vote EoS6/P090 rank-power branch; hidden-test capable public source; nonconstant public dry-run; not submitted under this source. Submitted ref `53063923`; public dry-run `3x235`, hash `62274b98d6a4f39c`.
3. **v623 — `mohamadmatali/bc2026-claude-anthony-m5only-fork`** — Recent Anthony/M5-only fork; chosen as one representative after duplicate-output Anthony variants; nonconstant public dry-run. Submitted ref `53063925`; public dry-run `3x235`, hash `09ef02cb55ff66b7`.
4. **v624 — `hanijezo/bc2026-claude-haru-public-top2-p125-fork`** — Recent public-top2+P125 fork with hidden-test path; distinct P125/Perch/SED blend; nonconstant public dry-run. Submitted ref `53063927`; public dry-run `3x235`, hash `97cd802bb60f6b83`.
5. **v625 — `sultanalgizani/bc2026-claude-safar-0948-fork`** — Recent Safar-0948 fork; claimed near-plateau lineage; hidden-test path and nonconstant public dry-run; chosen over malformed/static candidates. Submitted ref `53063928`; public dry-run `3x235`, hash `89438737d0b97271`.

## Rejected / blocked during late scout
- `muhammadsaadalvi/birdclef-2026-wildsound-v8` — reject_status; 
- `udaken10/submit` — reject_status; 
- `ahmedkhudair121/bc2026-claude-karnak-hier-tax-fork` — reject_bad_csv; 0ee04c918f807616
- `ahmedkhudair121/bc2026-claude-nina-eos-8-fork` — reject_bad_csv; 0ee04c918f807616
- `karnakbaevarthur/hierarchical-taxonomy-post-processing-birdclef-2` — reject_bad_csv; 0ee04c918f807616
- `archishachanda04/notebook2d3524c2e1` — reject_status; 
- `nina2025/birdclef-2026-eos-7-sz` — reject_bad_csv; 5afa1de99305ffd1
- `mohamadmatali/bc2026-claude-anthony-ensemble-fork` — reject_bad_csv; 0ee04c918f807616
- `abdulrahmansu10/bc2026-claude-eos6sz-m74heavy-v1-fork` — reject_bad_csv; 0ee04c918f807616
- `sans6262q/bc2026-claude-eos6sz-balanced-v1-fork` — reject_bad_csv; 0ee04c918f807616
- `abdulrahmansu10/bc2026-claude-eos7sz-m52heavy-v1-fork` — reject_bad_csv; 5afa1de99305ffd1
- `hassan1417/bc2026-claude-eos7sz-m74heavy-v1-fork` — reject_bad_csv; 5afa1de99305ffd1
- `joriahmed/bc2026-claude-eos7sz-balanced-v1-fork` — reject_bad_csv; 5afa1de99305ffd1
- `archishachanda/birdie-clef` — reject_status; 
- `mohamadmatali/bc2026-claude-nina-eos-2-fork` — reject_bad_csv; 0ee04c918f807616
- `ahmedkhudair121/bc2026-claude-adarsh-v62-eos3-fork` — reject_bad_csv; 0ee04c918f807616
- `abdulrahmansu10/bc2026-claude-analytica-development-fork` — reject_bad_csv; 0ee04c918f807616
- `gendaijin/birdclef2026-day0527-yaroslav-v221` — reject_bad_csv; 0ee04c918f807616
- `gendaijin/birdclef2026-day0527-nina-eos7sz` — reject_bad_csv; 5afa1de99305ffd1
- `ahmedkhudair121/bc2026-claude-yaroslav-v221-fork` — reject_bad_csv; 0ee04c918f807616

## Critic + verifier
- Critic rejected submitting the train_soundscape sequence wrapper because it was `-0.00219` vs v616 in local proxy and not hidden-test packaged.
- Verifier accepted the five selected public source submissions for late-day exploratory use: no duplicate descriptions, no duplicate public-output hash within the submitted batch, finite/nonconstant CSV, hidden-test path present, and Kaggle accepted each code submission.
- Caveat: these are exploratory public-source submissions, not repo-owned trained models; scores were pending at report time.

## Next queue after scores
1. Monitor v621-v625 scores/errors and update the ledger.
2. If no >0.949 improvement, resume true hidden-safe 234-class DyMN10/AudioSet package or train_soundscape file/site branch that can run on hidden test.
3. Avoid resubmitting exact duplicate public sources/output matrices; rejected malformed/NaN candidates stay blocked.
