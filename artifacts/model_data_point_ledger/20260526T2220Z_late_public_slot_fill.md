# Model/candidate evaluation — late public source slot fill — 2026-05-26 22:20 UTC
## Summary
- **Experiment id:** `late-public-source-slot-fill-20260526T2220Z`
- **Branch family:** source-clean public code candidates / late-day exploratory slot fill.
- **Purpose:** use the remaining UTC submission budget inside the `<3h` reset window after repo-owned sequence/model branches failed verifier-grade submission gates.
- **Evidence level:** late-day exploratory / schema-verifier-grade for code path; LB pending at write time.
- **Slots:** start `0/5`, submitted `5`, end estimated `5/5`.

## Candidate preflight table
| Candidate | Source | Version | Dry-run rows | CSV check | Public hash | Submit ref | Decision |
|---|---|---:|---:|---|---|---|---|
| v621 | `pilkwang/birdclef-2026-eos7-sz-oof-gated-pcen-sidecar` | 22 | 3x235 | finite/nonconstant; uniq100=96 | `2cd2be250a4020a4` | `53063922` | submitted; LB pending |
| v622 | `beicicc/bc26-eos6-p090-may23` | 6 | 3x235 | finite/nonconstant; uniq100=96 | `62274b98d6a4f39c` | `53063923` | submitted; LB pending |
| v623 | `mohamadmatali/bc2026-claude-anthony-m5only-fork` | 1 | 3x235 | finite/nonconstant; uniq100=91 | `09ef02cb55ff66b7` | `53063925` | submitted; LB pending |
| v624 | `hanijezo/bc2026-claude-haru-public-top2-p125-fork` | 1 | 3x235 | finite/nonconstant; uniq100=91 | `97cd802bb60f6b83` | `53063927` | submitted; LB pending |
| v625 | `sultanalgizani/bc2026-claude-safar-0948-fork` | 1 | 3x235 | finite/nonconstant; uniq100=93 | `89438737d0b97271` | `53063928` | submitted; LB pending |

## Critic / verifier decision
Proceed for late-day slot fill. No repo-owned verifier-grade candidate was available; the sequence 72→234 wrapper lost to v616 locally and was rejected. The submitted public candidates were selected after rejecting ERROR, running, malformed, nonnumeric/NaN, empty, and duplicate-public-output candidates. Each selected source contains a hidden-test `test_soundscapes` path, reads `sample_submission.csv`, and produced finite/nonconstant `submission.csv` in public dry-run.

## Rejected/blocked examples from scout
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

## Artifacts
- Scout report: `artifacts/public_kernels_20260526_late_scout/late_scout_20260526T2216Z.json`
- Submit report: `artifacts/public_kernels_20260526_late_scout/submit_v621_v625_late_fill_20260526.json`
- Downloaded public dry-run outputs: `artifacts/public_kernels_20260526_late_scout/*_submission.csv`
- Source snapshots: `artifacts/public_kernels_20260526_late_scout/sources/`
