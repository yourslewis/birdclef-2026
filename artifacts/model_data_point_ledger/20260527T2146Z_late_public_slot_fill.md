# BirdCLEF late-day public source slot fill — 2026-05-27 21:46 UTC

## Status
- Live Kaggle check before submission: best public LB remained `0.949`; v621/v622/v623 tied `0.949`, v625 `0.948`, v624 `0.943`.
- UTC slots before action: `0/5`; time to reset: `2.23h`.
- No active local/trainer BirdCLEF jobs were found before action.
- Slot policy state: late UTC day (`<3h` to reset), so valid source-code exploratory candidates were eligible.

## Critic / verifier decision
- Critic: no trained train_soundscape branch had a positive v616-sidecar gate; direct all-class sequence sidecars remained negative vs v616. Letting five slots expire was lower-value than guarded source reruns.
- Verifier: submitted only Kaggle source-code reruns, not static public CSV uploads. Each selected source was COMPLETE, exposed `test_soundscapes`/sample/submission markers, had a finite/nonconstant public dry-run `submission.csv`, and did not duplicate recent submitted dry-run hashes.
- Evidence level: exploratory/source-clean, not verifier-grade model improvement.

## Submitted candidates

| Label | Source family | Kaggle source | Ref | Public dry-run rows | uniq_first100 | dry-run hash | Status |
|---|---|---|---:|---:|---:|---|---|
| v626 | Perch meta/probe soundscape source | `shahadaljayzani/bc2026-claude-jaejohn-perch-starter-fork` | 53097345 | 240 | 95 | `7439ae3b15a3f6c6` | pending |
| v627 | ProtoSSM source/fork | `abdulrahmansu10/bc2026-claude-hideyukizushi-protossm-src-fork` | 53097346 | 240 | 89 | `6b5910239e37bd4b` | pending |
| v628 | Gate-combo ensemble | `sultanalgizani/bc2026-claude-cliff-gate-combo-fork` | 53097347 | 3 | 95 | `aa41ccbbf2a84046` | pending |
| v629 | BirdNET/Yaroslav public source | `hassanalgizani/bc2026-claude-yaroslav-birdnet-3rd-fork` | 53097348 | 3 | 92 | `3224ced9a582e251` | pending |
| v630 | Distilled SED public source | `hassan1417/bc2026-claude-tucker-distilled-sed-fork` | 53097349 | 60 | 100 | `e293021c399fa925` | pending |


## Decisions and caveats
- Submitted `5/5` available daily slots: refs `53097345, 53097346, 53097347, 53097348, 53097349`.
- None of these rows supersede the repo-owned v616/v621-v623 tied baseline until Kaggle scores return.
- If any score exceeds `0.949`, promote it to the top comparison set and inspect source lineage/private-risk. If all tie/drop/error, resume repo-owned hidden-safe 234-class PANNs/fused sequence packaging or no-call/acoustic-context protocol.

## Artifacts
- Submit report: `artifacts/public_kernels_20260527_late_scout/submit_v626_v630_late_fill_20260527.json`
- Dry-run report: `artifacts/public_kernels_20260527_late_scout/dryrun_v626_v630_late_fill_20260527.json`
- Submitter script: `scripts/submit_v626_v630_late_slot_fill_20260527.py`
