# BirdCLEF Hill-Climb Cron Log

## 2026-05-25 setup
- Created per user directive to run ClawTeam hill-climb automation and use daily slots aggressively while preserving integrity guards.

## 2026-05-25 22:43 UTC — late-day slot fill to 5/5
- Live state: best remained `0.949`; `v616` scored `0.949`; 2026-05-25 UTC slots were `1/5`; time to reset was about `1.28h`.
- Duplicate-work check: no active local or trainer BirdCLEF jobs; Kaggle latest list had no pending submissions before this run.
- Fresh late scout saved `artifacts/public_kernels_20260525_late_scout/scan_20260525T2238Z.json`; guarded submit artifact saved `artifacts/public_kernels_20260525_late_scout/submit_v617_v620_late_slot_fill_20260525.json`.
- Critic decision: no verifier-grade new branch was ready, and preserving 4 slots inside the late window would violate the new slot-use policy. Proceed only with source-clean COMPLETE exploratory kernels that pass basic output guards.
- Rejected: WildSound v8 ERROR/no outputs; teacher P952 Exp070 cache kernels with 7992 train rows/no competition final path; Kijiang/P949/Gendaijin variants with malformed `submission.csv`; Samejima HGNetV2 final bad values; Om Modi all-zero final; Ykuroka zero-row final; Tulay mock/wrong shape; Viktoriia EfficientAT-marked inference final bad values.
- Submitted four late-day exploratory code submissions:
  - `v617: Exploratory direct Nina EoS7 sz sidecar source`, ref `53032516`.
  - `v618: Exploratory direct Kruzzcc Nina EoS4 BirdNET source`, ref `53032520`.
  - `v619: Exploratory direct Kruzzcc Mtoshi UMAP BirdNET source`, ref `53032523`.
  - `v620: Exploratory direct Kazuhiro Karnak rank fusion source`, ref `53032524`.
- Immediate post-submit check: 2026-05-25 UTC slots now `5/5`; v617-v620 pending; v616 complete `0.949`.
- Ranked queue artifact: `specs/birdclef-hillclimb-cron-20260525/ranked_queue_20260525T2243Z.md`.
- Next: monitor v617-v620 after scoring; if all tie/drop, stop EoS/ProtoSSM/SED public-source repeats and resume new-branch no-slot work (EfficientAT/PANNs event/no-call, broader no-call negative cache, non-Aves specialist, or 20s temporal branch).
