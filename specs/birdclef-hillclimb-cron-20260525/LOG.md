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

## 2026-05-25 data-point training policy
- User advised training the new models anyway to obtain more data points. Updated hill-climb spec: new distinct model families should be trained as measured data points even if not immediate submission-grade, with an experiment ledger for model family/init/rows/targets/window/loss/runtime/CV/correlation/export status/diversity value.

## 2026-05-25 23:07 UTC — capped slots + soundscape non-Aves/no-train data point

- Live check: best remains `0.949`; `v616` tied; 2026-05-25 UTC slots are `5/5`; `v617`-`v620` remain pending; ~52 min to reset.
- Active job check: no local/trainer BirdCLEF jobs before this run.
- Scout refresh: EfficientAT and PANNs/Cnn14 remain the best AudioSet event/no-call leads, but trainer venv currently lacks `panns_inference`, TensorFlow/TF-Hub, and PaSST packages, so a real AudioSet branch needs asset packaging first.
- Trained a distinct soundscape-native data point anyway: `soundscape-nonaves-notrain-b0-5s160-siteS08-ep3-20260525` using official `train_soundscapes` 5s labels scoped to 72 non-Aves/no-train classes.
- Result: 1,478 windows; site-holdout `S08` 120 rows; runtime 19.46s CUDA; TorchScript+ONNX export passed; ONNX checker OK; CPU TorchScript smoke OK. Macro AUC only `0.48865` overall / `0.47610` no-train, with highly uneven sonotype behavior.
- Decision: no submission/no scale. Keep as comparison-grade landscape data point; next after reset is monitor `v617`-`v620`, then package EfficientAT/PANNs AudioSet embedding branch or run site-balanced/group-DRO non-Aves smoke.

## 2026-05-26 00:22 UTC — Reset-day PANNs/Cnn14 AudioSet data point

- Live Kaggle check: best remains `0.949`; 2026-05-26 UTC slots `0/5`; v617/v620 tied `0.949`, v618 `0.946`, v619 `0.944`. No local/trainer active jobs.
- No submission made: early UTC day and no verifier-grade/high-info non-duplicate candidate ready.
- Public scout artifact: `artifacts/public_kernels_20260526_scout/scan_20260526T0020Z.json`; no fresh clean >0.949 lead.
- Trained PANNs/Cnn14 AudioSet embedding branch via `scripts/birdclef_panns_soundscape_embedding_train.py` and config `configs/birdclef/panns_cnn14_audioset_soundscape_nonaves_notrain_nocall_siteS08_ep12_20260526.json`.
- Result: 1,478 official soundscape windows, 72 labels + no-call aux, S08 macro AUC `0.517333`, no-train AUC `0.520824`, embedding extraction 49.84s CUDA, best val loss `0.45604`.
- Verifier: artifacts finite/nonconstant, TorchScript head smoke passed; not submission-format and no slot approved.
- Reports: `ranked_queue_20260526T0022Z.md`, `model_data_point_20260526T0022Z_panns_cnn14_audioset_soundscape.md`, ledger under `artifacts/model_data_point_ledger/`.

## 2026-05-26 02:20 UTC — Broad OOF negative/no-call mask + 1024-row control

- Live check: best remains `0.949`; v616 tied; v617/v620 tied, v618/v619 dropped; 2026-05-26 UTC slots `0/5`; no active BirdCLEF jobs locally/on trainer before run.
- Early-day slot decision: no submission. No verifier-grade candidate was ready; exact/tied replays are forbidden.
- Built broad OOF-teacher-derived negative/no-call mask from `b0v26_nfnetv29_w090010_intersection_cache.npz` using threshold `0.03` and cap `64` negatives per row. Capped coverage: 47,343 cells, 1,259/1,279 rows, 230/234 classes, 0 false-negative cells.
- Trained broad-neg B0 soft OOF-teacher student: 1,024 rows, 4 epochs, aux weight `0.01`; macro AUC `0.908278` over 122 classes; TorchScript/ONNX export and CPU smoke passed.
- Trained matched soft-only 1,024-row/4-epoch control: macro AUC `0.911067` over 122 classes; TorchScript/ONNX export and CPU smoke passed.
- Critic/verifier: broad mask solved coverage but aux weight slightly underperformed control; keep soft-only as better promotion candidate, no submission approved. Next: no-slot sidecar/v616 audit for soft-only B0 or distinct 20s temporal/localmax smoke.
