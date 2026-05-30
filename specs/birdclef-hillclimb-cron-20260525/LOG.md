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

## 2026-05-26 04:19 UTC — 20s temporal/localmax B0 data point

- Live check: best remains `0.949`; v616 is the tied baseline; v617/v620 tied, v618/v619 dropped; 2026-05-26 UTC slots `0/5`; no active BirdCLEF jobs locally/on trainer before run.
- Early-day slot decision: no submission. No verifier-grade/high-info non-duplicate candidate was ready; daily slots remain available for later candidates.
- Scout refresh: web/search scan surfaced no fresh clean >0.949 public lead; visible recent leads are already-tested Nina/EoS/PANNs/discussion/plateau families.
- Trained next distinct data point from the default queue: `sed-b0-oofteacher-b0v26-nfnetv29-soft-20s-localmax-512-ep3-20260526`.
- Setup: EfficientNet-B0 SED frame model with clip pooling `0.5*mean + 0.5*amax`; q3/cap80 external init; 512 OOF-teacher-backed train-audio files; all 234 classes; 20s/160-mel input; BCE, no mixup/no aux, 3 epochs.
- Result: runtime `20.778s` CUDA; best val loss `0.322308`; macro AUC `0.672996` over 72 valid classes. Correlation vs 5s soft-only 1024/ep4 B0 on 407 overlapping files: global Pearson `0.599986`, MAE `0.036360`.
- Export/runtime: TorchScript `15.389 MB`; ONNX checker OK; CPU TorchScript inference smoke on 4 files `0.301s` total / `0.075s` per file, finite 234-class output.
- Critic/verifier: accepted as no-slot landscape artifact, rejected as submission-grade. It is decorrelated but too weak; do not package unchanged. Revisit only with true local-window/offset pseudo-labels or multi-crop localmax aggregation.
- Reports: `ranked_queue_20260526T0419Z.md`, `model_data_point_20260526T0419Z_20s_localmax.md`, ledger `artifacts/model_data_point_ledger/20260526T0419Z_20s_localmax.md`.
- Next: package/audit the stronger `1024_ep4` soft-only B0 student as a raw 234-class sidecar against v616, or move to G124/V2S if B0 sidecar audit fails.

### 2026-05-26 06:34 UTC — G124/V2S target-design localmax data point + B0/G124 sidecar audit

- **Live state:** best remains **0.949**; `v616` is still the tied baseline. Latest scored submissions remain `v617=0.949`, `v618=0.946`, `v619=0.944`, `v620=0.949`. 2026-05-26 UTC slots used **0/5** with ~17.4h to reset. No active jobs remained after cleanup.
- **Scout refresh:** web/Kaggle-search queries for EfficientAT/PANNs/AudioSet and fresh 0.949+ leads found no clean new public lead; results were generic EDA/baseline or irrelevant.
- **Duplicate prevention:** prior G124 all-row V2S-init center pilot already existed on trainer, so it was not rerun.
- **Trained data point:** `g124-effv2s-public946-pseudo-pilot-20260526-v2sinit-power085-localmax-ep6` using EfficientNetV2-RW-S, external V2S init, 792 teacher train-soundscape rows, `teacher_power=0.85`, local-max radius 1 targets, focal BCE, 6 epochs. Best val AUC `0.960094` over 62 valid classes; all-row student AUC `0.944720`; student/teacher corr `0.847478`; TorchScript+ONNX export passed on trainer.
- **Sidecar audit:** generated raw train-soundscape predictions for the soft-only B0 `1024_ep4` student, filtered to the 240 v616 proxy rows, converted G124 center/localmax predictions to sidecar CSVs, and ran `audit_vs_v616_fast.json`. Best tiny G124-only recipe lifted local proxy from `0.993480668` to `0.993484059` (`+0.00000339`) with corr `0.999986`; soft-B0 weights did not help.
- **Decision:** no submission. The G124 signal is interesting but the local lift is too small and teacher/proxy-derived for an early-day Kaggle slot. Next: EfficientAT AudioSet embedding branch if assets are clean, otherwise a bounded G124 hard-confidence/power ablation.


## 2026-05-26 06:59 UTC EfficientAT MN10 AudioSet soundscape data point
- User explicitly requested training the EfficientAT embedding branch. Implemented `scripts/birdclef_efficientat_soundscape_embedding_train.py` and config `configs/birdclef/efficientat_mn10_audioset_soundscape_nonaves_notrain_nocall_siteS08_ep12_20260526.json`.
- Cloned EfficientAT to trainer `/home/yourslewis/external_models/EfficientAT`, installed missing `wget`, and used public AudioSet `mn10_as` checkpoint.
- Trained 72-label non-Aves/no-train/no-call soundscape head on 1,478 official 5s train-soundscape windows, site-holdout S08. Embedding extraction 13.30s; best val loss 0.487352 at epoch 5; macro AUC 0.488240 over 18 valid classes; no-train AUC 0.472842 over 17 classes.
- Verification: holdout predictions finite/nonconstant; TorchScript head smoke passed (`2x960 -> 2x72 + 2x1`). No submission: 72-label specialist only and weaker than PANNs/Cnn14 AudioSet branch (`0.517333`).

### 2026-05-26 08:20 UTC — EfficientAT DyMN10 AudioSet soundscape embedding branch

- **Live state:** best remains **0.949**. Bearer API listing shows no 2026-05-26 UTC submissions yet; latest scored are `v617=0.949`, `v618=0.946`, `v619=0.944`, `v620=0.949`, with `v616` still the tied repo-owned baseline. Slots used: **0/5** with ~15.7h to reset. No active BirdCLEF jobs were found locally/on trainer before the run.
- **Scout/critic:** role report `specs/birdclef-hillclimb-cron-20260525/reports/scout_critic_20260526T0815Z.md` recommended EfficientAT `dymn10_as` as the next bounded no-slot data point and rejected early-day submission.
- **Training:** added config `configs/birdclef/efficientat_dymn10_audioset_soundscape_nonaves_notrain_nocall_siteS08_ep12_20260526.json` and trained with existing `scripts/birdclef_efficientat_soundscape_embedding_train.py` on trainer. Used public EfficientAT `dymn10_as.pt`, official train-soundscape 5s windows, 72 non-Aves/no-train labels, no-call aux, site holdout `S08`, 12 epochs.
- **Result:** extracted `1478 x 960` embeddings in `36.23s` CUDA; best val loss `0.428341`; S08 macro AUC `0.568586` over 18 valid scoped classes; no-train AUC `0.553327`; no-call AUC invalid on S08.
- **Comparison:** DyMN10 beat EfficientAT MN10 (`0.488240`) and PANNs/Cnn14 (`0.517333`) on this same target contract, so AudioSet remains alive as a rare-slice sidecar lane.
- **Verifier:** finite/nonconstant holdout predictions shape `120 x 72`; TorchScript head smoke passed `(2,960)->(2,72)+(2,1)`. Not submission-format; no Kaggle slot approved.
- **Artifacts:** `artifacts/efficientat_soundscape_embeddings/efficientat-dymn10-audioset-soundscape-nonaves-notrain-nocall-siteS08-ep12-20260526/`, log `logs/efficientat_dymn10_audioset_soundscape_nonaves_notrain_nocall_siteS08_ep12_20260526.log`, ledger `artifacts/model_data_point_ledger/20260526T0820Z_efficientat_dymn10_soundscape.md`, queue `specs/birdclef-hillclimb-cron-20260525/ranked_queue_20260526T0820Z.md`.
- **Next:** run multi-site/leave-one-site evaluation for AudioSet heads and decide whether DyMN10 deserves a 234-class sidecar wrapper; otherwise pivot to G124 hard-confidence/power ablation.

## 2026-05-26 train-soundscape sequence/deeper queue correction
- User asked whether the data-driven train-soundscape directions are already in queue and whether there is a deeper training variant.
- Finding: partial queue existed (non-Aves/no-train B0, PANNs/EfficientAT embeddings, broader OOF negative/no-call, 20s localmax, G124/V2S), but no explicit top-priority sequence/file/site mining branch and no true deeper soundscape-native variant beyond row-level CNN/SED smokes.
- Updated spec and cron prompt so the top queue is now: (1) train-soundscape sequence/file/site mining with MIL/temporal/file context and leave-site/file validation, (2) deeper soundscape-native training variant with last-block/adapters/compact CNN/SED on task-aligned targets, then AudioSet reformulation and existing queues.

## 2026-05-26 10:20 UTC — train_soundscapes sequence/file/site mining data point

- Live state via Bearer API: best `0.949`, v616 tied baseline, latest v617/v620 tied and v618/v619 dropped; 2026-05-26 slots `0/5`.
- No early-day submission: no verifier-grade, competition-format, nonduplicate candidate exists.
- Implemented sequence-aware mining script/config: `scripts/birdclef_soundscape_sequence_mining.py`, `configs/birdclef/soundscape_sequence_dymn10_context_losite_ep16_20260526.json`.
- Trained/evaluated DyMN10 context features on 1,478 official train-soundscape windows grouped by 60 files / 9 sites, 72 non-Aves/no-train labels.
- Leave-site row AUC: row-only `0.578422`, context `0.601355`, delta `+0.022933`. File-MIL AUC: `0.563852` -> `0.632127`.
- Positive fold deltas S19/S23/S13; regressions S03/S22, so comparison-grade only.
- Verifier checks passed: finite/nonconstant `(1314,72)` predictions and TorchScript smoke `(2,5764)->(2,72)`. Not competition-format; no slot approved.
- Reports: `ranked_queue_20260526T1020Z.md`, `model_data_point_20260526T1020Z_soundscape_sequence_mining.md`, `reports/scout_critic_20260526T1020Z.md`, ledger `artifacts/model_data_point_ledger/20260526T1020Z_soundscape_sequence_mining.md`.

## 2026-05-26 16:20 UTC — compact soundscape-native B0 leave-site data point
- Live status via Kaggle Bearer API: best remains `0.949`; latest scored `v616=0.949`, `v617=0.949`, `v618=0.946`, `v619=0.944`, `v620=0.949`; 2026-05-26 UTC slots `0/5` with ~7.66h to reset; no active local/trainer BirdCLEF jobs after run.
- Scout/critic report `specs/birdclef-hillclimb-cron-20260525/reports/scout_critic_20260526T1615Z.md` found no fresh clean public lead and recommended a bounded compact soundscape-native branch, not another TCN/gating smoother.
- Added `scripts/birdclef_soundscape_native_losite_train.py` and config `configs/birdclef/soundscape_native_b0_losite_nonaves_notrain_ep4_20260526.json`.
- Trained `soundscape-native-b0-losite-nonaves-notrain-ep4-20260526`: EfficientNet-B0 SED-style compact CNN, q3/cap80 train-audio init, official `train_soundscapes` only, 1,478 5s windows / 66 files / 9 sites, 72 non-Aves/no-train labels, leave-site folds, BCE, observed-sqrt pos weights, label smoothing/mixup, 4 epochs.
- Result: 6 completed folds / 2 skipped; leave-site row macro AUC mean `0.558044`, no-train row AUC `0.573554`, file-MIL macro AUC `0.429828`; pooled row AUC `0.396540`, pooled no-train `0.305887`. This underperforms the DyMN10 context-MLP sequence artifact (`0.601355` row / `0.632127` file-MIL), so it is a negative/diagnostic data point.
- Verifier: leave-site predictions finite/nonconstant (`1314x72`, `72/72` nonconstant); TorchScript and ONNX export/check passed. No 234-class wrapper/v616 audit; no Kaggle submission.
- Reports/artifacts: `specs/birdclef-hillclimb-cron-20260525/ranked_queue_20260526T1620Z.md`, `artifacts/model_data_point_ledger/20260526T1620Z_soundscape_native_losite.md`, artifact root `artifacts/soundscape_native_losite/soundscape-native-b0-losite-nonaves-notrain-ep4-20260526/`.
- Decision: no early-day slot. Next exact action: use context-MLP as the control; either run one regularized/worst-site context ablation then wrapper/audit, or reformulate DyMN10/AudioSet into a multi-site 234-class sidecar.

## 2026-05-26 20:25 UTC — sequence sidecar wrapper audit
- Live check: best `0.949`; v616/v617/v620 tied, v618 `0.946`, v619 `0.944`; 2026-05-26 slots `0/5`; ~3.7h to reset.
- No valid nonduplicate submission-ready candidate; quick scout found no clean fresh >0.949 lead.
- Added/evaluated `scripts/birdclef_soundscape_sequence_sidecar_audit.py`, wrapping current train_soundscape sequence predictions into 234-class v616 proxy sidecars.
- Best combo `seq_context02_r201`: local macro AUC `0.991293583`, lift vs anchor `+0.000903076`, lift vs v616 `-0.002187085`. Best single context sidecar: `0.991279099`, lift vs v616 `-0.002201568`. Best single r2 sidecar: `0.991031704`, lift vs v616 `-0.002448964`.
- Decision: reject as slot candidate; update canonical performance table and ledger. Next exact action: late-day slot-fill review inside `<3h`, or a true hidden-safe 234-class DyMN10/AudioSet package.

## 2026-05-29 02:25 UTC — Soft1279 site-balanced native ablation

- Status: best public LB `0.949`; UTC slots `0/5`; no active BirdCLEF jobs; trainer GPUs free after run.
- Trained/evaluated `soundscape-native-b0-soft1279init-sitebalanced-losite-allcls-ep4-20260529` and package/audited its sidecar.
- Performance table: row AUC `0.569405`, no-train `0.559505`, non-Aves `0.545574`, file-MIL `0.513779`; package best w0.16 local AUC `0.993104`, lift vs v616 `-0.000377`.
- Decision: reject/no submission; site-balanced sampling worsened soft1279 adaptation and did not improve proxy gates.
- Artifacts: `ranked_queue_20260529T0225Z.md`, `artifacts/model_data_point_ledger/20260529T0225Z_soundscape_native_soft1279init_sitebalanced_allclass.md`, `artifacts/model_data_point_ledger/20260529T0225Z_soft1279init_sitebalanced_package_audit.md`.

## 2026-05-30 06:20 UTC — PANNs file-context no-MIL ablation + farneg20 no-call protocol

- Live check: public best still `0.949`; v636-v640 scored `0.944/0.943/0.939/0.944/0.945`; UTC slots `0/5`; no local/trainer BirdCLEF jobs and GPUs free.
- Trained `soundscape-sequence-panns-cnn14-allcls-r2-filectx-nomil-losite-ep20-20260530` on official train_soundscapes: 1,478 windows / 66 files / 9 sites / 234 labels, frozen PANNs/Cnn14 embeddings, radius-2 full file context, no file-MIL loss.
- Result: row AUC `0.626315`, no-train `0.604598`, non-Aves `0.685213`, file-MIL `0.649487`; vs PANNs filectx+fileMIL row `-0.017957` / file-MIL `-0.029401`.
- Sidecar audit best `allcls_seq_w0p0025`: local AUC `0.990783` / 42 valid, lift vs v616 `-0.002698`; rejected/no submission.
- Evaluated `soundscape-nocall-gate-soft1279native-agg-farneg20-losite-20260530`: 752 retained rows, only 13 weak negatives, 0 valid leave-site AUC folds. Rejected; distance-only negative filtering beyond 10–12s collapses multi-site validation.
- Updated performance table/jsonl, ledgers, ranked queue. Next: soft1279 head-loaded class/site movement diagnosis or curated multi-site no-call negatives; avoid more blind PANNs file-context variants.

## 2026-05-30 08:20 UTC — G124/V2S hard-confidence localmax target-shape ablation

- Live Kaggle Bearer check: best public LB remains `0.949`; latest v636-v640 completed `0.944/0.943/0.939/0.944/0.945`; 2026-05-30 UTC slots `0/5`; no active local/trainer BirdCLEF jobs; trainer GPUs free.
- Trained `g124-effv2s-public946-pseudo-pilot-20260530-v2sinit-hardconf90-localmax-ep6` on 792 train_soundscape teacher rows / 66 files / 9 sites / 234 labels, EfficientNetV2-RW-S from V2S init, localmax radius-1 pseudo targets, hard-confidence positives `>=0.90` and negatives `<=0.01` with row/class caps.
- Effective target mask was very sparse: `0.995%` cells (`618` positives, `1,226` negatives). Metrics: best val AUC `0.622851` / 67 valid, all-row student AUC `0.623120` / 75 valid, teacher same-row AUC `0.995541`, student/teacher corr `0.141234`.
- Student-pool blend audit: best teacher blend w0.005 AUC `0.997018` / 75 valid, lift vs teacher `-0.000000443`; site bootstrap q05 `-0.00000704`; leave-site q05 `-0.00000176`; only 3/9 held-out sites positive.
- Decision: reject/no submission. Hard-confidence-only target-shape training is much worse than prior G124 soft localmax (`0.960094` val / `0.944720` all-row) and is not package-worthy. Updated performance table/jsonl, ledger `artifacts/model_data_point_ledger/20260530T0820Z_g124_v2s_hardconf90_localmax.md`, and queue `specs/birdclef-hillclimb-cron-20260525/ranked_queue_20260530T0820Z.md`.

## 2026-05-30 10:20 UTC — G124/V2S soft-anchor localmax target-shape ablation

- Live check: public best still `0.949`; latest v636-v640 completed `0.944/0.943/0.939/0.944/0.945`; UTC slots `0/5`; no active local/trainer BirdCLEF jobs; trainer GPUs free.
- Trained `g124-effv2s-public946-pseudo-pilot-20260530-v2sinit-softanchor90-localmax-ep6` on 792 train_soundscape teacher rows / 66 files / 9 sites / 234 labels using EfficientNetV2-RW-S from V2S init, localmax radius-1 pseudo targets, and `soft_anchor` targets (`soft_label_weight=0.5`, anchors >=0.90/<=0.01 with row/class caps).
- Metrics: best val AUC `0.961641` / 67 valid; all-row student AUC `0.965053` / 75 valid; teacher AUC `0.995541`; corr `0.856930`. This slightly beats the prior G124 soft localmax training metric and massively beats hardconf90, confirming hardconf target starvation.
- Teacher-cache blend audit: best w`0.02` AUC `0.997042`, lift vs teacher `+0.00002330`; site bootstrap q05 `-0.00016009` and leave-site q05 `-0.00001043`.
- Decision: no submission. Positive diagnostic, but teacher-cache lift is tiny/unstable and not a v616 sidecar audit. Updated performance table/jsonl, ledger `artifacts/model_data_point_ledger/20260530T1020Z_g124_v2s_softanchor90_localmax.md`, and queue `specs/birdclef-hillclimb-cron-20260525/ranked_queue_20260530T1020Z.md`.

## 2026-05-30 12:20 UTC — G124/V2S soft-anchor center/localmax-mix diagnostic

- Status: Bearer API live best `0.949`; v636-v640 scored `0.944/0.943/0.939/0.944/0.945`; UTC slots `0/5`; trainer idle.
- Trained/evaluated `g124-effv2s-public946-pseudo-pilot-20260530-v2sinit-softanchor90-centerlocalmix75-ep6`.
- Metrics table row added: val AUC `0.959950` / 67; all-row `0.962337` / 75; best teacher blend lift `+0.00003724`; v616 sidecar lift `-0.002286`.
- Decision: no submission; early-day slot policy plus failed v616 gate. Next exact action: soft1279 head-loaded class/site movement diagnosis.

## 2026-05-30 14:20 UTC — soft1279 head-loaded movement diagnosis

- Live status: public LB best `0.949`; v636-v640 `0.944/0.943/0.939/0.944/0.945`; UTC slots `0/5`; no active local/trainer jobs; trainer GPUs idle.
- Evaluated strongest soft1279 sidecar `soft1279init_native_allcls_w0p16`: 240 proxy rows, 190 matched label rows, 20 files, 6 sites, 42 valid classes.
- Metrics: candidate AUC `0.995545`, v616 AUC `0.993481`, lift `+0.002064`; anchor AUC `0.990391`, lift `+0.005155`; raw sidecar AUC `0.994941`.
- Diagnosis: aggregate gain is concentrated (`S03`/`S22`), `S18` regresses, no-train-primary lift is only `+0.000190`, Mammalia mean lift is negative.
- Decision: no submission; comparison-grade only. Updated performance table/jsonl, ledger, ranked queue, docs log. Next: curated multi-site no-call negatives or robust caps limited to stable soft1279 winners.

## 2026-05-30 20:20 UTC

- Live check: public best still `0.949`; v636-v640 scored `0.944/0.943/0.939/0.944/0.945`; UTC slots `0/5`; no active local/trainer jobs and trainer GPUs idle.
- Trained `soundscape-sequence-panns-cnn14-allcls-r4-20s-localmeanmax-losite-ep20-20260530` on official train_soundscapes: 1,478 windows / 66 files / 9 sites / 234 labels, frozen PANNs/Cnn14 embeddings, radius-4 local mean+max temporal context (±20s), 20 epochs.
- Metrics: context row AUC `0.627559`, no-train `0.604764`, non-Aves `0.665293`, file-MIL `0.673926`; internal row-only row `0.626128` / file-MIL `0.671220`; below prior best PANNs filectx+fileMIL row `0.644272`.
- Sidecar audit best `allcls_seq_w0p0025` local AUC `0.991131` / 42 valid; lift vs v616 `-0.002349`; finite/nonconstant sidecar; `submit_approved=false`.
- Updated performance table/jsonl, ledger, ranked queue, and docs log. No submission under mid-day policy; next run should late-fill valid source candidates if `<3h` to UTC reset and no verifier-grade candidate appears.
