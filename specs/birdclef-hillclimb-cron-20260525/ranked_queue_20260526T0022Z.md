# BirdCLEF Hill-Climb Queue — 2026-05-26 00:22 UTC

## Live status

- Public LB best remains **0.949**; `v616` is still the tied baseline to beat.
- 2026-05-26 UTC slots used: **0/5**, ~23.7h to reset at live check.
- Late-day exploratory submissions from 2026-05-25 all completed:
  - `v617` Nina EoS7 sidecar source: `0.949` tie.
  - `v618` Kruzzcc Nina EoS4 BirdNET source: `0.946` drop.
  - `v619` Kruzzcc Mtoshi UMAP BirdNET source: `0.944` drop.
  - `v620` Kazuhiro Karnak rank fusion source: `0.949` tie.
- Latest scored control remains `v616=0.949`; no active local/trainer BirdCLEF jobs before/after this run.
- Slot decision: **no submission this early UTC run** because no verifier-grade or high-info valid candidate was ready; duplicates/replays of v617/v620/v616 are forbidden.

## Scout / model-source refresh

- Kaggle public kernel scan artifact: `artifacts/public_kernels_20260526_scout/scan_20260526T0020Z.json`.
- Recent public competition scan still shows the already-rejected/known recent cluster first: `WildSound-V8` (previously ERROR), `viktoriiahranadzer/birdclef-inference` (previously bad values), `tulayppppp/my-efficientnet-b0-weights` (previously mock/wrong-shape), plus training/EDA/baseline notebooks.
- Search `949` still surfaces Pilkwang/Nina/Jungchan/EoS/Prior Field plateau-family sources, not a new clean >0.949 clue. Search `950` and PANNs/EfficientAT-specific query returned no useful new public notebook lead.
- External AudioSet lane became packageable this run: installed `panns-inference==0.1.1` in trainer venv and downloaded Cnn14 AudioSet checkpoint to `/home/yourslewis/panns_data/Cnn14_mAP=0.431.pth`.

## Ranked queue after this run

| Rank | Candidate | Expected LB potential | Info/data-point value | Current decision |
|---:|---|---:|---:|---|
| 1 | PANNs/Cnn14 AudioSet event/no-call branch | Medium/high if wrapped into 234-class sidecar; decorrelated from v616 by external AudioSet pretrain | Very high | **DATA POINT TRAINED**; keep as landscape point, next needs stronger validation/wrapping |
| 2 | Broader negative/no-call cache | Medium; current negative aux was too sparse | High | Build broader mask coverage before another aux smoke |
| 3 | 20s temporal/localmax branch | Medium; tests temporal context not covered by v616 | Medium/high | Next distinct repo-owned model smoke if AudioSet branch stalls |
| 4 | Non-Aves/no-train soundscape specialist | Medium/high hidden-slice relevance but first B0 and PANNs held-out-site metrics weak | High | Two data points measured; do not scale unchanged |
| 5 | G124/V2S target-design mini-grid | Medium; prior all-row pilot technically good but low blend utility | Medium | Only with changed targets, not unchanged rerun |
| 6 | Fresh source-clean public candidates | Unknown; currently no new >0.949 clue from scan | Medium | Rescan later; do not duplicate v617/v620/v616 |
| 7 | Alexy sidecar | Low until source/checkpoint access is clean | Medium | Blocked; direct replay already scored 0.923 |

## Model data point trained this run

- Config: `configs/birdclef/panns_cnn14_audioset_soundscape_nonaves_notrain_nocall_siteS08_ep12_20260526.json`
- Script: `scripts/birdclef_panns_soundscape_embedding_train.py`
- Artifact root: `artifacts/panns_soundscape_embeddings/panns-cnn14-audioset-soundscape-nonaves-notrain-nocall-siteS08-ep12-20260526/`
- Family: PANNs/Cnn14 AudioSet-pretrained 2048-d embeddings plus small MLP multilabel head.
- Data/targets: 1,478 official `train_soundscapes` 5s windows, 72 non-Aves/no-train labels, plus no-scoped-label/no-call auxiliary target (30 no-call rows).
- Runtime: PANNs checkpoint download ~327 MB; embedding extraction 49.84s on CUDA; MLP head 12 epochs, best val loss `0.45604` at epoch 5.
- Validation: site-holdout `S08` with 120 validation windows; macro AUC `0.517333` over 18 valid scoped classes; no-train macro AUC `0.520824` over 17 classes; no-call aux AUC not valid on S08 because no-call target had a single class there.
- Export/runtime: `embedding_head_torchscript.pt` loads on trainer CPU and produces `(2,72)` label logits plus `(2,1)` no-call logits from `(2,2048)` embeddings. Holdout predictions are finite/nonconstant and aligned.
- Decision: **comparison-grade data point only**. It is more semantically distinct than the B0 soundscape specialist and slightly better on harsh S08 holdout (`0.517` vs `0.489`), but still not submission-grade and not yet a 234-class row-aligned competition output.

## Critic review

- The run correctly avoided spending early-day slots on duplicate/tied plateau sources after v617/v620 tied and v618/v619 dropped.
- Training PANNs anyway was the right data-point move: it measured the highest-ranked external AudioSet family instead of polishing v616-family recipes.
- However, the single-site S08 holdout remains harsh and noisy; a `0.517` macro AUC is not enough to scale or submit. The branch should only continue if the next step tests leave-one-site robustness or wraps it as a tiny capped sidecar and proves movement vs v616 in a no-slot audit.
- No-call auxiliary was under-informative in this split because S08 had no positive/negative variety for that target; future no-call work needs a split with valid no-call positives in validation.

## Verifier decision

- Competition integrity: **ACCEPTED** for no-slot training. Uses official train soundscapes/metadata plus public AudioSet PANNs checkpoint; no hidden/test labels, no private data, no Kaggle submission.
- Output/schema: **not submit-capable** yet; 72-label specialist head only and no hidden/test row-aligned 234-class output.
- Artifact checks: holdout predictions finite/nonconstant, 120×72; labels match metrics; TorchScript head smoke passed.
- Submission decision: **not approved** this run.

## Next exact action

Run a distinct next smoke rather than replaying this branch unchanged:

1. build a broader negative/no-call mask with much higher row coverage and retry aux loss, or
2. run the 20s temporal/localmax branch, or
3. wrap PANNs embeddings into a 234-class capped sidecar only after a split with valid no-call and leave-one-site evidence.
