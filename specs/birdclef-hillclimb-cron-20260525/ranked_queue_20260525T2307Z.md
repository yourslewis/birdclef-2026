# BirdCLEF Hill-Climb Queue — 2026-05-25 23:07 UTC

## Live status

- Public LB best: **0.949**; `v616` is still the tied baseline to beat.
- 2026-05-25 UTC slots: **5/5 used**, ~52 minutes to reset.
- Pending late-day exploratory submissions: `v617` `53032516`, `v618` `53032520`, `v619` `53032523`, `v620` `53032524` — all still pending at 23:07 UTC.
- Latest scored: `v616=0.949`, `v615=0.949`, `v614=0.949`, `v613=0.923`, `v612=0.949`, `v611=0.949`.
- Active local/trainer BirdCLEF jobs: none before this run; the new soundscape specialist smoke completed during this run.

## Scout / model-source refresh

Web/model-source refresh re-confirmed the external AudioSet lane:

1. EfficientAT (`fschmid56/EfficientAT`) advertises AudioSet-pretrained efficient CNNs for downstream training and embedding extraction.
2. PANNs/Cnn14 (`qiuqiangkong/audioset_tagging_cnn` / `panns_inference`) exposes Cnn14 AudioSet checkpoints and 2048-d embeddings.
3. Current trainer venv has `torch`, `torchaudio`, `timm`, `sklearn`, but **does not** have `panns_inference`, `tensorflow`, `tensorflow_hub`, or `hear21passt` installed. So a real EfficientAT/PANNs branch needs an asset packaging step before training/inference.

## Ranked queue after this run

| Rank | Candidate | Expected LB potential | Info/data-point value | Current decision |
|---:|---|---:|---:|---|
| 1 | Monitor `v617`-`v620` | Unknown; source-clean exploratory candidates already submitted | High while pending | **WAIT FOR SCORES** before using reset slots |
| 2 | EfficientAT/PANNs AudioSet event/no-call branch | High if packageable; most decorrelated from v616 plateau | Very high | **NEXT**: package one pretrained asset + extract/train shallow head |
| 3 | Broader negative/no-call cache | Medium; current negative aux was too sparse | High | Needs broader mask coverage before another aux smoke |
| 4 | Non-Aves/no-train soundscape specialist | Medium/high hidden-slice relevance but first held-out-site metric weak | High | **DATA POINT TRAINED** this run; no submit |
| 5 | 20s temporal/localmax branch | Medium; tests temporal context | Medium/high | Run after AudioSet or if package blocks |
| 6 | G124/V2S target-design mini-grid | Medium; prior all-row pilot technically good but no blend utility | Medium | Only with changed targets, not unchanged rerun |
| 7 | Alexy sidecar | Low until source/checkpoint access is clean | Medium | Blocked; direct replay already scored 0.923 |

## Model data point trained this run

- Config: `configs/birdclef/soundscape_nonaves_notrain_b0_5s160_siteS08_ep3_20260525.json`
- Script: `scripts/birdclef_soundscape_specialist_train.py`
- Artifact root: `artifacts/soundscape_specialists/soundscape-nonaves-notrain-b0-5s160-siteS08-ep3-20260525/`
- Family: EfficientNet-B0 SED-style classifier trained on official `train_soundscapes` 5s labeled windows, scoped to 72 non-Aves / no-train labels.
- Init: q3/cap80 external-pretrain TorchScript encoder; 352 keys loaded, head skipped.
- Data: 1,478 windows, 72 labels, 5,420 positive cells; site holdout `S08` with 120 validation windows.
- Runtime: 19.46s on CUDA; TorchScript + ONNX exported; ONNX checker OK; CPU TorchScript smoke 0.093s for 2 logmel samples.
- Validation: site-holdout macro AUC `0.4886` over 18 valid scoped classes; no-train AUC `0.4761` over 17 valid classes. Some sonotypes were learnable (`son22=0.988`, `son13=0.944`, `son11=0.910`), but others inverted badly (`son18=0.057`, `son25=0.092`, `son10=0.106`).
- Decision: **comparison-grade data point only**. It proves the soundscape-native/non-Aves pipeline and export path, but the held-out-site score is not submission-grade and output is 72-class specialist only.

## Critic review

- Proceeding to train this branch despite weak immediate submission odds was correct under the new data-point policy: it measures a genuinely different slice (official soundscapes, no-train/non-Aves labels) rather than another v616-family tweak.
- The single `S08` holdout is intentionally harsh and may overstate failure for site-specific sonotypes; however, it is exactly the hidden-risk we need to see. Do **not** interpret the few high sonotype AUCs as approval evidence.
- Next implementation should not scale this exact B0 site-holdout model. Better options: AudioSet embeddings (EfficientAT/PANNs) for these labels, site-balanced/group-DRO training, or multi-site leave-one-site sweeps.

## Verifier decision

- Competition integrity: **ACCEPTED** for no-slot training. Uses only official train metadata, train audio/soundscapes, and public external pretrain already in repo artifacts. No hidden/test labels, no private data, no Kaggle submission.
- Output/schema: not competition-submission format (72 labels), so **not submit-capable** yet.
- Export/runtime: TorchScript and ONNX produced; ONNX checker passed; CPU TorchScript smoke passed.
- Submission decision: **not approved**; slots are capped and candidate is not submission-grade.

## Next exact action

After reset and after `v617`-`v620` score, either:

1. if any of `v617`-`v620` improves, inspect/build a repo-owned confirmer for that family; otherwise
2. package EfficientAT or PANNs/Cnn14 AudioSet weights and train an embedding-head version of the non-Aves/no-train + no-call branch, then audit against v616.
