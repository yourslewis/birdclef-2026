# Model Data Point — EfficientAT DyMN10 AudioSet soundscape embedding branch

Timestamp: 2026-05-26 08:20 UTC

## Objective
Train the next EfficientAT AudioSet ablation as a no-slot BirdCLEF 2026 landscape data point, following the user correction to train distinct model branches even when they are not immediately submission-grade.

## Configuration
- Script: `scripts/birdclef_efficientat_soundscape_embedding_train.py`
- Config: `configs/birdclef/efficientat_dymn10_audioset_soundscape_nonaves_notrain_nocall_siteS08_ep12_20260526.json`
- Model family: EfficientAT DyMN `dymn10_as`, AudioSet-pretrained
- External source/checkpoint: public EfficientAT release `dymn10_as.pt`, downloaded by the EfficientAT loader on trainer
- Training data: official BirdCLEF `train_soundscapes_labels.csv` 5s windows only
- Labels/targets: 72 non-Aves or no-train labels plus no-call auxiliary target
- Input window: 5.0s audio, 32 kHz, EfficientAT 128-mel preprocessing
- Split: site holdout `S08`
- Rows: 1,478 windows; 1,358 train / 120 validation; 5,420 positive target cells; 30 no-call auxiliary-positive rows
- Head/loss: 2-layer MLP embedding head, hidden 256, dropout 0.15, BCE label loss plus no-call auxiliary BCE weight 0.2
- Epochs: 12; AdamW lr 0.001, weight decay 0.0001
- Runtime: embedding extraction 36.23s CUDA; full train log at `logs/efficientat_dymn10_audioset_soundscape_nonaves_notrain_nocall_siteS08_ep12_20260526.log`

## Results
- Best validation loss: `0.428341` at epoch 5
- Site-holdout macro AUC: `0.568586` over 18 valid scoped classes
- No-train macro AUC: `0.553327`
- Non-Aves macro AUC: `0.568586`
- No-call auxiliary AUC: invalid on S08 because the validation target lacks both classes
- Prediction stats: finite/nonconstant; label min `0.00000221`, max `0.99864215`, mean `0.04641411`; no-call max `0.917548`
- Artifacts: `artifacts/efficientat_soundscape_embeddings/efficientat-dymn10-audioset-soundscape-nonaves-notrain-nocall-siteS08-ep12-20260526/`

## Comparison / diversity value
- EfficientAT DyMN10 materially improved over EfficientAT MN10 on the same contract: `0.568586` vs `0.488240` S08 macro AUC.
- DyMN10 also beat PANNs/Cnn14 on this same S08 soundscape target: `0.568586` vs `0.517333`, with no-train AUC `0.553327` vs PANNs `0.520824`.
- This keeps the EfficientAT/AudioSet family alive as a rare-slice/non-Aves sidecar candidate, but the branch remains a 72-label specialist, not a 234-class competition output.

## Critic decision
**PROCEED only to multi-site evaluation / wrapper design; no Kaggle slot.** The improvement over MN10/PANNs is meaningful for the measured landscape, but S08 still lacks valid no-call AUC and the output is not submission-format. Do not spend a slot until a 234-class hidden-safe wrapper passes alignment, finite/nonconstant, duplicate, and v616-sidecar audit gates.

## Verifier notes
- Rule-safe inputs: official train soundscapes plus public EfficientAT AudioSet checkpoint; no hidden/test labels or disallowed private data.
- Holdout predictions passed finite/nonconstant checks: shape `120 x 72`, std `0.05203`.
- TorchScript head smoke passed on trainer: input `2 x 960` produced label logits `2 x 72` and no-call logits `2 x 1`.
- Export/runtime status: embedding head `.pt` and TorchScript saved; EfficientAT backbone packaging for Kaggle hidden inference is not yet implemented.

## Next exact action
Build a leave-one-site / site-balanced evaluation for AudioSet heads (DyMN10 vs PANNs vs MN10) and only then decide whether to wrap DyMN10 as a bounded 234-class sidecar; otherwise pivot to the G124 hard-confidence/power ablation.
