# Model Data Point — EfficientAT MN10 AudioSet soundscape embedding branch

Timestamp: 2026-05-26 06:59 UTC

## Objective

Train the requested EfficientAT embedding branch as a no-slot model data point for BirdCLEF 2026 hill climbing. The goal is not immediate submission, but a measured landscape point for the AudioSet/general-acoustic branch family.

## Configuration

- Script: `scripts/birdclef_efficientat_soundscape_embedding_train.py`
- Config: `configs/birdclef/efficientat_mn10_audioset_soundscape_nonaves_notrain_nocall_siteS08_ep12_20260526.json`
- Model family: EfficientAT MobileNetV3 `mn10_as`, AudioSet-pretrained
- External source: EfficientAT repo cloned on trainer at `/home/yourslewis/external_models/EfficientAT`
- Public checkpoint: EfficientAT release `mn10_as_mAP_471.pt`, downloaded by EfficientAT loader
- Training data: official BirdCLEF `train_soundscapes_labels.csv` 5s windows
- Scope: non-Aves or no-train labels, 72 labels
- Split: site holdout `S08`
- Rows: 1,478 windows; validation 120 windows
- Embedding shape: `1478 x 960`; AudioSet logits shape: `1478 x 527`
- Head: 2-layer MLP, hidden 256, dropout 0.15, no-call auxiliary weight 0.2
- Epochs: 12

## Results

- Embedding extraction time: 13.30s on CUDA
- Best validation loss: 0.487352 at epoch 5
- Site-holdout macro AUC: 0.488240 over 18 valid scoped classes
- No-train macro AUC: 0.472842 over 17 valid classes
- Non-Aves macro AUC: 0.488240 over 18 valid classes
- No-call auxiliary AUC: invalid on this split; validation target lacked both classes
- Prediction stats: finite/nonconstant; label min 0.0000012, max 0.9987551, mean 0.0475079

## Comparison

Nearest prior AudioSet embedding point:

- PANNs/Cnn14 AudioSet soundscape branch: macro AUC 0.517333, no-train AUC 0.520824 on same site-holdout target family.
- EfficientAT MN10 underperformed PANNs on this branch (`0.488240` vs `0.517333`) and was roughly comparable to the earlier B0 soundscape specialist (`0.48865`).

## Verifier notes

- Rule-safe no-slot training: official train soundscapes plus public AudioSet checkpoint only.
- Output is a 72-label specialist head, not a 234-class competition submission.
- Holdout predictions are finite and nonconstant.
- TorchScript head load/smoke passed on trainer: input `2 x 960` to outputs `[2,72]` and `[2,1]`.

## Decision

No submission and no scale unchanged. This is a useful EfficientAT branch data point, but the current `mn10_as` 5s soundscape embedding head is weaker than PANNs/Cnn14 on the same target/split. If continuing EfficientAT, the next useful ablation is `dymn10_as` or a site-balanced/leave-one-site ensemble of heads, not repeating `mn10_as` unchanged.
