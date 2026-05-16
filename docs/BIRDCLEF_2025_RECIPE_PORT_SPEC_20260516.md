# BirdCLEF 2025 Top-Team Recipe Port — 2026-05-16

Status: active prep spec  
Anchor: current BirdCLEF 2026 public best remains **0.946** from repo-owned public946 Perch/ProtoSSM + distilled SED rank-blend variants (`v541`, `v542`, `v558`, `v563` ties).  
Purpose: convert external 2025 solution evidence into repo-owned training experiments that can create signal beyond the copied public946 stack.

## Source scan

### Tekkix BirdCLEF+ 2025 top-5 overview

URL: https://tekkix.com/articles/ai/2025/07/birdclef-2025-overview-of-the-competition-a

Most relevant claims:

- Winner: SED models, multi-stage pseudo-labeling, MixUp/StochasticDepth, power scaling, multiple pseudo-label rounds, and separate insect/amphibian pipeline.
- Reported winner progression: baseline `0.872` -> pseudo-labeling `0.898` -> power scaling + pseudo-labeling rounds `0.930` -> separate insects/amphibians `0.933`.
- 2nd place: Xeno Archive / previous BirdCLEF data pretraining, `tf_efficientnetv2_s + RAdam`, `eca_nfnet_l0 + AdamW`, 50 epochs, Focal+BCE, cosine LR, sqrt sample/class balancing, TTA with 2.5s shifts.
- 4th place: SoftAUCLoss plus semi-supervised training; this supports AUC-aware loss as part of a broader recipe, not as a tiny one-off smoke.

### Max Melichov 2025 postmortem

URL: https://medium.com/@maxme006/how-i-climbed-to-the-top-2-in-birdclef-2025-every-failure-every-lesson-and-why-details-matter-273d781a33df

Most relevant claims:

- CNN + public SED blending with Quantile-Mix / rank+mean averaging was the practical breakthrough.
- Prior BirdCLEF data pretraining helped more than many architecture swaps.
- Segment selection and spectrogram details mattered; raw audio / Wav2Vec / GNN and overcomplicated augmentations were poor fits.
- EfficientNet-B0 remained useful for pseudo-labeling even when larger models looked better on a narrower local metric.

### DS@GT / arXiv: Distilling Spectrograms into Tokens

URL: https://arxiv.org/html/2507.08236v1

Most relevant claims:

- Bioacoustic Model Zoo transfer is useful, but CPU inference optimization is mandatory.
- TFLite gave Perch roughly 10x CPU speedup; ONNX/OpenVINO/TFLite export gates should be first-class.
- BirdSetEfficientNetB1 was their strongest zoo baseline, but token skip-gram was mainly an efficiency experiment, not a top-LB recipe.

## Lessons from our 2026 attempts

- Direct postprocess/local-gate lifts are rejection filters only: `v560` had positive local gates and still dropped to `0.945`.
- Weak sidecars hurt: ConvNeXt 7.5% dropped (`v564=0.942`, `v565=0.943`).
- Tiny random-init model-zoo smokes mostly failed or produced negligible blend value: MobileNetV3-small, RegNetY-002, TF-EfficientNet-Lite0, ECA-ResNet33TS, soft-anchor and soft-AUC ResNet18 variants are not slot-worthy.
- Plain ResNet18 is the only recent model-zoo diagnostic with borderline standalone signal, but its local blend lift is too small for a slot.

## Immediate implementation gap closed

`birdclef_pseudolabel_student_train.py` now supports 2025-style knobs:

- `loss_name="focal_bce"` for BCE + soft-label focal BCE.
- `focal_gamma` and `focal_loss_weight`.
- `class_weight_mode="sqrt_inv_prevalence"` or `"inv_prevalence"` with clipping and mean normalization.

These make the trainer capable of expressing the 2nd-place-style `Focal+BCE + sqrt balancing` recipe against the public946 teacher cache.

## Next experiments when GPU is free

Run smoke before scaling. Do **not** spend Kaggle slots directly from these unless a full diagnostic produces a real blend or packaging signal.

### 1. eca_nfnet_l0 focal+BCE sqrt-class-weight smoke

Config prepared:

- `configs/birdclef/pl_public946_sed85_rankblend15_nfnetl0_focalbce_sqrtcw_5s_m160_lr1e4_ep8_smoke_20260516.json`

Why:

- eca_nfnet_l0 appears in 2025 top-team recipes.
- It should be more plausible than tiny random-init backbones if trained with the same loss/balancing recipe.

Smoke gate:

- final all-row AUC should exceed plain weak smokes and preferably approach/beat ResNet18 smoke (`0.9426` ep8) without high teacher correlation.
- If standalone is lower but correlation is meaningfully low, run a full aligned blend audit before killing.

### 2. EfficientNetV2-S focal+BCE sqrt-class-weight smoke

Config prepared:

- `configs/birdclef/pl_public946_sed85_rankblend15_effv2s_focalbce_sqrtcw_5s_m160_lr3e4_ep8_smoke_20260516.json`

Why:

- `tf_efficientnetv2_s` is explicitly cited in 2025 strong recipes and is available in the trainer timm environment.
- Prior V2S direct sidecar failed as a low-weight public946 blend, but not under this broader Focal+BCE + balancing setup.

Smoke gate:

- require either competitive student AUC or a materially better diversity profile than the previous V2S/ConvNeXt sidecars.

## Medium-term lane

If either smoke passes:

1. Scale to ep20/full-row diagnostic.
2. Run aligned student pool blend audit against public946 teacher.
3. If packaging looks plausible, export ONNX/OpenVINO/TorchScript and estimate Kaggle CPU time before submitting.
4. For private edge, extend to external/pretrained data rather than more same-teacher micro-sweeps.

If both fail:

- Stop random-init student sidecars and move to external-data/pretraining infrastructure or taxon-specific specialist prep.
