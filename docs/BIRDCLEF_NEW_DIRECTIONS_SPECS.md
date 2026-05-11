# BirdCLEF 2026 New Directions Spec Pack

Status: draft v1  
Anchor: current stable public LB best = **0.927** from `v268`, `v367`, `v368`, and `v501`-`v504`  
Problem: the current inference/post-processing family is saturated around 0.926-0.927. The next meaningful jump likely requires new model signal, not more single-parameter post-process sweeps.


## 2026-05-11 Public Approach Reprioritization

A new public/shared approach scan found a much stronger public frontier around **0.946 LB**, centered on public distilled SED + Perch/ProtoSSM rank blending.  Treat the older 0.927 plateau language in this document as stale.

Read the addendum before choosing new work:

- `docs/BIRDCLEF_PUBLIC_APPROACH_REEVAL_20260511.md`

Updated priority order:

1. Reproduce/port a robust public **0.946 Perch/ProtoSSM + distilled SED rank-blend** baseline as a controlled repo-owned Kaggle kernel.
2. Use that reproduced 0.946 stack as the new teacher cache for pseudo-label/noisy-student training.
3. Blend our internal v517/v537/v538 streams into the public946 anchor as minority streams, not the reverse.
4. Evaluate CLAP/BirdNET/custom EffNet forks only after the core public946 baseline lands.
5. Continue OOF-teacher B0 sidecars only as queued diagnostics unless they tie/improve or blend well with public946.

---

## Executive Summary

The next set of work should split into two lanes:

1. **High-upside training lanes** that can plausibly move the leaderboard by multiple points:
   - real SED/frame-event models
   - pseudo-label/noisy-student rounds
   - external-data pretraining/fine-tuning
   - model zoo diversity beyond the current ProtoSSM/Perch axis

2. **Fast tuning lanes** that should be rerun only after new model predictions exist:
   - ensemble weights
   - Quantile-Mix alpha
   - power/gamma scaling
   - file-context alpha
   - temporal smoothing
   - top-k contrast / tail damping

The current v500-v508 family is useful as a strong teacher and inference baseline, but it should no longer be the only search axis.

---

## Global Experiment Rules

### Evaluation gates

Every trained model or pseudo-label round must produce these artifacts:

- `oof_predictions.parquet` or `.npz`
- `test_predictions.parquet` or `.npz` where possible
- per-fold macro ROC-AUC
- per-class AUC summary with rare-class diagnostics
- inference timing estimate on Kaggle CPU
- export artifact size estimate
- config snapshot
- training log
- seed/fold metadata

### Success criteria

A direction is worth continuing if at least one is true:

- OOF macro AUC improves by a meaningful margin over the comparable baseline.
- Blend with current best predictions improves OOF blend even if standalone is weaker.
- Prediction correlation with current best is low while AUC is competitive.
- Public LB improves beyond 0.927 or creates a safe 0.927 tie with genuinely new signal.

### Kill criteria

Stop a lane early if:

- OOF is worse and prediction correlation with current best is high.
- Kaggle inference cannot fit CPU/time constraints after reasonable optimization.
- Pseudo-label confidence filtering produces unstable class distributions.
- External-data training improves common species but harms rare/zero-shot species.

### Common knobs to always record

- backbone
- crop length
- hop length / overlap
- mel resolution
- sample rate
- loss
- class balancing strategy
- augmentations
- pseudo-label thresholding strategy
- external-data source mix
- fold seed
- inference TTA
- export type: TorchScript / ONNX / OpenVINO / cached features

---

## Spec A — Real SED Frame/Event Ensemble

### Hypothesis

The current local-event post-processing approximates SED behavior, but it is still built on clip/global predictions. A real frame-level SED model should add new temporal signal and better localize short calls, overlapping species, and weak secondary events.

### Why this can break 0.927

2025 solution notes repeatedly emphasize SED/frame-local event modeling. Our v501-v504 gains suggest that even fake SED-style local propagation helps; a real SED model should provide less-correlated signal.

### Data

- primary 2026 train audio
- optional external target-species audio after Spec C data cleaning
- generate frame labels from clip labels with weak-label MIL pooling
- optionally use high-confidence teacher events from current best ensemble as soft frame labels

### Model recipe

- Input: log-mel spectrogram, 32 kHz or 48 kHz source resampled consistently
- Segment lengths:
  - 5s crop for direct competition alignment
  - 10s crop for context
  - 30s crop for whole-recording context when memory allows
- Backbone candidates:
  - EfficientNet-B0 SED
  - EfficientNet-B3 SED
  - EfficientNetV2-S SED
  - eca_nfnet_l0 SED
- Neck:
  - framewise feature map -> BiGRU or lightweight Transformer encoder optional
  - attention pooling + max pooling + mean pooling heads
- Heads:
  - clip-level head for macro AUC
  - frame-level/event head for temporal aggregation
- Pooling variants:
  - attention pooling
  - linear softmax pooling
  - max + mean concat
  - noisy-or MIL pooling

### Initial hyperparameter grid

- crop length: `5s`, `10s`, `20s`
- mel bins: `128`, `160`, `224`
- hop size: `320`, `512`
- loss:
  - BCEWithLogits
  - Focal BCE: gamma `1.0`, `1.5`, `2.0`
  - BCE + class-balanced weights
- label smoothing: `0.0`, `0.005`, `0.01`, `0.02`
- mixup alpha: `0.0`, `0.2`, `0.4`
- specaugment:
  - freq masks: `1`, `2`
  - time masks: `1`, `2`, `4`
  - severity: light / medium
- optimizer:
  - AdamW lr `1e-4`, `3e-4`
  - RAdam lr `1e-3` for NFNet-like configs
- schedule:
  - cosine
  - cosine with warmup 5%
- epochs:
  - smoke: 2 epochs / 1 fold
  - pilot: 12-20 epochs / 3 folds
  - full: 40-50 epochs / 5 folds
- EMA: off/on with decay `0.999`
- SWA: off/on for final 20-30% epochs

### First experiments

- `sed-b0-5s-attn-v1`: EfficientNet-B0, 5s crops, attention pooling, BCE, light SpecAugment.
- `sed-b0-10s-attn-v2`: same but 10s crops.
- `sed-v2s-5s-focal-v1`: EfficientNetV2-S, Focal BCE, equal class balancing.
- `sed-nfnet-l0-5s-focal-v1`: eca_nfnet_l0, focal loss, sqrt class balancing.
- `sed-b3-10s-mil-v1`: EfficientNet-B3, noisy-or MIL pooling.

### Inference/export

- Preferred: export to ONNX or OpenVINO for Kaggle CPU.
- Fallback: precompute fold predictions offline and submit via Kaggle dataset if allowed by competition rules and code constraints.
- Kaggle inference should produce both:
  - clip-level predictions
  - frame/event-smoothed predictions aggregated to 5s rows

### Blend plan

- Blend SED predictions with current v504/v508 axis.
- Start with weights:
  - current_best: `0.70`, SED: `0.30`
  - current_best: `0.50`, SED: `0.50`
  - current_best: `0.30`, SED: `0.70`
- Then retune Quantile-Mix/gamma/context on the blended output.

---

## Spec B — Pseudo-Label / Noisy Student Rounds

### Hypothesis

The current best ensemble is strong enough to teach new CNN/SED students on unlabeled or external audio. Pseudo-labeling was a major jump in 2025-style solutions, especially when combined with confidence filtering, power scaling, and iterative rounds.

### Data sources

- current competition train audio
- competition test-style unlabeled audio if available and rules allow
- Xeno-Canto/iNat/prior BirdCLEF data after filtering by target species
- any existing cached Perch/ProtoSSM predictions already in the repo/datasets

### Teacher

Use a stable ensemble, not a single submission:

- base teacher: v504/v503/v502/v501 average
- optional include v367/v368/v268 if raw predictions are available
- apply current safe postprocessing:
  - immediate smoothing
  - file-context alpha around `0.25`
  - gamma around `0.825`
  - Quantile-Mix alpha around `0.50`

### Pseudo-label generation

Create two label sets:

1. **Soft labels**
   - keep full probability vector
   - temperature/power scaled variants
   - better for multi-label ambiguity

2. **Hard confident labels**
   - positives: class prob > `0.90`, `0.95`, `0.98`
   - negatives: class prob < `0.01`, `0.02`, `0.05`
   - per-class cap to avoid common-species domination

### Power/temperature variants

- power scale teacher probabilities: `0.75`, `0.85`, `1.0`, `1.15`, `1.30`
- temperature logits: `0.75`, `1.0`, `1.25`
- top-k positive retention: `1`, `3`, `5`, all above threshold

### Student training recipe

- Start with simple robust CNN/SED students:
  - EfficientNet-B0 SED
  - EfficientNetV2-S clip/SED
  - eca_nfnet_l0 clip/SED
- Loss:
  - soft-label BCE
  - hard-label BCE
  - mixed: `0.7 * hard + 0.3 * soft`
- Sampling:
  - real labeled examples always present
  - pseudo-labeled examples sampled at ratios `0.25`, `0.50`, `1.00` vs real data
  - rare species oversampling if pseudo-label counts are too low

### Iteration plan

- Round 0: current best teacher produces pseudo-label cache.
- Round 1: train 2-3 student models with conservative pseudo-labels.
- Round 2: blend teacher + best students; regenerate pseudo-labels with more data or slightly lower thresholds.
- Round 3: only if Round 2 OOF/blend improves.

### First experiments

- `pl-r1-b0-soft-p095`: EfficientNet-B0 SED, soft labels + positive threshold 0.95 cache.
- `pl-r1-v2s-hard-p098`: EfficientNetV2-S, hard positives above 0.98, pseudo ratio 0.5.
- `pl-r1-nfnet-soft-p090`: NFNet-L0, softer threshold, pseudo ratio 0.25.
- `pl-r1-b0-power085`: B0 with teacher probabilities power-scaled by 0.85.
- `pl-r1-b0-power115`: B0 with teacher probabilities power-scaled by 1.15.

### Validation guardrails

- Track pseudo-label class histogram vs train histogram.
- Reject runs where common species explode and rare species disappear.
- Require OOF blend improvement or low-correlation model value.
- Keep a no-pseudo baseline for every architecture.

---

## Spec C — External-Data Pretraining on Target Species

### Hypothesis

The model stack is likely underexposed to biological/audio diversity. External target-species pretraining can improve recognition robustness and rare-class coverage before fine-tuning on 2026.

### Sources

Prioritize sources that match target taxonomy and audio domain:

- Xeno-Canto target-species audio
- iNaturalist audio if available/licensed
- prior BirdCLEF train sets with overlapping species or close taxonomy
- CSA / regional wildlife datasets if already accessible
- Perch/BirdNET embeddings as auxiliary metadata if raw audio is hard to use

### Data cleaning

- map species names to competition taxonomy
- reject ambiguous or unmapped taxonomy unless manually reviewed
- deduplicate by recording ID, uploader, and audio fingerprint where possible
- remove obvious leaks if any overlap with validation/test sources is suspected
- clip long files into 5s/10s windows
- generate background/no-call windows from low-confidence regions

### Pretraining recipe

- Stage 1: train on external target-species audio.
- Stage 2: fine-tune on BirdCLEF 2026 train folds.
- Stage 3: optionally add pseudo-labels from Spec B.

### Backbones

- EfficientNet-B0 for fast iteration
- EfficientNet-B3/B4 for stronger final model
- EfficientNetV2-S pretrained on ImageNet21k if available
- eca_nfnet_l0 for less-correlated representation
- RegNetY-008/016 if implementation/export is straightforward

### Hyperparameter grid

- external/competition sampling ratio during fine-tune:
  - `0.0`, `0.25`, `0.50`, `1.0`
- freeze backbone warmup:
  - none
  - freeze first 1-3 epochs
- LR:
  - pretrain `1e-3`, `3e-4`
  - fine-tune `1e-4`, `3e-5`
- class balancing:
  - equal
  - sqrt inverse frequency
  - rare species oversample cap `2x`, `4x`
- augmentations:
  - light baseline
  - medium SpecAugment
  - noise mix/background mix

### First experiments

- `xc-b0-pretrain-ft-v1`: B0 pretrain on Xeno-Canto target matches, fine-tune 2026.
- `xc-v2s-pretrain-ft-v1`: EfficientNetV2-S external pretrain, 2026 fine-tune.
- `xc-nfnet-pretrain-ft-v1`: NFNet-L0 external pretrain, focal BCE.
- `xc-b0-rare-oversample-v1`: B0 with rare species oversampling during fine-tune.
- `xc-b0-noise-bg-v1`: B0 with external background/no-call negatives.

### Kill criteria

- If external pretraining improves external validation but harms 2026 OOF, reduce external mix or use external only for initialization.
- If taxonomy mapping noise is high, switch to pseudo-labeling external audio with current best teacher rather than trusting source labels.

---

## Spec D — Model Zoo Diversity Sweep

### Hypothesis

Current gains mostly reweight a limited model family. A diverse model zoo can improve blend rank even when individual models are slightly weaker, especially if prediction correlation is low.

### Candidate families

- EfficientNet-B0/B3/B4
- EfficientNetV2-S / EfficientNetV2-B3
- eca_nfnet_l0
- RegNetY-008 / RegNetY-016
- ConvNeXt-Tiny if export and speed are acceptable
- AST/PaSST only if inference constraints can be solved
- BirdNET/Perch embeddings as side channels rather than full models

### Training variants

For each architecture, run:

- no-pseudo baseline
- pseudo-label Round 1 variant
- external-pretrained variant if data exists
- SED variant for at least the most promising two backbones

### Model selection metrics

- standalone OOF AUC
- blend OOF AUC with current best
- correlation with current best predictions
- rare-class AUC
- file-level calibration quality
- inference cost

### First experiments

- `zoo-b0-baseline-v1`
- `zoo-b3-baseline-v1`
- `zoo-v2s-baseline-v1`
- `zoo-nfnet-l0-baseline-v1`
- `zoo-regnety008-baseline-v1`
- `zoo-v2s-pseudo-v1`
- `zoo-nfnet-pseudo-v1`

### Blend policy

Do not discard a model only because standalone AUC is lower. Keep if:

- it improves OOF blend at any positive weight, or
- correlation is meaningfully lower than existing models and rare-class behavior is better.

---

## Spec E — Non-Bird / Background / Taxon Gate

### Hypothesis

Some BirdCLEF solutions benefited from separate handling of non-bird taxa or background/no-call behavior. A lightweight gate can reduce false positives and improve calibration, especially for insects/amphibians/mammals/background-like clips.

### Targets

- no-call / low-confidence background
- bird vs non-bird if taxonomy supports it
- taxon group heads:
  - bird
  - amphibian
  - mammal
  - insect
  - unknown/background

### Data

- competition labels mapped to taxon group
- low-confidence windows as background candidates
- external background/noise clips if safe
- negative mining from teacher low-confidence regions

### Model recipe

- small EfficientNet-B0 or shallow CNN gate
- input same mel config as main models
- output:
  - group probabilities
  - no-call probability
- integrate as postprocess multiplier:
  - species score *= group probability
  - species score *= `(1 - no_call_prob * alpha)`

### Hyperparameter grid

- gate alpha: `0.10`, `0.20`, `0.30`, `0.50`
- no-call threshold: `0.50`, `0.70`, `0.85`
- group multiplier floor: `0.50`, `0.70`, `0.85`
- train negatives ratio: `0.25`, `0.50`, `1.00`

### First experiments

- `gate-b0-nocall-v1`: no-call/background gate only.
- `gate-b0-taxon-v1`: taxon group gate.
- `gate-post-alpha020-v1`: integrate gate with current best predictions at alpha 0.20.
- `gate-post-floor070-v1`: taxon multiplier floor 0.70.

### Kill criteria

- If macro AUC drops on rare species, reduce alpha/floor aggression.
- If gate mainly suppresses real rare calls, use it only for extreme low-confidence clips.

---

## Spec F — Retune Hyperparameters After New Signals

### Hypothesis

The existing postprocess tuning was done around the current Perch/ProtoSSM axis. Once SED/pseudo/external/zoo models exist, optimal blend and calibration knobs will shift.

### When to run

Only after at least one new model family has OOF/test predictions.

### Sweep dimensions

- model weights:
  - current_best weight: `0.20`-`0.80`
  - SED weight: `0.10`-`0.60`
  - external/zoo weight: `0.10`-`0.50`
- Quantile-Mix alpha:
  - `0.40`, `0.45`, `0.475`, `0.50`, `0.525`, `0.55`, `0.60`
- power gamma:
  - `0.75`, `0.80`, `0.825`, `0.85`, `0.875`, `0.90`, `0.95`
- file-context alpha:
  - `0.15`, `0.20`, `0.225`, `0.25`, `0.275`, `0.30`, `0.35`
- temporal smoothing:
  - none
  - immediate `[0, 0.15, 0.70, 0.15, 0]`
  - sharp `[0.05, 0.10, 0.70, 0.10, 0.05]`
  - broader `[0.10, 0.15, 0.50, 0.15, 0.10]`
- top-k contrast:
  - k: `3`, `5`, `7`
  - boost power: `0.85`, `0.90`, `0.95`
  - damp power: `1.05`, `1.10`, `1.15`

### Optimization strategy

- Stage 1: coordinate descent on OOF, not public LB.
- Stage 2: low-dimensional Bayesian/random search over the top 4-6 knobs.
- Stage 3: submit only 3-5 diverse candidates per day, not near-duplicates.

### First experiments after new predictions

- `blend-sed-current-grid-v1`
- `blend-zoo-current-grid-v1`
- `blend-sed-zoo-current-grid-v1`
- `post-newmodels-gamma-grid-v1`
- `post-newmodels-context-grid-v1`

---

## Spec G — Inference Packaging / Export Track

### Hypothesis

Training wins are useless if the model cannot run under Kaggle code competition constraints. Export/inference should be treated as a first-class workstream, not postponed until the end.

### Requirements

- CPU-compatible inference
- no internet
- deterministic output
- controlled package dependencies
- model artifact size fits Kaggle dataset limits
- full inference within competition time budget

### Export options

1. ONNX Runtime
   - likely best for PyTorch CNN/SED models
   - validate ops early

2. OpenVINO
   - potentially fastest CPU inference
   - may require conversion testing per architecture

3. TorchScript
   - simplest but may be slower

4. Feature cache approach
   - if model inference is too slow, precompute features where rules allow and use lightweight probes/blends in kernel

### Export smoke tests

For every new backbone:

- export one fold after 1-epoch smoke train
- run local inference on 3 sample audio files
- run Kaggle-style script locally without internet
- compare PyTorch vs exported output cosine/correlation
- estimate total runtime on test-size proxy

### First experiments

- `export-b0-onnx-smoke-v1`
- `export-v2s-onnx-smoke-v1`
- `export-nfnet-onnx-smoke-v1`
- `export-b0-openvino-smoke-v1`

---

## Recommended First Ten Work Items

1. Build a local training scaffold that emits OOF/test prediction artifacts in a consistent format.
2. Train `sed-b0-5s-attn-v1` as a 1-fold smoke run.
3. Export `sed-b0-5s-attn-v1` to ONNX and test Kaggle-style inference.
4. Train `zoo-b0-baseline-v1` and `zoo-v2s-baseline-v1` no-pseudo baselines.
5. Generate teacher pseudo-label cache from v501-v504 average.
6. Train `pl-r1-b0-soft-p095`.
7. Start Xeno-Canto target-species metadata collection and taxonomy mapping.
8. Train `sed-v2s-5s-focal-v1` if B0 export works.
9. Run OOF blend grid for current best + first SED + first zoo model.
10. Submit only the best 3-5 diverse candidates from the blend grid.

---

## Cron Update Recommendation

The autonomous cron should stop creating only micro postprocess variants. Recommended cron objective:

1. Check latest LB and queued kernel state.
2. If submissions are capped, do training/spec work instead of only waiting.
3. Prioritize work in this order:
   - SED smoke/export
   - pseudo-label cache
   - model zoo baseline
   - external-data metadata mapping
   - postprocess retune only after new predictions exist
4. Keep daily Kaggle submissions to diverse candidates, not adjacent single-knob variants.

Suggested cron prompt fragment:

```text
Objective: Improve BirdCLEF 2026 beyond the current 0.927 plateau.
Use docs/BIRDCLEF_NEW_DIRECTIONS_SPECS.md as the active spec. Prioritize new model signal over post-processing micro-tweaks: SED training/export, pseudo-label cache + students, external-data pretraining, and model-zoo diversity. Only run postprocess/grid tuning after a new prediction artifact exists. Always check latest submissions, diagnose failures, avoid duplicate queued variants, and submit only diverse candidates under the daily cap.
```

---

## Immediate Decision

Recommended next implementation target: **Spec A + Spec G smoke path**.

Reason: a real SED B0 smoke model plus ONNX/Kaggle-style export tells us quickly whether the highest-upside direction is operationally feasible. If export works, we can scale SED and pseudo-label students. If export fails, we pivot to cached-feature/model-zoo approaches before spending more training time.
