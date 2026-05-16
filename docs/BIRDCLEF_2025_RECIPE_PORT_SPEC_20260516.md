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


## Ranked candidate slate

These are ordered by expected information value, not just ease.  The first two are already prepared as runnable smokes; the later candidates are the next spec-to-code targets if the smoke results or GPU availability justify them.

### Candidate A — NFNetL0 focal/BCE noisy student from public946 teacher

**Hypothesis:** `eca_nfnet_l0` was repeatedly useful in 2025 top-team recipes.  Our earlier random-init NFNet-ish attempts were weak because they did not use the full recipe: focal+BCE, sqrt class balancing, and careful public946 teacher targets.  A proper NFNetL0 student may add less-correlated CNN/SED-ish signal to the Perch/ProtoSSM + distilled-SED anchor.

**Prepared config:**

- `configs/birdclef/pl_public946_sed85_rankblend15_nfnetl0_focalbce_sqrtcw_5s_m160_lr1e4_ep8_smoke_20260516.json`

**Initial recipe:**

- teacher: `public946-v540-teacher-cache66-v1/teacher_sed85_rankblend15.npz`
- backbone: `eca_nfnet_l0`
- target: soft labels, teacher power `1.0`
- loss: BCE + soft focal BCE, `focal_gamma=1.5`, `focal_loss_weight=1.0`
- class weights: `sqrt_inv_prevalence`, clip `5.0`
- crop: 5s, 160 mel, hop 512
- optimizer: AdamW, lr `1e-4`, weight decay `1e-4`
- smoke: 256 train-soundscape rows, 8 epochs, best-val restore

**Smoke gate:**

- pass if final AUC approaches/exceeds ResNet18 ep8 smoke (`0.9426`) with lower or comparable teacher correlation;
- conditional pass if standalone is lower but correlation is materially lower and val trajectory is still rising;
- fail if it behaves like ECA-ResNet33TS/RegNetY (`~0.91`) or creates a huge artifact/runtime problem.

**Scale plan if pass:**

1. full-row ep20 diagnostic with same recipe;
2. aligned student-pool blend audit against public946 teacher;
3. if blend lift is nontrivial, package as a tiny rank sidecar at `0.5%`, `1%`, and `2%` offline, but submit at most one variant.

**Submission bar:** no direct slot from smoke.  Needs full-row aligned blend evidence or a clear standalone improvement profile.

**Kill rule:** if full diagnostic blend lift is in MobileNetV3 territory (`~+0.000003`) or public946 correlation is too high, demote to training infrastructure only.

### Candidate B — EfficientNetV2-S focal/BCE noisy student from public946 teacher

**Hypothesis:** 2025 2nd-place evidence specifically cited `tf_efficientnetv2_s + RAdam` with Focal+BCE and balancing.  Our prior V2S low-weight sidecar failed (`v560=0.945`), but that was a direct sidecar result, not a full top-team-style distillation recipe.

**Prepared config:**

- `configs/birdclef/pl_public946_sed85_rankblend15_effv2s_focalbce_sqrtcw_5s_m160_lr3e4_ep8_smoke_20260516.json`

**Initial recipe:**

- teacher: same public946 rankblend teacher cache;
- backbone: `tf_efficientnetv2_s`;
- target/loss/class weights: same Focal+BCE + sqrt balancing recipe as Candidate A;
- optimizer currently AdamW lr `3e-4` for compatibility with trainer; if smoke is unstable, add RAdam support or lower lr before judging.

**Smoke gate:**

- pass if it beats prior weak V2S/ConvNeXt behavior and produces competitive AUC with acceptable runtime;
- conditional pass if it is slower but clearly more accurate/diverse than ResNet18;
- fail if it repeats Lite0/ConvNeXt-style failure or high-correlation weak sidecar behavior.

**Scale plan if pass:**

1. ep20 full-row diagnostic;
2. compare targets: rankblend teacher vs SED-only teacher if available;
3. package only after CPU export/time check, because V2-S may be runtime-riskier than NFNetL0.

**Submission bar:** stronger than Candidate A because prior V2S public sidecar already dropped; needs either clear blend evidence or a source-clean package that is genuinely distinct from v560.

### Candidate C — External/pretrained 2025-style CNN init, then public946 distillation

**Hypothesis:** The largest non-public-copy gains in 2025 came from previous BirdCLEF/Xeno Archive pretraining before pseudo-labeling.  Same-teacher random-init sidecars are saturated; a source-clean external/pretrained init is more likely to shift the representation enough to matter.

**Data/source plan:**

- start with existing accessible train-audio/external manifests already used in prior Spec C pilots;
- include prior BirdCLEF / Xeno-canto style audio only if source-clean and Kaggle-packageable;
- cap per class to avoid common-species domination;
- keep labels aligned to 2026 target taxonomy; log unmapped/ambiguous classes.

**Training recipe:**

1. supervised/external pretrain on mapped target species:
   - B0 as a fast sanity baseline, then NFNetL0 or EffV2-S;
   - Focal+BCE, sqrt/equal balancing, middle/random 5s crop for rare classes;
2. load checkpoint with `initial_checkpoint`, `initial_load_head=false`;
3. public946 soft-label distillation on train soundscapes;
4. optional second pseudo-label round with teacher power `0.85` / `1.15`.

**Implementation gaps:**

- trainer already supports supervised clip mixing and partial checkpoint load;
- may need a manifest audit script that reports class coverage, missing files, and per-class caps before training;
- may need RAdam/cosine support for closer 2025 parity.

**Smoke gate:**

- pretrain smoke must load real audio and improve over random-init at same row/epoch budget;
- distillation smoke must beat same-backbone random-init or materially lower teacher correlation without AUC collapse.

**Submission bar:** highest upside but no slot until export/runtime and full-row blend evidence exist.

**Kill rule:** if external/pretrain improves common species but worsens rare/zero-shot/taxon classes in per-class diagnostics, split by taxon or abandon that source mix.

### Candidate D — Non-bird / rare-taxon specialist correction model

**Hypothesis:** 2025 winner gained from a separate insect/amphibian pipeline.  Public946 may already be strong on common birds, while remaining headroom could be in insects/amphibians/mammals/reptiles or rare label groups.  A specialist correction may be safer than perturbing all 234 classes.

**Target scope:**

- build label groups from taxonomy / competition metadata:
  - birds vs Amphibia/Mammalia/Reptilia/insects/ambiguous sonotypes;
  - rare classes with low train prevalence or low public946 confidence;
- do not touch all columns globally; emit a correction matrix only for target classes.

**Model/feature options:**

1. lightweight CNN specialist trained only on target-taxon labels;
2. public946-derived residual correction: learn when public946 undercalls non-bird/rare classes;
3. rule-assisted sonotype/taxon mirroring only if crossfit non-negative.

**Smoke gate:**

- leave-one-file/site crossfit must be non-negative for the correction;
- targeted-class AUC/recall must improve without degrading global macro AUC beyond a tiny tolerance;
- distribution shift vs anchor must stay bounded.

**Submission plan:**

- package as a bounded postprocess on top of v558/v542, with max absolute logit/probability delta;
- submit only one small correction if offline gate is much stronger than failed v560 local lift.

**Kill rule:** if it resembles site/hour prior leakage or only improves in-sample labels, reject even if local macro AUC rises.

### Candidate E — Real SED/MIL frame-local student instead of clip-only sidecar

**Hypothesis:** Current public946 already uses distilled SED output, but our trainable students are mostly clip/global distillation.  A real frame/event or MIL pooling student can learn temporal localization/residual behavior that clip-only sidecars miss.

**Architecture sketch:**

- backbone: B0 first for speed, then NFNetL0/EffV2-S if promising;
- framewise feature map -> attention or max+mean MIL pooling;
- heads:
  - clip-level BCE/Focal+BCE head;
  - optional frame/local head from teacher spike windows or weak MIL targets;
- inputs: 5s and 10s contexts, 160/224 mel.

**Teacher targets:**

- public946 final rankblend for clip head;
- raw SED stream or local window maxima for frame/local head when available;
- power-scaled pseudo-labels for second round.

**Implementation gaps:**

- inspect `birdclef_sed_pilot_train.py` and decide whether to extend it or make a new `birdclef_public946_sed_student_train.py`;
- add metrics that separate clip AUC from frame/local consistency;
- export timing must be checked early.

**Smoke gate:**

- on a 3-5-file or 256-row pilot, verify training runs, output shape matches, and local metrics are not obviously worse than B0/ResNet18;
- if frame/local labels are noisy, require blend improvement rather than standalone approval.

**Submission bar:** no immediate submission.  This is a medium-term candidate that only gets a slot after export/runtime and full-row blend audit.

### Candidate F — Public946 Quantile-Mix / rank+mean ensemble refresh after new artifacts

**Hypothesis:** Max Melichov's writeup and public946 itself both support rank/mean blending, but our current public-output overlays are too small or leaky.  Quantile-Mix becomes worth revisiting only after Candidates A-E produce a genuinely new prediction artifact.

**Recipe:**

- anchor: v558 or v542 public946 output;
- new artifact: NFNetL0/EffV2-S/external/taxon/SED candidate output;
- blend forms:
  - rank-only;
  - mean-only;
  - Quantile-Mix alpha `0.25/0.50/0.75`;
  - bounded class/taxon-specific blend weights.

**Gate:**

- must beat local reject thresholds by more than v560's failed local lift (`+0.000222`) or have independent OOF/source evidence;
- displacement/correlation must be reported;
- no slot for public-public clones without a new artifact.

**Submission plan:**

- at most one candidate per new artifact family;
- preserve two daily slots for follow-up if it scores/ties well.

## Recommended execution order

1. **Candidate A** NFNetL0 focal+BCE sqrt-balancing smoke when GPU is free.
2. **Candidate B** EffV2-S focal+BCE smoke if A is weak/slow or to compare top 2025 backbones.
3. **Candidate C** external/pretrained init if A/B show any life, or immediately if A/B both fail as random-init students.
4. **Candidate D** taxon specialist in parallel as a bounded postprocess/training target, because it attacks a different failure mode.
5. **Candidate E** real SED/MIL student as the medium-term architecture project.
6. **Candidate F** Quantile-Mix only after a new artifact exists.

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
