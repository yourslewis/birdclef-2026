# Coordinator Synthesis — BirdCLEF New Branch Ideation

Date: 2026-05-25

## Coordinator decision

Phase 1 ideation is **ACCEPTED**. It produced a clear direction: stop perturbing the v616/0.949 plateau family and build no-slot smokes for genuinely different hidden-behavior branches.

No Kaggle submission is approved. The next work should produce raw branch outputs and run them through `scripts/birdclef_ensemble_strategy_audit.py` against both the v616 anchor and the already-tied v616 baseline.

## Why we need new branches

The data and validation reports agree on the failure mode:

- Current local proxy is too narrow: about `190` matched rows / `20` files / `6` sites / `42` valid classes.
- Train audio is overwhelmingly Aves-heavy, but soundscape positives are Amphibia/Insecta-heavy.
- `28` submission classes have zero train-audio primary labels: `517063`, `1491113`, `25073`, and `47158son01`–`47158son25`.
- v611/v612/v616 were hidden-safe, locally plausible sidecar blends and all tied `0.949`.
- Therefore the next useful branch must target hidden behavior not captured by v616: no-train/non-Aves, no-call/background, longer temporal context, site/domain robustness, or external/general acoustic embeddings.

## Ranked branch roadmap

### 1. EfficientAT/PANNs AudioSet event/no-call branch — first external/public-model smoke

**Why:** Most likely to be decorrelated from Perch/ProtoSSM/SED/Jung21 plateau. AudioSet models can represent background, no-call, insects, frogs, rain, machinery, and general acoustic events that bird-specialist rank blends may mishandle.

**Public leads:**
- EfficientAT: `https://github.com/fschmid56/EfficientAT`
- PANNs/Cnn14: `https://github.com/qiuqiangkong/audioset_tagging_cnn`
- YAMNet fallback: `https://github.com/tensorflow/models/tree/master/research/audioset/yamnet`

**First smoke:** extract embeddings/logits for train-soundscape windows, train a tiny no-call/non-Aves/residual head, emit `audio_event_raw.csv`, audit as a capped branch versus anchor and v616.

**Gate:** hidden-safe packaging feasibility, nonconstant 234-class output, lift vs v616 not just anchor, site/file/taxon checks, negative controls.

### 2. Non-Aves / no-train soundscape specialist — strongest data-grounded branch

**Why:** Directly targets the biggest data mismatch. The 28 no-train classes and non-Aves soundscape labels are structurally under-modeled by Aves-heavy train-audio approaches.

**Targets:**
- Amphibia/no-train: `517063`, `1491113`, `25073`
- Insect sonotypes: `47158son01`–`47158son25`
- High-count non-Aves: `65380`, `555146`, `22973`, `23158`, `24321`, `66971`
- Mammal/reptile checks: `516975`, `43435`, `116570`

**First smoke:** frozen acoustic embeddings + shallow multilabel heads over soundscape windows; leave-site/file validation; class/taxon-capped raw sidecar only.

**Gate:** target-slice lift transfers across site/file, no broad Aves degradation, no learned boosts for sparse classes without OOF evidence.

### 3. Broader OOF-teacher SED + negative/no-call cache — fastest repo-owned next experiment

**Why:** Existing B0 SED export path works, but current negative cache covered only `26/512` rows. Broadening no-call/negative coverage is the fastest way to test a different hidden behavior while staying repo-owned.

**First smoke:** regenerate broader negative cache from available OOF predictions, train `max_files=1024`, `epochs=4`, small `aux_negative_weight=0.01`, then export/infer and audit raw branch.

**Gate:** AUC materially above `0.8194`, negative coverage far above `5%`, TS/ONNX + CPU inference pass, branch helps vs v616 in audit.

### 4. 20s / local-window pseudo-label branch — temporal-context branch

**Why:** Current v616 family is row/rank centered. Longer context may capture persistent choruses, intermittent calls, and file-level event continuity differently.

**First smoke:** 20s B0 center-localmax pseudo-label smoke with 384 rows / 4 epochs; export/infer; emit raw branch; audit low-weight recipes.

**Gate:** acceptable inference speed, lower correlation/new movement vs v616, leave-file/time-bin stability, positive lift over v616 baseline.

### 5. General/foundation acoustic embeddings — second-wave external branch

**Why:** BEATs, PaSST, AST, HTS-AT, EAT, CLAP/AudioCLIP/AudioMAE can be decorrelated from bird-specialist pipelines, especially for no-train/non-Aves/background behavior.

**Public leads:**
- PaSST: `https://github.com/kkoutini/PaSST`
- AST: `https://github.com/YuanGongND/ast`
- HTS-AT: `https://github.com/RetroCirce/HTS-Audio-Transformer`
- BEATs: `https://github.com/microsoft/unilm/blob/master/beats/README.md`
- LAION CLAP: `https://github.com/LAION-AI/CLAP`

**First smoke:** one encoder only, shallow heads for no-train/non-Aves/background prompts; avoid model-zoo fishing.

**Gate:** public/offline weights, runtime margin, low/moderate correlation plus group-stable slice lift; no submission without verifier.

### 6. G124/V2S target-design mini-grid — hold as operational but not yet useful

**Why:** V2S/G124 path works technically, but previous all-row pilot had microscopic/unstable blend utility. Only worth target-design changes, not unchanged reruns.

**First smoke:** mini-grid over `teacher_power=0.85`, local-max targets, and hard-confidence targets.

**Gate:** useful sidecar weight `>=0.01` with site/file bootstrap positive; otherwise kill.

### 7. Alexy NS1 CNN/noisy-student — conditional on source access

**Why:** CNN/noisy-student branch is genuinely different, but direct v613 scored `0.923` and source/checkpoint access is blocked.

**First smoke:** only source/asset recovery; if available, extract CNN-only raw sidecar with very low capped weights.

**Gate:** no source = stop. Direct final replay remains vetoed.

### 8. WildSound/ConvNeXt offline repair — offline-only, kill early

**Why:** Different acoustic family, but public notebook errors and runtime/export risk are high.

**First smoke:** repo-owned `convnext_tiny` SED one-epoch export/runtime smoke, not in-kernel training.

**Gate:** train/export/CPU infer must pass quickly; otherwise stop.

## Explicit rejects

Reject as slot candidates:

- v616 scalar/rank/temperature/per-class variants.
- SYD52p / p949 clone increments.
- More ProtoSSM/SED/Jung21 recombinations without new raw signal.
- HGNet low-weight repeats.
- PCEN/EoS/visual/RankPower clone replays.
- Public-output-only branch blends without hidden-safe rerun.
- Clean-audio-only CV branches as approval evidence.

## Standard validation path for every new branch

1. Produce raw row-aligned branch CSV, not final submission.
2. Add to copied ensemble audit manifest.
3. Compare against both `anchor_only` and `v616_baseline`.
4. Run site/file bootstrap and leave-one-site/file.
5. Report target slices: non-Aves, no-train classes, time bins, taxon movement, rare/common classes.
6. Run negative controls: shuffled/inverted/class-shuffled sidecars.
7. Only then consider private verifier.
8. Kaggle submission remains blocked unless Coordinator + Verifier approve.

## Recommended immediate next action

Start with **EfficientAT/PANNs AudioSet event/no-call branch smoke** if assets can be fetched/packaged quickly. In parallel or as fallback, run the **broader OOF-teacher negative/no-call cache smoke**, because it uses repo-owned infrastructure and directly tests the same no-call/background hypothesis.

The first implementation task should be a no-slot protocol file for `audio_event_no_call_branch_20260525` with asset verification, embedding extraction, tiny head training, raw branch emission, and ensemble audit integration.
