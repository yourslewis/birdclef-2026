# Public Model / External Scout — BirdCLEF 2026 New Branch Ideation

_Date: 2026-05-25_  
_Role: Public Model / External Scout_  
_Output path: `specs/birdclef-new-branch-ideation-20260525/reports/public_model_scout.md`_

## 1. Executive recommendation

The scout recommendation is to **stop spending effort on more Perch/ProtoSSM/SED/Jung21/EoS/PCEN/HGNet rank/scalar replays**. The repo has now produced multiple hidden-safe, locally plausible sidecar blends (`v611`, `v612`, `v616`) and all tied the `0.949` plateau. `v616` is especially important: it had raw branches with low correlation to the visual anchor, site-bootstrap-positive local lift, and still no public LB lift.

The next useful branches should be raw branch producers whose behavior differs for reasons the current plateau does not cover well:

1. **General AudioSet / sound-event embeddings** (`EfficientAT`, `PANNs/Cnn14`, and, if runtime allows, `PaSST`/`AST`/`HTS-AT`/`BEATs`/`ATST`/`EAT`) for background, no-call, amphibian/insect-like, and non-vocal acoustic events.
2. **Perch2 / SurfPerch / Bioacoustics Model Zoo embeddings**, but only as new residual/few-shot heads, not another Perch+ProtoSSM final replay.
3. **BirdNET 2.4 embeddings/range-aware residuals**, not direct BirdNET score cloning.
4. **CLAP / AudioCLIP / AudioMAE-style semantic or self-supervised embeddings** for text/name/background behavior and non-Aves prompts.
5. **BirdCLEF 2025-style long-context SED/distillation**, because current 5-second EfficientNet-B0 SED smokes were operationally feasible but too weak.

All five are **no-slot candidates first**. The first smoke for any branch should produce row-aligned raw outputs, compare against both the anchor and `v616`, and require site/file/taxon stability before any private verifier. This report does **not** approve a Kaggle submission.

## 2. Public/model leads found with URLs/citations

### BirdNET / BirdNET-Analyzer / `birdnet`

- BirdNET-Analyzer: <https://github.com/birdnet-team/BirdNET-Analyzer>
  - Public analyzer for large acoustic datasets and single audio files.
  - Source code is MIT; official model assets are listed as CC BY-NC-SA 4.0 in the project README, so model-license use needs competition/legal review.
- `birdnet` Python package: <https://github.com/birdnet-team/birdnet>
  - Provides prediction scores and embeddings for `6,522` species, custom species lists, CPU/GPU execution, multiprocessing, and offline local model files.
- Cornell BirdNET Analyzer page: <https://birdnet.cornell.edu/analyzer/>

**Scout interpretation:** direct BirdNET/visual-style forks are partly exhausted by prior public946/visual/BirdNET-like plateau work. The still-interesting path is **BirdNET embeddings plus a residual head** for Aves classes, range/time filtering, or as an embedding-distance detector, not direct score blending.

### Perch / Perch2 / SurfPerch / Bioacoustics Model Zoo

- Google Research Perch: <https://github.com/google-research/perch>
  - Released Perch bird vocalization classifier is available from Kaggle Models.
  - The README cites bird species classification trained on over `10k` species and describes embedding/search/active-learning tooling.
  - It also references SurfPerch, trained on birds, coral reef sounds, and general audio.
- Google Perch Kaggle model: <https://www.kaggle.com/models/google/bird-vocalization-classifier>
- Bioacoustics Model Zoo: <https://github.com/kitzeslab/bioacoustics-model-zoo>
  - Provides a common interface for BirdNET, Perch/Perch2, YAMNet, BirdSet, HawkEars, and related bioacoustic models.
- BirdCLEF 2024 DS@GT working note: <https://arxiv.org/html/2407.06291v1>
  - Used Google Bird Vocalization Classifier, BirdNET, and EnCodec; reports a BirdNET-embedding/pseudo-label approach.
- BirdCLEF 2025 DS@GT working note: <https://arxiv.org/html/2507.08236v1>
  - Reports Bioacoustics Model Zoo baselines; TFLite Perch speedup, BirdSetEfficientNetB1 as the best zoo model among their tested baselines.

**Scout interpretation:** direct Perch+ProtoSSM/SED branches are saturated in this repo (`v614`, `v615`, `v616` lineage). New Perch value must come from **different embedding heads, SurfPerch/general-audio transfer, or no-train/non-Aves few-shot use**.

### PANNs / Cnn14 and efficient AudioSet CNNs

- PANNs / AudioSet Tagging CNN: <https://github.com/qiuqiangkong/audioset_tagging_cnn>
  - Pretrained on AudioSet; README reports Cnn14 checkpoints, 2048-dimensional embeddings, audio tagging, and sound-event-detection variants.
- PANNs inference helper: <https://github.com/qiuqiangkong/panns_inference>
- EfficientAT: <https://github.com/fschmid56/EfficientAT>
  - Publishes AudioSet-pretrained efficient CNNs for downstream training and audio embedding extraction.
  - Explicitly positions models as faster/leaner alternatives to transformer audio taggers.

**Scout interpretation:** this is one of the best immediate external branch families because it is **not bird-specialized** and can supply background/no-call/general-event signals that visual/Perch/ProtoSSM branches may ignore.

### PaSST / AST / HTS-AT / BEATs / ATST / EAT

- PaSST: <https://github.com/kkoutini/PaSST>
  - AudioSet-pretrained Patchout Spectrogram Transformer; search results cite a pretrained `passt_s_swa_p16_128_ap476` model.
- AST: <https://github.com/YuanGongND/ast>
  - Audio Spectrogram Transformer with pretrained inference examples.
- HTS-AT: <https://github.com/RetroCirce/HTS-Audio-Transformer>
  - Hierarchical token-semantic audio transformer; README describes AudioSet and sound-event localization performance.
- BEATs: <https://github.com/microsoft/unilm/blob/master/beats/README.md>
  - Official PyTorch implementation and pretrained/fine-tuned AudioSet checkpoints.
- PretrainedSED / ATST: <https://github.com/fschmid56/PretrainedSED>
  - Pretrained transformer examples for downstream sound event detection.
- EAT: <https://github.com/cwx-worst-one/EAT>
  - Efficient Audio Transformer with AudioSet fine-tuned checkpoints / Hugging Face availability.

**Scout interpretation:** these are strong representation candidates but heavier than PANNs/EfficientAT/YAMNet. Use them first as **offline embedding extractors** or as a single small raw branch, not as a full Kaggle CPU path until runtime is proven.

### CLAP / AudioCLIP / AudioMAE-style embeddings

- LAION CLAP: <https://github.com/LAION-AI/CLAP>
  - Contrastive language-audio pretraining; README describes audio/text latent representations and a PyPI package.
- Microsoft CLAP: <https://github.com/microsoft/CLAP>
  - Provides text/audio embedding extraction and audio-text similarity API examples.
- AudioCLIP: <https://github.com/AndreyGuzhov/AudioCLIP>
- AudioMAE: <https://github.com/facebookresearch/AudioMAE>
  - Masked autoencoder audio representations with AudioSet fine-tuning examples.

**Scout interpretation:** CLAP/AudioCLIP could behave differently because they can score **text prompts** such as species common names, frog/insect descriptors, "rain", "silence", "insect chorus", or "distant bird call". Risk is high: zero-shot semantic labels may be noisy for obscure taxonomy IDs, and runtime/package size must be controlled.

### YAMNet / AudioSet general audio models

- YAMNet TensorFlow Models repo: <https://github.com/tensorflow/models/tree/master/research/audioset/yamnet>
- TensorFlow Hub YAMNet tutorial: <https://www.tensorflow.org/hub/tutorials/yamnet>
  - Search result describes YAMNet as predicting `521` AudioSet event classes with a MobileNetV1 architecture.
- Bioacoustics Model Zoo includes YAMNet support: <https://github.com/kitzeslab/bioacoustics-model-zoo>

**Scout interpretation:** YAMNet is unlikely to directly classify BirdCLEF species, but it is attractive as a **cheap no-call/background/audio-scene sidecar** and for general event priors.

### Bioacoustic no-call/background/event-detection leads

- Weakly supervised BirdCLEF 2021 soundscape detection paper: <https://arxiv.org/pdf/2107.04878>
  - Search result describes robustness against background sounds such as airplanes/rain and BirdCLEF 2021 soundscape classification/detection.
- BirdCLEF 2021 adaptation repo: <https://github.com/fraank/kaggle-birdclef-2021>
  - Search result notes use of background clips where no bird call was found.
- DCASE / Pretrained SED leads:
  - <https://github.com/fschmid56/PretrainedSED>
  - <https://github.com/Audio-WestlakeU/ATST-SED>

**Scout interpretation:** current local labels are positive-heavy and do not measure no-call behavior well. A background/no-call branch can be valuable even if it does not predict classes directly, by suppressing false positives or gating low-confidence windows.

### BirdCLEF 2025 / 2024 public solution ingredients

- BirdCLEF 2025 5th place solution: <https://github.com/myso1987/BirdCLEF-2025-5th-place-solution>
  - Public solution repo describes SED EfficientNet-B0/B3/EfficientNetV2-B3/EfficientNetV2-S models, 30/60-second crops, pseudo-label training, and model conversion assets.
- BirdCLEF 2024 3rd place solution component: <https://github.com/TheoViel/kaggle_birdclef2024>
  - Uses unlabeled soundscapes for pseudo-labeling and distillation; uses varied CNN backbones over log-mel spectrograms.
- BirdCLEF 2024 DS@GT working note: <https://arxiv.org/html/2407.06291v1>
- BirdCLEF 2025 DS@GT working note: <https://arxiv.org/html/2507.08236v1>

**Scout interpretation:** repo smokes show naive 5-second B0 SED is too weak (`0.754` supervised, `0.819` soft OOF teacher). The public-solution ingredient worth trying is **longer context + pseudo-label distillation + stronger/varied backbones**, not scaling the current weak B0 config.

### BirdCLEF 2026 public notebook leads already in the lineage

- Public Perch/Proto/SED examples were visible in search, e.g. Kaggle notebooks such as `BirdCLEF+ 2026 | ONNX + Perch+Proto+SED` and `BirdCLEF+ 2026 — Perch v2 + ProtoSSM · 0.925`.
- Repo artifacts already tested/ported the strongest local lineage: Samejima visual anchor, Jungchan Model21, Raunak/Samejima SED, HGNet sidecars, SYD52p, PCEN/EoS/RankPower variants.

**Scout interpretation:** these are valuable provenance references but mostly **not new branch candidates** unless they expose a raw branch not already duplicated by `v616`/SYD/ProtoSSM/SED outputs.

## 3. Already exhausted vs blocked vs genuinely new

### Already exhausted / not new enough

- **v616 bundle:** Samejima visual anchor + Jung21 raw + Samejima/Raunak SED raw. It is hidden-safe and diverse at raw-branch level, but the final scored `0.949`.
- **Jungchan/Raunak/Samejima/SYD ProtoSSM/SED branch clones:** repo comparison found exact or near-exact duplicates among key raw branch outputs; SYD52p added only microscopic local lift over `v616`.
- **EoS/PCEN/PriorField/RankPower/visual clone families:** several variants tied `0.949` and are considered one saturated plateau family.
- **HGNet low-weight sidecars:** `v611`/`v612` were hidden-safe and tied `0.949`; standalone HGNet was much worse.
- **Per-class/scalar/taxon/rank-power tweaks on v616:** per-class selector had in-sample lift but essentially zero leave-site transfer.
- **Direct Alexy NS1 replay:** `v613` scored `0.923`; source access is currently blocked.
- **Current G124/V2S student outputs:** training/export worked, but useful blend weight/lift was essentially zero and group stability was weak.
- **Current B0 SED smokes:** export path works, model gate is weak (`0.754` supervised, `0.819` soft OOF teacher, `0.819` with sparse negative aux).

### Blocked / not currently hidden-safe

- **Alexy NS1 sidecar extraction:** interesting CNN/noisy-student lineage, but source/API access was blocked and direct score was weak.
- **True G124 assets:** missing private/public checkpoint assets remain the blocker; direct wrappers without the real asset are not useful.
- **S14 / BidirProtoSSM + Snowflake SED:** potentially distinct but source/output access and hidden-safe rerun remain unresolved.
- **BirdNET official models:** model license is CC BY-NC-SA 4.0 per BirdNET-Analyzer README; must verify competition compatibility before packaging.
- **Some transformer/CLAP assets:** checkpoint licenses, Kaggle offline packaging, and runtime need explicit preflight before any private verifier.

### Genuinely new or still worth a no-slot smoke

- **General AudioSet embedding/no-call branch** from EfficientAT/PANNs/YAMNet or a single transformer. This directly targets hidden no-call/background and non-bird acoustic events.
- **Perch2/SurfPerch/Bioacoustics Model Zoo embedding residuals** trained against soundscape rows/no-train classes rather than replaying Perch+ProtoSSM finals.
- **BirdNET 2.4 embeddings/range residuals** as a bird-only correction or embedding-distance detector, not direct BirdNET scores.
- **CLAP/AudioCLIP semantic prompt branch** for common names, taxon descriptors, no-call/background prompts, and non-Aves sound classes.
- **Long-context SED/distillation branch** inspired by 2025/2024 public solutions, using 30–60s context and teacher/pseudo-label design changes.
- **Explicit background/no-call calibrator** using AudioSet/DCASE-style event models or mined negative windows.

## 4. Why these could produce hidden behavior different from the v616 plateau

The data-informed strategy report makes the key hidden-behavior case:

- The local proxy is narrow: about `190` matched rows, `20` files, `6` sites, `42` valid AUC classes.
- Train audio is Aves-heavy, while soundscape positives are strongly non-Aves-heavy.
- `28` submission classes have no train-audio primary labels, including frogs and `47158son*` insect sonotypes.
- Current artifact overlap misses some labeled sites and may not represent hidden sites/backgrounds.
- No-call/background behavior is undermeasured locally.

Candidate-specific hidden-behavior rationale:

1. **General AudioSet embeddings / YAMNet / PANNs / EfficientAT / transformers**
   - These models are trained to detect broad acoustic events, not only bird species. They may better separate rain, machinery, insects, choruses, silence, or other background regimes that can distort rank-blended species models.
   - They can power a no-call/background gate or non-Aves residual without copying Perch/ProtoSSM behavior.

2. **Perch2 / SurfPerch / Bioacoustics Model Zoo residuals**
   - Perch itself is partly in the plateau, but SurfPerch/general-bioacoustic representations and few-shot embedding heads may respond differently to frogs/insects/general audio.
   - Embedding kNN/logistic heads trained on soundscape windows can target classes not learnable from train-audio primary labels.

3. **BirdNET embeddings/range residuals**
   - Direct scores likely overlap visual/BirdNET-style public946 branches, but embeddings and range/time priors can supply a different decision surface for Aves classes and suppress improbable detections.
   - It may be useful as a calibrated bird-only correction, leaving non-Aves to other branches.

4. **CLAP / AudioCLIP / AudioMAE-style semantic/self-supervised branch**
   - Text-audio models can score prompts and descriptors that are not available as training labels, e.g. "frog chorus", "insect trill", "distant bird", "rain", "no animal call".
   - Even noisy semantic similarities could decorrelate from v616 and become useful as a low-weight gate or residual if locally stable.

5. **BirdCLEF 2025-style long-context SED/distillation**
   - v616 is fundamentally a 5-second rank blend. Long-context training/inference can model persistent choruses, temporal co-occurrence, and event continuity.
   - Prior public solutions emphasized 30/60-second crops, pseudo-labels, and distillation; current repo smokes have not yet tested that stronger formulation.

6. **Explicit no-call/background detector**
   - A suppression branch may improve hidden behavior without improving positive-only local rows much. If hidden public/private contains more sparse or background-heavy windows, this could be more important than another class-score sidecar.

## 5. Feasibility: assets, Kaggle data, runtime, license/source risk

| Family | Public assets | Kaggle/offline feasibility | Runtime risk | License/source risk | Scout status |
|---|---|---|---|---|---|
| BirdNET / `birdnet` | GitHub, PyPI, official models/Zenodo | Good if model packaged offline; CPU/GPU support advertised | Medium; must benchmark 5s windows over test soundscapes | Model CC BY-NC-SA 4.0 needs review | Hold for embedding residual smoke, not direct scores |
| Perch / Perch2 | Kaggle Models, Google Research code, Bioacoustics Model Zoo | Good; Perch already common in Kaggle | Medium; TFLite/ONNX preferred | Verify Kaggle model license/version | Direct Perch exhausted; residual/few-shot still useful |
| SurfPerch | Referenced by Perch README / Kaggle model | Need asset/version discovery | Medium | Verify model license | Newer/general-audio path worth smoke if accessible |
| Bioacoustics Model Zoo / BirdSet / HawkEars | Public Python package/repo | Good for offline extraction if deps packaged | Low–medium depending model | Verify individual model licenses | Good wrapper for quick comparative smoke |
| PANNs Cnn14 | GitHub + Zenodo checkpoints + `panns_inference` | Good PyTorch path; package checkpoint as dataset | Low–medium; Cnn14 embeddings feasible | Check checkpoint license | Top candidate for immediate smoke |
| EfficientAT | Public code/checkpoints | Good if dependencies minimal | Low–medium; designed for efficient inference | Verify checkpoint license | Top candidate for immediate smoke |
| YAMNet | TF Models / TF Hub | Good if TF Hub model cached/offline or BMZ wrapper works | Low; MobileNetV1 | Apache/source likely OK but verify asset license | Good no-call/background smoke |
| PaSST / AST / HTS-AT / BEATs / ATST / EAT | Public repos/checkpoints | Feasible offline with careful packaging | Medium–high; transformer CPU path may be too slow | Verify checkpoint terms; some checkpoints on OneDrive/GDrive/HF | Use only after smaller AudioSet smoke passes |
| CLAP / AudioCLIP / AudioMAE | Public repos/packages/checkpoints | Feasible offline but package size/deps need audit | Medium–high | Verify commercial/model licenses | Semantic smoke only; no direct verifier until stable |
| BirdCLEF 2025 long-context SED | Public solution repos | Repo already has SED infrastructure; needs config changes | Medium; train-time and inference windowing need budget | Public code/checkpoint license review | Promising but needs model-gate improvement |
| No-call/background detector | Can be built from AudioSet/DCASE/BirdCLEF background methods | Local mining possible; no submission needed | Low if simple gate | Low if trained repo-owned; external assets if pretrained | Strong hidden-behavior rationale |

## 6. Top 5 recommended external/public-model branches and first smoke experiment

### 1. Efficient AudioSet event/no-call branch (`EfficientAT` first; `PANNs Cnn14` fallback)

**Why this is top-ranked:** It is the cleanest non-plateau signal. AudioSet models should react to background/no-call/general event structure that v616's visual/Perch/ProtoSSM/SED lineage may not represent. It also directly addresses the local validation blind spot: positive-heavy rows and weak no-call measurement.

**Public assets:**

- EfficientAT: <https://github.com/fschmid56/EfficientAT>
- PANNs/Cnn14: <https://github.com/qiuqiangkong/audioset_tagging_cnn>

**First smoke:**

1. Extract embeddings and 527-class AudioSet logits for the same local soundscape windows used by the ensemble audit.
2. Train only a tiny logistic/ridge/LightGBM head for:
   - no-call/background score;
   - non-Aves score;
   - optional per-class residual for classes with soundscape positives.
3. Emit `audio_event_raw.csv` aligned to the v616 row IDs.
4. Run the ensemble audit harness against `anchor`, `v616`, and `audio_event_raw` with site/file bootstrap.
5. Promotion only if it improves vs `v616`, not just anchor, and shows a clear slice lift for no-train/non-Aves/background-sensitive rows.

### 2. Perch2 / SurfPerch / Bioacoustics Model Zoo few-shot embedding residual

**Why:** Direct Perch/ProtoSSM is saturated, but embedding residuals can be trained on BirdCLEF 2026 soundscape windows and target the `28` no-train-primary classes and non-Aves events. SurfPerch/general-audio embeddings are especially interesting if accessible because they are less bird-only.

**Public assets:**

- Perch: <https://github.com/google-research/perch>
- Kaggle Perch model: <https://www.kaggle.com/models/google/bird-vocalization-classifier>
- Bioacoustics Model Zoo: <https://github.com/kitzeslab/bioacoustics-model-zoo>

**First smoke:**

1. Use Bioacoustics Model Zoo or a cached Kaggle model to embed the local train-soundscape windows.
2. Train a leave-site-validated kNN/logistic head for no-train Amphibia/Insecta soundscape labels and a separate Aves residual head.
3. Emit `perch_embedding_residual_raw.csv` and compare against `v616` with strict site/file/taxon gates.
4. If SurfPerch is accessible, run the same head with Perch vs SurfPerch embeddings and keep only the more decorrelated branch.

### 3. BirdNET 2.4 embedding/range-aware Aves residual

**Why:** BirdNET's direct predictions are likely too close to known visual/BirdNET-style plateau behavior, but BirdNET embeddings and its geo/time/range machinery may make a useful **bird-only residual**. This can complement, not replace, a non-Aves branch.

**Public assets:**

- `birdnet` package: <https://github.com/birdnet-team/birdnet>
- BirdNET-Analyzer: <https://github.com/birdnet-team/BirdNET-Analyzer>

**First smoke:**

1. Package the model locally only after license review.
2. Run embeddings/predictions on the 20-file local overlap and map only submission Aves labels.
3. Build two raw outputs: direct BirdNET mapped scores and embedding-residual scores.
4. Kill direct-score path if correlation/behavior matches known visual anchor; keep only embedding residual if it shows Aves leave-site lift vs `v616` without harming non-Aves.

### 4. CLAP / AudioCLIP semantic prompt branch for non-Aves and background

**Why:** CLAP-style models can evaluate text prompts and descriptors where there is little/no supervised training data: frog/insect common names, "insect chorus", "frog call", "rain", "wind", "silence", "distant animal call". This is likely to be noisy, but if any prompt family has stable local slice signal it will be highly decorrelated from v616.

**Public assets:**

- LAION CLAP: <https://github.com/LAION-AI/CLAP>
- Microsoft CLAP: <https://github.com/microsoft/CLAP>
- AudioCLIP: <https://github.com/AndreyGuzhov/AudioCLIP>
- AudioMAE: <https://github.com/facebookresearch/AudioMAE>

**First smoke:**

1. Create a prompt bank from taxonomy common names, scientific names, taxon descriptors, and background/no-call prompts.
2. Score local soundscape windows with CLAP audio-text similarity.
3. Calibrate only on training soundscape rows using leave-site CV.
4. Emit `clap_prompt_raw.csv` plus a `clap_background_gate.csv` diagnostic.
5. Continue only if the branch is nonconstant, bounded, and shows slice-specific value vs `v616`.

### 5. BirdCLEF 2025-style long-context SED/distillation branch

**Why:** Current repo SED smokes proved the export/runtime path but failed the model gate. The public 2025/2024 solution ingredients suggest the missing piece is not just more B0 training; it is long context, pseudo-label/distillation strategy, and stronger/varied backbones.

**Public assets:**

- BirdCLEF 2025 5th solution: <https://github.com/myso1987/BirdCLEF-2025-5th-place-solution>
- BirdCLEF 2024 3rd-place component: <https://github.com/TheoViel/kaggle_birdclef2024>

**First smoke:**

1. Modify the existing repo SED smoke to test 30-second context with 5-second center-label output, not a new Kaggle kernel.
2. Use OOF-teacher soft labels plus a stronger EfficientNetV2-S/B3 or EfficientAT-initialized backbone if feasible.
3. Train a small 512-row smoke and require holdout macro AUC near `0.90` before scaling.
4. If it clears the smoke gate, export TorchScript/ONNX and emit a row-aligned raw branch for the ensemble harness.

## Bottom line

The public/model landscape still has useful external branches, but not in the already-saturated public leaderboard direction. The next scout work should be **embedding/residual/no-call/long-context branches**, especially general AudioSet and Bioacoustics Model Zoo variants, with `v616` treated as the control to beat rather than an anchor to perturb.

Recommended next concrete action: run the **EfficientAT/PANNs AudioSet event/no-call smoke** first because it has the best mix of public assets, likely decorrelation, runtime feasibility, and direct relevance to hidden background/non-Aves failure modes.
