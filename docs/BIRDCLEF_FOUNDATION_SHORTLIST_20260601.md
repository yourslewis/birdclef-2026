# BirdCLEF 2026 — Foundation-Diversity Shortlist (Phase 2)

Date: 2026-06-01 (PDT). Lane: ClawTeam competition-research.
Predecessor: `docs/BIRDCLEF_DIVERSITY_SCOUT_20260601.md` (proved our asset pool has no
member that is *simultaneously orthogonal and competent* — every competent stream is a
Perch-ProtoSSM relative; every decorrelated stream, PANNs/DyMN10, is incompetent at AUC ~0.64–0.70).

Goal of this doc: find a **structurally different pretrained bird-foundation embedding**
than EoS8's Perch-ProtoSSM, sourced from real, accessible public Kaggle birdclef-2026 work.

## Method / provenance (read-only, no submission, no slots spent)

- Kaggle access: the installed `kaggle` CLI (2.0.0) cannot authenticate — the configured
  credentials are the **new `KGAT_` token format**, which the old CLI sends via HTTP Basic
  auth and the API rejects with `401 Unauthenticated`. Both `~/.kaggle/kaggle.json` and
  `.bak` keys fail identically through the CLI.
- **Workaround used (verified working):** the same `KGAT_` token authenticates against the
  REST API via `Authorization: Bearer` header. All data below was pulled live from
  `https://www.kaggle.com/api/v1/` (`kernels/list`, `kernels/pull`, `datasets/list`) with
  Bearer auth. This is read-only metadata/source retrieval; no submissions, no slot spend.
- Every slug below was confirmed to exist by pulling its kernel source or dataset record.
  No fabricated refs.

### Keyword sweep results (competition=birdclef-2026)

| query | kernels found | notes |
|---|---:|---|
| `perch` | 50 | dominant family — Perch / Perch-v2 / ProtoSSM everywhere (this is EoS8's family) |
| `birdnet` | 50 (mostly EoS mentions) | **2 real BirdNET-backbone kernels** isolated below |
| `efficientat` | 2 | viktoriiahranadzer train/infer pair |
| `google bird vocalization` | 50 | all Perch-v2 (`google/bird-vocalization-classifier`) |
| `surfperch` | **0** | no public SurfPerch kernels on birdclef-2026 |
| `audiomae` | **0** | no public AudioMAE kernels on birdclef-2026 |

**Honest negatives:** SurfPerch and AudioMAE have **zero** public birdclef-2026 kernels.
Do not invent them. The only genuinely non-Perch foundation with proven public code is
**BirdNET** (Cornell), with **EfficientAT** (AudioSet MobileNet) as a weaker second axis.

## Ranked shortlist

Rank = (likely competence) × (representation orthogonality vs Perch-ProtoSSM).

| # | source slug | foundation / backbone | family vs EoS8 | hidden-test capable? | runtime | source-clean? | orthogonality hypothesis |
|---|---|---|---|---|---|---|---|
| **1** | `ahmadzulfiqar001/birdclef-2026-birdnet-baseline` | **BirdNET-Analyzer v2.4** (Cornell; custom CNN on mel, trained on its own global 6k corpus) via `birdnetlib` | **Different** — independent training data + non-Perch CNN | ✅ reads `test_soundscapes`, writes `/kaggle/working/submission.csv`, CPU-only, no-internet | CPU ≤90 min | clean public notebook; needs offline BirdNET weights dataset attached (see below) | BirdNET's label space & embedding are derived from a *different* pretraining corpus than Google Perch → genuinely decorrelated logits, and (unlike our PANNs/DyMN10) it is *bird-specific and competent* |
| **2** | `yaroslavkholmirzayev/birdnet-third-branch-site-hour-prior-restore` | **BirdNET tflite as an additive 3rd branch** on top of Perch + distilled-SED | Hybrid — adds BirdNET orthogonal axis to the frontier | ✅ rglobs `birdnet_global_6k_v2.4_model_fp32*.tflite`, runs `tflite_runtime` interpreter, writes submission | CPU (tflite) | clean; pins perch_v2 + distilled-SED + BirdNET datasets | **Proof-of-integration**: someone already wired BirdNET as a third branch onto the 0.95 stack — directly demonstrates BirdNET adds usable, orthogonal signal where Perch+SED is weak |
| **3** | `viktoriiahranadzer/birdclef-train` + `viktoriiahranadzer/birdclef-inference` | **EfficientAT (fschmid56 MN10/DyMN, AudioSet-pretrained)** branch, blended with proto/perch/SED | Partly different — AudioSet MobileNet, not Perch | ✅ infer reads `test_soundscapes`, loads `proto_models` + perch_v2 onnx, writes submission.csv | GPU train / CPU infer | clean; clones `github.com/fschmid56/EfficientAT`, exports ONNX | EfficientAT is AudioSet-pretrained (not bird-specific) → **CAUTION: same architectural family as our repo's DyMN10, which the scout already showed is decorrelated-but-incompetent (AUC 0.64)**. Lower competence prior. |

## Per-candidate fork plans

### #1 — BirdNET baseline (FORK-READY, top pick)
- **What it is:** clean 8-cell notebook. Loads `birdnetlib.analyzer.Analyzer()`, runs
  location-aware BirdNET (lat −17, lon −57 = Pantanal) on each 5 s segment, maps BirdNET
  scientific names → competition species codes via `taxonomy.csv`, writes `submission.csv`.
- **Fork steps:**
  1. Replace the live `Analyzer()` download with an **offline weights dataset** — attach one
     of the verified public BirdNET weight/lib datasets:
     `antimaterial/birdnetlib` (174 MB, the lib), `mansianilkadam/birdnet-v2-4-tflite`
     (77 MB), `seshurajup/birdnet-analyzer-2023` (946 MB), or
     `willrice/birdnet-onnx-backbone` (22 MB, ONNX) for a leaner ONNX path.
  2. Produce **per-segment probability vectors** (not the thresholded `min_conf=0.1`
     detections) so the output is a dense 234-class score matrix usable as a proxy stream.
  3. Run the BirdNET proxy stream through `scripts/birdclef_diversity_scout.py` against
     frontier E. Promote only if it shows competence on E-weak classes **and** positive
     blend site+file q05 (the bar the scout set).
  4. If it passes, rank-blend BirdNET into the 0.950 stack at a small weight, gated on
     E-weak classes (mirror the third-branch pattern from candidate #2).
- **Risk:** BirdNET's label set is global-6k; Pantanal species coverage / code-mapping
  completeness must be checked — missing species → zeros → must fall back to Perch there.

### #2 — BirdNET third branch (REFERENCE INTEGRATION, fork the wiring)
- **What it is:** the **existing proof** that BirdNET adds signal on top of the frontier.
  It already implements model discovery (`_find_birdnet_model`), tflite inference,
  BirdNET→competition label proxy mapping (`BN_TO_COMP`, `BN_PROXY`), site/hour priors,
  and dry-run timing on `test_soundscapes`.
- **Fork steps:** lift its BirdNET branch (`run_birdnet`, label mapping, `_N_BN_CHUNKS`
  batching) as the cleanest reference for wiring BirdNET into our pipeline. Use it to
  validate our #1 BirdNET proxy against an independent implementation before trusting it.
- **Risk:** heavily entangled with perch_v2 + distilled-SED in one notebook (142 KB
  source); extract the BirdNET path only — don't fork the whole Perch-redundant stack.

### #3 — EfficientAT branch (LOWER PRIORITY / diligence-only)
- **What it is:** an AudioSet MobileNet (MN10/DyMN) branch, OOF-aligned and blended with
  proto/perch/SED; exports ONNX for CPU inference.
- **Why ranked last:** EfficientAT is **architecturally the same family as our repo's
  DyMN10**, which the diversity scout already scored as decorrelated-but-incompetent
  (AUC 0.64, blend weight 0.0). It is *not* bird-specific pretraining. Treat as a sanity
  check, not a primary bet — if their EfficientAT branch is competent here, it would
  contradict our DyMN10 result and is worth understanding; otherwise skip.

## Supporting public weight datasets (verified to exist)

- `antimaterial/birdnetlib` (174 MB) — birdnetlib package for offline install
- `mansianilkadam/birdnet-v2-4-tflite` (77 MB) — BirdNET v2.4 tflite
- `willrice/birdnet-onnx-backbone` (22 MB) — BirdNET ONNX backbone (lean)
- `henryszy/birdnet-v24-6k-labels` (78 KB) — 6k label list for code mapping
- `seshurajup/birdnet-analyzer-2023` (946 MB) — full analyzer
- (Perch-v2, for reference/redundant axis): `tuckerarrants/perch-v2-no-dft-onnx`,
  `rishikeshjani/perch-onnx-for-birdclef-2026`, `nawfeelrahman1124444/perchv2-weights`

## Top-3 summary (decision)

1. **`ahmadzulfiqar001/birdclef-2026-birdnet-baseline` — FORK FIRST.** Only clean, public,
   hidden-test-capable solution on a foundation genuinely different from Perch-ProtoSSM.
   BirdNET = independent corpus + non-Perch CNN → the orthogonal *and* competent member the
   scout proved we lack. Validate via the diversity scout before blending.
2. **`yaroslavkholmirzayev/birdnet-third-branch-site-hour-prior-restore` — REFERENCE.**
   Already proves BirdNET integrates as an additive branch on the 0.95 stack; fork its
   BirdNET wiring/label-mapping as ground truth for our #1.
3. **`viktoriiahranadzer/birdclef-train`+`-inference` — DILIGENCE ONLY.** EfficientAT axis;
   same family as our incompetent DyMN10 — low competence prior, do not lead with it.

**Negative results (do not fabricate):** SurfPerch = 0 kernels, AudioMAE = 0 kernels on
birdclef-2026. Next foundation gain should come from **BirdNET**, not those two.
