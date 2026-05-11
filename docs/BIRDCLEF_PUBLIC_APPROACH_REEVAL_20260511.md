# BirdCLEF 2026 Public Approach Re-evaluation — 2026-05-11

Status: actionable spec addendum  
Owner branch at creation: `feature/v536-oof-ensemble-teacher-cache` / PR #221  
Current internal public LB anchor: **v517 = 0.930**  
New public frontier observed: **~0.946** public notebooks built around public distilled SED + Perch/ProtoSSM rank blending.

This document supersedes the earlier assumption that the active plateau is 0.927.  It does **not** replace `docs/BIRDCLEF_NEW_DIRECTIONS_SPECS.md`; it reprioritizes the next work based on newly inspected public/shared kernels and 2025 solution writeups.

---

## Sources inspected

Public/shared BirdCLEF 2026 kernels pulled via Kaggle API:

| Ref | Title / claim | Key sources |
|---|---:|---|
| `raunakdey07/birdclef-2026-onnx-perch-sed-blend` | ONNX Perch + SED blend | `tuckerarrants/bc2026-distilled-sed-public`, Perch ONNX/no-DFT, Perch meta |
| `safar1/lb-score-0-946` | `[LB] [SCORE] [0.946]` | same core stack |
| `yaroslavkholmirzayev/0-946-replay-with-robust-inputs` | 0.946 replay with robust inputs | same core stack, more robust path resolution |
| `needless090/birdclef-2026-perch-sed-lb-0-946-clap` | Perch+SED LB 0.946 + CLAP | adds V5 and CLAP streams when available |
| `raunakdey07/birdclef-2026-birdnet-4-way-rank-blend` | BirdNET 4-way rank blend | adds BirdNET + custom EffNet streams |
| `imaadmahmood/birdclef-2026-perch-v2-protossm-0-925` | original Perch v2 + ProtoSSM baseline | our current family descends from this axis |

External 2025 solution references inspected:

- Medium postmortem by Max Melichov: Quantile-Mix / rank+mean blending, public SED models, and pretraining on previous BirdCLEF data were the largest practical gains.
- Tekkix 2025 top-team overview: winning-level recipes centered on SED, pseudo-labeling rounds, power scaling, class/taxon-specific pipelines, external data, TTA, and diverse EfficientNet/NFNet/RegNet ensembles.
- `VSydorskyy/BirdCLEF_2025_2nd_place`: OpenVINO/fp16 export, eca_nfnet_l0 + EfficientNetV2-S, Focal+BCE, sqrt/equal balancing, multi-round pseudo labels.
- `myso1987/BirdCLEF-2025-5th-place-solution`: SED model zoo with EfficientNet-B0/B3, EfficientNetV2-S/B3, pseudo-label stages, OpenVINO inference.

---

## What changed

The public frontier has moved beyond our internal v517=0.930.  The biggest difference is not a novel postprocess knob; it is a **new prediction artifact**:

> `tuckerarrants/bc2026-distilled-sed-public` + Perch/ProtoSSM, blended in rank space.

Common public 0.946 pattern:

1. Generate a strong Perch/ProtoSSM stream.
2. Generate public distilled SED predictions from ONNX SED folds.
3. Convert both streams to per-class ranks.
4. Blend roughly **60% ProtoSSM / 40% SED**.
5. Apply targeted rescue/gating rules:
   - **fake-only Proto rescue** when Proto is high but SED is near-zero.
   - **Proto temporal continuity rescue** using a fat-tailed +/-3-window context kernel.
   - **SED-only local spike rescue** when SED rank is very high and Proto rank is low.
   - **sonotype mirroring** for known similar insect label groups.
   - **rare taxon suppression** for Amphibia/Mammalia/Reptilia below a dynamic threshold.
6. Some forks add V5, CLAP, BirdNET, or custom EffNet streams, but the core jump is already in the Perch+distilled-SED rank blend.

---

## Priority order

### P0 — Reproduce public 0.946 baseline as a controlled internal kernel

**Hypothesis:** The public distilled SED + rank-blend stack is now the strongest available baseline and should replace v517 as the teacher/reference axis if it scores near its public claim.

**Action:** Create a repo-owned Kaggle kernel candidate that faithfully ports the robust public 0.946 stack, starting from the simplest reproducible two-way version:

- Inputs:
  - `tuckerarrants/bc2026-distilled-sed-public`
  - `jaejohn/perch-meta`
  - `tuckerarrants/perch-v2-no-dft-onnx` if used by chosen source kernel
  - `rishikeshjani/perch-onnx-for-birdclef-2026`
  - `ashok205/tf-wheels`
  - Perch v2 Kaggle model
- Base candidate: robust 2-way replay, not the most exotic CLAP/BirdNET fork.
- Required validation:
  - public dry-run `submission.csv` shape `(240,235)`.
  - log confirms SED ONNX folds loaded, not silently skipped.
  - log confirms rank blend and postprocess path executed.
  - wall time comfortably under 90-min CPU budget.
- Submission policy:
  - Submit as soon as current cap/queue allows, ahead of further internal OOF-teacher variants.
  - Do not submit multiple public-stack forks before the baseline score lands.

**Success:** LB >= 0.940, or ties the public 0.946 neighborhood.  If it lands only around 0.930, inspect row/label alignment, data source versions, and dry-run/hidden-test behavior before judging the lane.

---

### P1 — Make public 0.946 the new teacher cache

**Hypothesis:** Copying the public stack may improve public LB, but private edge requires training our own models from it.  2025 top solutions repeatedly gained from pseudo-label rounds and power-scaled/self-training.

**Action:** After P0 is verified, run the public 0.946 stack over train soundscapes / available pseudo-label targets and save a teacher cache:

- `artifacts/pseudolabels/public946-teacher-cache-v1/predictions.npz`
- include row IDs, labels, raw Proto stream, raw SED stream, final rank-blend output, and metadata.
- compute diagnostics vs labeled train soundscapes:
  - macro AUC over valid labels
  - top-k recall
  - confidence histogram
  - class/taxon distribution
  - correlation with v517 and v537/v538 streams

**Training candidates:**

1. `pl-public946-b0-5s-v1` — fast B0 sanity student.
2. `pl-public946-eca-nfnet-l0-v1` — 2025-proven high-upside backbone.
3. `pl-public946-v2s-v1` — V2S, but only if correlation/OOF diagnostics beat the prior v508-distilled V2S failure.

**Pseudo-label knobs:**

- soft labels: raw, power 0.85, power 1.0, power 1.15.
- hard positives: >0.90 / >0.95 / >0.98.
- hard negatives: <0.01 / <0.02 / <0.05.
- cap per class to avoid common-species domination.
- use source-fold or OOF-style splits to reduce teacher leakage.

**Success:** student has either competitive AUC or low enough correlation to improve a blend with the public946 teacher.  Do not package students whose only gain is a tiny local AUC delta with high teacher correlation.

---

### P2 — Build an internal blend on top of public 0.946, not the other way around

**Hypothesis:** Our v517/v537/v538 signals may still be useful as minority streams, but public946 should be the anchor.

**Action:** Once P0 output is available, run offline blend grids:

- public946 rank output + v517 output.
- public946 rank output + v537/v538 OOF-teacher sidecar output.
- public946 raw Proto + public distilled SED + our OOF-teacher B0 sidecar.
- optional BirdNET/CLAP fork outputs if locally reproducible.

**Initial weights:**

- public946 0.90 + v517 0.10
- public946 0.85 + v517 0.10 + v537/v538 0.05
- public946 0.80 + distilled SED variant 0.10 + OOF-teacher 0.10

**Rules:**

- Prefer rank-space blends for heterogeneous models.
- Submit only one blend variant after baseline public946 score lands.
- Kill any blend if it mainly reintroduces the old v517 bias and lowers class diversity.

---

### P3 — CLAP/BirdNET/custom EffNet streams: evaluate as optional diversity, not first-line

**Hypothesis:** CLAP/BirdNET can add acoustic diversity, but public kernels suggest they are optional layers on top of the core Perch+SED jump.

**Action:** Treat as controlled add-ons after P0/P1:

- CLAP fork: reproduce only if required data sources are available and per-file mask coverage is high.
- BirdNET 4-way fork: evaluate rank output correlation with public946 and v517.
- Custom EffNet stream: only package if it has a real external model source and does not blow runtime.

**Success:** adds >=0.001 in public or strong OOF blend without hidden-runtime risk.

---

### P4 — Continue OOF-teacher B0 sidecar only as a queued diagnostic

**Current state:** v537/v538 are valid and runtime-safe, but they are now lower priority because public946 is a much larger jump.

- Keep current queued v537/v538 submissions if already queued and cap slots are allocated.
- Do not create v539/v540 B0-sidecar weight sweeps unless v537/v538 tie/improve v517 or show a useful blend with public946.
- If v537/v538 underperform, demote the B0 OOF-teacher lane to training-data infrastructure only.

---

### P5 — 2025-style external pretraining remains a medium-term lane

2025 evidence supports:

- Xeno Archive / previous BirdCLEF pretraining.
- eca_nfnet_l0 + EfficientNetV2-S.
- Focal+BCE, sqrt/equal balancing.
- OpenVINO/fp16 or ONNX export.
- multi-round pseudo-labeling with power scaling.

But this requires more time than reproducing public946.  Keep it as the next major training lane after P0/P1.

---

## Immediate implementation plan

1. **Branch:** continue on PR #221 only if convenient, or create `feature/public946-replay-baseline` if the diff gets too large.
2. **Kernel:** create `kaggle-kernels/v539-public946-replay/` from the robust public two-way rank-blend source.
3. **Push:** real Kaggle kernel via Bearer API v1.
4. **Monitor:** add v539 to the queue **ahead of new experimental sidecars** after current already-queued items are safely handled.
5. **Cache:** once v539 output is verified, download/cache public946 predictions for local blend and student training.
6. **Student:** train one B0 or NFNet student from public946 teacher cache; do not start a large model zoo until the cache diagnostics are known.

---

## Kill / caution rules

- Do not blindly trust public notebook titles. Verify the exact attached data-source versions and log path.
- Do not submit CLAP/BirdNET forks before the core public946 baseline lands.
- Do not keep spending submission slots on v517/v537/v538 micro-variants if public946 reproduces.
- Do not merge any of this to `main` without Wenhao approval.
