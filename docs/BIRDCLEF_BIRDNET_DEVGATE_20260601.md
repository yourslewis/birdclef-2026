# BirdCLEF 2026 — BirdNET Foundation DEV-Gate (Phase 3)

Date: 2026-06-01 (PDT) · Lane: ClawTeam competition-research
Predecessors:
- `docs/BIRDCLEF_DIVERSITY_SCOUT_20260601.md` (proved our asset pool has no member that is
  simultaneously *orthogonal and competent*).
- `docs/BIRDCLEF_FOUNDATION_SHORTLIST_20260601.md` (identified **BirdNET** as the one
  structurally-different, competent foundation lever — independent corpus + non-Perch CNN).

**Goal of this run:** reproduce the public BirdNET baseline, emit a BirdNET
`train_soundscape` proxy prediction stream on the canonical 240-row / 234-species schema,
and DEV-gate it against the 0.950 frontier. **No LB submission.**

## Decision (TL;DR)

**DEMOTE BirdNET to a recorded datapoint (negative result).** BirdNET is the **most
decorrelated** stream we have ever scored (rank-decorrelation **0.8697**, vs the previous
high of 0.250 for fused PANNs+DyMN10) — but it is **incompetent where it matters**:
weak-class AUC **0.5729** (barely above the 0.5 chance line), site-bootstrap q05 **0.0**,
and file-bootstrap q05 **negative (−0.000482)**. It fails the promotion gate. This is the
**same two-cluster verdict** the diversity scout reached for PANNs/DyMN10 — except BirdNET
is the bird-specific, independent-corpus foundation that *should* have broken the pattern.
It did not, on this proxy.

## Promotion gate (all four must hold) — BirdNET result

| gate condition | required | BirdNET | pass? |
|---|---|---:|:--:|
| `cand_auc_on_E_weak_classes` meaningfully > 0.5 (real competence) | ≫ 0.5 | **0.5729** | ❌ (only +0.073 over chance) |
| `blend_best_lift` > 0 | > 0 | +0.000421 | ✅ (marginal, at w=0.05) |
| `blend_site_q05` > 0 | > 0 | **0.0** | ❌ |
| `blend_file_q05` > 0 | > 0 | **−0.000482** | ❌ |

→ **3 of 4 fail. DEMOTE.**

## Proxy build method (provenance, source-clean)

### Sources forked / reproduced (real, accessible Kaggle slugs)
- **Baseline reproduced:** `ahmadzulfiqar001/birdclef-2026-birdnet-baseline` — clean public
  notebook (27 cells, BirdNET-Analyzer v2.4 via `birdnetlib`, location-aware Pantanal
  inference, scientific-name→species-code mapping via `taxonomy.csv`). Pulled live via the
  Kaggle REST `kernels/pull` endpoint with `Authorization: Bearer <KGAT_ token>`.
- **Label-mapping/branch-wiring reference:** `yaroslavkholmirzayev/birdnet-third-branch-site-hour-prior-restore`
  (proves BirdNET integrates as an additive 3rd branch on the 0.95 stack; used as the
  wiring reference for label-space proxy mapping).

### Kaggle access (the Bearer workaround held)
- The installed `kaggle` CLI (2.0.0) cannot authenticate the new `KGAT_` token (sends it as
  HTTP Basic → `401`). The foundation scout's **`Authorization: Bearer` REST workaround**
  was used for everything here: `kernels/list`, `kernels/pull`, and the official
  `train_soundscapes` audio via `competitions/data/download/birdclef-2026/<url-encoded path>`.
- `competitions/data/download-all` returns a 2.6 GB stream but it is **truncated** at our
  end (EOCD missing; `train_soundscapes` come after `train_audio` in stream order), so the
  20 proxy `.ogg` files were pulled **per-file** (all 20, HTTP 200, valid Ogg/Vorbis mono
  32 kHz). Read-only; no submissions.

### Inference (`scripts/birdnet_proxy_infer.py`)
- `birdnetlib==0.18.1` → `Analyzer()` loaded the genuine **BirdNET-Analyzer v2.4** weights
  (**6522 labels** confirmed) on CPU (TF-Lite/XNNPACK). Location-aware inference at the
  Pantanal centroid (lat −17.0, lon −57.0), `week_48=-1`.
- **Difference from the public baseline:** the baseline emits thresholded `min_conf=0.1`
  detections (sparse). For a usable proxy we emit **dense per-segment probability vectors**:
  `min_conf=0.01`, and for each canonical 5 s segment we take the **max BirdNET confidence**
  over all 3 s detection windows overlapping that segment. Output row_id schema and the
  234-species column order are forced to exactly match the canonical proxy
  (`…/v644_yaroslav_0950/submission_protossm.csv`): **240 rows × 235 cols**, row_id order
  identical (asserted).
- Detection density was healthy (14–198 detections/file; **833 nonzero score cells across
  78 active species columns**, max confidence 0.942) — this is a real signal stream, not a
  fallback/constant.

### Label mapping (BirdNET 6522 → competition 234)
- **161 / 234 species mapped:**
  - **157 direct** scientific-name matches (lowercased/stripped vs `taxonomy.csv`).
  - **3 common-name fallback** (BirdNET uses a newer split genus): `ocecra1` (Ocellated
    Crake → *Micropygia schomburgkii*), `sptnig1` (Spot-tailed Nightjar → *Hydropsalis
    maculicaudus*), `yehcar1` (Yellow-headed Caracara → *Milvago chimachima*).
  - **1 taxonomic synonym** (manual, documented): `brnowl` American Barn Owl
    *Tyto furcata* → BirdNET *Tyto alba* (recent split).
- **73 / 234 unmapped → constant low prior `1e-4`:** these are the **non-Aves taxa** BirdNET
  structurally cannot detect (35 Amphibia, 28 Insecta, 8 Mammalia, 1 Reptilia) plus **1 bird
  with no BirdNET equivalent** (`dwatin1`, Dwarf Tinamou, monotypic *Taoniscus nanus*). The
  constant low prior keeps the column finite/non-degenerate and is documented in
  `artifacts/diversity_scout/birdnet_proxy/birdnet_label_mapping.json`.
- Caveat: of the 42 *valid* (label-bearing) classes the scout actually scores, BirdNET only
  has competence on the bird subset; non-Aves valid classes are flat-prior, which caps its
  achievable competence — but that is precisely the honest picture of "BirdNET as a
  foundation for THIS competition's 234-class Pantanal target."

## DEV table (diversity scout vs 0.950 frontier E)

Frontier E = rank blend `proto*0.60 + sed*0.40` of the v644 EoS8/PowerOptimization 0.950
branches; proxy base macro-AUC ≈ 0.99303 (42 valid classes, 190 matched proxy rows).
`--neg-weights --bootstrap 200`.

| metric | BirdNET baseline |
|---|---:|
| cand_auc (all valid) | **0.563917** |
| cand_auc_on_E_weak_classes | **0.572896** |
| cand_auc_on_E_weak_rows | 0.578520 |
| rank_decorrelation `1−ρ` | **0.869671** |
| residual_error_corr | 0.510016 |
| competence_above_chance_on_E_weak | 0.072896 |
| blend_best_weight | 0.05 |
| blend_best_lift | +0.000421 |
| blend_best_auc | 0.993450 |
| blend_site_q05 (p>0) | **0.0** (0.88) |
| blend_file_q05 | **−0.000482** |
| DEV_score | 0.001055 |
| gate_pass | **False** |

### Reading: BirdNET lands in the *decorrelated-but-incompetent* cluster

This is the cleanest confirmation yet of the shared-ceiling / cross-family hypothesis, but
with a twist. The scout previously split candidates into two clusters:
1. **competent + redundant** (SED/proto/jung21, AUC ~0.99, decorr ~0.1–0.23), and
2. **decorrelated + incompetent** (our PANNs/DyMN10, AUC 0.64–0.70, decorr ~0.11–0.25).

BirdNET pushes cluster 2's **decorrelation to a new extreme (0.870)** — it genuinely sees a
different representation — yet its competence on the frontier's weak classes (0.573) is the
**lowest** of any candidate, *below* even our incompetent PANNs/DyMN10 (0.69–0.73). The DEV
product-guard correctly keeps the bonus tiny; the weight-optimised blend can only justify
w=0.05 and the **per-file bootstrap q05 is negative**, i.e. on a leave-file-out basis the
blend more often *hurts* than helps. BirdNET is orthogonal but not competent **on this
240-row Pantanal proxy**.

### Why (honest caveats, not excuses)
- **Domain shift:** BirdNET's global-6k corpus under-covers Pantanal species; 73/234 classes
  (incl. all non-Aves valid targets) are flat-prior, which structurally caps weak-class AUC.
- **Sparse high-confidence detections:** even dense-mode BirdNET only lights 78 columns; the
  frontier's *weak* classes (where lift would come from) overlap heavily with the rare/quiet
  Pantanal species BirdNET barely fires on.
- **Tiny proxy:** 20 files / 6 sites / 42 valid classes — the file-bootstrap is high-variance;
  the negative file-q05 is a soft signal, but combined with chance-level competence it is
  enough to deny promotion.

## Decision and follow-ups

**DEMOTE BirdNET** to a recorded datapoint. Logged to
`artifacts/model_data_point_ledger/performance_table.md` + `.jsonl`.

This is a *valuable* negative result: it extends the diversity-scout conclusion from "our own
wrappers" to "an independent bird foundation" — even a non-Perch, bird-specific foundation
does not add competent orthogonal signal on this proxy. Before abandoning BirdNET entirely,
the one remaining lever that could flip it (out of scope for this no-submission run):
- Re-score on a **larger label-matched proxy** (more sites/files) where BirdNET's bird subset
  carries more valid classes, to de-noise the file-bootstrap.
- Restrict the blend to the **bird-only subset of weak classes** (where BirdNET is actually
  represented) instead of all 42 valid classes, mirroring the third-branch reference's
  scoped `BN_PROXY` wiring — a scoped blend might clear file-q05 even if the global one cannot.

Both are *diligence* follow-ups; on the current evidence the next foundation gain is **not**
BirdNET-as-a-global-branch, and SurfPerch/AudioMAE remain 0-kernel dead ends. The honest
state: **no currently-accessible foundation has cleared the diversity gate.**

## Reproduce

```bash
# 1. Pull baseline source + audio (Bearer REST; KGAT_ token)
#    scripts/birdnet_proxy_infer.py expects 20 train_soundscapes .ogg in --audio-dir

# 2. Build the dense BirdNET proxy
.venv_birdnet/bin/python scripts/birdnet_proxy_infer.py \
  --audio-dir <dir with 20 train_soundscapes .ogg> \
  --proxy-csv artifacts/source_winner_audit_20260531T0416Z/session_outputs/v644_yaroslav_0950/submission_protossm.csv \
  --out-csv artifacts/diversity_scout/birdnet_proxy/birdnet_train_soundscape_proxy.csv \
  --map-json artifacts/diversity_scout/birdnet_proxy/birdnet_label_mapping.json

# 3. DEV-gate
.venv_scout/bin/python scripts/birdclef_diversity_scout.py \
  --proto-csv .../v644_yaroslav_0950/submission_protossm.csv \
  --sed-csv   .../v644_yaroslav_0950/submission_sed.csv \
  --candidate "birdnet_baseline=artifacts/diversity_scout/birdnet_proxy/birdnet_train_soundscape_proxy.csv" \
  --neg-weights --bootstrap 200 \
  --out-dir artifacts/diversity_scout/birdnet_<stamp>
```

## Compact summary

```
cand_auc                      = 0.563917
cand_auc_on_E_weak_classes    = 0.572896   (chance=0.5; +0.073 over chance)
rank_decorrelation            = 0.869671   (highest of any candidate ever scored)
blend_best_weight / lift      = 0.05 / +0.000421
site_q05                      = 0.0        (p_gt_0 = 0.88)
file_q05                      = -0.000482  (negative → blend hurts leave-file-out)
DEV_score                     = 0.001055
decision                      = DEMOTE (decorrelated-but-incompetent; no LB submission)
```
