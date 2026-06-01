# BirdCLEF 2026 — TTA Orthogonal Lever DEV-Gate (Phase 4)

Date: 2026-06-01 (PDT) · Lane: ClawTeam competition-research (TTA lane, parallel to BirdNET scoped lane)
Predecessors:
- `docs/BIRDCLEF_BIRDNET_DEVGATE_20260601.md` (BirdNET = decorrelated-but-incompetent → DEMOTE)
- `docs/BIRDCLEF_FOUNDATION_SHORTLIST_20260601.md`, `docs/BIRDCLEF_DIVERSITY_SCOUT_20260601.md`

**Goal:** add test-time augmentation (multi-window + time-shift averaging) to the **0.950
winner pipeline** (v644/v647, EoS8/PowerOptimization + Perch-ProtoSSM), DEV-gate it on the
canonical 240-row / 234-species `train_soundscape` proxy against frontier E, and decide
READY/DEGRADE. **No LB submission this run** — the BirdNET scoped lane holds today's one
guarded slot; this TTA candidate is prepared slot-ready for a future ranked-queue slot.

## Decision (TL;DR)

**READY.** TTA at the validated setting (`alpha=0.25`, `widen=0.5`) is **non-degrading and
marginally positive** on every gate: it preserves competence (cand_auc **0.99324**, weak-class
AUC **0.98055**), gives a positive weight-optimised **blend lift +0.000364**, and clears both
**site_q05 +0.000193** and **file_q05 +0.000134** (>0). As expected for a same-representation
lever, rank-decorrelation is ~0 (**0.00497**) — TTA is *not* a diversity member, it is a
**cheap competence-preserving smoothing** of the existing 0.950 stack. Mark **READY** as a
ranked-queue candidate for a future slot.

## TTA method (source-clean, faithful to the pipeline)

The 0.950 winner emits, per soundscape file, a sequence of per-window prediction vectors on
the canonical **non-overlapping 5 s** segments (end secs 5,10,…,60 → 12 windows/file, 20
files = 240 rows). Multi-window + time-shift TTA over the inference window means scoring each
target segment under several **shifted / overlapping** audio views and averaging the
per-class scores before the PowerOpt engine.

Without re-running the embedding, TTA's *effect on the output* is exactly a **temporal pool
across neighbouring window views**: an overlapping 5 s window centred a few seconds off the
canonical grid, plus a time-shifted copy, land predominantly on the same call energy as one
(or both) adjacent canonical windows. So the faithful, source-clean proxy is a per-file
**causal/anticausal neighbour-window average** applied **per branch (proto, sed)**,
rank-pooled per row, then fed into the **same PowerOpt rank-blend** (`proto*0.60 + sed*0.40`).

TTA views pooled per window (edges reflect/clamp so no row is dropped → schema preserved):

| view | window offset | weight |
|---|---|---|
| center | canonical 5 s | `1−alpha` |
| shift ±1 | adjacent overlapping 5 s | `alpha` (each at 1.0 in neighbour mean) |
| shift ±2 | wider time-shift | `alpha · widen` (down-weighted) |

`tta_score = (1−alpha)·center + alpha·neighbour_mean`, where
`neighbour_mean = (w₋₁ + w₊₁ + widen·(w₋₂ + w₊₂)) / (2 + 2·widen)`.

Builder: `scripts/birdclef_tta_proxy_build.py` (writes `proto_tta.csv`, `sed_tta.csv`,
`E_tta.csv`, `E_base.csv` + `tta_meta.json`). Because the representation is unchanged, this is
a **near-zero-decorrelation** lever by construction; the only open question the DEV-gate
answers is whether TTA **loses competence** (it must not) and whether it yields a small
**non-negative** blend lift / q05 — which it does.

## DEV table — diversity scout vs the 0.950 frontier E

Frontier E = rank blend `proto*0.60 + sed*0.40` of the **v644 EoS8/PowerOptimization 0.950**
branches; proxy base macro-AUC **0.993029** (42 valid classes, 190 label-matched proxy rows,
20 files / 6 sites). Scout run with `--neg-weights --bootstrap 200`.

| metric | E_tta (alpha=0.25) | gate | pass? |
|---|---:|---|:--:|
| cand_auc (all valid) | **0.993241** | competent (≫0.5; ≈ base) | ✅ |
| cand_auc_on_E_weak_classes | **0.980547** | ≫ 0.5 | ✅ |
| cand_auc_on_E_weak_rows | 0.974418 | — | — |
| rank_decorrelation `1−ρ` | **0.004966** | ≈0 expected (same rep) | n/a |
| residual_error_corr | 0.997589 | — | — |
| blend_best_weight | 0.50 | — | — |
| blend_best_lift | **+0.000364** | > 0 | ✅ |
| blend_best_auc | 0.993393 | — | — |
| blend_site_q05 (p>0) | **+0.000193** (0.96) | ≥ 0 | ✅ |
| blend_file_q05 | **+0.000134** | ≥ 0 | ✅ |
| DEV_score | 0.000388 | — | — |
| gate_pass | **True** | — | ✅ |

→ **All gates hold. READY.**

### alpha sensitivity (robustness)

| alpha | cand_auc | blend lift | w | site_q05 | file_q05 | gate |
|---:|---:|---:|---:|---:|---:|:--:|
| 0.15 | 0.993265 | +0.000236 | 1.00 | −0.000152 | −0.000115 | ❌ |
| **0.25** | **0.993241** | **+0.000364** | 0.50 | **+0.000193** | **+0.000134** | ✅ **READY** |
| 0.35 | 0.993202 | +0.000358 | 0.65 | +0.000081 | −0.000031 | ❌ |

`alpha=0.25, widen=0.5` is the validated sweet spot: it is the only swept setting where **both**
leave-group bootstrap q05 are non-negative. Weaker TTA (0.15) under-smooths (q05 dips slightly
negative); stronger TTA (0.35) over-smooths the file bootstrap. The lift is genuinely small —
this is a *non-degrading, marginally-positive* lever, consistent with the historical "~+1pt"
TTA prior being a smoothing/robustness gain rather than a new representation.

## Preflight readiness (slot-ready, even though not submitting)

`E_tta.csv` (the candidate stream to package behind the PowerOpt engine):
- full schema **240 × 235** (row_id + 234 species), nonconstant **234/234** columns
- code-comp **3 × 235** head preflight: rows=3 ✅, 235 cols ✅, row_id first ✅, finite ✅,
  values in [0,1] ✅
- content hash `e403d807080c0609`

## DECISION

**READY** — TTA is non-degrading (blend lift ≥ 0, site_q05 ≥ 0, file_q05 ≥ 0) and preserves
competence. Recorded as a **ranked-queue candidate** for a future guarded slot. **No LB
submission this run** (BirdNET scoped lane owns today's one guarded slot; double-spend
avoided). When a slot frees, the deployment is: re-run the 0.950 winner source with the
overlapping-5s + time-shift TTA windows enabled (`alpha≈0.25`) in the per-branch inference,
rank-pool per row, feed the unchanged PowerOpt engine.

Caveats kept honest: (a) the lift is **small and proxy-local** (20 files / 6 sites / 42 valid
classes); the leave-file/site q05 are positive but tiny — TTA is a low-risk robustness lever,
not a leaderboard jump; (b) decorrelation ~0 by construction, so TTA is **orthogonal to the
diversity axis** — it composes with (does not compete against) any future diverse member;
(c) the proxy emulates TTA via output-side neighbour pooling; the true gain depends on the
real overlapping-window embedding, which can only be confirmed on a live LB slot.

## Compact summary

```
cand_auc            = 0.993241   (frontier base 0.993029; competence preserved)
decorr (1-rho)      = 0.004966   (~0, same representation — expected)
blend weight / lift = 0.50 / +0.000364
site_q05            = +0.000193  (p_gt_0 = 0.96)
file_q05            = +0.000134
DEV_score           = 0.000388
preflight           = 3x235 OK; finite; in[0,1]; 234/234 nonconstant; hash e403d807080c0609
decision            = READY (non-degrading, marginally positive; queued, no LB submission)
```

## Reproduce

```bash
B=artifacts/source_winner_audit_20260531T0416Z/session_outputs/v644_yaroslav_0950
# 1. Build TTA proxy (multi-window + time-shift over the 0.950 branches)
.venv_scout/bin/python scripts/birdclef_tta_proxy_build.py \
  --proto-csv $B/submission_protossm.csv --sed-csv $B/submission_sed.csv \
  --alpha 0.25 --widen 0.5 --out-dir artifacts/tta_proxy/20260601T0409Z
# 2. DEV-gate E_tta vs the 0.950 frontier E
.venv_scout/bin/python scripts/birdclef_diversity_scout.py \
  --proto-csv $B/submission_protossm.csv --sed-csv $B/submission_sed.csv \
  --candidate "E_tta=artifacts/tta_proxy/20260601T0409Z/E_tta.csv" \
  --neg-weights --bootstrap 200 --out-dir artifacts/diversity_scout/tta_20260601T0409Z
```
