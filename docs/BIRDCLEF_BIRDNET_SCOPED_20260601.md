# BirdCLEF 2026 — Scoped BirdNET Blend Re-Gate (Phase 3b)

Date: 2026-06-01 (PDT) · Lane: ClawTeam competition-research
Predecessor: `docs/BIRDCLEF_BIRDNET_DEVGATE_20260601.md` (full BirdNET proxy DEMOTED:
rank_decorr 0.870 highest-ever, but weak-class AUC 0.573 ~ chance, site_q05 0.0,
file_q05 −0.000482). Binding constraint there was **competence**, hypothesised to be
diluted by the 73 flat-prior / unmapped (mostly non-Aves) classes.

**Goal:** restrict the BirdNET stream to its ~78 active/competent columns, leave every
other class at the 0.950 frontier value (no flat priors injected into the blend),
re-DEV-gate, and queue exactly ONE guarded LB slot **only if** the scoped blend clears
the gate or is provably non-harmful.

## Decision (TL;DR)

**DEMOTE — no LB slot spent.** Scoping **fixed the competence problem** exactly as
hypothesised (weak-class AUC **0.573 → 0.919**), but the blend's **out-of-sample harm
signal is unchanged** and its **diversity collapsed**:

- At the DEV-best weight (w=0.05) the scoped blend is numerically identical to the full
  blend on the scored valid classes — **lift +0.000421, site_q05 0.0, file_q05 −0.000482**.
- Scoping the stream to where BirdNET agrees with the frontier drove
  **rank_decorrelation 0.870 → 0.177** (< 0.3), so the non-harmful clause's diversity
  retention requirement also fails.

So neither promotion path holds:
- **Gate (a)** (weak-class AUC > 0.5 AND lift > 0 AND site_q05 > 0 AND file_q05 > 0):
  fails on site_q05 = 0 and file_q05 < 0.
- **Gate (b)** non-harmful (file_q05 ≥ 0 AND site_q05 ≥ 0 while decorr > 0.3):
  fails on file_q05 < 0 AND decorr 0.177 < 0.3.

The honest reading: BirdNET's *real* competent signal lives on the 10 active∩valid classes
where it already **agrees** with the 0.950 frontier (decorr collapses), and on the rare/quiet
Pantanal classes where lift would actually come from, it still does not fire. Scoping trades
orthogonality for competence on the same cells the frontier already wins — it does not
manufacture new complementary, out-of-sample-robust signal. **No slot is justified.**

## Scoped method (source-clean, no flat priors in the blend)

- **Frontier E** = rank blend `proto*0.60 + sed*0.40` of the v644 EoS8/PowerOptimization
  0.950 branches (`submission_protossm.csv`, `submission_sed.csv`), re-ranked. Proxy base
  macro-AUC ≈ 0.99303 (42 valid classes, 190 label-matched rows of 240).
- **Active set** = the **78** BirdNET columns with real signal (max confidence > the
  `1e-4` floor; 833 nonzero score cells). Of the 42 *valid* (label-bearing) classes the
  scout scores, **10 are active** (BirdNET-represented) and **32 are non-active**.
- **Scoped candidate** (`artifacts/diversity_scout/birdnet_proxy/birdnet_scoped_proxy.csv`):
  for the 78 active columns use the BirdNET proxy value; for **all other 156 columns pass
  the frontier-E value unchanged** (asserted `np.allclose` vs E). No `1e-4` flat priors
  enter the blend — that was the dilution the predecessor flagged.
- Re-gated with `scripts/birdclef_diversity_scout.py` (`--neg-weights --bootstrap 200`)
  against the same frontier E. Full BirdNET re-scored alongside as a control.

## DEV table (scoped vs full, vs 0.950 frontier E)

| metric | full BirdNET (prior) | **scoped BirdNET** |
|---|---:|---:|
| cand_auc (all 42 valid) | 0.563917 | **0.941057** |
| cand_auc_on_E_weak_classes | 0.572896 | **0.918920** |
| cand_auc_on_E_weak_rows | 0.578520 | 0.793373 |
| rank_decorrelation `1−ρ` | **0.869671** | **0.176605** |
| residual_error_corr | 0.510016 | 0.892159 |
| competence_above_chance_on_E_weak | 0.072896 | **0.418920** |
| blend_best_weight | 0.05 | 0.05 |
| blend_best_lift | +0.000421 | **+0.000421** |
| blend_best_auc | 0.993450 | 0.993450 |
| blend_site_q05 (p>0) | 0.0 (0.88) | **0.0 (0.88)** |
| blend_file_q05 | −0.000482 | **−0.000482** |
| DEV_score | 0.001055 | 0.001161 |
| gate_pass | False | **False** |

Output: `artifacts/diversity_scout/birdnet_scoped_20260601T040851Z/diversity_scout_summary.json`

### Why scoped lift/q05 equal the full case
At w=0.05 the blend on the scored valid classes is driven by the 10 active∩valid columns
(identical BirdNET values in both streams); the 32 non-active valid columns are E in the
scoped stream and near-constant flat prior in the full stream, but at small weight and after
per-column ranking they contribute no measurable AUC change. So scoping **buys competence on
the scout's headline AUC** (which is dominated by active columns) **but cannot move the
robustness verdict**: the out-of-sample file bootstrap still goes negative because the
genuine lift cells overlap the rare Pantanal classes BirdNET does not detect.

## Decision and rationale

- **No gate path clears.** Promote-to-slot (a) needs site_q05 > 0 and file_q05 > 0 — both
  fail. Non-harmful (b) needs file_q05 ≥ 0 AND decorr > 0.3 — both fail (file_q05 −0.000482,
  decorr 0.177). The constraints' AND-logic is decisive.
- **Slot conserved.** Per the run's hard constraint (≤ 1 LB slot, no
  duplicate/static/harmful submission), and because the scoped blend is neither
  gate-clearing nor provably non-harmful, **no submission is queued**. The proven-unreliable
  proxy caveat (v610 clean AUC 0.967 → public 0.852) would justify a guarded LB shot for a
  *record-orthogonal* stream that fails only the proxy — but scoping **removed** the
  orthogonality (decorr 0.177), so that escape hatch no longer applies. This is now an
  ordinary redundant-competent stream that also shows a negative file bootstrap.
- **Net:** a clean negative result. Scoping confirmed the dilution diagnosis (competence is
  recoverable) but also confirmed it is **mutually exclusive with the orthogonality** that
  was the only reason to chase BirdNET as a foundation lever. The competent BirdNET signal is
  redundant with the 0.950 frontier; the orthogonal BirdNET signal is incompetent. No
  currently-accessible foundation has cleared the diversity gate.

## Compact summary

```
weak-class AUC (scoped)       = 0.918920   (full was 0.572896 — scoping fixed competence)
rank_decorrelation (scoped)   = 0.176605   (full was 0.869671 — orthogonality collapsed)
blend_best_weight / lift      = 0.05 / +0.000421
site_q05                      = 0.0        (p_gt_0 = 0.88)
file_q05                      = -0.000482  (negative → blend hurts leave-file-out)
DEV_score                     = 0.001161
decision                      = DEMOTE (competent-but-redundant + negative file-q05; no slot)
slot spent                    = NO   (ref: none — no submission queued)
```

## Reproduce

```bash
# Build scoped proxy: 78 active BirdNET cols injected; all other cols = frontier E (no flat priors)
.venv_scout/bin/python - <<'PY'  # (inline builder; see run log) -> birdnet_scoped_proxy.csv
PY
# Re-gate
.venv_scout/bin/python scripts/birdclef_diversity_scout.py \
  --proto-csv .../v644_yaroslav_0950/submission_protossm.csv \
  --sed-csv   .../v644_yaroslav_0950/submission_sed.csv \
  --candidate "birdnet_scoped=artifacts/diversity_scout/birdnet_proxy/birdnet_scoped_proxy.csv" \
  --candidate "birdnet_full=artifacts/diversity_scout/birdnet_proxy/birdnet_train_soundscape_proxy.csv" \
  --neg-weights --bootstrap 200 \
  --out-dir artifacts/diversity_scout/birdnet_scoped_20260601T040851Z
```
