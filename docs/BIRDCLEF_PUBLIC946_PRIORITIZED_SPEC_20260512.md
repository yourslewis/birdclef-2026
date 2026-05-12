# BirdCLEF 2026 Public946 Prioritized Spec — 2026-05-12

Status: active planning spec  
Owner branch: `feature/v539-public946-replay` / PR #223  
Companion triage: `docs/BIRDCLEF_PUBLIC946_DIVERSITY_STREAM_TRIAGE_20260512.md`  
BirdNET fallback plan: `docs/BIRDCLEF_PUBLIC946_BIRDNET3_PORT_PLAN_20260512.md`  
Current scored anchor: **v539 = 0.943 public LB**  
Pending anchor candidates: **v541** complete/queued, **v542** complete/queued  
Deprecated premise: the old 0.927 plateau is no longer the search target.

---

## 0. Current state

### Scored leaderboard anchors

| Candidate | Public LB | Interpretation |
|---|---:|---|
| `v539` public946 replay baseline | **0.943** | New repo-owned anchor; validates public Perch/ProtoSSM + distilled SED rank-blend transfer. |
| `v527`, `v531`, `v537`, `v517` | 0.930 | Old internal tier; useful only as low-weight private-diversity diagnostics. |
| `v532`, `v526` | timeout / no score | Do not extend these runtime-risky lanes unless a clear fix is needed. |

### Live queue

1. `v541` — public946 replay with sonotype mirroring + rare-taxon adaptive thresholding.
   - Status: COMPLETE, verified, waiting for daily cap reset.
   - Purpose: restore public 0.946 postprocess paths omitted by v539.
2. `v542` — Afr1ste updated public946 V8 replay.
   - Status: COMPLETE and verified; queued after v541.
   - Verification refresh 2026-05-12 09:55 UTC: output files present; SED folds loaded; standard 60/40 rank blend executed; sonotype mirroring applied to 10 columns; rare thresholding applied to 44 species; full dry-run `submission.csv` shape `(240,235)` with no NaNs; runtime about 528s.
   - Purpose: controlled port of `afr1ste/birdclef-2026-0-946-updated-perch-sed`, which documents 0.946 V8 and 50/50 rank-blend ablations.
3. `v538` — old OOF-teacher B0 sidecar diagnostic.
   - Keep queued only after the public946 candidates; do not spend fresh work here unless it unexpectedly helps.

---

## 1. Priority order

### P0 — Score and lock the public946 anchor

**Goal:** determine whether v539, v541, or v542 should be the canonical public946 anchor.

**Actions:**

1. Let the monitor submit `v541` first at the next UTC reset.
2. Verify `v542` completion and output. If valid, keep it queued immediately after `v541`.
3. After both score, pick anchor by this rule:
   - If either scores `>=0.946`, make it canonical.
   - If both score around `0.943`, keep the higher of v539/v541/v542.
   - If v541/v542 underperform v539, freeze public-postprocess forks and move to ensemble/student work.

**Do not:** submit more public946 micro-forks before v541/v542 scores unless a pushed kernel fails and needs a minimal replacement.

---

### P1 — Public open-solution mining, but only for distinct signal

Public kernels inspected/found:

| Ref | Priority | Reason |
|---|---:|---|
| `afr1ste/birdclef-2026-0-946-updated-perch-sed` | DONE as v542 | Fresh updated 0.946 Perch+SED V8 source; close to v541 but preserves full train-row dry-run output and documents ablations. |
| `nina2025/birdclef-2026-ensemble-of-solutions-3` | Inspected; hold until v541/v542 score | Shows 0.946 via direct ensemble of Model_61/62. Offline reconstruction suggests the clean Model_61/62 idea is essentially a 50/50-ish public Proto/SED rank blend, not a new model stream. Avoid porting the kitchen-sink notebook unless v541/v542 miss and we need one clean v543 weight test. |
| `needless090/birdclef-2026-perch-sed-lb-0-946-clap` | Medium-high | Adds V5 and optional CLAP streams; true acoustic diversity but runtime/coverage risk. |
| `raunakdey07/birdclef-2026-birdnet-4-way-rank-blend` | Medium | Adds BirdNET + custom EffNet streams; potentially diverse but more model/data-source brittleness. |
| `mtoshidesu/testbirdclef-2026-ensemble-of-solutions-3` | Low/observe | Fork of Nina; only useful if it exposes a cleaner variant. |

**Nina mining result:**

1. Extracted the small testable idea: Model_61/Model_62 use two 0.946 variants with xSED weights around `0.54/0.46` and `0.46/0.54`, then direct-blend them.
2. Offline reconstruction with `scripts/birdclef_public946_weight_grid.py` on v542 dry-run rows shows the Model_61/62 direct proxy behaves almost like a 50/50 Proto/SED rank blend (`corr=0.993` vs v542 60/40) rather than a genuinely distinct stream.
3. On dry-run labels, SED-heavy weights look best (`proto0.40/sed0.60` AUC `0.994484`; 50/50/Nina proxy around `0.99362`; v542 60/40 AUC `0.992525`), but this likely reflects train-label leakage because public ablations say 50/50 tied 0.946 while 70/30 and 80/20 were lower.
4. If v541/v542 do not reach 0.946, package at most one clean `v543` weight test (`50/50` or SED-heavy `40/60`) without importing unrelated Nina model blocks. Do not port the full Nina notebook unless it contains a genuinely new high-signal stream and can run under the code-competition CPU budget.

---

### P2 — Ensemble on top of public946

**Principle:** public946 is the anchor. Old internal streams are minority diagnostics, not the base.

#### Candidate ensemble families

1. **Public-public blend** — highest near-term chance.
   - `v539/v541/v542` variants.
   - Rank-space blend or direct average only if outputs differ materially.
   - Submit only after v541/v542 scores.

2. **Public946 + V5/CLAP** — best diversity if sources are resolvable.
   - Use `needless090` fork as source; see companion diversity triage doc.
   - Required gate: log must show V5 sessions loaded and CLAP coverage; must not silently skip into plain 2-way public946.
   - Current audit: the extra V5/CLAP sources show as blank dataset refs in `GetKernel`; likely numeric datasetVersion IDs are visible in notebook JSON, but Bearer dataset lookup for likely slugs returns 403. Do not queue until source refs are resolved or recreated.
   - Abort if projected hidden runtime exceeds safe budget.

3. **Public946 + BirdNET/custom EffNet** — BirdNET-only path now cleaner than full 4-way.
   - Use `raunakdey07` fork for the 4-way idea and `claudedevore/birdclef-2026-r0946-birdnet-3way-submit` for a cleaner BirdNET-only reference; see companion diversity triage doc.
   - BirdNET source is resolved: `shadiakiki1/birdnet-analyzer/TfLite/birdnet_global_6k_v2.4_model_fp32-1/3`.
   - Custom EffNet source remains blocked by 403 notebook-output lookup; do not require it for the first source-clean BirdNET candidate.
   - Required gate: BirdNET TFLite and label mapping resolve; `submission_birdnet.csv` aligns; explicit 3-way rank blend is written; wall time stays safe.

4. **Public946 + internal 0.930 streams** — private robustness only.
   - Initial weights: public `0.98` + internal `0.02`, public `0.95` + internal `0.05`.
   - Candidate internal streams: `v517/v527` taxon gate, `v537/v538` OOF-teacher sidecar.
   - Submit at most one if local correlation/diversity looks favorable.

**Kill rule:** if an ensemble mostly reintroduces old v517 bias and lowers public score/class diversity, stop that branch.

---

### P3 — AutoResearch on the latest best model

**Goal:** create private-LB edge from public946, not just copy public notebooks.

#### Teacher cache

Use public946 outputs as the teacher source:

- `teacher_sed.npz` for local label fidelity.
- `teacher_rankblend.npz` for leaderboard-shaped target.
- Preserve raw Proto, SED, and final/rankblend streams whenever possible.

Current dry-run diagnostics from v541 on 190 train-overlap rows / 42 valid classes:

| Stream | Macro AUC | Top3 row recall | Note |
|---|---:|---:|---|
| Proto | 0.983987 | 0.6263 | Strong but not locally dominant. |
| SED | **0.995976** | **0.9895** | Best local teacher. |
| Rankblend | 0.992734 | 0.6421 | More LB-oriented than local-label optimized. |

#### Student experiments

Run small smoke first, then scale only if the smoke passes.

| Candidate | Priority | Config sketch | Success gate |
|---|---:|---|---|
| `public946-eca-nfnet-l0` | High | 5s or 10s, 160/256 mel, soft teacher, focal+BCE option | Low correlation to teacher or useful blend, not just high local AUC. |
| `public946-efficientnetv2-s` | High | 5s/10s, soft labels, external/pretrained init if available | Better diversity than B0 and packageable. |
| `public946-b0` | Done-ish baseline | B0 full 792-row student learned but is below teacher | Use only as sanity/sidecar unless it blends. |
| `public946-convnext-tiny` | Medium | Only if smoke correlation is low | Kill if it repeats prior V2S/ConvNeXt weak behavior. |

#### Knobs to sweep

- Teacher stream: SED-only vs rankblend vs mixed target.
- Teacher power: `0.85`, `1.0`, `1.15`.
- Crop length: `5s`, `10s`.
- Loss: BCE vs focal+BCE.
- Label smoothing: `0.0`, `0.01`, `0.03`.
- Mixup: `0.0`, `0.2`.
- Rare-class balancing: sqrt/equal weighting, class caps.

**Do not package** students whose only win is local AUC against train-soundscape labels with high correlation to public946. Package only if they add diversity or make a better private-risk blend.

---

## 2. Concrete next-run checklist

1. Keep queue order: `v541 -> v542 -> v538`.
   - `v541` and `v542` are both COMPLETE/no failure and verified.
   - Monitor pid `95675` is alive and sleeping on the daily cap after attempting `v541`.
2. Do not add another submission candidate until v541/v542 scores unless a queued candidate fails or the monitor dies.
3. Nina notebook mining is complete enough for now: Model_61/62 is effectively a 50/50-ish public rank-blend idea, not a new stream. Hold any `v543` until v541/v542 scores.
4. Public946 NFNet/V2S smokes are complete; keep `rankblend->NFNet 5s power1.0 ep20` as the only current student sidecar candidate.
5. If v541/v542 both miss, choose between one clean public weight test and one source-clean BirdNET-only 3-way rank-blend candidate; V5/CLAP remains blocked until source refs are resolvable. Use `docs/BIRDCLEF_PUBLIC946_BIRDNET3_PORT_PLAN_20260512.md` for the BirdNET port recipe if selected.
6. Update this spec after v541 and v542 scores land.

---

## 3. Submission-slot policy

Daily slots are now valuable. Use this ordering until updated:

1. `v541` — already complete, first pending.
2. `v542` — if complete and verified.
3. One genuinely distinct public ensemble fork (`v543`) only after v541/v542 results.
4. One public946+student/internal minority blend only after local diagnostics.
5. Old internal sidecars (`v538`, etc.) last.

No more old 0.930-axis micro-sweeps unless they are explicitly tied to public946 as a minority stream.
