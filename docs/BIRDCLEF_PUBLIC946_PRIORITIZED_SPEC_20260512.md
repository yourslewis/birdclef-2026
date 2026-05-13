# BirdCLEF 2026 Public946 Prioritized Spec — 2026-05-12

Status: active execution spec
Merged spec branch: `feature/v539-public946-replay-resolved` / PR #224
Current update branch: `feature/v543-public946-birdnet3`
Companion triage: `docs/BIRDCLEF_PUBLIC946_DIVERSITY_STREAM_TRIAGE_20260512.md`
BirdNET fallback plan: `docs/BIRDCLEF_PUBLIC946_BIRDNET3_PORT_PLAN_20260512.md`
Canonical scored anchor: **v541 = 0.946 public LB**
Independent confirmation: **v542 = 0.946 public LB**
Deprecated premise: the old 0.927 plateau is no longer the search target.

---

## 0. Current state

### Scored leaderboard anchors

| Candidate | Public LB | Interpretation |
|---|---:|---|
| `v541` public946 mirror/rare replay | **0.946** | Canonical repo-owned public946 anchor; restores sonotype mirroring and rare-taxon thresholding omitted by v539. |
| `v542` Afr1ste updated public946 V8 replay | **0.946** | Independent confirmation of the public946 Perch+SED V8 stack; preserves full train-row dry-run output and documented ablations. |
| `v543` public946 + BirdNET 3-way | 0.946 | Source-clean BirdNET 6K minority stream: Proto/SED/BirdNET rank blend 52/38/10; tied the anchor, ref `52600158`. |
| `v544` public946 + BirdNET 5% | pending | Safer follow-up from local grid: Proto/SED/BirdNET rank blend 56/39/5; submitted ref `52603058`. |
| `v539` public946 replay baseline | 0.943 | Validated baseline transfer, but superseded by v541/v542. |
| `v527`, `v531`, `v537`, `v538`, `v517` | 0.930 | Old internal tier; useful only as low-weight private-diversity diagnostics. |
| `v532`, `v526` | timeout / no score | Do not extend these runtime-risky lanes unless a clear fix is needed. |

### Queue state after 2026-05-13 UTC reset

1. `v541` — public946 replay with sonotype mirroring + rare-taxon adaptive thresholding.
   - Status: COMPLETE/scored `0.946`, ref `52594869`.
   - Purpose: canonical public946 anchor.
2. `v542` — Afr1ste updated public946 V8 replay.
   - Status: COMPLETE/scored `0.946`, ref `52594882`.
   - Verification refresh 2026-05-12 09:55 UTC: output files present; SED folds loaded; standard 60/40 rank blend executed; sonotype mirroring applied to 10 columns; rare thresholding applied to 44 species; full dry-run `submission.csv` shape `(240,235)` with no NaNs; runtime about 528s.
   - Purpose: independent confirmation of `afr1ste/birdclef-2026-0-946-updated-perch-sed`, which documents 0.946 V8 and 50/50 rank-blend ablations.
3. `v538` — old OOF-teacher B0 sidecar diagnostic.
   - Status: COMPLETE/scored `0.930`, ref `52594896`.
   - Interpretation: confirms the old internal OOF-teacher sidecar remains a 0.930-tier diagnostic, not a new anchor.
4. `v543` — public946 + source-clean BirdNET 6K 3-way rank blend.
   - Status: COMPLETE/scored `0.946`, ref `52600158`.
   - Verification refresh 2026-05-13 04:10 UTC: BirdNET model source resolved, `157/234` labels mapped, `submission_birdnet.csv` written, final `submission.csv` shape `(240,235)` with no NaNs, 3-way blend `52/38/10`, runtime about `546s`.
   - Interpretation: safe tie, but not an improvement.
5. `v544` — public946 + source-clean BirdNET 5% 3-way rank blend.
   - Status: kernel COMPLETE/no failure and submitted as ref `52603058`; score pending at first post-submit poll.
   - Verification refresh 2026-05-13 05:56 UTC: output files present, final `submission.csv` shape `(240,235)` with no NaNs, 3-way blend `56/39/5`, runtime about `556s`.
   - Purpose: smaller BirdNET perturbation selected from local grid after v543 tied the anchor.

---

## 1. Priority order

### P0 — Score and lock the public946 anchor — DONE

**Result:** `v541` and `v542` both scored **0.946 public LB** after the 2026-05-13 UTC reset. Treat `v541` as the canonical repo-owned public946 anchor and `v542` as independent confirmation.

**Locked decisions:**

1. Supersede `v539` (0.943) with `v541`/`v542` (0.946).
2. Do not spend a slot on the previously discussed clean `v543` public weight test unless later diagnostics show a clearly distinct public-public blend.
3. Freeze old 0.930-axis internal sidecars as diagnostics only; `v538` scored 0.930 and did not create a new anchor.

**Next:** move from anchor reproduction to distinct-signal work: teacher-cache/student experiments, carefully gated BirdNET/V5/CLAP diversity, or locally justified public-public blend diagnostics.

---

### P1 — Public open-solution mining, but only for distinct signal

Public kernels inspected/found:

| Ref | Priority | Reason |
|---|---:|---|
| `afr1ste/birdclef-2026-0-946-updated-perch-sed` | DONE as v542; scored 0.946 | Fresh updated 0.946 Perch+SED V8 source; confirms the public946 anchor. |
| `nina2025/birdclef-2026-ensemble-of-solutions-3` | Inspected; no immediate slot | Shows 0.946 via direct ensemble of Model_61/62. Offline reconstruction suggests the clean Model_61/62 idea is essentially a 50/50-ish public Proto/SED rank blend, not a new model stream. Since v541/v542 reached 0.946, do not spend the next slot on a plain public weight clone unless diagnostics show material divergence. |
| `needless090/birdclef-2026-perch-sed-lb-0-946-clap` | Medium-high | Adds V5 and optional CLAP streams; true acoustic diversity but runtime/coverage risk. |
| `raunakdey07/birdclef-2026-birdnet-4-way-rank-blend` | Medium | Adds BirdNET + custom EffNet streams; potentially diverse but more model/data-source brittleness. |
| `mtoshidesu/testbirdclef-2026-ensemble-of-solutions-3` | Low/observe | Fork of Nina; only useful if it exposes a cleaner variant. |

**Nina mining result:**

1. Extracted the small testable idea: Model_61/Model_62 use two 0.946 variants with xSED weights around `0.54/0.46` and `0.46/0.54`, then direct-blend them.
2. Offline reconstruction with `scripts/birdclef_public946_weight_grid.py` on v542 dry-run rows shows the Model_61/62 direct proxy behaves almost like a 50/50 Proto/SED rank blend (`corr=0.993` vs v542 60/40) rather than a genuinely distinct stream.
3. On dry-run labels, SED-heavy weights look best (`proto0.40/sed0.60` AUC `0.994484`; 50/50/Nina proxy around `0.99362`; v542 60/40 AUC `0.992525`), but this likely reflects train-label leakage because public ablations say 50/50 tied 0.946 while 70/30 and 80/20 were lower.
4. Since v541/v542 reached 0.946, do **not** package the simple Nina-style `v543` weight test as the next submission. Keep it as a fallback diagnostic only if future public-public output comparisons show material divergence without added runtime risk.

---

### P2 — Ensemble on top of public946

**Principle:** public946 is the anchor. Old internal streams are minority diagnostics, not the base.

#### Candidate ensemble families

1. **Public-public blend** — diagnostic only unless outputs differ materially.
   - `v541/v542` are both 0.946; `v539` is 0.943 and may add only minor postprocess diversity.
   - Rank-space blend or direct average only if offline output comparison shows nontrivial divergence and no runtime/submission-risk increase.
   - Do not spend the next slot on a public-public clone by default; prefer distinct signal.

2. **Public946 + V5/CLAP** — best diversity if sources are resolvable.
   - Use `needless090` fork as source; see companion diversity triage doc.
   - Required gate: log must show V5 sessions loaded and CLAP coverage; must not silently skip into plain 2-way public946.
   - Current audit: the extra V5/CLAP sources show as blank dataset refs in `GetKernel`; likely numeric datasetVersion IDs are visible in notebook JSON, but Bearer dataset lookup for likely slugs returns 403. Do not queue until source refs are resolved or recreated.
   - Abort if projected hidden runtime exceeds safe budget.

3. **Public946 + BirdNET/custom EffNet** — BirdNET-only path now cleaner than full 4-way.
   - `v543` (`yourslewis/bc26-v543-public946-birdnet-3way`) used Proto `0.52` / SED `0.38` / BirdNET `0.10` and tied 0.946.
   - Current candidate: `v544` (`yourslewis/bc26-v544-public946-birdnet-5pct`) uses the local-grid-favored smaller BirdNET perturbation: Proto `0.56` / SED `0.39` / BirdNET `0.05`; kernel COMPLETE and submitted, score pending.
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

1. Treat `v541`/`v542` at 0.946 as the locked public946 anchor pair.
2. Do not queue a plain public weight clone next; the simple Nina/50-50 idea is now a fallback diagnostic, not the main path.
3. Prefer one of these distinct-signal next moves:
   - build/refresh a public946 teacher cache from v541/v542 outputs and run the next student-diversity smoke (`rankblend->NFNet 5s power1.0 ep20` remains the current sidecar candidate),
   - source-clean BirdNET-only 3-way rank-blend if model/label mapping and wall-time gates pass,
   - V5/CLAP only if its blocked source refs are resolved or recreated.
4. If spending a Kaggle slot, require a pre-submit gate: output alignment, no silent fallback, and clear distinction from v541/v542.
5. Keep old 0.930 internal streams as minority diagnostics only; `v538` scored 0.930 and should not be extended directly.

---

## 3. Submission-slot policy

Daily slots are now valuable. Use this ordering until updated:

1. One genuinely distinct public946 diversity candidate only after local gates pass (BirdNET-only 3-way or resolved V5/CLAP, not a silent 2-way fallback).
2. One public946+student/internal minority blend only after local diagnostics show useful divergence.
3. Public-public weight clone (`v543`-style 50/50 or 40/60) only as a fallback if output comparisons justify it.
4. Old internal sidecars (`v538`, etc.) last.

No more old 0.930-axis micro-sweeps unless they are explicitly tied to public946 as a minority stream.
