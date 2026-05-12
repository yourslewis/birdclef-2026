# BirdCLEF 2026 Public946 Prioritized Spec — 2026-05-12

Status: active planning spec  
Owner branch: `feature/v539-public946-replay` / PR #223  
Current scored anchor: **v539 = 0.943 public LB**  
Pending anchor candidates: **v541** complete/queued, **v542** running at creation time  
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
   - Status at spec creation: pushed as `yourslewis/bc26-v542-afr1ste-updated-public946`, version 1, running.
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
| `nina2025/birdclef-2026-ensemble-of-solutions-3` | High, inspect-only next | Shows 0.946 via direct ensemble of Model_61/62 and includes many public model blocks. Mine weights/ideas; avoid porting the entire kitchen-sink notebook unless sources/runtime are clean. |
| `needless090/birdclef-2026-perch-sed-lb-0-946-clap` | Medium-high | Adds V5 and optional CLAP streams; true acoustic diversity but runtime/coverage risk. |
| `raunakdey07/birdclef-2026-birdnet-4-way-rank-blend` | Medium | Adds BirdNET + custom EffNet streams; potentially diverse but more model/data-source brittleness. |
| `mtoshidesu/testbirdclef-2026-ensemble-of-solutions-3` | Low/observe | Fork of Nina; only useful if it exposes a cleaner variant. |

**Nina mining plan:**

1. Extract only the small, testable idea first: Model_61/Model_62 use two 0.946 variants with xSED weights around `0.54/0.46` and `0.46/0.54`, then direct-blend them.
2. Check whether this reduces to approximately a 50/50 public rank blend. Afr1ste notes 50/50 tied 0.946, while 70/30 and 80/20 fell to 0.944/0.942.
3. If v541/v542 do not reach 0.946, package one clean `v543` that tests the best Nina/Afr1ste weight insight without importing unrelated models.
4. Only port the full Nina ensemble if it contains a genuinely new high-signal stream and can run under the code-competition CPU budget.

---

### P2 — Ensemble on top of public946

**Principle:** public946 is the anchor. Old internal streams are minority diagnostics, not the base.

#### Candidate ensemble families

1. **Public-public blend** — highest near-term chance.
   - `v539/v541/v542` variants.
   - Rank-space blend or direct average only if outputs differ materially.
   - Submit only after v541/v542 scores.

2. **Public946 + V5/CLAP** — best diversity bet.
   - Use `needless090` fork as source.
   - Required gate: log must show V5 sessions loaded and CLAP coverage; must not silently skip into plain 2-way public946.
   - Abort if projected hidden runtime exceeds safe budget.

3. **Public946 + BirdNET/custom EffNet** — diversity with higher fragility.
   - Use `raunakdey07` fork as source.
   - Required gate: BirdNET TFLite and custom EffNet source mount resolve; output rows align; wall time stays safe.

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

1. Poll `v542` until COMPLETE/ERROR.
   - If COMPLETE, verify output files/log:
     - SED folds loaded.
     - rank blend executed.
     - sonotype/rare paths executed.
     - output row/column shape sane.
   - Keep queue order: `v541 -> v542 -> v538`.
2. Do not add another submission candidate until v541/v542 scores unless v542 fails.
3. Inspect Nina notebook narrowly:
   - extract Model_61/62 definitions and final direct/rank ensemble math.
   - decide if there is a clean `v543` or if it is redundant with v541/v542.
4. Start a public946 NFNet or EfficientNetV2-S smoke as a background GPU job only after confirming no urgent Kaggle failure needs attention.
5. Update this spec after v541 and v542 scores land.

---

## 3. Submission-slot policy

Daily slots are now valuable. Use this ordering until updated:

1. `v541` — already complete, first pending.
2. `v542` — if complete and verified.
3. One genuinely distinct public ensemble fork (`v543`) only after v541/v542 results.
4. One public946+student/internal minority blend only after local diagnostics.
5. Old internal sidecars (`v538`, etc.) last.

No more old 0.930-axis micro-sweeps unless they are explicitly tied to public946 as a minority stream.
