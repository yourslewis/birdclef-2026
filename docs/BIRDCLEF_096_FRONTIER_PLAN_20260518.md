# BirdCLEF 0.96 Frontier Plan — 2026-05-18

Status: active after `v574`/`v575`/`v576` confirmed **0.949**.
Target: beat **0.960 public LB**.

## Current position

Confirmed best public LB is **0.949**:

- `v574`: guarded direct Nina EoS5 source replay — `0.949`
- `v575`: repo-owned EoS5 confirmation — `0.949`
- `v576`: repo-owned EoS5 Model5-only ablation — `0.949`

Interpretation:

- The EoS5 uplift is real and repo-owned reproducible.
- `v576` shows the top-level `Model_2` complement is not required; the score is carried by the EoS5 `Model_5` / Karnakbayev PowerOptimization path.
- The old public946 sidecar and train-audio student lanes are no longer credible primary paths to 0.960. They remain diagnostic assets only.

## Immediate slot policy

Do **not** spend a fresh slot on low-upside EoS5 scalar tuning unless explicitly approved or all higher-upside lanes are blocked.

This includes:

- `v577`: rank-aware power `0.6 -> 0.55`
- `v578`: `lambda_prior=0.5 -> 0.55`
- tiny top-level complement grids

Expected upside for these is likely around `0.949–0.951`, not enough for the new target.

## Primary track: public/source frontier discovery

The successful pattern was:

1. discover a structurally new public lineage (`nina2025/birdclef-2026-eos-5`),
2. preflight hidden-test and output-format safety,
3. run one guarded direct/source replay,
4. port to repo-owned confirmation.

Repeat that, but require stronger evidence than another EoS5 microvariant.

### Source scan method

Use Kaggle Bearer API v1 rather than unauthenticated CLI when needed:

- List visible BirdCLEF notebooks:
  `GET https://www.kaggle.com/api/v1/kernels/list?competition=birdclef-2026&pageSize=30&sortBy=scoreDescending`
- Pull source:
  `GET https://www.kaggle.com/api/v1/kernels/pull/<owner>/<slug>`

For each candidate, record:

- owner/slug/title/current version
- data, kernel, model sources
- source hash and source length
- top-level `solutions` block or equivalent final ensemble
- hidden-test path handling (`test_soundscapes/*.ogg`)
- dry-run/sample alignment guard
- final `submission.csv` verifier and row/column checks
- whether it is structurally distinct from EoS5 or a near clone

## 2026-05-18 first frontier scan

Pulled/summarized top visible public kernels into ignored artifacts under:

- `artifacts/public_kernels_20260518_frontier_refresh/`

High-signal observations:

### EoS5 clone / near-clone cluster

These are mostly not enough for 0.960 by themselves:

- `nina2025/birdclef-2026-eos-5` — known `0.949` source.
- `anthonytherrien/birdclef-2026-ensemble` — same EoS5-style `Model_2/Model_5` blend with tiny numeric differences.
- `kijiang/birdclef2026-v337` — EoS5-like source.
- `apachikoff/birdclef-2026-eos-5` — EoS5-like source.
- `beicicc/bc26-nina-eos5-may18`, `beicicc/bc26-s103-eos5-may18`, `beicicc/bc26-v63-nina-eos5-may18` — EoS5-like with `Model_2` weight around `0.04`.

Candidate action: do **not** submit clones blindly. Only inspect if they contain a real hidden-safe structural change not already covered by `v576`.

### SafeAlign/S106 variants

- `itshyao/birdclef-2026-s106-eos5-0949-safealign2`
- `beicicc/bc26-s106-eos5-sa2-may18`

Source features:

- EoS5-like source length around `330k` chars.
- Uses `Model_2` weight around `0.04` and `Model_5` dominant blend.
- Some sources include explicit `SafeAlign` markers.

Candidate action:

- Inspect diff versus our repo-owned v575/v576 before any slot.
- If SafeAlign is only output alignment / schema hardening, it is not a 0.96 candidate.
- If SafeAlign changes hidden-time aggregation or row alignment materially, prepare a repo-owned diagnostic. Otherwise skip.

### V6 / Model7 / BirdNET-PowerOptimization cluster

- `raunakdey07/birdclef-2026-v6`
- `youssefmo942009/birdclef-2026-v6`
- `yaroslavkholmirzayev/v6-0949-replay`
- `apachikoff/birdclef-2026-v6`
- `sunderekkiz/birdclef-2026-exp019-eos4-rank-power-06`
- `nina2025/birdclef-2026-eos-4`

Source features:

- Single `Model_7` style path, often labeled around `0.948`.
- BirdNET-heavy / PowerOptimization markers.
- We already tested some direct sibling behavior; not enough evidence for 0.960.

Candidate action:

- Deprioritize unless a newer variant claims/materially demonstrates >0.949.
- Do not submit direct BirdNET-heavy clones after previous BirdNET/V6-family drops/ties.

### Acoustic time-window / rank-fusion lane

- `pilkwang/birdclef-26-acoustic-time-window-rank-fusion`

Source features:

- `Karnakbayev_PowerOptimization_LB0948` as the single score-driving branch.
- Mentions acoustic time-window rank fusion.
- Smaller source and potentially distinct aggregation logic.

Candidate action:

- This is worth a deeper diff because it may encode a time-window/rank fusion idea we can port into EoS5 Model5 rather than submit as-is.
- First inspect whether it is simply a `0.948` single-branch replay. If yes, skip direct submission; extract any time-window logic only if distinct.

### Visual / UMAP / Mtoshi lane

- `mtoshidesu/birdclef-2026-visual-cpu-inference`
- `mtoshidesu/test-0-948`

Source features:

- Visual/UMAP/BirdNET-heavy markers.
- Prior direct Mtoshi-family submitted variants did not beat the plateau and one had resource/no-score issues.

Candidate action:

- Deprioritize for slots.
- Keep as possible idea mining only if source has a lightweight, source-clean aggregation trick.

## Candidate queue

1. **Diff SafeAlign/S106 against EoS5/v576**
   - Goal: determine if SafeAlign is real inference logic or only alignment hardening.
   - Submit only if structurally meaningful and hidden-safe.

2. **Inspect Pilkwang acoustic time-window rank fusion**
   - Goal: extract time-window/rank-fusion logic into EoS5 Model5 if distinct.
   - Avoid direct submission if it is just `0.948` PowerOptimization replay.

3. **Continue broader source scan for actual 0.96 lineage**
   - Search beyond top-30 visible notebooks and inspect latest forks/versions after EoS5.
   - Look for claims/metadata around `0.95`, `0.96`, `S114`, `SafeAlign`, `advanced ensemble`, `time window`, `rank fusion`, `EoS6`, or new datasets/model sources.

4. **Training lane only if source scan stalls**
   - New artifact must have stronger blend/correlation/stability evidence than failed sidecars.
   - Old local-positive sidecars are not approval evidence.

## Next run requirements

- Check live submissions and UTC slots.
- Confirm no stale v577/v578 submit monitor is alive.
- If a new slot is open, do not automatically submit v577.
- Work the source frontier first: SafeAlign/S106 diff, then acoustic time-window rank fusion diff, then broader 0.96 search.
