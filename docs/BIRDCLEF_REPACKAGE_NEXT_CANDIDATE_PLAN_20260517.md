# BirdCLEF Repackaged Next-Candidate Plan — 2026-05-17

Status: draft / no-submit prep
Anchor: public best remains **0.946**; direct public-kernel lane is stopped after `v566` tie, `v567` drop, `v568` hidden error, and `v570` RAM failure.

## Why this exists

The 2026-05-17 direct public submissions proved that `submission.csv` preflight is not enough for code-competition safety:

- `v566` (`kruzzcc/bc26-nina-eos4-fixed`) tied `0.946`.
- `v567` (`kruzzcc/bc26-mtoshi-umap-bn-a`) dropped to `0.944`.
- `v568` (`meenalsinha/birdclef-2026-improved`) failed hidden rerun with `totalBytes=0`.
- `v569` (`pilkwang/birdclef-2026-safe-ensemble`) failed direct preflight (`ListKernelFiles` 404), so it was not submitted.
- `v570` (`mtoshidesu/lb-improved`) failed hidden rerun from RAM exhaustion.

Therefore the next useful public-source action is not another direct public submission. It is to repackage a specific idea into our own repo-owned kernel and verify hidden-safe behavior.

## Best remaining public-source idea to repackage

### Candidate: repo-owned `v571` safe xSED / stacker-inspired rank blend

Source inspiration: `pilkwang/birdclef-2026-safe-ensemble` (`v569` direct candidate).

Why this one, despite no direct submission:

- It was the more interesting remaining candidate before direct preflight failed.
- Prior output triage showed lower sample correlation vs Mtoshi (`corr≈0.872`) than the Mtoshi-identical v570 path.
- Sample MAE vs Mtoshi was bounded (`≈0.0081`), not a wild global shift.
- Logs mention an xSED rank blend around `0.5964 Proto / 0.4036 SED` and an LB-weighted safe blend line (`['0.928', '0.947']`, weights around `[0.037, 0.963]`).
- Its public log showed BirdNET unavailable / zero, meaning the useful signal is likely a Proto/SED/xSED stacker/rank recipe rather than BirdNET itself.

Why it must be repackaged:

- Direct file-list preflight 404 means the submitter cannot safely operate on the public kernel.
- The public notebook is large (`~270k` source chars) and RAM-heavy-looking; direct public notebooks have already failed hidden rerun.
- A repo-owned kernel can remove visuals, caches, broad notebooks cells, fallback ambiguity, and unnecessary outputs.

## Proposed implementation path

Start from our known hidden-safe repo-owned public946 implementation:

- `kaggle-kernels/v542-afr1ste-updated-public946/script.py`

Create a new repo-owned kernel directory, e.g.:

- `kaggle-kernels/v571-public946-safe-xsed-rankblend/`

Minimal change target:

1. Keep the v542 input discovery, sample submission alignment, Perch/ProtoSSM branch, SED branch, and final CSV schema unchanged.
2. Replace only the final rank-blend formula with a small set of source-backed recipes:
   - baseline `0.60 Proto / 0.40 SED` (for exact compatibility check);
   - xSED-style `0.5964 Proto / 0.4036 SED`;
   - optional safe weighted blend if it can be reproduced from source code without output-only leakage.
3. Do **not** add BirdNET unless the model/data path is repo-owned and memory-safe; v566 already showed BirdNET-family direct routes tie at best.
4. Remove plotting/markdown-only notebook behavior and avoid writing large intermediate caches unless needed.
5. Keep a strong output verifier in-kernel:
   - `submission.csv` exists;
   - row count equals hidden/test row count × 12 windows (or dry-run expected rows);
   - columns exactly match `sample_submission.csv`;
   - finite numeric values only;
   - no fallback to 3-row sample output in formal submit mode.

## Smoke / validation bar before using the last slot

Do not submit `v571` unless all of these pass:

1. Local or Kaggle public dry-run completes and writes a competition-shaped `submission.csv`.
2. Public dry-run output is close to v542 for the baseline formula and the xSED variant shifts predictions only modestly.
3. Runtime and memory are no worse than v542/v566-safe levels; no large hidden RAM risk.
4. Source diff is limited and reviewable: final blend + verifier only, not a wholesale notebook import.
5. If possible, compare 3-row public-output correlation against the saved `pilkwang` output to confirm the intended formula is reproduced.

## Kill criteria

Kill the repackage if:

- it requires output-only public predictions rather than recomputing from allowed inputs;
- it relies on unavailable public-kernel file outputs;
- it exceeds v542 runtime/memory materially;
- it cannot reproduce the intended xSED/safe-blend behavior on sample rows;
- it is only another micro-weight tweak without a source-backed structural difference.

## Slot policy

Preserve the remaining 2026-05-17 slot until a repo-owned/repackaged candidate clears the above bar. Do not spend it on another direct public notebook.
