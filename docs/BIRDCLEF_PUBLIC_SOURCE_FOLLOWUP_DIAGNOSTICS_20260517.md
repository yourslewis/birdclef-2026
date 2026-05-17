# BirdCLEF Public Source Follow-up Diagnostics — 2026-05-17

Status: active stop-rule note
Anchor: current public best remains **0.946** from repo-owned/public946 anchors and tied variants (`v541`/`v542`/`v558`/`v563`/`v566`).

## Context

After the 2026-05-16 public-source sweep, we used 2026-05-17 UTC slots for a small controlled set of direct public-kernel submissions. The goal was to get distinct LB datapoints without spending all daily slots on correlated candidates.

The result is mixed-to-negative: direct public kernels can still tie, but hidden rerun/runtime behavior and output/rank-family correlation make them unsafe as a broad lane.

## Submission outcomes

| Variant | Source | Result | Lesson |
|---|---|---:|---|
| `v566` | `kruzzcc/bc26-nina-eos4-fixed` v2 | `0.946` | Tied plateau; useful confirmation but no improvement. |
| `v567` | `kruzzcc/bc26-mtoshi-umap-bn-a` v1 | `0.944` | Dropped; public BirdNET/UMAP direct path is weaker than repo-owned anchor. |
| `v568` | `meenalsinha/birdclef-2026-improved` v9 | no score / hidden rerun error | Public `submission.csv` preflight is not enough; hidden code rerun failed with generic unhandled error and `totalBytes=0`. |
| `v569` | `pilkwang/birdclef-2026-safe-ensemble` v4 | not submitted | Targeted preflight failed: `ListKernelFiles` returned 404. |
| `v570` | `mtoshidesu/lb-improved` v5 | no score / RAM error | Lower-priority fallback was submitted after v569 preflight failure; hidden rerun exceeded RAM. |

## Submitter hardening

The accidental lower-priority fallback to `v570` exposed a submitter safety issue. The submitter now supports:

- `--labels vXYZ` for exact targeted follow-ups;
- conservative stop-on-preflight-failure by default;
- explicit `--skip-preflight-failures` only when fallback behavior is intended.

Validation performed:

- `python3 -m py_compile scripts/submit_public_sweep_candidates_when_slots_available.py`
- `python3 scripts/submit_public_sweep_candidates_when_slots_available.py --help`
- targeted dry-run `--labels v569 --max-submissions 1`, which stops at the `v569` preflight 404 and does **not** fall through to `v570`;
- `git diff --check`.

## Decision

Stop direct public-kernel submissions for this lane unless there is materially stronger evidence or the kernel is repackaged into a repo-owned, hidden-rerun-safe candidate.

Do **not** submit more direct BirdNET / Mtoshi-family public notebooks just because a slot is idle. Preserve remaining slots for:

1. repo-owned/repackaged kernels with verified hidden/code-submission behavior;
2. genuinely new model/source signal from the 2025-style recipe lanes;
3. a targeted follow-up only if a candidate improves or there is new independent evidence.

## Next useful work

- If continuing public-source mining: port/repackage the idea rather than direct-submit the public notebook.
- If using GPU time: prefer external/pretraining, real SED/MIL targets, or another source-backed specialist over same-teacher micro-sidecars.
- If no GPU is available: prepare audits/specs and code hygiene; do not spend the last slot on low-confidence direct public notebooks.
