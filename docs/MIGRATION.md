# Migration Notes

Date: 2026-04-27

## Source

Migrated from Don's OpenClaw workspace:

- Source path: `/Users/yourslewis/.openclaw/workspace-don/kaggle/birdclef-2026`
- Canonical path: `/Users/yourslewis/Documents/birdclef-2026`
- GitHub remote: <https://github.com/yourslewis/birdclef-2026>

## What was migrated

- BirdCLEF 2026 Kaggle kernel/script variants under `kaggle-kernels/`.
- Useful local helper scripts under `scripts/`.
- Current project/submission context in `README.md` and this file.

## What was intentionally not migrated

- Python virtual environments.
- Local Kaggle credentials and token files.
- Large generated data/model/output artifacts.
- The unrelated 2.8GB `playground-series-s6e3` working tree.
- Simulated one-off submit scripts from the OpenClaw workspace root, except the legacy real-submit attempt kept as diagnostic context.

## History/context preservation

The source BirdCLEF tree was not meaningfully tracked in a standalone Git repository, so there was no clean prior git history to preserve. The migration preserves practical context by retaining versioned experiment folders (for example `v111`, `v193`, and `v200`-`v236`) and documenting the pre-migration leaderboard anchors.
