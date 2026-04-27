# BirdCLEF 2026 Kaggle Project

Canonical repository for Wenhao's BirdCLEF 2026 Kaggle competition work.

## Layout

- `kaggle-kernels/` — migrated Kaggle notebook/script variants and kernel metadata from Don's OpenClaw workspace.
- `scripts/` — local helper scripts for status/submission diagnostics.
- `docs/` — migration notes, submission notes, and operational context.

## Current known leaderboard anchors

Recent visible submissions before migration:

- `v111` — 0.922 LB, per-class temperature sharpening
- `v118` — 0.922 LB, mixup 0.3 + cutmix 0.25 + ensemble weight 0.6
- `v116` — 0.921 LB, SoftAUC loss + ensemble weight 0.6
- `v117` — 0.886 LB, larger GPU model d384/4L/8H
- `v113v3` — 0.895 LB, ProtoSSMv5 d256/3L/4H cross-attention

## GitHub workflow policy

This repo follows Wenhao's repo-management rule:

1. Work on feature/fix branches.
2. Open a pull request into `main`.
3. Do not merge into `main` without Wenhao's approval.
4. Default branch protection requires pull requests and code-owner review from `@yourslewis`.

## Kaggle code-competition submission note

BirdCLEF 2026 is a code competition: direct CSV/script upload is not accepted. The expected path is:

1. Push a Kaggle notebook/kernel that creates `submission.csv`.
2. Wait for the kernel version to complete successfully.
3. Submit that kernel version via `competition_submit_code(...)`.

The working auth path found during migration uses Kaggle API `2.x` with `KAGGLE_API_TOKEN` for the current `KGAT_*` token. The old Kaggle CLI/API `1.6.x` path returns `401 Unauthorized` with this token type.
