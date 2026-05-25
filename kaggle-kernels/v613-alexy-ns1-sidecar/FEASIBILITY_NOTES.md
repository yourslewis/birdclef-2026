# v613 Alexy NS1 sidecar feasibility notes — 2026-05-25 08UTC

Goal: convert `alexycactus/birdclef-2026-ns1-ensemble` into a hidden-safe repo-owned sidecar/private verifier rather than relying on direct replay.

## Live status

- Direct exploratory submission v613 scored `0.923`, far below the current `0.949` frontier. It is not a direct-slot family.
- Public preflight from 2026-05-24 showed a valid/nonconstant dry-run `submission.csv` with shape `192x235`, plus `submission_no_postproc.csv` and `diagnostics_nb21.json`.
- Log evidence says it uses five CNN checkpoints from `alexycactus/birdclef-2026-cnn-ns1-checkpoints`, runs a Perch+MLP stream, then blends `0.5*CNN + 0.5*(Perch+MLP)` and applies `TOP_K=1` postprocessing.

## 2026-05-25 access blocker

Current Kaggle Bearer API calls now return 403 for:

- `GetKernel` on `alexycactus/birdclef-2026-ns1-ensemble`
- `GetKernelSessionStatus`
- `ListKernelSessionOutput`
- `kaggle kernels pull`

The public web page is also behind reCAPTCHA for unauthenticated fetch. Because we cannot currently pull source or raw branch output, a repo-owned hidden-safe sidecar cannot be built without manually restoring source/code access or reimplementing the architecture from logs and checkpoint artifacts.

## Decision

Do **not** spend another competition slot on Alexy/NS1. It already scored `0.923`, and source/output access is blocked. Keep as an idea-mining lane only if source access becomes available.

Next better no-slot tracks while v616 is pending:

1. Per-class capped residual selector / OOF grid on v616 raw branches.
2. Fresh source/artifact scout for genuinely new 0.95/0.96 lineages.
3. Real SED/export smoke if no source candidate appears.
