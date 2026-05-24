# v611 anchored HGNet sidecar feasibility notes

Created from `docs/BIRDCLEF_096_ANCHORED_BLEND_IMPLEMENTATION_PLAN_20260524.md`.

## What is implemented

- Repo-owned kernel scaffold: `kaggle-kernels/v611-anchored-hgnet-sidecar/`.
- Anchor source: Samejima `birdclef-2026-visual-cpu-inference` pulled via Kaggle API and converted to script cells 1-14. The anchor final output is preserved as `submission_anchor_raw.csv`.
- Sidecar source: Praxel/Kosuke HGNet raw branch was inspected. The scaffold reimplements the Praxel/TTAhara OpenVINO `best_model_fold*.xml` inference path in streaming form instead of relying on public output files.
- Final blend: columnwise rank blend `0.94 * rank(anchor) + 0.06 * rank(submission_prax_hgnet_raw)`.
- Diagnostics/branch outputs: `submission_anchor_raw.csv`, `submission_prax_hgnet_raw.csv`, `submission_before_alignment.csv`, `submission.csv`.

## Feasibility judgment

Feasible as a hidden-safe scaffold for the exact low-weight Samejima-anchor + Praxel-HGNet-raw direction. It does not yet include Praxel `submission_blend_raw.csv`/`submission_pc010_raw.csv`, because those require running Praxel's separate ProtoSSM/SED branch; that is extra runtime and not necessary for the first hidden-safety probe.

## Blockers before any push/submit

1. Needs a Kaggle validation run to verify that `ttahara/birdclef-2026-hgnetv2-b0-baseline-inference` still exposes all four `best_model_fold*.xml/.bin` artifacts under `/kaggle/input`.
2. Runtime is unmeasured for full hidden size. The sidecar streams per audio and per fold to avoid v594-style hidden RAM failure, but OpenVINO + Samejima anchor may still be near the CPU time limit.
3. This scaffold intentionally does not push to Kaggle or submit; run only a private validation kernel first.
4. If the exact audit best is required, add Praxel `submission_pc010_raw.csv` and `submission_blend_raw.csv` as optional sidecars after this HGNet-only version passes hidden-safety/runtime.
