# BirdCLEF 2026 0.96 Frontier — Hidden-Safe Anchored Blend Implementation Plan

_Last updated: 2026-05-24 10:20 UTC_

## Objective

Prepare a repo-owned candidate only if it can rerun on hidden `test_soundscapes` and preserve the confirmed `0.949` plateau while adding a small, structurally distinct sidecar signal. Do **not** submit public-output wrappers or sample-aligned dry-run CSVs.

Current confirmed best: `0.949`. Target: `0.960`.

## Why this exists

The 08UTC anchored movement audit found positive local movement when small Praxel/Jungchan sidecars were rank-blended into a Samejima visual/plateau anchor:

- Samejima visual anchor local AUC: `0.9903905`, top3 row recall `0.4526` on `190` matched train-soundscape rows / `42` valid classes.
- Praxel-only best: `prax_hgnet=0.06`, `prax_blend=0.02`, `prax_pc010=0.02` → local AUC `0.9935637` (`+0.00317`), top3 `0.6263`, corr `0.99844`, MAE `0.0140`.
- Jungchan+Praxel best: `prax_hgnet=0.06`, `jung21=0.04` → local AUC `0.9936394` (`+0.00325`), top3 `0.6368`, corr `0.99834`, MAE `0.0150`.
- S14 sidecar best: `s14=0.20` → local AUC `0.9931105` (`+0.00272`), top3 `0.4789`, corr `0.99743`, MAE `0.0141`.

Treat this as **rejection-only evidence**. Prior local-positive lanes (v560/v573/v610) did not transfer. The audit is useful only for prioritizing a hidden-safe implementation plan.

## Source/artifact map

### Anchor: Samejima visual CPU inference

Ref: `samejimatink0/birdclef-2026-visual-cpu-inference` v14.

Metadata:

- Datasets: `tuckerarrants/bc2026-distilled-sed-public`, `mtoshidesu/birdclef-flow-diagram`, `jaejohn/perch-meta`, `tuckerarrants/perch-v2-no-dft-onnx`, `rishikeshjani/perch-onnx-for-birdclef-2026`.
- Kernel source: `ashok205/tf-wheels`.
- Model: `google/bird-vocalization-classifier/TensorFlow2/perch_v2_cpu/1`.
- Outputs include `submission.csv`, `submission_birdnet.csv`, `submission_protossm.csv`, `submission_sed.csv`, and `train8_models/*` artifacts.
- Hidden-test path exists: source discovers `BASE / "test_soundscapes"`; dry-run falls back to train files only when hidden test is absent.

Role:

- Use as anchor behavior/reference, not as a new direct submission. Samejima/visual family is already covered by v608 `0.949` and is highly correlated with v608.

### Sidecar A: Praxel HGNet/raw branch

Ref: `praxel/birdclef-2026-kosuke-v15-hgnet` v2.

Metadata:

- Datasets: `tuckerarrants/bc2026-distilled-sed-public`, `skidive/no-wav-use`, `jaejohn/perch-meta`, `rishikeshjani/perch-onnx-for-birdclef-2026`.
- Kernel sources: `ashok205/tf-wheels`, `ttahara/birdclef-2026-download-wheels`, `ttahara/birdclef-2026-hgnetv2-b0-baseline-inference`.
- Model: `google/bird-vocalization-classifier/TensorFlow2/perch_v2_cpu/1`.
- Outputs include valid full `240x235` raw branch files: `submission_hgnet_raw.csv`, `submission_blend_raw.csv`, `submission_pc010_raw.csv`, plus final sample-aligned `submission.csv`.
- Source robustly searches `/kaggle/input/**/best_model_fold0.xml` and offline wheel requirements; dry-run sample-aligns only final `submission.csv`, while raw branch files preserve real row IDs.

Role:

- Best local sidecar source. Candidate weights from audit: `HGNet 0.06` plus optional `blend_raw 0.02` and `pc010 0.02`.
- Must be rerun on hidden test via code, not copied from public output.

Risks:

- Prior standalone HGNet inference v598 scored `0.860`; HGNet must remain a low-weight sidecar anchored to a 0.949-family output.
- Need verify hidden runtime with OpenVINO/HGNet plus anchor stays under limit.

### Sidecar B: Jungchan Model21 / EoS branch

Ref: `jungchanryu/birdclef-first` v19 / v18 lineage.

Metadata:

- Datasets: `tuckerarrants/bc2026-distilled-sed-public`, `tuckerarrants/birdclef-2026-waveform-cache`, `jaejohn/perch-meta`, `tuckerarrants/perch-v2-no-dft-onnx`, `rishikeshjani/perch-onnx-for-birdclef-2026`, `hideyukizushi/sgkfk-202604041716`.
- Kernel sources: `ashok205/tf-wheels`, `hideyukizushi/bird26-reprod-perch-proto-residualssm-train-s7177`.
- Model: `google/bird-vocalization-classifier/TensorFlow2/perch_v2_cpu/1`.
- v19 outputs: `subm_21.csv`, `subm_52p.csv`, `submission_protossm.csv`, `submission_sed.csv`, final sample-shaped `submission.csv`.
- Source solution weights are anchor-heavy: Model21 LB `0.928`, Model52/74 LB `0.949`; public final is not direct-slot-worthy.

Role:

- Low-weight sidecar only. Best local audit used `jung21=0.04` with `prax_hgnet=0.06`.

Risks:

- Direct EoS/Jungchan family is saturated and mostly plateau duplication.
- Public final is sample-shaped/constant in dry-run; only branch-generation code can be used.

### Sidecar C: StudyExchange S14

Ref: `studyexchange/birdclef-2026-infer-s14` v26.

Metadata:

- Datasets: `tuckerarrants/bc2026-distilled-sed-public`, `tsubasatech/birdclef-2026-snowflake-sed`, `rishikeshjani/perch-onnx-for-birdclef-2026`.
- Output valid `240x235`, no failure; runtime about `669s`; dry-run OOF AUC `0.991722`.
- Source expectation is only `~0.943 -> 0.946+`, not 0.96.

Role:

- Hold as code-mining / optional low-weight sidecar. Not direct slot-worthy.

## Candidate designs

### Candidate P1 — Samejima anchor + Praxel HGNet low-weight sidecar

Blend in rank space:

```text
final = 0.92 * rank(anchor) + 0.06 * rank(praxel_hgnet_raw) + 0.02 * rank(praxel_blend_raw)
```

Optional third sidecar if runtime/implementation is already free:

```text
+ 0.02 * rank(praxel_pc010_raw)
```

Preferred because it uses the strongest local sidecar and avoids adding saturated EoS branches.

### Candidate P2 — Samejima anchor + Praxel HGNet + Jungchan Model21

Blend in rank space:

```text
final = 0.90 * rank(anchor) + 0.06 * rank(praxel_hgnet_raw) + 0.04 * rank(jungchan_subm_21)
```

Highest local AUC movement, but more complex and more EoS-family overlap.

### Candidate P3 — Samejima anchor + S14 low-weight sidecar

Blend in rank space:

```text
final = 0.80 * rank(anchor) + 0.20 * rank(s14_output)
```

Lower priority. S14 source expectation is below current best, and standalone local AUC is weaker than Praxel/Jungchan sidecars.

## Implementation rules

1. Start from a fresh clone/worktree; do not use the broken `/Users/yourslewis/Documents/birdclef-2026-v545` git metadata for commits.
2. Create a repo-owned kernel directory, e.g. `kaggle-kernels/v611-anchored-hgnet-sidecar/`.
3. Reuse known hidden-safe anchor code from Samejima/v608 family and preserve its final row/column guards.
4. Add Praxel HGNet branch by copying only the minimal OpenVINO/HGNet inference path needed to produce `submission_hgnet_raw.csv` on the same hidden test rows.
5. Rank-normalize each branch columnwise and blend with fixed small weights.
6. Write diagnostic branch files:
   - `submission_anchor_raw.csv`
   - `submission_prax_hgnet_raw.csv`
   - optional `submission_prax_blend_raw.csv`
   - optional `submission_jung21_raw.csv`
   - `submission_before_alignment.csv`
   - final `submission.csv`
7. If hidden test is absent, dry-run may use train soundscapes, but final public notebook output must still pass shape/finite/non-constant checks. Never submit a fixed sample-output fallback.
8. Hard-fail instead of silently dropping sidecars if a required artifact/model is missing.

## Required pre-submit gates

A v611-style candidate is slot-eligible only if all gates pass:

- Kernel COMPLETE with no failure.
- Final `submission.csv` has exactly the competition columns, finite values, no all-constant columns, and valid row count for the environment.
- Raw branch outputs exist and row-align with the anchor before any sample alignment.
- Logs show hidden-test branch will execute when `/test_soundscapes` is mounted.
- Runtime is comfortably below competition limit; avoid v609-style TTA/training timeout.
- Delta vs anchor is bounded: corr should remain high (`>=0.997` for the preferred low-weight candidate) and MAE around `0.01–0.02`, not a standalone HGNet takeover.
- No local metric is treated as approval; it is only a rejection screen.

## Current decision

Do **not** submit yet. The audit identifies Praxel HGNet/raw + optional Jungchan Model21 as the best extraction direction, but the next step is code implementation and hidden-safety preflight, not a direct public-source replay.
