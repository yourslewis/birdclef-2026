# BirdCLEF 2026 0.96 Promising Recipes AutoResearch Plan — 2026-05-23

Current public best: **0.949**.
Target: **0.960+**.

This plan operationalizes the post-0.949 pivot: stop spending slots on clone/postprocess lanes and focus on artifact-backed new signal.

## Summary of current evidence

- v599-v602 tested R0952 / Pilkwang / Meenal visual-prior / NFNet public lanes; all tied `0.949`.
- v604 tested the Pilkwang PCEN sidecar as a repo-owned verifier; it tied `0.949`.
- v595 tested the `public0952`/Exp070 Perch-probe lane and scored `0.899`; treat that title as non-reproduced.
- v607 is the latest repaired Eslam v26C verifier; it completed private-kernel output preflight with valid `submission.csv` and has been submitted as the next distinct source test.

## Primary hypothesis

The public 0.949 stack is saturated because most public forks share the same dominant Perch/ProtoSSM/Karnak rank-power anchor. The most credible route above 0.949 is not another scalar; it is a missing or newly trained sidecar artifact with enough independent signal to reorder top classes.

## Recipe 1 — G124/S124 reconstruction

### Why it matters

The only credible >0.949 public claim found so far is the S124/G124 line around `0.952`.

Public source points to a private asset dataset:

- `itshyao/birdclef2026-g124-effv2s-2025pre-pseudo-assets`

API status:

- direct dataset view: `403 datasets.get denied`
- dataset search by slug / G124 terms: no public hit

Likely artifact contents:

- `g124_fold1_fp16.pt`
- `_best.pt`
- `submission_g124_effv2s_fold1_s124.csv`
- EfficientNetV2-S sidecar training/inference utilities

### Action plan

1. Keep searching for exact artifact filenames and forks.
2. Build a repo-owned G124 approximation:
   - EffNetV2-S backbone
   - 5s mel windows, 160 mel bins first
   - pseudo-label teacher from current 0.949 anchor + public946 cache
   - optional 2025/external init using existing repo configs
3. Export a Kaggle-fit checkpoint and sidecar CSV.
4. Apply S124 rank-blend with strict movement guards.
5. Submit only after blend audit shows a meaningful change versus the 0.949 anchor.

### Initial specs to implement

- smoke: one fold, small class/sample subset, verify training/export/inference
- pilot: one fold, all classes, 8-12 epochs
- full: 5 folds only if pilot sidecar is promising

## Recipe 2 — Eslam v26C repaired public-source verifier

### Current state

- v605 failed: `proto_model` undefined
- v606 failed: missing `submission_protossm.csv`
- v607 fixed both and completed output preflight:
  - outputs: `submission.csv`, `submission_protossm.csv`, `submission_sed.csv`, cache files
  - no traceback
  - `submission.csv` stats: `240x235`, finite, non-constant, unique row IDs

### Caveat

The original source references optional student ONNX folds from `eslamelokpy/birdclef2026-student-onnx`, but Kaggle rejected that dataset source. v607 falls back to a 2-way ProtoSSM + SED blend.

### Decision rule

- If v607 improves: port/tune Eslam lineage and investigate student ONNX recreation.
- If v607 ties: mark safe but plateau; do not keep patching without student artifact.
- If v607 drops/no-scores: kill Eslam repair lane.

## Recipe 3 — Fresh source/artifact scout

Run each loop:

- Kaggle code search for `0.960`, `0.96`, `0.952`, `0.95`, `g124`, `s124`, `v26c`, `protossm`, `new submission`.
- Pull with Bearer API v1.
- Extract:
  - dataset/kernel/model sources
  - exact `/kaggle/input` paths
  - checkpoint filenames
  - final writer behavior
  - dry-run/sample fallback behavior

Promote only if:

- source is hidden-test safe;
- output writer is competition-format;
- dependencies are public/attachable or reproducible;
- lineage is not a clone of PCEN/EoS/NFNet/R0952 plateau family.

## Explicitly deprioritized

- PCEN forks after v604 tied `0.949`
- EoS5/EoS6 scalar power/rank tweaks
- NFNet LPrior clones after v602 tied `0.949`
- `public0952` Perch-probe clones after v595 scored `0.899`
- LLM-labeling path: skipped because no audio-capable labeler is available

## Open questions

1. Can the private G124 asset be found under another fork/owner?
2. Can a repo-owned EffV2-S sidecar reproduce enough of G124 to beat 0.949?
3. Does v607's repaired 2-way Eslam path score above plateau?
4. Are JGuevara `08-winning-tta-submission-pipeline` or Soundscape Finetune posts source-safe and artifact-backed, or just training notebooks?

## 2026-05-23 08:15 UTC concrete recipe specs

Added concrete G124 reconstruction configs:

- `configs/birdclef/g124_effv2s_public946_pseudo_smoke_20260523.json`
- `configs/birdclef/g124_effv2s_public946_pseudo_pilot_20260523.json`

Fresh public source audit conclusion:

- S14 fresh-Perch source is structurally interesting but self-reported below the 0.949 frontier, so hold for idea-mining.
- Henry G124 protect-delta source has the right protected-delta mechanism but still lacks the private G124 checkpoint/infer asset; fallback preserves the anchor and is not a high-upside slot by itself.
- New PCEN/EoS/NFNet forks remain plateau-like after v599-v604.

Next actionable training step is to run the G124 EffV2-S smoke config where the repo and public946 teacher cache are both available, then run sidecar/rank-blend audit before any Kaggle slot.
