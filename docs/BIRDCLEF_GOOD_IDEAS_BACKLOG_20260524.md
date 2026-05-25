# BirdCLEF 2026 Good Ideas Backlog — 2026-05-24

Current confirmed best: **0.949**. Target: **0.960+**.

Principle: **do not let the queue become empty just because public replay candidates are exhausted.** If no source-safe public kernel is slot-worthy, pivot to no-slot repo-owned validation: training smoke tests, anchored sidecar audits, export/runtime checks, source repair, OOF grids, and artifact discovery. Kaggle slots are for promoted candidates only.

## Immediate top 3 no-slot tests

### 1. Alexy NS1 CNN/noisy-student sidecar extraction

Why it is good:

- Structurally distinct from the exhausted EoS/PCEN/HGNet/rank-power families.
- Public source contains a CNN/noisy-student ensemble path and Perch blending logic.
- Direct public source is not slot-ready (`LB 0.922` header and public output shape issues), but it is a useful repo-owned sidecar candidate.

First artifact:

- `artifacts/anchored_blend_audit/alexy_ns1_sidecar_grid.json`

First implementation step:

```bash
mkdir -p kaggle-kernels/v613-alexy-ns1-sidecar
# Pull/rewrite the public source into a hidden-safe branch producer, then:
python3 -m py_compile kaggle-kernels/v613-alexy-ns1-sidecar/script.py
```

Promotion gate:

- Hidden-safe branch output on real `test_soundscapes`, not copied public `(192,235)` output.
- Branch CSV row-aligns with the anchor.
- Low-weight rank blend has bounded displacement, target corr `>=0.997` vs anchor.
- Local lift must be stable under file/site bootstrap; still rejection-only, not approval.

Kill gate:

- Any fixed sample-output fallback, non-rerunnable checkpoint path, or broad anchor takeover.

### 2. G124 EffV2-S reconstruction from strong init

Why it is good:

- The missing S124/G124 line remains the clearest public clue above the 0.949 plateau.
- Scratch EffV2-S failed; the next reasonable test is not another scratch run, but V2S/external-init plus pseudo labels.

Existing config checked:

- `configs/birdclef/g124_effv2s_public946_pseudo_smoke_20260523.json`
  - `backbone=tf_efficientnetv2_s`
  - `epochs=4`
  - `batch_size=8`
  - output: `artifacts/pseudolabels/students/g124-effv2s-public946-pseudo-smoke-20260523`

Candidate init config to copy/borrow from:

- `configs/birdclef/xc_v2s_q3_cap80_external_pretrain_balanced_ep12_bestloss.json`
  - `backbone=efficientnetv2_rw_s`
  - output: `artifacts/external_pretrain/xc-v2s-q3-cap80-external-pretrain-balanced-ep12-bestloss`

First command, after creating a config copy with the V2S init checkpoint wired in:

```bash
source ~/kaggle_envs/s6e3/bin/activate
python scripts/birdclef_pseudolabel_student_train.py \
  --config configs/birdclef/g124_effv2s_public946_pseudo_smoke_20260523_v2sinit.json
```

Promotion gate:

- Smoke validation AUC at least in the `0.90–0.93` range.
- Sidecar blend weight `>=0.01` is useful in OOF/anchored grid.
- Export/runtime path is feasible; no ONNX hang/regression.

Kill gate:

- Val AUC remains scratch-like/noisy, correlation useless, export hangs, or CPU/Kaggle inference is infeasible.

### 3. Unified anchored sidecar validation harness

Why it is good:

- v611/v612 showed that local-positive sidecars can tie publicly. We need a stronger no-slot rejection harness before spending v613+ slots.
- It lets us compare Praxel HGNet, Samejima v57, S14, Jungchan, Alexy NS1, and future sidecars under one protocol.

Existing script checked:

- `scripts/birdclef_public946_multi_sidecar_weight_grid.py` compiles.

First command template:

```bash
python scripts/birdclef_public946_multi_sidecar_weight_grid.py \
  --base-csv <anchor.csv> \
  --sidecar prax_hgnet:<praxel_hgnet_raw.csv> \
  --sidecar alexy_ns1:<alexy_ns1_raw.csv> \
  --labels-csv <train_soundscape_labels.csv> \
  --bootstrap-iters 1000 \
  --leave-one-group site \
  --output-json artifacts/anchored_blend_audit/next_sidecar_grid.json
```

Promotion gate:

- Positive aggregate lift plus stable file/site bootstrap.
- No catastrophic held-out group.
- Bounded displacement from the 0.949 anchor.
- Treat this as rejection-only unless backed by independent artifact/source evidence.

## Broader backlog of good ideas

1. **Alexy NS1 CNN sidecar extraction** — new CNN/noisy-student signal; mine as low-weight branch, not direct replay.
2. **G124 EffV2-S reconstruction with external/V2S init** — best missing >0.949 clue; avoid scratch-only retry.
3. **Unified anchored sidecar validation harness** — compare all sidecars before slot promotion.
4. **Per-class residual/sidecar selector** — global 6% sidecars tied; per-class capped weights may capture classes where each sidecar helps. Gate with strong regularization and leave-site validation.
5. **Real SED OOF refresh, all-class, export-first** — still one of the few plausible routes to a true jump. Use OOF/export gates before any submission.
6. **Pseudo-label threshold/cache redesign** — existing students cloned the teacher or failed transfer; improve label distribution before training more students.
7. **Hard-negative/no-call residual gate** — distinct from taxon scalar gates; train/validate a background suppressor with rare-call protection.
8. **External-data manifest refresh + short V2S/B0 pretrain** — 2025 recipes leaned on external diversity; measure only by 2026 OOF/blend, not external-val alone.
9. **WildSound repair as offline lane** — public notebook errors, but ConvNeXt/WildSound idea is distinct; repair locally only if no in-kernel training timeout.
10. **StudyExchange S14 branch extraction** — direct source likely below frontier, but BidirProtoSSM/Tucker/Snowflake SED branch may be useful as sidecar.
11. **ONNX/OpenVINO export smoke matrix** — prevent v609-style timeouts and v594-style packaging failures before slots.
12. **OOF ensemble grid across all repo-owned prediction NPZs** — build comparable private metrics instead of trusting leaderboard slots.

## Families to avoid unless new artifact evidence appears

- EoS5/EoS6 rank-power scalar tweaks.
- PCEN forks after v604 tied.
- Visual/BirdNET/Mtoshi forks after v608 tied.
- Simple HGNet low-weight repeats after v611/v612 tied.
- Standalone clean-audio checkpoints after v610 scored `0.852`.
- `public0952` / Exp070 Perch-probe clones after v595 scored `0.899`.
- Direct G124 wrappers without the actual `g124_fold1_fp16.pt` / sidecar artifact.

## Environment notes

- Local neutral clone: `/Users/yourslewis/.openclaw/repos/birdclef-2026`.
- Mac default `python3` currently lacks runtime ML deps such as `pandas`; scripts compile but do not import-run there.
- Trainer venv works for ML deps: `source ~/kaggle_envs/s6e3/bin/activate` has `pandas`, `torch`, `timm`, `sklearn`, and `numpy`.
- Trainer path `~/birdclef-2026` exists but is not a git repo; sync intentionally before long runs.
