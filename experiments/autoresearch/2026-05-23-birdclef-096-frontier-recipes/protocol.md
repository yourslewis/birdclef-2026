# Autoresearch Protocol: BirdCLEF 2026 0.96 Frontier Recipes

Date: 2026-05-23
Owner: Don / OpenClaw
Repo: `/Users/yourslewis/Documents/birdclef-2026-v545`
Status: active

## Goal

Find or create a distinct BirdCLEF 2026 signal capable of beating the current public LB plateau of **0.949** and moving toward **0.960+**.

This protocol intentionally deprioritizes small scalar/postprocess sweeps until a new artifact or prediction stream exists.

## Metric

- Name: Kaggle BirdCLEF 2026 public leaderboard macro AUC
- Direction: maximize
- Current baseline: `0.949`
- Target: `>0.960`
- Minimum useful public delta: `+0.001` if the source is genuinely new; `+0.003` for same-family postprocess changes
- Extraction:
  - Kaggle Bearer API v1: `https://www.kaggle.com/api/v1/competitions/submissions/list/birdclef-2026`
  - local candidate gates: OOF macro AUC, sidecar/anchor correlation, rank-blend delta, output schema validity

## Fixed Budget

### Per short source-verification trial
- Pull + audit source: <= 20 min
- Push private verifier: <= 20 min
- Kaggle run: <= 2h, fail fast if ERROR/no output
- Public slot use: only after COMPLETE + valid non-constant `submission.csv` + no traceback

### Per G124 reconstruction smoke
- One smoke config, one fold/seed: <= 90 min GPU time
- Kill timeout: 2h
- Must produce checkpoint + sidecar validation predictions + blend audit

### Per G124 pilot
- 1-2 folds, 8-12 epochs: <= 6h
- Kill timeout: 8h
- Continue only if blend/correlation evidence beats prior failed sidecars

## Trial Runner

### Status check
```bash
python3 - <<'PY'
import json, os, requests
with open(os.path.expanduser('~/.kaggle/kaggle.json')) as f: token=json.load(f)['key']
subs=requests.get('https://www.kaggle.com/api/v1/competitions/submissions/list/birdclef-2026',headers={'Authorization':f'Bearer {token}'},timeout=120).json()
for s in subs[:12]: print(s.get('ref'), s.get('description'), s.get('status'), s.get('publicScore'), s.get('errorDescription'))
PY
```

### Public-source audit
```bash
# Pull through Kaggle Bearer API v1, write source + summary JSON under artifacts/public_kernels_*/
# Existing examples: artifacts/public_kernels_higher_score_claims_20260523/summary.json
```

### Existing training scripts to reuse
```bash
python scripts/birdclef_pseudolabel_student_train.py --config <config.json>
python scripts/birdclef_student_pool_blend_audit.py --help
python scripts/birdclef_public946_multi_sidecar_weight_grid.py --help
python scripts/birdclef_sed_pilot_train.py --config <config.json>
```

## Result Contract

Every candidate must write one JSON/markdown record containing:

- candidate id / source URL / git sha
- hypothesis
- mutable files changed
- attached datasets/kernels/models
- source preflight result
- kernel status + output files
- `submission.csv` stats: rows, cols, finite, min/max, zero count, unique row IDs, unique first-100 values
- if trained: OOF macro AUC, prediction correlation vs current anchor, blend audit, runtime, artifact size
- final decision: submit / hold / kill / reproduce artifact

## Mutable Files

Allowed:

- `docs/BIRDCLEF_AUTORESEARCH_LOG.md`
- `docs/BIRDCLEF_096_FRONTIER_PLAN_20260518.md`
- `docs/BIRDCLEF_096_PROMISING_RECIPES_AUTORESEARCH_20260523.md`
- `experiments/autoresearch/2026-05-23-birdclef-096-frontier-recipes/**`
- `kaggle-kernels/v6xx-*`
- `scripts/push_v6xx_*.py`
- `scripts/submit_v6xx_*.py`
- new configs under `configs/birdclef/`
- ignored artifacts under `artifacts/`

Forbidden without explicit approval:

- merge to `main`
- destructive cleanup
- submitting scalar-only EoS5/v577/v578-style variants
- blind direct public-source submissions without output preflight

## Read-Only / Reference Files

- `docs/BIRDCLEF_NEW_DIRECTIONS_SPECS.md`
- `docs/BIRDCLEF_2025_RECIPE_PORT_SPEC_20260516.md`
- `artifacts/public_kernels_higher_score_claims_20260523/summary.json`
- public source audit text files under `artifacts/public_kernels_*`

## Search Space / Recipe Priorities

### P0 — G124/S124 0.952 artifact reconstruction

Hypothesis: the only credible >0.949 public claim is the S124/G124 line. The missing private dataset appears to contain the actual high-value sidecar.

Known missing/private artifact:

- `itshyao/birdclef2026-g124-effv2s-2025pre-pseudo-assets` — Kaggle API returns `403`, dataset search returns no public hit.

Expected contents from public reverse-engineering source:

- `g124_fold1_fp16.pt`
- `_best.pt` / fold checkpoint(s)
- `submission_g124_effv2s_fold1_s124.csv`
- G124 EfficientNetV2-S inference utility / rank-blend code

Plan:

1. Continue artifact search by exact filenames and dataset slug.
2. If not found, recreate an EffNetV2-S / G124-style sidecar:
   - backbone: `tf_efficientnetv2_s` or equivalent `timm` EffV2-S
   - crop: 5s primary, optional 20s context
   - pseudo labels: current 0.949 anchor + public946 teacher cache; no audio-LLM labels
   - external init: use existing public/pretrained configs where possible
   - output: `g124_fold1_fp16.pt`, sidecar `submission_g124_effv2s_fold1_s124.csv`
3. Apply strict S124 rank-blend to the 0.949 anchor.
4. Submit only if blend audit shows meaningful, bounded, low-correlation movement.

Kill criteria:

- sidecar/anchor correlation too high and blend weight optimum <= 0.005
- local lift only on train-soundscape gate with no leave-file/leave-site robustness
- artifact too slow/large for Kaggle CPU packaging

### P1 — Mechanical public-source repair only when structurally distinct

Current example: Eslam v26C.

Known sequence:

- v605 failed: missing `proto_model`
- v606 failed: missing `submission_protossm.csv`
- v607 repaired both and completed output preflight; submitted as `v607` if/when code-submission guard passes

Rules:

- Patch only mechanical blockers that prevent a credible source from writing valid output.
- Stop if the source degenerates into sample/fallback/random output.
- Stop after cascading nontrivial missing-model/dependency errors unless the next fix is obviously mechanical.

### P2 — Fresh source scouting

Continue Kaggle code scans every loop, but score candidates by independence and attachability:

High priority signs:

- public artifact dataset with model checkpoints
- full source + hidden-test path handling
- output includes final and branch submissions
- source claims a real submitted public LB, not just OOF
- not dominated by EoS/Karnak/Perch/ProtoSSM rank-power family

Low priority / hold signs:

- PCEN/EoS/NFNet public forks after v599-v604 plateau
- `public0952` titles with source identical to the v595 lane
- random placeholders
- sample-submission fallback
- private/403 dependencies with no reproducible training recipe

### P3 — Training lanes without audio LLM labels

We explicitly skip LLM-audio labeling because no audio-capable labeler is available.

Allowed training lanes:

- G124/EffV2-S reconstruction
- NFNet/EffV2-S pseudo-label student with lower correlation than prior sidecars
- real SED/frame-event only if it can produce new predictions and a Kaggle-fit artifact

## Git Policy

- Keep work on current feature branch unless a fresh branch is required by merged PR state.
- Commit specs and kept verifiers.
- Do not rewrite unrelated work.
- Do not merge PRs without Wenhao approval.

## Resource Policy

- Use GPU server `yourslewis@192.168.0.10` for training.
- Use venv `~/kaggle_envs/s6e3`.
- Mac data mount maps `/Volumes/ExternalSSD/data` to `/mnt/mac_data/`.
- For long training, use `nohup` or durable tmux and heartbeat every 15 minutes.
- Smoke 3-5 samples/configs before full training.

## Logging

Append each loop to:

- `docs/BIRDCLEF_AUTORESEARCH_LOG.md`
- `docs/BIRDCLEF_096_FRONTIER_PLAN_20260518.md`
- `memory/2026-05-22.md` until a new daily file exists

## Current Decision Queue

1. Watch `v607` score.
   - If >0.949: port/confirm/tune Eslam lineage.
   - If =0.949: treat as safe but plateau; no further Eslam repairs unless student ONNX becomes available.
   - If <0.949/no-score: kill Eslam repair lane.
2. Start G124 artifact scout + reconstruction spec implementation.
3. Only use remaining daily slots for source-safe distinct candidates.
