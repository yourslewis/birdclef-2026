# Implementation Feasibility Report — New BirdCLEF Branch Families

Date: 2026-05-25  
Role: Implementation Feasibility Engineer  
Repo: `/Users/yourslewis/.openclaw/repos/birdclef-2026`

## Executive decision

There are implementable no-slot branches left, but the fast path is **not another v616/SYD/PCEN/EoS/HGNet scalar blend**. The highest speed-to-signal candidates are repo-owned branches that can emit a hidden-safe raw CSV and then enter `scripts/birdclef_ensemble_strategy_audit.py` as a new member against both `anchor_only` and `v616_baseline`.

Recommended implementation order:

1. Broaden the soft OOF-teacher SED student with a better negative/no-call cache.
2. Re-test the existing 20s/local-window pseudo-label student as a branch producer, not as a direct model score.
3. Run a small G124/EffV2-S target-design grid; do not repeat the already-rejected all-row config unchanged.
4. Recover Alexy NS1 source/checkpoint access and extract CNN raw only if the access blocker clears.
5. Repair WildSound/ConvNeXt offline only; never train it in-kernel first.

## Current implementation constraints from repo state

- Best public LB remains `0.949`; v616 tied and is now the tied baseline to beat.
- Existing audit workbench is ready: `scripts/birdclef_ensemble_strategy_audit.py` with manifest `configs/birdclef/ensemble_strategy_20260525.json`.
- Current local overlap is narrow: `190` rows / `20` files / `6` sites / `42` valid AUC classes, so all local gates are rejection/comparison evidence, not approval evidence.
- Phase 2 audit showed `sakur_restored` was only `+0.0000556` local AUC over v616 and failed all submission gates.
- Exportable B0 SED path is operational: current TorchScript exports are `15.389 MB`, ONNX exports are `0.56 MB`, and CPU smoke has already passed.
- Mac default Python may lack ML deps; trainer venv should be used for actual ML runs:

```bash
source ~/kaggle_envs/s6e3/bin/activate
cd ~/birdclef-2026  # or synced repo copy on trainer
```

---

## Concrete branch implementation candidates

Each candidate below includes first smoke commands/configs, required assets, hidden-test safety, runtime/export risk, and ensemble audit integration.

## Candidate 1 — Soft OOF-teacher SED student + broader negative/no-call auxiliary

### Branch family

Repo-owned EfficientNet-B0 SED student trained from OOF teacher probabilities plus a masked negative/no-call auxiliary loss. This is different from the v616 public Samejima SED raw stream because the branch weights are trained locally from repo OOF caches and can be packaged as a TorchScript/ONNX bundle.

### Existing grounding

- Trainer: `scripts/birdclef_sed_pilot_train.py`
- Existing configs:
  - `configs/birdclef/sed_b0_oofteacher_b0v26_nfnetv29_soft_512_ep3_20260525.json`
  - `configs/birdclef/sed_b0_oofteacher_b0v26_nfnetv29_soft_negaux002_512_ep3_20260525.json`
- Existing outputs:
  - `artifacts/sed_oof_teacher_students/sed-b0-oofteacher-b0v26-nfnetv29-soft-512-ep3-20260525/`
  - `artifacts/sed_oof_teacher_students/sed-b0-oofteacher-b0v26-nfnetv29-soft-negaux002-512-ep3-20260525/`
- Smoke results already observed:
  - soft-only: macro AUC `0.8190205` over `80` valid classes, runtime `23.381s`, TS `15.389 MB`, ONNX `0.56 MB`.
  - negaux002: macro AUC `0.8194102` over `80` valid classes, runtime `16.726s`, TS `15.389 MB`, ONNX `0.56 MB`.
  - current negative cache covered only `26/512` rows (`5.08%`), so the next implementation step must broaden negative/no-call coverage.

### Required data/assets/model weights

- Data root on trainer: `/home/yourslewis/birdclef-2026/data`
- OOF teacher cache: `artifacts/pseudolabels/oof-teacher-cache/b0v26_nfnetv29_w090010_intersection_cache.npz`
- Negative cache: `artifacts/pseudolabels/oof-negative-cache/v13v15_neg005_pos095_cache.npz` or regenerated broader cache.
- External B0 init: `artifacts/external_pretrain/xc-b0-q3-cap80-external-pretrain-balanced-ep12/model_torchscript.pt`

Verification:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
for p in [
  'artifacts/pseudolabels/oof-teacher-cache/b0v26_nfnetv29_w090010_intersection_cache.npz',
  'artifacts/pseudolabels/oof-negative-cache/v13v15_neg005_pos095_cache.npz',
  'artifacts/external_pretrain/xc-b0-q3-cap80-external-pretrain-balanced-ep12/model_torchscript.pt',
]:
    print(p, Path(p).exists())
z = np.load('artifacts/pseudolabels/oof-teacher-cache/b0v26_nfnetv29_w090010_intersection_cache.npz', allow_pickle=True)
print(z.files, z['teacher_pred'].shape, z['labels'].shape)
PY
```

### First smoke command/config

Fast unchanged baseline rerun:

```bash
python scripts/birdclef_sed_pilot_train.py \
  --config configs/birdclef/sed_b0_oofteacher_b0v26_nfnetv29_soft_negaux002_512_ep3_20260525.json
```

Better next smoke: create a config copy with `max_files=1024`, `epochs=4`, `aux_negative_weight=0.01`, and a regenerated broader negative cache. Regenerate candidate cache first:

```bash
python scripts/birdclef_oof_negative_cache.py \
  --oof b0v26:artifacts/sed_oof/sed-b0-q3cap80-ep12init-oof-v26-10s-160-allfiles-ep5/oof_predictions.npz:0.5 \
  --oof nfnetv29:artifacts/sed_oof/sed-nfnet-allobserved-v29-20s-128-181cls-10per-ep5/oof_predictions.npz:0.5 \
  --negative-threshold 0.10 \
  --max-neg-per-row 96 \
  --max-neg-per-class 4000 \
  --output artifacts/pseudolabels/oof-negative-cache/b0v26_nfnetv29_neg010_cap96_cache.npz
```

If the referenced OOF NPZ names differ, locate them with:

```bash
find artifacts/sed_oof -name '*oof*.npz' -o -name 'oof_predictions.npz'
```

### Hidden-test safety plan

- Train/export locally; hidden Kaggle kernel must load only the packaged TorchScript/ONNX bundle, not training data labels.
- Hidden verifier must derive row IDs from `sample_submission.csv` and `test_soundscapes/*.ogg`.
- If hidden test is absent, dry-run on train soundscapes is allowed for validation only, but branch CSV must be named raw/debug and never copied as final static output.
- Hard-fail on missing hidden audio, row mismatch, nonfinite cells, duplicate row IDs, constant class columns, or fallback/sample-shaped output.
- Preserve raw branch CSV such as `submission_oofteacher_sed_raw.csv` before blending.

### Runtime/export risk and mitigations

- Low runtime risk for B0: current smokes are under a minute on GPU; CPU inference smoke is in the `0.05–0.08s/file` range for the small exported B0 bundles.
- Export path already passes TorchScript and ONNX; prefer TorchScript for simplest no-dependency CPU path, ONNX only after checker/runtime smoke.
- Main modeling risk is not runtime but target leakage/coverage: keep OOF-only cache construction and report coverage stats.

### Feeding ensemble audit harness

1. Run `scripts/birdclef_sed_infer_torchscript.py` on train soundscape rows or hidden/private verifier output to create a branch CSV with the same `row_id` and 234 class columns as anchor.
2. Add to a copied manifest, e.g. `configs/birdclef/ensemble_strategy_oofteacher_candidate_20260525.json`:

```json
"oofteacher_sed_raw": {
  "path": "artifacts/<candidate>/submission_oofteacher_sed_raw.csv",
  "role": "branch"
}
```

3. Add capped recipes such as `0.94 anchor + 0.03 v616_sed + 0.03 oofteacher_sed_raw` and `0.96 anchor + 0.04 oofteacher_sed_raw`.
4. Run:

```bash
PY=/Users/yourslewis/.openclaw/workspace-don/kaggle/playground-series-s6e3/.venv/bin/python
PYTHONPATH=scripts $PY scripts/birdclef_ensemble_strategy_audit.py \
  --manifest configs/birdclef/ensemble_strategy_oofteacher_candidate_20260525.json \
  --output-dir artifacts/ensemble_strategy_oofteacher_candidate_20260525 \
  --bootstrap-iters 1000 \
  --emit-candidate-csvs
```

---

## Candidate 2 — 20s/local-window pseudo-label student branch

### Branch family

Longer-context B0 SED/MIL student trained from public946 teacher rows with local-window target smoothing (`center_localmax_mix`). This is a temporal-context branch, not a scalar retune. It can behave differently on hidden soundscape event timing because it sees 20s context and learns from neighbor windows rather than isolated 5s labels.

### Existing grounding

- Trainer: `scripts/birdclef_pseudolabel_student_train.py`
- Existing strong config template:
  - `configs/birdclef/pl_public946_sed85_rankblend15_b0_centerlocalmax_r1_cw075_20s_m160_lr3e4_ep20_20260518.json`
- Supporting docs:
  - `docs/BIRDCLEF_PUBLIC946_20S_PSEUDOLABEL_20260518.md`
  - `docs/BIRDCLEF_FRAMEHEAD_20S_NEXT_SCALE_20260518.md`
- Existing conclusion: earlier 20s work was useful packaging/modeling infrastructure but not a direct slot. Reframe as a raw branch entering the ensemble audit, not direct final.

### Required data/assets/model weights

- Teacher NPZ: `artifacts/pseudolabels/public946-v540-teacher-cache66-v1/teacher_sed85_rankblend15.npz`
- Labels: `/home/yourslewis/birdclef-2026/data/train_soundscapes_labels.csv`
- Soundscapes: `/home/yourslewis/birdclef-2026/data/train_soundscapes`
- B0 external init: `artifacts/external_pretrain/xc-b0-q3-cap80-manifest20260517-external-pretrain-balanced-ep18-20260517/model_torchscript.pt`

Verification:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
for p in [
 'artifacts/pseudolabels/public946-v540-teacher-cache66-v1/teacher_sed85_rankblend15.npz',
 'artifacts/external_pretrain/xc-b0-q3-cap80-manifest20260517-external-pretrain-balanced-ep18-20260517/model_torchscript.pt',
 '/home/yourslewis/birdclef-2026/data/train_soundscapes_labels.csv',
]:
    print(p, Path(p).exists())
z=np.load('artifacts/pseudolabels/public946-v540-teacher-cache66-v1/teacher_sed85_rankblend15.npz')
print(z.files, z['probs'].shape, z['labels'].shape)
PY
```

### First smoke command/config

Do not start with full ep20. Make a smoke copy with:

- `experiment_id`: `pl-public946-sed85-rankblend15-b0-centerlocalmax-r1-cw075-20s-m160-smoke-20260525`
- `output_dir`: matching smoke artifact dir
- `epochs`: `4`
- `max_rows`: `384`
- `export_onnx`: `true`
- keep `duration_sec=20.0`, `temporal_target_mode=center_localmax_mix`, `temporal_neighbor_radius=1`, `temporal_center_weight=0.75`.

Run:

```bash
python scripts/birdclef_pseudolabel_student_train.py \
  --config configs/birdclef/pl_public946_sed85_rankblend15_b0_centerlocalmax_r1_cw075_20s_m160_smoke_20260525.json
```

Export/infer smoke after training:

```bash
python scripts/birdclef_sed_infer_torchscript.py \
  --manifest artifacts/pseudolabels/students/pl-public946-sed85-rankblend15-b0-centerlocalmax-r1-cw075-20s-m160-smoke-20260525/sed_bundle_manifest.json \
  --audio-dir /home/yourslewis/birdclef-2026/data/train_soundscapes \
  --max-files 4 \
  --output artifacts/pseudolabels/students/pl-public946-sed85-rankblend15-b0-centerlocalmax-r1-cw075-20s-m160-smoke-20260525/infer_smoke_probs.csv \
  --device cpu \
  --torch-threads 2
```

### Hidden-test safety plan

- Hidden inference must use the same 20s extraction policy around each 5s endpoint; when near file boundaries, pad deterministically.
- Do not read public dry-run outputs or static teacher submission CSVs in hidden kernel.
- Branch should be a raw CSV only; final blending must be capped and audited.
- Validate that 20s context does not change row count or row ID semantics.

### Runtime/export risk and mitigations

- Runtime is higher than 5s B0 because each row decodes 20s context. Mitigate by caching decoded file waveform per file and extracting all endpoint windows in one pass in the eventual verifier.
- TS size should remain ~`15.4 MB`; ONNX checker must pass before any private verifier.
- Use `max_rows=384/792` pilots before all-row training.

### Feeding ensemble audit harness

- Produce `submission_longctx_b0_raw.csv` on the same 240 dry-run/private verifier rows.
- Add as a new member against `anchor_v616_raw` and `v616_final`.
- First recipes:
  - `longctx_02`: `0.98 anchor + 0.02 longctx_b0_raw`
  - `longctx_sed_04`: `0.94 anchor + 0.03 sed_raw + 0.03 longctx_b0_raw`
- Require lift over v616 baseline, not merely over anchor.

---

## Candidate 3 — G124/EffV2-S reconstruction from V2S external init

### Branch family

EfficientNetV2-S / G124-style student trained from public946/pseudo labels, initialized from repo external V2S pretrain. This targets the missing G124 lineage without relying on unavailable `g124_fold1_fp16.pt` assets.

### Existing grounding

- Trainer: `scripts/birdclef_pseudolabel_student_train.py`
- Existing configs:
  - `configs/birdclef/g124_effv2s_public946_pseudo_smoke_20260525_v2sinit.json`
  - `configs/birdclef/g124_effv2s_public946_pseudo_pilot_20260525_v2sinit_allrows_ep8.json`
- Existing V2S init config:
  - `configs/birdclef/xc_v2s_q3_cap80_external_pretrain_balanced_ep12_bestloss.json`
- Current results from prior smoke/pilot:
  - 384-row smoke: loaded `786` init keys, val macro AUC `0.956867`, final-all AUC `0.962116`, corr `0.830097`, MAE `0.033398`, exported TS+ONNX, but blend lift only `+0.0000021` at weight `0.01`.
  - all-row ep8: final-all AUC `0.947190`, teacher `0.997018`, corr `0.878257`, MAE `0.031810`; best student weight `0.0025`, site bootstrap q05 negative; rejected as a slot candidate.

### Required data/assets/model weights

- V2S init: `artifacts/external_pretrain/xc-v2s-q3-cap80-external-pretrain-balanced-ep12-bestloss/model_torchscript.pt`
- Teacher NPZ: `artifacts/pseudolabels/public946-v540-teacher-cache66-v1/teacher_sed85_rankblend15.npz`
- Labels/soundscapes under `/home/yourslewis/birdclef-2026/data`
- No actual G124 public checkpoint currently available. If `g124_fold1_fp16.pt` or matching dataset appears, treat it as a separate asset-verification lane.

Verification:

```bash
python -m json.tool configs/birdclef/g124_effv2s_public946_pseudo_smoke_20260525_v2sinit.json >/dev/null
python - <<'PY'
from pathlib import Path
for p in [
 'artifacts/external_pretrain/xc-v2s-q3-cap80-external-pretrain-balanced-ep12-bestloss/model_torchscript.pt',
 'artifacts/pseudolabels/public946-v540-teacher-cache66-v1/teacher_sed85_rankblend15.npz',
]: print(p, Path(p).exists())
PY
```

### First smoke command/config

Do **not** rerun the unchanged all-row ep8 config. First useful smoke is a small target-design grid:

- Copy `g124_effv2s_public946_pseudo_smoke_20260525_v2sinit.json` into three smoke configs:
  1. `teacher_power=0.85`, `target_mode=soft`, `max_rows=384`
  2. `temporal_target_mode=local_max`, `temporal_neighbor_radius=1`, `max_rows=384`
  3. `target_mode=hard_conf`, `positive_threshold=0.80`, `negative_threshold=0.05`, `max_positive_per_row=5`, `max_rows=384`

Run one at a time:

```bash
python scripts/birdclef_pseudolabel_student_train.py \
  --config configs/birdclef/g124_effv2s_public946_pseudo_smoke_20260525_v2sinit_power085.json
```

### Hidden-test safety plan

- Package only trained TS/ONNX weights and preprocessing manifest.
- Hidden kernel must infer raw branch from hidden `test_soundscapes`, not from public teacher outputs.
- Since prior G124-like all-row branch hurt in group stability, cap initial blend weights at `0.005–0.02` in audit and require positive site/file bootstrap.
- If actual G124 assets are discovered, verify exact source/dataset license and shape; do not silently mix static submission files.

### Runtime/export risk and mitigations

- V2S is much heavier than B0; prior V2S/NFNet docs show TS sizes around `~88 MB` for some students.
- ONNX export has passed for the recent G124 V2S smoke, but Kaggle CPU inference may be the gating risk.
- Mitigate with ORT/TorchScript CPU inference on 4, then 16, then full dry-run files before any private verifier.

### Feeding ensemble audit harness

- Use `student_predictions.npz` for local/train-soundscape diagnostics.
- For audit, create a 240-row raw branch CSV from the same private verifier/dry-run rows and add member `g124_v2s_raw`.
- Initial recipes:
  - `g124_v2s_005`: `0.995 anchor + 0.005 g124_v2s_raw`
  - `g124_v2s_01`: `0.99 anchor + 0.01 g124_v2s_raw`
  - `g124_v2s_sed_02`: `0.94 anchor + 0.04 sed_raw + 0.02 g124_v2s_raw`

---

## Candidate 4 — Alexy NS1 CNN/noisy-student sidecar extraction

### Branch family

Five-fold CNN/noisy-student branch plus Perch+MLP logic from `alexycactus/birdclef-2026-ns1-ensemble`. This is genuinely distinct acoustically, but currently blocked by source/API access and weak direct LB.

### Existing grounding

- Feasibility notes: `kaggle-kernels/v613-alexy-ns1-sidecar/FEASIBILITY_NOTES.md`
- Public audit artifact: `artifacts/public_kernels_20260524_frontier_candidates/source_audit_20260524T2200Z_newleads/alexycactus__birdclef-2026-ns1-ensemble.json`
- Direct exploratory v613 scored `0.923`, so this is not a direct family.
- Log evidence says:
  - checkpoints under `alexycactus/birdclef-2026-cnn-ns1-checkpoints/fold0_best.pth` ... `fold4_best.pth`
  - CNN loaded 5 folds on CPU.
  - CNN inference on 16 dry-run files took `83s`; full kernel completed around `460s`.
  - final public dry-run output was `(192,235)` with `BC2026_Train_*` row IDs, so public final is not hidden-safe as-is.
- Current access blocker: Kaggle Bearer/API and `kaggle kernels pull` returned `403`; public web page hit reCAPTCHA.

### Required data/assets/model weights

- Source notebook/code for `alexycactus/birdclef-2026-ns1-ensemble`, currently blocked.
- Checkpoint dataset `alexycactus/birdclef-2026-cnn-ns1-checkpoints`, five `fold*_best.pth` files.
- Usual Perch/metadata assets if retaining the Perch+MLP side: `jaejohn/perch-meta`, `tuckerarrants/perch-v2-no-dft-onnx`, `rishikeshjani/perch-onnx-for-birdclef-2026`, and Perch model.

Verification once access is available:

```bash
kaggle datasets files alexycactus/birdclef-2026-cnn-ns1-checkpoints
kaggle kernels pull alexycactus/birdclef-2026-ns1-ensemble -p artifacts/source_audits/alexy_ns1_source_retry --metadata
find artifacts/source_audits/alexy_ns1_source_retry -type f -maxdepth 2 -print
```

### First smoke command/config

No production code change yet. First smoke is source/asset recovery plus a branch-only skeleton:

```bash
mkdir -p kaggle-kernels/v613-alexy-ns1-sidecar
# After source is available, extract CNN-only inference into script.py and hard-disable final TOP_K=1 postprocess.
python3 -m py_compile kaggle-kernels/v613-alexy-ns1-sidecar/script.py
```

If source remains blocked, stop here. Reimplementing from logs alone is too risky.

### Hidden-test safety plan

- Never use the `(192,235)` public dry-run output as a submission member.
- Hidden verifier must rerun CNN inference on current `test_soundscapes`; no static CSV reads.
- Output both `submission_alexy_cnn_raw.csv` and optional `submission_alexy_perch_mlp_raw.csv` before any blend.
- Disable or separately audit TOP_K=1 postprocessing; it is a major distribution shift and likely contributed to v613 weakness.
- Hard-fail if source falls back to train-row IDs, sample fallback, or missing checkpoints.

### Runtime/export risk and mitigations

- CPU runtime is material: `83s` for 16 dry-run files for CNN only; full public run around `460s`.
- Mitigate by extracting CNN-only first, batching by file, and avoiding Perch recomputation if the branch can be evaluated standalone.
- High maintenance risk because source access is blocked; keep as conditional lane only.

### Feeding ensemble audit harness

- Once branch CSV exists, add `alexy_cnn_raw` to copied ensemble manifest.
- Because direct v613 scored `0.923`, cap initial branch weights very low:
  - `alexy_005`: `0.995 anchor + 0.005 alexy_cnn_raw`
  - `alexy_01`: `0.99 anchor + 0.01 alexy_cnn_raw`
- Require positive lift over `v616_baseline`; do not trust aggregate anchor lift alone.

---

## Candidate 5 — WildSound/ConvNeXt repaired offline branch

### Branch family

ConvNeXt/WildSound-style acoustic branch repaired into repo-owned offline training/export. This is distinct from Perch/ProtoSSM/SED rank-blend lineages, but public notebook currently errors and may train in-kernel if used naively.

### Existing grounding

- Public audit artifact: `artifacts/public_kernels_20260524_frontier_candidates/source_audit_20260524T2200Z_newleads/muhammadsaadalvi__birdclef-2026-wildsound-v8.json`
- Error: missing `/kaggle/input/birdclef-2026/train_metadata.csv`.
- Repo trainer can already instantiate timm models through `scripts/birdclef_sed_pilot_train.py`; configs with `convnext_tiny` exist in `configs/birdclef/pl_public946_sed85_rankblend15_convnext_tiny_*`.

### Required data/assets/model weights

- Source notebook/code for WildSound v8 if recoverable.
- Repo data: `train_audio`, `train_soundscapes`, `train_metadata.csv` equivalent under `/home/yourslewis/birdclef-2026/data`.
- Timm ConvNeXt weights if using `pretrained=true`; otherwise external internet/model cache must be handled before Kaggle no-internet inference.

Verification:

```bash
python - <<'PY'
from pathlib import Path
for p in ['/home/yourslewis/birdclef-2026/data/train_audio', '/home/yourslewis/birdclef-2026/data/train_soundscapes']:
    print(p, Path(p).exists())
PY
python - <<'PY'
import timm
print('convnext_tiny' in timm.list_models('*convnext_tiny*'))
PY
```

### First smoke command/config

Fast repo-owned approximation using existing trainer, not public notebook:

- Copy `configs/birdclef/sed_b0_q3cap80_ep12init_exportsmoke_5s_160_allcls_20260525.json` to `configs/birdclef/wildsound_convnext_tiny_repair_smoke_20260525.json`.
- Change:
  - `backbone`: `convnext_tiny`
  - `experiment_id`: `wildsound-convnext-tiny-repair-smoke-20260525`
  - `output_dir`: `artifacts/sed_oof/wildsound-convnext-tiny-repair-smoke-20260525`
  - `epochs`: `1`
  - `max_files`: `256`
  - `batch_size`: `4`
  - `export_onnx`: `true`
  - `initial_checkpoint`: empty unless a compatible ConvNeXt checkpoint exists.

Run:

```bash
python scripts/birdclef_sed_pilot_train.py \
  --config configs/birdclef/wildsound_convnext_tiny_repair_smoke_20260525.json
```

### Hidden-test safety plan

- Do not train in a Kaggle inference kernel.
- Export trained weights to TS/ONNX and package as a dataset.
- Hidden kernel only runs inference and writes raw branch CSV.
- Hard-fail if model uses missing metadata paths or tries internet/download at inference time.

### Runtime/export risk and mitigations

- ConvNeXt may be slower and larger than B0. Start with `convnext_tiny`, 5s windows, small batch.
- ONNX export may fail depending on timm ops; TorchScript smoke is the fallback.
- If TS size/runtime is large or AUC is weak, kill before private verifier.

### Feeding ensemble audit harness

- Add `wildsound_convnext_raw` member once a row-aligned branch CSV exists.
- Initial recipes: `0.99 anchor + 0.01 wildsound_convnext_raw`, then `0.98/0.02` only if bootstrap is robust.

---

## Candidate 6 — Train-audio / external-data specialist head over missing or rare classes

### Branch family

Small specialist trained on train audio + external XC manifest, focused on classes with weak/absent train-soundscape labels or undercovered classes. This should emit a raw specialist probability branch, not a taxon scalar gate.

### Existing grounding

- External manifest/pretrain infrastructure:
  - `scripts/birdclef_external_pretrain_manifest.py`
  - `configs/birdclef/xc_b0_q3_cap80_external_pretrain_balanced_ep12.json`
  - `configs/birdclef/xc_v2s_q3_cap80_external_pretrain_balanced_ep12_bestloss.json`
- SED trainer supports manifest selection via `selection_strategy=manifest` and portable path resolution.
- Spec context notes `28` submission classes lack train-audio primary labels, so this candidate must explicitly report which classes it can and cannot cover.

### Required data/assets/model weights

- External pretrain manifest: `artifacts/external_pretrain/manifest_q3_cap80/external_pretrain_manifest.csv`
- External audio mirrored under `/home/yourslewis/birdclef-2026/data` or `/mnt/mac_data/...`.
- Existing B0/V2S external pretrain TorchScript weights.

Verification:

```bash
python - <<'PY'
from pathlib import Path
p=Path('artifacts/external_pretrain/manifest_q3_cap80/external_pretrain_manifest.csv')
print(p, p.exists())
if p.exists():
    import pandas as pd
    df=pd.read_csv(p)
    print(df.shape, df.columns.tolist()[:8], df['primary_label'].nunique())
PY
```

### First smoke command/config

Use `scripts/birdclef_sed_pilot_train.py` with a manifest-selected B0 specialist, max `256–512` files, `epochs=2`, `max_classes` restricted to classes undercovered by the current local labels. This likely needs a small manifest filtering step before training; no production code change is required if a filtered CSV is written under artifacts.

### Hidden-test safety plan

- Specialist must output all 234 classes; uncovered classes should be low-confidence/raw, not copied from anchor.
- Blend audit must measure per-class and group behavior; do not use unrestricted per-class selector after v616 selector overfit.
- Hidden kernel must package fixed weights and manifest labels only.

### Runtime/export risk and mitigations

- B0 specialist is low risk; V2S specialist is medium/high CPU risk.
- If many classes are uncovered, branch may be sparse or constant; audit script will hard-fail constant columns, so add small calibrated background values only if justified and documented.

### Feeding ensemble audit harness

- Add `rare_specialist_raw` as member.
- Use capped global weights first (`0.005–0.02`), then inspect per-class lift summary in the audit JSON manually. Avoid automatic per-class selector unless it passes leave-site/file gates.

---

## Shared hidden-safe branch contract

Every candidate private verifier or inference script should follow this contract before any submission can be considered:

1. **Inputs**: read `sample_submission.csv`, `taxonomy.csv` if needed, and current `test_soundscapes/*.ogg`; no static public output CSV as a branch source.
2. **Dry-run behavior**: if hidden test is absent, use a clearly logged train-soundscape dry-run; never let dry-run rows become a competition final without row validation.
3. **Raw outputs**: write one raw branch CSV per model family before final blend, e.g. `submission_<branch>_raw.csv`.
4. **Schema gates**: exact row ID set/order after alignment, exact class columns, finite values, no duplicate rows, all 234 class columns nonconstant.
5. **Runtime gates**: log per-file/sec, model size, number of folds/models, and device/thread settings.
6. **Audit gates**: candidate must beat both `anchor_only` and `v616_baseline` in `scripts/birdclef_ensemble_strategy_audit.py`; v616-like micro-lifts are rejected.
7. **No Kaggle submission** during this ideation phase.

---

## Ensemble audit integration template

For any candidate branch, create a copied manifest instead of editing the Phase 2 control manifest in place:

```bash
cp configs/birdclef/ensemble_strategy_20260525.json \
   configs/birdclef/ensemble_strategy_<candidate>_20260525.json
# edit members + recipes in the copied manifest only
```

Minimum manifest additions:

```json
"members": {
  "<candidate>_raw": {
    "path": "artifacts/<candidate>/submission_<candidate>_raw.csv",
    "role": "branch"
  }
},
"recipes": [
  {
    "name": "<candidate>_01",
    "type": "rank_blend",
    "weights": {
      "anchor_v616_raw": 0.99,
      "<candidate>_raw": 0.01
    }
  }
]
```

Run:

```bash
PY=/Users/yourslewis/.openclaw/workspace-don/kaggle/playground-series-s6e3/.venv/bin/python
PYTHONPATH=scripts $PY scripts/birdclef_ensemble_strategy_audit.py \
  --manifest configs/birdclef/ensemble_strategy_<candidate>_20260525.json \
  --output-dir artifacts/ensemble_strategy_<candidate>_20260525 \
  --bootstrap-iters 1000 \
  --emit-candidate-csvs
```

Promotion remains blocked unless a future verifier approves. Current quantitative minimums from coordinator: lift vs anchor `>= +0.0060`, lift vs v616 `>= +0.0010`, site bootstrap q05 `>= +0.0030`, file bootstrap q05 `>= +0.0015`, leave-one-site min `>= +0.0030`, leave-one-file q05 `>= +0.0010`, plus hidden-safe rerun.

---

## Top 5 actionable next experiments, ordered by speed-to-signal

### 1. Broader negative/no-call OOF-teacher B0 smoke

**Why first:** existing negaux smoke is already implemented/exportable and almost free; only blocker was sparse negative coverage.

Action:

```bash
find artifacts/sed_oof -name '*oof*.npz' -o -name 'oof_predictions.npz'
python scripts/birdclef_oof_negative_cache.py \
  --oof b0v26:<B0_V26_OOF_NPZ>:0.5 \
  --oof nfnetv29:<NFNET_V29_OOF_NPZ>:0.5 \
  --negative-threshold 0.10 \
  --max-neg-per-row 96 \
  --max-neg-per-class 4000 \
  --output artifacts/pseudolabels/oof-negative-cache/b0v26_nfnetv29_neg010_cap96_cache.npz
python scripts/birdclef_sed_pilot_train.py \
  --config configs/birdclef/sed_b0_oofteacher_b0v26_nfnetv29_soft_negaux001_1024_ep4_20260525.json
```

Gate: AUC materially above `0.8194`, coverage well above `5%`, TS/ONNX export passes, branch CSV nonconstant and auditable.

### 2. 20s center-localmax B0 pseudo-label branch smoke

**Why second:** existing config and trainer support this; it tests temporal context rather than another v616 branch tweak.

Action:

```bash
python scripts/birdclef_pseudolabel_student_train.py \
  --config configs/birdclef/pl_public946_sed85_rankblend15_b0_centerlocalmax_r1_cw075_20s_m160_smoke_20260525.json
```

Gate: export works, inference speed acceptable, branch has lower correlation/new movement vs v616, and audit lift is positive over v616 baseline.

### 3. G124/V2S target-design mini-grid

**Why third:** current V2S path is operational, but unchanged all-row G124 was rejected. Target design is the fastest thing that can change behavior.

Action:

```bash
python scripts/birdclef_pseudolabel_student_train.py --config configs/birdclef/g124_effv2s_public946_pseudo_smoke_20260525_v2sinit_power085.json
python scripts/birdclef_pseudolabel_student_train.py --config configs/birdclef/g124_effv2s_public946_pseudo_smoke_20260525_v2sinit_localmax.json
```

Gate: branch weight `>=0.01` must help in site/file bootstrap; otherwise kill.

### 4. Alexy NS1 access recovery + CNN-only branch extraction

**Why fourth:** genuinely distinct CNN/noisy-student signal, but currently source-blocked and direct v613 was weak.

Action:

```bash
kaggle datasets files alexycactus/birdclef-2026-cnn-ns1-checkpoints
kaggle kernels pull alexycactus/birdclef-2026-ns1-ensemble -p artifacts/source_audits/alexy_ns1_source_retry --metadata
```

Gate: if source/checkpoints are accessible, build CNN-only raw branch; if not, stop immediately.

### 5. WildSound/ConvNeXt offline repair smoke

**Why fifth:** highest implementation uncertainty, but it is a new acoustic family and the public failure is a fixable path issue.

Action:

```bash
python scripts/birdclef_sed_pilot_train.py \
  --config configs/birdclef/wildsound_convnext_tiny_repair_smoke_20260525.json
```

Gate: one-epoch smoke must train, export, and CPU-infer; otherwise do not spend more time.

## Final implementation recommendation

Start with Candidate 1 and Candidate 2 because they use existing repo scripts, existing assets, and known export paths. Treat Candidate 3 as a quick target-design grid, not a scale run. Keep Candidate 4 conditional on source access. Keep Candidate 5 as offline-only and kill early if export/runtime is ugly.
