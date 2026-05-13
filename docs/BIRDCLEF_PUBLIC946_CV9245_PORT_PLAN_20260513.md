# BirdCLEF 2026 Public946 + CV9245 Sidecar Port Plan — 2026-05-13

Status: prepared, **do not push before v545 scores**.

## Why this is next

`v541`/`v542` locked the public946 anchor, and `v543`/`v544` showed BirdNET 10%/5% is a safe tie but not an improvement. `v545` is the single queued CLAP INT8 probe. If `v545` ties or drops, the next source-clean diversity candidate should shift away from BirdNET/CLAP widening and toward the public CV9245 sidecar.

CV9245 is attractive because it adds a different public sidecar over shared Perch outputs, is much smaller than Snowflake SED, and has an existing public implementation pattern in Zeyad's two-branch public notebook.

## Source audit

Public dataset: `chaneyma/birdclef-2026-cv9245-moe-artifacts`

Bearer Dataset API verified it is public/attachable and contains:

- `README.md`
- `pantanal_infer_only_submission.py`
- `moe_p0.60_c0.25_r0.15_post_p0.45_fold1.pt`
- `moe_p0.60_c0.25_r0.15_post_p0.45_fold2.pt`
- `moe_p0.60_c0.25_r0.15_post_p0.45_fold3.pt`
- `moe_p0.60_c0.25_r0.15_post_p0.45_fold4.pt`
- `student_cnn_2025_plus_2026_nodistill_keepperch_ed28_cv45_c6_m45_ps0.40_seed42.pt`
- `student_crnn_2025_plus_2026_nodistill_keepperch_ed28_cv45_c6_m45_ps0.40_seed42.pt`

Local ignored audit copies:

- `artifacts/public946_cv9245_audit_20260513/README.md`
- `artifacts/public946_cv9245_audit_20260513/pantanal_infer_only_submission.py`

The script statically parses and compiles. Imports are standard for this project path: `numpy`, `pandas`, `soundfile`, `tensorflow`, `torch`, `torchaudio`.

## Integration pattern

Use `kaggle-kernels/v542-afr1ste-updated-public946/` as the base, not a BirdNET fork.

Borrow only the CV9245 sidecar path from `zeyadmohamadezzat/birdclef-2026-two-branch-perch-sed-sidecar`:

1. Locate CV9245 artifacts recursively under `/kaggle/input`:
   - fold weights matching `moe_p0.60_c0.25_r0.15_post_p0.45_fold*.pt`
   - `pantanal_infer_only_submission.py`
   - `student_cnn_2025_plus_2026_nodistill_keepperch*.pt`
2. Import the public CV9245 script with `importlib.util.spec_from_file_location`.
3. Reuse already-computed public946 Perch outputs from the base notebook:
   - `sc_te` / ProtoSSM logits shaped to `(n_files, N_WINDOWS, n_classes)`
   - `emb_te` / Perch embeddings shaped to `(n_files, N_WINDOWS, 1536)`
4. Run CV9245 on CPU with:
   - 4 MoE folds
   - student CNN logits mixed into teacher logits at Zeyad's public setting: `0.80 * Perch logits + 0.20 * CNN logits`
   - CV9245 prior scale `0.45`
   - `CV9245_BATCH_FILES=4`
5. Save `submission_cv9245_cnnonly_sharedperch.csv` and final rank-blend into the public946 anchor.

## Candidate weights

Start with a small rank sidecar because v541/v542 are already strong:

| Candidate | Final rank blend | Rationale |
|---|---|---|
| `v546-cv9245-w002` | public946 `0.98` + CV9245 `0.02` | safest first probe if v545 drops or CLAP looks harmful |
| `v546-cv9245-w005` | public946 `0.95` + CV9245 `0.05` | follows Zeyad's public sidecar setting; higher upside, higher anchor displacement |

Do not queue both blindly. Run the sidecar gate once CV9245 dry-run output exists:

```bash
python scripts/birdclef_public946_sidecar_weight_grid.py \
  --base-csv artifacts/kaggle_outputs/v542-afr1ste-updated-public946/submission.csv \
  --sidecar-csv artifacts/kaggle_outputs/v546-public946-cv9245/submission_cv9245_cnnonly_sharedperch.csv \
  --labels-csv /Volumes/ExternalSSD/data/workspace_don/kaggle_birdclef2026/data/train_soundscapes_labels.csv \
  --weights 0,0.01,0.02,0.035,0.05,0.075 \
  --output-json artifacts/blend_grids/public946_cv9245_weight_grid_20260513.json
```

## Runtime and failure gates

Required Kaggle log gates before submission:

- CV9245 artifacts found under `/kaggle/input`.
- `pantanal_infer_only_submission.py` imported successfully.
- Four fold weights loaded.
- Student CNN weight loaded.
- `submission_cv9245_cnnonly_sharedperch.csv` saved with exact `(rows, 235)` shape.
- Final log states explicit public946 + CV9245 rank blend and weight.
- No silent fallback to plain public946.
- Wall time remains comfortably below hidden-test budget. Zeyad's public path uses `CV9245_START_CUTOFF_MIN=65` and batch size 4; keep the same or stricter guard.

## Decision rule after v545

- If `v545 > 0.946`: do not immediately use CV9245; compare lower CLAP (`0.01`/`0.02`) against CV9245 gate outputs first.
- If `v545 = 0.946`: CV9245 `0.02` is the safest next probe unless local CV9245 gate strongly favors `0.05`.
- If `v545 < 0.946`: stop CLAP and use CV9245 or train-audio-head as the next distinct stream.

Do not submit another BirdNET-only variant; both BirdNET 10% and 5% tied 0.946.
