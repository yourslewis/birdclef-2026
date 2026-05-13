# BirdCLEF 2026 Public946 + Train-Audio-Head Plan — 2026-05-13

Status: prepared, **do not push before v545 scores**.

## Why this is a contender

`v541`/`v542` established the 0.946 public946 anchor. BirdNET 10% (`v543`) and BirdNET 5% (`v544`) both tied 0.946 but did not improve. `v545` is the single queued CLAP INT8 probe.

The train-audio-head branch is attractive because it is tiny, source-clean, and explicitly trained from the much larger `train_audio` corpus rather than just the small train-soundscape labels. Henry's public fork claims a hidden tie-break rank improvement while still displaying 0.946 after raising the head voter from 3% to 5%.

## Source audit

Public kernel audited: `henryszy/bc2026-raunak0946-direct-v44`

Relevant source snippets in the audited notebook:

- `HEAD_RANK_BLEND = 0.05`
- searches `/kaggle/input/**/head_weights_train_audio.npz`
- loads `W`, `b`, and optional `trained_mask`
- requires `emb_te` to be available
- computes `head_logits = emb_te @ W.T + b`
- rank-normalizes `head_probs`
- blends only trained classes: `pred = (1.0 - head_weight) * pred + head_weight * head_rank`

Public dataset audited: `konbu17/bird26-train-audio-head-v1`

Bearer Dataset API verified it is public/attachable and contains:

- `head_weights_train_audio.npz` — about 1.44 MB

Downloaded ignored local audit copy:

- `artifacts/public946_train_audio_head_audit_20260513/head_weights_train_audio.npz`

NPZ inspection:

| Key | Shape | Dtype | Notes |
|---|---:|---|---|
| `W` | `(234, 1536)` | `float32` | linear head weights over Perch embeddings |
| `b` | `(234,)` | `float32` | class biases |
| `trained_mask` | `(234,)` | `bool` | restrict blend to trained classes |
| `feature_dim` | scalar | `int32` | `1536` |
| `notes` | scalar | string | metadata |

## Integration pattern

Use `kaggle-kernels/v542-afr1ste-updated-public946/` as the base.

1. Attach dataset `konbu17/bird26-train-audio-head-v1`.
2. Locate `head_weights_train_audio.npz` recursively under `/kaggle/input`.
3. Require public946 Perch embeddings `emb_te`; hard-fail or skip candidate before submission if unavailable.
4. Compute:

```python
head_logits = emb_te.astype(np.float32) @ head_W.T + head_b[None, :]
head_probs = sigmoid(head_logits)
head_rank = rank_pct(head_probs, axis=0)
head_weight = HEAD_RANK_BLEND * trained_mask
pred = (1 - head_weight) * pred + head_weight * head_rank
```

5. Save `submission_train_audio_head.csv` for validation and final `submission.csv` for competition output.

## Candidate weights

Start with a tiny sweep because this is a linear head and the public946 anchor is strong:

| Candidate | Final rank blend | Rationale |
|---|---|---|
| `head-w003` | public946 `0.97` + head `0.03` on trained classes | safer if v545 drops and we want minimal displacement |
| `head-w005` | public946 `0.95` + head `0.05` on trained classes | Henry v48 setting; claimed hidden tie-break improvement |

Run the sidecar gate before any Kaggle slot:

```bash
python scripts/birdclef_public946_sidecar_weight_grid.py \
  --base-csv artifacts/kaggle_outputs/v542-afr1ste-updated-public946/submission.csv \
  --sidecar-csv artifacts/kaggle_outputs/v546-public946-train-head/submission_train_audio_head.csv \
  --labels-csv /Volumes/ExternalSSD/data/workspace_don/kaggle_birdclef2026/data/train_soundscapes_labels.csv \
  --weights 0,0.01,0.02,0.03,0.05,0.075 \
  --output-json artifacts/blend_grids/public946_train_audio_head_weight_grid_20260513.json
```


## Local gate prerequisite

A local sidecar grid needs row-aligned `submission_train_audio_head.csv`. The current downloaded public946 artifacts preserve train-cache `perch_arrays.npz` but not the dry-run `emb_te` matrix used for `submission.csv`, so the head CSV cannot be reconstructed exactly from existing local files alone. The v546 implementation should therefore write `submission_train_audio_head.csv` during the Kaggle dry-run, download it, and then run `scripts/birdclef_public946_sidecar_weight_grid.py` before submission.

## Runtime and failure gates

Required before submission:

- `head_weights_train_audio.npz` found under `/kaggle/input`.
- `W.shape == (234, 1536)` and `b.shape == (234,)`.
- `emb_te` exists and has feature dimension `1536`.
- `trained_mask` aligns with the class columns.
- `submission_train_audio_head.csv` row-aligns with the public946 final output.
- Final log states explicit public946 + train-audio-head blend and weight.
- No silent fallback to plain public946.
- No material runtime risk; the branch is a matrix multiply over existing embeddings and should be cheap.

## Decision rule after v545

- If `v545 > 0.946`: compare lower CLAP (`0.01`/`0.02`) against CV9245 and train-audio-head gates before choosing v546.
- If `v545 = 0.946`: train-audio-head `0.05` and CV9245 `0.02` are the leading source-clean v546 candidates.
- If `v545 < 0.946`: stop CLAP; choose between train-audio-head and CV9245 using local gate metrics and output displacement.

Do not submit another BirdNET-only variant; both BirdNET probes tied 0.946.
