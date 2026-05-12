# BirdCLEF 2026 Public946 + BirdNET 3-Way Port Plan — 2026-05-12

Status: prepared plan only; **do not queue before v541/v542 scores**  
Owner branch: `feature/v539-public946-replay` / PR #223  
Base candidate if activated: `v543` or later  
Current scored anchor: `v539 = 0.943`; `v541 -> v542 -> v538` still queued

---

## 0. Why this plan exists

V5/CLAP is the best distinct public stream in principle, but its extra datasets are currently source-blocked from our account. BirdNET is the cleanest source-resolved diversity option:

- resolved model source: `shadiakiki1/birdnet-analyzer/TfLite/birdnet_global_6k_v2.4_model_fp32-1/3`
- no internet required
- small artifact: about 52 MB uncompressed
- public reference: `claudedevore/birdclef-2026-r0946-birdnet-3way-submit`

This is a preparation artifact so the next loop can implement quickly if `v541/v542` do not establish a stronger anchor.

---

## 1. Candidate policy

Do **not** push or queue this candidate until:

1. `v541` and `v542` have both scored, or one fails in a way that needs replacement.
2. The selected anchor is known.
3. The daily cap has room for one genuinely distinct stream.

If activated, prefer BirdNET-only 3-way before the full BirdNET+EffNet 4-way because custom EffNet remains blocked by a 403 notebook-output source.

---

## 2. Base kernel and metadata changes

Use `kaggle-kernels/v542-afr1ste-updated-public946/` as the base because it preserves full dry-run rows and has just been revalidated.

Create a new folder, e.g.:

- `kaggle-kernels/v543-public946-birdnet3/`

Start from v542 metadata and add the BirdNET model source:

```json
"model_sources": [
  "google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1",
  "shadiakiki1/birdnet-analyzer/TfLite/birdnet_global_6k_v2.4_model_fp32-1/3"
]
```

Recommended title/id if activated:

```json
"id": "yourslewis/bc26-v543-public946-birdnet3",
"title": "BC26 v543 Public946 BirdNET 3Way"
```

Keep CPU-only, internet-off, same public946 dataset/kernel sources.

---

## 3. BirdNET inference insertion point

Insert a new BirdNET block after `submission_sed.csv` is written and before the final rank blend cell.

Reference lines:

- v542 writes SED at `kaggle-kernels/v542-afr1ste-updated-public946/script.py`, around the `sed_sub.to_csv("submission_sed.csv", index=False)` block.
- final blend starts at the block defining:
  - `PROTOSSM_CSV = "submission_protossm.csv"`
  - `SED_CSV = "submission_sed.csv"`
  - `OUT_CSV = "submission.csv"`

Do not import the whole ClaudeDevore notebook. Extract only its BirdNET inference idea.

---

## 4. BirdNET block sketch

The reference public code maps BirdNET scientific names to competition labels, runs each 60s file as 12 windows, and uses the central 3 seconds of each 5-second window.

Key code behavior to preserve:

```python
birdnet_model_path = Path(
    "/kaggle/input/models/shadiakiki1/birdnet-analyzer/"
    "tflite/birdnet_global_6k_v2.4_model_fp32-1/3/"
    "BIRDNET_GLOBAL_6K_V2.4_Model_FP32.tflite"
)

label_files = list(birdnet_model_path.parent.glob("*.txt"))
if not label_files:
    label_files = list(birdnet_model_path.parent.parent.glob("**/*.txt"))

with open(label_files[0], "r") as f:
    bn_labels = [line.strip().split("_")[0] for line in f.readlines()]
```

Mapping gate:

```python
bn_idx_map = []
mapped_mask = []
for label in PRIMARY_LABELS:
    sci_name = tax_df.loc[tax_df["primary_label"] == label, "scientific_name"].values[0]
    if sci_name in bn_labels:
        bn_idx_map.append(bn_labels.index(sci_name))
        mapped_mask.append(True)
    else:
        bn_idx_map.append(0)
        mapped_mask.append(False)
print(f"BirdNET mapped classes: {sum(mapped_mask)} / {len(PRIMARY_LABELS)}")
```

Runtime gate:

- If the model path or label file is missing, fail loudly during candidate validation. Do not silently fallback to a 2-way output in the pushed candidate.
- For development-only smoke, an explicit `BIRDNET_OPTIONAL=False/True` flag can control this, but final candidate should require the source.

Audio preprocessing:

```python
if sr0 != 48000:
    y = librosa.resample(y, orig_sr=sr0, target_sr=48000)
# pad/truncate to 60s, reshape into 12 x 5s, take central 3s:
chunks = y.reshape(N_WINDOWS, 5 * 48000)
chunks_3s = chunks[:, 48000:192000]
```

Output:

```python
bn_sub = pd.DataFrame(np.clip(bn_preds, 0.0, 1.0), columns=PRIMARY_LABELS)
bn_sub.insert(0, "row_id", bn_rows)
bn_sub.to_csv("submission_birdnet.csv", index=False)
print("BirdNET 6K Inference Complete. Saved submission_birdnet.csv")
```

---

## 5. Final 3-way blend sketch

Modify the v542 final rank blend to include `submission_birdnet.csv`.

Base v542 currently does:

```python
rank_proto = rank(p_proto)
rank_sed = rank(p_sed)
pred = 0.60 * rank_proto + 0.40 * rank_sed
```

Initial BirdNET 3-way candidate should use a conservative rank blend:

```python
BIRDNET_CSV = "submission_birdnet.csv"
df_bn = pd.read_csv(BIRDNET_CSV)
df_bn = df_bn.set_index("row_id").loc[df_proto["row_id"]].reset_index()
p_bn = np.clip(df_bn[cols].to_numpy(np.float32), EPS, 1.0 - EPS)
rank_bn = pd.DataFrame(p_bn).rank(axis=0, pct=True).to_numpy(np.float32)

print("Executing 3-way rank blend (52% Proto / 38% SED / 10% BirdNET)...")
pred = (rank_proto * 0.52) + (rank_sed * 0.38) + (rank_bn * 0.10)
```

Why `0.10` BirdNET first:

- BirdNET is a true external acoustic stream, but label mapping is sparse and can be brittle.
- The public BirdNET 4-way reference used `0.15` BirdNET when also using custom EffNet; starting with `0.10` reduces public-risk while testing diversity.
- If dry-run validation and later LB are safe, a second candidate could try `0.15` BirdNET, but do not queue two before seeing the first score.

Keep v542’s existing post-blend gates:

- `fake_only` Proto rescue
- proto continuity context
- SED-only rescue
- sonotype mirroring
- adaptive rare thresholding

But update conditions to account for BirdNET only if clearly justified. First candidate should avoid extra gates based on BirdNET confidence.

---

## 6. Required validation before push/queue

Before pushing a real Kaggle kernel:

1. Local static checks:
   - metadata includes BirdNET model source exactly.
   - script compiles.
   - no duplicate stale EffNet-skip cells imported.
2. Kaggle dry-run output checks:
   - log prints `BirdNET mapped classes: X / 234`.
   - TFLite interpreter initializes.
   - `submission_birdnet.csv` exists.
   - `submission_birdnet.csv`, `submission_protossm.csv`, `submission_sed.csv`, and final `submission.csv` row_ids align.
   - final `submission.csv` shape is `(240,235)` on public dry-run, no NaNs.
   - final log says `3-way rank blend`, not `2-way`.
   - runtime remains comfortably under the hidden CPU limit.
3. Queue policy:
   - only insert after `v541` and `v542` have scored, unless replacing a failed queued candidate.

---

## 7. Kill / pivot rules

Kill BirdNET 3-way if:

- BirdNET maps too few useful classes or the label file parsing is unstable.
- runtime materially threatens hidden timeout.
- dry-run output degenerates to near-constant ranks or strongly mismatched row_ids.
- v541/v542 already reach `>=0.946` and a safer public-anchor decision is needed before spending a slot.

If killed, next fallback after v541/v542 is one clean public weight test (`50/50` or `40/60`) or public946+student sidecar diagnostics, not full EffNet 4-way.
