# BirdCLEF 2026 Public946 Diversity Stream Triage — 2026-05-12

Status: planning / no new submission candidate yet  
Owner branch: `feature/v539-public946-replay` / PR #223  
Current scored anchor: `v539 = 0.943`  
Pending score gates: `v541`, then `v542`

---

## Decision summary

Do **not** push or queue a new diversity-stream kernel before `v541` and `v542` score.

The next genuinely distinct public-stream candidates are:

1. **V5/CLAP fork** from `needless090/birdclef-2026-perch-sed-lb-0-946-clap`.
2. **BirdNET/custom EffNet fork** from `raunakdey07/birdclef-2026-birdnet-4-way-rank-blend`.

Nina Model_61/62 has already been mined and appears to be a high-correlation 50/50-ish Proto/SED rank-blend idea, not a distinct stream.

---

## Candidate A — V5 / CLAP public fork

Source: `needless090/birdclef-2026-perch-sed-lb-0-946-clap`  
Local source: `artifacts/public_kernels_20260511/birdclef-2026-perch-sed-lb-0-946-clap.py`

### What it adds

After the standard public946 Proto/SED paths, it optionally adds:

- **V5 SED stream** from `birdclef2026-sed-v5-trio`.
  - Expected files:
    - `v5_cluster_aware.onnx`
    - `v5_focal.onnx`
    - `v5_pseudo2.onnx`
    - `v5_pseudo.onnx`
    - `v5_external.onnx`
  - Runtime path probes:
    - `/kaggle/input/datasets/needless090/birdclef2026-sed-v5-trio`
    - `/kaggle/input/birdclef2026-sed-v5-trio`
- **CLAP probe stream** from `birdclef2026-clap-probe`.
  - Expected files/config:
    - HuggingFace `ClapProcessor` / `ClapModel` assets.
    - `clap_probe_W.npy`
    - `clap_probe_b.npy`
    - `clap_probe_fitmask.npy`
  - Runtime path probes:
    - `/kaggle/input/datasets/needless090/birdclef2026-clap-probe`
    - `/kaggle/input/birdclef2026-clap-probe`
  - Has dynamic abort budget: `_CLAP_BUDGET = 45 * 60` seconds.

### Blend weights

The fork uses public946 rank streams plus optional V5/CLAP:

- If V5 exists and CLAP does not:
  - Proto rank: `1 - SED_W - V5_W`
  - SED rank: `SED_W`
  - V5 rank: `V5_W`
- If CLAP exists for a file:
  - Proto rank: `1 - SED_W - V5_W - CLAP_W`
  - SED rank: `SED_W`
  - V5 rank: `V5_W`
  - CLAP rank: `CLAP_W`
- Constants observed:
  - `V5_W = 0.15`
  - `CLAP_W = 0.10`
  - `SED_W` comes from the public946 final-blend block, usually `0.40`.

So the intended active-file 5-way blend is approximately:

- Proto `0.35`
- SED `0.40`
- V5 `0.15`
- CLAP `0.10`

If CLAP is missing, it falls back to about:

- Proto `0.45`
- SED `0.40`
- V5 `0.15`

### Source availability audit — 2026-05-12 08:55 UTC

Kaggle kernel metadata for `needless090/birdclef-2026-perch-sed-lb-0-946-clap` exposes the extra inputs only as blank dataset refs via `GetKernel`:

- dataset refs: `""`, `""`, `tuckerarrants/bc2026-distilled-sed-public`, `tuckerarrants/perch-v2-no-dft-onnx`, `rishikeshjani/perch-onnx-for-birdclef-2026`
- model refs: `google/bird-vocalization-classifier/TensorFlow2/perch_v2_cpu/1`
- kernel refs: `ashok205/tf-wheels`

The embedded notebook JSON has numeric source IDs for the blank extras:

- datasetVersion `sourceId=16013757`, `datasetId=10267502`, `databundleVersionId=16978012`
- datasetVersion `sourceId=16003884`, `datasetId=10025194`, `databundleVersionId=16967278`

Attempts to resolve likely slugs `needless090/birdclef2026-sed-v5-trio` and `needless090/birdclef2026-clap-probe` with Bearer Dataset API returned `403 Forbidden`; public dataset search returned no matching rows. Interpretation: the code path is visible, but the extra V5/CLAP inputs are not currently attachable by clean public slug from our account. A repo-owned port should not be queued until source refs are resolved or replaced by our own equivalent datasets.

### Risks

- The extra V5/CLAP datasets are not currently attachable by clean public slug; otherwise the code silently skips into plain public946/V5-missing fallback.
- CLAP may only cover part of hidden test due dynamic abort; code saves `clap_filemask.npy` and applies CLAP only for covered files.
- Hidden test runtime risk is real. The CLAP budget is 45 minutes on top of public946 and V5.

### Required validation before queueing

Do not queue a V5/CLAP candidate unless the public dry-run log confirms:

- `v5 loaded:` for at least one V5 ONNX, ideally all five.
- `Saved submission_v5.csv` with expected row count.
- Either:
  - `CLAP stream loaded, mask coverage X/Y files`, or
  - clear/intentional CLAP skipped behavior with V5 still active.
- Final blend log says `3-way` or `5-way`; not plain 2-way.
- Output shape aligns to sample submission / dry-run behavior.
- Wall time leaves comfortable hidden CPU budget.

### Priority

High after `v541/v542` score if both do not settle the 0.946 anchor. This is the most promising true diversity stream because V5/CLAP are new prediction artifacts, not just another Proto/SED weight sweep.

---

## Candidate B — BirdNET / custom EffNet public forks

Primary source: `raunakdey07/birdclef-2026-birdnet-4-way-rank-blend`  
Safer reproducible source: `claudedevore/birdclef-2026-r0946-birdnet-3way-submit`  
Local source: `artifacts/public_kernels_20260511/birdclef-2026-birdnet-4-way-rank-blend.py`  
Local audit source: `artifacts/public_kernels_20260511/birdclef-2026-r0946-birdnet-3way-submit.py` (ignored artifact)

### What it adds

After standard public946 Proto/SED paths, it adds:

- **BirdNET 6K TFLite stream**.
  - Code path:
    - `/kaggle/input/models/shadiakiki1/birdnet-analyzer/tflite/birdnet_global_6k_v2.4_model_fp32-1/3/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite`
  - Produces `submission_birdnet.csv`.
- **Custom EfficientNet-B0 ONNX stream**.
  - Code path:
    - `/kaggle/input/notebooks/raunakdey07/offline-training-efficientnet-b0-focal-recording/efficientnet_b0_birdclef.onnx`
  - Produces `submission_effnet.csv`.

### Blend weights

If both extra streams exist:

- Proto rank: `0.40`
- SED rank: `0.30`
- BirdNET rank: `0.15`
- custom EffNet rank: `0.15`

If either extra stream is missing, the fork falls back to standard:

- Proto rank: `0.60`
- SED rank: `0.40`

### Source availability audit — 2026-05-12 09:05 UTC

BirdNET itself is now resolved as an attachable Kaggle model source:

- `shadiakiki1/birdnet-analyzer/TfLite/birdnet_global_6k_v2.4_model_fp32-1/3`
- model API confirms version `3`, instance `birdnet_global_6k_v2.4_model_fp32-1`, framework `MODEL_FRAMEWORK_TF_LITE`, about `51.99 MB` uncompressed.
- Expected model path matches code:
  `/kaggle/input/models/shadiakiki1/birdnet-analyzer/tflite/birdnet_global_6k_v2.4_model_fp32-1/3/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite`

The custom EffNet branch is not cleanly reproducible yet:

- Raunak 4-way notebook embedded JSON includes kernelVersion source `317846744`, corresponding to the referenced notebook-output path.
- `GetKernel` for `raunakdey07/offline-training-efficientnet-b0-focal-recording` returned `403 Forbidden`.
- Public kernel search finds related forks but not a clean attachable source ref for the EfficientNet ONNX.

A useful public fork exists:

- `claudedevore/birdclef-2026-r0946-birdnet-3way-submit`
- metadata includes BirdNET model source plus the usual public946 inputs.
- it explicitly skips the unavailable custom EfficientNet branch and runs a reproducible Proto + SED + BirdNET path.
- caveat: the fetched notebook snapshot appears to contain repeated “EffNet skipped” cells and did not expose a clean final blend cell in the extracted source, so do not port blindly; extract only the BirdNET inference block and write our own final blend.

### Risks

- BirdNET label mapping into the 234 BirdCLEF target labels may be brittle; we need log confirmation and row/column checks.
- BirdNET alone is attachable, but custom EffNet is blocked by a 403 notebook-output source.
- Runtime is probably safer than CLAP: the ClaudeDevore snapshot shows BirdNET inference around 17 seconds on dry-run public execution after the main public946 branches, but hidden runtime still needs validation.
- Do not import the ClaudeDevore notebook wholesale; build a minimal repo-owned BirdNET block and explicit 3-way rank blend.

### Required validation before queueing

For a BirdNET-only 3-way candidate, dry-run must confirm:

- BirdNET TFLite source exists and interpreter initializes.
- BirdNET label file is found.
- mapping count from BirdNET scientific names into 234 targets is printed.
- `submission_birdnet.csv` is written and row-aligned with `submission_protossm.csv`.
- final blend uses explicit 3-way weights, not fallback 2-way.
- wall time is comfortably under CPU budget.

For a full BirdNET/EffNet 4-way candidate, additionally require:

- custom EffNet ONNX source exists and loads.
- `submission_effnet.csv` is written and aligned.
- final blend uses 4-way weights, not fallback 2-way.

### Priority

Medium-high for **BirdNET-only 3-way** if v541/v542 do not settle the anchor and V5/CLAP remains source-blocked. Medium/low for full BirdNET+EffNet 4-way until the custom EffNet output source is resolvable.

---

## Candidate C — Nina Model_61/62 public ensemble

Source: `nina2025/birdclef-2026-ensemble-of-solutions-3`  
Status: mined enough for now.

Offline reconstruction on v542 dry-run rows showed:

- `proto0.40_sed0.60`: AUC `0.994484`
- `proto0.46_sed0.54`: AUC `0.993964`
- Nina Model_61/62 direct proxy: AUC `0.993627`
- exact `50/50`: AUC `0.993616`
- v542-style `60/40`: AUC `0.992525`
- Nina proxy correlation vs v542 60/40: about `0.993`

Interpretation: the clean extractable Nina idea is not a distinct model stream; it is a high-correlation weight perturbation. Hold until v541/v542 scores. If both miss 0.946, consider one clean `50/50` or `40/60` weight test, not the full notebook.

---

## Next action after v541/v542 score

1. If either `v541` or `v542` scores `>=0.946`:
   - Make it the canonical public946 anchor.
   - Next submit should be a genuinely distinct diversity stream only if validation is clean.
2. If both land around `0.943`:
   - Consider exactly one `v543` simple weight test (`50/50` or `40/60`) **or** a diversity stream.
3. Diversity stream ordering after this source audit:
   - V5/CLAP only if hidden dataset refs are resolved or recreated.
   - Otherwise prefer a minimal BirdNET-only 3-way rank-blend candidate using the resolved `shadiakiki1/birdnet-analyzer/TfLite/.../3` model source.
   - Defer custom EffNet 4-way until the 403 notebook-output source is resolved.
4. If either public candidate underperforms v539:
   - Keep v539 as anchor.
   - Stop public postprocess forks and prioritize source-clean BirdNET or private-robustness student sidecar.
