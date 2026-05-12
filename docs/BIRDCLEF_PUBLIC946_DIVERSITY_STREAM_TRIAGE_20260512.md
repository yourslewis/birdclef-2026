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

### Risks

- The downloaded metadata includes blank dataset sources for the extra V5/CLAP datasets. A repo-owned port must explicitly attach the actual datasets; otherwise the code silently skips into plain public946/V5-missing fallback.
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

## Candidate B — BirdNET / custom EffNet public fork

Source: `raunakdey07/birdclef-2026-birdnet-4-way-rank-blend`  
Local source: `artifacts/public_kernels_20260511/birdclef-2026-birdnet-4-way-rank-blend.py`

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

### Risks

- The public metadata inspected locally does **not** list the BirdNET model source or custom EfficientNet notebook source, despite the code referencing `/kaggle/input/models/...` and `/kaggle/input/notebooks/...` paths. A repo-owned port must resolve exact attachable source refs first.
- BirdNET label mapping into the 234 BirdCLEF target labels may be brittle; we need log confirmation and row/column checks.
- Runtime may be heavier than V5-only, because it runs public946 + BirdNET TFLite + custom EffNet ONNX.

### Required validation before queueing

Do not queue a BirdNET/EffNet candidate unless dry-run confirms:

- BirdNET TFLite source exists and interpreter initializes.
- BirdNET label file is found.
- `submission_birdnet.csv` is written and aligned.
- custom EffNet ONNX source exists and loads.
- `submission_effnet.csv` is written and aligned.
- final blend uses 4-way weights, not fallback 2-way.
- wall time is comfortably under CPU budget.

### Priority

Medium. It is a real diversity candidate, but source attachment and runtime are riskier than V5/CLAP.

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
   - Next submit should be a genuinely distinct diversity stream only if validation is clean; prefer V5/CLAP over BirdNET.
2. If both land around `0.943`:
   - Consider exactly one `v543` simple weight test (`50/50` or `40/60`) **or** a V5-only/CLAP-clean port if source mounts validate.
3. If either public candidate underperforms v539:
   - Keep v539 as anchor.
   - Stop public postprocess forks and prioritize V5/CLAP source validation or private-robustness student sidecar.
