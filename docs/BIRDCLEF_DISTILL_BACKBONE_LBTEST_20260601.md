# BirdCLEF 2026 — Distinct-Backbone Distill LB-Test Attempt (CORRECTIVE)

Date: 2026-06-01 (PDT). Lane: ClawTeam competition-research, subagent depth 1/1.
Goal handed down: spend ONE leaderboard slot on a *genuinely different base-embedding*
candidate (the distinct-backbone distilled family), not a head-knob variant, to get the
real LB read on diversity transfer. Preferred candidate: distill 3-backbone rank stack
(ConvNeXt-w0.7 + ConvNeXt-w0.85 + RegNetY-008, soft1279-teacher distilled) blended
conservatively with the 0.950 frontier E.

## LIVE STATUS (verified via Kaggle Bearer API v1)

- UTC now: 2026-06-01T17:42Z. **Submissions today (UTC): 0.** Slots used 0/5, **5 remaining.**
- Latest 5 submissions are all 2026-05-31: v651–v655 EoS8 PowerOpt proto/sed frontier
  verifiers (0.941–0.949). No TTA, no BirdNET, no different-backbone submission exists today
  or in the recent window. The prior BirdNET LB-test subagent left **no slot spent** —
  confirmed.

## CANDIDATE EVIDENCE (already on record, re-verified)

| stream | base embedding | weak-class AUC | rank_decorr | DEV | blend wt |
|---|---|---:|---:|---:|---:|
| ConvNeXt-nano distill (w0.7) | ConvNeXt-nano front-end | 0.7997 | 0.7679 | 0.00230 | 0.0 |
| ConvNeXt distill w0.85 | ConvNeXt-nano | 0.8035 | 0.7647 | 0.00232 | 0.0 |
| RegNetY-008 distill | RegNetY-008 front-end | 0.7917 | 0.6853 | 0.00207 | 0.02 |
| **distill 3-backbone stack** | ConvNeXt+ConvNeXt+RegNetY | **0.8319** | **0.7374** | **0.00245** | 0.0 |

These are the strongest diversity leads in the program: first streams ever *simultaneously*
orthogonal AND competent, breaking the two-cluster law. The blend optimizer pins weight ~0
only because the **42/234-valid sparse proxy** saturates — the binding limiter is the proxy,
not model competence. This is precisely why a live hidden LB read is the only honest next step.

## BLOCKER (why no slot was spent this run) — HARD packaging gap

A hidden-test-capable Kaggle kernel must load the trained distinct-backbone model assets and
run them over `test_soundscapes` to write `submission.csv`. **Those model assets are not
reachable from this machine.**

1. **Trained weights are not local.** `metrics.json` for the ConvNeXt/RegNetY distill runs
   declares its export at
   `/home/yourslewis/birdclef-2026/artifacts/soundscape_native_losite/soundscape-native-convnextnano-soft1279teacher-distill-losite-allcls-ep6-20260601/model_torchscript.pt`
   (60.7 MB TS). That path is the **remote GPU training host** (`/home/yourslewis/...`),
   which **does not exist** on this control machine (`ls /home/yourslewis` → No such file).
   Only the *OOF artifacts* (`leave_site_predictions.npz`, `E_*_distill.csv`, summaries)
   were synced back into the repo — **no `.pt`/`.onnx` for any convnext/regnety/distill_stack
   backbone exists anywhere locally** (exhaustive `find` over repo + home dirs returned none).

2. **No Kaggle dataset holds the weights.** `datasets/list?user=yourslewis` returns an empty
   set for `distill`, `convnext`, `regnety`, `soundscape-native` — the distinct-backbone
   weights were never uploaded as a Kaggle dataset, so a kernel cannot attach them either.

3. **No local inference runtime.** `python3 -c "import torch"` → `ModuleNotFoundError`. This
   box cannot even re-export/convert weights if they were present, nor build/test a torch
   inference path.

4. **The only packageable artifact is the proxy sidecar.** `E_distill_stack3.csv` is a
   240-row anchor-filled OOF proxy (row_id = file_stem + end_seconds over train_soundscapes),
   not a hidden-test model. Submitting it would be exactly the **static-proxy / public-output
   fallback the task forbids.** It cannot read `test_soundscapes`; its row space is the train
   proxy, not the hidden test. Refused.

Conclusion: a hidden-safe different-base-embedding kernel **cannot be packaged within this
run** because the backbone weights are stranded on the offline GPU host and were never
exported to a local file or a Kaggle dataset. Faking it with the proxy CSV is explicitly
out of bounds. **No slot spent (correctly).**

## EXACT NEXT COMMAND(S) TO UNBLOCK

On the **GPU training host** (`/home/yourslewis/birdclef-2026`), where the weights and a
torch runtime live, do the wrapper+upload there, then push from this box:

1. Confirm the TS asset exists on the host and bundle the 3 backbones + label order:
   ```bash
   ssh <gpu-host> 'ls -la /home/yourslewis/birdclef-2026/artifacts/soundscape_native_losite/*convnext*distill*/model_torchscript.pt \
     /home/yourslewis/birdclef-2026/artifacts/soundscape_native_losite/*regnety*distill*/model_torchscript.pt'
   ```
2. Build a Kaggle **dataset** of the 3 TS weights + `labels.json` (234-col order from the
   npz `labels`) + mel config (sr=32000, n_fft=1024, hop=512, n_mels=160, 5 s windows), e.g.
   `kaggle datasets create -p <bundle_dir>` (Bearer API `datasets/create/upload`).
3. Author the inference kernel: for each `test_soundscapes` ogg → 5 s logmel windows →
   run the 3 TS backbones → per-window 234-logit → rank-mean across backbones → write
   `submission.csv` (235 cols incl. row_id; tta_shifts optional). Blend with the 0.950
   frontier E branches at small weight (w≈0.02–0.05 from the documented non-harmful setting),
   rank-space.
4. Preflight on this box via `birdclef_kernel_output_verify.py` + 3×235 dry-run schema,
   finite/nonconstant, species-column match, nonduplicate matrix hash & description vs v644/
   v616/v655, then push via the existing `push_v*.py` Bearer pattern and submit ONE slot.

The intellectual case for the LB probe stands; the only gap is **asset locality**, which is
a host/transfer problem, not a modeling one.

## PREFLIGHT FACTS GATHERED THIS RUN (for the eventual real kernel)

- distill_stack proxy: 240×234, finite, 234 nonconstant cols, first-100 rows all unique,
  matrix hash `90673358feb42fa9` (proxy only — NOT for submission).
- 234-label order available from npz `labels` (U10), matches taxonomy scope (206 train-primary
  + 28 no-train; 72 non-Aves; 75 soundscape-positive).
- Mel front-end config (from `config.resolved.json`): sr 32000, n_fft 1024, hop 512,
  n_mels 160, dur 5.0 s, backbones `convnext_nano` / `regnety_008`.
