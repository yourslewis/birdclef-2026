# PANNs/Cnn14 no-train row-only h384 isolation data point — 2026-05-31 10:16 UTC

## Status
- **Experiment id:** `soundscape-sequence-panns-cnn14-notrain-rowonly-h384-losite-ep24-20260531`
- **Branch family:** train_soundscapes sequence/file/site targeted AudioSet mining
- **Purpose:** isolate whether the stronger 08:20 72-label PANNs row-only result came from wider h384/regularization or from the non-Aves auxiliary/multitask labels.
- **Decision:** reject direct/sidecar submission; keep as a negative/diagnostic data point. The h384 no-train-only head improves over the old 28-label no-train row-only point, but trails the 72-label multitask row-only no-train slice and remains far below v616 as a sidecar.

## Live status before work
- Kaggle Bearer API: current best public LB remains `0.950`, tied by v644/v647 from 2026-05-30.
- 2026-05-31 UTC slots used: `0/5` at live check.
- Trainer: GPU0 occupied by unrelated HSTU job; GPU1 used via `CUDA_VISIBLE_DEVICES=1`.

## Training setup
- **Data:** official `train_soundscapes`, `1,478` windows / `66` files / `9` sites.
- **Target scope:** `28` no-train-primary labels (`3` Amphibia + `25` insect sonotypes).
- **Model/init:** frozen PANNs/Cnn14 AudioSet embeddings from `artifacts/panns_soundscape_embeddings/panns-cnn14-audioset-soundscape-nonaves-notrain-nocall-siteS08-ep12-20260526/panns_embeddings.npz`; row-only MLP head, hidden dim `384`, dropout `0.40`.
- **Split:** leave-one-site; valid folds `S03`, `S08`, `S13`, `S19`, `S22`, `S23`.
- **Config:** `configs/birdclef/soundscape_sequence_panns_cnn14_notrain_rowonly_h384_losite_ep24_20260531.json`
- **Training artifact:** `artifacts/soundscape_sequence_mining/soundscape-sequence-panns-cnn14-notrain-rowonly-h384-losite-ep24-20260531/`

## Metrics

| Metric | Value | Comparison |
|---|---:|---|
| row macro AUC mean | `0.590497` | +`0.016661` vs old 28-label PANNs row-only `0.573836`; -`0.009984` vs 72-label PANNs row-only no-train slice `0.600481` |
| file-MIL AUC mean | `0.640872` | +`0.073734` vs old 28-label row-only `0.567138`; -`0.050284` vs 72-label row-only file-MIL `0.691156` |
| no-train AUC | `0.590497` | target scope is exactly no-train-only |
| non-Aves AUC | `0.590497` | no-train labels are non-Aves here |
| sidecar best local AUC | `0.990402` / 42 valid | lift vs v616 `-0.003079`; lift vs anchor `+0.000011` |

Fold row AUCs: S03 `0.576948`, S08 `0.495797`, S13 `0.799479`, S19 `0.611206`, S22 `0.508636`, S23 `0.550918`.

## Sidecar/package audit
- Built 28→234 anchor-filled sidecar from leave-site predictions using `scripts/birdclef_soundscape_sequence_sidecar_audit.py`.
- Audit artifact: `artifacts/model_data_point_ledger/20260531T1016Z_panns_notrain_rowonly_h384_sidecar_audit/audit_summary.json`
- Best recipe: `seq_context_w01` (1% scoped sidecar).
- Local proxy AUC: `0.9904016095` / 42 valid classes.
- Lift vs v616: `-0.0030790581`.
- Rank corr vs v616: `0.9996887909`; MAE vs v616 `0.0062570721`.
- `submit_approved=false`; not hidden-test package grade.

## Per-class selector diagnostics
- Existing 72-label row-only PANNs sidecar per-class selector:
  - site CV lift `+0.00007865`, q05 `0.0`, positive groups `50%`; file CV lift `+0.00009510`, but held-file lift summary q05 `0.0` and `p_gt_0=0.0`.
  - selected all-row classes: `517063` (0.04), `555146` (0.04), `47144` (0.02).
  - Decision: comparison-only; no slot.
- New 28-label h384 sidecar per-class selector:
  - site CV lift `+0.00002245`; file CV lift `+0.00003207`, but file held-group q05 `-0.022727` and min `-0.090909`.
  - selected only `517063` at 0.04.
  - Decision: reject; selector evidence weaker than 72-label multitask sidecar and not robust.

## Critic / verifier
- **Critic:** The result argues against narrowing to no-train-only labels; auxiliary non-Aves labels in the 72-label row-only model appear helpful for no-train generalization. Do not keep scaling no-train-only PANNs heads unless a new data/architecture hypothesis appears.
- **Verifier:** Artifacts are finite/nonconstant; sidecar row/columns align on the 240-row v616 proxy. Candidate fails promotion and is not submission-grade.

## Next exact action
Prioritize v950/EoS8 PowerOptimization source-family exploitation or class/site-constrained selectors over another blind PANNs wrapper. If training another data point, use a genuinely new signal (e.g., source-winner PowerOptimization/proto path, curated multi-site no-call negatives, or a non-PANNs acoustic encoder), not another no-train-only PANNs capacity tweak.
