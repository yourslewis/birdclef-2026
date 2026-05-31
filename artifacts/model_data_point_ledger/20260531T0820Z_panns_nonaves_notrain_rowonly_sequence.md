# Model data point — PANNs/Cnn14 non-Aves/no-train row-only — 2026-05-31 08:20 UTC

## Experiment
- **Experiment id:** `soundscape-sequence-panns-cnn14-nonaves-notrain-rowonly-losite-ep24-20260531`
- **Branch family:** train_soundscapes sequence/file/site targeted AudioSet mining; row-only PANNs/Cnn14 embedding head.
- **Purpose:** isolate the row-only 72-label non-Aves/no-train signal after the r2 file-context/file-MIL variant underperformed its own row-only baseline. This gives a cleaner data point before adding context/MIL back.
- **Evidence level:** comparison-grade no-slot validation.

## Data / model
- Official `train_soundscapes` only.
- Rows/windows: `1,478`; files: `66`; sites: `9`.
- Target scope: `72` non-Aves or no-train labels; `5,420` positive target cells; `30` rows with no scoped label.
- Init/features: frozen PANNs/Cnn14 AudioSet embeddings from `artifacts/panns_soundscape_embeddings/panns-cnn14-audioset-soundscape-nonaves-notrain-nocall-siteS08-ep12-20260526/panns_embeddings.npz`.
- Head: MLP hidden `384`, dropout `0.40`, BCE with pos weight power `0.40` clip `12`, site-balanced sampling, no temporal/file/context features, no file-MIL loss.
- Split: leave-one-site; valid sites/folds: `6` (`S03`, `S08`, `S13`, `S19`, `S22`, `S23`).
- Runtime/export: trained on trainer GPU1; TorchScript head exported; final prediction columns nonconstant `72/72`.

## Metrics
- Row macro AUC: `0.674485` / 6 folds.
- File-MIL macro AUC: `0.691156` / 6 folds.
- No-train AUC: `0.600481` / 6 folds.
- Non-Aves AUC: `0.674485` / 6 folds.
- Fold range: row AUC min/max `0.519968` / `0.864060`.

## Sidecar audit vs v616
- Wrapped 72-label OOF predictions into 234-column anchor-filled proxy sidecar.
- Matched proxy rows: `156/240`; valid local classes: `42`.
- Best audited recipe: `seq_context_w01` / `seq_r2_w01` (same row-only sidecar at 1%).
- Local macro AUC: `0.990950`.
- Lift vs anchor: `+0.000560`.
- Lift vs v616: `-0.002530`.
- Rank corr vs v616: `0.999677`; MAE `0.006152`.
- `submit_approved=false` by policy: below v616 and not hidden-safe as a final package.

## Comparison / decision
- Compared with previous targeted r2 filectx+fileMIL PANNs 72-label point (`20260531T0620Z`): row AUC improves `+0.042893` (`0.674485` vs `0.631592`), file-MIL improves `+0.001151` (`0.691156` vs `0.690005`), no-train improves `+0.058851` (`0.600481` vs `0.541630`).
- Sidecar audit is still worse than v616 and close to prior scoped row-only wrappers, so this is **not** a slot candidate.
- **Decision:** keep as a useful targeted landscape point; reject direct/sidecar submission. Next work should be class/site movement diagnostics or v950/EoS8 source-family verifier, not more blind PANNs wrappers.

## Artifacts
- Config: `configs/birdclef/soundscape_sequence_panns_cnn14_nonaves_notrain_rowonly_losite_ep24_20260531.json`
- Metrics: `artifacts/soundscape_sequence_mining/soundscape-sequence-panns-cnn14-nonaves-notrain-rowonly-losite-ep24-20260531/metrics.json`
- OOF predictions: `artifacts/soundscape_sequence_mining/soundscape-sequence-panns-cnn14-nonaves-notrain-rowonly-losite-ep24-20260531/leave_site_predictions.npz`
- Sidecar audit: `artifacts/model_data_point_ledger/20260531T0820Z_panns_nonaves_notrain_rowonly_sidecar_audit/audit_summary.json`
