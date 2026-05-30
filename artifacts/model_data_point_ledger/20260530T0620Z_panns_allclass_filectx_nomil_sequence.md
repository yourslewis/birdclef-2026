# Model Data Point — PANNs all-class file-context no-MIL ablation

Timestamp: 2026-05-30 06:20 UTC

## Summary

Trained a targeted ablation of the PANNs/Cnn14 all-class train_soundscapes sequence/file-context branch: same radius-2 prev/next/local/file mean+max context as the 00:20 UTC file-context+file-MIL run, but with `file_mil_loss_weight=0.0`. This isolates whether the latest file-MIL loss was responsible for the file-level gain or whether full file context alone was doing the work.

Result: context row AUC improved only slightly over its own row-only control (`0.626315` vs `0.621201`, `+0.005115`), but file-MIL AUC decreased (`0.649487` vs row-only `0.653436`, `-0.003949`). Compared with the prior PANNs file-context+file-MIL run (`0.644272` row / `0.678888` file-MIL), removing the file-MIL loss costs about `-0.017957` row AUC and `-0.029401` file-MIL AUC. Sidecar audit remains below v616 (`-0.002698`), so no submission.

## Ledger

- **Branch family:** train_soundscapes sequence/file/site AudioSet mining, all-class PANNs file-context no-MIL ablation.
- **Training data:** official `train_soundscapes`, 1,478 labeled 5s windows, 66 files, 9 sites.
- **Targets:** all 234 competition labels.
- **Model/init:** frozen PANNs/Cnn14 AudioSet embeddings; MLP head over radius-2 previous/next/local mean/local max/file mean/file max/time features; no site one-hot; no file-MIL loss.
- **Validation split:** leave-one-site, site-balanced sampling; 7 valid folds.
- **Primary metric:** context row macro ROC-AUC `0.626315`.
- **Secondary metrics:**
  - no-train fold mean AUC `0.604598`.
  - non-Aves fold mean AUC `0.685213`.
  - file-MIL fold mean AUC `0.649487`.
  - pooled all-class AUC `0.293246`; pooled no-train AUC `0.184789`.
  - sidecar local AUC `0.990783` / 42 valid; lift vs v616 `-0.002698`; lift vs anchor `+0.000392`.
- **Baseline/delta:** vs PANNs file-context+file-MIL row `-0.017957` / file-MIL `-0.029401`; vs own row-only `+0.005115` row / `-0.003949` file-MIL.
- **Export/runtime status:** TorchScript context head written; finite/nonconstant final predictions (`234/234`); sidecar audit finite (`240 x 234`), `submit_approved=false`.
- **Decision:** **reject/no submission.** File-MIL loss is needed for the useful PANNs file-context file-level signal, but even with it this lane remains below v616 in sidecar audit; avoid more blind PANNs file-context variants.

## Artifacts

- Sequence artifact: `artifacts/soundscape_sequence_mining/soundscape-sequence-panns-cnn14-allcls-r2-filectx-nomil-losite-ep20-20260530/`
- Config: `configs/birdclef/soundscape_sequence_panns_cnn14_allcls_r2_filectx_nomil_losite_ep20_20260530.json`
- Sidecar audit: `artifacts/soundscape_sequence_sidecar_audit/20260530T0620Z_panns_allclass_filectx_nomil_sidecar_audit/`
