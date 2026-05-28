# Model Data Point — B0 OOF-teacher soft1279 soundscape package/audit

Timestamp: 2026-05-28 12:28 UTC

## Summary

Packaged and evaluated the strongest 1,279-row B0 OOF-teacher soft-label control as a TorchScript soundscape sidecar. This run tested whether the strong random-split OOF-teacher validation score transfers into a v616-compatible train_soundscapes proxy branch. It did not clear promotion gates.

## Ledger

- **Model family:** EfficientNet-B0 OOF-teacher SED control / hidden-safe soundscape inference package audit.
- **Source model:** `sed-b0-oofteacher-b0v26-nfnetv29-soft-1279-ep4-20260528`; q3/cap80 external-pretrain encoder init, all 234 taxonomy labels.
- **Packaging/inference:** single TorchScript model manifest, 5s/160-mel preprocessing via `scripts/birdclef_sed_soundscape_infer.py`; inferred the 66 labeled train_soundscapes files into 792 row predictions in 4.332s (0.066s/file) on trainer GPU1.
- **Wrapper/audit data:** v616 proxy matrix, 240 rows x 234 classes; matched 240/240 proxy rows; finite and 234/234 nonconstant columns.
- **Primary metric:** best low-weight sidecar recipe `soft1279_w0p005` local macro AUC `0.990644` over `42` valid classes.
- **Secondary metrics:** direct raw member local AUC `0.587109`; anchor local AUC `0.990391`; v616 local AUC `0.993481`; lift vs anchor `+0.000253`; lift vs v616 `-0.002837`; rank corr vs v616 `0.999678`.
- **Bootstrap/gates:** site bootstrap q05 vs anchor `-0.001305`; file bootstrap q05 vs anchor `-0.000666`. Promotion failed lift-vs-anchor, lift-vs-v616, and stability gates.
- **Baseline/delta:** compared against submitted v616 local proxy (`0.993481`); best sidecar blend delta `-0.002837`.
- **Decision:** **reject as slot candidate / do not submit**. Keep as packaging evidence: strong train-audio OOF-teacher AUC does not transfer as a direct soundscape sidecar without calibration/domain adaptation.

## Artifacts

- Manifest: `artifacts/sed_soundscape_packaging_audit/20260528T1220Z_sed_soft1279_soundscape_package/sed_soft1279_manifest.json`
- Soundscape predictions: `artifacts/sed_soundscape_packaging_audit/20260528T1220Z_sed_soft1279_soundscape_package/train_soundscapes_soft1279.csv`
- Sidecar CSV: `artifacts/sed_soundscape_packaging_audit/20260528T1220Z_sed_soft1279_soundscape_package/sidecars/soft1279_soundscape_sidecar_234_anchorfill.csv`
- Audit JSON: `artifacts/sed_soundscape_packaging_audit/20260528T1220Z_sed_soft1279_soundscape_package/audit/ensemble_strategy_audit.json`
- Build report: `artifacts/sed_soundscape_packaging_audit/20260528T1220Z_sed_soft1279_soundscape_package/sidecar_build_report.json`
