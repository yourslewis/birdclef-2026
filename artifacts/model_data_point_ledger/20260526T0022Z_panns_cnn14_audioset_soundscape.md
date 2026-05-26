# Model Data Point — PANNs/Cnn14 AudioSet Soundscape Non-Aves / No-Train / No-Call

Timestamp: 2026-05-26 00:22 UTC

## Summary

Trained a bounded PANNs/Cnn14 AudioSet embedding-head branch as a measured search-landscape data point. This probes whether external AudioSet semantic embeddings help the non-Aves/no-train/no-call hidden-behavior slice better than repo-owned B0 soundscape training.

## Ledger

- **Model family:** PANNs/Cnn14 AudioSet-pretrained audio tagging model for frozen 2048-d embeddings; small MLP multilabel head with no-call auxiliary output.
- **Init/source:** public `panns-inference==0.1.1`; Cnn14 AudioSet checkpoint downloaded to `/home/yourslewis/panns_data/Cnn14_mAP=0.431.pth` from the package's default Zenodo URL.
- **Train rows:** 1,478 official `train_soundscapes` 5s windows.
- **Labels/targets:** 72 `nonaves_or_no_train` labels from taxonomy; multilabel targets from `train_soundscapes_labels.csv`; 5,420 positive target cells; no-call auxiliary target has 30 positive rows (rows with zero scoped labels).
- **Input window/features:** 5s waveform at 32 kHz; PANNs/Cnn14 clipwise AudioSet logits and 2048-d embeddings.
- **Augmentations:** none; frozen embedding extraction plus MLP dropout `0.15`.
- **Loss:** BCE multilabel loss + `0.2`× BCE no-call auxiliary loss.
- **Epochs/runtime:** 12 epochs for the MLP head; embedding extraction 49.84s CUDA after checkpoint download; best val loss `0.45604` at epoch 5.
- **Validation/proxy:** site-holdout `S08` (120 windows); macro AUC `0.517333` over 18 valid scoped classes; no-train macro AUC `0.520824` over 17 valid classes; non-Aves macro AUC `0.517333`; no-call AUC invalid on S08 because validation target lacked both classes.
- **Prediction artifacts:** `artifacts/panns_soundscape_embeddings/panns-cnn14-audioset-soundscape-nonaves-notrain-nocall-siteS08-ep12-20260526/holdout_predictions.npz`.
- **Export/runtime status:** MLP head saved as `embedding_head.pt` and `embedding_head_torchscript.pt`; trainer CPU TorchScript smoke passed with `(2,2048)` input -> `(2,72)` label logits and `(2,1)` no-call logits. Holdout predictions are finite, nonconstant, and label-aligned.
- **Correlation/blend vs anchor/v616:** not run. Branch is not yet a row-aligned 234-class output and cannot be blended/submitted directly.
- **Diversity value:** high model-source diversity (external AudioSet Cnn14, not v616/Perch/ProtoSSM/SED plateau); high slice relevance for non-Aves/no-train/no-call; only modest validation signal.
- **Critic decision:** useful landscape data point; do not scale unchanged. Continue only with leave-one-site/no-call-valid split or as a tiny capped sidecar wrapper with no-slot v616 audit.
- **Verifier decision:** no-slot training is rule-safe; output is not competition format; no submission approved.

## Artifact paths

- Config: `configs/birdclef/panns_cnn14_audioset_soundscape_nonaves_notrain_nocall_siteS08_ep12_20260526.json`
- Script: `scripts/birdclef_panns_soundscape_embedding_train.py`
- Metrics: `artifacts/panns_soundscape_embeddings/panns-cnn14-audioset-soundscape-nonaves-notrain-nocall-siteS08-ep12-20260526/metrics.json`
- Holdout predictions: `artifacts/panns_soundscape_embeddings/panns-cnn14-audioset-soundscape-nonaves-notrain-nocall-siteS08-ep12-20260526/holdout_predictions.npz`
- TorchScript head: `artifacts/panns_soundscape_embeddings/panns-cnn14-audioset-soundscape-nonaves-notrain-nocall-siteS08-ep12-20260526/embedding_head_torchscript.pt`
- Remote/local log: `logs/panns_cnn14_audioset_soundscape_nonaves_notrain_nocall_siteS08_ep12_20260526.log`
