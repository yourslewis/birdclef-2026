# Model data point — G124/V2S soft-anchor90 localmax target-shape ablation — 2026-05-30 10:20 UTC

## Summary

Trained the follow-up G124/V2S-init pseudo-label student requested by the critic after the hard-confidence-only run failed. This variant keeps the full soft `teacher_power=0.85` local-max target surface, but anchors high-confidence positives/negatives with higher loss weight instead of masking away almost all cells.

Result: soft-anchor fixes the hardconf starvation failure and slightly beats the prior soft localmax G124 training metric, but teacher-cache blend lift remains tiny and site-bootstrap stability is not submission-grade.

## Model/data contract

- **Experiment id:** `g124-effv2s-public946-pseudo-pilot-20260530-v2sinit-softanchor90-localmax-ep6`
- **Family:** EfficientNetV2-RW-S SED noisy-student / G124 reconstruction target-shape ablation.
- **Init/source:** `artifacts/external_pretrain/xc-v2s-q3-cap80-external-pretrain-balanced-ep12-bestloss/model_torchscript.pt`, encoder loaded with head skipped.
- **Training data:** `792` train_soundscape teacher rows from `teacher_sed85_rankblend15.npz`; split `634` train / `158` val.
- **Targets:** `234` BirdCLEF labels; `local_max` temporal target transform radius `1`; `target_mode=soft_anchor`; positives `>=0.90`, negatives `<=0.01`, row caps `3` positive / `20` negative, class caps `100` positive / `80` negative; `soft_label_weight=0.5`, `anchor_positive_weight=2.0`, `anchor_negative_weight=1.0`.
- **Effective weighted target mask:** fraction `0.508310`; positive cells `13,463`; negative cells `80,740`.
- **Input/training:** 5s audio, 32 kHz, 160 mel bins, EfficientNetV2-RW-S, focal BCE (`gamma=1.5`), sqrt inverse prevalence class weights clipped to `5.0`, 6 epochs, seed `530`.
- **Runtime/export:** trainer GPU CUDA; runtime `23.950s`; TorchScript `88.74 MB`; ONNX exported (`1.203 MB`).

## Metrics

- **Best validation AUC:** `0.961641` / 67 valid classes at epoch `6`.
- **All-row student-vs-truth AUC:** `0.965053` / 75 valid classes.
- **Teacher-vs-truth AUC on same rows:** `0.995541` / 75 valid classes.
- **Student/teacher correlation:** `0.856930`; MAE `0.039308`.
- **Student pool blend audit vs teacher cache:** best blend weight `0.02`, AUC `0.997042` / 75 valid, lift vs teacher `+0.00002330`, corr `0.999939`.
- **Stability:** site bootstrap q05 lift `-0.00016009`, p(lift>0) `0.54`; leave-site q05 `-0.00001043`, p(lift>0) `0.89`. Worst held-out site is `S15` (`-0.00001964`); best is `S22` (`+0.00004264`).

## Comparison

- Versus prior G124/V2S soft localmax (`20260526T0634Z`): val AUC `+0.001547`, all-row AUC `+0.020333`, student/teacher corr `+0.009452`.
- Versus hard-confidence-only localmax (`20260530T0820Z`): val AUC `+0.338790`, all-row AUC `+0.341933`; confirms the hardconf failure was target starvation.
- The best teacher-cache blend lift is only `+0.00002330`, and bootstrap q05 is negative, so this is **not** a slot candidate yet.

## Critic / verifier decision

- **Evidence level:** comparison-grade positive diagnostic data point.
- **Decision:** **continue only as a diagnostic / no submission.** Soft-anchor is the correct G124 target-shape direction if revisiting this lane, but promotion requires a real v616 sidecar/package audit and robust site/file gates.
- **Next:** do soft1279 head-loaded class/site movement diagnosis or run a G124 soft-anchor v616-sidecar audit only if packaging time is cheap; do not submit from teacher-cache lift alone.

## Artifacts

- Config: `configs/birdclef/g124_effv2s_public946_pseudo_pilot_20260530_v2sinit_softanchor90_localmax_ep6.json`
- Metrics: `artifacts/pseudolabels/students/g124-effv2s-public946-pseudo-pilot-20260530-v2sinit-softanchor90-localmax-ep6/metrics.json`
- Training log: `logs/g124_softanchor90_localmax_20260530T1020Z.log`
- Student predictions: `artifacts/pseudolabels/students/g124-effv2s-public946-pseudo-pilot-20260530-v2sinit-softanchor90-localmax-ep6/student_predictions.npz`
- Blend audit: `artifacts/pseudolabels/audits/g124_softanchor90_localmax_blend_audit_20260530T1020Z.json`
