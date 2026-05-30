# Model data point — G124/V2S hard-confidence localmax target-shape ablation — 2026-05-30 08:20 UTC

## Summary

Trained a distinct G124/V2S-init pseudo-label student to test whether sparse high-confidence pseudo-label anchors improve the G124 reconstruction lane versus the previous soft `teacher_power=0.85` localmax run. They do not: the hard-confidence target mask is too sparse (`0.995%` of cells), standalone validation collapses, and tiny student blends are flat/slightly negative versus the teacher cache.

## Model/data contract

- **Experiment id:** `g124-effv2s-public946-pseudo-pilot-20260530-v2sinit-hardconf90-localmax-ep6`
- **Family:** EfficientNetV2-RW-S SED noisy-student / G124 reconstruction target-shape ablation.
- **Init/source:** `artifacts/external_pretrain/xc-v2s-q3-cap80-external-pretrain-balanced-ep12-bestloss/model_torchscript.pt`, encoder loaded with head skipped (`786` keys loaded, `2` head keys skipped).
- **Training data:** `792` train_soundscape teacher rows from `teacher_sed85_rankblend15.npz`; split `634` train / `158` val.
- **Targets:** `234` BirdCLEF labels; `local_max` temporal target transform radius `1`; hard-confidence positives `>=0.90`, negatives `<=0.01`, row caps `3` positive / `20` negative, class caps `100` positive / `80` negative.
- **Effective target mask:** fraction `0.009950`; positive cells `618`; negative cells `1,226`.
- **Input/training:** 5s audio, 32 kHz, 160 mel bins, EfficientNetV2-RW-S, focal BCE (`gamma=1.5`), sqrt inverse prevalence weights clipped to `5.0`, 6 epochs, seed `530`.
- **Runtime/export:** trainer GPU CUDA; decode `25.134s`, train runtime `25.143s`; TorchScript `88.74 MB`; ONNX exported (`1.203 MB` + data file on trainer).

## Metrics

- **Best validation AUC:** `0.622851` / 67 valid classes at epoch 6.
- **All-row student-vs-truth AUC:** `0.623120` / 75 valid classes.
- **Teacher-vs-truth AUC on same rows:** `0.995541` / 75 valid classes.
- **Student/teacher correlation:** `0.141234`; MAE `0.420776`.
- **Student pool blend audit vs teacher cache:** best tiny blend weight `0.005`, AUC `0.997018` / 75 valid, lift vs teacher `-0.000000443`; corr vs teacher `0.999764`.
- **Stability:** site bootstrap q05 lift `-0.00000704`; leave-site q05 `-0.00000176`; leave-site positive fraction `0.333`. Only held-out S22 was meaningfully positive (`+0.000109`); S09/S19/S08 were slightly negative.

## Comparison

- Previous G124/V2S soft localmax (`20260526T0634Z`) reached best val AUC `0.960094`, all-row student-vs-truth `0.944720`, student/teacher corr `0.847478`, and only microscopic sidecar lift vs v616 `+0.00000339`.
- Hard-confidence localmax is worse by `-0.337243` val AUC and `-0.321600` all-row AUC, with far lower teacher correlation. The hard target mask is the likely failure mode.

## Critic / verifier decision

- **Evidence level:** comparison-grade negative data point.
- **Decision:** **reject unchanged / no submission.** Hard-confidence sparse target-shape training does not improve the G124 lane and is not package-worthy.
- **Next:** if revisiting G124, use soft-anchor (not hard-only) or source-level S124/G124 artifact recovery; otherwise prioritize soft1279 class/site movement diagnosis and curated multi-site no-call negatives.

## Artifacts

- Config: `configs/birdclef/g124_effv2s_public946_pseudo_pilot_20260530_v2sinit_hardconf90_localmax_ep6.json`
- Metrics: `artifacts/pseudolabels/students/g124-effv2s-public946-pseudo-pilot-20260530-v2sinit-hardconf90-localmax-ep6/metrics.json`
- Training log: `artifacts/pseudolabels/students/g124-effv2s-public946-pseudo-pilot-20260530-v2sinit-hardconf90-localmax-ep6/training_log.jsonl`
- Student predictions: `artifacts/pseudolabels/students/g124-effv2s-public946-pseudo-pilot-20260530-v2sinit-hardconf90-localmax-ep6/student_predictions.npz`
- Blend audit: `artifacts/pseudolabels/audits/g124_hardconf90_localmax_blend_audit_20260530T0820Z.json`
- Trainer log: `logs/g124_hardconf90_localmax_20260530T0820Z.log`
