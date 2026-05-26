# Model data point — G124/V2S target-design localmax — 2026-05-26 06:34 UTC

## Summary
- Experiment id: `g124-effv2s-public946-pseudo-pilot-20260526-v2sinit-power085-localmax-ep6`.
- Config: `configs/birdclef/g124_effv2s_public946_pseudo_pilot_20260526_v2sinit_power085_localmax_ep6.json`.
- Trainer artifact root: `artifacts/pseudolabels/students/g124-effv2s-public946-pseudo-pilot-20260526-v2sinit-power085-localmax-ep6/` on `yourslewis@192.168.0.10`.
- Local synced metrics/logs: same artifact path without the large model files.

## Model/data contract
- Family: EfficientNetV2-RW-S SED noisy-student / G124 reconstruction lane.
- Init/source: `artifacts/external_pretrain/xc-v2s-q3-cap80-external-pretrain-balanced-ep12-bestloss/model_torchscript.pt`; loaded 786 non-head keys, skipped 2 head keys.
- Train rows: 792 public946 teacher train-soundscape rows; train/val split 634/158.
- Labels/targets: 234 BirdCLEF labels; soft teacher targets from `teacher_sed85_rankblend15.npz`.
- Target construction: `teacher_power=0.85`, `temporal_target_mode=local_max`, neighbor radius 1.
- Input: 5s audio, 32 kHz, 160 mel bins, hop 512.
- Augmentation/loss: no mixup; focal BCE, gamma `1.5`, sqrt inverse prevalence class weights clipped to `5.0`.
- Epochs/runtime: 6 epochs; feature decode `24.557s`, training runtime `24.564s` (reported by script).

## Metrics
- Best validation AUC: `0.960093947` over 62 valid classes at epoch 6.
- All-row student-vs-truth macro AUC: `0.944719618` over 75 valid classes.
- Teacher-vs-truth macro AUC on same all-row frame: `0.995540783` over 75 classes.
- Final student/teacher correlation: `0.847478085`; MAE `0.037472747`.
- Export/runtime: TorchScript `88.74 MB`; ONNX exported (`1.203 MB` + data file on trainer).

## Sidecar audit vs v616
- Audit artifact: `artifacts/anchored_blend_audit/20260526T0615Z_b0_g124_sidecar/audit_vs_v616_fast.json`.
- Base v616 local proxy: macro AUC `0.993480668`, top10 row recall `0.784211` on 190 matched rows / 42 valid classes.
- Best recipe: `soft_b0=0`, `g124_center=0.0025`, `g124_lmax=0.005`.
  - Macro AUC `0.993484059`, lift `+0.00000339` vs base.
  - Corr vs anchor `0.999985925`, MAE `0.00149156`, max abs displacement `0.0070`.
  - Top10 row recall `0.810526`.
- Soft-B0 sidecar did not appear in top recipes; nonzero soft-B0 weights generally reduced macro AUC.

## Critic/verifier decision
- Evidence level: **comparison-grade**, not verifier/submission-grade.
- Diversity value: moderate. G124 localmax changes target construction and remains a distinct V2S/G124 lane; however, it is still teacher-derived and close to the plateau family when blended at safe weights.
- Decision: **no submission/no early-day slot**. Keep G124 as promising but require a clearer verifier package, hard-confidence/power ablation, or hidden-safe code package before spending a slot.
