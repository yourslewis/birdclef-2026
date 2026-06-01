# MobileViT-v2-050 soft1279-teacher distill DEV gate — 2026-06-01 22:22 UTC

## Summary

- Experiment: `soundscape-native-mobilevitv2_050-soft1279teacher-distill-losite-allcls-ep6-20260601`
- Branch family: representation-level diversity / distinct MobileViT-v2 front-end distilled from competent soft1279 native-B0 teacher.
- Training data: official `train_soundscapes`, 1,410 5s OOF windows, 66 files, 9 sites, 234 labels.
- Model/init: `mobilevitv2_050`, ImageNet pretrained through timm, soft teacher weight 0.7, BCE with observed-sqrt class weights, site-balanced sampling, 6 epochs.
- Validation: leave-one-site OOF plus canonical 240-row train_soundscape DEV scout vs 0.950 frontier E (`proto*0.60 + sed*0.40` rankblend), 200 site/file bootstraps, neg weights.

## Metrics

| field | value |
|---|---:|
| row AUC mean | 0.653518 |
| file-MIL AUC mean | 0.720675 |
| no-train row AUC | 0.609974 |
| non-Aves row AUC | 0.654489 |
| pooled row macro AUC | 0.665982 / 71 valid |
| pooled no-train AUC | 0.741513 / 28 valid |
| DEV cand_auc | 0.750924 |
| DEV cand_auc_on_E_weak_classes | 0.714601 |
| DEV rank_decorrelation | 0.642349 |
| DEV blend_best_weight | 0.02 |
| DEV blend_best_lift | +0.0000706 |
| DEV blend_site_q05 | -0.000372 |
| DEV blend_file_q05 | -0.000315 |
| DEV_score | 0.001449 |
| gate_pass | false |

## Decision

**DEMOTE / data point only.** MobileViT-v2 is a genuinely different front-end and lands in the same broad distill family pattern as ConvNeXt/RegNet: moderately orthogonal and competent, but not robustly additive to frontier E. Both leave-site and leave-file q05 are negative, so it fails the promotion gate and the non-harmful clause. No Kaggle slot was available anyway (`5/5` used), and this would not qualify for the next reset without a stronger blend hypothesis.

## Artifacts

- Metrics: `artifacts/diversity_scout/mobilevitv2_050_distill_20260601/metrics.json`
- OOF predictions: `artifacts/diversity_scout/mobilevitv2_050_distill_20260601/leave_site_predictions.npz`
- Proxy sidecar: `artifacts/diversity_scout/mobilevitv2_050_distill_20260601/E_mobilevitv2_050_distill.csv`
- DEV scout: `artifacts/diversity_scout/mobilevitv2_050_distill_20260601/scout/diversity_scout_summary.json`
- TorchScript: `artifacts/diversity_scout/mobilevitv2_050_distill_20260601/model_torchscript.pt` (5.242 MB; finite smoke OK)
