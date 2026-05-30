# Model data point — G124/V2S soft-anchor90 center/localmax-mix target ablation — 2026-05-30 12:20 UTC

## Summary

Trained a G124/V2S-init pseudo-label student that keeps the successful soft-anchor target formulation from the 10:20 UTC run but softens the temporal target from pure neighbor `local_max` to `75%` center + `25%` local-max mix. This tests whether less aggressive temporal target broadening improves teacher-cache and v616-sidecar stability without reintroducing hard-confidence target starvation.

Result: the center/localmax mix is a valid positive diagnostic data point, but it is slightly worse than pure soft-anchor localmax on row validation and remains far below v616 in the local proxy sidecar audit. It is not a submission candidate.

## Model/data contract

- **Experiment id:** `g124-effv2s-public946-pseudo-pilot-20260530-v2sinit-softanchor90-centerlocalmix75-ep6`
- **Family:** EfficientNetV2-RW-S SED noisy-student / G124 reconstruction temporal-target ablation.
- **Init/source:** `artifacts/external_pretrain/xc-v2s-q3-cap80-external-pretrain-balanced-ep12-bestloss/model_torchscript.pt`, encoder loaded with head skipped.
- **Training data:** `792` train_soundscape teacher rows from `teacher_sed85_rankblend15.npz`; split `634` train / `158` val.
- **Targets:** `234` BirdCLEF labels; `target_mode=soft_anchor`; temporal target `center_localmax_mix`, radius `1`, center weight `0.75`; positives `>=0.90`, negatives `<=0.01`, row caps `3` positive / `20` negative, class caps `100` positive / `80` negative.
- **Effective weighted target mask:** fraction `0.509820`; positive cells `12,574`; negative cells `81,909`.
- **Input/training:** 5s audio, 32 kHz, 160 mel bins, EfficientNetV2-RW-S, focal BCE (`gamma=1.5`), sqrt inverse prevalence class weights clipped to `5.0`, 6 epochs, seed `530`.
- **Runtime/export:** trainer GPU CUDA; runtime `27.168s`; TorchScript `88.74 MB`; ONNX `exported` (`1.203 MB`).

## Metrics

- **Best validation AUC:** `0.959950` / 67 valid classes at epoch `6`.
- **All-row student-vs-truth AUC:** `0.962337` / 75 valid classes.
- **Teacher-vs-truth AUC on same rows:** `0.996798` / 75 valid classes.
- **Student/teacher correlation:** `0.856843`; MAE `0.038214`.
- **Student pool blend audit vs teacher cache:** best blend weight `0.04`, AUC `0.997056` / 75 valid, lift vs teacher `+0.00003724`, corr `0.999773`.
- **Teacher-cache stability:** site bootstrap q05 lift `-0.00015185`, p(lift>0) `0.635`; leave-site q05 `-0.00002109`, p(lift>0) `0.889`. Worst held-out site `S15` lift `-0.00004173`.
- **v616 local sidecar audit:** matched `240`/`240` proxy rows, 234/234 nonconstant columns. Best sidecar blend `g124_sidecar_w0p01` local AUC `0.991195` / 42 valid, lift vs v616 `-0.002286`, lift vs anchor `+0.000804`; submit approved `False`.

## Comparison

- Versus prior G124/V2S soft-anchor90 pure localmax (`20260530T1020Z`): val AUC `-0.001691`, all-row AUC `-0.002716`.
- Versus hard-confidence-only localmax (`20260530T0820Z`): val AUC `+0.337099`; target starvation remains fixed.
- v616 proxy sidecar remains strongly negative (`-0.002286`), so teacher-cache micro-lift does not transfer to the current local-v616 gate.

## Critic / verifier decision

- **Evidence level:** comparison-grade diagnostic data point.
- **Critic:** pure localmax remains the better soft-anchor target; center/localmax mix gives slightly more teacher-cache blend lift but worse raw validation and worse v616-sidecar behavior.
- **Verifier:** finite exports/predictions, 240/240 proxy rows matched, 234 nonconstant columns; no hidden/test labels or disallowed data; `submit_approved=false`.
- **Decision:** **reject as slot candidate; keep as diagnostic.** Do not continue G124 temporal-target variants unless a package path beats v616. Next exact action should pivot back to soft1279 head-loaded class/site movement diagnosis or curated multi-site no-call negatives.

## Artifacts

- Config: `configs/birdclef/g124_effv2s_public946_pseudo_pilot_20260530_v2sinit_softanchor90_centerlocalmix75_ep6.json`
- Metrics: `artifacts/pseudolabels/students/g124-effv2s-public946-pseudo-pilot-20260530-v2sinit-softanchor90-centerlocalmix75-ep6/metrics.json`
- Training log: `logs/g124_softanchor90_centerlocalmix75_20260530T1220Z.log`
- Student predictions: `artifacts/pseudolabels/students/g124-effv2s-public946-pseudo-pilot-20260530-v2sinit-softanchor90-centerlocalmix75-ep6/student_predictions.npz`
- Teacher-cache blend audit: `artifacts/pseudolabels/audits/g124_softanchor90_centerlocalmix75_blend_audit_20260530T1220Z.json`
- v616 sidecar audit: `artifacts/model_data_point_ledger/20260530T1220Z_g124_softanchor90_centerlocalmix75_v616_sidecar_audit/audit_smoke20_summary.json`
