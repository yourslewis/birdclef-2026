# Prediction ensemble numeric tables

Source JSON: `/Users/yourslewis/.openclaw/repos/birdclef-2026/specs/birdclef-ensemble-strategy-20260525/reports/prediction_ensemble_numeric_analysis.json`

## Unique usable candidates (sorted by local rank AUC)

| name | AUC | lift | rank corr vs anchor | rank MAE | prob MAE | top3 recall | duplicate? |
|---|---:|---:|---:|---:|---:|---:|---|
| `raunak_sed` | 0.995976 | 0.005585 | 0.778911 | 0.144987 | 0.488670 | 0.973684 |  |
| `v616_submission` | 0.993481 | 0.003090 | 0.999805 | 0.003161 | 0.039663 | 0.942105 |  |
| `samejima_visual_anchor` | 0.990391 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.931579 |  |
| `jungchan_model21` | 0.987426 | -0.002964 | 0.821531 | 0.126921 | 0.299041 | 0.931579 |  |
| `jungchan_protossm` | 0.986253 | -0.004137 | 0.900163 | 0.096253 | 0.437583 | 0.921053 |  |
| `sakur_visual` | 0.984775 | -0.005615 | 0.954954 | 0.061103 | 0.059673 | 0.931579 |  |
| `raunak_protossm` | 0.984640 | -0.005750 | 0.898378 | 0.096941 | 0.403804 | 0.921053 |  |
| `ahmedkhudair121:bc2026-p949-syd-effv2-a03:subm_52p` | 0.983729 | -0.006661 | 0.899258 | 0.096649 | 0.403721 | 0.921053 |  |
| `samejima_protossm` | 0.982155 | -0.008236 | 0.899268 | 0.096823 | 0.403491 | 0.931579 |  |
| `sakur_protossm` | 0.978405 | -0.011986 | 0.866397 | 0.110622 | 0.347998 | 0.915789 |  |

## Most different selected pairs (lowest rank correlation)

| a | b | rank corr | rank MAE | prob MAE |
|---|---|---:|---:|---:|
| `raunak_sed` | `sakur_protossm` | 0.455904 | 0.232427 | 0.160888 |
| `jungchan_model21` | `raunak_sed` | 0.466481 | 0.229944 | 0.239259 |
| `raunak_sed` | `samejima_protossm` | 0.475964 | 0.227503 | 0.096602 |
| `raunak_sed` | `ahmedkhudair121:bc2026-p949-syd-effv2-a03:subm_52p` | 0.481783 | 0.226114 | 0.093682 |
| `raunak_sed` | `jungchan_protossm` | 0.481898 | 0.225862 | 0.057116 |
| `raunak_sed` | `raunak_protossm` | 0.483183 | 0.225711 | 0.092694 |
| `sakur_visual` | `raunak_sed` | 0.757199 | 0.149853 | 0.489270 |
| `samejima_visual_anchor` | `raunak_sed` | 0.778911 | 0.144987 | 0.488670 |
| `raunak_sed` | `v616_submission` | 0.783959 | 0.143364 | 0.489477 |
| `sakur_visual` | `jungchan_model21` | 0.815316 | 0.128541 | 0.295684 |
| `samejima_visual_anchor` | `jungchan_model21` | 0.821531 | 0.126921 | 0.299041 |
| `jungchan_model21` | `v616_submission` | 0.824325 | 0.126321 | 0.311011 |
| `sakur_visual` | `jungchan_protossm` | 0.861044 | 0.112233 | 0.438724 |
| `sakur_visual` | `samejima_protossm` | 0.861166 | 0.111938 | 0.404148 |
| `sakur_visual` | `raunak_protossm` | 0.863907 | 0.111192 | 0.404443 |
| `sakur_visual` | `ahmedkhudair121:bc2026-p949-syd-effv2-a03:subm_52p` | 0.863969 | 0.111008 | 0.404433 |
| `sakur_protossm` | `v616_submission` | 0.864083 | 0.111760 | 0.352977 |
| `samejima_visual_anchor` | `sakur_protossm` | 0.866397 | 0.110622 | 0.347998 |
| `sakur_visual` | `sakur_protossm` | 0.870972 | 0.108136 | 0.346634 |
| `jungchan_model21` | `raunak_protossm` | 0.883513 | 0.082882 | 0.186132 |
| `jungchan_model21` | `ahmedkhudair121:bc2026-p949-syd-effv2-a03:subm_52p` | 0.884679 | 0.082099 | 0.187015 |
| `jungchan_model21` | `jungchan_protossm` | 0.886162 | 0.081987 | 0.209204 |
| `jungchan_model21` | `samejima_protossm` | 0.894202 | 0.078997 | 0.189706 |
| `raunak_protossm` | `v616_submission` | 0.895096 | 0.098696 | 0.405917 |
| `jungchan_model21` | `sakur_protossm` | 0.895467 | 0.074687 | 0.150437 |
| `v616_submission` | `ahmedkhudair121:bc2026-p949-syd-effv2-a03:subm_52p` | 0.895930 | 0.098419 | 0.405791 |
| `samejima_protossm` | `v616_submission` | 0.896010 | 0.098547 | 0.405530 |
| `jungchan_protossm` | `v616_submission` | 0.896801 | 0.098029 | 0.438830 |
| `samejima_visual_anchor` | `raunak_protossm` | 0.898378 | 0.096941 | 0.403804 |
| `samejima_visual_anchor` | `ahmedkhudair121:bc2026-p949-syd-effv2-a03:subm_52p` | 0.899258 | 0.096649 | 0.403721 |
| `samejima_visual_anchor` | `samejima_protossm` | 0.899268 | 0.096823 | 0.403491 |
| `samejima_visual_anchor` | `jungchan_protossm` | 0.900163 | 0.096253 | 0.437583 |
| `samejima_visual_anchor` | `sakur_visual` | 0.954954 | 0.061103 | 0.059673 |
| `sakur_visual` | `v616_submission` | 0.955782 | 0.060729 | 0.083120 |
| `sakur_protossm` | `jungchan_protossm` | 0.956708 | 0.053369 | 0.105252 |
| `raunak_protossm` | `sakur_protossm` | 0.959912 | 0.051016 | 0.069753 |
| `sakur_protossm` | `ahmedkhudair121:bc2026-p949-syd-effv2-a03:subm_52p` | 0.962057 | 0.049755 | 0.068746 |
| `samejima_protossm` | `sakur_protossm` | 0.963864 | 0.048475 | 0.065748 |
| `raunak_protossm` | `samejima_protossm` | 0.993409 | 0.011152 | 0.006864 |
| `raunak_protossm` | `jungchan_protossm` | 0.993614 | 0.012493 | 0.036840 |

## Duplicate matrices

| name | duplicate_of | path |
|---|---|---|
| `samejima_sed` | `raunak_sed` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/anchored_blend_audit/sidecar_grid_inputs_20260525T0200Z/samejima_sed.csv` |
| `v616_subm_21` | `jungchan_model21` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/subm_21.csv` |
| `v616_submission_anchor_raw` | `samejima_visual_anchor` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission_anchor_raw.csv` |
| `v616_submission_before_alignment` | `v616_submission` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission_before_alignment.csv` |
| `v616_submission_jung21_raw` | `jungchan_model21` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission_jung21_raw.csv` |
| `v616_submission_samejima_sed_raw` | `raunak_sed` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission_samejima_sed_raw.csv` |
| `v616_submission_sed` | `raunak_sed` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission_sed.csv` |
| `ahmedkhudair121:bc2026-p949-syd-effv2-a03:subm_21` | `jungchan_model21` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/ahmedkhudair121__bc2026-p949-syd-effv2-a03__subm_21.csv` |
| `ahmedkhudair121:bc2026-p949-syd-effv2-a03:submission_protossm` | `jungchan_protossm` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/ahmedkhudair121__bc2026-p949-syd-effv2-a03__submission_protossm.csv` |
| `ahmedkhudair121:bc2026-p949-syd-effv2-a03:submission_sed` | `raunak_sed` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/ahmedkhudair121__bc2026-p949-syd-effv2-a03__submission_sed.csv` |
| `hanijezo:bc2026-p949-syd-eca-a03:subm_21` | `jungchan_model21` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/hanijezo__bc2026-p949-syd-eca-a03__subm_21.csv` |
| `hanijezo:bc2026-p949-syd-eca-a03:subm_52p` | `ahmedkhudair121:bc2026-p949-syd-effv2-a03:subm_52p` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/hanijezo__bc2026-p949-syd-eca-a03__subm_52p.csv` |
| `hanijezo:bc2026-p949-syd-eca-a03:submission_protossm` | `jungchan_protossm` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/hanijezo__bc2026-p949-syd-eca-a03__submission_protossm.csv` |
| `hanijezo:bc2026-p949-syd-eca-a03:submission_sed` | `raunak_sed` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/hanijezo__bc2026-p949-syd-eca-a03__submission_sed.csv` |
| `hassan1417:bc2026-p949-syd-ort-effv2-a04:subm_21` | `jungchan_model21` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/hassan1417__bc2026-p949-syd-ort-effv2-a04__subm_21.csv` |
| `hassan1417:bc2026-p949-syd-ort-effv2-a04:subm_52p` | `ahmedkhudair121:bc2026-p949-syd-effv2-a03:subm_52p` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/hassan1417__bc2026-p949-syd-ort-effv2-a04__subm_52p.csv` |
| `hassan1417:bc2026-p949-syd-ort-effv2-a04:submission_protossm` | `jungchan_protossm` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/hassan1417__bc2026-p949-syd-ort-effv2-a04__submission_protossm.csv` |
| `hassan1417:bc2026-p949-syd-ort-effv2-a04:submission_sed` | `raunak_sed` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/hassan1417__bc2026-p949-syd-ort-effv2-a04__submission_sed.csv` |
| `hassanalgizani:bc2026-p949-fast-anchor-profile:subm_21` | `jungchan_model21` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/hassanalgizani__bc2026-p949-fast-anchor-profile__subm_21.csv` |
| `hassanalgizani:bc2026-p949-fast-anchor-profile:subm_52p` | `ahmedkhudair121:bc2026-p949-syd-effv2-a03:subm_52p` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/hassanalgizani__bc2026-p949-fast-anchor-profile__subm_52p.csv` |
| `hassanalgizani:bc2026-p949-fast-anchor-profile:submission_protossm` | `jungchan_protossm` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/hassanalgizani__bc2026-p949-fast-anchor-profile__submission_protossm.csv` |
| `hassanalgizani:bc2026-p949-fast-anchor-profile:submission_sed` | `raunak_sed` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/hassanalgizani__bc2026-p949-fast-anchor-profile__submission_sed.csv` |
| `joriahmed:bc2026-p949-syd-ort-effv2-a08:subm_21` | `jungchan_model21` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/joriahmed__bc2026-p949-syd-ort-effv2-a08__subm_21.csv` |
| `joriahmed:bc2026-p949-syd-ort-effv2-a08:subm_52p` | `ahmedkhudair121:bc2026-p949-syd-effv2-a03:subm_52p` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/joriahmed__bc2026-p949-syd-ort-effv2-a08__subm_52p.csv` |
| `joriahmed:bc2026-p949-syd-ort-effv2-a08:submission_protossm` | `jungchan_protossm` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/joriahmed__bc2026-p949-syd-ort-effv2-a08__submission_protossm.csv` |
| `joriahmed:bc2026-p949-syd-ort-effv2-a08:submission_sed` | `raunak_sed` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/joriahmed__bc2026-p949-syd-ort-effv2-a08__submission_sed.csv` |
| `kijiang:birdclef2026-v353:subm_2` | `jungchan_model21` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/kijiang__birdclef2026-v353__subm_2.csv` |
| `kijiang:birdclef2026-v353:submission_protossm` | `ahmedkhudair121:bc2026-p949-syd-effv2-a03:subm_52p` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/kijiang__birdclef2026-v353__submission_protossm.csv` |
| `kijiang:birdclef2026-v353:submission_sed` | `raunak_sed` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/kijiang__birdclef2026-v353__submission_sed.csv` |
| `kijiang:birdclef2026-v354:subm_2` | `jungchan_model21` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/kijiang__birdclef2026-v354__subm_2.csv` |
| `kijiang:birdclef2026-v354:submission_protossm` | `ahmedkhudair121:bc2026-p949-syd-effv2-a03:subm_52p` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/kijiang__birdclef2026-v354__submission_protossm.csv` |
| `kijiang:birdclef2026-v354:submission_sed` | `raunak_sed` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/kijiang__birdclef2026-v354__submission_sed.csv` |
| `mohamadmatali:bc2026-p949-syd-eca-a06:subm_21` | `jungchan_model21` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/mohamadmatali__bc2026-p949-syd-eca-a06__subm_21.csv` |
| `mohamadmatali:bc2026-p949-syd-eca-a06:subm_52p` | `ahmedkhudair121:bc2026-p949-syd-effv2-a03:subm_52p` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/mohamadmatali__bc2026-p949-syd-eca-a06__subm_52p.csv` |
| `mohamadmatali:bc2026-p949-syd-eca-a06:submission_protossm` | `jungchan_protossm` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/mohamadmatali__bc2026-p949-syd-eca-a06__submission_protossm.csv` |
| `mohamadmatali:bc2026-p949-syd-eca-a06:submission_sed` | `raunak_sed` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/mohamadmatali__bc2026-p949-syd-eca-a06__submission_sed.csv` |
| `sans6262q:bc2026-p949-syd-ort-effv2-a02:subm_21` | `jungchan_model21` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/sans6262q__bc2026-p949-syd-ort-effv2-a02__subm_21.csv` |
| `sans6262q:bc2026-p949-syd-ort-effv2-a02:subm_52p` | `ahmedkhudair121:bc2026-p949-syd-effv2-a03:subm_52p` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/sans6262q__bc2026-p949-syd-ort-effv2-a02__subm_52p.csv` |
| `sans6262q:bc2026-p949-syd-ort-effv2-a02:submission_protossm` | `jungchan_protossm` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/sans6262q__bc2026-p949-syd-ort-effv2-a02__submission_protossm.csv` |
| `sans6262q:bc2026-p949-syd-ort-effv2-a02:submission_sed` | `raunak_sed` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/sans6262q__bc2026-p949-syd-ort-effv2-a02__submission_sed.csv` |
| `shahadaljayzani:bc2026-p949-syd-effv2-a06:subm_21` | `jungchan_model21` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/shahadaljayzani__bc2026-p949-syd-effv2-a06__subm_21.csv` |
| `shahadaljayzani:bc2026-p949-syd-effv2-a06:subm_52p` | `ahmedkhudair121:bc2026-p949-syd-effv2-a03:subm_52p` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/shahadaljayzani__bc2026-p949-syd-effv2-a06__subm_52p.csv` |
| `shahadaljayzani:bc2026-p949-syd-effv2-a06:submission_protossm` | `jungchan_protossm` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/shahadaljayzani__bc2026-p949-syd-effv2-a06__submission_protossm.csv` |
| `shahadaljayzani:bc2026-p949-syd-effv2-a06:submission_sed` | `raunak_sed` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/shahadaljayzani__bc2026-p949-syd-effv2-a06__submission_sed.csv` |
| `ykuroka:birdclef-2026-eos6-bz-no-sidecar:submission_protossm` | `jungchan_protossm` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/ykuroka__birdclef-2026-eos6-bz-no-sidecar__submission_protossm.csv` |
| `ykuroka:birdclef-2026-eos6-bz-no-sidecar:submission_sed` | `raunak_sed` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/ykuroka__birdclef-2026-eos6-bz-no-sidecar__submission_sed.csv` |
| `ykuroka:birdclef-2026-yukiz05-nosidecar:submission_protossm` | `jungchan_protossm` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/ykuroka__birdclef-2026-yukiz05-nosidecar__submission_protossm.csv` |
| `ykuroka:birdclef-2026-yukiz05-nosidecar:submission_sed` | `raunak_sed` | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/ykuroka__birdclef-2026-yukiz05-nosidecar__submission_sed.csv` |

## Rejected/unusable CSVs

| name | shape | reason | path |
|---|---:|---|---|
| `ahmedkhudair121:bc2026-p949-syd-effv2-a03:subm_52` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/ahmedkhudair121__bc2026-p949-syd-effv2-a03__subm_52.csv` |
| `ahmedkhudair121:bc2026-p949-syd-effv2-a03:subm_74` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/ahmedkhudair121__bc2026-p949-syd-effv2-a03__subm_74.csv` |
| `ahmedkhudair121:bc2026-p949-syd-effv2-a03:subm_karnakbayev_power_optimization` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/ahmedkhudair121__bc2026-p949-syd-effv2-a03__subm_karnakbayev_power_optimization.csv` |
| `ahmedkhudair121:bc2026-p949-syd-effv2-a03:subm_karnakbayev_power_optimization_74` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/ahmedkhudair121__bc2026-p949-syd-effv2-a03__subm_karnakbayev_power_optimization_74.csv` |
| `ahmedkhudair121:bc2026-p949-syd-effv2-a03:submission` | 243x235 | nonfinite_values:56862 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/ahmedkhudair121__bc2026-p949-syd-effv2-a03__submission.csv` |
| `hanijezo:bc2026-p949-syd-eca-a03:subm_52` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/hanijezo__bc2026-p949-syd-eca-a03__subm_52.csv` |
| `hanijezo:bc2026-p949-syd-eca-a03:subm_74` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/hanijezo__bc2026-p949-syd-eca-a03__subm_74.csv` |
| `hanijezo:bc2026-p949-syd-eca-a03:subm_karnakbayev_power_optimization` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/hanijezo__bc2026-p949-syd-eca-a03__subm_karnakbayev_power_optimization.csv` |
| `hanijezo:bc2026-p949-syd-eca-a03:subm_karnakbayev_power_optimization_74` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/hanijezo__bc2026-p949-syd-eca-a03__subm_karnakbayev_power_optimization_74.csv` |
| `hanijezo:bc2026-p949-syd-eca-a03:submission` | 243x235 | nonfinite_values:56862 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/hanijezo__bc2026-p949-syd-eca-a03__submission.csv` |
| `hassan1417:bc2026-p949-syd-ort-effv2-a04:subm_52` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/hassan1417__bc2026-p949-syd-ort-effv2-a04__subm_52.csv` |
| `hassan1417:bc2026-p949-syd-ort-effv2-a04:subm_74` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/hassan1417__bc2026-p949-syd-ort-effv2-a04__subm_74.csv` |
| `hassan1417:bc2026-p949-syd-ort-effv2-a04:subm_karnakbayev_power_optimization` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/hassan1417__bc2026-p949-syd-ort-effv2-a04__subm_karnakbayev_power_optimization.csv` |
| `hassan1417:bc2026-p949-syd-ort-effv2-a04:subm_karnakbayev_power_optimization_74` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/hassan1417__bc2026-p949-syd-ort-effv2-a04__subm_karnakbayev_power_optimization_74.csv` |
| `hassan1417:bc2026-p949-syd-ort-effv2-a04:submission` | 243x235 | nonfinite_values:56862 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/hassan1417__bc2026-p949-syd-ort-effv2-a04__submission.csv` |
| `hassanalgizani:bc2026-p949-fast-anchor-profile:subm_52` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/hassanalgizani__bc2026-p949-fast-anchor-profile__subm_52.csv` |
| `hassanalgizani:bc2026-p949-fast-anchor-profile:subm_74` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/hassanalgizani__bc2026-p949-fast-anchor-profile__subm_74.csv` |
| `hassanalgizani:bc2026-p949-fast-anchor-profile:subm_karnakbayev_power_optimization` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/hassanalgizani__bc2026-p949-fast-anchor-profile__subm_karnakbayev_power_optimization.csv` |
| `hassanalgizani:bc2026-p949-fast-anchor-profile:subm_karnakbayev_power_optimization_74` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/hassanalgizani__bc2026-p949-fast-anchor-profile__subm_karnakbayev_power_optimization_74.csv` |
| `hassanalgizani:bc2026-p949-fast-anchor-profile:submission` | 243x235 | nonfinite_values:56862 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/hassanalgizani__bc2026-p949-fast-anchor-profile__submission.csv` |
| `jacqueszhelinzhang:birdclef26-deepcnn:submission` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/jacqueszhelinzhang__birdclef26-deepcnn__submission.csv` |
| `joriahmed:bc2026-p949-syd-ort-effv2-a08:subm_52` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/joriahmed__bc2026-p949-syd-ort-effv2-a08__subm_52.csv` |
| `joriahmed:bc2026-p949-syd-ort-effv2-a08:subm_74` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/joriahmed__bc2026-p949-syd-ort-effv2-a08__subm_74.csv` |
| `joriahmed:bc2026-p949-syd-ort-effv2-a08:subm_karnakbayev_power_optimization` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/joriahmed__bc2026-p949-syd-ort-effv2-a08__subm_karnakbayev_power_optimization.csv` |
| `joriahmed:bc2026-p949-syd-ort-effv2-a08:subm_karnakbayev_power_optimization_74` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/joriahmed__bc2026-p949-syd-ort-effv2-a08__subm_karnakbayev_power_optimization_74.csv` |
| `joriahmed:bc2026-p949-syd-ort-effv2-a08:submission` | 243x235 | nonfinite_values:56862 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/joriahmed__bc2026-p949-syd-ort-effv2-a08__submission.csv` |
| `kijiang:birdclef2026-v353:subm_5` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/kijiang__birdclef2026-v353__subm_5.csv` |
| `kijiang:birdclef2026-v353:subm_karnakbayev_power_optimization` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/kijiang__birdclef2026-v353__subm_karnakbayev_power_optimization.csv` |
| `kijiang:birdclef2026-v353:submission` | 243x235 | nonfinite_values:56862 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/kijiang__birdclef2026-v353__submission.csv` |
| `kijiang:birdclef2026-v354:subm_5` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/kijiang__birdclef2026-v354__subm_5.csv` |
| `kijiang:birdclef2026-v354:subm_karnakbayev_power_optimization` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/kijiang__birdclef2026-v354__subm_karnakbayev_power_optimization.csv` |
| `kijiang:birdclef2026-v354:submission` | 243x235 | nonfinite_values:56862 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/kijiang__birdclef2026-v354__submission.csv` |
| `mohamadmatali:bc2026-p949-syd-eca-a06:subm_52` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/mohamadmatali__bc2026-p949-syd-eca-a06__subm_52.csv` |
| `mohamadmatali:bc2026-p949-syd-eca-a06:subm_74` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/mohamadmatali__bc2026-p949-syd-eca-a06__subm_74.csv` |
| `mohamadmatali:bc2026-p949-syd-eca-a06:subm_karnakbayev_power_optimization` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/mohamadmatali__bc2026-p949-syd-eca-a06__subm_karnakbayev_power_optimization.csv` |
| `mohamadmatali:bc2026-p949-syd-eca-a06:subm_karnakbayev_power_optimization_74` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/mohamadmatali__bc2026-p949-syd-eca-a06__subm_karnakbayev_power_optimization_74.csv` |
| `mohamadmatali:bc2026-p949-syd-eca-a06:submission` | 243x235 | nonfinite_values:56862 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/mohamadmatali__bc2026-p949-syd-eca-a06__submission.csv` |
| `sans6262q:bc2026-p949-syd-ort-effv2-a02:subm_52` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/sans6262q__bc2026-p949-syd-ort-effv2-a02__subm_52.csv` |
| `sans6262q:bc2026-p949-syd-ort-effv2-a02:subm_74` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/sans6262q__bc2026-p949-syd-ort-effv2-a02__subm_74.csv` |
| `sans6262q:bc2026-p949-syd-ort-effv2-a02:subm_karnakbayev_power_optimization` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/sans6262q__bc2026-p949-syd-ort-effv2-a02__subm_karnakbayev_power_optimization.csv` |
| `sans6262q:bc2026-p949-syd-ort-effv2-a02:subm_karnakbayev_power_optimization_74` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/sans6262q__bc2026-p949-syd-ort-effv2-a02__subm_karnakbayev_power_optimization_74.csv` |
| `sans6262q:bc2026-p949-syd-ort-effv2-a02:submission` | 243x235 | nonfinite_values:56862 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/sans6262q__bc2026-p949-syd-ort-effv2-a02__submission.csv` |
| `shahadaljayzani:bc2026-p949-syd-effv2-a06:subm_52` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/shahadaljayzani__bc2026-p949-syd-effv2-a06__subm_52.csv` |
| `shahadaljayzani:bc2026-p949-syd-effv2-a06:subm_74` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/shahadaljayzani__bc2026-p949-syd-effv2-a06__subm_74.csv` |
| `shahadaljayzani:bc2026-p949-syd-effv2-a06:subm_karnakbayev_power_optimization` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/shahadaljayzani__bc2026-p949-syd-effv2-a06__subm_karnakbayev_power_optimization.csv` |
| `shahadaljayzani:bc2026-p949-syd-effv2-a06:subm_karnakbayev_power_optimization_74` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/shahadaljayzani__bc2026-p949-syd-effv2-a06__subm_karnakbayev_power_optimization_74.csv` |
| `shahadaljayzani:bc2026-p949-syd-effv2-a06:submission` | 243x235 | nonfinite_values:56862 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/shahadaljayzani__bc2026-p949-syd-effv2-a06__submission.csv` |
| `ykuroka:birdclef-2026-eos6-bz-no-sidecar:subm_karnakbayev_power_optimization` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/ykuroka__birdclef-2026-eos6-bz-no-sidecar__subm_karnakbayev_power_optimization.csv` |
| `ykuroka:birdclef-2026-eos6-bz-no-sidecar:subm_poweropt_model52_full` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/ykuroka__birdclef-2026-eos6-bz-no-sidecar__subm_poweropt_model52_full.csv` |
| `ykuroka:birdclef-2026-eos6-bz-no-sidecar:subm_poweropt_model52_pssm` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/ykuroka__birdclef-2026-eos6-bz-no-sidecar__subm_poweropt_model52_pssm.csv` |
| `ykuroka:birdclef-2026-eos6-bz-no-sidecar:subm_yaroslav_v6_prior065` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/ykuroka__birdclef-2026-eos6-bz-no-sidecar__subm_yaroslav_v6_prior065.csv` |
| `ykuroka:birdclef-2026-eos6-bz-no-sidecar:subm_yaroslav_v6_prior065_core` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/ykuroka__birdclef-2026-eos6-bz-no-sidecar__subm_yaroslav_v6_prior065_core.csv` |
| `ykuroka:birdclef-2026-eos6-bz-no-sidecar:subm_yukiz_perch_proto_res` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/ykuroka__birdclef-2026-eos6-bz-no-sidecar__subm_yukiz_perch_proto_res.csv` |
| `ykuroka:birdclef-2026-eos6-bz-no-sidecar:submission` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/ykuroka__birdclef-2026-eos6-bz-no-sidecar__submission.csv` |
| `ykuroka:birdclef-2026-eos6-bz-no-sidecar:submission_before_all_sidecars` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/ykuroka__birdclef-2026-eos6-bz-no-sidecar__submission_before_all_sidecars.csv` |
| `ykuroka:birdclef-2026-yukiz05-nosidecar:subm_karnakbayev_power_optimization` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/ykuroka__birdclef-2026-yukiz05-nosidecar__subm_karnakbayev_power_optimization.csv` |
| `ykuroka:birdclef-2026-yukiz05-nosidecar:subm_poweropt_model52_full` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/ykuroka__birdclef-2026-yukiz05-nosidecar__subm_poweropt_model52_full.csv` |
| `ykuroka:birdclef-2026-yukiz05-nosidecar:subm_poweropt_model52_pssm` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/ykuroka__birdclef-2026-yukiz05-nosidecar__subm_poweropt_model52_pssm.csv` |
| `ykuroka:birdclef-2026-yukiz05-nosidecar:subm_yaroslav_v6_prior065` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/ykuroka__birdclef-2026-yukiz05-nosidecar__subm_yaroslav_v6_prior065.csv` |
| `ykuroka:birdclef-2026-yukiz05-nosidecar:subm_yaroslav_v6_prior065_core` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/ykuroka__birdclef-2026-yukiz05-nosidecar__subm_yaroslav_v6_prior065_core.csv` |
| `ykuroka:birdclef-2026-yukiz05-nosidecar:subm_yukiz_perch_proto_res` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/ykuroka__birdclef-2026-yukiz05-nosidecar__subm_yukiz_perch_proto_res.csv` |
| `ykuroka:birdclef-2026-yukiz05-nosidecar:submission` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/ykuroka__birdclef-2026-yukiz05-nosidecar__submission.csv` |
| `ykuroka:birdclef-2026-yukiz05-nosidecar:submission_before_all_sidecars` | 3x235 | wrong_shape:3x235 | `/Users/yourslewis/.openclaw/repos/birdclef-2026/artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/ykuroka__birdclef-2026-yukiz05-nosidecar__submission_before_all_sidecars.csv` |
