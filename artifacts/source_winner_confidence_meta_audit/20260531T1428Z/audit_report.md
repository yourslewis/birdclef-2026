# Source-winner Proto/SED confidence meta audit — 2026-05-31
## Scope
Audits v644/v647 EoS8 source-winner intermediate ProtoSSM/SED train-soundscape streams against the v616 local proxy; trains a tiny leave-site logistic meta calibrator; no Kaggle submission.
## Data
- Proxy rows: 240; label-matched rows: 190; matched files/sites: 20/6; labels: 234 with 42 valid local AUC classes.
## Stream metrics
```text
stream             local_auc   lift_v616   nonaves_auc  top5_recall
anchor_raw        0.990391  -0.003090  0.992347  0.836842
v616_final        0.993481  +0.000000  0.995284  0.636842
source_proto      0.986299  -0.007182  0.991651  0.805263
source_sed        0.995976  +0.002495  0.996866  0.994737
source_rankblend  0.992723  -0.000758  0.994842  0.773684
```
## Leave-site meta
- Meta OOF AUC: 0.990463 / 42 valid; lift vs v616: -0.003018; fitted site-class models: 206; fallback cells: 38504.
## Top sidecar grid results
```text
source            weight   local_auc   lift_v616   site_q05    file_q05    top5
source_sed       0.8000  0.996059  +0.002578  +0.000450  +0.000083  0.742105
source_sed       1.0000  0.995976  +0.002495  +0.000244  -0.000249  0.700000
source_sed       0.6500  0.995822  +0.002341  +0.000332  +0.000239  0.731579
source_sed       0.5000  0.995713  +0.002233  +0.000552  +0.000430  0.721053
source_sed       0.4000  0.995474  +0.001994  +0.000542  +0.000577  0.715789
source_sed       0.3200  0.995086  +0.001605  +0.000509  +0.000591  0.710526
source_sed       0.2400  0.994717  +0.001236  +0.000312  +0.000574  0.710526
source_sed       0.1600  0.994280  +0.000799  +0.000154  +0.000326  0.700000
source_sed       0.0800  0.993766  +0.000285  +0.000053  +0.000042  0.689474
source_rankblend 0.0800  0.993397  -0.000083  -0.000208  -0.000208  0.678947
```
## Decision
Comparison-grade only; submit_approved=false. The raw source SED stream is locally strong, but this audit did not build a hidden-test package and v616-family SED/local-proxy gains are known to over-transfer. Next useful action is a private kernel verifier or a source-code fork that changes EoS8 SED/PowerOpt weights, not a direct competition slot from this CSV.
