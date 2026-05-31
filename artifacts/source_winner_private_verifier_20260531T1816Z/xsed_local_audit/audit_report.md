# EoS8 PowerOpt xSED weight audit

Baseline v616 local AUC: 0.993481 / 42 valid
Local scope: reconstructed PowerOptimization proto/sed branch only; public final/yukiZ files are sample-session sized.

```text
candidate          local_auc  delta_v616  site_q05   file_q05   corr_v616  mae_v616
proto060_sed040    0.992172  -0.001308  -0.005381  -0.004068  0.991423  0.052462
proto040_sed060    0.994267  +0.000787  +0.000324  -0.000226  0.956836  0.070585
proto020_sed080    0.995210  +0.001729  +0.000487  -0.000105  0.879130  0.101581
```

Decision: comparison-grade local verifier only; hidden-safe source forks still require public-session runtime/schema preflight before any competition submission.
