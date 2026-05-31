# EoS8 PowerOpt xSED weight audit

Baseline v616 local AUC: 0.993481 / 42 valid
Local scope: reconstructed PowerOptimization proto/sed branch only; public final/yukiZ files are sample-session sized.

```text
candidate          local_auc  delta_v616  site_q05   file_q05   corr_v616  mae_v616
proto080_sed020    0.990031  -0.003450  -0.011526  -0.007532  0.963329  0.063863
proto070_sed030    0.991227  -0.002253  -0.008898  -0.006307  0.983168  0.054012
proto060_sed040    0.992172  -0.001308  -0.005381  -0.004068  0.991423  0.052462
proto050_sed050    0.993290  -0.000191  -0.001668  -0.001458  0.982309  0.058984
```

Decision: comparison-grade local verifier only; hidden-safe source forks still require public-session runtime/schema preflight before any competition submission.
