# EoS8 PowerOptimization xSED source-fork verifier — 2026-05-31 18:16 UTC

## Scope
Audited and packaged hidden-safe source forks of the v647 EoS8/PowerOptimization source by changing the inner ProtoSSM/SED xSED rank weights. Public-session kernels were private-pushed, completed, and schema-verified before competition submission.

## Data and validation
- Local proxy: 240 v616/source rows, 190 label-matched train-soundscape rows, 20 files, 6 sites (`S03`, `S08`, `S09`, `S13`, `S18`, `S22`).
- Target scope: 234 labels; 42 local-valid AUC classes; non-Aves/no-train slices available.
- Local metric: reconstructed PowerOptimization proto/sed branch only; public final/yukiZ files are sample-session sized, so outer hidden behavior is verified via source runtime and Kaggle code submission.
- Public-session verifier: both private kernels `COMPLETE`; `submission.csv` finite/nonconstant 3x235; hidden submissions refs `53228552`/`53228555` are now scored.

## Performance table
```text
candidate          local_auc  delta_v616  site_q05   file_q05   nonAves   noTrain   rank_corr  public_verifier  submission
control/proto060_sed040 0.992172  -0.001308  -0.005381  -0.004068  0.994569  0.996838  0.991423  n/a                       no submit
v652/proto040_sed060 0.994267  +0.000787  +0.000324  -0.000226  0.996324  0.997397  0.956836  ok hash 734f7b0ef74fd52b  public LB 0.948 ref 53228552
v651/proto020_sed080 0.995210  +0.001729  +0.000487  -0.000105  0.996484  0.997161  0.879130  ok hash 85da9a4d397d2aea  public LB 0.941 ref 53228555
```

## Critic / verifier decision
- `v652` (proto0.40/sed0.60) is the safer first slot: modest local lift, smaller movement than v651, public verifier hash `734f7b0ef74fd52b`.
- `v651` (proto0.20/sed0.80) is the stronger local/high-information slot: larger lift but high displacement, public verifier hash `85da9a4d397d2aea`.
- Both were submitted because UTC usage was 0/5, reset was ~5.5h away, and both are hidden-safe source forks rather than static/fallback CSVs.
- Score readout: `v652` scored `0.948` and `v651` scored `0.941`, both below the live `0.950` frontier (`v644`/`v647`). The local SED-heavy xSED improvements did not transfer; demote this fork direction unless new hidden evidence appears.

## Artifacts
- Local audit: `artifacts/source_winner_private_verifier_20260531T1816Z/xsed_local_audit/audit_summary.json`
- Public verifier status: `artifacts/source_winner_private_verifier_20260531T1816Z/session_outputs/verifier_status.json`
- Submit report: `artifacts/source_winner_private_verifier_20260531T1816Z/submit_v651_v652_report.json`
- Kernel folders: `kaggle-kernels/v651-eos8-sedheavy-proto020/`, `kaggle-kernels/v652-eos8-sedheavy-proto040/`

