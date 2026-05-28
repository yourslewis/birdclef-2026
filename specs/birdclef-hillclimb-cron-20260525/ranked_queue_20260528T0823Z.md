# BirdCLEF Hill-Climb Ranked Queue — 20260528T0823Z

## Live status
- Best public LB remains **0.949** (v616/v621/v622/v623 tied best; latest v626-v630 all below best: `0.899/0.928/0.940/0.946/0.917`).
- 2026-05-28 UTC slots: **0/5 used**; early UTC-day policy applies (~15.6h to reset at post-run check).
- Active BirdCLEF jobs before launch: none local/trainer; trainer accepted the bounded CUDA_VISIBLE_DEVICES=1 run.
- New work this run: added file-MIL regularization support to `scripts/birdclef_soundscape_sequence_mining.py`, trained/evaluated `PANNs/Cnn14 all-class localmax + file-MIL`, and ran a 234-class v616 proxy sidecar audit.

## New result summary
- Model row AUC: **0.644053** LOSO / 7 folds; file-MIL **0.665302**; no-train **0.613032**; non-Aves **0.670490**.
- Deltas: vs row-only **+0.026498** row but **-0.010328** file-MIL; vs prior PANNs all-class localmax **+0.002552** row and **-0.016451** file-MIL.
- Best sidecar recipe: `allcls_seq_w0p0025`, local proxy AUC **0.991363** / 42 valid; lift vs anchor **+0.000973**, lift vs v616 **-0.002117**.
- Decision: **no submission**. The intended file-MIL improvement failed, and v616 sidecar promotion is still negative.

## Comparable top-5 by leave-site row AUC
1. `Native B0 soundscape-positive` — row **0.658165**, file-MIL **0.676383**; 75-class target, no direct package.
2. `PANNs/Cnn14 all-class r2 no-file` — row **0.647816**, file-MIL **0.670723**; best broad PANNs row baseline.
3. `PANNs/Cnn14 all-class localmax + file-MIL` — row **0.644053**, file-MIL **0.665302**; new data point, rejected for file-MIL/sidecar regression.
4. `PANNs/Cnn14 soundpos localmax` — row **0.642375**, file-MIL **0.662504**; 75-class target, sidecar weak.
5. `PANNs/Cnn14 all-class r2 file-context` — row **0.642202**, file-MIL **0.652651**; file context hurt file-MIL.

## Ranked queue after this run
1. **No-call/background protocol audit** — now higher than more PANNs localmax variants. Need trusted background/no-call coverage, no-call calibration diagnostics, and reject rules before training suppressors.
2. **Hidden-safe package/wrapper for PANNs/native clues** — only if it genuinely operates on hidden test rows and is not another OOF/proxy replay. PANNs localmax and native soundpos remain useful clues, but direct sidecars keep losing to v616.
3. **Site-robust native/PANNs hybrid target revision** — native soundpos remains strongest row AUC; pursue only with explicit S08/site robustness and packageability changes.
4. **Broader OOF negative/no-call SED student** — train as a distinct data point if no-call protocol defines safe targets.
5. **Late-day public/source slot fill** — only inside <3h to reset, after schema/runtime/dedup/source guards; yesterday's public fills were weak, so this remains last resort.

## Critic / red-team decision
- The file-MIL branch is **not** a promotion candidate: it increased row AUC slightly but decreased the target file-MIL metric and worsened sidecar lift vs the prior PANNs localmax sidecar.
- Repeating direct OOF/proxy sidecars is now low-value unless the hidden-test wrapper changes the inference path. The next useful data point should be no-call/background or hidden-safe packaging, not another minor PANNs feature tweak.

## Verifier decision
- Model guards passed: finite/nonconstant OOF predictions `1410x234`, final predictions `1478x234`, TorchScript context head exported, `234/234` nonconstant columns.
- Sidecar guards passed: finite `240x234`, 156/240 proxy rows matched, 234 nonconstant columns.
- Submission gate failed: best lift vs v616 is **-0.002117**, so **reject slot candidate**.

## Artifacts
- Model ledger: `artifacts/model_data_point_ledger/20260528T0823Z_panns_cnn14_allclass_localmax_filemil_sequence.md`
- Sidecar ledger: `artifacts/model_data_point_ledger/20260528T0823Z_panns_localmax_filemil_sidecar_audit.md`
- Metrics: `artifacts/soundscape_sequence_mining/soundscape-sequence-panns-cnn14-allcls-r2-localmax-filemil-losite-ep20-20260528/metrics.json`
- Audit: `artifacts/soundscape_sequence_sidecar_audit/20260528T0820Z_panns_localmax_filemil/audit_summary.json`
- Logs: `logs/soundscape_sequence_panns_cnn14_allcls_localmax_filemil_losite_ep20_20260528.log`, `logs/soundscape_sequence_panns_cnn14_allcls_localmax_filemil_sidecar_audit_20260528T0820Z.log`
