# Soundscape-positive DyMN10 sequence target-redesign data point — 2026-05-27 10:18 UTC

## Live context
- Kaggle Bearer live check at run start: best public LB remains **0.949**. Latest scored submissions: v621/v622/v623 tied 0.949, v625 0.948, v624 0.943; v616 remains the tied baseline to beat.
- 2026-05-27 UTC slots used at check: **0/5**, ~13.7h to reset. Early-day policy: no submission unless verifier-grade/high-info and non-duplicate.
- No active local or trainer BirdCLEF jobs before this run.

## Experiment
- Experiment id: `soundscape-sequence-dymn10-soundpos-r2-filectx-losite-ep20-20260527`
- Branch family: train_soundscapes sequence/file/site mining; S03/S08-aware target redesign attempt.
- Data: official `train_soundscapes`, 1,478 5s windows / 66 files / 9 sites.
- Target scope: `soundscape_positive`, 75 labels observed in soundscape labels, including 28 no-train labels and 47 non-Aves labels in scope.
- Model/init: cached EfficientAT `dymn10_as` frozen embeddings + radius-2 context MLP with local mean/max, prev/next, file mean, and time features; hidden_dim=256, dropout=0.30, weight_decay=0.001, pos_weight_power=0.30.
- Validation: leave-one-site, 7 completed folds; valid classes per fold 4/19/6/12/17/29/13.

## Metrics
- Row-only mean AUC: `0.514539`.
- Context mean AUC: `0.518121`; delta vs row-only `+0.003582`.
- No-train context AUC: `0.528556` vs row-only `0.494239`.
- Non-Aves context AUC: `0.537355` vs row-only `0.549162`.
- File-MIL context AUC: `0.512164` vs row-only `0.609112`; delta `-0.096949`.
- Fold deltas: S03 -0.035092, S08 -0.097247, S13 -0.078824, S15 +0.090237, S19 +0.036365, S22 +0.078737, S23 +0.030896.

## Sidecar/proxy audit
- Wrapped the 75-label leave-site predictions into the v616 proxy matrix using `scripts/birdclef_soundscape_sequence_sidecar_audit.py` with anchor fill for other classes/rows.
- Build: 240 proxy rows / 234 columns; 156 matched sequence rows and 84 anchor-filled rows; finite=True; nonconstant_columns=234.
- Best audited recipe: `seq_context_w01` local AUC `0.990665` / 42 valid classes; lift vs anchor `+0.000274`, lift vs v616 `-0.002816`, rank corr vs v616 `0.999675`.

## Verifier / Critic decision
- Verifier checks passed for the no-slot artifacts: leave-site predictions finite/nonconstant (`1410x75`, 75/75 nonconstant), final head TorchScript smoke `(2, 75)` finite, sidecar CSV finite/nonconstant `240x234`.
- Critic decision: **reject as slot candidate / keep as comparison-grade data point**. The target redesign slightly improves row AUC and no-train AUC, but file-MIL regresses sharply and the sidecar loses to v616 by ~0.0028 local AUC. Early-day slot use would be leaderboard probing.
- Next exact action: stop direct OOF sidecar wrappers for this family; move to a true hidden-test package/inference path or a different acoustic-context/no-call branch. Revisit late-day public/source slot fill only inside the `<3h` reset window.

## Artifacts
- Model root: `artifacts/soundscape_sequence_mining/soundscape-sequence-dymn10-soundpos-r2-filectx-losite-ep20-20260527/`
- Audit root: `artifacts/soundscape_sequence_sidecar_audit/20260527T1018Z_soundpos/`
- Config: `configs/birdclef/soundscape_sequence_dymn10_soundpos_r2_filectx_losite_ep20_20260527.json`
- Trainer log: `logs/soundscape_sequence_dymn10_soundpos_r2_filectx_losite_ep20_20260527.log`
