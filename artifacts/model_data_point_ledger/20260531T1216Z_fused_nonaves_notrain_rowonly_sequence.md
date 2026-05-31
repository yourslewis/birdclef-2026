# BirdCLEF model data point — fused DyMN10+PANNs 72-label row-only sequence — 2026-05-31 12:16 UTC

## Summary
Trained/evaluated `soundscape-sequence-fused-dymn10-panns-nonaves-notrain-rowonly-losite-ep24-20260531` as a targeted train_soundscapes row-only landscape point.

## Role / hypothesis
- **Coordinator:** test whether fused DyMN10+PANNs embeddings add complementary file/site signal after PANNs-only row-only looked locally strong but sidecar-weak.
- **Data & feature scientist:** official `train_soundscapes` only, preserved as files/sites; temporal/file context disabled to isolate encoder fusion.
- **Critic:** comparison-grade data point, not a slot probe; promotion requires local proxy lift vs v616 plus group robustness.
- **Verifier:** artifacts finite/nonconstant/aligned; no submission approved.

## Data / model
- Data: official train_soundscapes, `1,478` windows / `66` files / `9` sites.
- Target scope: `72` non-Aves/no-train labels.
- Model/init: frozen fused EfficientAT DyMN10 + PANNs/Cnn14 embeddings (`context_dim=3008`) + row-only MLP h384/dropout0.40.
- Split: leave-one-site, `6` valid folds; final all-row train runtime `6.134s`.

## Comparable performance table

| UTC | Experiment | Branch family | Data / targets | Split | Primary metric | Secondary metrics | Baseline / delta | Export | Decision | Artifact |
|---|---|---|---|---|---:|---|---|---|---|---|
| 2026-05-31 12:16 | `soundscape-sequence-fused-dymn10-panns-nonaves-notrain-rowonly-losite-ep24-20260531` | Sequence/file/site targeted fused AudioSet mining | 1,478 windows / 66 files / 9 sites; 72 labels | leave-one-site | row AUC `0.616166` / 6 folds | no-train `0.491181`; non-Aves `0.616166`; file-MIL `0.723917`; sidecar N/A | vs PANNs 72 row-only `-0.058319` row / `+0.032761` file-MIL; vs PANNs 72 filectx `-0.015425` row / `+0.033912` file-MIL | TorchScript smoke OK; final 72/72 nonconstant | keep file-MIL clue; reject direct slot | `artifacts/soundscape_sequence_mining/soundscape-sequence-fused-dymn10-panns-nonaves-notrain-rowonly-losite-ep24-20260531/metrics.json` |
| 2026-05-31 12:19 | fused 72→234 sidecar audit | Sequence/file/site scoped wrapper audit | 240 proxy rows; 156 matched rows; 72→234 anchor-filled | v616 local proxy + 200 boot | best sidecar AUC `0.990059` / 42 valid | lift vs v616 `-0.003422`; lift vs anchor `-0.000332`; rank corr `0.999613` | v616 local `0.993481`; below baseline | finite/nonconstant 240x234; submit_approved=false | reject slot candidate | `artifacts/model_data_point_ledger/20260531T1216Z_fused_nonaves_notrain_rowonly_sidecar_audit/audit_summary.json` |

## Interpretation
Fusion hurt row-level generalization compared with PANNs-only 72-label row-only (`0.616166` vs `0.674485`) and underperformed the 72-label filectx+fileMIL context row metric (`0.631592`). It improved file-MIL (`0.723917`), so the encoder mix may carry per-file presence signal, but the wrapped sidecar regressed vs v616.

## Decision
`REJECT` as direct/sidecar submission; `KEEP` as measured file-MIL landscape point. Next exact action: v950 PowerOptimization/source-winner confidence verifier, not another blind PANNs/fusion wrapper.
