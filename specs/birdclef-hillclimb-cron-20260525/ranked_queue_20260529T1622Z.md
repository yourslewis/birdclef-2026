# Ranked queue — 2026-05-29 16:22 UTC

## Live status
- Public LB best: `0.949`; v616 remains the tied repo-owned baseline to beat; v634 is the only latest v631-v635 tie.
- Latest completed submissions: v631 `0.926`, v632 `0.940`, v633 `0.946`, v634 `0.949`, v635 `0.941`.
- 2026-05-29 UTC slots used: `0/5`; about `7.7h` to reset, so early/mid-day policy is active.
- Active jobs before run: no local/trainer BirdCLEF jobs; trainer GPUs free.

## Current run result
- Trained `soundscape-sequence-fused-dymn10-panns-notrain-r2-nofile-losite-ep24-20260529`.
  - Data: official train_soundscapes, 1,478 windows / 66 files / 9 sites / 28 no-train labels.
  - Model/features: fused EfficientAT DyMN10 + PANNs/Cnn14 embeddings, radius-2 prev/next + local mean/max + time features, MLP hidden 256, site-balanced sampling, 24 epochs.
  - Leave-site result: row AUC `0.554429` / 6 folds; file-MIL `0.660711`; no-train/non-Aves `0.554429`.
  - Context lift over same-run row-only: row `+0.066813`, file-MIL `+0.044756`.
  - Comparator: weaker row than PANNs no-train context by `-0.046876`, but stronger file-MIL by `+0.044562`; strongest current 28-label no-train file-MIL clue, but not row-strong.
- Wrapped fused no-train OOF predictions as an anchor-preserved 28→234 sidecar and audited against v616 proxy.
  - Best non-control `seq_context_w01`: local AUC `0.990398` / 42 valid, lift vs v616 `-0.003083`, lift vs anchor `+0.000007`, rank corr vs v616 `0.999687`.
  - `submit_approved=false`; no submission.

## Ranked next actions
1. **No-train file-MIL / sonotype movement diagnosis** — fused no-train is row-weak but file-MIL-best; inspect which files/classes create the file-MIL gain and why rank sidecar is still below v616. Expected LB potential: medium-low; evidence value: high.
2. **Soft1279 head-loaded constrained selector / movement diagnosis** — original head-loaded sidecar remains the only current-day local branch above v616; diagnose per-class/site displacement before any early-day slot. Expected LB potential: medium; evidence value: high.
3. **Hand/stricter no-call negative audit** — farneg10 suppression lift remains tiny; broaden/verify true background negatives before another suppression sidecar. Expected LB potential: medium; evidence value: high.
4. **Sonotype/site-pair specialist validation** — scoped non-Aves/no-train branches repeatedly expose S08/S22/site inversion; isolate problematic site pairs or anti-site shortcuts. Expected LB potential: medium-low; data-point value: high.
5. **Late-day public/source slot fill** — only after `<3h` to reset if no verifier-grade candidate exists; source must pass schema/runtime/dedup guards and cap recount.

## Critic / verifier decision
- Critic: the fused branch answers a useful question (file-level fusion can help no-train labels) but its v616 sidecar is locally below baseline; an early-day submission would be probing, not a gated candidate.
- Verifier: artifacts are finite/nonconstant and row/column aligned through audit, but the best sidecar is below v616 by `-0.003083`; no Kaggle submission.

## Artifacts
- Ledger: `artifacts/model_data_point_ledger/20260529T1622Z_fused_notrain_sequence.md`
- Training metrics: `artifacts/soundscape_sequence_mining/soundscape-sequence-fused-dymn10-panns-notrain-r2-nofile-losite-ep24-20260529/metrics.json`
- Sidecar audit: `artifacts/soundscape_sequence_sidecar_audit/20260529T1618Z_fused_notrain_sidecar/audit_summary.json`
- Canonical performance table updated: `artifacts/model_data_point_ledger/performance_table.md` and `.jsonl`
