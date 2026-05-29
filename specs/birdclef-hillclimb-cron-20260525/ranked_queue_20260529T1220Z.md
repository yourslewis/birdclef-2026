# Ranked queue — 2026-05-29 12:20 UTC

## Live status
- Public LB best: `0.949`; v616 remains the tied repo-owned baseline to beat; v634 is the only latest v631-v635 tie.
- Latest completed submissions: v631 `0.926`, v632 `0.940`, v633 `0.946`, v634 `0.949`, v635 `0.941`.
- 2026-05-29 UTC slots used: `0/5`; about `11.7h` to reset, so early-day policy is active.
- Active jobs before run: no local/trainer BirdCLEF jobs; trainer GPUs free.

## Current run result
- Trained `soundscape-sequence-panns-cnn14-notrain-r2-localmaxonly-losite-ep24-20260529`.
  - Data: official train_soundscapes, 1,478 windows / 66 files / 9 sites / 28 no-train labels.
  - Model/features: frozen PANNs/Cnn14 embeddings, radius-2 local-max temporal context, time features, site-balanced sampling, 24 epochs.
  - Leave-site result: context row AUC `0.582799` / 6 folds; file-MIL `0.615630`; no-train/non-Aves `0.582799`.
  - Comparator: row-only internal `0.604460` row / `0.628347` file-MIL, so localmax context hurt `-0.021661` row and `-0.012717` file-MIL.
  - Compared with prior PANNs no-train context: row `-0.018506`, file-MIL `-0.000519`.
- Wrapped scoped OOF predictions as an anchor-preserved 28→234 sidecar and audited against v616 proxy.
  - Best non-control `seq_context_w01`: local AUC `0.990398` / 42 valid, lift vs v616 `-0.003082`, lift vs anchor `+0.000008`, rank corr vs v616 `0.999689`.
  - `submit_approved=false`; no submission.

## Ranked next actions
1. **No-train row-only/localmax class-site movement diagnosis** — localmax context hurt the aggregate but row-only was competitive with the prior PANNs no-train branch; inspect classes/sites where S08 improves and S03/S13 regress. Expected LB potential: medium-low; evidence value: high.
2. **Soft1279 head-loaded constrained selector / movement diagnosis** — original head-loaded sidecar remains the only current-day local branch above v616; diagnose per-class/site displacement before any slot. Expected LB potential: medium; evidence value: high.
3. **Hand/stricter no-call negative audit** — farneg10 suppression lift remains tiny; broaden/verify true background negatives before another suppression sidecar. Expected LB potential: medium; evidence value: high.
4. **Sonotype/site-pair specialist validation** — scoped non-Aves/no-train branches repeatedly expose site/sonotype inversion; isolate problematic site pairs or anti-site shortcuts. Expected LB potential: medium-low; data-point value: high.
5. **Late-day public/source slot fill** — only after `<3h` to reset if no verifier-grade candidate exists; source must pass schema/runtime/dedup guards and cap recount.

## Critic / verifier decision
- Critic: the no-train localmax branch answered a useful question but did not improve transfer; context improved only S08 and regressed the frog-heavy sites, so this is not slot-worthy.
- Verifier: training and sidecar artifacts are finite, nonconstant, and row/column aligned, but the best sidecar is below v616 by `-0.003082`; no Kaggle submission.

## Artifacts
- Ledger: `artifacts/model_data_point_ledger/20260529T1220Z_panns_notrain_localmax_sequence.md`
- Training metrics: `artifacts/soundscape_sequence_mining/soundscape-sequence-panns-cnn14-notrain-r2-localmaxonly-losite-ep24-20260529/metrics.json`
- Sidecar audit: `artifacts/soundscape_sequence_sidecar_audit/20260529T1220Z_panns_notrain_localmax_sidecar/audit_summary.json`
- Canonical performance table updated: `artifacts/model_data_point_ledger/performance_table.md` and `.jsonl`
