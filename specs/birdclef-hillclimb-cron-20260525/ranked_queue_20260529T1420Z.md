# Ranked queue — 2026-05-29 14:20 UTC

## Live status
- Public LB best: `0.949`; v616 remains the tied repo-owned baseline to beat; v634 is the only latest v631-v635 tie.
- Latest completed submissions: v631 `0.926`, v632 `0.940`, v633 `0.946`, v634 `0.949`, v635 `0.941`.
- 2026-05-29 UTC slots used: `0/5`; about `9.7h` to reset, so early-day policy is active.
- Active jobs before run: no local/trainer BirdCLEF jobs; trainer GPUs free.

## Current run result
- Trained `soundscape-sequence-panns-cnn14-notrain-rowonly-losite-ep24-20260529`.
  - Data: official train_soundscapes, 1,478 windows / 66 files / 9 sites / 28 no-train labels.
  - Model/features: frozen PANNs/Cnn14 AudioSet embeddings only (`context_radius=0`, no temporal/file/time/site features), site-balanced sampling, 24 epochs.
  - Leave-site result: row AUC `0.573836` / 6 folds; file-MIL `0.567138`; no-train/non-Aves `0.573836`.
  - Comparator: worse than the 12:20 localmax internal row-only baseline by `-0.030624` row and `-0.061209` file-MIL; worse than prior PANNs no-train context by `-0.027469` row and `-0.049011` file-MIL.
- Wrapped row-only OOF predictions as an anchor-preserved 28→234 sidecar and audited against v616 proxy.
  - Best non-control `seq_context_w01`: local AUC `0.990405` / 42 valid, lift vs v616 `-0.003076`, lift vs anchor `+0.000014`, rank corr vs v616 `0.999688`.
  - `submit_approved=false`; no submission.

## Ranked next actions
1. **No-train sonotype class/site movement diagnosis** — row-only, context, and localmax variants are unstable and often below v616 as sidecars; inspect class-site deltas (S08 strengths vs S03/S13/S19/S22/S23 failures), especially labels `47158son10/23/05/21/25/16` and `517063`/`1491113`. Expected LB potential: medium-low; evidence value: high.
2. **Soft1279 head-loaded constrained selector / movement diagnosis** — original head-loaded sidecar remains the only current-day local branch above v616; diagnose per-class/site displacement before any early-day slot. Expected LB potential: medium; evidence value: high.
3. **Hand/stricter no-call negative audit** — farneg10 suppression lift remains tiny; broaden/verify true background negatives before another suppression sidecar. Expected LB potential: medium; evidence value: high.
4. **Sonotype/site-pair specialist validation** — scoped non-Aves/no-train branches repeatedly expose site/sonotype inversion; isolate problematic site pairs or anti-site shortcuts. Expected LB potential: medium-low; data-point value: high.
5. **Late-day public/source slot fill** — only after `<3h` to reset if no verifier-grade candidate exists; source must pass schema/runtime/dedup guards and cap recount.

## Critic / verifier decision
- Critic: explicit row-only export confirmed that removing temporal context does not recover the no-train sonotype sidecar; the branch is weaker than the prior no-train context and the internal row-only baseline from the localmax run, so another blind no-train wrapper is not the best next use.
- Verifier: artifacts are finite/nonconstant and row/column aligned, but the best sidecar is below v616 by `-0.003076`; no Kaggle submission.

## Artifacts
- Ledger: `artifacts/model_data_point_ledger/20260529T1416Z_panns_notrain_rowonly_sequence.md`
- Training metrics: `artifacts/soundscape_sequence_mining/soundscape-sequence-panns-cnn14-notrain-rowonly-losite-ep24-20260529/metrics.json`
- Sidecar audit: `artifacts/soundscape_sequence_sidecar_audit/20260529T1416Z_panns_notrain_rowonly_sidecar/audit_summary.json`
- Canonical performance table updated: `artifacts/model_data_point_ledger/performance_table.md` and `.jsonl`
