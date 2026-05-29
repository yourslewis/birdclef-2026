# Ranked queue — 2026-05-29 18:20 UTC

## Live status
- Public LB best: `0.949`; v616 remains tied repo-owned baseline to beat; v634 is the only latest v631-v635 tie.
- Latest completed submissions: v631 `0.926`, v632 `0.940`, v633 `0.946`, v634 `0.949`, v635 `0.941`.
- 2026-05-29 UTC slots used: `0/5`; about `5.7h` to reset, so mid-day policy is active.
- Active jobs before run: no local/trainer BirdCLEF jobs; trainer GPUs free.

## Current run result
- Trained `soundscape-sequence-fused-dymn10-panns-allcls-r2-localmaxonly-losite-ep20-20260529`.
  - Data: official train_soundscapes, 1,478 windows / 66 files / 9 sites / 234 labels.
  - Model/features: fused DyMN10 + PANNs embeddings, localmax-only radius-2 temporal feature, time features, MLP hidden 384, site-balanced sampling, 20 epochs.
  - Leave-site result: row AUC `0.572762` / 7 folds; file-MIL `0.669996`; no-train `0.550756`; non-Aves `0.632578`.
  - Context lift over same-run row-only: row `+0.025190`, file-MIL `+0.024853`.
  - Comparator: weaker than fused all-class r2 no-file by row `-0.023880` / file `-0.005986`, and weaker than PANNs all-class localmax by row `-0.068739` / file `-0.011757`.
- Wrapped all-class OOF predictions into a v616 proxy sidecar and audited.
  - Best non-control `allcls_seq_w0p005`: local AUC `0.991500` / 42 valid, lift vs v616 `-0.001981`, lift vs anchor `+0.001109`, rank corr `0.999671`.
  - `submit_approved=false`; no submission.

## Ranked next actions
1. **Soft1279 head-loaded constrained selector / class-site movement diagnosis** — still the only current-day branch with material positive local lift vs v616 (`w0.16` +0.002064), but failed strict anchor/site gates. Expected LB potential: medium; evidence value: high.
2. **No-train file-MIL / sonotype movement diagnosis** — fused no-train file-MIL is strong while sidecar is below v616; identify classes/files/sites creating file-MIL gain and inversion. Expected LB potential: medium-low; evidence value: high.
3. **Hand/stricter no-call negative audit** — farneg10 suppression lift remains tiny but directionally positive; broaden/verify true background negatives before another suppression sidecar. Expected LB potential: medium; evidence value: high.
4. **Sonotype/site-pair specialist validation** — isolate S08/S22/S23 inversions and class-pair shortcuts before more no-train wrappers. Expected LB potential: medium-low; data-point value: high.
5. **Late-day public/source slot fill** — only after `<3h` to reset if no verifier-grade candidate exists; source must pass schema/runtime/dedup guards and cap recount.

## Critic / verifier decision
- Critic: localmax-only fused all-class is a valid one-variable data point, but it underperforms both prior fused context and PANNs localmax. It should not displace the soft1279 positive local lead.
- Verifier: sidecar artifacts are finite/nonconstant and row/column aligned, but best lift vs v616 is `-0.001981` and all promotion gates fail; no Kaggle submission.

## Artifacts
- Ledger: `artifacts/model_data_point_ledger/20260529T1820Z_fused_allclass_localmax_sequence.md`
- Training metrics: `artifacts/soundscape_sequence_mining/soundscape-sequence-fused-dymn10-panns-allcls-r2-localmaxonly-losite-ep20-20260529/metrics.json`
- Sidecar audit: `artifacts/soundscape_allclass_sidecar_audit/20260529T1815Z_fused_allclass_localmax/audit_summary.json`
- Canonical performance table updated: `artifacts/model_data_point_ledger/performance_table.md` and `.jsonl`
