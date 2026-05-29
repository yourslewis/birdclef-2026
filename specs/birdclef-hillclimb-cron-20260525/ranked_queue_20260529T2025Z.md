# Ranked queue — 2026-05-29 20:25 UTC

## Live status
- Public LB best: `0.949`; v616 remains tied repo-owned baseline to beat; v634 is the only latest v631-v635 tie.
- Latest completed submissions: v631 `0.926`, v632 `0.940`, v633 `0.946`, v634 `0.949`, v635 `0.941`.
- 2026-05-29 UTC slots used: `0/5`; ~3.6-3.7h to reset during this run, so mid-day policy still applied. Late-day fill policy activates under 3h if no verifier-grade candidate appears.
- Active jobs before run: no local/trainer BirdCLEF jobs; trainer GPUs free.

## Current run result
- Trained `soundscape-sequence-fused-dymn10-panns-allcls-r2-filectx-filemil-losite-ep20-20260529`.
  - Data: official train_soundscapes, 1,478 windows / 66 files / 9 sites / 234 labels.
  - Model/features: fused DyMN10 + PANNs embeddings, radius-2 prev/next + local mean/max + file mean/max + time features, MLP hidden 384, site-balanced sampling, file-MIL BCE weight 0.35, 20 epochs.
  - Leave-site result: row AUC `0.594204` / 7 folds; file-MIL `0.678623`; no-train `0.574279`; non-Aves `0.645232`.
  - Context lift over same-run row-only: row `+0.021849`, file-MIL `+0.033370`.
  - Comparator: vs fused all-class r2 no-file row `-0.002438` / file `+0.002641`; vs fused localmax row `+0.021442` / file `+0.008627`.
- Wrapped all-class OOF predictions into a v616 proxy sidecar and audited.
  - Best non-control `allcls_seq_w0p0025`: local AUC `0.990981` / 42 valid, lift vs v616 `-0.002499`, lift vs anchor `+0.000591`, rank corr `0.999689`.
  - `submit_approved=false`; no submission.

## Ranked next actions
1. **Soft1279 head-loaded class/site movement diagnosis** — still the only current-day branch with material positive local lift vs v616 (`w0.16` +0.002064), but failed strict anchor/site gates. Expected LB potential: medium; evidence value: highest.
2. **Late-day valid source/code slot fill after <3h to reset** — slots are still `0/5`; if no verifier-grade repo candidate appears, use highest-ranked guarded nonduplicate source candidates when the late policy triggers. Expected LB potential: medium-low; slot value: high near reset.
3. **No-train file-MIL / sonotype movement diagnosis** — file-MIL signals remain stronger than row/sidecar transfer; isolate classes/files/sites before more no-train wrappers. Expected LB potential: medium-low; evidence value: high.
4. **Hand/stricter no-call negative audit** — farneg10 suppression lift is tiny but directionally positive; only continue if negatives become less site-skewed/weak. Expected LB potential: medium; evidence value: medium-high.
5. **Sonotype/site-pair specialist validation** — targeted S08/S22/S23 inversion diagnostics; avoid broad blind wrappers.

## Critic / verifier decision
- Critic: the filectx+file-MIL fused run was a justified one-variable-ish follow-up to explain file-MIL gains, but the result is not promotion-grade: file-MIL improved while row/proxy transfer stayed weak. It should not displace the soft1279 head-loaded positive local lead.
- Verifier: training and sidecar artifacts are finite/nonconstant and row/column aligned; best sidecar lift vs v616 is `-0.002499`; all promotion gates fail. No Kaggle submission.

## Artifacts
- Ledger: `artifacts/model_data_point_ledger/20260529T2025Z_fused_allclass_filectx_filemil_sequence.md`
- Training metrics: `artifacts/soundscape_sequence_mining/soundscape-sequence-fused-dymn10-panns-allcls-r2-filectx-filemil-losite-ep20-20260529/metrics.json`
- Sidecar audit: `artifacts/soundscape_allclass_sidecar_audit/20260529T2025Z_fused_allclass_filectx_filemil/audit_summary.json`
- Canonical performance table updated: `artifacts/model_data_point_ledger/performance_table.md` and `.jsonl`
