# Ranked queue — 2026-05-30 20:20 UTC

## Live status

- Public LB best verified by Bearer API at run start: `0.949`; latest v636-v640 completed `0.944/0.943/0.939/0.944/0.945`.
- 2026-05-30 UTC slots used at run start: `0/5`; estimated time to reset: ~3.75h.
- Active jobs at start: no local/trainer BirdCLEF jobs; trainer GPUs idle (`0 MiB` used on both 4090s).
- Slot policy: still outside the `<3h` late-fill window; no comparison-grade-only slot use. No candidate cleared verifier/submission gates.

## Evaluated/trained data points this run

### PANNs all-class r4 20s local mean+max sequence data point

- Artifact: `artifacts/model_data_point_ledger/20260530T2020Z_panns_allclass_r4_20s_localmeanmax_sequence.md`.
- Data: official train_soundscapes, `1,478` windows / `66` files / `9` sites / `234` labels.
- Model: frozen PANNs/Cnn14 embeddings with radius-4 local mean+max context (±20s), no prev/next singleton features, no file mean/max, time features, context MLP, 20 epochs.
- LOSO metrics: context row AUC `0.627559`; no-train `0.604764`; non-Aves `0.665293`; file-MIL `0.673926`.
- Internal baseline: row-only row `0.626128`, file-MIL `0.671220`; context lift `+0.001430` row / `+0.002706` file-MIL.
- Prior baseline delta: vs PANNs all-class filectx+fileMIL row `0.644272`, delta `-0.016713`; vs localmax-only row `0.641501`, delta `-0.013942`.
- Site deltas: positive on `S03`, `S08`, `S23`; negative on `S13`, `S15`, `S19`, slight negative on `S22`.
- Sidecar audit: best `allcls_seq_w0p0025` local AUC `0.991131` / 42 valid; lift vs anchor `+0.000741`; lift vs v616 `-0.002349`; finite/nonconstant 240x234; `submit_approved=false`.
- Decision: reject/no submission. This is a useful 20s temporal diagnostic but not a slot candidate.

## Comparable top-5 context — PANNs sequence row metrics

1. PANNs all-class filectx+fileMIL: row `0.644272`, file-MIL `0.678888`, sidecar lift vs v616 `-0.002529`.
2. PANNs all-class localmax+fileMIL: row `0.644053`, file-MIL `0.665302`, sidecar lift vs v616 `-0.002117`.
3. PANNs all-class localmax-only: row `0.641501`, file-MIL `0.681753`, sidecar lift vs v616 `-0.001728`.
4. **New r4 20s local mean+max:** row `0.627559`, file-MIL `0.673926`, sidecar lift vs v616 `-0.002349`.
5. PANNs all-class filectx no-MIL: row `0.626315`, file-MIL `0.649487`, sidecar lift vs v616 `-0.002698`.

## Ranked next actions

1. **Late-day guarded source fill when `<3h` to reset** — Expected LB potential: low; information value: low-medium. Since slots are still `0/5`, next cron should activate the late-fill policy if no verifier-grade candidate appears and only use source-clean, schema-safe, nonduplicate candidates.
2. **Hand/teacher-audited multi-site no-call negatives** — Expected LB potential: medium; information value: high. Threshold-only weak negatives are exhausted; build actual negative evidence across more than S09/S18/S22 before another no-call model.
3. **Soft1279 class/site robust recipe with explicit site-risk controls** — Expected LB potential: low-medium; information value: medium. Continue only with explicit penalties/exclusions from movement diagnostics, not another broad selector grid.
4. **Genuinely different encoder/adapter soundscape data point** — Expected LB potential: low-medium; information value: medium. Avoid more PANNs/B0 objective/context knobs unless a new transfer hypothesis exists; consider a compact adapter/foundation encoder only if assets/runtime are clean.
5. **G124/teacher-cache lane** — Expected LB potential: low. Hold until there is a new transfer mechanism; target-shape ablations did not transfer to v616.

## Critic / verifier decision

- Critic: radius-4 20s temporal context answers a queued branch question, but the result is below prior PANNs local/file-context variants and has site-skewed movement. Do not spend a slot.
- Verifier: training/export artifacts exist, final predictions are finite/nonconstant, sidecar audit matched the proxy and failed v616 lift gates. `submit_approved=false`.
- Submission decision: **no submission this run** under mid-day policy; preserve slots for late-fill or a verifier-grade candidate in the next run.
