# Ranked queue — 2026-05-30 12:20 UTC

## Live status

- Public LB best verified by Bearer API: `0.949` (v616/v621/v622/v623/v634 tied lineage; latest v636-v640 scored `0.944/0.943/0.939/0.944/0.945`).
- 2026-05-30 UTC slots used: `0/5`; time to reset at run start: ~11.7h.
- Active jobs: no local/trainer BirdCLEF jobs before launch; trainer GPUs effectively idle.
- This is early UTC day, so no comparison-grade-only slot fill. No candidate cleared verifier/submission-grade gates this run.

## New data point this run

Trained `g124-effv2s-public946-pseudo-pilot-20260530-v2sinit-softanchor90-centerlocalmix75-ep6`.

- Branch: G124/V2S-init soft-anchor temporal target ablation: `75%` center + `25%` localmax mix.
- Metrics: best val AUC `0.959950` / 67; all-row `0.962337` / 75; corr vs teacher `0.856843`.
- Teacher-cache blend: best w`0.04`, AUC `0.997056`, lift vs teacher `+0.00003724`, but site bootstrap q05 `-0.00015185` and leave-site q05 `-0.00002109`.
- v616 local proxy sidecar: matched `240/240`, finite/nonconstant; best sidecar w`0.01` AUC `0.991195` / 42, lift vs v616 `-0.002286`, submit approved `false`.
- Decision: reject slot; keep as diagnostic. Pure soft-anchor localmax remains slightly better on validation, and neither transfers to v616 proxy.

## Ranked next actions

1. **Soft1279 head-loaded class/site movement diagnosis** — Expected LB potential: medium; information value: high. Rationale: original head-loaded sidecar is the only recent branch with local lift vs v616 (`+0.002064`) before strict gates failed. Need class/site attribution, movement caps, and why S/site gates fail rather than more blind retraining.
2. **Curated multi-site no-call/background negatives** — Expected LB potential: medium; information value: high. Farneg10 was a tiny positive suppression clue; farneg20 collapsed to S09-only. Next useful step is hand/teacher-audited multi-site negative protocol, not stricter distance-only filtering.
3. **G124 soft-anchor v616 packaging only if package path changes** — Expected LB potential: low after this run; information value: medium. Teacher-cache micro-lifts do not transfer to v616 local proxy; do not train more target-shape knobs without a packaging/audit hypothesis.
4. **Train-soundscape sequence/file/site mining with a genuinely different encoder/objective** — Expected LB potential: low-medium; information value: medium. Prior PANNs/fused/file-context variants are mostly negative but still map the landscape; avoid duplicate PANNs file-context variants.
5. **Late-day guarded source fill** — Expected LB potential: low; information value: low-medium. Activate only under `<3h` to reset if no verifier-grade candidate exists and source/runtime/schema guards pass.

## Critic / verifier decision

- Critic: G124 temporal-target lane is now low priority; the center/localmax mix does not fix the v616-transfer failure.
- Verifier: all new artifacts finite, 234 columns nonconstant, no hidden/test labels or disallowed data; no submission because `submit_approved=false` and early-day slot policy favors verifier-grade candidates.
