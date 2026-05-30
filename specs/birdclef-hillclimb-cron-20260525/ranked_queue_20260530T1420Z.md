# Ranked queue — 2026-05-30 14:20 UTC

## Live status

- Public LB best verified by Bearer API: `0.949` (v616/v621/v622/v623/v634 tied lineage; latest v636-v640 scored `0.944/0.943/0.939/0.944/0.945`).
- 2026-05-30 UTC slots used: `0/5`; time to reset at run start: ~9.7h.
- Active jobs: no local/trainer BirdCLEF jobs; trainer GPUs effectively idle.
- Early UTC day policy: no comparison-grade-only slot fill. No candidate cleared verifier/submission-grade gates this run.

## Evaluated data point this run

Evaluated `soft1279-headloaded-movement-diagnosis-20260530T1420Z`.

- Branch: soft1279-head-loaded native all-class sidecar class/site movement diagnosis for strongest recent candidate (`soft1279init_native_allcls_w0p16`).
- Evaluation data: 240 v616 proxy rows; 190 matched labeled rows; 20 files; 6 sites; 42 valid classes.
- Comparable metrics: anchor AUC `0.990391`, v616 AUC `0.993481`, raw sidecar AUC `0.994941`, w0.16 candidate AUC `0.995545`; lift vs v616 `+0.002064`, lift vs anchor `+0.005155`.
- Attribution: site lift concentrated in `S03` `+0.011364` and `S22` `+0.012645`; `S18` regressed `-0.005000`; `S09` flat. No-train-primary classes averaged only `+0.000190` lift with `0.26` of classes positive.
- Decision: reject as submission candidate; keep diagnostic. The global weight remains comparison-grade and too concentrated for early-day slot use.

## Ranked next actions

1. **Curated multi-site no-call/background negatives** — Expected LB potential: medium; information value: high. Rationale: farneg10 gave a tiny positive suppression clue but farneg20 collapsed to S09-only; next useful step is teacher/hand-audited multi-site background negatives, not stricter distance-only filters.
2. **Robust soft1279 class/site caps from stable winners only** — Expected LB potential: low-medium; information value: high. Use the movement diagnosis to test a non-global cap limited to classes/sites with repeatable lift; do not reuse the prior low-cap selector blindly.
3. **Train-soundscape sequence/file/site mining with a genuinely different encoder/objective** — Expected LB potential: low-medium; information value: medium. Avoid duplicate PANNs/file-context variants; only continue if the model family/objective is meaningfully new.
4. **G124 soft-anchor packaging only with a new transfer hypothesis** — Expected LB potential: low; information value: low-medium. Current teacher-cache gains do not transfer to v616 proxy.
5. **Late-day guarded source fill** — Expected LB potential: low; information value: low-medium. Activate only under `<3h` to reset if no verifier-grade candidate exists and source/runtime/schema guards pass.

## Critic / verifier decision

- Critic: soft1279 global w0.16 is not robust enough; it wins aggregate local AUC but does not directly solve no-train/non-Aves hidden behavior and is site-concentrated.
- Verifier: candidate CSV is finite/aligned and already audited, but `submit_approved=false`; early-day slot policy blocks it.
- Submission decision: no submission this run.
