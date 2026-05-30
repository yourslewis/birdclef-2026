# Ranked queue — 2026-05-30 18:22 UTC

## Live status

- Public LB best verified by Bearer API at run start: `0.949` (v616/v621/v622/v623/v634 tied lineage; latest v636-v640 scored `0.944/0.943/0.939/0.944/0.945`).
- 2026-05-30 UTC slots used at run start: `0/5`; time to reset: ~5.7h.
- Active jobs at start: no local/trainer BirdCLEF jobs; trainer GPUs effectively idle.
- Slot policy: mid UTC-day, no comparison-grade-only slot fill. No candidate cleared verifier/submission-grade gates this run.

## Evaluated/trained data points this run

### 1. Soft1279 stable-winner capped selector

- Artifact: `artifacts/model_data_point_ledger/20260530T1816Z_soft1279_stable_winner_caps.md`.
- Data: 190 matched proxy rows / 20 files / 6 sites / 42 valid local classes.
- Method: stricter per-class selector over the head-loaded soft1279 raw sidecar; weights up to 0.12 but only classes with >=5 train positives, >=20 train negatives, and >=0.002 train lift can move.
- Site-CV: AUC `0.993629`, lift vs v616 `+0.000148`, leave-site q05 `-0.000054`.
- File-CV: AUC `0.994132`, lift `+0.000651`; top3 recall `0.947368` vs base `0.942105`.
- All-row diagnostic: lift `+0.000986`; selected classes `116570`, `chacha1`, `555146`, `undtin1`.
- Decision: reject/no submission. It recovers some file-CV signal, but site robustness remains too small and below promotion gates.

### 2. Native B0 soft1279-init focal2 all-class data point + package audit

- Artifact: `artifacts/model_data_point_ledger/20260530T1820Z_soundscape_native_soft1279init_focal2_allclass.md`.
- Training data: official train_soundscapes, 1,478 windows / 66 files / 9 sites / 234 labels.
- Model/init: EfficientNet-B0 SED, soft1279 OOF-teacher checkpoint with head loaded, focal BCE gamma 2.0, full model trainable.
- LOSO row AUC: `0.599447` over 7 completed folds; no-train `0.550339`, non-Aves `0.583959`, file-MIL `0.540916`.
- Baseline delta: vs original soft1279-head-loaded native all-class row `-0.000913`; file-MIL `-0.064889`.
- Package audit: valid TorchScript soundscape inference, 240/240 proxy rows matched, finite/nonconstant 234 columns. Best rank blend `w0.08` local AUC `0.992675` / 42 valid; lift vs v616 `-0.000805`.
- Decision: reject/no submission. Focal objective did not preserve the useful soft1279 sidecar signal and materially hurts file-MIL.

## Comparable top-5 context

1. Soft1279 head-loaded global w0.16 package/audit: local AUC `0.995545`, lift vs v616 `+0.002064`, but site concentration failed promotion.
2. Original soft1279-head-loaded native all-class LOSO: row `0.600360`, file-MIL `0.605805`; package w0.16 local `0.995545`.
3. New stable-winner caps: site-CV `0.993629`, lift `+0.000148`; file-CV `0.994132`, lift `+0.000651`.
4. New focal2 native all-class: row `0.599447`, file-MIL `0.540916`; package best `0.992675`, lift vs v616 `-0.000805`.
5. Balanced farneg5 no-call suppression: local `0.993510`, lift vs v616 `+0.000029`; rejected.

## Ranked next actions

1. **Hand/teacher-audited multi-site no-call negatives** — Expected LB potential: medium; information value: high. Threshold-only weak negatives are exhausted; next protocol must add actual negative evidence across more than S09/S18/S22.
2. **Soft1279 class/site robust recipe with explicit site-risk controls** — Expected LB potential: low-medium; information value: medium-high. Current selector finds four plausible classes but not enough site robustness; only continue if adding site-risk exclusion/penalty from movement diagnostics, not another unconstrained grid.
3. **New encoder/objective train_soundscape data point** — Expected LB potential: low-medium; information value: medium. Focal-B0 is negative; avoid more B0 objective knobs unless they test a new transfer hypothesis. If training, use genuinely different encoder/adapter or label protocol.
4. **Late-day guarded source fill** — Expected LB potential: low; information value: low-medium. Activate under `<3h` to reset if no verifier-grade candidate exists and source/runtime/schema/dedup guards pass.
5. **G124/teacher-cache lane** — Expected LB potential: low; information value: low. Soft-anchor target shape trains, but v616 transfer failed; only revisit with a new transfer mechanism.

## Critic / verifier decision

- Critic: the run tested the two most plausible post-16:25 lanes. Stable caps are safer but too weak; focal-B0 tests a real objective change and comes back negative. Do not submit either.
- Verifier: package audit matched 240/240 proxy rows with finite/nonconstant 234-class output, but best focal sidecar is below v616; selector has no submission artifact and fails site robustness. `submit_approved=false`.
- Submission decision: **no submission this run** under mid-day slot policy.
