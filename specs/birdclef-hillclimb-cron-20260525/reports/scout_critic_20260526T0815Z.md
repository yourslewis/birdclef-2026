# Scout + Critic Report — BirdCLEF hill-climb cron — 2026-05-26 08:15 UTC

## Scope / evidence read
- Spec: `specs/birdclef-hillclimb-cron-20260525/spec.md`.
- Recent queue/model reports: `ranked_queue_20260526T0022Z.md`, `ranked_queue_20260526T0220Z.md`, `ranked_queue_20260526T0419Z.md`, `ranked_queue_20260526T0634Z.md`, plus model data points through `model_data_point_20260526T0659Z_efficientat_mn10_soundscape.md`.
- Ledger tail: `artifacts/model_data_point_ledger/20260525T2307Z_*` through `20260526T0659Z_*`.
- Autoresearch log tail through the EfficientAT MN10 data point.
- Public scout artifacts: `artifacts/public_kernels_20260526_scout/scan_20260526T0020Z.json`; quick web search at report time found no fresh EfficientAT/PANNs/BirdCLEF 0.95+ public notebook lead beyond already-known EoS/ONNX/Perch/Proto/SED plateau results.

Evidence level: **comparison-grade landscape synthesis**. No new verifier package or live submission was created by this role.

## Verified latest candidate landscape
- Public LB best remains **0.949** in the current records. `v616` is the strongest repo-owned tied baseline; late-day public replays/data-family probes ended `v617=0.949`, `v618=0.946`, `v619=0.944`, `v620=0.949`.
- 2026-05-26 slots were **0/5 used** in the latest recorded run; no post-0659Z submit-ready candidate is documented.
- Public/source landscape remains saturated around EoS / Perch / ProtoSSM / SED / Jungchan / Nina / Karnak / SYD variants. Fresh 20260526 scout did not surface a clean direct public lead above 0.949; known public outputs include malformed finals, exact/near branch duplicates, public-output-only artifacts, or already-submitted tied/dropped families.
- New-model landscape since reset:
  - **B0 OOF-teacher 1024 ep4 soft-only** is the strongest recent all-234 repo model data point: macro AUC `0.911067` over 122 valid classes; export/runtime passed. It failed to help in the later v616-sidecar audit when represented as a soft-B0 sidecar, so it is not a slot candidate unchanged.
  - **G124/V2S localmax/power085** is the strongest new proxy model: val AUC `0.960094` over 62 valid classes, but all-row student AUC `0.944720` remains far below teacher `0.995541`, and v616 sidecar lift was only `+0.00000339` local AUC. Interesting, not submit-grade.
  - **AudioSet embedding soundscape heads** are diverse but weak on S08: PANNs/Cnn14 `0.517333`, EfficientAT MN10 `0.488240`, B0 soundscape specialist `0.488650`. They are high-info for non-Aves/no-train/no-call slices, but currently not competition-format and not strong enough to wrap blindly.
  - **20s temporal/localmax B0** is decorrelated but too weak: macro AUC `0.672996`, corr `0.599986` vs 5s soft-only. Do not package unchanged.
  - **Broad negative/no-call aux** solved coverage but hurt the matched control: `0.908278` vs soft-only `0.911067`; do not scale unchanged.

## Top 5 ranked next actions by expected LB + information value

1. **Run EfficientAT `dymn10_as` AudioSet embedding data point on the same soundscape/non-Aves/no-train/no-call contract.**
   - Expected LB potential: low/medium directly, because MN10 and PANNs are weak; medium as rare-slice sidecar if it improves materially.
   - Info value: **high**. It cleanly tests whether the weaker MN10 result was architecture/checkpoint-specific. A config already exists: `configs/birdclef/efficientat_dymn10_audioset_soundscape_nonaves_notrain_nocall_siteS08_ep12_20260526.json`; no artifact exists yet.
   - Gate: must beat PANNs `0.517333` on S08 or show better leave-site/no-call behavior before any wrapper.

2. **G124/V2S hard-confidence / target-power ablation, not another unchanged rerun.**
   - Expected LB potential: medium if it produces a clearer independent branch than the tiny localmax weights.
   - Info value: **high**. Current G124 localmax has the best proxy AUC but negligible v616 blend lift; ablation can separate target-shape effect from `teacher_power=0.85` and identify whether any G124 signal is real.
   - Gate: require sidecar audit lift materially above noise, not `+3e-6`; include stability/leave-site before private verifier.

3. **Build a leave-one-site / site-balanced evaluation harness for AudioSet soundscape heads before more single-site conclusions.**
   - Expected LB potential: medium through better slice selection/calibration, not direct submission.
   - Info value: **high**. S08 makes no-call AUC invalid and may punish/boost sonotypes arbitrarily; this is blocking the AudioSet branch from being interpreted.
   - Gate: PANNs/EfficientAT/B0 soundscape heads should be compared across multiple sites with no-call-positive validation folds.

4. **Mine fresh public model assets/datasets rather than public code replays: BirdNET embeddings, Bioacoustics Model Zoo/Perch2/SurfPerch, PaSST/HTS-AT/BEATs/ATST.**
   - Expected LB potential: medium/high only if packageable as a true hidden-safe sidecar; direct Kaggle replay potential is currently low.
   - Info value: medium/high. The public code frontier is saturated; pretrained model-source diversity is more likely to shift hidden behavior.
   - Gate: license/rules check + reproducible embedding extraction + 234-class wrapper/audit. No static public output CSVs.

5. **Prepare late-day fallback queue now, but mark it lower priority until <3h reset.**
   - Expected LB potential: low; information value medium if slots would otherwise expire.
   - Use only source-clean, non-duplicate, non-malformed candidates. Avoid v616/v617/v620 replays, SYD52p micro-increments, per-class/scalar v616 tweaks, and P949/EoS/ProtoSSM/SED family duplicates unless they are demonstrably different and pass guards.
   - This is slot-policy hygiene, not an early-day action.

## Specific critique of the likely next train/submit choice

**If the next choice is EfficientAT `dymn10_as`: PROCEED, but only as a bounded data point.**
- Why proceed: MN10 underperformed PANNs, but DyMN10 is the most specific untested EfficientAT ablation and uses an already-prepared config. It is cheap, externally pretrained, and high-information for whether EfficientAT should remain in the queue.
- Do not oversell it: the current AudioSet branch metrics are far below anything directly submit-worthy, and the validation split lacks no-call AUC. A DyMN10 score near MN10/PANNs should terminate the EfficientAT-on-S08 path unchanged.
- Required success criteria: beat PANNs on the same contract (`>0.5173` S08 macro AUC), or show a clear per-site/no-call advantage in a follow-up multi-site harness. Otherwise, log as negative and move to G124 ablation or a different pretrained family.

**If the next choice is a Kaggle submission: REJECT for now.**
- No current candidate is verifier/submission-grade. The best G124/v616 audit lift is effectively zero; B0 sidecar did not help; AudioSet heads are not 234-class competition outputs; 20s branch is weak; public candidates are duplicated/malformed/plateau.
- Early-day submission from this state would be leaderboard probing, not a high-information authorized slot use. Revisit only if a private verifier package passes schema/runtime/alignment and has a nontrivial audit signal, or when late-day slot-fill policy applies.

## Rule / duplication / integrity risks
- **Duplicate matrix/description risk:** `v616`, `v617`, and `v620` tied at 0.949; exact replays or same-family cosmetic relabels violate the spec's duplicate guard.
- **Near-duplicate plateau risk:** SYD52p/P949/Kijiang/Jungchan/Nina/EoS/ProtoSSM/SED branch variants are mostly the same public branch soup. Tiny local AUC increments over v616 are not evidence after v616 itself tied hidden LB.
- **Static public-output-only risk:** branch CSVs from public notebook dry-runs must not be blended into finals unless the source reruns hidden `test_soundscapes` and writes aligned raw outputs.
- **Validation overfit risk:** v616 local sidecar gate and per-class selector already showed strong local/in-sample behavior with no LB lift. Treat local train-soundscape AUC as rejection/comparison evidence only.
- **No-call validation gap:** PANNs/EfficientAT S08 split lacks both no-call classes, so no-call auxiliary claims are currently unsupported.
- **Rule-safety:** recent no-slot training uses official train data plus public/existing pretrained weights/caches. Continue to verify licenses/assets before packaging external model zoo branches.

## Bottom line
Proceed with **EfficientAT DyMN10 as the next bounded no-slot data point**, then force a decision: if it does not beat PANNs or unlock a multi-site/no-call advantage, stop AudioSet soundscape repetition and move to **G124 hard-confidence/power ablation** or a genuinely new pretrained model family. No Kaggle submission is justified at 08:15Z.
