# Validation Skeptic / Promotion Gate — New Branch Ideation

Date: 2026-05-25  
Role: Validation Skeptic / Promotion Gate  
Posture: strict, veto-capable. Local AUC is a rejection screen, not a Kaggle-submission approval signal.

## Executive verdict

The next useful branch must change hidden-test behavior, not just produce another locally positive perturbation of the `0.949` plateau. The evidence base is now strong enough to be pessimistic:

- `v611`, `v612`, and `v616` were hidden-safe, repo-owned sidecar blends; all passed local/schema/runtime gates and all scored `0.949`.
- `v616` had local lift `+0.0030902`, site bootstrap q05 about `+0.00176`, and leave-one-site all positive, yet still tied.
- The current prediction-overlap proxy is only about `190` matched rows / `20` files / `6` sites / `42` valid classes.
- Clean/train-audio CV can be badly misaligned: `v610` had high clean-audio fold AUC but scored `0.852`.
- Per-class adaptation already showed classic overfit: all-row lift near `+0.0029`, leave-site CV lift only around `+0.000002–0.000004`.

Therefore, I would approve only two kinds of next work:

1. **No-slot private verifier / offline smoke** for branches with genuine new signal and hidden-safe implementation potential.
2. **Kaggle submission** only after the candidate beats both the anchor and the already-tied `v616` baseline under stronger group/slice/negative-control evidence or has independent external evidence stronger than the current local proxy.

No current v616-family scalar, per-class, SYD, ProtoSSM, SED/Jung21/HGNet recombination is slot-worthy.

## 1. Likely false friends / high-overfit-risk archetypes

### 1. v616/SYD/ProtoSSM/SED/Jung21 scalar variants — **veto**

Examples: `v616 + SYD52p`, `sed_only_capped`, `sed_jung_tighter`, more `0.90/0.02/0.04/0.04` style mixes, rank-power/temperature variants, tiny Sakur/visual restoration.

Why false friend:

- They optimize the same narrow local overlap that already failed to transfer.
- Phase 2 showed the best `sakur_restored` recipe improves only `+0.0000556` local AUC over `v616`, which just tied public LB.
- SYD/P949 clone branches are mostly duplicate or near-duplicate matrices of known Jung21/SED/ProtoSSM lineages.

Decision: offline sensitivity rows only; no private verifier unless a truly new hidden-safe branch is added; no Kaggle slot.

### 2. Per-class adaptive selectors / taxon scalars from the current proxy — **veto unless independently OOF-backed**

Why false friend:

- The current selector’s in-sample lift did not survive leave-site CV.
- Only `42` valid proxy classes are available, while the submission has `234` classes.
- Many no-train sonotypes and rare taxa are saturated or class-sparse in the proxy; learned class weights will chase noise.

Decision: rejection-only diagnostic. Do not package as a kernel/submission unless both leave-site and leave-file CV show material lift and no sparse-class leakage.

### 3. Public-output-only branch blending — **private verifier blocked**

Why false friend:

- Static `240x235` public/dry-run branch CSVs can be useful for analysis, but they do not prove the branch can rerun on hidden `test_soundscapes`.
- Malformed public finals (`243x235`, constant/sample fallbacks, nonfinite cells) have repeatedly appeared in scout outputs.

Decision: analysis-only until rewritten as hidden-safe inference that derives row IDs from `sample_submission.csv` / test audio and writes raw branch outputs.

### 4. Clean-audio CV / train-audio-only checkpoint branches — **high risk**

Why false friend:

- `train.csv` is mostly isolated recordings and Aves-heavy; hidden rows are multilabel soundscapes.
- `v610` is the hard negative example: strong clean-audio fold evidence still scored `0.852`.
- The supervised SED export smoke packaged correctly but had weak holdout macro AUC (`0.754` over 79 valid classes).

Decision: useful for model development only. Not approval evidence without soundscape-style OOF/teacher validation and anchored hidden-safe branch audit.

### 5. Direct Alexy NS1 replay — **currently veto**

Why false friend:

- Direct exploratory `v613` scored `0.923`, far below plateau.
- Current source/API access was blocked, so hidden-safe extraction is not yet reproducible.
- Shape/source risks are already known.

Decision: can become a private-verifier candidate only if source/assets are recovered and the raw CNN/noisy-student branch is extracted as a low-weight sidecar rather than replaying the bad final.

### 6. G124/V2S pseudo-student optimism from smoke metrics — **hold, not submit**

Why false friend:

- The V2S-init smoke/pilot improved over scratch and exported, but anchored blend audits found only microscopic or negative sidecar utility.
- The all-row pilot had best useful weight around `0.0025`, with site-bootstrap q05 negative and some held-out sites negative.

Decision: credible research lane, not a slot candidate until it produces independent sidecar lift at usable weight and stable group behavior.

### 7. Sparse hard-negative/no-call auxiliary losses — **hold**

Why false friend:

- The negative-mask auxiliary smoke covered only `26/512` rows (`5.08%`) and added noise-sized lift over soft-only (`0.819410` vs `0.819021`).
- Current labeled soundscape rows are all positive; unlabeled artifact rows are not proven no-call negatives.

Decision: promising only after broader negative/no-call coverage and rare-call protection are demonstrated.

### 8. WildSound/ConvNeXt/foundation-model repairs without export/runtime proof — **hold**

Why false friend:

- Structurally different does not matter if the branch cannot run hidden-safely within Kaggle runtime or depends on missing/private assets.
- Several public notebooks already fail, fallback, or train in-kernel without clear output guarantees.

Decision: offline repair/export smoke first; no private verifier until assets, runtime, and raw output path are proven.

## 2. Minimum evidence by promising archetype

The table ranks evidence needed before **private verifier** and before **Kaggle submission**. “Private verifier” means no-slot hidden-safe kernel or local/trainer artifact gate; it does **not** imply submission approval.

| Archetype | Before private verifier | Before Kaggle submission |
|---|---|---|
| **A. Non-Aves + no-train soundscape specialist** | Predeclare target labels/slices: the 28 no-train-primary classes (`517063`, `1491113`, `25073`, `47158son01`–`47158son25`) plus rare/non-Aves classes. Train on soundscape-style windows, not just train-audio clips. Use fixed taxon/class caps, not learned per-class weights. Require leave-site and leave-file lift on target slices and no broad Aves/common-class damage. | Private verifier COMPLETE with hidden-safe raw branch; aggregate lift vs v616 `>=+0.001` or strong independent slice evidence; target no-train/non-Aves lift transfers under site/file; taxon movement reviewed; sparse classes with `<3` positives or `<20` negatives remain anchor-only/fixed-capped; negative controls fail. |
| **B. Soundscape-native OOF/teacher SED refresh** | Exports TorchScript/ONNX; inference smoke emits full 234 columns; OOF/holdout uses soundscape-style windows; `>=60` valid classes preferred; preliminary local/blend lift `>=+0.001` vs anchor or clear new low-corr raw signal; no degenerate/no-call collapse. | Private verifier COMPLETE; raw branch reruns on hidden rows; lift vs anchor `>=+0.006` or >=2× v616 lift; lift vs `v616` `>=+0.001`; site q05 `>=+0.003`; file q05 `>=+0.0015`; leave-one-site min `>=+0.003`; leave-one-file q05 `>=+0.001`; taxon/no-train slices non-degrading; negative controls fail to improve. |
| **C. Broader hard-negative/no-call residual gate** | Negative/background mask covers a meaningful share of rows/classes, not `5%`; reports predicted mass on labeled-positive, sparse-positive, and verified/background-like rows; rare-call recall protected; branch displacement bounded when blended. | Demonstrates hidden-safe residual branch or calibration gate; improves no-call/background proxies without degrading rare/taxon classes; group-stable lift vs both anchor and v616; no evidence that it merely suppresses true positives; passes shuffle/inversion controls. |
| **D. Long-context / temporal-localization branch** | Rerunnable 30–60s/segment-aware branch; validates time bins (`<=20s`, `25–40s`, `>=45s`), file groups, endpoint logic, and persistent-chorus vs intermittent-call behavior; runtime margin on hidden rows. | Beats v616 on file/time-position slices, not just aggregate AUC; leave-one-file q05 clears threshold; no hard-coded row/window assumptions; bounded final displacement. |
| **E. External/foundation audio embedding branch (BEATs/AudioMAE/HTS-AT/PaSST/PANNs/YAMNet/CLAP/Perch/BirdNET-style)** | Frozen or lightweight model can run/export within Kaggle constraints; public/attachable weights; branch outputs 234 classes with no fallback; local raw predictions have low/moderate corr and nontrivial no-train/non-Aves/site-shift signal. | Strong independent source evidence or group-stable lift vs v616; runtime safely below kernel limit; no reliance on private embeddings; negative controls show signal is not validation leakage. |
| **F. Site/domain-adversarial or group-DRO branch/constraint** | Uses site-balanced/group-DRO/adversarial training as a predeclared constraint on a real branch, not as post-hoc tuning. Reports worst-site, missing-site (`S15/S19/S23`) where predictions can be materialized, and leave-site behavior. | Worst-site lift improves without average/taxon collapse; no hidden-unsafe site priors; group metrics clear submission thresholds vs v616. Best used as a constraint on A/B/E, not a standalone submission. |
| **G. G124 / EffV2-S reconstruction with strong init** | Full or sufficiently broad pilot, not a 384-row-only smoke; export/runtime path works; useful aligned sidecar weight `>=0.01` in OOF/blend audit; group q05 nonnegative; raw branch not just teacher clone. | Independent evidence that the reconstructed family differs from v616 and transfers: sidecar lift vs v616 `>=+0.001`, stable site/file/taxon behavior, no negative held-out sites/files, hidden-safe generation from public/attachable assets. |
| **H. Alexy NS1 CNN/noisy-student extraction** | Source/assets recovered; hidden-safe raw sidecar generated; does not replay direct `0.923` final; low-weight rank blend displacement bounded; no sample-shape/fallback behavior. | Must explain why direct `0.923` does not apply to the extracted branch; branch ablation shows useful contribution beyond v616; clears group gates vs v616, not only vs anchor. |
| **I. WildSound/ConvNeXt repair** | Public failure repaired offline; checkpoints/assets exist; no in-kernel training timeout; export/runtime smoke passes; raw branch valid and row-aligned. | Needs strong independent lift or slice-specific value after repair; otherwise structurally different but untrusted. |
| **J. Co-occurrence/context prior branch** | Only after a stronger raw branch exists. Must be predeclared and validated as a negative-control-prone prior, not a scalar tweak. | Requires branch ablation showing lift beyond raw model and no site/co-occurrence leakage; otherwise analysis-only. |

## 3. Testing hidden-behavior difference beyond local AUC

Every serious branch should be tested against **both** the Samejima/v616 anchor and the **nearest tied/dropped same-lineage recipe**. For current work, the nearest prior is usually `v616`, not the anchor.

### A. Correlation and displacement

Required metrics:

- Raw branch rank corr vs anchor and vs v616 branches.
- Final rank corr vs anchor and vs v616 final.
- Rank/probability MAE and max-abs displacement vs anchor and v616.
- Top-k row recall at `k=1,3,5,10`.
- Per-class lift distribution and worst moved classes.

Interpretation rules:

- Final corr `>0.99975` plus incremental lift vs v616 `<+0.00025` => near-duplicate veto.
- Final corr `<0.995` or MAE `>0.025` => require independent proof the sidecar is not a hidden-LB dropper.
- Raw low correlation is helpful only when paired with stable group lift and credible source lineage.

### B. Group validation

Submission-grade, not smoke-grade, validation should use:

- Site bootstrap, preferably `>=5000` iterations.
- File bootstrap, preferably `>=5000` iterations.
- Leave-one-site and leave-one-file.
- If source families are mixed: branch-family ablations.
- If taxonomy metadata is available: taxon/family summaries.

Minimum local thresholds for a submission from the current proxy:

- Lift vs anchor `>=+0.0060` macro AUC, or at least 2× v616 lift if using the same proxy.
- Lift vs v616 / nearest tied recipe `>=+0.0010`.
- Site bootstrap q05 `>=+0.0030`.
- File bootstrap q05 `>=+0.0015`.
- Leave-one-site all positive with min `>=+0.0030`.
- Leave-one-file at least 90% positive and q05 `>=+0.0010`.
- Valid classes `>=60` preferred; `<42` is verifier-only.
- If site count `<8`, site validation is explicitly not approval-grade.

### C. Slice and taxon checks

Report these for every candidate:

- Site slices: especially `S03`, `S08`, `S09`, `S13`, `S18`, `S22`, and any newly matched sites.
- File-level held-out behavior, not just site aggregates.
- Time-position bins: early `<=20s`, mid `25–40s`, late `>=45s`.
- Taxon slices: Aves, Amphibia, Insecta, Mammalia, Reptilia.
- Train-count bins: no-train primary, `1–5`, `6–20`, `21–100`, `>100`.
- Soundscape-positive rarity bins: `1–2`, `3–10`, `11–50`, `>50` rows.
- The 28 submission classes with no train-audio primary labels.

Known slice warnings:

- Jung21 is conditional and can hurt Aves/common classes.
- SED is locally useful for weak/rare soundscape slices but already failed to lift public LB when blended in v616.
- Insect/no-train sonotypes are locally near-saturated; aggressive movement is likely overfit.

### D. Negative controls

At least two controls are mandatory before any promotion:

- Anchor-only reconstruction must exactly match anchor metrics.
- Shuffled/permuted sidecar rows should not improve.
- Inverted/anti-rank sidecar should not improve.
- Class-shuffled sidecar should not improve.
- Branch ablation must show the new branch contributes lift beyond v616.
- Known bad/drop controls (`v610` clean-audio style, direct Alexy `0.923`, simple HGNet sidecar ties) should remain rejected by the same protocol.

If a negative control improves similarly to the candidate, the validation proxy is contaminated or too weak; no submission.

## 4. Strict ranking by validation credibility

This ranking is by **validation credibility**, not speculative upside.

1. **Non-Aves + no-train soundscape specialist** — strongest validation target because it directly addresses a real data gap: 28 submission classes have zero train-audio primary examples, while soundscape positives are Amphibia/Insecta-heavy. Credible if predeclared, capped, and evaluated by leave-site/file against v616.
2. **Soundscape-native OOF/teacher SED refresh with exportable hidden-safe inference** — best general-purpose validation fit because it can be evaluated on broader soundscape-style rows. Current supervised smoke is weak; current soft-teacher smoke is only a starting point.
3. **Broader hard-negative/no-call residual gate** — most plausible way to change hidden behavior that positive-only local AUC barely tests, but current negative coverage is far too sparse. Build the negative audit before modeling.
4. **Long-context / temporal-localization branch** — credible if it genuinely models file/segment context and passes leave-file/time-bin checks. Good hidden-diversity hypothesis; currently more idea than evidence.
5. **External/foundation audio embedding branch** — credible if weights/runtime are public and exportable and if it is focused on no-train/non-Aves/site-shift slices. Avoid model-zoo fishing.
6. **Site/domain-adversarial or group-DRO constraint** — useful as a training constraint for the above branches; not persuasive as a standalone post-hoc tweak.
7. **G124/EffV2-S reconstruction with strong external/V2S init** — credible research lane because it targets a missing stronger lineage and has working export smoke, but current blend utility is microscopic/unstable. Needs a better full-pilot or new target design.
8. **Alexy NS1 CNN/noisy-student extraction** — structurally different, but direct `0.923` result and source access issues make it low credibility until raw sidecar extraction is proven.
9. **WildSound/ConvNeXt repair** — structurally interesting but blocked by public errors/runtime/asset uncertainty. Offline repair only.
10. **Co-occurrence/context priors, HGNet/S14/ProtoSSM additional sidecars** — diagnostic only unless paired with a genuinely new raw branch.
11. **v616/SYD/scalar/per-class/rank-power variants** — reject. These consume slots without a credible hidden-behavior change.

## 5. Explicit veto rules

Reject without private verifier if any condition holds:

1. Same branches as a tied/dropped submission, only weights/powers/temperatures/thresholds/scalars changed.
2. Incremental local lift vs v616 or nearest tied recipe `<+0.0005`.
3. Final corr vs v616 `>=0.9995` and MAE `<=0.005`.
4. Metrics are reported only vs anchor, not vs v616 / nearest tied recipe.
5. Candidate evidence is only train-audio CV, clean-recording fold AUC, or a single random split.
6. Candidate uses static public/dry-run CSVs as hidden-test branch inputs.
7. Output relies on hard-coded row IDs, fixed sample fallback, malformed public final, nonfinite cells, constant columns, or private/missing assets.
8. Valid local AUC classes `<42`, or matched rows `<190`, unless the candidate is strictly private-verifier/research-only.
9. Site count `<8` and no independent evidence beyond the current proxy.
10. Any leave-one-site group is negative for a submission candidate.
11. File bootstrap q05 is negative or leave-one-file shows broad failures.
12. Per-class adaptive weights improve all-row/in-sample but leave-site or leave-file CV lift is `<+0.001`.
13. Sparse classes (`<3` positives or `<20` negatives) receive learned boosts without independent OOF evidence.
14. Taxon movement has obvious rare-class degradation or mean movement >2× global median without manual signoff.
15. A same-lineage candidate is pending or just tied/dropped and the new candidate cannot explain why the repeat will differ.
16. Negative controls improve similarly to the candidate.
17. Runtime/export is unproven or near kernel limit without margin.
18. Submission would consume a Kaggle slot before Coordinator + independent Verifier signoff.

## 6. Recommended promotion gate language for Coordinator

For the ideation synthesis, I recommend labeling candidates as:

- **Promote to no-slot smoke/private verifier:** archetypes 1–4 only if their first experiment is evidence-building, not submission-oriented.
- **Hold as repair/research:** foundation/long-context/Alexy/WildSound until hidden-safe raw branch generation is proven.
- **Reject as slot candidates:** all v616-family scalar/per-class/SYD/ProtoSSM/HGNet repeats.

The next branch roadmap should optimize for a new validation source or new hidden-safe raw signal. If the only evidence is another `+0.003` local lift on the same 190-row proxy, I would veto submission even if the private verifier output is clean.
