# Phase 1C — Data, Feature, and Slice Report

_Date: 2026-05-25_  
_Role: Data & Feature Scientist_

## Executive takeaways

The available validation signal is useful for **rejecting bad ensembles**, but it is too narrow to approve another leaderboard submission by itself. The strongest current proxy has only **190 matched artifact rows / 20 files / 6 sites / 42 valid AUC classes**, while the competition predicts **234 classes** and hidden rows include unseen site/time structure. This explains why `v611`, `v612`, and `v616` could look locally strong and still tie `0.949`.

Data-driven ensemble direction:

1. Keep one dominant Samejima/visual-style plateau anchor.
2. Allow only **small capped sidecars** from genuinely different raw branches (`SED`, `Jung21`, maybe one HGNet comparator) and judge them with site/file/taxon/rarity gates.
3. Do **not** trust per-class adaptive weights unless they transfer under leave-site and leave-file CV; current `v616` per-class selector failed that test.
4. Do not use train-audio-only CV or clean-audio fold AUC as approval evidence. The useful proxy is train-soundscape/file/site robustness plus hidden-safe rerun evidence.
5. The highest missing analysis before any final submission is a broader group bootstrap/leave-file/taxon report on the exact Phase 2 candidate versus both the anchor and nearest tied recipe.

Supporting one-off numeric analysis was saved here:

- `specs/birdclef-ensemble-strategy-20260525/reports/data_feature_slices_analysis.json`

## Sources inspected

Requested sources plus local artifacts:

- `specs/birdclef-ensemble-strategy-20260525/spec.md`
- `docs/BIRDCLEF_TWO_DAY_EXPERIMENT_SPEC_20260524.md`
- `docs/BIRDCLEF_AUTORESEARCH_LOG.md` and `docs/BIRDCLEF_096_FRONTIER_PLAN_20260518.md` recent relevant sections
- External local data: `/Volumes/ExternalSSD/data/workspace_don/kaggle_birdclef2026/data/{train.csv,train_soundscapes_labels.csv,taxonomy.csv,sample_submission.csv}`
- `artifacts/anchored_blend_audit/sidecar_grid_inputs_20260525T0200Z/*.csv`
- `artifacts/anchored_blend_audit/sidecar_grid_20260525T0200Z_fast.json`
- `artifacts/anchored_blend_audit/sidecar_grid_20260525T0200Z_top_stability.json`
- `artifacts/anchored_blend_audit/v616_per_class_selector_20260525T0810Z.json`
- `artifacts/anchored_blend_audit/v616_syd52p_*_20260525T1000Z.json`
- `artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/*`
- `artifacts/sed_oof/sed-b0-q3cap80-ep12init-exportsmoke-5s-160-allcls-20260525/metrics.json`
- `artifacts/sed_oof_teacher_students/sed-b0-oofteacher-b0v26-nfnetv29-soft-512-ep3-20260525/metrics.json`
- `artifacts/sed_oof_teacher_students/sed-b0-oofteacher-b0v26-nfnetv29-soft-negaux002-512-ep3-20260525/metrics.json`
- `artifacts/pseudolabels/threshold_sweeps/*.json`

Note: the canonical repo currently has no `data/` directory. The usable local data lives on `/Volumes/ExternalSSD/...`; several repo scripts/configs refer to `data/...` as the trainer/Kaggle-side relative path.

## 1. Available labeled validation/proxy data and group fields

### Core data files

| File | Shape | Useful fields | Group/slice fields present |
|---|---:|---|---|
| `/Volumes/ExternalSSD/data/workspace_don/kaggle_birdclef2026/data/train.csv` | `35549 x 15` | `primary_label`, `secondary_labels`, `type`, `latitude`, `longitude`, `scientific_name`, `common_name`, `class_name`, `inat_taxon_id`, `rating`, `filename`, `collection` | species/taxon, collection, rating, geography; **no site/time** except whatever can be inferred indirectly from filename/source |
| `/Volumes/ExternalSSD/data/workspace_don/kaggle_birdclef2026/data/train_soundscapes_labels.csv` | raw `1478 x 4`; collapsed to `739` unique 5s row IDs after unioning duplicate rows | `filename`, `start`, `end`, semicolon labels in `primary_label` | file, site parsed from filename, timestamp parsed from filename, segment end time, multilabel density, species |
| `/Volumes/ExternalSSD/data/workspace_don/kaggle_birdclef2026/data/taxonomy.csv` | `234 x 5` | `primary_label`, `inat_taxon_id`, names, `class_name` | taxon/class group for all submission labels |
| `/Volumes/ExternalSSD/data/workspace_don/kaggle_birdclef2026/data/sample_submission.csv` | `3 x 235` | `row_id` + 234 class columns | hidden/test row format; sample rows use site `S05` |

### Train-audio metadata characteristics

- `train.csv` has **35,549** recordings across **206** primary labels, while submission/taxonomy have **234** classes.
- Collection imbalance: `XC=23,043`, `iNat=12,506`.
- Taxon imbalance in train audio is extreme:
  - `Aves=34,799`
  - `Amphibia=451`
  - `Insecta=199`
  - `Mammalia=99`
  - `Reptilia=1`
- Primary-label count distribution is broad: min `1`, median `125`, 90th percentile `477`, max `499`.
- Rare classes in train audio: `4` labels have exactly one recording, `18` have `<=5`, `36` have `<=20`, while `124` have `>=100`.
- `secondary_labels` is nonempty in `4,372` rows, so single-primary training can underuse co-occurrence labels.
- Rating distribution has many low-quality/unknown rows: median `3.5`, 25th percentile `0.0`, mean `2.60`.

### Classes with no train-audio primary examples

There are **28 submission classes absent from `train.csv` primary labels**, and all 28 appear in soundscape labels:

- Amphibia: `1491113`, `25073`, `517063`
- Insecta/sonotypes: `47158son01` through `47158son25`

This is a major ensemble constraint: any member trained only from `train.csv` primary labels can be blind or under-calibrated for these classes unless it uses taxonomy/soundscape/pseudo-label side information.

### Labeled train-soundscape proxy

After collapsing duplicate label rows by `(filename,end)`:

- **739** unique labeled 5s rows
- **66** files
- **9** sites: `S03`, `S08`, `S09`, `S13`, `S15`, `S18`, `S19`, `S22`, `S23`
- **75** unique positive labels
- **3,122** positive row-label cells
- Multilabel density is high: rows have `1` to `10` labels; counts by row label count are:
  - `1:78`, `2:49`, `3:100`, `4:182`, `5:160`, `6:95`, `7:57`, `8:14`, `9:3`, `10:1`
- Top soundscape labels by row count include `65380` (`333`), `517063` (`313`), `22973` (`213`), `555146` (`210`), `23158` (`175`), `24279` (`173`), `24321` (`172`), `22967` (`155`), `66971` (`149`).
- Several labels are extremely sparse in soundscape rows: e.g. `rutjac1`, `plcjay1`, `wfwduc1`, `sibtan2`, `ruther1` each appear once.

### Current artifact validation overlap

Using `artifacts/anchored_blend_audit/sidecar_grid_inputs_20260525T0200Z/samejima_visual_anchor.csv` as the anchor:

- Anchor artifact rows: **240**
- Rows matched to labeled soundscape truth: **190**
- Unmatched rows: **50**
- Matched files: **20**
- Matched sites: `S03=24`, `S08=60`, `S09=19`, `S13=24`, `S18=15`, `S22=48`
- Valid AUC classes for the common proxy: **42**

This small overlap is the most important limitation in every numeric claim below.

## 2. Slice risks and opportunity areas for ensemble weighting

### A. Species/taxon imbalance risk

Risk:

- Macro AUC weights classes evenly, but train data is dominated by `Aves` and frequent labels.
- `28/234` submission classes have no train-audio primary examples, including all `47158son*` sonotypes.
- The artifact overlap has only **42 valid AUC classes**, so any per-class tuning is underdetermined.

Opportunity:

- Use taxonomy-aware caps instead of free per-class weights.
- If a sidecar is strong for rare or no-train classes, add it under a **small class/taxon cap**, not as an unconstrained selector.
- Require minimum positives/negatives before changing class weights. Current selector evidence supports this: `v616` per-class selector had in-sample lift but only `+0.0000035` leave-site CV lift.

### B. Site/file/domain shift risk

Risk:

- The labeled proxy sites are `S03/S08/S09/S13/S15/S18/S19/S22/S23`; current artifact overlap uses only six of them.
- The sample hidden/test row format includes `S05`, absent from current labeled artifact overlap.
- `v616` had excellent site-stable local evidence and still tied hidden LB, so site bootstrap on six sites is not approval-grade.

Opportunity:

- Site/file bootstrap is still useful as a **rejection screen**.
- Sidecars that lift weak sites without hurting strong sites are good candidates for private verifiers.
- The current `v616` blend improved local AUC on `S09`, `S13`, and `S22`, but not `S03` or `S18`; this argues for either lower/capped global sidecar weights or site-robust validation rather than site-specific gates.

### C. Train-audio vs train-soundscape mismatch

Risk:

- `train.csv` is mostly isolated recordings with one primary label, collection/rating biases, and no site/time groups.
- `train_soundscapes_labels.csv` is multilabel, background-heavy, and site/time structured.
- Historical evidence confirms this mismatch: clean-audio/high-CV checkpoint inference (`v610`) scored `0.852`; standalone Samejima/HGNet (`v598`) scored `0.860` despite valid output.

Opportunity:

- Prefer members that either directly model soundscape rows or have been rerun through hidden-safe soundscape inference.
- Treat train-audio OOF and clean recording CV as model-development signals, not submission approval.
- OOF teacher/cache work should be evaluated through row-level soundscape-style windows and negative/background behavior.

### D. No-call/background behavior

Risk:

- The collapsed soundscape labels have no clean empty/no-call rows in the current proxy; every labeled row has at least one positive label.
- The 50 unmatched artifact rows are **unlabeled**, not proven no-call negatives.
- Rank-scaled public/visual outputs have means near `0.5`, so their raw values are not calibrated probabilities. They cannot be used directly for no-call calibration.

Opportunity:

- Negative/background regularization should come from clean OOF negative masks, not from assuming unlabeled artifact rows are no-call.
- Current negative-mask auxiliary smoke was too sparse: only `26/512` rows covered (`5.08%`), `1,664` negative cells, and AUC was flat (`0.819410` vs soft-only `0.819021`). Broader coverage is needed before this becomes ensemble-relevant.

### E. Candidate-family saturation risk

Risk:

- EoS/PCEN/visual/RankPower/Jungchan/Raunak/SYD families contain many near-duplicates.
- `samejima_sed`, `raunak_sed`, and `v616_sameji_sed_raw` are exactly identical in local artifacts (`corr=1.0`, `MAE=0`).
- `jungchan_model21` and `v616_jung21_raw` are exactly identical.
- SYD `subm_21` duplicates Jungchan Model21 and `subm_52p` is near-duplicate ProtoSSM lineage.

Opportunity:

- Use raw branches once, not many final notebooks.
- For Phase 2, compare every candidate to the **nearest tied same-lineage recipe**, not just the anchor.

## 3. Candidate family behavior by slice

### Overall local behavior on the 190-row / 42-class proxy

| Candidate / family | Local macro AUC | Top-3 recall in this analysis | Corr vs visual anchor | Notes |
|---|---:|---:|---:|---|
| Samejima/Raunak SED raw | `0.995976` | `0.989` | `0.259` | Strongest local proxy model and low probability correlation, but using it at low weight in `v616` still tied public LB. Treat as overfit-prone but useful branch. |
| `v616` final blend | `0.993481` | `0.484` probability-top3 here; rank-table top3 is `0.942` | `0.988` prob corr; rank corr `0.9998` in prediction report | Local AUC lift did not transfer to public LB. Use as control, not approval. |
| Samejima visual anchor | `0.990391` | `0.721` probability-top3; rank-table top3 `0.932` | `1.000` | Current stable plateau anchor. |
| Jungchan Model21 raw | `0.987426` | `0.516` probability-top3 | `0.403` | Diverse raw branch; helps some rare/taxon slices but hurts others. |
| Jungchan ProtoSSM | `0.986253` | `0.779` | `0.432` | Better top-k than Model21 in probability space, weaker macro AUC. |
| Sakur visual | `0.984775` | `0.358` | `0.958` | Visual-family near-anchor; not enough novelty. |
| Raunak ProtoSSM | `0.984640` | `0.721` | `0.470` | Similar family to other ProtoSSM branches; not a standalone lift. |
| Samejima ProtoSSM | `0.982155` | `0.705` | `0.451` | Lowest of the inspected core branches. |

Important nuance: top-k recall differs between this probability-space analysis and the prediction team's rank-space tables. Use rank-space metrics for rank-blend decisions, but the probability-space contrast is still useful for identifying calibration/no-call limitations.

### Site slices

| Site | Rows / files | Anchor AUC | `v616` final | Jung21 | SED raw | Main read |
|---|---:|---:|---:|---:|---:|---|
| `S03` | `24 / 2` | `0.9623` | `0.9623` | `0.9964` | `0.9794` | Jung21 looks strong, but `v616` weight did not move this site. |
| `S08` | `60 / 5` | `0.9842` | `0.9856` | `0.9893` | `0.9866` | Small positive sidecar movement. |
| `S09` | `19 / 5` | `0.9487` | `1.0000` | `0.9003` | `1.0000` | SED carries this site; Jung21 hurts. Very small valid-class count (`5`). |
| `S13` | `24 / 2` | `0.9503` | `0.9923` | `0.9844` | `0.9808` | Visual Sakur/Proto also high; site may be easy to overfit. |
| `S18` | `15 / 2` | `0.9050` | `0.9050` | `0.8308` | `0.9286` | Weak/small site; global blend did not help despite SED raw being better. |
| `S22` | `48 / 4` | `0.9467` | `0.9630` | `0.9267` | `0.9744` | SED helps; Jung21 hurts. |

Slice implication: SED is the most consistently useful branch on weak sites, while Jung21 is site-conditional. A global Jung21 cap should remain small unless a future candidate proves robustness outside this overlap.

### Time-position slices within 60s files

| Segment bin | Rows | Anchor | `v616` final | Jung21 | SED raw | Read |
|---|---:|---:|---:|---:|---:|---|
| early `<=20s` | `68` | `0.9913` | `0.9940` | `0.9940` | `0.9959` | Both sidecars help locally. |
| mid `25–40s` | `59` | `0.9945` | `0.9970` | `0.9925` | `0.9966` | SED helps; Jung21 below anchor. |
| late `>=45s` | `63` | `0.9919` | `0.9939` | `0.9935` | `0.9950` | Sidecars help locally; SED best. |

No obvious time-bin failure emerges, but the bins are too small for approvals. Time should be included in bootstrap/leave-file diagnostics to catch file-position leakage.

### Train-recording rarity slices

| Train-count bin among valid classes | Valid classes | Anchor | `v616` final | Jung21 | SED raw | Read |
|---|---:|---:|---:|---:|---:|---|
| no train primary (`n=0`) | `19` | `0.99683` | `0.99681` | `0.99672` | `0.99700` | Already near-saturated; do not over-tune no-train sonotypes from this proxy. |
| `1–5` | `2` | `0.98778` | `0.99153` | `0.99842` | `0.99639` | Potential sidecar value, but only two classes. |
| `6–20` | `5` | `0.97906` | `0.98804` | `0.98749` | `0.99491` | Best rare-class opportunity; SED especially useful. |
| `21–100` | `7` | `0.98977` | `0.99550` | `0.99703` | `0.99570` | Sidecars look useful. |
| `>100` | `9` | `0.98416` | `0.98834` | `0.95787` | `0.99454` | Jung21 is weak on common-train classes; SED is strong. |

Slice implication: if using class-aware constraints, allow SED more room for `6–20` and common train-count classes, but keep Jung21 capped on common classes unless independently validated.

### Soundscape-row rarity slices

| Soundscape positive-row bin | Valid classes | Anchor | `v616` final | Jung21 | SED raw | Read |
|---|---:|---:|---:|---:|---:|---|
| `1–2` rows | `6` | `0.97741` | `0.98848` | `0.93886` | `0.99778` | SED strongly helps rare soundscape positives; Jung21 hurts. |
| `3–10` rows | `3` | `0.98810` | `0.99138` | `0.99820` | `0.99429` | Very small; do not overfit. |
| `11–50` rows | `23` | `0.99574` | `0.99675` | `0.99811` | `0.99705` | Jung21/SED both viable. |
| `>50` rows | `10` | `0.98657` | `0.98960` | `0.98876` | `0.99293` | SED best. |

Slice implication: SED is the only branch that appears consistently useful for soundscape-rare labels. This supports a capped SED sidecar, but `v616` shows the public LB may already encode much of this signal.

### Taxon slices

| Taxon | Valid classes | Anchor | `v616` final | Jung21 | SED raw | Read |
|---|---:|---:|---:|---:|---:|---|
| Amphibia | `10` | `0.98784` | `0.99175` | `0.99038` | `0.99509` | SED strongest; moderate sidecar opportunity. |
| Aves | `11` | `0.98488` | `0.98840` | `0.96396` | `0.99347` | Jung21 hurts Aves on this proxy; cap it. |
| Insecta | `17` | `0.99766` | `0.99764` | `0.99828` | `0.99797` | Already saturated; no aggressive moves. |
| Mammalia | `3` | `0.98286` | `0.99779` | `0.99809` | `0.99792` | Looks promising but only three classes. |
| Reptilia | `1` | `0.97556` | `0.98306` | `0.99944` | `0.99278` | Single-class anecdote only. |

Taxon implication: broad sidecar movement should be reviewed by taxon. Aves is the main risk for Jung21; non-Aves slices look more sidecar-positive but are class-sparse.

### Per-class signals worth noting

Largest local `v616` lifts over anchor include:

- `74113` Mammalia (`train_n=10`, soundscape rows `2`): `+0.0319`
- `23158` Amphibia (`train_n=25`, soundscape rows `175`): `+0.0163`
- `grekis` Aves (`train_n=482`, soundscape rows `2`): `+0.0133`
- `plcjay1` Aves (`train_n=186`, soundscape rows `1`): `+0.0132`, driven by SED while Jung21 was negative
- `555146` Amphibia (`train_n=18`, soundscape rows `210`): SED raw lift `+0.0486`, final lift `+0.0130`

Classes where the local story warns against naive branch selection:

- `strher2` Aves: SED raw `+0.0505`, Jung21 `-0.1277`, final basically flat due blend mechanics.
- `517063` Amphibia/no-train: Jung21 `-0.0115`, SED `-0.0021`, final near flat.
- Several `47158son*` insect/no-train classes are already near AUC `1.0` locally; moving them is more likely overfit than useful.

## 4. Safe ensemble constraints recommended from data/slices

### Global constraints

- Keep anchor mass high: **at least `0.90`**, preferably `0.92–0.96` for plateau-family sidecars.
- Total sidecar cap: **`<=0.10`** for any submission candidate unless an independent validation source clears a higher bar.
- Per-branch cap for known plateau-adjacent branches:
  - SED raw: `<=0.04–0.06` verifier range; submit default `<=0.04` unless file/site/taxon q05 improves clearly.
  - Jung21 raw: `<=0.04`; lower if Aves/common-class degradation persists.
  - ProtoSSM siblings/SYD52p: `<=0.02` audit-only unless a new hidden-safe source appears.
  - HGNet sidecars: `<=0.04–0.06` audit/private verifier only; prior `0.06` anchored submissions tied.

### Class/taxon constraints

- No per-class adaptive weights for submission unless **both** leave-site and leave-file CV lift materially clear zero. Current `v616` selector does not.
- For classes with fewer than `3` positives or fewer than `20` negatives in the proxy, use anchor-only or a fixed family-level cap; do not learn class weights.
- Taxon cap defaults:
  - Aves: cap Jung21 tightly because Aves slice was weak (`0.96396` vs anchor `0.98488`).
  - Insecta/no-train sonotypes: avoid large movement; proxy already saturated.
  - Amphibia/Mammalia/Reptilia: sidecars can be helpful, but class counts are small, so require manual class audit.
- No more than `25%` of valid proxy classes should receive nonzero adaptive sidecar boosts without OOF evidence.

### Group-robust validation constraints

- Report site and file bootstrap versus **both** anchor and nearest tied recipe.
- Require leave-one-site and leave-one-file summaries before any submission.
- Treat `<8` sites or `<60` valid classes as private-verifier evidence only, not submission approval.
- Include time-position bins (`<=20s`, `25–40s`, `>=45s`) as sanity checks, especially for models using endpoint/window logic.

### No-call/background constraints

- Do not calibrate no-call behavior from rank-scaled outputs or unlabeled artifact rows.
- Negative/background losses need broad OOF negative coverage. The current negative mask covered only `26/512` rows and produced noise-sized lift; do not scale or ensemble from it yet.
- Any future no-call/background sidecar must report predicted mass on labeled-positive, labeled-sparse, and genuinely empty/no-call proxy rows if such rows are available.

### Hidden-safe implementation constraints

- Use raw hidden-safe branch outputs, not public dry-run CSVs, in any candidate kernel.
- Every kernel should write raw branch outputs and final output:
  - `submission_anchor_raw.csv`
  - `submission_<branch>_raw.csv`
  - `submission_before_alignment.csv`
  - `submission.csv`
- Hard-fail on missing/misaligned/nonfinite/constant branch output.

## 5. Missing but valuable data/feature analysis before final submission

1. **Leave-file bootstrap and leave-file CV for the exact Phase 2 candidate.** Current evidence is site-heavy; files are the real leakage unit within sites.
2. **Broader validation rows/classes.** The current 190-row/42-class overlap is too narrow. Use the full 739 collapsed labeled soundscape rows where branch generation can be materialized.
3. **OOF-based class/taxon residual analysis.** Need class-level residuals from true OOF/cache artifacts, not only dry-run train-soundscape overlap.
4. **A real no-call/background proxy.** Current labels lack empty rows after collapse. Need either trusted no-call rows or a negative-only row set to tune suppression safely.
5. **Per-family ablation on exact hidden-safe outputs.** For any candidate, report anchor-only, +SED, +Jung21, +HGNet, and pairwise combinations versus the nearest tied recipe.
6. **Hidden/test row distribution comparison.** Compare parsed site/date/time/file counts from `sample_submission.csv`/private verifier dry-run logs against train-soundscape sites. Current sample shows `S05`, unseen in artifact overlap.
7. **Calibration analysis in rank space and probability space.** Rank blends are evaluated differently from raw probabilities; both should be tracked, but no-call/background conclusions need probability-calibrated models only.
8. **Taxon-weight movement report.** Before submission, summarize mean/max movement per taxon and flag any taxon with movement above `2x` global median.
9. **Source-family duplicate audit after candidate assembly.** The SYD52p and Raunak/Samejima examples show how easily new-looking branches duplicate existing signals.
10. **External-train/no-train class coverage plan.** The 28 no-train classes require explicit handling; either anchor them, use soundscape/OOF teacher signal, or cap movement.

## Bottom line for coordinator

From the data/slice perspective, the safest Phase 2 ensemble is still an anchored blend, but it should be **constraint-first** rather than local-AUC-first:

- Primary: Samejima/visual-style anchor.
- Sidecar candidates: SED raw first, Jung21 second, HGNet only as a comparator if available.
- Reject: per-class adaptive `v616`, SYD52p micro-gains, and any repeat of EoS/PCEN/visual/HGNet scalar variants without new source evidence.

A candidate should not be submitted unless it improves over both the anchor and nearest tied recipe under file/site/taxon constraints, with enough matched classes to make the result meaningful.
