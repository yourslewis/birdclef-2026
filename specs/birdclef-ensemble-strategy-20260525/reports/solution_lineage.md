# Phase 1A — Solution Lineage and Diversity Report

_Date: 2026-05-25_

## Executive conclusion

The current BirdCLEF 2026 `0.949` plateau is not one model; it is a cluster of closely related public-rank/ProtoSSM/SED/Perch/RankPower/visual recipes. We have several hidden-safe repo-owned candidates that preserve the plateau (`v575`, `v576`, `v604`, `v611`, `v612`, `v616`) and several public/source replays that tie it (`v608`, `v614`, `v615`, many earlier S/G-sidecar variants). None of the low-weight sidecar additions has improved hidden public LB despite strong train-soundscape/local lift.

For the ensemble search, treat the `0.949` anchor family as a single saturated family. Include only genuinely different raw branches as *diagnostic/search axes*, not as separate approval evidence. The most useful current reusable artifacts are:

1. Samejima/visual-style `0.949` anchor raw output from `v616`.
2. Jungchan Model21 raw branch from `v616` / public `subm_21.csv`.
3. Samejima/Raunak SED raw branch from `v616` / public branch outputs.
4. Praxel/Samejima HGNet raw sidecars from `v611`/`v612` only as held comparison axes.
5. S14 / Alexy / G124 / exportable SED lanes as future-source research, not immediate slot candidates.

Do **not** submit scalar/per-class variants of `v616`, `SYD52p`, or another EoS/PCEN/visual/HGNet replay without new independent hidden-safe signal. The main lesson from `v611`, `v612`, and `v616`: local train-soundscape movement is a rejection screen, not approval evidence.

## Sources inspected

Minimum requested sources were inspected, plus relevant artifacts:

- `specs/birdclef-ensemble-strategy-20260525/spec.md`
- `docs/BIRDCLEF_TWO_DAY_EXPERIMENT_SPEC_20260524.md`
- `docs/BIRDCLEF_096_FRONTIER_PLAN_20260518.md` tail/relevant sections
- `docs/BIRDCLEF_AUTORESEARCH_LOG.md` tail/relevant sections
- `docs/BIRDCLEF_096_ANCHORED_BLEND_IMPLEMENTATION_PLAN_20260524.md`
- `artifacts/anchored_blend_audit/sidecar_manifest_20260525T0000Z.json`
- `artifacts/anchored_blend_audit/sidecar_grid_20260525T0200Z_fast.json`
- `artifacts/anchored_blend_audit/sidecar_grid_20260525T0200Z_top_stability.json`
- `artifacts/anchored_blend_audit/v616_syd52p_grid_fast_20260525T1000Z.json`
- `artifacts/anchored_blend_audit/v616_syd52p_top_stability_20260525T1000Z.json`
- `artifacts/anchored_blend_audit/v616_per_class_selector_20260525T0810Z.json`
- `artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/validation_stats.json`
- `artifacts/source_audits/v616_hidden_safe_branch_extraction_20260525T0400Z/summary.json`
- `artifacts/source_audits/v616_hidden_safe_branch_extraction_20260525T0400Z/jungchan_excerpts.json`
- `artifacts/source_audits/v616_hidden_safe_branch_extraction_20260525T0400Z/jungchan_model21_block_summary.json`
- `artifacts/public_kernels_20260524_frontier_candidates/source_audit_20260524T2200Z_newleads/summary.json`
- `artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/summary.json`
- `artifacts/pseudolabels/threshold_sweeps/*.json`
- SED/export smoke artifacts under `artifacts/sed_oof/` and `artifacts/sed_oof_teacher_students/`

Safety labels used below:

- **hidden-safe**: already reruns on current Kaggle mount / hidden `test_soundscapes` path with guards, or scored as a valid competition submission.
- **private-verifier-ready**: repo-owned verifier/kernel/config exists, but it is not an approved competition candidate.
- **public-output-only**: only dry-run/public branch CSVs are currently available; not submit-safe unless ported/rerun hidden-safely.
- **invalid**: malformed output, timeout/no-score, constant/fallback, or scored far below frontier.
- **blocked**: missing source/API/assets or unresolved dependency prevents hidden-safe verification.

## Candidate family matrix

| Family | Members / public scores | Key artifact paths | Independence / diversity | Hidden-test safety | Ensemble decision | Notes |
|---|---:|---|---|---|---|---|
| EoS5 / Model5 / RankPower / Karnak plateau anchor | `v574=0.949`, `v575=0.949`, `v576=0.949`; many EoS/RankPower derivatives | `kaggle-kernels/v575-eos5-repo-confirm/`; `kaggle-kernels/v576-eos5-model5-only/`; history in `docs/BIRDCLEF_AUTORESEARCH_LOG.md` around `v574-v576` | **Low** within family | **hidden-safe** | **Include exactly one anchor equivalent; reject duplicates** | `v576` proved Model5-only preserves `0.949`; later SafeAlign/scalar/EoS siblings are near-duplicates. Useful as baseline lineage, not as multiple ensemble members. |
| Public946 / visual / Samejima-style anchor | `v608=0.949`; Samejima visual used as anchor; `v616` anchor raw | `artifacts/anchored_blend_audit/sidecar_grid_inputs_20260525T0200Z/samejima_visual_anchor.csv`; `artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission_anchor_raw.csv`; `artifacts/source_audits/v616_hidden_safe_branch_extraction_20260525T0400Z/samejima_visual.py.txt` | **Low-Medium** vs EoS; anchor-level, not fresh signal | **hidden-safe** via repo-owned v616 scaffold; public CSVs alone are public-output-only | **Include as primary search anchor/control** | Best current anchor for branch grids. It tied but is stable; all sidecar claims should be measured against it. |
| PCEN / PriorField / EoS6 sidecar plateau | `v604=0.949`; `v599-v602=0.949`; many Pilkwang/Beicicc/Gendaijin/Ykuroka clones | `kaggle-kernels/v604-pilkwang-pcen-sidecar-verify/`; `docs/BIRDCLEF_AUTORESEARCH_LOG.md` `v599-v604`; `artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/ykuroka__*.csv` | **Low** | **hidden-safe** for v604; many clones invalid/sample-shaped | **Reject for new ensemble slots** | Verified safe enough to tie, but saturated. Do not duplicate PCEN/PriorField forks unless new non-overlapping branch appears. |
| Jungchan / Raunak Model21 / ProtoSSM / SED branch line | `v614=0.949` Raunak direct; `v615=0.949` Jungchan direct; `v616=0.949` repo-owned anchored Jung21+SED | `kaggle-kernels/v616-anchored-jung21-sed-blend/`; `artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission_jung21_raw.csv`; `.../submission_samejima_sed_raw.csv`; public grid inputs under `artifacts/anchored_blend_audit/sidecar_grid_inputs_20260525T0200Z/` | **Medium** at raw branch level; **Low** at final LB level | **hidden-safe** for v616 raw branches; public branch inputs are public-output-only | **Include raw branches in search; reject scalar variants** | Raw branches are genuinely different from anchor by correlation (`v616_jung21` corr vs anchor `0.401`, SED corr `0.239` in `validation_stats.json`), but hidden public score still tied. Use once as branch axes, not as proof of improvement. |
| v616 fixed rank blend | `v616=0.949` | `artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission.csv`; `validation_stats.json`; `artifacts/anchored_blend_audit/v616_per_class_selector_20260525T0810Z.json` | **Medium branches, low final novelty** | **hidden-safe** | **Hold as baseline/control; do not resubmit variants** | Strong actual-output local lift (`AUC 0.9934807`, q05 bootstrap `+0.0017568`) still tied public LB. Per-class selector had all-row lift but essentially zero leave-site CV lift (`+0.0000035`), so adaptive variant is overfit. |
| Praxel HGNet / Samejima HGNet sidecars | `v598=0.860` standalone; `v611=0.949`; `v612=0.949` | `kaggle-kernels/v598-samejima-hgnet-openvino-artifact/`; `kaggle-kernels/v611-anchored-hgnet-sidecar/`; `kaggle-kernels/v612-anchored-sameji-hgnet57-pt/`; plan in `docs/BIRDCLEF_096_ANCHORED_BLEND_IMPLEMENTATION_PLAN_20260524.md` | **Medium-High raw sidecar**, **Low final lift** | **hidden-safe** for v611/v612; standalone HGNet is valid but bad | **Hold for diagnostics; reject repeat slots** | HGNet raw sidecars have low-ish anchor correlation and strong local gates, but both anchored submissions tied. Standalone HGNet dropped badly. Include only if future ensemble needs an orthogonal diagnostic sidecar, not as immediate candidate. |
| SYD52p / P949 SYD / Kijiang branch clones | No competition submission; fresh scout rejected after `v616` tie | `artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/joriahmed__bc2026-p949-syd-ort-effv2-a08__subm_52p.csv`; `artifacts/anchored_blend_audit/v616_syd52p_grid_fast_20260525T1000Z.json` | **Low**; branch clones of Jungchan/Raunak/Samejima | **invalid** final outputs; branch CSVs are public-output-only | **Reject** | Fresh scout showed malformed finals (`243x235`, many bad values) and branches identical/near-identical to existing lines. `syd52p` added only `+0.000020` local AUC over already-tied v616. |
| Alexy NS1 CNN/noisy-student | `v613=0.923` direct | `artifacts/public_kernels_20260524_frontier_candidates/source_audit_20260524T2200Z_newleads/alexycactus__birdclef-2026-ns1-ensemble.json`; `kaggle-kernels/v613-alexy-ns1-sidecar/FEASIBILITY_NOTES.md` | **High** diversity vs anchor | **blocked** for repo-owned source/API; direct output valid-ish but `192x235` and weak | **Hold for source recovery only** | True CNN/noisy-student signal is independent, but direct family is far below plateau. Current Kaggle API 403 blocks source/output recovery. Do not spend a slot unless source is recovered and a low-weight hidden-safe sidecar passes grid/stability. |
| StudyExchange S14 / BidirProtoSSM + Snowflake SED | No direct submission; source expectation `~0.946+`; valid public output | Mentioned in `docs/BIRDCLEF_096_ANCHORED_BLEND_IMPLEMENTATION_PLAN_20260524.md`; prior output/audit referenced in `docs/BIRDCLEF_096_FRONTIER_PLAN_20260518.md`; `artifacts/anchored_blend_audit/sidecar_manifest_20260525T0000Z.json` | **Medium** | **public-output-only / partly blocked** | **Hold as optional sidecar if source/output accessible** | Valid `240x235`, corr vs visual anchor `0.9345`, local AUC `0.991722`, but self-expectation below current best. Use only as low-weight branch after hidden-safe source repair. |
| S114/G116/G123/G124 protected deltas/rankblend | `v587=0.949`, `v588=0.949`, `v589=0.949`, `v593=0.949`; public `0.952` lead did not reproduce | History in `docs/BIRDCLEF_AUTORESEARCH_LOG.md`; G124 configs `configs/birdclef/g124_effv2s_public946_pseudo_smoke_20260525_v2sinit.json`, `configs/birdclef/g124_effv2s_public946_pseudo_pilot_20260525_v2sinit_allrows_ep8.json` | **Medium** if real G124 asset exists; current wrappers low | **blocked** by missing private G124 assets; current reproductions hidden-safe but plateau | **Hold research; reject wrapper replays** | Missing asset path remains the blocker (`g124_fold1_fp16.pt` / private pseudo assets). V2S-init training smoke fixed scratch training but blend lift was effectively zero/unstable. |
| G124 EffV2-S V2S-init student | No competition submission; smoke/pilot rejected | Configs above; trainer logs cited in docs: `logs/g124_v2sinit_smoke_20260525T0005Z.log`, `logs/g124_v2sinit_allrows_ep8_20260525T0025Z.log`; audit paths in docs under `artifacts/pseudolabels/audits/` | **Medium-High** as a model-training lane | **private-verifier-ready for training/export**, not submission-ready | **Hold for different data/architecture, not current ensemble** | Smoke AUC improved dramatically over scratch (`0.956867` subset), but all-row pilot best blend weight was `0.0025` with only `+0.00000066` lift and negative group stability. |
| Real/exportable SED students | No competition submission; model gates failed | `configs/birdclef/sed_b0_q3cap80_ep12init_exportsmoke_5s_160_allcls_20260525.json`; `configs/birdclef/sed_b0_oofteacher_b0v26_nfnetv29_soft_512_ep3_20260525.json`; `configs/birdclef/sed_b0_oofteacher_b0v26_nfnetv29_soft_negaux002_512_ep3_20260525.json`; artifacts under `artifacts/sed_oof/` and `artifacts/sed_oof_teacher_students/` | **High** if quality improves | **private-verifier-ready/export-safe**, not score-ready | **Hold for target-design work** | Export/runtime path is good (TorchScript/ONNX + CPU smoke), but AUCs `0.754`, `0.819021`, `0.819410` are below scale/submit threshold. |
| PerchFusion v951 / TTA heavy | `v609` timeout/no score | `docs/BIRDCLEF_096_FRONTIER_PLAN_20260518.md` `v609`; `docs/BIRDCLEF_AUTORESEARCH_LOG.md` | **Medium-High** structurally | **invalid** as-is due runtime | **Reject direct; hold only if precomputed/simplified** | Source is distinct, but hidden runtime timeout makes direct rerun unsafe. |
| Gandharva B3 clean-audio checkpoints | `v610=0.852` | `kaggle-kernels/v610-gandharva-b3-infer/`; docs/logs around `v610` | **Medium** architecture, but poor domain transfer | **hidden-safe but bad** | **Reject** | Clean-audio fold AUC did not transfer to hidden soundscape LB. |
| Eslam v26C ProtoSSM repair | `v607=0.934` | `kaggle-kernels/v605-eslam-v26c-verify/`; `kaggle-kernels/v606-eslam-v26c-proto-repair/`; `kaggle-kernels/v607-eslam-v26c-proto-save/` | **Low-Medium** | **hidden-safe after repair but bad** | **Reject unless missing student ONNX assets appear** | Repaired mechanically, but fallback two-way ProtoSSM+SED scored below plateau. |
| Cheny public0952 / Perch-prior-probe | `v595=0.899` | Docs/logs around `v595`; 2026-05-21 notes in `docs/BIRDCLEF_AUTORESEARCH_LOG.md` | **Medium** on paper | **hidden-safe but bad** | **Reject** | Explicit `public0952` clue did not reproduce. |
| Misc fresh invalid/fallback notebooks | Deepanshu, WildSound, Tulay, Scott, Anatoly, Jacques, atfujita/babych probes | `artifacts/public_kernels_20260524_frontier_candidates/source_audit_20260524T2200Z_newleads/summary.json`; `artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/summary.json` | N/A | **invalid** | **Reject** | Missing files, mock/all-zero/all-0.5/constant/report-only/no-output/malformed finals. |

## Near-duplicate evidence from local artifacts

A focused CSV comparison over available local branch outputs supports the duplicate grouping above:

- `samejima_sed` vs `raunak_sed`: `corr=1.000000`, `MAE=0`, `maxabs=0` using `artifacts/anchored_blend_audit/sidecar_grid_inputs_20260525T0200Z/{samejima_sed.csv,raunak_sed.csv}`.
- `v616_sed` vs `raunak_sed`: `corr=1.000000`, `MAE=0`, `maxabs=0` using `artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission_samejima_sed_raw.csv` and the 02UTC Raunak SED input.
- `jungchan_model21` vs SYD `subm_21`: `corr=1.000000`, `MAE=0`, `maxabs=0` using `artifacts/anchored_blend_audit/sidecar_grid_inputs_20260525T0200Z/jungchan_model21.csv` and `artifacts/public_kernels_20260525_fresh_scout/source_output_audit_20260525T1000Z/joriahmed__bc2026-p949-syd-ort-effv2-a08__subm_21.csv`.
- `raunak_protossm` vs SYD `subm_52p`: `corr=0.994349`, `MAE=0.002410`, `maxabs=0.318697`, so SYD52p is near-duplicate ProtoSSM-lineage, not a new model family.
- Against Samejima visual anchor, raw `Jung21` and `SED` look diverse (`corr=0.401` and `0.239` respectively), but the hidden-safe final blend still tied public LB (`v616=0.949`). Diversity at raw branch level is necessary but not sufficient.

## Include / hold / reject recommendations

### Include in immediate ensemble search

Use these as the first search pool, with fixed small weights and no competition submission until synthesis/verifier approval:

1. **Anchor:** `artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission_anchor_raw.csv` or 02UTC `samejima_visual_anchor.csv`.
2. **Jung21 raw:** `artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission_jung21_raw.csv`.
3. **Samejima/Raunak SED raw:** `artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission_samejima_sed_raw.csv`.
4. **Praxel/Samejima HGNet raw sidecars:** include as comparison axes only if outputs are available/recreated; do not prioritize a new HGNet-only submission.
5. **S14 raw output** only if source/output access is restored and it can be rerun hidden-safely.

Rules for this pool:

- Treat EoS5, EoS6, PCEN, PriorField, RankPower, and public946 final submissions as one plateau anchor family; do not add multiple final submissions as independent members.
- Prefer raw branches over final blended notebooks.
- Any candidate that relies on downloaded public dry-run CSVs is **not** hidden-safe until ported/rerun in a repo-owned verifier.

### Hold for later / research queue

- **Alexy NS1 CNN/noisy-student:** high diversity but direct `0.923` and current API/source access blocked. Resume only if source/assets can be recovered.
- **G124 EffV2-S / missing G124 assets:** high-upside idea, but current V2S-init pilots do not add useful blend signal. Needs real asset discovery or a fundamentally different student/target design.
- **Exportable SED / OOF-teacher students:** operationally viable export path, weak model gate. Continue only via target/data redesign with small-smoke AUC closer to `0.90`.
- **S14:** structurally interesting; hold for source repair and hidden-safe rerun, not direct public-output blending.
- **PerchFusion:** hold only if precomputed/simplified to solve runtime timeout.

### Reject / do not spend more slots

- Scalar variants of `v616`, including per-class selector variants: leave-site CV lift was effectively zero.
- `SYD52p` / P949 SYD branch increments: near-duplicate, microscopic local gain over tied `v616`, malformed public finals.
- Additional EoS/PCEN/PriorField/RankPower/visual clone replays without new assets.
- Standalone HGNet (`v598`), Gandharva B3 (`v610`), Eslam v26C repair (`v607`), Cheny public0952 (`v595`), Alexy direct replay (`v613`), and malformed/mock/constant/no-output kernels.

## Top 5 genuinely different candidates/branches to prioritize

1. **v616 hidden-safe raw branch bundle as the control pool**
   - Paths: `artifacts/kaggle_outputs/v616-anchored-jung21-sed-blend/submission_anchor_raw.csv`, `submission_jung21_raw.csv`, `submission_samejima_sed_raw.csv`, `validation_stats.json`.
   - Why: best currently reproducible hidden-safe branch bundle; raw branches are diverse from anchor, row-aligned, finite, nonconstant.
   - Caveat: fixed `v616` final tied public LB, so use as control/search input, not submit-ready proof.

2. **Praxel/Samejima HGNet raw sidecar family**
   - Paths: `kaggle-kernels/v611-anchored-hgnet-sidecar/`, `kaggle-kernels/v612-anchored-sameji-hgnet57-pt/`, implementation plan in `docs/BIRDCLEF_096_ANCHORED_BLEND_IMPLEMENTATION_PLAN_20260524.md`.
   - Why: most distinct low-weight artifact sidecar tested before `v616`; raw HGNet sidecars had strong local gates and low/moderate anchor correlation.
   - Caveat: standalone HGNet `v598=0.860`; anchored `v611/v612` tied, so only keep as a comparison branch.

3. **Alexy NS1 CNN/noisy-student sidecar, if source access is recovered**
   - Paths: `artifacts/public_kernels_20260524_frontier_candidates/source_audit_20260524T2200Z_newleads/alexycactus__birdclef-2026-ns1-ensemble.json`, `kaggle-kernels/v613-alexy-ns1-sidecar/FEASIBILITY_NOTES.md`.
   - Why: highest true architecture diversity (CNN/noisy-student, not just another RankPower/ProtoSSM branch).
   - Caveat: direct score `0.923`, output row behavior `192x235`, and current API 403 make it **hold**, not immediate include.

4. **StudyExchange S14 / BidirProtoSSM + Snowflake SED**
   - Paths: referenced in `docs/BIRDCLEF_096_ANCHORED_BLEND_IMPLEMENTATION_PLAN_20260524.md` and `artifacts/anchored_blend_audit/sidecar_manifest_20260525T0000Z.json`.
   - Why: moderately different S14/BidirProtoSSM/Snowflake SED branch; prior valid `240x235` output with corr `0.9345` vs visual anchor.
   - Caveat: own expected score below current frontier; needs hidden-safe source/access before use.

5. **G124 / exportable SED student lane as future new-signal research**
   - Paths: `configs/birdclef/g124_effv2s_public946_pseudo_smoke_20260525_v2sinit.json`, `configs/birdclef/g124_effv2s_public946_pseudo_pilot_20260525_v2sinit_allrows_ep8.json`, `configs/birdclef/sed_b0_oofteacher_b0v26_nfnetv29_soft_512_ep3_20260525.json`, `configs/birdclef/sed_b0_oofteacher_b0v26_nfnetv29_soft_negaux002_512_ep3_20260525.json`.
   - Why: most plausible path to genuinely new hidden-safe signal beyond public branch recombination.
   - Caveat: current runs fail promotion gates; do not include current outputs in ensemble search until a better small-smoke/model gate appears.

## Bottom line for coordinator

The actionable ensemble search should start small: `Samejima/v616 anchor + Jung21 raw + Samejima/Raunak SED raw`, optionally comparing a single HGNet raw sidecar if available. That pool is genuinely more diverse than another public `0.949` final blend, but `v616` already showed it can tie rather than lift. Therefore the coordinator should require validation evidence beyond local AUC before Phase 2 promotion, and should block any near-duplicate scalar/per-class/SYD/EoS/PCEN replay.
