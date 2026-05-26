# Ranked queue — BirdCLEF hill-climb cron — 2026-05-26 08:20 UTC

## Live state verified
- Best public LB remains **0.949**; `v616` remains the repo-owned tied baseline to beat.
- Latest live Kaggle Bearer API listing: `v617=0.949`, `v618=0.946`, `v619=0.944`, `v620=0.949`; no post-reset submissions on 2026-05-26 UTC.
- UTC daily slots used: **0/5** at ~08:20Z; about **15.7h** to reset.
- Active job check: no local BirdCLEF/Kaggle/ClawTeam jobs and no relevant trainer jobs before launching this run; trainer only showed unrelated LTX uvicorn.
- Fresh public scout: quick web searches for BirdCLEF 2026 EfficientAT/PANNs/0.949 notebooks surfaced no clean new public notebook lead; the scout+critic report is `specs/birdclef-hillclimb-cron-20260525/reports/scout_critic_20260526T0815Z.md`.

## Slot decision
No Kaggle submission this early UTC run. No candidate is verifier-grade or competition-format: DyMN10 is a 72-label specialist; G124 local sidecar lift remains noise-sized; B0 soft-only did not help v616 audit; public candidates are tied/dropped/duplicate/malformed. Submitting now would be leaderboard probing rather than high-information slot use.

## Work completed this run
1. Prevented duplicate work: checked local/trainer processes and recent artifacts; no active training/submission jobs were found.
2. Verified Kaggle state with Bearer API v1: best stays `0.949`, daily slots `0/5`, no pending submissions.
3. Ran scout+critic role bundle; it recommended EfficientAT `dymn10_as` as the next bounded no-slot data point and rejected early-day submission.
4. Trained EfficientAT DyMN10 AudioSet soundscape head:
   - Config: `configs/birdclef/efficientat_dymn10_audioset_soundscape_nonaves_notrain_nocall_siteS08_ep12_20260526.json`.
   - 1,478 official train-soundscape 5s windows; 72 non-Aves/no-train labels plus no-call aux; site holdout `S08`; 12 epochs.
   - Metrics: S08 macro AUC `0.568586` over 18 valid classes; no-train AUC `0.553327`; best val loss `0.428341`.
   - Verification: finite/nonconstant `120 x 72` holdout preds; TorchScript head smoke passed `(2,960)->(2,72)+(2,1)`.

## Ranked queue after this run

| Rank | Candidate | Evidence / value | Decision |
|---:|---|---|---|
| 1 | **AudioSet multi-site evaluation + DyMN10 wrapper decision** | DyMN10 beat MN10 and PANNs on the same S08 contract (`0.568586` vs `0.488240`/`0.517333`), but no-call AUC is still invalid and output is 72-label only. | **ACCEPTED next no-slot analysis** |
| 2 | **G124/V2S hard-confidence / power ablation** | G124 localmax had strong proxy AUC (`0.960094`) but v616 blend lift was only `+0.00000339`; needs target-shape/power separation. | **ACCEPTED bounded data point** |
| 3 | **234-class DyMN10 sidecar wrapper/proxy audit** | Potential rare-slice sidecar if multi-site confirms; requires hidden-safe feature extraction and 234-column wrapper. | **NEEDS REVISION** |
| 4 | **Fresh pretrained model-family scout** | Public code frontier is saturated; BirdNET/Bioacoustics Model Zoo/Perch2/PaSST/HTS-AT/BEATs could add genuinely new signal if source-clean. | **ACCEPTED research lane** |
| 5 | **Late-day guarded exploratory fallback queue** | Only for <3h-to-reset if no verifier-grade package emerges; avoid v616/v617/v620 replays and malformed/static outputs. | **DEFER** |
| 6 | **B0 soft-only raw sidecar** | Strong B0 smoke (`0.911067`) but did not help v616 sidecar audit. | **LOWER PRIORITY** |

## Critic / Red Team
- DyMN10 is the first AudioSet soundscape head that clearly beats PANNs on S08, so it deserves follow-up, but the validation split is narrow and no-call AUC remains unsupported.
- A 72-label specialist cannot be submitted or blended into hidden output without a 234-class wrapper and alignment audit.
- Early-day slot use is still unjustified: no current output is competition-format, duplicate-safe, and verifier-approved.

## Verifier decision
- No submission approved.
- Training inputs are competition-rule safe: official train soundscapes plus public AudioSet EfficientAT checkpoint.
- Local artifact checks passed: finite/nonconstant predictions and TorchScript head smoke.
- Required before slot: 234-column output path, row/column alignment, nonfinite/constant guards, duplicate-matrix check, source/license check, and v616-sidecar audit with non-noise lift or high information value.

## Next exact action
Run a multi-site / leave-one-site evaluation harness for PANNs vs EfficientAT MN10 vs EfficientAT DyMN10; if DyMN10 remains strongest, design a small 234-class sidecar wrapper for no-slot audit against v616. If it collapses, pivot to G124 hard-confidence/power ablation.
