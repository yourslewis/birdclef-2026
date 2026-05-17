# BirdCLEF Next Signal After v571 — 2026-05-17

Status: planning / no submission
Anchor: public best remains **0.946**. v571 was hidden-safe and tied `0.946`, proving repo-owned packaging works but tiny Proto/SED rank-weight tweaks are exhausted.

## Latest evidence

- Direct public-kernel lane is stopped:
  - v566 tied `0.946`;
  - v567 dropped to `0.944`;
  - v568 failed hidden rerun;
  - v569 preflight 404;
  - v570 failed hidden rerun from RAM.
- Repo-owned repackage lane is safer but needs real signal:
  - v571 changed only `0.60/0.40` Proto/SED rank blend to `0.5964/0.4036` and tied `0.946`.
  - Conclusion: do not spend more slots on rank-weight micro-nudges.
- Same-teacher random-init/focal sidecar lane is weak:
  - NFNetL0 focal+BCE sqrt-cw smoke failed practical gate (`best_val_auc≈0.8988`, corr `0.7073`).
  - EffV2-S focal+BCE sqrt-cw smoke failed (`best_val_auc≈0.7078`, corr `0.2458`).
  - External-init B0/public946 full diagnostics are strong standalone but too teacher-correlated; local V2S lifts have already failed to transfer (`v560=0.945`).

## Decision

Next useful work must create **new source signal**, not another public946 reshuffle.

The next candidate family should be one of:

1. **External/pretrained CNN → public946 distillation, but with cross-site gates**
2. **OOF teacher / real SED-MIL student with frame/local target structure**
3. **Rare/non-bird specialist using new features or new model outputs, not public946-only calibration**

## Recommended next candidate: v572 external-pretrained B0/B3 robustness diagnostic

### Hypothesis

Prior BirdCLEF/XC-style external pretraining was repeatedly useful in 2025 writeups. Our same-teacher public946 students are too correlated. A better external-pretrained diagnostic should test whether the representation adds stable signal across file/site/taxon splits before any Kaggle slot.

### Why this is the best next step

- Existing tooling already supports target-species external manifests (`scripts/birdclef_external_pretrain_manifest.py`).
- Existing configs show B0 and B3 external-pretrain lanes are already in the repo:
  - `xc_b0_q3_cap80_external_pretrain_balanced_ep12.json`
  - `xc_b3_q3_cap80_external_pretrain_lr1e4_smoke.json`
  - public946 B3 pretrained configs exist.
- This attacks the known failure mode: public946-only sidecars have local lift but do not transfer to LB.
- It is cheaper and more controllable than starting a new full MIL architecture immediately.

### Concrete plan

#### Phase 1 — Source/manifest audit only

Run or refresh manifest summary without training:

```bash
python3 scripts/birdclef_external_pretrain_manifest.py \
  --data-root /home/yourslewis/birdclef-2026/data \
  --output-dir artifacts/external_pretrain/manifest_q3_cap80_20260517 \
  --max-per-species 80 \
  --min-rating 3.0 \
  --prefer-quality
```

Gate:

- report species coverage, class/taxon coverage, missing files, capped species;
- reject if non-bird/rare classes remain too sparse to justify training;
- keep train/val split deterministic.

#### Phase 2 — B0/B3 pretrain comparison smoke

Run short smokes only, no Kaggle submission:

- B0 baseline using `efficientnet_b0`, 5s/128 or 160 mel, q>=3 cap80, balanced ep4/ep6.
- B3 smoke only if B0 learns and runtime remains manageable.

Gate:

- validation improves over random-init at comparable budget;
- per-class/taxon coverage is not just common-bird memorization;
- checkpoint export works.

#### Phase 3 — Distill from public946 with external init

Load the best external checkpoint into public946 student training:

- `initial_checkpoint`: best external-pretrain checkpoint
- `initial_load_head=false`
- teacher: public946 `teacher_sed85_rankblend15.npz`
- duration/mel: match best pretrain geometry first; avoid changing geometry and loss simultaneously.

Gate:

- full-row student AUC must be competitive;
- teacher correlation must be meaningfully below B0/V2S saturated sidecars, or blend lift must clear prior failed local-lift bar by a margin;
- leave-one-file/site or at least source/taxon stability must be non-negative.

#### Phase 4 — Only then package

Package at most one slot candidate if diagnostics pass:

- repo-owned kernel only;
- hidden-safe verifier from v571;
- no direct public notebooks;
- one weight only, chosen from full-row/crossfit evidence.

## Alternate medium-term candidate: real SED/MIL v573

If external-pretrain diagnostics still fail, stop public946 clip-student work and move to a true frame/local SED-MIL student:

- extend `scripts/birdclef_sed_pilot_train.py` or create a dedicated public946 SED-MIL trainer;
- use raw SED stream / local window maxima / MIL pooling targets rather than only clip-level final rankblend;
- validate on small file batches, then full-row blend audit;
- submission only after export/runtime validation.

## Stop rules

Do not submit if:

- the only evidence is a tiny local blend lift against public946;
- the model is highly teacher-correlated and resembles v560/V2S failure;
- the candidate changes only final rank weights or gate thresholds;
- hidden-size robustness is untested;
- the artifact relies on direct public notebook outputs.

## Next concrete action

Run the external manifest audit and summarize coverage. If coverage is healthy, run a short B0 external-pretrain smoke. No Kaggle submission before those gates.

## 2026-05-17 execution update: manifest audit + q0 smoke

Phase 1 manifest audit was run on the GPU server.

High-quality q>=3 cap80 manifest:

- Output: `artifacts/external_pretrain/manifest_q3_cap80_20260517/`
- Rows: `2659` total, `2470` train, `189` val
- Class coverage: `Aves=2605`, `Amphibia=38`, `Mammalia=16`, `Insecta=0`, `Reptilia=0`
- Weakness: `72` target species have fewer than five available rows and `54` target species are missing after filter.

Broader q>=0 cap80 manifest:

- Output: `artifacts/external_pretrain/manifest_q0_cap80_20260517/`
- Rows: `3388` total, `3050` train, `338` val
- Class coverage: `Aves=3138`, `Amphibia=182`, `Mammalia=49`, `Insecta=18`, `Reptilia=1`
- Weakness: still `28` missing target species, and `652` rows are unrated/zero-quality.

A B0 q0/cap80 smoke was run with config `configs/birdclef/xc_b0_q0_cap80_external_pretrain_smoke_20260517.json`:

- `512` manifest files, 2 epochs, pretrained EfficientNet-B0, 5s/128 mel, Focal BCE, pos_weight_sqrt
- Output: `artifacts/external_pretrain/xc-b0-q0-cap80-external-pretrain-smoke-20260517/`
- Result: holdout macro AUC `0.482335` over `88` valid classes, runtime `18.2s`, TorchScript `15.389 MB`

Conclusion: q0 improves non-bird coverage but is too noisy/sparse to scale blindly. Do not submit or package a q0-derived external-pretrain candidate. If continuing external pretraining, prefer either existing q3 bestloss/high-quality checkpoints for bird representation or build a targeted non-bird/rare data audit instead of lowering quality globally.

## 2026-05-17 execution update: rare/non-bird source audit

A CPU-only rare/non-bird audit was run on the trainer and saved to ignored artifact `artifacts/external_pretrain/rare_nonbird_audit_20260517.json`.

Target-species source rows by taxon:

- Amphibia: `35` species, `451` total rows, `57` q>=3 rows, `49` q>=4 rows, `393` zero/unrated rows
- Aves: `162` species, `34799` total rows, `21217` q>=3 rows, `16980` q>=4 rows, `12190` zero/unrated rows
- Insecta: `28` species, `199` total rows, `0` q>=3 rows, `0` q>=4 rows, `199` zero/unrated rows
- Mammalia: `8` species, `99` total rows, `21` q>=3 rows, `19` q>=4 rows, `66` zero/unrated rows
- Reptilia: `1` species, `1` total row, `0` q>=3 rows, `0` q>=4 rows, `1` zero/unrated row

Coverage summary:

- `28` target species have no train rows at all.
- `49` target species have no q>=3 rows.
- `69` non-bird target species have fewer than five q>=3 rows.
- Only `3` bird target species have fewer than five q>=3 rows.

Interpretation: the external-pretrain bottleneck is not a general bird-representation problem; it is heavily concentrated in non-bird/rare taxa. A global q0/q>=0 pretrain mostly injects noisy/unrated non-bird labels and already failed the B0 smoke. The next credible source-signal step should therefore be targeted non-bird/rare data acquisition or specialist construction, not lower-quality global pretraining.

Revised next action: build a bounded non-bird/rare specialist plan (Amphibia/Mammalia first, Insecta/Reptilia require external discovery or conservative abstention), then only train if it has enough verified examples or a reliable source-backed pseudo-label target.

## 2026-05-17 execution update: student stability helper

`birdclef_student_pool_blend_audit.py` now supports optional stability checks for the ranked best blends:

- `--bootstrap-iters`, `--bootstrap-group`, `--bootstrap-seed`
- `--leave-one-group`
- `--stability-top-n` to avoid running expensive stability on every aligned artifact

Site-stability audit output:

- `artifacts/pseudolabels/audits/public946_sed85_rankblend15_student_pool_site_stability_20260517T0855Z.json`
- `108` scanned student files, `40` row/label-aligned
- stability computed for top `8` blends with `50` site-bootstrap iterations and leave-one-site

Top local-stable candidates:

1. `pl-r2-v2s-v508-soft-p100-5s-pretrained-lr1e4-ep20-bestval`, weight `0.05`
   - local lift `+0.000168656`
   - student/teacher corr `0.3752`
   - leave-one-site `p_lift_gt_0=1.0`, min lift `+0.000063181`
   - site-bootstrap `p_lift_gt_0=0.94`, q05 lift `-0.000020597`
2. `pl-r1-convnext-tiny-v508-soft-p100-lr3e4-nomix-ep20-bestval`, weight `0.05`
   - local lift `+0.000100392`
   - student/teacher corr `0.3929`
   - leave-one-site `p_lift_gt_0=1.0`, min lift `+0.000039548`
   - site-bootstrap `p_lift_gt_0=0.96`, q05 lift `+0.000007196`

Interpretation: cross-site stability is necessary but still not sufficient. V2S/ConvNeXt sidecar families have already under-transferred on public LB (`v560=0.945`, `v564=0.942`, `v565=0.943`), so these local-stable blends should not be submitted directly. Use this helper as a rejection/triage gate for future genuinely new-source candidates.

## 2026-05-17 execution update: reproducible rare/non-bird audit script

Added `scripts/birdclef_rare_nonbird_source_audit.py` to make the rare/non-bird coverage audit reproducible. The script writes:

- `rare_nonbird_source_summary.json`
- `rare_nonbird_species_coverage.csv`
- `rare_nonbird_species_only.csv`
- `amphibia_mammalia_q3_existing_manifest.csv`

Run on trainer:

```bash
python3 scripts/birdclef_rare_nonbird_source_audit.py \
  --data-root /home/yourslewis/birdclef-2026/data \
  --output-dir artifacts/external_pretrain/rare_nonbird_source_audit_20260517T0955Z
```

Key results:

- `234` target species; `72` non-bird species.
- `28` target species have no source rows at all.
- `49` target species have no q>=3 rows.
- `69/72` non-bird species have fewer than five q>=3 source rows and fewer than five q>=3 verified local audio rows.
- Non-bird status counts:
  - `needs_external_discovery=28`
  - `source_sparse_or_low_quality=30`
  - `trainable_low_quality_only=11`
  - `trainable_verified_q3=3`
- Amphibia/Mammalia q>=3 verified manifest contains only `54` rows across `21` species.

Conclusion: current local source data is not enough for a reliable Amphibia/Mammalia specialist submission candidate. It is enough to define a targeted acquisition/abstention plan. Do not train/package a non-bird specialist until external discovery or source-backed pseudo-label targets improve coverage.

## 2026-05-17 execution update: raw-SED 10s B0 diagnostic

Tested the alternate real SED/MIL-ish lane by training against raw public946 SED teacher output rather than final rankblend.

Configs:

- `configs/birdclef/pl_public946_sedraw_b0_10s_m160_lr3e4_ep8_smoke_20260517.json`
- `configs/birdclef/pl_public946_sedraw_b0_10s_m160_lr3e4_ep20_20260517.json`

Trainer outputs:

- Smoke: `artifacts/pseudolabels/students/pl-public946-sedraw-b0-10s-m160-lr3e4-ep8-smoke-20260517/`
  - best/last val AUC `0.943722` over `30` valid classes
  - final-all AUC `0.904111` over `42`
  - student/teacher corr `0.8385`
  - runtime `6.274s`
- Full-row ep20: `artifacts/pseudolabels/students/pl-public946-sedraw-b0-10s-m160-lr3e4-ep20-20260517/`
  - val AUC peaked around `0.993693` over `60` classes
  - final-all AUC `0.988647` over `75`
  - student/teacher corr `0.977247`
  - runtime `27.642s`, TorchScript `15.391 MB`

Blend/stability audit:

- Output: `artifacts/pseudolabels/audits/public946_sedraw_b0_10s_ep20_blend_audit_20260517T1102Z.json`
- Best tested student weight: `0.0025`
- Lift vs sed85/rankblend teacher: `-0.000003042`
- Leave-one-site p_lift_gt_0: `0.0`; every site-held-out lift was negative.

Conclusion: raw-SED 10s B0 is a strong mimic but not additive. Do not package or submit it. A true v573 needs a different frame/local target structure or model output, not merely raw SED row-level distillation.

## 2026-05-17 execution update: refreshed q3 B3 external-pretrain diagnostic

The B3 external-pretrain lane exposed a useful manifest hygiene issue and a marginal-but-not-submittable new-source signal.

Manifest hygiene:

- Older `artifacts/external_pretrain/manifest_q3_cap80/external_pretrain_manifest.csv`: `976` balanced q>=3 candidate rows but only `295` resolved to local files on trainer.
- Refreshed `artifacts/external_pretrain/manifest_q3_cap80_20260517/external_pretrain_manifest.csv`: the same balanced cap resolves all `976` rows.
- Future external-pretrain comparisons should use the refreshed manifest or explicitly verify resolved-file counts.

B3 q3 external-pretrain runs:

- `configs/birdclef/xc_b3_q3_cap80_manifest20260517_external_pretrain_balanced_ep6_20260517.json`
  - `976` examples
  - val macro AUC `0.650746` / `117` classes
  - best epoch `6`
  - runtime `37.4s`, TorchScript `41.991 MB`
- `configs/birdclef/xc_b3_q3_cap80_manifest20260517_external_pretrain_balanced_ep18_20260517.json`
  - val macro AUC `0.722691` / `117` classes
  - best val loss at epoch `10`
  - runtime `48.109s`, TorchScript `41.991 MB`

Interpretation: refreshed-manifest B3 learns real external signal and is much better than the old 128-row smoke, but it still trails the existing B0 q3 ep18-bestloss checkpoint (`0.747224` over `122`). B3 is not a better external representation yet.

Public946 B3-extinit distill:

- Config: `configs/birdclef/pl_public946_sed85_rankblend15_b3_xc_q3_manifest20260517_extinit_5s_m128_lr1e4_ep20_20260517.json`
- Output: `artifacts/pseudolabels/students/pl-public946-sed85-rankblend15-b3-xc-q3-manifest20260517-extinit-5s-m128-lr1e4-ep20-20260517/`
- Final-all student AUC `0.968505` over `75`
- Student/teacher corr `0.936244`
- Runtime `31.243s`, TorchScript `41.995 MB`

Blend/stability audit:

- Output: `artifacts/pseudolabels/audits/public946_b3_xc_q3_manifest20260517_extinit_blend_audit_20260517T1125Z.json`
- Best tested weight `0.05`
- Local lift `+0.000045896`
- Site-bootstrap p_lift_gt_0 `0.80`, q05 `-0.000060676`
- Leave-one-site p_lift_gt_0 `0.8889`; worst site `S09` lift `-0.000011376`

Conclusion: B3 refreshed-q3 extinit is mildly additive locally but not robust enough for a slot. Do not package/submit it yet. If the external lane continues, compare against a refreshed-manifest B0 rerun or move to a different architecture/target structure; do not assume old-manifest metrics are apples-to-apples.

## 2026-05-17 execution update: refreshed q3 B0 apples-to-apples diagnostic

A same-refreshed-manifest B0 check was run to compare against the B3 external-pretrain result.

B0 q3 external-pretrain:

- Config: `configs/birdclef/xc_b0_q3_cap80_manifest20260517_external_pretrain_balanced_ep18_20260517.json`
- Output: `artifacts/external_pretrain/xc-b0-q3-cap80-manifest20260517-external-pretrain-balanced-ep18-20260517/`
- `976` examples
- val macro AUC `0.717722` over `117` classes
- best val loss at epoch `13`
- runtime `28.725s`, TorchScript `15.389 MB`

Interpretation: on the same refreshed manifest and seed as the B3 run, B0 is slightly behind B3 (`0.722691`) but much smaller. Both refreshed seed73 runs are below the older B0 seed42 metric (`0.747224`) and should not be over-interpreted as an architecture ranking without fold/seed replication.

Public946 B0-extinit distill:

- Config: `configs/birdclef/pl_public946_sed85_rankblend15_b0_xc_q3_manifest20260517_extinit_5s_m128_lr3e4_ep20_20260517.json`
- Output: `artifacts/pseudolabels/students/pl-public946-sed85-rankblend15-b0-xc-q3-manifest20260517-extinit-5s-m128-lr3e4-ep20-20260517/`
- Final-all student AUC `0.992896` over `75`
- Student/teacher corr `0.970497`
- Runtime `14.896s`, TorchScript `15.391 MB`

Blend/stability audit:

- Output: `artifacts/pseudolabels/audits/public946_b0_xc_q3_manifest20260517_extinit_blend_audit_20260517T1208Z.json`
- Best tested weight `0.01`
- Local lift `+0.000028672`
- Leave-one-site p_lift_gt_0 `1.0`, min lift `+0.000001733`
- Site-bootstrap p_lift_gt_0 `0.78`, q05 `-0.000050795`

Conclusion: B0 refreshed-q3 extinit is a safer but very small local addition. It is not strong enough to override the sidecar under-transfer lesson, but it is a reasonable low-weight fallback candidate for a future reset if no stronger frame/local or source-backed candidate is available. Preferred next work remains true frame/local SED-MIL target structure or fold/seed replication before spending a slot.

## 2026-05-17 execution update: local-window SED/MIL target diagnostic

To move beyond plain row-level distillation, `scripts/birdclef_pseudolabel_student_train.py` now supports local-window target transforms:

- `temporal_target_mode`: `center`, `local_max`, `local_mean`, `center_localmax_mix`
- `temporal_neighbor_radius`
- `temporal_center_weight`

The default remains `center`, preserving historical configs. Non-center modes are intended for weak SED/MIL-style diagnostics where a 10s context window learns from neighboring 5s teacher rows.

Smoke:

- Config: `configs/birdclef/pl_public946_sed85_rankblend15_b0_centerlocalmax_r1_10s_m160_lr3e4_ep8_smoke_20260517.json`
- B0, refreshed-q3 B0 init, 10s/160mel, center/localmax mix radius 1, center weight 0.5
- `256` rows, `8` epochs
- val AUC `0.953302` over `28` classes
- final-all AUC `0.947316` over `42`
- student/teacher corr `0.871117`
- runtime `7.431s`

Full-row:

- Config: `configs/birdclef/pl_public946_sed85_rankblend15_b0_centerlocalmax_r1_10s_m160_lr3e4_ep20_20260517.json`
- Output: `artifacts/pseudolabels/students/pl-public946-sed85-rankblend15-b0-centerlocalmax-r1-10s-m160-lr3e4-ep20-20260517/`
- final-all student AUC `0.990436` over `75`
- student/teacher corr `0.967644`
- runtime `26.702s`, TorchScript `15.391 MB`

Blend/stability audit:

- Output: `artifacts/pseudolabels/audits/public946_centerlocalmax_r1_10s_b0_blend_audit_20260517T1310Z.json`
- Best tested weight `0.01`
- Local lift `+0.000020186`
- Site-bootstrap p_lift_gt_0 `0.66`, q05 `-0.000073194`
- Leave-one-site p_lift_gt_0 `0.7778`; worst sites `S22=-0.000016085`, `S09=-0.000004629`

Conclusion: the local-window target mechanism is useful and passed smoke, but this exact center/localmax-r1 B0 candidate is not robust enough for a Kaggle slot. Next variants should alter the target structure rather than submit this one: e.g. lower neighbor influence (`temporal_center_weight=0.75`), `local_mean`, or a true frame-head model.

## 2026-05-17 execution update: queued local-window variants

Two follow-up local-window target variants are prepared but not yet evaluated:

- `configs/birdclef/pl_public946_sed85_rankblend15_b0_centerlocalmax_r1_cw075_10s_m160_lr3e4_ep8_smoke_20260517.json`
  - Same as the prior center/localmax-r1 smoke, but weaker neighbor influence: `temporal_center_weight=0.75`.
- `configs/birdclef/pl_public946_sed85_rankblend15_b0_localmean_r1_10s_m160_lr3e4_ep8_smoke_20260517.json`
  - Uses `temporal_target_mode=local_mean`, radius `1`.

Both use B0, refreshed-q3 B0 init, 10s/160mel, 256 rows, and 8 epochs. They were intentionally not scored: both GPUs were occupied by unrelated LRM P30 work, and CPU-only execution was too slow/heavy. Partial cw075 output reached only epoch 4 and is not evidence. Run these on GPU before making any decision.

## 2026-05-17 target-transform audit while GPU monitor waits

The queued local-window GPU smokes are still waiting for a free GPU. A CPU-light target distribution audit was run instead:

- Output: `artifacts/pseudolabels/audits/local_window_target_transform_summary_20260517T1555Z.json`
- Center baseline: mean `0.08917`, row-top mean `0.80162`, `>=0.95` cells `280`
- Center/localmax r1 cw0.50: mean `0.09396`, row-top mean `0.81324`, `>=0.95` cells `317`, mean abs delta `0.00479`
- Center/localmax r1 cw0.75: mean `0.09156`, row-top mean `0.80716`, `>=0.95` cells `296`, mean abs delta `0.00239`
- Local mean r1: mean `0.08918`, row-top mean `0.79646`, `>=0.95` cells `255`, mean abs delta `0.00644`

Interpretation: cw0.75 is a gentler version of the previously fragile cw0.50 local-max target; local_mean is a smoothing/control target that reduces extreme confidence. Both remain worth GPU smoke-testing, but no decision should be made from distribution stats alone.

## 2026-05-17 execution update: cw0.75 local-window full diagnostic

The queued GPU smokes completed after the GPU freed up:

- `center_localmax_mix` cw0.75 smoke: best val AUC `0.926609` over `29`, final-all AUC `0.948212` over `42`, corr `0.863944`, runtime `5.709s`.
- `local_mean` radius1 smoke: best val AUC `0.951101` over `35`, but final-all AUC only `0.924528` over `42`, corr `0.809669`; kill this exact local-mean control.

Scaled cw0.75:

- Config: `configs/birdclef/pl_public946_sed85_rankblend15_b0_centerlocalmax_r1_cw075_10s_m160_lr3e4_ep20_20260517.json`
- Output: `artifacts/pseudolabels/students/pl-public946-sed85-rankblend15-b0-centerlocalmax-r1-cw075-10s-m160-lr3e4-ep20-20260517/`
- `792` rows
- best epoch `15`
- best val AUC `0.992308` over `61`
- final-all student AUC `0.991336` over `75`
- student/teacher corr `0.964665`
- runtime `30.460s`, TorchScript `15.391 MB`

Blend/stability audit:

- Output: `artifacts/pseudolabels/audits/public946_centerlocalmax_r1_cw075_10s_b0_blend_audit_20260517T2158Z.json`
- Best tested weight `0.0025`
- Local lift `+0.000015339`
- Site-bootstrap p_lift_gt_0 `0.78`, q05 `-0.000028233`
- Leave-one-site p_lift_gt_0 `0.8889`, min lift `-0.000002217` on `S09`

Conclusion: cw0.75 is gentler and slightly more leave-one-site stable than cw0.50, but the additive lift is smaller and still bootstrap-fragile. It is a low-weight fallback candidate only; prefer stronger true frame/head or new-source signals for the next UTC reset.
