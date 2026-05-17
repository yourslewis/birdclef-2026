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
