# Validation & Metrics Report — BirdCLEF Ensemble Strategy

Role: **Validation & Metrics Scientist**  
Date: 2026-05-25  
Decision posture: **strict / veto-capable**. Local train-soundscape lift is a rejection screen, not a submission approval signal.

## Executive decision

The next ensemble strategy must not be promoted from a single positive train-soundscape grid, even if the grid has clean site bootstrap. The evidence from v611/v612/v616 is now decisive: three hidden-safe, repo-owned sidecar blends passed local gates and all tied the public LB at `0.949`.

A future candidate can earn a **no-slot private verifier** with promising local/diversity evidence. It can earn a **Kaggle submission** only with:

1. hidden-test-safe rerunnable branch generation;
2. strict schema/runtime/output checks;
3. stronger group validation than v616;
4. evidence of new signal beyond a near-duplicate plateau-family perturbation;
5. explicit Coordinator + Verifier approval.

## 1. Postmortem: why the previous local gates were insufficient

### Observed failure pattern

| Candidate | Local / private evidence | Public result | Lesson |
|---|---:|---:|---|
| v611 Samejima anchor + Praxel HGNet | Anchor AUC `0.9903905`; final AUC `0.9935681` on 190 rows / 42 classes; raw sidecar low rank corr vs anchor | `0.949` tie | A real hidden-safe low-corr sidecar can preserve the anchor without lifting hidden LB. |
| v612 Samejima anchor + HGNet-v57 PT | Raw sidecar strong on local subset (`0.96489`); final AUC `0.94089` vs anchor `0.93315` on 190 rows / 11 classes | `0.949` tie | Strong fold/checkpoint/local raw sidecar evidence is not enough, especially with few valid classes. |
| v616 Samejima anchor + Jung21 + SED | Final AUC `0.9934807`, lift `+0.0030902`; site bootstrap q05 `+0.0017568`; leave-one-site all positive | `0.949` tie | Even the cleanest local/site-stable sidecar gate so far did not transfer. |
| v616 + SYD52p near-duplicate | AUC `0.9935006`, only `+0.000020` over v616 recipe | rejected no-submit | Incremental local gains over a just-tied recipe are not actionable. |
| v610 Gandharva B3 | clean/train-audio fold AUC around `0.966–0.969` | `0.852` | Clean-audio CV can be badly misaligned with hidden soundscape scoring. |
| earlier public946 sidecars / v560-like gates | grouped bootstrap could be positive locally while public tied/dropped | tied/drop | Bootstrap over the same narrow overlap detects fragility but does not prove hidden lift. |

### Root causes

1. **The labeled overlap is too narrow.** Most sidecar grids use about `190` matched train-soundscape rows and `42` valid classes; v612 had only `11` valid classes. That is a small, biased validation proxy for 234 classes and hidden soundscapes.
2. **Site bootstrap has too few sites.** The current train-soundscape overlap often has only `6` site groups. A positive leave-one-site result is useful, but the confidence interval is still conditioned on a tiny set of sites.
3. **Anchor dominance masks failure.** Low-weight rank blends with corr `>0.999` can look safe and locally positive while mostly preserving a known `0.949` plateau prediction.
4. **Branch diversity is necessary but not sufficient.** v611/v612/v616 raw sidecars had low/moderate correlation to the anchor, yet the final hidden result tied. Low correlation alone does not imply useful hidden ordering.
5. **Local AUC rewards the same public/dry-run family repeatedly.** Samejima/Jungchan/Raunak/SYD branches share lineage and dry-run behavior. The grid can re-optimize on the known overlap without adding hidden-test signal.
6. **Per-class adaptation overfits immediately.** v616 per-class selector had all-row in-sample lift around `+0.0029`, but leave-site CV lift was only `~+0.000002–0.000003`; this is classic selector leakage.
7. **Output validity was over-weighted.** COMPLETE/no-failure, row alignment, finite/nonconstant outputs, and bounded displacement are mandatory operational gates, but they do not prove scoreboard improvement.

## 2. Stricter ensemble promotion protocol

### Stage A — Candidate registration and lineage screen

A candidate must be recorded before private verifier work with:

- raw branch names and artifact paths;
- source/kernel/dataset lineage;
- previous public scores for the same family;
- whether each branch is hidden-safe rerunnable, public-output-only, train/dry-run-only, or blocked/private;
- nearest known tied/dropped recipe to compare against, not just the anchor.

**Immediate veto:** any candidate whose only change is scalar weight, power, temperature, rank exponent, threshold, or per-class tuning on the same branches as a tied/dropped recipe is rejected unless it clears the near-duplicate rules in Section 4.

### Stage B — Hidden-test safety gate

Required before any no-slot private verifier is considered complete:

- No static public dry-run CSVs as hidden-test branch inputs.
- Branches must rerun from mounted public/attachable assets on `test_soundscapes`.
- Row IDs must be derived from `sample_submission.csv` / test audio, not hard-coded.
- Hard fail on missing assets, empty test set, schema drift, nonfinite values, constant branches, or row misalignment.
- Write raw branch outputs, not just final `submission.csv`:
  - `submission_anchor_raw.csv`
  - one `submission_<branch>_raw.csv` per sidecar
  - `submission_before_alignment.csv`
  - final `submission.csv`
- Log explicit branch counts, runtime, source paths, row/column counts, and blend weights.
- Dry-run runtime must leave margin. Any dry-run near v616 scale (`~1088s` including nbconvert) requires an extrapolation note before submission.

### Stage C — Required metrics

All metrics must be computed against both:

1. the **primary anchor**, and
2. the **nearest tied/dropped same-lineage recipe**.

Required output metrics:

- rows/columns/unique row IDs;
- finite/bad cell count;
- nonconstant class columns;
- min/max/mean;
- per-branch and final rank corr vs anchor;
- MAE and max-abs displacement vs anchor;
- corr/MAE vs nearest prior tied recipe;
- local macro AUC, valid class count, matched row count;
- top-k row recall for `k=1,3,5,10`;
- per-class lift distribution;
- taxon/family/order lift distribution where metadata is available;
- worst moved classes and largest rank displacements.

### Stage D — Group validation requirements

Run at least these group checks for any promoted candidate:

- site bootstrap, `>=5000` iterations preferred (`1000` is smoke only);
- file bootstrap, `>=5000` iterations preferred;
- leave-one-site;
- leave-one-file;
- if source families are mixed, leave-one-family/branch ablation where feasible;
- if taxon metadata is available, leave-one-taxon/family summary for high-impact taxa.

Minimum thresholds for **Kaggle submission** from local evidence alone:

- aggregate local lift vs anchor: `>= +0.0060` macro AUC, or at least `2x` the v616 lift if using the same validation proxy;
- aggregate local lift vs nearest tied recipe: `>= +0.0010` macro AUC;
- site bootstrap q05 lift: `>= +0.0030`;
- file bootstrap q05 lift: `>= +0.0015`;
- leave-one-site: all groups positive and min lift `>= +0.0030`;
- leave-one-file: at least `90%` positive groups and q05 `>= +0.0010`;
- valid classes: `>=60` preferred; `<42` is verifier-only unless supported by independent evidence;
- matched rows: `>=190`; less is verifier-only;
- if site group count `<8`, local site validation is explicitly **not approval-grade**.

For candidates with strong independent evidence (e.g. a genuinely new public/source lineage scoring above current best, or an OOF setup not sharing the train-soundscape proxy), the Coordinator may lower the aggregate lift threshold, but not the hidden-safety/output gates.

### Stage E — Displacement and correlation constraints

For final low-weight anchored blends:

- final rank corr vs anchor should usually be in `[0.9970, 0.9997]`;
- final MAE vs anchor target: `0.004–0.020`;
- final max-abs vs anchor target: `<=0.10`, and `<=0.06` for same-family plateau branches;
- if final rank corr `>0.99975` and incremental lift vs nearest tied recipe `<+0.00025`, reject as near-duplicate;
- if final rank corr `<0.995` or MAE `>0.025`, require separate evidence that the sidecar family is hidden-safe and not a public LB dropper;
- raw branch low correlation is positive evidence only when paired with group-stable lift and credible source lineage.

### Stage F — Class/taxon caps

Default caps unless a stronger OOF protocol justifies otherwise:

- total sidecar weight `<=0.10` for anchored rank blends;
- same-family plateau sidecars: total sidecar weight `<=0.08` and per-branch `<=0.04`;
- near-duplicate branch additions after a tie: per-branch `<=0.02`, and only for verifier experiments;
- per-class adaptive weights are **not submission-safe** unless leave-site and leave-file CV both clear thresholds;
- for classes with `<3` positives or `<20` negatives in the validation proxy, keep anchor-only unless independent OOF evidence exists;
- per-class total sidecar cap `<=0.06` for submission candidates (`0.08` is audit-only);
- no more than `25%` of valid classes may receive nonzero adaptive sidecar weights without OOF validation;
- taxon/family mean absolute movement must be reviewed; any taxon with mean movement `>2x` global median or obvious rare-class degradation needs manual approval or cap reduction.

### Stage G — Negative controls / sanity checks

Every serious candidate should include at least two negative controls:

- shuffled sidecar rows or permuted row IDs should not improve;
- inverted/anti-rank sidecar should not improve;
- zero sidecar / anchor-only reconstruction must match the anchor metric;
- branch ablation must show the claimed branch contributes lift beyond the existing tied recipe.

If a negative control improves similarly to the candidate, the validation proxy is contaminated or too weak; no submission.

## 3. Evidence enough for private verifier vs Kaggle submission

### Enough for a no-slot private verifier

A candidate can get a private verifier if it has all of:

- hidden-safe implementation appears feasible from public/attachable sources;
- raw branch is not a static public output file;
- preliminary local lift `>= +0.0010` or clearly novel low-correlation raw signal;
- bounded planned displacement (`corr >=0.997`, MAE roughly `<=0.02`);
- no known direct public result far below plateau unless used only as a tiny sidecar;
- no active pending candidate testing the same branch set.

Private verifiers are cheap relative to Kaggle slots and should be used to validate runtime/schema/raw outputs. Passing a private verifier does **not** imply submission.

### Enough for actual Kaggle submission

A candidate can be submitted only if all are true:

- private verifier COMPLETE/no failure;
- required raw outputs exist and validate;
- hidden-safe logs prove branches reran on current mounted rows;
- no static public-output blending;
- group metrics clear Stage D thresholds or have stronger independent evidence approved by Coordinator;
- displacement/correlation constraints clear Stage E;
- class/taxon caps clear Stage F;
- candidate is not a near-duplicate scalar variant;
- no pending or recently scored same-lineage candidate is unresolved;
- daily slot budget allows it and reserve slots remain;
- independent Verifier signs off.

### Explicit veto examples under this protocol

- v616 + SYD52p-like candidate: veto; only `+0.000020` over v616 recipe after v616 tied.
- v616 per-class selector: veto; all-row in-sample lift does not transfer in leave-site CV.
- another `0.94/0.06` HGNet sidecar with same anchor: veto unless new raw branch/source evidence clears thresholds vs v611/v612, not only vs anchor.
- public-output-only branch blend: private verifier blocked until rerunnable.

## 4. Reject rules for near-duplicate scalar variants

Reject without private verifier if any of these hold:

1. Same branches as a tied/dropped public submission and only scalar weights changed.
2. Same branches and only rank power, temperature, threshold, exponent, smoothing, or taxon scalar changed.
3. Final candidate corr vs prior tied recipe `>=0.9995` and MAE `<=0.005`.
4. Incremental local lift vs prior tied recipe `<+0.0005`.
5. Bootstrap/leave-group metrics are reported only vs the anchor, not vs the prior tied recipe.
6. Candidate improves all-row/in-sample selector metrics but leave-site/file CV lift is `<+0.001` or has negative groups.
7. Candidate adds a branch from the same public clone cluster with malformed/fallback final output and no new hidden-safe source logic.
8. Candidate's only evidence is local train-soundscape lift on `<60` valid classes.
9. Candidate consumes a daily slot while a same-lineage candidate is pending.
10. Candidate cannot explain why the previous same-family tie/drop should not repeat.

Near-duplicate variants can be kept only as **offline audit rows** if they help estimate sensitivity; they should not become `v617`-style submissions.

## 5. Suggested implementation checks / commands for Phase 2 and Verifier

Run these from the canonical repo:

```bash
cd /Users/yourslewis/.openclaw/repos/birdclef-2026
```

Use the trainer venv or another environment with pandas/sklearn available.

### A. Compile/static checks

```bash
python3 -m py_compile \
  scripts/birdclef_public946_multi_sidecar_weight_grid.py \
  scripts/birdclef_per_class_sidecar_selector.py

python3 -m py_compile kaggle-kernels/<candidate>/script.py
python3 -m json.tool kaggle-kernels/<candidate>/kernel-metadata.json >/dev/null
```

### B. Fast multi-sidecar grid

```bash
python3 scripts/birdclef_public946_multi_sidecar_weight_grid.py \
  --base-csv <anchor_raw.csv> \
  --sidecar branch_a=<branch_a_raw.csv> \
  --sidecar branch_b=<branch_b_raw.csv> \
  --weights branch_a=0,0.01,0.02,0.04,0.06 \
  --weights branch_b=0,0.01,0.02,0.04,0.06 \
  --max-total-weight 0.10 \
  --labels-csv <train_soundscapes_labels.csv> \
  --output-json artifacts/anchored_blend_audit/<candidate>_grid_fast.json
```

### C. Site and file stability for the selected recipe

Run exact-weight stability twice, once by site and once by file:

```bash
python3 scripts/birdclef_public946_multi_sidecar_weight_grid.py \
  --base-csv <anchor_raw.csv> \
  --sidecar branch_a=<branch_a_raw.csv> \
  --sidecar branch_b=<branch_b_raw.csv> \
  --weights branch_a=0.04 \
  --weights branch_b=0.04 \
  --max-total-weight 0.10 \
  --labels-csv <train_soundscapes_labels.csv> \
  --bootstrap-iters 5000 \
  --bootstrap-group site \
  --leave-one-group site \
  --holdout-detail 999 \
  --output-json artifacts/anchored_blend_audit/<candidate>_stability_site.json

python3 scripts/birdclef_public946_multi_sidecar_weight_grid.py \
  --base-csv <anchor_raw.csv> \
  --sidecar branch_a=<branch_a_raw.csv> \
  --sidecar branch_b=<branch_b_raw.csv> \
  --weights branch_a=0.04 \
  --weights branch_b=0.04 \
  --max-total-weight 0.10 \
  --labels-csv <train_soundscapes_labels.csv> \
  --bootstrap-iters 5000 \
  --bootstrap-group file \
  --leave-one-group file \
  --holdout-detail 999 \
  --output-json artifacts/anchored_blend_audit/<candidate>_stability_file.json
```

### D. Per-class selector as rejection-only screen

```bash
python3 scripts/birdclef_per_class_sidecar_selector.py \
  --base-csv <anchor_raw.csv> \
  --sidecar branch_a=<branch_a_raw.csv> \
  --sidecar branch_b=<branch_b_raw.csv> \
  --labels-csv <train_soundscapes_labels.csv> \
  --weights 0,0.005,0.01,0.02,0.04,0.06 \
  --max-total-weight 0.06 \
  --group site \
  --min-train-pos 3 \
  --min-train-neg 20 \
  --min-lift 0.001 \
  --output-json artifacts/anchored_blend_audit/<candidate>_per_class_site.json

python3 scripts/birdclef_per_class_sidecar_selector.py \
  --base-csv <anchor_raw.csv> \
  --sidecar branch_a=<branch_a_raw.csv> \
  --sidecar branch_b=<branch_b_raw.csv> \
  --labels-csv <train_soundscapes_labels.csv> \
  --weights 0,0.005,0.01,0.02,0.04,0.06 \
  --max-total-weight 0.06 \
  --group file \
  --min-train-pos 3 \
  --min-train-neg 20 \
  --min-lift 0.001 \
  --output-json artifacts/anchored_blend_audit/<candidate>_per_class_file.json
```

If per-class all-row lift is positive but site/file CV is near zero, reject adaptive weighting.

### E. Output/schema/displacement verifier snippet

```bash
python3 - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd

anchor = pd.read_csv('<anchor_raw.csv>')
cand = pd.read_csv('<candidate_submission.csv>')
assert cand.shape[1] == anchor.shape[1], (cand.shape, anchor.shape)
assert 'row_id' in cand.columns
assert cand['row_id'].is_unique
assert cand['row_id'].tolist() == anchor['row_id'].tolist()
cols = [c for c in cand.columns if c != 'row_id']
arr = cand[cols].to_numpy(float)
base = anchor[cols].to_numpy(float)
assert np.isfinite(arr).all()
assert (arr >= 0).all() and (arr <= 1).all()
nonconst = sum(np.nanstd(arr[:, j]) > 0 for j in range(arr.shape[1]))
rank_cand = pd.DataFrame(arr).rank(axis=0, pct=True).to_numpy()
rank_base = pd.DataFrame(base).rank(axis=0, pct=True).to_numpy()
print({
  'rows': len(cand),
  'cols': cand.shape[1],
  'nonconstant_cols': nonconst,
  'corr_vs_anchor_rank': float(np.corrcoef(rank_base.ravel(), rank_cand.ravel())[0,1]),
  'mae_vs_anchor_rank': float(np.mean(np.abs(rank_base-rank_cand))),
  'max_abs_vs_anchor_rank': float(np.max(np.abs(rank_base-rank_cand))),
})
assert nonconst == len(cols)
PY
```

### F. Verifier submit decision checklist

Before any submit script runs, the Verifier should confirm:

- [ ] candidate has private verifier COMPLETE/no-failure;
- [ ] raw branch outputs exist and are row-aligned;
- [ ] logs prove rerun inference, not copied public CSVs;
- [ ] site + file stability JSONs exist;
- [ ] metrics clear thresholds vs anchor and nearest tied recipe;
- [ ] no near-duplicate rule is triggered;
- [ ] runtime has hidden-test margin;
- [ ] source assets are public/attachable and not 403/private;
- [ ] daily slot budget/reserve policy is satisfied;
- [ ] Coordinator and independent Verifier explicitly approve.

## Bottom line

The validation bar must move from “positive local train-soundscape lift with bounded displacement” to “new hidden-safe signal with robust multi-group evidence and clear improvement over the last tied recipe.” Under this stricter bar, v616-like scalar, per-class, and clone-branch variants are rejected. The next viable submission must either introduce a genuinely new rerunnable branch family or improve the validation proxy itself.
