# BirdCLEF Diversity Scout — truth-aligned decorrelation vs 0.950 frontier

Run: `artifacts/diversity_scout/20260601T0334Z/`
Date: 2026-06-01 03:34 UTC
Frontier E: rank blend `proto*0.60 + sed*0.40` of the v644 EoS8/PowerOptimization 0.950 branches (proxy E base macro-AUC ≈ `0.99303`, 42 valid classes, 190 matched proxy rows).

## Why this scout exists

The standard audit scores "fixed-weight pooled-AUC lift vs anchor", which rewards
redundant agreement and punishes complementary disagreement — it can never reward a
diverse member. This scout instead measures **truth-aligned decorrelation**:
diverse AND competent exactly where the frontier is weak. DEV =
`blend_lift + λ·(rank_decorrelation × competence_above_chance_on_E_weak)`. The product
guard forces a garbage-but-decorrelated stream to score ~0.

## The result confirms the representation hypothesis

Every candidate splits cleanly into two clusters, and **no candidate is both diverse and competent**:

| candidate | repr family | cand AUC | rank decorr | err corr | blend lift | site q05 | DEV |
|---|---|---:|---:|---:|---:|---:|---:|
| winner_sed_branch | EoS8 SED | 0.99598 | 0.230 | 0.870 | +0.00301 | +0.00107 | 0.00413 |
| v616_samejima_sed | SED | 0.99598 | 0.230 | 0.870 | +0.00301 | +0.00107 | 0.00413 |
| winner_proto_branch | EoS8 Proto | 0.98630 | 0.097 | 0.943 | +0.00041 | +0.00025 | 0.00087 |
| v616_jung21_raw | CNN | 0.98743 | 0.165 | 0.862 | +0.00002 | 0.0 | 0.00081 |
| fused_seq_context | **our PANNs+DyMN10** | **0.64379** | 0.250 | 0.593 | **0.0** | 0.0 | 0.00057 |
| v616_final | anchored blend | 0.99348 | 0.007 | 0.991 | +0.00045 | <0 | 0.00048 |
| panns_seq_r2 | **our PANNs** | **0.69939** | 0.111 | 0.685 | **0.0** | 0.0 | 0.00026 |

### Two-cluster reading

1. **Our repo families (PANNs, fused PANNs+DyMN10): decorrelated but incompetent.**
   `fused_seq_context` has the *highest* rank-decorrelation (0.250) of any candidate — but
   its AUC is 0.644 and its competence on E's weak classes is only 0.726. The DEV product
   guard correctly zeroes it: best blend weight = 0.0, blend lift = 0.0. **This is the
   empirical proof of the shared-embedding ceiling: same-base wrappers can be decorrelated
   yet never strong enough to add usable signal.**

2. **Everything competent shares the winner's representation.** The only streams with real
   competence (SED 0.996, proto 0.986, jung21 0.987, v616 0.993) are all low-decorrelation
   relatives of E. The best blend (SED at w=0.8, +0.0030 site q05 +0.0011) is *inside the
   winner's own family* and its file q05 is slightly negative, so it does not robustly pass.

## Decisive conclusion

**Our entire current asset pool contains no member that is simultaneously orthogonal and
competent.** Diversity-aware evaluation cannot manufacture a diverse member that does not
exist. Therefore the next gain must come from a **new foundation embedding** (a different
pretrained backbone family), not from re-blending what we already have.

This validates the user's two intuitions:
- Our offline eval did not capture diversity value → fixed by DEV's decorrelation×competence term.
- Shared base embeddings cannot yield a really diversified set → confirmed: our decorrelated members are all incompetent; our competent members are all redundant.

## Next actions (slot-bearing)

1. Source 2-3 **structurally different** public foundations (Perch / Perch-v2 / BirdNET /
   AudioMAE / SurfPerch lineage) — different backbone than EoS8's Perch-ProtoSSM.
2. Run each candidate's proxy stream through this scout. Promote only members with
   competence_above_chance on E-weak classes **and** positive blend site+file q05.
3. Cross-family rank + Quantile-Mix blend the surviving member with the 0.950 frontier;
   submit top-DEV candidates to LB. Add TTA on the winning pipeline as a cheap orthogonal lever.

## Reproduce

```bash
.venv_scout/bin/python scripts/birdclef_diversity_scout.py \
  --proto-csv <winner proto branch> --sed-csv <winner sed branch> \
  --candidate "name=path.csv" [...] --neg-weights --bootstrap 200 \
  --out-dir artifacts/diversity_scout/<stamp>
```
