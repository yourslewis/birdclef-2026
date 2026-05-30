# Soft1279 head-loaded sidecar movement diagnosis — 2026-05-30 14:20 UTC
## Scope
- Evaluated strongest recent sidecar: `soft1279init_native_allcls_w0p16` (anchor rank blend + soft1279-head-loaded native sidecar).
- Matched `190` proxy rows / `20` files / `6` sites / `42` valid classes.
- Evidence level: comparison-grade; no Kaggle submission.

## Comparable metrics
- Anchor local AUC: `0.990391`
- v616 local AUC: `0.993481`
- Raw soft1279 sidecar AUC: `0.994941` (raw lift vs v616 `+0.001461`)
- Candidate w0.16 AUC: `0.995545`; lift vs v616 `+0.002064`; lift vs anchor `+0.005155`

## Site attribution vs v616
| site | rows | valid cls | v616 AUC | cand AUC | lift | pos Δ |
|---|---:|---:|---:|---:|---:|---:|
| S18 | 15 | 4 | 0.905032 | 0.900032 | -0.005000 | +0.001325 |
| S09 | 19 | 5 | 1.000000 | 1.000000 | +0.000000 | -0.002933 |
| S13 | 24 | 6 | 0.992323 | 0.993075 | +0.000752 | -0.002189 |
| S08 | 60 | 19 | 0.985618 | 0.987298 | +0.001680 | -0.000147 |
| S03 | 24 | 4 | 0.962338 | 0.973701 | +0.011364 | +0.001242 |
| S22 | 48 | 15 | 0.963022 | 0.975668 | +0.012645 | -0.000068 |

## Class/taxon attribution
| group | classes | positives | mean lift | p(lift>0) | min | max |
|---|---:|---:|---:|---:|---:|---:|
| Mammalia | 3 | 22 | -0.001277 | 0.00 | -0.002660 | +0.000000 |
| no_train_primary | 19 | 376 | +0.000190 | 0.26 | +0.000000 | +0.001424 |
| Amphibia | 8 | 169 | +0.002784 | 0.38 | -0.001812 | +0.022703 |
| Reptilia | 1 | 10 | +0.004722 | 1.00 | +0.004722 | +0.004722 |
| Aves | 11 | 93 | +0.005449 | 0.64 | -0.000869 | +0.037234 |

### Biggest class wins vs v616
| class | taxon | positives | sites | lift | pos Δ |
|---|---|---:|---|---:|---:|
| strher2 | Aves | 2 | S22 | +0.037234 | +0.009750 |
| 555146 | Amphibia | 5 | S18,S22 | +0.022703 | +0.006867 |
| undtin1 | Aves | 5 | S22 | +0.011892 | +0.009867 |
| bunibi1 | Aves | 2 | S22 | +0.005319 | +0.005833 |
| 116570 | Reptilia | 10 | S08 | +0.004722 | -0.003567 |
| chacha1 | Aves | 22 | S08,S09,S22 | +0.003247 | +0.000235 |
| trsowl | Aves | 13 | S22 | +0.002608 | -0.000154 |
| 47158son17 | Insecta | 43 | S08 | +0.001424 | +0.000422 |

### Biggest class losses vs v616
| class | taxon | positives | sites | lift | pos Δ |
|---|---|---:|---|---:|---:|
| 74113 | Mammalia | 2 | S22 | -0.002660 | +0.001667 |
| 23158 | Amphibia | 6 | S13 | -0.001812 | +0.000778 |
| 47144 | Mammalia | 7 | S22 | -0.001171 | +0.000429 |
| compau | Aves | 13 | S22 | -0.000869 | -0.000590 |
| 22967 | Amphibia | 47 | S03,S18,S22 | -0.000298 | -0.000170 |
| 65380 | Amphibia | 32 | S13,S22 | -0.000000 | -0.000292 |
| 47158son20 | Insecta | 12 | S08 | +0.000000 | -0.000000 |
| grekis | Aves | 2 | S09 | +0.000000 | -0.005333 |

## Critic / verifier decision
- Critic: global w0.16 is not a robust hidden-behavior bet; it over-moves calibration/rank mass and still depends on narrow proxy sites.
- Verifier: finite/aligned candidate exists, but `submit_approved=false`; early UTC slot policy blocks comparison-grade submission.
- Decision: reject as submission candidate; keep diagnostic. Next best action is robust class/site caps for only stable winners, or curated multi-site no-call negatives.
