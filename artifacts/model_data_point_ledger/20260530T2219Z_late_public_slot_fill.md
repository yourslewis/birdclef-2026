# Late public-source slot fill — 2026-05-30 22:19 UTC

Late-day policy was active (~1.68h to UTC reset). Started at 0/5 UTC slots; submitted 5/5 source-code candidates after guarded preflight. All are Kaggle code submissions (hidden rerun), not static CSV uploads.

## Submitted candidates

| Label | Source | Public LB | Preflight | Kaggle ref | Decision |
|---|---|---:|---|---:|---|
| v641 | `hassanalgizani/bc2026-claude-nina-eos-1-fork` | pending | uniq100 91; hash `07f51c964cdf249e` | 53197129 | submitted/monitor |
| v642 | `sans6262q/bc2026-claude-nina-eos-4-fork` | pending | uniq100 91; hash `fb94c3843d36f980` | 53197131 | submitted/monitor |
| v643 | `shahadaljayzani/bc2026-claude-raunak-v7-fork` | pending | uniq100 92; hash `21d11c1a70aad873` | 53197133 | submitted/monitor |
| v644 | `yaroslavkholmirzayev/0950-replay` | pending | uniq100 95; hash `87e9ac1be51a0b15` | 53197162 | submitted/monitor |
| v647 | `ryutoyoda/birdclef-2026-exp013-eos8-sidecar` | pending | uniq100 95; hash `3db34d13e789ae73` | 53197164 | submitted/monitor |

## Rejected during live preflight

- v645 `nina2025/birdclef-2026-eos-9`: public `submission.csv` malformed/nonfinite 243 rows (`bad_values=56862`, hash `0ee04c918f807616`).
- v646 `anthonytherrien/birdclef-2026-ensemble-0-950`: same malformed/nonfinite 243-row output hash `0ee04c918f807616`.
- v648/v649/v650 similar EoS/fork candidates rejected in dry-run for malformed/nonfinite 243-row public output.

## Critic / verifier

- Critic: these are not verifier-grade repo-owned gains, but under <3h late-day policy with 0/5 slots used, valid source-clean exploratory fills dominate letting slots expire.
- Verifier: source had `test_soundscapes`, `sample_submission`, and `submission.csv` markers; kernels COMPLETE; output finite/nonconstant 235-column CSV; descriptions and dry-run hashes nonduplicate; submitted as source-code hidden reruns.
- No model-training data point was launched after submissions because daily cap reached and trainer remains idle for next UTC reset.
