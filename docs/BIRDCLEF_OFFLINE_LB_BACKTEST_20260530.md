# BirdCLEF Offline Validation → Public LB Backtest — 2026-05-30

## Purpose

Calibrate recorded offline validation signals against actual public leaderboard outcomes. This is a backtest of the validation strategy, not a new submission candidate.

The generated artifact directory is:

```text
artifacts/offline_lb_backtest/latest/
```

Key generated files:

```text
artifacts/offline_lb_backtest/latest/offline_lb_correlation_2d.png
artifacts/offline_lb_backtest/latest/offline_lb_backtest_records.csv
artifacts/offline_lb_backtest/latest/offline_lb_backtest_correlations.json
artifacts/offline_lb_backtest/latest/kaggle_submissions_snapshot.json
```

## Coverage

The script fetched the latest 200 Kaggle submissions and mapped 175 versioned `vNNN` submissions.

Backtest records:

- 6 manually curated local/proxy validation records with durable historical offline evidence.
- 20 late public-source preflight records from `performance_table.jsonl`.
- 26 total records in the report CSV.

Correlation-ready subsets:

- Local/proxy lift vs public delta: `n=5` records (`v560`, `v573`, `v611`, `v612`, `v616`).
- Late public-source dry-run uniqueness vs public delta: `n=15` records (`v621`-`v630`, `v636`-`v640`; `v631`-`v635` lacked uniqueness in JSONL).

## Correlation results

### Local/proxy lift vs public LB delta

```text
n        = 5
Pearson  = 0.8023738463415447
Spearman = 0.8660254037844386
```

Important interpretation: this positive correlation **does not mean offline lift predicted improvements**. None of these local-lift candidates beat the public best. The correlation mostly separates two tiny-lift droppers (`v560`, `v573`) from larger-lift ties (`v611`, `v612`, `v616`). It supports using offline lift as a veto/triage signal, not as an approval oracle.

### Late source dry-run uniqueness vs public LB delta

```text
n        = 15
Pearson  = -0.31794973605451166
Spearman = -0.09519590102890572
```

Dry-run uniqueness/schema preflight is operationally necessary but is not a score proxy.

### Dry-run rows vs public LB delta

```text
n        = 15
Pearson  = -0.6311831991504844
Spearman = -0.5933561447616332
```

This is not actionable as a model-quality metric; it mostly reflects that some late source kernels emitted 240 dry-run rows and still dropped.

## Main conclusion

Offline validation remains useful as a **rejection and triage tool**:

- It can detect broken, duplicate, near-duplicate, unstable, or locally harmful candidates.
- It can separate tiny micro-sidecars that are not worth slots from stronger tied/plateau-preserving sidecars.
- It does **not** reliably identify public LB improvers from the current narrow proxy.

The current strategy should therefore be:

```text
Offline validation = veto / triage / information value
Kaggle LB slots    = final empirical test, especially late-day guarded exploration
```

## How to rerun

```bash
python3 scripts/birdclef_offline_lb_backtest.py --out-dir artifacts/offline_lb_backtest/latest
```

The script writes a 2D scatter plot and machine-readable correlation JSON.
