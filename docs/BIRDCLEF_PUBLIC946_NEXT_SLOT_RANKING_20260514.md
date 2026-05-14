# BirdCLEF Public946 Next Slot Ranking — 2026-05-14 16:20 UTC

Status: planning while UTC submission cap is exhausted and guarded `v551` monitor is waiting for reset.

## Submission / monitor state

- Current public best remains **0.946**.
- Recent scored submissions: `v549=0.946`, `v548=0.946`, `v547=0.946`, `v546=0.946`, `v545=0.944`.
- Completed candidate kernels with `submission.csv`: `v551`, `v552`, `v553`.
- Guarded submit monitor: `scripts/submit_v551_when_ready.py`, pid `96890`, log `logs/submit_v551_when_ready_restart_20260514T155229Z.log`; it hit daily cap and is sleeping until reset.

## Local gate comparison

| Candidate | Local macro AUC | Corr / MAE vs anchor | Top3 / Top5 | Interpretation |
|---|---:|---:|---:|---|
| Public946 gate retune full-top | `0.993325` | `corr=0.9946`, `MAE=0.0127`, `max_abs=0.353` | `0.647 / 0.753` | Strongest dry-run label overlap, but high overfit risk and large max displacement. Good next implementation candidate only after comparing hidden-risk gates. |
| Task239 Snowflake agreement best | `0.992567` | `corr=0.9882`, `MAE=0.0439`, `max_abs=0.269` | `0.526 / 0.632` | Tiny AUC lift but too much displacement and weak top-k vs reconstructed V8. Research-only unless tightened strongly. |
| `v551` tiny CLAP sidecar `0.005` | `0.992549` | `corr=0.999987`, `MAE=0.00169`, `max_abs=0.00496` | `0.511 / 0.632` | Best low-displacement completed candidate. Keep as next automatic reset submission. |
| Public946 anchor | `0.992525` | baseline | `0.521 / 0.632` | Known 0.946 public LB floor. |
| `v552` ConvNeXt student | best is anchor-only | baseline | baseline | Do not submit automatically. |
| `v553` ConvNeXt + taxon gate | best is anchor-only | baseline | baseline | Hold/no-submit. |
| CV9245 / train-audio-head further solo brackets | tied 0.946 publicly | n/a | n/a | Exhausted for display-score purposes unless combined with a clearly better gate. |

## Recommendation

1. **Keep v551 as the next reset submission.** It is already complete, monitored, low-displacement, and not yet visible in submissions.
2. **Do not auto-submit v552/v553.** Their local gates are negative.
3. If `v551` ties/drops, the next useful implementation should be either:
   - a source-clean **public946 gate-retune candidate** from the full gate sweep, with conservative displacement/rare-taxon safeguards; or
   - a much tighter **Snowflake agreement-gated candidate** that reduces displacement, uses the exact mapped/proxy mask, and is compared against the gate-retune before any Kaggle slot.
4. Avoid another single-family CV9245/BirdNET/CLAP weight sweep unless a leaderboard result changes the evidence.
