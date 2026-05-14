# BirdCLEF 2026 Next After CV9245 Bracket — 2026-05-14

Status: active planning after UTC daily quota was exhausted by `v545`-`v549`.

## Current public state

- Locked public best remains **0.946**.
- `v545` CLAP 5% dropped to `0.944`; stop CLAP-only follow-ups.
- `v546` train-audio-head 5% tied `0.946`.
- `v547` CV9245 2% tied `0.946`.
- `v548` CV9245 0.5% tied `0.946`.
- `v549` CV9245 1% is submitted and pending at this planning point.
- UTC submissions used: `5/5`.

## Interpretation

The public946 anchor is robust. BirdNET, train-audio-head, and CV9245 all appear safe at small weights, but none has moved the displayed public score yet. That means the next useful work should avoid another single-family weight bracket unless `v549` unexpectedly improves or hidden rank evidence suggests a clear reason.

## Next-day priority order

1. **Wait for v549 score.**
   - If `v549 > 0.946`: bracket narrowly around CV9245 1% only if a leaderboard/tie-break rank move is visible.
   - If `v549 = 0.946`: treat CV9245 as exhausted for display-score purposes; keep it as a hidden-diversity ingredient, not another solo sweep.
   - If `v549 < 0.946`: stop CV9245 public slots.

2. **Snowflake SED sidecar dry-run (preferred next new signal).**
   - Source audit says `tsubasatech/birdclef-2026-snowflake-sed` is public/attachable with ConvNeXt-Tiny + EfficientNetV2-M SED ONNX.
   - Do not submit blindly. Build a source-clean dry-run that writes `submission_snowflake_sed.csv`, then run `scripts/birdclef_public946_sidecar_weight_grid.py` against v542.
   - Candidate weights: `0.005`, `0.01`, `0.02` first.

3. **Combined low-weight hidden-diversity ensemble.**
   - Only after all component outputs are available.
   - Test public946 + CV9245 + train-audio-head with tiny weights, e.g. head `0.01` + CV9245 `0.005/0.01`.
   - Submit only if displacement is tiny and top-k/hidden-rank rationale is stronger than the individual ties.

4. **Public946 final-gate retune fallback.**
   - Low runtime, but high overfit risk to train-soundscape dry-run labels.
   - Use only if no distinct sidecar passes gates.

## Stop rules

- No more CLAP-only public slots unless a new CLAP source/model is introduced.
- No more BirdNET-only public slots; 10% and 5% tied but did not improve.
- No more CV9245-only weight bracketing if `v549` also ties/drops.
- No old 0.930-axis submissions unless they are a tiny sidecar into public946 with strong evidence.
