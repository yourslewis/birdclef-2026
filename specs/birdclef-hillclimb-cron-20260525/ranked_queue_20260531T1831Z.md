# BirdCLEF hill-climb ranked queue — 2026-05-31 18:31 UTC

## Live status
- Public best before this run: `0.950`, tied by `v644`/`v647` EoS8/PowerOptimization public-source submissions.
- UTC slots at start: 0/5 used; ~5.5h to reset.
- Actions this run: pushed/verifier-ran two repo-owned private source forks and submitted both after schema/runtime preflight.
  - `v652` ref `53228552`: EoS8 PowerOpt `proto=0.40`, `sed=0.60`, public LB pending.
  - `v651` ref `53228555`: EoS8 PowerOpt `proto=0.20`, `sed=0.80`, public LB pending.
- Remaining slots after submissions: 3/5 estimated; hold until pending scores resolve or late-day fill window.

## Ranked queue after this run

1. **Score readout + v651/v652 response plan** — highest immediate value.
   - If either beats/ties `0.950`, audit output lineage, compare hidden sensitivity, and build one follow-up midpoint/neighbor fork (`proto=0.30` or outer PowerOpt-only) only after dedup/source verifier.
   - If both underperform, demote SED-heavy hidden forks and retain source-SED clue as proxy-overfit diagnostic.

2. **Train-soundscape sequence/file/site mining branch** — best distinct-model next step.
   - Build a new sequence-aware MIL/temporal-pooling/file-context branch over full train_soundscapes, with leave-site/file gates and explicit file/site diagnostics.
   - Prioritize a model that mines sequences/files/sites rather than isolated rows.

3. **Deeper soundscape-native training variant**.
   - Fine-tune more than a shallow head (last blocks/adapters or compact CNN/SED) on task-aligned soundscape/OOF-teacher targets.
   - Gate with leave-site/file metrics and no-call/non-Aves slices.

4. **Source-winner SED local sidecar follow-up**.
   - The local `v616 + source_sed` rank grid remains the strongest proxy metric (`0.996059`, +0.002578 vs v616) but lacks hidden-safe packaging except via source fork; use only after v651/v652 score evidence.

5. **No-call / non-Aves specialist**.
   - Keep as landscape/rare-slice datapoints; no current robust slot candidate.

6. **20s temporal/localmax / G124 target-shape ablations**.
   - Useful diagnostics but currently below v616 local proxy; continue only if next train branch needs them as components.

## Critic / verifier notes
- v652 was submitted before v651 because it is lower-movement and still locally positive.
- v651 was submitted as high-information because it had the best local reconstructed PowerOpt AUC (+0.001729 vs v616) and slots were unused.
- Both source forks are hidden-test capable, public-session `COMPLETE`, finite/nonconstant, and not static/sample/fallback-only submissions.
