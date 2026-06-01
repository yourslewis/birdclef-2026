# BirdCLEF Hill-Climb Ranked Queue — 2026-06-01 08:15 UTC

## Live status
- UTC slots: **0/5 used** (reset ~15.8h away). Public best **0.950** (v644/v647). Latest v653/v654/v655 = 0.947/0.949/0.949.
- No active BirdCLEF trainer jobs (GPUs on unrelated HSTU/LRM). Diversity foundation lane exhausted (BirdNET fully closed: geo + scope + no-location all DEMOTE; SurfPerch/AudioMAE absent; Perch-v2 redundant).
- Re-scout this run: no genuinely-new accessible non-Perch foundation kernel/dataset on birdclef-2026 (searched birdclef-2026/birdnet/audiomae/surfperch/aves-embedding). Confirmed nothing new to fork.

## This run's results
1. **TTA alpha-sweep {0.20,0.22,0.25,0.28,0.30}** — alpha=**0.25 confirmed robust DEV optimum**: lift +0.000364, site_q05 +0.0001926, file_q05 +0.0001342, DEV 0.000388, gate_pass=true. Only point with BOTH q05 strongly positive (0.20 site-neg, 0.22 file-neg; 0.28/0.30 pass but weaker). Hardens the TTA candidate.
2. **CRITIC FINDING (high value):** the 0.950 source `yaroslavkholmirzayev/0950-replay` ALREADY runs `tta_shifts=[0,1,-1,2,-2]` (circular embedding-sequence shift through ProtoSSM). Our output-pool TTA proxy therefore PARTIALLY DUPLICATES realized gain on the proto branch. A naive "re-enable tta_shifts" kernel = near-duplicate mechanism → NOT a clean new LB datapoint. The genuine remaining lever is **audio-overlap Perch RE-EXTRACTION** (score time-shifted/overlapping raw 5s windows through Perch, average), which the winner does NOT do.

## Refreshed ranked queue (by DEV potential)
1. **Genuine audio-overlap Perch TTA kernel (NEW build).** Re-extract Perch embeddings on raw windows centered off the canonical grid (e.g. +/-2.5s overlap) and average per segment, then feed the existing PowerOpt engine. Distinct from the winner's circular shift → a true front-end/representation lever, not a head knob, not a duplicate. Risk: ~2x Perch wall-time on hidden test → must verify wall-time safety. DEV alpha=0.25 analog. Top slot candidate once packaged + COMPLETE-verified.
2. **BirdNET embedding-feature head (queue #4 carry).** Use BirdNET penultimate embeddings (not thresholded detections) as features for a learned head on E-weak classes; could decouple record orthogonality from detection-incompetence. BLOCKED: birdnetlib not installed in .venv_scout; needs trainer-side build. No slot.
3. **Representation-changing soundscape-native student (file-cal teacher → deeper student).** Best 72-label data point (row 0.712/file-MIL 0.783); distill into block/adapter fine-tune. In-scope (changes representation). Gate vs E.
4. **Late UTC slot fill** (near reset): highest DEV-ranked valid candidate; audio-overlap TTA preferred. Never a STOP-RULE head variant.

## Decision this run
- **No slot spent.** Early UTC (15.8h to reset); the only proxy-ready candidate (output-pool TTA) was shown to partially duplicate the winner's existing TTA, so submitting it risks a near-duplicate/low-info slot. Conserve slots; build the genuine audio-overlap Perch TTA kernel as the next concrete action (verifier-grade, wall-time-checked) for the preferred slot.
- STOP rule intact: no shared-embedding head variants.

## Artifacts
- Sweep: `artifacts/diversity_scout/tta_alpha_sweep_20260601T0815Z/sweep.json`
- Per-alpha scout runs: `artifacts/diversity_scout/sweep2_a{0.20..0.30}/`
- Performance table rows appended (md + jsonl).
