# BirdCLEF 2026 New Branch Ideation Spec — 2026-05-25

## Goal
Identify genuinely new model/branch families likely to have different hidden-test behavior from the saturated 0.949 plateau, using:
- available local data/slice insights;
- existing repo artifacts and ensemble audit reports;
- public models, public notebooks, papers, and related acoustic/event-detection models.

## Current context
- Best public LB remains 0.949.
- Existing v616-family sidecar blends are not slot-worthy.
- The local proxy is narrow: ~190 matched artifact rows, 20 files, 6 sites, 42 valid classes.
- Train audio is Aves-heavy and 28 submission classes lack train-audio primary labels.
- SED raw is locally useful but already in v616; Jung21 is conditional; per-class tuning overfits.

## Non-goals
- No Kaggle submission.
- No kernel launch unless explicitly approved later.
- No near-duplicate v616/SYD/PCEN/EoS/HGNet scalar branch proposals.

## Phase 1 deliverable
A ranked list of new branch candidates with:
- branch idea and model family;
- why hidden behavior should differ;
- public model/source references;
- data/slice rationale;
- implementation path in this repo;
- hidden-test safety risks;
- validation plan using the ensemble audit harness;
- recommended first smoke experiment.

## Required roles
1. Public Model / External Scout
2. Data-Informed Branch Strategist
3. Implementation Feasibility Engineer
4. Validation Skeptic / Promotion Gate
5. Coordinator synthesis
