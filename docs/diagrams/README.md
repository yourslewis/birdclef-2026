# BirdCLEF Diagrams

Generated for code-reading/debugging. These diagrams explain the Kaggle kernel workflow and model/ensemble architecture without requiring Graphviz, Chrome, Mermaid, or network access.

## Quick start

From the repo root:

```bash
# Regenerate v237 diagrams
python3 docs/diagrams/generate_v237_diagrams.py

# Regenerate v238 diagrams
python3 docs/diagrams/generate_v238_diagrams.py
```

Both generators write PNG, SVG, and standalone HTML outputs in `docs/diagrams/`.

Useful options:

```bash
# Only workflow diagram, SVG + HTML
python3 docs/diagrams/generate_v238_diagrams.py --diagram workflow --formats svg html

# Only architecture diagram, all formats, custom output folder
python3 docs/diagrams/generate_v238_diagrams.py --diagram architecture --outdir /tmp/birdclef-diagrams

# Point at a different script.py variant
python3 docs/diagrams/generate_v238_diagrams.py --script kaggle-kernels/v238-file-context-boost/script.py
```

## Outputs

### v237 — weighted ensemble / mount-fix baseline

- `birdclef_v237_workflow.png/svg/html` — end-to-end Kaggle workflow: mounts, setup, feature extraction, priors, probes, ProtoSSM, ensemble, post-processing, submission.
- `birdclef_v237_architecture.png/svg/html` — model/ensemble architecture: Perch foundation model, metadata prior branch, MLP probe bank, ProtoSSM temporal branch, ensemble and calibration head.

### v238 — stronger file-context boost experiment

- `birdclef_v238_workflow.png/svg/html` — same overall workflow as v237, with the v238 script/metadata and file-context-boost experiment context.
- `birdclef_v238_architecture.png/svg/html` — same model stack as v237, highlighting the calibrated ensemble/post-processing path used by v238.

## Tools

- `generate_v237_diagrams.py` parses constants from `kaggle-kernels/v237-weighted-ensemble/script.py`.
- `generate_v238_diagrams.py` parses constants from `kaggle-kernels/v238-file-context-boost/script.py`.
- Both generators use Pillow for PNG and write SVG/HTML directly.
