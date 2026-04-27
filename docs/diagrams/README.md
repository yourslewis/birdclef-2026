# BirdCLEF v237 Diagrams

Generated for code-reading/debugging.

## Quick start

From the repo root:

```bash
python3 docs/diagrams/generate_v237_diagrams.py
```

This regenerates PNG, SVG, and standalone HTML outputs in `docs/diagrams/`.

Useful options:

```bash
# Only workflow diagram, SVG + HTML
python3 docs/diagrams/generate_v237_diagrams.py --diagram workflow --formats svg html

# Only architecture diagram, all formats, custom output folder
python3 docs/diagrams/generate_v237_diagrams.py --diagram architecture --outdir /tmp/birdclef-diagrams

# Point at a different script.py variant
python3 docs/diagrams/generate_v237_diagrams.py --script kaggle-kernels/v237-weighted-ensemble/script.py
```

## Outputs

- `birdclef_v237_workflow.png/svg/html` — end-to-end Kaggle workflow: mounts, setup, feature extraction, priors, probes, ProtoSSM, ensemble, post-processing, submission.
- `birdclef_v237_architecture.png/svg/html` — model/ensemble architecture: Perch foundation model, metadata prior branch, MLP probe bank, ProtoSSM temporal branch, ensemble and calibration head.

## Tool

- `generate_v237_diagrams.py` is dependency-light: it uses Pillow for PNG and writes SVG/HTML directly. It does **not** require Graphviz, Chrome, Mermaid, or network access.
- It parses a few constants from `kaggle-kernels/v237-weighted-ensemble/script.py` (PCA dimension, ProtoSSM config, ensemble weight) so labels stay synchronized with the code.
