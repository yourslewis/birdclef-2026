# BirdCLEF Model Data-Point Ledger

This directory records bounded training/data-point runs for distinct model families.

Each entry should capture:
- timestamp
- branch/config/script
- model family and initialization
- data subset / row count / class count
- targets and loss
- window/features/augmentations
- epochs/runtime/compute
- CV/proxy metrics
- prediction artifact paths
- correlation or blend audit vs anchor/v616 when available
- export/runtime status
- critic/verifier notes
- submit decision or why not

Purpose: create a measured search landscape for hill climbing, not only submit-ready candidates.
