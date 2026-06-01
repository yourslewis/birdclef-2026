# BirdCLEF hill-climb ranked queue — 2026-06-01 22:22 UTC

## Live status
- UTC slots: **5/5 used**; reset in ~1.63h.
- Public best remains **0.950** from v644/v647. v616 remains only a secondary baseline at ~0.949.
- Active BirdCLEF jobs: none locally/trainer. Trainer GPU1 was used for this run's MobileViT-v2 data point; GPU0 is occupied by unrelated HSTU/LRM work.
- Pending submissions: `v657` and `v658` still pending. `v659`/`v660` scored and are below frontier.

## Latest Kaggle submissions
- `v657` ref `53267235` `jungchanryu/birdclef-submission2`: **pending**; BirdNET+sidecar public source, hash `01c39b92aa147bb4`.
- `v658` ref `53267236` `pilkwang/birdclef-2026-eos-oof-gated-pcen`: **pending**; PCEN public source, hash `abad620908ef0d89`.
- `v659` ref `53267249` `alexycactus/birdclef-2026-cnn-inference-regnety`: **0.860**, complete; delta −0.090 vs 0.950 E / −0.089 vs v616 → reject.
- `v660` ref `53267251` `kruzzcc/bc26-convnext-v3r3-active-a03`: **0.946**, complete; delta −0.004 vs 0.950 E / −0.003 vs v616 → reject below frontier, keep data point.
- `v656` ref `53264588` distinct-backbone distill standalone: **0.529**, complete; lane-closing standalone result.

## New representation-level data point this run
**MobileViT-v2-050 soft1279-teacher distill** (`artifacts/model_data_point_ledger/20260601T2222Z_mobilevitv2_050_distill_dev_gate.md`)
- Train/val: 1,410 train_soundscape OOF windows; 66 files; 9 sites; 234 labels; leave-site OOF.
- Row AUC 0.653518; file-MIL 0.720675; no-train row 0.609974; non-Aves row 0.654489; pooled row 0.665982 / 71 valid.
- DEV: cand_auc 0.750924; weak-class AUC 0.714601; rank_decorrelation 0.642349; blend w 0.02; lift +0.0000706; site_q05 −0.000372; file_q05 −0.000315; DEV 0.001449; gate_pass=false.
- Decision: **DEMOTE**. It is structurally different and moderately competent, but not robustly additive to frontier E; fails both promotion and non-harmful gates.

## Ranked next actions
1. **Poll v657/v658** after scoring and ledger deltas vs 0.950 E and v616 immediately. If either ties/improves, inspect source lineage for a reusable representation/data-source lever; otherwise demote.
2. **Do not spend reset slots on another standalone distill CNN.** v656 (0.529), public RegNetY (0.860), public ConvNeXt (0.946), and owned MobileViT DEV fail now give enough evidence that distinct-backbone standalone competence does not transfer reliably to hidden LB/frontier blend.
3. If a next-reset slot is needed, prefer a **source-clean, verifier-grade public 0.950-adjacent candidate with genuine front-end/data processing difference** and full dry-run preflight, not a STOP-rule head/power variant.
4. Continue foundation scout, but current accessible landscape remains exhausted: BirdNET global/scoped/no-location demoted; SurfPerch/AudioMAE absent; Perch-v2 redundant; EfficientAT/DyMN only diligence.
5. Medium-term: only revisit distill family as a tiny in-kernel blend with the actual 0.950 E pipeline if a credible runtime package can emit both streams; proxy/global weight evidence is weak, so EV is low.

## STOP-rule status
The STOP rule was respected: no pure PANNs/B0/DyMN/no-call-gate/power/head-knob submission or training was done. The only new training was representation-level (MobileViT-v2 front-end).
