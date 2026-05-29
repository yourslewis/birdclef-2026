# Ranked queue — 2026-05-29 04:25 UTC

## Live status
- Public LB best: `0.949`; v616 and v634 tied baseline to beat.
- 2026-05-29 UTC slots used: `0/5` at start of this run.
- Time to reset at run start: about 19.7h; early-day policy active.
- Active jobs: no local/trainer BirdCLEF jobs before launch; trainer GPU1 free.

## Current run result
- Trained/evaluated `soundscape-native-b0-soft1279init-calibnone-losite-allcls-ep4-20260529` as a calibration-focused soft1279 adaptation data point.
- LOSO row AUC `0.585879`; no-train `0.561323`; non-Aves `0.534311`; file-MIL `0.526705`.
- Package best `soft1279init_calibnone_native_allcls_w0p12` local AUC `0.992844`, lift vs v616 `-0.000637`. Rejected/no submission.

## Ranked next actions
1. **Upgrade no-call/background negatives** — highest information value. Current weak background labels are site-skewed; build stricter/hand-verifiable negative protocol, then rerun suppression sidecar. Expected LB potential: medium; evidence value: high.
2. **Soft1279 head-loaded sidecar stability/class diagnosis** — only recent branch with positive local lift vs v616 (`w0.16` +0.002064). Diagnose per-site/per-class failure before any late-day exploratory slot. Expected LB potential: medium; submission grade: no.
3. **Train-soundscape sequence/file/site mining with different acoustic encoder or stronger file objective** — PANNs localmax remains the best sequence/file clue, but direct sidecars are below v616. Expected LB potential: medium-low; data-point value: high.
4. **Non-Aves/no-train specialist with stricter leave-site/site-pair gates** — still strategically aligned with under-mined soundscapes; avoid direct OOF sidecars unless wrapper passes v616 audit. Expected LB potential: medium-low; data-point value: high.
5. **Late-day public/source slot fill** — only after <3h to reset if no verifier-grade candidate exists; source must pass schema/runtime/dedup guards.

## Critic/verifier decision
- Critic: calibration-none ablation was worth measuring but lower priority now; repeated soft1279 fine-tune knobs are producing negative evidence.
- Verifier: no slot. Candidate is finite/nonconstant and packageable, but below v616 local proxy and promotion gates fail.
