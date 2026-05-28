# Ranked Queue — BirdCLEF ClawTeam Hill-Climb — 2026-05-28 12:28 UTC

## Live state

- Public best: `0.949` (v616/v621-v623 family still tied best; v626-v630 below best).
- UTC slots used today at run start: `0/5`; ~11.7h to reset.
- Slot policy decision: early/mid UTC day, no submission without verifier-grade or high-info nonduplicate candidate.

## Result of this run

Evaluated/package-audited the strong `sed-b0-oofteacher-b0v26-nfnetv29-soft-1279-ep4-20260528` model as a soundscape sidecar. Best low-weight recipe `soft1279_w0p005` scored local AUC `0.990644` / 42 valid, lift vs anchor `+0.000253`, lift vs v616 `-0.002837`. Rejected for submission.

## Ranked next actions

1. **Trusted no-call/background label protocol (highest information value)** — Build/evaluate an explicit no-call/any-call protocol on soundscapes/unlabeled rows before more negative-aux training. Current direct soft OOF-teacher package has poor raw transfer and only tiny anchor lift.
2. **Soundscape-native calibration/domain adaptation for soft1279** — If revisiting OOF-teacher SED, adapt/calibrate on labeled train_soundscapes with leave-site/file gates instead of direct train-audio teacher output.
3. **PANNs/localmax hidden-test-operable wrapper** — Still the best direct train_soundscape sequence clue by v616 sidecar lift (`-0.001728`) but needs a true hidden-test row package, not OOF proxy leakage.
4. **Broader non-Aves/no-train specialist with strict site gates** — Focus on S08/S15/S19/S23 sonotypes and S22/S18 frogs; reject if only site-prior lift.
5. **Late-day slot fill (<3h to reset only)** — Use source-clean, nonduplicate public/kernel candidates if no verifier-grade internal candidate exists.

## Critic / verifier decision

- Critic: strong random-split train-audio OOF score is not sufficient evidence for soundscape hidden transfer.
- Verifier: sidecar is finite/nonconstant, fully row/column aligned, and package inference works; promotion gates fail and `submit_approved=false`.
