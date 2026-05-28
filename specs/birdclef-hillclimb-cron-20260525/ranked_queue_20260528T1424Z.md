# Ranked Queue — BirdCLEF ClawTeam Hill-Climb — 2026-05-28 14:24 UTC

## Live state

- Public best: `0.949` (v616/v621-v623 family still tied best; v626-v630 all below best at `0.899/0.928/0.940/0.946/0.917`).
- UTC slots used at run start: `0/5`; ~9.7h to reset.
- Active jobs before run: none locally or on trainer for BirdCLEF.
- Slot policy decision: early/mid UTC day. Use no submission slot unless a verifier-grade/high-info nonduplicate candidate clears gates.

## Result of this run

Trained and packaged a **soft1279-initialized soundscape-native all-class B0** adaptation:

- Training LOSO metrics regressed: row AUC `0.600360`, file-MIL `0.605805`, no-train `0.568181`, non-Aves `0.604565`; vs native B0 all-class q3 init row `-0.035801`, file-MIL `-0.067951`.
- Package/audit signal improved sharply vs the unadapted soft1279 sidecar: best rank blend `soft1279init_native_allcls_w0p08` local AUC `0.994813` / 42 valid, lift vs v616 `+0.001332`, lift vs anchor `+0.004422`.
- Verifier/critic: no submit. The best recipe still failed strict lift-vs-anchor and site-bootstrap q05 promotion gates (`submit_approved=false`), and early/mid-day policy requires a stronger gate pass.

## Ranked next actions

1. **Stability audit for soft1279-init native sidecar** — Highest immediate information value. Inspect site/file bootstrap failures and per-site/per-class contributions for `w0p04/w0p08`; determine whether this is a real hidden-safe calibration clue or proxy overfit. If the only failures are overly strict gate thresholds, reserve for late-day consideration, not early submit.
2. **Conservative calibrated blend grid around native package** — Test smaller/mid weights plus probability/rank calibration (`w0.04-0.12`, clipping/temperature, site-balanced objective) using the same packaging helper. Require site-bootstrap and anchor-lift gates before any submission.
3. **Trusted no-call/background label protocol** — Still needed before more negative-aux training: define any-call/no-call/background labels from soundscape/site/file evidence rather than random train-audio OOF negatives.
4. **PANNs/localmax hidden-test-operable wrapper** — Prior best direct sequence clue by v616 sidecar lift before this run (`-0.001728`), but now deprioritized unless it can be packaged and calibrated like the native sidecar.
5. **Late-day source/sidecar slot fill (<3h to reset only)** — If no gate-clearing candidate appears, compare the soft1279-init native `w0p08` sidecar against source-clean public candidates; only fill if verifier accepts the residual bootstrap risk.

## Critic / verifier decision

- Critic: training LOSO metrics are weaker, so the local proxy win may be calibration/proxy-specific; do not overreact to a single proxy AUC lift.
- Verifier: package path is finite/nonconstant, fully row/column aligned, and hidden-test-operable in principle, but promotion gates failed and `submit_approved=false`; no submission this run.
