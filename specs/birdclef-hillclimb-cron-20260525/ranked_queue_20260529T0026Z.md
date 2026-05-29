# Ranked Queue — BirdCLEF ClawTeam Hill-Climb — 2026-05-29 00:26 UTC

## Live state

- Public best: `0.949` (v616/v621-v623/v634 tied baseline; v634 newly confirmed tied, not better).
- Latest scored late fills: v631 `0.926`, v632 `0.940`, v633 `0.946`, v634 `0.949`, v635 `0.941`.
- UTC slots used today: `0/5` at start of the new UTC day; ~23.7h to reset.
- Active jobs: no local/trainer BirdCLEF jobs; trainer GPUs free at start.
- Git branch: `feature/birdclef-20260524-20utc-v612-submit`.

## Actions this run

1. **Updated late-day pending results** — v631-v635 are complete; only v634 tied best, none beat `0.949`.
2. **Built no-call suppression verifier** — added `scripts/birdclef_nocall_suppression_sidecar_audit.py` and ran a bounded 16-candidate no-slot audit.
3. **No submission** — early UTC day and no candidate passed promotion gates.

## Ranked queue after this run

1. **Upgrade no-call/background protocol before any suppression submit** — current aggregate gate is real but weak-label/site-skewed; best suppression lift vs v616 is only `+0.000066`. Need hand-verified/stricter negatives or better background labels before spending a slot.
2. **Train next distinct soundscape-native/domain-adaptation data point** — target a branch with more independent raw movement than a tiny suppression cap. Recommended: regularized soundscape-native model with site-balanced sampling or a compact adapter/last-block update, not another direct OOF sidecar.
3. **Soft1279-init w0.16 sidecar remains best local repo-owned candidate but held** — local AUC `0.995545`, lift vs v616 `+0.002064`, but failed strict gates and is static/proxy-only; do not submit early-day.
4. **Public/source-code queue refresh only for late-day slots** — v634 tied `0.949`, but recent source fills mostly underperformed; do not burn early-day slots on more public reruns unless a clearly stronger source appears.
5. **Longer-context / file-site sequence branch** — still useful as a distinct data point if no-call negative upgrade stalls, but direct PANNs/DyMN10 sidecars have repeatedly failed v616.

## Compact performance table for this run

```text
exp / candidate                         metric                         vs baseline       decision
v631 Maryna two-pass source              public LB 0.926                -0.023 vs 0.949   reject
v632 Vyanktesh source                    public LB 0.940                -0.009 vs 0.949   reject
v633 Raunak multi-model source           public LB 0.946                -0.003 vs 0.949   reject
v634 MeenalSinha improved source         public LB 0.949                +0.000 vs 0.949   tied best
v635 Mattia 943 source                   public LB 0.941                -0.008 vs 0.949   reject
nocall final non-Aves/no-train α0.01     local AUC 0.993546 / 42 valid  +0.000066 vs v616 reject gates
```

## Top comparable local sidecar/package audits

```text
candidate / recipe                    local AUC  lift vs v616   status
soft1279init native w0.16             0.995545   +0.002064      hold; strict gates failed
soft1279init native w0.08             0.994813   +0.001332      hold; strict gates failed
soft1279 obspos package w0.16         0.993906   +0.000425      reject
nocall suppression nonaves α0.01      0.993546   +0.000066      reject; tiny lift
v616 baseline                         0.993481   +0.000000      tied public best
```

## Critic / verifier

- Critic: conservative no-call suppression is the right thing to test, but the result is too small and too dependent on weak, site-skewed background labels. The all-class suppression variants are harmful; only non-Aves/no-train suppression is barely positive.
- Verifier: best candidate finite/nonconstant `240x234`, matched all gate rows and passed basic audit mechanics, but `submit_approved=false`; failed lift-vs-v616, lift-vs-anchor, site/file bootstrap q05 gates, and top-5 recall regressed vs v616.
- Submission decision: **not approved**; early-day slots should stay unused until a verifier-grade/high-info candidate exists.

## Artifacts

- Ledger: `artifacts/model_data_point_ledger/20260529T0026Z_nocall_suppression_sidecar_audit.md`
- Audit summary: `artifacts/soundscape_nocall_suppression/soundscape-nocall-suppression-v616-agg-20260529T0035Z/audit_summary.json`
- Full audit: `artifacts/soundscape_nocall_suppression/soundscape-nocall-suppression-v616-agg-20260529T0035Z/audit/ensemble_strategy_audit.json`
- Script: `scripts/birdclef_nocall_suppression_sidecar_audit.py`

## Next exact action

Train the next distinct soundscape-native/domain-adaptation data point with a clearer hidden-behavior hypothesis, or first hand-audit/upgrade no-call negatives if suppression remains the chosen lane.
