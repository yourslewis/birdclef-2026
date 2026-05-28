# Ranked Queue — BirdCLEF ClawTeam Hill-Climb — 2026-05-28 22:18 UTC

## Live state

- Public best: `0.949` (v616/v621-v623 tied baseline).
- UTC slots: started `0/5`, ended `5/5`; latest v631-v635 pending.
- Active jobs: no local/trainer BirdCLEF jobs at start; source submissions are now Kaggle-side pending.

## Actions / ranked queue result

1. **Late-day source-code slot fill accepted and executed** — v631-v635 submitted after guards; scores pending.
2. **Repo-owned soft1279-init w0.16 sidecar held** — best local sidecar but not hidden-safe/submission-grade; static CSV route rejected by verifier.
3. **No-call suppression sidecar remains next repo-owned branch** — aggregate gate is comparison-grade but needs conservative suppression verifier and stronger negative labeling.
4. **After reset:** if v631-v635 do not improve >0.949, resume soundscape-native/no-call verifier work rather than more direct OOF sidecars.

## Compact performance table for this run

```text
exp   branch family                  dry rows  status       baseline delta
v631  Two-pass SSM public source     240      pending      vs 0.949 pending
v632  Vyanktesh public source        240      pending      vs 0.949 pending
v633  Raunak multi-model public sour 3        pending      vs 0.949 pending
v634  MeenalSinha improved public so 240      pending      vs 0.949 pending
v635  Mattia 943 blend public source 240      pending      vs 0.949 pending
```

## Critic / verifier

- Critic: source-code reruns have lower strategic novelty than repo-owned model sidecars, but at <2h to reset they are the highest valid slot-use option because Kaggle reruns hidden-test code.
- Verifier: all submitted candidates passed source/status/schema/finite/nonconstant/dedup/description/cap guards; no static public CSV or duplicate matrix was submitted.