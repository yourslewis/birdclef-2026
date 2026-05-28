# BirdCLEF late public source slot fill — 2026-05-28 22:18 UTC

## Live state

- Public best before submissions: `0.949` (v616/v621-v623 tied baseline).
- UTC slot check before fill: `0/5` used; after guarded submits: `5/5` used.
- Time to reset at first submit: ~1.7h, so late-day slot policy was active.
- External action type: Kaggle **source-code submissions** via `create_code_submission`; no static CSV upload.

## Submitted candidates

```text
id    ref       source kernel                                            dry rows  uniq100  hash              status
v631  53130259  joriahmed/bc2026-claude-maryna-twopass-ssm-fork         240      98      9fd71fb24d94ca92 pending
v632  53130260  abdulrahmansu10/bc2026-claude-vyanktesh-fork            240      98      c6867f1d294b8ee5 pending
v633  53130262  ahmedkhudair121/bc2026-claude-raunak-multi-model-fork   3        91      30cc8796efbbaee8 pending
v634  53130270  mohamadmatali/bc2026-claude-meenalsinha-improved-fork   240      94      48e8eb7f8409ac11 pending
v635  53130272  joriahmed/bc2026-claude-mattia-943-blend-fork           240      97      4ea48f143ed5b877 pending
```

## Verifier / guard summary

- All submitted source kernels were live-preflighted as COMPLETE and had `submission.csv`, `sample_submission`, and `test_soundscapes` source markers.
- Public dry-run outputs were finite, 235-column, nonconstant, and unique vs recent submitted dry-run hashes.
- Descriptions were non-duplicate; cap was rechecked via Bearer submissions list after submission (`5/5`).
- Evidence level: exploratory late-day slot-fill; public LB scores pending.

## Performance table for this run

```text
exp   family                         dry-run rows  primary metric       baseline/delta       decision
v631  Two-pass SSM public source     240          LB pending           v616 0.949 / pending submitted; monitor
v632  Vyanktesh public source        240          LB pending           v616 0.949 / pending submitted; monitor
v633  Raunak multi-model public sour 3            LB pending           v616 0.949 / pending submitted; monitor
v634  MeenalSinha improved public so 240          LB pending           v616 0.949 / pending submitted; monitor
v635  Mattia 943 blend public source 240          LB pending           v616 0.949 / pending submitted; monitor
```

## Top comparable public/source submissions

```text
candidate  family/source                              public LB  delta vs 0.949
v621       Pilkwang EoS7 OOF-gated PCEN source         0.949     +0.000
v622       Beicicc EoS6 P090 source                    0.949     +0.000
v623       Anthony M5-only public source               0.949     +0.000
v625       Safar 0948 public fork                      0.948     -0.001
v629       Yaroslav BirdNET third source               0.946     -0.003
```

## Critic / verifier decision

- Critic: repo-owned sidecars still lack hidden-safe submission packaging; source-code reruns are the safer late-day use of slots than static local sidecar CSVs.
- Verifier: accepted late exploratory source submissions after guards; no known malformed/static/fallback-only/duplicate output was submitted by this run.
- Next exact action: monitor v631-v635 scores after completion, then update the table rows from `pending` to public LB results.
