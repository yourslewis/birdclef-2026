# Critic queue review — BirdCLEF hill-climb cron — 2026-05-30 00:16Z

Role: Critic / validation / queue reviewer. Scope: rank the next model/data-point actions only. I did **not** train, submit, or edit the canonical performance table.

## Precondition

- First monitor v636-v640 and update pending rows. They consumed the 2026-05-29 slots and were still pending in the latest queue. If any beats `0.949`, re-rank around that signal before more training.
- If none beats `0.949`, continue no-slot model/data-point work below. Current best public LB remains `0.949` in the latest completed evidence.

## Ranked next 5 actions

### 1) Run the prepared PANNs all-class full file-context + file-MIL sequence data point

**Decision:** PROCEED as the next bounded training data point after pending-score check.

**Why this is not exhausted:** PANNs all-class remains the strongest train_soundscapes sequence family by row AUC, and the exact full file-context + file-MIL combination exists as an unrun `20260530` config. We have tested PANNs no-file, file-context, localmax-only, and localmax+file-MIL; we have tested **fused** full file-context+file-MIL, but not PANNs-only full file-context+file-MIL.

**Evidence quality:** Comparison-grade. Strong enough to rank a no-slot data point; not enough for submission without sidecar audit.

**Best comparable baselines:**
- PANNs no-file all-class: row `0.647816`, file-MIL `0.670723`; sidecar lift vs v616 `-0.002538`.
- PANNs localmax-only all-class: row `0.641501`, file-MIL `0.681753`; best sequence sidecar lift vs v616 `-0.001728`.
- PANNs localmax+file-MIL: row `0.644053`, file-MIL `0.665302`; sidecar lift vs v616 `-0.002117`.
- Fused full file-context+file-MIL: row `0.594204`, file-MIL `0.678623`; sidecar lift vs v616 `-0.002499`.

**Promotion gate:** continue/package only if it beats the nearest PANNs baseline on at least one primary sequence metric without row collapse: row AUC `>= 0.648` **or** file-MIL `>= 0.682`, and final predictions are finite/nonconstant. Promote to sidecar audit if sidecar lift is better than current best sequence sidecar (`>-0.001728` vs v616) or shows clear anchor lift with low displacement.

**Kill gate:** reject unchanged if row `<0.642` and file-MIL `<0.670`, or if sidecar lift vs v616 is `<= -0.0020` after audit.

**Exact next command if config is synced to trainer:**

```bash
ssh yourslewis@192.168.0.10 'cd ~/birdclef-2026 && source ~/kaggle_envs/s6e3/bin/activate && CUDA_VISIBLE_DEVICES=1 python scripts/birdclef_soundscape_sequence_mining.py --config configs/birdclef/soundscape_sequence_panns_cnn14_allcls_r2_filectx_filemil_losite_ep20_20260530.json'
```

---

### 2) Diagnose the original soft1279 head-loaded sidecar movement before more soft1279 fine-tune knobs

**Decision:** PROCEED with diagnostics; REJECT more blind soft1279 ablations until this is understood.

**Why this is not exhausted:** The original head-loaded soft1279-native all-class package is still the only repo-owned branch with material positive v616 proxy lift. Follow-up training knobs mostly hurt: site-balanced, calibration-none, observed-positive, encoder-only, and head-only all regressed. The missing action is not another knob; it is a class/site/file movement diagnosis explaining where `w0.16` helps and why the anchor gate fails.

**Evidence quality:** Comparison-grade, close to verifier-grade but failed strict gates.

**Best comparable baseline:**
- Head-loaded stability grid `soft1279init_native_allcls_w0p16`: local AUC `0.995545`, lift vs v616 `+0.002064`, lift vs anchor `+0.005155`; site/file q05 positive but `lift_vs_anchor_min` gate requires `>= +0.006`.
- Per-class selector: site-CV lift only `+0.000280`, q05 `-0.003768`, p>0 `0.167`; file-CV lift `+0.001571`.

**Promotion gate:** only promote a derived cap/selector/calibration if it passes the existing manifest gates: lift vs anchor `>= +0.006`, lift vs v616 `>= +0.001`, site bootstrap q05 `>= +0.003`, file bootstrap q05 `>= +0.0015`, and no leave-site/file negative tail. Also require top-5 recall not to regress.

**Kill gate:** if the diagnostic shows the `+0.002064` lift is concentrated in S22 or a few already-overfit classes with negative leave-site q05, freeze this lane except as ensemble diversity.

**Exact next command:** no single existing command fully answers the movement question. Use existing artifacts from `20260528T1618Z_soft1279init_native_allcls_stability_grid` and `20260528T1424Z_soft1279init_native_allcls_package`; create a compact class×site/file movement report before any new training.

---

### 3) Upgrade no-call/background negatives before another suppression-sidecar attempt

**Decision:** REVISE protocol first; do not submit or train another no-call sidecar until negatives are less weak/site-skewed.

**Why this is not exhausted:** No-call is directionally plausible for non-Aves/no-train suppression, but current negative labels are too weak. Farneg10 improved the aggregate gate (`0.963346` vs prior `0.950469`) yet still trails a raw confidence baseline and uses only 23 background negatives across 3 sites.

**Evidence quality:** Exploratory to comparison-grade; not verifier-grade.

**Best comparable baselines:**
- Aggregate farneg10 gate: no-call AUC `0.963346`, site min/q05 `0.703601/0.726574`.
- Raw confidence baseline: soft1279enc max AUC `0.985645`, still better by `+0.022298`.
- Best farneg10 suppression smoke: lift vs v616 `+0.000084`, lift vs anchor `+0.003174`, top5 recall regressed.

**Promotion gate:** require at least 5 sites with negatives and preferably `>=50` negatives, no-call gate AUC competitive with raw max confidence or site q05 `>=0.80`, then suppression lift vs v616 `>= +0.001` with non-regressing top5 recall and positive site/file q05.

**Kill gate:** reject if valid no-call sites remain `<=3`, suppression lift stays `<+0.0002`, or top5 recall regresses again.

**Exact next command if only rerunning current farneg10 smoke:**

```bash
python3 scripts/birdclef_nocall_suppression_sidecar_audit.py --experiment-id soundscape-nocall-suppression-v616-agg-farneg10-extended-20260530 --output-dir artifacts/soundscape_nocall_suppression/soundscape-nocall-suppression-v616-agg-farneg10-extended-20260530 --gate-predictions artifacts/soundscape_nocall_gate/soundscape-nocall-gate-soft1279native-agg-farneg10-losite-20260529/nocall_gate_predictions.csv --bootstrap-iters 500
```

But the critic recommendation is to **not** spend the next cycle on this command unless a stricter/hand-verified negative set is added first.

---

### 4) Build a head-preserved all-class native model weighted toward non-Aves/no-train, rather than another 72-label reinitialized specialist

**Decision:** REVISE/IMPLEMENT only if small code/config support is added; do not repeat the current scoped-specialist pattern.

**Why this is not exhausted:** The 72-label non-Aves/no-train specialist used soft1279 encoder init with a reinitialized head. The evidence suggests the useful soft1279 signal comes from preserving the calibrated 234-class head and allowing encoder movement, not from encoder-only or head-only constraints. A better specialist would keep 234 outputs/head-loaded init, but upweight non-Aves/no-train losses or diagnostics.

**Evidence quality:** Exploratory, based on ablation contrast.

**Best comparable baselines:**
- Soft1279enc 72-label non-Aves/no-train: row `0.609793`, no-train `0.613437`, file-MIL `0.551016`; package lift vs v616 `+0.000347` raw anchor-preserved member.
- Head-loaded all-class soft1279 sidecar: lift vs v616 `+0.002064`, but failed anchor robustness.
- PANNs no-train sequence: row `0.601305`; fused no-train file-MIL `0.660711` but poor row/proxy transfer.

**Promotion gate:** improve scoped LOSO row above `0.613` or file-MIL above `0.661` while keeping all-class head/package sidecar lift vs v616 `>= +0.001`; non-Aves/no-train class movement must be positive under site-CV.

**Kill gate:** reject if scoped row `<0.60`, file-MIL `<0.60`, or package lift `<+0.0003`; also kill if pooled diagnostics show the same sonotype/site inversion as the current specialist.

**Exact next command:** not obvious with current config contract. The existing trainer can run all-class or scoped heads, but not a 234-output head-loaded model with loss weighting targeted to non-Aves/no-train. Add a small config/script option first rather than forcing `class_scope=nonaves_or_no_train` with `initial_load_head=true`, because the head shape will not load.

---

### 5) Run a no-train/file-MIL class-site movement diagnostic before any more no-train wrappers

**Decision:** PROCEED with diagnostics; REJECT more blind no-train wrappers for now.

**Why this is not exhausted:** No-train branches show conflicting signals: PANNs no-train has the best row signal, fused no-train has the best file-MIL, and soft1279enc scoped has the best anchor-preserved local lift. The issue is likely class/site inversion, not lack of another wrapper variant.

**Evidence quality:** Comparison-grade for rejecting blind wrappers; exploratory for selecting a repair.

**Best comparable baselines:**
- PANNs no-train context: row `0.601305`, file-MIL `0.616149`.
- PANNs no-train localmax/row-only follow-ups regressed (`0.582799` / `0.573836`).
- Fused no-train: row `0.554429`, file-MIL `0.660711`, sidecar lift vs v616 `-0.003083`.
- Soft1279enc non-Aves/no-train: row `0.609793`, package lift vs v616 `+0.000347`.

**Promotion gate:** identify classes/sites where file-MIL improves without row inversion; only then test a targeted blend/cap. Require site-CV lift `>0`, q05 nonnegative, and sidecar lift better than `+0.0005` before promotion.

**Kill gate:** if gains are isolated to one site or sonotype family and leave-site q05 is negative, stop no-train training and use the evidence only for private ensembling/error analysis.

**Exact next command:** no existing single command is ideal. Use existing metrics/OOF artifacts to produce a class×site table first; avoid another training run until the inversion source is known.

## Overall critic decision

**REVISE queue, then PROCEED with action #1.** The queue should not spend the next cycle on another soft1279 fine-tune knob, another no-call suppression smoke, or another blind no-train wrapper. The highest-EV bounded data point is the prepared PANNs all-class full file-context+file-MIL sequence run. In parallel conceptually, but not as training, diagnose the head-loaded soft1279 movement because it remains the only branch near promotion gates.

## Files read / evidence sources

- `specs/birdclef-hillclimb-cron-20260525/spec.md`
- `specs/birdclef-hillclimb-cron-20260525/ranked_queue_20260529T2219Z.md`
- `artifacts/model_data_point_ledger/performance_table.md`
- `docs/BIRDCLEF_AUTORESEARCH_LOG.md` tail
- Relevant configs and summaries under `configs/birdclef/`, `artifacts/soundscape_sequence_mining/`, `artifacts/sed_soundscape_packaging_audit/`, and `artifacts/soundscape_nocall_*`.
