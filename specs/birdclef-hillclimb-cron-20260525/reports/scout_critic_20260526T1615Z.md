# Scout + Critic Report — BirdCLEF hill-climb cron — 2026-05-26 16:15 UTC

## Scope / evidence read
- Spec: `specs/birdclef-hillclimb-cron-20260525/spec.md`, including the corrected default queue: train_soundscapes sequence/file/site mining first, then deeper soundscape-native variants, then AudioSet reformulation and older lanes.
- Recent queues: `ranked_queue_20260526T1020Z.md`, `ranked_queue_20260526T1220Z.md`, `ranked_queue_20260526T1430Z.md`.
- Recent reports/ledgers: `reports/scout_critic_20260526T0815Z.md`, `reports/scout_critic_20260526T1020Z.md`, and ledgers through `artifacts/model_data_point_ledger/20260526T1430Z_soundscape_gated_sequence_mining.md`.
- Public/model evidence: `artifacts/public_kernels_20260526_scout/scan_20260526T0020Z.json`, `specs/birdclef-new-branch-ideation-20260525/reports/public_model_scout.md`, repo `rg` over docs/specs/artifacts for EfficientAT/PANNs/BirdNET/Perch2/SurfPerch/PaSST/HTS-AT/BEATs/CLAP/YAMNet/WildSound/Alexy, plus a light web refresh for BirdCLEF 2026 0.950/EfficientAT/PANNs/SurfPerch leads.

Evidence level: **comparison-grade scout/critic synthesis**. This role did not train, submit, or modify source; it only writes this report.

## Live assumptions to verify before the coordinator acts
- Re-check Kaggle state before any submission or training handoff: last recorded best is still **0.949**; `v616`, `v617`, and `v620` tied; `v618=0.946`, `v619=0.944`; 2026-05-26 slots were **0/5** at 14:30Z.
- Re-check active local/trainer jobs. The last queue says no active BirdCLEF jobs, but this report did not query the trainer.
- Re-check whether any post-14:30Z artifact/queue exists. Repo file scan at report time showed latest relevant artifact remains `ranked_queue_20260526T1430Z.md` / gated-sequence ledger.
- If a candidate is promoted, verify competition schema, row alignment, nonconstant/finite outputs, source/runtime status, and non-duplication versus v616/v617/v620. Current soundscape artifacts are still 72-label landscape artifacts, not submissions.
- Re-verify external asset licenses before packaging BirdNET/Bioacoustics/CLAP/transformer model zoo branches. Existing docs flag BirdNET model licensing as a review item.

## Public/model scout finding
No obvious fresh public/model lead emerged from repo docs/artifacts or the light web refresh.

- The 2026-05-26 public scan lists recent kernels already categorized as rejected/plateau/blocked: WildSound V8 had ERROR/no outputs, Viktoriia EfficientAT-marked inference had bad final values, Tulay had mock/wrong-shape output, Kijiang/Gendaijin/P952-style finals were malformed/cache/static-risk, and Nina/EoS/BirdNET/Karnak variants already tied or dropped.
- Repo docs repeatedly converge on the same conclusion: direct EoS/Perch/ProtoSSM/SED/Jungchan/Nina/Karnak/SYD families are saturated around the 0.949 plateau; static public-output-only finals and exact/near duplicates remain invalid.
- External model leads remain research lanes, not fresh submit candidates: EfficientAT/PANNs have now produced useful but weak/72-label soundscape data points; BirdNET/Perch2/SurfPerch/Bioacoustics Model Zoo/PaSST/HTS-AT/BEATs/CLAP/YAMNet are possible future asset smokes, but none is documented as packageable and verifier-grade today.
- Light web refresh did not surface a clean Kaggle 0.950+ code lead. Search results were empty/noisy or repeated the general passive-acoustic dataset paper already noted in recent queues, not a hidden-safe BirdCLEF 2026 source.

## Critique of the current next action
The 14:30Z queue's next action — **pivot to a compact deeper soundscape-native CNN/SED or adapter branch** — is directionally right, but it should be tightened before implementation.

Why it is right:
- The user correction has now been validated: treating `train_soundscapes` as files/sites/sequences produced the best soundscape-specific signal so far. The context branch improved leave-site row AUC from `0.578422` to `0.601355` and file-MIL from `0.563852` to `0.632127`.
- Two follow-up sequence postprocessors were useful negatives. The per-file TCN fell to `0.547582` row / `0.606240` file-MIL; the gated residual smoother fell to `0.556907` row / `0.591958` file-MIL and failed the S03/S22 guard. That argues against another shallow TCN/gating tweak.
- The current best sequence signal still depends on frozen DyMN10 embeddings plus engineered context. A distinct model data point that learns from soundscape spectrograms/windows/files directly is the next high-information branch.

Required revision:
- Do **not** frame this as blind full fine-tuning of a large AudioSet encoder on 1,478 sparse rows. That is likely to overfit site/label quirks.
- Make it a bounded data-point branch with strong regularization: compact CNN/SED or adapter/last-block fine-tune, site-balanced file sampling, leave-site plus leave-file validation, sequence/file MIL outputs, and explicit S22/S03/S19 guard reporting.
- Keep it 72-label/comparison-grade unless/until it produces stable leave-site/file gains and then earns a 234-class wrapper + v616 audit.
- Train it anyway even if not submission-grade, because the standing policy is to measure distinct model families. The success criterion is useful landscape information or a promotable raw sidecar, not immediate LB readiness.

## Ranked next branches
1. **Compact deeper soundscape-native CNN/SED or adapter data point — PROCEED, revised exact next action.**
   - Train on official `train_soundscapes` as ordered files/sites, not isolated rows.
   - Use 72 non-Aves/no-train labels plus a carefully documented background/no-call protocol if available.
   - Prefer a small logmel CNN/SED, EfficientAT last-block/adapter, or shallow task head over cached embeddings with learnable spectrogram layers; avoid unrestricted full fine-tune.
   - Validation gates: leave-site, leave-file or file bootstrap, site-balanced metrics, fold deltas vs the context-MLP baseline, finite/nonconstant export smoke.
   - Promotion gate: only build a 234-class wrapper/audit if it beats the context MLP mean and does not create large S22/S03/S19 regressions.

2. **Context-MLP robustness/calibration audit — KEEP AS CONTROL, not main branch.**
   - The 10:20 context MLP remains the best sequence artifact. Use it as the baseline/control for any deeper branch.
   - A tiny regularization/dropout/worst-site objective ablation is acceptable only if it changes one variable and is cheap; it should not displace the deeper-native data point.

3. **AudioSet reformulation into multi-site features / 234-class sidecar — DEFER until soundscape-native branch is measured.**
   - DyMN10 beat PANNs and MN10 on S08 (`0.568586` vs `0.517333`/`0.488240`) and enabled the context branch, so AudioSet remains alive.
   - But more frozen single-site 72-label heads are low EV. Next AudioSet work should be multi-site/no-call features or wrapper audit, not another S08 embedding MLP.

4. **Fresh pretrained asset smoke: YAMNet/Bioacoustics Model Zoo/SurfPerch/CLAP/PaSST/HTS-AT/BEATs — RESEARCH LANE, not next training unless assets are already clean.**
   - Potentially high diversity, but current repo evidence has no license/runtime/package-ready artifact.
   - Best use: small asset/license/runtime preflight, then a no-slot embedding comparison on the same leave-site soundscape protocol.

5. **G124/V2S hard-confidence / target-power ablation — FALLBACK.**
   - Keep as fallback if deeper soundscape-native setup is blocked.
   - Prior G124 proxy AUC was strong, but v616 sidecar lift was only `+0.00000339`; do not repeat unchanged target design.

6. **Late-day guarded slot-fill queue — ONLY near reset.**
   - Early-day submission remains rejected from current evidence.
   - Near reset, use only source-clean, nonduplicate, nonmalformed candidates that pass output guards; avoid exact v616/v617/v620 replays and static/public-output-only finals.

## Opportunity-cost critique
- Another TCN/residual/gated sequence smoother is now low EV. Two attempts after the context MLP failed the aggregate metric and created fold regressions; the remaining uncertainty is model/data representation, not another gating trick.
- Jumping directly to a 234-class wrapper is premature. The best soundscape artifacts are 72-label, group-unstable, and not audited against v616. A wrapper before a stable leave-site/file signal risks turning useful landscape evidence into noisy LB probing.
- Generic public-code scouting has diminishing returns today. The public kernel frontier has been scanned repeatedly and mostly yields plateau, malformed, duplicate, or static-output-risk candidates. Use scout time for model-asset feasibility only if it can create a genuinely new raw branch.
- Blindly scaling/finetuning a large pretrained model would be worse than a bounded data point. Sparse labels and S22 dominance mean small controlled experiments are more informative than a heavy overfit-prone run.
- Submission slots are available, but early-day use from this state would be poor information value. The correct use of the cron right now is no-slot model landscape measurement.

## Proceed / revise / reject decision
**PROCEED WITH REVISION.** Proceed to the deeper soundscape-native branch, but define it as a bounded, group-validated data-point experiment over `train_soundscapes` files/sites/sequences. Reject any Kaggle submission at this gate. Reject another shallow sequence postprocessor as the immediate next action unless it is only a tiny control against the deeper branch.

Next exact recommended action: implement/train one compact soundscape-native CNN/SED or adapter-style data point with leave-site/file gates and compare it directly against the 10:20 context-MLP baseline; report it as landscape evidence first, not a slot candidate.
