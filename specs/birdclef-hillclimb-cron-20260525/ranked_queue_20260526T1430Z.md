# BirdCLEF Hill-Climb Ranked Queue — 2026-05-26 14:30 UTC

## Live status
- Coordinator status: early UTC-day no-slot training pass; no leaderboard submission approved.
- Kaggle Bearer API check: best known public LB remains **0.949**. Latest scored submissions: `v616=0.949`, `v617=0.949`, `v618=0.946`, `v619=0.944`, `v620=0.949`.
- UTC slot usage at check: **0/5** used on 2026-05-26; ~9.6h until reset.
- Active jobs: no active local BirdCLEF/Kaggle job found; no active BirdCLEF trainer process found.
- Git branch: `feature/birdclef-20260524-20utc-v612-submit`.

## Public scout / model-source refresh
- Web scout queries for fresh BirdCLEF 2026 `0.950`/solution notebook leads surfaced no clean new Kaggle code lead above the tied 0.949 plateau.
- EfficientAT/PANNs/DyMN10 query again surfaced only the general passive-acoustic paper and irrelevant/generic results, not a packageable hidden-safe Kaggle source.
- Current public/code landscape remains saturated around already-tested EoS/Perch/ProtoSSM/SED/Jungchan/Nina/Karnak/SYD families; duplicate/static/malformed/public-output-only candidates remain rejected.

Evidence level: **comparison-grade** for queue ranking; no public-source submission candidate became verifier-grade.

## Action taken this run — model data point
Trained the next train_soundscapes sequence/file/site branch: a guarded residual/gated smoother intended to combine the context MLP's stronger mean performance with a bounded per-file TCN residual.

- Script: `scripts/birdclef_soundscape_gated_sequence_mining.py`
- Config: `configs/birdclef/soundscape_gated_sequence_dymn10_context_tcn_losite_ep18_20260526.json`
- Artifact root: `artifacts/soundscape_sequence_mining/soundscape-gated-sequence-dymn10-context-tcn-losite-ep18-20260526/`
- Input: cached EfficientAT DyMN10 embeddings over official `train_soundscapes` 5s windows; 1,478 rows, 66 files, 9 sites.
- Target scope: 72 non-Aves/no-train labels.
- Model: row context MLP + bounded gated TCN residual, site-balanced file sampling, BCE with clipped positive weights, gate/residual regularization, 18 epochs.

Results vs context-MLP sequence-mining artifact:

- Leave-site row macro AUC mean: `0.556907` vs context `0.601355` (`-0.044448`).
- File-MIL macro AUC mean: `0.591958` vs context `0.632127` (`-0.040168`).
- Fold deltas vs context: S03 `-0.097429`, S08 `-0.047846`, S13 `+0.042120`, S19 `-0.149320`, S22 `+0.037136`, S23 `-0.051347`.

Verifier checks:
- Finite/nonconstant final predictions: 72/72 nonconstant columns, min `2.35e-14`, max `0.9999998`, std `0.20662`.
- TorchScript export completed: `gated_sequence_torchscript.pt`.
- Not competition-format; no 234-class wrapper or v616 audit. **No submission approved.**

## Ranked queue after this run

1. **Deeper soundscape-native training variant (ACCEPTED next)**
   - Expected LB potential: medium/high if it produces real non-Aves/no-train signal beyond frozen DyMN10 shallow heads.
   - Information value: high. The last three sequence experiments show `train_soundscapes` carries signal, but shallow context/TCN/gating is not enough.
   - Next exact experiment: compact CNN/SED or adapter/last-block fine-tune on official `train_soundscapes`, with leave-site/file gates, site-balanced sampling, strong regularization, and export/runtime smoke. Avoid blind full fine-tuning of huge AudioSet encoders.

2. **AudioSet branch reformulation into multi-site features/234-class wrapper**
   - Expected LB potential: medium as broad acoustic/no-call/rare-slice sidecar.
   - Evidence: DyMN10 > PANNs > MN10 on S08 and context MLP improved leave-site mean, but current artifacts are 72-label only. Needs multi-site wrapper/audit rather than another single-site shallow head.

3. **Residual/regularized context-only ablation**
   - Expected LB potential: low/medium. Context MLP remains best sequence artifact, but guarded TCN failed S03 and S19.
   - Only pursue if it changes one variable: stronger dropout/weight decay or worst-site objective on the context MLP, not another TCN-style residual.

4. **G124/V2S hard-confidence / target-power ablation**
   - Expected LB potential: medium if target design creates a sidecar with non-noise lift.
   - Evidence: prior G124 V2S/localmax had strong proxy AUC but only `+0.00000339` local sidecar lift vs v616; proceed only with changed target contract and sidecar audit.

5. **Calibrated no-call/background detector**
   - Expected LB potential: low/medium but strategically distinct.
   - Evidence: broad negative aux hurt the matched soft-only control; no-call remains undermeasured. Needs a better trusted-negative protocol before another aux rerun.

6. **Late-day slot-fill candidate queue**
   - Activate only under slot policy near reset or if a verifier-grade package appears. Avoid exact v616/v617/v620 duplicates, malformed/static/public-output-only finals, and SYD/P949/EoS/ProtoSSM/SED clone increments.

## Critic decision
**REVISE, not submit.** The gated smoother failed its own S03/S22 guard and underperformed the context MLP on both row and file-MIL leave-site means. The opportunity-cost winner is now a compact deeper soundscape-native branch, not another sequence postprocessor.

## Verifier decision
**ACCEPTED as no-slot artifact; REJECTED as submission.** Output is finite/nonconstant and exported, but it is 72-label only, not competition schema, not v616-audited, and not group-stable enough for a slot.
