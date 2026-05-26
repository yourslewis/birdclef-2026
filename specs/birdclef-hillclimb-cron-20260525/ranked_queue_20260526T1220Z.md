# BirdCLEF Hill-Climb Ranked Queue — 2026-05-26 12:20 UTC

## Live status
- Coordinator status: early UTC-day no-slot training pass; no leaderboard submission approved.
- Latest Kaggle Bearer API check: best known public LB remains **0.949**. Latest scored submissions: `v616=0.949`, `v617=0.949`, `v618=0.946`, `v619=0.944`, `v620=0.949`.
- UTC slot usage at check: **0/5** used on 2026-05-26; ~11.7h until reset.
- Active jobs: no active local BirdCLEF/Kaggle job found; trainer GPUs idle and no active BirdCLEF trainer process found.
- Git branch: `feature/birdclef-20260524-20utc-v612-submit`.

## Public scout / model-source refresh
- Quick 2026-05-26 public search for fresh BirdCLEF 2026 0.949/0.950+ code and EfficientAT/PANNs/DyMN10 leads surfaced no clean new Kaggle code lead above the tied 0.949 plateau.
- The only fresh-ish external hit was a general passive-acoustic dataset paper, not a directly packageable BirdCLEF hidden-safe source.
- Existing public/code landscape remains saturated around EoS/Perch/ProtoSSM/SED/Jungchan/Nina/Karnak/SYD families; direct replays already tied/dropped or are malformed/static/duplicate.

Evidence level: **comparison-grade** for queue ranking; no public-source submission candidate became verifier-grade.

## Action taken this run — model data point
Trained the next `train_soundscapes` sequence/file/site branch as a compact per-file temporal convolutional model:

- Script: `scripts/birdclef_soundscape_tcn_mining.py`
- Config: `configs/birdclef/soundscape_tcn_dymn10_losite_ep20_20260526.json`
- Artifact root: `artifacts/soundscape_sequence_mining/soundscape-tcn-dymn10-losite-ep20-20260526/`
- Input: cached EfficientAT DyMN10 embeddings over official `train_soundscapes` 5s windows; 1,478 rows, 66 files, 9 sites.
- Target scope: 72 non-Aves/no-train labels.
- Model: per-file residual TCN, hidden 256, 3 dilated layers, time features, site-balanced file sampling, BCE with clipped positive weights, 20 epochs.
- Runtime on trainer: ~14.4s summed folds + final export.

Results vs previous context-MLP sequence-mining artifact:

| metric | previous context MLP | compact file TCN | delta |
|---|---:|---:|---:|
| leave-site row macro AUC mean | 0.601355 | 0.547582 | -0.053773 |
| leave-site file-MIL macro AUC mean | 0.632127 | 0.606240 | -0.025887 |

Fold deltas vs context MLP:
- `S03`: **+0.195896** (TCN fixes the worst prior regression site)
- `S08`: -0.076063
- `S13`: -0.053791
- `S19`: -0.085756
- `S22`: -0.021799
- `S23`: -0.281125

Verifier checks:
- Finite/nonconstant final predictions: 72/72 nonconstant columns, min `1.23e-6`, max `0.9997`, std `0.2103`.
- TorchScript smoke passed on trainer: `(2,12,input_dim)->(2,12,72)`.
- Not competition-format; no 234-class wrapper or v616 audit. **No submission approved.**

## Ranked queue after this run

1. **Residual/gated train_soundscape sequence model (ACCEPTED next)**
   - Expected LB potential: medium as a low-weight rare-slice sidecar if it preserves the context-MLP mean while stealing the TCN's S03 gain.
   - Information value: high. Current evidence says temporal sequence modeling contains useful site-specific signal, but naive TCN overfits/hurts S08/S19/S23.
   - Next exact experiment: train/audit a residual or gated smoother over row/context/TCN features with explicit S03/S22 guard and leave-site fold reporting; do not package unless it beats context MLP mean and avoids S22/S03 regressions.

2. **Deeper soundscape-native training variant**
   - Expected LB potential: medium/high if it produces real non-Aves/no-train signal beyond frozen embeddings.
   - Information value: high. The top data source is still `train_soundscapes`; move beyond shallow heads, but avoid blind full fine-tune of huge AudioSet encoders.
   - Gate: compact CNN/SED or adapter/last-block fine-tune with strong regularization, leave-site/file validation, export/runtime smoke.

3. **AudioSet branch reformulation into multi-site features/234-class wrapper**
   - Expected LB potential: medium as broad acoustic/no-call/rare-slice sidecar, low as current 72-label heads.
   - Evidence: DyMN10 > PANNs > MN10 on S08, but still not submission-format. Need multi-site validation and wrapper/audit, not another single-site shallow repeat.

4. **G124/V2S hard-confidence / target-power ablation**
   - Expected LB potential: medium if target design creates a sidecar with non-noise lift.
   - Evidence: prior G124 V2S/localmax had strong proxy AUC but only `+0.00000339` local sidecar lift vs v616; proceed only with a changed target contract and sidecar audit.

5. **Broader OOF negative/no-call SED student**
   - Expected LB potential: low/medium; previous broad-neg aux hurt soft-only control, but no-call remains strategically important.
   - Gate: better negative protocol or calibrated any-call detector; do not rerun same aux unchanged.

6. **Late-day slot-fill candidate queue**
   - Only activate under the slot policy near reset or if a verifier-grade package appears. Avoid exact v616/v617/v620 duplicates, malformed/static/public-output-only finals, and SYD/P949/EoS/ProtoSSM/SED clone increments.

## Critic decision
**REVISE, not submit.** The TCN is an important negative/diagnostic data point: it proves per-file temporal modeling can help S03, but the broad fold regressions mean it is not a competition candidate unchanged. The correct next step is a guarded residual/gated sequence branch or a deeper soundscape-native model, not spending an early-day slot.

## Verifier decision
**ACCEPTED as no-slot artifact; REJECTED as submission.** Output is finite, nonconstant, and export-smoked, but it is 72-label landscape evidence only and lacks competition schema, 234-class wrapper, v616-sidecar audit, and group-stable lift.
