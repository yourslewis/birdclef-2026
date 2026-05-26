# BirdCLEF Hill-Climb Ranked Queue — 2026-05-26 18:20 UTC

## Live status
- Coordinator status: mid UTC-day no-slot training pass; no leaderboard submission approved.
- Kaggle Bearer API check: best known public LB remains **0.949**. Latest scored submissions: `v616=0.949`, `v617=0.949`, `v618=0.946`, `v619=0.944`, `v620=0.949`.
- UTC slot usage at check: **0/5** used on 2026-05-26; ~5.7h until reset.
- Active jobs before implementation: no active local BirdCLEF/Kaggle job and trainer GPUs idle.
- Git branch: `feature/birdclef-20260524-20utc-v612-submit`.

## Critic / scout decision
- No fresh verifier-grade public/code candidate was available for an early/mid-day slot.
- Direct repeats of `v616`/`v617`/`v620` are forbidden as duplicates/near-duplicates.
- The best data-point action after the 16:20 native-B0 negative result was to use the current DyMN10 context MLP as control and run a bounded robustness ablation before any wrapper decision.

Evidence level: **comparison-grade no-slot model training**.

## Action taken this run — model data point
Trained `soundscape-sequence-dymn10-r2-nofile-reg-losite-ep20-20260526`.

- Script: `scripts/birdclef_soundscape_sequence_mining.py`
- Config: `configs/birdclef/soundscape_sequence_dymn10_r2_nofile_reg_losite_ep20_20260526.json`
- Artifact root: `artifacts/soundscape_sequence_mining/soundscape-sequence-dymn10-r2-nofile-reg-losite-ep20-20260526/`
- Data: official `train_soundscapes`, `1,478` windows / `66` files / `9` sites, `72` non-Aves/no-train labels.
- Model: EfficientAT DyMN10 embeddings + regularized context MLP; radius-2 local mean/max context; no file mean/max; no site one-hot.

Results:
- Row-only leave-site AUC mean: `0.567307`.
- Context leave-site AUC mean: `0.587753`, delta `+0.020445` vs row-only.
- File-MIL context AUC mean: `0.664545`.
- No-train context AUC mean: `0.489591`.
- Compared with the 10:20 DyMN10 context control: row AUC `-0.013602`, file-MIL `+0.032418`.
- Fold guard: S03 improved strongly (`+0.075345` vs row-only), but S22 remains negative (`-0.063999`), so no wrapper/submission yet.

Verifier checks:
- Leave-site context predictions `1314 x 72`; finite, nonconstant `72/72` columns.
- TorchScript artifact exists (`context_head_torchscript.pt`, ~5.31 MB).
- Not competition format and not v616-audited; **no Kaggle submission approved**.

## Ranked queue after this run

1. **Cautious 72→234 wrapper/audit for DyMN10 sequence context (NEEDS_REVISION)**
   - Potential: medium only if S22/no-train risk is capped.
   - Evidence: baseline context remains best row AUC (`0.601355`), new r2/no-file reg is best file-MIL (`0.664545`).
   - Next exact action: build an offline wrapper/audit that emits only scoped non-Aves/no-train sidecar predictions and evaluates vs v616; reject if S22/no-train or v616 delta is unstable.

2. **Multi-site AudioSet/DyMN10 234-class sidecar reformulation (ACCEPTED fallback)**
   - Potential: medium as a broad acoustic/context sidecar.
   - Avoid more single-site shallow heads; use leave-site and v616 audit.

3. **Late-day guarded slot-fill review (<3h reset)**
   - If no package appears by ~21:00 UTC, scout source-clean, nonduplicate exploratory candidates and submit only those passing schema/rule guards.

4. **Fresh pretrained asset preflight (YAMNet / SurfPerch / PaSST / HTS-AT / BEATs / CLAP)**
   - Potential high but setup risk; do asset/license/runtime preflight only.

5. **G124/V2S hard-confidence target ablation**
   - Lower priority after tiny `+0.00000339` v616-sidecar lift.

## Top-5 comparable historical model data points
By most comparable soundscape/sequence row or file-MIL evidence:
1. `DyMN10 r2 no-file regularized context` — row `0.587753`, file-MIL `0.664545` (this run; best file-MIL, S22 risk).
2. `DyMN10 context MLP` — row `0.601355`, file-MIL `0.632127` (best row control).
3. `EfficientAT DyMN10 S08 head` — S08 row `0.568586` (single-site frozen embedding head).
4. `Soundscape-native B0 LOSO` — row `0.558044`, file-MIL `0.429828` (native CNN negative data point).
5. `Gated sequence smoother` — row `0.556907`, file-MIL `0.591958` (rejected TCN/gated variant).

## Submission decision
**Rejected for now.** Slots are available, but this is not competition-format, not v616-audited, and still fails a key site guard. Continue no-slot packaging/audit or prepare late-day source-clean slot fill if nothing verifier-grade exists near reset.
