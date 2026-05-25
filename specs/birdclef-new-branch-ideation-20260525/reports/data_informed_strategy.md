# Data-Informed Branch Strategy — BirdCLEF 2026 New Branch Ideation

_Date: 2026-05-25_  
_Role: Data-Informed Branch Strategist_

## Executive read

The next useful branch should not be another small perturbation of the `0.949` anchor/SED/Jung21/ProtoSSM cluster. The hidden-behavior bet needs to be about **data coverage and domain mismatch**, not marginal local AUC on the same 190-row overlap.

Most important facts from the current audit:

- Local proxy is narrow: `190` matched artifact rows, `20` files, `6` sites, `42` valid AUC classes, versus `234` submission classes.
- Train audio is strongly Aves-heavy: `34,799/35,549` train recordings are Aves; non-Aves train rows are sparse (`451` Amphibia, `199` Insecta, `99` Mammalia, `1` Reptilia).
- Submission taxonomy has `28` classes with **zero train-audio primary labels**: `3` Amphibia plus all `25` `47158son*` insect sonotypes. They have no secondary-label mentions in `train.csv`, but they do appear in soundscape labels.
- Soundscape positives are non-Aves-heavy: collapsed soundscape positive cells are dominated by Amphibia and Insecta, not birds.
- Artifact overlap misses important labeled sites (`S15`, `S19`, `S23`) and sample hidden format shows `S05`, which is not in the current labeled artifact overlap.
- v611/v612/v616 had plausible local sidecar evidence and still tied hidden public LB at `0.949`; local rank-blend lift is now rejection evidence only, not promotion evidence.

Therefore the strongest ideation direction is: build branches that specialize in **no-train labels, non-Aves soundscape events, no-call/background calibration, long-context temporal behavior, and unseen-site robustness**. Keep them as raw sidecars with strict caps until they beat v616 under site/file/taxon gates.

## 1. Data gaps and hidden-behavior hypotheses that matter most

### H1 — No-train primary classes are structurally under-modeled

The `28` no-train-primary classes cannot be learned from `train.csv` primary-label supervision. They are:

- Amphibia: `517063` Southern Orange-legged Leaf Frog (`313` soundscape rows), `1491113` Guaraní leaf-litter frog (`79`), `25073` Chiasmocleis mehelyi (`12`).
- Insecta sonotypes: `47158son01`–`47158son25`, with highest observed soundscape counts including `son25` (`84`), `son07` (`48`), `son17` (`43`), `son11`/`son13` (`36` each), `son03`/`son10` (`33` each), `son22`/`son23`/`son24` (`24` each).

Hidden hypothesis: current plateau models may rank these classes well only because of public soundscape leakage/proxies or site priors, not because they recognize the event acoustics. Hidden rows from different sites/times can break that. A branch trained directly on soundscape windows or external acoustic embeddings for these classes should behave differently.

### H2 — Non-Aves dominates soundscape labels despite train-audio being Aves-heavy

Train audio says “birds”; labeled soundscapes say “frogs/insects matter.” Positive cells by taxon in labeled soundscapes are approximately:

- Amphibia: `2,087`
- Insecta: `568`
- Aves: `412`
- Mammalia: `42`
- Reptilia: `13`

Hidden hypothesis: general bird-centric anchors can be locally strong but fragile for frog choruses, insect sonotypes, mammals, and reptiles. Non-Aves specialists may shift hidden ordering in ways plateau recipes do not.

### H3 — Site/domain shift is under-sampled by the current artifact overlap

Current artifact validation uses only six sites (`S03`, `S08`, `S09`, `S13`, `S18`, `S22`). Labeled data also includes `S15`, `S19`, and `S23`, and sample hidden rows include `S05`.

Site/taxon structure is sharp:

- `S22`: Amphibia-heavy, including `517063`, `65380`, `555146`, `22973`, `23158`, `24321`, `66971`.
- `S08`: Insecta + Reptilia, including many sonotypes and `116570` Southern Spectacled Caiman.
- `S15`: Aves + `47158son07`/`son08`; absent from artifact overlap.
- `S19`: mixed Amphibia + sonotypes; absent from artifact overlap.
- `S23`: Insecta + Mammalia + Aves + Reptilia; absent from artifact overlap.
- `S09`: Mammalia/Capuchin-heavy in labels, and current anchor is weak-ish locally.
- `S18`: small Amphibia site where SED helps locally but v616 did not move enough.

Hidden hypothesis: site-specific acoustics/backgrounds drive part of the leaderboard. We need branches whose training objective explicitly resists site shortcuts or benefits from unseen-site acoustic invariance.

### H4 — No-call/background behavior is currently almost unmeasured

Collapsed labeled soundscape rows all contain at least one positive label. The `50` unmatched artifact rows are unlabeled, not proven empty. Rank-scaled outputs near mean `0.5` cannot be interpreted as calibrated probabilities. The prior negative-mask auxiliary smoke covered only `26/512` rows and was flat.

Hidden hypothesis: public hidden has many sparse/empty/background-heavy rows or rows with only weak distant calls. Plateau rank blends can look excellent on positive-only local rows while misallocating mass on no-call or background rows. A calibrated background/no-call branch would be genuinely different, even if it only acts as a suppression sidecar.

### H5 — Five-second row scoring may need longer temporal context

Rows are scored at 5-second granularity, but many labels are persistent choruses or context-dependent events. Current time-position bins did not reveal a clean failure, but the validation overlap is too small. High multilabel density (`1`–`10` labels per row, median multi-label behavior) suggests co-occurring temporal context matters.

Hidden hypothesis: a model using 30–60s context, sequence smoothing, or event-boundary logic could rank persistent and intermittent calls differently from per-window rank blends.

### H6 — Existing branch family saturation is real

The audit found many exact or near duplicates:

- Samejima SED = Raunak SED = v616 SED raw.
- Jungchan Model21 = v616 Jung21 raw = SYD `subm_21`.
- SYD52p is ProtoSSM-like and gave only `+0.000020` local AUC over already-tied v616.
- Per-class selector had in-sample lift but essentially zero leave-site transfer.

Hidden hypothesis: another scalar/rank/clone branch will preserve `0.949`, not break it.

## 2. New branch archetypes mapped to hypotheses

### A. No-train class / sonotype soundscape branch

**Mapped hypotheses:** H1, H2, H3.  
**Model family:** soundscape-window specialist; one-vs-rest or shallow heads over pretrained acoustic embeddings; optionally a compact CNN/SED trained only for no-train and very rare classes.  
**Targets:**

- Primary no-train Amphibia: `517063`, `1491113`, `25073`.
- Sonotypes: all `47158son01`–`47158son25`; prioritize high-count `son25`, `son07`, `son17`, `son11`, `son13`, `son03`, `son10`; include low-count `son05`, `son12`, `son19`, `son09`, `son02` only under strong caps.
- Sites: `S08`, `S15`, `S19`, `S23` for sonotypes; `S03`, `S13`, `S18`, `S19`, `S22` for no-train Amphibia.

**Why hidden behavior should differ:** it directly trains on the labels the train-audio-primary pipeline cannot see. It should not share the same class priors as Aves-heavy public anchors.

**Implementation path:** build features for all labeled train-soundscape 5s windows using a frozen acoustic encoder and train small label heads for the 28 no-train classes plus nearby rare non-Aves classes. Compare raw outputs as a low-weight taxon/class-capped sidecar against anchor and v616.

**Safety risks:** local no-train/sonotype proxy is site-correlated and some sonotypes are locally saturated; aggressive movement can overfit. Do not promote if lift comes only from `S22` or one insect site.

**First smoke:** train/evaluate only the 28 no-train labels with leave-site and leave-file splits; require improvement on at least one held-out sonotype site and no degradation on high-count Amphibia.

### B. Non-Aves acoustic specialist branch

**Mapped hypotheses:** H2, H3, H5.  
**Model family:** taxon-specialized SED/CNN/embedding head with class-balanced sampling and non-Aves augmentations; separate heads for Amphibia/Insecta/Mammalia/Reptilia.  
**Targets:**

- Amphibia: `65380`, `517063`, `22973`, `555146`, `23158`, `24279`, `24321`, `22967`, `66971`, `1491113`.
- Insecta: `47158son*`, especially `son25`, `son07`, `son17`, `son11`, `son13`, `son03`, `son10`.
- Mammalia: `516975` Hooded Capuchin on `S09`, `43435` Black Howling Monkey on `S23`, plus rare mammal taxonomy classes.
- Reptilia: `116570` Southern Spectacled Caiman, especially `S08`/`S23`.
- Weak/currently informative sites: `S18`, `S09`, `S22`; hidden-risk sites absent from artifact overlap: `S15`, `S19`, `S23`.

**Why hidden behavior should differ:** it reverses the dominant train-audio bias and treats frogs/insects/mammals/reptiles as first-class targets instead of side effects of bird-trained embeddings.

**Implementation path:** train non-Aves-only or non-Aves-upweighted heads from train_soundscape labels plus rare train_audio clips. Evaluate by taxon and site. Blend as taxon-capped sidecar, not global class selector.

**Safety risks:** few Mammalia/Reptilia positives make local metrics anecdotal; Insecta local AUC is already saturated. Need manual taxon movement report and strict caps.

**First smoke:** Amphibia+Insecta head only, evaluated leave-site on `S03/S13/S18/S19/S22` for frogs and `S08/S15/S19/S23` for insects; no Kaggle packaging until it clears file/site gates.

### C. Calibrated no-call/background suppression branch

**Mapped hypotheses:** H4, H3.  
**Model family:** probability-calibrated background/no-call detector or per-class presence gate; not rank-space output. Could be a binary “any target call” model plus taxon gates.  
**Targets:**

- Sparse or empty hidden rows, background-heavy rows, low-confidence rows where rank blends spread mass across many classes.
- Classes likely to false-positive under broad rank blends: rare no-train sonotypes, Mammalia/Reptilia, and low-signal non-Aves classes.
- Sites/backgrounds: derive from unlabeled or weakly labeled soundscape segments only after a trusted negative protocol is defined; do not assume unmatched rows are negatives.

**Why hidden behavior should differ:** current plateau outputs are class-wise ranks and cannot know when to suppress all classes. A calibrated detector changes mass allocation on hidden rows that local positive-only AUC barely tests.

**Implementation path:** create a trusted negative/background set using OOF teacher agreement, distant segments, and/or manually audited empty segments; train a lightweight any-call/no-target detector. Use it only as a cap/suppression sidecar and report predicted mass on positive, sparse, and trusted-negative rows.

**Safety risks:** wrong negatives will destroy recall; the existing negative-mask smoke was too sparse and flat. This is high-diversity but needs data curation before modeling.

**First smoke:** build a negative-set audit, not a model first. Require broad coverage beyond `26/512` rows and verify that shuffled/inverted controls do not look good.

### D. Long-context sequence SED / temporal smoothing branch

**Mapped hypotheses:** H5, H2, H3.  
**Model family:** 30–60s context SED, temporal transformer/TCN over 5s embeddings, or post-hoc sequence smoother that consumes raw per-frame logits rather than final ranks.  
**Targets:**

- Persistent choruses: Amphibia on `S22`, `S03`, `S13`, `S18`, `S19`.
- Intermittent mammals/birds where context helps distinguish isolated calls: `516975`, `43435`, `chacha1`, `whtdov`, `undtin1`, `plcjay1`, `grekis`.
- Time-position slices `<=20s`, `25–40s`, `>=45s`; require no endpoint-specific leakage.

**Why hidden behavior should differ:** anchor/sidecar rank blends mostly act per row/class. A sequence branch can exploit persistence, repetition, call spacing, and co-occurrence across adjacent 5s rows.

**Implementation path:** generate per-5s embeddings/logits from a frozen encoder, train a sequence model or simple HMM/CRF-like smoother on files. Validate leave-file and leave-site. Output raw probabilities, then rank-blend at low weight.

**Safety risks:** high risk of overfitting file/site temporal patterns because labeled files are few. Must include leave-file validation and negative controls.

**First smoke:** use existing raw logits/embeddings if available; train a file-level temporal smoother on full 739 collapsed rows and compare against anchor/v616 on leave-file, not all-row AUC.

### E. Self-supervised acoustic embedding branch

**Mapped hypotheses:** H1, H2, H3, H4.  
**Model family:** frozen public acoustic/audio foundation embeddings with shallow classifiers: e.g. BEATs/AudioMAE/HTS-AT/PaSST/PANNs/YAMNet/BirdNET/Perch-like embeddings, but use a branch that is not already Jung21/SED/ProtoSSM lineage.  
**Targets:**

- No-train labels and rare classes where supervised train-audio examples are absent or weak.
- Non-Aves taxa, especially insect/frog timbres that bird-specific public models may not represent well.
- Unseen-site robustness; embeddings should be evaluated with site adversarial/leave-site criteria.

**Why hidden behavior should differ:** frozen general audio representations can capture broad acoustic texture and background domains, not just known BirdCLEF species priors. This is one of the better ways to escape the public plateau family without needing a full new large model.

**Implementation path:** extract embeddings for train_soundscape windows and selected train_audio clips; train lightweight multilabel heads with class-balanced loss. Compare to anchor and v616; keep outputs raw for branch diversity analysis.

**Safety risks:** many public encoders are heavy or poorly matched to bioacoustics; clean train-audio CV can be misleading. The smoke must be soundscape-first and runtime-aware.

**First smoke:** 1–2 encoders only, no zoo. Train shallow heads for non-Aves/no-train labels and require lift under leave-site/file, not train_audio CV.

### F. Site/domain-adversarial branch

**Mapped hypotheses:** H3 plus all model-overfit modes.  
**Model family:** domain-invariant embedding head, group-DRO training, leave-site early stopping, or ensemble member optimized for worst-site lift rather than average AUC.  
**Targets:**

- Hidden/unseen site `S05` risk.
- Labeled but artifact-missing sites `S15`, `S19`, `S23`.
- Weak/unstable local sites `S18`, `S09`, `S03`.
- Taxa with site concentration: frogs on `S22`, sonotypes on `S08/S15/S19/S23`, mammals on `S09/S23`.

**Why hidden behavior should differ:** it explicitly penalizes site shortcut learning, while previous gates only measured a tiny six-site overlap after the fact.

**Implementation path:** train any of the above specialist branches with site-adversarial loss or group-balanced sampling; optimize worst-site/leave-site metric; report feature/site classifier accuracy as a diagnostic.

**Safety risks:** can reduce useful site priors and hurt public LB if hidden site distribution matches training priors. Best used as a sidecar or training constraint for other branches, not a standalone final.

**First smoke:** run the no-train/non-Aves embedding branch with and without group-DRO/site-adversarial training; promote only if worst-site lift improves without average collapse.

### G. Co-occurrence/context prior branch — low priority

**Mapped hypotheses:** H5.  
**Model family:** taxonomy/co-occurrence graph prior or label-smoothing postprocessor.  
**Targets:** high-density rows and stable co-occurring frog/insect sets.  
**Why hidden behavior should differ:** could improve multilabel consistency.  
**Why low priority:** likely to become another scalar/prior tweak and overfit site co-occurrence. Use only as diagnostic after stronger raw branches exist.

## 3. What each branch should target

### Highest-priority class/taxon slices

1. **No-train primary labels (`n=0`)**
   - `517063`, `1491113`, `25073`, `47158son01`–`47158son25`.
   - Use class caps because local no-train AUC is already near-saturated for many sonotypes.

2. **Rare train primary labels (`1–5` recordings)**
   - Amphibia: `23150`, `23724`, `24321`, `70711`, `25214`, `476521`, `555123`, `64898`, `23176`, `1595929`, `23154`, `66971`.
   - Aves: `sptnig1`.
   - Mammalia: `516975`, `209233`, `74580`, `738183`.
   - Reptilia: `116570`.

3. **Soundscape-dominant Amphibia**
   - `65380`, `517063`, `22973`, `555146`, `23158`, `24279`, `24321`, `22967`, `66971`, `1491113`.

4. **Insect sonotypes**
   - High-count: `47158son25`, `son07`, `son17`, `son11`, `son13`, `son03`, `son10`.
   - Low-count/cap-only: `son05`, `son12`, `son19`, `son09`, `son02`.

5. **Mammalia/Reptilia anecdotal but potentially hidden-diverse**
   - `516975` Hooded Capuchin (`S09`), `43435` Black Howling Monkey (`S23`), `116570` Southern Spectacled Caiman (`S08/S23`).

### Highest-priority site/domain slices

- `S15`, `S19`, `S23`: labeled sites absent from current artifact overlap; useful for new branch validation if raw predictions can be materialized.
- `S05`: hidden sample site; no local artifact evidence.
- `S18`: weak small Amphibia site; SED raw helped locally but final blend did not.
- `S09`: Mammalia-heavy slice; Jung21/SED behavior differs and local sample is tiny.
- `S22`: large Amphibia-heavy site; high risk of local dominance/overfit, but important for frog behavior.

## 4. What not to do again

1. **Do not submit another v616 scalar variant.** Changing weights, rank power, temperature, caps, or tiny sidecars around `anchor + Jung21 + SED` is not enough after v616 tied.
2. **Do not treat site-bootstrap on six sites as approval.** It rejected some bad ideas, but v616 cleared it and still tied.
3. **Do not use per-class adaptive weights from the 190-row overlap.** The current selector had in-sample lift and essentially zero leave-site transfer.
4. **Do not trust train-audio CV or clean-audio fold AUC for soundscape hidden behavior.** v610 is the warning example.
5. **Do not add public clone branches as “diversity.”** SYD52p/ProtoSSM/EoS/PCEN/visual/HGNet clone increments are saturated unless the raw source is truly new and hidden-safe.
6. **Do not calibrate no-call behavior from rank-scaled outputs or unlabeled unmatched rows.** Build a real negative/background protocol first.
7. **Do not optimize only all-row AUC.** Every serious branch needs leave-file, leave-site, taxon movement, and negative controls against both anchor and v616 baseline.
8. **Do not move saturated sonotypes aggressively.** No-train sonotypes are strategically important, but local proxy may already be too easy/site-coded.

## 5. Top 5 branch ideas ranked by expected hidden diversity and feasibility

### 1. Non-Aves + no-train soundscape specialist

**Rank rationale:** best balance of hidden diversity and feasibility. It directly addresses the largest mismatch: Aves-heavy training versus non-Aves/no-train soundscape labels.

- **Expected hidden diversity:** High.
- **Feasibility:** Medium-high; can start with frozen embeddings + shallow heads.
- **Targets:** `517063`, `1491113`, `25073`, `47158son*`, `65380`, `555146`, `24321`, `66971`, `516975`, `116570`.
- **Sites:** frog sites `S03/S13/S18/S19/S22`; insect sites `S08/S15/S19/S23`; mammal/reptile sites `S09/S23/S08`.
- **First smoke:** train no-train + non-Aves heads on train_soundscape windows, evaluate leave-site/file, compare raw branch against v616 baseline.
- **Promotion condition:** improves non-Aves/no-train slices without Aves degradation and has positive leave-site/file lift against v616, not just anchor.

### 2. Calibrated no-call/background suppression branch

**Rank rationale:** likely very different hidden behavior because current validation barely tests it. Feasibility depends on curating trusted negatives.

- **Expected hidden diversity:** Very high.
- **Feasibility:** Medium-low until negative protocol is built.
- **Targets:** sparse/empty/background hidden rows; false-positive-prone rare taxa/classes.
- **Sites:** all sites; especially unmatched/hidden-like backgrounds and labeled sites outside artifact overlap.
- **First smoke:** build trusted negative/background audit and an any-call detector; do not train from assumed unlabeled negatives.
- **Promotion condition:** calibrated mass separation on positive vs trusted-negative rows, no recall collapse, negative controls fail as expected.

### 3. Long-context sequence SED branch

**Rank rationale:** good diversity from row-wise rank blends and directly suited to chorus/persistent event structure.

- **Expected hidden diversity:** High.
- **Feasibility:** Medium; can start as a smoother over existing embeddings/logits before training a full model.
- **Targets:** persistent Amphibia choruses, insect choruses, intermittent mammal/bird calls.
- **Sites:** `S22`, `S03`, `S13`, `S18`, `S19`, plus `S09/S23` for mammals.
- **First smoke:** file-level temporal smoother or TCN over 5s embeddings; evaluate leave-file first.
- **Promotion condition:** file-held-out lift and stable time-bin behavior without endpoint leakage.

### 4. Self-supervised/general acoustic embedding branch

**Rank rationale:** potentially the cleanest way to escape public plateau lineage without a full new training stack.

- **Expected hidden diversity:** High.
- **Feasibility:** Medium; runtime/model availability determines practicality.
- **Targets:** no-train labels, non-Aves, site-shifted backgrounds.
- **First smoke:** one or two frozen encoders only; shallow multilabel heads for no-train/non-Aves; no model zoo.
- **Promotion condition:** low enough correlation to anchor/v616 plus group-stable lift; clean hidden-safe extraction path.

### 5. Site/domain-adversarial specialist or training constraint

**Rank rationale:** directly addresses hidden site risk, but may sacrifice useful site priors if overdone. Best as a training constraint for ideas 1 or 4.

- **Expected hidden diversity:** Medium-high.
- **Feasibility:** Medium.
- **Targets:** `S15/S19/S23` missing from artifact overlap, `S05` hidden-risk, weak sites `S18/S09/S03`.
- **First smoke:** compare non-Aves/no-train branch with group-DRO/site-adversarial sampling versus normal training.
- **Promotion condition:** improves worst-site and leave-site metrics without average/taxon collapse.

## Recommended immediate queue

1. Build a **soundscape-native non-Aves/no-train embedding-head smoke**. This should combine ideas 1 and 4 and optionally test site-balanced sampling from idea 5.
2. In parallel or immediately after, build the **negative/background data audit** for idea 2 before any model work.
3. If the first branch produces usable raw outputs, run the ensemble workbench against anchor and v616 with site/file/taxon gates and duplicate/correlation checks.
4. Do not submit to Kaggle from this phase. Treat results as branch discovery only.
