# Ranked Queue — BirdCLEF Hill-Climb Cron — 2026-05-25 22:43 UTC

## Live status
- Current public LB best: `0.949`.
- Latest verified best/tied baseline: `v616` scored `0.949`.
- UTC daily slots before late fill: `1/5` used (`v616`).
- Time to reset at submit: about `1.28h`, so late-day fill policy applies.
- Active local/trainer jobs: none found.
- Git branch: `feature/birdclef-20260524-20utc-v612-submit`.

## Critic / opportunity-cost decision
- The high-priority new-branch lanes (EfficientAT/PANNs, non-Aves/no-train specialist, broader OOF negative/no-call, 20s temporal branch) were not submission-ready within the final UTC window.
- The last completed no-slot lanes (v616/SYD52p/per-class/SED/OOF-teacher/negative aux) did not produce a verifier-grade candidate above `0.949`.
- With <3h to reset and 4 unused slots, preserving slots would violate the current cron policy. Use highest-ranked source-clean exploratory candidates that pass guards.

## Verifier guard applied
For submitted candidates, preflight required:
- source kernel `COMPLETE`, no failure;
- source includes competition/test-soundscape path markers and Kaggle competition data source;
- `submission.csv` exists in public session output;
- public final is finite, nonconstant, 235 columns, no ragged rows or bad values;
- duplicate submission description absent;
- no static public-output-only upload path used — these are Kaggle code submissions, rerun by Kaggle.

## Ranked candidates

### 1. v617 — Nina EoS7 sz sidecar source — SUBMITTED
- Source: `nina2025/birdclef-2026-eos-7-sz`, version `4`.
- Expected LB potential: medium; EoS/sidecar plateau-family, but fresh EoS7 variant and likely strongest valid late-day public source.
- Information value: medium; tests EoS7 + sidecar behavior after v616/Jung21/SED tie.
- Evidence level: exploratory/source-preflight.
- Risk: likely near plateau; public dry-run has only 3 sample rows, but source has hidden/test path and raw branch outputs.
- Verifier: COMPLETE, no failure, nonconstant final, raw ProtoSSM/SED/BirdNET sidecars present.
- Submit decision: submit late-day exploratory.

### 2. v618 — Kruzzcc Nina EoS4 BirdNET source — SUBMITTED
- Source: `kruzzcc/bc26-nina-eos4-fixed`, version `2`.
- Expected LB potential: low-medium; EoS4 + BirdNET blend, valid hidden path.
- Information value: medium; BirdNET branch behavior distinct enough for late-day filler.
- Evidence level: exploratory/source-preflight.
- Risk: plateau-family and may tie/drop; acceptable only because slots would expire.
- Verifier: COMPLETE, no failure, nonconstant final, nonconstant raw BirdNET/ProtoSSM/SED branches.
- Submit decision: submit late-day exploratory.

### 3. v619 — Kruzzcc Mtoshi UMAP BirdNET source — SUBMITTED
- Source: `kruzzcc/bc26-mtoshi-umap-bn-a`, version `1`.
- Expected LB potential: low-medium; Mtoshi/UMAP/BirdNET source variant.
- Information value: medium; separate BirdNET/UMAP branch signal vs EoS/v616 cluster.
- Evidence level: exploratory/source-preflight.
- Risk: likely below/near plateau; acceptable as late-day source-clean slot use.
- Verifier: COMPLETE, no failure, finite/nonconstant public final, raw branch files present.
- Submit decision: submit late-day exploratory.

### 4. v620 — Kazuhiro Karnak rank-fusion source — SUBMITTED
- Source: `kazuhirokuriyama/birdclef2026-karnak-rank-fusion`, version `2`.
- Expected LB potential: low-medium; rank-fusion of known strong branches.
- Information value: low-medium; late-day test of Karnak rank fusion not previously submitted under this source.
- Evidence level: exploratory/source-preflight.
- Risk: near-duplicate/plateau; chosen after stricter candidates were malformed/blocked.
- Verifier: COMPLETE, no failure, public final finite/nonconstant, raw branch outputs present.
- Submit decision: submit as final slot filler.

## Rejected / blocked during late scout
- `muhammadsaadalvi/birdclef-2026-wildsound-v8`: ERROR/no outputs; reject despite high branch diversity.
- `udaken10/new-preprocessing-90`: CANCEL_ACKNOWLEDGED/no competition final; reject.
- `sultanalgizani/mohamadmatali/hassanalgizani bc2026-teacher-p952-exp070-*`: teacher/cache artifacts with `7992` train-soundscape rows, no test-soundscape final path; reject as not competition submission.
- `kijiang/birdclef2026-v355..v358` and `gendaijin/birdclef2026-day0525-nina-eos6-bz`: public `submission.csv` malformed (`243x235` with bad values); reject direct submission.
- `samejimatink0/birdclef-2026-hgnetv2-b0-baseline-inference`: final has bad values; reject.
- `ommodi07/birdclef2026`: constant all-zero final; reject.
- `ykuroka/birdclef-2026-iter-pseudo-oof`: zero-row `submission.csv`; reject.
- `tulayppppp/my-efficientnet-b0-weights`: wrong shape/mock output; reject.
- `viktoriiahranadzer/birdclef-inference`: final has bad values; reject despite EfficientAT marker.

## Submission refs
- `v617`: ref `53032516`, status pending at immediate post-submit check.
- `v618`: ref `53032520`, status pending at immediate post-submit check.
- `v619`: ref `53032523`, status pending at immediate post-submit check.
- `v620`: ref `53032524`, status pending at immediate post-submit check.

## Next exact action
After reset / next cron: monitor `v617`-`v620` scores. If any beats `0.949`, inspect source/branch movement and build a repo-owned hidden-safe confirmer. If all tie/drop, continue with truly new-branch no-slot work: EfficientAT/PANNs event/no-call or broader negative/no-call cache, not more EoS/ProtoSSM/SED repeats.
