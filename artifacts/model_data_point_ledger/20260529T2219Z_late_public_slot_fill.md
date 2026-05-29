# 2026-05-29T22:19Z late public source slot fill v636-v640
## Live status
- UTC slots before submit: `0/5`; hours to reset: `1.68`.
- Best public LB before submit: `0.949` (v616 tied baseline; v634 tied latest public-source fill).
- No active local/trainer BirdCLEF jobs; late-day policy active, so guarded source-code rerun candidates were valid slot fills.

## Submitted candidates
- **v636-late-public-source-20260529T2219Z** — ref `53165843`; source `mohamadmatali/bc2026-claude-mtoshi-947-repro-fork`; status pending. Dry-run `3x235`, finite/nonconstant, uniq100 `92`, hash `e5c937e6d87cb4fc`; min/max/mean `0.4466501`/`0.53982186`/`0.498501`.
- **v637-late-public-source-20260529T2219Z** — ref `53165844`; source `hassan1417/bc2026-claude-yaroslav-946-replay-fork`; status pending. Dry-run `3x235`, finite/nonconstant, uniq100 `94`, hash `f0947cc50457ccdd`; min/max/mean `0.50208336`/`0.5091626`/`0.503889`.
- **v638-late-public-source-20260529T2219Z** — ref `53165846`; source `sultanalgizani/bc2026-claude-5branch-ensemble-fork`; status pending. Dry-run `3x235`, finite/nonconstant, uniq100 `94`, hash `d0545f2c89ce36b8`; min/max/mean `0.2602615`/`0.34930405`/`0.289221`.
- **v639-late-public-source-20260529T2219Z** — ref `53165850`; source `shahadaljayzani/bc2026-claude-mtoshi-941-fork`; status pending. Dry-run `240x235`, finite/nonconstant, uniq100 `94`, hash `6dcda7328bb22532`; min/max/mean `0.004166667`/`1.0`/`0.504008`.
- **v640-late-public-source-20260529T2219Z** — ref `53165851`; source `hanijezo/bc2026-claude-imaad-perch-protossm-fork`; status pending. Dry-run `3x235`, finite/nonconstant, uniq100 `91`, hash `c166a6c8e22078c1`; min/max/mean `0.47800776`/`0.5619778`/`0.501614`.

## Critic / verifier decision
- Critic: no repo-owned candidate was verifier-grade; with <2h to reset and `0/5` slots used, preserving slots was lower value than source-clean exploratory reruns. Candidate ordering favored distinct, nonduplicate public source families with known/claimed 0.947/0.946/ensemble signal before lower-info fallbacks.
- Verifier: preflight required COMPLETE public kernels, hidden-test/source markers, `submission.csv`, finite/nonconstant 235-column public-session output, nonduplicate dry-run hashes against v621-v635, and unique descriptions. All five submitted candidates passed. These are source-code submissions; Kaggle reruns hidden test, not static CSV uploads.

## Decision
- Submission-grade under late-day exploratory slot-fill policy only; score pending. Monitor v636-v640 and update canonical table when completed.

## Artifact
- Submission/preflight artifact: `artifacts/public_kernels_20260529_late_scout/submit_v636_v640_late_fill_20260529.json`
