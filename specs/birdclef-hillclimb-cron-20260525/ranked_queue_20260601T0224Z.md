# BirdCLEF hill-climb ranked queue — 2026-06-01 02:24 UTC

## Live status
- Bearer API submission check: latest scored submissions are `v653=0.947`, `v654=0.949`, `v655=0.949`, `v651=0.941`, `v652=0.948`; public best remains `0.950` from `v644`/`v647`.
- UTC daily slots: `0/5` used for 2026-06-01; ~21.7h to reset at the start of this run.
- Active work check: no local/trainer BirdCLEF training jobs found; all latest BirdCLEF Kaggle submissions are complete. Trainer has unrelated HSTU/LRM GPU work, so this run stayed local/no-slot.

## Role synthesis
- **Coordinator:** early UTC day; no spend on below-best scalar EoS8 neighbors. Continue data-point training/evaluation in train_soundscapes sequence/file/site lane.
- **Data & Feature Scientist:** file-MIL signal from DyMN10 is real but did not transfer through raw sidecars. Test file-level calibration/mapping before another neural run.
- **Validation & Metrics Scientist:** use leave-one-site fold mean for comparable row/file metrics; pooled row AUC is site-prior-confounded and tracked only as a caveat.
- **Prediction & Ensemble Analyst:** selected mappings must also pass 72→234 v616 proxy sidecar audit before any slot consideration.
- **Experiment Engineer:** implemented `scripts/birdclef_file_level_calibration_diagnostic.py` and ran selected sidecar audits.
- **Critic / Red Team:** improvement in local row/file validation is not enough; require hidden-safe package or robust v616-sidecar lift.
- **Verifier / Skeptic:** sidecars finite/schema-valid, but leave-site OOF proxy artifacts are not submission-grade; no slot.

## Current run result
- Evaluated file-level calibration/mapping candidates from PANNs row-only OOF + DyMN10/fused file evidence.
- Best comparable row mapping: `pannsrow__dymn_filemean__a35`, row AUC `0.687350`, file-MIL `0.757145`, no-train `0.599442`.
- Delta: vs PANNs 72 row-only `+0.012865` row and `+0.065989` file-MIL; vs DyMN10 filectx `+0.011441` file-MIL for the best-row mapping.
- Best grid file-MIL was `0.784044` (`pannsrow__dymn_filemax__a50`), but its sidecar did not pass v616 proxy.
- Best selected 72→234 sidecar audit: `0.991112` local AUC / 42 valid, lift vs v616 `-0.002368`, `submit_approved=false`.
- Decision: **keep landscape point; no Kaggle submission.** This is the strongest row/file validation clue in the targeted 72-label lane, but it still needs a hidden-safe inference/package formulation.

## Refreshed ranked queue

1. **Hidden-safe file-calibration package / student teacher path — selected next.**
   - Rationale: the file-level mapping improved row and file-MIL validation, but OOF sidecars cannot be submitted. Convert the mapping into a reproducible hidden-test path: either package PANNs row head + DyMN10 file evidence inference, or distill mapping targets into a compact train_soundscape-native student.
   - Gate: must produce finite 234-class predictions and improve v616 proxy sidecar or a stronger private verifier; no static/OFF proxy output.

2. **Deeper soundscape-native calibrated student with file-level targets.**
   - Rationale: prior native B0 soft1279 variants were weak, but the new teacher mapping gives better row/file targets than raw soft labels. Try only with site/file gates and regularization.
   - Gate: beat `0.687350` row or show robust sidecar lift; otherwise reject.

3. **Class/file movement diagnostic for why sidecar remains below v616.**
   - Rationale: row/file validation improved while v616 proxy sidecar did not. Need class/site attribution to identify classes where file mapping helps vs harms before packaging.
   - Gate: find a selector that has positive site/file CV lift and avoids broad rank-correlation-only drift.

4. **Public source scout only for source-clean candidates materially distinct from v644/v647.**
   - Rationale: EoS8 scalar frontier (`v651`-`v655`) failed to beat 0.950. Avoid early-day slots on nearby public-source perturbations.

5. **Late UTC slot fills.**
   - Rationale: still allowed by policy near reset, but only after verifier/schema/dedup checks and only if no higher-grade candidate emerges.

## Artifacts
- Ledger: `artifacts/model_data_point_ledger/20260601T0220Z_file_level_calibration_diagnostic.md`
- Metrics/audit: `artifacts/model_data_point_ledger/20260601T0220Z_file_level_calibration_diagnostic/metrics.json`, `.../audit_summary.json`
- Canonical performance table/jsonl updated.
