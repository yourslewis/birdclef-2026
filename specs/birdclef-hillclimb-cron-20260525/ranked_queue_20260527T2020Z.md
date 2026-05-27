# BirdCLEF 2026 ranked queue — 2026-05-27 20:20 UTC

## Live status
- Kaggle Bearer API check at 20:17 UTC: latest scored submissions v621/v622/v623 tied `0.949`; v624 `0.943`; v625 `0.948`; v616 remains tied best `0.949`.
- UTC slots used today: `0/5`; time to reset at check: about `3.7h`.
- Slot decision: **no submission yet**. It is still outside the `<3h` late-fill window and no verifier-grade package is ready.
- Active BirdCLEF jobs: none before this run; unrelated trainer jobs were present but GPU 1 was free.

## ClawTeam synthesis / critic decision
- Scout/critic still ranks true hidden-test package/eval for **PANNs all-class no-file sequence** as the best near-term promotion lane.
- Coordinator accepted a bounded **deeper soundscape-native all-class B0** data point this run because no package was ready and the model-data-point policy says to train distinct families rather than idle.
- Red-team caution: native B0's fold mean is competitive, but pooled OOF is very weak, so it is **not** a submission candidate without a separate hidden-test package/audit story.

## Model trained/evaluated this run
- `soundscape-native-b0-losite-allcls-ep4-20260527`
  - Row AUC `0.636161`; no-train `0.626084`; non-Aves `0.618037`; file-MIL `0.673756`.
  - Baseline delta vs PANNs no-file: row `-0.011655`, file-MIL `+0.003033`.
  - Export: TS/ONNX passed on trainer; finite/nonconstant OOF predictions `1410x234`.
  - Decision: keep data point; no direct submission.

## Ranked next queue
1. **PANNs all-class no-file true hidden-test package/eval** — highest row AUC (`0.647816`) and strong file-MIL (`0.670723`); current sidecar audits are negative, so needs real package path, not another proxy sidecar sweep.
2. **No-call / acoustic-background protocol** — most distinct unmined behavior lane; focus on background/no-call features, not narrow 72-label wrapper repeats.
3. **Native B0 all-class follow-up only if package path emerges** — this run's native CNN is close on fold mean and file-MIL, but pooled OOF makes it risky; use as diagnostic/ensemble diversity, not primary.
4. **Fused DyMN10+PANNs file-MIL diagnostic** — best file-MIL so far (`0.675982`), row weaker and proxy sidecar loses vs v616; keep as secondary clue.
5. **Late source-clean public slot fill after `<3h`** — if no verifier-grade package is ready near reset, scan fresh source/kernel candidates and fill remaining valid slots after schema/runtime/dedup checks.

## Top-5 comparable all-class model table
| Rank | Model | Row AUC | File-MIL | Decision |
|---:|---|---:|---:|---|
| 1 | PANNs all-class r2 no-file | 0.647816 | 0.670723 | best package target |
| 2 | PANNs all-class r2 file-context | 0.642202 | 0.652651 | keep; lower than no-file |
| 3 | Native B0 all-class LOSO | 0.636161 | 0.673756 | new data point; no submit |
| 4 | DyMN10 all-class r2 no-file | 0.597633 | 0.635285 | backup package idea |
| 5 | Fused DyMN10+PANNs r2 no-file | 0.596642 | 0.675982 | file-MIL clue only |

## Artifacts
- Ledger: `artifacts/model_data_point_ledger/20260527T2020Z_soundscape_native_b0_allclass.md`
- Canonical table: `artifacts/model_data_point_ledger/performance_table.md`
- JSONL row: `artifacts/model_data_point_ledger/performance_table.jsonl`
- Metrics root: `artifacts/soundscape_native_losite/soundscape-native-b0-losite-allcls-ep4-20260527/`
