# Ranked queue — BirdCLEF ClawTeam hill-climb — 2026-05-26 20:25 UTC

## Live status
- Public LB best remains `0.949`; v616 is still the tied repo-owned baseline.
- Latest scored submissions: v617 `0.949`, v618 `0.946`, v619 `0.944`, v620 `0.949`; no 2026-05-26 UTC submissions yet.
- UTC slots used: `0/5`; time to reset at live check: ~`3.7h`.
- Active jobs: no active BirdCLEF local/trainer jobs. Trainer GPU memory is mostly occupied by unrelated LRM inference processes, but GPU utilization was `0%`; this run used only a light audit.
- Public scout web refresh: no clean fresh >0.949 BirdCLEF 2026 public lead surfaced in quick web searches.

## Run decision
Mid/late-day but not yet inside the `<3h` late-fill window. No verifier-grade, competition-format, nonduplicate candidate was ready. Instead of idling, evaluated the next exact action from the 18:20 run: a cautious 72→234 wrapper/audit for the best train_soundscape sequence heads.

## Ranked candidate queue

1. **Late-day guarded slot-fill review near reset** — highest slot-policy priority once `<3h` remains. Re-scan candidate pool and submit only source-clean, nonduplicate, non-malformed candidates. Current sequence wrapper is rejected and should not be used for this.
2. **True hidden-safe 234-class DyMN10/AudioSet sidecar package** — convert the sequence/AudioSet signal into a real inference path for hidden test, not leave-site OOF proxy rows. Requires multi-site validation and v616 audit before slot.
3. **Multi-site AudioSet/DyMN10 sidecar reformulation** — broader no-call/non-Aves features or 234-class wrapper with site/file gates. DyMN10 remains the best frozen AudioSet embedding data point (`0.568586` S08; context MLP `0.601355` LOSO row / `0.632127` file-MIL).
4. **Context-MLP robustness or S22/no-train cap diagnostics** — useful for science, but current wrapper lost to v616 (`-0.00219` best combo), so only continue if it informs a real package.
5. **G124/V2S hard-confidence/power ablation** — operational path exists, but prior local sidecar lift vs v616 was only `+0.00000339`; lower priority unless no better package appears.
6. **Broader no-call/background negative branch** — broad mask coverage is fixed, but aux hurt the matched B0 control; revisit only with better negative protocol.
7. **Direct plateau-family public replays / scalar tweaks** — rejected unless late-fill verifier finds a genuinely nonduplicate, rule-clean, high-info source candidate.

## Model/evaluation result this run
- Evaluated two sequence sidecar wrappers and combo recipes using `scripts/birdclef_soundscape_sequence_sidecar_audit.py`.
- Best recipe: `seq_context02_r201` (`97%` anchor + `2%` context + `1%` r2) local macro AUC `0.991293583`, lift vs anchor `+0.000903076`, lift vs v616 `-0.002187085`.
- Best single sidecar: `seq_context_w01` local macro AUC `0.991279099`, lift vs v616 `-0.002201568`.
- Decision: reject as slot candidate; useful comparison-grade wrapper data point only.

## Artifacts
- Ledger: `artifacts/model_data_point_ledger/20260526T2025Z_soundscape_sequence_sidecar_audit.md`
- Audit root: `artifacts/soundscape_sequence_sidecar_audit/20260526T2025Z/`
- Canonical performance table updated: `artifacts/model_data_point_ledger/performance_table.md` and `.jsonl`
