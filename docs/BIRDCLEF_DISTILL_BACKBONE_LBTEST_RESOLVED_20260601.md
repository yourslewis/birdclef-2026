# BirdCLEF 2026 — v656 Distinct-Backbone Distill 3-Stack LB Probe (RESOLVED)

Date: 2026-06-01 (PDT). Lane: ClawTeam competition-research hill-climb.

## HEADLINE CORRECTION

The prior run (`docs/BIRDCLEF_DISTILL_BACKBONE_LBTEST_20260601.md`, commit abe7d87) declared
the distinct-backbone TorchScript weights **"stranded on an offline GPU host"** and spent **0
slots**. **That blocker was FALSE.** The trainer host `yourslewis@192.168.0.10` is reachable,
has torch 2.10.0+cu128, and holds all three distill TorchScript exports:

- `convnext_w07.pt` (60.7 MB) — ConvNeXt-nano soft1279-teacher distill w0.7
- `convnext_w085.pt` (60.7 MB) — ConvNeXt-nano distill w0.85
- `regnety008.pt` (23.4 MB) — RegNetY-008 distill

All three load and produce valid (clip_logits 234, frame_logits) on `(B,160,frames)` logmel
input. Verified live on the trainer.

## WHAT WAS DONE

1. Bundled 3 TS weights + `labels.json` (234-label order, exact match to
   `sample_submission` columns) and uploaded as Kaggle dataset
   `yourslewis/bc26-distill-backbone-stack-v1` (144 MB, private, v1) via Bearer `/blobs/upload`.
2. Authored standalone CPU inference kernel `kaggle-kernels/v656-distill-backbone-stack/script.py`:
   - mel front-end reproduced from training (`sr 32000, n_fft 1024, hop 512, n_mels 160, 5s, 12 windows/file`)
   - per-window: run 3 backbones → column-rank-normalize each → mean across backbones → sigmoid-range [1e-6, 1-1e-6]
   - robust bundle path resolution via `rglob` (v1 failed on hardcoded mount path; v2 fixed)
   - dry-run aligns to sample_submission schema; real path emits 12 rows/file × 235 cols
3. Smoke-tested on trainer: real path → 234/234 nonconstant cols, finite, 1.3s/3 files.
4. Pushed kernel v2 → COMPLETE on Kaggle (dry-run on 3 train files, 56.7s wall).
5. Submitted to code competition: `CreateCodeSubmission` RPC on `www.kaggle.com` (the
   `api.kaggle.com` gRPC host returns 401 for KGAT tokens — use www host). **ref 53264588.**

## STATUS AT REPORT TIME

- **Slots: 1/5 used** (was 0/5). Submission `53264588` PENDING (hidden-test rerun in progress).
- This is the **first genuinely distinct base-embedding family** (non-Perch-ProtoSSM) ever to
  reach the BC26 hidden LB. Candidate DEV evidence on record: weak-class AUC 0.8319,
  rank_decorr 0.7374, DEV 0.002447 (highest diversity stream measured).
- Expectation: standalone score likely well below the 0.950 frontier E (these distill streams
  are weak alone). The value is the **honest live read** on a distinct foundation and the
  hidden-test domain — directly testing whether the 42/234-valid proxy ceiling (the documented
  binding limiter that pinned blend weight to 0) was masking real transfer, since the proxy is
  a known two-way liar.

## NEXT EXACT ACTION

- When `53264588` scores: record LB in `performance_table.md`/`.jsonl`.
  - If it lands competitively (e.g. >0.90) → the distinct foundation has real hidden-test
    competence; next slot = small-weight rank-mix of this stack WITH the 0.950 frontier E
    (graft distill ranks into a v655-style EoS8 kernel at w≈0.02–0.05).
  - If it lands weak (e.g. <0.85) → confirms standalone distill insufficiency; the diversity
    value only materializes as a blend member; pursue the in-kernel blend probe directly.
- Remaining 4 slots today: prefer DEV-passing diverse/representation candidates; the in-kernel
  distill×E blend is the highest-info follow-up once the standalone read is in.
