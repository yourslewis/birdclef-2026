#!/usr/bin/env python3
"""Queue #2: per-class / E-weak-only blend of the ConvNeXt-distill stream vs frontier E.

The diversity scout's global optimizer applies ONE scalar weight across all 42 valid
classes, so the distilled stream's E-weak competence (weak-AUC 0.80) is diluted by
E-strong columns and the optimizer picks w=0. This script tests the hypothesis that a
TARGETED per-class blend -- mixing the distilled stream ONLY into E-weak columns where it
is competent, leaving E-strong columns untouched -- lifts overall macro-AUC and is robust
under leave-site / leave-file bootstrap.

Reuses the scout's exact E construction and metrics so results are comparable.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import birdclef_diversity_scout as ds  # noqa: E402

PROTO = ROOT / "artifacts/source_winner_audit_20260531T0416Z/session_outputs/v644_yaroslav_0950/submission_protossm.csv"
SED = ROOT / "artifacts/source_winner_audit_20260531T0416Z/session_outputs/v644_yaroslav_0950/submission_sed.csv"
CAND = ROOT / "artifacts/diversity_scout/convnext_distill_20260601/E_convnext_distill.csv"
LABELS = ROOT / "data/train_soundscapes_labels.csv"
OUT = ROOT / "artifacts/diversity_scout/convnext_distill_perclass_20260601"


def boot_q05(y, base, cand, valid, groups, n_boot, seed):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(groups.astype(str))))
    lifts = []
    for _ in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in sampled])
        ac, _ = ds.macro_auc(y[idx], cand[idx], valid)
        ab, _ = ds.macro_auc(y[idx], base[idx], valid)
        if ac is not None and ab is not None:
            lifts.append(ac - ab)
    arr = np.asarray(lifts)
    return {"n": int(arr.size), "mean": float(arr.mean()), "q05": float(np.quantile(arr, 0.05)),
            "p_gt_0": float(np.mean(arr > 0))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    proto = pd.read_csv(PROTO)
    cols = ds.pred_cols(proto)
    row_ids = proto["row_id"].astype(str)
    sed = ds.align_like(pd.read_csv(SED), row_ids, cols, "sed")

    proto_rank = ds.rank_cols(proto[cols].to_numpy(np.float32))
    sed_rank = ds.rank_cols(sed[cols].to_numpy(np.float32))
    e_rank_full = ds.rank_cols((0.60 * proto_rank + 0.40 * sed_rank).astype(np.float32))

    sites_full = row_ids.map(ds.site_id).to_numpy()
    files_full = row_ids.map(ds.file_id).to_numpy()

    labels_wide = ds.load_long_labels(LABELS, cols)
    matched = row_ids.isin(labels_wide["row_id"].astype(str))
    midx = np.flatnonzero(matched.to_numpy())
    mrows = row_ids.iloc[midx].tolist()
    y = labels_wide.set_index("row_id").loc[mrows, cols].to_numpy(np.uint8)
    valid = ds.valid_class_indices(y)

    e_m = e_rank_full[midx]
    e_auc, e_valid_n = ds.macro_auc(y, e_m, valid)
    e_class_auc = ds.per_class_auc(y, e_m, valid)
    weak_classes = sorted(valid, key=lambda j: e_class_auc.get(j, 1.0))[: max(1, len(valid) // 3)]

    cand = ds.align_like(pd.read_csv(CAND), row_ids, cols, "convnext_distill")
    c_rank_full = ds.rank_cols(cand[cols].to_numpy(np.float32))
    c_m = c_rank_full[midx]

    sites_m = sites_full[midx]
    files_m = files_full[midx]

    # Per-class AUC of candidate on weak classes
    c_class_auc = ds.per_class_auc(y, c_m, valid)
    weak_detail = []
    for j in weak_classes:
        weak_detail.append({"col": cols[j], "E_auc": e_class_auc.get(j), "cand_auc": c_class_auc.get(j),
                            "cand_better": (c_class_auc.get(j, 0) > e_class_auc.get(j, 1))})
    n_cand_better = sum(1 for d in weak_detail if d["cand_better"])

    results = {"frontier_E_auc": e_auc, "valid_classes": e_valid_n, "weak_classes": len(weak_classes),
               "weak_cols_cand_better": n_cand_better, "weak_detail": weak_detail, "sweeps": []}

    # Strategy A: blend candidate into ALL weak columns at weight w (E-strong untouched).
    # Strategy B: blend candidate ONLY into weak columns where cand_auc > E_auc.
    weak_better = [j for j in weak_classes if c_class_auc.get(j, 0) > e_class_auc.get(j, 1)]
    weights = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    best_overall = {"strategy": None, "weight": 0.0, "auc": e_auc, "lift": 0.0, "cols": 0}

    for strat_name, target_cols in [("A_all_weak", weak_classes), ("B_weak_better", weak_better)]:
        for w in weights:
            blended = e_m.copy()
            for j in target_cols:
                blended[:, j] = (1.0 - w) * e_m[:, j] + w * c_m[:, j]
            blended = ds.rank_cols(blended)
            auc, _ = ds.macro_auc(y, blended, valid)
            lift = auc - e_auc
            entry = {"strategy": strat_name, "weight": w, "target_cols": len(target_cols),
                     "auc": auc, "lift": lift}
            results["sweeps"].append(entry)
            if lift > best_overall["lift"]:
                best_overall = {"strategy": strat_name, "weight": w, "auc": auc, "lift": lift,
                                "cols": len(target_cols)}

    results["best_overall"] = best_overall

    # Bootstrap the single best per-class blend under leave-site / leave-file
    if best_overall["strategy"] is not None and best_overall["lift"] > 0:
        target_cols = weak_classes if best_overall["strategy"] == "A_all_weak" else weak_better
        w = best_overall["weight"]
        blended = e_m.copy()
        for j in target_cols:
            blended[:, j] = (1.0 - w) * e_m[:, j] + w * c_m[:, j]
        blended = ds.rank_cols(blended)
        site_q05 = boot_q05(y, e_m, blended, valid, sites_m, 200, 17)
        file_q05 = boot_q05(y, e_m, blended, valid, files_m, 200, 29)
        results["best_site_q05"] = site_q05
        results["best_file_q05"] = file_q05
        results["gate_pass_nonharmful"] = bool(site_q05["q05"] >= 0 and file_q05["q05"] >= 0)
        results["gate_pass_strong"] = bool(site_q05["q05"] > 0 and file_q05["q05"] > 0)
    else:
        results["best_site_q05"] = {"q05": 0.0}
        results["best_file_q05"] = {"q05": 0.0}
        results["gate_pass_nonharmful"] = False
        results["gate_pass_strong"] = False

    (OUT / "perclass_blend_summary.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: results[k] for k in ["frontier_E_auc", "valid_classes", "weak_classes",
          "weak_cols_cand_better", "best_overall", "best_site_q05", "best_file_q05",
          "gate_pass_nonharmful", "gate_pass_strong"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
