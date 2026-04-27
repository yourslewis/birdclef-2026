#!/usr/bin/env python3
"""
BirdCLEF 2026 — v193: Offline Hyperparameter Sweep
====================================================
Takes the v111 script, runs OOF evaluation only, and sweeps
post-processing hyperparameters to find the optimal config.

This modifies ONLY post-processing params that don't affect training:
- PER_CLASS_TEMPS temperature values (0.5-1.0)
- POWER_GAMMA (0.7-1.0)
- FILE_CONTEXT_ALPHA (0.0-0.3)
- QUANTILE_MIX_ALPHA (0.0-1.0)
- PROTOSSM_ENSEMBLE_WEIGHT (0.3-0.7)
- PROBE_ALPHA (0.2-0.6)
- SMOOTH_TEXTURE_ALPHA (0.1-0.5)
- GAUSSIAN_WEIGHTS kernel width

All of these can be tuned post-hoc on OOF predictions without retraining.
"""

import itertools
import numpy as np
import json
from pathlib import Path

# We'll reconstruct the OOF evaluation pipeline from the v111 script
# but only the post-processing portion, since probe_scores and protossm_scores
# are the two ingredients that get combined.

# First, let's check if we can load cached OOF arrays
WORK_DIR = Path('/Users/yourslewis/.openclaw/workspace-don/kaggle/birdclef-2026')
CACHE_SEARCH = [
    WORK_DIR / 'v111' / 'oof_cache.npz',
    WORK_DIR / 'oof_cache.npz',
]

print("=== BirdCLEF v193: Offline HP Sweep ===")
print("This script needs OOF predictions to sweep post-processing params.")
print("We need to first generate OOF predictions from v111 locally.")
print()

# The approach: we modify v111 to ONLY do the OOF evaluation (no test inference)
# and save the intermediate arrays, then sweep params on those arrays.

# Since we can't run the full v111 locally (requires Kaggle datasets/Perch model),
# let's instead identify which params can be safely grid-searched by modifying
# the Kaggle notebook itself to do an internal sweep before writing submission.csv

# Strategy: Inject a grid search block into the v111 script AFTER OOF computation
# but BEFORE test inference, that finds optimal params on OOF, then applies them.

print("STRATEGY: Inject in-notebook OOF grid search into v111")
print("This adds ~10 seconds of compute (pure numpy) but finds optimal params")
print("on the training data before applying them to the hidden test set.")
print()

# Key post-processing params to sweep (no retraining needed):
sweep_config = {
    'POWER_GAMMA': [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00],
    'FILE_CONTEXT_ALPHA': [0.00, 0.05, 0.10, 0.15, 0.20, 0.25],
    'QUANTILE_MIX_ALPHA': [0.3, 0.4, 0.5, 0.6, 0.7],
    'PROTOSSM_ENSEMBLE_WEIGHT': [0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
    'PROBE_ALPHA': [0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    'SHARP_TEMP': [0.5, 0.6, 0.7, 0.8, 0.9],
}

# Total combos (full grid would be too large)
total = 1
for k, v in sweep_config.items():
    total *= len(v)
    print(f"  {k}: {len(v)} values -> {v}")
print(f"\nFull grid: {total:,} combos (too large for in-notebook)")
print("Will use coordinate descent (one param at a time) instead.")
print("Coordinate descent: ~", sum(len(v) for v in sweep_config.values()), "evaluations")
print()

# Generate the injection code block
injection_code = '''
# === V193: In-Notebook OOF Hyperparameter Sweep (Coordinate Descent) ===
print("\\n=== V193: OOF Hyperparameter Sweep ===")
_sweep_start = time.time()

# Reconstruct OOF probe + protossm scores for sweep
# (These are already computed above in the OOF section)

# OOF ProtoSSM predictions
oof_protossm = np.zeros_like(oof_base)
for model_i, protossm_model in enumerate(protossm_models):
    seed_scores = np.zeros_like(oof_base)
    if n_full_files > 0:
        emb_oof_files = emb_full[:n_full_files * WINDOWS_PER_FILE].reshape(
            n_full_files, WINDOWS_PER_FILE, -1)
        scores_oof_files = scores_full_raw[:n_full_files * WINDOWS_PER_FILE].reshape(
            n_full_files, WINDOWS_PER_FILE, -1)
        with torch.no_grad():
            X_oof_t = torch.FloatTensor(emb_oof_files)
            logits_oof_t = torch.FloatTensor(scores_oof_files)
            ssm_pred = protossm_model(X_oof_t, logits_oof_t)
            seed_scores[:n_full_files * WINDOWS_PER_FILE] = ssm_pred.numpy().reshape(-1, N_CLASSES)
    oof_protossm += seed_scores
oof_protossm /= len(protossm_models)

# OOF Probe predictions
oof_probe = oof_base.copy()
for cls_idx, clf in probe_models.items():
    proto_sim = None
    if cls_idx in CLASS_PROTOTYPES:
        proto_sim = cosine_sim_to_prototype(Z_FULL, CLASS_PROTOTYPES[cls_idx])
    family_name = CLASS_FAMILY.get(cls_idx, 'Unknown')
    family_idxs = FAMILY_IDX_MAP.get(family_name, np.array([]))
    other_family = family_idxs[family_idxs != cls_idx]
    family_mean = oof_base[:, other_family].mean(axis=1) if len(other_family) > 0 else None
    X = build_class_features(
        Z_FULL, scores_full_raw[:, cls_idx],
        oof_prior[:, cls_idx], oof_base[:, cls_idx],
        proto_sim_col=proto_sim, family_mean_col=family_mean)
    try:
        proba = clf.predict_proba(X)
        if proba.shape[1] == 2:
            pred = np.log(proba[:, 1] / (proba[:, 0] + 1e-8) + 1e-8)
        else:
            pred = np.zeros(len(X))
    except Exception:
        pred = np.zeros(len(X))
    oof_probe[:, cls_idx] = pred.astype(np.float32)

# Now sweep post-processing params using coordinate descent
def evaluate_config(probe_scores_raw, protossm_scores, Y_true,
                    probe_alpha, ew, qma, sharp_temp, power_gamma, fca):
    """Evaluate a full post-processing config on OOF data."""
    # Step 1: Blend probe with base
    ps = (1 - probe_alpha) * oof_base + probe_alpha * probe_scores_raw
    # Step 2: Ensemble
    simple = (1 - ew) * ps + ew * protossm_scores
    rank = rank_average_ensemble([ps, protossm_scores], weights=[1 - ew, ew])
    final = qma * simple + (1 - qma) * rank
    # Step 3: Temperature
    temps = np.ones(N_CLASSES, dtype=np.float32)
    for idx in SHARPENED:
        temps[idx] = sharp_temp
    temps[18] = min(sharp_temp + 0.1, 1.0)
    final = final / temps[np.newaxis, :]
    # Step 4: Gaussian smooth
    final = gauss_smooth_logits(final)
    # Step 5: Sigmoid + file context + power
    probs = sigmoid(final)
    probs = file_context_boost(probs, alpha=fca)
    probs = np.clip(probs, 1e-8, 1.0 - 1e-8)
    probs = np.power(probs, power_gamma)
    return macro_auc(Y_true, probs)

# Coordinate descent
best_params = {
    'PROBE_ALPHA': PROBE_ALPHA,
    'PROTOSSM_ENSEMBLE_WEIGHT': PROTOSSM_ENSEMBLE_WEIGHT,
    'QUANTILE_MIX_ALPHA': QUANTILE_MIX_ALPHA,
    'SHARP_TEMP': 0.7,
    'POWER_GAMMA': POWER_GAMMA,
    'FILE_CONTEXT_ALPHA': FILE_CONTEXT_ALPHA,
}

param_grids = {
    'PROBE_ALPHA': [0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    'PROTOSSM_ENSEMBLE_WEIGHT': [0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
    'QUANTILE_MIX_ALPHA': [0.3, 0.4, 0.5, 0.6, 0.7],
    'SHARP_TEMP': [0.5, 0.6, 0.7, 0.8, 0.9],
    'POWER_GAMMA': [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00],
    'FILE_CONTEXT_ALPHA': [0.00, 0.05, 0.10, 0.15, 0.20, 0.25],
}

best_auc = evaluate_config(oof_probe, oof_protossm, Y_FULL,
    best_params['PROBE_ALPHA'], best_params['PROTOSSM_ENSEMBLE_WEIGHT'],
    best_params['QUANTILE_MIX_ALPHA'], best_params['SHARP_TEMP'],
    best_params['POWER_GAMMA'], best_params['FILE_CONTEXT_ALPHA'])
print(f"  Baseline OOF AUC: {best_auc:.6f}")

for sweep_round in range(3):  # 3 rounds of coordinate descent
    improved = False
    for param_name, values in param_grids.items():
        round_best_val = best_params[param_name]
        round_best_auc = best_auc
        for val in values:
            test_params = best_params.copy()
            test_params[param_name] = val
            auc = evaluate_config(oof_probe, oof_protossm, Y_FULL,
                test_params['PROBE_ALPHA'], test_params['PROTOSSM_ENSEMBLE_WEIGHT'],
                test_params['QUANTILE_MIX_ALPHA'], test_params['SHARP_TEMP'],
                test_params['POWER_GAMMA'], test_params['FILE_CONTEXT_ALPHA'])
            if auc > round_best_auc:
                round_best_auc = auc
                round_best_val = val
        if round_best_auc > best_auc:
            print(f"  Round {sweep_round+1}: {param_name} {best_params[param_name]:.3f} -> {round_best_val:.3f} (AUC {best_auc:.6f} -> {round_best_auc:.6f})")
            best_params[param_name] = round_best_val
            best_auc = round_best_auc
            improved = True
    if not improved:
        print(f"  Round {sweep_round+1}: No improvement, stopping.")
        break

print(f"\\n  OPTIMAL PARAMS: {best_params}")
print(f"  OPTIMAL OOF AUC: {best_auc:.6f}")

# Apply optimal params for test inference
PROBE_ALPHA = best_params['PROBE_ALPHA']
PROTOSSM_ENSEMBLE_WEIGHT = best_params['PROTOSSM_ENSEMBLE_WEIGHT']
QUANTILE_MIX_ALPHA = best_params['QUANTILE_MIX_ALPHA']
POWER_GAMMA = best_params['POWER_GAMMA']
FILE_CONTEXT_ALPHA = best_params['FILE_CONTEXT_ALPHA']
_opt_sharp_temp = best_params['SHARP_TEMP']
for idx in SHARPENED:
    PER_CLASS_TEMPS[idx] = _opt_sharp_temp
PER_CLASS_TEMPS[18] = min(_opt_sharp_temp + 0.1, 1.0)

_sweep_time = time.time() - _sweep_start
print(f"  Sweep time: {_sweep_time:.1f}s")
# === End V193 Sweep ===
'''

print("Generated injection code block.")
print(f"Block size: {len(injection_code)} chars, ~{injection_code.count(chr(10))} lines")
print()
print("This block should be injected into v111/script.py AFTER the OOF computation")
print("cell (after 'OOF AUC (base fusion)' print) and BEFORE the test inference cell.")
print()
print("The block will:")
print("1. Reconstruct OOF probe + protossm predictions")
print("2. Run 3 rounds of coordinate descent over 6 post-processing params")
print("3. Apply the optimal params before test inference")
print("4. Total added compute: ~10-30 seconds (pure numpy)")
print()

# Save the injection code
with open(WORK_DIR / 'v193-hp-sweep' / 'injection_block.py', 'w') as f:
    f.write(injection_code)
print(f"Saved injection block to: {WORK_DIR / 'v193-hp-sweep' / 'injection_block.py'}")
