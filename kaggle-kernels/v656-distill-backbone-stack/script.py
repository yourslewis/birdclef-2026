"""BC26 v656 — Distinct-backbone distill 3-stack hidden-test inference (LB probe).

First genuinely DIFFERENT base-embedding family to reach the BirdCLEF-2026 hidden LB:
soundscape-native ConvNeXt-nano (distill w0.7 + w0.85) + RegNetY-008, each soft1279-teacher
distilled, leave-site trained on train_soundscapes 5s logmel windows. Output = per-window
234-logit rank-mean across the 3 backbones -> sigmoid -> submission.csv.

This is a representation-level diversity datapoint, NOT a head-knob variant. Standalone weak
expected (proxy row-AUC ~0.73, weak-class AUC ~0.83) — the value is the live hidden-LB read on
a distinct foundation and whether it composes with the 0.950 frontier later.
"""
import os, re, gc, time
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch

t0 = time.time()
INPUT = Path("/kaggle/input")

# ---- competition dir ----
def find_comp():
    for p in [Path("/kaggle/input/birdclef-2026"), Path("/kaggle/input/competitions/birdclef-2026")]:
        if (p / "sample_submission.csv").exists():
            return p
    for s in sorted(INPUT.rglob("sample_submission.csv")):
        if (s.parent / "taxonomy.csv").exists():
            return s.parent
    raise FileNotFoundError("competition dir not found")

BASE = find_comp()
sample_sub = pd.read_csv(BASE / "sample_submission.csv")
PRIMARY = sample_sub.columns[1:].tolist()
N_CLASSES = len(PRIMARY)

# ---- model bundle ----
import json as _json
def find_bundle():
    cand = Path("/kaggle/input/bc26-distill-backbone-stack-v1")
    if (cand / "labels.json").exists():
        return cand
    for p in sorted(INPUT.rglob("labels.json")):
        d = p.parent
        if (d / "convnext_w07.pt").exists() and (d / "regnety008.pt").exists():
            return d
    raise FileNotFoundError("distill bundle not found under /kaggle/input")
BUNDLE = find_bundle()
print("Bundle:", BUNDLE)
labels = _json.loads((BUNDLE / "labels.json").read_text())
assert labels == PRIMARY, "label order mismatch between bundle and sample_submission"

DEVICE = "cpu"
MODELS = []
for fn in ["convnext_w07.pt", "convnext_w085.pt", "regnety008.pt"]:
    m = torch.jit.load(str(BUNDLE / fn), map_location=DEVICE).eval()
    MODELS.append(m)
print("Loaded", len(MODELS), "distill backbones")

# ---- audio / mel front-end (matches training cfg) ----
SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES = 60 * SR
N_WINDOWS = 12
N_FFT = 1024
HOP = 512
N_MELS = 160

def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)

def mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)

def make_mel_filter(sr, n_fft, n_mels, fmin=20.0, fmax=None):
    if fmax is None:
        fmax = sr / 2
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    n_freq = n_fft // 2 + 1
    fb = np.zeros((n_mels, n_freq), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for k in range(left, min(center, n_freq)):
            fb[m - 1, k] = (k - left) / max(center - left, 1)
        for k in range(center, min(right, n_freq)):
            fb[m - 1, k] = (right - k) / max(right - center, 1)
    return torch.from_numpy(fb)

MEL_FB = make_mel_filter(SR, N_FFT, N_MELS)
WINDOW = torch.hann_window(N_FFT)

def waveform_to_logmel(wave):
    spec = torch.stft(wave, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
                      window=WINDOW, center=True, return_complex=True)
    mel = MEL_FB @ spec.abs().pow(2.0)
    logmel = torch.log1p(mel)
    return (logmel - logmel.mean()) / (logmel.std() + 1e-6)

def read_60s(path):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if len(y) < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    else:
        y = y[:FILE_SAMPLES]
    return y

def rankify(mat):
    # column-wise rank normalize to [0,1] over rows within a file batch
    order = mat.argsort(axis=0).argsort(axis=0).astype(np.float32)
    n = mat.shape[0]
    return order / max(n - 1, 1)

# ---- test files (dry-run fallback) ----
test_paths = sorted((BASE / "test_soundscapes").glob("*.ogg"))
IS_DRY = len(test_paths) == 0
if IS_DRY:
    test_paths = sorted((BASE / "train_soundscapes").glob("*.ogg"))[:3]
    print("Dry-run on", len(test_paths), "train soundscapes")
else:
    print("Hidden test files:", len(test_paths))

row_ids = []
all_logits = []  # per-window averaged-rank logits

for path in test_paths:
    y = read_60s(path)
    wav = torch.from_numpy(y.reshape(N_WINDOWS, WINDOW_SAMPLES))
    logmels = torch.stack([waveform_to_logmel(wav[i]) for i in range(N_WINDOWS)])  # (12, 160, frames)
    per_backbone = []
    with torch.no_grad():
        for m in MODELS:
            clip, _ = m(logmels)  # (12, 234)
            per_backbone.append(clip.numpy().astype(np.float32))
    # rank-mean across backbones (per file, per window-set)
    ranked = [rankify(pb) for pb in per_backbone]
    fused = np.mean(ranked, axis=0)  # (12, 234) in [0,1]
    stem = path.stem
    for t in range(N_WINDOWS):
        row_ids.append(f"{stem}_{(t + 1) * 5}")
    all_logits.append(fused)
    del logmels, wav
    gc.collect()

probs = np.vstack(all_logits).astype(np.float32)
probs = np.clip(probs, 1e-6, 1 - 1e-6)

sub = pd.DataFrame(probs, columns=PRIMARY)
sub.insert(0, "row_id", row_ids)

if IS_DRY:
    # align to sample_submission schema for dry-run validation
    template = sub[PRIMARY].mean(axis=0).astype(np.float32)
    sub = sample_sub.copy()
    for c in PRIMARY:
        sub[c] = template[c]

sub[PRIMARY] = sub[PRIMARY].astype(np.float32)
assert sub.columns.tolist() == ["row_id"] + PRIMARY
assert not sub.isna().any().any()
assert np.isfinite(sub[PRIMARY].values).all()
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv", sub.shape, "range", float(probs.min()), float(probs.max()),
      "wall", round(time.time() - t0, 1), "s")
print(sub.iloc[:2, :6])
