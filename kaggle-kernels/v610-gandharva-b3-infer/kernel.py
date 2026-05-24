# BC26 v610: repo-owned inference writer for Gandharva EfficientNet-B3 SED checkpoints.
# Uses public training artifact gandharvakhedekar/birdclef2026-new; no direct slot until verifier passes.
import os, re, json, math, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

ROOT_CANDIDATES = [Path('/kaggle/input/competitions/birdclef-2026'), Path('/kaggle/input/birdclef-2026')]
BASE = next((p for p in ROOT_CANDIDATES if (p/'sample_submission.csv').exists() and (p/'taxonomy.csv').exists()), None)
if BASE is None:
    for hit in Path('/kaggle/input').rglob('sample_submission.csv'):
        if (hit.parent/'taxonomy.csv').exists():
            BASE = hit.parent; break
if BASE is None:
    raise FileNotFoundError('BirdCLEF competition input not found')

SAMPLE = pd.read_csv(BASE/'sample_submission.csv')
TAX = pd.read_csv(BASE/'taxonomy.csv')
SAMPLE_COLS = [c for c in SAMPLE.columns if c != 'row_id']
TAX_COLS = TAX['primary_label'].astype(str).tolist()
N_CLASSES = len(TAX_COLS)
print('BASE', BASE, 'sample', SAMPLE.shape, 'taxonomy classes', N_CLASSES)

SR=32000; DURATION=5; SAMPLES=SR*DURATION; N_MELS=128; HOP_LENGTH=320; N_FFT=1024; FMIN=20; FMAX=16000
BATCH_SIZE=24
DEVICE='cpu'

# Imports that may be slow; fail loudly so verifier does not submit bad output.
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import timm
except Exception as e:
    raise RuntimeError('timm unavailable in Kaggle image; cannot run Gandharva B3 inference') from e
try:
    import soundfile as sf
except Exception:
    sf = None
import librosa


def load_audio_fast(path):
    if sf is not None:
        try:
            audio, sr = sf.read(str(path), dtype='float32')
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SR:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
            return audio.astype(np.float32)
        except Exception:
            pass
    audio, _ = librosa.load(str(path), sr=SR, mono=True)
    return audio.astype(np.float32)


def fix_audio_length(audio, target_samples=SAMPLES):
    if len(audio) < target_samples:
        reps = target_samples // max(1, len(audio)) + 1
        audio = np.tile(audio, reps)
    if len(audio) > target_samples:
        start = (len(audio) - target_samples)//2
        audio = audio[start:start+target_samples]
    return audio.astype(np.float32)


def audio_to_mel(audio):
    mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT, fmin=FMIN, fmax=FMAX)
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    if mel.shape[1] < 500:
        mel = np.pad(mel, ((0,0),(0,500-mel.shape[1])), mode='edge')
    mel = mel[:, :500]
    return mel.astype(np.float32)


class BirdCLEFModel(nn.Module):
    def __init__(self, backbone_name='efficientnet_b3', num_classes=N_CLASSES, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, in_chans=1, num_classes=0, global_pool='')
        feat_dim = self.backbone.num_features
        self.att = nn.Linear(feat_dim, num_classes)
        self.cls = nn.Linear(feat_dim, num_classes)
    def forward(self, x):
        feat = self.backbone(x)
        feat = feat.mean(dim=2).permute(0,2,1)
        att = torch.softmax(self.att(feat), dim=1)
        logit = self.cls(feat)
        return (att * logit).sum(dim=1)


def find_checkpoints():
    roots = [Path('/kaggle/input'), Path('/kaggle/working')]
    ckpts=[]
    for root in roots:
        if not root.exists(): continue
        ckpts.extend(root.rglob('fold*_ep*_auc*.pth'))
    # Prefer Gandharva notebook outputs and one best per fold.
    parsed=[]
    pat=re.compile(r'fold(\d+)_ep(\d+)_auc([0-9.]+)\.pth')
    for p in ckpts:
        m=pat.search(p.name)
        if m:
            parsed.append((int(m.group(1)), float(m.group(3)), p))
    best={}
    for fold, auc, p in parsed:
        if fold not in best or auc > best[fold][0]:
            best[fold]=(auc,p)
    out=[best[k][1] for k in sorted(best)]
    print('checkpoints:', [str(p) for p in out])
    if not out:
        raise FileNotFoundError('No Gandharva fold*_auc*.pth checkpoints found')
    return out


def load_model(path):
    model = BirdCLEFModel(pretrained=False).to(DEVICE)
    ckpt = torch.load(str(path), map_location=DEVICE)
    state = None
    if isinstance(ckpt, dict):
        for key in ['model_state_dict','state_dict','model']:
            if key in ckpt and isinstance(ckpt[key], dict):
                state=ckpt[key]; break
    if state is None and isinstance(ckpt, dict):
        state = ckpt
    if state is None:
        raise RuntimeError(f'Cannot locate model state in {path}')
    # Strip common prefixes.
    clean={}
    for k,v in state.items():
        nk=k.replace('module.','')
        clean[nk]=v
    missing, unexpected = model.load_state_dict(clean, strict=False)
    print('loaded', path.name, 'missing', len(missing), 'unexpected', len(unexpected))
    if len(unexpected) > 50 or len(missing) > 50:
        raise RuntimeError('checkpoint key mismatch too large')
    model.eval()
    return model


def iter_windows(path):
    audio = load_audio_fast(path)
    # Hidden test soundscapes are 60s; create 12 non-overlap 5s chunks ending at 5..60.
    rows=[]; mels=[]
    stem=Path(path).stem
    for end_sec in range(5, 65, 5):
        start=(end_sec-5)*SR; end=end_sec*SR
        clip = audio[start:end]
        clip = fix_audio_length(clip)
        rows.append(f'{stem}_{end_sec}')
        mels.append(audio_to_mel(clip))
    return rows, mels


def predict_mels(models, mels):
    preds=[]
    with torch.no_grad():
        for i in range(0, len(mels), BATCH_SIZE):
            x=np.stack(mels[i:i+BATCH_SIZE]).astype(np.float32)
            xb=torch.from_numpy(x[:,None,:,:]).to(DEVICE)
            fold_probs=[]
            for model in models:
                fold_probs.append(torch.sigmoid(model(xb)).cpu().numpy())
            # simple arithmetic mean; stable and avoids OOF-weight overfit from source notebook
            preds.append(np.mean(fold_probs, axis=0))
    return np.concatenate(preds, axis=0)


ckpts=find_checkpoints()
models=[load_model(p) for p in ckpts]

test_dir=BASE/'test_soundscapes'
test_paths=sorted(test_dir.glob('*.ogg'))
IS_DRY=not test_paths
if IS_DRY:
    print('No hidden test mounted; dry-run on first train soundscape then align to sample')
    test_paths=sorted((BASE/'train_soundscapes').glob('*.ogg'))[:1]
else:
    print('Hidden test files', len(test_paths))

all_rows=[]; all_mels=[]
for p in test_paths:
    rows,mels=iter_windows(p)
    all_rows.extend(rows); all_mels.extend(mels)
print('windows', len(all_rows), 'mels', np.array(all_mels[:1]).shape if all_mels else None)

pred_tax = predict_mels(models, all_mels)
pred_tax = np.clip(pred_tax, 0.0, 1.0).astype(np.float32)
raw = pd.DataFrame(pred_tax, columns=TAX_COLS)
raw.insert(0, 'row_id', all_rows)
raw.to_csv('submission_gandharva_b3_raw.csv', index=False)

# Reorder to sample columns; for dry-run, use per-class mean template on sample rows.
if IS_DRY:
    out=SAMPLE.copy()
    means=raw[TAX_COLS].mean(axis=0)
    for c in SAMPLE_COLS:
        out[c]=float(means.get(c, 0.0))
else:
    raw=raw.set_index('row_id')
    missing=[rid for rid in SAMPLE['row_id'].astype(str) if rid not in raw.index]
    if missing:
        raise RuntimeError(f'missing hidden row_ids in predictions: {missing[:5]} count={len(missing)}')
    out=SAMPLE[['row_id']].copy()
    for c in SAMPLE_COLS:
        out[c]=raw.loc[SAMPLE['row_id'].astype(str), c].to_numpy(np.float32) if c in raw.columns else 0.0

vals=out[SAMPLE_COLS].to_numpy(np.float32)
assert out['row_id'].is_unique
assert vals.shape[1] == len(SAMPLE_COLS)
assert np.isfinite(vals).all()
assert vals.min() >= 0.0 and vals.max() <= 1.0
out.to_csv('submission.csv', index=False)
print('submission.csv', out.shape, 'min', float(vals.min()), 'max', float(vals.max()), 'mean', float(vals.mean()), 'uniq6', len(np.unique(np.round(vals.ravel()[:10000], 6))))
