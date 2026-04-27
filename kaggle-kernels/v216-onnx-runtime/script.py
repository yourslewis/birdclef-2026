#!/usr/bin/env python3
"""
BirdCLEF 2026 — v94: Exact v67 Parameters + ONNX Perch (Minimal Change)
=============================================================================

Based on v90 with CRITICAL speed improvement:
- ONNX Perch replaces TensorFlow Perch for test inference (9x faster)
- Eliminates timeout risk on hidden test
- Keeps TF Perch for labels.csv mapping only (or reads directly)
- All v90 improvements: tuned fusion, event smoothing, 3 seeds

Target: 0.925+ LB. CPU-only, ~3 min with cache + fast ONNX inference.
"""

# === Cell: Install TF 2.20 ===
import subprocess, sys, glob as _glob
from pathlib import Path

# Dynamic TF wheel detection
TF_CANDIDATES = [
    Path('/kaggle/input/bc26-tensorflow-2-20-0/wheel'),
    Path('/kaggle/input/notebooks/kdmitrie/bc26-tensorflow-2-20-0/wheel'),
    Path('/kaggle/input/datasets/kdmitrie/bc26-tensorflow-2-20-0/wheel'),
    Path('/kaggle/input/notebooks/ashok205/tf-wheels/tf_wheels'),
]
tf_dir = next((p for p in TF_CANDIDATES if p.exists()), None)
if tf_dir:
    for pattern in ['tensorboard*.whl', 'tensorflow*.whl']:
        for whl in sorted(tf_dir.glob(pattern)):
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', str(whl)], check=True)
    print(f'Installed TF from {tf_dir}')
else:
    print('WARNING: No TF wheel directory found')

# === Cell: Install ONNX Runtime (9x faster than TF Perch) ===
import glob as _glob_onnx
_whl_candidates = _glob_onnx.glob('/kaggle/input/**/onnxruntime*cp312*x86_64*.whl', recursive=True)
if _whl_candidates:
    _whl_candidates.sort(reverse=True)
    print(f'Installing onnxruntime from: {_whl_candidates[0]}')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', _whl_candidates[0]], check=True)
else:
    print('WARNING: No onnxruntime wheel found. Will fall back to TF Perch.')

# === Cell: Imports ===
import gc, json, os, random, re, time, warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf

try:
    from scipy.ndimage import convolve1d
except ImportError:
    def convolve1d(arr, weights, axis=0, mode='nearest'):
        w = np.asarray(weights, dtype=arr.dtype)
        half = len(w) // 2
        out = np.zeros_like(arr)
        n = arr.shape[axis]
        for i in range(n):
            val = 0.0
            for j, wj in enumerate(w):
                idx = i + j - half
                idx = max(0, min(n - 1, idx))
                if axis == 0:
                    val = val + wj * arr[idx]
                else:
                    val = val + wj * arr[:, idx]
            if axis == 0:
                out[i] = val
            else:
                out[:, i] = val
        return out

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
tf.experimental.numpy.experimental_enable_numpy_behavior()

_WALL_START = time.time()

def time_remaining():
    return 90*60 - (time.time() - _WALL_START)

print('TensorFlow :', tf.__version__)
print('NumPy      :', np.__version__)
print('V94: Tuned Fusion + Event Smoothing + 3 Seeds (from 0.943 analysis)')

# === Cell: Settings ===
SEED = 42
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)

# Robust path detection
BASE_CANDIDATES = [
    Path('/kaggle/input/birdclef-2026'),
    Path('/kaggle/input/competitions/birdclef-2026'),
]
BASE = next((p for p in BASE_CANDIDATES if (p / 'taxonomy.csv').exists()), BASE_CANDIDATES[0])

MODEL_CANDIDATES = [
    Path('/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1'),
    Path('/kaggle/input/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1'),
    Path('/kaggle/input/bird-vocalization-classifier/TensorFlow2/perch_v2_cpu/1'),
]
MODEL_DIR = next((p for p in MODEL_CANDIDATES if p.exists()), MODEL_CANDIDATES[0])

CACHE_CANDIDATES = [
    Path('/kaggle/input/perch-meta'),
    Path('/kaggle/input/datasets/jaejohn/perch-meta'),
    Path('/kaggle/input/notebooks/jaejohn/perch-meta'),
]
CACHE_DIR = next(
    (d for d in CACHE_CANDIDATES
     if (d / 'full_perch_meta.parquet').exists() and (d / 'full_perch_arrays.npz').exists()),
    None,
)
CACHE_EXISTS = CACHE_DIR is not None
WORK_CACHE_DIR = Path('/kaggle/working/cache')
WORK_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Audio constants
SR = 32_000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES = 60 * SR
N_WINDOWS = 12
BATCH_FILES = 16
DRYRUN_N_FILES = 20

# Prior fusion — UPDATED from 0.943 notebook's best_fusion config
LAMBDA_EVENT = 0.45       # was 0.4 in v85
LAMBDA_TEXTURE = 1.1      # was 1.0 in v85
LAMBDA_PROXY_TEXTURE = 0.9  # was 0.8 in v85
SMOOTH_TEXTURE_ALPHA = 0.35
SMOOTH_EVENT_ALPHA = 0.0  # DISABLED — v67 didn't have this

# Probe params
PROBE_PCA_DIM = 64
PROBE_MIN_POS = 8
PROBE_C = 0.50
PROBE_ALPHA = 0.40

# ProtoSSM params — v90: restore 3 seeds for robustness (like v67)
PROTOSSM_ENSEMBLE_WEIGHT = 0.50
PROTOSSM_D_MODEL = 128
PROTOSSM_D_STATE = 16
PROTOSSM_N_LAYERS = 2
PROTOSSM_DROPOUT = 0.15
PROTOSSM_EPOCHS = 60  # v67 used 60
PROTOSSM_LR = 0.001
PROTOSSM_PATIENCE = 10
PROTOSSM_SEEDS = [42, 137, 2026]  # Exact v67 seeds
PROTOSSM_MIXUP_ALPHA = 0.0  # v67 didn't use mixup
PROTOSSM_LABEL_SMOOTH = 0.0  # v67 didn't use label smoothing
PROTOSSM_SWA_START = 999  # Disabled — v67 didn't use SWA

# Post-processing
GAUSSIAN_WEIGHTS = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float32)
POWER_GAMMA = 0.85
FILE_CONTEXT_ALPHA = 0.15
QUANTILE_MIX_ALPHA = 0.5

# Proxy taxa
PROXY_TAXA = {'Amphibia', 'Insecta', 'Aves'}

print(f'BASE: {BASE}')
print(f'MODEL: {MODEL_DIR}')
print(f'CACHE: {CACHE_DIR} (exists={CACHE_EXISTS})')
print(f'V94: PCA={PROBE_PCA_DIM}, C={PROBE_C}, alpha={PROBE_ALPHA}')
print(f'V94: ProtoSSM d_model={PROTOSSM_D_MODEL}, ensemble_weight={PROTOSSM_ENSEMBLE_WEIGHT}')
print(f'V94: power_gamma={POWER_GAMMA}, file_context_alpha={FILE_CONTEXT_ALPHA}')
print(f'V94: mixup_alpha={PROTOSSM_MIXUP_ALPHA}, label_smooth={PROTOSSM_LABEL_SMOOTH}, SWA_start={PROTOSSM_SWA_START}')
print(f'V94: seeds={PROTOSSM_SEEDS}, epochs={PROTOSSM_EPOCHS}, patience={PROTOSSM_PATIENCE}')

# === Cell: Data Loading ===
taxonomy = pd.read_csv(BASE / 'taxonomy.csv')
sample_sub = pd.read_csv(BASE / 'sample_submission.csv')
soundscape_raw = pd.read_csv(BASE / 'train_soundscapes_labels.csv')
soundscape_lbls = soundscape_raw.drop_duplicates().reset_index(drop=True)

PRIMARY_LABELS = sample_sub.columns[1:].tolist()
N_CLASSES = len(PRIMARY_LABELS)
label_to_idx = {c: i for i, c in enumerate(PRIMARY_LABELS)}

# === Cell: Parse Labels ===
FNAME_RE = re.compile(r'BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg')

def parse_labels(x):
    if pd.isna(x): return []
    return [t.strip() for t in str(x).split(';') if t.strip()]

def union_labels(series):
    return sorted(set(lbl for x in series for lbl in parse_labels(x)))

def parse_soundscape_filename(name):
    m = FNAME_RE.match(name)
    if not m: return {'site': None, 'hour_utc': -1}
    _, site, _, hms = m.groups()
    return {'site': site, 'hour_utc': int(hms[:2])}

sc_clean = (
    soundscape_lbls
    .groupby(['filename', 'start', 'end'])['primary_label']
    .apply(union_labels)
    .reset_index(name='label_list')
)
sc_clean['end_sec'] = pd.to_timedelta(sc_clean['end']).dt.total_seconds().astype(int)
sc_clean['row_id'] = sc_clean['filename'].str.replace('.ogg', '', regex=False) + '_' + sc_clean['end_sec'].astype(str)
meta_cols = sc_clean['filename'].apply(parse_soundscape_filename).apply(pd.Series)
sc_clean = pd.concat([sc_clean, meta_cols], axis=1)

wpf = sc_clean.groupby('filename').size()
full_files = sorted(wpf[wpf == N_WINDOWS].index.tolist())
sc_clean['file_fully_labeled'] = sc_clean['filename'].isin(full_files)

Y_SC = np.zeros((len(sc_clean), N_CLASSES), dtype=np.uint8)
for i, labels in enumerate(sc_clean['label_list']):
    for lbl in labels:
        if lbl in label_to_idx:
            Y_SC[i, label_to_idx[lbl]] = 1

full_truth = (
    sc_clean[sc_clean['file_fully_labeled']]
    .sort_values(['filename', 'end_sec'])
    .reset_index(drop=False)
)
Y_FULL_TRUTH = Y_SC[full_truth['index'].to_numpy()]

print(f'Files: {len(full_files)}, Windows: {len(full_truth)}, Active: {(Y_FULL_TRUTH.sum(0) > 0).sum()}')

# === Cell: Load Perch & Mapping ===
print('Loading Perch model...')
birdclassifier = tf.saved_model.load(str(MODEL_DIR))
infer_fn = birdclassifier.signatures['serving_default']
print('Perch loaded.')

bc_labels = (
    pd.read_csv(MODEL_DIR / 'assets' / 'labels.csv')
    .reset_index()
    .rename(columns={'index': 'bc_index', 'inat2024_fsd50k': 'scientific_name'})
)
NO_LABEL_INDEX = len(bc_labels)

taxonomy_ = taxonomy.copy()
taxonomy_['scientific_name'] = taxonomy_['scientific_name'].astype(str)
mapping = taxonomy_.merge(bc_labels[['scientific_name', 'bc_index']], on='scientific_name', how='left')
mapping['bc_index'] = mapping['bc_index'].fillna(NO_LABEL_INDEX).astype(int)

label_to_bc = mapping.set_index('primary_label')['bc_index']
BC_INDICES = np.array([int(label_to_bc.loc[c]) for c in PRIMARY_LABELS], dtype=np.int32)

MAPPED_MASK = BC_INDICES != NO_LABEL_INDEX
MAPPED_POS = np.where(MAPPED_MASK)[0].astype(np.int32)
UNMAPPED_POS = np.where(~MAPPED_MASK)[0].astype(np.int32)
MAPPED_BC_INDICES = BC_INDICES[MAPPED_MASK].astype(np.int32)

print(f'Mapped: {MAPPED_MASK.sum()}/{N_CLASSES}')

# === Cell: Class Types & Extended Proxies ===
CLASS_NAME_MAP = taxonomy_.set_index('primary_label')['class_name'].to_dict()
TEXTURE_TAXA = {'Amphibia', 'Insecta'}
ACTIVE_CLASSES = [PRIMARY_LABELS[i] for i in np.where(Y_SC.sum(0) > 0)[0]]

idx_active_texture = np.array([label_to_idx[c] for c in ACTIVE_CLASSES if CLASS_NAME_MAP.get(c) in TEXTURE_TAXA], dtype=np.int32)
idx_active_event = np.array([label_to_idx[c] for c in ACTIVE_CLASSES if CLASS_NAME_MAP.get(c) not in TEXTURE_TAXA], dtype=np.int32)
idx_mapped_active_texture = idx_active_texture[MAPPED_MASK[idx_active_texture]]
idx_mapped_active_event = idx_active_event[MAPPED_MASK[idx_active_event]]
idx_unmapped_active_texture = idx_active_texture[~MAPPED_MASK[idx_active_texture]]
idx_unmapped_active_event = idx_active_event[~MAPPED_MASK[idx_active_event]]
idx_unmapped_inactive = np.array([i for i in UNMAPPED_POS if PRIMARY_LABELS[i] not in ACTIVE_CLASSES], dtype=np.int32)

unmapped_df = mapping[mapping['bc_index'] == NO_LABEL_INDEX].copy()
unmapped_non_sonotype = unmapped_df[~unmapped_df['primary_label'].astype(str).str.contains('son', na=False)]

proxy_map = {}
for _, row in unmapped_non_sonotype.iterrows():
    genus = str(row['scientific_name']).split()[0]
    hits = bc_labels[bc_labels['scientific_name'].str.match(rf'^{re.escape(genus)}\s', na=False)]
    if len(hits) > 0:
        proxy_map[str(row['primary_label'])] = hits['bc_index'].astype(int).tolist()

SELECTED_PROXY_TARGETS = sorted([t for t in proxy_map if CLASS_NAME_MAP.get(t) in PROXY_TAXA])
selected_proxy_pos = np.array([label_to_idx[c] for c in SELECTED_PROXY_TARGETS], dtype=np.int32)
selected_proxy_pos_to_bc = {label_to_idx[t]: np.array(proxy_map[t], dtype=np.int32) for t in SELECTED_PROXY_TARGETS}

idx_selected_proxy_active_texture = np.intersect1d(selected_proxy_pos, idx_active_texture)
idx_selected_prioronly_active_texture = np.setdiff1d(idx_unmapped_active_texture, selected_proxy_pos)
idx_selected_prioronly_active_event = np.setdiff1d(idx_unmapped_active_event, selected_proxy_pos)

print(f'Proxy targets: {len(SELECTED_PROXY_TARGETS)}')

# === Cell: Family taxonomy ===
FAMILY_MAP = taxonomy_.set_index('primary_label')['class_name'].to_dict()
FAMILY_GROUPS = {}
for ci, label in enumerate(PRIMARY_LABELS):
    family = FAMILY_MAP.get(label, 'Unknown')
    FAMILY_GROUPS.setdefault(family, []).append(ci)

FAMILY_IDX_MAP = {fam: np.array(idxs, dtype=np.int32) for fam, idxs in FAMILY_GROUPS.items()}
CLASS_FAMILY = {ci: FAMILY_MAP.get(label, 'Unknown') for ci, label in enumerate(PRIMARY_LABELS)}
print(f'Family groups: {len(FAMILY_GROUPS)} — {sorted(FAMILY_GROUPS.keys())}')

# === Cell: Utilities ===
def macro_auc(y_true, y_score):
    keep = y_true.sum(0) > 0
    return roc_auc_score(y_true[:, keep], y_score[:, keep], average='macro')

def smooth_cols(scores, cols, alpha=0.35):
    if alpha <= 0 or len(cols) == 0: return scores.copy()
    s = scores.copy()
    view = s.reshape(-1, N_WINDOWS, s.shape[1])
    x = view[:, :, cols]
    prev = np.concatenate([x[:, :1, :], x[:, :-1, :]], axis=1)
    nxt = np.concatenate([x[:, 1:, :], x[:, -1:, :]], axis=1)
    view[:, :, cols] = (1 - alpha) * x + 0.5 * alpha * (prev + nxt)
    return s

def seq_features_1d(v):
    x = v.reshape(-1, N_WINDOWS)
    prev = np.concatenate([x[:, :1], x[:, :-1]], axis=1).reshape(-1)
    nxt = np.concatenate([x[:, 1:], x[:, -1:]], axis=1).reshape(-1)
    mean_v = np.repeat(x.mean(1), N_WINDOWS)
    max_v = np.repeat(x.max(1), N_WINDOWS)
    min_v = np.repeat(x.min(1), N_WINDOWS)
    range_v = max_v - min_v
    return prev, nxt, mean_v, max_v, min_v, range_v

def cosine_sim_to_prototype(Z, prototype):
    Z_norm = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    p_norm = prototype / (np.linalg.norm(prototype) + 1e-8)
    return Z_norm @ p_norm

def build_class_features(Z, raw_col, prior_col, base_col,
                         proto_sim_col=None, family_mean_col=None):
    p, n, m, mx, mn, rng = seq_features_1d(base_col)
    parts = [
        Z,
        raw_col[:, None], prior_col[:, None], base_col[:, None],
        p[:, None], n[:, None], m[:, None], mx[:, None],
        mn[:, None], rng[:, None],
    ]
    if proto_sim_col is not None:
        parts.append(proto_sim_col[:, None])
    if family_mean_col is not None:
        parts.append(family_mean_col[:, None])
    return np.concatenate(parts, axis=1).astype(np.float32)

def gauss_smooth_logits(scores, weights=GAUSSIAN_WEIGHTS):
    n_windows = N_WINDOWS
    n_rows = scores.shape[0]
    n_complete_files = n_rows // n_windows
    remainder = n_rows % n_windows
    if n_complete_files > 0:
        complete = scores[:n_complete_files * n_windows].reshape(-1, n_windows, scores.shape[1]).copy()
        for i in range(complete.shape[0]):
            complete[i] = convolve1d(complete[i], weights, axis=0, mode='nearest')
        result = complete.reshape(-1, scores.shape[1])
        if remainder > 0:
            result = np.vstack([result, scores[n_complete_files * n_windows:]])
        return result
    return scores.copy()

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def file_context_boost(probs, alpha=FILE_CONTEXT_ALPHA):
    if alpha <= 0:
        return probs
    n_rows = probs.shape[0]
    n_complete_files = n_rows // N_WINDOWS
    remainder = n_rows % N_WINDOWS
    if n_complete_files > 0:
        view = probs[:n_complete_files * N_WINDOWS].reshape(-1, N_WINDOWS, probs.shape[1]).copy()
        file_max = view.max(axis=1, keepdims=True)
        boosted = (1.0 - alpha) * view + alpha * file_max
        result = boosted.reshape(-1, probs.shape[1])
        if remainder > 0:
            result = np.vstack([result, probs[n_complete_files * N_WINDOWS:]])
        return result
    return probs.copy()

# === Cell: Perch Inference ===
def read_soundscape_60s(path):
    y, sr = sf.read(path, dtype='float32', always_2d=False)
    if y.ndim == 2: y = y.mean(axis=1)
    if len(y) < FILE_SAMPLES: y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    return y[:FILE_SAMPLES]

def infer_perch_batch(paths, verbose=True):
    paths = [Path(p) for p in paths]
    n_files = len(paths)
    n_rows = n_files * N_WINDOWS
    row_ids = np.empty(n_rows, dtype=object)
    filenames = np.empty(n_rows, dtype=object)
    sites = np.empty(n_rows, dtype=object)
    hours = np.empty(n_rows, dtype=np.int16)
    scores = np.zeros((n_rows, N_CLASSES), dtype=np.float32)
    embeddings = np.zeros((n_rows, 1536), dtype=np.float32)
    write_row = 0
    itr = tqdm(range(0, n_files, BATCH_FILES), desc='Perch', disable=not verbose)
    for start in itr:
        batch = paths[start:start + BATCH_FILES]
        bn = len(batch)
        x = np.empty((bn * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
        bstart = write_row
        for bi, path in enumerate(batch):
            audio = read_soundscape_60s(path)
            x[bi*N_WINDOWS:(bi+1)*N_WINDOWS] = audio.reshape(N_WINDOWS, WINDOW_SAMPLES)
            meta = parse_soundscape_filename(path.name)
            row_ids[write_row:write_row+N_WINDOWS] = [f'{path.stem}_{t}' for t in range(5, 65, 5)]
            filenames[write_row:write_row+N_WINDOWS] = path.name
            sites[write_row:write_row+N_WINDOWS] = meta['site']
            hours[write_row:write_row+N_WINDOWS] = meta['hour_utc']
            write_row += N_WINDOWS
        out = infer_fn(inputs=tf.convert_to_tensor(x))
        logits = out['label'].numpy().astype(np.float32)
        emb = out['embedding'].numpy().astype(np.float32)
        scores[bstart:write_row, MAPPED_POS] = logits[:write_row-bstart, MAPPED_BC_INDICES]
        embeddings[bstart:write_row] = emb
        for pos, bc_idx_arr in selected_proxy_pos_to_bc.items():
            scores[bstart:write_row, pos] = logits[:write_row-bstart, bc_idx_arr].max(axis=1)
        del x, out, logits, emb; gc.collect()
    meta_df = pd.DataFrame({'row_id': row_ids, 'filename': filenames, 'site': sites, 'hour_utc': hours})
    return meta_df, scores, embeddings

# === Cell: ONNX Perch Inference (9x faster) ===
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print('WARNING: onnxruntime not available, will use TF Perch')

# Search for ONNX model
ONNX_MODEL_PATH = None
if HAS_ONNX:
    _onnx_candidates = [
        Path('/kaggle/input/perch-onnx-for-birdclef-2026/perch_v2.onnx'),
        Path('/kaggle/input/datasets/rishikeshjani/perch-onnx-for-birdclef-2026/perch_v2.onnx'),
        Path('/kaggle/input/birdclef26-perch-onnx/perch_v2.onnx'),
        Path('/kaggle/input/datasets/yuriygreben/birdclef26-perch-onnx/perch_v2.onnx'),
        Path('/kaggle/input/perch-onnx/perch_v2.onnx'),
        Path('/kaggle/input/datasets/mlnjsh/perch-onnx/perch_v2.onnx'),
    ]
    ONNX_MODEL_PATH = next((p for p in _onnx_candidates if p.exists()), None)
    if ONNX_MODEL_PATH:
        print(f'ONNX model found: {ONNX_MODEL_PATH}')
        _onnx_t0 = time.time()
        _sess_options = ort.SessionOptions()
        _sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _sess_options.intra_op_num_threads = 1
        _sess_options.inter_op_num_threads = 1
        _sess_options.enable_mem_pattern = False
        _sess_options.enable_cpu_mem_arena = False
        ort_session = ort.InferenceSession(str(ONNX_MODEL_PATH), _sess_options)
        print(f'ONNX session loaded in {time.time()-_onnx_t0:.1f}s')
    else:
        HAS_ONNX = False
        print('WARNING: ONNX model not found, will use TF Perch')

def infer_perch_onnx(paths, verbose=True):
    """ONNX-based Perch inference — 9x faster than TF SavedModel."""
    from concurrent.futures import ThreadPoolExecutor
    paths = [Path(p) for p in paths]
    n_files = len(paths)
    n_rows = n_files * N_WINDOWS
    row_ids = np.empty(n_rows, dtype=object)
    filenames = np.empty(n_rows, dtype=object)
    sites = np.empty(n_rows, dtype=object)
    hours = np.empty(n_rows, dtype=np.int16)
    scores = np.zeros((n_rows, N_CLASSES), dtype=np.float32)
    embeddings = np.zeros((n_rows, 1536), dtype=np.float32)
    write_row = 0
    batch_size = 4  # Smaller batches for ONNX (less RAM)
    itr = tqdm(range(0, n_files, batch_size), desc='Perch-ONNX', disable=not verbose)
    for start in itr:
        batch = paths[start:start + batch_size]
        bn = len(batch)
        x = np.empty((bn * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
        bstart = write_row
        # Parallel file loading
        def _load(p):
            return read_soundscape_60s(p)
        with ThreadPoolExecutor(max_workers=min(4, bn)) as pool:
            batch_audio = list(pool.map(_load, batch))
        for bi, path in enumerate(batch):
            audio = batch_audio[bi]
            x[bi*N_WINDOWS:(bi+1)*N_WINDOWS] = audio.reshape(N_WINDOWS, WINDOW_SAMPLES)
            meta = parse_soundscape_filename(path.name)
            row_ids[write_row:write_row+N_WINDOWS] = [f'{path.stem}_{t}' for t in range(5, 65, 5)]
            filenames[write_row:write_row+N_WINDOWS] = path.name
            sites[write_row:write_row+N_WINDOWS] = meta['site']
            hours[write_row:write_row+N_WINDOWS] = meta['hour_utc']
            write_row += N_WINDOWS
        # ONNX inference
        emb, logits = ort_session.run(
            ['embedding', 'label'], {'inputs': x.astype(np.float32)})
        emb = emb.astype(np.float32)
        logits = logits.astype(np.float32)
        scores[bstart:write_row, MAPPED_POS] = logits[:write_row-bstart, MAPPED_BC_INDICES]
        embeddings[bstart:write_row] = emb
        for pos, bc_idx_arr in selected_proxy_pos_to_bc.items():
            scores[bstart:write_row, pos] = logits[:write_row-bstart, bc_idx_arr].max(axis=1)
        del x, logits, emb; gc.collect()
    meta_df = pd.DataFrame({'row_id': row_ids, 'filename': filenames, 'site': sites, 'hour_utc': hours})
    return meta_df, scores, embeddings

print(f'V94: ONNX={HAS_ONNX and ONNX_MODEL_PATH is not None}')

# === Cell: Load Cache or Infer ===
if CACHE_EXISTS:
    print(f'Loading Perch cache from: {CACHE_DIR}')
    meta_full = pd.read_parquet(CACHE_DIR / 'full_perch_meta.parquet')
    arr = np.load(CACHE_DIR / 'full_perch_arrays.npz')
    scores_full_raw = arr['scores_full_raw'].astype(np.float32)
    emb_full = arr['emb_full'].astype(np.float32)
else:
    print('No cache. Running Perch on fully-labeled files...')
    full_paths = [BASE / 'train_soundscapes' / fn for fn in full_files]
    meta_full, scores_full_raw, emb_full = infer_perch_batch(full_paths)
    meta_full.to_parquet(WORK_CACHE_DIR / 'full_perch_meta.parquet', index=False)
    np.savez_compressed(WORK_CACHE_DIR / 'full_perch_arrays.npz',
                        scores_full_raw=scores_full_raw, emb_full=emb_full)

full_truth_aligned = full_truth.set_index('row_id').loc[meta_full['row_id']].reset_index()
Y_FULL = Y_SC[full_truth_aligned['index'].to_numpy()]
print(f'scores: {scores_full_raw.shape}, emb: {emb_full.shape}, Y: {Y_FULL.shape}')

# === Cell: Prior Fusion ===
def fit_prior_tables(prior_df, Y_prior):
    prior_df = prior_df.reset_index(drop=True)
    global_p = Y_prior.mean(0).astype(np.float32)
    site_keys = sorted(prior_df['site'].dropna().astype(str).unique())
    hour_keys = sorted(prior_df['hour_utc'].dropna().astype(int).unique())
    site_to_i, site_n, site_p = {}, [], []
    for s in site_keys:
        mask = prior_df['site'].astype(str).values == s
        site_to_i[s] = len(site_n)
        site_n.append(mask.sum())
        site_p.append(Y_prior[mask].mean(0))
    site_n = np.array(site_n, dtype=np.float32)
    site_p = np.stack(site_p).astype(np.float32) if site_p else np.zeros((0, Y_prior.shape[1]), np.float32)
    hour_to_i, hour_n, hour_p = {}, [], []
    for h in hour_keys:
        mask = prior_df['hour_utc'].astype(int).values == h
        hour_to_i[h] = len(hour_n)
        hour_n.append(mask.sum())
        hour_p.append(Y_prior[mask].mean(0))
    hour_n = np.array(hour_n, dtype=np.float32)
    hour_p = np.stack(hour_p).astype(np.float32) if hour_p else np.zeros((0, Y_prior.shape[1]), np.float32)
    sh_to_i, sh_n_list, sh_p_list = {}, [], []
    for (s, h), idx in prior_df.groupby(['site', 'hour_utc']).groups.items():
        sh_to_i[(str(s), int(h))] = len(sh_n_list)
        idx = np.array(list(idx))
        sh_n_list.append(len(idx))
        sh_p_list.append(Y_prior[idx].mean(0))
    sh_n = np.array(sh_n_list, dtype=np.float32)
    sh_p = np.stack(sh_p_list).astype(np.float32) if sh_p_list else np.zeros((0, Y_prior.shape[1]), np.float32)
    return dict(global_p=global_p, site_to_i=site_to_i, site_n=site_n, site_p=site_p,
                hour_to_i=hour_to_i, hour_n=hour_n, hour_p=hour_p,
                sh_to_i=sh_to_i, sh_n=sh_n, sh_p=sh_p)

def prior_logits(sites, hours, tables, eps=1e-4):
    n = len(sites)
    p = np.repeat(tables['global_p'][None, :], n, axis=0).astype(np.float32, copy=True)
    si = np.fromiter((tables['site_to_i'].get(str(s), -1) for s in sites), np.int32, n)
    hi = np.fromiter(
        (tables['hour_to_i'].get(int(h), -1) if int(h) >= 0 else -1 for h in hours),
        np.int32, n)
    shi = np.fromiter(
        (tables['sh_to_i'].get((str(s), int(h)), -1) if int(h) >= 0 else -1
         for s, h in zip(sites, hours)), np.int32, n)
    valid = hi >= 0
    if valid.any():
        nh = tables['hour_n'][hi[valid]][:, None]
        p[valid] = nh/(nh+8)*tables['hour_p'][hi[valid]] + (1-nh/(nh+8))*p[valid]
    valid = si >= 0
    if valid.any():
        ns = tables['site_n'][si[valid]][:, None]
        p[valid] = ns/(ns+8)*tables['site_p'][si[valid]] + (1-ns/(ns+8))*p[valid]
    valid = shi >= 0
    if valid.any():
        nsh = tables['sh_n'][shi[valid]][:, None]
        p[valid] = nsh/(nsh+4)*tables['sh_p'][shi[valid]] + (1-nsh/(nsh+4))*p[valid]
    np.clip(p, eps, 1-eps, out=p)
    return (np.log(p) - np.log1p(-p)).astype(np.float32)

def fuse_scores(base, sites, hours, tables):
    scores = base.copy()
    prior = prior_logits(sites, hours, tables)
    if len(idx_mapped_active_event):
        scores[:, idx_mapped_active_event] += LAMBDA_EVENT * prior[:, idx_mapped_active_event]
    if len(idx_mapped_active_texture):
        scores[:, idx_mapped_active_texture] += LAMBDA_TEXTURE * prior[:, idx_mapped_active_texture]
    if len(idx_selected_proxy_active_texture):
        scores[:, idx_selected_proxy_active_texture] += LAMBDA_PROXY_TEXTURE * prior[:, idx_selected_proxy_active_texture]
    if len(idx_selected_prioronly_active_event):
        scores[:, idx_selected_prioronly_active_event] = LAMBDA_EVENT * prior[:, idx_selected_prioronly_active_event]
    if len(idx_selected_prioronly_active_texture):
        scores[:, idx_selected_prioronly_active_texture] = LAMBDA_TEXTURE * prior[:, idx_selected_prioronly_active_texture]
    if len(idx_unmapped_inactive):
        scores[:, idx_unmapped_inactive] = -8.0
    scores = smooth_cols(scores, idx_active_texture, alpha=SMOOTH_TEXTURE_ALPHA)
    # V94: Add event smoothing for bird classes (from 0.943 notebook)
    scores = smooth_cols(scores, idx_active_event, alpha=SMOOTH_EVENT_ALPHA)
    return scores.astype(np.float32), prior

# === Cell: OOF Computation ===
gkf = GroupKFold(n_splits=5)
groups = meta_full['site'].to_numpy()

oof_base = np.zeros_like(scores_full_raw, dtype=np.float32)
oof_prior = np.zeros_like(scores_full_raw, dtype=np.float32)

for _, va_idx in tqdm(list(gkf.split(scores_full_raw, groups=groups)), desc='OOF folds'):
    va_idx = np.sort(va_idx)
    val_sites = set(meta_full.iloc[va_idx]['site'].tolist())
    prior_m = ~sc_clean['site'].isin(val_sites).values
    tables = fit_prior_tables(sc_clean.loc[prior_m].reset_index(drop=True), Y_SC[prior_m])
    sites_va = meta_full.iloc[va_idx]['site'].to_numpy()
    hours_va = meta_full.iloc[va_idx]['hour_utc'].to_numpy()
    oof_base[va_idx], oof_prior[va_idx] = fuse_scores(
        scores_full_raw[va_idx], sites_va, hours_va, tables)

auc_base = macro_auc(Y_FULL, oof_base)
print(f'\nOOF AUC (base fusion): {auc_base:.6f}')

# === Cell: PCA + Prototypes ===
emb_scaler = StandardScaler()
emb_scaled = emb_scaler.fit_transform(emb_full)

n_comp = min(PROBE_PCA_DIM, emb_scaled.shape[0] - 1, emb_scaled.shape[1])
emb_pca = PCA(n_components=n_comp)
Z_FULL = emb_pca.fit_transform(emb_scaled).astype(np.float32)
print(f'PCA: {n_comp} dims, var={emb_pca.explained_variance_ratio_.sum():.4f}')

CLASS_PROTOTYPES = {}
for ci in range(N_CLASSES):
    pos_mask = Y_FULL[:, ci] == 1
    if pos_mask.sum() >= PROBE_MIN_POS:
        CLASS_PROTOTYPES[ci] = Z_FULL[pos_mask].mean(axis=0)
print(f'Class prototypes: {len(CLASS_PROTOTYPES)} classes')

# === Cell: Probe Training ===
print(f'\n=== V94: Training MLP probes ===')

pos_counts = Y_FULL.sum(0)
probe_idx = np.where(pos_counts >= PROBE_MIN_POS)[0].astype(np.int32)
probe_models = {}

for cls_idx in tqdm(probe_idx, desc='Training MLP probes'):
    y = Y_FULL[:, cls_idx]
    if y.sum() == 0 or y.sum() == len(y):
        continue

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

    clf = MLPClassifier(
        hidden_layer_sizes=(128,), activation='relu',
        solver='adam', alpha=1e-3, max_iter=200,
        early_stopping=True, n_iter_no_change=10,
        random_state=42, verbose=False)
    clf.fit(X, y)
    probe_models[cls_idx] = clf

print(f'MLP probes: {len(probe_models)} / {N_CLASSES} classes')

# === Cell: ProtoSSM v2 — State Space Model + Mixup + SWA ===
print(f'\n=== V94: Training ProtoSSM v2 + Mixup + SWA ===')

import torch
import torch.nn as nn

import onnxruntime as ort

def convert_and_load_onnx(model, dummy_input, onnx_path="model.onnx"):
    # Export to ONNX
    model.eval()
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    # Load with ONNX Runtime
    providers = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    ort_session = ort.InferenceSession(onnx_path, providers=providers)
    return ort_session

def predict_onnx(ort_session, x):
    ort_inputs = {ort_session.get_inputs()[0].name: x.cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    return torch.tensor(ort_outs[0])

import torch.nn.functional as F

class S4DKernel(nn.Module):
    def __init__(self, d_model, d_state=16, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        log_A_real = torch.log(0.5 * torch.ones(d_model, d_state))
        A_imag = torch.pi * torch.arange(d_state).float().unsqueeze(0).repeat(d_model, 1)
        self.log_A_real = nn.Parameter(log_A_real)
        self.A_imag = nn.Parameter(A_imag)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        log_dt = torch.rand(d_model) * (np.log(dt_max) - np.log(dt_min)) + np.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        self.D = nn.Parameter(torch.randn(d_model) * 0.01)

    def forward(self, L):
        dt = torch.exp(self.log_dt)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag
        dtA = A * dt.unsqueeze(-1)
        C = self.C.to(dtype=torch.cfloat)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=dtA.device).float()
        K = torch.exp(K)
        K = torch.einsum('dn,dnl->dl', C, K).real
        return K

class S4DLayer(nn.Module):
    def __init__(self, d_model, d_state=16, dropout=0.1):
        super().__init__()
        self.kernel = S4DKernel(d_model, d_state)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        B, L, D = x.shape
        K = self.kernel(L)
        x_fft = torch.fft.rfft(x.transpose(1, 2), n=2*L)
        K_fft = torch.fft.rfft(K, n=2*L)
        y_fft = x_fft * K_fft.unsqueeze(0)
        y = torch.fft.irfft(y_fft, n=2*L)[..., :L]
        y = y.transpose(1, 2)
        y = y + x * self.kernel.D.unsqueeze(0).unsqueeze(0)
        y = self.output_linear(self.dropout(F.gelu(y)))
        return residual + self.dropout(y)

class ProtoSSM(nn.Module):
    def __init__(self, input_dim, n_classes, d_model=128, d_state=16,
                 n_layers=2, dropout=0.15, n_families=5):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            S4DLayer(d_model, d_state, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)
        self.family_head = nn.Linear(d_model, n_families)
        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))
        self.proto_temp = nn.Parameter(torch.tensor(5.0))

    def forward(self, x, base_logits=None):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.head(x)
        if base_logits is not None:
            alpha = torch.sigmoid(self.fusion_alpha)
            logits = alpha * logits + (1 - alpha) * base_logits
        return logits

# Reshape training data to file-level sequences
WINDOWS_PER_FILE = 12
n_full_files = len(emb_full) // WINDOWS_PER_FILE
emb_files = emb_full[:n_full_files * WINDOWS_PER_FILE].reshape(n_full_files, WINDOWS_PER_FILE, -1)
scores_files = scores_full_raw[:n_full_files * WINDOWS_PER_FILE].reshape(n_full_files, WINDOWS_PER_FILE, -1)
labels_files = Y_FULL[:n_full_files * WINDOWS_PER_FILE].reshape(n_full_files, WINDOWS_PER_FILE, -1)

print(f'File-level shapes: emb={emb_files.shape}, labels={labels_files.shape}')

# Train ProtoSSM — v85: single seed with mixup + SWA
INPUT_DIM = emb_files.shape[-1]  # 1536
protossm_models = []

X_train_t = torch.FloatTensor(emb_files)
logits_train_t = torch.FloatTensor(scores_files)
Y_train_t = torch.FloatTensor(labels_files)

n_val = max(1, int(n_full_files * 0.15))
val_idx = list(range(n_full_files - n_val, n_full_files))
train_idx = list(range(n_full_files - n_val))

protossm_start = time.time()

for seed_i, seed_val in enumerate(PROTOSSM_SEEDS):
    print(f'\n  ProtoSSM seed {seed_i+1}/{len(PROTOSSM_SEEDS)} (seed={seed_val})')
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)

    model = ProtoSSM(
        input_dim=INPUT_DIM, n_classes=N_CLASSES,
        d_model=PROTOSSM_D_MODEL, d_state=PROTOSSM_D_STATE,
        n_layers=PROTOSSM_N_LAYERS, dropout=PROTOSSM_DROPOUT
    )
    if seed_i == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f'  ProtoSSM parameters: {n_params:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=PROTOSSM_LR, weight_decay=0.002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PROTOSSM_EPOCHS)

    best_val_loss = float('inf')
    wait = 0
    best_state = None
    swa_state = None
    swa_n = 0

    # Precompute pos_weight from original labels
    pos_counts_t = Y_train_t[train_idx].sum(dim=(0, 1)).clamp(min=1)
    neg_counts_t = (Y_train_t[train_idx].shape[0] * Y_train_t[train_idx].shape[1]) - pos_counts_t
    pos_weight = (neg_counts_t / pos_counts_t).clamp(max=30.0)

    for epoch in range(PROTOSSM_EPOCHS):
        model.train()
        optimizer.zero_grad()

        X_batch = X_train_t[train_idx]
        logits_batch = logits_train_t[train_idx]
        Y_batch = Y_train_t[train_idx].clone()

        # Mixup augmentation (training only)
        if PROTOSSM_MIXUP_ALPHA > 0:
            lam = np.random.beta(PROTOSSM_MIXUP_ALPHA, PROTOSSM_MIXUP_ALPHA)
            lam = max(lam, 1 - lam)
            perm = torch.randperm(X_batch.shape[0])
            X_batch = lam * X_batch + (1 - lam) * X_batch[perm]
            logits_batch = lam * logits_batch + (1 - lam) * logits_batch[perm]
            Y_batch = lam * Y_batch + (1 - lam) * Y_batch[perm]

        # Label smoothing
        if PROTOSSM_LABEL_SMOOTH > 0:
            Y_batch = Y_batch * (1 - PROTOSSM_LABEL_SMOOTH) + 0.5 * PROTOSSM_LABEL_SMOOTH

        pred = model(X_batch, logits_batch)
        loss = F.binary_cross_entropy_with_logits(pred, Y_batch, pos_weight=pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Validation (no mixup, no label smoothing)
        model.eval()
        with torch.no_grad():
            val_pred = model(X_train_t[val_idx], logits_train_t[val_idx])
            val_loss = F.binary_cross_entropy_with_logits(val_pred, Y_train_t[val_idx]).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if wait >= PROTOSSM_PATIENCE:
            print(f'    Early stop at epoch {epoch+1} (best val_loss={best_val_loss:.4f})')
            break

        # SWA: running average after warmup
        if epoch >= PROTOSSM_SWA_START:
            if swa_state is None:
                swa_state = {k: v.clone() for k, v in model.state_dict().items()}
                swa_n = 1
            else:
                swa_n += 1
                for k in swa_state:
                    swa_state[k] = swa_state[k] + (model.state_dict()[k] - swa_state[k]) / swa_n

        if (epoch + 1) % 10 == 0:
            print(f'    Epoch {epoch+1}: train={loss.item():.4f} val={val_loss:.4f} wait={wait} swa_n={swa_n}')

    # Load SWA average if available and time permits, otherwise best early-stop
    if swa_state is not None and swa_n > 1 and time_remaining() > 20 * 60:
        print(f'    Using SWA average over {swa_n} checkpoints (epochs {PROTOSSM_SWA_START}+)')
        model.load_state_dict(swa_state)
    else:
        if swa_state is not None and time_remaining() <= 20 * 60:
            print(f'    Skipping SWA (time_remaining={time_remaining()/60:.1f}min < 20min)')
        print(f'    Using best early-stop checkpoint (val_loss={best_val_loss:.4f})')
        model.load_state_dict(best_state)

    model.eval()
    protossm_models.append(model)
    print(f'    Seed {seed_val}: best_val_loss={best_val_loss:.4f}')

protossm_time = time.time() - protossm_start
print(f'ProtoSSM training time: {protossm_time:.1f}s ({len(PROTOSSM_SEEDS)} seed(s))')

# === Cell: Test Inference ===
final_tables = fit_prior_tables(sc_clean.reset_index(drop=True), Y_SC)

test_dir = BASE / 'test_soundscapes'
test_paths = sorted(test_dir.glob('*.ogg')) if test_dir.exists() else []
IS_TEST = len(test_paths) > 0

if not IS_TEST:
    print(f'No test files. Dry-run on {DRYRUN_N_FILES} train files.')
    test_paths = sorted((BASE / 'train_soundscapes').glob('*.ogg'))[:DRYRUN_N_FILES]
else:
    print(f'Test files: {len(test_paths)}')

meta_test, scores_test_raw, emb_test = infer_perch_onnx(test_paths) if (HAS_ONNX and ONNX_MODEL_PATH) else infer_perch_batch(test_paths)

test_base, test_prior = fuse_scores(
    scores_test_raw, meta_test['site'].to_numpy(),
    meta_test['hour_utc'].to_numpy(), final_tables)

Z_TEST = emb_pca.transform(emb_scaler.transform(emb_test)).astype(np.float32)

# === Cell: Apply Probes ===
print('\n=== V94: Applying MLP probes + ProtoSSM ensemble ===')

# Step 1: MLP probe predictions
probe_scores = test_base.copy()

for cls_idx, clf in tqdm(probe_models.items(), desc='MLP probes'):
    proto_sim = None
    if cls_idx in CLASS_PROTOTYPES:
        proto_sim = cosine_sim_to_prototype(Z_TEST, CLASS_PROTOTYPES[cls_idx])

    family_name = CLASS_FAMILY.get(cls_idx, 'Unknown')
    family_idxs = FAMILY_IDX_MAP.get(family_name, np.array([]))
    other_family = family_idxs[family_idxs != cls_idx]
    family_mean = test_base[:, other_family].mean(axis=1) if len(other_family) > 0 else None

    X = build_class_features(
        Z_TEST, scores_test_raw[:, cls_idx],
        test_prior[:, cls_idx], test_base[:, cls_idx],
        proto_sim_col=proto_sim, family_mean_col=family_mean)

    try:
        proba = clf.predict_proba(X)
        if proba.shape[1] == 2:
            pred = np.log(proba[:, 1] / (proba[:, 0] + 1e-8) + 1e-8)
        else:
            pred = np.zeros(len(X))
    except Exception:
        pred = np.zeros(len(X))

    probe_scores[:, cls_idx] = (1 - PROBE_ALPHA) * test_base[:, cls_idx] + PROBE_ALPHA * pred.astype(np.float32)

# Step 2: ProtoSSM predictions (multi-seed averaged)
n_test_windows = len(emb_test)
WINDOWS_PER_FILE_TEST = WINDOWS_PER_FILE
n_test_files = n_test_windows // WINDOWS_PER_FILE_TEST
remainder = n_test_windows % WINDOWS_PER_FILE_TEST

protossm_scores = np.zeros_like(test_base)

for model_i, protossm_model in enumerate(protossm_models):
    seed_scores = np.zeros_like(test_base)

    if n_test_files > 0:
        emb_test_files = emb_test[:n_test_files * WINDOWS_PER_FILE_TEST].reshape(
            n_test_files, WINDOWS_PER_FILE_TEST, -1)
        scores_test_files = scores_test_raw[:n_test_files * WINDOWS_PER_FILE_TEST].reshape(
            n_test_files, WINDOWS_PER_FILE_TEST, -1)

        with torch.no_grad():
            X_test_t = torch.FloatTensor(emb_test_files)
            logits_test_t = torch.FloatTensor(scores_test_files)
            ssm_pred = protossm_model(X_test_t, logits_test_t)
            seed_scores[:n_test_files * WINDOWS_PER_FILE_TEST] = ssm_pred.numpy().reshape(-1, N_CLASSES)

    if remainder > 0:
        start = n_test_files * WINDOWS_PER_FILE_TEST
        pad_emb = np.zeros((WINDOWS_PER_FILE_TEST, emb_test.shape[-1]), dtype=np.float32)
        pad_scores = np.zeros((WINDOWS_PER_FILE_TEST, N_CLASSES), dtype=np.float32)
        pad_emb[:remainder] = emb_test[start:]
        pad_scores[:remainder] = scores_test_raw[start:]

        with torch.no_grad():
            pred_pad = protossm_model(
                torch.FloatTensor(pad_emb).unsqueeze(0),
                torch.FloatTensor(pad_scores).unsqueeze(0)
            )
            seed_scores[start:] = pred_pad[0, :remainder].numpy()

    protossm_scores += seed_scores

protossm_scores /= len(protossm_models)

print(f'ProtoSSM score range: {protossm_scores.min():.3f} to {protossm_scores.max():.3f}')
print(f'Probe score range: {probe_scores.min():.3f} to {probe_scores.max():.3f}')

# Step 3: RANK-AVERAGE Ensemble (from v67)
def rank_average_ensemble(scores_list, weights=None):
    try:
        from scipy.stats import rankdata as _rankdata
    except (ImportError, ModuleNotFoundError):
        def _rankdata(a, method='average'):
            n = len(a)
            order = np.argsort(a)
            ranks = np.empty(n, dtype=np.float64)
            ranks[order] = np.arange(1, n + 1, dtype=np.float64)
            if method == 'average':
                sorted_a = a[order]
                i = 0
                while i < n:
                    j = i
                    while j < n - 1 and sorted_a[j + 1] == sorted_a[j]:
                        j += 1
                    if j > i:
                        avg_rank = np.mean(ranks[order[i:j+1]])
                        ranks[order[i:j+1]] = avg_rank
                    i = j + 1
            return ranks

    n = scores_list[0].shape[0]
    n_classes = scores_list[0].shape[1]

    if weights is None:
        weights = [1.0 / len(scores_list)] * len(scores_list)
    w_sum = sum(weights)
    weights = [w / w_sum for w in weights]

    ranked = []
    for scores in scores_list:
        r = np.zeros_like(scores)
        for j in range(n_classes):
            r[:, j] = _rankdata(scores[:, j], method='average') / n
        ranked.append(r)

    result = np.zeros_like(scores_list[0])
    for r, w in zip(ranked, weights):
        result += w * r
    return result

simple_ensemble = (1 - PROTOSSM_ENSEMBLE_WEIGHT) * probe_scores + PROTOSSM_ENSEMBLE_WEIGHT * protossm_scores

rank_ensemble = rank_average_ensemble(
    [probe_scores, protossm_scores],
    weights=[1 - PROTOSSM_ENSEMBLE_WEIGHT, PROTOSSM_ENSEMBLE_WEIGHT]
)

final_scores = QUANTILE_MIX_ALPHA * simple_ensemble + (1 - QUANTILE_MIX_ALPHA) * rank_ensemble
print(f'Simple ensemble range: {simple_ensemble.min():.3f} to {simple_ensemble.max():.3f}')
print(f'Rank ensemble range: {rank_ensemble.min():.3f} to {rank_ensemble.max():.3f}')
print(f'Quantile-mix range: {final_scores.min():.3f} to {final_scores.max():.3f}')

# === Cell: Post-processing + Submission ===
print('\n=== V111: Post-processing with per-class temperatures ===')

# Per-class temperatures from 0.943 cache (oof_temps_v19.npy)
# Sharpens predictions for 31 classes where model is confident
PER_CLASS_TEMPS = np.ones(N_CLASSES, dtype=np.float32)
SHARPENED = [1,3,8,9,10,15,21,24,29,32,33,36,39,40,42,46,54,57,60,63,64,97,100,103,139,141,148,214,215,225]
for idx in SHARPENED:
    PER_CLASS_TEMPS[idx] = 0.7
PER_CLASS_TEMPS[18] = 0.8

# Apply temperature scaling before sigmoid
final_scores_tempered = final_scores / PER_CLASS_TEMPS[np.newaxis, :]
final_scores_smoothed = gauss_smooth_logits(final_scores_tempered)
probs = sigmoid(final_scores_smoothed)
probs = file_context_boost(probs)
probs = np.clip(probs, 1e-8, 1.0 - 1e-8)
probs = np.power(probs, POWER_GAMMA)

print(f'Final prob range: {probs.min():.6f} to {probs.max():.6f}, mean: {probs.mean():.4f}')

# === Cell: Build Submission ===
submission = pd.DataFrame(
    data=probs.astype(np.float32),
    columns=PRIMARY_LABELS
)
submission.insert(0, 'row_id', meta_test['row_id'].values)

if IS_TEST:
    expected_ids = sample_sub['row_id'].values
    our_ids = set(submission['row_id'].values)
    expected_set = set(expected_ids)

    if our_ids != expected_set:
        print(f'WARNING: row_id mismatch! ours={len(our_ids)}, expected={len(expected_set)}')
        submission = submission.set_index('row_id').reindex(expected_ids, fill_value=0.0).reset_index()

    print(f'Submission aligned with sample_submission: {len(submission)} rows')

assert submission.columns.tolist() == ['row_id'] + PRIMARY_LABELS, 'Column mismatch!'
assert not submission.isna().any().any(), 'NaN values in submission!'
assert (submission[PRIMARY_LABELS] >= 0).all().all(), 'Negative values!'
assert (submission[PRIMARY_LABELS] <= 1).all().all(), 'Values > 1!'

submission.to_csv('submission.csv', index=False)

wall_time = time.time() - _WALL_START
print(f'\nsubmission.csv saved — {submission.shape}')
print(f'Wall time: {wall_time:.1f}s ({wall_time/60:.1f} min)')
print(f'\n=== V94 Summary ===')
print(f'  ProtoSSM + Mixup + SWA + MLP Probes + Rank-Average (PCA-64)')
print(f'  Base: v67 (proven 0.923) + mixup/SWA from v65')
print(f'  PCA={n_comp}, C={PROBE_C}, alpha={PROBE_ALPHA}')
print(f'  ProtoSSM: d_model={PROTOSSM_D_MODEL}, seeds={PROTOSSM_SEEDS}, epochs={PROTOSSM_EPOCHS}')
print(f'  Mixup alpha={PROTOSSM_MIXUP_ALPHA}, Label smooth={PROTOSSM_LABEL_SMOOTH}, SWA start={PROTOSSM_SWA_START}')
print(f'  Ensemble: quantile-mix ({QUANTILE_MIX_ALPHA} simple + {1-QUANTILE_MIX_ALPHA} rank-avg)')
print(f'  Power gamma={POWER_GAMMA}, File context alpha={FILE_CONTEXT_ALPHA}')
print(f'  Probes: {len(probe_models)} MLP + {len(protossm_models)}x ProtoSSM SSM')
print(f'  Post: Gaussian smooth → sigmoid → file-context boost → power transform')
print(submission.iloc[:3, :8])
