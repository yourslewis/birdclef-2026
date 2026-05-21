#!/usr/bin/env python3
"""
BirdCLEF 2026 v542 Afr1ste updated public946 replay.

Repo-owned controlled port of afr1ste/birdclef-2026-0-946-updated-perch-sed
(version 3), the updated Perch+distilled-SED V8-style 0.946 public stack.
This differs from v541 mainly by preserving the public notebook's full dry-run
train-row submission output, useful for diagnostics/cache validation.
"""


# %% markdown
# # BirdCLEF+ 2026 0.946 | Updated Perch + SED Pipeline
# 
# **Update note.** This is an in-place update of my existing public BirdCLEF notebook. The URL slug still contains `0.941` because that was the original public version; the current notebook content documents the stronger **0.946 public LB** line validated later.
# 
# I am updating the existing high-vote page rather than publishing a second near-duplicate notebook, so readers have one stable public reference and the vote history stays attached to the same educational resource.
# 
# ## What changed from the original 0.941 public version
# 
# | Area | Update |
# |---|---|
# | Public score anchor | Current confirmed best is `mtoshi_test_v8_v1 = 0.946` from my account. |
# | Pipeline family | Keeps the Perch + temporal sequence + SED idea, but updates the implementation to the stronger V8-style recipe. |
# | Public resources | Still all-addable Kaggle inputs, with internet disabled and CPU execution. |
# | Explanation | Adds a score card, reproducibility contract, dependency/credit table, and ablation lessons. |
# | Practical lesson | Component-only and simple weight sweeps did not beat the original blended structure. |
# 
# ## Score card from my account
# 
# | Candidate | Public LB | What it tested |
# |---|---:|---|
# | `mtoshi_test_v8_v1` | **0.946** | Strongest confirmed Perch/ProtoSSM + distilled SED V8-style blend. |
# | `mtoshi_v8_rank50p50_v1` | **0.946** | Stronger dry-run proxy, but only tied the original public score. |
# | `mtoshi_v8_rank70p30_v1` | 0.944 | More ProtoSSM weight hurt transfer. |
# | `mtoshi_v8_rank80p20_v1` | 0.942 | Even more ProtoSSM weight hurt further. |
# | `mtoshi_v8_proto_only_v1` | 0.929 | Temporal Perch branch alone is not enough. |
# | `mtoshi_v8_sed_only_v1` | 0.926 | Distilled SED branch alone is not enough. |
# 
# ## High-level flow
# 
# ```text
# Kaggle public inputs
#   |
#   +-- taxonomy / sample submission / soundscape labels
#   |       +-- label matrix and row_id schema
#   |
#   +-- hidden test audio in submit mode
#   |       +-- 60-second file -> 12 windows of 5 seconds
#   |
#   +-- Perch v2 model + ONNX export
#   |       +-- logits and 1536-dimensional embeddings
#   |
#   +-- lightweight temporal branch
#   |       +-- MLP probes
#   |       +-- ProtoSSM / ResidualSSM style sequence modeling
#   |       +-- site-hour priors
#   |       +-- isotonic calibration and temporal smoothing
#   |
#   +-- distilled SED branch
#   |       +-- mel features and public ONNX SED folds
#   |
#   +-- final rank blend
#           +-- continuity gates
#           +-- sonotype mirroring
#           +-- rare-class suppression
#           +-- submission.csv
# ```
# 
# ## Reading guide
# 
# Each code cell is preceded by a compact explanation block. When you fork the notebook, start by checking the input paths, then the output shape, then whether the final `submission.csv` has the same labels and row ordering as the sample submission.

# %% cell 1
# ============================================================
# Visual Pipeline Map
# ============================================================
# This lightweight cell draws the notebook's processing flow.
# It is only for understanding the pipeline; it does not affect predictions.

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

steps = [
    ("1. Inputs", "Kaggle data\nPublic Perch / SED assets"),
    ("2. Labels", "taxonomy.csv\nsample_submission.csv\ntrain labels"),
    ("3. Perch", "12 windows/file\nlogits + embeddings"),
    ("4. Sequence", "ProtoSSM\nMLP probes\nResidualSSM"),
    ("5. Calibration", "Priors\nthresholds\nsmoothing"),
    ("6. SED", "Mel spectrogram\npublic ONNX folds"),
    ("7. Blend", "Rank blend\npost-process\nsubmission.csv"),
]

fig, ax = plt.subplots(figsize=(16, 3.8))
ax.set_xlim(0, len(steps))
ax.set_ylim(0, 1)
ax.axis("off")

for i, (title, body) in enumerate(steps):
    x = i + 0.05
    box = FancyBboxPatch(
        (x, 0.25), 0.9, 0.5,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.5,
        facecolor="white",
        edgecolor="black",
    )
    ax.add_patch(box)
    ax.text(x + 0.45, 0.60, title, ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(x + 0.45, 0.43, body, ha="center", va="center", fontsize=9)
    if i < len(steps) - 1:
        ax.annotate("", xy=(i + 1.04, 0.5), xytext=(i + 0.96, 0.5),
                    arrowprops=dict(arrowstyle="->", linewidth=1.5))

plt.title("BirdCLEF 2026 Inference Pipeline — Big Picture", fontsize=14, fontweight="bold")
plt.show()

# %% markdown
# # Method, lineage, and compliance
# 
# This notebook is the single public continuation of the earlier 0.941 ONNX Perch Sequence + SED write-up. The code below is the stronger 0.946 V8-style line, but the educational purpose is the same: a runnable, inspectable, all-public BirdCLEF inference pipeline.
# 
# ## Architecture in one paragraph
# 
# The notebook first converts each soundscape into 5-second windows, then uses Perch to extract logits and dense embeddings. Lightweight in-notebook learners model temporal structure over the 12-window sequence, while a public distilled SED ensemble contributes spectrogram-local evidence. The final submission is not a raw average: predictions are rank blended, smoothed, gated for continuity, mirrored across selected sonotypes, clipped, and written in the exact sample-submission order.
# 
# ## Public attribution chain
# 
# Please keep this attribution if you fork or reuse the approach.
# 
# | Component | Public source |
# |---|---|
# | Updated V8 direction and final recipe | m-toshi / `testbirdclef-2026-v8` style public notebook lineage |
# | Original public page and sequence-modeling foundation | Vyanktesh Dwivedi, `birdclef-2026-onnx-perch-sequence-modeling` |
# | Distilled SED ONNX folds | Tucker Arrants, `bc2026-distilled-sed-public` |
# | Perch metadata and ONNX export | Jaejohn `perch-meta`, Rishikesh Jani `perch-onnx-for-birdclef-2026` |
# | TensorFlow wheel helper | Ashok205, `tf-wheels` |
# | Base audio embedding model | Google Bird Vocalization Classifier, Perch v2 CPU |
# 
# ## Reproducibility contract
# 
# | Check | Expected result |
# |---|---|
# | Kaggle inputs | Competition source plus the public datasets/models listed in the sidebar. |
# | Internet | Disabled. |
# | GPU | Not required for this notebook version. |
# | Public dry-run output | 240 rows x 235 columns, zero NaNs. |
# | Formal submit output | One row per hidden test file x 12 windows, `row_id` plus 234 class columns. |
# 
# Dry-run metrics on train soundscapes are useful for catching broken rows, NaNs, and obvious regressions. They are not a reliable leaderboard oracle; the SED-heavy dry-run proxy is especially optimistic.

# %% markdown
# ### 🔎 Offline wheel installation
# 
# | Item | Details |
# |---|---|
# | **Purpose** | Installs ONNX Runtime and TensorFlow from attached Kaggle wheel assets. |
# | **Input** | Wheel files under `/kaggle/input`. |
# | **Output** | Available `onnxruntime` and TensorFlow runtime. |
# | **Risk / check** | If wheel versions or Python version mismatch, install may fail. |

# %% cell 4
import subprocess, sys, os
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")

def find_wheel(pattern):
    for p in INPUT_ROOT.rglob(pattern):
        return p
    raise FileNotFoundError(pattern)

ONNX_WHL = Path("/kaggle/input/datasets/rishikeshjani/perch-onnx-for-birdclef-2026/onnxruntime-1.24.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl")
if ONNX_WHL.exists():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(ONNX_WHL)], check=True)
    print("ONNX Runtime installed")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                str(find_wheel("tensorboard-2.20.0-*.whl"))], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                str(find_wheel("tensorflow-2.20.0-*.whl"))], check=True)
print("TF 2.20 installed")

try:
    import onnxruntime as ort
    _ONNX_AVAILABLE = True
    print("ONNX Runtime available")
except ImportError:
    _ONNX_AVAILABLE = False
    print("ONNX not available, falling back to TF")

# %% markdown
# ### 🔎 Reproducibility seed setup
# 
# | Item | Details |
# |---|---|
# | **Purpose** | Fixes Python, NumPy, and PyTorch random seeds to make training behavior more stable. |
# | **Input** | Seed value, default `42`. |
# | **Output** | Deterministic random state for later lightweight training cells. |
# | **Risk / check** | Full determinism is not guaranteed across hardware/library versions, but this reduces run-to-run variance. |

# %% cell 6
import random
import os
import numpy as np
import torch

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)
print("Global random seed set to 42")

# %% markdown
# ### 🔎 Execution mode selector
# 
# | Item | Details |
# |---|---|
# | **Purpose** | Switches between training/debug behavior and submission behavior. |
# | **Input** | `MODE`, usually `submit` for Kaggle submission. |
# | **Output** | Validated mode used by configuration cells. |
# | **Risk / check** | Use `train` only when you want OOF evaluation and are willing to spend more runtime. |

# %% cell 8
MODE = "submit"   
 
assert MODE in {"train", "submit"}
print("MODE =", MODE)

# %% markdown
# ### 🔎 Global imports, paths, constants, and configuration
# 
# | Item | Details |
# |---|---|
# | **Purpose** | Defines the core runtime constants: audio sample rate, window size, class/window counts, paths, and model hyperparameters. |
# | **Input** | Kaggle input directory and selected execution mode. |
# | **Output** | `BASE`, `MODEL_DIR`, `WORK_DIR`, `SR`, `N_WINDOWS`, `CFG`, and TensorFlow runtime settings. |
# | **Risk / check** | Most downstream shape errors come from changing `N_WINDOWS`, `WINDOW_SEC`, or class order inconsistently. |

# %% cell 10
import os, re, gc, time, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
 
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm
 
tf.experimental.numpy.experimental_enable_numpy_behavior()
try: tf.config.set_visible_devices([], "GPU")
except: pass
 
_WALL_START = time.time()
 
BASE      = Path("/kaggle/input/competitions/birdclef-2026")
MODEL_DIR = Path("/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1")
WORK_DIR  = Path("/kaggle/working/cache")
WORK_DIR.mkdir(parents=True, exist_ok=True)
 
SR             = 32_000
WINDOW_SEC     = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES   = 60 * SR
N_WINDOWS      = 12          
 
CFG = {
    "batch_files": 16,
    "oof_n_splits": 5   if MODE == "train" else 3,
    "dryrun_n_files": 20 if MODE == "train" else 0,
    "run_oof": MODE == "train",
    "verbose": MODE == "train",
    "proto_ssm_train": {
        "n_epochs":        80  if MODE == "train" else 40,
        "lr":              8e-4,
        "weight_decay":    1e-3,
        "val_ratio":       0.15,
        "patience":        20  if MODE == "train" else 8,
        "pos_weight_cap":  25.0,
        "distill_weight":  0.15,
        "proto_margin":    0.15,
        "label_smoothing": 0.03,
        "oof_n_splits":    5   if MODE == "train" else 3,
        "mixup_alpha":     0.4,
        "focal_gamma":     2.5,
        "swa_start_frac":  0.65,
        "swa_lr":          4e-4,
        "use_cosine_restart": True,
        "restart_period":  20,
    },
    "residual_ssm": {
        "d_model": 128, "d_state": 16, "n_ssm_layers": 2,
        "dropout": 0.1, "correction_weight": 0.35,
        "n_epochs": 40  if MODE == "train" else 20,
        "lr": 8e-4,
        "patience": 12  if MODE == "train" else 6,
    },
    "mlp_params": {
        "hidden_layer_sizes": (256, 128), "activation": "relu",
        "max_iter": 500  if MODE == "train" else 200,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 20  if MODE == "train" else 10,
        "random_state": 42,
        "learning_rate_init": 5e-4,
        "alpha": 0.005,
    },
}
print("CFG loaded")
print(f"  n_epochs={CFG['proto_ssm_train']['n_epochs']}  "
      f"patience={CFG['proto_ssm_train']['patience']}  "
      f"oof_n_splits={CFG['proto_ssm_train']['oof_n_splits']}  "
      f"mlp_max_iter={CFG['mlp_params']['max_iter']}")
 
print("Config ready")
print(f"  run_oof={CFG['run_oof']}  verbose={CFG['verbose']}  dryrun={CFG['dryrun_n_files']}")

# %% markdown
# ## Data and Label Setup
# 
# The competition target has 234 scored classes. The notebook reads `taxonomy.csv` and `sample_submission.csv`, then constructs the row/column order required by Kaggle. It also parses train soundscape labels for the internal dry-run and for building sequence-model training targets.
# 
# The hidden test set is not available during ordinary public execution. This is expected for a Kaggle code competition: the same notebook will see real `test_soundscapes` only when Kaggle reruns it as a formal competition submission.
# 
# 
# ---

# %% markdown
# ### 🔎 Data schema and label matrix setup
# 
# | Item | Details |
# |---|---|
# | **Purpose** | Builds the scored class list, parses soundscape filenames, groups labels, and creates the multi-label target matrix. |
# | **Input** | `taxonomy.csv`, `sample_submission.csv`, and `train_soundscapes_labels.csv`. |
# | **Output** | `PRIMARY_LABELS`, `N_CLASSES`, `label_to_idx`, `sc`, `Y_SC`, `full_files`, `Y_FULL`. |
# | **Risk / check** | The class order must follow `sample_submission.csv`; changing it can silently corrupt submissions. |

# %% cell 13
taxonomy          = pd.read_csv(BASE / "taxonomy.csv")
sample_sub        = pd.read_csv(BASE / "sample_submission.csv")
soundscape_labels = pd.read_csv(BASE / "train_soundscapes_labels.csv")
 
PRIMARY_LABELS = sample_sub.columns[1:].tolist()
N_CLASSES      = len(PRIMARY_LABELS)
label_to_idx   = {c: i for i, c in enumerate(PRIMARY_LABELS)}
 
FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")
 
def parse_fname(name):
    m = FNAME_RE.match(name)
    if not m: return {"site": "unknown", "hour_utc": -1}
    _, site, _, hms = m.groups()
    return {"site": site, "hour_utc": int(hms[:2])}
 
def union_labels(series):
    out = set()
    for x in series:
        if pd.notna(x):
            for t in str(x).split(";"):
                t = t.strip()
                if t: out.add(t)
    return sorted(out)
 
sc = (soundscape_labels
      .groupby(["filename", "start", "end"])["primary_label"]
      .apply(union_labels)
      .reset_index(name="label_list"))
 
sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)
sc["row_id"]  = sc["filename"].str.replace(".ogg", "", regex=False) + "_" + sc["end_sec"].astype(str)
 
_meta = sc["filename"].apply(parse_fname).apply(pd.Series)
sc = pd.concat([sc, _meta], axis=1)
 
Y_SC = np.zeros((len(sc), N_CLASSES), dtype=np.uint8)
for i, lbls in enumerate(sc["label_list"]):
    for lbl in lbls:
        if lbl in label_to_idx:
            Y_SC[i, label_to_idx[lbl]] = 1
 
windows_per_file = sc.groupby("filename").size()
full_files = sorted(windows_per_file[windows_per_file == N_WINDOWS].index.tolist())
sc["fully_labeled"] = sc["filename"].isin(full_files)
 
full_rows = (sc[sc["fully_labeled"]]
             .sort_values(["filename", "end_sec"])
             .reset_index(drop=False))
Y_FULL = Y_SC[full_rows["index"].to_numpy()]
 
print(f"Classes: {N_CLASSES} | Fully-labeled files: {len(full_files)}")
print(f"Full-file windows: {len(full_rows)} | Active classes: {int((Y_FULL.sum(0) > 0).sum())}")

# %% markdown
# ## Perch Backbone
# 
# Perch provides strong acoustic embeddings and class logits. The notebook prefers an attached ONNX export for speed, while still keeping the Kaggle TensorFlow Perch model available. In our submitted run, ONNX Perch was used and the train cache was built in roughly 2.5 minutes.
# 
# A useful implementation detail is the species mapping step. Most target classes map directly to Perch logits; a few unmapped targets can borrow genus-level proxy signal; the remaining unmapped species are handled by the downstream learned and prior components rather than by direct Perch logits.
# 
# 
# ---

# %% markdown
# ### 🔎 Perch backbone loading and target-species mapping
# 
# | Item | Details |
# |---|---|
# | **Purpose** | Loads Perch and maps BirdCLEF target species to available Perch logits. |
# | **Input** | Google Perch SavedModel, optional Perch ONNX file, taxonomy, and Perch `labels.csv`. |
# | **Output** | `USE_ONNX`, `ONNX_SESSION`, `BC_INDICES`, `MAPPED_MASK`, mapped Perch class positions. |
# | **Risk / check** | If ONNX is missing, the notebook falls back to TensorFlow SavedModel, usually slower. |

# %% cell 16
birdclassifier = tf.saved_model.load(str(MODEL_DIR))
infer_fn       = birdclassifier.signatures["serving_default"]

ONNX_PERCH_PATH = Path("/kaggle/input/datasets/rishikeshjani/perch-onnx-for-birdclef-2026/perch_v2.onnx")
USE_ONNX = _ONNX_AVAILABLE and ONNX_PERCH_PATH.exists()

if USE_ONNX:
    _so = ort.SessionOptions()
    _so.intra_op_num_threads = 4
    ONNX_SESSION    = ort.InferenceSession(str(ONNX_PERCH_PATH), sess_options=_so,
                                            providers=["CPUExecutionProvider"])
    ONNX_INPUT_NAME = ONNX_SESSION.get_inputs()[0].name
    ONNX_OUT_MAP    = {o.name: i for i, o in enumerate(ONNX_SESSION.get_outputs())}
    print("Using ONNX Perch (150x faster)")
else:
    print("Using TF SavedModel Perch")

bc_labels = (pd.read_csv(MODEL_DIR / "assets" / "labels.csv")
             .reset_index()
             .rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"}))
NO_LABEL = len(bc_labels)

mapping = (taxonomy
           .merge(bc_labels.rename(columns={"scientific_name": "scientific_name"}),
                  on="scientific_name", how="left"))
mapping["bc_index"] = mapping["bc_index"].fillna(NO_LABEL).astype(int)
lbl2bc = mapping.set_index("primary_label")["bc_index"]

BC_INDICES    = np.array([int(lbl2bc.loc[c]) for c in PRIMARY_LABELS], dtype=np.int32)
MAPPED_MASK   = BC_INDICES != NO_LABEL
MAPPED_POS    = np.where(MAPPED_MASK)[0].astype(np.int32)
MAPPED_BC_IDX = BC_INDICES[MAPPED_MASK].astype(np.int32)

print(f"Mapped: {MAPPED_MASK.sum()} / {N_CLASSES} species have a Perch logit")

# %% markdown
# ### 🔎 Genus-level proxy mapping for unmapped species
# 
# | Item | Details |
# |---|---|
# | **Purpose** | Finds fallback Perch logits for target species that lack an exact Perch class match. |
# | **Input** | Unmapped taxonomy rows and Perch label metadata. |
# | **Output** | `proxy_map` for genus-level signal transfer. |
# | **Risk / check** | Proxy signal is weaker than direct mapping; inspect printed proxy targets for sanity. |

# %% cell 18
import re as _re
UNMAPPED_POS  = np.where(~MAPPED_MASK)[0].astype(np.int32)

CLASS_NAME_MAP = taxonomy.set_index("primary_label")["class_name"].to_dict()
TEXTURE_TAXA   = {"Amphibia", "Insecta"}

proxy_map = {}   

unmapped_df = (taxonomy[taxonomy["primary_label"]
               .isin([PRIMARY_LABELS[i] for i in UNMAPPED_POS])]
               .copy())

for _, row in unmapped_df.iterrows():
    target = row["primary_label"]
    sci    = str(row["scientific_name"])
    genus  = sci.split()[0]
    hits = bc_labels[
        bc_labels["scientific_name"]
        .astype(str)
        .str.match(rf"^{_re.escape(genus)}\s", na=False)
    ]
    
    if len(hits) > 0:
        proxy_map[label_to_idx[target]] = hits["bc_index"].astype(int).tolist()

PROXY_TAXA = {"Amphibia", "Insecta", "Aves"}
proxy_map  = {
    idx: bc_idxs
    for idx, bc_idxs in proxy_map.items()
    if CLASS_NAME_MAP.get(PRIMARY_LABELS[idx]) in PROXY_TAXA
}

print(f"Unmapped species total:        {len(UNMAPPED_POS)}")
print(f"Species with genus proxy:      {len(proxy_map)}")
print(f"Species still without signal:  {len(UNMAPPED_POS) - len(proxy_map)}")
print("\nProxy targets:")
for idx, bc_idxs in list(proxy_map.items())[:8]:
    label = PRIMARY_LABELS[idx]
    cls   = CLASS_NAME_MAP.get(label, "?")
    print(f"  {label:12s} ({cls:10s}) ← {len(bc_idxs)} Perch genus matches")

# %% markdown
# ## Window-Level Inference Cache
# 
# The inference engine splits each 60-second soundscape into twelve 5-second windows. Caching the Perch scores and 1536-dimensional embeddings is important because later cells train several lightweight models over the same windows.
# 
# The cache also makes the notebook easier to debug: if a later modeling cell changes, the expensive audio pass does not have to be repeated inside the same run.
# 
# 
# ---

# %% markdown
# ### 🔎 Genus-level proxy mapping for unmapped species
# 
# | Item | Details |
# |---|---|
# | **Purpose** | Finds fallback Perch logits for target species that lack an exact Perch class match. |
# | **Input** | Unmapped taxonomy rows and Perch label metadata. |
# | **Output** | `proxy_map` for genus-level signal transfer. |
# | **Risk / check** | Proxy signal is weaker than direct mapping; inspect printed proxy targets for sanity. |

# %% cell 21
import concurrent.futures

def read_60s(path):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2: y = y.mean(axis=1)
    if len(y) < FILE_SAMPLES: y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    else:                      y = y[:FILE_SAMPLES]
    return y

def run_perch(paths, batch_files=16, verbose=True):
    paths  = [Path(p) for p in paths]
    n_rows = len(paths) * N_WINDOWS

    row_ids   = np.empty(n_rows, dtype=object)
    filenames = np.empty(n_rows, dtype=object)
    sites     = np.empty(n_rows, dtype=object)
    hours     = np.zeros(n_rows, dtype=np.int16)
    scores    = np.zeros((n_rows, N_CLASSES), dtype=np.float32)
    embs      = np.zeros((n_rows, 1536),      dtype=np.float32)

    wr  = 0
    itr = tqdm(range(0, len(paths), batch_files), desc="Perch") if verbose else range(0, len(paths), batch_files)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as io_executor:
        next_paths   = paths[0:batch_files]
        future_audio = [io_executor.submit(read_60s, p) for p in next_paths]

        for start in itr:
            batch_paths  = next_paths
            batch_n      = len(batch_paths)
            batch_audio  = [f.result() for f in future_audio]
            next_start = start + batch_files
            if next_start < len(paths):
                next_paths   = paths[next_start:next_start + batch_files]
                future_audio = [io_executor.submit(read_60s, p) for p in next_paths]

            x  = np.empty((batch_n * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
            br = wr

            for bi, path in enumerate(batch_paths):
                y    = batch_audio[bi]
                meta = parse_fname(path.name)
                stem = path.stem
                x[bi * N_WINDOWS:(bi + 1) * N_WINDOWS] = y.reshape(N_WINDOWS, WINDOW_SAMPLES)
                row_ids  [wr:wr + N_WINDOWS] = [f"{stem}_{t}" for t in range(5, 65, 5)]
                filenames[wr:wr + N_WINDOWS] = path.name
                sites    [wr:wr + N_WINDOWS] = meta["site"]
                hours    [wr:wr + N_WINDOWS] = meta["hour_utc"]
                wr += N_WINDOWS

            if USE_ONNX:
                outs   = ONNX_SESSION.run(None, {ONNX_INPUT_NAME: x})
                logits = outs[ONNX_OUT_MAP["label"]].astype(np.float32)
                emb    = outs[ONNX_OUT_MAP["embedding"]].astype(np.float32)
            else:
                out    = infer_fn(inputs=tf.convert_to_tensor(x))
                logits = out["label"].numpy().astype(np.float32)
                emb    = out["embedding"].numpy().astype(np.float32)

            scores[br:wr, MAPPED_POS] = logits[:, MAPPED_BC_IDX]
            embs  [br:wr]             = emb

            for pos_idx, bc_idxs in proxy_map.items():
                bc_arr = np.array(bc_idxs, dtype=np.int32)
                scores[br:wr, pos_idx] = logits[:, bc_arr].max(axis=1)

            del x, logits, emb, batch_audio
            gc.collect()

    meta_df = pd.DataFrame({"row_id": row_ids, "filename": filenames,
                             "site": sites, "hour_utc": hours})
    return meta_df, scores, embs

print("Perch inference engine defined")

# %% markdown
# ### 🔎 Perch cache discovery, build, and alignment
# 
# | Item | Details |
# |---|---|
# | **Purpose** | Uses an existing public/local Perch cache when available, otherwise builds one from train soundscapes. |
# | **Input** | External cache directories, local working cache, `full_files`, and `run_perch()`. |
# | **Output** | `meta_tr`, `sc_tr`, `emb_tr`, and `Y_FULL_aligned`. |
# | **Risk / check** | Cache row IDs must align with labels. If alignment fails, rebuild the local cache. |

# %% cell 23
print(f"USE_ONNX = {USE_ONNX}  "
      f"(cache will be built with {'ONNX' if USE_ONNX else 'TF SavedModel'})")

EXTERNAL_CACHE_DIRS = [
    Path("/kaggle/input/notebooks/vyankteshdwivedi/notebook1b25083f0d"),
    Path("/kaggle/input/datasets/jaejohn/perch-meta"),
]
CACHE_META_LOCAL = WORK_DIR / "perch_meta.parquet"
CACHE_NPZ_LOCAL  = WORK_DIR / "perch_arrays.npz"

def _find_external_cache():
    for d in EXTERNAL_CACHE_DIRS:
        meta = d / "perch_meta.parquet"
        npz  = d / "perch_arrays.npz"
        if meta.exists() and npz.exists():
            return meta, npz
    return None, None

SCORE_KEYS = ["scores", "sc", "logits", "perch_scores", "preds", "arr_0"]
EMB_KEYS   = ["embs", "emb", "embeddings", "features", "perch_embs", "arr_1"]

def _pick_array(arr, candidates, shape_hint_cols):
    for k in candidates:
        if k in arr.files:
            return arr[k], k

    for k in arr.files:
        v = arr[k]
        if v.ndim == 2 and v.shape[1] == shape_hint_cols:
            return v, k

    raise KeyError(f"None of {candidates} found in npz. Available keys: {arr.files}")


def _build_cache():
    print(f"Building Perch cache from {len(full_files)} training files…")
    train_paths = [BASE / "train_soundscapes" / fn for fn in full_files]
    train_paths = [p for p in train_paths if p.exists()]
    t0 = time.time()
    meta_built, sc_built, emb_built = run_perch(
        train_paths,
        batch_files=CFG["batch_files"],
        verbose=True
    )
    print(f"  Perch pass done in {time.time()-t0:.1f}s  "
          f"scores={sc_built.shape} embs={emb_built.shape}")

    meta_built.to_parquet(CACHE_META_LOCAL)

    np.savez(
        CACHE_NPZ_LOCAL,
        scores=sc_built.astype(np.float32),
        embs=emb_built.astype(np.float32),
        primary_labels=np.array(PRIMARY_LABELS)
    )
    print(f"  Cache saved to {WORK_DIR}")
    return CACHE_META_LOCAL, CACHE_NPZ_LOCAL


ext_meta, ext_npz = _find_external_cache()

if ext_meta is not None:
    CACHE_META, CACHE_NPZ = ext_meta, ext_npz
    print(f"Using external cache: {CACHE_META.parent}")
elif CACHE_META_LOCAL.exists() and CACHE_NPZ_LOCAL.exists():
    CACHE_META, CACHE_NPZ = CACHE_META_LOCAL, CACHE_NPZ_LOCAL
    print(f"Using local cache: {WORK_DIR}")
else:
    print("No cache found — building from scratch (~1.5 min)")
    CACHE_META, CACHE_NPZ = _build_cache()

print("Loading Perch cache from:", CACHE_META.parent)

meta_tr = pd.read_parquet(CACHE_META)
_arr    = np.load(CACHE_NPZ)

sc_tr_raw,  sk = _pick_array(_arr, SCORE_KEYS, N_CLASSES)
emb_tr_raw, ek = _pick_array(_arr, EMB_KEYS,   1536)

print(f"  scores ← '{sk}'  shape={sc_tr_raw.shape}")
print(f"  embs   ← '{ek}'  shape={emb_tr_raw.shape}")

sc_tr  = sc_tr_raw.astype(np.float32)
emb_tr = emb_tr_raw.astype(np.float32)

if "primary_labels" in _arr.files:
    if _arr["primary_labels"].tolist() != PRIMARY_LABELS:
        print("  WARNING: cached primary_labels differ — scores columns may not align!")
    else:
        print("  primary_labels schema OK")


if "row_id" not in meta_tr.columns:
    print("  row_id missing — reconstructing")

    if "end_sec" in meta_tr.columns:
        end_sec = meta_tr["end_sec"].astype(int)
    elif "window_idx" in meta_tr.columns:
        end_sec = (meta_tr["window_idx"].astype(int) + 1) * 5
    else:
        end_sec = np.tile(np.arange(5, 65, 5), len(meta_tr) // N_WINDOWS)

    meta_tr["row_id"] = (
        meta_tr["filename"].str.replace(".ogg", "", regex=False)
        + "_" + end_sec.astype(str)
    )

row_id_to_index = full_rows.set_index("row_id")["index"]
missing_rows = set(meta_tr["row_id"]) - set(row_id_to_index.index)

if missing_rows:
    raise RuntimeError(
        f"Cache has {len(missing_rows)} row_ids not in labeled set. "
        f"Delete {CACHE_META_LOCAL} and {CACHE_NPZ_LOCAL} to rebuild."
    )

Y_FULL_aligned = Y_SC[
    row_id_to_index.loc[meta_tr["row_id"]].to_numpy()
]
print(f"sc_tr: {sc_tr.shape}  emb_tr: {emb_tr.shape}  Y_FULL_aligned: {Y_FULL_aligned.shape}")

# %% markdown
# ## Validation and Post-Processing Helpers
# 
# The local dry-run metric is useful for finding broken outputs, but it is not a replacement for public LB. Several SED-only candidates scored extremely high on train-soundscape dry-runs and then underperformed on the official public LB. For this reason, the dry-run is treated as a sanity check plus ranking clue, not as proof.
# 
# The post-processing stack combines:
# 
# - temporal smoothing across adjacent 5-second windows,
# - site/hour priors from train soundscape metadata,
# - file-level confidence scaling,
# - per-taxon temperature scaling,
# - rank-aware scaling,
# - adaptive smoothing that reacts to prediction deltas.
# 
# 
# ---

# %% markdown
# ### 🔎 Validation metric and temporal smoothing helpers
# 
# | Item | Details |
# |---|---|
# | **Purpose** | Defines macro AUC, GroupKFold OOF evaluation, and simple neighbor smoothing. |
# | **Input** | Prediction matrices, label matrices, metadata grouped by filename. |
# | **Output** | `macro_auc()`, `honest_oof_auc()`, `smooth_predictions()`. |
# | **Risk / check** | Validation must group by filename; random row splits leak windows from the same file. |

# %% cell 26
def macro_auc(y_true, y_score):
    """
    Exact replica of the competition metric:
    macro-averaged ROC-AUC, skipping classes with no positive labels.
    This is the ONLY number you should track locally.
    """
    keep = y_true.sum(axis=0) > 0
    return roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro")
 
 
def honest_oof_auc(scores, Y, meta_df, n_splits=5, label="scores"):
    """
    GroupKFold by filename — files never split across folds.
    This is the only correct way to estimate LB performance locally.
    Leaking a file across train/val inflates AUC by ~0.01–0.03.
    """
    groups = meta_df["filename"].to_numpy()
    gkf    = GroupKFold(n_splits=n_splits)
    oof    = np.zeros_like(scores, dtype=np.float32)
 
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(scores, groups=groups), 1):
        oof[va_idx] = scores[va_idx]
 
    auc = macro_auc(Y, oof)
    print(f"[{label}] honest OOF macro-AUC: {auc:.6f}")
    return auc, oof

# %% markdown
# ### 🔎 Code Cell
# 
# | Item | Details |
# |---|---|
# | **Purpose** | Runs one step of the BirdCLEF pipeline. |
# | **Input** | Variables/files created by earlier cells. |
# | **Output** | Variables used by later cells. |
# | **Risk / check** | Check printed shapes, paths, and whether expected files exist. |

# %% cell 28
def smooth_predictions(probs, n_windows=12, alpha=0.3):
    """
    For each file's 12 windows, blend each window with its neighbors.
    new[t] = (1 - alpha) * old[t] + 0.5*alpha * (old[t-1] + old[t+1])
    alpha=0: no smoothing (your current baseline)
    alpha=0.3: moderate smoothing (good starting point)
    Shape: (n_files * 12, n_classes) → same shape output
    """
    N, C = probs.shape
    assert N % n_windows == 0, f"Expected multiple of {n_windows}, got {N}"
    view = probs.reshape(-1, n_windows, C).copy()
    prev_w = np.concatenate([view[:, :1, :],  view[:, :-1, :]], axis=1)  
    next_w = np.concatenate([view[:, 1:,  :], view[:, -1:, :]], axis=1) 
    smoothed = (1 - alpha) * view + 0.5 * alpha * (prev_w + next_w)
    return smoothed.reshape(N, C)

print("Temporal smoothing helper defined")

# %% markdown
# ## Prior Probability Tables
# 
# This section calculates the frequency of species occurrences based on site and time of day. We construct a 3-tier prior: global frequency, independent site and hour frequencies, and a joint site-hour bucket. 
# 
# Because the joint site-hour combinations have fewer samples, we apply a tighter Bayesian shrinkage factor to prevent overfitting to sparse acoustic environments. These priors are converted to log-odds and added directly to the raw Perch logits.
# 
# ---

# %% markdown
# ### 🔎 Site/hour prior probability tables
# 
# | Item | Details |
# |---|---|
# | **Purpose** | Creates global, site, hour, and site-hour priors for ecological context. |
# | **Input** | Soundscape metadata and label matrix. |
# | **Output** | `build_prior_tables()` and `apply_prior()`. |
# | **Risk / check** | Priors help when train/test ecology matches, but too much prior weight can overfit local metadata. |

# %% cell 31
def build_prior_tables(sc_df, Y_labels):
    sc_df = sc_df.reset_index(drop=True)
    global_p = Y_labels.mean(axis=0).astype(np.float32)

    site_keys = sorted(sc_df["site"].dropna().astype(str).unique())
    site_to_i = {k: i for i, k in enumerate(site_keys)}
    site_p = np.zeros((len(site_keys), Y_labels.shape[1]), dtype=np.float32)
    site_n = np.zeros(len(site_keys), dtype=np.float32)
    for s in site_keys:
        i = site_to_i[s]
        mask = sc_df["site"].astype(str).values == s
        site_n[i] = mask.sum()
        site_p[i] = Y_labels[mask].mean(axis=0)

    hour_keys = sorted(sc_df["hour_utc"].dropna().astype(int).unique())
    hour_to_i = {h: i for i, h in enumerate(hour_keys)}
    hour_p = np.zeros((len(hour_keys), Y_labels.shape[1]), dtype=np.float32)
    hour_n = np.zeros(len(hour_keys), dtype=np.float32)
    for h in hour_keys:
        i = hour_to_i[h]
        mask = sc_df["hour_utc"].astype(int).values == h
        hour_n[i] = mask.sum()
        hour_p[i] = Y_labels[mask].mean(axis=0)

    sh_keys = sorted({(str(s), int(h)) for s, h in zip(sc_df["site"].dropna(), sc_df["hour_utc"].dropna())
                      if not pd.isna(s) and not pd.isna(h)})
    sh_to_i = {k: i for i, k in enumerate(sh_keys)}
    sh_p = np.zeros((len(sh_keys), Y_labels.shape[1]), dtype=np.float32)
    sh_n = np.zeros(len(sh_keys), dtype=np.float32)
    for (s, h) in sh_keys:
        i = sh_to_i[(s, h)]
        mask = (sc_df["site"].astype(str).values == s) & (sc_df["hour_utc"].astype(int).values == h)
        sh_n[i] = mask.sum()
        sh_p[i] = Y_labels[mask].mean(axis=0)

    return {
        "global_p": global_p,
        "site_to_i": site_to_i, "site_p": site_p, "site_n": site_n,
        "hour_to_i": hour_to_i, "hour_p": hour_p, "hour_n": hour_n,
        "sh_to_i": sh_to_i, "sh_p": sh_p, "sh_n": sh_n,
    }

def apply_prior(scores, sites, hours, tables, lambda_prior=0.4):
    eps = 1e-4
    n = len(scores)
    out = scores.copy()

    p = np.tile(tables["global_p"], (n, 1))

    for i, h in enumerate(hours):
        h = int(h)
        if h in tables["hour_to_i"]:
            j = tables["hour_to_i"][h]
            nh = tables["hour_n"][j]
            w = nh / (nh + 8.0)
            p[i] = w * tables["hour_p"][j] + (1 - w) * tables["global_p"]

    for i, s in enumerate(sites):
        s = str(s)
        if s in tables["site_to_i"]:
            j = tables["site_to_i"][s]
            ns = tables["site_n"][j]
            w = ns / (ns + 8.0)
            p[i] = w * tables["site_p"][j] + (1 - w) * p[i]

    if "sh_to_i" in tables:
        for i, (s, h) in enumerate(zip(sites, hours)):
            key = (str(s), int(h))
            if key in tables["sh_to_i"]:
                j = tables["sh_to_i"][key]
                nsh = tables["sh_n"][j]
                w = nsh / (nsh + 4.0)
                p[i] = w * tables["sh_p"][j] + (1 - w) * p[i]

    p = np.clip(p, eps, 1 - eps)
    logit_prior = np.log(p) - np.log1p(-p)
    out += lambda_prior * logit_prior

    return out.astype(np.float32)

print("Prior tables defined")

# %% markdown
# ### 🔎 File-level confidence scaling
# 
# | Item | Details |
# |---|---|
# | **Purpose** | Suppresses uncertain files by using top-k file-level confidence as a scale factor. |
# | **Input** | Window-level probabilities. |
# | **Output** | `file_confidence_scale()`. |
# | **Risk / check** | Aggressive power values can suppress rare true positives. |

# %% cell 33
def file_confidence_scale(probs, n_windows=12, top_k=2, power=0.4):
    """
    Scale each window's predictions by how confident the file is overall.
    
    Steps:
    1. For each file, find the top-k highest scores across all 12 windows
    2. Compute their mean → "file confidence"
    3. Multiply every window's scores by (file_confidence ** power)
    
    power=0: no effect (baseline)
    power=0.4: moderate suppression of uncertain files
    
    Why top-k and not max?
    Max is noisy (one lucky spike). Top-2 mean is more robust.
    """
    N, C = probs.shape
    assert N % n_windows == 0
    
    view      = probs.reshape(-1, n_windows, C)       
    sorted_v  = np.sort(view, axis=1)                 
    top_k_mean = sorted_v[:, -top_k:, :].mean(axis=1, keepdims=True)  
    
    scale  = np.power(top_k_mean, power)              
    scaled = view * scale                             
    
    return scaled.reshape(N, C)

print("File-level confidence scaling defined")

# %% markdown
# ### 🔎 Taxon-aware temperature scaling
# 
# | Item | Details |
# |---|---|
# | **Purpose** | Applies different score temperatures to texture-like taxa and event-like species. |
# | **Input** | Taxonomy class names and target label list. |
# | **Output** | `temperatures` vector of length `N_CLASSES`. |
# | **Risk / check** | Changing temperature values changes calibration, not just ranking. |

# %% cell 35
CLASS_NAME_MAP = taxonomy.set_index("primary_label")["class_name"].to_dict()
TEXTURE_TAXA   = {"Amphibia", "Insecta"}   

temperatures = np.ones(N_CLASSES, dtype=np.float32)
for ci, label in enumerate(PRIMARY_LABELS):
    cls = CLASS_NAME_MAP.get(label, "Aves")
    if cls in TEXTURE_TAXA:
        temperatures[ci] = 0.95   
    else:
        temperatures[ci] = 1.10   

n_texture = (temperatures < 1.0).sum()
n_event   = (temperatures > 1.0).sum()
print(f"Temperatures: {n_event} event species (T=1.10), {n_texture} texture species (T=0.95)")

# %% markdown
# ## Lightweight Perch-Embedding Learners
# 
# The notebook trains small models inside the inference notebook rather than relying on private checkpoints. This keeps the solution self-contained and public-input compliant.
# 
# The MLP probe branch uses PCA-compressed Perch embeddings and only trains classes with enough positive windows. The calibration block then uses isotonic calibration and class-level threshold estimates to keep probabilities better behaved before the final blend.
# 
# 
# ---

# %% markdown
# ### 🔎 Embedding-based MLP probe training
# 
# | Item | Details |
# |---|---|
# | **Purpose** | Trains small per-class MLP probes using compressed Perch embeddings plus temporal score features. |
# | **Input** | Perch embeddings, raw scores, labels, PCA/scaler settings. |
# | **Output** | `train_mlp_probes()` and `apply_mlp_probes()`. |
# | **Risk / check** | Classes with too few positive examples are skipped to avoid unstable probes. |

# %% cell 38
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

def build_class_freq_weights(Y, cap=10.0):
    total     = Y.shape[0]
    pos_count = Y.sum(axis=0).astype(np.float32) + 1.0
    freq      = pos_count / total
    weights   = 1.0 / (freq ** 0.5)
    weights   = np.clip(weights, 1.0, cap)
    weights   = weights / weights.mean()
    return weights.astype(np.float32)


def build_sequential_features(scores_col, n_windows=12):
    N = len(scores_col)
    assert N % n_windows == 0
    x     = scores_col.reshape(-1, n_windows)
    prev  = np.concatenate([x[:, :1], x[:, :-1]], axis=1)
    next_ = np.concatenate([x[:, 1:], x[:, -1:]], axis=1)
    mean  = np.repeat(x.mean(axis=1), n_windows)
    max_  = np.repeat(x.max(axis=1),  n_windows)
    std   = np.repeat(x.std(axis=1),  n_windows)
    return prev.reshape(-1), next_.reshape(-1), mean, max_, std


def train_mlp_probes(emb, scores_raw, Y, min_pos=5, pca_dim=64, alpha_blend=0.4):
    """
    CHANGE 1: Upgraded MLP probe.
    - pca_dim: 32 → 64  (more embedding information)
    - hidden:  (32,) → (128, 64)  (more capacity)
    - max_iter: 100 → 300  (longer training)
    - min_pos: 8 → 5  (catches more rare species)
    """
    scaler = StandardScaler()
    emb_s  = scaler.fit_transform(emb)
    pca    = PCA(n_components=min(pca_dim, emb_s.shape[1] - 1))
    Z      = pca.fit_transform(emb_s).astype(np.float32)
    print(f"Embedding: {emb.shape} → PCA: {Z.shape}  "
          f"(variance retained: {pca.explained_variance_ratio_.sum():.2%})")

    class_weights = build_class_freq_weights(Y, cap=10.0)

    probe_models = {}
    active = np.where(Y.sum(axis=0) >= min_pos)[0]
    print(f"Training MLP probes for {len(active)} species (>= {min_pos} pos windows)...")

    MAX_ROWS = 3000   

    for ci in tqdm(active, desc="MLP probes"):
        y = Y[:, ci]
        if y.sum() == 0 or y.sum() == len(y):
            continue

        prev, next_, mean, max_, std = build_sequential_features(scores_raw[:, ci])
        X = np.hstack([
            Z,
            scores_raw[:, ci:ci+1],
            prev[:, None], next_[:, None],
            mean[:, None], max_[:, None], std[:, None],
        ])

        n_pos = int(y.sum()); n_neg = len(y) - n_pos
        pos_idx = np.where(y == 1)[0]

        w      = float(class_weights[ci])
        repeat = max(1, int(round(w * n_neg / max(n_pos, 1))))
        repeat = min(repeat, 8)
        if n_pos * repeat + len(y) > MAX_ROWS:
            repeat = max(1, (MAX_ROWS - len(y)) // max(n_pos, 1))

        X_bal = np.vstack([X, np.tile(X[pos_idx], (repeat, 1))])
        y_bal = np.concatenate([y, np.ones(n_pos * repeat, dtype=y.dtype)])

        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),   
            activation="relu",
            max_iter=300,                   
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,            
            random_state=42,
            learning_rate_init=5e-4,        
            alpha=0.005,                    
        )
        clf.fit(X_bal, y_bal)
        probe_models[ci] = clf

    print(f"Trained {len(probe_models)} MLP probes")
    return probe_models, scaler, pca, alpha_blend


def apply_mlp_probes(emb_test, scores_test, probe_models, scaler, pca, alpha_blend=0.4):
    emb_s  = scaler.transform(emb_test)
    Z_test = pca.transform(emb_s).astype(np.float32)
    result = scores_test.copy()
    for ci, clf in probe_models.items():
        prev, next_, mean, max_, std = build_sequential_features(scores_test[:, ci])
        X_test = np.hstack([
            Z_test, scores_test[:, ci:ci+1],
            prev[:, None], next_[:, None],
            mean[:, None], max_[:, None], std[:, None],
        ])
        prob  = clf.predict_proba(X_test)[:, 1].astype(np.float32)
        logit = np.log(prob + 1e-7) - np.log(1 - prob + 1e-7)
        result[:, ci] = (1 - alpha_blend) * scores_test[:, ci] + alpha_blend * logit
    return result

print("Upgraded MLP probe (pca_dim=64, hidden=(128,64), max_iter=300, min_pos=5)")

# %% markdown
# ### 🔎 Vectorized MLP probe inference
# 
# | Item | Details |
# |---|---|
# | **Purpose** | Converts many per-class MLPs into batched PyTorch matrix operations for faster inference. |
# | **Input** | Trained sklearn MLP probe models, scaler, PCA, and test scores. |
# | **Output** | `VectorizedMLPProbes` and `apply_mlp_probes_vectorized()`. |
# | **Risk / check** | This assumes all trained MLPs share the same architecture. |

# %% cell 40
import torch
import torch.nn as nn

class VectorizedMLPProbes(nn.Module):
    """Stacks all per-class MLP weights into a single batched PyTorch model.
    Replaces the slow Python for-loop over probe_models at inference time."""
    def __init__(self, probe_models):
        super().__init__()
        self.valid_classes = sorted(probe_models.keys())
        V = len(self.valid_classes)
        if V == 0:
            self.weights = nn.ParameterList()
            self.biases  = nn.ParameterList()
            self.n_layers = 0
            return

        sample = probe_models[self.valid_classes[0]]
        self.n_layers = len(sample.coefs_)
        self.weights  = nn.ParameterList()
        self.biases   = nn.ParameterList()

        for layer_idx in range(self.n_layers):
            W = np.stack([probe_models[c].coefs_[layer_idx]
                          for c in self.valid_classes], axis=0)       
            b = np.stack([probe_models[c].intercepts_[layer_idx]
                          for c in self.valid_classes], axis=0)       
            self.weights.append(nn.Parameter(
                torch.tensor(W, dtype=torch.float32), requires_grad=False))
            self.biases.append(nn.Parameter(
                torch.tensor(b, dtype=torch.float32), requires_grad=False))

    def forward(self, x):
        h = x
        for i in range(self.n_layers):
            h = torch.bmm(h, self.weights[i]) + self.biases[i].unsqueeze(1)
            if i < self.n_layers - 1:
                h = torch.relu(h)
        return h.squeeze(-1)   

def apply_mlp_probes_vectorized(emb_test, scores_test, probe_models,
                                 scaler, pca, alpha_blend=0.4):
    """
    Drop-in replacement for apply_mlp_probes().
    Uses batched PyTorch matrix multiply instead of a Python for-loop —
    ~10-50x faster at inference time.
    """
    if len(probe_models) == 0:
        return scores_test.copy()

    emb_s  = scaler.transform(emb_test)
    Z_test = pca.transform(emb_s).astype(np.float32)

    valid_classes = sorted(probe_models.keys())
    V = len(valid_classes)
    N = len(scores_test)

    raw  = scores_test[:, valid_classes].T          
    n_files = N // N_WINDOWS
    raw_view = raw.reshape(V, n_files, N_WINDOWS)
    prev = np.concatenate([raw_view[:, :, :1], raw_view[:, :, :-1]], axis=2).reshape(V, N)
    nxt  = np.concatenate([raw_view[:, :, 1:], raw_view[:, :, -1:]], axis=2).reshape(V, N)
    mean = np.repeat(raw_view.mean(axis=2), N_WINDOWS, axis=1)
    mx   = np.repeat(raw_view.max(axis=2),  N_WINDOWS, axis=1)
    std  = np.repeat(raw_view.std(axis=2),  N_WINDOWS, axis=1)

    scalar_feats = np.stack([raw, prev, nxt, mean, mx, std], axis=-1).astype(np.float32)
    Z_expanded = np.broadcast_to(Z_test, (V, N, Z_test.shape[1]))
    X_all = np.concatenate(
        [Z_expanded.astype(np.float32), scalar_feats], axis=-1)

    vec_probe = VectorizedMLPProbes(probe_models)
    vec_probe.eval()
    with torch.no_grad():
        preds = vec_probe(torch.tensor(X_all)).numpy()   

    result = scores_test.copy()
    base_valid = scores_test[:, valid_classes]           
    result[:, valid_classes] = (
        (1.0 - alpha_blend) * base_valid +
        alpha_blend * preds.T
    )
    return result

print("Vectorized MLP probe inference defined")

# %% markdown
# ### 🔎 Isotonic calibration and per-class thresholds
# 
# | Item | Details |
# |---|---|
# | **Purpose** | Calibrates file-level probabilities and estimates per-class thresholds for sharpening. |
# | **Input** | OOF-like probabilities, labels, and threshold grid. |
# | **Output** | `calibrate_and_optimize_thresholds()` and `apply_per_class_thresholds()`. |
# | **Risk / check** | Calibration quality depends on whether the input probabilities are truly out-of-fold-like. |

# %% cell 42
from sklearn.isotonic import IsotonicRegression

def calibrate_and_optimize_thresholds(oof_probs, Y_FULL, 
                                       threshold_grid=None, n_windows=12):
    """
    For each species:
    1. Fit isotonic regression on OOF scores (calibrates overconfident/underconfident classes)
    2. Grid-search F1-optimal threshold over calibrated probs
    Returns: thresholds array of shape (n_classes,)
    """
    if threshold_grid is None:
        threshold_grid = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    
    n_samples, n_cls = oof_probs.shape
    thresholds = np.full(n_cls, 0.5, dtype=np.float32)
    n_files    = n_samples // n_windows
    file_oof   = oof_probs.reshape(n_files, n_windows, n_cls).max(axis=1)
    file_y     = Y_FULL.reshape(n_files, n_windows, n_cls).max(axis=1)
    
    n_calibrated = 0
    for c in range(n_cls):
        y_true = file_y[:, c]
        y_prob = file_oof[:, c]
        if y_true.sum() < 3:
            continue
        try:
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(y_prob, y_true)
            y_cal = ir.transform(y_prob)
        except Exception:
            y_cal = y_prob
        
        best_f1, best_t = 0.0, 0.5
        for t in threshold_grid:
            pred = (y_cal >= t).astype(int)
            tp = ((pred==1) & (y_true==1)).sum()
            fp = ((pred==1) & (y_true==0)).sum()
            fn = ((pred==0) & (y_true==1)).sum()
            prec = tp / (tp + fp + 1e-8)
            rec  = tp / (tp + fn + 1e-8)
            f1   = 2 * prec * rec / (prec + rec + 1e-8)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = best_t
        n_calibrated += 1
    
    print(f"Calibrated {n_calibrated} classes")
    print(f"Mean threshold: {thresholds.mean():.3f}")
    print(f"Range: [{thresholds.min():.2f}, {thresholds.max():.2f}]")
    return thresholds

def apply_per_class_thresholds(scores, thresholds):
    """
    Sharpens probabilities around the per-class threshold:
    - above threshold → push toward 1
    - below threshold → push toward 0
    """
    C = scores.shape[1]
    assert C == len(thresholds)
    scaled = np.copy(scores)
    for c in range(C):
        t = thresholds[c]
        above = scores[:, c] > t
        scaled[ above, c] = 0.5 + 0.5 * (scores[ above, c] - t) / (1 - t + 1e-8)
        scaled[~above, c] = 0.5 * scores[~above, c] / (t + 1e-8)
    return np.clip(scaled, 0.0, 1.0)

print("Isotonic calibration + per-class threshold optimization defined")

# %% markdown
# ### 🔎 Rank-aware scaling
# 
# | Item | Details |
# |---|---|
# | **Purpose** | Uses file-level peak confidence to scale all windows for each species. |
# | **Input** | Window-level probabilities. |
# | **Output** | `rank_aware_scaling()`. |
# | **Risk / check** | Strong scaling can reduce low-confidence but valid detections. |

# %% cell 44
def rank_aware_scaling(probs, n_windows=12, power=0.4):
    """
    Scale each window by the file's single peak confidence.

    How it works:
      1. For each file, find the MAX score across all 12 windows (per species)
      2. Raise it to power → scale factor
      3. Multiply every window's score by that scale factor

    Example for one species across 12 windows:
      Confident file:  max=0.90 → scale=0.90^0.4=0.96 → mild boost
      Uncertain file:  max=0.10 → scale=0.10^0.4=0.40 → strong suppression

    power=0.0 → no effect (baseline)
    power=0.4 → moderate suppression of uncertain files (recommended start)
    power=1.0 → multiply directly by file max (very aggressive)
    """
    N, C = probs.shape
    assert N % n_windows == 0, f"Expected multiple of {n_windows}, got {N}"

    view     = probs.reshape(-1, n_windows, C)              
    file_max = view.max(axis=1, keepdims=True)              

    scale  = np.power(file_max, power)                      
    scaled = view * scale                                   

    return scaled.reshape(N, C)

print("Rank-aware scaling defined")

# %% markdown
# ### 🔎 Adaptive delta smoothing
# 
# | Item | Details |
# |---|---|
# | **Purpose** | Smooths uncertain windows more than confident peaks, preserving sharp biological calls. |
# | **Input** | Window-level probabilities. |
# | **Output** | `adaptive_delta_smooth()`. |
# | **Risk / check** | Check that the number of rows remains divisible by `N_WINDOWS`. |

# %% cell 46
def adaptive_delta_smooth(probs, n_windows=12, base_alpha=0.20):
    """
    Smooth uncertain windows toward their neighbors,
    while leaving confident windows almost untouched.

    How it works:
      For each window t:
        conf  = max probability across all 234 species at window t
        alpha = base_alpha * (1 - conf)   ← KEY: adapts to confidence
        new[t] = (1 - alpha) * old[t] + alpha * avg(old[t-1], old[t+1])

    Why alpha adapts to confidence:
      Confident window (max=0.90):
        alpha = 0.20 * (1 - 0.90) = 0.02  → barely smoothed, peak preserved
      Uncertain window (max=0.10):
        alpha = 0.20 * (1 - 0.10) = 0.18  → smoothed more, noise reduced

    base_alpha=0.0  → no smoothing (baseline)
    base_alpha=0.20 → recommended starting point
    """
    N, C = probs.shape
    assert N % n_windows == 0, f"Expected multiple of {n_windows}, got {N}"

    result = probs.copy()
    view   = probs.reshape(-1, n_windows, C)    
    out    = result.reshape(-1, n_windows, C)  

    for t in range(n_windows):
        conf = view[:, t, :].max(axis=-1, keepdims=True)   
        alpha = base_alpha * (1.0 - conf)                  
        if t == 0:
            neighbor_avg = (view[:, t, :] + view[:, t+1, :]) / 2.0
        elif t == n_windows - 1:
            neighbor_avg = (view[:, t-1, :] + view[:, t, :]) / 2.0
        else:
            neighbor_avg = (view[:, t-1, :] + view[:, t+1, :]) / 2.0
        out[:, t, :] = (1.0 - alpha) * view[:, t, :] + alpha * neighbor_avg
    return result

print("Adaptive delta smoothing defined")

# %% markdown
# ## Sequence Modeling: LightProtoSSM with Cross-Attention
# 
# This section defines the core sequence learners responsible for processing the Perch embeddings. The primary model, LightProtoSSM, utilizes a selective state space architecture enhanced with multi-head cross-attention. This mechanism allows the model to actively attend to different 5-second windows across the entire soundscape, effectively separating isolated noise spikes from persistent biological calls. 
# 
# Additionally, a ResidualSSM is instantiated to learn an additive correction layer. It targets the systematic errors remaining after the first-pass ensemble, acting as a final polish on the temporal probabilities. Test-Time Augmentation (TTA) is also defined here, applying circular shifts to expose different context patterns to the sequence models.
# 
# ---

# %% markdown
# ### 🔎 Sequence models: ProtoSSM, TTA, and ResidualSSM
# 
# | Item | Details |
# |---|---|
# | **Purpose** | Defines the main temporal sequence learner and the residual correction model. |
# | **Input** | Perch embeddings, Perch logits, site IDs, hour IDs, and labels. |
# | **Output** | `LightProtoSSM`, `train_light_proto_ssm()`, `run_tta_proto()`, `ResidualSSM`, `train_residual_ssm()`. |
# | **Risk / check** | This is the most complex part; shape consistency is critical: files × 12 windows × features/classes. |

# %% cell 49
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(d_model, d_model, d_conv, padding=d_conv - 1, groups=d_model)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B_sz, T, D = x.shape
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)
        dt = F.softplus(self.dt_proj(x_conv))
        A = -torch.exp(self.A_log)
        B = self.B_proj(x_conv)
        C = self.C_proj(x_conv)
        h = torch.zeros(B_sz, D, self.d_state, device=x.device)
        ys = []
        for t in range(T):
            dA = torch.exp(A[None] * dt[:, t, :, None])
            dB = dt[:, t, :, None] * B[:, t, None, :]
            h = h * dA + x[:, t, :, None] * dB
            ys.append((h * C[:, t, None, :]).sum(-1))
        y = torch.stack(ys, dim=1)
        return y + x * self.D[None, None, :]

class LightProtoSSM(nn.Module):
    def __init__(self, d_input=1536, d_model=128, d_state=16, n_classes=234, n_windows=12, dropout=0.15, n_sites=20, meta_dim=16, use_cross_attn=True, cross_attn_heads=2):
        super().__init__()
        self.n_classes = n_classes
        self.n_windows = n_windows
        self.use_cross_attn = use_cross_attn
        self.input_proj = nn.Sequential(nn.Linear(d_input, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        self.pos_enc  = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)
        self.ssm_fwd  = nn.ModuleList([SelectiveSSM(d_model, d_state) for _ in range(2)])
        self.ssm_bwd  = nn.ModuleList([SelectiveSSM(d_model, d_state) for _ in range(2)])
        self.ssm_merge= nn.ModuleList([nn.Linear(2 * d_model, d_model) for _ in range(2)])
        self.ssm_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.drop     = nn.Dropout(dropout)
        
        if use_cross_attn:
            self.cross_attn = nn.ModuleList([nn.MultiheadAttention(d_model, num_heads=cross_attn_heads, dropout=dropout, batch_first=True) for _ in range(2)])
            self.cross_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
            
        self.prototypes   = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        self.proto_temp   = nn.Parameter(torch.tensor(5.0))
        self.class_bias   = nn.Parameter(torch.zeros(n_classes))
        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))

    def init_prototypes(self, emb_tensor, labels_tensor):
        with torch.no_grad():
            h = self.input_proj(emb_tensor)
            for c in range(self.n_classes):
                mask = labels_tensor[:, c] > 0.5
                if mask.sum() > 0:
                    self.prototypes.data[c] = F.normalize(h[mask].mean(0), dim=0)

    def forward(self, emb, perch_logits=None, site_ids=None, hours=None):
        B, T, _ = emb.shape
        h = self.input_proj(emb) + self.pos_enc[:, :T, :]
        if site_ids is not None and hours is not None:
            meta = self.meta_proj(torch.cat([self.site_emb(site_ids), self.hour_emb(hours)], dim=-1))
            h = h + meta[:, None, :]
            
        for i, (fwd, bwd, merge, norm) in enumerate(zip(self.ssm_fwd, self.ssm_bwd, self.ssm_merge, self.ssm_norm)):
            res = h
            h_f = fwd(h)
            h_b = bwd(h.flip(1)).flip(1)
            h   = self.drop(merge(torch.cat([h_f, h_b], dim=-1)))
            h   = norm(h + res)
            if self.use_cross_attn:
                attn_out, _ = self.cross_attn[i](h, h, h)
                h = self.cross_norm[i](h + attn_out)
                
        h_n = F.normalize(h, dim=-1)
        p_n = F.normalize(self.prototypes, dim=-1)
        sim = (torch.matmul(h_n, p_n.T) * F.softplus(self.proto_temp) + self.class_bias[None, None, :])
        
        if perch_logits is not None:
            alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]
            out   = alpha * sim + (1 - alpha) * perch_logits
        else:
            out = sim
        return out

def train_light_proto_ssm(emb_full, scores_full, Y_full, meta_full, n_epochs=40, patience=8, lr=1e-3, n_sites=20, verbose=False):
    n_files = len(emb_full) // N_WINDOWS
    emb_f   = emb_full.reshape(n_files, N_WINDOWS, -1)
    log_f   = scores_full.reshape(n_files, N_WINDOWS, -1)
    lab_f   = Y_full.reshape(n_files, N_WINDOWS, -1).astype(np.float32)
    
    fnames  = meta_full["filename"].unique()
    sites_u = sorted(meta_full["site"].unique())
    site2i  = {s: i + 1 for i, s in enumerate(sites_u)}
    site_ids = np.array([min(site2i.get(meta_full.loc[meta_full["filename"]==fn,"site"].iloc[0], 0), n_sites-1) for fn in fnames], dtype=np.int64)
    hour_ids = np.array([int(meta_full.loc[meta_full["filename"]==fn,"hour_utc"].iloc[0]) % 24 for fn in fnames], dtype=np.int64)
    
    model = LightProtoSSM(n_classes=N_CLASSES, n_sites=n_sites, use_cross_attn=True, cross_attn_heads=2)
    model.init_prototypes(torch.tensor(emb_full, dtype=torch.float32), torch.tensor(Y_full, dtype=torch.float32))
    
    emb_t  = torch.tensor(emb_f, dtype=torch.float32)
    log_t  = torch.tensor(log_f, dtype=torch.float32)
    lab_t  = torch.tensor(lab_f, dtype=torch.float32)
    site_t = torch.tensor(site_ids, dtype=torch.long)
    hour_t = torch.tensor(hour_ids, dtype=torch.long)
    
    pos_cnt    = lab_t.sum(dim=(0, 1))
    total      = lab_t.shape[0] * lab_t.shape[1]
    pos_weight = ((total - pos_cnt) / (pos_cnt + 1)).clamp(max=25.0)
    
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=n_epochs, steps_per_epoch=1, pct_start=0.1, anneal_strategy="cos")
    
    best_loss, best_state, wait = float("inf"), None, 0
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_start = int(n_epochs * 0.65)
    swa_sched = torch.optim.swa_utils.SWALR(opt, swa_lr=4e-4)
    
    for ep in range(n_epochs):
        model.train()
        out  = model(emb_t, log_t, site_ids=site_t, hours=hour_t)
        loss = F.binary_cross_entropy_with_logits(out, lab_t, pos_weight=pos_weight[None, None, :]) + 0.15 * F.mse_loss(out, log_t)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        if ep >= swa_start:
            swa_model.update_parameters(model)
            swa_sched.step()
        else:
            sched.step()
            
        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            
        if wait >= patience:
            break
            
    if ep >= swa_start:
        torch.optim.swa_utils.update_bn(emb_t.unsqueeze(0), swa_model)
        model = swa_model
    else:
        model.load_state_dict(best_state)
        
    model.eval()
    return model, site2i

def run_tta_proto(proto_model, emb_files, sc_files, site_t, hour_t, shifts=[0, 1, -1, 2, -2]):
    proto_model.eval()
    all_preds = []
    emb_t  = torch.tensor(emb_files, dtype=torch.float32)
    sc_t   = torch.tensor(sc_files,  dtype=torch.float32)
    
    for shift in shifts:
        if shift == 0:
            e_shifted = emb_t
            s_shifted = sc_t
        else:
            e_shifted = torch.roll(emb_t, shift, dims=1)
            s_shifted = torch.roll(sc_t,  shift, dims=1)
            
        with torch.no_grad():
            out = proto_model(e_shifted, s_shifted, site_ids=site_t, hours=hour_t).numpy()
            
        if shift != 0:
            out = np.roll(out, -shift, axis=1)
            
        all_preds.append(out)
        
    return np.mean(all_preds, axis=0)

class ResidualSSM(nn.Module):
    def __init__(self, d_input=1536, d_scores=234, d_model=64, d_state=8, n_classes=234, n_windows=12, dropout=0.1, n_sites=20, meta_dim=8):
        super().__init__()
        self.n_classes = n_classes
        self.input_proj = nn.Sequential(nn.Linear(d_input + d_scores, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        self.site_emb  = nn.Embedding(n_sites, meta_dim)
        self.hour_emb  = nn.Embedding(24,      meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)
        self.pos_enc   = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        self.ssm_fwd   = SelectiveSSM(d_model, d_state)
        self.ssm_bwd   = SelectiveSSM(d_model, d_state)
        self.ssm_merge = nn.Linear(2 * d_model, d_model)
        self.ssm_norm  = nn.LayerNorm(d_model)
        self.ssm_drop  = nn.Dropout(dropout)
        self.output_head = nn.Linear(d_model, n_classes)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(self, emb, first_pass, site_ids=None, hours=None):
        B, T, _ = emb.shape
        x = torch.cat([emb, first_pass], dim=-1)
        h = self.input_proj(x) + self.pos_enc[:, :T, :]
        if site_ids is not None and hours is not None:
            meta = self.meta_proj(torch.cat([self.site_emb(site_ids.clamp(0, self.site_emb.num_embeddings-1)), self.hour_emb(hours.clamp(0, 23))], dim=-1))
            h = h + meta.unsqueeze(1)
            
        res = h
        h_f = self.ssm_fwd(h)
        h_b = self.ssm_bwd(h.flip(1)).flip(1)
        h   = self.ssm_drop(self.ssm_merge(torch.cat([h_f, h_b], dim=-1)))
        h   = self.ssm_norm(h + res)
        return self.output_head(h)

def train_residual_ssm(emb_full, first_pass_flat, Y_full, site_ids, hour_ids, n_epochs=30, patience=8, lr=1e-3, correction_weight=0.30, verbose=False):
    n_files    = len(emb_full) // N_WINDOWS
    emb_f      = emb_full.reshape(n_files, N_WINDOWS, -1)
    fp_f       = first_pass_flat.reshape(n_files, N_WINDOWS, -1)
    lab_f      = Y_full.reshape(n_files, N_WINDOWS, -1).astype(np.float32)
    fp_prob    = 1.0 / (1.0 + np.exp(-np.clip(fp_f, -30, 30)))
    residuals  = lab_f - fp_prob
    n_val    = max(1, int(n_files * 0.15))
    rng      = torch.Generator()
    rng.manual_seed(42)
    perm     = torch.randperm(n_files, generator=rng).numpy()
    val_i    = perm[:n_val]
    train_i  = perm[n_val:]
    
    emb_t    = torch.tensor(emb_f, dtype=torch.float32)
    fp_t     = torch.tensor(fp_f, dtype=torch.float32)
    res_t    = torch.tensor(residuals, dtype=torch.float32)
    site_t   = torch.tensor(site_ids, dtype=torch.long)
    hour_t   = torch.tensor(hour_ids, dtype=torch.long)
    model    = ResidualSSM(n_classes=N_CLASSES)
    opt      = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched    = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=n_epochs, steps_per_epoch=1, pct_start=0.1, anneal_strategy="cos")
    
    best_loss, best_state, wait = float("inf"), None, 0
    for ep in range(n_epochs):
        model.train()
        corr = model(emb_t[train_i], fp_t[train_i], site_ids=site_t[train_i], hours=hour_t[train_i])
        loss = F.mse_loss(corr, res_t[train_i])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        
        model.eval()
        with torch.no_grad():
            val_corr = model(emb_t[val_i], fp_t[val_i], site_ids=site_t[val_i], hours=hour_t[val_i])
            val_loss = F.mse_loss(val_corr, res_t[val_i])
            
        if val_loss.item() < best_loss:
            best_loss  = val_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            
        if wait >= patience:
            break
            
    model.load_state_dict(best_state)
    return model, correction_weight

print("Sequence Models Initialized")

# %% markdown
# ### 🔎 Optional honest OOF baseline evaluation
# 
# | Item | Details |
# |---|---|
# | **Purpose** | Computes raw Perch OOF AUC when `MODE == train`. |
# | **Input** | `sc_tr`, `Y_FULL_aligned`, `meta_tr`, and GroupKFold split count. |
# | **Output** | Printed baseline AUC and `oof_raw`. |
# | **Risk / check** | Skipped in submit mode to save runtime. |

# %% cell 51
baseline_auc = None
oof_raw      = None
 
if CFG["run_oof"]:
    print("Running honest OOF evaluation on training data…")
    baseline_auc, oof_raw = honest_oof_auc(
        sc_tr, Y_FULL_aligned, meta_tr,
        n_splits=CFG["oof_n_splits"],
        label="raw Perch"
    )
    print(f"\nBaseline OOF AUC: {baseline_auc:.6f}  ← your starting point")
else:
    print("Submit mode: skipping OOF evaluation")

# %% markdown
# ### 🔎 Embedding-based MLP probe training
# 
# | Item | Details |
# |---|---|
# | **Purpose** | Trains small per-class MLP probes using compressed Perch embeddings plus temporal score features. |
# | **Input** | Perch embeddings, raw scores, labels, PCA/scaler settings. |
# | **Output** | `train_mlp_probes()` and `apply_mlp_probes()`. |
# | **Risk / check** | Classes with too few positive examples are skipped to avoid unstable probes. |

# %% cell 53
def run_pipeline_oof(emb_full, sc_full, Y_full, meta_full, n_splits=5):
    """
    Proper full-pipeline OOF.
    Trains ProtoSSM + MLP on K-1 folds, predicts on held-out fold.
    """
    file_meta = (
        meta_full.drop_duplicates("filename")
        .reset_index(drop=True)
    )

    gkf = GroupKFold(n_splits=n_splits)
    oof_probs = np.zeros((len(sc_full), N_CLASSES), dtype=np.float32)

    for fold, (tr_f, va_f) in enumerate(
        gkf.split(file_meta, groups=file_meta["filename"]), 1
    ):
        tr_fnames = set(file_meta.iloc[tr_f]["filename"])
        va_fnames = set(file_meta.iloc[va_f]["filename"])

        tr_mask = meta_full["filename"].isin(tr_fnames).values
        va_mask = meta_full["filename"].isin(va_fnames).values

        emb_tr_f = emb_full[tr_mask]
        sc_tr_f = sc_full[tr_mask]
        Y_tr_f = Y_full[tr_mask]
        meta_tr_f = meta_full[tr_mask].reset_index(drop=True)

        emb_va_f = emb_full[va_mask]
        sc_va_f = sc_full[va_mask]
        meta_va_f = meta_full[va_mask].reset_index(drop=True)

        proto_model, site2i = train_light_proto_ssm(
            emb_tr_f,
            sc_tr_f,
            Y_tr_f,
            meta_tr_f,
            n_epochs=40,
            patience=8,
            lr=1e-3,
            verbose=False,
        )

        n_va = len(emb_va_f) // N_WINDOWS

        va_fn_list = (
            meta_va_f.drop_duplicates("filename")["filename"].tolist()
        )

        va_site_ids = np.array(
            [
                min(
                    site2i.get(
                        meta_va_f.loc[
                            meta_va_f["filename"] == fn, "site"
                        ].iloc[0],
                        0,
                    ),
                    19,
                )
                for fn in va_fn_list
            ],
            dtype=np.int64,
        )

        va_hour_ids = np.array(
            [
                int(
                    meta_va_f.loc[
                        meta_va_f["filename"] == fn, "hour_utc"
                    ].iloc[0]
                )
                % 24
                for fn in va_fn_list
            ],
            dtype=np.int64,
        )

        proto_model.eval()
        with torch.no_grad():
            proto_va = proto_model(
                torch.tensor(
                    emb_va_f.reshape(n_va, N_WINDOWS, -1),
                    dtype=torch.float32,
                ),
                torch.tensor(
                    sc_va_f.reshape(n_va, N_WINDOWS, -1),
                    dtype=torch.float32,
                ),
                site_ids=torch.tensor(va_site_ids, dtype=torch.long),
                hours=torch.tensor(va_hour_ids, dtype=torch.long),
            ).numpy().reshape(-1, N_CLASSES)

        probe_models, emb_scaler, emb_pca, alpha_blend = train_mlp_probes(
            emb_tr_f,
            sc_tr_f,
            Y_tr_f,
            min_pos=5,
            pca_dim=64,
            alpha_blend=0.4,
        )

        sc_va_mlp = apply_mlp_probes_vectorized(
            emb_va_f,
            sc_va_f,
            probe_models,
            emb_scaler,
            emb_pca,
            alpha_blend,
        )

        first_pass = 0.5 * proto_va + 0.5 * sc_va_mlp
        probs_va = 1.0 / (1.0 + np.exp(-np.clip(first_pass, -30, 30)))
        oof_probs[va_mask] = probs_va

        fold_auc = macro_auc(Y_full[va_mask], probs_va)
        print(
            f"  Fold {fold}/{n_splits}  val files={len(va_fnames)}  AUC={fold_auc:.6f}"
        )

    overall = macro_auc(Y_full, oof_probs)
    print(f"\nFull pipeline OOF AUC: {overall:.6f}")
    return overall, oof_probs


if CFG["run_oof"]:
    pipeline_auc, oof_pipeline = run_pipeline_oof(
        emb_tr,
        sc_tr,
        Y_FULL_aligned,
        meta_tr,
        n_splits=5,
    )

# %% markdown
# ## Test or Dry-Run Inference
# 
# In formal Kaggle reruns, `test_soundscapes` contains the hidden test audio. In normal public notebook runs, that folder is empty, so the notebook falls back to train soundscapes for shape and runtime verification.
# 
# ---

# %% markdown
# ### 🔎 Test inference or public dry-run file selection
# 
# | Item | Details |
# |---|---|
# | **Purpose** | Finds hidden test audio during Kaggle rerun, or falls back to train soundscapes for public dry-run validation. |
# | **Input** | `test_soundscapes` or first train soundscape files. |
# | **Output** | `meta_te`, `sc_te`, `emb_te`. |
# | **Risk / check** | Dry-run output is only for shape/runtime checking, not LB validation. |

# %% cell 56
test_paths = sorted((BASE / "test_soundscapes").glob("*.ogg"))
IS_DRY_RUN = len(test_paths) == 0
 
if IS_DRY_RUN:
    n = CFG["dryrun_n_files"] or 20
    print(f"No hidden test — dry-run on {n} train files")
    test_paths = sorted((BASE / "train_soundscapes").glob("*.ogg"))[:n]
else:
    print(f"Hidden test files: {len(test_paths)}")
 
meta_te, sc_te, emb_te = run_perch(test_paths, CFG["batch_files"], verbose=CFG["verbose"])
print(f"Test scores: {sc_te.shape}")

# %% markdown
# ## ProtoSSM Execution & Isotonic Sharpening
# 
# This cell executes the full sequence modeling branch. It combines the LightProtoSSM, the MLP probes, the joint priors, and the ResidualSSM. 
# 
# An important architectural note: Test-Time Augmentation (TTA) is specifically applied to the training data to generate the inputs for the `ResidualSSM`, but is intentionally omitted from the final test inference. This asymmetry shifts the calibration landscape in a way that optimizes the final probability distributions. Following adaptive delta smoothing, the pipeline applies the optimized isotonic thresholds to sharpen the final output.
# 
# ---

# %% markdown
# ### 🔎 Embedding-based MLP probe training
# 
# | Item | Details |
# |---|---|
# | **Purpose** | Trains small per-class MLP probes using compressed Perch embeddings plus temporal score features. |
# | **Input** | Perch embeddings, raw scores, labels, PCA/scaler settings. |
# | **Output** | `train_mlp_probes()` and `apply_mlp_probes()`. |
# | **Risk / check** | Classes with too few positive examples are skipped to avoid unstable probes. |

# %% cell 59
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

t0 = time.time()
proto_model, site2i_tr = train_light_proto_ssm(
    emb_tr, sc_tr, Y_FULL_aligned, meta_tr,
    n_epochs=40, patience=8, lr=1e-3, verbose=False)
print(f"ProtoSSM training: {time.time()-t0:.1f}s")

n_test_files  = len(sc_te) // N_WINDOWS
emb_te_f      = emb_te.reshape(n_test_files, N_WINDOWS, -1)
sc_te_f       = sc_te.reshape(n_test_files, N_WINDOWS, -1)

test_fnames   = meta_te.drop_duplicates("filename")["filename"].tolist()
n_sites_cap   = 20
test_site_ids = np.array([
    min(site2i_tr.get(
        meta_te.loc[meta_te["filename"]==fn,"site"].iloc[0], 0),
        n_sites_cap-1)
    for fn in test_fnames], dtype=np.int64)
test_hour_ids = np.array([
    int(meta_te.loc[meta_te["filename"]==fn,"hour_utc"].iloc[0]) % 24
    for fn in test_fnames], dtype=np.int64)

proto_model.eval()
with torch.no_grad():
    proto_out = proto_model(
        torch.tensor(emb_te_f, dtype=torch.float32),
        torch.tensor(sc_te_f,  dtype=torch.float32),
        site_ids=torch.tensor(test_site_ids, dtype=torch.long),
        hours   =torch.tensor(test_hour_ids, dtype=torch.long),
    ).numpy()
proto_scores_flat = proto_out.reshape(-1, N_CLASSES).astype(np.float32)

prior_tables   = build_prior_tables(sc, Y_SC)
sc_te_adjusted = apply_prior(
    sc_te,
    sites=meta_te["site"].to_numpy(),
    hours=meta_te["hour_utc"].to_numpy(),
    tables=prior_tables,
    lambda_prior=0.4,
)

probe_models, emb_scaler, emb_pca, alpha_blend = train_mlp_probes(
    emb=emb_tr, scores_raw=sc_tr, Y=Y_FULL_aligned,
    min_pos=5, pca_dim=64, alpha_blend=0.4,
)
sc_te_adjusted = apply_mlp_probes_vectorized(
    emb_te, sc_te_adjusted,
    probe_models, emb_scaler, emb_pca, alpha_blend,
)

ENSEMBLE_W      = 0.5
first_pass_flat = (ENSEMBLE_W * proto_scores_flat
                   + (1.0 - ENSEMBLE_W) * sc_te_adjusted)

n_tr_files    = len(sc_tr) // N_WINDOWS
emb_tr_f      = emb_tr.reshape(n_tr_files, N_WINDOWS, -1)
sc_tr_f       = sc_tr.reshape(n_tr_files, N_WINDOWS, -1)

tr_fnames     = meta_tr.drop_duplicates("filename")["filename"].tolist()
tr_site_ids   = np.array([
    min(site2i_tr.get(
        meta_tr.loc[meta_tr["filename"]==fn,"site"].iloc[0], 0),
        n_sites_cap-1)
    for fn in tr_fnames], dtype=np.int64)
tr_hour_ids   = np.array([
    int(meta_tr.loc[meta_tr["filename"]==fn,"hour_utc"].iloc[0]) % 24
    for fn in tr_fnames], dtype=np.int64)

proto_tr_out = run_tta_proto(
    proto_model, emb_tr_f, sc_tr_f,
    site_t=torch.tensor(tr_site_ids, dtype=torch.long),
    hour_t=torch.tensor(tr_hour_ids, dtype=torch.long),
    shifts=[0, 1, -1, 2, -2],
)

proto_tr_flat = proto_tr_out.reshape(-1, N_CLASSES).astype(np.float32)

sc_tr_prior   = apply_prior(
    sc_tr,
    sites=meta_tr["site"].to_numpy(),
    hours=meta_tr["hour_utc"].to_numpy(),
    tables=prior_tables,
    lambda_prior=0.4,
)
sc_tr_mlp = apply_mlp_probes_vectorized(
    emb_tr, sc_tr_prior,
    probe_models, emb_scaler, emb_pca, alpha_blend,
)
first_pass_tr = (ENSEMBLE_W * proto_tr_flat
                 + (1.0 - ENSEMBLE_W) * sc_tr_mlp)

train_probs_for_calib = sigmoid(first_pass_tr)
PER_CLASS_THRESHOLDS = calibrate_and_optimize_thresholds(
    oof_probs=train_probs_for_calib,
    Y_FULL=Y_FULL_aligned,
    threshold_grid=[0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
    n_windows=N_WINDOWS,
)

t0 = time.time()
res_model, correction_weight = train_residual_ssm(
    emb_full=emb_tr,
    first_pass_flat=first_pass_tr,
    Y_full=Y_FULL_aligned,
    site_ids=tr_site_ids,
    hour_ids=tr_hour_ids,
    n_epochs=30,
    patience=8,
    lr=1e-3,
    correction_weight=0.30,
    verbose=False,
)
print(f"ResidualSSM training: {time.time()-t0:.1f}s")

first_pass_te_f  = first_pass_flat.reshape(n_test_files, N_WINDOWS, -1)
res_model.eval()
with torch.no_grad():
    test_correction = res_model(
        torch.tensor(emb_te_f,         dtype=torch.float32),
        torch.tensor(first_pass_te_f,  dtype=torch.float32),
        site_ids=torch.tensor(test_site_ids, dtype=torch.long),
        hours   =torch.tensor(test_hour_ids, dtype=torch.long),
    ).numpy()

correction_flat = test_correction.reshape(-1, N_CLASSES).astype(np.float32)
final_scores    = (first_pass_flat
                   + correction_weight * correction_flat)

final_scores = final_scores / temperatures[None, :]
probs = sigmoid(final_scores)

probs = file_confidence_scale(probs, n_windows=N_WINDOWS,
                               top_k=2,       power=0.4)
probs = rank_aware_scaling(   probs, n_windows=N_WINDOWS,
                               power=0.4)
probs = adaptive_delta_smooth(probs, n_windows=N_WINDOWS,
                               base_alpha=0.20)
probs = np.clip(probs, 0.0, 1.0)

probs = apply_per_class_thresholds(probs, PER_CLASS_THRESHOLDS)

sub = pd.DataFrame(probs.astype(np.float32), columns=PRIMARY_LABELS)
sub.insert(0, "row_id", meta_te["row_id"].values)
sub.to_csv("submission_protossm.csv", index=False)                                                                                        

print("ProtoSSM execution complete")

# %% markdown
# ## Distilled SED Branch
# 
# The second branch evaluates mel-spectrogram visual features using Tucker Arrants' public distilled SED ONNX folds. Because the SED models lack temporal context across the 12 windows, this branch provides highly localized, complementary evidence to the broader Perch sequence model.
# 
# ---

# %% markdown
# ### 🔎 Distilled SED branch execution
# 
# | Item | Details |
# |---|---|
# | **Purpose** | Converts audio chunks to mel spectrograms and runs public distilled SED ONNX folds. |
# | **Input** | Test/dry-run audio files, SED ONNX fold files, and audio preprocessing constants. |
# | **Output** | `submission_sed.csv` containing the SED-branch predictions. |
# | **Risk / check** | Requires SED ONNX files and ONNX Runtime. Spectrogram shape must match the model input. |

# %% cell 62
import librosa
from scipy.ndimage import gaussian_filter1d

N_MELS_SED = 256
N_FFT_SED  = 2048
HOP_SED    = 512
FMIN_SED   = 20
FMAX_SED   = 16000
TOP_DB_SED = 80

def find_sed_dir():
    hits = sorted(Path("/kaggle/input").rglob("sed_fold0.onnx"))
    if not hits:
        raise FileNotFoundError("sed_fold0.onnx not found.")
    return hits[0].parent

def make_sed_session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])

def audio_to_mel(chunks):
    mels = []
    for x in chunks:
        s = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=N_FFT_SED, hop_length=HOP_SED, n_mels=N_MELS_SED, fmin=FMIN_SED, fmax=FMAX_SED, power=2.0)
        s = librosa.power_to_db(s, top_db=TOP_DB_SED)
        s = (s - s.mean()) / (s.std() + 1e-6)
        mels.append(s)
    return np.stack(mels)[:, None].astype(np.float32)

def file_to_sed_chunks(path):
    y, sr0 = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr0 != SR:
        y = librosa.resample(y, orig_sr=sr0, target_sr=SR)
    n = 60 * SR
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    else:
        y = y[:n]
    chunks = y.reshape(N_WINDOWS, WINDOW_SAMPLES)
    ends   = np.arange(1, N_WINDOWS + 1) * WINDOW_SEC
    return chunks, ends

def sigmoid_sed(x):
    return (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))).astype(np.float32)

test_paths = sorted((BASE / "test_soundscapes").glob("*.ogg"))
IS_DRY_RUN = len(test_paths) == 0

if IS_DRY_RUN:
    dry_n = CFG["dryrun_n_files"] if "CFG" in locals() else 20
    test_paths = sorted((BASE / "train_soundscapes").glob("*.ogg"))[:(dry_n or 20)]

sed_dir = find_sed_dir()
sed_fold_paths = sorted(sed_dir.glob("sed_fold*.onnx"), key=lambda p: int(re.search(r"sed_fold(\d+)", p.name).group(1)))
sed_sessions = [make_sed_session(p) for p in sed_fold_paths]

sed_rows, sed_preds = [], []

for i, path in enumerate(test_paths, 1):
    chunks, ends = file_to_sed_chunks(path)
    mel = audio_to_mel(chunks)
    p_sum = np.zeros((len(chunks), N_CLASSES), dtype=np.float32)

    for sess in sed_sessions:
        outs = sess.run(None, {sess.get_inputs()[0].name: mel})
        clip_logits = outs[0]
        frame_max   = outs[1].max(axis=1)
        p_sum += 0.5 * sigmoid_sed(clip_logits) + 0.5 * sigmoid_sed(frame_max)

    p_mean = p_sum / len(sed_sessions)

    if len(p_mean) > 1:
        p_mean = gaussian_filter1d(p_mean, sigma=0.65, axis=0, mode="nearest").astype(np.float32)

    stem = path.stem
    sed_rows.extend([f"{stem}_{int(t)}" for t in ends])
    sed_preds.append(p_mean)

sed_preds_arr = np.concatenate(sed_preds, axis=0)
sed_sub = pd.DataFrame(np.clip(sed_preds_arr, 0.0, 1.0), columns=PRIMARY_LABELS)
sed_sub.insert(0, "row_id", sed_rows)
sed_sub.to_csv("submission_sed.csv", index=False)

print("Distilled SED Processing Complete.")

# %% markdown
# ## Rank Blend & Fat-Tail Continuity Gates
# 
# This final step computes a 2-way rank-based ensemble combining our sequence model (60%) and the primary distilled SED branch (40%). 
# 
# We apply the following post-processing pipeline to extract maximum signal:
# 1. **Noise Suppression:** Pure noise suppression if SED strongly disagrees with ProtoSSM.
# 2. **Temporal Continuity:** A t-distribution kernel provides a fat-tailed 35-second context to protect continuous calls.
# 3. **SED Spike Preservation:** Isolated SED spike preservation for brief, high-confidence visual anomalies.
# 4. **Sonotype Mirroring:** Max-pooling across visually identical species groups (e.g., specific insects/frogs) to unify their probabilities.
# 5. **Adaptive Thresholding:** Aggressive suppression of low-confidence noise specifically targeting rare classes (Amphibia, Mammalia, Reptilia).
# 
# ---

# %% markdown
# ### 🔎 Final rank blend and submission creation
# 
# | Item | Details |
# |---|---|
# | **Purpose** | Aligns ProtoSSM and SED outputs by row ID, rank-blends them, applies final gates, and writes `submission.csv`. |
# | **Input** | `submission_protossm.csv`, `submission_sed.csv`, taxonomy, and sample submission in dry-run mode. |
# | **Output** | Final `submission.csv`. |
# | **Risk / check** | Row alignment is essential. The final CSV must match Kaggle sample submission columns and row count. |


# %% markdown
# ### v591 HGNetV2-B0 distilled sidecar
#
# This sidecar ports the complete Qiuzi HGNetV2-B0 distillation training artifacts
# (4 folds, rank OOF AUC 0.96727) into the public946 ProtoSSM/SED anchor.  It is
# intentionally conservative: write a standalone `submission_hgnet.csv`, require
# real model weights, then use only a small rank-blend weight in the final output.

# %% cell 65a
import os
import gc
import math
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
import timm
from torch import nn
from torchvision.transforms import v2 as tvt_v2

HGNET_WEIGHT_DIRS = [p.parent for p in Path('/kaggle/input').rglob('best_model_fold0.pt')]
print('HGNet candidate weight dirs:', [str(p) for p in HGNET_WEIGHT_DIRS[:8]])
if not HGNET_WEIGHT_DIRS:
    raise FileNotFoundError('v591 HGNet weights not found: expected best_model_fold0.pt under /kaggle/input')
HGNET_WEIGHT_DIR = HGNET_WEIGHT_DIRS[0]
for fold in range(4):
    wp = HGNET_WEIGHT_DIR / f'best_model_fold{fold}.pt'
    if not wp.exists():
        raise FileNotFoundError(f'Missing HGNet fold weight: {wp}')

HGNET_SEGMENT_SEC = 5
HGNET_SR = 32000
HGNET_BATCH = 16
HGNET_N_CLASSES = 234
HGNET_OUT_CSV = 'submission_hgnet.csv'
# Kaggle's current CUDA/PyTorch image throws cudaErrorNoKernelImageForDevice
# for HGNetV2 ops on the available GPU.  Validate this sidecar CPU-only; if
# runtime is too high, the next path is OpenVINO/ONNX export rather than a slot.
HGNET_DEVICE = torch.device('cpu')
HGNET_PREPROCESS_DEVICE = torch.device('cpu')
print(f'v591 HGNet sidecar using weights from {HGNET_WEIGHT_DIR}; model_device={HGNET_DEVICE}; preprocess_device={HGNET_PREPROCESS_DEVICE}')

class HGNetLogMelSpectrogramTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=32000, n_fft=2048, win_length=626, hop_length=313,
            f_min=20, n_mels=256, power=2.0, center=True, pad_mode='reflect',
            norm='slaney', mel_scale='htk')
        self.db = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80.0)
        self.resize = tvt_v2.Resize(size=(256, 256))

    @torch.no_grad()
    def forward(self, wave):
        mel_spec = self.mel_transform(wave)
        lms = self.db(mel_spec)
        lms = self.resize(lms)
        batch_size = lms.shape[0]
        flat = lms.reshape(batch_size, -1)
        lmin = flat.min(dim=1)[0][:, None, None]
        lmax = flat.max(dim=1)[0][:, None, None]
        lms = (lms - lmin) / (lmax - lmin + 1e-7)
        return lms[:, None, :, :]

class HGNetLSEPooling(nn.Module):
    def __init__(self, pool_axis=1, temperature=1.0):
        super().__init__()
        self.pool_axis = pool_axis
        self.temperature = temperature
    def forward(self, x):
        return self.temperature * (
            torch.logsumexp(x / self.temperature, dim=self.pool_axis)
            - math.log(x.shape[self.pool_axis])
        )

class HGNetLSEHead(nn.Module):
    def __init__(self, num_features, num_classes=234, dropout=0.5, lse_temperature=1.0):
        super().__init__()
        self.lse_pool = HGNetLSEPooling(pool_axis=1, temperature=lse_temperature)
        self.cls_fc = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes),
        )
    def forward(self, h):
        h = torch.mean(h, axis=2)
        h = h.transpose(1, 2)
        timewise_logits = self.cls_fc(h)
        return self.lse_pool(timewise_logits)

class HGNetLSEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'hgnetv2_b0.ssld_stage2_ft_in1k', pretrained=False, in_chans=1,
            global_pool='', num_classes=0, drop_path_rate=0.0)
        with torch.no_grad():
            dummy = torch.randn(1, 1, 256, 256)
            out = self.backbone(dummy)
        self.head = HGNetLSEHead(out.shape[1], HGNET_N_CLASSES, dropout=0.5, lse_temperature=1.0)
    def forward(self, x):
        return self.head(self.backbone(x))

def _parse_row_id_for_audio(row_id: str):
    parts = str(row_id).split('_')
    if len(parts) < 2:
        raise ValueError(f'Cannot parse row_id {row_id!r}')
    # BirdCLEF rows are <soundscape_id>_<end_second>.  Public946 dry-run rows
    # look like BC2026_Train_0001_S08_20250606_030007_5, so the end second is
    # the final token, not the second-to-last token.
    end_sec = int(parts[-1])
    audio_id = '_'.join(parts[:-1])
    return audio_id, end_sec

def _load_segment(audio_id: str, end_sec: int, audio_cache: dict, is_dry_run: bool):
    if audio_id not in audio_cache:
        candidates = [BASE / 'test_soundscapes' / f'{audio_id}.ogg', BASE / 'train_soundscapes' / f'{audio_id}.ogg']
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(f'HGNet audio file not found for {audio_id}; checked {candidates}')
        else:
            wav, sr = sf.read(path, dtype='float32')
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            audio_cache[audio_id] = (wav.astype(np.float32), int(sr))
    wav, sr = audio_cache[audio_id]
    if sr != HGNET_SR:
        tw = torch.tensor(wav, dtype=torch.float32)
        wav = torchaudio.functional.resample(tw, sr, HGNET_SR).numpy().astype(np.float32)
        sr = HGNET_SR
        audio_cache[audio_id] = (wav, sr)
    start = max(0, int((end_sec - HGNET_SEGMENT_SEC) * HGNET_SR))
    stop = int(end_sec * HGNET_SR)
    seg = wav[start:stop]
    target_len = HGNET_SR * HGNET_SEGMENT_SEC
    if len(seg) < target_len:
        pad = np.zeros(target_len, dtype=np.float32)
        pad[:len(seg)] = seg
        seg = pad
    elif len(seg) > target_len:
        seg = seg[:target_len]
    return seg.astype(np.float32)

def _rank_normalize_matrix(x: np.ndarray) -> np.ndarray:
    return pd.DataFrame(x).rank(axis=0, pct=True).to_numpy(np.float32)

# Use the already-created ProtoSSM submission as the exact row/column target.
_hgnet_target = pd.read_csv('submission_protossm.csv')
_hgnet_cols = [c for c in _hgnet_target.columns if c != 'row_id']
_hgnet_row_ids = _hgnet_target['row_id'].astype(str).tolist()
_hgnet_test_paths = list((BASE / 'test_soundscapes').glob('*.ogg'))
_hgnet_is_dry_run = len(_hgnet_test_paths) == 0
print(f'v591 HGNet target rows={len(_hgnet_row_ids)}, cols={len(_hgnet_cols)}, dry_run={_hgnet_is_dry_run}')

_audio_cache = {}
_segments = []
for rid in _hgnet_row_ids:
    aid, end_sec = _parse_row_id_for_audio(rid)
    _segments.append(_load_segment(aid, end_sec, _audio_cache, _hgnet_is_dry_run))
_waves = torch.tensor(np.stack(_segments), dtype=torch.float32)
del _segments, _audio_cache
gc.collect()

# Keep STFT/log-mel/resize preprocessing on CPU: Kaggle's current CUDA image can
# throw cudaErrorNoKernelImageForDevice inside torch.stft/pad on the GPU.  Only
# move the finished 1x256x256 tensor batch into the model device.
_lms = HGNetLogMelSpectrogramTransform().eval().to(HGNET_PREPROCESS_DEVICE)
_fold_probs = []
with torch.no_grad():
    for fold in range(4):
        model = HGNetLSEModel().to(HGNET_DEVICE)
        state = torch.load(HGNET_WEIGHT_DIR / f'best_model_fold{fold}.pt', map_location='cpu')
        model.load_state_dict(state, strict=True)
        model.eval()
        logits_batches = []
        for start in range(0, len(_waves), HGNET_BATCH):
            batch = _waves[start:start + HGNET_BATCH].to(HGNET_PREPROCESS_DEVICE)
            lms = _lms(batch).to(HGNET_DEVICE, non_blocking=True)
            logits_batches.append(model(lms).detach().float().cpu().numpy())
        logits = np.concatenate(logits_batches, axis=0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        _fold_probs.append(probs.astype(np.float32))
        print(f'v591 HGNet fold {fold}: probs shape={probs.shape}, min={float(np.nanmin(probs)):.6f}, max={float(np.nanmax(probs)):.6f}')
        del model, logits_batches, logits, probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

_hgnet_probs = np.mean(np.stack(_fold_probs, axis=0), axis=0).astype(np.float32)
if _hgnet_probs.shape != (len(_hgnet_row_ids), len(_hgnet_cols)):
    raise RuntimeError(f'HGNet prediction shape mismatch: {_hgnet_probs.shape} vs {(len(_hgnet_row_ids), len(_hgnet_cols))}')
if not np.isfinite(_hgnet_probs).all():
    raise RuntimeError('HGNet predictions contain non-finite values')
if float(np.max(_hgnet_probs) - np.min(_hgnet_probs)) <= 1e-8:
    raise RuntimeError('HGNet predictions are constant; refusing silent fallback')

_hgnet_sub = _hgnet_target.copy()
_hgnet_sub[_hgnet_cols] = _hgnet_probs
_hgnet_sub.to_csv(HGNET_OUT_CSV, index=False)
print(f'v591 wrote {HGNET_OUT_CSV}: shape={_hgnet_sub.shape}, min={float(_hgnet_probs.min()):.6f}, max={float(_hgnet_probs.max()):.6f}')
del _waves, _fold_probs, _hgnet_probs, _hgnet_sub, _lms
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# %% cell 65
import os
import numpy as np
import pandas as pd
from pathlib import Path

PROTOSSM_CSV = "submission_protossm.csv"
SED_CSV      = "submission_sed.csv"
OUT_CSV      = "submission.csv"

EPS = 1e-5

df_proto = pd.read_csv(PROTOSSM_CSV)
df_sed   = pd.read_csv(SED_CSV)

cols = [c for c in df_proto.columns if c != "row_id"]

df_sed = df_sed.set_index("row_id").loc[df_proto["row_id"]].reset_index()
p_proto = np.clip(df_proto[cols].to_numpy(np.float32), EPS, 1.0 - EPS)
p_sed   = np.clip(df_sed[cols].to_numpy(np.float32), EPS, 1.0 - EPS)

rank_proto = pd.DataFrame(p_proto).rank(axis=0, pct=True).to_numpy(np.float32)
rank_sed   = pd.DataFrame(p_sed).rank(axis=0, pct=True).to_numpy(np.float32)

HGNET_CSV = "submission_hgnet.csv"
HGNET_RANK_WEIGHT = 0.025
if not os.path.exists(HGNET_CSV):
    raise FileNotFoundError(f"Expected v591 HGNet sidecar output {HGNET_CSV}")
df_hgnet = pd.read_csv(HGNET_CSV)
df_hgnet = df_hgnet.set_index("row_id").loc[df_proto["row_id"]].reset_index()
p_hgnet = np.clip(df_hgnet[cols].to_numpy(np.float32), EPS, 1.0 - EPS)
rank_hgnet = pd.DataFrame(p_hgnet).rank(axis=0, pct=True).to_numpy(np.float32)

base_mass = 1.0 - HGNET_RANK_WEIGHT
proto_w = 0.60 * base_mass
sed_w = 0.40 * base_mass
print(f"Executing v591 3-way rank blend: Proto={proto_w:.4f} SED={sed_w:.4f} HGNet={HGNET_RANK_WEIGHT:.4f}")
pred = (rank_proto * proto_w) + (rank_sed * sed_w) + (rank_hgnet * HGNET_RANK_WEIGHT)

row_ids = df_proto["row_id"].astype(str).to_numpy()
file_ids = np.array(["_".join(r.split("_")[:-1]) for r in row_ids])

fake_only = (p_proto > 0.50) & (p_sed < 0.05)
pred = np.where(fake_only, (1.0 - 0.08) * pred + 0.08 * rank_proto, pred)

offs = np.arange(-3, 4, dtype=np.float32)
proto_kernel = (1.0 + (offs / 1.20) ** 2 / 2.0) ** (-1.5)
proto_kernel = (proto_kernel / proto_kernel.sum()).astype(np.float32)

pa_ctx = p_proto.copy()
for fid in pd.unique(file_ids):
    m = file_ids == fid
    x = p_proto[m]
    if len(x) > 1:
        xp = np.pad(x, ((3, 3), (0, 0)), mode="edge")
        pa_ctx[m] = sum(proto_kernel[i] * xp[i:i + len(x)] for i in range(7))

xctx = pd.DataFrame(pa_ctx).rank(axis=0, pct=True).to_numpy(np.float32)
proto_cont = (xctx > 0.88) & (rank_proto > 0.75) & (p_sed < 0.12) & (~fake_only)
pred = np.where(proto_cont, (1.0 - 0.15) * pred + 0.15 * np.maximum(rank_proto, xctx), pred)

sed_only = (rank_sed > 0.95) & (rank_proto < 0.80) & (~fake_only) & (~proto_cont)
pred = np.where(sed_only, (1.0 - 0.12) * pred + 0.12 * rank_sed, pred)

sub = df_proto.copy()
sub[cols] = pred.astype(np.float32)

MIRROR_PAIRS = (
    ("47158son15", "47158son16"), 
    ("47158son09", "47158son12"),
    ("47158son02", "47158son14"), 
    ("47158son13", "47158son21", "47158son22", "47158son23")
)
col_to_idx = {l: i for i, l in enumerate(cols)}

mirror_count = 0
for group in MIRROR_PAIRS:
    valid_idx = [col_to_idx[s] for s in group if s in col_to_idx]
    if len(valid_idx) >= 2:
        group_max = sub[cols].iloc[:, valid_idx].max(axis=1).to_numpy(np.float32)
        for idx in valid_idx:
            sub.iloc[:, idx + 1] = group_max
        mirror_count += len(valid_idx)
print(f"Sonotype mirroring applied to {mirror_count} columns.")

try:
    tax_df = pd.read_csv(BASE / "taxonomy.csv").set_index("primary_label")
    rare_classes = {"Amphibia", "Mammalia", "Reptilia"}
    rare_count = 0
    
    for ci, species in enumerate(cols):
        if species in tax_df.index and tax_df.loc[species, "class_name"] in rare_classes:
            col_idx = ci + 1
            vals = sub.iloc[:, col_idx].to_numpy(np.float32)
            thr = vals.mean() + 0.05
            sub.iloc[:, col_idx] = np.where(vals < thr, vals * 0.9, vals)
            rare_count += 1
    print(f"Adaptive thresholding applied to {rare_count} rare species.")
except Exception as e:
    print(f"Adaptive thresholding skipped: {e}")

test_paths = list((BASE / "test_soundscapes").glob("*.ogg"))
IS_DRY_RUN = len(test_paths) == 0

if IS_DRY_RUN:
    print("Dry-run: keeping full train-soundscape row_ids for validation.")

sub.to_csv(OUT_CSV, index=False)
print("Blend and post-processing complete. Ready for submission!")

# %% markdown
# ## What the ablations taught
# 
# I used the five 2026-05-10 submissions to test whether the 0.946 result was coming from one branch or from the final blend. The answer was clear:
# 
# | Experiment | Public LB | Interpretation |
# |---|---:|---|
# | Original V8 blend | **0.946** | Best confirmed anchor. |
# | 50/50 ProtoSSM-SED rank blend | **0.946** | Competitive, but not an improvement. |
# | 70/30 and 80/20 ProtoSSM-heavy blends | 0.944 / 0.942 | More temporal-branch weight overfits the public dry-run proxy. |
# | ProtoSSM only | 0.929 | The Perch sequence branch needs the SED complement. |
# | SED only | 0.926 | The SED branch alone is far too weak despite optimistic dry-run behavior. |
# 
# For practical forking, I would therefore change the blend only when there is a genuinely new signal: a new public model family, a stronger addable checkpoint, or a validation scheme that is less tied to train-soundscape leakage. Simple weight sweeps are already mostly exhausted here.

# %% markdown
# ## Final checklist before submission
# 
# | Check | Expected result |
# |---|---|
# | `submission.csv` exists | The final blend cell writes this file in `/kaggle/working`. |
# | Column order | `row_id` followed by the exact labels from `sample_submission.csv`. |
# | Row count | Hidden rerun: one row per test file x 12 windows. Public dry-run: aligned to the 240-row sample contract. |
# | Value range | Prediction columns are clipped to `[0, 1]`. |
# | NaNs | Zero NaNs. |
# | Runtime | Submit mode skips the expensive optional OOF branch. |
# 
# ## Debugging map
# 
# ```text
# Path error
#   -> Check the attached Kaggle datasets, model source, and notebook source.
# 
# Shape error
#   -> Check N_WINDOWS, row_id order, and whether every file produces 12 windows.
# 
# Cache error
#   -> Delete /kaggle/working/cache and rebuild the local Perch cache.
# 
# Submission error
#   -> Compare submission.csv columns and rows with sample_submission.csv.
# 
# Unexpected low score
#   -> Verify that hidden test audio was mounted and that you did not submit a dry-run/debug output.
# ```
# 
# When reusing this notebook, please keep the public attribution chain intact and verify the final output shape before submitting.
