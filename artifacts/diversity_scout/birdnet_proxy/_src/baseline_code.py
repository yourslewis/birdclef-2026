# ── Install dependencies ──────────────────────────────────────────────────────
# birdnetlib: clean Python wrapper around BirdNET-Analyzer (Cornell Lab)
# librosa:    audio loading and feature extraction
# soundfile:  fast .ogg/.wav reading

!pip install birdnetlib --quiet
!pip install librosa soundfile --quiet

# ── Standard imports ──────────────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings('ignore')
print('✅ All libraries imported successfully')

# ===CELL===
class CFG:
    # ── Dataset Paths ─────────────────────────────────────────────────────────
    BASE_DIR          = '/kaggle/input/competitions/birdclef-2026'
    TRAIN_CSV         = f'{BASE_DIR}/train.csv'
    TAXONOMY_CSV      = f'{BASE_DIR}/taxonomy.csv'
    SAMPLE_SUB_CSV    = f'{BASE_DIR}/sample_submission.csv'
    TRAIN_LABELS_CSV  = f'{BASE_DIR}/train_soundscapes_labels.csv'
    RECORDING_LOC_TXT = f'{BASE_DIR}/recording_location.txt'

    # Audio folders (populate once you see full dataset structure)
    # These are the EXPECTED paths based on previous BirdCLEF years
    TRAIN_AUDIO_DIR   = f'{BASE_DIR}/train_audio'        # individual species recordings
    TRAIN_SOUNDSCAPES = f'{BASE_DIR}/train_soundscapes'  # labeled soundscape .ogg files
    TEST_SOUNDSCAPES  = f'{BASE_DIR}/test_soundscapes'   # hidden test soundscapes
    UNLABELED_DIR     = f'{BASE_DIR}/unlabeled_soundscapes'  # optional unlabeled data

    # ── Audio Parameters ──────────────────────────────────────────────────────
    SR              = 32000   # sample rate (32kHz is BirdCLEF standard)
    SEGMENT_DURATION = 5      # seconds per segment (competition standard)
    HOP_DURATION    = 5       # no overlap for baseline (can reduce for better coverage)

    # ── Mel Spectrogram Parameters ────────────────────────────────────────────
    N_FFT       = 1024
    HOP_LENGTH  = 512
    N_MELS      = 128
    FMIN        = 20
    FMAX        = 16000

    # ── Inference ─────────────────────────────────────────────────────────────
    MIN_CONFIDENCE = 0.1    # BirdNET confidence threshold
    BATCH_SIZE     = 8      # for CPU, keep low

    # ── Output ────────────────────────────────────────────────────────────────
    OUTPUT_DIR  = '/kaggle/working'
    SUBMISSION  = f'{OUTPUT_DIR}/submission.csv'

cfg = CFG()
print('✅ Config loaded')
print(f'   Base dir exists: {os.path.exists(cfg.BASE_DIR)}')
print(f'   Segment size   : {cfg.SEGMENT_DURATION}s @ {cfg.SR}Hz')

# ===CELL===
# ── Load all metadata CSVs ────────────────────────────────────────────────────
train_df    = pd.read_csv(cfg.TRAIN_CSV)
taxonomy_df = pd.read_csv(cfg.TAXONOMY_CSV)
sample_sub  = pd.read_csv(cfg.SAMPLE_SUB_CSV)
labels_df   = pd.read_csv(cfg.TRAIN_LABELS_CSV)

# ── Recording locations ───────────────────────────────────────────────────────
with open(cfg.RECORDING_LOC_TXT, 'r') as f:
    print('📍 Recording Locations:')
    print(f.read())

# ===CELL===
# ── train.csv ─────────────────────────────────────────────────────────────────
print('=== train.csv ===')
print(f'Shape: {train_df.shape}')
print(f'Columns: {list(train_df.columns)}')
display(train_df.head(5))

# Species count
if 'species_code' in train_df.columns:
    print(f'\n🐦 Unique species: {train_df["species_code"].nunique()}')
elif 'primary_label' in train_df.columns:
    print(f'\n🐦 Unique species: {train_df["primary_label"].nunique()}')

# ===CELL===
# ── taxonomy.csv ──────────────────────────────────────────────────────────────
print('=== taxonomy.csv ===')
print(f'Shape: {taxonomy_df.shape}')
print(f'Columns: {list(taxonomy_df.columns)}')
display(taxonomy_df.head(5))

# ===CELL===
# ── sample_submission.csv ─────────────────────────────────────────────────────
# VERY IMPORTANT — this defines the exact format we must output
print('=== sample_submission.csv ===')
print(f'Shape: {sample_sub.shape}')
print(f'Columns: {list(sample_sub.columns[:10])} ... ({len(sample_sub.columns)} total)')
display(sample_sub.head(3))

# Extract the species columns (everything except row_id)
species_cols = [c for c in sample_sub.columns if c != 'row_id']
print(f'\n🔑 Number of species to predict: {len(species_cols)}')
print(f'First 10 species codes: {species_cols[:10]}')

# ===CELL===
# ── train_soundscapes_labels.csv ──────────────────────────────────────────────
print('=== train_soundscapes_labels.csv ===')
print(f'Shape: {labels_df.shape}')
print(f'Columns: {list(labels_df.columns)}')
display(labels_df.head(5))

# These are our labeled validation soundscapes
# row_id format is typically: soundscape_XXXXX_5, soundscape_XXXXX_10 ...
# where the number = end-second of each 5s segment

# ===CELL===
# ── Class imbalance check ─────────────────────────────────────────────────────
# How many recordings per species? (crucial for understanding difficulty)
label_col = 'primary_label' if 'primary_label' in train_df.columns else 'species_code'

species_counts = train_df[label_col].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Distribution
axes[0].hist(species_counts.values, bins=50, color='steelblue', edgecolor='white')
axes[0].set_title('Distribution of recordings per species', fontsize=13)
axes[0].set_xlabel('Number of recordings')
axes[0].set_ylabel('Number of species')

# Top/bottom species
top20 = species_counts.head(20)
axes[1].barh(top20.index[::-1], top20.values[::-1], color='coral')
axes[1].set_title('Top 20 species by recording count', fontsize=13)
axes[1].set_xlabel('Number of recordings')

plt.tight_layout()
plt.show()

print(f'\n📊 Species with < 5 recordings : {(species_counts < 5).sum()}')
print(f'📊 Species with < 10 recordings: {(species_counts < 10).sum()}')
print(f'📊 Max recordings (most common): {species_counts.max()}')
print(f'📊 Min recordings (rarest)     : {species_counts.min()}')

# ===CELL===
# ── Helper: visualize one audio file ─────────────────────────────────────────
def visualize_audio(filepath, title='Audio Sample', max_duration=30):
    """
    Load and plot waveform + mel spectrogram for a given audio file.
    max_duration: only visualize first N seconds (keeps it fast)
    """
    y, sr = librosa.load(filepath, sr=cfg.SR, duration=max_duration)
    duration = librosa.get_duration(y=y, sr=sr)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))

    # Waveform
    librosa.display.waveshow(y, sr=sr, ax=axes[0], color='steelblue')
    axes[0].set_title(f'{title} — Waveform ({duration:.1f}s)', fontsize=12)
    axes[0].set_xlabel('Time (s)')

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(
        mel_db, sr=sr, hop_length=cfg.HOP_LENGTH,
        x_axis='time', y_axis='mel',
        fmin=cfg.FMIN, fmax=cfg.FMAX,
        ax=axes[1], cmap='magma'
    )
    axes[1].set_title('Mel Spectrogram', fontsize=12)
    fig.colorbar(img, ax=axes[1], format='%+2.0f dB')

    plt.tight_layout()
    plt.show()
    print(f'Duration: {duration:.2f}s | Sample rate: {sr}Hz | Samples: {len(y)}')

print('✅ visualize_audio() helper defined')

# ===CELL===
# ── Visualize a sample training audio recording ───────────────────────────────
# We pick the first file from train.csv

filename_col = 'filename' if 'filename' in train_df.columns else 'filepath'

sample_file = train_df[filename_col].iloc[0]

# Build the full path
if not sample_file.startswith('/'):
    sample_path = os.path.join(cfg.TRAIN_AUDIO_DIR, sample_file)
else:
    sample_path = sample_file

print(f'Loading: {sample_path}')

if os.path.exists(sample_path):
    label_val = train_df[label_col].iloc[0]
    visualize_audio(sample_path, title=f'Species: {label_val}')
else:
    print(f'⚠️  File not found at {sample_path}')
    print('   Check that TRAIN_AUDIO_DIR is correct in CFG')
    # List what's actually in BASE_DIR
    print('\n📁 Contents of BASE_DIR:')
    for item in sorted(os.listdir(cfg.BASE_DIR)):
        print(f'   {item}')

# ===CELL===
# ── Visualize a train soundscape ──────────────────────────────────────────────
# Soundscapes are the long continuous recordings we actually predict on

soundscape_files = []
if os.path.exists(cfg.TRAIN_SOUNDSCAPES):
    soundscape_files = list(Path(cfg.TRAIN_SOUNDSCAPES).glob('*.ogg'))
    print(f'Found {len(soundscape_files)} train soundscapes')

if soundscape_files:
    visualize_audio(str(soundscape_files[0]), title='Train Soundscape', max_duration=30)
else:
    print('⚠️  No soundscape files found. Check TRAIN_SOUNDSCAPES path.')

# ===CELL===
def load_and_segment_audio(filepath, sr=cfg.SR, segment_duration=cfg.SEGMENT_DURATION):
    """
    Load a full soundscape and split it into fixed-length segments.

    Returns:
        segments : list of np.arrays, each of length sr*segment_duration
        end_times: list of ints, the end-second for each segment
                   (used to build row_id)
    """
    # Load full audio at target sample rate
    y, _ = librosa.load(filepath, sr=sr, mono=True)
    total_duration = len(y) / sr

    segment_samples = sr * segment_duration
    segments  = []
    end_times = []

    # Slide window across audio with no overlap
    for start_sample in range(0, len(y), segment_samples):
        end_sample = start_sample + segment_samples
        segment = y[start_sample:end_sample]

        # Pad last segment if shorter than expected
        if len(segment) < segment_samples:
            segment = np.pad(segment, (0, segment_samples - len(segment)))

        end_sec = (start_sample // segment_samples + 1) * segment_duration

        segments.append(segment)
        end_times.append(end_sec)

    return segments, end_times, total_duration


def audio_to_melspectrogram(y, sr=cfg.SR):
    """
    Convert a 1D audio array to a 2D mel spectrogram (in dB scale).
    This is the 'image' we feed into models.
    """
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


print('✅ Segmentation helpers defined')

# Quick test on a soundscape (if available)
if soundscape_files:
    test_path = str(soundscape_files[0])
    segs, ends, dur = load_and_segment_audio(test_path)
    print(f'\nSoundscape: {Path(test_path).name}')
    print(f'  Total duration : {dur:.1f}s')
    print(f'  Segments (5s)  : {len(segs)}')
    print(f'  End times      : {ends[:5]} ...')
    print(f'  Segment shape  : {segs[0].shape}')

# ===CELL===
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer

# Load BirdNET model (downloads weights once, cached offline after)
# ⚠️  For submission: pre-download weights into a Kaggle dataset and load from there
print('Loading BirdNET analyzer...')
analyzer = Analyzer()
print('✅ BirdNET loaded')

# What species does BirdNET know about?
print(f'BirdNET covers {len(analyzer.labels)} species labels')

# ===CELL===
# ── Map BirdNET labels → competition species codes ────────────────────────────
#
# BirdNET uses 'Scientific Name_Common Name' format
# Competition uses short species codes (e.g. 'amered', 'houspa')
# We build a lookup dictionary via taxonomy.csv

# Display taxonomy to understand the mapping columns
print('Taxonomy columns:', list(taxonomy_df.columns))
display(taxonomy_df.head(3))

# Build mapping: scientific_name → species_code
# (Adjust column names below based on actual taxonomy.csv structure)
sci_col  = 'scientific_name' if 'scientific_name' in taxonomy_df.columns else taxonomy_df.columns[1]
code_col = 'species_code'    if 'species_code'    in taxonomy_df.columns else taxonomy_df.columns[0]

sci_to_code = dict(zip(
    taxonomy_df[sci_col].str.lower().str.strip(),
    taxonomy_df[code_col]
))

print(f'\n✅ Built {len(sci_to_code)} species mappings')
print('Sample mappings:', list(sci_to_code.items())[:3])

# ===CELL===
def run_birdnet_on_file(filepath, lat=None, lon=None, week=-1, min_conf=cfg.MIN_CONFIDENCE):
    """
    Run BirdNET on a single audio file.

    BirdNET can be location-aware (lat/lon/week) which improves accuracy.
    The Pantanal is roughly: lat=-17.0, lon=-57.0

    Returns:
        detections: list of dicts with keys:
            start_time, end_time, scientific_name, common_name, confidence
    """
    recording = Recording(
        analyzer,
        filepath,
        lat=lat or -17.0,       # Pantanal center latitude
        lon=lon or -57.0,       # Pantanal center longitude
        week_48=week,           # week of year (1-48), -1 = ignore seasonality
        min_conf=min_conf
    )
    recording.analyze()
    return recording.detections


def detections_to_scores(detections, species_cols, soundscape_name, total_secs):
    """
    Convert BirdNET detections into submission format rows.

    For each 5s segment, we create a row:
        row_id = soundscape_XXXXX_10  (end second)
        species columns = confidence scores (0 if not detected)

    Returns: pd.DataFrame with submission rows for this soundscape
    """
    segment_secs = range(cfg.SEGMENT_DURATION, total_secs + 1, cfg.SEGMENT_DURATION)
    rows = []

    for end_sec in segment_secs:
        start_sec = end_sec - cfg.SEGMENT_DURATION
        row_id = f'{soundscape_name}_{end_sec}'

        # Initialize all species scores at 0
        row = {col: 0.0 for col in species_cols}
        row['row_id'] = row_id

        # Find detections that overlap this time window
        for det in detections:
            if det['start_time'] >= start_sec and det['end_time'] <= end_sec + 1:
                # Map scientific name to competition code
                sci_name = det.get('scientific_name', '').lower().strip()
                code = sci_to_code.get(sci_name)
                if code and code in species_cols:
                    # Take max confidence if multiple detections for same species in segment
                    row[code] = max(row[code], det['confidence'])

        rows.append(row)

    df = pd.DataFrame(rows)
    # Ensure correct column order
    df = df[['row_id'] + species_cols]
    return df


print('✅ BirdNET inference helpers defined')

# ===CELL===
# ── Get test soundscape files ─────────────────────────────────────────────────
test_files = sorted(Path(cfg.TEST_SOUNDSCAPES).glob('*.ogg'))
print(f'Found {len(test_files)} test soundscapes')

if len(test_files) == 0:
    print('⚠️  No test files found — check TEST_SOUNDSCAPES path')
    print('   This is expected if the hidden test set is not yet populated')
    print('   During actual submission, test_soundscapes/ will contain the hidden files')

# ===CELL===
# ── Get species columns from sample_submission.csv ────────────────────────────
species_cols = [c for c in sample_sub.columns if c != 'row_id']
print(f'Species to predict: {len(species_cols)}')

# ── Main inference loop ───────────────────────────────────────────────────────
all_predictions = []

for filepath in tqdm(test_files, desc='Processing soundscapes'):
    soundscape_name = filepath.stem  # e.g. 'soundscape_12345'

    try:
        # Get total duration of this soundscape
        y_temp, _ = librosa.load(str(filepath), sr=cfg.SR, mono=True)
        total_secs = int(len(y_temp) / cfg.SR)

        # Run BirdNET
        detections = run_birdnet_on_file(str(filepath))
        print(f'  {soundscape_name}: {len(detections)} detections in {total_secs}s')

        # Convert to submission rows
        df_pred = detections_to_scores(
            detections, species_cols, soundscape_name, total_secs
        )
        all_predictions.append(df_pred)

    except Exception as e:
        print(f'  ❌ Error on {soundscape_name}: {e}')
        continue

print(f'\n✅ Inference complete on {len(all_predictions)} files')

# ===CELL===
# ── Concatenate all predictions ───────────────────────────────────────────────
if all_predictions:
    submission_df = pd.concat(all_predictions, ignore_index=True)
else:
    print('⚠️  No predictions made — using sample_submission as fallback')
    submission_df = sample_sub.copy()

print(f'Predictions shape: {submission_df.shape}')
display(submission_df.head(3))

# ── Align with sample_submission format ──────────────────────────────────────
submission_df = sample_sub[['row_id']].merge(
    submission_df,
    on='row_id',
    how='left'
)
submission_df[species_cols] = submission_df[species_cols].fillna(0.0)
submission_df[species_cols] = submission_df[species_cols].clip(0.0, 1.0)

# Final validation
print('=== Submission Validation ===')
print(f'Shape           : {submission_df.shape}')
print(f'Expected shape  : {sample_sub.shape}')
print(f'NaN values      : {submission_df.isnull().sum().sum()}')
print(f'Columns match   : {list(submission_df.columns) == list(sample_sub.columns)}')
display(submission_df.head(5))

# ── Save submission ───────────────────────────────────────────────────────────
# ALWAYS save to /kaggle/working/ — hardcoded, not from cfg
SAVE_PATH = '/kaggle/working/submission.csv'
submission_df.to_csv(SAVE_PATH, index=False)

# Verify it actually exists on disk
import os
assert os.path.exists(SAVE_PATH), "❌ submission.csv was NOT saved!"
print(f'✅ Submission saved to: {SAVE_PATH}')
print(f'   File size: {os.path.getsize(SAVE_PATH) / 1024:.1f} KB')

verify = pd.read_csv(SAVE_PATH)
print(f'   Verified shape: {verify.shape}')
print('\n🎉 Ready to submit!')