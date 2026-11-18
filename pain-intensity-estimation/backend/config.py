"""Central configuration for the Continuous Pain-Intensity Estimation System (PMPDB)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
RAW_DATASET_DIR = WORKSPACE_ROOT / "PMPDB" / "raw-data"

DATASET_DIR = PROJECT_ROOT / "dataset"
PROCESSED_DIR = DATASET_DIR / "processed"
METADATA_DIR = DATASET_DIR / "metadata"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Ground-truth PMPDB schema (verified from all 90 recordings)
SAMPLING_RATE_HZ = 250.0
DOWNSAMPLED_RATE_HZ = 50.0  # For deep learning temporal encoders (300 steps per 6s window)
WINDOW_SECONDS = 6.0        # Causal [t - 6.0s, t] window aligned to each pain rating
WINDOW_SAMPLES_RAW = int(SAMPLING_RATE_HZ * WINDOW_SECONDS)        # 1500
WINDOW_SAMPLES_DL = int(DOWNSAMPLED_RATE_HZ * WINDOW_SECONDS)      # 300

SIGNAL_COLUMNS = [
    "Bvp",
    "Eda_E4",
    "Tmp",
    "Resp",
    "Eda_RB",
    "Bvp_RB",
    "Emg",
    "Grip",
]

# Explicit modality groupings (no ECG or EEG exists in PMPDB)
MODALITY_GROUPS = {
    "BVP": ["Bvp", "Bvp_RB"],
    "EDA": ["Eda_E4", "Eda_RB"],
    "Resp": ["Resp"],
    "Tmp": ["Tmp"],
    "EMG": ["Emg"],
    "Grip": ["Grip"],
}

MODALITY_ORDER = ["BVP", "EDA", "Resp", "Tmp", "EMG", "Grip"]

MODALITY_DESCRIPTIONS = {
    "BVP": "Blood Volume Pulse / PPG (Empatica E4 wrist + RespiBAN optical sensor)",
    "EDA": "Electrodermal Activity / Skin Conductance (Empatica E4 + RespiBAN)",
    "Resp": "Respiration (RespiBAN inductive chest belt)",
    "Tmp": "Peripheral Skin Temperature in °C (Empatica E4)",
    "EMG": "Surface Electromyography (RespiBAN)",
    "Grip": "Grip Force / Hand Dynamometer (RespiBAN)",
}

ABSENT_MODALITIES = {
    "ECG": "Not recorded in PMPDB. Pulse rate and pulse rate variability (PRV) are extracted from BVP/PPG instead.",
    "EEG": "Not recorded in PMPDB. No brain/cortical features or EEG channel explanations are claimed.",
}

# Label scale: PMPDB 'Pain rates' are 1–10 NRS self-reports -> mapped linearly to 0–100 (x 10)
PAIN_SCALE_FACTOR = 10.0
PAIN_MIN = 0.0
PAIN_MAX = 100.0

RANDOM_SEED = 42
N_CV_FOLDS = 5

