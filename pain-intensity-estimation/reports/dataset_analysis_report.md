# PMPDB Dataset Analysis Report

## 1. Overview & Participant Count
- **Dataset Detected**: `PMPDB (PainMonit Physiotherapy Database)`
- **Location**: `C:\Users\Amol\Downloads\PMCD\PMPDB\raw-data`
- **File Format**: Semicolon-delimited CSV (sep=';', decimal=',') + plain-text threshold files
- **Number of Unique Participants**: **49** (`P01` through `P49`)
  - Participants with 2 sessions (`_1`, `_2`): **41**
  - Participants with 1 session: **8** (P11 (session 1), P15 (session 1), P31 (session 1), P33 (session 1), P38 (session 2), P41 (session 2), P43 (session 2), P48 (session 1))

## 2. Number of Recordings
- **Main Session Recordings (`<ID>.csv`)**: **90** (2,969,065 rows)
- **Pre-Session Baseline/Calibration Recordings (`<ID>_runUp.csv`)**: **90** (789,764 rows)
- **Per-Session Threshold Metadata**: 90 `noPainThreshold.txt` and 90 `severePainThreshold.txt` files

## 3. Available Modalities (and Explicitly Absent Modalities)
All 90 recordings share an identical 11-column schema (`Seconds`, 8 physiological/biomechanical channels, `Pain rates`, `Pain labels`):

| Modality | Columns | Sensor Source | Min | Max | Mean | Std |
|---|---|---|---|---:|---:|---:|
| **BVP** | `Bvp` | Blood Volume Pulse / PPG (Empatica E4 wrist + RespiBAN optical sensor) | -736.5695 | 489.9283 | -0.0051 | 41.7828 |
| **BVP** | `Bvp_RB` | Blood Volume Pulse / PPG (Empatica E4 wrist + RespiBAN optical sensor) | -1.4977 | 1.1776 | -0.0015 | 0.1458 |
| **EDA** | `Eda_E4` | Electrodermal Activity / Skin Conductance (Empatica E4 + RespiBAN) | -0.0000 | 61.4561 | 5.8922 | 9.1018 |
| **EDA** | `Eda_RB` | Electrodermal Activity / Skin Conductance (Empatica E4 + RespiBAN) | 2.0961 | 24.6011 | 10.1444 | 4.4973 |
| **Resp** | `Resp` | Respiration (RespiBAN inductive chest belt) | -21.9958 | 17.2358 | -0.0266 | 2.3686 |
| **Tmp** | `Tmp` | Peripheral Skin Temperature in °C (Empatica E4) | 29.3100 | 36.3900 | 33.2781 | 1.2922 |
| **EMG** | `Emg` | Surface Electromyography (RespiBAN) | -0.4099 | 0.4296 | 0.0011 | 0.0108 |
| **Grip** | `Grip` | Grip Force / Hand Dynamometer (RespiBAN) | -2.6441 | 14.9462 | -1.8353 | 0.9464 |

### Explicitly Absent Modalities
- **ECG**: **Not present** in PMPDB. Cardiac/autonomic dynamics are captured via dual Blood Volume Pulse (BVP/PPG) sensors (`Bvp` and `Bvp_RB`), from which pulse rate (PR) and pulse rate variability (PRV) are derived.
- **EEG**: **Not present** in PMPDB. No cortical/EEG signals or EEG channel explanations are fabricated or claimed.

## 4. Sampling Frequencies & Synchronization
- **Unified Sampling Frequency**: **250.0 Hz** (`dt = 0.004 s`) across all 90 main recordings and 90 run-up recordings (`max std(dt) = 0.00000000 s`).
- **Cross-Device Synchronization**: Empatica E4 wearable streams (`Bvp`, `Eda_E4`, `Tmp` — natively sampled at 64 Hz, 4 Hz, and 4 Hz respectively) are already time-aligned and upsampled/interpolated onto the 250 Hz RespiBAN (`Resp`, `Eda_RB`, `Bvp_RB`, `Emg`, `Grip`) timebase (`Seconds`). Note that `Tmp` shows a lower effective value-change rate (~32–38 Hz due to quantized 0.02 °C steps and interpolation).

## 5. Missing Values
| Column | Missing Samples | Missing % | Interpretation |
|---|---:|---:|---|
| `Seconds` | 0 | 0.0000% | Zero missing samples across all 90 recordings |
| `Bvp` | 0 | 0.0000% | Zero missing samples across all 90 recordings |
| `Eda_E4` | 0 | 0.0000% | Zero missing samples across all 90 recordings |
| `Tmp` | 0 | 0.0000% | Zero missing samples across all 90 recordings |
| `Resp` | 0 | 0.0000% | Zero missing samples across all 90 recordings |
| `Eda_RB` | 0 | 0.0000% | Zero missing samples across all 90 recordings |
| `Bvp_RB` | 0 | 0.0000% | Zero missing samples across all 90 recordings |
| `Emg` | 0 | 0.0000% | Zero missing samples across all 90 recordings |
| `Grip` | 0 | 0.0000% | Zero missing samples across all 90 recordings |
| `Pain rates` | 2,967,641 | 99.9520% | Event-stamped sparse self-reports (NaN between rating timestamps) |
| `Pain labels` | 2,967,598 | 99.9506% | Event-stamped sparse self-reports (NaN between rating timestamps) |

## 6. Recording Duration
- **Main Sessions**: min = 69.52 s, median = 129.82 s, mean = 131.95 s, max = 233.62 s
- **Total Main Session Duration**: **11,875.9 s (3.30 hours)**
- **Total Pre-Session Run-Up Duration**: **0.88 hours**

## 7. Label Distribution & Rating Structure
- **Rating Modality**: Discrete event-stamped NRS self-reports (1–10 scale), NOT continuous 250 Hz CoVAS.
- **Total Numeric `Pain rates` Events**: **1,424** (range 1–10 NRS, mean = 4.699, std = 2.459, median = 5.0).
- **Total Ordinal `Pain labels` Events**: **1,467** (`0` No/Low: 348, `1` Moderate: 844, `2` Severe: 275).
- **Label-without-Rate Events**: 43 rows across 20 recordings carry `Pain labels = 0` with `Pain rates = NaN`. To avoid fabricating numeric regression targets, primary regression evaluation uses the **1,424 verified numeric ratings** (mapped linearly via `NRS × 10` to the 0–100 scale).
- **Threshold Mapping Audit**: 1424/1424 exact matches (0 if rate < noPainThreshold; 1 if noPainThreshold <= rate <= severePainThreshold; 2 if rate > severePainThreshold).
- **Inter-Rating Intervals**: min = 0.0 s, P25 = 4.8 s, median = 6.98 s, mean = 7.8 s, P75 = 10.0 s, max = 46.01 s.

### NRS 1–10 Distribution
| NRS Rating | Mapped 0–100 Score | Count | Share |
|---:|---:|---:|---:|
| 1 | 10 | 146 | 10.3% |
| 2 | 20 | 180 | 12.6% |
| 3 | 30 | 198 | 13.9% |
| 4 | 40 | 179 | 12.6% |
| 5 | 50 | 180 | 12.6% |
| 6 | 60 | 173 | 12.1% |
| 7 | 70 | 149 | 10.5% |
| 8 | 80 | 110 | 7.7% |
| 9 | 90 | 75 | 5.3% |
| 10 | 100 | 34 | 2.4% |

## 8. Data Quality & Integrity
- **Zero NaN physiological samples** across all 2,969,065 main samples and 789,764 run-up samples.
- **Zero flatlined channels** (`std < 1e-5`).
- **Out-of-range / anomaly flags detected**: **0**.
- **Windowing Implication**: Because consecutive pain ratings are spaced by a median of 6.98 s (IQR 4.80–10.00 s), a **6.0-second causal window `[t - 6.0s, t]`** ending at each self-report timestamp `t` captures full autonomic (EDA, BVP) and biomechanical (EMG, Grip, Resp) response dynamics without future leakage.

## 9. Synchronization & Baseline Normalization Strategy
- All 8 channels are pre-synchronized at 250 Hz on a shared monotonic `Seconds` axis.
- Every main session has a paired pre-session `<ID>_runUp.csv` recording (mean 35.1 s) recorded prior to pain ratings. Using each recording's own `runUp` mean/std for physiological baseline-shift features provides subject-specific calibration with **zero data leakage** across train/test splits.