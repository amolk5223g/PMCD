# Research Methodology Report — Explainable Multimodal Continuous Pain-Intensity Estimation

## 1. Problem Formulation
Objective, continuous pain-intensity estimation from non-invasive physiological and biomechanical sensors is an active research challenge in biomedical engineering and physiotherapy monitoring. Self-reported pain scales (such as the 0–10 Numerical Rating Scale, NRS) remain the clinical gold standard but are intermittent, subjective, and impractical for patients who are sedated, non-verbal, or engaged in physical therapy exercises.

This research prototype models continuous pain intensity on a standardized **0–100 scale** directly from the local **PMPDB (PainMonit Physiotherapy Database)**, combining:
1. **Domain-grounded biomedical signal processing** across all 6 available peripheral modalities (`BVP`, `EDA`, `Resp`, `Tmp`, `EMG`, `Grip`).
2. **Participant-independent validation** (5-Fold GroupKFold and 49-Fold Leave-One-Subject-Out) with zero data leakage.
3. **Comparative machine learning and multimodal deep learning architectures** with explicit modality-specific encoders and missing-modality resilience.
4. **Calibrated uncertainty estimation** (MC Dropout) and **multi-method explainability** (SHAP + Integrated Gradients + neural gating).

## 2. Dataset Reality Check & Integrity Policy
Inspection of `C:\Users\Amol\Downloads\PMCD\PMPDB\raw-data` established key empirical facts that govern our methodology:
- **Participants & Sessions**: 49 unique participants (`P01`–`P49`), 90 main sessions (3.30 hours total) and 90 pre-session `runUp` calibration recordings (0.88 hours total).
- **Available Modalities**: 8 channels uniformly synchronized at **250.0 Hz** (`Bvp`, `Eda_E4`, `Tmp` from Empatica E4; `Resp`, `Eda_RB`, `Bvp_RB`, `Emg`, `Grip` from RespiBAN), with **0.00% missing physiological values**.
- **Explicitly Absent Modalities**: Neither **ECG** nor **EEG** is recorded in PMPDB. Cardiac/autonomic dynamics are extracted via PPG/BVP pulse rate and pulse rate variability (PRV), and no EEG channel explanations are claimed.
- **Label Structure**: Unlike continuous 250 Hz CoVAS sliders, PMPDB provides **1,424 discrete event-stamped NRS ratings** (integer values 1–10, median inter-rating interval 6.98 s) plus **1,467 3-class ordinal labels** (`0/1/2`) deterministically mapped via participant-specific `noPainThreshold.txt` and `severePainThreshold.txt` (verified 1,424/1,424 exact matches). Forty-three rows carry `Pain labels = 0` without a numeric `Pain rates` value; we exclude those from numeric regression rather than fabricating target values.

## 3. Biomedical Signal Processing & Windowing Methodology
To prevent filter edge artifacts, all digital filters are applied to each full 250 Hz recording prior to window slicing:
- **BVP / PPG (`Bvp`, `Bvp_RB`)**: 4th-order zero-phase Butterworth bandpass (`0.5–8.0 Hz`); systolic peak detection (`distance ≥ 0.35 s`); Pulse Rate (BPM), PRV (`SDNN`, `RMSSD`, `pNN50`), normalized peak prominence, and cardiac spectral power ratio (`0.7–3.5 Hz`).
- **EDA (`Eda_E4`, `Eda_RB`)**: 4th-order Butterworth lowpass (`5.0 Hz`); Tonic Skin Conductance Level (SCL) separated via 2nd-order lowpass (`0.05 Hz`); Phasic Skin Conductance Response (SCR = cleaned − tonic); SCR peak count, prominence, positive AUC, slope, and derivative statistics.
- **Respiration (`Resp`)**: 3rd-order bandpass (`0.05–1.0 Hz`); breath-peak rate with Welch PSD fallback for short windows, inter-breath variability, peak-to-peak amplitude, and RMS ratio.
- **Skin Temperature (`Tmp`)**: 3rd-order lowpass (`0.5 Hz`); mean, linear trend (°C/s), delta, and `runUp`-relative temperature shift (°C and z-score).
- **Surface EMG (`Emg`)**: 50 Hz IIR notch filter (European powerline frequency) + 4th-order bandpass (`20–120 Hz`, respecting 125 Hz Nyquist); RMS, MAV, Waveform Length, Zero-Crossing Rate, iEMG, spectral mean/median frequency, and `runUp` RMS ratio.
- **Grip Force (`Grip`)**: 4th-order lowpass (`10.0 Hz`); mean, std, range, slope, delta, and `runUp` z-score shifts.

### Causal Windowing & Zero-Leakage Subject Calibration
- Each ground-truth pain event at timestamp $t$ is aligned with a strictly **causal 6.0-second window `[t − 6.0 s, t]`** (1,500 samples at 250 Hz; resampled with anti-aliasing to 300 samples at 50 Hz for deep networks).
- Causal inter-window deltas compare `[t − 6s, t]` against `[t − 12s, t − 6s]` to capture physiological escalation without future leakage.
- Subject-specific baseline calibration uses strictly each recording's own pre-session `<ID>_runUp.csv` recording.

## 4. Modeling, Uncertainty, & Missing-Modality Strategy
- **Loss**: Huber loss (`SmoothL1Loss`, $\beta = 10.0$) provides robustness to occasional outlier self-reports while preserving smooth gradients near zero error.
- **Missing-Modality Robustness**: Both `MM_Gated` (via 30% modality dropout + binary masking) and `xgb_missing_resilient` (via native NaN split routing trained with simulated modality dropout) handle missing/disconnected sensors without fabricating synthetic waveforms.
- **Uncertainty**: MC Dropout ($T = 30$) quantifies epistemic variance; residual scaling on out-of-fold validation predictions calibrates 95% prediction intervals, and missing modalities automatically widen reported uncertainty.

## 5. Limitations & Ethical Considerations
1. **Subjectivity of Pain**: Pain is a subjective biopsychosocial experience. Two individuals with identical autonomic or muscular activation may legitimately report different NRS pain levels (as evidenced by PMPDB's individualized thresholds).
2. **Event-Stamped vs. Dense Continuous Labels**: Model evaluation occurs strictly at the 1,424 verified self-report timestamps; continuous trajectories between events represent sliding-window model estimates rather than dense ground truth.
3. **Domain & Physiotherapy Context**: EMG and Grip carry strong predictive signal during physiotherapy sessions where pain elicits protective muscle guarding; performance in passive or non-physiotherapy clinical settings would require domain-specific validation.
4. **Non-Clinical Use**: This prototype is strictly for **academic and research demonstration** and must never replace patient self-report or clinical judgment.

