# Explainable Multimodal AI for Continuous Pain-Intensity Estimation (PMPDB)

An end-to-end, reproducible biomedical signal processing, multimodal machine learning & deep learning, uncertainty quantification, and explainable AI (XAI) research platform trained directly on the local **PMPDB (PainMonit Physiotherapy Database)**, integrated with a **FastAPI** backend and **React + Vite + Tailwind CSS + Recharts** medical monitoring dashboard.

> **Research Disclaimer**: This project is an **academic research prototype only**. It is **not** intended for clinical diagnosis, patient triage, or medical treatment decisions.

---

## 1. Project Overview & Problem Statement

Objective estimation of continuous pain intensity from non-invasive physiological and biomechanical sensors can complement intermittent subjective self-reports (such as the 0–10 Numerical Rating Scale, NRS) in rehabilitation and research settings. However, practical deployment requires:
- **Participant-independent generalization** (strictly avoiding train/test subject leakage).
- **Multimodal fusion** across heterogeneous peripheral sensors with graceful handling of missing or noisy channels (without fabricating fake waveforms).
- **Uncertainty quantification** so users know when estimates are reliable vs. uncertain.
- **Transparent explainability** showing which modalities and physiological features drove each prediction.

---

## 2. Dataset Description & Audit Summary (PMPDB)

- **Detected Raw Location**: `C:\Users\Amol\Downloads\PMCD\PMPDB\raw-data`
- **Participants**: **49 unique participants** (`P01`–`P49`; 41 with 2 sessions, 8 with 1 session).
- **Recordings**: **90 main sessions** (`<ID>.csv`, 2,969,065 rows, 3.30 hours total; duration 69.5 s–233.6 s) + **90 pre-session baseline/calibration recordings** (`<ID>_runUp.csv`, 789,764 rows, 0.88 hours total) + 90 pairs of participant-specific threshold files (`noPainThreshold.txt`, `severePainThreshold.txt`).
- **Format & Sampling**: Semicolon-delimited CSV (`sep=';'`, `decimal=','`), uniformly synchronized at **250.0 Hz** (`max std(dt) = 0.0`). **0.00% missing physiological samples** and 0 flatline channels.
- **Available Modalities (6 Modalities / 8 Synchronized Channels)**:
  - **BVP / PPG**: `Bvp` (Empatica E4), `Bvp_RB` (RespiBAN)
  - **EDA**: `Eda_E4` (Empatica E4), `Eda_RB` (RespiBAN)
  - **Respiration**: `Resp` (RespiBAN)
  - **Skin Temperature**: `Tmp` (Empatica E4, °C)
  - **Surface EMG**: `Emg` (RespiBAN)
  - **Grip Force**: `Grip` (RespiBAN)
- **Explicitly Absent Modalities**: **No ECG** (pulse rate & PRV are derived from BVP/PPG instead) and **No EEG** (no cortical features or EEG channel explanations are claimed).
- **Labels**: 1,424 discrete event-stamped `Pain rates` (1–10 NRS, median spacing 6.98 s, mapped linearly via `NRS × 10` to the **0–100** continuous target) + 1,467 3-class ordinal `Pain labels` (`0` No/Low, `1` Moderate, `2` Severe; 1,424/1,424 exact match against participant threshold files). Forty-three rows with `Pain labels = 0` and `Pain rates = NaN` are excluded from numeric regression rather than fabricated.

See [reports/dataset_analysis_report.md](reports/dataset_analysis_report.md) for the complete 9-section audit.

---

## 3. Project Structure

```text
pain-intensity-estimation/
├── dataset/
│   ├── raw/README.md             # Pointer to C:\Users\Amol\Downloads\PMCD\PMPDB\raw-data
│   ├── processed/                # features.parquet, features.csv, windows_dl.npz, feature_metadata.json
│   └── metadata/                 # dataset_summary.json, recordings_catalog.csv, feature_pain_correlations.csv
├── backend/
│   ├── config.py                 # Paths, verified PMPDB schema, sampling/window parameters
│   ├── main.py                   # FastAPI entrypoint + CORS + static figures mount
│   ├── api/routes.py             # REST API endpoints
│   ├── preprocessing/            # inspect_pmdb.py, signal_pipeline.py
│   ├── models/deep_models.py     # CNN1D, CNN_BiGRU, TCN, MM_Concat, MM_Attention, MM_Gated
│   ├── training/                 # train_baselines.py, train_deep.py
│   ├── evaluation/               # metrics.py, eda_plots.py, generate_eval_report.py
│   ├── explainability/           # xai_engine.py (SHAP + Integrated Gradients)
│   ├── services/model_service.py # Live checkpoint inference, playback, missing-modality detection
│   ├── tests/test_api.py         # Automated end-to-end API integration tests
│   └── requirements.txt
├── frontend/                     # React 18 + Vite 6 + Tailwind CSS 3 + Recharts + Lucide Icons
│   └── src/
│       ├── components/           # PainGauge.jsx, SensorStatusBar.jsx
│       ├── pages/                # Overview, SignalMonitoring, PainPrediction, Explainability, DatasetExplorer, ModelPerformance
│       └── services/api.js
├── checkpoints/                  # Actual trained .pt and .joblib model checkpoints
├── experiments/                  # OOF predictions, JSON evaluation metrics, global XAI summaries
├── reports/
│   ├── dataset_analysis_report.md
│   ├── model_evaluation_report.md
│   ├── architecture_documentation.md
│   ├── research_methodology_report.md
│   └── figures/                  # 12 publication-grade EDA, XAI, and evaluation plots
└── notebooks/
    └── 01_pmdb_eda_and_modeling.ipynb
```

---

## 4. Preprocessing & Feature Methodology

All filtering runs on full 250 Hz recordings prior to windowing to avoid edge transients:
- **BVP (`Bvp`, `Bvp_RB`)**: 4th-order Butterworth bandpass `0.5–8.0 Hz` → systolic peak detection, Pulse Rate (BPM), PRV (`SDNN`, `RMSSD`, `pNN50`), normalized amplitude, cardiac band power ratio.
- **EDA (`Eda_E4`, `Eda_RB`)**: 4th-order lowpass `5.0 Hz` → Tonic (`SCL`, `<0.05 Hz`) vs. Phasic (`SCR`) decomposition, SCR peak count/amplitude, AUC, slope, derivative velocity.
- **Respiration (`Resp`)**: 3rd-order bandpass `0.05–1.0 Hz` → respiratory rate (peak + Welch PSD fallback), inter-breath variability, peak-to-peak amplitude.
- **Skin Temperature (`Tmp`)**: 3rd-order lowpass `0.5 Hz` → mean, slope (°C/s), delta, `runUp` baseline shift.
- **Surface EMG (`Emg`)**: 50 Hz IIR notch + 4th-order bandpass `20–120 Hz` → RMS, MAV, Waveform Length, ZCR, iEMG, spectral mean/median frequency, `runUp` RMS ratio.
- **Grip Force (`Grip`)**: 4th-order lowpass `10.0 Hz` → mean, std, range, slope, `runUp` z-score shifts.
- **Windowing & Zero-Leakage Calibration**: Strictly causal **6.0-second windows `[t − 6.0 s, t]`** (1,500 raw samples; anti-aliased to 300 samples @ 50 Hz for deep models), normalized using each recording's own pre-session `runUp` calibration file, producing **88 engineered features** and `(1424, 8, 300)` waveform tensors.

---

## 5. Experimental Results (Participant-Independent Validation)

All 10 models were evaluated on identical **5-Fold GroupKFold** splits grouped by `participant_id` (plus **49-Fold LOSO** for ML baselines) across all 1,424 verified pain ratings (0–100 scale):

| Model | Family | MAE ↓ | RMSE ↓ | Pearson r ↑ | Lin's CCC ↑ | R² ↑ | LOSO MAE / r |
|---|---|---:|---:|---:|---:|---:|---:|
| **MM_Gated** | Multimodal DL | **17.47** | **21.52** | **0.491** | **0.424** | **0.234** | — |
| **Hybrid_MM_Gated_XGB** | Hybrid Ensemble | 17.57 | 21.54 | 0.486 | 0.400 | 0.232 | — |
| **MM_Attention** | Multimodal DL | 17.74 | 21.56 | 0.485 | 0.406 | 0.231 | — |
| **MM_Concat** | Multimodal DL | 18.22 | 22.23 | 0.436 | 0.353 | 0.182 | — |
| **CNN1D** | Deep Learning | 18.59 | 22.75 | 0.396 | 0.320 | 0.144 | — |
| **CNN_BiGRU** | Deep Learning | 18.68 | 22.82 | 0.393 | 0.321 | 0.138 | — |
| **RandomForest** | Baseline ML | 18.72 | 22.73 | 0.388 | 0.289 | 0.145 | 18.90 / 0.372 |
| **XGBoost** | Baseline ML | 19.23 | 23.71 | 0.352 | 0.297 | 0.070 | 19.25 / 0.365 |
| **SVR** | Baseline ML | 19.27 | 23.40 | 0.354 | 0.288 | 0.094 | 19.48 / 0.332 |
| **TCN** | Deep Learning | 19.30 | 23.53 | 0.322 | 0.244 | 0.084 | — |

- **Calibrated MC Dropout Uncertainty (T=30)**: `MM_Gated` achieves **95.0% PICP** (95% prediction interval coverage), and epistemic uncertainty automatically widens when sensors drop out (from `σ = 22.25` with all 6 modalities to `σ = 24.24` without EMG and `σ = 26.50` with E4-only wearable sensors).
- **Single-Modality Ablation**: `EMG` is the strongest individual physiotherapy pain predictor (`MAE = 18.89, r = 0.390, CCC = 0.338`), followed by `Grip` (`r = 0.207`) and `Resp` (`r = 0.138`); full multimodal deep fusion (`MM_Gated`) improves Pearson $r$ from `0.390` → `0.491` and CCC from `0.338` → `0.424`.

See [reports/model_evaluation_report.md](reports/model_evaluation_report.md), [reports/architecture_documentation.md](reports/architecture_documentation.md), and [reports/research_methodology_report.md](reports/research_methodology_report.md) for full details.

---

## 6. Installation & Reproducible Pipeline Commands

### Backend & ML Pipeline Setup
```bash
cd pain-intensity-estimation
pip install -r backend/requirements.txt

# Phase 1-2: Audit PMPDB & generate Dataset Analysis Report
python -m backend.preprocessing.inspect_pmdb

# Phase 3: Preprocess all 90 recordings -> features.parquet & windows_dl.npz
python -m backend.preprocessing.signal_pipeline

# Phase 4: Generate 7 EDA figures
python -m backend.evaluation.eda_plots

# Phase 5: Train Baseline ML models (5-Fold GroupKFold, 49-Fold LOSO, Modality Ablation)
python -m backend.training.train_baselines

# Phase 6-7: Train Deep Learning & Multimodal Fusion models + Missing-Modality Stress Test
python -m backend.training.train_deep

# Phase 8: Compute Global SHAP + Integrated Gradients XAI
python -m backend.explainability.xai_engine

# Phase 9: Hybrid Ensemble OOF Evaluation & Model Evaluation Report
python -m backend.evaluation.generate_eval_report

# Run Backend Integration Tests
python -m backend.tests.test_api
```

### Start FastAPI Backend
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
API docs available at `http://127.0.0.1:8000/docs`.

### Start React Dashboard Frontend
```bash
cd frontend
npm install
npm run dev
```
Dashboard available at `http://127.0.0.1:5173`.

---

## 7. API Reference

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Health status & research disclaimer |
| `GET` | `/api/models/status` | Loaded checkpoint inventory & available/absent modalities |
| `GET` | `/api/dataset/summary` | Full PMPDB audit statistics |
| `GET` | `/api/dataset/participants` | List of all 49 participants & 90 sessions |
| `GET` | `/api/dataset/recordings/{id}` | Cleaned 6-modality waveforms + observed pain ratings |
| `POST` | `/api/preprocess` | Run biomedical filters & sensor availability detection |
| `POST` | `/api/predict/recording` | Single-window prediction + MC Dropout CI + SHAP + IG |
| `POST` | `/api/predict/timeline` | Continuous sliding-window pain trajectory across a session |
| `POST` | `/api/predict/custom` | Predict on custom sensor arrays with missing-modality detection |
| `GET` | `/api/explain/global` | Global SHAP, Integrated Gradients, and neural gate attributions |
| `POST` | `/api/explain/window` | Local SHAP & Integrated Gradients explanation for a window |
| `GET` | `/api/evaluation/summary` | Complete model comparison, ablation, stress test, & 49-subject metrics |
| `GET` / `DELETE` | `/api/predictions/history` | Prediction log retrieval & clearing |

---

## 8. Limitations & Future Improvements

- **Inter-Subject Pain Subjectivity**: Participant-independent regression cannot observe a patient's personal pain threshold without calibration ratings; adding few-shot subject-specific adaptation (e.g., 2–3 calibration prompts per session) is a natural next step.
- **Sparse Event Labels**: PMPDB provides event-stamped NRS ratings (~7 s median spacing) rather than dense continuous CoVAS streams; sliding-window curves between rating events represent model estimates.
- **Physiotherapy Domain Bias**: Strong EMG and Grip contributions reflect protective muscular guarding during physiotherapy; non-movement clinical pain contexts will exhibit different modality weights.
- **Future Work**: Sequence-to-sequence inter-window transformers across full sessions, conformal prediction intervals for finite-sample coverage guarantees, and multi-dataset transfer validation (e.g., BioVid + PainMonit heat/pressure benchmarks).

