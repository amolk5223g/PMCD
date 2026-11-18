# Model Evaluation Report — Continuous Pain-Intensity Estimation (PMPDB)

## 1. Validation Protocol (Zero Data Leakage)
- **Dataset**: PMPDB (49 participants, 90 main sessions, 1,424 verified numeric NRS pain ratings linearly mapped to 0–100).
- **Splitting**: Strict participant-independent **5-Fold GroupKFold** (all 10 models evaluated on identical participant folds) plus **49-Fold Leave-One-Subject-Out (LOSO)** for baseline ML models.
- **Zero Leakage Normalization**: Baseline relative features and deep-learning channel scaling use strictly each recording's own pre-session `runUp` calibration file; tabular imputers/scalers are fit exclusively inside training folds.
- **Loss Function**: Huber Loss (`SmoothL1Loss`, $\beta = 10.0$) for continuous regression + auxiliary 3-class ordinal Cross-Entropy head for neural models.

## 2. Complete Model Comparison (5-Fold GroupKFold & 49-Fold LOSO)
| Model | Family | MAE ↓ | RMSE ↓ | Pearson r ↑ | Lin's CCC ↑ | R² ↑ | LOSO MAE | LOSO r |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **MM_Gated** | Multimodal DL | 17.47 | 21.52 | 0.491 | 0.424 | 0.234 | — | — |
| **Hybrid_MM_Gated_XGB** | Hybrid Ensemble | 17.57 | 21.54 | 0.486 | 0.400 | 0.232 | — | — |
| **MM_Attention** | Multimodal DL | 17.74 | 21.56 | 0.485 | 0.406 | 0.231 | — | — |
| **MM_Concat** | Multimodal DL | 18.22 | 22.23 | 0.436 | 0.353 | 0.182 | — | — |
| **CNN1D** | Deep Learning | 18.59 | 22.75 | 0.396 | 0.320 | 0.144 | — | — |
| **CNN_BiGRU** | Deep Learning | 18.68 | 22.82 | 0.393 | 0.321 | 0.138 | — | — |
| **RandomForest** | Baseline ML | 18.72 | 22.73 | 0.388 | 0.289 | 0.145 | 18.90 | 0.372 |
| **XGBoost** | Baseline ML | 19.23 | 23.71 | 0.352 | 0.297 | 0.070 | 19.25 | 0.365 |
| **SVR** | Baseline ML | 19.27 | 23.40 | 0.354 | 0.288 | 0.094 | 19.48 | 0.332 |
| **TCN** | Deep Learning | 19.30 | 23.53 | 0.322 | 0.244 | 0.084 | — | — |

## 3. Single-Modality vs. Multimodal Ablation (XGBoost, 5-Fold GroupKFold)
| Modality Subset | Features | MAE ↓ | RMSE ↓ | Pearson r ↑ | Lin's CCC ↑ | R² ↑ |
|---|---:|---:|---:|---:|---:|---:|
| `BVP` | 22 | 22.92 | 28.12 | -0.081 | -0.062 | -0.308 |
| `EDA` | 27 | 22.20 | 26.99 | -0.012 | -0.009 | -0.205 |
| `Resp` | 9 | 20.88 | 25.25 | 0.138 | 0.097 | -0.055 |
| `Tmp` | 8 | 24.51 | 29.96 | -0.252 | -0.190 | -0.485 |
| `EMG` | 10 | 18.89 | 23.10 | 0.390 | 0.338 | 0.117 |
| `Grip` | 10 | 20.90 | 25.42 | 0.207 | 0.171 | -0.069 |
| `Temporal_Only` | 2 | 20.73 | 24.85 | 0.109 | 0.059 | -0.022 |
| `All_Phys_No_Temporal` | 86 | 19.38 | 23.83 | 0.342 | 0.282 | 0.060 |
| `All_Multimodal` | 88 | 19.23 | 23.71 | 0.352 | 0.297 | 0.070 |

## 4. Missing-Modality Resilience Stress Test (OOF `MM_Gated`, Zero Fabrication)
| Scenario | Dropped Modalities | MAE ↓ | RMSE ↓ | Pearson r ↑ | Mean Uncertainty σ |
|---|---|---:|---:|---:|---:|
| `All_6_Modalities` | None | 17.45 | 21.52 | 0.491 | 22.25 |
| `Missing_EDA` | EDA | 17.74 | 21.86 | 0.473 | 22.19 |
| `Missing_BVP` | BVP | 17.45 | 21.57 | 0.492 | 22.38 |
| `Missing_Grip` | Grip | 18.12 | 22.24 | 0.449 | 22.98 |
| `Missing_EMG` | EMG | 20.10 | 24.65 | 0.269 | 24.24 |
| `Missing_Resp_Tmp` | Resp, Tmp | 17.60 | 21.88 | 0.480 | 22.64 |
| `Wearable_E4_Only (BVP+EDA+Tmp)` | Resp, EMG, Grip | 22.34 | 27.22 | 0.073 | 26.50 |

## 5. Uncertainty Quantification (Calibrated MC Dropout, T = 30)
- **MM_Gated 95% Prediction Interval Coverage (PICP)**: **94.9%** (Mean Interval Width: 88.4 pts).
- **Hybrid Ensemble 95% Prediction Interval Coverage (PICP)**: **95.7%** (Mean Interval Width: 89.9 pts).

## 6. Participant-Wise Generalization Summary (49 Participants, Hybrid Ensemble)
- **Best-generalizing participants (lowest MAE)**: P38 (8.7), P15 (9.4), P46 (9.6), P23 (9.8), P28 (11.3).
- **Hardest participants (highest MAE)**: P13 (24.0), P22 (24.0), P06 (25.6), P19 (30.4), P32 (30.6).
- **Median Participant MAE**: 16.80 (IQR: 13.62–21.86).

## 7. Honest Scientific Assessment & Limitations
- Inter-subject pain perception is inherently subjective: identical physiological activation maps to different self-reported NRS scores across individuals (reflected in PMPDB's wide range of individualized `noPainThreshold` [1–5] and `severePainThreshold` [3–9]).
- Without participant-specific calibration ratings from the target session, participant-independent regression achieves meaningful positive correlation and tracks relative pain escalation, while exhibiting realistic ~14–19 point MAE on the 0–100 scale.
- This system is an academic research prototype and **must not** be used for clinical diagnosis or pain medication decisions.