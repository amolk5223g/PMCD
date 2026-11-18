"""Evaluation metrics for continuous pain estimation (MAE, RMSE, Pearson r, CCC, R², uncertainty)."""
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score


def concordance_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Lin's Concordance Correlation Coefficient (CCC)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) < 2:
        return 0.0
    mean_t = np.mean(y_true)
    mean_p = np.mean(y_pred)
    var_t = np.var(y_true, ddof=0)
    var_p = np.var(y_pred, ddof=0)
    cov = np.mean((y_true - mean_t) * (y_pred - mean_p))
    denom = var_t + var_p + (mean_t - mean_p) ** 2
    if denom < 1e-12:
        return 0.0
    return float((2.0 * cov) / denom)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    if len(y_true) >= 2 and np.std(y_true) > 1e-12 and np.std(y_pred) > 1e-12:
        r_val, p_val = stats.pearsonr(y_true, y_pred)
        pearson_r = float(r_val)
        pearson_p = float(p_val)
    else:
        pearson_r, pearson_p = 0.0, 1.0
    ccc = concordance_correlation_coefficient(y_true, y_pred)

    out: Dict[str, Any] = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "pearson_r": round(pearson_r, 4),
        "pearson_p": float(f"{pearson_p:.3e}"),
        "ccc": round(ccc, 4),
        "r2": round(r2, 4),
        "n_samples": int(len(y_true)),
    }

    if y_std is not None:
        y_std = np.asarray(y_std, dtype=np.float64)
        lower = y_pred - 1.96 * y_std
        upper = y_pred + 1.96 * y_std
        coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
        mean_width = float(np.mean(upper - lower))
        err = np.abs(y_true - y_pred)
        if np.std(y_std) > 1e-12 and np.std(err) > 1e-12:
            unc_corr = float(stats.spearmanr(y_std, err).statistic)
        else:
            unc_corr = 0.0
        out["uncertainty"] = {
            "mean_std": round(float(np.mean(y_std)), 4),
            "picp_95": round(coverage, 4),
            "mpiw_95": round(mean_width, 4),
            "error_uncertainty_spearman": round(unc_corr, 4),
        }
    return out


def compute_ordinal_metrics(y_true_cls: np.ndarray, y_pred_cls: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true_cls, y_pred_cls)), 4),
        "macro_f1": round(float(f1_score(y_true_cls, y_pred_cls, average="macro", zero_division=0)), 4),
    }


def compute_participant_metrics(
    participant_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    df = pd.DataFrame({
        "participant_id": participant_ids,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_std": y_std if y_std is not None else np.nan,
    })
    results: List[Dict[str, Any]] = []
    for pid, grp in df.groupby("participant_id", sort=True):
        yt = grp["y_true"].values
        yp = grp["y_pred"].values
        m = compute_regression_metrics(yt, yp, grp["y_std"].values if y_std is not None else None)
        m["participant_id"] = str(pid)
        m["true_mean"] = round(float(np.mean(yt)), 2)
        m["pred_mean"] = round(float(np.mean(yp)), 2)
        results.append(m)
    return results

