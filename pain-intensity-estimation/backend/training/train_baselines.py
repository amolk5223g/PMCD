"""Phase 5: Train and Evaluate Baseline ML Models (Random Forest, XGBoost, SVR).

Uses strict participant-independent validation (zero data leakage):
1. 5-Fold GroupKFold grouped by participant_id (identical folds as deep learning models)
2. 49-Fold Leave-One-Subject-Out (LOSO) validation across all 49 participants
3. Single-Modality vs. Multimodal ablation study (BVP, EDA, Resp, Tmp, EMG, Grip vs All)
4. Saves trained full-dataset checkpoints + OOF predictions + per-participant metrics
"""
import json
import time
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from backend.config import (
    CHECKPOINTS_DIR,
    EXPERIMENTS_DIR,
    MODALITY_ORDER,
    N_CV_FOLDS,
    PROCESSED_DIR,
    RANDOM_SEED,
)
from backend.evaluation.metrics import compute_participant_metrics, compute_regression_metrics


def _make_models() -> Dict[str, Any]:
    return {
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=250,
                max_depth=12,
                min_samples_leaf=4,
                max_features="sqrt",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )),
        ]),
        "XGBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(
                n_estimators=240,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.75,
                min_child_weight=5,
                reg_alpha=1.0,
                reg_lambda=5.0,
                objective="reg:squarederror",
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbosity=0,
            )),
        ]),
        "SVR": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", SVR(kernel="rbf", C=25.0, epsilon=3.0, gamma="scale")),
        ]),
    }


def _rf_tree_uncertainty(pipe: Pipeline, X: np.ndarray) -> np.ndarray:
    X_imp = pipe.named_steps["imputer"].transform(X)
    rf = pipe.named_steps["model"]
    preds = np.stack([est.predict(X_imp) for est in rf.estimators_], axis=0)
    return np.std(preds, axis=0)


def run_cv_for_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    splitter,
) -> Dict[str, Any]:
    oof_pred = np.zeros(len(y), dtype=np.float64)
    oof_std = np.zeros(len(y), dtype=np.float64) if model_name == "RandomForest" else None

    for tr_idx, va_idx in splitter.split(X, y, groups):
        pipe = _make_models()[model_name]
        pipe.fit(X[tr_idx], y[tr_idx])
        pred = np.clip(pipe.predict(X[va_idx]), 0.0, 100.0)
        oof_pred[va_idx] = pred
        if oof_std is not None:
            oof_std[va_idx] = _rf_tree_uncertainty(pipe, X[va_idx])

    metrics = compute_regression_metrics(y, oof_pred, y_std=oof_std)
    return {
        "metrics": metrics,
        "oof_pred": oof_pred,
        "oof_std": oof_std,
    }


def train_and_evaluate_baselines() -> Dict[str, Any]:
    t0 = time.time()
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    with open(PROCESSED_DIR / "feature_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_cols: List[str] = meta["feature_columns"]
    feat_mod_map: Dict[str, str] = meta["feature_modality_map"]

    X = df[feature_cols].values.astype(np.float64)
    y = df["pain_score_0_100"].values.astype(np.float64)
    groups = df["participant_id"].values

    gkf = GroupKFold(n_splits=N_CV_FOLDS)
    logo = LeaveOneGroupOut()

    results: Dict[str, Any] = {
        "validation_strategy": f"{N_CV_FOLDS}-Fold GroupKFold & 49-Fold LOSO (participant-independent)",
        "n_samples": int(len(y)),
        "n_participants": int(len(np.unique(groups))),
        "n_features": len(feature_cols),
        "groupkfold": {},
        "loso": {},
        "modality_ablation": {},
        "participant_metrics": {},
    }

    oof_df = df[["window_id", "recording_id", "participant_id", "timestamp_sec", "pain_score_0_100", "pain_label"]].copy()

    print("=== Phase 5: Training Baseline ML Models (5-Fold GroupKFold & 49-Fold LOSO) ===")
    for name in ["RandomForest", "XGBoost", "SVR"]:
        t_m = time.time()
        gkf_res = run_cv_for_model(name, X, y, groups, gkf)
        loso_res = run_cv_for_model(name, X, y, groups, logo)

        results["groupkfold"][name] = gkf_res["metrics"]
        results["loso"][name] = loso_res["metrics"]
        results["participant_metrics"][name] = compute_participant_metrics(
            groups, y, loso_res["oof_pred"], loso_res["oof_std"]
        )

        oof_df[f"{name}_gkf_pred"] = gkf_res["oof_pred"]
        oof_df[f"{name}_loso_pred"] = loso_res["oof_pred"]
        if loso_res["oof_std"] is not None:
            oof_df[f"{name}_loso_std"] = loso_res["oof_std"]

        # Fit final production checkpoint on all 49 participants
        final_pipe = _make_models()[name]
        final_pipe.fit(X, y)
        ckpt_path = CHECKPOINTS_DIR / f"baseline_{name.lower()}.joblib"
        joblib.dump(
            {
                "pipeline": final_pipe,
                "feature_columns": feature_cols,
                "feature_modality_map": feat_mod_map,
                "model_name": name,
                "cv_metrics_gkf": gkf_res["metrics"],
                "cv_metrics_loso": loso_res["metrics"],
            },
            ckpt_path,
        )
        print(
            f"  [{name:12s}] GKF5: MAE={gkf_res['metrics']['mae']:.2f}, RMSE={gkf_res['metrics']['rmse']:.2f}, "
            f"r={gkf_res['metrics']['pearson_r']:.3f}, CCC={gkf_res['metrics']['ccc']:.3f}, R2={gkf_res['metrics']['r2']:.3f} | "
            f"LOSO: MAE={loso_res['metrics']['mae']:.2f}, r={loso_res['metrics']['pearson_r']:.3f}, CCC={loso_res['metrics']['ccc']:.3f} "
            f"({time.time() - t_m:.1f}s)"
        )

    # Single-Modality vs. Multimodal Ablation using XGBoost (5-Fold GroupKFold)
    print("\n=== Single-Modality vs. Multimodal Ablation (XGBoost, 5-Fold GroupKFold) ===")
    ablation_groups = {mod: [c for c in feature_cols if feat_mod_map[c] == mod] for mod in MODALITY_ORDER}
    ablation_groups["Temporal_Only"] = [c for c in feature_cols if feat_mod_map[c] == "Temporal"]
    ablation_groups["All_Phys_No_Temporal"] = [c for c in feature_cols if feat_mod_map[c] != "Temporal"]
    ablation_groups["All_Multimodal"] = feature_cols

    for subset_name, cols in ablation_groups.items():
        if not cols:
            continue
        X_sub = df[cols].values.astype(np.float64)
        ab_res = run_cv_for_model("XGBoost", X_sub, y, groups, gkf)
        m = ab_res["metrics"]
        m["n_features"] = len(cols)
        results["modality_ablation"][subset_name] = m
        print(
            f"  [{subset_name:22s}] ({len(cols):2d} feats): MAE={m['mae']:.2f}, RMSE={m['rmse']:.2f}, "
            f"r={m['pearson_r']:.3f}, CCC={m['ccc']:.3f}, R2={m['r2']:.3f}"
        )

    # Also train a modality-dropout-resilient XGBoost checkpoint (handles NaN features natively when a sensor is missing!)
    print("\nTraining missing-modality-resilient XGBoost (native NaN sparsity + simulated modality dropout)...")
    rng = np.random.default_rng(RANDOM_SEED)
    X_aug_list = [X]
    y_aug_list = [y]
    # Add 2 copies with random 1-2 modalities masked to NaN so XGBoost learns optimal NaN split directions
    mod_col_indices = {
        mod: [i for i, c in enumerate(feature_cols) if feat_mod_map[c] == mod]
        for mod in MODALITY_ORDER
    }
    for _ in range(2):
        X_drop = X.copy()
        for row_i in range(len(X_drop)):
            n_drop = int(rng.integers(1, 3))
            drop_mods = rng.choice(MODALITY_ORDER, size=n_drop, replace=False)
            for dm in drop_mods:
                X_drop[row_i, mod_col_indices[dm]] = np.nan
        X_aug_list.append(X_drop)
        y_aug_list.append(y)
    X_aug = np.vstack(X_aug_list)
    y_aug = np.concatenate(y_aug_list)

    xgb_missing_resilient = XGBRegressor(
        n_estimators=260,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.75,
        min_child_weight=5,
        reg_alpha=1.0,
        reg_lambda=5.0,
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_missing_resilient.fit(X_aug, y_aug)
    joblib.dump(
        {
            "model": xgb_missing_resilient,
            "feature_columns": feature_cols,
            "feature_modality_map": feat_mod_map,
            "note": "Trained with modality-dropout NaN masking so missing sensors pass NaN natively without fabrication.",
        },
        CHECKPOINTS_DIR / "xgb_missing_resilient.joblib",
    )

    oof_df.to_parquet(EXPERIMENTS_DIR / "baseline_oof_predictions.parquet", index=False)
    with open(EXPERIMENTS_DIR / "baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Baseline training finished in {time.time() - t0:.1f}s. Saved to {EXPERIMENTS_DIR / 'baseline_results.json'}")
    return results


if __name__ == "__main__":
    train_and_evaluate_baselines()

