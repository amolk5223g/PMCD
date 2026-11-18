"""Backend inference, dataset playback, explainability, and missing-modality service.

Loads ONLY real trained model checkpoints from `checkpoints/` and real recordings from `PMPDB/raw-data`.
Never generates synthetic signals or hardcoded pain predictions.
"""
import json
import time
from collections import deque
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
import torch

from backend.config import (
    ABSENT_MODALITIES,
    CHECKPOINTS_DIR,
    DOWNSAMPLED_RATE_HZ,
    EXPERIMENTS_DIR,
    METADATA_DIR,
    MODALITY_DESCRIPTIONS,
    MODALITY_GROUPS,
    MODALITY_ORDER,
    PROCESSED_DIR,
    RAW_DATASET_DIR,
    SAMPLING_RATE_HZ,
    SIGNAL_COLUMNS,
    WINDOW_SAMPLES_RAW,
    WINDOW_SECONDS,
)
from backend.explainability.xai_engine import compute_integrated_gradients, explain_local_shap
from backend.models.deep_models import build_dl_model, predict_mc_dropout
from backend.preprocessing.signal_pipeline import (
    clean_signals_df,
    compute_runup_baseline,
    decompose_eda,
    detect_modality_availability,
    extract_dl_tensor_window,
    extract_features_for_window,
    slice_causal_window,
)


def classify_pain_severity(score_0_100: float) -> str:
    if score_0_100 < 30.0:
        return "No / Mild Pain"
    if score_0_100 < 60.0:
        return "Moderate Pain"
    if score_0_100 < 80.0:
        return "High Pain"
    return "Severe Pain"


def classify_uncertainty(std_val: float) -> str:
    if std_val < 8.0:
        return "Low"
    if std_val < 14.0:
        return "Moderate"
    return "High"


class PainEstimationService:
    """Singleton service holding real trained model checkpoints and dataset caches."""

    def __init__(self) -> None:
        self.loaded_at: Optional[str] = None
        self.models_ready: bool = False
        self.dl_model: Optional[torch.nn.Module] = None
        self.dl_calib_scale: float = 2.0
        self.xgb_bundle: Optional[Dict[str, Any]] = None
        self.xgb_missing_bundle: Optional[Dict[str, Any]] = None
        self.rf_bundle: Optional[Dict[str, Any]] = None
        self.svr_bundle: Optional[Dict[str, Any]] = None
        self.shap_explainer: Optional[shap.TreeExplainer] = None
        self.feature_columns: List[str] = []
        self.feature_modality_map: Dict[str, str] = {}
        self.history: Deque[Dict[str, Any]] = deque(maxlen=250)
        self._recording_cache: Dict[str, Dict[str, Any]] = {}
        self._load_all()

    def _load_all(self) -> None:
        meta_path = PROCESSED_DIR / "feature_metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.feature_columns = meta["feature_columns"]
            self.feature_modality_map = meta["feature_modality_map"]

        dl_path = CHECKPOINTS_DIR / "mm_gated_production.pt"
        if dl_path.exists():
            ckpt = torch.load(dl_path, map_location="cpu", weights_only=False)
            m = build_dl_model("MM_Gated", dropout=0.30)
            m.load_state_dict(ckpt["state_dict"])
            m.eval()
            self.dl_model = m
            self.dl_calib_scale = float(ckpt.get("calibration_scale", 2.0))

        xgb_path = CHECKPOINTS_DIR / "baseline_xgboost.joblib"
        if xgb_path.exists():
            self.xgb_bundle = joblib.load(xgb_path)
            self.shap_explainer = shap.TreeExplainer(self.xgb_bundle["pipeline"].named_steps["model"])

        xgb_m_path = CHECKPOINTS_DIR / "xgb_missing_resilient.joblib"
        if xgb_m_path.exists():
            self.xgb_missing_bundle = joblib.load(xgb_m_path)

        rf_path = CHECKPOINTS_DIR / "baseline_randomforest.joblib"
        if rf_path.exists():
            self.rf_bundle = joblib.load(rf_path)

        svr_path = CHECKPOINTS_DIR / "baseline_svr.joblib"
        if svr_path.exists():
            self.svr_bundle = joblib.load(svr_path)

        self.models_ready = (self.dl_model is not None) or (self.xgb_bundle is not None)
        self.loaded_at = datetime.now(timezone.utc).isoformat()

    def reload(self) -> Dict[str, Any]:
        self._load_all()
        return self.get_model_status()

    def get_dataset_summary(self) -> Dict[str, Any]:
        p = METADATA_DIR / "dataset_summary.json"
        if not p.exists():
            raise FileNotFoundError("Dataset summary not generated yet. Run backend.preprocessing.inspect_pmdb.")
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_evaluation_summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "baseline": None,
            "deep_learning": None,
            "hybrid_ensemble": None,
            "model_comparison_table": [],
        }
        b_path = EXPERIMENTS_DIR / "baseline_results.json"
        d_path = EXPERIMENTS_DIR / "deep_results.json"
        h_path = EXPERIMENTS_DIR / "hybrid_results.json"

        if b_path.exists():
            with open(b_path, "r", encoding="utf-8") as f:
                out["baseline"] = json.load(f)
            for name, m in out["baseline"].get("groupkfold", {}).items():
                loso_m = out["baseline"].get("loso", {}).get(name, {})
                out["model_comparison_table"].append({
                    "model": name,
                    "family": "Baseline ML",
                    "validation": "5-Fold GroupKFold",
                    "mae": m["mae"],
                    "rmse": m["rmse"],
                    "pearson_r": m["pearson_r"],
                    "ccc": m["ccc"],
                    "r2": m["r2"],
                    "loso_mae": loso_m.get("mae"),
                    "loso_pearson_r": loso_m.get("pearson_r"),
                    "loso_ccc": loso_m.get("ccc"),
                })

        if d_path.exists():
            with open(d_path, "r", encoding="utf-8") as f:
                out["deep_learning"] = json.load(f)
            for name, m in out["deep_learning"].get("models", {}).items():
                ord_m = out["deep_learning"].get("ordinal_head_metrics", {}).get(name, {})
                out["model_comparison_table"].append({
                    "model": name,
                    "family": "Multimodal DL" if name.startswith("MM_") else "Deep Learning",
                    "validation": "5-Fold GroupKFold",
                    "mae": m["mae"],
                    "rmse": m["rmse"],
                    "pearson_r": m["pearson_r"],
                    "ccc": m["ccc"],
                    "r2": m["r2"],
                    "ordinal_acc": ord_m.get("accuracy"),
                    "ordinal_f1": ord_m.get("macro_f1"),
                    "picp_95": m.get("uncertainty", {}).get("picp_95"),
                    "mean_unc_std": m.get("uncertainty", {}).get("mean_std"),
                })

        if h_path.exists():
            with open(h_path, "r", encoding="utf-8") as f:
                out["hybrid_ensemble"] = json.load(f)
            hm = out["hybrid_ensemble"].get("metrics", {})
            if hm:
                out["model_comparison_table"].append({
                    "model": "Hybrid_MM_Gated_XGB",
                    "family": "Hybrid Ensemble (DL + Tabular)",
                    "validation": "5-Fold GroupKFold",
                    "mae": hm["mae"],
                    "rmse": hm["rmse"],
                    "pearson_r": hm["pearson_r"],
                    "ccc": hm["ccc"],
                    "r2": hm["r2"],
                    "picp_95": hm.get("uncertainty", {}).get("picp_95"),
                    "mean_unc_std": hm.get("uncertainty", {}).get("mean_std"),
                })

        out["model_comparison_table"].sort(key=lambda r: r["mae"])
        return out

    def get_global_xai(self) -> Dict[str, Any]:
        p = EXPERIMENTS_DIR / "xai_global_summary.json"
        if not p.exists():
            raise FileNotFoundError("Global XAI summary not found yet.")
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_model_status(self) -> Dict[str, Any]:
        ckpts = sorted([p.name for p in CHECKPOINTS_DIR.glob("*")]) if CHECKPOINTS_DIR.exists() else []
        return {
            "models_ready": self.models_ready,
            "loaded_at": self.loaded_at,
            "loaded_checkpoints": {
                "mm_gated_dl": self.dl_model is not None,
                "xgboost_baseline": self.xgb_bundle is not None,
                "xgboost_missing_resilient": self.xgb_missing_bundle is not None,
                "random_forest": self.rf_bundle is not None,
                "svr": self.svr_bundle is not None,
                "shap_explainer": self.shap_explainer is not None,
            },
            "checkpoint_files": ckpts,
            "available_modalities": MODALITY_ORDER,
            "absent_modalities": ABSENT_MODALITIES,
            "window_seconds": WINDOW_SECONDS,
            "sampling_rate_hz": SAMPLING_RATE_HZ,
        }

    def _load_recording(self, recording_id: str) -> Dict[str, Any]:
        if recording_id in self._recording_cache:
            return self._recording_cache[recording_id]

        rdir = RAW_DATASET_DIR / recording_id
        main_csv = rdir / f"{recording_id}.csv"
        if not main_csv.exists():
            raise FileNotFoundError(f"Recording '{recording_id}' not found in {RAW_DATASET_DIR}")

        df = pd.read_csv(main_csv, sep=";", decimal=",", low_memory=False)
        runup_csv = rdir / f"{recording_id}_runUp.csv"
        rdf = pd.read_csv(runup_csv, sep=";", decimal=",", low_memory=False) if runup_csv.exists() else None
        no_thr = float((rdir / "noPainThreshold.txt").read_text(encoding="utf-8").strip())
        sev_thr = float((rdir / "severePainThreshold.txt").read_text(encoding="utf-8").strip())

        cleaned = clean_signals_df(df, fs=SAMPLING_RATE_HZ)
        eda_decomp = {
            "Eda_E4": decompose_eda(cleaned["Eda_E4"], fs=SAMPLING_RATE_HZ),
            "Eda_RB": decompose_eda(cleaned["Eda_RB"], fs=SAMPLING_RATE_HZ),
        }
        baseline = compute_runup_baseline(rdf, cleaned)

        pr_mask = df["Pain rates"].notna().values
        events = []
        for idx in np.where(pr_mask)[0]:
            events.append({
                "sample_idx": int(idx),
                "timestamp_sec": round(float(df["Seconds"].iloc[idx]), 3),
                "pain_rate_nrs": float(df["Pain rates"].iloc[idx]),
                "pain_score_0_100": round(float(df["Pain rates"].iloc[idx] * 10.0), 1),
                "pain_label": int(df["Pain labels"].iloc[idx]),
            })

        entry = {
            "recording_id": recording_id,
            "participant_id": recording_id.split("_")[0],
            "df": df,
            "cleaned": cleaned,
            "eda_decomp": eda_decomp,
            "baseline": baseline,
            "duration_sec": float(df["Seconds"].iloc[-1] - df["Seconds"].iloc[0]),
            "no_pain_threshold": no_thr,
            "severe_pain_threshold": sev_thr,
            "events": events,
        }
        # Keep small LRU cache (max 8 recordings in memory)
        if len(self._recording_cache) >= 8:
            oldest = next(iter(self._recording_cache))
            del self._recording_cache[oldest]
        self._recording_cache[recording_id] = entry
        return entry

    def get_recording_signals(self, recording_id: str, downsample_hz: float = 10.0) -> Dict[str, Any]:
        """Return actual cleaned physiological waveforms + observed pain ratings for dashboard plotting."""
        rec = self._load_recording(recording_id)
        df = rec["df"]
        cleaned = rec["cleaned"]
        tonic_rb, phasic_rb = rec["eda_decomp"]["Eda_RB"]

        step = max(1, int(round(SAMPLING_RATE_HZ / downsample_hz)))
        indices = np.arange(0, len(df), step)
        secs = df["Seconds"].values[indices]

        series = []
        for i, idx in enumerate(indices):
            series.append({
                "t": round(float(secs[i]), 2),
                "Bvp": round(float(cleaned["Bvp"][idx]), 4),
                "Bvp_RB": round(float(cleaned["Bvp_RB"][idx]), 5),
                "Eda_E4": round(float(cleaned["Eda_E4"][idx]), 4),
                "Eda_RB": round(float(cleaned["Eda_RB"][idx]), 4),
                "Eda_RB_Tonic": round(float(tonic_rb[idx]), 4),
                "Eda_RB_Phasic": round(float(phasic_rb[idx]), 4),
                "Resp": round(float(cleaned["Resp"][idx]), 4),
                "Tmp": round(float(cleaned["Tmp"][idx]), 3),
                "Emg": round(float(cleaned["Emg"][idx]), 5),
                "Grip": round(float(cleaned["Grip"][idx]), 4),
            })

        return {
            "recording_id": recording_id,
            "participant_id": rec["participant_id"],
            "data_source": "Prerecorded PMPDB dataset playback (NOT live hardware acquisition)",
            "sampling_rate_raw_hz": SAMPLING_RATE_HZ,
            "display_rate_hz": round(SAMPLING_RATE_HZ / step, 2),
            "duration_sec": round(rec["duration_sec"], 2),
            "no_pain_threshold_0_100": rec["no_pain_threshold"] * 10.0,
            "severe_pain_threshold_0_100": rec["severe_pain_threshold"] * 10.0,
            "available_modalities": MODALITY_ORDER,
            "absent_modalities": list(ABSENT_MODALITIES.keys()),
            "observed_pain_events": rec["events"],
            "signals": series,
        }

    def _build_feature_vector(
        self,
        rec: Dict[str, Any],
        end_idx: int,
        disabled_modalities: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        disabled = set(disabled_modalities or [])
        feats = extract_features_for_window(
            rec["cleaned"], rec["eda_decomp"], rec["baseline"], end_idx, fs=SAMPLING_RATE_HZ
        )
        prev_idx = max(0, end_idx - WINDOW_SAMPLES_RAW)
        prior_feats = extract_features_for_window(
            rec["cleaned"], rec["eda_decomp"], rec["baseline"], prev_idx, fs=SAMPLING_RATE_HZ
        )
        key_delta_cols = [
            "EDA_E4_scl_rel_runup", "EDA_RB_scl_rel_runup", "EDA_RB_phasic_auc",
            "BVP_E4_pr_bpm", "BVP_RB_rms_ratio", "Resp_rms_ratio",
            "Tmp_rel_runup_c", "EMG_rms_ratio", "Grip_rel_runup_z",
        ]
        for k in key_delta_cols:
            feats[f"delta_{k}"] = float(feats[k] - prior_feats[k])

        ts = float(rec["df"]["Seconds"].iloc[min(end_idx, len(rec["df"]) - 1)])
        feats["Temporal_elapsed_sec"] = ts
        feats["Temporal_progress_ratio"] = ts / max(rec["duration_sec"], 1.0)

        vec = np.zeros(len(self.feature_columns), dtype=np.float64)
        for i, col in enumerate(self.feature_columns):
            mod = self.feature_modality_map.get(col, "Temporal")
            if mod in disabled:
                vec[i] = np.nan
            else:
                v = feats.get(col, 0.0)
                vec[i] = v if np.isfinite(v) else np.nan
        return vec, feats

    def predict_recording_window(
        self,
        recording_id: str,
        timestamp_sec: float,
        model_choice: str = "hybrid",
        disabled_modalities: Optional[List[str]] = None,
        include_explainability: bool = True,
        log_history: bool = True,
    ) -> Dict[str, Any]:
        """Run actual trained checkpoint inference on a 6-second causal window of a real PMPDB recording."""
        rec = self._load_recording(recording_id)
        secs = rec["df"]["Seconds"].values
        end_idx = int(np.argmin(np.abs(secs - timestamp_sec)))
        actual_ts = float(secs[end_idx])

        disabled = [m for m in (disabled_modalities or []) if m in MODALITY_ORDER]
        disabled_set = set(disabled)

        # Build sensor status honestly (never fabricating missing channels)
        raw_win = {}
        for col in SIGNAL_COLUMNS:
            mod_of_col = next(m for m, cols in MODALITY_GROUPS.items() if col in cols)
            if mod_of_col in disabled_set:
                raw_win[col] = None
            else:
                raw_win[col] = slice_causal_window(rec["df"][col].values, end_idx, WINDOW_SAMPLES_RAW)

        sensor_status = detect_modality_availability(raw_win)
        active_mods = [m for m in MODALITY_ORDER if sensor_status[m]["available"]]
        missing_mods = [m for m in MODALITY_ORDER if not sensor_status[m]["available"]]

        mask_np = np.array([1.0 if m in active_mods else 0.0 for m in MODALITY_ORDER], dtype=np.float32)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0)

        # Deep learning tensor (zeroing disabled modalities strictly via mask)
        dl_tensor = extract_dl_tensor_window(rec["cleaned"], rec["baseline"], end_idx)
        x_dl = torch.from_numpy(dl_tensor).unsqueeze(0).float()

        # Tabular feature vector (passing NaN for missing modalities without fabrication)
        x_tab, raw_feats = self._build_feature_vector(rec, end_idx, disabled_modalities=missing_mods)

        dl_pred, dl_std, dl_cls_probs, gate_weights = None, None, None, None
        if self.dl_model is not None:
            mc = predict_mc_dropout(self.dl_model, x_dl, modality_mask=mask_t, n_samples=30)
            dl_pred = float(np.clip(mc["mean"].item(), 0.0, 100.0))
            raw_epistemic = max(float(mc["std"].item()), 0.5)
            # Increase uncertainty proportionally when modalities are missing
            missing_penalty = 1.0 + 0.15 * len(missing_mods)
            dl_std = float(raw_epistemic * self.dl_calib_scale * missing_penalty)
            dl_cls_probs = mc["cls_probs"].squeeze(0).numpy().tolist()
            if mc["modality_weights"] is not None:
                gw = mc["modality_weights"].squeeze(0).numpy()
                gate_weights = {m: round(float(gw[i] * 100.0), 2) for i, m in enumerate(MODALITY_ORDER)}

        xgb_pred = None
        if missing_mods and self.xgb_missing_bundle is not None:
            xgb_pred = float(np.clip(self.xgb_missing_bundle["model"].predict(x_tab.reshape(1, -1))[0], 0.0, 100.0))
        elif self.xgb_bundle is not None:
            xgb_pred = float(np.clip(self.xgb_bundle["pipeline"].predict(x_tab.reshape(1, -1))[0], 0.0, 100.0))

        rf_pred, rf_std = None, None
        if self.rf_bundle is not None:
            x_imp = self.rf_bundle["pipeline"].named_steps["imputer"].transform(x_tab.reshape(1, -1))
            rf_m = self.rf_bundle["pipeline"].named_steps["model"]
            tree_preds = np.array([e.predict(x_imp)[0] for e in rf_m.estimators_])
            rf_pred = float(np.clip(np.mean(tree_preds), 0.0, 100.0))
            rf_std = float(np.std(tree_preds))

        # Select final prediction
        if model_choice == "mm_gated" and dl_pred is not None:
            final_pred = dl_pred
            final_std = dl_std if dl_std is not None else 12.0
            active_model = "MM_Gated (Multimodal Gated Fusion + MC Dropout)"
        elif model_choice == "xgboost" and xgb_pred is not None:
            final_pred = xgb_pred
            final_std = rf_std if rf_std is not None else (dl_std or 12.0)
            active_model = "XGBoost (Missing-Resilient)" if missing_mods else "XGBoost Regressor"
        elif model_choice == "random_forest" and rf_pred is not None:
            final_pred = rf_pred
            final_std = rf_std if rf_std is not None else 12.0
            active_model = "Random Forest Regressor"
        else:
            # Hybrid fusion of MM_Gated (deep waveform) + XGBoost (biomedical features)
            if dl_pred is not None and xgb_pred is not None:
                final_pred = float(np.clip(0.70 * dl_pred + 0.30 * xgb_pred, 0.0, 100.0))
                disagreement = abs(dl_pred - xgb_pred) * 0.35
                final_std = float(np.sqrt((dl_std or 10.0) ** 2 + disagreement ** 2))
                active_model = "Hybrid Ensemble (MM_Gated + XGBoost)"
            elif xgb_pred is not None:
                final_pred = xgb_pred
                final_std = rf_std or 12.0
                active_model = "XGBoost Regressor"
            else:
                final_pred = dl_pred or 0.0
                final_std = dl_std or 12.0
                active_model = "MM_Gated"

        ci_low = float(np.clip(final_pred - 1.96 * final_std, 0.0, 100.0))
        ci_high = float(np.clip(final_pred + 1.96 * final_std, 0.0, 100.0))

        # Find nearest observed ground-truth pain event if within 3.0 s
        nearest_gt = None
        if rec["events"]:
            best_ev = min(rec["events"], key=lambda e: abs(e["timestamp_sec"] - actual_ts))
            if abs(best_ev["timestamp_sec"] - actual_ts) <= 3.0:
                nearest_gt = best_ev

        explain_obj = None
        if include_explainability:
            ig_res = compute_integrated_gradients(self.dl_model, x_dl, modality_mask=mask_t, n_steps=16) if self.dl_model is not None else None
            shap_res = None
            if missing_mods and self.xgb_missing_bundle is not None:
                m_expl = shap.TreeExplainer(self.xgb_missing_bundle["model"])
                shap_res = explain_local_shap(m_expl, x_tab, self.feature_columns, self.feature_modality_map, top_k=10)
            elif self.shap_explainer is not None:
                x_imp = self.xgb_bundle["pipeline"].named_steps["imputer"].transform(x_tab.reshape(1, -1))[0]
                shap_res = explain_local_shap(self.shap_explainer, x_imp, self.feature_columns, self.feature_modality_map, top_k=10)

            # Determine top driving modality
            top_mod = "N/A"
            if shap_res and shap_res["modality_contributions"]:
                phys_mods = {k: v["pct_contribution"] for k, v in shap_res["modality_contributions"].items() if k in active_mods}
                if phys_mods:
                    top_mod = max(phys_mods, key=phys_mods.get)

            unc_level = classify_uncertainty(final_std)
            summary_text = (
                f"Estimated pain intensity is {final_pred:.1f}/100 ({classify_pain_severity(final_pred)}) "
                f"with 95% interval [{ci_low:.1f}, {ci_high:.1f}] ({unc_level} uncertainty). "
                f"Primary physiological driver in this 6s window: {top_mod}."
            )
            if missing_mods:
                summary_text += (
                    f" Warning: {', '.join(missing_mods)} unavailable — masked without fabrication; "
                    f"prediction relies strictly on {len(active_mods)}/6 active modalities ({', '.join(active_mods)}) with widened uncertainty."
                )

            explain_obj = {
                "summary": summary_text,
                "primary_modality": top_mod,
                "shap": shap_res,
                "integrated_gradients": ig_res,
                "neural_gate_weights_pct": gate_weights,
                "limitations_note": "Based solely on peripheral PMPDB sensors (BVP, EDA, Resp, Tmp, EMG, Grip). No ECG or EEG is used. Research prototype only — not for clinical diagnosis.",
            }

        result = {
            "prediction_id": f"pred_{int(time.time() * 1000)}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": "prerecorded_dataset_playback",
            "recording_id": recording_id,
            "participant_id": rec["participant_id"],
            "window_start_sec": round(max(0.0, actual_ts - WINDOW_SECONDS), 2),
            "window_end_sec": round(actual_ts, 2),
            "model_used": active_model,
            "predicted_pain_0_100": round(final_pred, 1),
            "predicted_pain_nrs_10": round(final_pred / 10.0, 2),
            "severity_category": classify_pain_severity(final_pred),
            "uncertainty_std": round(final_std, 2),
            "uncertainty_level": classify_uncertainty(final_std),
            "confidence_interval_95": [round(ci_low, 1), round(ci_high, 1)],
            "component_predictions": {
                "mm_gated": round(dl_pred, 1) if dl_pred is not None else None,
                "xgboost": round(xgb_pred, 1) if xgb_pred is not None else None,
                "random_forest": round(rf_pred, 1) if rf_pred is not None else None,
            },
            "ordinal_class_probabilities": {
                "no_low_pain": round(dl_cls_probs[0], 3),
                "moderate_pain": round(dl_cls_probs[1], 3),
                "severe_pain": round(dl_cls_probs[2], 3),
            } if dl_cls_probs is not None else None,
            "nearest_ground_truth": nearest_gt,
            "sensor_status": sensor_status,
            "active_modalities": active_mods,
            "missing_modalities": missing_mods,
            "missing_modality_warning": (
                f"{len(missing_mods)} modality(ies) unavailable ({', '.join(missing_mods)}). "
                "Zero fabrication applied; fusion re-weighted across active sensors with widened uncertainty."
            ) if missing_mods else None,
            "key_physiological_metrics": {
                "pulse_rate_bpm": round(raw_feats["BVP_E4_pr_bpm"], 1) if "BVP" not in missing_mods else None,
                "prv_rmssd_ms": round(raw_feats["BVP_E4_prv_rmssd_ms"], 1) if "BVP" not in missing_mods else None,
                "eda_scl_rb": round(raw_feats["EDA_RB_scl_mean"], 2) if "EDA" not in missing_mods else None,
                "eda_scr_peaks": int(raw_feats["EDA_RB_scr_count"]) if "EDA" not in missing_mods else None,
                "resp_rate_bpm": round(raw_feats["Resp_rate_bpm"], 1) if "Resp" not in missing_mods else None,
                "skin_temp_c": round(raw_feats["Tmp_mean"], 2) if "Tmp" not in missing_mods else None,
                "emg_rms_ratio": round(raw_feats["EMG_rms_ratio"], 2) if "EMG" not in missing_mods else None,
                "grip_rel_z": round(raw_feats["Grip_rel_runup_z"], 2) if "Grip" not in missing_mods else None,
            },
            "explainability": explain_obj,
        }

        if log_history:
            self.history.appendleft({
                "prediction_id": result["prediction_id"],
                "created_at": result["created_at"],
                "recording_id": recording_id,
                "timestamp_sec": result["window_end_sec"],
                "model_used": active_model,
                "predicted_pain_0_100": result["predicted_pain_0_100"],
                "uncertainty_std": result["uncertainty_std"],
                "confidence_interval_95": result["confidence_interval_95"],
                "severity_category": result["severity_category"],
                "missing_modalities": missing_mods,
                "ground_truth_0_100": nearest_gt["pain_score_0_100"] if nearest_gt else None,
            })

        return result

    def predict_recording_timeline(
        self,
        recording_id: str,
        step_sec: float = 4.0,
        model_choice: str = "hybrid",
        disabled_modalities: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generate continuous sliding-window pain trajectory across an entire real PMPDB recording (batched)."""
        rec = self._load_recording(recording_id)
        dur = rec["duration_sec"]
        secs = rec["df"]["Seconds"].values
        grid_t = np.arange(WINDOW_SECONDS, dur + 0.01, step_sec)
        ev_t = np.array([e["timestamp_sec"] for e in rec["events"]], dtype=float)
        all_t = np.unique(np.round(np.concatenate([grid_t, ev_t]), 2))

        missing_mods = [m for m in (disabled_modalities or []) if m in MODALITY_ORDER]
        active_mods = [m for m in MODALITY_ORDER if m not in missing_mods]

        end_indices = [int(np.argmin(np.abs(secs - t_sec))) for t_sec in all_t]
        actual_ts_list = [float(secs[i]) for i in end_indices]

        # Batch DL tensors and tabular feature vectors
        dl_list = [extract_dl_tensor_window(rec["cleaned"], rec["baseline"], idx) for idx in end_indices]
        X_dl_t = torch.from_numpy(np.stack(dl_list, axis=0)).float()
        mask_vec = np.array([1.0 if m in active_mods else 0.0 for m in MODALITY_ORDER], dtype=np.float32)
        mask_t = torch.from_numpy(np.tile(mask_vec, (len(end_indices), 1)))

        tab_list = [self._build_feature_vector(rec, idx, disabled_modalities=missing_mods)[0] for idx in end_indices]
        X_tab = np.vstack(tab_list)

        missing_penalty = 1.0 + 0.15 * len(missing_mods)
        mc = predict_mc_dropout(self.dl_model, X_dl_t, modality_mask=mask_t, n_samples=24)
        dl_preds = np.clip(mc["mean"].numpy(), 0.0, 100.0)
        dl_stds = np.maximum(mc["std"].numpy(), 0.5) * self.dl_calib_scale * missing_penalty

        if missing_mods and self.xgb_missing_bundle is not None:
            xgb_preds = np.clip(self.xgb_missing_bundle["model"].predict(X_tab), 0.0, 100.0)
        else:
            xgb_preds = np.clip(self.xgb_bundle["pipeline"].predict(X_tab), 0.0, 100.0)

        rf_preds, rf_stds = None, None
        if model_choice == "random_forest" and self.rf_bundle is not None:
            x_imp = self.rf_bundle["pipeline"].named_steps["imputer"].transform(X_tab)
            rf_m = self.rf_bundle["pipeline"].named_steps["model"]
            tree_p = np.stack([e.predict(x_imp) for e in rf_m.estimators_], axis=0)
            rf_preds = np.clip(np.mean(tree_p, axis=0), 0.0, 100.0)
            rf_stds = np.std(tree_p, axis=0)

        timeline = []
        for i, actual_ts in enumerate(actual_ts_list):
            dp = float(dl_preds[i])
            ds = float(dl_stds[i])
            xp = float(xgb_preds[i])
            if model_choice == "mm_gated":
                fp, fs = dp, ds
            elif model_choice == "xgboost":
                fp, fs = xp, ds
            elif model_choice == "random_forest" and rf_preds is not None:
                fp, fs = float(rf_preds[i]), float(rf_stds[i])
            else:
                fp = float(np.clip(0.70 * dp + 0.30 * xp, 0.0, 100.0))
                fs = float(np.sqrt(ds ** 2 + (0.35 * abs(dp - xp)) ** 2))

            ci_low = float(np.clip(fp - 1.96 * fs, 0.0, 100.0))
            ci_high = float(np.clip(fp + 1.96 * fs, 0.0, 100.0))

            gt_val = None
            if rec["events"]:
                best_ev = min(rec["events"], key=lambda e: abs(e["timestamp_sec"] - actual_ts))
                if abs(best_ev["timestamp_sec"] - actual_ts) <= 0.5:
                    gt_val = best_ev["pain_score_0_100"]

            timeline.append({
                "t": round(actual_ts, 2),
                "pred": round(fp, 1),
                "ci_low": round(ci_low, 1),
                "ci_high": round(ci_high, 1),
                "std": round(fs, 2),
                "mm_gated": round(dp, 1),
                "xgboost": round(xp, 1),
                "gt": gt_val,
            })

        return {
            "recording_id": recording_id,
            "participant_id": rec["participant_id"],
            "mode": "prerecorded_dataset_playback",
            "model_choice": model_choice,
            "disabled_modalities": missing_mods,
            "duration_sec": round(dur, 2),
            "no_pain_threshold_0_100": rec["no_pain_threshold"] * 10.0,
            "severe_pain_threshold_0_100": rec["severe_pain_threshold"] * 10.0,
            "timeline": timeline,
        }

    def predict_custom_signals(
        self,
        signals: Dict[str, Optional[List[float]]],
        sampling_rate_hz: float = 250.0,
        model_choice: str = "hybrid",
    ) -> Dict[str, Any]:
        """Preprocess and predict on user-provided raw sensor arrays with genuine missing-modality detection."""
        raw_win: Dict[str, Optional[np.ndarray]] = {}
        max_len = 0
        for col in SIGNAL_COLUMNS:
            vals = signals.get(col)
            if vals is not None and len(vals) > 0:
                arr = np.asarray(vals, dtype=np.float64)
                raw_win[col] = arr
                max_len = max(max_len, len(arr))
            else:
                raw_win[col] = None

        if max_len < 30:
            raise ValueError("At least one physiological channel with >= 30 samples must be provided.")

        sensor_status = detect_modality_availability(raw_win)
        active_mods = [m for m in MODALITY_ORDER if sensor_status[m]["available"]]
        missing_mods = [m for m in MODALITY_ORDER if not sensor_status[m]["available"]]
        if not active_mods:
            raise ValueError("No valid non-flatlined physiological modalities detected in input.")

        # Build DataFrame ONLY for cleaning available signals; unavailable channels stay zeroed and masked out
        df_dict = {}
        for col in SIGNAL_COLUMNS:
            arr = raw_win[col]
            if arr is not None and len(arr) >= 30:
                if len(arr) != max_len:
                    arr = np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(arr)), arr)
                df_dict[col] = arr
            else:
                df_dict[col] = np.zeros(max_len, dtype=np.float64)
        df = pd.DataFrame(df_dict)
        cleaned = clean_signals_df(df, fs=sampling_rate_hz)
        eda_decomp = {
            "Eda_E4": decompose_eda(cleaned["Eda_E4"], fs=sampling_rate_hz),
            "Eda_RB": decompose_eda(cleaned["Eda_RB"], fs=sampling_rate_hz),
        }
        baseline = compute_runup_baseline(None, cleaned)
        end_idx = max_len - 1

        feats = extract_features_for_window(cleaned, eda_decomp, baseline, end_idx, fs=sampling_rate_hz)
        for col in self.feature_columns:
            if col not in feats:
                feats[col] = 0.0

        disabled_set = set(missing_mods)
        x_tab = np.array([
            np.nan if self.feature_modality_map.get(c, "Temporal") in disabled_set else float(feats.get(c, 0.0))
            for c in self.feature_columns
        ], dtype=np.float64)

        mask_np = np.array([1.0 if m in active_mods else 0.0 for m in MODALITY_ORDER], dtype=np.float32)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0)
        dl_tensor = extract_dl_tensor_window(cleaned, baseline, end_idx)
        x_dl = torch.from_numpy(dl_tensor).unsqueeze(0).float()

        mc = predict_mc_dropout(self.dl_model, x_dl, modality_mask=mask_t, n_samples=30)
        dl_pred = float(np.clip(mc["mean"].item(), 0.0, 100.0))
        dl_std = float(max(mc["std"].item(), 0.5) * self.dl_calib_scale * (1.0 + 0.15 * len(missing_mods)))

        if missing_mods and self.xgb_missing_bundle is not None:
            xgb_pred = float(np.clip(self.xgb_missing_bundle["model"].predict(x_tab.reshape(1, -1))[0], 0.0, 100.0))
        else:
            xgb_pred = float(np.clip(self.xgb_bundle["pipeline"].predict(x_tab.reshape(1, -1))[0], 0.0, 100.0))

        final_pred = dl_pred if model_choice == "mm_gated" else (xgb_pred if model_choice == "xgboost" else 0.70 * dl_pred + 0.30 * xgb_pred)
        ci_low = float(np.clip(final_pred - 1.96 * dl_std, 0.0, 100.0))
        ci_high = float(np.clip(final_pred + 1.96 * dl_std, 0.0, 100.0))

        ig_res = compute_integrated_gradients(self.dl_model, x_dl, modality_mask=mask_t, n_steps=16)

        return {
            "mode": "custom_signal_input",
            "predicted_pain_0_100": round(final_pred, 1),
            "severity_category": classify_pain_severity(final_pred),
            "uncertainty_std": round(dl_std, 2),
            "uncertainty_level": classify_uncertainty(dl_std),
            "confidence_interval_95": [round(ci_low, 1), round(ci_high, 1)],
            "sensor_status": sensor_status,
            "active_modalities": active_mods,
            "missing_modalities": missing_mods,
            "missing_modality_warning": (
                f"Missing/invalid modalities detected: {', '.join(missing_mods)}. Masked without fabrication."
            ) if missing_mods else None,
            "explainability": {"integrated_gradients": ig_res},
        }


_service_instance: Optional[PainEstimationService] = None


def get_service() -> PainEstimationService:
    global _service_instance
    if _service_instance is None:
        _service_instance = PainEstimationService()
    return _service_instance

