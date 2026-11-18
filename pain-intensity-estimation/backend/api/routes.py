"""FastAPI route definitions for the Continuous Pain-Intensity Estimation API."""
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.config import MODALITY_ORDER, SAMPLING_RATE_HZ
from backend.preprocessing.signal_pipeline import clean_signals_df, decompose_eda, detect_modality_availability
from backend.services.model_service import get_service

router = APIRouter(prefix="/api", tags=["Pain Estimation API"])


class RecordingPredictRequest(BaseModel):
    recording_id: str = Field(..., examples=["P01_1"])
    timestamp_sec: float = Field(..., ge=0.0, examples=[56.46])
    model_choice: str = Field(default="hybrid", description="hybrid | mm_gated | xgboost | random_forest")
    disabled_modalities: List[str] = Field(default_factory=list, description="Optional list of modalities to simulate sensor dropout")
    include_explainability: bool = True


class TimelinePredictRequest(BaseModel):
    recording_id: str = Field(..., examples=["P01_1"])
    step_sec: float = Field(default=4.0, ge=1.0, le=20.0)
    model_choice: str = Field(default="hybrid")
    disabled_modalities: List[str] = Field(default_factory=list)


class CustomSignalRequest(BaseModel):
    signals: Dict[str, Optional[List[float]]] = Field(
        ...,
        description="Dictionary mapping PMPDB channels (Bvp, Eda_E4, Tmp, Resp, Eda_RB, Bvp_RB, Emg, Grip) to numeric arrays or null if unavailable.",
    )
    sampling_rate_hz: float = Field(default=SAMPLING_RATE_HZ, gt=10.0)
    model_choice: str = Field(default="hybrid")


class PreprocessRequest(BaseModel):
    recording_id: Optional[str] = None
    start_sec: float = Field(default=0.0, ge=0.0)
    duration_sec: float = Field(default=10.0, gt=0.5, le=60.0)
    signals: Optional[Dict[str, Optional[List[float]]]] = None
    sampling_rate_hz: float = Field(default=SAMPLING_RATE_HZ, gt=10.0)


@router.get("/health")
def health_check():
    svc = get_service()
    return {
        "status": "ok" if svc.models_ready else "degraded_no_checkpoints",
        "models_ready": svc.models_ready,
        "loaded_at": svc.loaded_at,
        "research_disclaimer": "Academic research prototype only. Not for clinical diagnosis or medical treatment.",
    }


@router.get("/models/status")
def model_status():
    return get_service().get_model_status()


@router.post("/models/reload")
def reload_models():
    return get_service().reload()


@router.get("/dataset/summary")
def dataset_summary():
    try:
        summary = get_service().get_dataset_summary()
        # Omit heavy per-recording signal_stats in top-level summary for fast UI rendering
        slim = {k: v for k, v in summary.items() if k != "recordings"}
        slim["recordings_count"] = len(summary.get("recordings", []))
        return slim
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/dataset/participants")
def dataset_participants():
    try:
        summary = get_service().get_dataset_summary()
        grouped: Dict[str, List[Dict]] = {}
        for rec in summary.get("recordings", []):
            pid = rec["participant_id"]
            grouped.setdefault(pid, []).append({
                "recording_id": rec["recording_id"],
                "session": rec["session"],
                "duration_sec": rec["duration_sec"],
                "n_pain_rates": rec["n_pain_rates"],
                "pain_rate_min": rec["pain_rate_min"],
                "pain_rate_max": rec["pain_rate_max"],
                "pain_rate_mean": rec["pain_rate_mean"],
                "no_pain_threshold": rec["no_pain_threshold"],
                "severe_pain_threshold": rec["severe_pain_threshold"],
            })
        return {
            "num_participants": len(grouped),
            "participants": [
                {"participant_id": pid, "recordings": recs}
                for pid, recs in sorted(grouped.items())
            ],
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/dataset/recordings/{recording_id}")
def get_recording(recording_id: str, downsample_hz: float = Query(default=8.0, ge=1.0, le=50.0)):
    try:
        return get_service().get_recording_signals(recording_id, downsample_hz=downsample_hz)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/preprocess")
def preprocess_signals(req: PreprocessRequest):
    svc = get_service()
    if req.recording_id:
        try:
            rec = svc._load_recording(req.recording_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        df = rec["df"]
        mask = (df["Seconds"] >= req.start_sec) & (df["Seconds"] <= req.start_sec + req.duration_sec)
        sub = df[mask]
        if len(sub) < 30:
            raise HTTPException(status_code=400, detail="Requested time slice has too few samples.")
        cleaned = clean_signals_df(sub, fs=SAMPLING_RATE_HZ)
        tonic, phasic = decompose_eda(cleaned["Eda_RB"], fs=SAMPLING_RATE_HZ)
        step = max(1, int(SAMPLING_RATE_HZ / 10.0))
        return {
            "recording_id": req.recording_id,
            "samples_processed": len(sub),
            "sampling_rate_hz": SAMPLING_RATE_HZ,
            "filters_applied": {
                "BVP": "4th-order Butterworth bandpass 0.5-8.0 Hz",
                "EDA": "4th-order lowpass 5.0 Hz + 0.05 Hz tonic/phasic decomposition",
                "Resp": "3rd-order bandpass 0.05-1.0 Hz",
                "Tmp": "3rd-order lowpass 0.5 Hz",
                "EMG": "50 Hz IIR notch + 4th-order bandpass 20-120 Hz",
                "Grip": "4th-order lowpass 10.0 Hz",
            },
            "preview_10hz": {
                "t": [round(float(x), 2) for x in sub["Seconds"].values[::step]],
                "Bvp_clean": [round(float(x), 4) for x in cleaned["Bvp"][::step]],
                "Eda_RB_tonic": [round(float(x), 4) for x in tonic[::step]],
                "Eda_RB_phasic": [round(float(x), 4) for x in phasic[::step]],
                "Emg_clean": [round(float(x), 5) for x in cleaned["Emg"][::step]],
            },
        }
    elif req.signals:
        import numpy as np
        raw_win = {k: (np.asarray(v, dtype=float) if v else None) for k, v in req.signals.items()}
        status = detect_modality_availability(raw_win)
        return {"sensor_status": status, "available_modalities": [m for m in MODALITY_ORDER if status[m]["available"]]}
    raise HTTPException(status_code=400, detail="Provide either recording_id or signals.")


@router.post("/predict/recording")
def predict_recording(req: RecordingPredictRequest):
    try:
        return get_service().predict_recording_window(
            recording_id=req.recording_id,
            timestamp_sec=req.timestamp_sec,
            model_choice=req.model_choice,
            disabled_modalities=req.disabled_modalities,
            include_explainability=req.include_explainability,
            log_history=True,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/predict/timeline")
def predict_timeline(req: TimelinePredictRequest):
    try:
        return get_service().predict_recording_timeline(
            recording_id=req.recording_id,
            step_sec=req.step_sec,
            model_choice=req.model_choice,
            disabled_modalities=req.disabled_modalities,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/predict/custom")
def predict_custom(req: CustomSignalRequest):
    try:
        return get_service().predict_custom_signals(
            signals=req.signals,
            sampling_rate_hz=req.sampling_rate_hz,
            model_choice=req.model_choice,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/explain/global")
def explain_global():
    try:
        return get_service().get_global_xai()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/explain/window")
def explain_window(req: RecordingPredictRequest):
    try:
        res = get_service().predict_recording_window(
            recording_id=req.recording_id,
            timestamp_sec=req.timestamp_sec,
            model_choice=req.model_choice,
            disabled_modalities=req.disabled_modalities,
            include_explainability=True,
            log_history=False,
        )
        return {
            "recording_id": res["recording_id"],
            "window_end_sec": res["window_end_sec"],
            "predicted_pain_0_100": res["predicted_pain_0_100"],
            "uncertainty_std": res["uncertainty_std"],
            "confidence_interval_95": res["confidence_interval_95"],
            "sensor_status": res["sensor_status"],
            "explainability": res["explainability"],
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/evaluation/summary")
def evaluation_summary():
    return get_service().get_evaluation_summary()


@router.get("/predictions/history")
def prediction_history(limit: int = Query(default=50, ge=1, le=250)):
    svc = get_service()
    items = list(svc.history)[:limit]
    return {"count": len(items), "items": items}


@router.delete("/predictions/history")
def clear_prediction_history():
    svc = get_service()
    svc.history.clear()
    return {"cleared": True}

