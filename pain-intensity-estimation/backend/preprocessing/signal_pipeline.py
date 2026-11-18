"""Phase 3: Biomedical Signal Preprocessing, Feature Extraction, and Windowing for PMPDB.

Handles the 8 verified PMPDB channels (6 modalities):
- BVP: Bvp (Empatica E4), Bvp_RB (RespiBAN) -> bandpass 0.5-8 Hz, systolic peaks, PR, PRV
- EDA: Eda_E4 (Empatica E4), Eda_RB (RespiBAN) -> lowpass 5 Hz, tonic/phasic decomposition, SCRs
- Resp: Resp (RespiBAN) -> bandpass 0.05-1.0 Hz, breath rate, amplitude, variability
- Tmp: Tmp (Empatica E4) -> lowpass 0.5 Hz, trend, slope, runUp baseline shift
- EMG: Emg (RespiBAN) -> 50 Hz notch + bandpass 20-120 Hz, RMS, MAV, WL, ZCR, spectral freq
- Grip: Grip (RespiBAN) -> lowpass 10 Hz, mean, peak, slope, impulse, runUp baseline shift

Explicitly does NOT fabricate ECG or EEG signals (neither exists in PMPDB).
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal as sp_signal

from backend.config import (
    DOWNSAMPLED_RATE_HZ,
    METADATA_DIR,
    MODALITY_GROUPS,
    MODALITY_ORDER,
    PAIN_SCALE_FACTOR,
    PROCESSED_DIR,
    RAW_DATASET_DIR,
    SAMPLING_RATE_HZ,
    SIGNAL_COLUMNS,
    WINDOW_SAMPLES_DL,
    WINDOW_SAMPLES_RAW,
    WINDOW_SECONDS,
)


def _safe_butter_filter(
    x: np.ndarray,
    fs: float,
    btype: str,
    cutoff: Any,
    order: int = 4,
) -> np.ndarray:
    """Apply zero-phase Butterworth filter safely even on short or constant arrays."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 30 or np.std(x) < 1e-12:
        return x.copy()
    nyq = 0.5 * fs
    if isinstance(cutoff, (list, tuple)):
        wn = [max(1e-4, min(0.99, c / nyq)) for c in cutoff]
        if wn[0] >= wn[1]:
            return x.copy()
    else:
        wn = max(1e-4, min(0.99, cutoff / nyq))
    sos = sp_signal.butter(order, wn, btype=btype, output="sos")
    try:
        return sp_signal.sosfiltfilt(sos, x)
    except ValueError:
        return sp_signal.sosfilt(sos, x)


def _notch_50hz(x: np.ndarray, fs: float = 250.0, q: float = 30.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 30 or np.std(x) < 1e-12 or fs <= 105.0:
        return x.copy()
    b, a = sp_signal.iirnotch(50.0 / (0.5 * fs), q)
    try:
        return sp_signal.filtfilt(b, a, x)
    except ValueError:
        return x.copy()


def clean_signals_df(df: pd.DataFrame, fs: float = SAMPLING_RATE_HZ) -> Dict[str, np.ndarray]:
    """Clean all available physiological channels on a full recording (before windowing)."""
    cleaned: Dict[str, np.ndarray] = {}
    for col in SIGNAL_COLUMNS:
        if col not in df.columns:
            continue
        raw = pd.to_numeric(df[col], errors="coerce").interpolate(limit_direction="both").fillna(0.0).values.astype(np.float64)
        if col in ("Bvp", "Bvp_RB"):
            cleaned[col] = _safe_butter_filter(raw, fs, "bandpass", (0.5, 8.0), order=4)
        elif col in ("Eda_E4", "Eda_RB"):
            cleaned[col] = _safe_butter_filter(raw, fs, "lowpass", 5.0, order=4)
        elif col == "Resp":
            cleaned[col] = _safe_butter_filter(raw, fs, "bandpass", (0.05, 1.0), order=3)
        elif col == "Tmp":
            cleaned[col] = _safe_butter_filter(raw, fs, "lowpass", 0.5, order=3)
        elif col == "Emg":
            notched = _notch_50hz(raw, fs=fs, q=30.0)
            cleaned[col] = _safe_butter_filter(notched, fs, "bandpass", (20.0, 120.0), order=4)
        elif col == "Grip":
            cleaned[col] = _safe_butter_filter(raw, fs, "lowpass", 10.0, order=4)
        else:
            cleaned[col] = raw
    return cleaned


def decompose_eda(eda_clean: np.ndarray, fs: float = SAMPLING_RATE_HZ) -> Tuple[np.ndarray, np.ndarray]:
    """Separate cleaned EDA into Tonic (SCL) and Phasic (SCR) components."""
    if len(eda_clean) < 30 or np.std(eda_clean) < 1e-12:
        return eda_clean.copy(), np.zeros_like(eda_clean)
    tonic = _safe_butter_filter(eda_clean, fs, "lowpass", 0.05, order=2)
    phasic = eda_clean - tonic
    return tonic, phasic


def compute_runup_baseline(runup_df: Optional[pd.DataFrame], fallback_cleaned: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """Compute subject-specific baseline statistics from pre-session runUp recording (zero leakage)."""
    source = clean_signals_df(runup_df) if runup_df is not None and len(runup_df) >= 250 else fallback_cleaned
    stats: Dict[str, Dict[str, float]] = {}
    for col in SIGNAL_COLUMNS:
        arr = source.get(col)
        if arr is None or len(arr) == 0:
            stats[col] = {"mean": 0.0, "std": 1.0, "median": 0.0, "iqr": 1.0}
            continue
        std = float(np.std(arr))
        q75, q25 = np.percentile(arr, [75, 25])
        iqr = float(q75 - q25)
        stats[col] = {
            "mean": float(np.mean(arr)),
            "std": std if std > 1e-6 else 1.0,
            "median": float(np.median(arr)),
            "iqr": iqr if iqr > 1e-6 else (std if std > 1e-6 else 1.0),
        }
    return stats


def _linear_slope(y: np.ndarray, fs: float) -> float:
    if len(y) < 2:
        return 0.0
    t = np.arange(len(y), dtype=np.float64) / fs
    t_c = t - t.mean()
    denom = float(np.dot(t_c, t_c))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(t_c, y - y.mean()) / denom)


def _extract_bvp_features(sig: np.ndarray, prefix: str, fs: float, base: Dict[str, float]) -> Dict[str, float]:
    std = float(np.std(sig))
    mean = float(np.mean(sig))
    rms = float(np.sqrt(np.mean(sig ** 2)))
    ptp = float(np.ptp(sig))
    # Systolic peak detection
    min_dist = int(0.35 * fs)  # up to ~171 BPM
    prom = max(0.25 * std, 1e-6)
    peaks, props = sp_signal.find_peaks(sig, distance=min_dist, prominence=prom)
    if len(peaks) >= 2:
        ibis = np.diff(peaks) / fs  # seconds
        valid_ibis = ibis[(ibis >= 0.35) & (ibis <= 1.8)]
    else:
        valid_ibis = np.array([], dtype=np.float64)

    if len(valid_ibis) >= 1:
        mean_ibi = float(np.mean(valid_ibis))
        pr_bpm = float(60.0 / max(mean_ibi, 0.3))
        sdnn = float(np.std(valid_ibis) * 1000.0) if len(valid_ibis) >= 2 else 0.0
        diffs = np.diff(valid_ibis)
        rmssd = float(np.sqrt(np.mean(diffs ** 2)) * 1000.0) if len(diffs) >= 1 else 0.0
        pnn50 = float(np.mean(np.abs(diffs) > 0.05)) if len(diffs) >= 1 else 0.0
    else:
        pr_bpm, sdnn, rmssd, pnn50 = 75.0, 0.0, 0.0, 0.0

    peak_amp = float(np.mean(props["prominences"])) if len(peaks) > 0 and "prominences" in props else ptp
    # Band power via Welch PSD
    freqs, psd = sp_signal.welch(sig, fs=fs, nperseg=min(len(sig), 512))
    total_p = float(np.trapezoid(psd, freqs)) + 1e-12
    cardiac_mask = (freqs >= 0.7) & (freqs <= 3.5)
    cardiac_ratio = float(np.trapezoid(psd[cardiac_mask], freqs[cardiac_mask]) / total_p)

    return {
        f"BVP_{prefix}_std": std,
        f"BVP_{prefix}_rms": rms,
        f"BVP_{prefix}_ptp": ptp,
        f"BVP_{prefix}_pr_bpm": pr_bpm,
        f"BVP_{prefix}_prv_sdnn_ms": sdnn,
        f"BVP_{prefix}_prv_rmssd_ms": rmssd,
        f"BVP_{prefix}_prv_pnn50": pnn50,
        f"BVP_{prefix}_peak_amp_norm": peak_amp / base["std"],
        f"BVP_{prefix}_cardiac_power_ratio": cardiac_ratio,
        f"BVP_{prefix}_rms_ratio": rms / base["std"],
    }


def _extract_eda_features(
    clean_seg: np.ndarray,
    tonic_seg: np.ndarray,
    phasic_seg: np.ndarray,
    prefix: str,
    fs: float,
    base: Dict[str, float],
) -> Dict[str, float]:
    scl_mean = float(np.mean(tonic_seg))
    scl_std = float(np.std(tonic_seg))
    scl_slope = _linear_slope(tonic_seg, fs)
    scl_delta = float(tonic_seg[-1] - tonic_seg[0]) if len(tonic_seg) > 1 else 0.0
    scl_rel_shift = (scl_mean - base["mean"]) / base["std"]

    # Phasic SCR peaks
    ph_std = float(np.std(phasic_seg))
    min_dist = int(0.8 * fs)
    prom = max(0.015, 0.35 * ph_std)
    peaks, props = sp_signal.find_peaks(phasic_seg, distance=min_dist, prominence=prom)
    scr_count = float(len(peaks))
    scr_amp_mean = float(np.mean(props["prominences"])) if len(peaks) > 0 else 0.0
    scr_amp_max = float(np.max(props["prominences"])) if len(peaks) > 0 else 0.0
    scr_pos_auc = float(np.mean(np.maximum(phasic_seg, 0.0)))

    deriv = np.diff(clean_seg) * fs
    deriv_std = float(np.std(deriv)) if len(deriv) > 0 else 0.0
    deriv_max = float(np.max(deriv)) if len(deriv) > 0 else 0.0

    return {
        f"EDA_{prefix}_scl_mean": scl_mean,
        f"EDA_{prefix}_scl_std": scl_std,
        f"EDA_{prefix}_scl_slope": scl_slope,
        f"EDA_{prefix}_scl_delta": scl_delta,
        f"EDA_{prefix}_scl_rel_runup": scl_rel_shift,
        f"EDA_{prefix}_phasic_std": ph_std,
        f"EDA_{prefix}_phasic_auc": scr_pos_auc,
        f"EDA_{prefix}_scr_count": scr_count,
        f"EDA_{prefix}_scr_amp_mean": scr_amp_mean,
        f"EDA_{prefix}_scr_amp_max": scr_amp_max,
        f"EDA_{prefix}_deriv_std": deriv_std,
        f"EDA_{prefix}_deriv_max": deriv_max,
    }


def _extract_resp_features(sig: np.ndarray, fs: float, base: Dict[str, float]) -> Dict[str, float]:
    std = float(np.std(sig))
    rms = float(np.sqrt(np.mean(sig ** 2)))
    ptp = float(np.ptp(sig))
    min_dist = int(1.2 * fs)
    peaks, _ = sp_signal.find_peaks(sig, distance=min_dist, prominence=max(0.2 * std, 1e-4))
    if len(peaks) >= 2:
        ibis = np.diff(peaks) / fs
        resp_rate = float(60.0 / max(np.mean(ibis), 0.5))
        ibi_std = float(np.std(ibis))
    else:
        # Spectral fallback for 6s window (often 1-2 breaths)
        freqs, psd = sp_signal.welch(sig, fs=fs, nperseg=len(sig))
        band = (freqs >= 0.08) & (freqs <= 0.8)
        if np.any(band):
            dom_f = float(freqs[band][np.argmax(psd[band])])
            resp_rate = dom_f * 60.0
        else:
            resp_rate = 15.0
        ibi_std = 0.0

    slope = _linear_slope(sig, fs)
    return {
        "Resp_std": std,
        "Resp_rms": rms,
        "Resp_ptp": ptp,
        "Resp_rate_bpm": resp_rate,
        "Resp_ibi_std": ibi_std,
        "Resp_slope": slope,
        "Resp_rms_ratio": rms / base["std"],
        "Resp_mean_rel_runup": (float(np.mean(sig)) - base["mean"]) / base["std"],
    }


def _extract_tmp_features(sig: np.ndarray, fs: float, base: Dict[str, float]) -> Dict[str, float]:
    mean = float(np.mean(sig))
    std = float(np.std(sig))
    ptp = float(np.ptp(sig))
    slope = _linear_slope(sig, fs)
    delta = float(sig[-1] - sig[0]) if len(sig) > 1 else 0.0
    rel_shift_c = mean - base["mean"]
    rel_z = rel_shift_c / base["std"]
    return {
        "Tmp_mean": mean,
        "Tmp_std": std,
        "Tmp_ptp": ptp,
        "Tmp_slope": slope,
        "Tmp_delta": delta,
        "Tmp_rel_runup_c": rel_shift_c,
        "Tmp_rel_runup_z": rel_z,
    }


def _extract_emg_features(sig: np.ndarray, fs: float, base: Dict[str, float]) -> Dict[str, float]:
    abs_sig = np.abs(sig)
    rms = float(np.sqrt(np.mean(sig ** 2)))
    mav = float(np.mean(abs_sig))
    peak = float(np.max(abs_sig))
    wl = float(np.mean(np.abs(np.diff(sig)))) if len(sig) > 1 else 0.0
    zcr = float(np.mean(np.diff(np.signbit(sig)))) if len(sig) > 1 else 0.0
    iemg = float(np.sum(abs_sig) / fs)

    freqs, psd = sp_signal.welch(sig, fs=fs, nperseg=min(len(sig), 512))
    psd_sum = float(np.sum(psd)) + 1e-12
    mean_freq = float(np.sum(freqs * psd) / psd_sum)
    cdf = np.cumsum(psd) / psd_sum
    med_idx = min(len(freqs) - 1, int(np.searchsorted(cdf, 0.5)))
    med_freq = float(freqs[med_idx]) if len(freqs) > 0 else 0.0

    return {
        "EMG_rms": rms,
        "EMG_mav": mav,
        "EMG_peak": peak,
        "EMG_wl": wl,
        "EMG_zcr": zcr,
        "EMG_iemg": iemg,
        "EMG_mean_freq": mean_freq,
        "EMG_median_freq": med_freq,
        "EMG_rms_ratio": rms / base["std"],
    }


def _extract_grip_features(sig: np.ndarray, fs: float, base: Dict[str, float]) -> Dict[str, float]:
    mean = float(np.mean(sig))
    std = float(np.std(sig))
    vmin = float(np.min(sig))
    vmax = float(np.max(sig))
    ptp = vmax - vmin
    slope = _linear_slope(sig, fs)
    delta = float(sig[-1] - sig[0]) if len(sig) > 1 else 0.0
    rel_shift = (mean - base["mean"]) / base["std"]
    peak_rel = (vmax - base["mean"]) / base["std"]
    return {
        "Grip_mean": mean,
        "Grip_std": std,
        "Grip_min": vmin,
        "Grip_max": vmax,
        "Grip_ptp": ptp,
        "Grip_slope": slope,
        "Grip_delta": delta,
        "Grip_rel_runup_z": rel_shift,
        "Grip_peak_rel_runup_z": peak_rel,
    }


def slice_causal_window(arr: np.ndarray, end_idx: int, win_len: int) -> np.ndarray:
    """Extract causal window [end_idx - win_len + 1, end_idx + 1], edge-padding left if needed."""
    end = min(len(arr), max(1, end_idx + 1))
    start = end - win_len
    if start >= 0:
        return arr[start:end].copy()
    seg = arr[0:end]
    pad_len = win_len - len(seg)
    return np.pad(seg, (pad_len, 0), mode="edge")


def extract_features_for_window(
    cleaned_full: Dict[str, np.ndarray],
    eda_decomp: Dict[str, Tuple[np.ndarray, np.ndarray]],
    baseline: Dict[str, Dict[str, float]],
    end_idx: int,
    fs: float = SAMPLING_RATE_HZ,
) -> Dict[str, float]:
    """Extract all 6-modality engineered features for a single 6-second causal window."""
    feats: Dict[str, float] = {}

    # BVP (E4 + RespiBAN)
    for col, pref in [("Bvp", "E4"), ("Bvp_RB", "RB")]:
        seg = slice_causal_window(cleaned_full[col], end_idx, WINDOW_SAMPLES_RAW)
        feats.update(_extract_bvp_features(seg, pref, fs, baseline[col]))

    # EDA (E4 + RespiBAN)
    for col, pref in [("Eda_E4", "E4"), ("Eda_RB", "RB")]:
        c_seg = slice_causal_window(cleaned_full[col], end_idx, WINDOW_SAMPLES_RAW)
        t_seg = slice_causal_window(eda_decomp[col][0], end_idx, WINDOW_SAMPLES_RAW)
        p_seg = slice_causal_window(eda_decomp[col][1], end_idx, WINDOW_SAMPLES_RAW)
        feats.update(_extract_eda_features(c_seg, t_seg, p_seg, pref, fs, baseline[col]))

    # Resp
    r_seg = slice_causal_window(cleaned_full["Resp"], end_idx, WINDOW_SAMPLES_RAW)
    feats.update(_extract_resp_features(r_seg, fs, baseline["Resp"]))

    # Tmp
    t_seg = slice_causal_window(cleaned_full["Tmp"], end_idx, WINDOW_SAMPLES_RAW)
    feats.update(_extract_tmp_features(t_seg, fs, baseline["Tmp"]))

    # EMG
    e_seg = slice_causal_window(cleaned_full["Emg"], end_idx, WINDOW_SAMPLES_RAW)
    feats.update(_extract_emg_features(e_seg, fs, baseline["Emg"]))

    # Grip
    g_seg = slice_causal_window(cleaned_full["Grip"], end_idx, WINDOW_SAMPLES_RAW)
    feats.update(_extract_grip_features(g_seg, fs, baseline["Grip"]))

    return feats


def extract_dl_tensor_window(
    cleaned_full: Dict[str, np.ndarray],
    baseline: Dict[str, Dict[str, float]],
    end_idx: int,
) -> np.ndarray:
    """Build normalized (8, 300) float32 tensor at 50 Hz for deep learning models.

    Normalization uses the recording's own pre-session runUp baseline (zero train/test leakage),
    clipped to [-8, 8] to guard against extreme motion artifacts.
    """
    tensor = np.zeros((len(SIGNAL_COLUMNS), WINDOW_SAMPLES_DL), dtype=np.float32)
    for i, col in enumerate(SIGNAL_COLUMNS):
        seg = slice_causal_window(cleaned_full[col], end_idx, WINDOW_SAMPLES_RAW)
        # Anti-aliased resample 1500 -> 300 (5x decimation: 250 Hz -> 50 Hz)
        ds = sp_signal.resample_poly(seg, up=1, down=5)
        if len(ds) != WINDOW_SAMPLES_DL:
            ds = sp_signal.resample(seg, WINDOW_SAMPLES_DL)
        b = baseline[col]
        norm = (ds - b["mean"]) / b["std"]
        norm = np.clip(norm, -8.0, 8.0)
        tensor[i] = norm.astype(np.float32)
    return tensor


def detect_modality_availability(
    raw_window: Dict[str, Optional[np.ndarray]],
) -> Dict[str, Dict[str, Any]]:
    """Inspect raw sensor inputs and report modality availability without fabricating values."""
    status: Dict[str, Dict[str, Any]] = {}
    for mod in MODALITY_ORDER:
        cols = MODALITY_GROUPS[mod]
        ch_details = {}
        available_cols = 0
        for col in cols:
            arr = raw_window.get(col)
            if arr is None or len(arr) == 0:
                ch_details[col] = {"available": False, "reason": "missing_channel"}
                continue
            a = np.asarray(arr, dtype=np.float64)
            nan_ratio = float(np.mean(~np.isfinite(a)))
            if nan_ratio > 0.5:
                ch_details[col] = {"available": False, "reason": f"high_missingness ({nan_ratio*100:.1f}%)"}
                continue
            finite = a[np.isfinite(a)]
            if len(finite) < 10 or float(np.std(finite)) < 1e-7:
                ch_details[col] = {"available": False, "reason": "flatline_or_constant"}
                continue
            available_cols += 1
            ch_details[col] = {"available": True, "reason": "ok", "nan_pct": round(nan_ratio * 100.0, 2)}

        status[mod] = {
            "available": available_cols > 0,
            "active_channels": available_cols,
            "total_channels": len(cols),
            "channels": ch_details,
        }
    return status


def map_feature_to_modality(feature_name: str) -> str:
    for mod in MODALITY_ORDER:
        if feature_name.startswith(f"{mod}_") or feature_name.startswith(f"delta_{mod}_"):
            return mod
    return "Temporal"


def build_processed_dataset() -> Dict[str, Any]:
    """Run full Phase 3 preprocessing across all 90 PMPDB recordings and save structured outputs."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    rec_dirs = sorted([d for d in RAW_DATASET_DIR.iterdir() if d.is_dir()], key=lambda p: p.name)

    rows: List[Dict[str, Any]] = []
    dl_windows: List[np.ndarray] = []

    key_delta_cols = [
        "EDA_E4_scl_rel_runup",
        "EDA_RB_scl_rel_runup",
        "EDA_RB_phasic_auc",
        "BVP_E4_pr_bpm",
        "BVP_RB_rms_ratio",
        "Resp_rms_ratio",
        "Tmp_rel_runup_c",
        "EMG_rms_ratio",
        "Grip_rel_runup_z",
    ]

    for idx_r, rdir in enumerate(rec_dirs):
        rec_id = rdir.name
        m = re.match(r"(P\d+)_(\d+)", rec_id)
        pid, sess = (m.group(1), int(m.group(2))) if m else (rec_id, 0)

        main_df = pd.read_csv(rdir / f"{rec_id}.csv", sep=";", decimal=",", low_memory=False)
        runup_path = rdir / f"{rec_id}_runUp.csv"
        runup_df = pd.read_csv(runup_path, sep=";", decimal=",", low_memory=False) if runup_path.exists() else None
        no_thr = float((rdir / "noPainThreshold.txt").read_text(encoding="utf-8").strip())
        sev_thr = float((rdir / "severePainThreshold.txt").read_text(encoding="utf-8").strip())

        cleaned = clean_signals_df(main_df, fs=SAMPLING_RATE_HZ)
        eda_decomp = {
            "Eda_E4": decompose_eda(cleaned["Eda_E4"], fs=SAMPLING_RATE_HZ),
            "Eda_RB": decompose_eda(cleaned["Eda_RB"], fs=SAMPLING_RATE_HZ),
        }
        baseline = compute_runup_baseline(runup_df, cleaned)

        dur_sec = float(main_df["Seconds"].iloc[-1] - main_df["Seconds"].iloc[0])
        pr_indices = np.where(main_df["Pain rates"].notna().values)[0]

        prev_feats: Optional[Dict[str, float]] = None
        for ev_i, row_idx in enumerate(pr_indices):
            ts = float(main_df["Seconds"].iloc[row_idx])
            pr_nrs = float(main_df["Pain rates"].iloc[row_idx])
            pl_cls = int(main_df["Pain labels"].iloc[row_idx])
            pain_0_100 = float(np.clip(pr_nrs * PAIN_SCALE_FACTOR, 0.0, 100.0))

            feats = extract_features_for_window(cleaned, eda_decomp, baseline, int(row_idx), fs=SAMPLING_RATE_HZ)

            # Causal inter-window temporal deltas (from immediately preceding 6s window [t-12s, t-6s])
            prev_end_idx = max(0, int(row_idx) - WINDOW_SAMPLES_RAW)
            if int(row_idx) >= WINDOW_SAMPLES_RAW:
                prior_feats = extract_features_for_window(cleaned, eda_decomp, baseline, prev_end_idx, fs=SAMPLING_RATE_HZ)
            else:
                prior_feats = prev_feats if prev_feats is not None else feats

            for k in key_delta_cols:
                feats[f"delta_{k}"] = float(feats[k] - prior_feats[k])

            # Causal recording progression context
            feats["Temporal_elapsed_sec"] = ts
            feats["Temporal_progress_ratio"] = ts / max(dur_sec, 1.0)
            prev_feats = feats

            rec_row: Dict[str, Any] = {
                "window_id": len(rows),
                "recording_id": rec_id,
                "participant_id": pid,
                "session": sess,
                "event_idx": ev_i,
                "sample_idx": int(row_idx),
                "timestamp_sec": round(ts, 4),
                "duration_sec": round(dur_sec, 3),
                "pain_rate_nrs": pr_nrs,
                "pain_score_0_100": pain_0_100,
                "pain_label": pl_cls,
                "no_pain_threshold": no_thr,
                "severe_pain_threshold": sev_thr,
            }
            rec_row.update(feats)
            rows.append(rec_row)

            dl_tensor = extract_dl_tensor_window(cleaned, baseline, int(row_idx))
            dl_windows.append(dl_tensor)

        if (idx_r + 1) % 15 == 0 or (idx_r + 1) == len(rec_dirs):
            print(f"Processed {idx_r + 1}/{len(rec_dirs)} recordings ({len(rows)} windows)")

    feat_df = pd.DataFrame(rows)
    meta_cols = [
        "window_id", "recording_id", "participant_id", "session", "event_idx",
        "sample_idx", "timestamp_sec", "duration_sec", "pain_rate_nrs",
        "pain_score_0_100", "pain_label", "no_pain_threshold", "severe_pain_threshold",
    ]
    feature_cols = [c for c in feat_df.columns if c not in meta_cols]

    # Guard against any rare NaN/inf in engineered features
    for c in feature_cols:
        feat_df[c] = pd.to_numeric(feat_df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    feat_df.to_parquet(PROCESSED_DIR / "features.parquet", index=False)
    feat_df.to_csv(PROCESSED_DIR / "features.csv", index=False)

    X_dl = np.stack(dl_windows, axis=0)  # (N, 8, 300)
    np.savez_compressed(
        PROCESSED_DIR / "windows_dl.npz",
        X=X_dl,
        y_reg=feat_df["pain_score_0_100"].values.astype(np.float32),
        y_cls=feat_df["pain_label"].values.astype(np.int64),
        participant_ids=feat_df["participant_id"].values,
        recording_ids=feat_df["recording_id"].values,
        timestamps=feat_df["timestamp_sec"].values.astype(np.float32),
        channels=np.array(SIGNAL_COLUMNS),
    )

    feature_modality_map = {c: map_feature_to_modality(c) for c in feature_cols}
    mod_feature_counts = {}
    for c, m_name in feature_modality_map.items():
        mod_feature_counts[m_name] = mod_feature_counts.get(m_name, 0) + 1

    proc_meta = {
        "n_windows": int(len(feat_df)),
        "n_participants": int(feat_df["participant_id"].nunique()),
        "n_recordings": int(feat_df["recording_id"].nunique()),
        "window_seconds": WINDOW_SECONDS,
        "raw_fs_hz": SAMPLING_RATE_HZ,
        "dl_fs_hz": DOWNSAMPLED_RATE_HZ,
        "dl_tensor_shape": list(X_dl.shape),
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "feature_modality_map": feature_modality_map,
        "modality_feature_counts": mod_feature_counts,
        "channels_order": SIGNAL_COLUMNS,
    }
    with open(PROCESSED_DIR / "feature_metadata.json", "w", encoding="utf-8") as f:
        json.dump(proc_meta, f, indent=2)

    print(f"Saved {len(feat_df)} windows, {len(feature_cols)} features, DL tensor {X_dl.shape}")
    print("Features per modality:", mod_feature_counts)
    return proc_meta


if __name__ == "__main__":
    build_processed_dataset()
