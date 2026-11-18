"""Phase 4: Exploratory Data Analysis visualizations on actual PMPDB recordings."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal as sp_signal, stats

from backend.config import FIGURES_DIR, METADATA_DIR, PROCESSED_DIR, RAW_DATASET_DIR, SAMPLING_RATE_HZ
from backend.preprocessing.signal_pipeline import clean_signals_df, decompose_eda

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.bbox"] = "tight"


def generate_all_eda_plots() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    feat_df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    with open(PROCESSED_DIR / "feature_metadata.json", "r", encoding="utf-8") as f:
        feat_meta = json.load(f)
    feature_cols = feat_meta["feature_columns"]

    # 1. Multimodal waveforms on actual recording P01_1
    rec_id = "P01_1"
    raw_df = pd.read_csv(RAW_DATASET_DIR / rec_id / f"{rec_id}.csv", sep=";", decimal=",")
    cleaned = clean_signals_df(raw_df, fs=SAMPLING_RATE_HZ)
    t = raw_df["Seconds"].values
    ds_step = 5  # 50 Hz plotting resolution
    t_ds = t[::ds_step]
    pr_mask = raw_df["Pain rates"].notna().values
    pr_t = t[pr_mask]
    pr_y = raw_df.loc[pr_mask, "Pain rates"].values * 10.0

    fig, axes = plt.subplots(7, 1, figsize=(13, 14), sharex=True)
    panels = [
        ("BVP (Empatica E4, 0.5–8 Hz)", cleaned["Bvp"][::ds_step], "#ef4444"),
        ("BVP (RespiBAN, 0.5–8 Hz)", cleaned["Bvp_RB"][::ds_step], "#f97316"),
        ("EDA (Empatica E4 & RespiBAN, µS)", None, "#0ea5e9"),
        ("Respiration (RespiBAN, 0.05–1 Hz)", cleaned["Resp"][::ds_step], "#10b981"),
        ("Skin Temperature (Empatica E4, °C)", cleaned["Tmp"][::ds_step], "#8b5cf6"),
        ("Surface EMG (RespiBAN, 20–120 Hz)", cleaned["Emg"][::ds_step], "#eab308"),
        ("Grip Force (RespiBAN) + Observed Pain Ratings (0–100)", cleaned["Grip"][::ds_step], "#6366f1"),
    ]
    for i, (title, sig, color) in enumerate(panels):
        ax = axes[i]
        if i == 2:
            ax.plot(t_ds, cleaned["Eda_E4"][::ds_step], color="#0ea5e9", lw=1.1, label="Eda_E4 (wrist)")
            ax2 = ax.twinx()
            ax2.plot(t_ds, cleaned["Eda_RB"][::ds_step], color="#0284c7", lw=1.1, ls="--", label="Eda_RB (hand)")
            ax2.set_ylabel("Eda_RB", color="#0284c7", fontsize=8)
            ax.set_ylabel("Eda_E4", color="#0ea5e9", fontsize=8)
            ax.legend(loc="upper left", fontsize=8)
        elif i == 6:
            ax.plot(t_ds, sig, color=color, lw=1.0, label="Grip Force")
            ax.set_ylabel("Grip", color=color, fontsize=8)
            ax2 = ax.twinx()
            ax2.plot(pr_t, pr_y, color="#dc2626", marker="o", lw=1.8, ms=5, label="Pain Score (0–100)")
            ax2.set_ylim(0, 105)
            ax2.set_ylabel("Pain (0–100)", color="#dc2626", fontsize=8)
            ax2.legend(loc="upper right", fontsize=8)
        else:
            ax.plot(t_ds, sig, color=color, lw=0.9)
        for pt in pr_t:
            ax.axvline(pt, color="#dc2626", alpha=0.14, lw=0.8)
        ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
    axes[-1].set_xlabel("Recording Time (seconds)")
    fig.suptitle(f"PMPDB Recording {rec_id}: Synchronized 6-Modality Signals & Pain Ratings (Note: No ECG/EEG in PMPDB)", fontsize=12, fontweight="bold", y=1.005)
    fig.savefig(FIGURES_DIR / "01_multimodal_waveforms_P01_1.png")
    plt.close(fig)

    # 2. EDA Tonic/Phasic Decomposition
    tonic_rb, phasic_rb = decompose_eda(cleaned["Eda_RB"], fs=SAMPLING_RATE_HZ)
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)
    axes[0].plot(t_ds, raw_df["Eda_RB"].values[::ds_step], color="#94a3b8", lw=0.8, label="Raw Eda_RB")
    axes[0].plot(t_ds, cleaned["Eda_RB"][::ds_step], color="#0284c7", lw=1.2, label="Cleaned (<5 Hz)")
    axes[0].plot(t_ds, tonic_rb[::ds_step], color="#0f172a", lw=1.5, ls="--", label="Tonic (SCL <0.05 Hz)")
    axes[0].set_title(f"{rec_id}: RespiBAN EDA Cleaning & Tonic (SCL) Extraction", loc="left", fontweight="bold")
    axes[0].legend(fontsize=8)

    axes[1].plot(t_ds, phasic_rb[::ds_step], color="#06b6d4", lw=1.0, label="Phasic Component (SCR)")
    peaks, _ = sp_signal.find_peaks(phasic_rb, distance=int(0.8 * SAMPLING_RATE_HZ), prominence=max(0.015, 0.35 * np.std(phasic_rb)))
    axes[1].scatter(t[peaks], phasic_rb[peaks], color="#ef4444", s=20, zorder=5, label=f"Detected SCR Peaks (n={len(peaks)})")
    axes[1].set_title("Phasic Skin Conductance Responses (SCR)", loc="left", fontweight="bold")
    axes[1].legend(fontsize=8)

    axes[2].step(pr_t, pr_y, where="post", color="#dc2626", lw=2.0, marker="o", ms=5, label="Pain Score (0–100)")
    axes[2].set_ylim(0, 105)
    axes[2].set_ylabel("Pain (0–100)")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_title("Observed Self-Reported Pain Intensity", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_eda_decomposition_P01_1.png")
    plt.close(fig)

    # 3. BVP Systolic Peak Detection (15-second zoom)
    mask_15 = (t >= 45.0) & (t <= 60.0)
    t_z = t[mask_15]
    bvp_e4_z = cleaned["Bvp"][mask_15]
    bvp_rb_z = cleaned["Bvp_RB"][mask_15]
    p_e4, _ = sp_signal.find_peaks(bvp_e4_z, distance=int(0.35 * SAMPLING_RATE_HZ), prominence=0.25 * np.std(bvp_e4_z))
    p_rb, _ = sp_signal.find_peaks(bvp_rb_z, distance=int(0.35 * SAMPLING_RATE_HZ), prominence=0.25 * np.std(bvp_rb_z))

    fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True)
    axes[0].plot(t_z, bvp_e4_z, color="#ef4444", lw=1.2, label="Cleaned Bvp (E4)")
    axes[0].scatter(t_z[p_e4], bvp_e4_z[p_e4], color="#1e293b", s=30, zorder=5, label=f"Systolic Peaks (PR ≈ {60/np.mean(np.diff(t_z[p_e4])):.1f} BPM)")
    axes[0].set_title(f"{rec_id} (45–60 s): Empatica E4 BVP & Systolic Peak Detection", loc="left", fontweight="bold")
    axes[0].legend(fontsize=8)

    axes[1].plot(t_z, bvp_rb_z, color="#f97316", lw=1.2, label="Cleaned Bvp_RB (RespiBAN)")
    axes[1].scatter(t_z[p_rb], bvp_rb_z[p_rb], color="#1e293b", s=30, zorder=5, label=f"Systolic Peaks (PR ≈ {60/np.mean(np.diff(t_z[p_rb])):.1f} BPM)")
    axes[1].set_title("RespiBAN BVP & Systolic Peak Detection (PPG-based PR/PRV; No ECG in PMPDB)", loc="left", fontweight="bold")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_bvp_pulse_detection.png")
    plt.close(fig)

    # 4. Pain Intensity Over Time across 6 representative recordings
    sample_recs = ["P01_1", "P02_1", "P08_1", "P16_1", "P25_1", "P40_1"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.5), sharey=True)
    for ax, r in zip(axes.flat, sample_recs):
        sub = feat_df[feat_df["recording_id"] == r].sort_values("timestamp_sec")
        ax.plot(sub["timestamp_sec"], sub["pain_score_0_100"], marker="o", lw=1.8, ms=4, color="#dc2626")
        ax.axhline(sub["no_pain_threshold"].iloc[0] * 10, color="#10b981", ls="--", lw=1.0, label="No-Pain Thr")
        ax.axhline(sub["severe_pain_threshold"].iloc[0] * 10, color="#f59e0b", ls="--", lw=1.0, label="Severe Thr")
        ax.set_title(f"{r} (n={len(sub)} ratings)", fontweight="bold", fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=7, loc="upper left")
    axes[0, 0].set_ylabel("Pain Intensity (0–100)")
    axes[1, 0].set_ylabel("Pain Intensity (0–100)")
    fig.suptitle("Observed Self-Reported Pain Trajectories Across Representative PMPDB Sessions", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_pain_intensity_over_time.png")
    plt.close(fig)

    # 5. Pain Label & Key Physiological Feature Distributions
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    sns.countplot(x="pain_rate_nrs", data=feat_df, ax=axes[0, 0], color="#ef4444")
    axes[0, 0].set_title("NRS Pain Rating Distribution (1–10)", fontweight="bold")
    axes[0, 0].set_xlabel("Pain Rate (NRS)")

    label_names = feat_df["pain_label"].map({0: "0: No/Low", 1: "1: Moderate", 2: "2: Severe"})
    sns.countplot(x=label_names, order=["0: No/Low", "1: Moderate", "2: Severe"], ax=axes[0, 1], palette=["#10b981", "#f59e0b", "#ef4444"])
    axes[0, 1].set_title("Ordinal Pain Label Distribution", fontweight="bold")
    axes[0, 1].set_xlabel("Pain Class")

    sns.histplot(feat_df["EDA_RB_scl_mean"].clip(0, 35), bins=35, kde=True, ax=axes[0, 2], color="#0284c7")
    axes[0, 2].set_title("RespiBAN EDA Tonic Level (SCL)", fontweight="bold")

    sns.histplot(feat_df["BVP_E4_pr_bpm"].clip(40, 150), bins=30, kde=True, ax=axes[1, 0], color="#ef4444")
    axes[1, 0].set_title("BVP-Derived Pulse Rate (BPM)", fontweight="bold")

    sns.histplot(feat_df["Tmp_mean"], bins=30, kde=True, ax=axes[1, 1], color="#8b5cf6")
    axes[1, 1].set_title("Skin Temperature Distribution (°C)", fontweight="bold")

    sns.histplot(np.log1p(feat_df["EMG_rms"] * 1000), bins=30, kde=True, ax=axes[1, 2], color="#eab308")
    axes[1, 2].set_title("Log Surface EMG RMS (mV)", fontweight="bold")

    fig.suptitle("PMPDB Target & Physiological Signal Distributions (n=1,424 windows)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_signal_and_label_distributions.png")
    plt.close(fig)

    # 6. Feature-Pain Correlations
    corrs = []
    y = feat_df["pain_score_0_100"].values
    for c in feature_cols:
        x = feat_df[c].values
        if np.std(x) > 1e-12:
            r, p = stats.pearsonr(x, y)
            rho, _ = stats.spearmanr(x, y)
            corrs.append({"feature": c, "modality": feat_meta["feature_modality_map"][c], "pearson_r": r, "spearman_rho": rho, "abs_r": abs(r), "p_value": p})
    corr_df = pd.DataFrame(corrs).sort_values("abs_r", ascending=False)
    corr_df.to_csv(METADATA_DIR / "feature_pain_correlations.csv", index=False)

    top20 = corr_df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#ef4444" if v > 0 else "#0284c7" for v in top20["pearson_r"]]
    ax.barh(top20["feature"], top20["pearson_r"], color=colors)
    ax.axvline(0, color="#334155", lw=0.8)
    ax.set_xlabel("Pearson Correlation (r) with Continuous Pain Intensity (0–100)")
    ax.set_title("Top 20 Physiological & Temporal Features Correlated with Pain Intensity", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "06_feature_pain_correlations.png")
    plt.close(fig)

    # 7. Participant-wise variations
    p_stats = feat_df.groupby("participant_id").agg(
        pain_mean=("pain_score_0_100", "mean"),
        pain_std=("pain_score_0_100", "std"),
        eda_mean=("EDA_RB_scl_mean", "mean"),
        tmp_mean=("Tmp_mean", "mean"),
        pr_mean=("BVP_E4_pr_bpm", "mean"),
        n_windows=("window_id", "count"),
    ).reset_index()

    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    x_pos = np.arange(len(p_stats))
    axes[0].bar(x_pos, p_stats["pain_mean"], yerr=p_stats["pain_std"].fillna(0), capsize=2, color="#ef4444", alpha=0.85)
    axes[0].set_ylabel("Pain (0–100)")
    axes[0].set_title("Participant-Wise Pain Intensity (Mean ± SD across all 49 Participants)", loc="left", fontweight="bold")

    axes[1].bar(x_pos, p_stats["eda_mean"], color="#0284c7", alpha=0.85)
    axes[1].set_ylabel("EDA_RB SCL")
    axes[1].set_title("Participant-Wise Baseline Electrodermal Activity (High Inter-Subject Variability)", loc="left", fontweight="bold")

    axes[2].bar(x_pos, p_stats["tmp_mean"], color="#8b5cf6", alpha=0.85)
    axes[2].set_ylim(26, 37)
    axes[2].set_ylabel("Temp (°C)")
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(p_stats["participant_id"], rotation=90, fontsize=8)
    axes[2].set_title("Participant-Wise Mean Skin Temperature (°C)", loc="left", fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "07_participant_variations.png")
    plt.close(fig)

    print(f"Generated 7 EDA figures in {FIGURES_DIR}")
    print("Top 10 correlated features:")
    print(corr_df[["feature", "modality", "pearson_r", "spearman_rho"]].head(10).to_string(index=False))


if __name__ == "__main__":
    generate_all_eda_plots()

