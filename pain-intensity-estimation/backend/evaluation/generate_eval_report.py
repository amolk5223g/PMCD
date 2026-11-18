"""Phase 9: Hybrid OOF Ensemble Evaluation, Comparison Plots, and Model Evaluation Report."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backend.config import EXPERIMENTS_DIR, FIGURES_DIR, REPORTS_DIR
from backend.evaluation.metrics import compute_participant_metrics, compute_regression_metrics

plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.bbox"] = "tight"


def generate_evaluation_report() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(EXPERIMENTS_DIR / "baseline_results.json", "r", encoding="utf-8") as f:
        base_res = json.load(f)
    with open(EXPERIMENTS_DIR / "deep_results.json", "r", encoding="utf-8") as f:
        deep_res = json.load(f)

    base_oof = pd.read_parquet(EXPERIMENTS_DIR / "baseline_oof_predictions.parquet")
    deep_oof = pd.read_parquet(EXPERIMENTS_DIR / "deep_oof_predictions.parquet")

    # Verify exact row alignment across OOF predictions
    assert len(base_oof) == len(deep_oof), "OOF alignment mismatch!"
    y_true = base_oof["pain_score_0_100"].values.astype(np.float64)
    groups = base_oof["participant_id"].values

    # Strictly Out-of-Fold Hybrid Ensemble (0.70 * MM_Gated + 0.30 * XGBoost)
    p_dl = deep_oof["MM_Gated_pred"].values.astype(np.float64)
    std_dl = deep_oof["MM_Gated_std"].values.astype(np.float64)
    p_xgb = base_oof["XGBoost_gkf_pred"].values.astype(np.float64)

    p_hyb = np.clip(0.70 * p_dl + 0.30 * p_xgb, 0.0, 100.0)
    std_hyb = np.sqrt(std_dl ** 2 + (0.35 * np.abs(p_dl - p_xgb)) ** 2)

    hyb_metrics = compute_regression_metrics(y_true, p_hyb, y_std=std_hyb)
    hyb_part = compute_participant_metrics(groups, y_true, p_hyb, std_hyb)

    hyb_out = {
        "model": "Hybrid_MM_Gated_XGB",
        "formula": "0.70 * MM_Gated (deep waveforms) + 0.30 * XGBoost (88 engineered features)",
        "validation": "5-Fold GroupKFold (strictly out-of-fold, identical participant splits)",
        "metrics": hyb_metrics,
        "participant_metrics": hyb_part,
    }
    with open(EXPERIMENTS_DIR / "hybrid_results.json", "w", encoding="utf-8") as f:
        json.dump(hyb_out, f, indent=2)

    # Assemble unified comparison table
    rows = []
    for name, m in base_res["groupkfold"].items():
        loso = base_res["loso"][name]
        rows.append({
            "Model": name,
            "Family": "Baseline ML",
            "MAE": m["mae"],
            "RMSE": m["rmse"],
            "Pearson_r": m["pearson_r"],
            "CCC": m["ccc"],
            "R2": m["r2"],
            "LOSO_MAE": loso["mae"],
            "LOSO_r": loso["pearson_r"],
            "LOSO_CCC": loso["ccc"],
        })
    for name, m in deep_res["models"].items():
        rows.append({
            "Model": name,
            "Family": "Multimodal DL" if name.startswith("MM_") else "Deep Learning",
            "MAE": m["mae"],
            "RMSE": m["rmse"],
            "Pearson_r": m["pearson_r"],
            "CCC": m["ccc"],
            "R2": m["r2"],
            "LOSO_MAE": None,
            "LOSO_r": None,
            "LOSO_CCC": None,
        })
    rows.append({
        "Model": "Hybrid_MM_Gated_XGB",
        "Family": "Hybrid Ensemble",
        "MAE": hyb_metrics["mae"],
        "RMSE": hyb_metrics["rmse"],
        "Pearson_r": hyb_metrics["pearson_r"],
        "CCC": hyb_metrics["ccc"],
        "R2": hyb_metrics["r2"],
        "LOSO_MAE": None,
        "LOSO_r": None,
        "LOSO_CCC": None,
    })
    comp_df = pd.DataFrame(rows).sort_values("MAE")

    # Plot 10: Model Comparison (MAE & Pearson r / CCC)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fam_colors = {
        "Baseline ML": "#0284c7",
        "Deep Learning": "#f97316",
        "Multimodal DL": "#10b981",
        "Hybrid Ensemble": "#7c3aed",
    }
    colors = [fam_colors[f] for f in comp_df["Family"]]

    axes[0].barh(comp_df["Model"][::-1], comp_df["MAE"][::-1], color=colors[::-1])
    axes[0].set_xlabel("MAE (0–100 Scale, lower is better)")
    axes[0].set_title("5-Fold GroupKFold MAE Comparison", fontweight="bold")

    y_pos = np.arange(len(comp_df))
    axes[1].barh(y_pos + 0.18, comp_df["Pearson_r"][::-1], height=0.35, label="Pearson r", color="#10b981")
    axes[1].barh(y_pos - 0.18, comp_df["CCC"][::-1], height=0.35, label="Lin's CCC", color="#6366f1")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(comp_df["Model"][::-1])
    axes[1].set_xlabel("Correlation / Concordance (higher is better)")
    axes[1].set_title("Pearson r & Lin's CCC Comparison", fontweight="bold")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "10_model_comparison.png")
    plt.close(fig)

    # Plot 11: Modality Ablation & Missing-Modality Stress Test
    ab = base_res["modality_ablation"]
    st = deep_res["missing_modality_stress_test"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))

    ab_names = list(ab.keys())
    axes[0].bar(range(len(ab_names)), [ab[k]["mae"] for k in ab_names], color="#0284c7")
    axes[0].set_xticks(range(len(ab_names)))
    axes[0].set_xticklabels(ab_names, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("MAE (0–100)")
    axes[0].set_title("Single-Modality vs. Multimodal Ablation (XGBoost)", fontweight="bold")

    st_names = [k.replace("Wearable_E4_Only (BVP+EDA+Tmp)", "E4_Only") for k in st.keys()]
    axes[1].bar(range(len(st_names)), [st[k]["mae"] for k in st.keys()], color="#f97316")
    axes[1].set_xticks(range(len(st_names)))
    axes[1].set_xticklabels(st_names, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("MAE (0–100)")
    axes[1].set_title("Missing-Modality Stress Test (OOF MM_Gated)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "11_modality_ablation_and_stress.png")
    plt.close(fig)

    # Plot 12: Participant-wise MAE across all 49 participants
    p_df = pd.DataFrame(hyb_part).sort_values("mae")
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.bar(p_df["participant_id"], p_df["mae"], color="#0ea5e9")
    ax.axhline(hyb_metrics["mae"], color="#dc2626", ls="--", lw=1.5, label=f"Overall Mean MAE = {hyb_metrics['mae']:.2f}")
    ax.set_ylabel("MAE (0–100)")
    ax.set_xlabel("Participant ID (sorted by MAE)")
    ax.set_title("Participant-Independent Out-of-Fold MAE Across All 49 PMPDB Participants (Hybrid Ensemble)", fontweight="bold")
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "12_participant_mae_distribution.png")
    plt.close(fig)

    # Generate Model Evaluation Report Markdown
    md = [
        "# Model Evaluation Report — Continuous Pain-Intensity Estimation (PMPDB)",
        "",
        "## 1. Validation Protocol (Zero Data Leakage)",
        "- **Dataset**: PMPDB (49 participants, 90 main sessions, 1,424 verified numeric NRS pain ratings linearly mapped to 0–100).",
        "- **Splitting**: Strict participant-independent **5-Fold GroupKFold** (all 10 models evaluated on identical participant folds) plus **49-Fold Leave-One-Subject-Out (LOSO)** for baseline ML models.",
        "- **Zero Leakage Normalization**: Baseline relative features and deep-learning channel scaling use strictly each recording's own pre-session `runUp` calibration file; tabular imputers/scalers are fit exclusively inside training folds.",
        "- **Loss Function**: Huber Loss (`SmoothL1Loss`, $\\beta = 10.0$) for continuous regression + auxiliary 3-class ordinal Cross-Entropy head for neural models.",
        "",
        "## 2. Complete Model Comparison (5-Fold GroupKFold & 49-Fold LOSO)",
        "| Model | Family | MAE ↓ | RMSE ↓ | Pearson r ↑ | Lin's CCC ↑ | R² ↑ | LOSO MAE | LOSO r |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in comp_df.iterrows():
        l_mae = f"{r['LOSO_MAE']:.2f}" if pd.notna(r["LOSO_MAE"]) else "—"
        l_r = f"{r['LOSO_r']:.3f}" if pd.notna(r["LOSO_r"]) else "—"
        md.append(
            f"| **{r['Model']}** | {r['Family']} | {r['MAE']:.2f} | {r['RMSE']:.2f} | "
            f"{r['Pearson_r']:.3f} | {r['CCC']:.3f} | {r['R2']:.3f} | {l_mae} | {l_r} |"
        )

    md += [
        "",
        "## 3. Single-Modality vs. Multimodal Ablation (XGBoost, 5-Fold GroupKFold)",
        "| Modality Subset | Features | MAE ↓ | RMSE ↓ | Pearson r ↑ | Lin's CCC ↑ | R² ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for k, m in ab.items():
        md.append(
            f"| `{k}` | {m['n_features']} | {m['mae']:.2f} | {m['rmse']:.2f} | "
            f"{m['pearson_r']:.3f} | {m['ccc']:.3f} | {m['r2']:.3f} |"
        )

    md += [
        "",
        "## 4. Missing-Modality Resilience Stress Test (OOF `MM_Gated`, Zero Fabrication)",
        "| Scenario | Dropped Modalities | MAE ↓ | RMSE ↓ | Pearson r ↑ | Mean Uncertainty σ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for k, m in st.items():
        dropped = ", ".join(m["dropped_modalities"]) if m["dropped_modalities"] else "None"
        md.append(
            f"| `{k}` | {dropped} | {m['mae']:.2f} | {m['rmse']:.2f} | "
            f"{m['pearson_r']:.3f} | {m['uncertainty']['mean_std']:.2f} |"
        )

    md += [
        "",
        "## 5. Uncertainty Quantification (Calibrated MC Dropout, T = 30)",
        f"- **MM_Gated 95% Prediction Interval Coverage (PICP)**: **{deep_res['models']['MM_Gated']['uncertainty']['picp_95']*100:.1f}%** (Mean Interval Width: {deep_res['models']['MM_Gated']['uncertainty']['mpiw_95']:.1f} pts).",
        f"- **Hybrid Ensemble 95% Prediction Interval Coverage (PICP)**: **{hyb_metrics['uncertainty']['picp_95']*100:.1f}%** (Mean Interval Width: {hyb_metrics['uncertainty']['mpiw_95']:.1f} pts).",
        "",
        "## 6. Participant-Wise Generalization Summary (49 Participants, Hybrid Ensemble)",
        f"- **Best-generalizing participants (lowest MAE)**: {', '.join(f'{r.participant_id} ({r.mae:.1f})' for r in p_df.head(5).itertuples())}.",
        f"- **Hardest participants (highest MAE)**: {', '.join(f'{r.participant_id} ({r.mae:.1f})' for r in p_df.tail(5).itertuples())}.",
        f"- **Median Participant MAE**: {p_df['mae'].median():.2f} (IQR: {p_df['mae'].quantile(0.25):.2f}–{p_df['mae'].quantile(0.75):.2f}).",
        "",
        "## 7. Honest Scientific Assessment & Limitations",
        "- Inter-subject pain perception is inherently subjective: identical physiological activation maps to different self-reported NRS scores across individuals (reflected in PMPDB's wide range of individualized `noPainThreshold` [1–5] and `severePainThreshold` [3–9]).",
        "- Without participant-specific calibration ratings from the target session, participant-independent regression achieves meaningful positive correlation and tracks relative pain escalation, while exhibiting realistic ~14–19 point MAE on the 0–100 scale.",
        "- This system is an academic research prototype and **must not** be used for clinical diagnosis or pain medication decisions.",
    ]

    (REPORTS_DIR / "model_evaluation_report.md").write_text("\n".join(md), encoding="utf-8")
    print("Saved hybrid_results.json, 3 evaluation figures, and reports/model_evaluation_report.md")
    print(comp_df[["Model", "Family", "MAE", "RMSE", "Pearson_r", "CCC", "R2"]].to_string(index=False))


if __name__ == "__main__":
    generate_evaluation_report()

