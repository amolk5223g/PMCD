"""Phase 8: Explainable AI (SHAP + Integrated Gradients + Neural Gate Attribution).

Provides:
1. Global & Local SHAP TreeExplainer explanations on trained XGBoost / RandomForest checkpoints
2. PyTorch Integrated Gradients (IG) on MultimodalGatedNet (per-channel, per-modality, and temporal salience)
3. Modality contribution synthesis (combining IG, neural gating weights, and SHAP modality sums)
4. Generates XAI & evaluation comparison figures saved in reports/figures/

Explicitly reports ONLY verified PMPDB modalities (BVP, EDA, Resp, Tmp, EMG, Grip).
Never claims EEG or ECG channel explanations.
"""
import json
from typing import Any, Dict, List, Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

from backend.config import (
    CHECKPOINTS_DIR,
    EXPERIMENTS_DIR,
    FIGURES_DIR,
    MODALITY_GROUPS,
    MODALITY_ORDER,
    PROCESSED_DIR,
    SIGNAL_COLUMNS,
)
from backend.models.deep_models import MODALITY_CHANNEL_INDICES, build_dl_model

plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.bbox"] = "tight"


def compute_integrated_gradients(
    model: torch.nn.Module,
    x_single: torch.Tensor,
    modality_mask: Optional[torch.Tensor] = None,
    n_steps: int = 24,
) -> Dict[str, Any]:
    """Compute Integrated Gradients for a single (1, 8, 300) tensor against zero baseline."""
    model.eval()
    device = next(model.parameters()).device
    x = x_single.to(device).float()
    if x.ndim == 2:
        x = x.unsqueeze(0)
    baseline = torch.zeros_like(x)
    mask = modality_mask.to(device).float() if modality_mask is not None else None

    alphas = torch.linspace(0.0, 1.0, steps=n_steps, device=device)
    total_grads = torch.zeros_like(x)

    for alpha in alphas:
        interp = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        pred_reg, _ = model(interp, modality_mask=mask)
        pred_reg.sum().backward()
        if interp.grad is not None:
            total_grads += interp.grad.detach()

    avg_grads = total_grads / float(n_steps)
    ig = ((x - baseline) * avg_grads).squeeze(0).cpu().numpy()  # (8, 300)

    # Channel-level attribution (mean absolute IG)
    ch_abs = np.mean(np.abs(ig), axis=1)  # (8,)
    ch_signed = np.mean(ig, axis=1)       # (8,)
    total_abs = float(np.sum(ch_abs)) + 1e-12

    channel_attr = {
        SIGNAL_COLUMNS[i]: {
            "abs_importance": round(float(ch_abs[i]), 5),
            "pct_contribution": round(float(100.0 * ch_abs[i] / total_abs), 2),
            "signed_effect": round(float(ch_signed[i]), 5),
        }
        for i in range(len(SIGNAL_COLUMNS))
    }

    # Aggregate to 6 modalities
    mod_attr = {}
    for mod_name, ch_ids in MODALITY_CHANNEL_INDICES:
        val = float(np.sum(ch_abs[ch_ids]))
        sval = float(np.sum(ch_signed[ch_ids]))
        mod_attr[mod_name] = {
            "abs_importance": round(val, 5),
            "pct_contribution": round(float(100.0 * val / total_abs), 2),
            "signed_effect": round(sval, 5),
        }

    # Temporal salience binned into 12 half-second segments across the 6s causal window
    temp_abs = np.sum(np.abs(ig), axis=0)  # (300,)
    bins = temp_abs.reshape(12, 25).mean(axis=1)
    bins_norm = bins / (np.sum(bins) + 1e-12)
    temporal_profile = [
        {"window_offset_sec": round(-6.0 + (i + 0.5) * 0.5, 2), "salience": round(float(bins_norm[i]), 4)}
        for i in range(12)
    ]

    return {
        "channel_attributions": channel_attr,
        "modality_attributions": mod_attr,
        "temporal_salience": temporal_profile,
    }


def explain_local_shap(
    explainer: shap.TreeExplainer,
    x_row: np.ndarray,
    feature_columns: List[str],
    feature_modality_map: Dict[str, str],
    top_k: int = 12,
) -> Dict[str, Any]:
    """Compute exact local SHAP values for one feature vector."""
    arr = np.asarray(x_row, dtype=np.float64).reshape(1, -1)
    sv = explainer.shap_values(arr)
    if isinstance(sv, list):
        sv = sv[0]
    sv = np.asarray(sv).reshape(-1)
    base_val = float(np.asarray(explainer.expected_value).reshape(-1)[0])

    # Top-k features by |SHAP|
    order = np.argsort(np.abs(sv))[::-1]
    top_features = []
    for idx in order[:top_k]:
        fname = feature_columns[idx]
        val = float(arr[0, idx]) if np.isfinite(arr[0, idx]) else None
        top_features.append({
            "feature": fname,
            "modality": feature_modality_map.get(fname, "Temporal"),
            "feature_value": round(val, 4) if val is not None else None,
            "shap_value": round(float(sv[idx]), 4),
            "direction": "increases_pain" if sv[idx] >= 0 else "decreases_pain",
        })

    # Modality-level SHAP aggregation
    mod_abs: Dict[str, float] = {}
    mod_signed: Dict[str, float] = {}
    for idx, fname in enumerate(feature_columns):
        m = feature_modality_map.get(fname, "Temporal")
        mod_abs[m] = mod_abs.get(m, 0.0) + float(abs(sv[idx]))
        mod_signed[m] = mod_signed.get(m, 0.0) + float(sv[idx])
    total_abs = sum(mod_abs.values()) + 1e-12
    modality_shap = {
        m: {
            "abs_shap": round(mod_abs[m], 4),
            "signed_shap": round(mod_signed[m], 4),
            "pct_contribution": round(100.0 * mod_abs[m] / total_abs, 2),
        }
        for m in sorted(mod_abs.keys(), key=lambda k: mod_abs[k], reverse=True)
    }

    return {
        "base_value": round(base_val, 3),
        "shap_sum_prediction": round(float(np.clip(base_val + np.sum(sv), 0.0, 100.0)), 3),
        "top_features": top_features,
        "modality_contributions": modality_shap,
    }


def generate_global_xai_artifacts() -> Dict[str, Any]:
    """Run Phase 8 global SHAP + Integrated Gradients analysis & save JSON + figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    feat_df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    with open(PROCESSED_DIR / "feature_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_cols: List[str] = meta["feature_columns"]
    feat_mod_map: Dict[str, str] = meta["feature_modality_map"]

    # 1. Global SHAP on trained XGBoost checkpoint
    xgb_ckpt = joblib.load(CHECKPOINTS_DIR / "baseline_xgboost.joblib")
    xgb_pipe = xgb_ckpt["pipeline"]
    X = feat_df[feature_cols].values.astype(np.float64)
    X_imp = xgb_pipe.named_steps["imputer"].transform(X)
    xgb_model = xgb_pipe.named_steps["model"]

    explainer = shap.TreeExplainer(xgb_model)
    shap_vals = explainer.shap_values(X_imp)
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

    feat_imp = []
    for idx, fname in enumerate(feature_cols):
        feat_imp.append({
            "feature": fname,
            "modality": feat_mod_map[fname],
            "mean_abs_shap": round(float(mean_abs_shap[idx]), 4),
            "xgb_gain_importance": round(float(xgb_model.feature_importances_[idx]), 5),
        })
    feat_imp.sort(key=lambda d: d["mean_abs_shap"], reverse=True)

    mod_shap: Dict[str, float] = {}
    for item in feat_imp:
        m = item["modality"]
        mod_shap[m] = mod_shap.get(m, 0.0) + item["mean_abs_shap"]
    tot_mod_shap = sum(mod_shap.values()) + 1e-12
    shap_modality_pct = {
        m: round(100.0 * v / tot_mod_shap, 2)
        for m, v in sorted(mod_shap.items(), key=lambda kv: kv[1], reverse=True)
    }

    # 2. Global Integrated Gradients on production MM_Gated checkpoint (sample 120 stratified windows)
    npz = np.load(PROCESSED_DIR / "windows_dl.npz", allow_pickle=True)
    X_dl = npz["X"].astype(np.float32)
    dl_ckpt = torch.load(CHECKPOINTS_DIR / "mm_gated_production.pt", map_location="cpu", weights_only=False)
    dl_model = build_dl_model("MM_Gated", dropout=0.30)
    dl_model.load_state_dict(dl_ckpt["state_dict"])
    dl_model.eval()

    sample_indices = np.linspace(0, len(X_dl) - 1, num=min(120, len(X_dl)), dtype=int)
    ig_mod_accum = {m: 0.0 for m in MODALITY_ORDER}
    ig_ch_accum = {c: 0.0 for c in SIGNAL_COLUMNS}
    gate_accum = np.zeros(len(MODALITY_ORDER), dtype=np.float64)

    for idx in sample_indices:
        xt = torch.from_numpy(X_dl[idx : idx + 1]).float()
        ig_res = compute_integrated_gradients(dl_model, xt, n_steps=16)
        for m in MODALITY_ORDER:
            ig_mod_accum[m] += ig_res["modality_attributions"][m]["abs_importance"]
        for c in SIGNAL_COLUMNS:
            ig_ch_accum[c] += ig_res["channel_attributions"][c]["abs_importance"]
        with torch.no_grad():
            _, _, w = dl_model(xt, return_weights=True)
            gate_accum += w.squeeze(0).numpy()

    ig_mod_tot = sum(ig_mod_accum.values()) + 1e-12
    ig_modality_pct = {m: round(100.0 * ig_mod_accum[m] / ig_mod_tot, 2) for m in MODALITY_ORDER}
    ig_ch_tot = sum(ig_ch_accum.values()) + 1e-12
    ig_channel_pct = {c: round(100.0 * ig_ch_accum[c] / ig_ch_tot, 2) for c in SIGNAL_COLUMNS}
    gate_mean = gate_accum / max(len(sample_indices), 1)
    gate_modality_pct = {m: round(100.0 * float(gate_mean[i]), 2) for i, m in enumerate(MODALITY_ORDER)}

    xai_summary = {
        "note": "Explanations reflect strictly the 6 physiological/biomechanical modalities present in PMPDB (BVP, EDA, Resp, Tmp, EMG, Grip). No EEG or ECG channels exist.",
        "shap_top_25_features": feat_imp[:25],
        "shap_all_features": feat_imp,
        "shap_modality_contribution_pct": shap_modality_pct,
        "integrated_gradients_modality_pct": ig_modality_pct,
        "integrated_gradients_channel_pct": ig_channel_pct,
        "neural_gate_modality_weights_pct": gate_modality_pct,
    }

    with open(EXPERIMENTS_DIR / "xai_global_summary.json", "w", encoding="utf-8") as f:
        json.dump(xai_summary, f, indent=2)

    # Plot 08: Global SHAP Top 20 Features
    top20 = feat_imp[:20][::-1]
    mod_colors = {
        "EMG": "#eab308", "Grip": "#6366f1", "EDA": "#0ea5e9",
        "BVP": "#ef4444", "Resp": "#10b981", "Tmp": "#8b5cf6", "Temporal": "#64748b",
    }
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(
        [d["feature"] for d in top20],
        [d["mean_abs_shap"] for d in top20],
        color=[mod_colors.get(d["modality"], "#64748b") for d in top20],
    )
    ax.set_xlabel("Mean |SHAP Value| (Impact on 0–100 Pain Score)")
    ax.set_title("Global SHAP Feature Importance (Trained XGBoost Regressor on PMPDB)", fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=m) for m, c in mod_colors.items()]
    ax.legend(handles=handles, title="Modality", loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "08_shap_global_importance.png")
    plt.close(fig)

    # Plot 09: Modality Attribution Comparison (SHAP vs Integrated Gradients vs Neural Gates)
    mods = MODALITY_ORDER
    x_pos = np.arange(len(mods))
    w_bar = 0.26
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x_pos - w_bar, [shap_modality_pct.get(m, 0.0) for m in mods], width=w_bar, label="SHAP (XGBoost)", color="#0284c7")
    ax.bar(x_pos, [ig_modality_pct.get(m, 0.0) for m in mods], width=w_bar, label="Integrated Gradients (MM_Gated)", color="#f97316")
    ax.bar(x_pos + w_bar, [gate_modality_pct.get(m, 0.0) for m in mods], width=w_bar, label="Neural Gate Weights (MM_Gated)", color="#10b981")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(mods, fontweight="bold")
    ax.set_ylabel("Modality Contribution (%)")
    ax.set_title("Multimodal Explainability: Modality Contributions Across XAI Methods (PMPDB)", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "09_modality_attribution_comparison.png")
    plt.close(fig)

    print("Global XAI summary saved to", EXPERIMENTS_DIR / "xai_global_summary.json")
    print("SHAP Modality %:", shap_modality_pct)
    print("IG Modality %:  ", ig_modality_pct)
    print("Gate Modality %:", gate_modality_pct)
    return xai_summary


if __name__ == "__main__":
    generate_global_xai_artifacts()

