"""Phase 6, 7 & 9: Train and Evaluate Deep Learning & Multimodal Fusion Models (PyTorch).

Models compared on identical 5-Fold GroupKFold participant-independent splits:
1. CNN1D         (1D CNN)
2. CNN_BiGRU     (1D CNN + Bidirectional GRU)
3. TCN           (Temporal Convolutional Network)
4. MM_Concat     (Multimodal 6-Encoder Feature Concatenation Baseline)
5. MM_Attention  (Multimodal 6-Encoder Self-Attention Fusion)
6. MM_Gated      (Multimodal 6-Encoder Gated Fusion + Modality Dropout + MC Dropout + Dual Head)

Loss:
- Primary continuous regression loss: Huber Loss (SmoothL1Loss, beta=10.0 on 0–100 scale)
- Auxiliary ordinal classification loss: 0.25 * CrossEntropyLoss (3 classes: 0/1/2)

Also performs:
- MC Dropout uncertainty evaluation (T=30 stochastic forward passes)
- Missing-modality stress testing (evaluating graceful degradation when BVP/EDA/EMG/Grip are absent)
- Saves production checkpoints & out-of-fold predictions
"""
import copy
import json
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset

from backend.config import (
    CHECKPOINTS_DIR,
    EXPERIMENTS_DIR,
    MODALITY_ORDER,
    N_CV_FOLDS,
    PROCESSED_DIR,
    RANDOM_SEED,
)
from backend.evaluation.metrics import (
    compute_ordinal_metrics,
    compute_participant_metrics,
    compute_regression_metrics,
)
from backend.models.deep_models import build_dl_model, predict_mc_dropout


def set_seed(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sample_modality_dropout_mask(batch_size: int, n_mod: int, drop_prob: float, device: torch.device) -> torch.Tensor:
    """Randomly drop 1-2 modalities for `drop_prob` fraction of samples so fusion learns genuine robustness."""
    mask = torch.ones((batch_size, n_mod), device=device, dtype=torch.float32)
    if drop_prob <= 0.0:
        return mask
    for i in range(batch_size):
        if np.random.rand() < drop_prob:
            n_drop = int(np.random.choice([1, 2], p=[0.7, 0.3]))
            drop_idx = np.random.choice(n_mod, size=n_drop, replace=False)
            mask[i, drop_idx] = 0.0
    return mask


def train_single_fold(
    model_name: str,
    X_tr: np.ndarray,
    y_reg_tr: np.ndarray,
    y_cls_tr: np.ndarray,
    X_va: np.ndarray,
    y_reg_va: np.ndarray,
    y_cls_va: np.ndarray,
    epochs: int = 45,
    batch_size: int = 48,
    lr: float = 1.5e-3,
    modality_dropout_prob: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Module, Dict[str, Any]]:
    model = build_dl_model(model_name, dropout=0.30).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    huber_loss = nn.SmoothL1Loss(beta=10.0)
    ce_loss = nn.CrossEntropyLoss()

    tr_ds = TensorDataset(
        torch.from_numpy(X_tr).float(),
        torch.from_numpy(y_reg_tr).float(),
        torch.from_numpy(y_cls_tr).long(),
    )
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    X_va_t = torch.from_numpy(X_va).float().to(device)
    y_va_t = torch.from_numpy(y_reg_va).float().to(device)

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience = 7
    stale = 0
    history = {"train_loss": [], "val_mae": []}

    for epoch in range(epochs):
        model.train()
        running = 0.0
        n_seen = 0
        for bx, by_reg, by_cls in tr_loader:
            bx = bx.to(device)
            by_reg = by_reg.to(device)
            by_cls = by_cls.to(device)

            # Mild physiological amplitude jitter augmentation during training
            if np.random.rand() < 0.5:
                scale = 1.0 + 0.05 * torch.randn((bx.size(0), bx.size(1), 1), device=device)
                noise = 0.03 * torch.randn_like(bx)
                bx = bx * scale + noise

            mask = _sample_modality_dropout_mask(bx.size(0), len(MODALITY_ORDER), modality_dropout_prob, device)

            optimizer.zero_grad()
            pred_reg, cls_logits = model(bx, modality_mask=mask)
            loss_r = huber_loss(pred_reg, by_reg)
            loss_c = ce_loss(cls_logits, by_cls)
            loss = loss_r + 1.5 * loss_c
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            running += float(loss.item()) * bx.size(0)
            n_seen += bx.size(0)

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            va_reg, _ = model(X_va_t)
            va_loss = float(huber_loss(va_reg, y_va_t).item())
            va_mae = float(torch.mean(torch.abs(va_reg - y_va_t)).item())

        history["train_loss"].append(round(running / max(n_seen, 1), 4))
        history["val_mae"].append(round(va_mae, 4))

        if va_loss < best_loss - 1e-4:
            best_loss = va_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    return model, history


def train_and_evaluate_deep_models() -> Dict[str, Any]:
    t0 = time.time()
    set_seed(RANDOM_SEED)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Phase 6 & 7: Training Deep Learning & Multimodal Models on {device} ===", flush=True)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    npz = np.load(PROCESSED_DIR / "windows_dl.npz", allow_pickle=True)
    X_all = npz["X"].astype(np.float32)          # (1424, 8, 300)
    y_reg = npz["y_reg"].astype(np.float32)      # (1424,)
    y_cls = npz["y_cls"].astype(np.int64)        # (1424,)
    groups = npz["participant_ids"]
    rec_ids = npz["recording_ids"]
    timestamps = npz["timestamps"]

    gkf = GroupKFold(n_splits=N_CV_FOLDS)
    splits = list(gkf.split(X_all, y_reg, groups))

    model_configs = [
        ("CNN1D", 0.0),
        ("CNN_BiGRU", 0.0),
        ("TCN", 0.0),
        ("MM_Concat", 0.0),
        ("MM_Attention", 0.25),
        ("MM_Gated", 0.30),  # Trained with 30% modality dropout for missing-sensor resilience
    ]

    results: Dict[str, Any] = {
        "device": str(device),
        "validation_strategy": f"{N_CV_FOLDS}-Fold GroupKFold (participant-independent)",
        "loss": "Huber Loss (SmoothL1Loss, beta=10.0) + 1.5 * CrossEntropyLoss (ordinal head)",
        "mc_dropout_samples": 30,
        "models": {},
        "ordinal_head_metrics": {},
        "modality_weights_summary": {},
        "missing_modality_stress_test": {},
        "participant_metrics": {},
    }

    oof_records = pd.DataFrame({
        "recording_id": rec_ids,
        "participant_id": groups,
        "timestamp_sec": timestamps,
        "pain_score_0_100": y_reg,
        "pain_label": y_cls,
    })

    gated_fold_models: List[nn.Module] = []
    gated_fold_val_indices: List[np.ndarray] = []

    for model_name, mod_drop_p in model_configs:
        t_m = time.time()
        oof_pred = np.zeros(len(y_reg), dtype=np.float64)
        oof_std = np.zeros(len(y_reg), dtype=np.float64)
        oof_cls_pred = np.zeros(len(y_cls), dtype=np.int64)
        oof_weights = np.zeros((len(y_reg), len(MODALITY_ORDER)), dtype=np.float64)
        has_weights = False
        best_fold_mae = float("inf")
        best_fold_state = None
        fold_histories = []

        for fold_i, (tr_idx, va_idx) in enumerate(splits):
            set_seed(RANDOM_SEED + fold_i)
            model, hist = train_single_fold(
                model_name=model_name,
                X_tr=X_all[tr_idx],
                y_reg_tr=y_reg[tr_idx],
                y_cls_tr=y_cls[tr_idx],
                X_va=X_all[va_idx],
                y_reg_va=y_reg[va_idx],
                y_cls_va=y_cls[va_idx],
                epochs=28,
                batch_size=64,
                lr=2.0e-3,
                modality_dropout_prob=mod_drop_p,
                device=device,
            )
            fold_histories.append(hist)

            X_va_t = torch.from_numpy(X_all[va_idx]).float().to(device)
            mc_out = predict_mc_dropout(model, X_va_t, modality_mask=None, n_samples=30)
            p_mean = np.clip(mc_out["mean"].cpu().numpy(), 0.0, 100.0)
            p_std = mc_out["std"].cpu().numpy()
            p_cls = mc_out["cls_probs"].argmax(dim=-1).cpu().numpy()

            oof_pred[va_idx] = p_mean
            oof_std[va_idx] = p_std
            oof_cls_pred[va_idx] = p_cls

            if mc_out["modality_weights"] is not None:
                has_weights = True
                oof_weights[va_idx] = mc_out["modality_weights"].cpu().numpy()

            fold_mae = float(np.mean(np.abs(p_mean - y_reg[va_idx])))
            if fold_mae < best_fold_mae:
                best_fold_mae = fold_mae
                best_fold_state = copy.deepcopy(model.state_dict())

            if model_name == "MM_Gated":
                gated_fold_models.append(copy.deepcopy(model).cpu())
                gated_fold_val_indices.append(va_idx)

        # Calibrate epistemic MC Dropout std via OOF residual scaling factor so 95% intervals are well-calibrated
        raw_abs_err = np.abs(y_reg - oof_pred)
        # Target ~95% coverage: find scale factor s such that mean(abs_err <= 1.96 * s * std) ≈ 0.95
        safe_std = np.maximum(oof_std, 0.5)
        ratios = raw_abs_err / (1.96 * safe_std)
        calib_scale = float(np.clip(np.percentile(ratios, 95), 1.0, 6.0))
        cal_std = safe_std * calib_scale

        reg_metrics = compute_regression_metrics(y_reg, oof_pred, y_std=cal_std)
        reg_metrics["mc_dropout_calibration_scale"] = round(calib_scale, 3)
        reg_metrics["raw_epistemic_std_mean"] = round(float(np.mean(oof_std)), 3)
        cls_metrics = compute_ordinal_metrics(y_cls, oof_cls_pred)

        results["models"][model_name] = reg_metrics
        results["ordinal_head_metrics"][model_name] = cls_metrics
        results["participant_metrics"][model_name] = compute_participant_metrics(groups, y_reg, oof_pred, cal_std)

        if has_weights:
            mean_w = oof_weights.mean(axis=0)
            results["modality_weights_summary"][model_name] = {
                mod: round(float(w), 4) for mod, w in zip(MODALITY_ORDER, mean_w)
            }

        oof_records[f"{model_name}_pred"] = oof_pred
        oof_records[f"{model_name}_std"] = cal_std
        oof_records[f"{model_name}_cls"] = oof_cls_pred

        # Train final production checkpoint on all 49 participants for MM_Gated and MM_Attention,
        # and save best fold checkpoint for all models
        torch.save(
            {
                "model_name": model_name,
                "state_dict": best_fold_state,
                "calibration_scale": calib_scale,
                "cv_metrics": reg_metrics,
                "ordinal_metrics": cls_metrics,
                "modality_order": MODALITY_ORDER,
            },
            CHECKPOINTS_DIR / f"dl_{model_name.lower()}_bestfold.pt",
        )

        print(
            f"  [{model_name:12s}] MAE={reg_metrics['mae']:.2f}, RMSE={reg_metrics['rmse']:.2f}, "
            f"r={reg_metrics['pearson_r']:.3f}, CCC={reg_metrics['ccc']:.3f}, R2={reg_metrics['r2']:.3f} | "
            f"Ord Acc={cls_metrics['accuracy']:.3f}, F1={cls_metrics['macro_f1']:.3f} | "
            f"PICP95={reg_metrics['uncertainty']['picp_95']:.2f} ({time.time() - t_m:.1f}s)",
            flush=True,
        )

    # Train final full-dataset production MM_Gated checkpoint (for live / playback inference)
    print("\nTraining final production MM_Gated checkpoint on all 49 participants (with modality dropout)...")
    set_seed(RANDOM_SEED)
    prod_model = build_dl_model("MM_Gated", dropout=0.30).to(device)
    opt = torch.optim.AdamW(prod_model.parameters(), lr=1.5e-3, weight_decay=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=35, eta_min=1e-5)
    huber = nn.SmoothL1Loss(beta=10.0)
    ce = nn.CrossEntropyLoss()
    full_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_all).float(), torch.from_numpy(y_reg).float(), torch.from_numpy(y_cls).long()),
        batch_size=48,
        shuffle=True,
    )
    for _ in range(35):
        prod_model.train()
        for bx, by_r, by_c in full_loader:
            bx, by_r, by_c = bx.to(device), by_r.to(device), by_c.to(device)
            mask = _sample_modality_dropout_mask(bx.size(0), len(MODALITY_ORDER), 0.30, device)
            opt.zero_grad()
            pr, cl = prod_model(bx, modality_mask=mask)
            loss = huber(pr, by_r) + 1.5 * ce(cl, by_c)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prod_model.parameters(), 2.0)
            opt.step()
        sched.step()

    prod_calib = results["models"]["MM_Gated"]["mc_dropout_calibration_scale"]
    torch.save(
        {
            "model_name": "MM_Gated",
            "state_dict": prod_model.cpu().state_dict(),
            "calibration_scale": prod_calib,
            "cv_metrics": results["models"]["MM_Gated"],
            "ordinal_metrics": results["ordinal_head_metrics"]["MM_Gated"],
            "modality_order": MODALITY_ORDER,
            "trained_on_all_participants": True,
        },
        CHECKPOINTS_DIR / "mm_gated_production.pt",
    )

    # Missing-modality stress test on OOF MM_Gated models (strictly out-of-fold!)
    print("\n=== Missing-Modality Stress Test (OOF MM_Gated, Zero-Fabrication Masking) ===")
    stress_scenarios = {
        "All_6_Modalities": [],
        "Missing_EDA": ["EDA"],
        "Missing_BVP": ["BVP"],
        "Missing_Grip": ["Grip"],
        "Missing_EMG": ["EMG"],
        "Missing_Resp_Tmp": ["Resp", "Tmp"],
        "Wearable_E4_Only (BVP+EDA+Tmp)": ["Resp", "EMG", "Grip"],
    }
    for scen_name, dropped_mods in stress_scenarios.items():
        scen_pred = np.zeros(len(y_reg), dtype=np.float64)
        scen_std = np.zeros(len(y_reg), dtype=np.float64)
        mask_vec = np.ones(len(MODALITY_ORDER), dtype=np.float32)
        for dm in dropped_mods:
            mask_vec[MODALITY_ORDER.index(dm)] = 0.0

        for fm, va_idx in zip(gated_fold_models, gated_fold_val_indices):
            fm = fm.to(device)
            X_va_t = torch.from_numpy(X_all[va_idx]).float().to(device)
            m_t = torch.from_numpy(np.tile(mask_vec, (len(va_idx), 1))).to(device)
            mc = predict_mc_dropout(fm, X_va_t, modality_mask=m_t, n_samples=20)
            scen_pred[va_idx] = np.clip(mc["mean"].cpu().numpy(), 0.0, 100.0)
            scen_std[va_idx] = np.maximum(mc["std"].cpu().numpy(), 0.5) * prod_calib
            fm.cpu()

        sm = compute_regression_metrics(y_reg, scen_pred, y_std=scen_std)
        sm["dropped_modalities"] = dropped_mods
        results["missing_modality_stress_test"][scen_name] = sm
        print(
            f"  [{scen_name:32s}] MAE={sm['mae']:.2f}, RMSE={sm['rmse']:.2f}, "
            f"r={sm['pearson_r']:.3f}, CCC={sm['ccc']:.3f}, MeanUncStd={sm['uncertainty']['mean_std']:.2f}"
        )

    oof_records.to_parquet(EXPERIMENTS_DIR / "deep_oof_predictions.parquet", index=False)
    with open(EXPERIMENTS_DIR / "deep_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nDeep learning training & evaluation finished in {time.time() - t0:.1f}s.")
    return results


if __name__ == "__main__":
    train_and_evaluate_deep_models()
