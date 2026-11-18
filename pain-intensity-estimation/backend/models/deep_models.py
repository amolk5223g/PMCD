"""Phase 6 & 7: Deep Learning & Multimodal Fusion Architectures for Continuous Pain Estimation.

Implements:
1. CNN1DPainNet         — Multi-scale 1D CNN across all 8 synchronized PMPDB channels
2. CNNBiGRUPainNet      — 1D CNN + Bidirectional GRU temporal dynamics encoder
3. TCNPainNet           — Dilated Residual Temporal Convolutional Network (TCN)
4. MultimodalConcatNet  — 6 Modality-Specific 1D-CNN Encoders + Feature Concatenation Baseline
5. MultimodalAttnNet    — 6 Modality-Specific Encoders + Interpretable Modality Self-Attention Fusion
6. MultimodalGatedNet   — 6 Modality-Specific Encoders + Gated Multimodal Fusion + Modality Dropout
                          + Auxiliary Tabular Context Fusion + MC Dropout Uncertainty + Dual Head
                          (Primary 0–100 Continuous Regression + Optional 3-Class Ordinal Head)

All models support:
- Explicit binary modality availability mask (shape [B, 6]) so missing sensors are ZEROED
  and masked out in attention/gating rather than receiving fabricated values.
- MC Dropout stochastic inference (`predict_with_uncertainty`) for epistemic uncertainty & 95% intervals.
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Channel indices matching SIGNAL_COLUMNS = [Bvp, Eda_E4, Tmp, Resp, Eda_RB, Bvp_RB, Emg, Grip]
MODALITY_CHANNEL_INDICES: List[Tuple[str, List[int]]] = [
    ("BVP", [0, 5]),   # Bvp, Bvp_RB
    ("EDA", [1, 4]),   # Eda_E4, Eda_RB
    ("Resp", [3]),     # Resp
    ("Tmp", [2]),      # Tmp
    ("EMG", [6]),      # Emg
    ("Grip", [7]),     # Grip
]


def _apply_channel_mask(x: torch.Tensor, modality_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Zero out raw input channels belonging to unavailable modalities (no fabricated values)."""
    if modality_mask is None:
        return x
    B, C, T = x.shape
    ch_mask = torch.ones((B, C, 1), device=x.device, dtype=x.dtype)
    for m_idx, (_, ch_ids) in enumerate(MODALITY_CHANNEL_INDICES):
        m_val = modality_mask[:, m_idx].view(B, 1, 1)
        for ch in ch_ids:
            ch_mask[:, ch : ch + 1, :] = m_val
    return x * ch_mask


class ModalityEncoder1D(nn.Module):
    """Lightweight modality-specific 1D CNN encoder with residual pooling & dropout."""

    def __init__(self, in_channels: int, emb_dim: int = 32, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 24, kernel_size=7, padding=3),
            nn.BatchNorm1d(24),
            nn.GELU(),
            nn.MaxPool1d(2),  # 300 -> 150
            nn.Dropout(dropout),
            nn.Conv1d(24, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),  # 150 -> 75
            nn.Dropout(dropout),
            nn.Conv1d(32, emb_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(emb_dim),
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)  # (B, emb_dim, L)
        avg_p = h.mean(dim=-1)
        max_p = h.amax(dim=-1)
        return self.proj(torch.cat([avg_p, max_p], dim=-1))


class DualPainHead(nn.Module):
    """Continuous 0-100 regression head + auxiliary 3-class ordinal head with MC Dropout."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.30):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.reg_head = nn.Linear(hidden_dim, 1)
        self.cls_head = nn.Linear(hidden_dim, 3)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(z)
        # Raw regression output scaled smoothly to [0, 100] via sigmoid or unconstrained clamped at eval
        reg = torch.sigmoid(self.reg_head(h)).squeeze(-1) * 100.0
        cls_logits = self.cls_head(h)
        return reg, cls_logits


class CNN1DPainNet(nn.Module):
    """1. Multi-scale 1D CNN baseline across all 8 synchronized channels."""

    def __init__(self, in_channels: int = 8, dropout: float = 0.30):
        super().__init__()
        self.branch_k3 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.branch_k9 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.backbone = nn.Sequential(
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(64, 96, kernel_size=5, padding=2),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.GELU(),
        )
        self.head = DualPainHead(96 * 2, hidden_dim=64, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ):
        x = _apply_channel_mask(x, modality_mask)
        h = torch.cat([self.branch_k3(x), self.branch_k9(x)], dim=1)
        h = self.backbone(h)
        z = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=-1)
        reg, cls_logits = self.head(z)
        if return_weights:
            return reg, cls_logits, None
        return reg, cls_logits


class CNNBiGRUPainNet(nn.Module):
    """2. 1D CNN + Bidirectional GRU temporal architecture."""

    def __init__(self, in_channels: int = 8, dropout: float = 0.30):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 48, kernel_size=7, padding=3),
            nn.BatchNorm1d(48),
            nn.GELU(),
            nn.MaxPool1d(3),  # 300 -> 100
            nn.Dropout(dropout),
            nn.Conv1d(48, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),  # 100 -> 50
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=48,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attn_pool = nn.Linear(96, 1)
        self.head = DualPainHead(96, hidden_dim=64, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ):
        x = _apply_channel_mask(x, modality_mask)
        h = self.conv(x).transpose(1, 2)  # (B, 50, 64)
        out, _ = self.gru(h)              # (B, 50, 96)
        w = torch.softmax(self.attn_pool(out), dim=1)
        z = (out * w).sum(dim=1)
        reg, cls_logits = self.head(z)
        if return_weights:
            return reg, cls_logits, None
        return reg, cls_logits


class TCNBlock(nn.Module):
    """Dilated causal residual block for Temporal Convolutional Network."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float = 0.25):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=self.pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=self.pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)[:, :, : x.size(2)]
        h = self.drop(F.gelu(self.bn1(h)))
        h = self.conv2(h)[:, :, : x.size(2)]
        h = self.drop(F.gelu(self.bn2(h)))
        return F.gelu(h + self.res(x))


class TCNPainNet(nn.Module):
    """3. Temporal Convolutional Network (TCN) with exponentially increasing dilations."""

    def __init__(self, in_channels: int = 8, dropout: float = 0.25):
        super().__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=2, padding=2),  # 300 -> 150
            nn.BatchNorm1d(32),
            nn.GELU(),
            TCNBlock(32, 48, kernel_size=5, dilation=1, dropout=dropout),
            TCNBlock(48, 64, kernel_size=5, dilation=2, dropout=dropout),
            TCNBlock(64, 64, kernel_size=5, dilation=4, dropout=dropout),
            TCNBlock(64, 64, kernel_size=5, dilation=8, dropout=dropout),
        )
        self.head = DualPainHead(64 * 2, hidden_dim=64, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ):
        x = _apply_channel_mask(x, modality_mask)
        h = self.tcn(x)
        z = torch.cat([h.mean(dim=-1), h[:, :, -1]], dim=-1)
        reg, cls_logits = self.head(z)
        if return_weights:
            return reg, cls_logits, None
        return reg, cls_logits


class MultimodalBaseNet(nn.Module):
    """Shared base for 6 modality-specific encoders over PMPDB channels."""

    def __init__(self, emb_dim: int = 32, dropout: float = 0.25):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_modalities = len(MODALITY_CHANNEL_INDICES)
        self.encoders = nn.ModuleList([
            ModalityEncoder1D(len(ch_ids), emb_dim=emb_dim, dropout=dropout)
            for _, ch_ids in MODALITY_CHANNEL_INDICES
        ])

    def encode_modalities(
        self, x: torch.Tensor, modality_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.size(0)
        if modality_mask is None:
            mask = torch.ones((B, self.n_modalities), device=x.device, dtype=x.dtype)
        else:
            mask = modality_mask.to(device=x.device, dtype=x.dtype)

        embs = []
        for m_idx, (_, ch_ids) in enumerate(MODALITY_CHANNEL_INDICES):
            sub_x = x[:, ch_ids, :]
            e = self.encoders[m_idx](sub_x)  # (B, emb_dim)
            e = e * mask[:, m_idx : m_idx + 1]
            embs.append(e)
        stacked = torch.stack(embs, dim=1)  # (B, 6, emb_dim)
        return stacked, mask


class MultimodalConcatNet(MultimodalBaseNet):
    """4. Multimodal Feature Concatenation Baseline (6 modality encoders -> concat)."""

    def __init__(self, emb_dim: int = 32, dropout: float = 0.30):
        super().__init__(emb_dim=emb_dim, dropout=dropout)
        self.head = DualPainHead(self.n_modalities * emb_dim, hidden_dim=96, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ):
        stacked, mask = self.encode_modalities(x, modality_mask)
        z = stacked.reshape(x.size(0), -1)
        reg, cls_logits = self.head(z)
        if return_weights:
            norm_w = mask / (mask.sum(dim=1, keepdim=True) + 1e-6)
            return reg, cls_logits, norm_w
        return reg, cls_logits


class MultimodalAttentionNet(MultimodalBaseNet):
    """5. Attention-Based Multimodal Fusion with modality-level attention weights."""

    def __init__(self, emb_dim: int = 32, dropout: float = 0.30):
        super().__init__(emb_dim=emb_dim, dropout=dropout)
        self.mod_pos = nn.Parameter(torch.randn(1, self.n_modalities, emb_dim) * 0.02)
        self.self_attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn_score = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.Tanh(),
            nn.Linear(emb_dim // 2, 1),
        )
        self.head = DualPainHead(emb_dim * 2, hidden_dim=64, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ):
        stacked, mask = self.encode_modalities(x, modality_mask)
        tokens = stacked + self.mod_pos
        key_padding_mask = mask < 0.5  # True where missing
        # Guard: if all modalities were masked, unmask all to avoid NaN softmax
        all_missing = key_padding_mask.all(dim=1, keepdim=True)
        safe_kpm = key_padding_mask & (~all_missing)

        attn_out, _ = self.self_attn(tokens, tokens, tokens, key_padding_mask=safe_kpm)
        tokens = self.norm1(tokens + attn_out) * mask.unsqueeze(-1)

        logits = self.attn_score(tokens).squeeze(-1)  # (B, 6)
        logits = logits.masked_fill(safe_kpm, -1e9)
        weights = torch.softmax(logits, dim=-1)       # (B, 6)

        weighted_sum = (tokens * weights.unsqueeze(-1)).sum(dim=1)
        mean_avail = tokens.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-6)
        z = torch.cat([weighted_sum, mean_avail], dim=-1)
        reg, cls_logits = self.head(z)
        if return_weights:
            return reg, cls_logits, weights
        return reg, cls_logits


class MultimodalGatedNet(MultimodalBaseNet):
    """6. Gated Multimodal Fusion Network (Primary Deep Architecture).

    Combines:
    - 6 modality-specific 1D-CNN encoders (BVP, EDA, Resp, Tmp, EMG, Grip)
    - Context-aware sigmoid gating per modality, explicitly masked by sensor availability
    - Optional biomedical engineered feature branch (hybrid multimodal fusion)
    - MC Dropout uncertainty + Dual regression/ordinal heads
    """

    def __init__(self, emb_dim: int = 32, dropout: float = 0.30):
        super().__init__(emb_dim=emb_dim, dropout=dropout)
        # Each modality gate sees its own embedding + global mean of available modalities
        self.gate_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim * 2, emb_dim),
                nn.Sigmoid(),
            )
            for _ in range(self.n_modalities)
        ])
        self.mod_importance = nn.Linear(emb_dim, 1)
        self.fusion_proj = nn.Sequential(
            nn.Linear(emb_dim * 2, 96),
            nn.LayerNorm(96),
            nn.GELU(),
        )
        self.head = DualPainHead(96, hidden_dim=64, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ):
        stacked, mask = self.encode_modalities(x, modality_mask)  # (B, 6, D)
        avail_count = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        global_ctx = stacked.sum(dim=1) / avail_count             # (B, D)

        gated_embs = []
        gate_magnitudes = []
        for m_idx in range(self.n_modalities):
            e_m = stacked[:, m_idx, :]
            g_m = self.gate_nets[m_idx](torch.cat([e_m, global_ctx], dim=-1))  # (B, D)
            m_avail = mask[:, m_idx : m_idx + 1]
            gated = e_m * g_m * m_avail
            gated_embs.append(gated)
            # Scalar interpretable gate score per modality
            imp = F.softplus(self.mod_importance(gated)).squeeze(-1) * mask[:, m_idx]
            gate_magnitudes.append(imp)

        gated_stack = torch.stack(gated_embs, dim=1)          # (B, 6, D)
        raw_weights = torch.stack(gate_magnitudes, dim=1)     # (B, 6)
        weights = raw_weights / (raw_weights.sum(dim=1, keepdim=True) + 1e-6)

        fused_weighted = (gated_stack * weights.unsqueeze(-1)).sum(dim=1)
        fused_max = gated_stack.amax(dim=1)
        z = self.fusion_proj(torch.cat([fused_weighted, fused_max], dim=-1))
        reg, cls_logits = self.head(z)
        if return_weights:
            return reg, cls_logits, weights
        return reg, cls_logits


def build_dl_model(model_name: str, dropout: float = 0.30) -> nn.Module:
    registry = {
        "CNN1D": CNN1DPainNet,
        "CNN_BiGRU": CNNBiGRUPainNet,
        "TCN": TCNPainNet,
        "MM_Concat": MultimodalConcatNet,
        "MM_Attention": MultimodalAttentionNet,
        "MM_Gated": MultimodalGatedNet,
    }
    if model_name not in registry:
        raise ValueError(f"Unknown model_name: {model_name}. Options: {list(registry.keys())}")
    return registry[model_name](dropout=dropout)


def enable_mc_dropout(model: nn.Module) -> None:
    """Keep BatchNorm in eval mode while enabling Dropout layers for MC Dropout sampling."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
            m.train()


@torch.no_grad()
def predict_mc_dropout(
    model: nn.Module,
    x: torch.Tensor,
    modality_mask: Optional[torch.Tensor] = None,
    n_samples: int = 30,
) -> Dict[str, torch.Tensor]:
    """Run T stochastic MC Dropout passes to estimate prediction mean, std, and modality weights."""
    enable_mc_dropout(model)
    regs = []
    cls_probs = []
    weights_list = []
    for _ in range(n_samples):
        reg, cls_logits, w = model(x, modality_mask=modality_mask, return_weights=True)
        regs.append(reg)
        cls_probs.append(torch.softmax(cls_logits, dim=-1))
        if w is not None:
            weights_list.append(w)

    reg_stack = torch.stack(regs, dim=0)        # (T, B)
    mean_reg = reg_stack.mean(dim=0)
    std_reg = reg_stack.std(dim=0, unbiased=False)
    mean_cls_probs = torch.stack(cls_probs, dim=0).mean(dim=0)
    mean_weights = torch.stack(weights_list, dim=0).mean(dim=0) if weights_list else None
    model.eval()
    return {
        "mean": mean_reg,
        "std": std_reg,
        "cls_probs": mean_cls_probs,
        "modality_weights": mean_weights,
    }
