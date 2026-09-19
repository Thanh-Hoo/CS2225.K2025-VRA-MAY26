import torch
import torch.nn as nn


class AdaptiveModalityWeighting(nn.Module):
    """Predicts a per-sample scalar weight for RGB/NIR/TIR local features.

    Given the three modalities' global (CLS) representations g_rgb, g_nir, g_tir
    (each [B, C]), a shared scoring network produces one score per modality, which
    is turned into a softmax distribution p (sums to 1) and then rescaled to
    alpha = 3 * p (sums to 3), so that alpha == [1, 1, 1] reproduces the unweighted
    IDEA baseline exactly.
    """

    def __init__(self, feat_dim, reduction_ratio=8, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        hidden_dim = max(feat_dim // reduction_ratio, 32)
        self.gate = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # Zero-init the final layer so scores start at 0 -> softmax is uniform
        # -> alpha starts at [1, 1, 1], matching the IDEA baseline at epoch 0.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, g_rgb, g_nir, g_tir):
        """
        Args:
            g_rgb, g_nir, g_tir: [B, C] global features for each modality.
        Returns:
            alpha_rgb, alpha_nir, alpha_tir: [B] scaled coefficients (sum to 3 per sample).
            alpha: [B, 3] stacked coefficients, in (rgb, nir, tir) order.
        """
        stacked = torch.stack([g_rgb, g_nir, g_tir], dim=1)  # [B, 3, C]
        scores = self.gate(stacked).squeeze(-1)  # [B, 3]
        p = torch.softmax(scores / self.temperature, dim=1)  # [B, 3], sums to 1
        alpha = 3.0 * p  # [B, 3], sums to 3
        alpha_rgb, alpha_nir, alpha_tir = alpha.unbind(dim=1)
        return alpha_rgb, alpha_nir, alpha_tir, alpha


def select_alpha(gate, g_rgb, g_nir, g_tir, force_uniform=False, static_weights=()):
    """Ablation-aware alpha selection, shared by IDEA.apply_adaptive_weighting and
    by tests (kept dependency-free so it is unit-testable without the rest of IDEA).

    - static_weights (3 floats): Ablation D, fixed non-learned coefficients.
    - force_uniform=True: Ablation B, bypass the gate, alpha == [1, 1, 1].
    - otherwise: Ablation C, alpha = 3 * softmax(gate(features) / temperature).

    Returns (alpha_rgb, alpha_nir, alpha_tir), each [B].
    """
    batch_size = g_rgb.size(0)
    device = g_rgb.device
    dtype = g_rgb.dtype
    if len(static_weights) == 3:
        static = torch.tensor(static_weights, device=device, dtype=dtype)
        return static[0].expand(batch_size), static[1].expand(batch_size), static[2].expand(batch_size)
    if force_uniform:
        ones = torch.ones(batch_size, device=device, dtype=dtype)
        return ones, ones, ones
    alpha_rgb, alpha_nir, alpha_tir, _ = gate(g_rgb, g_nir, g_tir)
    return alpha_rgb, alpha_nir, alpha_tir
