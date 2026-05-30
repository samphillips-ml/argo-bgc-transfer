import torch
import torch.nn as nn
from config import LATENT_DIM, DECODER_HIDDEN, TARGET_VARS


class OxygenDecoderHead(nn.Module):
    """
    Frozen-encoder probe decoder for TARGET_VARS (default: Oxygen).

    Architecture mirrors the T/S decoder — depth in meters concatenated
    to latent vector p at every level, MLP maps to len(TARGET_VARS) outputs.

    Predicts in original physical units (no normalization) so that loss
    and metrics are interpretable (e.g. µmol/kg for oxygen).
    """

    def __init__(self, latent_dim=LATENT_DIM, hidden=DECODER_HIDDEN):
        super().__init__()
        n_out = len(TARGET_VARS)

        layers = []
        in_dim = latent_dim + 1    # +1 for depth in meters
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, n_out)]

        self.mlp = nn.Sequential(*layers)

    def forward(self, p, depth_levels):
        """
        p            : (batch, latent_dim)
        depth_levels : (depth,) float tensor — DEPTH_GRID in meters
        returns      : (batch, depth, n_target_vars)
        """
        batch = p.shape[0]
        depth = depth_levels.shape[0]

        p_expanded = p.unsqueeze(1).expand(-1, depth, -1)         # (B, D, latent_dim)
        d = depth_levels.view(1, -1, 1).expand(batch, -1, -1)     # (B, D, 1)
        inp = torch.cat([p_expanded, d], dim=-1)                  # (B, D, latent_dim+1)

        return self.mlp(inp)                                       # (B, D, n_target_vars)