import torch
import torch.nn as nn
from config import LATENT_DIM, ODE_HIDDEN


class GRUDynamics(nn.Module):
    """
    GRU-based discrete dynamics model. Drop-in replacement for ODEFunc.

    Takes the initial latent vector p0 and steps it forward T-1 times,
    producing a trajectory of shape (T, B, latent_dim) — same as odeint output
    after stripping lat/lon dims, so the training loop is identical to NODE.

    lat/lon are concatenated to the input at each step, mirroring how ODEFunc
    uses them as conditioning signals.

    Input at each step : cat(p_t, lat, lon) — (B, latent_dim + 2)
    GRU hidden state   : (B, hidden_dim)
    Output projection  : hidden -> p_{t+1}  (B, latent_dim)
    """

    def __init__(self, latent_dim=LATENT_DIM, hidden=ODE_HIDDEN):
        super().__init__()
        hidden_dim = hidden[0]   # use first hidden size — GRU is a single layer

        self.gru  = nn.GRUCell(input_size=latent_dim + 2, hidden_size=hidden_dim)
        self.proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, p0, lat, lon, n_steps):
        """
        p0     : (B, latent_dim)  — initial latent state
        lat    : (B,)
        lon    : (B,)
        n_steps: int — number of steps to unroll (T - 1)

        Returns trajectory : (T, B, latent_dim) including p0 as t=0
        """
        B = p0.shape[0]
        lat = lat.unsqueeze(-1)   # (B, 1)
        lon = lon.unsqueeze(-1)   # (B, 1)

        traj = [p0]
        p_t  = p0
        h_t  = torch.zeros(B, self.gru.hidden_size, device=p0.device)

        for _ in range(n_steps):
            inp = torch.cat([p_t, lat, lon], dim=-1)   # (B, latent_dim + 2)
            h_t = self.gru(inp, h_t)                   # (B, hidden_dim)
            p_t = self.proj(h_t)                       # (B, latent_dim)
            traj.append(p_t)

        return torch.stack(traj, dim=0)                # (T, B, latent_dim)
