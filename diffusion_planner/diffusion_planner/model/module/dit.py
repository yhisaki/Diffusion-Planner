import math

import torch
import torch.nn as nn
from timm.layers import Mlp


def modulate(x, shift, scale):
    x = x * (1 + scale) + shift
    return x


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning for ego and Cross-Attention.
    """

    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp1 = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.norm3 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm4 = nn.LayerNorm(dim)

        self.mlp2 = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
        )

    def forward(self, x, cross_c, y, attn_mask):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            y
        ).chunk(6, dim=2)

        modulated_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = (
            x
            + gate_msa
            * self.attn(modulated_x, modulated_x, modulated_x, key_padding_mask=attn_mask)[0]
        )

        modulated_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp1(modulated_x)

        x = x + self.cross_attn(self.norm3(x), cross_c, cross_c)[0]
        x = x + self.mlp2(self.norm4(x))

        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4, bias=True),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, output_size, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, y):
        B, P, _ = x.shape

        shift, scale = self.adaLN_modulation(y).chunk(2, dim=2)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.proj(x)
        return x


class DiT(nn.Module):
    def __init__(
        self,
        depth,
        output_dim,
        hidden_dim=192,
        heads=6,
        dropout=0.1,
        mlp_ratio=4.0,
    ):
        super().__init__()

        T = 81
        D = 4
        self.agent_embedding = nn.Embedding(4, hidden_dim)
        self.preproj = Mlp(
            in_features=T * D,
            hidden_features=512,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.t_embedder = Mlp(
            in_features=T,
            hidden_features=512,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.ego_state_proj = nn.Linear(2, hidden_dim)
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, heads, dropout, mlp_ratio) for i in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_dim, output_dim)

    def forward(
        self,
        x,
        t,
        cross_c,
        neighbor_current_mask,
        ego_current_state,
        agent_class,
        current_states,
    ):
        """
        Forward pass of DiT.
        x: (B, P, T, D)   -> Embedded out of DiT
        t: (B, P, T, 1)
        cross_c: (B, N, D)      -> Cross-Attention context
        ego_current_state: (B, 10) -> Ego current state [x, y, cos, sin, vx, vy, ax, ay, steering, yaw_rate]
        agent_class: (B, P) class ids. 0=ego, 1=vehicle, 2=pedestrian, 3=bicycle
        current_states: (B, P, 4) current states used as x0 offset for neighbors
        """
        assert x.dim() == 4, f"{x.dim()=}"
        assert t.dim() == 4, f"{t.dim()=}"
        assert x.shape[2] == t.shape[2], f"{x.shape[2]=} {t.shape[2]=}"
        B, P, T, D = x.shape

        x = x.reshape(B, P, T * D)  # (B, P, T*D)
        t = t.reshape(B, P, T)  # (B, P, T)

        x = self.preproj(x)  # (B, P, hidden_dim)
        t = self.t_embedder(t)  # (B, P, hidden_dim)

        x_embedding = self.agent_embedding(agent_class)  # (B, P, hidden_dim)
        x = x + x_embedding

        ego_vx_ax = ego_current_state[:, [4, 6]]
        x[:, 0] = x[:, 0] + self.ego_state_proj(ego_vx_ax)

        ego_mask = torch.zeros((B, 1), dtype=torch.bool, device=x.device)
        attn_mask = torch.cat([ego_mask, neighbor_current_mask], dim=1)

        for block in self.blocks:
            x = block(x, cross_c, t, attn_mask)

        x = self.final_layer(x, t) # (B, P, output_dim)
        x = x.reshape(B, P, T, D)
        if P > 1 and T > 1:
            neighbor_xy = x[:, 1:, 1:, :2] + current_states[:, 1:, None, :2]
            neighbor_future = torch.cat([neighbor_xy, x[:, 1:, 1:, 2:]], dim=-1)
            neighbors = torch.cat([x[:, 1:, :1], neighbor_future], dim=2)
            x = torch.cat([x[:, :1], neighbors], dim=1)
        return x
