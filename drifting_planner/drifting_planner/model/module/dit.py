import torch
import torch.nn as nn
from timm.layers.mlp import Mlp


def modulate(x, shift, scale):
    x = x * (1 + scale) + shift
    return x


class DiTBlock(nn.Module):
    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.norm3 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm4 = nn.LayerNorm(dim)

        self.mlp2 = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0
        )

    def forward(self, x, cond, cross_c, attn_mask):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            cond
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

    def forward(self, x, cond):
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=2)
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
        self.agent_embedding = nn.Embedding(2, hidden_dim)
        self.preproj = Mlp(
            in_features=T * D,
            hidden_features=512,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, heads, dropout, mlp_ratio) for i in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_dim, output_dim)

    def forward(self, x, cond, cross_c, neighbor_current_mask):
        assert x.dim() == 4, f"{x.dim()=}"
        B, P, T, D = x.shape

        x = x.reshape(B, P, T * D)

        x = self.preproj(x)

        x_embedding = torch.cat(
            [
                self.agent_embedding.weight[0][None, :],
                self.agent_embedding.weight[1][None, :].expand(P - 1, -1),
            ],
            dim=0,
        )
        x_embedding = x_embedding[None, :, :].expand(B, -1, -1)
        x = x + x_embedding

        attn_mask = torch.zeros((B, P), dtype=torch.bool, device=x.device)
        attn_mask[:, 1:] = neighbor_current_mask

        for block in self.blocks:
            x = block(x, cond, cross_c, attn_mask)

        x = self.final_layer(x, cond)
        x = x.reshape(B, P, T, D)
        return x
