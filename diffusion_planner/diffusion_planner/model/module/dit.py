import torch
import torch.nn as nn
from timm.layers import Mlp


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    x = x * (1 + scale) + shift
    return x


def _approx_gelu() -> nn.Module:
    return nn.GELU(approximate="tanh")


class DiTBlock(nn.Module):
    """
    A DiT block with agent self-attention, timestep conditioning, and encoder cross-attention.
    """

    def __init__(
        self,
        dim: int = 192,
        heads: int = 6,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=_approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.norm3 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm4 = nn.LayerNorm(dim)

        self.mlp2 = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=_approx_gelu,
            drop=0,
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        self_attn_mask: torch.Tensor,
        cross_c: torch.Tensor,
        cross_c_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Agent trajectory tokens, shape (B, P, hidden_dim).
            y: Diffusion-time conditioning, shape (B, P, hidden_dim).
            self_attn_mask: Agent-to-agent attention mask, shape (B * num_heads, P, P).
                True blocks attention from a query row to a key column.
            cross_c: Encoder context tokens, shape (B, N, hidden_dim).
            cross_c_mask: Encoder context mask, shape (B, N). True means invalid.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            y
        ).chunk(6, dim=2)

        modulated_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = (
            x
            + gate_msa
            * self.attn(
                modulated_x,
                modulated_x,
                modulated_x,
                attn_mask=self_attn_mask,
            )[0]
        )

        modulated_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp1(modulated_x)

        x = x + self.cross_attn(self.norm3(x), cross_c, cross_c, key_padding_mask=cross_c_mask)[0]
        x = x + self.mlp2(self.norm4(x))

        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4, bias=True),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, output_size, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Apply the final adaLN-conditioned projection to trajectory tokens.

        Args:
            x: Agent trajectory tokens, shape (B, P, hidden_size).
            y: Diffusion-time conditioning, shape (B, P, hidden_size).

        Returns:
            Per-agent trajectory output, shape (B, P, output_size).
        """
        shift, scale = self.adaLN_modulation(y).chunk(2, dim=2)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.proj(x)
        return x


class DiT(nn.Module):
    def __init__(
        self,
        depth: int,
        output_dim: int,
        hidden_dim: int = 192,
        heads: int = 6,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
        trajectory_len: int = 81,
        state_dim: int = 4,
    ) -> None:
        super().__init__()
        assert state_dim == 4, (
            f"DiT assumes agent states are (x, y, cos, sin); received {state_dim=}"
        )
        assert output_dim == trajectory_len * state_dim, (
            f"{output_dim=} must equal {trajectory_len=} * {state_dim=}"
        )

        self.heads = heads
        self.trajectory_len = trajectory_len
        self.state_dim = state_dim
        self.agent_embedding = nn.Embedding(4, hidden_dim)
        self.preproj = Mlp(
            in_features=trajectory_len * state_dim,
            hidden_features=512,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.t_embedder = Mlp(
            in_features=trajectory_len,
            hidden_features=512,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, heads, dropout, mlp_ratio) for i in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_dim, output_dim)

    def _make_agent_self_attention_mask(
        self,
        neighbor_current_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build the structural agent attention mask.

        The ego token is index 0. Ego may attend to neighbors, but neighbor queries
        cannot attend to the ego key. This encodes the assumption that ego behavior can
        depend on predicted neighbors, while neighbor predictions do not depend on ego.

        Invalid neighbors are masked as keys for valid queries. For invalid neighbor
        queries, only their own key is left unmasked to avoid all-masked attention rows,
        which would otherwise produce NaNs inside MultiheadAttention.
        """
        B, neighbor_num = neighbor_current_mask.shape
        agent_num = neighbor_num + 1
        device = neighbor_current_mask.device

        ego_invalid_mask = torch.zeros((B, 1), dtype=torch.bool, device=device)
        invalid_agent_mask = torch.cat([ego_invalid_mask, neighbor_current_mask], dim=1)

        mask = invalid_agent_mask[:, None, :].expand(B, agent_num, agent_num).clone()
        mask[:, 1:, 0] = True

        index = torch.arange(agent_num, device=device)
        self_only_mask = index.view(1, agent_num, 1) != index.view(1, 1, agent_num)
        invalid_query_mask = invalid_agent_mask[:, :, None]
        mask = (mask & ~invalid_query_mask) | (self_only_mask & invalid_query_mask)

        return mask.repeat_interleave(self.heads, dim=0)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cross_c: torch.Tensor,
        cross_c_mask: torch.Tensor,
        neighbor_current_mask: torch.Tensor,
        agent_class: torch.Tensor,
        current_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict denoised trajectories conditioned on encoder context.

        Args:
            x: Sampled current/future trajectories, shape (B, P, T, 4), where the state
                channels are (x, y, cos, sin).
            t: Diffusion time per trajectory step, shape (B, P, T, 1).
            cross_c: Encoder context tokens, shape (B, N, hidden_dim).
            cross_c_mask: Encoder context mask, shape (B, N). True means invalid.
            neighbor_current_mask: Neighbor-agent mask, shape (B, P - 1). True means invalid.
            agent_class: Agent class ids, shape (B, P). 0=ego, 1=vehicle,
                2=pedestrian, 3=bicycle.
            current_states: Current agent states, shape (B, P, 4). Neighbor predicted xy
                offsets are converted back to ego-frame positions using current xy.

        Returns:
            Predicted trajectories, shape (B, P, T, 4).
        """
        B, P, T, D = x.shape

        x = x.reshape(B, P, T * D)  # (B, P, T*D)
        t = t.reshape(B, P, T)  # (B, P, T)

        x = self.preproj(x)  # (B, P, hidden_dim)
        t = self.t_embedder(t)  # (B, P, hidden_dim)

        x_embedding = self.agent_embedding(agent_class)  # (B, P, hidden_dim)
        x = x + x_embedding

        self_attn_mask = self._make_agent_self_attention_mask(neighbor_current_mask)

        for block in self.blocks:
            x = block(
                x,
                t,
                self_attn_mask,
                cross_c,
                cross_c_mask,
            )

        x = self.final_layer(x, t)  # (B, P, output_dim)
        x = x.reshape(B, P, T, D)
        # if P > 1 and T > 1:
        #     neighbor_xy = x[:, 1:, 1:, :2] + current_states[:, 1:, None, :2]
        #     neighbor_future = torch.cat([neighbor_xy, x[:, 1:, 1:, 2:]], dim=-1)
        #     neighbors = torch.cat([x[:, 1:, :1], neighbor_future], dim=2)
        #     x = torch.cat([x[:, :1], neighbors], dim=1)
        return x
