from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.drop = nn.Dropout(dropout)

        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = Mlp(
            in_features=hidden_dim,
            hidden_features=hidden_dim * 4,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.norm_q(x)
        kv = self.norm_kv(memory)

        attn_out, _ = self.attn(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=memory_mask,
            need_weights=False,
        )

        x = x + self.drop(attn_out)
        x = x + self.ffn(self.norm_ffn(x))
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(hidden_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.drop = nn.Dropout(dropout)

        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = Mlp(
            in_features=hidden_dim,
            hidden_features=hidden_dim * 4,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.norm(x)

        attn_out, _ = self.attn(
            query=q,
            key=q,
            value=q,
            key_padding_mask=mask,
            need_weights=False,
        )

        x = x + self.drop(attn_out)
        x = x + self.ffn(self.norm_ffn(x))

        if mask is not None:
            x = torch.where(
                (~mask).unsqueeze(-1),
                x,
                torch.zeros_like(x),
            )

        return x


class SpeedPredictor(nn.Module):
    """Predict ego future velocity from scene tokens and distance-domain ego path.

    encoding:
        Scene / agent / map tokens, shape (B, N, hidden_dim).
        Invalid tokens are zero vectors.

    ego_agent_future:
        Distance-domain future path, shape (B, S, path_dim).
        Usually S = 80 for 80m at 1m interval.

    output:
        Time-domain velocity, shape (B, T, velocity_dim).
        Usually T = 80 for 8s at 0.1s interval.
    """

    def __init__(
        self,
        future_len: int,
        hidden_dim: int,
        num_heads: int,
        depth: int = 2,
        dropout: float = 0.0,
        velocity_dim: int = 1,
        path_dim: int = 4,
        path_len: int = 80,
    ) -> None:
        super().__init__()

        self.future_len = future_len
        self.path_len = path_len
        self.velocity_dim = velocity_dim
        self.hidden_dim = hidden_dim

        # ------------------------------------------------------------
        # Path encoder: flatten the full distance-domain path into one token,
        # matching DiT's sampled-trajectory encoding.
        # ------------------------------------------------------------
        self.path_preproj = Mlp(
            in_features=path_len * path_dim,
            hidden_features=512,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.path_output_norm = nn.LayerNorm(hidden_dim)

        # ------------------------------------------------------------
        # Velocity decoder: time domain, 8s / 0.1s
        # ------------------------------------------------------------
        self.time_query = nn.Parameter(torch.zeros(1, future_len, hidden_dim))

        self.time_pos_embed = nn.Parameter(torch.zeros(1, future_len, hidden_dim))

        self.decoder_blocks = nn.ModuleList([
            nn.ModuleDict({
                "cross": CrossAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                ),
                "self": SelfAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                ),
            })
            for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)

        # Use softplus so the predicted velocity remains non-negative.
        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, velocity_dim),
        )

        # Logit for explicitly predicting a complete stop.
        self.stop_logit_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, velocity_dim),
        )

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.trunc_normal_(self.time_query, std=0.02)
        nn.init.trunc_normal_(self.time_pos_embed, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Avoid biasing the initial state too strongly toward stopping at all points.
        last_stop = self.stop_logit_head[-1]
        if isinstance(last_stop, nn.Linear) and last_stop.bias is not None:
            nn.init.constant_(last_stop.bias, -4.0)

    @staticmethod
    def _make_encoding_mask(encoding: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.ne(encoding, 0), dim=-1) == 0

    def forward(
        self,
        encoding: torch.Tensor,
        ego_agent_future: torch.Tensor,
        encoding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoding:
                Encoder tokens, shape (B, N, hidden_dim).
                Invalid tokens may be zeroed.

            ego_agent_future:
                Ego future path, shape (B, path_len, path_dim).

            encoding_mask:
                Optional encoder mask, shape (B, N).
                True means invalid.
                If None, the mask is recovered from zero tokens.

        Returns:
            If the module is in training mode:
                Tuple of predicted ego future velocity and stop logits, each with shape
                (B, T, velocity_dim).
            Otherwise:
                Predicted ego future velocity, shape (B, T, velocity_dim), with stopped
                steps overwritten to zero.
        """
        B, S, _ = ego_agent_future.shape
        if S != self.path_len:
            raise ValueError(f"ego_agent_future length must be {self.path_len}, got {S}")

        if encoding_mask is None:
            encoding_mask = self._make_encoding_mask(encoding)

        # ------------------------------------------------------------
        # 1. Encode the distance-domain path as one token
        # ------------------------------------------------------------
        path_tokens = self.path_preproj(ego_agent_future.reshape(B, -1)).unsqueeze(1)
        path_tokens = self.path_output_norm(path_tokens)

        # ------------------------------------------------------------
        # 2. Concatenate scene tokens and the compressed path token
        # ------------------------------------------------------------
        memory = torch.cat(
            [
                encoding,
                path_tokens,
            ],
            dim=1,
        )

        memory_mask = torch.cat(
            [
                encoding_mask,
                torch.zeros((B, 1), dtype=torch.bool, device=ego_agent_future.device),
            ],
            dim=1,
        )

        # ------------------------------------------------------------
        # 3. Create time-domain velocity query tokens
        # ------------------------------------------------------------
        T = self.future_len

        x = self.time_query.expand(B, -1, -1)
        x = x + self.time_pos_embed[:, :T]

        # ------------------------------------------------------------
        # 4. Decode time-domain velocity from fused memory
        # ------------------------------------------------------------
        for block in self.decoder_blocks:
            x = block["cross"](
                x=x,
                memory=memory,
                memory_mask=memory_mask,
            )
            x = block["self"](x)

        x = self.final_norm(x)

        # ------------------------------------------------------------
        # 5. Predict velocity and stop logit
        # ------------------------------------------------------------
        raw_velocity = self.velocity_head(x)
        velocity = F.softplus(raw_velocity)

        stop_logit = self.stop_logit_head(x)

        if self.training:
            return velocity, stop_logit

        stop_mask = torch.sigmoid(stop_logit) >= 0.5
        velocity = torch.where(stop_mask, torch.zeros_like(velocity), velocity)

        return velocity
