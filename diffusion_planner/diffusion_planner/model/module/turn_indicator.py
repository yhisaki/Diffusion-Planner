import torch
import torch.nn as nn
from timm.layers import Mlp

from diffusion_planner.dimensions import TURN_INDICATOR_OUTPUT_DIM


class TurnIndicatorPredictor(nn.Module):
    """Predict turn-indicator logits from ego future xy and detached encoder tokens."""

    def __init__(self, future_len: int, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.future_len = future_len
        ego_feature_dim = 2 * (future_len // 10)
        self.ego_proj = Mlp(
            in_features=ego_feature_dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.encoder_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(2 * hidden_dim)
        self.head = Mlp(
            in_features=2 * hidden_dim,
            hidden_features=2 * hidden_dim,
            out_features=TURN_INDICATOR_OUTPUT_DIM,
            act_layer=nn.GELU,
            drop=0.0,
        )

    def forward(
        self,
        ego_trajectory: torch.Tensor,
        encoding: torch.Tensor,
        encoding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            ego_trajectory: Flattened ego xy samples, shape (B, 2 * (future_len // 10)).
            encoding: Encoder tokens, shape (B, N, hidden_dim). Detached before use so
                turn-indicator loss does not update the encoder.
            encoding_mask: Encoder context mask, shape (B, N). True means invalid.

        Returns:
            Turn-indicator logits, shape (B, TURN_INDICATOR_OUTPUT_DIM).
        """
        encoding = encoding.detach()
        ego_token = self.ego_proj(ego_trajectory).unsqueeze(1)
        context_token = self.encoder_attn(
            ego_token,
            encoding,
            encoding,
            key_padding_mask=encoding_mask,
        )[0]
        x = torch.cat([ego_token.squeeze(1), context_token.squeeze(1)], dim=-1)
        return self.head(self.out_norm(x))
