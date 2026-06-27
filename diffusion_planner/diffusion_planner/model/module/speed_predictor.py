from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp


class SpeedPredictor(nn.Module):
    """Predict ego future velocity from encoder tokens and ego future path shape."""

    def __init__(
        self,
        future_len: int,
        hidden_dim: int,
        num_heads: int,
        depth: int = 2,
        dropout: float = 0.0,
        velocity_dim: int = 1,
        path_dim: int = 4,
    ) -> None:
        super().__init__()
        self.future_len = future_len
        self.velocity_dim = velocity_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.ego_future_proj = Mlp(
            in_features=path_dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=dropout,
        )
        self.ego_future_pos_emb = nn.Parameter(torch.zeros(1, future_len, hidden_dim))
        self.token_type_emb = nn.Embedding(3, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.head = Mlp(
            in_features=hidden_dim,
            hidden_features=hidden_dim,
            out_features=future_len * velocity_dim,
            act_layer=nn.GELU,
            drop=dropout,
        )

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.ego_future_pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.token_type_emb.weight, mean=0.0, std=0.02)

        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

        self.apply(_basic_init)

    @staticmethod
    def _make_encoding_mask(encoding: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.ne(encoding, 0), dim=-1) == 0

    def forward(
        self,
        encoding: torch.Tensor,
        ego_agent_future: torch.Tensor,
        encoding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            encoding: Encoder tokens, shape (B, N, hidden_dim). Invalid tokens may be zeroed.
            ego_agent_future: Ego future path, shape (B, T, path_dim).
            encoding_mask: Optional encoder mask, shape (B, N). True means invalid.

        Returns:
            Predicted ego future velocity, shape (B, T, velocity_dim).
        """
        if ego_agent_future.shape[1] != self.future_len:
            raise ValueError(
                f"ego_agent_future length must be {self.future_len}, "
                f"got {ego_agent_future.shape[1]}"
            )
        B = encoding.shape[0]
        if encoding_mask is None:
            encoding_mask = self._make_encoding_mask(encoding)

        cls_token = self.cls_token.expand(B, -1, -1)
        path_tokens = self.ego_future_proj(ego_agent_future)
        path_tokens = path_tokens + self.ego_future_pos_emb

        cls_type = self.token_type_emb.weight[0].view(1, 1, -1)
        path_type = self.token_type_emb.weight[1].view(1, 1, -1)
        context_type = self.token_type_emb.weight[2].view(1, 1, -1)

        tokens = torch.cat(
            [
                cls_token + cls_type,
                path_tokens + path_type,
                encoding + context_type,
            ],
            dim=1,
        )
        cls_path_mask = torch.zeros(
            B,
            1 + self.future_len,
            dtype=torch.bool,
            device=encoding.device,
        )
        token_mask = torch.cat([cls_path_mask, encoding_mask], dim=1)

        x = self.transformer(tokens, src_key_padding_mask=token_mask)
        velocity = F.softplus(self.head(self.out_norm(x[:, 0])))
        return velocity.view(B, self.future_len, self.velocity_dim)
