import torch
import torch.nn as nn


class StopPredictor(nn.Module):
    """Predict stop logits at selected future timesteps."""

    def __init__(
        self,
        hidden_dim: int,
        num_steps: int,  # len(STOP_INDICES)
        num_heads: int = 4,
        depth: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_steps = num_steps

        self.query_tokens = nn.Parameter(torch.zeros(1, num_steps, hidden_dim))
        self.time_embedding = nn.Parameter(torch.zeros(1, num_steps, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=depth,
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.pred_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        nn.init.trunc_normal_(self.time_embedding, std=0.02)

    def forward(
        self,
        encoding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoding: (B, N, D)

        Returns:
            logits: (B, num_steps)
        """
        B = encoding.shape[0]

        key_padding_mask = encoding.abs().sum(dim=-1) == 0

        queries = self.query_tokens + self.time_embedding
        queries = queries.expand(B, -1, -1)  # (B, num_steps, D)

        x = torch.cat([queries, encoding], dim=1)  # (B, num_steps + N, D)

        # Query tokens are always valid.
        query_mask = torch.zeros(
            B,
            self.num_steps,
            dtype=torch.bool,
            device=encoding.device,
        )
        padding_mask = torch.cat([query_mask, key_padding_mask], dim=1)

        x = self.transformer(x, src_key_padding_mask=padding_mask)

        query_out = x[:, : self.num_steps]  # (B, num_steps, D)
        query_out = self.norm(query_out)

        logits = self.pred_head(query_out).squeeze(-1)  # (B, num_steps)
        return logits
