from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from drifting_planner.model.module.decoder import Decoder
from drifting_planner.model.module.encoder import Encoder


class DriftingPlanner(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(
        self, inputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        encoder_outputs = self.encoder(inputs)
        decoder_outputs = self.decoder(encoder_outputs, inputs)

        return encoder_outputs, decoder_outputs
