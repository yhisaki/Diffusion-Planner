import torch.nn as nn

from drifting_planner.model.module.decoder import Decoder
from drifting_planner.model.module.encoder import Encoder


class DriftingPlanner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, inputs):
        encoder_outputs = self.encoder(inputs)
        decoder_outputs = self.decoder(encoder_outputs, inputs)

        return encoder_outputs, decoder_outputs
