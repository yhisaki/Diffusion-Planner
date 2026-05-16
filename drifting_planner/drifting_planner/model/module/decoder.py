import torch
import torch.nn as nn

from drifting_planner.dimensions import TURN_INDICATOR_OUTPUT_DIM
from drifting_planner.model.module.dit import DiT
from drifting_planner.utils.normalizer import StateNormalizer, TrajectoryNormalizer


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        dpr = config.decoder_drop_path_rate
        self._predicted_neighbor_num = config.predicted_neighbor_num
        self._future_len = config.future_len
        self._state_normalizer: StateNormalizer = config.state_normalizer
        self._trajectory_normalizer: TrajectoryNormalizer = config.trajectory_normalizer
        self._observation_normalizer = config.observation_normalizer

        self.dit = DiT(
            depth=config.decoder_depth,
            output_dim=(config.future_len + 1) * 4,
            hidden_dim=config.hidden_dim,
            heads=config.num_heads,
            dropout=dpr,
        )
        self.turn_indicator_predictor = nn.Linear(
            2 * (self._future_len // 10) + config.hidden_dim, TURN_INDICATOR_OUTPUT_DIM
        )

        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        self.apply(_basic_init)

        nn.init.constant_(self.dit.final_layer.proj[-1].weight, 0)
        nn.init.constant_(self.dit.final_layer.proj[-1].bias, 0)

    def _prepare_current_states(self, inputs):
        denorm_inputs = self._observation_normalizer.inverse(
            {
                "ego_current_state": inputs["ego_current_state"],
                "neighbor_agents_past": inputs["neighbor_agents_past"],
            }
        )
        ego_current_raw = denorm_inputs["ego_current_state"][:, None, :4]
        neighbors_current_raw = denorm_inputs["neighbor_agents_past"][
            :, : self._predicted_neighbor_num, -1, :4
        ]
        neighbor_current_mask = torch.sum(torch.ne(neighbors_current_raw[..., :4], 0), dim=-1) == 0
        inputs["neighbor_current_mask"] = neighbor_current_mask

        current_states_raw = torch.cat([ego_current_raw, neighbors_current_raw], dim=1)
        current_states = self._state_normalizer(current_states_raw[:, :, None, :])[:, :, 0, :]

        return current_states, neighbor_current_mask, ego_current_raw, neighbors_current_raw

    def _compute_turn_indicator(self, ego_trajectory, encoding_pooled):
        turn_indicator_input = torch.cat([ego_trajectory, encoding_pooled], dim=-1)
        return self.turn_indicator_predictor(turn_indicator_input)

    def _forward(self, encoding, inputs, neighbor_current_mask, encoding_pooled):
        B = encoding.shape[0]
        P = 1 + self._predicted_neighbor_num
        T = self._future_len

        sampled_trajectories = inputs["sampled_trajectories"].reshape(B, P, (1 + T), 4)

        gt_trajectories = inputs.get("gt_trajectories")
        if gt_trajectories is not None:
            gt_trajectories = gt_trajectories.reshape(B, P, (1 + T), 4)
            ego_trajectory = gt_trajectories[:, 0, 1::10, :2].reshape(B, 2 * (T // 10))
        else:
            ego_trajectory = sampled_trajectories[:, 0, 1::10, :2].reshape(B, 2 * (T // 10))
        turn_indicator_logit = self._compute_turn_indicator(ego_trajectory, encoding_pooled)

        return {
            "model_output": self.dit(
                sampled_trajectories,
                encoding_pooled.unsqueeze(1).expand(-1, P, -1),
                encoding,
                neighbor_current_mask,
            ).reshape(B, P, -1, 4),
            "turn_indicator_logit": turn_indicator_logit,
        }

    def forward(self, encoding, inputs):
        current_states, neighbor_current_mask, ego_current, neighbors_current = (
            self._prepare_current_states(inputs)
        )

        B, P, _ = current_states.shape
        assert P == (1 + self._predicted_neighbor_num)

        encoding_pooled = torch.mean(encoding, dim=1)

        if self.training:
            return self._forward(encoding, inputs, neighbor_current_mask, encoding_pooled)
        else:
            T = self._future_len
            noise = torch.randn(B, P, T, 4, device=current_states.device)
            sampled_trajectories = torch.cat([current_states[:, :, None, :], noise], dim=2)

            merged_inputs = {
                **inputs,
                "sampled_trajectories": sampled_trajectories.reshape(B, P, (1 + T) * 4),
            }

            result = self._forward(encoding, merged_inputs, neighbor_current_mask, encoding_pooled)
            prediction_delta_norm = result["model_output"][:, :, 1:, :].clone()
            current_states_raw = torch.cat([ego_current, neighbors_current], dim=1)
            prediction = self._trajectory_normalizer.inverse_future(
                prediction_delta_norm, current_states_raw
            )
            prediction[:, 1:] = prediction[:, 1:].masked_fill(
                neighbor_current_mask[..., None, None], 0.0
            )
            result["prediction"] = prediction
            return result
