from typing import Any, Callable, TypeAlias

import torch
import torch.nn as nn

import diffusion_planner.model.diffusion_utils.dpm_solver_pytorch as dpm
from diffusion_planner.model.module.dit import DiT
from diffusion_planner.model.module.speed_predictor import SpeedPredictor
from diffusion_planner.model.module.turn_indicator import TurnIndicatorPredictor
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer

TensorDict: TypeAlias = dict[str, torch.Tensor]
DecoderOutput: TypeAlias = dict[str, torch.Tensor]
CurrentStateInfo: TypeAlias = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def replace_current_state(x: torch.Tensor, current_states: torch.Tensor) -> torch.Tensor:
    """
    Return trajectories whose first timestep is fixed to the current agent states.

    Args:
        x: Trajectories of shape (B, P, T, 4), with state channels (x, y, cos, sin).
        current_states: Current states of shape (B, P, 4).

    Returns:
        Trajectories of shape (B, P, T, 4).
    """
    return torch.cat([current_states[:, :, None, :], x[:, :, 1:, :]], dim=2)


class Decoder(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()

        dpr = config.decoder_drop_path_rate
        self._predicted_neighbor_num = config.predicted_neighbor_num
        self._future_len = config.future_len
        self._state_dim = 4

        self.dit = DiT(
            depth=config.decoder_depth,
            output_dim=(config.future_len + 1) * self._state_dim,
            hidden_dim=config.hidden_dim,
            heads=config.num_heads,
            dropout=dpr,
            trajectory_len=config.future_len + 1,
            state_dim=self._state_dim,
        )
        self.turn_indicator_predictor = TurnIndicatorPredictor(
            future_len=self._future_len,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
        )
        self.speed_predictor = SpeedPredictor(
            future_len=self._future_len,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            depth=getattr(config, "speed_predictor_depth", 2),
            dropout=dpr,
        )

        self._state_normalizer: StateNormalizer = config.state_normalizer
        self._observation_normalizer: ObservationNormalizer = config.observation_normalizer

        self._guidance_fn: Callable[..., torch.Tensor] | None = getattr(config, "guidance_fn", None)
        self._guidance_scale = getattr(config, "guidance_scale", 0.0)

        self._init_parameters()

    def _init_parameters(self) -> None:
        """Initialize decoder modules and zero the DiT output head."""
        self.apply(self._init_module)

        output_layer = self.dit.final_layer.proj[-1]
        nn.init.constant_(output_layer.weight, 0)
        nn.init.constant_(output_layer.bias, 0)

    @staticmethod
    def _init_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _prepare_current_states(self, inputs: TensorDict) -> CurrentStateInfo:
        """Extract current agent states, validity masks, and class ids.

        Args:
            inputs: Model inputs. Requires:
                - ego_current_state: (B, >=4)
                - neighbor_agents_past: (B, Pn, V, >=11), where the last step stores
                  current neighbor state and channels 8:11 store type scores.

        Returns:
            Tuple of:
                - current_states: (B, P, 4), concatenated ego and neighbor current states.
                - neighbor_current_mask: (B, Pn), True for invalid neighbors.
                - agent_class: (B, P), with 0=ego, 1=vehicle, 2=pedestrian, 3=bicycle.
        """
        B = inputs["ego_current_state"].shape[0]
        ego_current = inputs["ego_current_state"][:, None, :4]
        neighbors_current = inputs["neighbor_agents_past"][
            :, : self._predicted_neighbor_num, -1, :4
        ]
        neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0

        current_states = torch.cat([ego_current, neighbors_current], dim=1)  # [B, P, 4]

        neighbor_type = inputs["neighbor_agents_past"][:, : self._predicted_neighbor_num, -1, 8:11]
        neighbor_class = neighbor_type.argmax(dim=-1) + 1
        neighbor_class = torch.where(
            neighbor_current_mask,
            torch.ones_like(neighbor_class),
            neighbor_class,
        )
        ego_class = torch.zeros((B, 1), dtype=torch.long, device=neighbor_class.device)
        agent_class = torch.cat([ego_class, neighbor_class.long()], dim=1)

        return current_states, neighbor_current_mask, agent_class

    def _make_encoding_mask(self, encoding: torch.Tensor) -> torch.Tensor:
        """
        Infer invalid encoder tokens.

        Encoder tokens are zeroed when invalid, so an all-zero token is masked out for
        cross-attention and mean pooling.
        """
        return torch.sum(torch.ne(encoding, 0), dim=-1) == 0

    def _reshape_trajectory(self, trajectory: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Reshape a sampled or target trajectory to (B, P, 1 + future_len, 4).

        The decoder accepts either the structured shape (B, P, T, 4) or the flattened
        shape (B, P, T * 4).
        """
        agent_num = 1 + self._predicted_neighbor_num
        trajectory_len = 1 + self._future_len

        if trajectory.dim() == 4:
            return trajectory

        return trajectory.reshape(batch_size, agent_num, trajectory_len, self._state_dim)

    def _forward_training(
        self,
        encoding: torch.Tensor,
        inputs: TensorDict,
        current_states: torch.Tensor,
        agent_class: torch.Tensor,
        neighbor_current_mask: torch.Tensor,
        encoding_mask: torch.Tensor,
    ) -> DecoderOutput:
        """Forward pass for training mode.

        Args:
            encoding: Encoder tokens, shape (B, N, hidden_dim).
            inputs: Training inputs. Requires sampled_trajectories, gt_trajectories,
                and diffusion_time. The first timestep of sampled_trajectories is
                overwritten with current_states before DiT.
            current_states: Current ego and neighbor states, shape (B, P, 4).
            agent_class: DiT class ids, shape (B, P).
            neighbor_current_mask: Invalid-neighbor mask, shape (B, P - 1).
            encoding_mask: Invalid encoder token mask, shape (B, N).

        Returns:
            Dictionary with model_output of shape (B, P, 1 + future_len, 4) and
            turn_indicator_logit of shape (B, TURN_INDICATOR_OUTPUT_DIM).
        """
        B = encoding.shape[0]
        P = 1 + self._predicted_neighbor_num

        sampled_trajectories = self._reshape_trajectory(inputs["sampled_trajectories"], B)
        sampled_trajectories = replace_current_state(sampled_trajectories, current_states)
        diffusion_time = inputs["diffusion_time"]

        gt_trajectories = self._reshape_trajectory(inputs["gt_trajectories"], B)
        ego_trajectory = gt_trajectories[:, 0, 1::10, :2].reshape(B, 2 * (self._future_len // 10))
        turn_indicator_logit = self.turn_indicator_predictor(
            ego_trajectory,
            encoding,
            encoding_mask,
        )
        ego_velocity_future_prediction = self.speed_predictor(
            encoding,
            gt_trajectories[:, 0, 1:, :],
            encoding_mask,
        )

        return {
            "model_output": self.dit(
                x=sampled_trajectories,
                t=diffusion_time,
                cross_c=encoding,
                cross_c_mask=encoding_mask,
                neighbor_current_mask=neighbor_current_mask,
                agent_class=agent_class,
            ).reshape(B, P, -1, self._state_dim),
            "turn_indicator_logit": turn_indicator_logit,
            "ego_velocity_future_prediction": ego_velocity_future_prediction,
        }

    def _inference_x_start(
        self,
        encoding: torch.Tensor,
        inputs: TensorDict,
        current_states: torch.Tensor,
        agent_class: torch.Tensor,
        neighbor_current_mask: torch.Tensor,
        encoding_mask: torch.Tensor,
        sampled_trajectories: torch.Tensor,
    ) -> DecoderOutput:
        """Inference using X-Start (DPM Solver) approach.

        Args:
            encoding: Encoder tokens, shape (B, N, hidden_dim).
            inputs: Inference inputs passed through to optional guidance functions.
            current_states: Current ego and neighbor states, shape (B, P, 4).
            agent_class: DiT class ids, shape (B, P).
            neighbor_current_mask: Invalid-neighbor mask, shape (B, P - 1).
            encoding_mask: Invalid encoder token mask, shape (B, N).
            sampled_trajectories: Flattened sampled trajectories, shape
                (B, P, (1 + future_len) * 4).

        Returns:
            Dictionary with prediction of shape (B, P, future_len, 4) and
            turn_indicator_logit of shape (B, TURN_INDICATOR_OUTPUT_DIM).
        """
        B = encoding.shape[0]
        P = 1 + self._predicted_neighbor_num

        action_prefix = self._reshape_trajectory(sampled_trajectories, B)
        action_prefix = replace_current_state(action_prefix, current_states)
        xT = action_prefix.reshape(B, P, (1 + self._future_len) * self._state_dim)

        B, P, _, _ = action_prefix.shape

        def prefix_constraint(xt: torch.Tensor, t: torch.Tensor, step: int) -> torch.Tensor:
            xt = xt.reshape(B, P, 1 + self._future_len, self._state_dim)
            xt = replace_current_state(xt, current_states)
            return xt

        model_wrapper_params = {
            "classifier_fn": self._guidance_fn,
            "classifier_kwargs": {
                "model": self.dit,
                "model_condition": {
                    "cross_c": encoding,
                    "cross_c_mask": encoding_mask,
                    "neighbor_current_mask": neighbor_current_mask,
                    "agent_class": agent_class,
                },
                "inputs": inputs,
                "observation_normalizer": self._observation_normalizer,
                "state_normalizer": self._state_normalizer,
            },
            "guidance_scale": self._guidance_scale,
            "guidance_type": "classifier" if self._guidance_fn is not None else "uncond",
        }

        noise_schedule = dpm.NoiseScheduleVP()

        model_fn = dpm.model_wrapper(
            model=self.dit,
            noise_schedule=noise_schedule,
            model_type="x_start",
            model_kwargs={
                "cross_c": encoding,
                "cross_c_mask": encoding_mask,
                "neighbor_current_mask": neighbor_current_mask,
                "agent_class": agent_class,
            },
            **model_wrapper_params,
        )

        dpm_solver = dpm.DPM_Solver(
            model_fn=model_fn,
            noise_schedule=noise_schedule,
            correcting_xt_fn=prefix_constraint,
        )

        x0 = dpm_solver.sample(x=xT, steps=10, skip_type="logSNR")

        x0 = x0.reshape(B, P, (1 + self._future_len), self._state_dim)
        ego_trajectory = x0[:, 0, 1::10, :2].reshape(B, 2 * (self._future_len // 10))
        turn_indicator_logit = self.turn_indicator_predictor(
            ego_trajectory,
            encoding,
            encoding_mask,
        )
        ego_velocity_future_prediction = self.speed_predictor(
            encoding,
            x0[:, 0, 1:, :],
            encoding_mask,
        )
        x0 = self._state_normalizer.inverse(x0)[:, :, 1:]

        return {
            "prediction": x0,
            "turn_indicator_logit": turn_indicator_logit,
            "ego_velocity_future_prediction": ego_velocity_future_prediction,
        }

    def _forward_inference(
        self,
        encoding: torch.Tensor,
        inputs: TensorDict,
        current_states: torch.Tensor,
        agent_class: torch.Tensor,
        neighbor_current_mask: torch.Tensor,
        encoding_mask: torch.Tensor,
    ) -> DecoderOutput:
        """Forward pass for inference mode.

        Args:
            encoding: Encoder tokens, shape (B, N, hidden_dim).
            inputs: Inference inputs. Requires sampled_trajectories.
            current_states: Current ego and neighbor states, shape (B, P, 4).
            agent_class: DiT class ids, shape (B, P).
            neighbor_current_mask: Invalid-neighbor mask, shape (B, P - 1).
            encoding_mask: Invalid encoder token mask, shape (B, N).

        Returns:
            Dictionary with prediction and turn_indicator_logit.
        """
        B = encoding.shape[0]
        P = 1 + self._predicted_neighbor_num

        sampled_trajectories = self._reshape_trajectory(inputs["sampled_trajectories"], B).reshape(
            B, P, (1 + self._future_len) * self._state_dim
        )

        return self._inference_x_start(
            encoding=encoding,
            inputs=inputs,
            current_states=current_states,
            agent_class=agent_class,
            neighbor_current_mask=neighbor_current_mask,
            encoding_mask=encoding_mask,
            sampled_trajectories=sampled_trajectories,
        )

    def forward(self, encoding: torch.Tensor, inputs: TensorDict) -> DecoderOutput:
        """
        Run the diffusion decoder.

        Args:
            encoding: Encoder tokens, shape (B, N, hidden_dim). Invalid tokens are all zero.
            inputs: Model inputs. Required keys:
                - ego_current_state: (B, >=4)
                - neighbor_agents_past: (B, Pn, V, >=11)
                - sampled_trajectories: (B, P, 1 + future_len, 4) or flattened equivalent.
                Training additionally requires:
                - gt_trajectories: (B, P, 1 + future_len, 4)
                - diffusion_time: (B, P, 1 + future_len, 1)

        Returns:
            Training: model_output and turn_indicator_logit.
            Inference: prediction and turn_indicator_logit.
        """
        current_states, neighbor_current_mask, agent_class = self._prepare_current_states(inputs)

        B, P, _ = current_states.shape

        encoding_mask = self._make_encoding_mask(encoding)  # [B, N]

        if self.training:
            return self._forward_training(
                encoding=encoding,
                inputs=inputs,
                current_states=current_states,
                agent_class=agent_class,
                neighbor_current_mask=neighbor_current_mask,
                encoding_mask=encoding_mask,
            )
        else:
            return self._forward_inference(
                encoding=encoding,
                inputs=inputs,
                current_states=current_states,
                agent_class=agent_class,
                neighbor_current_mask=neighbor_current_mask,
                encoding_mask=encoding_mask,
            )
