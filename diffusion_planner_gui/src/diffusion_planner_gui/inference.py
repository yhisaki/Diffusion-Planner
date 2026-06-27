"""Model loading and deterministic inference for the result visualizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils.config import Config


class Predictor:
    """Thin wrapper around a loaded Diffusion Planner checkpoint."""

    def __init__(self, model_path: str | Path, device_name: str = "auto") -> None:
        self.device = _select_device(device_name)
        self.model_path = Path(model_path).expanduser().resolve()
        self.model_args = _load_config(self.model_path)
        self.model = Diffusion_Planner(self.model_args)
        _load_checkpoint(self.model_path, self.model, self.device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        npz_path: str | Path | None = None,
        data: dict[str, np.ndarray] | None = None,
        noise_scale: float = 1.0,
        noise_seed: int = 0,
    ) -> np.ndarray:
        prediction, _ = self.predict_with_velocity(
            npz_path=npz_path,
            data=data,
            noise_scale=noise_scale,
            noise_seed=noise_seed,
        )
        return prediction

    @torch.no_grad()
    def predict_with_velocity(
        self,
        npz_path: str | Path | None = None,
        data: dict[str, np.ndarray] | None = None,
        noise_scale: float = 1.0,
        noise_seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        inputs = _prepare_input_data(npz_path=npz_path, data=data, device=self.device)
        inputs = self.model_args.observation_normalizer(inputs)

        batch_size = inputs["ego_current_state"].shape[0]
        agent_count = 1 + int(self.model_args.predicted_neighbor_num)
        future_len = int(self.model_args.future_len)
        sampled_trajectories = torch.zeros(
            batch_size,
            agent_count,
            future_len + 1,
            4,
            dtype=torch.float32,
            device=self.device,
        )
        if noise_scale > 0.0:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(noise_seed))
            sampled_trajectories[:, :, 1:, :] = float(noise_scale) * torch.randn(
                batch_size,
                agent_count,
                future_len,
                4,
                dtype=torch.float32,
                device=self.device,
                generator=generator,
            )
        inputs["sampled_trajectories"] = sampled_trajectories

        _, outputs = self.model(inputs)
        prediction = outputs["prediction"][0].detach().cpu().numpy()
        velocity = outputs.get("ego_velocity_future_prediction")
        if velocity is not None:
            velocity = velocity[0].detach().cpu().numpy()
        return prediction, velocity


def _select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _load_config(model_path: Path) -> Config:
    args_path = model_path.parent / "args.json"
    if not args_path.is_file():
        raise FileNotFoundError(f"args.json not found in model directory: {args_path}")
    return Config(str(args_path))


def _load_checkpoint(model_path: Path, model: Diffusion_Planner, device: torch.device) -> None:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "ema_state_dict" in checkpoint:
        state_dict = checkpoint["ema_state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format: {model_path}")

    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)


def _prepare_input_data(
    npz_path: str | Path | None,
    data: dict[str, np.ndarray] | None,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if data is None:
        if npz_path is None:
            raise ValueError("Either npz_path or data must be provided.")
        with np.load(str(npz_path)) as loaded:
            source = {key: loaded[key] for key in loaded.files}
    else:
        source = data

    tensor_data: dict[str, torch.Tensor] = {}
    for key, value in source.items():
        if key in {"map_name", "token", "delay"}:
            continue
        if not isinstance(value, np.ndarray) or value.dtype.kind in {"O", "U", "S"}:
            continue
        tensor_data[key] = torch.as_tensor(np.expand_dims(value, axis=0), device=device)

    if "goal_pose" in tensor_data:
        tensor_data["goal_pose"] = heading_to_cos_sin(tensor_data["goal_pose"])
    if "ego_agent_past" in tensor_data:
        tensor_data["ego_agent_past"] = heading_to_cos_sin(tensor_data["ego_agent_past"])
    tensor_data["delay"] = torch.zeros(1, dtype=torch.long, device=device)

    return tensor_data
