from types import SimpleNamespace

import torch
from diffusion_planner.dimensions import INPUT_T, SEGMENT_POINT_DIM
from diffusion_planner.model.module.encoder import CLASS_TYPE_EGO, Encoder, VectorEncoder


def _encoder_config(velocity_dropout_ratio: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_dim=8,
        num_heads=2,
        encoder_drop_path_rate=0.0,
        encoder_mixer_depth=1,
        encoder_neighbor_attention_depth=1,
        encoder_fusion_depth=1,
        use_turn_indicators=True,
        velocity_dropout_ratio=velocity_dropout_ratio,
        time_len=2,
        agent_num=1,
        static_objects_num=1,
        static_objects_state_dim=10,
        lane_num=1,
        lane_len=2,
        route_num=1,
        route_len=2,
        polygon_num=1,
        polygon_len=2,
        line_string_num=1,
        line_string_len=2,
    )


def _encoder_inputs(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "neighbor_agents_past": torch.zeros(batch_size, 1, 2, 11),
        "static_objects": torch.zeros(batch_size, 1, 10),
        "lanes": torch.zeros(batch_size, 1, 2, SEGMENT_POINT_DIM),
        "lanes_speed_limit": torch.zeros(batch_size, 1, 1),
        "lanes_has_speed_limit": torch.zeros(batch_size, 1, 1, dtype=torch.bool),
        "route_lanes": torch.zeros(batch_size, 1, 2, SEGMENT_POINT_DIM),
        "route_lanes_speed_limit": torch.zeros(batch_size, 1, 1),
        "route_lanes_has_speed_limit": torch.zeros(batch_size, 1, 1, dtype=torch.bool),
        "polygons": torch.zeros(batch_size, 1, 2, 3),
        "line_strings": torch.zeros(batch_size, 1, 2, 4),
        "goal_pose": torch.zeros(batch_size, 4),
        "ego_shape": torch.ones(batch_size, 3),
        "ego_current_state": torch.tensor(
            [[0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * batch_size
        ),
        "turn_indicators": torch.zeros(batch_size, INPUT_T + 1),
    }


def test_encoder_does_not_require_ego_history():
    encoder = Encoder(_encoder_config())
    encoder.eval()

    outputs = encoder(_encoder_inputs())

    assert outputs.shape == (2, encoder.token_num, 8)


def test_velocity_vector_encoder_dropout_masks_velocity_token():
    encoder = VectorEncoder(
        num_float=1,
        hidden_dim=8,
        class_type=CLASS_TYPE_EGO,
        dropout_ratio=1.0,
    )
    encoder.train()

    encoding, mask, _ = encoder(torch.ones(2, 1))

    assert mask.tolist() == [[True], [True]]
    assert torch.count_nonzero(encoding) == 0
