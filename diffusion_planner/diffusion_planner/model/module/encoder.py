from typing import Any, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp

from diffusion_planner.dimensions import *
from diffusion_planner.model.module.mixer import MixerBlock

CLASS_TYPE_EGO = 0
CLASS_TYPE_NEIGHBOR_VEHICLE = 1
CLASS_TYPE_NEIGHBOR_PEDESTRIAN = 2
CLASS_TYPE_NEIGHBOR_BICYCLE = 3
CLASS_TYPE_STATIC = 4
CLASS_TYPE_LANE = 5
CLASS_TYPE_ROUTE = 6
CLASS_TYPE_POLYGON = 7
CLASS_TYPE_LINE_STRING = 8
CLASS_TYPE_GOAL_POSE = 9
CLASS_TYPE_EGO_SHAPE = 10
CLASS_TYPE_TURN_INDICATOR = 11
CLASS_TYPE_EGO_SPEED = 12
CLASS_TYPE_NUM = 13

EncoderOutput: TypeAlias = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def add_class_type(x: torch.Tensor, class_type: int) -> torch.Tensor:
    """
    Add class type to the input tensor.
    Args:
        x: Tensor of shape (B, T, D=4) where D=4 represents (x, y, cos, sin)
        class_type: Class type to add (int)
    Returns:
        x: Tensor with class type added at the end
    """
    B, T, D = x.shape
    assert D == 4, "Input tensor must have 4 features (x, y, cos, sin)"
    class_type_tensor = F.one_hot(
        torch.full((B, T), class_type, device=x.device, dtype=torch.long),
        num_classes=CLASS_TYPE_NUM,
    ).to(dtype=x.dtype)
    return torch.cat([x, class_type_tensor], dim=-1)


def add_neighbor_class_type(x: torch.Tensor, neighbor_type: torch.Tensor) -> torch.Tensor:
    """
    Add neighbor-specific class type to the input tensor.
    Args:
        x: Tensor of shape (B, P, D=4) where D=4 represents (x, y, cos, sin)
        neighbor_type: Tensor of shape (B, P, 3) one-hot type (vehicle, pedestrian, bicycle)
    Returns:
        x: Tensor with neighbor class type added at the end
    """
    B, P, D = x.shape
    assert D == 4, "Input tensor must have 4 features (x, y, cos, sin)"

    type_idx = neighbor_type.argmax(dim=-1) + CLASS_TYPE_NEIGHBOR_VEHICLE
    class_type_tensor = F.one_hot(type_idx, num_classes=CLASS_TYPE_NUM).to(dtype=x.dtype)
    return torch.cat([x, class_type_tensor], dim=-1)


class Encoder(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()

        self.hidden_dim = config.hidden_dim

        self.use_ego_history = config.use_ego_history
        self.ego_history_dropout_rate = config.ego_history_dropout_rate
        self.use_turn_indicators = config.use_turn_indicators
        self.ego_history_len = 6

        self.token_num = (
            1  # Ego agent past token
            + config.agent_num
            + config.static_objects_num
            + config.lane_num
            + config.route_num
            + config.polygon_num
            + config.line_string_num
            + 1  # Goal pose token
            + 1  # Ego shape token
            + 1  # Turn indicator token
            + 1  # Ego speed token
        )

        self.ego_encoder = EgoHistoryEncoder(
            self.ego_history_len,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.encoder_mixer_depth,
        )
        self.neighbor_encoder = NeighborEncoder(
            config.time_len,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.encoder_mixer_depth,
        )
        self.static_encoder = StaticEncoder(
            config.static_objects_state_dim,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
        )
        self.lane_encoder = LaneEncoder(
            config.lane_len,
            class_type=CLASS_TYPE_LANE,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.encoder_mixer_depth,
        )
        self.route_encoder = RouteEncoder(
            config.route_len,
            route_num=config.route_num,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.encoder_mixer_depth,
        )
        self.polygon_encoder = LineEncoder(
            config.polygon_len,
            class_type=CLASS_TYPE_POLYGON,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.encoder_mixer_depth,
            point_dim=2 + POLYGON_TYPE_NUM,
        )
        self.line_string_encoder = LineEncoder(
            config.line_string_len,
            class_type=CLASS_TYPE_LINE_STRING,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.encoder_mixer_depth,
            point_dim=2 + LINE_STRING_TYPE_NUM,
        )
        self.goal_pose_encoder = VectorEncoder(
            num_float=4,
            hidden_dim=config.hidden_dim,
            class_type=CLASS_TYPE_GOAL_POSE,
            use_input_as_pos=True,
        )
        self.ego_shape_encoder = VectorEncoder(
            num_float=3,
            hidden_dim=config.hidden_dim,
            class_type=CLASS_TYPE_EGO_SHAPE,
        )
        self.turn_indicator_encoder = VectorEncoder(
            num_float=INPUT_T,
            hidden_dim=config.hidden_dim,
            class_type=CLASS_TYPE_TURN_INDICATOR,
        )
        self.ego_speed_encoder = VectorEncoder(
            num_float=1,
            hidden_dim=config.hidden_dim,
            class_type=CLASS_TYPE_EGO_SPEED,
        )

        # Pose/type embedding encodes x, y, cos, sin, and class type.
        self.pose_type_emb = nn.Linear(4 + CLASS_TYPE_NUM, config.hidden_dim)

        self._init_parameters()

    def _init_parameters(self) -> None:
        def _basic_init(m: nn.Module) -> None:
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

        nn.init.normal_(self.pose_type_emb.weight, std=0.02)
        nn.init.normal_(self.lane_encoder.speed_limit_emb.weight, std=0.02)
        nn.init.normal_(self.lane_encoder.attribute_emb.weight, std=0.02)
        nn.init.normal_(self.route_encoder.lane_encoder.speed_limit_emb.weight, std=0.02)
        nn.init.normal_(self.route_encoder.lane_encoder.attribute_emb.weight, std=0.02)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode scene observations into context tokens for the decoder.

        Each sub-encoder returns three tensors:
            encoding: (B, N, hidden_dim)
            mask:     (B, N), bool, True means the token is invalid or padded
            pos:      (B, N, 4 + CLASS_TYPE_NUM), used only for positional/type embedding

        The returned tensor has shape (B, self.token_num, hidden_dim). Invalid tokens are
        kept as all-zero vectors so the decoder can recover the same mask from the output.
        """
        # ego agent
        ego_agent_past = inputs["ego_agent_past"][:, -self.ego_history_len :].clone()
        if not self.use_ego_history:
            ego_agent_past = torch.zeros_like(ego_agent_past)

        # agents
        neighbors = inputs["neighbor_agents_past"].clone()  # (B, P, T, 11)

        # static objects
        static = inputs["static_objects"]  # (B, P, 10)

        # vector maps
        lanes = inputs["lanes"]  # (B, P, V, D)
        lanes_speed_limit = inputs["lanes_speed_limit"]  # (B, P, 1)
        lanes_has_speed_limit = inputs["lanes_has_speed_limit"]  # (B, P, 1)

        # route
        route = inputs["route_lanes"]  # (B, P, V, D)
        route_speed_limit = inputs["route_lanes_speed_limit"]  # (B, P, 1)
        route_has_speed_limit = inputs["route_lanes_has_speed_limit"]  # (B, P, 1)

        # polygons
        polygons = inputs["polygons"]  # (B, P, V, D)

        # line strings
        line_strings = inputs["line_strings"]  # (B, P, V, D)

        # goal pose
        goal_pose = inputs["goal_pose"]  # (B, D=4)

        # ego shape
        ego_shape = inputs["ego_shape"]  # (B, D=3)

        # ego speed
        ego_speed = inputs["ego_current_state"][:, 4:5]  # (B, D=1)

        # turn indicator
        turn_indicator = inputs["turn_indicators"][:, :-1]  # (B, T)
        turn_indicator = turn_indicator.float()
        if not self.use_turn_indicators:
            turn_indicator = torch.zeros_like(turn_indicator)

        B = neighbors.shape[0]

        encoding_ego, ego_mask, ego_pos = self.ego_encoder(ego_agent_past)

        if self.ego_history_dropout_rate > 0:
            encoding_ego = F.dropout(
                encoding_ego, p=self.ego_history_dropout_rate, training=self.training
            )

        encoding_neighbors, neighbors_mask, neighbor_pos = self.neighbor_encoder(neighbors)
        encoding_static, static_mask, static_pos = self.static_encoder(static)
        encoding_lanes, lanes_mask, lane_pos = self.lane_encoder(
            lanes, lanes_speed_limit, lanes_has_speed_limit
        )
        encoding_route, route_mask, route_pos = self.route_encoder(
            route, route_speed_limit, route_has_speed_limit
        )
        encoding_polygon, polygon_mask, polygon_pos = self.polygon_encoder(polygons)
        encoding_line_string, line_string_mask, line_string_pos = self.line_string_encoder(
            line_strings
        )

        encoding_goal_pose, goal_pose_mask, goal_pose_pos = self.goal_pose_encoder(goal_pose)
        encoding_ego_shape, ego_shape_mask, ego_shape_pos = self.ego_shape_encoder(ego_shape)
        encoding_turn_indicator, turn_indicator_mask, turn_indicator_pos = (
            self.turn_indicator_encoder(turn_indicator)
        )
        encoding_ego_speed, ego_speed_mask, ego_speed_pos = self.ego_speed_encoder(ego_speed)

        encoding_input = torch.cat(
            [
                encoding_ego,
                encoding_neighbors,
                encoding_static,
                encoding_lanes,
                encoding_route,
                encoding_polygon,
                encoding_line_string,
                encoding_goal_pose,
                encoding_ego_shape,
                encoding_turn_indicator,
                encoding_ego_speed,
            ],
            dim=1,
        )

        # All masks use PyTorch attention-mask polarity: True means invalid/padded.
        encoding_mask = torch.cat(
            [
                ego_mask,
                neighbors_mask,
                static_mask,
                lanes_mask,
                route_mask,
                polygon_mask,
                line_string_mask,
                goal_pose_mask,
                ego_shape_mask,
                turn_indicator_mask,
                ego_speed_mask,
            ],
            dim=1,
        ).view(-1)

        # Add geometry/type positional embedding only to valid tokens.
        pose_type_input = torch.cat(
            [
                ego_pos,
                neighbor_pos,
                static_pos,
                lane_pos,
                route_pos,
                polygon_pos,
                line_string_pos,
                goal_pose_pos,
                ego_shape_pos,
                turn_indicator_pos,
                ego_speed_pos,
            ],
            dim=1,
        ).view(B * self.token_num, -1)
        pose_type_embedding = self.pose_type_emb(pose_type_input)
        pose_type_embedding = torch.where(
            (~encoding_mask).unsqueeze(-1),
            pose_type_embedding,
            torch.zeros_like(pose_type_embedding),
        )

        encoder_outputs = encoding_input + pose_type_embedding.view(B, self.token_num, -1)

        # Keep invalid tokens exactly zero after normalization. Decoder uses this invariant
        # to build its cross-attention mask without a separate encoder mask output.
        encoder_outputs = torch.where(
            (~encoding_mask).view(B, self.token_num, 1),
            encoder_outputs,
            torch.zeros_like(encoder_outputs),
        )

        return encoder_outputs


class EgoHistoryEncoder(nn.Module):
    def __init__(
        self,
        time_len: int,
        drop_path_rate: float,
        hidden_dim: int,
        depth: int,
    ) -> None:
        super().__init__()
        tokens_mlp_dim = 64
        channels_mlp_dim = 128

        self._hidden_dim = hidden_dim

        self.channel_pre_project = Mlp(
            in_features=4,
            hidden_features=channels_mlp_dim,
            out_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.token_pre_project = Mlp(
            in_features=time_len,
            hidden_features=tokens_mlp_dim,
            out_features=tokens_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        self.blocks = nn.ModuleList([
            MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for i in range(depth)
        ])

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(
            in_features=channels_mlp_dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        """
        Args:
            x: Ego history, shape (B, T, 4), fields are x, y, cos, sin.

        Returns:
            encoding: (B, 1, hidden_dim)
            mask: (B, 1), always False because ego history is a required token
            pos: (B, 1, 4 + CLASS_TYPE_NUM), current ego pose plus class type
        """
        B, T, D = x.shape
        pos = x[:, -1].clone()  # (B, D=4[x, y, cos, sin])
        pos = pos.unsqueeze(1)  # (B, 1, D=4)
        pos = add_class_type(pos, CLASS_TYPE_EGO)

        mask = torch.zeros((B, 1), dtype=torch.bool, device=x.device)

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)

        for block in self.blocks:
            x = block(x)

        # pooling
        x = torch.mean(x, dim=1, keepdim=True)  # (B, 1, C=channels_mlp_dim)

        x = self.emb_project(self.norm(x))  # (B, hidden_dim)

        return x, mask, pos


class NeighborEncoder(nn.Module):
    def __init__(
        self,
        time_len: int,
        drop_path_rate: float,
        hidden_dim: int,
        depth: int,
    ) -> None:
        super().__init__()
        tokens_mlp_dim = 64
        channels_mlp_dim = 128

        self._hidden_dim = hidden_dim

        self.channel_pre_project = Mlp(
            in_features=8 + 1,
            hidden_features=channels_mlp_dim,
            out_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.token_pre_project = Mlp(
            in_features=time_len,
            hidden_features=tokens_mlp_dim,
            out_features=tokens_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        self.blocks = nn.ModuleList([
            MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for i in range(depth)
        ])

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(
            in_features=channels_mlp_dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        """
        Args:
            x: Neighbor histories, shape (B, P, V, 11), fields are
                x, y, cos, sin, vx, vy, width, length, type_one_hot(3).
                P is the number of neighbor slots; V is the history length.

        Returns:
            encoding: (B, P, hidden_dim)
            mask: (B, P), True when every timestep for that neighbor is empty
            pos: (B, P, 4 + CLASS_TYPE_NUM), latest neighbor pose plus class type
        """
        neighbor_type = x[:, :, -1, 8:]
        x = x[..., :8]

        pos = x[:, :, -1, :4].clone()  # x, y, cos, sin
        pos = add_neighbor_class_type(pos, neighbor_type)

        B, P, V, _ = x.shape
        history_step_invalid_mask = torch.sum(torch.ne(x[..., :8], 0), dim=-1).to(x.device) == 0
        neighbor_invalid_mask = torch.sum(~history_step_invalid_mask, dim=-1) == 0
        x = torch.cat([x, (~history_step_invalid_mask).float().unsqueeze(-1)], dim=-1)
        x = x.view(B * P, V, -1)
        x = torch.cat([x[..., :4], torch.zeros_like(x[..., 4:6]), x[..., 6:]], dim=-1)

        valid_slot_mask = ~neighbor_invalid_mask.view(-1)
        x = torch.where(valid_slot_mask.view(-1, 1, 1), x, torch.zeros_like(x))

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)

        # pooling
        x = torch.mean(x, dim=1)

        x = self.emb_project(self.norm(x))
        x_result = x * valid_slot_mask.float().unsqueeze(-1)

        return x_result.view(B, P, -1), neighbor_invalid_mask.reshape(B, -1), pos.view(B, P, -1)


class StaticEncoder(nn.Module):
    def __init__(self, dim: int, drop_path_rate: float, hidden_dim: int) -> None:
        super().__init__()

        self._hidden_dim = hidden_dim

        self.projection = Mlp(
            in_features=dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        """
        Args:
            x: Static objects, shape (B, P, D), with pose and object attributes.
                P is the number of static-object slots.

        Returns:
            encoding: (B, P, hidden_dim)
            mask: (B, P), True when the static-object slot is empty
            pos: (B, P, 4 + CLASS_TYPE_NUM), object pose plus class type
        """
        B, P, _ = x.shape

        pos = x[:, :, :4].clone()  # x, y, cos, sin
        pos = add_class_type(pos, CLASS_TYPE_STATIC)

        static_invalid_mask = torch.sum(torch.ne(x[..., :10], 0), dim=-1).to(x.device) == 0
        valid_slot_mask = ~static_invalid_mask.view(-1)

        x = x.view(B * P, -1)
        x = torch.where(valid_slot_mask.view(-1, 1), x, torch.zeros_like(x))
        x_result = self.projection(x)
        x_result = x_result * valid_slot_mask.float().unsqueeze(-1)

        return x_result.view(B, P, -1), static_invalid_mask.view(B, P), pos.view(B, P, -1)


class LaneEncoder(nn.Module):
    def __init__(
        self,
        lane_len: int,
        class_type: int,
        drop_path_rate: float,
        hidden_dim: int,
        depth: int,
    ) -> None:
        super().__init__()
        tokens_mlp_dim = 64
        channels_mlp_dim = 128

        assert class_type in [CLASS_TYPE_LANE, CLASS_TYPE_ROUTE], (
            "Invalid class type for LaneEncoder"
        )

        self._lane_len = lane_len
        self._class_type = class_type

        self.speed_limit_emb = nn.Linear(1, channels_mlp_dim)
        self.unknown_speed_emb = nn.Embedding(1, channels_mlp_dim)
        self.attribute_emb = nn.Linear(5 + 2 * 10, channels_mlp_dim)  # traffic_light and line type

        self.channel_pre_project = Mlp(
            in_features=8,
            hidden_features=channels_mlp_dim,
            out_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.token_pre_project = Mlp(
            in_features=lane_len,
            hidden_features=tokens_mlp_dim,
            out_features=tokens_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        self.blocks = nn.ModuleList([
            MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for i in range(depth)
        ])

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(
            in_features=channels_mlp_dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

    def forward(
        self,
        x: torch.Tensor,
        speed_limit: torch.Tensor,
        has_speed_limit: torch.Tensor,
    ) -> EncoderOutput:
        """
        Args:
            x: Lane or route polylines, shape (B, P, V, D). The first 8 fields are
                center/left/right geometry; remaining fields are traffic and line type.
                P is the number of lane/route slots; V is the number of points per polyline.
            speed_limit: (B, P, 1)
            has_speed_limit: (B, P, 1), bool-like tensor

        Returns:
            encoding: (B, P, hidden_dim)
            mask: (B, P), True when every point in the lane/route slot is empty
            pos: (B, P, 4 + CLASS_TYPE_NUM), midpoint pose plus class type
        """
        attribute = x[:, :, 0, 8:]
        x = x[..., :8]

        pos = x[:, :, int(self._lane_len / 2), :4].clone()  # x, y, x'-x, y'-y
        heading = torch.atan2(pos[..., 3], pos[..., 2])
        pos = torch.stack(
            [pos[..., 0], pos[..., 1], torch.cos(heading), torch.sin(heading)], dim=-1
        )
        pos = add_class_type(pos, self._class_type)

        B, P, V, _ = x.shape
        polyline_point_invalid_mask = torch.sum(torch.ne(x[..., :8], 0), dim=-1).to(x.device) == 0
        lane_invalid_mask = torch.sum(~polyline_point_invalid_mask, dim=-1) == 0
        valid_slot_mask = ~lane_invalid_mask.view(-1)

        x = x.view(B * P, V, -1)

        # Preserve fixed tensor size while zeroing invalid slots.
        x = torch.where(valid_slot_mask.view(-1, 1, 1), x, torch.zeros_like(x))

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)

        x = torch.mean(x, dim=1)

        # Reshape speed_limit and traffic to match flattened dimensions
        speed_limit = speed_limit.view(B * P, 1)
        has_speed_limit = has_speed_limit.view(B * P, 1)
        attribute = attribute.view(B * P, -1)

        # Create embeddings for all positions
        speed_limit_emb = self.speed_limit_emb(speed_limit)
        unknown_speed_emb = self.unknown_speed_emb(
            torch.zeros(B * P, dtype=torch.long, device=x.device)
        )
        speed_limit_embedding = torch.where(has_speed_limit, speed_limit_emb, unknown_speed_emb)

        # Process traffic lights for all positions
        traffic_light_embedding = self.attribute_emb(attribute)

        x = x + speed_limit_embedding + traffic_light_embedding
        x = self.emb_project(self.norm(x))

        # Invalid slots stay zero so downstream mask recovery remains reliable.
        x = x * valid_slot_mask.float().unsqueeze(-1)

        return x.view(B, P, -1), lane_invalid_mask.reshape(B, -1), pos.view(B, P, -1)


class RouteEncoder(nn.Module):
    def __init__(
        self,
        route_len: int,
        route_num: int,
        drop_path_rate: float,
        hidden_dim: int,
        depth: int,
    ) -> None:
        super().__init__()

        self.lane_encoder = LaneEncoder(
            route_len,
            class_type=CLASS_TYPE_ROUTE,
            drop_path_rate=drop_path_rate,
            hidden_dim=hidden_dim,
            depth=depth,
        )
        self.route_position_embedding = nn.Parameter(torch.randn(1, route_num, hidden_dim))

    def forward(
        self,
        x: torch.Tensor,
        speed_limit: torch.Tensor,
        has_speed_limit: torch.Tensor,
    ) -> EncoderOutput:
        """
        Encode route lane tokens and add a learnable order embedding.

        Route lanes share the same geometry encoder as lanes. The extra route-position
        embedding tells the decoder where each token sits in the ordered route sequence.
        The embedding is applied only to valid route slots; mask uses True == invalid.
        """
        encoding, mask, pos = self.lane_encoder(x, speed_limit, has_speed_limit)

        route_token_num = encoding.shape[1]
        route_position_emb = self.route_position_embedding[:, :route_token_num]
        route_position_emb = route_position_emb.expand(encoding.shape[0], -1, -1)
        valid_route_slot_mask = ~mask
        encoding = encoding + route_position_emb * valid_route_slot_mask.unsqueeze(-1).float()

        return encoding, mask, pos


class LineEncoder(nn.Module):
    def __init__(
        self,
        line_len: int,
        class_type: int,
        drop_path_rate: float,
        hidden_dim: int,
        depth: int,
        point_dim: int = 2,
    ) -> None:
        super().__init__()
        self._class_type = class_type
        tokens_mlp_dim = 64
        channels_mlp_dim = 128

        self._line_len = line_len

        self.channel_pre_project = Mlp(
            in_features=point_dim,
            hidden_features=channels_mlp_dim,
            out_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.token_pre_project = Mlp(
            in_features=line_len,
            hidden_features=tokens_mlp_dim,
            out_features=tokens_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        self.blocks = nn.ModuleList([
            MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for i in range(depth)
        ])

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(
            in_features=channels_mlp_dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        """
        Args:
            x: Polygon or line-string points, shape (B, P, V, D). The first two fields
                are x, y; optional remaining fields are type indicators.
                P is the number of polygon/line-string slots; V is the number of points.

        Returns:
            encoding: (B, P, hidden_dim)
            mask: (B, P), True when every point in the slot is empty
            pos: (B, P, 4 + CLASS_TYPE_NUM), midpoint xy with neutral heading plus class type
        """
        B, P, V, D = x.shape

        pos_xy = x[:, :, int(self._line_len / 2), :2].clone()
        pos = torch.stack(
            [
                pos_xy[..., 0],
                pos_xy[..., 1],
                torch.ones_like(pos_xy[..., 0]),
                torch.zeros_like(pos_xy[..., 0]),
            ],
            dim=-1,
        )
        pos = add_class_type(pos, self._class_type)

        point_invalid_mask = torch.sum(torch.ne(x[..., :2], 0), dim=-1).to(x.device) == 0
        line_invalid_mask = torch.sum(~point_invalid_mask, dim=-1) == 0
        valid_slot_mask = ~line_invalid_mask.view(-1)

        x = x.view(B * P, V, -1)

        # Preserve fixed tensor size while zeroing invalid slots.
        x = torch.where(valid_slot_mask.view(-1, 1, 1), x, torch.zeros_like(x))

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)

        x = torch.mean(x, dim=1)

        x = self.emb_project(self.norm(x))

        # Invalid slots stay zero so downstream mask recovery remains reliable.
        x = x * valid_slot_mask.float().unsqueeze(-1)

        return x.view(B, P, -1), line_invalid_mask.reshape(B, -1), pos.view(B, P, -1)


class VectorEncoder(nn.Module):
    def __init__(
        self,
        num_float: int,
        hidden_dim: int,
        class_type: int,
        use_input_as_pos: bool = False,
    ) -> None:
        super().__init__()
        assert class_type in [
            CLASS_TYPE_GOAL_POSE,
            CLASS_TYPE_EGO_SHAPE,
            CLASS_TYPE_TURN_INDICATOR,
            CLASS_TYPE_EGO_SPEED,
        ], "Invalid class type for VectorEncoder"
        assert not use_input_as_pos or num_float >= 4, (
            "VectorEncoder requires at least 4 inputs when using input as position"
        )

        self._hidden_dim = hidden_dim
        self._class_type = class_type
        self._use_input_as_pos = use_input_as_pos

        self.projection = Mlp(
            in_features=num_float,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        """
        Encode a dense vector as one required context token.

        Args:
            x: Dense vector input, shape (B, D). Used for goal pose, ego shape,
                turn indicators, and ego speed.

        Returns:
            encoding: (B, 1, hidden_dim)
            mask: (B, 1), always False because these scalar/vector tokens are required
            pos: (B, 1, 4 + CLASS_TYPE_NUM). Uses input pose when use_input_as_pos=True;
                otherwise uses a neutral pose at the origin.
        """
        B, D = x.shape
        if self._use_input_as_pos:
            pos = x[:, :4].clone()
        else:
            pos = torch.cat(
                [
                    torch.zeros((B, 2), device=x.device, dtype=x.dtype),
                    torch.ones((B, 1), device=x.device, dtype=x.dtype),
                    torch.zeros((B, 1), device=x.device, dtype=x.dtype),
                ],
                dim=-1,
            )
        pos = pos.unsqueeze(1)  # (B, 1, D=4)
        pos = add_class_type(pos, self._class_type)

        mask = torch.zeros((B, 1), dtype=torch.bool, device=x.device)

        x = self.projection(x).unsqueeze(1)  # (B, 1, hidden_dim)

        return x, mask, pos
