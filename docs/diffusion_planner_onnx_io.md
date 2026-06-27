# Diffusion Planner ONNX I/O

`ros_scripts/torch2onnx.py` exports the following ONNX files for each `latest.pth`.

| File | Purpose |
|---|---|
| `{prefix}.onnx` | Full model wrapper |
| `{prefix}_encoder.onnx` | Encoder only |
| `{prefix}_decoder.onnx` | One DiT denoising evaluation |
| `{prefix}_turn_indicator.onnx` | Turn-indicator head |
| `{prefix}_speed_predictor.onnx` | Ego future velocity head |

`B` is dynamic batch size. Other dimensions are fixed by the model config used at export time.

## Constants

Default exported dimensions from `diffusion_planner.dimensions`:

| Symbol | Value | Meaning |
|---|---:|---|
| `INPUT_T` | `30` | Past horizon excluding current |
| `OUTPUT_T` | `80` | Future horizon |
| `POSE_DIM` | `4` | `x, y, cos(yaw), sin(yaw)` |
| `MAX_NUM_NEIGHBORS` | `320` | Neighbor slots |
| `MAX_NUM_AGENTS` | `321` | Ego + neighbors |
| `NUM_STATIC_OBJECTS` | `5` | Static object slots |
| `NUM_SEGMENTS_IN_LANE` | `140` | Lane slots |
| `NUM_SEGMENTS_IN_ROUTE` | `25` | Route lane slots |
| `NUM_POLYGONS` | `10` | Polygon slots |
| `NUM_LINE_STRINGS` | `60` | Line string slots |
| `POINTS_PER_LANELET` | `20` | Lane points |
| `POINTS_PER_POLYGON` | `40` | Polygon points |
| `POINTS_PER_LINE_STRING` | `20` | Line string points |
| `SEGMENT_POINT_DIM` | `33` | Lane/route point feature dim |

## Preprocessing

Inputs must match the tensors used by training/inference:

- `ego_agent_past` and `goal_pose` are expected as `x, y, cos(yaw), sin(yaw)`.
- `ego_agent_future` for speed predictor is expected as `x, y, cos(yaw), sin(yaw)`.
- `turn_indicators` length is `INPUT_T + 1`; encoder consumes `turn_indicators[:, :-1]`.
- `ego_velocity_past` length is `INPUT_T + 1`; only `vx` is used by the ego velocity encoder.
- Observation inputs should be normalized with the same `observation_normalizer` stats from `args.json`.
- `encoding_mask` is not an ONNX input. It is reconstructed inside wrappers from zero encoder tokens.

## Full Model: `{prefix}.onnx`

Inputs:

| Name | Shape | Dtype | Notes |
|---|---|---|---|
| `sampled_trajectories` | `[B, MAX_NUM_AGENTS, OUTPUT_T + 1, 4]` | `float32` | Diffusion state. Current slot is replaced internally. |
| `ego_agent_past` | `[B, INPUT_T + 1, 4]` | `float32` | `x, y, cos, sin` |
| `ego_velocity_past` | `[B, INPUT_T + 1, 2]` | `float32` | `vx, vy`; only `vx` is used |
| `ego_current_state` | `[B, 10]` | `float32` | Current ego state |
| `neighbor_agents_past` | `[B, MAX_NUM_NEIGHBORS, INPUT_T + 1, 11]` | `float32` | Neighbor history |
| `static_objects` | `[B, NUM_STATIC_OBJECTS, 10]` | `float32` | Static objects |
| `lanes` | `[B, NUM_SEGMENTS_IN_LANE, POINTS_PER_LANELET, SEGMENT_POINT_DIM]` | `float32` | Lane vectors |
| `lanes_speed_limit` | `[B, NUM_SEGMENTS_IN_LANE, 1]` | `float32` | Lane speed limit |
| `lanes_has_speed_limit` | `[B, NUM_SEGMENTS_IN_LANE, 1]` | `bool` | Speed-limit validity |
| `route_lanes` | `[B, NUM_SEGMENTS_IN_ROUTE, POINTS_PER_LANELET, SEGMENT_POINT_DIM]` | `float32` | Route vectors |
| `route_lanes_speed_limit` | `[B, NUM_SEGMENTS_IN_ROUTE, 1]` | `float32` | Route speed limit |
| `route_lanes_has_speed_limit` | `[B, NUM_SEGMENTS_IN_ROUTE, 1]` | `bool` | Speed-limit validity |
| `polygons` | `[B, NUM_POLYGONS, POINTS_PER_POLYGON, 3]` | `float32` | `2 + POLYGON_TYPE_NUM` |
| `line_strings` | `[B, NUM_LINE_STRINGS, POINTS_PER_LINE_STRING, 4]` | `float32` | `2 + LINE_STRING_TYPE_NUM` |
| `goal_pose` | `[B, 4]` | `float32` | `x, y, cos, sin` |
| `ego_shape` | `[B, 3]` | `float32` | wheelbase, length, width |
| `turn_indicators` | `[B, INPUT_T + 1]` | `float32` | Indicator sequence |

Outputs:

| Name | Shape | Dtype |
|---|---|---|
| `prediction` | `[B, MAX_NUM_AGENTS, OUTPUT_T, 4]` | `float32` |
| `turn_indicator_logit` | `[B, 5]` | `float32` |

## Encoder: `{prefix}_encoder.onnx`

Inputs are the same observation inputs as the full model, excluding `sampled_trajectories`.

Outputs:

| Name | Shape | Dtype | Notes |
|---|---|---|---|
| `encoding` | `[B, token_num, hidden_dim]` | `float32` | Invalid tokens are all zero |

`token_num` is:

```text
1
+ agent_num
+ static_objects_num
+ lane_num
+ route_num
+ polygon_num
+ line_string_num
+ 1  # goal pose
+ 1  # ego shape
+ 1  # turn indicator
```

For the default config this is `1 + 320 + 5 + 140 + 25 + 10 + 60 + 3 = 564`.

## Decoder: `{prefix}_decoder.onnx`

This runs one DiT denoising-network evaluation. The external caller is responsible for the diffusion loop.

Inputs:

| Name | Shape | Dtype | Notes |
|---|---|---|---|
| `encoding` | `[B, token_num, hidden_dim]` | `float32` | Encoder output |
| `sampled_trajectories` | `[B, MAX_NUM_AGENTS, OUTPUT_T + 1, 4]` | `float32` | Current slot should be present |
| `diffusion_time` | `[B, MAX_NUM_AGENTS, OUTPUT_T + 1, 1]` | `float32` | Diffusion time |
| `neighbor_agents_past` | `[B, MAX_NUM_NEIGHBORS, INPUT_T + 1, 11]` | `float32` | Used for neighbor masks/classes |

Outputs:

| Name | Shape | Dtype |
|---|---|---|
| `model_output` | `[B, MAX_NUM_AGENTS, OUTPUT_T + 1, 4]` | `float32` |

## Turn Indicator: `{prefix}_turn_indicator.onnx`

Inputs:

| Name | Shape | Dtype |
|---|---|---|
| `encoding` | `[B, token_num, hidden_dim]` | `float32` |
| `final_x0` | `[B, MAX_NUM_AGENTS, OUTPUT_T + 1, 4]` | `float32` |

Outputs:

| Name | Shape | Dtype |
|---|---|---|
| `turn_indicator_logit` | `[B, 5]` | `float32` |

## Speed Predictor: `{prefix}_speed_predictor.onnx`

Inputs:

| Name | Shape | Dtype | Notes |
|---|---|---|---|
| `encoding` | `[B, token_num, hidden_dim]` | `float32` | Encoder output. Zero tokens are masked internally. |
| `ego_agent_future` | `[B, OUTPUT_T, 4]` | `float32` | Ego future path, `x, y, cos, sin` |

Outputs:

| Name | Shape | Dtype | Notes |
|---|---|---|---|
| `ego_velocity_future_prediction` | `[B, OUTPUT_T, 1]` | `float32` | Non-negative longitudinal velocity |

## Split Inference Flow

When using the split ONNX files, run them in this order:

1. Run `{prefix}_encoder.onnx` with observation inputs to get `encoding`.
2. Run `{prefix}_decoder.onnx` repeatedly from the external diffusion loop. Pass the same `encoding`, the current `sampled_trajectories`, current `diffusion_time`, and `neighbor_agents_past`.
3. Use the final denoised trajectory as `final_x0`.
4. Run `{prefix}_turn_indicator.onnx` with `encoding` and `final_x0`.
5. Run `{prefix}_speed_predictor.onnx` with `encoding` and `final_x0[:, 0, 1:, :]` as `ego_agent_future`.

The split decoder, turn-indicator, and speed-predictor wrappers all rebuild `encoding_mask` internally by treating all-zero `encoding` tokens as masked. Do not pass `encoding_mask` as an ONNX input.

## Shape Notes

- ONNX export marks only batch dimension `B` as dynamic. Time, agent, map, token, and feature dimensions are fixed by the exported checkpoint/config.
- `hidden_dim` is the model hidden size from `args.json`.
- `token_num` depends on config fields such as `agent_num`, `static_objects_num`, `lane_num`, `route_num`, `polygon_num`, and `line_string_num`. The default value is `564`.
- `prediction` from the full model does not include the current state slot. `model_output` and `final_x0` in the split pipeline include the current state slot at index `0`.
- The speed predictor is exported as a separate ONNX file. The full model ONNX output remains `prediction` and `turn_indicator_logit`.
