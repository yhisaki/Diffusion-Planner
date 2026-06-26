// Copyright 2026 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cli/converter_options.hpp"

#include <CLI/CLI.hpp>

#include <filesystem>
#include <string>

void ConverterOptions::add_converter_options(CLI::App & app)
{
  app.add_option("--step", step, "Frame sampling interval in 10 Hz ticks.");
  app.add_option(
    "--limit", limit,
    "Maximum number of rosbag messages to read. Use -1 to read "
    "all messages.");
  app.add_option(
    "--min_frames", min_frames,
    "Minimum number of assembled frames required to accept a sequence.");
  app.add_option(
    "--min_distance", min_distance,
    "Minimum traveled ego distance in meters required to accept a sequence.");
  app.add_option(
    "--future_distance_horizon_m", future_distance_horizon_m,
    "Distance horizon in meters for ego_agent_future. If less future path is available, "
    "the per-frame sampling interval is reduced to fit the available distance.");
  app.add_option(
    "--green_light_stop_line_distance_m", green_light_stop_line_distance_m,
    "Skip frames when the ego front bumper is within this stop-line distance on a green route "
    "and ego does not start. Use 0 to disable.");
  app.add_option(
    "--green_light_no_start_duration_s", green_light_no_start_duration_s,
    "Future duration in seconds used to detect no-start behavior at a green light. "
    "Use 0 to disable.");
  app.add_option(
    "--stopped_future_drop_probability", stopped_future_drop_probability,
    "Deterministic drop probability for frames whose ego_velocity_future is fully stopped.");
  app.add_option(
    "--search_nearest_route", search_nearest_route,
    "Use the latest route message at or before each frame "
    "timestamp when non-zero.");
  app.add_option(
    "--convert_yellow", convert_yellow,
    "Deprecated compatibility option. Legacy red/yellow stopped-frame skip is disabled.");
  app.add_option(
    "--convert_red", convert_red,
    "Deprecated compatibility option. Legacy red/yellow stopped-frame skip is disabled.");
  app.add_option_function<int64_t>(
    "--interpolation",
    [this](const int64_t value) {
      interpolation = value;
      use_interpolation = static_cast<bool>(interpolation);
    },
    "Use timestamp-based interpolation for ego past and future "
    "trajectories when non-zero.");
  app.add_option("--ego_wheel_base", ego_wheel_base, "Ego vehicle wheel base in meters.");
  app.add_option("--ego_length", ego_length, "Ego vehicle length in meters.");
  app.add_option("--ego_width", ego_width, "Ego vehicle width in meters.");
  app.add_option(
    "--static_object_margin", static_object_margin,
    "Additional margin in meters for static-object collision filtering.");
  app.add_option(
    "--neighbor_margin", neighbor_margin,
    "Additional margin in meters for neighbor-agent collision filtering.");
  app.add_option(
    "--disable_neighbor_collision", disable_neighbor_collision,
    "Disable neighbor-agent collision filtering when non-zero.");
  app.add_option(
    "--road_border_margin", road_border_margin,
    "Additional margin in meters for road-border collision filtering.");
  app.add_option(
    "--collision_time_stride", collision_time_stride,
    "Sample stride used when checking trajectory collision filters.");
  app.add_option(
    "--offlane_max_score", offlane_max_score,
    "Maximum average distance in meters from lane centerlines "
    "before a frame is skipped.");
  app.add_option(
    "--offlane_time_stride", offlane_time_stride,
    "Sample stride used when checking the off-lane filter.");
  app.add_option(
    "--write_skipped_npz", write_skipped_npz,
    "Also write npz files for skipped frames when non-zero. "
    "Intended for inspection.");
}

std::string ConverterPaths::get_rosbag_dir_name() const
{
  return std::filesystem::path(rosbag_path).filename();
}

ConverterOptions ConverterOptions::default_converter_options()
{
  ConverterOptions options;
  options.step = 3;
  options.limit = -1;
  options.min_frames = 1700;
  options.search_nearest_route = 1;
  options.convert_yellow = 0;
  options.convert_red = 0;
  options.interpolation = 1;
  options.min_distance = 50.0;
  options.future_distance_horizon_m = 80.0;
  options.green_light_stop_line_distance_m = 5.0;
  options.green_light_no_start_duration_s = 0.3;
  options.stopped_future_drop_probability = 0.9;
  options.ego_wheel_base = -1.0f;
  options.ego_length = -1.0f;
  options.ego_width = -1.0f;

  // Collision-free filter defaults match filter_collision_free_npz.py.
  options.static_object_margin = -100.0f;
  options.neighbor_margin = -100.0f;
  options.road_border_margin = -0.0f;
  options.disable_neighbor_collision = true;
  options.collision_time_stride = 5;

  // In-lanelet filter defaults match filter_in_lanelet_npz.py.
  options.offlane_max_score = 6.0f;
  options.offlane_time_stride = 1;

  // Inspection-only: production keeps this off so skipped frames write no npz.
  options.write_skipped_npz = false;
  options.use_interpolation = static_cast<bool>(options.interpolation);
  return options;
}

std::optional<std::string> validate_options(const ConverterOptions & opts)
{
  if (opts.ego_wheel_base < 0.0 || opts.ego_length < 0.0 || opts.ego_width < 0.0) {
    return "Ego vehicle dimensions must be non-negative.";
  }
  if (opts.future_distance_horizon_m <= 0.0) {
    return "Future distance horizon must be positive.";
  }
  if (opts.green_light_stop_line_distance_m < 0.0) {
    return "Green-light stop-line distance must be non-negative.";
  }
  if (opts.green_light_no_start_duration_s < 0.0) {
    return "Green-light no-start duration must be non-negative.";
  }
  if (opts.stopped_future_drop_probability < 0.0 || opts.stopped_future_drop_probability > 1.0) {
    return "Stopped future drop probability must be in [0, 1].";
  }
  return std::nullopt;
}
