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

#ifndef PROCESSING__EGO_SEQUENCE_HPP_
#define PROCESSING__EGO_SEQUENCE_HPP_

#include "types/frame_data.hpp"

#include <Eigen/Core>
#include <rclcpp/time.hpp>

#include <cstdint>
#include <optional>
#include <vector>

struct EgoDistanceSequenceResult
{
  std::vector<float> sequence;
  double interval_m;
  double available_distance_m;
};

std::optional<std::vector<float>> create_ego_sequence(
  const std::vector<FrameData> & data_list, const int64_t start_idx, const size_t num_timesteps,
  const Eigen::Matrix4d & map2bl_matrix, const rclcpp::Time & reference_time,
  const bool use_interpolation);

std::optional<EgoDistanceSequenceResult> create_ego_distance_sequence(
  const std::vector<FrameData> & data_list, const int64_t current_idx, const size_t num_timesteps,
  const Eigen::Matrix4d & map2bl_matrix, const double distance_horizon_m);

std::optional<std::vector<float>> create_ego_velocity_sequence(
  const std::vector<FrameData> & data_list, const int64_t start_idx, const size_t num_timesteps,
  const rclcpp::Time & reference_time, const bool use_interpolation);

std::optional<std::vector<float>> create_ego_acceleration_sequence(
  const std::vector<FrameData> & data_list, const int64_t start_idx, const size_t num_timesteps,
  const rclcpp::Time & reference_time, const bool use_interpolation);

#endif  // PROCESSING__EGO_SEQUENCE_HPP_
