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

#include "processing/ego_sequence.hpp"

#include <autoware/diffusion_planner/preprocessing/preprocessing_utils.hpp>
#include <autoware/diffusion_planner/utils/utils.hpp>

#include <algorithm>
#include <cmath>
#include <deque>

namespace
{

constexpr double kPi = 3.14159265358979323846;

struct EgoPathPoint
{
  double x;
  double y;
  double yaw;
  double distance;
};

double normalize_angle(double angle)
{
  while (angle > kPi) {
    angle -= 2.0 * kPi;
  }
  while (angle < -kPi) {
    angle += 2.0 * kPi;
  }
  return angle;
}

EgoPathPoint pose_to_path_point(
  const geometry_msgs::msg::Pose & pose, const Eigen::Matrix4d & map2bl_matrix)
{
  const Eigen::Matrix4d pose_in_map = autoware::diffusion_planner::utils::pose_to_matrix4d(pose);
  const Eigen::Matrix4d pose_in_bl = map2bl_matrix * pose_in_map;
  const double yaw = std::atan2(pose_in_bl(1, 0), pose_in_bl(0, 0));
  return {pose_in_bl(0, 3), pose_in_bl(1, 3), yaw, 0.0};
}

}  // namespace

std::optional<std::vector<float>> create_ego_sequence(
  const std::vector<FrameData> & data_list, const int64_t start_idx, const size_t num_timesteps,
  const Eigen::Matrix4d & map2bl_matrix, const rclcpp::Time & reference_time,
  const bool use_interpolation)
{
  std::deque<nav_msgs::msg::Odometry> odom_deque;

  if (use_interpolation) {
    // Collect odom messages from start_idx until timestamp >= reference_time
    for (size_t j = static_cast<size_t>(std::max(int64_t(0), start_idx)); j < data_list.size();
         ++j) {
      odom_deque.push_back(data_list[j].kinematic_state);
      if (rclcpp::Time(data_list[j].kinematic_state.header.stamp) >= reference_time) {
        break;
      }
    }

    // Error: data doesn't cover the reference_time
    if (odom_deque.empty() || rclcpp::Time(odom_deque.back().header.stamp) < reference_time) {
      return std::nullopt;
    }

    return autoware::diffusion_planner::preprocess::create_ego_agent_past(
      odom_deque, num_timesteps, map2bl_matrix, reference_time);
  }

  // Without interpolation: collect exactly num_timesteps frames by index
  for (size_t j = 0; j < num_timesteps; ++j) {
    const int64_t index =
      std::min(start_idx + static_cast<int64_t>(j), static_cast<int64_t>(data_list.size()) - 1);
    if (index < 0) {
      return std::nullopt;
    }
    odom_deque.push_back(data_list[index].kinematic_state);
  }

  if (odom_deque.empty()) {
    return std::nullopt;
  }

  return autoware::diffusion_planner::preprocess::create_ego_agent_past(
    odom_deque, num_timesteps, map2bl_matrix);
}

std::optional<EgoDistanceSequenceResult> create_ego_distance_sequence(
  const std::vector<FrameData> & data_list, const int64_t current_idx, const size_t num_timesteps,
  const Eigen::Matrix4d & map2bl_matrix, const double distance_horizon_m)
{
  if (
    current_idx < 0 || current_idx >= static_cast<int64_t>(data_list.size()) ||
    num_timesteps == 0 || distance_horizon_m <= 0.0) {
    return std::nullopt;
  }

  std::vector<EgoPathPoint> path;
  path.reserve(data_list.size() - static_cast<size_t>(current_idx));
  for (size_t j = static_cast<size_t>(current_idx); j < data_list.size(); ++j) {
    EgoPathPoint point = pose_to_path_point(data_list[j].kinematic_state.pose.pose, map2bl_matrix);
    if (!path.empty()) {
      const EgoPathPoint & prev = path.back();
      const double dx = point.x - prev.x;
      const double dy = point.y - prev.y;
      point.distance = prev.distance + std::sqrt(dx * dx + dy * dy);
    }
    path.push_back(point);
    if (point.distance >= distance_horizon_m) {
      break;
    }
  }

  if (path.empty()) {
    return std::nullopt;
  }

  const double available_distance_m = path.back().distance;
  const double effective_horizon_m = std::min(distance_horizon_m, available_distance_m);
  const double interval_m =
    effective_horizon_m > 1.0e-6 ? effective_horizon_m / static_cast<double>(num_timesteps) : 0.0;

  std::vector<float> sequence(num_timesteps * 4, 0.0f);
  size_t segment_idx = 0;
  for (size_t t = 0; t < num_timesteps; ++t) {
    const double target_distance = interval_m * static_cast<double>(t + 1);
    while (segment_idx + 1 < path.size() && path[segment_idx + 1].distance < target_distance) {
      ++segment_idx;
    }

    const EgoPathPoint & p0 = path[segment_idx];
    const EgoPathPoint & p1 = path[std::min(segment_idx + 1, static_cast<size_t>(path.size() - 1))];
    const double segment_length = std::max(p1.distance - p0.distance, 1.0e-6);
    const double ratio = std::clamp((target_distance - p0.distance) / segment_length, 0.0, 1.0);
    const double x = p0.x + ratio * (p1.x - p0.x);
    const double y = p0.y + ratio * (p1.y - p0.y);
    const double yaw = p0.yaw + ratio * normalize_angle(p1.yaw - p0.yaw);

    sequence[t * 4 + 0] = static_cast<float>(x);
    sequence[t * 4 + 1] = static_cast<float>(y);
    sequence[t * 4 + 2] = static_cast<float>(std::cos(yaw));
    sequence[t * 4 + 3] = static_cast<float>(std::sin(yaw));
  }

  return EgoDistanceSequenceResult{sequence, interval_m, available_distance_m};
}

std::optional<std::vector<float>> create_ego_velocity_sequence(
  const std::vector<FrameData> & data_list, const int64_t start_idx, const size_t num_timesteps,
  const rclcpp::Time & reference_time, const bool use_interpolation)
{
  std::deque<nav_msgs::msg::Odometry> odom_deque;

  if (use_interpolation) {
    for (size_t j = static_cast<size_t>(std::max(int64_t(0), start_idx)); j < data_list.size();
         ++j) {
      odom_deque.push_back(data_list[j].kinematic_state);
      if (rclcpp::Time(data_list[j].kinematic_state.header.stamp) >= reference_time) {
        break;
      }
    }

    if (odom_deque.empty() || rclcpp::Time(odom_deque.back().header.stamp) < reference_time) {
      return std::nullopt;
    }

    return autoware::diffusion_planner::preprocess::create_ego_velocity(
      odom_deque, num_timesteps, reference_time);
  }

  for (size_t j = 0; j < num_timesteps; ++j) {
    const int64_t index =
      std::min(start_idx + static_cast<int64_t>(j), static_cast<int64_t>(data_list.size()) - 1);
    if (index < 0) {
      return std::nullopt;
    }
    odom_deque.push_back(data_list[index].kinematic_state);
  }

  if (odom_deque.empty()) {
    return std::nullopt;
  }

  return autoware::diffusion_planner::preprocess::create_ego_velocity(odom_deque, num_timesteps);
}

std::optional<std::vector<float>> create_ego_acceleration_sequence(
  const std::vector<FrameData> & data_list, const int64_t start_idx, const size_t num_timesteps,
  const rclcpp::Time & reference_time, const bool use_interpolation)
{
  std::deque<geometry_msgs::msg::AccelWithCovarianceStamped> accel_deque;

  if (use_interpolation) {
    for (size_t j = static_cast<size_t>(std::max(int64_t(0), start_idx)); j < data_list.size();
         ++j) {
      accel_deque.push_back(data_list[j].acceleration);
      if (rclcpp::Time(data_list[j].acceleration.header.stamp) >= reference_time) {
        break;
      }
    }

    if (accel_deque.empty() || rclcpp::Time(accel_deque.back().header.stamp) < reference_time) {
      return std::nullopt;
    }

    return autoware::diffusion_planner::preprocess::create_ego_acceleration(
      accel_deque, num_timesteps, reference_time);
  }

  for (size_t j = 0; j < num_timesteps; ++j) {
    const int64_t index =
      std::min(start_idx + static_cast<int64_t>(j), static_cast<int64_t>(data_list.size()) - 1);
    if (index < 0) {
      return std::nullopt;
    }
    accel_deque.push_back(data_list[index].acceleration);
  }

  if (accel_deque.empty()) {
    return std::nullopt;
  }

  return autoware::diffusion_planner::preprocess::create_ego_acceleration(
    accel_deque, num_timesteps);
}
