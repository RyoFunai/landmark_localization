#include "landmark_localization/landmark_localization.hpp"
#include "landmark_localization/ransac.hpp"
#include "pointcloud_processor/pointcloud_processor.hpp"
#include <tf2/utils.h>
#include <tf2/LinearMath/Quaternion.h>
#include <chrono>
#include <random>
#include <cmath>
#include <chrono>

LandmarkLocalization::LandmarkLocalization() : Node("landmark_localization")
{
  subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/camera/camera/depth/color/points", 10,
      std::bind(&LandmarkLocalization::pointcloud_callback, this, std::placeholders::_1));
  odom_subscription_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/odom", 10,
      std::bind(&LandmarkLocalization::odom_callback, this, std::placeholders::_1));

  downsampled_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("downsampled_points", 10);
  plane_marker_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("plane_marker", 10);
  detected_plane_marker_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("detected_plane_marker", 10);
  inliers_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("inlier_points", 10);
  timer_ = this->create_wall_timer(100ms, std::bind(&LandmarkLocalization::timer_callback, this));

  // ロボット位置用のパブリッシャーを初期化
  robot_marker_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("robot_marker", 10);
  marker_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("laser_estimated_marker", 10);

  load_parameters();
  pose_fuser_.setup(params_.laser_weight, params_.odom_weight_liner, params_.odom_weight_angler);
}

void LandmarkLocalization::pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  auto start = std::chrono::high_resolution_clock::now();
  PointCloudProcessor processor(params_);
  std::vector<Point3D> processed_points = processor.process_pointcloud(*msg);
  std::vector<Point3D> downsampled_points = processor.get_downsampled_points();
  RCLCPP_INFO(this->get_logger(), "processed_points.size(): %ld", processed_points.size());
  RCLCPP_INFO(this->get_logger(), "downsampled_points.size(): %ld", downsampled_points.size());

  // RANSAC を実行して平面を推定
  std::array<float, 4> plane_coefficients;
  std::vector<Point3D> plane_inliers;
  if (ransac->perform_ransac(downsampled_points, plane_coefficients, plane_inliers))
  {
    std::vector<LaserPoint> inliers_2d;
    for (auto pt : plane_inliers)
    {
      LaserPoint laser_point = {pt.x, pt.y};
      inliers_2d.push_back(laser_point);
    }

    // Yaw角の推定をRANSACで行う
    double angle = 0.0;
    if (ransac->perform_line_ransac(inliers_2d, angle))
    {
      angle = ransac->normalize_angle(ransac->arrange_angle(angle));
      std::vector<LaserPoint> rotated_inliers = ransac->rotate_points(inliers_2d, angle);
      double width = 0.0;
      double height = 0.0;
      if (!ransac->check_plane_size(plane_inliers, rotated_inliers, width, height))
        return;
      // インライア点群の重心を計算
      std::array<double, 2> centroid = ransac->calculate_centroid(rotated_inliers);

      // インライア点群を重心で圧縮（原点に移動）
      translate_points<LaserPoint>(inliers_2d, centroid);
      translate_points<LaserPoint>(rotated_inliers, centroid);

      /////////////////////////////////////////////////////////////////////////////////////////////
      Vector3d current_scan_odom = odom + est_diff_sum;

      Vector3d robot_position_vec = {-centroid[0], -centroid[1], angle};
      if (first_time_)
      {
        first_time_ = false;
        est_diff_sum = robot_position_vec;
      }
      std::vector<LaserPoint> global_points;
      for (auto &pt : inliers_2d)
      {
        LaserPoint laser_point = {pt.x, pt.y};
        global_points.push_back(laser_point);
      }
      Vector3d estimated = pose_fuser_.fuse_pose(robot_position_vec, current_scan_odom, vt, wt, inliers_2d, rotated_inliers);
      Vector3d est_diff = estimated - current_scan_odom; // 直線からの推定値がデフォルト
      est_diff_sum += est_diff;
      /////////////////////////////////////////////////////////////////////////////////////////////
      publish_robot_markers(robot_position_vec);
      publish_plane_marker(plane_coefficients);
      publish_detected_plane_marker(width, height);
      translate_points<Point3D>(plane_inliers, centroid);
      publish_inliers(plane_inliers);
      publish_downsampled_points(downsampled_points);
    }
    else
    {
      RCLCPP_WARN(this->get_logger(), "Yaw角の推定に失敗しました。");
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  RCLCPP_INFO(this->get_logger(), "localization time: %ld ms", duration);
}

void LandmarkLocalization::timer_callback()
{
  Vector3d marker_position = odom + est_diff_sum;
  publish_marker(marker_position);
}

void LandmarkLocalization::odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  // オドメトリデータからx, y, yawを取得
  double x = msg->pose.pose.position.x;
  double y = msg->pose.pose.position.y;
  vt = msg->twist.twist.linear.x;
  wt = msg->twist.twist.angular.z;

  // クォータニオンからヨー角を計算
  tf2::Quaternion q(
      msg->pose.pose.orientation.x,
      msg->pose.pose.orientation.y,
      msg->pose.pose.orientation.z,
      msg->pose.pose.orientation.w);
  double yaw = tf2::getYaw(q);

  diff_odom[0] = x - last_odom[0];
  diff_odom[1] = y - last_odom[1];
  diff_odom[2] = yaw - last_odom[2];

  odom[0] += diff_odom[0];
  odom[1] += diff_odom[1];
  odom[2] += diff_odom[2];
  last_odom[0] = x;
  last_odom[1] = y;
  last_odom[2] = yaw;
}

void LandmarkLocalization::load_parameters()
{
  this->declare_parameter("min_x", -10.0);
  this->declare_parameter("max_x", 10.0);
  this->declare_parameter("min_y", -10.0);
  this->declare_parameter("max_y", 10.0);
  this->declare_parameter("min_z", -2.0);
  this->declare_parameter("max_z", 5.0);
  this->declare_parameter("D_voxel_size_x", 0.1);
  this->declare_parameter("D_voxel_size_y", 0.1);
  this->declare_parameter("D_voxel_size_z", 0.1);

  this->declare_parameter("landmark_width", 0.9);
  this->declare_parameter("landmark_height", 0.91);
  this->declare_parameter("width_tolerance", 0.1);
  this->declare_parameter("height_tolerance", 0.4);
  this->declare_parameter("laser_weight", 1.0);
  this->declare_parameter("odom_weight_liner", 1.0e-2);
  this->declare_parameter("odom_weight_angler", 1.0e-2);
  this->declare_parameter("plane_iterations", 100);
  this->declare_parameter("line_iterations", 100);

  params_.min_x = this->get_parameter("min_x").as_double();
  params_.max_x = this->get_parameter("max_x").as_double();
  params_.min_y = this->get_parameter("min_y").as_double();
  params_.max_y = this->get_parameter("max_y").as_double();
  params_.min_z = this->get_parameter("min_z").as_double();
  params_.max_z = this->get_parameter("max_z").as_double();
  params_.D_voxel_size_x = this->get_parameter("D_voxel_size_x").as_double();
  params_.D_voxel_size_y = this->get_parameter("D_voxel_size_y").as_double();
  params_.D_voxel_size_z = this->get_parameter("D_voxel_size_z").as_double();

  params_.landmark_width = this->get_parameter("landmark_width").as_double();
  params_.landmark_height = this->get_parameter("landmark_height").as_double();
  params_.laser_weight = this->get_parameter("laser_weight").as_double();
  params_.odom_weight_liner = this->get_parameter("odom_weight_liner").as_double();
  params_.odom_weight_angler = this->get_parameter("odom_weight_angler").as_double();
  params_.width_tolerance = this->get_parameter("width_tolerance").as_double();
  params_.height_tolerance = this->get_parameter("height_tolerance").as_double();
  params_.plane_iterations = this->get_parameter("plane_iterations").as_int();
  params_.line_iterations = this->get_parameter("line_iterations").as_int();

  ransac = std::make_unique<Ransac>(params_);
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LandmarkLocalization>());
  rclcpp::shutdown();
  return 0;
}

void LandmarkLocalization::publish_downsampled_points(const std::vector<Point3D> &downsampled_points)
{
  PointCloudProcessor processor(params_);
  sensor_msgs::msg::PointCloud2 downsampled_msg = processor.vector_to_PC2(downsampled_points);
  downsampled_msg.header.frame_id = "map";
  downsampled_publisher_->publish(downsampled_msg);
}

void LandmarkLocalization::publish_inliers(const std::vector<Point3D> &inliers)
{
  PointCloudProcessor processor(params_);
  sensor_msgs::msg::PointCloud2 inliers_msg = processor.vector_to_PC2(inliers);
  inliers_msg.header.frame_id = "map";
  inliers_publisher_->publish(inliers_msg);
}

void LandmarkLocalization::publish_plane_marker(const std::array<float, 4> &plane_coefficients)
{
  visualization_msgs::msg::Marker plane_marker;
  plane_marker.header.frame_id = "map";
  plane_marker.ns = "planes";
  plane_marker.id = 0;
  plane_marker.type = visualization_msgs::msg::Marker::CUBE;
  plane_marker.action = visualization_msgs::msg::Marker::ADD;

  // 平面のサイズ（横910mm、縦600mm）
  plane_marker.scale.x = 0.6;                // 横
  plane_marker.scale.y = 0.91;               // 縦
  plane_marker.scale.z = distance_threshold; // 厚さ

  // 平面の位置を原点に設定（重心を基準に移動済み）
  geometry_msgs::msg::Point point;
  point.x = 0.0;
  point.y = 0.0;
  point.z = 0.0;
  plane_marker.pose.position = point;

  // 平面の向きをクォータニオンで設定
  // 平面の係数 ax + by + cz + d = 0 から法線ベクトルを取得
  float a = plane_coefficients[0];
  float b = plane_coefficients[1];
  float c = plane_coefficients[2];
  float d = plane_coefficients[3];

  Eigen::Vector3f normal(a, b, c);
  normal.normalize();
  Eigen::Vector3f up(0.0, 0.0, 1.0);
  tf2::Quaternion q;
  q.setRPY(0.0, M_PI / 2, 0.0);
  plane_marker.pose.orientation.x = q.x();
  plane_marker.pose.orientation.y = q.y();
  plane_marker.pose.orientation.z = q.z();
  plane_marker.pose.orientation.w = q.w();

  // 色と透過性を設定
  plane_marker.color.r = 0.0f;
  plane_marker.color.g = 1.0f;
  plane_marker.color.b = 0.0f;
  plane_marker.color.a = 0.5f;

  // 平面マーカーをパブリッシュ
  plane_marker_publisher_->publish(plane_marker);
}

void LandmarkLocalization::publish_detected_plane_marker(double width, double height)
{
  visualization_msgs::msg::Marker plane_marker;
  plane_marker.header.frame_id = "map";
  plane_marker.ns = "detected_plane";
  plane_marker.id = 0;
  plane_marker.type = visualization_msgs::msg::Marker::CUBE;
  plane_marker.action = visualization_msgs::msg::Marker::ADD;

  plane_marker.scale.x = height;             // 横
  plane_marker.scale.y = width;              // 縦
  plane_marker.scale.z = distance_threshold; // 厚さ

  geometry_msgs::msg::Point point;
  point.x = 0.0;
  point.y = 0.0;
  point.z = 0.0;
  plane_marker.pose.position = point;

  tf2::Quaternion q;
  q.setRPY(0.0, M_PI / 2, 0.0);
  plane_marker.pose.orientation.x = q.x();
  plane_marker.pose.orientation.y = q.y();
  plane_marker.pose.orientation.z = q.z();
  plane_marker.pose.orientation.w = q.w();

  // 色と透過性を設定
  plane_marker.color.r = 0.0f;
  plane_marker.color.g = 0.0f;
  plane_marker.color.b = 1.0f;
  plane_marker.color.a = 0.5f;

  // 平面マーカーをパブリッシュ
  detected_plane_marker_publisher_->publish(plane_marker);
}

void LandmarkLocalization::publish_robot_markers(Vector3d &robot_position)
{
  // ロボットの位置マーカー（球体）
  visualization_msgs::msg::Marker robot_marker;
  robot_marker.header.frame_id = "map";
  robot_marker.ns = "robot";
  robot_marker.id = 1;
  robot_marker.type = visualization_msgs::msg::Marker::SPHERE;
  robot_marker.action = visualization_msgs::msg::Marker::ADD;

  // ロボットのサイズを設定（例: 0.2m の球体）
  robot_marker.scale.x = 0.2;
  robot_marker.scale.y = 0.2;
  robot_marker.scale.z = 0.2;

  // ロボットの位置を設定
  geometry_msgs::msg::Point position;
  position.x = robot_position[0];
  position.y = robot_position[1];
  robot_marker.pose.position = position;

  // Yaw角を調整してロボットの向きを設定
  // 平面がロボットに向かっている時にYaw角が0度になるように調整

  // クォータニオンをYaw角から計算
  geometry_msgs::msg::Quaternion orientation;
  orientation.x = 0.0;
  orientation.y = 0.0;
  orientation.z = sin(robot_position[2] / 2.0);
  orientation.w = cos(robot_position[2] / 2.0);
  robot_marker.pose.orientation = orientation;

  // 色と透過性を設定（例: 青色、透明度 1.0）
  robot_marker.color.r = 0.0f;
  robot_marker.color.g = 0.0f;
  robot_marker.color.b = 1.0f;
  robot_marker.color.a = 1.0f;

  // ロボットマーカーをパブリッシュ
  robot_marker_publisher_->publish(robot_marker);

  // ロボットの向きを示すベクトルマーカーを作成
  visualization_msgs::msg::Marker robot_vector;
  robot_vector.header.frame_id = "map";
  robot_vector.ns = "robot_vector";
  robot_vector.id = 2;
  robot_vector.type = visualization_msgs::msg::Marker::ARROW;
  robot_vector.action = visualization_msgs::msg::Marker::ADD;

  // 矢印の開始点をロボットの位置に設定
  geometry_msgs::msg::Point start_point;
  start_point.x = robot_position[0];
  start_point.y = robot_position[1];
  start_point.z = 0.0;

  // 矢印の終了点を調整したYaw角に基づいて設定
  double arrow_length = 0.5; // 矢印の長さを0.5mに設定
  geometry_msgs::msg::Point end_point;
  end_point.x = robot_position[0] + arrow_length * std::cos(robot_position[2]);
  end_point.y = robot_position[1] + arrow_length * std::sin(robot_position[2]);
  end_point.z = 0.0;

  robot_vector.points.push_back(start_point);
  robot_vector.points.push_back(end_point);

  // 矢印のサイズを設定
  robot_vector.scale.x = 0.05; // 矢印の径
  robot_vector.scale.y = 0.1;  // 矢じりの幅
  robot_vector.scale.z = 0.1;  // 矢じりの高さ

  // 矢印の色と透明度を設定（例: 赤色、透明度 1.0）
  robot_vector.color.r = 1.0f;
  robot_vector.color.g = 0.0f;
  robot_vector.color.b = 0.0f;
  robot_vector.color.a = 1.0f;

  // 矢印マーカーをパブリッシュ
  robot_marker_publisher_->publish(robot_vector);
}

void LandmarkLocalization::publish_marker(Vector3d &marker_position)
{
  visualization_msgs::msg::Marker marker;

  // マーカーの基本設定
  marker.header.frame_id = "map";
  marker.header.stamp = this->get_clock()->now();
  marker.ns = "laser_estimated";
  marker.id = 0;
  marker.type = visualization_msgs::msg::Marker::SPHERE;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.pose.position.x = marker_position[0];
  marker.pose.position.y = marker_position[1];
  marker.pose.position.z = 0.0;

  // クォータニオンの設定
  tf2::Quaternion q;
  q.setRPY(0, 0, marker_position[2]);
  marker.pose.orientation.x = q.x();
  marker.pose.orientation.y = q.y();
  marker.pose.orientation.z = q.z();
  marker.pose.orientation.w = q.w();

  // サイズとカラーの設定
  marker.scale.x = 0.2;
  marker.scale.y = 0.2;
  marker.scale.z = 0.2;
  marker.color.a = 1.0; // 不透明
  marker.color.r = 1.0;
  marker.color.g = 0.0;
  marker.color.b = 0.0;

  // マーカーをパブリッシュ
  marker_publisher_->publish(marker);
}