#include "landmark_localization/landmark_localization.hpp"
#include "landmark_localization/ransac.hpp"
#include "pointcloud_processor/pointcloud_processor.hpp"
#include <tf2/utils.h>
#include <tf2/LinearMath/Quaternion.h>
#include <chrono>
#include <random>
#include <cmath>
#include <chrono>

namespace landmark_localization
{
  LandmarkLocalization::LandmarkLocalization(const rclcpp::NodeOptions &options) : LandmarkLocalization("", options) {}

  LandmarkLocalization::LandmarkLocalization(const string &name_space, const rclcpp::NodeOptions &options)
      : rclcpp::Node("landmark_localization", name_space, options)
  {
    RCLCPP_INFO(this->get_logger(), "landmark_localization initializing...");
    subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar", 10,
        std::bind(&LandmarkLocalization::pointcloud_callback, this, std::placeholders::_1));
    odom_subscription_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/odom", 10,
        std::bind(&LandmarkLocalization::odom_callback, this, std::placeholders::_1));

    downsampled_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("downsampled_points", 10);
    plane_marker_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("plane_marker", 10);
    detected_plane_marker_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("detected_plane_marker", 10);
    inliers_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("inlier_points", 10);
    pose_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("self_pose", 10);
    laser_pose_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("laser_pose", 10);
    timer_ = this->create_wall_timer(50ms, std::bind(&LandmarkLocalization::timer_callback, this));

    load_parameters();
    pose_fuser_.setup(params_.laser_weight, params_.odom_weight_liner, params_.odom_weight_angler);
    ransac = std::make_unique<Ransac>(params_);

    RCLCPP_INFO(this->get_logger(), "landmark_localization initialized");
  }

  void LandmarkLocalization::pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    auto start = std::chrono::high_resolution_clock::now();
    PointCloudProcessor processor(params_);
    std::vector<Point3D> tmp_points = processor.PC2_to_vector(*msg);
    std::vector<Point3D> downsampled_points;
    if (!first_detect_plane)
    {
      downsampled_points = processor.filter_points_pre(tmp_points);
    }
    else{
      downsampled_points = processor.filter_points_base_origin(self_pose[0], self_pose[1], self_pose[2], tmp_points);
    }
    publish_downsampled_points(downsampled_points);

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
        std::array<double, 2> centroid = ransac->calculate_centroid(inliers_2d);

        // インライア点群を重心で圧縮（原点に移動）
        translate_points<LaserPoint>(inliers_2d, centroid);
        translate_points<LaserPoint>(rotated_inliers, centroid);

        /////////////////////////////////////////////////////////////////////////////////////////////

        Vector3d robot_position_vec = {-centroid[0], -centroid[1], angle};
        if (!first_detect_plane)
        {
          first_detect_plane = true;
          est_diff_sum = robot_position_vec;
        }

        // std::vector<LaserPoint> global_points;
        // for (auto &pt : inliers_2d)
        // {
        //   LaserPoint laser_point = {pt.x, pt.y};
        //   global_points.push_back(laser_point);
        // }
        Vector3d current_scan_odom = odom + est_diff_sum;

        vt = sqrt(pow(diff_odom[0], 2) + pow(diff_odom[1], 2)) / duration;
        wt = abs(diff_odom[2]) / duration;
        if (!isfinite(vt))
        {
          vt = 0.0;
        }
        if (!isfinite(wt))
        {
          wt = 0.0;
        }
        Vector3d estimated = pose_fuser_.fuse_pose(robot_position_vec, current_scan_odom, vt, wt, inliers_2d, rotated_inliers);
        Vector3d est_diff = estimated - current_scan_odom;
        est_diff_sum += est_diff;
        /////////////////////////////////////////////////////////////////////////////////////////////
        publish_plane_marker(plane_coefficients);
        publish_detected_plane_marker(width, height);
        translate_points<Point3D>(plane_inliers, centroid);
        publish_inliers(plane_inliers);
      }
      else
      {
        RCLCPP_WARN(this->get_logger(), "Yaw角の推定に失敗しました。");
      }
    }
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    // RCLCPP_INFO(this->get_logger(), "localization time: %ld ms", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count());
  }

  void LandmarkLocalization::timer_callback()
  {
    robot_pose[0] = est_diff_sum[0] - params_.odom2laser_x * std::cos(est_diff_sum[2]) - params_.odom2laser_y * std::sin(est_diff_sum[2]); // laserの位置が求まったので、odomの位置に変換
    robot_pose[1] = est_diff_sum[1] - params_.odom2laser_x * std::sin(est_diff_sum[2]) + params_.odom2laser_y * std::cos(est_diff_sum[2]);
    robot_pose[2] = est_diff_sum[2];
    self_pose = odom + robot_pose;
    Vector3d laser_pose = odom + est_diff_sum;
    geometry_msgs::msg::PoseStamped pose_msg = convert_to_pose_stamped(self_pose);
    geometry_msgs::msg::PoseStamped laser_pose_msg = convert_to_pose_stamped(laser_pose);
    pose_publisher_->publish(pose_msg);
    laser_pose_publisher_->publish(laser_pose_msg);
  }

  geometry_msgs::msg::PoseStamped LandmarkLocalization::convert_to_pose_stamped(Vector3d &pose)
  {
    // PoseStampedメッセージの作成と公開
    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = this->get_clock()->now();
    pose_msg.header.frame_id = "map";

    pose_msg.pose.position.x = pose[0];
    pose_msg.pose.position.y = pose[1];
    pose_msg.pose.position.z = 0.0;

    tf2::Quaternion q;
    q.setRPY(0, 0, pose[2]);
    pose_msg.pose.orientation.x = q.x();
    pose_msg.pose.orientation.y = q.y();
    pose_msg.pose.orientation.z = q.z();
    pose_msg.pose.orientation.w = q.w();
    return pose_msg;
  }

  void LandmarkLocalization::odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    // オドメトリデータからx, y, yawを取得
    double x = msg->pose.pose.position.x;
    double y = msg->pose.pose.position.y;

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
    // this->declare_parameter("min_x", -10.0);
    // this->declare_parameter("max_x", 10.0);
    // this->declare_parameter("min_y", -10.0);
    // this->declare_parameter("max_y", 10.0);
    // this->declare_parameter("min_z", -2.0);
    // this->declare_parameter("max_z", 5.0);
    // this->declare_parameter("D_voxel_size_x", 0.1);
    // this->declare_parameter("D_voxel_size_y", 0.1);
    // this->declare_parameter("D_voxel_size_z", 0.1);

    // this->declare_parameter("landmark_width", 0.9);
    // this->declare_parameter("landmark_height", 0.91);
    // this->declare_parameter("width_tolerance", 0.1);
    // this->declare_parameter("height_tolerance", 0.4);
    // this->declare_parameter("laser_weight", 1.0);
    // this->declare_parameter("odom_weight_liner", 1.0e-2);
    // this->declare_parameter("odom_weight_angler", 1.0e-2);
    // this->declare_parameter("plane_iterations", 100);
    // this->declare_parameter("line_iterations", 100);
    // this->declare_parameter("odom2laser_x", 0.0);
    // this->declare_parameter("odom2laser_y", 0.0);

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
    params_.odom2laser_x = this->get_parameter("odom2laser_x").as_double();
    params_.odom2laser_y = this->get_parameter("odom2laser_y").as_double();
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
}

// int main(int argc, char **argv)
// {
//   rclcpp::init(argc, argv);
//   rclcpp::spin(std::make_shared<LandmarkLocalization>());
//   rclcpp::shutdown();
//   return 0;
// }
