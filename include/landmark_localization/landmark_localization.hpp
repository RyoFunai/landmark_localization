#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include "pointcloud_processor/types.hpp"
#include <eigen3/Eigen/Dense>
#include "landmark_localization/pose_fuser.hpp"
class LandmarkLocalization : public rclcpp::Node
{
public:
  LandmarkLocalization();

private:
  void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void load_parameters();
  bool perform_ransac(const std::vector<Point3D> &points, std::array<float, 4> &plane_coefficients, std::vector<Point3D> &inliers);
  bool perform_line_ransac(const std::vector<Point3D> &points, double &angle);
  double normalize_angle(double angle);
  double arrange_angle(double &angle);
  std::vector<Point3D> rotate_points(std::vector<Point3D> &points, double angle);

  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
  void publish_downsampled_points(const std::vector<Point3D> &downsampled_points);
  void publish_inliers(const std::vector<Point3D> &inliers);
  void publish_plane_marker(const std::array<float, 4> &plane_coefficients, const std::array<double, 3> &centroid);
  void publish_robot_markers(const std::array<double, 3> &robot_position, double robot_yaw);
  void publish_marker(double x, double y, double yaw);

  std::array<double, 3> calculate_centroid(const std::vector<Point3D> &points);
  void translate_points(std::vector<Point3D> &points, const std::array<double, 3> &centroid);
  bool check_plane_size(const std::vector<Point3D> &plane_inliers, const std::vector<Point3D> &rotated_inliers);
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr downsampled_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr plane_marker_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr inliers_publisher_;

  // ロボット位置用のパブリッシャーを追加
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr robot_marker_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_publisher_;

  Parameters params_;
  PoseFuser pose_fuser_;
  double vertical_threshold_deg_;
  Vector3d current_scan_odom_vec = Vector3d(0.0, 0.0, 0.0);
  double vt = 0.0;
  double wt = 0.0;
};
