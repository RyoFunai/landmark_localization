#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include "pointcloud_processor/types.hpp"
#include <eigen3/Eigen/Dense>

class LandmarkLocalization : public rclcpp::Node
{
public:
  LandmarkLocalization();

private:
  void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void load_parameters();
  bool perform_ransac(const std::vector<Point3D> &points, std::array<float, 4> &plane_coefficients, std::vector<Point3D> &inliers);
  void create_plane_marker(const std::array<float, 4> &plane_coefficients, const std::array<double, 3> &centroid);
  void create_robot_marker(const std::array<double, 3> &robot_position, double robot_yaw);
  bool perform_line_ransac(const std::vector<Point3D> &points, double &yaw_angle);
  double normalize_angle(double angle);

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr downsampled_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr plane_marker_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr inliers_publisher_;

  // ロボット位置用のパブリッシャーを追加
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr robot_marker_publisher_;

  Parameters params_;

  double vertical_threshold_deg_;
};
