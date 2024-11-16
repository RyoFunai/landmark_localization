#ifndef LANDMARK_LOCALIZATION_HPP
#define LANDMARK_LOCALIZATION_HPP

#include "landmark_localization/visibility.h"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include "pointcloud_processor/types.hpp"
#include <eigen3/Eigen/Dense>
#include "landmark_localization/pose_fuser.hpp"
#include "landmark_localization/ransac.hpp"
#include "landmark_localization/visibility.h"

class LANDMARK_LOCALIZATION_PUBLIC LandmarkLocalization : public rclcpp::Node
{
public:
  LandmarkLocalization();

private:
  void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void load_parameters();

  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
  void publish_downsampled_points(const std::vector<Point3D> &downsampled_points);
  void publish_inliers(const std::vector<Point3D> &inliers);
  void publish_plane_marker(const std::array<float, 4> &plane_coefficients);
  void publish_detected_plane_marker(double width, double height);
  void publish_robot_markers(Vector3d &robot_position);
  void publish_marker(Vector3d &marker_position);
  void timer_callback();

  template <typename PointT>
  void translate_points(std::vector<PointT> &points, const std::array<double, 2> &centroid)
  {
    for (auto &pt : points)
    {
      pt.x -= centroid[0];
      pt.y -= centroid[1];
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr downsampled_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr plane_marker_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr detected_plane_marker_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr inliers_publisher_;

  // ロボット位置用のパブリッシャーを追加
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr robot_marker_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_publisher_;
  rclcpp::TimerBase::SharedPtr timer_;

  Parameters params_;
  PoseFuser pose_fuser_;
  std::unique_ptr<Ransac> ransac; // Ransac クラスのメンバ追加
  Vector3d current_scan_odom_vec = Vector3d::Zero();
  double vt = 0.0;
  double wt = 0.0;
  Vector3d diff_odom = Vector3d::Zero();
  Vector3d odom = Vector3d::Zero();
  Vector3d last_odom = Vector3d::Zero();
  Vector3d est_diff_sum = Vector3d::Zero();
  bool first_time_ = true;
  const float distance_threshold = 0.02;
};

#endif // LANDMARK_LOCALIZATION_HPP