#include "landmark_localization/landmark_localization.hpp"
#include "pointcloud_processor/pointcloud_processor.hpp"
#include <tf2/utils.h>
#include <tf2/LinearMath/Quaternion.h>
#include <chrono>
#include <random>
#include <cmath>

const float DISTANCE_THRESHOLD = 0.02;

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
  inliers_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("inlier_points", 10);

  // ロボット位置用のパブリッシャーを初期化
  robot_marker_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("robot_marker", 10);
  marker_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("laser_estimated_marker", 10);

  load_parameters();
  double laser_weight = 1.0;
  double odom_weight_liner = 1.0e-2;
  double odom_weight_angler = 1.0e-2;
  pose_fuser_.setup(laser_weight, odom_weight_liner, odom_weight_angler);
}

void LandmarkLocalization::pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  PointCloudProcessor processor(params_);
  std::vector<Point3D> processed_points = processor.process_pointcloud(*msg);
  std::vector<Point3D> downsampled_points = processor.get_downsampled_points();

  publish_downsampled_points(downsampled_points);

  // RANSAC を実行して平面を推定
  std::array<float, 4> plane_coefficients;
  std::vector<Point3D> plane_inliers;
  if (perform_ransac(downsampled_points, plane_coefficients, plane_inliers))
  {
    std::vector<Point3D> inliers_2d = plane_inliers;

    for (auto &pt : inliers_2d)
    {
      pt.z = 0.0;
    }

    // Yaw角の推定をRANSACで行う
    double angle = 0.0;
    if (perform_line_ransac(inliers_2d, angle))
    {
      angle = normalize_angle(arrange_angle(angle));
      std::vector<Point3D> rotated_inliers = rotate_points(inliers_2d, angle);
      if (!check_plane_size(plane_inliers, rotated_inliers))
        return;
      // インライア点群の重心を計算
      std::array<double, 3> centroid = calculate_centroid(rotated_inliers);

      // インライア点群を重心で圧縮（原点に移動）
      translate_points(rotated_inliers, centroid);

      // ロボットの相対的な位置を計算（重心の逆）
      std::array<double, 3> robot_position = {-centroid[0], -centroid[1], -centroid[2]};

      /////////////////////////////////////////////////////////////////////////////////////////////
      Vector3d robot_position_vec = {robot_position[0], robot_position[1], angle};
      double vt = 0.0;
      double wt = 0.0;
      std::vector<LaserPoint> rotated_inliers_lp;
      std::vector<LaserPoint> global_points;
      for (auto &pt : rotated_inliers)
      {
        LaserPoint laser_point = {pt.x, pt.y};
        rotated_inliers_lp.push_back(laser_point);
      }
      for (auto &pt : inliers_2d)
      {
        LaserPoint laser_point = {pt.x, pt.y};
        global_points.push_back(laser_point);
      }
      Vector3d laser_estimated = pose_fuser_.fuse_pose(robot_position_vec, current_scan_odom_vec, vt, wt, rotated_inliers_lp, global_points);
      publish_marker(laser_estimated[0], laser_estimated[1], laser_estimated[2]);

      /////////////////////////////////////////////////////////////////////////////////////////////
      publish_robot_markers(robot_position, angle);
      publish_plane_marker(plane_coefficients, centroid);

      publish_inliers(plane_inliers);
    }
    else
    {
      RCLCPP_WARN(this->get_logger(), "Yaw角の推定に失敗しました。");
    }
  }
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
  current_scan_odom_vec = Vector3d(x, y, yaw);
}

bool LandmarkLocalization::check_plane_size(const std::vector<Point3D> &plane_inliers, const std::vector<Point3D> &rotated_inliers)
{
    // 平面のサイズを計算
    double min_z = 100;
    double max_z = -100;
    double min_y = 100;
    double max_y = -100;

    for (const auto &pt : plane_inliers)
    {
      if (pt.z < min_z)
        min_z = pt.z;
      if (pt.z > max_z)
        max_z = pt.z;
    }
    for (const auto &pt : rotated_inliers)
    {
      if (pt.y < min_y)
        min_y = pt.y;
      if (pt.y > max_y)
        max_y = pt.y;
    }

    double width = max_y - min_y;  // 横幅
    double height = max_z - min_z; // 縦幅

    // 期待するサイズ
    const double expected_width = 0.91;
    const double expected_height = 0.6;
    const double tolerance = 0.4; // ±40%

    // サイズの許容範囲
    double min_width = expected_width * (1.0 - tolerance);
    double max_width = expected_width * (1.0 + tolerance);
    double min_height = expected_height * (1.0 - tolerance);
    double max_height = expected_height * (1.0 + tolerance);

    // サイズチェック
    if (width < min_width || width > max_width || height < min_height || height > max_height)
    {
      RCLCPP_INFO(this->get_logger(), "max_z: %f, min_z: %f", max_z, min_z);
      RCLCPP_INFO(this->get_logger(), "max_y: %f, min_y: %f", max_y, min_y);
      RCLCPP_ERROR(this->get_logger(), "width: %f, height: %f", width, height);
      RCLCPP_WARN(this->get_logger(), "検出した面のサイズが規定範囲外です。スキップします。");
      return false;
    }
    return true;
  }

  double LandmarkLocalization::arrange_angle(double &angle)
  {
    if (angle < 0)
      angle += M_PI / 2;
    else
      angle -= M_PI / 2;

    return angle;
  }

  std::vector<Point3D> LandmarkLocalization::rotate_points(std::vector<Point3D> & points, double angle)
  {
    std::vector<Point3D> rotated_points = points;
    double cos_yaw = std::cos(-angle);
    double sin_yaw = std::sin(-angle);

    for (auto &pt : rotated_points)
    {
      double x_rot = pt.x * cos_yaw - pt.y * sin_yaw;
      double y_rot = pt.x * sin_yaw + pt.y * cos_yaw;
      pt.x = x_rot;
      pt.y = y_rot;
    }
    return rotated_points;
  }

  double LandmarkLocalization::normalize_angle(double angle)
  {
    while (angle > M_PI)
      angle -= 2.0 * M_PI;
    while (angle <= -M_PI)
      angle += 2.0 * M_PI;
    return angle;
  }

  bool LandmarkLocalization::perform_ransac(const std::vector<Point3D> &points, std::array<float, 4> &plane_coefficients, std::vector<Point3D> &inliers)
  {
    if (points.size() < 30)
    {
      RCLCPP_WARN(this->get_logger(), "Not enough points for RANSAC.");
      return false;
    }

    const int max_iterations = 100;
    size_t best_inliers = 0;
    std::array<float, 4> best_plane = {0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<Point3D> best_inlier_points;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, points.size() - 1);

    Eigen::Vector3f up(0.0f, 0.0f, 1.0f); // 地面の法線ベクトル

    // 度をラジアンに変換し、コサインを計算
    float vertical_threshold_cos = std::cos(vertical_threshold_deg_ * M_PI / 180.0f);

    for (int i = 0; i < max_iterations; ++i)
    {
      // 3点をランダムに選択
      int idx1 = dis(gen);
      int idx2 = dis(gen);
      int idx3 = dis(gen);
      if (idx1 == idx2 || idx1 == idx3 || idx2 == idx3)
        continue;

      const Point3D &p1 = points[idx1];
      const Point3D &p2 = points[idx2];
      const Point3D &p3 = points[idx3];

      // 平面の法線ベクトルを計算
      float ux = p2.x - p1.x;
      float uy = p2.y - p1.y;
      float uz = p2.z - p1.z;

      float vx = p3.x - p1.x;
      float vy = p3.y - p1.y;
      float vz = p3.z - p1.z;

      // 外積で法線ベクトルを求める
      float a = uy * vz - uz * vy;
      float b = uz * vx - ux * vz;
      float c = ux * vy - uy * vx;

      // 法線ベクトルが0の場合スキップ
      if (a == 0 && b == 0 && c == 0)
        continue;

      // 平面方程式: ax + by + cz + d = 0
      float d = -(a * p1.x + b * p1.y + c * p1.z);

      // 正規化
      float norm = std::sqrt(a * a + b * b + c * c);
      if (norm == 0)
        continue;
      a /= norm;
      b /= norm;
      c /= norm;
      d /= norm;

      // 法線ベクトルの垂直性をチェック
      Eigen::Vector3f normal(a, b, c);
      float dot_product = std::abs(normal.dot(up));
      if (dot_product > vertical_threshold_cos)
      {
        // 法線ベクトルが垂直ではない（水平に近い）ためスキップ
        continue;
      }

      // インライアーを数える
      size_t inliers_count = 0;
      std::vector<Point3D> current_inliers;
      for (const auto &pt : points)
      {
        float distance = std::abs(a * pt.x + b * pt.y + c * pt.z + d);
        if (distance < DISTANCE_THRESHOLD)
        {
          inliers_count++;
          current_inliers.push_back(pt);
        }
      }

      if (inliers_count > best_inliers)
      {
        best_inliers = inliers_count;
        best_plane = {a, b, c, d};
        best_inlier_points = current_inliers;
      }
    }

    // インライアー数が閾値を超えた場合に採用
    if (best_inliers > points.size() * 0.3) // 例: 30%以上のインライアー
    {
      plane_coefficients = best_plane;
      inliers = best_inlier_points;
      return true;
    }
    else
    {
      RCLCPP_WARN(this->get_logger(), "RANSAC failed to find a plane.");
      return false;
    }
  }

  bool LandmarkLocalization::perform_line_ransac(const std::vector<Point3D> &points, double &angle)
  {
    if (points.size() < 2)
    {
      RCLCPP_WARN(this->get_logger(), "線のRANSACに必要な点が不足しています。");
      return false;
    }

    const int max_iterations = 100;
    const float distance_threshold = 0.02;
    size_t best_inliers = 0;
    double best_slope = 0.0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, points.size() - 1);

    for (int i = 0; i < max_iterations; ++i)
    {
      int idx1 = dis(gen);
      int idx2 = dis(gen);
      if (idx1 == idx2)
        continue;

      const Point3D &p1 = points[idx1];
      const Point3D &p2 = points[idx2];

      // y = mx + b の形式で線を定義
      double delta_x = p2.x - p1.x;
      double delta_y = p2.y - p1.y;

      if (delta_x == 0)
        continue; // 垂直な線はスキップ

      double slope = delta_y / delta_x;
      double intercept = p1.y - slope * p1.x;

      // インライアーを数える
      size_t inliers_count = 0;
      for (const auto &pt : points)
      {
        double distance = std::abs(slope * pt.x - pt.y + intercept) / std::sqrt(slope * slope + 1);
        if (distance < distance_threshold)
          inliers_count++;
      }

      if (inliers_count > best_inliers)
      {
        best_inliers = inliers_count;
        best_slope = slope;
      }
    }

    // インライアー数が一定以上
    if (best_inliers > points.size() * 0.3)
    {
      angle = std::atan(best_slope);
      return true;
    }
    else
    {
      RCLCPP_WARN(this->get_logger(), "線のRANSACが線を見つけるのに失敗しました。");
      return false;
    }
  }

  std::array<double, 3> LandmarkLocalization::calculate_centroid(const std::vector<Point3D> &points)
  {
    double centroid_x = 0.0, centroid_y = 0.0, centroid_z = 0.0;
    for (const auto &pt : points)
    {
      centroid_x += pt.x;
      centroid_y += pt.y;
    }
    size_t inlier_size = points.size();
    centroid_x /= inlier_size;
    centroid_y /= inlier_size;
    centroid_z = 0.0;

    return {centroid_x, centroid_y, centroid_z};
  }

  void LandmarkLocalization::translate_points(std::vector<Point3D> & points, const std::array<double, 3> &centroid)
  {
    for (auto &pt : points)
    {
      pt.x -= centroid[0];
      pt.y -= centroid[1];
    }
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

  void LandmarkLocalization::publish_plane_marker(const std::array<float, 4> &plane_coefficients, const std::array<double, 3> &centroid)
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
    plane_marker.scale.z = DISTANCE_THRESHOLD; // 厚さ

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
    Eigen::Quaternionf q;
    q.setFromTwoVectors(up, normal);
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

  void LandmarkLocalization::publish_robot_markers(const std::array<double, 3> &robot_position, double robot_yaw)
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
    position.z = robot_position[2];
    robot_marker.pose.position = position;

    // Yaw角を調整してロボットの向きを設定
    // 平面がロボットに向かっている時にYaw角が0度になるように調整
    double adjusted_yaw = robot_yaw; // 必要に応じて調整

    // クォータニオンをYaw角から計算
    geometry_msgs::msg::Quaternion orientation;
    double half_yaw = adjusted_yaw / 2.0;
    orientation.x = 0.0;
    orientation.y = 0.0;
    orientation.z = sin(half_yaw);
    orientation.w = cos(half_yaw);
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
    start_point.z = robot_position[2];

    // 矢印の終了点を調整したYaw角に基づいて設定
    double arrow_length = 0.5; // 矢印の長さを0.5mに設定
    geometry_msgs::msg::Point end_point;
    end_point.x = robot_position[0] + arrow_length * std::cos(adjusted_yaw);
    end_point.y = robot_position[1] + arrow_length * std::sin(adjusted_yaw);
    end_point.z = robot_position[2]; // 同じ高さに設定

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

  void LandmarkLocalization::publish_marker(double x, double y, double yaw)
  {
    visualization_msgs::msg::Marker marker;

    // マーカーの基本設定
    marker.header.frame_id = "map";
    marker.header.stamp = this->get_clock()->now();
    marker.ns = "laser_estimated";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position.x = x;
    marker.pose.position.y = y;
    marker.pose.position.z = 0.0;

    // クォータニオンの設定
    tf2::Quaternion q;
    q.setRPY(0, 0, yaw);
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

    // ライフタイムの設定（必要に応じて）
    // marker.lifetime = rclcpp::Duration();

    // マーカーをパブリッシュ
    marker_publisher_->publish(marker);
  }

  int main(int argc, char **argv)
  {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LandmarkLocalization>());
    rclcpp::shutdown();
    return 0;
  }

  void LandmarkLocalization::load_parameters()
  {
    this->declare_parameter("min_x", -10.0);
    this->declare_parameter("max_x", 10.0);
    this->declare_parameter("min_y", -10.0);
    this->declare_parameter("max_y", 10.0);
    this->declare_parameter("min_z", -2.0);
    this->declare_parameter("max_z", 5.0);
    this->declare_parameter("voxel_size_x", 0.1);
    this->declare_parameter("voxel_size_y", 0.1);
    this->declare_parameter("voxel_size_z", 0.1);
    this->declare_parameter("D_voxel_size_x", 0.05);
    this->declare_parameter("D_voxel_size_y", 0.05);
    this->declare_parameter("D_voxel_size_z", 0.05);
    this->declare_parameter("voxel_search_range", 3);
    this->declare_parameter("ball_radius", 0.1);

    // 新しいパラメータの宣言
    this->declare_parameter("vertical_threshold_deg", 10.0); // 例: 10度

    params_.min_x = this->get_parameter("min_x").as_double();
    params_.max_x = this->get_parameter("max_x").as_double();
    params_.min_y = this->get_parameter("min_y").as_double();
    params_.max_y = this->get_parameter("max_y").as_double();
    params_.min_z = this->get_parameter("min_z").as_double();
    params_.max_z = this->get_parameter("max_z").as_double();
    params_.voxel_size_x = this->get_parameter("voxel_size_x").as_double();
    params_.voxel_size_y = this->get_parameter("voxel_size_y").as_double();
    params_.voxel_size_z = this->get_parameter("voxel_size_z").as_double();
    params_.D_voxel_size_x = this->get_parameter("D_voxel_size_x").as_double();
    params_.D_voxel_size_y = this->get_parameter("D_voxel_size_y").as_double();
    params_.D_voxel_size_z = this->get_parameter("D_voxel_size_z").as_double();
    params_.voxel_search_range = this->get_parameter("voxel_search_range").as_int();
    params_.ball_radius = this->get_parameter("ball_radius").as_double();

    // 新しいパラメータの取得
    vertical_threshold_deg_ = this->get_parameter("vertical_threshold_deg").as_double();
  }