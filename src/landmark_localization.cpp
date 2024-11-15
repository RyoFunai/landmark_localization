#include "landmark_localization/landmark_localization.hpp"
#include "pointcloud_processor/pointcloud_processor.hpp"

#include <chrono>
#include <random>
#include <cmath>

const float DISTANCE_THRESHOLD = 0.02;

LandmarkLocalization::LandmarkLocalization() : Node("landmark_localization")
{
  subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/camera/camera/depth/color/points", 10,
      std::bind(&LandmarkLocalization::pointcloud_callback, this, std::placeholders::_1));

  downsampled_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("downsampled_points", 10);
  plane_marker_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("plane_marker", 10);
  inliers_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("inlier_points", 10);

  // ロボット位置用のパブリッシャーを初期化
  robot_marker_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("robot_marker", 10);

  load_parameters();
}

void LandmarkLocalization::pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  PointCloudProcessor processor(params_);
  std::vector<Point3D> processed_points = processor.process_pointcloud(*msg);
  std::vector<Point3D> downsampled_points = processor.get_downsampled_points();

  RCLCPP_INFO(this->get_logger(), "Received pointcloud with %zu points", msg->width * msg->height);
  RCLCPP_INFO(this->get_logger(), "Downsampled pointcloud with %zu points", downsampled_points.size());

  // downsampled_points をパブリッシュ
  sensor_msgs::msg::PointCloud2 downsampled_msg = processor.vector_to_PC2(downsampled_points);
  downsampled_publisher_->publish(downsampled_msg);

  // RANSAC を実行して平面を推定
  std::array<float, 4> plane_coefficients;
  std::vector<Point3D> inliers;
  if (perform_ransac(downsampled_points, plane_coefficients, inliers))
  {
    // インライア点群の重心を計算
    double centroid_x = 0.0, centroid_y = 0.0, centroid_z = 0.0;
    for (const auto &pt : inliers)
    {
      centroid_x += pt.x;
      centroid_y += pt.y;
      centroid_z += pt.z;
    }
    size_t inlier_size = inliers.size();
    centroid_x /= inlier_size;
    centroid_y /= inlier_size;
    centroid_z /= inlier_size;

    std::array<double, 3> centroid = {centroid_x, centroid_y, centroid_z};

    // インライア点群を重心で圧縮（原点に移動）
    for (auto &pt : inliers)
    {
      pt.x -= centroid_x;
      pt.y -= centroid_y;
      pt.z -= centroid_z;
    }

    // ロボットの相対的な位置を計算（重心の逆）
    double robot_rel_x = -centroid_x;
    double robot_rel_y = -centroid_y;
    double robot_rel_z = -centroid_z;

    RCLCPP_INFO(this->get_logger(), "Robot relative position: x=%.2f, y=%.2f, z=%.2f", robot_rel_x, robot_rel_y, robot_rel_z);

    // ロボットの相対位置をマーカーとしてパブリッシュ
    std::array<double, 3> robot_position = {robot_rel_x, robot_rel_y, robot_rel_z};
    create_robot_marker(robot_position);

    // マーカーを作成してパブリッシュ
    create_plane_marker(plane_coefficients, centroid);

    // 圧縮後のインライア点群をパブリッシュ
    sensor_msgs::msg::PointCloud2 inliers_msg = processor.vector_to_PC2(inliers);
    inliers_msg.header.frame_id = "map";
    inliers_publisher_->publish(inliers_msg);
  }
}

bool LandmarkLocalization::perform_ransac(const std::vector<Point3D> &points, std::array<float, 4> &plane_coefficients, std::vector<Point3D> &inliers)
{
  if (points.size() < 3)
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

void LandmarkLocalization::create_plane_marker(const std::array<float, 4> &plane_coefficients, const std::array<double, 3> &centroid)
{
  // RViz2 用のマーカを作成
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

  // 平面の位置を重心に設定
  geometry_msgs::msg::Point point;
  // point.x = centroid[0];
  // point.y = centroid[1];
  // point.z = centroid[2];
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

  plane_marker_publisher_->publish(plane_marker);
}

void LandmarkLocalization::create_robot_marker(const std::array<double, 3> &robot_position)
{
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

  // ロボットの向きはアイデンティティクォータニオン
  robot_marker.pose.orientation.x = 0.0;
  robot_marker.pose.orientation.y = 0.0;
  robot_marker.pose.orientation.z = 0.0;
  robot_marker.pose.orientation.w = 1.0;

  // 色と透過性を設定（例: 青色、透明度 1.0）
  robot_marker.color.r = 0.0f;
  robot_marker.color.g = 0.0f;
  robot_marker.color.b = 1.0f;
  robot_marker.color.a = 1.0f;

  // ロボットマーカーをパブリッシュ
  robot_marker_publisher_->publish(robot_marker);
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LandmarkLocalization>());
  rclcpp::shutdown();
  return 0;
}