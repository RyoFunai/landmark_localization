#include "landmark_localization/ransac.hpp"
#include <rclcpp/rclcpp.hpp>
#include <random>
#include <cmath>
#include <Eigen/Dense>

Ransac::Ransac(const Parameters &params) : params_(params) {}

double Ransac::normalize_angle(double angle)
{
  while (angle > M_PI)
    angle -= 2.0 * M_PI;
  while (angle <= -M_PI)
    angle += 2.0 * M_PI;
  return angle;
}

double Ransac::arrange_angle(double &angle)
{
  if (angle < 0)
    angle += M_PI / 2;
  else
    angle -= M_PI / 2;

  return -angle;
}

std::vector<LaserPoint> Ransac::rotate_points(std::vector<LaserPoint> &points, double angle)
{
  std::vector<LaserPoint> rotated_points = points;
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

bool Ransac::perform_ransac(const std::vector<Point3D> &points, std::array<float, 4> &plane_coefficients, std::vector<Point3D> &inliers)
{
  if (points.size() < 10)
  {
    RCLCPP_WARN(rclcpp::get_logger("ransac"), "Not enough points for RANSAC.");
    return false;
  }

  size_t best_inliers = 0;
  std::array<float, 4> best_plane = {0.0f, 0.0f, 0.0f, 0.0f};
  std::vector<Point3D> best_inlier_points;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, points.size() - 1);

  Eigen::Vector3f up(0.0f, 0.0f, 1.0f); // 地面の法線ベクトル

  double vertical_threshold_deg_ = 70.0;  // 70度以上を検出
  float vertical_threshold_cos = std::cos(vertical_threshold_deg_ * M_PI / 180.0f);

  for (int i = 0; i < params_.plane_iterations; ++i)
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
      if (distance < 0.02)
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
    RCLCPP_WARN(rclcpp::get_logger("ransac"), "RANSAC failed to find a plane.");
    return false;
  }
}

bool Ransac::perform_line_ransac(const std::vector<LaserPoint> &points, double &angle)
{
  if (points.size() < 2)
  {
    RCLCPP_WARN(rclcpp::get_logger("ransac"), "線のRANSACに必要な点が不足しています。");
    return false;
  }

  const float distance_threshold = 0.02;
  size_t best_inliers = 0;
  double best_slope = 0.0;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, points.size() - 1);

  for (int i = 0; i < params_.line_iterations; ++i)
  {
    int idx1 = dis(gen);
    int idx2 = dis(gen);
    if (idx1 == idx2)
      continue;

    const LaserPoint &p1 = points[idx1];
    const LaserPoint &p2 = points[idx2];

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
    RCLCPP_WARN(rclcpp::get_logger("ransac"), "線のRANSACが線を見つけるのに失敗しました。");
    return false;
  }
}

bool Ransac::check_plane_size(const std::vector<Point3D> &plane_inliers, const std::vector<LaserPoint> &rotated_inliers, double &width, double &height)
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

  width = max_y - min_y;  // 横幅
  height = max_z - min_z; // 縦幅

  // サイズの許容範囲
  double min_width = params_.landmark_width * (1.0 - params_.width_tolerance);
  double max_width = params_.landmark_width * (1.0 + params_.width_tolerance);
  double min_height = params_.landmark_height * (1.0 - params_.height_tolerance);
  double max_height = params_.landmark_height * (1.0 + params_.height_tolerance);

  // サイズチェック
  if (width < min_width || width > max_width || height < min_height || height > max_height)
  {
    RCLCPP_INFO(rclcpp::get_logger("ransac"), "max_z: %f, min_z: %f", max_z, min_z);
    RCLCPP_INFO(rclcpp::get_logger("ransac"), "max_y: %f, min_y: %f", max_y, min_y);
    RCLCPP_ERROR(rclcpp::get_logger("ransac"), "width: %f, height: %f", width, height);
    RCLCPP_WARN(rclcpp::get_logger("ransac"), "検出した面のサイズが規定範囲外です。スキップします。");
    return false;
  }
  return true;
}

std::array<double, 2> Ransac::calculate_mean(const std::vector<LaserPoint> &points)
{
  double centroid_x = 0.0, centroid_y = 0.0;
  for (const auto &pt : points)
  {
    centroid_x += pt.x;
    centroid_y += pt.y;
  }
  size_t inlier_size = points.size();
  centroid_x /= inlier_size;
  centroid_y /= inlier_size;

  return {centroid_x, centroid_y};
}

std::array<double, 2> Ransac::calculate_centroid(const std::vector<LaserPoint> &points)
{
  if (points.empty())
  {
    return {0.0, 0.0}; // 空の場合は原点を返す
  }

  // x座標とy座標それぞれについて、すべての点の値を集める
  std::vector<double> x_values;
  std::vector<double> y_values;
  x_values.reserve(points.size());
  y_values.reserve(points.size());

  for (const auto &pt : points)
  {
    x_values.push_back(pt.x);
    y_values.push_back(pt.y);
  }

  // 各座標についてソートして中央値を取得
  std::sort(x_values.begin(), x_values.end());
  std::sort(y_values.begin(), y_values.end());

  size_t mid = points.size() / 2;
  double median_x, median_y;

  if (points.size() % 2 == 0)
  {
    // 要素数が偶数の場合、中央の2つの値の平均を取る
    median_x = (x_values[mid - 1] + x_values[mid]) / 2.0;
    median_y = (y_values[mid - 1] + y_values[mid]) / 2.0;
  }
  else
  {
    // 要素数が奇数の場合、中央の値を取る
    median_x = x_values[mid];
    median_y = y_values[mid];
  }

  return {median_x, median_y};
}