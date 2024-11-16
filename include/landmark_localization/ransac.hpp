#pragma once

#include <vector>
#include <array>
#include "pointcloud_processor/types.hpp"

class Ransac
{
public:
  Ransac(double vertical_threshold_deg);

  bool perform_ransac(const std::vector<Point3D> &points, std::array<float, 4> &plane_coefficients, std::vector<Point3D> &inliers);
  bool perform_line_ransac(const std::vector<LaserPoint> &points, double &angle);
  std::vector<LaserPoint> rotate_points(std::vector<LaserPoint> &points, double angle);
  double normalize_angle(double angle);
  double arrange_angle(double &angle);
  bool check_plane_size(const std::vector<Point3D> &plane_inliers, const std::vector<LaserPoint> &rotated_inliers, double &width, double &height);
  std::array<double, 2> calculate_mean(const std::vector<LaserPoint> &points);
  std::array<double, 2> calculate_centroid(const std::vector<LaserPoint> &points);

private:
  double vertical_threshold_deg_;
};