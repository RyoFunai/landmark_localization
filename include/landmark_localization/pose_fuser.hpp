#pragma once
#include <vector>
#include <eigen3/Eigen/Dense>
#include "pointcloud_processor/types.hpp"
using namespace std;
using namespace Eigen;

struct CorrespondLaserPoint{
  double x;
  double y;
  double nx;
  double ny;
  CorrespondLaserPoint():x(0.0),y(0.0),nx(0.0),ny(0.0){}
};
struct NormalVector{
  double normalize_x;
  double normalize_y;
  NormalVector():normalize_x(0.0),normalize_y(0.0){}
};

class PoseFuser{
public:
  PoseFuser(){}
  ~PoseFuser(){}
  void setup(const double laser_weight, const double odom_weight_liner, const double odom_weight_angler);
  void init();
  Vector3d fuse_pose(Vector3d &laser_estimated, const Vector3d &current_scan_odom, double &vt, double &wt, const vector<LaserPoint> &src_points, const vector<LaserPoint> &global_points);

private:
  void find_correspondence(const vector<LaserPoint> &src_points, const vector<LaserPoint> &global_points, vector<CorrespondLaserPoint> &current_points, vector<CorrespondLaserPoint> &reference_points);
  CorrespondLaserPoint find_closest_vertical_point(CorrespondLaserPoint global);
  Matrix3d calc_laser_cov(const Vector3d &laser_estimated, vector<CorrespondLaserPoint> &current_points, vector<CorrespondLaserPoint> &reference_points);
  double calc_vertical_distance(const CorrespondLaserPoint current, const CorrespondLaserPoint reference, double x, double y, double yaw);
  Matrix3d calc_motion_cov(double vt, double wt);
  Matrix3d svdInverse(const Matrix3d &A);
  Matrix3d rotate_cov(const Vector3d &laser_estimated, Matrix3d &scan_odom_motion_cov);
  Vector3d fuse(Vector3d &laser_estimated, const Matrix3d &laser_cov, const Vector3d &current_scan_odom, const Matrix3d &rotate_scan_odom_motion_cov);

  vector<CorrespondLaserPoint> current_points;   //対応がとれた現在スキャンの点群
  vector<CorrespondLaserPoint> reference_points;   //対応がとれた参照スキャンの点群

  double laser_weight_;
  double odom_weight_liner_;
  double odom_weight_angler_;
  string robot_type_;
  bool accum{false};
  int count=0;
};
