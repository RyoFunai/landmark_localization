#include "landmark_localization/pose_fuser.hpp"
#include <iostream>
void PoseFuser::setup(const double laser_weight, const double odom_weight_liner, const double odom_weight_angler)
{
  laser_weight_ = laser_weight;
  odom_weight_liner_ = odom_weight_liner;
  odom_weight_angler_ = odom_weight_angler;
}

void PoseFuser::init()
{
  current_points.clear();
  reference_points.clear();
}

Vector3d PoseFuser::fuse_pose(Vector3d &laser_estimated, const Vector3d &current_scan_odom, double &vt, double &wt, const vector<LaserPoint> &src_points, const vector<LaserPoint> &global_points)
{
  init();
  if (global_points.size() == 0)
    return current_scan_odom;
  find_correspondence(src_points, global_points, current_points, reference_points);
  Matrix3d laser_cov = laser_weight_ * calc_laser_cov(laser_estimated, current_points, reference_points);
  Matrix3d scan_odom_motion_cov = calc_motion_cov(vt, wt);
  Matrix3d rotate_scan_odom_motion_cov = rotate_cov(laser_estimated, scan_odom_motion_cov);
  return fuse(laser_estimated, laser_cov, current_scan_odom, rotate_scan_odom_motion_cov);
}

void PoseFuser::find_correspondence(const vector<LaserPoint> &src_points, const vector<LaserPoint> &global_points, vector<CorrespondLaserPoint> &current_points, vector<CorrespondLaserPoint> &reference_points)
{
  CorrespondLaserPoint global;
  CorrespondLaserPoint current;
  double sum_x = 0.0;
  double sum_y = 0.0;
  for (size_t i = 0; i < global_points.size(); i++)
  {
    current.x = src_points[i].x;
    current.y = src_points[i].y;
    global.x = global_points[i].x;
    global.y = global_points[i].y;
    CorrespondLaserPoint closest_reference = find_closest_vertical_point(global);
    current_points.push_back(current);
    reference_points.push_back(closest_reference);
  }
}

CorrespondLaserPoint PoseFuser::find_closest_vertical_point(CorrespondLaserPoint global)
{
  CorrespondLaserPoint closest;
  CorrespondLaserPoint vertical_distance;
  double distance_min = 100.0;
  double map_point_x = 0.0;
  double map_point_y = 0.0;
  vertical_distance.x = fabs(map_point_x - global.x);
  vertical_distance.y = fabs(map_point_y - global.y);
  if (vertical_distance.x < distance_min)
  {
    distance_min = vertical_distance.x;
    closest.x = map_point_x;
    closest.y = global.y;
  }
  if (vertical_distance.y < distance_min)
  {
    distance_min = vertical_distance.y;
    closest.x = global.x;
    closest.y = map_point_y;
  }
  if (closest.x == map_point_x)
  {
    closest.nx = 1.0;
    closest.ny = 0.0;
  }
  else
  {
    closest.nx = 0.0;
    closest.ny = 1.0;
  }
  return closest;
}

Matrix3d PoseFuser::calc_laser_cov(const Vector3d &laser_estimated, vector<CorrespondLaserPoint> &current_points, vector<CorrespondLaserPoint> &reference_points)
{
  const double dd = 1e-6; // 数値微分の刻み
  vector<double> Jx;      // ヤコビ行列のxの列
  vector<double> Jy;      // ヤコビ行列のyの列
  vector<double> Jyaw;    // ヤコビ行列のyawの列

  for (size_t i = 0; i < current_points.size(); i++)
  {
    double vertical_distance = calc_vertical_distance(current_points[i], reference_points[i], laser_estimated[0], laser_estimated[1], laser_estimated[2]);
    double vertical_distance_x = calc_vertical_distance(current_points[i], reference_points[i], laser_estimated[0] + dd, laser_estimated[1], laser_estimated[2]);
    double vertical_distance_y = calc_vertical_distance(current_points[i], reference_points[i], laser_estimated[0], laser_estimated[1] + dd, laser_estimated[2]);
    double vertical_distance_yaw = calc_vertical_distance(current_points[i], reference_points[i], laser_estimated[0], laser_estimated[1], laser_estimated[2] + dd);
    Jx.push_back((vertical_distance_x - vertical_distance) / dd);
    Jy.push_back((vertical_distance_y - vertical_distance) / dd);
    Jyaw.push_back((vertical_distance_yaw - vertical_distance) / dd);
  }
  // ヘッセ行列の近似J^TJの計算
  Matrix3d hes = Matrix3d::Zero(3, 3);
  for (size_t i = 0; i < Jx.size(); i++)
  {
    hes(0, 0) += Jx[i] * Jx[i];
    hes(0, 1) += Jx[i] * Jy[i];
    hes(0, 2) += Jx[i] * Jyaw[i];
    hes(1, 1) += Jy[i] * Jy[i];
    hes(1, 2) += Jy[i] * Jyaw[i];
    hes(2, 2) += Jyaw[i] * Jyaw[i];
  }
  // J^TJが対称行列であることを利用
  hes(1, 0) = hes(0, 1);
  hes(2, 0) = hes(0, 2);
  hes(2, 1) = hes(1, 2);
  const double esp = 1e-6;
  hes += esp * Matrix3d::Identity(); // 行列の要素が0になり、逆行列が求まらない場合の対処
  return svdInverse(hes);
}

double PoseFuser::calc_vertical_distance(const CorrespondLaserPoint current, const CorrespondLaserPoint reference, double x, double y, double yaw)
{
  const double x_ = cos(yaw) * current.x - sin(yaw) * current.y + x;
  const double y_ = sin(yaw) * current.x + cos(yaw) * current.y + y;
  return (x_ - reference.x) * reference.nx + (y_ - reference.y) * reference.ny;
}

Matrix3d PoseFuser::calc_motion_cov(double vt, double wt)
{
  const double thre = 1.; // 低速時、分散を大きくしないための閾値
  if (vt < thre)
    vt = thre;
  // if (wt < thre) wt = thre;
  wt += 1; // lidarが回転に弱いため、回転時ジャイロの信頼度を上げる。
  double wt_ = wt * wt * wt * wt;
  Matrix3d C1;
  C1.setZero();
  C1(0, 0) = odom_weight_liner_ * vt / (wt_); // 並進成分x
  C1(1, 1) = odom_weight_liner_ * vt / (wt_); // 並進成分y
  C1(2, 2) = odom_weight_angler_ / (wt_);     // 回転成分

  return C1;
}

Matrix3d PoseFuser::rotate_cov(const Vector3d &laser_estimated, Matrix3d &scan_odom_motion_cov)
{
  const double cs = cos(laser_estimated[2]); // poseの回転成分thによるcos
  const double sn = sin(laser_estimated[2]);
  Matrix3d J; // 回転のヤコビ行列
  J << cs, -sn, 0,
      sn, cs, 0,
      0, 0, 1;
  Matrix3d JT = J.transpose();
  return J * scan_odom_motion_cov * JT; // 回転変換
}

Vector3d PoseFuser::fuse(Vector3d &laser_estimated, const Matrix3d &laser_cov, const Vector3d &current_scan_odom, const Matrix3d &rotate_scan_odom_motion_cov)
{
  // 共分散行列の融合
  Matrix3d IC1 = svdInverse(laser_cov);
  Matrix3d IC2 = svdInverse(rotate_scan_odom_motion_cov);
  Matrix3d IC = IC1 + IC2;
  Matrix3d fused_cov = svdInverse(IC);

  // 角度を連続に保つ
  double da = current_scan_odom[2] - laser_estimated[2];
  if (da > M_PI)
    laser_estimated[2] += 2 * M_PI;
  else if (da < -M_PI)
    laser_estimated[2] -= 2 * M_PI;

  // 平均を算出
  Vector3d nu1 = IC1 * laser_estimated;
  Vector3d nu2 = IC2 * current_scan_odom;
  Vector3d estimated = fused_cov * (nu1 + nu2);
  // estimated[2] = normalize_yaw(estimated[2]);
  return estimated;
}

// SVDを用いた逆行列計算
Matrix3d PoseFuser::svdInverse(const Matrix3d &A)
{
  size_t m = A.rows();
  size_t n = A.cols();
  JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
  MatrixXd eU = svd.matrixU();
  MatrixXd eV = svd.matrixV();
  VectorXd eS = svd.singularValues();
  MatrixXd M1(m, n);
  for (size_t i = 0; i < n; i++)
  {
    for (size_t j = 0; j < n; j++)
    {
      M1(i, j) = eU(j, i) / eS[i];
    }
  }
  Matrix3d IA;
  for (size_t i = 0; i < n; i++)
  {
    for (size_t j = 0; j < n; j++)
    {
      IA(i, j) = 0;
      for (size_t k = 0; k < n; k++)
        IA(i, j) += eV(i, k) * M1(k, j);
    }
  }
  return (IA);
}
