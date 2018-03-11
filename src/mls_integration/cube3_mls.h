#ifndef CUBE3_MLS_H
#define CUBE3_MLS_H

#include "vector"
#include "cube3_mls_l.h"
#include "cube3_mls_q.h"

class cube3_mls_t
{
  std::vector<double> x_;
  std::vector<double> y_;
  std::vector<double> z_;

  int order_;
  int points_per_cube_;

  int cubes_in_x_;
  int cubes_in_y_;
  int cubes_in_z_;
  int cubes_total_;

  int points_in_x_;
  int points_in_y_;
  int points_in_z_;
  int points_total_;

  bool check_for_curvature_;

public:

  std::vector<cube3_mls_l_t *> cubes_l_;
  std::vector<cube3_mls_q_t *> cubes_q_;

  std::vector<double> x_grid_;
  std::vector<double> y_grid_;
  std::vector<double> z_grid_;

  cube3_mls_t() {}
  cube3_mls_t(double xyz_min[], double xyz_max[], int mnk[], int order)
  { initialize(xyz_min, xyz_max, mnk, order); }

  void initialize (double xyz_min[], double xyz_max[], int mnk[], int order);

  ~cube3_mls_t();

  inline void get_x_coord(std::vector<double> &x) { x = x_grid_; }
  inline void get_y_coord(std::vector<double> &y) { y = y_grid_; }
  inline void get_z_coord(std::vector<double> &z) { z = z_grid_; }

  inline void set_check_for_curvature(bool value) { check_for_curvature_ = value; }

  void reconstruct(std::vector<double> &phi, std::vector<action_t> &acn, std::vector<int> &clr);

  void quadrature_over_domain      (                              std::vector<double> &W, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z);
  void quadrature_over_interface   (int num0,                     std::vector<double> &W, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z);
  void quadrature_over_intersection(int num0, int num1,           std::vector<double> &W, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z);
  void quadrature_over_intersection(int num0, int num1, int num2, std::vector<double> &W, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z);
  void quadrature_in_dir           (int dir,                      std::vector<double> &W, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z);
};

#endif // CUBE3_MLS_H
