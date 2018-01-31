#ifndef CUBE2_MLS_H
#define CUBE2_MLS_H

#include "vector"
#include "cube2_mls_l.h"
#include "cube2_mls_q.h"

class cube2_mls_t
{
  std::vector<double> x_;
  std::vector<double> y_;


  int order_;
  int points_per_cube_;

  int cubes_in_x_;
  int cubes_in_y_;
  int cubes_total_;

  int points_in_x_;
  int points_in_y_;
  int points_total_;

public:

  std::vector<cube2_mls_l_t *> cubes_l_;
  std::vector<cube2_mls_q_t *> cubes_q_;

  std::vector<double> x_grid_;
  std::vector<double> y_grid_;

  cube2_mls_t(double xyz_min[], double xyz_max[], int mnk[], int order);

  ~cube2_mls_t();

  inline void get_x_coord(std::vector<double> &x) { x = x_grid_; }
  inline void get_y_coord(std::vector<double> &y) { y = y_grid_; }

  void reconstruct(std::vector<double> &phi, std::vector<action_t> &acn, std::vector<int> &clr);

  void quadrature_over_domain      (                    std::vector<double> &W, std::vector<double> &X, std::vector<double> &Y);
  void quadrature_over_interface   (int num0,           std::vector<double> &W, std::vector<double> &X, std::vector<double> &Y);
  void quadrature_over_intersection(int num0, int num1, std::vector<double> &W, std::vector<double> &X, std::vector<double> &Y);
  void quadrature_in_dir           (int dir,            std::vector<double> &W, std::vector<double> &X, std::vector<double> &Y);
};

#endif // CUBE2_MLS_H
