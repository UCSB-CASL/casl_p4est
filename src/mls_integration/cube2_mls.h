#ifndef CUBE2_MLS_H
#define CUBE2_MLS_H

#include "vector"
#include "cube2_mls_l.h"
#include "cube2_mls_q.h"

class cube2_mls_t
{
  std::vector<double> x_;
  std::vector<double> y_;


  unsigned int order_;
  unsigned int points_per_cube_;

  unsigned int cubes_in_x_;
  unsigned int cubes_in_y_;
  unsigned int cubes_total_;

  unsigned int points_in_x_;
  unsigned int points_in_y_;
  unsigned int points_total_;

public:

  std::vector<cube2_mls_l_t *> cubes_l_;
  std::vector<cube2_mls_q_t *> cubes_q_;

  std::vector<double> x_grid_;
  std::vector<double> y_grid_;

  cube2_mls_t() {}
  cube2_mls_t(double xyz_min[], double xyz_max[], int mnk[], int order);

  void initialize(const double xyz_min[], const double xyz_max[], const int mnk[], const int& order);

  ~cube2_mls_t();

  inline void get_x_coord(std::vector<double> &x) { x = x_grid_; }
  inline void get_y_coord(std::vector<double> &y) { y = y_grid_; }

  void reconstruct(const std::vector<double> &phi, const std::vector<action_t> &acn, const std::vector<int> &clr);

  void quadrature_over_domain      (                    std::vector<double> &W, std::vector<double> &X, std::vector<double> &Y);
  void quadrature_over_interface   (int num0,           std::vector<double> &W, std::vector<double> &X, std::vector<double> &Y);
  void quadrature_over_intersection(int num0, int num1, std::vector<double> &W, std::vector<double> &X, std::vector<double> &Y);
  void quadrature_in_dir           (int dir,            std::vector<double> &W, std::vector<double> &X, std::vector<double> &Y);
  void save_vtk                    (const std::string& directory, const std::string& suffix) const;
};

#endif // CUBE2_MLS_H
