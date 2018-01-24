#include "cube2_mls.h"

cube2_mls_t::cube2_mls_t(double xyz_min[], double xyz_max[], int mnk[], int order)
{
  order_ = order;
  points_per_cube_ = (order_+1)*(order_+1);

  cubes_in_x_  = mnk[0];
  cubes_in_y_  = mnk[1];
  cubes_total_ = cubes_in_x_*cubes_in_y_;

  points_in_x_  = cubes_in_x_*order_+1;
  points_in_y_  = cubes_in_y_*order_+1;
  points_total_ = points_in_x_*points_in_y_;

  // create grid
  x_.resize(points_in_x_, 0);
  y_.resize(points_in_y_, 0);

  double dx = (xyz_max[0]-xyz_min[0]) / (double) (points_in_x_ - 1);
  double dy = (xyz_max[1]-xyz_min[1]) / (double) (points_in_y_ - 1);

  for (int i = 0; i < points_in_x_; ++i) x_[i] = xyz_min[0] + (double) i * dx;
  for (int i = 0; i < points_in_y_; ++i) y_[i] = xyz_min[1] + (double) i * dy;

  x_grid_.resize(points_total_, 0);
  y_grid_.resize(points_total_, 0);

  for (int i = 0; i < points_in_x_; ++i)
    for (int j = 0; j < points_in_y_; ++j)
    {
      int idx = j * points_in_x_ + i;
      x_grid_[idx] = x_[i];
      y_grid_[idx] = y_[j];
    }
}

cube2_mls_t::~cube2_mls_t()
{
  if      (order_ == 1) { for (int idx = 0; idx < cubes_total_; ++idx) delete cubes_l_[idx]; }
  else if (order_ == 2) { for (int idx = 0; idx < cubes_total_; ++idx) delete cubes_q_[idx]; }
}


void cube2_mls_t::reconstruct(std::vector<double> &phi, std::vector<action_t> &acn, std::vector<int> &clr)
{
  int num_phi = acn.size();

  if (clr.size() != num_phi) throw;
  if (phi.size() != num_phi*points_total_) throw;

  std::vector<double> phi_cube(points_per_cube_*num_phi, -1);

  if      (order_ == 1) cubes_l_.resize(cubes_total_, NULL);
  else if (order_ == 2) cubes_q_.resize(cubes_total_, NULL);
  else throw;

  for (int i = 0; i < cubes_in_x_; ++i)
    for (int j = 0; j < cubes_in_y_; ++j)
    {
      int idx = j*cubes_in_x_ + i;

      // create new cube
      if      (order_ == 1) cubes_l_[idx] = new cube2_mls_l_t(x_[i*order_], x_[(i+1)*order_], y_[j*order_], y_[(j+1)*order_]);
      else if (order_ == 2) cubes_q_[idx] = new cube2_mls_q_t(x_[i*order_], x_[(i+1)*order_], y_[j*order_], y_[(j+1)*order_]);
      else throw;

      // get values of level-set functions for a cube
      for (int phi_idx = 0; phi_idx < num_phi; ++phi_idx)
        for (int ii = 0; ii < order_+1; ++ii)
          for (int jj = 0; jj < order_+1; ++jj)
          {
            phi_cube[phi_idx*points_per_cube_ + jj*(order_+1) + ii] = phi[phi_idx*points_total_ + (j*order_+jj)*points_in_x_ + i*order_+ii];
          }

      // feed level-set functions values to a cube for reconstruction
      if      (order_ == 1) cubes_l_[idx]->construct_domain(phi_cube, acn, clr);
      else if (order_ == 2) cubes_q_[idx]->construct_domain(phi_cube, acn, clr);
      else throw;
    }
}

void cube2_mls_t::quadrature_over_domain(std::vector<double> &W, std::vector<double> &X, std::vector<double> &Y)
{
  W.clear();
  X.clear();
  Y.clear();

  if      (order_ == 1) { for (int idx = 0; idx < cubes_total_; ++idx) cubes_l_[idx]->quadrature_over_domain(W, X, Y); }
  else if (order_ == 2) { for (int idx = 0; idx < cubes_total_; ++idx) cubes_q_[idx]->quadrature_over_domain(W, X, Y); }
  else throw;
}

void cube2_mls_t::quadrature_over_interface(int num0, std::vector<double> &W, std::vector<double> &X, std::vector<double> &Y)
{
  W.clear();
  X.clear();
  Y.clear();

  if      (order_ == 1) { for (int idx = 0; idx < cubes_total_; ++idx) cubes_l_[idx]->quadrature_over_interface(num0, W, X, Y); }
  else if (order_ == 2) { for (int idx = 0; idx < cubes_total_; ++idx) cubes_q_[idx]->quadrature_over_interface(num0, W, X, Y); }
  else throw;
}

void cube2_mls_t::quadrature_over_intersection(int num0, int num1, std::vector<double> &W, std::vector<double> &X, std::vector<double> &Y)
{
  W.clear();
  X.clear();
  Y.clear();

  if      (order_ == 1) { for (int idx = 0; idx < cubes_total_; ++idx) cubes_l_[idx]->quadrature_over_intersection(num0, num1, W, X, Y); }
  else if (order_ == 2) { for (int idx = 0; idx < cubes_total_; ++idx) cubes_q_[idx]->quadrature_over_intersection(num0, num1, W, X, Y); }
  else throw;
}


void cube2_mls_t::quadrature_in_dir(int dir, std::vector<double> &W, std::vector<double> &X, std::vector<double> &Y)
{
  W.clear();
  X.clear();
  Y.clear();

  int i_start = 0, i_total = cubes_in_x_;
  int j_start = 0, j_total = cubes_in_y_;

  switch (dir)
  {
    case 0:
      i_start = 0;
      i_total = 1;
      break;
    case 1:
      i_start = cubes_in_x_-1;
      i_total = 1;
      break;
    case 2:
      j_start = 0;
      j_total = 1;
      break;
    case 3:
      j_start = cubes_in_y_-1;
      j_total = 1;
      break;
    default:
      throw;
  }

  for (int i = i_start; i < i_start + i_total; ++i)
    for (int j = j_start; j < j_start + j_total; ++j)
    {
      int idx = j*cubes_in_x_ + i;

      if      (order_ == 1) { cubes_l_[idx]->quadrature_in_dir(dir, W, X, Y); }
      else if (order_ == 2) { cubes_q_[idx]->quadrature_in_dir(dir, W, X, Y); }
    }
}
