#include "cube3_mls.h"
#include "vtk/simplex3_mls_l_vtk.h"
#include "vtk/simplex3_mls_q_vtk.h"

void cube3_mls_t::initialize(const double xyz_min[], const double xyz_max[], const int mnk[], const int& order)
{
  check_for_curvature_ = true;

  for (unsigned int idx = 0; idx < cubes_l_.size(); ++idx) delete cubes_l_[idx];
  for (unsigned int idx = 0; idx < cubes_q_.size(); ++idx) delete cubes_q_[idx];

  order_ = order;
  points_per_cube_ = (order_+1)*(order_+1)*(order_+1);

  cubes_in_x_  = mnk[0];
  cubes_in_y_  = mnk[1];
  cubes_in_z_  = mnk[2];
  cubes_total_ = cubes_in_x_*cubes_in_y_*cubes_in_z_;

  points_in_x_  = cubes_in_x_*order_+1;
  points_in_y_  = cubes_in_y_*order_+1;
  points_in_z_  = cubes_in_z_*order_+1;
  points_total_ = points_in_x_*points_in_y_*points_in_z_;

  // create grid
  x_.resize(points_in_x_, 0);
  y_.resize(points_in_y_, 0);
  z_.resize(points_in_z_, 0);

  double dx = (xyz_max[0]-xyz_min[0]) / (double) (points_in_x_ - 1);
  double dy = (xyz_max[1]-xyz_min[1]) / (double) (points_in_y_ - 1);
  double dz = (xyz_max[2]-xyz_min[2]) / (double) (points_in_z_ - 1);

  for (unsigned int i = 0; i < points_in_x_; ++i) x_[i] = xyz_min[0] + (double) i * dx;
  for (unsigned int i = 0; i < points_in_y_; ++i) y_[i] = xyz_min[1] + (double) i * dy;
  for (unsigned int i = 0; i < points_in_z_; ++i) z_[i] = xyz_min[2] + (double) i * dz;

  x_grid_.resize(points_total_, 0);
  y_grid_.resize(points_total_, 0);
  z_grid_.resize(points_total_, 0);

  for (unsigned int i = 0; i < points_in_x_; ++i)
    for (unsigned int j = 0; j < points_in_y_; ++j)
      for (unsigned int k = 0; k < points_in_z_; ++k)
      {
        unsigned int idx = k*points_in_x_*points_in_y_ + j * points_in_x_ + i;
        x_grid_[idx] = x_[i];
        y_grid_[idx] = y_[j];
        z_grid_[idx] = z_[k];
      }
}

cube3_mls_t::~cube3_mls_t()
{
  for (unsigned int idx = 0; idx < cubes_l_.size(); ++idx) delete cubes_l_[idx];
  for (unsigned int idx = 0; idx < cubes_q_.size(); ++idx) delete cubes_q_[idx];
}


void cube3_mls_t::reconstruct(const std::vector<double> &phi, const std::vector<action_t> &acn, const std::vector<int> &clr)
{
  unsigned int num_phi = acn.size();

  if (clr.size() != num_phi) throw;
  if (phi.size() != num_phi*points_total_) throw;

  std::vector<double> phi_cube(points_per_cube_*num_phi, -1);

  if      (order_ == 1) cubes_l_.resize(cubes_total_, NULL);
  else if (order_ == 2) cubes_q_.resize(cubes_total_, NULL);
  else throw;

  for (unsigned int i = 0; i < cubes_in_x_; ++i)
    for (unsigned int j = 0; j < cubes_in_y_; ++j)
      for (unsigned int k = 0; k < cubes_in_z_; ++k)
      {
        int idx = k*cubes_in_x_*cubes_in_y_ + j*cubes_in_x_ + i;

        // create new cube
        if      (order_ == 1) cubes_l_[idx] = new cube3_mls_l_t(x_[i*order_], x_[(i+1)*order_], y_[j*order_], y_[(j+1)*order_], z_[k*order_], z_[(k+1)*order_]);
        else if (order_ == 2) cubes_q_[idx] = new cube3_mls_q_t(x_[i*order_], x_[(i+1)*order_], y_[j*order_], y_[(j+1)*order_], z_[k*order_], z_[(k+1)*order_]);
        else throw;

        // get values of level-set functions for a cube
        for (unsigned int phi_idx = 0; phi_idx < num_phi; ++phi_idx)
          for (unsigned int ii = 0; ii < order_+1; ++ii)
            for (unsigned int jj = 0; jj < order_+1; ++jj)
              for (unsigned int kk = 0; kk < order_+1; ++kk)
              {
                phi_cube[phi_idx*points_per_cube_ + kk*(order_+1)*(order_+1) + jj*(order_+1) + ii]
                    = phi[phi_idx*points_total_ + (k*order_+kk)*points_in_x_*points_in_y_ + (j*order_+jj)*points_in_x_ + i*order_+ii];
              }

        // feed level-set functions values to a cube for reconstruction
        if      (order_ == 1) { cubes_l_[idx]->construct_domain(phi_cube, acn, clr); }
        else if (order_ == 2) { cubes_q_[idx]->set_check_for_curvature(check_for_curvature_);
                                cubes_q_[idx]->construct_domain(phi_cube, acn, clr); }
        else throw;
      }

}

void cube3_mls_t::quadrature_over_domain(std::vector<double> &W, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z)
{
  W.clear();
  X.clear();
  Y.clear();
  Z.clear();

  if      (order_ == 1) { for (unsigned int idx = 0; idx < cubes_total_; ++idx) cubes_l_[idx]->quadrature_over_domain(W, X, Y, Z); }
  else if (order_ == 2) { for (unsigned int idx = 0; idx < cubes_total_; ++idx) cubes_q_[idx]->quadrature_over_domain(W, X, Y, Z); }
  else throw;
}

void cube3_mls_t::quadrature_over_interface(int num0, std::vector<double> &W, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z)
{
  W.clear();
  X.clear();
  Y.clear();
  Z.clear();

  if      (order_ == 1) { for (unsigned int idx = 0; idx < cubes_total_; ++idx) cubes_l_[idx]->quadrature_over_interface(num0, W, X, Y, Z); }
  else if (order_ == 2) { for (unsigned int idx = 0; idx < cubes_total_; ++idx) cubes_q_[idx]->quadrature_over_interface(num0, W, X, Y, Z); }
  else throw;
}

void cube3_mls_t::quadrature_over_intersection(int num0, int num1, std::vector<double> &W, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z)
{
  W.clear();
  X.clear();
  Y.clear();
  Z.clear();

  if      (order_ == 1) { for (unsigned int idx = 0; idx < cubes_total_; ++idx) cubes_l_[idx]->quadrature_over_intersection(num0, num1, W, X, Y, Z); }
  else if (order_ == 2) { for (unsigned int idx = 0; idx < cubes_total_; ++idx) cubes_q_[idx]->quadrature_over_intersection(num0, num1, W, X, Y, Z); }
  else throw;
}

void cube3_mls_t::quadrature_over_intersection(int num0, int num1, int num2, std::vector<double> &W, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z)
{
  W.clear();
  X.clear();
  Y.clear();
  Z.clear();

  if      (order_ == 1) { for (unsigned int idx = 0; idx < cubes_total_; ++idx) cubes_l_[idx]->quadrature_over_intersection(num0, num1, num2, W, X, Y, Z); }
  else if (order_ == 2) { for (unsigned int idx = 0; idx < cubes_total_; ++idx) cubes_q_[idx]->quadrature_over_intersection(num0, num1, num2, W, X, Y, Z); }
  else throw;
}


void cube3_mls_t::quadrature_in_dir(int dir, std::vector<double> &W, std::vector<double> &X, std::vector<double> &Y, std::vector<double> &Z)
{
  W.clear();
  X.clear();
  Y.clear();
  Z.clear();

  unsigned int i_start = 0, i_total = cubes_in_x_;
  unsigned int j_start = 0, j_total = cubes_in_y_;
  unsigned int k_start = 0, k_total = cubes_in_z_;

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
    case 4:
      k_start = 0;
      k_total = 1;
      break;
    case 5:
      k_start = cubes_in_z_-1;
      k_total = 1;
      break;
    default:
      throw;
  }

  for (unsigned int i = i_start; i < i_start + i_total; ++i)
    for (unsigned int j = j_start; j < j_start + j_total; ++j)
      for (unsigned int k = k_start; k < k_start + k_total; ++k)
      {
        int idx = k*cubes_in_x_*cubes_in_y_ + j*cubes_in_x_ + i;

        if      (order_ == 1) { cubes_l_[idx]->quadrature_in_dir(dir, W, X, Y, Z); }
        else if (order_ == 2) { cubes_q_[idx]->quadrature_in_dir(dir, W, X, Y, Z); }
      }

}

void cube3_mls_t::save_vtk(const std::string& directory, const std::string& suffix) const
{
  if(order_ == 1)
  {
    std::vector<simplex3_mls_l_t*>  tmp;
    for(size_t k = 0; k < cubes_l_.size(); k++)
    {
      for(size_t uu = 0; uu < cubes_l_[k]->simplex.size(); uu++)
        tmp.push_back(&cubes_l_[k]->simplex[uu]);
    }
    simplex3_mls_l_vtk::write_simplex_geometry(tmp, directory, suffix);
  }
  else if(order_ == 2)
  {
    std::vector<simplex3_mls_q_t*>  tmp;
    for(size_t k = 0; k < cubes_q_.size(); k++)
    {
      for(size_t uu = 0; uu < cubes_q_[k]->simplex.size(); uu++)
        tmp.push_back(&cubes_q_[k]->simplex[uu]);
    }
    simplex3_mls_q_vtk::write_simplex_geometry(tmp, directory, suffix);
  }
  else
    throw;
}
