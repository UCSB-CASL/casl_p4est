#include "cube2_refined_mls.h"

void cube2_refined_mls_t::construct_domain(int nx_, int ny_, double *phi, std::vector<action_t> &action, std::vector<int> &color)
{
  nx = nx_;
  ny = ny_;

  n_cubes = nx*ny;
  n_nodes = (nx+1)*(ny+1);

  v_mm = 0;
  v_pm = nx;
  v_mp = ny*(nx+1);
  v_pp = (nx+1)*(ny+1)-1;

  //std::vector<double> x_coord;
  //std::vector<double> y_coord;

  double dx = (x1-x0)/(double)(nx); //for (int i = 0; i < nx+1; i++) {x_coord.push_back(x0 + dx*(double)(i));}
  double dy = (y1-y0)/(double)(ny); //for (int j = 0; j < ny+1; j++) {y_coord.push_back(y0 + dy*(double)(j));}

  /* Split the cube into sub-cubes */
  cubes.clear();
  cubes.reserve(n_cubes);

  int n_phi = action.size();

  std::vector<double> phi_cube(n_phi*4, -1);

  for (int j = 0; j < ny; j++)
    for (int i = 0; i < nx; i++)
    {
      cubes.push_back(cube2_mls_t(0, dx, 0, dy));
      for (int k = 0; k < n_phi; k++)
      {
        phi_cube[k*4 + 0] = phi[k*n_nodes + (j+0)*(nx+1) + i+0];
        phi_cube[k*4 + 1] = phi[k*n_nodes + (j+0)*(nx+1) + i+1];
        phi_cube[k*4 + 2] = phi[k*n_nodes + (j+1)*(nx+1) + i+0];
        phi_cube[k*4 + 3] = phi[k*n_nodes + (j+1)*(nx+1) + i+1];
      }
      cubes.back().construct_domain(phi_cube.data(), action, color);
    }

  bool all_out = true;
  bool all_in = true;

  for (int i = 0; i < n_cubes; i++)
  {
    all_out = all_out && (cubes[i].loc == OUT);
    all_in  = all_in  && (cubes[i].loc == INS);
  }

  if (all_out) loc = OUT;
  else if (all_in) loc = INS;
  else loc = FCE;
}

double cube2_refined_mls_t::integrate_over_domain(double* f)
{
  double result = 0;

  switch (loc)
  {
  case INS: result = (x1-x0)*(y1-y0)*(f[v_mm]+f[v_pm]+f[v_mp]+f[v_pp])/4.0;       break;
  case OUT: result = 0.0;                                                         break;
  case FCE:
  {
    double f_cube[4];
    for (int j = 0; j < ny; j++)
      for (int i = 0; i < nx; i++)
      {
        f_cube[0] = f[(j+0)*(nx+1) + i+0];
        f_cube[1] = f[(j+0)*(nx+1) + i+1];
        f_cube[2] = f[(j+1)*(nx+1) + i+0];
        f_cube[3] = f[(j+1)*(nx+1) + i+1];
        result += cubes[j*(nx)+i].integrate_over_domain(f_cube);
      }
  } break;
  }

  return result;
}

double cube2_refined_mls_t::integrate_over_interface(double *f, int num)
{
  double result = 0;

  if (loc == FCE)
  {
    double f_cube[4];
    for (int j = 0; j < ny; j++)
      for (int i = 0; i < nx; i++)
      {
        f_cube[0] = f[(j+0)*(nx+1) + i+0];
        f_cube[1] = f[(j+0)*(nx+1) + i+1];
        f_cube[2] = f[(j+1)*(nx+1) + i+0];
        f_cube[3] = f[(j+1)*(nx+1) + i+1];
        result += cubes[j*(nx)+i].integrate_over_interface(f_cube, num);
      }
  }

  return result;
}

double cube2_refined_mls_t::integrate_over_colored_interface(double *f, int num0, int num1)
{
  double result = 0;

  if (loc == FCE)
  {
    double f_cube[4];
    for (int j = 0; j < ny; j++)
      for (int i = 0; i < nx; i++)
      {
        f_cube[0] = f[(j+0)*(nx+1) + i+0];
        f_cube[1] = f[(j+0)*(nx+1) + i+1];
        f_cube[2] = f[(j+1)*(nx+1) + i+0];
        f_cube[3] = f[(j+1)*(nx+1) + i+1];
        result += cubes[j*(nx)+i].integrate_over_colored_interface(f_cube, num0, num1);
      }
  }

  return result;
}

double cube2_refined_mls_t::integrate_over_intersection(double *f, int num0, int num1)
{
  double result = 0;
  if (loc == FCE && num_non_trivial > 1)
  {
    double f_cube[4];
    for (int j = 0; j < ny; j++)
      for (int i = 0; i < nx; i++)
      {
        f_cube[0] = f[(j+0)*(nx+1) + i+0];
        f_cube[1] = f[(j+0)*(nx+1) + i+1];
        f_cube[2] = f[(j+1)*(nx+1) + i+0];
        f_cube[3] = f[(j+1)*(nx+1) + i+1];
        result += cubes[j*(nx)+i].integrate_over_intersection(f_cube, num0, num1);
      }
  }
  return result;
}

double cube2_refined_mls_t::integrate_in_dir(double *f, int dir)
{
  double result = 0;
  double f_cube[4];
  switch (dir)
  {
  case 0:
  {
    int i = 0;
    for (int j = 0; j < ny; j++)
    {
      f_cube[0] = f[(j+0)*(nx+1) + i+0];
      f_cube[1] = f[(j+0)*(nx+1) + i+1];
      f_cube[2] = f[(j+1)*(nx+1) + i+0];
      f_cube[3] = f[(j+1)*(nx+1) + i+1];
      result += cubes[j*(nx)+i].integrate_in_dir(f_cube, dir);
    }
    break;
  }
  case 1:
  {
    int i = nx-1;
    for (int j = 0; j < ny; j++)
    {
      f_cube[0] = f[(j+0)*(nx+1) + i+0];
      f_cube[1] = f[(j+0)*(nx+1) + i+1];
      f_cube[2] = f[(j+1)*(nx+1) + i+0];
      f_cube[3] = f[(j+1)*(nx+1) + i+1];
      result += cubes[j*(nx)+i].integrate_in_dir(f_cube, dir);
    }
    break;
  }
  case 2:
  {
    int j = 0;
    for (int i = 0; i < nx; i++)
    {
      f_cube[0] = f[(j+0)*(nx+1) + i+0];
      f_cube[1] = f[(j+0)*(nx+1) + i+1];
      f_cube[2] = f[(j+1)*(nx+1) + i+0];
      f_cube[3] = f[(j+1)*(nx+1) + i+1];
      result += cubes[j*(nx)+i].integrate_in_dir(f_cube, dir);
    }
    break;
  }
  case 3:
  {
    int j = ny-1;
    for (int i = 0; i < nx; i++)
    {
      f_cube[0] = f[(j+0)*(nx+1) + i+0];
      f_cube[1] = f[(j+0)*(nx+1) + i+1];
      f_cube[2] = f[(j+1)*(nx+1) + i+0];
      f_cube[3] = f[(j+1)*(nx+1) + i+1];
      result += cubes[j*(nx)+i].integrate_in_dir(f_cube, dir);
    }
    break;
  }
  }
  return result;
}

double cube2_refined_mls_t::measure_of_domain()
{
  double result = 0;

  switch (loc)
  {
  case INS: result = (x1-x0)*(y1-y0);     break;
  case OUT: result = 0.0;                 break;
  case FCE:
    for (int j = 0; j < ny; j++)
      for (int i = 0; i < nx; i++)
        result += cubes[j*(nx)+i].measure_of_domain();
    break;
  }

  return result;
}

double cube2_refined_mls_t::measure_of_interface(int num)
{
  double result = 0;

  if (loc == FCE)
    for (int j = 0; j < ny; j++)
      for (int i = 0; i < nx; i++)
        result += cubes[j*(nx)+i].measure_of_interface(num);

  return result;
}

double cube2_refined_mls_t::measure_of_colored_interface(int num0, int num1)
{
  double result = 0;

  if (loc == FCE)
    for (int j = 0; j < ny; j++)
      for (int i = 0; i < nx; i++)
        result += cubes[j*(nx)+i].measure_of_colored_interface(num0, num1);

  return result;
}

double cube2_refined_mls_t::measure_in_dir(int dir)
{
  double result = 0;

  switch (dir)
  {
  case 0:
  {
    int i = 0;
    for (int j = 0; j < ny; j++)
      result += cubes[j*(nx)+i].measure_in_dir(dir);
    break;
  }
  case 1:
  {
    int i = nx-1;
    for (int j = 0; j < ny; j++)
      result += cubes[j*(nx)+i].measure_in_dir(dir);
     break;
  }
  case 2:
  {
    int j = 0;
    for (int i = 0; i < nx; i++)
      result += cubes[j*(nx)+i].measure_in_dir(dir);
    break;
  }
  case 3:
  {
    int j = ny-1;
    for (int i = 0; i < nx; i++)
      result += cubes[j*(nx)+i].measure_in_dir(dir);
    break;
  }
  }

  return result;
}
