#include "cube3_refined_mls.h"

void cube3_refined_mls_t::construct_domain(int nx_, int ny_, int nz_, double *phi, std::vector<action_t> &action, std::vector<int> &color)
{
  nx = nx_;
  ny = ny_;
  nz = nz_;

  n_cubes = nx*ny*nz;
  n_nodes = (nx+1)*(ny+1)*(nz+1);

  v_mmm = 00 + 00*(nx+1) + 00*(nx+1)*(ny+1);
  v_pmm = nx + 00*(nx+1) + 00*(nx+1)*(ny+1);
  v_mpm = 00 + ny*(nx+1) + 00*(nx+1)*(ny+1);
  v_ppm = nx + ny*(nx+1) + 00*(nx+1)*(ny+1);
  v_mmp = 00 + 00*(nx+1) + nz*(nx+1)*(ny+1);
  v_pmp = nx + 00*(nx+1) + nz*(nx+1)*(ny+1);
  v_mpp = 00 + ny*(nx+1) + nz*(nx+1)*(ny+1);
  v_ppp = nx + ny*(nx+1) + nz*(nx+1)*(ny+1);

  double dx = (x1-x0)/(double)(nx);
  double dy = (y1-y0)/(double)(ny);
  double dz = (z1-z0)/(double)(nz);

  /* Split the cube into sub-cubes */
  cubes.clear();
  cubes.reserve(n_cubes);

  int n_phi = action.size();

  std::vector<double> phi_cube(n_phi*8, -1);

  for (int k = 0; k < nz; k++)
    for (int j = 0; j < ny; j++)
      for (int i = 0; i < nx; i++)
      {
        cubes.push_back(cube3_mls_t(0, dx, 0, dy, 0, dz));
        for (int n = 0; n < n_phi; n++)
        {
          phi_cube[n*8 + 0] = phi[n*n_nodes + i+0 + (j+0)*(nx+1) + (k+0)*(nx+1)*(ny+1)];
          phi_cube[n*8 + 1] = phi[n*n_nodes + i+1 + (j+0)*(nx+1) + (k+0)*(nx+1)*(ny+1)];
          phi_cube[n*8 + 2] = phi[n*n_nodes + i+0 + (j+1)*(nx+1) + (k+0)*(nx+1)*(ny+1)];
          phi_cube[n*8 + 3] = phi[n*n_nodes + i+1 + (j+1)*(nx+1) + (k+0)*(nx+1)*(ny+1)];
          phi_cube[n*8 + 4] = phi[n*n_nodes + i+0 + (j+0)*(nx+1) + (k+1)*(nx+1)*(ny+1)];
          phi_cube[n*8 + 5] = phi[n*n_nodes + i+1 + (j+0)*(nx+1) + (k+1)*(nx+1)*(ny+1)];
          phi_cube[n*8 + 6] = phi[n*n_nodes + i+0 + (j+1)*(nx+1) + (k+1)*(nx+1)*(ny+1)];
          phi_cube[n*8 + 7] = phi[n*n_nodes + i+1 + (j+1)*(nx+1) + (k+1)*(nx+1)*(ny+1)];
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

double cube3_refined_mls_t::integrate_over_domain(double* f)
{
  double result = 0;

  switch (loc)
  {
  case INS: result = (x1-x0)*(y1-y0)*(z1-z0)*
                     (f[v_mmm]+f[v_pmm]+f[v_mpm]+f[v_ppm]+f[v_mmp]+f[v_pmp]+f[v_mpp]+f[v_ppp])/8.0;   break;
  case OUT: result = 0.0;                                                         break;
  case FCE:
  {
    double f_cube[8];
    for (int k = 0; k < nz; k++)
      for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
        {
          f_cube[0] = f[i+0 + (j+0)*(nx+1) + (k+0)*(nx+1)*(ny+1)];
          f_cube[1] = f[i+1 + (j+0)*(nx+1) + (k+0)*(nx+1)*(ny+1)];
          f_cube[2] = f[i+0 + (j+1)*(nx+1) + (k+0)*(nx+1)*(ny+1)];
          f_cube[3] = f[i+1 + (j+1)*(nx+1) + (k+0)*(nx+1)*(ny+1)];
          f_cube[4] = f[i+0 + (j+0)*(nx+1) + (k+1)*(nx+1)*(ny+1)];
          f_cube[5] = f[i+1 + (j+0)*(nx+1) + (k+1)*(nx+1)*(ny+1)];
          f_cube[6] = f[i+0 + (j+1)*(nx+1) + (k+1)*(nx+1)*(ny+1)];
          f_cube[7] = f[i+1 + (j+1)*(nx+1) + (k+1)*(nx+1)*(ny+1)];
          result += cubes[i+j*nx+k*nx*ny].integrate_over_domain(f_cube);
        }
  } break;
  }

  return result;
}

double cube3_refined_mls_t::integrate_over_interface(double *f, int num)
{
  double result = 0;

  if (loc == FCE)
  {
    double f_cube[8];
    for (int k = 0; k < nz; k++)
      for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
        {
          f_cube[0] = f[i+0 + (j+0)*(nx+1) + (k+0)*(nx+1)*(ny+1)];
          f_cube[1] = f[i+1 + (j+0)*(nx+1) + (k+0)*(nx+1)*(ny+1)];
          f_cube[2] = f[i+0 + (j+1)*(nx+1) + (k+0)*(nx+1)*(ny+1)];
          f_cube[3] = f[i+1 + (j+1)*(nx+1) + (k+0)*(nx+1)*(ny+1)];
          f_cube[4] = f[i+0 + (j+0)*(nx+1) + (k+1)*(nx+1)*(ny+1)];
          f_cube[5] = f[i+1 + (j+0)*(nx+1) + (k+1)*(nx+1)*(ny+1)];
          f_cube[6] = f[i+0 + (j+1)*(nx+1) + (k+1)*(nx+1)*(ny+1)];
          f_cube[7] = f[i+1 + (j+1)*(nx+1) + (k+1)*(nx+1)*(ny+1)];
          result += cubes[i+j*nx+k*nx*ny].integrate_over_interface(f_cube, num);
        }
  }

  return result;
}

double cube3_refined_mls_t::integrate_over_colored_interface(double *f, int num0, int num1)
{
  double result = 0;

  if (loc == FCE)
  {
    double f_cube[8];
    for (int k = 0; k < nz; k++)
      for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
        {
          f_cube[0] = f[i+0 + (j+0)*(nx+1) + (k+0)*(nx+1)*(ny+1)];
          f_cube[1] = f[i+1 + (j+0)*(nx+1) + (k+0)*(nx+1)*(ny+1)];
          f_cube[2] = f[i+0 + (j+1)*(nx+1) + (k+0)*(nx+1)*(ny+1)];
          f_cube[3] = f[i+1 + (j+1)*(nx+1) + (k+0)*(nx+1)*(ny+1)];
          f_cube[4] = f[i+0 + (j+0)*(nx+1) + (k+1)*(nx+1)*(ny+1)];
          f_cube[5] = f[i+1 + (j+0)*(nx+1) + (k+1)*(nx+1)*(ny+1)];
          f_cube[6] = f[i+0 + (j+1)*(nx+1) + (k+1)*(nx+1)*(ny+1)];
          f_cube[7] = f[i+1 + (j+1)*(nx+1) + (k+1)*(nx+1)*(ny+1)];
          result += cubes[i+j*nx+k*nx*ny].integrate_over_colored_interface(f_cube, num0, num1);
        }
  }

  return result;
}

double cube3_refined_mls_t::measure_of_domain()
{
  double result = 0;

  switch (loc)
  {
  case INS: result = (x1-x0)*(y1-y0)*(z1-z0); break;
  case OUT: result = 0.0;                     break;
  case FCE:
    for (int k = 0; k < nz; k++)
      for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
          result += cubes[i+j*nx+k*nx*ny].measure_of_domain();
    break;
  }

  return result;
}

double cube3_refined_mls_t::measure_of_interface(int num)
{
  double result = 0;

  if (loc == FCE)
    for (int k = 0; k < nz; k++)
      for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
          result += cubes[i+j*nx+k*nx*ny].measure_of_interface(num);

  return result;
}

double cube3_refined_mls_t::measure_of_colored_interface(int num0, int num1)
{
  double result = 0;

  if (loc == FCE)
    for (int k = 0; k < nz; k++)
      for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
          result += cubes[i+j*nx+k*nx*ny].measure_of_colored_interface(num0, num1);

  return result;
}

double cube3_refined_mls_t::measure_in_dir(int dir)
{
  double result = 0;

  switch (dir)
  {
  case 0:
  {
    int i = 0;
    for (int k = 0; k < nz; k++)
      for (int j = 0; j < ny; j++)
//        for (int i = 0; i < nx; i++)
         result += cubes[i+j*nx+k*nx*ny].measure_in_dir(dir);
    break;
  }
  case 1:
  {
    int i = nx-1;
    for (int k = 0; k < nz; k++)
      for (int j = 0; j < ny; j++)
//        for (int i = 0; i < nx; i++)
         result += cubes[i+j*nx+k*nx*ny].measure_in_dir(dir);
    break;
  }
  case 2:
  {
    int j = 0;
    for (int k = 0; k < nz; k++)
//      for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
         result += cubes[i+j*nx+k*nx*ny].measure_in_dir(dir);
    break;
  }
  case 3:
  {
    int j = ny-1;
    for (int k = 0; k < nz; k++)
//      for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
         result += cubes[i+j*nx+k*nx*ny].measure_in_dir(dir);
    break;
  }
  case 4:
  {
    int k = 0;
//    for (int k = 0; k < nz; k++)
      for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
         result += cubes[i+j*nx+k*nx*ny].measure_in_dir(dir);
    break;
  }
  case 5:
  {
    int k = nz-1;
//    for (int k = 0; k < nz; k++)
      for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
         result += cubes[i+j*nx+k*nx*ny].measure_in_dir(dir);
    break;
  }
  }

  return result;
}
