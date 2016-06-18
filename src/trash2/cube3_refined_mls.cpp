#include "cube3_refined_mls.h"

void cube3_refined_mls_t::construct_domain(int nx_, int ny_, int nz_, int level)
{
  int n_phis = action->size();

  nx = nx_; ny = ny_; nz = nz_;

  double dx = (x1-x0)/(double)(nx);
  double dy = (y1-y0)/(double)(ny);
  double dz = (z1-z0)/(double)(nz);

  dx_min = dx/pow(2.,(double)(level+1));
  dy_min = dy/pow(2.,(double)(level+1));
  dz_min = dz/pow(2.,(double)(level+1));

  x_coord.resize(nx+1, 0.); for (int i = 0; i < nx+1; i++) {x_coord[i]=(x0 + dx*(double)(i));}
  y_coord.resize(ny+1, 0.); for (int j = 0; j < ny+1; j++) {y_coord[j]=(y0 + dy*(double)(j));}
  z_coord.resize(nz+1, 0.); for (int k = 0; k < nz+1; k++) {z_coord[k]=(z0 + dz*(double)(k));}

  n_nodes = (nx+1)*(ny+1)*(nx+1);
  n_cubes = 0;
  n_leafs = nx*ny*nz;

  phi.resize    (n_phis, std::vector<double> (n_nodes, -1.));

  phi_xx.resize (n_phis, std::vector<double> (n_nodes, 0.));
  phi_yy.resize (n_phis, std::vector<double> (n_nodes, 0.));
  phi_zz.resize (n_phis, std::vector<double> (n_nodes, 0.));

  if (phi_cf_in == NULL)
  {
    for (int q = 0; q < n_phis; q++)
    for (int kk = 0; kk < nz+1; kk++)
      for (int jj = 0; jj < ny+1; jj++)
        for (int ii = 0; ii < nx+1; ii++)
        {
          int idx = ii + jj*(nx+1) + kk*(nx+1)*(ny+1);
          phi[q][idx] = interp.quadratic(phi_in->at(q).data(),
                                         phi_xx_in->at(q).data(),
                                         phi_yy_in->at(q).data(),
                                         phi_zz_in->at(q).data(),
                                         x_coord[ii], y_coord[jj], z_coord[kk]);

          phi_xx[q][idx] = interp.linear(phi_xx_in->at(q).data(), x_coord[ii], y_coord[jj], z_coord[kk]);
          phi_yy[q][idx] = interp.linear(phi_yy_in->at(q).data(), x_coord[ii], y_coord[jj], z_coord[kk]);
          phi_zz[q][idx] = interp.linear(phi_zz_in->at(q).data(), x_coord[ii], y_coord[jj], z_coord[kk]);
        }

  } else {
    for (int q = 0; q < n_phis; q++)
    for (int kk = 0; kk < nz+1; kk++)
      for (int jj = 0; jj < ny+1; jj++)
        for (int ii = 0; ii < nx+1; ii++)
        {
          int idx = ii + jj*(nx+1) + kk*(nx+1)*(ny+1);
          double phi_000 = (*phi_cf_in->at(q))(x_coord[ii], y_coord[jj], z_coord[kk]);
          double phi_m00 = (*phi_cf_in->at(q))(x_coord[ii]-dx_min, y_coord[jj], z_coord[kk]);
          double phi_p00 = (*phi_cf_in->at(q))(x_coord[ii]+dx_min, y_coord[jj], z_coord[kk]);
          double phi_0m0 = (*phi_cf_in->at(q))(x_coord[ii], y_coord[jj]-dy_min, z_coord[kk]);
          double phi_0p0 = (*phi_cf_in->at(q))(x_coord[ii], y_coord[jj]+dy_min, z_coord[kk]);
          double phi_00m = (*phi_cf_in->at(q))(x_coord[ii], y_coord[jj], z_coord[kk]-dz_min);
          double phi_00p = (*phi_cf_in->at(q))(x_coord[ii], y_coord[jj], z_coord[kk]+dz_min);

          phi[q][idx] = phi_000;

          phi_xx[q][idx] = (phi_p00+phi_m00-2.0*phi_000)/dx_min/dx_min;
          phi_yy[q][idx] = (phi_0p0+phi_0m0-2.0*phi_000)/dy_min/dy_min;
          phi_zz[q][idx] = (phi_00p+phi_00m-2.0*phi_000)/dz_min/dz_min;
        }
  }

  leaf_to_node.resize(n_leafs*N_CHILDREN,-1);
  get_cube.resize(n_leafs,-1);


  for (int kk = 0; kk < nz; kk++)
    for (int jj = 0; jj < ny; jj++)
      for (int ii = 0; ii < nx; ii++)
      {
        int idx = ii + jj*(nx) + kk*(nx)*(ny);
        leaf_to_node[idx*N_CHILDREN+0] = ( ii ) + ( jj )*(nx+1) + ( kk )*(nx+1)*(ny+1);
        leaf_to_node[idx*N_CHILDREN+1] = (ii+1) + ( jj )*(nx+1) + ( kk )*(nx+1)*(ny+1);
        leaf_to_node[idx*N_CHILDREN+2] = ( ii ) + (jj+1)*(nx+1) + ( kk )*(nx+1)*(ny+1);
        leaf_to_node[idx*N_CHILDREN+3] = (ii+1) + (jj+1)*(nx+1) + ( kk )*(nx+1)*(ny+1);
        leaf_to_node[idx*N_CHILDREN+4] = ( ii ) + ( jj )*(nx+1) + (kk+1)*(nx+1)*(ny+1);
        leaf_to_node[idx*N_CHILDREN+5] = (ii+1) + ( jj )*(nx+1) + (kk+1)*(nx+1)*(ny+1);
        leaf_to_node[idx*N_CHILDREN+6] = ( ii ) + (jj+1)*(nx+1) + (kk+1)*(nx+1)*(ny+1);
        leaf_to_node[idx*N_CHILDREN+7] = (ii+1) + (jj+1)*(nx+1) + (kk+1)*(nx+1)*(ny+1);
      }


  /* Split the cube into sub-cubes */
  cubes_mls.clear();
  cubes_mls.reserve(n_leafs);

  std::vector< std::vector<double> > phi_cube    (n_phis, std::vector<double> (N_CHILDREN, -1.));
  std::vector< std::vector<double> > phi_xx_cube (n_phis, std::vector<double> (N_CHILDREN, 0.));
  std::vector< std::vector<double> > phi_yy_cube (n_phis, std::vector<double> (N_CHILDREN, 0.));
  std::vector< std::vector<double> > phi_zz_cube (n_phis, std::vector<double> (N_CHILDREN, 0.));

  // reconstruct interfaces in leaf cubes
  for (int kk = 0; kk < nz; kk++)
    for (int jj = 0; jj < ny; jj++)
      for (int ii = 0; ii < nx; ii++)
      {
        int n = ii + jj*(nx) + kk*(nx)*(ny);
        // create a mls cube
        cubes_mls.push_back(cube3_mls_t(x_coord[ii], x_coord[ii+1], y_coord[jj], y_coord[jj+1], z_coord[kk], z_coord[kk+1]));

        // fetch function values at the corners of a cube
        for (int q = 0; q < n_phis; q++)
          for (int d = 0; d < N_CHILDREN; d++)
          {
            phi_cube   [q][d] = phi   [q][leaf_to_node[n*N_CHILDREN+d]];

            phi_xx_cube[q][d] = phi_xx[q][leaf_to_node[n*N_CHILDREN+d]];
            phi_yy_cube[q][d] = phi_yy[q][leaf_to_node[n*N_CHILDREN+d]];
            phi_zz_cube[q][d] = phi_zz[q][leaf_to_node[n*N_CHILDREN+d]];
          }

        cubes_mls.back().set_phi(phi_cube, phi_xx_cube, phi_yy_cube, phi_zz_cube, *action, *color);
        cubes_mls.back().construct_domain();
      }
}

void cube3_refined_mls_t::sample_all(std::vector<double> &f, std::vector<double> &f_values)
{
  for (int kk = 0; kk < nz+1; kk++)
    for (int jj = 0; jj < ny+1; jj++)
      for (int ii = 0; ii < nx+1; ii++)
        f_values[ii + jj*(nx+1) + kk*(nx+1)*(ny+1)] = interp.linear(f, x_coord[ii], y_coord[jj], z_coord[kk]);
}

void cube3_refined_mls_t::sample_all(CF_3 &f, std::vector<double> &f_values)
{
  for (int kk = 0; kk < nz+1; kk++)
    for (int jj = 0; jj < ny+1; jj++)
      for (int ii = 0; ii < nx+1; ii++)
        f_values[ii + jj*(nx+1) + kk*(nx+1)*(ny+1)] = f(x_coord[ii], y_coord[jj], z_coord[kk]);
}

double cube3_refined_mls_t::perform(std::vector<double> &f, int type, int num0, int num1, int num2)
{
  std::vector<double> f_values(n_nodes,1);
  sample_all(f, f_values);

  double result = 0;
  double f_cube[N_CHILDREN];

  for (int n = 0; n < n_leafs; n++)
  {
    // fetch function values at corners of a cube
    for (int d = 0; d < N_CHILDREN; d++)
      f_cube[d] = f_values[leaf_to_node[n*N_CHILDREN + d]];
    switch (type) {
    case 0: result += cubes_mls[n].integrate_over_domain            (f_cube); break;
    case 1: result += cubes_mls[n].integrate_over_interface         (f_cube, num0); break;
    case 2: result += cubes_mls[n].integrate_over_intersection      (f_cube, num0, num1); break;
    case 3: result += cubes_mls[n].integrate_over_intersection      (f_cube, num0, num1, num2); break;
    case 4: result += cubes_mls[n].integrate_over_colored_interface (f_cube, num0, num1); break;
//    case 5: result += cubes_mls[n].integrate_in_non_cart_dir        (f_cube, num0); break;
    }
  }

  return result;
}

double cube3_refined_mls_t::perform(CF_3 &f, int type, int num0, int num1, int num2)
{
  std::vector<double> f_values(n_nodes,1);
  sample_all(f, f_values);

  double result = 0;
  double f_cube[N_CHILDREN];

  for (int n = 0; n < n_leafs; n++)
  {
    // fetch function values at corners of a cube
    for (int d = 0; d < N_CHILDREN; d++)
      f_cube[d] = f_values[leaf_to_node[n*N_CHILDREN + d]];
    switch (type) {
    case 0: result += cubes_mls[n].integrate_over_domain            (f_cube); break;
    case 1: result += cubes_mls[n].integrate_over_interface         (f_cube, num0); break;
    case 2: result += cubes_mls[n].integrate_over_intersection      (f_cube, num0, num1); break;
    case 3: result += cubes_mls[n].integrate_over_intersection      (f_cube, num0, num1, num2); break;
    case 4: result += cubes_mls[n].integrate_over_colored_interface (f_cube, num0, num1); break;
//    case 5: result += cubes_mls[n].integrate_in_non_cart_dir        (f_cube, num0); break;
    }
  }

  return result;
}

double cube3_refined_mls_t::perform(double f, int type, int num0, int num1, int num2)
{
  double result = 0;
  double f_cube[N_CHILDREN] = {f,f,f,f,f,f,f,f};

  for (int n = 0; n < n_leafs; n++)
  {
    switch (type) {
    case 0: result += cubes_mls[n].integrate_over_domain            (f_cube); break;
    case 1: result += cubes_mls[n].integrate_over_interface         (f_cube, num0); break;
    case 2: result += cubes_mls[n].integrate_over_intersection      (f_cube, num0, num1); break;
    case 3: result += cubes_mls[n].integrate_over_intersection      (f_cube, num0, num1, num2); break;
    case 4: result += cubes_mls[n].integrate_over_colored_interface (f_cube, num0, num1); break;
//    case 5: result += cubes_mls[n].integrate_in_non_cart_dir        (f_cube, num0); break;
    }
  }

  return result;
}

double cube3_refined_mls_t::integrate_in_dir(std::vector<double> &f, int dir)
{
  std::vector<double> f_values(n_nodes,1);
  sample_all(f, f_values);

  double result = 0;
  double f_cube[N_CHILDREN];

//  for (int n = 0; n < n_leafs; n++)
//    if (cubes[get_cube[n]].wall[dir])
//    {
//      // fetch function values at corners of a cube
//      for (int d = 0; d < N_CHILDREN; d++)
//        f_cube[d] = f_values[leaf_to_node[n*N_CHILDREN + d]];
//      result += cubes_mls[n].integrate_in_dir(f_cube, dir);
//    }

  return result;
}

double cube3_refined_mls_t::integrate_in_dir(CF_3 &f, int dir)
{
  std::vector<double> f_values(n_nodes,1);
  sample_all(f, f_values);

  double result = 0;
  double f_cube[N_CHILDREN];

//  for (int n = 0; n < n_leafs; n++)
//    if (cubes[get_cube[n]].wall[dir])
//    {
//      // fetch function values at corners of a cube
//      for (int d = 0; d < N_CHILDREN; d++)
//        f_cube[d] = f_values[leaf_to_node[n*N_CHILDREN + d]];
//      result += cubes_mls[n].integrate_in_dir(f_cube, dir);
//    }

  return result;
}

double cube3_refined_mls_t::integrate_in_dir(double f, int dir)
{
  double result = 0;
  double f_cube[N_CHILDREN] = {f,f,f,f,f,f,f,f};

//  for (int n = 0; n < n_leafs; n++)
//    if (cubes[get_cube[n]].wall[dir])
//    {
//      result += cubes_mls[n].integrate_in_dir(f_cube, dir);
//    }

  return result;
}
