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

  std::vector<double> x_coord(nx+1, 0.); for (int i = 0; i < nx+1; i++) {x_coord[i]=(x0 + dx*(double)(i));}
  std::vector<double> y_coord(ny+1, 0.); for (int j = 0; j < ny+1; j++) {y_coord[j]=(y0 + dy*(double)(j));}
  std::vector<double> z_coord(nz+1, 0.); for (int k = 0; k < nz+1; k++) {z_coord[k]=(z0 + dz*(double)(k));}

  // do initial splitting
  nodes.reserve((nx+1)*(ny+1)*(nz+1));
  edges.reserve(nx*(ny+1)*(nz+1) + (nx+1)*ny*(nz+1) + (nx+1)*(ny+1)*(nz+1));
  faces.reserve((nx+1)*ny*nz + nx*(ny+1)*nz + nx*ny*(nz+1));
  cubes.reserve(nx*ny*nz);

  // create nodes
  for (int k = 0; k < nz+1; k++)
    for (int j = 0; j < ny+1; j++)
      for (int i = 0; i < nx+1; i++)
        nodes.push_back(node_t(x_coord[i],y_coord[j], z_coord[k]));

  // create x-edges
  for (int k = 0; k < nz+1; k++)
    for (int j = 0; j < ny+1; j++)
      for (int i = 0; i < nx; i++)
        edges.push_back(edge_t(k*(nx+1)*(ny+1) + j*(nx+1) + i, k*(nx+1)*(ny+1) + j*(nx+1) + i+1));

  // create y-edges
  for (int k = 0; k < nz+1; k++)
    for (int j = 0; j < ny; j++)
      for (int i = 0; i < nx+1; i++)
        edges.push_back(edge_t(k*(nx+1)*(ny+1) + j*(nx+1) + i, k*(nx+1)*(ny+1) + (j+1)*(nx+1) + i));

  // create z-edges
  for (int k = 0; k < nz; k++)
    for (int j = 0; j < ny+1; j++)
      for (int i = 0; i < nx+1; i++)
        edges.push_back(edge_t(k*(nx+1)*(ny+1) + j*(nx+1) + i, (k+1)*(nx+1)*(ny+1) + j*(nx+1) + i));

  int xe_offset = 0;
  int ye_offset = nx*(ny+1)*(nz+1);
  int ze_offset = nx*(ny+1)*(nz+1) + (nx+1)*ny*(nz+1);

  // create x-faces
  for (int k = 0; k < nz; k++)
    for (int j = 0; j < ny; j++)
      for (int i = 0; i < nx+1; i++)
        faces.push_back(face_t(ze_offset + k*(nx+1)*(ny+1) + j*(nx+1) + i, ze_offset + ( k )*(nx+1)*(ny+1) + (j+1)*(nx+1) + ( i ),
                               ye_offset + k*(nx+1)*( ny ) + j*(nx+1) + i, ye_offset + (k+1)*(nx+1)*( ny ) + ( j )*(nx+1) + ( i )));

  // create y-faces
  for (int k = 0; k < nz; k++)
    for (int j = 0; j < ny+1; j++)
      for (int i = 0; i < nx; i++)
        faces.push_back(face_t(ze_offset + k*(nx+1)*(ny+1) + j*(nx+1) + i, ze_offset + ( k )*(nx+1)*(ny+1) + ( j )*(nx+1) + (i+1),
                               xe_offset + k*( nx )*(ny+1) + j*( nx ) + i, xe_offset + (k+1)*( nx )*(ny+1) + ( j )*( nx ) + ( i )));

  // create z-faces
  for (int k = 0; k < nz+1; k++)
    for (int j = 0; j < ny; j++)
      for (int i = 0; i < nx; i++)
        faces.push_back(face_t(ye_offset + k*(nx+1)*( ny ) + j*(nx+1) + i, ye_offset + ( k )*(nx+1)*( ny ) + ( j )*(nx+1) + (i+1),
                               xe_offset + k*( nx )*(ny+1) + j*( nx ) + i, xe_offset + ( k )*( nx )*(ny+1) + (j+1)*( nx ) + ( i )));

  int xf_offset = 0;
  int yf_offset = (nx+1)*ny*nz;
  int zf_offset = (nx+1)*ny*nz + nx*(ny+1)*nz;

  // create cubes
  for (int k = 0; k < nz; k++)
    for (int j = 0; j < ny; j++)
      for (int i = 0; i < nx; i++)
        cubes.push_back(cube_t(xf_offset + k*(nx+1)*( ny ) + j*(nx+1) + i, xf_offset + ( k )*(nx+1)*( ny ) + ( j )*(nx+1) + (i+1),
                               yf_offset + k*( nx )*(ny+1) + j*( nx ) + i, yf_offset + ( k )*( nx )*(ny+1) + (j+1)*( nx ) + ( i ),
                               zf_offset + k*( nx )*( ny ) + j*( nx ) + i, zf_offset + (k+1)*( nx )*( ny ) + ( j )*( nx ) + ( i )));

  // mark cubes next to walls
  for (int k = 0; k < nz; k++)
    for (int j = 0; j < ny; j++)
    {
      int i;
      i = 0;    cubes[k*nx*ny + j*nx + i].wall[0] = true;
      i = nx-1; cubes[k*nx*ny + j*nx + i].wall[1] = true;
    }

  for (int k = 0; k < nz; k++)
    for (int i = 0; i < nx; i++)
    {
      int j;
      j = 0;    cubes[k*nx*ny + j*nx + i].wall[2] = true;
      j = ny-1; cubes[k*nx*ny + j*nx + i].wall[3] = true;
    }

  for (int j = 0; j < ny; j++)
    for (int i = 0; i < nx; i++)
    {
      int k;
      k = 0;    cubes[k*nx*ny + j*nx + i].wall[4] = true;
      k = nz-1; cubes[k*nx*ny + j*nx + i].wall[5] = true;
    }


  // do recursive refinement
  n_nodes = 0;
  n_cubes = 0;
  n_leafs = nx*ny*nz;

  phi.resize    (n_phis, std::vector<double> ((nx+1)*(ny+1)*(nz+1), -1.));

  phi_xx.resize (n_phis, std::vector<double> ((nx+1)*(ny+1)*(nz+1), 0.));
  phi_yy.resize (n_phis, std::vector<double> ((nx+1)*(ny+1)*(nz+1), 0.));
  phi_zz.resize (n_phis, std::vector<double> ((nx+1)*(ny+1)*(nz+1), 0.));

  for (int l = -1; l < level; l++)
  {
    int n_new = cubes.size();

    // check if no new cubes have been created
    if (n_new == n_cubes) break;

    if (l != -1)
    {
      // loop through new cubes
      for (int n = n_cubes; n < n_new; n++)
        if (need_split(n))
        {
          split_cube(n);
          n_leafs += N_CHILDREN-1;
        }

      n_cubes = n_new;
    }

    // interpolate LSF to new nodes
    for (int q = 0; q < n_phis; q++)
    {
      phi[q].resize   (nodes.size(),0.);

      phi_xx[q].resize(nodes.size(),0.);
      phi_yy[q].resize(nodes.size(),0.);
      phi_zz[q].resize(nodes.size(),0.);

      for (int i = n_nodes; i < nodes.size(); i++)
      {
        if (phi_cf_in == NULL)
        {
          phi[q][i] = interp.quadratic(phi_in->at(q).data(), phi_xx_in->at(q).data(), phi_yy_in->at(q).data(), phi_zz_in->at(q).data(),
                                       nodes[i].x, nodes[i].y, nodes[i].z);

          phi_xx[q][i] = interp.linear(phi_xx_in->at(q).data(), nodes[i].x, nodes[i].y, nodes[i].z);
          phi_yy[q][i] = interp.linear(phi_yy_in->at(q).data(), nodes[i].x, nodes[i].y, nodes[i].z);
          phi_zz[q][i] = interp.linear(phi_zz_in->at(q).data(), nodes[i].x, nodes[i].y, nodes[i].z);

        } else {

          double phi_000 = (*phi_cf_in->at(q))(nodes[i].x, nodes[i].y, nodes[i].z);
          double phi_m00 = (*phi_cf_in->at(q))(nodes[i].x-dx_min, nodes[i].y, nodes[i].z);
          double phi_p00 = (*phi_cf_in->at(q))(nodes[i].x+dx_min, nodes[i].y, nodes[i].z);
          double phi_0m0 = (*phi_cf_in->at(q))(nodes[i].x, nodes[i].y-dy_min, nodes[i].z);
          double phi_0p0 = (*phi_cf_in->at(q))(nodes[i].x, nodes[i].y+dy_min, nodes[i].z);
          double phi_00m = (*phi_cf_in->at(q))(nodes[i].x, nodes[i].y, nodes[i].z-dz_min);
          double phi_00p = (*phi_cf_in->at(q))(nodes[i].x, nodes[i].y, nodes[i].z+dz_min);

          phi[q][i] = phi_000;

          phi_xx[q][i] = (phi_p00+phi_m00-2.0*phi_000)/dx_min/dx_min;
          phi_yy[q][i] = (phi_0p0+phi_0m0-2.0*phi_000)/dy_min/dy_min;
          phi_zz[q][i] = (phi_00p+phi_00m-2.0*phi_000)/dz_min/dz_min;

        }
      }
    }

    n_nodes = nodes.size();
  }

  n_cubes = cubes.size();

  leaf_to_node.resize(n_leafs*N_CHILDREN,-1);
  get_cube.resize(n_leafs,-1);

  int i = 0;
  for (int n = 0; n < n_cubes; n++)
  {
    if (!cubes[n].is_split)
    {
      int f_m00 = cubes[n].f_m00;
      int f_p00 = cubes[n].f_p00;

      int e_mm0 = faces[f_m00].e_m0;
      int e_mp0 = faces[f_m00].e_p0;

      int e_pm0 = faces[f_p00].e_m0;
      int e_pp0 = faces[f_p00].e_p0;

      int v_mmm = edges[e_mm0].v0;
      int v_mmp = edges[e_mm0].v1;
      int v_pmm = edges[e_pm0].v0;
      int v_pmp = edges[e_pm0].v1;
      int v_mpm = edges[e_mp0].v0;
      int v_mpp = edges[e_mp0].v1;
      int v_ppm = edges[e_pp0].v0;
      int v_ppp = edges[e_pp0].v1;

      leaf_to_node[i*N_CHILDREN+0] = v_mmm;
      leaf_to_node[i*N_CHILDREN+1] = v_pmm;
      leaf_to_node[i*N_CHILDREN+2] = v_mpm;
      leaf_to_node[i*N_CHILDREN+3] = v_ppm;
      leaf_to_node[i*N_CHILDREN+4] = v_mmp;
      leaf_to_node[i*N_CHILDREN+5] = v_pmp;
      leaf_to_node[i*N_CHILDREN+6] = v_mpp;
      leaf_to_node[i*N_CHILDREN+7] = v_ppp;

      get_cube[i] = n;
      i++;
    }
  }


  /* Split the cube into sub-cubes */
  cubes_mls.clear();
  cubes_mls.reserve(n_leafs);

//  std::vector<double> phi_cube    (N_CHILDREN*n_phis, -1.);

//  std::vector<double> phi_xx_cube (N_CHILDREN*n_phis, 0.);
//  std::vector<double> phi_yy_cube (N_CHILDREN*n_phis, 0.);
//  std::vector<double> phi_zz_cube (N_CHILDREN*n_phis, 0.);

  std::vector< std::vector<double> > phi_cube    (n_phis, std::vector<double> (N_CHILDREN, -1.));
  std::vector< std::vector<double> > phi_xx_cube (n_phis, std::vector<double> (N_CHILDREN, 0.));
  std::vector< std::vector<double> > phi_yy_cube (n_phis, std::vector<double> (N_CHILDREN, 0.));
  std::vector< std::vector<double> > phi_zz_cube (n_phis, std::vector<double> (N_CHILDREN, 0.));

  // reconstruct interfaces in leaf cubes
  for (int n = 0; n < n_leafs; n++)
  {
    // fetch corners of a cube
    int v_mm = leaf_to_node[N_CHILDREN*n];
    int v_pp = leaf_to_node[N_CHILDREN*n + N_CHILDREN-1];

    // create a mls cube
    cubes_mls.push_back(cube3_mls_t(nodes[v_mm].x, nodes[v_pp].x, nodes[v_mm].y, nodes[v_pp].y, nodes[v_mm].z, nodes[v_pp].z));

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
//    cubes_mls.back().set_phi(phi_cube, *action, *color);
    cubes_mls.back().construct_domain();
  }
}

void cube3_refined_mls_t::sample_all(std::vector<double> &f, std::vector<double> &f_values)
{
  for (int i = 0; i < n_nodes; i++)
    f_values[i] = interp.linear(f, nodes[i].x, nodes[i].y, nodes[i].z);
}

void cube3_refined_mls_t::sample_all(CF_3 &f, std::vector<double> &f_values)
{
  for (int i = 0; i < n_nodes; i++)
    f_values[i] = f(nodes[i].x, nodes[i].y, nodes[i].z);
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

//double cube3_refined_mls_t::perform(double f, int type, int num0, int num1, int num2)
//{
//  double result = 0;
//  double f_cube[N_CHILDREN] = {f,f,f,f,f,f,f,f};
//  int n_phis = action->size();

//  std::vector<double> phi_cube    (N_CHILDREN*n_phis, -1.);

//  std::vector<double> phi_x_cube  (N_CHILDREN*n_phis, 0.);
//  std::vector<double> phi_y_cube  (N_CHILDREN*n_phis, 0.);
//  std::vector<double> phi_z_cube  (N_CHILDREN*n_phis, 0.);

//  std::vector<double> phi_xx_cube (N_CHILDREN*n_phis, 0.);
//  std::vector<double> phi_yy_cube (N_CHILDREN*n_phis, 0.);
//  std::vector<double> phi_zz_cube (N_CHILDREN*n_phis, 0.);


//  for (int n = 0; n < n_leafs; n++)
//  {
//    // fetch corners of a cube
//    int v_mm = leaf_to_node[N_CHILDREN*n];
//    int v_pp = leaf_to_node[N_CHILDREN*n + N_CHILDREN-1];

//    // create a mls cube
//    cube3_mls_t cube(nodes[v_mm].x, nodes[v_pp].x, nodes[v_mm].y, nodes[v_pp].y, nodes[v_mm].z, nodes[v_pp].z);

//    // fetch function values at the corners of a cube
//    for (int q = 0; q < n_phis; q++)
//      for (int d = 0; d < N_CHILDREN; d++)
//      {
//        phi_cube   [q*N_CHILDREN + d] = phi   [q][leaf_to_node[n*N_CHILDREN+d]];

//        phi_x_cube [q*N_CHILDREN + d] = phi_x [q][leaf_to_node[n*N_CHILDREN+d]];
//        phi_y_cube [q*N_CHILDREN + d] = phi_y [q][leaf_to_node[n*N_CHILDREN+d]];
//        phi_z_cube [q*N_CHILDREN + d] = phi_z [q][leaf_to_node[n*N_CHILDREN+d]];

//        phi_xx_cube[q*N_CHILDREN + d] = phi_xx[q][leaf_to_node[n*N_CHILDREN+d]];
//        phi_yy_cube[q*N_CHILDREN + d] = phi_yy[q][leaf_to_node[n*N_CHILDREN+d]];
//        phi_zz_cube[q*N_CHILDREN + d] = phi_zz[q][leaf_to_node[n*N_CHILDREN+d]];
//      }

//    cube.set_phi(phi_cube);
//    cube.set_phi_d(phi_x_cube, phi_y_cube, phi_z_cube);
//    cube.set_phi_dd(phi_xx_cube, phi_yy_cube, phi_zz_cube);

//    cube.construct_domain(*action, *color);

//    switch (type) {
//    case 0: result += cube.integrate_over_domain            (f_cube); break;
//    case 1: result += cube.integrate_over_interface         (f_cube, num0); break;
//    case 2: result += cube.integrate_over_intersection      (f_cube, num0, num1); break;
//    case 3: result += cube.integrate_over_intersection      (f_cube, num0, num1, num2); break;
//    case 4: result += cube.integrate_over_colored_interface (f_cube, num0, num1); break;
////    case 5: result += cubes_mls[n].integrate_in_non_cart_dir        (f_cube, num0); break;
//    }
//  }

//  return result;
//}

double cube3_refined_mls_t::integrate_in_dir(std::vector<double> &f, int dir)
{
  std::vector<double> f_values(n_nodes,1);
  sample_all(f, f_values);

  double result = 0;
  double f_cube[N_CHILDREN];

  for (int n = 0; n < n_leafs; n++)
    if (cubes[get_cube[n]].wall[dir])
    {
      // fetch function values at corners of a cube
      for (int d = 0; d < N_CHILDREN; d++)
        f_cube[d] = f_values[leaf_to_node[n*N_CHILDREN + d]];
      result += cubes_mls[n].integrate_in_dir(f_cube, dir);
    }

  return result;
}

double cube3_refined_mls_t::integrate_in_dir(CF_3 &f, int dir)
{
  std::vector<double> f_values(n_nodes,1);
  sample_all(f, f_values);

  double result = 0;
  double f_cube[N_CHILDREN];

  for (int n = 0; n < n_leafs; n++)
    if (cubes[get_cube[n]].wall[dir])
    {
      // fetch function values at corners of a cube
      for (int d = 0; d < N_CHILDREN; d++)
        f_cube[d] = f_values[leaf_to_node[n*N_CHILDREN + d]];
      result += cubes_mls[n].integrate_in_dir(f_cube, dir);
    }

  return result;
}

double cube3_refined_mls_t::integrate_in_dir(double f, int dir)
{
  double result = 0;
  double f_cube[N_CHILDREN] = {f,f,f,f,f,f,f,f};

  for (int n = 0; n < n_leafs; n++)
    if (cubes[get_cube[n]].wall[dir])
    {
      result += cubes_mls[n].integrate_in_dir(f_cube, dir);
    }

  return result;
}

void cube3_refined_mls_t::split_edge(int n)
{
  // check if an edge has already been split
  if (edges[n].is_split) return;

  // mark that an edge is split
  edges[n].is_split = true;

//  edges.reserve(edges.size()+2);

  // fetch coordinates of nodes
  double x_0 = nodes[edges[n].v0].x; double y_0 = nodes[edges[n].v0].y; double z_0 = nodes[edges[n].v0].z;
  double x_1 = nodes[edges[n].v1].x; double y_1 = nodes[edges[n].v1].y; double z_1 = nodes[edges[n].v1].z;

#ifdef CASL_THROWS
  if (x_0 > x_1) throw std::invalid_argument("[CASL_ERROR]: ");
  if (y_0 > y_1) throw std::invalid_argument("[CASL_ERROR]: ");
  if (z_0 > z_1) throw std::invalid_argument("[CASL_ERROR]: ");
#endif

  // create the splitting node
  nodes.push_back(node_t(0.5*(x_0+x_1), 0.5*(y_0+y_1), 0.5*(z_0+z_1)));
  edges[n].v01 = nodes.size()-1;

  // create child edges
  edges.push_back(edge_t(edges[n].v0,  edges[n].v01)); edges[n].e0 = edges.size()-1;
  edges.push_back(edge_t(edges[n].v01, edges[n].v1 )); edges[n].e1 = edges.size()-1;
}

void cube3_refined_mls_t::split_face(int n)
{
  // check if a cube has already been split
  if (faces[n].is_split) return;

  // mark that a cube is split
  faces[n].is_split = true;

  // split edges
  split_edge(faces[n].e_m0);
  split_edge(faces[n].e_p0);
  split_edge(faces[n].e_0m);
  split_edge(faces[n].e_0p);

//  edges.reserve(edges.size()+4);
//  faces.reserve(faces.size()+4);

  // fetch child elements of edges
  int v_mm = edges[faces[n].e_0m].v0;
//  int v_pm = edges[cubes[n].e_0m].v1;
//  int v_mp = edges[cubes[n].e_0p].v0;
  int v_pp = edges[faces[n].e_0p].v1;

  int v_m0 = edges[faces[n].e_m0].v01;
  int v_p0 = edges[faces[n].e_p0].v01;
  int v_0m = edges[faces[n].e_0m].v01;
  int v_0p = edges[faces[n].e_0p].v01;

  int e_m0_0m = edges[faces[n].e_m0].e0;
  int e_m0_0p = edges[faces[n].e_m0].e1;

  int e_p0_0m = edges[faces[n].e_p0].e0;
  int e_p0_0p = edges[faces[n].e_p0].e1;

  int e_0m_m0 = edges[faces[n].e_0m].e0;
  int e_0m_p0 = edges[faces[n].e_0m].e1;

  int e_0p_m0 = edges[faces[n].e_0p].e0;
  int e_0p_p0 = edges[faces[n].e_0p].e1;

  // create central node
  double x_0 = nodes[v_mm].x; double y_0 = nodes[v_mm].y; double z_0 = nodes[v_mm].z;
  double x_1 = nodes[v_pp].x; double y_1 = nodes[v_pp].y; double z_1 = nodes[v_pp].z;

  nodes.push_back(node_t(0.5*(x_0+x_1), 0.5*(y_0+y_1), 0.5*(z_0+z_1))); int v_00 = nodes.size()-1;
  faces[n].v_00 = v_00;

  // create internal child edges
  edges.push_back(edge_t(v_m0,v_00)); int e_00_m0 = edges.size()-1; faces[n].ce_m0 = e_00_m0;
  edges.push_back(edge_t(v_00,v_p0)); int e_00_p0 = edges.size()-1; faces[n].ce_p0 = e_00_p0;
  edges.push_back(edge_t(v_0m,v_00)); int e_00_0m = edges.size()-1; faces[n].ce_0m = e_00_0m;
  edges.push_back(edge_t(v_00,v_0p)); int e_00_0p = edges.size()-1; faces[n].ce_0p = e_00_0p;

  // create new cubes
  faces.push_back(face_t(e_m0_0m, e_00_0m, e_0m_m0, e_00_m0)); faces[n].f_mm = faces.size()-1;
  faces.push_back(face_t(e_00_0m, e_p0_0m, e_0m_p0, e_00_p0)); faces[n].f_pm = faces.size()-1;
  faces.push_back(face_t(e_m0_0p, e_00_0p, e_00_m0, e_0p_m0)); faces[n].f_mp = faces.size()-1;
  faces.push_back(face_t(e_00_0p, e_p0_0p, e_00_p0, e_0p_p0)); faces[n].f_pp = faces.size()-1;
}

void cube3_refined_mls_t::split_cube(int n)
{
  // check if a cube has already been split
  if (cubes[n].is_split) return;

  // mark that a cube is split
  cubes[n].is_split = true;

  // fetch faces
  int f_m00 = cubes[n].f_m00;
  int f_p00 = cubes[n].f_p00;
  int f_0m0 = cubes[n].f_0m0;
  int f_0p0 = cubes[n].f_0p0;
  int f_00m = cubes[n].f_00m;
  int f_00p = cubes[n].f_00p;

  // split faces
  split_face(f_m00);
  split_face(f_p00);
  split_face(f_0m0);
  split_face(f_0p0);
  split_face(f_00m);
  split_face(f_00p);

//  cubes.reserve(cubes.size()+N_CHILDREN);
//  edges.reserve(cubes.size()+6);
//  faces.reserve(faces.size()+12);

  // fetch sub-faces
  int f_m00_mm = faces[f_m00].f_mm;
  int f_m00_pm = faces[f_m00].f_pm;
  int f_m00_mp = faces[f_m00].f_mp;
  int f_m00_pp = faces[f_m00].f_pp;

  int f_p00_mm = faces[f_p00].f_mm;
  int f_p00_pm = faces[f_p00].f_pm;
  int f_p00_mp = faces[f_p00].f_mp;
  int f_p00_pp = faces[f_p00].f_pp;

  int f_0m0_mm = faces[f_0m0].f_mm;
  int f_0m0_pm = faces[f_0m0].f_pm;
  int f_0m0_mp = faces[f_0m0].f_mp;
  int f_0m0_pp = faces[f_0m0].f_pp;

  int f_0p0_mm = faces[f_0p0].f_mm;
  int f_0p0_pm = faces[f_0p0].f_pm;
  int f_0p0_mp = faces[f_0p0].f_mp;
  int f_0p0_pp = faces[f_0p0].f_pp;

  int f_00m_mm = faces[f_00m].f_mm;
  int f_00m_pm = faces[f_00m].f_pm;
  int f_00m_mp = faces[f_00m].f_mp;
  int f_00m_pp = faces[f_00m].f_pp;

  int f_00p_mm = faces[f_00p].f_mm;
  int f_00p_pm = faces[f_00p].f_pm;
  int f_00p_mp = faces[f_00p].f_mp;
  int f_00p_pp = faces[f_00p].f_pp;

  // fetch sub-edges
  int e_m00_m0 = faces[f_m00].ce_m0;
  int e_m00_p0 = faces[f_m00].ce_p0;
  int e_m00_0m = faces[f_m00].ce_0m;
  int e_m00_0p = faces[f_m00].ce_0p;

  int e_p00_m0 = faces[f_p00].ce_m0;
  int e_p00_p0 = faces[f_p00].ce_p0;
  int e_p00_0m = faces[f_p00].ce_0m;
  int e_p00_0p = faces[f_p00].ce_0p;

  int e_0m0_m0 = faces[f_0m0].ce_m0;
  int e_0m0_p0 = faces[f_0m0].ce_p0;
  int e_0m0_0m = faces[f_0m0].ce_0m;
  int e_0m0_0p = faces[f_0m0].ce_0p;

  int e_0p0_m0 = faces[f_0p0].ce_m0;
  int e_0p0_p0 = faces[f_0p0].ce_p0;
  int e_0p0_0m = faces[f_0p0].ce_0m;
  int e_0p0_0p = faces[f_0p0].ce_0p;

  int e_00m_m0 = faces[f_00m].ce_m0;
  int e_00m_p0 = faces[f_00m].ce_p0;
  int e_00m_0m = faces[f_00m].ce_0m;
  int e_00m_0p = faces[f_00m].ce_0p;

  int e_00p_m0 = faces[f_00p].ce_m0;
  int e_00p_p0 = faces[f_00p].ce_p0;
  int e_00p_0m = faces[f_00p].ce_0m;
  int e_00p_0p = faces[f_00p].ce_0p;

  // create sub-edges
  int v_m00 = faces[f_m00].v_00;
  int v_p00 = faces[f_p00].v_00;
  int v_0m0 = faces[f_0m0].v_00;
  int v_0p0 = faces[f_0p0].v_00;
  int v_00m = faces[f_00m].v_00;
  int v_00p = faces[f_00p].v_00;

  // create middle node
  nodes.push_back(node_t(nodes[v_0m0].x, nodes[v_m00].y, nodes[v_m00].z)); int v_000 = nodes.size()-1;

  edges.push_back(edge_t(v_m00, v_000)); int e_m00 = edges.size()-1;
  edges.push_back(edge_t(v_000, v_p00)); int e_p00 = edges.size()-1;
  edges.push_back(edge_t(v_0m0, v_000)); int e_0m0 = edges.size()-1;
  edges.push_back(edge_t(v_000, v_0p0)); int e_0p0 = edges.size()-1;
  edges.push_back(edge_t(v_00m, v_000)); int e_00m = edges.size()-1;
  edges.push_back(edge_t(v_000, v_00p)); int e_00p = edges.size()-1;

  // create sub-faces
  faces.push_back(face_t(e_0m0_0m, e_00m, e_00m_0m, e_0m0)); int f_0mm = faces.size()-1;
  faces.push_back(face_t(e_00m, e_0p0_0m, e_00m_0p, e_0p0)); int f_0pm = faces.size()-1;
  faces.push_back(face_t(e_0m0_0p, e_00p, e_0m0, e_00p_0m)); int f_0mp = faces.size()-1;
  faces.push_back(face_t(e_00p, e_0p0_0p, e_0p0, e_00p_0p)); int f_0pp = faces.size()-1;

  faces.push_back(face_t(e_m00_0m, e_00m, e_00m_m0, e_m00)); int f_m0m = faces.size()-1;
  faces.push_back(face_t(e_00m, e_p00_0m, e_00m_p0, e_p00)); int f_p0m = faces.size()-1;
  faces.push_back(face_t(e_m00_0p, e_00p, e_m00, e_00p_m0)); int f_m0p = faces.size()-1;
  faces.push_back(face_t(e_00p, e_p00_0p, e_p00, e_00p_p0)); int f_p0p = faces.size()-1;

  faces.push_back(face_t(e_m00_m0, e_0m0, e_0m0_m0, e_m00)); int f_mm0 = faces.size()-1;
  faces.push_back(face_t(e_0m0, e_p00_m0, e_0m0_p0, e_p00)); int f_pm0 = faces.size()-1;
  faces.push_back(face_t(e_m00_p0, e_0p0, e_m00, e_0p0_m0)); int f_mp0 = faces.size()-1;
  faces.push_back(face_t(e_0p0, e_p00_p0, e_p00, e_0p0_p0)); int f_pp0 = faces.size()-1;

  bool *wall = cubes[n].wall;

  // create cubes
  cubes.push_back(cube_t(f_m00_mm, f_0mm, f_0m0_mm, f_m0m, f_00m_mm, f_mm0));
  cubes.back().wall[0] = cubes[n].wall[0];
  cubes.back().wall[2] = cubes[n].wall[2];
  cubes.back().wall[4] = cubes[n].wall[4];
  cubes.push_back(cube_t(f_0mm, f_p00_mm, f_0m0_pm, f_p0m, f_00m_pm, f_pm0));
  cubes.back().wall[1] = cubes[n].wall[1];
  cubes.back().wall[2] = cubes[n].wall[2];
  cubes.back().wall[4] = cubes[n].wall[4];
  cubes.push_back(cube_t(f_m00_pm, f_0pm, f_m0m, f_0p0_mm, f_00m_mp, f_mp0));
  cubes.back().wall[0] = cubes[n].wall[0];
  cubes.back().wall[3] = cubes[n].wall[3];
  cubes.back().wall[4] = cubes[n].wall[4];
  cubes.push_back(cube_t(f_0pm, f_p00_pm, f_p0m, f_0p0_pm, f_00m_pp, f_pp0));
  cubes.back().wall[1] = cubes[n].wall[1];
  cubes.back().wall[3] = cubes[n].wall[3];
  cubes.back().wall[4] = cubes[n].wall[4];
  cubes.push_back(cube_t(f_m00_mp, f_0mp, f_0m0_mp, f_m0p, f_mm0, f_00p_mm));
  cubes.back().wall[0] = cubes[n].wall[0];
  cubes.back().wall[2] = cubes[n].wall[2];
  cubes.back().wall[5] = cubes[n].wall[5];
  cubes.push_back(cube_t(f_0mp, f_p00_mp, f_0m0_pp, f_p0p, f_pm0, f_00p_pm));
  cubes.back().wall[1] = cubes[n].wall[1];
  cubes.back().wall[2] = cubes[n].wall[2];
  cubes.back().wall[5] = cubes[n].wall[5];
  cubes.push_back(cube_t(f_m00_pp, f_0pp, f_m0p, f_0p0_mp, f_mp0, f_00p_mp));
  cubes.back().wall[0] = cubes[n].wall[0];
  cubes.back().wall[3] = cubes[n].wall[3];
  cubes.back().wall[5] = cubes[n].wall[5];
  cubes.push_back(cube_t(f_0pp, f_p00_pp, f_p0p, f_0p0_pp, f_pp0, f_00p_pp));
  cubes.back().wall[1] = cubes[n].wall[1];
  cubes.back().wall[3] = cubes[n].wall[3];
  cubes.back().wall[5] = cubes[n].wall[5];
}


bool cube3_refined_mls_t::need_split(int n)
{
  int v[N_CHILDREN];
  bool all_negative, all_positive;

  // TODO: fix the fetch of corner nodes

  int f_m00 = cubes[n].f_m00;
  int f_p00 = cubes[n].f_p00;

  int e_mm0 = faces[f_m00].e_m0;
  int e_mp0 = faces[f_m00].e_p0;

  int e_pm0 = faces[f_p00].e_m0;
  int e_pp0 = faces[f_p00].e_p0;

  v[0] = edges[e_mm0].v0;
  v[1] = edges[e_pm0].v0;
  v[2] = edges[e_mp0].v0;
  v[3] = edges[e_pp0].v0;
  v[4] = edges[e_mm0].v1;
  v[5] = edges[e_pm0].v1;
  v[6] = edges[e_mp0].v1;
  v[7] = edges[e_pp0].v1;

  bool result = false;
  double phi_eff = -10;

  double xm = nodes[v[0]].x, ym = nodes[v[0]].y, zm = nodes[v[0]].y;
  double xp = nodes[v[7]].x, yp = nodes[v[7]].y, zp = nodes[v[7]].y;

  double diag = sqrt((xp-xm)*(xp-xm)+(yp-ym)*(yp-ym)+(zp-zm)*(zp-zm));


  for (int i = 0; i < action->size(); i++)
  {
    all_negative = true;
    all_positive = true;

    for (int j = 0; j < N_CHILDREN; j++)
    {
      all_negative = (all_negative && (phi[i][v[j]] < 0.0));
      all_positive = (all_positive && (phi[i][v[j]] > 0.0));
    }

    if (all_positive)
    {
      if (action->at(i) == INTERSECTION)
      {
        loc = OUT;
        result = false;
      }
    }
    else if (all_negative)
    {
      if (action->at(i) == ADDITION)
      {
        loc = INS;
        result = false;
      }
    }
    else if (loc == FCE || (loc == INS && action->at(i) == INTERSECTION) || (loc == OUT && action->at(i) == ADDITION))
    {
      loc = FCE;
      result = true;
    }

    double phi_cur = 0;

    for (int j = 0; j < N_CHILDREN; j++)
    {
      phi_cur += phi[i][v[j]];
    }

    phi_cur /= 8.;

    if (action->at(i) == INTERSECTION)
    {
      phi_eff = MAX(phi_eff, phi_cur);
    }
    if (action->at(i) == ADDITION)
    {
      phi_eff = MIN(phi_eff, phi_cur);
    }

  }

  if (fabs(phi_eff) < 0.7*diag) result = true;

//  return true;
  return result;
}

//bool cube3_refined_mls_t::face_crossed(int n)
//{
//  int level_diff = faces[n].level - level_max;

//  int NX = pow(2, level_diff);
//  int NY = pow(2, level_diff);

//}
