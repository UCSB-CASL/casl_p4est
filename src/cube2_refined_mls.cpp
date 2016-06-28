#include "cube2_refined_mls.h"

void cube2_refined_mls_t::construct_domain(int nx_, int ny_, int level)
{
  // clean up everything
  nodes.clear();
  edges.clear();
  cubes.clear();
  cubes_mls.clear();

  leaf_to_node.clear();
  get_cube.clear();

  int n_phis = action->size();

  nx = nx_; ny = ny_;

  double dx = (x1-x0)/(double)(nx);
  double dy = (y1-y0)/(double)(ny);

  dx_min = dx/pow(2.,(double)(level+1));
  dy_min = dy/pow(2.,(double)(level+1));

  std::vector<double> x_coord(nx+1, 0.); for (int i = 0; i < nx+1; i++) {x_coord[i]=(x0 + dx*(double)(i));}
  std::vector<double> y_coord(ny+1, 0.); for (int j = 0; j < ny+1; j++) {y_coord[j]=(y0 + dy*(double)(j));}

  // do initial splitting
  nodes.reserve((nx+1)*(ny+1));
  edges.reserve(nx*(ny+1)+(nx+1)*ny);
  cubes.reserve(nx*ny);

  // create nodes
  for (int j = 0; j < ny+1; j++)
    for (int i = 0; i < nx+1; i++)
      nodes.push_back(node_t(x_coord[i],y_coord[j]));

  // create x-edges
  for (int j = 0; j < ny; j++)
    for (int i = 0; i < nx+1; i++)
      edges.push_back(edge_t(j*(nx+1)+i, (j+1)*(nx+1)+i));

  // create y-edges
  for (int j = 0; j < ny+1; j++)
    for (int i = 0; i < nx; i++)
      edges.push_back(edge_t(j*(nx+1)+i, j*(nx+1)+i+1));

  // create cubes
  int n_xedges = ny*(nx+1);
  for (int j = 0; j < ny; j++)
    for (int i = 0; i < nx; i++)
      cubes.push_back(cube_t(j*(nx+1) + i, j*(nx+1) + i+1, n_xedges + j*(nx+0) + i, n_xedges + (j+1)*(nx+0) + i));

  // mark cubes next to walls
  for (int j = 0; j < ny; j++)
  {
    int i = 0; cubes[j*nx+i].wall[0] = true;
    i = nx-1;  cubes[j*nx+i].wall[1] = true;
  }
  for (int i = 0; i < nx; i++)
  {
    int j = 0; cubes[j*nx+i].wall[2] = true;
    j = ny-1;  cubes[j*nx+i].wall[3] = true;
  }

  // do recursive refinement
  n_nodes = 0;
  n_cubes = 0;
  n_leafs = nx*ny;

  phi.resize    (n_phis, std::vector<double> ((nx+1)*(ny+1), -1.));
  phi_xx.resize (n_phis, std::vector<double> ((nx+1)*(ny+1), 0.));
  phi_yy.resize (n_phis, std::vector<double> ((nx+1)*(ny+1), 0.));

//  phi.resize    (n_phis);
//  phi_x.resize  (n_phis);
//  phi_y.resize  (n_phis);
//  phi_xx.resize (n_phis);
//  phi_yy.resize (n_phis);

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
      phi[q].resize(nodes.size());
      phi_xx[q].resize(nodes.size());
      phi_yy[q].resize(nodes.size());
      for (int i = n_nodes; i < nodes.size(); i++)
      {
        if (phi_cf_in == NULL)
        {
          phi[q][i] = interp.quadratic(phi_in->at(q), phi_xx_in->at(q), phi_yy_in->at(q), nodes[i].x, nodes[i].y);
          phi_xx[q][i] = interp.linear(phi_xx_in->at(q), nodes[i].x, nodes[i].y);
          phi_yy[q][i] = interp.linear(phi_yy_in->at(q), nodes[i].x, nodes[i].y);
        } else {
          double phi_000 = (*phi_cf_in->at(q))(nodes[i].x, nodes[i].y);
          double phi_m00 = (*phi_cf_in->at(q))(nodes[i].x-dx_min, nodes[i].y);
          double phi_p00 = (*phi_cf_in->at(q))(nodes[i].x+dx_min, nodes[i].y);
          double phi_0m0 = (*phi_cf_in->at(q))(nodes[i].x, nodes[i].y-dy_min);
          double phi_0p0 = (*phi_cf_in->at(q))(nodes[i].x, nodes[i].y+dy_min);
          phi[q][i] = phi_000;
          phi_xx[q][i] = (phi_p00+phi_m00-2.0*phi_000)/dx_min/dx_min;
          phi_yy[q][i] = (phi_0p0+phi_0m0-2.0*phi_000)/dy_min/dy_min;
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
      leaf_to_node[i*N_CHILDREN+0] = edges[cubes[n].e_m0].v0;
      leaf_to_node[i*N_CHILDREN+1] = edges[cubes[n].e_p0].v0;
      leaf_to_node[i*N_CHILDREN+2] = edges[cubes[n].e_m0].v1;
      leaf_to_node[i*N_CHILDREN+3] = edges[cubes[n].e_p0].v1;
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

  std::vector< std::vector<double> > phi_cube    (n_phis, std::vector<double> (N_CHILDREN, -1.));
  std::vector< std::vector<double> > phi_xx_cube (n_phis, std::vector<double> (N_CHILDREN, 0.));
  std::vector< std::vector<double> > phi_yy_cube (n_phis, std::vector<double> (N_CHILDREN, 0.));

  // reconstruct interfaces in leaf cubes
  for (int n = 0; n < n_leafs; n++)
  {
    // fetch corners of a cube
    int v_mm = leaf_to_node[N_CHILDREN*n];
    int v_pp = leaf_to_node[N_CHILDREN*n + N_CHILDREN-1];

    // create a mls cube
    cubes_mls.push_back(cube2_mls_t(nodes[v_mm].x, nodes[v_pp].x, nodes[v_mm].y, nodes[v_pp].y));

    // fetch function values at the corners of a cube
    for (int q = 0; q < n_phis; q++)
      for (int d = 0; d < N_CHILDREN; d++)
      {
        phi_cube   [q][d] = phi   [q][leaf_to_node[n*N_CHILDREN+d]];
        phi_xx_cube[q][d] = phi_xx[q][leaf_to_node[n*N_CHILDREN+d]];
        phi_yy_cube[q][d] = phi_yy[q][leaf_to_node[n*N_CHILDREN+d]];
      }

    cubes_mls.back().set_phi(phi_cube, phi_xx_cube, phi_yy_cube, *action, *color);
    cubes_mls.back().construct_domain();
  }
}

void cube2_refined_mls_t::sample_all(std::vector<double> &f, std::vector<double> &f_values)
{
  for (int i = 0; i < n_nodes; i++)
    f_values[i] = interp.linear(f, nodes[i].x, nodes[i].y);
}

void cube2_refined_mls_t::sample_all(double *f, std::vector<double> &f_values)
{
  for (int i = 0; i < n_nodes; i++)
    f_values[i] = interp.linear(f, nodes[i].x, nodes[i].y);
}

void cube2_refined_mls_t::sample_all(CF_2 &f, std::vector<double> &f_values)
{
  for (int i = 0; i < n_nodes; i++)
    f_values[i] = f(nodes[i].x, nodes[i].y);
}

double cube2_refined_mls_t::perform(std::vector<double> &f, int type, int num0, int num1)
{
  std::vector<double> f_values(n_nodes,1);
  sample_all(f, f_values);

  double result = 0;
  double f_cube[4];

  for (int n = 0; n < n_leafs; n++)
  {
    // fetch function values at corners of a cube
    for (int d = 0; d < N_CHILDREN; d++)
      f_cube[d] = f_values[leaf_to_node[n*N_CHILDREN + d]];
    switch (type) {
    case 0: result += cubes_mls[n].integrate_over_domain            (f_cube); break;
    case 1: result += cubes_mls[n].integrate_over_interface         (f_cube, num0); break;
    case 2: result += cubes_mls[n].integrate_over_intersection      (f_cube, num0, num1); break;
    case 3: result += cubes_mls[n].integrate_over_colored_interface (f_cube, num0, num1); break;
    case 4: result += cubes_mls[n].integrate_in_non_cart_dir        (f_cube, num0); break;
    }
  }

  return result;
}

double cube2_refined_mls_t::perform(double *f, int type, int num0, int num1)
{
  std::vector<double> f_values(n_nodes,1);
  sample_all(f, f_values);

  double result = 0;
  double f_cube[4];

  for (int n = 0; n < n_leafs; n++)
  {
    // fetch function values at corners of a cube
    for (int d = 0; d < N_CHILDREN; d++)
      f_cube[d] = f_values[leaf_to_node[n*N_CHILDREN + d]];
    switch (type) {
    case 0: result += cubes_mls[n].integrate_over_domain            (f_cube); break;
    case 1: result += cubes_mls[n].integrate_over_interface         (f_cube, num0); break;
    case 2: result += cubes_mls[n].integrate_over_intersection      (f_cube, num0, num1); break;
    case 3: result += cubes_mls[n].integrate_over_colored_interface (f_cube, num0, num1); break;
    case 4: result += cubes_mls[n].integrate_in_non_cart_dir        (f_cube, num0); break;
    }
  }

  return result;
}

double cube2_refined_mls_t::perform(CF_2 &f, int type, int num0, int num1)
{
  std::vector<double> f_values(n_nodes,1);
  sample_all(f, f_values);

  double result = 0;
  double f_cube[4];

  for (int n = 0; n < n_leafs; n++)
  {
    // fetch function values at corners of a cube
    for (int d = 0; d < N_CHILDREN; d++)
      f_cube[d] = f_values[leaf_to_node[n*N_CHILDREN + d]];
    switch (type) {
    case 0: result += cubes_mls[n].integrate_over_domain            (f_cube); break;
    case 1: result += cubes_mls[n].integrate_over_interface         (f_cube, num0); break;
    case 2: result += cubes_mls[n].integrate_over_intersection      (f_cube, num0, num1); break;
    case 3: result += cubes_mls[n].integrate_over_colored_interface (f_cube, num0, num1); break;
    case 4: result += cubes_mls[n].integrate_in_non_cart_dir        (f_cube, num0); break;
    }
  }

  return result;
}

double cube2_refined_mls_t::perform(double f, int type, int num0, int num1)
{
  double result = 0;
  double f_cube[4] = {f,f,f,f};

  for (int n = 0; n < n_leafs; n++)
  {
    switch (type)
    {
    case 0: result += cubes_mls[n].integrate_over_domain            (f_cube); break;
    case 1: result += cubes_mls[n].integrate_over_interface         (f_cube, num0); break;
    case 2: result += cubes_mls[n].integrate_over_intersection      (f_cube, num0, num1); break;
    case 3: result += cubes_mls[n].integrate_over_colored_interface (f_cube, num0, num1); break;
    case 4: result += cubes_mls[n].integrate_in_non_cart_dir        (f_cube, num0); break;
    }
  }

  return result;
}

double cube2_refined_mls_t::integrate_in_dir(std::vector<double> &f, int dir)
{
  std::vector<double> f_values(n_nodes,1);
  sample_all(f, f_values);

  double result = 0;
  double f_cube[4];

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

double cube2_refined_mls_t::integrate_in_dir(CF_2 &f, int dir)
{
  std::vector<double> f_values(n_nodes,1);
  sample_all(f, f_values);

  double result = 0;
  double f_cube[4];

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

double cube2_refined_mls_t::integrate_in_dir(double f, int dir)
{
  double result = 0;
  double f_cube[4] = {f,f,f,f};

  for (int n = 0; n < n_leafs; n++)
    if (cubes[get_cube[n]].wall[dir])
    {
      result += cubes_mls[n].integrate_in_dir(f_cube, dir);
    }

  return result;
}

void cube2_refined_mls_t::split_edge(int n)
{
  // check if an edge has already been split
  if (edges[n].is_split) return;

  // mark that an edge is split
  edges[n].is_split = true;

  // fetch coordinates of nodes
  double x_0 = nodes[edges[n].v0].x;  double y_0 = nodes[edges[n].v0].y;
  double x_1 = nodes[edges[n].v1].x;  double y_1 = nodes[edges[n].v1].y;

#ifdef CASL_THROWS
  if (x_0 > x_1) throw std::invalid_argument("[CASL_ERROR]: ");
  if (y_0 > y_1) throw std::invalid_argument("[CASL_ERROR]: ");
#endif

  // create the splitting node
  nodes.push_back(node_t(0.5*(x_0+x_1), 0.5*(y_0+y_1)));
  edges[n].v01 = nodes.size()-1;

  // create child edges
  edges.push_back(edge_t(edges[n].v0,  edges[n].v01)); edges[n].e0 = edges.size()-1;
  edges.push_back(edge_t(edges[n].v01, edges[n].v1 )); edges[n].e1 = edges.size()-1;
}

void cube2_refined_mls_t::split_cube(int n)
{
  // check if a cube has already been split
  if (cubes[n].is_split) return;

  // mark that a cube is split
  cubes[n].is_split = true;

  // split edges
  split_edge(cubes[n].e_m0);
  split_edge(cubes[n].e_p0);
  split_edge(cubes[n].e_0m);
  split_edge(cubes[n].e_0p);

  // fetch child elements of edges
//  int v_mm = edges[cubes[n].e_0m].v0;
//  int v_pm = edges[cubes[n].e_0m].v1;
//  int v_mp = edges[cubes[n].e_0p].v0;
//  int v_pp = edges[cubes[n].e_0p].v1;

  int v_m0 = edges[cubes[n].e_m0].v01;
  int v_p0 = edges[cubes[n].e_p0].v01;
  int v_0m = edges[cubes[n].e_0m].v01;
  int v_0p = edges[cubes[n].e_0p].v01;

  int e_m0_0m = edges[cubes[n].e_m0].e0;
  int e_m0_0p = edges[cubes[n].e_m0].e1;

  int e_p0_0m = edges[cubes[n].e_p0].e0;
  int e_p0_0p = edges[cubes[n].e_p0].e1;

  int e_0m_m0 = edges[cubes[n].e_0m].e0;
  int e_0m_p0 = edges[cubes[n].e_0m].e1;

  int e_0p_m0 = edges[cubes[n].e_0p].e0;
  int e_0p_p0 = edges[cubes[n].e_0p].e1;

  node_t *n_m0 = &nodes[v_m0];
  node_t *n_p0 = &nodes[v_p0];
  node_t *n_0m = &nodes[v_0m];
  node_t *n_0p = &nodes[v_0p];

  edge_t *E_m0 = &edges[cubes[n].e_m0];
  edge_t *E_p0 = &edges[cubes[n].e_p0];
  edge_t *E_0m = &edges[cubes[n].e_0m];
  edge_t *E_0p = &edges[cubes[n].e_0p];

  // create central node
  nodes.push_back(node_t(nodes[v_0p].x, nodes[v_p0].y)); int v_00 = nodes.size()-1;

  // create internal child edges
  edges.push_back(edge_t(v_m0,v_00)); int e_00_m0 = edges.size()-1;
  edges.push_back(edge_t(v_00,v_p0)); int e_00_p0 = edges.size()-1;
  edges.push_back(edge_t(v_0m,v_00)); int e_00_0m = edges.size()-1;
  edges.push_back(edge_t(v_00,v_0p)); int e_00_0p = edges.size()-1;

  bool wall[4] = {cubes[n].wall[0], cubes[n].wall[1], cubes[n].wall[2], cubes[n].wall[3]};

  // create new cubes
  cubes.push_back(cube_t(e_m0_0m, e_00_0m, e_0m_m0, e_00_m0)); cubes.back().wall[0] = wall[0]; cubes.back().wall[2] = wall[2];
  cubes.push_back(cube_t(e_00_0m, e_p0_0m, e_0m_p0, e_00_p0)); cubes.back().wall[1] = wall[1]; cubes.back().wall[2] = wall[2];
  cubes.push_back(cube_t(e_m0_0p, e_00_0p, e_00_m0, e_0p_m0)); cubes.back().wall[0] = wall[0]; cubes.back().wall[3] = wall[3];
  cubes.push_back(cube_t(e_00_0p, e_p0_0p, e_00_p0, e_0p_p0)); cubes.back().wall[1] = wall[1]; cubes.back().wall[3] = wall[3];
}

bool cube2_refined_mls_t::need_split(int n)
{
  int v[N_CHILDREN];
  bool all_negative, all_positive;

  v[0] = edges[cubes[n].e_m0].v0;
  v[1] = edges[cubes[n].e_p0].v0;
  v[2] = edges[cubes[n].e_m0].v1;
  v[3] = edges[cubes[n].e_p0].v1;

  bool result = false;

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
        result = false;
      }
    }
    else if (all_negative)
    {
      if (action->at(i) == ADDITION)
      {
        result = false;
      }
    }
    else if (action->at(i) == INTERSECTION || action->at(i) == ADDITION)
    {
      result = true;
    }
  }

  return result;
}
