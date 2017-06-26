// System
#include <stdexcept>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <iterator>
#include <set>
#include <time.h>
#include <stdio.h>

// p4est Library
#ifdef P4_TO_P8
#include <p8est_bits.h>
#include <p8est_extended.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_integration_mls.h>
#include <src/simplex3_mls_vtk.h>
#include <src/simplex3_mls_quadratic_vtk.h>
#else
#include <p4est_bits.h>
#include <p4est_extended.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_integration_mls.h>
#include <src/simplex2_mls_vtk.h>
//#include <src/simplex2_mls_quadratic_vtk.h>
#endif

#include <tools/plotting.h>

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

#include "geometry_one_circle.h"
#include "geometry_two_circles_union.h"
#include "geometry_two_circles_intersection.h"
#include "geometry_rose.h"
#include "geometry_two_circles_coloration.h"
#include "geometry_four_flowers.h"

//#include "geometry_four_flowers.cpp"

#undef MIN
#undef MAX

using namespace std;

/* grid and discretization */
#ifdef P4_TO_P8
int lmin = 4;
int lmax = 4;
int nb_splits = 5;
#else
int lmin = 5;
int lmax = 5;
int nb_splits = 10;
#endif

bool reinitialize_level_set = 0;

const int n_xyz[] = {1, 1, 1};
const int periodic[] = {0, 0, 0};

const double p_xyz_min[] = {-1, -1, -1};
const double p_xyz_max[] = { 1,  1,  1};

bool save_vtk = false;

// function to integrate
int func_num = 0;

#ifdef P4_TO_P8
class func_t: public CF_3
{
public:
  double operator()(double x, double y, double z) const
  {
    switch(func_num) {
      case 0: return 1;
      case 1: return x*x+y*y+z*z;
      case 2: return sin(x)*cos(y)*exp(z);
      case 3: return log(3.0+x*x-z*z)*(y-x);
    }
  }
} func;
#else
class func_t: public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch(func_num) {
      case 0: return 1;
      case 1: return x*x+y*y;
      case 2: return sin(x)*cos(y);
      case 3: return log(3.0+x*x-y*y)*(y-x);
    }
  }
} func;
#endif

/* Geometry:
 * 0 - union of two circles
 * 1 - intersection of two circles
 * 2 - coloration with two circles
 * 3 - four flower-shaped domains
 * 4 - rose-like domain
 * 5 - one circle
 */
int geometry_num = 5;

geometry_two_circles_union_t        geometry_two_circles_union;
geometry_two_circles_intersection_t geometry_two_circles_intersection;
geometry_two_circles_coloration_t   geometry_two_circles_coloration;
geometry_four_flowers_t             geometry_four_flowers;
geometry_rose_t                     geometry_rose;
geometry_one_circle_t               geometry_one_circle;


int num_of_domains;
#ifdef P4_TO_P8
vector<CF_3 *> *LSF;
#else
vector<CF_2 *> *LSF;
#endif
vector<action_t> *action;
vector<int> *color;


// exact values
class exact_t {
public:
  double ID;
  double IB;
  vector<double> ISB;
  vector<double> IX;
  vector<double> IX3;

  double n_subs;

  double n_Xs;
  vector<int> IXc0, IXc1;

  double n_X3s;
  vector<int> IX3c0, IX3c1, IX3c2;

  bool provided;
} exact;

// structure to keep results of integration
class result_t
{
public:
  vector<double> ID, IB;
  vector< vector<double> > ISB, IX, IX3;
} result, result_quadratic;

void set_parameters()
{
  // set geometry
  switch (geometry_num)
  {
    case 0:
      {
        LSF     = &geometry_two_circles_union.LSF;
        action  = &geometry_two_circles_union.action;
        color   = &geometry_two_circles_union.color;

        num_of_domains  = geometry_two_circles_union.n_subs;
        exact.n_subs    = geometry_two_circles_union.n_subs;
        exact.n_Xs      = geometry_two_circles_union.n_Xs;
        exact.IXc0      = geometry_two_circles_union.IXc0;
        exact.IXc1      = geometry_two_circles_union.IXc1;

        exact.n_X3s      = geometry_two_circles_union.n_X3s;
        exact.IX3c0      = geometry_two_circles_union.IX3c0;
        exact.IX3c1      = geometry_two_circles_union.IX3c1;
        exact.IX3c1      = geometry_two_circles_union.IX3c2;
      } break;
    case 1:
      {
        LSF     = &geometry_two_circles_intersection.LSF;
        action  = &geometry_two_circles_intersection.action;
        color   = &geometry_two_circles_intersection.color;

        num_of_domains  = geometry_two_circles_intersection.n_subs;
        exact.n_subs    = geometry_two_circles_intersection.n_subs;
        exact.n_Xs      = geometry_two_circles_intersection.n_Xs;
        exact.IXc0      = geometry_two_circles_intersection.IXc0;
        exact.IXc1      = geometry_two_circles_intersection.IXc1;

        exact.n_X3s      = geometry_two_circles_coloration.n_X3s;
        exact.IX3c0      = geometry_two_circles_coloration.IX3c0;
        exact.IX3c1      = geometry_two_circles_coloration.IX3c1;
        exact.IX3c1      = geometry_two_circles_coloration.IX3c2;
      } break;
    case 2:
      {
        LSF     = &geometry_two_circles_coloration.LSF;
        action  = &geometry_two_circles_coloration.action;
        color   = &geometry_two_circles_coloration.color;

        num_of_domains  = geometry_two_circles_coloration.n_subs;
        exact.n_subs    = geometry_two_circles_coloration.n_subs;
        exact.n_Xs      = geometry_two_circles_coloration.n_Xs;
        exact.IXc0      = geometry_two_circles_coloration.IXc0;
        exact.IXc1      = geometry_two_circles_coloration.IXc1;

        exact.n_X3s      = geometry_two_circles_coloration.n_X3s;
        exact.IX3c0      = geometry_two_circles_coloration.IX3c0;
        exact.IX3c1      = geometry_two_circles_coloration.IX3c1;
        exact.IX3c1      = geometry_two_circles_coloration.IX3c2;
      } break;
    case 3:
      {
        LSF     = &geometry_four_flowers.LSF;
        action  = &geometry_four_flowers.action;
        color   = &geometry_four_flowers.color;

        num_of_domains  = geometry_four_flowers.n_subs;
        exact.n_subs    = geometry_four_flowers.n_subs;
        exact.n_Xs      = geometry_four_flowers.n_Xs;
        exact.IXc0      = geometry_four_flowers.IXc0;
        exact.IXc1      = geometry_four_flowers.IXc1;

        exact.n_X3s      = geometry_four_flowers.n_X3s;
        exact.IX3c0      = geometry_four_flowers.IX3c0;
        exact.IX3c1      = geometry_four_flowers.IX3c1;
        exact.IX3c2      = geometry_four_flowers.IX3c2;
      } break;
    case 4:
      {
        LSF     = &geometry_rose.LSF;
        action  = &geometry_rose.action;
        color   = &geometry_rose.color;

        num_of_domains  = geometry_rose.n_subs;
        exact.n_subs    = geometry_rose.n_subs;
        exact.n_Xs      = geometry_rose.n_Xs;
        exact.IXc0      = geometry_rose.IXc0;
        exact.IXc1      = geometry_rose.IXc1;

        exact.n_X3s      = geometry_rose.n_X3s;
        exact.IX3c0      = geometry_rose.IX3c0;
        exact.IX3c1      = geometry_rose.IX3c1;
        exact.IX3c2      = geometry_rose.IX3c2;
      } break;
    case 5:
      {
        LSF     = &geometry_one_circle.LSF;
        action  = &geometry_one_circle.action;
        color   = &geometry_one_circle.color;

        num_of_domains  = geometry_one_circle.n_subs;
        exact.n_subs    = geometry_one_circle.n_subs;
        exact.n_Xs      = geometry_one_circle.n_Xs;
        exact.IXc0      = geometry_one_circle.IXc0;
        exact.IXc1      = geometry_one_circle.IXc1;

        exact.n_X3s      = geometry_one_circle.n_X3s;
        exact.IX3c0      = geometry_one_circle.IX3c0;
        exact.IX3c1      = geometry_one_circle.IX3c1;
        exact.IX3c2      = geometry_one_circle.IX3c2;
      } break;
  }

  exact.provided = false;

  // get exact values if available
  if (func_num == 0)
    switch (geometry_num)
    {
      case 0: {
          exact.ID  = geometry_two_circles_union.exact0.ID;
          exact.ISB = geometry_two_circles_union.exact0.ISB;
          exact.IX  = geometry_two_circles_union.exact0.IX;
          exact.provided = true;
        } break;
      case 1: {
          exact.ID  = geometry_two_circles_intersection.exact0.ID;
          exact.ISB = geometry_two_circles_intersection.exact0.ISB;
          exact.IX  = geometry_two_circles_intersection.exact0.IX;
          exact.provided = true;
        } break;
      case 2: {
          exact.ID  = geometry_two_circles_coloration.exact0.ID;
          exact.ISB = geometry_two_circles_coloration.exact0.ISB;
          exact.IX  = geometry_two_circles_coloration.exact0.IX;
          exact.provided = true;
        } break;
      case 5: {
          exact.ID  = geometry_one_circle.exact0.ID;
          exact.ISB = geometry_one_circle.exact0.ISB;
          exact.IX  = geometry_one_circle.exact0.IX;
          exact.provided = true;
        } break;
    }
  else if (func_num == 1)
    switch (geometry_num)
    {
      case 0: {
          exact.ID  = geometry_two_circles_union.exact1.ID;
          exact.ISB = geometry_two_circles_union.exact1.ISB;
          exact.IX  = geometry_two_circles_union.exact1.IX;
          exact.provided = true;
        } break;
      case 1: {
          exact.ID  = geometry_two_circles_intersection.exact1.ID;
          exact.ISB = geometry_two_circles_intersection.exact1.ISB;
          exact.IX  = geometry_two_circles_intersection.exact1.IX;
          exact.provided = true;
        } break;
      case 2: {
          exact.ID  = geometry_two_circles_coloration.exact1.ID;
          exact.ISB = geometry_two_circles_coloration.exact1.ISB;
          exact.IX  = geometry_two_circles_coloration.exact1.IX;
          exact.provided = true;
        } break;
      case 5: {
          exact.ID  = geometry_one_circle.exact1.ID;
          exact.ISB = geometry_one_circle.exact1.ISB;
          exact.IX  = geometry_one_circle.exact1.IX;
          exact.provided = true;
        } break;
    }

  // prepare arrays for numerical results
  result.ISB.clear();
  for (int i = 0; i < exact.n_subs; i++)
  {
    result.ISB.push_back(vector<double>());
  }

  result.IX.clear();
  for (int i = 0; i < exact.n_Xs; i++)
  {
    result.IX.push_back(vector<double>());
  }

  result.IX3.clear();
  for (int i = 0; i < exact.n_X3s; i++)
  {
    result.IX3.push_back(vector<double>());
  }

  // prepare arrays for numerical results
  result_quadratic.ISB.clear();
  for (int i = 0; i < exact.n_subs; i++)
  {
    result_quadratic.ISB.push_back(vector<double>());
  }

  result_quadratic.IX.clear();
  for (int i = 0; i < exact.n_Xs; i++)
  {
    result_quadratic.IX.push_back(vector<double>());
  }

  result_quadratic.IX3.clear();
  for (int i = 0; i < exact.n_X3s; i++)
  {
    result_quadratic.IX3.push_back(vector<double>());
  }
}

#ifdef P4_TO_P8
class level_set_tot_t : public CF_3
#else
class level_set_tot_t : public CF_2
#endif
{
#ifdef P4_TO_P8
  std::vector<CF_3 *>   *phi_cf;
#else
  std::vector<CF_2 *>   *phi_cf;
#endif
  std::vector<action_t> *action;
  std::vector<int>      *color;

public:

#ifdef P4_TO_P8
  level_set_tot_t(std::vector<CF_3 *> *phi_cf, std::vector<action_t> *action, std::vector<int> *color) :
#else
  level_set_tot_t(std::vector<CF_2 *> *phi_cf, std::vector<action_t> *action, std::vector<int> *color) :
#endif
    phi_cf(phi_cf), action(action), color(color) {}

#ifdef P4_TO_P8
  double operator()(double x, double y, double z) const
#else
  double operator()(double x, double y) const
#endif
  {
    double phi_total = -10;
    double phi_current = -10;
    for (short i = 0; i < color->size(); ++i)
    {
      if (action->at(i) == INTERSECTION)
      {
#ifdef P4_TO_P8
        phi_current = (*phi_cf->at(i))(x,y,z);
#else
        phi_current = (*phi_cf->at(i))(x,y);
#endif
        if (phi_current > phi_total) phi_total = phi_current;
      } else if (action->at(i) == ADDITION) {
#ifdef P4_TO_P8
        phi_current = (*phi_cf->at(i))(x,y,z);
#else
        phi_current = (*phi_cf->at(i))(x,y);
#endif
        if (phi_current < phi_total) phi_total = phi_current;
      }
    }
    return phi_total;
  }
};

vector<double> level, h;

// forward declaration
void save_VTK(p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes, my_p4est_brick_t *brick,
              std::vector<Vec> phi, Vec phi_tot,
              int compt);

int main (int argc, char* argv[])
{

  set_parameters();

  level_set_tot_t ls_tot(LSF, action, color);

  PetscErrorCode ierr;
  int mpiret;
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  parStopWatch w;
  w.start("total time");

  p4est_connectivity_t *connectivity;
  my_p4est_brick_t brick;

  connectivity = my_p4est_brick_new(n_xyz, p_xyz_min, p_xyz_max, &brick, periodic);

  p4est_t       *p4est;
  p4est_nodes_t *nodes;
  p4est_ghost_t *ghost;

  std::vector<double> phi_integr;
  std::vector<double> phi_integr_quadratic;
  for(int iter=0; iter<nb_splits; ++iter)
  {
    ierr = PetscPrintf(mpi.comm(), "Level %d / %d\n", lmin+iter, lmax+iter); CHKERRXX(ierr);
    p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

    splitting_criteria_cf_t data(0, lmax+iter, &ls_tot, 1.2);
//    splitting_criteria_cf_t data(lmin+iter, lmax+iter, &ls_tot, 1.2);
    p4est->user_pointer = (void*)(&data);

    my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);
    p4est_balance(p4est, P4EST_CONNECT_FULL, NULL);
    my_p4est_partition(p4est, P4EST_FALSE, NULL);

    ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
    my_p4est_ghost_expand(p4est, ghost);
    nodes = my_p4est_nodes_new(p4est, ghost);

    my_p4est_hierarchy_t hierarchy(p4est,ghost, &brick);
    my_p4est_node_neighbors_t ngbd_n(&hierarchy,nodes);

    /* function to integrate */
    Vec func_vec;
    ierr = VecCreateGhostNodes(p4est, nodes, &func_vec); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, func, func_vec);

    Vec fdd[P4EST_DIM];

    for (short dir = 0; dir < P4EST_DIM; ++dir)
    {
      ierr = VecCreateGhostNodes(p4est, nodes, &fdd[dir]); CHKERRXX(ierr);
    }

    ngbd_n.second_derivatives_central(func_vec, fdd);


    my_p4est_level_set_t ls(&ngbd_n);

    /* level-set functions */
    vector<Vec> phi_vec, phi_xx_vec, phi_yy_vec;
#ifdef P4_TO_P8
    vector<Vec> phi_zz_vec;
#endif

    for (int i = 0; i < num_of_domains; i++)
    {
      phi_vec.push_back(Vec());     ierr = VecCreateGhostNodes(p4est, nodes, &phi_vec[i]); CHKERRXX(ierr);
      phi_xx_vec.push_back(Vec());  ierr = VecCreateGhostNodes(p4est, nodes, &phi_xx_vec[i]); CHKERRXX(ierr);
      phi_yy_vec.push_back(Vec());  ierr = VecCreateGhostNodes(p4est, nodes, &phi_yy_vec[i]); CHKERRXX(ierr);
#ifdef P4_TO_P8
      phi_zz_vec.push_back(Vec());  ierr = VecCreateGhostNodes(p4est, nodes, &phi_zz_vec[i]); CHKERRXX(ierr);
#endif

      sample_cf_on_nodes(p4est, nodes, *LSF->at(i), phi_vec[i]);

      if (reinitialize_level_set)
        ls.reinitialize_1st_order_time_2nd_order_space(phi_vec.back());

#ifdef P4_TO_P8
      ngbd_n.second_derivatives_central(phi_vec[i], phi_xx_vec[i], phi_yy_vec[i], phi_zz_vec[i]);
#else
      ngbd_n.second_derivatives_central(phi_vec[i], phi_xx_vec[i], phi_yy_vec[i]);
#endif
    }

    Vec phi_tot;
    ierr = VecCreateGhostNodes(p4est, nodes, &phi_tot); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, ls_tot, phi_tot);

    my_p4est_integration_mls_t integration(p4est, nodes);

//    integration.set_phi(phi_vec, geometry.action, geometry.color);
//    integration.set_phi(geometry.LSF, geometry.action, geometry.color);
//#ifdef P4_TO_P8
//    integration.set_phi(phi_vec, phi_xx_vec, phi_yy_vec, phi_zz_vec, *action, *color);
//#else
//    integration.set_phi(phi_vec, phi_xx_vec, phi_yy_vec, *action, *color);
//#endif
#ifdef P4_TO_P8
    integration.set_phi(phi_vec, *action, *color);
#else
    integration.set_phi(phi_vec, *action, *color);
#endif
//    integration.set_use_cube_refined(0);

    my_p4est_integration_mls_t integration_quadratic(p4est, nodes);

#ifdef P4_TO_P8
    integration_quadratic.set_phi(phi_vec, phi_xx_vec, phi_yy_vec, phi_zz_vec, *action, *color);
#else
    integration_quadratic.set_phi(phi_vec, phi_xx_vec, phi_yy_vec, *action, *color);
#endif



    if (save_vtk)
    {
      integration.initialize();
#ifdef P4_TO_P8
      vector<simplex3_mls_t *> simplices;
      int n_sps = NTETS;
#else
      vector<simplex2_mls_t *> simplices;
      int n_sps = 2;
#endif

      for (int k = 0; k < integration.cubes_linear.size(); k++)
        if (integration.cubes_linear[k].loc == FCE)
          for (int l = 0; l < n_sps; l++)
            simplices.push_back(&integration.cubes_linear[k].simplex[l]);

#ifdef P4_TO_P8
      simplex3_mls_vtk::write_simplex_geometry(simplices, to_string(OUTPUT_DIR), to_string(iter));
#else
      simplex2_mls_vtk::write_simplex_geometry(simplices, to_string(OUTPUT_DIR), to_string(iter));
#endif
      save_VTK(p4est, ghost, nodes, &brick, phi_vec, phi_tot, iter);
    }

#ifdef P4_TO_P8
    if (save_vtk)
    {
      integration_quadratic.initialize();
#ifdef P4_TO_P8
      vector<simplex3_mls_quadratic_t *> simplices;
      int n_sps = NUM_TETS;
#else
      vector<simplex3_mls_quadratic_t *> simplices;
      int n_sps = 2;
#endif

      for (int k = 0; k < integration_quadratic.cubes_quadratic.size(); k++)
        if (integration_quadratic.cubes_quadratic[k].loc == FCE)
          for (int l = 0; l < n_sps; l++)
            simplices.push_back(&integration_quadratic.cubes_quadratic[k].simplex[l]);

#ifdef P4_TO_P8
      simplex3_mls_quadratic_vtk::write_simplex_geometry(simplices, to_string(OUTPUT_DIR), to_string(iter+1000));
#else
      simplex3_mls_quadratic_vtk::write_simplex_geometry(simplices, to_string(OUTPUT_DIR), to_string(iter+1000));
#endif
    }
#endif


    /* Calculate and store results */

#ifdef P4_TO_P8
    Vec phi_dd[P4EST_DIM] = { phi_xx_vec[0], phi_yy_vec[0], phi_zz_vec[0] };
#else
    Vec phi_dd[P4EST_DIM] = { phi_xx_vec[0], phi_yy_vec[0] };
#endif
//    phi_integr.push_back(integration_quadratic.integrate_over_interface(0, phi_vec[0], phi_dd)/integration_quadratic.integrate_over_interface(0, func_vec, fdd));
    phi_integr.push_back(integration_quadratic.integrate_over_interface(0, phi_vec[0], phi_dd));

    if (exact.provided || iter < nb_splits-1)
    {
      level.push_back(lmax+iter);
      h.push_back((p_xyz_max[0]-p_xyz_min[0])/pow(2.0,(double)(lmax+iter)));

      result.ID.push_back(integration.integrate_over_domain(func_vec));
      result_quadratic.ID.push_back(integration_quadratic.integrate_over_domain(func_vec, fdd));

      for (int i = 0; i < exact.n_subs; i++)
      {
        result.ISB[i].push_back(integration.integrate_over_interface(color->at(i), func_vec));
        result_quadratic.ISB[i].push_back(integration_quadratic.integrate_over_interface(color->at(i), func_vec, fdd));
      }

      for (int i = 0; i < exact.n_Xs; i++)
      {
#ifdef P4_TO_P8
        result.IX[i].push_back(integration.integrate_over_intersection(exact.IXc0[i], exact.IXc1[i], func_vec));
        result_quadratic.IX[i].push_back(integration_quadratic.integrate_over_intersection(exact.IXc0[i], exact.IXc1[i], func_vec, fdd));
#else
        result.IX[i].push_back(integration.integrate_over_intersection(exact.IXc0[i], exact.IXc1[i], func_vec));
        result_quadratic.IX[i].push_back(integration_quadratic.integrate_over_intersection(exact.IXc0[i], exact.IXc1[i], func_vec, fdd));
#endif
      }

#ifdef P4_TO_P8
      for (int i = 0; i < exact.n_X3s; i++)
//        double val = integration.integrate_over_intersection(func_vec, exact.IX3c0[i], exact.IX3c1[i], exact.IX3c2[i]);
        result.IX3[i].push_back(integration.integrate_over_intersection(exact.IX3c0[i], exact.IX3c1[i], exact.IX3c2[i], func_vec));
#endif
    }
    else if (iter == nb_splits-1)
    {
//      exact.ID  = (integration.integrate_over_domain(func_vec));
      exact.ID  = (integration_quadratic.integrate_over_domain(func_vec, fdd));

      for (int i = 0; i < exact.n_subs; i++)
//        exact.ISB.push_back(integration.integrate_over_interface(color->at(i), func_vec));
        exact.ISB.push_back(integration_quadratic.integrate_over_interface(color->at(i), func_vec, fdd));

      for (int i = 0; i < exact.n_Xs; i++)
#ifdef P4_TO_P8
        exact.IX.push_back(integration.integrate_over_intersection(exact.IXc0[i], exact.IXc1[i], func_vec));
#else
//        exact.IX.push_back(integration.integrate_over_intersection(exact.IXc0[i], exact.IXc1[i], func_vec));
        exact.IX.push_back(integration_quadratic.integrate_over_intersection(exact.IXc0[i], exact.IXc1[i], func_vec, fdd));
#endif

#ifdef P4_TO_P8
      for (int i = 0; i < exact.n_X3s; i++)
        exact.IX3.push_back(integration.integrate_over_intersection(exact.IX3c0[i], exact.IX3c1[i], exact.IX3c2[i], func_vec));
#endif
    }

    ierr = VecDestroy(func_vec); CHKERRXX(ierr);
    ierr = VecDestroy(phi_tot); CHKERRXX(ierr);

    for (short dir = 0; dir < P4EST_DIM; ++dir)
    {
      ierr = VecDestroy(fdd[dir]); CHKERRXX(ierr);
    }

    for (int i = 0; i < phi_vec.size(); i++)
    {
      ierr = VecDestroy(phi_vec[i]); CHKERRXX(ierr);
      ierr = VecDestroy(phi_xx_vec[i]); CHKERRXX(ierr);
      ierr = VecDestroy(phi_yy_vec[i]); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecDestroy(phi_zz_vec[i]); CHKERRXX(ierr);
#endif
    }
    phi_vec.clear();

    p4est_nodes_destroy(nodes);
    p4est_ghost_destroy(ghost);
    p4est_destroy      (p4est);
  }

  // make a plot
  int plot_color = 1;
  if (mpi.rank() == 0)
  {
    // plot convergence results for a quick check
    Gnuplot plot;
    print_Table("Convergence", exact.ID, level, h, "Domain", result.ID, 1, &plot);
    plot_color++;

    for (int i = 0; i < exact.n_subs; i++)
    {
      print_Table("Convergence", exact.ISB[i], level, h, "Sub-boundary #"+to_string(i), result.ISB[i], plot_color, &plot);
      plot_color++;
    }

    for (int i = 0; i < exact.n_Xs; i++)
    {
      print_Table("Convergence", exact.IX[i], level, h, "X of #"+to_string(exact.IXc0[i])+" and #"+to_string(exact.IXc1[i]), result.IX[i], plot_color, &plot);
      plot_color++;
    }


#ifdef P4_TO_P8
    for (int i = 0; i < exact.n_X3s; i++)
    {
      print_Table("Convergence", exact.IX3[i], level, h, "X of #"+to_string(exact.IX3c0[i])+" and #"+to_string(exact.IX3c1[i])+" and #"+to_string(exact.IX3c2[i]), result.IX3[i], plot_color, &plot);
      plot_color++;
    }
#endif

//    plot_color = 1;
    print_Table("Convergence", exact.ID, level, h, "Domain", result_quadratic.ID, plot_color, &plot);
    plot_color++;

    for (int i = 0; i < exact.n_subs; i++)
    {
      print_Table("Convergence", exact.ISB[i], level, h, "Sub-boundary #"+to_string(i), result_quadratic.ISB[i], plot_color, &plot);
      plot_color++;
    }

//    for (int i = 0; i < result_quadratic.ISB[0].size(); ++i)
//      result_quadratic.ISB[0][i] += result_quadratic.ISB[1][i];

//    print_Table("Convergence", exact.ISB[0]+exact.ISB[1], level, h, "Total Boundary", result_quadratic.ISB[0], plot_color, &plot);
//    plot_color++;

    for (int i = 0; i < exact.n_Xs; i++)
    {
      print_Table("Convergence", exact.IX[i], level, h, "X of #"+to_string(exact.IXc0[i])+" and #"+to_string(exact.IXc1[i]), result_quadratic.IX[i], plot_color, &plot);
      plot_color++;
    }


#ifdef P4_TO_P8
    for (int i = 0; i < exact.n_X3s; i++)
    {
      print_Table("Convergence", exact.IX3[i], level, h, "X of #"+to_string(exact.IX3c0[i])+" and #"+to_string(exact.IX3c1[i])+" and #"+to_string(exact.IX3c2[i]), result_quadratic.IX3[i], plot_color, &plot);
      plot_color++;
    }
#endif

    print_Table("Convergence", 0, level, h, "phi_integr", phi_integr, plot_color, &plot);

    // print all errors in compact form for plotting in matlab
    // step sizes
    for (int i = 0; i < h.size(); i++)
    {
      if (i != 0) cout << ", ";
      cout << h[i];
    }
    cout <<  ";" << endl;

    // domain
    for (int i = 0; i < h.size(); i++)
    {
      if (i != 0) cout << ", ";
      cout << fabs(result.ID[i]-exact.ID);
    }
    cout <<  ";" << endl;

    // sub-boundaries
    for (int j = 0; j < exact.n_subs; j++)
    {
      for (int i = 0; i < h.size(); i++)
      {
        if (i != 0) cout << ", ";
        cout << fabs(result.ISB[j][i]-exact.ISB[j]);
      }
      cout <<  ";" << endl;
    }

    // X of 2 sub-boundaries
    for (int j = 0; j < exact.n_Xs; j++)
    {
      for (int i = 0; i < h.size(); i++)
      {
        if (i != 0) cout << ", ";
        cout << fabs(result.IX[j][i]-exact.IX[j]);
      }
      cout <<  ";" << endl;
    }

#ifdef P4_TO_P8
    // X of 3 sub-boundaries
    for (int j = 0; j < exact.n_X3s; j++)
    {
      for (int i = 0; i < h.size(); i++)
      {
        if (i != 0) cout << ", ";
        cout << fabs(result.IX3[j][i]-exact.IX3[j]);
      }
      cout <<  ";" << endl;
    }
#endif

//    for (int i = 0; i < h.size(); i++)
//    {
//      cout << h[i] << ", "
//           << fabs(result.ID[i]-exact.ID);

//      for (int j = 0; j < exact.n_subs; j++)
//        cout << ", " << fabs(result.ISB[j][i]-exact.ISB[j]);

//      for (int j = 0; j < exact.n_Xs; j++)
//        cout << ", " << fabs(result.IX[j][i]-exact.IX[j]);

//      cout <<  ";" << endl;
//    }
    std::cin.get();
  }

  my_p4est_brick_destroy(connectivity, &brick);

  w.stop(); w.read_duration();

  return 0;
}



void save_VTK(p4est_t *p4est, p4est_ghost_t *ghost, p4est_nodes_t *nodes, my_p4est_brick_t *brick,
              std::vector<Vec> phi, Vec phi_tot,
              int compt)
{
  PetscErrorCode ierr;
#ifdef STAMPEDE
  char *out_dir;
  out_dir = getenv("OUT_DIR");
#else
  char out_dir[10000];
  sprintf(out_dir, OUTPUT_DIR);
#endif

  std::ostringstream oss;

  oss << out_dir
      << "/vtu/nodes_"
      << p4est->mpisize << "_"
      << brick->nxyztrees[0] << "x"
      << brick->nxyztrees[1] <<
       #ifdef P4_TO_P8
         "x" << brick->nxyztrees[2] <<
       #endif
         "." << compt;

  double *phi_p;
  std::vector<double *> point_data(phi.size(), NULL);
  std::vector<std::string> point_data_names;

  for (int i = 0; i < phi.size(); ++i)
  {
    ierr = VecGetArray(phi[i], &point_data[i]); CHKERRXX(ierr);
    point_data_names.push_back("phi"+to_string(i));
  }

  point_data.push_back(NULL);
  ierr = VecGetArray(phi_tot, &point_data.back()); CHKERRXX(ierr);
  point_data_names.push_back("phi_tot");

  /* save the size of the leaves */
  Vec leaf_level;
  ierr = VecCreateGhostCells(p4est, ghost, &leaf_level); CHKERRXX(ierr);
  double *l_p;
  ierr = VecGetArray(leaf_level, &l_p); CHKERRXX(ierr);

  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for( size_t q=0; q<tree->quadrants.elem_count; ++q)
    {
      const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
      l_p[tree->quadrants_offset+q] = quad->level;
    }
  }

  for(size_t q=0; q<ghost->ghosts.elem_count; ++q)
  {
    const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, q);
    l_p[p4est->local_num_quadrants+q] = quad->level;
  }

  std::vector<double *> cell_data;
  std::vector<std::string> cell_data_names;

  cell_data.push_back(l_p);
  cell_data_names.push_back("leaf_level");

  my_p4est_vtk_write_all_vector_form(p4est, nodes, ghost,
                                     P4EST_TRUE, P4EST_TRUE,
                                     oss.str().c_str(),
                                     point_data, point_data_names,
                                     cell_data,  cell_data_names);

//  my_p4est_vtk_write_all(p4est, nodes, ghost,
//                         P4EST_TRUE, P4EST_TRUE,
//                         1, 1, oss.str().c_str(),
//                         VTK_POINT_DATA, "phi", phi_p,
//                         VTK_CELL_DATA , "leaf_level", l_p);

  ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
  ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

  for (int i = 0; i < phi.size(); ++i)
  {
    ierr = VecRestoreArray(phi[i], &point_data[i]); CHKERRXX(ierr);
  }

  ierr = VecGetArray(phi_tot, &point_data.back()); CHKERRXX(ierr);

  PetscPrintf(p4est->mpicomm, "VTK saved in %s\n", oss.str().c_str());
}
