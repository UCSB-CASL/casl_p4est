
/*
 * Test the cell based multi level-set p4est.
 * Intersection of two circles
 *
 * run the program with the -help flag to see the available options
 */

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
#include <src/my_p8est_shapes.h>
#include <src/my_p8est_tools_mls.h>
#include <src/my_p8est_macros.h>
#include <src/my_p8est_semi_lagrangian.h>
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
#include <src/my_p4est_shapes.h>
#include <src/my_p4est_tools_mls.h>
#include <src/my_p4est_macros.h>
#include <src/my_p4est_semi_lagrangian.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/Parser.h>

#undef MIN
#undef MAX

#define CMD_OPTIONS for (short option_action = 0; option_action < 2; ++option_action)
#define ADD_OPTION(cmd, var, description) option_action == 0 ? cmd.add_option(#var, description) : (void) (var = cmd.get(#var, var));
#define PARSE_OPTIONS(cmd, argc, argv) if (option_action == 0) cmd.parse(argc, argv);
using namespace std;

// comptational domain

double xmin = 0;
double ymin = 0;
double zmin = 0;

double xmax = 1;
double ymax = 1;
double zmax = 1;

bool px = 0;
bool py = 0;
bool pz = 0;

int nx = 1;
int ny = 1;
int nz = 1;

// grid parameters
#ifdef P4_TO_P8
int lmin = 4;
int lmax = 4;
#else
int lmin = 1;
int lmax = 7;
#endif
double lip = 1.5;

// output parameters
bool save_vtk = 1;

int num_geometry = 0;
int num_contact_angle = 0;

int num_surfaces = 2;

int max_iterations = 1000;

/* geometry of interfaces */
class phi_intf_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (num_geometry)
    {
      case 0: return sqrt( SQR(x-.5) + SQR(y-.5) ) - 0.25;
      default: throw std::invalid_argument("Error: Invalid geometry number\n");
    }
  }
} phi_intf_cf;

class phi_wall_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (num_geometry)
    {
      case 0: return ((x-.5)*1. - (y-.3)*2.)/sqrt(SQR(1.) + SQR(2.));
      default: throw std::invalid_argument("Error: Invalid geometry number\n");
    }
  }
} phi_wall_cf;

class phi_eff_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
//    return MIN(fabs(phi_intf_cf(x,y)), fabs(phi_wall_cf(x,y)));
    return MAX((phi_intf_cf(x,y)), (phi_wall_cf(x,y)));
  }
} phi_eff_cf;

// contact angle
class cos_angle_cf_t : public CF_2
{
public:
  double operator()(double x, double y) const
  {
    switch (num_geometry)
    {
      case 0: return 0.99;
      default: throw std::invalid_argument("Error: Invalid geometry number\n");
    }
  }
} cos_angle_cf;

int main (int argc, char* argv[])
{
  PetscErrorCode ierr;
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  // create an output directory
  const char* out_dir = getenv("OUT_DIR");
  if (!out_dir)
  {
    ierr = PetscPrintf(mpi.comm(), "You need to set the environment variable OUT_DIR to save visuals\n");
    return -1;
  }

  std::ostringstream command;

  command << "mkdir -p " << out_dir << "/vtu";
  int ret_sys = system(command.str().c_str());
  if(ret_sys < 0)
    throw std::invalid_argument("Could not create directory");

  cmdParser cmd;
  CMD_OPTIONS
  {
    ADD_OPTION(cmd, nx, "number of trees in x-dimension");
    ADD_OPTION(cmd, ny, "number of trees in y-dimension");
#ifdef P4_TO_P8
    ADD_OPTION(cmd, nz, "number of trees in z-dimension");
#endif

    ADD_OPTION(cmd, px, "periodicity in x-dimension 0/1");
    ADD_OPTION(cmd, py, "periodicity in y-dimension 0/1");
#ifdef P4_TO_P8
    ADD_OPTION(cmd, pz, "periodicity in z-dimension 0/1");
#endif

    ADD_OPTION(cmd, xmin, "xmin"); ADD_OPTION(cmd, xmax, "xmax");
    ADD_OPTION(cmd, ymin, "ymin"); ADD_OPTION(cmd, ymax, "ymax");
#ifdef P4_TO_P8
    ADD_OPTION(cmd, zmin, "zmin"); ADD_OPTION(cmd, zmax, "zmax");
#endif

    ADD_OPTION(cmd, lmin, "min level of trees");
    ADD_OPTION(cmd, lmax, "max level of trees");
    ADD_OPTION(cmd, lip,  "Lipschitz constant");

    ADD_OPTION(cmd, save_vtk,  "save_vtk");

    PARSE_OPTIONS(cmd, argc, argv);
  }

#ifdef P4_TO_P8
  double xyz_min[] = { xmin, ymin, zmin };
  double xyz_max[] = { xmax, ymax, zmax };
  int nb_trees[] = { nx, ny, nz };
  int periodic[] = { px, py, pz };
#else
  double xyz_min[] = { xmin, ymin };
  double xyz_max[] = { xmax, ymax };
  int nb_trees[] = { nx, ny };
  int periodic[] = { px, py };
#endif

  /* create the p4est */
  my_p4est_brick_t brick;

  p4est_connectivity_t *connectivity = my_p4est_brick_new(nb_trees, xyz_min, xyz_max, &brick, periodic);
  p4est_t *p4est = my_p4est_new(mpi.comm(), connectivity, 0, NULL, NULL);

  splitting_criteria_cf_t data(lmin, lmax, &phi_eff_cf, lip);

  p4est->user_pointer = (void*)(&data);
  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);
  my_p4est_partition(p4est, P4EST_FALSE, NULL);

  p4est_ghost_t *ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes = my_p4est_nodes_new(p4est, ghost);

  my_p4est_hierarchy_t *hierarchy = new my_p4est_hierarchy_t(p4est,ghost, &brick);
  my_p4est_node_neighbors_t *ngbd = new my_p4est_node_neighbors_t(hierarchy,nodes);
  ngbd->init_neighbors();

  p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  double xmin_tree = p4est->connectivity->vertices[3*vm + 0];
  double ymin_tree = p4est->connectivity->vertices[3*vm + 1];
  double xmax_tree = p4est->connectivity->vertices[3*vp + 0];
  double ymax_tree = p4est->connectivity->vertices[3*vp + 1];
  double dx = (xmax_tree-xmin_tree) / pow(2., (double) data.max_lvl);
  double dy = (ymax_tree-ymin_tree) / pow(2., (double) data.max_lvl);
#ifdef P4_TO_P8
  double zmin_tree = p4est->connectivity->vertices[3*vm + 2];
  double zmax_tree = p4est->connectivity->vertices[3*vp + 2];
  double dz = (zmax_tree-zmin_tree) / pow(2.,(double) data.max_lvl);
#endif

  Vec phi_intf; ierr = VecCreateGhostNodes(p4est, nodes, &phi_intf); CHKERRXX(ierr);
  Vec phi_wall; ierr = VecCreateGhostNodes(p4est, nodes, &phi_wall); CHKERRXX(ierr);

  sample_cf_on_nodes(p4est, nodes, phi_intf_cf, phi_intf);
  sample_cf_on_nodes(p4est, nodes, phi_wall_cf, phi_wall);

  Vec phi_extd;

  ierr = VecDuplicate(phi_intf, &phi_extd);

  copy_ghosted_vec(phi_intf, phi_extd);

  int iteration = 0;

  double dt = 0.0025*dx;

  double volume_beg = 0;

  {
    std::vector<Vec> phi(2);
    std::vector<action_t> acn(2, INTERSECTION);
    std::vector<int> clr(2);

    phi[0] = phi_extd; clr[0] = 0;
    phi[1] = phi_wall; clr[1] = 1;

    my_p4est_integration_mls_t integration(p4est, nodes);
    integration.set_phi(phi, acn, clr);

    volume_beg = integration.measure_of_domain();
  }


  while (iteration < max_iterations)
  {
    Vec cos_angle; ierr = VecCreateGhostNodes(p4est, nodes, &cos_angle); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est, nodes, cos_angle_cf, cos_angle);

    std::vector<Vec> phi(2);
    std::vector<action_t> acn(2, INTERSECTION);
    std::vector<int> clr(2);

    phi[0] = phi_extd; clr[0] = 0;
    phi[1] = phi_wall; clr[1] = 1;

    my_p4est_integration_mls_t integration(p4est, nodes);
    integration.set_phi(phi, acn, clr);

    double volume_cur = integration.measure_of_domain();

    PetscPrintf(mpi.comm(), "Volume change: %f\n", (volume_cur-volume_beg)/volume_beg);

    /* normal and curvature */
    Vec normal[P4EST_DIM];
    Vec kappa;

    foreach_dimension(dim)
    {
      ierr = VecCreateGhostNodes(p4est, nodes, &normal[dim]); CHKERRXX(ierr);
    }

    ierr = VecCreateGhostNodes(p4est, nodes, &kappa); CHKERRXX(ierr);

    compute_normals_and_mean_curvature(*ngbd, phi_extd, normal, kappa);

    Vec kappa_tmp;

    ierr = VecDuplicate(kappa, &kappa_tmp); CHKERRXX(ierr);

    my_p4est_level_set_t ls(ngbd);
    ls.set_interpolation_on_interface(quadratic_non_oscillatory_continuous_v2);

    ls.extend_from_interface_to_whole_domain_TVD(phi_extd, kappa, kappa_tmp);

    ierr = VecDestroy(kappa); CHKERRXX(ierr);

    kappa = kappa_tmp;

    double kappa_avg = integration.integrate_over_interface(0, kappa)/integration.measure_of_interface(0);

    shift_ghosted_vec(kappa, -kappa_avg);

    if (save_vtk)
    {
      std::ostringstream oss;

      oss << out_dir
          << "/vtu/nodes_"
          << mpi.size() << "_"
          << brick.nxyztrees[0] << "x"
          << brick.nxyztrees[1] <<
       #ifdef P4_TO_P8
             "x" << brick.nxyztrees[2] <<
       #endif
             "." << iteration;

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

      double *phi_intf_ptr;
      double *phi_wall_ptr;
      double *phi_extd_ptr;
      double *kappa_ptr;

      ierr = VecGetArray(phi_intf, &phi_intf_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(phi_wall, &phi_wall_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(phi_extd, &phi_extd_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(kappa, &kappa_ptr); CHKERRXX(ierr);

      my_p4est_vtk_write_all(p4est, nodes, ghost,
                             P4EST_TRUE, P4EST_TRUE,
                             4, 1, oss.str().c_str(),
                             VTK_POINT_DATA, "phi_intf", phi_intf_ptr,
                             VTK_POINT_DATA, "phi_wall", phi_wall_ptr,
                             VTK_POINT_DATA, "phi_extd", phi_extd_ptr,
                             VTK_POINT_DATA, "kappa", kappa_ptr,
                             VTK_CELL_DATA , "leaf_level", l_p);

      ierr = VecRestoreArray(phi_intf, &phi_intf_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_wall, &phi_wall_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_extd, &phi_extd_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(kappa, &kappa_ptr); CHKERRXX(ierr);

      ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
      ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

      PetscPrintf(mpi.comm(), "VTK saved in %s\n", oss.str().c_str());
    }

    Vec velo[P4EST_DIM];

    double *normal_ptr;
    double *kappa_ptr;
    double *velo_ptr;

    foreach_dimension(dim)
    {
      ierr = VecDuplicate(normal[dim], &velo[dim]); CHKERRXX(ierr);

      ierr = VecGetArray(normal[dim], &normal_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(velo[dim], &velo_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(kappa, &kappa_ptr); CHKERRXX(ierr);

      foreach_node(n, nodes)
      {
        velo_ptr[n] = -kappa_ptr[n]*normal_ptr[n];
      }

      ierr = VecRestoreArray(normal[dim], &normal_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(velo[dim], &velo_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(kappa, &kappa_ptr); CHKERRXX(ierr);
    }

    p4est_t *p4est_np1 = p4est_copy(p4est, P4EST_FALSE);
    p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
    p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

    my_p4est_semi_lagrangian_t sl(&p4est_np1, &nodes_np1, &ghost_np1, ngbd);

//    sl.update_p4est(velo, dt, phi_extd);
    sl.update_p4est(velo, dt, phi, acn, 0);

    p4est_destroy(p4est);       p4est = p4est_np1;
    p4est_ghost_destroy(ghost); ghost = ghost_np1;
    p4est_nodes_destroy(nodes); nodes = nodes_np1;
    hierarchy->update(p4est, ghost);
    ngbd->update(hierarchy, nodes);

    my_p4est_level_set_t ls_new(ngbd);

    phi_extd = phi[0];
    phi_wall = phi[1];

    ls_new.enforce_contact_angle(phi_wall, phi_extd, cos_angle, 20);
    ls_new.reinitialize_1st_order_time_2nd_order_space(phi_extd, 20);

    iteration++;

    foreach_dimension(dim)
    {
      ierr = VecDestroy(velo[dim]); CHKERRXX(ierr);
      ierr = VecDestroy(normal[dim]); CHKERRXX(ierr);
    }
    ierr = VecDestroy(kappa); CHKERRXX(ierr);
    ierr = VecDestroy(cos_angle); CHKERRXX(ierr);
  }

  ierr = VecDestroy(phi_intf); CHKERRXX(ierr);
  ierr = VecDestroy(phi_wall); CHKERRXX(ierr);
  ierr = VecDestroy(phi_extd); CHKERRXX(ierr);

  delete ngbd;
  delete hierarchy;
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy      (p4est);
  my_p4est_brick_destroy(connectivity, &brick);

  return 0;
}
