
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
int lmin = 7;
int lmax = 7;
#endif
double lip = 1.5;

// output parameters
bool save_vtk = 1;

int num_geometry = 0;
int num_contact_angle = 0;

int num_surfaces = 2;

int max_iterations = 200;

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
    return MIN(fabs(phi_intf_cf(x,y)), fabs(phi_wall_cf(x,y)));
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
      case 0: return -0.9;
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
  Vec cos_angle; ierr = VecCreateGhostNodes(p4est, nodes, &cos_angle); CHKERRXX(ierr);

  sample_cf_on_nodes(p4est, nodes, phi_intf_cf, phi_intf);
  sample_cf_on_nodes(p4est, nodes, phi_wall_cf, phi_wall);
  sample_cf_on_nodes(p4est, nodes, cos_angle_cf, cos_angle);

  Vec phi_extd;

  ierr = VecDuplicate(phi_intf, &phi_extd);

  copy_ghosted_vec(phi_intf, phi_extd);

  my_p4est_level_set_t ls(ngbd);

  int iteration = 0;

  while (iteration < max_iterations)
  {

    //  ls.extend_Over_Interface_TVD(phi_wall, phi_intf, 100, 1);


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

      ierr = VecGetArray(phi_intf, &phi_intf_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(phi_wall, &phi_wall_ptr); CHKERRXX(ierr);
      ierr = VecGetArray(phi_extd, &phi_extd_ptr); CHKERRXX(ierr);

      my_p4est_vtk_write_all(p4est, nodes, ghost,
                             P4EST_TRUE, P4EST_TRUE,
                             3, 1, oss.str().c_str(),
                             VTK_POINT_DATA, "phi_intf", phi_intf_ptr,
                             VTK_POINT_DATA, "phi_wall", phi_wall_ptr,
                             VTK_POINT_DATA, "phi_extd", phi_extd_ptr,
                             VTK_CELL_DATA , "leaf_level", l_p);

      ierr = VecRestoreArray(phi_intf, &phi_intf_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_wall, &phi_wall_ptr); CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_extd, &phi_extd_ptr); CHKERRXX(ierr);

      ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
      ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

      PetscPrintf(mpi.comm(), "VTK saved in %s\n", oss.str().c_str());
    }

    ls.enforce_contact_angle(phi_wall, phi_extd, cos_angle, 1);
    ls.reinitialize_1st_order_time_2nd_order_space(phi_intf, 1);

    iteration++;
  }

  ierr = VecDestroy(phi_intf); CHKERRXX(ierr);
  ierr = VecDestroy(phi_wall); CHKERRXX(ierr);
  ierr = VecDestroy(phi_extd); CHKERRXX(ierr);
  ierr = VecDestroy(cos_angle); CHKERRXX(ierr);

  delete ngbd;
  delete hierarchy;
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy      (p4est);
  my_p4est_brick_destroy(connectivity, &brick);

  return 0;
}
