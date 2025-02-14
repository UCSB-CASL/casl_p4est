/*
 * Title: get_started_with_my_p4est
 * Description: illustration of data structure and data access with basic my_p4est_... objects
 * Author: Raphael Egan
 * Date Created: 10-01-2019
 */

#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_macros.h>
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_macros.h>
#endif

#include <src/Parser.h>
#include <src/casl_math.h>

using namespace std;

#ifdef P4_TO_P8
  struct:CF_3{
    double operator()(double x, double y, double z) const {
      return 0.5 - sqrt(SQR(x) + SQR(y) + SQR(z));
    }
  } circle;
#else
  struct:CF_2{
    double operator()(double x, double y) const {
      return 0.5 - sqrt(SQR(x) + SQR(y));
    }
  } circle;
#endif

void create_my_grid_my_ghost_and_my_nodes(const mpi_environment_t &mpi, p4est_connectivity_t* conn, const unsigned int &lmin, const unsigned int &lmax,
                                          p4est_t* &forest, p4est_ghost_t* &ghosts, p4est_nodes_t* &nodes)
{
  if(forest!=NULL)
    p4est_destroy(forest);
  if(nodes!=NULL)
    p4est_nodes_destroy(nodes);
  if(ghosts!=NULL)
    p4est_ghost_destroy(ghosts);
  // creation and refinement of the p4est structure is not the purpose of this illustrative example
  // create the forest
  forest = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL);
  // refine based on distance to a level-set
  splitting_criteria_cf_t sp(lmin, lmax, &circle);
  forest->user_pointer = &sp;
  for (unsigned int k = 0; k < lmax; ++k) {
    // refine the forest once more
    my_p4est_refine(forest, P4EST_FALSE, refine_levelset_cf, NULL);
    // partition the forest
    my_p4est_partition(forest, P4EST_TRUE, NULL);
  }

  // create ghost layer
  ghosts = my_p4est_ghost_new(forest, P4EST_CONNECT_FULL);

  // create node structure: ALWAYS use 'my_p4est_nodes_new', never use 'p4est_nodes_new'
  // this is critical to ensure consistency with the rest of my_p4est_... library
  nodes = my_p4est_nodes_new(forest, ghosts);
}

int main(int argc, char** argv) {

  // prepare parallel enviroment
  mpi_environment_t mpi;
  mpi.init(argc, argv);
  PetscErrorCode ierr;

  // p4est variables
  p4est_t*              p4est = NULL; // a pointer to the p4est object, representing the grid
  p4est_nodes_t*        nodes = NULL; // a pointer to the nodes
  p4est_ghost_t*        ghost = NULL; // a pointer to the ghost cells
  p4est_connectivity_t* conn;         // a pointer to the p4est connectivity (basic connectivity information of the macromesh required and created by p4est)
  my_p4est_brick_t      brick;        // our own description of the macromesh: uniform distribution of same-size root cells onto a Cartesian macro-grid

  // domain size information
  const int n_xyz[]      = { 2,  2,  2}; // number of root cells in the macromesh along x, y, z
  const double xyz_min[] = {-1, -1, -1}; // coordinates of the front, lower left corner of the computational domain
  const double xyz_max[] = { 1,  1,  1}; // coordinates of the back, upper right corner of the computational domain
  const int periodic[]   = { 0,  0,  0}; // periodicity flag along the x, y, z (0== no periodicity, 1==periodicity)
  conn = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic); // this builds the brick and create the p4est-required connectivity structure

  // now create the p4est grid for real, every tree in the forest will have refinement levels 3/8
  create_my_grid_my_ghost_and_my_nodes(mpi, conn, 3, 8, p4est, ghost, nodes);
  // in the following, "quadrant" == "quadrant (resp. octant) in 2D (resp. 3D)", or similarly, "quadrant" == "leaf cell"
  // p4est: contains tree structures which themselves contain arrays of locally owned quadrants (sorted by increasing value of their z-code)
  //
  // ghost: list of all the ghost quadrants layering the local partition of the current process (sorted by increasing value of their z-code)
  //
  // nodes: contains all the grid nodes that the current partition is aware of, i.e. all the corner vertices of the locally
  //        owned quadrants and of the ghost quadrants. Thanks to Mohammad Mirzadeh, the nodes in the indep_nodes inner array
  //        are sorted in the convenient following way:
  //        - the num_owned_indeps locally owned nodes are listed first (sorted by increazing value of their z-code)
  //        - the (indep_nodes.elem_count-num_owned_indeps) ghost nodes are then listed (sorted by increazing value of their z-code)

  // Now that the grid is created, let's sample a field at the nodes
  // so let's create a (ghosted) Petsc parallel vector for node-sampling
  Vec node_sampled_field;
  double *node_sampled_field_p;
  ierr = VecCreateGhostNodes(p4est, nodes, &node_sampled_field); CHKERRXX(ierr);
  ierr = VecGetArray(node_sampled_field, &node_sampled_field_p); CHKERRXX(ierr);
  // let's sample through the nodes of the grid: we will sample the 'circle' function  at the nodes
  // In our my_p4est framework, the local indices in the p4est_nodes_t structure matches the local array indices
  // in the parallel vectors built with VecCreateGhostNodes(), thanks to Mohammad Mirzadeh's work on my_p4est_nodes_new
  // [VERY IMPORTANT NOTE: always use my_p4esy_nodes_new, NEVER p4est_nodes_new, you would lose consistency]
//  // APPROACH ONE: loop through locally owned nodes, then through locally known ghost nodes
//  for (p4est_locidx_t k = 0; k < nodes->num_owned_indeps; ++k) {
//    double xyz_node[P4EST_DIM]; // P4EST_DIM will be 2 or 3 depending of whether you compile for 2d or 3d scenarii
//    // this routine calculates the coordinates of node k
//    node_xyz_fr_n(k, p4est, nodes, xyz_node);
//    // one can distinguish between 2D and 3D code by using the precompiler directive '#ifdef P4_TO_P8' since P4_TO_P8 is defined only in 3D
//#ifdef P4_TO_P8
//    node_sampled_field_p[k] = circle(xyz_node[0], xyz_node[1], xyz_node[2]); // 3D code
//#else
//    node_sampled_field_p[k] = circle(xyz_node[0], xyz_node[1]); // 2D code
//#endif
//  }
//  for (size_t ghost_node_idx = 0; ghost_node_idx < (nodes->indep_nodes.elem_count-nodes->num_owned_indeps); ++ghost_node_idx) {
//    double xyz_node[P4EST_DIM]; // P4EST_DIM will be 2 or 3 depending of whether you compile for 2d or 3d scenarii
//    // this routine calculates the coordinates of node k
//    node_xyz_fr_n(nodes->num_owned_indeps+ghost_node_idx, p4est, nodes, xyz_node);
//    // one can distinguish between 2D and 3D code by using the precompiler directive '#ifdef P4_TO_P8' since P4_TO_P8 is defined only in 3D
//#ifdef P4_TO_P8
//    node_sampled_field_p[nodes->num_owned_indeps+ghost_node_idx] = circle(xyz_node[0], xyz_node[1], xyz_node[2]); // 3D code
//#else
//    node_sampled_field_p[nodes->num_owned_indeps+ghost_node_idx] = circle(xyz_node[0], xyz_node[1]); // 2D code
//#endif
//  }
  // APPROACH TWO: loop through all locally known nodes (local and ghost right away)
  for (size_t k = 0; k < nodes->indep_nodes.elem_count; ++k) {
    double xyz_node[P4EST_DIM]; // P4EST_DIM will be 2 or 3 depending of whether you compile for 2d or 3d scenarii
    // this routine calculates the coordinates of node k
    node_xyz_fr_n(k, p4est, nodes, xyz_node);
    // one can distinguish between 2D and 3D code by using the precompiler directive '#ifdef P4_TO_P8' since P4_TO_P8 is defined only in 3D
#ifdef P4_TO_P8
    node_sampled_field_p[k] = circle(xyz_node[0], xyz_node[1], xyz_node[2]); // 3D code
#else
    node_sampled_field_p[k] = circle(xyz_node[0], xyz_node[1]); // 2D code
#endif
  }

  // Now, let's sample a field at the cell centers
  // In order to illustrate how to access the vertices of a given quadrant and the usage of 'local_nodes' within
  // the p4est_nodes_t structures, we will define the cell-sampled field, as the quadrant-average value of the
  // former node_sampled_field: for every known quadrant, we will find the node indices corresponding to its
  // P4EST_CHILDREN (==4 in 2d, 8 in 3d) vertices, find the corresponding values of the node-sampled-field and
  // add it to the cell average value.
  // So let's create a (ghosted) Petsc parallel vector for cell-sampling
  Vec cell_mean_sampled_field;
  double *cell_mean_sampled_field_p;
  ierr = VecCreateGhostCells(p4est, ghost, &cell_mean_sampled_field); CHKERRXX(ierr);
  ierr = VecGetArray(cell_mean_sampled_field, &cell_mean_sampled_field_p); CHKERRXX(ierr);
  // let's sample through the cells of the grid: in this case, we have no other choice than looping through the locally
  // owned quadrants, then the ghost quadrants

  // loop through locally owned quadrants: you loop through the locally known trees first, then through their own
  // array of quadrants
  for (p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx) {
    // note the unusual '<=' instead of '<' in the above loop termination criterion
    // access the tree:
    p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
    for (size_t q = 0; q < tree->quadrants.elem_count; ++q) {
      p4est_locidx_t local_quad_idx = tree->quadrants_offset + q;
      double cell_averaged_value = 0.0;
      for (int vv = 0; vv < P4EST_CHILDREN; ++vv) {
        // this is how you get the local node index of the vv^{th} vertex of the local quadrant of local index local_quad_idx:
        p4est_locidx_t local_index_of_vertex = nodes->local_nodes[P4EST_CHILDREN*local_quad_idx+vv];
        cell_averaged_value += node_sampled_field_p[local_index_of_vertex]/((double) P4EST_CHILDREN);
      }
      cell_mean_sampled_field_p[local_quad_idx] = cell_averaged_value;
    }
  }

  // loop through the locally-aware ghost quadrants
  for (size_t ghost_idx = 0; ghost_idx < ghost->ghosts.elem_count; ++ghost_idx) {
    p4est_locidx_t local_quad_idx = p4est->local_num_quadrants+ghost_idx;
    double cell_averaged_value = 0.0;
    for (int vv = 0; vv < P4EST_CHILDREN; ++vv) {
      p4est_locidx_t local_index_of_vertex = nodes->local_nodes[P4EST_CHILDREN*local_quad_idx+vv];
      cell_averaged_value += node_sampled_field_p[local_index_of_vertex]/((double) P4EST_CHILDREN);
    }
    cell_mean_sampled_field_p[local_quad_idx] = cell_averaged_value;
  }

  // save the grid into vtk
  my_p4est_vtk_write_all(p4est, nodes, ghost,
                         P4EST_TRUE, P4EST_TRUE,
                         1, 1, "visualization",
                         VTK_POINT_DATA, "levelset", node_sampled_field_p,
                         VTK_CELL_DATA, "cell-mean levelset", cell_mean_sampled_field_p);

  ierr = VecRestoreArray(cell_mean_sampled_field, &cell_mean_sampled_field_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(node_sampled_field, &node_sampled_field_p); CHKERRXX(ierr);
  // never forget to destroy what you have created:
  ierr = VecDestroy(cell_mean_sampled_field); CHKERRXX(ierr);
  ierr = VecDestroy(node_sampled_field); CHKERRXX(ierr);
  // destroy the structures
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy      (p4est);
  my_p4est_brick_destroy(conn, &brick);
}

