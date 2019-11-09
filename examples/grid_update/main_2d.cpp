/*
 * Title: grid_update
 * Description:
 * Author: Fernando Temprano
 * Date Created: 11-06-2019
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
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_semi_lagrangian.h>
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_macros.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_semi_lagrangian.h>
#endif

#include <iostream>
#include <iomanip>
#include <time.h>
#include <src/Parser.h>
#include <src/casl_math.h>
#include <src/parameter_list.h>
#include <src/petsc_compatibility.h>

using namespace std;

// -----------------------------------------------------------------------------------------------------------------------
// Description of the main file
// -----------------------------------------------------------------------------------------------------------------------

const static std::string main_description = "\
 In this example, we illustrate and test the procedures within the parCASL library to update a forest of quad-oc/trees \n\
 according to a given set of refinement criteria.                                                                      \n\
 Example of application of interest: Grid update from t=t_n to t_np1 via adaptive mesh refinement.                     \n\
 Developer: Fernando Temprano-Coleto (ftempranocoleto@ucsb.edu), November 2019.                                        \n ";

// -----------------------------------------------------------------------------------------------------------------------
// Definition of the parameters of the example
// -----------------------------------------------------------------------------------------------------------------------

// Declare the parameter list object
param_list_t pl;

// Grid parameters
param_t<int>          n          (pl, 2,    "n",          "Number of trees in each direction (default: 2)");
param_t<unsigned int> lmin       (pl, 4,    "lmin",       "Min. level of refinement (default: 4)");
param_t<unsigned int> lmax       (pl, 7,    "lmax",       "Max. level of refinement (default: 7)");
param_t<double>       lip        (pl, 1.2,  "lip",        "Lipschitz constant (default: 1.2)");
param_t<double>       b_width    (pl, 2.0,  "b_width",    "Bandwidth of uniform cells around the interface (default: 2.0)");
param_t<double>       thres      (pl, 15,   "threshold",  "Threshold value of the gradient of the scalar field for refinement (default: 15)");


// Method setup
param_t<double>       duration   (pl, 1.0,  "duration",   "Duration of the simulation (default: 1.0)");
param_t<bool>         save_vtk   (pl, 1,    "save_vtk",   "Duration of the simulation (default: 1)");
param_t<int>          num_it_vtk (pl, 1,    "num_it_vtk", "Save vtk files every num_it_vtk iterations (default: 1)");

// -----------------------------------------------------------------------------------------------------------------------
// Define auxiliary classes
// -----------------------------------------------------------------------------------------------------------------------

// Continuous signed-distance level-set function representing a circle (2D) or a sphere (3D)
#ifdef P4_TO_P8
struct sphere_ls : CF_3
{
  sphere_ls(double x0_, double y0_, double z0_, double R_, double r_, double omega_, double *time_)
    : x0(x0_), y0(y0_), z0(z0_), R(R_), r(r_), omega(omega_)
#else
struct sphere_ls : CF_2
{
  sphere_ls(double x0_, double y0_, double R_, double r_, double omega_, double *time_)
    : x0(x0_), y0(y0_), R(R_), r(r_), omega(omega_)
#endif
  {
    time = time_;
  }
#ifdef P4_TO_P8
  double operator()(double x, double y, double z) const
  {
    return R - sqrt(SQR(x-x0-r*cos(omega*(*time))) + SQR(y-y0-r*sin(omega*(*time))) + SQR(z-z0));
#else
  double operator()(double x, double y) const
  {
    return R - sqrt(SQR(x-x0-r*cos(omega*(*time))) + SQR(y-y0-r*sin(omega*(*time))));
#endif
  }
private:
  double x0, y0;
#ifdef P4_TO_P8
  double z0;
#endif
  double R;
  double r;
  double omega;
  double *time;
};

// Time-dependent scalar field for refinement purposes
#ifdef P4_TO_P8
struct scalar_t : CF_3
{
  scalar_ls(double delta_, double omega_, double *time_)
#else
struct scalar_t : CF_2
{
  scalar_t(double delta_, double omega_, double *time_)
#endif
    : delta(delta_), omega(omega_)
  {
    time = time_;
  }

#ifdef P4_TO_P8
  double operator()(double x, double y, double z) const
  {
    return PI + tanh( ( (x-0.5) + (y-0.5) + (z-0.5) - 0.5*sin(omega*(*time)) )/ delta );
#else
  double operator()(double x, double y) const
  {
    return PI + tanh( ( (x-0.5) + (y-0.5) - 0.5*sin(omega*(*time)) ) / delta );
#endif
  }

private:
  double delta;
  double omega;
  double *time;
};

// Custom splitting criteria
class splitting_criteria_custom_t : public splitting_criteria_tag_t
{
private:
  double band_width;
  double thres_scalar;
  void tag_quadrant(p4est_t *p4est, p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, p4est_nodes_t *nodes, const double *phi_p, const double *scalar_p)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index( &tree->quadrants, quad_idx-(tree->quadrants_offset) );

    if (quad->level < min_lvl)
      quad->p.user_int = REFINE_QUADRANT; // if coarser than min_lvl, mark for refinement

    else if (quad->level > max_lvl)
      quad->p.user_int = COARSEN_QUADRANT; // if finer than max_lvl, mark for coarsening

    else
    {
      double dxyz[P4EST_DIM];
      double dxyz_min, diag_min;
      get_dxyz_min(p4est, dxyz, dxyz_min, diag_min);
      const double quad_diag = diag_min*(1<<( max_lvl-(quad->level) ));

      bool coarsen = (quad->level > min_lvl);
      if(coarsen)
      {
        bool coar_intf   = true;
        bool coar_band   = true;
        bool coar_scalar = true;
        p4est_locidx_t node_idx;

        // check the P4EST_CHILDREN vertices of the quadrant and enforce all conditions in each
        for(unsigned short v=0; v < P4EST_CHILDREN; ++v)
        {
          node_idx  = nodes->local_nodes[P4EST_CHILDREN*quad_idx+v];

          coar_intf   = coar_intf   && ( fabs(phi_p[node_idx])  >= lip*2.0*quad_diag );
          coar_band   = coar_band   && ( fabs(phi_p[node_idx])  >  MAX(1.0,band_width)*diag_min );
          coar_scalar = coar_scalar && fabs(scalar_p[node_idx]) <  thres_scalar;

          coarsen = coar_intf && coar_band && coar_scalar; // Need ALL of the coarsening conditions satisfied to coarsen the quadrant
          if(!coarsen)
            break;
        }
      }

      bool refine = (quad->level < max_lvl);
      if(refine)
      {
        bool ref_intf   = false;
        bool ref_band   = false;
        bool ref_scalar = false;
        p4est_locidx_t node_idx;

        // check the P4EST_CHILDREN vertices of the quadrant and enforce all conditions in each
        for(unsigned short v=0; v < P4EST_CHILDREN; ++v)
        {
          node_idx  = nodes->local_nodes[P4EST_CHILDREN*quad_idx+v];

          ref_intf   = ref_intf   || ( fabs(phi_p[node_idx])  <= lip*quad_diag );
          ref_band   = ref_band   || ( fabs(phi_p[node_idx])  <  MAX(1.0,band_width)*diag_min );
          ref_scalar = ref_scalar || fabs(scalar_p[node_idx]) >  thres_scalar;

          refine = ref_intf || ref_band || ref_scalar; // Need AT LEAST ONE of the refining conditions satisfied to refine the quadrant
          if(refine)
            break;
        }
      }

      if (refine)
        quad->p.user_int = REFINE_QUADRANT;
      else if (coarsen)
        quad->p.user_int = COARSEN_QUADRANT;
      else
        quad->p.user_int = SKIP_QUADRANT;
    }
  }
public:
  splitting_criteria_custom_t(int min_lvl_, int max_lvl_, double lip_, double band_width_, double thres_scalar_)
    : splitting_criteria_tag_t(min_lvl_, max_lvl_, lip_), band_width(band_width_), thres_scalar(thres_scalar_) {}
  bool refine_and_coarsen(p4est_t* p4est, p4est_nodes_t* nodes, p4est_ghost_t* ghost, my_p4est_node_neighbors_t* ngbd, Vec phi, Vec scalar)
  {
    // Read phi to a pointer
    const double *phi_p;
    PetscErrorCode ierr;
    ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);

    // Compute norm of the gradient of the scalar
    Vec d_s[P4EST_DIM];
    for(unsigned short dir=0; dir<P4EST_DIM; ++dir) { ierr = VecCreateGhostNodes(p4est, nodes, &d_s[dir]); CHKERRXX(ierr); }
    ngbd->first_derivatives_central(scalar, d_s);

    Vec grad_norm_s; ierr = VecCreateGhostNodes(p4est, nodes, &grad_norm_s); CHKERRXX(ierr);
    double *grad_norm_s_p;
    const double *d_s_p[P4EST_DIM];
    for(unsigned short dir=0; dir<P4EST_DIM; ++dir) { ierr = VecGetArrayRead(d_s[dir], &d_s_p[dir]); CHKERRXX(ierr); }
    ierr = VecGetArray(grad_norm_s, &grad_norm_s_p); CHKERRXX(ierr);
    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    {
  #ifdef P4_TO_P8
      grad_norm_s_p[n] = sqrt(SQR(d_s_p[0][n])+SQR(d_s_p[1][n])+SQR(d_s_p[2][n]));
  #else
      grad_norm_s_p[n] = sqrt(SQR(d_s_p[0][n])+SQR(d_s_p[1][n]));
  #endif
    }
    ierr = VecRestoreArray(grad_norm_s, &grad_norm_s_p); CHKERRXX(ierr);
    for(unsigned short dir=0; dir<P4EST_DIM; ++dir) { ierr = VecRestoreArrayRead(d_s[dir], &d_s_p[dir]); CHKERRXX(ierr); }
    for(unsigned short dir=0; dir<P4EST_DIM; ++dir) { ierr = VecDestroy(d_s[dir]); CHKERRXX(ierr); }

    // Read the grad norm to a pointer
    const double *grad_norm_s_read_p;
    ierr = VecGetArrayRead(grad_norm_s, &grad_norm_s_read_p); CHKERRXX(ierr);

    // Tag the quadrants that need to be refined or coarsened
    for (p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
    {
      p4est_tree_t* tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
      for (size_t q = 0; q <tree->quadrants.elem_count; ++q)
      {
        p4est_locidx_t quad_idx  = q + tree->quadrants_offset;
        tag_quadrant(p4est, quad_idx, tree_idx, nodes, phi_p, grad_norm_s_read_p);
      }
    }
    ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(grad_norm_s, &grad_norm_s_read_p); CHKERRXX(ierr);

    my_p4est_coarsen(p4est, P4EST_FALSE, splitting_criteria_custom_t::coarsen_fn, splitting_criteria_custom_t::init_fn);
    my_p4est_refine (p4est, P4EST_FALSE, splitting_criteria_custom_t::refine_fn,  splitting_criteria_custom_t::init_fn);

    int is_grid_changed = false;
    for (p4est_topidx_t it = p4est->first_local_tree; it <= p4est->last_local_tree; ++it)
    {
      p4est_tree_t* tree = (p4est_tree_t*)sc_array_index(p4est->trees, it);
      for (size_t q = 0; q <tree->quadrants.elem_count; ++q)
      {
        p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
        if (quad->p.user_int == NEW_QUADRANT)
        {
          is_grid_changed = true;
          goto function_end;
        }
      }
    }

  function_end:
    MPI_Allreduce(MPI_IN_PLACE, &is_grid_changed, 1, MPI_INT, MPI_LOR, p4est->mpicomm);
    return is_grid_changed;
  }
};

// -----------------------------------------------------------------------------------------------------------------------
// Main function
// -----------------------------------------------------------------------------------------------------------------------

int main(int argc, char** argv) {

  // Prepare the parallel enviroment
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  // Declaration of the stopwatch object
  parStopWatch sw;
  sw.start("Running example: grid_update");

  // Declaration of the PETSc error flag variable
  PetscErrorCode ierr;

  // Display initial message
  char init_msg[1024];
  sprintf(init_msg, "\n"
                    "-------------------------------------------------------------------\n"
                    "------------------== parCASL: grid_update test ==------------------\n"
                    "-------------------------------------------------------------------\n"
                    "\n");
  ierr = PetscPrintf(mpi.comm(), init_msg); CHKERRXX(ierr);

  // Get parameter values from the run command
  cmdParser cmd;
  pl.initialize_parser(cmd);
  if (cmd.parse(argc, argv, main_description)) return 0;
  pl.set_from_cmd_all(cmd);

  // Domain size information
  const int n_xyz[]      = { n(), n(), n()};
  const double xyz_min[] = { 0.0, 0.0, 0.0};
  const double xyz_max[] = { 1.0, 1.0, 1.0};
  const int periodic[]   = {   0,   0,   0};

  // Define initial and final times
  double tn = 0.0;
  double tf = duration();

  // Define the initial interface
  double R  = 0.15;
  double x0 = 0.5;
  double y0 = 0.5;
  double r  = 0.25;
  double omega_sphere = 4*PI;
#ifdef P4_TO_P8
  double z0 = 0.0;
  sphere_ls geom(x0,y0,z0,R,r,omega_shpere,&tn);
#else
  sphere_ls geom(x0,y0,R,r,omega_sphere,&tn);
#endif

  // Declaration of the macromesh via the brick and connectivity objects
  my_p4est_brick_t      brick;
  p4est_connectivity_t* conn;
  conn = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

  // Display initial message
  char split_msg[1024];
  sprintf(split_msg, "\n Iteration %04d: t = %1.4f \n",
          lmin(), lmax(), 0, 0.0);
  ierr = PetscPrintf(mpi.comm(), split_msg); CHKERRXX(ierr);

  // Declaration of pointers to p4est variables
  p4est_t*       p4est_n;
  p4est_ghost_t* ghost_n;
  p4est_nodes_t* nodes_n;

  // Create the forest
  p4est_n = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL);

  // Refine based on distance to the interface
  splitting_criteria_cf_t sp(lmin(), lmax(), &geom);
  p4est_n->user_pointer = &sp;
  for(unsigned int iter=0; iter<lmax(); ++iter)
  {
    my_p4est_refine(p4est_n, P4EST_FALSE, refine_levelset_cf, NULL);
    my_p4est_partition(p4est_n, P4EST_FALSE, NULL);
  }

  // Create ghost layer and nodes structure
  ghost_n = my_p4est_ghost_new(p4est_n, P4EST_CONNECT_FULL);
  nodes_n = my_p4est_nodes_new(p4est_n, ghost_n);

  // Create my_p4est objects
  my_p4est_hierarchy_t *hierarchy_n = new my_p4est_hierarchy_t(p4est_n, ghost_n, &brick);
  my_p4est_node_neighbors_t *ngbd_n = new my_p4est_node_neighbors_t(hierarchy_n, nodes_n);
  ngbd_n->init_neighbors();

  // Compute grid size data
  double dxyz[P4EST_DIM];
  double dxyz_min;
  double diag_min;
  get_dxyz_min(p4est_n, dxyz, dxyz_min, diag_min);

  // Define the scalar
  double delta = 5*dxyz_min;
  double omega_scalar = PI*exp(2.0);
  scalar_t s_cf(delta, omega_scalar, &tn);

  // Declare data Vecs and pointers for read/write
  Vec phi_n; double *phi_n_p;
  Vec s_n; double *s_n_p;
  Vec grad_norm_s_n; double *grad_norm_s_n_p;

  // Allocate memory for the Vecs
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &phi_n);         CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &s_n);           CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &grad_norm_s_n); CHKERRXX(ierr);

  // Sample the level-set function at t=0
  sample_cf_on_nodes(p4est_n, nodes_n, geom, phi_n);

  // Sample the scalar at t=0
  sample_cf_on_nodes(p4est_n, nodes_n, s_cf, s_n);

  // Save the initial grid and fields into vtk
  if(save_vtk())
  {
    char name[1024];
    sprintf(name, "visualization_%d", 0);
    ierr = VecGetArray(phi_n,         &phi_n_p);         CHKERRXX(ierr);
    ierr = VecGetArray(s_n,           &s_n_p);           CHKERRXX(ierr);
    my_p4est_vtk_write_all(p4est_n, nodes_n, ghost_n,
                           P4EST_TRUE, P4EST_TRUE,
                           2, 0, name,
                           VTK_POINT_DATA, "phi", phi_n_p,
                           VTK_POINT_DATA, "s",   s_n_p);
    ierr = VecRestoreArray(phi_n, &phi_n_p);                 CHKERRXX(ierr);
    ierr = VecRestoreArray(s_n, &s_n_p);                     CHKERRXX(ierr);
    char msg[1024];
    sprintf(msg, " -> Saving vtu files in %s.vtu\n", name);
    ierr = PetscPrintf(mpi.comm(), msg); CHKERRXX(ierr);
  }

  // Define time stepping variables
  int iter = 0;
  int vtk_idx = 1;
  double dt_n = dxyz_min;

  while(tn+0.1*dt_n<tf)
  {
    // Clip time step if it's going to go over the final time
    if(tn+dt_n>tf)
      dt_n = tf-tn;

    // Declare the continuous-function refinement object
    splitting_criteria_custom_t sp(lmin(), lmax(), lip(), b_width(), thres());

    // Point the user user_pointer of p4est to refinement object
    p4est_n->user_pointer = &sp;

    // Update time and iteration counter
    tn += dt_n;
    iter++;

    // Re-sample the level-set function (since the time has changed)
    ierr = VecDestroy(phi_n);                             CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &phi_n); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est_n, nodes_n, geom, phi_n);

    // Re-sample the scalar function (since the time has changed)
    ierr = VecDestroy(s_n);                             CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &s_n); CHKERRXX(ierr);
    sample_cf_on_nodes(p4est_n, nodes_n, s_cf, s_n);

    // Grid update
    bool grid_is_changing = true;
    unsigned int iter_ref=0;
    while(grid_is_changing)
    {
      // Refine grid
      ierr = VecGetArray(phi_n, &phi_n_p); CHKERRXX(ierr);
      grid_is_changing = sp.refine_and_coarsen(p4est_n, nodes_n, ghost_n, ngbd_n, phi_n, s_n);
      ierr = VecRestoreArray(phi_n, &phi_n_p); CHKERRXX(ierr);

      if(grid_is_changing)
      {
        // Partition, create ghosts, create nodes
        my_p4est_partition(p4est_n, P4EST_FALSE, NULL);
        p4est_ghost_destroy(ghost_n); ghost_n = my_p4est_ghost_new(p4est_n, P4EST_CONNECT_FULL);
        p4est_nodes_destroy(nodes_n); nodes_n = my_p4est_nodes_new(p4est_n, ghost_n);
        delete hierarchy_n;           hierarchy_n = new my_p4est_hierarchy_t(p4est_n, ghost_n, &brick);
        delete ngbd_n;                ngbd_n      = new my_p4est_node_neighbors_t(hierarchy_n, nodes_n);

        // Re-sample the level-set function (since the grid has change and we need it for further refinement)
        ierr = VecDestroy(phi_n);                             CHKERRXX(ierr);
        ierr = VecCreateGhostNodes(p4est_n, nodes_n, &phi_n); CHKERRXX(ierr);
        sample_cf_on_nodes(p4est_n, nodes_n, geom, phi_n);

        // Re-sample the scalar function (since the grid has change and we need it for further refinement)
        ierr = VecDestroy(s_n);                             CHKERRXX(ierr);
        ierr = VecCreateGhostNodes(p4est_n, nodes_n, &s_n); CHKERRXX(ierr);
        sample_cf_on_nodes(p4est_n, nodes_n, s_cf, s_n);

        // Keep track of the iterations
        iter_ref++;
        if(iter_ref>lmax()-lmin()+1)
        {
          ierr = PetscPrintf(mpi.comm(), "[WARNING:] The grid update did not converge.");
          break;
        }
      }
    }

    // Display iteration message
    char iter_msg[1024];
    sprintf(iter_msg, " Iteration %04d: t = %1.4f \n",
            iter, tn);
    ierr = PetscPrintf(mpi.comm(), iter_msg); CHKERRXX(ierr);

    // Save to vtk format
    if(save_vtk() && (iter >= vtk_idx*num_it_vtk() || tn==tf))
    {
      char name[1024];
      sprintf(name, "visualization_%d", vtk_idx);
      ierr = VecGetArray(phi_n, &phi_n_p); CHKERRXX(ierr);
      ierr = VecGetArray(s_n, &s_n_p);     CHKERRXX(ierr);
      my_p4est_vtk_write_all(p4est_n, nodes_n, ghost_n,
                             P4EST_TRUE, P4EST_TRUE,
                             2, 0, name,
                             VTK_POINT_DATA, "phi", phi_n_p,
                             VTK_POINT_DATA, "s",   s_n_p);
      ierr = VecRestoreArray(phi_n, &phi_n_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(s_n, &s_n_p);     CHKERRXX(ierr);
      char msg[1024];
      sprintf(msg, " -> Saving vtu files in %s.vtu\n", name);
      ierr = PetscPrintf(mpi.comm(), msg); CHKERRXX(ierr);
      ++vtk_idx;
    }
  }

  // Destroy the dynamically allocated Vecs
  ierr = VecDestroy(phi_n);         CHKERRXX(ierr);
  ierr = VecDestroy(s_n);           CHKERRXX(ierr);
  ierr = VecDestroy(grad_norm_s_n); CHKERRXX(ierr);

  // Destroy the dynamically allocated p4est and my_p4est structures
  delete ngbd_n;
  delete hierarchy_n;
  p4est_nodes_destroy   (nodes_n);
  p4est_ghost_destroy   (ghost_n);
  p4est_destroy         (p4est_n);

  // Destroy the dynamically allocated brick and connectivity structures
  my_p4est_brick_destroy(conn, &brick);

  // Display end message
  char end_msg[1024];
  sprintf(end_msg, "\n -------------------------------------------------------------------\n"
                   "\n ");
  ierr = PetscPrintf(mpi.comm(), end_msg); CHKERRXX(ierr);

  // Stop and print global timer
  sw.stop();
  sw.read_duration();
}

