/*
 * Title: grid_creation_and_refinement
 * Description:
 * Author: ftc
 * Date Created: 10-22-2019
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
 In this example, we illustrate and test the main procedures to create and refine a forest of Quad/Oc-trees within the \n\
 parCASL library. We focus on creating a grid from scratch, not on updating the grid from a previous one, since that   \n\
 will be covered in the 'grid_update' example. Specifically, we address the two main paradigms of grid refinement      \n\
 within pasCASL, i.e. (i) refinement from a continuous function (setting 'method' as 1 or 2) and (ii) refinement from  \n\
 grid-sampled data via tagging each quadrant/octant (setting 'method' as 3). The script first creates the relevant     \n\
 p4est and my_p4est objects, and then refines the grid around a circle (in 2D) or sphere (in 3D) that is randomly      \n\
 placed in the domain.                                                                                                 \n\
 Example of application of interest: Creation of any computational grid in the parCASL library.                        \n\
 Developer: Fernando Temprano-Coleto (ftempranocoleto@ucsb.edu), October 2019.                                         \n ";

// -----------------------------------------------------------------------------------------------------------------------
// Definition of the parameters of the example
// -----------------------------------------------------------------------------------------------------------------------

// Declare the parameter list object
param_list_t pl;

// Grid parameters
param_t<int>          nx         (pl, 2,    "nx", "Number of trees in the x direction (default: 2)");
param_t<int>          ny         (pl, 2,    "ny", "Number of trees in the y direction (default: 2)");
#ifdef P4_TO_P8
param_t<int>          nz         (pl, 2,    "nz", "Number of trees in the z direction (default: 2)");
#endif
param_t<unsigned int> lmin       (pl, 3,    "lmin", "Min. level of refinement (default: 3)");
param_t<unsigned int> lmax       (pl, 7,    "lmax", "Max. level of refinement (default: 7)");
param_t<double>       lip        (pl, 1.2,  "lip",  "Lipschitz constant (default: 1.2)");

// Method setup
param_t<bool>         print_iter (pl, true, "print_iter", "Output each refinement iteration (1)\n"
                                                      "or only final grid (0) (default:1)");
param_t<unsigned int> method     (pl, 1,    "method", "Method of grid refinement (default: 1): \n\
                                                          0 - Do nothing,\n\
                                                          1 - Continuous function: interface,\n\
                                                          2 - Continuous function: interface and band of uniform cells,\n\
                                                          3 - Tag quadrants: interface.");

// -----------------------------------------------------------------------------------------------------------------------
// Define auxiliary classes
// -----------------------------------------------------------------------------------------------------------------------

// Random number generator
double random_gen(const double &min=0.0, const double &max=1.0)
{
  return (min+(max-min)*((double) rand())/((double) RAND_MAX));
}

// Continuous signed-distance level-set function representing a circle (2D) or a sphere (3D)
#ifdef P4_TO_P8
struct sphere_ls : CF_3
{
  sphere_ls(double x0_, double y0_, double z0_, double R_): x0(x0_), y0(y0_), z0(z0_), R(R_) {}
#else
struct sphere_ls : CF_2
{
  sphere_ls(double x0_, double y0_, double R_): x0(x0_), y0(y0_), R(R_) {}
#endif
#ifdef P4_TO_P8
  double operator()(double x, double y, double z) const
  {
    return R - sqrt(SQR(x-x0) + SQR(y-y0) + SQR(z-z0));
#else
  double operator()(double x, double y) const
  {
    return R - sqrt(SQR(x-x0) + SQR(y-y0));
#endif
  }
private:
  double x0, y0;
#ifdef P4_TO_P8
  double z0;
#endif
  double R;
};

// -----------------------------------------------------------------------------------------------------------------------
// Main function
// -----------------------------------------------------------------------------------------------------------------------

int main(int argc, char** argv) {

  // Prepare the parallel enviroment
  mpi_environment_t mpi;
  mpi.init(argc, argv);

  // Declaration of the stopwatch object
  parStopWatch w;
  w.start("Running example: grid_creation_and_refinement");

  // Get parameter values from the run command
  cmdParser cmd;
  pl.initialize_parser(cmd);
  if (cmd.parse(argc, argv, main_description)) return 0;
  pl.set_from_cmd_all(cmd);

  // Declaration of the PETSc error flag variable
  PetscErrorCode ierr;

  // Initialize the random seed generator
  srand(time(NULL));

  // Domain size information
#ifdef P4_TO_P8
  const int    n_xyz[]    = {     nx.val,      ny.val,      nz.val}; // Number of trees in each dimension from inputs
#else
  const int    n_xyz[]    = {     nx.val,      ny.val,          0 }; // Number of trees in each dimension from inputs
#endif
  const double xyz_min[]  = {        0.0,         0.0,         0.0}; // Cartesian coordinates of the domain corner with minimum x, y, and z
  const double xyz_max[]  = {n_xyz[0]*PI, n_xyz[1]*PI, n_xyz[2]*PI}; // Cartesian coordinates of the domain corner with maximum x, y, and z
  const int    periodic[] = {          0,           0,           0}; // No periodicity of the tree in any dimension

  // Declare continuous level-set function
  sphere_ls sphere(random_gen(xyz_min[0], xyz_max[0]),
                   random_gen(xyz_min[1], xyz_max[1]),
#ifdef P4_TO_P8
                   random_gen(xyz_min[2], xyz_max[2]),
#endif
                   PI/exp(1.0));

  // Declaration of the *macromesh* via the brick and connectivity objects
  my_p4est_brick_t      brick;
  p4est_connectivity_t* conn;
  conn = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

  // Declaration of pointers to p4est variables
  p4est_t*       p4est;
  p4est_ghost_t* ghost;
  p4est_nodes_t* nodes;

  // Create the forest
  p4est = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL);

  // Grid refinement methods
  switch(method.val)
  {
    case 0: /*------ DO NOTHING ----------------------------------------------------------------------------------------*/
    {
      // Partition, create ghosts, create nodes
      my_p4est_partition(p4est, P4EST_FALSE, NULL);
      ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
      nodes = my_p4est_nodes_new(p4est, ghost);

      // Output the grid
      my_p4est_vtk_write_all(p4est, nodes, ghost,
                             P4EST_TRUE, P4EST_TRUE,
                             0, 0, "visualization_0");

      break;
    }
    case 1: /*------ CONTINUOUS FUNCTION: INTERFACE --------------------------------------------------------------------*/
    {
      // Declare the continuous-function refinement object
      splitting_criteria_cf_t sp(lmin.val, lmax.val, &sphere, lip.val);

      // Point the custom user_pointer of p4est to refinement object
      p4est->user_pointer = &sp;

      if(!print_iter.val)
      {
        // Refine *recursively*
        my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);

        // Partition, create ghosts, create nodes
        my_p4est_partition(p4est, P4EST_FALSE, NULL);
        ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
        nodes = my_p4est_nodes_new(p4est, ghost);

        // Output to vtk
        my_p4est_vtk_write_all(p4est, nodes, ghost,
                               P4EST_TRUE, P4EST_TRUE,
                               0, 0, "visualization_0");
      }
      else
      {
        // Refine *non-recursively* in succesive iterations
        for(unsigned int iter=0; iter<lmax.val; ++iter)
        {
          my_p4est_refine(p4est, P4EST_FALSE, refine_levelset_cf, NULL);

          // Partition at each iteration
          my_p4est_partition(p4est, P4EST_FALSE, NULL);

          // Create ghosts and nodes
          if(iter>0){ p4est_nodes_destroy(nodes); p4est_ghost_destroy(ghost); }
          ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
          nodes = my_p4est_nodes_new(p4est, ghost);

          // Output each iteration to vtk
          char name[1024];
          sprintf(name, "visualization_%1d", iter);
          my_p4est_vtk_write_all(p4est, nodes, ghost,
                                 P4EST_TRUE, P4EST_TRUE,
                                 0, 0, name);
        }
      }

      break;
    }

    case 2: /*------ CONTINUOUS FUNCTION: INTERFACE AND UNIFORM BAND ---------------------------------------------------*/
    {
      throw std::invalid_argument("Not implemented yet");
    }

    case 3: /*------ TAGGING QUADRANTS: INTERFACE ----------------------------------------------------------------------*/
    {
      // Partition, create ghosts, create nodes
      my_p4est_partition(p4est, P4EST_FALSE, NULL);
      ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
      nodes = my_p4est_nodes_new(p4est, ghost);

      // Sample a level-set function at nodes
      Vec phi;
      ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
      sample_cf_on_nodes(p4est, nodes, sphere, phi);

      // Output the grid and data
      double *phi_p;

      // Declare the continuous-function refinement object
      splitting_criteria_tag_t sp(lmin.val, lmax.val, lip.val);

      // Point the custom user_pointer of p4est to refinement object
      p4est->user_pointer = &sp;

      bool grid_is_changing = true;
      unsigned int iter=0;
      while(grid_is_changing)
      {
        // Refine grid
        ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
        grid_is_changing = sp.refine(p4est, nodes, phi_p);
        ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

        if(grid_is_changing)
        {
          // Partition, create ghosts, create nodes
          my_p4est_partition(p4est, P4EST_FALSE, NULL);
          p4est_ghost_destroy(ghost); ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL);
          p4est_nodes_destroy(nodes); nodes = my_p4est_nodes_new(p4est, ghost);

          // Re-sample the level-set function (since the grid has changed)
          ierr = VecDestroy(phi); CHKERRXX(ierr);
          ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
          sample_cf_on_nodes(p4est, nodes, sphere, phi);

          // Output the grid and data
          if(print_iter.val)
          {
            char name[1024];
            sprintf(name, "visualization_%1d", iter);
            ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
            my_p4est_vtk_write_all(p4est, nodes, ghost,
                                   P4EST_TRUE, P4EST_TRUE,
                                   1, 0, name,
                                   VTK_POINT_DATA, "phi", phi_p);
            ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
          }

          // Keep track of the iterations
          iter++;
          if(iter>lmax.val)
          {
            ierr = PetscPrintf(mpi.comm(), "[WARNING:] The grid update did not converge.");
            break;
          }
        }

      }

      // Print the final grid if we did not print each iteration
      if(!print_iter.val)
      {
        ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
        my_p4est_vtk_write_all(p4est, nodes, ghost,
                               P4EST_TRUE, P4EST_TRUE,
                               1, 0, "visualization_0",
                               VTK_POINT_DATA, "phi", phi_p);
      }

      // Destry the dynamically allocated Vec
      ierr = VecDestroy(phi); CHKERRXX(ierr);

      break;
    }

    default: throw std::invalid_argument("Invalid refinement method");
  }

  // Destroy the dynamically allocated structures
  p4est_nodes_destroy   (nodes);
  p4est_ghost_destroy   (ghost);
  p4est_destroy         (p4est);
  my_p4est_brick_destroy(conn, &brick);

  // Stop and print global timer
  w.stop();
  w.read_duration();
}

