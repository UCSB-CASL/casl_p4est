/* 
 * Title: multialloy_with_fluids
 * Description:
 * Author: Elyce
 * Date Created: 08-06-2019
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
#include <src/petsc_compatibility.h>


using namespace std;

// Create a class for the level set function:

struct LEVEL_SET : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    return 0.5 - sqrt(SQR(x) + SQR(y));
  }
} level_set;

struct u_t : CF_DIM
{ double operator() (double x, double y) const{
  return 1.0;
  }

} u;

struct v_t: CF_DIM{
  double operator()(double x, double y) const
  {
    return 1.0;
  }
} v;

int main(int argc, char** argv) {


  // prepare parallel enviroment
  mpi_environment_t mpi;
  mpi.init(argc, argv);
  PetscErrorCode ierr;
  PetscViewer viewer;

  if (mpi.rank() ==0){
    std::cout<<"Elyce's first example is going! \n"<<std::endl;
  }

  // stopwatch
  parStopWatch w;
  w.start("Running example: multialloy_with_fluids");

  // -----------------------------------------------
  // Set up grid structure and partition:
  // -----------------------------------------------
  // p4est variables
  p4est_t*              p4est;
  p4est_nodes_t*        nodes;
  p4est_ghost_t*        ghost;
  p4est_connectivity_t* conn;
  my_p4est_brick_t      brick;

  // domain size information
  const int n_xyz[]      = { 1,  1,  1};
  const double xyz_min[] = {-1, -1, -1};
  const double xyz_max[] = { 1,  1,  1};
  const int periodic[]   = { 0,  0,  0};
  conn = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic); // same as Daniil

  // create the forest
  p4est = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL); // same as Daniil

  // refine based on distance to a level-set

  splitting_criteria_cf_t sp(3, 8, &level_set); // same as Daniil, minus lipschitz
  p4est->user_pointer = &sp; // save the pointer to the forst splitting criteria
  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL); // refine the level set according to the splitting criteria

  // partition the forest
  my_p4est_partition(p4est, P4EST_TRUE, NULL); // partition the forest but allow for coarsening --> Daniil does not allow (use P4EST_FALSE)

  // create ghost layer
  ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL); // same

  // create node structure
  nodes = my_p4est_nodes_new(p4est, ghost); //same

  // -----------------------------------------------
  // Initialize the Level Set function:
  // -----------------------------------------------
  Vec phi;

  ierr = VecCreateGhostNodes(p4est,nodes,&phi); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est,nodes,level_set,phi);

  // -----------------------------------------------
  // Initialize the Velocity field:
  // -----------------------------------------------
  Vec vel_n[P4EST_DIM];
  const CF_DIM *vel_cf[P4EST_DIM] = {&u, &v};

  for (int dir = 0; dir<P4EST_DIM; dir++){
      ierr = VecDuplicate(phi,&vel_n[dir]);CHKERRXX(ierr);
      sample_cf_on_nodes(p4est,nodes, *vel_cf[dir], vel_n[dir]);
    }

  // -----------------------------------------------
  // Initialize the output file:
  // -----------------------------------------------
  int out_idx = 0;
  char outdir[1000];
  sprintf(outdir,"/home/elyce/workspace/projects/build/advecting_a_LSF/output/advecting_a_LSF_snapshot_%d",out_idx);
  // -----------------------------------------------
  // Get local array to write initial LSF and velocity fields to vtk:
  // -----------------------------------------------

  double *phi_ptr;
  ierr = VecGetArray(phi,&phi_ptr); CHKERRXX(ierr); // Gets the part of the array which is on this processor so we can write it out

  // Get velocity data:
  double *vel_p[P4EST_DIM];
  for (int dir=0; dir< P4EST_DIM; dir++){
    ierr = VecGetArray(vel_n[dir], &vel_p[dir]);
    }


  //ierr = VecView(phi,viewer);
  // Write out the data:
  my_p4est_vtk_write_all(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,
                         3,0,outdir,
                         VTK_POINT_DATA,"phi",phi_ptr,
                         VTK_POINT_DATA,"vx",vel_p[0],
                         VTK_POINT_DATA,"vy",vel_p[1]);


  // Restores that part of the array once we don't need it anymore
  ierr = VecRestoreArray(phi,&phi_ptr); CHKERRXX(ierr);

  for (int dir=0; dir< P4EST_DIM; dir++){
    ierr = VecRestoreArray(vel_n[dir], &vel_p[dir]);
    }

  // -----------------------------------------------
  // Time to begin stepping through time and advecting that LSF!
  // -----------------------------------------------
  double tf = 0.5;
  int tstep = 0;
  int save = 1;
  double dt = 0.1;

  for (double t = 0; t<tf; t+=dt, tstep++){
      ierr = PetscPrintf(mpi.comm(),"Iteration %d , Time: %0.2f \n",tstep,t);

      // Save data every specified amout of timesteps:
      if (tstep % save ==0){}


      // Make a copy of the grid objects for the next timestep:
      p4est_t *p4est_np1 = p4est_copy(p4est,P4EST_FALSE); // copy the grid but not the data
      p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
      p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

      // Create the semi-lagrangian object and do the advection:




    } // End of for loop through time















  // -----------------------------------------------

  // destroy the structures
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy      (p4est);
  my_p4est_brick_destroy(conn, &brick);

  w.stop(); w.read_duration();
}

