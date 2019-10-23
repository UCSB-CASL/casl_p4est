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
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_trajectory_of_point.h>


#include <src/my_p4est_semi_lagrangian.h>


//#include <src/my_p4est_poisson_nodes_mls.h>
//#include <src/my_p4est_poisson_nodes.h>
//#include <src/my_p4est_interpolation_nodes.h>
//#include <src/my_p4est_navier_stokes.h>



#include <src/my_p4est_macros.h>
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8esT_l_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_level_set.h>

#include <src/my_p8esT_l_node_neighbors.h>
#include <src/my_p8est_macros.h>
#endif

#include <src/Parser.h>
#include <src/casl_math.h>
#include <src/petsc_compatibility.h>


using namespace std;

// Examples to run:
int example_ = 2;  // 0 - Ice cube melting in water, 1 - Frank sphere, 2 - water solidifying around cooled cylinder

int method_ = 0; // 0 - Backward Euler, 1 - Crank Nicholson

bool elyce_laptop = false; // Set to true if working on laptop --> changes the output path
// ---------------------------------------
// Define geometry:
// ---------------------------------------
double xmin; double xmax;
double ymin; double ymax;

int nx, ny;
int px, py;
double box_size; // equivalent width (in x) in meters

// For frank sphere:
double s0;
double T_inf;

// For ice cube:
double r0;
double Twall;
double Tinterface;
double back_wall_temp_flux;

// For solidifying ice problem:
double r_cyl;
double T_cyl;

// For surface tension:
double sigma;

void set_geometry(){

      xmin = -1.0; xmax = 1.0;
      ymin = -1.0; ymax = 1.0;

      nx = 1;
      ny = 1;

      px = 0;
      py = 0;
      r0 = 0.50;

}

bool refine_by_phi1;
bool refine_by_phi2;
bool refine_by_s1;
bool refine_by_s2;

double s1_coarsen_criteria;
double s1_refine_criteria;

double s2_coarsen_criteria;
double s2_refine_criteria;

compare_option_t  s1_coarsen_comp, s1_refine_comp, s2_coarsen_comp, s2_refine_comp;

compare_diagonal_option_t  s1_coarsen_diag_comp, s1_refine_diag_comp, s2_coarsen_diag_comp, s2_refine_diag_comp;

int num_fields; // Number of scalar fields to refine by

void set_refinement_options(double dxyz_smallest_min){
  refine_by_phi1 = true;
  refine_by_phi2 = false;
  refine_by_s1 = true;
  refine_by_s2 = true;

  num_fields = 2;

  // s1 coarsen options:
  s1_coarsen_criteria = 0.5;
  s1_coarsen_comp = GREATER_THAN;
  s1_coarsen_diag_comp = ABSOLUTE;

  // s1 refine options:
  s1_refine_criteria = dxyz_smallest_min*1.0;
  s1_refine_comp = LESS_THAN;
  s1_refine_diag_comp = ABSOLUTE;

  // s2 coarsen options:
  s2_coarsen_criteria = 0.5;
  s2_coarsen_comp = GREATER_THAN;
  s2_coarsen_diag_comp = ABSOLUTE;

  // s2 refine options:
  s2_refine_criteria = dxyz_smallest_min*1.0;
  s2_refine_comp = LESS_THAN;
  s2_refine_diag_comp = ABSOLUTE;

}
// ---------------------------------------
// Grid refinement:
// ---------------------------------------
int lmin = 4;
int lmax = 7;
double lip = 1.75;

// --------------------------------------------------------------------------------------------------------------
// LEVEL SET FUNCTIONS:
// --------------------------------------------------------------------------------------------------------------
struct LEVEL_SET_1 : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
      return r0 - sqrt(SQR(x) + SQR(y));
  }
} level_set_1;

struct LEVEL_SET_2 : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
      return (r0/2.0) - sqrt(SQR(x) + SQR(y));
  }
} level_set_2;

struct SINUSOID : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
      return sin(PI*x)*cos(PI*y);
  }
} sinusoid;

struct SINUSOID2 : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
      return sin(5.0*PI*x)*cos(5.0*PI*y);
  }
} sinusoid2;


void save_vtk(p4est_t* p4est, p4est_nodes_t* nodes, p4est_ghost_t* ghost, char filename[], vec_and_ptr_t phi_1, vec_and_ptr_t phi_2, vec_and_ptr_t s1, vec_and_ptr_t s2){
  // Get necessary arrays:

  phi_1.get_array();
  phi_2.get_array();

  s1.get_array();

  s2.get_array();

  // Write out the files to vtk:

  my_p4est_vtk_write_all(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,4,0,filename,
                         VTK_POINT_DATA, "phi1", phi_1.ptr,
                         VTK_POINT_DATA, "phi2", phi_2.ptr,
                         VTK_POINT_DATA, "sinusoid1",s1.ptr,
                         VTK_POINT_DATA, "sinusoid2",s2.ptr);


  // Restore necessary arrays:
  phi_1.restore_array();
  phi_2.restore_array();
  s1.restore_array();
  s2.restore_array();

}
// --------------------------------------------------------------------------------------------------------------
// BEGIN MAIN OPERATION:
// --------------------------------------------------------------------------------------------------------------
int main(int argc, char** argv) {
  // prepare parallel enviroment
  mpi_environment_t mpi;
  mpi.init(argc, argv);
  PetscErrorCode ierr;
  PetscViewer viewer;
  int mpi_ret; // Check mpi issues

  // stopwatch
  parStopWatch w;
  w.start("Running example: refine and coarsen with provided criteria \n \n ");

 // -----------------------------------------------
  // Set up grid structure and partition:
  // -----------------------------------------------
  // p4est variables
  p4est_t*              p4est;
  p4est_nodes_t*        nodes;
  p4est_ghost_t*        ghost;
  p4est_connectivity_t* conn;
  my_p4est_brick_t      brick;

  p4est_t*              p4est_np1;
  p4est_nodes_t*        nodes_np1;
  p4est_ghost_t*        ghost_np1;

  // domain size information
  set_geometry();
  const int n_xyz[]      = { nx,  ny,  0};
  const double xyz_min[] = {xmin, ymin, 0};
  const double xyz_max[] = {xmax,  ymax,  0};
  const int periodic[]   = { px,  py,  0};

  PetscPrintf(mpi.comm(),"nx and ny are: %d, %d \n"
                         "xmin and xmax are: %f, %f \n"
                         "ymin and ymanx are: %f, %f \n"
                         "px and py are : %d , %d \n \n \n ",
                         nx,ny,xmin,xmax,ymin,ymax,px,py);
  // Initialize output file name:
  int ex_no = 0;
  char out_dir[1000];
  sprintf(out_dir,"/home/elyce/workspace/projects/multialloy_with_fluids/solidif_with_fluids_output");
  char file_name[1000];
  sprintf(file_name,"%s/refinement_test_ex_%d",out_dir,ex_no);
  PetscPrintf(mpi.comm(),"Output file is %s \n",file_name);

  // -----------------------------------------------
  // Create the grid:
  // -----------------------------------------------
  conn = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic); // same as Daniil

  // create the forest
  p4est = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL); // same as Daniil

  // refine based on distance to a level-set
  splitting_criteria_cf_t sp(lmin, lmax, &level_set_1,lip);

  p4est->user_pointer = &sp; // save the pointer to the forst splitting criteria

  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL); // refine the grid according to the splitting criteria

  // partition the forest
  my_p4est_partition(p4est, P4EST_FALSE, NULL);                  // partition the forest, do not allow for coarsening --> Daniil does not allow (use P4EST_FALSE)

  // create ghost layer
  ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL); // same

  // Expand ghost layer -- FOR NAVIER STOKES:
  my_p4est_ghost_expand(p4est,ghost);

  // create node structure
  nodes = my_p4est_nodes_new(p4est, ghost); //same

  // Create hierarchy
  my_p4est_hierarchy_t *hierarchy = new my_p4est_hierarchy_t(p4est, ghost, &brick);

  // Get neighbors
  my_p4est_node_neighbors_t *ngbd = new my_p4est_node_neighbors_t(hierarchy, nodes);

  ngbd->init_neighbors();

  // -----------------------------------------------
  // Get the LSF at the nodes:
  // -----------------------------------------------
  // LSF:
  vec_and_ptr_t phi_1;
  phi_1.create(p4est,nodes);
  sample_cf_on_nodes(p4est,nodes,level_set_1,phi_1.vec);

  vec_and_ptr_t phi_2;
  phi_2.create(p4est,nodes);
  sample_cf_on_nodes(p4est,nodes,level_set_2,phi_2.vec);

  // -----------------------------------------------
  // Get the sinusoid function on the nodes:
  // -----------------------------------------------
  vec_and_ptr_t s_1;
  s_1.create(p4est,nodes);
  sample_cf_on_nodes(p4est,nodes,sinusoid,s_1.vec);

  vec_and_ptr_t s_2;
  s_2.create(p4est,nodes);
  sample_cf_on_nodes(p4est,nodes,sinusoid2,s_2.vec);

//  int phi_1_s, phi_2_s, s_1_s,s_2_s;
//  VecGetSize(phi_1.vec,&phi_1_s);
//  VecGetSize(phi_2.vec,&phi_2_s);
//  VecGetSize(s_1.vec,&s_1_s);
//  VecGetSize(s_2.vec,&s_2_s);
//  PetscPrintf(mpi.comm(),"INITIAL FUNCTIONS AND GRID SIZES : ------------------------------ \n \n \n ");
//  PetscPrintf(mpi.comm(),"The sizes are : %d, %d , %d, %d \n \n ",phi_1_s,phi_2_s,s_1_s,s_2_s);

//  int no_nodes = nodes->num_owned_indeps;
//  int global_nodes = 0;
//  MPI_Allreduce(&no_nodes,&global_nodes,1,MPI_INT,MPI_SUM,mpi.comm());
//  PetscPrintf(mpi.comm(),"Number of nodes in the initial grid: %d \n \n",global_nodes);

  // -----------------------------------------------
  // Save the initial grid and values, before trying any refinement:
  // -----------------------------------------------
  save_vtk(p4est,nodes,ghost,file_name,phi_1,phi_2,s_1,s_2);
  PetscPrintf(mpi.comm(),"Saved initial state to vtk \n \n ");


  // Make a copy of the initial grid to refine and coarsen:
  p4est_np1 = my_p4est_copy(p4est,P4EST_FALSE);
  ghost_np1 = my_p4est_ghost_new(p4est_np1,P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est_np1,ghost_np1); // CONFIRMED YES YOU NEED TO DO THIS
  nodes_np1 = my_p4est_nodes_new(p4est_np1,ghost_np1);
  my_p4est_hierarchy_t* hierarchy_np1 = new my_p4est_hierarchy_t(p4est_np1,ghost_np1,&brick);
  my_p4est_node_neighbors_t* ngbd_np1 = new my_p4est_node_neighbors_t(hierarchy_np1,nodes_np1);
  ngbd_np1->init_neighbors();

  // -----------------------------------------------
  // Example:
  // -----------------------------------------------
  // Get smallest grid cells:
  double dxyz_smallest[P4EST_DIM];
  dxyz_min(p4est_np1,dxyz_smallest);

  double dxyz_min_small = 10.0;
  foreach_dimension(d){
    dxyz_min_small = min(dxyz_min_small,dxyz_smallest[d]);
  }
  PetscPrintf(mpi.comm(),"Initial minimum grid size is %0.4f \n",dxyz_min_small);

  // Set the refinement options:
  set_refinement_options(dxyz_min_small);

  // Begin defining the refinement criteria:
  // Order is {coarsen_field_1, refine_field_1, coarsen_field_2, refine_field_2, ...}
  std::vector<compare_option_t> compare_opn;
  std::vector<compare_diagonal_option_t> diag_opn;
  std::vector<double> criteria;

  // Set the instructions:
  if(refine_by_s1) {
      compare_opn.push_back(s1_coarsen_comp);
      diag_opn.push_back(s1_coarsen_diag_comp);
      criteria.push_back(s1_coarsen_criteria);

      compare_opn.push_back(s1_refine_comp);
      diag_opn.push_back(s1_refine_diag_comp);
      criteria.push_back(s1_refine_criteria);

      PetscPrintf(mpi.comm(),"Will coarsen if fxn is greater than %0.4f \n",s1_coarsen_criteria);
      PetscPrintf(mpi.comm(),"Will refine if fxn is less than %0.4f \n",s1_refine_criteria);


    }
  if(refine_by_s2){
      compare_opn.push_back(s2_coarsen_comp);
      diag_opn.push_back(s2_coarsen_diag_comp);
      criteria.push_back(s2_coarsen_criteria);

      compare_opn.push_back(s2_refine_comp);
      diag_opn.push_back(s2_refine_diag_comp);
      criteria.push_back(s2_refine_criteria);
    }

  // Create the new criteria:
  splitting_criteria_tag_t sp1(sp.min_lvl,sp.max_lvl,sp.lip);

  // Refine and coarsen the grid according to the new criteria:

  vec_and_ptr_t phi_1_new;
  vec_and_ptr_t phi_2_new;
  vec_and_ptr_t s_1_new;

  vec_and_ptr_t phi_eff_old;
  vec_and_ptr_t phi_eff_new;

  // Create vector to pass fields to refine by:
  Vec fields_old;
  Vec fields_new;

  VecCreateGhostNodesBlock(p4est,nodes,num_fields,&fields_old);
  VecCreateGhostNodesBlock(p4est_np1,nodes_np1,num_fields,&fields_new);

  int num_fields_added = 0;

  // Create vectors of values on the new grid -- these values will be updated onto whatever current grid exists during the iteration of grid refinement
  if(refine_by_phi1 && !refine_by_phi2){
      phi_1_new.create(p4est_np1,nodes_np1);


      VecCopyGhost(phi_1.vec,phi_1_new.vec);
    }

  if(refine_by_phi2 && !refine_by_phi1){
      phi_2_new.create(p4est_np1,nodes_np1);
      VecCopyGhost(phi_2.vec,phi_2_new.vec);
    }

  if(refine_by_phi1 && refine_by_phi2){
      // Get an effective phi of both the LSF's to refine by, if we want to refine by both
      phi_eff_old.create(p4est,nodes);
      phi_eff_new.create(p4est,nodes);

      phi_eff_old.get_array();
      phi_1.get_array();
      phi_2.get_array();

      foreach_node(n,nodes){
        phi_eff_old.ptr[n] = min(phi_1.ptr[n],phi_2.ptr[n]);
      }

      phi_eff_old.restore_array();
      phi_1.restore_array();
      phi_2.restore_array();

      VecCopyGhost(phi_eff_old.vec,phi_eff_new.vec);
    }

  std::vector<double*> refinement_fields;

  if(refine_by_s1) {
      s_1_new.create(p4est_np1,nodes_np1);
      VecCopyGhost(s_1.vec,s_1_new.vec);

      s_1.get_array();
      refinement_fields.push_back(s_1.ptr);
      s_1.restore_array();
      num_fields_added++;
    }

  if(refine_by_s2){
      s_2.get_array();
      refinement_fields.push_back(s_2.ptr);
      s_2.restore_array();
      num_fields_added++;
    }

  // Create block vector of the fields:
  double* fields_old_ptr;
  VecGetArray(fields_old,&fields_old_ptr);
  foreach_node(n,nodes){
    for(unsigned long i = 0; i<num_fields; i++){
        fields_old_ptr[n*num_fields + i] = refinement_fields[i][n];
//        PetscPrintf(mpi.comm(),"Index %d in fields_old is set as %0.3f for refinement field %d \n \n ",n*num_fields + i, fields_old_ptr[n*num_fields + i],i);
      }
  }

  VecRestoreArray(fields_old,&fields_old_ptr);

  // Check fields_old to see if it has the correct values:
  double xyz_old[P4EST_DIM];
  VecGetArray(fields_old,&fields_old_ptr);
  phi_1.get_array();
  s_1.get_array();
  s_2.get_array();


  double err_s1, err_s2, err_phi1, err_fields;
  double err_s1_g = 0, err_s2_g = 0, err_phi1_g=0, err_fields_g = 0;
  foreach_node(n,nodes){
    node_xyz_fr_n(n,p4est,nodes,xyz_old);
    err_s1 = max(err_s1,s_1.ptr[n] - sinusoid(xyz_old[0],xyz_old[1]));
    err_s2 = max(err_s2,s_2.ptr[n] - sinusoid2(xyz_old[0],xyz_old[1]));
    err_phi1 = max(err_phi1,phi_1.ptr[n] - level_set_1(xyz_old[0],xyz_old[1]));
    for(int i=0;i<num_fields;i++){
        if(i ==0) err_fields = max(err_fields,fields_old_ptr[n*num_fields + i] - sinusoid(xyz_old[0],xyz_old[1]));
        else err_fields = max(err_fields,fields_old_ptr[n*num_fields + i] - sinusoid2(xyz_old[0],xyz_old[1]));
      }
  }
  VecRestoreArray(fields_old,&fields_old_ptr);
  phi_1.restore_array();
  s_1.restore_array();
  s_2.restore_array();

  double errors[4] = {err_s1,err_s2,err_phi1,err_fields};
  double global_errors[4] = {err_s1_g,err_s2_g,err_phi1_g,err_fields_g};
  MPI_Allreduce(&errors,&global_errors,4,MPI_INT,MPI_MAX,mpi.comm());

  PetscPrintf(mpi.comm()," Initial Errors are as follows: \n "
                         "On s1: %0.3e \n "
                         "On s2: %0.3e \n"
                         "On phi1: %0.3e \n"
                         "On fields: %0.3e \n \n",global_errors[0],global_errors[1],global_errors[2],global_errors[3]);


//  PetscPrintf(mpi.comm(),"Sets the blocked values \n");

//  // Check sizes:
//  int phi_size1;
//  int s_size1;
//  int numnodes1;
//  VecGetSize(phi_1_new.vec,&phi_size1);
//  VecGetSize(s_1_new.vec,&s_size1);
//  numnodes1 = nodes_np1->num_owned_indeps;

//  int numnodes_global1 = 0;

//  MPI_Allreduce(&numnodes1,&numnodes_global1,1,MPI_INT,MPI_SUM,mpi.comm());

//  PetscPrintf(mpi.comm(),"SIZES AFTER COPYING OVER FIELDS: ------------------------------- \n \n");
//  PetscPrintf(mpi.comm(),"Sizes of phi and s1 are %d and %d \n \n ",phi_size1,s_size1);
//  PetscPrintf(mpi.comm(),"Total nodes:  %d \n \n ",numnodes_global1);

  // Make sure we added as many fields as we said we were:
  PetscPrintf(mpi.comm(),"Num fields : %d, Num fields added : %d \n",num_fields,num_fields_added);
  P4EST_ASSERT(num_fields_added == num_fields);

  // Save new version of fields to update during grid iteration:
  VecCopyGhost(fields_old,fields_new);

  // Start refinement procedure for the grid:
  bool is_grid_changing = true;
  int no_grid_changes = 0;
  int intermediate_no=0;

  while(is_grid_changing){

//      // Check sizes:
//      int phi_size;
//      int s_size;
//      int numnodes;
//      VecGetSize(phi_1_new.vec,&phi_size);
//      VecGetSize(s_1_new.vec,&s_size);
//      numnodes = nodes_np1->num_owned_indeps;
//      int numnodes_global = 0;

//      MPI_Allreduce(&numnodes,&numnodes_global,1,MPI_INT,MPI_SUM,mpi.comm());

//      PetscPrintf(mpi.comm(),"SIZES BEFORE GRID CHANGE: ------------------------------- \n \n");
//      PetscPrintf(mpi.comm(),"Sizes of phi and s1 are %d and %d \n \n ",phi_size,s_size);
//      PetscPrintf(mpi.comm(),"Total nodes:  %d \n \n ",numnodes_global);

      // Get array for fields:
      double *fields_new_p;
      VecGetArray(fields_new,&fields_new_p);

      if(refine_by_phi1 && !refine_by_phi2) {
          phi_1_new.get_array();
//          s_1_new.get_array();
//          foreach_node(n,nodes_np1){
//            PetscPrintf(mpi.comm(),"s1new value is %0.3e \n",s_1_new.ptr[n]);
//          }

          //is_grid_changing = sp1.refine_and_coarsen(p4est_np1,nodes_np1,phi_1_new.ptr,num_fields,s_1_new.ptr,criteria,compare_opn,diag_opn);
          is_grid_changing = sp1.refine_and_coarsen(p4est_np1,nodes_np1,phi_1_new.ptr,num_fields,fields_new_p,criteria,compare_opn,diag_opn);

//          s_1_new.restore_array();
          phi_1_new.restore_array();
        }
      if(refine_by_phi2 && !refine_by_phi1) {phi_2_new.get_array();}
      if(refine_by_phi1 && refine_by_phi2) {phi_eff_new.get_array();}

      // Restore array for fields:
      VecRestoreArray(fields_new,&fields_new_p);

      PetscPrintf(mpi.comm(),"Did the grid change? --> %s \n \n ", is_grid_changing? "yes" : "no");

      if (is_grid_changing){
          no_grid_changes++;
          PetscPrintf(mpi.comm(),"Grid changed (%d time(s) ) \n \n ",no_grid_changes);

          // Repartition the grid:
          my_p4est_partition(p4est_np1,P4EST_TRUE,NULL);

          // Reset the grid:
          p4est_ghost_destroy(ghost_np1); ghost_np1 = my_p4est_ghost_new(p4est_np1,P4EST_CONNECT_FULL);
          p4est_ghost_expand(p4est_np1,ghost_np1);
          p4est_nodes_destroy(nodes_np1); nodes_np1 = my_p4est_nodes_new(p4est_np1,ghost_np1);

//          // Check the number of nodes after grid change:
//          no_nodes = nodes_np1->num_owned_indeps;
//          global_nodes = 0;
//          MPI_Allreduce(&no_nodes,&global_nodes,1,MPI_INT,MPI_SUM,mpi.comm());
//          PetscPrintf(mpi.comm(),"Number of nodes AFTER the grid change: %d \n \n",global_nodes);

          // Reset the values we need for refinement:
          VecDestroy(fields_new);
          VecCreateGhostNodesBlock(p4est_np1,nodes_np1,num_fields,&fields_new);
          s_1_new.destroy();
          s_1_new.create(p4est_np1,nodes_np1);

          if(refine_by_phi1 && !refine_by_phi2){
              phi_1_new.destroy();
              phi_1_new.create(p4est_np1,nodes_np1);
            }
          if(refine_by_phi2 && !refine_by_phi1){}
          if(refine_by_phi1 && refine_by_phi2){}

         // Interpolate values of phi and criteria from the old grid to current iteration of new grid:
          my_p4est_interpolation_nodes_t interp(ngbd);
          my_p4est_interpolation_nodes_t interp_block(ngbd);
          interp_block.set_input(&fields_old,quadratic_non_oscillatory_continuous_v2,num_fields);

          double xyz[P4EST_DIM];
          foreach_node(n,nodes_np1){
            node_xyz_fr_n(n,p4est_np1,nodes_np1,xyz);
            interp.add_point(n,xyz);
            interp_block.add_point(n,xyz);
          }
          interp.set_input(phi_1.vec,quadratic_non_oscillatory_continuous_v2);
          interp.interpolate(phi_1_new.vec);

          interp_block.interpolate(fields_new);

          interp.set_input(s_1.vec,quadratic_non_oscillatory_continuous_v2);
          interp.interpolate(s_1_new.vec);

//          interp.set_input(&fields_old,quadratic_non_oscillatory_continuous_v2,num_fields);
//          interp.interpolate(fields_new);

        } // End of "if grid changing"
      // Check sizes:
//      int phi_size2;
//      int s_size2;
//      int numnodes2;
//      VecGetSize(phi_1_new.vec,&phi_size2);
//      VecGetSize(s_1_new.vec,&s_size2);
//      numnodes2 = nodes_np1->num_owned_indeps;
//      int numnodes_global2 = 0;

//      MPI_Allreduce(&numnodes2,&numnodes_global2,1,MPI_INT,MPI_SUM,mpi.comm());

//      PetscPrintf(mpi.comm(),"SIZES AFTER GRID CHANGE: ------------------------------- \n \n");
//      PetscPrintf(mpi.comm(),"Sizes of phi and s1 are %d and %d \n \n ",phi_size2,s_size2);
//      PetscPrintf(mpi.comm(),"Total nodes:  %d \n \n ",numnodes_global2);


//      sprintf(file_name,"%s/refinement_test_ex_intermediate_%d",out_dir,intermediate_no);
//      phi_1_new.get_array();
//      s_1_new.get_array();
//      my_p4est_vtk_write_all(p4est_np1,nodes_np1,ghost_np1,P4EST_TRUE,P4EST_TRUE,2,0,file_name,
//                             VTK_POINT_DATA, "phi1",phi_1_new.ptr,
//                             VTK_POINT_DATA, "s1",s_1_new.ptr);
//      phi_1_new.restore_array();
//      s_1_new.restore_array();
//      PetscPrintf(mpi.comm(),"Saved intermediate grid file \n");
      intermediate_no++;
    is_grid_changing = false;
    } // End of "while grid changing"


  if(refine_by_s1 && refine_by_s2){
      double error = 0.0;
      double error_fields = 0.0;
      double error_phi = 0.0;
      double xyz1[P4EST_DIM];
      double* fields_new_p;
      VecGetArray(fields_new,&fields_new_p);
      phi_1_new.get_array();
      s_1_new.get_array();
      foreach_node(n,nodes_np1){
        node_xyz_fr_n(n,p4est_np1,nodes_np1,xyz1);
        error = max(error,s_1_new.ptr[n] - sinusoid(xyz1[0],xyz1[1]));

        for(int i=0;i<num_fields;i++){
            if(i==0) error_fields = max(error_fields,fields_new_p[n*num_fields + i] - sinusoid(xyz1[0],xyz1[1]));
            else error_fields = max(error_fields,fields_new_p[n*num_fields + i] - sinusoid2(xyz1[0],xyz1[1]));
          }

        error_phi = max(error_phi,phi_1_new.ptr[n] - level_set_1(xyz1[0],xyz1[1]));
      }
      phi_1_new.restore_array();
      s_1_new.restore_array();
      VecRestoreArray(fields_new,&fields_new_p);
      double global_error = 0.0;
      double global_fields_error = 0.0;
      double global_phi_error = 0.0;
      MPI_Allreduce(&error,&global_error,1,MPI_DOUBLE,MPI_MAX,mpi.comm());
      MPI_Allreduce(&error_fields,&global_fields_error,1,MPI_DOUBLE,MPI_MAX,mpi.comm());

      MPI_Allreduce(&error_phi,&global_phi_error,1,MPI_DOUBLE,MPI_MAX,mpi.comm());

      PetscPrintf(mpi.comm(),"Error on the interpolated sinusoid is %0.3e \n \n",global_error);
      PetscPrintf(mpi.comm(),"Error on the interpolated fields is %0.3e \n \n",global_fields_error);

      PetscPrintf(mpi.comm(),"Error on the interpolated LSF is %0.3e \n \n",global_phi_error);

    }

  // Interpolate remaining onto new grid:
  my_p4est_interpolation_nodes_t interp(ngbd);
  PetscPrintf(mpi.comm(),"Begin interpolation onto the new grid \n ");
  double xyz[P4EST_DIM];
  foreach_node(n,nodes_np1){
    interp.add_point(n,xyz);
  }
  // Make vectors to hold new values:
  phi_1_new.destroy(); phi_2_new.destroy();
  phi_1_new.create(p4est_np1,nodes_np1);
  phi_2_new.create(phi_1_new.vec);
  s_1_new.create(phi_1_new.vec);
  vec_and_ptr_t s_2_new(phi_1_new.vec);

  //Interpolate all the fields:
  interp.set_input(phi_1.vec,quadratic_non_oscillatory_continuous_v2); interp.interpolate(phi_1_new.vec);
  interp.set_input(phi_2.vec,quadratic_non_oscillatory_continuous_v2); interp.interpolate(phi_2_new.vec);
  interp.set_input(s_1.vec,quadratic_non_oscillatory_continuous_v2); interp.interpolate(s_1_new.vec);
  interp.set_input(s_2.vec,quadratic_non_oscillatory_continuous_v2); interp.interpolate(s_2_new.vec);

  PetscPrintf(mpi.comm(),"Finish interpolation onto the new grid \n \n ");

  // Update the grid accordingly:
  p4est_destroy(p4est); p4est = p4est_np1;
  p4est_ghost_destroy(ghost); ghost = ghost_np1;
  p4est_nodes_destroy(nodes); nodes = nodes_np1;
  hierarchy->update(p4est,ghost);
  ngbd->update(hierarchy,nodes);

//  //int phi_1_s, phi_2_s, s_1_s,s_2_s;
//  VecGetSize(phi_1_new.vec,&phi_1_s);
//  VecGetSize(phi_2_new.vec,&phi_2_s);
//  VecGetSize(s_1_new.vec,&s_1_s);
//  VecGetSize(s_2_new.vec,&s_2_s);

//  PetscPrintf(mpi.comm(),"The sizes are : %d, %d , %d, %d \n \n ",phi_1_s,phi_2_s,s_1_s,s_2_s);

//  no_nodes = nodes->num_owned_indeps;
//  global_nodes = 0;
//  MPI_Allreduce(&no_nodes,&global_nodes,1,MPI_INT,MPI_SUM,mpi.comm());

//  PetscPrintf(mpi.comm(),"Number of nodes in the new grid: %d \n \n",global_nodes);

  // Visualize:
  ex_no++;
  sprintf(file_name,"%s/refinement_test_ex_%d",out_dir,ex_no);
  PetscPrintf(mpi.comm(),"New filename is %s \n",file_name);
  save_vtk(p4est_np1,nodes_np1,ghost_np1,file_name,phi_1_new,phi_2_new,s_1_new,s_2_new);

//   Destroy vectors:
  VecDestroy(fields_old);
  VecDestroy(fields_new);

  // -----------------------------------------------
  // Ex. 2: Refine grid around two LSFs and the sinusoid
  // -----------------------------------------------
  // Get the "effective LSF"


  // Define the criteria:


  // Refine the grid:


  // Interpolate onto new grid:


  // Visualize:





  // -----------------------------------------------
  // Ex. 3: Refine the grid around first LSF, sinusoid, and paraboloid
  // -----------------------------------------------
  // Define the criteria:


  // Refine the grid:


  // Interpolate onto new grid:


  // Visualize:




  // -----------------------------------------------
  // Ex. 4: Refine grid around two LSFs, the sinusoid, and the paraboloid
  // -----------------------------------------------
  // Define the criteria:


  // Refine the grid:


  // Interpolate onto new grid:


  // Visualize:



  // -----------------------------------------------
  // Ex. 5:
  // -----------------------------------------------
  // Define the criteria:


  // Refine the grid:


  // Interpolate onto new grid:


  // Visualize:

  // -----------------------------------------------
  // Destroy the grid structures now that they are no longer in use:
  // -----------------------------------------------
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy      (p4est);

//  p4est_nodes_destroy(nodes_np1);
//  p4est_ghost_destroy(ghost_np1);
//  p4est_destroy      (p4est_np1);

  my_p4est_brick_destroy(conn, &brick);

  w.stop(); w.read_duration();


}

