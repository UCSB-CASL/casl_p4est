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


// For ice cube:
double r0;

void set_geometry(){

      xmin = -1.0; xmax = 1.0;
      ymin = -1.0; ymax = 1.0;

      nx = 1;
      ny = 1;

      px = 0;
      py = 0;
      r0 = 0.50;

}

bool test_refine_and_coarsen = false;

bool test_update_p4est = true;

bool refine_by_s1;
bool refine_by_s2;

bool use_block;

double s1_coarsen_criteria;
double s1_refine_criteria;

double s2_coarsen_criteria;
double s2_refine_criteria;

compare_option_t  s1_coarsen_comp, s1_refine_comp, s2_coarsen_comp, s2_refine_comp;

compare_diagonal_option_t  s1_coarsen_diag_comp, s1_refine_diag_comp, s2_coarsen_diag_comp, s2_refine_diag_comp;


// ---------------------------------------
// Grid refinement:
// ---------------------------------------
int lmin = 4;
int lmax = 7;
double lip = 1.75;
const unsigned int num_fields = 2; // Number of scalar fields to refine by
const unsigned int total_num_fields = 4; // Total number of fields we are looking at (including scalar fields and LSFs )


bool refine_by_phi1 = true;
bool refine_by_phi2 = true;
void set_refinement_options(double dxyz_smallest_min){
  refine_by_s1 = true;
  refine_by_s2 = true;

  if(!refine_by_phi1 && !refine_by_phi2){
      throw std::invalid_argument("You must select at least one level set function to refine around \n"); // FOR NOW.... maybe at some point we give an option to skip phi?
    }

  use_block = false;

  // s1 coarsen options:
  s1_coarsen_criteria = 0.5;
  s1_coarsen_comp = GREATER_THAN;
  s1_coarsen_diag_comp = ABSOLUTE;

  // s1 refine options:
  s1_refine_criteria = dxyz_smallest_min*1.0;
  s1_refine_comp = LESS_THAN;
  s1_refine_diag_comp = ABSOLUTE;

  // s2 coarsen options:
  s2_coarsen_criteria = 1.0*lip;
  s2_coarsen_comp = GREATER_THAN;
  s2_coarsen_diag_comp = MULTIPLY_BY;

  // s2 refine options:
  s2_refine_criteria = 0.5*lip;
  s2_refine_comp = LESS_THAN;
  s2_refine_diag_comp = MULTIPLY_BY;

}

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

struct FUNCTION1 : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
      return sin(PI*x)*cos(PI*y);
  }
} function1;

struct FUNCTION2 : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
      return r0 - sqrt(SQR(x/1.) + SQR(y/0.5));
      //return sin(2.*PI*x)*cos(2.*PI*y);
  }
} function2;

struct U_VELOCITY : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    return 0.0;
  }
} u_vel;

struct V_VELOCITY : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    return 0.5;
  }
} v_vel;
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
                         VTK_POINT_DATA, "function1",s1.ptr,
                         VTK_POINT_DATA, "function2",s2.ptr);


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
  conn = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic);

  // create the forest
  p4est = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL);

  // refine based on distance to a level-set
  CF_DIM *lsf;
  if(refine_by_phi1) {
      lsf = &level_set_1;}
  else{
      lsf = &level_set_2;}

  splitting_criteria_cf_t sp(lmin, lmax, lsf,lip);

  p4est->user_pointer = &sp; // save the p4est user pointer to the forest splitting criteria --> we will use this to tag quadrants for refinement or coarsening

  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL);  // refine the grid according to the splitting criteria

  // Partition the forest
  my_p4est_partition(p4est, P4EST_FALSE, NULL);                  // partition the forest, do not allow for coarsening

  // Create ghost layer
  ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL); // same

  // Expand ghost layer -- FOR NAVIER STOKES:
  my_p4est_ghost_expand(p4est,ghost);

  // Create node structure
  nodes = my_p4est_nodes_new(p4est, ghost); //same

  // Create hierarchy
  my_p4est_hierarchy_t *hierarchy = new my_p4est_hierarchy_t(p4est, ghost, &brick);

  // Get neighbors and initialize them
  my_p4est_node_neighbors_t *ngbd = new my_p4est_node_neighbors_t(hierarchy, nodes);
  ngbd->init_neighbors();

  // -----------------------------------------------
  // Get the LSF(s) at the nodes:
  // -----------------------------------------------
  // LSF:
  vec_and_ptr_t phi_1;
  phi_1.create(p4est,nodes);
  sample_cf_on_nodes(p4est,nodes,level_set_1,phi_1.vec);

  vec_and_ptr_t phi_2;
  phi_2.create(p4est,nodes);
  sample_cf_on_nodes(p4est,nodes,level_set_2,phi_2.vec);

  // -----------------------------------------------
  // Get the specified function(s) on the nodes:
  // -----------------------------------------------
  vec_and_ptr_t s_1;
  s_1.create(p4est,nodes);
  sample_cf_on_nodes(p4est,nodes,function1,s_1.vec);

  vec_and_ptr_t s_2;
  s_2.create(p4est,nodes);
  sample_cf_on_nodes(p4est,nodes,function2,s_2.vec);

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
  // Set up refinement options:
  // -----------------------------------------------
  // Get smallest grid cells size:
  double dxyz_smallest[P4EST_DIM];
  dxyz_min(p4est_np1,dxyz_smallest);

  double dxyz_min_small = 10.0;
  foreach_dimension(d){
    dxyz_min_small = min(dxyz_min_small,dxyz_smallest[d]);
  }
  PetscPrintf(mpi.comm(),"Initial minimum grid size is %0.4f \n",dxyz_min_small);

  // Set the refinement options defined by the user:
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

  // Initialize vectors to hold values of the fields on the new grid
  vec_and_ptr_t phi_1_new;
  vec_and_ptr_t phi_2_new;
  vec_and_ptr_t s_1_new;
  vec_and_ptr_t s_2_new;

  // Initialize phi effective vectors  -- These will be used to describe whichever LSF or combo of LSFs we want to refine around
  vec_and_ptr_t phi_eff_old;
  vec_and_ptr_t phi_eff_new;

  int num_fields_added = 0;


  // Create the phi effective vectors:
  phi_eff_old.create(p4est,nodes);
  phi_eff_new.create(p4est_np1,nodes_np1);

  // Sample fields on the new grid-- these values will be the ones updated onto whatever current grid exists during the iteration of grid refinement
  // The "old" fields will be saved for interpolation purposes
  phi_1_new.create(p4est_np1,nodes_np1);
  VecCopyGhost(phi_1.vec,phi_1_new.vec);
  phi_2_new.create(p4est_np1,nodes_np1);
  VecCopyGhost(phi_2.vec,phi_2_new.vec);

  if(refine_by_phi1 && !refine_by_phi2){
      VecCopyGhost(phi_1.vec,phi_eff_old.vec);
      VecCopyGhost(phi_1.vec,phi_eff_new.vec);
    }

  if(refine_by_phi2 && !refine_by_phi1){
      VecCopyGhost(phi_2.vec,phi_eff_old.vec);
      VecCopyGhost(phi_2.vec,phi_eff_new.vec);
    }

  if(refine_by_phi1 && refine_by_phi2){
      // Get the "effective phi" -- the local minimum of both LSF's
      phi_eff_old.get_array();
      phi_1.get_array();
      phi_2.get_array();

      foreach_node(n,nodes){
        phi_eff_old.ptr[n] = MIN(fabs(phi_1.ptr[n]),fabs(phi_2.ptr[n]));
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
      s_2_new.create(p4est_np1,nodes_np1);
      VecCopyGhost(s_2.vec,s_2_new.vec);

      s_2.get_array();
      refinement_fields.push_back(s_2.ptr);
      s_2.restore_array();
      num_fields_added++;
    }

  // Initialize block vector to pass fields to refine by: (for use_block case)
  Vec fields_old_block;
  Vec fields_new_block;
  double* fields_old_block_ptr;

  // Initialize array of PETSc vectors to pass fields to refine by: (for !use_block case)
  Vec fields_old[num_fields];
  Vec fields_new[num_fields];


  if(use_block){
  // Set the values of the block vector:
  VecCreateGhostNodesBlock(p4est,nodes,num_fields,&fields_old_block);
  VecCreateGhostNodesBlock(p4est_np1,nodes_np1,num_fields,&fields_new_block);

  // Set the values of the block vector:
  VecGetArray(fields_old_block,&fields_old_block_ptr);
  foreach_node(n,nodes){
    for(unsigned long i = 0; i<num_fields; i++){
        fields_old_block_ptr[n*num_fields + i] = refinement_fields[i][n];
      }
  }
  VecRestoreArray(fields_old_block,&fields_old_block_ptr);

  // Save new version of fields to update during grid iteration:
  VecCopyGhost(fields_old_block,fields_new_block);

  // Check fields_old_block to see if it has the correct values:
  double xyz_old[P4EST_DIM];
  VecGetArray(fields_old_block,&fields_old_block_ptr);
  phi_1.get_array();
  s_1.get_array();
  s_2.get_array();

  double err_s1, err_s2, err_phi1, err_fields;
  double err_s1_g = 0, err_s2_g = 0, err_phi1_g=0, err_fields_g = 0;
  foreach_node(n,nodes){
    node_xyz_fr_n(n,p4est,nodes,xyz_old);
    err_s1 = max(err_s1,s_1.ptr[n] - function1(xyz_old[0],xyz_old[1]));
    err_s2 = max(err_s2,s_2.ptr[n] - function2(xyz_old[0],xyz_old[1]));
    err_phi1 = max(err_phi1,phi_1.ptr[n] - level_set_1(xyz_old[0],xyz_old[1]));
    for(int i=0;i<num_fields;i++){
        if(i ==0) err_fields = max(err_fields,fields_old_block_ptr[n*num_fields + i] - function1(xyz_old[0],xyz_old[1]));
        else err_fields = max(err_fields,fields_old_block_ptr[n*num_fields + i] - function2(xyz_old[0],xyz_old[1]));
      }
  }
  VecRestoreArray(fields_old_block,&fields_old_block_ptr);
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
                         "On block fields: %0.3e \n \n",global_errors[0],global_errors[1],global_errors[2],global_errors[3]);


  PetscPrintf(mpi.comm(),"Sets the blocked values \n");
 } // end of "if use block"
  else{
  // Set the scalar fields to refine by in a PETSc vector:
  for(int k=0; k<num_fields; k++){
      VecCreateGhostNodes(p4est,nodes,&fields_old[k]);
    }
  VecCopyGhost(s_1.vec,fields_old[0]);
  VecCopyGhost(s_2.vec,fields_old[1]);

  for(int k=0; k<num_fields; k++){
      VecCreateGhostNodes(p4est,nodes,&fields_new[k]);
    }
  VecCopyGhost(s_1_new.vec,fields_new[0]);
  VecCopyGhost(s_1_new.vec,fields_new[1]);
  } // end of "else" for "if use block"

  // Make sure we added as many fields as we said we were:
  PetscPrintf(mpi.comm(),"Num fields : %d, Num fields added : %d \n",num_fields,num_fields_added);
  P4EST_ASSERT(num_fields_added == num_fields);

  // Start refinement procedure for the grid:
  if(test_refine_and_coarsen){
    bool is_grid_changing = true;
    int no_grid_changes = 0;
    int intermediate_no=0;
    while(is_grid_changing){
          if(use_block){
               // Using block vector:
              is_grid_changing = sp1.refine_and_coarsen(p4est_np1,nodes_np1,phi_eff_new.vec,num_fields,use_block,NULL,fields_new_block,criteria,compare_opn,diag_opn);
            }
          else{
              // Using vector of vectors:
              is_grid_changing = sp1.refine_and_coarsen(p4est_np1,nodes_np1,phi_eff_new.vec,num_fields,use_block,fields_new,NULL,criteria,compare_opn,diag_opn);
            }

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

            // Reset the values we need for refinement:
            if(use_block){
                VecDestroy(fields_new_block);
                VecCreateGhostNodesBlock(p4est_np1,nodes_np1,num_fields,&fields_new_block);
              }
            else{
                for(int k=0; k<num_fields; k++){
                  VecDestroy(fields_new[k]);
                  VecCreateGhostNodes(p4est_np1,nodes_np1,&fields_new[k]);
                  }
              }
            // CHANGE TO PHI_EFF:
            phi_eff_new.destroy();
            phi_eff_new.create(p4est_np1,nodes_np1);


           // Interpolate values of phi and criteria from the old grid to current iteration of new grid:
            my_p4est_interpolation_nodes_t interp(ngbd);
            my_p4est_interpolation_nodes_t interp_block(ngbd);
            my_p4est_interpolation_nodes_t interp_mult_fields(ngbd);

            interp.set_input(phi_eff_old.vec,quadratic_non_oscillatory_continuous_v2);
            if(use_block)interp_block.set_input(fields_old_block,quadratic_non_oscillatory_continuous_v2,num_fields);
            else interp_mult_fields.set_input(fields_old, quadratic_non_oscillatory_continuous_v2,num_fields);


            double xyz[P4EST_DIM];
            foreach_node(n,nodes_np1){
              node_xyz_fr_n(n,p4est_np1,nodes_np1,xyz);
              interp.add_point(n,xyz);
              interp_block.add_point(n,xyz);
              interp_mult_fields.add_point(n,xyz);
            }

            interp.interpolate(phi_eff_new.vec);
            if(use_block) interp_block.interpolate(fields_new_block);
            else interp_mult_fields.interpolate(fields_new);


          } // End of "if grid changing"
        intermediate_no++;

        // Save the intermediate grid:
        sprintf(file_name,"%s/intermediate_%d",out_dir,intermediate_no);
        my_p4est_vtk_write_all(p4est_np1,nodes_np1,ghost_np1,P4EST_TRUE,P4EST_FALSE,0,0,file_name);

        if(intermediate_no>15){ PetscPrintf(mpi.comm(),"Grid not converging \n");break;}
      } // End of "while grid changing"

    // Once grid is done changing, view final results:
    // Interpolate the fields onto the new grid:
    my_p4est_interpolation_nodes_t interp_final(ngbd);

    Vec all_fields_old[total_num_fields];
    for (int k=0; k<total_num_fields;k++){
      VecCreateGhostNodes(p4est,nodes,&all_fields_old[k]);
      }
    all_fields_old[0] = phi_1.vec;
    all_fields_old[1] = phi_2.vec;
    all_fields_old [2] = s_1.vec;
    all_fields_old[3] = s_2.vec;

    phi_1_new.destroy(); phi_1_new.create(p4est_np1,nodes_np1);
    phi_2_new.destroy(); phi_2_new.create(phi_1_new.vec);
    s_1_new.destroy(); s_1_new.create(phi_1_new.vec);
    s_2_new.destroy(); s_2_new.create(phi_1_new.vec);

    Vec all_fields_new[total_num_fields];
    for (int k=0; k<total_num_fields;k++){
      VecCreateGhostNodes(p4est_np1,nodes_np1,&all_fields_new[k]);
      }
    all_fields_new[0] = phi_1_new.vec;
    all_fields_new[1] = phi_2_new.vec;
    all_fields_new[2] = s_1_new.vec;
    all_fields_new[3] = s_2_new.vec;


    // Interpolate all fields onto new grid:
    double xyz[P4EST_DIM];
    interp_final.set_input(all_fields_old,quadratic_non_oscillatory_continuous_v2,4);
    PetscPrintf(mpi.comm(),"Sets inputs \n ");

    foreach_node(n,nodes_np1){
      node_xyz_fr_n(n,p4est_np1,nodes_np1,xyz);
      interp_final.add_point(n,xyz);
    }

    //Interpolate all the fields:
    interp_final.interpolate(all_fields_new);
    PetscPrintf(mpi.comm(),"Interpolation of all fields onto the final grid is complete \n");

    // Now, check the interpolation errors on the final grid to make sure nothing is weird:
    double local_errors_final[5] = {0.0,0.0,0.0,0.0,0.0};// Order is : phi1, phi2, s1, s2, block fields
    double global_errors_final[5] = {0.0,0.0,0.0,0.0,0.0};

    double xyz_final[P4EST_DIM];

    double *all_fields_new_p[4];
    for(unsigned int i = 0; i<4; i++){
        VecGetArray(all_fields_new[i],&all_fields_new_p[i]);
      }

    // For checking errors on interpolated block vector:
    const double *block_fields_p;
    if(use_block){  VecGetArrayRead(fields_new_block,&block_fields_p);}

    // Loop over each node and check the error:
    foreach_node(n,nodes_np1){
      node_xyz_fr_n(n,p4est_np1,nodes_np1,xyz_final);
      double xv = xyz_final[0];
      double yv = xyz_final[1];

      // Check phi1 error:
      local_errors_final[0] = max(local_errors_final[0],fabs(all_fields_new_p[0][n] - level_set_1(xv,yv)));

      // Check phi2 error:
      local_errors_final[1] = max(local_errors_final[1],fabs(all_fields_new_p[1][n] - level_set_2(xv,yv)));

      // Check s1 error:
      local_errors_final[2] = max(local_errors_final[2],fabs(all_fields_new_p[2][n] - function1(xv,yv)));

      // Check s2 error:
      local_errors_final[3] = max(local_errors_final[3],fabs(all_fields_new_p[3][n] - function2(xv,yv)));

      // Check block field error:
      double s = 10.0;
      double err = 0.0;
      if(use_block){
          for(int j = 0;j<num_fields;j++){
              if(j==0){
                  s = function1(xv,yv);
                }
              else if(j ==1){
                  s = function2(xv,yv);
                }
              err = fabs(block_fields_p[n*num_fields + j] - s);
              local_errors_final[4] = max(local_errors_final[4],fabs(block_fields_p[n*num_fields + j] - s));
            }
        }

    }
    for(unsigned int i = 0; i<4; i++){
        VecRestoreArray(all_fields_new[i],&all_fields_new_p[i]);
      }

    // Restore the block vector array, if we used a block vector and checked its error
    if(use_block) VecRestoreArrayRead(fields_new_block,&block_fields_p);


    // Reduce the errors globally:
    MPI_Allreduce(&local_errors_final,&global_errors_final,5,MPI_DOUBLE,MPI_MAX,mpi.comm());

    PetscPrintf(mpi.comm(),"\n \n The final interpolation errors are as follows: \n \n"
                           "phi1 : %0.3e \n "
                           "phi2 : %0.3e \n "
                           "s1   : %0.3e \n "
                           "s2   : %0.3e \n "
                           "block fields : %0.3e \n \n ",global_errors_final[0],global_errors_final[1],global_errors_final[2],global_errors_final[3],global_errors_final[4]);
    PetscPrintf(mpi.comm(),"Smallest grid size is : %0.3e \n",dxyz_min_small);

    // Update the grid accordingly:
    p4est_destroy(p4est); p4est = p4est_np1;
    p4est_ghost_destroy(ghost); ghost = ghost_np1;
    p4est_nodes_destroy(nodes); nodes = nodes_np1;
    hierarchy->update(p4est,ghost);
    ngbd->update(hierarchy,nodes);

    // Visualize the fields on the final new grid:
    ex_no++;
    sprintf(file_name,"%s/refinement_test_ex_%d",out_dir,ex_no);
    PetscPrintf(mpi.comm(),"New filename is %s \n",file_name);
    save_vtk(p4est_np1,nodes_np1,ghost_np1,file_name,phi_1_new,phi_2_new,s_1_new,s_2_new);

    // ---------------------------------------------------
    // Destroy objects now that they are no longer in use:
    // ---------------------------------------------------
    if(use_block){
        VecDestroy(fields_old_block);
        VecDestroy(fields_new_block);
      }
    else{
        for(unsigned int k=0;k<num_fields; k++){
            VecDestroy(fields_old[k]);
            VecDestroy(fields_new[k]);
          }
      }

    // This (vvv) takes care of destroying s_1, s_2, phi_1, phi_2, s_1_new, s_2_new, phi_1_new, phi_2_new as well bc those vectors are set in the all_fields_ arrays
    for(unsigned int k=0; k<total_num_fields; k++){
        VecDestroy(all_fields_old[k]);
        VecDestroy(all_fields_new[k]);
      }
  } // end of "if test refine and coarsen"

  if(test_update_p4est){
      int update_p4est_test_no = 0;
      sprintf(file_name,"%s/update_p4est_test_%d",out_dir,update_p4est_test_no);
      //VecView(s_2_new.vec,PETSC_VIEWER_STDOUT_WORLD);
      save_vtk(p4est,nodes,ghost,file_name,phi_1_new,phi_2_new,s_1_new,s_2_new);

      // First, create velocity vector to advect LSF by:
      vec_and_ptr_dim_t velocity(p4est_np1,nodes_np1);
      const CF_DIM *velocity_cf[P4EST_DIM] = {&u_vel,&v_vel};
      foreach_dimension(d){
        sample_cf_on_nodes(p4est_np1,nodes_np1,&velocity_cf[d],velocity.vec[d]);
      }

      // Get derivatives of phi_1_new:
      vec_and_ptr_dim_t phi_1_new_xx(velocity.vec);
      ngbd_np1->second_derivatives_central(phi_1_new.vec,phi_1_new_xx.vec);

      // Create the semi-lagrangian object:
      my_p4est_semi_lagrangian_t sl(&p4est_np1,&nodes_np1,&ghost_np1,ngbd);

      // Call the update_p4est function:
      PetscPrintf(mpi.comm(),"Calling the update_p4est function now ... \n");
      sl.update_p4est(velocity.vec,0.2,phi_1_new.vec,phi_1_new_xx.vec,NULL,num_fields,use_block,fields_new,fields_new_block,criteria,compare_opn,diag_opn);


      // Interpolate all other fields onto the new grid (besides phi_1_new, which has already been advected):
      // Once grid is done changing, view final results:
      // Interpolate the fields onto the new grid:
      my_p4est_interpolation_nodes_t interp_final(ngbd);

      Vec all_fields_old[total_num_fields-1];
      for (int k=0; k<total_num_fields-1;k++){
        VecCreateGhostNodes(p4est,nodes,&all_fields_old[k]);
        }
      all_fields_old[0] = phi_2.vec;
      all_fields_old [1] = s_1.vec;
      all_fields_old[2] = s_2.vec;

      phi_2_new.destroy(); phi_2_new.create(phi_1_new.vec);
      s_1_new.destroy(); s_1_new.create(phi_1_new.vec);
      s_2_new.destroy(); s_2_new.create(phi_1_new.vec);

      Vec all_fields_new[total_num_fields-1];
      for (int k=0; k<total_num_fields-1;k++){
        VecCreateGhostNodes(p4est_np1,nodes_np1,&all_fields_new[k]);
        }
      all_fields_new[0] = phi_2_new.vec;
      all_fields_new[1] = s_1_new.vec;
      all_fields_new[2] = s_2_new.vec;


      // Interpolate all fields onto new grid:
      double xyz[P4EST_DIM];
      interp_final.set_input(all_fields_old,quadratic_non_oscillatory_continuous_v2,total_num_fields -1);
      PetscPrintf(mpi.comm(),"Sets inputs \n ");

      foreach_node(n,nodes_np1){
        node_xyz_fr_n(n,p4est_np1,nodes_np1,xyz);
        interp_final.add_point(n,xyz);
      }

      //Interpolate all the fields:
      interp_final.interpolate(all_fields_new);
      PetscPrintf(mpi.comm(),"Interpolation of all fields onto the final grid is complete \n");


      // Print out the new grid:
      update_p4est_test_no++;

      sprintf(file_name,"%s/update_p4est_test_%d",out_dir,update_p4est_test_no);
      save_vtk(p4est_np1,nodes_np1,ghost_np1,file_name,phi_1_new,phi_2_new,s_1_new,s_2_new);
      //my_p4est_vtk_write_all(p4est_np1,nodes_np1,ghost_np1,P4EST_TRUE,P4EST_TRUE,0,0,file_name);



      p4est_nodes_destroy(nodes_np1);
      p4est_ghost_destroy(ghost_np1);
      p4est_destroy(p4est_np1);

    }
  // -----------------------------------------------
  // Destroy the grid structures now that they are no longer in use:
  // -----------------------------------------------
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy      (p4est);


  my_p4est_brick_destroy(conn, &brick);

  w.stop(); w.read_duration();


}

