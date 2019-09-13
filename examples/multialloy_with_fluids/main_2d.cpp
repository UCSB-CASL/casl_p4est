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


#include <src/my_p4est_poisson_nodes_mls.h>
#include <src/my_p4est_poisson_nodes.h>
#include <src/my_p4est_interpolation_nodes.h>



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
// Define geometry:
double xmin = -1.0;
double xmax = 1.0;
double ymin = -1.0;
double ymax = 1.0;

// Grid refinement:
int lmin = 4;
int lmax = 7;

// Time-stepping:
double tfinal = 0.010;
double delta_t = 0.001;

// Physical properties:
double alpha_s = (1.1820e-6); //ice - [m^2]/s
double alpha_l = (1.4547e-7); //water- [m^2]/s

double k_s = 2.22; // W/[m*K]
double k_l = 0.608; // W/[m*K]

bool move_interface_with_v_external = false;

bool do_advection = false;

bool output_information = true;


// Begin defining classes for necessary functions and boundary conditions...
// --------------------------------------------------------------------------------------------------------------
// LEVEL SET FUNCTION:
// --------------------------------------------------------------------------------------------------------------
struct LEVEL_SET : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    return 0.25 - sqrt(SQR(x) + SQR(y));
    //return 0.5 - sqrt(SQR(x) + SQR(y-3*ymin/4));
  }
} level_set;

struct MINI_LEVEL_SET : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    return 0.15 - sqrt(SQR(x) + SQR(y));
    //return 0.5 - sqrt(SQR(x) + SQR(y-3*ymin/4));
  }
} mini_level_set;

// --------------------------------------------------------------------------------------------------------------
// PRESCRIBED VELOCITY FIELD AT WHICH THE INTERFACE ADVANCES
// --------------------------------------------------------------------------------------------------------------
struct u_advance : CF_DIM
{ double operator() (double x, double y) const{
  return 0.5;
  }

} u_adv;

struct v_advance: CF_DIM{
  double operator()(double x, double y) const
  {
    return 0.0;
  }
} v_adv;

// --------------------------------------------------------------------------------------------------------------
// not sure what this is for ....
// --------------------------------------------------------------------------------------------------------------
class BC_interface_value: public CF_DIM{
public:
  double operator()(double x, double y) const
  { return 273.0;
  }
}bc_interface_val;

class BC_interface_coeff: public CF_DIM{
public:
  double operator()(double x, double y) const
  { return 1.0;
  }
}bc_interface_coeff;

class BC_WALL_TYPE: public WallBC2D
{
  BoundaryConditionType operator()(double, double) const{
    return DIRICHLET;
  }
} bc_wall_type;

// --------------------------------------------------------------------------------------------------------------
// WALL TEMPERATURE BOUNDARY CONDITION
// --------------------------------------------------------------------------------------------------------------
class WALL_BC_TYPE_TEMP: public WallBCDIM
{
public:
  BoundaryConditionType operator()(DIM(double x, double y, double z )) const
  {
//    if(fabs(y-ymin)<EPS){
//        return NEUMANN;
//      }
    return DIRICHLET;
  }
} wall_bc_type_temp;

class WALL_BC_VALUE_TEMP: public CF_DIM
{
public:
  double operator()(DIM(double x, double y, double z)) const
  {
    if ((fabs(y-ymax)<EPS) || (fabs(y-ymin)<EPS) || (fabs(x-xmin)<EPS) || (fabs(x-xmax)<EPS)){
        if (level_set(DIM(x,y,z)) < EPS){
          return 298.0;
          }
        else{return 255.0;}
      }
    if ((fabs(y) < EPS) && (fabs(x) < EPS)){ // Center of domain -- inside solid domain
      return 255.0;
    }

//    if (level_set(DIM(x,y,z)) > EPS){
//        return 255.0;
//      }
//    else{
//        double m = (298.0 - 255.0)/(level_set(DIM(xmin,ymin,z)));
//        return 255.0 + m*level_set(DIM(x,y,z));
//      }

//            double m = (298.0 - 255.0)/(level_set(DIM(xmin,ymin,z)));
//            return 255.0 + m*level_set(DIM(x,y,z));
  }

} wall_bc_value_temp;

// --------------------------------------------------------------------------------------------------------------
// TEMPERATURE INITIAL CONDITION
// --------------------------------------------------------------------------------------------------------------
class INITIAL_CONDITION_TEMP: public CF_DIM
{
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    double m;
    if (level_set(DIM(x,y,z)) > EPS){
        //m = (255.0 - 273.0)/(level_set(DIM(xmin,ymin,z)));
        //return 273.0 - m*level_set(DIM(x,y,z));
        return 255.0;
      }
    else{
        m = (298.0 - 273.0)/(level_set(DIM(xmin,ymin,z)));
        return 273.0 + m*level_set(DIM(x,y,z));
      }
  }
}IC_temp;

// --------------------------------------------------------------------------------------------------------------
// INTERFACIAL TEMPERATURE BOUNDARY CONDITION
// --------------------------------------------------------------------------------------------------------------

BoundaryConditionType interface_bc_type_temp = DIRICHLET;

// --------------------------------------------------------------------------------------------------------------
// Function for checking the temperature values during the solution process
// --------------------------------------------------------------------------------------------------------------
vector <double> check_T_values(vec_and_ptr_t phi, vec_and_ptr_t T, p4est_nodes* nodes) {

  vector <double> T_values;

  double t_norm = 0.0;

  int pts = 0;

  double t_max = 0.0;

  double t_min_signed = 1000.00;

  double t_min = 1000.0;

  phi.get_array(); T.get_array();

  foreach_node(n,nodes){
    if(phi.ptr[n]<EPS){
        pts++;
        t_norm +=SQR(T.ptr[n]);
        t_max = max(t_max,fabs(T.ptr[n]));
        t_min = min(t_min,fabs(T.ptr[n]));
        t_min_signed = min(t_min_signed,T.ptr[n]);
      }
  }

  phi.restore_array();
  T.restore_array();
  t_norm = sqrt(t_norm/pts);

  T_values.push_back(t_norm);
  T_values.push_back(t_max);
  T_values.push_back(t_min);
  T_values.push_back(t_min_signed);

  return T_values;

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

  p4est_t               *p4est_np1;
  p4est_nodes_t         *nodes_np1;
  p4est_ghost_t         *ghost_np1;


  // domain size information
  const int n_xyz[]      = { 1,  1,  0};
  const double xyz_min[] = {xmin, ymin, 0};
  const double xyz_max[] = {xmax,  ymax,  0};
  const int periodic[]   = { 0,  0,  0};

  // -----------------------------------------------
  // Set properties for the Poisson node problem:
  // -----------------------------------------------
  int cube_refinement = 4;
  interpolation_method interp_bw_grids = quadratic_non_oscillatory_continuous_v2;

  // -----------------------------------------------
  // Create the grid:
  // -----------------------------------------------
  conn = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic); // same as Daniil

  // create the forest
  p4est = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL); // same as Daniil

  // refine based on distance to a level-set
  splitting_criteria_cf_t sp(lmin, lmax, &level_set);           // same as Daniil, minus lipschitz
  p4est->user_pointer = &sp;                                    // save the pointer to the forst splitting criteria
  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL); // refine the level set according to the splitting criteria

  // partition the forest
  my_p4est_partition(p4est, P4EST_TRUE, NULL);                  // partition the forest but allow for coarsening --> Daniil does not allow (use P4EST_FALSE)

  // create ghost layer
  ghost = my_p4est_ghost_new(p4est, P4EST_CONNECT_FULL); // same

  // create node structure
  nodes = my_p4est_nodes_new(p4est, ghost); //same

  // Create hierarchy
  my_p4est_hierarchy_t *hierarchy = new my_p4est_hierarchy_t(p4est, ghost, &brick);

  // Get neighbors
  my_p4est_node_neighbors_t *ngbd = new my_p4est_node_neighbors_t(hierarchy, nodes);
  ngbd->init_neighbors();

  // -----------------------------------------------
  // Initialize the Level Set function:
  // -----------------------------------------------
  // LSF:
  vec_and_ptr_t phi;
  phi.create(p4est,nodes);
  sample_cf_on_nodes(p4est,nodes,level_set,phi.vec);

  // LSF for solid domain:
  vec_and_ptr_t phi_solid;

  // 2nd derivatives of LSF's
  vec_and_ptr_dim_t phi_dd;
  vec_and_ptr_dim_t phi_solid_dd;

//  vec_and_ptr_t mini_phi;
//  mini_phi.create(phi.vec);
//  sample_cf_on_nodes(p4est,nodes,mini_level_set,mini_phi.vec);

  // -----------------------------------------------
  // Initialize the Velocity field:
  // -----------------------------------------------
  vec_and_ptr_dim_t vel_n(p4est,nodes);

  const CF_DIM *vel_cf[P4EST_DIM] = {&u_adv, &v_adv};

  for (int dir = 0; dir<P4EST_DIM;dir++){
      sample_cf_on_nodes(p4est,nodes,*vel_cf[dir],vel_n.vec[dir]);
    }

  vec_and_ptr_dim_t v_interface(p4est,nodes);

  // -----------------------------------------------
  // Initialize the Temperature field:
  // -----------------------------------------------
  // Vectors for T_liquid:
  vec_and_ptr_t T_l_n;
  T_l_n.create(p4est,nodes);
  sample_cf_on_nodes(p4est,nodes,IC_temp,T_l_n.vec); // Sample this just so that we can save the initial temperature distribution
  vec_and_ptr_t rhs_Tl;

  // Vectors for T_solid:
  vec_and_ptr_t T_s_n;
  T_s_n.create(p4est,nodes);
  sample_cf_on_nodes(p4est,nodes,IC_temp,T_s_n.vec); // Sample this just so that we can save the initial temperature distribution
  vec_and_ptr_t rhs_Ts;

  // Vectors to hold T values on old grid (for interpolation purposes)
  vec_and_ptr_t T_l_old;
  vec_and_ptr_t T_s_old;

  // Vectors to hold first derivatives of T
  vec_and_ptr_dim_t T_l_d;
  vec_and_ptr_dim_t T_s_d;

  // Vectors to hold the normals of each domain:
  vec_and_ptr_dim_t liquid_normals;
  vec_and_ptr_dim_t solid_normals;

  // -----------------------------------------------
  // Initialize the output file:
  // -----------------------------------------------
  int out_idx = 0;
  char outdir[1000];
  sprintf(outdir,"/home/elyce/workspace/projects/multialloy_with_fluids/solidif_with_fluids_output/snapshot_%d",out_idx);

  // -----------------------------------------------
  // Initialize the needed solvers
  // -----------------------------------------------
  my_p4est_poisson_nodes_mls_t *solver_Tl;  // will solve poisson problem for Temperature in liquid domains
  my_p4est_poisson_nodes_mls_t *solver_Ts;  // will solve poisson problem for Temperature in solid domain

  // -----------------------------------------------
  // Begin stepping through time
  // -----------------------------------------------
  double tf = tfinal;
  int tstep = 0;
  int save = 1;
  double dt = delta_t;

  for (double t = 0; t<tf; t+=dt, tstep++){
      PetscPrintf(mpi.comm(),"\n");
      ierr = PetscPrintf(mpi.comm(),"Iteration %d , Time: %0.3f \n ------------------------------------------- \n",tstep,t);


      // Define some variables needed to specify how to extend across the interface:
      double dxyz[P4EST_DIM];
      dxyz_min(p4est, dxyz);
      double min_volume_ = MULTD(dxyz[0], dxyz[1], dxyz[2]);
      double extension_band_use_    = 8.*pow(min_volume_, 1./ double(P4EST_DIM)); //8
      double extension_band_extend_ = 10.*pow(min_volume_, 1./ double(P4EST_DIM)); //10
      double extension_band_check_  = 6.*pow(min_volume_, 1./ double(P4EST_DIM)); // 6

      /*
      PetscPrintf(mpi.comm(),"\n ");

      PetscPrintf(mpi.comm(),"Band use: %0.2f \n",extension_band_use_);
      PetscPrintf(mpi.comm(),"Band extend: %0.2f \n",extension_band_extend_);
      PetscPrintf(mpi.comm(),"Band check : %0.2f \n",extension_band_check_);
      PetscPrintf(mpi.comm(),"\n ");
      */
      // --------------------------------------------------------------------------------------------------------------
      // Extend Fields Across Interface:
      // --------------------------------------------------------------------------------------------------------------
      // Define LSF for the solid domain (as just the negative of the liquid one):
      phi_solid.create(p4est,nodes);
      VecScaleGhost(phi.vec,-1.0);
      VecCopyGhost(phi.vec,phi_solid.vec);
      VecScaleGhost(phi.vec,-1.0);


      vector <double> T_values;
      if (output_information){
        T_values = check_T_values(phi,T_l_n,nodes);

        MPI_Barrier(mpi.comm());
        for (int i=0; i<4; i++){
            mpi_ret = MPI_Allreduce(MPI_IN_PLACE,&T_values[i],1,MPI_DOUBLE,MPI_MAX,mpi.comm());
            SC_CHECK_MPI(mpi_ret);
          }


        PetscPrintf(mpi.comm(),"\n");
        PetscPrintf(mpi.comm(),"Before extending ----------:");
        PetscPrintf(mpi.comm(),"\n");

        PetscPrintf(mpi.comm(),"Tl data: \n");
        PetscPrintf(mpi.comm(),"norm : %0.2f \n",T_values[0]);
        PetscPrintf(mpi.comm(),"max magnitude: %0.2f \n",T_values[1]);
        PetscPrintf(mpi.comm(),"min magnitude: %0.2f \n",T_values[2]);
        PetscPrintf(mpi.comm(),"min value: %0.2f \n",T_values[3]);

        T_values = check_T_values(phi_solid,T_s_n,nodes);
        MPI_Barrier(mpi.comm());
        for (int i=0; i<4; i++){
            mpi_ret = MPI_Allreduce(MPI_IN_PLACE,&T_values[i],1,MPI_DOUBLE,MPI_MAX,mpi.comm());
            SC_CHECK_MPI(mpi_ret);
          }

        PetscPrintf(mpi.comm(),"Ts data: \n");
        PetscPrintf(mpi.comm(),"norm : %0.2f \n",T_values[0]);
        PetscPrintf(mpi.comm(),"max magnitude: %0.2f \n",T_values[1]);
        PetscPrintf(mpi.comm(),"min magnitude: %0.2f \n",T_values[2]);
        PetscPrintf(mpi.comm(),"min value: %0.2f \n",T_values[3]);


    }

      // Get second derivatives of both LSFs:
      phi_dd.create(p4est,nodes);
      phi_solid_dd.create(p4est,nodes);

      ngbd->dxx_central(phi.vec,phi_dd.vec[0]);
      ngbd->dyy_central(phi.vec,phi_dd.vec[1]);

      ngbd->dxx_central(phi_solid.vec,phi_solid_dd.vec[0]);
      ngbd->dyy_central(phi_solid.vec,phi_solid_dd.vec[1]);

      // Compute normals for each domain:
      liquid_normals.create(p4est,nodes);
      compute_normals(*ngbd,phi.vec,liquid_normals.vec);

      solid_normals.create(p4est,nodes);
      compute_normals(*ngbd,phi_solid.vec,solid_normals.vec);


      // Extend Temperature Fields across the interface:
      my_p4est_level_set_t ls(ngbd);
      ls.extend_Over_Interface_TVD_Full(phi.vec,T_l_n.vec,50,2,1.e-9,extension_band_use_,extension_band_extend_,extension_band_check_,liquid_normals.vec,NULL,NULL,false,NULL,NULL);

      ls.extend_Over_Interface_TVD_Full(phi_solid.vec,T_s_n.vec,50,2,1.e-9,extension_band_use_,extension_band_extend_,extension_band_check_,solid_normals.vec,NULL,NULL,false,NULL,NULL);

      // Delete data for normals since it is no longer needed:
      liquid_normals.destroy();
      solid_normals.destroy();


      if (output_information) {
        PetscPrintf(mpi.comm(),"\n");
        PetscPrintf(mpi.comm(),"After extending ----------:");
        PetscPrintf(mpi.comm(),"\n");

        T_values = check_T_values(phi,T_l_n,nodes);
        MPI_Barrier(mpi.comm());
        MPI_Barrier(mpi.comm());
        for (int i=0; i<4; i++){
            mpi_ret = MPI_Allreduce(MPI_IN_PLACE,&T_values[i],1,MPI_DOUBLE,MPI_MAX,mpi.comm());
            SC_CHECK_MPI(mpi_ret);
          }
        PetscPrintf(mpi.comm(),"Tl data: \n");
        PetscPrintf(mpi.comm(),"norm : %0.2f \n",T_values[0]);
        PetscPrintf(mpi.comm(),"max magnitude: %0.2f \n",T_values[1]);
        PetscPrintf(mpi.comm(),"min magnitude: %0.2f \n",T_values[2]);
        PetscPrintf(mpi.comm(),"min value: %0.2f \n",T_values[3]);

        T_values = check_T_values(phi_solid,T_s_n,nodes);
        MPI_Barrier(mpi.comm());
        MPI_Barrier(mpi.comm());
        for (int i=0; i<4; i++){
            mpi_ret = MPI_Allreduce(MPI_IN_PLACE,&T_values[i],1,MPI_DOUBLE,MPI_MAX,mpi.comm());
            SC_CHECK_MPI(mpi_ret);
          }

        PetscPrintf(mpi.comm(),"Ts data: \n");
        PetscPrintf(mpi.comm(),"norm : %0.2f \n",T_values[0]);
        PetscPrintf(mpi.comm(),"max magnitude: %0.2f \n",T_values[1]);
        PetscPrintf(mpi.comm(),"min magnitude: %0.2f \n",T_values[2]);
        PetscPrintf(mpi.comm(),"min value: %0.2f \n",T_values[3]);
        }

      // --------------------------------------------------------------------------------------------------------------
      // SAVING DATA: Save data every specified amout of timesteps: -- Do this after values are extended across interface to make visualization nicer
      // --------------------------------------------------------------------------------------------------------------

      if (tstep % save ==0){
          out_idx++;
          sprintf(outdir,"/home/elyce/workspace/projects/multialloy_with_fluids/solidif_with_fluids_output/snapshot_%d",out_idx);
          // -----------------------------------------------
          // Get local array to write initial LSF and velocity fields to vtk:
          // -----------------------------------------------

          phi.get_array();
          vel_n.get_array();
          T_l_n.get_array();
          T_s_n.get_array();

          // Write out the data:
          my_p4est_vtk_write_all(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,
                                            5,0,outdir,
                                            VTK_POINT_DATA,"phi",phi.ptr,
                                            VTK_POINT_DATA,"vx",vel_n.ptr[0],
                                            VTK_POINT_DATA,"vy",vel_n.ptr[1],
                                            VTK_POINT_DATA,"Tl",T_l_n.ptr,
                                            VTK_POINT_DATA,"Ts",T_s_n.ptr);

          phi.restore_array();
          vel_n.restore_array();
          T_l_n.restore_array();
          T_s_n.restore_array();

        }


      // --------------------------------------------------------------------------------------------------------------
      // Compute the jump in flux across the interface to use to advance the LSF:
      // --------------------------------------------------------------------------------------------------------------
      vec_and_ptr_dim_t jump;
      jump.create(p4est,nodes);

      T_l_d.create(jump.vec);
      T_s_d.create(jump.vec);

      // Get derivatives of the Temperature to see what's going on:
      ngbd->first_derivatives_central(T_l_n.vec,T_l_d.vec);
      ngbd->first_derivatives_central(T_s_n.vec,T_s_d.vec);

      if(output_information){

        T_l_d.get_array();
        T_s_d.get_array();

        double T_l_d_x = 0.0;
        double T_l_d_y = 0.0;
        double T_s_d_x = 0.0;
        double T_s_d_y = 0.0;

        double max_Tl_deriv = 0.0;
        double max_Ts_deriv = 0.0;

        int pts_T = 0;

        foreach_node(n,nodes){
          pts_T++;
          T_l_d_x += T_l_d.ptr[0][n];
          T_l_d_y += T_l_d.ptr[1][n];

          T_s_d_x += T_s_d.ptr[0][n];
          T_s_d_y += T_s_d.ptr[1][n];

          max_Tl_deriv = max(max_Tl_deriv,fabs(T_l_d.ptr[0][n]));
          max_Tl_deriv = max(max_Tl_deriv,fabs(T_l_d.ptr[1][n]));
          max_Ts_deriv = max(max_Ts_deriv,fabs(T_s_d.ptr[0][n]));
          max_Ts_deriv = max(max_Ts_deriv,fabs(T_s_d.ptr[1][n]));
        }
        T_l_d_x/=pts_T;
        T_l_d_y/=pts_T;

        T_s_d_x/=pts_T;
        T_s_d_y/=pts_T;

        MPI_Barrier(mpi.comm());
        mpi_ret = MPI_Allreduce(MPI_IN_PLACE,&T_l_d_x,1,MPI_DOUBLE,MPI_MAX,mpi.comm());
        SC_CHECK_MPI(mpi_ret);

        MPI_Barrier(mpi.comm());
        mpi_ret = MPI_Allreduce(MPI_IN_PLACE,&T_l_d_y,1,MPI_DOUBLE,MPI_MAX,mpi.comm());
        SC_CHECK_MPI(mpi_ret);

        MPI_Barrier(mpi.comm());
        mpi_ret = MPI_Allreduce(MPI_IN_PLACE,&T_s_d_x,1,MPI_DOUBLE,MPI_MAX,mpi.comm());
        SC_CHECK_MPI(mpi_ret);

        MPI_Barrier(mpi.comm());
        mpi_ret = MPI_Allreduce(MPI_IN_PLACE,&T_s_d_y,1,MPI_DOUBLE,MPI_MAX,mpi.comm());
        SC_CHECK_MPI(mpi_ret);

        MPI_Barrier(mpi.comm());
        mpi_ret = MPI_Allreduce(MPI_IN_PLACE,&max_Tl_deriv,1,MPI_DOUBLE,MPI_MAX,mpi.comm());
        SC_CHECK_MPI(mpi_ret);

        MPI_Barrier(mpi.comm());
        mpi_ret = MPI_Allreduce(MPI_IN_PLACE,&max_Ts_deriv,1,MPI_DOUBLE,MPI_MAX,mpi.comm());
        SC_CHECK_MPI(mpi_ret);

        PetscPrintf(mpi.comm(),"\n dT/dx l avg: %0.2f", T_l_d_x);
        PetscPrintf(mpi.comm(),"\n dT/dy l avg: %0.2f \n", T_l_d_y);

        PetscPrintf(mpi.comm(),"\n dT/dx s avg: %0.2f", T_s_d_x);
        PetscPrintf(mpi.comm(),"\n dT/dy s avg: %0.2f \n", T_s_d_y);

        PetscPrintf(mpi.comm(),"\n Max Tl derivative: %0.2f", max_Tl_deriv);
        PetscPrintf(mpi.comm(),"\n Max Ts derivatve : %0.2f \n", max_Ts_deriv);


        //PetscPrintf(mpi.comm(),"\n jump v norm: %0.2f \n", v_norm);
        T_l_d.restore_array(); T_s_d.restore_array();
        }
      // ----------------------


      jump.get_array();
      T_l_d.get_array();
      T_s_d.get_array();

      // First, compute jump in the layer nodes:
      quad_neighbor_nodes_of_node_t qnnn;
      for(size_t i=0; i<ngbd->get_layer_size();i++){
        p4est_locidx_t n = ngbd->get_layer_node(i);
        //ngbd->get_neighbors(n,qnnn);

        jump.ptr[0][n] = k_s*T_s_d.ptr[0][n] -k_l*T_l_d.ptr[0][n];
        jump.ptr[1][n] = k_s*T_s_d.ptr[1][n] -k_l*T_l_d.ptr[1][n];

        }

      // Begin updating the ghost values of the layer nodes:
      foreach_dimension(d){
        VecGhostUpdateBegin(jump.vec[d],INSERT_VALUES,SCATTER_FORWARD);
      }

      // Compute the jump in the local nodes:
      for(size_t i = 0; i<ngbd->get_local_size();i++){
          p4est_locidx_t n = ngbd->get_local_node(i);
          //ngbd->get_neighbors(n,qnnn);
          jump.ptr[0][n] = k_s*T_s_d.ptr[0][n] -k_l*T_l_d.ptr[0][n];
          jump.ptr[1][n] = k_s*T_s_d.ptr[1][n] -k_l*T_l_d.ptr[1][n];
        }

      // Finish updating the ghost values of the layer nodes:
      foreach_dimension(d){
        VecGhostUpdateEnd(jump.vec[d],INSERT_VALUES,SCATTER_FORWARD);
      }

      jump.restore_array();
      T_l_d.restore_array();
      T_s_d.restore_array();


      // Check the values of jump: ---

      if (output_information){
        jump.get_array();
        double avg_vint_x = 0.0;
        double avg_vint_y = 0.0;

        int pts_1 = 0;

        foreach_node(n,nodes){
          pts_1++;
          avg_vint_x += jump.ptr[0][n];
          avg_vint_y += jump.ptr[0][n];
        }
        avg_vint_x/=pts_1;
        avg_vint_y/=pts_1;

        MPI_Barrier(mpi.comm());
        mpi_ret = MPI_Allreduce(MPI_IN_PLACE,&avg_vint_x,1,MPI_DOUBLE,MPI_MAX,mpi.comm());
        SC_CHECK_MPI(mpi_ret);

        MPI_Barrier(mpi.comm());
        mpi_ret = MPI_Allreduce(MPI_IN_PLACE,&avg_vint_y,1,MPI_DOUBLE,MPI_MAX,mpi.comm());
        SC_CHECK_MPI(mpi_ret);

        PetscPrintf(mpi.comm(),"\n jump vx avg: %0.2f", avg_vint_x);
        PetscPrintf(mpi.comm(),"\n jump vy avg: %0.2f \n", avg_vint_y);
        jump.restore_array();
        }
      // ------


      v_interface.destroy();
      v_interface.create(p4est,nodes);
      // Extend the interfacial temperature to the whole domain for advection of the LSF:
      foreach_dimension(d){
         ls.extend_from_interface_to_whole_domain_TVD(phi.vec,jump.vec[d],v_interface.vec[d],50);
      }

      // Destroy values once no longer needed:
      T_l_d.destroy();
      T_s_d.destroy();
      jump.destroy();

      // Check the values of v_interface:
      v_interface.get_array();
      phi.get_array();

      double v_norm = 0.0;//double
      double max_v_norm = 0.0;

      int pts = 0; //int

      double dxyz_smallest[P4EST_DIM];
      dxyz_min(p4est,dxyz_smallest);

      double dxyz_close_to_interface = 1.2*max(dxyz_smallest[0],dxyz_smallest[1]);
      foreach_node(n,nodes){
        pts++;
        if (phi.ptr[n] < dxyz_close_to_interface && phi.ptr[n] > -1.0*dxyz_close_to_interface){
            max_v_norm = max(max_v_norm,sqrt(SQR(v_interface.ptr[0][n]) + SQR(v_interface.ptr[1][n])));
          }
        //v_norm+= sqrt(SQR(v_interface.ptr[0][n]) + SQR(v_interface.ptr[1][n]));
      }


      //v_norm/=pts;
      MPI_Barrier(mpi.comm());
      int mpi_ret = MPI_Allreduce(MPI_IN_PLACE,&max_v_norm,1,MPI_DOUBLE,MPI_MAX,mpi.comm());
      SC_CHECK_MPI(mpi_ret);

      PetscPrintf(mpi.comm(),"\n max v norm: %0.2f \n \n", max_v_norm);


      P4EST_ASSERT(max_v_norm < 100.0);

      // Compute new timestep:
      dt = 0.1*min(dxyz_smallest[0],dxyz_smallest[1])/max_v_norm;

      ierr = PetscPrintf(mpi.comm(),"Timestep : %0.3e \n ------------------------------------------- \n",dt);


      v_interface.restore_array();
      phi.restore_array();
      // --------------------------------------------------------------------------------------------------------------
      // Compute the timestep -- determined by velocity at the interface:
      // --------------------------------------------------------------------------------------------------------------



      // --------------------------------------------------------------------------------------------------------------
      // Store old grid values, and then reinitialize the Temperature vectors to hold the data on the new interpolated grid
      // --------------------------------------------------------------------------------------------------------------
      // Store temperature values on the old grid:
      T_l_old.create(p4est,nodes);
      T_s_old.create(T_l_old.vec); // Make objects to hold old grid values so T_l_n and T_s_n can be updated with values for the new grid
      VecCopyGhost(T_l_n.vec,T_l_old.vec);
      VecCopyGhost(T_s_n.vec,T_s_old.vec);




      // --------------------------------------------------------------------------------------------------------------
      // Advance the LSF:
      // --------------------------------------------------------------------------------------------------------------
      // Make a copy of the grid objects for the next timestep:
      p4est_np1 = p4est_copy(p4est,P4EST_FALSE); // copy the grid but not the data
      ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
      nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);


      // Create the semi-lagrangian object and do the advection:
      my_p4est_semi_lagrangian_t sl(&p4est_np1, &nodes_np1, &ghost_np1, ngbd); // is this really the correct way to do this?

      // Advect the grid under the velocity field:
      if (move_interface_with_v_external){
          sl.update_p4est(vel_n.vec,dt,phi.vec);
        }
      else{
          sl.update_p4est(v_interface.vec,dt,phi.vec);
        }


      // --------------------------------------------------------------------------------------------------------------
      // Interpolate Values onto New Grid:
      // -------------------------------------------------------------------------------------------------------------
      // Update the velocity field onto the new grid:
      vel_n.destroy();
      vel_n.create(p4est_np1,nodes_np1);
      for (int dir=0;dir<P4EST_DIM;dir++){
          sample_cf_on_nodes(p4est_np1,nodes_np1,*vel_cf[dir],vel_n.vec[dir]);
        }

      // Interpolate the Temperature values onto the new grid for the next timestep:-------------
      my_p4est_interpolation_nodes_t  interp_nodes(ngbd);
      double xyz[P4EST_DIM];

      // --> Grab the points on the new grid that we want to interpolate to:
      foreach_node(n,nodes_np1){
          node_xyz_fr_n(n,p4est_np1,nodes_np1,xyz);
          interp_nodes.add_point(n,xyz);
      }

      T_l_n.destroy();
      T_s_n.destroy();
      T_l_n.create(p4est_np1,nodes_np1);
      T_s_n.create(T_l_n.vec);

      interp_nodes.set_input(T_l_old.vec, interp_bw_grids); interp_nodes.interpolate(T_l_n.vec);
      interp_nodes.set_input(T_s_old.vec, interp_bw_grids); interp_nodes.interpolate(T_s_n.vec);

      // Destroy values stored at old grid, since they are no longer needed:
      T_l_old.destroy();
      T_s_old.destroy();

      if (output_information) {
        PetscPrintf(mpi.comm(),"\n");
        PetscPrintf(mpi.comm(),"After interpolating onto new grid ----------:");
        PetscPrintf(mpi.comm(),"\n");

        T_values = check_T_values(phi,T_l_n,nodes);
        MPI_Barrier(mpi.comm());
        MPI_Barrier(mpi.comm());
        for (int i=0; i<4; i++){
            mpi_ret = MPI_Allreduce(MPI_IN_PLACE,&T_values[i],1,MPI_DOUBLE,MPI_MAX,mpi.comm());
            SC_CHECK_MPI(mpi_ret);
          }
        PetscPrintf(mpi.comm(),"Tl data: \n");
        PetscPrintf(mpi.comm(),"norm : %0.2f \n",T_values[0]);
        PetscPrintf(mpi.comm(),"max magnitude: %0.2f \n",T_values[1]);
        PetscPrintf(mpi.comm(),"min magnitude: %0.2f \n",T_values[2]);
        PetscPrintf(mpi.comm(),"min value: %0.2f \n",T_values[3]);

        T_values = check_T_values(phi_solid,T_s_n,nodes);
        MPI_Barrier(mpi.comm());
        MPI_Barrier(mpi.comm());
        for (int i=0; i<4; i++){
            mpi_ret = MPI_Allreduce(MPI_IN_PLACE,&T_values[i],1,MPI_DOUBLE,MPI_MAX,mpi.comm());
            SC_CHECK_MPI(mpi_ret);
          }

        PetscPrintf(mpi.comm(),"Ts data: \n");
        PetscPrintf(mpi.comm(),"norm : %0.2f \n",T_values[0]);
        PetscPrintf(mpi.comm(),"max magnitude: %0.2f \n",T_values[1]);
        PetscPrintf(mpi.comm(),"min magnitude: %0.2f \n",T_values[2]);
        PetscPrintf(mpi.comm(),"min value: %0.2f \n",T_values[3]);
        }



      // --------------------------------------------------------------------------------------------------------------
      // Delete the old grid and reinitialize phi:
      // --------------------------------------------------------------------------------------------------------------

      // Delete the old grid and update with the new one:
      p4est_destroy(p4est); p4est = p4est_np1;
      p4est_ghost_destroy(ghost); ghost = ghost_np1;
      p4est_nodes_destroy(nodes); nodes = nodes_np1;

      delete hierarchy; hierarchy = new my_p4est_hierarchy_t(p4est,ghost,&brick);
      delete ngbd; ngbd = new my_p4est_node_neighbors_t(hierarchy, nodes);
      ngbd->init_neighbors();

      // Create level set object and reinitialize it:
      my_p4est_level_set_t ls_new(ngbd);
      ls_new.reinitialize_1st_order_time_2nd_order_space(phi.vec);

      // Get the new solid LSF:
      phi_solid.destroy();
      phi_solid.create(p4est,nodes);
      VecScaleGhost(phi.vec,-1.0);
      VecCopyGhost(phi.vec,phi_solid.vec);
      VecScaleGhost(phi.vec,-1.0);

      // --------------------------------------------------------------------------------------------------------------
      // Poisson Problem at Nodes: Setup and solve a Poisson problem on both the liquid and solidified subdomains
      // --------------------------------------------------------------------------------------------------------------

      // Compute the advection term for the RHS: (if we are considering advection) ---------------------
      // --------------------------------------------
      // Get derivative of the external velocity field: (for the semi-lagrangian backtrace)

      vec_and_ptr_t T_l_backtrace(p4est,nodes);

      if (do_advection){
        vec_and_ptr_dim_t v_n_dd[P4EST_DIM];
        foreach_dimension(d){
          v_n_dd[d].create(p4est,nodes);
          ngbd->second_derivatives_central(vel_n.vec[d],v_n_dd[d].vec); // computes derivatives of the dth component of the velocity
        }

        vector <double> xyz_d[P4EST_DIM];

        trajectory_from_np1_to_n(p4est,nodes,ngbd,dt,vel_n.vec,&v_n_dd->vec,xyz_d);

        PetscPrintf(mpi.comm(),"Grabbing the backtraced points: \n");
        // Find the temperature at the backtraced points
        my_p4est_interpolation_nodes_t SL_backtrace_interp(ngbd);
        foreach_local_node(n,nodes){
          double xyz_temp[P4EST_DIM];
          XCODE(xyz_temp[0] = xyz_d[0][n]);
          YCODE(xyz_temp[1] = xyz_d[1][n]);
          ZCODE(xyz_temp[2] = xyz_d[2][n]);

          SL_backtrace_interp.add_point(n,xyz_temp);
        }
        PetscPrintf(mpi.comm(),"Finishes grabbing the backtraced points: \n");

        // Now interpolate the Temperature data:
        PetscPrintf(mpi.comm(),"Attempting to interpolate backtraced Tl: \n");
        SL_backtrace_interp.set_input(T_l_n.vec,interp_bw_grids);
        SL_backtrace_interp.interpolate(T_l_backtrace.vec);

        PetscPrintf(mpi.comm(),"\n");
        PetscPrintf(mpi.comm(),"After Doing Backtrace: ----------");
        PetscPrintf(mpi.comm(),"\n");

        vector <double> T_values = check_T_values(phi,T_l_backtrace,nodes);
        PetscPrintf(mpi.comm(),"T backtrace data: \n");
        PetscPrintf(mpi.comm(),"norm : %0.2f \n",T_values[0]);
        PetscPrintf(mpi.comm(),"max magnitude: %0.2f \n",T_values[1]);
        PetscPrintf(mpi.comm(),"min magnitude: %0.2f \n",T_values[2]);
        PetscPrintf(mpi.comm(),"min value: %0.2f \n",T_values[3]);

      } // end of do_advection if statement

      // Now, create the RHS vector and diagonal vector for both the solid and the liquid sides:
      rhs_Tl.create(p4est,nodes);
      rhs_Ts.create(rhs_Tl.vec);

      rhs_Tl.get_array();
      rhs_Ts.get_array();
      T_l_n.get_array();
      T_s_n.get_array();

      // FIX: There is probably a cheaper way to do this than looping through all nodes? Via setting diag more wisely or etc.

      if(do_advection){T_l_backtrace.get_array();}
      foreach_local_node(n,nodes){
        if(do_advection){
            rhs_Tl.ptr[n]+=T_l_backtrace.ptr[n]/dt;
        }
        else{
            rhs_Tl.ptr[n] = T_l_n.ptr[n]/dt;
          }

        rhs_Ts.ptr[n] = T_s_n.ptr[n]/dt;


      }
      rhs_Tl.restore_array(); rhs_Ts.restore_array();
      T_l_n.restore_array(); T_s_n.restore_array();


      // Now, set up the solver(s):
      solver_Tl = new my_p4est_poisson_nodes_mls_t(ngbd);
      solver_Ts = new my_p4est_poisson_nodes_mls_t(ngbd);

      solver_Tl->add_boundary(MLS_INTERSECTION,phi.vec,phi_dd.vec[0],phi_dd.vec[1],DIRICHLET,bc_interface_val,bc_interface_coeff);
      solver_Ts->add_boundary(MLS_INTERSECTION,phi_solid.vec,phi_solid_dd.vec[0],phi_solid_dd.vec[1],DIRICHLET,bc_interface_val,bc_interface_coeff);

      // Set diagonal and diffusivity:
      solver_Tl->set_diag(1.0/dt);
      solver_Tl->set_mu(alpha_l);
      solver_Tl->set_rhs(rhs_Tl.vec);

      solver_Ts->set_diag(1.0/dt);
      solver_Ts->set_mu(alpha_l);
      solver_Ts->set_rhs(rhs_Ts.vec);

      // Set some other solver properties:
      solver_Tl->set_integration_order(1);
      solver_Tl->set_use_sc_scheme(0);
      solver_Tl->set_cube_refinement(cube_refinement);
      solver_Tl->set_store_finite_volumes(1);

      solver_Ts->set_integration_order(1);
      solver_Ts->set_use_sc_scheme(0);
      solver_Ts->set_cube_refinement(cube_refinement);
      solver_Ts->set_store_finite_volumes(1);

      // Set the wall BC and RHS:
      solver_Tl ->set_wc(wall_bc_type_temp,wall_bc_value_temp);
      solver_Ts ->set_wc(wall_bc_type_temp,wall_bc_value_temp);

      // Preassemble the linear system
      solver_Tl->preassemble_linear_system();
      solver_Ts->preassemble_linear_system();

      // Create vector to hold the solution:
      vec_and_ptr_t T_l_np1; vec_and_ptr_t T_s_np1;
      T_l_np1.create(p4est,nodes);
      T_s_np1.create(T_l_np1.vec);

      // Solve the system:
      solver_Tl->solve(T_l_np1.vec);
      solver_Ts->solve(T_s_np1.vec);

      // Destroy the T_n values now and update them with the solution for the next timestep:
      T_l_n.destroy(); T_s_n.destroy();
      T_l_n.create(p4est,nodes); T_s_n.create(T_l_n.vec);

      VecCopyGhost(T_l_np1.vec,T_l_n.vec);
      VecCopyGhost(T_s_np1.vec,T_s_n.vec);

      if (output_information){
        PetscPrintf(mpi.comm(),"\n");
        PetscPrintf(mpi.comm(),"After Getting Solution: ----------");
        PetscPrintf(mpi.comm(),"\n");

        T_values = check_T_values(phi,T_l_n,nodes);
        MPI_Barrier(mpi.comm());
        MPI_Barrier(mpi.comm());
        for (int i=0; i<4; i++){
            mpi_ret = MPI_Allreduce(MPI_IN_PLACE,&T_values[i],1,MPI_DOUBLE,MPI_MAX,mpi.comm());
            SC_CHECK_MPI(mpi_ret);
          }
        PetscPrintf(mpi.comm(),"Tl data: \n");
        PetscPrintf(mpi.comm(),"norm : %0.2f \n",T_values[0]);
        PetscPrintf(mpi.comm(),"max magnitude: %0.2f \n",T_values[1]);
        PetscPrintf(mpi.comm(),"min magnitude: %0.2f \n",T_values[2]);
        PetscPrintf(mpi.comm(),"min value: %0.2f \n",T_values[3]);

        T_values = check_T_values(phi_solid,T_s_n,nodes);
        MPI_Barrier(mpi.comm());
        MPI_Barrier(mpi.comm());
        for (int i=0; i<4; i++){
            mpi_ret = MPI_Allreduce(MPI_IN_PLACE,&T_values[i],1,MPI_DOUBLE,MPI_MAX,mpi.comm());
            SC_CHECK_MPI(mpi_ret);
          }

        PetscPrintf(mpi.comm(),"Ts data: \n");
        PetscPrintf(mpi.comm(),"norm : %0.2f \n",T_values[0]);
        PetscPrintf(mpi.comm(),"max magnitude: %0.2f \n",T_values[1]);
        PetscPrintf(mpi.comm(),"min magnitude: %0.2f \n",T_values[2]);
        PetscPrintf(mpi.comm(),"min value: %0.2f \n",T_values[3]);
      }


      // Destroy old information: (except phi, which gets updated by the update_p4est function)
      T_l_np1.destroy();
      T_s_np1.destroy();
      phi_solid.destroy();
      phi_dd.destroy();
      phi_solid_dd.destroy();
  // -----------------------------------------------
    } // End of for loop through time

  // destroy the structures
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy      (p4est);
  my_p4est_brick_destroy(conn, &brick);

  w.stop(); w.read_duration();
}

