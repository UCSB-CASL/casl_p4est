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

#include <src/my_p4est_semi_lagrangian.h>


#include <src/my_p4est_poisson_nodes_mls.h>
#include <src/my_p4est_poisson_nodes.h>
#include <src/my_p4est_interpolation_nodes.h>



#include <src/my_p4est_macros.h>
#else
#include <src/my_p8est_utils.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_log_wrappers.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_level_set.h>

#include <src/my_p8est_node_neighbors.h>
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
int lmin = 4;
int lmax = 7;
bool trying_out_with_interface = false;




// Begin defining classes for necessary functions and boundary conditions...
// --------------------------------------------------------------------------------------------------------------
// LEVEL SET FUNCTION:
// --------------------------------------------------------------------------------------------------------------
struct LEVEL_SET : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    //return 0.25 - sqrt(SQR(x) + SQR(y));
    return 0.5 - sqrt(SQR(x) + SQR(y-3*ymin/4));
  }
} level_set;

struct LEVEL_SET_DX : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    return (1./2.)*(1./sqrt(SQR(x) + SQR(y)))*(2*x);
  }
} level_set_dx;

struct LEVEL_SET_DY : CF_DIM {
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    return (1./2.)*(1./sqrt(SQR(x) + SQR(y)))*(2*y);
  }
} level_set_dy;

// --------------------------------------------------------------------------------------------------------------
// PRESCRIBED VELOCITY FIELD AT WHICH THE INTERFACE ADVANCES
// --------------------------------------------------------------------------------------------------------------
struct u_advance : CF_DIM
{ double operator() (double x, double y) const{
  return 0.0;
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
  { return 255.0;
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
//    if ((fabs(y-ymax)<EPS) || (fabs(y-ymin)<EPS) || (fabs(x-xmin)<EPS) || (fabs(x-xmax)<EPS)){
//        if (level_set(DIM(x,y,z)) < EPS){
//          return 298.0;
//          }
//        else{return 255.0;}
//      }

//    if (level_set(DIM(x,y,z)) > EPS){
//        return 255.0;
//      }
//    else{
//        double m = (298.0 - 255.0)/(level_set(DIM(xmin,ymin,z)));
//        return 255.0 + m*level_set(DIM(x,y,z));
//      }

            double m = (298.0 - 255.0)/(level_set(DIM(xmin,ymin,z)));
            return 255.0 + m*level_set(DIM(x,y,z));
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

    double m = (298.0 - 255.0)/(level_set(DIM(xmin,ymin,z)));
    return 255.0 + m*level_set(DIM(x,y,z));
//    if (level_set(DIM(x,y,z)) > EPS){
//        return 255.0;
//      }
//    else{
//        double m = (298.0 - 255.0)/(level_set(DIM(xmin,ymin,z)));
//        return 255.0 + m*level_set(DIM(x,y,z));
//      }
    //return 298.0;//(298.0 - 273.0)*(-level_set(DIM(x,y,z))) + 273.0;
  }
}IC_temp;

// --------------------------------------------------------------------------------------------------------------
// INTERFACIAL TEMPERATURE BOUNDARY CONDITION
// --------------------------------------------------------------------------------------------------------------
class INTERFACE_JUMPFLUX_VALUE_TEMP: public CF_DIM
{
  CF_DIM *phi_x_;
  CF_DIM *phi_y_;
  CF_DIM *phi_z_;
public:
  INTERFACE_JUMPFLUX_VALUE_TEMP(DIM(CF_DIM *phi_x, CF_DIM *phi_y, CF_DIM *phi_z)): DIM(phi_x_(phi_x), phi_y_(phi_y),phi_z_(phi_z)){}
  double operator() (DIM(double x, double y, double z)) const
  {
    double DIM(nx = (*phi_x_)(DIM(x,y,z)),
               ny = (*phi_y_)(DIM(x,y,z)),
               nz = (*phi_z_)(DIM(x,y,z)));
    double norm = sqrt(SUMD(nx*nx,ny*ny,nz*nz));
    nx/=norm; ny/=norm; CODE3D(nz/=norm);

    double flux = SUMD(nx*u_adv(DIM(x,y,z)),ny*v_adv(DIM(x,y,z)),nz*0.0);
    return flux;
  }
};

INTERFACE_JUMPFLUX_VALUE_TEMP interface_jumpflux_value_temp(&level_set_dx,&level_set_dy);

class INTERFACE_JUMP_VALUE_TEMP: public CF_DIM
{
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    return 0.0;
  }
}interface_jump_value_temp;


//class INTERFACE_BC_TYPE_TEMP: public WallBCDIM
//{
//public:
//  BoundaryConditionType operator() (DIM(double, double, double)) const
//  {
//    return DIRICHLET;
//  }
//}interface_bc_type_temp;

BoundaryConditionType interface_bc_type_temp = DIRICHLET;

class NULL_CF:public CF_DIM{
public:
  double operator() (DIM(double x, double y, double z)) const
  {
    return NULL;
  }
} null_cf;

// --------------------------------------------------------------------------------------------------------------
// BEGIN MAIN OPERATION:
// --------------------------------------------------------------------------------------------------------------


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
  double alpha_s = 0.0182;//1.1820e-6; //ice
  double alpha_l = 0.14547; //1.4547e-7; //water

  int cube_refinement = 4;
  interpolation_method interp_bw_grids = quadratic_non_oscillatory_continuous_v2;

  // -----------------------------------------------
  // Create the grid:
  // -----------------------------------------------

  conn = my_p4est_brick_new(n_xyz, xyz_min, xyz_max, &brick, periodic); // same as Daniil

  // create the forest
  p4est = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL); // same as Daniil

  // refine based on distance to a level-set

  splitting_criteria_cf_t sp(lmin, lmax, &level_set); // same as Daniil, minus lipschitz
  p4est->user_pointer = &sp; // save the pointer to the forst splitting criteria
  my_p4est_refine(p4est, P4EST_TRUE, refine_levelset_cf, NULL); // refine the level set according to the splitting criteria

  // partition the forest
  my_p4est_partition(p4est, P4EST_TRUE, NULL); // partition the forest but allow for coarsening --> Daniil does not allow (use P4EST_FALSE)

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

  vec_and_ptr_t phi;
  phi.create(p4est,nodes);
  sample_cf_on_nodes(p4est,nodes,level_set,phi.vec);

  // -----------------------------------------------
  // Initialize the Velocity field:
  // -----------------------------------------------

  vec_and_ptr_dim_t vel_n(p4est,nodes);

  const CF_DIM *vel_cf[P4EST_DIM] = {&u_adv, &v_adv};

  for (int dir = 0; dir<P4EST_DIM;dir++){
      sample_cf_on_nodes(p4est,nodes,*vel_cf[dir],vel_n.vec[dir]);
    }

  // -----------------------------------------------
  // Initialize the Temperature field:
  // -----------------------------------------------
  vec_and_ptr_t tl; vec_and_ptr_t ts;
  tl.create(p4est,nodes);
  ts.create(tl.vec);
  sample_cf_on_nodes(p4est,nodes,IC_temp,tl.vec);
  sample_cf_on_nodes(p4est,nodes,IC_temp,ts.vec);

  vec_and_ptr_t rhs_tl; vec_and_ptr_t rhs_ts;
  vec_and_ptr_t tl_np1; vec_and_ptr_t ts_np1;



  // -----------------------------------------------
  // Initialize the output file:
  // -----------------------------------------------
  int out_idx = 0;
  char outdir[1000];
  sprintf(outdir,"/home/elyce/workspace/projects/multialloy_with_fluids/solidif_with_fluids_output/snapshot_%d",out_idx);


  // -----------------------------------------------
  // Initialize the needed solvers
  // -----------------------------------------------
  my_p4est_poisson_nodes_mls_t *solver_T;  // will solve poisson problem for a variable T, which has jump conditions on the interface

  // -----------------------------------------------
  // Begin stepping through time and advecting that LSF!
  // -----------------------------------------------
  double tf = 0.1;
  int tstep = 0;
  int save = 1;
  double dt = 0.05;

  for (double t = 0; t<tf; t+=dt, tstep++){
      ierr = PetscPrintf(mpi.comm(),"Iteration %d , Time: %0.3f \n ------------------------------------------- \n",tstep,t);

      // --------------------------------------------------------------------------------------------------------------
      // SAVING DATA: Save data every specified amout of timesteps:
      // --------------------------------------------------------------------------------------------------------------

      if (tstep % save ==0){
          out_idx++;
          sprintf(outdir,"/home/elyce/workspace/projects/multialloy_with_fluids/solidif_with_fluids_output/snapshot_%d",out_idx);
          // -----------------------------------------------
          // Get local array to write initial LSF and velocity fields to vtk:
          // -----------------------------------------------

          phi.get_array();
          vel_n.get_array();
          tl.get_array();
          ts.get_array();

          // Write out the data:
          if(trying_out_with_interface){
          my_p4est_vtk_write_all(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,
                                 5,0,outdir,
                                 VTK_POINT_DATA,"phi",phi.ptr,
                                 VTK_POINT_DATA,"vx",vel_n.ptr[0],
                                 VTK_POINT_DATA,"vy",vel_n.ptr[1],
                                 VTK_POINT_DATA,"ts",ts.ptr,
                                 VTK_POINT_DATA,"tl",tl.ptr);
            }
          else{
              my_p4est_vtk_write_all(p4est,nodes,ghost,P4EST_TRUE,P4EST_TRUE,
                                                4,0,outdir,
                                                VTK_POINT_DATA,"phi",phi.ptr,
                                                VTK_POINT_DATA,"vx",vel_n.ptr[0],
                                                VTK_POINT_DATA,"vy",vel_n.ptr[1],
                                                VTK_POINT_DATA,"tl",tl.ptr);}

          phi.restore_array();
          vel_n.restore_array();
          tl.restore_array();
          ts.restore_array();

        }


      // --------------------------------------------------------------------------------------------------------------
      // PART I: POISSON PROBLEM AT NODES : Solve a Poisson problem on the negative subdomain
      // --------------------------------------------------------------------------------------------------------------

      // First, get second derivatives of phi on the grid, which we will need to specify the jump conditions:
      vec_and_ptr_dim_t phi_dd(p4est,nodes);
      ngbd->dxx_central(phi.vec,phi_dd.vec[0]);
      ngbd->dyy_central(phi.vec,phi_dd.vec[1]);

      // Now, get second derivatives of v on the grid, which we will need to specify the semi-lagrangian backtrace:
      //vec_and_ptr_dim_t v_n_dd(p4est,nodes);


      // Next, do advection of the Temperature field under the external velocity field, so we can add it to the RHS of the Poisson system:
      // The semi-lagrangian backtrace will find the backtraced point(s), then you can find the Temperature at the backtraced points via interpolation

      /* Process is as follows:
       * (1) Use my_p4est_trajectory_of_point to find the backtraced points
       * (2) Use interpolation to find temperature values at the backtraced points
       * (3) Add the appropriate values now to the RHS of the poisson problem discretization
       * */


      // Now, create the RHS vector for both the solid and the liquid sides:
      rhs_tl.create(p4est,nodes);
      rhs_ts.create(rhs_tl.vec);

      rhs_tl.get_array();
      rhs_ts.get_array();

      tl.get_array();
      ts.get_array();

      foreach_node(n,nodes){
        rhs_tl.ptr[n] = 0.;
        rhs_ts.ptr[n] = 0.;

        rhs_tl.ptr[n] += tl.ptr[n]/dt;
        rhs_ts.ptr[n] += ts.ptr[n]/dt;
      }
      rhs_tl.restore_array(); rhs_ts.restore_array();
      tl.restore_array(); ts.restore_array();


      // Now, set up the solver:
      solver_T = new my_p4est_poisson_nodes_mls_t(ngbd);
      //solver_T->add_interface(MLS_INTERSECTION,phi.vec,phi_dd.vec,zero_cf,zero_cf);

      if (trying_out_with_interface){
          solver_T->add_boundary(MLS_INTERSECTION,phi.vec,phi_dd.vec[0],phi_dd.vec[1],DIRICHLET,bc_interface_val,bc_interface_coeff);

          // Set diagonal and diffusivity:
          solver_T->set_diag(1.0/dt,1.0/dt);
          solver_T->set_mu(alpha_l,alpha_s);
          solver_T->set_rhs(rhs_tl.vec,rhs_ts.vec);
        }
      else{
          solver_T->set_diag(1.0/dt);
          solver_T->set_mu(alpha_l);
          solver_T->set_rhs(rhs_tl.vec);
        }



      // Set some other solver properties:
      solver_T->set_integration_order(1);
      solver_T->set_use_sc_scheme(0);
      solver_T->set_cube_refinement(cube_refinement);
      solver_T->set_store_finite_volumes(1);

      //solver_T->set_rhs()
      // Set the wall BC and RHS:
      solver_T ->set_wc(wall_bc_type_temp,wall_bc_value_temp);


      // Preassemble the linear system
      solver_T->preassemble_linear_system();

      // Create vector to hold the solution:
      vec_and_ptr_t soln;
      soln.create(tl.vec);

      // Get norm of the temperature solution to compare before and after extension:
      tl.get_array();
      double tl_norm = 0.0;
      double max_val = 0.0;
      foreach_node(n,nodes){
        tl_norm+= SQR(tl.ptr[n]);
        max_val = max(max_val,tl.ptr[n]);
      }
      MPI_Allreduce(&tl_norm,&tl_norm,1,MPI_DOUBLE,MPI_SUM,mpi.comm());
      MPI_Allreduce(&max_val,&max_val,1,MPI_DOUBLE,MPI_MAX,mpi.comm());


      tl_norm = sqrt(tl_norm);
      PetscPrintf(mpi.comm(),"Tl norm before solution is computed: %0.4f \n",tl_norm);
      PetscPrintf(mpi.comm(),"Tl max val before solution is computed: %0.4f \n",max_val);



      // Solve the system:
      solver_T->solve(soln.vec);


      // Copy the solution over to tl and ts:

      tl.get_array();
      ts.get_array();
      phi.get_array();
      soln.get_array();
      foreach_node(n,nodes){
        if(phi.ptr[n] < 0) tl.ptr[n] = soln.ptr[n];
        else{
            if (trying_out_with_interface){
                ts.ptr[n] = soln.ptr[n];
              }
          }
      }
      tl.restore_array();
      ts.restore_array();
      phi.restore_array();
      soln.restore_array();

      // Destroy unneeded things now:
      soln.destroy();
      rhs_tl.destroy();
      rhs_ts.destroy();

//      ierr = PetscPrintf(mpi.comm(),"Tl before extension: \n");

//      ierr = VecView(tl.vec,PETSC_VIEWER_STDOUT_WORLD);


      // Get norm of the temperature solution to compare before and after extension:
      tl.get_array();
      tl_norm = 0.0;
      max_val = 0.0;
      foreach_node(n,nodes){
        tl_norm+= SQR(tl.ptr[n]);
        max_val = max(max_val,tl.ptr[n]);
      }
      MPI_Allreduce(&tl_norm,&tl_norm,1,MPI_DOUBLE,MPI_SUM,mpi.comm());
      MPI_Allreduce(&max_val,&max_val,1,MPI_DOUBLE,MPI_MAX,mpi.comm());


      tl_norm = sqrt(tl_norm);
      PetscPrintf(mpi.comm(),"Tl norm before field extension: %0.4f \n",tl_norm);
      PetscPrintf(mpi.comm(),"Tl max val before field extension: %0.4f \n",max_val);

      tl.restore_array();


      vec_and_ptr_dim_t tl_d(p4est,nodes);
      vec_and_ptr_dim_t tl_dd(p4est,nodes);
      ngbd->dxx_central(tl.vec,tl_dd.vec[0]);
      ngbd->dyy_central(tl.vec,tl_dd.vec[1]);

      // Extend the fields over the interface:
      my_p4est_level_set_t ls(ngbd);
      ls.extend_Over_Interface_TVD_Full(phi.vec,tl.vec,50,2,1.e-9);




      // Get norm of the temperature solution to compare before and after extension:
      tl.get_array();
      tl_norm = 0.0;
      max_val = 0.0;
      foreach_node(n,nodes){
        tl_norm+= SQR(tl.ptr[n]);
        max_val = max(max_val,tl.ptr[n]);
      }

      MPI_Allreduce(&tl_norm,&tl_norm,1,MPI_DOUBLE,MPI_SUM,mpi.comm());
      MPI_Allreduce(&max_val,&max_val,1,MPI_DOUBLE,MPI_MAX,mpi.comm());

      tl_norm = sqrt(tl_norm);
      PetscPrintf(mpi.comm(),"\n Tl norm after field extension: %0.4f \n",tl_norm);
      PetscPrintf(mpi.comm(),"Tl max val after field extension: %0.4f \n",max_val);

      tl.restore_array();



         //--> Note: we briefly flip the sign of phi when extending ts over the interface, bc of how extend over interface works
        //-->        since we need to extend ts INTO the negative subdomain FROM the positive subdomain

//      ierr = PetscPrintf(mpi.comm(),"Tl after extension: \n");

//      ierr = VecView(tl.vec,PETSC_VIEWER_STDOUT_WORLD);
      if(trying_out_with_interface){
        VecScaleGhost(phi.vec,-1.0);
        ls.extend_Over_Interface_TVD_Full(phi.vec,ts.vec,50,2);
        VecScaleGhost(phi.vec,-1.0);
        }


      // --------------------------------------------------------------------------------------------------------------
      // PART II: ADVECTING THE LSF UNDER A VELOCITY FIELD -- Advect LSF and update grid
      // --------------------------------------------------------------------------------------------------------------

      // Make a copy of the grid objects for the next timestep:
      p4est_np1 = p4est_copy(p4est,P4EST_FALSE); // copy the grid but not the data
      ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
      nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

      // Create the semi-lagrangian object and do the advection:
      my_p4est_semi_lagrangian_t sl(&p4est_np1, &nodes_np1, &ghost_np1, ngbd); // is this really the correct way to do this?

      // Advect the grid under the velocity field:
      sl.update_p4est(vel_n.vec,dt,phi.vec);

      // (^^^) This operation erases the old phi and replaces it with phi(tn+1)
      // This will return the new grid which has been refined/coarsened and repartitioned

      // --------------------------------------------------------------------------------------------------------------
      // UPDATING NECESSARY FIELDS ONTO THE NEW GRID
      // --------------------------------------------------------------------------------------------------------------


      // Update the velocity field onto the new grid:
      vel_n.destroy();
      vel_n.create(p4est_np1,nodes_np1);
      for (int dir=0;dir<P4EST_DIM;dir++){
          sample_cf_on_nodes(p4est_np1,nodes_np1,*vel_cf[dir],vel_n.vec[dir]);
        }

      // Interpolate the Temperature values onto the new grid for the next timestep:
      my_p4est_interpolation_nodes_t  interp_nodes(ngbd);
      double xyz[P4EST_DIM];

      // Grab the points on the new grid that we want to interpolate to:

      foreach_node(n,nodes_np1){
          node_xyz_fr_n(n,p4est_np1,nodes_np1,xyz);
          interp_nodes.add_point(n,xyz);
      }

      // Create objects to hold the interpolated field at the next time step
      tl_np1.create(phi.vec);


      interp_nodes.set_input(tl.vec, interp_bw_grids); interp_nodes.interpolate(tl_np1.vec);
      if(trying_out_with_interface){
        ts_np1.create(phi.vec);
        interp_nodes.set_input(ts.vec, interp_bw_grids); interp_nodes.interpolate(ts_np1.vec);}

      // Update the objects with the new grid's values:
      tl.destroy();
      tl.create(phi.vec);  // note that the phi object was automatically transferred to the new grid
                                              //already via sl.update_p4est
      if (trying_out_with_interface){
      ts.destroy();ts.create(phi.vec);}

      // Copy updated solution over for the next timestep:
      VecCopy(tl_np1.vec,tl.vec);
      tl_np1.destroy();

      if(trying_out_with_interface){
        VecCopy(ts_np1.vec,ts.vec);
        ts_np1.destroy();
        }


      // --------------------------------------------------------------------------------------------------------------
      // DELETING THE OLD GRID OBJECTS AND REINITIALIZING THE NEW LSF:
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

    } // End of for loop through time


  // -----------------------------------------------

  // destroy the structures
  p4est_nodes_destroy(nodes);
  p4est_ghost_destroy(ghost);
  p4est_destroy      (p4est);
  my_p4est_brick_destroy(conn, &brick);

  w.stop(); w.read_duration();
}

