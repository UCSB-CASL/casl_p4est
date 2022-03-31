#include "my_p4est_stefan_with_fluids.h"


#ifndef P4_TO_P8
#include <src/my_p4est_utils.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_semi_lagrangian.h>

#include <src/my_p4est_log_wrappers.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_trajectory_of_point.h>


#include <src/my_p4est_poisson_nodes_mls.h>
#include <src/my_p4est_poisson_nodes.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_navier_stokes.h>
#include <src/my_p4est_multialloy.h>
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


// -------------------------------------------------------
// Constructor and other auxiliary initializations:
// -------------------------------------------------------

my_p4est_stefan_with_fluids_t::my_p4est_stefan_with_fluids_t()
{


}

// -------------------------------------------------------
// Destructor:
// -------------------------------------------------------


// -------------------------------------------------------
// Functions related to scalar temp/conc problem:
// -------------------------------------------------------
void my_p4est_stefan_with_fluids_t::setup_rhs_for_scalar_temp_conc_problem(){

  // In building RHS, if we are doing advection, we have two options:
  // (1) 1st order -- approx is (dT/dt + u dot grad(T)) ~ (T(n+1) - Td(n))/dt --> so we add Td/dt to the RHS
  // (2) 2nd order -- approx is (dT/dt + u dot grad(T)) ~ alpha*(T(n+1) - Td(n))/dt + beta*(Td(n) - Td(n-1))/dt_nm1
  //                       --> so we add Td(n)*(alpha/dt - beta/dt_nm1) + Td(n-1)*(beta/dt_nm1) to the RHS
  //               -- where alpha and beta are weights of the two timesteps
  // See Semi-Lagrangian backtrace advection schemes for more details

  // If we are not doing advection, then we have:
  // (1) dT/dt = (T(n+1) - T(n)/dt) --> which is a backward euler 1st order approximation (since the RHS is discretized spatially at T(n+1))
  // (2) dT/dt = alpha*laplace(T) ~ (T(n+1) - T(n)/dt) = (1/2)*(laplace(T(n)) + laplace(T(n+1)) )  ,
  //                              in which case we need the second derivatives of the temperature field at time n


  // Establish forcing terms if applicable:
  vec_and_ptr_t forcing_term_liquid;
  vec_and_ptr_t forcing_term_solid;

  if(there_is_user_provided_heat_source){
    forcing_term_liquid.create(p4est_np1, nodes_np1);
    sample_cf_on_nodes(p4est_np1, nodes_np1, *user_provided_external_heat_source[LIQUID_DOMAIN], forcing_term_liquid.vec);

    if(do_we_solve_for_Ts) {
      forcing_term_solid.create(p4est_np1, nodes_np1);
      sample_cf_on_nodes(p4est_np1, nodes_np1, *user_provided_external_heat_source[SOLID_DOMAIN], forcing_term_solid.vec);
    }
  }

  // Prep coefficients if we are doing 2nd order advection:
  // TO-DO: probably should move calculation of these coefficients elsewhere
  if(solve_navier_stokes && advection_sl_order==2){
    advection_alpha_coeff = (2.*dt + dt_nm1)/(dt + dt_nm1);
    advection_beta_coeff = (-1.*dt)/(dt + dt_nm1);
  }
  // Get Ts arrays:
  if(do_we_solve_for_Ts){
    T_s_n.get_array();
    rhs_Ts.get_array();
  }

  // Get Tl arrays:
  rhs_Tl.get_array();
  if(solve_navier_stokes){
    T_l_backtrace_n.get_array();
    if(advection_sl_order ==2) T_l_backtrace_nm1.get_array();
  }
  else{
    T_l_n.get_array();
  }

  if(analytical_IC_BC_forcing_term){
    forcing_term_liquid.get_array();
    if(do_we_solve_for_Ts) forcing_term_solid.get_array();
  }

  phi.get_array();
  // 3-7-22 : Elyce changed from foreach_local_node to foreach_node --> when I visualized rhs it was patchy ...
  foreach_node(n, nodes_np1){
    if(do_we_solve_for_Ts){
      // Backward Euler
      rhs_Ts.ptr[n] = T_s_n.ptr[n]/dt;
    }

    // Now for Tl depending on case:
    if(solve_navier_stokes){
      if(advection_sl_order ==2){
        rhs_Tl.ptr[n] = T_l_backtrace_n.ptr[n]*((advection_alpha_coeff/dt) - (advection_beta_coeff/dt_nm1)) + T_l_backtrace_nm1.ptr[n]*(advection_beta_coeff/dt_nm1);
      }
      else{
        rhs_Tl.ptr[n] = T_l_backtrace_n.ptr[n]/dt;
      }
    }
    else{
      // Backward Euler
      rhs_Tl.ptr[n] = T_l_n.ptr[n]/dt;
    }
    if(analytical_IC_BC_forcing_term){
      // Add forcing terms:
      rhs_Tl.ptr[n]+=forcing_term_liquid.ptr[n];
      if(do_we_solve_for_Ts) rhs_Ts.ptr[n]+=forcing_term_solid.ptr[n];
    }

  }// end of loop over nodes

  // Restore arrays:
  phi.restore_array();

  if(do_we_solve_for_Ts){
    T_s_n.restore_array();
    rhs_Ts.restore_array();
  }

  rhs_Tl.restore_array();
  if(solve_navier_stokes){
    T_l_backtrace_n.restore_array();
    if(advection_sl_order==2) T_l_backtrace_nm1.restore_array();
  }
  else{
    T_l_n.restore_array();
  }

  if(analytical_IC_BC_forcing_term){
    forcing_term_liquid.restore_array();

    if(do_we_solve_for_Ts) {
      forcing_term_solid.restore_array(); forcing_term_solid.destroy();
    }

    // Destroy these if they were created
    forcing_term_liquid.destroy();
  }
}

void my_p4est_stefan_with_fluids_t::do_backtrace_for_scalar_temp_conc_problem(){
  // -------------------
  // A note on notation:
  // -------------------
  // Recall that at this stage, we are computing backtrace points for
  // -- T_n (sampled on the grid np1) and T_nm1 (sampled on the grid n)
  // using the fluid velocities
  // -- v_n_NS (sampled on grid np1) and v_nm1_NS (sampled on the grid n)

  // This notation can be a bit confusing, but stems from the fact that the grid np1 has been chosen around the interface location at time np1,
  // and all the fields at n have been interpolated to this new grid to solve for fields at np1.
  // Thus, while T_n is sampled on the grid np1, it is indeed still the field at time n, simply transferred to the grid used to solve for the np1 fields.


  if(print_checkpoints) PetscPrintf(mpi.comm(),"Beginning to do backtrace \n");

  // Initialize objects we will use in this function:
  // PETSC Vectors for second derivatives
  vec_and_ptr_dim_t T_l_dd, T_l_dd_nm1;
  Vec v_dd[P4EST_DIM][P4EST_DIM];
  Vec v_dd_nm1[P4EST_DIM][P4EST_DIM];

  // Create vector to hold back-trace points:
  vector <double> xyz_d[P4EST_DIM];
  vector <double> xyz_d_nm1[P4EST_DIM];

  // Create the necessary interpolators
  my_p4est_interpolation_nodes_t SL_backtrace_interp(ngbd_np1); /*= NULL;*/
  my_p4est_interpolation_nodes_t SL_backtrace_interp_nm1(ngbd_n);/* = NULL;*/

  // Get the relevant second derivatives
  T_l_dd.create(p4est_np1, nodes_np1);
  ngbd_np1->second_derivatives_central(T_l_n.vec, T_l_dd.vec);

  if(advection_sl_order==2) {
    T_l_dd_nm1.create(p4est_n, nodes_n);
    ngbd_n->second_derivatives_central(T_l_nm1.vec,T_l_dd_nm1.vec);
  }

  foreach_dimension(d){
    foreach_dimension(dd){
      ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &v_dd[d][dd]); CHKERRXX(ierr); // v_n_dd will be a dxdxn object --> will hold the dxd derivative info at each node n
      if(advection_sl_order==2){
        ierr = VecCreateGhostNodes(p4est_n, nodes_n, &v_dd_nm1[d][dd]); CHKERRXX(ierr);
      }
    }
  }

  // v_dd[k] is the second derivative of the velocity components n along cartesian direction k
  // v_dd_nm1[k] is the second derivative of the velocity components nm1 along cartesian direction k

  ngbd_np1->second_derivatives_central(v_n.vec,v_dd[0],v_dd[1],P4EST_DIM);
  if(advection_sl_order ==2){
    ngbd_n->second_derivatives_central(v_nm1.vec, DIM(v_dd_nm1[0], v_dd_nm1[1], v_dd_nm1[2]), P4EST_DIM);
  }

  // Do the Semi-Lagrangian backtrace:
  if(advection_sl_order ==2){
    trajectory_from_np1_to_nm1(p4est_np1, nodes_np1, ngbd_n, ngbd_np1, v_nm1.vec, v_dd_nm1, v_n.vec, v_dd, dt_nm1, dt, xyz_d_nm1, xyz_d);
    if(print_checkpoints) PetscPrintf(p4est_np1->mpicomm,"Completes backtrace trajectory \n");
  }
  else{
    trajectory_from_np1_to_n(p4est_np1, nodes_np1, ngbd_np1, dt, v_n.vec, v_dd, xyz_d);
  }

  // Add backtrace points to the interpolator(s):
  foreach_local_node(n, nodes_np1){
    double xyz_temp[P4EST_DIM];
    double xyz_temp_nm1[P4EST_DIM];

    foreach_dimension(d){
      xyz_temp[d] = xyz_d[d][n];

      if(advection_sl_order ==2){
        xyz_temp_nm1[d] = xyz_d_nm1[d][n];
      }
    } // end of "for each dimension"

    SL_backtrace_interp.add_point(n,xyz_temp);
    if(advection_sl_order ==2 ) SL_backtrace_interp_nm1.add_point(n,xyz_temp_nm1);
  } // end of loop over local nodes

  // Interpolate the Temperature data to back-traced points:
  SL_backtrace_interp.set_input(T_l_n.vec, T_l_dd.vec[0], T_l_dd.vec[1],quadratic_non_oscillatory_continuous_v2);
  SL_backtrace_interp.interpolate(T_l_backtrace_n.vec);

  if(advection_sl_order ==2){
    SL_backtrace_interp_nm1.set_input(T_l_nm1.vec, T_l_dd_nm1.vec[0], T_l_dd_nm1.vec[1], quadratic_non_oscillatory_continuous_v2);
    SL_backtrace_interp_nm1.interpolate(T_l_backtrace_nm1.vec);
  }

  // Destroy velocity derivatives now that not needed:
  foreach_dimension(d){
    foreach_dimension(dd)
    {
      ierr = VecDestroy(v_dd[d][dd]); CHKERRXX(ierr); // v_n_dd will be a dxdxn object --> will hold the dxd derivative info at each node n
      if(advection_sl_order==2) ierr = VecDestroy(v_dd_nm1[d][dd]); CHKERRXX(ierr);
    }
  }

  // Destroy temperature derivatives
  T_l_dd.destroy();
  if(advection_sl_order==2) {
    T_l_dd_nm1.destroy();
  }

  // Clear interp points:
  xyz_d->clear();xyz_d->shrink_to_fit();
  xyz_d_nm1->clear();xyz_d_nm1->shrink_to_fit();

  // Clear and delete interpolators:
  SL_backtrace_interp.clear();
  SL_backtrace_interp_nm1.clear();

  if(print_checkpoints) PetscPrintf(p4est_np1->mpicomm,"Completes backtrace \n");
} // end of "do_backtrace_for_scalar_temp_conc_problem"



// -------------------------------------------------------
// Functions related to interfacial velocity and timestep:
// -------------------------------------------------------



// -------------------------------------------------------
// Functions related to Navier-Stokes problem:
// -------------------------------------------------------




// -------------------------------------------------------
// Functions related to LSF advection/grid update:
// -------------------------------------------------------







// -------------------------------------------------------
// Functions related to LSF regularization:
// -------------------------------------------------------





// -------------------------------------------------------
// Functions related to VTK saving:
// -------------------------------------------------------






// -------------------------------------------------------
// Functions related to save state/load state:
// -------------------------------------------------------












//  class interfacial_bc_temp: public CF_DIM{
//private:
//    my_p4est_node_neighbors_t* ngbd_bc_temp;

//    // Curvature interp:
//    my_p4est_interpolation_nodes_t* kappa_interp;

//    // Normals interp:
//    my_p4est_interpolation_nodes_t* nx_interp;
//    my_p4est_interpolation_nodes_t* ny_interp;
//    // TO-DO: add 3d case


//public:
//    void set_kappa_interp(my_p4est_node_neighbors_t* ngbd_, Vec &kappa){
//      ngbd_bc_temp = ngbd_;
//      kappa_interp = new my_p4est_interpolation_nodes_t(ngbd_bc_temp);
//      kappa_interp->set_input(kappa, linear);

//    }
//    void clear_kappa_interp(){
//      kappa_interp->clear();
//      delete kappa_interp;
//    }
//    void set_normals_interp(my_p4est_node_neighbors_t* ngbd_, Vec &nx, Vec &ny){
//      ngbd_bc_temp = ngbd_;
//      nx_interp = new my_p4est_interpolation_nodes_t(ngbd_bc_temp);
//      nx_interp->set_input(nx, linear);

//      ny_interp = new my_p4est_interpolation_nodes_t(ngbd_bc_temp);
//      ny_interp->set_input(ny, linear);
//    }
//    void clear_normals_interp(){
//      kappa_interp->clear();
//      delete kappa_interp;
//    }
//    double Gibbs_Thomson(double sigma_, DIM(double x, double y, double z)) const {
//      switch(problem_dimensionalization_type){
//      // Note slight difference in condition bw diff nondim types -- T0 vs Tinf
//      case NONDIM_BY_FLUID_VELOCITY:{
//        return (theta_interface - (sigma_/l_char)*((*kappa_interp)(x,y))*(theta_interface + T0/deltaT));
//      }
//      case NONDIM_BY_SCALAR_DIFFUSIVITY:{
//        return (theta_interface - (sigma_/l_char)*((*kappa_interp)(x,y))*(theta_interface + Tinfty/deltaT));
//      }
//      case DIMENSIONAL:{
//        return (Tinterface*(1 - sigma_*((*kappa_interp)(x,y))));
//      }
//      default:{
//        throw std::runtime_error("Gibbs_thomson: unrecognized problem dimensionalization type \n");
//      }
//      }
//    }

//  };


