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
// Functions related to scalar temp/conc problem: ( in order of their usage in the main step)
// -------------------------------------------------------
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


void my_p4est_stefan_with_fluids_t::poisson_nodes_step_for_scalar_temp_conc_problem(){
  // Create solvers:
  solver_Tl = new my_p4est_poisson_nodes_mls_t(ngbd_np1);
  if(do_we_solve_for_Ts) solver_Ts = new my_p4est_poisson_nodes_mls_t(ngbd_np1);

  // Add the appropriate interfaces and interfacial boundary conditions:
  solver_Tl->add_boundary(MLS_INTERSECTION, phi.vec,
                          phi_dd.vec[0], phi_dd.vec[1],
                          *bc_interface_type_temp[LIQUID_DOMAIN],
                          *bc_interface_val_temp[LIQUID_DOMAIN], *bc_interface_robin_coeff_temp[LIQUID_DOMAIN]);

  if(do_we_solve_for_Ts){
    solver_Ts->add_boundary(MLS_INTERSECTION, phi_solid.vec,
                            phi_solid_dd.vec[0], phi_solid_dd.vec[1],
                            *bc_interface_type_temp[SOLID_DOMAIN],
                            *bc_interface_val_temp[SOLID_DOMAIN],
                            *bc_interface_robin_coeff_temp[SOLID_DOMAIN]);
  }

  if(there_is_a_substrate){
    // Need to add this is the event that phi collapses onto the substrate and we need the phi_substrate BC to take over in that region
    solver_Tl->add_boundary(MLS_INTERSECTION, phi_substrate.vec,
                            phi_substrate_dd.vec[0], phi_substrate_dd.vec[1],
                            *bc_interface_type_temp_substrate[LIQUID_DOMAIN],
                            *bc_interface_val_temp_substrate[LIQUID_DOMAIN],
                            *bc_interface_robin_coeff_temp_substrate[LIQUID_DOMAIN]);
    if(do_we_solve_for_Ts){
      // Need to add this to fully define the solid domain (assuming solid is sitting on substrate, thus bounded by liquid and substrate)
      solver_Ts->add_boundary(MLS_INTERSECTION, phi_substrate.vec,
                              phi_substrate_dd.vec[0], phi_substrate_dd.vec[1],
                              *bc_interface_type_temp_substrate[SOLID_DOMAIN],
                              *bc_interface_val_temp_substrate[SOLID_DOMAIN],
                              *bc_interface_robin_coeff_temp_substrate[SOLID_DOMAIN]);
    }
  }

  // Set diagonal for Tl:
  if(solve_navier_stokes){ // Cases with advection use semi lagrangian advection discretization in time
    if(advection_sl_order ==2){ // 2nd order semi lagrangian (BDF2 coefficients)
      solver_Tl->set_diag(advection_alpha_coeff/dt);
    }
    else{ // 1st order semi lagrangian (Backward Euler but with backtrace)
      solver_Tl->set_diag(1./dt);
    }
  }
  else{ // Cases with no temperature advection
    // Backward Euler
    solver_Tl->set_diag(1./dt);
  }

  if(do_we_solve_for_Ts){
    // Set diagonal for Ts:
    // Backward Euler
    solver_Ts->set_diag(1./dt);
  }
  switch(problem_dimensionalization_type){
  case NONDIM_BY_FLUID_VELOCITY:{
    if(!is_dissolution_case){
      solver_Tl->set_mu(1./Pe);
      if(do_we_solve_for_Ts) solver_Ts->set_mu((1./Pe)*(alpha_s/alpha_l));
    }
    else{
      solver_Tl->set_mu(1./Pe);
      if(do_we_solve_for_Ts) solver_Ts->set_mu((1./Pe)*(Ds/Dl));
    }
    break;
  }
  case NONDIM_BY_SCALAR_DIFFUSIVITY:{
    if(!is_dissolution_case){
      solver_Tl->set_mu(1.);
      if(do_we_solve_for_Ts) solver_Ts->set_mu((alpha_s/alpha_l));
    }
    else{
      solver_Tl->set_mu(1.);
      if(do_we_solve_for_Ts) solver_Ts->set_mu((Ds/Dl));
    }
    break;
  }
  case DIMENSIONAL:{
    if(!is_dissolution_case){
      solver_Tl->set_mu(alpha_l);
      if(do_we_solve_for_Ts) solver_Ts->set_mu(alpha_s);
    }
    else{
      solver_Tl->set_mu(Dl);
      if(do_we_solve_for_Ts) solver_Ts->set_mu(Ds);
    }
    break;
  }
  default:{
    throw std::runtime_error("main_2d:poisson_step: unrecognized problem dimensionalization type when setting diffusion coefficients for poisson solver \n");
    break;
  }
  }


  // Set RHS:
  solver_Tl->set_rhs(rhs_Tl.vec);
  if(do_we_solve_for_Ts) solver_Ts->set_rhs(rhs_Ts.vec);

  // Set some other solver properties:
  solver_Tl->set_integration_order(1);
  solver_Tl->set_use_sc_scheme(0);
  solver_Tl->set_cube_refinement(cube_refinement);
  solver_Tl->set_store_finite_volumes(0);
  if(do_we_solve_for_Ts){
    solver_Ts->set_integration_order(1);
    solver_Ts->set_use_sc_scheme(0);
    solver_Ts->set_cube_refinement(cube_refinement);
    solver_Ts->set_store_finite_volumes(0);
  }


  // Set the wall BC and RHS:
  solver_Tl->set_wc(*bc_wall_type_temp[LIQUID_DOMAIN], *bc_wall_value_temp[LIQUID_DOMAIN]);
  if(do_we_solve_for_Ts) solver_Ts->set_wc(*bc_wall_type_temp[SOLID_DOMAIN], *bc_wall_value_temp[SOLID_DOMAIN]);

  // Preassemble the linear system
  solver_Tl->preassemble_linear_system();

  if(do_we_solve_for_Ts) solver_Ts->preassemble_linear_system();

  // Solve the system:
  solver_Tl->solve(T_l_n.vec, false, true, KSPBCGS, PCHYPRE);
  if(do_we_solve_for_Ts) solver_Ts->solve(T_s_n.vec, false, true, KSPBCGS, PCHYPRE);

  // Delete solvers:
  delete solver_Tl;
  if(do_we_solve_for_Ts) delete solver_Ts;
} // end of "poisson_nodes_step_for_scalar_temp_conc_problem()"


void my_p4est_stefan_with_fluids_t::setup_and_solve_poisson_nodes_problem_for_scalar_temp_conc()
{
  PetscErrorCode ierr;

  // -------------------------------
  // Create all vectors that will be used
  // strictly for the stefan step
  // (aka created and destroyed in stefan step)
  // -------------------------------

  // Solid LSF:
  phi_solid.create(p4est_np1,nodes_np1);

  //Curvature and normal for BC's and setting up solver:
  if(interfacial_temp_bc_requires_normal || interfacial_temp_bc_requires_curvature) normal.create(p4est_np1,nodes_np1);
  if(interfacial_temp_bc_requires_curvature)curvature.create(p4est_np1,nodes_np1);

  // Second derivatives of LSF's (for solver):
  phi_solid_dd.create(p4est_np1,nodes_np1);
  phi_dd.create(p4est_np1,nodes_np1);

  if(there_is_a_substrate){
    phi_substrate_dd.create(p4est_np1,nodes_np1);
  }
  if(solve_navier_stokes){
    T_l_backtrace_n.create(p4est_np1,nodes_np1);
    if(advection_sl_order ==2){
      T_l_backtrace_nm1.create(p4est_np1,nodes_np1);
    }
  }
  // Create arrays to hold the RHS:
  rhs_Tl.create(p4est_np1,nodes_np1);
  if(do_we_solve_for_Ts) rhs_Ts.create(p4est_np1,nodes_np1);

  // -------------------------------
  // Compute the normal and curvature of the interface
  //-- curvature is used in some of the interfacial boundary condition(s) on temperature
  // -------------------------------

  if(print_checkpoints) PetscPrintf(mpi.comm(),"Computing normal and curvature ... \n");
  // Get the new solid LSF:
  VecCopyGhost(phi.vec, phi_solid.vec);
  VecScaleGhost(phi_solid.vec, -1.0);

  // Compute normals on the interface (if needed) :
  if(interfacial_temp_bc_requires_normal || interfacial_temp_bc_requires_curvature){
    compute_normals(*ngbd_np1, phi_solid.vec, normal.vec); // normal here is outward normal of solid domain

    // Feed the normals if relevant
    if(interfacial_temp_bc_requires_normal){
      for(unsigned char d=0; d<2; d++){
        bc_interface_val_temp[d]->set_normals_interp(ngbd_np1, normal.vec[0], normal.vec[1]);
      }
    }
  }

  // Compute curvature if needed and feed to bc object:
  if(interfacial_temp_bc_requires_curvature){
    // We need curvature of the solid domain, so we use phi_solid and negative of normals
    compute_mean_curvature(*ngbd_np1, normal.vec, curvature.vec);

    for(unsigned char d=0;d<2;d++){
      bc_interface_val_temp[d]->set_kappa_interp(ngbd_np1, curvature.vec);
    }
  }


  // -------------------------------
  // Get most updated derivatives of the LSF's (on current grid)
  // -------------------------------
  if(print_checkpoints) PetscPrintf(mpi.comm(),"Beginning Poisson problem ... \n");

  // Get derivatives of liquid and solid LSF's
  if (print_checkpoints) PetscPrintf(mpi.comm(),"New solid LSF acquired \n");
  ngbd_np1->second_derivatives_central(phi.vec, phi_dd.vec);
  ngbd_np1->second_derivatives_central(phi_solid.vec, phi_solid_dd.vec);

  // Get inner LSF and derivatives if required:
  if(there_is_a_substrate){
    ngbd_np1->second_derivatives_central(phi_substrate.vec, phi_substrate_dd.vec);
  }

  // -------------------------------
  // Compute advection terms (if applicable):
  // -------------------------------
  if (solve_navier_stokes){
    if(print_checkpoints) PetscPrintf(mpi.comm(),"Computing advection terms ... \n");
    do_backtrace_for_scalar_temp_conc_problem();
  } // end of solve_navier_stokes if statement

  // -------------------------------
  // Set up the RHS for Poisson step:
  // -------------------------------
  if(print_checkpoints) PetscPrintf(mpi.comm(),"Setting up RHS for Poisson problem ... \n");

  setup_rhs_for_scalar_temp_conc_problem();


  // -------------------------------
  // Execute the Poisson step:
  // -------------------------------
  // Slide Temp fields:
  if(solve_navier_stokes && advection_sl_order==2){
    T_l_nm1.destroy();
    T_l_nm1.create(p4est_np1, nodes_np1);
    ierr = VecCopyGhost(T_l_n.vec, T_l_nm1.vec);CHKERRXX(ierr);
  }
  // Solve Poisson problem:
  if(print_checkpoints) PetscPrintf(mpi.comm(),"Beginning Poisson problem solution step... \n");

  poisson_nodes_step_for_scalar_temp_conc_problem();

  if(print_checkpoints) PetscPrintf(mpi.comm(),"Poisson step completed ... \n");

  // -------------------------------
  // Clear interfacial BC if needed (curvature, normals, or both depending on example)
  // -------------------------------
  if(interfacial_temp_bc_requires_curvature){
    for(unsigned char d=0;d<2;++d){
      if (bc_interface_val_temp[d]!=NULL){
        bc_interface_val_temp[d]->clear_kappa_interp();
      }
    }
  }
  if(interfacial_temp_bc_requires_normal){
    for(unsigned char d=0;d<2;++d){
      if (bc_interface_val_temp[d]!=NULL){
        bc_interface_val_temp[d]->clear_normals_interp();
      }
    }
  }

  // -------------------------------
  // Destroy all vectors
  // that were used strictly for the
  // stefan step (aka created and destroyed in stefan step)
  // -------------------------------
  // Solid LSF:
  phi_solid.destroy();

  // Curvature and normal for BC's and setting up solver:
  if(interfacial_temp_bc_requires_normal || interfacial_temp_bc_requires_curvature) normal.destroy();
  if(interfacial_temp_bc_requires_curvature) curvature.destroy();

  // Second derivatives of LSF's (for solver):
  phi_solid_dd.destroy();
  phi_dd.destroy();

  if(there_is_a_substrate){
    phi_substrate_dd.destroy();
  }

  if(solve_navier_stokes){
    T_l_backtrace_n.destroy();
    if(advection_sl_order ==2){
      T_l_backtrace_nm1.destroy();
    }
  }

  // Destroy arrays to hold the RHS:
  rhs_Tl.destroy();
  if(do_we_solve_for_Ts) rhs_Ts.destroy();

} // end of "setup_and_solve_poisson_nodes_problem_for_scalar_temp_conc()"


// -------------------------------------------------------
// Functions related to interfacial velocity and timestep:
// -------------------------------------------------------
double my_p4est_stefan_with_fluids_t::interfacial_velocity_expression(double Tl_d, double Ts_d){
  switch(problem_dimensionalization_type){
  // Note: removed curvature from Stefan condition after discussing w frederic and looking at Daniil's thesis 11/24/2020
  case NONDIM_BY_FLUID_VELOCITY:{
    if(!is_dissolution_case){
      return ( -1.*(St/Pe)*(alpha_s/alpha_l) * ((k_l/k_s)*Tl_d - Ts_d) );
    }
    else{
      return -1.*(gamma_diss/Pe)*Tl_d;
    }
  }
  case NONDIM_BY_SCALAR_DIFFUSIVITY:{
    if(!is_dissolution_case){
      return ( -1.*(St)*(alpha_s/alpha_l)*( (k_l/k_s)*Tl_d - Ts_d ) );
    }
    else{
      return -1.*gamma_diss*Tl_d;
    }
  }
  case DIMENSIONAL:{
    if(!is_dissolution_case){
      return (k_s*Ts_d -k_l*Tl_d)/(L*rho_s);
    }
    else{
      return -1.*molar_volume_diss*(Dl/stoich_coeff_diss)*Tl_d;
    }
  }

  default:{
    throw std::invalid_argument("interfacial_velocity_expression: Unrecognized stefan condition type case \n");
  }
  }
} // end of "interfacial_velocity_expression()"

bool my_p4est_stefan_with_fluids_t::compute_interfacial_velocity(){
  // Some vec_and_ptrs owned by this fxn:
  vec_and_ptr_t vgamma_n;

  // Begin calculation:
  if(!force_interfacial_velocity_to_zero){
    // Cut the extension band in half for region to actually compute vgamma:
    extension_band_extend_/=2;


    // Get the first derivatives to compute the jump
    T_l_d.create(p4est_np1, nodes_np1);
    ngbd_np1->first_derivatives_central(T_l_n.vec, T_l_d.vec);

    if(do_we_solve_for_Ts){
      T_s_d.create(T_l_d.vec);
      ngbd_np1->first_derivatives_central(T_s_n.vec, T_s_d.vec);
    }

    // Create vgamma and normals, and compute normals:
    vgamma_n.create(p4est_np1, nodes_np1);
    normal.create(p4est_np1, nodes_np1);
    // TO-DO: not sure how important this is, but it's possible we compute normals 2x per timestep in some cases which is not very efficient. Consider changing this later. Would need to be handled in main loop.
    // Could add a boolean flag for (are_normals_computed) or something
    compute_normals(*ngbd_np1, phi.vec, normal.vec);

    // Create vector to hold the jump values:
    jump.create(p4est_np1, nodes_np1);

    // Get arrays:
    normal.get_array();
    vgamma_n.get_array();
    jump.get_array();
    T_l_d.get_array();
    if(do_we_solve_for_Ts) T_s_d.get_array();
    phi.get_array();

    // First, compute jump in the layer nodes:
    for(size_t i=0; i<ngbd_np1->get_layer_size();i++){
      p4est_locidx_t n = ngbd_np1->get_layer_node(i);

      if(fabs(phi.ptr[n])<extension_band_extend_){ // TO-DO: should be nondim for ALL cases

        vgamma_n.ptr[n] = 0.; // Initialize
        foreach_dimension(d){
          jump.ptr[d][n] = interfacial_velocity_expression(T_l_d.ptr[d][n], do_we_solve_for_Ts?T_s_d.ptr[d][n]:0.);

          // Calculate V_gamma,n using dot product:
          vgamma_n.ptr[n] += jump.ptr[d][n] * normal.ptr[d][n];
        } // end of loop over dimensions

        // Now, go back and set jump equal to the enforced normal velocity (a scalar) multiplied by the normal --> to get a velocity vector:
        foreach_dimension(d){
          jump.ptr[d][n] = vgamma_n.ptr[n] * normal.ptr[d][n];
        }
      }
    }

    // Begin updating the ghost values of the layer nodes:
    foreach_dimension(d){
      VecGhostUpdateBegin(jump.vec[d],INSERT_VALUES,SCATTER_FORWARD);
    }

    // Compute the jump in the local nodes:
    for(size_t i = 0; i<ngbd_np1->get_local_size();i++){
      p4est_locidx_t n = ngbd_np1->get_local_node(i);
      if(fabs(phi.ptr[n])<extension_band_extend_){
        vgamma_n.ptr[n] = 0.; // initialize
        foreach_dimension(d){
          jump.ptr[d][n] = interfacial_velocity_expression(T_l_d.ptr[d][n], do_we_solve_for_Ts?T_s_d.ptr[d][n]:0.);

          // calculate the dot product to find V_gamma,n
          vgamma_n.ptr[n] += jump.ptr[d][n] * normal.ptr[d][n];

        } // end over loop on dimensions

        // Now, go back and set jump equal to the enforced normal velocity (a scalar) multiplied by the normal --> to get a velocity vector:
        foreach_dimension(d){
          jump.ptr[d][n] = vgamma_n.ptr[n] * normal.ptr[d][n];
        }
      }
    }

    // Finish updating the ghost values of the layer nodes:
    foreach_dimension(d){
      VecGhostUpdateEnd(jump.vec[d],INSERT_VALUES,SCATTER_FORWARD);
    }

    // Restore arrays:
    jump.restore_array();
    T_l_d.restore_array();
    if(do_we_solve_for_Ts) T_s_d.restore_array();

    // Elyce trying something:
    normal.restore_array();
    normal.destroy();
    vgamma_n.restore_array();
    vgamma_n.destroy();


    // Extend the interfacial velocity to the whole domain for advection of the LSF:
    foreach_dimension(d){
      ls->extend_from_interface_to_whole_domain_TVD((there_is_a_substrate? phi_eff.vec : phi.vec),
                                                   jump.vec[d], v_interface.vec[d]); // , 20/*, NULL, 2., 4.*/);
    }


    // Set to zero if we are inside the substrate:
    if(there_is_a_substrate){
      phi_substrate.get_array();
      v_interface.get_array();

      // Layer nodes:
      for(size_t i=0; i<ngbd_np1->get_layer_size();i++){
        p4est_locidx_t n = ngbd_np1->get_layer_node(i);

        foreach_dimension(d){
          if(phi_substrate.ptr[n]>0.){
            v_interface.ptr[d][n] = 0.;
          }
        }
      }

      // Begin communication:
      // Finish updating the ghost values of the layer nodes:
      foreach_dimension(d){
        VecGhostUpdateBegin(v_interface.vec[d],INSERT_VALUES,SCATTER_FORWARD);
      }
      // Local nodes:
      for(size_t i = 0; i<ngbd_np1->get_local_size();i++){
        p4est_locidx_t n = ngbd_np1->get_local_node(i);
        foreach_dimension(d){
          if(phi_substrate.ptr[n]>0.){
            v_interface.ptr[d][n] = 0.;
          }
        }
      }
      // End communication:
      // Finish updating the ghost values of the layer nodes:
      foreach_dimension(d){
        VecGhostUpdateEnd(v_interface.vec[d],INSERT_VALUES,SCATTER_FORWARD);
      }
      v_interface.get_array();
      phi_substrate.restore_array();
    }

    // Scale v_interface computed by appropriate sign if we are doing the coupled test case:
    if(fabs(scale_vgamma_by - 1.)>EPS){
      foreach_dimension(d){
        VecScaleGhost(v_interface.vec[d], scale_vgamma_by);
      }
    }

    // Destroy values once no longer needed:
    T_l_d.destroy();
    if(do_we_solve_for_Ts) T_s_d.destroy();
    jump.destroy();

  }
  else{ // Case where we are forcing interfacial velocity to zero
    foreach_dimension(d){
      VecScaleGhost(v_interface.vec[d], 0.0);
    }
  }


  bool did_crash = false;
  if(v_interface_max_norm>v_interface_max_allowed){
    did_crash=true;
  }
  return did_crash;


} // end of "compute_interfacial_velocity()"

void my_p4est_stefan_with_fluids_t::compute_timestep(){

  // Initialize variables and set max vint if known:
  double max_v_norm = 0.0;
  double global_max_vnorm = 0.0;

  // Compute dt_Stefan (and interfacial velocity if needed)
  if(solve_stefan){
    // Check the values of v_interface locally:
    v_interface.get_array();
    phi.get_array();
    foreach_local_node(n, nodes_np1){
      if (fabs(phi.ptr[n]) < uniform_band*dxyz_close_to_interface){
        max_v_norm = MAX(max_v_norm,sqrt(SQR(v_interface.ptr[0][n]) + SQR(v_interface.ptr[1][n])));
      }
    }
    v_interface.restore_array();
    phi.restore_array();

    // Get the maximum v norm across all the processors:
    int mpi_ret = MPI_Allreduce(&max_v_norm,&global_max_vnorm,1,MPI_DOUBLE,MPI_MAX,p4est_np1->mpicomm);
    SC_CHECK_MPI(mpi_ret);


    // Compute new Stefan timestep:
    dt_Stefan = cfl_Stefan*MIN(dxyz_smallest[0], dxyz_smallest[1])/global_max_vnorm;
  } // end of if solve stefan

  // Compute dt_NS if necessary
  if(solve_navier_stokes){
    ns->compute_dt();
    dt_NS = ns->get_dt();
    // Address the case where we are loading a simulation state
    if(tstep==load_tstep){
      dt_NS=dt_nm1;
    }
  }

  // Compute the timestep that will be used depending on what physics we have:
  if(solve_stefan && solve_navier_stokes){
    // Take the minimum timestep of the NS and Stefan (dt_Stefan computed previously):
    dt = MIN(dt_Stefan,dt_NS);
    dt = MIN(dt, dt_max_allowed);
  }
  else if(solve_stefan && !solve_navier_stokes){
    dt = MIN(dt_Stefan, dt_max_allowed);
  }
  else if(!solve_stefan && solve_navier_stokes){
    dt = MIN(dt_NS, dt_max_allowed);
  }
  else{
    throw std::runtime_error("setting the timestep : you are not solving any of the possible physics ... \n");
  }

  v_interface_max_norm = global_max_vnorm;

  // TO-DO: want to move last tstep and clipping business to outside of the class,
  // don't forget to make that consistent in the main

//  // Clip the timestep if we are near the end of our simulation, to get the proper end time:
//  if((tn + dt > tfinal) && (last_tstep<0)){

//    dt = max(tfinal - tn,dt_min_allowed);

//    // if time remaining is too small for one more step, end here. otherwise, do one more step and clip timestep to end on exact ending time
//    if(fabs(tfinal-tn)>dt_min_allowed){
//      last_tstep = tstep+1;
//    }
//    else{
//      last_tstep = tstep;
//    }

//    PetscPrintf(mpicomm,"Final tstep will be %d \n",last_tstep);
//  }

  // Print the interface velocity info:
  PetscPrintf(mpi.comm(),"\n"
                       "Computed interfacial velocity: \n"
                       " - %0.3e [nondim] "
                       " - %0.3e [m/s] "
                       " - %0.3e [mm/s] \n",
              v_interface_max_norm,
              v_interface_max_norm*vel_nondim_to_dim,
              v_interface_max_norm*vel_nondim_to_dim*1000.);

  // Print the timestep info:
  PetscPrintf(mpi.comm(),"\n"
                       "Computed timestep: \n"
                       " - dt used: %0.3e "
                       " - dt_Stefan: %0.3e "
                       " - dt_NS : %0.3e  "
                       " - dt_max_allowed : %0.3e \n"
                       " - dxyz close to interface : %0.3e "
                       "\n",
              dt, dt_Stefan, dt_NS, dt_max_allowed,
              dxyz_close_to_interface);
} // end of "compute_timestep()"


// -------------------------------------------------------
// Functions related to Navier-Stokes problem:
// -------------------------------------------------------

void my_p4est_stefan_with_fluids_t::set_ns_parameters(){
  switch(problem_dimensionalization_type){
  case NONDIM_BY_FLUID_VELOCITY:{
    ns->set_parameters((1./Re), 1.0, NS_advection_sl_order, uniform_band, vorticity_threshold, cfl_NS);
    break;
  }
  case NONDIM_BY_SCALAR_DIFFUSIVITY:{
    if(!is_dissolution_case){
      ns->set_parameters(Pr, 1.0, NS_advection_sl_order, uniform_band, vorticity_threshold, cfl_NS);
    }
    else{
      ns->set_parameters(Sc, 1.0, NS_advection_sl_order, uniform_band, vorticity_threshold, cfl_NS);
    }
    break;
  }
  case DIMENSIONAL:{
    ns->set_parameters(mu_l, rho_l, NS_advection_sl_order, uniform_band, vorticity_threshold, cfl_NS);
  }
  }// end switch case

} // end of "set_ns_parameters()"

void my_p4est_stefan_with_fluids_t::initialize_ns_solver(){

  // Create the initial neigbhors and faces (after first step, NS grid update will handle this internally)
  ngbd_c_np1 = new my_p4est_cell_neighbors_t(hierarchy_np1);
  faces_np1 = new my_p4est_faces_t(p4est_np1, ghost_np1, &brick, ngbd_c_np1);

  // Create the solver
  ns = new my_p4est_navier_stokes_t(ngbd_n,ngbd_np1,faces_np1);


  // Set the LSF:
  ns->set_phi((there_is_a_substrate ? phi_eff.vec:phi.vec));

  ns->set_dt(dt_nm1,dt);

  ns->set_velocities(v_nm1.vec, v_n.vec);

  PetscPrintf(mpi.comm(),"NS solver initialization: CFL_NS: %0.2f, rho : %0.2f, mu : %0.3e \n",cfl_NS,rho_l,mu_l);

  // Use a function to set ns parameters to avoid code duplication
  set_ns_parameters();

} // end of "initialize_ns_solver()"

bool my_p4est_stefan_with_fluids_t::navier_stokes_step(){

  // Destroy old pressure at nodes (if it exists) and create vector to hold new solns:
  press_nodes.destroy(); press_nodes.create(p4est_np1, nodes_np1);

  // Create vector to store old dxyz hodge:
  for (unsigned char d=0; d<P4EST_DIM; d++){
    ierr = VecCreateNoGhostFaces(p4est_np1, faces_np1, &dxyz_hodge_old[d], d); CHKERRXX(ierr);
  }

  hodge_tolerance = NS_norm*hodge_percentage_of_max_u;
  PetscPrintf(mpi.comm(),"Hodge tolerance is %e \n",hodge_tolerance);

  int hodge_iteration = 0;
  double convergence_check_on_dxyz_hodge = DBL_MAX;

  face_solver = NULL;
  cell_solver = NULL;

  // Update the parameters: (this is only done to update the cfl potentially)
  // Use a function to set ns parameters to avoid code duplication
  set_ns_parameters();

  // Enter the loop on the hodge variable and solve the NS equations
  while((hodge_iteration<hodge_max_it) && (convergence_check_on_dxyz_hodge>hodge_tolerance)){
    ns->copy_dxyz_hodge(dxyz_hodge_old);
    ns->solve_viscosity(face_solver,(face_solver!=NULL),face_solver_type,pc_face);

    convergence_check_on_dxyz_hodge=
        ns->solve_projection(cell_solver,(cell_solver!=NULL),cell_solver_type,pc_cell,
                             false,NULL,dxyz_hodge_old,uvw_components);

    ierr= PetscPrintf(mpi.comm(),"Hodge iteration : %d, (hodge error)/(NS_max): %0.3e \n",hodge_iteration,convergence_check_on_dxyz_hodge/NS_norm);CHKERRXX(ierr);
    hodge_iteration++;
  }
  ierr = PetscPrintf(mpi.comm(), "Hodge loop exited \n");

  for (unsigned char d=0;d<P4EST_DIM;d++){
    ierr = VecDestroy(dxyz_hodge_old[d]); CHKERRXX(ierr);
  }

  // Delete solvers:
  delete face_solver;
  delete cell_solver;


  // Compute velocity at the nodes
  ns->compute_velocity_at_nodes();

  // ------------------------
  // Slide velocity fields (for our use):
  // ------------------------
  // (a) get rid of old vnm1, now vn becomes the new vnm1
  // (b) no need to destroy vn, bc now we put vnp1 into vn's slot
  v_nm1.destroy();
  foreach_dimension(d){
    ns->get_node_velocities_n(v_nm1.vec[d], d);
    ns->get_node_velocities_np1(v_n.vec[d], d);
  }


  // ------------------------
  // Compute the pressure
  // TO-DO: need to make sure the user defines compute_pressure_ at every required step bc nothing internal handles this
  if(compute_pressure_){
    ns->compute_pressure(); // note: only compute pressure at nodes when we are saving to VTK (or evaluating some errors)
    ns->compute_pressure_at_nodes(&press_nodes.vec);
  }

  // Get the computed values of vorticity
  vorticity.vec = ns->get_vorticity();




  // Check the L2 norm of u to make sure nothing is blowing up
  NS_norm = ns->get_max_L2_norm_u();

  PetscPrintf(mpi.comm(),"\n Max NS velocity norm: \n"
                        " - Computational value: %0.4f  "
                        " - Physical value: %0.3e [m/s]  "
                        " - Physical value: %0.3f [mm/s] \n",NS_norm,NS_norm*vel_nondim_to_dim,NS_norm*vel_nondim_to_dim*1000.);

  // Stop simulation if things are blowing up
  bool did_crash;
  if(NS_norm>NS_max_allowed){
    MPI_Barrier(mpi.comm());
    PetscPrintf(mpi.comm(),"The simulation blew up ! ");
    did_crash = true;
  }
  else{
    did_crash = false;
  }
  return did_crash;

} // end of "navier_stokes_step()"

void my_p4est_stefan_with_fluids_t::setup_and_solve_navier_stokes_problem(){

  // -------------------------------
  // Set the NS timestep:
  // -------------------------------
  if(advection_sl_order ==2){
    ns->set_dt(dt_nm1,dt);
  }
  else{
    ns->set_dt(dt);
  }

  // -------------------------------
  // Update BC and RHS objects for navier-stokes problem:
  // -------------------------------
  // NOTE: we update NS grid first, THEN set new BCs and forces. This is because the update grid interpolation of the hodge variable
  // requires knowledge of the boundary conditions from that same timestep (the previous one, in our case)
  // -------------------------------
  // Setup velocity conditions
  for(unsigned char d=0;d<P4EST_DIM;d++){
    if(interfacial_vel_bc_requires_vint){
      bc_interface_value_velocity[d]->set(ngbd_np1, v_interface.vec[d]);
    }

    bc_velocity[d].setInterfaceType(*bc_interface_type_velocity[d]);
    bc_velocity[d].setInterfaceValue(*bc_interface_value_velocity[d]);
    bc_velocity[d].setWallValues(*bc_wall_value_velocity[d]);
    bc_velocity[d].setWallTypes(*bc_wall_type_velocity[d]);
  }
  // Setup pressure conditions:
  bc_pressure.setInterfaceType(bc_interface_type_pressure);
  bc_pressure.setInterfaceValue(*bc_interface_value_pressure);
  bc_pressure.setWallTypes(*bc_wall_type_pressure);
  bc_pressure.setWallValues(*bc_wall_value_pressure);

  // -------------------------------
  // Set BC's and external forces if relevant
  // (note: these are actually updated in the fxn dedicated to it, aka setup_analytical_ics_and_bcs_for_this_tstep() )
  // -------------------------------
  // Set the boundary conditions:
  ns->set_bc(bc_velocity,&bc_pressure);

  // Set the RHS:
  if(there_is_user_provided_external_force_NS){
    ns->set_external_forces(user_provided_external_forces_NS);
  }

  // -------------------------------
  // Handle the Boussinesq case setup for the RHS, if relevant:
  // ---------------------------
  if(use_boussinesq && (!there_is_user_provided_external_force_NS)){
    switch(problem_dimensionalization_type){
    case NONDIM_BY_FLUID_VELOCITY:{
      ns->boussinesq_approx=true;
      ierr = VecScaleGhost(T_l_n.vec, -1.);
      ns->set_external_forces_using_vector(T_l_n.vec);
      ierr = VecScaleGhost(T_l_n.vec, -1.);
      break;
    }
    case NONDIM_BY_SCALAR_DIFFUSIVITY:{
      ns->boussinesq_approx=true;
      if(!is_dissolution_case){
        ierr = VecScaleGhost(T_l_n.vec, -1.*RaT*Pr);
        ns->set_external_forces_using_vector(T_l_n.vec);
        ierr = VecScaleGhost(T_l_n.vec, -1./(RaT*Pr));
      }
      else{
        // Elyce to-do: 12/15/21 - this is a work in process, havent nailed down nondim def yet
        ierr = VecScaleGhost(T_l_n.vec, -1.*RaC*Sc);
        ns->set_external_forces_using_vector(T_l_n.vec);
        ierr = VecScaleGhost(T_l_n.vec, -1./(RaC*Sc));
      }

      break;
    }
    case DIMENSIONAL:{
      throw std::invalid_argument("AHHHHH!!! this is not fully developed yet. don't use this setup with natural convection \n");

      break;
    }
    default:{
      throw std::runtime_error("setting natural convection -- unrecognized problem dimensionalization formulation \n");
    }
    }
  }


  // -------------------------------
  // Solve the Navier-Stokes problem:
  // -------------------------------
  if(print_checkpoints) PetscPrintf(mpi.comm(),"Beginning Navier-Stokes solution step... \n");


  // Check if we are going to be saving to vtk for the next timestep... if so, we will compute pressure at nodes for saving

  bool did_crash = navier_stokes_step();


  // -------------------------------
  // Clear out the interfacial BC for the next timestep, if needed
  // -------------------------------
  if(interfacial_vel_bc_requires_vint){
    for(unsigned char d=0;d<P4EST_DIM;d++){
      bc_interface_value_velocity[d]->clear();
    }
  }

  if(print_checkpoints) PetscPrintf(mpi.comm(),"Completed Navier-Stokes step \n");

  if(did_crash){
    char crash_tag[10];
    sprintf(crash_tag, "NS");
    save_fields_to_vtk(did_crash, crash_tag);
  }


} // end of "setup_and_solve_navier_stokes_problem()"

// -------------------------------------------------------
// Functions related to LSF advection/grid update:
// -------------------------------------------------------

void prepare_refinement_fields(vec_and_ptr_t& phi, vec_and_ptr_t& vorticity, vec_and_ptr_t& vorticity_refine, vec_and_ptr_dim_t& T_l_dd, my_p4est_node_neighbors_t* ngbd_n, bool refine_by_vorticity_){
  PetscErrorCode ierr;

  // Get relevant arrays:
  if(refine_by_vorticity_){
    vorticity.get_array();
    vorticity_refine.get_array();
  }
  if(refine_by_d2T) {T_l_dd.get_array();}
  phi.get_array();

  // Compute proper refinement fields on layer nodes:
  for(size_t i = 0; i<ngbd_n->get_layer_size(); i++){
    p4est_locidx_t n = ngbd_n->get_layer_node(i);
    if(phi.ptr[n] < 0.){
      if(refine_by_vorticity_)vorticity_refine.ptr[n] = vorticity.ptr[n];
    }
    else{
      if(refine_by_vorticity_) vorticity_refine.ptr[n] = 0.0;
      if(refine_by_d2T){ // Set to 0 in solid subdomain, don't want to refine by T_l_dd in there
        foreach_dimension(d){
          T_l_dd.ptr[d][n]=0.;
        }
      }
    }
  } // end of loop over layer nodes

  // Begin updating the ghost values:
  if(refine_by_vorticity_)ierr = VecGhostUpdateBegin(vorticity_refine.vec,INSERT_VALUES,SCATTER_FORWARD);
  if(refine_by_d2T){
    foreach_dimension(d){
      ierr = VecGhostUpdateBegin(T_l_dd.vec[d],INSERT_VALUES,SCATTER_FORWARD);
    }
  }

  //Compute proper refinement fields on local nodes:
  for(size_t i = 0; i<ngbd_n->get_local_size(); i++){
    p4est_locidx_t n = ngbd_n->get_local_node(i);
    if(phi.ptr[n] < 0.){
      if(refine_by_vorticity_)vorticity_refine.ptr[n] = vorticity.ptr[n];
    }
    else{
      if(refine_by_vorticity_)vorticity_refine.ptr[n] = 0.0;
      if(refine_by_d2T){ // Set to 0 in solid subdomain, don't want to refine by T_l_dd in there
        foreach_dimension(d){
          T_l_dd.ptr[d][n]=0.;
        }
      }
    }
  } // end of loop over local nodes

  // Finish updating the ghost values:
  if(refine_by_vorticity_)ierr = VecGhostUpdateEnd(vorticity_refine.vec,INSERT_VALUES,SCATTER_FORWARD);
  if(refine_by_d2T){
    foreach_dimension(d){
      ierr = VecGhostUpdateEnd(T_l_dd.vec[d],INSERT_VALUES,SCATTER_FORWARD);
    }
  }

  // Restore appropriate arrays:
  if(refine_by_d2T) {T_l_dd.restore_array();}
  if(refine_by_vorticity_){
    vorticity.restore_array();
    vorticity_refine.restore_array();
  }
  phi.restore_array();
}


void perform_reinitialization(int mpi_comm, my_p4est_level_set_t ls, vec_and_ptr_t& phi){
  // Time to reinitialize in a clever way depending on the scenario:

  if(solve_stefan && !force_interfacial_velocity_to_zero){
    // If interface velocity *is* forced to zero, we do not reinitialize -- that way we don't degrade the LSF through unnecessary reinitializations



    // There are some cases where we may not want to reinitialize after every time iteration
    // i.e.) In the coupled case, if fluid velocity is much larger than interfacial velocity, may not need to reinitialize as much
    // (bc the timestepping is much smaller than necessary for the interface growth, and we don't want the reinitialization to govern more of the interface change than the actual physical interface change)
    // For this reason, we have the user option of reinit_every_iter (which is default set to 1)

    if((tstep % reinit_every_iter) == 0){
      ls.reinitialize_2nd_order(phi.vec);
      PetscPrintf(mpi_comm, "reinit every iter =%d, LSF was reinitialized \n", reinit_every_iter);
    }
  }
  else{
    // If only solving Navier-Stokes, or just no interface motion, only need to do this once, not every single timestep
    if(tstep==0) ls.reinitialize_2nd_order(phi.vec);
  }
};


void refine_and_coarsen_grid_and_advect_lsf_if_applicable(my_p4est_semi_lagrangian_t sl, splitting_criteria_cf_and_uniform_band_t sp,
                                                          p4est_t* &p4est_np1, p4est_nodes_t* &nodes_np1, p4est_ghost_t* &ghost_np1,
                                                          p4est_t* &p4est_n, p4est_nodes_t* &nodes_n,
                                                          vec_and_ptr_t &phi, vec_and_ptr_dim_t& v_interface,
                                                          vec_and_ptr_t& phi_substrate,
                                                          vec_and_ptr_dim_t& phi_dd,
                                                          vec_and_ptr_t& vorticity, vec_and_ptr_t& vorticity_refine,
                                                          vec_and_ptr_t& T_l_n,vec_and_ptr_dim_t& T_l_dd,
                                                          my_p4est_node_neighbors_t* ngbd){
  PetscErrorCode ierr;
  int mpi_comm = p4est_np1->mpicomm;
  // ------------------------------------------------------------
  // Define the things needed for the refinement/coarsening tool:
  // ------------------------------------------------------------
  if(!solve_stefan) refine_by_d2T=false; // override settings if there *is* no temperature field

  bool use_block = false;
  bool expand_ghost_layer = true;

  std::vector<compare_option_t> compare_opn;
  std::vector<compare_diagonal_option_t> diag_opn;
  std::vector<double> criteria;
  std::vector<int> custom_lmax;

  PetscInt num_fields = 0;
  bool refine_by_vorticity_ = solve_navier_stokes && vorticity.vec!=NULL;
  // -----------------------
  // Count number of refinement fields and create vectors for necessary fields:
  // ------------------------
  if(refine_by_vorticity_) {
    num_fields+=1;
    vorticity_refine.create(p4est_n, nodes_n);
  }// for vorticity
  if(refine_by_d2T){
    num_fields+=2;
    T_l_dd.create(p4est_n, nodes_n);
    ngbd->second_derivatives_central(T_l_n.vec,T_l_dd.vec);
  } // for second derivatives of temperature

  // Create array of fields we wish to refine by, to pass to the refinement tools
  Vec fields_[num_fields];

  // ------------------------------------------------------------
  // Begin preparing the refine/coarsen criteria:
  // ------------------------------------------------------------
  if(num_fields>0){
    // ------------------------------------------------------------
    // Prepare refinement fields:
    // ------------------------------------------------------------
    prepare_refinement_fields(phi,vorticity,vorticity_refine,T_l_dd,ngbd, refine_by_vorticity_);

    // ------------------------------------------------------------
    // Add our refinement fields to the array:
    // ------------------------------------------------------------
    PetscInt fields_idx = 0;
    if(refine_by_vorticity_)fields_[fields_idx++] = vorticity_refine.vec;
    if(refine_by_d2T){
      fields_[fields_idx++] = T_l_dd.vec[0];
      fields_[fields_idx++] = T_l_dd.vec[1];
    }

    P4EST_ASSERT(fields_idx ==num_fields);

    // ------------------------------------------------------------
    // Add our instructions:
    // ------------------------------------------------------------
    // Coarsening instructions: (for vorticity)
    if(refine_by_vorticity_){
      compare_opn.push_back(LESS_THAN);
      diag_opn.push_back(DIVIDE_BY);
      criteria.push_back(vorticity_threshold*NS_norm/2.);

      // Refining instructions: (for vorticity)
      compare_opn.push_back(GREATER_THAN);
      diag_opn.push_back(DIVIDE_BY);
      criteria.push_back(vorticity_threshold*NS_norm);

      if(lint>0){custom_lmax.push_back(lint);}
      else{custom_lmax.push_back(sp.max_lvl);}
    }
    if(refine_by_d2T){
      double dxyz_smallest[P4EST_DIM];
      dxyz_min(p4est_n,dxyz_smallest);

      double dTheta= fabs(theta_infty - theta_interface)>0 ? fabs(theta_infty - theta_interface): 1.0;
      dTheta/=SQR(min(dxyz_smallest[0],dxyz_smallest[1])); // max d2Theta in liquid subdomain

      // Define variables for the refine/coarsen instructions for d2T fields:
      compare_diagonal_option_t diag_opn_d2T = DIVIDE_BY;
      compare_option_t compare_opn_d2T = SIGN_CHANGE;
      double refine_criteria_d2T = dTheta*gradT_threshold;
      double coarsen_criteria_d2T = dTheta*gradT_threshold*0.1;

      // Coarsening instructions: (for d2T/dx2)
      compare_opn.push_back(compare_opn_d2T);
      diag_opn.push_back(diag_opn_d2T);
      criteria.push_back(coarsen_criteria_d2T); // did 0.1* () for the coarsen if no sign change OR below threshold case

      // Refining instructions: (for d2T/dx2)
      compare_opn.push_back(compare_opn_d2T);
      diag_opn.push_back(diag_opn_d2T);
      criteria.push_back(refine_criteria_d2T);
      if(lint>0){custom_lmax.push_back(lint);}
      else{custom_lmax.push_back(sp.max_lvl);}

      // Coarsening instructions: (for d2T/dy2)
      compare_opn.push_back(compare_opn_d2T);
      diag_opn.push_back(diag_opn_d2T);
      criteria.push_back(coarsen_criteria_d2T);

      // Refining instructions: (for d2T/dy2)
      compare_opn.push_back(compare_opn_d2T);
      diag_opn.push_back(diag_opn_d2T);
      criteria.push_back(refine_criteria_d2T);
      if(lint>0){custom_lmax.push_back(lint);}
      else{custom_lmax.push_back(sp.max_lvl);}
    }
  } // end of "if num_fields!=0"

  // -------------------------------
  // Call grid advection and update:
  // -------------------------------

  if(solve_stefan){
    // Create second derivatives for phi in the case that we are using update_p4est:
    phi_dd.create(p4est_n, nodes_n);
    ngbd->second_derivatives_central(phi.vec, phi_dd.vec);

    // Get inner substrate LSF if needed
    if(example_uses_inner_LSF){
      //      if(start_w_merged_grains){regularize_front(p4est, nodes, ngbd, phi_substrate.vec);}
    }
    // Call advection and refinement
    sl.update_p4est(v_interface.vec, dt,
                    phi.vec, phi_dd.vec, example_uses_inner_LSF ? phi_substrate.vec: NULL,
                    num_fields, use_block, true,
                    uniform_band, uniform_band*(1.5),
                    fields_, NULL,
                    criteria, compare_opn, diag_opn, custom_lmax,
                    expand_ghost_layer);

    if(print_checkpoints) PetscPrintf(mpi_comm,"Grid update completed \n");

    // Destroy 2nd derivatives of LSF now that not needed
    phi_dd.destroy();

  } // case for stefan or coupled
  else {
    // NS only case --> no advection --> do grid update iteration manually:
    splitting_criteria_tag_t sp_NS(sp.min_lvl, sp.max_lvl, sp.lip);

    // Create a new vector which will hold the updated values of the fields -- since we will interpolate with each grid iteration
    Vec fields_new_[num_fields];
    if(num_fields!=0)
    {
      for(unsigned int k = 0;k<num_fields; k++){
        ierr = VecCreateGhostNodes(p4est_np1,nodes_np1,&fields_new_[k]);
        ierr = VecCopyGhost(fields_[k],fields_new_[k]);
      }
    }

    // Create a vector which will hold the updated values of the LSF:
    vec_and_ptr_t phi_new;
    phi_new.create(p4est_n, nodes_n);
    ierr = VecCopyGhost(phi.vec, phi_new.vec);

    bool is_grid_changing = true;
    int no_grid_changes = 0;
    bool last_grid_balance = false;
    while(is_grid_changing){
      if(!last_grid_balance){
        is_grid_changing = sp_NS.refine_and_coarsen(p4est_np1, nodes_np1, phi_new.vec,
                                                    num_fields, use_block, true,
                                                    uniform_band, uniform_band*1.5,
                                                    fields_new_, NULL, criteria,
                                                    compare_opn, diag_opn, custom_lmax);

        if(no_grid_changes>0 && !is_grid_changing){
          last_grid_balance = true; // if the grid isn't changing anymore but it has changed, we need to do one more special interp of fields and balancing of the grid
        }
      }

      if(is_grid_changing || last_grid_balance){
        no_grid_changes++;
        PetscPrintf(mpi_comm,"NS grid changed %d times \n",no_grid_changes);
        if(last_grid_balance){
          p4est_balance(p4est_np1,P4EST_CONNECT_FULL,NULL);
          PetscPrintf(mpi_comm,"Does last grid balance \n");
        }

        my_p4est_partition(p4est_np1,P4EST_FALSE,NULL);
        p4est_ghost_destroy(ghost_np1); ghost_np1 = my_p4est_ghost_new(p4est_np1,P4EST_CONNECT_FULL);
        my_p4est_ghost_expand(p4est_np1,ghost_np1);
        p4est_nodes_destroy(nodes_np1); nodes_np1 = my_p4est_nodes_new(p4est_np1,ghost_np1);

        // Destroy fields_new and create it on the new grid:
        if(num_fields!=0){
          for(unsigned int k = 0; k<num_fields; k++){
            ierr = VecDestroy(fields_new_[k]);
            ierr = VecCreateGhostNodes(p4est_np1,nodes_np1,&fields_new_[k]);
          }
        }
        phi_new.destroy();
        phi_new.create(p4est_np1,nodes_np1);

        // Interpolate fields onto new grid:
        my_p4est_interpolation_nodes_t interp_refine_and_coarsen(ngbd);
        double xyz_interp[P4EST_DIM];
        foreach_node(n,nodes_np1){
          node_xyz_fr_n(n,p4est_np1,nodes_np1,xyz_interp);
          interp_refine_and_coarsen.add_point(n,xyz_interp);
        }
        if(num_fields!=0){
          interp_refine_and_coarsen.set_input(fields_,quadratic_non_oscillatory_continuous_v2,num_fields);
          // Interpolate fields
          interp_refine_and_coarsen.interpolate(fields_new_);
        }
        interp_refine_and_coarsen.set_input(phi.vec,quadratic_non_oscillatory_continuous_v2);
        interp_refine_and_coarsen.interpolate(phi_new.vec);

        if(last_grid_balance){
          last_grid_balance = false;
        }
      } // End of if grid is changing

      // Do last balancing of the grid, and final interp of phi:
      if(no_grid_changes>10) {PetscPrintf(mpi_comm,"NS grid did not converge!\n"); break;}
    } // end of while grid is changing

    // Update the LSF accordingly:
    phi.destroy();
    phi.create(p4est_np1,nodes_np1);
    ierr = VecCopyGhost(phi_new.vec,phi.vec);

    // Destroy the vectors we created for refine and coarsen:
    for(unsigned int k = 0;k<num_fields; k++){
      ierr = VecDestroy(fields_new_[k]);
    }
    phi_new.destroy();
  } // end of if only navier stokes

  // -------------------------------
  // Destroy refinement fields now that they're not in use:
  // -------------------------------
  if(refine_by_vorticity_){
    vorticity_refine.destroy();
  }
  if(refine_by_d2T){
    T_l_dd.destroy();
  }

  // -------------------------------
  // Clear up the memory from the std vectors holding refinement info:
  // -------------------------------
  compare_opn.clear(); diag_opn.clear(); criteria.clear();
  compare_opn.shrink_to_fit(); diag_opn.shrink_to_fit(); criteria.shrink_to_fit();
  custom_lmax.clear(); custom_lmax.shrink_to_fit();
};

void update_the_grid(splitting_criteria_cf_and_uniform_band_t sp,
                     p4est_t* &p4est_np1, p4est_nodes_t* &nodes_np1, my_p4est_node_neighbors_t* &ngbd_np1,
                     p4est_ghost_t* &ghost_np1, my_p4est_hierarchy_t* &hierarchy_np1,
                     p4est_t* &p4est_n, p4est_nodes_t* &nodes_n, my_p4est_node_neighbors_t* &ngbd_n,
                     p4est_ghost_t* &ghost_n, my_p4est_hierarchy_t* &hierarchy_n,
                     my_p4est_brick_t &brick, my_p4est_navier_stokes_t* ns,
                     vec_and_ptr_t &phi, vec_and_ptr_t &phi_nm1, vec_and_ptr_dim_t& v_interface,
                     vec_and_ptr_t& phi_substrate, vec_and_ptr_t &phi_eff,
                     vec_and_ptr_dim_t& phi_dd,
                     vec_and_ptr_t& vorticity, vec_and_ptr_t& vorticity_refine,
                     vec_and_ptr_t& T_l_n,vec_and_ptr_dim_t& T_l_dd){

  int ierr;
  int mpi_comm = p4est_np1->mpicomm;


  // --------------------------------
  // Destroy p4est at n and slide grids:
  // -----------------------------------
  p4est_destroy(p4est_n);
  p4est_ghost_destroy(ghost_n);
  p4est_nodes_destroy(nodes_n);
  delete ngbd_n;
  delete hierarchy_n;

  p4est_n = p4est_np1;
  ghost_n = ghost_np1;
  nodes_n = nodes_np1;

  hierarchy_n = hierarchy_np1;
  ngbd_n = ngbd_np1;

  // -------------------------------
  // Create the new p4est at time np1:
  // -------------------------------
  p4est_np1 = p4est_copy(p4est_n, P4EST_FALSE); // copy the grid but not the data
  ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est_np1,ghost_np1);
  nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

  // Get the new neighbors: // TO-DO : no need to do this here, is there ?
  hierarchy_np1 = new my_p4est_hierarchy_t(p4est_np1, ghost_np1, &brick);
  ngbd_np1 = new my_p4est_node_neighbors_t(hierarchy_np1,nodes_np1);

  // Initialize the neigbors:
  ngbd_np1->init_neighbors();

  // ------------------------------------------------------
  // Nullify the nm1 grid inside the NS solver if relevant:
  // ------------------------------------------------------
  if(solve_navier_stokes && (tstep>1)){
    ns->nullify_p4est_nm1(); // the nm1 grid has just been destroyed, but pointer within NS has not been updated, so it needs to be nullified (p4est_nm1 in NS == p4est in main)
  }

  // -------------------------------
  // Perform the advection/grid update:
  // -------------------------------
  // If solving NS, save the previous LSF to provide to NS solver, to correctly
  // interpolate hodge variable to new grid

  if(solve_navier_stokes){
    // Rochi addition 10/1/22 solved memory leak
    if (phi_nm1.vec!= NULL){
      phi_nm1.destroy();
    }
    phi_nm1.create(p4est_n, nodes_n);
    ierr = VecCopyGhost((example_uses_inner_LSF? phi_eff.vec : phi.vec), phi_nm1.vec); CHKERRXX(ierr); //--> this will need to be provided to NS update_from_tn_to_tnp1_grid_external
    // copy over phi eff if we are using a substrate
    // Note: this is done because the update_p4est destroys the old LSF, but we need to keep it
    // for NS update procedure
    if(print_checkpoints) ierr= PetscPrintf(mpi_comm,"Phi nm1 copy is created ... \n");
  }

  my_p4est_semi_lagrangian_t sl(&p4est_np1, &nodes_np1, &ghost_np1, ngbd_n);

  refine_and_coarsen_grid_and_advect_lsf_if_applicable(sl, sp,
                                                       p4est_np1, nodes_np1, ghost_np1,
                                                       p4est_n, nodes_n,
                                                       phi, v_interface,
                                                       phi_substrate, phi_dd,
                                                       vorticity, vorticity_refine,
                                                       T_l_n, T_l_dd,
                                                       ngbd_n);

  // -------------------------------
  // Update hierarchy and neighbors to match new updated grid:
  // -------------------------------
  hierarchy_np1->update(p4est_np1,ghost_np1);
  ngbd_np1->update(hierarchy_np1,nodes_np1);

  // Initialize the neigbors:
  ngbd_np1->init_neighbors();

};


void interpolate_fields_onto_new_grid(vec_and_ptr_t& T_l_n, vec_and_ptr_t& T_s_n,
                                      //                                      vec_and_ptr_dim_t& v_interface,
                                      vec_and_ptr_dim_t& v_n,
                                      p4est_nodes_t *nodes_np1, p4est_t *p4est_np1,
                                      my_p4est_node_neighbors_t *ngbd_n,interpolation_method interp_method/*,
                                      Vec *all_fields_old=NULL, Vec *all_fields_new=NULL*/){
  // Need neighbors of old grid to create interpolation object
  // Need nodes of new grid to get the points that we must interpolate to

  Vec all_fields_old[num_fields_interp];
  Vec all_fields_new[num_fields_interp];

  my_p4est_interpolation_nodes_t interp_nodes(ngbd_n);
  //  my_p4est_interpolation_nodes_t* interp_nodes = NULL;
  //  interp_nodes = new my_p4est_interpolation_nodes_t(ngbd_old_grid);



  // Set existing vectors as elements of the array of vectors: --------------------------
  unsigned int i = 0;
  if(solve_stefan){
    all_fields_old[i++] = T_l_n.vec; // Now, all_fields_old[0] and T_l both point to same object (where old T_l vec sits)
    if(do_we_solve_for_Ts) all_fields_old[i++] = T_s_n.vec;

  }
  if(solve_navier_stokes){
    foreach_dimension(d){
      all_fields_old[i++] = v_n.vec[d];
    }
  }
  P4EST_ASSERT(i == num_fields_interp);

  // Create the array of vectors to hold the new values: ------------------------------
  PetscErrorCode ierr;
  for(unsigned int j = 0;j<num_fields_interp;j++){
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &all_fields_new[j]);CHKERRXX(ierr);
  }

  // Do interpolation:--------------------------------------------
  interp_nodes.set_input(all_fields_old,interp_method,num_fields_interp);

  // Grab points on the new grid that we want to interpolate to:
  double xyz[P4EST_DIM];
  foreach_node(n, nodes_np1){
    node_xyz_fr_n(n, p4est_np1, nodes_np1,  xyz);
    interp_nodes.add_point(n,xyz);
  }

  interp_nodes.interpolate(all_fields_new);
  interp_nodes.clear();
  // Destroy the old fields no longer in use:------------------------
  for(unsigned int k=0;k<num_fields_interp;k++){
    ierr = VecDestroy(all_fields_old[k]); CHKERRXX(ierr); // Destroy objects where the old vectors were
  }
  // Slide the newly interpolated fields to back to their passed objects
  i = 0;
  if(solve_stefan){
    T_l_n.vec = all_fields_new[i++]; // Now, T_l points to (new T_l vec)
    if(do_we_solve_for_Ts) T_s_n.vec = all_fields_new[i++];

  }
  if(solve_navier_stokes){
    foreach_dimension(d){
      v_n.vec[d] = all_fields_new[i++];
    }
  }

  P4EST_ASSERT(i==num_fields_interp);
} // end of slide_and_interpolate_fields_onto_new_grid


void perform_lsf_advection_grid_update_and_interp_of_fields(mpi_environment_t &mpi, splitting_criteria_cf_and_uniform_band_t* &sp,
                                                            p4est_t* &p4est_np1, p4est_nodes_t* &nodes_np1, p4est_ghost_t* &ghost_np1,
                                                            my_p4est_node_neighbors_t* &ngbd_np1, my_p4est_hierarchy_t* &hierarchy_np1,
                                                            my_p4est_faces_t* &faces_np1, my_p4est_cell_neighbors_t* &ngbd_c_np1,
                                                            p4est_t* &p4est_n, p4est_nodes_t* &nodes_n, p4est_ghost_t* &ghost_n,
                                                            my_p4est_node_neighbors_t* &ngbd_n, my_p4est_hierarchy_t* &hierarchy_n,
                                                            my_p4est_brick_t& brick,
                                                            my_p4est_navier_stokes_t* &ns,
                                                            my_p4est_level_set_t* &ls,
                                                            vec_and_ptr_t& phi, vec_and_ptr_t& phi_nm1,
                                                            vec_and_ptr_t& phi_substrate, vec_and_ptr_t& phi_eff,
                                                            vec_and_ptr_dim_t& phi_dd,
                                                            vec_and_ptr_t& T_l_n, vec_and_ptr_t& T_s_n,
                                                            vec_and_ptr_dim_t& T_l_dd,
                                                            vec_and_ptr_dim_t& v_interface,
                                                            vec_and_ptr_dim_t& v_n, vec_and_ptr_dim_t& v_nm1,
                                                            vec_and_ptr_t& vorticity, vec_and_ptr_t& vorticity_refine,
                                                            double dxyz_smallest[P4EST_DIM], double& dxyz_close_to_interface,
                                                            int load_tstep, int last_tstep,
                                                            interpolation_method interp_bw_grids){
  // ---------------------------------------------------
  // (4) Advance the LSF and (5) Update the grid:
  // ---------------------------------------------------
  /* In Coupled case: advect the LSF and update the grid according to vorticity, d2T/dd2, and phi
       * In Stefan case:  advect the LSF and update the grid according to phi
       * In NS case:      update the grid according to phi (no advection)
      */

  // --------------------------------
  // (4/5a) Compute the timestep
  // (needed for the grid advection, and will be used as timestep for np1 step)
  // --------------------------------
  dt_nm1 = dt; // Slide the timestep

  // Compute stefan timestep:
  char stefan_timestep[1000];
  sprintf(stefan_timestep,"Computed interfacial velocity: \n"
                           " - Computational : %0.3e  "
                           "- Physical : %0.3e [m/s]  "
                           "- Physical : %0.3f  [mm/s] \n",
          v_interface_max_norm,
          v_interface_max_norm*vel_nondim_to_dim,
          v_interface_max_norm*vel_nondim_to_dim*1000.);

  compute_timestep(p4est_np1, nodes_np1,
                   phi, v_interface,
                   ns, dxyz_close_to_interface, dxyz_smallest,
                   load_tstep, last_tstep); // this function modifies the variable dt



  if(tstep!=last_tstep){
    //-------------------------------------------------------------
    // (4/5b) Update the grids so long as this is not the last timestep:
    //-------------------------------------------------------------
    if(print_checkpoints) PetscPrintf(mpi.comm(),"Beginning grid update process ... \n"
                              "Refine by d2T = %s \n",refine_by_d2T? "true": "false");
    update_the_grid(*sp, p4est_np1, nodes_np1, ngbd_np1, ghost_np1, hierarchy_np1,
                    p4est_n, nodes_n, ngbd_n, ghost_n, hierarchy_n,
                    brick, ns,
                    phi, phi_nm1, v_interface, phi_substrate, phi_eff, phi_dd,
                    vorticity, vorticity_refine, T_l_n, T_l_dd);

    // -------------------------------
    // (4/5c) Reinitialize the LSF on the new grid (if it has been advected):
    // -------------------------------
    if(print_checkpoints) PetscPrintf(mpi.comm(),"Reinitializing LSF... \n");
    ls->update(ngbd_np1);
    perform_reinitialization(mpi.comm(), *ls, phi);

    // Regularize the front (if we are doing that)
    if(use_regularize_front){
      PetscPrintf(mpi.comm(), "Calling regularlize front: \n");
      regularize_front(p4est_np1, nodes_np1, ngbd_np1, phi);
    }

    // Check collapse on the substrate (if we are doing that)
    if(example_uses_inner_LSF && use_collapse_onto_substrate){
      PetscPrintf(mpi.comm(), "Checking collapse \n ");
      //          if(start_w_merged_grains){regularize_front(p4est_np1, nodes_np1, ngbd_np1, phi_substrate.vec);}
      check_collapse_on_substrate(p4est_np1,nodes_np1,ngbd_np1, phi, phi_substrate);
    }

    //------------------------------------------------------
    // (4/5d) Destroy substrate LSF and phi_eff (if used) and re-create for upcoming timestep:
    //------------------------------------------------------
    if(example_uses_inner_LSF){
      phi_substrate.destroy();
      phi_eff.destroy();
      create_and_compute_phi_sub_and_phi_eff(p4est_np1, nodes_np1,
                                             ls,
                                             phi, phi_substrate, phi_eff);
    }

    // ---------------------------------------------------
    // (4/5e) Interpolate Values onto New Grid:
    // ---------------------------------------------------

    if(print_checkpoints) PetscPrintf(mpi.comm(),"Interpolating fields to new grid ... \n");

    interpolate_fields_onto_new_grid(T_l_n, T_s_n,
                                     /*v_interface,*/ v_n,
                                     nodes_np1, p4est_np1, ngbd_n, interp_bw_grids);
    if(solve_navier_stokes){
      ns->update_from_tn_to_tnp1_grid_external((example_uses_inner_LSF? phi_eff.vec : phi.vec), phi_nm1.vec,
                                               v_n.vec, v_nm1.vec,
                                               p4est_np1, nodes_np1, ghost_np1,
                                               ngbd_np1,
                                               faces_np1,ngbd_c_np1,
                                               hierarchy_np1);
    }
    if(print_checkpoints) PetscPrintf(mpi.comm(),"Done. \n");
  } // end of "if tstep !=last tstep"


} // end of "perform_lsf_advection_grid_update_and_interp_of_fields"


// -------------------------------------------------------
// Functions related to LSF regularization:
// -------------------------------------------------------





// -------------------------------------------------------
// Functions related to VTK saving:
// -------------------------------------------------------
void my_p4est_stefan_with_fluids_t::save_fields_to_vtk(bool is_crash, char crash_type[]){

  char output[1000];

  const char* out_dir = getenv("OUT_DIR_VTK");
  if(!out_dir){
    throw std::invalid_argument("You need to set the output directory for VTK: OUT_DIR_VTK");
  }
  sprintf(output, "%s/grid_lmin%d_lint%d_lmax%d", out_dir, lmin, lint, lmax);
  // Create outdir if it does not exist:
  if(!file_exists(output)){
    create_directory(output, mpi.rank(), mpi.comm());
  }
  if(!is_folder(output)){
    if(!create_directory(output, mpi.rank(), mpi.comm()))
    {
      char error_msg[1024];
      sprintf(error_msg, "saving geometry information: the path %s is invalid and the directory could not be created", output);
      throw std::invalid_argument(error_msg);
    }
  }
  // Now save to vtk:


  PetscPrintf(mpi.comm(),"Saving to vtk, outidx = %d ...\n",out_idx);

  //    if(example_uses_inner_LSF){
  //      if(start_w_merged_grains){regularize_front(p4est, nodes, ngbd, phi_substrate.vec);}
  //    }

  if(is_crash){
    sprintf(output,"%s/snapshot_lmin_%d_lmax_%d_%s_CRASH", output, lmin, lmax, crash_type);

  }
  else{
    sprintf(output,"%s/snapshot_lmin_%d_lmax_%d_outidx_%d", output, lmin, lmax, out_idx);
  }

  // Calculate curvature:
  vec_and_ptr_t kappa;
  vec_and_ptr_dim_t normal;

  kappa.create(p4est_np1, nodes_np1);
  normal.create(p4est_np1, nodes_np1);

  VecScaleGhost(phi.vec,-1.0);
  compute_normals(*ngbd_np1, phi.vec,normal.vec);
  compute_mean_curvature(*ngbd_np1, normal.vec,kappa.vec);

  VecScaleGhost(phi.vec,-1.0);

  // Save data:
  std::vector<Vec_for_vtk_export_t> point_fields;
  point_fields.push_back(Vec_for_vtk_export_t(phi.vec, "phi"));
  point_fields.push_back(Vec_for_vtk_export_t(kappa.vec, "kappa"));

  //phi substrate and phi eff
  if(there_is_a_substrate){
    point_fields.push_back(Vec_for_vtk_export_t(phi_eff.vec, "phi_eff"));
    point_fields.push_back(Vec_for_vtk_export_t(phi_substrate.vec, "phi_sub"));
  }
  // stefan related fields
  if(solve_stefan){
    point_fields.push_back(Vec_for_vtk_export_t(T_l_n.vec, "Tl"));
    if(do_we_solve_for_Ts){
      point_fields.push_back(Vec_for_vtk_export_t(T_s_n.vec, "Ts"));
    }
    point_fields.push_back(Vec_for_vtk_export_t(v_interface.vec[0], "v_interface_x"));
    point_fields.push_back(Vec_for_vtk_export_t(v_interface.vec[1], "v_interface_y"));
  }

  if(solve_navier_stokes){
    point_fields.push_back(Vec_for_vtk_export_t(v_n.vec[0], "u"));
    point_fields.push_back(Vec_for_vtk_export_t(v_n.vec[1], "v"));
    point_fields.push_back(Vec_for_vtk_export_t(vorticity.vec, "vorticity"));
    point_fields.push_back(Vec_for_vtk_export_t(press_nodes.vec, "pressure"));
  }
  if(track_evolving_geometries && !is_crash){
    point_fields.push_back(Vec_for_vtk_export_t(island_numbers.vec, "island_no"));
  }

  std::vector<Vec_for_vtk_export_t> cell_fields = {};

  my_p4est_vtk_write_all_lists(p4est_np1, nodes_np1, ghost_np1,
                               P4EST_TRUE,P4EST_TRUE, output,
                               point_fields, cell_fields);


  point_fields.clear();
  cell_fields.clear();

  kappa.destroy();
  normal.destroy();

  if(print_checkpoints) PetscPrintf(mpi.comm(),"Finishes saving to VTK \n");

};







// -------------------------------------------------------
// Functions related to save state/load state:
// -------------------------------------------------------




// -------------------------------------------------------
// Interfacial boundary condition for temperature:
// -------------------------------------------------------
double my_p4est_stefan_with_fluids_t::interfacial_bc_temp_t::Gibbs_Thomson(DIM(double x, double y, double z))const {
  switch(owner->problem_dimensionalization_type){
    // Note slight difference in condition bw diff nondim types -- T0 vs Tinf
    case NONDIM_BY_FLUID_VELOCITY:{
      return (owner->theta_interface - (owner->sigma/owner->l_char)*((*kappa_interp)(x,y))*(owner->theta_interface + owner->T0/owner->deltaT));
    }
    case NONDIM_BY_SCALAR_DIFFUSIVITY:{
      return (owner->theta_interface - (owner->sigma/owner->l_char)*((*kappa_interp)(x,y))*(owner->theta_interface + owner->Tinfty/owner->deltaT));
    }
    case DIMENSIONAL:{
      return (owner->Tinterface*(1 - owner->sigma*((*kappa_interp)(x,y))));
    }
    default:{
      throw std::runtime_error("Gibbs_thomson: unrecognized problem dimensionalization type \n");
    }
  }
}

// -------------------------------------------------------
// Interfacial boundary condition(s) for fluid velocity:
// -------------------------------------------------------
double my_p4est_stefan_with_fluids_t::interfacial_bc_fluid_velocity_t::Conservation_of_Mass(DIM(double x, double y, double z)) const{
  return (*v_interface_interp)(x,y)*(1. - (owner->rho_s/owner->rho_l));
}

double my_p4est_stefan_with_fluids_t::interfacial_bc_fluid_velocity_t::Strict_No_Slip(DIM(double x, double y, double z)) const{
  return (*v_interface_interp)(x,y);
}





