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





