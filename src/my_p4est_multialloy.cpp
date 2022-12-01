#ifdef P4_TO_P8
#include "my_p8est_multialloy.h"
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_save_load.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_macros.h>
#include <src/my_p8est_utils.h>
#else
#include "my_p4est_multialloy.h"
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_save_load.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_stefan_with_fluids.h>
#include <src/my_p4est_navier_stokes.h>
#include <src/my_p4est_faces.h>
#include <src/my_p4est_macros.h>
#include <src/my_p4est_utils.h>
#endif

#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_my_p4est_multialloy_one_step;
extern PetscLogEvent log_my_p4est_multialloy_compute_dt;
extern PetscLogEvent log_my_p4est_multialloy_compute_geometric_properties;
extern PetscLogEvent log_my_p4est_multialloy_compute_velocity;
extern PetscLogEvent log_my_p4est_multialloy_compute_solid;
extern PetscLogEvent log_my_p4est_multialloy_update_grid;
extern PetscLogEvent log_my_p4est_multialloy_update_grid_solid;
extern PetscLogEvent log_my_p4est_multialloy_save_vtk;
extern PetscLogEvent log_my_p4est_multialloy_update_grid_transfer_data;
extern PetscLogEvent log_my_p4est_multialloy_update_grid_regularize_front;
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif

my_p4est_node_neighbors_t *my_p4est_multialloy_t::v_ngbd;
double **my_p4est_multialloy_t::v_c_p, **my_p4est_multialloy_t::v_c0_d_p, **my_p4est_multialloy_t::v_c0_dd_p, **my_p4est_multialloy_t::v_normal_p;
double my_p4est_multialloy_t::v_factor;
int my_p4est_multialloy_t::v_num_comps;
double (*my_p4est_multialloy_t::v_part_coeff)(int, double *);

my_p4est_multialloy_t::my_p4est_multialloy_t(int num_comps, int time_order)
{
  num_comps_       = num_comps;
  num_time_layers_ = time_order+1;

  ts_.resize(num_time_layers_);
  tl_.resize(num_time_layers_);
  cl_.resize(num_time_layers_, vec_and_ptr_array_t(num_comps_));

  psi_cl_.resize(num_comps_);

  front_velo_.resize(num_time_layers_);
  front_velo_norm_.resize(num_time_layers_);

  dt_.resize(num_time_layers_);

  solid_cl_.resize(num_comps_);
  solid_part_coeff_.resize(num_comps_);

  solute_diff_.assign(num_comps_, 1.e-5);

  latent_heat_     = 2350;    // J.cm-3
  density_l_       = 8.88e-3; // kg.cm-3
  density_s_       = 8.88e-3; // kg.cm-3
  heat_capacity_l_ = 0.46e3;  // J.kg-1.K-1
  heat_capacity_s_ = 0.46e3;  // J.kg-1.K-1
  thermal_cond_l_  = 6.07e-1; // W.cm-1.K-1
  thermal_cond_s_  = 6.07e-1; // W.cm-1.K-1
  liquidus_slope_  = NULL;
  liquidus_value_  = NULL;
  part_coeff_      = NULL;
  vol_heat_gen_    = &zero_cf;

  num_seeds_ = 1;
  eps_c_.assign(1, &zero_cf);
  eps_v_.assign(1, &zero_cf);


  contr_bc_type_temp_ = NEUMANN;
  contr_bc_type_conc_ = NEUMANN;

  contr_bc_value_temp_ = NULL;
  contr_bc_value_conc_.assign(num_comps_, NULL);

//  wall_bc_type_temp_ = NEUMANN;
//  wall_bc_type_conc_ = NEUMANN;

  wall_bc_type_temp_ = NULL;
  wall_bc_type_conc_ = NULL; // changed to NULL bc we changed the wall bc types to be WallBCDIM, which is a class (instead of just a BoundaryConditionType)

  wall_bc_value_temp_ = NULL;
  wall_bc_value_conc_.assign(num_comps_, NULL);

  scaling_ = 1.;

  max_iterations_ = 50;

  bc_tolerance_   = 1.e-5;
  cfl_number_     = 0.5;

  time_   = 0;
  dt_max_ = DBL_MAX;
  dt_min_ = DBL_MIN;

  use_superconvergent_robin_ = false;
  use_points_on_interface_   = true;
  update_c0_robin_           = 0;
  enforce_planar_front_      = false;

  interpolation_between_grids_ = quadratic_non_oscillatory_continuous_v2;
  interpolation_between_grids_ = quadratic_non_oscillatory_continuous_v2;

  dendrite_cut_off_fraction_ = .75;
  dendrite_min_length_       = .05;

  proximity_smoothing_ = 1.0;

  connectivity_ = NULL;
  p4est_ = NULL;
  ghost_ = NULL;
  nodes_ = NULL;
  hierarchy_ = NULL;
  ngbd_ = NULL;

  p4est_nm1 = NULL;
  ghost_nm1 = NULL;
  nodes_nm1 = NULL;
  hierarchy_nm1 = NULL;
  ngbd_nm1 = NULL;

}

my_p4est_multialloy_t::~my_p4est_multialloy_t()
{
  //--------------------------------------------------
  // Geometry
  //--------------------------------------------------
  contr_phi_.destroy();
  front_phi_.destroy();
  front_curvature_.destroy();

  contr_phi_dd_.destroy();
  front_phi_dd_.destroy();
  front_normal_.destroy();

  //--------------------------------------------------
  // Physical fields
  //--------------------------------------------------
  for (int i = 0; i < num_time_layers_; ++i)
  {
    tl_[i].destroy();
    ts_[i].destroy();
    cl_[i].destroy();

    front_velo_     [i].destroy();
    front_velo_norm_[i].destroy();
  }

  cl0_grad_.destroy();

  psi_tl_.destroy();
  psi_ts_.destroy();
  psi_cl_.destroy();

  //--------------------------------------------------
  // Geometry on the auxiliary grid
  //--------------------------------------------------
  solid_front_phi_.destroy();
  solid_front_phi_nm1_.destroy();
  solid_front_curvature_.destroy();
  solid_front_velo_norm_.destroy();
  solid_tf_.destroy();
  solid_cl_.destroy();
  solid_part_coeff_.destroy();
  solid_seed_.destroy();
  solid_time_.destroy();
  solid_smoothed_nodes_.destroy();

  seed_map_.destroy();

  dendrite_number_.destroy();
  dendrite_tip_.destroy();
  bc_error_.destroy();
  smoothed_nodes_.destroy();
  front_phi_unsmooth_.destroy();

  /* destroy the p4est and its connectivity structure */
  delete ngbd_;
  delete hierarchy_;
  p4est_nodes_destroy(nodes_);
  p4est_ghost_destroy(ghost_);
  p4est_destroy      (p4est_);

  delete solid_ngbd_;
  delete solid_hierarchy_;
  p4est_nodes_destroy(solid_nodes_);
  p4est_ghost_destroy(solid_ghost_);
  p4est_destroy      (solid_p4est_);

  my_p4est_brick_destroy(connectivity_, &brick_);
  if(solve_with_fluids){
    if(stefan_w_fluids_solver!=nullptr) delete stefan_w_fluids_solver;

    foreach_dimension(d){
      // Interface:
      bc_interface_val_fluid_vel[d] = NULL;
      bc_interface_type_fluid_vel[d] = NULL;

      // Wall:
      bc_wall_value_velocity[d] = NULL;
      bc_wall_type_velocity[d] = NULL;
    }
    bc_interface_type_fluid_press=NOINTERFACE;
    bc_interface_val_fluid_press=NULL;
  }
  delete sp_crit_;
}
void my_p4est_multialloy_t::set_front(Vec phi)
{
  VecCopyGhost(phi, front_phi_.vec);
  compute_geometric_properties_front();

  my_p4est_interpolation_nodes_t interp(ngbd_);

  double xyz[P4EST_DIM];

  foreach_node(n, solid_nodes_)
  {
    node_xyz_fr_n(n, solid_p4est_, solid_nodes_, xyz);
    interp.add_point(n, xyz);
  }

  interp.set_input(front_phi_.vec, interpolation_between_grids_);
  interp.interpolate(solid_front_phi_.vec);

  VecCopyGhost(solid_front_phi_.vec, solid_front_phi_nm1_.vec);

  interp.set_input(front_curvature_.vec, interpolation_between_grids_);
  interp.interpolate(solid_front_curvature_.vec);
}

void my_p4est_multialloy_t::set_container(Vec phi)
{
  VecCopyGhost(phi, contr_phi_.vec);
  compute_geometric_properties_contr();

  my_p4est_interpolation_nodes_t interp(ngbd_);

  double xyz[P4EST_DIM];

  foreach_node(n, solid_nodes_)
  {
    node_xyz_fr_n(n, solid_p4est_, solid_nodes_, xyz);
    interp.add_point(n, xyz);
  }

  interp.set_input(front_phi_.vec, interpolation_between_grids_);
  interp.interpolate(solid_front_phi_.vec);

  VecCopyGhost(solid_front_phi_.vec, solid_front_phi_nm1_.vec);

  interp.set_input(front_curvature_.vec, interpolation_between_grids_);
  interp.interpolate(solid_front_curvature_.vec);
}

void my_p4est_multialloy_t::initialize(MPI_Comm mpi_comm, double xyz_min[], double xyz_max[], int nxyz[], int periodicity[], CF_2 &level_set, int lmin, int lmax, double lip, double band, bool solve_w_fluids)
{
  // Check if we solve with fluids:
  solve_with_fluids = solve_w_fluids;

  /* create main p4est grid */
  connectivity_ = my_p4est_brick_new(nxyz, xyz_min, xyz_max, &brick_, periodicity);
  p4est_        = my_p4est_new(mpi_comm, connectivity_, 0, NULL, NULL);

  sp_crit_ = new splitting_criteria_cf_t(lmin, lmax, &level_set, lip, band);

  p4est_->user_pointer = (void*)(sp_crit_);
  my_p4est_refine(p4est_, P4EST_TRUE, refine_levelset_cf, NULL);
  my_p4est_partition(p4est_, P4EST_FALSE, NULL);

  ghost_ = my_p4est_ghost_new(p4est_, P4EST_CONNECT_FULL);
  printf("should have expanded the ghost layer ? vvv solve_w_fluids = %d \n", solve_with_fluids);
  if(solve_with_fluids) {my_p4est_ghost_expand(p4est_, ghost_); printf("EXPANDS THE GHOST LAYER !!! \n");}

  printf("should have expanded the ghost layer ? ^^^ solve_w_fluids = %d \n", solve_with_fluids);
  // TO-DO MULTICOMP: eventually add extra expansions for ghost layer for CFL larger than 2
  nodes_ = my_p4est_nodes_new(p4est_, ghost_);

  hierarchy_ = new my_p4est_hierarchy_t(p4est_, ghost_, &brick_);
  ngbd_ = new my_p4est_node_neighbors_t(hierarchy_, nodes_);
  ngbd_->init_neighbors();

  /* create auxiliary p4est grid for keeping values in solid */
  solid_p4est_ = p4est_copy(p4est_, P4EST_FALSE);
  solid_ghost_ = my_p4est_ghost_new(solid_p4est_, P4EST_CONNECT_FULL);
  solid_nodes_ = my_p4est_nodes_new(solid_p4est_, solid_ghost_);

  solid_hierarchy_ = new my_p4est_hierarchy_t(solid_p4est_, solid_ghost_, &brick_);
  solid_ngbd_      = new my_p4est_node_neighbors_t(solid_hierarchy_, solid_nodes_);
  solid_ngbd_->init_neighbors();

  /* determine the smallest cell size */
  get_dxyz_min(p4est_, dxyz_, &dxyz_min_, &diag_);
  dxyz_max_ = dxyz_min_;

  dxyz_close_interface_ = 1.2*dxyz_max_;

  // front_phi_ and front_phi_dd_ are templates for all other vectors

  // allocate memory for physical fields
  //--------------------------------------------------
  // Geometry
  //--------------------------------------------------
  front_phi_.create(p4est_, nodes_);
  front_phi_dd_.create(p4est_, nodes_);
  front_curvature_.create(front_phi_.vec);
  front_normal_.create(front_phi_dd_.vec);

  contr_phi_.create(p4est_, nodes_);
  contr_phi_dd_.create(p4est_, nodes_);

  //--------------------------------------------------
  // Physical fields
  //--------------------------------------------------
  /* temperature */
  for (int i = 0; i < num_time_layers_; ++i)
  {
    tl_[i].create(front_phi_.vec);
    ts_[i].create(front_phi_.vec);
    cl_[i].create(front_phi_.vec);
    front_velo_     [i].create(front_phi_dd_.vec);
    front_velo_norm_[i].create(front_phi_.vec);
  }

  cl0_grad_.create(front_phi_dd_.vec);

  seed_map_.create(front_phi_.vec);

  dendrite_number_.create(front_phi_.vec);
  dendrite_tip_.create(front_phi_.vec);
  bc_error_.create(front_phi_.vec);
  smoothed_nodes_.create(front_phi_.vec);
  front_phi_unsmooth_.create(front_phi_.vec);

  psi_tl_.create(front_phi_.vec);
  psi_ts_.create(front_phi_.vec);
  psi_cl_.create(front_phi_.vec);

  VecSetGhost(psi_tl_.vec, 0.);
  VecSetGhost(psi_ts_.vec, 0.);
  for (int i = 0; i < num_comps_; ++i)
  {
    VecSetGhost(psi_cl_.vec[i], 0.);
  }

  //--------------------------------------------------
  // Geometry on the auxiliary grid
  //--------------------------------------------------
  solid_front_phi_.create(solid_p4est_, solid_nodes_);
  solid_front_phi_nm1_.create(solid_front_phi_.vec);
  solid_front_curvature_.create(solid_front_phi_.vec);
  solid_front_velo_norm_.create(solid_front_phi_.vec);

  solid_time_.create(solid_front_phi_.vec);
  solid_tf_.create(solid_front_phi_.vec);
  solid_cl_.create(solid_front_phi_.vec);
  solid_part_coeff_.create(solid_front_phi_.vec);
  solid_seed_.create(solid_front_phi_.vec);
  solid_smoothed_nodes_.create(solid_front_phi_.vec);

  VecSetGhost(solid_time_.vec, 0.);

}


void my_p4est_multialloy_t::initialize_for_fluids(my_p4est_stefan_with_fluids_t* stefan_w_fluids_solver_){

  iteration_w_fluids=0;
  // NOTE: calling this fxn assumes that initialize for multialloy
  // has already been performed
  //std:: cout << "mpi :: " << mpi_ <<"\n";
  if(mpi_ == NULL){
    throw std::runtime_error("You must set the mpi environment via multialloy:set_mpi_env() in order to create the stefan w fluids solver \n");
  }

  //stefan_w_fluids_solver = new my_p4est_stefan_with_fluids_t(mpi_);
  stefan_w_fluids_solver = stefan_w_fluids_solver_;

  // Set up the initial nm1 grids that we will need:
  p4est_nm1 = p4est_copy(p4est_, P4EST_FALSE); // copy the grid but not the data
  p4est_nm1->user_pointer = sp_crit_; // CHECK
  my_p4est_partition(p4est_nm1,P4EST_FALSE,NULL);

  ghost_nm1 = my_p4est_ghost_new(p4est_nm1, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est_nm1, ghost_nm1);

  // TO-DO MULTICOMP: eventually add extra expansions for ghost layer for CFL larger than 2
  nodes_nm1 = my_p4est_nodes_new(p4est_nm1, ghost_nm1);

  // Get the new neighbors:
  hierarchy_nm1 = new my_p4est_hierarchy_t(p4est_nm1, ghost_nm1, &brick_);
  ngbd_nm1 = new my_p4est_node_neighbors_t(hierarchy_nm1,nodes_nm1);

  // Initialize the neigbors:
  ngbd_nm1->init_neighbors();
  v_n.create(p4est_, nodes_);
  v_nm1.create(p4est_nm1, nodes_nm1);

  foreach_dimension(d){
    sample_cf_on_nodes(p4est_, nodes_, *initial_NS_velocity_n[d],v_n.vec[d]);
    sample_cf_on_nodes(p4est_nm1, nodes_nm1, *initial_NS_velocity_nm1[d],v_nm1.vec[d]);
  }

  if(0){
    // -------------------------------
    // TEMPORARY: save fields before backtrace
    // -------------------------------
    std::vector<Vec_for_vtk_export_t> point_fields;
    std::vector<Vec_for_vtk_export_t> cell_fields = {};

    point_fields.push_back(Vec_for_vtk_export_t(v_n.vec[0], "vx_n"));
    point_fields.push_back(Vec_for_vtk_export_t(v_n.vec[1], "vy_n"));

    const char* out_dir = getenv("OUT_DIR");
    if(!out_dir){
      throw std::invalid_argument("You need to set the output directory for VTK: OUT_DIR_VTK");
    }

    char filename[1000];
    sprintf(filename, "%s/snapshot_after_init_for_fluids_%d", out_dir, iteration_w_fluids);
    my_p4est_vtk_write_all_lists(p4est_, nodes_, ngbd_->get_ghost(), P4EST_TRUE, P4EST_TRUE, filename, point_fields, cell_fields);
    point_fields.clear();


    // nm1 fields now:
    point_fields.push_back(Vec_for_vtk_export_t(v_nm1.vec[0], "vx_nm1"));
    point_fields.push_back(Vec_for_vtk_export_t(v_nm1.vec[1], "vy_nm1"));

    char filename2[1000];
    sprintf(filename2, "%s/snapshot_after_init_for_fluids_nm1_%d", out_dir, iteration_w_fluids);
    my_p4est_vtk_write_all_lists(p4est_nm1, nodes_nm1, ngbd_nm1->get_ghost(), P4EST_TRUE, P4EST_TRUE, filename2, point_fields, cell_fields);
    point_fields.clear();
  }

  stefan_w_fluids_solver->set_phi(front_phi_);
  stefan_w_fluids_solver->set_v_n(v_n);
  stefan_w_fluids_solver->set_v_nm1(v_nm1);

  stefan_w_fluids_solver->set_use_boussinesq(false);
  // TO-DO: turn on boussinesq eventually
  stefan_w_fluids_solver->set_print_checkpoints(false);
  // ----------------------------------------------
  // Next will need to initialize the NS solver:
  // ----------------------------------------------
  // (1) Pass along the boundary conditions associated w the fluid to stefan_w_fluids:
  // -------------

  // TO-DO MULTICOMP: flesh this out
  // Interface velocity:

  // Rochi addition :: to define type of interface velocity
  /*BoundaryConditionType* bc_interface_type_velocity_[P4EST_DIM];
  std::cout<<"hello world 6\n";
  foreach_dimension(dim)
  {
    std::cout<<"hello world 6_1\n";
    *bc_interface_type_velocity_[dim]=DIRICHLET;
  }*/
  //std::cout<<"hello world 7\n";

  // Velocity
  stefan_w_fluids_solver->set_bc_interface_value_velocity(bc_interface_val_fluid_vel);
  stefan_w_fluids_solver->set_bc_interface_type_velocity(bc_interface_type_fluid_vel);

  stefan_w_fluids_solver->set_bc_wall_type_velocity(bc_wall_type_velocity);
  stefan_w_fluids_solver->set_bc_wall_value_velocity(bc_wall_value_velocity);
  printf("[MULTI] bc_wall_value_velocity = %p \n", bc_wall_value_velocity[0]);

  // Pressure
  stefan_w_fluids_solver->set_bc_wall_type_pressure(bc_wall_type_pressure);
  stefan_w_fluids_solver->set_bc_wall_value_pressure(bc_wall_value_pressure);

  stefan_w_fluids_solver->set_bc_interface_type_pressure(bc_interface_type_fluid_press);
  stefan_w_fluids_solver->set_bc_interface_value_pressure(bc_interface_val_fluid_press);

  // -------------
  // (2) Pass along all the necessary grid variables:
  // -------------
  stefan_w_fluids_solver->set_brick(brick_);

  stefan_w_fluids_solver->set_hierarchy_np1(hierarchy_);

  stefan_w_fluids_solver->set_p4est_np1(p4est_);

  stefan_w_fluids_solver->set_nodes_np1(nodes_);
  stefan_w_fluids_solver->set_ghost_np1(ghost_);
  stefan_w_fluids_solver->set_ngbd_np1(ngbd_);
  stefan_w_fluids_solver->set_hierarchy_n(hierarchy_nm1);
  stefan_w_fluids_solver->set_p4est_n(p4est_nm1);
  stefan_w_fluids_solver->set_nodes_n(nodes_nm1);
  stefan_w_fluids_solver->set_ghost_n(ghost_nm1);
  stefan_w_fluids_solver->set_ngbd_n(ngbd_nm1);



  l_char=1.0;
  PetscPrintf(p4est_->mpicomm, "WARNING: RED ALERT: LCHAR MANUALLY SET INSIDE INITIALIZE_FOR_FLUIDS \n");
  double thermal_diff_l = thermal_cond_l_/(density_l_*heat_capacity_l_);

  //printf("thermal cond l = %0.2e, density_l = %0.2e, "
    //     "heat_capacity_l = %0.2e, thermal diff = %0.2e, l_char = %0.2e \n",
      //   thermal_cond_l_, density_l_, heat_capacity_l_, thermal_diff_l, l_char);

  stefan_w_fluids_solver->set_vel_nondim_to_dim(1.0/*thermal_diff_l/SQR(l_char)*/);
  PetscPrintf(p4est_->mpicomm, "RED ALERT: ns max allowed is manually hard coded for now \n");
  stefan_w_fluids_solver->set_NS_max_allowed(1000.0);

  // NOTE -- If you want to visualize dimensional velocities, pressure, and vorticity, you will need to do the appropriate
  // scaling by hand before outputting to vtk

  // -------------
  // (3) pass along the level set function:
  // -------------
  stefan_w_fluids_solver->set_phi(front_phi_);

  // -------------
  // (4) pass along timestep
  // -------------
  PetscPrintf(p4est_->mpicomm, "RED ALERT: dt's passed in here are likely blank, restructure later, for now we pass in one_Step w fluids \n");
  stefan_w_fluids_solver->set_dt(dt_[0]);
  stefan_w_fluids_solver->set_dt_nm1(dt_[1]);

  // -------------
  // (5) pass along nondim type and relevant nondim parameters:
  // -------------
  // TO-DO MULTICOMP: will make this more user friendly later, but this serves as a first pass
  stefan_w_fluids_solver->set_problem_dimensionalization_type(NONDIM_BY_SCALAR_DIFFUSIVITY);

  // irrelevant since we don't know Re, we nondim by diffusivity stefan_w_fluids_solver->set_Re(1.);
  // ALERT :: Pr hard coded and set here
  Pr=23.1;
  PetscPrintf(p4est_->mpicomm, "ALERT ALERT: PRANDTL NUMBER IS HARD CODED AND SET TO 23.1. THIS IS A SHORT TERM FIX AND MUST BE UPDATED \n");
  stefan_w_fluids_solver->set_Pr(Pr);

  stefan_w_fluids_solver->set_NS_advection_order(2);

  stefan_w_fluids_solver->set_cfl_NS(2.); // TO-DO MULTICOMP: THIS SHOULD NOT BE HARD-CODED !!!

  stefan_w_fluids_solver->set_uniform_band(sp_crit_->band);


  // TO-DO MULTICOMP
  // Before calling initialize, we will need to provide:
  // - grid variables: hierarchy_np1, p4est_np1, ghost_np1, brick, ngbd_np1, ngbd_n
  // - level set function (with or without substrate, we will assume no substrate)
  // - dt and dtnm1
  // * This will call "set_ns_parameters"
  // TO-DO MULTICOMP
  // Parameters we need to actually provide:
  // rho, mu, SL order, uband, vort thresh(ignored), cfl_NS
  // -------------
  // Initialize the ns solver:
  // -------------
  stefan_w_fluids_solver->initialize_ns_solver(true);

  // -------------
  // Get back out the cell neigbors and faces:
  // -------------
  // Then we should grab back out the ngbd_c and faces so that multialloy has access to those pointers
  // since the actual objects get created by stefan_w_fluids and navier_stokes
  ngbd_c_ = stefan_w_fluids_solver->get_ngbd_c_np1();
  faces_ = stefan_w_fluids_solver->get_faces_np1();

}


void my_p4est_multialloy_t::compute_geometric_properties_front()
{
  ierr = PetscLogEventBegin(log_my_p4est_multialloy_compute_geometric_properties, 0, 0, 0, 0); CHKERRXX(ierr);

  ngbd_->second_derivatives_central(front_phi_.vec, front_phi_dd_.vec);
  compute_normals_and_mean_curvature(*ngbd_, front_phi_.vec, front_normal_.vec, front_curvature_.vec);

  // flatten curvature values
  my_p4est_level_set_t ls(ngbd_);
  ls.set_interpolation_on_interface(quadratic_non_oscillatory_continuous_v2);
  ls.extend_from_interface_to_whole_domain_TVD_in_place(front_phi_.vec, front_curvature_.vec, front_phi_.vec, 20);

//  front_curvature_.get_array();

//  double kappa_max = 1./dxyz_min_;

//  foreach_node(n, nodes_)
//  {
//    if      (front_curvature_.ptr[n] > kappa_max) front_curvature_.ptr[n] = kappa_max;
//    else if (front_curvature_.ptr[n] <-kappa_max) front_curvature_.ptr[n] =-kappa_max;
//  }

//  front_curvature_.restore_array();

  ierr = PetscLogEventEnd(log_my_p4est_multialloy_compute_geometric_properties, 0, 0, 0, 0); CHKERRXX(ierr);
}




void my_p4est_multialloy_t::compute_geometric_properties_contr()
{
  ierr = PetscLogEventBegin(log_my_p4est_multialloy_compute_geometric_properties, 0, 0, 0, 0); CHKERRXX(ierr);
  ngbd_->second_derivatives_central(contr_phi_.vec, contr_phi_dd_.vec);
  ierr = PetscLogEventEnd(log_my_p4est_multialloy_compute_geometric_properties, 0, 0, 0, 0); CHKERRXX(ierr);
}





void my_p4est_multialloy_t::compute_velocity()
{
  ierr = PetscLogEventBegin(log_my_p4est_multialloy_compute_velocity, 0, 0, 0, 0); CHKERRXX(ierr);
  PetscPrintf(p4est_->mpicomm, "Computing velocity... ");
  // TODO: implement a smarter extend from interface
  vec_and_ptr_dim_t c0_dd(front_normal_.vec);


  ngbd_->second_derivatives_central(cl_[0].vec[0], c0_dd.vec);


  // flattened interface concentration
  vec_and_ptr_t c_interface(front_phi_.vec);


  my_p4est_level_set_t ls(ngbd_);
  ls.set_interpolation_on_interface(quadratic_non_oscillatory_continuous_v2);
  ls.extend_from_interface_to_whole_domain_TVD(front_phi_.vec, cl_[0].vec[0], c_interface.vec);


  vec_and_ptr_dim_t front_velo_tmp(front_phi_dd_.vec);
  vec_and_ptr_t     front_velo_norm_tmp(front_phi_.vec);


  cl_[0]       .get_array();
  cl0_grad_    .get_array();
  front_normal_.get_array();
  c0_dd        .get_array();


  set_velo_interpolation(ngbd_, cl_[0].ptr.data(), cl0_grad_.ptr, c0_dd.ptr, front_normal_.ptr, solute_diff_[0]);
  ls.extend_from_interface_to_whole_domain_TVD(front_phi_.vec, front_velo_norm_tmp.vec, front_velo_norm_[0].vec, 50,
      NULL, 0, 0, &velo);


  cl_[0]       .restore_array();
  cl0_grad_    .restore_array();
  front_normal_.restore_array();
  c0_dd        .restore_array();


  c0_dd.destroy();


  //cl_[0]             .get_array();
  //cl0_grad_          .get_array();
  //front_normal_      .get_array();
  //c_interface        .get_array();
  //front_velo_tmp     .get_array();
  //front_velo_norm_tmp.get_array();
  //
  //double xyz[P4EST_DIM];
  //
  //quad_neighbor_nodes_of_node_t qnnn;
  //for(size_t i=0; i<ngbd_->get_layer_size(); ++i)
  //{
  //  p4est_locidx_t n = ngbd_->get_layer_node(i);
////    qnnn = ngbd_->get_neighbors(n);
  //
  //  XCODE( front_velo_tmp.ptr[0][n] = -cl0_grad_.ptr[0][n]*solute_diff_[0] / (1.-part_coeff_[0]) / MAX(c_interface.ptr[n], 1e-7) );
  //  YCODE( front_velo_tmp.ptr[1][n] = -cl0_grad_.ptr[1][n]*solute_diff_[0] / (1.-part_coeff_[0]) / MAX(c_interface.ptr[n], 1e-7) );
  //  ZCODE( front_velo_tmp.ptr[2][n] = -cl0_grad_.ptr[2][n]*solute_diff_[0] / (1.-part_coeff_[0]) / MAX(c_interface.ptr[n], 1e-7) );
  //
  //  front_velo_norm_tmp.ptr[n] = SUMD(front_velo_tmp.ptr[0][n]*front_normal_.ptr[0][n],
  //                                    front_velo_tmp.ptr[1][n]*front_normal_.ptr[1][n],
  //                                    front_velo_tmp.ptr[2][n]*front_normal_.ptr[2][n]);
  //}
  //
  //ierr = VecGhostUpdateBegin(front_velo_norm_tmp.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  //foreach_dimension(dim)
  //{
  //  ierr = VecGhostUpdateBegin(front_velo_tmp.vec[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  //}
  //
  //for(size_t i=0; i<ngbd_->get_local_size(); ++i)
  //{
  //  p4est_locidx_t n = ngbd_->get_local_node(i);
  //  //    qnnn = ngbd_->get_neighbors(n);
  //
  //  XCODE( front_velo_tmp.ptr[0][n] = -cl0_grad_.ptr[0][n]*solute_diff_[0] / (1.-part_coeff_[0]) / MAX(c_interface.ptr[n], 1e-7) );
  //  YCODE( front_velo_tmp.ptr[1][n] = -cl0_grad_.ptr[1][n]*solute_diff_[0] / (1.-part_coeff_[0]) / MAX(c_interface.ptr[n], 1e-7) );
  //  ZCODE( front_velo_tmp.ptr[2][n] = -cl0_grad_.ptr[2][n]*solute_diff_[0] / (1.-part_coeff_[0]) / MAX(c_interface.ptr[n], 1e-7) );
  //
  //  front_velo_norm_tmp.ptr[n] = SUMD(front_velo_tmp.ptr[0][n]*front_normal_.ptr[0][n],
  //                                    front_velo_tmp.ptr[1][n]*front_normal_.ptr[1][n],
  //                                    front_velo_tmp.ptr[2][n]*front_normal_.ptr[2][n]);
  //}
  //
  //ierr = VecGhostUpdateEnd(front_velo_norm_tmp.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  //foreach_dimension(dim)
  //{
  //  ierr = VecGhostUpdateEnd(front_velo_tmp.vec[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  //}
  //
  //cl_[0]             .restore_array();
  //cl0_grad_          .restore_array();
  //front_normal_      .restore_array();
  //c_interface        .restore_array();
  //front_velo_tmp     .restore_array();
  //front_velo_norm_tmp.restore_array();
  //
  //ls.extend_from_interface_to_whole_domain_TVD(front_phi_.vec, front_velo_norm_tmp.vec, front_velo_norm_[0].vec);


  foreach_dimension(dim)
  {
    VecPointwiseMultGhost(front_velo_[0].vec[dim], front_velo_norm_[0].vec, front_normal_.vec[dim]);
    //ls.extend_from_interface_to_whole_domain_TVD(front_phi_.vec, front_velo_tmp.vec[dim], front_velo_[0].vec[dim]);
  }

  c_interface.destroy();
  front_velo_norm_tmp.destroy();
  front_velo_tmp.destroy();

  if (enforce_planar_front_)
  {
    vec_and_ptr_t ones(front_phi_.vec);
    VecSetGhost(ones.vec, 1.);

    double velo_norm_avg = integrate_over_interface(p4est_, nodes_, front_phi_.vec, front_velo_norm_[0].vec)
                           /integrate_over_interface(p4est_, nodes_, front_phi_.vec, ones.vec);

    VecSetGhost(front_velo_norm_[0].vec, velo_norm_avg);

    foreach_dimension(dim)
    {
      VecCopyGhost(front_normal_.vec[dim], front_velo_[0].vec[dim]);
      VecScaleGhost(front_velo_[0].vec[dim], velo_norm_avg);
    }

    ones.destroy();
  }

  PetscPrintf(p4est_->mpicomm, "done!\n");

  ierr = PetscLogEventEnd(log_my_p4est_multialloy_compute_velocity, 0, 0, 0, 0); CHKERRXX(ierr);
}






void my_p4est_multialloy_t::compute_dt()
{
  ierr = PetscLogEventBegin(log_my_p4est_multialloy_compute_dt, 0, 0, 0, 0); CHKERRXX(ierr);

  // TO-DO MULTICOMP: add consideration of the Navier stokes timestep


  PetscPrintf(p4est_->mpicomm, "Computing time step: ");
  double velo_norm_max = 0;
  double curvature_max = 0;

  front_phi_         .get_array();
  front_curvature_   .get_array();
  front_velo_norm_[0].get_array();

  foreach_local_node(n, nodes_)
  {
    if (fabs(front_phi_.ptr[n]) < dxyz_close_interface_)
    {
//      printf("front velo norm [n] = %0.3e \n", front_velo_norm_[0].ptr[n]);
      velo_norm_max = MAX(velo_norm_max, fabs(front_velo_norm_[0].ptr[n]));
      curvature_max = MAX(curvature_max, fabs(front_curvature_.ptr[n]));
    }
  }

  double buffer[] = {velo_norm_max, curvature_max};
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, buffer, 2, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
  velo_norm_max = buffer[0];
  curvature_max = buffer[1];

  front_velo_norm_max_ = velo_norm_max;

  for (int i = 1; i < num_time_layers_; ++i)
  {
    dt_[i] = dt_[i-1];
  }
  dt_[0] = cfl_number_ * dxyz_min_/MAX(fabs(velo_norm_max),EPS);
  //std::cout<<"dt_ :: "<< dt_[0]<<"\n";
  //std::cout<<"dt_min :: "<< dt_min_<<"\n";
  //std::cout<<"dt_max :: "<< dt_max_<<"\n";
  dt_[0] = MIN(dt_[0], dt_max_);
  dt_[0] = MAX(dt_[0], dt_min_);

  PetscPrintf(p4est_->mpicomm, "dt_[0] = %0.3e, dt_max = %0.3e, dt_min = %0.3e \n", dt_[0], dt_max_, dt_min_);

  if(solve_with_fluids){
    stefan_w_fluids_solver->get_ns_solver()->compute_dt();
    double dt_NS= stefan_w_fluids_solver->get_ns_solver()->get_dt();
    //std::cout<<"dt_NS :: "<< dt_NS<<"\n";
    dt_[0]=MIN(dt_NS,dt_[0]);

    PetscPrintf(p4est_->mpicomm, "dt_NS = %0.3e \n", dt_NS);

  }

  double cfl_tmp = dt_[0]*MAX(fabs(velo_norm_max),EPS)/dxyz_min_;

  PetscPrintf(p4est_->mpicomm, "curvature max = %e, velo max = %e, dt = %e, eff cfl = %e done!\n", curvature_max, velo_norm_max/scaling_, dt_[0], cfl_tmp);

  //std::cout<< "Plotting contents of dt\n";
  //std::cout<< "cfl stefan :: " << cfl_number_ <<"\n";
  //std::cout<< "dxy_min :: " << dxyz_min_ <<"\n";;
  //std::cout<< "velocity max norm " << fabs(velo_norm_max) <<"\n";
  //std::cout<< "dt_ :"<< dt_[0] << "\n";
  //std::exit(EXIT_FAILURE);
  ierr = PetscLogEventEnd(log_my_p4est_multialloy_compute_dt, 0, 0, 0, 0); CHKERRXX(ierr);
}



void my_p4est_multialloy_t::update_grid()
{
  ierr = PetscLogEventBegin(log_my_p4est_multialloy_update_grid, 0, 0, 0, 0); CHKERRXX(ierr);

  PetscPrintf(p4est_->mpicomm, "Updating grid... ");

  // advect interface and update p4est
  p4est_t       *p4est_np1 = p4est_copy(p4est_, P4EST_FALSE);
  p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);

  my_p4est_semi_lagrangian_t sl(&p4est_np1, &nodes_np1, &ghost_np1, ngbd_, ngbd_);

  sl.set_phi_interpolation(quadratic_non_oscillatory_continuous_v2);
  sl.set_velo_interpolation(quadratic_non_oscillatory_continuous_v2);
  //  sl.set_velo_interpolation(linear);

  // save copy of old level-set function (used for front regularization later)
  // (gonna use front_curvature_ as tmp, it will be destroyed later anyway)
  ierr = VecCopyGhost(front_phi_.vec, front_curvature_.vec); CHKERRXX(ierr);

  if (num_time_layers_ == 2) {
    sl.update_p4est(front_velo_[0].vec, dt_[0], front_phi_.vec, NULL, contr_phi_.vec);
  } else {
    sl.update_p4est(front_velo_[1].vec, front_velo_[0].vec, dt_[1], dt_[0], front_phi_.vec, NULL, contr_phi_.vec);
  }

  /* interpolate the quantities onto the new grid */
  // also shifts n+1 -> n

  PetscPrintf(p4est_->mpicomm, "done!\n");
  PetscPrintf(p4est_->mpicomm, "Transfering data between grids... ");
  ierr = PetscLogEventBegin(log_my_p4est_multialloy_update_grid_transfer_data, 0, 0, 0, 0); CHKERRXX(ierr);
  my_p4est_interpolation_nodes_t interp(ngbd_);

  double xyz[P4EST_DIM];
  foreach_node(n, nodes_np1)
  {
    node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);
    interp.add_point(n, xyz);
  }

  vec_and_ptr_t front_phi_old;
  front_phi_old.create(front_phi_.vec);
  interp.set_input(front_curvature_.vec, interpolation_between_grids_);
  interp.interpolate(front_phi_old.vec);

  // interpolate old level-set function
  front_phi_dd_.destroy();
  front_phi_dd_.create(p4est_np1, nodes_np1);
  front_curvature_.destroy();
  front_curvature_.create(front_phi_.vec);
  front_normal_.destroy();
  front_normal_.create(front_phi_dd_.vec);
  /* temperature */
  for (int j = num_time_layers_-1; j > 0; --j)
  {
    tl_[j].destroy();
    tl_[j].create(front_phi_.vec);
    interp.set_input(tl_[j-1].vec, interpolation_between_grids_);
    interp.interpolate(tl_[j].vec);

    ts_[j].destroy();
    ts_[j].create(front_phi_.vec);
    interp.set_input(ts_[j-1].vec, interpolation_between_grids_);
    interp.interpolate(ts_[j].vec);

    cl_[j].destroy();
    cl_[j].create(front_phi_.vec);
    for (int i = 0; i < num_comps_; ++i)
    {
      interp.set_input(cl_[j-1].vec[i], interpolation_between_grids_);
      interp.interpolate(cl_[j].vec[i]);
    }

    front_velo_norm_[j].destroy();
    front_velo_norm_[j].create(front_phi_.vec);
    interp.set_input(front_velo_norm_[j-1].vec, interpolation_between_grids_);
    interp.interpolate(front_velo_norm_[j].vec);

    front_velo_[j].destroy();
    front_velo_[j].create(front_phi_dd_.vec);
    foreach_dimension(dim)
    {
      interp.set_input(front_velo_[j-1].vec[dim], interpolation_between_grids_);
      interp.interpolate(front_velo_[j].vec[dim]);
    }
  }

  tl_[0].destroy();
  tl_[0].create(front_phi_.vec);
  VecCopyGhost(tl_[1].vec, tl_[0].vec);
  ts_[0].destroy();
  ts_[0].create(front_phi_.vec);
  VecCopyGhost(ts_[1].vec, ts_[0].vec);
  cl_[0].destroy();
  cl_[0].create(front_phi_.vec);
  for (int i = 0; i < num_comps_; ++i)
  {
    VecCopyGhost(cl_[1].vec[i], cl_[0].vec[i]);
  }
  front_velo_[0].destroy();
  front_velo_[0].create(front_phi_dd_.vec);
  front_velo_norm_[0].destroy();
  front_velo_norm_[0].create(front_phi_.vec);

  vec_and_ptr_t contr_phi_tmp(p4est_np1, nodes_np1);
  interp.set_input(contr_phi_.vec, linear);
  interp.interpolate(contr_phi_tmp.vec);
  contr_phi_.destroy();
  contr_phi_.set(contr_phi_tmp.vec);
  contr_phi_dd_.destroy();
  contr_phi_dd_.create(p4est_np1, nodes_np1);

  cl0_grad_.destroy();
  cl0_grad_.create(front_phi_dd_.vec);

  dendrite_number_.destroy();
  dendrite_number_.create(front_phi_.vec);
  dendrite_tip_.destroy();
  dendrite_tip_.create(front_phi_.vec);
  bc_error_.destroy();
  bc_error_.create(front_phi_.vec);
  smoothed_nodes_.destroy();
  smoothed_nodes_.create(front_phi_.vec);
  front_phi_unsmooth_.destroy();
  front_phi_unsmooth_.create(front_phi_.vec);
  ierr = VecCopyGhost(front_phi_.vec, front_phi_unsmooth_.vec); CHKERRXX(ierr);

  if (num_seeds_ == 1)
  {
    seed_map_.destroy();
    seed_map_.create(front_phi_.vec);
    VecSetGhost(seed_map_.vec, 0.);
  } else {
    vec_and_ptr_t tmp(front_phi_.vec);
    interp.set_input(seed_map_.vec, linear);
    interp.interpolate(tmp.vec);
    seed_map_.destroy();
    seed_map_.set(tmp.vec);
  }

  vec_and_ptr_t psi_tl_tmp(front_phi_.vec);
  interp.set_input(psi_tl_.vec, linear);
  interp.interpolate(psi_tl_tmp.vec);
  psi_tl_.destroy();
  psi_tl_.set(psi_tl_tmp.vec);

  vec_and_ptr_t psi_ts_tmp(front_phi_.vec);
  interp.set_input(psi_ts_.vec, linear);
  interp.interpolate(psi_ts_tmp.vec);
  psi_ts_.destroy();
  psi_ts_.set(psi_ts_tmp.vec);

  vec_and_ptr_array_t psi_cl_tmp(num_comps_, front_phi_.vec);
  for (int i = 0; i < num_comps_; ++i)
  {
    interp.set_input(psi_cl_.vec[i], linear);
    interp.interpolate(psi_cl_tmp.vec[i]);
  }
  psi_cl_.destroy();
  psi_cl_.set(psi_cl_tmp.vec.data());

  p4est_destroy      (p4est_); p4est_ = p4est_np1;
  p4est_ghost_destroy(ghost_); ghost_ = ghost_np1;
  p4est_nodes_destroy(nodes_); nodes_ = nodes_np1;
  hierarchy_->update(p4est_, ghost_);
  ngbd_->update(hierarchy_, nodes_);

  PetscPrintf(p4est_->mpicomm, "done!\n");
  ierr = PetscLogEventEnd(log_my_p4est_multialloy_update_grid_transfer_data, 0, 0, 0, 0); CHKERRXX(ierr);
  regularize_front(front_phi_old.vec);
  front_phi_old.destroy();

  /* reinitialize phi */
  my_p4est_level_set_t ls_new(ngbd_);
//  ls_new.reinitialize_1st_order_time_2nd_order_space(front_phi_.vec, 20);
//  ls_new.reinitialize_1st_order_time_2nd_order_space(front_phi_.vec, 100);
  ls_new.reinitialize_2nd_order(front_phi_.vec, 50);

  if (num_seeds_ > 1)
  {
    VecScaleGhost(front_phi_.vec, -1.);

    ls_new.extend_Over_Interface_TVD_Full(front_phi_.vec, seed_map_.vec, 20, 1); // Rochi changing 0 to 1

    seed_map_.get_array();
    foreach_node(n, nodes_) seed_map_.ptr[n] = round(seed_map_.ptr[n]);
    seed_map_.restore_array();
    VecScaleGhost(front_phi_.vec, -1.);
  }

  /* second derivatives, normals, curvature, angles */
  compute_geometric_properties_front();
  compute_geometric_properties_contr();


  ierr = PetscLogEventEnd(log_my_p4est_multialloy_update_grid, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_multialloy_t::update_grid_w_fluids(){
  ierr = PetscLogEventBegin(log_my_p4est_multialloy_update_grid, 0, 0, 0, 0); CHKERRXX(ierr);

  PetscPrintf(p4est_->mpicomm, "Updating grid (with fluids)... ");
  int num_nodes = nodes_->num_owned_indeps;
  MPI_Allreduce(MPI_IN_PLACE,&num_nodes,1,MPI_INT,MPI_SUM, p4est_->mpicomm);
  PetscPrintf(p4est_->mpicomm, "(Number of nodes before: %d) \n", num_nodes);

  if(0){
    PetscPrintf(p4est_->mpicomm, "\n \n Saving fields before grid update \n");
    // -------------------------------
    // TEMPORARY: save fields before grid update
    // -------------------------------
    std::vector<Vec_for_vtk_export_t> point_fields;
    std::vector<Vec_for_vtk_export_t> cell_fields = {};

    point_fields.push_back(Vec_for_vtk_export_t(front_phi_.vec, "phi"));
    point_fields.push_back(Vec_for_vtk_export_t(cl_[0].vec[0], "cl_tn_0"));
    point_fields.push_back(Vec_for_vtk_export_t(cl_[0].vec[1], "cl_tn_1"));

    point_fields.push_back(Vec_for_vtk_export_t(cl_[1].vec[0], "cl_tnm1_0"));
    point_fields.push_back(Vec_for_vtk_export_t(cl_[1].vec[1], "cl_tnm1_1"));

    point_fields.push_back(Vec_for_vtk_export_t(tl_[0].vec, "tl_tn"));
    point_fields.push_back(Vec_for_vtk_export_t(tl_[1].vec, "tl_tnm1_1"));

    point_fields.push_back(Vec_for_vtk_export_t(v_n.vec[0], "vx_n"));
    point_fields.push_back(Vec_for_vtk_export_t(v_n.vec[1], "vy_n"));
    point_fields.push_back(Vec_for_vtk_export_t(vorticity.vec, "vort"));
//    point_fields.push_back(Vec_for_vtk_export_t(vorticity_refine.vec, "vort_refine"));
//    point_fields.push_back(Vec_for_vtk_export_t(v_nm1.vec[0], "vx_nm1"));
//    point_fields.push_back(Vec_for_vtk_export_t(v_nm1.vec[1], "vy_nm1"));


    const char* out_dir = getenv("OUT_DIR");
    if(!out_dir){
      throw std::invalid_argument("You need to set the output directory for VTK: OUT_DIR_VTK");
    }

    char filename[1000];
    sprintf(filename, "%s/snapshot_before_grid_update_%d", out_dir, iteration_w_fluids);
    my_p4est_vtk_write_all_lists(p4est_, nodes_, ngbd_->get_ghost(), P4EST_TRUE, P4EST_TRUE, filename, point_fields, cell_fields);
    point_fields.clear();




//    point_fields.push_back(Vec_for_vtk_export_t(v_nm1.vec[0], "vx_nm1"));
//    point_fields.push_back(Vec_for_vtk_export_t(v_nm1.vec[1], "vy_nm1"));

//    char filename2[1000];
//    sprintf(filename2, "%s/snapshot_before_grid_update_nm1", out_dir);
//    my_p4est_vtk_write_all_lists(p4est_nm1, nodes_nm1, ngbd_nm1->get_ghost(), P4EST_TRUE, P4EST_TRUE, filename2, point_fields, cell_fields);
//    point_fields.clear();


    PetscPrintf(p4est_->mpicomm, "Done! \n \n \n");

  }

  p4est_destroy(p4est_nm1);
  p4est_ghost_destroy(ghost_nm1); ghost_nm1 = NULL;
  p4est_nodes_destroy(nodes_nm1);
  delete ngbd_nm1;
  delete hierarchy_nm1;

  p4est_nm1 = p4est_;
  ghost_nm1 = ghost_;
  nodes_nm1 = nodes_;

  hierarchy_nm1 = hierarchy_;
  ngbd_nm1 = ngbd_;

  my_p4est_navier_stokes_t* ns = stefan_w_fluids_solver->get_ns_solver();
  ns->nullify_p4est_nm1();

  // advect interface and update p4est
//  p4est_t       *p4est_np1 = p4est_copy(p4est_, P4EST_FALSE);
//  p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
//  p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);
//  my_p4est_hierarchy_t* hierarchy_np1 = new my_p4est_hierarchy_t(p4est_np1, ghost_np1, &brick_);
//  my_p4est_node_neighbors_t* ngbd_np1 = new my_p4est_node_neighbors_t(hierarchy_np1,nodes_np1);

  p4est_ = p4est_copy(p4est_nm1, P4EST_FALSE);
  ghost_ = my_p4est_ghost_new(p4est_, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est_, ghost_);
  nodes_ = my_p4est_nodes_new(p4est_, ghost_);

  // Get the new neighbors and hierarchy
  hierarchy_ = new my_p4est_hierarchy_t(p4est_, ghost_, &brick_);
  ngbd_ = new my_p4est_node_neighbors_t(hierarchy_, nodes_);

  //initialize neightbors
  ngbd_-> init_neighbors();

  bool use_block = false;
  bool expand_ghost_layer = true;

  std::vector<compare_option_t> compare_opn;
  std::vector<compare_diagonal_option_t> diag_opn;
  std::vector<double> criteria;
  std::vector<int> custom_lmax;
  PetscInt num_fields = 0;
  //refine_by_vorticity = vorticity.vec!=NULL;
  refine_by_vorticity=false;
  refine_by_d2C=false;
  refine_by_d2T=false;
  // -----------------------
  // Count number of refinement fields and create vectors for necessary fields:
  // ------------------------
  if(refine_by_vorticity) {
    num_fields+=1;
    vorticity_refine.create(p4est_, nodes_);
  }// for vorticity
  if(refine_by_d2T){
    num_fields+=2;
    tl_dd.create(p4est_, nodes_);
    ngbd_->second_derivatives_central(tl_[num_time_layers_-1].vec,tl_dd.vec);
  }// for temperature
  if(refine_by_d2C){
    num_fields+=2;
    cl0_dd.create(p4est_,nodes_);
    ngbd_->second_derivatives_central(cl_[num_time_layers_-1].vec[0],cl0_dd.vec);
  }// for concentration (using the first component ) // might change later

  Vec fields_[num_fields];
  // preparing refinement fields
  if(num_fields>0){
    // Get relevant arrays:
    if(refine_by_vorticity){
      vorticity.get_array();
      vorticity_refine.get_array();
    }
    if(refine_by_d2T) {tl_dd.get_array();}
    front_phi_.get_array();

    // Compute proper refinement fields on layer nodes:
    for(size_t i = 0; i<ngbd_->get_layer_size(); i++){
      p4est_locidx_t n = ngbd_->get_layer_node(i);
      if(front_phi_.ptr[n] < 0.){
        if(refine_by_vorticity)vorticity_refine.ptr[n] = vorticity.ptr[n];
      }
      else{
        if(refine_by_vorticity) vorticity_refine.ptr[n] = 0.0;
        if(refine_by_d2T){ // Set to 0 in solid subdomain, don't want to refine by T_l_dd in there
          foreach_dimension(d){
            tl_dd.ptr[d][n]=0.;
          }
        }
      }
    } // end of loop over layer nodes

    // Begin updating the ghost values:
    if(refine_by_vorticity)ierr = VecGhostUpdateBegin(vorticity_refine.vec,INSERT_VALUES,SCATTER_FORWARD);
    if(refine_by_d2T){
      foreach_dimension(d){
        ierr = VecGhostUpdateBegin(tl_dd.vec[d],INSERT_VALUES,SCATTER_FORWARD);
      }
    }

    //Compute proper refinement fields on local nodes:
    for(size_t i = 0; i<ngbd_->get_local_size(); i++){
      p4est_locidx_t n = ngbd_->get_local_node(i);
      if(front_phi_.ptr[n] < 0.){
        if(refine_by_vorticity)vorticity_refine.ptr[n] = vorticity.ptr[n];
      }
      else{
        if(refine_by_vorticity)vorticity_refine.ptr[n] = 0.0;
        if(refine_by_d2T){ // Set to 0 in solid subdomain, don't want to refine by T_l_dd in there
          foreach_dimension(d){
            tl_dd.ptr[d][n]=0.;
          }
        }
      }
    } // end of loop over local nodes

    // Finish updating the ghost values:
    if(refine_by_vorticity)ierr = VecGhostUpdateEnd(vorticity_refine.vec,INSERT_VALUES,SCATTER_FORWARD);
    if(refine_by_d2T){
      foreach_dimension(d){
        ierr = VecGhostUpdateEnd(tl_dd.vec[d],INSERT_VALUES,SCATTER_FORWARD);
      }
    }

    // Restore appropriate arrays:
    if(refine_by_d2T) {tl_dd.restore_array();}
    if(refine_by_vorticity){
      vorticity.restore_array();
      vorticity_refine.restore_array();
    }
    front_phi_.restore_array();

    // ------------------------------------------------------------
    // Add our refinement fields to the array:
    // ------------------------------------------------------------
    PetscInt fields_idx = 0;
    if(refine_by_vorticity)fields_[fields_idx++] = vorticity_refine.vec;
    if(refine_by_d2T){
      fields_[fields_idx++] = tl_dd.vec[0];
      fields_[fields_idx++] = tl_dd.vec[1];
    }

    P4EST_ASSERT(fields_idx ==num_fields);
//    int lint= stefan_w_fluids_solver->get_lint();
//    int lmin= stefan_w_fluids_solver->get_lmin();
//    int lmax= stefan_w_fluids_solver->get_lmax();
    splitting_criteria_t* sp_new = (splitting_criteria_t*) p4est_->user_pointer;
    int lmin = sp_new->min_lvl;
    int lmax = sp_new->max_lvl;
    int lint = -1;
    PetscPrintf(p4est_->mpicomm, "Warning: for refinement, the lint option is not implemented. You might want this later. \n");
    PetscPrintf(p4est_->mpicomm, "INSIDE MULTI GRID UPDATE: "
           "lmin = %d, lmax = %d, lint = %d \n", lmin, lmax, lint);



    double NS_norm= stefan_w_fluids_solver->get_NS_norm();

    // ------------------------------------------------------------
    // Add our instructions:
    // ------------------------------------------------------------
    // Coarsening instructions: (for vorticity)
    PetscPrintf(p4est_->mpicomm,"\nRefine by vort = %d, refine_by_d2T = %d, refine_by_d2C= %d \n", refine_by_vorticity, refine_by_d2T, refine_by_d2C);
    if(refine_by_vorticity){
      compare_opn.push_back(LESS_THAN);
      diag_opn.push_back(DIVIDE_BY);
      criteria.push_back(vorticity_threshold*NS_norm/2.);

      // Refining instructions: (for vorticity)
      compare_opn.push_back(GREATER_THAN);
      diag_opn.push_back(DIVIDE_BY);
      criteria.push_back(vorticity_threshold*NS_norm);

      if(lint>0){custom_lmax.push_back(lint);}
      else{custom_lmax.push_back(lmax);}
    }
    if(refine_by_d2T){
      PetscPrintf(p4est_->mpicomm, "Warning: you have activated refine_by_d2T, but this has not been updated to the diimensional case \n");
      double dxyz_smallest[P4EST_DIM];
      dxyz_min(p4est_,dxyz_smallest);
      double theta_infty= stefan_w_fluids_solver->get_theta_infty();
      double theta_interface= stefan_w_fluids_solver->get_theta_interface();
      double theta0= stefan_w_fluids_solver->get_theta0();
      double deltaT= stefan_w_fluids_solver->get_deltaT();
      double dTheta= fabs(theta_infty - theta_interface)>0 ? fabs(theta_infty - theta_interface): 1.0;
      dTheta/=SQR(MIN(dxyz_smallest[0],dxyz_smallest[1])); // max d2Theta in liquid subdomain

      // Define variables for the refine/coarsen instructions for d2T fields:
      compare_diagonal_option_t diag_opn_d2T = DIVIDE_BY;
      compare_option_t compare_opn_d2T = SIGN_CHANGE;
      double refine_criteria_d2T = dTheta*d2T_threshold;
      double coarsen_criteria_d2T = dTheta*d2T_threshold*0.1;

      // Coarsening instructions: (for d2T/dx2)
      compare_opn.push_back(compare_opn_d2T);
      diag_opn.push_back(diag_opn_d2T);
      criteria.push_back(coarsen_criteria_d2T); // did 0.1* () for the coarsen if no sign change OR below threshold case

      // Refining instructions: (for d2T/dx2)
      compare_opn.push_back(compare_opn_d2T);
      diag_opn.push_back(diag_opn_d2T);
      criteria.push_back(refine_criteria_d2T);
      if(lint>0){custom_lmax.push_back(lint);}
      else{custom_lmax.push_back(lmax);}

      // Coarsening instructions: (for d2T/dy2)
      compare_opn.push_back(compare_opn_d2T);
      diag_opn.push_back(diag_opn_d2T);
      criteria.push_back(coarsen_criteria_d2T);

      // Refining instructions: (for d2T/dy2)
      compare_opn.push_back(compare_opn_d2T);
      diag_opn.push_back(diag_opn_d2T);
      criteria.push_back(refine_criteria_d2T);
      if(lint>0){custom_lmax.push_back(lint);}
      else{custom_lmax.push_back(lmax);}
    }
    if(refine_by_d2C){
      PetscPrintf(p4est_->mpicomm, "Warning: you have activated refine_by_d2C, but this has not been implemented. Bypassing ... \n");
    }

  } // end of "if num_fields!=0"

  // Create second derivatives for phi in the case that we are using update_p4est:
  front_phi_dd_.create(p4est_, nodes_);
  ngbd_->second_derivatives_central(front_phi_.vec, front_phi_dd_.vec);
  double uniform_band= stefan_w_fluids_solver->get_uniform_band();

  my_p4est_semi_lagrangian_t sl(&p4est_, &nodes_, &ghost_, ngbd_nm1);

  sl.set_phi_interpolation(quadratic_non_oscillatory_continuous_v2);
  sl.set_velo_interpolation(quadratic_non_oscillatory_continuous_v2);

  sl.update_p4est(front_velo_[0].vec, dt_[0],
                  front_phi_.vec, front_phi_dd_.vec,
                  NULL, num_fields, use_block, true, uniform_band, uniform_band*1.5,
                  fields_, NULL, criteria,
                  compare_opn, diag_opn, custom_lmax, expand_ghost_layer);

  //front_phi_dd_.destroy();
  if(refine_by_vorticity){
    vorticity_refine.destroy();
//printf("\n!!!! do not leave this uncommented !!! refine vorticity needs to be destroyed !! red alert\n");
  }
  if(refine_by_d2T){
    tl_dd.destroy();
  }
  if(refine_by_d2C){
    cl0_dd.destroy();
  }
  PetscPrintf(p4est_->mpicomm, "done!\n");
  ierr = PetscLogEventEnd(log_my_p4est_multialloy_update_grid, 0, 0, 0, 0); CHKERRXX(ierr);
  //end of update grid


  PetscPrintf(p4est_->mpicomm, "Transfering data between grids... ");

  // -------------------------------
  // Update hierarchy and neighbors to match new updated grid:
  // -------------------------------
//  hierarchy_np1->update(p4est_np1, ghost_np1);
//  ngbd_np1->update(hierarchy_np1, nodes_np1);

//  // Initialize the neigbors:
//  ngbd_np1->init_neighbors();


  hierarchy_->update(p4est_, ghost_);
  ngbd_->update(hierarchy_, nodes_);
  ngbd_->init_neighbors();
  printf("\n\n aaa \n");

  ierr = PetscLogEventBegin(log_my_p4est_multialloy_update_grid_transfer_data, 0, 0, 0, 0); CHKERRXX(ierr);
  my_p4est_interpolation_nodes_t interp(ngbd_nm1);

  double xyz[P4EST_DIM];
  foreach_node(n, nodes_)
  {
    node_xyz_fr_n(n, p4est_, nodes_, xyz);
    interp.add_point(n, xyz);
  }
  printf("bbb \n");

  vec_and_ptr_t front_phi_old;
  front_phi_old.create(front_phi_.vec);
  interp.set_input(front_curvature_.vec, interpolation_between_grids_);
  interp.interpolate(front_phi_old.vec);
  printf("ccc \n");

  // interpolate old level-set function
  front_phi_dd_.destroy();
  front_phi_dd_.create(p4est_, nodes_);
  front_curvature_.destroy();
  front_curvature_.create(front_phi_.vec);
  front_normal_.destroy();
  front_normal_.create(front_phi_dd_.vec);
  printf("ddd \n");

  /* temperature */
  for (int j = num_time_layers_-1; j > 0; --j)
  {
    tl_[j].destroy();
    tl_[j].create(front_phi_.vec);
    interp.set_input(tl_[j-1].vec, interpolation_between_grids_);
    interp.interpolate(tl_[j].vec);

    ts_[j].destroy();
    ts_[j].create(front_phi_.vec);
    interp.set_input(ts_[j-1].vec, interpolation_between_grids_);
    interp.interpolate(ts_[j].vec);

    cl_[j].destroy();
    cl_[j].create(front_phi_.vec);
    for (int i = 0; i < num_comps_; ++i)
    {
      interp.set_input(cl_[j-1].vec[i], interpolation_between_grids_);
      interp.interpolate(cl_[j].vec[i]);
    }

    front_velo_norm_[j].destroy();
    front_velo_norm_[j].create(front_phi_.vec);
    interp.set_input(front_velo_norm_[j-1].vec, interpolation_between_grids_);
    interp.interpolate(front_velo_norm_[j].vec);

    front_velo_[j].destroy();
    front_velo_[j].create(front_phi_dd_.vec);
    foreach_dimension(dim)
    {
      interp.set_input(front_velo_[j-1].vec[dim], interpolation_between_grids_);
      interp.interpolate(front_velo_[j].vec[dim]);
    }
  }

  printf("eee \n");
  tl_[0].destroy();
  tl_[0].create(front_phi_.vec);
  VecCopyGhost(tl_[1].vec, tl_[0].vec);
  ts_[0].destroy();
  ts_[0].create(front_phi_.vec);
  VecCopyGhost(ts_[1].vec, ts_[0].vec);
  cl_[0].destroy();
  cl_[0].create(front_phi_.vec);
  for (int i = 0; i < num_comps_; ++i)
  {
    VecCopyGhost(cl_[1].vec[i], cl_[0].vec[i]);
  }
  front_velo_[0].destroy();
  front_velo_[0].create(front_phi_dd_.vec);
  front_velo_norm_[0].destroy();
  front_velo_norm_[0].create(front_phi_.vec);

  vec_and_ptr_t contr_phi_tmp(p4est_, nodes_);
  interp.set_input(contr_phi_.vec, linear);
  interp.interpolate(contr_phi_tmp.vec);
  contr_phi_.destroy();
  contr_phi_.set(contr_phi_tmp.vec);
  contr_phi_dd_.destroy();
  contr_phi_dd_.create(p4est_, nodes_);

  cl0_grad_.destroy();
  cl0_grad_.create(front_phi_dd_.vec);

  dendrite_number_.destroy();
  dendrite_number_.create(front_phi_.vec);
  dendrite_tip_.destroy();
  dendrite_tip_.create(front_phi_.vec);
  bc_error_.destroy();
  bc_error_.create(front_phi_.vec);
  smoothed_nodes_.destroy();
  smoothed_nodes_.create(front_phi_.vec);
  front_phi_unsmooth_.destroy();
  front_phi_unsmooth_.create(front_phi_.vec);
  ierr = VecCopyGhost(front_phi_.vec, front_phi_unsmooth_.vec); CHKERRXX(ierr);

  if (num_seeds_ == 1)
  {
    seed_map_.destroy();
    seed_map_.create(front_phi_.vec);
    VecSetGhost(seed_map_.vec, 0.);
  } else {
    vec_and_ptr_t tmp(front_phi_.vec);
    interp.set_input(seed_map_.vec, linear);
    interp.interpolate(tmp.vec);
    seed_map_.destroy();
    seed_map_.set(tmp.vec);
  }

  vec_and_ptr_t psi_tl_tmp(front_phi_.vec);
  interp.set_input(psi_tl_.vec, linear);
  interp.interpolate(psi_tl_tmp.vec);
  psi_tl_.destroy();
  psi_tl_.set(psi_tl_tmp.vec);

  vec_and_ptr_t psi_ts_tmp(front_phi_.vec);
  interp.set_input(psi_ts_.vec, linear);
  interp.interpolate(psi_ts_tmp.vec);
  psi_ts_.destroy();
  psi_ts_.set(psi_ts_tmp.vec);

  vec_and_ptr_array_t psi_cl_tmp(num_comps_, front_phi_.vec);
  for (int i = 0; i < num_comps_; ++i)
  {
    interp.set_input(psi_cl_.vec[i], linear);
    interp.interpolate(psi_cl_tmp.vec[i]);
  }
  psi_cl_.destroy();
  psi_cl_.set(psi_cl_tmp.vec.data());




//  std::cout<<"line 1790 ok \n";
  //interpolating velocity fields to the new grid

//  int num_velocity_fields=P4EST_DIM; // vNSx, vNSy
  Vec velocity_fields_old[P4EST_DIM];
  Vec velocity_fields_new[P4EST_DIM];
  unsigned int i=0;
  foreach_dimension(d){
    velocity_fields_old[i++]=v_n.vec[d];
  }
  //std::cout<<"line 1799 ok \n";
  PetscErrorCode ierr;
  for(unsigned int j = 0;j<P4EST_DIM;j++){
    ierr = VecCreateGhostNodes(p4est_, nodes_, &velocity_fields_new[j]);CHKERRXX(ierr);
  }
  //std::cout<<"line 1804 ok \n";
  interp.set_input(velocity_fields_old, quadratic_non_oscillatory_continuous_v2,P4EST_DIM);

  double xyz_[P4EST_DIM];
  foreach_node(n, nodes_){
    node_xyz_fr_n(n, p4est_, nodes_,  xyz_);
    interp.add_point(n,xyz_);
  }
  //std::cout<<"line 1806 ok \n";

  interp.interpolate(velocity_fields_new);

  //std::cout<<"line 1808 ok \n";
  interp.clear();
  for(unsigned int k=0;k<P4EST_DIM;k++){
    ierr = VecDestroy(velocity_fields_old[k]); CHKERRXX(ierr); // Destroy objects where the old vectors were
  }

  foreach_dimension(d){
    v_n.vec[d] = velocity_fields_new[d];
  }

  // update vn in the stefan class
  stefan_w_fluids_solver->set_v_n(v_n);

  // update the ns grid (this will handle updating with our new vn inside the NS class)
//  printf("\n Addresses of vns vectors (multialloy, before passing to NS for grid update): \n "
//         "v_n.vec = %p, v_nm1.vec = %p \n", v_n.vec[0], v_nm1.vec[0]);

  ns->update_from_tn_to_tnp1_grid_external(front_phi_.vec, front_phi_old.vec,
                                           v_n.vec, v_nm1.vec,
                                           p4est_, nodes_, ghost_,
                                           ngbd_,
                                           faces_,ngbd_c_,
                                           hierarchy_);

  // Update stefan's copy of the faces:
  stefan_w_fluids_solver->set_faces_np1(faces_);

  compare_opn.clear(); diag_opn.clear(); criteria.clear();
  compare_opn.shrink_to_fit(); diag_opn.shrink_to_fit(); criteria.shrink_to_fit();
  custom_lmax.clear(); custom_lmax.shrink_to_fit();

  // Update the grids inside stefan with fluids solver:
  stefan_w_fluids_solver->set_p4est_n(p4est_nm1);
  stefan_w_fluids_solver->set_nodes_n(nodes_nm1);
  stefan_w_fluids_solver->set_ghost_n(ghost_nm1);
  stefan_w_fluids_solver->set_ngbd_n(ngbd_nm1);
  stefan_w_fluids_solver->set_hierarchy_n(hierarchy_nm1);

  stefan_w_fluids_solver->set_p4est_np1(p4est_);
  stefan_w_fluids_solver->set_nodes_np1(nodes_);
  stefan_w_fluids_solver->set_ghost_np1(ghost_);
  stefan_w_fluids_solver->set_ngbd_np1(ngbd_);
  stefan_w_fluids_solver->set_hierarchy_np1(hierarchy_);


  PetscPrintf(p4est_->mpicomm, "update_grid_w_fluids done!\n");
  ierr = PetscLogEventEnd(log_my_p4est_multialloy_update_grid_transfer_data, 0, 0, 0, 0); CHKERRXX(ierr);

  int num_nodes2 = nodes_->num_owned_indeps;
  MPI_Allreduce(MPI_IN_PLACE,&num_nodes2,1,MPI_INT,MPI_SUM, p4est_->mpicomm);
  PetscPrintf(p4est_->mpicomm, "Number of nodes after: %d \n", num_nodes2);


  if(0){
    PetscPrintf(p4est_->mpicomm," \n \n \n saving fields after grid update \n");
    // -------------------------------
    // TEMPORARY: save fields after grid update
    // -------------------------------
    std::vector<Vec_for_vtk_export_t> point_fields;
    std::vector<Vec_for_vtk_export_t> cell_fields = {};

    point_fields.push_back(Vec_for_vtk_export_t(front_phi_.vec, "phi"));
    point_fields.push_back(Vec_for_vtk_export_t(cl_[0].vec[0], "cl_tn_0"));
    point_fields.push_back(Vec_for_vtk_export_t(cl_[0].vec[1], "cl_tn_1"));

//    point_fields.push_back(Vec_for_vtk_export_t(cl_[1].vec[0], "cl_tnm1_0"));
//    point_fields.push_back(Vec_for_vtk_export_t(cl_[1].vec[1], "cl_tnm1_1"));

    point_fields.push_back(Vec_for_vtk_export_t(tl_[0].vec, "tl_tn"));
//    point_fields.push_back(Vec_for_vtk_export_t(tl_[1].vec, "tl_tnm1_1"));

    point_fields.push_back(Vec_for_vtk_export_t(v_n.vec[0], "vx_n"));
    point_fields.push_back(Vec_for_vtk_export_t(v_n.vec[1], "vy_n"));
//    point_fields.push_back(Vec_for_vtk_export_t(vorticity.vec, "vort"));

    //    point_fields.push_back(Vec_for_vtk_export_t(vorticity_refine.vec, "vort_refine"));

    //    point_fields.push_back(Vec_for_vtk_export_t(v_nm1.vec[0], "vx_nm1"));
    //    point_fields.push_back(Vec_for_vtk_export_t(v_nm1.vec[1], "vy_nm1"));


    const char* out_dir = getenv("OUT_DIR");
    if(!out_dir){
      throw std::invalid_argument("You need to set the output directory for VTK: OUT_DIR_VTK");
    }

    char filename[1000];
    sprintf(filename, "%s/snapshot_after_grid_update_%d", out_dir, iteration_w_fluids);
    my_p4est_vtk_write_all_lists(p4est_, nodes_, ngbd_->get_ghost(), P4EST_TRUE, P4EST_TRUE, filename, point_fields, cell_fields);
    point_fields.clear();


    //    point_fields.push_back(Vec_for_vtk_export_t(v_nm1.vec[0], "vx_nm1"));
    //    point_fields.push_back(Vec_for_vtk_export_t(v_nm1.vec[1], "vy_nm1"));

    //    char filename2[1000];
    //    sprintf(filename2, "%s/snapshot_after_grid_update_nm1", out_dir);
    //    my_p4est_vtk_write_all_lists(p4est_nm1, nodes_nm1, ngbd_nm1->get_ghost(), P4EST_TRUE, P4EST_TRUE, filename2, point_fields, cell_fields);
    //    point_fields.clear();

    PetscPrintf(p4est_->mpicomm, " Done \n");

//    vorticity_refine.destroy();

  }


  regularize_front(front_phi_old.vec);

  /* reinitialize phi */
  my_p4est_level_set_t ls_new(ngbd_);
//  ls_new.reinitialize_1st_order_time_2nd_order_space(front_phi_.vec, 20);
//  ls_new.reinitialize_1st_order_time_2nd_order_space(front_phi_.vec, 100);
  ls_new.reinitialize_2nd_order(front_phi_.vec, 50);

  if (num_seeds_ > 1)
  {
    VecScaleGhost(front_phi_.vec, -1.);

    ls_new.extend_Over_Interface_TVD_Full(front_phi_.vec, seed_map_.vec, 20, 1); // Rochi changing 0 to 1

    seed_map_.get_array();
    foreach_node(n, nodes_) seed_map_.ptr[n] = round(seed_map_.ptr[n]);
    seed_map_.restore_array();
    VecScaleGhost(front_phi_.vec, -1.);
  }

  /* second derivatives, normals, curvature, angles */
//  std::cout<<"ns update over \n";
  compute_geometric_properties_front();
  compute_geometric_properties_contr();
  front_phi_old.destroy();




}

void my_p4est_multialloy_t::update_grid_solid()
{
  ierr = PetscLogEventBegin(log_my_p4est_multialloy_update_grid_solid, 0, 0, 0, 0); CHKERRXX(ierr);
  PetscPrintf(p4est_->mpicomm, "Refining auxiliary p4est for storing data... ");

  Vec tmp = solid_front_phi_.vec;
  solid_front_phi_.vec = solid_front_phi_nm1_.vec;
  solid_front_phi_nm1_.vec = tmp;

  p4est_t       *solid_p4est_np1 = p4est_copy(solid_p4est_, P4EST_FALSE);
  p4est_ghost_t *solid_ghost_np1 = my_p4est_ghost_new(solid_p4est_np1, P4EST_CONNECT_FULL);
  p4est_nodes_t *solid_nodes_np1 = my_p4est_nodes_new(solid_p4est_np1, solid_ghost_np1);

  splitting_criteria_t* sp_old = (splitting_criteria_t*)solid_ngbd_->p4est->user_pointer;

  vec_and_ptr_t front_phi_np1(solid_p4est_np1, solid_nodes_np1);

  bool is_grid_changing = true;
  int  counter          = 0;

  while (is_grid_changing && counter < 5)
  {
    // interpolate from a coarse grid to a fine one
    front_phi_np1.get_array();

    my_p4est_interpolation_nodes_t interp(ngbd_);

    double xyz[P4EST_DIM];
    foreach_node(n, solid_nodes_np1)
    {
      node_xyz_fr_n(n, solid_p4est_np1, solid_nodes_np1, xyz);
      interp.add_point(n, xyz);
    }

    interp.set_input(front_phi_.vec, linear);
    interp.interpolate(front_phi_np1.ptr);

    splitting_criteria_tag_t sp(sp_old->min_lvl, sp_old->max_lvl, sp_old->lip);
    is_grid_changing = sp.refine(solid_p4est_np1, solid_nodes_np1, front_phi_np1.ptr);

    front_phi_np1.restore_array();

    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &is_grid_changing, 1, MPI_C_BOOL, MPI_LOR, p4est_->mpicomm); SC_CHECK_MPI(mpiret);

    if (is_grid_changing)
    {
      my_p4est_partition(solid_p4est_np1, P4EST_TRUE, NULL);

      // reset nodes, ghost, and phi
      p4est_ghost_destroy(solid_ghost_np1); solid_ghost_np1 = my_p4est_ghost_new(solid_p4est_np1, P4EST_CONNECT_FULL);
      p4est_nodes_destroy(solid_nodes_np1); solid_nodes_np1 = my_p4est_nodes_new(solid_p4est_np1, solid_ghost_np1);

      front_phi_np1.destroy();
      front_phi_np1.create(solid_p4est_np1, solid_nodes_np1);
    }

    counter++;
  }

  solid_p4est_np1->user_pointer = (void*) sp_old;

  solid_front_phi_.destroy();
  solid_front_phi_.set(front_phi_np1.vec);

  PetscPrintf(p4est_->mpicomm, "almost there... ");

  // transfer variables to the new grid
  my_p4est_interpolation_nodes_t solid_interp(solid_ngbd_);

  double xyz[P4EST_DIM];
  foreach_node(n, solid_nodes_np1)
  {
    node_xyz_fr_n(n, solid_p4est_np1, solid_nodes_np1, xyz);
    solid_interp.add_point(n, xyz);
  }

  vec_and_ptr_t tf_tmp(solid_front_phi_.vec);
  solid_interp.set_input(solid_tf_.vec, linear);
  solid_interp.interpolate(tf_tmp.vec);
  solid_tf_.destroy();
  solid_tf_.set(tf_tmp.vec);

  vec_and_ptr_array_t cs_tmp(num_comps_, solid_front_phi_.vec);
  for (int i = 0; i < num_comps_; ++i)
  {
    solid_interp.set_input(solid_cl_.vec[i], linear);
    solid_interp.interpolate(cs_tmp.vec[i]);
  }
  solid_cl_.destroy();
  solid_cl_.set(cs_tmp.vec.data());

  vec_and_ptr_array_t part_coeff_tmp(num_comps_, solid_front_phi_.vec);
  for (int i = 0; i < num_comps_; ++i)
  {
    solid_interp.set_input(solid_part_coeff_.vec[i], linear);
    solid_interp.interpolate(part_coeff_tmp.vec[i]);
  }
  solid_part_coeff_.destroy();
  solid_part_coeff_.set(part_coeff_tmp.vec.data());

  vec_and_ptr_t curv_tmp(solid_front_phi_.vec);
  solid_interp.set_input(solid_front_curvature_.vec, linear);
  solid_interp.interpolate(curv_tmp.vec);
  solid_front_curvature_.destroy();
  solid_front_curvature_.set(curv_tmp.vec);

  vec_and_ptr_t velo_tmp(solid_front_phi_.vec);
  solid_interp.set_input(solid_front_velo_norm_.vec, linear);
  solid_interp.interpolate(velo_tmp.vec);
  solid_front_velo_norm_.destroy();
  solid_front_velo_norm_.set(velo_tmp.vec);

  vec_and_ptr_t phi_nm1_tmp(solid_front_phi_.vec);
  solid_interp.set_input(solid_front_phi_nm1_.vec, linear);
  solid_interp.interpolate(phi_nm1_tmp.vec);
  solid_front_phi_nm1_.destroy();
  solid_front_phi_nm1_.set(phi_nm1_tmp.vec);

  vec_and_ptr_t seed_tmp(solid_front_phi_.vec);
  solid_interp.set_input(solid_seed_.vec, linear);
  solid_interp.interpolate(seed_tmp.vec);
  solid_seed_.destroy();
  solid_seed_.set(seed_tmp.vec);

  vec_and_ptr_t smoothed_nodes_tmp(solid_front_phi_.vec);
  solid_interp.set_input(solid_smoothed_nodes_.vec, linear);
  solid_interp.interpolate(smoothed_nodes_tmp.vec);
  solid_smoothed_nodes_.destroy();
  solid_smoothed_nodes_.set(smoothed_nodes_tmp.vec);

  vec_and_ptr_t time_tmp(solid_front_phi_.vec);
  solid_interp.set_input(solid_time_.vec, linear);
  solid_interp.interpolate(time_tmp.vec);
  solid_time_.destroy();
  solid_time_.set(time_tmp.vec);

  p4est_destroy(solid_p4est_);       solid_p4est_ = solid_p4est_np1;
  p4est_ghost_destroy(solid_ghost_); solid_ghost_ = solid_ghost_np1;
  p4est_nodes_destroy(solid_nodes_); solid_nodes_ = solid_nodes_np1;
  solid_hierarchy_->update(solid_p4est_, solid_ghost_);
  solid_ngbd_->update(solid_hierarchy_, solid_nodes_);
  PetscPrintf(p4est_->mpicomm, "done!\n");
  ierr = PetscLogEventEnd(log_my_p4est_multialloy_update_grid_solid, 0, 0, 0, 0); CHKERRXX(ierr);
}

int my_p4est_multialloy_t::one_step(int it_scheme, double *bc_error_max, double *bc_error_avg, std::vector<int> *num_pdes, std::vector<double> *bc_error_max_all, std::vector<double> *bc_error_avg_all)
{
  ierr = PetscLogEventBegin(log_my_p4est_multialloy_one_step, 0, 0, 0, 0); CHKERRXX(ierr);
  PetscPrintf(p4est_->mpicomm, "Solving nonlinear system:\n");
  time_ += dt_[0];

  int num_nodes = nodes_->num_owned_indeps;
  MPI_Allreduce(MPI_IN_PLACE,&num_nodes,1,MPI_INT,MPI_SUM, p4est_->mpicomm);

  PetscPrintf(p4est_->mpicomm, "\n Time = %3e, Number of Nodes = %d "
                               "\n -------------------------- \n", time_,num_nodes);

  PetscPrintf(p4est_->mpicomm, "dxyz_close_to_interface %3e \n \n", dxyz_close_interface_);
  // update time in interface and boundary conditions
//  gibbs_thomson_->t = time_;
  vol_heat_gen_ ->t = time_;
  contr_bc_value_temp_->t = time_;
  wall_bc_value_temp_ ->t = time_;
  for (int i = 0; i < num_comps_; ++i)
  {
    contr_bc_value_conc_[i]->t = time_;
    wall_bc_value_conc_ [i]->t = time_;
  }
  // compute right-hand sides
  vec_and_ptr_t       rhs_tl(tl_[0].vec);
  vec_and_ptr_t       rhs_ts(ts_[0].vec);
  vec_and_ptr_array_t rhs_cl(num_comps_, cl_[0].vec.data());

  rhs_tl.get_array();
  rhs_ts.get_array();
  rhs_cl.get_array();

  for (int i = 0; i < num_time_layers_; ++i)
  {
    tl_[i].get_array();
    ts_[i].get_array();
    cl_[i].get_array();
  }

  double xyz[P4EST_DIM];
  double heat_gen = 0;

  // get coefficients for time discretization
  std::vector<double> time_coeffs;

  variable_step_BDF_implicit(num_time_layers_-1, dt_, time_coeffs);


  foreach_node(n, nodes_)
  {
    if (vol_heat_gen_ != NULL)
    {
      node_xyz_fr_n(n, p4est_, nodes_, xyz);
      heat_gen = vol_heat_gen_->value(xyz);
    }

    rhs_tl.ptr[n] = 0;
    rhs_ts.ptr[n] = 0;

    for (int j = 0; j < num_comps_; ++j)
    {
      rhs_cl.ptr[j][n] = 0;
    }

    for (int i = 1; i < num_time_layers_; ++i)
    {
      rhs_tl.ptr[n] -= time_coeffs[i]*tl_[i].ptr[n];
      rhs_ts.ptr[n] -= time_coeffs[i]*ts_[i].ptr[n];

      for (int j = 0; j < num_comps_; ++j)
      {
        rhs_cl.ptr[j][n] -= time_coeffs[i]*cl_[i].ptr[j][n];
      }
    }

    rhs_tl.ptr[n] = rhs_tl.ptr[n]*density_l_*heat_capacity_l_/dt_[0] + heat_gen;
    rhs_ts.ptr[n] = rhs_ts.ptr[n]*density_s_*heat_capacity_s_/dt_[0] + heat_gen;

    for (int i = 0; i < num_comps_; ++i)
    {
      rhs_cl.ptr[i][n] = rhs_cl.ptr[i][n]/dt_[0];
    }
  }
  rhs_tl.restore_array();
  rhs_ts.restore_array();
  rhs_cl.restore_array();

  for (int i = 0; i < num_time_layers_; ++i)
  {
    tl_[i].restore_array();
    ts_[i].restore_array();
    cl_[i].restore_array();
  }
  //std::cout<< "Diagonal data conc:: " << time_coeffs[0]/dt_[0] <<"\n";
  vector<double> conc_diag(num_comps_, time_coeffs[0]/dt_[0]);

  // solve coupled system of equations
  my_p4est_poisson_nodes_multialloy_t solver_all_in_one(ngbd_, num_comps_);
  solver_all_in_one.set_iteration_scheme(it_scheme);

  solver_all_in_one.set_front(front_phi_.vec, front_phi_dd_.vec, front_normal_.vec, front_curvature_.vec);

  solver_all_in_one.set_composition_parameters(conc_diag.data(), solute_diff_.data());


  //std::cout<< "Diagonal data :: " << density_l_*heat_capacity_l_*time_coeffs[0]/dt_[0] <<"\n";
  //std::exit(EXIT_FAILURE);
  solver_all_in_one.set_thermal_parameters(latent_heat_,
                                           density_l_*heat_capacity_l_*time_coeffs[0]/dt_[0], thermal_cond_l_,
                                           density_s_*heat_capacity_s_*time_coeffs[0]/dt_[0], thermal_cond_s_);
  solver_all_in_one.set_density_parameters(density_l_,density_s_);
  solver_all_in_one.set_gibbs_thomson(zero_cf);
  solver_all_in_one.set_liquidus(liquidus_value_, liquidus_slope_, part_coeff_);
  solver_all_in_one.set_undercoolings(num_seeds_, seed_map_.vec, eps_v_.data(), eps_c_.data());

  solver_all_in_one.set_rhs(rhs_tl.vec, rhs_ts.vec, rhs_cl.vec.data());

  if (contr_phi_.vec != NULL)
  {
    solver_all_in_one.set_container(contr_phi_.vec, contr_phi_dd_.vec);
    solver_all_in_one.set_container_conditions_thermal(contr_bc_type_temp_, *contr_bc_value_temp_);
    solver_all_in_one.set_container_conditions_composition(contr_bc_type_conc_, contr_bc_value_conc_.data());
  }

  vector<CF_DIM *> zeros_cf(num_comps_, &zero_cf);

  solver_all_in_one.set_front_conditions(zero_cf, zero_cf, zeros_cf.data());

  solver_all_in_one.set_wall_conditions_thermal(wall_bc_type_temp_, *wall_bc_value_temp_);
  solver_all_in_one.set_wall_conditions_composition(wall_bc_type_conc_, wall_bc_value_conc_.data());

  solver_all_in_one.set_tolerance(bc_tolerance_, max_iterations_);
  solver_all_in_one.set_use_points_on_interface(use_points_on_interface_);
  solver_all_in_one.set_update_c0_robin(update_c0_robin_);
  solver_all_in_one.set_use_superconvergent_robin(use_superconvergent_robin_);

//  my_p4est_interpolation_nodes_t interp_c0_n(ngbd_);
//  interp_c0_n.set_input(cl_[1].vec[0], linear);
  solver_all_in_one.set_c0_guess(cl_[1].vec[0]);

//  bc_error_.destroy();
//  bc_error_.create(front_phi_.vec);

//  cl0_grad_.destroy();
//  cl0_grad_.create(p4est_, nodes_);

//  solver_all_in_one.set_verbose_mode(1);

  int one_step_iterations = solver_all_in_one.solve(tl_[0].vec, ts_[0].vec, cl_[0].vec.data(), cl0_grad_.vec, true,
      bc_error_.vec, bc_error_max, bc_error_avg,
      num_pdes, bc_error_max_all, bc_error_avg_all,
      psi_tl_.vec, psi_ts_.vec, psi_cl_.vec.data());
  rhs_tl.destroy();
  rhs_ts.destroy();
  rhs_cl.destroy();

  PetscPrintf(p4est_->mpicomm, "done!\n");

  // compute velocity
  compute_velocity();
  compute_solid();

  ierr = PetscLogEventEnd(log_my_p4est_multialloy_one_step, 0, 0, 0, 0); CHKERRXX(ierr);
  return one_step_iterations;
}


int my_p4est_multialloy_t::one_step_w_fluids(int it_scheme, double *bc_error_max, double *bc_error_avg, std::vector<int> *num_pdes, std::vector<double> *bc_error_max_all, std::vector<double> *bc_error_avg_all)
{
  ierr = PetscLogEventBegin(log_my_p4est_multialloy_one_step, 0, 0, 0, 0); CHKERRXX(ierr);

  time_ += dt_[0];

  int num_nodes = nodes_->num_owned_indeps;
  MPI_Allreduce(MPI_IN_PLACE,&num_nodes,1,MPI_INT,MPI_SUM, p4est_->mpicomm);

  PetscPrintf(p4est_->mpicomm, "\n ------------------------- \n Iteration = %d, Time = %3e, Number of Nodes = %d "
                               "\n -------------------------- \n", iteration_w_fluids, time_,num_nodes);
    PetscPrintf(p4est_->mpicomm, "Solving nonlinear system:\n");


  // update time in interface and boundary conditions
  //gibbs_thomson_->t = time_;
  vol_heat_gen_ ->t = time_;

  contr_bc_value_temp_->t = time_;
  wall_bc_value_temp_ ->t = time_;

  for (int i = 0; i < num_comps_; ++i)
  {
    contr_bc_value_conc_[i]->t = time_;
    wall_bc_value_conc_ [i]->t = time_;
  }

  // TO-DO MULTICOMP: // DID IT

  // Get the backtraced values for conc components and temperature from 
  // stefan with fluids solver
  
  // Create backtraced vectors:
  cl_backtrace_n.resize(num_comps_);
  cl_backtrace_nm1.resize(num_comps_);

  cl_backtrace_n.create(p4est_, nodes_);
  cl_backtrace_nm1.create(p4est_, nodes_);

  tl_backtrace_n.create(p4est_, nodes_);
  tl_backtrace_nm1.create(p4est_, nodes_);
  
  // Pass all relevant vectors/grid objects to stefan_w_fluids solver:
  // ---------------------------------
  // Concentrations:
  stefan_w_fluids_solver->set_Cl_n(cl_[0]);
  stefan_w_fluids_solver->set_Cl_nm1(cl_[1]);
  
  // Concentration backtraces:
  stefan_w_fluids_solver->set_Cl_backtrace_n(cl_backtrace_n);
  stefan_w_fluids_solver->set_Cl_backtrace_nm1(cl_backtrace_nm1);

  // Temperatures:
  stefan_w_fluids_solver->set_T_l_n(tl_[0]);
  stefan_w_fluids_solver->set_T_l_nm1(tl_[1]);

  // Temperature backtraces:
  stefan_w_fluids_solver->set_T_l_backtrace_n(tl_backtrace_n);
  stefan_w_fluids_solver->set_T_l_backtrace_nm1(tl_backtrace_nm1);

  // Set the phi (only relevant for visualization I think, but still let's just do it)
  stefan_w_fluids_solver->set_phi(front_phi_);

  // Relevant grid objects:
  stefan_w_fluids_solver->set_p4est_np1(p4est_);
  stefan_w_fluids_solver->set_nodes_np1(nodes_);
  stefan_w_fluids_solver->set_ngbd_np1(ngbd_);

  stefan_w_fluids_solver->set_p4est_n(p4est_nm1);
  stefan_w_fluids_solver->set_nodes_n(nodes_nm1);
  stefan_w_fluids_solver->set_ngbd_n(ngbd_nm1);

  stefan_w_fluids_solver->set_tstep(iteration_w_fluids);

   if(0){
     // -------------------------------
    // TEMPORARY: save fields before backtrace
    // -------------------------------
    std::vector<Vec_for_vtk_export_t> point_fields;
    std::vector<Vec_for_vtk_export_t> cell_fields = {};


    point_fields.push_back(Vec_for_vtk_export_t(cl_[0].vec[0], "cl_tn_0"));
    point_fields.push_back(Vec_for_vtk_export_t(cl_[0].vec[1], "cl_tn_1"));

    point_fields.push_back(Vec_for_vtk_export_t(tl_[0].vec, "tl_tn"));

    point_fields.push_back(Vec_for_vtk_export_t(cl_backtrace_n.vec[0], "cl_dn_0"));
    point_fields.push_back(Vec_for_vtk_export_t(cl_backtrace_n.vec[1], "cl_dn_1"));

    point_fields.push_back(Vec_for_vtk_export_t(tl_backtrace_n.vec, "tl_dn"));

    point_fields.push_back(Vec_for_vtk_export_t(v_n.vec[0], "vx_n"));
    point_fields.push_back(Vec_for_vtk_export_t(v_n.vec[1], "vy_n"));

    point_fields.push_back(Vec_for_vtk_export_t(cl_backtrace_nm1.vec[0], "cl_dnm1_0"));
    point_fields.push_back(Vec_for_vtk_export_t(cl_backtrace_nm1.vec[1], "cl_dnm1_1"));
    point_fields.push_back(Vec_for_vtk_export_t(tl_backtrace_nm1.vec, "tl_dnm1"));

    const char* out_dir = getenv("OUT_DIR");
    if(!out_dir){
      throw std::invalid_argument("You need to set the output directory for VTK: OUT_DIR_VTK");
    }

    char filename[1000];
    sprintf(filename, "%s/snapshot_before_backtrace_n_%d", out_dir, iteration_w_fluids);
    my_p4est_vtk_write_all_lists(p4est_, nodes_, ngbd_->get_ghost(), P4EST_TRUE, P4EST_TRUE, filename, point_fields, cell_fields);
    point_fields.clear();


    // nm1 fields now:
    point_fields.push_back(Vec_for_vtk_export_t(tl_[1].vec, "tl_tnm1_1"));
    point_fields.push_back(Vec_for_vtk_export_t(cl_[1].vec[0], "cl_tnm1_0"));
    point_fields.push_back(Vec_for_vtk_export_t(cl_[1].vec[1], "cl_tnm1_1"));

    point_fields.push_back(Vec_for_vtk_export_t(v_nm1.vec[0], "vx_nm1"));
    point_fields.push_back(Vec_for_vtk_export_t(v_nm1.vec[1], "vy_nm1"));

    char filename2[1000];
    sprintf(filename2, "%s/snapshot_before_backtrace_nm1_%d", out_dir, iteration_w_fluids);
    my_p4est_vtk_write_all_lists(p4est_nm1, nodes_nm1, ngbd_nm1->get_ghost(), P4EST_TRUE, P4EST_TRUE, filename2, point_fields, cell_fields);
    point_fields.clear();
  }

  stefan_w_fluids_solver->set_dt_nm1(dt_[1]);
  stefan_w_fluids_solver->set_dt(dt_[0]);

  stefan_w_fluids_solver->do_backtrace_for_scalar_temp_conc_problem(true, num_comps_, iteration_w_fluids);


  if(0){
    // -------------------------------
    // TEMPORARY: save fields after backtrace
    // -------------------------------
    std::vector<Vec_for_vtk_export_t> point_fields;
    std::vector<Vec_for_vtk_export_t> cell_fields = {};


    point_fields.push_back(Vec_for_vtk_export_t(cl_[0].vec[0], "cl_tn_0"));
    point_fields.push_back(Vec_for_vtk_export_t(cl_[0].vec[1], "cl_tn_1"));

    point_fields.push_back(Vec_for_vtk_export_t(tl_[0].vec, "tl_tn"));

    point_fields.push_back(Vec_for_vtk_export_t(cl_backtrace_n.vec[0], "cl_dn_0"));
    point_fields.push_back(Vec_for_vtk_export_t(cl_backtrace_n.vec[1], "cl_dn_1"));

    point_fields.push_back(Vec_for_vtk_export_t(tl_backtrace_n.vec, "tl_dn"));

    point_fields.push_back(Vec_for_vtk_export_t(v_n.vec[0], "vx_n"));
    point_fields.push_back(Vec_for_vtk_export_t(v_n.vec[1], "vy_n"));

    point_fields.push_back(Vec_for_vtk_export_t(cl_backtrace_nm1.vec[0], "cl_dnm1_0"));
    point_fields.push_back(Vec_for_vtk_export_t(cl_backtrace_nm1.vec[1], "cl_dnm1_1"));
    point_fields.push_back(Vec_for_vtk_export_t(tl_backtrace_nm1.vec, "tl_dnm1"));

    const char* out_dir = getenv("OUT_DIR");
    if(!out_dir){
      throw std::invalid_argument("You need to set the output directory for VTK: OUT_DIR_VTK");
    }

    char filename[1000];
    sprintf(filename, "%s/snapshot_after_backtrace_n_%d", out_dir, iteration_w_fluids);
    my_p4est_vtk_write_all_lists(p4est_, nodes_, ngbd_->get_ghost(), P4EST_TRUE, P4EST_TRUE, filename, point_fields, cell_fields);
    point_fields.clear();


    // nm1 fields now:
    point_fields.push_back(Vec_for_vtk_export_t(tl_[1].vec, "tl_tnm1_1"));
    point_fields.push_back(Vec_for_vtk_export_t(cl_[1].vec[0], "cl_tnm1_0"));
    point_fields.push_back(Vec_for_vtk_export_t(cl_[1].vec[1], "cl_tnm1_1"));

    point_fields.push_back(Vec_for_vtk_export_t(v_nm1.vec[0], "vx_nm1"));
    point_fields.push_back(Vec_for_vtk_export_t(v_nm1.vec[1], "vy_nm1"));

    char filename2[1000];
    sprintf(filename2, "%s/snapshot_after_backtrace_nm1_%d", out_dir, iteration_w_fluids);
    my_p4est_vtk_write_all_lists(p4est_nm1, nodes_nm1, ngbd_nm1->get_ghost(), P4EST_TRUE, P4EST_TRUE, filename2, point_fields, cell_fields);
    point_fields.clear();

  }

  // ----------------------

  // compute right-hand sides
  vec_and_ptr_t       rhs_tl(tl_[0].vec);
  vec_and_ptr_t       rhs_ts(ts_[0].vec);
  vec_and_ptr_array_t rhs_cl(num_comps_, cl_[0].vec.data());

  rhs_tl.get_array();
  rhs_ts.get_array();
  rhs_cl.get_array();

  for (int i = 0; i < num_time_layers_; ++i)
  {
//    tl_[i].get_array();
    ts_[i].get_array();
//    cl_[i].get_array();
  }

  // PASTE STARTING HERE 
  // Get the backtrace arrays:
  tl_backtrace_n.get_array();
  tl_backtrace_nm1.get_array();

  cl_backtrace_n.get_array();
  cl_backtrace_nm1.get_array();

  double xyz[P4EST_DIM];
  double heat_gen = 0;

  // get coefficients for time discretization
  std::vector<double> time_coeffs;
  // we will still use the usual BDF for the solid temperature
  variable_step_BDF_implicit(num_time_layers_-1, dt_, time_coeffs);


  // Compute the time coefficients associated with the Semi-Lagragian advective disc
  double SL_alpha = (2.*dt_[0] + dt_[1])/(dt_[0] + dt_[1]); // SL alpha coeff
  double SL_beta = (-1.*dt_[0])/(dt_[0] + dt_[1]); // SL beta coefff

  foreach_node(n, nodes_)
  {
    if (vol_heat_gen_ != NULL)
    {
      node_xyz_fr_n(n, p4est_, nodes_, xyz);
      heat_gen = vol_heat_gen_->value(xyz);
    }

    // Build the RHS's
    rhs_ts.ptr[n] = 0;
    rhs_tl.ptr[n] = 0;
    for (int j = 0; j < num_comps_; ++j)
    {
      rhs_cl.ptr[j][n] = 0;
    }

    // Daniils usual treatment for the solid temp:
    for (int i = 1; i < num_time_layers_; ++i){
      rhs_ts.ptr[n] -= time_coeffs[i]*ts_[i].ptr[n];
    }


    // Use SL disc for fluid temp:
    rhs_tl.ptr[n] = tl_backtrace_n.ptr[n]*((SL_alpha/dt_[0]) - (SL_beta/dt_[1])) +
                    tl_backtrace_nm1.ptr[n]*(SL_beta/dt_[1]);


//    printf("Tl_backtrace n = %0.2f, Tl_backtrace_nm1 = %0.2f \n", tl_backtrace_n.ptr[n], tl_backtrace_nm1.ptr[n]);


    // Use SL disc for fluid concentrations:
    for(int j = 0; j<num_comps_; ++j){
      rhs_cl.ptr[j][n] = cl_backtrace_n.ptr[j][n]*((SL_alpha/dt_[0]) - (SL_beta/dt_[1])) +
                    cl_backtrace_nm1.ptr[j][n]*(SL_beta/dt_[1]);

//      printf("rhs cl %d : %0.3e \n", j, rhs_cl.ptr[j][n]);
    }


//    printf("rhs cl %d : %0.3e, rhs tl : %0.3e \n", 0, rhs_cl.ptr[0][n], rhs_tl.ptr[n]);


    // Multiply by relevant quantities:
    // TO-DO MULTICOMP: will change this later to reflect nondim setup
    rhs_tl.ptr[n] = rhs_tl.ptr[n]*density_l_*heat_capacity_l_ + heat_gen;
    rhs_ts.ptr[n] = rhs_ts.ptr[n]*density_s_*heat_capacity_s_/dt_[0] + heat_gen;
    //std::cout<< rhs_tl.ptr[n]<<"\n";
    //std::cout << "conc rhs"<< rhs_cl.ptr[0][n]<<"\n";

//    PetscPrintf(p4est_->mpicomm, "rhs_ts[n] = %0.3e, heat_gen = %0.3e \n", rhs_ts.ptr[n], heat_gen);
//    PetscPrintf(p4est_->mpicomm, "SL_alpha = %0.2f, SL_beta = %0.2f \n", SL_alpha, SL_beta);

//    PetscPrintf(p4est_->mpicomm, "rhs_tl[n] = %0.3e, heat_gen = %0.3e \n", rhs_tl.ptr[n], heat_gen);

  }

//  PetscPrintf(p4est_->mpicomm, "RHS_TS: \n -------------------------------- \n");
//  VecView(rhs_ts.vec, PETSC_VIEWER_STDOUT_WORLD);

//  PetscPrintf(p4est_->mpicomm, "\n\nRHS_TL: \n -------------------------------- \n");
//      VecView(rhs_tl.vec, PETSC_VIEWER_STDOUT_WORLD);
    
  // Restore arrays:
  rhs_tl.restore_array();
  rhs_ts.restore_array();
  rhs_cl.restore_array();
  for (int i = 0; i < num_time_layers_; ++i)
  {
//    tl_[i].restore_array();
    ts_[i].restore_array();
//    cl_[i].restore_array();
  }

  //std::cout<< "Diagonal data :: " << SL_alpha/dt_[0] <<"\n";

  // MULTICOMP ALERT: we have changed the diag below
  //std::exit(EXIT_FAILURE);
  vector<double> conc_diag(num_comps_, SL_alpha/dt_[0]);

  // solve coupled system of equations
  my_p4est_poisson_nodes_multialloy_t solver_all_in_one(ngbd_, num_comps_);
  solver_all_in_one.set_iteration_scheme(it_scheme);

  solver_all_in_one.set_front(front_phi_.vec, front_phi_dd_.vec, front_normal_.vec, front_curvature_.vec);

  solver_all_in_one.set_composition_parameters(conc_diag.data(), solute_diff_.data());
  // MULTICOMP ALERT: we have changed the diag below
  //std::cout<< "Diagonal data :: " << density_l_*heat_capacity_l_*SL_alpha/dt_[0] <<"\n";
  solver_all_in_one.set_thermal_parameters(latent_heat_,
                                           density_l_*heat_capacity_l_*SL_alpha/dt_[0], thermal_cond_l_,
                                           density_s_*heat_capacity_s_*time_coeffs[0]/dt_[0], thermal_cond_s_);
  solver_all_in_one.set_density_parameters(density_l_,density_s_);
  solver_all_in_one.set_gibbs_thomson(zero_cf);
  solver_all_in_one.set_liquidus(liquidus_value_, liquidus_slope_, part_coeff_);
  solver_all_in_one.set_undercoolings(num_seeds_, seed_map_.vec, eps_v_.data(), eps_c_.data());

  solver_all_in_one.set_rhs(rhs_tl.vec, rhs_ts.vec, rhs_cl.vec.data());

  if (contr_phi_.vec != NULL)
  {
    solver_all_in_one.set_container(contr_phi_.vec, contr_phi_dd_.vec);
    solver_all_in_one.set_container_conditions_thermal(contr_bc_type_temp_, *contr_bc_value_temp_);
    solver_all_in_one.set_container_conditions_composition(contr_bc_type_conc_, contr_bc_value_conc_.data());
  }

  vector<CF_DIM *> zeros_cf(num_comps_, &zero_cf);

  solver_all_in_one.set_front_conditions(zero_cf, zero_cf, zeros_cf.data());

  solver_all_in_one.set_wall_conditions_thermal(wall_bc_type_temp_, *wall_bc_value_temp_);
  solver_all_in_one.set_wall_conditions_composition(wall_bc_type_conc_, wall_bc_value_conc_.data());

  solver_all_in_one.set_tolerance(bc_tolerance_, max_iterations_);
  solver_all_in_one.set_use_points_on_interface(use_points_on_interface_);
  solver_all_in_one.set_update_c0_robin(update_c0_robin_);
  solver_all_in_one.set_use_superconvergent_robin(use_superconvergent_robin_);

  solver_all_in_one.set_c0_guess(cl_[1].vec[0]);

  int one_step_iterations = solver_all_in_one.solve(tl_[0].vec, ts_[0].vec, cl_[0].vec.data(), cl0_grad_.vec, true,
      bc_error_.vec, bc_error_max, bc_error_avg,
      num_pdes, bc_error_max_all, bc_error_avg_all,
      psi_tl_.vec, psi_ts_.vec, psi_cl_.vec.data());


  rhs_tl.destroy();
  rhs_ts.destroy();
  rhs_cl.destroy();

  // destroy backtrace vectors since they are no longer needed:
  tl_backtrace_n.destroy();
  tl_backtrace_nm1.destroy();

  cl_backtrace_n.destroy();
  cl_backtrace_nm1.destroy();

  PetscPrintf(p4est_->mpicomm, "done!\n");

  // compute velocity
  compute_velocity();

  PetscPrintf(p4est_->mpicomm, "Forcing front velo to zero ... \n");
  foreach_dimension(d){
    ierr= VecScaleGhost(front_velo_[0].vec[d], 0.);
  }



 // TO-DO MULTICOMP:
  // Provide the computed interfacial velocity to the stefan_w_fluids solver
  // to use as a boundary condition
  // Solve the Navier-Stokes problem using stefan_w_fluids solver


  // (ASIDE) make sure the bc is properly setup in stefan w fluids so it actually
  // gets passed along 

  // (1) pass the computed interfacial velocity to stefan w fluids
  // Convert to a (vgamma,n) n to make it a vector still along the normal direction
//  std::cout<<"step_w_fluids line 2384\n";
  vec_and_ptr_dim_t vgamma_n_vec;
  vgamma_n_vec.create(p4est_, nodes_);

  front_normal_.get_array();
  vgamma_n_vec.get_array();
  foreach_dimension(dim){
    VecCopyGhost(front_normal_.vec[dim], vgamma_n_vec.vec[dim]);
    // VecScaleGhost(vgamma_n_vec.vec[dim], front_velo_norm_[0]);
    foreach_node(n, nodes_){
      vgamma_n_vec.ptr[dim][n]*=front_velo_norm_[0].ptr[n];
    }
  }
//  std::cout<<"step_w_fluids line 2400\n";
  front_normal_.restore_array();
  vgamma_n_vec.restore_array();
  stefan_w_fluids_solver->set_v_interface(vgamma_n_vec);
//  std::cout<<"step_w_fluids line 2404\n";

  // (2) provide stefan_w_fluids w all other relevant things to solve the ns problem:
  // Things we will need: 
  // - dt, dtnm1 
  // -
  // - 
  // -
  // TO-DO: put these below in initialize (I think)
  // - boundary conditions (these can just be passed by multialloy once) (think it is sufficient to do this in our initialize_for_fluids() fxn)
  // - any boussinesq options -- NEED TO PROVIDE THE BOUSSINESQ TERM
  // - problem dimensionalization type (did this in initialize)
  // - fluid velocity vectors, vorticity vector, pressure vector
  
  // (3) solve the NS 
  vec_and_ptr_t boussinesq_terms_rhs_for_ns;
  /*
  boussinesq_terms_rhs_for_ns.create(p4est_, nodes_);
  boussinesq_terms_rhs_for_ns.get_array();
  tl_[0].get_array();
  cl_[0].get_array();

  foreach_node(n, nodes_){

  }
  boussinesq_terms_rhs_for_ns.restore_array();
  tl_[0].restore_array();
  cl_[0].get_array();


  stefan_w_fluids_solver->setup_and_solve_navier_stokes_problem(true, boussinesq_terms_rhs_for_ns.vec);

  boussinesq_terms_rhs_for_ns.destroy(); // move this somewhere more appropriate later

*/
  PetscPrintf(p4est_->mpicomm, "RED ALERT: Boussinesq is currently non-operational. We will want to fix this later \n");



  stefan_w_fluids_solver->setup_and_solve_navier_stokes_problem(false, nullptr, true);

  // Now, get the velocity results back out of SWF (or do we need to? ) :


  // Slide and get things out of stefan with fluids:
  // the old v_nm1 has already been destroyed by stefan with fluids

  if(0){
    printf(" \n \n \n saving fields after ns solution \n");
    // -------------------------------
    // TEMPORARY: save fields after grid update
    // -------------------------------
    std::vector<Vec_for_vtk_export_t> point_fields;
    std::vector<Vec_for_vtk_export_t> cell_fields = {};
    point_fields.push_back(Vec_for_vtk_export_t(v_n.vec[0], "vx_n"));
    point_fields.push_back(Vec_for_vtk_export_t(v_n.vec[1], "vy_n"));

    const char* out_dir = getenv("OUT_DIR");
    if(!out_dir){
      throw std::invalid_argument("You need to set the output directory for VTK: OUT_DIR_VTK");
    }

    char filename[1000];
    sprintf(filename, "%s/snapshot_after_ns_solution_before_slide_n", out_dir);
    my_p4est_vtk_write_all_lists(p4est_, nodes_, ngbd_->get_ghost(), P4EST_TRUE, P4EST_TRUE, filename, point_fields, cell_fields);
    point_fields.clear();


    point_fields.push_back(Vec_for_vtk_export_t(v_nm1.vec[0], "vx_nm1"));
    point_fields.push_back(Vec_for_vtk_export_t(v_nm1.vec[1], "vy_nm1"));

    char filename2[1000];
    sprintf(filename2, "%s/snapshot_after_ns_solution_before_slide_nm1", out_dir);
    my_p4est_vtk_write_all_lists(p4est_nm1, nodes_nm1, ngbd_nm1->get_ghost(), P4EST_TRUE, P4EST_TRUE, filename2, point_fields, cell_fields);
    point_fields.clear();
  }

  v_n = stefan_w_fluids_solver->get_v_n();
  v_nm1 = stefan_w_fluids_solver->get_v_nm1();

  if(0){
    printf(" \n \n \n saving fields after grid update \n");
    // -------------------------------
    // TEMPORARY: save fields after grid update
    // -------------------------------
    std::vector<Vec_for_vtk_export_t> point_fields;
    std::vector<Vec_for_vtk_export_t> cell_fields = {};
    point_fields.push_back(Vec_for_vtk_export_t(v_n.vec[0], "vx_n"));
    point_fields.push_back(Vec_for_vtk_export_t(v_n.vec[1], "vy_n"));
    point_fields.push_back(Vec_for_vtk_export_t(v_nm1.vec[0], "vx_nm1"));
    point_fields.push_back(Vec_for_vtk_export_t(v_nm1.vec[1], "vy_nm1"));


    const char* out_dir = getenv("OUT_DIR");
    if(!out_dir){
      throw std::invalid_argument("You need to set the output directory for VTK: OUT_DIR_VTK");
    }

    char filename[1000];
    sprintf(filename, "%s/snapshot_after_ns_solution", out_dir);
    my_p4est_vtk_write_all_lists(p4est_, nodes_, ngbd_->get_ghost(), P4EST_TRUE, P4EST_TRUE, filename, point_fields, cell_fields);
    point_fields.clear();
  }

  // TO-DO:
  // (1) need to handle scaling the fluid velocities back and forth between dim and nondim for
  // the NS solution

  // (2) need to actually get the vn and vnm1 result out of the navier stokes

  // -- note: want to handle scaling from dim to nondim here, not inside SWF
  // --> to be consistent, we want to modify the backtrace appropach too

  // (4) get out the velocities, pressure, vort, for visualization
  press_nodes = stefan_w_fluids_solver->get_press_nodes();
  vorticity = stefan_w_fluids_solver->get_vorticity();

  compute_solid();

  // Increment the iteration count:
  iteration_w_fluids++;

  ierr = PetscLogEventEnd(log_my_p4est_multialloy_one_step, 0, 0, 0, 0); CHKERRXX(ierr);
  return one_step_iterations;
} // END OF ONE STEP W FLUIDS


void my_p4est_multialloy_t::save_VTK(int iter)
{
  ierr = PetscLogEventBegin(log_my_p4est_multialloy_save_vtk, 0, 0, 0, 0); CHKERRXX(ierr);
  //std:: cout<<"mas line 2017 \n";
  if(solve_with_fluids)
  {
    ierr = PetscPrintf(p4est_->mpicomm, " solve_w_fluids \n");
  }
  const char* out_dir = getenv("OUT_DIR");
  if (!out_dir)
  {
    ierr = PetscPrintf(p4est_->mpicomm, "You need to set the environment variable OUT_DIR to save visuals\n");
    return;
  }
  //std:: cout<<"mas line 2024 \n";
  splitting_criteria_t *data = (splitting_criteria_t*)p4est_->user_pointer;

  char name[1000];
  sprintf(name, "%s/vtu/multialloy_lvl_%d_%d.%05d", out_dir, data->min_lvl, data->max_lvl, iter);

  /*
  // cell data
  std::vector<const double *>    cell_data;
  std::vector<std::string> cell_data_names;
  */
  // new format for saving to vtk
  //std:: cout<<"mas line 2036 \n";
  std::vector<Vec_for_vtk_export_t> cell_fields = {};
  /* save the size of the leaves */
  Vec leaf_level;
  ierr = VecCreateGhostCells(p4est_, ghost_, &leaf_level); CHKERRXX(ierr);
  double *l_p;
  ierr = VecGetArray(leaf_level, &l_p); CHKERRXX(ierr);

  for(p4est_topidx_t tree_idx = p4est_->first_local_tree; tree_idx <= p4est_->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est_->trees, tree_idx);
    for( size_t q=0; q<tree->quadrants.elem_count; ++q)
    {
      const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
      l_p[tree->quadrants_offset+q] = quad->level;
    }
  }
  //std:: cout<<"mas line 2053 \n";
  for(size_t q=0; q<ghost_->ghosts.elem_count; ++q)
  {
    const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&ghost_->ghosts, q);
    l_p[p4est_->local_num_quadrants+q] = quad->level;
  }
  //std:: cout<<"mas line 2059 \n";
  //cell_data.push_back(l_p); cell_data_names.push_back("leaf_level");
  ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
  cell_fields.push_back(Vec_for_vtk_export_t(leaf_level, "leaf_level"));
  // point data
  /*std::vector<const double *>    point_data;
  std::vector<std::string> point_data_names;*/

  std::vector<Vec_for_vtk_export_t> point_fields;

  //front_phi_.get_array();
  //point_data.push_back(front_phi_.ptr); point_data_names.push_back("phi");
  point_fields.push_back(Vec_for_vtk_export_t(front_phi_.vec, "phi" ));
  if (contr_phi_.vec != NULL)
  {
    //contr_phi_.get_array(); // point_data.push_back(contr_phi_.ptr); point_data_names.push_back("contr");
    point_fields.push_back(Vec_for_vtk_export_t(contr_phi_.vec, "contr" ));
  }
  //std:: cout<<"mas line 2077 \n";
  //tl_[0].get_array(); //point_data.push_back(tl_[0].ptr); point_data_names.push_back("tl");
  point_fields.push_back(Vec_for_vtk_export_t(tl_[0].vec, "tl" ));
  //ts_[0].get_array(); //point_data.push_back(ts_[0].ptr); point_data_names.push_back("ts");
  point_fields.push_back(Vec_for_vtk_export_t(ts_[0].vec, "ts" ));
  //cl_[0].get_array();
  //std:: cout<<"mas line 2083 \n";
  for (int i = 0; i < num_comps_; ++i)
  {
    char numstr[21];
    sprintf(numstr, "%d", i);
    std::string name("cl");
    //point_data.push_back(cl_[0].ptr[i]); point_data_names.push_back(name + numstr);
    point_fields.push_back(Vec_for_vtk_export_t(cl_[0].vec[i], name+numstr));
  }
  //std:: cout<<"mas line 2092 \n";
  //front_velo_norm_[0]      .get_array(); //point_data.push_back(front_velo_norm_[0].ptr);       point_data_names.push_back("vn");
  point_fields.push_back(Vec_for_vtk_export_t(front_velo_norm_[0].vec, "vn" ));
  //std:: cout<<"mas line 2095 \n";
  //front_curvature_         .get_array(); //point_data.push_back(front_curvature_.ptr);          point_data_names.push_back("kappa");
  //std:: cout<<"mas line 2097 \n";
  point_fields.push_back(Vec_for_vtk_export_t(front_curvature_.vec, "kappa" ));
  //bc_error_                .get_array(); //point_data.push_back(bc_error_.ptr);                 point_data_names.push_back("bc_error");
  //std:: cout<<"mas line 2100 \n";
  point_fields.push_back(Vec_for_vtk_export_t(bc_error_.vec, "bc_error" ));
  //dendrite_number_         .get_array(); //point_data.push_back(dendrite_number_.ptr);          point_data_names.push_back("dendrite_number");
  //std:: cout<<"mas line 2103 \n";
  point_fields.push_back(Vec_for_vtk_export_t(dendrite_number_.vec, "dendrite_number" ));
  //dendrite_tip_            .get_array(); //point_data.push_back(dendrite_tip_.ptr);             point_data_names.push_back("dendrite_tip");
  //std:: cout<<"mas line 2106 \n";
  point_fields.push_back(Vec_for_vtk_export_t(dendrite_tip_.vec, "dendrite_tip" ));
  //seed_map_                .get_array(); //point_data.push_back(seed_map_.ptr);                 point_data_names.push_back("seed_num");
  //std:: cout<<"mas line 2109 \n";
  point_fields.push_back(Vec_for_vtk_export_t(seed_map_.vec, "seed_num" ));
  //smoothed_nodes_          .get_array(); //point_data.push_back(smoothed_nodes_.ptr);           point_data_names.push_back("smoothed_nodes");
  //std:: cout<<"mas line 2112 \n";
  point_fields.push_back(Vec_for_vtk_export_t(smoothed_nodes_.vec, "smoothed_nodes" ));
  //std:: cout<<"mas line 2114 \n";
  //front_phi_unsmooth_      .get_array(); //point_data.push_back(front_phi_unsmooth_.ptr);       point_data_names.push_back("phi_unsmooth");
  point_fields.push_back(Vec_for_vtk_export_t(front_phi_unsmooth_.vec, "phi_unsmooth" ));
  //std:: cout<<"mas line 2117\n";
  // if solving with fluids , output fluid velocity
  if (solve_with_fluids){
//        std:: cout<<"we are here \n";
        point_fields.push_back(Vec_for_vtk_export_t(v_n.vec[0], "u"));
        //std:: cout<<"mas line 2122\n";
        point_fields.push_back(Vec_for_vtk_export_t(v_n.vec[1], "v"));
        //std:: cout<<"mas line 2124\n";
        point_fields.push_back(Vec_for_vtk_export_t(vorticity.vec, "vorticity"));
        //std:: cout<<"mas line 2126\n";
        point_fields.push_back(Vec_for_vtk_export_t(press_nodes.vec, "pressure"));
  }
  VecScaleGhost(front_velo_norm_[0].vec, 1./scaling_);
  //std:: cout<<"mas line 2130\n";
  my_p4est_vtk_write_all_lists(p4est_, nodes_, ghost_,
                               P4EST_TRUE, P4EST_TRUE,
                               name,
                               point_fields,
                               cell_fields);
  //std:: cout<<"mas line 2136\n";
  VecScaleGhost(front_velo_norm_[0].vec, scaling_);

  //ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
  //ierr = VecDestroy(leaf_level); CHKERRXX(ierr);
  /*
  front_phi_.restore_array();

  if (contr_phi_.vec != NULL)
  {
    contr_phi_.restore_array();
  }

  tl_[0].restore_array();
  ts_[0].restore_array();
  cl_[0].restore_array();

  front_velo_norm_[0]      .restore_array();
  front_curvature_         .restore_array();
  bc_error_                .restore_array();
  dendrite_number_         .restore_array();
  dendrite_tip_            .restore_array();
  seed_map_                .restore_array();
  smoothed_nodes_          .restore_array();
  front_phi_unsmooth_      .restore_array();
  */
  PetscPrintf(p4est_->mpicomm, "VTK saved in %s\n", name);
  ierr = PetscLogEventEnd(log_my_p4est_multialloy_save_vtk, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_multialloy_t::save_VTK_solid(int iter)
{
  ierr = PetscLogEventBegin(log_my_p4est_multialloy_save_vtk, 0, 0, 0, 0); CHKERRXX(ierr);

  const char* out_dir = getenv("OUT_DIR");
  if (!out_dir)
  {
    ierr = PetscPrintf(solid_p4est_->mpicomm, "You need to set the environment variable OUT_DIR to save visuals\n");
    return;
  }

  splitting_criteria_t *data = (splitting_criteria_t*)solid_p4est_->user_pointer;

  char name[1000];
  sprintf(name, "%s/vtu/multialloy_solid_lvl_%d_%d.%05d", out_dir, data->min_lvl, data->max_lvl, iter);

  std::vector<Vec_for_vtk_export_t> cell_fields = {};
  Vec leaf_level;
  ierr = VecCreateGhostCells(solid_p4est_, solid_ghost_, &leaf_level); CHKERRXX(ierr);
  double *l_p;
  ierr = VecGetArray(leaf_level, &l_p); CHKERRXX(ierr);

  for(p4est_topidx_t tree_idx = solid_p4est_->first_local_tree; tree_idx <= solid_p4est_->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(solid_p4est_->trees, tree_idx);
    for( size_t q=0; q<tree->quadrants.elem_count; ++q)
    {
      const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
      l_p[tree->quadrants_offset+q] = quad->level;
    }
  }

  for(size_t q=0; q<solid_ghost_->ghosts.elem_count; ++q)
  {
    const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&solid_ghost_->ghosts, q);
    l_p[solid_p4est_->local_num_quadrants+q] = quad->level;
  }

  ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
  cell_fields.push_back(Vec_for_vtk_export_t(leaf_level, "leaf_level"));

  std::vector<Vec_for_vtk_export_t> point_fields;
  vec_and_ptr_t solid_contr_phi(solid_front_phi_.vec);
  my_p4est_interpolation_nodes_t interp(ngbd_);
  interp.add_all_nodes(solid_p4est_, solid_nodes_);
  interp.set_input(contr_phi_.vec, linear);
  interp.interpolate(solid_contr_phi.vec);

  point_fields.push_back(Vec_for_vtk_export_t(solid_contr_phi.vec, "contr"));
  point_fields.push_back(Vec_for_vtk_export_t(solid_front_phi_.vec, "phi"));

  point_fields.push_back(Vec_for_vtk_export_t(solid_front_curvature_.vec, "kappa"));

  point_fields.push_back(Vec_for_vtk_export_t(solid_front_velo_norm_.vec, "vn"));
  point_fields.push_back(Vec_for_vtk_export_t(solid_seed_.vec, "seed"));
  point_fields.push_back(Vec_for_vtk_export_t(solid_smoothed_nodes_.vec, "smoothed_nodes"));
  point_fields.push_back(Vec_for_vtk_export_t(solid_tf_.vec, "tf"));
  point_fields.push_back(Vec_for_vtk_export_t(solid_time_.vec, "time"));


  for (int i = 0; i < num_comps_; ++i)
  {
    char numstr[21];
    sprintf(numstr, "%d", i);
    std::string name("cl");
    point_fields.push_back(Vec_for_vtk_export_t(solid_cl_.vec[i], name + numstr));
  }
  for (int i = 0; i < num_comps_; ++i)
  {
    char numstr[21];
    sprintf(numstr, "%d", i);
    std::string name("kp");
    point_fields.push_back(Vec_for_vtk_export_t(solid_part_coeff_.vec[i], name + numstr));
  }

  VecScaleGhost(solid_front_velo_norm_.vec, 1./scaling_);

  my_p4est_vtk_write_all_lists(solid_p4est_, solid_nodes_, solid_ghost_,
                               P4EST_TRUE, P4EST_TRUE,
                               name,
                               point_fields, cell_fields);


  VecScaleGhost(solid_front_velo_norm_.vec, scaling_);

  //solid_contr_phi.destroy();

//  PetscPrintf(solid_p4est_->mpicomm, "Line 2207-> VTK saved in %s\n", name);

  ierr = PetscLogEventEnd(log_my_p4est_multialloy_save_vtk, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_multialloy_t::save_p4est(int iter)
{
//  ierr = PetscLogEventBegin(log_my_p4est_multialloy_save_vtk, 0, 0, 0, 0); CHKERRXX(ierr);
  const char* out_dir = getenv("OUT_DIR");
  if (!out_dir)
  {
    ierr = PetscPrintf(p4est_->mpicomm, "You need to set the environment variable OUT_DIR to save visuals\n");
    return;
  }

  splitting_criteria_t *data = (splitting_criteria_t*)p4est_->user_pointer;

  char name_grid[1000]; sprintf(name_grid, "%s/p4est/grid_lvl_%d_%d.%05d", out_dir, data->min_lvl, data->max_lvl, iter);
  char name_vecs[1000]; sprintf(name_vecs, "%s/p4est/vecs_lvl_%d_%d.%05d.petscbin", out_dir, data->min_lvl, data->max_lvl, iter);

  my_p4est_save_forest(name_grid, p4est_, nodes_, NULL);

  std::vector<Vec> vecs_to_save;
  vecs_to_save.push_back(front_phi_.vec);

  if (contr_phi_.vec != NULL) {
    vecs_to_save.push_back(contr_phi_.vec);
  }

  vecs_to_save.push_back(tl_[0].vec);
  vecs_to_save.push_back(ts_[0].vec);

  for (int i = 0; i < num_comps_; ++i) {
    vecs_to_save.push_back(cl_[0].vec[i]);
  }

  vecs_to_save.push_back(front_velo_norm_[0].vec);
  vecs_to_save.push_back(front_curvature_.vec);
  vecs_to_save.push_back(bc_error_.vec);
  vecs_to_save.push_back(seed_map_.vec);

  VecDump(name_vecs, vecs_to_save.size(), vecs_to_save.data());
//  ierr = PetscLogEventEnd(log_my_p4est_multialloy_save_vtk, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_multialloy_t::save_p4est_solid(int iter)
{
//  ierr = PetscLogEventBegin(log_my_p4est_multialloy_save_vtk, 0, 0, 0, 0); CHKERRXX(ierr);
  const char* out_dir = getenv("OUT_DIR");
  if (!out_dir)
  {
    ierr = PetscPrintf(p4est_->mpicomm, "You need to set the environment variable OUT_DIR to save visuals\n");
    return;
  }

  splitting_criteria_t *data = (splitting_criteria_t*)p4est_->user_pointer;

  char name_grid[1000]; sprintf(name_grid, "%s/p4est/grid_solid_lvl_%d_%d.%05d", out_dir, data->min_lvl, data->max_lvl, iter);
  char name_vecs[1000]; sprintf(name_vecs, "%s/p4est/vecs_solid_lvl_%d_%d.%05d.petscbin", out_dir, data->min_lvl, data->max_lvl, iter);

  my_p4est_save_forest(name_grid, solid_p4est_, solid_nodes_, NULL);

  std::vector<Vec> vecs_to_save;

  vec_and_ptr_t solid_contr_phi(solid_front_phi_.vec);
  my_p4est_interpolation_nodes_t interp(ngbd_);
  interp.add_all_nodes(solid_p4est_, solid_nodes_);
  interp.set_input(contr_phi_.vec, linear);
  interp.interpolate(solid_contr_phi.vec);

  vecs_to_save.push_back(solid_contr_phi.vec);
  vecs_to_save.push_back(solid_front_phi_.vec);
  vecs_to_save.push_back(solid_front_curvature_.vec);
  vecs_to_save.push_back(solid_front_velo_norm_.vec);
  vecs_to_save.push_back(solid_seed_.vec);
  vecs_to_save.push_back(solid_smoothed_nodes_.vec);
  vecs_to_save.push_back(solid_tf_.vec);
  vecs_to_save.push_back(solid_time_.vec);

  for (int i = 0; i < num_comps_; ++i) {
    vecs_to_save.push_back(solid_cl_.vec[i]);
    vecs_to_save.push_back(solid_part_coeff_.vec[i]);
  }

  VecScaleGhost(solid_front_velo_norm_.vec, 1./scaling_);

  VecDump(name_vecs, vecs_to_save.size(), vecs_to_save.data());

  VecScaleGhost(solid_front_velo_norm_.vec, scaling_);

//  ierr = PetscLogEventEnd(log_my_p4est_multialloy_save_vtk, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_multialloy_t::count_dendrites(int iter)
{
  // this code assumes dendrites grow in the positive y-direction

  // find boundaries of the "mushy zone"
  double mushy_zone_min = brick_.xyz_max[1];
  double mushy_zone_max = brick_.xyz_min[1];

  front_phi_.get_array();
  foreach_local_node(n, nodes_)
  {
    double y_coord = node_y_fr_n(n, p4est_, nodes_);

    if (front_phi_.ptr[n] > 0 && y_coord > mushy_zone_max) mushy_zone_max = y_coord;
    if (front_phi_.ptr[n] < 0 && y_coord < mushy_zone_min) mushy_zone_min = y_coord;
  }
  front_phi_.restore_array();

  double buffer[] = { mushy_zone_min, mushy_zone_max};
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &buffer, 2, MPI_DOUBLE, MPI_MIN, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
  mushy_zone_min = buffer[0];
  mushy_zone_max = buffer[1];

//  dendrite_number_.destroy();
//  dendrite_tip_   .destroy();

//  dendrite_number_.create(front_phi_.vec);
//  dendrite_tip_   .create(front_phi_.vec);

  VecSetGhost(dendrite_number_.vec, -1.);
  VecSetGhost(dendrite_tip_.vec, -1.);

  // cut off denrites' base
  double cut_off_length = mushy_zone_max - dendrite_cut_off_fraction_*MAX(dendrite_min_length_, mushy_zone_max-mushy_zone_min);

  vec_and_ptr_t phi_cut(front_phi_.vec);
  VecCopyGhost(front_phi_.vec, phi_cut.vec);

  phi_cut.get_array();
  foreach_node(n, nodes_)
  {
    if (node_y_fr_n(n, p4est_, nodes_) < cut_off_length) phi_cut.ptr[n] = -1;
  }
  phi_cut.restore_array();

  // enumerate dendrite by using Arthur's parallel region-growing code
  num_dendrites_ = 0;
  compute_islands_numbers(*ngbd_, phi_cut.vec, num_dendrites_, dendrite_number_.vec);
  phi_cut.destroy();

  ierr = PetscPrintf(p4est_->mpicomm, "%d prominent dendrites found.\n", num_dendrites_); CHKERRXX(ierr);

  // find the tip of every dendrite
  std::vector<double> x_tip(num_dendrites_, 0);
  std::vector<double> y_tip(num_dendrites_, 0);

  dendrite_number_.get_array();
  front_phi_      .get_array();

  foreach_local_node(n, nodes_)
  {
    int i = (int) dendrite_number_.ptr[n];
    if (i >= 0)
    {
      const quad_neighbor_nodes_of_node_t qnnn = ngbd_->get_neighbors(n);
      int j = (int) qnnn.f_0p0_linear(dendrite_number_.ptr);
      if (j < 0)
      {
        double phi_n   = front_phi_.ptr[n];
        double phi_nei = qnnn.f_0p0_linear(front_phi_.ptr);

        double y_coord = node_y_fr_n(n, p4est_, nodes_) + fabs(phi_n)/(fabs(phi_n)+fabs(phi_nei)) * dxyz_[1];

        if (y_coord > y_tip[i])
        {
          y_tip[i] = y_coord;
          x_tip[i] = node_x_fr_n(n, p4est_, nodes_);
        }
      }
    }
  }
  dendrite_number_.restore_array();
  front_phi_      .restore_array();

  std::vector<double> y_tip_g(num_dendrites_, 0);
  std::vector<double> x_tip_g(num_dendrites_, 0);

  mpiret = MPI_Allreduce(y_tip.data(), y_tip_g.data(), num_dendrites_, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);

  for (unsigned int i = 0; i < num_dendrites_; ++i)
  {
    if (y_tip[i] != y_tip_g[i]) x_tip[i] = -10;
  }

  mpiret = MPI_Allreduce(x_tip.data(), x_tip_g.data(), num_dendrites_, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);

  // auxiliary field that shows found tip locations
  dendrite_tip_.get_array();
  foreach_node(n, nodes_)
  {
    dendrite_tip_.ptr[n] = -1;
    double x_coord = node_x_fr_n(n, p4est_, nodes_);
    for (unsigned short i = 0; i < num_dendrites_; ++i)
    {
      if (fabs(x_tip_g[i] - x_coord) < EPS) dendrite_tip_.ptr[n] = i;
    }
  }
  dendrite_tip_.restore_array();

  // sample quantities along the tip of every dendrite
  // specifically: phi, c0, c1, t, vn, c0s, c1s, tf, kappa, velo

  splitting_criteria_t *data = (splitting_criteria_t*)p4est_->user_pointer;
  int nb_sample_points = brick_.nxyztrees[1]*pow(2, data->max_lvl)+1;

  std::vector<double> line_phi;
  std::vector<double> line_tl;
  std::vector<double> line_ts;
  std::vector<double> line_tf;
  std::vector<double> line_vn;
  std::vector<double> line_kappa;
  std::vector<double> line_vf;

  std::vector< std::vector<double> > line_cl(num_comps_);
  std::vector< std::vector<double> > line_cs(num_comps_);

  for (unsigned short dendrite_idx = 0; dendrite_idx < num_dendrites_; ++dendrite_idx)
  {
    double xyz0[P4EST_DIM] = { x_tip_g[dendrite_idx], brick_.xyz_min[1] };
    double xyz1[P4EST_DIM] = { x_tip_g[dendrite_idx], brick_.xyz_max[1] };

    double xyz[P4EST_DIM];
    double dxyz[P4EST_DIM];

    foreach_dimension(dim) dxyz[dim] = (xyz1[dim]-xyz0[dim])/((double) nb_sample_points - 1.);

    line_phi  .clear(); line_phi  .resize(nb_sample_points, -DBL_MAX);
    line_tl   .clear(); line_tl   .resize(nb_sample_points, -DBL_MAX);
    line_ts   .clear(); line_ts   .resize(nb_sample_points, -DBL_MAX);
    line_tf   .clear(); line_tf   .resize(nb_sample_points, -DBL_MAX);
    line_vn   .clear(); line_vn   .resize(nb_sample_points, -DBL_MAX);
    line_vf   .clear(); line_vf   .resize(nb_sample_points, -DBL_MAX);
    line_kappa.clear(); line_kappa.resize(nb_sample_points, -DBL_MAX);

    for (int i = 0; i < num_comps_; ++i)
    {
      line_cl[i].clear(); line_cl[i].resize(nb_sample_points, -DBL_MAX);
      line_cs[i].clear(); line_cs[i].resize(nb_sample_points, -DBL_MAX);
    }

    my_p4est_interpolation_nodes_t interp(ngbd_);
    my_p4est_interpolation_nodes_t h_interp(solid_ngbd_);

    for (unsigned int i = 0; i < nb_sample_points; ++i)
    {
      foreach_dimension(dim) xyz[dim] = xyz0[dim] + ((double) i) * dxyz[dim];
      interp.add_point_local(i, xyz);
      h_interp.add_point_local(i, xyz);
    }

    interp.set_input(front_phi_.vec, linear); interp.interpolate_local(line_phi.data());
    interp.set_input(front_velo_norm_[0].vec, linear); interp.interpolate_local(line_vn.data());
    interp.set_input(tl_[0].vec, linear); interp.interpolate_local(line_tl.data());
    interp.set_input(ts_[0].vec, linear); interp.interpolate_local(line_ts.data());
    for (int i = 0; i < num_comps_; ++i)
    {
      interp.set_input(cl_[0].vec[i], linear); interp.interpolate_local(line_cl[i].data());
    }

    h_interp.set_input(solid_tf_.vec, linear); h_interp.interpolate_local(line_tf.data());
    h_interp.set_input(solid_front_curvature_.vec, linear); h_interp.interpolate_local(line_kappa.data());
    h_interp.set_input(solid_front_velo_norm_.vec, linear); h_interp.interpolate_local(line_vf.data());
    for (int i = 0; i < num_comps_; ++i)
    {
      h_interp.set_input(solid_cl_.vec[i], linear); h_interp.interpolate_local(line_cs[i].data());
    }

    int mpiret;

    mpiret = MPI_Allreduce(MPI_IN_PLACE, line_phi.data(), nb_sample_points, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, line_tl .data(), nb_sample_points, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, line_ts .data(), nb_sample_points, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, line_tf .data(), nb_sample_points, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, line_vn .data(), nb_sample_points, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, line_vf .data(), nb_sample_points, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
    mpiret = MPI_Allreduce(MPI_IN_PLACE, line_kappa.data(), nb_sample_points, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);

    for (int i = 0; i < num_comps_; ++i)
    {
      mpiret = MPI_Allreduce(MPI_IN_PLACE, line_cl[i].data(), nb_sample_points, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
      mpiret = MPI_Allreduce(MPI_IN_PLACE, line_cs[i].data(), nb_sample_points, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
    }

    const char* out_dir = getenv("OUT_DIR");

    char dirname[1024];
    sprintf(dirname, "%s/dendrites/%05d", out_dir, iter);

    std::ostringstream command;
    command << "mkdir -p " << dirname;
    int ret_sys = system(command.str().c_str());
    if (ret_sys<0)
      throw std::invalid_argument("could not create directory");

    char filename[1024];
    if (p4est_->mpirank == 0)
    {
      std::ios_base::openmode mode = (dendrite_idx == 0 ? std::ios_base::out : std::ios_base::app);
      sprintf(filename, "%s/%s.txt", dirname, "phi"); save_vector(filename, line_phi, mode);
      sprintf(filename, "%s/%s.txt", dirname, "tl");  save_vector(filename, line_tl,  mode);
      sprintf(filename, "%s/%s.txt", dirname, "ts");  save_vector(filename, line_ts,  mode);
      sprintf(filename, "%s/%s.txt", dirname, "tf");  save_vector(filename, line_tf,  mode);
      sprintf(filename, "%s/%s.txt", dirname, "vn");  save_vector(filename, line_vn,  mode);
      sprintf(filename, "%s/%s.txt", dirname, "vf");  save_vector(filename, line_vf,  mode);
      sprintf(filename, "%s/%s.txt", dirname, "kappa"); save_vector(filename, line_kappa, mode);

      for (int i = 0; i < num_comps_; ++i)
      {
        sprintf(filename, "%s/%s%d.txt", dirname, "cl", i);  save_vector(filename, line_cl[i], mode);
        sprintf(filename, "%s/%s%d.txt", dirname, "cs", i);  save_vector(filename, line_cs[i], mode);
      }
    }
  }
}

void my_p4est_multialloy_t::sample_along_line(const double xyz0[], const double xyz1[], const unsigned int nb_points, Vec data, std::vector<double> out)
{
  out.clear();
  out.resize(nb_points, -DBL_MAX);

  my_p4est_interpolation_nodes_t interp(ngbd_);

  double xyz[P4EST_DIM];
  double dxyz[P4EST_DIM];

  foreach_dimension(dim) dxyz[dim] = (xyz1[dim]-xyz0[dim])/((double) nb_points - 1.);

  for (unsigned int i = 0; i < nb_points; ++i)
  {
    foreach_dimension(dim) xyz[dim] = xyz0[dim] + ((double) i) * dxyz[dim];
    interp.add_point(i, xyz);
  }

  interp.set_input(data, linear);
  interp.interpolate(out.data());

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, out.data(), nb_points, MPI_DOUBLE, MPI_MAX, p4est_->mpicomm); SC_CHECK_MPI(mpiret);
}

void my_p4est_multialloy_t::regularize_front(Vec front_phi_old)
{
  ierr = PetscLogEventBegin(log_my_p4est_multialloy_update_grid_regularize_front, 0, 0, 0, 0); CHKERRXX(ierr);
  ierr = PetscPrintf(p4est_->mpicomm, "Removing problem geometries... "); CHKERRXX(ierr);

  ierr = VecSetGhost(smoothed_nodes_.vec, 0); CHKERRXX(ierr);
  smoothed_nodes_.get_array();

  double new_phi_val = .5*dxyz_min_;

  int num_nodes_smoothed = 0;

  // first pass: bridge too narrow regions by lifting level-set function, reinitializing,
  // brigning it down and checking which nodes flipped sign
  // (note that it also smooths out too sharp corners which are usually formed by
  // solidifying front ``getting stuck'' on grid nodes)
  if (proximity_smoothing_ != 0) {

    // shift level-set upwards
    my_p4est_level_set_t ls(ngbd_);
    vec_and_ptr_t front_phi_tmp(front_phi_.vec);
    double shift = dxyz_min_*proximity_smoothing_;
    ierr = VecCopyGhost(front_phi_.vec, front_phi_tmp.vec); CHKERRXX(ierr);
    ierr = VecShiftGhost(front_phi_tmp.vec, shift); CHKERRXX(ierr);

    // eliminate small pools created by lifting
    int num_islands = 0;
    vec_and_ptr_t island_number(front_phi_.vec);

    VecScaleGhost(front_phi_tmp.vec, -1.);
    compute_islands_numbers(*ngbd_, front_phi_tmp.vec, num_islands, island_number.vec);
    VecScaleGhost(front_phi_tmp.vec, -1.);

    if (num_islands > 1) {
      ierr = PetscPrintf(p4est_->mpicomm, "%d subpools removed... ", num_islands-1); CHKERRXX(ierr);
      island_number.get_array();
      front_phi_tmp.get_array();

      // compute liquid pools areas
      // TODO: make it real area instead of number of points
      std::vector<double> island_area(num_islands, 0);

      foreach_local_node(n, nodes_) {
        if (island_number.ptr[n] >= 0) {
          ++island_area[ (int) island_number.ptr[n] ];
        }
      }

      int mpiret = MPI_Allreduce(MPI_IN_PLACE, island_area.data(), num_islands, MPI_DOUBLE, MPI_SUM, p4est_->mpicomm); SC_CHECK_MPI(mpiret);

      // find the biggest liquid pool
      int main_island = 0;
      int island_area_max = island_area[0];

      for (int i = 1; i < num_islands; ++i) {
        if (island_area[i] > island_area_max) {
          main_island     = i;
          island_area_max = island_area[i];
        }
      }

      if (main_island < 0) throw;

      // remove all but the biggest pool
      foreach_node(n, nodes_) {
        if (front_phi_tmp.ptr[n] < 0 && island_number.ptr[n] != main_island) {
          front_phi_tmp.ptr[n] = new_phi_val;
        }
      }

      island_number.restore_array();
      front_phi_tmp.restore_array();

      // TODO: make the decision whether to solidify a liquid pool or not independently
      // for each pool based on its size and shape
    }

    island_number.destroy();

    // reinitilize and bring it down
    ls.reinitialize_2nd_order(front_phi_tmp.vec, 50);
    VecShiftGhost(front_phi_tmp.vec, -shift);

    // "solidify" nodes that changed sign
    front_phi_.get_array();
    front_phi_tmp.get_array();

    foreach_node(n, nodes_) {
      if (front_phi_.ptr[n] < 0 && front_phi_tmp.ptr[n] > 0) {
        front_phi_.ptr[n] = front_phi_tmp.ptr[n];
        smoothed_nodes_.ptr[n] = 1;
        num_nodes_smoothed++;
      }
    }

    front_phi_.restore_array();
    front_phi_tmp.restore_array();
    front_phi_tmp.destroy();
  }

  // second pass (optional, not used anymore): smooth out too sharp protruding corners
  // (usually formed by melting front ``stuck'' on grid nodes)
  if (proximity_smoothing_ != 0 && 0) {
    double *front_phi_old_ptr;
    ierr = VecGetArray(front_phi_old, &front_phi_old_ptr); CHKERRXX(ierr);

    // shift level-set upwards and reinitialize
    my_p4est_level_set_t ls(ngbd_);
    vec_and_ptr_t front_phi_tmp(front_phi_.vec);
    double shift = -0.1*dxyz_min_*proximity_smoothing_;
    VecCopyGhost(front_phi_.vec, front_phi_tmp.vec);
    VecShiftGhost(front_phi_tmp.vec, shift);

    ls.reinitialize_2nd_order(front_phi_tmp.vec, 50);
    VecShiftGhost(front_phi_tmp.vec, -shift);

    // "solidify" nodes that changed sign
    front_phi_.get_array();
    front_phi_tmp.get_array();

    foreach_node(n, nodes_) {
      if ( (front_phi_.ptr[n] > 0) &&
           (front_phi_tmp.ptr[n] < 0) &&
           (front_phi_old_ptr[n] > 0) ) {
//        (front_phi_cur.ptr[n] < front_phi_old_ptr[n]) ) {
        front_phi_.ptr[n] = front_phi_tmp.ptr[n];
        smoothed_nodes_.ptr[n] = 1;
        num_nodes_smoothed++;
      }
    }

    front_phi_.restore_array();
    front_phi_tmp.restore_array();
    front_phi_tmp.destroy();

    ierr = VecRestoreArray(front_phi_old, &front_phi_old_ptr); CHKERRXX(ierr);
  }

  // third pass: look for isolated pools of liquid and remove them
  ierr = MPI_Allreduce(MPI_IN_PLACE, &num_nodes_smoothed, 1, MPI_INT, MPI_SUM, p4est_->mpicomm); SC_CHECK_MPI(ierr);

  if (num_nodes_smoothed > 0) // assuming such pools can form only due to the artificial bridging (I guess it's quite safe to say, but not entirely correct)
  {
    ierr = PetscPrintf(p4est_->mpicomm, "%d nodes smoothed... ", num_nodes_smoothed); CHKERRXX(ierr);
    int num_islands = 0;
    vec_and_ptr_t island_number(front_phi_.vec);

    VecScaleGhost(front_phi_.vec, -1.);
    compute_islands_numbers(*ngbd_, front_phi_.vec, num_islands, island_number.vec);
    VecScaleGhost(front_phi_.vec, -1.);

    if (num_islands > 1)
    {
      ierr = PetscPrintf(p4est_->mpicomm, "%d pools removed... ", num_islands-1); CHKERRXX(ierr);
      island_number.get_array();
      front_phi_.get_array();

      // compute liquid pools areas
      // TODO: make it real area instead of number of points
      std::vector<double> island_area(num_islands, 0);

      foreach_local_node(n, nodes_) {
        if (island_number.ptr[n] >= 0) {
          ++island_area[ (int) island_number.ptr[n] ];
        }
      }

      int mpiret = MPI_Allreduce(MPI_IN_PLACE, island_area.data(), num_islands, MPI_DOUBLE, MPI_SUM, p4est_->mpicomm); SC_CHECK_MPI(mpiret);

      // find the biggest liquid pool
      int main_island = 0;
      int island_area_max = island_area[0];

      for (int i = 1; i < num_islands; ++i) {
        if (island_area[i] > island_area_max) {
          main_island     = i;
          island_area_max = island_area[i];
        }
      }

      if (main_island < 0) throw;

      // solidify all but the biggest pool
      foreach_node(n, nodes_) {
        if (front_phi_.ptr[n] < 0 && island_number.ptr[n] != main_island) {
          front_phi_.ptr[n] = new_phi_val;
          smoothed_nodes_.ptr[n] = 2;
        }
      }

      island_number.restore_array();
      front_phi_.restore_array();

      // TODO: make the decision whether to solidify a liquid pool or not independently
      // for each pool based on its size and shape
    }

    island_number.destroy();
  }

  smoothed_nodes_.restore_array();

  ierr = VecGhostUpdateBegin(smoothed_nodes_.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(smoothed_nodes_.vec, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = PetscPrintf(p4est_->mpicomm, "done!\n"); CHKERRXX(ierr);
  ierr = PetscLogEventEnd(log_my_p4est_multialloy_update_grid_regularize_front, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_multialloy_t::compute_solid()
{
  ierr = PetscLogEventBegin(log_my_p4est_multialloy_compute_solid, 0, 0, 0, 0); CHKERRXX(ierr);
  // get new values for nodes that just solidified
  solid_front_phi_    .get_array();
  solid_front_phi_nm1_.get_array();

  // first, figure out which nodes just became solid
  my_p4est_interpolation_nodes_t interp(ngbd_);
  std::vector<p4est_locidx_t> freezing_nodes;
  double xyz[P4EST_DIM];
  foreach_node(n, solid_nodes_)
  {
    if (solid_front_phi_    .ptr[n] > 0 &&
        solid_front_phi_nm1_.ptr[n] < 0)
    {
      node_xyz_fr_n(n, solid_p4est_, solid_nodes_, xyz);
      interp.add_point(n, xyz);
      freezing_nodes.push_back(n);
    }
  }

  // get values concentrations and temperature at those nodes
  vec_and_ptr_array_t cl_new(num_comps_, solid_front_phi_.vec);
  vec_and_ptr_array_t cl_old(num_comps_, solid_front_phi_.vec);

  vec_and_ptr_t tl_new(solid_front_phi_.vec);
  vec_and_ptr_t tl_old(solid_front_phi_.vec);

  vec_and_ptr_t vn_new(solid_front_phi_.vec);
  vec_and_ptr_t vn_old(solid_front_phi_.vec);

  for (int i = 0; i < num_comps_; ++i)
  {
    interp.set_input(cl_[1].vec[i], linear); interp.interpolate(cl_old.vec[i]);
    interp.set_input(cl_[0].vec[i], linear); interp.interpolate(cl_new.vec[i]);
  }
  interp.set_input(tl_[1].vec, linear); interp.interpolate(tl_old.vec);
  interp.set_input(tl_[0].vec, linear); interp.interpolate(tl_new.vec);

  interp.set_input(front_curvature_.vec,    linear); interp.interpolate(solid_front_curvature_.vec);
  interp.set_input(front_velo_norm_[1].vec, linear); interp.interpolate(vn_old.vec);
  interp.set_input(front_velo_norm_[0].vec, linear); interp.interpolate(vn_new.vec);

  interp.set_input(seed_map_.vec, linear); interp.interpolate(solid_seed_.vec);
  interp.set_input(smoothed_nodes_.vec, linear); interp.interpolate(solid_smoothed_nodes_.vec);

  cl_old.get_array();
  cl_new.get_array();
  tl_old.get_array();
  tl_new.get_array();
  vn_old.get_array();
  vn_new.get_array();

  solid_cl_.get_array();
  solid_part_coeff_.get_array();
  solid_tf_.get_array();
  solid_time_.get_array();
  solid_front_velo_norm_.get_array();

  for (unsigned int i = 0; i < freezing_nodes.size(); ++i)
  {
    p4est_locidx_t n = freezing_nodes[i];

    // estimate the time when the interface crossed a node
    double tau = (fabs(solid_front_phi_nm1_.ptr[n]))
                 /(fabs(solid_front_phi_nm1_.ptr[n]) + fabs(solid_front_phi_.ptr[n]));

    vector<double> cl_all(num_comps_);
    for (int i = 0; i < num_comps_; ++i) {
      cl_all[i] = (cl_old.ptr[i][n]*(1.-tau) + cl_new.ptr[i][n]*tau);
    }

    for (int i = 0; i < num_comps_; ++i) {
      solid_part_coeff_.ptr[i][n] = part_coeff_(i, cl_all.data());
      solid_cl_.ptr[i][n] = cl_all[i];
    }
    solid_tf_.ptr[n] = tl_old.ptr[n]*(1.-tau) + tl_new.ptr[n]*tau;
    solid_time_.ptr[n] = (time_-dt_[0])*(1.-tau) + time_*tau;
    solid_front_velo_norm_.ptr[n] = vn_old.ptr[n]*(1.-tau) + vn_new.ptr[n]*tau;
  }

  cl_old.restore_array();
  cl_new.restore_array();
  tl_old.restore_array();
  tl_new.restore_array();
  vn_old.restore_array();
  vn_new.restore_array();

  solid_cl_.restore_array();
  solid_part_coeff_.restore_array();
  solid_tf_.restore_array();
  solid_time_.restore_array();
  solid_front_velo_norm_.restore_array();

  cl_old.destroy();
  cl_new.destroy();
  tl_old.destroy();
  tl_new.destroy();
  vn_old.destroy();
  vn_new.destroy();

  my_p4est_level_set_t ls(solid_ngbd_);
  VecScaleGhost(solid_front_phi_.vec, -1.);
  double band_use    = 4*diag_;
  double band_extend = 4*diag_;
  for (int i = 0; i < num_comps_; ++i)
  {
    ls.extend_Over_Interface_TVD(solid_front_phi_.vec, solid_cl_.vec[i], 5, 1, band_use, band_extend);
    ls.extend_Over_Interface_TVD(solid_front_phi_.vec, solid_part_coeff_.vec[i], 5, 1, band_use, band_extend);
  }
  ls.extend_Over_Interface_TVD(solid_front_phi_.vec, solid_tf_.vec,  5, 1, band_use, band_extend);
  ls.extend_Over_Interface_TVD(solid_front_phi_.vec, solid_front_curvature_.vec,  5, 1, band_use, band_extend);
  ls.extend_Over_Interface_TVD(solid_front_phi_.vec, solid_front_velo_norm_.vec,  5, 1, band_use, band_extend);
  ls.extend_Over_Interface_TVD(solid_front_phi_.vec, solid_seed_.vec,  5, 0, band_use, band_extend);
//  ls.extend_Over_Interface_TVD(solid_front_phi_.vec, solid_time_.vec,  5, 1);
  VecScaleGhost(solid_front_phi_.vec, -1.);
  ierr = PetscLogEventEnd(log_my_p4est_multialloy_compute_solid, 0, 0, 0, 0); CHKERRXX(ierr);
}
