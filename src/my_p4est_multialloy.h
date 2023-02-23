#ifndef MY_P4EST_MULTIALLOY_H
#define MY_P4EST_MULTIALLOY_H

#ifdef P4_TO_P8
#include <src/my_p8est_tools.h>
#include <p8est.h>
#include <p8est_ghost.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_poisson_nodes_multialloy.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_macros.h>
#else
#include <src/my_p4est_tools.h>
#include <p4est.h>
#include <p4est_ghost.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_poisson_nodes_multialloy.h>
#include <src/my_p4est_stefan_with_fluids.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_macros.h>
#endif

#include <src/casl_math.h>

using std::vector;

class my_p4est_multialloy_t
{
private:
  PetscErrorCode ierr;
  mpi_environment_t* mpi_;

  //--------------------------------------------------
  // Main grid
  //--------------------------------------------------
  my_p4est_brick_t             brick_;
  p4est_connectivity_t        *connectivity_;
  p4est_t                     *p4est_;
  p4est_ghost_t               *ghost_;
  p4est_nodes_t               *nodes_;
  my_p4est_hierarchy_t        *hierarchy_;
  my_p4est_node_neighbors_t   *ngbd_;

  splitting_criteria_t *sp_crit_;

  //--------------------------------------------------
  // nm1 grid (used for backtraced values when solving w fluids)
  //--------------------------------------------------
  p4est_t                     *p4est_nm1;
  p4est_ghost_t               *ghost_nm1;
  p4est_nodes_t               *nodes_nm1;
  my_p4est_hierarchy_t        *hierarchy_nm1;
  my_p4est_node_neighbors_t   *ngbd_nm1;

  //--------------------------------------------------
  // Auxiliary grid that does not coarsen to keep track of quantities inside the solid
  //--------------------------------------------------
  p4est_t                     *solid_p4est_;
  p4est_ghost_t               *solid_ghost_;
  p4est_nodes_t               *solid_nodes_;
  my_p4est_hierarchy_t        *solid_hierarchy_;
  my_p4est_node_neighbors_t   *solid_ngbd_;

  //--------------------------------------------------
  // Grid characteristics
  //--------------------------------------------------
  double dxyz_[P4EST_DIM];
  double dxyz_max_, dxyz_min_;
  double dxyz_close_interface_;
  double diag_;

  //--------------------------------------------------
  // Stefan with fluids solver (used when we are solving coupled case)
  //--------------------------------------------------
  my_p4est_stefan_with_fluids_t* stefan_w_fluids_solver;

  //--------------------------------------------------
  // Geometry
  //--------------------------------------------------
  vec_and_ptr_t contr_phi_;
  vec_and_ptr_t front_phi_;
  vec_and_ptr_t front_curvature_;

  vec_and_ptr_dim_t contr_phi_dd_;
  vec_and_ptr_dim_t front_phi_dd_;
  vec_and_ptr_dim_t front_normal_;

  //--------------------------------------------------
  // Physical fields
  //--------------------------------------------------
  int num_time_layers_;

  /* temperature */
  vector<vec_and_ptr_t> tl_;
  vec_and_ptr_dim_t tl_dd;
  vector<vec_and_ptr_t> ts_;

  // backtraced temperatures:
  vec_and_ptr_t tl_backtrace_n;
  vec_and_ptr_t tl_backtrace_nm1;

  /* concentrations */
  vector<vec_and_ptr_array_t> cl_;
  vec_and_ptr_dim_t           cl0_grad_;
  vec_and_ptr_dim_t cl0_dd;
  // backtraced concentrations:
  vec_and_ptr_array_t cl_backtrace_n;
  vec_and_ptr_array_t cl_backtrace_nm1;

  /* velocity */
  vector<vec_and_ptr_dim_t> front_velo_;
  vector<vec_and_ptr_t>     front_velo_norm_;

  //--------------------------------------------------
  // Navier-Stokes fields 
  //--------------------------------------------------
  vec_and_ptr_dim_t v_n;
  vec_and_ptr_dim_t v_nm1;
  vec_and_ptr_t vorticity;
  vec_and_ptr_t vorticity_refine;
  vec_and_ptr_t press_nodes;

  // Never actually use below, we will just get them from stefan_W_fluids after it 
  // initializes its fluid solver component
  my_p4est_cell_neighbors_t* ngbd_c_;
  my_p4est_faces_t* faces_;

  my_p4est_navier_stokes_t* ns;


  //--------------------------------------------------
  // Lagrangian Multipliers (for speeding up)
  //--------------------------------------------------
  /* temperature */
  vec_and_ptr_t psi_tl_;
  vec_and_ptr_t psi_ts_;

  /* concentrations */
  vec_and_ptr_array_t psi_cl_;

  //--------------------------------------------------
  // Geometry on the auxiliary grid
  //--------------------------------------------------
  vec_and_ptr_t       solid_front_phi_;
  vec_and_ptr_t       solid_front_phi_nm1_;
  vec_and_ptr_t       solid_front_curvature_;
  vec_and_ptr_t       solid_front_velo_norm_;
  vec_and_ptr_t       solid_time_; // time at which alloy solidified
  vec_and_ptr_t       solid_tf_; // temperature at which alloy solidified
  vec_and_ptr_array_t solid_cl_; // composition of solidified region
  vec_and_ptr_array_t solid_part_coeff_; // partition coefficient at freezing
  vec_and_ptr_t       solid_seed_; // seed tag
  vec_and_ptr_t       solid_smoothed_nodes_; // is used to track nodes that were artificially solidified during front regularization

  //--------------------------------------------------
  // physical parameters
  //--------------------------------------------------
  // composition parameters
  int            num_comps_;
  vector<double> solute_diff_;

  // thermal parameters
  double density_l_, heat_capacity_l_, thermal_cond_l_;
  double density_s_, heat_capacity_s_, thermal_cond_s_;
  double latent_heat_;

  // phase diagram
  double (*liquidus_value_)(double *);
  double (*liquidus_slope_)(int, double *);
  double (*part_coeff_)(int, double *);

  // undercoolings
  int              num_seeds_;
  vec_and_ptr_t    seed_map_;
  vector<CF_DIM *> eps_c_;
  vector<CF_DIM *> eps_v_;

  // volumetric heat generation
  CF_DIM *vol_heat_gen_;

  // boundary conditions at container
  BoundaryConditionType contr_bc_type_temp_;
  BoundaryConditionType contr_bc_type_conc_;
  BoundaryConditionType contr_bc_type_vel_;
  BoundaryConditionType contr_bc_type_pres_;

  CF_DIM           *contr_bc_value_temp_;
  vector<CF_DIM *>  contr_bc_value_conc_;
  CF_DIM           *contr_bc_value_pres_;
  CF_DIM           *contr_bc_value_vel_[P4EST_DIM];

  // boundary condtions at walls
//  BoundaryConditionType wall_bc_type_temp_;
//  BoundaryConditionType wall_bc_type_conc_;

  WallBCDIM* wall_bc_type_temp_;
  WallBCDIM* wall_bc_type_conc_;

  //BoundaryConditionType wall_bc_type_vel_;
  //BoundaryConditionType wall_bc_type_pres_;

  CF_DIM           *wall_bc_value_temp_;
  vector<CF_DIM *>  wall_bc_value_conc_;
  //CF_DIM           *wall_bc_value_pres_;
  //CF_DIM           *wall_bc_value_vel_[P4EST_DIM];

  // simulation scale
  double scaling_;

  // -------------------------------------------------
  // Terms related to the coupled convergence test
  // -------------------------------------------------
  bool there_is_convergence_test;
  // Source terms for the RHS of the PDE's:
  CF_DIM* convergence_external_source_conc[2]; // for conc0 and conc1
  CF_DIM* convergence_external_source_temp[2]; // for liquid domain and solid domain
  CF_DIM* convergence_external_forces_NS[2]; // one per cartesian direction

  // Source terms to play into the boundary conditions at the interface:
  CF_DIM* convergence_external_source_temperature_jump;
  CF_DIM* convergence_external_source_temperature_flux_jump;

  CF_DIM* convergence_external_source_conc_robin[2]; // for conc0 and conc1
  CF_DIM* convergence_external_source_Gibbs_Thomson; // source term in the Gibbs Thomson relation

  // -------------------------------------------------
  // Physical parameters and nondim groups for problem coupled with fluids
  // -------------------------------------------------
  double mu_l; // fluid viscosity
  double Pr;
  double Le_0, Le_1, Le_2, Le_3;
  double St;
  double deltaC_0, deltaC_1, deltaC_2, deltaC_3;
  double deltaT;

  double RaT, RaC_0, RaC_1, RaC_2, RaC_3;
  double l_char; // characteristic length scale

  //--------------------------------------------------
  // solver parameters
  //--------------------------------------------------
  int max_iterations_;
  int update_c0_robin_;

  double proximity_smoothing_;

  double bc_tolerance_;
  double cfl_number_;

  bool use_superconvergent_robin_;
  bool use_points_on_interface_;
  bool enforce_planar_front_;

  interpolation_method interpolation_between_grids_;

  bool solve_with_fluids; // To set if we are solving the coupled problem
  bool loading_from_previous_state;

  //--------------------------------------------------
  // Dendrite counting and profiling
  //--------------------------------------------------
  int    num_dendrites_;
  double dendrite_cut_off_fraction_;
  double dendrite_min_length_;

  vec_and_ptr_t dendrite_number_;
  vec_and_ptr_t dendrite_tip_;

  //--------------------------------------------------
  // Misc
  //--------------------------------------------------
  vec_and_ptr_t  bc_error_;
  vector<double> dt_;
  double         time_;
  double         dt_min_;
  double         dt_max_;
  double         front_velo_norm_max_;
  vec_and_ptr_t  smoothed_nodes_; // is used to track nodes that were artificially solidified during front regularization
  vec_and_ptr_t  front_phi_unsmooth_; // is used to track nodes that were artificially solidified during front regularization

  int iteration_w_fluids;
  int iteration_one_step;

  int vtk_idx;

  static my_p4est_node_neighbors_t *v_ngbd;
  static double **v_c_p, **v_c0_d_p, **v_c0_dd_p, **v_normal_p;
  static double v_factor;
  static double (*v_part_coeff)(int, double *);
  static int v_num_comps;
  static bool is_there_convergence_v;
  static CF_DIM* external_conc0_robin_term;

  void set_velo_interpolation(my_p4est_node_neighbors_t *ngbd, double **c_p, double **c0_d_p, double **c0_dd_p,
                              double **normal_p, double factor)
  {
    v_ngbd       = ngbd;
    v_c_p        = c_p;
    v_c0_d_p     = c0_d_p;
    v_c0_dd_p    = c0_dd_p;
    v_normal_p   = normal_p;
    v_factor     = factor;
    v_part_coeff = part_coeff_;
    v_num_comps  = num_comps_;

    is_there_convergence_v = there_is_convergence_test;
    if(is_there_convergence_v){
      external_conc0_robin_term = convergence_external_source_conc_robin[0];
    }
  }

  static double velo(p4est_locidx_t n, int dir, double dist)
  {
    const quad_neighbor_nodes_of_node_t &qnnn = (*v_ngbd)[n];
    vector<double> cl_all (v_num_comps);
    for (int j = 0; j < v_num_comps; ++j) {
      cl_all[j] = qnnn.interpolate_in_dir(dir, dist, v_c_p[j]);
    }

    double source_term=0.;
    if(is_there_convergence_v){
      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, v_ngbd->p4est, v_ngbd->nodes, xyz);
      source_term += (*external_conc0_robin_term)(xyz[0], xyz[1]);
    }

    // ELYCE TO-DO: add the source term for front_conc_flux here to get the correct interface velocity expression when including the source term hC1
    return -v_factor/(1.-v_part_coeff(0, cl_all.data()))*
        ( qnnn.interpolate_in_dir(dir, dist, v_c0_d_p[0])*qnnn.interpolate_in_dir(dir, dist, v_normal_p[0])
        + qnnn.interpolate_in_dir(dir, dist, v_c0_d_p[1])*qnnn.interpolate_in_dir(dir, dist, v_normal_p[1])
        - source_term)
        / MAX(qnnn.interpolate_in_dir(dir, dist, v_c_p[0], v_c0_dd_p[1]), 1e-7 ) ;
  }

  // input[] = { cl_{0}, ..., cl_{num_comps-1}, c0x, c0y, c0z, nx, ny, nz };
  static double velo2(double input[])
  {
    return -v_factor/(1.-v_part_coeff(0, input))*
        SUMD(input[v_num_comps + 0]*input[v_num_comps + P4EST_DIM + 0],
             input[v_num_comps + 1]*input[v_num_comps + P4EST_DIM + 1],
             input[v_num_comps + 2]*input[v_num_comps + P4EST_DIM + 2])
        / MAX(input[0], 1e-7);
  }


  // To-do w/fluids: actually write this:
  bool refine_by_vorticity;
  void set_refine_by_vorticity(bool refine_by_vorticity_){refine_by_vorticity = refine_by_vorticity_;}
  bool refine_by_d2T;
  void set_refine_by_d2T(bool ref_by_d2T_){refine_by_d2T = ref_by_d2T_;}
  bool refine_by_d2C;
  void set_refine_by_d2C(bool ref_by_d2C_){refine_by_d2C = ref_by_d2C_;}
  double vorticity_threshold;
  double d2T_threshold;
  double d2C_threshold;
  //void prepare_refinement_fields();

  CF_DIM* initial_NS_velocity_n[P4EST_DIM];
  CF_DIM* initial_NS_velocity_nm1[P4EST_DIM];

public:
  my_p4est_multialloy_t(int num_comps, int time_order);
  ~my_p4est_multialloy_t();

  void initialize(MPI_Comm mpi_comm, double xyz_min[], double xyz_max[], int nxyz[], int periodicity[], CF_2 &level_set, int lmin, int lmax, double lip, double band, bool solve_w_fluids=false);

  inline void set_mpi_env(mpi_environment_t* mpi_in){
    mpi_ = mpi_in;
  }

  void initialize_for_fluids(my_p4est_stefan_with_fluids_t* stefan_w_fluids_solver_);
  inline void set_scaling(double value) { scaling_ = value; }
  inline void set_composition_parameters(double solute_diff[])
  {
    for (int i = 0; i < num_comps_; ++i)
    {
      solute_diff_[i] = solute_diff[i];
    }
  }

  void set_iteration_one_step(int iter){
    iteration_one_step = iter;
  }
  int get_iteration_one_step(){
    return iteration_one_step;
  }
  int get_vtk_idx(){
    return vtk_idx;
  }

  vec_and_ptr_dim_t get_front_normals(){return front_normal_;}
//  my_p4est_node_neighbors_t* get_ngbd(){return ngbd_;}

  void set_initial_NS_velocity_n_(CF_DIM* init_vel_n_[P4EST_DIM]){

    foreach_dimension(d){
      //printf("cf address (provided) : %p, value before setting : %p \n", init_vel_n_[d], initial_NS_velocity_n[d]);
      initial_NS_velocity_n[d] = init_vel_n_[d];
      //printf("cf address (after setting) : %p \n ", initial_NS_velocity_n[d]);
    }
    //std::cout << "initial ns velocity n is set\n";


  }

  void set_initial_NS_velocity_nm1_(CF_DIM* init_vel_nm1_[P4EST_DIM]){
    foreach_dimension(d){
      initial_NS_velocity_nm1[d] = init_vel_nm1_[d];
    }
    //std::cout << "initial ns velocity nm1 is set\n";
  }

  inline void set_thermal_parameters(double latent_heat,
                                     double density_l, double heat_capacity_l, double thermal_cond_l,
                                     double density_s, double heat_capacity_s, double thermal_cond_s)
  {
    latent_heat_ = latent_heat;
    density_l_ = density_l; heat_capacity_l_ = heat_capacity_l; thermal_cond_l_ = thermal_cond_l;
    density_s_ = density_s; heat_capacity_s_ = heat_capacity_s; thermal_cond_s_ = thermal_cond_s;
  }

  inline void set_liquidus(double (*liquidus_value)(double *), double (*liquidus_slope)(int, double *), double (*part_coeff)(int, double *))
  {
    liquidus_value_ = liquidus_value;
    liquidus_slope_ = liquidus_slope;
    part_coeff_ = part_coeff;
  }

  inline void set_undercoolings(int num_seeds, Vec seed_map, CF_DIM *eps_v[], CF_DIM *eps_c[])
  {
    num_seeds_    = num_seeds;
    if(!loading_from_previous_state)VecCopyGhost(seed_map, seed_map_.vec);

    eps_v_.resize(num_seeds, NULL);
    eps_c_.resize(num_seeds, NULL);
    for (int i = 0; i < num_seeds; ++i)
    {
      eps_v_[i] = eps_v[i];
      eps_c_[i] = eps_c[i];
    }

    if(!loading_from_previous_state){
      // Only do this interp if we are not reloading.
      // If reloading, we have save/loaded this information
      my_p4est_interpolation_nodes_t interp(ngbd_);
      double xyz[P4EST_DIM];
      foreach_node(n, solid_nodes_)
      {
        node_xyz_fr_n(n, solid_p4est_, solid_nodes_, xyz);
        interp.add_point(n, xyz);
      }
      interp.set_input(seed_map_.vec, linear);
      interp.interpolate(solid_seed_.vec);
    }
  }

  // Setting functions for coupled problem with fluids:
  // ---------------------------------------------------
  inline void set_solve_with_fluids(){ solve_with_fluids = true;}
  void set_load_from_previous_state(bool load_from_prev){ loading_from_previous_state = load_from_prev;}

  void set_mu_l(double mu_l_){mu_l = mu_l_;}
  void set_nondimensional_groups(double Pr_, double St_,
                                 double Le_0_, double Le_1_, double Le_2_, double Le_3_,
                                 double RaT_,
                                 double RaC_0_, double RaC_1_, double RaC_2_, double RaC_3_,
                                 double deltaT_,
                                 double deltaC_0_, double deltaC_1_, double deltaC_2_, double deltaC_3_){
    Pr = Pr_;
    St = St_;

    Le_0 = Le_0_;
    Le_1 = Le_1_;
    Le_2 = Le_2_;
    Le_3 = Le_3_;

    RaT = RaT_;

    RaC_0 = RaC_0_;
    RaC_1 = RaC_1_;
    RaC_2 = RaC_2_;
    RaC_3 = RaC_3_;

    deltaT = deltaT_;

    deltaC_0 = deltaC_0_; deltaC_1 = deltaC_1_; deltaC_2 = deltaC_2_; deltaC_3 = deltaC_3_;
  }


  // ---------------------------------------------------
  inline void set_container_conditions_thermal(BoundaryConditionType bc_type, CF_DIM &bc_value)
  {
    contr_bc_type_temp_  =  bc_type;
    contr_bc_value_temp_ = &bc_value;
  }

  inline void set_container_conditions_composition(BoundaryConditionType bc_type, CF_DIM *bc_value[])
  {
    contr_bc_type_conc_ = bc_type;
    for (int i = 0; i < num_comps_; ++i)
    {
      contr_bc_value_conc_[i] = bc_value[i];
    }
  }

  inline void set_container_conditions_velocity(BoundaryConditionType bc_type, CF_DIM* bc_value[P4EST_DIM])
  {
    contr_bc_type_vel_ =  bc_type;
    foreach_dimension(d){
      contr_bc_value_vel_[d] = bc_value[d];

    }
  }

  inline void set_container_conditions_pressure(BoundaryConditionType bc_type, CF_DIM &bc_value)
  {
    contr_bc_type_pres_ =  bc_type;
    contr_bc_value_pres_ = &bc_value; 
  }
  /*
  inline void set_wall_conditions_velocity(BoundaryConditionType bc_type, CF_DIM* bc_value[P4EST_DIM])
  {
    wall_bc_type_vel_ =  bc_type;
    foreach_dimension(d){
      wall_bc_value_vel_[d] = bc_value[d];

    }
  }
  */
  /*inline void set_wall_conditions_pressure(BoundaryConditionType bc_type, CF_DIM &bc_value)
  {
    wall_bc_type_pres_ =  bc_type;
    wall_bc_value_pres_ = &bc_value;
  }*/


  // set fluid velocity interface bc
  my_p4est_stefan_with_fluids_t* return_stefan_solver(){
    return  stefan_w_fluids_solver;
  }
  my_p4est_stefan_with_fluids_t::interfacial_bc_fluid_velocity_t* bc_interface_val_fluid_vel[P4EST_DIM];
  BoundaryConditionType* bc_interface_type_fluid_vel[P4EST_DIM];

  CF_DIM* bc_interface_val_fluid_press;
  BoundaryConditionType bc_interface_type_fluid_press;

  CF_DIM* bc_wall_value_velocity[P4EST_DIM];
  WallBCDIM* bc_wall_type_velocity[P4EST_DIM];

  CF_DIM* bc_wall_value_pressure;
  WallBCDIM* bc_wall_type_pressure;

/*  void set_bc_interface_conditions_velocity(my_p4est_stefan_with_fluids_t::interfacial_bc_fluid_velocity_t* bc_interface_value_velocity_[P4EST_DIM]){
    foreach_dimension(d){
      bc_interface_val_fluid_vel[d] = bc_interface_value_velocity_[d];
    }
  }*/
  void set_bc_interface_value_velocity(my_p4est_stefan_with_fluids_t::interfacial_bc_fluid_velocity_t* bc_interface_val_fluid_vel_[P4EST_DIM]){
    foreach_dimension(d){
      bc_interface_val_fluid_vel[d] = bc_interface_val_fluid_vel_[d];
    }
  }

  void set_bc_interface_type_velocity(BoundaryConditionType* bc_interface_type_velocity_[P4EST_DIM]){

    foreach_dimension(d){
      bc_interface_type_fluid_vel[d] = bc_interface_type_velocity_[d];
    }
  }

  void set_bc_interface_value_pressure(CF_DIM* bc_interface_val_fluid_press_){
    bc_interface_val_fluid_press = bc_interface_val_fluid_press_;
  }
  void set_bc_interface_type_pressure(BoundaryConditionType bc_interface_type_fluid_press_){
    bc_interface_type_fluid_press = bc_interface_type_fluid_press_;
  }

  // ------
  // Wall velocity:
  // ------
  void set_bc_wall_value_velocity(CF_DIM* bc_wall_value_velocity_[P4EST_DIM]){
    foreach_dimension(d){
      bc_wall_value_velocity[d] = bc_wall_value_velocity_[d];
      double test_val = (*bc_wall_value_velocity[d])(1.0, 1.0);
    }
  }

  void set_bc_wall_type_velocity(WallBCDIM* bc_wall_type_velocity_[P4EST_DIM]){
    foreach_dimension(d){
      bc_wall_type_velocity[d] = bc_wall_type_velocity_[d];
    }
  }


  // ------
  // Wall pressure:
  // ------
  void set_bc_wall_value_pressure(CF_DIM* bc_wall_value_pressure_){
    bc_wall_value_pressure = bc_wall_value_pressure_;
  }
  void set_bc_wall_type_pressure(WallBCDIM* bc_wall_type_pressure_){
    bc_wall_type_pressure = bc_wall_type_pressure_;
  }



/*
  void set_bc_interface_conditions_pressure(BoundaryConditionType bc_type, CF_DIM& bc_value){

    bc_interface_type_fluid_press = bc_type;
    bc_interface_val_fluid_press = &bc_value;

  }*/

//  inline void set_wall_conditions_thermal(BoundaryConditionType bc_type, CF_DIM &bc_value)
//  {
//    wall_bc_type_temp_  =  bc_type;
//    wall_bc_value_temp_ = &bc_value;
//  }
  inline void set_wall_conditions_thermal(WallBCDIM* bc_type, CF_DIM &bc_value)
  {
    wall_bc_type_temp_  =  bc_type;
    wall_bc_value_temp_ = &bc_value;

//    double test_val = (bc_value)(1.0, 1.0);
    //printf("Sets wall bc temp: Testing wall bc value temp : %0.2f \n", test_val);
  }

//  inline void set_wall_conditions_composition(BoundaryConditionType bc_type, CF_DIM *bc_value[])
//  {
//    wall_bc_type_conc_ = bc_type;
//    for (int i = 0; i < num_comps_; ++i)
//    {
//      wall_bc_value_conc_[i] = bc_value[i];
//    }
//  }

  inline void set_wall_conditions_composition(WallBCDIM* bc_type, CF_DIM *bc_value[])
  {
    wall_bc_type_conc_ = bc_type;
    for (int i = 0; i < num_comps_; ++i)
    {
      wall_bc_value_conc_[i] = bc_value[i];
    }
  }

  /*
  inline void set_wall_conditions_velocity(BoundaryConditionType bc_type, CF_DIM &bc_value)
  {
    wall_bc_type_vel_  =  bc_type;
    wall_bc_value_vel_ = &bc_value;
  }*/
  /*
  inline void set_wall_conditions_pressure(BoundaryConditionType bc_type, CF_DIM &bc_value)
  {
    wall_bc_type_pres_  =  bc_type;
    wall_bc_value_pres_ = &bc_value;
  }*/

  void set_front(Vec phi);
  void set_container(Vec phi);

  inline void set_temperature(Vec tl, Vec ts)
  {
    for (int i = 0; i < num_time_layers_; ++i)
    {
      VecCopyGhost(tl, tl_[i].vec);
      VecCopyGhost(ts, ts_[i].vec);
    }

    my_p4est_interpolation_nodes_t interp(ngbd_);
    interp.add_all_nodes(solid_p4est_, solid_nodes_);
    interp.set_input(ts_[0].vec, linear);
    interp.interpolate(solid_tf_.vec);
  }

  inline void set_temperature(CF_DIM &tl, CF_DIM &ts, CF_DIM &tf)
  {
    for (int i = 0; i < num_time_layers_; ++i)
    {
      tl.t = -double(i)*dt_[0];
      ts.t = -double(i)*dt_[0];
      sample_cf_on_nodes(p4est_, nodes_, tl, tl_[i].vec);
      sample_cf_on_nodes(p4est_, nodes_, ts, ts_[i].vec);
    }

    sample_cf_on_nodes(solid_p4est_, solid_nodes_, tf, solid_tf_.vec);
  }

  inline void set_temperature_solve_w_fluids(CF_DIM &tl, CF_DIM &ts, CF_DIM &tf)
  {

    if(num_time_layers_ <=2){

      tl.t = 0.;
      ts.t = 0.;
      sample_cf_on_nodes(p4est_, nodes_, tl, tl_[0].vec);
      sample_cf_on_nodes(p4est_, nodes_, ts, ts_[0].vec);

      if(num_time_layers_>1){
        tl.t = -dt_[0];
        ts.t = -dt_[0];

        sample_cf_on_nodes(p4est_nm1, nodes_nm1, tl, tl_[0].vec);
        sample_cf_on_nodes(p4est_nm1, nodes_nm1, ts, ts_[0].vec);

      }
      sample_cf_on_nodes(solid_p4est_, solid_nodes_, tf, solid_tf_.vec);

    }
    else{
      throw std::invalid_argument("my_p4est_multialloy:set_temperature_solve_w_fluids -- you can only solve with fluids using 2 or less time layers");
    }

  }


  inline void set_concentration(Vec cl[], Vec cs[])
  {
    for (int j = 0; j < num_comps_; ++j)
    {
      for (int i = 0; i < num_time_layers_; ++i)
      {
        VecCopyGhost(cl[j], cl_[i].vec[j]);
      }
    }

    my_p4est_interpolation_nodes_t interp(ngbd_);
    interp.add_all_nodes(solid_p4est_, solid_nodes_);

    for (int j = 0; j < num_comps_; ++j)
    {
      interp.set_input(cs[j], linear);
      interp.interpolate(solid_cl_.vec[j]);
    }
  }

  inline void set_concentration(CF_DIM *cl[], CF_DIM *cs[])
  {
    for (int j = 0; j < num_comps_; ++j)
    {
      for (int i = 0; i < num_time_layers_; ++i)
      {
        cl[j]->t = -double(i)*dt_[0];
        sample_cf_on_nodes(p4est_, nodes_, *cl[j], cl_[i].vec[j]);
      }

      sample_cf_on_nodes(solid_p4est_, solid_nodes_, *cs[j], solid_cl_.vec[j]);
    }

    // compute partition coefficient inside solid
    solid_cl_.get_array();
    solid_part_coeff_.get_array();

    vector<double> cl_all(num_comps_);
    foreach_node(n, solid_nodes_) {
      for (int i = 0; i < num_comps_; ++i) {
        cl_all[i] = solid_cl_.ptr[i][n];
      }

      for (int i = 0; i < num_comps_; ++i) {
        solid_part_coeff_.ptr[i][n] = part_coeff_(i, cl_all.data());
      }
    }

    solid_cl_.restore_array();
    solid_part_coeff_.restore_array();
  }

  //  inline void set_concentration_solve_w_fluids(CF_DIM *cl[], CF_DIM *cs[])
  //  {
  //    for (int j = 0; j < num_comps_; ++j)
  //    {
  //      for (int i = 0; i < num_time_layers_; ++i)
  //      {
  //        cl[j]->t = -double(i)*dt_[0];
  //        sample_cf_on_nodes(p4est_, nodes_, *cl[j], cl_[i].vec[j]);
  //      }

  //      sample_cf_on_nodes(solid_p4est_, solid_nodes_, *cs[j], solid_cl_.vec[j]);
  //    }

  //    // compute partition coefficient inside solid
  //    solid_cl_.get_array();
  //    solid_part_coeff_.get_array();

  //    vector<double> cl_all(num_comps_);
  //    foreach_node(n, solid_nodes_) {
  //      for (int i = 0; i < num_comps_; ++i) {
  //        cl_all[i] = solid_cl_.ptr[i][n];
  //      }

  //      for (int i = 0; i < num_comps_; ++i) {
  //        solid_part_coeff_.ptr[i][n] = part_coeff_(i, cl_all.data());
  //      }
  //    }

  //    solid_cl_.restore_array();
  //    solid_part_coeff_.restore_array();
  //  }

  inline void set_normal_velocity(Vec v)
  {
    for (int i = 0; i < num_time_layers_; ++i)
    {
      VecCopyGhost(v, front_velo_norm_[i].vec);
      foreach_dimension(dim)
      {
        VecPointwiseMultGhost(front_velo_[i].vec[dim], v, front_normal_.vec[dim]);
      }
    }
  }

  inline void set_velocity(CF_DIM &vn, DIM(CF_DIM &vx, CF_DIM &vy, CF_DIM &vz), CF_DIM &vf)
  {
    for (int i = 0; i < num_time_layers_; ++i)
    {
      vn.t = -double(i)*dt_[0];
      EXECD( vx.t = -double(i)*dt_[0],
             vy.t = -double(i)*dt_[0],
             vz.t = -double(i)*dt_[0] );

      sample_cf_on_nodes(p4est_, nodes_, vn, front_velo_norm_[i].vec);
      EXECD( sample_cf_on_nodes(p4est_, nodes_, vx, front_velo_[i].vec[0]),
             sample_cf_on_nodes(p4est_, nodes_, vy, front_velo_[i].vec[1]),
             sample_cf_on_nodes(p4est_, nodes_, vz, front_velo_[i].vec[2]) );

      VecScaleGhost(front_velo_norm_[i].vec, -1.);

      if (i == 0) {
        compute_dt();
        for (int j = 1; j < num_time_layers_; ++j) {
          dt_[j] = dt_[0];
        }
      }
    }

    sample_cf_on_nodes(solid_p4est_, solid_nodes_, vf, solid_front_velo_norm_.vec);
  }

  inline void set_ft(CF_DIM &ft_cf)
  {
    sample_cf_on_nodes(solid_p4est_, solid_nodes_, ft_cf, solid_time_.vec);
  }

  inline p4est_t*       get_p4est() { return p4est_; }
  inline p4est_nodes_t* get_nodes() { return nodes_; }
  inline p4est_ghost_t* get_ghost() { return ghost_; }
  inline my_p4est_node_neighbors_t* get_ngbd()  { return ngbd_; }

  inline p4est_t*       get_solid_p4est() { return solid_p4est_; }
  inline p4est_nodes_t* get_solid_nodes() { return solid_nodes_; }
  inline p4est_ghost_t* get_solid_ghost() { return solid_ghost_; }
  inline my_p4est_node_neighbors_t* get_solid_ngbd()  { return solid_ngbd_; }

  inline Vec  get_contr_phi()    { return contr_phi_.vec; }
  inline Vec  get_front_phi()    { return front_phi_.vec; }
  inline Vec* get_front_phi_dd() { return front_phi_dd_.vec; }
  inline Vec  get_normal_velocity() { return front_velo_norm_[0].vec; }

  inline Vec  get_tl() { return tl_[0].vec; }
  inline Vec  get_ts() { return ts_[0].vec; }
  inline Vec* get_cl() { return cl_[0].vec.data(); }
  inline Vec  get_cl(int idx) { return cl_[0].vec[idx]; }

  inline Vec  get_ft() { return solid_time_.vec; }
  inline Vec  get_tf() { return solid_tf_.vec; }
  inline Vec  get_vf() { return solid_front_velo_norm_.vec; }
  inline Vec* get_cs() { return solid_cl_.vec.data(); }
  inline Vec  get_cs(int idx) { return solid_cl_.vec[idx]; }
  inline Vec* get_kps() { return solid_part_coeff_.vec.data(); }
  inline Vec  get_kps(int idx) { return solid_part_coeff_.vec[idx]; }

  inline double get_dt() { return dt_[0]; }
  inline double get_front_velocity_max() { return front_velo_norm_max_; }
  my_p4est_stefan_with_fluids_t* get_stefan_w_fluids_solver(){
    return stefan_w_fluids_solver;
  }
//  inline double get_max_interface_velocity() { return vgamma_max_; }

  inline void set_dt_all(double dt)
  {
    dt_.assign(num_time_layers_, dt);
  }

  inline void set_dt(double dt)
  {
    dt_[0] = dt;
  }

  inline void set_dt_limits(double dt_min, double dt_max)
  {
    dt_min_ = dt_min;
    dt_max_ = dt_max;
  }

  inline void set_use_superconvergent_robin(bool value)   { use_superconvergent_robin_ = value; }
  inline void set_use_points_on_interface  (bool value)   { use_points_on_interface_   = value; }
  inline void set_enforce_planar_front     (bool value)   { enforce_planar_front_      = value; }

  inline void set_update_c0_robin          (int value)    { update_c0_robin_           = value; }
  inline void set_max_iterations           (int value)    { max_iterations_            = value; }

  inline void set_bc_tolerance             (double value) { bc_tolerance_              = value; }
  inline void set_cfl                      (double value) { cfl_number_                = value; }
  inline void set_dendrite_cut_off_fraction(double value) { dendrite_cut_off_fraction_ = value; }
  inline void set_dendrite_min_length      (double value) { dendrite_min_length_       = value; }
  inline void set_volumetric_heat          (CF_DIM &value){ vol_heat_gen_              =&value; }
  inline void set_proximity_smoothing      (double value) { proximity_smoothing_       = value; }

  // ---------------------------------------------
  // For setting convergence test related things:
  // ---------------------------------------------
  void set_there_is_convergence_test(bool is_conv_test){
    there_is_convergence_test = is_conv_test;
    if(there_is_convergence_test){
      if(num_comps_ != 2){
        throw std::runtime_error("my_p4est_multialloy: you have attempted to run a convergence test with an invalid number of components. Currently, only one convergence test with 2 components has been implemented. \n");
      }
    }
  }

  void set_convergence_source_NS(CF_DIM* source_NS[2]){
    foreach_dimension(d){
      convergence_external_forces_NS[d] = source_NS[d];
    }
  }

  void set_convergence_source_conc(CF_DIM* source_conc[2]){
    for(unsigned int i = 0; i<2; i++){
      convergence_external_source_conc[i] = source_conc[i];
    }
  }

  void set_convergence_source_temp(CF_DIM* source_temp[2]){
    for(unsigned int i = 0; i<2; i++){
      convergence_external_source_temp[i] = source_temp[i];
    }
  }

  void set_convergence_source_temp_jump(CF_DIM* temp_jump){
    convergence_external_source_temperature_jump = temp_jump;
  }

  void set_convergence_source_temp_flux_jump(CF_DIM* temp_flux_jump){
    convergence_external_source_temperature_flux_jump = temp_flux_jump;
  }

  void set_convergence_source_conc_robin(CF_DIM* source_conc_robin[2]){
    for(unsigned int i=0; i<2; i++){
      convergence_external_source_conc_robin[i] = source_conc_robin[i];
    }
  }

  void set_convergence_source_Gibbs_Thomson(CF_DIM* source_Gibbs){
    convergence_external_source_Gibbs_Thomson = source_Gibbs;
  }

  // ---------------------------------------------
  void regularize_front(Vec front_phi_old);
  void compute_geometric_properties_front();
  void compute_geometric_properties_contr();
  void compute_velocity();
  void compute_solid();

  void compute_dt();
  void update_grid();
  void update_grid_w_fluids();
  void update_grid_w_fluids_v2();
  void update_grid_eno();
  void update_grid_solid();
  int  one_step(int it_scheme=2, double *bc_error_max=NULL, double *bc_error_avg=NULL, std::vector<int> *num_pdes=NULL, std::vector<double> *bc_error_max_all=NULL, std::vector<double> *bc_error_avg_all=NULL);
  int  one_step_w_fluids(int it_scheme, double *bc_error_max, double *bc_error_avg, std::vector<int> *num_pdes, std::vector<double> *bc_error_max_all, std::vector<double> *bc_error_avg_all);
  void save_VTK(int iter);
  void save_VTK_solid(int iter);
  void save_p4est(int iter);
  void save_p4est_solid(int iter);

  void count_dendrites(int iter);
  void sample_along_line(const double xyz0[], const double xyz1[], const unsigned int nb_points, Vec data, std::vector<double> out);

  // Functions for save/load of simulation state:
  void fill_or_load_double_parameters(save_or_load flag, PetscInt num, PetscReal *data);
  void fill_or_load_integer_parameters(save_or_load flag, PetscInt num, PetscInt *data);
  void save_or_load_parameters(const char* filename, save_or_load flag);
  void prepare_fields_for_save_or_load(vector<save_or_load_element_t> &fields_to_save_np1, vector<save_or_load_element_t> &fields_to_save_n, vector<save_or_load_element_t> &fields_to_save_solid);
  void load_state(const char* path_to_folder);
  void save_state(const char* path_to_directory, unsigned int n_saved);

};


#endif /* MY_P4EST_MULTIALLOY_H */
