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
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_macros.h>
#endif

#include <src/casl_math.h>

using std::vector;

class my_p4est_multialloy_t
{
private:
  PetscErrorCode ierr;
  int i,j,k;

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
  // Auxiliary grid that does not coarsen to keep track of quantities inside the solid
  //--------------------------------------------------
  p4est_t                     *history_p4est_;
  p4est_ghost_t               *history_ghost_;
  p4est_nodes_t               *history_nodes_;
  my_p4est_hierarchy_t        *history_hierarchy_;
  my_p4est_node_neighbors_t   *history_ngbd_;

  //--------------------------------------------------
  // Grid characteristics
  //--------------------------------------------------
  double dxyz_[P4EST_DIM];
  double dxyz_max_, dxyz_min_;
  double dxyz_close_interface_;
  double diag_;

  //--------------------------------------------------
  // Geometry
  //--------------------------------------------------
  vec_and_ptr_t contr_phi_;
  vec_and_ptr_t front_phi_;
  vec_and_ptr_t front_curvature_;
  vec_and_ptr_t front_curvature_filtered_;

  vec_and_ptr_dim_t contr_phi_dd_;
  vec_and_ptr_dim_t front_phi_dd_;
  vec_and_ptr_dim_t front_normal_;

  //--------------------------------------------------
  // Physical fields
  //--------------------------------------------------
  int num_time_layers_;

  /* temperature */
  vector<vec_and_ptr_t> tl_;
  vector<vec_and_ptr_t> ts_;

  /* concentrations */
  vector<vec_and_ptr_array_t> cl_;
  vec_and_ptr_dim_t           cl0_grad_;

  /* velocity */
  vector<vec_and_ptr_dim_t> front_velo_;
  vector<vec_and_ptr_t>     front_velo_norm_;

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
  vec_and_ptr_t       history_front_phi_;
  vec_and_ptr_t       history_front_phi_nm1_;
  vec_and_ptr_t       history_front_curvature_;
  vec_and_ptr_t       history_front_velo_norm_;
  vec_and_ptr_t       history_tf_; // temperature at which alloy solidified
  vec_and_ptr_array_t history_cs_; // composition of solidified region
  vec_and_ptr_t       history_seed_; // seed tag

  //--------------------------------------------------
  // physical parameters
  //--------------------------------------------------
  // composition parameters
  int            num_comps_;
  vector<double> solute_diff_;
  vector<double> part_coeff_;

  // thermal parameters
  double density_l_, heat_capacity_l_, thermal_cond_l_;
  double density_s_, heat_capacity_s_, thermal_cond_s_;
  double latent_heat_;

  // front conditions
  double melting_temp_;
  double (*liquidus_value_)(double *);
  double (*liquidus_slope_)(int, double *);

  // undercoolings
  int              num_seeds_;
  vec_and_ptr_t    seed_map_;
  vector<CF_DIM *> eps_c_;
  vector<CF_DIM *> eps_v_;

  // volumetric heat generation
  CF_DIM *vol_heat_gen_;

  // boundary conditions at container
  BoundaryConditionType         contr_bc_type_temp_;
  vector<BoundaryConditionType> contr_bc_type_conc_;

  CF_DIM           *contr_bc_value_temp_;
  vector<CF_DIM *>  contr_bc_value_conc_;

  // boundary condtions at walls
  WallBCDIM           *wall_bc_type_temp_;
  vector<WallBCDIM *>  wall_bc_type_conc_;

  CF_DIM           *wall_bc_value_temp_;
  vector<CF_DIM *>  wall_bc_value_conc_;

  // simulation scale
  double scaling_;

  //--------------------------------------------------
  // solver parameters
  //--------------------------------------------------
  int pin_every_n_iterations_;
  int max_iterations_;
  int update_c0_robin_;
  int front_smoothing_;

  double bc_tolerance_;
  double phi_thresh_;
  double cfl_number_;
  double curvature_smoothing_;
  double curvature_smoothing_steps_;

  bool use_superconvergent_robin_;
  bool use_points_on_interface_;
  bool save_history_;
  bool enforce_planar_front_;

  interpolation_method interpolation_between_grids_;

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

  static my_p4est_node_neighbors_t *v_ngbd;
  static double *v_c_p, **v_c_d_p, **v_c_dd_p, **v_normal_p;
  static double v_factor;

  void set_velo_interpolation(my_p4est_node_neighbors_t *ngbd, double *c_p, double **c_d_p, double **c_dd_p, double **normal_p, double factor)
  {
    v_ngbd     = ngbd;
    v_c_p      = c_p;
    v_c_d_p    = c_d_p;
    v_c_dd_p   = c_dd_p;
    v_normal_p = normal_p;
    v_factor   = factor;
  }

  static double velo(p4est_locidx_t n, int dir, double dist)
  {
    const quad_neighbor_nodes_of_node_t &qnnn = (*v_ngbd)[n];
    return -v_factor*
        ( qnnn.interpolate_in_dir(dir, dist, v_c_d_p[0])*qnnn.interpolate_in_dir(dir, dist, v_normal_p[0])
        + qnnn.interpolate_in_dir(dir, dist, v_c_d_p[1])*qnnn.interpolate_in_dir(dir, dist, v_normal_p[1]))
        / MAX(qnnn.interpolate_in_dir(dir, dist, v_c_p, v_c_dd_p), 1e-7);
  };


public:
  my_p4est_multialloy_t(int num_comps, int num_time_layers);
  ~my_p4est_multialloy_t();

  void initialize(MPI_Comm mpi_comm, double xyz_min[], double xyz_max[], int nxyz[], int periodicity[], CF_2 &level_set, int lmin, int lmax, double lip, double band);

  inline void set_scaling(double value) { scaling_ = value; }
  inline void set_composition_parameters(double solute_diff[], double part_coeff[])
  {
    for (int i = 0; i < num_comps_; ++i)
    {
      solute_diff_[i] = solute_diff[i];
      part_coeff_ [i] = part_coeff [i];
    }
  }

  inline void set_thermal_parameters(double latent_heat,
                                     double density_l, double heat_capacity_l, double thermal_cond_l,
                                     double density_s, double heat_capacity_s, double thermal_cond_s)
  {
    latent_heat_ = latent_heat;
    density_l_ = density_l; heat_capacity_l_ = heat_capacity_l; thermal_cond_l_ = thermal_cond_l;
    density_s_ = density_s; heat_capacity_s_ = heat_capacity_s; thermal_cond_s_ = thermal_cond_s;
  }

  inline void set_liquidus(double melting_temp, double (*liquidus_value)(double *), double (*liquidus_slope)(int, double *))
  {
    melting_temp_   = melting_temp;
    liquidus_value_ = liquidus_value;
    liquidus_slope_ = liquidus_slope;
  }

  inline void set_undercoolings(int num_seeds, Vec seed_map, CF_DIM *eps_v[], CF_DIM *eps_c[])
  {
    num_seeds_    = num_seeds;

    VecCopyGhost(seed_map, seed_map_.vec);

    eps_v_.resize(num_seeds, NULL);
    eps_c_.resize(num_seeds, NULL);

    for (int i = 0; i < num_seeds; ++i)
    {
      eps_v_[i] = eps_v[i];
      eps_c_[i] = eps_c[i];
    }

    my_p4est_interpolation_nodes_t interp(ngbd_);

    double xyz[P4EST_DIM];
    foreach_node(n, history_nodes_)
    {
      node_xyz_fr_n(n, history_p4est_, history_nodes_, xyz);
      interp.add_point(n, xyz);
    }

    interp.set_input(seed_map_.vec, linear);
    interp.interpolate(history_seed_.vec);
  }

  inline void set_container_conditions_thermal(BoundaryConditionType bc_type, CF_DIM &bc_value)
  {
    contr_bc_type_temp_  =  bc_type;
    contr_bc_value_temp_ = &bc_value;
  }

  inline void set_container_conditions_composition(BoundaryConditionType bc_type[], CF_DIM *bc_value[])
  {
    for (int i = 0; i < num_comps_; ++i)
    {
      contr_bc_type_conc_ [i] = bc_type [i];
      contr_bc_value_conc_[i] = bc_value[i];
    }
  }

  inline void set_wall_conditions_thermal(WallBCDIM &bc_type, CF_DIM &bc_value)
  {
    wall_bc_type_temp_  = &bc_type;
    wall_bc_value_temp_ = &bc_value;
  }

  inline void set_wall_conditions_composition(WallBCDIM *bc_type[], CF_DIM *bc_value[])
  {
    for (int i = 0; i < num_comps_; ++i)
    {
      wall_bc_type_conc_ [i] = bc_type [i];
      wall_bc_value_conc_[i] = bc_value[i];
    }
  }

  void set_front(Vec phi);
  void set_container(Vec phi);

  inline void set_temperature(Vec tl, Vec ts)
  {
    for (i = 0; i < num_time_layers_; ++i)
    {
      VecCopyGhost(tl, tl_[i].vec);
      VecCopyGhost(ts, ts_[i].vec);
    }

    my_p4est_interpolation_nodes_t interp(ngbd_);

    double xyz[P4EST_DIM];
    foreach_node(n, history_nodes_)
    {
      node_xyz_fr_n(n, history_p4est_, history_nodes_, xyz);
      interp.add_point(n, xyz);
    }

    interp.set_input(ts_[0].vec, linear);
    interp.interpolate(history_tf_.vec);
  }

  inline void set_concentration(Vec cl[], Vec cs[])
  {
    for (j = 0; j < num_comps_; ++j)
    {
      for (i = 0; i < num_time_layers_; ++i)
      {
        VecCopyGhost(cl[j], cl_[i].vec[j]);
      }
    }

    my_p4est_interpolation_nodes_t interp(ngbd_);

    double xyz[P4EST_DIM];
    foreach_node(n, history_nodes_)
    {
      node_xyz_fr_n(n, history_p4est_, history_nodes_, xyz);
      interp.add_point(n, xyz);
    }

    for (j = 0; j < num_comps_; ++j)
    {
      interp.set_input(cs[j], linear);
      interp.interpolate(history_cs_.vec[j]);
    }
  }

  inline void set_normal_velocity(Vec v)
  {
    for (i = 0; i < num_time_layers_; ++i)
    {
      VecCopyGhost(v, front_velo_norm_[i].vec);
      foreach_dimension(dim)
      {
        VecPointwiseMultGhost(front_velo_[i].vec[dim], v, front_normal_.vec[dim]);
      }
    }
    // todo
    // copy to history_front_velo
  }

  inline p4est_t*       get_p4est() { return p4est_; }
  inline p4est_nodes_t* get_nodes() { return nodes_; }
  inline p4est_ghost_t* get_ghost() { return ghost_; }
  inline my_p4est_node_neighbors_t* get_ngbd()  { return ngbd_; }

  inline Vec  get_contr_phi()    { return front_phi_.vec; }
  inline Vec  get_front_phi()    { return front_phi_.vec; }
  inline Vec* get_front_phi_dd() { return front_phi_dd_.vec; }
  inline Vec  get_normal_velocity() { return front_velo_norm_[0].vec; }

  inline double get_dt() { return dt_[0]; }
  inline double get_front_velocity_max() { return front_velo_norm_max_; }

//  inline double get_max_interface_velocity() { return vgamma_max_; }

  inline void set_dt(double dt)
  {
    dt_.assign(num_time_layers_, dt);
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
  inline void set_pin_every_n_iterations   (int value)    { pin_every_n_iterations_    = value; }
  inline void set_max_iterations           (int value)    { max_iterations_            = value; }
  inline void set_front_smoothing          (int value)    { front_smoothing_           = value; }
  inline void set_curvature_smoothing      (double value,
                                            int    steps) { curvature_smoothing_       = value;
                                                            curvature_smoothing_steps_ = steps; }

  inline void set_phi_thresh               (double value) { phi_thresh_                = value; }
  inline void set_bc_tolerance             (double value) { bc_tolerance_              = value; }
  inline void set_cfl                      (double value) { cfl_number_                = value; }
  inline void set_dendrite_cut_off_fraction(double value) { dendrite_cut_off_fraction_ = value; }
  inline void set_dendrite_min_length      (double value) { dendrite_min_length_       = value; }
  inline void set_volumetric_heat          (CF_DIM &value) { vol_heat_gen_              =&value; }


  void regularize_front();
  void compute_geometric_properties_front();
  void compute_geometric_properties_contr();
  void compute_filtered_curvature();
  void compute_velocity();
  void compute_solid();

  void compute_dt();
  void update_grid();
  void update_grid_eno();
  void update_grid_history();
  int  one_step();
  void save_VTK(int iter);
  void save_VTK_solid(int iter);

  void count_dendrites(int iter);
  void sample_along_line(const double xyz0[], const double xyz1[], const unsigned int nb_points, Vec data, std::vector<double> out);

  inline Vec  get_tl() { return tl_[0].vec; }
  inline Vec  get_ts() { return ts_[0].vec; }
  inline Vec* get_cl() { return cl_[0].vec.data(); }

};


#endif /* MY_P4EST_MULTIALLOY_H */
