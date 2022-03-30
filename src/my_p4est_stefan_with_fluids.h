#ifndef MY_P4EST_STEFAN_WITH_FLUIDS_H
#define MY_P4EST_STEFAN_WITH_FLUIDS_H


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


enum problem_dimensionalization_type_t{
  NONDIM_BY_FLUID_VELOCITY, // nondim by the characteristic fluid velocity
  NONDIM_BY_SCALAR_DIFFUSIVITY, // nondimensionalized by the temperature or concentration fluid diffusivity
  DIMENSIONAL // dimensional problem (highly not recommended)
};

class my_p4est_stefan_with_fluids_t
{
public:
  my_p4est_stefan_with_fluids_t();

private:
  // -----------------------------------------------
  // p4est variables
  // -----------------------------------------------

  p4est_t*              p4est;
  p4est_nodes_t*        nodes;
  p4est_ghost_t*        ghost;
  p4est_connectivity_t* conn;
  my_p4est_brick_t      brick;
  my_p4est_hierarchy_t* hierarchy;
  my_p4est_node_neighbors_t* ngbd;

  p4est_t               *p4est_np1;
  p4est_nodes_t         *nodes_np1;
  p4est_ghost_t         *ghost_np1;
  my_p4est_hierarchy_t* hierarchy_np1;
  my_p4est_node_neighbors_t* ngbd_np1;

  // Level set function(s):---------------------------
  vec_and_ptr_t phi;
  vec_and_ptr_t phi_nm1; // LSF for previous timestep... we must keep this so that hodge fields can be updated correctly in NS process

  vec_and_ptr_t phi_solid; // LSF for solid domain: -- This will be assigned within the loop as the negative of phi
  vec_and_ptr_t phi_cylinder;   // LSF for the inner cylinder, if applicable (example ICE_OVER_CYLINDER)

  vec_and_ptr_dim_t phi_dd;
  vec_and_ptr_dim_t phi_solid_dd;
  vec_and_ptr_dim_t phi_cylinder_dd;

  // Interface geometry:------------------------------
  vec_and_ptr_dim_t normal;
  vec_and_ptr_t curvature;

  vec_and_ptr_dim_t liquid_normals;
  vec_and_ptr_dim_t solid_normals;
  vec_and_ptr_dim_t cyl_normals;

  // Poisson problem:---------------------------------
  int cube_refinement = 1;
  my_p4est_poisson_nodes_mls_t *solver_Tl = NULL;  // will solve poisson problem for Temperature in liquid domains
  my_p4est_poisson_nodes_mls_t *solver_Ts = NULL;  // will solve poisson problem for Temperature in solid domain

  vec_and_ptr_t T_l_n;
  vec_and_ptr_t T_l_nm1;
  vec_and_ptr_t T_l_backtrace;
  vec_and_ptr_t T_l_backtrace_nm1;
  vec_and_ptr_t rhs_Tl;

  vec_and_ptr_t T_s_n;
  vec_and_ptr_t rhs_Ts;

  // Vectors to hold first derivatives of T
  vec_and_ptr_dim_t T_l_d;
  vec_and_ptr_dim_t T_s_d;
  vec_and_ptr_dim_t T_l_dd;

  // Stefan problem:------------------------------------
  vec_and_ptr_dim_t v_interface;;
  vec_and_ptr_dim_t jump;

  // Navier-Stokes problem:-----------------------------
  my_p4est_navier_stokes_t* ns = NULL;

  PCType pc_face = PCSOR;
  KSPType face_solver_type = KSPBCGS;
  PCType pc_cell = PCSOR;
  KSPType cell_solver_type = KSPBCGS;

  vec_and_ptr_dim_t v_n;
  vec_and_ptr_dim_t v_nm1;

  vec_and_ptr_t vorticity;
  vec_and_ptr_t vorticity_refine;

  vec_and_ptr_t press_nodes;

  Vec dxyz_hodge_old[P4EST_DIM];

  my_p4est_cell_neighbors_t *ngbd_c_np1 = NULL;
  my_p4est_faces_t *faces_np1 = NULL;

  // Related to domain: -------------------------------
  double xyz_min[P4EST_DIM]; double xyz_max[P4EST_DIM];
  int ntrees[P4EST_DIM];
  bool periodicity[P4EST_DIM];

  // Number of fields to transfer between grids
  int num_fields_interp;


  // Related to time/timestepping: --------------------
  double tn;
  double dt;
  double dt_nm1;
  double dt_Stefan;
  double dt_NS;

  int tstep;

  double cfl_Stefan;
  double cfl_NS;

  // Related to dimensionalization type:
  int problem_dimensionalization_type;

  // Converting nondim to dim:
  double time_nondim_to_dim;
  double vel_nondim_to_dim;

  // Nondimensional groups
  double Re; // Reynolds number (rho Uinf l_char)/mu_l
  double Pr; // Prandtl number - (mu_l/(rho_l * alpha_l)) = (nu_l/alpha_l)
  double Sc; // Schmidt number - (mu_l/(rho_l * D)) = (nu_l/D)
  double Pe; // Peclet number - (Uint lchar)/alpha_l , or Re*Pr
  double St; // Stefan number (cp_s deltaT/L)
  double Da; // Damkohler number (k_diss*l_char/D_diss)
  double RaT; // Rayleigh number by temperature TO-DO: add definition
  double RaC; // Rayleigh number by concentration TO-DO: add definition

  // Physical parameters:
  double l_char; // Characteristic length scale (assumed in meters)

  double T0; // characteristic solid temperature of the problem
  double Tinterface; // Interface temperature or concentration
  double Tinfty; // Freestream fluid temperature or concentration
  double Tflush; // Flush temperature (K) or concentration that inlet BC is changed to if flush_dim_time is activated

  double alpha_l, alpha_s; // Liquid and solid thermal diffusivities, [m^2/s]
  double k_l, k_s; // Liquid and solid thermal conductivities, [W/(mK)]
  double rho_l, rho_s; // Liquid and solid densities, [kg/m^3]
  double cp_s; // Solid heat capacity [J/(kg K)]
  double L; // Latent heat of fusion [J/kg]
  double mu_l; // Fluid viscosity [Pa s]

  double grav; // Gravity
  double beta_T; // Thermal expansion coefficient for the boussinesq approx
  double beta_C; // Concentration expansion coefficient for the boussinesq approx

  double gamma_diss; // TO-DO: add description
  double stoich_coeff_diss; // The stoichiometric coefficient of the dissolution reaction
  double molar_volume_diss; // The molar volume of the dissolving solid

  double Dl, Ds; //Concentration diffusion coefficient m^2/s,
  double k_diss; // Dissolution rate constant per unit area of reactive surface (m/s)


  // Interp method: -------------------------------------
  interpolation_method interp_bw_grids = quadratic_non_oscillatory_continuous_v2;


  // Variables for extension band and grid size: ---------
  double dxyz_smallest[P4EST_DIM];
  double dxyz_close_to_interface;

  double min_volume_;
  double extension_band_use_;
  double extension_band_extend_;
  double extension_band_check_;

  // Variables for refining the fields
  int lmin, lint, lmax;
  double uniform_band;

  bool use_uniform_band;
  bool refine_by_d2T;
  double vorticity_threshold;
  double gradT_threshold;



};

#endif // STEFAN_WITH_FLUIDS_H
