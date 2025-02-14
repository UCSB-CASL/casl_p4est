#ifndef EXAMPLES_FOR_STEFAN_WITH_FLUIDS_H
#define EXAMPLES_FOR_STEFAN_WITH_FLUIDS_H

#endif // EXAMPLES_FOR_STEFAN_WITH_FLUIDS_H


enum example_t:int {
  FRANK_SPHERE = 0,
  NS_GIBOU_EXAMPLE = 1,
  COUPLED_TEST_2 = 2,
  COUPLED_PROBLEM_EXAMPLE = 3,
  ICE_AROUND_CYLINDER = 4,
  FLOW_PAST_CYLINDER = 5,
  DENDRITE_TEST = 6,
  MELTING_ICE_SPHERE = 7,
  EVOLVING_POROUS_MEDIA = 8,
  PLANE_POIS_FLOW=9,
  DISSOLVING_DISK_BENCHMARK=10,
  MELTING_ICE_SPHERE_NAT_CONV=11,
  COUPLED_PROBLEM_WTIH_BOUSSINESQ_APP=12,
};

class example_case_for_stefan_with_fluids_t{
  public:
      example_t example_;
      // ----------------------------------------------------------------
      // Solver flag options:
      // ----------------------------------------------------------------

      bool solve_stefan, solve_navier_stokes;
      bool do_we_solve_for_Ts;
      bool use_boussinesq;   // Note: want to implement this in main file in such a way that these can be overwritten by the cmd line parameters

      bool is_dissolution_case;

      // ----------------------------------------------------------------
      // Flags to distinguish processes required depending on the example:
      // ----------------------------------------------------------------

      bool analytical_IC_BC_forcing_term; // example has analytical expression for IC/BC/forcing term(s)
      bool example_is_a_test_case; // meaning we output error files

      bool interfacial_temp_bc_requires_curvature; // temp problems using Gibbs-Thomson relation
      bool interfacial_temp_bc_requires_normal; // dendrite test

      bool interfacial_vel_bc_requires_vint; // interfacial BC on fluid velocity is dependent on stefan interfacial velocity

      bool example_uses_inner_LSF; // example uses a substrate (inner LSF)
      bool example_requires_area_computation; // We are interested in computing and outputting area info


      bool example_has_known_max_vint; // for analytical cases to enforce certain time-stepping

      // ----------------------------------------------------------------
      // Time-stepping and time discretization options:
      // ----------------------------------------------------------------
      double tn;
      double tfinal;

      double cfl_Stefan;
      double cfl_NS;

      int advection_sl_order; // advection order for temp/conc
      int NS_advection_order; // advection order for NS

      // ----------------------------------------------------------------
      // Porous media LSF options:
      // ----------------------------------------------------------------
      bool use_porous_media_substrate;
      double porous_media_initial_thickness_multiplier;
      int num_grains;

      // Specific to porous media example : xshifts, yshifts, rvals, and corresponding fxn. Maybe don't need to define in this big class.

      // ----------------------------------------------------------------
      // Grid refinement parameters:
      // ----------------------------------------------------------------
      bool use_uniform_band;
      double uniform_band;

      double vorticity_threshold;

      bool refine_by_d2T;
      double gradT_threshold;

      // ----------------------------------------------------------------
      // For solving Ts in time:
      // ----------------------------------------------------------------
      int Ts_soln_method; // Default is Backward Euler


      // ----------------------------------------------------------------
      // Problem dimensionalization type:
      // ----------------------------------------------------------------
      int nondim_type_used;


      // ----------------------------------------------------------------
      // Temperature/concentration variables:
      // ----------------------------------------------------------------
      // Dimensionless values used in code:
      double theta_infty;
      double theta_interface;
      double theta0;

      // User-specified values:
      double Tinfty;
      double Tinterface;
      double T0;

      // Function to define the theta values accordingly depending
      // on your nondimensionalization:
      virtual void set_nondim_temp_conc_values();
};


// EVOLVING POROUS MEDIA:
static class example_evolving_porous_media_t : public example_case_for_stefan_with_fluids_t{
  public:
      example_evolving_porous_media_t() : example_case_for_stefan_with_fluids_t(){
        example_ = EVOLVING_POROUS_MEDIA;
      }



} example_evolving_porous_media;


