#ifndef MY_P4EST_SURFACTANT_H
#define MY_P4EST_SURFACTANT_H

#include <petsc.h>

#ifdef P4_TO_P8
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_general_poisson_nodes_mls_solver.h>
#include <src/my_p8est_poisson_nodes_mls.h>
#include <src/my_p8est_save_load.h>
#include <src/my_p8est_log_wrappers.h>
#else
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_general_poisson_nodes_mls_solver.h>
#include <src/my_p4est_poisson_nodes_mls.h>
#include <src/my_p4est_save_load.h>
#include <src/my_p4est_log_wrappers.h>
#endif

//typedef enum
//{
//  SAVE=3541,
//  LOAD
//} save_or_load;

class my_p4est_surfactant_t
{
protected:

  class band
  {
  private:
    my_p4est_surfactant_t* _prnt;
  public:
    double min_band_width = 4.0;
    double band_width;
    band(my_p4est_surfactant_t* obj, double b_width) : _prnt(obj), band_width(b_width)
    {
      if( band_width < min_band_width )
      {
        char error_msg[1024];
        sprintf(error_msg, "[WARNING] my_p4est_surfactant::band: The width of the interface band must be of at least %4.2f cell diagonals.", min_band_width);
        throw std::invalid_argument(error_msg);
      }
    }
    inline double band_fn( const double& phi_val ) const { return fabs(phi_val) - 0.5*band_width*(_prnt->dxyz_diag); }
  };

  class splitting_criteria_surfactant_t : public splitting_criteria_tag_t
  {
  private:
    my_p4est_surfactant_t* _prnt;
    void tag_quadrant(p4est_t *p4est, p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, p4est_nodes_t *nodes, const double *phi_p);
  public:
    double min_uniform_padding = 1.5;
    double uniform_padding;
    splitting_criteria_surfactant_t(my_p4est_surfactant_t* obj, int min_lvl, int max_lvl, double lip, double uniform_padding_input)
      : splitting_criteria_tag_t(min_lvl, max_lvl, lip), _prnt(obj), uniform_padding(uniform_padding_input)
    {
      if( uniform_padding < min_uniform_padding )
      {
        char error_msg[1024];
        sprintf(error_msg, "[WARNING] my_p4est_surfactant::splitting_criteria_surfactant_t: The width of the uniform padding must be of at least %4.2f cell diagonals.", min_uniform_padding);
        throw std::invalid_argument(error_msg);
      }
    }
    bool refine_and_coarsen(p4est_t* p4est, p4est_nodes_t* nodes, Vec phi);
  };

  // Brick and connectivity structures
  my_p4est_brick_t brick;
  p4est_connectivity_t *conn; // Need to declare it as p8est for 3D??

  // Tree structures at time step nm1 (n-1)
  p4est_t *p4est_nm1;
  p4est_ghost_t *ghost_nm1;
  p4est_nodes_t *nodes_nm1;
  my_p4est_hierarchy_t *hierarchy_nm1;
  my_p4est_node_neighbors_t *ngbd_nm1;

  // Tree structures at time step n
  p4est_t *p4est_n;
  p4est_ghost_t *ghost_n;
  p4est_nodes_t *nodes_n;
  my_p4est_hierarchy_t *hierarchy_n;
  my_p4est_node_neighbors_t *ngbd_n;

  // Flags to "turn off" different terms in the transport equations. They're set to false by default.
  bool STATIC_GRID;
  bool NO_SURFACE_DIFFUSION;
  bool NO_SURFACE_ADVECTION;

  // Numerical variables
  int ITER_EXTENSION;
  double TOL_BAND_NODES;

  // Problem lengthscales, stored during the whole life span of the solver
  double dxyz[P4EST_DIM];
  double dxyz_min, dxyz_diag;
  splitting_criteria_surfactant_t *sp_crit;

  double dt_n;
  double dt_nm1;
  double CFL;

  double D_s;

  Vec phi;
  Vec phi_band;
  band phi_band_gen;
  int num_band_nodes;

  Vec normal[P4EST_DIM];
  Vec kappa;
  double max_abs_kappa;

  Vec vn_nodes  [P4EST_DIM];
  Vec vnm1_nodes[P4EST_DIM];
  Vec vn_s_nodes  [P4EST_DIM];
  Vec vnm1_s_nodes[P4EST_DIM];
  double max_L2_norm_u;

  Vec str_n;
  Vec str_nm1;

  Vec dd_vn_nodes[P4EST_DIM][P4EST_DIM];
  Vec dd_vnm1_nodes[P4EST_DIM][P4EST_DIM];
  Vec dd_vn_s_nodes[P4EST_DIM][P4EST_DIM];
  Vec dd_vnm1_s_nodes[P4EST_DIM][P4EST_DIM];

  Vec Gamma_np1;
  Vec Gamma_n;
  Vec Gamma_nm1;

  // Semi-lagrangian backtraced points for nodes (needed in the discretization of the advection terms, needs to be done only once)
  // No need to destroy these, not dynamically allocated...
  bool sl_dep_pts_computed;
  std::vector<double> xyz_dep_n[P4EST_DIM];
  std::vector<double> xyz_dep_nm1[P4EST_DIM]; // used only if sl_order == 2

  bool sl_dep_pts_s_computed;
  std::vector<double> xyz_dep_s_n[P4EST_DIM];
  std::vector<double> xyz_dep_s_nm1[P4EST_DIM]; // used only if sl_order == 2

  void update_phi_band_vector();

  void enforce_refinement_p4est_n(CF_DIM *ls);

public:
  my_p4est_surfactant_t(const mpi_environment_t& mpi,
                        const double xyz_min_[], const double xyz_max_[], const int n_xyz_[], const int periodicity_[],
                        const int& lmin, const int& lmax,
                        CF_DIM *ls_n,
                        const double& CFL_input=1.0,
                        const double& uniform_padding_input=1.5,
                        const double& band_width=4.0);
  ~my_p4est_surfactant_t();

  void set_no_surface_diffusion(const bool &flag);
  void set_no_surface_advection(const bool &flag);

  void set_phi(Vec level_set, const bool& do_reinit=true);
  void set_phi(CF_DIM *level_set, const bool& do_reinit=true);
  void compute_num_band_nodes();

  void compute_normal();
  void compute_curvature();

  void set_Gamma(CF_DIM *Gamma_nm1_input, CF_DIM *Gamma_n_input);

  void set_velocities(CF_DIM **vnm1, CF_DIM **vn);

  void compute_dd_vn();
  void compute_dd_vnm1();

  void compute_dd_vn_s();
  void compute_dd_vnm1_s();

  void compute_vnm1_s_from_phi_nm1(CF_DIM *ls_nm1, const bool& do_reinit=true);
  void compute_vn_s();
  void compute_initial_extended_velocities(CF_DIM *ls_nm1, const bool& do_reinit_nm1=true);
  //void compute_extended_velocities(CF_DIM *ls_nm1, CF_DIM *ls_n=NULL, const bool& do_reinit_nm1=true, const bool& do_reinit_n=true);

  void compute_stretching_term_nm1();
  void compute_stretching_term_n();

  double compute_adapted_dt_n();

  void advect_interface_one_step();

  void compute_one_step_Gamma();

  void update_from_tn_to_tnp1(CF_DIM **vnp1);

  inline double get_dt_n() { return dt_n; }
  inline double get_dt_nm1() { return dt_nm1; }

  inline p4est_t* get_p4est_n() { return p4est_n; }
  inline p4est_t* get_p4est_nm1() { return p4est_nm1; }

  inline p4est_nodes_t* get_nodes_n() { return nodes_n; }
  inline p4est_nodes_t* get_nodes_nm1() { return nodes_nm1; }

  inline p4est_ghost_t* get_ghost_n() { return ghost_n; }
  inline p4est_ghost_t* get_ghost_nm1() { return ghost_nm1; }

  inline my_p4est_node_neighbors_t* get_ngbd_n() { return ngbd_n; }
  inline my_p4est_node_neighbors_t* get_ngbd_nm1() { return ngbd_nm1; }

  inline Vec get_phi() { return phi; }
  inline Vec get_phi_band() { return phi_band; }

  inline Vec get_Gamma_nm1() { return Gamma_np1; }
  inline Vec get_Gamma_n() { return Gamma_n; }
  inline Vec get_Gamma_np1() { return Gamma_np1; }

  inline double get_band_width_num_diag() { return phi_band_gen.band_width; }
  inline double get_band_width_distance() { return phi_band_gen.band_width*dxyz_diag; }

  inline double get_uniform_padding_num_diag() { return sp_crit->uniform_padding; }
  inline double get_uniform_padding_distance() { return sp_crit->uniform_padding*dxyz_diag; }

  inline double get_integrated_Gamma_intf() { return integrate_over_interface(p4est_n, nodes_n, phi, Gamma_n); }
  inline double get_average_Gamma_intf() { return get_integrated_Gamma_intf()/interface_length(p4est_n, nodes_n, phi); }

  inline double get_integrated_Gamma_band() { return integrate_over_negative_domain(p4est_n, nodes_n, phi_band, Gamma_n)/((phi_band_gen.band_width)*(dxyz_diag)); }
  inline double get_average_Gamma_band() { return get_integrated_Gamma_band()/interface_length(p4est_n, nodes_n, phi); }

  void set_D_s(const double& D_s);

  void set_dt_nm1(const double& dt_nm1);

  void set_dt_n(const double& dt_n);

  inline void set_dt(const double& dt_nm1, const double& dt_n) { set_dt_nm1(dt_nm1); set_dt_n(dt_n);}

  void save_vtk(const char* name);

  // TO-DO:
  //   - compute str_n on the fly and don't extend, it will be extended in rhs anyway
  //   - higher-order interpolation when computing rhs
  //   - eliminate double calculation of second derivatives
  //   - optimized interface advection (no sl class)
  //   - interface advection with extended velocity
  //   - need to store normals? If not, wipe
  //   - add 'face-is-well-defined' style vector to only perform computations in nodes close to the tube
  //   - add variable thickness scaling with the inverse of curvature (or at least a warning)
  //   - extend_from_interface_to_whole_domain_TVD only on a band
  //   - Add load/save capabilities
  //   - Add logging capabilities
  //   - Change semi-lagrangian of phi in ls advection so that it is done only on new nodes
  //   - rebalance-vector instead of delete and create again phi_np1 in advect_interface_one_step
};

#endif /* MY_P4EST_SURFACTANT_H */
