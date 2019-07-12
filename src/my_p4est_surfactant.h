#ifndef MY_P4EST_SURFACTANT_H
#define MY_P4EST_SURFACTANT_H

#include <petsc.h>

#ifdef P4_TO_P8
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_interpolation_cells.h>
#include <src/my_p8est_interpolation_faces.h>
#include <src/my_p8est_poisson_cells.h>
#include <src/my_p8est_poisson_faces.h>
#include <src/my_p8est_save_load.h>
#else
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_interpolation_cells.h>
#include <src/my_p4est_interpolation_faces.h>
#include <src/my_p4est_poisson_cells.h>
#include <src/my_p4est_poisson_faces.h>
#include <src/my_p4est_save_load.h>
#endif

//typedef enum
//{
//  SAVE=3541,
//  LOAD
//} save_or_load;

class my_p4est_surfactant_t
{
protected:

#ifdef P4_TO_P8
  class band: public CF_3
#else
  class band: public CF_2
#endif
  {
  private:
    my_p4est_surfactant_t* _prnt;
  public:
    double band_width;
    band(my_p4est_surfactant_t* obj, double b_width) : _prnt(obj), band_width(b_width) {lip=1.2;}
#ifdef P4_TO_P8
    double operator()( double x, double y, double z ) const;
#else
    double operator()( double x, double y ) const;
#endif
  };

  class splitting_criteria_surfactant_t : public splitting_criteria_tag_t
  {
  private:
    my_p4est_surfactant_t* _prnt;
    void tag_quadrant(p4est_t *p4est, p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, my_p4est_interpolation_nodes_t &phi, band* band_p);
  public:
    splitting_criteria_surfactant_t(my_p4est_surfactant_t* obj, int min_lvl, int max_lvl, double lip)
      : splitting_criteria_tag_t(min_lvl, max_lvl, lip), _prnt(obj) {}
    bool refine_and_coarsen(p4est_t* p4est, my_p4est_node_neighbors_t *ngbd_n, Vec phi);
  };

  PetscErrorCode ierr;

  my_p4est_brick_t *brick;
  p4est_connectivity_t *conn;

  p4est_t *p4est_nm1;
  p4est_ghost_t *ghost_nm1;
  p4est_nodes_t *nodes_nm1;
  my_p4est_hierarchy_t *hierarchy_nm1;
  my_p4est_node_neighbors_t *ngbd_nm1;

  p4est_t *p4est_n;
  p4est_ghost_t *ghost_n;
  p4est_nodes_t *nodes_n;
  my_p4est_hierarchy_t *hierarchy_n;
  my_p4est_node_neighbors_t *ngbd_n;

  p4est_t *p4est_np1;
  p4est_ghost_t *ghost_np1;
  p4est_nodes_t *nodes_np1;
  my_p4est_hierarchy_t *hierarchy_np1;
  my_p4est_node_neighbors_t *ngbd_np1;

  splitting_criteria_cf_and_uniform_band_t* ref_data;

  bool NO_SURFACE_DIFFUSION;

  double dxyz[P4EST_DIM];
  double dxyz_min, dxyz_max;
  double xyz_min[P4EST_DIM];
  double xyz_max[P4EST_DIM];
  double convert_to_xyz[P4EST_DIM];

  double dt_n;
  double dt_nm1;
  double n_times_dt;
  bool   dt_updated;

  Vec phi;
  Vec phi_band;
  band phi_band_gen;

  Vec normal_n  [P4EST_DIM];

  Vec vn_nodes  [P4EST_DIM];
  Vec vnm1_nodes[P4EST_DIM];
  Vec vn_s_nodes  [P4EST_DIM];
  Vec vnm1_s_nodes[P4EST_DIM];

  Vec str_n;
  Vec str_nm1;

  Vec dd_vn_nodes[P4EST_DIM][P4EST_DIM];
  Vec dd_vnm1_nodes[P4EST_DIM][P4EST_DIM];
  Vec dd_vn_s_nodes[P4EST_DIM][P4EST_DIM];
  Vec dd_vnm1_s_nodes[P4EST_DIM][P4EST_DIM];

  Vec Gamma_np1;
  Vec Gamma_n;
  Vec Gamma_nm1;

    my_p4est_interpolation_nodes_t *interp_phi;

  // semi-lagrangian backtraced points for nodes (needed in the discretization of the advection terms, needs to be done only once)
  // no need to destroy these, not dynamically allocated...
  std::vector<double> xyz_dep_n[P4EST_DIM];
  std::vector<double> xyz_dep_nm1[P4EST_DIM]; // used only if sl_order == 2

  std::vector<double> xyz_dep_s_n[P4EST_DIM];
  std::vector<double> xyz_dep_s_nm1[P4EST_DIM]; // used only if sl_order == 2

  bool is_in_domain(const double xyz_[]) const;
  void update_phi_band_vector();

public:
  my_p4est_surfactant_t(my_p4est_node_neighbors_t *ngbd_nm1, my_p4est_node_neighbors_t *ngbd_n, const double& band_width=4.0);
#ifdef P4_TO_P8
  my_p4est_surfactant_t(const mpi_environment_t& mpi, my_p4est_brick_t *brick_input, p8est_connectivity *conn_input,
                          const int& lmin, const int& lmax, CF_3 *ls_n, CF_3 *ls_nm1=NULL, const bool& with_reinit=true, const double& band_width=5.0);
#else
  my_p4est_surfactant_t(const mpi_environment_t& mpi, my_p4est_brick_t *brick_input, p4est_connectivity *conn_input,
                          const int& lmin, const int& lmax, CF_2 *ls_n, CF_2 *ls_nm1=NULL, const bool& with_reinit=true, const double& band_width=5.0);
#endif
  ~my_p4est_surfactant_t();

  void set_grid_nm1(CF_2 *ls_nm1, const bool& is_ls_reinit=true);

  void set_no_surface_diffusion(const bool &flag);

  void set_phi(Vec level_set, const bool& do_reinit=true);

#ifdef P4_TO_P8
  void set_phi(CF_3 *level_set, const bool& do_reinit=true);
#else
  void set_phi(CF_2 *level_set, const bool& do_reinit=true);
#endif    

  void compute_initial_normals_from_phi();

  void compute_normal_np1();

#ifdef P4_TO_P8
  void set_Gamma(CF_3 *Gamma_nm1_input, CF_3 *Gamma_n_input);
#else
  void set_Gamma(CF_2 *Gamma_nm1_input, CF_2 *Gamma_n_input);
#endif

#ifdef P4_TO_P8
  void set_velocities(CF_3 **vnm1, CF_3 **vn);
#else
  void set_velocities(CF_2 **vnm1, CF_2 **vn);
#endif

#ifdef P4_TO_P8
  void compute_extended_velocities(CF_3 *ls_nm1, CF_3 *ls_n=NULL, const bool& do_reinit_nm1=true, const bool& do_reinit_n=true);
#else
  void compute_extended_velocities(CF_2 *ls_nm1, CF_2 *ls_n=NULL, const bool& do_reinit_nm1=true, const bool& do_reinit_n=true);
#endif

#ifdef P4_TO_P8
  void compute_stretching_term_nm1(CF_3 *ls_nm1);
  void compute_stretching_term_n(CF_3 *ls_nm1=NULL);
#else
  void compute_stretching_term_nm1(CF_2 *ls_nm1);
  void compute_stretching_term_n(CF_2 *ls_n=NULL);
#endif

  void compute_second_derivatives_for_interface_advection(Vec *dd_phi, Vec **dd_vn_nodes, Vec **dd_vnm1_nodes);
  void compute_second_derivatives_for_interface_advection(Vec *dd_phi);

  void advect_interface_one_step(bool second_order_SL = true);
  void advect_interface_one_step_TEST(bool second_order_SL = true);

  void compute_one_step_Gamma();

  inline double get_dt_n() { return dt_n; }

  inline double get_dt_nm1() { return dt_nm1; }

  void set_dt(const double& dt_nm1, const double& dt_n);

  void set_dt_n(const double& dt_n);

  void save_vtk(const char* name);

  // TO-DO:
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
