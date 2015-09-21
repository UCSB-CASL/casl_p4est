#ifndef MY_P4EST_NAVIER_STOKES_H
#define MY_P4EST_NAVIER_STOKES_H

#include <petsc.h>

#ifdef P4_TO_P8
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_faces.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_interpolation_cells.h>
#include <src/my_p8est_interpolation_faces.h>
#else
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_faces.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_interpolation_cells.h>
#include <src/my_p4est_interpolation_faces.h>
#endif





class my_p4est_navier_stokes_t
{
private:

  class splitting_criteria_vorticity_t : public splitting_criteria_tag_t
  {
  private:
    void tag_quadrant(p4est_t *p4est, p4est_locidx_t quad_idx, p4est_topidx_t tree_idx,
                      my_p4est_interpolation_nodes_t &phi, my_p4est_interpolation_nodes_t &vor,
                      my_p4est_interpolation_nodes_t *smo);
  public:
    double max_L2_norm_u;
    double threshold;
    double uniform_band;
    double smoke_thresh;
    splitting_criteria_vorticity_t(int min_lvl, int max_lvl, double lip, double uniform_band, double threshold, double max_L2_norm_u, double smoke_thresh);
    bool refine_and_coarsen(p4est_t* p4est, my_p4est_node_neighbors_t *ngbd_n, Vec phi, Vec vorticity, Vec smoke);
  };

#ifdef P4_TO_P8
  class wall_bc_value_hodge_t : public CF_3
    #else
  class wall_bc_value_hodge_t : public CF_2
    #endif
  {
  private:
    my_p4est_navier_stokes_t* _prnt;
  public:
    wall_bc_value_hodge_t(my_p4est_navier_stokes_t* obj) : _prnt(obj) {}
#ifdef P4_TO_P8
    double operator()( double x, double y, double z ) const;
#else
    double operator()( double x, double y ) const;
#endif
  };

#ifdef P4_TO_P8
  class interface_bc_value_hodge_t : public CF_3
    #else
  class interface_bc_value_hodge_t : public CF_2
    #endif
  {
  private:
    my_p4est_navier_stokes_t* _prnt;
  public:
    interface_bc_value_hodge_t(my_p4est_navier_stokes_t* obj) : _prnt(obj) {}
#ifdef P4_TO_P8
    double operator()( double x, double y, double z ) const;
#else
    double operator()( double x, double y ) const;
#endif
  };

  my_p4est_brick_t *brick;

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
  my_p4est_cell_neighbors_t *ngbd_c;
  my_p4est_faces_t *faces_n;

  double dxyz_min[P4EST_DIM];
  double xyz_min[P4EST_DIM];
  double xyz_max[P4EST_DIM];
  double convert_to_xyz[P4EST_DIM];

  double mu;
  double rho;
  double dt_n;
  double dt_nm1;
  double max_L2_norm_u;
  double uniform_band;
  double threshold_split_cell;
  double n_times_dt;
  bool dt_updated;

  Vec phi;
  Vec hodge;
  Vec dxyz_hodge[P4EST_DIM];

  Vec vstar[P4EST_DIM];
  Vec vnp1 [P4EST_DIM];

  Vec vnm1_nodes[P4EST_DIM];
  Vec vn_nodes  [P4EST_DIM];
  Vec vnp1_nodes[P4EST_DIM];

  Vec vorticity;

  Vec pressure;

  Vec smoke;
#ifdef P4_TO_P8
  CF_3 *bc_smoke;
#else
  CF_2 *bc_smoke;
#endif
  bool refine_with_smoke;
  double smoke_thresh;

  Vec face_is_well_defined[P4EST_DIM];

#ifdef P4_TO_P8
  BoundaryConditions3D *bc_pressure;
  BoundaryConditions3D bc_hodge;
  BoundaryConditions3D *bc_v;
#else
  BoundaryConditions2D *bc_pressure;
  BoundaryConditions2D bc_hodge;
  BoundaryConditions2D *bc_v;
#endif

  wall_bc_value_hodge_t wall_bc_value_hodge;
  interface_bc_value_hodge_t interface_bc_value_hodge;

#ifdef P4_TO_P8
  CF_3 *external_forces[P4EST_DIM];
#else
  CF_2 *external_forces[P4EST_DIM];
#endif

  my_p4est_interpolation_nodes_t *interp_phi;

  double compute_dxyz_hodge( p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, int dir);

  double compute_divergence(p4est_locidx_t quad_idx, p4est_topidx_t tree_idx);

  void compute_max_L2_norm_u();

  void compute_vorticity();

  void compute_norm_grad_v();

public:
  my_p4est_navier_stokes_t(my_p4est_node_neighbors_t *ngbd_nm1, my_p4est_node_neighbors_t *ngbd_n, my_p4est_faces_t *faces_n);
  ~my_p4est_navier_stokes_t();

  void set_parameters(double mu, double rho, double uniform_band, double threshold_split_cell, double n_times_dt);

#ifdef P4_TO_P8
  void set_smoke(Vec smoke, CF_3 *bc_smoke, bool refine_with_smoke=true, double smoke_thresh=.5);
#else
  void set_smoke(Vec smoke, CF_2 *bc_smoke, bool refine_with_smoke=true, double smoke_thresh=.5);
#endif

  void set_phi(Vec phi);

#ifdef P4_TO_P8
  void set_external_forces(CF_3 **external_forces);
#else
  void set_external_forces(CF_2 **external_forces);
#endif

#ifdef P4_TO_P8
  void set_bc(BoundaryConditions3D *bc_v, BoundaryConditions3D *bc_p);
#else
  void set_bc(BoundaryConditions2D *bc_v, BoundaryConditions2D *bc_p);
#endif

  void set_velocities(Vec *vnm1, Vec *vn);

#ifdef P4_TO_P8
  void set_velocities(CF_3 **vnm1, CF_3 **vn);
#else
  void set_velocities(CF_2 **vnm1, CF_2 **vn);
#endif

  void set_vstar(Vec *vstar);

  void set_hodge(Vec hodge);

  inline double get_dt() { return dt_n; }

  inline my_p4est_node_neighbors_t* get_ngbd_n() { return ngbd_n; }

  inline p4est_t *get_p4est() { return p4est_n; }

  inline p4est_t *get_p4est_nm1() { return p4est_nm1; }

  inline p4est_ghost_t *get_ghost() { return ghost_n; }

  inline p4est_nodes_t *get_nodes() { return nodes_n; }

  inline my_p4est_faces_t* get_faces() { return faces_n; }

  inline Vec get_phi() { return phi; }

  inline Vec* get_velocity() { return vn_nodes; }

  inline Vec* get_velocity_np1() { return vnp1_nodes; }

  inline Vec* get_vstar() { return vstar; }

  inline Vec* get_vnp1() { return vnp1; }

  inline Vec get_hodge() { return hodge; }

  inline Vec get_smoke() { return smoke; }

  inline double get_max_L2_norm_u() { return max_L2_norm_u; }

  void solve_viscosity();

  void solve_projection();

  void compute_velocity_at_nodes();

  void set_dt(double dt_nm1, double dt_n);

  void set_dt(double dt_n);

  void compute_dt();

  void advect_smoke(my_p4est_node_neighbors_t *ngbd_np1, Vec *v, Vec smoke, Vec smoke_np1);

#ifdef P4_TO_P8
  void update_from_tn_to_tnp1(const CF_3 *level_set=NULL, bool convergence_test=false);
#else
  void update_from_tn_to_tnp1(const CF_2 *level_set=NULL, bool convergence_test=false);
#endif

  void compute_pressure();

  void compute_forces(double *f);

  void save_vtk(const char* name);
};



#endif /* MY_P4EST_NAVIER_STOKES_H */
