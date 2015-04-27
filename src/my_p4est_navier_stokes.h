#ifndef MY_P4EST_NAVIER_STOKES_H
#define MY_P4EST_NAVIER_STOKES_H

#include <petsc.h>

#ifdef P4_TO_P8
#include <src/my_p8est_faces.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_interpolation_cells.h>
#include <src/my_p8est_interpolation_faces.h>
#else
#include <src/my_p4est_faces.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_interpolation_cells.h>
#include <src/my_p4est_interpolation_faces.h>
#endif





class my_p4est_navier_stokes_t
{
private:

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

#ifdef P4_TO_P8
  class wall_bc_value_vstar_t : public CF_3
    #else
  class wall_bc_value_vstar_t : public CF_2
    #endif
  {
  private:
    my_p4est_navier_stokes_t* _prnt;
    int dir;
  public:
    wall_bc_value_vstar_t(my_p4est_navier_stokes_t* obj, int dir) : _prnt(obj), dir(dir) {}
#ifdef P4_TO_P8
    double operator()( double x, double y, double z ) const;
#else
    double operator()( double x, double y ) const;
#endif
  };

#ifdef P4_TO_P8
  class interface_bc_value_vstar_t : public CF_3
    #else
  class interface_bc_value_vstar_t : public CF_2
    #endif
  {
  private:
    my_p4est_navier_stokes_t* _prnt;
    int dir;
  public:
    interface_bc_value_vstar_t(my_p4est_navier_stokes_t* obj, int dir) : _prnt(obj), dir(dir) {}
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

#ifdef P4_TO_P8
  Vec vorticity[P4EST_DIM];
#else
  Vec vorticity;
#endif

  Vec norm_grad_v;

  Vec face_is_well_defined[P4EST_DIM];

#ifdef P4_TO_P8
  BoundaryConditions3D bc_pressure;
  BoundaryConditions3D bc_hodge;
  BoundaryConditions3D bc_v[P4EST_DIM];
  BoundaryConditions3D bc_vstar[P4EST_DIM];
#else
  BoundaryConditions2D *bc_pressure;
  BoundaryConditions2D bc_hodge;
  BoundaryConditions2D *bc_v;
  BoundaryConditions2D bc_vstar[P4EST_DIM];
#endif

  wall_bc_value_hodge_t wall_bc_value_hodge;
  wall_bc_value_vstar_t *wall_bc_value_vstar[P4EST_DIM];
  interface_bc_value_hodge_t interface_bc_value_hodge;
  interface_bc_value_vstar_t *interface_bc_value_vstar[P4EST_DIM];

#ifdef P4_TO_P8
  CF_3 *external_forces;
#else
  CF_2 *external_forces;
#endif

  my_p4est_interpolation_nodes_t *interp_phi;

  my_p4est_interpolation_faces_t *interp_dxyz_hodge[P4EST_DIM];

  double compute_dxyz_hodge( p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, char dir);

  void compute_max_L2_norm_u();

  void compute_vorticity();

  void compute_norm_grad_v();

public:
  my_p4est_navier_stokes_t(my_p4est_node_neighbors_t *ngbd_nm1, my_p4est_node_neighbors_t *ngbd_n, my_p4est_faces_t *faces_n);
  ~my_p4est_navier_stokes_t();

  void set_parameters(double mu, double rho, double uniform_band, double threshold_split_cell, double n_times_dt);

  void set_phi(Vec phi);

#ifdef P4_TO_P8
  void set_external_forces(CF_3 *external_forces);
#else
  void set_external_forces(CF_2 *external_forces);
#endif

#ifdef P4_TO_P8
  void set_bc(BoundaryConditions3D *bc_v, BoundaryConditions3D *bc_p);
#else
  void set_bc(BoundaryConditions2D *bc_v, BoundaryConditions2D *bc_p);
#endif

  void set_velocities(Vec *vnm1, Vec *vn);

#ifdef P4_TO_P8
  void set_velocities(CF_3 *vnm1, CF_3 *vn);
#else
  void set_velocities(CF_2 *vnm1, CF_2 *vn);
#endif

  inline double get_dt() { return dt_n; }

  void solve_viscosity();

  void solve_projection();

  void compute_dt();

  void update_from_tn_to_tnp1();

  void compute_forces();

  void save_vtk(const char* name);
};



#endif /* MY_P4EST_NAVIER_STOKES_H */
