#ifndef MY_P4EST_NS_FREE_SURFACE_H
#define MY_P4EST_NS_FREE_SURFACE_H

#ifdef P4_TO_P8
#include <src/my_p8est_navier_stokes.h>
#else
#include <src/my_p4est_navier_stokes.h>
#endif

typedef enum
{
  HODGE,
  PRESSURE,
  VELOCITY_X,
  VELOCITY_Y
#ifdef P4_TO_P8
  , VELOCITY_Z
#endif
} field_type;

class my_p4est_ns_free_surface_t : public my_p4est_navier_stokes_t
{
private:

#ifdef P4_TO_P8
  class wall_bc_value_t : public CF_3
    #else
  class wall_bc_value_t : public CF_2
    #endif
  {
  private:
    my_p4est_ns_free_surface_t *const  _prnt;
    const field_type which;
  public:
    wall_bc_value_t(my_p4est_ns_free_surface_t * const obj, const field_type which_) : _prnt(obj), which(which_) {}// wall_bc_value class constructs by free_surface class object? not sure
#ifdef P4_TO_P8
    double operator()( double x, double y, double z ) const
#else
    double operator()( double x, double y ) const
#endif
    {
      switch (which) {
      case PRESSURE:
#ifdef P4_TO_P8
        return _prnt->bc_pressure->wallValue(x, y, z);
#else
        return _prnt->bc_pressure->wallValue(x, y);
#endif
        break;
      case VELOCITY_X:
#ifdef P4_TO_P8
        return _prnt->bc_v[0].wallValue(x, y, z);
#else
        return _prnt->bc_v[0].wallValue(x, y);
#endif
        break;
      case VELOCITY_Y:
#ifdef P4_TO_P8
        return _prnt->bc_v[1].wallValue(x, y, z);
#else
        return _prnt->bc_v[1].wallValue(x, y);
#endif
        break;
#ifdef P4_TO_P8
      case VELOCITY_Z:
#ifdef P4_TO_P8
        return _prnt->bc_v[2].wallValue(x, y, z);
#else
        return _prnt->bc_v[2].wallValue(x, y);
#endif
        break;
#endif
      default:
        throw std::runtime_error("my_p4est_ns_free_surface_t::wall_bc_value_t::operator: unknown field type.");
        break;
      }
    }
  };

#ifdef P4_TO_P8
  class mixed_interface_bc_t : public CF_3, public mixed_interface
    #else
  class mixed_interface_bc_t : public CF_2, public mixed_interface
    #endif
  {
  private:
    my_p4est_ns_free_surface_t* const _prnt;
    const field_type which;
  public:
    mixed_interface_bc_t(my_p4est_ns_free_surface_t* const obj, field_type which_) : _prnt(obj), which(which_) {}
#ifdef P4_TO_P8
    double operator()( double x, double y, double z ) const;
#else
    double operator()( double x, double y ) const;
#endif
    double operator()( double xyz[] ) const {
      return this->operator ()(xyz[0], xyz[1]
    #ifdef P4_TO_P8
          , xyz[2]
    #endif
          );
    }

    /*!
     * \brief mixed_type
     * \param xyz: physical location of the point of interest in the domain
     * \return the boundary condition type to be locaclly considered
     * [NOTE :] intersections of the free surface and the solid should not be allowed but in case of
     * intersection, the solid interface's condition always has priority!
     */
    BoundaryConditionType mixed_type(const double xyz[]) const;
  };

#ifdef P4_TO_P8
  class ZERO : public CF_3
  {
  public:
    double operator ()(double, double, double) const {return 0.0;}
  } zero;
#else
  class ZERO : public CF_2
  {
  public:
    double operator ()(double, double) const {return 0.0;}
  } zero;
#endif

  Vec fs_phi;
  Vec global_phi;
  Vec sigma_kappa;
  Vec physical_bc[P4EST_DIM];
  Vec vel_bc_check[P4EST_DIM];
  bool use_physical_bc;

  double surf_tension, finest_diag;

#ifdef P4_TO_P8
  BoundaryConditions3D bc_pressure_global, bc_hodge_global, bc_v_global[P4EST_DIM];
#else
  BoundaryConditions2D bc_pressure_global, bc_hodge_global, bc_v_global[P4EST_DIM];
#endif
  wall_bc_value_t wall_bc_value_pressure, wall_bc_value_vx, wall_bc_value_vy;
#ifdef P4_TO_P8
  wall_bc_value_t wall_bc_value_vz;
#endif
  wall_bc_value_t* wall_bc_value_v[P4EST_DIM];

  mixed_interface_bc_t interface_bc_hodge_global, interface_bc_pressure_global, interface_bc_vx_global, interface_bc_vy_global;
#ifdef P4_TO_P8
  mixed_interface_bc_t interface_bc_vz_global;
#endif
  mixed_interface_bc_t* interface_bc_v_global[P4EST_DIM];


  my_p4est_interpolation_nodes_t *interp_fs_phi, *interp_global_phi, *interp_sigma_kappa;
  my_p4est_interpolation_nodes_t *interp_physical_bc_x, *interp_physical_bc_y;
#ifdef P4_to_P8
  my_p4est_interpolation_nodes_t *interp_physical_bc_z;
#endif


  void build_global_phi_and_face_is_well_defined(p4est_nodes_t* nodes,  Vec phi_, Vec fs_phi_, Vec global_phi_);

  void calculate_second_derivatives_for_fs_advection(Vec *fs_phi_xx, Vec** vxx_np1_nodes, Vec** vxx_n_nodes, bool second_order_fs_advection);

  /*!
   * \brief enforce_velocity_boundary_and_solid_interface_condition: TBD
   */
  void enforce_velocity_boundary_and_solid_interface_condition();

  /*!
   * \brief mark the faces that are well defined, i.e. that are solved for in an implicit poisson solve with irregular interface.
   *   For Dirichlet b.c. the condition is phi(face)<0. For Neumann, the control volume of the face must be at least partially in the negative domain.
   * \param faces the faces structure
   * \param dir the cartesian direction treated, dir::x, dir::y or dir::z
   * \param bc_v the boundary condition structure for the boundary condition on the interface
   * \param is_well_defined a Vector the size of the number of faces in direction dir, to be filled
   */
  void check_if_faces_are_well_defined_for_free_surface(my_p4est_faces_t *faces, int dir,
                                                      #ifdef P4_TO_P8
                                                        BoundaryConditions3D bc_v,
                                                      #else
                                                        BoundaryConditions2D bc_v,
                                                      #endif
                                                        Vec is_well_defined);

  /*!
   * \brief compute_dxyz_hodge: calculates the component of the gradient of the hodge variable on the face of a quadrant
   * \param quad_idx: local index of the quadrant on which we operate (cumulative over the trees)
   * \param tree_idx: tree index of the quadrant
   * \param face_idx: local face index (i.e. in [0 ... P4EST_FACES[) on which the gradient component needs to be calculated
   * \return the value of the component of the gradient on the face
   */
  double compute_dxyz_hodge(p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, int face_idx);

  void update_sigma_kappa();

public:
  my_p4est_ns_free_surface_t(my_p4est_node_neighbors_t *ngbd_nm1, my_p4est_node_neighbors_t *ngbd_n, my_p4est_faces_t *faces_n);
  ~my_p4est_ns_free_surface_t();

  void set_parameters(double mu_, double rho, int sl_order, double uniform_band, double threshold_split_cell, double n_times_dt, double surf_tension_)
  {
    surf_tension = surf_tension_;
    my_p4est_navier_stokes_t::set_parameters(mu_, rho, sl_order, uniform_band, threshold_split_cell, n_times_dt);
  }

  /*!
   * \brief set_phi/set_solid_phi: sets the solid interface
   * \param phi: node-sampled values of the solid levelset
   */
  void set_phi(Vec phi);
  void set_solid_phi(Vec solid_phi) {set_phi(solid_phi);}

  /*!
   * \brief set_free_surface: sets the free-surface interface
   * \param fs_phi_: node-sampled values of the free-surface levelset
   */
  void set_free_surface(Vec fs_phi_);

  /*!
   * \brief set_phis: sets the solid and free_surface interfaces
   * \param solid_phi_: node-sampled values of the solid-interface levelset
   * \param fs_phi_: node-sampled values of the free-surface-interface levelset
   */
  void set_phis(Vec solid_phi_, Vec fs_phi_);

  /*!
   * \brief set_bc: sets the wall and SOLID boundary conditions for velocity and pressure
   * \param bc_v: boundary conditions for velocity components
   * \param bc_p: boundary conditions for pressure
   * NOTE 1: the boundary conditions for the free surface are handled within the class itself!
   * NOTE 2: since we have no viscosity for now, bc_v MUST be such that
   * - velocity components normal to walls MUST be DIRICHLET;
   * - velocity components tangential to walls MUST be NEUMANN;
   * - velocity components in the solide MUST be DIRICHLET.
   */
#ifdef P4_TO_P8
  void set_bc(BoundaryConditions3D *bc_v, BoundaryConditions3D *bc_p);
#else
  void set_bc(BoundaryConditions2D *bc_v, BoundaryConditions2D *bc_p);
#endif

  inline my_p4est_interpolation_nodes_t* get_interp_global_phi() { return interp_global_phi; }
  inline Vec get_global_phi() { return global_phi; }

  /*!
   * \brief compute_max_L2_norm_u: computes the absolute maximum local velocity in the computational domain
   */
  void compute_max_L2_norm_u();

  /*!
   * \brief solve_viscosity: solves the viscosity step, i.e. time stepping from {v_n, v_nm1} to v_star
   */
  void solve_viscosity();

  /*!
   * \brief solve_projection: solves the projection step, i.e. laplace Hodge = -div(v_star)
   * The user-set boundary condition is used on the solid interface, a Dirichlet boundary condition surf_tension*curvature is used
   * on the free surface.
   */
  void solve_projection();


  /*!
   * \brief compute_velocity_at_nodes: interpolates vnp1 from faces to nodes and extrapolate over the free surface (2nd order, tvd extrapolation)
   */
  void compute_velocity_at_nodes();

  /*!
   * \brief update_from_tn_to_tnp1: self-explanatory
   * \param level_set: continuous function to define the solid level-set at time step tnp1
   * \param second_order_fs_advection: flag activating a second-order semilagrangian routine for the advection of the free surface
   * (the user should activate it only if vnp1 AND vn are reasonably good --> not for the first advection step if harsh initialization of velocity fields)
   * (since the free surface is advected, the grid changes at every time step and it needs to be re-built at each update)
   */
#ifdef P4_TO_P8
  void update_from_tn_to_tnp1(const CF_3 *level_set, bool second_order_fs_advection);
#else
  void update_from_tn_to_tnp1(const CF_2 *level_set, bool second_order_fs_advection);
#endif


  /*!
   * \brief compute_pressure: self-explanatory
   */
  void compute_pressure();

  /*!
   * \brief save_vtk: save the visual data in the directory of path name
   * \param name: path to the directory where the results are saved
   */
  void save_vtk(const char* name);

  void compute_physical_bc();
//  void set_vnp1_to_vn();

  void physical_bc_on();
  void save_vtk_hodge(const char* name);
  void compute_vel_bc_value();

};

#endif // MY_P4EST_NS_FREE_SURFACE_H
