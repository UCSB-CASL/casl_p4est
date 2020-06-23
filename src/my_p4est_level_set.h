#ifndef MY_P4EST_LEVELSET_H
#define MY_P4EST_LEVELSET_H

#ifdef P4_TO_P8
#include <src/my_p8est_tools.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_poisson_nodes.h>
#else
#include <src/my_p4est_tools.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_poisson_nodes.h>
#endif
#include <src/types.h>

#include <vector>
#include <src/ipm_logging.h>
#include <src/casl_math.h>

class my_p4est_level_set_t {

  my_p4est_brick_t *myb;
  p4est_t *p4est;
  p4est_nodes_t *nodes;
  p4est_ghost_t *ghost;
  my_p4est_node_neighbors_t *ngbd;

#ifdef P4_TO_P8
  void compute_derivatives( Vec phi_petsc, Vec dxx_petsc, Vec dyy_petsc, Vec dzz_petsc) const;
  void compute_derivatives_above_threshold( Vec phi_petsc, double threshold, Vec dxx_petsc, Vec dyy_petsc, Vec dzz_petsc) const;
#else
  void compute_derivatives( Vec phi_petsc, Vec dxx_petsc, Vec dyy_petsc) const;
  void compute_derivatives_above_threshold( Vec phi_petsc, double threshold, Vec dxx_petsc, Vec dyy_petsc) const;
#endif

  void reinitialize_One_Iteration_First_Order( std::vector<p4est_locidx_t>& map, double *p0, double *pn, double *pnp1, double limit );
  void reinitialize_One_Iteration_First_Order_Above_Threshold( std::vector<p4est_locidx_t>& map, double *p0, double *pn, double *pnp1, double threshold, double limit );

  void reinitialize_One_Iteration_Second_Order( std::vector<p4est_locidx_t>& map,
                                              #ifdef P4_TO_P8
                                                const double *dxx0, const double *dyy0, const double *dzz0,
                                                const double *dxx,  const double *dyy,  const double *dzz,
                                              #else
                                                const double *dxx0, const double *dyy0,
                                                const double *dxx,  const double *dyy,
                                              #endif
                                                double *p0, double *pn, double *pnp1, double limit );

  void reinitialize_One_Iteration_Second_Order_Above_Threshold( std::vector<p4est_locidx_t>& map,
                                                                 #ifdef P4_TO_P8
                                                                   const double *dxx0, const double *dyy0, const double *dzz0,
                                                                   const double *dxx,  const double *dyy,  const double *dzz,
                                                                 #else
                                                                   const double *dxx0, const double *dyy0,
                                                                   const double *dxx,  const double *dyy,
                                                                 #endif
                                                                   double *p0, double *pn, double *pnp1, double threshold, double limit );

  void advect_in_normal_direction_one_iteration(std::vector<p4est_locidx_t>& map, const double *vn, double dt,
                                              #ifdef P4_TO_P8
                                                const double *dxx, const double *dyy, const double *dzz,
                                              #else
                                                const double *dxx, const double *dyy,
                                              #endif
                                                const double *pn, double *pnp1);

  // auxiliary flags and options
  interpolation_method interpolation_on_interface;
  bool                 use_neumann_for_contact_angle;
  int                  contact_angle_extension;
  bool                 show_convergence;
  double               show_convergence_band;
  bool                 use_two_step_extrapolation;

public:
  my_p4est_level_set_t(my_p4est_node_neighbors_t *ngbd_ )
    : myb(ngbd_->myb), p4est(ngbd_->p4est), nodes(ngbd_->nodes), ghost(ngbd_->ghost), ngbd(ngbd_),
      interpolation_on_interface(quadratic_non_oscillatory),
      use_neumann_for_contact_angle(true), contact_angle_extension(0),
      show_convergence(false), show_convergence_band(5.), use_two_step_extrapolation(false)
  {}

  inline void update(my_p4est_node_neighbors_t *ngbd_) {
    ngbd  = ngbd_;
    myb   = ngbd->myb;
    p4est = ngbd->p4est;
    nodes = ngbd->nodes;
    ghost = ngbd->ghost;
  }

  /* perturb the level set function by epsilon */
  void perturb_level_set_function( Vec phi_petsc, double epsilon );

  /* 2nd order in time, 1st order in space */
  void reinitialize_2nd_order_time_1st_order_space( Vec phi_petsc, int number_of_iteration=50, double limit=DBL_MAX );

  /* 1st order in time, 2nd order in space */
  void reinitialize_1st_order_time_2nd_order_space( Vec phi_petsc, int number_of_iteration=50, double limit=DBL_MAX );
  void reinitialize_1st_order_time_2nd_order_space_above_threshold( Vec phi_petsc, double threshold, int number_of_iteration=50, double limit=DBL_MAX );

  /* 1st order in time, 1st order in space */
  void reinitialize_1st_order( Vec phi_petsc, int number_of_iteration=50, double limit=DBL_MAX );
  void reinitialize_1st_order_above_threshold( Vec phi_petsc, double threshold, int number_of_iteration=50, double limit=DBL_MAX );

  /* 2nd order in time, 2nd order in space */
  /* this has not be thoroughly tested ... use with caution. It's also disastrous in terms of MPI communications */
  void reinitialize_2nd_order( Vec phi_petsc, int number_of_iteration=50, double limit=DBL_MAX );
  void reinitialize_2nd_order_above_threshold( Vec phi_petsc, double threshold, int number_of_iteration=50, double limit=DBL_MAX );


  /*!
   * \brief advect_in_normal_direction advects the level-set function in the normal direction using Godunov's scheme
   * \param vn     [in]      velocity in the normal direction (CF2)
   * \param phi    [in, out] level-set function
   * \param phi_xx [in]      dxx derivative of level-set function. will be computed if set to NULL
   * \param phi_yy [in]      dyy derivative of level-set function. will be computed if set to NULL
   * \return dt
   */
#ifdef P4_TO_P8
  double advect_in_normal_direction(const CF_3& vn, Vec phi, Vec phi_xx = NULL, Vec phi_yy = NULL, Vec phi_zz = NULL);
#else
  double advect_in_normal_direction(const CF_2& vn, Vec phi, Vec phi_xx = NULL, Vec phi_yy = NULL);
#endif

  /*!
   * \brief advect_in_normal_direction advects the level-set function in the normal direction using Godunov's scheme
   * \param vn     [in]      velocity in the normal direction (Vec)
   * \param phi    [in, out] level-set function
   * \param phi_xx [in]      dxx derivative of level-set function. will be computed if set to NULL
   * \param phi_yy [in]      dyy derivative of level-set function. will be computed if set to NULL
   * \return dt
   */
#ifdef P4_TO_P8
  double advect_in_normal_direction(const Vec vn, Vec phi, double dt, Vec phi_xx = NULL, Vec phi_yy = NULL, Vec phi_zz = NULL);
#else
  double advect_in_normal_direction(const Vec vn, Vec phi, double dt, Vec phi_xx = NULL, Vec phi_yy = NULL);
#endif

  /*!
   * \brief advect_in_normal_direction advects the level-set function in the normal direction using Godunov's scheme
   * (was it an attempt to have second order accurate advection? seems like it not entirely correct. Daniil)
   * \param vn     [in]      velocity in the normal direction (Vec)
   * \param vn_np1 [in]      velocity in the normal direction (Vec) at time n+1
   * \param phi    [in, out] level-set function
   * \param phi_xx [in]      dxx derivative of level-set function. will be computed if set to NULL
   * \param phi_yy [in]      dyy derivative of level-set function. will be computed if set to NULL
   * \return dt
   */
#ifdef P4_TO_P8
  double advect_in_normal_direction(const Vec vn, const Vec vn_np1, Vec phi, double dt, Vec phi_xx = NULL, Vec phi_yy = NULL, Vec phi_zz = NULL);
#else
  double advect_in_normal_direction(const Vec vn, const Vec vn_np1, Vec phi, double dt, Vec phi_xx = NULL, Vec phi_yy = NULL);
#endif

  /* extrapolate using geometrical extrapolation */
#ifdef P4_TO_P8
  void extend_Over_Interface( Vec phi_petsc, Vec q_petsc, BoundaryConditions3D &bc, int order=2, int band_to_extend=INT_MAX ) const;
#else
  void extend_Over_Interface( Vec phi_petsc, Vec q_petsc, BoundaryConditions2D &bc, int order=2, int band_to_extend=INT_MAX ) const;
#endif

  /* extrapolate using geometrical extrapolation */
  void extend_Over_Interface( Vec phi_petsc, Vec q_petsc, BoundaryConditionType bc_type, Vec bc_vec, int order=2, int band_to_extend=INT_MAX ) const;

  /* same as above except does not use boundary condition information */
  void extend_Over_Interface(Vec phi_petsc, Vec q_petsc, int order=2, int band_to_extend = INT_MAX ) const ;

  /* extend a quantity from the interface */
  void extend_from_interface_to_whole_domain( Vec phi_petsc, Vec q_petsc, Vec q_extended_petsc, int band_to_extend=INT_MAX) const;

  /* extend a quantity over the interface with the TVD algorithm */
  void extend_Over_Interface_TVD(Vec phi, Vec q, int iterations=20, int order=2,
                                 double tol=0.0, double band_use=-DBL_MAX, double band_extend=DBL_MAX, double band_check=DBL_MAX,
                                 Vec normal[P4EST_DIM]=NULL, Vec mask=NULL, boundary_conditions_t *bc=NULL,
                                 bool use_nonzero_guess=false, Vec q_n=NULL, Vec q_nn=NULL) const;

  /* extend a quantity over the interface with the TVD algorithm (all derivatives are extended, not just q_n and q_nn) */
  void extend_Over_Interface_TVD_Full(Vec phi, Vec q, int iterations=20, int order=2,
                                      double tol=0.0, double band_use=-DBL_MAX, double band_extend=DBL_MAX, double band_check=DBL_MAX,
                                      Vec normal[P4EST_DIM]=NULL, Vec mask=NULL, boundary_conditions_t *bc=NULL,
                                      bool use_nonzero_guess=false, Vec *q_d=NULL, Vec *q_dd=NULL) const;

  void extend_Over_Interface_TVD_not_parallel(Vec phi, Vec q, int iterations=20, int order=2) const;

  /*!
   * \brief extend_from_interface_to_whole_domain_TVD_one_iteration performs one iteration of extension in normal direction
   * \param map                   [in]
   * \param nx                    [in]
   * \param ny                    [in]
   * \param q_out_p               [out]
   * \param q_p                   [in]
   * \param qxx_p                 [in]
   * \param qyy_p                 [in]
   * \param map_grid_to_interface [in]  a look-up table that contains information about wheter a given node
   *                                    has interface points next to it
   * \param interface_directions  [in]
   * \param interface_distances   [in]
   * \param interface_values      [in]
   * \return
   */
  void extend_from_interface_to_whole_domain_TVD_one_iteration(const std::vector<int>& map, DIM(vector<double>& nx, vector<double>& ny, vector<double>& nz),
                                                               double *q_out_p, double *q_p, DIM(double *qxx_p, double *qyy_p, double *qzz_p),
                                                               std::vector<int>& map_grid_to_interface,
                                                               std::vector<int>& interface_directions,
                                                               std::vector<double>& interface_distances,
                                                               std::vector<double>& interface_values) const;

  /*!
   * \brief extend_from_interface_to_whole_domain_TVD extends a field from interface in normal direction (``flattening'')
   * \param phi         [in]  level-set function
   * \param qi          [in]  original field that needs to be ``flattened''
   *                          (can be NULL, but must provide func in this case)
   *                          (even when func is provided, qi can be used to set initial guess)
   * \param q           [out] ``flattened'' around interface field
   * \param mask        [in]  if not NULL, interface values where (mask > band_zero*diag_min) will be set to zero
   *                          (usefull when dealing with intersecting level-set function)
   * \param band_zero   [in]  the threshold defining which interface values will be set to zero
   * \param band_smooth [in]  transition width to zeros (in units of diag_min)
   * \param func        [in]  optional function to compute interface values
   * \return
   */
  void extend_from_interface_to_whole_domain_TVD(Vec phi, Vec qi, Vec q, int iterations=20,
                                                 Vec mask=NULL, double band_zero=2, double band_smooth=10,
                                                 double (*func)(p4est_locidx_t, int, double)=NULL) const;

  void enforce_contact_angle(Vec phi_wall, Vec phi_intf, Vec cos_angle, int iterations=20, Vec normal[] = NULL) const;
  void enforce_contact_angle2(Vec phi, Vec q, Vec cos_angle, int iterations=20, int order=2, Vec normal[] = NULL) const;
  void extend_Over_Interface_TVD_regional( Vec phi, Vec mask, Vec region, Vec q, int iterations = 20, int order = 2) const;

  double advect_in_normal_direction_with_contact_angle(const Vec vn, const Vec surf_tns, const Vec cos_angle, const Vec phi_wall, Vec phi, double dt);

  inline void extend_from_interface_to_whole_domain_TVD_in_place(Vec phi, Vec &q, Vec parent=NULL, int iterations=20, Vec mask=NULL) const
  {
    PetscErrorCode ierr;
    Vec tmp;

    if (parent == NULL) {
      ierr = VecCreateGhostNodes(p4est, nodes, &tmp); CHKERRXX(ierr);
    } else {
      ierr = VecDuplicate(parent, &tmp); CHKERRXX(ierr);
    }

    extend_from_interface_to_whole_domain_TVD(phi, q, tmp, iterations, mask);

    ierr = VecDestroy(q); CHKERRXX(ierr);
    q = tmp;
  }

  inline void set_interpolation_on_interface   (interpolation_method value) { interpolation_on_interface = value; }
  inline void set_use_neumann_for_contact_angle(bool   value) { use_neumann_for_contact_angle = value; }
  inline void set_contact_angle_extension      (int    value) { contact_angle_extension       = value; }
  inline void set_show_convergence             (bool   value) { show_convergence              = value; }
  inline void set_show_convergence_band        (double value) { show_convergence_band         = value; }
  inline void set_use_two_step_extrapolation   (bool   value) { use_two_step_extrapolation    = value; }
};

#endif // MY_P4EST_LEVELSET_H
