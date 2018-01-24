#ifndef MY_P4EST_LEVELSET_H
#define MY_P4EST_LEVELSET_H

#ifdef P4_TO_P8
#include <src/my_p8est_tools.h>
#include <src/my_p8est_nodes.h>
#include <src/my_p8est_node_neighbors.h>
#else
#include <src/my_p4est_tools.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_node_neighbors.h>
#endif
#include <src/types.h>

#include <vector>
#include <src/ipm_logging.h>
#include <src/math.h>

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
public:
  my_p4est_level_set_t(my_p4est_node_neighbors_t *ngbd_ )
    : myb(ngbd_->myb), p4est(ngbd_->p4est), nodes(ngbd_->nodes), ghost(ngbd_->ghost), ngbd(ngbd_)
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
  double advect_in_normal_direction(const Vec vn, Vec phi, double dt_max = DBL_MAX, Vec phi_xx = NULL, Vec phi_yy = NULL, Vec phi_zz = NULL);
#else
  double advect_in_normal_direction(const Vec vn, Vec phi, double dt_max = DBL_MAX, Vec phi_xx = NULL, Vec phi_yy = NULL);
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
  void extend_Over_Interface_TVD(Vec phi, Vec q, int iterations=20, int order=2) const;

  void extend_Over_Interface_TVD_not_parallel(Vec phi, Vec q, int iterations=20, int order=2) const;

  void extend_from_interface_to_whole_domain_TVD_one_iteration( const std::vector<int>& map, double *phi_p,
                                                                std::vector<double>& nx, std::vector<double>& ny,
                                                              #ifdef P4_TO_P8
                                                              std::vector<double>& nz,
                                                              #endif
                                                                double *q_out_p,
                                                                double *q_p, double *qxx_p, double *qyy_p,
                                                              #ifdef P4_TO_P8
                                                              double *qzz_p,
                                                              #endif
                                                                std::vector<double>& qi_m00, std::vector<double>& qi_p00,
                                                                std::vector<double>& qi_0m0, std::vector<double>& qi_0p0,
                                                              #ifdef P4_TO_P8
                                                                std::vector<double>& qi_00m, std::vector<double>& qi_00p,
                                                              #endif
                                                                std::vector<double>& s_m00, std::vector<double>& s_p00,
                                                                std::vector<double>& s_0m0, std::vector<double>& s_0p0
                                                              #ifdef P4_TO_P8
                                                                , std::vector<double>& s_00m, std::vector<double>& s_00p
                                                              #endif
                                                                ) const;
  void extend_from_interface_to_whole_domain_TVD( Vec phi, Vec q_interface, Vec q, int iterations=20 ) const;
};

#endif // MY_P4EST_LEVELSET_H
