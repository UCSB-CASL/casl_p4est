#include <src/CASL_math.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_nodes.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/casl_types.h>

#include <vector>

class my_p4est_level_set {

  my_p4est_brick_t *myb;
  p4est_t *p4est;
  p4est_nodes_t *nodes;
  p4est_ghost_t *ghost;
  my_p4est_node_neighbors_t *ngbd;

  /* order the nodes based on whether they are in another mpirank's ghost layer or not */
  std::vector<p4est_locidx_t> layer_nodes;
  std::vector<p4est_locidx_t> local_nodes;

  void compute_derivatives( Vec phi_petsc, Vec dxx_petsc, Vec dyy_petsc) const;

  void reinitialize_One_Iteration_First_Order( std::vector<p4est_locidx_t>& map, double *p0, double *pn, double *pnp1, double limit );

  void reinitialize_One_Iteration_Second_Order( std::vector<p4est_locidx_t>& map, const double *dxx0, const double *dyy0, const double *dxx, const double *dyy, double *p0, double *pn, double *pnp1, double limit );

  void advect_in_normal_direction_one_iteration(std::vector<p4est_locidx_t>& map, const double *vn, double dt, const double *dxx, const double *dyy, const double *pn, double *pnp1);
public:
  my_p4est_level_set(my_p4est_node_neighbors_t *ngbd_ )
    : myb(ngbd_->myb), p4est(ngbd_->p4est), nodes(ngbd_->nodes), ghost(ngbd_->ghost), ngbd(ngbd_)
  {
    for( p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n )
    {
      if(  (*ngbd)[n].node_m0_m >= nodes->num_owned_indeps ||
           (*ngbd)[n].node_m0_p >= nodes->num_owned_indeps ||
           (*ngbd)[n].node_p0_m >= nodes->num_owned_indeps ||
           (*ngbd)[n].node_p0_p >= nodes->num_owned_indeps ||
           (*ngbd)[n].node_0m_m >= nodes->num_owned_indeps ||
           (*ngbd)[n].node_0m_p >= nodes->num_owned_indeps ||
           (*ngbd)[n].node_0p_m >= nodes->num_owned_indeps ||
           (*ngbd)[n].node_0p_p >= nodes->num_owned_indeps )
        layer_nodes.push_back(n);
      else
        local_nodes.push_back(n);
    }
  }

  /* 1st order in time, 1st order in space */
  void reinitialize_1st_order( Vec phi_petsc, int number_of_iteration=20, double limit=DBL_MAX );

  /* 2nd order in time, 1st order in space */
  void reinitialize_2nd_order_time_1st_order_space( Vec phi_petsc, int number_of_iteration=20, double limit=DBL_MAX );

  /* 2nd order in time, 2nd order in space */
  /* this has not be thoroughly tested ... use with caution. It's also disastrous in terms of MPI communications */
  void reinitialize_2nd_order( Vec phi_petsc, int number_of_iteration=20, double limit=DBL_MAX );

  /*!
   * \brief advect_in_normal_direction advects the level-set function in the normal direction using Godunov's scheme
   * \param vn     [in]      velocity in the normal direction (CF2)
   * \param phi    [in, out] level-set function
   * \param phi_xx [in]      dxx derivative of level-set function. will be computed if set to NULL
   * \param phi_yy [in]      dyy derivative of level-set function. will be computed if set to NULL
   * \return dt
   */
  double advect_in_normal_direction(const CF_2& vn, Vec phi, Vec phi_xx = NULL, Vec phi_yy = NULL);

  /*!
   * \brief advect_in_normal_direction advects the level-set function in the normal direction using Godunov's scheme
   * \param vn     [in]      velocity in the normal direction (Vec)
   * \param phi    [in, out] level-set function
   * \param phi_xx [in]      dxx derivative of level-set function. will be computed if set to NULL
   * \param phi_yy [in]      dyy derivative of level-set function. will be computed if set to NULL
   * \return dt
   */
  double advect_in_normal_direction(const Vec vn, Vec phi, Vec phi_xx = NULL, Vec phi_yy = NULL);

  /* extrapolate using geometrical extrapolation */
  void extend_Over_Interface( Vec phi_petsc, Vec q_petsc, BoundaryConditions2D &bc, int order=2, int band_to_extend=INT_MAX ) const;

  /* extrapolate using geometrical extrapolation */
  void extend_Over_Interface( Vec phi_petsc, Vec q_petsc, BoundaryConditionType bc_type, Vec bc_vec, int order=2, int band_to_extend=INT_MAX ) const;

  /* extend a quantity from the interface */
  void extend_from_interface_to_whole_domain( Vec phi_petsc, Vec q_petsc, Vec q_extended_petsc, int band_to_extend=INT_MAX) const;
};
