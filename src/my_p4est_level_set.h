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

struct data_for_geometric_extapolation
{
  data_for_geometric_extapolation(const double &signed_distance_first_sample_in_negative_domain, const double &signed_distance_second_sample_in_negative_domain)
    : x_sample{signed_distance_first_sample_in_negative_domain, signed_distance_second_sample_in_negative_domain}
  {
    sample_value[0] = sample_value[1] = NAN; // initialize sampled values to NAN --> enable useful P4EST_ASSERT checks in debug
  }
  data_for_geometric_extapolation(const double signed_distances_sample_in_negative_domain[2]) : data_for_geometric_extapolation(signed_distances_sample_in_negative_domain[0], signed_distances_sample_in_negative_domain[1]) {}

  size_t        list_idx;
  double        phi;
  bc_sample     interface_bc;
  const double  x_sample[2];
  double        sample_value[2];
};

inline unsigned char number_of_samples_across_the_interface_for_geometric_extrapolation(const unsigned char &degree, const BoundaryConditionType &interface_bc_type)
{
  switch (interface_bc_type) {
  case DIRICHLET:
    return degree;
    break;
  case NEUMANN:
  case MIXED:
    return MAX(degree, (unsigned char) 1);
    break;
  default:
    throw std::invalid_argument("number_of_samples_across_the_interface_for_geometric_extrapolation: unknown interface type of boundary condition (not implemented yet).");
    break;
  }
}

inline double geometric_extrapolation(const unsigned char &degree, const data_for_geometric_extapolation &data)
{
  double extrapolated_q;
  P4EST_ASSERT(data.phi > 0.0);
  switch (data.interface_bc.type) {
  case DIRICHLET:
  {
    // using Newton's divided differences, interpolant through interface value, q1 and q2:
    //
    //     x_sample[1]                     x_sample[0]                    0                                       phi
    // --------|--------------------------------|-------------------------|----------------------------------------|--------------------------> // oriented axis (along the normal to the interface)
    //  second sampled value          first sampled value             interface                          point where extrapolation
    //    sample_value[1]                sample_value[0]          interface_bc.value                          is queried
    //     required only                  required only            always required                    --> that's what we compute here
    //    if degree >= 2                  if degree >= 1
    //      := dif2                          := dif1                   := dif0
    //
    extrapolated_q = data.interface_bc.value; // == dif0
    if(degree >= 1)
    {
      P4EST_ASSERT(!ISNAN(data.sample_value[0]));
      const double dif01 = (data.sample_value[0] - data.interface_bc.value)/data.x_sample[0]; // == (dif1 - dif0)/(x_sample[0] - 0)
      extrapolated_q += data.phi*dif01;
      if(degree >= 2)
      {
        P4EST_ASSERT(!ISNAN(data.sample_value[1]));
        const double dif12  = (data.sample_value[1] - data.sample_value[0])/(data.x_sample[1] - data.x_sample[0]); // == (dif2 - dif1)/(x_sample[1] - x_sample[0])
        const double dif012 = (dif12 - dif01) /data.x_sample[1];
        extrapolated_q += data.phi*(data.phi - data.x_sample[0])*dif012;
      }
    }
    PetscErrorCode ierr = PetscLogFlops((degree > 0 ? 5 : 0) + (degree > 1 ? 13 : 0)); CHKERRXX(ierr);
  }
    break;
  case NEUMANN:
  {
    // We construct the polynomial through x_sample's and 0 that matches the prescribed derivative at the interface
    //
    //
    //     x_sample[1]                     x_sample[0]                    0                                       phi
    // --------|--------------------------------|-------------------------|----------------------------------------|--------------------------> // oriented axis (along the normal to the interface)
    //  second sampled value          first sampled value             interface                          point where extrapolation
    //    sample_value[1]                sample_value[0]          interface_bc.value                          is queried
    //     required only                 always required           required only if                 --> that's what we compute here
    //    if degree >= 2                                              degree >= 1
    //
    // The general polynomial is
    //
    // p(x) = sample_value[0] + interface_bc.value*(x - x_sample[0]) + ((sample_value[1] - sample_value[0])/(x_sample[1]^2 - x_sample[0]^2) - interface_bc.value/(x_sample[1] + x_sample[0]))*(x^2 - x_sample[0]^2)
    //
    P4EST_ASSERT(!ISNAN(data.sample_value[0]));
    extrapolated_q = data.sample_value[0];
    if(degree >= 1)
    {
      extrapolated_q += (data.phi - data.x_sample[0])*data.interface_bc.value;
      if(degree >= 2)
      {
        P4EST_ASSERT(!ISNAN(data.sample_value[1]));
        extrapolated_q += ((data.sample_value[1] - data.sample_value[0])/(SQR(data.x_sample[1]) - SQR(data.x_sample[0])) - data.interface_bc.value/(data.x_sample[0] + data.x_sample[1]))*(SQR(data.phi) - SQR(data.x_sample[0]));
      }
    }
    PetscErrorCode ierr = PetscLogFlops(1 + (degree > 0 ? 4 : 0) + (degree > 1 ? 19 : 0)); CHKERRXX(ierr);
  }
    break;
  default:
    throw std::invalid_argument("compute_geometric_extrapolation: unknown interface type");
    break;
  }
  return extrapolated_q;
}

inline double build_extrapolation_data_and_compute_geometric_extrapolation(data_for_geometric_extapolation& data, const unsigned degree, const unsigned char &nsamples,
                                                                           const std::vector<double> &qsamples, const std::vector<bc_sample>* calculated_bc_samples,
                                                                           const std::vector<double> *calculated_correction_dirichlet_bc = NULL)
{
  const size_t list_idx = data.list_idx;
  if(calculated_bc_samples == NULL)
  {
    P4EST_ASSERT(data.interface_bc.type  = DIRICHLET); // this should have been set before hand if so
    data.interface_bc.value = qsamples[(nsamples + (calculated_bc_samples == NULL))*list_idx];
  }
  else
  {
    data.interface_bc = (*calculated_bc_samples)[list_idx];
    if(calculated_correction_dirichlet_bc != NULL && data.interface_bc.type == DIRICHLET)
      data.interface_bc.value += (*calculated_correction_dirichlet_bc)[list_idx];
  }
  if(nsamples >= 1)
    data.sample_value[0] = qsamples[(nsamples + (calculated_bc_samples == NULL))*list_idx + (calculated_bc_samples == NULL)];
  if(nsamples >= 2)
    data.sample_value[1] = qsamples[(nsamples + (calculated_bc_samples == NULL))*list_idx + 1 + (calculated_bc_samples == NULL)];

  return geometric_extrapolation(degree, data);
}

inline void add_dof_to_extrapolation_map(std::map<p4est_locidx_t, data_for_geometric_extapolation> &extrapolation_map, const p4est_locidx_t &dof_local_idx, const double *xyz_dof,
                                         const double &phi_dof, double *grad_phi_dof, const unsigned char nsamples, const double phi_level[2],
                                         my_p4est_interpolation_t *interp_q, my_p4est_interpolation_t *interp_bc, my_p4est_interpolation_t *interp_correction_dirichlet_bc = NULL)
{
  P4EST_ASSERT(phi_dof > 0.0); // sanity check in debug, can't be in negative domain
  const double mag_grad_phi = sqrt(SUMD(SQR(grad_phi_dof[0]), SQR(grad_phi_dof[1]), SQR(grad_phi_dof[2])));
  if(mag_grad_phi > EPS)
  {
    data_for_geometric_extapolation local_extrapolation_data(phi_level[0], phi_level[1]);
    local_extrapolation_data.list_idx  = extrapolation_map.size();
    local_extrapolation_data.phi       = phi_dof;
    double xyz_interface[P4EST_DIM] = {DIM(xyz_dof[0], xyz_dof[1], xyz_dof[2])};
    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
      grad_phi_dof[dim] /= mag_grad_phi;
      xyz_interface[dim] -= local_extrapolation_data.phi*grad_phi_dof[dim];
    }

    if(interp_bc != NULL)
    {
      interp_bc->add_point(extrapolation_map.size(), xyz_interface);
      if(interp_correction_dirichlet_bc != NULL) // sample at all interface points if data is given since we don't know the boundary condition type yet --> we'll need to check later on if we use the value or not
        interp_correction_dirichlet_bc->add_point(extrapolation_map.size(), xyz_interface);
    }
    else
    {
      local_extrapolation_data.interface_bc.type = DIRICHLET; // we use a DIRICHLET-like interpolant on the considered_zero_level-level of the levelset
      interp_q->add_point((nsamples + (interp_bc == NULL))*extrapolation_map.size(), xyz_interface);
    }

    // sample q value at level phi_level[0] (supposedly/hopefully in the negative domain)
    if(nsamples >= 1)
    {
      P4EST_ASSERT(phi_level[0] < 0.0);
      double qsampling_xyz[P4EST_DIM];
      for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
        qsampling_xyz[dim] = xyz_interface[dim] + grad_phi_dof[dim]*phi_level[0];
      interp_q->add_point((nsamples + (interp_bc == NULL))*extrapolation_map.size() + (interp_bc == NULL), qsampling_xyz);
    }

    // sample q value at level phi_level[1] (supposedly/hopefully in the negative domain)
    if(nsamples >= 2)
    {
      P4EST_ASSERT(phi_level[1] < 0.0);
      double qsampling_xyz[P4EST_DIM];
      for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
        qsampling_xyz[dim] = xyz_interface[dim] + grad_phi_dof[dim]*phi_level[1];
      interp_q->add_point((nsamples + (interp_bc == NULL))*extrapolation_map.size() + 1 + (interp_bc == NULL), qsampling_xyz);
    }

    std::pair<std::map<p4est_locidx_t, data_for_geometric_extapolation>::iterator, bool> ret;
    ret = extrapolation_map.insert(std::pair<p4est_locidx_t, data_for_geometric_extapolation>(dof_local_idx, local_extrapolation_data));
    P4EST_ASSERT(ret.second); // check that it was indeed sucessfully inserted in debug!
  }
}

class my_p4est_level_set_t {

  my_p4est_brick_t *myb;
  p4est_t *p4est;
  p4est_nodes_t *nodes;
  p4est_ghost_t *ghost;
  my_p4est_node_neighbors_t *ngbd;
  const double tree_dimensions[P4EST_DIM], zero_distance_threshold;

  void compute_derivatives(Vec phi, Vec phi_xxyyzz[P4EST_DIM], const u_int& blocksize = 1) const;

  /*!
   * \brief reinitialize_one_iteration performs one forward pseudotime-step of the reinitialization
   * algorithm in a list of nodes : phi_np1 = phi_n - dt*sign(phi_0)*(norm of gradient of phi_n - 1).
   * \param [out] phi_np1_p         : pointer to double array of local data of phi_np1;
   * \param [in] list_of_node_idx   : standard vector of node indices in which the operations need to be executed (typically lists local or layer nodes);
   * \param [in] phi_0_p            : pointer to const double array of local data of phi_0;
   * \param [in] phi_n_p            : pointer to const double array of local data of phi_n;
   * \param [in] phi_0_limit_high   : upper threshold value on phi_0 to trigger calculation of local reinitialization (see description here below)
   * \param [in] phi_0_limit_low    : lower threshold value on phi_0 to trigger calculation of local reinitialization (see description here below)
   * \param [in] phi_0_xxyyzz_p     : array of P4EST_DIM pointers to const double arrays of local data to second derivatives of phi_0 along the p4EST_DIM Cartesian directions
   *                                  (--> for interface localization and interface anchoring using Smereka's fix with higher order of accuracy, disregarded if NULL)
   * \param [in] phi_n_xxyyzz_p     : array of P4EST_DIM pointers to const double arrays of local data to second derivatives of phi_n along the p4EST_DIM Cartesian directions
   *                                  (--> for minmod correction to one-sided evaluated of gradients of phi_n and higher spatial accuracy, disregarded if NULL)
   *
   * -----------------------------
   * Description of the algorithm:
   * -----------------------------
   * For every node of index n in list_of_node_idx, the algorithm executes
   * /
   * |  if fabs(phi_0_p[n] < zero_distance_threshold)
   * |      --> set phi_np1[n] = 0.0; (anchor the interface at that node)
   * |  else if phi_0_limit_low < phi_0_p[n] < phi_0_limit_high
   * |      --> set phi_np1[n] = phi_n[n] - dt*(Godunov Hamiltonian of (norm of grad of phi_n - 1.0)) // see here below for more details about that
   * |      (+ force an inversion of sign of phi_np1[n] if it has unexepectedly changed sign compared to phi_0[n])
   * |  else
   * |      --> set phi_np1[n] = phi_0[n];
   * \
   * - The pseudotime-step dt is not constant in space but adapted to the local cell sizes around the point of interest,
   * it is locally set to MIN(dx^{-}, dx^{+}, dy^{-}, dy^{+} [, dz^{-}, dz^{+}])/P4EST_DIM.
   * - The Godunov Hamiltonian of (norm of grad of phi_n - 1.0) is calculated as follows:
   * 1) for every cartesian direction xyz, the partial derivative of phi_n with respect to xyz is calculated using one-sided
   *    difference formulas. Let dphin_dxyz_p and dphin_dxyz_m be the corresponding values results obtained using forward
   *    and backward differences respectively.
   *    Note 1: second-order minmod corrections are calculated if the second derivatives are provided to the function;
   *    Note 2: for nodes whose direct neighbors are across the interface, Smeraka's fix is used to anchor the levelset.
   *            The interface location between the two nodes of interest is calculated by finding the root of the
   *            - linear interpolant, if second-order derivatives aren't provided;
   *            - quadratic interpolant matching the appropriate (minmod) second derivative, otherwise.
   * 2) For every one of those derivative components, the sign of corresponding "advection velocity" of the reinitialization
   * equation is sgn(phi_0[n])*dphin_dxyz_*. Therefore,
   *    A) dphin_dxyz_p is a valid numerical approximation to consider for capturing a potential flow of characteristics
   *       only if sgn(phi_0[n])*dphin_dxyz_p is negative. If not, we deactivate it by overwriting it with 0.0;
   *    B) dphin_dxyz_m is a valid numerical approximation to consider for capturing a potential flow of characteristics
   *       only if sgn(phi_0[n])*dphin_dxyz_m is positive. If not, we deactivate it by overwriting it with 0.0.
   * (note : although unlikely, it is _not_ impossible to have either both dphin_dxyz_p and dphin_dxyz_m nonzero after this
   * step or both dphin_dxyz_p and dphin_dxyz_m zero, especially when dealing with a local kink)
   * 3) The Godunov Hamiltonian of (norm of grad of phi_n - 1.0) is then evaluated as
   *    sgn*(sqrt(MAX(SQR(dphin_dx_p), SQR(dphin_dx_m)) + MAX(SQR(dphin_dy_p), SQR(dphin_dy_m)) [+ MAX(SQR(dphin_dz_p), SQR(dphin_dz_m))]) - 1.0);
   */
  void reinitialize_one_iteration(double *phi_np1_p, const std::vector<p4est_locidx_t> &list_of_node_idx,
                                  const double *phi_0_p, const double *phi_n_p,
                                  const double &phi_0_limit_high, const double &phi_0_limit_low,
                                  const double *phi_0_xxyyzz_p[P4EST_DIM], const double *phi_n_xxyyzz_p[P4EST_DIM]) const;

  /*!
   * \brief reinitialize_within_range_of_phi_0 reinitializes the given node-sampled levelset function with
   * - linear/quadratic interface localization in space;
   * - Forward Euler or 2nd order explicit TVD Runge-Kutta steps in pseudo-time (with adaptive time-stepping);
   * \param [inout] phi               : on input, node-sampled value of the levelset function whose reinitialization is desired
   *                                    on output, results of the reinitialization algorithm described here below
   * \param [in] order_space          : desired order of accuracy in space. Accepted values are 1 or 2
   *                                   --> 1 : interface location as the root of linear interpolants and no quadratic minmod correction to gradients of phi
   *                                   --> 2 : interface location as the root of quadratic interpolants and use of quadratic minmod corrections to gradients of phi
   * \param [in] order_pseudotime     : desired order of accuracy in pseudotime. Accepted values are 1 or 2
   *                                   --> 1 : Explicit Forward Euler steps
   *                                   --> 2 : 2nd order explicit TVD Runge-Kutta steps
   * \param [in] phi_0_limit_high     : upper threshold value on phi_0 to trigger calculation of local reinitialization (see description of reinitialize_one_iteration)
   * \param [in] phi_0_limit_low      : lower threshold value on phi_0 to trigger calculation of local reinitialization (see description of reinitialize_one_iteration)
   * \param [in] number_of_iterations : number of Forward pseudo-time steps to be executed
   * \param [in] layer_nodes_p		  : optional pointer to a list of layer nodes that are updatable.
   * \param [in] local_nodes_p		  : optional pointer to a list of local nodes (not in shared layer) that are updatable.
   * \param [in] masked_layer_nodes_p : optional pointer to a list of layer nodes that are NOT updatable (i.e., masked).
   * \param [in] masked_local_nodes_p : optional pointer to a list of local nodes (not in shared layer) that are NOT updatable.
   * \note If at least one of the custom pointer to lists of layer/local nodes is present, then, all 4 lists must be given.
   * -----------------------------
   * Description of the algorithm:
   * -----------------------------
   * 0) create  phi_np1;
   *    create  phi_np2 if second accuracy in pseudotime is desired
   *    copy    phi into phi_0 for interface anchoring
   *    if second order accuracy in space is desired,
   *      create P4EST_DIM node-sampled vectors phi_0_xxyyzz[0, 1 {, 2}] for second serivatives of phi_0 and calculated them (more accuate interafce anchoring)
   *      create P4EST_DIM node-sampled vectors current_phi_xxyyzz[0, 1 {, 2}] for second serivatives of phi_n (minmod corrections in one-sided finite difference formulas)
   * 1) do number_of_iterations times:
   *    /
   *    | - if second order accuracy in space is desired, calculate second derivatives of phi_n and store into current_phi_xxyyzz
   *    | - call reinitialize_one_iteration(...) to calculate phi_np1 from phi_n (providing current_phi_xxyyzz as well, if second order accuracy in space is desired)
   *    |   and using phi_0 (and its second derivatives, if second order accuracy in space is desired) for interface anchoring
   *    | if second order accuracy in pseudo-time is desired,
   *    |   /
   *    |   | - if second order accuracy in space is desired, calculate second derivatives of phi_np1 and store into current_phi_xxyyzz
   *    |   | - call reinitialize_one_iteration(...) to calculate phi_np2 from phi_np1 (providing current_phi_xxyyzz as well, if second order accuracy in space is desired)
   *    |   |   and using phi_0 (and its second derivatives, if second order accuracy in space is desired) for interface anchoring
   *    |   | - set phi_n = 0.5*(phi_n + phi_np2)
   *    |   \
   *    | otherwise,
   *    |   - set phi_n = phi_np1
   *    \
   * 2) destroy locally created data and return;
   */
  void reinitialize_within_range_of_phi_0(Vec phi, const unsigned char order_space, const unsigned char order_pseudotime,
                                          const double &phi_0_limit_high, const double &phi_0_limit_low, const int &number_of_iterations,
                                          const std::vector<p4est_locidx_t> *layer_nodes_p=nullptr,
                                          const std::vector<p4est_locidx_t> *local_nodes_p=nullptr,
                                          const std::vector<p4est_locidx_t> *masked_layer_nodes_p=nullptr,
                                          const std::vector<p4est_locidx_t> *masked_local_nodes_p=nullptr ) const;


  void advect_in_normal_direction_one_iteration(const std::vector<p4est_locidx_t> &list_of_node_idx, const double *vn, const double &dt,
                                                const double *phi_n_xxyyzz_p[P4EST_DIM], const double *phi_n_p, double *phi_np1_p) const;

  void advect_in_normal_direction(Vec phi, const double &dt, DIM(Vec phi_xx, Vec phi_yy, Vec phi_zz), const double *node_sampled_vn_p, const double *node_sampled_vnp1_p = NULL) const;

  inline double geometric_extrapolation_local(data_for_geometric_extapolation& data, const unsigned char &degree, const unsigned char &nsamples_across,
                                              const unsigned char &nsamples_in_considered_negative_domain, const std::vector<double> &qsamples,
                                              const BoundaryConditionsDIM* bc, const std::vector<bc_sample>* calculated_bc_samples) const
  {
    const size_t list_idx = data.list_idx;
    if(bc == NULL)
      data.interface_bc.value = qsamples[nsamples_across*list_idx];
    else
    {
      P4EST_ASSERT(calculated_bc_samples != NULL);
      data.interface_bc = (*calculated_bc_samples)[list_idx];
    }
    if(nsamples_in_considered_negative_domain > 0)
      data.sample_value[0] = qsamples[nsamples_across*list_idx + (bc == NULL)];
    if(nsamples_in_considered_negative_domain > 1)
      data.sample_value[1] = qsamples[nsamples_across*list_idx + 1 + (bc == NULL)];
    return geometric_extrapolation(degree, data);
  }

  /*!
   * \brief geometric_extrapolation_over_interface extrapolates a node-sampled field q from the negative to positive domain
   * associated with the node-sampled levelset function phi.
   * \param [in] phi            : vector of node-sampled levelset values;
   * \param [inout] q           : vector of node-sampled field values to extrapolate, values are unchanged in the negative domain;
   * \param [in] bc             : pointer to a boundary condition object, used for determining and using the type of interface
   *                              boundary (disregarded if NULL);
   * \param [in] degree         : desired degree of the interpolant to be build (0, 1 or 2);
   * \param [in] band_to_extend : values of q are calculated and set only where 0 < phi < band_to_extend*(smallest diagonal).
   * Values at nodes such that 0 < phi < band_to_extend*(smallest diagonal) and where the normal is well-defined are calculated
   * by building an interpolant of desired degree along the normal calculated at the node of interest, passing by the nodes
   * 0) the interface condition point projected on the interface with priority 0 if it is locally DIRICHLET, 1 if it is locally NEUMANN
   * 1) the node 2*diag away from the interface along the normal with priority 1 if it is locally DIRICHLET, 0 if it is locally NEUMANN
   * 2) the node 3*diag away from the interface along the normal with priority 2 if it is locally DIRICHLET, 2 if it is locally NEUMANN
   * if the boundary condition is provided (i.e. if bc != NULL). Otherwise, the value is set by building an interpolant of desired
   * degree along the normal calculated at the node of interest, passing by the nodes
   * 0) the node 2*diag away from the interface along the normal with priority 0
   * 1) the node 3*diag away from the interface along the normal with priority 1
   * 2) the node 4*diag away from the interface along the normal with priority 2
   * considering a DIRICHLET-like "interface" which would correspond to the -2*diag level of the original levelset function (so that no
   * interface boundary condition is required).
   * */
  void geometric_extrapolation_over_interface(Vec phi, Vec q, const BoundaryConditionsDIM *bc, const unsigned char &degree, const unsigned int &band_to_extend) const;


  // auxiliary flags and options
  interpolation_method interpolation_on_interface;
  bool                 use_neumann_for_contact_angle;
  int                  contact_angle_extension;
  double               bc_rel_thresh;
  bool                 use_two_step_extrapolation;

public:
  /*!
   * \brief my_p4est_level_set_t constructor using a nonconstant node-neighborhood object. If the node neighbors are not, initialized,
   * this constructor will initialize them (reinitializations, extensions, extrapolations, etc. are *much* faster if the node neighbors
   * are initialized!)
   * \param ngbd_ pointer to a my_p4est_node_neighbors object, this constructor will initialize the node neighborhood information
   */
  my_p4est_level_set_t(my_p4est_node_neighbors_t* ngbd_)
    : myb(ngbd_->myb), p4est(ngbd_->p4est), nodes(ngbd_->nodes), ghost(ngbd_->ghost), ngbd(ngbd_),
      tree_dimensions{DIM((ngbd_->get_brick()->xyz_max[0] - ngbd_->get_brick()->xyz_min[0])/ngbd_->get_brick()->nxyztrees[0], (ngbd_->get_brick()->xyz_max[1] - ngbd_->get_brick()->xyz_min[1])/ngbd_->get_brick()->nxyztrees[1], (ngbd_->get_brick()->xyz_max[2] - ngbd_->get_brick()->xyz_min[2])/ngbd_->get_brick()->nxyztrees[2])},
      zero_distance_threshold(EPS*MIN(DIM((ngbd_->get_brick()->xyz_max[0] - ngbd_->get_brick()->xyz_min[0])/ngbd_->get_brick()->nxyztrees[0], (ngbd_->get_brick()->xyz_max[1] - ngbd_->get_brick()->xyz_min[1])/ngbd_->get_brick()->nxyztrees[1], (ngbd_->get_brick()->xyz_max[2] - ngbd_->get_brick()->xyz_min[2])/ngbd_->get_brick()->nxyztrees[2]))),
      interpolation_on_interface(quadratic_non_oscillatory_continuous_v2),
      use_neumann_for_contact_angle(true), contact_angle_extension(0),
      bc_rel_thresh(1.e-8), use_two_step_extrapolation(false)
  {}

   //11/22/21 -- Elyce and Rochi merge -- commented out the below code, because show_convergence is no longer part of  this function in the updated version from Daniil
   // 12/10/21 -- Elyce and Rochi merge -- added a const cast to this constructor to make it compatible with usage in the rest of the library. Essentially allows you to
   // pass the neighbors as a const, but removes that const qualifier when providing it to the level set
   // This was done to make two_phase_flows compatible with the updates done to the level set class
  /*!
   * \brief my_p4est_level_set_t constructor based on a pointer to a CONSTANT node-neighborhood object. Too bad for you if the node neighbors
   * are not initialized upon calling this constructor because you'll pay the price...
   * \param ngbd_ pointer to a (constant) my_p4est_node_neighbors object, this constructor will __NOT__ initialize the node neighborhood information
   */

  // TO-DO: have to somehow resolve two phase flows use of const neighbors with the rest of us using non const neighbors 11/23/21 Elyce and Rochi
  my_p4est_level_set_t(const my_p4est_node_neighbors_t* ngbd_)
    : myb(ngbd_->myb), p4est(ngbd_->p4est), nodes(ngbd_->nodes), ghost(ngbd_->ghost), /*ngbd(ngbd_),*/
      tree_dimensions{DIM((ngbd_->get_brick()->xyz_max[0] - ngbd_->get_brick()->xyz_min[0])/ngbd_->get_brick()->nxyztrees[0], (ngbd_->get_brick()->xyz_max[1] - ngbd_->get_brick()->xyz_min[1])/ngbd_->get_brick()->nxyztrees[1], (ngbd_->get_brick()->xyz_max[2] - ngbd_->get_brick()->xyz_min[2])/ngbd_->get_brick()->nxyztrees[2])},
      zero_distance_threshold(EPS*MIN(DIM((ngbd_->get_brick()->xyz_max[0] - ngbd_->get_brick()->xyz_min[0])/ngbd_->get_brick()->nxyztrees[0], (ngbd_->get_brick()->xyz_max[1] - ngbd_->get_brick()->xyz_min[1])/ngbd_->get_brick()->nxyztrees[1], (ngbd_->get_brick()->xyz_max[2] - ngbd_->get_brick()->xyz_min[2])/ngbd_->get_brick()->nxyztrees[2]))),
      interpolation_on_interface(quadratic_non_oscillatory_continuous_v2),
      use_neumann_for_contact_angle(true), contact_angle_extension(0),
      /*show_convergence(false), show_convergence_band(5.), */bc_rel_thresh(1.e-8),use_two_step_extrapolation(false)
  {
      this->ngbd = const_cast<my_p4est_node_neighbors_t*>(ngbd_);
  }

  inline void update(my_p4est_node_neighbors_t *ngbd_) {
    ngbd  = ngbd_;
    myb   = ngbd->myb;
    p4est = ngbd->p4est;
    nodes = ngbd->nodes;
    ghost = ngbd->ghost;
  }

  /*!
   * \brief reinitialize_1st_order reinitializes the given node-sampled levelset function with linear interface
   * localization in space and 1st order Forward Euler steps in pseudo-time (with adaptive time-stepping)
   * \param [inout] phi                     : on input, node-sampled value of the levelset function whose reinitialization is required
   *                                          on output, results of the reinitialization algorithm
   * \param [in] number_of_iterations       : number of Forward Euler steps in pseudo-time to be executed (default is 50)
   * \param [in] limit_absolute_value_phi_0 : the reinitization procedure is executed only at nodes where the local value of phi is such that
   *                                          fabs(phi_node_on_input) < limit_absolute_value_phi_0 (default is DBL_MAX)
   * (more details to be found in header's comments of reinitialize_within_range_of_phi_0)
   */
  inline void reinitialize_1st_order(Vec phi, const int &number_of_iterations = 50, const double &limit_absolute_value_phi_0 = DBL_MAX) const
  {
    reinitialize_within_range_of_phi_0(phi, 1, 1, limit_absolute_value_phi_0, -limit_absolute_value_phi_0, number_of_iterations);
    return;
  }
  /*!
   * \brief reinitialize_1st_order_above_threshold reinitializes the given node-sampled levelset function with linear interface
   * localization in space and 1st order Forward Euler steps in pseudo-time (with adaptive time-stepping)
   * \param [inout] phi                     : on input, node-sampled value of the levelset function whose reinitialization is required
   *                                          on output, results of the reinitialization algorithm
   * \param [in] threshold                  : the reinitization procedure is executed only at nodes where the local value of phi is such that
   *                                          threshold < phi_node_on_input
   * \param [in] number_of_iterations       : number of pseudotime steps to be executed (default is 50)
   * \param [in] upper_limit_phi_0 :          the reinitization procedure is executed only at nodes where the local value of phi is such that
   *                                          phi_node_on_input < upper_limit_phi_0 (default is DBL_MAX)
   * (more details to be found in header's comments of reinitialize_within_range_of_phi_0)
   */
  inline void reinitialize_1st_order_above_threshold(Vec phi, const double &threshold, const int &number_of_iterations = 50, const double &upper_limit_phi_0 = DBL_MAX) const
  {
    reinitialize_within_range_of_phi_0(phi, 1, 1, upper_limit_phi_0, threshold, number_of_iterations);
    return;
  }

  /*!
   * \brief reinitialize_2nd_order reinitializes the given node-sampled levelset function with quadratic interface
   * localization in space and minmod corrections to second-derivatives and with 2nd order TVD Runge-Kutta explicit
   * steps in pseudo-time (with adaptive time-stepping)
   * \param [inout] phi                     : on input, node-sampled value of the levelset function whose reinitialization is required
   *                                          on output, results of the reinitialization algorithm
   * \param [in] number_of_iterations       : number of pseudotime steps to be executed (default is 50)
   * \param [in] limit_absolute_value_phi_0 : the reinitization procedure is executed only at nodes where the local value of phi is such that
   *                                          fabs(phi_node_on_input) < limit_absolute_value_phi_0 (default is DBL_MAX)
   * (more details to be found in header's comments of reinitialize_within_range_of_phi_0)
   */
  inline void reinitialize_2nd_order(Vec phi, const int &number_of_iterations = 50, const double &limit_absolute_value_phi_0 = DBL_MAX) const
  {
    reinitialize_within_range_of_phi_0(phi, 2, 2, limit_absolute_value_phi_0, -limit_absolute_value_phi_0, number_of_iterations);
    return;
  }

  /**
   * @brief Reinitialize the given node-sampled level-set function with quadratic interface localization in space and
   * minmod corrections to second-derivatives and with 2nd order TVD Runge-Kutta explicit steps in pseudo-time (with
   * adaptive time-stepping), only for nodes that are not masked out.
   * @param [in,out] phi On input, node-sampled value of the level-set function whose reinitialization is required;
   * 		on output, results of the reinitialization algorithm.
   * @param [in] mask Parallel vector with 0s for nodes we don't want to update, !=0 for rest of updatable grid points.
   * @param [in] number_of_masked A hint to reserve space in vectors with indices for nonupdatable nodes.
   * @param [in] number_of_iterations Number of pseudotime steps to be executed (default is 50).
   * @param [in] limit_absolute_value_phi_0 The reinitization procedure is executed only at nodes where the local value
   * 		of phi is such that fabs(phi_node_on_input) < limit_absolute_value_phi_0 (default is DBL_MAX).
   * @note More details to be found in header's comments of reinitialize_within_range_of_phi_0.
   */
  inline void reinitialize_2nd_order_with_mask( Vec phi, Vec mask, const int& number_of_masked,
												const int& number_of_iterations=50,
												const double& limit_absolute_value_phi_0=DBL_MAX ) const
  {
	std::vector<p4est_locidx_t> customLayerNodes, customLocalNodes;		// Will store indices of updatable nodes after
	customLayerNodes.reserve( ngbd->layer_nodes.size() );				// removing masked nodes.
	customLocalNodes.reserve( ngbd->local_nodes.size() );

	std::vector<p4est_locidx_t> maskedLayerNodes, maskedLocalNodes;		// Store indices of nonupdatable nodes.
	maskedLayerNodes.reserve( number_of_masked );
	maskedLocalNodes.reserve( number_of_masked );

	PetscErrorCode ierr;
	const double *maskReadPtr;
	ierr = VecGetArrayRead( mask, &maskReadPtr );
	CHKERRXX( ierr );

	for( const auto& n : ngbd->layer_nodes )	// Building custom list of updatable and nonupdatable layer nodes.
	{
	  if( maskReadPtr[n] != 0 )
	  	customLayerNodes.push_back( n );
	  else
	  	maskedLayerNodes.push_back( n );
	}

	for( const auto& n : ngbd->local_nodes )	// Building custom list of updatable and nonupdatable local nodes.
	{
	  if( maskReadPtr[n] != 0 )
		customLocalNodes.push_back( n );
	  else
	  	maskedLocalNodes.push_back( n );
	}

	ierr = VecRestoreArrayRead( mask, &maskReadPtr );
	CHKERRXX( ierr );

	reinitialize_within_range_of_phi_0( phi, 2, 2, limit_absolute_value_phi_0, -limit_absolute_value_phi_0,
									 	number_of_iterations, &customLayerNodes, &customLocalNodes,
									 	&maskedLayerNodes, &maskedLocalNodes );
  }

  /*!
   * \brief reinitialize_2nd_order_above_threshold reinitializes the given node-sampled levelset function with quadratic interface
   * localization in space and minmod corrections to second-derivatives and with 2nd order TVD Runge-Kutta explicit
   * steps in pseudo-time (with adaptive time-stepping)
   * \param [inout] phi                     : on input, node-sampled value of the levelset function whose reinitialization is required
   *                                          on output, results of the reinitialization algorithm
   * \param [in] threshold                  : the reinitization procedure is executed only at nodes where the local value of phi is such that
   *                                          threshold < phi_node_on_input
   * \param [in] number_of_iterations       : number of pseudotime steps to be executed (default is 50)
   * \param [in] upper_limit_phi_0 :          the reinitization procedure is executed only at nodes where the local value of phi is such that
   *                                          phi_node_on_input < upper_limit_phi_0 (default is DBL_MAX)
   * (more details to be found in header's comments of reinitialize_within_range_of_phi_0)
   */
  inline void reinitialize_2nd_order_above_threshold(Vec phi, const double &threshold, const int &number_of_iterations = 50, const double &upper_limit_phi_0 = DBL_MAX) const
  {
    reinitialize_within_range_of_phi_0(phi, 2, 2, upper_limit_phi_0, threshold, number_of_iterations);
    return;
  }

  /*!
   * \brief reinitialize_2nd_order_time_1st_order_space reinitializes the given node-sampled levelset function with linear interface
   * localization in space and with 2nd order TVD Runge-Kutta explicit steps in pseudo-time (with adaptive time-stepping)
   * \param [inout] phi                     : on input, node-sampled value of the levelset function whose reinitialization is required
   *                                          on output, results of the reinitialization algorithm
   * \param [in] number_of_iterations       : number of pseudotime steps to be executed (default is 50)
   * \param [in] limit_absolute_value_phi_0 : the reinitization procedure is executed only at nodes where the local value of phi is such that
   *                                          phi_node_on_input < limit_absolute_value_phi_0 (default is DBL_MAX)
   * (more details to be found in header's comments of reinitialize_within_range_of_phi_0)
   */
  inline void reinitialize_2nd_order_time_1st_order_space(Vec phi, const int &number_of_iterations = 50, const double &limit_absolute_value_phi_0 = DBL_MAX) const
  {
    reinitialize_within_range_of_phi_0(phi, 1, 2, limit_absolute_value_phi_0, -limit_absolute_value_phi_0, number_of_iterations);
    return;
  }
  /*!
   * \brief reinitialize_2nd_order_time_1st_order_space_above_threshold reinitializes the given node-sampled levelset function with linear interface
   * localization in space and with 2nd order TVD Runge-Kutta explicit steps in pseudo-time (with adaptive time-stepping)
   * \param [inout] phi                     : on input, node-sampled value of the levelset function whose reinitialization is required
   *                                          on output, results of the reinitialization algorithm
   * \param [in] threshold                  : the reinitization procedure is executed only at nodes where the local value of phi is such that
   *                                          threshold < phi_node_on_input
   * \param [in] number_of_iterations       : number of pseudotime steps to be executed (default is 50)
   * \param [in] upper_limit_phi_0 :          the reinitization procedure is executed only at nodes where the local value of phi is such that
   *                                          phi_node_on_input < upper_limit_phi_0 (default is DBL_MAX)
   * (more details to be found in header's comments of reinitialize_within_range_of_phi_0)
   */
  inline void reinitialize_2nd_order_time_1st_order_space_above_threshold(Vec phi, const double &threshold, const int &number_of_iterations = 50, const double &upper_limit_phi_0 = DBL_MAX) const
  {
    reinitialize_within_range_of_phi_0(phi, 1, 2, upper_limit_phi_0, threshold, number_of_iterations);
    return;
  }

  /*!
   * \brief reinitialize_1st_order_time_2nd_order_space reinitializes the given node-sampled levelset function with quadratic interface
   * localization in space and minmod corrections to second-derivatives and 1st order Forward Euler steps in pseudo-time (with adaptive
   * time-stepping)
   * \param [inout] phi                     : on input, node-sampled value of the levelset function whose reinitialization is required
   *                                          on output, results of the reinitialization algorithm
   * \param [in] number_of_iterations       : number of pseudotime steps to be executed (default is 50)
   * \param [in] limit_absolute_value_phi_0 : the reinitization procedure is executed only at nodes where the local value of phi is such that
   *                                          fabs(phi_node_on_input) < limit_absolute_value_phi_0 (default is DBL_MAX)
   * (more details to be found in header's comments of reinitialize_within_range_of_phi_0)
   */
  inline void reinitialize_1st_order_time_2nd_order_space(Vec phi, const int &number_of_iterations = 50, const double &limit_absolute_value_phi_0 = DBL_MAX) const
  {
    reinitialize_within_range_of_phi_0(phi, 2, 1, limit_absolute_value_phi_0, -limit_absolute_value_phi_0, number_of_iterations);
    return;
  }
  /*!
   * \brief reinitialize_1st_order_time_2nd_order_space_above_threshold reinitializes the given node-sampled levelset function with quadratic
   * interface localization in space and minmod corrections to second-derivatives and 1st order Forward Euler steps in pseudo-time (with
   * adaptive time-stepping)
   * \param [inout] phi                     : on input, node-sampled value of the levelset function whose reinitialization is required
   *                                          on output, results of the reinitialization algorithm
   * \param [in] threshold                  : the reinitization procedure is executed only at nodes where the local value of phi is such that
   *                                          threshold < phi_node_on_input
   * \param [in] number_of_iterations       : number of pseudotime steps to be executed (default is 50)
   * \param [in] upper_limit_phi_0 :          the reinitization procedure is executed only at nodes where the local value of phi is such that
   *                                          phi_node_on_input < upper_limit_phi_0 (default is DBL_MAX)
   * (more details to be found in header's comments of reinitialize_within_range_of_phi_0)
   */
  inline void reinitialize_1st_order_time_2nd_order_space_above_threshold(Vec phi, const double &threshold, const int &number_of_iterations = 50, const double &upper_limit_phi_0 = DBL_MAX) const
  {
    reinitialize_within_range_of_phi_0(phi, 2, 1, upper_limit_phi_0, threshold, number_of_iterations);
    return;
  }

  /*!
   * \brief perturb_level_set_function ensures no node value of phi is smaller than epsilon. If a node value on input
   * is smaller than epsilon in absolute value, it is set to sng(phi[n])*epsilon
   * \param [inout] phi   node-sampled levelset values;
   * \param [in] epsilon  user-defined threshold, as described above.
   */
  void perturb_level_set_function(Vec phi, const double &epsilon) const;

  /*!
   * \brief advect_in_normal_direction advects the level-set function in the normal direction using Godunov's scheme
   * \param vn     [in]      velocity in the normal direction (CF_DIM function evaluated at nodes)
   * \param phi    [in, out] node-sample level-set function at time t_n on input, at time t_{n + 1} = t_n + dt on output
   * \param phi_xx [in]      dxx derivative of level-set function. will be computed internally if set to NULL
   * \param phi_yy [in]      dyy derivative of level-set function. will be computed internally if set to NULL
   * \param phi_zz [in]      dzz derivative of level-set function. will be computed internally if set to NULL
   * \return dt calculated as to satisfy "CFL number <= 0.8" for the calculated advection step
   */
  double advect_in_normal_direction(const CF_DIM& vn, Vec phi, DIM(Vec phi_xx = NULL, Vec phi_yy = NULL, Vec phi_zz = NULL)) const;

  /*!
   * \brief advect_in_normal_direction advects the level-set function in the normal direction using Godunov's scheme
   * \param vn     [in]      velocity in the normal direction (node-sampled vector)
   * \param phi    [in, out] node-sample level-set function at time t_n on input, at time t_{n + 1} = t_n + dt on output
   * \param dt     [in]      time-step used for advancing the advection equation
   * \param phi_xx [in]      dxx derivative of level-set function. will be computed internally if set to NULL
   * \param phi_yy [in]      dyy derivative of level-set function. will be computed internally if set to NULL
   * \param phi_zz [in]      dzz derivative of level-set function. will be computed internally if set to NULL
   * \return dt (unchanged)
   */
  double advect_in_normal_direction(const Vec vn, Vec phi, const double &dt, DIM(Vec phi_xx = NULL, Vec phi_yy = NULL, Vec phi_zz = NULL)) const;

  /*!
   * \brief advect_in_normal_direction advects the level-set function in the normal direction using Godunov's scheme
   * [Daniil : was it an attempt to have second order accurate advection? Probably, watch out for providing a correct
   * estimate if vn_np1 when using this though]
   * \param vn     [in]      velocity in the normal direction (Vec)
   * \param vn_np1 [in]      velocity in the normal direction (Vec) at time n + 1
   * \param phi    [in, out] level-set function
   * \param dt     [in]      time-step used for advancing the advection equation
   * \param phi_xx [in]      dxx derivative of level-set function. will be computed internally if set to NULL
   * \param phi_yy [in]      dyy derivative of level-set function. will be computed internally if set to NULL
   * \param phi_zz [in]      dzz derivative of level-set function. will be computed internally if set to NULL
   * \return dt (unchanged)
   */
  double advect_in_normal_direction(const Vec vn, const Vec vn_np1, Vec phi, const double &dt, DIM(Vec phi_xx = NULL, Vec phi_yy = NULL, Vec phi_zz = NULL)) const;

  /*!
   * \brief extend_Over_Interface extrapolates a node-sampled field q from the negative to positive domain
   * associated with the node-sampled levelset function phi.
   * \param [in] phi            : vector of node-sampled levelset values;
   * \param [inout] q           : vector of node-sampled field values to extrapolate, values are unchanged in the negative domain;
   * \param [in] bc             : pointer to a boundary condition object, used for determining and using the type of interface
   *                              boundary (disregarded if NULL);
   * \param [in] degree         : desired degree of the interpolant to be build (0, 1 or 2);
   * \param [in] band_to_extend : values of q are calculated and set only where 0 < phi < band_to_extend*(smallest diagonal);
   * (see more details in the header's comments of 'geometric_extrapolation_over_interface', which is called internally).
   */
  inline void extend_Over_Interface(Vec phi, Vec q, const BoundaryConditionsDIM &bc, const unsigned char &degree = 2, const unsigned int &band_to_extend = UINT_MAX) const
  {
    geometric_extrapolation_over_interface(phi, q, &bc, degree, band_to_extend);
    return;
  }

  /*!
   * \brief extend_Over_Interface extrapolates a node-sampled field q from the negative to positive domain
   * associated with the node-sampled levelset function phi.
   * \param [in] phi            : vector of node-sampled levelset values;
   * \param [inout] q           : vector of node-sampled field values to extrapolate, values are unchanged in the negative domain;
   * \param [in] bc_type        : type of boundary condition on q onto the interface, (hence assumed constant on the interface);
   * \param [in] bc_values      : vector of node-sampled values of the interface boundary condition, i.e. q values onto the interface
   *                              in case bc_type == DIRICHLET, or the normal derivatives of q if bc_type == NEUMANN);
   *                              --> those values are interpolated at interface points of interest using the 'quadratic_non_oscillatory'
   *                              interpolation procedure;
   * \param [in] degree         : desired degree of the interpolant to be build (0, 1 or 2);
   * \param [in] band_to_extend : values of q are calculated and set only where 0 < phi < band_to_extend*(smallest diagonal);
   * (see more details in the header's comments of 'geometric_extrapolation_over_interface', which is called internally).
   * */
  inline void extend_Over_Interface(Vec phi, Vec q, const BoundaryConditionType &bc_type, Vec bc_values, const unsigned char &degree = 2, const unsigned int &band_to_extend = UINT_MAX) const
  {
    // create a temporary BoundaryConditionsDIM object and use a local interpolator for the functor
    BoundaryConditionsDIM bc_tmp;
    my_p4est_interpolation_nodes_t interface_bc_value(ngbd); interface_bc_value.set_input(bc_values, quadratic_non_oscillatory); // create an operator
    bc_tmp.setInterfaceType(bc_type);
    bc_tmp.setInterfaceValue(interface_bc_value);
    geometric_extrapolation_over_interface(phi, q, &bc_tmp, degree, band_to_extend);
    return;
  }
  /*!
   * \brief extend_Over_Interface extrapolates a node-sampled field q from the negative to positive domain
   * associated with the node-sampled levelset function phi.
   * \param [in] phi            : vector of node-sampled levelset values;
   * \param [inout] q           : vector of node-sampled field values to extrapolate, values are unchanged in the negative domain;
   * \param [in] degree         : desired degree of the interpolant to be build (0, 1 or 2);
   * \param [in] band_to_extend : values of q are calculated and set only where 0 < phi < band_to_extend*(smallest diagonal);
   * WARNING : this function does not required (and therefore does not consider) any interface boundary condition on q
   * (see more details in the header's comments of 'geometric_extrapolation_over_interface', which is called internally).
   */
  inline void extend_Over_Interface(Vec phi, Vec q, const unsigned char &degree = 2, const unsigned int &band_to_extend = UINT_MAX) const
  {
    geometric_extrapolation_over_interface(phi, q, NULL, degree, band_to_extend);
    return;
  }

  /*!
   * \brief extend_from_interface_to_whole_domain evaluates the interface-projected point for every node in the grid
   * then interpolate the input field 'q' (using 'quadratic' interpolation) at that interface-projected point to set
   * the local value of 'q_extended'
   * --> "flattens" the input field q away from the interface in either direction
   * \param [in] phi            : vector of node-sampled levelset values;
   * \param [in] q              : vector of node-sampled interface field values to extend, values are unchanged;
   * \param [out] q_extended    : vector of node-sampled values of q interpolated at the corresponding interface-projected
   *                              points (quadratic interpolation);
   * \param [in] band_to_extend : values of q_extended are calculated and set only where fabs(phi) < band_to_extend*(smallest diagonal).
   */
  void extend_from_interface_to_whole_domain(Vec phi, Vec q, Vec q_extended, const unsigned int &band_to_extend = UINT_MAX) const;

  /* extend a quantity over the interface with the TVD algorithm */
  void extend_Over_Interface_TVD(Vec phi, Vec q, int iterations=20, int order=2,
                                 double band_use=-DBL_MAX, double band_extend=DBL_MAX,
                                 Vec normal[P4EST_DIM]=NULL, Vec mask=NULL, boundary_conditions_t *bc=NULL,
                                 bool use_nonzero_guess=false, Vec q_n=NULL, Vec q_nn=NULL) const;

  /*!
   * \brief extend_Over_Interface_TVD_Full extends a quantity over the interface with the TVD algorithm
   * (all derivatives are extended, not just q_n and q_nn)
   * \param phi               [in]      level-set function representing interface
   * \param q                 [in, out] field to extends
   * \param iterations        [in]      number of iterations
   * \param order             [out]     order of extrapolation
   * \param band_use          [in]      how deep
   * \param band_extend       [in]
   * \param normal            [in]      provides normal vector field (
   * \param mask              [in]
   * \param bc                [in]
   * \param use_nonzero_guess [in]
   * \param q_d               [in]
   * \param q_dd              [in]
   * \return
   */
  void extend_Over_Interface_TVD_Full(Vec phi, Vec q, int iterations=20, int order=2,
                                      double band_use=-DBL_MAX, double band_extend=DBL_MAX,
                                      Vec normal[P4EST_DIM]=NULL, Vec mask=NULL, boundary_conditions_t *bc=NULL,
                                      bool use_nonzero_guess=false, Vec *q_d=NULL, Vec *q_dd=NULL) const;

  void extend_Over_Interface_TVD_not_parallel(Vec phi, Vec q, int iterations=20, int order=2) const;

  void extend_from_interface_to_whole_domain_TVD_one_iteration(const std::vector<int>& map, const double *phi_p, const double* grad_phi_p,
                                                               double *q_out_p,
                                                               const double *q_p, const double *qxxyyzz_p[P4EST_DIM],
                                                               std::vector<double>& qi_m00, std::vector<double>& qi_p00,
                                                               std::vector<double>& qi_0m0, std::vector<double>& qi_0p0,
                                                             #ifdef P4_TO_P8
                                                               std::vector<double>& qi_00m, std::vector<double>& qi_00p,
                                                             #endif
                                                               std::vector<double>& s_m00, std::vector<double>& s_p00,
                                                               std::vector<double>& s_0m0, std::vector<double>& s_0p0,
                                                             #ifdef P4_TO_P8
                                                               std::vector<double>& s_00m, std::vector<double>& s_00p,
                                                             #endif
                                                               const u_int& blocksize) const;
  void extend_from_interface_to_whole_domain_TVD(Vec phi, Vec q_interface, Vec q, int iterations=20, Vec mask=NULL, double band_zero=2, double band_smooth=10, double (*cf)(p4est_locidx_t, int, double)=NULL,
                                                 Vec grad_phi_in = NULL, const u_int& blocksize = 1) const;

  
//  /*!
//   * \brief extend_from_interface_to_whole_domain_TVD_one_iteration performs one iteration of extension in normal direction
//   * \param map                   [in]
//   * \param nx                    [in]
//   * \param ny                    [in]
//   * \param q_out_p               [out]
//   * \param q_p                   [in]
//   * \param qxx_p                 [in]
//   * \param qyy_p                 [in]
//   * \param map_grid_to_interface [in]  a look-up table that contains information about wheter a given node
//   *                                    has interface points next to it
//   * \param interface_directions  [in]
//   * \param interface_distances   [in]
//   * \param interface_values      [in]
//   * \return
//   */
//  void extend_from_interface_to_whole_domain_TVD_one_iteration_daniil_ver(const std::vector<int>& map, DIM(vector<double>& nx, vector<double>& ny, vector<double>& nz),
//                                                               double *q_out_p, double *q_p, DIM(double *qxx_p, double *qyy_p, double *qzz_p),
//                                                               std::vector<int>& map_grid_to_interface,
//                                                               std::vector<int>& interface_directions,
//                                                               std::vector<double>& interface_distances,
//                                                               std::vector<double>& interface_values) const;

//  /*!
//   * \brief extend_from_interface_to_whole_domain_TVD extends a field from interface in normal direction (``flattening'')
//   * \param phi         [in]  level-set function
//   * \param qi          [in]  original field that needs to be ``flattened''
//   *                          (can be NULL, but must provide func in this case)
//   *                          (even when func is provided, qi can be used to set initial guess)
//   * \param q           [out] ``flattened'' around interface field
//   * \param mask        [in]  if not NULL, interface values where (mask > band_zero*diag_min) will be set to zero
//   *                          (usefull when dealing with intersecting level-set function)
//   * \param band_zero   [in]  the threshold defining which interface values will be set to zero
//   * \param band_smooth [in]  transition width to zeros (in units of diag_min)
//   * \param func        [in]  optional function to compute interface values
//   * \return
//   */
//  void extend_from_interface_to_whole_domain_TVD_daniil_ver(Vec phi, Vec qi, Vec q, int iterations=20,
//                                                 Vec mask=NULL, double band_zero=2, double band_smooth=10,
//                                                 double (*func)(p4est_locidx_t, int, double)=NULL) const;

  void enforce_contact_angle(Vec phi_wall, Vec phi_intf, Vec cos_angle, int iterations=20, Vec normal[] = NULL) const;
  void enforce_contact_angle2(Vec phi, Vec q, Vec cos_angle, int iterations=20, int order=2, Vec normal[] = NULL) const;
  void extend_Over_Interface_TVD_regional( Vec phi, Vec mask, Vec region, Vec q, int iterations = 20, int order = 2) const;

  void advect_in_normal_direction_with_contact_angle(const Vec vn, const Vec surf_tns, const Vec cos_angle, const Vec phi_wall, Vec phi, double dt);

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
  inline void set_bc_rel_thresh                (bool   value) { bc_rel_thresh                 = value; }
  inline void set_use_two_step_extrapolation   (bool   value) { use_two_step_extrapolation    = value; }
};

#endif // MY_P4EST_LEVELSET_H
