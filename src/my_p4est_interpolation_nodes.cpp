#ifdef P4_TO_P8
#include "my_p8est_interpolation_nodes.h"
#else
#include "my_p4est_interpolation_nodes.h"
#endif

#include "math.h"

my_p4est_interpolation_nodes_t::my_p4est_interpolation_nodes_t(const my_p4est_node_neighbors_t* ngbd_n)
  : my_p4est_interpolation_t(ngbd_n), nodes(ngbd_n->nodes),
    method(linear)
{
  for (u_char dim = 0; dim < P4EST_DIM; ++dim)
    Fxxyyzz[dim].resize(0);
  Fxxyyzz_block.resize(0);
}

void my_p4est_interpolation_nodes_t::update_neighbors(const my_p4est_node_neighbors_t* ngbd_n_)
{
  ngbd_n  = ngbd_n_;
  p4est   = ngbd_n->p4est;
  ghost   = ngbd_n->ghost;
  myb     = ngbd_n->myb;
  nodes   = ngbd_n->nodes;
}

void my_p4est_interpolation_nodes_t::set_input(Vec *F, const interpolation_method & method, const size_t &n_vecs_, const u_int &block_size_f)
{
  set_input_fields(F, n_vecs_, block_size_f);
  for (u_char dim = 0; dim < P4EST_DIM; ++dim)
    Fxxyyzz[dim].resize(0);
  Fxxyyzz_block.resize(0);
  this->method = method;
}

void my_p4est_interpolation_nodes_t::set_input(Vec *F, Vec *Fxxyyzz_block_, DIM(Vec *Fxx_, Vec *Fyy_, Vec *Fzz_), const interpolation_method & method, const size_t &n_vecs_, const u_int &block_size_f)
{
  // give the second derivatives either by P4EST-DIM block-structured vectors or component by component, but not both ways!
  // either Fxxyyzz_block_ == NULL or ((Fxx_ == NULL) && (Fyy_ == NULL) && (Fzz_ == NULL))
  P4EST_ASSERT(Fxxyyzz_block_ == NULL || ANDD(Fxx_ == NULL, Fyy_ == NULL, Fzz_ == NULL));
  set_input_fields(F, n_vecs_, block_size_f);
  if(Fxxyyzz_block_ == NULL)
  {
    for (u_char dim = 0; dim < P4EST_DIM; ++dim)
      Fxxyyzz[dim].resize(n_vecs_);
    for (size_t k = 0; k < Fi.size(); ++k) {
      P4EST_ASSERT(method == linear || Fxx_[k] != NULL); Fxxyyzz[0][k] = Fxx_[k];
      P4EST_ASSERT(method == linear || Fyy_[k] != NULL); Fxxyyzz[1][k] = Fyy_[k];
#ifdef P4_TO_P8
      P4EST_ASSERT(method == linear || Fzz_[k] != NULL); Fxxyyzz[2][k] = Fzz_[k];
#endif
    }
    Fxxyyzz_block.resize(0);
  }
  else
  {
    Fxxyyzz_block.resize(n_vecs_);
    for (u_char dim = 0; dim < P4EST_DIM; ++dim)
      Fxxyyzz[dim].resize(0);
    for (size_t k = 0; k < Fi.size(); ++k){
      P4EST_ASSERT(method == linear || Fxxyyzz_block_[k] != NULL); Fxxyyzz_block[k] = Fxxyyzz_block_[k];
    }
  }
  this->method = method;
}

void my_p4est_interpolation_nodes_t::operator()(const double *xyz, double* results, const u_int& comp) const
{
  int rank_found = -1;
  std::vector<p4est_quadrant_t> remote_matches;
  try
  {
    bool proceed_even_if_ghost = method == linear;
    // if not, then we need second derivatives and we can proceed safely in ghost layers
    // if and only if the second derivatives have been precomputed and provided to the object
    // --> we either use second derivatives given by block-structured vectors
    proceed_even_if_ghost = proceed_even_if_ghost || Fxxyyzz_block.size() == n_vecs();
    // --> or given components-by-components
    proceed_even_if_ghost = proceed_even_if_ghost || ANDD(Fxxyyzz[0].size() == n_vecs(), Fxxyyzz[1].size() == n_vecs(), Fxxyyzz[2].size() == n_vecs());
    clip_point_and_interpolate_all_on_the_fly(xyz, comp, results, rank_found, remote_matches, proceed_even_if_ghost);
  }
  catch (std::invalid_argument& e)
  {
    // The point could not be handled locally --> let the user know but let's be more specific about the origin of the issue
    std::ostringstream oss;
    oss << "my_p4est_interpolation_nodes_t::operator (): Point (" << xyz[0] << "," << xyz[1] << ONLY3D("," << xyz[2] <<) ") cannot be processed on-the-fly by process " << p4est->mpirank << ". ";
    if (rank_found != -1)
    {
      oss << "Only process " << rank_found << " can perform the interpolation locally. ";
      if (method != linear)
        oss << "This point is within the ghost layer though: either try using 'linear' interpolation or precompute "
               "the second derivatives and provide them to the interpolator along with the input field(s)." << std::endl;
    }
    else
    {
      oss << "Process(es) likely to own a quadrant containing the point is (are) =  ";
      for (size_t i = 0; i < remote_matches.size() - 1; i++) {
        oss << remote_matches[i].p.piggy1.owner_rank << ", ";
      }
      oss << remote_matches[remote_matches.size() - 1].p.piggy1.owner_rank << "." << std::endl;
    }
    throw std::invalid_argument(oss.str());
  }
  return;
}

void my_p4est_interpolation_nodes_t::interpolate(const p4est_quadrant_t &quad, const double *xyz, double* results, const u_int &comp) const
{
  PetscErrorCode ierr;
  const p4est_topidx_t &tree_idx = quad.p.piggy3.which_tree;
  const p4est_locidx_t &quad_idx = quad.p.piggy3.local_num;

  const size_t n_functions = n_vecs();
  P4EST_ASSERT(n_functions > 0);
  P4EST_ASSERT(bs_f > 0);
  P4EST_ASSERT(comp == ALL_COMPONENTS || comp < bs_f);

  std::vector<const double *> Fi_p(n_functions);
  for (size_t k = 0; k < n_functions; ++k) {
    ierr = VecGetArrayRead(Fi[k], &Fi_p[k]); CHKERRXX(ierr); }
  const size_t nelem_per_node = (comp == ALL_COMPONENTS && bs_f > 1 ? bs_f*n_functions: n_functions);
  std::vector<double> f(nelem_per_node*P4EST_CHILDREN); // f[k*bs_f*P4EST_CHILDREN+cc*P4EST_CHILDREN+j] = value of the ccth component of the kth block vector at node j

  for (u_char i = 0; i < P4EST_CHILDREN; i++) {
    p4est_locidx_t node_idx = nodes->local_nodes[quad_idx*P4EST_CHILDREN + i];
    if (bs_f == 1)
      for (size_t k = 0; k < n_functions; ++k)
        f[k*P4EST_CHILDREN + i] = Fi_p[k][node_idx];
    else if (comp == ALL_COMPONENTS)
      for (size_t k = 0; k < n_functions; ++k)
        for (u_int cc = 0; cc < bs_f; ++cc)
          f[k*bs_f*P4EST_CHILDREN + cc*P4EST_CHILDREN + i] = Fi_p[k][bs_f*node_idx + cc];
    else
      for (size_t k = 0; k < n_functions; ++k)
        f[k*P4EST_CHILDREN + i] = Fi_p[k][bs_f*node_idx + comp];
  }


  const double* xyz_min = get_xyz_min();
  const double* xyz_max = get_xyz_max();
  const bool* periodic  = get_periodicity();
  /* Fetch the interpolation point to feed the basic interpolation routines */
  double xyz_p[P4EST_DIM] = {DIM(xyz[0], xyz[1], xyz[2])};
  /* NOTE: the following is NOT a standard "clip_in_domain": we bring the point back in, !only if periodicity allows it!
   * If the domain is not periodic but one queries values out of the domain anyways, we use the standard interpolant built
   * on the closest smallest quadrant in the domain and use it as a way to "somehow extrapolate" the results out of the domain...
   * Moreover, in case of periodicity, we actually fetch the mirror point that is the closest to the provided owning quadrant.
   * Let the coordinate of the point be x, the corresponding coordinate of the quadrant front lower left corner be xq, we want to
   * find the integer k such that fabs(x + k*L_x - xq) is minimal where L_x is the domain dimension along the Cartesian direction
   * of interest. That integer is either pp = floor((xq - x)/L_x) or (pp + 1). The choice between pp or (pp + 1) is based on the
   * fact that the minimal value of fabs(x + k*L_x - xq)/L_x must be smaller than 0.5 */
  double xyz_q[P4EST_DIM]; // needed only in case of periodicity
  if(ORD(periodic[0], periodic[1], periodic[2]))
    quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, xyz_q);
  for(u_char dim = 0; dim < P4EST_DIM; ++dim)
    if(periodic[dim])
    {
      const double pp = (xyz_q[dim] - xyz[dim])/(xyz_max[dim] - xyz_min[dim]);
      xyz_p[dim] += (floor(pp) + (pp > floor(pp) + 0.5 ? 1.0 : 0.0))*(xyz_max[dim] - xyz_min[dim]);
    }

  /* compute derivatives */
  if (method != linear)
  {
    std::vector<double> fdd(nelem_per_node*P4EST_CHILDREN*P4EST_DIM);
    // description of the above data structure:
    // if (comp == ALL_COMPONENTS && bs_f > 1)
    //   fdd[k*bs_f*P4EST_CHILDREN*P4EST_DIM + cc*P4EST_CHILDREN*P4EST_DIM + j*P4EST_DIM + dim] = value of second derivative along cartesian direction dim of the ccth component of the kth block vector at node j
    // else
    //   fdd[k*P4EST_CHILDREN*P4EST_DIM + j*P4EST_DIM + dim] = value of second derivative along cartesian direction dim of the (component of interest of the) kth (block) vector at node j

    const bool use_precomputed_block_derivatives = (Fxxyyzz_block.size() == n_functions);
    bool use_precomputed_derivatives_by_components = !use_precomputed_block_derivatives;
    for (u_char dim = 0; use_precomputed_derivatives_by_components && dim < P4EST_DIM; ++dim)
      use_precomputed_derivatives_by_components = use_precomputed_derivatives_by_components && Fxxyyzz[dim].size() == n_functions;
    const double **Fxxyyzz_p[P4EST_DIM];
    const double **Fxxyyzz_block_p = NULL;
    if(use_precomputed_derivatives_by_components){
      for (u_char dim = 0; dim < P4EST_DIM; ++dim){
        Fxxyyzz_p[dim] = P4EST_ALLOC(const double *, n_functions);
        for (size_t k = 0; k < n_functions; ++k) {
          ierr = VecGetArrayRead(Fxxyyzz[dim][k], &Fxxyyzz_p[dim][k]); CHKERRXX(ierr);
        }
      }
    }
    if(use_precomputed_block_derivatives){
      Fxxyyzz_block_p = P4EST_ALLOC(const double *, n_functions);
      for (size_t k = 0; k < n_functions; ++k) {
        ierr = VecGetArrayRead(Fxxyyzz_block[k], &Fxxyyzz_block_p[k]); CHKERRXX(ierr);
      }
    }

    if (use_precomputed_derivatives_by_components || use_precomputed_block_derivatives) {
      for (u_char j = 0; j < P4EST_CHILDREN; j++) {
        p4est_locidx_t node_idx = nodes->local_nodes[quad_idx*P4EST_CHILDREN + j];
        if(bs_f == 1)
          for (size_t k = 0; k < n_functions; ++k)
            for (u_char dim = 0; dim < P4EST_DIM; ++dim)
              fdd[k*P4EST_CHILDREN*P4EST_DIM + j*P4EST_DIM + dim] = (use_precomputed_derivatives_by_components ? Fxxyyzz_p[dim][k][node_idx] : Fxxyyzz_block_p[k][node_idx*P4EST_DIM + dim]);
        else if (comp == ALL_COMPONENTS)
          for (size_t k = 0; k < n_functions; ++k)
            for (u_int cc = 0; cc < bs_f; ++cc)
              for (u_char dim = 0; dim < P4EST_DIM; ++dim)
                fdd[k*bs_f*P4EST_CHILDREN*P4EST_DIM + cc*P4EST_CHILDREN*P4EST_DIM + j*P4EST_DIM + dim] = (use_precomputed_derivatives_by_components ? Fxxyyzz_p[dim][k][bs_f*node_idx + cc] : Fxxyyzz_block_p[k][node_idx*bs_f*P4EST_DIM + cc*P4EST_DIM + dim]);
        else
          for (size_t k = 0; k < n_functions; ++k)
            for (u_char dim = 0; dim < P4EST_DIM; ++dim)
              fdd[k*P4EST_CHILDREN*P4EST_DIM + j*P4EST_DIM + dim] = (use_precomputed_derivatives_by_components ? Fxxyyzz_p[dim][k][bs_f*node_idx + comp] : Fxxyyzz_block_p[k][node_idx*bs_f*P4EST_DIM + comp*P4EST_DIM + dim]);
      }
      // restore arrays and release memory
      for (size_t k = 0; k < n_functions; ++k) {
        ierr = VecRestoreArrayRead(Fi[k], &Fi_p[k]); CHKERRXX(ierr);
        if(use_precomputed_derivatives_by_components){
          for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
            ierr = VecRestoreArrayRead(Fxxyyzz[dim][k], &Fxxyyzz_p[dim][k]); CHKERRXX(ierr);
          }
        }
        if(use_precomputed_block_derivatives){
          ierr = VecRestoreArrayRead(Fxxyyzz_block[k], &Fxxyyzz_block_p[k]); CHKERRXX(ierr);
        }
      }
      if(use_precomputed_derivatives_by_components)
        for (u_char dim = 0; dim < P4EST_DIM; ++dim)
          P4EST_FREE(Fxxyyzz_p[dim]);
      if(use_precomputed_block_derivatives)
        P4EST_FREE(Fxxyyzz_block_p);
    }
    else {
      const bool local_quad = quad_idx < p4est->local_num_quadrants;
      if(!local_quad)
        throw std::invalid_argument("my_p4est_interpolation_nodes_t::interpolate(): attempting to calculate second derivative of input fields in the ghost layer...");

      std::vector<double> tmp(nelem_per_node*P4EST_DIM);
      const bool neihbors_are_initialized = ngbd_n->neighbors_are_initialized();
      quad_neighbor_nodes_of_node_t qnnn;
      const quad_neighbor_nodes_of_node_t *qnnn_p = (neihbors_are_initialized ? NULL : &qnnn); // we'll avoid data copy if the neighbors are initialized!
      for (u_char j = 0; j < P4EST_CHILDREN; j++){
        const p4est_locidx_t node_idx = nodes->local_nodes[quad_idx*P4EST_CHILDREN + j];
        if(neihbors_are_initialized)
          ngbd_n->get_neighbors(node_idx, qnnn_p);
        else
          ngbd_n->get_neighbors(node_idx, qnnn);
        if(bs_f == 1 || (comp < bs_f && bs_f > 1))
        {
          (bs_f == 1 ? qnnn_p->laplace(Fi_p.data(), tmp.data(), n_functions) : qnnn_p->laplace_component(Fi_p.data(), tmp.data(), n_functions, bs_f, comp));
          for (size_t k = 0; k < n_functions; ++k)
            for (u_char dim = 0; dim < P4EST_DIM; ++dim)
              fdd[k*P4EST_CHILDREN*P4EST_DIM + j*P4EST_DIM + dim] = tmp[k*P4EST_DIM + dim];
        }
        else
        {
          qnnn_p->laplace_all_components(Fi_p.data(), tmp.data(), n_functions, bs_f);
          for (size_t k = 0; k < n_functions; ++k)
            for (u_int cc = 0; cc < bs_f; ++cc)
              for (u_char dim = 0; dim < P4EST_DIM; ++dim)
                fdd[k*bs_f*P4EST_CHILDREN*P4EST_DIM + cc*P4EST_CHILDREN*P4EST_DIM + j*P4EST_DIM + dim] = tmp[k*bs_f*P4EST_DIM + cc*P4EST_DIM + dim];
        }
      }
    }
    for (size_t k = 0; k < n_functions; ++k) {
      ierr = VecRestoreArrayRead(Fi[k], &Fi_p[k]); CHKERRXX(ierr);
    }

    if(method == quadratic) {
      quadratic_interpolation(p4est, tree_idx, quad, f.data(), fdd.data(), xyz_p, results, nelem_per_node);
      return;
    }
    else if (method == quadratic_non_oscillatory) {
      quadratic_non_oscillatory_interpolation(p4est, tree_idx, quad, f.data(), fdd.data(), xyz_p, results, nelem_per_node);
      return;
    }
    else if (method == quadratic_non_oscillatory_continuous_v1){
      quadratic_non_oscillatory_continuous_v1_interpolation(p4est, tree_idx, quad, f.data(), fdd.data(), xyz_p, results, nelem_per_node);
      return;
    }
    else if (method == quadratic_non_oscillatory_continuous_v2){
      quadratic_non_oscillatory_continuous_v2_interpolation(p4est, tree_idx, quad, f.data(), fdd.data(), xyz_p, results, nelem_per_node);
      return;
    }
  }

  for (size_t k = 0; k < n_functions; ++k) {
    ierr = VecRestoreArrayRead(Fi[k], &Fi_p[k]); CHKERRXX(ierr);
  }

  linear_interpolation(p4est, tree_idx, quad, f.data(), xyz_p, results, nelem_per_node);
  return;
}
