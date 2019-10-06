#ifdef P4_TO_P8
#include "my_p4est_interpolation_nodes.h"
#else
#include "my_p4est_interpolation_nodes.h"
#endif

#include "math.h"


my_p4est_interpolation_nodes_t::my_p4est_interpolation_nodes_t(const my_p4est_node_neighbors_t* ngbd_n)
  : my_p4est_interpolation_t(ngbd_n), nodes(ngbd_n->nodes),
    method(linear)
{
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    Fxxyyzz[dir].resize(0);
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

void my_p4est_interpolation_nodes_t::set_input(Vec *F, interpolation_method method, const unsigned int &n_vecs_, const unsigned int &block_size_f)
{
  set_input(F, n_vecs_, block_size_f);
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    Fxxyyzz[dir].resize(0);
  Fxxyyzz_block.resize(0);
  this->method = method;
}


#ifdef P4_TO_P8
void my_p4est_interpolation_nodes_t::set_input(Vec *F, Vec *Fxxyyzz_block_, Vec *Fxx_, Vec *Fyy_, Vec *Fzz_,  interpolation_method method, const unsigned int &n_vecs_, const unsigned int &block_size_f)
#else
void my_p4est_interpolation_nodes_t::set_input(Vec *F, Vec *Fxxyyzz_block_, Vec *Fxx_, Vec *Fyy_,             interpolation_method method, const unsigned int &n_vecs_, const unsigned int &block_size_f)
#endif
{
  // give the second derivatives either by P4EST-DIM block-structured vectors or component by component, but not both ways!
  // either Fxxyyzz_block_ == NULL or ((Fxx_ == NULL) && (Fyy_ == NULL) && (Fzz_ == NULL))
#ifdef P4_TO_P8
  P4EST_ASSERT((Fxxyyzz_block_ == NULL) || ((Fxx_ == NULL) && (Fyy_ == NULL) && (Fzz_ == NULL)));
#else
  P4EST_ASSERT((Fxxyyzz_block_ == NULL) || ((Fxx_ == NULL) && (Fyy_ == NULL)));
#endif
  set_input(F, n_vecs_, block_size_f);
  if(Fxxyyzz_block_==NULL)
  {
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      Fxxyyzz[dir].resize(n_vecs_);
    for (unsigned int k = 0; k < Fi.size(); ++k) {
      Fxxyyzz[0][k] = Fxx_[k];
      Fxxyyzz[1][k] = Fyy_[k];
#ifdef P4_TO_P8
      Fxxyyzz[2][k] = Fzz_[k];
#endif
    }
    Fxxyyzz_block.resize(0);
  }
  else
  {
    Fxxyyzz_block.resize(n_vecs_);
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      Fxxyyzz[dir].resize(0);
    for (unsigned int k = 0; k < Fi.size(); ++k)
      Fxxyyzz_block[k] = Fxxyyzz_block_[k];
  }
  this->method = method;
}


#ifdef P4_TO_P8
void my_p4est_interpolation_nodes_t::operator ()(double x, double y, double z, double* results) const
#else
void my_p4est_interpolation_nodes_t::operator ()(double x, double y, double* results) const
#endif
{
  PetscErrorCode ierr;

#ifdef P4_TO_P8
  double xyz [] = { x, y, z };
#else
  double xyz [] = { x, y };
#endif
  
  /* first clip the coordinates */
#ifdef P4_TO_P8
  double xyz_clip [] = { x, y, z };
#else
  double xyz_clip [] = { x, y };
#endif

  // clip to bounding box
  for (short i=0; i<P4EST_DIM; i++){
    if (xyz_clip[i] > xyz_max[i]) xyz_clip[i] = is_periodic(p4est,i) ? xyz_clip[i]-(xyz_max[i]-xyz_min[i]) : xyz_max[i];
    if (xyz_clip[i] < xyz_min[i]) xyz_clip[i] = is_periodic(p4est,i) ? xyz_clip[i]+(xyz_max[i]-xyz_min[i]) : xyz_min[i];
  }
  

  unsigned int n_functions = n_vecs();
  P4EST_ASSERT(n_functions > 0);
  P4EST_ASSERT(bs_f > 0);
  const double *Fi_p[n_functions];
  for (unsigned int k = 0; k < n_functions; ++k) {
    ierr = VecGetArrayRead(Fi[k], &Fi_p[k]); CHKERRXX(ierr);
  }

  bool use_precomputed_block_derivatives = (method == quadratic || method == quadratic_non_oscillatory) && (Fxxyyzz_block.size() == n_functions);
  bool use_precomputed_derivatives_by_components = (method == quadratic || method == quadratic_non_oscillatory) && !use_precomputed_block_derivatives;
  for (unsigned char dir = 0; use_precomputed_derivatives_by_components && (dir < P4EST_DIM); ++dir)
    use_precomputed_derivatives_by_components = use_precomputed_derivatives_by_components && (Fxxyyzz[dir].size() == n_functions);
  const double **Fxxyyzz_p[P4EST_DIM];
  const double **Fxxyyzz_block_p = NULL;
  if(use_precomputed_derivatives_by_components)
  {
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    {
      Fxxyyzz_p[dir] = P4EST_ALLOC(const double *, n_functions);
      for (unsigned int k = 0; k < n_functions; ++k) {
        ierr = VecGetArrayRead(Fxxyyzz[dir][k], &Fxxyyzz_p[dir][k]); CHKERRXX(ierr);
      }
    }
  }
  if(use_precomputed_block_derivatives)
  {
    Fxxyyzz_block_p = P4EST_ALLOC(const double *, n_functions);
    for (unsigned int k = 0; k < n_functions; ++k) {
      ierr = VecGetArrayRead(Fxxyyzz_block[k], &Fxxyyzz_block_p[k]); CHKERRXX(ierr);
    }
  }

  double f  [P4EST_CHILDREN*n_functions*bs_f]; // f[j*n_functions*bs_f+k*bs_f+comp] = value of the compth component of the kth block vector at node j
  double *fdd;
  if (method == quadratic || method == quadratic_non_oscillatory)
    fdd = P4EST_ALLOC(double, P4EST_CHILDREN*n_functions*bs_f*P4EST_DIM); // fdd[j*n_functions*bs_f*P4EST_DIM+k*bs_f*P4EST_DIM+comp*P4EST_DIM+dir] = value of second derivative along direction dir of the compth component of the kth block vector at node j

  p4est_quadrant_t best_match;
  vector<p4est_quadrant_t> remote_matches;
  int rank_found = ngbd_n->hierarchy->find_smallest_quadrant_containing_point(xyz_clip, best_match, remote_matches);
  
  if (rank_found == p4est->mpirank || (rank_found!=-1 && (method==linear || use_precomputed_derivatives_by_components || use_precomputed_block_derivatives) )) { // local quadrant
    p4est_locidx_t quad_idx;
    if(rank_found==p4est->mpirank)
    {
      p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, best_match.p.piggy3.which_tree);
      quad_idx = best_match.p.piggy3.local_num + tree->quadrants_offset;
    }
    else
    {
      quad_idx = best_match.p.piggy3.local_num + p4est->local_num_quadrants;
    }

    for (short i = 0; i<P4EST_CHILDREN; i++) {
      p4est_locidx_t node_idx = nodes->local_nodes[quad_idx*P4EST_CHILDREN + i];
      for (unsigned int k = 0; k < n_functions; ++k)
        for (unsigned int comp = 0; comp < bs_f; ++comp)
          f[i*n_functions*bs_f+k*bs_f+comp] = Fi_p[k][bs_f*node_idx+comp];
    }

    // compute derivatives
    if (method == quadratic || method == quadratic_non_oscillatory) {
      if (use_precomputed_derivatives_by_components || use_precomputed_block_derivatives) {
        for (short j = 0; j<P4EST_CHILDREN; j++) {
          p4est_locidx_t node_idx = nodes->local_nodes[quad_idx*P4EST_CHILDREN + j];
          for (unsigned int k = 0; k < n_functions; ++k)
            for (unsigned int comp = 0; comp < bs_f; ++comp)
              for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
                fdd[j*n_functions*bs_f*P4EST_DIM+k*bs_f*P4EST_DIM+comp*P4EST_DIM+dir] = use_precomputed_derivatives_by_components ? Fxxyyzz_p[dir][k][bs_f*node_idx+comp] : Fxxyyzz_block_p[k][node_idx*bs_f*P4EST_DIM+comp*P4EST_DIM+dir];
        }
      } else {
        quad_neighbor_nodes_of_node_t qnnn;
        for (short j = 0; j<P4EST_CHILDREN; j++) {
          p4est_locidx_t node_idx = nodes->local_nodes[quad_idx*P4EST_CHILDREN + j];
          ngbd_n->get_neighbors(node_idx, qnnn);
          (bs_f==1)? qnnn.laplace(Fi_p, (fdd+j*n_functions*P4EST_DIM), n_functions) : qnnn.laplace_all_components(Fi_p, (fdd+j*n_functions*bs_f*P4EST_DIM), n_functions, bs_f);
        }
      }
    }

    // restore arrays and release memory
    for (unsigned int k = 0; k < n_functions; ++k) {
      ierr = VecRestoreArrayRead(Fi[k], &Fi_p[k]); CHKERRXX(ierr);
      if(use_precomputed_derivatives_by_components){
        for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
          ierr = VecRestoreArrayRead(Fxxyyzz[dir][k], &Fxxyyzz_p[dir][k]); CHKERRXX(ierr);
        }
      }
      if(use_precomputed_block_derivatives){
        ierr = VecRestoreArrayRead(Fxxyyzz_block[k], &Fxxyyzz_block_p[k]); CHKERRXX(ierr);
      }
    }
    if(use_precomputed_derivatives_by_components)
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
        P4EST_FREE(Fxxyyzz_p[dir]);
    if(use_precomputed_block_derivatives)
      P4EST_FREE(Fxxyyzz_block_p);


    double xyz_q[P4EST_DIM];
    quad_xyz_fr_q(quad_idx, best_match.p.piggy3.which_tree, p4est, ghost, xyz_q);

    double xyz_p[P4EST_DIM];
    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      xyz_p[dir] = xyz[dir];
      if     (is_periodic(p4est,dir) && xyz[dir]-xyz_q[dir]>(xyz_max[dir]-xyz_min[dir])/2) xyz_p[dir] -= xyz_max[dir]-xyz_min[dir];
      else if(is_periodic(p4est,dir) && xyz_q[dir]-xyz[dir]>(xyz_max[dir]-xyz_min[dir])/2) xyz_p[dir] += xyz_max[dir]-xyz_min[dir];
    }

    if (method == linear) {
      linear_interpolation(p4est, best_match.p.piggy3.which_tree, best_match, f, xyz_p, results, n_functions*bs_f);
    } else if (method == quadratic) {
      quadratic_interpolation(p4est, best_match.p.piggy3.which_tree, best_match, f, fdd, xyz_p, results, n_functions*bs_f);
    } else if (method == quadratic_non_oscillatory) {
      quadratic_non_oscillatory_interpolation(p4est, best_match.p.piggy3.which_tree, best_match, f, fdd, xyz_p, results, n_functions*bs_f);
    }

    if (method == quadratic || method == quadratic_non_oscillatory)
      P4EST_FREE(fdd);
  } else {
    for (unsigned int k = 0; k < n_functions; ++k) {
      ierr = VecRestoreArrayRead(Fi[k], &Fi_p[k]); CHKERRXX(ierr);
      if(use_precomputed_derivatives_by_components){
        for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
          ierr = VecRestoreArrayRead(Fxxyyzz[dir][k], &Fxxyyzz_p[dir][k]); CHKERRXX(ierr);
        }
      }
      if(use_precomputed_block_derivatives){
        ierr = VecRestoreArrayRead(Fxxyyzz_block[k], &Fxxyyzz_block_p[k]); CHKERRXX(ierr);
      }
    }
    if(use_precomputed_derivatives_by_components)
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
        P4EST_FREE(Fxxyyzz_p[dir]);
    if(use_precomputed_block_derivatives)
      P4EST_FREE(Fxxyyzz_block_p);

    std::ostringstream oss;
    oss << "\n[ERROR]: Point (" << x << "," << y <<
       #ifdef P4_TO_P8
           "," << z <<
       #endif
           ") is not locally owned by processor "
        << p4est->mpirank << ". ";
    if (rank_found != -1) {
      oss << "Only processor " << rank_found
          << " can perform the interpolation locally. ";
      if (method != linear) {
        oss << "If this point is on the processor boundary, or within the ghost layer, "
               "try using 'linear' interpolation or alternatively precompute the "
               "second derivatives if using 'quadratic'' interpolations." << std::endl;
      }
    } else {
      oss << "Could not find appropriate remote owner. "
             "Possible matches are = ";
      for (size_t i = 0; i<remote_matches.size() - 1; i++) {
        oss << remote_matches[i].p.piggy1.owner_rank << ", ";
      }
      oss << remote_matches[remote_matches.size() - 1].p.piggy1.owner_rank << "." << std::endl;
    }
    throw std::invalid_argument(oss.str());
  }
}


void my_p4est_interpolation_nodes_t::interpolate(const p4est_quadrant_t &quad, const double *xyz, double* results, const unsigned int &comp) const
{
  PetscErrorCode ierr;

  p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, quad.p.piggy3.which_tree);
  p4est_locidx_t quad_idx = quad.p.piggy3.local_num + tree->quadrants_offset;

  unsigned int n_functions = n_vecs();
  P4EST_ASSERT(n_functions > 0);
  P4EST_ASSERT(bs_f > 0);
  P4EST_ASSERT(comp==ALL_COMPONENTS || comp < bs_f);

  const double *Fi_p[n_functions];
  for (unsigned int k = 0; k < n_functions; ++k) {
    ierr = VecGetArrayRead(Fi[k], &Fi_p[k]); CHKERRXX(ierr); }
  const unsigned int nelem_per_node = (comp==ALL_COMPONENTS && bs_f > 1) ? bs_f*n_functions: n_functions;
  double f[P4EST_CHILDREN*nelem_per_node]; // f[j*n_functions*bs_f+k*bs_f+comp] = value of the compth component of the kth block vector at node j

  for (short i = 0; i<P4EST_CHILDREN; i++) {
    p4est_locidx_t node_idx = nodes->local_nodes[quad_idx*P4EST_CHILDREN + i];
    if(comp==ALL_COMPONENTS && bs_f > 1)
      for (unsigned int k = 0; k < n_functions; ++k)
        for (unsigned int comp = 0; comp < bs_f; ++comp)
          f[i*n_functions*bs_f+k*bs_f+comp] = Fi_p[k][bs_f*node_idx+comp];
    else
      for (unsigned int k = 0; k < n_functions; ++k)
        f[i*n_functions+k] = Fi_p[k][bs_f*node_idx+comp];
  }

  /* enforce periodicity if necessary */
  double xyz_q[P4EST_DIM];
  quad_xyz_fr_q(quad_idx, quad.p.piggy3.which_tree, p4est, ghost, xyz_q);

  double xyz_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    xyz_p[dir] = xyz[dir];
    if     (is_periodic(p4est,dir) && xyz[dir]-xyz_q[dir]>(xyz_max[dir]-xyz_min[dir])/2) xyz_p[dir] -= xyz_max[dir]-xyz_min[dir];
    else if(is_periodic(p4est,dir) && xyz_q[dir]-xyz[dir]>(xyz_max[dir]-xyz_min[dir])/2) xyz_p[dir] += xyz_max[dir]-xyz_min[dir];
  }

  /* compute derivatives */
  if (method == quadratic || method == quadratic_non_oscillatory)
  {
    double fdd[P4EST_CHILDREN*nelem_per_node*P4EST_DIM];
    // description of the above data structure:
    // if (comp==ALL_COMPONENTS && bs_f > 1)
    //   fdd[j*n_functions*bs_f*P4EST_DIM+k*bs_f*P4EST_DIM+comp*P4EST_DIM+dir] = value of second derivative along direction dir of the compth component of the kth block vector at node j
    // else
    //   fdd[j*n_functions*P4EST_DIM+k*P4EST_DIM+dir] = value of second derivative along direction dir of the component of interest of the kth block vector at node j

    bool use_precomputed_block_derivatives = (Fxxyyzz_block.size() == n_functions);
    bool use_precomputed_derivatives_by_components = !use_precomputed_block_derivatives;
    for (unsigned char dir = 0; use_precomputed_derivatives_by_components && (dir < P4EST_DIM); ++dir)
      use_precomputed_derivatives_by_components = use_precomputed_derivatives_by_components && (Fxxyyzz[dir].size() == n_functions);
    const double **Fxxyyzz_p[P4EST_DIM];
    const double **Fxxyyzz_block_p=NULL;
    if(use_precomputed_derivatives_by_components){
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir){
        Fxxyyzz_p[dir] = P4EST_ALLOC(const double *, n_functions);
        for (unsigned int k = 0; k < n_functions; ++k) {
          ierr = VecGetArrayRead(Fxxyyzz[dir][k], &Fxxyyzz_p[dir][k]); CHKERRXX(ierr);
        }
      }
    }
    if(use_precomputed_block_derivatives){
      Fxxyyzz_block_p = P4EST_ALLOC(const double *, n_functions);
      for (unsigned int k = 0; k < n_functions; ++k) {
        ierr = VecGetArrayRead(Fxxyyzz_block[k], &Fxxyyzz_block_p[k]); CHKERRXX(ierr);
      }
    }

    if (use_precomputed_derivatives_by_components || use_precomputed_block_derivatives) {
      for (short j = 0; j<P4EST_CHILDREN; j++) {
        p4est_locidx_t node_idx = nodes->local_nodes[quad_idx*P4EST_CHILDREN + j];
        if(comp==ALL_COMPONENTS && bs_f > 1)
          for (unsigned int k = 0; k < n_functions; ++k)
            for (unsigned int comp = 0; comp < bs_f; ++comp)
              for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
                fdd[j*n_functions*bs_f*P4EST_DIM+k*bs_f*P4EST_DIM+comp*P4EST_DIM+dir] = use_precomputed_derivatives_by_components ? Fxxyyzz_p[dir][k][bs_f*node_idx+comp] : Fxxyyzz_block_p[k][node_idx*bs_f*P4EST_DIM+comp*P4EST_DIM+dir];
        else
          for (unsigned int k = 0; k < n_functions; ++k)
            for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
              fdd[j*n_functions*P4EST_DIM+k*P4EST_DIM+dir] = use_precomputed_derivatives_by_components ? Fxxyyzz_p[dir][k][bs_f*node_idx+comp] : Fxxyyzz_block_p[k][node_idx*bs_f*P4EST_DIM+comp*P4EST_DIM+dir];
      }
      // restore arrays and release memory
      for (unsigned int k = 0; k < n_functions; ++k) {
        ierr = VecRestoreArrayRead(Fi[k], &Fi_p[k]); CHKERRXX(ierr);
        if(use_precomputed_derivatives_by_components){
          for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
            ierr = VecRestoreArrayRead(Fxxyyzz[dir][k], &Fxxyyzz_p[dir][k]); CHKERRXX(ierr);
          }
        }
        if(use_precomputed_block_derivatives){
          ierr = VecRestoreArrayRead(Fxxyyzz_block[k], &Fxxyyzz_block_p[k]); CHKERRXX(ierr);
        }
      }
      if(use_precomputed_derivatives_by_components)
        for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
          P4EST_FREE(Fxxyyzz_p[dir]);
      if(use_precomputed_block_derivatives)
        P4EST_FREE(Fxxyyzz_block_p);
    }
    else{
      quad_neighbor_nodes_of_node_t qnnn;
      for (short j = 0; j<P4EST_CHILDREN; j++){
        p4est_locidx_t node_idx = nodes->local_nodes[quad_idx*P4EST_CHILDREN + j];
        ngbd_n->get_neighbors(node_idx, qnnn);
        if(bs_f == 1)
          qnnn.laplace(Fi_p, (fdd+j*n_functions*P4EST_DIM), n_functions) ;
        else if (bs_f > 1 && comp!=ALL_COMPONENTS)
          qnnn.laplace_component(Fi_p, (fdd+j*n_functions*P4EST_DIM), n_functions, bs_f, comp);
        else
          qnnn.laplace_all_components(Fi_p, (fdd+j*n_functions*bs_f*P4EST_DIM), n_functions, bs_f);
      }
    }
    for (unsigned int k = 0; k < n_functions; ++k) {
      ierr = VecRestoreArrayRead(Fi[k], &Fi_p[k]); CHKERRXX(ierr);
    }

    if(method==quadratic) {
      quadratic_interpolation(p4est, quad.p.piggy3.which_tree, quad, f, fdd, xyz_p, results, nelem_per_node);
      return;
    } else {
      quadratic_non_oscillatory_interpolation(p4est, quad.p.piggy3.which_tree, quad, f, fdd, xyz_p, results, nelem_per_node);
      return;
    }
  }

  for (unsigned int k = 0; k < n_functions; ++k) {
    ierr = VecRestoreArrayRead(Fi[k], &Fi_p[k]); CHKERRXX(ierr);
  }
  linear_interpolation(p4est, quad.p.piggy3.which_tree, quad, f, xyz_p, results, nelem_per_node);
  return;
}
