#ifdef P4_TO_P8
#include "my_p4est_interpolation_nodes.h"
#else
#include "my_p4est_interpolation_nodes.h"
#endif

#include "math.h"


my_p4est_interpolation_nodes_t::my_p4est_interpolation_nodes_t(const my_p4est_node_neighbors_t* ngbd_n)
  : my_p4est_interpolation_t(ngbd_n), nodes(ngbd_n->nodes),
    Fxx(vector<Vec>(0)), Fyy(vector<Vec>(0)),
    #ifdef P4_TO_P8
    Fzz(vector<Vec>(0)),
    #endif
    method(linear)
{

}

void my_p4est_interpolation_nodes_t::update_neighbors(const my_p4est_node_neighbors_t* ngbd_n_)
{
  ngbd_n  = ngbd_n_;
  p4est   = ngbd_n->p4est;
  ghost   = ngbd_n->ghost;
  myb     = ngbd_n->myb;
  nodes   = ngbd_n->nodes;
}

void my_p4est_interpolation_nodes_t::set_input(Vec *F, interpolation_method method, unsigned int n_vecs_) {
  set_input(F, n_vecs_);
  Fxx.resize(0);
  Fyy.resize(0);
#ifdef P4_TO_P8
  Fzz.resize(0);
#endif
  this->method = method;
}


#ifdef P4_TO_P8
void my_p4est_interpolation_nodes_t::set_input(Vec* F, Vec* Fxx_, Vec* Fyy_, Vec* Fzz_, interpolation_method method, unsigned int n_vecs_) {
#else
void my_p4est_interpolation_nodes_t::set_input(Vec* F, Vec* Fxx_, Vec* Fyy_, interpolation_method method, unsigned int n_vecs_) {
#endif
  set_input(F, n_vecs_);
  Fxx.resize(n_vecs_);
  Fyy.resize(n_vecs_);
#ifdef P4_TO_P8
  Fzz.resize(n_vecs_);
#endif
  for (unsigned int k = 0; k < Fi.size(); ++k) {
    Fxx[k] = Fxx_[k];
    Fyy[k] = Fyy_[k];
#ifdef P4_TO_P8
    Fzz[k] = Fzz_[k];
#endif
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
  const double *Fi_p[n_functions];
  for (unsigned int k = 0; k < n_functions; ++k) {
    ierr = VecGetArrayRead(Fi[k], &Fi_p[k]); CHKERRXX(ierr);
  }

  const double *Fxx_p[n_functions], *Fyy_p[n_functions];
#ifdef P4_TO_P8
  bool use_precomputed_derivatives = (Fxx.size() == n_functions) && (Fyy.size() == n_functions) && (Fzz.size() == n_functions);
  const double *Fzz_p[n_functions];
#else
  bool use_precomputed_derivatives = (Fxx.size() == n_functions) && (Fyy.size() == n_functions);
#endif

  if (use_precomputed_derivatives) {
    for (unsigned int k = 0; k < n_functions; ++k) {
      ierr = VecGetArrayRead(Fxx[k], &Fxx_p[k]); CHKERRXX(ierr);
      ierr = VecGetArrayRead(Fyy[k], &Fyy_p[k]); CHKERRXX(ierr);
  #ifdef P4_TO_P8
      ierr = VecGetArrayRead(Fzz[k], &Fzz_p[k]); CHKERRXX(ierr);
  #endif
    }
  }

  double f  [n_functions*P4EST_CHILDREN];
  double fdd[n_functions*P4EST_CHILDREN*P4EST_DIM];

  p4est_quadrant_t best_match;
  vector<p4est_quadrant_t> remote_matches;
  int rank_found = ngbd_n->hierarchy->find_smallest_quadrant_containing_point(xyz_clip, best_match, remote_matches);
  
  if (rank_found == p4est->mpirank || (rank_found!=-1 && (method==linear || use_precomputed_derivatives) )) { // local quadrant
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
        f[P4EST_CHILDREN*k+i] = Fi_p[k][node_idx];
    }

      // compute derivatives
    if (method == quadratic || method == quadratic_non_oscillatory || method == quadratic_non_oscillatory_continuous_v1 || method == quadratic_non_oscillatory_continuous_v2) {
      if (use_precomputed_derivatives) {
        for (short j = 0; j<P4EST_CHILDREN; j++) {
          p4est_locidx_t node_idx = nodes->local_nodes[quad_idx*P4EST_CHILDREN + j];
          for (unsigned int k = 0; k < n_functions; ++k) {
            fdd[k*P4EST_CHILDREN*P4EST_DIM+j*P4EST_DIM + 0] = Fxx_p[k][node_idx];
            fdd[k*P4EST_CHILDREN*P4EST_DIM+j*P4EST_DIM + 1] = Fyy_p[k][node_idx];
  #ifdef P4_TO_P8
            fdd[k*P4EST_CHILDREN*P4EST_DIM+j*P4EST_DIM + 2] = Fzz_p[k][node_idx];
  #endif
          }
        }
      } else {
        quad_neighbor_nodes_of_node_t qnnn;
        for (short j = 0; j<P4EST_CHILDREN; j++) {
          p4est_locidx_t node_idx = nodes->local_nodes[quad_idx*P4EST_CHILDREN + j];
          ngbd_n->get_neighbors(node_idx, qnnn);
          for (unsigned int k = 0; k < n_functions; ++k) {
            fdd[k*P4EST_CHILDREN*P4EST_DIM+j*P4EST_DIM + 0] = qnnn.dxx_central(Fi_p[k]);
            fdd[k*P4EST_CHILDREN*P4EST_DIM+j*P4EST_DIM + 1] = qnnn.dyy_central(Fi_p[k]);
  #ifdef P4_TO_P8
            fdd[k*P4EST_CHILDREN*P4EST_DIM+j*P4EST_DIM + 2] = qnnn.dzz_central(Fi_p[k]);
  #endif
          }
        }
      }
    }
     
    // restore arrays and release remote_maches
    for (unsigned int k = 0; k < n_functions; ++k) {
      ierr = VecRestoreArrayRead(Fi[k], &Fi_p[k]); CHKERRXX(ierr);
    }
  
    if (use_precomputed_derivatives) {
      for (unsigned int k = 0; k < n_functions; ++k) {
        ierr = VecRestoreArrayRead(Fxx[k], &Fxx_p[k]); CHKERRXX(ierr);
        ierr = VecRestoreArrayRead(Fyy[k], &Fyy_p[k]); CHKERRXX(ierr);
#ifdef P4_TO_P8
        ierr = VecRestoreArrayRead(Fzz[k], &Fzz_p[k]); CHKERRXX(ierr);
#endif
      }
    }

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
      linear_interpolation(p4est, best_match.p.piggy3.which_tree, best_match, f, xyz_p, results, n_functions);
    } else if (method == quadratic) {
      quadratic_interpolation(p4est, best_match.p.piggy3.which_tree, best_match, f, fdd, xyz_p, results, n_functions);
    } else if (method == quadratic_non_oscillatory) {
      quadratic_non_oscillatory_interpolation(p4est, best_match.p.piggy3.which_tree, best_match, f, fdd, xyz_p, results, n_functions);
    } else if (method == quadratic_non_oscillatory_continuous_v1) {
        quadratic_non_oscillatory_continuous_v1_interpolation(p4est, best_match.p.piggy3.which_tree, best_match, f, fdd, xyz_p, results, n_functions);
    } else if (method == quadratic_non_oscillatory_continuous_v2) {
        quadratic_non_oscillatory_continuous_v2_interpolation(p4est, best_match.p.piggy3.which_tree, best_match, f, fdd, xyz_p, results, n_functions);
    }
    
  } else {
    for (unsigned int k = 0; k < n_functions; ++k) {
      ierr = VecRestoreArrayRead(Fi[k], &Fi_p[k]); CHKERRXX(ierr);
      if (use_precomputed_derivatives) {
        ierr = VecRestoreArrayRead(Fxx[k], &Fxx_p[k]); CHKERRXX(ierr);
        ierr = VecRestoreArrayRead(Fyy[k], &Fyy_p[k]); CHKERRXX(ierr);
  #ifdef P4_TO_P8
        ierr = VecRestoreArrayRead(Fzz[k], &Fzz_p[k]); CHKERRXX(ierr);
  #endif
      }
    }

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


void my_p4est_interpolation_nodes_t::interpolate(const p4est_quadrant_t &quad, const double *xyz, double* results) const
{
  PetscErrorCode ierr;

  p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, quad.p.piggy3.which_tree);
  p4est_locidx_t quad_idx = quad.p.piggy3.local_num + tree->quadrants_offset;

  unsigned int n_functions = n_vecs();
  P4EST_ASSERT(n_functions > 0);
  double f[n_functions*P4EST_CHILDREN];

  const double *Fi_p[n_functions];
  for (unsigned int k = 0; k < n_functions; ++k) {
    ierr = VecGetArrayRead(Fi[k], &Fi_p[k]); CHKERRXX(ierr);
    for (short j = 0; j<P4EST_CHILDREN; j++) {
      p4est_locidx_t node_idx = nodes->local_nodes[quad_idx*P4EST_CHILDREN + j];
      f[P4EST_CHILDREN*k+j] = Fi_p[k][node_idx];
    }
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
  if (method == quadratic || method == quadratic_non_oscillatory || method == quadratic_non_oscillatory_continuous_v1 || method == quadratic_non_oscillatory_continuous_v2)
  {
    double fdd[n_functions*P4EST_CHILDREN*P4EST_DIM];

#ifdef P4_TO_P8
    if ((Fxx.size() == n_functions) && (Fyy.size() == n_functions) && (Fzz.size() == n_functions))
#else
    if ((Fxx.size() == n_functions) && (Fyy.size() == n_functions))
#endif
    {
      for (unsigned int k = 0; k < n_functions; ++k) {
        const double *Fxx_p;
        ierr = VecGetArrayRead(Fxx[k], &Fxx_p); CHKERRXX(ierr);
        const double *Fyy_p;
        ierr = VecGetArrayRead(Fyy[k], &Fyy_p); CHKERRXX(ierr);
  #ifdef P4_TO_P8
        const double *Fzz_p;
        ierr = VecGetArrayRead(Fzz[k], &Fzz_p); CHKERRXX(ierr);
  #endif
        for (short j = 0; j<P4EST_CHILDREN; j++) {
          p4est_locidx_t node_idx = nodes->local_nodes[quad_idx*P4EST_CHILDREN + j];

          fdd[k*P4EST_CHILDREN*P4EST_DIM+j*P4EST_DIM + 0] = Fxx_p[node_idx];
          fdd[k*P4EST_CHILDREN*P4EST_DIM+j*P4EST_DIM + 1] = Fyy_p[node_idx];
  #ifdef P4_TO_P8
          fdd[k*P4EST_CHILDREN*P4EST_DIM+j*P4EST_DIM + 2] = Fzz_p[node_idx];
  #endif
        }

        ierr = VecRestoreArrayRead(Fxx[k], &Fxx_p); CHKERRXX(ierr);
        ierr = VecRestoreArrayRead(Fyy[k], &Fyy_p); CHKERRXX(ierr);
  #ifdef P4_TO_P8
        ierr = VecRestoreArrayRead(Fzz[k], &Fzz_p); CHKERRXX(ierr);
  #endif
      }
    }
    else
    {
      quad_neighbor_nodes_of_node_t qnnn;
      for (short j = 0; j<P4EST_CHILDREN; j++)
      {
        p4est_locidx_t node_idx = nodes->local_nodes[quad_idx*P4EST_CHILDREN + j];
        ngbd_n->get_neighbors(node_idx, qnnn);
        for (unsigned int k = 0; k < n_functions; ++k) {
          fdd[k*P4EST_CHILDREN*P4EST_DIM+j*P4EST_DIM + 0] = qnnn.dxx_central(Fi_p[k]);
          fdd[k*P4EST_CHILDREN*P4EST_DIM+j*P4EST_DIM + 1] = qnnn.dyy_central(Fi_p[k]);
#ifdef P4_TO_P8
          fdd[k*P4EST_CHILDREN*P4EST_DIM+j*P4EST_DIM + 2] = qnnn.dzz_central(Fi_p[k]);
#endif
        }
      }
    }
    for (unsigned int k = 0; k < n_functions; ++k) {
      ierr = VecRestoreArrayRead(Fi[k], &Fi_p[k]); CHKERRXX(ierr);
    }

    if(method==quadratic)
    {
      quadratic_interpolation(p4est, quad.p.piggy3.which_tree, quad, f, fdd, xyz_p, results, n_functions);
      return;
    }
    else if (method==quadratic_non_oscillatory)
    {
      quadratic_non_oscillatory_interpolation(p4est, quad.p.piggy3.which_tree, quad, f, fdd, xyz_p, results, n_functions);
      return;
    }
    else if (method==quadratic_non_oscillatory_continuous_v1)
    {
        quadratic_non_oscillatory_continuous_v1_interpolation(p4est, quad.p.piggy3.which_tree, quad, f, fdd, xyz_p, results, n_functions);
      return;
    }
    else if (method==quadratic_non_oscillatory_continuous_v2)
    {
        quadratic_non_oscillatory_continuous_v2_interpolation(p4est, quad.p.piggy3.which_tree, quad, f, fdd, xyz_p, results, n_functions);
        return;
    }
  }
  for (unsigned int k = 0; k < n_functions; ++k) {
    ierr = VecRestoreArrayRead(Fi[k], &Fi_p[k]); CHKERRXX(ierr);
  }
  linear_interpolation(p4est, quad.p.piggy3.which_tree, quad, f, xyz_p, results, n_functions);
  return;
}
