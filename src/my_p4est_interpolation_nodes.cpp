#ifdef P4_TO_P8
#include "my_p4est_interpolation_nodes.h"
#else
#include "my_p4est_interpolation_nodes.h"
#endif

#include "math.h"


my_p4est_interpolation_nodes_t::my_p4est_interpolation_nodes_t(const my_p4est_node_neighbors_t* ngbd_n)
  : my_p4est_interpolation_t(ngbd_n), nodes(ngbd_n->nodes),
    Fxx(NULL), Fyy(NULL),
    #ifdef P4_TO_P8
    Fzz(NULL),
    #endif
    method(linear)
{

}


void my_p4est_interpolation_nodes_t::set_input(Vec F, interpolation_method method) {
  Fi = F;
  this->method = method;
}


#ifdef P4_TO_P8
void my_p4est_interpolation_nodes_t::set_input(Vec F, Vec Fxx, Vec Fyy, Vec Fzz, interpolation_method method) {
#else
void my_p4est_interpolation_nodes_t::set_input(Vec F, Vec Fxx, Vec Fyy, interpolation_method method) {
#endif
  Fi = F;
  this->Fxx = Fxx;
  this->Fyy = Fyy;
#ifdef P4_TO_P8
  this->Fzz = Fzz;
#endif

  this->method = method;
}




#ifdef P4_TO_P8
double my_p4est_interpolation_nodes_t::operator ()(double x, double y, double z) const
#else
double my_p4est_interpolation_nodes_t::operator ()(double x, double y) const
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
  bool periodic[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
    periodic[dir] = (p4est->connectivity->tree_to_tree[P4EST_FACES*0 + 2*dir]!=0);
  for (short i=0; i<P4EST_DIM; i++){
    if (xyz_clip[i] > xyz_max[i]) xyz_clip[i] = periodic[i] ? xyz_clip[i]-(xyz_max[i]-xyz_min[i]) : xyz_max[i];
    if (xyz_clip[i] < xyz_min[i]) xyz_clip[i] = periodic[i] ? xyz_clip[i]+(xyz_max[i]-xyz_min[i]) : xyz_min[i];
  }
  
  const double *Fi_p;
  ierr = VecGetArrayRead(Fi, &Fi_p); CHKERRXX(ierr);

  const double *Fxx_p, *Fyy_p;
#ifdef P4_TO_P8
  bool use_precomputed_derivatives = Fxx != NULL && Fyy != NULL && Fzz != NULL;
  const double *Fzz_p;
#else
  bool use_precomputed_derivatives = Fxx != NULL && Fyy != NULL;
#endif

  if (use_precomputed_derivatives) {
    ierr = VecGetArrayRead(Fxx, &Fxx_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(Fyy, &Fyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArrayRead(Fzz, &Fzz_p); CHKERRXX(ierr);
#endif
  }

  double f  [P4EST_CHILDREN];
  double fdd[P4EST_CHILDREN*P4EST_DIM];

  p4est_quadrant_t best_match;
  std::vector<p4est_quadrant_t> remote_matches;
  int rank_found = ngbd_n->hierarchy->find_smallest_quadrant_containing_point(xyz_clip, best_match, remote_matches);
  
  if (rank_found == p4est->mpirank || (rank_found!=-1 && method==linear)) { // local quadrant
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
      f[i] = Fi_p[node_idx];
    }

      // compute derivatives
    if (method == quadratic || method == quadratic_non_oscillatory) {
      if (use_precomputed_derivatives) {
        for (short j = 0; j<P4EST_CHILDREN; j++) {
          p4est_locidx_t node_idx = nodes->local_nodes[quad_idx*P4EST_CHILDREN + j];

          fdd[j*P4EST_DIM + 0] = Fxx_p[node_idx];
          fdd[j*P4EST_DIM + 1] = Fyy_p[node_idx];
#ifdef P4_TO_P8
          fdd[j*P4EST_DIM + 2] = Fzz_p[node_idx];
#endif          
        }
      } else {
        for (short j = 0; j<P4EST_CHILDREN; j++) {
          p4est_locidx_t node_idx = nodes->local_nodes[quad_idx*P4EST_CHILDREN + j];
          const quad_neighbor_nodes_of_node_t& qnnn = ngbd_n->get_neighbors(node_idx);

          fdd[j*P4EST_DIM + 0] = qnnn.dxx_central(Fi_p);
          fdd[j*P4EST_DIM + 1] = qnnn.dyy_central(Fi_p);
#ifdef P4_TO_P8
          fdd[j*P4EST_DIM + 2] = qnnn.dzz_central(Fi_p);
#endif
        }
      }
    }
     
    // restore arrays and release remote_maches
    ierr = VecRestoreArrayRead(Fi, &Fi_p); CHKERRXX(ierr);
  
    if (use_precomputed_derivatives) {
      ierr = VecRestoreArrayRead(Fxx, &Fxx_p); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(Fyy, &Fyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecRestoreArrayRead(Fzz, &Fzz_p); CHKERRXX(ierr);
#endif
    }

    double xyz_q[P4EST_DIM];
    quad_xyz_fr_q(quad_idx, best_match.p.piggy3.which_tree, p4est, ghost, xyz_q);

    double xyz_p[P4EST_DIM];
    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      xyz_p[dir] = xyz[dir];
      if     (periodic[dir] && xyz[dir]-xyz_q[dir]>(xyz_max[dir]-xyz_min[dir])/2) xyz_p[dir] -= xyz_max[dir]-xyz_min[dir];
      else if(periodic[dir] && xyz_q[dir]-xyz[dir]>(xyz_max[dir]-xyz_min[dir])/2) xyz_p[dir] += xyz_max[dir]-xyz_min[dir];
    }

    double value=0;
    if (method == linear) {
      value = linear_interpolation(p4est, best_match.p.piggy3.which_tree, best_match, f, xyz_p);
    } else if (method == quadratic) {
      value = quadratic_interpolation(p4est, best_match.p.piggy3.which_tree, best_match, f, fdd, xyz_p);
    } else if (method == quadratic_non_oscillatory) {
      value = quadratic_non_oscillatory_interpolation(p4est, best_match.p.piggy3.which_tree, best_match, f, fdd, xyz_p);
    }

    return value;
    
  } else {
    ierr = VecRestoreArrayRead(Fi, &Fi_p); CHKERRXX(ierr);
    if (use_precomputed_derivatives) {
      ierr = VecRestoreArrayRead(Fxx, &Fxx_p); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(Fyy, &Fyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecRestoreArrayRead(Fzz, &Fzz_p); CHKERRXX(ierr);
#endif
    }

    std::cout << "Buuug " << x << ", " << y << ", found " << remote_matches.size() << " possible matches" << std::endl;
    std::ostringstream oss;
    oss << "[ERROR]: Point (" << x << "," << y <<
       #ifdef P4_TO_P8
           "," << z <<
       #endif
           ") is not locally owned by processor. "
           << p4est->mpirank;
    if (rank_found != -1) {
      oss << "Remote owner's rank = " << rank_found << std::endl;
    } else {
      oss << "Possible remote owners are = ";
      for (size_t i = 0; i<remote_matches.size() - 1; i++) {
        oss << remote_matches[i].p.piggy1.owner_rank << ", ";
      }
      oss << remote_matches[remote_matches.size() - 1].p.piggy1.owner_rank << "." << std::endl;
    }
    throw std::invalid_argument(oss.str());
  }
}


double my_p4est_interpolation_nodes_t::interpolate(const p4est_quadrant_t &quad, const double *xyz) const
{
  PetscErrorCode ierr;

  p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, quad.p.piggy3.which_tree);
  p4est_locidx_t quad_idx = quad.p.piggy3.local_num + tree->quadrants_offset;

  double f[P4EST_CHILDREN];

  const double *Fi_p;
  ierr = VecGetArrayRead(Fi, &Fi_p); CHKERRXX(ierr);
  for (short j = 0; j<P4EST_CHILDREN; j++) {
    p4est_locidx_t node_idx = nodes->local_nodes[quad_idx*P4EST_CHILDREN + j];
    f[j] = Fi_p[node_idx];
  }

  /* enforce periodicity if necessary */
  bool periodic[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
    periodic[dir] = (p4est->connectivity->tree_to_tree[P4EST_FACES*0 + 2*dir]!=0);
  double xyz_q[P4EST_DIM];
  quad_xyz_fr_q(quad_idx, quad.p.piggy3.which_tree, p4est, ghost, xyz_q);

  double xyz_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    xyz_p[dir] = xyz[dir];
    if     (periodic[dir] && xyz[dir]-xyz_q[dir]>(xyz_max[dir]-xyz_min[dir])/2) xyz_p[dir] -= xyz_max[dir]-xyz_min[dir];
    else if(periodic[dir] && xyz_q[dir]-xyz[dir]>(xyz_max[dir]-xyz_min[dir])/2) xyz_p[dir] += xyz_max[dir]-xyz_min[dir];
  }

  /* compute derivatives */
  if (method == quadratic || method == quadratic_non_oscillatory)
  {
    double fdd[P4EST_CHILDREN*P4EST_DIM];

#ifdef P4_TO_P8
    if (Fxx!=NULL && Fyy!=NULL && Fzz!=NULL)
#else
    if (Fxx!=NULL && Fyy!=NULL)
#endif
    {
      const double *Fxx_p;
      ierr = VecGetArrayRead(Fxx, &Fxx_p); CHKERRXX(ierr);
      const double *Fyy_p;
      ierr = VecGetArrayRead(Fyy, &Fyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
      const double *Fzz_p;
      ierr = VecGetArrayRead(Fzz, &Fzz_p); CHKERRXX(ierr);
#endif
      for (short j = 0; j<P4EST_CHILDREN; j++) {
        p4est_locidx_t node_idx = nodes->local_nodes[quad_idx*P4EST_CHILDREN + j];

        fdd[j*P4EST_DIM + 0] = Fxx_p[node_idx];
        fdd[j*P4EST_DIM + 1] = Fyy_p[node_idx];
#ifdef P4_TO_P8
        fdd[j*P4EST_DIM + 2] = Fzz_p[node_idx];
#endif
      }

      ierr = VecRestoreArrayRead(Fxx, &Fxx_p); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(Fyy, &Fyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecRestoreArrayRead(Fzz, &Fzz_p); CHKERRXX(ierr);
#endif
    }
    else
    {
      for (short j = 0; j<P4EST_CHILDREN; j++)
      {
        p4est_locidx_t node_idx = nodes->local_nodes[quad_idx*P4EST_CHILDREN + j];
        const quad_neighbor_nodes_of_node_t& qnnn = ngbd_n->get_neighbors(node_idx);

        fdd[j*P4EST_DIM + 0] = qnnn.dxx_central(Fi_p);
        fdd[j*P4EST_DIM + 1] = qnnn.dyy_central(Fi_p);
#ifdef P4_TO_P8
        fdd[j*P4EST_DIM + 2] = qnnn.dzz_central(Fi_p);
#endif
      }
    }
    ierr = VecRestoreArrayRead(Fi, &Fi_p); CHKERRXX(ierr);

    if(method==quadratic) return quadratic_interpolation(p4est, quad.p.piggy3.which_tree, quad, f, fdd, xyz_p);
    else  return quadratic_non_oscillatory_interpolation(p4est, quad.p.piggy3.which_tree, quad, f, fdd, xyz_p);
  }
  ierr = VecRestoreArrayRead(Fi, &Fi_p); CHKERRXX(ierr);

  return linear_interpolation(p4est, quad.p.piggy3.which_tree, quad, f, xyz_p);
}
