#ifndef MY_P4EST_XGFM_CELLS_H
#define MY_P4EST_XGFM_CELLS_H

#include <petsc.h>

#ifdef P4_TO_P8
#include <src/my_p8est_cell_neighbors.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_faces.h>
#include <p8est_nodes.h>
#else
#include <src/my_p4est_cell_neighbors.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_faces.h>
#include <p4est_nodes.h>
#endif

#include <src/matrix.h>
#include <src/my_p4est_solve_lsqr.h>
#include <algorithm>
#include <map>

/*!
 * \brief The interface_neighbor struct contains all relevant data regarding
 * interface-neighbor, i.e. intersection between the interface and the segment
 * joining the actual neighbor cell of interest.
 * --> used in the extension routine
 * phi_q: value of the level-set at the cell-center of interest;
 * phi_tmp: value of the level-set at the center of the actual neighbor cell (across the interface);
 * theta: fraction of the grid spacing covered by the domain in which the cell of interest is;
 * mu_this_side: value of the diffusion coefficient as seen by the cell of interest;
 * mu_other_side: value of the diffusion coefficient as seen by the cell of interest;
 * int_value: value of u^- (or u^+) at the interface point;
 * quad_tmp_idx: local index of the actual neighbor cell in the computational grid (coarse grid)
 * mid_point_fine_node_idx: local index of the grid node in between those two cells in the interface_capturing grid (fine grid)
 * quad_fine_node_idx: local index of the grid node that coincides with the center of the cell of interest in the interface_capturing grid (fine grid)
 * tmp_fine_node_idx: local index of the grid node that coincides with the center of the actual neighbor cell across the interface in the interface_capturing grid (fine grid)
 */
struct interface_neighbor
{
  double  phi_q;
  double  phi_tmp;
  double  theta;
  double  mu_this_side;
  double  mu_other_side;
  double  int_value;
  p4est_locidx_t quad_tmp_idx;
  p4est_locidx_t mid_point_fine_node_idx;
  p4est_locidx_t quad_fine_node_idx;
  p4est_locidx_t tmp_fine_node_idx;
#ifdef DEBUG
  bool is_consistent_with_neighbor_across(const interface_neighbor nb_across) const
  {
    bool to_return = ((phi_q > 0.0) && (nb_across.phi_q <= 0.0)) || ((phi_q <= 0.0) && (nb_across.phi_q > 0.0));
    to_return = to_return && (fabs(phi_q - nb_across.phi_tmp) < EPS*MAX(fabs(phi_q), fabs(phi_tmp)));
    to_return = to_return && (fabs(phi_tmp - nb_across.phi_q) < EPS*MAX(fabs(phi_q), fabs(phi_tmp)));
    to_return = to_return && (fabs(theta - (1.0 - nb_across.theta)) < EPS);
    to_return = to_return && (fabs(mu_this_side - nb_across.mu_other_side) < EPS*MAX(mu_this_side, mu_other_side));
    to_return = to_return && (fabs(mu_other_side - nb_across.mu_this_side) < EPS*MAX(mu_this_side, mu_other_side));
    to_return = to_return && (fabs(int_value - nb_across.int_value) < EPS*MAX(fabs(int_value), 1.0));
    return to_return;
  }
#endif
};

#ifdef DEBUG
struct which_interface_nb
{
  p4est_locidx_t loc_idx;
  int dir;
};
#endif

struct extension_matrix_entry
{
  p4est_locidx_t loc_idx;
  double coeff;
};
struct extension_interface_value_entry
{
  int dir;
  double coeff;
};

struct extension_affine_map
{
  bool too_close;
  int forced_interface_value_dir;
  double diag_entry, dtau, phi_q;
  std::vector<extension_interface_value_entry> interface_entries;
  std::vector<extension_matrix_entry> quad_entries;
  extension_affine_map() {
    interface_entries.resize(0);
    quad_entries.resize(0);
    too_close = false;
    diag_entry = 0.0;
    forced_interface_value_dir = -1;
  }
};

struct interpolation_factor
{
  p4est_locidx_t quad_idx;
  double weight;
  bool operator ==(const interpolation_factor &other) const {return (this->quad_idx == other.quad_idx);}
};

class my_p4est_xgfm_cells_t
{
#ifdef P4_TO_P8
  class constant_scalar: public CF_3
#else
  class constant_scalar: public CF_2
#endif
  {
  private:
    double value;
  public:
    constant_scalar(){ value = -1.0; }
    constant_scalar(double value_){ value = value_; }
    void set(double value_) {this->value = value_; }
#ifdef P4_TO_P8
    double operator()(double, double, double) const
#else
    double operator()(double, double) const
#endif
    {
      return value;
    }
    double operator()(double *) const
    {
      return value;
    }
    double operator()(p4est_locidx_t) const
    {
      return value;
    }
    double get_value() const { return value;}
  };


  // defined on/from the computational grid
  const my_p4est_cell_neighbors_t *cell_ngbd;
  const my_p4est_node_neighbors_t *node_ngbd;
  p4est_t *p4est;
  p4est_nodes_t *nodes;
  p4est_ghost_t *ghost;
  // computational domain
  my_p4est_brick_t *brick;
  double xyz_min[P4EST_DIM], xyz_max[P4EST_DIM], tree_dimensions[P4EST_DIM];
  double dxyz_min[P4EST_DIM], dxyz_min_fine[P4EST_DIM];
  double d_min, diag_min, cell_volume_min;
  bool periodicity[P4EST_DIM];
  // equation parameters
  constant_scalar mu_m, mu_p, add_diag_m, add_diag_p;
  bool mu_m_is_larger, mu_m_and_mu_p_equal;
  // vectors of cell-centered values
  Vec rhs; // set by user, not owned by this object (hence not destroyed at destruction)
  Vec solution; // constructed and owned by solver except if returned to user, hence destroyed at destruction if not returned.

  // defined on/from the fine, interface-capturing grid
  p4est_t *fine_p4est;
  p4est_nodes_t *fine_nodes;
  p4est_ghost_t *fine_ghost;
  const my_p4est_node_neighbors_t *fine_node_ngbd;
  Vec phi, jump_u, normals[P4EST_DIM], phi_second_der[P4EST_DIM];
  bool phi_has_been_set, normals_have_been_set, mus_have_been_set, jumps_have_been_set, second_derivatives_of_phi_are_set;
  Vec corrected_rhs; // constructed and owned by solver, hence destroyed at destruction
  Vec jump_mu_grad_u[P4EST_DIM]; // constructed and owned by solver, hence destroyed at destruction
  Vec extension_cell_values, extension_on_fine_nodes; // constructed and owned by solver, hence destroyed at destruction

#ifdef P4_TO_P8
  BoundaryConditions3D *bc;
#else
  BoundaryConditions2D *bc;
#endif

  // solver monitoring
  std::vector<PetscInt>   numbers_of_ksp_iterations;
  std::vector<double> max_corrections, relative_residuals;


  // PETSc objects
  bool nullspace_use_fixed_point;
  Mat A;
  MatNullSpace A_null_space;
  Vec null_space;
  p4est_gloidx_t fixed_value_idx_g;
  p4est_gloidx_t fixed_value_idx_l;
  KSP ksp;
  PetscErrorCode ierr;

  // flags
  bool is_matrix_built;
  int matrix_has_nullspace; // type int because of required MPI operations
  bool interface_values_are_set, map_of_neighbors_is_initialized;
  bool solution_is_set, use_initial_guess;
  const bool activate_x;

  // map of interface_neighbors:
  // key    = local index of the considered quadrant
  // value  = another map such that
  //      - key   = direction of the interface neighbor;
  //      - value = structure encapsulating theta (normalized distance to the interface) and the interface value.
  std::map<p4est_locidx_t, std::map<int, interface_neighbor> > map_of_interface_neighbors;

  // memorized local extension operators
  std::vector<extension_affine_map> extension_entries;
  bool extension_entries_are_set;
  // memorized local interpolation operators
  std::vector< std::vector<interpolation_factor> > local_interpolator;
  bool local_interpolator_is_set;

  // disallow copy ctr and copy assignment
  my_p4est_xgfm_cells_t(const my_p4est_xgfm_cells_t& other);
  my_p4est_xgfm_cells_t& operator=(const my_p4est_xgfm_cells_t& other);

  // internal procedures
  void preallocate_matrix();
  void setup_negative_laplace_matrix_with_jumps();
  void setup_negative_laplace_rhsvec_with_jumps();

  inline p4est_gloidx_t compute_global_index(p4est_locidx_t quad_idx) const
  {
    if(quad_idx<p4est->local_num_quadrants)
      return p4est->global_first_quadrant[p4est->mpirank] + quad_idx;

    const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&ghost->ghosts, quad_idx-p4est->local_num_quadrants);
    return p4est->global_first_quadrant[quad_find_ghost_owner(ghost, quad_idx-p4est->local_num_quadrants)] + quad->p.piggy3.local_num;
  }


  bool multilinear_interpolation_weights(const p4est_topidx_t const_tree_qxyz[], const p4est_qcoord_t const_qxyz_node[], p4est_locidx_t node_indices[], double weights[])
  {
    splitting_criteria_t *data_fine   = (splitting_criteria_t*) fine_p4est->user_pointer;
    // create the logical coordinates of the fine node clamped in a quadrant of the fine grid
    // Far right/top/front wall nodes are mapped to the other mirror wall in case of periodic boundaries;
    p4est_topidx_t tree_qxyz[P4EST_DIM];
    p4est_qcoord_t qxyz_node[P4EST_DIM];
    // We also get the logical coordinates of the center of a (fictitious) finest quadrant on the fine grid containing the above node, called the "clamped node".
    // If accessible, we take the fictitious ppp quad. If not (because of walls), we take the mirror (p --> m)
    p4est_qcoord_t qxyz_clamped[P4EST_DIM];
    for (short dim = 0; dim < P4EST_DIM; ++dim) {
      P4EST_ASSERT((const_tree_qxyz[dim] >=0) && (const_tree_qxyz[dim] < brick->nxyztrees[dim]) && (const_qxyz_node[dim] >=0) && (const_qxyz_node[dim] <= P4EST_ROOT_LEN));
      qxyz_node[dim]          = const_qxyz_node[dim];
      tree_qxyz[dim]          = const_tree_qxyz[dim];
      if(qxyz_node[dim] <= P4EST_ROOT_LEN-P4EST_QUADRANT_LEN(data_fine->max_lvl))
        qxyz_clamped[dim]     = qxyz_node[dim] + P4EST_QUADRANT_LEN(data_fine->max_lvl+1);
      else
      {
        P4EST_ASSERT(qxyz_node[dim] == P4EST_ROOT_LEN);
        if(tree_qxyz[dim] < (brick->nxyztrees[dim] - 1))
        {
          tree_qxyz[dim]++;
          qxyz_node[dim]     -= P4EST_ROOT_LEN;
          qxyz_clamped[dim]   = qxyz_node[dim] + P4EST_QUADRANT_LEN(data_fine->max_lvl+1);
        }
        else
        {
          P4EST_ASSERT(tree_qxyz[dim] == brick->nxyztrees[dim]-1);
          if(periodicity[dim])
          {
            tree_qxyz[dim]    = 0;
            qxyz_node[dim]   -= P4EST_ROOT_LEN;
            qxyz_clamped[dim] = qxyz_node[dim] + P4EST_QUADRANT_LEN(data_fine->max_lvl+1);
          }
          else
            qxyz_clamped[dim] = qxyz_node[dim] - P4EST_QUADRANT_LEN(data_fine->max_lvl+1);
        }
      }
    }
#ifdef DEBUG
    for (short dim = 0; dim < P4EST_DIM; ++dim)
      P4EST_ASSERT((tree_qxyz[dim] >=0) && (tree_qxyz[dim] < brick->nxyztrees[dim]) &&
                   (qxyz_clamped[dim] > 0) && (qxyz_clamped[dim] < P4EST_ROOT_LEN) &&
                   (qxyz_node[dim] >=0) && (qxyz_node[dim] <= (((tree_qxyz[dim] < brick->nxyztrees[dim]-1) || periodicity[dim])? (P4EST_ROOT_LEN-P4EST_QUADRANT_LEN(data_fine->max_lvl)): P4EST_ROOT_LEN)));
#endif

    double xyz_clamped[P4EST_DIM];
    for (short dim = 0; dim < P4EST_DIM; ++dim)
      xyz_clamped[dim]   = xyz_min[dim] + tree_dimensions[dim]*(tree_qxyz[dim] + ((double) qxyz_clamped[dim])/((double) P4EST_ROOT_LEN));
#ifdef DEBUG
    for (short dim = 0; dim < P4EST_DIM; ++dim)
      P4EST_ASSERT((xyz_clamped[dim] > xyz_min[dim]) && (xyz_clamped[dim] < xyz_max[dim]));
#endif
    p4est_quadrant_t best_match;
    std::vector<p4est_quadrant_t> remote_matches;
    int rank_found = fine_node_ngbd->hierarchy->find_smallest_quadrant_containing_point(xyz_clamped, best_match, remote_matches);
    P4EST_ASSERT(rank_found != -1);
    p4est_locidx_t quad_idx;
    if(rank_found==fine_p4est->mpirank)
    {
      p4est_tree_t *tree = p4est_tree_array_index(fine_p4est->trees, best_match.p.piggy3.which_tree);
      quad_idx = best_match.p.piggy3.local_num + tree->quadrants_offset;
    }
    else
      quad_idx = best_match.p.piggy3.local_num + fine_p4est->local_num_quadrants;

    p4est_qcoord_t best_match_qsize = P4EST_QUADRANT_LEN(best_match.level);
    p4est_qcoord_t best_match_qxyz_min[P4EST_DIM];
#ifdef DEBUG
    p4est_topidx_t v_m  = p4est->connectivity->tree_to_vertex[best_match.p.piggy3.which_tree*P4EST_CHILDREN + 0];
    p4est_topidx_t best_match_tree_qxyz[P4EST_DIM];
    for (short dim = 0; dim < P4EST_DIM; ++dim) {
      double rel_top_idx = (p4est->connectivity->vertices[3*v_m + dim] - xyz_min[dim])/tree_dimensions[dim];
      P4EST_ASSERT(fabs(rel_top_idx - floor(rel_top_idx)) < EPS);
      best_match_tree_qxyz[dim] = (p4est_topidx_t) floor(rel_top_idx);
    }
#endif
    best_match_qxyz_min[0] = best_match.x;
    best_match_qxyz_min[1] = best_match.y;
#ifdef P4_TO_P8
    best_match_qxyz_min[2] = best_match.z;
#endif
#ifdef DEBUG
    for (short dim = 0; dim < P4EST_DIM; ++dim)
      P4EST_ASSERT((best_match_tree_qxyz[dim] == tree_qxyz[dim]) &&
                   (best_match_qxyz_min[dim] <= qxyz_node[dim]) && (best_match_qxyz_min[dim]+best_match_qsize >= qxyz_node[dim]));
#endif

    bool it_is_a_node_on_the_fine_grid = true;
    double d_m[P4EST_DIM];
    for (short dim = 0; dim < P4EST_DIM; ++dim){
      d_m[dim] = ((double) (qxyz_node[dim] - best_match_qxyz_min[dim]))/((double) best_match_qsize);
      it_is_a_node_on_the_fine_grid = it_is_a_node_on_the_fine_grid && ((qxyz_node[dim] - best_match_qxyz_min[dim] == 0) || (qxyz_node[dim] - best_match_qxyz_min[dim] == best_match_qsize));
    }

    short ccc_max = 0;
    double max_weight = 0.0;
    double sum_weight = 0.0;
    for (short ccc = 0; ccc < P4EST_CHILDREN; ++ccc) {
      node_indices[ccc] = fine_nodes->local_nodes[quad_idx*P4EST_CHILDREN + ccc];
#ifdef P4_TO_P8
      weights[ccc]      = (((ccc%2 == 0)? (1.0 - d_m[0]) : d_m[0])*(((ccc/2)%2 == 0) ? (1.0 - d_m[1]) : d_m[1])*(((ccc/4)%2 == 0) ? (1.0 - d_m[2]) : d_m[2]));
#else
      weights[ccc]      = (((ccc%2 == 0)? (1.0 - d_m[0]) : d_m[0])*(((ccc/2)%2 == 0) ? (1.0 - d_m[1]) : d_m[1]));
#endif
      if(weights[ccc] > max_weight)
      {
        max_weight      = weights[ccc];
        ccc_max         = ccc;
      }
      sum_weight       += weights[ccc];
    }
    P4EST_ASSERT(fabs(sum_weight - 1.0) < EPS);
    if(ccc_max != 0)
    {
      // put most important weight first
      double weight_tmp           = weights[0];
      p4est_locidx_t node_idx_tmp = node_indices[0];
      weights[0]                  = weights[ccc_max];
      node_indices[0]             = node_indices[ccc_max];
      weights[ccc_max]            = weight_tmp;
      node_indices[ccc_max]       = node_idx_tmp;
    }

    P4EST_ASSERT((it_is_a_node_on_the_fine_grid? fabs(weights[0] - 1.0) < EPS:true));
    // return true if the point is a node of the fine grid
    return it_is_a_node_on_the_fine_grid;
  }

  p4est_locidx_t fine_idx_of_direct_neighbor(const quad_neighbor_nodes_of_node_t& qnnn, int dir)
  {
    switch (dir) {
    case dir::f_m00:
#ifdef P4_TO_P8
      return ((fabs(qnnn.d_m00_m0) < EPS*tree_dimensions[1])? ((fabs(qnnn.d_m00_0m) < EPS*tree_dimensions[2])? qnnn.node_m00_mm : ((fabs(qnnn.d_m00_0p) < EPS*tree_dimensions[2])? qnnn.node_m00_mp: -1))
          :((fabs(qnnn.d_m00_p0) < EPS*tree_dimensions[1])? ((fabs(qnnn.d_m00_0m) < EPS*tree_dimensions[2])? qnnn.node_m00_pm : ((fabs(qnnn.d_m00_0p) < EPS*tree_dimensions[2])? qnnn.node_m00_pp: -1)) : -1));
#else
      return ((fabs(qnnn.d_m00_m0) < EPS*tree_dimensions[1]) ? qnnn.node_m00_mm : ((fabs(qnnn.d_m00_p0) < EPS*tree_dimensions[1]) ? qnnn.node_m00_pm : -1));
#endif
      break;
    case dir::f_p00:
#ifdef P4_TO_P8
      return ((fabs(qnnn.d_p00_m0) < EPS*tree_dimensions[1])? ((fabs(qnnn.d_p00_0m) < EPS*tree_dimensions[2])? qnnn.node_p00_mm : ((fabs(qnnn.d_p00_0p) < EPS*tree_dimensions[2])? qnnn.node_p00_mp: -1))
          :((fabs(qnnn.d_p00_p0) < EPS*tree_dimensions[1])? ((fabs(qnnn.d_p00_0m) < EPS*tree_dimensions[2])? qnnn.node_p00_pm : ((fabs(qnnn.d_p00_0p) < EPS*tree_dimensions[2])? qnnn.node_p00_pp: -1)) : -1));
#else
      return ((fabs(qnnn.d_p00_m0) < EPS*tree_dimensions[1]) ? qnnn.node_p00_mm : ((fabs(qnnn.d_p00_p0) < EPS*tree_dimensions[1]) ? qnnn.node_p00_pm : -1));
#endif
      break;
    case dir::f_0m0:
#ifdef P4_TO_P8
      return ((fabs(qnnn.d_0m0_m0) < EPS*tree_dimensions[0])? ((fabs(qnnn.d_0m0_0m) < EPS*tree_dimensions[2])? qnnn.node_0m0_mm : ((fabs(qnnn.d_0m0_0p) < EPS*tree_dimensions[2])? qnnn.node_0m0_mp: -1))
          :((fabs(qnnn.d_0m0_p0) < EPS*tree_dimensions[0])? ((fabs(qnnn.d_0m0_0m) < EPS*tree_dimensions[2])? qnnn.node_0m0_pm : ((fabs(qnnn.d_0m0_0p) < EPS*tree_dimensions[2])? qnnn.node_0m0_pp: -1)) : -1));
#else
      return ((fabs(qnnn.d_0m0_m0) < EPS*tree_dimensions[0]) ? qnnn.node_0m0_mm : ((fabs(qnnn.d_0m0_p0) < EPS*tree_dimensions[0]) ? qnnn.node_0m0_pm : -1));
#endif
      break;
    case dir::f_0p0:
#ifdef P4_TO_P8
      return ((fabs(qnnn.d_0p0_m0) < EPS*tree_dimensions[0])? ((fabs(qnnn.d_0p0_0m) < EPS*tree_dimensions[2])? qnnn.node_0p0_mm : ((fabs(qnnn.d_0p0_0p) < EPS*tree_dimensions[2])? qnnn.node_0p0_mp: -1))
          :((fabs(qnnn.d_0p0_p0) < EPS*tree_dimensions[0])? ((fabs(qnnn.d_0p0_0m) < EPS*tree_dimensions[2])? qnnn.node_0p0_pm : ((fabs(qnnn.d_0p0_0p) < EPS*tree_dimensions[2])? qnnn.node_0p0_pp: -1)) : -1));
#else
      return ((fabs(qnnn.d_0p0_m0) < EPS*tree_dimensions[0]) ? qnnn.node_0p0_mm : ((fabs(qnnn.d_0p0_p0) < EPS*tree_dimensions[0]) ? qnnn.node_0p0_pm : -1));
#endif
      break;
#ifdef P4_TO_P8
    case dir::f_00m:
      return ((fabs(qnnn.d_00m_m0) < EPS*tree_dimensions[0])? ((fabs(qnnn.d_00m_0m) < EPS*tree_dimensions[1])? qnnn.node_00m_mm : ((fabs(qnnn.d_00m_0p) < EPS*tree_dimensions[1])? qnnn.node_00m_mp: -1))
          :((fabs(qnnn.d_00m_p0) < EPS*tree_dimensions[0])? ((fabs(qnnn.d_00m_0m) < EPS*tree_dimensions[1])? qnnn.node_00m_pm : ((fabs(qnnn.d_00m_0p) < EPS*tree_dimensions[1])? qnnn.node_00m_pp: -1)) : -1));
      break;
    case dir::f_00p:
          return ((fabs(qnnn.d_00p_m0) < EPS*tree_dimensions[0])? ((fabs(qnnn.d_00p_0m) < EPS*tree_dimensions[1])? qnnn.node_00p_mm : ((fabs(qnnn.d_00p_0p) < EPS*tree_dimensions[1])? qnnn.node_00p_mp: -1))
              :((fabs(qnnn.d_00p_p0) < EPS*tree_dimensions[0])? ((fabs(qnnn.d_00p_0m) < EPS*tree_dimensions[1])? qnnn.node_00p_pm : ((fabs(qnnn.d_00p_0p) < EPS*tree_dimensions[1])? qnnn.node_00p_pp: -1)) : -1));
      break;
#endif
    default:
      throw std::invalid_argument("my_p4est_xgfm_cells_t::fine_idx_of_direct_neighbor(): unknown direction");
    }
  }

  inline bool is_quad_in_quad(const double xyz_candidate_container_quad[], const double dxyz_candidate_container_quad[], const int8_t& level_candidate_container_quad, const double xyz_other_quad[], const int8_t& level_other_quad) const
  {
      return ((xyz_other_quad[0] > xyz_candidate_container_quad[0] - .5*dxyz_candidate_container_quad[0])
        && (xyz_other_quad[0] < xyz_candidate_container_quad[0] + .5*dxyz_candidate_container_quad[0])
        && (xyz_other_quad[1] > xyz_candidate_container_quad[1] - .5*dxyz_candidate_container_quad[1])
        && (xyz_other_quad[1] < xyz_candidate_container_quad[1] + .5*dxyz_candidate_container_quad[1])
    #ifdef P4_TO_P8
        && (xyz_other_quad[2] > xyz_candidate_container_quad[2] - .5*dxyz_candidate_container_quad[2])
        && (xyz_other_quad[2] < xyz_candidate_container_quad[2] + .5*dxyz_candidate_container_quad[2])
    #endif
        && (level_other_quad >= level_candidate_container_quad));
  }

  inline double get_lsqr_interpolation_at(const double xyz[], const std::vector<p4est_quadrant_t>& ngbd_of_coarse_cells, const double *coarse_cell_data_read_p, std::vector<interpolation_factor>& interpolator)
  {
    matrix_t A;
    std::vector<double> lsqr_rhs;
    std::vector<double> data_points[P4EST_DIM];
    double scaling = DBL_MAX;

    P4EST_ASSERT(ngbd_of_coarse_cells.size() > 0);
    interpolator.resize(0);
    for(unsigned int m=0; m<ngbd_of_coarse_cells.size(); ++m)
      scaling = MIN(scaling, (double)P4EST_QUADRANT_LEN(ngbd_of_coarse_cells[m].level)/(double)P4EST_ROOT_LEN);

    lsqr_rhs.resize(0);
    for (short dim = 0; dim < P4EST_DIM; ++dim)
      data_points[dim].resize(0);
#ifdef P4_TO_P8
    A.resize(1,10);
    scaling *= .5*MIN(tree_dimensions[0], tree_dimensions[1], tree_dimensions[2]);
#else
    A.resize(1,6);
    scaling *= .5*MIN(tree_dimensions[0], tree_dimensions[1]);
#endif

    for(unsigned int m=0; m < ngbd_of_coarse_cells.size(); m++)
    {
      p4est_locidx_t qm_idx = ngbd_of_coarse_cells[m].p.piggy3.local_num;
      interpolation_factor interp_term; interp_term.quad_idx = qm_idx;
      if(std::find(interpolator.begin(), interpolator.end(), interp_term)==interpolator.end())
      {
        double xyz_t[P4EST_DIM];

        xyz_t[0] = quad_x_fr_q(ngbd_of_coarse_cells[m].p.piggy3.local_num, ngbd_of_coarse_cells[m].p.piggy3.which_tree, p4est, ghost);
        xyz_t[1] = quad_y_fr_q(ngbd_of_coarse_cells[m].p.piggy3.local_num, ngbd_of_coarse_cells[m].p.piggy3.which_tree, p4est, ghost);
#ifdef P4_TO_P8
        xyz_t[2] = quad_z_fr_q(ngbd_of_coarse_cells[m].p.piggy3.local_num, ngbd_of_coarse_cells[m].p.piggy3.which_tree, p4est, ghost);
#endif

        for(short dim=0; dim<P4EST_DIM; ++dim)
          xyz_t[dim] = (xyz_t[dim] - xyz[dim]) / scaling;

#ifdef P4_TO_P8
        double w = MAX(1.0e-6,1./MAX(1.0e-6,sqrt(SQR(xyz_t[0]) + SQR(xyz_t[1]) + SQR(xyz_t[2]))));
#else
        double w = MAX(1.0e-6,1./MAX(1.0e-6,sqrt(SQR(xyz_t[0]) + SQR(xyz_t[1]))));
#endif
#ifdef P4_TO_P8
        A.set_value(interpolator.size(), 0, 1                 * w);
        A.set_value(interpolator.size(), 1, xyz_t[0]          * w);
        A.set_value(interpolator.size(), 2, xyz_t[1]          * w);
        A.set_value(interpolator.size(), 3, xyz_t[2]          * w);
        A.set_value(interpolator.size(), 4, xyz_t[0]*xyz_t[0] * w);
        A.set_value(interpolator.size(), 5, xyz_t[0]*xyz_t[1] * w);
        A.set_value(interpolator.size(), 6, xyz_t[0]*xyz_t[2] * w);
        A.set_value(interpolator.size(), 7, xyz_t[1]*xyz_t[1] * w);
        A.set_value(interpolator.size(), 8, xyz_t[1]*xyz_t[2] * w);
        A.set_value(interpolator.size(), 9, xyz_t[2]*xyz_t[2] * w);
#else
        A.set_value(interpolator.size(), 0, 1                 * w);
        A.set_value(interpolator.size(), 1, xyz_t[0]          * w);
        A.set_value(interpolator.size(), 2, xyz_t[1]          * w);
        A.set_value(interpolator.size(), 3, xyz_t[0]*xyz_t[0] * w);
        A.set_value(interpolator.size(), 4, xyz_t[0]*xyz_t[1] * w);
        A.set_value(interpolator.size(), 5, xyz_t[1]*xyz_t[1] * w);
#endif
        interp_term.weight = w;
        interpolator.push_back(interp_term);
        lsqr_rhs.push_back(coarse_cell_data_read_p[qm_idx]*w);
        for(short dim=0; dim<P4EST_DIM; ++dim)
        {
          size_t kk;
          for (kk = 0; kk < data_points[dim].size(); ++kk)
            if(fabs(data_points[dim][kk] - xyz_t[dim]) < EPS)
              break;
          if(kk == data_points[dim].size())
            data_points[dim].push_back(xyz_t[dim]);
        }
      }
    }
    double abs_max = A.scale_by_maxabs(lsqr_rhs);

    P4EST_ASSERT(interpolator.size()>0);
    std::vector<double> interp_weights;
#ifdef P4_TO_P8
    double value_to_return = solve_lsqr_system_and_get_coefficients(A, lsqr_rhs, data_points[0].size(), data_points[1].size(), data_points[2].size(), interp_weights);
#else
    double value_to_return = solve_lsqr_system_and_get_coefficients(A, lsqr_rhs, data_points[0].size(), data_points[1].size(), interp_weights);
#endif
    interpolator.resize(interp_weights.size());
    for (size_t ii = 0; ii < interpolator.size(); ++ii)
      interpolator[ii].weight *= interp_weights[ii]/abs_max;

    return value_to_return;
  }

  void interpolate_cell_field_at_fine_node(const p4est_locidx_t &fine_node_idx, const p4est_indep_t* ni,
                                           const double *cell_field_read_p, double *fine_node_field_p,
                                           const bool& super_fine_node, const p4est_quadrant_t* coarse_quad, const p4est_locidx_t& coarse_quad_idx,
                                           const p4est_topidx_t& tree_idx_for_coarse_quad, const std::vector<p4est_quadrant_t>& ngbd_of_coarse_cells);

  // using PDE extrapolation
  void extend_interface_values(const double *solution_p, Vec new_cell_extension, const double* extension_on_fine_nodes_read_p, double threshold = 1.0e-10, uint niter_max = 20);
  // multilinear interpolation at superfine nodes, lsqr interpolation otherwise, no interface consideration
  void interpolate_coarse_cell_field_to_fine_nodes(const double *cell_field_read_p, Vec fine_node_field);
  // get the correction jump terms
  void get_corrected_rhs(Vec corrected_rhs_p, const double *fine_extension_interface_values_read_p);

#ifdef DEBUG
  int is_map_consistent() const;
#endif

  void correct_jump_mu_grad_u();

  void set_jump_mu_grad_u_for_nodes(const std::vector<p4est_locidx_t>& list_of_node_indices, double *jump_mu_grad_u_p[P4EST_DIM], const double *jump_normal_flux_read_p, const double *normals_read_p[P4EST_DIM], const double *jump_u_read_p);

  bool interface_neighbor_is_found(const p4est_locidx_t& quad_idx, const int& dir, interface_neighbor& int_nb);
  interface_neighbor get_interface_neighbor(const p4est_locidx_t& quad_idx, const int& dir,
                                            const p4est_locidx_t& tmp_quad_idx,
                                            const p4est_locidx_t& quad_fine_node_idx,
                                            const p4est_locidx_t& tmp_fine_node_idx,
                                            const double *phi_read_p, const double *phi_dd_read_p[P4EST_DIM]);

  void update_interface_values(Vec new_cell_extension, const double *solution_read_p, const double *extension_on_fine_nodes_read_p);
  void cell_TVD_extension_of_interface_values(Vec new_cell_extension, const double& threshold, const uint& niter_max);
public:

  /* ! VERY IMPORTANT ! QUALITY OF THE GRID, LAYER OF FINE CELLS, and so on... */

  my_p4est_xgfm_cells_t(const my_p4est_cell_neighbors_t *ngbd_c, const my_p4est_node_neighbors_t *ngbd_n, const my_p4est_node_neighbors_t *fine_ls, const bool activate_x_ = true);
  ~my_p4est_xgfm_cells_t();

#ifdef P4_TO_P8
  void set_phi(Vec phi_on_fine_mesh, Vec phi_xx_on_fine_mesh = NULL, Vec phi_yy_on_fine_mesh = NULL, Vec phi_zz_on_fine_mesh = NULL);
#else
  void set_phi(Vec phi_on_fine_mesh, Vec phi_xx_on_fine_mesh = NULL, Vec phi_yy_on_fine_mesh = NULL);
#endif

  void set_normals(Vec normals[]);
  void set_jumps(Vec jump_u, Vec jump_normal_flux);

#ifdef P4_TO_P8
  inline void set_bc(BoundaryConditions3D& bc)            {this->bc       = &bc; is_matrix_built = false;}
#else
  inline void set_bc(BoundaryConditions2D& bc)            {this->bc       = &bc; is_matrix_built = false;}
#endif
  inline void set_mus(double mu_m_, double mu_p_)         {
    jumps_have_been_set   = jumps_have_been_set && (fabs(mu_m_ - mu_m.get_value()) < EPS*MAX(mu_m_, mu_m.get_value())) && (fabs(mu_p_ - mu_p.get_value()) < EPS*MAX(mu_p_, mu_p.get_value()));
    this->mu_m.set(mu_m_); this->mu_p.set(mu_p_);
    mu_m_is_larger        = (mu_m_ >= mu_p_);
    mu_m_and_mu_p_equal   = fabs(mu_m_ - mu_p_) < EPS*MAX(fabs(mu_m_), fabs(mu_p_));
    is_matrix_built       = false;
    mus_have_been_set     = true;
  }
  inline void set_diagonals(double add_m_, double add_p_) { this->add_diag_m.set(add_m_); this->add_diag_p.set(add_p_); is_matrix_built = false;}
  inline void set_rhs(Vec rhs_)
  {
#ifdef CASL_THROWS
    // compare local size, global size and ghost layers
    PetscInt local_size, global_size;

    ierr = VecGetLocalSize(rhs_, &local_size); CHKERRXX(ierr);
    int my_error = ( ((PetscInt) p4est->local_num_quadrants) != local_size);

    ierr = VecGetSize(rhs_, &global_size); CHKERRXX(ierr);
    my_error = my_error || (global_size != ((PetscInt) p4est->global_num_quadrants));

    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &my_error, 1, MPI_INT, MPI_LOR, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    if(my_error)
      throw std::invalid_argument("my_p4est_xgfm_cells_t::set_rhs(Vec): the vector argument must be preallocated and have the same layout as if constructed with VecCreateMPI(p4est->mpicomm, p4est->local_num_quadrants, p4est->global_num_quadrants, &Vec) on the coarse p4est...");
#endif
    this->rhs = rhs_;
  }
  inline bool get_matrix_has_nullspace()                  { return matrix_has_nullspace; }

  /* Benchmark tests revealed that PCHYPRE is MUCH faster than PCSOR as PCType!
   * The linear systme is supposed to be symmetric positive (semi-) definite, so KSPCG is ok as KSPType
   * Note: a low threshold for tolerance_on_rel_residual is critical to ensure accuracy in cases with large differences in diffusion coefficients!
   * */
  void solve(KSPType ksp_type = KSPCG, PCType pc_type = PCHYPRE, double absolute_accuracy_threshold = 1e-8, double tolerance_on_rel_residual = 1e-12);
//  void solve(KSPType ksp_type = KSPBCGS, PCType pc_type = PCSOR, double absolute_accuracy_threshold = 1e-8, double tolerance_on_rel_residual = 1e-12);

  void get_extended_interface_values(Vec& cell_centered_extension, Vec& fine_node_sampled_extension)
  {
#ifdef CASL_THROWS
    int my_error = (solution_is_set && (solution == NULL) && (extension_cell_values == NULL));
    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &my_error, 1, MPI_INT, MPI_LOR, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    if(my_error)
      throw std::invalid_argument("my_p4est_xgfm_cells_t::get_extended_interface_values(Vec&, Vec&): the extended interface values cannot be calculated if the solution has been returned to the user beforehand.");
#endif
    if(!solution_is_set)
      solve();
    if(extension_cell_values == NULL)
    {
#ifdef CASL_THROWS
      for (short dim = 0; dim < P4EST_DIM; ++dim)
        my_error = my_error || (jump_mu_grad_u[dim] == NULL);
      if(my_error)
        throw std::invalid_argument("my_p4est_xgfm_cells_t::get_extended_interface_values(Vec&, Vec&): the extended interface values cannot be calculated if the jumps in mu*grad_u have been returned to the used beforehand.");
#endif
      P4EST_ASSERT((!activate_x || mu_m_and_mu_p_equal) && (extension_on_fine_nodes == NULL));

      const double *solution_read_p;
      ierr = VecGetArrayRead(solution, &solution_read_p); CHKERRXX(ierr);
      ierr = VecCreateGhostCells(p4est, ghost, &extension_cell_values); CHKERRXX(ierr);
      extend_interface_values(solution_read_p, extension_cell_values, NULL);
      const double *extension_cell_values_read_p;
      ierr = VecGetArrayRead(extension_cell_values, &extension_cell_values_read_p); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(fine_p4est, fine_nodes, &extension_on_fine_nodes); CHKERRXX(ierr);
      interpolate_coarse_cell_field_to_fine_nodes(extension_cell_values_read_p, extension_on_fine_nodes);
      ierr = VecRestoreArrayRead(solution, &solution_read_p); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(extension_cell_values, &extension_cell_values_read_p); CHKERRXX(ierr);
    }
    cell_centered_extension = extension_cell_values;
    extension_cell_values = NULL; // will be handled by the new owner (hopefully :-P)...
    fine_node_sampled_extension = extension_on_fine_nodes;
    extension_on_fine_nodes = NULL; // will be handled by the new owner (hopefully :-P)...
  }

  void get_jump_mu_grad_u(Vec to_return[P4EST_DIM])
  {
    if(activate_x && !solution_is_set)
      solve();
    for (short dim = 0; dim < P4EST_DIM; ++dim)
    {
      to_return[dim]      = jump_mu_grad_u[dim];
      jump_mu_grad_u[dim] = NULL; // will be handled by the new owner (hopefully :-P)...
    }
  };

  int get_number_of_corrections() const {return numbers_of_ksp_iterations.size()-1; }
  std::vector<PetscInt> get_numbers_of_ksp_iterations() const {return numbers_of_ksp_iterations; }
  std::vector<double> get_max_corrections() const {return max_corrections; }
  std::vector<double> get_relative_residuals() const {return relative_residuals; }

  void get_flux_components_and_subtract_them_from_velocities(Vec flux[P4EST_DIM], my_p4est_faces_t *faces, Vec vstar[P4EST_DIM] = NULL, Vec vnp1[P4EST_DIM] = NULL);
  void get_flux_components(Vec flux[P4EST_DIM], my_p4est_faces_t* faces)
  {
    get_flux_components_and_subtract_them_from_velocities(flux, faces);
  }

  Vec get_solution()
  {
    if(!solution_is_set)
      solve();
    Vec to_return = solution;
    solution = NULL; // will be handled by user, hopefully!
    return to_return;
  }

  void set_initial_guess(Vec& initial_guess);

};

#endif // MY_P4EST_XGFM_CELLS_H

