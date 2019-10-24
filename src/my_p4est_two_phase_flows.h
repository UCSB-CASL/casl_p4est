#ifndef MY_P4EST_TWO_PHASE_FLOWS_H
#define MY_P4EST_TWO_PHASE_FLOWS_H

#ifdef P4_TO_P8
#include <src/my_p8est_cell_neighbors.h>
#include <src/my_p8est_faces.h>
#include <src/my_p8est_interpolation_cells.h>
#include <src/my_p8est_interpolation_faces.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_xgfm_cells.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/voronoi3D.h>
#else
#include <src/my_p4est_cell_neighbors.h>
#include <src/my_p4est_faces.h>
#include <src/my_p4est_interpolation_cells.h>
#include <src/my_p4est_interpolation_faces.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_xgfm_cells.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/voronoi2D.h>
#endif

#if __cplusplus >= 201103L
#include <unordered_map> // if c++11 is fully supported, use unordered maps (i.e. hash tables) as they are apparently much faster
#else
#include <map>
#endif

#if __cplusplus >= 201103L
typedef std::unordered_map<p4est_locidx_t, p4est_locidx_t> computational_to_fine_node_t;
#else
typedef std::map<p4est_locidx_t, p4est_locidx_t> computational_to_fine_node_t;
#endif

using std::set;

typedef enum {
  PSEUDO_TIME = 426624,
  EXPLICIT_ITERATIVE
} extrapolation_technique;

typedef enum {
  OMEGA_MINUS = 2789,
  OMEGA_PLUS
} domain_side;

typedef  enum {
  velocity_field = 153759,
  hodge_field
} two_sided_field;

typedef struct
{
  double value;
  double distance;
} neighbor_value;

typedef struct
{
  double derivative;
  double theta;
  bool xgfm;
} sharp_derivative;


class my_p4est_two_phase_flows_t
{
private:
  class splitting_criteria_computational_grid_two_phase_t : public splitting_criteria_tag_t
  {
  private:
    void tag_quadrant(p4est_t *p4est_np1, p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, p4est_nodes_t* nodes_np1,
                      const double *phi_np1_p, const double *vorticities_np1_p);
    const my_p4est_two_phase_flows_t *owner;
  public:
    splitting_criteria_computational_grid_two_phase_t(my_p4est_two_phase_flows_t* parent_solver) :
      splitting_criteria_tag_t ((splitting_criteria_t*)(parent_solver->p4est_n->user_pointer)), owner(parent_solver) {}
    bool refine_and_coarsen(p4est_t* p4est_np1, p4est_nodes_t* nodes_np1, Vec phi_coarse_np1, Vec vorticities);
  };


  class wall_bc_value_hodge_t : public CF_DIM {
  private:
    my_p4est_two_phase_flows_t* _parent;
  public:
    wall_bc_value_hodge_t(my_p4est_two_phase_flows_t* obj) : _parent(obj) {}
    double operator()(DIM(double x, double y, double z)) const;
    double operator()(const double *xyz) const {return this->operator()(DIM(xyz[0], xyz[1], xyz[2]));}
  };

  my_p4est_brick_t *brick;
  p4est_connectivity_t *conn;

  p4est_t *p4est_nm1;
  p4est_ghost_t *ghost_nm1;
  p4est_nodes_t *nodes_nm1;
  my_p4est_hierarchy_t *hierarchy_nm1;
  my_p4est_node_neighbors_t *ngbd_nm1;

  p4est_t *p4est_n, *fine_p4est_n;
  p4est_ghost_t *ghost_n, *fine_ghost_n;
  p4est_nodes_t *nodes_n, *fine_nodes_n;
  my_p4est_hierarchy_t *hierarchy_n, *fine_hierarchy_n;
  my_p4est_node_neighbors_t *ngbd_n, *fine_ngbd_n;
  my_p4est_cell_neighbors_t *ngbd_c;
  my_p4est_faces_t *faces_n;

  double dxyz_min[P4EST_DIM];
  double xyz_min[P4EST_DIM];
  double xyz_max[P4EST_DIM];
  double convert_to_xyz[P4EST_DIM];
  double tree_diag;
  bool periodic[P4EST_DIM];

  double surface_tension;
  double mu_plus, mu_minus;
  double rho_plus, rho_minus;
  double dt_n;
  double dt_nm1;
  double max_L2_norm_u[2]; // 0:: minus, 1::plus
  double uniform_band_minus, uniform_band_plus;
  double threshold_split_cell;
  double cfl;
  bool   dt_updated;

  int sl_order;

  double threshold_dbl_max;
  const double threshold_norm_of_n = 1.0e-6;

  BoundaryConditionsDIM *bc_pressure;
  BoundaryConditionsDIM bc_hodge;
  BoundaryConditionsDIM *bc_v;

  wall_bc_value_hodge_t wall_bc_value_hodge;

  CF_DIM *external_forces[P4EST_DIM];
  my_p4est_interpolation_nodes_t *interp_phi;

  // -------------------------------------------------------------------
  // ----- FIELDS SAMPLED AT NODES OF THE INTERFACE-CAPTURING GRID -----
  // -------------------------------------------------------------------
  // scalar fields
  Vec fine_phi, fine_curvature, fine_jump_hodge, fine_jump_normal_flux_hodge, fine_mass_flux, fine_variable_surface_tension;
  // vector fields, P4EST_DIM-block-structured
  Vec fine_normal, fine_phi_xxyyzz;
  // tensor/matrix fields, (P4EST_DIM*P4EST_DIM)-block-structured
  // fine_jump_mu_grad_v_p[P4EST_DIM*P4EST_DIM*i+P4EST_DIM*dir+der] is the jump in mu \dfrac{\partial u_{dir}}{\partial x_{der}}, evaluated at local node i of fine_p4est_n
  Vec fine_jump_mu_grad_v;
  // -----------------------------------------------------------------------
  // ----- FIELDS SAMPLED AT NODES OF THE COMPUTATIONAL GRID AT TIME N -----
  // -----------------------------------------------------------------------
  // scalar fields
  Vec vorticities;
  // vector fields, P4EST_DIM-block-structured
  Vec vnp1_nodes_omega_minus,  vnp1_nodes_omega_plus;
  Vec vn_nodes_omega_minus,    vn_nodes_omega_plus;
  Vec interface_velocity_np1; // yes, np1, yes! (used right after compute_dt in update_from_n_to_np1, so it looks like n but it's actually np1)
  // tensor/matrix fields, (P4EST_DIM*P4EST_DIM)-block-structured
  // vn_nodes_omega_minus_xxyyzz_p[P4EST_DIM*P4EST_DIM*i+P4EST_DIM*dir+der] is the second derivative of u^{n, -}_{dir} with respect to cartesian direction x_{der}, evaluated at local node i of p4est_n
  Vec vn_nodes_omega_minus_xxyyzz, vn_nodes_omega_plus_xxyyzz, interface_velocity_np1_xxyyzz;
  // ------------------------------------------------------------------------------
  // ----- FIELDS SAMPLED AT CELL CENTERS OF THE COMPUTATIONAL GRID AT TIME N -----
  // ------------------------------------------------------------------------------
  // scalar fields
  Vec hodge;
  // ------------------------------------------------------------------------------
  // ----- FIELDS SAMPLED AT FACE CENTERS OF THE COMPUTATIONAL GRID AT TIME N -----
  // ------------------------------------------------------------------------------
  // vector fields
  Vec dxyz_hodge[P4EST_DIM], vstar[P4EST_DIM], vnp1_minus[P4EST_DIM], vnp1_plus[P4EST_DIM];
  // -------------------------------------------------------------------------
  // ----- FIELDS SAMPLED AT NODES OF THE COMPUTATIONAL GRID AT TIME NM1 -----
  // -------------------------------------------------------------------------
  // vector fields, P4EST_DIM-block-structured
  Vec vnm1_nodes_omega_minus,  vnm1_nodes_omega_plus;
  Vec interface_velocity_n; // yes, n, yes! (used right after compute_dt in update_from_n_to_np1, so it looks like nm1 but it's actually n)
  // tensor/matrix fields, (P4EST_DIM*P4EST_DIM)-block-structured
  // vnm1_nodes_omega_minus_xxyyzz_p[P4EST_DIM*P4EST_DIM*i+P4EST_DIM*dir+der] is the second derivative of u^{n-1, -}_{dir} with respect to cartesian direction x_{der}, evaluated at local node i of p4est_nm1
  Vec vnm1_nodes_omega_minus_xxyyzz, vnm1_nodes_omega_plus_xxyyzz, interface_velocity_n_xxyyzz;

  // semi-lagrangian backtraced points for faces (needed in viscosity step's setup, needs to be done only once)
  // no need to destroy these, not dynamically allocated...
  bool semi_lagrangian_backtrace_is_done;
  std::vector<double> xyz_n[P4EST_DIM];
  std::vector<double> xyz_nm1[P4EST_DIM];
  // coordinates of the points at time n and nm1 backtraced from the face of orientation dir and local index f_idx are
  // xyz_n[dir][P4EST_DIM*f_idx+0], xyz_n[dir][P4EST_DIM*f_idx+1] (, xyz_n[dir][P4EST_DIM*f_idx+2])
  // xyz_nm1[dir][P4EST_DIM*f_idx+0], xyz_nm1[dir][P4EST_DIM*f_idx+1] (, xyz_nm1[dir][P4EST_DIM*f_idx+2])

  computational_to_fine_node_t face_to_fine_node[P4EST_DIM], cell_to_fine_node, node_to_fine_node;
  bool face_to_fine_node_maps_are_set[P4EST_DIM], cell_to_fine_node_map_is_set, node_to_fine_node_map_is_set;


  //  inline bool get_close_coarse_node(const p4est_indep_t* fine_node, p4est_locidx_t& coarse_node_idx, bool subrefined [P4EST_DIM]) const
  //  {
  //    int lmax = ((const splitting_criteria_t*)p4est_n->user_pointer)->max_lvl;
  //    bool is_coarse = true;
  //    subrefined[0] = ((fine_node->x%P4EST_QUADRANT_LEN(lmax)) == P4EST_QUADRANT_LEN(lmax+1)); is_coarse = is_coarse && !subrefined[0];
  //    subrefined[1] = ((fine_node->y%P4EST_QUADRANT_LEN(lmax)) == P4EST_QUADRANT_LEN(lmax+1)); is_coarse = is_coarse && !subrefined[1];
  //#ifdef P4_TO_P8
  //    subrefined[2] = ((fine_node->z%P4EST_QUADRANT_LEN(lmax)) == P4EST_QUADRANT_LEN(lmax+1)); is_coarse = is_coarse && !subrefined[2];
  //#endif

  //    p4est_quadrant_t r;
  //    r.level = P4EST_MAXLEVEL;
  //    r.x = fine_node->x - (subrefined[0]?P4EST_QUADRANT_LEN(lmax+1):0);
  //    r.y = fine_node->y - (subrefined[1]?P4EST_QUADRANT_LEN(lmax+1):0);
  //#ifdef P4_TO_P8
  //    r.z = fine_node->z - (subrefined[2]?P4EST_QUADRANT_LEN(lmax+1):0);
  //#endif
  //    r.p.which_tree = fine_node->p.which_tree;
  //    // theoretically no need to canonicalize here, the quad center will always be INSIDE a tree
  //    // --> check for it in debug!
  //    P4EST_ASSERT((r.x!=P4EST_ROOT_LEN) && (r.y!=P4EST_ROOT_LEN));
  //#ifdef P4_TO_P8
  //    P4EST_ASSERT(r.z!=P4EST_ROOT_LEN);
  //#endif
  //    P4EST_ASSERT (p4est_quadrant_is_node (&r, 1));
  //    bool tmp = index_of_node(&r, nodes_n, coarse_node_idx);
  //    if(!tmp)
  //      throw std::runtime_error("my_p4est_two_phase_flows::get_close_coarse_node() could not find close coarse node.");

  //    return is_coarse;
  //  }

  //#ifdef P4_TO_P8
  //  inline p4est_locidx_t neighbor_coarse_node(const p4est_indep_t* coarse_node, const unsigned char& ii, const unsigned char& jj, const unsigned char& kk) const
  //#else
  //  inline p4est_locidx_t neighbor_coarse_node(const p4est_indep_t* coarse_node, const unsigned char& ii, const unsigned char& jj) const
  //#endif
  //  {
  //    P4EST_ASSERT(((ii == 0) || (ii == 1)) && ((jj == 0) || (jj == 1)));
  //#ifdef P4_TO_P8
  //    P4EST_ASSERT((kk == 0) || (kk == 1));
  //    P4EST_ASSERT((ii!=0) || (jj!=0) || (kk!=0));
  //#else
  //    P4EST_ASSERT((ii!=0) || (jj!=0));
  //#endif
  //    int lmax = ((const splitting_criteria_t*)p4est_n->user_pointer)->max_lvl;

  //    p4est_quadrant_t *tmp;
  //    p4est_quadrant_t r, n;
  //    r.level = P4EST_MAXLEVEL;
  //    r.x = coarse_node->x + ii*P4EST_QUADRANT_LEN(lmax);
  //    r.y = coarse_node->y + jj*P4EST_QUADRANT_LEN(lmax);
  //#ifdef P4_TO_P8
  //    r.z = coarse_node->z + kk*P4EST_QUADRANT_LEN(lmax);
  //#endif
  //    r.p.which_tree = coarse_node->p.which_tree;
  //#ifdef P4_TO_P8
  //    if(r.x==P4EST_ROOT_LEN || r.y==P4EST_ROOT_LEN || r.z==P4EST_ROOT_LEN)
  //#else
  //    if(r.x==P4EST_ROOT_LEN || r.y==P4EST_ROOT_LEN)
  //#endif
  //    {
  //      p4est_node_canonicalize(p4est_n, r.p.which_tree, &r, &n);
  //      tmp = &n;
  //    }
  //    else
  //      tmp = &r;
  //    P4EST_ASSERT (p4est_quadrant_is_node (tmp, 1));
  //    p4est_locidx_t return_idx;
  //    bool check = index_of_node(tmp, nodes_n, return_idx);
  //    if(!check)
  //      throw std::runtime_error("my_p4est_two_phase_flows::neighbor_coarse_node() could not find neighbor coarse node.");
  //    return return_idx;
  //  }

  inline bool is_subresolved(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, p4est_locidx_t& fine_center_idx, const p4est_quadrant_t* coarse_quad = NULL)
  {
    computational_to_fine_node_t::const_iterator got_it = cell_to_fine_node.find(quad_idx);
    if(got_it != cell_to_fine_node.end()) // found in map
    {
      fine_center_idx = got_it->second;
      return true;
    }
    else
    {
      if(cell_to_fine_node_map_is_set)
        return false;
      if(coarse_quad == NULL)
      {
        p4est_tree_t* tree = p4est_tree_array_index(p4est_n->trees, tree_idx);
        coarse_quad = p4est_quadrant_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);
      }
      if(coarse_quad->level < (((splitting_criteria_t*) p4est_n->user_pointer)->max_lvl))
        return false;
      p4est_quadrant_t r;
      r.level = P4EST_MAXLEVEL;
      r.x = coarse_quad->x + P4EST_QUADRANT_LEN(coarse_quad->level+1);
      r.y = coarse_quad->y + P4EST_QUADRANT_LEN(coarse_quad->level+1);
      ONLY3D(r.z = coarse_quad->z + P4EST_QUADRANT_LEN(coarse_quad->level+1));
      P4EST_ASSERT (p4est_quadrant_is_node (&r, 1));
      // theoretically no need to canonicalize here, the quad center will always be INSIDE a tree
      // --> check for it in debug!
      P4EST_ASSERT((r.x!=0) && (r.x!=P4EST_ROOT_LEN) && (r.y!=0) && (r.y!=P4EST_ROOT_LEN));
      ONLY3D(P4EST_ASSERT((r.z!=0) && (r.z!=P4EST_ROOT_LEN)));
      r.p.which_tree = tree_idx;
      bool to_return = index_of_node(&r, fine_nodes_n, fine_center_idx);
      if(to_return)
        cell_to_fine_node[quad_idx] = fine_center_idx;
      return to_return;
    }
  };

  inline bool get_fine_node_idx_of_logical_vertex(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, DIM(const char& vx, const char& vy, const char& vz), p4est_locidx_t& fine_vertex_idx,  const p4est_quadrant_t* coarse_quad)
  {
    unsigned char sum_v = abs(vx) + abs(vy);
    P4EST_ASSERT(vx==-1 || vx==0 || vx==1);
    P4EST_ASSERT(vy==-1 || vy==0 || vy==1);
    ONLY3D(P4EST_ASSERT(vz==-1 || vz==0 || vz==1));
    ONLY3D(sum_v += abs(vz));
    P4EST_ASSERT(sum_v <= P4EST_DIM);
    // looking for saved shortcuts first
    const bool is_center  = (sum_v == 0);
    const bool is_face    = (sum_v == 1);
#ifdef P4_TO_P8
    const bool is_edge    = (sum_v == 2);
#endif
    const bool is_corner  = (sum_v == P4EST_DIM);
    p4est_locidx_t face_idx, node_idx;
    char local_face_dir;
    if(is_center)
    {
      computational_to_fine_node_t::const_iterator got_it = cell_to_fine_node.find(quad_idx);
      if(got_it != cell_to_fine_node.end())
      {
        fine_vertex_idx = got_it->second;
        return true;
      }
      else if (cell_to_fine_node_map_is_set)
        return false;
    }
    if(is_face)
    {
#ifdef P4_TO_P8
      local_face_dir = ((abs(vx) == 1)? (1+vx) : ((abs(vy) == 1)? (5+vy) : (9+vz)))/2;
#else
      local_face_dir = ((abs(vx) == 1)? (1+vx) : (5+vy))/2;
#endif
      P4EST_ASSERT((local_face_dir >=0) && (local_face_dir < P4EST_FACES));
      face_idx = faces_n->q2f(quad_idx, local_face_dir);
      computational_to_fine_node_t::const_iterator got_it = face_to_fine_node[local_face_dir/2].find(face_idx);
      if(got_it != face_to_fine_node[local_face_dir/2].end())
      {
        fine_vertex_idx = got_it->second;
        return true;
      }
      else if (face_to_fine_node_maps_are_set[local_face_dir/2])
        return false;
    }
#ifdef P4_TO_P8
//    // TO BE IMPLEMENTED
//    if(is_edge)
//    {
//      throw std::invalid_argument("you have a bit more work in 3D, here...");
//    }
#endif
    if (is_corner)
    {
      char local_node_idx = SUMD((vx+1)/2, (vy+1), 2*(vz+1));
      P4EST_ASSERT((local_node_idx >= 0) && (local_node_idx < P4EST_CHILDREN));
      node_idx = nodes_n->local_nodes[P4EST_CHILDREN*quad_idx+local_node_idx];
      computational_to_fine_node_t::const_iterator got_it = node_to_fine_node.find(node_idx);
      if(got_it != node_to_fine_node.end())
      {
        fine_vertex_idx = got_it->second;
        return true;
      }
      else if (node_to_fine_node_map_is_set)
        return false;
    }
    // core of the routine here below
    if(coarse_quad == NULL)
    {
      if(quad_idx < p4est_n->local_num_quadrants)
      {
        p4est_tree_t* tree  = p4est_tree_array_index(p4est_n->trees, tree_idx);
        coarse_quad         = p4est_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);
      }
      else
        coarse_quad         = p4est_quadrant_array_index(&ghost_n->ghosts, quad_idx-p4est_n->local_num_quadrants);
    }
    p4est_quadrant_t *tmp_ptr;
    p4est_quadrant_t r;
    r.level = P4EST_MAXLEVEL;
    XCODE(r.x = coarse_quad->x + (vx+1)*P4EST_QUADRANT_LEN(coarse_quad->level+1));
    YCODE(r.y = coarse_quad->y + (vy+1)*P4EST_QUADRANT_LEN(coarse_quad->level+1));
    ZCODE(r.z = coarse_quad->z + (vz+1)*P4EST_QUADRANT_LEN(coarse_quad->level+1));
    r.p.which_tree = tree_idx;
    P4EST_ASSERT (p4est_quadrant_is_node (&r, 0));
    if(ORD((r.x == 0) || (r.x == P4EST_ROOT_LEN), (r.y == 0) || (r.y == P4EST_ROOT_LEN), (r.z == 0) || (r.z == P4EST_ROOT_LEN)))
    {
      p4est_quadrant_t n;
      p4est_node_canonicalize(fine_p4est_n, tree_idx, &r, &n);
      tmp_ptr = &n;
    }
    else
      tmp_ptr = &r;
    bool to_return = index_of_node(tmp_ptr, fine_nodes_n, fine_vertex_idx);
    if(to_return && is_center)
      cell_to_fine_node[quad_idx] = fine_vertex_idx;
    if(to_return && is_face)
      face_to_fine_node[local_face_dir/2][face_idx] = fine_vertex_idx;
#ifdef P4_TO_P8
    // TO BE IMPLEMENTED
//    if(to_return && is_edge)
//    {
//      throw std::invalid_argument("you have a bit more work in 3D, here...");
//    }
#endif
    if(to_return && is_corner)
      node_to_fine_node[node_idx] = fine_vertex_idx;

    return to_return;
  }

  inline bool get_fine_node_idx_node(const p4est_locidx_t& node_idx, p4est_locidx_t& fine_vertex_idx)
  {
    P4EST_ASSERT((node_idx >= 0) && (node_idx < nodes_n->num_owned_indeps));
    computational_to_fine_node_t::const_iterator got_it = node_to_fine_node.find(node_idx);
    if(got_it != node_to_fine_node.end())
    {
      fine_vertex_idx = got_it->second;
      return true;
    }
    else if (node_to_fine_node_map_is_set)
      return false;
    // core of the routine here below
    p4est_quadrant_t r;
    r = *((p4est_quadrant_t*) sc_array_index(&nodes_n->indep_nodes, node_idx));
    P4EST_ASSERT (p4est_quadrant_is_node (&r, 0));
    // theoretically no need to canonicalize here, the point should already be INSIDE
    // a tree
    P4EST_ASSERT(ANDD(r.x!=P4EST_ROOT_LEN, r.y!=P4EST_ROOT_LEN, r.z!=P4EST_ROOT_LEN));
    bool to_return = index_of_node(&r, fine_nodes_n, fine_vertex_idx);
    if(to_return)
      node_to_fine_node[node_idx] = fine_vertex_idx;
    return to_return;
  }



  inline bool get_fine_node_idx_of_face(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const unsigned char& loc_face_dir, p4est_locidx_t& fine_face_idx, const p4est_quadrant_t* coarse_quad = NULL, const p4est_locidx_t* face_idx_ptr = NULL)
  {
    computational_to_fine_node_t::const_iterator got_it = face_to_fine_node[loc_face_dir/2].find((face_idx_ptr!=NULL)? (*face_idx_ptr):faces_n->q2f(quad_idx, loc_face_dir));
    if(got_it != face_to_fine_node[loc_face_dir/2].end()) // found in map
    {
      fine_face_idx = got_it->second;
      return true;
    }
    else
    {
      if(face_to_fine_node_maps_are_set[loc_face_dir/2])
        return false;
      if(coarse_quad == NULL)
      {
        if(quad_idx < p4est_n->local_num_quadrants)
        {
          p4est_tree_t* tree = p4est_tree_array_index(p4est_n->trees, tree_idx);
          coarse_quad = p4est_quadrant_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);
        }
        else
          coarse_quad = p4est_quadrant_array_index(&ghost_n->ghosts, quad_idx-p4est_n->local_num_quadrants);
      }
      if(coarse_quad->level < (((splitting_criteria_t*) p4est_n->user_pointer)->max_lvl))
        return false;
      return get_fine_node_idx_of_logical_vertex(quad_idx, tree_idx, DIM(((loc_face_dir/2==dir::x)? ((loc_face_dir%2)? 1:-1):0), ((loc_face_dir/2==dir::y)? ((loc_face_dir%2)? 1:-1):0), ((loc_face_dir/2==dir::z)? ((loc_face_dir%2)? 1:-1):0)),
                                                 fine_face_idx, coarse_quad);
    }
  }
  inline bool signs_of_phi_are_different(const double& phi_0, const double& phi_1) const
  {
    return (((phi_0 > 0.0) && (phi_1<=0.0)) || ((phi_0 <= 0.0) && (phi_1 >0.0)));
  }
  inline bool face_is_across(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const unsigned char& dir, const double* fine_phi_p, p4est_locidx_t& fine_center_idx, p4est_locidx_t& fine_face_idx, const p4est_quadrant_t* coarse_quad = NULL)
  {
    if(!is_subresolved(quad_idx, tree_idx, fine_center_idx, coarse_quad))
      return false;
    get_fine_node_idx_of_face(quad_idx, tree_idx, dir, fine_face_idx, coarse_quad);
    return signs_of_phi_are_different(fine_phi_p[fine_center_idx], fine_phi_p[fine_face_idx]);
  }
  inline bool face_is_dirichlet_wall(const p4est_locidx_t& face_idx, const unsigned char& dir, const double* xyz_face) const
  {
    p4est_locidx_t quad_idx;
    p4est_topidx_t tree_idx;
    faces_n->f2q(face_idx, dir, quad_idx, tree_idx);
    unsigned char loc_face_idx = ((faces_n->q2f(quad_idx, 2*dir) == face_idx)? 2*dir: 2*dir+1);
    P4EST_ASSERT(faces_n->q2f(quad_idx, loc_face_idx) == face_idx);
    const p4est_quadrant_t* quad;
    if(quad_idx < p4est_n->local_num_quadrants)
    {
      p4est_tree_t* tree = p4est_tree_array_index(p4est_n->trees, tree_idx);
      quad = p4est_quadrant_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);
    }
    else
      quad = p4est_quadrant_array_index(&ghost_n->ghosts, quad_idx-p4est_n->local_num_quadrants);
    return (is_quad_Wall(p4est_n, tree_idx, quad, loc_face_idx) && (bc_v[dir].wallType(xyz_face) == DIRICHLET));
  }
  inline bool is_face_in_omega_minus(const p4est_locidx_t& face_idx, const unsigned char& dir, const double* fine_phi_p, p4est_locidx_t& fine_face_idx)
  {
    computational_to_fine_node_t::const_iterator got_it = face_to_fine_node[dir].find(face_idx);
    if(got_it != face_to_fine_node[dir].end()) // found in map
    {
      fine_face_idx = got_it->second;
      return (fine_phi_p[fine_face_idx] <=0.0);
    }
    else
    {
      p4est_locidx_t quad_idx;
      p4est_topidx_t tree_idx;
      faces_n->f2q(face_idx, dir, quad_idx, tree_idx);
      unsigned char loc_face_dir = ((faces_n->q2f(quad_idx, 2*dir) == face_idx)? 2*dir: 2*dir+1);
      P4EST_ASSERT(faces_n->q2f(quad_idx, loc_face_dir) == face_idx);
      const p4est_quadrant_t* coarse_quad;
      if(quad_idx<p4est_n->local_num_quadrants)
      {
        p4est_tree_t* tree = (p4est_tree_t*) sc_array_index(p4est_n->trees, tree_idx);
        coarse_quad = p4est_quadrant_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);
      }
      else
        coarse_quad = p4est_quadrant_array_index(&ghost_n->ghosts, quad_idx-p4est_n->local_num_quadrants);
      if ((!face_to_fine_node_maps_are_set[dir]) && (coarse_quad->level == ((splitting_criteria_t*) p4est_n->user_pointer)->max_lvl) && get_fine_node_idx_of_face(quad_idx, tree_idx, loc_face_dir, fine_face_idx, coarse_quad, &face_idx))
        return (fine_phi_p[fine_face_idx] <=0.0);
      else
      {
        double xyz_face [P4EST_DIM];
        faces_n->xyz_fr_f(face_idx, dir, xyz_face);
        return ((*interp_phi)(xyz_face) <= 0.0);
      }
    }
  }
  inline bool is_face_in_omega_minus(const p4est_locidx_t& face_idx, const unsigned char& dir, const double* fine_phi_p)
  {
    p4est_locidx_t dummy;
    return is_face_in_omega_minus(face_idx, dir, fine_phi_p, dummy);
  }

  inline double BDF_alpha() const {return ((sl_order ==1) ? 1.0: (2*dt_n+dt_nm1)/(dt_n+dt_nm1)); }
  inline double BDF_beta() const {return ((sl_order ==1) ? 0.0: -dt_n/(dt_n+dt_nm1)); }

  void get_velocity_seen_from_cell(neighbor_value& neighbor_velocity, const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const int& face_dir,
                                   const double *vstar_p, const double *fine_phi_p, const double *fine_phi_xxyyzz_p, const double *fine_jump_mu_grad_v_p);

  double compute_divergence(p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, const double *vstar_p[], const double *fine_phi_p, const double *fine_phi_xxyyzz_p, const double *fine_jump_mu_grad_v_p);

  double div_mu_grad_u_dir(const p4est_locidx_t& face_idx, const unsigned char& dir, const bool& face_is_in_omega_minus, const p4est_locidx_t & fine_idx_of_face,
                           const double *vn_dir_p, const double *fine_jump_mu_grad_vdir_p, const double *fine_phi_p, const double *fine_phi_xxyyzz_p);

  Voronoi_DIM compute_voronoi_cell(const p4est_locidx_t &face_idx, const unsigned char &dir,
                                   const bool &face_is_in_omega_minus, const p4est_locidx_t &fine_idx_of_face,
                                   const double *fine_phi_p, const double *fine_phi_xxyyzz_p,
                                   const double *vn_dir_p, const double *fine_jump_mu_grad_vdir_p,
                                   bool &xgfm_treatment_required, double xgfm_fluxes[P4EST_DIM]);

  void compute_normals_curvature_and_second_derivatives(const bool& set_second_derivatives);
  void compute_curvature();
  void normalize_normals();

  PetscErrorCode inline delete_and_nullify_vector(Vec& vv)
  {
    PetscErrorCode ierr = 0;
    if(vv != NULL){
      ierr = VecDestroy(vv); CHKERRQ(ierr);
      vv = NULL;
    }
    return ierr;
  }

  PetscErrorCode inline create_node_vector_if_needed(Vec& vv, const p4est_t* forest, const p4est_nodes_t* nodes, const unsigned int &block_size=1)
  {
    PetscErrorCode ierr = 0;
    // destroy and nullify the vector if it is not correctly set
    if(vv != NULL && !vectorIsWellSetForNodes(vv, nodes, forest->mpicomm, block_size)){
      ierr = VecDestroy(vv); CHKERRQ(ierr);
      vv = NULL;
    }
    if(vv == NULL){
      ierr = VecCreateGhostNodesBlock(forest, nodes, block_size, &vv); CHKERRQ(ierr);
    }
    return ierr;
  }

  void inline compute_gradient_and_second_derivatives(const p4est_locidx_t& fine_node_idx, const quad_neighbor_nodes_of_node_t *qnnn,
                                                      const double* fine_phi_p, double *fine_grad_phi_p, double *fine_phi_xxyyzz_p)
  {
    qnnn->gradient(fine_phi_p, (fine_grad_phi_p+P4EST_DIM*fine_node_idx));
    if(fine_phi_xxyyzz_p!=NULL)
      qnnn->laplace(fine_phi_p, (fine_phi_xxyyzz_p+P4EST_DIM*fine_node_idx));
  }

  void inline compute_local_curvature(const p4est_locidx_t& fine_node_idx, const quad_neighbor_nodes_of_node_t *qnnn,
                                      const double* fine_phi_p, const double *fine_phi_xxyyzz_p,
                                      const double *fine_grad_phi_p, double *fine_curvature_p)
  {
    // compute first derivatives
    double norm_of_grad = 0.0;
    double dx = fine_grad_phi_p[P4EST_DIM*fine_node_idx+0]; norm_of_grad += SQR(dx);
    double dy = fine_grad_phi_p[P4EST_DIM*fine_node_idx+1]; norm_of_grad += SQR(dy);
#ifdef P4_TO_P8
    double dz = fine_grad_phi_p[P4EST_DIM*fine_node_idx+2]; norm_of_grad += SQR(dz);
#endif
    norm_of_grad = sqrt(norm_of_grad);

    if(norm_of_grad > threshold_norm_of_n)
    {
      // compute second derivatives
      double dxxyyzz[P4EST_DIM];
      if(fine_phi_xxyyzz_p!=NULL){
        for (unsigned char der = 0; der < P4EST_DIM; ++der)
          dxxyyzz[der] = fine_phi_xxyyzz_p[P4EST_DIM*fine_node_idx+der];
      } else
        qnnn->laplace(fine_phi_p, dxxyyzz);


      double dxy = qnnn->dy_central_component(fine_grad_phi_p, P4EST_DIM, dir::x);
#ifdef P4_TO_P8
      double dxz = qnnn->dz_central_component(fine_grad_phi_p, P4EST_DIM, dir::x); // d/dz{d/dx}
      double dyz = qnnn->dz_central_component(fine_grad_phi_p, P4EST_DIM, dir::y); // d/dz{d/dy}
#endif
#ifdef P4_TO_P8
      fine_curvature_p[fine_node_idx] = ((dxxyyzz[1]+dxxyyzz[2])*SQR(dx) + (dxxyyzz[0]+dxxyyzz[2])*SQR(dy) + (dxxyyzz[0]+dxxyyzz[1])*SQR(dz) -
          2*(dx*dy*dxy + dx*dz*dxz + dy*dz*dyz)) / (norm_of_grad*norm_of_grad*norm_of_grad);
#else
      fine_curvature_p[fine_node_idx] = (dxxyyzz[1]*SQR(dx) + dxxyyzz[0]*SQR(dy) - 2*dx*dy*dxy) / (norm_of_grad*norm_of_grad*norm_of_grad);
#endif
    }
    else
      fine_curvature_p[fine_node_idx] = 0.0; // nothing better to suggest for now, sorry
  }

  void inline compute_local_jump_mu_grad_v_elements(const p4est_locidx_t& fine_node_idx, const quad_neighbor_nodes_of_node_t *fine_qnnn,
                                                    const my_p4est_interpolation_nodes_t &interp_grad_underlined_vn_nodes,
                                                    const double* fine_normal_p, const double *fine_mass_flux_p, const double *fine_mass_flux_times_normal_p,
                                                    const double *fine_variable_surface_tension_p, const double *fine_curvature_p,
                                                    double* fine_jump_mu_grad_v_p) const
  {
    const double overlined_mu = overlined_viscosity();
    double grad_underlined_u[P4EST_DIM*P4EST_DIM];            // grad_underlined_u[P4EST_DIM*i+der] = partical derivative of component i of underlined u along direction der
    double grad_mass_flux_times_normal[P4EST_DIM*P4EST_DIM];  // grad_mass_flux_times_normal[P4EST_DIM*i+der] = partical derivative of component i of grad_mass_flux_times_normal along direction der
    double grad_mass_flux[P4EST_DIM];
    double grad_surface_tension[P4EST_DIM];
    double xyz_fine_node[P4EST_DIM]; node_xyz_fr_n(fine_node_idx, fine_p4est_n, fine_nodes_n, xyz_fine_node);
    interp_grad_underlined_vn_nodes(xyz_fine_node, grad_underlined_u);
    if(fine_mass_flux_p!=NULL){
      fine_qnnn->gradient(fine_mass_flux_p, grad_mass_flux);
      P4EST_ASSERT(fine_mass_flux_times_normal_p!=NULL);
      fine_qnnn->gradient_all_components(fine_mass_flux_times_normal_p, grad_mass_flux_times_normal, P4EST_DIM);
    }
    if(fine_variable_surface_tension_p!=NULL)
      fine_qnnn->gradient(fine_variable_surface_tension_p, grad_surface_tension);

    // jump in div(u) is implicitly assumed to be 0.0! (only assumption)
    p4est_locidx_t dim_dim_fine_node_idx  = P4EST_DIM*P4EST_DIM*fine_node_idx;
    p4est_locidx_t dim_fine_node_idx      = P4EST_DIM*fine_node_idx;
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      const unsigned char dim_dir = P4EST_DIM*dir;
      for (unsigned char der = 0; der < P4EST_DIM; ++der) {
        fine_jump_mu_grad_v_p[dim_dim_fine_node_idx+dim_dir+der] = 0.0;
        if(fine_mass_flux_p!=NULL)
          fine_jump_mu_grad_v_p[dim_dim_fine_node_idx+dim_dir+der] -=
              overlined_mu*fine_normal_p[dim_fine_node_idx+dir]*fine_normal_p[dim_fine_node_idx+der]*fine_curvature_p[fine_node_idx]*fine_mass_flux_p[fine_node_idx]*jump_inverse_mass_density();
        for (unsigned char k = 0; k < P4EST_DIM; ++k) {
          if(fine_mass_flux_p!=NULL)
            fine_jump_mu_grad_v_p[dim_dim_fine_node_idx+dim_dir+der] +=
                overlined_mu*(((k==der)?1.0:0.0) - fine_normal_p[dim_fine_node_idx+der]*fine_normal_p[dim_fine_node_idx+k])*grad_mass_flux_times_normal[dim_dir+k]*jump_inverse_mass_density();
          fine_jump_mu_grad_v_p[dim_dim_fine_node_idx+dim_dir+der] +=
              jump_viscosity()*(((k==der)?1.0:0.0) - fine_normal_p[dim_fine_node_idx+der]*fine_normal_p[dim_fine_node_idx+k])*grad_underlined_u[dim_dir+k];
          if(fine_variable_surface_tension_p!=NULL)
            fine_jump_mu_grad_v_p[dim_dim_fine_node_idx+dim_dir+der] +=
                (((k==dir)?1.0:0.0) - fine_normal_p[dim_fine_node_idx+der]*fine_normal_p[dim_fine_node_idx+k])*grad_surface_tension[k]*fine_normal_p[dim_fine_node_idx+der];
          if(fine_mass_flux_p!=NULL)
            fine_jump_mu_grad_v_p[dim_dim_fine_node_idx+dim_dir+der] -=
                overlined_mu*fine_normal_p[dim_fine_node_idx+der]*(((dir==k)?1.0:0.0) - fine_normal_p[dim_fine_node_idx+der]*fine_normal_p[dim_fine_node_idx+k])*grad_mass_flux[k]*jump_inverse_mass_density();
          for (unsigned char r = 0; r < P4EST_DIM; ++r)
          {
            fine_jump_mu_grad_v_p[dim_dim_fine_node_idx+dim_dir+der] -=
                jump_viscosity()*fine_normal_p[dim_fine_node_idx+der]*(((dir==k)?1.0:0.0) - fine_normal_p[dim_fine_node_idx+der]*fine_normal_p[dim_fine_node_idx+k])*grad_underlined_u[P4EST_DIM*r+k]*fine_normal_p[dim_fine_node_idx+r];
            fine_jump_mu_grad_v_p[dim_dim_fine_node_idx+dim_dir+der] +=
                jump_viscosity()*fine_normal_p[dim_fine_node_idx+dir]*fine_normal_p[dim_fine_node_idx+der]*fine_normal_p[dim_fine_node_idx+k]*fine_normal_p[dim_fine_node_idx+r]*grad_underlined_u[P4EST_DIM*k+r];
          }
        }
      }
    }
  }

  inline double jump_mass_density() const { return (rho_plus - rho_minus);}
  inline double jump_inverse_mass_density() const { return (1.0/rho_plus - 1.0/rho_minus);}
  inline double jump_viscosity() const { return (mu_plus-mu_minus);}
  inline domain_side underlined_side(two_sided_field field_) const {
    return ((field_ == velocity_field)? ((mu_plus>=mu_minus)? OMEGA_PLUS : OMEGA_MINUS) : ((rho_plus>=rho_minus)?OMEGA_MINUS:OMEGA_PLUS));
  }
  inline domain_side overlined_side(two_sided_field field_) const {
    return ((underlined_side(field_)==OMEGA_PLUS? OMEGA_MINUS:OMEGA_PLUS));
  }
  inline double overlined_viscosity() const { return ((overlined_side(velocity_field) == OMEGA_PLUS)? mu_plus:mu_minus); }

  void interpolate_velocity_at_node(const p4est_locidx_t &node_idx, double *v_nodes_omega_plus_p, double *v_nodes_omega_minus_p,
                                    const double *vnp1_plus_p[P4EST_DIM], const double *vnp1_minus_p[P4EST_DIM]);

  /*
   * qm and qp must be defined and have their p.piggy3 filled!
   */
  void sharp_derivative_of_face_field(sharp_derivative &one_sided_derivative,
                                      const p4est_locidx_t &face_idx, const bool &face_is_in_omega_minus, const p4est_locidx_t &fine_idx_of_face, const uniform_face_ngbd *face_neighbors,
                                      const double *fine_phi_p, const double *fine_phi_xxyyzz_p,
                                      const unsigned char &der, const unsigned char &dir,
                                      const double *vn_dir_minus_p, const double *vn_dir_plus_p, const double *fine_jump_mu_grad_vdir_p,
                                      const p4est_quadrant_t &qm, const p4est_quadrant_t &qp,
                                      const double *fine_mass_flux_p = NULL, const double *fine_normal_dir_p = NULL);
  /*
   * qm and qp must be defined and have their p.piggy3 filled!
   */
  void get_velocity_seen_from_face(neighbor_value &neighbor_velocity, const p4est_locidx_t &face_idx, const p4est_locidx_t &neighbor_face_idx,
                                   const double *fine_phi_p, const double *fine_phi_xxyyzz_p,
                                   const unsigned char &der, const unsigned char &dir,
                                   const double *vn_dir_minus_p, const double *vn_dir_plus_p, const double *fine_jump_mu_grad_vdir_p,
                                   const double *fine_mass_flux_p = NULL, const double *fine_normal_dir_p = NULL);


  inline void add_faces_to_set_and_clear_vector_of_quad(const p4est_locidx_t& center_face_idx, const unsigned char& dir, set<p4est_locidx_t>& set_of_faces, vector<p4est_quadrant_t>& quad_ngbd) const
  {
    for (size_t k = 0; k < quad_ngbd.size(); ++k) {
      p4est_locidx_t f_tmp = faces_n->q2f(quad_ngbd[k].p.piggy3.local_num, 2*dir);
      if(f_tmp!= NO_VELOCITY && f_tmp != center_face_idx)
        set_of_faces.insert(f_tmp);
      f_tmp = faces_n->q2f(quad_ngbd[k].p.piggy3.local_num, 2*dir+1);
      if(f_tmp!= NO_VELOCITY && f_tmp != center_face_idx)
        set_of_faces.insert(f_tmp);
    }
    quad_ngbd.clear();
  }

  inline void add_all_faces_to_sets_and_clear_vector_of_quad(set<p4est_locidx_t> set_of_faces[P4EST_DIM], vector<p4est_quadrant_t>& quad_ngbd) const
  {
    p4est_locidx_t f_tmp;
    for (size_t k = 0; k < quad_ngbd.size(); ++k) {
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
        f_tmp = faces_n->q2f(quad_ngbd[k].p.piggy3.local_num, 2*dir);
        if(f_tmp!= NO_VELOCITY)
          set_of_faces[dir].insert(f_tmp);
        f_tmp = faces_n->q2f(quad_ngbd[k].p.piggy3.local_num, 2*dir+1);
        if(f_tmp!= NO_VELOCITY)
          set_of_faces[dir].insert(f_tmp);
      }
    }
    quad_ngbd.clear();
  }

  void initialize_normal_derivative_of_velocity_on_faces_local(const p4est_locidx_t &local_face_idx, const unsigned char &dir, const my_p4est_interpolation_nodes_t &interp_normal,
                                                               const double *fine_jump_mu_grad_vdir_p, const double *fine_phi_p, const double *fine_phi_xxyyzz_p,
                                                               double *vnp1_minus_p[P4EST_DIM], double *vnp1_plus_p[P4EST_DIM],
                                                               double *normal_derivative_of_vnp1_minus_p[P4EST_DIM], double *normal_derivative_of_vnp1_plus_p[P4EST_DIM]);

  void extrapolate_normal_derivatives_of_face_velocity_local_explicit_iterative(const p4est_locidx_t &local_face_idx, const unsigned char &dir, const my_p4est_interpolation_nodes_t &interp_normal, const double *fine_phi_p,
                                                                                double *normal_derivative_of_vnp1_minus_p[P4EST_DIM], double *normal_derivative_of_vnp1_plus_p[P4EST_DIM]);

  void solve_velocity_extrapolation_local_explicit_iterative(const p4est_locidx_t &local_face_idx, const unsigned char &dir, const my_p4est_interpolation_nodes_t &interp_normal,
                                                             const double *fine_jump_mu_grad_vdir_p, const double *fine_phi_p, const double *fine_phi_xxyyzz_p,
                                                             const double *normal_derivative_of_vnp1_minus_p[P4EST_DIM], const double *normal_derivative_of_vnp1_plus_p[P4EST_DIM],
                                                             double *vnp1_minus_p[P4EST_DIM], double *vnp1_plus_p[P4EST_DIM]);

  void extrapolate_normal_derivatives_of_face_velocity_local_pseudo_time(const p4est_locidx_t &local_face_idx, const unsigned char &dir, const my_p4est_interpolation_nodes_t &interp_normal, const double *fine_phi_p,
                                                                         double *normal_derivative_of_vnp1_minus_p[P4EST_DIM], double *normal_derivative_of_vnp1_plus_p[P4EST_DIM]);

  void solve_velocity_extrapolation_local_pseudo_time(const p4est_locidx_t &local_face_idx, const unsigned char &dir, const my_p4est_interpolation_nodes_t &interp_normal,
                                                      const double *fine_jump_mu_grad_vdir_p, const double *fine_phi_p, const double *fine_phi_xxyyzz_p,
                                                      const double *normal_derivative_of_vnp1_minus_p[P4EST_DIM], const double *normal_derivative_of_vnp1_plus_p[P4EST_DIM],
                                                      double *vnp1_minus_p[P4EST_DIM], double *vnp1_plus_p[P4EST_DIM]);

  void interpolate_linearly_from_fine_nodes_to_coarse_nodes(const Vec& vv_fine, Vec& vv_coarse);

  void trajectory_from_all_faces_two_phases(p4est_t *p4est_n, my_p4est_faces_t *faces_n, my_p4est_node_neighbors_t *ngbd_nm1, my_p4est_node_neighbors_t *ngbd_n,
                                            const double *fine_phi_p,
                                            Vec vnm1_nodes_omega_minus, Vec vnm1_nodes_omega_minus_xxyyzz,
                                            Vec vnm1_nodes_omega_plus, Vec vnm1_nodes_omega_plus_xxyyzz,
                                            Vec vn_nodes_omega_minus, Vec vn_nodes_omega_minus_xxyyzz,
                                            Vec vn_nodes_omega_plus, Vec vn_nodes_omega_plus_xxyyzz,
                                            double dt_nm1, double dt_n,
                                            std::vector<double> xyz_n[P4EST_DIM],
                                            std::vector<double> xyz_nm1[P4EST_DIM]);

  void get_interface_velocity(Vec interface_velocity);
  void advect_interface(p4est_t *fine_p4est_np1, p4est_nodes_t *fine_nodes_np1, Vec fine_phi_np1,
                        p4est_nodes_t *known_nodes, Vec known_phi_np1 = NULL);
  void compute_vorticities();

  void set_interface_velocity();

  inline bool no_wall_in_face_neighborhood(const uniform_face_ngbd *face_neighbors)
  {
    bool to_return = true;
    for (unsigned char dir = 0; dir < P4EST_FACES; ++dir)
      to_return = to_return && (face_neighbors->neighbor_face_idx[dir] >=0);
    return to_return;
  }

  void extend_vector_field_from_interface_to_whole_domain();

public:
  my_p4est_two_phase_flows_t(my_p4est_node_neighbors_t *ngbd_nm1, my_p4est_node_neighbors_t *ngbd_n, my_p4est_faces_t *faces, my_p4est_node_neighbors_t *fine_ngbd_n);
  ~my_p4est_two_phase_flows_t();

  void compute_dt(const double &min_value_for_u_max=1.0);
  void set_bc(BoundaryConditionsDIM *bc_v, BoundaryConditionsDIM *bc_p);
  void set_external_forces(CF_DIM *external_forces_[P4EST_DIM]);

  void set_dynamic_viscosities(double mu_omega_minus, double mu_omega_plus);
  void set_surface_tension(double surface_tension_);
  void set_densities(double rho_omega_minus, double rho_omega_plus);

  void set_phi(Vec fine_phi_, bool set_second_derivatives = false);
  void set_node_velocities(CF_DIM* vnm1_omega_minus[P4EST_DIM], CF_DIM* vn_omega_minus[P4EST_DIM], CF_DIM* vnm1_omega_plus[P4EST_DIM], CF_DIM* vn_omega_plus[P4EST_DIM]);
  void set_face_velocities_np1(CF_DIM* vnp1_omega_minus[P4EST_DIM], CF_DIM* vnp1_omega_plus[P4EST_DIM]);
  void set_jump_mu_grad_v(CF_DIM* jump_mu_grad_v_op[P4EST_DIM][P4EST_DIM]);
//  void set_node_vorticities(CF_DIM* vorticity_minus, CF_DIM* vorticity_plus);
  void compute_second_derivatives_of_n_velocities();
  void compute_second_derivatives_of_nm1_velocities();
  inline void compute_second_derivatives_of_n_and_nm1_velocities()
  {
    compute_second_derivatives_of_nm1_velocities();
    compute_second_derivatives_of_n_velocities();
  }
  void set_semi_lagrangian_order(int sl_);
  void set_uniform_bands(double uniform_band_minus_, double uniform_band_plus_);
  void set_uniform_band(double uniform_band_) {set_uniform_bands(uniform_band_, uniform_band_);}
  void set_vorticity_split_threshold(double thresh_);
  void set_cfl(double cfl_);
  void set_dt(double dt_nm1_, double dt_n_);
  inline void set_dt(double dt_n_) {dt_n = dt_n_; }

  inline double get_dt() { return dt_n; }
  inline double get_dtnm1() { return dt_nm1; }
  inline p4est_t* get_p4est() { return p4est_n; }
//  inline p4est_nodes_t* get_nodes() { return nodes_n; }

  //  inline p4est_nodes_t* get_fine_nodes() { return fine_nodes_n; }
  //  inline p4est_ghost_t* get_ghost() { return ghost_n; }
  inline Vec get_hodge() { return hodge; }
  //  inline Vec* get_normals() { return fine_normal; }
  inline p4est_t* get_p4est_n() const { return p4est_n; }
  inline Vec get_vnp1_nodes_omega_minus() const { return vnp1_nodes_omega_minus; }
  inline Vec get_vnp1_nodes_omega_plus() const { return vnp1_nodes_omega_plus; }
  inline my_p4est_node_neighbors_t* get_ngbd_n() const { return ngbd_n; }
  inline my_p4est_interpolation_nodes_t* get_interp_phi() const { return interp_phi; }
  inline p4est_nodes_t* get_nodes_n() const { return nodes_n; }
  inline p4est_ghost_t* get_ghost_n() const { return ghost_n; }
  inline double get_diag_min() const { return tree_diag/((double) (1<<(((splitting_criteria_t*)p4est_n->user_pointer)->max_lvl))); }
  //  inline Vec get_curvature() { return fine_curvature; }

  void compute_jump_mu_grad_v();
  void compute_jumps_hodge();
  void solve_viscosity_explicit();

  void solve_projection(const bool activate_xgfm)
  {
    my_p4est_xgfm_cells_t* cell_poisson_jump_solver = NULL;
    solve_projection(cell_poisson_jump_solver, activate_xgfm);
    delete cell_poisson_jump_solver;
  }
  void solve_projection(my_p4est_xgfm_cells_t* &cell_poisson_jump_solver, const bool activate_xgfm, const KSPType ksp = KSPCG, const PCType pc = PCHYPRE);

  void extrapolate_velocities_across_interface_in_finest_computational_cells_Aslam_PDE(const extrapolation_technique& extrapolation_method = PSEUDO_TIME, const unsigned int& n_iteration = 10);
  void compute_velocity_at_nodes();
  void save_vtk(const char* name, const bool& export_fine_grid = false, const char* name_fine = NULL);
  void update_from_tn_to_tnp1(const unsigned int &nnn);

  inline double get_max_velocity() const { return MAX(max_L2_norm_u[0], max_L2_norm_u[1]); }

};

#endif // MY_P4EST_TWO_PHASE_FLOWS_H
