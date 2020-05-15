#ifndef MY_P4EST_FACES_H
#define MY_P4EST_FACES_H

#ifdef P4_TO_P8
#include <p8est.h>
#include <p8est_ghost.h>
#include <src/casl_math.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_cell_neighbors.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/voronoi3D.h>
#else
#include <p4est.h>
#include <p4est_ghost.h>
#include <src/casl_math.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_cell_neighbors.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/voronoi2D.h>
#endif

#if __cplusplus >= 201103L
#include <unordered_map> // if c++11 is fully supported, use unordered maps (i.e. hash tables) as they are apparently much faster
#else
#include <map>
#endif

using std::vector;
#if __cplusplus >= 201103L
using std::unordered_map;
#else
using std::map;
#endif


#define NO_VELOCITY -1

typedef struct {
  p4est_locidx_t face_idx;
  double weight;
} face_interpolator_element;

typedef std::vector<face_interpolator_element> face_interpolator;

typedef struct {
  p4est_locidx_t neighbor_face_idx[P4EST_FACES]; // stored in the following order: m00, p00, 0m0, 0p0, 00m, 00p
} uniform_face_ngbd;

#if __cplusplus >= 201103L
typedef std::unordered_map<p4est_locidx_t, uniform_face_ngbd> map_to_uniform_face_ngbd_t;
#else
typedef std::map<p4est_locidx_t, uniform_face_ngbd> map_to_uniform_face_ngbd_t;
#endif

// Raphael: introducing these "types" for Voronoi cells to make my life easier in two-phase flows...
enum voro_cell_type
{
  dirichlet_wall_face,      // the face is on the wall and associated with Dirichlet BC     --> enforce value
  nonuniform,               // the Voronoi cell was constructed from a non-uniform neighborhood, without any wall neighbor
  parallelepiped_no_wall,   // the Voronoi cell is a parallelepiped, i.e., with a uniform neighborhood, without any wall neighbor
  parallelepiped_with_wall, // the Voronoi cell is a parallelepiped, i.e., with a uniform neighborhood, with a wall neighbor
  not_well_defined,         // the face is irrelevant for computing purposes (based on tag values --> when solving in one domain only (one-phase N-S))
};


class my_p4est_faces_t
{
  friend class my_p4est_poisson_faces_t;
  friend class my_p4est_interpolation_faces_t;
private:

  typedef struct face_quad_ngbd
  {
    /* the indices of the neighbor quadrant.
     * - if the quadrant is local, store its index and the corresponding tree.
     *  Note that the local index is cumulative over the trees, and not the index in the tree.
     * - if the quadrant is ghost, store its local index, i.e. local_num_quadrants + ghost_index.
     *
     * If a face has two well defined neighboring quadrants, the local one is prefered over the ghost one.
     * To find out which direction the quadrant is in, use the q2f_ structure to check which index matches.
     */
    p4est_locidx_t quad_idx;
    p4est_topidx_t tree_idx;
    face_quad_ngbd() { quad_idx = -1; tree_idx = -1; }
  } face_quad_ngbd;

  typedef struct faces_comm_1
  {
    p4est_locidx_t local_num;
    unsigned char dir;
  } faces_comm_1_t;

  typedef struct faces_comm_2
  {
    p4est_locidx_t local_num[P4EST_FACES];
    int rank[P4EST_FACES];
  } faces_comm_2_t;

#ifdef CASL_THROWS
  p4est_t *p4est;
#else
  const p4est_t *p4est;
#endif
  const uint8_t max_p4est_lvl;
  const double smallest_dxyz[P4EST_DIM];
  const double* xyz_min;
  const double* xyz_max;
  const double* tree_dimensions;
  const bool*   periodic;
  p4est_ghost_t *ghost;
  const my_p4est_brick_t *myb;
  my_p4est_cell_neighbors_t *ngbd_c;

  void init_faces(bool initialize_neighborhoods_of_fine_faces);

  map_to_uniform_face_ngbd_t uniform_face_neighbors[P4EST_DIM];

  bool finest_faces_neighborhoods_are_set;
  vector<p4est_locidx_t> local_layer_face_index[P4EST_DIM]; // local_layer_face_index[dir][k] = local index of the kth local face of orientation dir that is a ghost face for (an)other(s) process(es)
  vector<p4est_locidx_t> local_inner_face_index[P4EST_DIM]; // local_inner_face_index[dir][k] = local index of the kth local face of orientation dir that is NOT a ghost face for any other process

  inline p4est_bool_t face_neighborhood_is_valid(const unsigned char& dir, const map_to_uniform_face_ngbd_t::const_iterator& my_iterator) const
  {
    p4est_bool_t to_return = P4EST_TRUE;
    p4est_locidx_t local_face_idx = my_iterator->first;
    uniform_face_ngbd face_neighborhood = my_iterator->second;
    double xyz_face[P4EST_DIM]; xyz_fr_f(local_face_idx, dir, xyz_face);
    for (unsigned char k = 0; to_return && k < P4EST_FACES; ++k) {
      double xyz_other_face[P4EST_DIM];
      if(face_neighborhood.neighbor_face_idx[k] >= 0)
      {
        xyz_fr_f(face_neighborhood.neighbor_face_idx[k], dir, xyz_other_face);
        for (unsigned char dim = 0; to_return && dim < P4EST_DIM; ++dim)
        {
          double dim_distance_between_dof = (xyz_other_face[dim] - xyz_face[dim]);
          if(periodic[dim])
          {
            const double pp = dim_distance_between_dof/(xyz_max[dim] - xyz_min[dim]);
            dim_distance_between_dof -= (floor(pp) + (pp > floor(pp) + 0.5 ? 1.0 : 0.0))*(xyz_max[dim] - xyz_min[dim]);
          }
          to_return = to_return && fabs(dim_distance_between_dof - (k/2 == dim ? (k%2 == 1 ? +1.0:-1.0)*smallest_dxyz[dim] : 0.0)) < 0.01*smallest_dxyz[dim]; // we use 0.01*dxyz_min[dim] as tolerance
        }
      }
      else
      {
        p4est_locidx_t quad_idx = f2q_[dir][local_face_idx].quad_idx;
        p4est_topidx_t tree_idx = f2q_[dir][local_face_idx].tree_idx;
        const p4est_quadrant_t* quad = NULL;
        if(quad_idx < p4est->local_num_quadrants)
        {
          p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
          quad = p4est_quadrant_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);
        }
        else
          quad = p4est_quadrant_array_index(&ghost->ghosts, quad_idx-p4est->local_num_quadrants);
        to_return = to_return && is_quad_Wall(p4est, tree_idx, quad, k);
      }
    }
    return to_return;
  }

  void find_fine_face_neighbors_and_store_it(const p4est_topidx_t& tree_idx, const p4est_locidx_t& quad_idx, p4est_tree_t*tree, const unsigned char& face_dir, const p4est_locidx_t& local_face_idx);

  /*!
   * \brief found_finest_face_neighbor: looks for the (finest) face neighbor of a given face in a cartesian direction if the given face is 'finest' itself.
   * (This function will search for the desired face neighbor even if finest_faces_neighborhoods_are_set is false)
   * \param quad              [in]  pointer to the quadrant owning the face of interest
   * \param quad_idx          [in]  local index of the quadrant owning the face of interest
   * \param tree_idx          [in]  index of the tree in which the quadrant owning the face lies
   * \param local_face_idx    [in]  local face index of the face whose neighbor is looked for
   * \param dir               [in]  cartesian direction of the face normal (dir::x, dir::y or dir::z)
   * \param oriented_dir      [in]  oriented direction in which the neighbor is search (dir::f_m00, dir::f_p00, dir::f_0m0, etc.)
   * \param neighbor_face_idx [out] local index of the neighbor face on output if neighbor is found
   * \return true if the neighbor is found, false otherwise.
   */
  inline bool found_finest_face_neighbor(const p4est_quadrant_t* quad, const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx,
                                         const p4est_locidx_t& local_face_idx, const unsigned char& dir,
                                         const unsigned char& oriented_dir, p4est_locidx_t& neighbor_face_idx) const
  {
    if (quad->level < max_p4est_lvl)
      return false;
    const unsigned char face_dir      = (q2f_[2*dir][quad_idx] == local_face_idx ? 2*dir : 2*dir + 1);
    const unsigned char dual_face_dir = (face_dir%2 == 1 ? face_dir - 1 : face_dir + 1);
    P4EST_ASSERT(q2f_[face_dir][quad_idx] == local_face_idx);
    if(oriented_dir/2 == dir)
    {
      if(oriented_dir != face_dir)
      {
        P4EST_ASSERT(oriented_dir == dual_face_dir);
        neighbor_face_idx = q2f_[dual_face_dir][quad_idx];
        return true;
      }
      if(is_quad_Wall(p4est, tree_idx, quad, face_dir))
      {
        neighbor_face_idx = WALL_idx(face_dir);
        return true;
      }
      set_of_neighboring_quadrants cell_neighbor; cell_neighbor.clear();
      ngbd_c->find_neighbor_cells_of_cell(cell_neighbor, quad_idx, tree_idx, face_dir);
      P4EST_ASSERT(cell_neighbor.size() <= 1);
      if(cell_neighbor.size() > 0 && cell_neighbor.begin()->level == max_p4est_lvl)
      {
        P4EST_ASSERT(q2f_[face_dir][cell_neighbor.begin()->p.piggy3.local_num] != NO_VELOCITY && q2f_[dual_face_dir][cell_neighbor.begin()->p.piggy3.local_num] == local_face_idx);
        neighbor_face_idx = q2f_[face_dir][cell_neighbor.begin()->p.piggy3.local_num];
        return true;
      }
      return false;
    }
    if(is_quad_Wall(p4est, tree_idx, quad, oriented_dir))
    {
      neighbor_face_idx = WALL_idx(oriented_dir);
      return true;
    }
    set_of_neighboring_quadrants cell_neighbor; cell_neighbor.clear();
    ngbd_c->find_neighbor_cells_of_cell(cell_neighbor, quad_idx, tree_idx, oriented_dir);
    P4EST_ASSERT(cell_neighbor.size() <= 1);
    if(cell_neighbor.size() > 0 && cell_neighbor.begin()->level == max_p4est_lvl)
    {
      P4EST_ASSERT(q2f_[face_dir][cell_neighbor.begin()->p.piggy3.local_num] != NO_VELOCITY);
      neighbor_face_idx = q2f_[face_dir][cell_neighbor.begin()->p.piggy3.local_num];
      return true;
    }
    P4EST_ASSERT(cell_neighbor.size() == 0 || cell_neighbor.begin()->level < max_p4est_lvl);
    cell_neighbor.clear();
    char search_dir[P4EST_DIM];
    for (unsigned char k = 0; k < P4EST_DIM; ++k) {
      if(dir == k)
        search_dir[k] = 2*(face_dir%2) - 1;
      else if (oriented_dir/2 == k)
        search_dir[k] = 2*(oriented_dir%2) - 1;
      else
        search_dir[k] = 0;
    }
    ngbd_c->find_neighbor_cells_of_cell(cell_neighbor, quad_idx, tree_idx, DIM(search_dir[0], search_dir[1], search_dir[2]));
    P4EST_ASSERT(cell_neighbor.size()<=1);
    if(cell_neighbor.size() > 0 && cell_neighbor.begin()->level == max_p4est_lvl)
    {
      P4EST_ASSERT(q2f_[dual_face_dir][cell_neighbor.begin()->p.piggy3.local_num] != NO_VELOCITY);
      neighbor_face_idx = q2f_[dual_face_dir][cell_neighbor.begin()->p.piggy3.local_num];
      return true;
    }
    return false;
  }

public:

  inline bool finest_faces_neighborhoods_have_been_set() const { return finest_faces_neighborhoods_are_set ; }

  /* the remote local number of the ghost velocities
   * For ghost faces, face_idx>=num_local[dim], and
   * ghost_local_num[dim][face_idx-num_local[dim]] = local index of the ghost face in the proc that owns it
   */
  vector<p4est_locidx_t> ghost_local_num[P4EST_DIM];

  /* q2f[P4EST_FACES][quad_idx]
   * q2f_[dir][quad_idx] = local index of the face in direction dir of quadrant quad_idx
   */
  vector<p4est_locidx_t> q2f_[P4EST_FACES];

  /* f2q_[P4EST_DIM][u_idx]
   * f2q_[dim][face_idx].quad_idx = local index of the quadrant that owns the face (cumulative over the trees)
   * f2q_[dim][face_idx].tree_idx = tree index of the quadrant that owns the face
   * e.g. f2q_[1][12] is the quadrant whose face in the y direction has index 12
   */
  vector<my_p4est_faces_t::face_quad_ngbd> f2q_[P4EST_DIM];

  /* Store which process the ghost faces belong to.
   * For ghost faces, face_idx>=num_local[dim], and
   * nonlocal_ranks[dim][face_idx-num_local[dim]] = rank of the process that owns it
   */
  vector<int> nonlocal_ranks[P4EST_DIM];

  /* Store the number of owned faces for each rank.
   * global_owned_indeps[dim][r] = number of faces of orientation dim owned by process of rank r
   */
  vector<p4est_locidx_t> global_owned_indeps[P4EST_DIM];

  /* Store the offset for global indices, for each rank.
   * proc_offset[dim][r] = offset of global index for faces of orientation dim, on process of rank r
   * (proc_offset[dim][mpisize] = global number of faces of orientation dim)
   */
  vector<p4est_gloidx_t> proc_offset[P4EST_DIM];

  /* num_local[dim] contains the number of local faces of orientation dim */
  p4est_locidx_t num_local[P4EST_DIM];

  /* num_ghost[dim] contains the number of ghost faces of orientation dim  */
  p4est_locidx_t num_ghost[P4EST_DIM];

  /* IMPORTANT NOTE: this constructor assumes that p4est->user_pointer already points to a (splitting_criteria_t) type of object with valid max_lvl, when being called.
   * --> important for restart!*/
  my_p4est_faces_t(p4est_t *p4est_, p4est_ghost_t *ghost_, const my_p4est_brick_t *myb_, my_p4est_cell_neighbors_t *ngbd_c_, bool initialize_neighborhoods_of_fine_faces = false);
  my_p4est_faces_t(p4est_t *p4est_, p4est_ghost_t *ghost_, my_p4est_cell_neighbors_t *ngbd_c_, bool initialize_neighborhoods_of_fine_faces = false)
    : my_p4est_faces_t(p4est_, ghost_, ngbd_c_->get_brick(), ngbd_c_, initialize_neighborhoods_of_fine_faces) {}

  /*!
   * \brief q2f return the face of quadrant quad_idx in the direction dir
   * \param quad_idx the quadrant index in the local p4est (cumulative over the trees)
   * \param dir the direction of the face, dir::f_m00, dir::f_p00, dir::f_0m0 ...
   * \return the local index of the face of quadrant quad_idx in direction dir, return NO_VELOCITY if there is many small neighbor quadrant in the direction dir
   */
  inline p4est_locidx_t q2f(p4est_locidx_t quad_idx, const unsigned char &dir) const
  {
    return q2f_[dir][quad_idx];
  }

  /*!
   * \brief get the local index of the neighbor cell
   * \param f_idx the local index of the face
   * \param dim the cartesian direction of the face (dir::x, dir::y or dir::z)
   * \return the local index of the neighbor cell and the index of three of that neighbor cell
   */
  inline void f2q(p4est_locidx_t f_idx, const unsigned char &dim, p4est_locidx_t& quad_idx, p4est_topidx_t& tree_idx) const
  {
    quad_idx = f2q_[dim][f_idx].quad_idx;
    tree_idx = f2q_[dim][f_idx].tree_idx;
  }

  /*!
   * \brief get the coordinates of the center of a face
   * \param f_idx the index of the face
   * \param dir the cartesian direction of the face (dir::x, dir::y or dir::z)
   * \return the coordinates of the center of face f_idx
   */
  double x_fr_f(p4est_locidx_t f_idx, const unsigned char &dir) const;
  double y_fr_f(p4est_locidx_t f_idx, const unsigned char &dir) const;
#ifdef P4_TO_P8
  double z_fr_f(p4est_locidx_t f_idx, const unsigned char &dir) const;
#endif

  void xyz_fr_f(p4est_locidx_t f_idx, const unsigned char &dir, double* xyz) const;

  /*!
   * \brief rel_qxyz_face_fr_node calculates the relative cartesian coordinates between a face and a given grid node (very useful for lsqr interpolation).
   * The method also returns the cartesian differences in terms of logical coordinate units (in order to efficiently and unambiguously count the number
   * of independent points along Cartesian directions).
   * \param f_idx               [in]  local index of the face of interest
   * \param dir                 [in]  cartesian direction of the face normal (dir::x, dir::y or dir::z)
   * \param xyz_rel             [out] pointer to an array of P4EST_DIM doubles: difference of Cartesian coordinates between the face and the point in physical units
   * \param xyz_node            [in]  pointer to an array of P4EST_DIM doubles: cartesian cooordinates of the grid node
   * \param node                [in]  pointer to the grid node of interest
   * \param logical_qcoord_diff [out] pointer to an array of P4EST_DIM int64_t: difference of Cartesian coordinates between the face and the point in logical units
   * NOTE: logical_qcoord_diff must point to int64_t type to make sure that logical differences and calculations across trees are correct.
   */
  void rel_qxyz_face_fr_node(const p4est_locidx_t& f_idx, const unsigned char& dir, double* xyz_rel, const double* xyz_node, const p4est_indep_t* node, int64_t* logical_qcoord_diff) const;

  /*!
   * \brief calculates the area of the face in negative domain
   * \param f_idx: local index of the face
   * \param dir: the cartesian direction of the face (dir::x, dir::y or dir::z)
   * \param phi_p  [optional]: pointer to read-only node-sampled levelset-values (assumed to be all negative if ignored)
   * \param nodes  [optional]: p4est_node structure
   * \param phi_dd [optional, in 2D only, disregarded in 3D]: array of pointers to the second derivatives of the levelset
   * function to more accurate interface localization
   * \return the area of the face in negative domain (just the area of the face if the optional inputs are disregarded)
   */
#ifdef P4_TO_P8
  double face_area_in_negative_domain(p4est_locidx_t f_idx, const unsigned char &dir, const double *phi_p=NULL, const p4est_nodes_t* nodes = NULL) const;
#else
  double face_area_in_negative_domain(p4est_locidx_t f_idx, const unsigned char &dir, const double *phi_p=NULL, const p4est_nodes_t* nodes = NULL, const double *phi_dd[] = NULL) const;
#endif
  double face_area(p4est_locidx_t f_idx, const unsigned char &dir) const {return face_area_in_negative_domain(f_idx, dir);}

  inline size_t get_layer_size(const unsigned char& dir) const { return local_layer_face_index[dir].size(); }
  inline size_t get_local_size(const unsigned char& dir) const { return local_inner_face_index[dir].size(); }
  inline p4est_locidx_t get_layer_face(const unsigned char& dir, const size_t& i) const {
#ifdef CASL_THROWS
    return local_layer_face_index[dir].at(i);
#endif
    return local_layer_face_index[dir][i];
  }
  inline p4est_locidx_t get_local_face(const unsigned char &dir, const size_t& i) const {
#ifdef CASL_THROWS
    return local_inner_face_index[dir].at(i);
#endif
    return local_inner_face_index[dir][i];
  }

  void set_finest_face_neighborhoods();

  p4est_bool_t finest_face_neighborhoods_are_valid() const
  {
    p4est_bool_t to_return = P4EST_TRUE;
    for (unsigned char dir = 0; to_return && dir < P4EST_DIM; ++dir)
      for (map_to_uniform_face_ngbd_t::const_iterator my_iterator = uniform_face_neighbors[dir].begin(); to_return && my_iterator != uniform_face_neighbors[dir].end(); ++my_iterator)
        to_return = to_return && face_neighborhood_is_valid(dir, my_iterator);
    return  to_return;
  }

  /*!
   * \brief found_uniform_face_neighborhood: looks for the face neighborhood of a given face (finest faces only).
   * (This function will NOT search and build the desired face neighborhood if finest_faces_neighborhoods_are_set is false
   * --> relevant only if finest_faces_neighborhoods_are_set == true)
   * \param local_face_idx  [in]  local index of the face whose neighborhood is sought
   * \param dir             [in]  cartesian direction of the face normal (dir::x, dir::y or dir::z)
   * \param face_ngbd       [out] pointer to the face neighborhood on output if found, NULL on output otherwise
   * \return  true if the neighborhood was found, false otherwise
   */
  inline bool found_uniform_face_neighborhood(const p4est_locidx_t& local_face_idx, const unsigned char& dir, const uniform_face_ngbd* &face_ngbd) const
  {
    bool to_return = finest_faces_neighborhoods_are_set;
    map_to_uniform_face_ngbd_t::const_iterator dummy = uniform_face_neighbors[dir].find(local_face_idx);
    to_return = to_return && dummy != uniform_face_neighbors[dir].end();
    face_ngbd = (to_return ? &(dummy->second) : NULL);
    return  to_return;
  }

  /*!
   * \brief found_finest_face_neighbor: looks for the (finest) face neighbor of a given face in a cartesian direction if the given face is 'finest' itself.
   * (This function will search for the desired face neighbor even if finest_faces_neighborhoods_are_set is false)
   * \param local_face_idx    [in]  local index of the face whose neighbor is looked for
   * \param dir               [in]  cartesian direction of the face normal (dir::x, dir::y or dir::z)
   * \param oriented_dir      [in]  oriented direction in which the neighbor is search (dir::f_m00, dir::f_p00, dir::f_0m0, etc.)
   * \param neighbor_face_idx [out] local index of the neighbor face on output if neighbor is found
   * \return true if the neighbor is found, false otherwise.
   */
  inline bool found_finest_face_neighbor(const p4est_locidx_t& local_face_idx, const unsigned char& dir, const unsigned char& oriented_dir, p4est_locidx_t& neighbor_face_idx) const
  {
    const uniform_face_ngbd* face_neighborhood;
    if(finest_faces_neighborhoods_are_set && found_uniform_face_neighborhood(local_face_idx, dir, face_neighborhood))
    {
      neighbor_face_idx = face_neighborhood->neighbor_face_idx[oriented_dir];
      return true;
    }
    p4est_locidx_t quad_idx;
    p4est_topidx_t tree_idx;
    const p4est_quadrant_t* quad;
    f2q(local_face_idx, dir, quad_idx, tree_idx);
    if(quad_idx < p4est->local_num_quadrants)
    {
      p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
      quad = p4est_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);
    }
    else
      quad = p4est_quadrant_array_index(&ghost->ghosts, quad_idx-p4est->local_num_quadrants);
    return found_finest_face_neighbor(quad, quad_idx, tree_idx, local_face_idx, dir, oriented_dir, neighbor_face_idx);
  }

  /*!
   * \brief global_index calculates the global index of a face
   * \param f_idx [in] local index of the face
   * \param dir   [in]  orientation of the face
   * \return the global index of the face
   */
  inline p4est_gloidx_t global_index(p4est_locidx_t f_idx, const unsigned char &dir) const
  {
    if(f_idx < num_local[dir])
      return f_idx + proc_offset[dir][p4est->mpirank];
    f_idx -= num_local[dir];
    return ghost_local_num[dir][f_idx] + proc_offset[dir][nonlocal_ranks[dir][f_idx]];
  }

  /*!
   * \brief find_quads_touching_face find the quadrant(s) (two at most) on either side of a face. The queried face MUST be connected to at least
   * one quadrant in the domain (i.e. should work for any locally own face).
   * \param face_idx  [in]    the idx of the local face that is queried;
   * \param dir       [in]    cartesian direction of the normal of the queried face (dir::x, dir::y or dir::z)
   * \param qm        [inout] the quadrant touching the face in (oriented) direction 2*dir
   * \param qp        [inout] the quadrant touching the face in (oriented) direction 2*dir+1
   * NOTE: the p.piggy3 member of qm and qp are filled with the corresponding local quad index and their tree idx (if found, otherwise, both set to -1 and their level too)
   */
  inline void find_quads_touching_face(const p4est_locidx_t& face_idx, const unsigned char& dir, p4est_quadrant_t& qm, p4est_quadrant_t& qp) const
  {
    set_of_neighboring_quadrants ngbd;
    p4est_locidx_t quad_idx;
    p4est_topidx_t tree_idx;
    f2q(face_idx, dir, quad_idx, tree_idx);
#ifdef P4EST_DEBUG
    if(quad_idx > p4est->local_num_quadrants)
      throw std::invalid_argument("my_p4est_faces::find_quads_touching_face() called for a face that does not touch local quadrant");
#endif
    p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
    const p4est_quadrant_t * quad = p4est_quadrant_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);

    if(q2f(quad_idx, 2*dir) == face_idx)
    {
      qp = *quad; qp.p.piggy3.local_num = quad_idx; qp.p.piggy3.which_tree = tree_idx;
      ngbd.clear();
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 2*dir);
      /* note that the potential neighbor has to be the same size or bigger */
      if(ngbd.size() > 0)
      {
        P4EST_ASSERT(ngbd.size() == 1);
        qm = *ngbd.begin();
      }
      else
      {
        qm.level                = -1;
        qm.p.piggy3.local_num   = -1;
        qm.p.piggy3.which_tree  = -1;
      }
    }
    else
    {
      qm = *quad; qm.p.piggy3.local_num = quad_idx; qm.p.piggy3.which_tree = tree_idx;
      ngbd.clear();
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 2*dir + 1);
      /* note that the potential neighbor has to be the same size or bigger */
      if(ngbd.size() > 0)
      {
        P4EST_ASSERT(ngbd.size() == 1);
        qp = *ngbd.begin();
      }
      else
      {
        qp.level                = -1;
        qp.p.piggy3.local_num   = -1;
        qp.p.piggy3.which_tree  = -1;
      }
    }
    P4EST_ASSERT(qm.p.piggy3.local_num != -1 || qp.p.piggy3.local_num != -1);
  }

#ifdef CASL_THROWS
  inline p4est_t* get_p4est() const { return  p4est; }
#else
  inline const p4est_t* get_p4est() const { return  p4est; }
#endif
  inline my_p4est_cell_neighbors_t* get_ngbd_c() const { return ngbd_c;}
  inline const double* get_smallest_dxyz() const { return  smallest_dxyz; }
  inline const double* get_xyz_max() const { return  xyz_max; }
  inline const double* get_xyz_min() const { return  xyz_min; }
  inline const double* get_tree_dimensions() const { return  tree_dimensions; }
  inline const bool* get_periodicity() const { return periodic; }
  inline bool periodicity(const unsigned char &dim) const { P4EST_ASSERT(ORD(dim == dir::x, dim == dir::y, dim == dir::z)); return periodic[dim]; }

  size_t memory_estimate() const
  {
    size_t memory = 0;
    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
    {
      memory += ghost_local_num[dim].size()*sizeof (p4est_locidx_t);
      memory += q2f_[dim].size()*sizeof (p4est_locidx_t);
      memory += f2q_[dim].size()*sizeof (face_quad_ngbd);
      memory += nonlocal_ranks[dim].size()*sizeof (int);
      memory += global_owned_indeps[dim].size()*sizeof (p4est_locidx_t);
      memory += proc_offset[dim].size()*sizeof (p4est_gloidx_t);
      memory += local_inner_face_index[dim].size()*sizeof (p4est_locidx_t);
      memory += local_layer_face_index[dim].size()*sizeof (p4est_locidx_t);
      memory += uniform_face_neighbors[dim].size()*(sizeof (p4est_locidx_t) + sizeof (uniform_face_ngbd)) + sizeof (uniform_face_neighbors[dim]);
    }
    memory += sizeof (bool); // finest_faces_neighborhoods_are_set
    memory += sizeof (uint8_t); // max_p4est_lvl
    memory += P4EST_DIM*sizeof (double);  // smallest_dxyz_min
    memory += P4EST_DIM*sizeof (double);  // tree_dimensions
    memory += P4EST_DIM*sizeof (double);  // xyz_min
    memory += P4EST_DIM*sizeof (double);  // xyz_max
    memory += P4EST_DIM*sizeof (bool);    // periodic
    memory += 2*P4EST_DIM*sizeof (p4est_locidx_t); // num_local and num_ghost;
    return memory;
  }
};

PetscErrorCode VecCreateGhostFacesBlock     (const p4est_t *p4est, const my_p4est_faces_t *faces, PetscInt block_size, Vec* v, const unsigned char &dir);
inline PetscErrorCode VecCreateGhostFaces   (const p4est_t *p4est, const my_p4est_faces_t *faces, Vec* v, const unsigned char &dir)
{
  return VecCreateGhostFacesBlock(p4est, faces, 1, v, dir);
}
PetscErrorCode VecCreateNoGhostFacesBlock   (const p4est_t *p4est, const my_p4est_faces_t *faces, PetscInt block_size, Vec* v, const unsigned char &dir);
inline PetscErrorCode VecCreateNoGhostFaces (const p4est_t *p4est, const my_p4est_faces_t *faces, Vec* v, const unsigned char &dir)
{
  return VecCreateNoGhostFacesBlock(p4est, faces, 1, v, dir);
}

inline bool VecsAreSetForFaces(const Vec v[P4EST_DIM], const my_p4est_faces_t* faces, const unsigned int& blocksize, const bool& ghosted = true)
{
  P4EST_ASSERT(v != NULL && ANDD(v[0] != NULL, v[1] != NULL, v[2] != NULL));
  P4EST_ASSERT(blocksize > 0);
  PetscInt local_size, ghosted_size;
  int my_test = 1;
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
    VecGetLocalAndGhostSizes(v[dim], local_size, ghosted_size, ghosted);
    my_test = my_test && (local_size == (PetscInt) (blocksize*faces->num_local[dim]) && (!ghosted || ghosted_size == (PetscInt) (blocksize*(faces->num_local[dim] + faces->num_ghost[dim]))) ? 1 : 0);
  }
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &my_test, 1, MPI_INT, MPI_LAND, faces->get_p4est()->mpicomm); SC_CHECK_MPI(mpiret);
  return my_test;
}


inline bool local_face_is_well_defined(const p4est_locidx_t &f_idx, const my_p4est_faces_t *faces, const my_p4est_interpolation_nodes_t &interp_phi,
                                       const unsigned char &dir, const BoundaryConditionsDIM &bc_dir)
{
  double xyz_face[P4EST_DIM];
  faces->xyz_fr_f(f_idx, dir, xyz_face);
  const double *dxyz = faces->get_smallest_dxyz();
  const double phi_f = interp_phi(xyz_face);
  bool well_defined = phi_f <= 0.0; // any face in negative domain is well-defined independently of
  if(!well_defined && (bc_dir.interfaceType() == NEUMANN || bc_dir.interfaceType() == MIXED))
  {
    // the face may be well defined if the control volume associated with the current cell
    // is partly in negative domain. In such a case, the cell is assumed to be (and should
    // be)the smallest possible and we check the levelset values at the corners
    bool at_least_one_corner_in_negative_domain = false;
    bool at_least_one_corner_is_neumann         = false;
    for (char xxx = -1; xxx < 2 && !well_defined; xxx += 2)
      for (char yyy = -1; yyy < 2 && !well_defined; yyy += 2)
#ifdef P4_TO_P8
        for (char zzz = -1; zzz < 2 && !well_defined; zzz += 2)
#endif
        {
          double xyz_eval[P4EST_DIM] = {DIM(xyz_face[0] + xxx*0.5*dxyz[0], xyz_face[1] + yyy*0.5*dxyz[1], xyz_face[2] + zzz*0.5*dxyz[2])};
          at_least_one_corner_in_negative_domain = at_least_one_corner_in_negative_domain || interp_phi(xyz_eval) <= 0.0;
          at_least_one_corner_is_neumann = at_least_one_corner_is_neumann || (bc_dir.interfaceType(xyz_eval) == NEUMANN);
          well_defined = at_least_one_corner_in_negative_domain && at_least_one_corner_is_neumann;
        }
  }
  return well_defined;
}
/*!
 * \brief mark the faces that are well defined, i.e. that are solved for in an implicit poisson solve with irregular interface.
 * Any face where phi(face) <= 0 is marked well-defined. For Neumann boundary conditions, the control volume of the face must
 * be at least partially in the negative domain. (In case of MIXED boundary conditions, it can be marked well-defined if the at
 * least one corner value of the levelset is negative and at least one cornet boundary condition type is NEUMANN, not necessarily
 * the same cornet).
 * \param [in] faces                  : the faces structure
 * \param [in] dir                    : the cartesian direction treated, dir::x, dir::y or dir::z
 * \param [in] interp_phi             : a node-interpolator for the node-sampled level-set function
 * \param [out] face_is_well_defined  : a face-sampling PetSc Vector for face of orientation dir, to be filled. The values are
 *                                      either 1.0 (if the face is well-defined) or 0.0 (if not).
 */
void check_if_faces_are_well_defined(const my_p4est_faces_t *faces, const unsigned char &dir, const my_p4est_interpolation_nodes_t &interp_phi,
                                     const BoundaryConditionsDIM& bc, Vec face_is_well_defined);
inline void check_if_faces_are_well_defined(my_p4est_node_neighbors_t *ngbd_n, my_p4est_faces_t *faces, const unsigned char &dir,
                                            Vec phi, BoundaryConditionType interface_type, Vec face_is_well_defined)
{
  my_p4est_interpolation_nodes_t interp_phi(ngbd_n);
  interp_phi.set_input(phi, linear);
  BoundaryConditionsDIM bc_tmp; bc_tmp.setInterfaceType(interface_type);
  check_if_faces_are_well_defined(faces, dir, interp_phi, bc_tmp, face_is_well_defined);
  return;
}

// NOTE: reusing the interpolator_from_faces is ok afterwards for Neumann-BC nodes ONLY if the Neumann bc is HOMOGENEOUS!
double interpolate_velocity_at_node_n(my_p4est_faces_t *faces, my_p4est_node_neighbors_t *ngbd_n, p4est_locidx_t node_idx, Vec velocity_component, const unsigned char &dir,
                                      Vec face_is_well_defined = NULL, int order = 2, BoundaryConditionsDIM *bc = NULL, face_interpolator* interpolator_from_faces = NULL);

inline void add_faces_to_set_and_clear_set_of_quad(const my_p4est_faces_t* faces, const p4est_locidx_t& center_face_idx, const unsigned char& dir, std::set<indexed_and_located_face>& set_of_faces, set_of_neighboring_quadrants& quad_ngbd)
{
  indexed_and_located_face tmp_neighbor_seed;
  for (set_of_neighboring_quadrants::const_iterator it = quad_ngbd.begin(); it != quad_ngbd.end(); ++it) {
    tmp_neighbor_seed.face_idx = faces->q2f(it->p.piggy3.local_num, 2*dir);
    if(tmp_neighbor_seed.face_idx != NO_VELOCITY && tmp_neighbor_seed.face_idx != center_face_idx && set_of_faces.find(tmp_neighbor_seed) == set_of_faces.end())
    {
      faces->xyz_fr_f(tmp_neighbor_seed.face_idx, dir, tmp_neighbor_seed.xyz_face);
      set_of_faces.insert(tmp_neighbor_seed);
    }
    tmp_neighbor_seed.face_idx = faces->q2f(it->p.piggy3.local_num, 2*dir + 1);
    if(tmp_neighbor_seed.face_idx != NO_VELOCITY && tmp_neighbor_seed.face_idx != center_face_idx && set_of_faces.find(tmp_neighbor_seed) == set_of_faces.end())
    {
      faces->xyz_fr_f(tmp_neighbor_seed.face_idx, dir, tmp_neighbor_seed.xyz_face);
      set_of_faces.insert(tmp_neighbor_seed);
    }
  }
  quad_ngbd.clear();
}

inline void add_all_faces_to_sets_and_clear_set_of_quad(const my_p4est_faces_t* faces, std::set<p4est_locidx_t> set_of_faces[P4EST_DIM], set_of_neighboring_quadrants& quad_ngbd)
{
  p4est_locidx_t f_tmp;
  for (set_of_neighboring_quadrants::const_iterator it = quad_ngbd.begin(); it != quad_ngbd.end(); ++it) {
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      f_tmp = faces->q2f(it->p.piggy3.local_num, 2*dir);
      if(f_tmp != NO_VELOCITY)
        set_of_faces[dir].insert(f_tmp);
      f_tmp = faces->q2f(it->p.piggy3.local_num, 2*dir + 1);
      if(f_tmp != NO_VELOCITY)
        set_of_faces[dir].insert(f_tmp);
    }
  }
  quad_ngbd.clear();
}

inline bool no_wall_in_face_neighborhood(const uniform_face_ngbd *face_neighbors)
{
  bool to_return = true;
  for (unsigned char dir = 0; dir < P4EST_FACES && to_return; ++dir)
    to_return = to_return && face_neighbors->neighbor_face_idx[dir] >= 0;
  return to_return;
}

inline bool extra_layer_in_tranverse_directon_may_be_required(const unsigned dir, const double *tree_dim)
{
  // [Raphael :]
  // In case of very stretched grid, the standard check for local uniformity may fail
  // For instance, consider in 2D
  // |---------------------------|-------------|-------------|---------------------------|---------------------------|
  // |                           |             |             |                           |                           |
  // |                           |-------------|-------------|---------------------------|---------------------------|
  // |                           |             x             |                           x                           |
  // |---------------------------|-------------|-------------|---------------------------|---------------------------|
  // |                           |                           |                           |                           |
  // |                           |                           |                           |                           |
  // |                           |                           |                           |                           |
  // |---------------------------|---------------------------|---------------------------|---------------------------|
  // |                           |                           |                           |                           |
  // |                           |                           O                           |                           |
  // |                           |                           |                           |                           |
  // |---------------------------|---------------------------|---------------------------|---------------------------|
  // |                           |                           |                           |                           |
  // |                           |                           |                           |                           |
  // |                           |                           |                           |                           |
  // |---------------------------|---------------------------|---------------------------|---------------------------|
  //
  // One can show that if dy/dx < sqrt(21.0/4.0), the faces marked 'x' actually come into play for the construction of
  // the Voronoi cell associated with the face marked 'O' here above.
  // A similar analysis can be done in 3D, it is very tedious and basically impossible to illustrate like the above in
  // a simple comment but, except if I made a mistake, the result here below should be correct.
  //
  // -->notice that one may need to access THIRD-degree neighbor quadrants in such a case

#ifdef P4_TO_P8
  return (MIN(0.25*21.0*SQR(tree_dim[(dir + 1)%P4EST_DIM]/tree_dim[dir]) - 0.5*SQR(tree_dim[(dir + 2)%P4EST_DIM]/tree_dim[dir]), 0.25*21.0*SQR(tree_dim[(dir + 2)%P4EST_DIM]/tree_dim[dir]) - 0.5*SQR(tree_dim[(dir + 1)%P4EST_DIM]/tree_dim[dir])) < 1.0);
#else
  return (0.25*21.0*SQR(tree_dim[(dir + 1)%P4EST_DIM]/tree_dim[dir]) < 1.0);
#endif
}

inline bool check_past_sharing_quad_is_required(const unsigned dir, const double *tree_dim)
{
  // [Raphael :]
  // In case of very stretched grid, one may need to fetch neighbor faces past the parallel face across the quadrant sharing the faces,
  // even if that faces is resolved
  // For instance, consider in 2D
  // |-------------|-------------|---------------------------|
  // |             |             |                           |
  // |-------------|-------------|                           |
  // |             |             |                           |
  // |-------------|------x------|---------------------------|
  // |                           |                           |
  // |                           |                           |
  // |                           |                           |
  // |-------------$-----------------------------------------|
  // |                                                       |
  // |                                                       |
  // |                                                       |
  // |                                                       |
  // |                                                       |
  // |                                                       |
  // |                                                       |
  // |---------------------------O---------------------------|
  //
  // One can show that, if dx/dy > 4*sqrt(3/5), the face marked 'O' actually comes into play for the construction of the Voronoi cell associated with the face marked 'x' here above.
  // (and the other way around). (The standard construction procedure does not fetch it)
  // A similar analysis can be done in 3D (although it's hard to figure out if it is general enough), but it is basically
  // impossible to illustrate like the above in a simple comment but, except if I made a mistake, the result here below should be correct.

  // check if alpha^2 (+ beta^2) < 48/5 where alpha (beta) is (are) the tranverse aspect ratio(s)
  return (SQR(tree_dim[(dir + 1)%P4EST_DIM]/tree_dim[dir]) ONLY3D(+ SQR(tree_dim[(dir + 2)%P4EST_DIM]/tree_dim[dir])) > 48.0/5.0);
}

inline bool third_degree_ghost_are_required(const double *tree_dim)
{
  bool to_return = false;
  for (unsigned char dir = 0; dir < P4EST_DIM && !to_return; ++dir)
    to_return = to_return || extra_layer_in_tranverse_directon_may_be_required(dir, tree_dim) || check_past_sharing_quad_is_required(dir, tree_dim);
  return to_return;
}

voro_cell_type compute_voronoi_cell(Voronoi_DIM &voronoi_cell, const my_p4est_faces_t* faces, const p4est_locidx_t &f_idx, const unsigned char & dir, const BoundaryConditionsDIM *bc, const PetscScalar *face_is_well_defined_p);

#endif /* MY_P4EST_FACES_H */
