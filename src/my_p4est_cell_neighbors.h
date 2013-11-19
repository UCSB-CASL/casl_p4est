#ifndef MY_P4EST_CELL_NEIGHBORS_H
#define MY_P4EST_CELL_NEIGHBORS_H

#ifdef P4_TO_P8
#include <p8est.h>
#include <p8est_ghost.h>
#include <src/my_p8est_hierarchy.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_node_neighbors.h>
#else
#include <p4est.h>
#include <p4est_ghost.h>
#include <src/my_p4est_hierarchy.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_node_neighbors.h>
#endif

#include <vector>

class my_p4est_cell_neighbors_t {
  friend class PoissonSolverCellBase;
  friend class InterpolatingFunctionCellBase;

  my_p4est_hierarchy_t *hierarchy;
  p4est_t *p4est;
  p4est_ghost_t *ghost;
  my_p4est_brick_t *myb;

  /* local quadrants from 0 .. local_number_of_quadrants - 1
   * ghosts from local_number_of_quadrants .. local_number_of_quadrants + num_ghosts
   */
  struct quad_info_t{
    p4est_locidx_t locidx;
    p4est_gloidx_t gloidx;
    p4est_topidx_t tree_idx;
    const p4est_quadrant_t *quad;
  };
  struct cell_neighbors_t{
    std::vector<quad_info_t> neighbors;
    std::vector<p4est_locidx_t> offsets;
  };

  cell_neighbors_t faces, corners;
#ifdef P4_TO_P8
  cell_neighbors_t edges;
#endif

  p4est_locidx_t n_quads;

  void initialize_neighbors();

  /**
     * perform the recursive search to find the neighboring cells of a cell
     */
  void find_face_neighbor_cells_of_cell_recursive(p4est_topidx_t tr, int ind, int dir_f);

  /**
     * find the neighboring cells of cell in a given direction, the possible directions are :
     * \param [in] q      the index of the quadrant whose neighbors are queried
     * \param [in] tr     the tree in which the quadrant is located
     * \param [in] dir    the direction to investigate. must be in [0, P4EST_FACES).
     */
  void find_face_neighbor_cells_of_cell( const p4est_quadrant_t* quad, p4est_locidx_t q, p4est_topidx_t tr, int dir_f);

  /**
     * perform the recursive search to find the neighboring cells of a cell
     */
  void find_corner_neighbor_cells_of_cell_recursive(p4est_topidx_t tr, int ind, int dir_c);

  /**
     * find the neighboring cells of cell in a given direction, the possible directions are :
     * \param [in] q      the index of the quadrant whose neighbors are queried
     * \param [in] tr     the tree in which the quadrant is located
     * \param [in] dir    the direction to investigate. must b ein [0, P4EST_CHILDREN)
     */
  void find_corner_neighbor_cells_of_cell( const p4est_quadrant_t* quad, p4est_locidx_t q, p4est_topidx_t tr, int dir_c);

#ifdef P4_TO_P8
  /**
     * perform the recursive search to find the neighboring cells of a cell
     */
  void find_edge_neighbor_cells_of_cell_recursive(p4est_topidx_t tr, int ind, int dir_e);

  /**
     * find the neighboring cells of cell in a given direction, the possible directions are :
     * \param [in] q      the index of the quadrant whose neighbors are queried
     * \param [in] tr     the tree in which the quadrant is located
     * \param [in] dir    the direction to investigate. must be in [0, P8EST_EDGES).
     */
  void find_edge_neighbor_cells_of_cell( const p4est_quadrant_t* quad, p4est_locidx_t q, p4est_topidx_t tr, int dir_e);
#endif

public:
  my_p4est_cell_neighbors_t( my_p4est_hierarchy_t *hierarchy_ )
    : hierarchy(hierarchy_), p4est(hierarchy_->p4est), ghost(hierarchy_->ghost), myb(hierarchy_->myb),
      n_quads(p4est->local_num_quadrants + ghost->ghosts.elem_count)
  {
    faces.neighbors.reserve(P4EST_FACES * n_quads);
    faces.offsets.resize(P4EST_FACES*n_quads + 1, 0);

    corners.neighbors.reserve(P4EST_CHILDREN * n_quads);
    corners.offsets.resize(P4EST_CHILDREN*n_quads + 1, 0);

#ifdef P4_TO_P8
    edges.neighbors.reserve(P8EST_EDGES * n_quads);
    edges.offsets.resize(P8EST_EDGES*n_quads + 1, 0);
#endif

    initialize_neighbors();
  }

  inline const quad_info_t* face_begin(p4est_locidx_t q, int dir_f) const {
#ifdef CASL_THROWS
    if (dir_f < 0 || dir_f >= P4EST_FACES)
      throw std::invalid_argument("invalid face direction index.");
    if (q < 0 || q >= n_quads)
      throw std::invalid_argument("given quadrant index is neigher local nor ghost.");
#endif
    return &faces.neighbors[faces.offsets[q*P4EST_FACES + dir_f]];
  }

  inline const quad_info_t* face_end(p4est_locidx_t q, int dir_f) const {
#ifdef CASL_THROWS
    if (dir_f < 0 || dir_f >= P4EST_FACES)
      throw std::invalid_argument("invalid face direction index.");
    if (q < 0 || q >= n_quads)
      throw std::invalid_argument("given quadrant index does is neigher local nor ghost.");
#endif
    return &faces.neighbors[faces.offsets[q*P4EST_FACES + dir_f + 1]];
  }

  inline size_t face_size(p4est_locidx_t q, int dir_f) const {
    return faces.offsets[q*P4EST_FACES + dir_f + 1] - faces.offsets[q*P4EST_FACES + dir_f];
  }

#ifdef P4_TO_P8
  inline const quad_info_t* edge_begin(p4est_locidx_t q, int dir_e) const {
#ifdef CASL_THROWS
    if (dir_e < 0 || dir_e >= P8EST_EDGES)
      throw std::invalid_argument("invalid edge direction index.");
    if (q < 0 || q >= n_quads)
      throw std::invalid_argument("given quadrant index is neigher local nor ghost.");
#endif
    return &edges.neighbors[edges.offsets[q*P8EST_EDGES + dir_e]];
  }

  inline const quad_info_t* edge_end(p4est_locidx_t q, int dir_e) const {
#ifdef CASL_THROWS
    if (dir_e < 0 || dir_e >= P8EST_EDGES)
      throw std::invalid_argument("invalid edge direction index.");
    if (q < 0 || q >= n_quads)
      throw std::invalid_argument("given quadrant index does is neigher local nor ghost.");
#endif
    return &edges.neighbors[edges.offsets[q*P8EST_EDGES + dir_e + 1]];
  }

  inline size_t edge_size(p4est_locidx_t q, int dir_e) const {
#ifdef CASL_THROWS
    if (dir_e < 0 || dir_e >= P8EST_EDGES)
      throw std::invalid_argument("invalid edge direction index.");
    if (q < 0 || q >= n_quads)
      throw std::invalid_argument("given quadrant index does is neigher local nor ghost.");
#endif
    return edges.offsets[q*P8EST_EDGES + dir_e + 1] - edges.offsets[q*P8EST_EDGES + dir_e];
  }
#endif

  inline const quad_info_t* corner_begin(p4est_locidx_t q, int dir_c) const {
#ifdef CASL_THROWS
    if (dir_c < 0 || dir_c >= P4EST_CHILDREN)
      throw std::invalid_argument("invalid corner direction index.");
    if (q < 0 || q >= n_quads)
      throw std::invalid_argument("given quadrant index does is neigher local nor ghost.");
#endif
      return &corners.neighbors[corners.offsets[q*P4EST_CHILDREN + dir_c]];
  }

  inline const quad_info_t* corner_end(p4est_locidx_t q, int dir_c) const {
#ifdef CASL_THROWS
    if (dir_c < 0 || dir_c >= P4EST_CHILDREN)
      throw std::invalid_argument("invalid corner direction index.");
    if (q < 0 || q >= n_quads)
      throw std::invalid_argument("given quadrant index does is neigher local nor ghost.");
#endif
      return &corners.neighbors[corners.offsets[q*P4EST_CHILDREN + dir_c + 1]];
  }

  inline size_t corner_size(p4est_locidx_t q, int dir_c) const {
#ifdef CASL_THROWS
    if (dir_c < 0 || dir_c >= P4EST_CHILDREN)
      throw std::invalid_argument("invalid corner direction index.");
    if (q < 0 || q >= n_quads)
      throw std::invalid_argument("given quadrant index does is neigher local nor ghost.");
#endif
    return corners.offsets[q*P4EST_CHILDREN + dir_c + 1] - corners.offsets[q*P4EST_CHILDREN + dir_c];
  }

  void print_debug(p4est_locidx_t q, FILE* stream = stdout);
  void write_cell_neighbors_vtk(p4est_locidx_t qu, p4est_topidx_t tr, const char* filename);
  void write_triangulation(const my_p4est_node_neighbors_t& qnnn, const char* filename);
  void write_cell_triangulation_vtk_m00(p4est_locidx_t qu, p4est_topidx_t tr, const char *filename);
  void write_cell_triangulation_vtk_p00(p4est_locidx_t qu, p4est_topidx_t tr, const char *filename);
  void write_cell_triangulation_vtk_0m0(p4est_locidx_t qu, p4est_topidx_t tr, const char *filename);
  void write_cell_triangulation_vtk_0p0(p4est_locidx_t qu, p4est_topidx_t tr, const char *filename);
  void write_cell_triangulation_vtk_00m(p4est_locidx_t qu, p4est_topidx_t tr, const char *filename);
  void write_cell_triangulation_vtk_00p(p4est_locidx_t qu, p4est_topidx_t tr, const char *filename);
#ifndef P4_TO_P8
  void write_triangulation(const char* filename);
#endif
};

#endif /* !MY_P4EST_CELL_NEIGHBORS_H */
