#ifndef NEIGHBORS_H
#define NEIGHBORS_H

#include <p4est_mesh.h>
#include <src/my_p4est_nodes.h>

#include <vector>
#include <stdexcept>
#include <sstream>
#include <map>

using std::map; using std::cout; using std::endl; using std::vector;
using std::invalid_argument; using std::runtime_error;

class CellNeighbors{
public:
  typedef vector<p4est_quadrant_t*> quad_array;
private:

  p4est_ghost_t *ghost;
  p4est_mesh_t  *mesh;
  p4est_t *p4est;

  int8_t         *q2f; // quad-to-face
  p4est_locidx_t *q2q; // quad-to-quad
  sc_array_t     *q2h; // quad-to-half
  p4est_locidx_t *g2i; // ghost-to-index
  int            *g2p; // ghost-to-proc

  int rank;
  p4est_locidx_t n_local,n_ghost;
  p4est_gloidx_t *global_proc_offset;

  vector<quad_array> ngbd [4];
  std::map<p4est_gloidx_t, p4est_locidx_t> global2local_map;

  bool initialized;
public:

  CellNeighbors(p4est_t *p4est_)
    : p4est(p4est_)
  {
    // First generate the ghosted layer
    ghost = p4est_ghost_new(p4est, P4EST_CONNECT_DEFAULT);
    // Create a mesh data structure
    mesh = p4est_mesh_new(p4est, ghost, P4EST_CONNECT_DEFAULT);

    q2f = mesh->quad_to_face;
    q2q = mesh->quad_to_quad;
    q2h = mesh->quad_to_half;
    g2i = mesh->ghost_to_index;
    g2p = mesh->ghost_to_proc;

    n_local = mesh->local_num_quadrants;
    n_ghost = mesh->ghost_num_quadrants;

    for (short i=0; i<P4EST_FACES; ++i)
      ngbd[i].resize(n_local);

    initialized = false;

    rank = p4est->mpirank;
    global_proc_offset = p4est->global_first_quadrant;

    for (p4est_locidx_t i=0; i<n_ghost; ++i){
      global2local_map.insert(std::make_pair(g2i[i] + global_proc_offset[g2p[i]], n_local+i));
    }
  }
  ~CellNeighbors(){
    // Destroy p4est objects
    p4est_ghost_destroy(ghost);
    p4est_mesh_destroy(mesh);

    initialized = false;
  }

  void Init();

  inline p4est_gloidx_t local2global(const p4est_locidx_t& locidx){
    if (locidx >= 0 && locidx < n_local){
      return global_proc_offset[rank] + locidx;
    } else if (locidx >= n_local && locidx < n_local+n_ghost){
      return g2i[locidx - n_local] + global_proc_offset[g2p[locidx - n_local]];
    } else {
#ifdef CASL_THROWS
      std::ostringstream oss;
      oss << "[CASL_ERROR]: local index '" << locidx << "' is neither local nor ghost";
      throw std::invalid_argument(oss.str());
#endif
    }
  }

  inline p4est_locidx_t global2local(const p4est_gloidx_t& gloidx){
    if (gloidx >= global_proc_offset[rank] && gloidx < global_proc_offset[rank+1]){
      return gloidx - global_proc_offset[rank];
    } else if (global2local_map.find(gloidx) != global2local_map.end()) {
      return global2local_map[gloidx];
    } else {
#ifdef CASL_THROWS
      std::ostringstream oss;
      oss << "[CASL_ERROR]: global index '" << gloidx << "' is neither local nor ghost";
      throw std::invalid_argument(oss.str());
#endif
    }
  }

  inline const vector<quad_array>&
  get_face_neighbors(const short& face_id) {
#ifdef CASL_THROWS
    if (!initialized)
      throw runtime_error("[CASL_ERROR]: List is not initilized. You must call Init() first.");
    if (face_id>=4 || face_id<0)
      throw runtime_error("[CASL_ERROR]: face_id must either be 0, 1, 2, or 3.");
#endif
    return ngbd[face_id];
  }

  inline const vector<quad_array>&
  get_m0_neighbors() {
#ifdef CASL_THROWS
    if (!initialized)
      throw runtime_error("[CASL_ERROR]: List is not initilized. You must call Init() first.");
#endif
    return ngbd[0];
  }

  inline const vector<quad_array>&
  get_p0_neighbors() {
#ifdef CASL_THROWS
    if (!initialized)
      throw runtime_error("[CASL_ERROR]: List is not initilized. You must call Init() first.");
#endif
    return ngbd[1];
  }

  inline const vector<quad_array>&
  get_0m_neighbors() {
#ifdef CASL_THROWS
    if (!initialized)
      throw runtime_error("[CASL_ERROR]: List is not initilized. You must call Init() first.");
#endif
    return ngbd[2];
  }

  inline const vector<quad_array>&
  get_0p_neighbors() {
#ifdef CASL_THROWS
    if (!initialized)
      throw runtime_error("[CASL_ERROR]: List is not initilized. You must call Init() first.");
#endif
    return ngbd[3];
  }

  friend class NodeNeighbors;
};

class NodeNeighbors{
public:
    typedef vector<p4est_indep_t*> node_array;

private:

  CellNeighbors *cell_ngbds;
  p4est_nodes_t *nodes;
  p4est_locidx_t *local_index;

  p4est_locidx_t num_indep, num_hang;

  bool initialized;

public:
  NodeNeighbors(CellNeighbors* cell_ngbds_)
    : cell_ngbds(cell_ngbds_)
  {
    // Set up the node data structure
    nodes = p4est_nodes_new(cell_ngbds->p4est, cell_ngbds->ghost);
    local_index = nodes->local_nodes;
    num_indep   = nodes->indep_nodes.elem_count;
    num_hang    = nodes->face_hangings.elem_count;
    initialized = false;


  }
  ~NodeNeighbors(){
    p4est_nodes_destroy(nodes);
    initialized = false;
  }

  void Init();
};

#endif // NEIGHBORS_H
