#ifdef P4_TO_P8
#include "my_p8est_cell_neighbors.h"
#else
#include "my_p4est_cell_neighbors.h"
#endif

#ifdef P4_TO_P8
#define VTK_CELL_TYPE 10 /* VTL_TETRA */
#else
#define VTK_CELL_TYPE 5  /* VTL_TRIANGLE */
#endif


void my_p4est_cell_neighbors_t::initialize_neighbors()
{
  // find neighboring quadrants of local quadrants
  for( p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx )
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for( size_t q=0; q<tree->quadrants.elem_count; ++q ){
      const p4est_quadrant_t* quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
      p4est_locidx_t q_locidx = q + tree->quadrants_offset;

      /* neighbors across faces */
      for(int f = 0; f<P4EST_FACES; ++f )
        find_face_neighbor_cells_of_cell(quad, q_locidx, tree_idx, f);

      /* neighbors across corners */
      for(int c = 0; c<P4EST_CHILDREN; ++c)
        find_corner_neighbor_cells_of_cell(quad, q_locidx, tree_idx, c);

#ifdef P4_TO_P8
      /* neighbors across edges (3D only) */
      for(int e = 0; e<P8EST_EDGES; ++e)
        find_edge_neighbor_cells_of_cell(quad, q_locidx, tree_idx, e);
#endif
    }
  }

  // find neighboring quadrants of ghost quadrants
  for (size_t q = 0; q < ghost->ghosts.elem_count; ++q){
    const p4est_quadrant_t* quad = (const p4est_quadrant_t*)sc_array_index(&ghost->ghosts, q);
    p4est_locidx_t q_locidx = q + p4est->local_num_quadrants;

    /* neighbors across faces */
    for(int f = 0; f<P4EST_FACES; ++f )
      find_face_neighbor_cells_of_cell(quad, q_locidx, quad->p.piggy3.which_tree, f);

    /* neighbors across corners */
    for(int c = 0; c<P4EST_CHILDREN; ++c)
      find_corner_neighbor_cells_of_cell(quad, q_locidx, quad->p.piggy3.which_tree, c);

#ifdef P4_TO_P8
    /* neighbors across edges (3D only) */
    for(int e = 0; e<P8EST_EDGES; ++e)
      find_edge_neighbor_cells_of_cell(quad, q_locidx, quad->p.piggy3.which_tree, e);
#endif
  }
}

void my_p4est_cell_neighbors_t::find_face_neighbor_cells_of_cell(const p4est_quadrant_t* quad, p4est_locidx_t q, p4est_topidx_t tree_idx, int dir_f )
{
  faces.offsets[q*P4EST_FACES + dir_f + 1] = faces.neighbors.size();

  const static char fi [] = {-1,  1,  0,  0,  0,  0};
  const static char fj [] = { 0,  0, -1,  1,  0,  0};
#ifdef P4_TO_P8
  const static char fk [] = { 0,  0,  0,  0, -1,  1};
#endif

  /* construct the coordinate of the neighbor cell of the same size in the given direction */
  p4est_qcoord_t size = P4EST_QUADRANT_LEN(quad->level);

  p4est_topidx_t nb_tree_idx = tree_idx;
  p4est_qcoord_t i_nb = quad->x + fi[dir_f]*size;
  p4est_qcoord_t j_nb = quad->y + fj[dir_f]*size;
#ifdef P4_TO_P8
  p4est_qcoord_t k_nb = quad->z + fk[dir_f]*size;
#endif

  /* check if quadrant is on a boundary */
  if(quad->x == 0 && dir_f == dir::f_m00)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_m00];
    if(nb_tree_idx == tree_idx) return;
    i_nb = P4EST_ROOT_LEN - size;
  }
  else if(quad->x+size == P4EST_ROOT_LEN && dir_f == dir::f_p00)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_p00];
    if(nb_tree_idx == tree_idx) return;
    i_nb = 0;
  }
  else if(quad->y == 0 && dir_f == dir::f_0m0)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_0m0];
    if(nb_tree_idx == tree_idx) return;
    j_nb = P4EST_ROOT_LEN - size;
  }
  else if(quad->y+size == P4EST_ROOT_LEN && dir_f == dir::f_0p0)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_0p0];
    if(nb_tree_idx == tree_idx) return;
    j_nb = 0;
  }
#ifdef P4_TO_P8
  else if(quad->z == 0 && dir_f == dir::f_00m)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_00m];
    if(nb_tree_idx == tree_idx) return;
    k_nb = P4EST_ROOT_LEN - size;
  }
  else if(quad->z+size == P4EST_ROOT_LEN && dir_f == dir::f_00p)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_00p];
    if(nb_tree_idx == tree_idx) return;
    k_nb = 0;
  }
#endif

  /* find the constructed neighbor cell of the same size */
  int ind = 0;
  while( hierarchy->trees[nb_tree_idx][ind].level != quad->level && hierarchy->trees[nb_tree_idx][ind].child != CELL_LEAF )
  {
    p4est_qcoord_t half_size = P4EST_QUADRANT_LEN(hierarchy->trees[nb_tree_idx][ind].level) / 2;
    bool i_search = ( i_nb >= hierarchy->trees[nb_tree_idx][ind].imin + half_size );
    bool j_search = ( j_nb >= hierarchy->trees[nb_tree_idx][ind].jmin + half_size );
#ifdef P4_TO_P8
    bool k_search = ( k_nb >= hierarchy->trees[nb_tree_idx][ind].kmin + half_size );
#endif
#ifdef P4_TO_P8
    ind = hierarchy->trees[nb_tree_idx][ind].child + 4*k_search + 2*j_search + i_search;
#else
    ind = hierarchy->trees[nb_tree_idx][ind].child + 2*j_search + i_search;
#endif
  }

  /* now find the children of this constructed cell in the desired direction and add them to the list */
  find_face_neighbor_cells_of_cell_recursive( nb_tree_idx, ind, dir_f );
  faces.offsets[q*P4EST_FACES + dir_f + 1] = faces.neighbors.size();
}


void my_p4est_cell_neighbors_t::find_face_neighbor_cells_of_cell_recursive(p4est_topidx_t tr, int ind, int dir_f )
{

  if (hierarchy->trees[tr][ind].child == CELL_LEAF)
  {
    if (hierarchy->trees[tr][ind].quad == NOT_A_P4EST_QUADRANT) return;

    quad_info_t qinfo;
    qinfo.locidx = hierarchy->trees[tr][ind].quad;
    qinfo.tree_idx = tr;

    const p4est_quadrant_t *quad = NULL;

    if (qinfo.locidx < p4est->local_num_quadrants) {// local quadrant
      p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tr);
      quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, qinfo.locidx - tree->quadrants_offset);
      qinfo.gloidx = p4est->global_first_quadrant[p4est->mpirank] + qinfo.locidx;
    } else {
      quad = (const p4est_quadrant_t*)sc_array_index(&ghost->ghosts, qinfo.locidx - p4est->local_num_quadrants);
      qinfo.gloidx = p4est->global_first_quadrant[hierarchy->trees[tr][ind].owner_rank] + quad->p.piggy3.local_num;
    }
    qinfo.quad = quad;

    faces.neighbors.push_back(qinfo);
    return;
  }

  switch(dir_f)
  {
  case dir::f_m00:
    find_face_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_pmm, dir_f);
    find_face_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_ppm, dir_f);
#ifdef P4_TO_P8
    find_face_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_pmp, dir_f);
    find_face_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_ppp, dir_f);
#endif
    break;
  case dir::f_p00:
    find_face_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mmm, dir_f);
    find_face_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mpm, dir_f);
#ifdef P4_TO_P8
    find_face_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mmp, dir_f);
    find_face_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mpp, dir_f);
#endif
    break;
  case dir::f_0m0:
    find_face_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mpm, dir_f);
    find_face_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_ppm, dir_f);
#ifdef P4_TO_P8
    find_face_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mpp, dir_f);
    find_face_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_ppp, dir_f);
#endif
    break;
  case dir::f_0p0:
    find_face_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mmm, dir_f);
    find_face_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_pmm, dir_f);
#ifdef P4_TO_P8
    find_face_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mmp, dir_f);
    find_face_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_pmp, dir_f);
#endif
    break;
#ifdef P4_TO_P8
  case dir::f_00m:
    find_face_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mmp, dir_f);
    find_face_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_pmp, dir_f);
    find_face_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mpp, dir_f);
    find_face_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_ppp, dir_f);
    break;
  case dir::f_00p:
    find_face_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mmm, dir_f);
    find_face_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_pmm, dir_f);
    find_face_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mpm, dir_f);
    find_face_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_ppm, dir_f);
    break;
#endif
  }
}

void my_p4est_cell_neighbors_t::find_corner_neighbor_cells_of_cell(const p4est_quadrant_t* quad, p4est_locidx_t q, p4est_topidx_t tree_idx, int dir_c )
{
  corners.offsets[q*P4EST_CHILDREN + dir_c + 1] = corners.neighbors.size();

  const static char ci [] = {-1,  1, -1,  1, -1,  1, -1,  1};
  const static char cj [] = {-1, -1,  1,  1, -1, -1,  1,  1};
#ifdef P4_TO_P8
  const static char ck [] = {-1, -1, -1, -1,  1,  1,  1,  1};
#endif

  /* construct the coordinate of the neighbor cell of the same size in the given direction */
  p4est_qcoord_t size = P4EST_QUADRANT_LEN(quad->level);

  p4est_topidx_t nb_tree_idx = tree_idx;
  p4est_qcoord_t i_nb = quad->x + ci[dir_c]*size;
  p4est_qcoord_t j_nb = quad->y + cj[dir_c]*size;
#ifdef P4_TO_P8
  p4est_qcoord_t k_nb = quad->z + ck[dir_c]*size;
#endif

  /* check if quadrant is on a boundary. we need to check 26 cases in 3D or 8 cases in 2D
   * 8(4) corners in 3D(2D)
   * 12 edges in 3D
   * 6(4) faces in 3D(2D)
   */
  \
  /* search in the corner directions */
  // 0) mmm
  if (quad->x == 0  && ci[dir_c] == -1 &&
      quad->y == 0  && cj[dir_c] == -1
    #ifdef P4_TO_P8
      &&
      quad->z == 0  && ck[dir_c] == -1
    #endif
      )
  {
    p4est_topidx_t tmp_tree_idx[P4EST_DIM];
    nb_tree_idx = tmp_tree_idx[0] = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_m00]; // m00
    if(tmp_tree_idx[0] == tree_idx)        return;
    nb_tree_idx = tmp_tree_idx[1] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0m0]; // 0m0
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) return;
#ifdef P4_TO_P8
    nb_tree_idx = tmp_tree_idx[2] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[1] + dir::f_00m]; // 00m
    if(tmp_tree_idx[2] == tmp_tree_idx[1]) return;
#endif

    i_nb = P4EST_ROOT_LEN - size;
    j_nb = P4EST_ROOT_LEN - size;
#ifdef P4_TO_P8
    k_nb = P4EST_ROOT_LEN - size;
#endif
  }
  // 1) pmm
  else if (quad->x == P4EST_ROOT_LEN - size && ci[dir_c] ==  1 &&
           quad->y == 0                     && cj[dir_c] == -1
         #ifdef P4_TO_P8
           &&
           quad->z == 0                     && ck[dir_c] == -1
         #endif
           )
  {
    p4est_topidx_t tmp_tree_idx[P4EST_DIM];
    nb_tree_idx = tmp_tree_idx[0] = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_p00]; // p00
    if(tmp_tree_idx[0] == tree_idx)        return;
    nb_tree_idx = tmp_tree_idx[1] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0m0]; // 0m0
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) return;
#ifdef P4_TO_P8
    nb_tree_idx = tmp_tree_idx[2] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[1] + dir::f_00m]; // 00m
    if(tmp_tree_idx[2] == tmp_tree_idx[1]) return;
#endif

    i_nb = 0;
    j_nb = P4EST_ROOT_LEN - size;
#ifdef P4_TO_P8
    k_nb = P4EST_ROOT_LEN - size;
#endif
  }
  // 2) mpm
  else if (quad->x == 0                     && ci[dir_c] == -1 &&
           quad->y == P4EST_ROOT_LEN - size && cj[dir_c] ==  1
         #ifdef P4_TO_P8
           &&
           quad->z == 0                     && ck[dir_c] == -1
         #endif
           )
  {
    p4est_topidx_t tmp_tree_idx[P4EST_DIM];
    nb_tree_idx = tmp_tree_idx[0] = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_m00]; // m00
    if(tmp_tree_idx[0] == tree_idx)        return;
    nb_tree_idx = tmp_tree_idx[1] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0p0]; // 0p0
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) return;
#ifdef P4_TO_P8
    nb_tree_idx = tmp_tree_idx[2] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[1] + dir::f_00m]; // 00m
    if(tmp_tree_idx[2] == tmp_tree_idx[1]) return;
#endif

    i_nb = P4EST_ROOT_LEN - size;
    j_nb = 0;
#ifdef P4_TO_P8
    k_nb = P4EST_ROOT_LEN - size;
#endif
  }
  // 3) ppm
  else if (quad->x == P4EST_ROOT_LEN - size && ci[dir_c] ==  1 &&
           quad->y == P4EST_ROOT_LEN - size && cj[dir_c] ==  1
         #ifdef P4_TO_P8
           &&
           quad->z == 0                     && ck[dir_c] == -1
         #endif
           )
  {
    p4est_topidx_t tmp_tree_idx[P4EST_DIM];
    nb_tree_idx = tmp_tree_idx[0] = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_p00]; // p00
    if(tmp_tree_idx[0] == tree_idx)        return;
    nb_tree_idx = tmp_tree_idx[1] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0p0]; // 0p0
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) return;
#ifdef P4_TO_P8
    nb_tree_idx = tmp_tree_idx[2] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[1] + dir::f_00m]; // 00m
    if(tmp_tree_idx[2] == tmp_tree_idx[1]) return;
#endif

    i_nb = 0;
    j_nb = 0;
#ifdef P4_TO_P8
    k_nb = P4EST_ROOT_LEN - size;
#endif
  }
#ifdef P4_TO_P8
  // 4) mmp
  else if (quad->x == 0                     && ci[dir_c] == -1 &&
           quad->y == 0                     && cj[dir_c] == -1 &&
           quad->z == P4EST_ROOT_LEN - size && ck[dir_c] ==  1 )
  {
    p4est_topidx_t tmp_tree_idx[P4EST_DIM];
    nb_tree_idx = tmp_tree_idx[0] = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_m00]; // m00
    if(tmp_tree_idx[0] == tree_idx)        return;
    nb_tree_idx = tmp_tree_idx[1] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0m0]; // 0m0
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) return;
    nb_tree_idx = tmp_tree_idx[2] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[1] + dir::f_00p]; // 00p
    if(tmp_tree_idx[2] == tmp_tree_idx[1]) return;

    i_nb = P4EST_ROOT_LEN - size;
    j_nb = P4EST_ROOT_LEN - size;
    k_nb = 0;
  }
  // 5) pmp
  else if (quad->x == P4EST_ROOT_LEN - size && ci[dir_c] ==  1 &&
           quad->y == 0                     && cj[dir_c] == -1 &&
           quad->z == P4EST_ROOT_LEN - size && ck[dir_c] ==  1 )
  {
    p4est_topidx_t tmp_tree_idx[P4EST_DIM];
    nb_tree_idx = tmp_tree_idx[0] = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_p00]; // p00
    if(tmp_tree_idx[0] == tree_idx)        return;
    nb_tree_idx = tmp_tree_idx[1] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0m0]; // 0m0
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) return;
    nb_tree_idx = tmp_tree_idx[2] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[1] + dir::f_00p]; // 00p
    if(tmp_tree_idx[2] == tmp_tree_idx[1]) return;

    i_nb = 0;
    j_nb = P4EST_ROOT_LEN - size;
    k_nb = 0;
  }
  // 6) mpp
  else if (quad->x == 0                     && ci[dir_c] == -1 &&
           quad->y == P4EST_ROOT_LEN - size && cj[dir_c] ==  1 &&
           quad->z == P4EST_ROOT_LEN - size && ck[dir_c] ==  1 )
  {
    p4est_topidx_t tmp_tree_idx[P4EST_DIM];
    nb_tree_idx = tmp_tree_idx[0] = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_m00]; // m00
    if(tmp_tree_idx[0] == tree_idx)        return;
    nb_tree_idx = tmp_tree_idx[1] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0p0]; // 0p0
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) return;
    nb_tree_idx = tmp_tree_idx[2] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[1] + dir::f_00p]; // 00p
    if(tmp_tree_idx[2] == tmp_tree_idx[1]) return;

    i_nb = P4EST_ROOT_LEN - size;
    j_nb = 0;
    k_nb = 0;
  }
  // 7) ppp
  else if (quad->x == P4EST_ROOT_LEN - size && ci[dir_c] ==  1 &&
           quad->y == P4EST_ROOT_LEN - size && cj[dir_c] ==  1 &&
           quad->z == P4EST_ROOT_LEN - size && ck[dir_c] ==  1 )
  {
    p4est_topidx_t tmp_tree_idx[P4EST_DIM];
    nb_tree_idx = tmp_tree_idx[0] = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_p00]; // p00
    if(tmp_tree_idx[0] == tree_idx)        return;
    nb_tree_idx = tmp_tree_idx[1] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0p0]; // 0p0
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) return;
    nb_tree_idx = tmp_tree_idx[2] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[1] + dir::f_00p]; // 00p
    if(tmp_tree_idx[2] == tmp_tree_idx[1]) return;

    i_nb = 0;
    j_nb = 0;
    k_nb = 0;
  }

  /* search in the edge directions */
  // 0) m0m
  else if (quad->x == 0                     && ci[dir_c] == -1 &&
           quad->z == 0                     && ck[dir_c] == -1)
  {
    p4est_topidx_t tmp_tree_idx[P4EST_DIM - 1];
    nb_tree_idx = tmp_tree_idx[0] = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_m00]; // m00
    if(tmp_tree_idx[0] == tree_idx)        return;
    nb_tree_idx = tmp_tree_idx[1] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_00m]; // 00m
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) return;

    i_nb = P4EST_ROOT_LEN - size;
    k_nb = P4EST_ROOT_LEN - size;
  }
  // 1) p0m
  else if (quad->x == P4EST_ROOT_LEN - size && ci[dir_c] ==  1 &&
           quad->z == 0                     && ck[dir_c] == -1)
  {
    p4est_topidx_t tmp_tree_idx[P4EST_DIM - 1];
    nb_tree_idx = tmp_tree_idx[0] = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_p00]; // p00
    if(tmp_tree_idx[0] == tree_idx)        return;
    nb_tree_idx = tmp_tree_idx[1] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_00m]; // 00m
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) return;

    i_nb = 0;
    k_nb = P4EST_ROOT_LEN - size;
  }
  // 2) 0mm
  else if (quad->y == 0                     && cj[dir_c] == -1 &&
           quad->z == 0                     && ck[dir_c] == -1)
  {
    p4est_topidx_t tmp_tree_idx[P4EST_DIM - 1];
    nb_tree_idx = tmp_tree_idx[0] = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_0m0]; // 0m0
    if(tmp_tree_idx[0] == tree_idx)        return;
    nb_tree_idx = tmp_tree_idx[1] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_00m]; // 00m
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) return;

    j_nb = P4EST_ROOT_LEN - size;
    k_nb = P4EST_ROOT_LEN - size;
  }
  // 3) 0pm
  else if (quad->y == P4EST_ROOT_LEN - size && cj[dir_c] ==  1 &&
           quad->z == 0                     && ck[dir_c] == -1)
  {
    p4est_topidx_t tmp_tree_idx[P4EST_DIM - 1];
    nb_tree_idx = tmp_tree_idx[0] = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_0p0]; // 0p0
    if(tmp_tree_idx[0] == tree_idx)        return;
    nb_tree_idx = tmp_tree_idx[1] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_00m]; // 00m
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) return;

    j_nb = 0;
    k_nb = P4EST_ROOT_LEN - size;
  }
  // 4) mm0
  else if (quad->x == 0                     && ci[dir_c] == -1 &&
           quad->y == 0                     && cj[dir_c] == -1)
  {
    p4est_topidx_t tmp_tree_idx[P4EST_DIM - 1];
    nb_tree_idx = tmp_tree_idx[0] = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_m00]; // m00
    if(tmp_tree_idx[0] == tree_idx)        return;
    nb_tree_idx = tmp_tree_idx[1] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0m0]; // 0m0
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) return;

    i_nb = P4EST_ROOT_LEN - size;
    j_nb = P4EST_ROOT_LEN - size;
  }
  // 5) pm0
  else if (quad->x == P4EST_ROOT_LEN - size && ci[dir_c] ==  1 &&
           quad->y == 0                     && cj[dir_c] == -1)
  {
    p4est_topidx_t tmp_tree_idx[P4EST_DIM - 1];
    nb_tree_idx = tmp_tree_idx[0] = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_p00]; // p00
    if(tmp_tree_idx[0] == tree_idx)        return;
    nb_tree_idx = tmp_tree_idx[1] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0m0]; // 0m0
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) return;

    i_nb = 0;
    j_nb = P4EST_ROOT_LEN - size;
  }
  // 6) mp0
  else if (quad->x == 0                     && ci[dir_c] == -1 &&
           quad->y == P4EST_ROOT_LEN - size && cj[dir_c] ==  1)
  {
    p4est_topidx_t tmp_tree_idx[P4EST_DIM - 1];
    nb_tree_idx = tmp_tree_idx[0] = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_m00]; // m00
    if(tmp_tree_idx[0] == tree_idx)        return;
    nb_tree_idx = tmp_tree_idx[1] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0p0]; // 0p0
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) return;

    i_nb = P4EST_ROOT_LEN - size;
    j_nb = 0;
  }
  // 7) pp0
  else if (quad->x == P4EST_ROOT_LEN - size && ci[dir_c] ==  1 &&
           quad->y == P4EST_ROOT_LEN - size && cj[dir_c] ==  1)
  {
    p4est_topidx_t tmp_tree_idx[P4EST_DIM - 1];
    nb_tree_idx = tmp_tree_idx[0] = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_p00]; // p00
    if(tmp_tree_idx[0] == tree_idx)        return;
    nb_tree_idx = tmp_tree_idx[1] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0p0]; // 0p0
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) return;

    i_nb = 0;
    j_nb = 0;
  }
  // 8) m0p
  else if (quad->x == 0                     && ci[dir_c] == -1 &&
           quad->z == P4EST_ROOT_LEN - size && ck[dir_c] ==  1)
  {
    p4est_topidx_t tmp_tree_idx[P4EST_DIM - 1];
    nb_tree_idx = tmp_tree_idx[0] = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_m00]; // m00
    if(tmp_tree_idx[0] == tree_idx)        return;
    nb_tree_idx = tmp_tree_idx[1] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_00p]; // 00p
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) return;

    i_nb = P4EST_ROOT_LEN - size;
    k_nb = 0;
  }
  // 9) p0p
  else if (quad->x == P4EST_ROOT_LEN - size && ci[dir_c] ==  1 &&
           quad->z == P4EST_ROOT_LEN - size && ck[dir_c] ==  1)
  {
    p4est_topidx_t tmp_tree_idx[P4EST_DIM - 1];
    nb_tree_idx = tmp_tree_idx[0] = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_p00]; // p00
    if(tmp_tree_idx[0] == tree_idx)        return;
    nb_tree_idx = tmp_tree_idx[1] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_00p]; // 00p
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) return;

    i_nb = 0;
    k_nb = 0;
  }
  // 10) 0mp
  else if (quad->y == 0                     && cj[dir_c] == -1 &&
           quad->z == P4EST_ROOT_LEN - size && ck[dir_c] ==  1)
  {
    p4est_topidx_t tmp_tree_idx[P4EST_DIM - 1];
    nb_tree_idx = tmp_tree_idx[0] = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_0m0]; // 0m0
    if(tmp_tree_idx[0] == tree_idx)        return;
    nb_tree_idx = tmp_tree_idx[1] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_00p]; // 00p
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) return;

    j_nb = P4EST_ROOT_LEN - size;
    k_nb = 0;
  }
  // 11) 0pp
  else if (quad->y == P4EST_ROOT_LEN - size && cj[dir_c] ==  1 &&
           quad->z == P4EST_ROOT_LEN - size && ck[dir_c] ==  1)
  {
    p4est_topidx_t tmp_tree_idx[P4EST_DIM - 1];
    nb_tree_idx = tmp_tree_idx[0] = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_0p0]; // 0p0
    if(tmp_tree_idx[0] == tree_idx)        return;
    nb_tree_idx = tmp_tree_idx[1] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_00p]; // 00p
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) return;

    j_nb = 0;
    k_nb = 0;
  }
#endif

  /* search in the face directions */
  // 0) m00
  else if(quad->x == 0 && ci[dir_c] == -1)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_m00];
    if(nb_tree_idx == tree_idx) return;
    i_nb = P4EST_ROOT_LEN - size;
  }
  // 1) p00
  else if(quad->x+size == P4EST_ROOT_LEN && ci[dir_c] == 1)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_p00];
    if(nb_tree_idx == tree_idx) return;
    i_nb = 0;
  }
  // 2) 0m0
  else if(quad->y == 0 && cj[dir_c] == -1)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_0m0];
    if(nb_tree_idx == tree_idx) return;
    j_nb = P4EST_ROOT_LEN - size;
  }
  // 3) 0p0
  else if(quad->y+size == P4EST_ROOT_LEN && cj[dir_c] == 1)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_0p0];
    if(nb_tree_idx == tree_idx) return;
    j_nb = 0;
  }
#ifdef P4_TO_P8
  // 4) 00m
  else if(quad->z == 0 && ck[dir_c] == -1)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_00m];
    if(nb_tree_idx == tree_idx) return;
    k_nb = P4EST_ROOT_LEN - size;
  }
  // 5) 00p
  else if(quad->z+size == P4EST_ROOT_LEN && ck[dir_c] == 1)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_00p];
    if(nb_tree_idx == tree_idx) return;
    k_nb = 0;
  }
#endif



  /* find the constructed neighbor cell of the same size */
  int ind = 0;
  while( hierarchy->trees[nb_tree_idx][ind].level != quad->level && hierarchy->trees[nb_tree_idx][ind].child != CELL_LEAF )
  {
    p4est_qcoord_t half_size = P4EST_QUADRANT_LEN(hierarchy->trees[nb_tree_idx][ind].level) / 2;
    bool i_search = ( i_nb >= hierarchy->trees[nb_tree_idx][ind].imin + half_size );
    bool j_search = ( j_nb >= hierarchy->trees[nb_tree_idx][ind].jmin + half_size );
#ifdef P4_TO_P8
    bool k_search = ( k_nb >= hierarchy->trees[nb_tree_idx][ind].kmin + half_size );
#endif
#ifdef P4_TO_P8
    ind = hierarchy->trees[nb_tree_idx][ind].child + 4*k_search + 2*j_search + i_search;
#else
    ind = hierarchy->trees[nb_tree_idx][ind].child + 2*j_search + i_search;
#endif
  }

  /* now find the children of this constructed cell in the desired direction and add them to the list */
  find_corner_neighbor_cells_of_cell_recursive( nb_tree_idx, ind, dir_c );
  corners.offsets[q*P4EST_CHILDREN + dir_c + 1] = corners.neighbors.size();
}

void my_p4est_cell_neighbors_t::find_corner_neighbor_cells_of_cell_recursive(p4est_topidx_t tr, int ind, int dir_c )
{

  if (hierarchy->trees[tr][ind].child == CELL_LEAF)
  {
    if (hierarchy->trees[tr][ind].quad == NOT_A_P4EST_QUADRANT) return;

    quad_info_t qinfo;
    qinfo.locidx = hierarchy->trees[tr][ind].quad;
    qinfo.tree_idx = tr;

    const p4est_quadrant_t *quad = NULL;

    if (qinfo.locidx < p4est->local_num_quadrants) {// local quadrant
      p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tr);
      quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, qinfo.locidx - tree->quadrants_offset);
      qinfo.gloidx = p4est->global_first_quadrant[p4est->mpirank] + qinfo.locidx;
    } else {
      quad = (const p4est_quadrant_t*)sc_array_index(&ghost->ghosts, qinfo.locidx - p4est->local_num_quadrants);
      qinfo.gloidx = p4est->global_first_quadrant[hierarchy->trees[tr][ind].owner_rank] + quad->p.piggy3.local_num;
    }
    qinfo.quad = quad;

    corners.neighbors.push_back(qinfo);
    return;
  }

  switch(dir_c)
  {
  case dir::v_mmm:
#ifdef P4_TO_P8
    find_corner_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_ppp, dir_c);
#else
    find_corner_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_ppm, dir_c);
#endif
    break;
  case dir::v_pmm:
#ifdef P4_TO_P8
    find_corner_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mpp, dir_c);
#else
    find_corner_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mpm, dir_c);
#endif
    break;
  case dir::v_mpm:
#ifdef P4_TO_P8
    find_corner_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_pmp, dir_c);
#else
    find_corner_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_pmm, dir_c);
#endif
    break;
  case dir::v_ppm:
#ifdef P4_TO_P8
    find_corner_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mmp, dir_c);
#else
    find_corner_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mmm, dir_c);
#endif
    break;
#ifdef P4_TO_P8
  case dir::v_mmp:
    find_corner_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_ppm, dir_c);
    break;
  case dir::v_pmp:
    find_corner_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mpm, dir_c);
    break;
  case dir::v_mpp:
    find_corner_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_pmm, dir_c);
    break;
  case dir::v_ppp:
    find_corner_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mmm, dir_c);
    break;
#endif
  }
}

#ifdef P4_TO_P8
void my_p4est_cell_neighbors_t::find_edge_neighbor_cells_of_cell(const p4est_quadrant_t* quad, p4est_locidx_t q, p4est_topidx_t tree_idx, int dir_e )
{
  edges.offsets[q*P8EST_EDGES + dir_e + 1] = edges.neighbors.size();

  const static char ci [] = {-1,  1,  0,  0, -1,  1, -1,  1, -1,  1,  0,  0};
  const static char cj [] = { 0,  0, -1,  1, -1, -1,  1,  1,  0,  0, -1,  1};
  const static char ck [] = {-1, -1, -1, -1,  0,  0,  0,  0,  1,  1,  1,  1};

  /* construct the coordinate of the neighbor cell of the same size in the given direction */
  p4est_qcoord_t size = P4EST_QUADRANT_LEN(quad->level);

  p4est_topidx_t nb_tree_idx = tree_idx;
  p4est_qcoord_t i_nb = quad->x + ci[dir_e]*size;
  p4est_qcoord_t j_nb = quad->y + cj[dir_e]*size;
  p4est_qcoord_t k_nb = quad->z + ck[dir_e]*size;

  /* check if quadrant is on a boundary. we need to check 18 cases in 3D
   * 12 edges
   * 6 faces
   */
  /* search in the edge directions */
  // 0) m0m
  if (quad->x == 0                     && ci[dir_e] == -1 &&
      quad->z == 0                     && ck[dir_e] == -1)
  {
    p4est_topidx_t tmp_tree_idx[P4EST_DIM - 1];
    nb_tree_idx = tmp_tree_idx[0] = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_m00]; // m00
    if(tmp_tree_idx[0] == tree_idx)        return;
    nb_tree_idx = tmp_tree_idx[1] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_00m]; // 00m
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) return;

    i_nb = P4EST_ROOT_LEN - size;
    k_nb = P4EST_ROOT_LEN - size;
  }
  // 1) p0m
  else if (quad->x == P4EST_ROOT_LEN - size && ci[dir_e] ==  1 &&
           quad->z == 0                     && ck[dir_e] == -1)
  {
    p4est_topidx_t tmp_tree_idx[P4EST_DIM - 1];
    nb_tree_idx = tmp_tree_idx[0] = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_p00]; // p00
    if(tmp_tree_idx[0] == tree_idx)        return;
    nb_tree_idx = tmp_tree_idx[1] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_00m]; // 00m
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) return;

    i_nb = 0;
    k_nb = P4EST_ROOT_LEN - size;
  }
  // 2) 0mm
  else if (quad->y == 0                     && cj[dir_e] == -1 &&
           quad->z == 0                     && ck[dir_e] == -1)
  {
    p4est_topidx_t tmp_tree_idx[P4EST_DIM - 1];
    nb_tree_idx = tmp_tree_idx[0] = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_0m0]; // 0m0
    if(tmp_tree_idx[0] == tree_idx)        return;
    nb_tree_idx = tmp_tree_idx[1] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_00m]; // 00m
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) return;

    j_nb = P4EST_ROOT_LEN - size;
    k_nb = P4EST_ROOT_LEN - size;
  }
  // 3) 0pm
  else if (quad->y == P4EST_ROOT_LEN - size && cj[dir_e] ==  1 &&
           quad->z == 0                     && ck[dir_e] == -1)
  {
    p4est_topidx_t tmp_tree_idx[P4EST_DIM - 1];
    nb_tree_idx = tmp_tree_idx[0] = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_0p0]; // 0p0
    if(tmp_tree_idx[0] == tree_idx)        return;
    nb_tree_idx = tmp_tree_idx[1] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_00m]; // 00m
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) return;

    j_nb = 0;
    k_nb = P4EST_ROOT_LEN - size;
  }
  // 4) mm0
  else if (quad->x == 0                     && ci[dir_e] == -1 &&
           quad->y == 0                     && cj[dir_e] == -1)
  {
    p4est_topidx_t tmp_tree_idx[P4EST_DIM - 1];
    nb_tree_idx = tmp_tree_idx[0] = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_m00]; // m00
    if(tmp_tree_idx[0] == tree_idx)        return;
    nb_tree_idx = tmp_tree_idx[1] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0m0]; // 0m0
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) return;

    i_nb = P4EST_ROOT_LEN - size;
    j_nb = P4EST_ROOT_LEN - size;
  }
  // 5) pm0
  else if (quad->x == P4EST_ROOT_LEN - size && ci[dir_e] ==  1 &&
           quad->y == 0                     && cj[dir_e] == -1)
  {
    p4est_topidx_t tmp_tree_idx[P4EST_DIM - 1];
    nb_tree_idx = tmp_tree_idx[0] = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_p00]; // p00
    if(tmp_tree_idx[0] == tree_idx)        return;
    nb_tree_idx = tmp_tree_idx[1] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0m0]; // 0m0
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) return;

    i_nb = 0;
    j_nb = P4EST_ROOT_LEN - size;
  }
  // 6) mp0
  else if (quad->x == 0                     && ci[dir_e] == -1 &&
           quad->y == P4EST_ROOT_LEN - size && cj[dir_e] ==  1)
  {
    p4est_topidx_t tmp_tree_idx[P4EST_DIM - 1];
    nb_tree_idx = tmp_tree_idx[0] = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_m00]; // m00
    if(tmp_tree_idx[0] == tree_idx)        return;
    nb_tree_idx = tmp_tree_idx[1] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0p0]; // 0p0
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) return;

    i_nb = P4EST_ROOT_LEN - size;
    j_nb = 0;
  }
  // 7) pp0
  else if (quad->x == P4EST_ROOT_LEN - size && ci[dir_e] ==  1 &&
           quad->y == P4EST_ROOT_LEN - size && cj[dir_e] ==  1)
  {
    p4est_topidx_t tmp_tree_idx[P4EST_DIM - 1];
    nb_tree_idx = tmp_tree_idx[0] = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_p00]; // p00
    if(tmp_tree_idx[0] == tree_idx)        return;
    nb_tree_idx = tmp_tree_idx[1] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_0p0]; // 0p0
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) return;

    i_nb = 0;
    j_nb = 0;
  }
  // 8) m0p
  else if (quad->x == 0                     && ci[dir_e] == -1 &&
           quad->z == P4EST_ROOT_LEN - size && ck[dir_e] ==  1)
  {
    p4est_topidx_t tmp_tree_idx[P4EST_DIM - 1];
    nb_tree_idx = tmp_tree_idx[0] = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_m00]; // m00
    if(tmp_tree_idx[0] == tree_idx)        return;
    nb_tree_idx = tmp_tree_idx[1] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_00p]; // 00p
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) return;

    i_nb = P4EST_ROOT_LEN - size;
    k_nb = 0;
  }
  // 9) p0p
  else if (quad->x == P4EST_ROOT_LEN - size && ci[dir_e] ==  1 &&
           quad->z == P4EST_ROOT_LEN - size && ck[dir_e] ==  1)
  {
    p4est_topidx_t tmp_tree_idx[P4EST_DIM - 1];
    nb_tree_idx = tmp_tree_idx[0] = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_p00]; // p00
    if(tmp_tree_idx[0] == tree_idx)        return;
    nb_tree_idx = tmp_tree_idx[1] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_00p]; // 00p
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) return;

    i_nb = 0;
    k_nb = 0;
  }
  // 10) 0mp
  else if (quad->y == 0                     && cj[dir_e] == -1 &&
           quad->z == P4EST_ROOT_LEN - size && ck[dir_e] ==  1)
  {
    p4est_topidx_t tmp_tree_idx[P4EST_DIM - 1];
    nb_tree_idx = tmp_tree_idx[0] = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_0m0]; // 0m0
    if(tmp_tree_idx[0] == tree_idx)        return;
    nb_tree_idx = tmp_tree_idx[1] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_00p]; // 00p
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) return;

    j_nb = P4EST_ROOT_LEN - size;
    k_nb = 0;
  }
  // 11) 0pp
  else if (quad->y == P4EST_ROOT_LEN - size && cj[dir_e] ==  1 &&
           quad->z == P4EST_ROOT_LEN - size && ck[dir_e] ==  1)
  {
    p4est_topidx_t tmp_tree_idx[P4EST_DIM - 1];
    nb_tree_idx = tmp_tree_idx[0] = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx        + dir::f_0p0]; // 0p0
    if(tmp_tree_idx[0] == tree_idx)        return;
    nb_tree_idx = tmp_tree_idx[1] = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_tree_idx[0] + dir::f_00p]; // 00p
    if(tmp_tree_idx[1] == tmp_tree_idx[0]) return;

    j_nb = 0;
    k_nb = 0;
  }

  /* search in the face directions */
  // 0) m00
  else if(quad->x == 0 && ci[dir_e] == -1)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_m00];
    if(nb_tree_idx == tree_idx) return;
    i_nb = P4EST_ROOT_LEN - size;
  }
  // 1) p00
  else if(quad->x+size == P4EST_ROOT_LEN && ci[dir_e] == 1)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_p00];
    if(nb_tree_idx == tree_idx) return;
    i_nb = 0;
  }
  // 2) 0m0
  else if(quad->y == 0 && cj[dir_e] == -1)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_0m0];
    if(nb_tree_idx == tree_idx) return;
    j_nb = P4EST_ROOT_LEN - size;
  }
  // 3) 0p0
  else if(quad->y+size == P4EST_ROOT_LEN && cj[dir_e] == 1)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_0p0];
    if(nb_tree_idx == tree_idx) return;
    j_nb = 0;
  }
  // 4) 00m
  else if(quad->z == 0 && ck[dir_e] == -1)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_00m];
    if(nb_tree_idx == tree_idx) return;
    k_nb = P4EST_ROOT_LEN - size;
  }
  // 5) 00p
  else if(quad->z+size == P4EST_ROOT_LEN && ck[dir_e] == 1)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_00p];
    if(nb_tree_idx == tree_idx) return;
    k_nb = 0;
  }


  /* find the constructed neighbor cell of the same size */
  int ind = 0;
  while( hierarchy->trees[nb_tree_idx][ind].level != quad->level && hierarchy->trees[nb_tree_idx][ind].child != CELL_LEAF )
  {
    p4est_qcoord_t half_size = P4EST_QUADRANT_LEN(hierarchy->trees[nb_tree_idx][ind].level) / 2;
    bool i_search = ( i_nb >= hierarchy->trees[nb_tree_idx][ind].imin + half_size );
    bool j_search = ( j_nb >= hierarchy->trees[nb_tree_idx][ind].jmin + half_size );
    bool k_search = ( k_nb >= hierarchy->trees[nb_tree_idx][ind].kmin + half_size );
    ind = hierarchy->trees[nb_tree_idx][ind].child + 4*k_search + 2*j_search + i_search;
  }

  /* now find the children of this constructed cell in the desired direction and add them to the list */
  find_edge_neighbor_cells_of_cell_recursive( nb_tree_idx, ind, dir_e );
  edges.offsets[q*P8EST_EDGES + dir_e + 1] = edges.neighbors.size();
}

void my_p4est_cell_neighbors_t::find_edge_neighbor_cells_of_cell_recursive(p4est_topidx_t tr, int ind, int dir_e )
{

  if (hierarchy->trees[tr][ind].child == CELL_LEAF)
  {
    if (hierarchy->trees[tr][ind].quad == NOT_A_P4EST_QUADRANT) return;

    quad_info_t qinfo;
    qinfo.locidx = hierarchy->trees[tr][ind].quad;
    qinfo.tree_idx = tr;

    const p4est_quadrant_t *quad = NULL;

    if (qinfo.locidx < p4est->local_num_quadrants) {// local quadrant
      p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tr);
      quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, qinfo.locidx - tree->quadrants_offset);
      qinfo.gloidx = p4est->global_first_quadrant[p4est->mpirank] + qinfo.locidx;
    } else {
      quad = (const p4est_quadrant_t*)sc_array_index(&ghost->ghosts, qinfo.locidx - p4est->local_num_quadrants);
      qinfo.gloidx = p4est->global_first_quadrant[hierarchy->trees[tr][ind].owner_rank] + quad->p.piggy3.local_num;
    }
    qinfo.quad = quad;

    edges.neighbors.push_back(qinfo);
    return;
  }

  switch(dir_e)
  {
  case dir::e_m0m:
    find_edge_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_pmp, dir_e);
    find_edge_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_ppp, dir_e);
    break;
  case dir::e_p0m:
    find_edge_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mmp, dir_e);
    find_edge_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mpp, dir_e);
    break;
  case dir::e_0mm:
    find_edge_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mpp, dir_e);
    find_edge_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_ppp, dir_e);
    break;
  case dir::e_0pm:
    find_edge_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mmp, dir_e);
    find_edge_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_pmp, dir_e);
    break;
  case dir::e_mm0:
    find_edge_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_ppm, dir_e);
    find_edge_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_ppp, dir_e);
    break;
  case dir::e_pm0:
    find_edge_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mpm, dir_e);
    find_edge_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mpp, dir_e);
    break;
  case dir::e_mp0:
    find_edge_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_pmm, dir_e);
    find_edge_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_pmp, dir_e);
    break;
  case dir::e_pp0:
    find_edge_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mmm, dir_e);
    find_edge_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mmp, dir_e);
    break;
  case dir::e_m0p:
    find_edge_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_pmm, dir_e);
    find_edge_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_ppm, dir_e);
    break;
  case dir::e_p0p:
    find_edge_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mmm, dir_e);
    find_edge_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mpm, dir_e);
    break;
  case dir::e_0mp:
    find_edge_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mpm, dir_e);
    find_edge_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_ppm, dir_e);
    break;
  case dir::e_0pp:
    find_edge_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mmm, dir_e);
    find_edge_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_pmm, dir_e);
    break;
  }
}
#endif

void my_p4est_cell_neighbors_t::print_debug(p4est_locidx_t q, FILE *stream)
{
  fprintf(stream, " ---------- quad = %d ---------- \n", q);
  fprintf(stream, " faces: ");
  for (int f = 0; f<P4EST_FACES; ++f){
    fprintf(stream, "f_%d = ",f);
    for (p4est_locidx_t qi = faces.offsets[q*P4EST_FACES + f]; qi < faces.offsets[q*P4EST_FACES + f + 1]; ++qi)
      fprintf(stream, "%d ", faces.neighbors[qi].locidx);
    fprintf(stream, "\n");
  }
  fprintf(stream, "\n");

#ifdef P4_TO_P8
  fprintf(stream, " edges: ");
  for (int e = 0; e<P8EST_EDGES; ++e){
    fprintf(stream, "e_%d = ",e);
    for (p4est_locidx_t qi = edges.offsets[q*P8EST_EDGES + e]; qi < edges.offsets[q*P8EST_EDGES + e + 1]; ++qi)
      fprintf(stream, "%d ", edges.neighbors[qi].locidx);
    fprintf(stream, "\n");
  }
  fprintf(stream, "\n");
#endif

  fprintf(stream, " corners: ");
  for (int c = 0; c<P4EST_CHILDREN; ++c){
    fprintf(stream, "c_%d = ",c);
    for (p4est_locidx_t qi = corners.offsets[q*P4EST_CHILDREN + c]; qi < corners.offsets[q*P4EST_CHILDREN + c + 1]; ++qi)
      fprintf(stream, "%d ", corners.neighbors[qi].locidx);
    fprintf(stream, "\n");
  }
  fprintf(stream, "\n");
}

void my_p4est_cell_neighbors_t::write_cell_neighbors_vtk(p4est_locidx_t qu, p4est_topidx_t tr, const char *filename)
{
  p4est_connectivity_t *conn = p4est->connectivity;
  p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tr);
  p4est_locidx_t q = qu + tree->quadrants_offset;

  char vtkname[1024];
  sprintf(vtkname, "%s_%04d.%d.vtk", filename, p4est->mpirank, q);

  FILE *vtk = fopen(vtkname, "w");

  fprintf(vtk, "# vtk DataFile Version 2.0 \n");
  fprintf(vtk, "Triangulation \n");
  fprintf(vtk, "ASCII \n");
  fprintf(vtk, "DATASET UNSTRUCTURED_GRID \n");

  /* get neighboring cells */
  const quad_info_t *f_begin = this->face_begin(q, 0);
  const quad_info_t *f_end   = this->face_end(q, P4EST_FACES - 1); size_t f_size = f_end - f_begin;

#ifdef P4_TO_P8
  const quad_info_t *e_begin = this->edge_begin(q, 0);
  const quad_info_t *e_end   = this->edge_end(q, P8EST_EDGES - 1); size_t e_size = e_end - e_begin;
#endif

  const quad_info_t *c_begin = this->corner_begin(q, 0);
  const quad_info_t *c_end   = this->corner_end(q, P4EST_CHILDREN - 1); size_t c_size = c_end - c_begin;


  /* total number of nodes */
#ifdef P4_TO_P8
  size_t num_ngbds = f_size + e_size + c_size + 1;
#else
  size_t num_ngbds = f_size + c_size + 1;
#endif

  fprintf(vtk, "POINTS %ld double \n", P4EST_CHILDREN*num_ngbds);

  /* central cell */
  {
    const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, qu);
    p4est_topidx_t v_mm = conn->tree_to_vertex[P4EST_CHILDREN*tr];
    double tree_xmin = conn->vertices[3*v_mm + 0];
    double tree_ymin = conn->vertices[3*v_mm + 1];
#ifdef P4_TO_P8
    double tree_zmin = conn->vertices[3*v_mm + 2];
#endif

    double qh   = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
    double xmin = quad_x_fr_i(quad) + tree_xmin;
    double ymin = quad_y_fr_j(quad) + tree_ymin;
#ifdef P4_TO_P8
    double zmin = quad_z_fr_k(quad) + tree_zmin;
#endif

#ifdef P4_TO_P8
    for (short ck = 0; ck<2; ck++)
#endif
      for (short cj = 0; cj<2; cj++)
        for (short ci = 0; ci<2; ci++)
#ifdef P4_TO_P8
          fprintf(vtk, "%lf %lf %lf\n", xmin + ci*qh, ymin + cj*qh, zmin + ck*qh);
#else
          fprintf(vtk, "%lf %lf 0\n", xmin + ci*qh, ymin + cj*qh);
#endif
  }

  /* neighboring cells */
  // faces
  for (const quad_info_t *it = f_begin; it != f_end; ++it){
    p4est_topidx_t v_mm = conn->tree_to_vertex[P4EST_CHILDREN*it->tree_idx];
    double tree_xmin = conn->vertices[3*v_mm + 0];
    double tree_ymin = conn->vertices[3*v_mm + 1];
#ifdef P4_TO_P8
    double tree_zmin = conn->vertices[3*v_mm + 2];
#endif

    double qh   = (double)P4EST_QUADRANT_LEN(it->quad->level)/(double)P4EST_ROOT_LEN;
    double xmin = quad_x_fr_i(it->quad) + tree_xmin;
    double ymin = quad_y_fr_j(it->quad) + tree_ymin;
#ifdef P4_TO_P8
    double zmin = quad_z_fr_k(it->quad) + tree_zmin;
#endif

#ifdef P4_TO_P8
    for (short ck = 0; ck<2; ck++)
#endif
      for (short cj = 0; cj<2; cj++)
        for (short ci = 0; ci<2; ci++)
#ifdef P4_TO_P8
          fprintf(vtk, "%lf %lf %lf\n", xmin + ci*qh, ymin + cj*qh, zmin + ck*qh);
#else
          fprintf(vtk, "%lf %lf 0\n", xmin + ci*qh, ymin + cj*qh);
#endif
  }

#ifdef P4_TO_P8
  // edges
  for (const quad_info_t *it = e_begin; it != e_end; ++it){
    p4est_topidx_t v_mm = conn->tree_to_vertex[P4EST_CHILDREN*it->tree_idx];
    double tree_xmin = conn->vertices[3*v_mm + 0];
    double tree_ymin = conn->vertices[3*v_mm + 1];
    double tree_zmin = conn->vertices[3*v_mm + 2];

    double qh   = (double)P4EST_QUADRANT_LEN(it->quad->level)/(double)P4EST_ROOT_LEN;
    double xmin = quad_x_fr_i(it->quad) + tree_xmin;
    double ymin = quad_y_fr_j(it->quad) + tree_ymin;
    double zmin = quad_z_fr_k(it->quad) + tree_zmin;

    for (short ck = 0; ck<2; ck++)
      for (short cj = 0; cj<2; cj++)
        for (short ci = 0; ci<2; ci++)
          fprintf(vtk, "%lf %lf %lf\n", xmin + ci*qh, ymin + cj*qh, zmin + ck*qh);
  }
#endif

  // corners
  for (const quad_info_t *it = c_begin; it != c_end; ++it){
    p4est_topidx_t v_mm = conn->tree_to_vertex[P4EST_CHILDREN*it->tree_idx];
    double tree_xmin = conn->vertices[3*v_mm + 0];
    double tree_ymin = conn->vertices[3*v_mm + 1];
#ifdef P4_TO_P8
    double tree_zmin = conn->vertices[3*v_mm + 2];
#endif

    double qh   = (double)P4EST_QUADRANT_LEN(it->quad->level)/(double)P4EST_ROOT_LEN;
    double xmin = quad_x_fr_i(it->quad) + tree_xmin;
    double ymin = quad_y_fr_j(it->quad) + tree_ymin;
#ifdef P4_TO_P8
    double zmin = quad_z_fr_k(it->quad) + tree_zmin;
#endif

#ifdef P4_TO_P8
    for (short ck = 0; ck<2; ck++)
#endif
      for (short cj = 0; cj<2; cj++)
        for (short ci = 0; ci<2; ci++)
#ifdef P4_TO_P8
          fprintf(vtk, "%lf %lf %lf\n", xmin + ci*qh, ymin + cj*qh, zmin + ck*qh);
#else
          fprintf(vtk, "%lf %lf 0\n", xmin + ci*qh, ymin + cj*qh);
#endif
  }

  /* write the connectivity information (a.k.a. elements */
  fprintf(vtk, "CELLS %ld %ld \n", num_ngbds, (1+P4EST_CHILDREN)*num_ngbds);

  for (size_t i=0; i<num_ngbds; ++i)
  {
    fprintf(vtk, "%d ", P4EST_CHILDREN);
    for (short j = 0; j<P4EST_CHILDREN; j++)
      fprintf(vtk, "%ld ", i*P4EST_CHILDREN + j);
    fprintf(vtk, "\n");
  }

  fprintf(vtk, "CELL_TYPES %ld\n", num_ngbds);
  for (size_t i=0; i<num_ngbds; ++i)
    fprintf(vtk, "%d\n", P4EST_VTK_CELL_TYPE);

  fprintf(vtk, "CELL_DATA %ld\n", num_ngbds);
  fprintf(vtk, "SCALARS cell double 1\n");
  fprintf(vtk, "LOOKUP_TABLE default\n");
  fprintf(vtk, "0\n");
  for (size_t i=1; i<num_ngbds; ++i)
    fprintf(vtk, "1\n");

  fclose(vtk);
}

#ifdef P4_TO_P8
void my_p4est_cell_neighbors_t::write_cell_triangulation_vtk_00p(p4est_locidx_t qu, p4est_topidx_t tr, const char *filename)
{
  p4est_connectivity_t *conn = p4est->connectivity;
  p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tr);
  const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, qu);
  p4est_locidx_t q = qu + tree->quadrants_offset;

  char vtkname[1024];
  sprintf(vtkname, "%s_%04d.%d.vtk", filename, p4est->mpirank, q);
  FILE *vtk = fopen(vtkname, "w");

  fprintf(vtk, "# vtk DataFile Version 2.0 \n");
  fprintf(vtk, "Triangulation \n");
  fprintf(vtk, "ASCII \n");
  fprintf(vtk, "DATASET UNSTRUCTURED_GRID \n");

  const quad_info_t *cells[P4EST_FACES + 1];
  for (short i = 0; i<P4EST_FACES; i++)
    cells[i] = this->face_begin(q, i);
  cells[P4EST_FACES] = this->face_end(q, P4EST_FACES - 1);

  const quad_info_t *begin_m00 = cells[dir::f_m00  ];
  const quad_info_t *end_m00   = cells[dir::f_m00+1]; size_t n_m00 = end_m00 - begin_m00;
  const quad_info_t *begin_p00 = cells[dir::f_p00  ];
  const quad_info_t *end_p00   = cells[dir::f_p00+1]; size_t n_p00 = end_p00 - begin_p00;
  const quad_info_t *begin_0m0 = cells[dir::f_0m0  ];
  const quad_info_t *end_0m0   = cells[dir::f_0m0+1]; size_t n_0m0 = end_0m0 - begin_0m0;
  const quad_info_t *begin_0p0 = cells[dir::f_0p0  ];
  const quad_info_t *end_0p0   = cells[dir::f_0p0+1]; size_t n_0p0 = end_0p0 - begin_0p0;
  const quad_info_t *begin_00p = cells[dir::f_00p  ];
  const quad_info_t *end_00p   = cells[dir::f_00p+1]; size_t n_00p = end_00p - begin_00p;

  size_t num_points = cells[P4EST_FACES]-cells[0] + 1;
  fprintf(vtk, "POINTS %ld double\n", num_points);

  std::vector<p4est_locidx_t> elements;
  elements.reserve(4*num_points); // worst-case estimate

  std::map<p4est_locidx_t, int> map;

  int c = 0;
  /* write the center location of the central point */
  {
    double qh = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;

    p4est_topidx_t v_mm = conn->tree_to_vertex[P4EST_CHILDREN*tr];
    double tree_xmin = conn->vertices[3*v_mm + 0];
    double tree_ymin = conn->vertices[3*v_mm + 1];
    double tree_zmin = conn->vertices[3*v_mm + 2];

    double x  = quad_x_fr_i(quad) + 0.5*qh + tree_xmin;
    double y  = quad_y_fr_j(quad) + 0.5*qh + tree_ymin;
    double z  = quad_z_fr_k(quad) + 0.5*qh + tree_zmin;

    fprintf(vtk, "%lf %lf %lf\n", x, y, z);
    map.insert(std::make_pair(q,c++));
  }

  /* write center location and create mapping */
  for (const quad_info_t *it = cells[0]; it != cells[P4EST_FACES]; ++it)
  {
    double qh = (double)P4EST_QUADRANT_LEN(it->quad->level)/(double)P4EST_ROOT_LEN;

    p4est_topidx_t v_mm = conn->tree_to_vertex[P4EST_CHILDREN*it->tree_idx];
    double tree_xmin = conn->vertices[3*v_mm + 0];
    double tree_ymin = conn->vertices[3*v_mm + 1];
    double tree_zmin = conn->vertices[3*v_mm + 2];

    double x  = quad_x_fr_i(it->quad) + 0.5*qh + tree_xmin;
    double y  = quad_y_fr_j(it->quad) + 0.5*qh + tree_ymin;
    double z  = quad_z_fr_k(it->quad) + 0.5*qh + tree_zmin;

    fprintf(vtk, "%lf %lf %lf\n", x, y, z);
    map.insert(std::make_pair(it->locidx, c++));
  }

  /* loop over all cells and find neighbors in the p00, 0p0, and pp0 directions */
  for (const quad_info_t *it = begin_00p; it != end_00p; ++it)
  {
    std::vector<p4est_locidx_t> ng_p00;
    std::vector<p4est_locidx_t> ng_0p0;
    int8_t l_p0, l_0p;

    bool is_p00_boundary = it->quad->x + P4EST_QUADRANT_LEN(it->quad->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level);
    bool is_0p0_boundary = it->quad->y + P4EST_QUADRANT_LEN(it->quad->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level);

    /* p00 */
    if (!is_p00_boundary)
    {
      const quad_info_t *begin_ = this->face_begin(it->locidx, dir::f_p00);
      const quad_info_t *end_   = this->face_end(it->locidx, dir::f_p00);
      ng_p00.reserve(end_ - begin_);
      for (const quad_info_t *it_ = begin_; it_ != end_; ++it_)
        if (it_->quad->z == it->quad->z){
          ng_p00.push_back(it_->locidx);
          l_p0 = it_->quad->level;
        }
    }

    /* 0p0 */
    if (!is_0p0_boundary)
    {
      const quad_info_t *begin_ = this->face_begin(it->locidx, dir::f_0p0);
      const quad_info_t *end_   = this->face_end(it->locidx, dir::f_0p0);
      ng_0p0.reserve(end_ - begin_);
      for (const quad_info_t *it_ = begin_; it_ != end_; ++it_)
        if (it_->quad->z == it->quad->z){
          ng_0p0.push_back(it_->locidx);
          l_0p = it_->quad->level;
        }
    }

    /* pp0 */
    if (!is_p00_boundary && !is_0p0_boundary)
    {
      const quad_info_t *it_;
      if (l_p0 > l_0p)
        it_ = this->face_begin(ng_p00[ng_p00.size() - 1], dir::f_0p0);
      else
        it_ = this->face_begin(ng_0p0[ng_0p0.size() - 1], dir::f_p00);

      ng_p00.push_back(it_->locidx);
      ng_0p0.push_back(it_->locidx);
    }

    /* now that we have all cells, construct the elements */
    if (ng_p00.size() != 0)
      for (size_t i=0; i<ng_p00.size()-1; i++){
        elements.push_back(q);
        elements.push_back(it->locidx);
        elements.push_back(ng_p00[i]);
        elements.push_back(ng_p00[i+1]);
      }

    if (ng_0p0.size() != 0)
      for (size_t i=0; i<ng_0p0.size()-1; i++){
        elements.push_back(q);
        elements.push_back(it->locidx);
        elements.push_back(ng_0p0[i]);
        elements.push_back(ng_0p0[i+1]);
      }
  }

  /* now we do stiching! */
  /* 1 - across m00 direction */
  if (n_00p != 0) {
    {
      /* first locate top cells along the edge */
      std::vector<p4est_locidx_t> ng1;
      ng1.reserve(end_00p - begin_00p);
      for (const quad_info_t *it = begin_00p; it != end_00p; ++it)
        if (it->quad->x == quad->x){
          ng1.push_back(it->locidx);
        }

      std::vector<p4est_locidx_t> ng2;
      ng2.reserve(end_m00 - begin_m00);
      for (const quad_info_t *it = begin_m00; it != end_m00; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->quad->level) == quad->z + P4EST_QUADRANT_LEN(quad->level)){
          ng2.push_back(it->locidx);
        }

      /* now construct the triangulation */
      if (ng1.size() != 0 && ng2.size() != 0){
        if (ng1.size() > ng2.size())
          std::swap(ng1, ng2);

        // forward
        int s = (ng2.size()-1) / ng1.size();
        int r = (ng2.size()-1) % ng1.size();
        int c = 0;
        std::vector<int> ng2s; ng2s.reserve(ng1.size() - 1);
        if (ng2.size() > 1){
          for (int i=0; i<r; i++){
            for (int j = 0; j<s+1; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s+1;
            ng2s.push_back(c);
          }
          for (size_t i = r; i<ng1.size(); i++){
            for (int j = 0; j<s; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s;
            ng2s.push_back(c);
          }
        }
        //backward
        if (ng1.size() > 1)
          for (size_t i=0; i<ng1.size() - 1; i++){
            elements.push_back(q);
            elements.push_back(ng1[i]);
            elements.push_back(ng1[i+1]);
            elements.push_back(ng2[ng2s[i]]);
          }
      }
    }
    /* 2 - across p00 direction */
    {
      /* first locate top cells along the edge */
      std::vector<p4est_locidx_t> ng1;
      ng1.reserve(end_00p - begin_00p);
      for (const quad_info_t *it = begin_00p; it != end_00p; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->quad->level) == quad->x + P4EST_QUADRANT_LEN(quad->level)){
          ng1.push_back(it->locidx);
        }

      std::vector<p4est_locidx_t> ng2;
      ng2.reserve(end_p00 - begin_p00);
      for (const quad_info_t *it = begin_p00; it != end_p00; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->quad->level) == quad->z + P4EST_QUADRANT_LEN(quad->level)){
          ng2.push_back(it->locidx);
        }

      /* now construct the triangulation */
      if (ng1.size() != 0 && ng2.size() != 0){
        if (ng1.size() > ng2.size())
          std::swap(ng1, ng2);

        // forward
        int s = (ng2.size()-1) / ng1.size();
        int r = (ng2.size()-1) % ng1.size();
        int c = 0;
        std::vector<int> ng2s; ng2s.reserve(ng1.size() - 1);
        if (ng2.size() > 1){
          for (int i=0; i<r; i++){
            for (int j = 0; j<s+1; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s+1;
            ng2s.push_back(c);
          }
          for (size_t i = r; i<ng1.size(); i++){
            for (int j = 0; j<s; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s;
            ng2s.push_back(c);
          }
        }
        //backward
        if (ng1.size() > 1)
          for (size_t i=0; i<ng1.size() - 1; i++){
            elements.push_back(q);
            elements.push_back(ng1[i]);
            elements.push_back(ng1[i+1]);
            elements.push_back(ng2[ng2s[i]]);
          }
      }
    }
    /* 3 - across 0m0 direction */
    {
      /* first locate top cells along the edge */
      std::vector<p4est_locidx_t> ng1;
      ng1.reserve(end_00p - begin_00p);
      for (const quad_info_t *it = begin_00p; it != end_00p; ++it)
        if (it->quad->y == quad->y){
          ng1.push_back(it->locidx);
        }

      std::vector<p4est_locidx_t> ng2;
      ng2.reserve(end_0m0 - begin_0m0);
      for (const quad_info_t *it = begin_0m0; it != end_0m0; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->quad->level) == quad->z + P4EST_QUADRANT_LEN(quad->level)){
          ng2.push_back(it->locidx);
        }

      /* now construct the triangulation */
      if (ng1.size() != 0 && ng2.size() != 0){
        if (ng1.size() > ng2.size())
          std::swap(ng1, ng2);

        // forward
        int s = (ng2.size()-1) / ng1.size();
        int r = (ng2.size()-1) % ng1.size();
        int c = 0;
        std::vector<int> ng2s; ng2s.reserve(ng1.size() - 1);
        if (ng2.size() > 1){
          for (int i=0; i<r; i++){
            for (int j = 0; j<s+1; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s+1;
            ng2s.push_back(c);
          }
          for (size_t i = r; i<ng1.size(); i++){
            for (int j = 0; j<s; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s;
            ng2s.push_back(c);
          }
        }
        //backward
        if (ng1.size() > 1)
          for (size_t i=0; i<ng1.size() - 1; i++){
            elements.push_back(q);
            elements.push_back(ng1[i]);
            elements.push_back(ng1[i+1]);
            elements.push_back(ng2[ng2s[i]]);
          }
      }
    }
    /* 4 - across 0p0 direction */
    {
      /* first locate top cells along the edge */
      std::vector<p4est_locidx_t> ng1;
      ng1.reserve(end_00p - begin_00p);
      for (const quad_info_t *it = begin_00p; it != end_00p; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->quad->level) == quad->y + P4EST_QUADRANT_LEN(quad->level)){
          ng1.push_back(it->locidx);
        }

      std::vector<p4est_locidx_t> ng2;
      ng2.reserve(end_0p0 - begin_0p0);
      for (const quad_info_t *it = begin_0p0; it != end_0p0; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->quad->level) == quad->z + P4EST_QUADRANT_LEN(quad->level)){
          ng2.push_back(it->locidx);
        }

      /* now construct the triangulation */
      if (ng1.size() != 0 && ng2.size() != 0){
        if (ng1.size() > ng2.size())
          std::swap(ng1, ng2);

        // forward
        int s = (ng2.size()-1) / ng1.size();
        int r = (ng2.size()-1) % ng1.size();
        int c = 0;
        std::vector<int> ng2s; ng2s.reserve(ng1.size() - 1);
        if (ng2.size() > 1){
          for (int i=0; i<r; i++){
            for (int j = 0; j<s+1; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s+1;
            ng2s.push_back(c);
          }
          for (size_t i = r; i<ng1.size(); i++){
            for (int j = 0; j<s; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s;
            ng2s.push_back(c);
          }
        }
        //backward
        if (ng1.size() > 1)
          for (size_t i=0; i<ng1.size() - 1; i++){
            elements.push_back(q);
            elements.push_back(ng1[i]);
            elements.push_back(ng1[i+1]);
            elements.push_back(ng2[ng2s[i]]);
          }
      }
    }
    /* 5 - across mm0 */
    if (n_m00 != 0 && n_0m0 != 0)
    {
      elements.push_back(q);
      elements.push_back(begin_00p->locidx);

      for (const quad_info_t *it = begin_m00; it != end_m00; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->quad->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level)){
          elements.push_back(it->locidx);
          break;
        }

      for (const quad_info_t *it = begin_0m0; it != end_0m0; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->quad->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level)){
          elements.push_back(it->locidx);
          break;
        }
    }

    /* 6 - across pm0 */
    if (n_p00 != 0 && n_0m0 != 0)
    {
      elements.push_back(q);
      for (const quad_info_t *it = begin_00p; it != end_00p; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->quad->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level)){
          elements.push_back(it->locidx);
          break;
        }

      for (const quad_info_t *it = begin_p00; it != end_p00; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->quad->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level)){
          elements.push_back(it->locidx);
          break;
        }

      elements.push_back((end_0m0-1)->locidx);
    }
    /* 7 - across mp0 */
    if (n_m00 != 0 && n_0p0 != 0)
    {
      elements.push_back(q);
      for (const quad_info_t *it = begin_00p; it != end_00p; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->quad->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level)){
          elements.push_back(it->locidx);
          break;
        }

      elements.push_back((end_m00-1)->locidx);

      for (const quad_info_t *it = begin_0p0; it != end_0p0; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->quad->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level)){
          elements.push_back(it->locidx);
          break;
        }
    }
    /* 8 - across pp0 */
    if (n_p00 != 0 && n_0p0 != 0)
    {
      elements.push_back(q);
      elements.push_back((end_00p-1)->locidx);
      elements.push_back((end_p00-1)->locidx);
      elements.push_back((end_0p0-1)->locidx);
    }
  }
  /* write the connectivity information (a.k.a. elements) */
  size_t num_elements = elements.size()/4;
  fprintf(vtk, "CELLS %ld %ld \n", num_elements, (1+4)*num_elements);

  for (size_t i=0; i<num_elements; ++i)
  {
    fprintf(vtk, "%d ", 4);
    for (short j = 0; j<4; j++)
      fprintf(vtk, "%d ", map[elements[4*i + j]]);
    fprintf(vtk, "\n");
  }

  fprintf(vtk, "CELL_TYPES %ld\n", num_elements);
  for (size_t i=0; i<num_elements; ++i)
    fprintf(vtk, "%d\n", VTK_CELL_TYPE);
  fclose(vtk);
}

void my_p4est_cell_neighbors_t::write_cell_triangulation_vtk_00m(p4est_locidx_t qu, p4est_topidx_t tr, const char *filename)
{
  p4est_connectivity_t *conn = p4est->connectivity;
  p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tr);
  const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, qu);
  p4est_locidx_t q = qu + tree->quadrants_offset;

  char vtkname[1024];
  sprintf(vtkname, "%s_%04d.%d.vtk", filename, p4est->mpirank, q);
  FILE *vtk = fopen(vtkname, "w");

  fprintf(vtk, "# vtk DataFile Version 2.0 \n");
  fprintf(vtk, "Triangulation \n");
  fprintf(vtk, "ASCII \n");
  fprintf(vtk, "DATASET UNSTRUCTURED_GRID \n");

  const quad_info_t *cells[P4EST_FACES + 1];
  for (short i = 0; i<P4EST_FACES; i++)
    cells[i] = this->face_begin(q, i);
  cells[P4EST_FACES] = this->face_end(q, P4EST_FACES - 1);

  const quad_info_t *begin_m00 = cells[dir::f_m00  ];
  const quad_info_t *end_m00   = cells[dir::f_m00+1]; size_t n_m00 = end_m00 - begin_m00;
  const quad_info_t *begin_p00 = cells[dir::f_p00  ];
  const quad_info_t *end_p00   = cells[dir::f_p00+1]; size_t n_p00 = end_p00 - begin_p00;
  const quad_info_t *begin_0m0 = cells[dir::f_0m0  ];
  const quad_info_t *end_0m0   = cells[dir::f_0m0+1]; size_t n_0m0 = end_0m0 - begin_0m0;
  const quad_info_t *begin_0p0 = cells[dir::f_0p0  ];
  const quad_info_t *end_0p0   = cells[dir::f_0p0+1]; size_t n_0p0 = end_0p0 - begin_0p0;
  const quad_info_t *begin_00m = cells[dir::f_00m  ];
  const quad_info_t *end_00m   = cells[dir::f_00m+1]; size_t n_00m = end_00m - begin_00m;

  size_t num_points = cells[P4EST_FACES]-cells[0] + 1;
  fprintf(vtk, "POINTS %ld double\n", num_points);

  std::vector<p4est_locidx_t> elements;
  elements.reserve(4*num_points); // worst-case estimate

  std::map<p4est_locidx_t, int> map;

  int c = 0;
  /* write the center location of the central point */
  {
    double qh = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;

    p4est_topidx_t v_mm = conn->tree_to_vertex[P4EST_CHILDREN*tr];
    double tree_xmin = conn->vertices[3*v_mm + 0];
    double tree_ymin = conn->vertices[3*v_mm + 1];
    double tree_zmin = conn->vertices[3*v_mm + 2];

    double x  = quad_x_fr_i(quad) + 0.5*qh + tree_xmin;
    double y  = quad_y_fr_j(quad) + 0.5*qh + tree_ymin;
    double z  = quad_z_fr_k(quad) + 0.5*qh + tree_zmin;

    fprintf(vtk, "%lf %lf %lf\n", x, y, z);
    map.insert(std::make_pair(q,c++));
  }

  /* write center location and create mapping */
  for (const quad_info_t *it = cells[0]; it != cells[P4EST_FACES]; ++it)
  {
    double qh = (double)P4EST_QUADRANT_LEN(it->quad->level)/(double)P4EST_ROOT_LEN;

    p4est_topidx_t v_mm = conn->tree_to_vertex[P4EST_CHILDREN*it->tree_idx];
    double tree_xmin = conn->vertices[3*v_mm + 0];
    double tree_ymin = conn->vertices[3*v_mm + 1];
    double tree_zmin = conn->vertices[3*v_mm + 2];

    double x  = quad_x_fr_i(it->quad) + 0.5*qh + tree_xmin;
    double y  = quad_y_fr_j(it->quad) + 0.5*qh + tree_ymin;
    double z  = quad_z_fr_k(it->quad) + 0.5*qh + tree_zmin;

    fprintf(vtk, "%lf %lf %lf\n", x, y, z);
    map.insert(std::make_pair(it->locidx, c++));
  }

  /* loop over all cells and find neighbors in the p00, 0p0, and pp0 directions */
  for (const quad_info_t *it = begin_00m; it != end_00m; ++it)
  {
    std::vector<p4est_locidx_t> ng_p00;
    std::vector<p4est_locidx_t> ng_0p0;
    int8_t l_p0, l_0p;

    bool is_p00_boundary = it->quad->x + P4EST_QUADRANT_LEN(it->quad->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level);
    bool is_0p0_boundary = it->quad->y + P4EST_QUADRANT_LEN(it->quad->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level);

    /* p00 */
    if (!is_p00_boundary)
    {
      const quad_info_t *begin_ = this->face_begin(it->locidx, dir::f_p00);
      const quad_info_t *end_   = this->face_end(it->locidx, dir::f_p00);
      ng_p00.reserve(end_ - begin_);
      for (const quad_info_t *it_ = begin_; it_ != end_; ++it_)
        if (it_->quad->z + P4EST_QUADRANT_LEN(it_->quad->level) == it->quad->z + P4EST_QUADRANT_LEN(it->quad->level) ){
          ng_p00.push_back(it_->locidx);
          l_p0 = it_->quad->level;
        }
    }

    /* 0p0 */
    if (!is_0p0_boundary)
    {
      const quad_info_t *begin_ = this->face_begin(it->locidx, dir::f_0p0);
      const quad_info_t *end_   = this->face_end(it->locidx, dir::f_0p0);
      ng_0p0.reserve(end_ - begin_);
      for (const quad_info_t *it_ = begin_; it_ != end_; ++it_)
        if (it_->quad->z + P4EST_QUADRANT_LEN(it_->quad->level) == it->quad->z + P4EST_QUADRANT_LEN(it->quad->level)){
          ng_0p0.push_back(it_->locidx);
          l_0p = it_->quad->level;
        }
    }

    /* pp0 */
    if (!is_p00_boundary && !is_0p0_boundary)
    {
      const quad_info_t *it_;
      if (l_p0 > l_0p)
        it_ = this->face_begin(ng_p00[ng_p00.size() - 1], dir::f_0p0);
      else
        it_ = this->face_begin(ng_0p0[ng_0p0.size() - 1], dir::f_p00);

      ng_p00.push_back(it_->locidx);
      ng_0p0.push_back(it_->locidx);
    }

    /* now that we have all cells, construct the elements */
    if (ng_p00.size() != 0)
      for (size_t i=0; i<ng_p00.size()-1; i++){
        elements.push_back(q);
        elements.push_back(it->locidx);
        elements.push_back(ng_p00[i]);
        elements.push_back(ng_p00[i+1]);
      }

    if (ng_0p0.size() != 0)
      for (size_t i=0; i<ng_0p0.size()-1; i++){
        elements.push_back(q);
        elements.push_back(it->locidx);
        elements.push_back(ng_0p0[i]);
        elements.push_back(ng_0p0[i+1]);
      }
  }

  /* now we do stiching! */
  /* 1 - across m00 direction */
  if (n_00m != 0) {
    {
      /* first locate top cells along the edge */
      std::vector<p4est_locidx_t> ng1;
      ng1.reserve(end_00m - begin_00m);
      for (const quad_info_t *it = begin_00m; it != end_00m; ++it)
        if (it->quad->x == quad->x){
          ng1.push_back(it->locidx);
        }

      std::vector<p4est_locidx_t> ng2;
      ng2.reserve(end_m00 - begin_m00);
      for (const quad_info_t *it = begin_m00; it != end_m00; ++it)
        if (it->quad->z == quad->z){
          ng2.push_back(it->locidx);
        }

      /* now construct the triangulation */
      if (ng1.size() != 0 && ng2.size() != 0){
        if (ng1.size() > ng2.size())
          std::swap(ng1, ng2);

        // forward
        int s = (ng2.size()-1) / ng1.size();
        int r = (ng2.size()-1) % ng1.size();
        int c = 0;
        std::vector<int> ng2s; ng2s.reserve(ng1.size() - 1);
        if (ng2.size() > 1){
          for (int i=0; i<r; i++){
            for (int j = 0; j<s+1; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s+1;
            ng2s.push_back(c);
          }
          for (size_t i = r; i<ng1.size(); i++){
            for (int j = 0; j<s; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s;
            ng2s.push_back(c);
          }
        }
        //backward
        if (ng1.size() > 1)
          for (size_t i=0; i<ng1.size() - 1; i++){
            elements.push_back(q);
            elements.push_back(ng1[i]);
            elements.push_back(ng1[i+1]);
            elements.push_back(ng2[ng2s[i]]);
          }
      }
    }
    /* 2 - across p00 direction */
    {
      /* first locate top cells along the edge */
      std::vector<p4est_locidx_t> ng1;
      ng1.reserve(end_00m - begin_00m);
      for (const quad_info_t *it = begin_00m; it != end_00m; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->quad->level) == quad->x + P4EST_QUADRANT_LEN(quad->level)){
          ng1.push_back(it->locidx);
        }

      std::vector<p4est_locidx_t> ng2;
      ng2.reserve(end_p00 - begin_p00);
      for (const quad_info_t *it = begin_p00; it != end_p00; ++it)
        if (it->quad->z == quad->z){
          ng2.push_back(it->locidx);
        }

      /* now construct the triangulation */
      if (ng1.size() != 0 && ng2.size() != 0){
        if (ng1.size() > ng2.size())
          std::swap(ng1, ng2);

        // forward
        int s = (ng2.size()-1) / ng1.size();
        int r = (ng2.size()-1) % ng1.size();
        int c = 0;
        std::vector<int> ng2s; ng2s.reserve(ng1.size() - 1);
        if (ng2.size() > 1){
          for (int i=0; i<r; i++){
            for (int j = 0; j<s+1; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s+1;
            ng2s.push_back(c);
          }
          for (size_t i = r; i<ng1.size(); i++){
            for (int j = 0; j<s; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s;
            ng2s.push_back(c);
          }
        }
        //backward
        if (ng1.size() > 1)
          for (size_t i=0; i<ng1.size() - 1; i++){
            elements.push_back(q);
            elements.push_back(ng1[i]);
            elements.push_back(ng1[i+1]);
            elements.push_back(ng2[ng2s[i]]);
          }
      }
    }
    /* 3 - across 0m0 direction */
    {
      /* first locate top cells along the edge */
      std::vector<p4est_locidx_t> ng1;
      ng1.reserve(end_00m - begin_00m);
      for (const quad_info_t *it = begin_00m; it != end_00m; ++it)
        if (it->quad->y == quad->y){
          ng1.push_back(it->locidx);
        }

      std::vector<p4est_locidx_t> ng2;
      ng2.reserve(end_0m0 - begin_0m0);
      for (const quad_info_t *it = begin_0m0; it != end_0m0; ++it)
        if (it->quad->z == quad->z){
          ng2.push_back(it->locidx);
        }

      /* now construct the triangulation */
      if (ng1.size() != 0 && ng2.size() != 0){
        if (ng1.size() > ng2.size())
          std::swap(ng1, ng2);

        // forward
        int s = (ng2.size()-1) / ng1.size();
        int r = (ng2.size()-1) % ng1.size();
        int c = 0;
        std::vector<int> ng2s; ng2s.reserve(ng1.size() - 1);
        if (ng2.size() > 1){
          for (int i=0; i<r; i++){
            for (int j = 0; j<s+1; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s+1;
            ng2s.push_back(c);
          }
          for (size_t i = r; i<ng1.size(); i++){
            for (int j = 0; j<s; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s;
            ng2s.push_back(c);
          }
        }
        //backward
        if (ng1.size() > 1)
          for (size_t i=0; i<ng1.size() - 1; i++){
            elements.push_back(q);
            elements.push_back(ng1[i]);
            elements.push_back(ng1[i+1]);
            elements.push_back(ng2[ng2s[i]]);
          }
      }
    }
    /* 4 - across 0p0 direction */
    {
      /* first locate top cells along the edge */
      std::vector<p4est_locidx_t> ng1;
      ng1.reserve(end_00m - begin_00m);
      for (const quad_info_t *it = begin_00m; it != end_00m; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->quad->level) == quad->y + P4EST_QUADRANT_LEN(quad->level)){
          ng1.push_back(it->locidx);
        }

      std::vector<p4est_locidx_t> ng2;
      ng2.reserve(end_0p0 - begin_0p0);
      for (const quad_info_t *it = begin_0p0; it != end_0p0; ++it)
        if (it->quad->z == quad->z){
          ng2.push_back(it->locidx);
        }

      /* now construct the triangulation */
      if (ng1.size() != 0 && ng2.size() != 0){
        if (ng1.size() > ng2.size())
          std::swap(ng1, ng2);

        // forward
        int s = (ng2.size()-1) / ng1.size();
        int r = (ng2.size()-1) % ng1.size();
        int c = 0;
        std::vector<int> ng2s; ng2s.reserve(ng1.size() - 1);
        if (ng2.size() > 1){
          for (int i=0; i<r; i++){
            for (int j = 0; j<s+1; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s+1;
            ng2s.push_back(c);
          }
          for (size_t i = r; i<ng1.size(); i++){
            for (int j = 0; j<s; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s;
            ng2s.push_back(c);
          }
        }
        //backward
        if (ng1.size() > 1)
          for (size_t i=0; i<ng1.size() - 1; i++){
            elements.push_back(q);
            elements.push_back(ng1[i]);
            elements.push_back(ng1[i+1]);
            elements.push_back(ng2[ng2s[i]]);
          }
      }
    }
    /* 5 - across mm0 */
    if (n_m00 != 0 && n_0m0 != 0)
    {
      elements.push_back(q);
      elements.push_back(begin_00m->locidx);

      for (const quad_info_t *it = begin_m00; it != end_m00; ++it)
        if (it->quad->z <= quad->z){
          elements.push_back(it->locidx);
          break;
        }

      for (const quad_info_t *it = begin_0m0; it != end_0m0; ++it)
        if (it->quad->z <= quad->z){
          elements.push_back(it->locidx);
          break;
        }
    }

    /* 6 - across pm0 */
    if (n_p00 != 0 && n_0m0 != 0)
    {
      elements.push_back(q);
      for (const quad_info_t *it = begin_00m; it != end_00m; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->quad->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level)){
          elements.push_back(it->locidx);
          break;
        }

      elements.push_back(begin_p00->locidx);

      for (const quad_info_t *it = begin_0m0; it != end_0m0; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->quad->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level)){
          elements.push_back(it->locidx);
          break;
        }
    }
    /* 7 - across mp0 */
    if (n_m00 != 0 && n_0p0 != 0)
    {
      elements.push_back(q);
      for (const quad_info_t *it = begin_00m; it != end_00m; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->quad->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level)){
          elements.push_back(it->locidx);
          break;
        }

      for (const quad_info_t* it = begin_m00; it != end_m00; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->quad->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level)){
          elements.push_back(it->locidx);
          break;
        }

      for (const quad_info_t *it = begin_0p0; it != end_0p0; ++it)
        if (it->quad->z <= quad->z){
          elements.push_back(it->locidx);
          break;
        }
    }
    /* 8 - across pp0 */
    if (n_p00 != 0 && n_0p0 != 0)
    {
      elements.push_back(q);
      elements.push_back((end_00m-1)->locidx);
      for (const quad_info_t *it = begin_p00; it != end_p00; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->quad->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level)){
          elements.push_back(it->locidx);
          break;
        }
      for (const quad_info_t *it = begin_0p0; it != end_0p0; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->quad->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level)){
          elements.push_back(it->locidx);
          break;
        }
    }
  }
  /* write the connectivity information (a.k.a. elements) */
  size_t num_elements = elements.size()/4;
  fprintf(vtk, "CELLS %ld %ld \n", num_elements, (1+4)*num_elements);

  for (size_t i=0; i<num_elements; ++i)
  {
    fprintf(vtk, "%d ", 4);
    for (short j = 0; j<4; j++)
      fprintf(vtk, "%d ", map[elements[4*i + j]]);
    fprintf(vtk, "\n");
  }

  fprintf(vtk, "CELL_TYPES %ld\n", num_elements);
  for (size_t i=0; i<num_elements; ++i)
    fprintf(vtk, "%d\n", VTK_CELL_TYPE);
  fclose(vtk);
}

void my_p4est_cell_neighbors_t::write_cell_triangulation_vtk_m00(p4est_locidx_t qu, p4est_topidx_t tr, const char *filename)
{
  p4est_connectivity_t *conn = p4est->connectivity;
  p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tr);
  const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, qu);
  p4est_locidx_t q = qu + tree->quadrants_offset;

  char vtkname[1024];
  sprintf(vtkname, "%s_%04d.%d.vtk", filename, p4est->mpirank, q);
  FILE *vtk = fopen(vtkname, "w");

  fprintf(vtk, "# vtk DataFile Version 2.0 \n");
  fprintf(vtk, "Triangulation \n");
  fprintf(vtk, "ASCII \n");
  fprintf(vtk, "DATASET UNSTRUCTURED_GRID \n");

  const quad_info_t *cells[P4EST_FACES + 1];
  for (short i = 0; i<P4EST_FACES; i++)
    cells[i] = this->face_begin(q, i);
  cells[P4EST_FACES] = this->face_end(q, P4EST_FACES - 1);

  const quad_info_t *begin_m00 = cells[dir::f_m00  ];
  const quad_info_t *end_m00   = cells[dir::f_m00+1]; size_t n_m00 = end_m00 - begin_m00;
  const quad_info_t *begin_p00 = cells[dir::f_p00  ];
  const quad_info_t *end_p00   = cells[dir::f_p00+1]; size_t n_p00 = end_p00 - begin_p00;
  const quad_info_t *begin_0m0 = cells[dir::f_0m0  ];
  const quad_info_t *end_0m0   = cells[dir::f_0m0+1]; size_t n_0m0 = end_0m0 - begin_0m0;
  const quad_info_t *begin_0p0 = cells[dir::f_0p0  ];
  const quad_info_t *end_0p0   = cells[dir::f_0p0+1]; size_t n_0p0 = end_0p0 - begin_0p0;
  const quad_info_t *begin_00m = cells[dir::f_00m  ];
  const quad_info_t *end_00m   = cells[dir::f_00m+1]; size_t n_00m = end_00m - begin_00m;
  const quad_info_t *begin_00p = cells[dir::f_00p  ];
  const quad_info_t *end_00p   = cells[dir::f_00p+1]; size_t n_00p = end_00p - begin_00p;

  size_t num_points = cells[P4EST_FACES]-cells[0] + 1;
  fprintf(vtk, "POINTS %ld double\n", num_points);

  std::vector<p4est_locidx_t> elements;
  elements.reserve(4*num_points); // worst-case estimate

  std::map<p4est_locidx_t, int> map;

  int c = 0;
  /* write the center location of the central point */
  {
    double qh = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;

    p4est_topidx_t v_mm = conn->tree_to_vertex[P4EST_CHILDREN*tr];
    double tree_xmin = conn->vertices[3*v_mm + 0];
    double tree_ymin = conn->vertices[3*v_mm + 1];
    double tree_zmin = conn->vertices[3*v_mm + 2];

    double x  = quad_x_fr_i(quad) + 0.5*qh + tree_xmin;
    double y  = quad_y_fr_j(quad) + 0.5*qh + tree_ymin;
    double z  = quad_z_fr_k(quad) + 0.5*qh + tree_zmin;

    fprintf(vtk, "%lf %lf %lf\n", x, y, z);
    map.insert(std::make_pair(q,c++));
  }

  /* write center location and create mapping */
  for (const quad_info_t *it = cells[0]; it != cells[P4EST_FACES]; ++it)
  {
    double qh = (double)P4EST_QUADRANT_LEN(it->quad->level)/(double)P4EST_ROOT_LEN;

    p4est_topidx_t v_mm = conn->tree_to_vertex[P4EST_CHILDREN*it->tree_idx];
    double tree_xmin = conn->vertices[3*v_mm + 0];
    double tree_ymin = conn->vertices[3*v_mm + 1];
    double tree_zmin = conn->vertices[3*v_mm + 2];

    double x  = quad_x_fr_i(it->quad) + 0.5*qh + tree_xmin;
    double y  = quad_y_fr_j(it->quad) + 0.5*qh + tree_ymin;
    double z  = quad_z_fr_k(it->quad) + 0.5*qh + tree_zmin;

    fprintf(vtk, "%lf %lf %lf\n", x, y, z);
    map.insert(std::make_pair(it->locidx, c++));
  }

  /* loop over all cells and find neighbors in the p00, 0p0, and pp0 directions */
  for (const quad_info_t *it = begin_m00; it != end_m00; ++it)
  {
    std::vector<p4est_locidx_t> ng_0p0;
    std::vector<p4est_locidx_t> ng_00p;
    int8_t l_0p0, l_00p;

    bool is_boundary_0p0 = it->quad->y + P4EST_QUADRANT_LEN(it->quad->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level);
    bool is_boundary_00p = it->quad->z + P4EST_QUADRANT_LEN(it->quad->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level);

    /* p00 */
    if (!is_boundary_0p0)
    {
      const quad_info_t *begin_ = this->face_begin(it->locidx, dir::f_0p0);
      const quad_info_t *end_   = this->face_end(it->locidx, dir::f_0p0);
      ng_0p0.reserve(end_ - begin_);
      for (const quad_info_t *it_ = begin_; it_ != end_; ++it_)
        if (it_->quad->x + P4EST_QUADRANT_LEN(it_->quad->level) == it->quad->x + P4EST_QUADRANT_LEN(it->quad->level) ){
          ng_0p0.push_back(it_->locidx);
          l_0p0 = it_->quad->level;
        }
    }

    /* 0p0 */
    if (!is_boundary_00p)
    {
      const quad_info_t *begin_ = this->face_begin(it->locidx, dir::f_00p);
      const quad_info_t *end_   = this->face_end(it->locidx, dir::f_00p);
      ng_00p.reserve(end_ - begin_);
      for (const quad_info_t *it_ = begin_; it_ != end_; ++it_)
        if (it_->quad->x + P4EST_QUADRANT_LEN(it_->quad->level) == it->quad->x + P4EST_QUADRANT_LEN(it->quad->level)){
          ng_00p.push_back(it_->locidx);
          l_00p = it_->quad->level;
        }
    }

    /* pp0 */
    if (!is_boundary_0p0 && !is_boundary_00p)
    {
      const quad_info_t *it_;
      if (l_0p0 > l_00p)
        it_ = this->face_begin(ng_0p0[ng_0p0.size() - 1], dir::f_00p);
      else
        it_ = this->face_begin(ng_00p[ng_00p.size() - 1], dir::f_0p0);

      ng_0p0.push_back(it_->locidx);
      ng_00p.push_back(it_->locidx);
    }

    /* now that we have all cells, construct the elements */
    if (ng_0p0.size() != 0)
      for (size_t i=0; i<ng_0p0.size()-1; i++){
        elements.push_back(q);
        elements.push_back(it->locidx);
        elements.push_back(ng_0p0[i]);
        elements.push_back(ng_0p0[i+1]);
      }

    if (ng_00p.size() != 0)
      for (size_t i=0; i<ng_00p.size()-1; i++){
        elements.push_back(q);
        elements.push_back(it->locidx);
        elements.push_back(ng_00p[i]);
        elements.push_back(ng_00p[i+1]);
      }
  }

  /* now we do stiching! */
  /* 1 - across m00 direction */
  if (n_m00 != 0) {
    {
      /* first locate top cells along the edge */
      std::vector<p4est_locidx_t> ng1;
      ng1.reserve(end_m00 - begin_m00);
      for (const quad_info_t *it = begin_m00; it != end_m00; ++it)
        if (it->quad->y <= quad->y){
          ng1.push_back(it->locidx);
        }

      std::vector<p4est_locidx_t> ng2;
      ng2.reserve(end_0m0 - begin_0m0);
      for (const quad_info_t *it = begin_0m0; it != end_0m0; ++it)
        if (it->quad->x <= quad->x){
          ng2.push_back(it->locidx);
        }

      /* now construct the triangulation */
      if (ng1.size() != 0 && ng2.size() != 0){
        if (ng1.size() > ng2.size())
          std::swap(ng1, ng2);

        // forward
        int s = (ng2.size()-1) / ng1.size();
        int r = (ng2.size()-1) % ng1.size();
        int c = 0;
        std::vector<int> ng2s; ng2s.reserve(ng1.size() - 1);
        if (ng2.size() > 1){
          for (int i=0; i<r; i++){
            for (int j = 0; j<s+1; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s+1;
            ng2s.push_back(c);
          }
          for (size_t i = r; i<ng1.size(); i++){
            for (int j = 0; j<s; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s;
            ng2s.push_back(c);
          }
        }
        //backward
        if (ng1.size() > 1)
          for (size_t i=0; i<ng1.size() - 1; i++){
            elements.push_back(q);
            elements.push_back(ng1[i]);
            elements.push_back(ng1[i+1]);
            elements.push_back(ng2[ng2s[i]]);
          }
      }
    }
    /* 2 - across p00 direction */
    {
      /* first locate top cells along the edge */
      std::vector<p4est_locidx_t> ng1;
      ng1.reserve(end_m00 - begin_m00);
      for (const quad_info_t *it = begin_m00; it != end_m00; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->quad->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level)){
          ng1.push_back(it->locidx);
        }

      std::vector<p4est_locidx_t> ng2;
      ng2.reserve(end_0p0 - begin_0p0);
      for (const quad_info_t *it = begin_0p0; it != end_0p0; ++it)
        if (it->quad->x <= quad->x){
          ng2.push_back(it->locidx);
        }

      /* now construct the triangulation */
      if (ng1.size() != 0 && ng2.size() != 0){
        if (ng1.size() > ng2.size())
          std::swap(ng1, ng2);

        // forward
        int s = (ng2.size()-1) / ng1.size();
        int r = (ng2.size()-1) % ng1.size();
        int c = 0;
        std::vector<int> ng2s; ng2s.reserve(ng1.size() - 1);
        if (ng2.size() > 1){
          for (int i=0; i<r; i++){
            for (int j = 0; j<s+1; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s+1;
            ng2s.push_back(c);
          }
          for (size_t i = r; i<ng1.size(); i++){
            for (int j = 0; j<s; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s;
            ng2s.push_back(c);
          }
        }
        //backward
        if (ng1.size() > 1)
          for (size_t i=0; i<ng1.size() - 1; i++){
            elements.push_back(q);
            elements.push_back(ng1[i]);
            elements.push_back(ng1[i+1]);
            elements.push_back(ng2[ng2s[i]]);
          }
      }
    }
    /* 3 - across 0m0 direction */
    {
      /* first locate top cells along the edge */
      std::vector<p4est_locidx_t> ng1;
      ng1.reserve(end_m00 - begin_m00);
      for (const quad_info_t *it = begin_m00; it != end_m00; ++it)
        if (it->quad->z <= quad->z){
          ng1.push_back(it->locidx);
        }

      std::vector<p4est_locidx_t> ng2;
      ng2.reserve(end_00m- begin_00m);
      for (const quad_info_t *it = begin_00m; it != end_00m; ++it)
        if (it->quad->x <= quad->x){
          ng2.push_back(it->locidx);
        }

      /* now construct the triangulation */
      if (ng1.size() != 0 && ng2.size() != 0){
        if (ng1.size() > ng2.size())
          std::swap(ng1, ng2);

        // forward
        int s = (ng2.size()-1) / ng1.size();
        int r = (ng2.size()-1) % ng1.size();
        int c = 0;
        std::vector<int> ng2s; ng2s.reserve(ng1.size() - 1);
        if (ng2.size() > 1){
          for (int i=0; i<r; i++){
            for (int j = 0; j<s+1; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s+1;
            ng2s.push_back(c);
          }
          for (size_t i = r; i<ng1.size(); i++){
            for (int j = 0; j<s; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s;
            ng2s.push_back(c);
          }
        }
        //backward
        if (ng1.size() > 1)
          for (size_t i=0; i<ng1.size() - 1; i++){
            elements.push_back(q);
            elements.push_back(ng1[i]);
            elements.push_back(ng1[i+1]);
            elements.push_back(ng2[ng2s[i]]);
          }
      }
    }
    /* 4 - across 0p0 direction */
    {
      /* first locate top cells along the edge */
      std::vector<p4est_locidx_t> ng1;
      ng1.reserve(end_m00 - begin_m00);
      for (const quad_info_t *it = begin_m00; it != end_m00; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->quad->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level)){
          ng1.push_back(it->locidx);
        }

      std::vector<p4est_locidx_t> ng2;
      ng2.reserve(end_00p - begin_00p);
      for (const quad_info_t *it = begin_00p; it != end_00p; ++it)
        if (it->quad->x <= quad->x){
          ng2.push_back(it->locidx);
        }

      /* now construct the triangulation */
      if (ng1.size() != 0 && ng2.size() != 0){
        if (ng1.size() > ng2.size())
          std::swap(ng1, ng2);

        // forward
        int s = (ng2.size()-1) / ng1.size();
        int r = (ng2.size()-1) % ng1.size();
        int c = 0;
        std::vector<int> ng2s; ng2s.reserve(ng1.size() - 1);
        if (ng2.size() > 1){
          for (int i=0; i<r; i++){
            for (int j = 0; j<s+1; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s+1;
            ng2s.push_back(c);
          }
          for (size_t i=r; i<ng1.size(); i++){
            for (int j = 0; j<s; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s;
            ng2s.push_back(c);
          }
        }
        //backward
        if (ng1.size() > 1)
          for (size_t i=0; i<ng1.size() - 1; i++){
            elements.push_back(q);
            elements.push_back(ng1[i]);
            elements.push_back(ng1[i+1]);
            elements.push_back(ng2[ng2s[i]]);
          }
      }
    }
    /* 5 - across mm0 */

    if (n_0m0 != 0 && n_00m != 0)
    {
      elements.push_back(q);
      elements.push_back(begin_m00->locidx);
      elements.push_back(begin_0m0->locidx);
      elements.push_back(begin_00m->locidx);
    }

    /* 6 - across pm0 */
    if (n_0p0 != 0 && n_00m != 0)
    {
      elements.push_back(q);
      for (const quad_info_t *it = begin_m00; it != end_m00; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->quad->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level)){
          elements.push_back(it->locidx);
          break;
        }

      elements.push_back(begin_0p0->locidx);

      for (const quad_info_t *it = begin_00m; it != end_00m; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->quad->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level)){
          elements.push_back(it->locidx);
          break;
        }
    }
    /* 7 - across mp0 */
    if (n_0m0 != 0 && n_00p != 0)
    {
      elements.push_back(q);
      for (const quad_info_t *it = begin_m00; it != end_m00; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->quad->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level)){
          elements.push_back(it->locidx);
          break;
        }

      for (const quad_info_t *it = begin_0m0; it != end_0m0; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->quad->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level)){
          elements.push_back(it->locidx);
          break;
        }

      elements.push_back(begin_00p->locidx);
    }
    /* 8 - across pp0 */
    if (n_p00 != 0 && n_0p0 != 0)
    {
      elements.push_back(q);
      elements.push_back((end_m00-1)->locidx);
      for (const quad_info_t *it = begin_0p0; it != end_0p0; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->quad->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level)){
          elements.push_back(it->locidx);
          break;
        }
      for (const quad_info_t *it = begin_00p; it != end_00p; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->quad->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level)){
          elements.push_back(it->locidx);
          break;
        }
    }
  }
  /* write the connectivity information (a.k.a. elements) */
  size_t num_elements = elements.size()/4;
  fprintf(vtk, "CELLS %ld %ld \n", num_elements, (1+4)*num_elements);

  for (size_t i=0; i<num_elements; ++i)
  {
    fprintf(vtk, "%d ", 4);
    for (short j = 0; j<4; j++)
      fprintf(vtk, "%d ", map[elements[4*i + j]]);
    fprintf(vtk, "\n");
  }

  fprintf(vtk, "CELL_TYPES %ld\n", num_elements);
  for (size_t i=0; i<num_elements; ++i)
    fprintf(vtk, "%d\n", VTK_CELL_TYPE);
  fclose(vtk);
}

void my_p4est_cell_neighbors_t::write_cell_triangulation_vtk_p00(p4est_locidx_t qu, p4est_topidx_t tr, const char *filename)
{
  p4est_connectivity_t *conn = p4est->connectivity;
  p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tr);
  const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, qu);
  p4est_locidx_t q = qu + tree->quadrants_offset;

  char vtkname[1024];
  sprintf(vtkname, "%s_%04d.%d.vtk", filename, p4est->mpirank, q);
  FILE *vtk = fopen(vtkname, "w");

  fprintf(vtk, "# vtk DataFile Version 2.0 \n");
  fprintf(vtk, "Triangulation \n");
  fprintf(vtk, "ASCII \n");
  fprintf(vtk, "DATASET UNSTRUCTURED_GRID \n");

  const quad_info_t *cells[P4EST_FACES + 1];
  for (short i = 0; i<P4EST_FACES; i++)
    cells[i] = this->face_begin(q, i);
  cells[P4EST_FACES] = this->face_end(q, P4EST_FACES - 1);

  const quad_info_t *begin_p00 = cells[dir::f_p00  ];
  const quad_info_t *end_p00   = cells[dir::f_p00+1]; size_t n_p00 = end_p00 - begin_p00;
  const quad_info_t *begin_0m0 = cells[dir::f_0m0  ];
  const quad_info_t *end_0m0   = cells[dir::f_0m0+1]; size_t n_0m0 = end_0m0 - begin_0m0;
  const quad_info_t *begin_0p0 = cells[dir::f_0p0  ];
  const quad_info_t *end_0p0   = cells[dir::f_0p0+1]; size_t n_0p0 = end_0p0 - begin_0p0;
  const quad_info_t *begin_00m = cells[dir::f_00m  ];
  const quad_info_t *end_00m   = cells[dir::f_00m+1]; size_t n_00m = end_00m - begin_00m;
  const quad_info_t *begin_00p = cells[dir::f_00p  ];
  const quad_info_t *end_00p   = cells[dir::f_00p+1]; size_t n_00p = end_00p - begin_00p;

  size_t num_points = cells[P4EST_FACES]-cells[0] + 1;
  fprintf(vtk, "POINTS %ld double\n", num_points);

  std::vector<p4est_locidx_t> elements;
  elements.reserve(4*num_points); // worst-case estimate

  std::map<p4est_locidx_t, int> map;

  int c = 0;
  /* write the center location of the central point */
  {
    double qh = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;

    p4est_topidx_t v_mm = conn->tree_to_vertex[P4EST_CHILDREN*tr];
    double tree_xmin = conn->vertices[3*v_mm + 0];
    double tree_ymin = conn->vertices[3*v_mm + 1];
    double tree_zmin = conn->vertices[3*v_mm + 2];

    double x  = quad_x_fr_i(quad) + 0.5*qh + tree_xmin;
    double y  = quad_y_fr_j(quad) + 0.5*qh + tree_ymin;
    double z  = quad_z_fr_k(quad) + 0.5*qh + tree_zmin;

    fprintf(vtk, "%lf %lf %lf\n", x, y, z);
    map.insert(std::make_pair(q,c++));
  }

  /* write center location and create mapping */
  for (const quad_info_t *it = cells[0]; it != cells[P4EST_FACES]; ++it)
  {
    double qh = (double)P4EST_QUADRANT_LEN(it->quad->level)/(double)P4EST_ROOT_LEN;

    p4est_topidx_t v_mm = conn->tree_to_vertex[P4EST_CHILDREN*it->tree_idx];
    double tree_xmin = conn->vertices[3*v_mm + 0];
    double tree_ymin = conn->vertices[3*v_mm + 1];
    double tree_zmin = conn->vertices[3*v_mm + 2];

    double x  = quad_x_fr_i(it->quad) + 0.5*qh + tree_xmin;
    double y  = quad_y_fr_j(it->quad) + 0.5*qh + tree_ymin;
    double z  = quad_z_fr_k(it->quad) + 0.5*qh + tree_zmin;

    fprintf(vtk, "%lf %lf %lf\n", x, y, z);
    map.insert(std::make_pair(it->locidx, c++));
  }

  /* loop over all cells and find neighbors in the p00, 0p0, and pp0 directions */
  for (const quad_info_t *it = begin_p00; it != end_p00; ++it)
  {
    std::vector<p4est_locidx_t> ng_0p0;
    std::vector<p4est_locidx_t> ng_00p;
    int8_t l_0p0, l_00p;

    bool is_boundary_0p0 = it->quad->y + P4EST_QUADRANT_LEN(it->quad->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level);
    bool is_boundary_00p = it->quad->z + P4EST_QUADRANT_LEN(it->quad->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level);

    /* p00 */
    if (!is_boundary_0p0)
    {
      const quad_info_t *begin_ = this->face_begin(it->locidx, dir::f_0p0);
      const quad_info_t *end_   = this->face_end(it->locidx, dir::f_0p0);
      ng_0p0.reserve(end_ - begin_);
      for (const quad_info_t *it_ = begin_; it_ != end_; ++it_)
        if (it_->quad->x == it->quad->x){
          ng_0p0.push_back(it_->locidx);
          l_0p0 = it_->quad->level;
        }
    }

    /* 0p0 */
    if (!is_boundary_00p)
    {
      const quad_info_t *begin_ = this->face_begin(it->locidx, dir::f_00p);
      const quad_info_t *end_   = this->face_end(it->locidx, dir::f_00p);
      ng_00p.reserve(end_ - begin_);
      for (const quad_info_t *it_ = begin_; it_ != end_; ++it_)
        if (it_->quad->x == it->quad->x){
          ng_00p.push_back(it_->locidx);
          l_00p = it_->quad->level;
        }
    }

    /* pp0 */
    if (!is_boundary_0p0 && !is_boundary_00p)
    {
      const quad_info_t *it_;
      if (l_0p0 > l_00p)
        it_ = this->face_begin(ng_0p0[ng_0p0.size() - 1], dir::f_00p);
      else
        it_ = this->face_begin(ng_00p[ng_00p.size() - 1], dir::f_0p0);

      ng_0p0.push_back(it_->locidx);
      ng_00p.push_back(it_->locidx);
    }

    /* now that we have all cells, construct the elements */
    if (ng_0p0.size() != 0)
      for (size_t i=0; i<ng_0p0.size()-1; i++){
        elements.push_back(q);
        elements.push_back(it->locidx);
        elements.push_back(ng_0p0[i]);
        elements.push_back(ng_0p0[i+1]);
      }

    if (ng_00p.size() != 0)
      for (size_t i=0; i<ng_00p.size()-1; i++){
        elements.push_back(q);
        elements.push_back(it->locidx);
        elements.push_back(ng_00p[i]);
        elements.push_back(ng_00p[i+1]);
      }
  }

  /* now we do stiching! */
  /* 1 - across m00 direction */
  if (n_p00 != 0) {
    {
      /* first locate top cells along the edge */
      std::vector<p4est_locidx_t> ng1;
      ng1.reserve(end_p00 - begin_p00);
      for (const quad_info_t *it = begin_p00; it != end_p00; ++it)
        if (it->quad->y <= quad->y){
          ng1.push_back(it->locidx);
        }

      std::vector<p4est_locidx_t> ng2;
      ng2.reserve(end_0m0 - begin_0m0);
      for (const quad_info_t *it = begin_0m0; it != end_0m0; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->quad->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level)){
          ng2.push_back(it->locidx);
        }

      /* now construct the triangulation */
      if (ng1.size() != 0 && ng2.size() != 0){
        if (ng1.size() > ng2.size())
          std::swap(ng1, ng2);

        // forward
        int s = (ng2.size()-1) / ng1.size();
        int r = (ng2.size()-1) % ng1.size();
        int c = 0;
        std::vector<int> ng2s; ng2s.reserve(ng1.size() - 1);
        if (ng2.size() > 1){
          for (int i=0; i<r; i++){
            for (int j = 0; j<s+1; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s+1;
            ng2s.push_back(c);
          }
          for (size_t i = r; i<ng1.size(); i++){
            for (int j = 0; j<s; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s;
            ng2s.push_back(c);
          }
        }
        //backward
        if (ng1.size() > 1)
          for (size_t i=0; i<ng1.size() - 1; i++){
            elements.push_back(q);
            elements.push_back(ng1[i]);
            elements.push_back(ng1[i+1]);
            elements.push_back(ng2[ng2s[i]]);
          }
      }
    }
    /* 2 - across p00 direction */
    {
      /* first locate top cells along the edge */
      std::vector<p4est_locidx_t> ng1;
      ng1.reserve(end_p00 - begin_p00);
      for (const quad_info_t *it = begin_p00; it != end_p00; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->quad->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level)){
          ng1.push_back(it->locidx);
        }

      std::vector<p4est_locidx_t> ng2;
      ng2.reserve(end_0p0 - begin_0p0);
      for (const quad_info_t *it = begin_0p0; it != end_0p0; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->quad->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level)){
          ng2.push_back(it->locidx);
        }

      /* now construct the triangulation */
      if (ng1.size() != 0 && ng2.size() != 0){
        if (ng1.size() > ng2.size())
          std::swap(ng1, ng2);

        // forward
        int s = (ng2.size()-1) / ng1.size();
        int r = (ng2.size()-1) % ng1.size();
        int c = 0;
        std::vector<int> ng2s; ng2s.reserve(ng1.size() - 1);
        if (ng2.size() > 1){
          for (int i=0; i<r; i++){
            for (int j = 0; j<s+1; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s+1;
            ng2s.push_back(c);
          }
          for (size_t i = r; i<ng1.size(); i++){
            for (int j = 0; j<s; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s;
            ng2s.push_back(c);
          }
        }
        //backward
        if (ng1.size() > 1)
          for (size_t i=0; i<ng1.size() - 1; i++){
            elements.push_back(q);
            elements.push_back(ng1[i]);
            elements.push_back(ng1[i+1]);
            elements.push_back(ng2[ng2s[i]]);
          }
      }
    }
    /* 3 - across 0m0 direction */
    {
      /* first locate top cells along the edge */
      std::vector<p4est_locidx_t> ng1;
      ng1.reserve(end_p00 - begin_p00);
      for (const quad_info_t *it = begin_p00; it != end_p00; ++it)
        if (it->quad->z <= quad->z){
          ng1.push_back(it->locidx);
        }

      std::vector<p4est_locidx_t> ng2;
      ng2.reserve(end_00m- begin_00m);
      for (const quad_info_t *it = begin_00m; it != end_00m; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->quad->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level)){
          ng2.push_back(it->locidx);
        }

      /* now construct the triangulation */
      if (ng1.size() != 0 && ng2.size() != 0){
        if (ng1.size() > ng2.size())
          std::swap(ng1, ng2);

        // forward
        int s = (ng2.size()-1) / ng1.size();
        int r = (ng2.size()-1) % ng1.size();
        int c = 0;
        std::vector<int> ng2s; ng2s.reserve(ng1.size() - 1);
        if (ng2.size() > 1){
          for (int i=0; i<r; i++){
            for (int j = 0; j<s+1; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s+1;
            ng2s.push_back(c);
          }
          for (size_t i = r; i<ng1.size(); i++){
            for (int j = 0; j<s; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s;
            ng2s.push_back(c);
          }
        }
        //backward
        if (ng1.size() > 1)
          for (size_t i=0; i<ng1.size() - 1; i++){
            elements.push_back(q);
            elements.push_back(ng1[i]);
            elements.push_back(ng1[i+1]);
            elements.push_back(ng2[ng2s[i]]);
          }
      }
    }
    /* 4 - across 0p0 direction */
    {
      /* first locate top cells along the edge */
      std::vector<p4est_locidx_t> ng1;
      ng1.reserve(end_p00 - begin_p00);
      for (const quad_info_t *it = begin_p00; it != end_p00; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->quad->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level)){
          ng1.push_back(it->locidx);
        }

      std::vector<p4est_locidx_t> ng2;
      ng2.reserve(end_00p - begin_00p);
      for (const quad_info_t *it = begin_00p; it != end_00p; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->quad->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level)){
          ng2.push_back(it->locidx);
        }

      /* now construct the triangulation */
      if (ng1.size() != 0 && ng2.size() != 0){
        if (ng1.size() > ng2.size())
          std::swap(ng1, ng2);

        // forward
        int s = (ng2.size()-1) / ng1.size();
        int r = (ng2.size()-1) % ng1.size();
        int c = 0;
        std::vector<int> ng2s; ng2s.reserve(ng1.size() - 1);
        if (ng2.size() > 1){
          for (int i=0; i<r; i++){
            for (int j = 0; j<s+1; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s+1;
            ng2s.push_back(c);
          }
          for (size_t i=r; i<ng1.size(); i++){
            for (int j = 0; j<s; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s;
            ng2s.push_back(c);
          }
        }
        //backward
        if (ng1.size() > 1)
          for (size_t i=0; i<ng1.size() - 1; i++){
            elements.push_back(q);
            elements.push_back(ng1[i]);
            elements.push_back(ng1[i+1]);
            elements.push_back(ng2[ng2s[i]]);
          }
      }
    }
    /* 5 - across mm0 */
    if (n_0m0 != 0 && n_00m != 0)
    {
      elements.push_back(q);
      elements.push_back(begin_p00->locidx);

      for (const quad_info_t *it = begin_0m0; it != end_0m0; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->quad->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level)){
          elements.push_back(it->locidx);
          break;
        }

      for (const quad_info_t *it = begin_00m; it != end_00m; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->quad->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level)){
          elements.push_back(it->locidx);
          break;
        }
    }

    /* 6 - across pm0 */
    if (n_0p0 != 0 && n_00m != 0)
    {
      elements.push_back(q);
      for (const quad_info_t *it = begin_p00; it != end_p00; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->quad->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level)){
          elements.push_back(it->locidx);
          break;
        }

      for (const quad_info_t *it = begin_0p0; it != end_0p0; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->quad->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level)){
          elements.push_back(it->locidx);
          break;
        }

      elements.push_back((end_00m-1)->locidx);
    }
    /* 7 - across mp0 */
    if (n_0m0 != 0 && n_00p != 0)
    {
      elements.push_back(q);
      for (const quad_info_t *it = begin_p00; it != end_p00; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->quad->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level)){
          elements.push_back(it->locidx);
          break;
        }

      elements.push_back((end_0m0-1)->locidx);
      for (const quad_info_t *it = begin_00p; it != end_00p; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->quad->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level)){
          elements.push_back(it->locidx);
          break;
        }
    }
    /* 8 - across pp0 */
    if (n_0p0 != 0 && n_0p0 != 0)
    {
      elements.push_back(q);
      elements.push_back((end_p00-1)->locidx);
      elements.push_back((end_0p0-1)->locidx);
      elements.push_back((end_00p-1)->locidx);
    }
  }
  /* write the connectivity information (a.k.a. elements) */
  size_t num_elements = elements.size()/4;
  fprintf(vtk, "CELLS %ld %ld \n", num_elements, (1+4)*num_elements);

  for (size_t i=0; i<num_elements; ++i)
  {
    fprintf(vtk, "%d ", 4);
    for (short j = 0; j<4; j++)
      fprintf(vtk, "%d ", map[elements[4*i + j]]);
    fprintf(vtk, "\n");
  }

  fprintf(vtk, "CELL_TYPES %ld\n", num_elements);
  for (size_t i=0; i<num_elements; ++i)
    fprintf(vtk, "%d\n", VTK_CELL_TYPE);
  fclose(vtk);
}

void my_p4est_cell_neighbors_t::write_cell_triangulation_vtk_0m0(p4est_locidx_t qu, p4est_topidx_t tr, const char *filename)
{
  p4est_connectivity_t *conn = p4est->connectivity;
  p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tr);
  const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, qu);
  p4est_locidx_t q = qu + tree->quadrants_offset;

  char vtkname[1024];
  sprintf(vtkname, "%s_%04d.%d.vtk", filename, p4est->mpirank, q);
  FILE *vtk = fopen(vtkname, "w");

  fprintf(vtk, "# vtk DataFile Version 2.0 \n");
  fprintf(vtk, "Triangulation \n");
  fprintf(vtk, "ASCII \n");
  fprintf(vtk, "DATASET UNSTRUCTURED_GRID \n");

  const quad_info_t *cells[P4EST_FACES + 1];
  for (short i = 0; i<P4EST_FACES; i++)
    cells[i] = this->face_begin(q, i);
  cells[P4EST_FACES] = this->face_end(q, P4EST_FACES - 1);

  const quad_info_t *begin_m00 = cells[dir::f_m00  ];
  const quad_info_t *end_m00   = cells[dir::f_m00+1]; size_t n_m00 = end_m00 - begin_m00;
  const quad_info_t *begin_p00 = cells[dir::f_p00  ];
  const quad_info_t *end_p00   = cells[dir::f_p00+1]; size_t n_p00 = end_p00 - begin_p00;
  const quad_info_t *begin_0m0 = cells[dir::f_0m0  ];
  const quad_info_t *end_0m0   = cells[dir::f_0m0+1]; size_t n_0m0 = end_0m0 - begin_0m0;
  const quad_info_t *begin_0p0 = cells[dir::f_0p0  ];
  const quad_info_t *end_0p0   = cells[dir::f_0p0+1]; size_t n_0p0 = end_0p0 - begin_0p0;
  const quad_info_t *begin_00m = cells[dir::f_00m  ];
  const quad_info_t *end_00m   = cells[dir::f_00m+1]; size_t n_00m = end_00m - begin_00m;
  const quad_info_t *begin_00p = cells[dir::f_00p  ];
  const quad_info_t *end_00p   = cells[dir::f_00p+1]; size_t n_00p = end_00p - begin_00p;

  size_t num_points = cells[P4EST_FACES]-cells[0] + 1;
  fprintf(vtk, "POINTS %ld double\n", num_points);

  std::vector<p4est_locidx_t> elements;
  elements.reserve(4*num_points); // worst-case estimate

  std::map<p4est_locidx_t, int> map;

  int c = 0;
  /* write the center location of the central point */
  {
    double qh = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;

    p4est_topidx_t v_mm = conn->tree_to_vertex[P4EST_CHILDREN*tr];
    double tree_xmin = conn->vertices[3*v_mm + 0];
    double tree_ymin = conn->vertices[3*v_mm + 1];
    double tree_zmin = conn->vertices[3*v_mm + 2];

    double x  = quad_x_fr_i(quad) + 0.5*qh + tree_xmin;
    double y  = quad_y_fr_j(quad) + 0.5*qh + tree_ymin;
    double z  = quad_z_fr_k(quad) + 0.5*qh + tree_zmin;

    fprintf(vtk, "%lf %lf %lf\n", x, y, z);
    map.insert(std::make_pair(q,c++));
  }

  /* write center location and create mapping */
  for (const quad_info_t *it = cells[0]; it != cells[P4EST_FACES]; ++it)
  {
    double qh = (double)P4EST_QUADRANT_LEN(it->quad->level)/(double)P4EST_ROOT_LEN;

    p4est_topidx_t v_mm = conn->tree_to_vertex[P4EST_CHILDREN*it->tree_idx];
    double tree_xmin = conn->vertices[3*v_mm + 0];
    double tree_ymin = conn->vertices[3*v_mm + 1];
    double tree_zmin = conn->vertices[3*v_mm + 2];

    double x  = quad_x_fr_i(it->quad) + 0.5*qh + tree_xmin;
    double y  = quad_y_fr_j(it->quad) + 0.5*qh + tree_ymin;
    double z  = quad_z_fr_k(it->quad) + 0.5*qh + tree_zmin;

    fprintf(vtk, "%lf %lf %lf\n", x, y, z);
    map.insert(std::make_pair(it->locidx, c++));
  }

  /* loop over all cells and find neighbors in the p00, 0p0, and pp0 directions */
  for (const quad_info_t *it = begin_0m0; it != end_0m0; ++it)
  {
    std::vector<p4est_locidx_t> ng_p00;
    std::vector<p4est_locidx_t> ng_00p;
    int8_t l_p00, l_00p;

    bool is_boundary_p00 = it->quad->x + P4EST_QUADRANT_LEN(it->quad->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level);
    bool is_boundary_00p = it->quad->z + P4EST_QUADRANT_LEN(it->quad->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level);

    /* p00 */
    if (!is_boundary_p00)
    {
      const quad_info_t *begin_ = this->face_begin(it->locidx, dir::f_p00);
      const quad_info_t *end_   = this->face_end(it->locidx, dir::f_p00);
      ng_p00.reserve(end_ - begin_);
      for (const quad_info_t *it_ = begin_; it_ != end_; ++it_)
        if (it_->quad->y + P4EST_QUADRANT_LEN(it_->quad->level) == it->quad->y + P4EST_QUADRANT_LEN(it->quad->level) ){
          ng_p00.push_back(it_->locidx);
          l_p00 = it_->quad->level;
        }
    }

    /* 0p0 */
    if (!is_boundary_00p)
    {
      const quad_info_t *begin_ = this->face_begin(it->locidx, dir::f_00p);
      const quad_info_t *end_   = this->face_end(it->locidx, dir::f_00p);
      ng_00p.reserve(end_ - begin_);
      for (const quad_info_t *it_ = begin_; it_ != end_; ++it_)
        if (it_->quad->y + P4EST_QUADRANT_LEN(it_->quad->level) == it->quad->y + P4EST_QUADRANT_LEN(it->quad->level)){
          ng_00p.push_back(it_->locidx);
          l_00p = it_->quad->level;
        }
    }

    /* pp0 */
    if (!is_boundary_p00 && !is_boundary_00p)
    {
      const quad_info_t *it_;
      if (l_p00 > l_00p)
        it_ = this->face_begin(ng_p00[ng_p00.size() - 1], dir::f_00p);
      else
        it_ = this->face_begin(ng_00p[ng_00p.size() - 1], dir::f_p00);

      ng_p00.push_back(it_->locidx);
      ng_00p.push_back(it_->locidx);
    }

    /* now that we have all cells, construct the elements */
    if (ng_p00.size() != 0)
      for (size_t i=0; i<ng_p00.size()-1; i++){
        elements.push_back(q);
        elements.push_back(it->locidx);
        elements.push_back(ng_p00[i]);
        elements.push_back(ng_p00[i+1]);
      }

    if (ng_00p.size() != 0)
      for (size_t i=0; i<ng_00p.size()-1; i++){
        elements.push_back(q);
        elements.push_back(it->locidx);
        elements.push_back(ng_00p[i]);
        elements.push_back(ng_00p[i+1]);
      }
  }

  /* now we do stiching! */
  /* 1 - across m00 direction */
  if (n_0m0 != 0) {
    {
      /* first locate top cells along the edge */
      std::vector<p4est_locidx_t> ng1;
      ng1.reserve(end_0m0 - begin_0m0);
      for (const quad_info_t *it = begin_0m0; it != end_0m0; ++it)
        if (it->quad->x <= quad->x){
          ng1.push_back(it->locidx);
        }

      std::vector<p4est_locidx_t> ng2;
      ng2.reserve(end_m00 - begin_m00);
      for (const quad_info_t *it = begin_m00; it != end_m00; ++it)
        if (it->quad->y <= quad->y){
          ng2.push_back(it->locidx);
        }

      /* now construct the triangulation */
      if (ng1.size() != 0 && ng2.size() != 0){
        if (ng1.size() > ng2.size())
          std::swap(ng1, ng2);

        // forward
        int s = (ng2.size()-1) / ng1.size();
        int r = (ng2.size()-1) % ng1.size();
        int c = 0;
        std::vector<int> ng2s; ng2s.reserve(ng1.size() - 1);
        if (ng2.size() > 1){
          for (int i=0; i<r; i++){
            for (int j = 0; j<s+1; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s+1;
            ng2s.push_back(c);
          }
          for (size_t i = r; i<ng1.size(); i++){
            for (int j = 0; j<s; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s;
            ng2s.push_back(c);
          }
        }
        //backward
        if (ng1.size() > 1)
          for (size_t i=0; i<ng1.size() - 1; i++){
            elements.push_back(q);
            elements.push_back(ng1[i]);
            elements.push_back(ng1[i+1]);
            elements.push_back(ng2[ng2s[i]]);
          }
      }
    }
    /* 2 - across p00 direction */
    {
      /* first locate top cells along the edge */
      std::vector<p4est_locidx_t> ng1;
      ng1.reserve(end_0m0 - begin_0m0);
      for (const quad_info_t *it = begin_0m0; it != end_0m0; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->quad->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level)){
          ng1.push_back(it->locidx);
        }

      std::vector<p4est_locidx_t> ng2;
      ng2.reserve(end_p00 - begin_p00);
      for (const quad_info_t *it = begin_p00; it != end_p00; ++it)
        if (it->quad->y <= quad->y){
          ng2.push_back(it->locidx);
        }

      /* now construct the triangulation */
      if (ng1.size() != 0 && ng2.size() != 0){
        if (ng1.size() > ng2.size())
          std::swap(ng1, ng2);

        // forward
        int s = (ng2.size()-1) / ng1.size();
        int r = (ng2.size()-1) % ng1.size();
        int c = 0;
        std::vector<int> ng2s; ng2s.reserve(ng1.size() - 1);
        if (ng2.size() > 1){
          for (int i=0; i<r; i++){
            for (int j = 0; j<s+1; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s+1;
            ng2s.push_back(c);
          }
          for (size_t i = r; i<ng1.size(); i++){
            for (int j = 0; j<s; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s;
            ng2s.push_back(c);
          }
        }
        //backward
        if (ng1.size() > 1)
          for (size_t i=0; i<ng1.size() - 1; i++){
            elements.push_back(q);
            elements.push_back(ng1[i]);
            elements.push_back(ng1[i+1]);
            elements.push_back(ng2[ng2s[i]]);
          }
      }
    }
    /* 3 - across 0m0 direction */
    {
      /* first locate top cells along the edge */
      std::vector<p4est_locidx_t> ng1;
      ng1.reserve(end_0m0 - begin_0m0);
      for (const quad_info_t *it = begin_0m0; it != end_0m0; ++it)
        if (it->quad->z <= quad->z){
          ng1.push_back(it->locidx);
        }

      std::vector<p4est_locidx_t> ng2;
      ng2.reserve(end_00m- begin_00m);
      for (const quad_info_t *it = begin_00m; it != end_00m; ++it)
        if (it->quad->y <= quad->y){
          ng2.push_back(it->locidx);
        }

      /* now construct the triangulation */
      if (ng1.size() != 0 && ng2.size() != 0){
        if (ng1.size() > ng2.size())
          std::swap(ng1, ng2);

        // forward
        int s = (ng2.size()-1) / ng1.size();
        int r = (ng2.size()-1) % ng1.size();
        int c = 0;
        std::vector<int> ng2s; ng2s.reserve(ng1.size() - 1);
        if (ng2.size() > 1){
          for (int i=0; i<r; i++){
            for (int j = 0; j<s+1; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s+1;
            ng2s.push_back(c);
          }
          for (size_t i = r; i<ng1.size(); i++){
            for (int j = 0; j<s; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s;
            ng2s.push_back(c);
          }
        }
        //backward
        if (ng1.size() > 1)
          for (size_t i=0; i<ng1.size() - 1; i++){
            elements.push_back(q);
            elements.push_back(ng1[i]);
            elements.push_back(ng1[i+1]);
            elements.push_back(ng2[ng2s[i]]);
          }
      }
    }
    /* 4 - across 0p0 direction */
    {
      /* first locate top cells along the edge */
      std::vector<p4est_locidx_t> ng1;
      ng1.reserve(end_0m0 - begin_0m0);
      for (const quad_info_t *it = begin_0m0; it != end_0m0; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->quad->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level)){
          ng1.push_back(it->locidx);
        }

      std::vector<p4est_locidx_t> ng2;
      ng2.reserve(end_00p - begin_00p);
      for (const quad_info_t *it = begin_00p; it != end_00p; ++it)
        if (it->quad->y <= quad->y){
          ng2.push_back(it->locidx);
        }

      /* now construct the triangulation */
      if (ng1.size() != 0 && ng2.size() != 0){
        if (ng1.size() > ng2.size())
          std::swap(ng1, ng2);

        // forward
        int s = (ng2.size()-1) / ng1.size();
        int r = (ng2.size()-1) % ng1.size();
        int c = 0;
        std::vector<int> ng2s; ng2s.reserve(ng1.size() - 1);
        if (ng2.size() > 1){
          for (int i=0; i<r; i++){
            for (int j = 0; j<s+1; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s+1;
            ng2s.push_back(c);
          }
          for (size_t i=r; i<ng1.size(); i++){
            for (int j = 0; j<s; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s;
            ng2s.push_back(c);
          }
        }
        //backward
        if (ng1.size() > 1)
          for (size_t i=0; i<ng1.size() - 1; i++){
            elements.push_back(q);
            elements.push_back(ng1[i]);
            elements.push_back(ng1[i+1]);
            elements.push_back(ng2[ng2s[i]]);
          }
      }
    }
    /* 5 - across mm0 */

    if (n_m00 != 0 && n_00m != 0)
    {
      elements.push_back(q);
      elements.push_back(begin_m00->locidx);
      elements.push_back(begin_0m0->locidx);
      elements.push_back(begin_00m->locidx);
    }

    /* 6 - across pm0 */
    if (n_p00 != 0 && n_00m != 0)
    {
      elements.push_back(q);
      for (const quad_info_t *it = begin_0m0; it != end_0m0; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->quad->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level)){
          elements.push_back(it->locidx);
          break;
        }

      elements.push_back(begin_p00->locidx);

      for (const quad_info_t *it = begin_00m; it != end_00m; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->quad->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level)){
          elements.push_back(it->locidx);
          break;
        }
    }
    /* 7 - across mp0 */
    if (n_m00 != 0 && n_00p != 0)
    {
      elements.push_back(q);
      for (const quad_info_t *it = begin_0m0; it != end_0m0; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->quad->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level)){
          elements.push_back(it->locidx);
          break;
        }

      for (const quad_info_t *it = begin_m00; it != end_m00; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->quad->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level)){
          elements.push_back(it->locidx);
          break;
        }

      elements.push_back(begin_00p->locidx);
    }
    /* 8 - across pp0 */
    if (n_p00 != 0 && n_0p0 != 0)
    {
      elements.push_back(q);
      elements.push_back((end_0m0-1)->locidx);
      for (const quad_info_t *it = begin_p00; it != end_p00; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->quad->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level)){
          elements.push_back(it->locidx);
          break;
        }
      for (const quad_info_t *it = begin_00p; it != end_00p; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->quad->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level)){
          elements.push_back(it->locidx);
          break;
        }
    }
  }
  /* write the connectivity information (a.k.a. elements) */
  size_t num_elements = elements.size()/4;
  fprintf(vtk, "CELLS %ld %ld \n", num_elements, (1+4)*num_elements);

  for (size_t i=0; i<num_elements; ++i)
  {
    fprintf(vtk, "%d ", 4);
    for (short j = 0; j<4; j++)
      fprintf(vtk, "%d ", map[elements[4*i + j]]);
    fprintf(vtk, "\n");
  }

  fprintf(vtk, "CELL_TYPES %ld\n", num_elements);
  for (size_t i=0; i<num_elements; ++i)
    fprintf(vtk, "%d\n", VTK_CELL_TYPE);
  fclose(vtk);
}

void my_p4est_cell_neighbors_t::write_cell_triangulation_vtk_0p0(p4est_locidx_t qu, p4est_topidx_t tr, const char *filename)
{
  p4est_connectivity_t *conn = p4est->connectivity;
  p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tr);
  const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, qu);
  p4est_locidx_t q = qu + tree->quadrants_offset;

  char vtkname[1024];
  sprintf(vtkname, "%s_%04d.%d.vtk", filename, p4est->mpirank, q);
  FILE *vtk = fopen(vtkname, "w");

  fprintf(vtk, "# vtk DataFile Version 2.0 \n");
  fprintf(vtk, "Triangulation \n");
  fprintf(vtk, "ASCII \n");
  fprintf(vtk, "DATASET UNSTRUCTURED_GRID \n");

  const quad_info_t *cells[P4EST_FACES + 1];
  for (short i = 0; i<P4EST_FACES; i++)
    cells[i] = this->face_begin(q, i);
  cells[P4EST_FACES] = this->face_end(q, P4EST_FACES - 1);

  const quad_info_t *begin_m00 = cells[dir::f_m00  ];
  const quad_info_t *end_m00   = cells[dir::f_m00+1]; size_t n_m00 = end_m00 - begin_m00;
  const quad_info_t *begin_p00 = cells[dir::f_p00  ];
  const quad_info_t *end_p00   = cells[dir::f_p00+1]; size_t n_p00 = end_p00 - begin_p00;
  const quad_info_t *begin_0p0 = cells[dir::f_0p0  ];
  const quad_info_t *end_0p0   = cells[dir::f_0p0+1]; size_t n_0p0 = end_0p0 - begin_0p0;
  const quad_info_t *begin_00m = cells[dir::f_00m  ];
  const quad_info_t *end_00m   = cells[dir::f_00m+1]; size_t n_00m = end_00m - begin_00m;
  const quad_info_t *begin_00p = cells[dir::f_00p  ];
  const quad_info_t *end_00p   = cells[dir::f_00p+1]; size_t n_00p = end_00p - begin_00p;

  size_t num_points = cells[P4EST_FACES]-cells[0] + 1;
  fprintf(vtk, "POINTS %ld double\n", num_points);

  std::vector<p4est_locidx_t> elements;
  elements.reserve(4*num_points); // worst-case estimate

  std::map<p4est_locidx_t, int> map;

  int c = 0;
  /* write the center location of the central point */
  {
    double qh = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;

    p4est_topidx_t v_mm = conn->tree_to_vertex[P4EST_CHILDREN*tr];
    double tree_xmin = conn->vertices[3*v_mm + 0];
    double tree_ymin = conn->vertices[3*v_mm + 1];
    double tree_zmin = conn->vertices[3*v_mm + 2];

    double x  = quad_x_fr_i(quad) + 0.5*qh + tree_xmin;
    double y  = quad_y_fr_j(quad) + 0.5*qh + tree_ymin;
    double z  = quad_z_fr_k(quad) + 0.5*qh + tree_zmin;

    fprintf(vtk, "%lf %lf %lf\n", x, y, z);
    map.insert(std::make_pair(q,c++));
  }

  /* write center location and create mapping */
  for (const quad_info_t *it = cells[0]; it != cells[P4EST_FACES]; ++it)
  {
    double qh = (double)P4EST_QUADRANT_LEN(it->quad->level)/(double)P4EST_ROOT_LEN;

    p4est_topidx_t v_mm = conn->tree_to_vertex[P4EST_CHILDREN*it->tree_idx];
    double tree_xmin = conn->vertices[3*v_mm + 0];
    double tree_ymin = conn->vertices[3*v_mm + 1];
    double tree_zmin = conn->vertices[3*v_mm + 2];

    double x  = quad_x_fr_i(it->quad) + 0.5*qh + tree_xmin;
    double y  = quad_y_fr_j(it->quad) + 0.5*qh + tree_ymin;
    double z  = quad_z_fr_k(it->quad) + 0.5*qh + tree_zmin;

    fprintf(vtk, "%lf %lf %lf\n", x, y, z);
    map.insert(std::make_pair(it->locidx, c++));
  }

  /* loop over all cells and find neighbors in the p00, 0p0, and pp0 directions */
  for (const quad_info_t *it = begin_0p0; it != end_0p0; ++it)
  {
    std::vector<p4est_locidx_t> ng_p00;
    std::vector<p4est_locidx_t> ng_00p;
    int8_t l_p00, l_00p;

    bool is_boundary_p00 = it->quad->x + P4EST_QUADRANT_LEN(it->quad->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level);
    bool is_boundary_00p = it->quad->z + P4EST_QUADRANT_LEN(it->quad->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level);

    /* p00 */
    if (!is_boundary_p00)
    {
      const quad_info_t *begin_ = this->face_begin(it->locidx, dir::f_p00);
      const quad_info_t *end_   = this->face_end(it->locidx, dir::f_p00);
      ng_p00.reserve(end_ - begin_);
      for (const quad_info_t *it_ = begin_; it_ != end_; ++it_)
        if (it_->quad->y == it->quad->y ){
          ng_p00.push_back(it_->locidx);
          l_p00 = it_->quad->level;
        }
    }

    /* 0p0 */
    if (!is_boundary_00p)
    {
      const quad_info_t *begin_ = this->face_begin(it->locidx, dir::f_00p);
      const quad_info_t *end_   = this->face_end(it->locidx, dir::f_00p);
      ng_00p.reserve(end_ - begin_);
      for (const quad_info_t *it_ = begin_; it_ != end_; ++it_)
        if (it_->quad->y == it->quad->y){
          ng_00p.push_back(it_->locidx);
          l_00p = it_->quad->level;
        }
    }

    /* pp0 */
    if (!is_boundary_p00 && !is_boundary_00p)
    {
      const quad_info_t *it_;
      if (l_p00 > l_00p)
        it_ = this->face_begin(ng_p00[ng_p00.size() - 1], dir::f_00p);
      else
        it_ = this->face_begin(ng_00p[ng_00p.size() - 1], dir::f_p00);

      ng_p00.push_back(it_->locidx);
      ng_00p.push_back(it_->locidx);
    }

    /* now that we have all cells, construct the elements */
    if (ng_p00.size() != 0)
      for (size_t i=0; i<ng_p00.size()-1; i++){
        elements.push_back(q);
        elements.push_back(it->locidx);
        elements.push_back(ng_p00[i]);
        elements.push_back(ng_p00[i+1]);
      }

    if (ng_00p.size() != 0)
      for (size_t i=0; i<ng_00p.size()-1; i++){
        elements.push_back(q);
        elements.push_back(it->locidx);
        elements.push_back(ng_00p[i]);
        elements.push_back(ng_00p[i+1]);
      }
  }

  /* now we do stiching! */
  /* 1 - across m00 direction */
  if (n_0p0 != 0) {
    {
      /* first locate top cells along the edge */
      std::vector<p4est_locidx_t> ng1;
      ng1.reserve(end_0p0 - begin_0p0);
      for (const quad_info_t *it = begin_0p0; it != end_0p0; ++it)
        if (it->quad->x <= quad->x){
          ng1.push_back(it->locidx);
        }

      std::vector<p4est_locidx_t> ng2;
      ng2.reserve(end_m00 - begin_m00);
      for (const quad_info_t *it = begin_m00; it != end_m00; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->quad->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level)){
          ng2.push_back(it->locidx);
        }

      /* now construct the triangulation */
      if (ng1.size() != 0 && ng2.size() != 0){
        if (ng1.size() > ng2.size())
          std::swap(ng1, ng2);

        // forward
        int s = (ng2.size()-1) / ng1.size();
        int r = (ng2.size()-1) % ng1.size();
        int c = 0;
        std::vector<int> ng2s; ng2s.reserve(ng1.size() - 1);
        if (ng2.size() > 1){
          for (int i=0; i<r; i++){
            for (int j = 0; j<s+1; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s+1;
            ng2s.push_back(c);
          }
          for (size_t i = r; i<ng1.size(); i++){
            for (int j = 0; j<s; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s;
            ng2s.push_back(c);
          }
        }
        //backward
        if (ng1.size() > 1)
          for (size_t i=0; i<ng1.size() - 1; i++){
            elements.push_back(q);
            elements.push_back(ng1[i]);
            elements.push_back(ng1[i+1]);
            elements.push_back(ng2[ng2s[i]]);
          }
      }
    }
    /* 2 - across p00 direction */
    {
      /* first locate top cells along the edge */
      std::vector<p4est_locidx_t> ng1;
      ng1.reserve(end_0p0 - begin_0p0);
      for (const quad_info_t *it = begin_0p0; it != end_0p0; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->quad->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level)){
          ng1.push_back(it->locidx);
        }

      std::vector<p4est_locidx_t> ng2;
      ng2.reserve(end_p00 - begin_p00);
      for (const quad_info_t *it = begin_p00; it != end_p00; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->quad->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level)){
          ng2.push_back(it->locidx);
        }

      /* now construct the triangulation */
      if (ng1.size() != 0 && ng2.size() != 0){
        if (ng1.size() > ng2.size())
          std::swap(ng1, ng2);

        // forward
        int s = (ng2.size()-1) / ng1.size();
        int r = (ng2.size()-1) % ng1.size();
        int c = 0;
        std::vector<int> ng2s; ng2s.reserve(ng1.size() - 1);
        if (ng2.size() > 1){
          for (int i=0; i<r; i++){
            for (int j = 0; j<s+1; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s+1;
            ng2s.push_back(c);
          }
          for (size_t i = r; i<ng1.size(); i++){
            for (int j = 0; j<s; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s;
            ng2s.push_back(c);
          }
        }
        //backward
        if (ng1.size() > 1)
          for (size_t i=0; i<ng1.size() - 1; i++){
            elements.push_back(q);
            elements.push_back(ng1[i]);
            elements.push_back(ng1[i+1]);
            elements.push_back(ng2[ng2s[i]]);
          }
      }
    }
    /* 3 - across 0m0 direction */
    {
      /* first locate top cells along the edge */
      std::vector<p4est_locidx_t> ng1;
      ng1.reserve(end_0p0 - begin_0p0);
      for (const quad_info_t *it = begin_0p0; it != end_0p0; ++it)
        if (it->quad->z <= quad->z){
          ng1.push_back(it->locidx);
        }

      std::vector<p4est_locidx_t> ng2;
      ng2.reserve(end_00m- begin_00m);
      for (const quad_info_t *it = begin_00m; it != end_00m; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->quad->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level)){
          ng2.push_back(it->locidx);
        }

      /* now construct the triangulation */
      if (ng1.size() != 0 && ng2.size() != 0){
        if (ng1.size() > ng2.size())
          std::swap(ng1, ng2);

        // forward
        int s = (ng2.size()-1) / ng1.size();
        int r = (ng2.size()-1) % ng1.size();
        int c = 0;
        std::vector<int> ng2s; ng2s.reserve(ng1.size() - 1);
        if (ng2.size() > 1){
          for (int i=0; i<r; i++){
            for (int j = 0; j<s+1; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s+1;
            ng2s.push_back(c);
          }
          for (size_t i = r; i<ng1.size(); i++){
            for (int j = 0; j<s; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s;
            ng2s.push_back(c);
          }
        }
        //backward
        if (ng1.size() > 1)
          for (size_t i=0; i<ng1.size() - 1; i++){
            elements.push_back(q);
            elements.push_back(ng1[i]);
            elements.push_back(ng1[i+1]);
            elements.push_back(ng2[ng2s[i]]);
          }
      }
    }
    /* 4 - across 0p0 direction */
    {
      /* first locate top cells along the edge */
      std::vector<p4est_locidx_t> ng1;
      ng1.reserve(end_0p0 - begin_0p0);
      for (const quad_info_t *it = begin_0p0; it != end_0p0; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->quad->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level)){
          ng1.push_back(it->locidx);
        }

      std::vector<p4est_locidx_t> ng2;
      ng2.reserve(end_00p - begin_00p);
      for (const quad_info_t *it = begin_00p; it != end_00p; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->quad->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level)){
          ng2.push_back(it->locidx);
        }

      /* now construct the triangulation */
      if (ng1.size() != 0 && ng2.size() != 0){
        if (ng1.size() > ng2.size())
          std::swap(ng1, ng2);

        // forward
        int s = (ng2.size()-1) / ng1.size();
        int r = (ng2.size()-1) % ng1.size();
        int c = 0;
        std::vector<int> ng2s; ng2s.reserve(ng1.size() - 1);
        if (ng2.size() > 1){
          for (int i=0; i<r; i++){
            for (int j = 0; j<s+1; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s+1;
            ng2s.push_back(c);
          }
          for (size_t i=r; i<ng1.size(); i++){
            for (int j = 0; j<s; j++){
              elements.push_back(q);
              elements.push_back(ng1[i]);
              elements.push_back(ng2[c+j]);
              elements.push_back(ng2[c+j+1]);
            }
            c += s;
            ng2s.push_back(c);
          }
        }
        //backward
        if (ng1.size() > 1)
          for (size_t i=0; i<ng1.size() - 1; i++){
            elements.push_back(q);
            elements.push_back(ng1[i]);
            elements.push_back(ng1[i+1]);
            elements.push_back(ng2[ng2s[i]]);
          }
      }
    }
    /* 5 - across mm0 */

    if (n_m00 != 0 && n_00m != 0)
    {
      elements.push_back(q);
      for (const quad_info_t *it = begin_m00; it != end_m00; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->quad->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level)){
          elements.push_back(it->locidx);
          break;
        }

      elements.push_back(begin_0p0->locidx);

      for (const quad_info_t *it = begin_00m; it != end_00m; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->quad->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level)){
          elements.push_back(it->locidx);
          break;
        }
    }

    /* 6 - across pm0 */
    if (n_p00 != 0 && n_00m != 0)
    {
      elements.push_back(q);
      for (const quad_info_t *it = begin_p00; it != end_p00; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->quad->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level)){
          elements.push_back(it->locidx);
          break;
        }

      for (const quad_info_t *it = begin_0p0; it != end_0p0; ++it)
        if (it->quad->x + P4EST_QUADRANT_LEN(it->quad->level) >= quad->x + P4EST_QUADRANT_LEN(quad->level)){
          elements.push_back(it->locidx);
          break;
        }
      elements.push_back((end_00m-1)->locidx);
    }
    /* 7 - across mp0 */
    if (n_m00 != 0 && n_00p != 0)
    {
      elements.push_back(q);
      elements.push_back((end_m00-1)->locidx);

      for (const quad_info_t *it = begin_0p0; it != end_0p0; ++it)
        if (it->quad->z + P4EST_QUADRANT_LEN(it->quad->level) >= quad->z + P4EST_QUADRANT_LEN(quad->level)){
          elements.push_back(it->locidx);
          break;
        }

      for (const quad_info_t *it = begin_00p; it != end_00p; ++it)
        if (it->quad->y + P4EST_QUADRANT_LEN(it->quad->level) >= quad->y + P4EST_QUADRANT_LEN(quad->level)){
          elements.push_back(it->locidx);
          break;
        }
    }
    /* 8 - across pp0 */
    if (n_p00 != 0 && n_00p != 0)
    {
      elements.push_back(q);
      elements.push_back((end_p00-1)->locidx);
      elements.push_back((end_0p0-1)->locidx);
      elements.push_back((end_00p-1)->locidx);
    }
  }
  /* write the connectivity information (a.k.a. elements) */
  size_t num_elements = elements.size()/4;
  fprintf(vtk, "CELLS %ld %ld \n", num_elements, (1+4)*num_elements);

  for (size_t i=0; i<num_elements; ++i)
  {
    fprintf(vtk, "%d ", 4);
    for (short j = 0; j<4; j++)
      fprintf(vtk, "%d ", map[elements[4*i + j]]);
    fprintf(vtk, "\n");
  }

  fprintf(vtk, "CELL_TYPES %ld\n", num_elements);
  for (size_t i=0; i<num_elements; ++i)
    fprintf(vtk, "%d\n", VTK_CELL_TYPE);
  fclose(vtk);
}
#endif

#ifndef P4_TO_P8
void my_p4est_cell_neighbors_t::write_triangulation(const char *filename)
{
  p4est_connectivity_t *conn = p4est->connectivity;

  char vtkname[1024];
  sprintf(vtkname, "%s_%04d.vtk", filename, p4est->mpirank);

  FILE *vtk = fopen(vtkname, "w");

  fprintf(vtk, "# vtk DataFile Version 2.0 \n");
  fprintf(vtk, "Triangulation \n");
  fprintf(vtk, "ASCII \n");
  fprintf(vtk, "DATASET UNSTRUCTURED_GRID \n");
  fprintf(vtk, "POINTS %ld double \n", p4est->local_num_quadrants + ghost->ghosts.elem_count);

  /* create triangulations */
  std::vector<p4est_locidx_t> elements;
  elements.reserve(3*p4est->local_num_quadrants/2); // estimate

  for (p4est_topidx_t tr_id = p4est->first_local_tree; tr_id <= p4est->last_local_tree; ++tr_id){
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tr_id);

    p4est_topidx_t v_mm = conn->tree_to_vertex[P4EST_CHILDREN*tr_id];
    double tree_xmin = conn->vertices[3*v_mm + 0];
    double tree_ymin = conn->vertices[3*v_mm + 1];

    for (size_t q = 0; q < tree->quadrants.elem_count; ++q){
      const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
      p4est_locidx_t q_locidx = q + tree->quadrants_offset;

      /* compute the coordinates for the vertices */
      double qh = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
      double x  = quad_x_fr_i(quad) + 0.5*qh + tree_xmin;
      double y  = quad_y_fr_j(quad) + 0.5*qh + tree_ymin;
      fprintf(vtk, "%lf %lf 0.0\n", x, y);

      /* construct the elemets */
      /* all the cells in the p0 direction */
      if (!is_quad_xpWall(p4est, tr_id, quad)){
        const quad_info_t *begin = this->face_begin(q_locidx, dir::f_p00);
        const quad_info_t *end   = this->face_end(q_locidx, dir::f_p00);

        for (const quad_info_t* it = begin; it != end - 1; ++it){
          elements.push_back(q_locidx);
          elements.push_back(it->locidx);
          elements.push_back((it+1)->locidx);
        }
      }

      /* all the cells in the 0p direction */
      if (!is_quad_ypWall(p4est, tr_id, quad)){
        const quad_info_t *begin = this->face_begin(q_locidx, dir::f_0p0);
        const quad_info_t *end   = this->face_end(q_locidx, dir::f_0p0);

        for (const quad_info_t* it = begin; it != end - 1; ++it){
          elements.push_back(q_locidx);
          elements.push_back(it->locidx);
          elements.push_back((it+1)->locidx);
        }
      }

      /* corner in the pp direction */
      if (!is_quad_xpWall(p4est, tr_id, quad) && !is_quad_ypWall(p4est, tr_id, quad)){
        const quad_info_t *it_p0  = this->face_begin(q_locidx, dir::f_0m0) - 1;
        const quad_info_t *it_0p  = this->face_end(q_locidx, dir::f_0p0) - 1;
        const quad_info_t *it_pp;
        if (it_p0->quad->level > it_0p->quad->level)
          it_pp = this->face_begin(it_p0->locidx, dir::f_0p0);
        else
          it_pp = this->face_begin(it_0p->locidx, dir::f_p00);

        if (it_pp != it_p0){
          elements.push_back(q_locidx);
          elements.push_back(it_p0->locidx);
          elements.push_back(it_pp->locidx);
        }

        if (it_pp != it_0p){
          elements.push_back(q_locidx);
          elements.push_back(it_0p->locidx);
          elements.push_back(it_pp->locidx);
        }
      }
    }
  }

  /* write the point information for the ghost cells */
  for (size_t g = 0; g < ghost->ghosts.elem_count; ++g){
    const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&ghost->ghosts, g);

    p4est_topidx_t v_mm = conn->tree_to_vertex[P4EST_CHILDREN*quad->p.piggy3.which_tree];
    double tree_xmin = conn->vertices[3*v_mm + 0];
    double tree_ymin = conn->vertices[3*v_mm + 1];

    /* compute the coordinates for the vertices */
    double qh = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
    double x  = quad_x_fr_i(quad) + 0.5*qh + tree_xmin;
    double y  = quad_y_fr_j(quad) + 0.5*qh + tree_ymin;
    fprintf(vtk, "%lf %lf 0.0\n", x, y);
  }

  /* write the connectivity information (a.k.a. elements */
  size_t num_elements = elements.size()/3;
  fprintf(vtk, "CELLS %ld %ld \n", num_elements, (1+3)*num_elements);

  for (size_t i=0; i<num_elements; ++i)
  {
    fprintf(vtk, "%d ", 3);
    fprintf(vtk, "%d ", elements[3*i + 0]);
    fprintf(vtk, "%d ", elements[3*i + 1]);
    fprintf(vtk, "%d ", elements[3*i + 2]);
    fprintf(vtk, "\n");
  }

  fprintf(vtk, "CELL_TYPES %ld\n", num_elements);
  for (size_t i=0; i<num_elements; ++i)
    fprintf(vtk, "%d\n",VTK_CELL_TYPE);
  fclose(vtk);
}
#endif
