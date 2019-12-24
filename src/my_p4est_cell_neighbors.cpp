#ifdef P4_TO_P8
#include "my_p8est_cell_neighbors.h"
#else
#include "my_p4est_cell_neighbors.h"
#endif


void my_p4est_cell_neighbors_t::find_neighbor_cells_of_cell(std::vector<p4est_quadrant_t>& ngbd, p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, DIM(char dir_x, char dir_y, char dir_z) ) const
{
  const p4est_quadrant_t *quad;
  if(quad_idx<p4est->local_num_quadrants)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);
  }
  else
    quad = (const p4est_quadrant_t*)sc_array_index(&ghost->ghosts, quad_idx-p4est->local_num_quadrants);

  /* construct the coordinate of the neighbor cell of the same size in the given direction */
  p4est_qcoord_t size = P4EST_QUADRANT_LEN(quad->level);

  p4est_topidx_t nb_tree_idx = tree_idx;
  p4est_qcoord_t i_nb = quad->x + ( dir_x == -1 ? -size : ( dir_x == 1 ? size : 0) );
  p4est_qcoord_t j_nb = quad->y + ( dir_y == -1 ? -size : ( dir_y == 1 ? size : 0) );
#ifdef P4_TO_P8
  p4est_qcoord_t k_nb = quad->z + ( dir_z == -1 ? -size : ( dir_z == 1 ? size : 0) );
#endif

  bool px = is_periodic(p4est, 0);
  bool py = is_periodic(p4est, 1);
#ifdef P4_TO_P8
  bool pz = is_periodic(p4est, 2);
#endif

  /* check if quadrant is on a boundary */
#ifdef P4_TO_P8
  if     (quad->x==0 && dir_x==-1 && quad->y==0 && dir_y==-1 && quad->z==0 && dir_z==-1)
  {
    if((!px && (myb->nxyztrees[0]==1)) || (!py && (myb->nxyztrees[1]==1)) || (!pz && (myb->nxyztrees[2]==1))) return;
    p4est_topidx_t corner = p4est->connectivity->tree_to_corner[P4EST_CHILDREN*tree_idx + dir::v_mmm];
    if(corner==-1) return;
    p4est_topidx_t offset = p4est->connectivity->ctt_offset[corner];
    nb_tree_idx = p4est->connectivity->corner_to_tree[offset + dir::v_mmm];
    i_nb = P4EST_ROOT_LEN - size;
    j_nb = P4EST_ROOT_LEN - size;
    k_nb = P4EST_ROOT_LEN - size;
  }
  else if(quad->x+size==P4EST_ROOT_LEN && dir_x==1 && quad->y==0 && dir_y==-1 && quad->z==0 && dir_z==-1)
  {
    if((!px && (myb->nxyztrees[0]==1)) || (!py && (myb->nxyztrees[1]==1)) || (!pz && (myb->nxyztrees[2]==1))) return;
    p4est_topidx_t corner = p4est->connectivity->tree_to_corner[P4EST_CHILDREN*tree_idx + dir::v_pmm];
    if(corner==-1) return;
    p4est_topidx_t offset = p4est->connectivity->ctt_offset[corner];
    nb_tree_idx = p4est->connectivity->corner_to_tree[offset + dir::v_pmm];
    i_nb = 0;
    j_nb = P4EST_ROOT_LEN - size;
    k_nb = P4EST_ROOT_LEN - size;
  }
  else if(quad->x==0 && dir_x==-1 && quad->y+size==P4EST_ROOT_LEN && dir_y==1 && quad->z==0 && dir_z==-1)
  {
    if((!px && (myb->nxyztrees[0]==1)) || (!py && (myb->nxyztrees[1]==1)) || (!pz && (myb->nxyztrees[2]==1)))  return;
    p4est_topidx_t corner = p4est->connectivity->tree_to_corner[P4EST_CHILDREN*tree_idx + dir::v_mpm];
    if(corner==-1) return;
    p4est_topidx_t offset = p4est->connectivity->ctt_offset[corner];
    nb_tree_idx = p4est->connectivity->corner_to_tree[offset + dir::v_mpm];
    i_nb = P4EST_ROOT_LEN - size;
    j_nb = 0;
    k_nb = P4EST_ROOT_LEN - size;
  }
  else if(quad->x+size==P4EST_ROOT_LEN && dir_x==1 && quad->y+size==P4EST_ROOT_LEN && dir_y==1 && quad->z==0 && dir_z==-1)
  {
    if((!px && (myb->nxyztrees[0]==1)) || (!py && (myb->nxyztrees[1]==1)) || (!pz && (myb->nxyztrees[2]==1)))  return;
    p4est_topidx_t corner = p4est->connectivity->tree_to_corner[P4EST_CHILDREN*tree_idx + dir::v_ppm];
    if(corner==-1) return;
    p4est_topidx_t offset = p4est->connectivity->ctt_offset[corner];
    nb_tree_idx = p4est->connectivity->corner_to_tree[offset + dir::v_ppm];
    i_nb = 0;
    j_nb = 0;
    k_nb = P4EST_ROOT_LEN - size;
  }
  else if(quad->x==0 && dir_x==-1 && quad->y==0 && dir_y==-1 && quad->z+size==P4EST_ROOT_LEN && dir_z==1)
  {
    if((!px && (myb->nxyztrees[0]==1)) || (!py && (myb->nxyztrees[1]==1)) || (!pz && (myb->nxyztrees[2]==1)))  return;
    p4est_topidx_t corner = p4est->connectivity->tree_to_corner[P4EST_CHILDREN*tree_idx + dir::v_mmp];
    if(corner==-1) return;
    p4est_topidx_t offset = p4est->connectivity->ctt_offset[corner];
    nb_tree_idx = p4est->connectivity->corner_to_tree[offset + dir::v_mmp];
    i_nb = P4EST_ROOT_LEN - size;
    j_nb = P4EST_ROOT_LEN - size;
    k_nb = 0;
  }
  else if(quad->x+size==P4EST_ROOT_LEN && dir_x==1 && quad->y==0 && dir_y==-1 && quad->z==P4EST_ROOT_LEN && dir_z==1)
  {
    if((!px && (myb->nxyztrees[0]==1)) || (!py && (myb->nxyztrees[1]==1)) || (!pz && (myb->nxyztrees[2]==1)))  return;
    p4est_topidx_t corner = p4est->connectivity->tree_to_corner[P4EST_CHILDREN*tree_idx + dir::v_pmp];
    if(corner==-1) return;
    p4est_topidx_t offset = p4est->connectivity->ctt_offset[corner];
    nb_tree_idx = p4est->connectivity->corner_to_tree[offset + dir::v_pmp];
    i_nb = 0;
    j_nb = P4EST_ROOT_LEN - size;
    k_nb = 0;
  }
  else if(quad->x==0 && dir_x==-1 && quad->y+size==P4EST_ROOT_LEN && dir_y==1 && quad->z==P4EST_ROOT_LEN && dir_z==1)
  {
    if((!px && (myb->nxyztrees[0]==1)) || (!py && (myb->nxyztrees[1]==1)) || (!pz && (myb->nxyztrees[2]==1)))  return;
    p4est_topidx_t corner = p4est->connectivity->tree_to_corner[P4EST_CHILDREN*tree_idx + dir::v_mpp];
    if(corner==-1) return;
    p4est_topidx_t offset = p4est->connectivity->ctt_offset[corner];
    nb_tree_idx = p4est->connectivity->corner_to_tree[offset + dir::v_mpp];
    i_nb = P4EST_ROOT_LEN - size;
    j_nb = 0;
    k_nb = 0;
  }
  else if(quad->x+size==P4EST_ROOT_LEN && dir_x==1 && quad->y+size==P4EST_ROOT_LEN && dir_y==1 && quad->z==P4EST_ROOT_LEN && dir_z==1)
  {
    if((!px && (myb->nxyztrees[0]==1)) || (!py && (myb->nxyztrees[1]==1)) || (!pz && (myb->nxyztrees[2]==1)))  return;
    p4est_topidx_t corner = p4est->connectivity->tree_to_corner[P4EST_CHILDREN*tree_idx + dir::v_ppp];
    if(corner==-1) return;
    p4est_topidx_t offset = p4est->connectivity->ctt_offset[corner];
    nb_tree_idx = p4est->connectivity->corner_to_tree[offset + dir::v_ppp];
    i_nb = 0;
    j_nb = 0;
    k_nb = 0;
  }
  else if(quad->x==0 && dir_x==-1 && quad->y==0 && dir_y==-1)
#else
  if(quad->x==0 && dir_x==-1 && quad->y==0 && dir_y==-1)
#endif
  /* edges/corner directions */
  {
    p4est_topidx_t tmp_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx    + dir::f_m00];
    if(!px && tmp_idx==tree_idx) return;
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_idx + dir::f_0m0];
    if(!py && nb_tree_idx==tmp_idx) return;
    i_nb = P4EST_ROOT_LEN - size;
    j_nb = P4EST_ROOT_LEN - size;
  }
  else if(quad->x+size==P4EST_ROOT_LEN && dir_x==1 && quad->y==0 && dir_y==-1)
  {
    p4est_topidx_t tmp_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_p00];
    if(!px && tmp_idx==tree_idx) return;
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_idx + dir::f_0m0];
    if(!py && nb_tree_idx==tmp_idx) return;
    i_nb = 0;
    j_nb = P4EST_ROOT_LEN - size;
  }
  else if(quad->x==0 && dir_x==-1 && quad->y+size==P4EST_ROOT_LEN && dir_y==1)
  {
    p4est_topidx_t tmp_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_m00];
    if(!px && tmp_idx==tree_idx) return;
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_idx + dir::f_0p0];
    if(!py && nb_tree_idx==tmp_idx) return;
    i_nb = P4EST_ROOT_LEN - size;
    j_nb = 0;
  }
  else if(quad->x+size==P4EST_ROOT_LEN && dir_x==1 && quad->y+size==P4EST_ROOT_LEN && dir_y==1)
  {
    p4est_topidx_t tmp_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_p00];
    if(!px && tmp_idx==tree_idx) return;
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_idx + dir::f_0p0];
    if(!py && nb_tree_idx==tmp_idx) return;
    i_nb = 0;
    j_nb = 0;
  }
#ifdef P4_TO_P8
  else if(quad->x==0 && dir_x==-1 && quad->z==0 && dir_z==-1)
  {
    p4est_topidx_t tmp_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_m00];
    if(!px && tmp_idx==tree_idx) return;
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_idx + dir::f_00m];
    if(!pz && nb_tree_idx==tmp_idx) return;
    i_nb = P4EST_ROOT_LEN - size;
    k_nb = P4EST_ROOT_LEN - size;
  }
  else if(quad->x+size==P4EST_ROOT_LEN && dir_x==1 && quad->z==0 && dir_z==-1)
  {
    p4est_topidx_t tmp_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_p00];
    if(!px && tmp_idx==tree_idx) return;
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_idx + dir::f_00m];
    if(!pz && nb_tree_idx==tmp_idx) return;
    i_nb = 0;
    k_nb = P4EST_ROOT_LEN - size;
  }
  else if(quad->x==0 && dir_x==-1 && quad->z+size==P4EST_ROOT_LEN && dir_z==1)
  {
    p4est_topidx_t tmp_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_m00];
    if(!px && tmp_idx==tree_idx) return;
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_idx + dir::f_00p];
    if(!pz && nb_tree_idx==tmp_idx) return;
    i_nb = P4EST_ROOT_LEN - size;
    k_nb = 0;
  }
  else if(quad->x+size==P4EST_ROOT_LEN && dir_x==1 && quad->z+size==P4EST_ROOT_LEN && dir_z==1)
  {
    p4est_topidx_t tmp_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_p00];
    if(!px && tmp_idx==tree_idx) return;
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_idx + dir::f_00p];
    if(!pz && nb_tree_idx==tmp_idx) return;
    i_nb = 0;
    k_nb = 0;
  }
  else if(quad->y==0 && dir_y==-1 && quad->z==0 && dir_z==-1)
  {
    p4est_topidx_t tmp_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_0m0];
    if(!py && tmp_idx==tree_idx) return;
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_idx + dir::f_00m];
    if(!pz && nb_tree_idx==tmp_idx) return;
    j_nb = P4EST_ROOT_LEN - size;
    k_nb = P4EST_ROOT_LEN - size;
  }
  else if(quad->y+size==P4EST_ROOT_LEN && dir_y==1 && quad->z==0 && dir_z==-1)
  {
    p4est_topidx_t tmp_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_0p0];
    if(!py && tmp_idx==tree_idx) return;
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_idx + dir::f_00m];
    if(!pz && nb_tree_idx==tmp_idx) return;
    j_nb = 0;
    k_nb = P4EST_ROOT_LEN - size;
  }
  else if(quad->y==0 && dir_y==-1 && quad->z+size==P4EST_ROOT_LEN && dir_z==1)
  {
    p4est_topidx_t tmp_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_0m0];
    if(!py && tmp_idx==tree_idx) return;
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_idx + dir::f_00p];
    if(!pz && nb_tree_idx==tmp_idx) return;
    j_nb = P4EST_ROOT_LEN - size;
    k_nb = 0;
  }
  else if(quad->y+size==P4EST_ROOT_LEN && dir_y==1 && quad->z+size==P4EST_ROOT_LEN && dir_z==1)
  {
    p4est_topidx_t tmp_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_0p0];
    if(!py && tmp_idx==tree_idx) return;
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tmp_idx + dir::f_00p];
    if(!pz && nb_tree_idx==tmp_idx) return;
    j_nb = 0;
    k_nb = 0;
  }
#endif
  /* faces directions */
  else if(quad->x == 0 && dir_x == -1)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_m00];
    if(!px && nb_tree_idx == tree_idx) return;
    i_nb = P4EST_ROOT_LEN - size;
  }
  else if(quad->x+size == P4EST_ROOT_LEN && dir_x == 1)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_p00];
    if(!px && nb_tree_idx == tree_idx) return;
    i_nb = 0;
  }
  else if(quad->y == 0 && dir_y == -1)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_0m0];
    if(!py && nb_tree_idx == tree_idx) return;
    j_nb = P4EST_ROOT_LEN - size;
  }
  else if(quad->y+size == P4EST_ROOT_LEN && dir_y == 1)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_0p0];
    if(!py && nb_tree_idx == tree_idx) return;
    j_nb = 0;
  }
#ifdef P4_TO_P8
  else if(quad->z == 0 && dir_z == -1)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_00m];
    if(!pz && nb_tree_idx == tree_idx) return;
    k_nb = P4EST_ROOT_LEN - size;
  }
  else if(quad->z+size == P4EST_ROOT_LEN && dir_z == 1)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_00p];
    if(!pz && nb_tree_idx == tree_idx) return;
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
  find_neighbor_cells_of_cell_recursive( ngbd, nb_tree_idx, ind, DIM(dir_x, dir_y, dir_z) );
}

void my_p4est_cell_neighbors_t::find_neighbor_cells_of_cell_recursive( std::vector<p4est_quadrant_t>& ngbd, p4est_topidx_t tr, int ind, DIM(char dir_x, char dir_y, char dir_z) ) const
{
  if (hierarchy->trees[tr][ind].child == CELL_LEAF)
  {
    if (hierarchy->trees[tr][ind].quad == NOT_A_P4EST_QUADRANT) return;

    p4est_locidx_t locid = hierarchy->trees[tr][ind].quad;

    for(unsigned int n=0; n<ngbd.size(); ++n)
      if(ngbd[n].p.piggy3.local_num==locid)
        return;

    p4est_quadrant_t quad;
    if (locid < p4est->local_num_quadrants) {// local quadrant
      p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tr);
      quad = *(const p4est_quadrant_t*)sc_array_index(&tree->quadrants, locid - tree->quadrants_offset);
    } else {
      quad = *(const p4est_quadrant_t*)sc_array_index(&ghost->ghosts, locid - p4est->local_num_quadrants);
    }
    quad.p.piggy3.local_num = locid;
    quad.p.piggy3.which_tree = tr;
    ngbd.push_back(quad);

    return;
  }

#ifdef P4_TO_P8
  /* corner directions */
  if     (dir_x==-1 && dir_y==-1 && dir_z==-1) find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_ppp, -1, -1, -1);
  else if(dir_x== 1 && dir_y==-1 && dir_z==-1) find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_mpp,  1, -1, -1);
  else if(dir_x==-1 && dir_y== 1 && dir_z==-1) find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_pmp, -1,  1, -1);
  else if(dir_x== 1 && dir_y== 1 && dir_z==-1) find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_mmp,  1,  1, -1);
  else if(dir_x==-1 && dir_y==-1 && dir_z== 1) find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_ppm, -1, -1,  1);
  else if(dir_x== 1 && dir_y==-1 && dir_z== 1) find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_mpm,  1, -1,  1);
  else if(dir_x==-1 && dir_y== 1 && dir_z== 1) find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_pmm, -1,  1,  1);
  else if(dir_x== 1 && dir_y== 1 && dir_z== 1) find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_mmm,  1,  1,  1);
  else if(dir_x==-1 && dir_y==-1)
#else
  if     (dir_x==-1 && dir_y==-1)
#endif
  /* edges/corner directions */
  {
#ifdef P4_TO_P8
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_ppm, -1, -1,  0);
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_ppp, -1, -1,  0);
#else
    find_neighbor_cells_of_cell_recursive(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_ppm, -1, -1);
#endif
  }
  else if(dir_x== 1 && dir_y==-1)
  {
#ifdef P4_TO_P8
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_mpm,  1, -1,  0);
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_mpp,  1, -1,  0);
#else
    find_neighbor_cells_of_cell_recursive(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_mpm,  1, -1);
#endif
  }
  else if(dir_x==-1 && dir_y== 1)
  {
#ifdef P4_TO_P8
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_pmm, -1,  1,  0);
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_pmp, -1,  1,  0);
#else
    find_neighbor_cells_of_cell_recursive(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_pmm, -1,  1);
#endif
  }
  else if(dir_x== 1 && dir_y== 1)
  {
#ifdef P4_TO_P8
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_mmm,  1,  1,  0);
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_mmp,  1,  1,  0);
#else
    find_neighbor_cells_of_cell_recursive(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_mmm,  1,  1);
#endif
  }
#ifdef P4_TO_P8
  else if(dir_x== 1 && dir_z== 1)
  {
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_mmm,  1,  0,  1);
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_mpm,  1,  0,  1);
  }
  else if(dir_x==-1 && dir_z== 1)
  {
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_pmm, -1,  0,  1);
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_ppm, -1,  0,  1);
  }
  else if(dir_x== 1 && dir_z==-1)
  {
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_mmp,  1,  0, -1);
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_mpp,  1,  0, -1);
  }
  else if(dir_x==-1 && dir_z==-1)
  {
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_pmp, -1,  0, -1);
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_ppp, -1,  0, -1);
  }
  else if(dir_y== 1 && dir_z== 1)
  {
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_mmm,  0,  1,  1);
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_pmm,  0,  1,  1);
  }
  else if(dir_y==-1 && dir_z== 1)
  {
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_mpm,  0, -1,  1);
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_ppm,  0, -1,  1);
  }
  else if(dir_y== 1 && dir_z==-1)
  {
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_mmp,  0,  1, -1);
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_pmp,  0,  1, -1);
  }
  else if(dir_y==-1 && dir_z==-1)
  {
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_mpp,  0, -1, -1);
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_ppp,  0, -1, -1);
  }
#endif
  /* faces directions */
  else if(dir_x==-1)
  {
#ifdef P4_TO_P8
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_pmm, -1,  0,  0);
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_ppm, -1,  0,  0);
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_pmp, -1,  0,  0);
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_ppp, -1,  0,  0);
#else
    find_neighbor_cells_of_cell_recursive(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_pmm, -1,  0);
    find_neighbor_cells_of_cell_recursive(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_ppm, -1,  0);
#endif
  }
  else if(dir_x== 1)
  {
#ifdef P4_TO_P8
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_mmm,  1,  0,  0);
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_mpm,  1,  0,  0);
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_mmp,  1,  0,  0);
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_mpp,  1,  0,  0);
#else
    find_neighbor_cells_of_cell_recursive(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_mmm,  1,  0);
    find_neighbor_cells_of_cell_recursive(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_mpm,  1,  0);
#endif
  }
  else if(dir_y==-1)
  {
#ifdef P4_TO_P8
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_mpm,  0, -1,  0);
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_ppm,  0, -1,  0);
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_mpp,  0, -1,  0);
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_ppp,  0, -1,  0);
#else
    find_neighbor_cells_of_cell_recursive(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_mpm,  0, -1);
    find_neighbor_cells_of_cell_recursive(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_ppm,  0, -1);
#endif
  }
  else if(dir_y== 1)
  {
#ifdef P4_TO_P8
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_mmm,  0,  1,  0);
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_pmm,  0,  1,  0);
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_mmp,  0,  1,  0);
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_pmp,  0,  1,  0);
#else
    find_neighbor_cells_of_cell_recursive(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_mmm,  0,  1);
    find_neighbor_cells_of_cell_recursive(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_pmm,  0,  1);
#endif
  }
#ifdef P4_TO_P8
  else if(dir_z==-1)
  {
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_mmp,  0,  0, -1);
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_pmp,  0,  0, -1);
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_mpp,  0,  0, -1);
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_ppp,  0,  0, -1);
  }
  else if(dir_z== 1)
  {
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_mmm,  0,  0,  1);
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_pmm,  0,  0,  1);
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_mpm,  0,  0,  1);
    find_neighbor_cells_of_cell_recursive_test(ngbd, tr, hierarchy->trees[tr][ind].child + dir::v_ppm,  0,  0,  1);
  }
#endif
}
