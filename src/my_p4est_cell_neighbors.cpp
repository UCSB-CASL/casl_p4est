#ifdef P4_TO_P8
#include "my_p8est_cell_neighbors.h"
#else
#include "my_p4est_cell_neighbors.h"
#endif

void my_p4est_cell_neighbors_t::init_neighbors()
{
  neighbor_cells.reserve(P4EST_FACES * n_quads);
  offsets.resize(P4EST_FACES*n_quads + 1, 0);

  // find neighboring quadrants of local quadrants
  for( p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx )
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for( size_t q=0; q<tree->quadrants.elem_count; ++q ){
      const p4est_quadrant_t* quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
      for(int f = 0; f<P4EST_FACES; ++f )
        find_neighbor_cells_of_cell(quad, q + tree->quadrants_offset, tree_idx, f);
    }
  }
  // find neighboring quadrants of ghost quadrants
  for (size_t q = 0; q < ghost->ghosts.elem_count; ++q){
    const p4est_quadrant_t* quad = (const p4est_quadrant_t*)sc_array_index(&ghost->ghosts, q);
    for(int f = 0; f<P4EST_FACES; ++f )
      find_neighbor_cells_of_cell(quad, q + p4est->local_num_quadrants, quad->p.piggy3.which_tree, f);
  }
}


void my_p4est_cell_neighbors_t::find_neighbor_cells_of_cell(const p4est_quadrant_t* quad, p4est_locidx_t q, p4est_topidx_t tree_idx, int dir_f )
{
  offsets[q*P4EST_FACES + dir_f + 1] = neighbor_cells.size();
  /* construct the coordinate of the neighbor cell of the same size in the given direction */
  p4est_qcoord_t size = P4EST_QUADRANT_LEN(quad->level);

  p4est_topidx_t nb_tree_idx = tree_idx;
  p4est_qcoord_t i_nb = quad->x + ( dir_f == dir::f_m00 ? -size : ( dir_f == dir::f_p00 ? size : 0) );
  p4est_qcoord_t j_nb = quad->y + ( dir_f == dir::f_0m0 ? -size : ( dir_f == dir::f_0p0 ? size : 0) );
#ifdef P4_TO_P8
  p4est_qcoord_t k_nb = quad->z + ( dir_f == dir::f_00m ? -size : ( dir_f == dir::f_00p ? size : 0) );
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
  find_neighbor_cells_of_cell_recursive( nb_tree_idx, ind, dir_f );
  offsets[q*P4EST_FACES + dir_f + 1] = neighbor_cells.size();
}


void my_p4est_cell_neighbors_t::find_neighbor_cells_of_cell_recursive(p4est_topidx_t tr, int ind, int dir_f )
{

  if (hierarchy->trees[tr][ind].child == CELL_LEAF)
  {
    if (hierarchy->trees[tr][ind].quad == NOT_A_P4EST_QUADRANT) return;

    quad_info_t qinfo;
    qinfo.locidx = hierarchy->trees[tr][ind].quad;

    if (qinfo.locidx < p4est->local_num_quadrants) {// local quadrant
      p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tr);
      const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, qinfo.locidx - tree->quadrants_offset);
      qinfo.level = quad->level;
      qinfo.gloidx = p4est->global_first_quadrant[p4est->mpirank] + qinfo.locidx;
    } else {
      const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&ghost->ghosts, qinfo.locidx - p4est->local_num_quadrants);
      qinfo.gloidx = p4est->global_first_quadrant[hierarchy->trees[tr][ind].owner_rank] + quad->p.piggy3.local_num;
      qinfo.level = quad->level;
    }

    neighbor_cells.push_back(qinfo);
    return;
  }

  switch(dir_f)
  {
  case dir::f_m00:
    find_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_pmm, dir_f);
    find_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_ppm, dir_f);
#ifdef P4_TO_P8
    find_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_pmp, dir_f);
    find_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_ppp, dir_f);
#endif
    break;
  case dir::f_p00:
    find_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mmm, dir_f);
    find_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mpm, dir_f);
#ifdef P4_TO_P8
    find_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mmp, dir_f);
    find_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mpp, dir_f);
#endif
    break;
  case dir::f_0m0:
    find_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mpm, dir_f);
    find_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_ppm, dir_f);
#ifdef P4_TO_P8
    find_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mpp, dir_f);
    find_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_ppp, dir_f);
#endif
    break;
  case dir::f_0p0:
    find_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mmm, dir_f);
    find_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_pmm, dir_f);
#ifdef P4_TO_P8
    find_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mmp, dir_f);
    find_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_pmp, dir_f);
#endif
    break;
#ifdef P4_TO_P8
  case dir::f_00m:
    find_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mmp, dir_f);
    find_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_pmp, dir_f);
    find_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mpp, dir_f);
    find_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_ppp, dir_f);
    break;
  case dir::f_00p:
    find_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mmm, dir_f);
    find_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_pmm, dir_f);
    find_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mpm, dir_f);
    find_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_ppm, dir_f);
    break;
#endif
  }
}





#ifdef P4_TO_P8
void my_p4est_cell_neighbors_t::find_neighbor_cells_of_cell_test(std::vector<p4est_quadrant_t>& ngbd, p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, char dir_x, char dir_y, char dir_z ) const
#else
void my_p4est_cell_neighbors_t::find_neighbor_cells_of_cell_test(std::vector<p4est_quadrant_t>& ngbd, p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, char dir_x, char dir_y ) const
#endif
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

  /* check if quadrant is on a boundary */
#ifdef P4_TO_P8
  if     (quad->x==0 && dir_x==-1 && quad->y==0 && dir_y==-1 && quad->z==0 && dir_z==-1)
  {
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
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx    + dir::f_m00];
    if(nb_tree_idx==tree_idx) return;
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*nb_tree_idx + dir::f_0m0];
    i_nb = P4EST_ROOT_LEN - size;
    j_nb = P4EST_ROOT_LEN - size;
  }
  else if(quad->x+size==P4EST_ROOT_LEN && dir_x==1 && quad->y==0 && dir_y==-1)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx    + dir::f_p00];
    if(nb_tree_idx==tree_idx) return;
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*nb_tree_idx + dir::f_0m0];
    i_nb = 0;
    j_nb = P4EST_ROOT_LEN - size;
  }
  else if(quad->x==0 && dir_x==-1 && quad->y+size==P4EST_ROOT_LEN && dir_y==1)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx    + dir::f_m00];
    if(nb_tree_idx==tree_idx) return;
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*nb_tree_idx + dir::f_0p0];
    i_nb = P4EST_ROOT_LEN - size;
    j_nb = 0;
  }
  else if(quad->x+size==P4EST_ROOT_LEN && dir_x==1 && quad->y+size==P4EST_ROOT_LEN && dir_y==1)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx    + dir::f_p00];
    if(nb_tree_idx==tree_idx) return;
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*nb_tree_idx + dir::f_0p0];
    i_nb = 0;
    j_nb = 0;
  }
#ifdef P4_TO_P8
  else if(quad->x==0 && dir_x==-1 && quad->z==0 && dir_z==-1)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx    + dir::f_m00];
    if(nb_tree_idx==tree_idx) return;
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*nb_tree_idx + dir::f_00m];
    i_nb = P4EST_ROOT_LEN - size;
    k_nb = P4EST_ROOT_LEN - size;
  }
  else if(quad->x+size==P4EST_ROOT_LEN && dir_x==1 && quad->z==0 && dir_z==-1)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx    + dir::f_p00];
    if(nb_tree_idx==tree_idx) return;
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*nb_tree_idx + dir::f_00m];
    i_nb = 0;
    k_nb = P4EST_ROOT_LEN - size;
  }
  else if(quad->x==0 && dir_x==-1 && quad->z+size==P4EST_ROOT_LEN && dir_z==1)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx    + dir::f_m00];
    if(nb_tree_idx==tree_idx) return;
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*nb_tree_idx + dir::f_00p];
    i_nb = P4EST_ROOT_LEN - size;
    k_nb = 0;
  }
  else if(quad->x+size==P4EST_ROOT_LEN && dir_x==1 && quad->z+size==P4EST_ROOT_LEN && dir_z==1)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx    + dir::f_p00];
    if(nb_tree_idx==tree_idx) return;
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*nb_tree_idx + dir::f_00p];
    i_nb = 0;
    k_nb = 0;
  }
  else if(quad->y==0 && dir_y==-1 && quad->z==0 && dir_z==-1)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx    + dir::f_0m0];
    if(nb_tree_idx==tree_idx) return;
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*nb_tree_idx + dir::f_00m];
    j_nb = P4EST_ROOT_LEN - size;
    k_nb = P4EST_ROOT_LEN - size;
  }
  else if(quad->y+size==P4EST_ROOT_LEN && dir_y==1 && quad->z==0 && dir_z==-1)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx    + dir::f_0p0];
    if(nb_tree_idx==tree_idx) return;
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*nb_tree_idx + dir::f_00m];
    j_nb = 0;
    k_nb = P4EST_ROOT_LEN - size;
  }
  else if(quad->y==0 && dir_y==-1 && quad->z+size==P4EST_ROOT_LEN && dir_z==1)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx    + dir::f_0m0];
    if(nb_tree_idx==tree_idx) return;
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*nb_tree_idx + dir::f_00p];
    j_nb = P4EST_ROOT_LEN - size;
    k_nb = 0;
  }
  else if(quad->y+size==P4EST_ROOT_LEN && dir_y==1 && quad->z+size==P4EST_ROOT_LEN && dir_z==1)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx    + dir::f_0p0];
    if(nb_tree_idx==tree_idx) return;
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*nb_tree_idx + dir::f_00p];
    j_nb = 0;
    k_nb = 0;
  }
#endif
  /* faces directions */
  else if(quad->x == 0 && dir_x == -1)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_m00];
    if(nb_tree_idx == tree_idx) return;
    i_nb = P4EST_ROOT_LEN - size;
  }
  else if(quad->x+size == P4EST_ROOT_LEN && dir_x == 1)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_p00];
    if(nb_tree_idx == tree_idx) return;
    i_nb = 0;
  }
  else if(quad->y == 0 && dir_y == -1)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_0m0];
    if(nb_tree_idx == tree_idx) return;
    j_nb = P4EST_ROOT_LEN - size;
  }
  else if(quad->y+size == P4EST_ROOT_LEN && dir_y == 1)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_0p0];
    if(nb_tree_idx == tree_idx) return;
    j_nb = 0;
  }
#ifdef P4_TO_P8
  else if(quad->z == 0 && dir_z == -1)
  {
    nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_00m];
    if(nb_tree_idx == tree_idx) return;
    k_nb = P4EST_ROOT_LEN - size;
  }
  else if(quad->z+size == P4EST_ROOT_LEN && dir_z == 1)
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
#ifdef P4_TO_P8
  find_neighbor_cells_of_cell_recursive_test( ngbd, nb_tree_idx, ind, dir_x, dir_y, dir_z );
#else
  find_neighbor_cells_of_cell_recursive( ngbd, nb_tree_idx, ind, dir_x, dir_y );
#endif
}


#ifdef P4_TO_P8
void my_p4est_cell_neighbors_t::find_neighbor_cells_of_cell_recursive_test( std::vector<p4est_quadrant_t>& ngbd, p4est_topidx_t tr, int ind, char dir_x, char dir_y, char dir_z ) const
#else
void my_p4est_cell_neighbors_t::find_neighbor_cells_of_cell_recursive( std::vector<p4est_quadrant_t>& ngbd, p4est_topidx_t tr, int ind, char dir_x, char dir_y ) const
#endif
{
  if (hierarchy->trees[tr][ind].child == CELL_LEAF)
  {
    if (hierarchy->trees[tr][ind].quad == NOT_A_P4EST_QUADRANT) return;

    p4est_quadrant_t quad;
    p4est_locidx_t locid = hierarchy->trees[tr][ind].quad;
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



void my_p4est_cell_neighbors_t::print_debug(p4est_locidx_t q, FILE *stream)
{
  fprintf(stream, " ---------- quad = %d ---------- \n", q);
  fprintf(stream, " x-: ");
  for (p4est_locidx_t qi = offsets[q*P4EST_FACES + 0]; qi < offsets[q*P4EST_FACES + 1]; ++qi)
    fprintf(stream, "%d ", neighbor_cells[qi].locidx);
  fprintf(stream, "\n");

  fprintf(stream, " x+: ");
  for (p4est_locidx_t qi = offsets[q*P4EST_FACES + 1]; qi < offsets[q*P4EST_FACES + 2]; ++qi)
    fprintf(stream, "%d ", neighbor_cells[qi].locidx);
  fprintf(stream, "\n");

  fprintf(stream, " y-: ");
  for (p4est_locidx_t qi = offsets[q*P4EST_FACES + 2]; qi < offsets[q*P4EST_FACES + 3]; ++qi)
    fprintf(stream, "%d ", neighbor_cells[qi].locidx);
  fprintf(stream, "\n");

  fprintf(stream, " y+: ");
  for (p4est_locidx_t qi = offsets[q*P4EST_FACES + 3]; qi < offsets[q*P4EST_FACES + 4]; ++qi)
    fprintf(stream, "%d ", neighbor_cells[qi].locidx);
  fprintf(stream, "\n");

#ifdef P4_TO_P8
  fprintf(stream, " z-: ");
  for (p4est_locidx_t qi = offsets[q*P4EST_FACES + 4]; qi < offsets[q*P4EST_FACES + 5]; ++qi)
    fprintf(stream, "%d ", neighbor_cells[qi].locidx);
  fprintf(stream, "\n");

  fprintf(stream, " z+: ");
  for (p4est_locidx_t qi = offsets[q*P4EST_FACES + 5]; qi < offsets[q*P4EST_FACES + 6]; ++qi)
    fprintf(stream, "%d ", neighbor_cells[qi].locidx);
  fprintf(stream, "\n");
#endif
}
