#ifdef P4_TO_P8
#include "my_p8est_cell_neighbors.h"
#else
#include "my_p4est_cell_neighbors.h"
#endif

void my_p4est_cell_neighbors_t::initialize_neighbors()
{
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
    find_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_pmm, dir_f);
#ifdef P4_TO_P8
    find_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_mpp, dir_f);
    find_neighbor_cells_of_cell_recursive(tr, hierarchy->trees[tr][ind].child + dir::v_pmp, dir_f);
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
