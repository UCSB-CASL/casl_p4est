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
    qinfo.tree_idx = tr;

    if (qinfo.locidx < p4est->local_num_quadrants) {// local quadrant
      p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tr);
      qinfo.quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, qinfo.locidx - tree->quadrants_offset);
      qinfo.level = qinfo.quad->level;
      qinfo.gloidx = p4est->global_first_quadrant[p4est->mpirank] + qinfo.locidx;
    } else {
      qinfo.quad = (const p4est_quadrant_t*)sc_array_index(&ghost->ghosts, qinfo.locidx - p4est->local_num_quadrants);
      qinfo.gloidx = p4est->global_first_quadrant[hierarchy->trees[tr][ind].owner_rank] + qinfo.quad->p.piggy3.local_num;
      qinfo.level = qinfo.quad->level;
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

#ifdef P4_TO_P8
struct triangle{
  p4est_locidx_t p0, p1, p2;
  bool operator  =(const triangle& other) { return (p0 == other.p0 && p1 == other.p1 && p2 == other.p2); }
};

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
        const quad_info_t *begin = this->begin(q_locidx, dir::f_p00);
        const quad_info_t *end   = this->end(q_locidx, dir::f_p00);

        for (const quad_info_t* it = begin; it != end - 1; ++it){
          elements.push_back(q_locidx);
          elements.push_back(it->locidx);
          elements.push_back((it+1)->locidx);
        }
      }

      /* all the cells in the 0p direction */
      if (!is_quad_ypWall(p4est, tr_id, quad)){
        const quad_info_t *begin = this->begin(q_locidx, dir::f_0p0);
        const quad_info_t *end   = this->end(q_locidx, dir::f_0p0);

        for (const quad_info_t* it = begin; it != end - 1; ++it){
          elements.push_back(q_locidx);
          elements.push_back(it->locidx);
          elements.push_back((it+1)->locidx);
        }
      }

      /* corner in the pp direction */
      if (!is_quad_xpWall(p4est, tr_id, quad) && !is_quad_ypWall(p4est, tr_id, quad)){
        const quad_info_t *it_p0  = this->begin(q_locidx, dir::f_0m0) - 1;
        const quad_info_t *it_0p  = this->end(q_locidx, dir::f_0p0) - 1;
        const quad_info_t *it_pp;
        if (it_p0->level > it_0p->level)
          it_pp = this->begin(it_p0->locidx, dir::f_0p0);
        else
          it_pp = this->begin(it_0p->locidx, dir::f_p00);

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
