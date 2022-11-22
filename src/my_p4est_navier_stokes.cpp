#ifdef P4_TO_P8
#include <src/my_p8est_poisson_cells.h>
#include <src/my_p8est_poisson_faces.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_level_set_cells.h>
#include <src/my_p8est_level_set_faces.h>
#include <src/my_p8est_trajectory_of_point.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_macros.h>
#include <p8est_extended.h>
#include <p8est_algorithms.h>
#include "my_p8est_navier_stokes.h"
#else
#include <src/my_p4est_poisson_cells.h>
#include <src/my_p4est_poisson_faces.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_level_set_cells.h>
#include <src/my_p4est_level_set_faces.h>
#include <src/my_p4est_trajectory_of_point.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_macros.h>
#include <p4est_extended.h>
#include <p4est_algorithms.h>
#include "my_p4est_navier_stokes.h"
#endif

#include <algorithm>

#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_my_p4est_navier_stokes_viscosity;
extern PetscLogEvent log_my_p4est_navier_stokes_projection;
extern PetscLogEvent log_my_p4est_navier_stokes_update;
#endif

my_p4est_navier_stokes_t::splitting_criteria_vorticity_t::splitting_criteria_vorticity_t(int min_lvl, int max_lvl, double lip, double uniform_band, double threshold_vorticity, double max_L2_norm_u, double smoke_thresh, double threshold_norm_grad_u)
  : splitting_criteria_tag_t(min_lvl, max_lvl, lip)
{
  this->uniform_band          = uniform_band;
  this->threshold_vorticity   = threshold_vorticity;
  this->max_L2_norm_u         = max_L2_norm_u;
  this->smoke_thresh          = smoke_thresh;
  this->threshold_norm_grad_u = threshold_norm_grad_u;
}

void my_p4est_navier_stokes_t::splitting_criteria_vorticity_t::tag_quadrant(p4est_t *p4est, p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, p4est_nodes_t* nodes,
                                                                            const double* tree_dimensions,
                                                                            const double *phi_p, const double *vorticity_p, const double *smoke_p, const double* norm_grad_u_p)
{
  p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tree_idx);
  p4est_quadrant_t *quad = p4est_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);

  if (quad->level < min_lvl)
    quad->p.user_int = REFINE_QUADRANT;

  else if (quad->level > max_lvl)
    quad->p.user_int = COARSEN_QUADRANT;

  else
  {
    const double quad_diag          = sqrt(SUMD(SQR(tree_dimensions[0]), SQR(tree_dimensions[1]), SQR(tree_dimensions[2])))*(((double) P4EST_QUADRANT_LEN(quad->level))/((double) P4EST_ROOT_LEN));
    const double quad_dxyz_max      = MAX(DIM(tree_dimensions[0],tree_dimensions[1], tree_dimensions[2]))*(((double) P4EST_QUADRANT_LEN(quad->level))/((double) P4EST_ROOT_LEN));
    const double smallest_dxyz_max  = MAX(DIM(tree_dimensions[0],tree_dimensions[1], tree_dimensions[2]))*(((double) P4EST_QUADRANT_LEN((int8_t) max_lvl))/((double) P4EST_ROOT_LEN));

    bool coarsen = (quad->level > min_lvl);
    if(coarsen)
    {
      bool all_pos    = true;
      bool cor_vort   = true;
      bool cor_band   = true;
      bool cor_intf   = true;
      bool cor_smok   = true;
      bool cor_grad_u = true;
      p4est_locidx_t node_idx;

      for (unsigned char k = 0; k < P4EST_CHILDREN; ++k) {
        node_idx    = nodes->local_nodes[P4EST_CHILDREN*quad_idx + k];
        cor_vort    = cor_vort && fabs(vorticity_p[node_idx])*2.0*quad_dxyz_max/max_L2_norm_u < threshold_vorticity;
        cor_band    = cor_band && fabs(phi_p[node_idx]) > uniform_band*smallest_dxyz_max;
        cor_intf    = cor_intf && fabs(phi_p[node_idx]) >= lip*2.0*quad_diag;
        if(smoke_p != NULL)
          cor_smok  = cor_smok && smoke_p[node_idx] < smoke_thresh;
        if(norm_grad_u_p != NULL)
          cor_grad_u = cor_grad_u && norm_grad_u_p[node_idx]*2.0*quad_dxyz_max/max_L2_norm_u < threshold_norm_grad_u;

        all_pos = all_pos && phi_p[node_idx] > MAX(2.0, uniform_band)*smallest_dxyz_max;
        // [RAPHAEL:] modified to enforce two layers of finest level in positive domain as well (better extrapolation etc.), also required for Neumann BC in the face-solvers
        coarsen = (cor_vort && cor_band && cor_intf && cor_smok && cor_grad_u) || all_pos;
        if(!coarsen)
          break;
      }
    }


    bool refine = (quad->level < max_lvl);
    if(refine)
    {
      bool is_neg       = false;
      bool ref_vort     = false;
      bool ref_band     = false;
      bool ref_intf     = false;
      bool ref_smok     = false;
      bool ref_grad_u   = false;
      p4est_locidx_t node_idx;
      bool node_found;
      // check possibly finer points
      const p4est_qcoord_t mid_qh = P4EST_QUADRANT_LEN (quad->level + 1);
#ifdef P4_TO_P8
      for(unsigned char k = 0; k < 3; ++k)
#endif
        for(unsigned char j = 0; j < 3; ++j)
          for(unsigned char i = 0; i < 3; ++i)
          {
            if(ANDD(i == 1, j == 1, k == 1))
              continue;
            if(ANDD(i == 0 || i == 2, j == 0 || j == 2, k == 0 || k == 2))
            {
              node_found=true;
              node_idx = nodes->local_nodes[P4EST_CHILDREN*quad_idx + SUMD(i/2, 2*(j/2), 4*(k/2))]; // integer divisions!
            }
            else
            {
              p4est_quadrant_t r, c;
              r.level = P4EST_MAXLEVEL;
              r.x = quad->x + i*mid_qh;
              r.y = quad->y + j*mid_qh;
#ifdef P4_TO_P8
              r.z = quad->z + k*mid_qh;
#endif
              P4EST_ASSERT (p4est_quadrant_is_node (&r, 0));
              p4est_node_canonicalize(p4est, tree_idx, &r, &c);
              node_found = index_of_node(&c, nodes, node_idx);
            }
            if(node_found)
            {
              P4EST_ASSERT(node_idx < ((p4est_locidx_t) nodes->indep_nodes.elem_count));
              ref_vort        = ref_vort || fabs(vorticity_p[node_idx])*quad_dxyz_max/max_L2_norm_u > threshold_vorticity;
              ref_band        = ref_band || fabs(phi_p[node_idx]) < uniform_band*smallest_dxyz_max;
              ref_intf        = ref_intf || fabs(phi_p[node_idx]) <= lip*quad_diag;
              if(smoke_p != NULL)
                ref_smok      = ref_smok || smoke_p[node_idx] >= smoke_thresh;
              if(norm_grad_u_p != NULL)
                ref_grad_u    = ref_grad_u || fabs(norm_grad_u_p[node_idx])*quad_dxyz_max/max_L2_norm_u > threshold_norm_grad_u;
              is_neg = is_neg || phi_p[node_idx]< MAX(2.0, uniform_band)*smallest_dxyz_max; // [RAPHAEL:] same comment as before

              refine = is_neg && (ref_vort || ref_band || ref_intf || ref_smok || ref_grad_u);
              if(refine)
                goto end_of_function;
            }
          }
    }
end_of_function:

    if (refine)
      quad->p.user_int = REFINE_QUADRANT;
    else if (coarsen)
      quad->p.user_int = COARSEN_QUADRANT;
    else
      quad->p.user_int = SKIP_QUADRANT;
  }
}

bool my_p4est_navier_stokes_t::splitting_criteria_vorticity_t::refine_and_coarsen(p4est_t* p4est, p4est_nodes_t* nodes, Vec phi, Vec vorticity, Vec smoke, Vec norm_grad_u)
{
  const double *phi_p, *vorticity_p, *smoke_p, *norm_grad_u_p;
  phi_p = vorticity_p = smoke_p = norm_grad_u_p = NULL;
  PetscErrorCode ierr;
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(vorticity, &vorticity_p); CHKERRXX(ierr);
  if(smoke != NULL){
    ierr = VecGetArrayRead(smoke, &smoke_p); CHKERRXX(ierr); }
  if(norm_grad_u != NULL){
    ierr = VecGetArrayRead(norm_grad_u, &norm_grad_u_p); CHKERRXX(ierr); }

  double tree_dimensions[P4EST_DIM];
  p4est_locidx_t* t2v = p4est->connectivity->tree_to_vertex;
  double* v2c = p4est->connectivity->vertices;
  /* tag the quadrants that need to be refined or coarsened */
  for (p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx) {
    p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      tree_dimensions[dir] = v2c[3*t2v[P4EST_CHILDREN*tree_idx + P4EST_CHILDREN -1] + dir] - v2c[3*t2v[P4EST_CHILDREN*tree_idx + 0] + dir];
    for (size_t q = 0; q < tree->quadrants.elem_count; ++q) {
      p4est_locidx_t quad_idx  = q + tree->quadrants_offset;
      tag_quadrant(p4est, quad_idx, tree_idx, nodes, tree_dimensions, phi_p, vorticity_p, smoke_p, norm_grad_u_p);
    }
  }

  my_p4est_coarsen(p4est, P4EST_FALSE, splitting_criteria_vorticity_t::coarsen_fn, splitting_criteria_vorticity_t::init_fn);
  my_p4est_refine (p4est, P4EST_FALSE, splitting_criteria_vorticity_t::refine_fn,  splitting_criteria_vorticity_t::init_fn);

  int is_grid_changed = false;
  for (p4est_topidx_t it = p4est->first_local_tree; it <= p4est->last_local_tree; ++it) {
    p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, it);
    for (size_t q = 0; q < tree->quadrants.elem_count; ++q) {
      p4est_quadrant_t *quad = p4est_quadrant_array_index(&tree->quadrants, q);
      if (quad->p.user_int == NEW_QUADRANT) {
        is_grid_changed = true;
        goto function_end;
      }
    }
  }

function_end:
  MPI_Allreduce(MPI_IN_PLACE, &is_grid_changed, 1, MPI_INT, MPI_LOR, p4est->mpicomm);

  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(vorticity, &vorticity_p); CHKERRXX(ierr);
  if(smoke != NULL){
    ierr = VecRestoreArrayRead(smoke, &smoke_p); CHKERRXX(ierr); }
  if(norm_grad_u != NULL){
    ierr = VecRestoreArrayRead(norm_grad_u, &norm_grad_u_p); CHKERRXX(ierr); }

  return is_grid_changed;
}

double my_p4est_navier_stokes_t::wall_bc_value_hodge_t::operator ()(DIM(double x, double y, double z)) const
{
  return _prnt->bc_pressure->wallValue(DIM(x, y, z)) * _prnt->dt_n / (_prnt->alpha()*_prnt->rho);
}

double my_p4est_navier_stokes_t::interface_bc_value_hodge_t::operator ()(DIM(double x, double y, double z)) const
{
  return _prnt->bc_pressure->interfaceValue(DIM(x, y, z)) * _prnt->dt_n / (_prnt->alpha()*_prnt->rho);
}

my_p4est_navier_stokes_t::my_p4est_navier_stokes_t(my_p4est_node_neighbors_t *ngbd_nm1, my_p4est_node_neighbors_t *ngbd_n, my_p4est_faces_t *faces)
  : brick(ngbd_n->myb), conn(ngbd_n->p4est->connectivity),
    p4est_nm1(ngbd_nm1->p4est), ghost_nm1(ngbd_nm1->ghost), nodes_nm1(ngbd_nm1->nodes),
    hierarchy_nm1(ngbd_nm1->hierarchy), ngbd_nm1(ngbd_nm1),
    p4est_n(ngbd_n->p4est), ghost_n(ngbd_n->ghost), nodes_n(ngbd_n->nodes),
    hierarchy_n(ngbd_n->hierarchy), ngbd_n(ngbd_n),
    ngbd_c(faces->get_ngbd_c()), faces_n(faces), semi_lagrangian_backtrace_is_done(false), interpolators_from_face_to_nodes_are_set(false),
    wall_bc_value_hodge(this), interface_bc_value_hodge(this)
{
  PetscErrorCode ierr;

  mu = 1;
  rho = 1;
  uniform_band = 0;
  vorticity_threshold_split_cell = 0.04;
  norm_grad_u_threshold_split_cell = DBL_MAX;
  n_times_dt = 5;
  dt_updated = false;
  max_L2_norm_u = 0;

  sl_order = 1;
  interp_v_viscosity  = quadratic;
  interp_v_update     = quadratic;

  double *v2c = p4est_n->connectivity->vertices;
  p4est_topidx_t *t2v = p4est_n->connectivity->tree_to_vertex;
  p4est_topidx_t first_tree = 0, last_tree = p4est_n->trees->elem_count - 1;
  p4est_topidx_t first_vertex = 0, last_vertex = P4EST_CHILDREN - 1;

  splitting_criteria_t *data = (splitting_criteria_t*)p4est_n->user_pointer;

  for (unsigned char dir = 0; dir < P4EST_DIM; dir++)
  {
    xyz_min[dir] = v2c[3*t2v[P4EST_CHILDREN*first_tree + first_vertex] + dir];
    xyz_max[dir] = v2c[3*t2v[P4EST_CHILDREN*last_tree  + last_vertex ] + dir];

    double xyz_tmp = v2c[3*t2v[P4EST_CHILDREN*first_tree + last_vertex] + dir];
    dxyz_min[dir] = (xyz_tmp - xyz_min[dir]) / (1<<data->max_lvl);
    convert_to_xyz[dir] = xyz_tmp - xyz_min[dir];
    // initialize this one
    interpolator_from_face_to_nodes[dir].resize(0);
  }

  dt_nm1 = dt_n = .5 * MIN(DIM(dxyz_min[0], dxyz_min[1], dxyz_min[2]));

  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &phi); CHKERRXX(ierr);
  ierr = VecCreateGhostNodesBlock(p4est_n, nodes_n, P4EST_DIM, &grad_phi); CHKERRXX(ierr);
  Vec vec_loc, vec_loc_grad;
  ierr = VecGhostGetLocalForm(phi, &vec_loc); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(grad_phi, &vec_loc_grad); CHKERRXX(ierr);
  ierr = VecSet(vec_loc, -1); CHKERRXX(ierr);
  ierr = VecSet(vec_loc_grad, 0.0); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(grad_phi, &vec_loc_grad); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(phi, &vec_loc); CHKERRXX(ierr);

  ierr = VecCreateGhostCells(p4est_n, ghost_n, &hodge); CHKERRXX(ierr);
  ierr = VecCreateGhostCells(p4est_n, ghost_n, &pressure); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(hodge, &vec_loc); CHKERRXX(ierr);
  ierr = VecSet(vec_loc, 0); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(hodge, &vec_loc); CHKERRXX(ierr);

  bc_v = NULL;
  bc_pressure = NULL;

  vorticity = NULL;
  norm_grad_u = NULL;
  smoke = NULL;
  bc_smoke = NULL;
  refine_with_smoke = false;

  for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
  {
    external_forces_per_unit_volume[dir]  = NULL;
    external_forces_per_unit_mass[dir]    = NULL;
    vstar[dir] = NULL;
    vnp1[dir] = NULL;

    vnm1_nodes[dir] = NULL;
    vn_nodes  [dir] = NULL;
    vnp1_nodes[dir] = NULL;
    for (unsigned char dd = 0; dd < P4EST_DIM; ++dd) {
      second_derivatives_vn_nodes[dd][dir]    = NULL;
      second_derivatives_vnm1_nodes[dd][dir]  = NULL;
    }

    ierr = VecCreateGhostFaces(p4est_n, faces_n, &face_is_well_defined[dir], dir); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(face_is_well_defined[dir], &vec_loc); CHKERRXX(ierr);
    ierr = VecSet(vec_loc, 1); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(face_is_well_defined[dir], &vec_loc); CHKERRXX(ierr);

    ierr = VecCreateGhostFaces(p4est_n, faces_n, &dxyz_hodge[dir], dir); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(dxyz_hodge[dir], &vec_loc); CHKERRXX(ierr);
    ierr = VecSet(vec_loc, 0); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(dxyz_hodge[dir], &vec_loc); CHKERRXX(ierr);
  }

  ngbd_n->init_neighbors();
  ngbd_nm1->init_neighbors();

  interp_phi = new my_p4est_interpolation_nodes_t(ngbd_n);
  interp_phi->set_input(phi, linear);
  interp_grad_phi = new my_p4est_interpolation_nodes_t(ngbd_n);
  interp_grad_phi->set_input(grad_phi, linear, P4EST_DIM);
}

my_p4est_navier_stokes_t::my_p4est_navier_stokes_t(const mpi_environment_t& mpi, const char* path_to_save_state, double &simulation_time)
  : semi_lagrangian_backtrace_is_done(false), interpolators_from_face_to_nodes_are_set(false), wall_bc_value_hodge(this), interface_bc_value_hodge(this)
{
  PetscErrorCode ierr;
  // we need to initialize those to NULL, otherwise the loader will freak out
  // no risk of memory leak here, this is a CONSTRUCTOR
  brick = NULL; conn = NULL;
  p4est_n = NULL; ghost_n = NULL; nodes_n = NULL;
  hierarchy_n = NULL; ngbd_n = NULL;
  ngbd_c = NULL; faces_n = NULL;
  p4est_nm1 = NULL; ghost_nm1 = NULL; nodes_nm1 = NULL;
  hierarchy_nm1 = NULL; ngbd_nm1 = NULL;
  phi = NULL; hodge = NULL; smoke = NULL;
  for (unsigned char kk = 0; kk < P4EST_DIM; ++kk) {
    vnm1_nodes[kk] = NULL;
    vn_nodes[kk] = NULL;
    for (unsigned char dd = 0; dd < P4EST_DIM; ++dd) {
      second_derivatives_vn_nodes[dd][kk]    = NULL;
      second_derivatives_vnm1_nodes[dd][kk]  = NULL;
    }
  }
  dt_updated = false;
  load_state(mpi, path_to_save_state, simulation_time);
  ierr = VecCreateGhostCells(p4est_n, ghost_n, &pressure); CHKERRXX(ierr);

  bc_v = NULL;
  bc_pressure = NULL;

  vorticity = NULL;
  norm_grad_u = NULL;
  bc_smoke = NULL;

  Vec vec_loc;
  for(unsigned int dir = 0; dir < P4EST_DIM; ++dir)
  {
    external_forces_per_unit_volume[dir]  = NULL;
    external_forces_per_unit_mass[dir]    = NULL;
    vstar[dir] = NULL;
    vnp1[dir] = NULL;
    vnp1_nodes[dir] = NULL;

    ierr = VecCreateGhostFaces(p4est_n, faces_n, &face_is_well_defined[dir], dir); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(face_is_well_defined[dir], &vec_loc); CHKERRXX(ierr);
    ierr = VecSet(vec_loc, 1); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(face_is_well_defined[dir], &vec_loc); CHKERRXX(ierr);

    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &vnp1_nodes[dir]); CHKERRXX(ierr);

    ierr = VecCreateGhostFaces(p4est_n, faces_n, &vstar[dir], dir); CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est_n, faces_n, &vnp1[dir], dir); CHKERRXX(ierr);

    for (unsigned char dd = 0; dd < P4EST_DIM; ++dd) {
      if(second_derivatives_vn_nodes[dd][dir] != NULL)
      {
        ierr = VecDestroy(second_derivatives_vn_nodes[dd][dir]); CHKERRXX(ierr);
      }
      if(second_derivatives_vnm1_nodes[dd][dir] != NULL)
      {
        ierr = VecDestroy(second_derivatives_vnm1_nodes[dd][dir]); CHKERRXX(ierr);
      }
      ierr = VecCreateGhostNodes(p4est_n, nodes_n, &second_derivatives_vn_nodes[dd][dir]); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est_nm1, nodes_nm1, &second_derivatives_vnm1_nodes[dd][dir]); CHKERRXX(ierr);
    }
  }
  ngbd_n->second_derivatives_central(vn_nodes, DIM(second_derivatives_vn_nodes[0], second_derivatives_vn_nodes[1], second_derivatives_vn_nodes[2]), P4EST_DIM);
  ngbd_nm1->second_derivatives_central(vnm1_nodes, DIM(second_derivatives_vnm1_nodes[0], second_derivatives_vnm1_nodes[1], second_derivatives_vnm1_nodes[2]), P4EST_DIM);

  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &vorticity); CHKERRXX(ierr);
  if(norm_grad_u_threshold_split_cell <= largest_dbl_smaller_than_dbl_max){
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &norm_grad_u); CHKERRXX(ierr); }


  interp_phi = new my_p4est_interpolation_nodes_t(ngbd_n);
  interp_phi->set_input(phi, linear);

  interp_grad_phi = new my_p4est_interpolation_nodes_t(ngbd_n);
  ierr = VecCreateGhostNodesBlock(p4est_n, nodes_n, P4EST_DIM, &grad_phi);
  ngbd_n->first_derivatives_central(phi, grad_phi);
  interp_grad_phi->set_input(grad_phi, linear, P4EST_DIM);
}

my_p4est_navier_stokes_t::~my_p4est_navier_stokes_t()
{
  PetscErrorCode ierr;
  if(phi != NULL)       { ierr = VecDestroy(phi);         CHKERRXX(ierr); }
  if(grad_phi != NULL)  { ierr = VecDestroy(grad_phi);    CHKERRXX(ierr); }
  if(hodge != NULL)     { ierr = VecDestroy(hodge);       CHKERRXX(ierr); }
  if(pressure != NULL)  { ierr = VecDestroy(pressure);    CHKERRXX(ierr); }
  if(vorticity != NULL) { ierr = VecDestroy(vorticity);   CHKERRXX(ierr); }
  if(norm_grad_u!= NULL){ ierr = VecDestroy(norm_grad_u); CHKERRXX(ierr); }
  if(smoke != NULL)     { ierr = VecDestroy(smoke);       CHKERRXX(ierr); }

  for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
  {
    if(dxyz_hodge[dir] != NULL) { ierr = VecDestroy(dxyz_hodge[dir]);                         CHKERRXX(ierr); }
    if(vstar[dir] != NULL)      { ierr = VecDestroy(vstar[dir]);                              CHKERRXX(ierr); }
    if(vnp1[dir] != NULL)       { ierr = VecDestroy(vnp1[dir]);                               CHKERRXX(ierr); }
    if(vnm1_nodes[dir] != NULL) { ierr = VecDestroy(vnm1_nodes[dir]);                         CHKERRXX(ierr); }
    if(vn_nodes[dir] != NULL)   { ierr = VecDestroy(vn_nodes[dir]);                           CHKERRXX(ierr); }
    if(vnp1_nodes[dir] != NULL) { ierr = VecDestroy(vnp1_nodes[dir]);                         CHKERRXX(ierr); }
    if(face_is_well_defined[dir] != NULL)
                                { ierr = VecDestroy(face_is_well_defined[dir]);               CHKERRXX(ierr); }
    for (unsigned char dd = 0; dd < P4EST_DIM; ++dd) {
      if(second_derivatives_vn_nodes[dd][dir] != NULL)
                                { ierr = VecDestroy(second_derivatives_vn_nodes[dd][dir]);    CHKERRXX(ierr); }
      if(second_derivatives_vnm1_nodes[dd][dir] != NULL)
                                { ierr = VecDestroy(second_derivatives_vnm1_nodes[dd][dir]);  CHKERRXX(ierr); }
    }
  }

  if(interp_phi != NULL)      delete interp_phi;
  if(interp_grad_phi != NULL) delete interp_grad_phi;

  if(ngbd_nm1 != ngbd_n)
    delete ngbd_nm1;
  delete ngbd_n;
  if(hierarchy_nm1 != hierarchy_n)
    delete hierarchy_nm1;
  delete hierarchy_n;
  if(nodes_nm1 != nodes_n)
    p4est_nodes_destroy(nodes_nm1);
  p4est_nodes_destroy(nodes_n);
  if(p4est_nm1 != p4est_n)
    p4est_destroy(p4est_nm1);
  p4est_destroy(p4est_n);
  if(ghost_nm1 != ghost_n)
    p4est_ghost_destroy(ghost_nm1);
  p4est_ghost_destroy(ghost_n);

  delete faces_n;
  delete ngbd_c;

  my_p4est_brick_destroy(conn, brick);
}

void my_p4est_navier_stokes_t::set_parameters(double mu_, double rho_, int sl_order_, double uniform_band_, double vorticity_threshold_split_cell_, double n_times_dt_, double norm_grad_u_threshold_split_cell_)
{
  P4EST_ASSERT(mu_ >= 0.0);                               this->mu = mu_;
  P4EST_ASSERT(rho_ > 0.0);                               this->rho = rho_;
  P4EST_ASSERT(sl_order_ == 1 || sl_order_ == 2);         this->sl_order = sl_order_;
  P4EST_ASSERT(uniform_band_ >= 0.0);                     this->uniform_band = uniform_band_;
  P4EST_ASSERT(vorticity_threshold_split_cell > 0.0);     this->vorticity_threshold_split_cell = vorticity_threshold_split_cell_;
  P4EST_ASSERT(n_times_dt_ > 0.0);                        this->n_times_dt = n_times_dt_;
  P4EST_ASSERT(norm_grad_u_threshold_split_cell_ > 0.0);  this->norm_grad_u_threshold_split_cell = norm_grad_u_threshold_split_cell_;
}

void my_p4est_navier_stokes_t::set_smoke(Vec smoke, CF_DIM *bc_smoke, bool refine_with_smoke, double smoke_thresh)
{
  if(this->smoke != NULL && this->smoke != smoke){
    PetscErrorCode ierr= VecDestroy(this->smoke); CHKERRXX(ierr); }
  this->smoke = smoke;
  this->bc_smoke = bc_smoke;
  this->refine_with_smoke = refine_with_smoke;
  this->smoke_thresh = smoke_thresh;
}

void my_p4est_navier_stokes_t::set_phi(Vec phi)
{
  PetscErrorCode ierr;
  if(this->phi != NULL && this->phi != phi) {
    ierr = VecDestroy(this->phi); CHKERRXX(ierr); }
  this->phi = phi;
  ngbd_n->first_derivatives_central(phi, grad_phi);
  interp_phi->set_input(phi, linear);
  interp_grad_phi->set_input(grad_phi, linear, P4EST_DIM);

  if(bc_v != NULL)
    for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      check_if_faces_are_well_defined(faces_n, dir, *interp_phi, bc_v[dir], face_is_well_defined[dir]);
  return;
}

void my_p4est_navier_stokes_t::set_external_forces_using_vector(Vec f)
{
  PetscErrorCode ierr;
  ierr = VecCreateGhostNodes(p4est_n,nodes_n,&external_force_per_unit_volume); CHKERRXX(ierr);
  ierr = VecDuplicate(f,&external_force_per_unit_volume);
  double *external_force_per_unit_volume_p;
  ierr= VecGetArray(external_force_per_unit_volume,&external_force_per_unit_volume_p); CHKERRXX(ierr);
  double *f_p;
  ierr= VecGetArray(f,&f_p); CHKERRXX(ierr);
  for (size_t n=0; n < nodes_n->indep_nodes.elem_count; ++n){
      external_force_per_unit_volume_p[n]=f_p[n];
  }
  ierr= VecRestoreArray(external_force_per_unit_volume,&external_force_per_unit_volume_p); CHKERRXX(ierr);
  ierr= VecRestoreArray(f,&f_p); CHKERRXX(ierr);
}

void my_p4est_navier_stokes_t::set_external_forces_per_unit_volume(CF_DIM **external_forces_per_unit_volume_)
{
  for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
  {
    external_forces_per_unit_mass[dir]    = NULL;
    external_forces_per_unit_volume[dir]  = external_forces_per_unit_volume_[dir];
  }
}

void my_p4est_navier_stokes_t::set_external_forces_per_unit_mass(CF_DIM **external_forces_per_unit_mass_)
{
  for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
  {
    external_forces_per_unit_mass[dir]    = external_forces_per_unit_mass_[dir];
    external_forces_per_unit_volume[dir]  = NULL;
  }
}

void my_p4est_navier_stokes_t::set_bc(BoundaryConditionsDIM *bc_v, BoundaryConditionsDIM *bc_p)
{
  this->bc_v = bc_v;
  this->bc_pressure = bc_p;

  bc_hodge.setWallTypes(bc_pressure->getWallType());
  bc_hodge.setWallValues(wall_bc_value_hodge);
  bc_hodge.setInterfaceType(bc_pressure->interfaceType());
  bc_hodge.setInterfaceValue(interface_bc_value_hodge);

  if(phi != NULL)
  {
    P4EST_ASSERT(interp_phi != NULL);
    for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      check_if_faces_are_well_defined(faces_n, dir, *interp_phi, bc_v[dir], face_is_well_defined[dir]);
  }
}

void my_p4est_navier_stokes_t::set_velocities(Vec *vnm1_nodes_, Vec *vn_nodes_, const double *max_L2_norm_u_from_user)
{
  PetscErrorCode ierr;

  for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
  {
    // destroy current vectors if needed
    if(this->vn_nodes[dir] != NULL && this->vn_nodes[dir] != vn_nodes_[dir]){
      ierr = VecDestroy(this->vn_nodes[dir]); CHKERRXX(ierr); }
    if(this->vnm1_nodes[dir] != NULL && this->vnm1_nodes[dir] != vnm1_nodes_[dir]){
      ierr = VecDestroy(this->vnm1_nodes[dir]); CHKERRXX(ierr); }
    if(this->vnp1_nodes[dir] != NULL){
      ierr = VecDestroy(this->vnp1_nodes[dir]); CHKERRXX(ierr); }
    if(this->vstar[dir] != NULL){
      ierr = VecDestroy(this->vstar[dir]); CHKERRXX(ierr); }
    if(this->vnp1[dir] != NULL){
      ierr = VecDestroy(this->vnp1[dir]); CHKERRXX(ierr); }
    // set to given vectors or create new vectors
    this->vn_nodes[dir]   = vn_nodes_[dir];
    this->vnm1_nodes[dir] = vnm1_nodes_[dir];
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &vnp1_nodes [dir]); CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est_n, faces_n, &vstar[dir], dir); CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est_n, faces_n, &vnp1[dir], dir); CHKERRXX(ierr);

    // create vectors for second derivatives of node-sampled velocities
    for (unsigned char dd = 0; dd < P4EST_DIM; ++dd) {
      if(second_derivatives_vn_nodes[dd][dir] != NULL){
        ierr = VecDestroy(second_derivatives_vn_nodes[dd][dir]); CHKERRXX(ierr); }
      if(second_derivatives_vnm1_nodes[dd][dir] != NULL) {
        ierr = VecDestroy(second_derivatives_vnm1_nodes[dd][dir]); CHKERRXX(ierr); }

      ierr = VecCreateGhostNodes(p4est_n, nodes_n, &second_derivatives_vn_nodes[dd][dir]); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est_nm1, nodes_nm1, &second_derivatives_vnm1_nodes[dd][dir]); CHKERRXX(ierr);
    }
  }
  ngbd_n->second_derivatives_central(vn_nodes, DIM(second_derivatives_vn_nodes[0], second_derivatives_vn_nodes[1], second_derivatives_vn_nodes[2]), P4EST_DIM);
  ngbd_nm1->second_derivatives_central(vnm1_nodes, DIM(second_derivatives_vnm1_nodes[0], second_derivatives_vnm1_nodes[1], second_derivatives_vnm1_nodes[2]), P4EST_DIM);


  if(max_L2_norm_u_from_user != NULL) {
    max_L2_norm_u = *max_L2_norm_u_from_user;
#ifdef P4EST_DEBUG
    // just to make sure in debug, but otherwise we assume the user knows it should be synchronized
    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &max_L2_norm_u, 1, MPI_DOUBLE, MPI_MAX, p4est_n->mpicomm); SC_CHECK_MPI(mpiret);
#endif
  }

  if(this->vorticity != NULL){
    ierr = VecDestroy(this->vorticity); CHKERRXX(ierr); }
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &vorticity); CHKERRXX(ierr);
  if(norm_grad_u_threshold_split_cell < largest_dbl_smaller_than_dbl_max){
    if(this->norm_grad_u != NULL){
      ierr = VecDestroy(this->norm_grad_u); CHKERRXX(ierr); }
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &norm_grad_u); CHKERRXX(ierr);
  }
}

void my_p4est_navier_stokes_t::set_velocities(CF_DIM **vnm1, CF_DIM **vn, const bool set_max_L2_norm_u)
{
  PetscErrorCode ierr;

  double *vnm1_p[P4EST_DIM], *vn_p[P4EST_DIM];
  double max_velocity = 0.0;
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
  {
    // destroy current vectors if needed
    if(this->vnm1_nodes[dir] != NULL){
      ierr = VecDestroy(this->vnm1_nodes[dir]); CHKERRXX(ierr); }
    if(this->vn_nodes[dir] != NULL){
      ierr = VecDestroy(this->vn_nodes[dir]); CHKERRXX(ierr); }
    if(this->vnp1_nodes[dir] != NULL){
      ierr = VecDestroy(this->vnp1_nodes[dir]); CHKERRXX(ierr); }
    if(this->vstar[dir] != NULL){
      ierr = VecDestroy(this->vstar[dir]); CHKERRXX(ierr); }
    if(this->vnp1[dir] != NULL){
      ierr = VecDestroy(this->vnp1[dir]); CHKERRXX(ierr); }
    // create vectors
    ierr = VecCreateGhostNodes(p4est_nm1, nodes_nm1, &vnm1_nodes[dir]); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &vn_nodes[dir]);       CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &vnp1_nodes[dir]);     CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est_n, faces_n, &vstar[dir], dir);     CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est_n, faces_n, &vnp1[dir], dir);      CHKERRXX(ierr);

    // get pointers for sampling given functions at nodes
    ierr = VecGetArray(vnm1_nodes[dir], &vnm1_p[dir]); CHKERRXX(ierr);
    ierr = VecGetArray(vn_nodes[dir], &vn_p[dir]); CHKERRXX(ierr);
  }

  for(size_t n = 0; n < MAX(nodes_nm1->indep_nodes.elem_count, nodes_n->indep_nodes.elem_count); ++n)
  {
    double xyz[P4EST_DIM];
    if(n < nodes_nm1->indep_nodes.elem_count)
    {
      node_xyz_fr_n(n, p4est_nm1, nodes_nm1, xyz);
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
        vnm1_p[dir][n] = (*vnm1[dir])(xyz);
    }
    if(n < nodes_n->indep_nodes.elem_count)
    {
      node_xyz_fr_n(n, p4est_n, nodes_n, xyz);
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
        vn_p[dir][n] = (*vn[dir])(xyz);
      if(set_max_L2_norm_u)
        max_velocity = MAX(max_velocity, sqrt(SUMD(SQR(vn_p[dir::x][n]), SQR(vn_p[dir::y][n]), SQR(vn_p[dir::z][n]))));
    }
  }

  for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
  {
    // restore pointers
    ierr = VecRestoreArray(vnm1_nodes[dir], &vnm1_p[dir]);  CHKERRXX(ierr);
    ierr = VecRestoreArray(vn_nodes[dir], &vn_p[dir]);      CHKERRXX(ierr);
    // create vectors for second derivatives of node-sampled velocities
    for (unsigned char dd = 0; dd < P4EST_DIM; ++dd) {
      if(second_derivatives_vn_nodes[dd][dir] != NULL){
        ierr = VecDestroy(second_derivatives_vn_nodes[dd][dir]); CHKERRXX(ierr); }
      if(second_derivatives_vnm1_nodes[dd][dir] != NULL){
        ierr = VecDestroy(second_derivatives_vnm1_nodes[dd][dir]); CHKERRXX(ierr); }
      ierr = VecCreateGhostNodes(p4est_n, nodes_n, &second_derivatives_vn_nodes[dd][dir]); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est_nm1, nodes_nm1, &second_derivatives_vnm1_nodes[dd][dir]); CHKERRXX(ierr);
    }
  }

  ngbd_n->second_derivatives_central(vn_nodes, DIM(second_derivatives_vn_nodes[0], second_derivatives_vn_nodes[1], second_derivatives_vn_nodes[2]), P4EST_DIM);
  ngbd_nm1->second_derivatives_central(vnm1_nodes, DIM(second_derivatives_vnm1_nodes[0], second_derivatives_vnm1_nodes[1], second_derivatives_vnm1_nodes[2]), P4EST_DIM);

  if(set_max_L2_norm_u) {
    int mpiret = MPI_Allreduce(&max_velocity, &max_L2_norm_u, 1, MPI_DOUBLE, MPI_MAX, p4est_n->mpicomm); SC_CHECK_MPI(mpiret); }

  if(this->vorticity != NULL){
    ierr = VecDestroy(this->vorticity); CHKERRXX(ierr); }
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &vorticity); CHKERRXX(ierr);
  if(norm_grad_u_threshold_split_cell < largest_dbl_smaller_than_dbl_max){
    if(this->norm_grad_u != NULL){
      ierr = VecDestroy(this->norm_grad_u); CHKERRXX(ierr); }
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &norm_grad_u); CHKERRXX(ierr);
  }
}

void my_p4est_navier_stokes_t::set_vstar(Vec *vstar)
{
  for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
  {
    if(this->vstar[dir] != NULL && this->vstar[dir] != vstar[dir]){
      PetscErrorCode ierr = VecDestroy(this->vstar[dir]); CHKERRXX(ierr); }
    this->vstar[dir] = vstar[dir];
  }
}

void my_p4est_navier_stokes_t::set_hodge(Vec hodge)
{
  if(this->hodge != NULL && this->hodge != hodge){
    PetscErrorCode ierr = VecDestroy(this->hodge); CHKERRXX(ierr); }
  this->hodge = hodge;
}

void my_p4est_navier_stokes_t::compute_max_L2_norm_u()
{
  PetscErrorCode ierr;
  max_L2_norm_u = 0;

  const double *phi_p;
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);

  double dxmax = MAX(DIM(dxyz_min[0], dxyz_min[1], dxyz_min[2]));

  const double *v_p[P4EST_DIM];
  for(unsigned char dir = 0; dir < P4EST_DIM; ++dir) { ierr = VecGetArrayRead(vnp1_nodes[dir], &v_p[dir]); CHKERRXX(ierr); }

  for(p4est_locidx_t n = 0; n < nodes_n->num_owned_indeps; ++n)
    if(phi_p[n] < dxmax)
      max_L2_norm_u = MAX(max_L2_norm_u, sqrt(SUMD(SQR(v_p[0][n]), SQR(v_p[1][n]), SQR(v_p[2][n]))));

  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
  for(unsigned char dir = 0; dir < P4EST_DIM; ++dir) { ierr = VecRestoreArrayRead(vnp1_nodes[dir], &v_p[dir]); CHKERRXX(ierr); }

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &max_L2_norm_u, 1, MPI_DOUBLE, MPI_MAX, p4est_n->mpicomm); SC_CHECK_MPI(mpiret);
}

void my_p4est_navier_stokes_t::compute_vorticity()
{
  PetscErrorCode ierr;

  quad_neighbor_nodes_of_node_t qnnn;

  const double *vnp1_p[P4EST_DIM];
  for(unsigned char dir = 0; dir < P4EST_DIM; dir++) { ierr = VecGetArrayRead(vnp1_nodes[dir], &vnp1_p[dir]); CHKERRXX(ierr); }

  double *vorticity_p, *norm_grad_u_p;
  ierr = VecGetArray(vorticity, &vorticity_p); CHKERRXX(ierr);
  norm_grad_u_p = NULL;
  if(norm_grad_u != NULL){
    ierr = VecGetArray(norm_grad_u, &norm_grad_u_p); CHKERRXX(ierr); }

  double** grad_v_loc = new double*[P4EST_DIM];
  for(u_char dim = 0; dim < P4EST_DIM; dim++)
    grad_v_loc[dim] = new double [P4EST_DIM];

  for(size_t i=0; i<ngbd_n->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n->get_layer_node(i);
    ngbd_n->get_neighbors(n, qnnn);
    qnnn.gradient(vnp1_p, grad_v_loc, P4EST_DIM);
    double vz = grad_v_loc[1][0] - grad_v_loc[0][1];
#ifdef P4_TO_P8
    double vx = grad_v_loc[2][1] - grad_v_loc[1][2];
    double vy = grad_v_loc[0][2] - grad_v_loc[2][0];
    vorticity_p[n] = sqrt(vx*vx + vy*vy + vz*vz);
#else
    vorticity_p[n] = vz;
#endif
    if(norm_grad_u_p != NULL)
      norm_grad_u_p[n] = SUMD(SUMD(fabs(grad_v_loc[0][0]), fabs(grad_v_loc[0][1]), fabs(grad_v_loc[0][2])),
          SUMD(fabs(grad_v_loc[1][0]), fabs(grad_v_loc[1][1]), fabs(grad_v_loc[1][2])),
          SUMD(fabs(grad_v_loc[2][0]), fabs(grad_v_loc[2][1]), fabs(grad_v_loc[2][2])));
  }

  ierr = VecGhostUpdateBegin(vorticity, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  if(norm_grad_u_p != NULL){
    ierr = VecGhostUpdateBegin(norm_grad_u, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr); }

  for(size_t i=0; i<ngbd_n->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n->get_local_node(i);
    ngbd_n->get_neighbors(n, qnnn);
    qnnn.gradient(vnp1_p, grad_v_loc, P4EST_DIM);
    double vz = grad_v_loc[1][0] - grad_v_loc[0][1];
#ifdef P4_TO_P8
    double vx = grad_v_loc[2][1] - grad_v_loc[1][2];
    double vy = grad_v_loc[0][2] - grad_v_loc[2][0];
    vorticity_p[n] = sqrt(vx*vx + vy*vy + vz*vz);
#else
    vorticity_p[n] = vz;
#endif
    if(norm_grad_u_p != NULL)
      norm_grad_u_p[n] = SUMD(SUMD(fabs(grad_v_loc[0][0]), fabs(grad_v_loc[0][1]), fabs(grad_v_loc[0][2])),
          SUMD(fabs(grad_v_loc[1][0]), fabs(grad_v_loc[1][1]), fabs(grad_v_loc[1][2])),
          SUMD(fabs(grad_v_loc[2][0]), fabs(grad_v_loc[2][1]), fabs(grad_v_loc[2][2])));
  }

  ierr = VecGhostUpdateEnd(vorticity, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecRestoreArray(vorticity, &vorticity_p); CHKERRXX(ierr);
  if(norm_grad_u != NULL){
    ierr = VecGhostUpdateEnd(norm_grad_u, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecRestoreArray(norm_grad_u, &norm_grad_u_p); CHKERRXX(ierr);
  }

  for(unsigned char dir = 0; dir < P4EST_DIM; dir++) {
    ierr = VecRestoreArrayRead(vnp1_nodes[dir], &vnp1_p[dir]); CHKERRXX(ierr);
    delete [] grad_v_loc[dir];
  }
  delete [] grad_v_loc;
}

void my_p4est_navier_stokes_t::compute_Q_and_lambda_2_value(Vec& Q_value_nodes, Vec& lambda_2_nodes, const double U_scaling, const double x_scaling) const
{
  PetscErrorCode ierr;

  quad_neighbor_nodes_of_node_t qnnn;

  const double *vnp1_p[P4EST_DIM];
  for(unsigned char dir = 0; dir < P4EST_DIM; dir++) { ierr = VecGetArrayRead(vnp1_nodes[dir], &vnp1_p[dir]); CHKERRXX(ierr); }

  double *Q_value_nodes_p, *lambda_2_nodes_p;
  ierr = VecGetArray(Q_value_nodes, &Q_value_nodes_p); CHKERRXX(ierr);
  ierr = VecGetArray(lambda_2_nodes, &lambda_2_nodes_p); CHKERRXX(ierr);

  for(size_t i=0; i<ngbd_n->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n->get_layer_node(i);
    ngbd_n->get_neighbors(n, qnnn);

    get_Q_and_lambda_2_values(qnnn, vnp1_p, x_scaling, U_scaling, Q_value_nodes_p[n], lambda_2_nodes_p[n]);
  }

  ierr = VecGhostUpdateBegin(Q_value_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateBegin(lambda_2_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  for(size_t i=0; i<ngbd_n->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n->get_local_node(i);
    ngbd_n->get_neighbors(n, qnnn);

    get_Q_and_lambda_2_values(qnnn, vnp1_p, x_scaling, U_scaling, Q_value_nodes_p[n], lambda_2_nodes_p[n]);
  }

  ierr = VecGhostUpdateEnd(Q_value_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(lambda_2_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  for(unsigned char dir = 0; dir < P4EST_DIM; dir++) { ierr = VecRestoreArrayRead(vnp1_nodes[dir], &vnp1_p[dir]); CHKERRXX(ierr); }
  ierr = VecRestoreArray(Q_value_nodes,   &Q_value_nodes_p);  CHKERRXX(ierr);
  ierr = VecRestoreArray(lambda_2_nodes,  &lambda_2_nodes_p); CHKERRXX(ierr);
}


double my_p4est_navier_stokes_t::compute_dxyz_hodge(const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const unsigned char& dir)
{
  PetscErrorCode ierr;

  p4est_quadrant_t *quad;
  if(quad_idx < p4est_n->local_num_quadrants)
  {
    p4est_tree_t *tree = p4est_tree_array_index(p4est_n->trees, tree_idx);
    quad = p4est_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);
  }
  else
    quad = p4est_quadrant_array_index(&ghost_n->ghosts, quad_idx - p4est_n->local_num_quadrants);

  const double *hodge_p;
  ierr = VecGetArrayRead(hodge, &hodge_p); CHKERRXX(ierr);

  if(is_quad_Wall(p4est_n, tree_idx, quad, dir))
  {
    double xyz_wall[P4EST_DIM];
    quad_xyz_fr_q(quad_idx, tree_idx, p4est_n, ghost_n, xyz_wall);
    const double hh = convert_to_xyz[dir/2]*((double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN);
    xyz_wall[dir/2] += (dir%2 == 1 ? +0.5 : -0.5)*hh;
    const double hodge_q = hodge_p[quad_idx];
    ierr = VecRestoreArrayRead(hodge, &hodge_p); CHKERRXX(ierr);
    if(bc_hodge.wallType(xyz_wall) == NEUMANN)
      return (dir%2 == 1 ? +1.0 : -1.0)*bc_hodge.wallValue(xyz_wall);
    else
    {
      P4EST_ASSERT(bc_hodge.wallType(xyz_wall) == DIRICHLET);
      return (dir%2 == 1 ? +1.0 : -1.0)*(bc_hodge.wallValue(xyz_wall) - hodge_q)*2.0/hh;
    }
  }
  else
  {
    set_of_neighboring_quadrants ngbd; ngbd.clear();
    ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, dir);

    /* multiple neighbor cells should never happen since this function is called for a given face,
     * and the faces are defined only for small cells.
     */
    if(ngbd.size() > 1)
      throw std::invalid_argument("my_p4est_navier_stokes::compute_dxyz_hodge: invalid case.");
    /* one neighbor cell of same size, check for interface */
    else if(ngbd.begin()->level == quad->level)
    {
      double xyz_q[P4EST_DIM], xyz_0[P4EST_DIM];
      quad_xyz_fr_q(quad_idx, tree_idx, p4est_n, ghost_n, xyz_q);
      quad_xyz_fr_q(ngbd.begin()->p.piggy3.local_num, ngbd.begin()->p.piggy3.which_tree, p4est_n, ghost_n, xyz_0);

      const double phi_q = (*interp_phi)(xyz_q);
      const double phi_0 = (*interp_phi)(xyz_0);

      const double hh = convert_to_xyz[dir/2]*((double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN);

      if(bc_hodge.interfaceType() == DIRICHLET && phi_q*phi_0 < 0.0)
      {
        const double theta = MAX(EPS, MIN(fraction_Interval_Covered_By_Irregular_Domain(phi_q, phi_0, hh, hh), 1.0));
        double xyz_interface[P4EST_DIM];
        for (unsigned char dd = 0; dd < P4EST_DIM; ++dd)
        {
          xyz_interface[dd] = (phi_q < 0.0 ? xyz_q[dd] : xyz_0[dd]);
          if(dd == dir/2)
            xyz_interface[dd] += (dir%2 == (phi_q < 0.0 ? 1 : 0) ? +1.0 : -1.0)*hh*theta;
        }
        const double val_interface = bc_hodge.interfaceValue(xyz_interface);

        double grad_hodge;
        if(phi_q < 0.0)
          grad_hodge = (dir%2 == 1 ? +1.0 : -1.0)*(val_interface - hodge_p[quad_idx])/(hh*theta);
        else
          grad_hodge = (dir%2 == 1 ? +1.0 : -1.0)*(hodge_p[ngbd.begin()->p.piggy3.local_num] - val_interface)/(hh*theta);

        ierr = VecRestoreArrayRead(hodge, &hodge_p); CHKERRXX(ierr);
        return grad_hodge;
      }
      else
      {
        double grad_hodge = (dir%2 == 1 ? +1.0 : -1.0)*(hodge_p[ngbd.begin()->p.piggy3.local_num] - hodge_p[quad_idx])/hh;

        ierr = VecRestoreArrayRead(hodge, &hodge_p); CHKERRXX(ierr);
        return grad_hodge;
      }
    }
    /* one neighbor cell that is bigger, get common neighbors */
    else
    {
      p4est_quadrant_t quad_tmp = *ngbd.begin();
      ngbd.clear();
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_tmp.p.piggy3.local_num, quad_tmp.p.piggy3.which_tree, dir%2 == 0 ? dir + 1 : dir - 1);

      double dist = 0.0;
      double grad_hodge = 0.0;
      double d0 = (double)P4EST_QUADRANT_LEN(quad_tmp.level)/(double)P4EST_ROOT_LEN;

      for(set_of_neighboring_quadrants::const_iterator it = ngbd.begin(); it != ngbd.end(); ++it)
      {
        double dm = (double)P4EST_QUADRANT_LEN(it->level)/(double)P4EST_ROOT_LEN;
        dist += pow(dm, P4EST_DIM - 1)*0.5*(d0 + dm);
        grad_hodge += (dir%2 == 1 ? 1.0 : -1.0)*(hodge_p[quad_tmp.p.piggy3.local_num] - hodge_p[it->p.piggy3.local_num])*pow(dm, P4EST_DIM - 1);
      }
      grad_hodge /= (convert_to_xyz[dir/2]*dist);

      ierr = VecRestoreArrayRead(hodge, &hodge_p); CHKERRXX(ierr);
      return grad_hodge;
    }
  }
}

double my_p4est_navier_stokes_t::compute_divergence(p4est_locidx_t quad_idx, p4est_topidx_t tree_idx)
{
  PetscErrorCode ierr;
  p4est_tree_t *tree = p4est_tree_array_index(p4est_n->trees, tree_idx);
  p4est_quadrant_t *quad = p4est_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);

  double dmin = (double)P4EST_QUADRANT_LEN(quad->level) / (double)P4EST_ROOT_LEN;
  double val = 0;

  set_of_neighboring_quadrants ngbd;
  p4est_quadrant_t quad_tmp;

  for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
  {
    const double *vstar_p;
    ierr = VecGetArrayRead(vstar[dir], &vstar_p); CHKERRXX(ierr);

    double vm = 0;
    if(is_quad_Wall(p4est_n, tree_idx, quad, 2*dir))
      vm = vstar_p[faces_n->q2f(quad_idx, 2*dir)];
    else if(faces_n->q2f(quad_idx, 2*dir) != NO_VELOCITY)
    {
      ngbd.clear();
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 2*dir);
      quad_tmp = *ngbd.begin();
      ngbd.clear();
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_tmp.p.piggy3.local_num, quad_tmp.p.piggy3.which_tree, 2*dir + 1);
      for(set_of_neighboring_quadrants::const_iterator it = ngbd.begin(); it != ngbd.end(); ++it)
        vm += pow((double)P4EST_QUADRANT_LEN(it->level)/(double)P4EST_ROOT_LEN, P4EST_DIM - 1) * vstar_p[faces_n->q2f(it->p.piggy3.local_num, 2*dir)];
      vm /= pow((double)P4EST_QUADRANT_LEN(quad_tmp.level)/(double)P4EST_ROOT_LEN, P4EST_DIM - 1);
    }
    else
    {
      ngbd.clear();
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 2*dir);
      for(set_of_neighboring_quadrants::const_iterator it = ngbd.begin(); it != ngbd.end(); ++it)
        vm += pow((double)P4EST_QUADRANT_LEN(it->level)/(double)P4EST_ROOT_LEN, P4EST_DIM - 1) * vstar_p[faces_n->q2f(it->p.piggy3.local_num, 2*dir + 1)];
      vm /= pow((double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN, P4EST_DIM - 1);
    }

    double vp = 0;
    if(is_quad_Wall(p4est_n, tree_idx, quad, 2*dir + 1))
      vp = vstar_p[faces_n->q2f(quad_idx, 2*dir + 1)];
    else if(faces_n->q2f(quad_idx, 2*dir + 1) != NO_VELOCITY)
    {
      ngbd.clear();
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 2*dir + 1);
      quad_tmp = *ngbd.begin();
      ngbd.clear();
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_tmp.p.piggy3.local_num, quad_tmp.p.piggy3.which_tree, 2*dir);
      for(set_of_neighboring_quadrants::const_iterator it = ngbd.begin(); it != ngbd.end(); ++it)
        vp += pow((double)P4EST_QUADRANT_LEN(it->level)/(double)P4EST_ROOT_LEN, P4EST_DIM - 1) * vstar_p[faces_n->q2f(it->p.piggy3.local_num, 2*dir + 1)];
      vp /= pow((double)P4EST_QUADRANT_LEN(quad_tmp.level)/(double)P4EST_ROOT_LEN, P4EST_DIM - 1);
    }
    else
    {
      ngbd.clear();
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 2*dir + 1);
      for(set_of_neighboring_quadrants::const_iterator it = ngbd.begin(); it != ngbd.end(); ++it)
        vp += pow((double)P4EST_QUADRANT_LEN(it->level)/(double)P4EST_ROOT_LEN, P4EST_DIM - 1) * vstar_p[faces_n->q2f(it->p.piggy3.local_num, 2*dir)];
      vp /= pow((double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN, P4EST_DIM - 1);
    }

    ierr = VecRestoreArrayRead(vstar[dir], &vstar_p); CHKERRXX(ierr);

    val += (vp - vm)/(convert_to_xyz[dir] * dmin);
  }

  return val;
}

void my_p4est_navier_stokes_t::solve_viscosity(my_p4est_poisson_faces_t* &face_poisson_solver, const bool& use_initial_guess, const KSPType& ksp, const PCType& pc)
{
  PetscErrorCode ierr;

  ierr = PetscLogEventBegin(log_my_p4est_navier_stokes_viscosity, 0, 0, 0, 0); CHKERRXX(ierr);
  if(ns_time_step_analyzer.is_on())
    ns_time_step_analyzer.start(viscous_step);

  /* construct the right hand side */

  vector<double> xyz_n[P4EST_DIM][P4EST_DIM];   // xyz_n[dir][comp][f_idx] = comp^th Cartesian coordinate of the backtraced point at time n calculated from the f_idx^th face of orientation dir
  vector<double> xyz_nm1[P4EST_DIM][P4EST_DIM]; // xyz_nm1[dir][comp][f_idx] = comp^th Cartesian coordinate of the backtraced point time (n - 1) calculated from the f_idx^th face of orientation dir (required only if sl_order == 2)
  if(!semi_lagrangian_backtrace_is_done)
    trajectory_from_np1_all_faces(faces_n, ngbd_nm1, ngbd_n,
                                  vnm1_nodes, second_derivatives_vnm1_nodes,
                                  vn_nodes, second_derivatives_vn_nodes, dt_nm1, dt_n,
                                  xyz_n, (sl_order == 2 ? xyz_nm1 : NULL));

  // rhs is modified by the solver so it should be reset every-time (could be modified, if required)
  Vec rhs[P4EST_DIM];
  for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
  {
    if(!semi_lagrangian_backtrace_is_done)
    {
      /* find the velocity at the backtraced points */
      my_p4est_interpolation_nodes_t interp_nm1(ngbd_nm1);
      my_p4est_interpolation_nodes_t interp_n  (ngbd_n  );
      for(p4est_locidx_t f_idx = 0; f_idx < faces_n->num_local[dir]; ++f_idx)
      {
        double xyz_tmp[P4EST_DIM];

        if(sl_order == 2)
        {
          for (unsigned char dd = 0; dd < P4EST_DIM; ++dd)
            xyz_tmp[dd] = xyz_nm1[dir][dd][f_idx];
          interp_nm1.add_point(f_idx, xyz_tmp);
        }

        for (unsigned char dd = 0; dd < P4EST_DIM; ++dd)
          xyz_tmp[dd] = xyz_n[dir][dd][f_idx];

        interp_n.add_point(f_idx, xyz_tmp);
      }

      if(sl_order == 2)
      {
        backtraced_v_nm1[dir].resize(faces_n->num_local[dir]);
        interp_nm1.set_input(vnm1_nodes[dir], DIM(second_derivatives_vnm1_nodes[0][dir], second_derivatives_vnm1_nodes[1][dir], second_derivatives_vnm1_nodes[2][dir]), interp_v_viscosity);
        interp_nm1.interpolate(backtraced_v_nm1[dir].data());
      }

      backtraced_v_n[dir].resize(faces_n->num_local[dir]);
      interp_n.set_input(vn_nodes[dir], DIM(second_derivatives_vn_nodes[0][dir], second_derivatives_vn_nodes[1][dir], second_derivatives_vn_nodes[2][dir]), interp_v_viscosity);
      interp_n.interpolate(backtraced_v_n[dir].data());

      if (boussinesq_approx) {
          if (dir == 1){


              for(unsigned char direction = 0; direction < P4EST_DIM; ++direction){
                  //if (second_derivatives_external_force_per_unit_volume[direction]!=NULL){
                  //    ierr = VecDestroy(second_derivatives_external_force_per_unit_volume[direction]); CHKERRXX(ierr);
                  ierr = VecCreateGhostNodes(p4est_n,nodes_n,&second_derivatives_external_force_per_unit_volume[direction]); CHKERRXX(ierr);
                  //}
              }

              ngbd_n->second_derivatives_central(external_force_per_unit_volume,DIM(second_derivatives_external_force_per_unit_volume[0],second_derivatives_external_force_per_unit_volume[1],second_derivatives_external_force_per_unit_volume[2]));

              ext_force_term_interpolation_from_nodes_to_faces.resize(faces_n->num_local[dir]);

              interp_n.set_input(external_force_per_unit_volume,DIM(second_derivatives_external_force_per_unit_volume[0],second_derivatives_external_force_per_unit_volume[1],second_derivatives_external_force_per_unit_volume[2]),quadratic);

              interp_n.interpolate(ext_force_term_interpolation_from_nodes_to_faces.data());
              for (unsigned char direction=0; direction < P4EST_DIM; ++direction){
                  ierr= VecDestroy(second_derivatives_external_force_per_unit_volume[direction]); CHKERRXX(ierr);
              }

              ierr= VecDestroy(external_force_per_unit_volume); CHKERRXX(ierr);
            }
        }

     }

//    if (boussinesq_approx){
//        PetscPrintf(p4est_n->mpicomm,"Hello world 4\n");
//        if (dir::y){
//          PetscPrintf(p4est_n->mpicomm,"Hello world 5\n");
//          my_p4est_interpolation_nodes_t interp_external_force_per_unit_volume  (ngbd_n  );
//          for(p4est_locidx_t f_idx=0; f_idx < faces_n->num_local[dir]; ++f_idx){
//              double xyz_tmp_[P4EST_DIM];
//              for (unsigned char dd = 0; dd < P4EST_DIM; ++dd){
//                 xyz_tmp_[dd] = xyz_n[dir][dd][f_idx];
//              }
//              interp_external_force_per_unit_volume.add_point(f_idx, xyz_tmp_);
//          }
//          ext_force_term_interpolation_from_nodes_to_faces.resize(faces_n->num_local[dir]);
//          for (unsigned char dd=0; dd < P4EST_DIM; ++dd){
//              ierr= VecCreateGhostNodes(p4est_n,nodes_n,&second_derivatives_external_force_per_unit_volume[dd]); CHKERRXX(ierr);
//          }
//          ngbd_n->second_derivatives_central(external_force_per_unit_volume,DIM(second_derivatives_external_force_per_unit_volume[0],second_derivatives_external_force_per_unit_volume[1],second_derivatives_external_force_per_unit_volume[2]));
//          interp_external_force_per_unit_volume.set_input(external_force_per_unit_volume,DIM(second_derivatives_external_force_per_unit_volume[0],second_derivatives_external_force_per_unit_volume[1],second_derivatives_external_force_per_unit_volume[2]),quadratic);
//          interp_external_force_per_unit_volume.interpolate(ext_force_term_interpolation_from_nodes_to_faces.data());
//          for (unsigned char dd=0; dd < P4EST_DIM; ++dd){
//              ierr= VecDestroy(second_derivatives_external_force_per_unit_volume[dd]); CHKERRXX(ierr);
//          }
//        }
//    }


    /* assemble the right-hand-side */
    ierr = VecCreateNoGhostFaces(p4est_n, faces_n, &rhs[dir], dir); CHKERRXX(ierr);

    const PetscScalar *face_is_well_defined_p;
    ierr = VecGetArrayRead(face_is_well_defined[dir], &face_is_well_defined_p); CHKERRXX(ierr);

    double *rhs_p;
    ierr = VecGetArray(rhs[dir], &rhs_p); CHKERRXX(ierr);

    for(p4est_locidx_t f_idx = 0; f_idx < faces_n->num_local[dir]; ++f_idx)
    {
      if(face_is_well_defined_p[f_idx])
      {
        if(sl_order == 1)
          rhs_p[f_idx] = rho/dt_n * backtraced_v_n[dir][f_idx]; // alpha == 1, beta == 0 if sl_order == 1
        else
          rhs_p[f_idx] = -rho * ((-alpha()/dt_n + beta()/dt_nm1)*backtraced_v_n[dir][f_idx] - beta()/dt_nm1*backtraced_v_nm1[dir][f_idx]);

        if(external_forces_per_unit_volume[dir] != NULL || external_forces_per_unit_mass[dir] != NULL)
        {
          P4EST_ASSERT((external_forces_per_unit_volume[dir] != NULL && external_forces_per_unit_mass[dir] == NULL) || (external_forces_per_unit_volume[dir] == NULL && external_forces_per_unit_mass[dir] != NULL));
          double xyz[P4EST_DIM]; faces_n->xyz_fr_f(f_idx, dir, xyz);
          rhs_p[f_idx] += (external_forces_per_unit_volume[dir] != NULL ? (*external_forces_per_unit_volume[dir])(xyz) : rho*(*external_forces_per_unit_mass[dir])(xyz));
        }

        if (boussinesq_approx)
        {
            if (dir==1){
                rhs_p[f_idx]+=ext_force_term_interpolation_from_nodes_to_faces[f_idx];
            }
        }
      }
      else
        rhs_p[f_idx] = 0;
    }

    ierr = VecRestoreArray(rhs[dir], &rhs_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(face_is_well_defined[dir], &face_is_well_defined_p); CHKERRXX(ierr);
    //ierr = VecDestroy(external_force_per_unit_volume[dir]); CHKERRXX(ierr);
  }
  semi_lagrangian_backtrace_is_done = true;

  if(face_poisson_solver == NULL)
  {
    face_poisson_solver = new my_p4est_poisson_faces_t(faces_n, ngbd_n);
    face_poisson_solver->set_phi(phi);
    face_poisson_solver->set_bc(bc_v, dxyz_hodge, face_is_well_defined, &bc_hodge);
// [Raphael:] I decided to deactivate this, I don't see why it was done like this in the
// first place except for avoiding memory issues perhaps but that would be surprising, imo...
//#if defined(COMET) || defined(STAMPEDE) || defined(POD_CLUSTER)
//    face_poisson_solver->set_compute_partition_on_the_fly(true);
//#else
    face_poisson_solver->set_compute_partition_on_the_fly(false);
//#endif
  }

  face_poisson_solver->set_mu(mu);
  face_poisson_solver->set_diagonal(alpha()*rho/dt_n);
  face_poisson_solver->set_rhs(rhs);

  face_poisson_solver->solve(vstar, use_initial_guess, ksp, pc);

  for(unsigned char dir = 0; dir < P4EST_DIM; ++dir){
    ierr = VecDestroy(rhs[dir]); CHKERRXX(ierr);
  }

  if(bc_pressure->interfaceType() != NOINTERFACE)
  {
    my_p4est_level_set_faces_t lsf(ngbd_n, faces_n);
    P4EST_ASSERT(interp_phi != NULL && interp_grad_phi != NULL);
    for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      lsf.geometric_extrapolation_over_interface(vstar[dir], *interp_phi, *interp_grad_phi, bc_v[dir], dir, face_is_well_defined[dir], dxyz_hodge[dir], 2, 2); //lsf.extend_over_interface(phi, vstar[dir], bc_v[dir], dir, face_is_well_defined[dir], dxyz_hodge[dir], 2, 2);
  }

  if(ns_time_step_analyzer.is_on())
    ns_time_step_analyzer.stop();
  ierr = PetscLogEventEnd(log_my_p4est_navier_stokes_viscosity, 0, 0, 0, 0); CHKERRXX(ierr);
}

/* solve the projection step
 * laplace Hodge = -div(vstar)
 */
double my_p4est_navier_stokes_t::solve_projection(my_p4est_poisson_cells_t* &cell_solver, const bool& use_initial_guess, const KSPType& ksp, const PCType& pc,
                                                  const bool& shift_to_zero_mean_if_floating, Vec hodge_old, Vec former_dxyz_hodge[P4EST_DIM], const hodge_control& hodge_chek)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_navier_stokes_projection, 0, 0, 0, 0); CHKERRXX(ierr);
  if(ns_time_step_analyzer.is_on())
    ns_time_step_analyzer.start(projection_step);

  double convergence_check_on_hodge = 0.0;

  Vec rhs;
  ierr = VecCreateNoGhostCells(p4est_n, &rhs); CHKERRXX(ierr);

  double *rhs_p;
  ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);

  /* compute the right-hand-side */
  for(p4est_topidx_t tree_idx = p4est_n->first_local_tree; tree_idx <= p4est_n->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = p4est_tree_array_index(p4est_n->trees, tree_idx);
    for(size_t q_idx = 0; q_idx < tree->quadrants.elem_count; ++q_idx)
    {
      p4est_locidx_t quad_idx = q_idx + tree->quadrants_offset;
      rhs_p[quad_idx] = -compute_divergence(quad_idx, tree_idx);
    }
  }
  ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);

  /* solve the linear system */
  if(cell_solver == NULL)
  {
    cell_solver = new my_p4est_poisson_cells_t(ngbd_c, ngbd_n);
    cell_solver->set_phi(phi);
    cell_solver->set_bc(bc_hodge);
    cell_solver->set_nullspace_use_fixed_point(false);
  }
  cell_solver->set_mu(1.0);
  cell_solver->set_rhs(rhs);
  cell_solver->solve(hodge, use_initial_guess, ksp, pc);

  ierr = VecDestroy(rhs); CHKERRXX(ierr);

  // if desired and needed, shift the hodge variable to a zero average
  // compute difference in hodge variables on-the-fly if the information was given
  const bool shift_to_zero_mean = (shift_to_zero_mean_if_floating && cell_solver->get_matrix_has_nullspace());
  const bool compute_max_diff_in_hodge_value = (hodge_chek == hodge_value && hodge_old != NULL);
  if(shift_to_zero_mean || compute_max_diff_in_hodge_value)
  {
    double average = 0.0;
    if(shift_to_zero_mean)
    {
      my_p4est_level_set_cells_t lsc(ngbd_c, ngbd_n);
      average = lsc.integrate(phi, hodge) / area_in_negative_domain(p4est_n, nodes_n, phi);
    }
    double *hodge_p;
    const double *phi_p, *hodge_old_p = NULL;
    ierr = VecGetArray(hodge, &hodge_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
    if(compute_max_diff_in_hodge_value){
      ierr = VecGetArrayRead(hodge_old, &hodge_old_p); CHKERRXX(ierr); }

    for (size_t i = 0; i < hierarchy_n->get_layer_size(); ++i) {
      p4est_locidx_t quad_idx = hierarchy_n->get_local_index_of_layer_quadrant(i);
      if(shift_to_zero_mean)
        hodge_p[quad_idx] -= average;
      if(compute_max_diff_in_hodge_value && quadrant_value_is_well_defined(bc_hodge, p4est_n, ghost_n, nodes_n, quad_idx, hierarchy_n->get_tree_index_of_layer_quadrant(i), phi_p))
        convergence_check_on_hodge = MAX(convergence_check_on_hodge, fabs(hodge_p[quad_idx] - hodge_old_p[quad_idx]));
    }

    if(shift_to_zero_mean){
      ierr = VecGhostUpdateBegin(hodge, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr); }

    for (size_t i = 0; i < hierarchy_n->get_inner_size(); ++i) {
      p4est_locidx_t quad_idx = hierarchy_n->get_local_index_of_inner_quadrant(i);
      if(shift_to_zero_mean)
        hodge_p[quad_idx] -= average;
      if(compute_max_diff_in_hodge_value && quadrant_value_is_well_defined(bc_hodge, p4est_n, ghost_n, nodes_n, quad_idx, hierarchy_n->get_tree_index_of_inner_quadrant(i), phi_p))
        convergence_check_on_hodge = MAX(convergence_check_on_hodge, fabs(hodge_p[quad_idx] - hodge_old_p[quad_idx]));
    }

    if(shift_to_zero_mean){
      ierr = VecGhostUpdateEnd(hodge, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr); }

    ierr = VecRestoreArray(hodge, &hodge_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
    if(compute_max_diff_in_hodge_value){
      ierr = VecRestoreArrayRead(hodge_old, &hodge_old_p); CHKERRXX(ierr); }
  }

  if(bc_pressure->interfaceType() != NOINTERFACE)
  {
    my_p4est_level_set_cells_t lsc(ngbd_c, ngbd_n);
    P4EST_ASSERT(interp_grad_phi != NULL);
    lsc.geometric_extrapolation_over_interface(hodge, phi, *interp_grad_phi, bc_hodge, 2, 2);
//    lsc.extend_over_interface(phi, hodge, &bc_hodge, 2, 2);
  }

  /* project vstar */
  double *dxyz_hodge_p[P4EST_DIM];
  const double *vstar_p[P4EST_DIM], *former_dxyz_hodge_p[P4EST_DIM];
  const PetscScalar *face_is_well_defined_p[P4EST_DIM];
  bool compute_max_diff_in_grad_hodge[P4EST_DIM];
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    compute_max_diff_in_grad_hodge[dir] = former_dxyz_hodge != NULL && (hodge_chek == dir|| hodge_chek == uvw_components);

  double *vnp1_p[P4EST_DIM];
  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;
  // Projection on faces that are shared with other processors
  for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
  {
    ierr = VecGetArrayRead(vstar[dir],  &vstar_p[dir]); CHKERRXX(ierr);
    ierr = VecGetArray(dxyz_hodge[dir], &dxyz_hodge_p[dir]); CHKERRXX(ierr);
    ierr = VecGetArray(vnp1[dir],       &vnp1_p[dir]); CHKERRXX(ierr);
    if(former_dxyz_hodge != NULL)
    {
      ierr = VecGetArrayRead(former_dxyz_hodge[dir], &former_dxyz_hodge_p[dir]); CHKERRXX(ierr);
      ierr = VecGetArrayRead(face_is_well_defined[dir], &face_is_well_defined_p[dir]); CHKERRXX(ierr);
    }
    for(size_t i = 0; i < faces_n->get_layer_size(dir); ++i)
    {
      p4est_locidx_t f_idx = faces_n->get_layer_face(dir, i);
      faces_n->f2q(f_idx, dir, quad_idx, tree_idx);
      int tmp = (faces_n->q2f(quad_idx, 2*dir) == f_idx ? 0 : 1);
      dxyz_hodge_p[dir][f_idx]  = compute_dxyz_hodge(quad_idx, tree_idx, 2*dir + tmp);
      vnp1_p[dir][f_idx]        = vstar_p[dir][f_idx] - dxyz_hodge_p[dir][f_idx];
      if(compute_max_diff_in_grad_hodge[dir] && face_is_well_defined_p[dir][f_idx])
        convergence_check_on_hodge = MAX(convergence_check_on_hodge, fabs(former_dxyz_hodge_p[dir][f_idx] - dxyz_hodge_p[dir][f_idx]));
    }
  }
  // update ghost values on other processes
  for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
  {
    ierr = VecGhostUpdateBegin(dxyz_hodge[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(vnp1[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }
  // Projection on local faces that are not shared with other processors
  for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
  {
    for(size_t i = 0; i < faces_n->get_local_size(dir); ++i)
    {
      p4est_locidx_t f_idx = faces_n->get_local_face(dir, i);
      faces_n->f2q(f_idx, dir, quad_idx, tree_idx);
      int tmp = (faces_n->q2f(quad_idx, 2*dir) == f_idx ? 0 : 1);
      dxyz_hodge_p[dir][f_idx]  = compute_dxyz_hodge(quad_idx, tree_idx, 2*dir + tmp);
      vnp1_p[dir][f_idx]        = vstar_p[dir][f_idx] - dxyz_hodge_p[dir][f_idx];
      if(compute_max_diff_in_grad_hodge[dir] && face_is_well_defined_p[dir][f_idx])
        convergence_check_on_hodge = MAX(convergence_check_on_hodge, fabs(former_dxyz_hodge_p[dir][f_idx] - dxyz_hodge_p[dir][f_idx]));
    }
    ierr = VecRestoreArray(vnp1[dir],       &vnp1_p[dir]); CHKERRXX(ierr);
    ierr = VecRestoreArray(dxyz_hodge[dir], &dxyz_hodge_p[dir]); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(vstar[dir],  &vstar_p[dir]); CHKERRXX(ierr);
    if(former_dxyz_hodge != NULL)
    {
      ierr = VecRestoreArrayRead(former_dxyz_hodge[dir], &former_dxyz_hodge_p[dir]); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(face_is_well_defined[dir], &face_is_well_defined_p[dir]); CHKERRXX(ierr);
    }
  }
  // finish the ghost updates on other processes
  for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
  {
    ierr = VecGhostUpdateEnd(dxyz_hodge[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(vnp1[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }
  if(compute_max_diff_in_hodge_value || ORD(compute_max_diff_in_grad_hodge[0], compute_max_diff_in_grad_hodge[1], compute_max_diff_in_grad_hodge[2]))
  {
    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &convergence_check_on_hodge, 1, MPI_DOUBLE, MPI_MAX, p4est_n->mpicomm); SC_CHECK_MPI(mpiret);
  }
  else
    convergence_check_on_hodge = DBL_MAX; // we have not calculated any convergence check in that case --> don't fool the user...

  if(ns_time_step_analyzer.is_on())
    ns_time_step_analyzer.stop();
  ierr = PetscLogEventEnd(log_my_p4est_navier_stokes_projection, 0, 0, 0, 0); CHKERRXX(ierr);
  return convergence_check_on_hodge;
}

double my_p4est_navier_stokes_t::get_correction_in_hodge_derivative_for_enforcing_mass_flow(const unsigned char& force_direction, const double& desired_mean_velocity, const double* current_mass_flow_p)
{
  P4EST_ASSERT(force_direction < P4EST_DIM);
  double current_mass_flow;
  if(!is_periodic(p4est_n, force_direction))
  {
#ifdef CASL_THROWS
    throw std::invalid_argument("my_p4est_navier_stokes_t::enforce_mass_flow: this function cannot be called to enforce mass flow in a nonperiodic direction");
#else
    return 0.0;
#endif
  }
  if(current_mass_flow_p == NULL)
  {
    double section = xyz_min[force_direction];
    global_mass_flow_through_slice(force_direction, section, current_mass_flow);
  }
  else
    current_mass_flow = *current_mass_flow_p;

  const double current_mean_velocity  = current_mass_flow/MULTD(rho, (xyz_max[(force_direction + 1)%P4EST_DIM] - xyz_min[(force_direction + 1)%P4EST_DIM]), (xyz_max[(force_direction + 2)%P4EST_DIM] - xyz_min[(force_direction + 2)%P4EST_DIM]));

  return current_mean_velocity - desired_mean_velocity;
}

void my_p4est_navier_stokes_t::compute_velocity_at_nodes(const bool store_interpolators)
{
  PetscErrorCode ierr;
  if(ns_time_step_analyzer.is_on())
    ns_time_step_analyzer.start(velocity_interpolation);

  /* interpolate vnp1 from faces to nodes */
  for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
  {
    double *vnp1_nodes_dir_p;
    const double *vnp1_dir_read_p;
    ierr = VecGetArray(vnp1_nodes[dir], &vnp1_nodes_dir_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(vnp1[dir], &vnp1_dir_read_p); CHKERRXX(ierr);

    if(store_interpolators && !interpolators_from_face_to_nodes_are_set)
      interpolator_from_face_to_nodes[dir].resize(nodes_n->num_owned_indeps);

    for(size_t i = 0; i < ngbd_n->get_layer_size(); ++i)
    {
      p4est_locidx_t node_idx = ngbd_n->get_layer_node(i);
      vnp1_nodes_dir_p[node_idx] = compute_velocity_at_local_node(node_idx, dir, vnp1_dir_read_p, store_interpolators);
    }

    ierr = VecGhostUpdateBegin(vnp1_nodes[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    for(size_t i = 0; i < ngbd_n->get_local_size(); ++i)
    {
      p4est_locidx_t node_idx = ngbd_n->get_local_node(i);
      vnp1_nodes_dir_p[node_idx] = compute_velocity_at_local_node(node_idx, dir, vnp1_dir_read_p, store_interpolators);
    }

    ierr = VecGhostUpdateEnd(vnp1_nodes[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecRestoreArray(vnp1_nodes[dir], &vnp1_nodes_dir_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(vnp1[dir], &vnp1_dir_read_p); CHKERRXX(ierr);

    if(bc_pressure->interfaceType() != NOINTERFACE)
    {
      my_p4est_level_set_t lsn(ngbd_n);
//      int phi_size, vnm1_size, vn_size, vnp1_size;
//      VecGetSize(phi, &phi_size);
//      VecGetSize(vnm1_nodes[dir], &vnm1_size);
//      VecGetSize(vn_nodes[dir], &vn_size);
//      VecGetSize(vnp1_nodes[dir], &vnp1_size);
//      printf("phi size = %d, dir = %d, vnm1 = %d, vn = %d, vnp1 = %d \n", phi_size, dir, vnm1_size, vn_size, vnp1_size);
      lsn.extend_Over_Interface_TVD(phi, vnp1_nodes[dir]);
    }
  }

  if(store_interpolators && !interpolators_from_face_to_nodes_are_set)
    interpolators_from_face_to_nodes_are_set = true;

  compute_vorticity();
  compute_max_L2_norm_u();

//  PetscPrintf(p4est_n->mpicomm, "Addresses on rank %d: p4est_n = %p, vnm1 = %p, vn = %p, vnp1 = %p \n",
//              p4est_n->mpirank, p4est_n, vnm1_nodes[0], vn_nodes[0], vnp1_nodes[0]);


  if(ns_time_step_analyzer.is_on())
    ns_time_step_analyzer.stop();

  return;
}

void my_p4est_navier_stokes_t::set_dt(double dt_nm1, double dt_n)
{
  this->dt_nm1 = dt_nm1;
  this->dt_n = dt_n;
  dt_updated = true;
}


void my_p4est_navier_stokes_t::set_dt(double dt_n)
{
  this->dt_n = dt_n;
  dt_updated = true;
}


void my_p4est_navier_stokes_t::set_grids(my_p4est_node_neighbors_t *ngbd_nm1_, my_p4est_node_neighbors_t *ngbd_n_, my_p4est_faces_t *faces){
  // Overall grid properties:
  brick = ngbd_n_->myb;
  conn = ngbd_n_->p4est->connectivity;

  // The nm1 grid:
  p4est_nm1 = ngbd_nm1_->p4est;
  ghost_nm1 = ngbd_nm1_->ghost;
  nodes_nm1 = ngbd_nm1_->nodes;
  hierarchy_nm1 = ngbd_nm1_->hierarchy;
  ngbd_nm1 = ngbd_nm1_;

  // The n grid:
  p4est_n = ngbd_n_->p4est;
  ghost_n = ngbd_n_->ghost;
  nodes_n = ngbd_n_->nodes;
  hierarchy_n = ngbd_n_->hierarchy;
  ngbd_n = ngbd_n_;

  // Cell and face info:
  ngbd_c = faces->get_ngbd_c();
  faces_n = faces;
}


void my_p4est_navier_stokes_t::compute_adapted_dt(double min_value_for_umax)
{
  dt_nm1 = dt_n;
  dt_n = +DBL_MAX;
  double dxmin = MIN(DIM(dxyz_min[0], dxyz_min[1], dxyz_min[2]));
  splitting_criteria_t* data = (splitting_criteria_t*) p4est_n->user_pointer;
  PetscErrorCode ierr;
  const double* v_p[P4EST_DIM];
  for (unsigned char dd = 0; dd < P4EST_DIM; ++dd) {
    ierr = VecGetArrayRead(vnp1_nodes[dd], &v_p[dd]); CHKERRXX(ierr);
  }
  for (p4est_topidx_t tree_idx = p4est_n->first_local_tree; tree_idx <= p4est_n->last_local_tree; ++tree_idx) {
    p4est_tree_t* tree = p4est_tree_array_index(p4est_n->trees, tree_idx);
    for (size_t qq = 0; qq < tree->quadrants.elem_count; ++qq) {
      p4est_locidx_t quad_idx = tree->quadrants_offset + qq;
      double max_local_velocity_magnitude = -1.0;
      p4est_quadrant_t* quad = p4est_quadrant_array_index(&tree->quadrants, qq);
      for (unsigned char child_idx = 0; child_idx < P4EST_CHILDREN; ++child_idx) {
        p4est_locidx_t node_idx = nodes_n->local_nodes[P4EST_CHILDREN*quad_idx + child_idx];
        double node_vel_mag = 0.0;
        for (unsigned char dd = 0; dd < P4EST_DIM; ++dd)
          node_vel_mag += SQR(v_p[dd][node_idx]);
        node_vel_mag = sqrt(node_vel_mag);
        max_local_velocity_magnitude = MAX(max_local_velocity_magnitude, node_vel_mag);
      }
      dt_n = MIN(dt_n, MIN(1.0/max_local_velocity_magnitude, 1.0/min_value_for_umax)*n_times_dt*dxmin*((double) (1<<(data->max_lvl - quad->level))));
    }
  }
  for (unsigned char dd = 0; dd < P4EST_DIM; ++dd) {
    ierr = VecRestoreArrayRead(vnp1_nodes[dd], &v_p[dd]); CHKERRXX(ierr);
  }
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &dt_n, 1, MPI_DOUBLE, MPI_MIN, p4est_n->mpicomm); SC_CHECK_MPI(mpiret);
  dt_updated = true;
}

void my_p4est_navier_stokes_t::compute_dt(double min_value_for_umax)
{
  dt_nm1 = dt_n;
  dt_n = MIN(1/min_value_for_umax, 1/max_L2_norm_u) * n_times_dt * MIN(DIM(dxyz_min[0], dxyz_min[1], dxyz_min[2]));

  dt_updated = true;
}

void my_p4est_navier_stokes_t::advect_smoke(my_p4est_node_neighbors_t* ngbd_n_np1, Vec* vnp1, Vec smoke_np1)
{
  PetscErrorCode ierr;
  std::vector<double> xyz_d[P4EST_DIM];
  trajectory_from_np1_to_n(ngbd_n_np1->p4est, ngbd_n_np1->nodes, ngbd_n_np1, dt_n, vnp1, xyz_d);

  my_p4est_interpolation_nodes_t interp_nodes(ngbd_n);
  for(p4est_locidx_t n = 0; n < ngbd_n_np1->nodes->num_owned_indeps; ++n)
  {
    double xyz_d_[P4EST_DIM] = {DIM(xyz_d[0][n], xyz_d[1][n], xyz_d[2][n])};
    interp_nodes.add_point(n, xyz_d_);
  }
  interp_nodes.set_input(smoke, linear);
  interp_nodes.interpolate(smoke_np1);

  /* enforce boundary condition */
  double *smoke_np1_p;
  ierr = VecGetArray(smoke_np1, &smoke_np1_p); CHKERRXX(ierr);
  for(size_t i = 0; i < ngbd_n_np1->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n_np1->get_layer_node(i);
    double xyz[P4EST_DIM];
    node_xyz_fr_n(n, ngbd_n_np1->p4est, ngbd_n_np1->nodes, xyz);
    smoke_np1_p[n] = MAX(smoke_np1_p[n], (*bc_smoke)(xyz));
  }
  ierr = VecGhostUpdateBegin(smoke_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for(size_t i = 0; i < ngbd_n_np1->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n_np1->get_local_node(i);
    double xyz[P4EST_DIM];
    node_xyz_fr_n(n, ngbd_n_np1->p4est, ngbd_n_np1->nodes, xyz);
    smoke_np1_p[n] = MAX(smoke_np1_p[n], (*bc_smoke)(xyz));
  }
  ierr = VecGhostUpdateEnd(smoke_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecRestoreArray(smoke_np1, &smoke_np1_p); CHKERRXX(ierr);
}

/*
// [RAPHAEL]: this function is not even called anywhere... it flattens boundary conditions for velocity components on the
// interface towards the positive domain --> I've decided to comment it out for clarity
void my_p4est_navier_stokes_t::extrapolate_bc_v(my_p4est_node_neighbors_t *ngbd, Vec *v, Vec phi)
{
  PetscErrorCode ierr;

  double *phi_p;
  ierr = VecGetArray(phi, &phi_p);
  quad_neighbor_nodes_of_node_t qnnn;

  double *v_p[P4EST_DIM];
  for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
  {
    ierr = VecGetArray(v[dir], &v_p[dir]); CHKERRXX(ierr);
  }

  for(size_t i = 0; i < ngbd->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd->get_layer_node(i);
    if(phi_p[n] > 0.0)
    {
      ngbd->get_neighbors(n, qnnn);

      double x = node_x_fr_n(n, ngbd->p4est, ngbd->nodes) - phi_p[n]*qnnn.dx_central(phi_p);
      double y = node_y_fr_n(n, ngbd->p4est, ngbd->nodes) - phi_p[n]*qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
      double z = node_z_fr_n(n, ngbd->p4est, ngbd->nodes) - phi_p[n]*qnnn.dz_central(phi_p);
#endif

      for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
        v_p[dir][n] = bc_v[dir].interfaceValue(DIM(x, y, z));
    }
  }

  for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
  {
    ierr = VecGhostUpdateBegin(v[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  for(size_t i = 0; i < ngbd->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd->get_local_node(i);
    if(phi_p[n] > 0.0)
    {
      ngbd->get_neighbors(n, qnnn);

      double x = node_x_fr_n(n, ngbd->p4est, ngbd->nodes) - phi_p[n]*qnnn.dx_central(phi_p);
      double y = node_y_fr_n(n, ngbd->p4est, ngbd->nodes) - phi_p[n]*qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
      double z = node_z_fr_n(n, ngbd->p4est, ngbd->nodes) - phi_p[n]*qnnn.dz_central(phi_p);
#endif

      for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
        v_p[dir][n] = bc_v[dir].interfaceValue(DIM(x, y, z));
    }
  }

  for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArray(v[dir], &v_p[dir]); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(v[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
}
*/


bool my_p4est_navier_stokes_t::update_from_tn_to_tnp1(const CF_DIM *level_set, bool keep_grid_as_such, bool do_reinitialization)
{
  PetscErrorCode ierr;

  ierr = PetscLogEventBegin(log_my_p4est_navier_stokes_update, 0, 0, 0, 0); CHKERRXX(ierr);
  if(ns_time_step_analyzer.is_on())
  {
    ns_time_step_analyzer.reset(); // this is typically the beginning of a new time step
    ns_time_step_analyzer.start(grid_update);
  }

  if(!dt_updated)
    compute_dt();
  dt_updated = false;

  /* initialize the new forest and nodes, at time np1 */
  p4est_t *p4est_np1 = p4est_n;
  p4est_nodes_t *nodes_np1 = NULL;
  Vec phi_np1 = NULL; double *phi_np1_p = NULL;
  Vec smoke_np1 = NULL;
  my_p4est_interpolation_nodes_t interp_nodes(ngbd_n);
  std::vector<Vec> interp_inputs; interp_inputs.resize(0);
  std::vector<Vec> interp_outputs; interp_outputs.resize(0);

  bool grid_is_unchanged = true;
  bool iterative_grid_update_converged = false;
  if(!keep_grid_as_such)
  {
    splitting_criteria_t *data = (splitting_criteria_t*)p4est_n->user_pointer;
    splitting_criteria_vorticity_t criteria(data->min_lvl, data->max_lvl, data->lip, uniform_band, vorticity_threshold_split_cell, max_L2_norm_u, smoke_thresh, norm_grad_u_threshold_split_cell);
    /* construct a new forest */
    p4est_np1 = p4est_copy(p4est_n, P4EST_FALSE); // very efficient operation, costs almost nothing
    p4est_np1->connectivity = p4est_n->connectivity; // connectivity is not duplicated by p4est_copy, the pointer (i.e. the memory-address) of connectivity seems to be copied from my understanding of the source file of p4est_copy, but I feel this is a bit safer [Raphael Egan]
    p4est_np1->user_pointer = (void*)&criteria;
    Vec vorticity_np1 = NULL;
    Vec norm_grad_u_np1 = NULL;
    unsigned int iter=0;
    p4est_ghost_t *ghost_np1 = NULL;
    my_p4est_hierarchy_t *hierarchy_np1 = NULL;
    my_p4est_node_neighbors_t* ngbd_n_np1 = NULL;
    while(!iterative_grid_update_converged)
    {
      /* ---   FIND THE NEXT ADAPTIVE GRID   --- */
      /* For the very first iteration of the grid-update procedure, p4est_np1 is a
       * simple pure copy of p4est_n, so no node creation nor data interpolation is
       * required. Hence the "if(iter>0)..." statements and the ternary statements
       * (iter > 0 ? : ) */
      // partition the grid if it has changed...
      if(iter > 0)
        my_p4est_partition(p4est_np1, P4EST_FALSE, NULL);
      // ghost_np1, hierarchy_np1 and ngbd_n_np1 are required for the grid update if and only if smoke is defined and refinement with smoke is activated
      if (iter > 0 && smoke != NULL && refine_with_smoke)
        ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
      // get nodes_np1
      // reset nodes_np1 if needed
      if (nodes_np1 != NULL && nodes_np1 != nodes_n){
        p4est_nodes_destroy(nodes_np1); CHKERRXX(ierr); }
      nodes_np1 = ((iter > 0) ? my_p4est_nodes_new(p4est_np1, ghost_np1) : nodes_n);
      // reset phi_np1 if needed
      if (phi_np1 != NULL && phi_np1 != phi){
        ierr = VecDestroy(phi_np1); CHKERRXX(ierr); phi_np1 = NULL; }
      // get phi_np1
      if(iter > 0 || level_set != NULL)
      {
        ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);
        if(level_set != NULL){
          ierr = VecGetArray(phi_np1, &phi_np1_p); CHKERRXX(ierr); }
        for(size_t n=0; n<nodes_np1->indep_nodes.elem_count; ++n)
        {
          double xyz[P4EST_DIM];
          node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);
          if(iter > 0)
            interp_nodes.add_point(n, xyz);
          if(level_set != NULL)
            phi_np1_p[n] = (*level_set)(xyz);
        }
        if(level_set != NULL){
          ierr = VecRestoreArray(phi_np1, &phi_np1_p); phi_np1_p = NULL; CHKERRXX(ierr); }
        else
        {
          interp_inputs.push_back(phi);
          interp_outputs.push_back(phi_np1);
        }
      }
      else
        phi_np1 = phi;
      // reset vorticity_np1 if needed
      if (vorticity_np1 != NULL && vorticity_np1 != vorticity){
        ierr = VecDestroy(vorticity_np1); CHKERRXX(ierr); vorticity_np1 = NULL; }
      if (norm_grad_u_np1 != NULL && norm_grad_u_np1 != norm_grad_u){
        ierr = VecDestroy(norm_grad_u_np1); CHKERRXX(ierr); norm_grad_u_np1 = NULL; }
      // get vorticity_np1
      if(iter > 0)
      {
        ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &vorticity_np1); CHKERRXX(ierr);
        interp_inputs.push_back(vorticity);
        interp_outputs.push_back(vorticity_np1);
        if(norm_grad_u != NULL){
          ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &norm_grad_u_np1); CHKERRXX(ierr);
          interp_inputs.push_back(norm_grad_u);
          interp_outputs.push_back(norm_grad_u_np1);
        }
      }
      else
      {
        vorticity_np1 = vorticity;
        norm_grad_u_np1 = norm_grad_u;
      }
      // reset smoke_np1 if needed
      if (smoke_np1 != NULL && smoke_np1 != smoke){
        ierr = VecDestroy(smoke_np1); CHKERRXX(ierr); smoke_np1 = NULL; }
      // get smoke_np1 (if required)
      Vec vtmp[P4EST_DIM];
      if(smoke != NULL && refine_with_smoke)
      {
        ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &smoke_np1); CHKERRXX(ierr);
        for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
        {
          if(iter > 0)
          {
            ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &vtmp[dir]); CHKERRXX(ierr);
            interp_inputs.push_back(vnp1_nodes[dir]);
            interp_outputs.push_back(vtmp[dir]);
          }
          else
            vtmp[dir] = vnp1_nodes[dir];
        }
      }
      if(iter > 0)
      {
        interp_nodes.set_input(interp_inputs, linear);
        interp_nodes.interpolate(interp_outputs);
        interp_inputs.resize(0); interp_outputs.resize(0);
        interp_nodes.clear();
      }
      if(smoke != NULL && refine_with_smoke)
      {
        P4EST_ASSERT(ghost_np1 != NULL || iter == 0);
        hierarchy_np1 = (iter > 0 ? new my_p4est_hierarchy_t(p4est_np1, ghost_np1, brick) : hierarchy_n);
        ngbd_n_np1    = (iter > 0 ? new my_p4est_node_neighbors_t(hierarchy_np1, nodes_np1) : ngbd_n);
        advect_smoke(ngbd_n_np1, vtmp, smoke_np1);
        for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
          if(vtmp[dir] != NULL && vtmp[dir] != vnp1_nodes[dir]){
            ierr = VecDestroy(vtmp[dir]); CHKERRXX(ierr); }
        if(iter > 0)
        {
          p4est_ghost_destroy(ghost_np1);
          delete  hierarchy_np1;
          delete  ngbd_n_np1;
        }
      }
      iterative_grid_update_converged = !criteria.refine_and_coarsen(p4est_np1, nodes_np1, phi_np1, vorticity_np1, smoke_np1, norm_grad_u_np1);
      grid_is_unchanged = grid_is_unchanged && iterative_grid_update_converged;
      iter++;

      if(iter>((unsigned int) 2+data->max_lvl-data->min_lvl)) // increase the rhs by one to account for the very first step that used to be out of the loop, [Raphael]
      {
        ierr = PetscPrintf(p4est_n->mpicomm, "ooops ... the grid update did not converge\n"); CHKERRXX(ierr);
        break;
      }
    }
    if(vorticity_np1 != NULL && vorticity_np1 != vorticity){
      ierr = VecDestroy(vorticity_np1); CHKERRXX(ierr); } // destroy it if created in the iterative process
    if(norm_grad_u_np1 != NULL && norm_grad_u_np1 != norm_grad_u){
      ierr = VecDestroy(norm_grad_u_np1); CHKERRXX(ierr); } // destroy it if created in the iterative process
    // done refining using the specific grid-refinement criterion, reset the original data as user-pointer
    p4est_np1->user_pointer = data;
  }

  int smoke_np1_is_still_valid = (smoke_np1 != NULL && iterative_grid_update_converged ? 1 : 0); // collectively the same at this stage: the vectors are still ok if the iterative procedure exited properly by grid_is_changing turning false;
  if(!grid_is_unchanged)
  {
    /* Get the final forest at time np1: balance (2:1 neighbor cell sizes) the results of the iterative grid update
     * and repartition it.
     * If grid_is_unchanged is true, there is no need to do that, since the np1 forest is strictly the same
     * as the forest at time n in such a case (which is supposedly already balanced and partitioned)! */
    p4est_gloidx_t num_global_quadrants_np1_before_balance_and_partition = p4est_np1->global_num_quadrants;
    p4est_balance(p4est_np1, P4EST_CONNECT_FULL, NULL);
    my_p4est_partition(p4est_np1, P4EST_FALSE, NULL);
    if(smoke_np1_is_still_valid == 1)
    {
      smoke_np1_is_still_valid = ((p4est_np1->global_num_quadrants == num_global_quadrants_np1_before_balance_and_partition)? 1: 0);
      int mpiret = MPI_Allreduce(MPI_IN_PLACE, &smoke_np1_is_still_valid, 1, MPI_INT, MPI_LAND, p4est_np1->mpicomm); SC_CHECK_MPI(mpiret);
      // the smoke_np1 vector is still ok if the balance step did not create new quadrants: we might need to rescatter it but it is still ok, no need to recalculate it...
    }
  }
  else
  {
    P4EST_ASSERT(p4est_is_equal(p4est_n, p4est_np1, P4EST_FALSE));
    if(p4est_np1 != p4est_n)
      p4est_destroy(p4est_np1);// no need to keep two identical forests in memory if they are exactly the same, just make p4est_nm1 and p4est_n point to the same one eventually and handle memory correspondingly...
    p4est_np1     = p4est_n;
  }

  /* Get the ghost cells at time np1, */
  p4est_ghost_t *ghost_np1 = NULL;
  if(!grid_is_unchanged)
  {
    ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
    my_p4est_ghost_expand(p4est_np1, ghost_np1);
    if(third_degree_ghost_are_required(convert_to_xyz))
      my_p4est_ghost_expand(p4est_np1, ghost_np1);
  }
  else
    ghost_np1 = ghost_n;
  if(!grid_is_unchanged)
  {
    p4est_nodes_destroy(nodes_np1);
    nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);
  }
  else
  {
    P4EST_ASSERT(nodes_np1 == NULL || nodes_np1 == nodes_n);
    nodes_np1 = nodes_n;
  }
  my_p4est_hierarchy_t *hierarchy_np1 = NULL;
  if(!grid_is_unchanged)
    hierarchy_np1 = new my_p4est_hierarchy_t(p4est_np1, ghost_np1, brick);
  else
    hierarchy_np1 = hierarchy_n;
  my_p4est_node_neighbors_t *ngbd_np1 = NULL;
  if(!grid_is_unchanged)
    ngbd_np1 = new my_p4est_node_neighbors_t(hierarchy_np1, nodes_np1);
  else
    ngbd_np1 = ngbd_n;
  ngbd_np1->init_neighbors();

  my_p4est_cell_neighbors_t *ngbd_c_np1 = NULL;
  if(!grid_is_unchanged)
    ngbd_c_np1 = new my_p4est_cell_neighbors_t(hierarchy_np1);
  else
    ngbd_c_np1 = ngbd_c;
  my_p4est_faces_t *faces_np1 = NULL;
  if(!grid_is_unchanged)
    faces_np1 = new my_p4est_faces_t(p4est_np1, ghost_np1, brick, ngbd_c_np1);
  else
    faces_np1 = faces_n;

  /* slide relevant fiels and grids in time: nm1 data are disregarded, n data becomes nm1 data and np1 data become n data...
   * In particular, if grid_is_unchanged is false, the np1 grid is different than grid at time n, we need to
   * re-construct its faces and cell-neighbors and the solvers we have used will need to be destroyed... */

  // 1) scalar field phi, possible scenarii:
  //    i)  if the grid is unchanged and phi_np1 is not defined yet (iterative grid update forcibly skipped) --> sample the given valid levelset if it is given, and/or simply set phi_np1 = phi
  //    ii) if the grid is changed, the node interpolator input buffer needs to be reset anyways (for future interpolation) and phi_np1 is recreated as a side result (if previously evaluated
  //        from a valid given levelset, this is a small overhead; if previously obtained from linear interpolation it is no longer valid anyways since non-oscillatory quadratic interpolation
  //        is desired eventually)
  // Reinitialization is performed only if desired and if either the grid has changed or if a valid given levelset function has been given!
  bool phi_may_have_changed = true;
  if(grid_is_unchanged)
  {
    P4EST_ASSERT(ghosts_are_equal(ghost_n, ghost_np1));
    P4EST_ASSERT(nodes_are_equal(p4est_n->mpisize, nodes_n, nodes_np1));
    P4EST_ASSERT(phi != NULL);
    P4EST_ASSERT((phi_np1 == NULL && keep_grid_as_such) || (!keep_grid_as_such && ((phi_np1 == phi && level_set == NULL) || (phi_np1 != phi && phi_np1 != NULL && level_set != NULL))));
    if (phi_np1 == NULL) // in this case, the grid is forcibly kept as such
    {
      if(level_set != NULL) // if keep_grid_as_such is true but a valid CF_ levelset is given, sample it on phi (which exists and is of appropriate size)...
        sample_cf_on_nodes(p4est_np1, nodes_np1, *level_set, phi);
      else
        phi_may_have_changed = false; // in this case (ONLY), no change at all in phi, for sure
      // set phi_np1 == phi, to enable reinitialization thereafter if desired
      phi_np1 = phi;
    }
  }
  else
  {
    P4EST_ASSERT(phi_np1 != NULL && phi_np1 != phi);
    ierr = VecDestroy(phi_np1); CHKERRXX(ierr); // even if it existed, it is no longer valid, since non-oscillatory interpolation is desired on the final np1 grid
    // build the new phi, either by direct evaluation if levelset is given
    // or by non-oscillatory interpolation from current state if not.
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);
    if(level_set != NULL){
      ierr = VecGetArray(phi_np1, &phi_np1_p); CHKERRXX(ierr); }
    for(size_t n=0; n<nodes_np1->indep_nodes.elem_count; ++n)
    {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);
      interp_nodes.add_point(n, xyz);
      if(level_set != NULL)
        phi_np1_p[n] = (*level_set)(xyz);
    }
    if(level_set != NULL){
      ierr = VecRestoreArray(phi_np1, &phi_np1_p); CHKERRXX(ierr); }
    else
    {
      interp_nodes.set_input(phi, quadratic_non_oscillatory);
      interp_nodes.interpolate(phi_np1);
    }
  }

  if(do_reinitialization && phi_may_have_changed) // we don't reinitialize if phi is unchanged
  {
    my_p4est_level_set_t lsn(ngbd_np1);
    lsn.reinitialize_1st_order_time_2nd_order_space(phi_np1);
    lsn.perturb_level_set_function(phi_np1, EPS);
  }

  // we also recalculate the gradient of phi if needed
  Vec grad_phi_np1 = grad_phi; // if phi is unchanged, so is grad_phi
  if(phi_may_have_changed)
  {
    // if not, build and calculate the new one:
    ierr = VecCreateGhostNodesBlock(p4est_np1, nodes_np1, P4EST_DIM, &grad_phi_np1); CHKERRXX(ierr);
    ngbd_np1->first_derivatives_central(phi_np1, grad_phi_np1);
  }
  my_p4est_interpolation_nodes_t* interp_phi_np1      = (ngbd_n != ngbd_np1 ? new my_p4est_interpolation_nodes_t(ngbd_np1) : interp_phi);
  my_p4est_interpolation_nodes_t* interp_grad_phi_np1 = (ngbd_n != ngbd_np1 ? new my_p4est_interpolation_nodes_t(ngbd_np1) : interp_grad_phi);
  interp_phi_np1->set_input(phi_np1, linear);
  interp_grad_phi_np1->set_input(grad_phi_np1, linear, P4EST_DIM);

  // 2) scalar field vorticity: reset it to the appropriate size if the grid has changed, nothing to do otherwise
  if(!grid_is_unchanged){
    ierr = VecDestroy(vorticity); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &vorticity); CHKERRXX(ierr);
    if(norm_grad_u != NULL)
    {
      ierr = VecDestroy(norm_grad_u); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &norm_grad_u); CHKERRXX(ierr);
    }
  }
  // 3) velocity fields at the nodes (and their second derivatives)
  for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
  {
    ierr = VecDestroy(vnm1_nodes[dir]); CHKERRXX(ierr);
    vnm1_nodes[dir] = vn_nodes[dir]; // At this point, both vnm1_nodes and vn_nodes point to the same object
    if(!grid_is_unchanged){
      ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &vn_nodes[dir]); CHKERRXX(ierr); } // At this point, we now create a new object, which vn_nodes points to now.
    else
      vn_nodes[dir]   = vnp1_nodes[dir];
    for (unsigned char dd = 0; dd < P4EST_DIM; ++dd) {
      ierr = VecDestroy(second_derivatives_vnm1_nodes[dd][dir]); CHKERRXX(ierr);
      second_derivatives_vnm1_nodes[dd][dir] = second_derivatives_vn_nodes[dd][dir];
      ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &second_derivatives_vn_nodes[dd][dir]); CHKERRXX(ierr);
    }
  }
  if(!grid_is_unchanged)
  {
    interp_nodes.set_input(vnp1_nodes, interp_v_update, P4EST_DIM);
    interp_nodes.interpolate(vn_nodes); CHKERRXX(ierr);
  }
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    if(!grid_is_unchanged){
      ierr = VecDestroy(vnp1_nodes[dir]); CHKERRXX(ierr); }
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &vnp1_nodes[dir]); CHKERRXX(ierr);
  }
  ngbd_np1->second_derivatives_central(vn_nodes, DIM(second_derivatives_vn_nodes[0], second_derivatives_vn_nodes[1], second_derivatives_vn_nodes[2]), P4EST_DIM);
  if(!grid_is_unchanged)
    interp_nodes.clear();

  // [Raphael]: the following was already commented in the original version. If uncommented, I think it should be called here...
  /* set velocity inside solid to bc_v */
  //  extrapolate_bc_v(ngbd_np1, vn_nodes, phi_np1);

  // 4) scalar field smoke (if needed)
  if(smoke != NULL)
  {
    // if smoke_np1 already exists but is no longer useful, destroy it...
    if(smoke_np1 != NULL && !smoke_np1_is_still_valid){
      ierr = VecDestroy(smoke_np1); CHKERRXX(ierr); smoke_np1 = NULL; }
    if(smoke_np1 == NULL) // calculate it if not already done or if needs to be redone...
    {
      ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &smoke_np1); CHKERRXX(ierr);
      advect_smoke(ngbd_np1, vn_nodes, smoke_np1); // lighter call if the neighborhood is already known...
      ierr = VecDestroy(smoke); CHKERRXX(ierr); // you can now destroy this one (after it has been advected)
      smoke = smoke_np1;
    }
    else // smoke_np1 already exists and is valid
    {
      ierr = VecDestroy(smoke); CHKERRXX(ierr);
      if(grid_is_unchanged) // it is good as such if grid is unchanged (since nodes_np1 == nodes_n)
        smoke = smoke_np1;
      else // just needs to be re-scattered in this case, do not recalculate all the advection equations...
      {
        ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &smoke); CHKERRXX(ierr);
        VecScatter ctx;
        ierr = VecScatterCreateChangeLayout(p4est_np1->mpicomm, smoke_np1, smoke, &ctx); CHKERRXX(ierr);
        ierr = VecGhostChangeLayoutBegin(ctx, smoke_np1, smoke); CHKERRXX(ierr);
        ierr = VecGhostChangeLayoutEnd(ctx, smoke_np1, smoke); CHKERRXX(ierr);
        ierr = VecScatterDestroy(ctx); CHKERRXX(ierr);
        ierr = VecDestroy(smoke_np1); CHKERRXX(ierr); smoke_np1 = NULL;
      }
    }
  }

  // 5) cell-centered hodge variable, face-centered dxyz_hodge and face-centered face_is_well_defined vectors
  // if the grid is changed, the first two variables are interpolated (cells to cells and faces to faces) and the
  // face_is_well_defined vectors are recalculated and reset.
  // If the grid is unchanged, hodge and dxyz_hodge are good as such, no need to recalculate them. However, the
  // face_is_well_defined vectors need to be recalculated if the levelset has been reset...
  if(!grid_is_unchanged) // if the grid has changed, we need to re-interpolate cell and face data to new cells and new faces
  {
    /* interpolate the Hodge variable on the new forest (for good initial guess for next projection step)
     * build a new cell-centered pressure vector for calculating the next pressure field */
    my_p4est_interpolation_cells_t interp_cell(ngbd_c, ngbd_n);
    for (p4est_topidx_t tree_idx = p4est_np1->first_local_tree; tree_idx <= p4est_np1->last_local_tree; ++tree_idx) {
      p4est_tree_t *tree = p4est_tree_array_index(p4est_np1->trees, tree_idx);
      for (size_t q = 0; q < tree->quadrants.elem_count; ++q) {
        p4est_locidx_t quad_idx = tree->quadrants_offset + q;
        double xyz_c[P4EST_DIM];
        quad_xyz_fr_q(quad_idx, tree_idx, p4est_np1, ghost_np1, xyz_c);
        interp_cell.add_point(quad_idx, xyz_c);
      }
    }
    interp_cell.set_input(hodge, phi, &bc_hodge);
    Vec hodge_tmp;
    ierr = VecCreateGhostCells(p4est_np1, ghost_np1, &hodge_tmp); CHKERRXX(ierr);
    interp_cell.interpolate(hodge_tmp);
    interp_cell.clear();
    ierr = VecDestroy(hodge); CHKERRXX(ierr);
    hodge = hodge_tmp;
    ierr = VecGhostUpdateBegin(hodge, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    // create a new pressure vector...
    ierr = VecDestroy(pressure); CHKERRXX(ierr);
    ierr = VecCreateGhostCells(p4est_np1, ghost_np1, &pressure); CHKERRXX(ierr);

    /* interpolate the gradient of the Hodge variable on the new forest
     * The (Dirichlet) velocity boundary conditions depend on the components of the gradient of Hodge, so they are
     * required to condition the solver for the next step...
     * [Raphael's note:] it might be better and more efficient to recalculate them from the interpolated Hodge here
     * above for better consistent conditioning of the iterative solver (since this is how the solver itself defines
     * its components within a solve step) --> ask Frederic's opinion! */
    my_p4est_interpolation_faces_t interp_faces(ngbd_n, faces_n);

    for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    {
      Vec dxyz_hodge_tmp;
      ierr = VecCreateGhostFaces(p4est_np1, faces_np1, &dxyz_hodge_tmp, dir); CHKERRXX(ierr);
      for(p4est_locidx_t f_idx=0; f_idx<faces_np1->num_local[dir]; ++f_idx)
      {
        double xyz[P4EST_DIM];
        faces_np1->xyz_fr_f(f_idx, dir, xyz);
        interp_faces.add_point(f_idx, xyz);
      }
      interp_faces.set_input(dxyz_hodge[dir], dir, 1, face_is_well_defined[dir]);
      interp_faces.interpolate(dxyz_hodge_tmp);
      interp_faces.clear();

      ierr = VecDestroy(dxyz_hodge[dir]); CHKERRXX(ierr);
      dxyz_hodge[dir] = dxyz_hodge_tmp;

      ierr = VecGhostUpdateBegin(dxyz_hodge[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      ierr = VecDestroy(face_is_well_defined[dir]); CHKERRXX(ierr);
      ierr = VecCreateGhostFaces(p4est_np1, faces_np1, &face_is_well_defined[dir], dir); CHKERRXX(ierr);
      check_if_faces_are_well_defined(faces_np1, dir, *interp_phi_np1, bc_v[dir], face_is_well_defined[dir]);

      ierr = VecDestroy(vstar[dir]); CHKERRXX(ierr);
      ierr = VecCreateGhostFaces(p4est_np1, faces_np1, &vstar[dir], dir); CHKERRXX(ierr);

      ierr = VecDestroy(vnp1[dir]); CHKERRXX(ierr);
      ierr = VecCreateGhostFaces(p4est_np1, faces_np1, &vnp1[dir], dir); CHKERRXX(ierr);
    }
    // finish communicating ghost values for hodge and dxyz_hodge
    ierr = VecGhostUpdateEnd(hodge, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      ierr = VecGhostUpdateEnd(dxyz_hodge[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }
  }
  else if(level_set != NULL)
  {
    // the grid is unchanged but the levelset might have changed, hence possibly affecting the face_is_well_defined vectors --> the solvers cannot be reused safely
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      check_if_faces_are_well_defined(faces_np1, dir, *interp_phi_np1, bc_v[dir], face_is_well_defined[dir]);
  }

  /* update the variables */
  if(p4est_nm1 != p4est_n)
    p4est_destroy(p4est_nm1);
  p4est_nm1 = p4est_n; p4est_n = p4est_np1;
  if(ghost_nm1 != ghost_n)
    p4est_ghost_destroy(ghost_nm1);
  ghost_nm1 = ghost_n; ghost_n = ghost_np1;
  if(nodes_nm1 != nodes_n)
    p4est_nodes_destroy(nodes_nm1);
  nodes_nm1 = nodes_n; nodes_n = nodes_np1;
  if(hierarchy_nm1 != hierarchy_n)
    delete hierarchy_nm1;
  hierarchy_nm1 = hierarchy_n; hierarchy_n = hierarchy_np1;
  if(ngbd_nm1 != ngbd_n)
    delete ngbd_nm1;
  ngbd_nm1 = ngbd_n; ngbd_n = ngbd_np1;
  if(ngbd_c != ngbd_c_np1)
    delete ngbd_c;
  ngbd_c = ngbd_c_np1;
  if(faces_n != faces_np1)
    delete faces_n;
  faces_n = faces_np1;
  // we can finally slide phi, grad phi and reset the interpolators...
  if(phi != phi_np1){
    ierr = VecDestroy(phi); CHKERRXX(ierr); }
  phi = phi_np1;
  if(grad_phi != grad_phi_np1){
    ierr = VecDestroy(grad_phi); CHKERRXX(ierr); }
  grad_phi = grad_phi_np1;
  // reset the phi-interpolator tools
  if(interp_phi != interp_phi_np1)
    delete interp_phi;
  interp_phi = interp_phi_np1;
  if(interp_grad_phi != interp_grad_phi_np1)
    delete interp_grad_phi;
  interp_grad_phi = interp_grad_phi_np1;


  semi_lagrangian_backtrace_is_done         = false;
  interpolators_from_face_to_nodes_are_set  = interpolators_from_face_to_nodes_are_set && grid_is_unchanged;
  ierr = PetscLogEventEnd(log_my_p4est_navier_stokes_update, 0, 0, 0, 0); CHKERRXX(ierr);
  if(ns_time_step_analyzer.is_on())
    ns_time_step_analyzer.stop();

  return (grid_is_unchanged && level_set == NULL);
}

// ELYCE TRYING SOMETHING:
void my_p4est_navier_stokes_t::update_from_tn_to_tnp1_grid_external(Vec phi_np1, Vec phi_n,
                                                                    Vec v_n_nodes_[P4EST_DIM], Vec v_nm1_nodes_[P4EST_DIM],
                                                                    p4est_t* p4est_np1, p4est_nodes_t* nodes_np1, p4est_ghost_t* ghost_np1,
                                                                    my_p4est_node_neighbors_t* ngbd_np1,
                                                                    my_p4est_faces_t* &faces_np1, my_p4est_cell_neighbors_t* &ngbd_c_np1, my_p4est_hierarchy_t* hierarchy_np1)
{
  PetscErrorCode ierr;

  ierr = PetscLogEventBegin(log_my_p4est_navier_stokes_update, 0, 0, 0, 0); CHKERRXX(ierr);


  printf("\n Addresses of vns vectors (NS, before grid update): \n "
         "v_n (NS), v_nm1 (NS) = %p, %p \n", vn_nodes[0], vnm1_nodes[0]);

  // Create new faces and ngbd_c for NS to use based on new grid provided: (old ones will be deleted when we slide the fields)
  // Get the new cell neighbors:
  ngbd_c_np1 = new my_p4est_cell_neighbors_t(hierarchy_np1);

  // Create new the faces:
  faces_np1 = new my_p4est_faces_t(p4est_np1,ghost_np1,ngbd_c_np1);

  //------------------------------------------------------
  // (0) Interpolate the cell-centered hodge variable onto the new grid (cells to cells)
  //------------------------------------------------------

  /*NOTE: this interpolation from old cells to new cells requires the *old* level set function phi,
   * and thus must be executed BEFORE we update the LSF phi with the new provided LSF phi_np1 */
  /* interpolate the Hodge variable on the new forest (for good initial guess for next projection step)
   * build a new cell-centered pressure vector for calculating the next pressure field */


  my_p4est_interpolation_cells_t interp_cell(ngbd_c, ngbd_n);
  for (p4est_topidx_t tree_idx = p4est_np1->first_local_tree; tree_idx <= p4est_np1->last_local_tree; ++tree_idx) {
    p4est_tree_t *tree = p4est_tree_array_index(p4est_np1->trees, tree_idx);
    for (size_t q = 0; q < tree->quadrants.elem_count; ++q) {
      p4est_locidx_t quad_idx = tree->quadrants_offset + q;
      double xyz_c[P4EST_DIM];
      quad_xyz_fr_q(quad_idx, tree_idx, p4est_np1, ghost_np1, xyz_c);
      interp_cell.add_point(quad_idx, xyz_c);
    }
  }

  interp_cell.set_input(hodge, phi_n, &bc_hodge);
  Vec hodge_tmp;
  ierr = VecCreateGhostCells(p4est_np1, ghost_np1, &hodge_tmp); CHKERRXX(ierr);
  interp_cell.interpolate(hodge_tmp);
  interp_cell.clear();

  ierr = VecDestroy(hodge); CHKERRXX(ierr);
  hodge = hodge_tmp;
  ierr = VecGhostUpdateBegin(hodge, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  // Now that the hodge variable has been updated using the old LSF,
  // we may proceed with sliding the rest of the fields

  //------------------------------------------------------
  // (1) Set phi as the new phi on new grid
  //------------------------------------------------------

  phi = phi_np1;

  delete interp_phi;
  interp_phi = new my_p4est_interpolation_nodes_t(ngbd_np1);
  interp_phi->set_input(phi_np1,linear);

  // reset grad phi
  if(grad_phi != NULL){
    ierr = VecDestroy(grad_phi); CHKERRXX(ierr);
  }
  ierr = VecCreateGhostNodesBlock(p4est_np1, nodes_np1, P4EST_DIM, &grad_phi);
  ngbd_np1->first_derivatives_central(phi_np1, grad_phi);
  delete interp_grad_phi;
  interp_grad_phi = new my_p4est_interpolation_nodes_t(ngbd_np1);
  interp_grad_phi->set_input(grad_phi, linear, P4EST_DIM);

  //------------------------------------------------------
  // (2) Reset the scalar vorticity field onto the new grid provided
  //------------------------------------------------------
  // Note -- vorticity will be computed by user as desired after solution is obtained
  if(vorticity!=NULL) ierr = VecDestroy(vorticity); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &vorticity); CHKERRXX(ierr);
  if(norm_grad_u != NULL)
  {
    ierr = VecDestroy(norm_grad_u); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &norm_grad_u); CHKERRXX(ierr);
  }

  //------------------------------------------------------
  // (3) Slide velocity fields at nodes (and their second derivatives)
  //------------------------------------------------------
  // NOTE: We want to slide velocity fields inside the NS solver such that:
  // And the user provides the vnm1 and vn
  // vnm1_nodes[d] = vn_nodes[d]
  // vnp_nodes[d] = vnp1_nodes[d]
  // vnp1_nodes[d] --> we will create a new one here on the new grid provided by the user to hold the forthcoming new soln at vnp1

  foreach_dimension(d){
    // Slide the velocities:
    vnm1_nodes[d] = v_nm1_nodes_[d];
    vn_nodes[d] = v_n_nodes_[d];

    // Slide 2nd derivatives and create new vecs for the second derivatives to come
    foreach_dimension(dd){
      ierr = VecDestroy(second_derivatives_vnm1_nodes[d][dd]); CHKERRXX(ierr);
      second_derivatives_vnm1_nodes[d][dd] = second_derivatives_vn_nodes[d][dd];
      ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &second_derivatives_vn_nodes[d][dd]); CHKERRXX(ierr);
    }
  }

  // Create vector to hold the new vnp1 at the nodes (for next time!):
  // Note: this is safe to do bc outside of this fxn, the vn_nodes has been slid to vnm1_nodes and vnp1_nodes to vn_nodes, so now we need to create a new vector for the new vnp1 forthcoming
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &vnp1_nodes[dir]); CHKERRXX(ierr);
  }


  // (3.6) Compute the new second derivatives of vn on the new grid
  ngbd_np1->second_derivatives_central(vn_nodes, DIM(second_derivatives_vn_nodes[0], second_derivatives_vn_nodes[1], second_derivatives_vn_nodes[2]), P4EST_DIM);


  // [Raphael]: the following was already commented in the original version. If uncommented, I think it should be called here...
  /* set velocity inside solid to bc_v */
  //  extrapolate_bc_v(ngbd_np1, vn_nodes, phi_np1);
  //------------------------------------------------------
  // (4) Smoke should be handled here -- [Elyce: Ignoring this for now, will come back to it later]
  //------------------------------------------------------


  //------------------------------------------------------
  // (5) face-centered dxyz_hodge and face-centered face_is_well_defined vectors
  //------------------------------------------------------
  // the first variable is interpolated (faces to faces) and the
  // face_is_well_defined vectors are recalculated and reset.

  // create a new pressure vector...
  ierr = VecDestroy(pressure); CHKERRXX(ierr);
  ierr = VecCreateGhostCells(p4est_np1, ghost_np1, &pressure); CHKERRXX(ierr);

  /* interpolate the gradient of the Hodge variable on the new forest
   * The (Dirichlet) velocity boundary conditions depend on the components of the gradient of Hodge, so they are
   * required to condition the solver for the next step...
   * [Raphael's note:] it might be better and more efficient to recalculate them from the interpolated Hodge here
   * above for better consistent conditioning of the iterative solver (since this is how the solver itself defines
   * its components within a solve step) --> ask Frederic's opinion! */

  my_p4est_interpolation_faces_t interp_faces(ngbd_n, faces_n);

  for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
  {
    // (5.1) Create temporary dxyz_hodge at faces for np1 grid
    Vec dxyz_hodge_tmp;
    ierr = VecCreateGhostFaces(p4est_np1, faces_np1, &dxyz_hodge_tmp, dir); CHKERRXX(ierr);

    // (5.2) Set up face interpolator to interp dxyz_hodge values
    for(p4est_locidx_t f_idx=0; f_idx<faces_np1->num_local[dir]; ++f_idx)
    {
      double xyz[P4EST_DIM];
      faces_np1->xyz_fr_f(f_idx, dir, xyz);
      interp_faces.add_point(f_idx, xyz);
    }
    // (5.3) Perform the interpolation
    interp_faces.set_input(dxyz_hodge[dir], dir, 1, face_is_well_defined[dir]);

    interp_faces.interpolate(dxyz_hodge_tmp);

    interp_faces.clear();

    // (5.4) Destroy old dxyz_hodge and update variable with the new values
    ierr = VecDestroy(dxyz_hodge[dir]); CHKERRXX(ierr);
    dxyz_hodge[dir] = dxyz_hodge_tmp;

    ierr = VecGhostUpdateBegin(dxyz_hodge[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    // (5.5) Destroy face_is_well_defined at previous grid and check at the new grid
    ierr = VecDestroy(face_is_well_defined[dir]); CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est_np1, faces_np1, &face_is_well_defined[dir], dir); CHKERRXX(ierr);
    check_if_faces_are_well_defined(faces_np1, dir, *interp_phi, bc_v[dir], face_is_well_defined[dir]);

    //------------------------------------------------------
    // (6) Create new vectors for vstar (intermediate velocity) and vnp1 on new grid for next solution step
    //------------------------------------------------------
    ierr = VecDestroy(vstar[dir]); CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est_np1, faces_np1, &vstar[dir], dir); CHKERRXX(ierr);

    ierr = VecDestroy(vnp1[dir]); CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est_np1, faces_np1, &vnp1[dir], dir); CHKERRXX(ierr);
  }
  //------------------------------------------------------
  // finish communicating ghost values for hodge and dxyz_hodge
  //------------------------------------------------------
  ierr = VecGhostUpdateEnd(hodge, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecGhostUpdateEnd  (dxyz_hodge[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  }

  printf("\n Addresses of vns vectors (NS, after grid update): \n "
         "v_n (provided), v_nm1 (provided) = %p, %p \n"
         "v_n (NS), v_nm1 (NS) = %p, %p \n",
         v_n_nodes_[0], v_nm1_nodes_[0], vn_nodes[0], vnm1_nodes[0]);


  //------------------------------------------------------
  //(7) Slide the grid objects
  //------------------------------------------------------
  // Note: destruction of p4est_nm1 is handled externally
  p4est_nm1 = p4est_n; p4est_n = p4est_np1;

  ghost_nm1 = ghost_n; ghost_n = ghost_np1;

  nodes_nm1 = nodes_n; nodes_n = nodes_np1;

  hierarchy_nm1 = hierarchy_n; hierarchy_n = hierarchy_np1;

  ngbd_nm1 = ngbd_n; ngbd_n = ngbd_np1;

  delete ngbd_c;
  ngbd_c = ngbd_c_np1;

  delete faces_n;
  faces_n = faces_np1;

  semi_lagrangian_backtrace_is_done         = false;
  ierr = PetscLogEventEnd(log_my_p4est_navier_stokes_update, 0, 0, 0, 0); CHKERRXX(ierr);
}

// DONE ELYCE TRYING SOMETHING.
void my_p4est_navier_stokes_t::compute_pressure()
{
  PetscErrorCode ierr;

  const double *hodge_p;
  ierr = VecGetArrayRead(hodge, &hodge_p); CHKERRXX(ierr);

  double *pressure_p;
  ierr = VecGetArray(pressure, &pressure_p); CHKERRXX(ierr);

  for (size_t i = 0; i < hierarchy_n->get_layer_size(); ++i) {
    p4est_locidx_t quad_idx = hierarchy_n->get_local_index_of_layer_quadrant(i);
    p4est_topidx_t tree_idx = hierarchy_n->get_tree_index_of_layer_quadrant(i);
    pressure_p[quad_idx] = alpha()*rho/dt_n*hodge_p[quad_idx] - mu*compute_divergence(quad_idx, tree_idx);
  }
  ierr = VecGhostUpdateBegin(pressure, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for (size_t i = 0; i < hierarchy_n->get_inner_size(); ++i) {
    p4est_locidx_t quad_idx = hierarchy_n->get_local_index_of_inner_quadrant(i);
    p4est_topidx_t tree_idx = hierarchy_n->get_tree_index_of_inner_quadrant(i);
    pressure_p[quad_idx] = alpha()*rho/dt_n*hodge_p[quad_idx] - mu*compute_divergence(quad_idx, tree_idx);
  }
  ierr = VecGhostUpdateEnd (pressure, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecRestoreArray(pressure, &pressure_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(hodge, &hodge_p); CHKERRXX(ierr);

  if(bc_pressure->interfaceType() != NOINTERFACE)
  {
    my_p4est_level_set_cells_t lsc(ngbd_c, ngbd_n);
    lsc.geometric_extrapolation_over_interface(pressure, phi, *interp_grad_phi, *bc_pressure, 2, 3);
  }
}

void my_p4est_navier_stokes_t::compute_pressure_at_nodes(Vec *pressure_nodes){
  PetscErrorCode ierr;

  double *pressure_nodes_p;
  ierr = VecGetArray(*pressure_nodes, &pressure_nodes_p); CHKERRXX(ierr);

  for (size_t k = 0; k < ngbd_n->get_layer_size(); ++k) {
    p4est_locidx_t node_idx = ngbd_n->get_layer_node(k);
    pressure_nodes_p[node_idx] = interpolate_cell_field_at_node(node_idx, ngbd_c, ngbd_n, pressure, bc_pressure, phi);
  }
  ierr = VecGhostUpdateBegin(*pressure_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for (size_t k = 0; k < ngbd_n->get_local_size(); ++k) {
    p4est_locidx_t node_idx = ngbd_n->get_local_node(k);
    pressure_nodes_p[node_idx] = interpolate_cell_field_at_node(node_idx, ngbd_c, ngbd_n, pressure, bc_pressure, phi);
  }
  ierr = VecGhostUpdateEnd(*pressure_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecRestoreArray(*pressure_nodes, &pressure_nodes_p); CHKERRXX(ierr);


}
void my_p4est_navier_stokes_t::calculate_viscous_stress_at_local_nodes(const p4est_locidx_t& node_idx, const double* phi_read_p, const double *grad_phi_read_p,
                                                                       const double* vnodes_read_p[P4EST_DIM], double* viscous_stress_p[P4EST_DIM]) const
{
  for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    viscous_stress_p[dir][node_idx] = 0.0; // initialize value (note: also 0.0 if the interpolated normal is not well-behaved)
  if(fabs(phi_read_p[node_idx]) < 10.0*MAX(DIM(dxyz_min[0], dxyz_min[1], dxyz_min[2]))) // we calculate it only close to the interface, we don't need it elsewhere
  {
    const quad_neighbor_nodes_of_node_t* qnnn;
    ngbd_n->get_neighbors(node_idx, qnnn);
    double grad_v[P4EST_DIM][P4EST_DIM]; // grad_v[dir][comp] : partial derivative of dir^th velocity component along cartesian direction comp
    const double* in[P4EST_DIM] = {DIM(vnodes_read_p[0], vnodes_read_p[1], vnodes_read_p[2])};
    double* out[P4EST_DIM]      = {DIM(grad_v[0], grad_v[1], grad_v[2])};
    qnnn->gradient(in, out, P4EST_DIM);

    const double *local_grad_phi = (grad_phi_read_p + P4EST_DIM*node_idx); // so local_grad_phi[k] = grad_phi_read_p[P4EST_DIM*node_idx + k]
    const double mag_grad_phi = sqrt(SUMD(SQR(local_grad_phi[0]), SQR(local_grad_phi[1]), SQR(local_grad_phi[2])));
    // calculate mu*(strain-rate tensor) dot negative_normal
    if(mag_grad_phi > EPS)
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
        for (unsigned char comp = 0; comp < P4EST_DIM; ++comp)
          viscous_stress_p[dir][node_idx] -= mu*(grad_v[dir][comp] + grad_v[comp][dir])*local_grad_phi[comp]/mag_grad_phi; // "-=" because the solid's normal is -grad(phi)/norm(grad_phi)

  }
  return;
}

void my_p4est_navier_stokes_t::compute_forces(double *f)
{
  PetscErrorCode ierr;

  const double *phi_read_p, *grad_phi_read_p;
  ierr = VecGetArrayRead(phi, &phi_read_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(grad_phi, &grad_phi_read_p); CHKERRXX(ierr);

  Vec viscous_stress[P4EST_DIM]; double *viscous_stress_p[P4EST_DIM];
  const double *vnodes_read_p[P4EST_DIM];

  for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
  {
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &viscous_stress[dir]); CHKERRXX(ierr);
    ierr = VecGetArrayRead(vnp1_nodes[dir], &vnodes_read_p[dir]); CHKERRXX(ierr);
    ierr = VecGetArray(viscous_stress[dir], &viscous_stress_p[dir]); CHKERRXX(ierr);
  }

  for(size_t i = 0; i < ngbd_n->get_layer_size(); ++i)
    calculate_viscous_stress_at_local_nodes(ngbd_n->get_layer_node(i), phi_read_p, grad_phi_read_p, vnodes_read_p, viscous_stress_p);

  for(unsigned char dir = 0; dir < P4EST_DIM; ++dir){
    ierr = VecGhostUpdateBegin(viscous_stress[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  for(size_t i = 0; i < ngbd_n->get_local_size(); ++i)
    calculate_viscous_stress_at_local_nodes(ngbd_n->get_local_node(i), phi_read_p, grad_phi_read_p, vnodes_read_p, viscous_stress_p);

  ierr = VecRestoreArrayRead(phi, &phi_read_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(phi, &grad_phi_read_p); CHKERRXX(ierr);
  for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArrayRead(vnp1_nodes[dir], &vnodes_read_p[dir]); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(viscous_stress[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecRestoreArray(viscous_stress[dir], &viscous_stress_p[dir]); CHKERRXX(ierr);
    f[dir] = integrate_over_interface(p4est_n, nodes_n, phi, viscous_stress[dir]);
    ierr = VecDestroy(viscous_stress[dir]); CHKERRXX(ierr);
  }

  my_p4est_level_set_cells_t lsc(ngbd_c, ngbd_n);
  double integral_of_p_normal[P4EST_DIM]; // which is equal to "integral of negative p times solid's normal"
  lsc.normal_vector_weighted_integral_over_interface(phi, pressure, integral_of_p_normal, grad_phi);

  for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    f[dir] += integral_of_p_normal[dir];

  return;
}

void my_p4est_navier_stokes_t::save_vtk(const char* name, bool with_Q_and_lambda_2_value, const double U_scaling_for_Q_and_lambda_2, const double x_scaling_for_Q_and_lambda_2)
{
  PetscErrorCode ierr;

  const double *phi_p;
  const double *vn_p[P4EST_DIM];
  const double *hodge_p;

  Vec Q_value_nodes               = NULL;
  Vec lambda_2_nodes              = NULL;
  const double *Q_value_nodes_p   = NULL;
  const double *lambda_2_nodes_p  = NULL;
  if(with_Q_and_lambda_2_value)
  {
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &Q_value_nodes); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &lambda_2_nodes); CHKERRXX(ierr);
    compute_Q_and_lambda_2_value(Q_value_nodes, lambda_2_nodes, U_scaling_for_Q_and_lambda_2, x_scaling_for_Q_and_lambda_2);
    ierr = VecGetArrayRead(Q_value_nodes, &Q_value_nodes_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(lambda_2_nodes, &lambda_2_nodes_p); CHKERRXX(ierr);
  }

  ierr = VecGetArrayRead(phi  , &phi_p  ); CHKERRXX(ierr);
  ierr = VecGetArrayRead(hodge, &hodge_p); CHKERRXX(ierr);

  for(unsigned char dir = 0; dir < P4EST_DIM; ++dir){
    ierr = VecGetArrayRead(vnp1_nodes[dir], &vn_p[dir]); CHKERRXX(ierr);
  }

  const double *vort_p;
  ierr = VecGetArrayRead(vorticity, &vort_p); CHKERRXX(ierr);

  /* compute the pressure at nodes for visualization */
  Vec pressure_nodes;
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &pressure_nodes); CHKERRXX(ierr);

  //  // [Raphael:] this is how it was done before, I don't like it:
//  // 1) no synchronization of ghost node values --> sometimes very ugly in paraview, at processors' boundaries
//  // 2) the quadrant neighborhood of the node depends on the quadrant claimed to be the 'owner' of the node
//  //    --> this is implementation-dependent (dependence on the particular implementation of 'find_smallest_quadrant_containing_point')
//  //    --> doesn't smooth out possible cell-discontinuities that the projection had to put in to ensure divergence-free face-based velocities (especially at T-junctions)
//  my_p4est_interpolation_cells_t interp_c(ngbd_c, ngbd_n);
//  interp_c.set_input(pressure, phi, bc_pressure);
//  for(size_t n = 0; n < nodes_n->indep_nodes.elem_count; ++n)
//  {
//    double xyz[P4EST_DIM];
//    node_xyz_fr_n(n, p4est_n, nodes_n, xyz);
//    interp_c.add_point(n, xyz);
//  }
//  interp_c.interpolate(pressure_nodes);

//  // So I changed it to this: [Elyce : Raphael -- I moved your new interpolation into it's own function but preserved the behavior (I believe) so it could be called for independent usage without having code duplication/ I've left the original code in comments below though]
  compute_pressure_at_nodes(&pressure_nodes);

  const double *pressure_nodes_p;
  ierr = VecGetArrayRead(pressure_nodes, &pressure_nodes_p); CHKERRXX(ierr);
  /*
  for (size_t k = 0; k < ngbd_n->get_layer_size(); ++k) {
    p4est_locidx_t node_idx = ngbd_n->get_layer_node(k);
    pressure_nodes_p[node_idx] = interpolate_cell_field_at_node(node_idx, ngbd_c, ngbd_n, pressure, bc_pressure, phi);
  }
  ierr = VecGhostUpdateBegin(pressure_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for (size_t k = 0; k < ngbd_n->get_local_size(); ++k) {
    p4est_locidx_t node_idx = ngbd_n->get_local_node(k);
    pressure_nodes_p[node_idx] = interpolate_cell_field_at_node(node_idx, ngbd_c, ngbd_n, pressure, bc_pressure, phi);
  }
  ierr = VecGhostUpdateEnd(pressure_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  */

  Vec leaf_level;
  ierr = VecCreateGhostCells(p4est_n, ghost_n, &leaf_level); CHKERRXX(ierr);
  PetscScalar *l_p;
  ierr = VecGetArray(leaf_level, &l_p); CHKERRXX(ierr);

  for(p4est_topidx_t tree_idx = p4est_n->first_local_tree; tree_idx <= p4est_n->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = p4est_tree_array_index(p4est_n->trees, tree_idx);
    for(size_t q = 0; q < tree->quadrants.elem_count; ++q)
      l_p[tree->quadrants_offset + q] = p4est_quadrant_array_index(&tree->quadrants, q)->level;
  }

  for(size_t q = 0; q < ghost_n->ghosts.elem_count; ++q)
    l_p[p4est_n->local_num_quadrants + q] = p4est_quadrant_array_index(&ghost_n->ghosts, q)->level;

  const double *smoke_p;
  if(smoke != NULL)
  {
    ierr = VecGetArrayRead(smoke, &smoke_p); CHKERRXX(ierr);

    if(with_Q_and_lambda_2_value)
      my_p4est_vtk_write_all_general(p4est_n, nodes_n, ghost_n,
                                     P4EST_TRUE, P4EST_TRUE,
                                     6, /* number of VTK_NODE_SCALAR */
                                     1, /* number of VTK_NODE_VECTOR_BY_COMPONENTS */
                                     0, /* number of VTK_NODE_VECTOR_BY_BLOCK */
                                     1, /* number of VTK_CELL_SCALAR */
                                     0, /* number of VTK_CELL_VECTOR_BY_COMPONENTS */
                                     0, /* number of VTK_CELL_VECTOR_BY_BLOCK */
                                     name,
                                     VTK_NODE_SCALAR, "phi", phi_p,
                                     VTK_NODE_SCALAR, "pressure", pressure_nodes_p,
                                     VTK_NODE_SCALAR, "smoke", smoke_p,
                                     VTK_NODE_SCALAR, "vorticity", vort_p,
                                     VTK_NODE_SCALAR, "Q-value", Q_value_nodes_p,
                                     VTK_NODE_SCALAR, "lambda_2", lambda_2_nodes_p,
                                     VTK_NODE_VECTOR_BY_COMPONENTS, "velocity", DIM(vn_p[0], vn_p[1], vn_p[2]),
          VTK_CELL_SCALAR, "leaf_level", l_p );
    else
      my_p4est_vtk_write_all_general(p4est_n, nodes_n, ghost_n,
                                     P4EST_TRUE, P4EST_TRUE,
                                     4, /* number of VTK_NODE_SCALAR */
                                     1, /* number of VTK_NODE_VECTOR_BY_COMPONENTS */
                                     0, /* number of VTK_NODE_VECTOR_BY_BLOCK */
                                     1, /* number of VTK_CELL_SCALAR */
                                     0, /* number of VTK_CELL_VECTOR_BY_COMPONENTS */
                                     0, /* number of VTK_CELL_VECTOR_BY_BLOCK */
                                     name,
                                     VTK_NODE_SCALAR, "phi", phi_p,
                                     VTK_NODE_SCALAR, "pressure", pressure_nodes_p,
                                     VTK_NODE_SCALAR, "smoke", smoke_p,
                                     VTK_NODE_SCALAR, "vorticity", vort_p,
                                     VTK_NODE_VECTOR_BY_COMPONENTS, "velocity", DIM(vn_p[0], vn_p[1], vn_p[2]),
          VTK_CELL_DATA, "leaf_level", l_p);
    ierr = VecRestoreArrayRead(smoke, &smoke_p); CHKERRXX(ierr);
  }
  else
  {
    if(with_Q_and_lambda_2_value)
      my_p4est_vtk_write_all_general(p4est_n, nodes_n, ghost_n,
                                     P4EST_TRUE, P4EST_TRUE,
                                     5, /* number of VTK_NODE_SCALAR */
                                     1, /* number of VTK_NODE_VECTOR_BY_COMPONENTS */
                                     0, /* number of VTK_NODE_VECTOR_BY_BLOCK */
                                     1, /* number of VTK_CELL_SCALAR */
                                     0, /* number of VTK_CELL_VECTOR_BY_COMPONENTS */
                                     0, /* number of VTK_CELL_VECTOR_BY_BLOCK */
                                     name,
                                     VTK_NODE_SCALAR, "phi", phi_p,
                                     VTK_NODE_SCALAR, "pressure", pressure_nodes_p,
                                     VTK_NODE_SCALAR, "vorticity", vort_p,
                                     VTK_NODE_SCALAR, "Q-value", Q_value_nodes_p,
                                     VTK_NODE_SCALAR, "lambda_2", lambda_2_nodes_p,
                                     VTK_NODE_VECTOR_BY_COMPONENTS, "velocity", DIM(vn_p[0], vn_p[1], vn_p[2]),
          VTK_CELL_DATA, "leaf_level", l_p);
    else
      my_p4est_vtk_write_all_general(p4est_n, nodes_n, ghost_n,
                                     P4EST_TRUE, P4EST_TRUE,
                                     3, /* number of VTK_NODE_SCALAR */
                                     1, /* number of VTK_NODE_VECTOR_BY_COMPONENTS */
                                     0, /* number of VTK_NODE_VECTOR_BY_BLOCK */
                                     1, /* number of VTK_CELL_SCALAR */
                                     0, /* number of VTK_CELL_VECTOR_BY_COMPONENTS */
                                     0, /* number of VTK_CELL_VECTOR_BY_BLOCK */
                                     name,
                                     VTK_NODE_SCALAR, "phi", phi_p,
                                     VTK_NODE_SCALAR, "pressure", pressure_nodes_p,
                                     VTK_NODE_SCALAR, "vorticity", vort_p,
                                     VTK_NODE_VECTOR_BY_COMPONENTS, "velocity", DIM(vn_p[0], vn_p[1], vn_p[2]),
          VTK_CELL_DATA, "leaf_level", l_p);
  }

  ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
  ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

  ierr = VecRestoreArrayRead(phi  , &phi_p  ); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(hodge, &hodge_p); CHKERRXX(ierr);

  ierr = VecRestoreArrayRead(vorticity, &vort_p); CHKERRXX(ierr);

  ierr = VecRestoreArrayRead(pressure_nodes, &pressure_nodes_p); CHKERRXX(ierr);
  ierr = VecDestroy(pressure_nodes); CHKERRXX(ierr);

  for(unsigned char dir = 0; dir < P4EST_DIM; ++dir){
    ierr = VecRestoreArrayRead(vn_nodes[dir], &vn_p[dir]); CHKERRXX(ierr);
  }

  if(with_Q_and_lambda_2_value)
  {
    ierr = VecRestoreArrayRead(Q_value_nodes, &Q_value_nodes_p); CHKERRXX(ierr); Q_value_nodes_p = NULL;
    ierr = VecRestoreArrayRead(lambda_2_nodes, &lambda_2_nodes_p); CHKERRXX(ierr); lambda_2_nodes_p = NULL;
    ierr = VecDestroy(Q_value_nodes); CHKERRXX(ierr); Q_value_nodes = NULL;
    ierr = VecDestroy(lambda_2_nodes); CHKERRXX(ierr); lambda_2_nodes = NULL;
  }

  ierr = PetscPrintf(p4est_n->mpicomm, "Saved visual data in ... %s\n", name); CHKERRXX(ierr);
}

void my_p4est_navier_stokes_t::global_mass_flow_through_slice(const unsigned int& dir, double& section, double& mass_flow) const
{
  PetscErrorCode ierr;
#ifdef CASL_THROWS
  if (dir >= P4EST_DIM)
    throw std::invalid_argument("my_p4est_navier_stokes_t::global_mass_flow_through_slice: the queried direction MUST be strictly smaller than P4EST_DIM");
#endif
  const splitting_criteria_t *data = (const splitting_criteria_t*) p4est_n->user_pointer;

  const double coarsest_cell_size   = convert_to_xyz[dir]/((double) (1 << data->min_lvl));
  const double comparison_threshold = 0.5*convert_to_xyz[dir]/((double) (1 << data->max_lvl));

#ifdef CASL_THROWS
  if((section < xyz_min[dir]) || (section > xyz_max[dir]))
    throw std::invalid_argument("my_p4est_navier_stokes_t::global_mass_flow_through_slice: the slice section must be in the computational domain!");
#endif
  const int tree_dim_idx = (int) floor((section - xyz_min[dir])/convert_to_xyz[dir]);
  double should_be_integer = (section - (xyz_min[dir] + tree_dim_idx*convert_to_xyz[dir]))/coarsest_cell_size;
  if(fabs(should_be_integer - (int) should_be_integer) > 1e-6)
  {
#ifdef CASL_THROWS
    throw std::invalid_argument("my_p4est_navier_stokes_t::global_mass_flow_through_slice: the mass flux can be evaluated only through a slice in the \n computational domain that coincides with cell faces of the coarsest cells: choose a valid section!");
#else
    section = xyz_min[dir] + tree_dim_idx*convert_to_xyz[dir] + ((int) should_be_integer)*(section - (xyz_min[dir] + tree_dim_idx*convert_to_xyz[dir]))/coarsest_cell_size;
    if(p4est_n->mpirank == 0)
      std::cerr << "my_p4est_navier_stokes_t::global_mass_flow_through_slice: the section for calculating the mass flow has been relocated!" << std::endl;
#endif
  }
  mass_flow = 0.0; // initialization

  const double *vel_p, *phi_p;
  double face_coordinate = DBL_MAX;
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(vnp1[dir], &vel_p); CHKERRXX(ierr);
  for (p4est_locidx_t face_idx = 0; face_idx < faces_n->num_local[dir]; ++face_idx) {
    switch (dir) {
    case dir::x:
      face_coordinate = faces_n->x_fr_f(face_idx, dir);
      break;
    case dir::y:
      face_coordinate = faces_n->y_fr_f(face_idx, dir);
      break;
#ifdef P4_TO_P8
    case dir::z:
      face_coordinate = faces_n->z_fr_f(face_idx, dir);
      break;
#endif
    default:
#ifdef CASL_THROWS
      throw std::invalid_argument("my_p4est_navier_stokes_t::global_mass_flow_through_slice: the queried direction MUST be strictly smaller than P4EST_DIM");
#endif
      break;
    }

    bool check_for_periodic_wrapping = is_periodic(p4est_n, dir) && (fabs(face_coordinate - xyz_min[dir]) < comparison_threshold || fabs(face_coordinate - xyz_max[dir]) < comparison_threshold);
    if (check_for_periodic_wrapping && (fabs(section - xyz_min[dir]) < comparison_threshold || fabs(section - xyz_max[dir]) < comparison_threshold))
      face_coordinate = section;
    if (fabs(face_coordinate - section) < comparison_threshold)
      mass_flow += rho*vel_p[face_idx]*faces_n->face_area_in_negative_domain(face_idx, dir, phi_p, nodes_n);
  }
  ierr = VecRestoreArrayRead(vnp1[dir], &vel_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &mass_flow, 1, MPI_DOUBLE, MPI_SUM, p4est_n->mpicomm); SC_CHECK_MPI(mpiret);
}

void my_p4est_navier_stokes_t::get_noslip_wall_forces(double wall_force[], const bool with_pressure) const
{
  const splitting_criteria_t *data = (const splitting_criteria_t*) p4est_n->user_pointer;
  double xyz_stencil[P4EST_DIM], xyz_w[P4EST_DIM], xyz_f[P4EST_DIM];
  const double rr = 0.95;
  const double *pressure_p;
  const double *vnp1_p[P4EST_DIM];
  PetscErrorCode ierr;
  if(with_pressure)
  {
    ierr = VecGetArrayRead(pressure, &pressure_p); CHKERRXX(ierr);
  }
  for (unsigned char dd = 0; dd < P4EST_DIM; ++dd) {
    ierr = VecGetArrayRead(vnp1[dd], &vnp1_p[dd]); CHKERRXX(ierr);
  }

  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    // initialize calculation of the force component
    wall_force[dir] = 0.0;
    for (p4est_locidx_t normal_face_idx = 0; normal_face_idx < faces_n->num_local[dir]; ++normal_face_idx) {
      p4est_locidx_t quad_idx;
      p4est_topidx_t tree_idx;
      faces_n->f2q(normal_face_idx, dir, quad_idx, tree_idx);
      p4est_quadrant_t *quad;
      if(quad_idx < p4est_n->local_num_quadrants)
      {
        p4est_tree_t *tree = p4est_tree_array_index(p4est_n->trees, tree_idx);
        quad = p4est_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);
      }
      else
        quad = p4est_quadrant_array_index(&ghost_n->ghosts, quad_idx - p4est_n->local_num_quadrants);

      bool is_quad_wall[P4EST_FACES];
      bool no_wall = true;
      for (unsigned char ff = 0; ff < P4EST_FACES; ++ff)
      {
        is_quad_wall[ff] = is_quad_Wall(p4est_n,tree_idx, quad, ff);
        no_wall = no_wall && !is_quad_wall[ff];
      }

      if(no_wall)
        continue;

      faces_n->xyz_fr_f(normal_face_idx, dir, xyz_f);
      int tmp = (faces_n->q2f(quad_idx, 2*dir) == normal_face_idx ? 0 : 1);

      for (unsigned char wall_dir = 0; wall_dir < P4EST_DIM; ++wall_dir) {
        if(is_quad_wall[2*wall_dir] || is_quad_wall[2*wall_dir + 1])
        {
          // if the normal face is parallel to the considered wall normal, then it MUST be the wall face, otherwise skip it (the appropriate one is another one)
          if(dir == wall_dir && !(tmp == 0 ? is_quad_wall[2*wall_dir] : is_quad_wall[2*wall_dir + 1]))
            continue;

          // if the normal face is a wall face, and if it is a no-slip wall face (i.e. xyz_w = xyz_f)
          if(dir == wall_dir && is_no_slip(xyz_f)) // always in the domain boundary by definition in this case
          {
            double fraction_noslip = 0.0;
            for (unsigned char dd = 0; dd < P4EST_DIM; ++dd)
              xyz_stencil[dd] = xyz_f[dd];
            for (char ii = -1; ii < 2; ii+=2) {
              int first_transverse_dir = (dir + 1)%P4EST_DIM;
              xyz_stencil[first_transverse_dir] = xyz_f[first_transverse_dir] + ((double) ii)*0.5*rr*dxyz_min[first_transverse_dir]*((double) (1<<(data->max_lvl - quad->level)));
#ifdef P4_TO_P8
              if(is_no_slip(xyz_stencil)) // always in the domain boundary by definition in this case
              {
                int second_transverse_dir = (dir+2)%P4EST_DIM;
                for (char jj = -1; jj < 2; jj+=2) {
                  xyz_stencil[second_transverse_dir] = xyz_f[second_transverse_dir] + ((double) jj)*0.5*rr*dxyz_min[second_transverse_dir]*((double) (1<<(data->max_lvl - quad->level)));
                  fraction_noslip += ((is_no_slip(xyz_stencil))?0.25:0); // always in the domain boundary by definition in this case
                }
                xyz_stencil[second_transverse_dir] = xyz_f[second_transverse_dir]; // reset that one
              }
#else
              fraction_noslip += ((is_no_slip(xyz_stencil))?0.5:0.0); // always in the domain boundary by definition in this case
#endif
            }
            if(fraction_noslip < EPS)
              throw std::runtime_error("my_p4est_navier_stokes_t::get_noslip_wall_forces: this case is under-resolved, a wall-face has its center no-slip but no other point in that face is no-slip...");
            double element_drag = 0.0;
            if(with_pressure)
              element_drag -= (pressure_p[quad_idx] + bc_pressure->wallValue(xyz_f)*0.5*dxyz_min[dir]*((double) (1<<(data->max_lvl - quad->level)))); // NEUMANN on pressure, since no-slip wall...
            p4est_locidx_t opposite_face_idx = faces_n->q2f(quad_idx, 2*dir + 1 - tmp);
            P4EST_ASSERT(opposite_face_idx != NO_VELOCITY);
            double dvdirddir = (tmp == 1 ? +1.0 : -1.0)*(vnp1_p[dir][normal_face_idx] - vnp1_p[dir][opposite_face_idx])/(dxyz_min[dir]*((double) (1 << (data->max_lvl - quad->level))));
            element_drag += 2*mu*dvdirddir;
            element_drag *= faces_n->face_area(normal_face_idx, dir)*fraction_noslip;
            wall_force[dir] += (tmp == 1 ? +1.0 : -1.0)*element_drag;
          }
          if(dir != wall_dir)
          {
            for (unsigned char kk = 0; kk < 2; ++kk) {
              if(!is_quad_wall[2*wall_dir + kk])
                continue; // to handle dumbass cases with only 1 cell in transverse direction that some insane person might want to try one day...
              for (unsigned char dd = 0; dd < P4EST_DIM; ++dd)
                xyz_w[dd] = xyz_f[dd];
              xyz_w[wall_dir] += (2.0*kk - 1.0)*0.5*dxyz_min[wall_dir]*((double) (1<<(data->max_lvl - quad->level)));
              if(is_no_slip(xyz_w)) // always in the domain boundary by definition as wall
              {
                // first term: mu*dv[dir]/dwall_dir
                double element_drag = mu*((double) (1-2*kk))*(vnp1_p[dir][normal_face_idx] - bc_v[dir].wallValue(xyz_w))/(0.5*dxyz_min[wall_dir]*((double) (1<<(data->max_lvl - quad->level))));
                // second term: dv[wall_dir]/ddir
                double transverse_derivative = 0.0;
                double fraction_noslip = 0.0;
                unsigned int n_terms = 0;
                for (unsigned char dd = 0; dd < P4EST_DIM; ++dd)
                  xyz_stencil[dd] = xyz_w[dd];
#ifdef P4_TO_P8
                int second_transverse_dir = ((dir + 1)%P4EST_DIM == wall_dir ? (dir + 2)%P4EST_DIM : (dir + 1)%P4EST_DIM);
#endif
                for (char ii = -1; ii < 2; ii+=2){
                  xyz_stencil[dir] = xyz_w[dir] + ((double) ii)*0.5*rr*dxyz_min[dir]*((double) (1<<(data->max_lvl - quad->level)));
#ifdef P4_TO_P8
                  if(is_in_domain(xyz_stencil) && is_no_slip(xyz_stencil)) // could be out of the domain (consider a corner in a cavity flow, for instance)
                  {
                    for (char jj = -1; jj < 2; jj+=2) {
                      xyz_stencil[second_transverse_dir] = xyz_w[second_transverse_dir] + ((double) jj)*0.5*rr*dxyz_min[second_transverse_dir]*((double) (1<<(data->max_lvl - quad->level)));
                      fraction_noslip += ((is_in_domain(xyz_stencil) && is_no_slip(xyz_stencil))?0.25:0); // could be out of the domain (consider a corner in a cavity flow, for instance)
                    }
                    xyz_stencil[second_transverse_dir] = xyz_w[second_transverse_dir]; // reset that one
                  }
#else
                  fraction_noslip += (is_in_domain(xyz_stencil) && is_no_slip(xyz_stencil) ? 0.5 : 0.0); // could be out of the domain (consider a corner in a cavity flow, for instance)
#endif
                  if(is_in_domain(xyz_stencil) && bc_v[wall_dir].wallType(xyz_stencil) == DIRICHLET) // consistency for that component at least
                  {
                    transverse_derivative += ((double) ii)*(bc_v[wall_dir].wallValue(xyz_stencil) - bc_v[wall_dir].wallValue(xyz_w))/(0.5*rr*dxyz_min[dir]*((double) (1<<(data->max_lvl - quad->level))));
                    n_terms++;
                  }
                }

                if(n_terms == 0)
                  std::runtime_error("my_p4est_navier_stokes_t::get_noslip_wall_forces: this case is under-resolved, less than one-cell size DIRICHLET condition for x-velocity on an x-wall");
                if(n_terms == 2)
                  transverse_derivative *= 0.5;
                if(fraction_noslip < EPS)
                  throw std::runtime_error("my_p4est_navier_stokes_t::get_noslip_wall_forces: this case is under-resolved, a wall-element associated with a projected face center has its center no-slip but no other point in that element is no-slip...");

                element_drag += mu*transverse_derivative;
                element_drag *= dxyz_min[dir]*((double) (1<<(data->max_lvl - quad->level)));
#ifdef P4_TO_P8
                element_drag *= dxyz_min[second_transverse_dir]*((double) (1<<(data->max_lvl - quad->level)));
#endif
                wall_force[dir] += (2.0*kk - 1.0)*element_drag*fraction_noslip;
              }
            }
          }
        }
      }
    }
  }

  for (unsigned char dd = 0; dd < P4EST_DIM; ++dd) {
    ierr = VecRestoreArrayRead(vnp1[dd], &vnp1_p[dd]); CHKERRXX(ierr);
  }
  if(with_pressure)
  {
    ierr = VecRestoreArrayRead(pressure, &pressure_p); CHKERRXX(ierr);
  }

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &wall_force[0], P4EST_DIM, MPI_DOUBLE, MPI_SUM, p4est_n->mpicomm); SC_CHECK_MPI(mpiret);
}

void my_p4est_navier_stokes_t::save_state(const char* path_to_root_directory, double tn, unsigned int n_saved)
{
  if(!is_folder(path_to_root_directory))
  {
    if(!create_directory(path_to_root_directory, p4est_n->mpirank, p4est_n->mpicomm))
    {
      char error_msg[1024];
      sprintf(error_msg, "save_state: the path %s is invalid and the directory could not be created", path_to_root_directory);
      throw std::invalid_argument(error_msg);
    }
  }

  unsigned int backup_idx = 0;

  if(p4est_n->mpirank == 0)
  {
    unsigned int n_backup_subfolders = 0;
    // get the current number of backups already present
    // delete the extra ones that may exist for whatever reason
    std::vector<std::string> subfolders; subfolders.resize(0);
    get_subdirectories_in(path_to_root_directory, subfolders);
    char temp_backup_folder_to_delete[PATH_MAX]; unsigned int to_delete_idx = 0;
    for (size_t idx = 0; idx < subfolders.size(); ++idx) {
      if(!subfolders[idx].compare(0, 7, "backup_"))
      {
        unsigned int backup_idx;
        sscanf(subfolders[idx].c_str(), "backup_%ud", &backup_idx);
        if(backup_idx >= n_saved)
        {
          // delete extra backups existing for whatever reasons (renamed to temporary folders beforehand to avoid issues)
          char full_path[PATH_MAX];
          sprintf(full_path, "%s/%s", path_to_root_directory, subfolders[idx].c_str());
          sprintf(temp_backup_folder_to_delete, "%s/temp_backup_folder_to_delete_%d", path_to_root_directory, to_delete_idx++);
          rename(full_path, temp_backup_folder_to_delete);
          delete_directory(temp_backup_folder_to_delete, p4est_n->mpirank, p4est_n->mpicomm, true);
        }
        else
          n_backup_subfolders++;
      }
    }

    // check that they are successively indexed if less than the max number
    if(n_backup_subfolders < n_saved)
    {
      backup_idx = 0;
      for (unsigned int idx = 0; idx < n_backup_subfolders; ++idx) {
        char expected_dir[PATH_MAX];
        sprintf(expected_dir, "%s/backup_%d", path_to_root_directory, (int) idx);
        if(!is_folder(expected_dir))
          break; // well, it's a mess in there, but I can't really do any better...
        backup_idx++;
      }
    }
    if (n_saved > 1 && n_backup_subfolders == n_saved)
    {
      // delete the 0th (renamed to a temporary folder beforehand to avoid issues)
      char full_path_zeroth_index[PATH_MAX];
      sprintf(full_path_zeroth_index, "%s/backup_0", path_to_root_directory);
      sprintf(temp_backup_folder_to_delete, "%s/temp_backup_folder_to_delete_%d", path_to_root_directory, to_delete_idx++);
      rename(full_path_zeroth_index, temp_backup_folder_to_delete);
      delete_directory(temp_backup_folder_to_delete, p4est_n->mpirank, p4est_n->mpicomm, true);
      // shift the others
      for (size_t idx = 1; idx < n_saved; ++idx) {
        char old_name[PATH_MAX], new_name[PATH_MAX];
        sprintf(old_name, "%s/backup_%d", path_to_root_directory, (int) idx);
        sprintf(new_name, "%s/backup_%d", path_to_root_directory, (int) (idx - 1));
        rename(old_name, new_name);
      }
      backup_idx = n_saved - 1;
    }
  }
  int mpiret = MPI_Bcast(&backup_idx, 1, MPI_INT, 0, p4est_n->mpicomm); SC_CHECK_MPI(mpiret);// acts as a MPI_Barrier, too

  char path_to_folder[PATH_MAX];
  sprintf(path_to_folder, "%s/backup_%d", path_to_root_directory, (int) backup_idx);
  create_directory(path_to_folder, p4est_n->mpirank, p4est_n->mpicomm);


  char filename[PATH_MAX];
  // save the solver parameters
  sprintf(filename, "%s/solver_parameters", path_to_folder);
  save_or_load_parameters(filename, (splitting_criteria_t*) p4est_n->user_pointer, SAVE, tn);
  // save p4est_n and all corresponding data
  if(smoke == NULL)
    my_p4est_save_forest_and_data(path_to_folder, p4est_n, nodes_n, faces_n,
                                  "p4est_n", 4,
                                  "phi", 1, &phi,
                                  "hodge", 1, &hodge,
                                  "dxyz_hodge", P4EST_DIM, dxyz_hodge,
                                  "vn_nodes", P4EST_DIM, vn_nodes);
  else
    my_p4est_save_forest_and_data(path_to_folder, p4est_n, nodes_n, faces_n,
                                  "p4est_n", 5,
                                  "phi", 1, &phi,
                                  "hodge", 1, &hodge,
                                  "dxyz_hodge", P4EST_DIM, dxyz_hodge,
                                  "vn_nodes", P4EST_DIM, vn_nodes,
                                  "smoke", 1, &smoke);
  // save p4est_nm1
  my_p4est_save_forest_and_data(path_to_folder, p4est_nm1, nodes_nm1,
                                "p4est_nm1", 1,
                                "vnm1_nodes", P4EST_DIM, vnm1_nodes);
  PetscErrorCode ierr = PetscPrintf(p4est_n->mpicomm, "Saved solver state in ... %s\n", path_to_folder); CHKERRXX(ierr);
}

void my_p4est_navier_stokes_t::fill_or_load_double_parameters(save_or_load flag, std::vector<PetscReal>& data, splitting_criteria_t *splitting_criterion, double &tn)
{
  size_t idx = 0;
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
  {
    switch (flag) {
    case SAVE:
      data[idx++] = dxyz_min[dim];
      break;
    case LOAD:
      dxyz_min[dim] = data[idx++];
      break;
    default:
      throw std::runtime_error("my_p4est_navier_stokes_t::fill_or_load_double_data: unknown flag value");
      break;
    }
  }
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
  {
    switch (flag) {
    case SAVE:
      data[idx++] = xyz_min[dim];
      break;
    case LOAD:
      xyz_min[dim] = data[idx++];
      break;
    default:
      throw std::runtime_error("my_p4est_navier_stokes_t::fill_or_load_double_data: unknown flag value");
      break;
    }
  }
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
  {
    switch (flag) {
    case SAVE:
      data[idx++] = xyz_max[dim];
      break;
    case LOAD:
      xyz_max[dim] = data[idx++];
      break;
    default:
      throw std::runtime_error("my_p4est_navier_stokes_t::fill_or_load_double_data: unknown flag value");
      break;
    }
  }
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
  {
    switch (flag) {
    case SAVE:
      data[idx++] = convert_to_xyz[dim];
      break;
    case LOAD:
      convert_to_xyz[dim] = data[idx++];
      break;
    default:
      throw std::runtime_error("my_p4est_navier_stokes_t::fill_or_load_double_data: unknown flag value");
      break;
    }
  }
  {
    switch (flag) {
    case SAVE:
    {
      data[idx++] = mu;
      data[idx++] = rho;
      data[idx++] = tn;
      data[idx++] = dt_n;
      data[idx++] = dt_nm1;
      data[idx++] = max_L2_norm_u;
      data[idx++] = uniform_band;
      data[idx++] = vorticity_threshold_split_cell;
      data[idx++] = n_times_dt;
      data[idx++] = smoke_thresh;
      data[idx++] = splitting_criterion->lip;
      data[idx++] = norm_grad_u_threshold_split_cell;
      break;
    }
    case LOAD:
    {
      mu                              = data[idx++];
      rho                             = data[idx++];
      tn                              = data[idx++];
      dt_n                            = data[idx++];
      dt_nm1                          = data[idx++];
      max_L2_norm_u                   = data[idx++];
      uniform_band                    = data[idx++];
      vorticity_threshold_split_cell  = data[idx++];
      n_times_dt                      = data[idx++];
      smoke_thresh                    = data[idx++];
      splitting_criterion->lip        = data[idx++];
      norm_grad_u_threshold_split_cell= data[idx++];
      break;
    }
    default:
      throw std::runtime_error("my_p4est_navier_stokes_t::fill_or_load_double_data: unknown flag value");
      break;
    }
  }
  P4EST_ASSERT(idx == n_double_parameter_for_restart);
  return;
}

void my_p4est_navier_stokes_t::fill_or_load_integer_parameters(save_or_load flag, std::vector<PetscInt>& data, splitting_criteria_t* splitting_criterion)
{
  size_t idx = 0;
  switch (flag) {
  case SAVE:
  {
    data[idx++] = P4EST_DIM;
    data[idx++] = (PetscInt) refine_with_smoke;
    data[idx++] = splitting_criterion->min_lvl;
    data[idx++] = splitting_criterion->max_lvl;
    data[idx++] = sl_order;
    data[idx++] = static_cast<int>(interp_v_viscosity);
    data[idx++] = static_cast<int>(interp_v_update);
    break;
  }
  case LOAD:
  {
    PetscInt P4EST_DIM_COPY       = data[idx++];
    if(P4EST_DIM_COPY != P4EST_DIM)
      throw std::runtime_error("You're trying to load 2D (resp. 3D) data with a 3D (resp. 2D) program...");
    refine_with_smoke             = (bool) data[idx++];
    splitting_criterion->min_lvl  = data[idx++];
    splitting_criterion->max_lvl  = data[idx++];
    sl_order                      = data[idx++];
    interp_v_viscosity            = static_cast<interpolation_method>(data[idx++]);
    interp_v_update               = static_cast<interpolation_method>(data[idx++]);
    break;
  }
  default:
    throw std::runtime_error("my_p4est_navier_stokes_t::fill_or_load_integer_data: unknown flag value");
    break;
  }
  P4EST_ASSERT(idx == n_integer_parameter_for_restart);
}

void my_p4est_navier_stokes_t::save_or_load_parameters(const char* filename, splitting_criteria_t* splitting_criterion, save_or_load flag, double &tn, const mpi_environment_t* mpi)
{
  PetscErrorCode ierr;
  std::vector<PetscReal> double_parameters(n_double_parameter_for_restart);
  std::vector<PetscInt> integer_parameters(n_integer_parameter_for_restart);
  int fd;
  char diskfilename[PATH_MAX];
  switch (flag) {
  case SAVE:
  {
    if(p4est_n->mpirank == 0)
    {
      sprintf(diskfilename, "%s_integers", filename);
      fill_or_load_integer_parameters(flag, integer_parameters, splitting_criterion);
      ierr = PetscBinaryOpen(diskfilename, FILE_MODE_WRITE, &fd); CHKERRXX(ierr);
      ierr = PetscBinaryWrite(fd, integer_parameters.data(), n_integer_parameter_for_restart, PETSC_INT, PETSC_TRUE); CHKERRXX(ierr);
      ierr = PetscBinaryClose(fd); CHKERRXX(ierr);
      // Then we save the double parameters
      sprintf(diskfilename, "%s_doubles", filename);
      fill_or_load_double_parameters(flag, double_parameters, splitting_criterion, tn);
      ierr = PetscBinaryOpen(diskfilename, FILE_MODE_WRITE, &fd); CHKERRXX(ierr);
      ierr = PetscBinaryWrite(fd, double_parameters.data(), n_double_parameter_for_restart, PETSC_DOUBLE, PETSC_TRUE); CHKERRXX(ierr);
      ierr = PetscBinaryClose(fd); CHKERRXX(ierr);
    }
    break;
  }
  case LOAD:
  {
    sprintf(diskfilename, "%s_integers", filename);
    if(!file_exists(diskfilename))
      throw std::invalid_argument("my_p4est_navier_stokes_t::save_or_load_parameters: the file storing the solver's integer parameters could not be found");
    if(mpi->rank() == 0)
    {
      PetscInt count;
      ierr = PetscBinaryOpen(diskfilename, FILE_MODE_READ, &fd); CHKERRXX(ierr);
      ierr = PetscBinaryRead(fd, integer_parameters.data(), n_integer_parameter_for_restart, &count, PETSC_INT); CHKERRXX(ierr);
      ierr = PetscBinaryClose(fd); CHKERRXX(ierr);
      if(count < (PetscInt) min_n_integer_parameter_for_restart)
        throw std::runtime_error("my_p4est_navier_stokes_t::save_or_load_parameters: the file storing the solver's integer parameters is not valid (not enough data therein)...");
      for(int k = count; k < (PetscInt) n_integer_parameter_for_restart; k++)
      {
        switch (k) {
        case 5:
        case 6:
          integer_parameters[k] = static_cast<int>(quadratic); // default was quadratic interpolation, before
          break;
        default:
          throw std::runtime_error("my_p4est_navier_stokes_t::save_or_load_parameters: unknown default parameter to fill in missing data from backup file...");
          break;
        }
      }
    }
    int mpiret = MPI_Bcast(integer_parameters.data(), n_integer_parameter_for_restart, MPIU_INT, 0, mpi->comm()); SC_CHECK_MPI(mpiret); // "MPIU_INT" so that it still works if PetSc uses 64-bit integers (correct MPI type defined in Petscsys.h for you!)
    fill_or_load_integer_parameters(flag, integer_parameters, splitting_criterion);
    // Then we save the double parameters
    sprintf(diskfilename, "%s_doubles", filename);
    if(!file_exists(diskfilename))
      throw std::invalid_argument("my_p4est_navier_stokes_t::save_or_load_parameters: the file storing the solver's double parameters could not be found");
    if(mpi->rank() == 0)
    {
      PetscInt count;
      ierr = PetscBinaryOpen(diskfilename, FILE_MODE_READ, &fd); CHKERRXX(ierr);
      ierr = PetscBinaryRead(fd, double_parameters.data(), n_double_parameter_for_restart, &count, PETSC_DOUBLE); CHKERRXX(ierr);
      ierr = PetscBinaryClose(fd); CHKERRXX(ierr);
      if(count < (PetscInt) min_n_double_parameter_for_restart)
        throw std::runtime_error("my_p4est_navier_stokes_t::save_or_load_parameters: the file storing the solver's double parameters is not valid (not enough data therein)...");
      for(int k = count; k < (PetscInt) n_double_parameter_for_restart; k++)
      {
        switch (k) {
        case 4*P4EST_DIM + 11:
          double_parameters[k] = DBL_MAX; // default was not using that, before
          break;
        default:
          throw std::runtime_error("my_p4est_navier_stokes_t::save_or_load_parameters: unknown default parameter to fill in missing data from backup file...");
          break;
        }
      }
    }
    mpiret = MPI_Bcast(double_parameters.data(), n_double_parameter_for_restart, MPI_DOUBLE, 0, mpi->comm()); SC_CHECK_MPI(mpiret);
    fill_or_load_double_parameters(flag, double_parameters, splitting_criterion, tn);
    break;
  }
  default:
    throw std::runtime_error("my_p4est_navier_stokes_t::save_or_load_parameters: unknown flag value");
    break;
    break;
  }
}

void my_p4est_navier_stokes_t::load_state(const mpi_environment_t& mpi, const char* path_to_folder, double& tn)
{
  PetscErrorCode ierr;
  char filename[PATH_MAX];

  if(!is_folder(path_to_folder))
    throw std::invalid_argument("my_p4est_navier_stokes_t::load_state: path_to_folder is invalid.");

  // load general solver parameters first
  splitting_criteria_t* data = new splitting_criteria_t;
  sprintf(filename, "%s/solver_parameters", path_to_folder);
  save_or_load_parameters(filename, data, LOAD, tn, &mpi);

  sprintf(filename, "%s/smoke.petscbin", path_to_folder);

  // load p4est_n and the corresponding objects
  if(refine_with_smoke && file_exists(filename))
  {
    my_p4est_load_forest_and_data(mpi.comm(), path_to_folder,
                                  p4est_n, conn,
                                  P4EST_TRUE, ghost_n, nodes_n,
                                  P4EST_TRUE, brick, P4EST_TRUE, faces_n, hierarchy_n, ngbd_c,
                                  "p4est_n", 5,
                                  "phi", NODE_DATA, 1, &phi,
                                  "hodge", CELL_DATA, 1, &hodge,
                                  "dxyz_hodge", FACE_DATA, P4EST_DIM, dxyz_hodge,
                                  "vn_nodes", NODE_DATA, P4EST_DIM, vn_nodes,
                                  "smoke", NODE_DATA, 1, &smoke);
  }
  else
  {
    if(refine_with_smoke) // the original solver was refined with smoke, but the smoke was not exported for some reason...
      refine_with_smoke = false;
    my_p4est_load_forest_and_data(mpi.comm(), path_to_folder,
                                  p4est_n, conn,
                                  P4EST_TRUE, ghost_n, nodes_n,
                                  P4EST_TRUE, brick, P4EST_TRUE, faces_n, hierarchy_n, ngbd_c,
                                  "p4est_n", 4,
                                  "phi", NODE_DATA, 1, &phi,
                                  "hodge", CELL_DATA, 1, &hodge,
                                  "dxyz_hodge", FACE_DATA, P4EST_DIM, dxyz_hodge,
                                  "vn_nodes", NODE_DATA, P4EST_DIM, vn_nodes);
  }
  P4EST_ASSERT(find_max_level(p4est_n) == data->max_lvl);

  if(ngbd_n != NULL)
    delete ngbd_n;
  ngbd_n = new my_p4est_node_neighbors_t(hierarchy_n, nodes_n);
  ngbd_n->init_neighbors();
  // load p4est_nm1 and the corresponding objects
  p4est_connectivity_t* conn_nm1 = NULL;
  my_p4est_load_forest_and_data(mpi.comm(), path_to_folder,
                                p4est_nm1, conn_nm1,
                                P4EST_TRUE, ghost_nm1, nodes_nm1,
                                "p4est_nm1", 1,
                                "vnm1_nodes", NODE_DATA, P4EST_DIM, vnm1_nodes);
  p4est_connectivity_destroy(conn_nm1);

  p4est_nm1->connectivity = conn;
  if(hierarchy_nm1 != NULL)
    delete hierarchy_nm1;
  hierarchy_nm1 = new my_p4est_hierarchy_t(p4est_nm1, ghost_nm1, brick);
  if(ngbd_nm1 != NULL)
    delete ngbd_nm1;
  ngbd_nm1 = new my_p4est_node_neighbors_t(hierarchy_nm1, nodes_nm1);
  ngbd_nm1->init_neighbors();

  p4est_n->user_pointer = (void*) data;
  p4est_nm1->user_pointer = (void*) data;

  ierr = PetscPrintf(mpi.comm(), "Loaded solver state from ... %s\n", path_to_folder); CHKERRXX(ierr);
}

double my_p4est_navier_stokes_t::compute_velocity_at_local_node(const p4est_locidx_t& node_idx, const unsigned char& dir, const double *vnp1_dir_read_p, const bool& store_interpolators)
{
  if(!interpolators_from_face_to_nodes_are_set)
    return interpolate_velocity_at_node_n(faces_n, ngbd_n, node_idx, vnp1[dir], dir, face_is_well_defined[dir], 2, bc_v, (store_interpolators? &interpolator_from_face_to_nodes[dir][node_idx] : NULL));
  else
  {
    const face_interpolator& my_interpolator = interpolator_from_face_to_nodes[dir][node_idx];
    if(my_interpolator.size() == 0) // technically should be possible and tolerated only if far inside positive domain...
      return 0.0;
#ifdef P4EST_DEBUG
    const p4est_indep_t* node = (p4est_indep_t*) sc_array_index(&nodes_n->indep_nodes, node_idx);
#endif
    P4EST_ASSERT(my_interpolator.size() > 0);
    if(my_interpolator.size() == 1) // must be in positive domain OR a Dirichlet wall boundary condition
    {
      double xyz_node[P4EST_DIM]; node_xyz_fr_n(node_idx, p4est_n, nodes_n, xyz_node);
      if(my_interpolator[0].face_idx < 0) // it was a DIRICHLET wall node
      {
        P4EST_ASSERT(my_interpolator[0].face_idx == -1 && bc_v != NULL && is_node_Wall(p4est_n, node) && bc_v[dir].wallType(xyz_node) == DIRICHLET);
        return bc_v[dir].wallValue(xyz_node);
      }
      else
        return my_interpolator[0].weight*vnp1_dir_read_p[my_interpolator[0].face_idx];
    }
    else
    {
#ifdef P4EST_DEBUG
      if(is_node_Wall(p4est_n, node))
      {
        double xyz_node[P4EST_DIM]; node_xyz_fr_n(node_idx, p4est_n, nodes_n, xyz_node);
        P4EST_ASSERT(bc_v != NULL);
        if(bc_v[dir].wallType(xyz_node) == DIRICHLET) // should have been dealt with in the condition above...
          throw std::runtime_error("my_p4est_navier_stokes_t::compute_velocity_at_local_node: unexpected behavior when reusing interpolator_from_face_to_nodes on a Dirichlet wall node");
        if(bc_v[dir].wallType(xyz_node) == NEUMANN && fabs(bc_v[dir].wallValue(xyz_node)) > EPS)
          throw std::runtime_error("my_p4est_navier_stokes_t::compute_velocity_at_local_node: when reusing interpolator_from_face_to_nodes on a Neumann wall node, the Neumann boundary value MUST be 0.0");
      }
#endif
      double to_return = 0.0;
      for (size_t k = 0; k < my_interpolator.size(); ++k)
        to_return += my_interpolator[k].weight*vnp1_dir_read_p[my_interpolator[k].face_idx];
      return to_return;
    }
  }
}

void my_p4est_navier_stokes_t::refine_coarsen_grid_after_restart(const CF_DIM *level_set, bool do_reinitialization)
{
  double dt_nm1_saved             = dt_nm1;
  double dt_n_saved               = dt_n;
  p4est_t* p4est_nm1_saved        = p4est_copy(p4est_nm1, P4EST_FALSE);
  p4est_ghost_t* ghost_nm1_saved  = my_p4est_ghost_new(p4est_nm1_saved, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est_nm1_saved, ghost_nm1_saved);
  if(third_degree_ghost_are_required(convert_to_xyz))
    my_p4est_ghost_expand(p4est_nm1_saved, ghost_nm1_saved);
  P4EST_ASSERT(ghosts_are_equal(ghost_nm1_saved, ghost_nm1));
  p4est_nodes_t* nodes_nm1_saved  = my_p4est_nodes_new(p4est_nm1_saved, ghost_nm1_saved);
  P4EST_ASSERT(nodes_are_equal(p4est_n->mpisize, nodes_nm1, nodes_nm1_saved));

  Vec vnm1_nodes_saved[P4EST_DIM];
  Vec second_derivatives_vnm1_nodes_saved[P4EST_DIM][P4EST_DIM];

  PetscErrorCode ierr;
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecCopyGhost(vn_nodes[dir], vnp1_nodes[dir]); CHKERRXX(ierr);

    ierr = VecCreateGhostNodes(p4est_nm1_saved, nodes_nm1_saved, &vnm1_nodes_saved[dir]); CHKERRXX(ierr);
    ierr = VecCopyGhost(vnm1_nodes[dir], vnm1_nodes_saved[dir]); CHKERRXX(ierr);
    for (unsigned char dd = 0; dd < P4EST_DIM; ++dd) {
      ierr = VecCreateGhostNodes(p4est_nm1_saved, nodes_nm1_saved, &second_derivatives_vnm1_nodes_saved[dd][dir]); CHKERRXX(ierr);
      ierr = VecCopyGhost(second_derivatives_vnm1_nodes[dd][dir], second_derivatives_vnm1_nodes_saved[dd][dir]); CHKERRXX(ierr);
    }
  }
  compute_vorticity();

  update_from_tn_to_tnp1(level_set, false, do_reinitialization);

  const p4est_topidx_t* t2v = conn->tree_to_vertex;
  const double* v2c = conn->vertices;
  for (unsigned char dir = 0; dir < P4EST_DIM; dir++)
    dxyz_min[dir] = (v2c[3*t2v[P4EST_CHILDREN*0 + P4EST_CHILDREN - 1] + dir] - xyz_min[dir]) / (1 << (((splitting_criteria_t*) p4est_n->user_pointer)->max_lvl));
  dt_nm1    = dt_nm1_saved;
  dt_n      = dt_n_saved;
  if(p4est_nm1 != p4est_n)
    p4est_destroy(p4est_nm1);
  p4est_nm1 = p4est_nm1_saved;
  if(ghost_nm1 != ghost_n)
    p4est_ghost_destroy(ghost_nm1);
  ghost_nm1 = ghost_nm1_saved;
  if(nodes_nm1 != nodes_nm1_saved)
    p4est_nodes_destroy(nodes_nm1);
  nodes_nm1 = nodes_nm1_saved;
  if(hierarchy_nm1 != hierarchy_n)
    delete  hierarchy_nm1;
  hierarchy_nm1 = new my_p4est_hierarchy_t(p4est_nm1, ghost_nm1, brick);
  if(ngbd_nm1 != ngbd_n)
    delete  ngbd_nm1;
  ngbd_nm1 = new my_p4est_node_neighbors_t(hierarchy_nm1, nodes_nm1);

  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecDestroy(vnm1_nodes[dir]); CHKERRXX(ierr);
    vnm1_nodes[dir] = vnm1_nodes_saved[dir];
    for (unsigned char dd = 0; dd < P4EST_DIM; ++dd) {
      ierr = VecDestroy(second_derivatives_vnm1_nodes[dd][dir]); CHKERRXX(ierr);
      second_derivatives_vnm1_nodes[dd][dir] = second_derivatives_vnm1_nodes_saved[dd][dir];
    }
  }
}

unsigned long int my_p4est_navier_stokes_t::memory_estimate() const
{
  unsigned long int memory_used = 0;
  memory_used += my_p4est_brick_memory_estimate(brick);
  memory_used += p4est_connectivity_memory_used(conn);
  if(p4est_nm1 != p4est_n)
    memory_used += p4est_memory_used(p4est_nm1);
  memory_used += p4est_memory_used(p4est_n);
  if(ghost_nm1 != ghost_n)
    memory_used += p4est_ghost_memory_used(ghost_nm1);
  memory_used += p4est_ghost_memory_used(ghost_n);
  if(hierarchy_nm1 != hierarchy_n)
    memory_used += hierarchy_nm1->memory_estimate();
  memory_used+= hierarchy_n->memory_estimate();
  if(ngbd_nm1 != ngbd_n)
    memory_used += ngbd_nm1->memory_estimate();
  memory_used += ngbd_n->memory_estimate();
  // cell-neighbors memory footage is negligible, only contains pointers to other objects...
  memory_used += faces_n->memory_estimate();

  // internal variables dxyz_min, xyz_min, xyz_max, convert_to_xyz
  memory_used += 4*P4EST_DIM*sizeof (double);
  // other internal variables
  memory_used += sizeof (mu) + sizeof (rho) + sizeof (dt_n) + sizeof (dt_nm1) +
      sizeof (max_L2_norm_u) + sizeof (uniform_band) + sizeof (vorticity_threshold_split_cell) + sizeof (norm_grad_u_threshold_split_cell) +
      sizeof (n_times_dt) + sizeof (dt_updated) + sizeof (refine_with_smoke) + sizeof (smoke_thresh) +
      sizeof (sl_order);
  // xyz_n, xyz_nm1, face interpolators
  memory_used += sizeof(semi_lagrangian_backtrace_is_done) + sizeof(interpolators_from_face_to_nodes_are_set);
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    memory_used += backtraced_v_n[dir].size()*sizeof (double);
    memory_used += backtraced_v_nm1[dir].size()*sizeof (double);
    for (unsigned int k = 0; k < interpolator_from_face_to_nodes[dir].size(); ++k)
      memory_used += interpolator_from_face_to_nodes[dir][k].size()*sizeof (face_interpolator_element);
  }

  // petsc node vectors at time n: phi, grad_phi, vn_nodes[P4EST_DIM], vnp1_nodes[P4EST_DIM], vorticity, smoke
  memory_used += (1 + 3*P4EST_DIM + 1 + (smoke != NULL ? 1:0))*(nodes_n->indep_nodes.elem_count)*sizeof (PetscScalar);
  // petsc node vectors at time nm1: vnm1_nodes[P4EST_DIM],
  memory_used += P4EST_DIM*(nodes_nm1->indep_nodes.elem_count)*sizeof (PetscScalar);
  // petsc cell vectors at time n: hodge, pressure
  memory_used += 2*(p4est_n->local_num_quadrants + ghost_n->ghosts.elem_count)*sizeof (PetscScalar);
  // petsc face vectors at time n: dxyz_hodge[P4EST_DIM], vstar[P4EST_DIM], vnp1[P4EST_DIM], face_is_well_defined[P4EST_DIM]
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
    memory_used += 4*(faces_n->num_local[dim] + faces_n->num_ghost[dim])*sizeof (PetscScalar);

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &memory_used, 1, MPI_UNSIGNED_LONG, MPI_SUM, p4est_n->mpicomm); SC_CHECK_MPI(mpiret);

  return memory_used;
}

void my_p4est_navier_stokes_t::get_slice_averaged_vnp1_profile(const unsigned char& vel_component, const unsigned char& axis, std::vector<double>& avg_velocity_profile, const double u_scaling)
{
  P4EST_ASSERT(vel_component < P4EST_DIM && axis < P4EST_DIM && vel_component != axis && is_periodic(p4est_n, vel_component));
#ifdef P4_TO_P8
  for (unsigned char dd = 0; dd < P4EST_DIM; ++dd) {
    if(dd == vel_component || dd == axis)
      continue;
    P4EST_ASSERT(is_periodic(p4est_n, dd));
  }
#endif
  splitting_criteria_t* data = (splitting_criteria_t*) p4est_n->user_pointer;
  unsigned int ndouble = brick->nxyztrees[axis]*(1<<data->max_lvl); // equivalent number of data for a uniform grid with finest level of refinement
#ifdef P4EST_ENABLE_DEBUG
  avg_velocity_profile.resize(ndouble*(1 + 1), 0.0); // slice-averaged velocity component + area of the slice
#else
  avg_velocity_profile.resize(ndouble, 0.0); // slice-averaged velocity component
#endif
  const double elementary_area = MULTD(1.0, dxyz_min[(axis + 1)%P4EST_DIM], dxyz_min[(axis+2)%P4EST_DIM]);
  for (size_t k = 0; k < avg_velocity_profile.size(); ++k)
    avg_velocity_profile[k] = 0.0;

  PetscErrorCode ierr;
  const double* velocity_component_p;
  ierr = VecGetArrayRead(vnp1[vel_component], &velocity_component_p); CHKERRXX(ierr);
  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;
#ifdef P4EST_ENABLE_DEBUG
  p4est_topidx_t nb_tree_idx = -1;
#endif
  set_of_neighboring_quadrants ngbd;
  p4est_quadrant_t quad, nb_quad;

  for (p4est_locidx_t face_idx = 0; face_idx < faces_n->num_local[vel_component]; ++face_idx) {
    faces_n->f2q(face_idx, vel_component, quad_idx, tree_idx);
    const p4est_quadrant_t* quad_ptr;
    P4EST_ASSERT(quad_idx < p4est_n->local_num_quadrants);
    p4est_tree_t* tree = p4est_tree_array_index(p4est_n->trees, tree_idx);
    quad_ptr = p4est_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);
    if(faces_n->q2f(quad_idx, 2*vel_component) == face_idx)
    {
      quad = *quad_ptr; quad.p.piggy3.local_num = quad_idx;
      ngbd.clear();
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 2*vel_component);
      /* note that the potential neighbor has to be the same size or bigger and there MUST be a neighbor*/
      P4EST_ASSERT(ngbd.size() == 1);
      nb_quad = *ngbd.begin();
#ifdef P4EST_ENABLE_DEBUG
      nb_tree_idx = nb_quad.p.piggy3.which_tree;
#endif
    }
    else
    {
      P4EST_ASSERT(faces_n->q2f(quad_idx, 2*vel_component + 1) == face_idx);
      quad = *quad_ptr; quad.p.piggy3.local_num = quad_idx;
      ngbd.clear();
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 2*vel_component + 1);
      /* note that the potential neighbor has to be the same size or bigger and there MUST be a neighbor*/
      P4EST_ASSERT(ngbd.size() == 1);
      nb_quad = *ngbd.begin();
#ifdef P4EST_ENABLE_DEBUG
      nb_tree_idx = nb_quad.p.piggy3.which_tree;
#endif
    }
    p4est_topidx_t cartesian_tree_idx_along_axis = -1;
#ifdef P4EST_ENABLE_DEBUG
    p4est_topidx_t cartesian_nb_tree_idx_along_axis = -1;
#endif
    bool is_found = false;
    for (p4est_topidx_t tt = 0; tt < conn->num_trees; ++tt) {
      if (brick->nxyz_to_treeid[tt] == tree_idx)
        cartesian_tree_idx_along_axis = tt;
#ifdef P4EST_ENABLE_DEBUG
      if (brick->nxyz_to_treeid[tt] == nb_tree_idx)
        cartesian_nb_tree_idx_along_axis = tt;
      is_found = is_found || (cartesian_tree_idx_along_axis != -1 && cartesian_nb_tree_idx_along_axis != -1);
#else
      is_found = is_found || cartesian_tree_idx_along_axis != -1;
#endif
      if (is_found)
        break;
    }
    P4EST_ASSERT(is_found);
    switch (axis) {
    case dir::x:
      cartesian_tree_idx_along_axis = cartesian_tree_idx_along_axis%brick->nxyztrees[0];
#ifdef P4EST_ENABLE_DEBUG
      cartesian_nb_tree_idx_along_axis = cartesian_nb_tree_idx_along_axis%brick->nxyztrees[0];
#endif
      break;
    case dir::y:
      cartesian_tree_idx_along_axis = (cartesian_tree_idx_along_axis/brick->nxyztrees[0])%brick->nxyztrees[1];
#ifdef P4EST_ENABLE_DEBUG
      cartesian_nb_tree_idx_along_axis = (cartesian_nb_tree_idx_along_axis/brick->nxyztrees[0])%brick->nxyztrees[1];
#endif
      break;
#ifdef P4_TO_P8
    case dir::z:
      cartesian_tree_idx_along_axis = cartesian_tree_idx_along_axis/(brick->nxyztrees[0]*brick->nxyztrees[1]);
#ifdef P4EST_ENABLE_DEBUG
      cartesian_nb_tree_idx_along_axis = cartesian_nb_tree_idx_along_axis/(brick->nxyztrees[0]*brick->nxyztrees[1]);
#endif
      break;
#endif
    default:
      throw std::invalid_argument("my_p4est_navier_stokes_t::get_slice_averaged_vnp1_profile: unknown axis...");
      break;
    }
    P4EST_ASSERT(cartesian_tree_idx_along_axis == cartesian_nb_tree_idx_along_axis);
    unsigned int idx_in_profile;
    for (int k = 0; k < (1<<(data->max_lvl - quad.level)); ++k) {
      switch (axis) {
      case dir::x:
        idx_in_profile = cartesian_tree_idx_along_axis*(1<<data->max_lvl) + (quad.x/(1<<(P4EST_MAXLEVEL - data->max_lvl))) + k;
        break;
      case dir::y:
        idx_in_profile = cartesian_tree_idx_along_axis*(1<<data->max_lvl) + (quad.y/(1<<(P4EST_MAXLEVEL - data->max_lvl))) + k;
        break;
#ifdef P4_TO_P8
      case dir::z:
        idx_in_profile = cartesian_tree_idx_along_axis*(1<<data->max_lvl) + (quad.z/(1<<(P4EST_MAXLEVEL - data->max_lvl))) + k;
        break;
#endif
      default:
        throw std::invalid_argument("my_p4est_navier_stokes_t::get_slice_averaged_vnp1_profile: unknown axis...");
        break;
      }
      double weighting_area = 0.5*MULTD(elementary_area, ((1 << (data->max_lvl - nb_quad.level)) + (1 << (data->max_lvl - quad.level))), (1 << (data->max_lvl - quad.level)));
      avg_velocity_profile[idx_in_profile]          += velocity_component_p[face_idx]*weighting_area/u_scaling;
#ifdef P4EST_ENABLE_DEBUG
      avg_velocity_profile[ndouble+idx_in_profile]  +=weighting_area;
#endif
    }
  }
  ierr = VecRestoreArrayRead(vnp1[vel_component], &velocity_component_p); CHKERRXX(ierr);
  int mpiret;
  if(p4est_n->mpirank == 0){
    mpiret = MPI_Reduce(MPI_IN_PLACE, avg_velocity_profile.data(), avg_velocity_profile.size(), MPI_DOUBLE, MPI_SUM, 0, p4est_n->mpicomm); SC_CHECK_MPI(mpiret); }
  else{
    mpiret = MPI_Reduce(avg_velocity_profile.data(), avg_velocity_profile.data(), avg_velocity_profile.size(), MPI_DOUBLE, MPI_SUM, 0, p4est_n->mpicomm); SC_CHECK_MPI(mpiret); }
  const double expected_slice_area = (xyz_max[(axis + 1)%P4EST_DIM]-xyz_min[(axis + 1)%P4EST_DIM]) ONLY3D(*(xyz_max[(axis+2)%P4EST_DIM]-xyz_min[(axis+2)%P4EST_DIM]));
  if(!p4est_n->mpirank)
    for (unsigned int k = 0; k < ndouble; ++k) {
      P4EST_ASSERT(fabs(avg_velocity_profile[ndouble+k] - expected_slice_area) < 10.0*EPS*MAX(avg_velocity_profile[ndouble+k], expected_slice_area));
      avg_velocity_profile[k] /= expected_slice_area;
    }
}

void my_p4est_navier_stokes_t::get_line_averaged_vnp1_profiles(DIM(const unsigned char& vel_component, const unsigned char& axis, const unsigned char& averaging_direction),
                                                               const std::vector<unsigned int>& bin_index, std::vector< std::vector<double> >& avg_velocity_profile, const double u_scaling)
{
  P4EST_ASSERT(vel_component < P4EST_DIM && axis < P4EST_DIM && vel_component != axis && is_periodic(p4est_n, vel_component));
#ifdef P4_TO_P8
  P4EST_ASSERT((averaging_direction < P4EST_DIM) && (averaging_direction != axis) && is_periodic(p4est_n, averaging_direction));
  unsigned char transverse_direction = (3 & ~axis) & ~averaging_direction; // bitwise binary operation
  P4EST_ASSERT(transverse_direction != averaging_direction);
  P4EST_ASSERT(vel_component == averaging_direction || vel_component == transverse_direction);
#else
  unsigned char transverse_direction = (axis == dir::x ? dir::y : dir::x);
  P4EST_ASSERT(vel_component == transverse_direction);
#endif
  P4EST_ASSERT(transverse_direction < P4EST_DIM && transverse_direction != axis);
  splitting_criteria_t* data = (splitting_criteria_t*) p4est_n->user_pointer;
  if((brick->nxyztrees[transverse_direction]*(1<<data->max_lvl))%bin_index.size() != 0)
    throw std::invalid_argument("my_p4est_navier_stokes_t::get_line_averaged_vnp1_profiles(...): invalid size of bin indices, it does not match domain length by periodic repetition!");
#ifdef P4EST_ENABLE_DEBUG
  for (size_t k = 0; k < bin_index.size(); ++k)
    P4EST_ASSERT(bin_index[k]<avg_velocity_profile.size());
#endif
  unsigned int ndouble = brick->nxyztrees[axis]*(1<<data->max_lvl); // equivalent number of data for a uniform grid with finest level of refinement
  for (unsigned int profile_idx = 0; profile_idx < avg_velocity_profile.size(); ++profile_idx)
  {
#ifdef P4EST_ENABLE_DEBUG
    avg_velocity_profile[profile_idx].resize(ndouble*(1 + 1), 0.0); // line-averaged velocity component profiles + averaging lengths
#else
    avg_velocity_profile[profile_idx].resize(ndouble, 0.0); // line-averaged velocity component profiles
#endif
    for (size_t k = 0; k < avg_velocity_profile[profile_idx].size(); ++k)
      avg_velocity_profile[profile_idx][k] = 0.0;
  }

  PetscErrorCode ierr;
  const double* velocity_component_p;
  ierr = VecGetArrayRead(vnp1[vel_component], &velocity_component_p); CHKERRXX(ierr);
  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx, nb_tree_idx;
  set_of_neighboring_quadrants ngbd;
  p4est_quadrant_t quad, nb_quad;

  p4est_locidx_t bounds_along_axis[2];
#ifdef P4_TO_P8
  // if (vel_component == averaging_direction)
  unsigned int bounds_transverse_direction[2] = {0, 0}; // we declare these variables unsigned int to avoid misbehaved results due to type conversion during the periodic mapping hereunder
#endif
  // else, i.e., if (vel_component == transverse_direction)
  unsigned int logical_idx_of_face = 0; // we declare these variables unsigned int to avoid misbehaved results due to type conversion during the periodic mapping hereunder
  unsigned int negative_coverage = 0;
  unsigned int positive_coverage = 0;
  double covering_length;

  for (p4est_locidx_t face_idx = 0; face_idx < faces_n->num_local[vel_component]; ++face_idx) {
    faces_n->f2q(face_idx, vel_component, quad_idx, tree_idx);
    const p4est_quadrant_t* quad_ptr;
    P4EST_ASSERT(quad_idx < p4est_n->local_num_quadrants);
    p4est_tree_t* tree = p4est_tree_array_index(p4est_n->trees, tree_idx);
    quad_ptr = p4est_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);
    if(faces_n->q2f(quad_idx, 2*vel_component) == face_idx)
    {
      quad = *quad_ptr; quad.p.piggy3.local_num = quad_idx;
      ngbd.clear();
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 2*vel_component);
      /* note that the potential neighbor has to be the same size or bigger and there MUST be a neighbor*/
      P4EST_ASSERT(ngbd.size() == 1);
      nb_quad = *ngbd.begin();
      nb_tree_idx = ngbd.begin()->p.piggy3.which_tree;
    }
    else
    {
      P4EST_ASSERT(faces_n->q2f(quad_idx, 2*vel_component + 1) == face_idx);
      quad = *quad_ptr; quad.p.piggy3.local_num = quad_idx;
      ngbd.clear();
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 2*vel_component + 1);
      /* note that the potential neighbor has to be the same size or bigger and there MUST be a neighbor*/
      P4EST_ASSERT(ngbd.size() == 1);
      nb_quad = *ngbd.begin();
      nb_tree_idx = ngbd.begin()->p.piggy3.which_tree;
    }
    p4est_topidx_t cartesian_ordered_tree_idx = -1;
    p4est_topidx_t cartesian_ordered_nb_tree_idx = -1;
    bool are_found = false;
    for (p4est_topidx_t tt = 0; tt < conn->num_trees; ++tt) {
      if (brick->nxyz_to_treeid[tt] == tree_idx)
        cartesian_ordered_tree_idx = tt;
      if (brick->nxyz_to_treeid[tt] == nb_tree_idx)
        cartesian_ordered_nb_tree_idx = tt;
      are_found = are_found || (cartesian_ordered_tree_idx != -1 && cartesian_ordered_nb_tree_idx != -1);
      if (are_found)
        break;
    }
    P4EST_ASSERT(are_found);
    // logical bounds along the axis of the velocity profiles
    p4est_topidx_t cartesian_tree_idx_along_axis;
#ifdef P4EST_ENABLE_DEBUG
    p4est_topidx_t cartesian_nb_tree_idx_along_axis;
#endif
    switch (axis) {
    case dir::x:
      cartesian_tree_idx_along_axis     = cartesian_ordered_tree_idx%brick->nxyztrees[0];
      bounds_along_axis[0]              = quad.x/(1<<(P4EST_MAXLEVEL - data->max_lvl));
#ifdef P4EST_ENABLE_DEBUG
      cartesian_nb_tree_idx_along_axis  = cartesian_ordered_nb_tree_idx%brick->nxyztrees[0];
#endif
      break;
    case dir::y:
      cartesian_tree_idx_along_axis     = (cartesian_ordered_tree_idx/brick->nxyztrees[0])%brick->nxyztrees[1];
      bounds_along_axis[0]              = quad.y/(1<<(P4EST_MAXLEVEL - data->max_lvl));
#ifdef P4EST_ENABLE_DEBUG
      cartesian_nb_tree_idx_along_axis  = (cartesian_ordered_nb_tree_idx/brick->nxyztrees[0])%brick->nxyztrees[1];
#endif
      break;
#ifdef P4_TO_P8
    case dir::z:
      cartesian_tree_idx_along_axis     = cartesian_ordered_tree_idx/(brick->nxyztrees[0]*brick->nxyztrees[1]);
      bounds_along_axis[0]              = quad.z/(1<<(P4EST_MAXLEVEL - data->max_lvl));
#ifdef P4EST_ENABLE_DEBUG
      cartesian_nb_tree_idx_along_axis  = cartesian_ordered_nb_tree_idx/(brick->nxyztrees[0]*brick->nxyztrees[1]);
#endif
      break;
#endif
    default:
      throw std::invalid_argument("my_p4est_navier_stokes_t::get_line_averaged_vnp1_profiles: unknown axis...");
      break;
    }
    bounds_along_axis[0]               += cartesian_tree_idx_along_axis*(1<<data->max_lvl);
    bounds_along_axis[1]                = bounds_along_axis[0] + (1<<(data->max_lvl - quad.level));
    P4EST_ASSERT(cartesian_tree_idx_along_axis == cartesian_nb_tree_idx_along_axis);
    // logical bounds along transverse direction
    // if (vel_component == averaging_direction), the bounds are the owning quadrant's bounds in the transverse direction
    // otherwise, i.e. if (vel_component == transverse_direction), the bounds are of the form
    // [logical_idx_of_face-/+(2**(lmax-quad.level))/2 : logical_idx_of_face+/-(2**(lmax-nb_quad.level))/2] with weight
    // factors of 0.5 for the extremities if different than the logical index of the face...
    p4est_topidx_t cartesian_tree_idx_along_transverse_dir;
#if defined(P4_TO_P8) && defined(P4EST_ENABLE_DEBUG)
    p4est_topidx_t cartesian_nb_tree_idx_along_transverse_dir;
#endif
    switch (transverse_direction) {
    case dir::x:
      cartesian_tree_idx_along_transverse_dir     = cartesian_ordered_tree_idx%brick->nxyztrees[0];
#ifdef P4_TO_P8
      if(vel_component == averaging_direction)
        bounds_transverse_direction[0]            = quad.x/(1<<(P4EST_MAXLEVEL - data->max_lvl));
      else
        logical_idx_of_face                       = (quad.x + ((faces_n->q2f(quad_idx, 2*vel_component + 1) == face_idx)? P4EST_QUADRANT_LEN(quad.level) : 0))/(1<<(P4EST_MAXLEVEL - data->max_lvl));
#ifdef P4EST_ENABLE_DEBUG
      cartesian_nb_tree_idx_along_transverse_dir  = cartesian_ordered_nb_tree_idx%brick->nxyztrees[0];
#endif
#else
      logical_idx_of_face                         = (quad.x + ((faces_n->q2f(quad_idx, 2*vel_component + 1) == face_idx)? P4EST_QUADRANT_LEN(quad.level) : 0))/(1<<(P4EST_MAXLEVEL - data->max_lvl));
#endif
      break;
    case dir::y:
      cartesian_tree_idx_along_transverse_dir     = (cartesian_ordered_tree_idx/brick->nxyztrees[0])%brick->nxyztrees[1];
#ifdef P4_TO_P8
      if(vel_component == averaging_direction)
        bounds_transverse_direction[0]            = quad.y/(1<<(P4EST_MAXLEVEL - data->max_lvl));
      else
        logical_idx_of_face                       = (quad.y + ((faces_n->q2f(quad_idx, 2*vel_component + 1) == face_idx)? P4EST_QUADRANT_LEN(quad.level) : 0))/(1<<(P4EST_MAXLEVEL - data->max_lvl));
#ifdef P4EST_ENABLE_DEBUG
      cartesian_nb_tree_idx_along_transverse_dir  = (cartesian_ordered_nb_tree_idx/brick->nxyztrees[0])%brick->nxyztrees[1];
#endif
#else
      logical_idx_of_face                         = (quad.y + ((faces_n->q2f(quad_idx, 2*vel_component + 1) == face_idx)? P4EST_QUADRANT_LEN(quad.level) : 0))/(1<<(P4EST_MAXLEVEL - data->max_lvl));
#endif
      break;
#ifdef P4_TO_P8
    case dir::z:
      cartesian_tree_idx_along_transverse_dir     = cartesian_ordered_tree_idx/(brick->nxyztrees[0]*brick->nxyztrees[1]);
      if(vel_component == averaging_direction)
        bounds_transverse_direction[0]            = quad.z/(1<<(P4EST_MAXLEVEL - data->max_lvl));
      else
        logical_idx_of_face                       = (quad.z + ((faces_n->q2f(quad_idx, 2*vel_component + 1) == face_idx)? P4EST_QUADRANT_LEN(quad.level) : 0))/(1<<(P4EST_MAXLEVEL - data->max_lvl));
#ifdef P4EST_ENABLE_DEBUG
      cartesian_nb_tree_idx_along_transverse_dir  = cartesian_ordered_nb_tree_idx/(brick->nxyztrees[0]*brick->nxyztrees[1]);
#endif
      break;
#endif
    default:
      throw std::invalid_argument("my_p4est_navier_stokes_t::get_line_averaged_vnp1_profiles: unknown transverse direction...");
      break;
    }
#ifdef P4_TO_P8
    if(vel_component == averaging_direction)
    {
      P4EST_ASSERT(cartesian_tree_idx_along_transverse_dir == cartesian_nb_tree_idx_along_transverse_dir);
      bounds_transverse_direction[0]             += cartesian_tree_idx_along_transverse_dir*(1<<data->max_lvl);
      bounds_transverse_direction[1]              = bounds_transverse_direction[0] + (1<<(data->max_lvl - quad.level));
      covering_length                             = 0.5*dxyz_min[averaging_direction]*((double) (1<<(data->max_lvl-quad.level)) + (double) (1<<(data->max_lvl-nb_quad.level)));
    }
    else
    {
      P4EST_ASSERT(vel_component == transverse_direction);
      logical_idx_of_face                        += cartesian_tree_idx_along_transverse_dir*(1<<data->max_lvl);
      negative_coverage                           = data->max_lvl - ((faces_n->q2f(quad_idx, 2*vel_component + 1) == face_idx)? quad.level: nb_quad.level);
      negative_coverage                           = (negative_coverage > 0 ? (1 << (negative_coverage - 1)) : 0);
      positive_coverage                           = data->max_lvl - ((faces_n->q2f(quad_idx, 2*vel_component + 1) == face_idx)? nb_quad.level: quad.level);
      positive_coverage                           = ((positive_coverage > 0)? (1<<(positive_coverage - 1)) : 0);
      covering_length                             = dxyz_min[averaging_direction]*((double) (1<<(data->max_lvl-quad.level)));
    }
#else
    logical_idx_of_face                          += cartesian_tree_idx_along_transverse_dir*(1<<data->max_lvl);
    negative_coverage                             = data->max_lvl - ((faces_n->q2f(quad_idx, 2*vel_component + 1) == face_idx)? quad.level: nb_quad.level);
    negative_coverage                             = (negative_coverage > 0 ? (1 << (negative_coverage - 1)) : 0);
    positive_coverage                             = data->max_lvl - ((faces_n->q2f(quad_idx, 2*vel_component + 1) == face_idx)? nb_quad.level: quad.level);
    positive_coverage                             = ((positive_coverage > 0)? (1<<(positive_coverage - 1)) : 0);
    covering_length                               = 1.0;
#endif

    for (p4est_locidx_t idx_in_profile = bounds_along_axis[0]; idx_in_profile < bounds_along_axis[1]; ++idx_in_profile) {
      size_t wrapped_idx;
#ifdef P4_TO_P8
      if(vel_component == averaging_direction)
      {
        for (unsigned int transverse_logical_idx = bounds_transverse_direction[0]; transverse_logical_idx < bounds_transverse_direction[1]; ++transverse_logical_idx) {
          wrapped_idx = transverse_logical_idx%bin_index.size();
          avg_velocity_profile[bin_index.at(wrapped_idx)][idx_in_profile] += covering_length*velocity_component_p[face_idx]/u_scaling;
#ifdef P4EST_ENABLE_DEBUG
          avg_velocity_profile[bin_index.at(wrapped_idx)][idx_in_profile+ndouble] += covering_length;
#endif
        }
      }
      else
      {
        wrapped_idx = logical_idx_of_face%bin_index.size();
        avg_velocity_profile[bin_index.at(wrapped_idx)][idx_in_profile] += covering_length*velocity_component_p[face_idx]/u_scaling;
#ifdef P4EST_ENABLE_DEBUG
        avg_velocity_profile[bin_index.at(wrapped_idx)][idx_in_profile+ndouble] += covering_length;
#endif
        for (unsigned int k = 1; k <= negative_coverage; ++k) {
          wrapped_idx = (logical_idx_of_face + (k > logical_idx_of_face ? ((k - logical_idx_of_face)/bin_index.size() + 1)*bin_index.size() : 0) - k)%bin_index.size(); // avoid negative intermediary result...
          avg_velocity_profile[bin_index.at(wrapped_idx)][idx_in_profile] += (k == negative_coverage ? 0.5: 1.0)*covering_length*velocity_component_p[face_idx]/u_scaling;
#ifdef P4EST_ENABLE_DEBUG
          avg_velocity_profile[bin_index.at(wrapped_idx)][idx_in_profile+ndouble] += (k == negative_coverage ? 0.5 : 1.0)*covering_length;
#endif
        }
        for (unsigned int k = 1; k <= positive_coverage; ++k) {
          wrapped_idx = (logical_idx_of_face + k)%bin_index.size();
          avg_velocity_profile[bin_index.at(wrapped_idx)][idx_in_profile] += (k == positive_coverage ? 0.5: 1.0)*covering_length*velocity_component_p[face_idx]/u_scaling;
#ifdef P4EST_ENABLE_DEBUG
          avg_velocity_profile[bin_index.at(wrapped_idx)][idx_in_profile+ndouble] += (k == positive_coverage ? 0.5: 1.0)*covering_length;
#endif
        }
      }
#else
      wrapped_idx = logical_idx_of_face%bin_index.size();
      avg_velocity_profile[bin_index.at(wrapped_idx)][idx_in_profile] += covering_length*velocity_component_p[face_idx]/u_scaling;
#ifdef P4EST_ENABLE_DEBUG
      avg_velocity_profile[bin_index.at(wrapped_idx)][idx_in_profile+ndouble] += covering_length;
#endif
      for (unsigned int k = 1; k <= negative_coverage; ++k) {
        wrapped_idx = (logical_idx_of_face + (k > logical_idx_of_face ? ((k - logical_idx_of_face)/bin_index.size() + 1)*bin_index.size() : 0) - k)%bin_index.size(); // avoid negative intermediary result...
        avg_velocity_profile[bin_index.at(wrapped_idx)][idx_in_profile] += (k == negative_coverage ? 0.5 : 1.0)*covering_length*velocity_component_p[face_idx]/u_scaling;
#ifdef P4EST_ENABLE_DEBUG
        avg_velocity_profile[bin_index.at(wrapped_idx)][idx_in_profile+ndouble] += (k == negative_coverage ? 0.5 : 1.0)*covering_length;
#endif
      }
      for (unsigned int k = 1; k <= positive_coverage; ++k) {
        wrapped_idx = (logical_idx_of_face + k)%bin_index.size();
        avg_velocity_profile[bin_index.at(wrapped_idx)][idx_in_profile] += (k == positive_coverage ? 0.5 : 1.0)*covering_length*velocity_component_p[face_idx]/u_scaling;
#ifdef P4EST_ENABLE_DEBUG
        avg_velocity_profile[bin_index.at(wrapped_idx)][idx_in_profile+ndouble] += (k == positive_coverage ? 0.5 : 1.0)*covering_length;
#endif
      }
#endif
    }
  }
  ierr = VecRestoreArrayRead(vnp1[vel_component], &velocity_component_p); CHKERRXX(ierr);

  int mpiret;
  std::vector<MPI_Request> requests(avg_velocity_profile.size());
  for (size_t profile_idx = 0; profile_idx < avg_velocity_profile.size(); ++profile_idx) {
    int rcv_rank = profile_idx%p4est_n->mpisize;
    if(p4est_n->mpirank == rcv_rank){
      mpiret = MPI_Ireduce(MPI_IN_PLACE, avg_velocity_profile[profile_idx].data(), avg_velocity_profile[profile_idx].size(), MPI_DOUBLE, MPI_SUM, rcv_rank, p4est_n->mpicomm, &requests[profile_idx]); SC_CHECK_MPI(mpiret); }
    else{
      mpiret = MPI_Ireduce(avg_velocity_profile[profile_idx].data(), avg_velocity_profile[profile_idx].data(), avg_velocity_profile[profile_idx].size(), MPI_DOUBLE, MPI_SUM, rcv_rank, p4est_n->mpicomm, &requests[profile_idx]); SC_CHECK_MPI(mpiret); }
  }
  mpiret = MPI_Waitall(avg_velocity_profile.size(), &requests[0], MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret);

  std::vector<double> expected_averaging_lengths(avg_velocity_profile.size(), 0.0);
  for (size_t k = 0; k < bin_index.size(); ++k)
  {
#ifdef P4_TO_P8
    expected_averaging_lengths[bin_index[k]] += (xyz_max[averaging_direction]-xyz_min[averaging_direction])*((double) (brick->nxyztrees[transverse_direction]*(1<<data->max_lvl))/bin_index.size());
#else
    expected_averaging_lengths[bin_index[k]] += (brick->nxyztrees[transverse_direction]*(1<<data->max_lvl))/bin_index.size();
#endif
  }
  for (size_t profile_idx = 0; profile_idx < avg_velocity_profile.size(); ++profile_idx) {
    int rcv_rank = profile_idx%p4est_n->mpisize;
    if(p4est_n->mpirank == rcv_rank)
    {
      for (unsigned int k = 0; k < ndouble; ++k) {
        P4EST_ASSERT(fabs(avg_velocity_profile[profile_idx][ndouble+k] - expected_averaging_lengths[profile_idx]) < 10.0*EPS*MAX(avg_velocity_profile[profile_idx][ndouble+k], expected_averaging_lengths[profile_idx]));
        avg_velocity_profile[profile_idx][k] /= expected_averaging_lengths[profile_idx];
      }
    }
  }
}

void my_p4est_navier_stokes_t::coupled_problem_partial_destructor()
{
  PetscErrorCode ierr;
//  if(hodge != PETSC_NULL)     { ierr = VecDestroy(hodge);     CHKERRXX(ierr); }

  for(unsigned char dir = 0; dir < P4EST_DIM; ++dir)
  {
    if(dxyz_hodge[dir] != NULL) { ierr = VecDestroy(dxyz_hodge[dir]);                         CHKERRXX(ierr); }
    if(vstar[dir] != NULL)      { ierr = VecDestroy(vstar[dir]);                              CHKERRXX(ierr); }
    if(face_is_well_defined[dir] != NULL)
                                { ierr = VecDestroy(face_is_well_defined[dir]);               CHKERRXX(ierr); }
    for (unsigned char dd = 0; dd < P4EST_DIM; ++dd) {
      if(second_derivatives_vn_nodes[dd][dir] != NULL)
                                { ierr = VecDestroy(second_derivatives_vn_nodes[dd][dir]);    CHKERRXX(ierr); }
      if(second_derivatives_vnm1_nodes[dd][dir] != NULL)
                                { ierr = VecDestroy(second_derivatives_vnm1_nodes[dd][dir]);  CHKERRXX(ierr); }
    }
  }

  if(interp_phi != NULL)      delete interp_phi;
  if(interp_grad_phi != NULL) delete interp_grad_phi;

  delete faces_n;
  delete ngbd_c;

}
