#ifdef P4_TO_P8
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_trajectory_of_point.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_macros.h>
#include <p8est_extended.h>
#include <p8est_algorithms.h>
#include "my_p8est_navier_stokes_nodes.h"
#else
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_trajectory_of_point.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_macros.h>
#include <p4est_extended.h>
#include <p4est_algorithms.h>
#include "my_p4est_navier_stokes_nodes.h"
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

my_p4est_navier_stokes_nodes_t::splitting_criteria_vorticity_nodes_t::splitting_criteria_vorticity_nodes_t(int min_lvl, int max_lvl, double lip, double uniform_band, double threshold_vorticity, double max_L2_norm_u, double smoke_thresh, double threshold_norm_grad_u)
  : splitting_criteria_tag_t(min_lvl, max_lvl, lip)
{
  this->uniform_band          = uniform_band;
  this->threshold_vorticity   = threshold_vorticity;
  this->max_L2_norm_u         = max_L2_norm_u;
  this->smoke_thresh          = smoke_thresh;
  this->threshold_norm_grad_u = threshold_norm_grad_u;
}

void my_p4est_navier_stokes_nodes_t::splitting_criteria_vorticity_nodes_t::tag_quadrant(p4est_t *p4est, p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, p4est_nodes_t* nodes,
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

bool my_p4est_navier_stokes_nodes_t::splitting_criteria_vorticity_nodes_t::refine_and_coarsen(p4est_t* p4est, p4est_nodes_t* nodes, Vec phi, Vec vorticity, Vec smoke, Vec norm_grad_u)
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

  my_p4est_coarsen(p4est, P4EST_FALSE, splitting_criteria_vorticity_nodes_t::coarsen_fn, splitting_criteria_vorticity_nodes_t::init_fn);
  my_p4est_refine (p4est, P4EST_FALSE, splitting_criteria_vorticity_nodes_t::refine_fn,  splitting_criteria_vorticity_nodes_t::init_fn);

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

double my_p4est_navier_stokes_nodes_t::wall_bc_value_hodge_t::operator ()(DIM(double x, double y, double z)) const
{
  return _prnt->bc_pressure->wallValue(DIM(x, y, z)) * _prnt->dt_n / (_prnt->alpha()*_prnt->rho);
}

double my_p4est_navier_stokes_nodes_t::interface_bc_value_hodge_t::operator ()(DIM(double x, double y, double z)) const
{
  return _prnt->bc_pressure->interfaceValue(DIM(x, y, z)) * _prnt->dt_n / (_prnt->alpha()*_prnt->rho);
}

my_p4est_navier_stokes_nodes_t::my_p4est_navier_stokes_nodes_t(const mpi_environment_t& mpi, const char* path_to_save_state, double &simulation_time)
  : semi_lagrangian_backtrace_is_done(false), wall_bc_value_hodge(this), interface_bc_value_hodge(this)
{
  PetscErrorCode ierr;
  // we need to initialize those to NULL, otherwise the loader will freak out
  // no risk of memory leak here, this is a CONSTRUCTOR
  brick = NULL; conn = NULL;
  p4est_n = NULL; ghost_n = NULL; nodes_n = NULL;
  hierarchy_n = NULL; ngbd_n = NULL;
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


    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &vnp1_nodes[dir]); CHKERRXX(ierr);

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
  interp_phi = new my_p4est_interpolation_nodes_t(ngbd_n);
  interp_phi->set_input(phi, linear);

  interp_grad_phi = new my_p4est_interpolation_nodes_t(ngbd_n);
  ierr = VecCreateGhostNodesBlock(p4est_n, nodes_n, P4EST_DIM, &grad_phi);
  ngbd_n->first_derivatives_central(phi, grad_phi);
  interp_grad_phi->set_input(grad_phi, linear, P4EST_DIM);
}
my_p4est_navier_stokes_nodes_t::~my_p4est_navier_stokes_nodes_t()
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
    if(vnp1_nodes[dir] != NULL) { ierr = VecDestroy(vnp1_nodes[dir]);                CHKERRXX(ierr); }
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
  my_p4est_brick_destroy(conn, brick);
}
void my_p4est_navier_stokes_nodes_t::set_parameters(double mu_, double rho_, int sl_order_, double uniform_band_, double vorticity_threshold_split_cell_, double n_times_dt_, double norm_grad_u_threshold_split_cell_)
{
  P4EST_ASSERT(mu_ >= 0.0);                               this->mu = mu_;
  P4EST_ASSERT(rho_ > 0.0);                               this->rho = rho_;
  P4EST_ASSERT(sl_order_ == 1 || sl_order_ == 2);         this->sl_order = sl_order_;
  P4EST_ASSERT(uniform_band_ >= 0.0);                     this->uniform_band = uniform_band_;
  P4EST_ASSERT(n_times_dt_ > 0.0);                        this->n_times_dt = n_times_dt_;
}

void my_p4est_navier_stokes_nodes_t::set_smoke(Vec smoke, CF_DIM *bc_smoke, bool refine_with_smoke, double smoke_thresh)
{
  if(this->smoke != NULL && this->smoke != smoke){
    PetscErrorCode ierr= VecDestroy(this->smoke); CHKERRXX(ierr); }
  this->smoke = smoke;
  this->bc_smoke = bc_smoke;
  this->refine_with_smoke = refine_with_smoke;
  this->smoke_thresh = smoke_thresh;
}

void my_p4est_navier_stokes_nodes_t::set_phi(Vec phi)
{
  PetscErrorCode ierr;
  if(this->phi != NULL && this->phi != phi) {
    ierr = VecDestroy(this->phi); CHKERRXX(ierr); }
  this->phi = phi;
  ngbd_n->first_derivatives_central(phi, grad_phi);
  interp_phi->set_input(phi, linear);
  interp_grad_phi->set_input(grad_phi, linear, P4EST_DIM);
  return;
}

