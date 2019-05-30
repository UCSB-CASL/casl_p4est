#include "my_p4est_navier_stokes.h"

#ifdef P4_TO_P8
#include <src/my_p8est_poisson_cells.h>
#include <src/my_p8est_poisson_faces.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_level_set_cells.h>
#include <src/my_p8est_level_set_faces.h>
#include <src/my_p8est_trajectory_of_point.h>
#include <src/my_p8est_vtk.h>
#include <p8est_extended.h>
#include <p8est_algorithms.h>
#else
#include <src/my_p4est_poisson_cells.h>
#include <src/my_p4est_poisson_faces.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_level_set_cells.h>
#include <src/my_p4est_level_set_faces.h>
#include <src/my_p4est_trajectory_of_point.h>
#include <src/my_p4est_vtk.h>
#include <p4est_extended.h>
#include <p4est_algorithms.h>
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



my_p4est_navier_stokes_t::splitting_criteria_vorticity_t::splitting_criteria_vorticity_t(int min_lvl, int max_lvl, double lip, double uniform_band, double threshold, double max_L2_norm_u, double smoke_thresh)
  : splitting_criteria_tag_t(min_lvl, max_lvl, lip)
{
  this->uniform_band  = uniform_band;
  this->threshold     = threshold;
  this->max_L2_norm_u = max_L2_norm_u;
  this->smoke_thresh  = smoke_thresh;
}


void my_p4est_navier_stokes_t::splitting_criteria_vorticity_t::tag_quadrant(p4est_t *p4est, p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, p4est_nodes_t* nodes,
                                                                            const double* tree_dimensions,
                                                                            const double *phi_p, const double *vorticity_p, const double *smoke_p)
{
  p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
  p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);

  if (quad->level < min_lvl)
    quad->p.user_int = REFINE_QUADRANT;

  else if (quad->level > max_lvl)
    quad->p.user_int = COARSEN_QUADRANT;

  else
  {
#ifdef P4_TO_P8
    const double quad_diag          = sqrt(SQR(tree_dimensions[0]) + SQR(tree_dimensions[1]) + SQR(tree_dimensions[2]))*(((double) P4EST_QUADRANT_LEN(quad->level))/((double) P4EST_ROOT_LEN));
    const double quad_dxyz_max      = MAX(tree_dimensions[0],tree_dimensions[1], tree_dimensions[2])*(((double) P4EST_QUADRANT_LEN(quad->level))/((double) P4EST_ROOT_LEN));
    const double smallest_dxyz_max  = MAX(tree_dimensions[0],tree_dimensions[1], tree_dimensions[2])*(((double) P4EST_QUADRANT_LEN((int8_t) max_lvl))/((double) P4EST_ROOT_LEN));
#else
    const double quad_diag          = sqrt(SQR(tree_dimensions[0]) + SQR(tree_dimensions[1]))*(((double) P4EST_QUADRANT_LEN(quad->level))/((double) P4EST_ROOT_LEN));
    const double quad_dxyz_max      = MAX(tree_dimensions[0],tree_dimensions[1])*(((double) P4EST_QUADRANT_LEN(quad->level))/((double) P4EST_ROOT_LEN));
    const double smallest_dxyz_max  = MAX(tree_dimensions[0],tree_dimensions[1])*(((double) P4EST_QUADRANT_LEN((int8_t) max_lvl))/((double) P4EST_ROOT_LEN));
#endif

//    double xyz_quad[P4EST_DIM];
//    xyz_quad[0] = p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx] + 0] + tree_dimensions[0]*quad_x_fr_i(quad);
//    xyz_quad[1] = p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx] + 1] + tree_dimensions[1]*quad_y_fr_j(quad);
//    bool print = (fabs(xyz_quad[0]-9.0) < 0.5*smallest_dxyz_max) && (fabs(xyz_quad[1]-0.25) < 0.5*smallest_dxyz_max);

    bool coarsen = (quad->level > min_lvl);
    if(coarsen)
    {
      bool all_pos  = true;
      bool cor_vort = true;
      bool cor_band = true;
      bool cor_intf = true;
      bool cor_smok = true;
      p4est_locidx_t node_idx;

      for (unsigned short k = 0; k < P4EST_CHILDREN; ++k) {
        node_idx    = nodes->local_nodes[P4EST_CHILDREN*quad_idx+k];
        cor_vort    = cor_vort && fabs(vorticity_p[node_idx])*2.0*quad_dxyz_max/max_L2_norm_u<threshold;
        cor_band    = cor_band && fabs(phi_p[node_idx])>uniform_band*smallest_dxyz_max;
        cor_intf    = cor_intf && fabs(phi_p[node_idx])>=lip*2.0*quad_diag;
        if(smoke_p!=NULL)
          cor_smok  = cor_smok && smoke_p[node_idx]<smoke_thresh;
        all_pos = all_pos && phi_p[node_idx]>MAX(2.0, uniform_band)*smallest_dxyz_max;
        // [RAPHAEL:] modified to enforce two layers of finest level in positive domain as well (better extrapolation etc.), also required for Neumann BC in the face-solvers
        coarsen = (cor_vort && cor_band && cor_intf && cor_smok) || all_pos;
        //    coarsen = ((cor_vort && cor_band && cor_smok) || all_pos) && cor_intf;
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
      p4est_locidx_t node_idx;
      bool node_found;
      // check possibly finer points
      const p4est_qcoord_t mid_qh = P4EST_QUADRANT_LEN (quad->level+1);
#ifdef P4_TO_P8
      for(unsigned short k=0; k<3; ++k)
#endif
        for(unsigned short j=0; j<3; ++j)
          for(unsigned short i=0; i<3; ++i)
          {
#ifdef P4_TO_P8
            if((k==1)&&(j==1)&&(i==1))
#else
            if((j==1)&&(i==1))
#endif
              continue;
#ifdef P4_TO_P8
            if(((k==0)||(k==2)) && ((j==0)||(j==2)) && ((i==0)||(i==2)))
#else
            if(((j==0)||(j==2)) && ((i==0)||(i==2)))
#endif
            {
              node_found=true;
#ifdef P4_TO_P8
              node_idx = nodes->local_nodes[P4EST_CHILDREN*quad_idx+4*(k/2)+2*(j/2)+(i/2)];
#else
              node_idx = nodes->local_nodes[P4EST_CHILDREN*quad_idx+2*(j/2)+(i/2)];
#endif
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
              ref_vort        = ref_vort || fabs(vorticity_p[node_idx])*quad_dxyz_max/max_L2_norm_u>threshold;
              ref_band        = ref_band || fabs(phi_p[node_idx])<uniform_band*smallest_dxyz_max;
              ref_intf        = ref_intf || fabs(phi_p[node_idx])<=lip*quad_diag;
              if(smoke_p!=NULL)
                ref_smok      = ref_smok || smoke_p[node_idx]>=smoke_thresh;
              is_neg = is_neg || phi_p[node_idx]< MAX(2.0, uniform_band)*smallest_dxyz_max; // [RAPHAEL:] same comment as before

              refine = is_neg && (ref_vort || ref_band || ref_intf || ref_smok);
              //    refine = ((ref_vort || ref_band || ref_smok) && is_neg) || ref_intf;
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


bool my_p4est_navier_stokes_t::splitting_criteria_vorticity_t::refine_and_coarsen(p4est_t* p4est, p4est_nodes_t* nodes, Vec phi, Vec vorticity, Vec smoke)
{
  const double *phi_p, *vorticity_p, *smoke_p;
  PetscErrorCode ierr;
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(vorticity, &vorticity_p); CHKERRXX(ierr);
  if(smoke != NULL){
    ierr = VecGetArrayRead(smoke, &smoke_p); CHKERRXX(ierr); }
  else
    smoke_p = NULL;

  double tree_dimensions[P4EST_DIM];
  p4est_locidx_t* t2v = p4est->connectivity->tree_to_vertex;
  double* v2c = p4est->connectivity->vertices;
  /* tag the quadrants that need to be refined or coarsened */
  for (p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx) {
    p4est_tree_t* tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for (short dir = 0; dir < P4EST_DIM; ++dir)
      tree_dimensions[dir] = v2c[3*t2v[P4EST_CHILDREN*tree_idx + P4EST_CHILDREN -1] + dir] - v2c[3*t2v[P4EST_CHILDREN*tree_idx + 0] + dir];
    for (size_t q = 0; q <tree->quadrants.elem_count; ++q) {
      p4est_locidx_t quad_idx  = q + tree->quadrants_offset;
      tag_quadrant(p4est, quad_idx, tree_idx, nodes, tree_dimensions, phi_p, vorticity_p, smoke_p);
    }
  }

  my_p4est_coarsen(p4est, P4EST_FALSE, splitting_criteria_vorticity_t::coarsen_fn, splitting_criteria_vorticity_t::init_fn);
  my_p4est_refine (p4est, P4EST_FALSE, splitting_criteria_vorticity_t::refine_fn,  splitting_criteria_vorticity_t::init_fn);

  int is_grid_changed = false;
  for (p4est_topidx_t it = p4est->first_local_tree; it <= p4est->last_local_tree; ++it) {
    p4est_tree_t* tree = (p4est_tree_t*)sc_array_index(p4est->trees, it);
    for (size_t q = 0; q <tree->quadrants.elem_count; ++q) {
      p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
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

  return is_grid_changed;
}


#ifdef P4_TO_P8
double my_p4est_navier_stokes_t::wall_bc_value_hodge_t::operator ()(double x, double y, double z) const
#else
double my_p4est_navier_stokes_t::wall_bc_value_hodge_t::operator ()(double x, double y) const
#endif
{
  double alpha = _prnt->sl_order==1 ? 1 : (2*_prnt->dt_n + _prnt->dt_nm1)/(_prnt->dt_n + _prnt->dt_nm1);
#ifdef P4_TO_P8
  return _prnt->bc_pressure->wallValue(x,y,z) * _prnt->dt_n / (alpha*_prnt->rho);
#else
  return _prnt->bc_pressure->wallValue(x,y)   * _prnt->dt_n / (alpha*_prnt->rho);
#endif
}




#ifdef P4_TO_P8
double my_p4est_navier_stokes_t::interface_bc_value_hodge_t::operator ()(double x, double y, double z) const
#else
double my_p4est_navier_stokes_t::interface_bc_value_hodge_t::operator ()(double x, double y) const
#endif
{
  double alpha = _prnt->sl_order==1 ? 1 : (2*_prnt->dt_n + _prnt->dt_nm1)/(_prnt->dt_n + _prnt->dt_nm1);
#ifdef P4_TO_P8
  return _prnt->bc_pressure->interfaceValue(x,y,z) * _prnt->dt_n / (alpha*_prnt->rho);
#else
  return _prnt->bc_pressure->interfaceValue(x,y)   * _prnt->dt_n / (alpha*_prnt->rho);
#endif
}


my_p4est_navier_stokes_t::my_p4est_navier_stokes_t(my_p4est_node_neighbors_t *ngbd_nm1, my_p4est_node_neighbors_t *ngbd_n, my_p4est_faces_t *faces)
  : brick(ngbd_n->myb), conn(ngbd_n->p4est->connectivity),
    p4est_nm1(ngbd_nm1->p4est), ghost_nm1(ngbd_nm1->ghost), nodes_nm1(ngbd_nm1->nodes),
    hierarchy_nm1(ngbd_nm1->hierarchy), ngbd_nm1(ngbd_nm1),
    p4est_n(ngbd_n->p4est), ghost_n(ngbd_n->ghost), nodes_n(ngbd_n->nodes),
    hierarchy_n(ngbd_n->hierarchy), ngbd_n(ngbd_n),
    ngbd_c(faces->ngbd_c), faces_n(faces), semi_lagrangian_backtrace_is_done(false), interpolators_from_face_to_nodes_are_set(false),
    wall_bc_value_hodge(this), interface_bc_value_hodge(this)
{
  PetscErrorCode ierr;

  mu = 1;
  rho = 1;
  uniform_band = 0;
  threshold_split_cell = 0.04;
  n_times_dt = 5;
  dt_updated = false;
  max_L2_norm_u = 0;

  sl_order = 1;

  double *v2c = p4est_n->connectivity->vertices;
  p4est_topidx_t *t2v = p4est_n->connectivity->tree_to_vertex;
  p4est_topidx_t first_tree = 0, last_tree = p4est_n->trees->elem_count-1;
  p4est_topidx_t first_vertex = 0, last_vertex = P4EST_CHILDREN - 1;

  splitting_criteria_t *data = (splitting_criteria_t*)p4est_n->user_pointer;

  for (int dir=0; dir<P4EST_DIM; dir++)
  {
    xyz_min[dir] = v2c[3*t2v[P4EST_CHILDREN*first_tree + first_vertex] + dir];
    xyz_max[dir] = v2c[3*t2v[P4EST_CHILDREN*last_tree  + last_vertex ] + dir];

    double xyz_tmp = v2c[3*t2v[P4EST_CHILDREN*first_tree + last_vertex] + dir];
    dxyz_min[dir] = (xyz_tmp-xyz_min[dir]) / (1<<data->max_lvl);
    convert_to_xyz[dir] = xyz_tmp-xyz_min[dir];
    // initialize this one
    interpolator_from_face_to_nodes[dir].resize(0);
  }

#ifdef P4_TO_P8
  dt_nm1 = dt_n = .5 * MIN(dxyz_min[0], dxyz_min[1], dxyz_min[2]);
#else
  dt_nm1 = dt_n = .5 * MIN(dxyz_min[0], dxyz_min[1]);
#endif

  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &phi); CHKERRXX(ierr);
  Vec vec_loc;
  ierr = VecGhostGetLocalForm(phi, &vec_loc); CHKERRXX(ierr);
  ierr = VecSet(vec_loc, -1); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(phi, &vec_loc); CHKERRXX(ierr);

  ierr = VecCreateGhostCells(p4est_n, ghost_n, &hodge); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(hodge, &vec_loc); CHKERRXX(ierr);
  ierr = VecSet(vec_loc, 0); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(hodge, &vec_loc); CHKERRXX(ierr);

  ierr = VecDuplicate(hodge, &pressure); CHKERRXX(ierr);

  bc_v = NULL;
  bc_pressure = NULL;

  vorticity = NULL;
  smoke = NULL;
  bc_smoke = NULL;

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    external_forces[dir] = NULL;
    vstar[dir] = NULL;
    vnp1[dir] = NULL;

    vnm1_nodes[dir] = NULL;
    vn_nodes  [dir] = NULL;
    vnp1_nodes[dir] = NULL;
    for (short dd = 0; dd < P4EST_DIM; ++dd) {
      second_derivatives_vn_nodes[dd][dir]    = NULL;
      second_derivatives_vnm1_nodes[dd][dir]  = NULL;
    }

    ierr = VecCreateGhostFaces(p4est_n, faces_n, &face_is_well_defined[dir], dir); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(face_is_well_defined[dir], &vec_loc); CHKERRXX(ierr);
    ierr = VecSet(vec_loc, 1); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(face_is_well_defined[dir], &vec_loc); CHKERRXX(ierr);

    ierr = VecDuplicate(face_is_well_defined[dir], &dxyz_hodge[dir]); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(dxyz_hodge[dir], &vec_loc); CHKERRXX(ierr);
    ierr = VecSet(vec_loc, 0); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(dxyz_hodge[dir], &vec_loc); CHKERRXX(ierr);
  }

  ngbd_n->init_neighbors();
  ngbd_nm1->init_neighbors();

  interp_phi = new my_p4est_interpolation_nodes_t(ngbd_n);
  interp_phi->set_input(phi, linear);
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
  for (short kk = 0; kk < P4EST_DIM; ++kk) {
    vnm1_nodes[kk] = NULL;
    vn_nodes[kk] = NULL;
    for (short dd = 0; dd < P4EST_DIM; ++dd) {
      second_derivatives_vn_nodes[dd][kk]    = NULL;
      second_derivatives_vnm1_nodes[dd][kk]  = NULL;
    }
  }
  dt_updated = false;
  load_state(mpi, path_to_save_state, simulation_time);
  ierr = VecDuplicate(hodge, &pressure); CHKERRXX(ierr);

  bc_v = NULL;
  bc_pressure = NULL;

  vorticity = NULL;
  bc_smoke = NULL;

  Vec vec_loc;
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    external_forces[dir] = NULL;
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

    for (short dd = 0; dd < P4EST_DIM; ++dd) {
      if(second_derivatives_vn_nodes[dd][dir]!= NULL)
      {
        ierr = VecDestroy(second_derivatives_vn_nodes[dd][dir]); CHKERRXX(ierr);
      }
      if(second_derivatives_vnm1_nodes[dd][dir]!= NULL)
      {
        ierr = VecDestroy(second_derivatives_vnm1_nodes[dd][dir]); CHKERRXX(ierr);
      }
      ierr = VecCreateGhostNodes(p4est_n, nodes_n, &second_derivatives_vn_nodes[dd][dir]); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est_nm1, nodes_nm1, &second_derivatives_vnm1_nodes[dd][dir]); CHKERRXX(ierr);
    }
  }
#ifdef P4_TO_P8
  ngbd_n->second_derivatives_central(vn_nodes, second_derivatives_vn_nodes[0], second_derivatives_vn_nodes[1], second_derivatives_vn_nodes[2], P4EST_DIM);
  ngbd_nm1->second_derivatives_central(vnm1_nodes, second_derivatives_vnm1_nodes[0], second_derivatives_vnm1_nodes[1], second_derivatives_vnm1_nodes[2], P4EST_DIM);
#else
  ngbd_n->second_derivatives_central(vn_nodes, second_derivatives_vn_nodes[0], second_derivatives_vn_nodes[1], P4EST_DIM);
  ngbd_nm1->second_derivatives_central(vnm1_nodes, second_derivatives_vnm1_nodes[0], second_derivatives_vnm1_nodes[1], P4EST_DIM);
#endif

  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &vorticity); CHKERRXX(ierr);

  interp_phi = new my_p4est_interpolation_nodes_t(ngbd_n);
  interp_phi->set_input(phi, linear);
}


my_p4est_navier_stokes_t::~my_p4est_navier_stokes_t()
{
  PetscErrorCode ierr;
  if(phi!=NULL)       { ierr = VecDestroy(phi);       CHKERRXX(ierr); }
  if(hodge!=NULL)     { ierr = VecDestroy(hodge);     CHKERRXX(ierr); }
  if(pressure!=NULL)  { ierr = VecDestroy(pressure);  CHKERRXX(ierr); }
  if(vorticity!=NULL) { ierr = VecDestroy(vorticity); CHKERRXX(ierr); }
  if(smoke!=NULL)     { ierr = VecDestroy(smoke);     CHKERRXX(ierr); }

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    if(dxyz_hodge[dir]!=NULL) { ierr = VecDestroy(dxyz_hodge[dir]);                         CHKERRXX(ierr); }
    if(vstar[dir]!=NULL)      { ierr = VecDestroy(vstar[dir]);                              CHKERRXX(ierr); }
    if(vnp1[dir]!=NULL)       { ierr = VecDestroy(vnp1[dir]);                               CHKERRXX(ierr); }
    if(vnm1_nodes[dir]!=NULL) { ierr = VecDestroy(vnm1_nodes[dir]);                         CHKERRXX(ierr); }
    if(vn_nodes[dir]!=NULL)   { ierr = VecDestroy(vn_nodes[dir]);                           CHKERRXX(ierr); }
    if(vnp1_nodes[dir]!=NULL) { ierr = VecDestroy(vnp1_nodes[dir]);                         CHKERRXX(ierr); }
    if(face_is_well_defined[dir]!=NULL)
                              { ierr = VecDestroy(face_is_well_defined[dir]);               CHKERRXX(ierr); }
    for (short dd = 0; dd < P4EST_DIM; ++dd) {
      if(second_derivatives_vn_nodes[dd][dir]!=NULL)
                              { ierr = VecDestroy(second_derivatives_vn_nodes[dd][dir]);    CHKERRXX(ierr); }
      if(second_derivatives_vnm1_nodes[dd][dir]!=NULL)
                              { ierr = VecDestroy(second_derivatives_vnm1_nodes[dd][dir]);  CHKERRXX(ierr); }
    }
  }

  if(interp_phi!=NULL) delete interp_phi;

  if(ngbd_nm1!=ngbd_n)
    delete ngbd_nm1;
  delete ngbd_n;
  if(hierarchy_nm1!=hierarchy_n)
    delete hierarchy_nm1;
  delete hierarchy_n;
  if(nodes_nm1!=nodes_n)
    p4est_nodes_destroy(nodes_nm1);
  p4est_nodes_destroy(nodes_n);
  if(p4est_nm1!=p4est_n)
    p4est_destroy(p4est_nm1);
  p4est_destroy(p4est_n);
  if(ghost_nm1!=ghost_n)
    p4est_ghost_destroy(ghost_nm1);
  p4est_ghost_destroy(ghost_n);

  delete faces_n;
  delete ngbd_c;

  my_p4est_brick_destroy(conn, brick);
}


void my_p4est_navier_stokes_t::set_parameters(double mu, double rho, int sl_order, double uniform_band, double threshold_split_cell, double n_times_dt)
{
  this->mu = mu;
  this->rho = rho;
  this->sl_order = sl_order;
  this->uniform_band = uniform_band;
  this->threshold_split_cell = threshold_split_cell;
  this->n_times_dt = n_times_dt;
}

#ifdef P4_TO_P8
void my_p4est_navier_stokes_t::set_smoke(Vec smoke, CF_3 *bc_smoke, bool refine_with_smoke, double smoke_thresh)
#else
void my_p4est_navier_stokes_t::set_smoke(Vec smoke, CF_2 *bc_smoke, bool refine_with_smoke, double smoke_thresh)
#endif
{
  if(this->smoke!=NULL){
    PetscErrorCode ierr= VecDestroy(this->smoke); CHKERRXX(ierr); }
  this->smoke = smoke;
  this->bc_smoke = bc_smoke;
  this->refine_with_smoke = refine_with_smoke;
  this->smoke_thresh = smoke_thresh;
}


void my_p4est_navier_stokes_t::set_phi(Vec phi)
{
  PetscErrorCode ierr;
  if(this->phi!=NULL) { ierr = VecDestroy(this->phi); CHKERRXX(ierr); }
  this->phi = phi;

  if(bc_v!=NULL)
  {
    for(int dir=0; dir<P4EST_DIM; ++dir)
      check_if_faces_are_well_defined(p4est_n, ngbd_n, faces_n, dir, phi, bc_v[dir].interfaceType(), face_is_well_defined[dir]);
  }

  interp_phi->set_input(phi, linear);
}


#ifdef P4_TO_P8
void my_p4est_navier_stokes_t::set_external_forces(CF_3 **external_forces)
#else
void my_p4est_navier_stokes_t::set_external_forces(CF_2 **external_forces)
#endif
{
  for(int dir=0; dir<P4EST_DIM; ++dir)
    this->external_forces[dir] = external_forces[dir];
}


#ifdef P4_TO_P8
void my_p4est_navier_stokes_t::set_bc(BoundaryConditions3D *bc_v, BoundaryConditions3D *bc_p)
#else
void my_p4est_navier_stokes_t::set_bc(BoundaryConditions2D *bc_v, BoundaryConditions2D *bc_p)
#endif
{
  this->bc_v = bc_v;
  this->bc_pressure = bc_p;

  bc_hodge.setWallTypes(bc_pressure->getWallType());
  bc_hodge.setWallValues(wall_bc_value_hodge);
  bc_hodge.setInterfaceType(bc_pressure->interfaceType());
  bc_hodge.setInterfaceValue(interface_bc_value_hodge);

  if(phi!=NULL)
  {
    for(int dir=0; dir<P4EST_DIM; ++dir)
      check_if_faces_are_well_defined(p4est_n, ngbd_n, faces_n, dir, phi, bc_v[dir].interfaceType(), face_is_well_defined[dir]);
  }
}


void my_p4est_navier_stokes_t::set_velocities(Vec *vnm1_nodes, Vec *vn_nodes)
{
  PetscErrorCode ierr;

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    if(this->vn_nodes[dir]!=NULL){
      ierr = VecDestroy(this->vn_nodes[dir]); CHKERRXX(ierr); }
    this->vn_nodes[dir]   = vn_nodes[dir];
    if(this->vnm1_nodes[dir]!=NULL){
      ierr = VecDestroy(this->vnm1_nodes[dir]); CHKERRXX(ierr); }
    this->vnm1_nodes[dir] = vnm1_nodes[dir];
    if(this->vnp1_nodes[dir]!=NULL){
      ierr = VecDestroy(this->vnp1_nodes[dir]); CHKERRXX(ierr); }
    ierr = VecDuplicate(vn_nodes[dir], &vnp1_nodes [dir]); CHKERRXX(ierr);
    if(this->vstar[dir]!=NULL){
      ierr = VecDestroy(this->vstar[dir]); CHKERRXX(ierr); }
    ierr = VecDuplicate(face_is_well_defined[dir], &vstar[dir]); CHKERRXX(ierr);
    if(this->vnp1[dir]!=NULL){
      ierr = VecDestroy(this->vnp1[dir]); CHKERRXX(ierr); }
    ierr = VecDuplicate(face_is_well_defined[dir], &vnp1[dir]); CHKERRXX(ierr);

    for (short dd = 0; dd < P4EST_DIM; ++dd) {
      if(second_derivatives_vn_nodes[dd][dir]!= NULL)
      {
        ierr = VecDestroy(second_derivatives_vn_nodes[dd][dir]); CHKERRXX(ierr);
      }
      if(second_derivatives_vnm1_nodes[dd][dir]!= NULL)
      {
        ierr = VecDestroy(second_derivatives_vnm1_nodes[dd][dir]); CHKERRXX(ierr);
      }
      ierr = VecCreateGhostNodes(p4est_n, nodes_n, &second_derivatives_vn_nodes[dd][dir]); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est_nm1, nodes_nm1, &second_derivatives_vnm1_nodes[dd][dir]); CHKERRXX(ierr);
    }
  }
#ifdef P4_TO_P8
  ngbd_n->second_derivatives_central(vn_nodes, second_derivatives_vn_nodes[0], second_derivatives_vn_nodes[1], second_derivatives_vn_nodes[2], P4EST_DIM);
  ngbd_nm1->second_derivatives_central(vnm1_nodes, second_derivatives_vnm1_nodes[0], second_derivatives_vnm1_nodes[1], second_derivatives_vnm1_nodes[2], P4EST_DIM);
#else
  ngbd_n->second_derivatives_central(vn_nodes, second_derivatives_vn_nodes[0], second_derivatives_vn_nodes[1], P4EST_DIM);
  ngbd_nm1->second_derivatives_central(vnm1_nodes, second_derivatives_vnm1_nodes[0], second_derivatives_vnm1_nodes[1], P4EST_DIM);
#endif

  if(this->vorticity!=NULL){
    ierr = VecDestroy(this->vorticity); CHKERRXX(ierr); }
  ierr = VecDuplicate(phi, &vorticity); CHKERRXX(ierr);
}

#ifdef P4_TO_P8
void my_p4est_navier_stokes_t::set_velocities(CF_3 **vnm1, CF_3 **vn)
#else
void my_p4est_navier_stokes_t::set_velocities(CF_2 **vnm1, CF_2 **vn)
#endif
{
  PetscErrorCode ierr;

  double *v_p;
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    if(this->vnm1_nodes[dir]!=NULL){
      ierr = VecDestroy(this->vnm1_nodes[dir]); CHKERRXX(ierr); }
    ierr = VecCreateGhostNodes(p4est_nm1, nodes_nm1, &vnm1_nodes[dir]); CHKERRXX(ierr);
    ierr = VecGetArray(vnm1_nodes[dir], &v_p); CHKERRXX(ierr);
    for(size_t n=0; n<nodes_nm1->indep_nodes.elem_count; ++n)
    {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, p4est_nm1, nodes_nm1, xyz);
#ifdef P4_TO_P8
      v_p[n] = (*vnm1[dir])(xyz[0], xyz[1], xyz[2]);
#else
      v_p[n] = (*vnm1[dir])(xyz[0], xyz[1]);
#endif
    }
    ierr = VecRestoreArray(vnm1_nodes[dir], &v_p); CHKERRXX(ierr);

    if(this->vn_nodes[dir]!=NULL){
      ierr = VecDestroy(this->vn_nodes[dir]); CHKERRXX(ierr); }
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &vn_nodes[dir]); CHKERRXX(ierr);
    ierr = VecGetArray(vn_nodes[dir], &v_p); CHKERRXX(ierr);
    for(size_t n=0; n<nodes_n->indep_nodes.elem_count; ++n)
    {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, p4est_n, nodes_n, xyz);
#ifdef P4_TO_P8
      v_p[n] = (*vn[dir])(xyz[0], xyz[1], xyz[2]);
#else
      v_p[n] = (*vn[dir])(xyz[0], xyz[1]);
#endif
    }
    ierr = VecRestoreArray(vn_nodes[dir], &v_p); CHKERRXX(ierr);

    if(this->vnp1_nodes[dir]!=NULL){
      ierr = VecDestroy(this->vnp1_nodes[dir]); CHKERRXX(ierr); }
    ierr = VecDuplicate(vn_nodes[dir], &vnp1_nodes [dir]); CHKERRXX(ierr);
    if(this->vstar[dir]!=NULL){
      ierr = VecDestroy(this->vstar[dir]); CHKERRXX(ierr); }
    ierr = VecDuplicate(face_is_well_defined[dir], &vstar[dir]); CHKERRXX(ierr);
    if(this->vnp1[dir]!=NULL){
      ierr = VecDestroy(this->vnp1[dir]); CHKERRXX(ierr); }
    ierr = VecDuplicate(face_is_well_defined[dir], &vnp1[dir]); CHKERRXX(ierr);

    for (short dd = 0; dd < P4EST_DIM; ++dd) {
      if(second_derivatives_vn_nodes[dd][dir]!= NULL)
      {
        ierr = VecDestroy(second_derivatives_vn_nodes[dd][dir]); CHKERRXX(ierr);
      }
      if(second_derivatives_vnm1_nodes[dd][dir]!= NULL)
      {
        ierr = VecDestroy(second_derivatives_vnm1_nodes[dd][dir]); CHKERRXX(ierr);
      }
      ierr = VecCreateGhostNodes(p4est_n, nodes_n, &second_derivatives_vn_nodes[dd][dir]); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est_nm1, nodes_nm1, &second_derivatives_vnm1_nodes[dd][dir]); CHKERRXX(ierr);
    }
  }
#ifdef P4_TO_P8
  ngbd_n->second_derivatives_central(vn_nodes, second_derivatives_vn_nodes[0], second_derivatives_vn_nodes[1], second_derivatives_vn_nodes[2], P4EST_DIM);
  ngbd_nm1->second_derivatives_central(vnm1_nodes, second_derivatives_vnm1_nodes[0], second_derivatives_vnm1_nodes[1], second_derivatives_vnm1_nodes[2], P4EST_DIM);
#else
  ngbd_n->second_derivatives_central(vn_nodes, second_derivatives_vn_nodes[0], second_derivatives_vn_nodes[1], P4EST_DIM);
  ngbd_nm1->second_derivatives_central(vnm1_nodes, second_derivatives_vnm1_nodes[0], second_derivatives_vnm1_nodes[1], P4EST_DIM);
#endif

  if(this->vorticity!=NULL){
    ierr = VecDestroy(this->vorticity); CHKERRXX(ierr); }
  ierr = VecDuplicate(phi, &vorticity); CHKERRXX(ierr);
}

void my_p4est_navier_stokes_t::set_vstar(Vec *vstar)
{
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    if(this->vstar[dir]!=NULL){
      PetscErrorCode ierr = VecDestroy(this->vstar[dir]); CHKERRXX(ierr); }
    this->vstar[dir] = vstar[dir];
  }
}

void my_p4est_navier_stokes_t::set_hodge(Vec hodge)
{
  if(this->hodge!=NULL){
    PetscErrorCode ierr = VecDestroy(this->hodge); CHKERRXX(ierr); }
  this->hodge = hodge;
}

void my_p4est_navier_stokes_t::compute_max_L2_norm_u()
{
  PetscErrorCode ierr;
  max_L2_norm_u = 0;

  const double *phi_p;
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);

#ifdef P4_TO_P8
  double dxmax = MAX(dxyz_min[0], dxyz_min[1], dxyz_min[2]);
#else
  double dxmax = MAX(dxyz_min[0], dxyz_min[1]);
#endif

  const double *v_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir) { ierr = VecGetArrayRead(vnp1_nodes[dir], &v_p[dir]); CHKERRXX(ierr); }

  for(p4est_locidx_t n=0; n<nodes_n->num_owned_indeps; ++n)
  {
    if(phi_p[n]<dxmax)
#ifdef P4_TO_P8
      max_L2_norm_u = MAX(max_L2_norm_u, sqrt(SQR(v_p[0][n]) + SQR(v_p[1][n]) + SQR(v_p[2][n])));
#else
      max_L2_norm_u = MAX(max_L2_norm_u, sqrt(SQR(v_p[0][n]) + SQR(v_p[1][n])));
#endif
  }

  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
  for(int dir=0; dir<P4EST_DIM; ++dir) { ierr = VecRestoreArrayRead(vnp1_nodes[dir], &v_p[dir]); CHKERRXX(ierr); }

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &max_L2_norm_u, 1, MPI_DOUBLE, MPI_MAX, p4est_n->mpicomm); SC_CHECK_MPI(mpiret);
}

void my_p4est_navier_stokes_t::compute_vorticity()
{
  PetscErrorCode ierr;

  quad_neighbor_nodes_of_node_t qnnn;

  const double *vnp1_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; dir++) { ierr = VecGetArrayRead(vnp1_nodes[dir], &vnp1_p[dir]); CHKERRXX(ierr); }

  double *vorticity_p;
  ierr = VecGetArray(vorticity, &vorticity_p); CHKERRXX(ierr);

  for(size_t i=0; i<ngbd_n->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n->get_layer_node(i);
    ngbd_n->get_neighbors(n, qnnn);
#ifdef P4_TO_P8
    double vx = qnnn.dy_central(vnp1_p[2]) - qnnn.dz_central(vnp1_p[1]);
    double vy = qnnn.dz_central(vnp1_p[0]) - qnnn.dx_central(vnp1_p[2]);
    double vz = qnnn.dx_central(vnp1_p[1]) - qnnn.dy_central(vnp1_p[0]);
    vorticity_p[n] = sqrt(vx*vx + vy*vy + vz*vz);
#else
    vorticity_p[n] = qnnn.dx_central(vnp1_p[1]) - qnnn.dy_central(vnp1_p[0]);
#endif
  }

  ierr = VecGhostUpdateBegin(vorticity, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  for(size_t i=0; i<ngbd_n->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n->get_local_node(i);
    ngbd_n->get_neighbors(n, qnnn);
#ifdef P4_TO_P8
    double vx = qnnn.dy_central(vnp1_p[2]) - qnnn.dz_central(vnp1_p[1]);
    double vy = qnnn.dz_central(vnp1_p[0]) - qnnn.dx_central(vnp1_p[2]);
    double vz = qnnn.dx_central(vnp1_p[1]) - qnnn.dy_central(vnp1_p[0]);
    vorticity_p[n] = sqrt(vx*vx + vy*vy + vz*vz);
#else
    vorticity_p[n] = qnnn.dx_central(vnp1_p[1]) - qnnn.dy_central(vnp1_p[0]);
#endif
  }

  ierr = VecGhostUpdateEnd(vorticity, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  for(int dir=0; dir<P4EST_DIM; dir++) { ierr = VecRestoreArrayRead(vnp1_nodes[dir], &vnp1_p[dir]); CHKERRXX(ierr); }
}

void my_p4est_navier_stokes_t::compute_Q_and_lambda_2_value(Vec& Q_value_nodes, Vec& lambda_2_nodes, const double U_scaling, const double x_scaling) const
{
  PetscErrorCode ierr;

  quad_neighbor_nodes_of_node_t qnnn;

  const double *vnp1_p[P4EST_DIM];
  for(unsigned short dir=0; dir<P4EST_DIM; dir++) { ierr = VecGetArrayRead(vnp1_nodes[dir], &vnp1_p[dir]); CHKERRXX(ierr); }

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


  for(unsigned short dir=0; dir<P4EST_DIM; dir++) { ierr = VecRestoreArrayRead(vnp1_nodes[dir], &vnp1_p[dir]); CHKERRXX(ierr); }
  ierr = VecRestoreArray(Q_value_nodes,   &Q_value_nodes_p);  CHKERRXX(ierr);
  ierr = VecRestoreArray(lambda_2_nodes,  &lambda_2_nodes_p); CHKERRXX(ierr);

}


double my_p4est_navier_stokes_t::compute_dxyz_hodge(p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, int dir)
{
  PetscErrorCode ierr;

  p4est_quadrant_t *quad;
  if(quad_idx<p4est_n->local_num_quadrants)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est_n->trees, tree_idx);
    quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);
  }
  else
    quad = (p4est_quadrant_t*)sc_array_index(&ghost_n->ghosts, quad_idx-p4est_n->local_num_quadrants);

  const double *hodge_p;
  ierr = VecGetArrayRead(hodge, &hodge_p); CHKERRXX(ierr);

  if(is_quad_Wall(p4est_n, tree_idx, quad, dir))
  {
    double x = quad_x_fr_q(quad_idx, tree_idx, p4est_n, ghost_n);
    double y = quad_y_fr_q(quad_idx, tree_idx, p4est_n, ghost_n);
#ifdef P4_TO_P8
    double z = quad_z_fr_q(quad_idx, tree_idx, p4est_n, ghost_n);
#endif

    double dmin = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
    double dx = convert_to_xyz[0] * dmin;
    double dy = convert_to_xyz[1] * dmin;
#ifdef P4_TO_P8
    double dz = convert_to_xyz[2] * dmin;
#endif

    double hodge_q = hodge_p[quad_idx];
    ierr = VecRestoreArrayRead(hodge, &hodge_p); CHKERRXX(ierr);

    switch(dir)
    {
#ifdef P4_TO_P8
    case dir::f_m00:
      if(bc_hodge.wallType(x-dx/2,y,z)==NEUMANN) return -bc_hodge.wallValue(x-dx/2,y,z);
      else                                       return (hodge_q - bc_hodge.wallValue(x-dx/2,y,z)) * 2 / dx;
    case dir::f_p00:
      if(bc_hodge.wallType(x+dx/2,y,z)==NEUMANN) return  bc_hodge.wallValue(x+dx/2,y,z);
      else                                       return (bc_hodge.wallValue(x+dx/2,y,z) - hodge_q) * 2 / dx;
    case dir::f_0m0:
      if(bc_hodge.wallType(x,y-dy/2,z)==NEUMANN) return -bc_hodge.wallValue(x,y-dy/2,z);
      else                                       return (hodge_q - bc_hodge.wallValue(x,y-dy/2,z)) * 2 / dy;
    case dir::f_0p0:
      if(bc_hodge.wallType(x,y+dy/2,z)==NEUMANN) return  bc_hodge.wallValue(x,y+dy/2,z);
      else                                       return (bc_hodge.wallValue(x,y+dy/2,z) - hodge_q) * 2 / dy;
    case dir::f_00m:
      if(bc_hodge.wallType(x,y,z-dz/2)==NEUMANN) return -bc_hodge.wallValue(x,y,z-dz/2);
      else                                       return (hodge_q - bc_hodge.wallValue(x,y,z-dz/2)) * 2 / dz;
    case dir::f_00p:
      if(bc_hodge.wallType(x,y,z+dz/2)==NEUMANN) return  bc_hodge.wallValue(x,y,z+dz/2);
      else                                       return (bc_hodge.wallValue(x,y,z+dz/2) - hodge_q) * 2 / dz;
#else
    case dir::f_m00:
      if(bc_hodge.wallType(x-dx/2,y)==NEUMANN) return -bc_hodge.wallValue(x-dx/2,y);
      else                                     return (hodge_q - bc_hodge.wallValue(x-dx/2,y)) * 2 / dx;
    case dir::f_p00:
      if(bc_hodge.wallType(x+dx/2,y)==NEUMANN) return  bc_hodge.wallValue(x+dx/2,y);
      else                                     return (bc_hodge.wallValue(x+dx/2,y) - hodge_q) * 2 / dx;
    case dir::f_0m0:
      if(bc_hodge.wallType(x,y-dy/2)==NEUMANN) return -bc_hodge.wallValue(x,y-dy/2);
      else                                     return (hodge_q - bc_hodge.wallValue(x,y-dy/2)) * 2 / dy;
    case dir::f_0p0:
      if(bc_hodge.wallType(x,y+dy/2)==NEUMANN) return  bc_hodge.wallValue(x,y+dy/2);
      else                                     return (bc_hodge.wallValue(x,y+dy/2) - hodge_q) * 2 / dy;
#endif
    default:
      throw std::invalid_argument("[ERROR]: my_p4est_navier_stokes_t->dxyz_hodge: unknown direction.");
    }
  }
  else
  {
    std::vector<p4est_quadrant_t> ngbd;
    ngbd.resize(0);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, dir);

    /* multiple neighbor cells should never happen since this function is called for a given face,
     * and the faces are defined only for small cells.
     */
    if(ngbd.size()>1)
    {
      throw std::invalid_argument("[ERROR]: my_p4est_navier_stokes->compute_dxyz_hodge: invalid case.");
    }
    /* one neighbor cell of same size, check for interface */
    else if(ngbd[0].level == quad->level)
    {
      double xq = quad_x_fr_q(quad_idx, tree_idx, p4est_n, ghost_n);
      double yq = quad_y_fr_q(quad_idx, tree_idx, p4est_n, ghost_n);

      double x0 = quad_x_fr_q(ngbd[0].p.piggy3.local_num, ngbd[0].p.piggy3.which_tree, p4est_n, ghost_n);
      double y0 = quad_y_fr_q(ngbd[0].p.piggy3.local_num, ngbd[0].p.piggy3.which_tree, p4est_n, ghost_n);

#ifdef P4_TO_P8
      double zq = quad_z_fr_q(quad_idx, tree_idx, p4est_n, ghost_n);
      double z0 = quad_z_fr_q(ngbd[0].p.piggy3.local_num, ngbd[0].p.piggy3.which_tree, p4est_n, ghost_n);
      double phi_q = (*interp_phi)(xq, yq, zq);
      double phi_0 = (*interp_phi)(x0, y0, z0);
#else
      double phi_q = (*interp_phi)(xq, yq);
      double phi_0 = (*interp_phi)(x0, y0);
#endif

      double dmin = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
      double dx = convert_to_xyz[dir/2] * dmin;

      if(bc_hodge.interfaceType()==DIRICHLET && phi_q*phi_0<0)
      {
        if(phi_q>0)
        {
          double phi_tmp = phi_q; phi_q = phi_0; phi_0 = phi_tmp;
          dir = dir%2==0 ? dir+1 : dir-1;
          quad_idx = ngbd[0].p.piggy3.local_num;
          switch(dir)
          {
          case dir::f_m00: case dir::f_p00: xq = x0; break;
          case dir::f_0m0: case dir::f_0p0: yq = y0; break;
#ifdef P4_TO_P8
          case dir::f_00m: case dir::f_00p: zq = z0; break;
#endif
          }
        }

        double theta = fraction_Interval_Covered_By_Irregular_Domain(phi_q, phi_0, dx, dx);
        if(theta<EPS)
          theta = EPS;
        if(theta>1)
          theta = 1.0;
        double val_interface;
        double dist = dx*theta;
        switch(dir)
        {
#ifdef P4_TO_P8
        case dir::f_m00: case dir::f_p00: val_interface = bc_hodge.interfaceValue(xq + (dir%2==0 ? -1 : 1)*dist, yq, zq); break;
        case dir::f_0m0: case dir::f_0p0: val_interface = bc_hodge.interfaceValue(xq, yq + (dir%2==0 ? -1 : 1)*dist, zq); break;
        case dir::f_00m: case dir::f_00p: val_interface = bc_hodge.interfaceValue(xq, yq, zq + (dir%2==0 ? -1 : 1)*dist); break;
#else
        case dir::f_m00: case dir::f_p00: val_interface = bc_hodge.interfaceValue(xq + (dir%2==0 ? -1 : 1)*dist, yq); break;
        case dir::f_0m0: case dir::f_0p0: val_interface = bc_hodge.interfaceValue(xq, yq + (dir%2==0 ? -1 : 1)*dist); break;
#endif
        default:
          throw std::invalid_argument("[ERROR]: my_p4est_navier_stokes_t->dxyz_hodge: uknown direction.");
        }

        double grad_hodge = hodge_p[quad_idx] - val_interface;

        ierr = VecRestoreArrayRead(hodge, &hodge_p); CHKERRXX(ierr);
        return dir%2==0 ? grad_hodge/dist : -grad_hodge/dist;
      }
      else
      {
        double grad_hodge = hodge_p[quad_idx] - hodge_p[ngbd[0].p.piggy3.local_num];

        ierr = VecRestoreArrayRead(hodge, &hodge_p); CHKERRXX(ierr);
        return dir%2==0 ? grad_hodge/dx : -grad_hodge/dx;
      }
    }
    /* one neighbor cell that is bigger, get common neighbors */
    else
    {
      p4est_quadrant_t quad_tmp = ngbd[0];
      ngbd.resize(0);
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_tmp.p.piggy3.local_num, quad_tmp.p.piggy3.which_tree, dir%2==0 ? dir+1 : dir-1);

      double dist = 0;
      double grad_hodge = 0;
      double d0 = (double)P4EST_QUADRANT_LEN(quad_tmp.level)/(double)P4EST_ROOT_LEN;

      for(unsigned int m=0; m<ngbd.size(); ++m)
      {
        double dm = (double)P4EST_QUADRANT_LEN(ngbd[m].level)/(double)P4EST_ROOT_LEN;
        dist += pow(dm,P4EST_DIM-1) * .5*(d0+dm);
        grad_hodge += (hodge_p[ngbd[m].p.piggy3.local_num] - hodge_p[quad_tmp.p.piggy3.local_num]) * pow(dm,P4EST_DIM-1);
      }
      dist *= convert_to_xyz[dir/2];

      ierr = VecRestoreArrayRead(hodge, &hodge_p); CHKERRXX(ierr);
      return dir%2==0 ? grad_hodge/dist : -grad_hodge/dist;
    }
  }
}

double my_p4est_navier_stokes_t::compute_divergence(p4est_locidx_t quad_idx, p4est_topidx_t tree_idx)
{
  PetscErrorCode ierr;
  p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est_n->trees, tree_idx);
  p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);

  double dmin = (double)P4EST_QUADRANT_LEN(quad->level) / (double)P4EST_ROOT_LEN;
  double val = 0;

  std::vector<p4est_quadrant_t> ngbd;
  p4est_quadrant_t quad_tmp;

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    const double *vstar_p;
    ierr = VecGetArrayRead(vstar[dir], &vstar_p); CHKERRXX(ierr);

    double vm = 0;
    if(is_quad_Wall(p4est_n, tree_idx, quad, 2*dir))
    {
      vm = vstar_p[faces_n->q2f(quad_idx, 2*dir)];
    }
    else if(faces_n->q2f(quad_idx, 2*dir)!=NO_VELOCITY)
    {
      ngbd.resize(0);
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 2*dir);
      quad_tmp = ngbd[0];
      ngbd.resize(0);
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_tmp.p.piggy3.local_num, quad_tmp.p.piggy3.which_tree, 2*dir+1);
      for(unsigned int m=0; m<ngbd.size(); ++m)
        vm += pow((double)P4EST_QUADRANT_LEN(ngbd[m].level)/(double)P4EST_ROOT_LEN, P4EST_DIM-1) * vstar_p[faces_n->q2f(ngbd[m].p.piggy3.local_num, 2*dir)];
      vm /= pow((double)P4EST_QUADRANT_LEN(quad_tmp.level)/(double)P4EST_ROOT_LEN, P4EST_DIM-1);
    }
    else
    {
      ngbd.resize(0);
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 2*dir);
      for(unsigned int m=0; m<ngbd.size(); ++m)
        vm += pow((double)P4EST_QUADRANT_LEN(ngbd[m].level)/(double)P4EST_ROOT_LEN, P4EST_DIM-1) * vstar_p[faces_n->q2f(ngbd[m].p.piggy3.local_num, 2*dir+1)];
      vm /= pow((double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN, P4EST_DIM-1);
    }

    double vp = 0;
    if(is_quad_Wall(p4est_n, tree_idx, quad, 2*dir+1))
    {
      vp = vstar_p[faces_n->q2f(quad_idx, 2*dir+1)];
    }
    else if(faces_n->q2f(quad_idx, 2*dir+1)!=NO_VELOCITY)
    {
      ngbd.resize(0);
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 2*dir+1);
      quad_tmp = ngbd[0];
      ngbd.resize(0);
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_tmp.p.piggy3.local_num, quad_tmp.p.piggy3.which_tree, 2*dir);

      for(unsigned int m=0; m<ngbd.size(); ++m)
        vp += pow((double)P4EST_QUADRANT_LEN(ngbd[m].level)/(double)P4EST_ROOT_LEN, P4EST_DIM-1) * vstar_p[faces_n->q2f(ngbd[m].p.piggy3.local_num, 2*dir+1)];
      vp /= pow((double)P4EST_QUADRANT_LEN(quad_tmp.level)/(double)P4EST_ROOT_LEN, P4EST_DIM-1);
    }
    else
    {
      ngbd.resize(0);
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 2*dir+1);
      for(unsigned int m=0; m<ngbd.size(); ++m)
        vp += pow((double)P4EST_QUADRANT_LEN(ngbd[m].level)/(double)P4EST_ROOT_LEN, P4EST_DIM-1) * vstar_p[faces_n->q2f(ngbd[m].p.piggy3.local_num, 2*dir)];
      vp /= pow((double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN, P4EST_DIM-1);
    }

    ierr = VecRestoreArrayRead(vstar[dir], &vstar_p); CHKERRXX(ierr);

    val += (vp-vm)/(convert_to_xyz[dir] * dmin);
  }

  return val;
}

void my_p4est_navier_stokes_t::solve_viscosity(my_p4est_poisson_faces_t* &face_poisson_solver, const bool use_initial_guess, const KSPType ksp, const PCType pc)
{
  PetscErrorCode ierr;

  ierr = PetscLogEventBegin(log_my_p4est_navier_stokes_viscosity, 0, 0, 0, 0); CHKERRXX(ierr);

  double alpha;
  double beta;
  if(sl_order==1)
  {
    alpha = 1;
    beta = 0;
  }
  else
  {
    alpha = (2*dt_n+dt_nm1)/(dt_n+dt_nm1);
    beta = -dt_n/(dt_n+dt_nm1);
  }

  /* construct the right hand side */
  if(!semi_lagrangian_backtrace_is_done)
  {
    trajectory_from_np1_all_faces(p4est_n, faces_n, ngbd_nm1, ngbd_n,
                                  vnm1_nodes, second_derivatives_vnm1_nodes,
                                  vn_nodes, second_derivatives_vn_nodes, dt_nm1, dt_n,
                                  xyz_n, ((sl_order == 2)? xyz_nm1 : NULL));
    semi_lagrangian_backtrace_is_done = true;
  }

  // rhs is modified by the solver so it should be reset every-time (could be modified, if required)
  Vec rhs[P4EST_DIM];
  for(short dir=0; dir<P4EST_DIM; ++dir)
  {
    /* find the velocity at the backtraced points */
    my_p4est_interpolation_nodes_t interp_nm1(ngbd_nm1);
    my_p4est_interpolation_nodes_t interp_n  (ngbd_n  );
    for(p4est_locidx_t f_idx=0; f_idx<faces_n->num_local[dir]; ++f_idx)
    {
      double xyz_tmp[P4EST_DIM];

      if(sl_order==2)
      {
#ifdef P4_TO_P8
        xyz_tmp[0] = xyz_nm1[dir][0][f_idx]; xyz_tmp[1] = xyz_nm1[dir][1][f_idx]; xyz_tmp[2] = xyz_nm1[dir][2][f_idx];
#else
        xyz_tmp[0] = xyz_nm1[dir][0][f_idx]; xyz_tmp[1] = xyz_nm1[dir][1][f_idx];
#endif
        interp_nm1.add_point(f_idx, xyz_tmp);
      }

#ifdef P4_TO_P8
      xyz_tmp[0] = xyz_n[dir][0][f_idx]; xyz_tmp[1] = xyz_n[dir][1][f_idx]; xyz_tmp[2] = xyz_n[dir][2][f_idx];
#else
      xyz_tmp[0] = xyz_n[dir][0][f_idx]; xyz_tmp[1] = xyz_n[dir][1][f_idx];
#endif
      interp_n.add_point(f_idx, xyz_tmp);
    }

    std::vector<double> vnm1_faces(faces_n->num_local[dir]);
    if(sl_order==2)
    {
      vnm1_faces.resize(faces_n->num_local[dir]);
      interp_nm1.set_input(vnm1_nodes[dir], second_derivatives_vnm1_nodes[0][dir], second_derivatives_vnm1_nodes[1][dir],
    #ifdef P4_TO_P8
          second_derivatives_vnm1_nodes[2][dir],
    #endif
          quadratic);
      interp_nm1.interpolate(vnm1_faces.data());
    }

    std::vector<double> vn_faces(faces_n->num_local[dir]);
    interp_n.set_input(vn_nodes[dir], second_derivatives_vn_nodes[0][dir], second_derivatives_vn_nodes[1][dir],
    #ifdef P4_TO_P8
        second_derivatives_vn_nodes[2][dir],
    #endif
        quadratic);
    interp_n.interpolate(vn_faces.data());

    /* assemble the right-hand-side */
    ierr = VecDuplicate(vstar[dir], &rhs[dir]); CHKERRXX(ierr);

    const PetscScalar *face_is_well_defined_p;
    ierr = VecGetArrayRead(face_is_well_defined[dir], &face_is_well_defined_p); CHKERRXX(ierr);

    double *rhs_p;
    ierr = VecGetArray(rhs[dir], &rhs_p); CHKERRXX(ierr);

    for(p4est_locidx_t f_idx=0; f_idx<faces_n->num_local[dir]; ++f_idx)
    {
      if(face_is_well_defined_p[f_idx])
      {
        if(sl_order==1)
          rhs_p[f_idx] = rho/dt_n * vn_faces[f_idx];
        else
          rhs_p[f_idx] = -rho * ( (-alpha/dt_n + beta/dt_nm1)*vn_faces[f_idx] - beta/dt_nm1*vnm1_faces[f_idx]);

        if(external_forces[dir]!=NULL)
        {
          double xyz[P4EST_DIM];
          faces_n->xyz_fr_f(f_idx, dir, xyz);
#ifdef P4_TO_P8
          rhs_p[f_idx] += (*external_forces[dir])(xyz[0], xyz[1], xyz[2]);
#else
          rhs_p[f_idx] += (*external_forces[dir])(xyz[0], xyz[1]);
#endif
        }
      }
      else
        rhs_p[f_idx] = 0;
    }

    ierr = VecRestoreArray(rhs[dir], &rhs_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(face_is_well_defined[dir], &face_is_well_defined_p); CHKERRXX(ierr);
  }

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
  face_poisson_solver->set_diagonal(alpha * rho/dt_n);
  face_poisson_solver->set_rhs(rhs);
  face_poisson_solver->solve(vstar, use_initial_guess, ksp, pc);

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecDestroy(rhs[dir]); CHKERRXX(ierr);
  }

  if(bc_pressure->interfaceType()!=NOINTERFACE)
  {
    my_p4est_level_set_faces_t lsf(ngbd_n, faces_n);
    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      lsf.extend_Over_Interface(phi, vstar[dir], bc_v[dir], dir, face_is_well_defined[dir], dxyz_hodge[dir], 2, 2);
    }
  }

  ierr = PetscLogEventEnd(log_my_p4est_navier_stokes_viscosity, 0, 0, 0, 0); CHKERRXX(ierr);
}

/* solve the projection step
 * laplace Hodge = -div(vstar)
 */
void my_p4est_navier_stokes_t::solve_projection(my_p4est_poisson_cells_t* &cell_solver, const bool use_initial_guess, const KSPType ksp, const PCType pc)
{
  PetscErrorCode ierr;

  ierr = PetscLogEventBegin(log_my_p4est_navier_stokes_projection, 0, 0, 0, 0); CHKERRXX(ierr);

  Vec rhs;
  ierr = VecDuplicate(hodge, &rhs); CHKERRXX(ierr);

  double *rhs_p;
  ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);

  /* compute the right-hand-side */
  for(p4est_topidx_t tree_idx=p4est_n->first_local_tree; tree_idx<=p4est_n->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est_n->trees, tree_idx);
    for(size_t q_idx=0; q_idx<tree->quadrants.elem_count; ++q_idx)
    {
      p4est_locidx_t quad_idx = q_idx+tree->quadrants_offset;

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

  /* if needed, shift the hodge variable to a zero average */
  if(1 && cell_solver->get_matrix_has_nullspace())
  {
    my_p4est_level_set_cells_t lsc(ngbd_c, ngbd_n);
    double average = lsc.integrate(phi, hodge) / area_in_negative_domain(p4est_n, nodes_n, phi);
    double *hodge_p;
    ierr = VecGetArray(hodge, &hodge_p); CHKERRXX(ierr);
    for(p4est_locidx_t quad_idx=0; quad_idx<p4est_n->local_num_quadrants; ++quad_idx)
      hodge_p[quad_idx] -= average;
    ierr = VecRestoreArray(hodge, &hodge_p); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(hodge, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (hodge, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  if(bc_pressure->interfaceType()!=NOINTERFACE)
  {
    my_p4est_level_set_cells_t lsc(ngbd_c, ngbd_n);
    lsc.extend_Over_Interface(phi, hodge, &bc_hodge, 2, 2);
  }

  /* project vstar */
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    double *dxyz_hodge_p;
    ierr = VecGetArray(dxyz_hodge[dir], &dxyz_hodge_p); CHKERRXX(ierr);

    const double *vstar_p;
    ierr = VecGetArrayRead(vstar[dir], &vstar_p); CHKERRXX(ierr);

    double *vnp1_p;
    ierr = VecGetArray(vnp1[dir], &vnp1_p); CHKERRXX(ierr);

    p4est_locidx_t quad_idx;
    p4est_topidx_t tree_idx;

    for(p4est_locidx_t f_idx=0; f_idx<faces_n->num_local[dir]; ++f_idx)
    {
      faces_n->f2q(f_idx, dir, quad_idx, tree_idx);
      int tmp = faces_n->q2f(quad_idx, 2*dir)==f_idx ? 0 : 1;
      dxyz_hodge_p[f_idx] = compute_dxyz_hodge(quad_idx, tree_idx, 2*dir+tmp);
    }

    ierr = VecGhostUpdateBegin(dxyz_hodge[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    for(p4est_locidx_t f_idx=0; f_idx<faces_n->num_local[dir]; ++f_idx)
    {
      vnp1_p[f_idx] = vstar_p[f_idx] - dxyz_hodge_p[f_idx];
    }

    ierr = VecGhostUpdateEnd  (dxyz_hodge[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecRestoreArray(vnp1[dir], &vnp1_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(vstar[dir], &vstar_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(dxyz_hodge[dir], &dxyz_hodge_p); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(vnp1[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (vnp1[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    //    if(bc_pressure->interfaceType()!=NOINTERFACE)
    //    {
    //      my_p4est_level_set_faces_t lsf(ngbd_n, faces_n);
    //      lsf.extend_Over_Interface(phi, vnp1[dir], bc_v[dir], dir, face_is_well_defined[dir], NULL, 2, 8);
    //    }
  }

  ierr = PetscLogEventEnd(log_my_p4est_navier_stokes_projection, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_navier_stokes_t::enforce_mass_flow(const bool* force_in_direction, const double* desired_mean_velocity, double* forcing_mean_hodge_gradient, double* mass_flow)
{
  double current_mass_flow;
  PetscErrorCode ierr;
  double *vel_p;
  for (short unsigned dir = 0; dir < P4EST_DIM; ++dir) {
    if(!force_in_direction[dir])
    {
      forcing_mean_hodge_gradient[dir] = 0.0;
      continue;
    }
    if(!is_periodic(p4est_n, dir))
    {
#ifdef CASL_THROWS
      throw std::invalid_argument("my_p4est_navier_stokes_t::enforce_mass_flow: this function cannot be called to enforce mass flow in a nonperiodic direction");
#else
      forcing_mean_hodge_gradient[dir] = 0.0;
      continue;
#endif
    }
    if(mass_flow == NULL)
    {
      double section = xyz_min[dir];
      global_mass_flow_through_slice(dir, section, current_mass_flow);
    }
    else
      current_mass_flow = mass_flow[dir];

    double current_mean_velocity      = current_mass_flow/(xyz_max[(dir+1)%P4EST_DIM] - xyz_min[(dir+1)%P4EST_DIM]);
#ifdef P4_TO_P8
    current_mean_velocity            /= (xyz_max[(dir+2)%P4EST_DIM] - xyz_min[(dir+2)%P4EST_DIM]);
#endif
    current_mean_velocity            /= rho;

    forcing_mean_hodge_gradient[dir]  = current_mean_velocity - desired_mean_velocity[dir];
    ierr = VecGetArray(vnp1[dir], &vel_p); CHKERRXX(ierr);
    for (p4est_locidx_t k = 0; k < (faces_n->num_local[dir] + faces_n->num_ghost[dir]); ++k)
      vel_p[k] -= forcing_mean_hodge_gradient[dir];
    if(mass_flow!=NULL)
    {
      mass_flow[dir] = desired_mean_velocity[dir]*(xyz_max[(dir+1)%P4EST_DIM] - xyz_min[(dir+1)%P4EST_DIM]);
#ifdef P4_TO_P8
      mass_flow[dir] *= (xyz_max[(dir+2)%P4EST_DIM] - xyz_min[(dir+2)%P4EST_DIM]);
#endif
    }
    ierr = VecRestoreArray(vnp1[dir], &vel_p); CHKERRXX(ierr);
  }
}

void my_p4est_navier_stokes_t::compute_velocity_at_nodes(const bool store_interpolators)
{
  /* interpolate vnp1 from faces to nodes */
  for(unsigned short dir=0; dir<P4EST_DIM; ++dir)
  {
    PetscErrorCode ierr;
    double *v_p;
    const double *vnp1_read_p;
    ierr = VecGetArray(vnp1_nodes[dir], &v_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(vnp1[dir], &vnp1_read_p); CHKERRXX(ierr);

    if(store_interpolators && !interpolators_from_face_to_nodes_are_set)
      interpolator_from_face_to_nodes[dir].resize(nodes_n->num_owned_indeps);

    for(size_t i=0; i<ngbd_n->get_layer_size(); ++i)
    {
      p4est_locidx_t n = ngbd_n->get_layer_node(i);
      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, p4est_n, nodes_n, xyz);
      if(!interpolators_from_face_to_nodes_are_set)
        v_p[n] = interpolate_f_at_node_n(p4est_n, ghost_n, nodes_n, faces_n, ngbd_c, ngbd_n, n,
                                         vnp1[dir], dir, face_is_well_defined[dir], 2, bc_v, (store_interpolators? &interpolator_from_face_to_nodes[dir][n]: NULL));
      else
      {
        const face_interpolator& my_interpolator = interpolator_from_face_to_nodes[dir][n];
        p4est_indep_t *node = (p4est_indep_t*) sc_array_index(&nodes_n->indep_nodes, n);
        if(my_interpolator.size() == 0)
        {
          v_p[n] = 0.0;
          continue;
        }
        P4EST_ASSERT(my_interpolator.size()>0);
        if((my_interpolator.size() == 1))
        {
          P4EST_ASSERT(my_interpolator[0].face_idx <0 && bc_v!=NULL && is_node_Wall(p4est_n, node) && bc_v[dir].wallType(xyz)==DIRICHLET);
          v_p[n] = bc_v[dir].wallValue(xyz);
        }
        else
        {
          if(is_node_Wall(p4est_n, node))
          {
            P4EST_ASSERT(bc_v!=NULL);
            if(bc_v[dir].wallType(xyz) == DIRICHLET) // should have been dealt with above...
              throw std::runtime_error("my_p4est_navier_stokes_t::compute_velocity_at_nodes: unexpected behavior when reusing interpolator_from_face_to_nodes on a Dirichlet wall node");
            if((bc_v[dir].wallType(xyz) == NEUMANN) && (fabs(bc_v[dir].wallValue(xyz)) > EPS))
              throw std::runtime_error("my_p4est_navier_stokes_t::compute_velocity_at_nodes: when reusing interpolator_from_face_to_nodes on a Neumann wall node, the Neumann boundary value MUST be 0.0");
          }
          v_p[n] = 0.0;
          for (unsigned int k = 0; k < my_interpolator.size(); ++k)
            v_p[n] += my_interpolator[k].weight*vnp1_read_p[my_interpolator[k].face_idx];
        }
      }
    }

    ierr = VecGhostUpdateBegin(vnp1_nodes[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    for(size_t i=0; i<ngbd_n->get_local_size(); ++i)
    {
      p4est_locidx_t n = ngbd_n->get_local_node(i);
      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, p4est_n, nodes_n, xyz);
      if(!interpolators_from_face_to_nodes_are_set)
        v_p[n] = interpolate_f_at_node_n(p4est_n, ghost_n, nodes_n, faces_n, ngbd_c, ngbd_n, n,
                                         vnp1[dir], dir, face_is_well_defined[dir], 2, bc_v, (store_interpolators? &interpolator_from_face_to_nodes[dir][n]: NULL));
      else
      {
        const face_interpolator& my_interpolator = interpolator_from_face_to_nodes[dir][n];
        p4est_indep_t *node = (p4est_indep_t*) sc_array_index(&nodes_n->indep_nodes, n);
        if(my_interpolator.size() == 0)
        {
          v_p[n] = 0.0;
          continue;
        }
        P4EST_ASSERT(my_interpolator.size()>0);
        if((my_interpolator.size() == 1))
        {
          P4EST_ASSERT(my_interpolator[0].face_idx <0 && bc_v!=NULL && is_node_Wall(p4est_n, node) && bc_v[dir].wallType(xyz)==DIRICHLET);
          v_p[n] = bc_v[dir].wallValue(xyz);
        }
        else
        {
          if(is_node_Wall(p4est_n, node))
          {
            P4EST_ASSERT(bc_v!=NULL);
            if(bc_v[dir].wallType(xyz) == DIRICHLET) // should have been dealt with above...
              throw std::runtime_error("my_p4est_navier_stokes_t::compute_velocity_at_nodes: unexpected behavior when reusing interpolator_from_face_to_nodes on a Dirichlet wall node");
            if((bc_v[dir].wallType(xyz) == NEUMANN) && (fabs(bc_v[dir].wallValue(xyz)) > EPS))
              throw std::runtime_error("my_p4est_navier_stokes_t::compute_velocity_at_nodes: when reusing interpolator_from_face_to_nodes on a Neumann wall node, the Neumann boundary value MUST be 0.0");
          }
          v_p[n] = 0.0;
          for (unsigned int k = 0; k < my_interpolator.size(); ++k)
            v_p[n] += my_interpolator[k].weight*vnp1_read_p[my_interpolator[k].face_idx];
        }
      }
    }

    ierr = VecGhostUpdateEnd(vnp1_nodes[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecRestoreArray(vnp1_nodes[dir], &v_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(vnp1[dir], &vnp1_read_p); CHKERRXX(ierr);

    if(bc_pressure->interfaceType()!=NOINTERFACE)
    {
      my_p4est_level_set_t lsn(ngbd_n);
      lsn.extend_Over_Interface_TVD(phi, vnp1_nodes[dir]);
    }
  }

  if(store_interpolators && !interpolators_from_face_to_nodes_are_set)
    interpolators_from_face_to_nodes_are_set = true;

  compute_vorticity();
  compute_max_L2_norm_u();
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


void my_p4est_navier_stokes_t::compute_adapted_dt(double min_value_for_umax)
{
  dt_nm1 = dt_n;
  dt_n = +DBL_MAX;
#ifdef P4_TO_P8
  double dxmin = MIN(dxyz_min[0], dxyz_min[1], dxyz_min[2]);
#else
  double dxmin = MIN(dxyz_min[0], dxyz_min[1]);
#endif
  splitting_criteria_t* data = (splitting_criteria_t*) p4est_n->user_pointer;
  PetscErrorCode ierr;
  const double* v_p[P4EST_DIM];
  for (short dd = 0; dd < P4EST_DIM; ++dd) {
    ierr = VecGetArrayRead(vnp1_nodes[dd], &v_p[dd]); CHKERRXX(ierr);
  }
  for (p4est_topidx_t tree_idx = p4est_n->first_local_tree; tree_idx <= p4est_n->last_local_tree; ++tree_idx) {
    p4est_tree_t* tree = p4est_tree_array_index(p4est_n->trees, tree_idx);
    for (size_t qq = 0; qq < tree->quadrants.elem_count; ++qq) {
      p4est_locidx_t quad_idx = tree->quadrants_offset + qq;
      double max_local_velocity_magnitude = -1.0;
      p4est_quadrant_t* quad = p4est_quadrant_array_index(&tree->quadrants, qq);
      for (short child_idx = 0; child_idx < P4EST_CHILDREN; ++child_idx) {
        p4est_locidx_t node_idx = nodes_n->local_nodes[P4EST_CHILDREN*quad_idx + child_idx];
        double node_vel_mag = 0.0;
        for (short dd = 0; dd < P4EST_DIM; ++dd)
          node_vel_mag += SQR(v_p[dd][node_idx]);
        node_vel_mag = sqrt(node_vel_mag);
        max_local_velocity_magnitude = MAX(max_local_velocity_magnitude, node_vel_mag);
      }
      dt_n = MIN(dt_n, MIN(1.0/max_local_velocity_magnitude, 1.0/min_value_for_umax)*n_times_dt*dxmin*((double) (1<<(data->max_lvl - quad->level))));
    }
  }
  for (short dd = 0; dd < P4EST_DIM; ++dd) {
    ierr = VecRestoreArrayRead(vnp1_nodes[dd], &v_p[dd]); CHKERRXX(ierr);
  }
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &dt_n, 1, MPI_DOUBLE, MPI_MIN, p4est_n->mpicomm); SC_CHECK_MPI(mpiret);
  dt_updated = true;
}

void my_p4est_navier_stokes_t::compute_dt(double min_value_for_umax)
{
  dt_nm1 = dt_n;
#ifdef P4_TO_P8
  dt_n = MIN(1/min_value_for_umax, 1/max_L2_norm_u) * n_times_dt * MIN(dxyz_min[0], dxyz_min[1], dxyz_min[2]);
#else
  dt_n = MIN(1/min_value_for_umax, 1/max_L2_norm_u) * n_times_dt * MIN(dxyz_min[0], dxyz_min[1]);
#endif

  dt_updated = true;
}

void my_p4est_navier_stokes_t::advect_smoke(my_p4est_node_neighbors_t* ngbd_n_np1, Vec* vnp1, Vec smoke_np1)
{
  PetscErrorCode ierr;
  std::vector<double> xyz_d[P4EST_DIM];
  trajectory_from_np1_to_n(ngbd_n_np1->p4est, ngbd_n_np1->nodes, ngbd_n_np1, dt_n, vnp1, xyz_d);

  my_p4est_interpolation_nodes_t interp_nodes(ngbd_n);
  for(p4est_locidx_t n=0; n< ngbd_n_np1->nodes->num_owned_indeps; ++n)
  {
#ifdef P4_TO_P8
    double xyz_d_[] = {xyz_d[0][n], xyz_d[1][n], xyz_d[2][n]};
#else
    double xyz_d_[] = {xyz_d[0][n], xyz_d[1][n]};
#endif
    interp_nodes.add_point(n, xyz_d_);
  }
  interp_nodes.set_input(smoke, linear);
  interp_nodes.interpolate(smoke_np1);

  /* enforce boundary condition */
  double *smoke_np1_p;
  ierr = VecGetArray(smoke_np1, &smoke_np1_p); CHKERRXX(ierr);
  for(size_t i=0; i<ngbd_n_np1->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n_np1->get_layer_node(i);
    double xyz[P4EST_DIM];
    node_xyz_fr_n(n, ngbd_n_np1->p4est, ngbd_n_np1->nodes, xyz);
#ifdef P4_TO_P8
    smoke_np1_p[n] = MAX(smoke_np1_p[n], (*bc_smoke)(xyz[0],xyz[1],xyz[2]));
#else
    smoke_np1_p[n] = MAX(smoke_np1_p[n], (*bc_smoke)(xyz[0],xyz[1]));
#endif
  }
  ierr = VecGhostUpdateBegin(smoke_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for(size_t i=0; i<ngbd_n_np1->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n_np1->get_local_node(i);
    double xyz[P4EST_DIM];
    node_xyz_fr_n(n, ngbd_n_np1->p4est, ngbd_n_np1->nodes, xyz);
#ifdef P4_TO_P8
    smoke_np1_p[n] = MAX(smoke_np1_p[n], (*bc_smoke)(xyz[0],xyz[1],xyz[2]));
#else
    smoke_np1_p[n] = MAX(smoke_np1_p[n], (*bc_smoke)(xyz[0],xyz[1]));
#endif
  }
  ierr = VecGhostUpdateEnd(smoke_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecRestoreArray(smoke_np1, &smoke_np1_p); CHKERRXX(ierr);
}

// [RAPHAEL]: this function is not even called anywhere... it flattens boundary conditions for velocity components on the
// interface towards the positive domain
void my_p4est_navier_stokes_t::extrapolate_bc_v(my_p4est_node_neighbors_t *ngbd, Vec *v, Vec phi)
{
  PetscErrorCode ierr;

  double *phi_p;
  ierr = VecGetArray(phi, &phi_p);
  quad_neighbor_nodes_of_node_t qnnn;

  double *v_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGetArray(v[dir], &v_p[dir]); CHKERRXX(ierr);
  }

  for(size_t i=0; i<ngbd->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd->get_layer_node(i);
    if(phi_p[n]>0)
    {
      ngbd->get_neighbors(n, qnnn);

      double x = node_x_fr_n(n, ngbd->p4est, ngbd->nodes) - phi_p[n]*qnnn.dx_central(phi_p);
      double y = node_y_fr_n(n, ngbd->p4est, ngbd->nodes) - phi_p[n]*qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
      double z = node_z_fr_n(n, ngbd->p4est, ngbd->nodes) - phi_p[n]*qnnn.dz_central(phi_p);
#endif

      for(int dir=0; dir<P4EST_DIM; ++dir)
      {
#ifdef P4_TO_P8
        v_p[dir][n] = bc_v[dir].interfaceValue(x,y,z);
#else
        v_p[dir][n] = bc_v[dir].interfaceValue(x,y);
#endif
      }
    }
  }

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGhostUpdateBegin(v[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  for(size_t i=0; i<ngbd->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd->get_local_node(i);
    if(phi_p[n]>0)
    {
      ngbd->get_neighbors(n, qnnn);

      double x = node_x_fr_n(n, ngbd->p4est, ngbd->nodes) - phi_p[n]*qnnn.dx_central(phi_p);
      double y = node_y_fr_n(n, ngbd->p4est, ngbd->nodes) - phi_p[n]*qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
      double z = node_z_fr_n(n, ngbd->p4est, ngbd->nodes) - phi_p[n]*qnnn.dz_central(phi_p);
#endif

      for(int dir=0; dir<P4EST_DIM; ++dir)
      {
#ifdef P4_TO_P8
        v_p[dir][n] = bc_v[dir].interfaceValue(x,y,z);
#else
        v_p[dir][n] = bc_v[dir].interfaceValue(x,y);
#endif
      }
    }
  }

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArray(v[dir], &v_p[dir]); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(v[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
}


#ifdef P4_TO_P8
bool my_p4est_navier_stokes_t::update_from_tn_to_tnp1(const CF_3 *level_set, bool keep_grid_as_such, bool do_reinitialization)
#else
bool my_p4est_navier_stokes_t::update_from_tn_to_tnp1(const CF_2 *level_set, bool keep_grid_as_such, bool do_reinitialization)
#endif
{
  PetscErrorCode ierr;

  ierr = PetscLogEventBegin(log_my_p4est_navier_stokes_update, 0, 0, 0, 0); CHKERRXX(ierr);

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
    splitting_criteria_vorticity_t criteria(data->min_lvl, data->max_lvl, data->lip, uniform_band, threshold_split_cell, max_L2_norm_u, smoke_thresh);
    /* construct a new forest */
    p4est_np1 = p4est_copy(p4est_n, P4EST_FALSE); // very efficient operation, costs almost nothing
    p4est_np1->connectivity = p4est_n->connectivity; // connectivity is not duplicated by p4est_copy, the pointer (i.e. the memory-address) of connectivity seems to be copied from my understanding of the source file of p4est_copy, but I feel this is a bit safer [Raphael Egan]
    p4est_np1->user_pointer = (void*)&criteria;
    Vec vorticity_np1 = NULL;
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
       * ((iter > 0) ? : ) */
      // partition the grid if it has changed...
      if(iter > 0)
        my_p4est_partition(p4est_np1, P4EST_FALSE, NULL);
      // ghost_np1, hierarchy_np1 and ngbd_n_np1 are required for the grid update if and only if smoke is defined and refinement with smoke is activated
      if ((iter>0) && (smoke!=NULL) && refine_with_smoke)
        ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
      // get nodes_np1
      // reset nodes_np1 if needed
      if ((nodes_np1 != NULL) && (nodes_np1!=nodes_n)){
        p4est_nodes_destroy(nodes_np1); CHKERRXX(ierr); }
      nodes_np1 = ((iter > 0) ? my_p4est_nodes_new(p4est_np1, ghost_np1) : nodes_n);
      // reset phi_np1 if needed
      if ((phi_np1 != NULL) && (phi_np1!= phi)){
        ierr = VecDestroy(phi_np1); CHKERRXX(ierr); phi_np1 = NULL; }
      // get phi_np1
      if(iter > 0 || level_set!=NULL)
      {
        ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);
        if(level_set!= NULL){
          ierr = VecGetArray(phi_np1, &phi_np1_p); CHKERRXX(ierr); }
        for(size_t n=0; n<nodes_np1->indep_nodes.elem_count; ++n)
        {
          double xyz[P4EST_DIM];
          node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);
          if(iter > 0)
            interp_nodes.add_point(n, xyz);
          if(level_set != NULL)
#ifdef P4_TO_P8
            phi_np1_p[n] = (*level_set)(xyz[0], xyz[1], xyz[2]);
#else
            phi_np1_p[n] = (*level_set)(xyz[0], xyz[1]);
#endif
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
      if ((vorticity_np1 != NULL) && (vorticity_np1!=vorticity)){
        ierr = VecDestroy(vorticity_np1); CHKERRXX(ierr); vorticity_np1 = NULL; }
      // get vorticity_np1
      if(iter > 0)
      {
        ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &vorticity_np1); CHKERRXX(ierr);
        interp_inputs.push_back(vorticity);
        interp_outputs.push_back(vorticity_np1);
      }
      else
        vorticity_np1 = vorticity;
      // reset smoke_np1 if needed
      if ((smoke_np1 != NULL) && (smoke_np1!=smoke)){
        ierr = VecDestroy(smoke_np1); CHKERRXX(ierr); smoke_np1 = NULL; }
      // get smoke_np1 (if required)
      Vec vtmp[P4EST_DIM];
      if(smoke!=NULL && refine_with_smoke)
      {
        ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &smoke_np1); CHKERRXX(ierr);
        for(int dir=0; dir<P4EST_DIM; ++dir)
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
      if(smoke!=NULL && refine_with_smoke)
      {
        P4EST_ASSERT((ghost_np1!=NULL) || (iter == 0));
        hierarchy_np1 = ((iter>0)? (new my_p4est_hierarchy_t(p4est_np1, ghost_np1, brick)): hierarchy_n);
        ngbd_n_np1    = ((iter>0)? (new my_p4est_node_neighbors_t(hierarchy_np1, nodes_np1)): ngbd_n);
        advect_smoke(ngbd_n_np1, vtmp, smoke_np1);
        for(short dir=0; dir<P4EST_DIM; ++dir)
          if((vtmp[dir] != NULL) && (vtmp[dir] != vnp1_nodes[dir])){
            ierr = VecDestroy(vtmp[dir]); CHKERRXX(ierr); }
        if(iter > 0)
        {
          p4est_ghost_destroy(ghost_np1);
          delete  hierarchy_np1;
          delete  ngbd_n_np1;
        }
      }
      iterative_grid_update_converged = !criteria.refine_and_coarsen(p4est_np1, nodes_np1, phi_np1, vorticity_np1, smoke_np1);
      grid_is_unchanged = grid_is_unchanged && iterative_grid_update_converged;
      iter++;

      if(iter>((unsigned int) 2+data->max_lvl-data->min_lvl)) // increase the rhs by one to account for the very first step that used to be out of the loop, [Raphael]
      {
        ierr = PetscPrintf(p4est_n->mpicomm, "ooops ... the grid update did not converge\n"); CHKERRXX(ierr);
        break;
      }
    }
    if((vorticity_np1!= NULL) && (vorticity_np1!=vorticity)){
      ierr = VecDestroy(vorticity_np1); CHKERRXX(ierr); } // destroy it if created in the iterative process
    // done refining using the specific grid-refinement criterion, reset the original data as user-pointer
    p4est_np1->user_pointer = data;
  }

  int smoke_np1_is_still_valid = (((smoke_np1!= NULL) && iterative_grid_update_converged)? 1 : 0); // collectively the same at this stage: the vectors are still ok if the iterative procedure exited properly by grid_is_changing turning false;
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
    if(p4est_np1!=p4est_n)
      p4est_destroy(p4est_np1);// no need to keep two identical forests in memory if they are exactly the same, just make p4est_nm1 and p4est_n point to the same one eventually and handle memory correspondingly...
    p4est_np1     = p4est_n;
  }

  /* Get the ghost cells at time np1, */
  p4est_ghost_t *ghost_np1 = NULL;
  if(!grid_is_unchanged)
  {
    ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
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
    P4EST_ASSERT((nodes_np1 == NULL) || (nodes_np1 == nodes_n));
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
  //    i)  if the grid is unchanged and phi_np1 is not defined yet (iterative grid update forcibly skipped) --> sample the given valid levelset if it is given, or simply set phi_np1 = phi
  //    ii) if the grid is changed, the node interpolator input buffer needs to be reset anyways (for future interpolation) and phi_np1 is recreated as a side result (if previously evaluated
  //        from a valid given levelset, this is a small overhead; if previously obtained from linear interpolation it is no longer valid anyways since non-oscillatory quadratic interpolation
  //        is desired eventually)
  // Reinitialization is performed only if desired and if either the grid has changed or if a valid given levelset function has been given!
  if(grid_is_unchanged)
  {
    P4EST_ASSERT(ghosts_are_equal(ghost_n, ghost_np1));
    P4EST_ASSERT(nodes_are_equal(p4est_n->mpisize, nodes_n, nodes_np1));
    P4EST_ASSERT(phi!=NULL);
    P4EST_ASSERT(((phi_np1 == NULL) && keep_grid_as_such) || (!keep_grid_as_such && (((phi_np1==phi) && (level_set==NULL)) || ((phi_np1!=phi) && (phi_np1!=NULL) && (level_set!=NULL)))));
    if (phi_np1 == NULL) // in this case, the grid is forcibly kept as such
    {
      if(level_set != NULL) // if keep_grid_as_such is true but a valid CF_ levelset is given...
        sample_cf_on_nodes(p4est_np1, nodes_np1, *level_set, phi);
      else
        phi_np1 = phi;
    }
  }
  else
  {
    P4EST_ASSERT((phi_np1!= NULL) && (phi_np1!=phi));
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
      if(level_set!= NULL)
  #ifdef P4_TO_P8
        phi_np1_p[n] = (*level_set)(xyz[0], xyz[1], xyz[2]);
  #else
        phi_np1_p[n] = (*level_set)(xyz[0], xyz[1]);
  #endif
    }
    if(level_set != NULL){
      ierr = VecRestoreArray(phi_np1, &phi_np1_p); CHKERRXX(ierr); }
    else
    {
      interp_nodes.set_input(phi, quadratic_non_oscillatory);
      interp_nodes.interpolate(phi_np1);
    }

  }
  if(do_reinitialization && (!grid_is_unchanged || (level_set != NULL)))
  {
    my_p4est_level_set_t lsn(ngbd_np1);
    lsn.reinitialize_1st_order_time_2nd_order_space(phi_np1);
    lsn.perturb_level_set_function(phi_np1, EPS);
  }
  // we can finally slide phi, current phi is no longer needed (IF different than phi_np1)...
  if(phi!=phi_np1){
    ierr = VecDestroy(phi); CHKERRXX(ierr); }
  // reset the phi-interpolator tool if needed
  if(ngbd_n!=ngbd_np1)
  {
    delete interp_phi;
    interp_phi = new my_p4est_interpolation_nodes_t(ngbd_np1);
  }
  // reset the phi-interpolator input and slide phi!
  interp_phi->set_input(phi_np1, linear);
  phi = phi_np1;

  // 2) scalar field vorticity: reset it to the appropriate size if the grid has changed, nothing to do otherwise
  if(!grid_is_unchanged){
    ierr = VecDestroy(vorticity); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &vorticity); CHKERRXX(ierr);
  }
  // 3) velocity fields at the nodes (and their second derivatives)
  for(short dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecDestroy(vnm1_nodes[dir]); CHKERRXX(ierr);
    vnm1_nodes[dir] = vn_nodes[dir];
    if(!grid_is_unchanged){
      ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &vn_nodes[dir]); CHKERRXX(ierr); }
    else
      vn_nodes[dir]   = vnp1_nodes[dir];
    for (short dd = 0; dd < P4EST_DIM; ++dd) {
      ierr = VecDestroy(second_derivatives_vnm1_nodes[dd][dir]); CHKERRXX(ierr);
      second_derivatives_vnm1_nodes[dd][dir] = second_derivatives_vn_nodes[dd][dir];
      ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &second_derivatives_vn_nodes[dd][dir]); CHKERRXX(ierr);
    }
  }
  if(!grid_is_unchanged)
  {
    interp_nodes.set_input(vnp1_nodes, quadratic, P4EST_DIM);
    interp_nodes.interpolate(vn_nodes, P4EST_DIM); CHKERRXX(ierr);
  }
  for (short dir = 0; dir < P4EST_DIM; ++dir) {
    if(!grid_is_unchanged){
      ierr = VecDestroy(vnp1_nodes[dir]); CHKERRXX(ierr); }
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &vnp1_nodes[dir]); CHKERRXX(ierr);
  }
#ifdef P4_TO_P8
  ngbd_np1->second_derivatives_central(vn_nodes, second_derivatives_vn_nodes[0], second_derivatives_vn_nodes[1], second_derivatives_vn_nodes[2], P4EST_DIM);
#else
  ngbd_np1->second_derivatives_central(vn_nodes, second_derivatives_vn_nodes[0], second_derivatives_vn_nodes[1], P4EST_DIM);
#endif
  if(!grid_is_unchanged)
    interp_nodes.clear();

  // [Raphael]: the following was already commented in the original version. If uncommented, I think it should be called here...
  /* set velocity inside solid to bc_v */
  //  extrapolate_bc_v(ngbd_np1, vn_nodes, phi_np1);

  // 4) scalar field smoke (if needed)
  if(smoke!=NULL)
  {
    // if smoke_np1 already exists but is no longer useful, destroy it...
    if((smoke_np1!=NULL) && !smoke_np1_is_still_valid){
      ierr = VecDestroy(smoke_np1); CHKERRXX(ierr); smoke_np1 = NULL; }
    if(smoke_np1 == NULL) // calculate it if not already done or if needs to be redone...
    {
      ierr = VecDuplicate(phi_np1, &smoke_np1); CHKERRXX(ierr);
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


    for(int dir=0; dir<P4EST_DIM; ++dir)
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
      check_if_faces_are_well_defined(p4est_np1, ngbd_np1, faces_np1, dir, phi_np1, bc_v[dir].interfaceType(), face_is_well_defined[dir]);

      ierr = VecDestroy(vstar[dir]); CHKERRXX(ierr);
      ierr = VecCreateGhostFaces(p4est_np1, faces_np1, &vstar[dir], dir); CHKERRXX(ierr);

      ierr = VecDestroy(vnp1[dir]); CHKERRXX(ierr);
      ierr = VecCreateGhostFaces(p4est_np1, faces_np1, &vnp1[dir], dir); CHKERRXX(ierr);
    }
    // finish communicating ghost values for hodge and dxyz_hodge
    ierr = VecGhostUpdateEnd(hodge, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    for (short dir = 0; dir < P4EST_DIM; ++dir) {
      ierr = VecGhostUpdateEnd  (dxyz_hodge[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }
  }
  else if(level_set!=NULL)
  {
    // the grid is unchanged but the levelset might have changed, hence possibly affecting the face_is_well_defined vectors --> the solvers cannot be reused safely
    for (short dir = 0; dir < P4EST_DIM; ++dir)
      check_if_faces_are_well_defined(p4est_np1, ngbd_np1, faces_np1, dir, phi_np1, bc_v[dir].interfaceType(), face_is_well_defined[dir]);
  }

  /* update the variables */
  if(p4est_nm1!=p4est_n)
    p4est_destroy(p4est_nm1);
  p4est_nm1 = p4est_n; p4est_n = p4est_np1;
  if(ghost_nm1!=ghost_n)
    p4est_ghost_destroy(ghost_nm1);
  ghost_nm1 = ghost_n; ghost_n = ghost_np1;
  if(nodes_nm1!=nodes_n)
    p4est_nodes_destroy(nodes_nm1);
  nodes_nm1 = nodes_n; nodes_n = nodes_np1;
  if(hierarchy_nm1!=hierarchy_n)
    delete hierarchy_nm1;
  hierarchy_nm1 = hierarchy_n; hierarchy_n = hierarchy_np1;
  if(ngbd_nm1!= ngbd_n)
    delete ngbd_nm1;
  ngbd_nm1 = ngbd_n; ngbd_n = ngbd_np1;
  if(ngbd_c!= ngbd_c_np1)
    delete ngbd_c;
  ngbd_c = ngbd_c_np1;
  if(faces_n!= faces_np1)
    delete faces_n;
  faces_n = faces_np1;

  semi_lagrangian_backtrace_is_done         = false;
  interpolators_from_face_to_nodes_are_set  = interpolators_from_face_to_nodes_are_set && grid_is_unchanged;
  ierr = PetscLogEventEnd(log_my_p4est_navier_stokes_update, 0, 0, 0, 0); CHKERRXX(ierr);

  return (grid_is_unchanged && (level_set == NULL));
}

void my_p4est_navier_stokes_t::compute_pressure()
{
  PetscErrorCode ierr;

  double alpha = sl_order==1 ? 1 : (2*dt_n+dt_nm1)/(dt_n+dt_nm1);

  const double *hodge_p;
  ierr = VecGetArrayRead(hodge, &hodge_p); CHKERRXX(ierr);

  double *pressure_p;
  ierr = VecGetArray(pressure, &pressure_p); CHKERRXX(ierr);

  for(p4est_topidx_t tree_idx=p4est_n->first_local_tree; tree_idx<=p4est_n->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est_n->trees, tree_idx);
    for(size_t q_idx=0; q_idx<tree->quadrants.elem_count; ++q_idx)
    {
      p4est_locidx_t quad_idx = q_idx+tree->quadrants_offset;
      pressure_p[quad_idx] = alpha*rho/dt_n*hodge_p[quad_idx] - mu*compute_divergence(quad_idx, tree_idx);
    }
  }

  ierr = VecRestoreArray(pressure, &pressure_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(hodge, &hodge_p); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(pressure, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (pressure, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  if(bc_pressure->interfaceType()!=NOINTERFACE)
  {
    my_p4est_level_set_cells_t lsc(ngbd_c, ngbd_n);
    lsc.extend_Over_Interface(phi, pressure, bc_pressure, 2, 3);
  }
}


void my_p4est_navier_stokes_t::compute_forces(double *f)
{
  PetscErrorCode ierr;

  const double *phi_p;
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);

  Vec forces[P4EST_DIM];
  const double *v_p[P4EST_DIM];
  double *forces_p[P4EST_DIM];

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &forces[dir]); CHKERRXX(ierr);
    ierr = VecGetArrayRead(vnp1_nodes[dir], &v_p[dir]); CHKERRXX(ierr);
    ierr = VecGetArray(forces[dir], &forces_p[dir]); CHKERRXX(ierr);
  }


  quad_neighbor_nodes_of_node_t qnnn;

  for(size_t i=0; i<ngbd_n->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n->get_layer_node(i);
#ifdef P4_TO_P8
    if(fabs(phi_p[n])<10*MAX(dxyz_min[0],dxyz_min[1],dxyz_min[2]))
#else
    if(fabs(phi_p[n])<10*MAX(dxyz_min[0],dxyz_min[1]))
#endif
    {
      ngbd_n->get_neighbors(n, qnnn);

      double nx = -qnnn.dx_central(phi_p);
      double ny = -qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
      double nz = -qnnn.dz_central(phi_p);
      double norm = sqrt(nx*nx + ny*ny + nz*nz);
#else
      double norm = sqrt(nx*nx + ny*ny);
#endif

      nx = norm>EPS ? nx/norm : 0;
      ny = norm>EPS ? ny/norm : 0;
#ifdef P4_TO_P8
      nz = norm>EPS ? nz/norm : 0;
#endif

      double ux = qnnn.dx_central(v_p[0]);
      double uy = qnnn.dy_central(v_p[0]);
      double vx = qnnn.dx_central(v_p[1]);
      double vy = qnnn.dy_central(v_p[1]);

#ifdef P4_TO_P8
      double uz = qnnn.dz_central(v_p[0]);
      double vz = qnnn.dz_central(v_p[1]);
      double wx = qnnn.dx_central(v_p[2]);
      double wy = qnnn.dy_central(v_p[2]);
      double wz = qnnn.dz_central(v_p[2]);
      forces_p[0][n] = 2*mu*ux*nx + mu*(uy+vx)*ny + mu*(uz+wx)*nz;
      forces_p[1][n] = 2*mu*vy*ny + mu*(uy+vx)*nx + mu*(vz+wy)*nz;
      forces_p[2][n] = 2*mu*wz*nz + mu*(uz+wx)*nx + mu*(vz+wy)*ny;
#else
      forces_p[0][n] = 2*mu*ux*nx + mu*(uy+vx)*ny;
      forces_p[1][n] = 2*mu*vy*ny + mu*(uy+vx)*nx;
#endif
    }
    else
    {
      forces_p[0][n] = 0;
      forces_p[1][n] = 0;
#ifdef P4_TO_P8
      forces_p[2][n] = 0;
#endif
    }
  }

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGhostUpdateBegin(forces[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  for(size_t i=0; i<ngbd_n->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n->get_local_node(i);
#ifdef P4_TO_P8
    if(fabs(phi_p[n])<10*MAX(dxyz_min[0],dxyz_min[1],dxyz_min[2]))
#else
    if(fabs(phi_p[n])<10*MAX(dxyz_min[0],dxyz_min[1]))
#endif
    {
      ngbd_n->get_neighbors(n, qnnn);

      double nx = -qnnn.dx_central(phi_p);
      double ny = -qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
      double nz = -qnnn.dz_central(phi_p);
      double norm = sqrt(nx*nx + ny*ny + nz*nz);
#else
      double norm = sqrt(nx*nx + ny*ny);
#endif

      nx = norm>EPS ? nx/norm : 0;
      ny = norm>EPS ? ny/norm : 0;
#ifdef P4_TO_P8
      nz = norm>EPS ? nz/norm : 0;
#endif

      double ux = qnnn.dx_central(v_p[0]);
      double uy = qnnn.dy_central(v_p[0]);
      double vx = qnnn.dx_central(v_p[1]);
      double vy = qnnn.dy_central(v_p[1]);

#ifdef P4_TO_P8
      double uz = qnnn.dz_central(v_p[0]);
      double vz = qnnn.dz_central(v_p[1]);
      double wx = qnnn.dx_central(v_p[2]);
      double wy = qnnn.dy_central(v_p[2]);
      double wz = qnnn.dz_central(v_p[2]);
      forces_p[0][n] = 2*mu*ux*nx + mu*(uy+vx)*ny + mu*(uz+wx)*nz;
      forces_p[1][n] = 2*mu*vy*ny + mu*(uy+vx)*nx + mu*(vz+wy)*nz;
      forces_p[2][n] = 2*mu*wz*nz + mu*(uz+wx)*nx + mu*(vz+wy)*ny;
#else
      forces_p[0][n] = 2*mu*ux*nx + mu*(uy+vx)*ny;
      forces_p[1][n] = 2*mu*vy*ny + mu*(uy+vx)*nx;
#endif
    }
    else
    {
      forces_p[0][n] = 0;
      forces_p[1][n] = 0;
#ifdef P4_TO_P8
      forces_p[2][n] = 0;
#endif
    }
  }

  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArrayRead(vnp1_nodes[dir], &v_p[dir]); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(forces[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecRestoreArray(forces[dir], &forces_p[dir]); CHKERRXX(ierr);
    f[dir] = integrate_over_interface(p4est_n, nodes_n, phi, forces[dir]);
    ierr = VecDestroy(forces[dir]); CHKERRXX(ierr);
  }

  my_p4est_level_set_cells_t lsc(ngbd_c, ngbd_n);
  double integral[P4EST_DIM];
  lsc.integrate_over_interface(phi, pressure, integral);

  for(int dir=0; dir<P4EST_DIM; ++dir)
    f[dir] -= integral[dir];
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

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGetArrayRead(vnp1_nodes[dir], &vn_p[dir]); CHKERRXX(ierr);
  }

  const double *vort_p;
  ierr = VecGetArrayRead(vorticity, &vort_p); CHKERRXX(ierr);

  /* compute the pressure at nodes for visualization */
  Vec pressure_nodes;
  ierr = VecDuplicate(phi, &pressure_nodes); CHKERRXX(ierr);

  my_p4est_interpolation_cells_t interp_c(ngbd_c, ngbd_n);
  for(size_t n=0; n<nodes_n->indep_nodes.elem_count; ++n)
  {
    double xyz[P4EST_DIM];
    node_xyz_fr_n(n, p4est_n, nodes_n, xyz);
    interp_c.add_point(n, xyz);
  }
  interp_c.set_input(pressure, phi, bc_pressure);
  interp_c.interpolate(pressure_nodes);

  const double *pressure_nodes_p;
  ierr = VecGetArrayRead(pressure_nodes, &pressure_nodes_p); CHKERRXX(ierr);

  Vec leaf_level;
  ierr = VecDuplicate(hodge, &leaf_level); CHKERRXX(ierr);
  PetscScalar *l_p;
  ierr = VecGetArray(leaf_level, &l_p); CHKERRXX(ierr);

  for(p4est_topidx_t tree_idx = p4est_n->first_local_tree; tree_idx <= p4est_n->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est_n->trees, tree_idx);
    for(size_t q=0; q<tree->quadrants.elem_count; ++q)
    {
      const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
      l_p[tree->quadrants_offset+q] = quad->level;
    }
  }

  for(size_t q=0; q<ghost_n->ghosts.elem_count; ++q)
  {
    const p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&ghost_n->ghosts, q);
    l_p[p4est_n->local_num_quadrants+q] = quad->level;
  }

  const double *smoke_p;
  if(smoke!=NULL)
  {
    ierr = VecGetArrayRead(smoke, &smoke_p); CHKERRXX(ierr);

    if(with_Q_and_lambda_2_value)
      my_p4est_vtk_write_all(p4est_n, nodes_n, ghost_n,
                             P4EST_TRUE, P4EST_TRUE,
                             6+P4EST_DIM, /* number of VTK_POINT_DATA */
                             1, /* number of VTK_CELL_DATA  */
                             name,
                             VTK_POINT_DATA, "phi", phi_p,
                             VTK_POINT_DATA, "pressure", pressure_nodes_p,
                             VTK_POINT_DATA, "smoke", smoke_p,
                             VTK_POINT_DATA, "vorticity", vort_p,
                             VTK_POINT_DATA, "Q-value", Q_value_nodes_p,
                             VTK_POINT_DATA, "lambda_2", lambda_2_nodes_p,
                             VTK_POINT_DATA, "vx", vn_p[0],
          VTK_POINT_DATA, "vy", vn_p[1],
      #ifdef P4_TO_P8
          VTK_POINT_DATA, "vz", vn_p[2],
      #endif
          VTK_CELL_DATA, "leaf_level", l_p
          );
    else
      my_p4est_vtk_write_all(p4est_n, nodes_n, ghost_n,
                             P4EST_TRUE, P4EST_TRUE,
                             4+P4EST_DIM, /* number of VTK_POINT_DATA */
                             1, /* number of VTK_CELL_DATA  */
                             name,
                             VTK_POINT_DATA, "phi", phi_p,
                             VTK_POINT_DATA, "pressure", pressure_nodes_p,
                             VTK_POINT_DATA, "smoke", smoke_p,
                             VTK_POINT_DATA, "vorticity", vort_p,
                             VTK_POINT_DATA, "vx", vn_p[0],
          VTK_POINT_DATA, "vy", vn_p[1],
    #ifdef P4_TO_P8
          VTK_POINT_DATA, "vz", vn_p[2],
    #endif
          VTK_CELL_DATA, "leaf_level", l_p
          );
    ierr = VecRestoreArrayRead(smoke, &smoke_p); CHKERRXX(ierr);
  }
  else
  {
    if(with_Q_and_lambda_2_value)
      my_p4est_vtk_write_all(p4est_n, nodes_n, ghost_n,
                             P4EST_TRUE, P4EST_TRUE,
                             5+P4EST_DIM, /* number of VTK_POINT_DATA */
                             1, /* number of VTK_CELL_DATA  */
                             name,
                             VTK_POINT_DATA, "phi", phi_p,
                             VTK_POINT_DATA, "pressure", pressure_nodes_p,
                             VTK_POINT_DATA, "vorticity", vort_p,
                             VTK_POINT_DATA, "Q-value", Q_value_nodes_p,
                             VTK_POINT_DATA, "lambda_2", lambda_2_nodes_p,
                             VTK_POINT_DATA, "vx", vn_p[0],
          VTK_POINT_DATA, "vy", vn_p[1],
    #ifdef P4_TO_P8
          VTK_POINT_DATA, "vz", vn_p[2],
    #endif
          VTK_CELL_DATA, "leaf_level", l_p
          );
    else
      my_p4est_vtk_write_all(p4est_n, nodes_n, ghost_n,
                             P4EST_TRUE, P4EST_TRUE,
                             3+P4EST_DIM, /* number of VTK_POINT_DATA */
                             1, /* number of VTK_CELL_DATA  */
                             name,
                             VTK_POINT_DATA, "phi", phi_p,
                             VTK_POINT_DATA, "pressure", pressure_nodes_p,
                             VTK_POINT_DATA, "vorticity", vort_p,
                             VTK_POINT_DATA, "vx", vn_p[0],
          VTK_POINT_DATA, "vy", vn_p[1],
    #ifdef P4_TO_P8
          VTK_POINT_DATA, "vz", vn_p[2],
    #endif
          VTK_CELL_DATA, "leaf_level", l_p
          );
  }

  ierr = VecRestoreArray(leaf_level, &l_p); CHKERRXX(ierr);
  ierr = VecDestroy(leaf_level); CHKERRXX(ierr);

  ierr = VecRestoreArrayRead(phi  , &phi_p  ); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(hodge, &hodge_p); CHKERRXX(ierr);

  ierr = VecRestoreArrayRead(vorticity, &vort_p); CHKERRXX(ierr);

  ierr = VecRestoreArrayRead(pressure_nodes, &pressure_nodes_p); CHKERRXX(ierr);
  ierr = VecDestroy(pressure_nodes); CHKERRXX(ierr);

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
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
  double *v2c = p4est_n->connectivity->vertices;
  p4est_topidx_t *t2v = p4est_n->connectivity->tree_to_vertex;

  const double size_of_tree         = (v2c[3*t2v[P4EST_CHILDREN*0 + P4EST_CHILDREN - 1] + dir] - v2c[3*t2v[P4EST_CHILDREN*0 + 0] + dir]);
  const double coarsest_cell_size   = size_of_tree/((double) (1<<data->min_lvl));
  const double comparison_threshold = 0.5*size_of_tree/((double) (1<<data->max_lvl));

#ifdef CASL_THROWS
  if((section < xyz_min[dir]) || (section > xyz_max[dir]))
    throw std::invalid_argument("my_p4est_navier_stokes_t::global_mass_flow_through_slice: the slice section must be in the computational domain!");
#endif
  int tree_dim_idx = (int) floor((section-xyz_min[dir])/size_of_tree);
  double should_be_integer = (section-(xyz_min[dir] + tree_dim_idx*size_of_tree))/coarsest_cell_size;
  if(fabs(should_be_integer - ((int) should_be_integer)) > 1e-6)
  {
#ifdef CASL_THROWS
    throw std::invalid_argument("my_p4est_navier_stokes_t::global_mass_flow_through_slice: the mass flux can be evaluated only through a slice in the \n computational domain that coincides with cell faces of the coarsest cells: choose a valid section!");
#else
    section = xyz_min[dir] + tree_dim_idx*size_of_tree + ((int) should_be_integer)*(section-(xyz_min[dir] + tree_dim_idx*size_of_tree))/coarsest_cell_size;
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

    bool check_for_periodic_wrapping = is_periodic(p4est_n, dir) && ((fabs(face_coordinate - xyz_min[dir]) < comparison_threshold) || (fabs(face_coordinate - xyz_max[dir]) < comparison_threshold));
    if(check_for_periodic_wrapping && ((fabs(section - xyz_min[dir]) < comparison_threshold) || (fabs(section - xyz_max[dir]) < comparison_threshold)))
      face_coordinate = section;
    if(fabs(face_coordinate - section) < comparison_threshold)
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
  for (short dd = 0; dd < P4EST_DIM; ++dd) {
    ierr = VecGetArrayRead(vnp1[dd], &vnp1_p[dd]); CHKERRXX(ierr);
  }

  for (short dir = 0; dir < P4EST_DIM; ++dir) {
    // initialize calculation of the force component
    wall_force[dir] = 0.0;
    for (p4est_locidx_t normal_face_idx = 0; normal_face_idx < faces_n->num_local[dir]; ++normal_face_idx) {
      p4est_locidx_t quad_idx;
      p4est_topidx_t tree_idx;
      faces_n->f2q(normal_face_idx, dir, quad_idx, tree_idx);
      p4est_quadrant_t *quad;
      if(quad_idx<p4est_n->local_num_quadrants)
      {
        p4est_tree_t *tree = p4est_tree_array_index(p4est_n->trees, tree_idx);
        quad = p4est_quadrant_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);
      }
      else
        quad = p4est_quadrant_array_index(&ghost_n->ghosts, quad_idx-p4est_n->local_num_quadrants);

      bool is_quad_wall[P4EST_FACES];
      bool no_wall = true;
      for (short ff = 0; ff < P4EST_FACES; ++ff)
      {
        is_quad_wall[ff] = is_quad_Wall(p4est_n,tree_idx, quad, ff);
        no_wall = no_wall && !is_quad_wall[ff];
      }

      if(no_wall)
        continue;

      faces_n->xyz_fr_f(normal_face_idx, dir, xyz_f);
      int tmp = ((faces_n->q2f(quad_idx, 2*dir)==normal_face_idx)? 0 : 1);

      for (short wall_dir = 0; wall_dir < P4EST_DIM; ++wall_dir) {
        if(is_quad_wall[2*wall_dir] || is_quad_wall[2*wall_dir+1])
        {
          // if the normal face is parallel to the considered wall normal, then it MUST be the wall face, otherwise skip it (the appropriate one is another one)
          if(dir == wall_dir && !((tmp == 0)? is_quad_wall[2*wall_dir] : is_quad_wall[2*wall_dir+1]))
            continue;

          // if the normal face is a wall face, and if it is a no-slip wall face (i.e. xyz_w = xyz_f)
          if(dir == wall_dir && is_no_slip(xyz_f)) // always in the domain boundary by definition in this case
          {
            double fraction_noslip = 0.0;
            for (short dd = 0; dd < P4EST_DIM; ++dd)
              xyz_stencil[dd] = xyz_f[dd];
            for (short ii = -1; ii < 2; ii+=2) {
              int first_transverse_dir = (dir+1)%P4EST_DIM;
              xyz_stencil[first_transverse_dir] = xyz_f[first_transverse_dir] + ((double) ii)*0.5*rr*dxyz_min[first_transverse_dir]*((double) (1<<(data->max_lvl - quad->level)));
#ifdef P4_TO_P8
              if(is_no_slip(xyz_stencil)) // always in the domain boundary by definition in this case
              {
                int second_transverse_dir = (dir+2)%P4EST_DIM;
                for (short jj = -1; jj < 2; jj+=2) {
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
            p4est_locidx_t opposite_face_idx = faces_n->q2f(quad_idx, 2*dir+1-tmp);
            P4EST_ASSERT(opposite_face_idx != NO_VELOCITY);
            double dvdirddir = ((tmp==1)?+1.0:-1.0)*(vnp1_p[dir][normal_face_idx] - vnp1_p[dir][opposite_face_idx])/(dxyz_min[dir]*((double) (1<<(data->max_lvl - quad->level))));
            element_drag += 2*mu*dvdirddir;
            element_drag *= faces_n->face_area(normal_face_idx, dir)*fraction_noslip;
            wall_force[dir] += ((tmp == 1)?+1.0:-1.0)*element_drag;
          }
          if(dir != wall_dir)
          {
            for (short kk = 0; kk < 2; ++kk) {
              if(!is_quad_wall[2*wall_dir+kk])
                continue; // to handle dumbass cases with only 1 cell in transverse direction that some insane person might want to try one day...
              for (short dd = 0; dd < P4EST_DIM; ++dd)
                xyz_w[dd] = xyz_f[dd];
              xyz_w[wall_dir] += ((double) (2*kk-1))*0.5*dxyz_min[wall_dir]*((double) (1<<(data->max_lvl - quad->level)));
              if(is_no_slip(xyz_w)) // always in the domain boundary by definition as wall
              {
                // first term: mu*dv[dir]/dwall_dir
                double element_drag = mu*((double) (1-2*kk))*(vnp1_p[dir][normal_face_idx] - bc_v[dir].wallValue(xyz_w))/(0.5*dxyz_min[wall_dir]*((double) (1<<(data->max_lvl - quad->level))));
                // second term: dv[wall_dir]/ddir
                double transverse_derivative = 0.0;
                double fraction_noslip = 0.0;
                unsigned int n_terms = 0;
                for (short dd = 0; dd < P4EST_DIM; ++dd)
                  xyz_stencil[dd] = xyz_w[dd];
#ifdef P4_TO_P8
                int second_transverse_dir = ((((dir+1)%P4EST_DIM)==wall_dir)? ((dir+2)%P4EST_DIM) : ((dir+1)%P4EST_DIM));
#endif
                for (short ii = -1; ii < 2; ii+=2){
                  xyz_stencil[dir] = xyz_w[dir] + ((double) ii)*0.5*rr*dxyz_min[dir]*((double) (1<<(data->max_lvl - quad->level)));
#ifdef P4_TO_P8
                  if(is_in_domain(xyz_stencil) && is_no_slip(xyz_stencil)) // could be out of the domain (consider a corner in a cavity flow, for instance)
                  {
                    for (short jj = -1; jj < 2; jj+=2) {
                      xyz_stencil[second_transverse_dir] = xyz_w[second_transverse_dir] + ((double) jj)*0.5*rr*dxyz_min[second_transverse_dir]*((double) (1<<(data->max_lvl - quad->level)));
                      fraction_noslip += ((is_in_domain(xyz_stencil) && is_no_slip(xyz_stencil))?0.25:0); // could be out of the domain (consider a corner in a cavity flow, for instance)
                    }
                    xyz_stencil[second_transverse_dir] = xyz_w[second_transverse_dir]; // reset that one
                  }
#else
                  fraction_noslip += ((is_in_domain(xyz_stencil) && is_no_slip(xyz_stencil))?0.5:0.0); // could be out of the domain (consider a corner in a cavity flow, for instance)
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
                wall_force[dir] += ((double) (2*kk-1))*element_drag*fraction_noslip;
              }
            }
          }
        }
      }
    }
  }

  for (short dd = 0; dd < P4EST_DIM; ++dd) {
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
    for (size_t idx = 0; idx < subfolders.size(); ++idx) {
      if(!subfolders[idx].compare(0, 7, "backup_"))
      {
        unsigned int backup_idx;
        sscanf(subfolders[idx].c_str(), "backup_%d", &backup_idx);
        if(backup_idx >= n_saved)
        {
          char full_path[PATH_MAX];
          sprintf(full_path, "%s/%s", path_to_root_directory, subfolders[idx].c_str());
          delete_directory(full_path, p4est_n->mpirank, p4est_n->mpicomm, true);
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
    if ((n_saved > 1) && (n_backup_subfolders == n_saved))
    {
      char full_path_zeroth_index[PATH_MAX];
      sprintf(full_path_zeroth_index, "%s/backup_0", path_to_root_directory);
      // delete the 0th
      delete_directory(full_path_zeroth_index, p4est_n->mpirank, p4est_n->mpicomm, true);
      // shift the others
      for (size_t idx = 1; idx < n_saved; ++idx) {
        char old_name[PATH_MAX], new_name[PATH_MAX];
        sprintf(old_name, "%s/backup_%d", path_to_root_directory, (int) idx);
        sprintf(new_name, "%s/backup_%d", path_to_root_directory, (int) (idx-1));
        rename(old_name, new_name);
      }
      backup_idx = n_saved-1;
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

void my_p4est_navier_stokes_t::fill_or_load_double_parameters(save_or_load flag, PetscReal *data, splitting_criteria_t *splitting_criterion, double &tn)
{
  size_t idx = 0;
  for (short dim = 0; dim < P4EST_DIM; ++dim)
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
  for (short dim = 0; dim < P4EST_DIM; ++dim)
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
  for (short dim = 0; dim < P4EST_DIM; ++dim)
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
  for (short dim = 0; dim < P4EST_DIM; ++dim)
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
      data[idx++] = threshold_split_cell;
      data[idx++] = n_times_dt;
      data[idx++] = smoke_thresh;
      data[idx++] = splitting_criterion->lip;
      break;
    }
    case LOAD:
    {
      mu                        = data[idx++];
      rho                       = data[idx++];
      tn                        = data[idx++];
      dt_n                      = data[idx++];
      dt_nm1                    = data[idx++];
      max_L2_norm_u             = data[idx++];
      uniform_band              = data[idx++];
      threshold_split_cell      = data[idx++];
      n_times_dt                = data[idx++];
      smoke_thresh              = data[idx++];
      splitting_criterion->lip  = data[idx++];
      break;
    }
    default:
      throw std::runtime_error("my_p4est_navier_stokes_t::fill_or_load_double_data: unknown flag value");
      break;
    }
  }
  P4EST_ASSERT(idx==4*P4EST_DIM+11);
}

void my_p4est_navier_stokes_t::fill_or_load_integer_parameters(save_or_load flag, PetscInt *data, splitting_criteria_t* splitting_criterion)
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
    break;
  }
  default:
    throw std::runtime_error("my_p4est_navier_stokes_t::fill_or_load_integer_data: unknown flag value");
    break;
  }
  P4EST_ASSERT(idx==5);
}

void my_p4est_navier_stokes_t::save_or_load_parameters(const char* filename, splitting_criteria_t* splitting_criterion, save_or_load flag, double &tn, const mpi_environment_t* mpi)
{
  PetscErrorCode ierr;
  // dxyz_min, xyz_min, xyz_max, convert_to_xyz, mu, rho, tn, dt_n, dt_nm1, max_L2_norm_u, uniform_band, threshold_split_cell, n_times_dt, smoke_thresh, data->lip
  // that makes 4*P4EST_DIM+11 doubles to save
  PetscReal double_parameters[4*P4EST_DIM+11];
  // P4EST_DIM, refine_with_smoke (converted to int), data->min_lvl, data->max_lvl, sl_order
  // that makes 5 integers
  PetscInt integer_parameters[5];
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
      ierr = PetscBinaryWrite(fd, integer_parameters, 5, PETSC_INT, PETSC_TRUE); CHKERRXX(ierr);
      ierr = PetscBinaryClose(fd); CHKERRXX(ierr);
      // Then we save the double parameters
      sprintf(diskfilename, "%s_doubles", filename);
      fill_or_load_double_parameters(flag, double_parameters, splitting_criterion, tn);
      ierr = PetscBinaryOpen(diskfilename, FILE_MODE_WRITE, &fd); CHKERRXX(ierr);
      ierr = PetscBinaryWrite(fd, double_parameters, 4*P4EST_DIM+11, PETSC_DOUBLE, PETSC_TRUE); CHKERRXX(ierr);
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
      ierr = PetscBinaryOpen(diskfilename, FILE_MODE_READ, &fd); CHKERRXX(ierr);
      ierr = PetscBinaryRead(fd, integer_parameters, 5, PETSC_INT); CHKERRXX(ierr);
      ierr = PetscBinaryClose(fd); CHKERRXX(ierr);
    }
    int mpiret = MPI_Bcast(integer_parameters, 5, MPI_INT, 0, mpi->comm()); SC_CHECK_MPI(mpiret);
    fill_or_load_integer_parameters(flag, integer_parameters, splitting_criterion);
    // Then we save the double parameters
    sprintf(diskfilename, "%s_doubles", filename);
    if(!file_exists(diskfilename))
      throw std::invalid_argument("my_p4est_navier_stokes_t::save_or_load_parameters: the file storing the solver's double parameters could not be found");
    if(mpi->rank() == 0)
    {
      ierr = PetscBinaryOpen(diskfilename, FILE_MODE_READ, &fd); CHKERRXX(ierr);
      ierr = PetscBinaryRead(fd, double_parameters, 4*P4EST_DIM+11, PETSC_DOUBLE); CHKERRXX(ierr);
      ierr = PetscBinaryClose(fd); CHKERRXX(ierr);
    }
    mpiret = MPI_Bcast(double_parameters, 4*P4EST_DIM+11, MPI_DOUBLE, 0, mpi->comm()); SC_CHECK_MPI(mpiret);
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

#ifdef P4_TO_P8
void my_p4est_navier_stokes_t::refine_coarsen_grid_after_restart(const CF_3 *level_set, bool do_reinitialization)
#else
void my_p4est_navier_stokes_t::refine_coarsen_grid_after_restart(const CF_2 *level_set, bool do_reinitialization)
#endif
{
  double dt_nm1_saved             = dt_nm1;
  double dt_n_saved               = dt_n;
  p4est_t* p4est_nm1_saved        = p4est_copy(p4est_nm1, P4EST_FALSE);
  p4est_ghost_t* ghost_nm1_saved  = my_p4est_ghost_new(p4est_nm1_saved, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est_nm1_saved, ghost_nm1_saved);
  P4EST_ASSERT(ghosts_are_equal(ghost_nm1_saved, ghost_nm1));
  p4est_nodes_t* nodes_nm1_saved  = my_p4est_nodes_new(p4est_nm1_saved, ghost_nm1_saved);
  P4EST_ASSERT(nodes_are_equal(p4est_n->mpisize, nodes_nm1, nodes_nm1_saved));

  Vec vnm1_nodes_saved[P4EST_DIM];
  Vec second_derivatives_vnm1_nodes_saved[P4EST_DIM][P4EST_DIM];

  PetscErrorCode ierr;
  for (unsigned short dir = 0; dir < P4EST_DIM; ++dir) {
    Vec from_loc, to_loc;
    ierr = VecGhostGetLocalForm(vn_nodes[dir], &from_loc); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(vnp1_nodes[dir], &to_loc); CHKERRXX(ierr);
    ierr = VecCopy(from_loc, to_loc); CHKERRXX(ierr);

    ierr = VecDuplicate(vnm1_nodes[dir], &vnm1_nodes_saved[dir]); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(vnm1_nodes[dir], &from_loc); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(vnm1_nodes_saved[dir], &to_loc); CHKERRXX(ierr);
    ierr = VecCopy(from_loc, to_loc); CHKERRXX(ierr);
    for (unsigned short dd = 0; dd < P4EST_DIM; ++dd) {
      ierr = VecDuplicate(second_derivatives_vnm1_nodes[dd][dir], &second_derivatives_vnm1_nodes_saved[dd][dir]); CHKERRXX(ierr);
      ierr = VecGhostGetLocalForm(second_derivatives_vnm1_nodes[dd][dir], &from_loc); CHKERRXX(ierr);
      ierr = VecGhostGetLocalForm(second_derivatives_vnm1_nodes_saved[dd][dir], &to_loc); CHKERRXX(ierr);
      ierr = VecCopy(from_loc, to_loc); CHKERRXX(ierr);
    }
  }
  compute_vorticity();

  update_from_tn_to_tnp1(level_set, false, do_reinitialization);

  const p4est_topidx_t* t2v = conn->tree_to_vertex;
  const double* v2c = conn->vertices;
  for (int dir=0; dir<P4EST_DIM; dir++)
    dxyz_min[dir] = (v2c[3*t2v[P4EST_CHILDREN*0+P4EST_CHILDREN-1] + dir]-xyz_min[dir]) / (1<<(((splitting_criteria_t*) p4est_n->user_pointer)->max_lvl));
  dt_nm1    = dt_nm1_saved;
  dt_n      = dt_n_saved;
  if(p4est_nm1!=p4est_n)
    p4est_destroy(p4est_nm1);
  p4est_nm1 = p4est_nm1_saved;
  if(ghost_nm1!=ghost_n)
    p4est_ghost_destroy(ghost_nm1);
  ghost_nm1 = ghost_nm1_saved;
  if(nodes_nm1!= nodes_nm1_saved)
    p4est_nodes_destroy(nodes_nm1);
  nodes_nm1 = nodes_nm1_saved;
  if(hierarchy_nm1!=hierarchy_n)
    delete  hierarchy_nm1;
  hierarchy_nm1 = new my_p4est_hierarchy_t(p4est_nm1, ghost_nm1, brick);
  if(ngbd_nm1!=ngbd_n)
    delete  ngbd_nm1;
  ngbd_nm1 = new my_p4est_node_neighbors_t(hierarchy_nm1, nodes_nm1);

  for (unsigned short dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecDestroy(vnm1_nodes[dir]); CHKERRXX(ierr);
    vnm1_nodes[dir] = vnm1_nodes_saved[dir];
    for (unsigned short dd = 0; dd < P4EST_DIM; ++dd) {
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
  if(p4est_nm1!=p4est_n)
    memory_used += p4est_memory_used(p4est_nm1);
  memory_used += p4est_memory_used(p4est_n);
  if(ghost_nm1!=ghost_n)
    memory_used += p4est_ghost_memory_used(ghost_nm1);
  memory_used += p4est_ghost_memory_used(ghost_n);
  if(hierarchy_nm1!=hierarchy_n)
    memory_used += hierarchy_nm1->memory_estimate();
  memory_used+= hierarchy_n->memory_estimate();
  if(ngbd_nm1!=ngbd_n)
    memory_used += ngbd_nm1->memory_estimate();
  memory_used += ngbd_n->memory_estimate();
  // cell-neighbors memory footage is negligible, only contains pointers to other objects...
  memory_used += faces_n->memory_estimate();

  // internal variables dxyz_min, xyz_min, xyz_max, convert_to_xyz
  memory_used += 4*P4EST_DIM*sizeof (double);
  // other internal variables
  memory_used += sizeof (mu) + sizeof (rho) + sizeof (dt_n) + sizeof (dt_nm1) +
      sizeof (max_L2_norm_u) + sizeof (uniform_band) + sizeof (threshold_split_cell) +
      sizeof (n_times_dt) + sizeof (dt_updated) + sizeof (refine_with_smoke) + sizeof (smoke_thresh) +
      sizeof (sl_order);
  // xyz_n, xyz_nm1, face interpolators
  memory_used += sizeof(semi_lagrangian_backtrace_is_done) + sizeof(interpolators_from_face_to_nodes_are_set);
  for (unsigned short dir = 0; dir < P4EST_DIM; ++dir) {
    memory_used += xyz_n[dir]->size()*sizeof (double);
    memory_used += xyz_nm1[dir]->size()*sizeof (double);
    for (unsigned int k = 0; k < interpolator_from_face_to_nodes[dir].size(); ++k)
      memory_used += interpolator_from_face_to_nodes[dir][k].size()*sizeof (face_interpolator_element);
  }

  // petsc node vectors at time n: phi, vn_nodes[P4EST_DIM], vnp1_nodes[P4EST_DIM], vorticity, smoke
  memory_used += (1+2*P4EST_DIM+1+((smoke!=NULL)? 1:0))*(nodes_n->indep_nodes.elem_count)*sizeof (PetscScalar);
  // petsc node vectors at time nm1: vnm1_nodes[P4EST_DIM],
  memory_used += P4EST_DIM*(nodes_nm1->indep_nodes.elem_count)*sizeof (PetscScalar);
  // petsc cell vectors at time n: hodge, pressure
  memory_used += 2*(p4est_n->local_num_quadrants + ghost_n->ghosts.elem_count)*sizeof (PetscScalar);
  // petsc face vectors at time n: dxyz_hodge[P4EST_DIM], vstar[P4EST_DIM], vnp1[P4EST_DIM], face_is_well_defined[P4EST_DIM]
  for (unsigned short dim = 0; dim < P4EST_DIM; ++dim)
    memory_used += 4*(faces_n->num_local[dim] + faces_n->num_ghost[dim])*sizeof (PetscScalar);

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &memory_used, 1, MPI_UNSIGNED_LONG, MPI_SUM, p4est_n->mpicomm); SC_CHECK_MPI(mpiret);

  return memory_used;
}

void my_p4est_navier_stokes_t::get_slice_averaged_vnp1_profile(const unsigned short& vel_component, const unsigned short& axis, std::vector<double>& avg_velocity_profile, const double u_scaling)
{
  P4EST_ASSERT((vel_component<P4EST_DIM) && (axis<P4EST_DIM) && (vel_component!=axis) && is_periodic(p4est_n, vel_component));
#ifdef P4_TO_P8
  for (unsigned short dd = 0; dd < P4EST_DIM; ++dd) {
    if((dd == vel_component) || (dd == axis))
      continue;
    P4EST_ASSERT(is_periodic(p4est_n, dd));
  }
#endif
  splitting_criteria_t* data = (splitting_criteria_t*) p4est_n->user_pointer;
  unsigned int ndouble = brick->nxyztrees[axis]*(1<<data->max_lvl); // equivalent number of data for a uniform grid with finest level of refinement
#ifdef P4EST_ENABLE_DEBUG
  avg_velocity_profile.resize(ndouble*(1+1), 0.0); // slice-averaged velocity component + area of the slice
#else
  avg_velocity_profile.resize(ndouble, 0.0); // slice-averaged velocity component
#endif
#ifdef P4_TO_P8
  const double elementary_area = dxyz_min[(axis+1)%P4EST_DIM]*dxyz_min[(axis+2)%P4EST_DIM];
#else
  const double elementary_area = dxyz_min[(axis+1)%P4EST_DIM];
#endif
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
  vector<p4est_quadrant_t> ngbd;
  p4est_quadrant_t quad, nb_quad;

  for (p4est_locidx_t face_idx = 0; face_idx < faces_n->num_local[vel_component]; ++face_idx) {
    faces_n->f2q(face_idx, vel_component, quad_idx, tree_idx);
    const p4est_quadrant_t* quad_ptr;
    P4EST_ASSERT(quad_idx < p4est_n->local_num_quadrants);
    p4est_tree_t* tree = (p4est_tree_t*) sc_array_index(p4est_n->trees, tree_idx);
    quad_ptr = (const p4est_quadrant_t*) sc_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);
    if(faces_n->q2f(quad_idx, 2*vel_component)==face_idx)
    {
      quad = *quad_ptr; quad.p.piggy3.local_num = quad_idx;
      ngbd.clear();
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 2*vel_component);
      /* note that the potential neighbor has to be the same size or bigger and there MUST be a neighbor*/
      P4EST_ASSERT(ngbd.size()==1);
      nb_quad = ngbd[0];
#ifdef P4EST_ENABLE_DEBUG
      nb_tree_idx = ngbd[0].p.piggy3.which_tree;
#endif
    }
    else
    {
      P4EST_ASSERT(faces_n->q2f(quad_idx, 2*vel_component+1)==face_idx);
      quad = *quad_ptr; quad.p.piggy3.local_num = quad_idx;
      ngbd.clear();
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 2*vel_component+1);
      /* note that the potential neighbor has to be the same size or bigger and there MUST be a neighbor*/
      P4EST_ASSERT(ngbd.size()==1);
      nb_quad = ngbd[0];
#ifdef P4EST_ENABLE_DEBUG
      nb_tree_idx = ngbd[0].p.piggy3.which_tree;
#endif
    }
    p4est_topidx_t cartesian_tree_idx_along_axis=-1;
#ifdef P4EST_ENABLE_DEBUG
    p4est_topidx_t cartesian_nb_tree_idx_along_axis=-1;
#endif
    bool is_found = false;
    for (p4est_topidx_t tt = 0; tt < conn->num_trees; ++tt) {
      if (brick->nxyz_to_treeid[tt] == tree_idx)
        cartesian_tree_idx_along_axis = tt;
#ifdef P4EST_ENABLE_DEBUG
      if (brick->nxyz_to_treeid[tt] == nb_tree_idx)
        cartesian_nb_tree_idx_along_axis = tt;
      is_found = is_found || ((cartesian_tree_idx_along_axis!=-1) && (cartesian_nb_tree_idx_along_axis!=-1));
#else
      is_found = is_found || (cartesian_tree_idx_along_axis!=-1);
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
#ifdef P4_TO_P8
      double weighting_area = elementary_area*(1<<(data->max_lvl-quad.level))*((1<<(data->max_lvl-nb_quad.level)) + (1<<(data->max_lvl-quad.level)))*0.5;
#else
      double weighting_area = elementary_area*((1<<(data->max_lvl-nb_quad.level)) + (1<<(data->max_lvl-quad.level)))*0.5;
#endif
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
  const double expected_slice_area = (xyz_max[(axis+1)%P4EST_DIM]-xyz_min[(axis+1)%P4EST_DIM])
    #ifdef P4_TO_P8
      *(xyz_max[(axis+2)%P4EST_DIM]-xyz_min[(axis+2)%P4EST_DIM])
    #endif
      ;
  if(!p4est_n->mpirank)
    for (unsigned int k = 0; k < ndouble; ++k) {
      P4EST_ASSERT(fabs(avg_velocity_profile[ndouble+k] - expected_slice_area) < 10.0*EPS*MAX(avg_velocity_profile[ndouble+k], expected_slice_area));
      avg_velocity_profile[k] /= expected_slice_area;
    }
}

void my_p4est_navier_stokes_t::get_line_averaged_vnp1_profiles(const unsigned short& vel_component, const unsigned short& axis,
                                                               #ifdef P4_TO_P8
                                                               const unsigned short& averaging_direction,
                                                               #endif
                                                               const std::vector<unsigned int>& bin_index, std::vector< std::vector<double> >& avg_velocity_profile, const double u_scaling)
{
  P4EST_ASSERT((vel_component<P4EST_DIM) && (axis<P4EST_DIM) && (vel_component!=axis) && is_periodic(p4est_n, vel_component));
#ifdef P4_TO_P8
  P4EST_ASSERT((averaging_direction<P4EST_DIM) && (averaging_direction!=axis) && is_periodic(p4est_n, averaging_direction));
  unsigned short transverse_direction = (3 & ~axis) & ~averaging_direction; // bitwise binary operation
  P4EST_ASSERT(transverse_direction != averaging_direction);
  P4EST_ASSERT((vel_component == averaging_direction) || (vel_component == transverse_direction));
#else
  unsigned short transverse_direction = ((axis==dir::x) ? dir::y : dir::x);
  P4EST_ASSERT(vel_component == transverse_direction);
#endif
  P4EST_ASSERT((transverse_direction < P4EST_DIM) && (transverse_direction != axis));
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
    avg_velocity_profile[profile_idx].resize(ndouble*(1+1), 0.0); // line-averaged velocity component profiles + averaging lengths
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
  vector<p4est_quadrant_t> ngbd;
  p4est_quadrant_t quad, nb_quad;

  p4est_locidx_t bounds_along_axis[2];
#ifdef P4_TO_P8
  // if (vel_component==averaging_direction)
  unsigned int bounds_transverse_direction[2]; // we declare these variables unsigned int to avoid misbehaved results due to type conversion during the periodic mapping hereunder
#endif
  // else, i.e., if (vel_component==transverse_direction)
  unsigned int logical_idx_of_face; // we declare these variables unsigned int to avoid misbehaved results due to type conversion during the periodic mapping hereunder
  unsigned int negative_coverage, positive_coverage;
  double covering_length;

  for (p4est_locidx_t face_idx = 0; face_idx < faces_n->num_local[vel_component]; ++face_idx) {
    faces_n->f2q(face_idx, vel_component, quad_idx, tree_idx);
    const p4est_quadrant_t* quad_ptr;
    P4EST_ASSERT(quad_idx < p4est_n->local_num_quadrants);
    p4est_tree_t* tree = (p4est_tree_t*) sc_array_index(p4est_n->trees, tree_idx);
    quad_ptr = (const p4est_quadrant_t*) sc_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);
    if(faces_n->q2f(quad_idx, 2*vel_component)==face_idx)
    {
      quad = *quad_ptr; quad.p.piggy3.local_num = quad_idx;
      ngbd.clear();
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 2*vel_component);
      /* note that the potential neighbor has to be the same size or bigger and there MUST be a neighbor*/
      P4EST_ASSERT(ngbd.size()==1);
      nb_quad = ngbd[0];
      nb_tree_idx = ngbd[0].p.piggy3.which_tree;
    }
    else
    {
      P4EST_ASSERT(faces_n->q2f(quad_idx, 2*vel_component+1)==face_idx);
      quad = *quad_ptr; quad.p.piggy3.local_num = quad_idx;
      ngbd.clear();
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 2*vel_component+1);
      /* note that the potential neighbor has to be the same size or bigger and there MUST be a neighbor*/
      P4EST_ASSERT(ngbd.size()==1);
      nb_quad = ngbd[0];
      nb_tree_idx = ngbd[0].p.piggy3.which_tree;
    }
    p4est_topidx_t cartesian_ordered_tree_idx=-1;
    p4est_topidx_t cartesian_ordered_nb_tree_idx=-1;
    bool are_found = false;
    for (p4est_topidx_t tt = 0; tt < conn->num_trees; ++tt) {
      if (brick->nxyz_to_treeid[tt] == tree_idx)
        cartesian_ordered_tree_idx = tt;
      if (brick->nxyz_to_treeid[tt] == nb_tree_idx)
        cartesian_ordered_nb_tree_idx = tt;
      are_found = are_found || ((cartesian_ordered_tree_idx!=-1) && (cartesian_ordered_nb_tree_idx!=-1));
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
    P4EST_ASSERT(cartesian_tree_idx_along_axis==cartesian_nb_tree_idx_along_axis);
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
        logical_idx_of_face                       = (quad.x + ((faces_n->q2f(quad_idx, 2*vel_component+1) == face_idx)? P4EST_QUADRANT_LEN(quad.level) : 0))/(1<<(P4EST_MAXLEVEL - data->max_lvl));
#ifdef P4EST_ENABLE_DEBUG
      cartesian_nb_tree_idx_along_transverse_dir  = cartesian_ordered_nb_tree_idx%brick->nxyztrees[0];
#endif
#else
      logical_idx_of_face                         = (quad.x + ((faces_n->q2f(quad_idx, 2*vel_component+1) == face_idx)? P4EST_QUADRANT_LEN(quad.level) : 0))/(1<<(P4EST_MAXLEVEL - data->max_lvl));
#endif
      break;
    case dir::y:
      cartesian_tree_idx_along_transverse_dir     = (cartesian_ordered_tree_idx/brick->nxyztrees[0])%brick->nxyztrees[1];
#ifdef P4_TO_P8
      if(vel_component == averaging_direction)
        bounds_transverse_direction[0]            = quad.y/(1<<(P4EST_MAXLEVEL - data->max_lvl));
      else
        logical_idx_of_face                       = (quad.y + ((faces_n->q2f(quad_idx, 2*vel_component+1) == face_idx)? P4EST_QUADRANT_LEN(quad.level) : 0))/(1<<(P4EST_MAXLEVEL - data->max_lvl));
#ifdef P4EST_ENABLE_DEBUG
      cartesian_nb_tree_idx_along_transverse_dir  = (cartesian_ordered_nb_tree_idx/brick->nxyztrees[0])%brick->nxyztrees[1];
#endif
#else
      logical_idx_of_face                         = (quad.y + ((faces_n->q2f(quad_idx, 2*vel_component+1) == face_idx)? P4EST_QUADRANT_LEN(quad.level) : 0))/(1<<(P4EST_MAXLEVEL - data->max_lvl));
#endif
      break;
#ifdef P4_TO_P8
    case dir::z:
      cartesian_tree_idx_along_transverse_dir     = cartesian_ordered_tree_idx/(brick->nxyztrees[0]*brick->nxyztrees[1]);
      if(vel_component == averaging_direction)
        bounds_transverse_direction[0]            = quad.z/(1<<(P4EST_MAXLEVEL - data->max_lvl));
      else
        logical_idx_of_face                       = (quad.z + ((faces_n->q2f(quad_idx, 2*vel_component+1) == face_idx)? P4EST_QUADRANT_LEN(quad.level) : 0))/(1<<(P4EST_MAXLEVEL - data->max_lvl));
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
      P4EST_ASSERT(cartesian_tree_idx_along_transverse_dir==cartesian_nb_tree_idx_along_transverse_dir);
      bounds_transverse_direction[0]             += cartesian_tree_idx_along_transverse_dir*(1<<data->max_lvl);
      bounds_transverse_direction[1]              = bounds_transverse_direction[0] + (1<<(data->max_lvl - quad.level));
      covering_length                             = 0.5*dxyz_min[averaging_direction]*((double) (1<<(data->max_lvl-quad.level)) + (double) (1<<(data->max_lvl-nb_quad.level)));
    }
    else
    {
      P4EST_ASSERT(vel_component == transverse_direction);
      logical_idx_of_face                        += cartesian_tree_idx_along_transverse_dir*(1<<data->max_lvl);
      negative_coverage                           = data->max_lvl - ((faces_n->q2f(quad_idx, 2*vel_component+1) == face_idx)? quad.level: nb_quad.level);
      negative_coverage                           = ((negative_coverage > 0)? (1<<(negative_coverage-1)): 0);
      positive_coverage                           = data->max_lvl - ((faces_n->q2f(quad_idx, 2*vel_component+1) == face_idx)? nb_quad.level: quad.level);
      positive_coverage                           = ((positive_coverage > 0)? (1<<(positive_coverage-1)): 0);
      covering_length                             = dxyz_min[averaging_direction]*((double) (1<<(data->max_lvl-quad.level)));
    }
#else
    logical_idx_of_face                          += cartesian_tree_idx_along_transverse_dir*(1<<data->max_lvl);
    negative_coverage                             = data->max_lvl - ((faces_n->q2f(quad_idx, 2*vel_component+1) == face_idx)? quad.level: nb_quad.level);
    negative_coverage                             = ((negative_coverage > 0)? (1<<(negative_coverage-1)): 0);
    positive_coverage                             = data->max_lvl - ((faces_n->q2f(quad_idx, 2*vel_component+1) == face_idx)? nb_quad.level: quad.level);
    positive_coverage                             = ((positive_coverage > 0)? (1<<(positive_coverage-1)): 0);
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
          wrapped_idx = (logical_idx_of_face+((k>logical_idx_of_face)?(((k-logical_idx_of_face)/bin_index.size()+1)*bin_index.size()):0)-k)%bin_index.size(); // avoid negative intermediary result...
          avg_velocity_profile[bin_index.at(wrapped_idx)][idx_in_profile] += ((k == negative_coverage)? 0.5: 1.0)*covering_length*velocity_component_p[face_idx]/u_scaling;
#ifdef P4EST_ENABLE_DEBUG
          avg_velocity_profile[bin_index.at(wrapped_idx)][idx_in_profile+ndouble] += ((k == negative_coverage)? 0.5: 1.0)*covering_length;
#endif
        }
        for (unsigned int k = 1; k <= positive_coverage; ++k) {
          wrapped_idx = (logical_idx_of_face+k)%bin_index.size();
          avg_velocity_profile[bin_index.at(wrapped_idx)][idx_in_profile] += ((k == positive_coverage)? 0.5: 1.0)*covering_length*velocity_component_p[face_idx]/u_scaling;
#ifdef P4EST_ENABLE_DEBUG
          avg_velocity_profile[bin_index.at(wrapped_idx)][idx_in_profile+ndouble] += ((k == positive_coverage)? 0.5: 1.0)*covering_length;
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
        wrapped_idx = (logical_idx_of_face+((k>logical_idx_of_face)?(((k-logical_idx_of_face)/bin_index.size()+1)*bin_index.size()):0)-k)%bin_index.size(); // avoid negative intermediary result...
        avg_velocity_profile[bin_index.at(wrapped_idx)][idx_in_profile] += ((k == negative_coverage)? 0.5: 1.0)*covering_length*velocity_component_p[face_idx]/u_scaling;
#ifdef P4EST_ENABLE_DEBUG
        avg_velocity_profile[bin_index.at(wrapped_idx)][idx_in_profile+ndouble] += ((k == negative_coverage)? 0.5: 1.0)*covering_length;
#endif
      }
      for (unsigned int k = 1; k <= positive_coverage; ++k) {
        wrapped_idx = (logical_idx_of_face+k)%bin_index.size();
        avg_velocity_profile[bin_index.at(wrapped_idx)][idx_in_profile] += ((k == positive_coverage)? 0.5: 1.0)*covering_length*velocity_component_p[face_idx]/u_scaling;
#ifdef P4EST_ENABLE_DEBUG
        avg_velocity_profile[bin_index.at(wrapped_idx)][idx_in_profile+ndouble] += ((k == positive_coverage)? 0.5: 1.0)*covering_length;
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

