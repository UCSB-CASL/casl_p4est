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
#else
#include <src/my_p4est_poisson_cells.h>
#include <src/my_p4est_poisson_faces.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_level_set_cells.h>
#include <src/my_p4est_level_set_faces.h>
#include <src/my_p4est_trajectory_of_point.h>
#include <src/my_p4est_vtk.h>
#include <p4est_extended.h>
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
  this->uniform_band = uniform_band;
  this->threshold = threshold;
  this->max_L2_norm_u = max_L2_norm_u;
  this->smoke_thresh = smoke_thresh;
}


void my_p4est_navier_stokes_t::splitting_criteria_vorticity_t::tag_quadrant(p4est_t *p4est, p4est_locidx_t quad_idx, p4est_topidx_t tree_idx,
                                                                            my_p4est_interpolation_nodes_t &phi, my_p4est_interpolation_nodes_t &vor,
                                                                            my_p4est_interpolation_nodes_t *smo)
{
  p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
  p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);

  if (quad->level < min_lvl)
    quad->p.user_int = REFINE_QUADRANT;

  else if (quad->level > max_lvl)
    quad->p.user_int = COARSEN_QUADRANT;

  else
  {
    p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0];
    p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + P4EST_CHILDREN-1];
    double xmin = p4est->connectivity->vertices[3*vm + 0];
    double ymin = p4est->connectivity->vertices[3*vm + 1];
    double xmax = p4est->connectivity->vertices[3*vp + 0];
    double ymax = p4est->connectivity->vertices[3*vp + 1];
  #ifdef P4_TO_P8
    double zmin = p4est->connectivity->vertices[3*vm + 2];
    double zmax = p4est->connectivity->vertices[3*vp + 2];
  #endif

    double dmin = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
    double dx = (xmax-xmin) * dmin;
    double dy = (ymax-ymin) * dmin;
#ifdef P4_TO_P8
    double dz = (zmax-zmin) * dmin;
#endif

#ifdef P4_TO_P8
    double d = sqrt(dx*dx + dy*dy + dz*dz);
#else
    double d = sqrt(dx*dx + dy*dy);
#endif

#ifdef P4_TO_P8
    double dxyz_min = MAX(xmax-xmin, ymax-ymin, zmax-zmin) / (1<<max_lvl);
#else
    double dxyz_min = MAX(xmax-xmin, ymax-ymin) / (1<<max_lvl);
#endif

    double x = (xmax-xmin)*quad_x_fr_i(quad) + xmin;
    double y = (ymax-ymin)*quad_y_fr_j(quad) + ymin;
#ifdef P4_TO_P8
    double z = (zmax-zmin)*quad_z_fr_k(quad) + zmin;
#endif

    bool all_pos = true;
    bool cor_vort = true;
    bool cor_band = true;
    bool cor_intf = true;
    bool cor_smok = true;
    for(int i=0; i<2; ++i)
      for(int j=0; j<2; ++j)
#ifdef P4_TO_P8
        for(int k=0; k<2; ++k)
        {
        cor_vort = cor_vort && fabs(vor(x+i*dx, y+j*dy, z+k*dz))*2*MAX(dx,dy,dz)/max_L2_norm_u<threshold;
        cor_band = cor_band && fabs(phi(x+i*dx, y+j*dy, z+k*dz))>uniform_band*dxyz_min;
        cor_intf = cor_intf && fabs(phi(x+i*dx, y+j*dy, z+k*dz))>=lip*2*d;
        if(smo!=NULL)
          cor_smok = cor_smok && (*smo)(x+i*dx, y+j*dy, z+k*dz)<smoke_thresh;
        all_pos = all_pos && phi(x+i*dx, y+j*dy, z+k*dz)> MAX(2.0, uniform_band)*dxyz_min; // [RAPHAEL:] modified to enforce at least two layers of finest level in positive domain as well (better extrapolation etc.), also required for Neumann BC in the face-solvers
#else
      {
        cor_vort = cor_vort && fabs(vor(x+i*dx, y+j*dy))*2*MAX(dx,dy)/max_L2_norm_u<threshold;
        cor_band = cor_band && fabs(phi(x+i*dx, y+j*dy))>uniform_band*dxyz_min;
        cor_intf = cor_intf && fabs(phi(x+i*dx, y+j*dy))>=lip*2*d;
        if(smo!=NULL)
          cor_smok = cor_smok && (*smo)(x+i*dx, y+j*dy)<smoke_thresh;
        all_pos = all_pos && phi(x+i*dx, y+j*dy)> MAX(2.0, uniform_band)*dxyz_min; // [RAPHAEL:] modified to enforce two layers of finest level in positive domain as well (better extrapolation etc.), also required for Neumann BC in the face-solvers
#endif
      }

    bool coarsen = true;
    coarsen = (cor_vort && cor_band && cor_intf && cor_smok) || all_pos;
//    coarsen = ((cor_vort && cor_band && cor_smok) || all_pos) && cor_intf;
    coarsen = coarsen && quad->level > min_lvl;

    bool is_neg = false;
    bool ref_vort = false;
    bool ref_band = false;
    bool ref_intf = false;
    bool ref_smok = false;
    for(int i=0; i<3; ++i)
      for(int j=0; j<3; ++j)
#ifdef P4_TO_P8
        for(int k=0; k<3; ++k)
        {
          ref_vort = ref_vort || fabs(vor(x+i*dx/2, y+j*dy/2, z+k*dz/2))*MAX(dx,dy,dz)/max_L2_norm_u>threshold;
          ref_band = ref_band || fabs(phi(x+i*dx/2, y+j*dy/2, z+k*dz/2))<uniform_band*dxyz_min;
          ref_intf = ref_intf || fabs(phi(x+i*dx/2, y+j*dy/2, z+k*dz/2))<=lip*d;
          if(smo!=NULL)
            ref_smok = ref_smok || (*smo)(x+i*dx/2, y+j*dy/2, z+k*dz/2)>=smoke_thresh;
          is_neg = is_neg || phi(x+i*dx/2, y+j*dy/2, z+k*dz/2)< MAX(2.0, uniform_band)*dxyz_min; // [RAPHAEL:] same comment as before
#else
      {
        ref_vort = ref_vort || fabs(vor(x+i*dx/2, y+j*dy/2))*MAX(dx,dy)/max_L2_norm_u>threshold;
        ref_band = ref_band || fabs(phi(x+i*dx/2, y+j*dy/2))<uniform_band*dxyz_min;
        ref_intf = ref_intf || fabs(phi(x+i*dx/2, y+j*dy/2))<=lip*d;
        if(smo!=NULL)
          ref_smok = ref_smok || (*smo)(x+i*dx/2, y+j*dy/2)>=smoke_thresh;
        is_neg = is_neg || phi(x+i*dx/2, y+j*dy/2)< MAX(2.0, uniform_band)*dxyz_min; // [RAPHAEL:] same comment as before
#endif
      }

    bool refine = false;
    refine = (ref_vort || ref_band || ref_intf || ref_smok) && is_neg;
//    refine = ((ref_vort || ref_band || ref_smok) && is_neg) || ref_intf;
    refine = refine && quad->level < max_lvl;

    if (refine)
      quad->p.user_int = REFINE_QUADRANT;

    else if (coarsen)
      quad->p.user_int = COARSEN_QUADRANT;

    else
      quad->p.user_int = SKIP_QUADRANT;
  }
}


bool my_p4est_navier_stokes_t::splitting_criteria_vorticity_t::refine_and_coarsen(p4est_t* p4est, my_p4est_node_neighbors_t *ngbd_n, Vec phi, Vec vorticity, Vec smoke)
{
  my_p4est_interpolation_nodes_t interp_phi(ngbd_n);
  interp_phi.set_input(phi, linear);

  my_p4est_interpolation_nodes_t interp_vorticity(ngbd_n);
  interp_vorticity.set_input(vorticity, linear);

  my_p4est_interpolation_nodes_t *interp_smoke = NULL;
  if(smoke!=NULL)
  {
    interp_smoke = new my_p4est_interpolation_nodes_t(ngbd_n);
    interp_smoke->set_input(smoke, linear);
  }

  /* tag the quadrants that need to be refined or coarsened */
  for (p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx) {
    p4est_tree_t* tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for (size_t q = 0; q <tree->quadrants.elem_count; ++q) {
      p4est_locidx_t quad_idx  = q + tree->quadrants_offset;
      tag_quadrant(p4est, quad_idx, tree_idx, interp_phi, interp_vorticity, interp_smoke);
    }
  }

  if(smoke!=NULL)
    delete interp_smoke;

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
    ngbd_c(faces->ngbd_c), faces_n(faces),
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
  ierr = VecGhostRestoreLocalForm(phi, &vec_loc); CHKERRXX(ierr);

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

    ierr = VecCreateGhostFaces(p4est_n, faces_n, &face_is_well_defined[dir], dir); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(face_is_well_defined[dir], &vec_loc); CHKERRXX(ierr);
    ierr = VecSet(vec_loc, 1); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(face_is_well_defined[dir], &vec_loc); CHKERRXX(ierr);

    ierr = VecDuplicate(face_is_well_defined[dir], &dxyz_hodge[dir]); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(dxyz_hodge[dir], &vec_loc); CHKERRXX(ierr);
    ierr = VecSet(vec_loc, 0); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(dxyz_hodge[dir], &vec_loc); CHKERRXX(ierr);
  }

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
    if(dxyz_hodge[dir]!=NULL) { ierr = VecDestroy(dxyz_hodge[dir]); CHKERRXX(ierr); }
    if(vstar[dir]!=NULL)      { ierr = VecDestroy(vstar[dir]);      CHKERRXX(ierr); }
    if(vnp1[dir]!=NULL)       { ierr = VecDestroy(vnp1[dir]);       CHKERRXX(ierr); }
    if(vnm1_nodes[dir]!=NULL) { ierr = VecDestroy(vnm1_nodes[dir]); CHKERRXX(ierr); }
    if(vn_nodes[dir]!=NULL)   { ierr = VecDestroy(vn_nodes[dir]);   CHKERRXX(ierr); }
    if(vnp1_nodes[dir]!=NULL) { ierr = VecDestroy(vnp1_nodes[dir]); CHKERRXX(ierr); }
    if(face_is_well_defined[dir]!=NULL) { ierr = VecDestroy(face_is_well_defined[dir]); CHKERRXX(ierr); }
  }

  if(interp_phi!=NULL) delete interp_phi;


  delete ngbd_nm1;
  delete hierarchy_nm1;
  p4est_nodes_destroy(nodes_nm1);
  p4est_ghost_destroy(ghost_nm1);
  p4est_destroy(p4est_nm1);

  delete faces_n;
  delete ngbd_c;
  delete ngbd_n;
  delete hierarchy_n;
  p4est_nodes_destroy(nodes_n);
  p4est_ghost_destroy(ghost_n);
  p4est_destroy(p4est_n);

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
    this->vn_nodes[dir]   = vn_nodes[dir];
    this->vnm1_nodes[dir] = vnm1_nodes[dir];
    ierr = VecDuplicate(vn_nodes[dir], &vnp1_nodes [dir]); CHKERRXX(ierr);

    ierr = VecDuplicate(face_is_well_defined[dir], &vstar[dir]); CHKERRXX(ierr);
    ierr = VecDuplicate(face_is_well_defined[dir], &vnp1[dir]); CHKERRXX(ierr);
  }

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


    ierr = VecDuplicate(vn_nodes[dir], &vnp1_nodes [dir]); CHKERRXX(ierr);
    ierr = VecDuplicate(face_is_well_defined[dir], &vstar[dir]); CHKERRXX(ierr);
    ierr = VecDuplicate(face_is_well_defined[dir], &vnp1[dir]); CHKERRXX(ierr);
  }

  ierr = VecDuplicate(phi, &vorticity); CHKERRXX(ierr);
}

void my_p4est_navier_stokes_t::set_vstar(Vec *vstar)
{
  for(int dir=0; dir<P4EST_DIM; ++dir)
    this->vstar[dir] = vstar[dir];
}

void my_p4est_navier_stokes_t::set_hodge(Vec hodge)
{
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

void my_p4est_navier_stokes_t::solve_viscosity()
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
  std::vector<double> xyz_nm1[P4EST_DIM];
  std::vector<double> xyz_n  [P4EST_DIM];
  Vec rhs[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    /* backtrace the nodes with semi-Lagrangian / BDF scheme */
    for(int dd=0; dd<P4EST_DIM; ++dd)
    {
      if(sl_order==2) xyz_nm1[dd].resize(faces_n->num_local[dir]);
      xyz_n  [dd].resize(faces_n->num_local[dir]);
    }
    if(sl_order==2)
      trajectory_from_np1_to_nm1(p4est_n, faces_n, ngbd_nm1, ngbd_n, vnm1_nodes, vn_nodes, dt_nm1, dt_n, xyz_nm1, xyz_n, dir);
    else
      trajectory_from_np1_to_n(p4est_n, faces_n, ngbd_nm1, ngbd_n, vnm1_nodes, vn_nodes, dt_nm1, dt_n, xyz_n, dir);

    /* find the velocity at the backtraced points */
    my_p4est_interpolation_nodes_t interp_nm1(ngbd_nm1);
    my_p4est_interpolation_nodes_t interp_n  (ngbd_n  );
    for(p4est_locidx_t f_idx=0; f_idx<faces_n->num_local[dir]; ++f_idx)
    {
      double xyz_tmp[P4EST_DIM];

      if(sl_order==2)
      {
#ifdef P4_TO_P8
        xyz_tmp[0] = xyz_nm1[0][f_idx]; xyz_tmp[1] = xyz_nm1[1][f_idx]; xyz_tmp[2] = xyz_nm1[2][f_idx];
#else
        xyz_tmp[0] = xyz_nm1[0][f_idx]; xyz_tmp[1] = xyz_nm1[1][f_idx];
#endif
        interp_nm1.add_point(f_idx, xyz_tmp);
      }

#ifdef P4_TO_P8
      xyz_tmp[0] = xyz_n[0][f_idx]; xyz_tmp[1] = xyz_n[1][f_idx]; xyz_tmp[2] = xyz_n[2][f_idx];
#else
      xyz_tmp[0] = xyz_n[0][f_idx]; xyz_tmp[1] = xyz_n[1][f_idx];
#endif
      interp_n.add_point(f_idx, xyz_tmp);
    }

    std::vector<double> vnm1_faces(faces_n->num_local[dir]);
    if(sl_order==2)
    {
      vnm1_faces.resize(faces_n->num_local[dir]);
      interp_nm1.set_input(vnm1_nodes[dir], quadratic);
      interp_nm1.interpolate(vnm1_faces.data());
    }

    std::vector<double> vn_faces(faces_n->num_local[dir]);
    interp_n.set_input(vn_nodes[dir], quadratic);
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

  my_p4est_poisson_faces_t solver(faces_n, ngbd_n);
  solver.set_phi(phi);
  solver.set_mu(mu);
  solver.set_diagonal(alpha * rho/dt_n);
  solver.set_bc(bc_v, dxyz_hodge, face_is_well_defined);
  solver.set_rhs(rhs);
#if defined(COMET) || defined(STAMPEDE) || defined(POD_CLUSTER)
  solver.set_compute_partition_on_the_fly(true);
#else
  solver.set_compute_partition_on_the_fly(false);
#endif

  solver.solve(vstar);

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
void my_p4est_navier_stokes_t::solve_projection()
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
  my_p4est_poisson_cells_t solver(ngbd_c, ngbd_n);
  solver.set_phi(phi);
  solver.set_mu(1);
  solver.set_bc(bc_hodge);
  solver.set_rhs(rhs);
  solver.set_nullspace_use_fixed_point(false);

  solver.solve(hodge);

  ierr = VecDestroy(rhs); CHKERRXX(ierr);

  /* if needed, shift the hodge variable to a zero average */
  if(1 && solver.get_matrix_has_nullspace())
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


void my_p4est_navier_stokes_t::compute_velocity_at_nodes()
{
  /* interpolate vnp1 from faces to nodes */
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    PetscErrorCode ierr;
    double *v_p;
    ierr = VecGetArray(vnp1_nodes[dir], &v_p); CHKERRXX(ierr);

    for(size_t i=0; i<ngbd_n->get_layer_size(); ++i)
    {
      p4est_locidx_t n = ngbd_n->get_layer_node(i);
      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, p4est_n, nodes_n, xyz);
      v_p[n] = interpolate_f_at_node_n(p4est_n, ghost_n, nodes_n, faces_n, ngbd_c, ngbd_n, n,
                                       vnp1[dir], dir, face_is_well_defined[dir], 2, bc_v);
    }

    ierr = VecGhostUpdateBegin(vnp1_nodes[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    for(size_t i=0; i<ngbd_n->get_local_size(); ++i)
    {
      p4est_locidx_t n = ngbd_n->get_local_node(i);
      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, p4est_n, nodes_n, xyz);
      v_p[n] = interpolate_f_at_node_n(p4est_n, ghost_n, nodes_n, faces_n, ngbd_c, ngbd_n, n,
                                       vnp1[dir], dir, face_is_well_defined[dir], 2, bc_v);
    }

    ierr = VecGhostUpdateEnd(vnp1_nodes[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecRestoreArray(vnp1_nodes[dir], &v_p); CHKERRXX(ierr);

    if(bc_pressure->interfaceType()!=NOINTERFACE)
    {
      my_p4est_level_set_t lsn(ngbd_n);
      lsn.extend_Over_Interface_TVD(phi, vnp1_nodes[dir]);
    }
  }

  compute_vorticity();
  compute_max_L2_norm_u();
}


void my_p4est_navier_stokes_t::set_dt(double dt_nm1, double dt_n)
{
  this->dt_nm1 = dt_nm1;
  this->dt_n = dt_n;
}


void my_p4est_navier_stokes_t::set_dt(double dt_n)
{
  this->dt_n = dt_n;
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

void my_p4est_navier_stokes_t::advect_smoke(my_p4est_node_neighbors_t *ngbd_np1, Vec *v, Vec smoke, Vec smoke_np1)
{
  PetscErrorCode ierr;

  std::vector<double> xyz_d[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
    xyz_d[dir].resize(ngbd_np1->nodes->num_owned_indeps);
  trajectory_from_np1_to_n(ngbd_np1->p4est, ngbd_np1->nodes, ngbd_np1, dt_n, v, xyz_d);

  my_p4est_interpolation_nodes_t interp_nodes(ngbd_n);
  for(p4est_locidx_t n=0; n<ngbd_np1->nodes->num_owned_indeps; ++n)
  {
#ifdef P4_TO_P8
    double xyz_n[] = {xyz_d[0][n], xyz_d[1][n], xyz_d[2][n]};
#else
    double xyz_n[] = {xyz_d[0][n], xyz_d[1][n]};
#endif
    interp_nodes.add_point(n, xyz_n);
  }
  interp_nodes.set_input(smoke, linear);
  interp_nodes.interpolate(smoke_np1);

  /* enforce boundary condition */
  double *smoke_p;
  ierr = VecGetArray(smoke_np1, &smoke_p); CHKERRXX(ierr);
  for(size_t i=0; i<ngbd_np1->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd_np1->get_layer_node(i);
    double xyz[P4EST_DIM];
    node_xyz_fr_n(n, ngbd_np1->p4est, ngbd_np1->nodes, xyz);
#ifdef P4_TO_P8
    smoke_p[n] = MAX(smoke_p[n], (*bc_smoke)(xyz[0],xyz[1],xyz[2]));
#else
    smoke_p[n] = MAX(smoke_p[n], (*bc_smoke)(xyz[0],xyz[1]));
#endif
  }
  ierr = VecGhostUpdateBegin(smoke_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for(size_t i=0; i<ngbd_np1->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd_np1->get_local_node(i);
    double xyz[P4EST_DIM];
    node_xyz_fr_n(n, ngbd_np1->p4est, ngbd_np1->nodes, xyz);
#ifdef P4_TO_P8
    smoke_p[n] = MAX(smoke_p[n], (*bc_smoke)(xyz[0],xyz[1],xyz[2]));
#else
    smoke_p[n] = MAX(smoke_p[n], (*bc_smoke)(xyz[0],xyz[1]));
#endif
  }
  ierr = VecGhostUpdateEnd(smoke_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecRestoreArray(smoke_np1, &smoke_p); CHKERRXX(ierr);
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
void my_p4est_navier_stokes_t::update_from_tn_to_tnp1(const CF_3 *level_set, bool convergence_test, bool do_reinitialization)
#else
void my_p4est_navier_stokes_t::update_from_tn_to_tnp1(const CF_2 *level_set, bool convergence_test, bool do_reinitialization)
#endif
{
  PetscErrorCode ierr;

  ierr = PetscLogEventBegin(log_my_p4est_navier_stokes_update, 0, 0, 0, 0); CHKERRXX(ierr);

  if(!dt_updated)
    compute_dt();

  dt_updated = false;

  splitting_criteria_t *data = (splitting_criteria_t*)p4est_n->user_pointer;

  /* construct the new forest */
  splitting_criteria_vorticity_t criteria(data->min_lvl, data->max_lvl, data->lip, uniform_band, threshold_split_cell, max_L2_norm_u, smoke_thresh);
  p4est_t *p4est_np1 = p4est_copy(p4est_n, P4EST_FALSE);
  p4est_np1->connectivity = p4est_n->connectivity; // connectivity is not duplicated by p4est_copy, the pointer (i.e. the memory-address) of connectivity seems to be copied from my understanding of the source file of p4est_copy, but I feel this is a bit safer [Raphael Egan]

  p4est_ghost_t *ghost_np1 = NULL;
  p4est_nodes_t *nodes_np1 = NULL;
  my_p4est_hierarchy_t *hierarchy_np1 = NULL;
  my_p4est_node_neighbors_t *ngbd_np1 = NULL;

  Vec phi_np1 = NULL;
  Vec smoke_np1 = NULL;
  my_p4est_interpolation_nodes_t interp_nodes(ngbd_n);
  bool grid_is_changing = false;

  if(convergence_test==false)
  {
    ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
    nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);
    hierarchy_np1 = new my_p4est_hierarchy_t(p4est_np1, ghost_np1, brick);
    ngbd_np1 = new my_p4est_node_neighbors_t(hierarchy_np1, nodes_np1);

    p4est_np1->user_pointer = (void*)&criteria;
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);

    Vec vorticity_np1;
    ierr = VecDuplicate(phi_np1, &vorticity_np1); CHKERRXX(ierr);

    for(size_t n=0; n<nodes_np1->indep_nodes.elem_count; ++n)
    {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);
      interp_nodes.add_point(n, xyz);
    }
    interp_nodes.set_input(vorticity, linear);
    interp_nodes.interpolate(vorticity_np1);

    if(level_set==NULL)
    {
      interp_nodes.set_input(phi, linear);
      interp_nodes.interpolate(phi_np1);
    }
    else
    {
      sample_cf_on_nodes(p4est_np1, nodes_np1, *level_set, phi_np1);
    }

    if(smoke!=NULL && refine_with_smoke)
    {
      ierr = VecDuplicate(phi_np1, &smoke_np1); CHKERRXX(ierr);

      Vec vtmp[P4EST_DIM];
      for(int dir=0; dir<P4EST_DIM; ++dir)
      {
        ierr = VecDuplicate(phi_np1, &vtmp[dir]); CHKERRXX(ierr);
        interp_nodes.set_input(vnp1_nodes[dir], linear);
        interp_nodes.interpolate(vtmp[dir]);
      }

      advect_smoke(ngbd_np1, vtmp, smoke, smoke_np1);

      for(int dir=0; dir<P4EST_DIM; ++dir)
      {
        ierr = VecDestroy(vtmp[dir]); CHKERRXX(ierr);
      }
    }

    grid_is_changing = criteria.refine_and_coarsen(p4est_np1, ngbd_np1, phi_np1, vorticity_np1, smoke_np1);
    int iter=0;
    while(1 && grid_is_changing)
    {
      my_p4est_partition(p4est_np1, P4EST_FALSE, NULL);
      p4est_ghost_destroy(ghost_np1); ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
      p4est_nodes_destroy(nodes_np1); nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);
      delete hierarchy_np1; hierarchy_np1 = new my_p4est_hierarchy_t(p4est_np1, ghost_np1, brick);
      delete ngbd_np1; ngbd_np1 = new my_p4est_node_neighbors_t(hierarchy_np1, nodes_np1);

      ierr = VecDestroy(phi_np1); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);

      ierr = VecDestroy(vorticity_np1); CHKERRXX(ierr);
      ierr = VecDuplicate(phi_np1, &vorticity_np1); CHKERRXX(ierr);

      interp_nodes.clear();
      for(size_t n=0; n<nodes_np1->indep_nodes.elem_count; ++n)
      {
        double xyz[P4EST_DIM];
        node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);
        interp_nodes.add_point(n, xyz);
      }

      interp_nodes.set_input(vorticity, linear);
      interp_nodes.interpolate(vorticity_np1);

      if(level_set==NULL)
      {
        interp_nodes.set_input(phi, linear);
        interp_nodes.interpolate(phi_np1);
      }
      else
      {
        sample_cf_on_nodes(p4est_np1, nodes_np1, *level_set, phi_np1);
      }

      if(smoke!=NULL && refine_with_smoke)
      {
        ierr = VecDestroy(smoke_np1); CHKERRXX(ierr);
        ierr = VecDuplicate(phi_np1, &smoke_np1); CHKERRXX(ierr);

        Vec vtmp[P4EST_DIM];
        for(int dir=0; dir<P4EST_DIM; ++dir)
        {
          ierr = VecDuplicate(phi_np1, &vtmp[dir]); CHKERRXX(ierr);
          interp_nodes.set_input(vnp1_nodes[dir], linear);
          interp_nodes.interpolate(vtmp[dir]);
        }

        advect_smoke(ngbd_np1, vtmp, smoke, smoke_np1);

        for(int dir=0; dir<P4EST_DIM; ++dir)
        {
          ierr = VecDestroy(vtmp[dir]); CHKERRXX(ierr);
        }
      }

      grid_is_changing = criteria.refine_and_coarsen(p4est_np1, ngbd_np1, phi_np1, vorticity_np1, smoke_np1);

      iter++;

      if(iter>1+data->max_lvl-data->min_lvl)
      {
        ierr = PetscPrintf(p4est_n->mpicomm, "ooops ... the grid update did not converge\n"); CHKERRXX(ierr);
        break;
      }
    }
    ierr = VecDestroy(vorticity_np1);
  }

  p4est_np1->user_pointer = data;

  /* balance the forest and expand the ghost layer */
  p4est_balance(p4est_np1, P4EST_CONNECT_FULL, NULL);
  my_p4est_partition(p4est_np1, P4EST_FALSE, NULL);
  if(ghost_np1!=NULL)
    p4est_ghost_destroy(ghost_np1);
  ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est_np1, ghost_np1);
  if(nodes_np1!=NULL)
    p4est_nodes_destroy(nodes_np1);
  nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);
  if(hierarchy_np1!=NULL)
    delete hierarchy_np1;
  hierarchy_np1 = new my_p4est_hierarchy_t(p4est_np1, ghost_np1, brick);
  if(ngbd_np1!=NULL)
    delete ngbd_np1;
  ngbd_np1 = new my_p4est_node_neighbors_t(hierarchy_np1, nodes_np1);

  if(phi_np1!=NULL)
  {
    ierr = VecDestroy(phi_np1); CHKERRXX(ierr);
  }
  ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);

  interp_nodes.clear();

  for(size_t n=0; n<nodes_np1->indep_nodes.elem_count; ++n)
  {
    double xyz[P4EST_DIM];
    node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);
    interp_nodes.add_point(n, xyz);
  }

  my_p4est_cell_neighbors_t *ngbd_c_np1 = new my_p4est_cell_neighbors_t(hierarchy_np1);
  my_p4est_faces_t *faces_np1 = new my_p4est_faces_t(p4est_np1, ghost_np1, brick, ngbd_c_np1);

  ierr = VecDestroy(vorticity); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_np1, &vorticity); CHKERRXX(ierr);

  /* interpolate the quantities on the new forest at the nodes */
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecDestroy(vnm1_nodes[dir]); CHKERRXX(ierr);
    vnm1_nodes[dir] = vn_nodes[dir];

    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &vn_nodes[dir]); CHKERRXX(ierr);
    interp_nodes.set_input(vnp1_nodes[dir], quadratic);
    interp_nodes.interpolate(vn_nodes[dir]); CHKERRXX(ierr);

    ierr = VecDestroy(vnp1_nodes[dir]); CHKERRXX(ierr);
    ierr = VecDuplicate(vn_nodes[dir], &vnp1_nodes[dir]); CHKERRXX(ierr);
  }

  interp_nodes.set_input(phi, quadratic_non_oscillatory);
  interp_nodes.interpolate(phi_np1);

  interp_nodes.clear();

  /* set velocity inside solid to bc_v */
//  extrapolate_bc_v(ngbd_np1, vn_nodes, phi_np1);

  if(level_set!=NULL)
  {
    sample_cf_on_nodes(p4est_np1, nodes_np1, *level_set, phi_np1);
  }
  if(do_reinitialization)
  {
    my_p4est_level_set_t lsn(ngbd_np1);
    lsn.reinitialize_1st_order_time_2nd_order_space(phi_np1);
    lsn.perturb_level_set_function(phi_np1, EPS);
  }

  /* advect smoke */
  if(smoke!=NULL)
  {
    if(smoke_np1!=NULL)
    {
      ierr = VecDestroy(smoke_np1); CHKERRXX(ierr);
    }
    ierr = VecDuplicate(phi_np1, &smoke_np1); CHKERRXX(ierr);
    advect_smoke(ngbd_np1, vn_nodes, smoke, smoke_np1);
  }

  /* interpolate the quantities on the new forest at the cells */
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
  ierr = VecDestroy(hodge); CHKERRXX(ierr); hodge = hodge_tmp;
  ierr = VecGhostUpdateBegin(hodge, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(hodge, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecDestroy(pressure); CHKERRXX(ierr);
  ierr = VecCreateGhostCells(p4est_np1, ghost_np1, &pressure); CHKERRXX(ierr);

  /* interpolate the quantities on the new forest at the faces */
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
    ierr = VecGhostUpdateEnd  (dxyz_hodge[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecDestroy(face_is_well_defined[dir]); CHKERRXX(ierr);
		ierr = VecDuplicate(dxyz_hodge[dir], &face_is_well_defined[dir]); CHKERRXX(ierr);
    check_if_faces_are_well_defined(p4est_np1, ngbd_np1, faces_np1, dir, phi_np1, bc_v[dir].interfaceType(), face_is_well_defined[dir]);

    ierr = VecDestroy(vstar[dir]); CHKERRXX(ierr);
    ierr = VecDuplicate(dxyz_hodge[dir], &vstar[dir]); CHKERRXX(ierr);

    ierr = VecDestroy(vnp1[dir]); CHKERRXX(ierr);
    ierr = VecDuplicate(dxyz_hodge[dir], &vnp1[dir]); CHKERRXX(ierr);
  }

  /* update the variables */
  p4est_destroy(p4est_nm1); p4est_nm1 = p4est_n; p4est_n = p4est_np1;
  p4est_ghost_destroy(ghost_nm1); ghost_nm1 = ghost_n; ghost_n = ghost_np1;
  p4est_nodes_destroy(nodes_nm1); nodes_nm1 = nodes_n; nodes_n = nodes_np1;
  delete hierarchy_nm1; hierarchy_nm1 = hierarchy_n; hierarchy_n = hierarchy_np1;
  delete ngbd_nm1; ngbd_nm1 = ngbd_n; ngbd_n = ngbd_np1;
  delete ngbd_c; ngbd_c = ngbd_c_np1;
  delete faces_n; faces_n = faces_np1;

  ierr = VecDestroy(phi); CHKERRXX(ierr);
  phi = phi_np1;

  if(smoke!=NULL)
  {
    ierr = VecDestroy(smoke); CHKERRXX(ierr);
    smoke = smoke_np1;
  }

  delete interp_phi;
  interp_phi = new my_p4est_interpolation_nodes_t(ngbd_n);
  interp_phi->set_input(phi, linear);

  ierr = PetscLogEventEnd(log_my_p4est_navier_stokes_update, 0, 0, 0, 0); CHKERRXX(ierr);
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



void my_p4est_navier_stokes_t::save_vtk(const char* name)
{
  PetscErrorCode ierr;

  const double *phi_p;
  const double *vn_p[P4EST_DIM];
  const double *hodge_p;

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
    my_p4est_vtk_write_all(p4est_n, nodes_n, ghost_n,
                           P4EST_TRUE, P4EST_TRUE,
//                           P4EST_FALSE, P4EST_FALSE,
                           3+P4EST_DIM, /* number of VTK_POINT_DATA */
                           1, /* number of VTK_CELL_DATA  */
                           name,
                           VTK_POINT_DATA, "phi", phi_p,
                           VTK_POINT_DATA, "pressure", pressure_nodes_p,
                           VTK_POINT_DATA, "smoke", smoke_p,
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
    my_p4est_vtk_write_all(p4est_n, nodes_n, ghost_n,
                           P4EST_TRUE, P4EST_TRUE,
//                           P4EST_FALSE, P4EST_FALSE,
                           2+P4EST_DIM, /* number of VTK_POINT_DATA */
                           1, /* number of VTK_CELL_DATA  */
                           name,
                           VTK_POINT_DATA, "phi", phi_p,
                           VTK_POINT_DATA, "pressure", pressure_nodes_p,
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

  ierr = PetscPrintf(p4est_n->mpicomm, "Saved visual data in ... %s\n", name); CHKERRXX(ierr);
}

void my_p4est_navier_stokes_t::global_mass_flow_through_slice(const unsigned int& dir, std::vector<double>& section, std::vector<double>& mass_flows) const
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

  if(mass_flows.size() != section.size())
    mass_flows.resize(section.size(), 0.0);
  for (size_t ii = 0; ii < section.size(); ++ii) {
#ifdef CASL_THROWS
    if((section[ii] < xyz_min[dir]) || (section[ii] > xyz_max[dir]))
      throw std::invalid_argument("my_p4est_navier_stokes_t::global_mass_flow_through_slice: the slice section must be in the computational domain!");
#endif
    int tree_dim_idx = (int) floor((section[ii]-xyz_min[dir])/size_of_tree);
    double should_be_integer = (section[ii]-(xyz_min[dir] + tree_dim_idx*size_of_tree))/coarsest_cell_size;
    if(fabs(should_be_integer - ((int) should_be_integer)) > 1e-6)
    {
#ifdef CASL_THROWS
      throw std::invalid_argument("my_p4est_navier_stokes_t::global_mass_flow_through_slice: the mass flux can be evaluated only through slices in the \n computational domain that coincide with cell faces of the coarsest cells: choose a valid section!");
#else
      section[ii] = xyz_min[dir] + tree_dim_idx*size_of_tree + ((int) should_be_integer)*(section[ii]-(xyz_min[dir] + tree_dim_idx*size_of_tree))/coarsest_cell_size;
      if(p4est_n->mpirank == 0)
        std::cerr << "my_p4est_navier_stokes_t::global_mass_flow_through_slice: the section for calculating the mass flow has been relocated!" << std::endl;
#endif
    }
    mass_flows[ii] = 0.0; // initialization
  }

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
    for (size_t ii = 0; ii < section.size(); ++ii)
    {
      if(check_for_periodic_wrapping && ((fabs(section[ii] - xyz_min[dir]) < comparison_threshold) || (fabs(section[ii] - xyz_max[dir]) < comparison_threshold)))
        face_coordinate = section[ii];
      if(fabs(face_coordinate - section[ii]) < comparison_threshold)
        mass_flows[ii] += rho*vel_p[face_idx]*faces_n->face_area_in_negative_domain(face_idx, dir, phi_p, nodes_n);
    }
  }
  ierr = VecRestoreArrayRead(vnp1[dir], &vel_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &mass_flows[0], mass_flows.size(), MPI_DOUBLE, MPI_SUM, p4est_n->mpicomm); SC_CHECK_MPI(mpiret);
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
                  fraction_noslip += ((is_no_slip(xyz_stencil))?0.25:0) // always in the domain boundary by definition in this case
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
              if(is_no_slip(xyz_w)) // always in the domain boundary by defintion as wall
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
                      fraction_noslip += ((is_in_domain(xyz_stencil) && is_no_slip(xyz_stencil))?0.25:0) // could be out of the domain (consider a corner in a cavity flow, for instance)
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
          char full_path[1024];
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
        char expected_dir[1024];
        sprintf(expected_dir, "%s/backup_%d", path_to_root_directory, (int) idx);
        if(!is_folder(expected_dir))
          break; // well, it's a mess in there, but I can't really do any better...
        backup_idx++;
      }
    }
    if ((n_saved > 1) && (n_backup_subfolders == n_saved))
    {
      char full_path_zeroth_index[1024];
      sprintf(full_path_zeroth_index, "%s/backup_0", path_to_root_directory);
      // delete the 0th
      delete_directory(full_path_zeroth_index, p4est_n->mpirank, p4est_n->mpicomm, true);
      // shift the others
      for (size_t idx = 1; idx < n_saved; ++idx) {
        char old_name[1024], new_name[1024];
        sprintf(old_name, "%s/backup_%d", path_to_root_directory, (int) idx);
        sprintf(new_name, "%s/backup_%d", path_to_root_directory, (int) (idx-1));
        rename(old_name, new_name);
      }
      backup_idx = n_saved-1;
    }
  }
  int mpiret = MPI_Bcast(&backup_idx, 1, MPI_INT, 0, p4est_n->mpicomm); SC_CHECK_MPI(mpiret);// acts as a MPI_Barrier, too

  char path_to_folder[1024];
  sprintf(path_to_folder, "%s/backup_%d", path_to_root_directory, (int) backup_idx);
  create_directory(path_to_folder, p4est_n->mpirank, p4est_n->mpicomm);

  PetscErrorCode ierr;
  char filename[1024];

  // save general solver parameters first
  sprintf(filename, "%s/solver_parameters.petsc", path_to_folder);
  save_or_load_parameters(filename, (splitting_criteria_t*) p4est_n->user_pointer, SAVE, tn);
  sprintf(filename, "%s/solver_parameters.ascii", path_to_folder);
  save_or_load_parameters(filename, (splitting_criteria_t*) p4est_n->user_pointer, SAVE_ASCII, tn);
  // save connectivity
  sprintf(filename, "%s/connectivity", path_to_folder);
  p4est_connectivity_save(filename, p4est_n->connectivity);
  // save p4est_n
  sprintf(filename, "%s/p4est_n", path_to_folder);
  p4est_save_ext(filename, p4est_n, P4EST_FALSE, P4EST_TRUE); // no cell-data saved
  // save p4est_nm1
  sprintf(filename, "%s/p4est_nm1", path_to_folder);
  p4est_save_ext(filename, p4est_nm1, P4EST_FALSE, P4EST_TRUE); // no cell-data saved
  // save phi
  sprintf(filename, "%s/phi.petsc", path_to_folder);
  ierr = VecDump(filename, phi); CHKERRXX(ierr);
  // save hodge
  sprintf(filename, "%s/hodge.petsc", path_to_folder);
  ierr = VecDump(filename, hodge); CHKERRXX(ierr);
  // save vnm1_nodes
  sprintf(filename, "%s/vnm1_nodes.petsc", path_to_folder);
  ierr = VecDump(filename, P4EST_DIM, vnm1_nodes); CHKERRXX(ierr);
  // save vn_nodes
  sprintf(filename, "%s/vn_nodes.petsc", path_to_folder);
  ierr = VecDump(filename, P4EST_DIM, vn_nodes); CHKERRXX(ierr);

  // save smoke if used
  if (smoke!=NULL)
  {
    sprintf(filename, "%s/smoke.petsc", path_to_folder);
    ierr = VecDump(filename, smoke); CHKERRXX(ierr);
  }

  ierr = PetscPrintf(p4est_n->mpicomm, "Saved solver state in ... %s\n", path_to_folder); CHKERRXX(ierr);
}

void my_p4est_navier_stokes_t::save_or_load_parameters(const char* filename, splitting_criteria_t* data, save_or_load flag, double &tn)
{
  PetscViewer viewer;
  PetscErrorCode ierr;
  struct
  {
    PetscErrorCode operator()(save_or_load flag_, PetscViewer& viewer_, void* data_, PetscInt n_data, PetscDataType dtype, const char *var_name = NULL)
    {
      switch (flag_) {
      case SAVE:
      {
        return PetscViewerBinaryWrite(viewer_, data_, n_data, dtype, PETSC_FALSE);
        break;
      }
      case SAVE_ASCII:
      {
        if(var_name == NULL)
          throw std::runtime_error("my_p4est_navier_stokes_t::save_or_load_parameters: the variable name MUST be given when exporting ASCII data");
        PetscErrorCode iierr;
        iierr = PetscViewerASCIIPrintf(viewer_, "%s:", var_name);
        switch (dtype) {
        case PETSC_INT:
        {
          for (PetscInt var_idx = 0; var_idx < n_data; ++var_idx)
            iierr = iierr || PetscViewerASCIIPrintf(viewer_, " %d", ((PetscInt*)data_)[var_idx]);
          iierr = PetscViewerASCIIPrintf(viewer_, "\n", var_name);
          return iierr;
          break;
        }
        case PETSC_BOOL:
        {
          for (PetscInt var_idx = 0; var_idx < n_data; ++var_idx)
            iierr = iierr || PetscViewerASCIIPrintf(viewer_, " %s", ((((PetscBool*) data_)[var_idx])? "true" : "false"));
          iierr = PetscViewerASCIIPrintf(viewer_, "\n", var_name);
          return iierr;
          break;
        }
        case PETSC_DOUBLE:
        {
          for (PetscInt var_idx = 0; var_idx < n_data; ++var_idx)
            iierr = iierr || PetscViewerASCIIPrintf(viewer_, " %g", ((PetscReal*) data_)[var_idx]);
          iierr = PetscViewerASCIIPrintf(viewer_, "\n", var_name);
          return iierr;
          break;
        }
        default:
          throw std::runtime_error("my_p4est_navier_stokes_t::save_or_load_parameters: unknonw dataype when exporting parameters in ASCII format.");
          break;
        }
        break;
      }
      case LOAD:
      {
        PetscInt num_read;
        PetscErrorCode iierr = PetscViewerBinaryRead(viewer_, data_, n_data, &num_read, dtype);
        P4EST_ASSERT(num_read == n_data);
        return iierr;
        break;
      }
      default:
        throw std::runtime_error("my_p4est_navier_stokes_t::save_or_load_parameters: unknown flag value");
        break;
      }

    }
  } elementary_operation;
  int P4EST_DIM_COPY;
  switch (flag) {
  case SAVE:
  {
    ierr = PetscViewerBinaryOpen(p4est_n->mpicomm, filename, FILE_MODE_WRITE, &viewer); CHKERRXX(ierr);
    P4EST_DIM_COPY = P4EST_DIM;
    elementary_operation(flag, viewer, &P4EST_DIM_COPY, 1, PETSC_INT);
    break;
  }
  case SAVE_ASCII:
  {
    ierr = PetscViewerASCIIOpen(p4est_n->mpicomm, filename, &viewer); CHKERRXX(ierr);
    P4EST_DIM_COPY = P4EST_DIM;
    elementary_operation(flag, viewer, &P4EST_DIM_COPY, 1, PETSC_INT, "P4EST_DIM");
    break;
  }
  case LOAD:
  {
    ierr = PetscViewerBinaryOpen(p4est_n->mpicomm, filename, FILE_MODE_READ, &viewer); CHKERRXX(ierr);
    elementary_operation(flag, viewer, &P4EST_DIM_COPY, 1, PETSC_INT);
    if(P4EST_DIM_COPY != P4EST_DIM)
      throw std::runtime_error("my_p4est_navier_stokes_t::save_or_load_parameters: you are trying to read 2D (resp. 3D) data with a 3D (resp. 2D) program...");
    break;
  }
  default:
    throw std::runtime_error("my_p4est_navier_stokes_t::save_or_load_parameters: unknown flag value");
    break;
  }
  ierr = elementary_operation(flag, viewer, dxyz_min,               P4EST_DIM,  PETSC_DOUBLE, "dxyz_min");              CHKERRXX(ierr);
  ierr = elementary_operation(flag, viewer, xyz_min,                P4EST_DIM,  PETSC_DOUBLE, "xyz_min");               CHKERRXX(ierr);
  ierr = elementary_operation(flag, viewer, xyz_max,                P4EST_DIM,  PETSC_DOUBLE, "xyz_max");               CHKERRXX(ierr);
  ierr = elementary_operation(flag, viewer, convert_to_xyz,         P4EST_DIM,  PETSC_DOUBLE, "convert_to_xyz");        CHKERRXX(ierr);
  ierr = elementary_operation(flag, viewer, &mu,                    1,          PETSC_DOUBLE, "mu");                    CHKERRXX(ierr);
  ierr = elementary_operation(flag, viewer, &rho,                   1,          PETSC_DOUBLE, "rho");                   CHKERRXX(ierr);
  ierr = elementary_operation(flag, viewer, &tn,                    1,          PETSC_DOUBLE, "tn");                    CHKERRXX(ierr);
  ierr = elementary_operation(flag, viewer, &dt_n,                  1,          PETSC_DOUBLE, "dt_n");                  CHKERRXX(ierr);
  ierr = elementary_operation(flag, viewer, &dt_nm1,                1,          PETSC_DOUBLE, "dt_nm1");                CHKERRXX(ierr);
  ierr = elementary_operation(flag, viewer, &max_L2_norm_u,         1,          PETSC_DOUBLE, "max_L2_norm_u");         CHKERRXX(ierr);
  ierr = elementary_operation(flag, viewer, &uniform_band,          1,          PETSC_DOUBLE, "uniform_band");          CHKERRXX(ierr);
  ierr = elementary_operation(flag, viewer, &threshold_split_cell,  1,          PETSC_DOUBLE, "threshold_split_cell");  CHKERRXX(ierr);
  ierr = elementary_operation(flag, viewer, &n_times_dt,            1,          PETSC_DOUBLE, "n_times_dt");            CHKERRXX(ierr);
  ierr = elementary_operation(flag, viewer, &refine_with_smoke,     1,          PETSC_BOOL,   "refine_with_smoke");     CHKERRXX(ierr);
  ierr = elementary_operation(flag, viewer, &smoke_thresh,          1,          PETSC_DOUBLE, "smoke_thresh");          CHKERRXX(ierr);
  ierr = elementary_operation(flag, viewer, &data->min_lvl,         1,          PETSC_INT,    "data->min_lvl");         CHKERRXX(ierr);
  ierr = elementary_operation(flag, viewer, &data->max_lvl,         1,          PETSC_INT,    "data->max_lvl");         CHKERRXX(ierr);
  ierr = elementary_operation(flag, viewer, &data->lip,             1,          PETSC_DOUBLE, "data->lip");             CHKERRXX(ierr);
  ierr = elementary_operation(flag, viewer, &sl_order,              1,          PETSC_INT,    "sl_order");              CHKERRXX(ierr);
  ierr = PetscViewerDestroy(viewer); CHKERRXX(ierr);
}

void my_p4est_navier_stokes_t::load_state(const mpi_environment_t& mpi, const char* path_to_folder, splitting_criteria_t* data, p4est_connectivity_t* conn, double& tn)
{

  PetscErrorCode ierr;
  char filename[1024];

  // load general solver parameters first
  if(data != NULL)
    delete data;
  sprintf(filename, "%s/solver_parameters.petsc", path_to_folder);
  save_or_load_parameters(filename, data, LOAD, tn);
  // load connectivity
  if(conn != NULL)
    p4est_connectivity_destroy(conn);
  sprintf(filename, "%s/connectivity", path_to_folder);
  conn = p4est_connectivity_load(filename, NULL);
  // load p4est_n
  if(p4est_n != NULL)
    p4est_destroy(p4est_n);
  sprintf(filename, "%s/p4est_n", path_to_folder);
  p4est_n = p4est_load_ext(filename, mpi.comm(), 0, P4EST_FALSE, P4EST_FALSE, P4EST_FALSE, (void*) &data, &conn); // no data load, because we assume no data saved, we don't ignore the saved partition, in case we load it with the same number of processes
  // load p4est_nm1
  if(p4est_nm1 != NULL)
    p4est_destroy(p4est_nm1);
  sprintf(filename, "%s/p4est_nm1", path_to_folder);
  p4est_nm1 = p4est_load_ext(filename, mpi.comm(), 0, P4EST_FALSE, P4EST_FALSE, P4EST_FALSE, (void*) &data, &conn);
  // load phi
  if(phi != NULL)
  {
    ierr = VecDestroy(phi); CHKERRXX(ierr);
  }
  sprintf(filename, "%s/phi.petsc", path_to_folder);
  ierr = VecLoad(mpi.comm(), filename, &phi); CHKERRXX(ierr);
  // check if phi is consistent with p4est_n, ghost it and rebuilt it with updated ghost values
  // load hodge
  sprintf(filename, "%s/hodge.petsc", path_to_folder);
  ierr = VecLoad(mpi.comm(), filename, &hodge); CHKERRXX(ierr);
  // check if phi is consistent with p4est_n, ghost it and rebuilt it with updated ghost values
  // load vnm1_nodes
  for (short kk = 0; kk < P4EST_DIM; ++kk) {
    if(vnm1_nodes[kk] != NULL)
    {
      ierr = VecDestroy(vnm1_nodes[kk]); CHKERRXX(ierr);
    }
  }
  sprintf(filename, "%s/vnm1_nodes.petsc", path_to_folder);
  ierr = VecLoad(mpi.comm(), filename, P4EST_DIM, vnm1_nodes); CHKERRXX(ierr);
  // check if every component of vnm1_nodes is consistent with p4est_n, ghost it and rebuilt it with updated ghost values
  // load vn_nodes
  for (short kk = 0; kk < P4EST_DIM; ++kk) {
    if (vn_nodes[kk] != NULL) {
      ierr = VecDestroy(vnm1_nodes[kk]); CHKERRXX(ierr);
    }
  }
  sprintf(filename, "%s/vn_nodes.petsc", path_to_folder);
  ierr = VecLoad(mpi.comm(), filename, P4EST_DIM, vn_nodes); CHKERRXX(ierr);
  // check if every component of vn_nodes is consistent with p4est_n, ghost it and rebuilt it with updated ghost values
  // load smoke was used
  sprintf(filename, "%s/smoke.petsc", path_to_folder);
  if (file_exists(filename))
  {
    ierr = VecLoad(mpi.comm(), filename, &smoke); CHKERRXX(ierr);
    // check if smoke is consistent with p4est_n, ghost it and rebuild it
  }

  ierr = PetscPrintf(mpi.comm(), "Loaded solver state from ... %s\n", path_to_folder); CHKERRXX(ierr);
}

