#include "my_p4est_navier_stokes.h"

#ifdef P4_TO_P8
#include <src/my_p8est_poisson_cells.h>
#include <src/my_p8est_poisson_faces.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_level_set_cells.h>
#include <src/my_p8est_level_set_faces.h>
#include <src/my_p8est_trajectory_of_point.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_vtk.h>
#else
#include <src/my_p4est_poisson_cells.h>
#include <src/my_p4est_poisson_faces.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_level_set_cells.h>
#include <src/my_p4est_level_set_faces.h>
#include <src/my_p4est_trajectory_of_point.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_vtk.h>
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
        all_pos = all_pos && phi(x+i*dx, y+j*dy, z+k*dz)>0;
#else
      {
        cor_vort = cor_vort && fabs(vor(x+i*dx, y+j*dy))*2*MAX(dx,dy)/max_L2_norm_u<threshold;
        cor_band = cor_band && fabs(phi(x+i*dx, y+j*dy))>uniform_band*dxyz_min;
        cor_intf = cor_intf && fabs(phi(x+i*dx, y+j*dy))>=lip*2*d;
        if(smo!=NULL)
          cor_smok = cor_smok && (*smo)(x+i*dx, y+j*dy)<smoke_thresh;
        all_pos = all_pos && phi(x+i*dx, y+j*dy)>0;
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
          is_neg = is_neg || phi(x+i*dx/2, y+j*dy/2, z+k*dz/2)<0;
#else
      {
        ref_vort = ref_vort || fabs(vor(x+i*dx/2, y+j*dy/2))*MAX(dx,dy)/max_L2_norm_u>threshold;
        ref_band = ref_band || fabs(phi(x+i*dx/2, y+j*dy/2))<uniform_band*dxyz_min;
        ref_intf = ref_intf || fabs(phi(x+i*dx/2, y+j*dy/2))<=lip*d;
        if(smo!=NULL)
          ref_smok = ref_smok || (*smo)(x+i*dx/2, y+j*dy/2)>=smoke_thresh;
        is_neg = is_neg || phi(x+i*dx/2, y+j*dy/2)<0;
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
      max_L2_norm_u = MAX(max_L2_norm_u, sqrt(SQR(v_p[0][n]) + SQR(v_p[1][n])) );
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
        if(theta<EPS) theta = EPS; if(theta>1) theta = 1;
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
    if(sl_order==2) trajectory_from_np1_to_nm1(p4est_n, faces_n, ngbd_nm1, ngbd_n, vnm1_nodes, vn_nodes, dt_nm1, dt_n, xyz_nm1, xyz_n, dir);
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
#if defined(COMET) || defined(STAMPEDE)
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


void my_p4est_navier_stokes_t::compute_dt()
{
  dt_nm1 = dt_n;
#ifdef P4_TO_P8
  dt_n = MIN(1., 1/max_L2_norm_u) * n_times_dt * MIN(dxyz_min[0], dxyz_min[1], dxyz_min[2]);
#else
  dt_n = MIN(1., 1/max_L2_norm_u) * n_times_dt * MIN(dxyz_min[0], dxyz_min[1]);
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
void my_p4est_navier_stokes_t::update_from_tn_to_tnp1(const CF_3 *level_set, bool convergence_test)
#else
void my_p4est_navier_stokes_t::update_from_tn_to_tnp1(const CF_2 *level_set, bool convergence_test)
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
  if(ghost_np1!=NULL) p4est_ghost_destroy(ghost_np1); ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est_np1, ghost_np1);
  if(nodes_np1!=NULL) p4est_nodes_destroy(nodes_np1); nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);
  if(hierarchy_np1!=NULL) delete hierarchy_np1; hierarchy_np1 = new my_p4est_hierarchy_t(p4est_np1, ghost_np1, brick);
  if(ngbd_np1!=NULL) delete ngbd_np1; ngbd_np1 = new my_p4est_node_neighbors_t(hierarchy_np1, nodes_np1);

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
  my_p4est_level_set_t lsn(ngbd_np1);
  lsn.reinitialize_1st_order_time_2nd_order_space(phi_np1);
  lsn.perturb_level_set_function(phi_np1, EPS);

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
  ierr = VecDestroy(hodge); CHKERRXX(ierr);
  ierr = VecCreateGhostCells(p4est_np1, ghost_np1, &hodge); CHKERRXX(ierr);

  ierr = VecDestroy(pressure); CHKERRXX(ierr);
  ierr = VecDuplicate(hodge, &pressure); CHKERRXX(ierr);

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
