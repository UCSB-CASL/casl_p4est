#include "my_p4est_navier_stokes.h"

#ifdef P4_TO_P8
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_poisson_cells.h>
#include <src/my_p8est_poisson_faces.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_level_set_cells.h>
#include <src/my_p8est_level_set_faces.h>
#include <src/my_p8est_trajectory_of_point.h>
#include <src/my_p8est_vtk.h>
#else
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_poisson_cells.h>
#include <src/my_p4est_poisson_faces.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_level_set_cells.h>
#include <src/my_p4est_level_set_faces.h>
#include <src/my_p4est_trajectory_of_point.h>
#include <src/my_p4est_vtk.h>
#endif




#ifdef P4_TO_P8
double my_p4est_navier_stokes_t::wall_bc_value_hodge_t::operator ()(double x, double y, double z) const
#else
double my_p4est_navier_stokes_t::wall_bc_value_hodge_t::operator ()(double x, double y) const
#endif
{
  double alpha = (2*_prnt->dt_n + _prnt->dt_nm1)/(_prnt->dt_n + _prnt->dt_nm1);
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
  double alpha = (2*_prnt->dt_n + _prnt->dt_nm1)/(_prnt->dt_n + _prnt->dt_nm1);
#ifdef P4_TO_P8
  return _prnt->bc_pressure->interfaceValue(x,y,z) * _prnt->dt_n / (alpha*_prnt->rho);
#else
  return _prnt->bc_pressure->interfaceValue(x,y)   * _prnt->dt_n / (alpha*_prnt->rho);
#endif
}



#ifdef P4_TO_P8
double my_p4est_navier_stokes_t::wall_bc_value_vstar_t::operator ()(double x, double y, double z) const
#else
double my_p4est_navier_stokes_t::wall_bc_value_vstar_t::operator ()(double x, double y) const
#endif
{
#ifdef P4_TO_P8
  if(_prnt->bc_v[dir].wallType(x,y,z)==DIRICHLET)
    return _prnt->bc_v[dir].wallValue(x,y,z) + (*_prnt->interp_dxyz_hodge[dir])(x,y,z);
  else
    return _prnt->bc_v[dir].wallValue(x,y,z);
#else
  if(_prnt->bc_v[dir].wallType(x,y)==DIRICHLET)
    return _prnt->bc_v[dir].wallValue(x,y) + (*_prnt->interp_dxyz_hodge[dir])(x,y);
  else
    return _prnt->bc_v[dir].wallValue(x,y);
#endif
}



#ifdef P4_TO_P8
double my_p4est_navier_stokes_t::interface_bc_value_vstar_t::operator ()(double x, double y, double z) const
#else
double my_p4est_navier_stokes_t::interface_bc_value_vstar_t::operator ()(double x, double y) const
#endif
{
#ifdef P4_TO_P8
  if(_prnt->bc_v[dir].interfaceType()==DIRICHLET)
    return _prnt->bc_v[dir].interfaceValue(x,y,z) + (*_prnt->interp_dxyz_hodge[dir])(x,y,z);
  else
    return _prnt->bc_v[dir].interfaceValue(x,y,z);
#else
  if(_prnt->bc_v[dir].interfaceType()==DIRICHLET)
    return _prnt->bc_v[dir].interfaceValue(x,y) + (*_prnt->interp_dxyz_hodge[dir])(x,y);
  else
    return _prnt->bc_v[dir].interfaceValue(x,y);
#endif
}



my_p4est_navier_stokes_t::my_p4est_navier_stokes_t(my_p4est_node_neighbors_t *ngbd_nm1, my_p4est_node_neighbors_t *ngbd_n, my_p4est_faces_t *faces)
  : brick(ngbd_n->myb),
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
  n_times_dt = 1;
  dt_updated = false;
  max_L2_norm_u = 0;

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
    dxyz_min[dir] = (xyz_tmp-xyz_min[dir]) / pow(2., (double)data->max_lvl);
  }

#ifdef P4_TO_P8
  dt_nm1 = dt_n = MIN(1., 1/max_L2_norm_u) * n_times_dt * MIN(dxyz_min[0], dxyz_min[1], dxyz_min[2]);
#else
  dt_nm1 = dt_n = MIN(1., 1/max_L2_norm_u) * n_times_dt * MIN(dxyz_min[0], dxyz_min[1]);
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

  bc_v = NULL;
  bc_pressure = NULL;
  external_forces = NULL;

#ifndef P4_TO_P8
  vorticity = NULL;
#endif

  norm_grad_v = NULL;

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    vstar[dir] = NULL;
    vnp1[dir] = NULL;
#ifdef P4_TO_P8
    vorticity[dir] = NULL;
#endif

    vnm1_nodes[dir] = NULL;
    vn_nodes  [dir] = NULL;
    vnp1_nodes[dir] = NULL;

    wall_bc_value_vstar[dir] = new wall_bc_value_vstar_t(this, dir);
    interface_bc_value_vstar[dir] = new interface_bc_value_vstar_t(this, dir);

    ierr = VecCreateGhostFaces(p4est_n, faces, &dxyz_hodge[dir], dir); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(dxyz_hodge[dir], &vec_loc); CHKERRXX(ierr);
    ierr = VecSet(vec_loc, 0); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(dxyz_hodge[dir], &vec_loc); CHKERRXX(ierr);

    ierr = VecDuplicate(dxyz_hodge[dir], &face_is_well_defined[dir]); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(face_is_well_defined[dir], &vec_loc); CHKERRXX(ierr);
    ierr = VecSet(vec_loc, 1); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(face_is_well_defined[dir], &vec_loc); CHKERRXX(ierr);

    /* NOTE: bousouf in the original CASL code, dx_hodge is interpolated using extrapolated values */
    interp_dxyz_hodge[dir] = new my_p4est_interpolation_faces_t(ngbd_n, faces);
    interp_dxyz_hodge[dir]->set_input(dxyz_hodge[dir], face_is_well_defined[dir], dir);
  }

  interp_phi = new my_p4est_interpolation_nodes_t(ngbd_n);
  interp_phi->set_input(phi, linear);
}


my_p4est_navier_stokes_t::~my_p4est_navier_stokes_t()
{
  PetscErrorCode ierr;
  if(phi!=NULL) { ierr = VecDestroy(phi); CHKERRXX(ierr); }
  if(hodge!=NULL) { ierr = VecDestroy(hodge); CHKERRXX(ierr); }
#ifndef P4_TO_P8
  if(vorticity!=NULL) { ierr = VecDestroy(vorticity); CHKERRXX(ierr); }
#endif

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    if(dxyz_hodge[dir]!=NULL) { ierr = VecDestroy(dxyz_hodge[dir]); CHKERRXX(ierr); }
    if(vstar[dir]!=NULL) { ierr = VecDestroy(vstar[dir]); CHKERRXX(ierr); }
    if(vnp1[dir]!=NULL) { ierr = VecDestroy(vnp1[dir]); CHKERRXX(ierr); }
    if(vnm1_nodes[dir]!=NULL) { ierr = VecDestroy(vnm1_nodes[dir]); CHKERRXX(ierr); }
    if(vn_nodes[dir]!=NULL) { ierr = VecDestroy(vn_nodes[dir]); CHKERRXX(ierr); }
    if(vnp1_nodes[dir]!=NULL) { ierr = VecDestroy(vnp1_nodes[dir]); CHKERRXX(ierr); }
#ifdef P4_TO_P8
    if(vorticity[dir]!=NULL) { ierr = VecDestroy(vorticity[dir]); CHKERRXX(ierr); }
#endif
    if(face_is_well_defined[dir]!=NULL) { ierr = VecDestroy(face_is_well_defined[dir]); CHKERRXX(ierr); }

    if(interp_dxyz_hodge[dir]!=NULL) delete interp_dxyz_hodge[dir];
    if(wall_bc_value_vstar[dir]!=NULL) delete wall_bc_value_vstar[dir];
    if(interface_bc_value_vstar[dir]!=NULL) delete interface_bc_value_vstar[dir];
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
}


void my_p4est_navier_stokes_t::set_parameters(double mu, double rho, double uniform_band, double threshold_split_cell, double n_times_dt)
{
  this->mu = mu;
  this->rho = rho;
  this->uniform_band = uniform_band;
  this->threshold_split_cell = threshold_split_cell;
  this->n_times_dt = n_times_dt;
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
void my_p4est_navier_stokes_t::set_external_forces(CF_3 *external_forces)
#else
void my_p4est_navier_stokes_t::set_external_forces(CF_2 *external_forces)
#endif
{
  this->external_forces = external_forces;
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

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    bc_vstar[dir].setWallTypes(bc_v[dir].getWallType());
    bc_vstar[dir].setWallValues(*wall_bc_value_vstar[dir]);
    bc_vstar[dir].setInterfaceType(bc_v[dir].interfaceType());
    bc_vstar[dir].setInterfaceValue(*interface_bc_value_vstar[dir]);
  }

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

    ierr = VecDuplicate(dxyz_hodge[dir], &vstar[dir]); CHKERRXX(ierr);
    ierr = VecDuplicate(dxyz_hodge[dir], &vnp1[dir]); CHKERRXX(ierr);

#ifdef P4_TO_P8
    ierr = VecDuplicate(phi, &vorticity[dir]); CHKERRXX(ierr);
#endif
  }

#ifndef P4_TO_P8
  ierr = VecDuplicate(phi, &vorticity); CHKERRXX(ierr);
#endif

  ierr = VecDuplicate(phi, &norm_grad_v); CHKERRXX(ierr);
}


#ifdef P4_TO_P8
void my_p4est_navier_stokes_t::set_velocities(CF_3 *vnm1, CF_3 *vn)
#else
void my_p4est_navier_stokes_t::set_velocities(CF_2 *vnm1, CF_2 *vn)
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
      v_p[n] = vn[dir](xyz[0], xyz[1], xyz[2]);
#else
      v_p[n] = vn[dir](xyz[0], xyz[1]);
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
      v_p[n] = vnm1[dir](xyz[0], xyz[1], xyz[2]);
#else
      v_p[n] = vnm1[dir](xyz[0], xyz[1]);
#endif
    }
    ierr = VecRestoreArray(vn_nodes[dir], &v_p); CHKERRXX(ierr);


    ierr = VecDuplicate(vn_nodes[dir], &vnp1_nodes [dir]); CHKERRXX(ierr);
    ierr = VecDuplicate(dxyz_hodge[dir], &vstar[dir]); CHKERRXX(ierr);
    ierr = VecDuplicate(dxyz_hodge[dir], &vnp1[dir]); CHKERRXX(ierr);

#ifdef P4_TO_P8
    ierr = VecDuplicate(phi, &vorticity[dir]); CHKERRXX(ierr);
#endif
  }

#ifndef P4_TO_P8
  ierr = VecDuplicate(phi, &vorticity); CHKERRXX(ierr);
#endif

  ierr = VecDuplicate(phi, &norm_grad_v); CHKERRXX(ierr);
}



void my_p4est_navier_stokes_t::compute_max_L2_norm_u()
{
  PetscErrorCode ierr;
  max_L2_norm_u = 0;

  const double *phi_p;
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);

#ifdef P4_TO_P8
  double dxmax = MAX(dxyz_min[0], dxyz_min[1]);
#else
  double dxmax = MAX(dxyz_min[0], dxyz_min[1], dxyz_min[2]);
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

  ierr = PetscPrintf(p4est_n->mpicomm, "max norm velocity : %g\n", max_L2_norm_u); CHKERRXX(ierr);
}


void my_p4est_navier_stokes_t::compute_vorticity()
{
  PetscErrorCode ierr;

  quad_neighbor_nodes_of_node_t qnnn;

  const double *vnp1_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; dir++) { ierr = VecGetArrayRead(vnp1_nodes[dir], &vnp1_p[dir]); CHKERRXX(ierr); }

#ifdef P4_TO_P8
  double *vorticity_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; dir++) { ierr = VecGetArray(vorticity[dir], &vorticity_p[dir]); CHKERRXX(ierr); }

  for(size_t i=0; i<ngbd_n->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n->get_layer_node(i);
    ngbd_n->get_neighbors(n, qnnn);
    vorticity_p[0] = qnnn.dy_central(vnp1_p[2]) - qnnn.dz_central(vnp1_p[1]);
    vorticity_p[1] = qnnn.dz_central(vnp1_p[0]) - qnnn.dx_central(vnp1_p[2]);
    vorticity_p[2] = qnnn.dx_central(vnp1_p[1]) - qnnn.dy_central(vnp1_p[0]);
  }

  for(int dir=0; dir<P4EST_DIM; ++dir) { ierr = VecGhostUpdateBegin(vorticity[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr); }

  for(size_t i=0; i<ngbd_n->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n->get_local_node(i);
    ngbd_n->get_neighbors(n, qnnn);
    vorticity_p[0] = qnnn.dy_central(vnp1_p[2]) - qnnn.dz_central(vnp1_p[1]);
    vorticity_p[1] = qnnn.dz_central(vnp1_p[0]) - qnnn.dx_central(vnp1_p[2]);
    vorticity_p[2] = qnnn.dx_central(vnp1_p[1]) - qnnn.dy_central(vnp1_p[0]);
  }

  for(int dir=0; dir<P4EST_DIM; ++dir) { ierr = VecGhostUpdateEnd(vorticity[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr); }
#else
  double *vorticity_p;
  ierr = VecGetArray(vorticity, &vorticity_p); CHKERRXX(ierr);


  for(size_t i=0; i<ngbd_n->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n->get_layer_node(i);
    ngbd_n->get_neighbors(n, qnnn);
    vorticity_p[n] = qnnn.dx_central(vnp1_p[1]) - qnnn.dy_central(vnp1_p[0]);
  }

  ierr = VecGhostUpdateBegin(vorticity, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  for(size_t i=0; i<ngbd_n->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n->get_local_node(i);
    ngbd_n->get_neighbors(n, qnnn);
    vorticity_p[n] = qnnn.dx_central(vnp1_p[1]) - qnnn.dy_central(vnp1_p[0]);
  }

  ierr = VecGhostUpdateEnd(vorticity, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#endif

  for(int dir=0; dir<P4EST_DIM; dir++) { ierr = VecRestoreArrayRead(vnp1_nodes[dir], &vnp1_p[dir]); CHKERRXX(ierr); }
}




void my_p4est_navier_stokes_t::compute_norm_grad_v()
{
  PetscErrorCode ierr;

  double *norm_grad_v_p;
  ierr = VecGetArray(norm_grad_v, &norm_grad_v_p); CHKERRXX(ierr);

  const double *v_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir) { ierr = VecGetArrayRead(vnp1_nodes[dir], &v_p[dir]); CHKERRXX(ierr); }

  quad_neighbor_nodes_of_node_t qnnn;

  for(size_t i=0; i<ngbd_n->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n->get_layer_node(i);
    ngbd_n->get_neighbors(n, qnnn);
    norm_grad_v_p[n] = 0;
    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
#ifdef P4_TO_P8
      norm_grad_v_p[n] += ABS(qnnn.dx_central(v_p[dir])) + ABS(qnnn.dy_central(v_p[dir])) + ABS(qnnn.dz_central(v_p[dir]));
#else
      norm_grad_v_p[n] += ABS(qnnn.dx_central(v_p[dir])) + ABS(qnnn.dy_central(v_p[dir]));
#endif
    }
  }

  ierr = VecGhostUpdateBegin(norm_grad_v, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  for(size_t i=0; i<ngbd_n->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n->get_local_node(i);
    ngbd_n->get_neighbors(n, qnnn);
    norm_grad_v_p[n] = 0;
    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
#ifdef P4_TO_P8
      norm_grad_v_p[n] += ABS(qnnn.dx_central(v_p[dir])) + ABS(qnnn.dy_central(v_p[dir])) + ABS(qnnn.dz_central(v_p[dir]));
#else
      norm_grad_v_p[n] += ABS(qnnn.dx_central(v_p[dir])) + ABS(qnnn.dy_central(v_p[dir]));
#endif
    }
  }

  ierr = VecGhostUpdateEnd(norm_grad_v, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  for(int dir=0; dir<P4EST_DIM; ++dir) { ierr = VecRestoreArrayRead(vnp1_nodes[dir], &v_p[dir]); CHKERRXX(ierr); }

  ierr = VecRestoreArray(norm_grad_v, &norm_grad_v_p); CHKERRXX(ierr);
}




double my_p4est_navier_stokes_t::compute_dxyz_hodge(p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, char dir)
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
    double z = quad_z_fr_q(quad_idx, tree_idx, p4est, ghost);
#endif

    double dx = (double) P4EST_QUADRANT_LEN(quad->level)/(double) P4EST_ROOT_LEN;
    double dy = dx;
#ifdef P4_TO_P8
      double dz = dx;
#endif

    double hodge_q = hodge_p[quad_idx];
    ierr = VecRestoreArrayRead(hodge, &hodge_p); CHKERRXX(ierr);

    switch(dir)
    {
#ifdef P4_TO_P8
    case dir::f_m00:
      if(bc_hodge.wallType(x-dx,y,z)==NEUMANN) return -bc_hodge.wallValue(x-dx,y,z);
      else                                     return (hodge_q - bc_hodge.wallValue(x-dx,y,z)) * 2 / dx;
    case dir::f_p00:
      if(bc_hodge.wallType(x+dx,y,z)==NEUMANN) return  bc_hodge.wallValue(x+dx,y,z);
      else                                     return (bc_hodge.wallValue(x+dx,y,z) - hodge_q) * 2 / dx;
    case dir::f_0m0:
      if(bc_hodge.wallType(x,y-dy,z)==NEUMANN) return -bc_hodge.wallValue(x,y-dy,z);
      else                                     return (hodge_q - bc_hodge.wallValue(x,y-dy,z)) * 2 / dy;
    case dir::f_0p0:
      if(bc_hodge.wallType(x,y+dy,z)==NEUMANN) return  bc_hodge.wallValue(x,y+dy,z);
      else                                     return (bc_hodge.wallValue(x,y+dy,z) - hodge_q) * 2 / dy;
    case dir::f_00m:
      if(bc_hodge.wallType(x,y,z-dz)==NEUMANN) return -bc_hodge.wallValue(x,y,z-dz);
      else                                     return (hodge_q - bc_hodge.wallValue(x,y,z-dz)) * 2 / dz;
    case dir::f_00p:
      if(bc_hodge.wallType(x,y,z+dz)==NEUMANN) return  bc_hodge.wallValue(x,y,z+dz);
      else                                     return (bc_hodge.wallValue(x,y,z+dz) - hodge_q) * 2 / dz;
#else
    case dir::f_m00:
      if(bc_hodge.wallType(x-dx,y)==NEUMANN) return -bc_hodge.wallValue(x-dx,y);
      else                                   return (hodge_q - bc_hodge.wallValue(x-dx,y)) * 2 / dx;
    case dir::f_p00:
      if(bc_hodge.wallType(x+dx,y)==NEUMANN) return  bc_hodge.wallValue(x+dx,y);
      else                                   return (bc_hodge.wallValue(x+dx,y) - hodge_q) * 2 / dx;
    case dir::f_0m0:
      if(bc_hodge.wallType(x,y-dy)==NEUMANN) return -bc_hodge.wallValue(x,y-dy);
      else                                   return (hodge_q - bc_hodge.wallValue(x,y-dy)) * 2 / dy;
    case dir::f_0p0:
      if(bc_hodge.wallType(x,y+dy)==NEUMANN) return  bc_hodge.wallValue(x,y+dy);
      else                                   return (bc_hodge.wallValue(x,y+dy) - hodge_q) * 2 / dy;
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

    /* multiple neighbor cells */
    if(ngbd.size()>1)
    {
      double dist = 0;
      double grad_hodge = 0;
      for(unsigned int m=0; m<ngbd.size(); ++m)
      {
        dist += (double) ngbd[m].level * .5*((double)P4EST_QUADRANT_LEN(quad->level) + (double)P4EST_QUADRANT_LEN(ngbd[m].level));
        grad_hodge += (hodge_p[quad_idx] - hodge_p[ngbd[m].p.piggy3.local_num]) * (double) ngbd[m].level;
      }
      ierr = VecRestoreArrayRead(hodge, &hodge_p); CHKERRXX(ierr);
      return dir%2==0 ? grad_hodge/dist : -grad_hodge/dist;
    }
    /* one neighbor cell of same size, check for interface */
    else if(ngbd[0].level == quad->level)
    {
      double xq = quad_x_fr_q(quad_idx, tree_idx, p4est_n, ghost_n);
      double yq = quad_y_fr_q(quad_idx, tree_idx, p4est_n, ghost_n);

      double x0 = quad_x_fr_q(ngbd[0].p.piggy3.local_num, ngbd[0].p.piggy3.which_tree, p4est_n, ghost_n);
      double y0 = quad_y_fr_q(ngbd[0].p.piggy3.local_num, ngbd[0].p.piggy3.which_tree, p4est_n, ghost_n);

#ifdef P4_TO_P8
      double zq = quad_z_fr_q(quad_idx, tree_idx, p4est, ghost);
      double z0 = quad_z_fr_q(ngbd[0].p.piggy3.local_num, ngbd[0].p.piggy3.which_tree, p4est, ghost);
      double phi_q = (*interp_phi)(xq, yq, zq);
      double phi_0 = (*interp_phi)(x0, y0, z0);
#else
      double phi_q = (*interp_phi)(xq, yq);
      double phi_0 = (*interp_phi)(x0, y0);
#endif

      double dx = (double) P4EST_QUADRANT_LEN(quad->level)/(double) P4EST_ROOT_LEN;
      double dy = dx;
#ifdef P4_TO_P8
      double dz = dx;
#endif

      if(bc_hodge.interfaceType()==DIRICHLET && phi_q*phi_0<0)
      {
        if(phi_q>0)
        {
          double phi_tmp = phi_q; phi_q = phi_0; phi_0 = phi_tmp;
          dir += 1;
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

        double theta = fraction_Interval_Covered_By_Irregular_Domain(phi_q, phi_0, dx, dy);
        if(theta<EPS) theta = EPS; if(theta>1) theta = 1;
        double val_interface;
        double dist;
        switch(dir)
        {
#ifdef P4_TO_P8
        case dir::f_m00: case dir::f_p00: dist = dx*theta; val_interface = bc_hodge.interfaceValue(xq + (dir%2==0 ? -1 : 1)*theta*dx, yq, zq); break;
        case dir::f_0m0: case dir::f_0p0: dist = dy*theta; val_interface = bc_hodge.interfaceValue(xq, yq + (dir%2==0 ? -1 : 1)*theta*dy, zq); break;
        case dir::f_00m: case dir::f_00p: dist = dz*theta; val_interface = bc_hodge.interfaceValue(xq, yq, zq + (dir%2==0 ? -1 : 1)*theta*dz); break;
#else
        case dir::f_m00: case dir::f_p00: dist = dx*theta; val_interface = bc_hodge.interfaceValue(xq + (dir%2==0 ? -1 : 1)*theta*dx, yq); break;
        case dir::f_0m0: case dir::f_0p0: dist = dy*theta; val_interface = bc_hodge.interfaceValue(xq, yq + (dir%2==0 ? -1 : 1)*theta*dy); break;
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
      for(unsigned int m=0; m<ngbd.size(); ++m)
      {
        dist += (double) ngbd[m].level * .5*((double)P4EST_QUADRANT_LEN(quad_tmp.level) + (double)P4EST_QUADRANT_LEN(ngbd[m].level));
        grad_hodge += (hodge_p[ngbd[m].p.piggy3.local_num] - hodge_p[quad_tmp.p.piggy3.local_num]) * (double) ngbd[m].level;
      }
      ierr = VecRestoreArrayRead(hodge, &hodge_p); CHKERRXX(ierr);
      return dir%2==0 ? grad_hodge/dist : -grad_hodge/dist;
    }
  }
}





void my_p4est_navier_stokes_t::solve_viscosity()
{
  PetscErrorCode ierr;

  double alpha = (2*dt_n+dt_nm1)/(dt_n+dt_nm1);
  double beta = -dt_n/(dt_n+dt_nm1);

  /* construct the right hand side */
  std::vector<double> xyz_nm1[P4EST_DIM];
  std::vector<double> xyz_n  [P4EST_DIM];
  Vec rhs[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    /* backtrace the nodes with semi-Lagrangian / BDF scheme */
    for(int dd=0; dd<P4EST_DIM; ++dd)
    {
      xyz_nm1[dd].resize(faces_n->num_local[dir]);
      xyz_n  [dd].resize(faces_n->num_local[dir]);
    }
    trajectory_from_np1_to_nm1(p4est_n, faces_n, ngbd_nm1, ngbd_n, vnm1_nodes, vn_nodes, dt_nm1, dt_n, xyz_nm1, xyz_n, dir);

    /* find the velocity at the backtraced points */
    my_p4est_interpolation_nodes_t interp_nm1(ngbd_nm1);
    my_p4est_interpolation_nodes_t interp_n  (ngbd_n  );
    for(p4est_locidx_t f_idx=0; f_idx<faces_n->num_local[dir]; ++f_idx)
    {
#ifdef P4_TO_P8
      double xyz_tmp[] = {xyz_nm1[0][f_idx], xyz_nm1[1][f_idx], xyz_nm1[2][f_idx]};
#else
      double xyz_tmp[] = {xyz_nm1[0][f_idx], xyz_nm1[1][f_idx]};
#endif
      interp_nm1.add_point(f_idx, xyz_tmp);

#ifdef P4_TO_P8
      xyz_tmp[0] = xyz_nm1[0][f_idx]; xyz_tmp[1] = xyz_nm1[1][f_idx]; xyz_tmp[2] = xyz_nm1[2][f_idx];
#else
      xyz_tmp[0] = xyz_nm1[0][f_idx]; xyz_tmp[1] = xyz_nm1[1][f_idx];
#endif
      interp_n.add_point(f_idx, xyz_tmp);
    }

    std::vector<double> vnm1_faces(faces_n->num_local[dir]);
    interp_nm1.set_input(vnm1_nodes[dir], quadratic);
    interp_nm1.interpolate(vnm1_faces.data());

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
        rhs_p[f_idx] = -rho * ( (-alpha/dt_n + beta/dt_nm1)*vn_faces[f_idx] - beta/dt_nm1*vnm1_faces[f_idx]);

        if(external_forces!=NULL)
        {
          double xyz[P4EST_DIM];
          faces_n->xyz_fr_f(f_idx, dir, xyz);
#ifdef P4_TO_P8
          rhs_p[f_idx] += external_forces[dir](xyz[0], xyz[1], xyz[2]);
#else
          rhs_p[f_idx] += external_forces[dir](xyz[0], xyz[1]);
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
  solver.set_bc(bc_vstar);
  solver.set_rhs(rhs);

  solver.solve(vstar);

  for(int dir=0; dir<P4EST_DIM; ++dir)
    if(VecIsNan(vstar[dir]))
      std::cout << "NAN in vstar[" << dir << "]" << std::endl;
  // bousouf
//  ierr = VecView(rhs[0], PETSC_VIEWER_STDOUT_WORLD);

  if(bc_pressure->interfaceType()!=NOINTERFACE)
  {
    my_p4est_level_set_faces_t lsf(ngbd_n, faces_n);
    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      lsf.extend_Over_Interface(phi, vstar[dir], bc_vstar[dir], dir, face_is_well_defined[dir], 2, 8);
      ierr = VecDestroy(rhs[dir]); CHKERRXX(ierr);
    }
  }
}




/* solve the projection step
 * laplace Hodge = -div(vstar)
 */
void my_p4est_navier_stokes_t::solve_projection()
{
  PetscErrorCode ierr;

  Vec rhs;
  ierr = VecDuplicate(hodge, &rhs); CHKERRXX(ierr);

  double *rhs_p;
  ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);

  const double *vstar_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGetArrayRead(vstar[dir], &vstar_p[dir]); CHKERRXX(ierr);
  }

  std::vector<p4est_quadrant_t> ngbd;
  p4est_quadrant_t quad_tmp;

  /* compute the right-hand-side */
  for(p4est_topidx_t tree_idx=p4est_n->first_local_tree; tree_idx<=p4est_n->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est_n->trees, tree_idx);
    for(size_t q_idx=0; q_idx<tree->quadrants.elem_count; ++q_idx)
    {
      p4est_locidx_t quad_idx = q_idx+tree->quadrants_offset;
      p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q_idx);

      rhs_p[quad_idx] = 0;
      double dx = (double)P4EST_QUADRANT_LEN(quad->level) / (double)P4EST_ROOT_LEN;

      for(int dir=0; dir<P4EST_DIM; ++dir)
      {
        double vm = 0;
        if(is_quad_Wall(p4est_n, tree_idx, quad, 2*dir))
        {
          vm = vstar_p[dir][faces_n->q2f(quad_idx, 2*dir)];
        }
        else if(faces_n->q2f(quad_idx, 2*dir)!=NO_VELOCITY)
        {
          ngbd.resize(0);
          ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 2*dir);
          quad_tmp = ngbd[0];
          ngbd.resize(0);
          ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_tmp.p.piggy3.local_num, quad_tmp.p.piggy3.which_tree, 2*dir+1);
          for(unsigned int m=0; m<ngbd.size(); ++m)
            vm += (double)ngbd[m].level * vstar_p[dir][faces_n->q2f(ngbd[m].p.piggy3.local_num, 2*dir)];
          vm /= (double) quad->level;
        }
        else
        {
          ngbd.resize(0);
          ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 2*dir);
          for(unsigned int m=0; m<ngbd.size(); ++m)
            vm += (double)ngbd[m].level * vstar_p[dir][faces_n->q2f(ngbd[m].p.piggy3.local_num, 2*dir+1)];
          vm /= (double) quad->level;
        }

        double vp = 0;
        if(is_quad_Wall(p4est_n, tree_idx, quad, 2*dir+1))
        {
          vp = vstar_p[dir][faces_n->q2f(quad_idx, 2*dir+1)];
        }
        else if(faces_n->q2f(quad_idx, 2*dir+1)!=NO_VELOCITY)
        {
          ngbd.resize(0);
          ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 2*dir+1);
          quad_tmp = ngbd[0];
          ngbd.resize(0);
          ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_tmp.p.piggy3.local_num, quad_tmp.p.piggy3.which_tree, 2*dir);
          for(unsigned int m=0; m<ngbd.size(); ++m)
            vp += (double)ngbd[m].level * vstar_p[dir][faces_n->q2f(ngbd[m].p.piggy3.local_num, 2*dir+1)];
          vp /= (double) quad->level;
        }
        else
        {
          ngbd.resize(0);
          ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 2*dir+1);
          for(unsigned int m=0; m<ngbd.size(); ++m)
            vp += (double)ngbd[m].level * vstar_p[dir][faces_n->q2f(ngbd[m].p.piggy3.local_num, 2*dir)];
          vp /= (double) quad->level;
        }

        rhs_p[quad_idx] -= (vp-vm)/dx;
      }
    }
  }

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArrayRead(vstar[dir], &vstar_p[dir]); CHKERRXX(ierr);
  }

  ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);

  /* solve the linear system */
  my_p4est_poisson_cells_t solver(ngbd_c, ngbd_n);
  solver.set_phi(phi);
  solver.set_mu(1);
  solver.set_bc(bc_hodge);
  solver.set_rhs(rhs);

  solver.solve(hodge);

  ierr = VecDestroy(rhs); CHKERRXX(ierr);

  if(bc_pressure->interfaceType()!=NOINTERFACE)
  {
    my_p4est_level_set_cells_t lsc(ngbd_c, ngbd_n);
    lsc.extend_Over_Interface(phi, hodge, &bc_hodge, 2, 8);
  }

  /* project vstar */
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    double *d_hodge_p;
    ierr = VecGetArray(dxyz_hodge[dir], &d_hodge_p); CHKERRXX(ierr);

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
      d_hodge_p[f_idx] = compute_dxyz_hodge(quad_idx, tree_idx, 2*dir+tmp);
    }

    ierr = VecGhostUpdateBegin(dxyz_hodge[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    for(p4est_locidx_t f_idx=0; f_idx<faces_n->num_local[dir]; ++f_idx)
    {
      vnp1_p[f_idx] = vstar_p[f_idx] - d_hodge_p[f_idx];
    }

    ierr = VecGhostUpdateEnd  (dxyz_hodge[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecRestoreArray(vnp1[dir], &vnp1_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(vstar[dir], &vstar_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(dxyz_hodge[dir], &d_hodge_p); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(vnp1[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (vnp1[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    if(bc_pressure->interfaceType()!=NOINTERFACE)
    {
      my_p4est_level_set_faces_t lsf(ngbd_n, faces_n);
      lsf.extend_Over_Interface(phi, vnp1[dir], bc_v[dir], dir, face_is_well_defined[dir], 2, 8);
    }
  }

  for(int dir=0; dir<P4EST_DIM; ++dir)
    if(VecIsNan(vnp1[dir]))
      std::cout << "NAN in vnp1[" << dir << "]" << std::endl;
}



void my_p4est_navier_stokes_t::compute_dt()
{
  /* interpolate vnp1 from faces to nodes */
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    my_p4est_interpolation_faces_t interp(ngbd_n, faces_n);
    for(p4est_locidx_t n=0; n<nodes_n->num_owned_indeps; ++n)
    {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, p4est_n, nodes_n, xyz);
      interp.add_point(n, xyz);
    }
    interp.set_input(vnp1[dir], face_is_well_defined[dir], dir);
    interp.interpolate(vnp1_nodes[dir]);

    if(bc_pressure->interfaceType()!=NOINTERFACE)
    {
      my_p4est_level_set_t lsn(ngbd_n);
      lsn.extend_Over_Interface_TVD(phi, vnp1_nodes[dir]);
    }

    PetscErrorCode ierr;
    ierr = VecGhostUpdateBegin(vnp1_nodes[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (vnp1_nodes[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  for(int dir=0; dir<P4EST_DIM; ++dir)
    if(VecIsNan(vn_nodes[dir]))
      std::cout << "NAN in vnp1_nodes[" << dir << "]" << std::endl;

  compute_vorticity();
  compute_norm_grad_v();
  compute_max_L2_norm_u();

  dt_nm1 = dt_n;
#ifdef P4_TO_P8
  dt_n = MIN(1., 1/max_L2_norm_u) * n_times_dt * MIN(dxyz_min[0], dxyz_min[1], dxyz_min[2]);
#else
  dt_n = MIN(1., 1/max_L2_norm_u) * n_times_dt * MIN(dxyz_min[0], dxyz_min[1]);
#endif

  dt_updated = true;
}




void my_p4est_navier_stokes_t::update_from_tn_to_tnp1()
{
  PetscErrorCode ierr;

  if(!dt_updated)
    compute_dt();

  dt_updated = false;

  /* construct the new forest */
  p4est_t *p4est_np1 = p4est_copy(p4est_n, P4EST_FALSE);
  p4est_np1->user_pointer = p4est_n->user_pointer;
  my_p4est_partition(p4est_np1, P4EST_FALSE, NULL);
  p4est_ghost_t *ghost_np1 = p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(p4est_np1, ghost_np1);
  p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);
  my_p4est_hierarchy_t *hierarchy_np1 = new my_p4est_hierarchy_t(p4est_np1, ghost_np1, brick);
  my_p4est_cell_neighbors_t *ngbd_c_np1 = new my_p4est_cell_neighbors_t(hierarchy_np1);
  my_p4est_node_neighbors_t *ngbd_np1 = new my_p4est_node_neighbors_t(hierarchy_np1, nodes_np1);
  my_p4est_faces_t *faces_np1 = new my_p4est_faces_t(p4est_np1, ghost_np1, brick, ngbd_c_np1);

  my_p4est_interpolation_nodes_t interp_new_phi(ngbd_np1);
  for(p4est_locidx_t n=0; n<nodes_np1->num_owned_indeps; ++n)
  {
    double xyz[P4EST_DIM];
    node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);
    interp_new_phi.add_point(n, xyz);
  }
  Vec phi_np1;
  ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);
  interp_new_phi.set_input(phi, quadratic_non_oscillatory);
  interp_new_phi.interpolate(phi_np1);
  ierr = VecDestroy(phi); CHKERRXX(ierr);
  phi = phi_np1;
  interp_new_phi.clear();

  ierr = VecGhostUpdateBegin(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  /* interpolate the quantities on the new forest at the nodes */
  my_p4est_interpolation_nodes_t interp_nodes(ngbd_n);
  for(p4est_locidx_t n=0; n<nodes_np1->num_owned_indeps; ++n)
  {
    double xyz[P4EST_DIM];
    node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);
    interp_nodes.add_point(n, xyz);
  }


  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecDestroy(vnm1_nodes[dir]); CHKERRXX(ierr);
    vnm1_nodes[dir] = vn_nodes[dir];

    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &vn_nodes[dir]); CHKERRXX(ierr);
    interp_nodes.set_input(vnp1_nodes[dir], quadratic_non_oscillatory);
    interp_nodes.interpolate(vn_nodes[dir]); CHKERRXX(ierr);


    if(VecIsNan(vn_nodes[dir]))
      std::cout << "NAN middle update in vn_nodes[" << dir << "]" << std::endl;

    ierr = VecGhostUpdateBegin(vn_nodes[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  ierr = VecGhostUpdateEnd(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecDestroy(vnp1_nodes[dir]); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &vnp1_nodes[dir]); CHKERRXX(ierr);
#ifdef P4_TO_P8
    Vec vort;
    ierr = VecDuplicate(phi, &vort); CHKERRXX(ierr);
    interp_nodes.set_input(vorticity[dir], quadratic_non_oscillatory);
    interp_nodes.interpolate(vort);

    ierr = VecDestroy(vorticity[dir]); CHKERRXX(ierr);
    vorticity[dir] = vort;
#endif
  }

#ifndef P4_TO_P8
    Vec vort;
    ierr = VecDuplicate(phi, &vort); CHKERRXX(ierr);
    interp_nodes.set_input(vorticity, quadratic_non_oscillatory);
    interp_nodes.interpolate(vort);

    ierr = VecDestroy(vorticity); CHKERRXX(ierr);
    vorticity = vort;
#endif
  interp_nodes.clear();

  /* interpolate the quantities on the new forest at the cells */
  my_p4est_interpolation_cells_t interp_cells(ngbd_c, ngbd_n);
  Vec hodge_np1;
  ierr = VecCreateGhostCells(p4est_np1, ghost_np1, &hodge_np1); CHKERRXX(ierr);
  for(p4est_topidx_t tree_idx=p4est_np1->first_local_tree; tree_idx<=p4est_np1->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est_np1->trees, tree_idx);
    for(size_t q=0; q<tree->quadrants.elem_count; ++q)
    {
      p4est_locidx_t quad_idx = q+tree->quadrants_offset;
      double xyz[P4EST_DIM];
      quad_xyz_fr_q(quad_idx, tree_idx, p4est_np1, ghost_np1, xyz);
      interp_cells.add_point(quad_idx, xyz);
    }
  }
  /* NOTE: bousouf in the original code the extrapolated values are used ... */
  interp_cells.set_input(hodge, phi, &bc_hodge);
  interp_cells.interpolate(hodge_np1);
  interp_cells.clear();

  ierr = VecDestroy(hodge); CHKERRXX(ierr);
  hodge = hodge_np1;
  ierr = VecGhostUpdateBegin(hodge, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);


  /* interpolate the quantities on the new forest at the faces */
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecDestroy(face_is_well_defined[dir]); CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est_np1, faces_np1, &face_is_well_defined[dir], dir); CHKERRXX(ierr);
    check_if_faces_are_well_defined(p4est_np1, ngbd_np1, faces_np1, dir, phi, bc_v[dir].interfaceType(), face_is_well_defined[dir]);

    my_p4est_interpolation_faces_t interp_faces(ngbd_n, faces_n);
    for(p4est_locidx_t f=0; f<faces_np1->num_local[f]; ++f)
    {
      double xyz[P4EST_DIM];
      faces_np1->xyz_fr_f(f, dir, xyz);
      interp_faces.add_point(f, xyz);
    }
    Vec dxyz_hodge_tmp;
    ierr = VecDuplicate(face_is_well_defined[dir], &dxyz_hodge_tmp); CHKERRXX(ierr);
    interp_faces.set_input(dxyz_hodge[dir], face_is_well_defined[dir], dir);
    interp_faces.interpolate(dxyz_hodge_tmp);


    ierr = VecDestroy(dxyz_hodge[dir]); CHKERRXX(ierr);
    dxyz_hodge[dir] = dxyz_hodge_tmp;

    ierr = VecGhostUpdateBegin(dxyz_hodge[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecDestroy(vstar[dir]); CHKERRXX(ierr);
    ierr = VecDuplicate(face_is_well_defined[dir], &vstar[dir]); CHKERRXX(ierr);

    ierr = VecDestroy(vnp1[dir]); CHKERRXX(ierr);
    ierr = VecDuplicate(face_is_well_defined[dir], &vnp1[dir]); CHKERRXX(ierr);
  }

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGhostUpdateEnd(vn_nodes[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(dxyz_hodge[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }
  ierr = VecGhostUpdateEnd(hodge, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  /* update the variables */
  p4est_destroy(p4est_nm1); p4est_nm1 = p4est_n; p4est_n = p4est_np1;
  p4est_ghost_destroy(ghost_nm1); ghost_nm1 = ghost_n; ghost_n = ghost_np1;
  p4est_nodes_destroy(nodes_nm1); nodes_nm1 = nodes_n; nodes_n = nodes_np1;
  delete hierarchy_nm1; hierarchy_nm1 = hierarchy_n; hierarchy_n = hierarchy_np1;
  delete ngbd_nm1; ngbd_nm1 = ngbd_n; ngbd_n = ngbd_np1;
  delete ngbd_c; ngbd_c = ngbd_c_np1;
  delete faces_n; faces_n = faces_np1;

  delete interp_phi;
  interp_phi = new my_p4est_interpolation_nodes_t(ngbd_n);
  interp_phi->set_input(phi, linear);

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    delete interp_dxyz_hodge[dir];
    interp_dxyz_hodge[dir] = new my_p4est_interpolation_faces_t(ngbd_n, faces_n);
    interp_dxyz_hodge[dir]->set_input(dxyz_hodge[dir], face_is_well_defined[dir], dir);
  }

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    if(VecIsNan(vn_nodes[dir]))
      std::cout << "NAN after update in vn_nodes[" << dir << "]" << std::endl;

    if(VecIsNan(vnm1_nodes[dir]))
      std::cout << "NAN after update in vnm1_nodes[" << dir << "]" << std::endl;

    if(VecIsNan(vnp1[dir]))
      std::cout << "NAN after update in vnp1[" << dir << "]" << std::endl;
  }
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
    ierr = VecGetArrayRead(vn_nodes[dir], &vn_p[dir]); CHKERRXX(ierr);
  }

  my_p4est_vtk_write_all(p4est_n, nodes_n, ghost_n,
                         P4EST_TRUE, P4EST_TRUE,
                         1+P4EST_DIM, /* number of VTK_POINT_DATA */
                         1, /* number of VTK_CELL_DATA  */
                         name,
                         VTK_POINT_DATA, "phi", phi_p,
                         VTK_POINT_DATA, "vx", vn_p[0],
                         VTK_POINT_DATA, "vy", vn_p[1],
    #ifdef P4_TO_P8
                         VTK_POINT_DATA, "vz", vn_p[2],
    #endif
                         VTK_CELL_DATA, "hodge", hodge_p);


  ierr = VecRestoreArrayRead(phi  , &phi_p  ); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(hodge, &hodge_p); CHKERRXX(ierr);

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArrayRead(vn_nodes[dir], &vn_p[dir]); CHKERRXX(ierr);
  }

  ierr = PetscPrintf(p4est_n->mpicomm, "Saved visual data in ... %s\n", name); CHKERRXX(ierr);
}
