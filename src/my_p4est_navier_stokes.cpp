#include "my_p4est_navier_stokes.h"

#ifdef P4_TO_P8
#include <src/my_p8est_poisson_cells.h>
#include <src/my_p8est_poisson_faces.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_level_set_cells.h>
#include <src/my_p8est_level_set_faces.h>
#else
#include <src/my_p4est_poisson_cells.h>
#include <src/my_p4est_poisson_faces.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_level_set_cells.h>
#include <src/my_p4est_level_set_faces.h>
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



my_p4est_navier_stokes_t::my_p4est_navier_stokes_t(my_p4est_node_neighbors_t *ngbd_n, my_p4est_faces_t *faces)
  : brick(ngbd_n->myb), p4est(ngbd_n->p4est), ghost(ngbd_n->ghost), nodes(ngbd_n->nodes),
    ngbd_c(faces->ngbd_c), ngbd_n(ngbd_n), faces(faces),
    wall_bc_value_hodge(this), interface_bc_value_hodge(this)
{
  PetscErrorCode ierr;

  ierr = VecCreateGhostNodes(p4est, nodes, &phi); CHKERRXX(ierr);
  Vec vec_loc;
  ierr = VecGhostGetLocalForm(phi, &vec_loc); CHKERRXX(ierr);
  ierr = VecSet(vec_loc, -1); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(phi, &vec_loc); CHKERRXX(ierr);

  ierr = VecCreateGhostCells(p4est, ghost, &hodge); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(hodge, &vec_loc); CHKERRXX(ierr);
  ierr = VecSet(vec_loc, 0); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(phi, &vec_loc); CHKERRXX(ierr);

  bc_v = NULL;
  external_forces = NULL;

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    vnp1[dir] = NULL;
    vstar[dir] = NULL;
#ifdef P4_TO_P8
    vorticity[dir] = NULL;
#endif

    wall_bc_value_vstar[dir] = new wall_bc_value_vstar_t(this, dir);
    interface_bc_value_vstar[dir] = new interface_bc_value_vstar_t(this, dir);

    ierr = VecCreateGhostFaces(p4est, faces, &dxyz_hodge[dir], dir); CHKERRXX(ierr);
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

void my_p4est_navier_stokes_t::set_parameters(double mu, double rho, double uniform_band, double threshold_split_cell)
{
  this->mu = mu;
  this->rho = rho;
  this->uniform_band = uniform_band;
  this->threshold_split_cell = threshold_split_cell;
}


void my_p4est_navier_stokes_t::set_phi(Vec phi)
{
  PetscErrorCode ierr;
  if(this->phi!=NULL) { ierr = VecDestroy(this->phi); CHKERRXX(ierr); }
  this->phi = phi;

  if(bc_v!=NULL)
  {
    for(int dir=0; dir<P4EST_DIM; ++dir)
      check_if_faces_are_well_defined(p4est, ngbd_n, faces, dir, phi, bc_v[dir].interfaceType(), face_is_well_defined[dir]);
  }
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
      check_if_faces_are_well_defined(p4est, ngbd_n, faces, dir, phi, bc_v[dir].interfaceType(), face_is_well_defined[dir]);
  }
}


void my_p4est_navier_stokes_t::set_velocities(Vec *vnm1_nodes, Vec *vn_nodes)
{
  PetscErrorCode ierr;
#ifdef CASL_THROWS

#endif

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

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
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
}


void my_p4est_navier_stokes_t::compute_vorticity()
{
  PetscErrorCode ierr;

  quad_neighbor_nodes_of_node_t qnnn;

  const double *vnp1_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; dir++) { ierr = VecGetArrayRead(vnp1[dir], &vnp1_p[dir]); CHKERRXX(ierr); }

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

  for(int dir=0; dir<P4EST_DIM; dir++) { ierr = VecRestoreArrayRead(vnp1[dir], &vnp1_p[dir]); CHKERRXX(ierr); }
}



double my_p4est_navier_stokes_t::compute_dxyz_hodge(p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, char dir)
{
  PetscErrorCode ierr;

  p4est_quadrant_t *quad;
  if(quad_idx<p4est->local_num_quadrants)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);
  }
  else
    quad = (p4est_quadrant_t*)sc_array_index(&ghost->ghosts, quad_idx-p4est->local_num_quadrants);

  const double *hodge_p;
  ierr = VecGetArrayRead(hodge, &hodge_p); CHKERRXX(ierr);

  if(is_quad_Wall(p4est, tree_idx, quad, dir))
  {
    double x = quad_x_fr_q(quad_idx, tree_idx, p4est, ghost);
    double y = quad_y_fr_q(quad_idx, tree_idx, p4est, ghost);
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
      double xq = quad_x_fr_q(quad_idx, tree_idx, p4est, ghost);
      double yq = quad_y_fr_q(quad_idx, tree_idx, p4est, ghost);

      double x0 = quad_x_fr_q(ngbd[0].p.piggy3.local_num, ngbd[0].p.piggy3.which_tree, p4est, ghost);
      double y0 = quad_y_fr_q(ngbd[0].p.piggy3.local_num, ngbd[0].p.piggy3.which_tree, p4est, ghost);

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

  Vec rhs[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecDuplicate(vstar[dir], &rhs[dir]); CHKERRXX(ierr);

    double *rhs_p;
    ierr = VecGetArray(rhs[dir], &rhs_p); CHKERRXX(ierr);

    const PetscScalar *face_is_well_defined_p;
    ierr = VecGetArrayRead(face_is_well_defined[dir], &face_is_well_defined_p); CHKERRXX(ierr);

    /* compute the advection term with semi lagrangian */

    ierr = VecRestoreArray(rhs[dir], &rhs_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(face_is_well_defined[dir], &face_is_well_defined_p); CHKERRXX(ierr);
  }

  my_p4est_poisson_faces_t solver(faces, ngbd_n);
  solver.set_phi(phi);
  solver.set_mu(mu);
  solver.set_diagonal(alpha * rho/dt_n);
  solver.set_bc(bc_vstar);
  solver.set_rhs(rhs);

  solver.solve(vstar);

  my_p4est_level_set_faces_t lsf(ngbd_n, faces);

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    lsf.extend_Over_Interface(phi, vstar[dir], bc_vstar[dir], dir, face_is_well_defined[dir], 2, 8);
    ierr = VecDestroy(rhs[dir]); CHKERRXX(ierr);
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
  for(p4est_topidx_t tree_idx=p4est->first_local_tree; tree_idx<=p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for(size_t q_idx=0; q_idx<tree->quadrants.elem_count; ++q_idx)
    {
      p4est_locidx_t quad_idx = q_idx+tree->quadrants_offset;
      p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q_idx);

      rhs_p[quad_idx] = 0;
      double dx = (double)P4EST_QUADRANT_LEN(quad->level) / (double)P4EST_ROOT_LEN;

      for(int dir=0; dir<P4EST_DIM; ++dir)
      {
        double vm = 0;
        if(is_quad_Wall(p4est, tree_idx, quad, 2*dir))
        {
          vm = vstar_p[dir][faces->q2f(quad_idx, 2*dir)];
        }
        else if(faces->q2f(quad_idx, 2*dir)!=NO_VELOCITY)
        {
          ngbd.resize(0);
          ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 2*dir);
          quad_tmp = ngbd[0];
          ngbd.resize(0);
          ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_tmp.p.piggy3.local_num, quad_tmp.p.piggy3.which_tree, 2*dir+1);
          for(unsigned int m=0; m<ngbd.size(); ++m)
            vm += (double)ngbd[m].level * vstar_p[dir][faces->q2f(ngbd[m].p.piggy3.local_num, 2*dir)];
          vm /= (double) quad->level;
        }
        else
        {
          ngbd.resize(0);
          ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 2*dir);
          for(unsigned int m=0; m<ngbd.size(); ++m)
            vm += (double)ngbd[m].level * vstar_p[dir][faces->q2f(ngbd[m].p.piggy3.local_num, 2*dir+1)];
          vm /= (double) quad->level;
        }

        double vp = 0;
        if(is_quad_Wall(p4est, tree_idx, quad, 2*dir+1))
        {
          vp = vstar_p[dir][faces->q2f(quad_idx, 2*dir+1)];
        }
        else if(faces->q2f(quad_idx, 2*dir+1)!=NO_VELOCITY)
        {
          ngbd.resize(0);
          ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 2*dir+1);
          quad_tmp = ngbd[0];
          ngbd.resize(0);
          ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_tmp.p.piggy3.local_num, quad_tmp.p.piggy3.which_tree, 2*dir);
          for(unsigned int m=0; m<ngbd.size(); ++m)
            vp += (double)ngbd[m].level * vstar_p[dir][faces->q2f(ngbd[m].p.piggy3.local_num, 2*dir+1)];
          vp /= (double) quad->level;
        }
        else
        {
          ngbd.resize(0);
          ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, 2*dir+1);
          for(unsigned int m=0; m<ngbd.size(); ++m)
            vp += (double)ngbd[m].level * vstar_p[dir][faces->q2f(ngbd[m].p.piggy3.local_num, 2*dir)];
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

  my_p4est_level_set_cells_t lsc(ngbd_c, ngbd_n);
  lsc.extend_Over_Interface(phi, hodge, &bc_hodge, 2, 8);

  my_p4est_level_set_faces_t lsf(ngbd_n, faces);

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

    for(p4est_locidx_t f_idx=0; f_idx<faces->num_local[dir]; ++f_idx)
    {
      faces->f2q(f_idx, dir, quad_idx, tree_idx);
      int tmp = faces->q2f(quad_idx, 2*dir)==f_idx ? 0 : 1;
      d_hodge_p[f_idx] = compute_dxyz_hodge(quad_idx, tree_idx, 2*dir+tmp);
    }

    ierr = VecGhostUpdateBegin(dxyz_hodge[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    for(p4est_locidx_t f_idx=0; f_idx<faces->num_local[dir]; ++f_idx)
    {
      vnp1_p[f_idx] = vstar_p[f_idx] - d_hodge_p[f_idx];
    }

    ierr = VecGhostUpdateEnd  (dxyz_hodge[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecRestoreArray(vnp1[dir], &vnp1_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(vstar[dir], &vstar_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(dxyz_hodge[dir], &d_hodge_p); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(vnp1[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (vnp1[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    lsf.extend_Over_Interface(phi, vnp1[dir], bc_v[dir], dir, face_is_well_defined[dir], 2, 8);
  }
}
