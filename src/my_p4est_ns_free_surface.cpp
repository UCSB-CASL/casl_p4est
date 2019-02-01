#include "my_p4est_ns_free_surface.h"

#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_my_p4est_ns_free_surface_advection;
extern PetscLogEvent log_my_p4est_ns_free_surface_projection;
extern PetscLogEvent log_my_p4est_ns_free_surface_update
#endif

#ifdef P4_TO_P8
#include <src/my_p8est_poisson_cells.h>
#include <src/my_p8est_poisson_faces.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_level_set_cells.h>
#include <src/my_p8est_level_set_faces.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_trajectory_of_point.h>
#include <src/my_p8est_vtk.h>
#else
#include <src/my_p4est_poisson_cells.h>
#include <src/my_p4est_poisson_faces.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_level_set_cells.h>
#include <src/my_p4est_level_set_faces.h>
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_trajectory_of_point.h>
#include <src/my_p4est_vtk.h>
#endif

my_p4est_ns_free_surface_t::my_p4est_ns_free_surface_t(my_p4est_node_neighbors_t *ngbd_nm1, my_p4est_node_neighbors_t *ngbd_n, my_p4est_faces_t *faces_n):
  my_p4est_navier_stokes_t(ngbd_nm1, ngbd_n, faces_n),
  wall_bc_value_pressure(this, PRESSURE), wall_bc_value_vx(this, VELOCITY_X), wall_bc_value_vy(this, VELOCITY_Y),
  #ifdef P4_TO_P8
  wall_bc_value_vz(this, VELOCITY_Z),
  #endif
  interface_bc_hodge_global(this, HODGE), interface_bc_pressure_global(this, PRESSURE), interface_bc_vx_global(this, VELOCITY_X), interface_bc_vy_global(this, VELOCITY_Y)
#ifdef P4_TO_P8
, interface_bc_vz_global(this, VELOCITY_Z)
#endif
{
  PetscErrorCode ierr;
  surf_tension = 1.0;
  interface_bc_v_global[0] = &interface_bc_vx_global; wall_bc_value_v[0] = &wall_bc_value_vx;
  interface_bc_v_global[1] = &interface_bc_vy_global; wall_bc_value_v[1] = &wall_bc_value_vy;
#ifdef P4_TO_P8
  interface_bc_v_global[2] = &interface_bc_vz_global; wall_bc_value_v[2] = &wall_bc_value_vz;
  finest_diag = sqrt(SQR(dxyz_min[0]) + SQR(dxyz_min[1]) + SQR(dxyz_min[2]));
#else
  finest_diag = sqrt(SQR(dxyz_min[0]) + SQR(dxyz_min[1]));
#endif

  Vec vec_loc;
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &fs_phi);
  ierr = VecGhostGetLocalForm(fs_phi, &vec_loc); CHKERRXX(ierr);
  ierr = VecSet(vec_loc, -1.0); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(fs_phi, &vec_loc); CHKERRXX(ierr);
  interp_fs_phi = new my_p4est_interpolation_nodes_t(ngbd_n);
  interp_fs_phi->set_input(fs_phi, linear);

  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &global_phi);
  ierr = VecGhostGetLocalForm(global_phi, &vec_loc); CHKERRXX(ierr);
  ierr = VecSet(vec_loc, -1.0); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(global_phi, &vec_loc); CHKERRXX(ierr);
  interp_global_phi = new my_p4est_interpolation_nodes_t(ngbd_n);
  interp_global_phi->set_input(global_phi, linear);
}

my_p4est_ns_free_surface_t::~my_p4est_ns_free_surface_t()
{
  PetscErrorCode ierr;
  if(fs_phi != NULL) {ierr = VecDestroy(fs_phi); CHKERRXX(ierr);}
  if(global_phi != NULL) {ierr = VecDestroy(global_phi); CHKERRXX(ierr);}
  if(interp_fs_phi!=NULL) delete interp_fs_phi;
  if(interp_global_phi!=NULL) delete interp_global_phi;
}

// by-pass the "well-defined" faces from the parent class (all faces within the solid are well-defined in this case, forced equal to solid velocity).
// set phi for interp_phi input
// update face_is_well_defined
// (re-)build the global phi
void my_p4est_ns_free_surface_t::set_phi(Vec phi_)
{
  PetscErrorCode ierr;
  if(this->phi!=NULL) { ierr = VecDestroy(this->phi); CHKERRXX(ierr); }
  this->phi = phi_;
  interp_phi->set_input(phi, linear);

  build_global_phi_and_face_is_well_defined(nodes_n, phi, fs_phi, global_phi);
}

// the faces are well-defined where fs_phi < 0 or phi > 0, we'll use constant extrapolation where faces are NOT well defined
void my_p4est_ns_free_surface_t::set_free_surface(Vec fs_phi_)
{
  PetscErrorCode ierr;
  if(this->fs_phi!=NULL) { ierr = VecDestroy(this->fs_phi); CHKERRXX(ierr); }
  this->fs_phi = fs_phi_;
  interp_fs_phi->set_input(fs_phi_, linear);

  build_global_phi_and_face_is_well_defined(nodes_n, phi, fs_phi, global_phi);
}

void my_p4est_ns_free_surface_t::set_phis(Vec solid_phi_, Vec fs_phi_)
{
  PetscErrorCode ierr;
  if(this->phi!=NULL) { ierr = VecDestroy(this->phi); CHKERRXX(ierr); }
  if(this->fs_phi!=NULL) { ierr = VecDestroy(this->fs_phi); CHKERRXX(ierr); }
  this->phi = solid_phi_;
  this->fs_phi = fs_phi_;
  interp_phi->set_input(phi, linear);
  interp_fs_phi->set_input(fs_phi_, linear);

  build_global_phi_and_face_is_well_defined(nodes_n, phi, fs_phi, global_phi);
}

void my_p4est_ns_free_surface_t::build_global_phi_and_face_is_well_defined(p4est_nodes_t* nodes,  Vec phi_, Vec fs_phi_, Vec global_phi_)
{
  PetscErrorCode ierr;
  const double *phi_read_p, *fs_phi_read_p;
  double *global_phi_p;
  ierr = VecGetArrayRead(phi_, &phi_read_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(fs_phi_, &fs_phi_read_p); CHKERRXX(ierr);
  ierr = VecGetArray(global_phi_, &global_phi_p); CHKERRXX(ierr);

  for (size_t ni = 0; ni < nodes->indep_nodes.elem_count; ++ni)
    global_phi_p[ni] = ((phi_read_p[ni] > 0.0) && (fs_phi_read_p[ni] > 0.0))? MIN(phi_read_p[ni], fs_phi_read_p[ni]) : MAX(phi_read_p[ni], fs_phi_read_p[ni]);

  ierr = VecRestoreArray(global_phi_, &global_phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(fs_phi_, &fs_phi_read_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(phi_, &phi_read_p); CHKERRXX(ierr);

  interp_global_phi->set_input(global_phi_, linear);

  if(bc_v != NULL)
    for(int dir=0; dir<P4EST_DIM; ++dir)
      check_if_faces_are_well_defined_for_free_surface(faces_n, dir, bc_v_global[dir], face_is_well_defined[dir]);
}


BoundaryConditionType my_p4est_ns_free_surface_t::mixed_interface_bc_t::mixed_type(const double xyz[]) const
{
  // always prioriterize the solid in case of intersection (arbitrary choice!)
  switch (which) {
  case HODGE:
    return (((*_prnt->interp_phi)(xyz) > -1.2*_prnt->finest_diag)? _prnt->bc_hodge.interfaceType() : DIRICHLET); // DIRICHLET ON THE FREE SURFACE FOR THE HODGE VARIABLE
    break;
  case PRESSURE:
    return (((*_prnt->interp_phi)(xyz) > -1.2*_prnt->finest_diag)? _prnt->bc_pressure->interfaceType() : DIRICHLET); // DIRICHLET ON THE FREE SURFACE FOR THE PRESSURE
    break;
  case VELOCITY_X:
    return (((*_prnt->interp_phi)(xyz) > -1.2*_prnt->finest_diag)? _prnt->bc_v[0].interfaceType() : NEUMANN); // NEUMANN ON THE FREE SURFACE FOR THE VELOCITY COMPONENTS
    break;
  case VELOCITY_Y:
    return (((*_prnt->interp_phi)(xyz) > -1.2*_prnt->finest_diag)? _prnt->bc_v[1].interfaceType() : NEUMANN); // NEUMANN ON THE FREE SURFACE FOR THE VELOCITY COMPONENTS
    break;
#ifdef P4_TO_P8
  case VELOCITY_Z:
    return (((*_prnt->interp_phi)(xyz) > -1.2*_prnt->finest_diag)? _prnt->bc_v[2].interfaceType() : NEUMANN); // NEUMANN ON THE FREE SURFACE FOR THE VELOCITY COMPONENTS
    break;
#endif
  default:
    throw std::runtime_error("my_p4est_ns_free_surface_t::mixed_interface_bc_t::mixed_type: unknonw field type...");
    break;
  }
}

#ifdef P4_TO_P8
double my_p4est_ns_free_surface_t::mixed_interface_bc_t::operator ()(double x, double y, double z) const
#else
double my_p4est_ns_free_surface_t::mixed_interface_bc_t::operator ()(double x, double y) const
#endif
{
  // same as before, always prioriterize the solid in case of intersection (arbitrary choice!)
  switch (which) {
  case HODGE:
    // hodge and pressure interface value = 0 for now
#ifdef P4_TO_P8
    return (((*_prnt->interp_phi)(x, y, z) > -1.2*_prnt->finest_diag)? _prnt->bc_hodge.interfaceValue(x, y, z) : _prnt->zero(x, y, z));
#else
    return (((*_prnt->interp_phi)(x, y) > -1.2*_prnt->finest_diag)? _prnt->bc_hodge.interfaceValue(x, y) : _prnt->zero(x, y));
#endif
    break;
  case PRESSURE:
    // hodge and pressure interface value = 0 for now
#ifdef P4_TO_P8
    return (((*_prnt->interp_phi)(x, y, z) > -1.2*_prnt->finest_diag)? _prnt->bc_pressure->interfaceValue(x, y, z) : _prnt->zero(x, y, z));
#else
    return (((*_prnt->interp_phi)(x, y) > -1.2*_prnt->finest_diag)? _prnt->bc_pressure->interfaceValue(x, y) : _prnt->zero(x, y));
#endif
    break;
  case VELOCITY_X:
    // homogeneous Neumann for now
#ifdef P4_TO_P8
    return (((*_prnt->interp_phi)(x, y, z) > -1.2*_prnt->finest_diag)? _prnt->bc_v[0].interfaceValue(x, y, z) : _prnt->zero(x, y, z));
#else
    return (((*_prnt->interp_phi)(x, y) > -1.2*_prnt->finest_diag)? _prnt->bc_v[0].interfaceValue(x, y) : _prnt->zero(x, y));
#endif
    break;
  case VELOCITY_Y:
    // homogeneous Neumann for now
#ifdef P4_TO_P8
    return (((*_prnt->interp_phi)(x, y, z) > -1.2*_prnt->finest_diag)? _prnt->bc_v[1].interfaceValue(x, y, z) : _prnt->zero(x, y, z));
#else
    return (((*_prnt->interp_phi)(x, y) > -1.2*_prnt->finest_diag)? _prnt->bc_v[1].interfaceValue(x, y) : _prnt->zero(x, y));
#endif
    break;
#ifdef P4_TO_P8
  case VELOCITY_Z:
    return (((*_prnt->interp_phi)(x, y, z) > -1.2*_prnt->finest_diag)? _prnt->bc_v[2].interfaceValue(x, y, z) : _prnt->zero(x, y, z)); // homogeneous Neumann extension for now
    break;
#endif
  default:
    throw std::runtime_error("my_p4est_ns_free_surface_t::mixed_interface_bc_t::operator (): unknown field type...");
  }
}


// build our own mixed boundary condition suff
// build our own mixed face_is_well_defined vectors
#ifdef P4_TO_P8
void my_p4est_ns_free_surface_t::set_bc(BoundaryConditions3D *bc_v, BoundaryConditions3D *bc_p)
#else
void my_p4est_ns_free_surface_t::set_bc(BoundaryConditions2D *bc_v, BoundaryConditions2D *bc_p)
#endif
{
  this->bc_v = bc_v;
  this->bc_pressure = bc_p;

  bc_hodge.setWallTypes(bc_pressure->getWallType());
  bc_hodge.setWallValues(wall_bc_value_hodge);
  bc_hodge.setInterfaceType(bc_pressure->interfaceType());
  bc_hodge.setInterfaceValue(interface_bc_value_hodge);

  // additional stuff for our application!
  bc_pressure_global.setWallTypes(bc_pressure->getWallType());
  bc_pressure_global.setWallValues(wall_bc_value_pressure);
  bc_pressure_global.setInterfaceType(MIXED, &interface_bc_pressure_global);
  bc_pressure_global.setInterfaceValue(interface_bc_pressure_global);

  bc_hodge_global.setWallTypes(bc_pressure->getWallType());
  bc_hodge_global.setWallValues(wall_bc_value_hodge);
  bc_hodge_global.setInterfaceType(MIXED, &interface_bc_hodge_global);
  bc_hodge_global.setInterfaceValue(interface_bc_hodge_global);

  for (short dir = 0; dir < P4EST_DIM; ++dir) {
    bc_v_global[dir].setWallTypes(this->bc_v[dir].getWallType());
    bc_v_global[dir].setWallValues(*wall_bc_value_v[dir]);
    bc_v_global[dir].setInterfaceType(MIXED, interface_bc_v_global[dir]);
    bc_v_global[dir].setInterfaceValue(*interface_bc_v_global[dir]);
    check_if_faces_are_well_defined_for_free_surface(faces_n, dir, bc_v_global[dir], face_is_well_defined[dir]);
  }
}

// use global_phi instead of phi
void my_p4est_ns_free_surface_t::compute_max_L2_norm_u()
{
  PetscErrorCode ierr;
  max_L2_norm_u = 0;

  const double *global_phi_p;
  ierr = VecGetArrayRead(global_phi, &global_phi_p); CHKERRXX(ierr);

  const double *v_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir) { ierr = VecGetArrayRead(vnp1_nodes[dir], &v_p[dir]); CHKERRXX(ierr); }

  for(p4est_locidx_t n=0; n<nodes_n->num_owned_indeps; ++n)
  {
    if(global_phi_p[n]<finest_diag)
#ifdef P4_TO_P8
      max_L2_norm_u = MAX(max_L2_norm_u, sqrt(SQR(v_p[0][n]) + SQR(v_p[1][n]) + SQR(v_p[2][n])));
#else
      max_L2_norm_u = MAX(max_L2_norm_u, sqrt(SQR(v_p[0][n]) + SQR(v_p[1][n])));
#endif
  }

  ierr = VecRestoreArrayRead(global_phi, &global_phi_p); CHKERRXX(ierr);
  for(int dir=0; dir<P4EST_DIM; ++dir) { ierr = VecRestoreArrayRead(vnp1_nodes[dir], &v_p[dir]); CHKERRXX(ierr); }

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &max_L2_norm_u, 1, MPI_DOUBLE, MPI_MAX, p4est_n->mpicomm); SC_CHECK_MPI(mpiret);
}

// allow for mixed interface type, contrary to parent's function
// use global bc for hodge!
double my_p4est_ns_free_surface_t::compute_dxyz_hodge(p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, int face_idx)
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

  if(is_quad_Wall(p4est_n, tree_idx, quad, face_idx))
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

    switch(face_idx)
    {
#ifdef P4_TO_P8
    case dir::f_m00:
      if(bc_hodge_global.wallType(x-dx/2,y,z)==NEUMANN) return -bc_hodge_global.wallValue(x-dx/2,y,z);
      else                                              return (hodge_q - bc_hodge_global.wallValue(x-dx/2,y,z)) * 2.0 / dx;
    case dir::f_p00:
      if(bc_hodge_global.wallType(x+dx/2,y,z)==NEUMANN) return  bc_hodge_global.wallValue(x+dx/2,y,z);
      else                                              return (bc_hodge_global.wallValue(x+dx/2,y,z) - hodge_q) * 2.0 / dx;
    case dir::f_0m0:
      if(bc_hodge_global.wallType(x,y-dy/2,z)==NEUMANN) return -bc_hodge_global.wallValue(x,y-dy/2,z);
      else                                              return (hodge_q - bc_hodge_global.wallValue(x,y-dy/2,z)) * 2.0 / dy;
    case dir::f_0p0:
      if(bc_hodge_global.wallType(x,y+dy/2,z)==NEUMANN) return  bc_hodge_global.wallValue(x,y+dy/2,z);
      else                                              return (bc_hodge_global.wallValue(x,y+dy/2,z) - hodge_q) * 2.0 / dy;
    case dir::f_00m:
      if(bc_hodge_global.wallType(x,y,z-dz/2)==NEUMANN) return -bc_hodge_global.wallValue(x,y,z-dz/2);
      else                                              return (hodge_q - bc_hodge_global.wallValue(x,y,z-dz/2)) * 2.0 / dz;
    case dir::f_00p:
      if(bc_hodge_global.wallType(x,y,z+dz/2)==NEUMANN) return  bc_hodge_global.wallValue(x,y,z+dz/2);
      else                                              return (bc_hodge_global.wallValue(x,y,z+dz/2) - hodge_q) * 2.0 / dz;
#else
    case dir::f_m00:
      if(bc_hodge_global.wallType(x-dx/2,y)==NEUMANN) return -bc_hodge_global.wallValue(x-dx/2,y);
      else                                            return (hodge_q - bc_hodge_global.wallValue(x-dx/2,y)) * 2.0 / dx;
    case dir::f_p00:
      if(bc_hodge_global.wallType(x+dx/2,y)==NEUMANN) return  bc_hodge_global.wallValue(x+dx/2,y);
      else                                            return (bc_hodge_global.wallValue(x+dx/2,y) - hodge_q) * 2.0 / dx;
    case dir::f_0m0:
      if(bc_hodge_global.wallType(x,y-dy/2)==NEUMANN) return -bc_hodge_global.wallValue(x,y-dy/2);
      else                                            return (hodge_q - bc_hodge_global.wallValue(x,y-dy/2)) * 2.0 / dy;
    case dir::f_0p0:
      if(bc_hodge_global.wallType(x,y+dy/2)==NEUMANN) return  bc_hodge_global.wallValue(x,y+dy/2);
      else                                            return (bc_hodge_global.wallValue(x,y+dy/2) - hodge_q) * 2.0 / dy;
#endif
    default:
      throw std::invalid_argument("[ERROR]: my_p4est_ns_free_surface_t::dxyz_hodge: unknown local face index.");
    }
  }
  else
  {
    std::vector<p4est_quadrant_t> ngbd;
    ngbd.resize(0);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, face_idx);

    /* multiple neighbor cells should never happen since this function is called for a given face,
     * and the faces are defined only for small cells.
     */
    if(ngbd.size()>1)
    {
      throw std::invalid_argument("[ERROR]: my_p4est_ns_free_surface_t:compute_dxyz_hodge: function called for a subdivided face.");
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
      double phi_q = (*interp_global_phi)(xq, yq, zq);
      double phi_0 = (*interp_global_phi)(x0, y0, z0);
#else
      double phi_q = (*interp_global_phi)(xq, yq);
      double phi_0 = (*interp_global_phi)(x0, y0);
#endif

      double dmin = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
      double dx = convert_to_xyz[face_idx/2] * dmin;

      bool interface_is_crossed = (((phi_q <= 0.0) & (phi_0 > 0.0)) || ((phi_q > 0.0) & (phi_0 <= 0.0)));
      double xyz_int[] = {xq, yq
                         #ifdef P4_TO_P8
                          , zq
                         #endif
                         };
      if(interface_is_crossed)
        xyz_int[face_idx/2] += ((face_idx%2 == 0)?-1.0:+1.0)*((phi_q <= 0.0)?fraction_Interval_Covered_By_Irregular_Domain(phi_q, phi_0, dx, dx):(1.0-fraction_Interval_Covered_By_Irregular_Domain(phi_q, phi_0, dx, dx)))*dx;


      if(interface_is_crossed && (bc_hodge_global.interfaceType(xyz_int)==DIRICHLET))
      {

        if(phi_q>0)
        {
          double phi_tmp = phi_q; phi_q = phi_0; phi_0 = phi_tmp;
          face_idx = face_idx%2==0 ? face_idx+1 : face_idx-1;
          quad_idx = ngbd[0].p.piggy3.local_num;
          switch(face_idx)
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
          theta = 1;
        double val_interface;
        double dist = dx*theta;
        switch(face_idx)
        {
        case dir::f_m00: case dir::f_p00: val_interface = bc_hodge_global.interfaceValue(xyz_int); break;
        case dir::f_0m0: case dir::f_0p0: val_interface = bc_hodge_global.interfaceValue(xyz_int); break;
#ifdef P4_TO_P8
        case dir::f_00m: case dir::f_00p: val_interface = bc_hodge_global.interfaceValue(xyz_int); break;
#endif
        default:
          throw std::invalid_argument("[ERROR]: my_p4est_ns_free_surface_t::compute_dxyz_hodge: uknown local face index.");
        }

        double grad_hodge = hodge_p[quad_idx] - val_interface;

        ierr = VecRestoreArrayRead(hodge, &hodge_p); CHKERRXX(ierr);
        return face_idx%2==0 ? grad_hodge/dist : -grad_hodge/dist;
      }
      else
      {
        double grad_hodge = hodge_p[quad_idx] - hodge_p[ngbd[0].p.piggy3.local_num];

        ierr = VecRestoreArrayRead(hodge, &hodge_p); CHKERRXX(ierr);
        return face_idx%2==0 ? grad_hodge/dx : -grad_hodge/dx;
      }
    }
    /* one neighbor cell that is bigger, get common neighbors */
    else
    {
      p4est_quadrant_t quad_tmp = ngbd[0];
      ngbd.resize(0);
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_tmp.p.piggy3.local_num, quad_tmp.p.piggy3.which_tree, face_idx%2==0 ? face_idx+1 : face_idx-1);

      double dist = 0;
      double grad_hodge = 0;
      double d0 = (double)P4EST_QUADRANT_LEN(quad_tmp.level)/(double)P4EST_ROOT_LEN;

      for(unsigned int m=0; m<ngbd.size(); ++m)
      {
        double dm = (double)P4EST_QUADRANT_LEN(ngbd[m].level)/(double)P4EST_ROOT_LEN;
        dist += pow(dm,P4EST_DIM-1) * .5*(d0+dm);
        grad_hodge += (hodge_p[ngbd[m].p.piggy3.local_num] - hodge_p[quad_tmp.p.piggy3.local_num]) * pow(dm,P4EST_DIM-1);
      }
      dist *= convert_to_xyz[face_idx/2];

      ierr = VecRestoreArrayRead(hodge, &hodge_p); CHKERRXX(ierr);
      return face_idx%2==0 ? grad_hodge/dist : -grad_hodge/dist;
    }
  }
}

// Use global phi and allow for mixed interface type, contrary to parent's function
// use global bc for velocity components!
void my_p4est_ns_free_surface_t::solve_viscosity()
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
  solver.set_phi(global_phi);
  solver.set_mu(mu);
  solver.set_diagonal(alpha * rho/dt_n);
  solver.set_bc(bc_v_global, dxyz_hodge, face_is_well_defined);
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

  my_p4est_level_set_faces_t lsf(ngbd_n, faces_n);
  for(int dir=0; dir<P4EST_DIM; ++dir)
    lsf.extend_Over_Interface(global_phi, vstar[dir], bc_v_global[dir], dir, face_is_well_defined[dir], dxyz_hodge[dir], 2, 2);

  ierr = PetscLogEventEnd(log_my_p4est_navier_stokes_viscosity, 0, 0, 0, 0); CHKERRXX(ierr);
}

/* solve the projection step
 * laplace Hodge = -div(vstar)
 *
 * Differences with parent's method:
 * - Use global_phi with MIXED boundary conditions instead of phi for interface!
 * - velocity field extended after every solve_projection because
 *   i) interpolation from faces to node will disregard interface boundary conditions
 *  ii) a regular extension of vnp1 across the fress surface will be required to capture the correct boundary condition on the free surface
 */
void my_p4est_ns_free_surface_t::solve_projection()
{
  PetscErrorCode ierr;

  ierr = PetscLogEventBegin(log_my_p4est_ns_free_surface_projection, 0, 0, 0, 0); CHKERRXX(ierr);

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
  solver.set_phi(global_phi);
  solver.set_mu(1);
  solver.set_bc(bc_hodge_global);
  solver.set_rhs(rhs);
  solver.set_nullspace_use_fixed_point(false);
  solver.solve(hodge, false, KSPCG, PCHYPRE);

  ierr = VecDestroy(rhs); CHKERRXX(ierr);

  /* if needed, shift the hodge variable to a zero average */
  if(solver.get_matrix_has_nullspace())
  {
    my_p4est_level_set_cells_t lsc(ngbd_c, ngbd_n);
    double average = lsc.integrate(global_phi, hodge) / area_in_negative_domain(p4est_n, nodes_n, global_phi);
    double *hodge_p;
    ierr = VecGetArray(hodge, &hodge_p); CHKERRXX(ierr);
    for(p4est_locidx_t quad_idx=0; quad_idx<p4est_n->local_num_quadrants; ++quad_idx)
      hodge_p[quad_idx] -= average;
    ierr = VecRestoreArray(hodge, &hodge_p); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(hodge, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (hodge, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  my_p4est_level_set_cells_t lsc(ngbd_c, ngbd_n);
  lsc.extend_Over_Interface(global_phi, hodge, &bc_hodge_global, 2, 2);

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
      vnp1_p[f_idx] = vstar_p[f_idx] - dxyz_hodge_p[f_idx];

    ierr = VecGhostUpdateEnd  (dxyz_hodge[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecRestoreArray(vnp1[dir], &vnp1_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(vstar[dir], &vstar_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(dxyz_hodge[dir], &dxyz_hodge_p); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(vnp1[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (vnp1[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    // extend the velocity field over the interfaces
    // (will be required for proper interpolation to nodes AND for correction of free surface boundary condition, i.e. Mingru's task)
    my_p4est_level_set_faces_t lsf(ngbd_n, faces_n);
    lsf.extend_Over_Interface(global_phi, vnp1[dir], bc_v_global[dir], dir, face_is_well_defined[dir], NULL, 2, 8);
    // Do not set the second-to-last argument to dxyz_hodge!
    // (dxyz_hodge has already been subtracted from vstar, here above)
  }

  ierr = PetscLogEventEnd(log_my_p4est_ns_free_surface_projection, 0, 0, 0, 0); CHKERRXX(ierr);
}

/*
 * Differences with parent's method:
 * - Disregard face_is_well_defined vectors in the interpolation, so that lsqr interpolation is used everywhere based on the extensions calculated before-hand!
 * */
void my_p4est_ns_free_surface_t::compute_velocity_at_nodes()
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
                                       vnp1[dir], dir, NULL, 2, bc_v_global);
    }

    ierr = VecGhostUpdateBegin(vnp1_nodes[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    for(size_t i=0; i<ngbd_n->get_local_size(); ++i)
    {
      p4est_locidx_t n = ngbd_n->get_local_node(i);
      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, p4est_n, nodes_n, xyz);
      v_p[n] = interpolate_f_at_node_n(p4est_n, ghost_n, nodes_n, faces_n, ngbd_c, ngbd_n, n,
                                       vnp1[dir], dir, NULL, 2, bc_v_global);
    }

    ierr = VecGhostUpdateEnd(vnp1_nodes[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecRestoreArray(vnp1_nodes[dir], &v_p); CHKERRXX(ierr);

    my_p4est_level_set_t lsn(ngbd_n);
    lsn.extend_Over_Interface(phi, vnp1_nodes[dir], bc_v_global[dir]);
  }

  compute_vorticity();
  compute_max_L2_norm_u();
}


#ifdef P4_TO_P8
void my_p4est_ns_free_surface_t::update_from_tn_to_tnp1(const CF_3 *level_set, bool second_order_fs_advection)
#else
void my_p4est_ns_free_surface_t::update_from_tn_to_tnp1(const CF_2 *level_set, bool second_order_fs_advection)
#endif
{
  PetscErrorCode ierr;

  ierr = PetscLogEventBegin(log_my_p4est_ns_free_surface_update, 0, 0, 0, 0); CHKERRXX(ierr);

  if(!dt_updated)
    compute_dt(MAX(max_L2_norm_u, 1.0));

  dt_updated = false;
  // so, at this stage, we have already swapped
  // dt_nm1 <-- dt_n
  // dt_n   <-- new value, just calculated previously
  // HOWEVER, the relevant fields have not been swapped yet! they will be at the end of this function
  // so, currently, what is in memory under a "_n" name represents nm1 fields for the updated time
  // and what is under a "_np1" name represents n fields for the updated time

  splitting_criteria_t *data = (splitting_criteria_t*)p4est_n->user_pointer;
  // new refinement criterion to be attached to the new forest
  splitting_criteria_vorticity_t criteria(data->min_lvl, data->max_lvl, data->lip, uniform_band, threshold_split_cell, max_L2_norm_u, smoke_thresh);

  // build the second derivatives required for advection of the free-surface levelset (semi-lagrangian advection uses (non-oscillatory) quadratic interpolation and thus requires the following second derivatives)
  Vec fs_phi_xx[P4EST_DIM], *vxx_np1_nodes[P4EST_DIM], *vxx_n_nodes[P4EST_DIM];
  for (short dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &fs_phi_xx[dir]); CHKERRXX(ierr);
    vxx_np1_nodes[dir] = new Vec[P4EST_DIM];
    if(second_order_fs_advection)
      vxx_n_nodes[dir] = new Vec[P4EST_DIM];
    for (short der = 0; der < P4EST_DIM; ++der) {
      ierr = VecCreateGhostNodes(p4est_n, nodes_n, &vxx_np1_nodes[dir][der]); CHKERRXX(ierr);
      if(second_order_fs_advection)
      {
        // this one is required only for second-order-in-time advection
        ierr = VecCreateGhostNodes(p4est_n, nodes_n, &vxx_n_nodes[dir][der]); CHKERRXX(ierr);
      }
    }
  }
  // calculate them now...
  calculate_second_derivatives_for_fs_advection(fs_phi_xx, vxx_np1_nodes, vxx_n_nodes, second_order_fs_advection);

  // GRID UPDATE STARTS HERE
  // advect what needs to be advected, construct the new forest etc. (iterative procedure)
  p4est_t *p4est_np1 = p4est_copy(p4est_n, P4EST_FALSE);

  p4est_ghost_t *ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);
  my_p4est_hierarchy_t *hierarchy_np1 = new my_p4est_hierarchy_t(p4est_np1, ghost_np1, brick);
  my_p4est_node_neighbors_t *ngbd_np1 = new my_p4est_node_neighbors_t(hierarchy_np1, nodes_np1);
  p4est_np1->user_pointer = (void*)&criteria;

  Vec phi_np1; // solid interface at t_{n+1}
  ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);
  Vec fs_phi_np1; // free surface interface at t_{n+1}
  ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &fs_phi_np1); CHKERRXX(ierr);
  Vec global_phi_np1; // global levelset at t_{n+1}
  ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &global_phi_np1); CHKERRXX(ierr);
  Vec smoke_np1 = NULL; // smoke at t_{n+1}, might not be required if no smoke, that's why NULL

  // interpolator agent: interpolator from nodes at time t_n
  my_p4est_interpolation_nodes_t interp_nodes(ngbd_n);

  Vec vorticity_np1;
  ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &vorticity_np1); CHKERRXX(ierr);

  for(size_t n=0; n<nodes_np1->indep_nodes.elem_count; ++n)
  {
    double xyz[P4EST_DIM];
    node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);
    interp_nodes.add_point(n, xyz);
  }
  // the vorticity as owned by the current object has been calculated right after the end of the compute_velocity_at_nodes() function
  // which is supposed to be called after the full solver: so it is actually vorticity_{n+1}, a simple interpolation is ok
  interp_nodes.set_input(vorticity, linear);
  interp_nodes.interpolate(vorticity_np1);

  if(level_set==NULL) // no interface was provided, interpolate what it was previously (-1 everywhere if no interface at all)
  {
    interp_nodes.set_input(phi, linear);
    interp_nodes.interpolate(phi_np1);
  }
  else
    sample_cf_on_nodes(p4est_np1, nodes_np1, *level_set, phi_np1);

  // advect the free surface:
  // As seen from the current object status, we want to know fs_phi_{n+2},
  // advect from fs_phi (which is at t_{n+1}) using vnp1 and possible vn if second order is required
  // both these fields are defined in ngbd_n!

  my_p4est_semi_lagrangian_t sl(&p4est_np1, &nodes_np1, &ghost_np1, ngbd_n, ((second_order_fs_advection)? ngbd_n : NULL));
  double *fs_phi_np1_p;
  ierr = VecGetArray(fs_phi_np1, &fs_phi_np1_p); CHKERRXX(ierr);
  if(second_order_fs_advection)
    sl.advect_from_n_to_np1(dt_nm1, dt_n, vn_nodes, vxx_n_nodes, vnp1_nodes, vxx_np1_nodes, fs_phi, fs_phi_xx, fs_phi_np1_p);
  else
    sl.advect_from_n_to_np1(dt_n, vnp1_nodes, vxx_np1_nodes, fs_phi, fs_phi_xx, fs_phi_np1_p);
  ierr = VecRestoreArray(fs_phi_np1, &fs_phi_np1_p); CHKERRXX(ierr);

  // build the global phi from the known solid boundary and the advected free surface!
  build_global_phi_and_face_is_well_defined(nodes_np1, phi_np1, fs_phi_np1, global_phi_np1);

  if(smoke!=NULL && refine_with_smoke)
  {
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &smoke_np1); CHKERRXX(ierr);

    Vec vtmp[P4EST_DIM];
    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &vtmp[dir]); CHKERRXX(ierr);
      interp_nodes.set_input(vnp1_nodes[dir], linear);
      interp_nodes.interpolate(vtmp[dir]);
    }

    advect_smoke(ngbd_np1, vtmp, smoke_np1);

    for(int dir=0; dir<P4EST_DIM; ++dir)
    {
      ierr = VecDestroy(vtmp[dir]); CHKERRXX(ierr);
    }
  }

  bool grid_is_changing = criteria.refine_and_coarsen(p4est_np1, nodes_np1, global_phi_np1, vorticity_np1, smoke_np1); // refine with the GLOBAL phi!!!!!
  int iter=0;
  while(grid_is_changing)
  {
    my_p4est_partition(p4est_np1, P4EST_FALSE, NULL);
    p4est_ghost_destroy(ghost_np1); ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
    p4est_nodes_destroy(nodes_np1); nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);
    delete hierarchy_np1; hierarchy_np1 = new my_p4est_hierarchy_t(p4est_np1, ghost_np1, brick);
    delete ngbd_np1; ngbd_np1 = new my_p4est_node_neighbors_t(hierarchy_np1, nodes_np1);

    ierr = VecDestroy(phi_np1); CHKERRXX(ierr);
    ierr = VecDestroy(fs_phi_np1); CHKERRXX(ierr);
    ierr = VecDestroy(global_phi_np1); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &fs_phi_np1); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &global_phi_np1); CHKERRXX(ierr);


    ierr = VecDestroy(vorticity_np1); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1,  &vorticity_np1); CHKERRXX(ierr);

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
      sample_cf_on_nodes(p4est_np1, nodes_np1, *level_set, phi_np1);

    my_p4est_semi_lagrangian_t updated_sl(&p4est_np1, &nodes_np1, &ghost_np1, ngbd_n, ((second_order_fs_advection)? ngbd_n : NULL));
    ierr = VecGetArray(fs_phi_np1, &fs_phi_np1_p); CHKERRXX(ierr);
    if(second_order_fs_advection)
      updated_sl.advect_from_n_to_np1(dt_nm1, dt_n, vn_nodes, vxx_n_nodes, vnp1_nodes, vxx_np1_nodes, fs_phi, fs_phi_xx, fs_phi_np1_p);
    else
      updated_sl.advect_from_n_to_np1(dt_n, vnp1_nodes, vxx_np1_nodes, fs_phi, fs_phi_xx, fs_phi_np1_p);
    ierr = VecRestoreArray(fs_phi_np1, &fs_phi_np1_p); CHKERRXX(ierr);

    build_global_phi_and_face_is_well_defined(nodes_np1, phi_np1, fs_phi_np1, global_phi_np1);

    if(smoke!=NULL && refine_with_smoke)
    {
      ierr = VecDestroy(smoke_np1); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &smoke_np1); CHKERRXX(ierr);

      Vec vtmp[P4EST_DIM];
      for(int dir=0; dir<P4EST_DIM; ++dir)
      {
        ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &vtmp[dir]); CHKERRXX(ierr);
        interp_nodes.set_input(vnp1_nodes[dir], linear);
        interp_nodes.interpolate(vtmp[dir]);
      }

      advect_smoke(ngbd_np1, vtmp, smoke_np1);

      for(int dir=0; dir<P4EST_DIM; ++dir)
      {
        ierr = VecDestroy(vtmp[dir]); CHKERRXX(ierr);
      }
    }

    grid_is_changing = criteria.refine_and_coarsen(p4est_np1, nodes_np1, global_phi_np1, vorticity_np1, smoke_np1); // refine with the GLOBAL phi!!!!!

    iter++;

    if(iter>1+data->max_lvl-data->min_lvl)
    {
      ierr = PetscPrintf(p4est_n->mpicomm, "ooops ... the grid update did not converge\n"); CHKERRXX(ierr);
      break;
    }
  }
  ierr = VecDestroy(vorticity_np1);

  // GRID UPDATE ENDS HERE

  // no further refinement required so re-assigned the regular data pointer to use_pointer
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
  ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &vorticity); CHKERRXX(ierr);

  // needs to change the layout of the following since the forest has been balanced and repartitioned!
  // could use a rebalance-vector function instead, would be more efficient
  ierr = VecDestroy(phi_np1); CHKERRXX(ierr);
  ierr = VecDestroy(fs_phi_np1); CHKERRXX(ierr);
  ierr = VecDestroy(global_phi_np1); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &fs_phi_np1); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &global_phi_np1); CHKERRXX(ierr);
  if(level_set==NULL)
  {
    interp_nodes.set_input(phi, linear);
    interp_nodes.interpolate(phi_np1);
  }
  else
    sample_cf_on_nodes(p4est_np1, nodes_np1, *level_set, phi_np1);
  my_p4est_semi_lagrangian_t updated_sl(&p4est_np1, &nodes_np1, &ghost_np1, ngbd_n, ((second_order_fs_advection)? ngbd_n : NULL));
  ierr = VecGetArray(fs_phi_np1, &fs_phi_np1_p); CHKERRXX(ierr);
  if(second_order_fs_advection)
    updated_sl.advect_from_n_to_np1(dt_nm1, dt_n, vn_nodes, vxx_n_nodes, vnp1_nodes, vxx_np1_nodes, fs_phi, fs_phi_xx, fs_phi_np1_p);
  else
    updated_sl.advect_from_n_to_np1(dt_n, vnp1_nodes, vxx_np1_nodes, fs_phi, fs_phi_xx, fs_phi_np1_p);
  ierr = VecRestoreArray(fs_phi_np1, &fs_phi_np1_p); CHKERRXX(ierr);
  build_global_phi_and_face_is_well_defined(nodes_np1, phi_np1, fs_phi_np1, global_phi_np1);
  // finally reinitialize all needed levelsets
  my_p4est_level_set_t lsn(ngbd_np1);
  lsn.reinitialize_2nd_order(global_phi_np1);
  lsn.perturb_level_set_function(global_phi_np1, EPS);
  lsn.reinitialize_2nd_order(phi_np1);
  lsn.perturb_level_set_function(phi_np1, EPS);
  lsn.reinitialize_2nd_order(fs_phi_np1);
  lsn.perturb_level_set_function(fs_phi_np1, EPS);

  /* interpolate the quantities on the new forest at the nodes */
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecDestroy(vnm1_nodes[dir]); CHKERRXX(ierr);
    vnm1_nodes[dir] = vn_nodes[dir];

    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &vn_nodes[dir]); CHKERRXX(ierr);
#ifdef P4_TO_P8
    interp_nodes.set_input(vnp1_nodes[dir], vxx_np1_nodes[dir][0], vxx_np1_nodes[dir][1], vxx_np1_nodes[dir][2], quadratic);
#else
    interp_nodes.set_input(vnp1_nodes[dir], vxx_np1_nodes[dir][0], vxx_np1_nodes[dir][1], quadratic);
#endif
    interp_nodes.interpolate(vn_nodes[dir]); CHKERRXX(ierr);

    ierr = VecDestroy(vnp1_nodes[dir]); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &vnp1_nodes[dir]); CHKERRXX(ierr);
  }

  interp_nodes.clear();


  /* set velocity inside solid to bc_v */
//  extrapolate_bc_v(ngbd_np1, vn_nodes, phi_np1);

  /* advect smoke */
  if(smoke!=NULL)
  {
    if(smoke_np1!=NULL)
    {
      ierr = VecDestroy(smoke_np1); CHKERRXX(ierr);
    }
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &smoke_np1); CHKERRXX(ierr);
    advect_smoke(ngbd_np1, vn_nodes, smoke_np1);
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
  interp_cell.set_input(hodge, global_phi_np1, &bc_hodge_global);
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
    interp_faces.set_input(dxyz_hodge[dir], dir, 1, NULL /*face_is_well_defined[dir]*/); // [RAPHAEL] : deactivated the face_is_well_defined thing to avoid off-centered stencil when interpolating for faces close to the free surface
    interp_faces.interpolate(dxyz_hodge_tmp);
    interp_faces.clear();

    ierr = VecDestroy(dxyz_hodge[dir]); CHKERRXX(ierr);
    dxyz_hodge[dir] = dxyz_hodge_tmp;

    ierr = VecGhostUpdateBegin(dxyz_hodge[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (dxyz_hodge[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

//    ierr = VecDestroy(face_is_well_defined[dir]); CHKERRXX(ierr);
//    ierr = VecCreateGhostFaces(p4est_np1, faces_np1, &face_is_well_defined[dir], dir); CHKERRXX(ierr);
//    check_if_faces_are_well_defined_for_free_surface(faces_np1, dir, face_is_well_defined[dir]);
////    check_if_faces_are_well_defined(p4est_np1, ngbd_np1, faces_np1, dir, fs_phi_np1, DIRICHLET, face_is_well_defined[dir]);
//    // we'll need the face_is_well_defined defined as such for extrapolation purposes!
//    // "DIRICHLET" has no mathematicl/physical interpretation in the case above, it's just the correct input for the expected outcome

    ierr = VecDestroy(vstar[dir]); CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est_np1, faces_np1, &vstar[dir], dir); CHKERRXX(ierr);

    ierr = VecDestroy(vnp1[dir]); CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est_np1, faces_np1, &vnp1[dir], dir); CHKERRXX(ierr);

    // destroy the second derivatives that were built for the advection of the free-surface levelset
    ierr = VecDestroy(fs_phi_xx[dir]); CHKERRXX(ierr);
    for (short der = 0; der < P4EST_DIM; ++der) {
      ierr = VecDestroy(vxx_np1_nodes[dir][der]); CHKERRXX(ierr);
      if(second_order_fs_advection)
      {
        // this one is required only for second-order-in-time advection
        ierr = VecDestroy(vxx_n_nodes[dir][der]); CHKERRXX(ierr);
      }
    }
    delete[] vxx_np1_nodes[dir];
    if(second_order_fs_advection)
      delete[] vxx_n_nodes[dir];
  }

  /* update the variables */
  p4est_destroy(p4est_nm1); p4est_nm1 = p4est_n; p4est_n = p4est_np1;
  p4est_ghost_destroy(ghost_nm1); ghost_nm1 = ghost_n; ghost_n = ghost_np1;
  p4est_nodes_destroy(nodes_nm1); nodes_nm1 = nodes_n; nodes_n = nodes_np1;
  delete hierarchy_nm1; hierarchy_nm1 = hierarchy_n; hierarchy_n = hierarchy_np1;
  delete ngbd_nm1; ngbd_nm1 = ngbd_n; ngbd_n = ngbd_np1;
  delete ngbd_c; ngbd_c = ngbd_c_np1;
  delete faces_n; faces_n = faces_np1;

  ierr = VecDestroy(phi); CHKERRXX(ierr); phi = phi_np1;
  ierr = VecDestroy(fs_phi); CHKERRXX(ierr); fs_phi = fs_phi_np1;
  ierr = VecDestroy(global_phi); CHKERRXX(ierr); global_phi = global_phi_np1;

  if(smoke!=NULL)
  {
    ierr = VecDestroy(smoke); CHKERRXX(ierr);
    smoke = smoke_np1;
  }

  delete interp_phi;
  interp_phi = new my_p4est_interpolation_nodes_t(ngbd_n);
  interp_phi->set_input(phi, linear);
  delete interp_fs_phi;
  interp_fs_phi = new my_p4est_interpolation_nodes_t(ngbd_n);
  interp_fs_phi->set_input(fs_phi, linear);
  delete interp_global_phi;
  interp_global_phi = new my_p4est_interpolation_nodes_t(ngbd_n);
  interp_global_phi->set_input(global_phi_np1, linear);

  // needs to be done AFTER the update of interpolators since interp_global_phi is used in check_if_faces_are_well_defined_for_free_surface
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecDestroy(face_is_well_defined[dir]); CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est_np1, faces_np1, &face_is_well_defined[dir], dir); CHKERRXX(ierr);
    check_if_faces_are_well_defined_for_free_surface(faces_np1, dir, bc_v_global[dir], face_is_well_defined[dir]);
  }


  ierr = PetscLogEventEnd(log_my_p4est_ns_free_surface_update, 0, 0, 0, 0); CHKERRXX(ierr);

}


void my_p4est_ns_free_surface_t::calculate_second_derivatives_for_fs_advection(Vec *fs_phi_xx, Vec **vxx_np1_nodes, Vec **vxx_n_nodes, bool second_order_fs_advection)
{
  PetscErrorCode ierr;
  const double *fs_phi_read_p, *vnp1_nodes_read_p[P4EST_DIM], *vn_nodes_read_p[P4EST_DIM];
  double *fs_phi_xx_p[P4EST_DIM], *vxx_np1_nodes_p[P4EST_DIM][P4EST_DIM], *vxx_n_nodes_p[P4EST_DIM][P4EST_DIM];

  // get all pointers
  ierr = VecGetArrayRead(fs_phi, &fs_phi_read_p); CHKERRXX(ierr);
  for (short dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecGetArrayRead(vnp1_nodes[dir], &vnp1_nodes_read_p[dir]); CHKERRXX(ierr);
    ierr = VecGetArray(fs_phi_xx[dir], &fs_phi_xx_p[dir]); CHKERRXX(ierr);
    if(second_order_fs_advection)
    {
      ierr = VecGetArrayRead(vn_nodes[dir], &vn_nodes_read_p[dir]); CHKERRXX(ierr);
    }

    for (short der = 0; der < P4EST_DIM; ++der) {
      ierr = VecGetArray(vxx_np1_nodes[dir][der], &vxx_np1_nodes_p[dir][der]); CHKERRXX(ierr);
      if(second_order_fs_advection)
      {
        ierr = VecGetArray(vxx_n_nodes[dir][der], &vxx_n_nodes_p[dir][der]); CHKERRXX(ierr);
      }
    }
  }

  quad_neighbor_nodes_of_node_t qnnn;
  // loop over the layer nodes
  for (size_t n = 0; n < ngbd_n->get_layer_size(); ++n) {
    p4est_locidx_t node_idx = ngbd_n->get_layer_node(n);
    ngbd_n->get_neighbors(node_idx, qnnn);
    fs_phi_xx_p[0][node_idx] = qnnn.dxx_central(fs_phi_read_p);
    fs_phi_xx_p[1][node_idx] = qnnn.dyy_central(fs_phi_read_p);
#ifdef P4_TO_P8
    fs_phi_xx_p[2][node_idx] = qnnn.dzz_central(fs_phi_read_p);
#endif
    for (int dir = 0; dir < P4EST_DIM; ++dir) {
      vxx_np1_nodes_p[dir][0][node_idx] = qnnn.dxx_central(vnp1_nodes_read_p[dir]);
      vxx_np1_nodes_p[dir][1][node_idx] = qnnn.dyy_central(vnp1_nodes_read_p[dir]);
#ifdef P4_TO_P8
      vxx_np1_nodes_p[dir][2][node_idx] = qnnn.dzz_central(vnp1_nodes_read_p[dir]);
#endif
      if(second_order_fs_advection)
      {
        vxx_n_nodes_p[dir][0][node_idx] = qnnn.dxx_central(vn_nodes_read_p[dir]);
        vxx_n_nodes_p[dir][1][node_idx] = qnnn.dyy_central(vn_nodes_read_p[dir]);
#ifdef P4_TO_P8
        vxx_n_nodes_p[dir][2][node_idx] = qnnn.dzz_central(vn_nodes_read_p[dir]);
#endif
      }
    }
  }
  // Begin updating the ghost layers
  for (short dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecGhostUpdateBegin(fs_phi_xx[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    for (short der = 0; der < P4EST_DIM; ++der) {
      ierr = VecGhostUpdateBegin(vxx_np1_nodes[dir][der], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      if(second_order_fs_advection)
      {
        ierr = VecGhostUpdateBegin(vxx_n_nodes[dir][der], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      }
    }
  }
  // loop over the local nodes
  for (size_t n = 0; n < ngbd_n->get_local_size(); ++n) {
    p4est_locidx_t node_idx = ngbd_n->get_local_node(n);
    ngbd_n->get_neighbors(node_idx, qnnn);
    fs_phi_xx_p[0][node_idx] = qnnn.dxx_central(fs_phi_read_p);
    fs_phi_xx_p[1][node_idx] = qnnn.dyy_central(fs_phi_read_p);
#ifdef P4_TO_P8
    fs_phi_xx_p[2][node_idx] = qnnn.dzz_central(fs_phi_read_p);
#endif
    for (int dir = 0; dir < P4EST_DIM; ++dir) {
      vxx_np1_nodes_p[dir][0][node_idx] = qnnn.dxx_central(vnp1_nodes_read_p[dir]);
      vxx_np1_nodes_p[dir][1][node_idx] = qnnn.dyy_central(vnp1_nodes_read_p[dir]);
#ifdef P4_TO_P8
      vxx_np1_nodes_p[dir][2][node_idx] = qnnn.dzz_central(vnp1_nodes_read_p[dir]);
#endif
      if(second_order_fs_advection)
      {
        vxx_n_nodes_p[dir][0][node_idx] = qnnn.dxx_central(vn_nodes_read_p[dir]);
        vxx_n_nodes_p[dir][1][node_idx] = qnnn.dyy_central(vn_nodes_read_p[dir]);
#ifdef P4_TO_P8
        vxx_n_nodes_p[dir][2][node_idx] = qnnn.dzz_central(vn_nodes_read_p[dir]);
#endif
      }
    }
  }
  // End updating the ghost layers
  for (short dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecGhostUpdateEnd(fs_phi_xx[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    for (short der = 0; der < P4EST_DIM; ++der) {
      ierr = VecGhostUpdateEnd(vxx_np1_nodes[dir][der], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      if(second_order_fs_advection)
      {
        ierr = VecGhostUpdateEnd(vxx_n_nodes[dir][der], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      }
    }
  }

  // restore pointers
  ierr = VecRestoreArrayRead(fs_phi, &fs_phi_read_p); CHKERRXX(ierr);
  for (short dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecRestoreArrayRead(vnp1_nodes[dir], &vnp1_nodes_read_p[dir]); CHKERRXX(ierr);
    ierr = VecRestoreArray(fs_phi_xx[dir], &fs_phi_xx_p[dir]); CHKERRXX(ierr);
    if(second_order_fs_advection)
    {
      ierr = VecRestoreArrayRead(vn_nodes[dir], &vn_nodes_read_p[dir]); CHKERRXX(ierr);
    }

    for (short der = 0; der < P4EST_DIM; ++der) {
      ierr = VecRestoreArray(vxx_np1_nodes[dir][der], &vxx_np1_nodes_p[dir][der]); CHKERRXX(ierr);
      if(second_order_fs_advection)
      {
        ierr = VecRestoreArray(vxx_n_nodes[dir][der], &vxx_n_nodes_p[dir][der]); CHKERRXX(ierr);
      }
    }
  }

}

// different with parent's allow for mixed interface type when extending over interface + use global phi
void my_p4est_ns_free_surface_t::compute_pressure()
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


  my_p4est_level_set_cells_t lsc(ngbd_c, ngbd_n);
  lsc.extend_Over_Interface(global_phi, pressure, &bc_pressure_global, 2, 3);
}


void my_p4est_ns_free_surface_t::save_vtk(const char* name)
{
  PetscErrorCode ierr;

  const double *phi_p;
  const double *fs_phi_p;
  const double *vn_p[P4EST_DIM];
//  const double *hodge_p;
//  const double *pressure_p;

  ierr = VecGetArrayRead(phi    , &phi_p    ); CHKERRXX(ierr);
  ierr = VecGetArrayRead(fs_phi , &fs_phi_p ); CHKERRXX(ierr);
//  ierr = VecGetArrayRead(hodge  , &hodge_p  ); CHKERRXX(ierr);
//  ierr = VecGetArrayRead(pressure  , &pressure_p  ); CHKERRXX(ierr);

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
  interp_c.set_input(pressure, phi, &bc_pressure_global);
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
                           4+P4EST_DIM, /* number of VTK_POINT_DATA */
                           1, /* number of VTK_CELL_DATA  */
                           name,
                           VTK_POINT_DATA, "solid", phi_p,
                           VTK_POINT_DATA, "free surface", fs_phi_p,
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
                           3+P4EST_DIM, /* number of VTK_POINT_DATA */
                           1, /* number of VTK_CELL_DATA  */
                           name,
                           VTK_POINT_DATA, "solid", phi_p,
                           VTK_POINT_DATA, "free surface", phi_p,
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

  ierr = VecRestoreArrayRead(phi    , &phi_p    ); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(fs_phi , &fs_phi_p ); CHKERRXX(ierr);
//  ierr = VecRestoreArrayRead(hodge  , &hodge_p  ); CHKERRXX(ierr);
//  ierr = VecRestoreArrayRead(pressure  , &pressure_p  ); CHKERRXX(ierr);

  ierr = VecRestoreArrayRead(vorticity, &vort_p); CHKERRXX(ierr);

  ierr = VecRestoreArrayRead(pressure_nodes, &pressure_nodes_p); CHKERRXX(ierr);
  ierr = VecDestroy(pressure_nodes); CHKERRXX(ierr);

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArrayRead(vn_nodes[dir], &vn_p[dir]); CHKERRXX(ierr);
  }

  ierr = PetscPrintf(p4est_n->mpicomm, "Saved visual data in ... %s\n", name); CHKERRXX(ierr);
}

// modification of the check_if_faces_are_well_defined function from class my_p4est_faces to allow for mixed interface types
void my_p4est_ns_free_surface_t::check_if_faces_are_well_defined_for_free_surface(my_p4est_faces_t *faces, int dir,
                                                                                  #ifdef P4_TO_P8
                                                                                  BoundaryConditions3D bc_v,
                                                                                  #else
                                                                                  BoundaryConditions2D bc_v,
                                                                                  #endif
                                                                                  Vec face_is_well_defined)
{
  PetscErrorCode ierr;

  PetscScalar *face_is_well_defined_p;
  ierr = VecGetArray(face_is_well_defined, &face_is_well_defined_p); CHKERRXX(ierr);

  for(p4est_locidx_t f_idx=0; f_idx<faces->num_local[dir]; ++f_idx)
  {
    double xyz_face[] = {
      faces->x_fr_f(f_idx,dir)
      , faces->y_fr_f(f_idx,dir)
  #ifdef P4_TO_P8
      , faces->z_fr_f(f_idx,dir)
  #endif
    };
    switch (bc_v.interfaceType(xyz_face)) {
    case DIRICHLET:
      face_is_well_defined_p[f_idx] = (*interp_global_phi)(xyz_face) <= 0.0;
      break;
    case NEUMANN:
    {
      double xyz_corner[P4EST_DIM];
      bool volume_is_crossed = false;
      for (short ddx = -1; ddx < 2; ddx+=2) {
        xyz_corner[0] = xyz_face[0] + ((double) ddx)*0.5*dxyz_min[0];
        for (short ddy = -1; ddy < 2; ddy+=2) {
          xyz_corner[1] = xyz_face[1] + ((double) ddy)*0.5*dxyz_min[1];
#ifdef P4_TO_P8
          for (short ddz = -1; ddz < 2; ddz+=2) {
            xyz_corner[2] = xyz_face[2] + ((double) ddz)*0.5*dxyz_min[2];
#endif
            volume_is_crossed = volume_is_crossed || ((*interp_global_phi)(xyz_corner) <= 0.0);
#ifdef P4_TO_P8
          }
#endif
        }
      }
      face_is_well_defined_p[f_idx] = (PetscScalar) volume_is_crossed;
      break;
    }
    default:
      throw std::runtime_error("my_p4est_ns_free_surface_t::check_if_faces_are_well_defined_for_free_surface: unknown boundary condition type for interface...");
      break;
    }
  }

  ierr = VecRestoreArray(face_is_well_defined, &face_is_well_defined_p); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(face_is_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (face_is_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
}
