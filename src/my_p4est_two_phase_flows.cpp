#ifdef P4_TO_P8
#include "my_p8est_two_phase_flows.h"
#include "my_p8est_trajectory_of_point.h"
#else
#include "my_p4est_two_phase_flows.h"
#include "my_p4est_trajectory_of_point.h"
#endif

#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_my_p4est_two_phase_flows_viscosity;
extern PetscLogEvent log_my_p4est_two_phase_flows_projection;
extern PetscLogEvent log_my_p4est_two_phase_compute_voronoi_cell;
#endif

my_p4est_two_phase_flows_t::my_p4est_two_phase_flows_t(my_p4est_node_neighbors_t *ngbd_nm1, my_p4est_node_neighbors_t *ngbd_n, my_p4est_faces_t *faces, my_p4est_node_neighbors_t *fine_ngbd_n)
  : brick(ngbd_n->myb), conn(ngbd_n->p4est->connectivity),
    p4est_nm1(ngbd_nm1->p4est), ghost_nm1(ngbd_nm1->ghost), nodes_nm1(ngbd_nm1->nodes), hierarchy_nm1(ngbd_nm1->hierarchy), ngbd_nm1(ngbd_nm1), 
    p4est_n(ngbd_n->p4est), fine_p4est_n(fine_ngbd_n->p4est),
    ghost_n(ngbd_n->ghost), fine_ghost_n(fine_ngbd_n->ghost),
    nodes_n(ngbd_n->nodes), fine_nodes_n(fine_ngbd_n->nodes),
    hierarchy_n(ngbd_n->hierarchy), fine_hierarchy_n(fine_ngbd_n->hierarchy),
    ngbd_n(ngbd_n), fine_ngbd_n(fine_ngbd_n),
    ngbd_c(faces->get_ngbd_c()), faces_n(faces),
    wall_bc_value_hodge(this)
{
  PetscErrorCode ierr;

  mu_minus = 1.0; mu_plus = 1.0;
  rho_minus = 1.0; rho_plus = 1.0;
  uniform_band_minus = 0.0; uniform_band_plus = 0.0;
  threshold_split_cell = 0.04;
  cfl = 1.0;
  dt_updated = false;
  max_L2_norm_u = 0;

  sl_order = 2;

  double p = 1.0;
  while (DBL_MAX-p == DBL_MAX) p*=2.0;
  threshold_dbl_max = DBL_MAX -p;

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

  bc_v = NULL;
  bc_pressure = NULL;
  for(unsigned char dir=0; dir<P4EST_DIM; ++dir)
    external_forces[dir]  = NULL;

  // -------------------------------------------------------------------
  // ----- FIELDS SAMPLED AT NODES OF THE INTERFACE-CAPTURING GRID -----
  // -------------------------------------------------------------------
  // scalar fields
  fine_phi                    = NULL;
  fine_curvature              = NULL;
  jump_hodge                  = NULL;
  jump_normal_flux_hodge      = NULL;
  mass_flux                   = NULL;
  nonconstant_surface_tension = NULL;
  // vector fields
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    normal[dir] = NULL;
    fine_phi_xxyyzz[dir] = NULL;
    grad_surface_tension[dir] = NULL;
    // tensor/matrix fields
    for (unsigned char der = 0; der < P4EST_DIM; ++der) {
      jump_mu_grad_v[dir][der] = NULL;
    }
  }
  // -----------------------------------------------------------------------
  // ----- FIELDS SAMPLED AT NODES OF THE COMPUTATIONAL GRID AT TIME N -----
  // -----------------------------------------------------------------------
  // scalar fields
  vorticity = NULL;
  // vector fields
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    vnp1_nodes_omega_minus[dir] = NULL;
    vnp1_nodes_omega_plus[dir]  = NULL;
    vn_nodes_omega_minus[dir]   = NULL;
    vn_nodes_omega_plus[dir]    = NULL;
    // tensor/matrix fields
    for (unsigned char der = 0; der < P4EST_DIM; ++der) {
      vn_nodes_omega_minus_xxyyzz[dir][der] = NULL;
      vn_nodes_omega_plus_xxyyzz[dir][der]  = NULL;
    }
  }
  // ------------------------------------------------------------------------------
  // ----- FIELDS SAMPLED AT CELL CENTERS OF THE COMPUTATIONAL GRID AT TIME N -----
  // ------------------------------------------------------------------------------
  // scalar fields
  Vec vec_loc;
  // cell-sampled fields, computational grid n
  ierr = VecCreateGhostCells(p4est_n, ghost_n, &hodge); CHKERRXX(ierr);
  // hodge
  ierr = VecGhostGetLocalForm(hodge, &vec_loc); CHKERRXX(ierr);
  ierr = VecSet(vec_loc, 0); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(hodge, &vec_loc); CHKERRXX(ierr);
  ierr = VecDuplicate(hodge, &pressure); CHKERRXX(ierr);
  // ------------------------------------------------------------------------------
  // ----- FIELDS SAMPLED AT FACE CENTERS OF THE COMPUTATIONAL GRID AT TIME N -----
  // ------------------------------------------------------------------------------
  // vector fields
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    // gradient of hodge
    ierr = VecCreateGhostFaces(p4est_n, faces_n, &dxyz_hodge[dir], dir); CHKERRXX(ierr);
    // vstar
    ierr = VecCreateGhostFaces(p4est_n, faces_n, &vstar[dir], dir); CHKERRXX(ierr);
    // vnp1 minus and plus
    ierr = VecCreateGhostFaces(p4est_n, faces_n, &vnp1_minus[dir],  dir); CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est_n, faces_n, &vnp1_plus[dir],   dir); CHKERRXX(ierr);
// //     sign_of_phi
//    ierr = VecCreateGhostFaces(p4est_n, faces_n, &sign_of_phi_face[dir], dir); CHKERRXX(ierr);
  }
  // -------------------------------------------------------------------------
  // ----- FIELDS SAMPLED AT NODES OF THE COMPUTATIONAL GRID AT TIME NM1 -----
  // -------------------------------------------------------------------------
  // vector fields
  for (unsigned char dir  = 0; dir  < P4EST_DIM; ++dir ) {
    vnm1_nodes_omega_minus[dir] = NULL;
    vnm1_nodes_omega_plus[dir]  = NULL;
  // tensor/matrix fields
    for (unsigned char der = 0; der < P4EST_DIM; ++der) {
      vnm1_nodes_omega_minus_xxyyzz[dir][der] = NULL;
      vnm1_nodes_omega_plus_xxyyzz[dir][der] = NULL;
    }
  }

  // clear backtracing points and computational-to-fine-grid maps
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    for (unsigned char comp = 0; comp < P4EST_DIM; ++comp) {
      xyz_n_omega_minus[dir][comp].clear();
      xyz_n_omega_plus[dir][comp].clear();
      xyz_nm1_omega_minus[dir][comp].clear();;
      xyz_nm1_omega_plus[dir][comp].clear();
    }
    face_to_fine_node[dir].clear();
    face_to_fine_node_maps_are_set[dir] = false;
  }
  cell_to_fine_node.clear();
  node_to_fine_node.clear();
  cell_to_fine_node_map_is_set = false;
  node_to_fine_node_map_is_set = false;
  semi_lagrangian_backtrace_is_done = false;

  ngbd_n->init_neighbors();
  ngbd_nm1->init_neighbors();
  interp_phi = new my_p4est_interpolation_nodes_t(fine_ngbd_n);
  if(!faces_n->finest_faces_neighborhoods_have_been_set())
    faces_n->set_finest_face_neighborhoods();
}

#ifdef P4_TO_P8
double my_p4est_two_phase_flows_t::wall_bc_value_hodge_t::operator ()(double x, double y, double z) const
#else
double my_p4est_two_phase_flows_t::wall_bc_value_hodge_t::operator ()(double x, double y) const
#endif
{  
#ifdef P4_TO_P8
  return _prnt->bc_pressure->wallValue(x,y,z) * _prnt->dt_n / (_prnt->BDF_alpha());
#else
  return _prnt->bc_pressure->wallValue(x,y) * _prnt->dt_n / (_prnt->BDF_alpha());
#endif
}

#ifdef P4_TO_P8
void my_p4est_two_phase_flows_t::set_external_forces(CF_3 *external_forces_[P4EST_DIM])
#else
void my_p4est_two_phase_flows_t::set_external_forces(CF_2 *external_forces_[P4EST_DIM])
#endif
{
  for(unsigned char dir=0; dir<P4EST_DIM; ++dir)
    this->external_forces[dir] = external_forces_[dir];
}

#ifdef P4_TO_P8
void my_p4est_two_phase_flows_t::set_bc(BoundaryConditions3D *bc_v, BoundaryConditions3D *bc_p)
#else
void my_p4est_two_phase_flows_t::set_bc(BoundaryConditions2D *bc_v, BoundaryConditions2D *bc_p)
#endif
{
  this->bc_v          = bc_v;
  this->bc_pressure   = bc_p;

  bc_hodge.setWallTypes(bc_pressure->getWallType());
  bc_hodge.setWallValues(wall_bc_value_hodge);
}

my_p4est_two_phase_flows_t::~my_p4est_two_phase_flows_t()
{
  PetscErrorCode ierr;
  // node-sampled fields on the interface-capturing grid
  if(fine_phi!=NULL)                      { ierr = VecDestroy(fine_phi);                      CHKERRXX(ierr); }
  if(fine_curvature!=NULL)                { ierr = VecDestroy(fine_curvature);                CHKERRXX(ierr); }
  if(jump_hodge!=NULL)                    { ierr = VecDestroy(jump_hodge);                    CHKERRXX(ierr); }
  if(jump_normal_flux_hodge!=NULL)        { ierr = VecDestroy(jump_normal_flux_hodge);        CHKERRXX(ierr); }
  if(mass_flux!=NULL)                     { ierr = VecDestroy(mass_flux);                     CHKERRXX(ierr); }
  if(nonconstant_surface_tension!=NULL)   { ierr = VecDestroy(nonconstant_surface_tension);   CHKERRXX(ierr); }
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    if(normal[dir]!=NULL)                 { ierr = VecDestroy(normal[dir]);                   CHKERRXX(ierr); }
    if(fine_phi_xxyyzz[dir] != NULL)      { ierr = VecDestroy(fine_phi_xxyyzz[dir]);          CHKERRXX(ierr); }
    for (unsigned char der = 0; der < P4EST_DIM; ++der)
      if(jump_mu_grad_v[dir][der]!=NULL)  { ierr = VecDestroy(jump_mu_grad_v[dir][der]);      CHKERRXX(ierr); }
  }
  // node-sampled fields on the computational grids n and nm1
  if(vorticity!=NULL)                     { ierr = VecDestroy(vorticity);                     CHKERRXX(ierr); }
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    if(vnm1_nodes_omega_minus[dir]!=NULL) { ierr = VecDestroy(vnm1_nodes_omega_minus[dir]);   CHKERRXX(ierr); }
    if(vnm1_nodes_omega_plus[dir]!=NULL)  { ierr = VecDestroy(vnm1_nodes_omega_plus[dir]);    CHKERRXX(ierr); }
    if(vn_nodes_omega_minus[dir]!=NULL)   { ierr = VecDestroy(vn_nodes_omega_minus[dir]);     CHKERRXX(ierr); }
    if(vn_nodes_omega_plus[dir]!=NULL)    { ierr = VecDestroy(vn_nodes_omega_plus[dir]);      CHKERRXX(ierr); }
    if(vnp1_nodes_omega_minus[dir]!=NULL) { ierr = VecDestroy(vnp1_nodes_omega_minus[dir]);   CHKERRXX(ierr); }
    if(vnp1_nodes_omega_plus[dir]!=NULL)  { ierr = VecDestroy(vnp1_nodes_omega_plus[dir]);    CHKERRXX(ierr); }
    for (unsigned char der = 0; der < P4EST_DIM; ++der) {
      if(vnm1_nodes_omega_minus_xxyyzz[dir][der] != NULL)
                                          { ierr = VecDestroy(vnm1_nodes_omega_minus_xxyyzz[dir][der]); CHKERRXX(ierr); }
      if(vnm1_nodes_omega_plus_xxyyzz[dir][der] != NULL)
                                          { ierr = VecDestroy(vnm1_nodes_omega_plus_xxyyzz[dir][der]);  CHKERRXX(ierr); }
      if(vn_nodes_omega_minus_xxyyzz[dir][der] != NULL)
                                          { ierr = VecDestroy(vn_nodes_omega_minus_xxyyzz[dir][der]);   CHKERRXX(ierr); }
      if(vn_nodes_omega_plus_xxyyzz[dir][der] != NULL)
                                          { ierr = VecDestroy(vn_nodes_omega_plus_xxyyzz[dir][der]);    CHKERRXX(ierr); }
    }
  }
  // cell-sampled fields, computational grid n
  if(hodge!=NULL)                         { ierr = VecDestroy(hodge);                     CHKERRXX(ierr); }
  if(pressure!=NULL)                      { ierr = VecDestroy(pressure);                  CHKERRXX(ierr); }
  // face-sampled fields, computational grid n
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    if(dxyz_hodge[dir]!=NULL)             { ierr = VecDestroy(dxyz_hodge[dir]);           CHKERRXX(ierr); }
    if(vstar[dir]!=NULL)                  { ierr = VecDestroy(vstar[dir]);                CHKERRXX(ierr); }
    if(vnp1_minus[dir]!=NULL)             { ierr = VecDestroy(vnp1_minus[dir]);           CHKERRXX(ierr); }
    if(vnp1_plus[dir]!=NULL)              { ierr = VecDestroy(vnp1_plus[dir]);            CHKERRXX(ierr); }
//    if(sign_of_phi_face[dir]!=NULL)       { ierr = VecDestroy(sign_of_phi_face[dir]);     CHKERRXX(ierr); }
  }

  if(interp_phi!=NULL) delete interp_phi;

  if(ngbd_nm1!=ngbd_n)
    delete ngbd_nm1;
  delete ngbd_n;
  delete fine_ngbd_n;
  if(hierarchy_nm1!=hierarchy_n)
    delete hierarchy_nm1;
  delete hierarchy_n;
  delete fine_hierarchy_n;
  if(nodes_nm1!=nodes_n)
    p4est_nodes_destroy(nodes_nm1);
  p4est_nodes_destroy(nodes_n);
  p4est_nodes_destroy(fine_nodes_n);
  if(p4est_nm1!=p4est_n)
    p4est_destroy(p4est_nm1);
  p4est_destroy(p4est_n);
  p4est_destroy(fine_p4est_n);
  if(ghost_nm1!=ghost_n)
    p4est_ghost_destroy(ghost_nm1);
  p4est_ghost_destroy(ghost_n);
  p4est_ghost_destroy(fine_ghost_n);

  delete faces_n;
  delete ngbd_c;

  my_p4est_brick_destroy(conn, brick);
}

void my_p4est_two_phase_flows_t::set_dynamic_viscosities(double mu_omega_minus, double mu_omega_plus)
{
  mu_minus = mu_omega_minus;
  mu_plus = mu_omega_plus;
}

void my_p4est_two_phase_flows_t::set_surface_tension(double surface_tension_)
{
  surface_tension = surface_tension_;
}

void my_p4est_two_phase_flows_t::set_densities(double rho_omega_minus, double rho_omega_plus)
{
  rho_minus = rho_omega_minus;
  rho_plus = rho_omega_plus;
}

void my_p4est_two_phase_flows_t::set_semi_lagrangian_order(int sl_)
{
  sl_order = sl_;
}

void my_p4est_two_phase_flows_t::set_uniform_bands(double uniform_band_minus_, double uniform_band_plus_)
{
  uniform_band_minus = uniform_band_minus_;
  uniform_band_plus = uniform_band_plus_;
}

void my_p4est_two_phase_flows_t::set_vorticity_split_threshold(double thresh_)
{
  threshold_split_cell = thresh_;
}

void my_p4est_two_phase_flows_t::set_cfl(double cfl_)
{
  cfl = cfl_;
}


void my_p4est_two_phase_flows_t::set_phi(Vec fine_phi_, bool set_second_derivatives)
{
  PetscErrorCode ierr;
  if(this->fine_phi!=NULL){   ierr = VecDestroy(this->fine_phi);  CHKERRXX(ierr); }
  this->fine_phi = fine_phi_;
  compute_normals_curvature_and_second_derivatives(set_second_derivatives);
  if(set_second_derivatives)
#ifdef P4_TO_P8
    interp_phi->set_input(fine_phi, fine_phi_xxyyzz[0], fine_phi_xxyyzz[1], fine_phi_xxyyzz[2], quadratic_non_oscillatory);
#else
    interp_phi->set_input(fine_phi, fine_phi_xxyyzz[0], fine_phi_xxyyzz[1], quadratic_non_oscillatory);
#endif
  else
    interp_phi->set_input(fine_phi, linear);
}

#ifdef P4_TO_P8
void my_p4est_two_phase_flows_t::set_velocities(CF_3* vnm1_omega_minus[P4EST_DIM], CF_3* vn_omega_minus[P4EST_DIM], CF_3* vnm1_omega_plus[P4EST_DIM], CF_3* vn_omega_plus[P4EST_DIM])
#else
void my_p4est_two_phase_flows_t::set_velocities(CF_2* vnm1_omega_minus[P4EST_DIM], CF_2* vn_omega_minus[P4EST_DIM], CF_2* vnm1_omega_plus[P4EST_DIM], CF_2* vn_omega_plus[P4EST_DIM])
#endif
{
  PetscErrorCode ierr;
  double xyz_node[P4EST_DIM];
  double *vnm1_nodes_omega_minus_p, *vnm1_nodes_omega_plus_p, *vn_nodes_omega_minus_p, *vn_nodes_omega_plus_p;
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = create_p4est_nm1_node_vector_if_needed(vnm1_nodes_omega_minus[dir]);   CHKERRXX(ierr);
    ierr = create_p4est_nm1_node_vector_if_needed(vnm1_nodes_omega_plus[dir]);    CHKERRXX(ierr);
    ierr = create_p4est_n_node_vector_if_needed(vn_nodes_omega_minus[dir]);     CHKERRXX(ierr);
    ierr = create_p4est_n_node_vector_if_needed(vn_nodes_omega_plus[dir]);      CHKERRXX(ierr);
    ierr = VecGetArray(vnm1_nodes_omega_minus[dir], &vnm1_nodes_omega_minus_p);       CHKERRXX(ierr);
    ierr = VecGetArray(vnm1_nodes_omega_plus[dir], &vnm1_nodes_omega_plus_p);         CHKERRXX(ierr);
    ierr = VecGetArray(vn_nodes_omega_minus[dir], &vn_nodes_omega_minus_p);           CHKERRXX(ierr);
    ierr = VecGetArray(vn_nodes_omega_plus[dir], &vn_nodes_omega_plus_p);             CHKERRXX(ierr);
    for (unsigned int n = 0; n < MAX(nodes_n->indep_nodes.elem_count, nodes_nm1->indep_nodes.elem_count); ++n) {
      if(n < nodes_n->indep_nodes.elem_count)
      {
        node_xyz_fr_n(n, p4est_n, nodes_n, xyz_node);
        vn_nodes_omega_minus_p[n]     = (*vn_omega_minus[dir])(xyz_node);
        vn_nodes_omega_plus_p[n]      = (*vn_omega_plus[dir])(xyz_node);
      }
      if(n < nodes_nm1->indep_nodes.elem_count)
      {
        node_xyz_fr_n(n, p4est_nm1, nodes_nm1, xyz_node);
        vnm1_nodes_omega_minus_p[n]   = (*vnm1_omega_minus[dir])(xyz_node);
        vnm1_nodes_omega_plus_p[n]    = (*vnm1_omega_plus[dir])(xyz_node);
      }
    }
    ierr = VecRestoreArray(vnm1_nodes_omega_minus[dir], &vnm1_nodes_omega_minus_p);   CHKERRXX(ierr);
    ierr = VecRestoreArray(vnm1_nodes_omega_plus[dir], &vnm1_nodes_omega_plus_p);     CHKERRXX(ierr);
    ierr = VecRestoreArray(vn_nodes_omega_minus[dir], &vn_nodes_omega_minus_p);       CHKERRXX(ierr);
    ierr = VecRestoreArray(vn_nodes_omega_plus[dir], &vn_nodes_omega_plus_p);         CHKERRXX(ierr);
  }

  double *vnm1_nodes_omega_minus_dd_p, *vnm1_nodes_omega_plus_dd_p, *vn_nodes_omega_minus_dd_p, *vn_nodes_omega_plus_dd_p;
  p4est_locidx_t fine_node_idx;
  const quad_neighbor_nodes_of_node_t* qnnn;
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecGetArray(vnm1_nodes_omega_minus[dir], &vnm1_nodes_omega_minus_p);   CHKERRXX(ierr);
    ierr = VecGetArray(vnm1_nodes_omega_plus[dir], &vnm1_nodes_omega_plus_p);     CHKERRXX(ierr);
    ierr = VecGetArray(vn_nodes_omega_minus[dir], &vn_nodes_omega_minus_p);       CHKERRXX(ierr);
    ierr = VecGetArray(vn_nodes_omega_plus[dir], &vn_nodes_omega_plus_p);         CHKERRXX(ierr);
    for (unsigned char der = 0; der < P4EST_DIM; ++der) {
      ierr = create_p4est_nm1_node_vector_if_needed(vnm1_nodes_omega_minus_xxyyzz[dir][der]); CHKERRXX(ierr);
      ierr = create_p4est_nm1_node_vector_if_needed(vnm1_nodes_omega_plus_xxyyzz[dir][der]); CHKERRXX(ierr);
      ierr = create_p4est_n_node_vector_if_needed(vn_nodes_omega_minus_xxyyzz[dir][der]); CHKERRXX(ierr);
      ierr = create_p4est_n_node_vector_if_needed(vn_nodes_omega_plus_xxyyzz[dir][der]); CHKERRXX(ierr);

      ierr = VecGetArray(vnm1_nodes_omega_minus_xxyyzz[dir][der], &vnm1_nodes_omega_minus_dd_p);   CHKERRXX(ierr);
      ierr = VecGetArray(vnm1_nodes_omega_plus_xxyyzz[dir][der], &vnm1_nodes_omega_plus_dd_p);     CHKERRXX(ierr);
      ierr = VecGetArray(vn_nodes_omega_minus_xxyyzz[dir][der], &vn_nodes_omega_minus_dd_p);       CHKERRXX(ierr);
      ierr = VecGetArray(vn_nodes_omega_plus_xxyyzz[dir][der], &vn_nodes_omega_plus_dd_p);         CHKERRXX(ierr);

      for (unsigned int k = 0; k < MAX(ngbd_n->get_layer_size(), ngbd_nm1->get_layer_size()); ++k) {
        if(k < ngbd_n->get_layer_size())
        {
          fine_node_idx = ngbd_n->get_layer_node(k);
          ngbd_n->get_neighbors(fine_node_idx, qnnn);
          vn_nodes_omega_minus_dd_p[fine_node_idx]    = qnnn->d_central(der, vn_nodes_omega_minus_p);
          vn_nodes_omega_plus_dd_p[fine_node_idx]     = qnnn->d_central(der, vn_nodes_omega_plus_p);
        }
        if(k < ngbd_nm1->get_layer_size())
        {
          fine_node_idx = ngbd_nm1->get_layer_node(k);
          ngbd_nm1->get_neighbors(fine_node_idx, qnnn);
          vnm1_nodes_omega_minus_dd_p[fine_node_idx]  = qnnn->d_central(der, vnm1_nodes_omega_minus_p);
          vnm1_nodes_omega_plus_dd_p[fine_node_idx]   = qnnn->d_central(der, vnm1_nodes_omega_plus_p);
        }
      }
      ierr = VecGhostUpdateBegin(vnm1_nodes_omega_minus_xxyyzz[dir][der], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(vnm1_nodes_omega_plus_xxyyzz[dir][der], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(vn_nodes_omega_minus_xxyyzz[dir][der], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(vn_nodes_omega_plus_xxyyzz[dir][der], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      for (unsigned int k = 0; k < MAX(ngbd_n->get_local_size(), ngbd_nm1->get_local_size()); ++k) {
        if(k < ngbd_n->get_local_size())
        {
          fine_node_idx = ngbd_n->get_local_node(k);
          ngbd_n->get_neighbors(fine_node_idx, qnnn);
          vn_nodes_omega_minus_dd_p[fine_node_idx]    = qnnn->d_central(der, vn_nodes_omega_minus_p);
          vn_nodes_omega_plus_dd_p[fine_node_idx]     = qnnn->d_central(der, vn_nodes_omega_plus_p);
        }
        if(k < ngbd_nm1->get_local_size())
        {
          fine_node_idx = ngbd_nm1->get_local_node(k);
          ngbd_nm1->get_neighbors(fine_node_idx, qnnn);
          vnm1_nodes_omega_minus_dd_p[fine_node_idx]  = qnnn->d_central(der, vnm1_nodes_omega_minus_p);
          vnm1_nodes_omega_plus_dd_p[fine_node_idx]   = qnnn->d_central(der, vnm1_nodes_omega_plus_p);
        }
      }
      ierr = VecGhostUpdateEnd(vnm1_nodes_omega_minus_xxyyzz[dir][der], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(vnm1_nodes_omega_plus_xxyyzz[dir][der], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(vn_nodes_omega_minus_xxyyzz[dir][der], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(vn_nodes_omega_plus_xxyyzz[dir][der], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      ierr = VecRestoreArray(vnm1_nodes_omega_minus_xxyyzz[dir][der], &vnm1_nodes_omega_minus_dd_p);   CHKERRXX(ierr);
      ierr = VecRestoreArray(vnm1_nodes_omega_plus_xxyyzz[dir][der], &vnm1_nodes_omega_plus_dd_p);     CHKERRXX(ierr);
      ierr = VecRestoreArray(vn_nodes_omega_minus_xxyyzz[dir][der], &vn_nodes_omega_minus_dd_p);       CHKERRXX(ierr);
      ierr = VecRestoreArray(vn_nodes_omega_plus_xxyyzz[dir][der], &vn_nodes_omega_plus_dd_p);         CHKERRXX(ierr);
    }
    ierr = VecRestoreArray(vnm1_nodes_omega_minus[dir], &vnm1_nodes_omega_minus_p);   CHKERRXX(ierr);
    ierr = VecRestoreArray(vnm1_nodes_omega_plus[dir], &vnm1_nodes_omega_plus_p);     CHKERRXX(ierr);
    ierr = VecRestoreArray(vn_nodes_omega_minus[dir], &vn_nodes_omega_minus_p);       CHKERRXX(ierr);
    ierr = VecRestoreArray(vn_nodes_omega_plus[dir], &vn_nodes_omega_plus_p);         CHKERRXX(ierr);
  }
}

void my_p4est_two_phase_flows_t::compute_normals_curvature_and_second_derivatives(const bool& set_second_derivatives)
{
  PetscErrorCode ierr;
  // make sure normals are properly allocated
  // same for second derivatives if desired
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = create_fine_node_vector_if_needed(normal[dir]); CHKERRXX(ierr);
    if(set_second_derivatives)
    {
      ierr = create_fine_node_vector_if_needed(fine_phi_xxyyzz[dir]); CHKERRXX(ierr);
    }
    else
    {
      ierr = delete_and_nullify_vector_if_needed(fine_phi_xxyyzz[dir]); CHKERRXX(ierr);
    }
  }
  // get the normals now, and the second derivatives, if desired
  // get the pointers
  double *normal_p[P4EST_DIM], *fine_phi_xxyyzz_p[P4EST_DIM];
  const double* fine_phi_read_p;
  ierr = VecGetArrayRead(fine_phi, &fine_phi_read_p); CHKERRXX(ierr);
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecGetArray(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
    if(set_second_derivatives){
      ierr = VecGetArray(fine_phi_xxyyzz[dir], &fine_phi_xxyyzz_p[dir]); CHKERRXX(ierr);}
    else
      fine_phi_xxyyzz_p[dir] = NULL;
  }
  p4est_locidx_t fine_node_idx;
  const quad_neighbor_nodes_of_node_t* qnnn;
  // get the values for the layer nodes
  for (size_t k = 0; k < fine_ngbd_n->layer_nodes.size(); ++k) {
    fine_node_idx = fine_ngbd_n->get_layer_node(k);
    fine_ngbd_n->get_neighbors(fine_node_idx, qnnn);
    compute_gradient_and_second_derivatives(fine_node_idx, qnnn, fine_phi_read_p, normal_p, fine_phi_xxyyzz_p);
  }
  for (unsigned char der = 0; der < P4EST_DIM; ++der) {
    ierr = VecGhostUpdateBegin(normal[der], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    if(set_second_derivatives){
      ierr = VecGhostUpdateBegin(fine_phi_xxyyzz[der], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr); }
  }
  for (size_t k = 0; k < fine_ngbd_n->local_nodes.size(); ++k) {
    fine_node_idx = fine_ngbd_n->get_local_node(k);
    fine_ngbd_n->get_neighbors(fine_node_idx, qnnn);
    compute_gradient_and_second_derivatives(fine_node_idx, qnnn, fine_phi_read_p, normal_p, fine_phi_xxyyzz_p);
  }
  for (unsigned char der = 0; der < P4EST_DIM; ++der) {
    ierr = VecGhostUpdateEnd(normal[der], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    if(set_second_derivatives){
      ierr = VecGhostUpdateEnd(fine_phi_xxyyzz[der], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr); }
  }
  ierr = VecRestoreArrayRead(fine_phi, &fine_phi_read_p); CHKERRXX(ierr);
  for (unsigned char der = 0; der < P4EST_DIM; ++der) {
    ierr = VecRestoreArray(normal[der], &normal_p[der]); CHKERRXX(ierr);
    if(set_second_derivatives){
      ierr = VecRestoreArray(fine_phi_xxyyzz[der], &fine_phi_xxyyzz_p[der]); CHKERRXX(ierr);}
  }
  compute_curvature();
  normalize_normals();
}


void my_p4est_two_phase_flows_t::compute_curvature()
{
  PetscErrorCode ierr;
  // make sure fine curvature is properly allocated
  ierr = create_fine_node_vector_if_needed(fine_curvature); CHKERRXX(ierr);
  double *fine_curvature_p;
  const double *fine_phi_read_p, *normal_p[P4EST_DIM], *fine_phi_xxyyzz_p[P4EST_DIM];
  ierr = VecGetArray(fine_curvature, &fine_curvature_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(fine_phi, &fine_phi_read_p); CHKERRXX(ierr);
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecGetArrayRead(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
    if(fine_phi_xxyyzz[dir]!=NULL){
      ierr = VecGetArrayRead(fine_phi_xxyyzz[dir], &fine_phi_xxyyzz_p[dir]); CHKERRXX(ierr);
    }
    else
      fine_phi_xxyyzz_p[dir] = NULL;
  }
  p4est_locidx_t fine_node_idx;
  const quad_neighbor_nodes_of_node_t* qnnn;
  // get the values for the layer nodes
  for (size_t k = 0; k < fine_ngbd_n->layer_nodes.size(); ++k) {
    fine_node_idx = fine_ngbd_n->get_layer_node(k);
    fine_ngbd_n->get_neighbors(fine_node_idx, qnnn);
    compute_local_curvature(fine_node_idx, qnnn, fine_phi_read_p, fine_phi_xxyyzz_p, normal_p, fine_curvature_p);
  }
  ierr = VecGhostUpdateBegin(fine_curvature, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for (size_t k = 0; k < fine_ngbd_n->local_nodes.size(); ++k) {
    fine_node_idx = fine_ngbd_n->get_local_node(k);
    fine_ngbd_n->get_neighbors(fine_node_idx, qnnn);
    compute_local_curvature(fine_node_idx, qnnn, fine_phi_read_p, fine_phi_xxyyzz_p, normal_p, fine_curvature_p);
  }
  ierr = VecGhostUpdateEnd(fine_curvature, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecRestoreArray(fine_curvature, &fine_curvature_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(fine_phi, &fine_phi_read_p); CHKERRXX(ierr);
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecRestoreArrayRead(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
    if(fine_phi_xxyyzz[dir]!=NULL){
      ierr = VecRestoreArrayRead(fine_phi_xxyyzz[dir], &fine_phi_xxyyzz_p[dir]); CHKERRXX(ierr);
    }
  }
}

void my_p4est_two_phase_flows_t::normalize_normals()
{
  PetscErrorCode ierr;
  double norm_of_grad;
  double *normal_p[P4EST_DIM];
  for (unsigned char der = 0; der < P4EST_DIM; ++der) {
    ierr = VecGetArray(normal[der], &normal_p[der]); CHKERRXX(ierr);
  }
  for (unsigned int n = 0; n < fine_nodes_n->indep_nodes.elem_count; ++n)
  {
    norm_of_grad = 0.0;
    for (unsigned int der = 0; der < P4EST_DIM; ++der)
      norm_of_grad += SQR(normal_p[der][n]);
    norm_of_grad = sqrt(norm_of_grad);
    for (unsigned int der = 0; der < P4EST_DIM; ++der)
      normal_p[der][n] = ((norm_of_grad > threshold_norm_of_n)? (normal_p[der][n]/norm_of_grad): 0.0);
  }
  for (unsigned char der = 0; der < P4EST_DIM; ++der) {
    ierr = VecRestoreArray(normal[der], &normal_p[der]); CHKERRXX(ierr);
  }
}

void my_p4est_two_phase_flows_t::set_dt(double dt_nm1_, double dt_n_)
{
  dt_nm1  = dt_nm1_;
  dt_n    = dt_n_;
}

void my_p4est_two_phase_flows_t::get_velocity_seen_from_cell(neighbor_value& neighbor_velocity, const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const int& face_dir,
                                                             const double *vstar_p, const double *fine_phi_p, const double *fine_phi_dirdir_p, const double *jump_mu_grad_v_p)
{
  P4EST_ASSERT(quad_idx < p4est_n->local_num_quadrants);
  p4est_tree_t* tree = p4est_tree_array_index(p4est_n->trees, tree_idx);
  const p4est_quadrant_t* quad = p4est_quadrant_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);
  neighbor_velocity.distance  = convert_to_xyz[face_dir/2]*((double) P4EST_QUADRANT_LEN(quad->level+1)/((double) P4EST_ROOT_LEN));
  neighbor_velocity.value     = 0.0;
  if(is_quad_Wall(p4est_n, tree_idx, quad, face_dir))
    neighbor_velocity.value   = vstar_p[faces_n->q2f(quad_idx, face_dir)];
  else if(faces_n->q2f(quad_idx, face_dir)!=NO_VELOCITY)
  {
    p4est_locidx_t fine_center_idx, fine_face_idx;
    if(face_is_across(quad_idx, tree_idx, face_dir, fine_phi_p, fine_center_idx, fine_face_idx, quad))
    {
      const double phi_center = fine_phi_p[fine_center_idx];
      double theta;
      if(fine_phi_dirdir_p!=NULL)
        theta = fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(phi_center, fine_phi_p[fine_face_idx], fine_phi_dirdir_p[fine_center_idx], fine_phi_dirdir_p[fine_face_idx], neighbor_velocity.distance);
      else
        theta = fraction_Interval_Covered_By_Irregular_Domain(phi_center, fine_phi_p[fine_face_idx], neighbor_velocity.distance, neighbor_velocity.distance);
      theta = ((phi_center>0.0)? (1.0-theta):theta);
      theta = ((theta < EPS)?0.0:MIN(theta, 1.0));
      neighbor_velocity.distance *= theta;
      p4est_locidx_t opposite_fine_face_idx;
      if(face_is_across(quad_idx, tree_idx, ((face_dir%2==0)? (face_dir+1): (face_dir-1)), fine_phi_p, fine_center_idx, opposite_fine_face_idx, quad))
        throw std::runtime_error("get_velocity_seen_from_cell:: subrefined cell, not implemented yet...");
      double jump_flux_comp_across = (1.0-theta)*jump_mu_grad_v_p[fine_center_idx] + theta*jump_mu_grad_v_p[fine_face_idx];
      theta = 0.5*(1.0+theta);
      neighbor_velocity.value = ((1.0-theta)*((phi_center > 0.0)? mu_plus: mu_minus)*vstar_p[faces_n->q2f(quad_idx, ((face_dir%2==0)? (face_dir+1): (face_dir-1)))]
          + theta*((phi_center > 0.0)? mu_minus: mu_plus)*vstar_p[faces_n->q2f(quad_idx, face_dir)]
          + ((face_dir%2 == 0) ? ((phi_center <= 0) ? +1.0 : -1.0) :
                                 ((phi_center <= 0) ? -1.0 : +1.0))*theta*(1.0-theta)*dxyz_min[face_dir/2]*jump_flux_comp_across
          )/((1.0-theta)*((phi_center > 0.0)? mu_plus: mu_minus) + theta*((phi_center > 0.0)? mu_minus:mu_plus));
    }
    else
    {
      std::vector<p4est_quadrant_t> ngbd(0);
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, face_dir);
      P4EST_ASSERT(ngbd.size()==1);
      p4est_quadrant_t quad_tmp = ngbd[0];
      ngbd.resize(0);
      ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_tmp.p.piggy3.local_num, quad_tmp.p.piggy3.which_tree, ((face_dir%2==1)? face_dir-1:face_dir+1));
      double inv_norm = 1.0/((double) (1<<(quad_tmp.level*(P4EST_DIM-1))));
      for(unsigned int m=0; m<ngbd.size(); ++m)
        neighbor_velocity.value += ((double) (1<<(ngbd[m].level*(P4EST_DIM-1))))*inv_norm*vstar_p[faces_n->q2f(ngbd[m].p.piggy3.local_num, face_dir)];
    }
  }
  else
  {
    std::vector<p4est_quadrant_t> ngbd(0);
    ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, face_dir);
    P4EST_ASSERT(ngbd.size()>1);
    double inv_norm = 1.0/((double) (1<<(quad->level*(P4EST_DIM-1))));
    for(unsigned int m=0; m<ngbd.size(); ++m)
      neighbor_velocity.value += ((double) (1<<(ngbd[m].level*(P4EST_DIM-1))))*inv_norm*vstar_p[faces_n->q2f(ngbd[m].p.piggy3.local_num, ((face_dir%2==1)? face_dir-1:face_dir+1))];
  }
  return;
}

double my_p4est_two_phase_flows_t::compute_divergence(p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, const double *vstar_p[], const double *fine_phi_p,
                                                      const double *fine_phi_xxyyzz_p[], const double *jump_mu_grad_v_p[])
{
  double val = 0;
  neighbor_value nb_p, nb_m;
  for(unsigned char dir=0; dir<P4EST_DIM; ++dir)
  {
    get_velocity_seen_from_cell(nb_p, quad_idx, tree_idx, 2*dir+1,  vstar_p[dir], fine_phi_p, fine_phi_xxyyzz_p[dir], jump_mu_grad_v_p[dir]);
    get_velocity_seen_from_cell(nb_m, quad_idx, tree_idx, 2*dir,    vstar_p[dir], fine_phi_p, fine_phi_xxyyzz_p[dir], jump_mu_grad_v_p[dir]);
    val += (nb_p.value-nb_m.value)/(nb_p.distance + nb_m.distance);
  }
  return val;
}

/* solve the projection step
 * laplace Hodge = -div(vstar)
 * jump_hodge = (dt_n/(alpha*rho))*jump_pressure
 * jump_normal_flux_hodge = (dt_n/(alpha*rho))*jump_normal_flux_pressure = 0.0!
 */
void my_p4est_two_phase_flows_t::solve_projection(my_p4est_xgfm_cells_t* &cell_poisson_jump_solver, const bool activate_xgfm, const KSPType ksp, const PCType pc)
{
  PetscErrorCode ierr;

  ierr = PetscLogEventBegin(log_my_p4est_two_phase_flows_projection, 0, 0, 0, 0); CHKERRXX(ierr);

  Vec rhs;
  ierr = VecDuplicate(hodge, &rhs); CHKERRXX(ierr);

  double *rhs_p;
  const double *vstar_p[P4EST_DIM], *jump_mu_grad_v_p[P4EST_DIM];
  const double *fine_phi_xxyyzz_p[P4EST_DIM];
  const double *fine_phi_p;
  ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecGetArrayRead(vstar[dir], &vstar_p[dir]); CHKERRXX(ierr);
    ierr = VecGetArrayRead(jump_mu_grad_v[dir][dir], &jump_mu_grad_v_p[dir]); CHKERRXX(ierr);
    if(fine_phi_xxyyzz[dir] != NULL)
    {
      ierr = VecGetArrayRead(fine_phi_xxyyzz[dir], &fine_phi_xxyyzz_p[dir]); CHKERRXX(ierr);
    }
    else
      fine_phi_xxyyzz_p[dir] = NULL;
  }
  ierr = VecGetArrayRead(fine_phi, &fine_phi_p); CHKERRXX(ierr);
  /* compute the right-hand-side */
  for(p4est_topidx_t tree_idx=p4est_n->first_local_tree; tree_idx<=p4est_n->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est_n->trees, tree_idx);
    for(size_t q_idx=0; q_idx<tree->quadrants.elem_count; ++q_idx)
    {
      p4est_locidx_t quad_idx = q_idx+tree->quadrants_offset;
      rhs_p[quad_idx] = -compute_divergence(quad_idx, tree_idx, vstar_p, fine_phi_p, fine_phi_xxyyzz_p, jump_mu_grad_v_p);
    }
  }
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecRestoreArrayRead(vstar[dir], &vstar_p[dir]); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(jump_mu_grad_v[dir][dir], &jump_mu_grad_v_p[dir]); CHKERRXX(ierr);
    if(fine_phi_xxyyzz[dir] != NULL)
    {
      ierr = VecRestoreArrayRead(fine_phi_xxyyzz[dir], &fine_phi_xxyyzz_p[dir]); CHKERRXX(ierr);
    }
  }
  ierr = VecRestoreArrayRead(fine_phi, &fine_phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);

  /* solve the linear system */
  if(cell_poisson_jump_solver == NULL)
  {
    cell_poisson_jump_solver = new my_p4est_xgfm_cells_t(ngbd_c, ngbd_n, fine_ngbd_n, activate_xgfm);
    cell_poisson_jump_solver->set_phi(fine_phi);
    cell_poisson_jump_solver->set_normals(normal);
    cell_poisson_jump_solver->set_diagonals(0.0, 0.0);
    cell_poisson_jump_solver->set_mus(1.0/rho_minus, 1.0/rho_plus);
    cell_poisson_jump_solver->set_bc(bc_hodge);
    cell_poisson_jump_solver->set_jumps(jump_hodge, jump_normal_flux_hodge);
  }
  cell_poisson_jump_solver->set_rhs(rhs);
  cell_poisson_jump_solver->solve(ksp, pc);

  cell_poisson_jump_solver->get_flux_components_and_subtract_them_from_velocities(dxyz_hodge, faces_n, vstar, vnp1_minus, vnp1_plus);
  hodge = cell_poisson_jump_solver->get_solution();
  ierr = VecDestroy(rhs); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_two_phase_flows_projection, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_two_phase_flows_t::compute_jump_mu_grad_v()
{
  PetscErrorCode ierr;
  Vec mass_flux_times_normal[P4EST_DIM];
  if(mass_flux!=NULL)
  {
    Vec loc_result, loc_m, loc_n;
    ierr = VecGhostGetLocalForm(mass_flux, &loc_m); CHKERRXX(ierr);
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      ierr = VecCreateGhostNodes(fine_p4est_n, fine_nodes_n, &mass_flux_times_normal[dir]); CHKERRXX(ierr);
      ierr = VecGhostGetLocalForm(mass_flux_times_normal[dir], &loc_result); CHKERRXX(ierr);
      ierr = VecGhostGetLocalForm(normal[dir], &loc_n); CHKERRXX(ierr);
      ierr = VecPointwiseMult(loc_result, loc_m, loc_n); CHKERRXX(ierr);
      ierr = VecGhostRestoreLocalForm(mass_flux_times_normal[dir], &loc_result); CHKERRXX(ierr);
      ierr = VecGhostRestoreLocalForm(normal[dir], &loc_n); CHKERRXX(ierr);
    }
    ierr = VecGhostRestoreLocalForm(mass_flux, &loc_m); CHKERRXX(ierr);
  }

  const double *fine_curvature_p;
  const double *normal_p[P4EST_DIM];
  const double *vn_nodes_underlined_p[P4EST_DIM];
  const double *mass_flux_p, *nonconstant_surface_tension_p, *mass_flux_times_normal_p[P4EST_DIM];
  double* jump_mu_grad_v_p[P4EST_DIM][P4EST_DIM];
  const quad_neighbor_nodes_of_node_t* qnnn;

  if(mass_flux!=NULL)
  {
    ierr = VecGetArrayRead(mass_flux, &mass_flux_p); CHKERRXX(ierr);
  }
  else
    mass_flux_p = NULL;
  if(nonconstant_surface_tension!=NULL){
    ierr = VecGetArrayRead(nonconstant_surface_tension, &nonconstant_surface_tension_p); CHKERRXX(ierr); }
  else
    nonconstant_surface_tension_p = NULL;
  ierr = VecGetArrayRead(fine_curvature, &fine_curvature_p); CHKERRXX(ierr);
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    for (unsigned char der = 0; der < P4EST_DIM; ++der) {
      ierr = create_fine_node_vector_if_needed(jump_mu_grad_v[dir][der]);
      ierr = VecGetArray(jump_mu_grad_v[dir][der], &jump_mu_grad_v_p[dir][der]); CHKERRXX(ierr);
    }
    ierr = VecGetArrayRead(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
    ierr = VecGetArrayRead(((underlined_side(velocity_field) == OMEGA_PLUS)? vn_nodes_omega_plus[dir]:vn_nodes_omega_minus[dir]), &vn_nodes_underlined_p[dir]); CHKERRXX(ierr);
    if(mass_flux!=NULL)
    {
      ierr = VecGetArrayRead(mass_flux_times_normal[dir], &mass_flux_times_normal_p[dir]); CHKERRXX(ierr);
    }
    else
    {
      mass_flux_times_normal_p[dir] = NULL;
    }
  }
  for (unsigned int kk = 0; kk < fine_ngbd_n->get_layer_size(); ++kk) {
    p4est_locidx_t fine_node_idx = fine_ngbd_n->get_layer_node(kk);
    fine_ngbd_n->get_neighbors(fine_node_idx, qnnn);
    compute_local_jump_mu_grad_v_elements(fine_node_idx, qnnn,
                                          vn_nodes_underlined_p, normal_p,
                                          mass_flux_p, mass_flux_times_normal_p,
                                          nonconstant_surface_tension_p, fine_curvature_p,
                                          jump_mu_grad_v_p);
  }
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    for (unsigned char der = 0; der < P4EST_DIM; ++der) {
      ierr = VecGhostUpdateBegin(jump_mu_grad_v[dir][der], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }
  }
  for (unsigned int kk = 0; kk < fine_ngbd_n->get_local_size(); ++kk) {
    p4est_locidx_t fine_node_idx = fine_ngbd_n->get_local_node(kk);
    fine_ngbd_n->get_neighbors(fine_node_idx, qnnn);
    compute_local_jump_mu_grad_v_elements(fine_node_idx, qnnn,
                                          vn_nodes_underlined_p, normal_p,
                                          mass_flux_p, mass_flux_times_normal_p,
                                          nonconstant_surface_tension_p, fine_curvature_p,
                                          jump_mu_grad_v_p);
  }
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    for (unsigned char der = 0; der < P4EST_DIM; ++der) {
      ierr = VecGhostUpdateEnd(jump_mu_grad_v[dir][der], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }
  }

  if(mass_flux!=NULL)
  {
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      ierr = VecRestoreArrayRead(mass_flux_times_normal[dir], &mass_flux_times_normal_p[dir]); CHKERRXX(ierr);
      ierr = VecDestroy(mass_flux_times_normal[dir]); CHKERRXX(ierr);
    }
    ierr = VecRestoreArrayRead(mass_flux, &mass_flux_p); CHKERRXX(ierr);
  }
  if(nonconstant_surface_tension!=NULL)
  {
    ierr = VecRestoreArrayRead(nonconstant_surface_tension, &nonconstant_surface_tension_p); CHKERRXX(ierr);
  }
  ierr = VecRestoreArrayRead(fine_curvature, &fine_curvature_p); CHKERRXX(ierr);
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    for (unsigned char der = 0; der < P4EST_DIM; ++der) {
      ierr = VecRestoreArray(jump_mu_grad_v[dir][der], &jump_mu_grad_v_p[dir][der]); CHKERRXX(ierr);
    }
    ierr = VecRestoreArrayRead(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(((underlined_side(velocity_field) == OMEGA_PLUS)? vn_nodes_omega_plus[dir]:vn_nodes_omega_minus[dir]), &vn_nodes_underlined_p[dir]); CHKERRXX(ierr);
  }
}

void my_p4est_two_phase_flows_t::compute_jumps_hodge()
{
  PetscErrorCode ierr;
  ierr = create_fine_node_vector_if_needed(jump_hodge);                   CHKERRXX(ierr);
  ierr = create_fine_node_vector_if_needed(jump_normal_flux_hodge);       CHKERRXX(ierr);

  const double *fine_curvature_p;
  const double *fine_phi_p;
  double *jump_hodge_p, *jump_normal_flux_hodge_p;
  ierr = VecGetArray(jump_hodge, &jump_hodge_p);                          CHKERRXX(ierr);
  ierr = VecGetArray(jump_normal_flux_hodge, &jump_normal_flux_hodge_p);  CHKERRXX(ierr);
  ierr = VecGetArrayRead(fine_curvature, &fine_curvature_p);              CHKERRXX(ierr);
  ierr = VecGetArrayRead(fine_phi, &fine_phi_p);              CHKERRXX(ierr);
  for (unsigned int k = 0; k < fine_nodes_n->indep_nodes.elem_count; ++k) {
    jump_hodge_p[k] = surface_tension*fine_curvature_p[k]*dt_n/(BDF_alpha());
    jump_normal_flux_hodge_p[k] = 0.0;
  }
  ierr = VecRestoreArrayRead(fine_curvature, &fine_curvature_p);              CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(fine_phi, &fine_phi_p);              CHKERRXX(ierr);
  ierr = VecRestoreArray(jump_hodge, &jump_hodge_p);                          CHKERRXX(ierr);
  ierr = VecRestoreArray(jump_normal_flux_hodge, &jump_normal_flux_hodge_p);  CHKERRXX(ierr);
}

void my_p4est_two_phase_flows_t::solve_viscosity_explicit()
{
  PetscErrorCode ierr;

  ierr = PetscLogEventBegin(log_my_p4est_two_phase_flows_viscosity, 0, 0, 0, 0); CHKERRXX(ierr);

  double alpha  = BDF_alpha();
  double beta   = BDF_beta();


  /* construct the right hand side */
  if(!semi_lagrangian_backtrace_is_done)
  {
    trajectory_from_np1_all_faces_transposed_second_derivatives(p4est_n, faces_n, ngbd_nm1, ngbd_n,
                                                                vnm1_nodes_omega_minus, vnm1_nodes_omega_minus_xxyyzz,
                                                                vn_nodes_omega_minus, vn_nodes_omega_minus_xxyyzz, dt_nm1, dt_n,
                                                                xyz_n_omega_minus, ((sl_order == 2)? xyz_nm1_omega_minus : NULL));
    trajectory_from_np1_all_faces_transposed_second_derivatives(p4est_n, faces_n, ngbd_nm1, ngbd_n,
                                                                vnm1_nodes_omega_plus, vnm1_nodes_omega_plus_xxyyzz,
                                                                vn_nodes_omega_plus, vn_nodes_omega_plus_xxyyzz, dt_nm1, dt_n,
                                                                xyz_n_omega_plus, ((sl_order == 2)? xyz_nm1_omega_plus : NULL));
    semi_lagrangian_backtrace_is_done = true;
  }
  const double *fine_phi_p;
  const double *fine_phi_xxyyzz_p[P4EST_DIM];
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    if(fine_phi_xxyyzz[dir] != NULL){
      ierr = VecGetArrayRead(fine_phi_xxyyzz[dir], &fine_phi_xxyyzz_p[dir]); CHKERRXX(ierr); }
    else
      fine_phi_xxyyzz_p[dir] = NULL;
  }
  const double *jump_mu_grad_vdir_p[P4EST_DIM];
  ierr = VecGetArrayRead(fine_phi, &fine_phi_p); CHKERRXX(ierr);
  for(unsigned char dir=0; dir<P4EST_DIM; ++dir)
  {
    for (unsigned char der = 0; der < P4EST_DIM; ++der) {
      ierr = VecGetArrayRead(jump_mu_grad_v[dir][der], &jump_mu_grad_vdir_p[der]); CHKERRXX(ierr); }
    /* find the velocity at the backtraced points */
    my_p4est_interpolation_nodes_t interp_nm1_omega_minus(ngbd_nm1), interp_nm1_omega_plus(ngbd_nm1);
    my_p4est_interpolation_nodes_t interp_n_omega_minus  (ngbd_n),   interp_n_omega_plus  (ngbd_n);
    for(p4est_locidx_t f_idx=0; f_idx<faces_n->num_local[dir]; ++f_idx)
    {
      double xyz_tmp[P4EST_DIM];
      const bool face_is_in_omega_minus = is_face_in_omega_minus(f_idx, dir, fine_phi_p);

      if(sl_order==2)
      {
        for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
          xyz_tmp[dim] = (face_is_in_omega_minus? xyz_nm1_omega_minus[dir][dim][f_idx] : xyz_nm1_omega_plus[dir][dim][f_idx]);
        if(face_is_in_omega_minus)
          interp_nm1_omega_minus.add_point(f_idx, xyz_tmp);
        else
          interp_nm1_omega_plus.add_point(f_idx, xyz_tmp);
      }

      for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
        xyz_tmp[dim] = (face_is_in_omega_minus? xyz_n_omega_minus[dir][dim][f_idx] : xyz_n_omega_plus[dir][dim][f_idx]);
      if(face_is_in_omega_minus)
        interp_n_omega_minus.add_point(f_idx, xyz_tmp);
      else
        interp_n_omega_plus.add_point(f_idx, xyz_tmp);
    }

    std::vector<double> vnm1_faces(faces_n->num_local[dir]);
    if(sl_order==2)
    {
      vnm1_faces.resize(faces_n->num_local[dir]);
#ifdef P4_TO_P8
      interp_nm1_omega_minus.set_input(vnm1_nodes_omega_minus[dir], vnm1_nodes_omega_minus_xxyyzz[0][dir], vnm1_nodes_omega_minus_xxyyzz[1][dir], vnm1_nodes_omega_minus_xxyyzz[2][dir], quadratic);
      interp_nm1_omega_plus.set_input(vnm1_nodes_omega_plus[dir], vnm1_nodes_omega_plus_xxyyzz[0][dir], vnm1_nodes_omega_plus_xxyyzz[1][dir], vnm1_nodes_omega_plus_xxyyzz[2][dir], quadratic);
#else
      interp_nm1_omega_minus.set_input(vnm1_nodes_omega_minus[dir], vnm1_nodes_omega_minus_xxyyzz[0][dir], vnm1_nodes_omega_minus_xxyyzz[1][dir], quadratic);
      interp_nm1_omega_plus.set_input(vnm1_nodes_omega_plus[dir], vnm1_nodes_omega_plus_xxyyzz[0][dir], vnm1_nodes_omega_plus_xxyyzz[1][dir], quadratic);
#endif
      interp_nm1_omega_minus.interpolate(vnm1_faces.data());
      interp_nm1_omega_plus.interpolate(vnm1_faces.data());
    }
    Vec vn_faces;
    double *vn_faces_p;
    ierr = VecCreateGhostFaces(p4est_n, faces_n, &vn_faces, dir); CHKERRXX(ierr);
    ierr = VecGetArray(vn_faces, &vn_faces_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    interp_n_omega_minus.set_input(vn_nodes_omega_minus[dir], vn_nodes_omega_minus_xxyyzz[0][dir], vn_nodes_omega_minus_xxyyzz[1][dir], vn_nodes_omega_minus_xxyyzz[2][dir], quadratic);
    interp_n_omega_plus.set_input(vn_nodes_omega_plus[dir], vn_nodes_omega_plus_xxyyzz[0][dir], vn_nodes_omega_plus_xxyyzz[1][dir], vn_nodes_omega_plus_xxyyzz[2][dir], quadratic);
#else
    interp_n_omega_minus.set_input(vn_nodes_omega_minus[dir], vn_nodes_omega_minus_xxyyzz[0][dir], vn_nodes_omega_minus_xxyyzz[1][dir], quadratic);
    interp_n_omega_plus.set_input(vn_nodes_omega_plus[dir], vn_nodes_omega_plus_xxyyzz[0][dir], vn_nodes_omega_plus_xxyyzz[1][dir], quadratic);
#endif
    interp_n_omega_minus.interpolate(vn_faces_p /*.data()*/);
    interp_n_omega_plus.interpolate(vn_faces_p /*.data()*/);

    ierr = VecGhostUpdateBegin(vn_faces, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(vn_faces, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    double* vstar_p;
    ierr = VecGetArray(vstar[dir], &vstar_p); CHKERRXX(ierr);
    for(p4est_locidx_t f_idx=0; f_idx<faces_n->num_local[dir]; ++f_idx)
    {
      double xyz[P4EST_DIM];
      faces_n->xyz_fr_f(f_idx, dir, xyz);
      if(!face_is_dirichlet_wall(f_idx, dir, xyz))
      {
        p4est_locidx_t fine_idx_of_face;
        const bool face_is_in_omega_minus = is_face_in_omega_minus(f_idx, dir, fine_phi_p, fine_idx_of_face);
        vstar_p[f_idx] = + (dt_n/(alpha*(face_is_in_omega_minus?rho_minus:rho_plus)))*div_mu_grad_u_dir(f_idx, dir, face_is_in_omega_minus, fine_idx_of_face, vn_faces_p, jump_mu_grad_vdir_p, fine_phi_p, fine_phi_xxyyzz_p) + (beta*dt_n/(alpha*dt_nm1))*vnm1_faces[f_idx] + (1.0-(beta*dt_n/(alpha*dt_nm1)))*vn_faces_p[f_idx];
        if (external_forces[dir]!=NULL)
        {
#ifdef P4_TO_P8
          vstar_p[f_idx] += (dt_n/(alpha*(face_is_in_omega_minus?rho_minus:rho_plus)))*(*external_forces[dir])(xyz[0], xyz[1], xyz[2]);
#else
          vstar_p[f_idx] += (dt_n/(alpha*(face_is_in_omega_minus?rho_minus:rho_plus)))*(*external_forces[dir])(xyz[0], xyz[1]);
#endif
        }
      }
      else
        vstar_p[f_idx] = bc_v[dir].wallValue(xyz);
    }

    ierr = VecRestoreArray(vstar[dir], &vstar_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(vn_faces, &vn_faces_p); CHKERRXX(ierr);
    for (unsigned char der = 0; der < P4EST_DIM; ++der) {
      ierr = VecRestoreArrayRead(jump_mu_grad_v[dir][der], &jump_mu_grad_vdir_p[der]); CHKERRXX(ierr); }
    ierr = VecDestroy(vn_faces); CHKERRXX(ierr);
  }
  ierr = VecRestoreArrayRead(fine_phi, &fine_phi_p); CHKERRXX(ierr);
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    if(fine_phi_xxyyzz[dir] != NULL){
      ierr = VecRestoreArrayRead(fine_phi_xxyyzz[dir], &fine_phi_xxyyzz_p[dir]); CHKERRXX(ierr); }
  }

  ierr = PetscLogEventEnd(log_my_p4est_two_phase_flows_viscosity, 0, 0, 0, 0); CHKERRXX(ierr);
}

double my_p4est_two_phase_flows_t::div_mu_grad_u_dir(const p4est_locidx_t& face_idx, const unsigned char& dir,
                                                     const bool& face_is_in_omega_minus, const p4est_locidx_t & fine_idx_of_face,
                                                     const double *vn_dir_p, const double *jump_mu_grad_vdir_p[P4EST_DIM],
                                                     const double *fine_phi_p, const double *fine_phi_xxyyzz_p[P4EST_DIM])
{
  bool xgfm_treatment_required;
  double xgfm_fluxes[P4EST_FACES];
#ifdef P4_TO_P8
  Voronoi3D voro_cell = compute_voronoi_cell(face_idx, dir, face_is_in_omega_minus, fine_idx_of_face, fine_phi_p, fine_phi_xxyyzz_p, vn_dir_p, jump_mu_grad_vdir_p, xgfm_treatment_required, xgfm_fluxes);
  const vector<ngbd3Dseed> *points;
#else
  Voronoi2D voro_cell = compute_voronoi_cell(face_idx, dir, face_is_in_omega_minus, fine_idx_of_face, fine_phi_p, fine_phi_xxyyzz_p, vn_dir_p, jump_mu_grad_vdir_p, xgfm_treatment_required, xgfm_fluxes);
  vector<ngbd2Dseed> *points;
  vector<Point2> *partition;
  voro_cell.get_partition(partition);
#endif
  voro_cell.get_neighbor_seeds(points);

  double to_return = 0.0;
  if(!xgfm_treatment_required)
  {
    const double voro_volume = voro_cell.get_volume();
    double voro_face_area, voro_neighbor_distance;
    for (size_t m = 0; m < points->size(); ++m) {
#ifdef P4_TO_P8
      voro_face_area = (*points)[m].s;
#else
      unsigned int k = mod(m-1, points->size());
      voro_face_area = ((*partition)[m] - (*partition)[k]).norm_L2();
#endif
      voro_neighbor_distance = ((*points)[m].p - voro_cell.get_center_point()).norm_L2();
      if((*points)[m].n >=0)
        to_return += (face_is_in_omega_minus? mu_minus:mu_plus)*voro_face_area*(vn_dir_p[(*points)[m].n]-vn_dir_p[face_idx])/voro_neighbor_distance;
      else
      {
        double xyz_wall[P4EST_DIM];
        xyz_wall[0] = voro_cell.get_center_point().x;
        xyz_wall[1] = voro_cell.get_center_point().y;
#ifdef P4_TO_P8
        xyz_wall[2] = voro_cell.get_center_point().z;
#endif
        xyz_wall[(-1-(*points)[m].n)/2] = (((-1-(*points)[m].n)%2==0)?xyz_min[(-1-(*points)[m].n)/2] : xyz_max[(-1-(*points)[m].n)/2]);
        P4EST_ASSERT(bc_v[dir].wallType(xyz_wall) == DIRICHLET);
        to_return += (face_is_in_omega_minus? mu_minus:mu_plus)*voro_face_area*(bc_v[dir].wallValue(xyz_wall)-vn_dir_p[face_idx])/(voro_neighbor_distance/2);
      }
    }
    to_return /= voro_volume;
  }
  else
    for (unsigned char ff = 0; ff < P4EST_DIM; ++ff)
      to_return += (xgfm_fluxes[2*ff+1] - xgfm_fluxes[2*ff])/dxyz_min[ff];

  return to_return;
}

#ifdef P4_TO_P8
Voronoi3D my_p4est_two_phase_flows_t::compute_voronoi_cell(const p4est_locidx_t& face_idx, const unsigned char& dir,
                                                           const bool& face_is_in_omega_minus, const p4est_locidx_t & fine_idx_of_face,
                                                           const double* fine_phi_p, const double *fine_phi_xxyyzz_p[P4EST_DIM],
                                                           const double* vn_dir_p, const double* jump_mu_grad_vdir_p[P4EST_DIM],
                                                           bool& xgfm_treatment_required, double xgfm_fluxes[P4EST_DIM])
#else
Voronoi2D my_p4est_two_phase_flows_t::compute_voronoi_cell(const p4est_locidx_t& face_idx, const unsigned char& dir,
                                                           const bool& face_is_in_omega_minus, const p4est_locidx_t & fine_idx_of_face,
                                                           const double* fine_phi_p, const double *fine_phi_xxyyzz_p[P4EST_DIM],
                                                           const double* vn_dir_p, const double* jump_mu_grad_vdir_p[P4EST_DIM],
                                                           bool& xgfm_treatment_required, double xgfm_fluxes[P4EST_DIM])
#endif
{
  PetscErrorCode ierr;

  ierr = PetscLogEventBegin(log_my_p4est_two_phase_compute_voronoi_cell, 0, 0, 0, 0); CHKERRXX(ierr);

  xgfm_treatment_required = false;

#ifdef P4_TO_P8
  Voronoi3D voro_tmp;
#else
  Voronoi2D voro_tmp;
#endif
  voro_tmp.clear();

  double xyz_face[P4EST_DIM], dxyz[P4EST_DIM];
  faces_n->xyz_fr_f(face_idx, dir, xyz_face);
  p4est_quadrant_t qm, qp;
  faces_n->find_quads_touching_face(face_idx, dir, qm, qp);
  for (unsigned char dd = 0; dd < P4EST_DIM; ++dd)
    dxyz[dd] = convert_to_xyz[dd]*((double) P4EST_QUADRANT_LEN(MAX(qm.level, qp.level)))/((double) P4EST_ROOT_LEN);

  /* check for walls */
  if((qm.p.piggy3.local_num==-1 && bc_v[dir].wallType(xyz_face)==DIRICHLET) || (qp.p.piggy3.local_num==-1 && bc_v[dir].wallType(xyz_face)==DIRICHLET))
    throw std::invalid_argument(" my_p4est_two_phase_flows_t::compute_voronoi_cell(): called for a Dirichlet wall cell...");

  // check if the face of interest is one of the finest quadrants' in the uniform region
  const uniform_face_ngbd* face_neighbors;
  if(faces_n->found_uniform_face_neighborhood(face_idx, dir, face_neighbors)) // is it in the finest uniform tesselation? (possible presence of interface, no need to search for neighbors already done)
  {
#ifdef P4_TO_P8
    vector<ngbd3Dseed> points(P4EST_FACES);
#else
    vector<ngbd2Dseed> points(P4EST_FACES);
    vector<Point2> partition(P4EST_FACES);
#endif
    for (unsigned char face_dir = 0; face_dir < P4EST_FACES; ++face_dir) {
      points[face_dir].n = face_neighbors->neighbor_face_idx[face_dir];
      faces_n->point_fr_f(points[face_dir].n, dir, points[face_dir].p);
#ifdef P4_TO_P8
      points[face_dir].s      = ((face_dir/2==dir::x) ? dy*dz : ((face_dir/2==dir::y) ? dx*dz : dx*dy));
#else
      points[face_dir].theta  = ((double)(face_dir/2))*0.5*PI + ((double)(1-face_dir%2))*PI;
      partition[face_dir].x   = points[face_dir].p.x + ((face_dir/2==1) ? ((1.0 - 2.0*((double) (face_dir%2)))*0.5*dxyz[0]): 0.0);
      partition[face_dir].y   = points[face_dir].p.y + ((face_dir/2==0) ? ((((double) (2*(face_dir%2)))-1.0)*0.5*dxyz[1]): 0.0);
#endif
    }
#ifdef P4_TO_P8
    voro_tmp.set_cell(points, dxyz[0]*dxyz[1]*dxyz[2]);
#else
    voro_tmp.set_neighbors_and_partition(points, partition, dxyz[0]*dxyz[1]);
    voro_tmp.reorder_neighbors_and_partition_from_faces_to_counterclock_cycle();
#endif

    sharp_derivative local_one_sided_derivative;
    for (unsigned char ff = 0; ff < P4EST_FACES; ++ff) {
      sharp_derivative_of_face_field(local_one_sided_derivative, face_idx, face_is_in_omega_minus, fine_idx_of_face, face_neighbors,
                                     fine_phi_p, fine_phi_xxyyzz_p, ff, dir,
                                     vn_dir_p, vn_dir_p, jump_mu_grad_vdir_p,
                                     qm, qp, NULL, NULL); // use vn_dir_p for both, since before projection step
      xgfm_fluxes[ff] = (face_is_in_omega_minus? mu_minus:mu_plus)*local_one_sided_derivative.derivative;
      xgfm_treatment_required = local_one_sided_derivative.xgfm;
    }
  }
  else
  {
    /* find direct neighbors */
    vector<p4est_quadrant_t> ngbd_m_[2*(P4EST_DIM-1)]; // neighbors of qm in transverse cartesian direction
    vector<p4est_quadrant_t> ngbd_p_[2*(P4EST_DIM-1)]; // neighbors of qp in transverse cartesian direction
#ifdef P4_TO_P8
    P4EST_ASSERT((dir==dir::x) || (dir==dir::y) || (dir==dir::z));
#else
    P4EST_ASSERT((dir==dir::x) || (dir==dir::y));
#endif
    unsigned char ngbd_idx = 0;
    for (unsigned char tranverse_dir = 0; tranverse_dir < P4EST_DIM; ++tranverse_dir) {
      if(tranverse_dir == dir)
        continue;
      for (char cartesian_search = -1; cartesian_search < 2; cartesian_search+=2)
      {
        if(qm.p.piggy3.local_num != -1)
#ifdef P4_TO_P8
          ngbd_c->find_neighbor_cells_of_cell(ngbd_m_[ngbd_idx], qm.p.piggy3.local_num, qm.p.piggy3.which_tree, ((tranverse_dir==dir::x)?cartesian_search:0), ((tranverse_dir==dir::y)?cartesian_search:0), ((tranverse_dir==dir::z)?cartesian_search:0));
#else
          ngbd_c->find_neighbor_cells_of_cell(ngbd_m_[ngbd_idx], qm.p.piggy3.local_num, qm.p.piggy3.which_tree, ((tranverse_dir==dir::x)?cartesian_search:0), ((tranverse_dir==dir::y)?cartesian_search:0));
#endif
        if(qp.p.piggy3.local_num != 1)
#ifdef P4_TO_P8
          ngbd_c->find_neighbor_cells_of_cell(ngbd_p_[ngbd_idx], qp.p.piggy3.local_num, qp.p.piggy3.which_tree, ((tranverse_dir==dir::x)?cartesian_search:0), ((tranverse_dir==dir::y)?cartesian_search:0), ((tranverse_dir==dir::z)?cartesian_search:0));
#else
          ngbd_c->find_neighbor_cells_of_cell(ngbd_p_[ngbd_idx], qp.p.piggy3.local_num, qp.p.piggy3.which_tree, ((tranverse_dir==dir::x)?cartesian_search:0), ((tranverse_dir==dir::y)?cartesian_search:0));
#endif
        ngbd_idx++;
      }
    }
    P4EST_ASSERT(ngbd_idx==2*(P4EST_DIM-1));

    /* now gather the neighbor cells to get the potential voronoi neighbors */
#ifdef P4_TO_P8
    voro_tmp.set_center_point(face_idx, xyz_face);
#else
    voro_tmp.set_center_point(xyz_face);
#endif

    /* check for uniform case, if so build voronoi partition by hand */
    bool is_uniform = (qp.level == qm.level);
    for (unsigned char k = 0; is_uniform && (k < 2*(P4EST_DIM-1)); ++k)
      is_uniform = is_uniform && (ngbd_m_[k].size() == 1) && (ngbd_p_[k].size() == 1) && (ngbd_m_[k][0].level == qm.level) && (ngbd_p_[k][0].level == qp.level) && (faces_n->q2f(ngbd_m_[k][0].p.piggy3.local_num, 2*dir) != NO_VELOCITY) && (faces_n->q2f(ngbd_p_[k][0].p.piggy3.local_num, 2*dir+1) != NO_VELOCITY);
    is_uniform = is_uniform && (faces_n->q2f(qm.p.piggy3.local_num, 2*dir)!=NO_VELOCITY) && (faces_n->q2f(qp.p.piggy3.local_num, 2*dir+1)!=NO_VELOCITY);
    if(is_uniform)
    {
      P4EST_ASSERT(qm.level < ((splitting_criteria_t*) p4est_n->user_pointer)->max_lvl); // finest level of uniform face neighborhoods must have been addressed previously!
#ifdef P4EST_DEBUG
      for (unsigned char k = 0; k < 2*(P4EST_DIM-1); ++k)
        P4EST_ASSERT(faces_n->q2f(ngbd_m_[k][0].p.piggy3.local_num, 2*dir+1) == faces_n->q2f(ngbd_p_[k][0].p.piggy3.local_num, 2*dir));
#endif

#ifdef P4_TO_P8
      vector<ngbd3Dseed> points(P4EST_FACES);
#else
      vector<ngbd2Dseed> points(P4EST_FACES);
      vector<Point2> partition(P4EST_FACES);
#endif
      ngbd_idx = 0;
      for (unsigned char cartesian_dir = 0; cartesian_dir < P4EST_DIM; ++cartesian_dir) {
        if(cartesian_dir == dir)
        {
          points[2*dir].n  = faces_n->q2f(qm.p.piggy3.local_num, 2*dir);
          faces_n->point_fr_f(points[2*dir].n, dir, points[2*dir].p);
#ifdef P4_TO_P8
          points[2*dir].s     = ((dir==dir::x) ? dxyz[1]*dxyz[2] : ((dir==dir::y) ? dxyz[0]*dxyz[2] : dxyz[0]*dxyz[1]));
#else
          points[2*dir].theta = ((double) dir)*0.5*PI + PI;
          partition[2*dir].x   = points[2*dir].p.x + ((dir==1) ? 0.5*dxyz[0]: 0.0);
          partition[2*dir].y   = points[2*dir].p.y - ((dir==0) ? 0.5*dxyz[1]: 0.0);
#endif
          points[2*dir+1].n  = faces_n->q2f(qp.p.piggy3.local_num, 2*dir+1);
          faces_n->point_fr_f(points[2*dir+1].n, dir, points[2*dir+1].p);
#ifdef P4_TO_P8
          points[2*dir+1].s     = ((dir==dir::x) ? dxyz[1]*dxyz[2] : ((dir==dir::y) ? dxyz[0]*dxyz[2] : dxyz[0]*dxyz[1]));
#else
          points[2*dir+1].theta = ((double) dir)*0.5*PI;
          partition[2*dir+1].x   = points[2*dir+1].p.x - ((dir==1) ? 0.5*dxyz[0]: 0.0);
          partition[2*dir+1].y   = points[2*dir+1].p.y + ((dir==0) ? 0.5*dxyz[1]: 0.0);
#endif
        }
        else
          for (char cart_search = -1; cart_search < 2; cart_search+=2) {
            unsigned char ff_idx = 2*cartesian_dir + ((cart_search == 1) ? 1:0);
            points[ff_idx].n   = faces_n->q2f(ngbd_p_[ngbd_idx][0].p.piggy3.local_num, 2*dir); // we loop through ngbd_p_ in the same order as when it was built
            faces_n->point_fr_f(points[ff_idx].n, dir, points[ff_idx].p);
#ifdef P4_TO_P8
            points[ff_idx].s      = ((cartesian_dir==dir::x) ? dxyz[1]*dxyz[2] : ((cartesian_dir==dir::y) ? dxyz[0]*dxyz[2] : dxyz[0]*dxyz[1]));
#else
            points[ff_idx].theta  =((double)(ff_idx/2))*0.5*PI + ((double)(1-ff_idx%2))*PI;
            partition[ff_idx].x   = points[ff_idx].p.x + ((ff_idx/2==1) ? ((1.0 - 2.0*((double) (ff_idx%2)))*0.5*dxyz[0]): 0.0);
            partition[ff_idx].y   = points[ff_idx].p.y + ((ff_idx/2==0) ? ((((double) (2*(ff_idx%2)))-1.0)*0.5*dxyz[1]): 0.0);
#endif
            ngbd_idx++;
          }
      }
      P4EST_ASSERT(ngbd_idx == 2*(P4EST_DIM-1));

#ifdef P4_TO_P8
      voro_tmp.set_cell(points, dxyz[0]*dxyz[1]*dxyz[2]);
#else
      voro_tmp.set_neighbors_and_partition(points, partition, dxyz[0]*dxyz[1]);
      voro_tmp.reorder_neighbors_and_partition_from_faces_to_counterclock_cycle();
#endif
    }
    /* otherwise, there is a T-junction and the grid is not uniform, need to compute the voronoi cell */
    else
    {
      const bool periodic[] = {is_periodic(p4est_n, dir::x), is_periodic(p4est_n, dir::y)
                           #ifdef P4_TO_P8
                               , is_periodic(p4est_n, dir::z)
                           #endif
                              };
      /* note that the walls are dealt with by voro++ in 3D */
#ifndef P4_TO_P8
      const unsigned char other_cartesian_dir = ((dir==dir::x)? (dir::y):(dir::x));
      if(qm.p.piggy3.local_num==-1 && bc_v[dir].wallType(xyz_face)==NEUMANN)
        voro_tmp.push(WALL_idx(2*dir), (xyz_face[0] - ((dir==dir::x)? dxyz[0]:0.0)), (xyz_face[1] - ((dir==dir::y)? dxyz[1]:0.0)), periodic, xyz_min, xyz_max);
      if(qp.p.piggy3.local_num==-1 && bc_v[dir].wallType(xyz_face)==NEUMANN)
        voro_tmp.push(WALL_idx(2*dir+1), (xyz_face[0] + ((dir==dir::x)? dxyz[0]:0.0)), (xyz_face[1] + ((dir==dir::y)? dxyz[1]:0.0)), periodic, xyz_min, xyz_max);
      if (((qm.p.piggy3.local_num==-1) || is_quad_Wall(p4est_n, qm.p.piggy3.which_tree, &qm, 2*other_cartesian_dir)) || ((qp.p.piggy3.local_num==-1) || is_quad_Wall(p4est_n, qp.p.piggy3.which_tree, &qp, 2*other_cartesian_dir)))
        voro_tmp.push(WALL_idx(2*other_cartesian_dir), (xyz_face[0] - ((other_cartesian_dir==dir::x)? dxyz[0]:0.0)), (xyz_face[1] - ((other_cartesian_dir==dir::y)? dxyz[1]:0.0)), periodic, xyz_min, xyz_max);
      if (((qm.p.piggy3.local_num==-1) || is_quad_Wall(p4est_n, qm.p.piggy3.which_tree, &qm, 2*other_cartesian_dir+1)) || ((qp.p.piggy3.local_num==-1) || is_quad_Wall(p4est_n, qp.p.piggy3.which_tree, &qp, 2*other_cartesian_dir+1)))
        voro_tmp.push(WALL_idx(2*other_cartesian_dir+1), (xyz_face[0] + ((other_cartesian_dir==dir::x)? dxyz[0]:0.0)), (xyz_face[1] + ((other_cartesian_dir==dir::y)? dxyz[1]:0.0)), periodic, xyz_min, xyz_max);
#endif

      /* gather neighbor cells:
       * find neighbor quadrants of touching quadrants in (all possible) transverse orientations + one more layer of such in the positive and negative face-normals
       */
      vector<p4est_quadrant_t> ngbd(0);
      std::set<p4est_locidx_t> set_of_faces; set_of_faces.clear();
      // we add faces to the set as we find them and clear the list of neighbor quadrants after every search to avoid vectors growing very large and slowing down the O(n) searches implemented in find_neighbor_cells_of_cell
      for (char face_touch = -1; face_touch < 2; face_touch+=2) {
        p4est_locidx_t quadrant_touch_idx = ((face_touch==-1)? qm.p.piggy3.local_num : qp.p.piggy3.local_num);
        if(quadrant_touch_idx!=-1)
        {
          p4est_topidx_t tree_touch_idx = ((face_touch==-1)? qm.p.piggy3.which_tree : qp.p.piggy3.which_tree);
          p4est_quadrant_t* quad_touch = ((face_touch==-1)? &qm: &qp);
          const unsigned char dir_touch = 2*dir + ((face_touch==-1)? 0: 1);
          ngbd.push_back(*quad_touch);

          // in face normal direction if needed
          if(faces_n->q2f(quadrant_touch_idx, dir_touch)==NO_VELOCITY)
            ngbd_c->find_neighbor_cells_of_cell(ngbd, quadrant_touch_idx, tree_touch_idx, dir_touch);
          add_faces_to_set_and_clear_vector_of_quad(face_idx, dir, set_of_faces, ngbd);
          // in all tranverse cartesian directions
          for (unsigned char k = 0; k < 2*(P4EST_DIM-1); ++k)
            add_faces_to_set_and_clear_vector_of_quad(face_idx, dir, set_of_faces, ((face_touch==-1)? ngbd_m_[k]: ngbd_p_[k]));
          char search[P4EST_DIM];
#ifdef P4_TO_P8
          // in all transverse "diagonal directions"
          search[dir] = 0;
          for (char iii = -1; iii < 2; iii+=2) {
            search[(dir+1)%P4EST_DIM] = iii;
            for (char jjj = -1; jjj < 2; jjj+=2) {
              search[(dir+2)%P4EST_DIM] = iii;
              ngbd_c->find_neighbor_cells_of_cell(ngbd, quadrant_touch_idx, tree_touch_idx, search[0], search[1], search[2]);
              add_faces_to_set_and_clear_vector_of_quad(face_idx, dir, set_of_faces, ngbd);
            }
          }
#endif
          // extra layer
          search[dir] = face_touch;
          for (char iii = -1; iii < 2; ++iii)
          {
            search[(dir+1)%P4EST_DIM] = iii;
#ifdef P4_TO_P8
            for (char jjj = -1; jjj < 2; ++jjj)
            {
              if((iii==0) && (jjj == 0))
                continue;
              search[(dir+2)%P4EST_DIM] = iii;
              ngbd_c->find_neighbor_cells_of_cell(ngbd, quadrant_touch_idx, tree_touch_idx, search[0], search[1], search[2]);
              add_faces_to_set_and_clear_vector_of_quad(face_idx, dir, set_of_faces, ngbd);
            }
#else
            if(iii == 0)
              continue;
            ngbd_c->find_neighbor_cells_of_cell(ngbd, quadrant_touch_idx, tree_touch_idx, search[0], search[1]);
            add_faces_to_set_and_clear_vector_of_quad(face_idx, dir, set_of_faces, ngbd);
#endif
          }
        }
      }
      voro_tmp.assemble_from_set_of_faces(dir, set_of_faces, faces_n, periodic, xyz_min, xyz_max);

#ifdef P4_TO_P8
      voro_tmp.construct_partition(xyz_min, xyz_max, periodic);
#else
      voro_tmp.construct_partition();
      voro_tmp.compute_volume();
#endif
    }
  }

  ierr = PetscLogEventEnd(log_my_p4est_two_phase_flow_compute_voronoi_cell, 0, 0, 0, 0); CHKERRXX(ierr);
  return voro_tmp;
}


void my_p4est_two_phase_flows_t::sharp_derivative_of_face_field(sharp_derivative& one_sided_derivative,
                                                                const p4est_locidx_t& face_idx, const bool& face_is_in_omega_minus, const p4est_locidx_t& fine_idx_of_face, const uniform_face_ngbd* face_neighbors,
                                                                const double *fine_phi_p, const double *fine_phi_xxyyzz_p[P4EST_DIM],
                                                                const unsigned char &der, const unsigned char &dir,
                                                                const double *vn_dir_minus_p, const double *vn_dir_plus_p, const double *jump_mu_grad_vdir_p[P4EST_DIM],
                                                                const p4est_quadrant& qm, const p4est_quadrant& qp,
                                                                const double* mass_flux_p, const double* normal_dir_p)
{
  p4est_locidx_t fine_idx_of_other_face = -1;
  const double* vn_dir_this_side_p  = (face_is_in_omega_minus? vn_dir_minus_p : vn_dir_plus_p );
  const double* vn_dir_other_side_p = (face_is_in_omega_minus? vn_dir_plus_p  : vn_dir_minus_p);
  const bool other_face_is_in_omega_minus = is_face_in_omega_minus(face_neighbors->neighbor_face_idx[der], dir, fine_phi_p, fine_idx_of_other_face);
  if((face_is_in_omega_minus && !other_face_is_in_omega_minus) || (!face_is_in_omega_minus && other_face_is_in_omega_minus))
  {
    P4EST_ASSERT(fine_idx_of_face != -1);
    P4EST_ASSERT(fine_idx_of_other_face!=-1);
    P4EST_ASSERT((qm.p.piggy3.local_num!=-1) && (qp.p.piggy3.local_num!=-1));
    p4est_locidx_t fine_intermediary_node = -1;
    if(der==2*dir)
    {
#ifdef P4_TO_P8
      get_fine_node_idx_of_logical_vertex(qm.p.piggy3.local_num, qm.p.piggy3.which_tree, 0, 0, 0, fine_intermediary_node, &qm);
#else
      get_fine_node_idx_of_logical_vertex(qm.p.piggy3.local_num, qm.p.piggy3.which_tree, 0, 0, fine_intermediary_node, &qm);
#endif
    }
    else if(der==2*dir+1)
#ifdef P4_TO_P8
      get_fine_node_idx_of_logical_vertex(qp.p.piggy3.local_num, qp.p.piggy3.which_tree, 0, 0, 0, fine_intermediary_node, &qp);
#else
      get_fine_node_idx_of_logical_vertex(qp.p.piggy3.local_num, qp.p.piggy3.which_tree, 0, 0, fine_intermediary_node, &qp);
#endif
    else
      get_fine_node_idx_of_logical_vertex(qp.p.piggy3.local_num, qp.p.piggy3.which_tree,
                                          ((dir==dir::x)?-1:((der/2==dir::x)? (2*(der%2)-1):0)),
                                          ((dir==dir::y)?-1:((der/2==dir::y)? (2*(der%2)-1):0)),
                                    #ifdef P4_TO_P8
                                          ((dir==dir::z)?-1:((der/2==dir::z)? (2*(der%2)-1):0)),
                                    #endif
                                          fine_intermediary_node, &qp);
    P4EST_ASSERT(fine_intermediary_node != -1);
    P4EST_ASSERT((mass_flux_p == NULL) || ((mass_flux_p != NULL) && (normal_dir_p != NULL)));

    const double dsub = dxyz_min[der/2]*0.5;
    double theta;
    double jump_flux_comp_across;
    double jump_v_dir;
    const double phi_intermediary = fine_phi_p[fine_intermediary_node];
    const double phi_face         = fine_phi_p[fine_idx_of_face];
    const double phi_other_face   = fine_phi_p[fine_idx_of_other_face];
    if(signs_of_phi_are_different(phi_face, phi_intermediary))
    {
      P4EST_ASSERT(!signs_of_phi_are_different(phi_intermediary, phi_other_face));
      if(fine_phi_xxyyzz_p[der/2]!=NULL)
        theta = fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(phi_face, phi_intermediary, fine_phi_xxyyzz_p[der/2][fine_idx_of_face], fine_phi_xxyyzz_p[der/2][fine_intermediary_node], dsub);
      else
        theta = fraction_Interval_Covered_By_Irregular_Domain(phi_face, phi_intermediary, dsub, dsub);
      theta = ((phi_face>0.0)? (1.0-theta):theta);
      theta = ((theta < EPS)?0.0:MIN(theta, 1.0));
      jump_flux_comp_across = (1.0-theta)*jump_mu_grad_vdir_p[der/2][fine_idx_of_face] + theta*jump_mu_grad_vdir_p[der/2][fine_intermediary_node];
      jump_v_dir            = ((mass_flux_p == NULL)? 0.0 : (((1.0-theta)*mass_flux_p[fine_idx_of_face]*normal_dir_p[fine_idx_of_face] + theta*mass_flux_p[fine_intermediary_node]*normal_dir_p[fine_intermediary_node])*inverse_mass_density_jump()));
      theta *= 0.5;
    }
    else
    {
      P4EST_ASSERT(signs_of_phi_are_different(phi_intermediary, phi_other_face));
      if(fine_phi_xxyyzz_p[der/2]!=NULL)
        theta = fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(phi_intermediary, phi_other_face, fine_phi_xxyyzz_p[der/2][fine_intermediary_node], fine_phi_xxyyzz_p[der/2][fine_idx_of_other_face], dsub);
      else
        theta = fraction_Interval_Covered_By_Irregular_Domain(phi_intermediary, phi_other_face, dsub, dsub);
      theta = ((phi_intermediary>0.0)? (1.0-theta):theta);
      theta = ((theta < EPS)?0.0:MIN(theta, 1.0));
      jump_flux_comp_across = (1.0-theta)*jump_mu_grad_vdir_p[der/2][fine_intermediary_node] + theta*jump_mu_grad_vdir_p[der/2][fine_idx_of_other_face];
      jump_v_dir            = ((mass_flux_p == NULL)? 0.0 : (((1.0-theta)*mass_flux_p[fine_intermediary_node]*normal_dir_p[fine_intermediary_node] + theta*mass_flux_p[fine_idx_of_other_face]*normal_dir_p[fine_idx_of_other_face])*inverse_mass_density_jump()));
      theta = 0.5*(1.0+theta);
    }
    P4EST_ASSERT((theta >=0.0) && (theta <=1.0));

    double mu_tilde   = (1-theta)*(face_is_in_omega_minus? mu_minus:mu_plus) + theta*(face_is_in_omega_minus? mu_plus:mu_minus);
    double mu_across  = (face_is_in_omega_minus? mu_plus : mu_minus);

    P4EST_ASSERT((fabs(vn_dir_this_side_p[face_idx]) < threshold_dbl_max) && (fabs(vn_dir_other_side_p[face_neighbors->neighbor_face_idx[der]]) < threshold_dbl_max));
    one_sided_derivative.derivative =
        ((der%2==1)?+1.0:-1.0)*(mu_across/mu_tilde)*(vn_dir_other_side_p[face_neighbors->neighbor_face_idx[der]] - vn_dir_this_side_p[face_idx] + (face_is_in_omega_minus? -1.0: +1.0)*jump_v_dir)/dxyz_min[der/2]
        + (face_is_in_omega_minus? -1.0: +1.0)*(1.0-theta)*jump_flux_comp_across/mu_tilde;
    one_sided_derivative.theta = theta;
    one_sided_derivative.xgfm  = true;
  }
  else
  {
    one_sided_derivative.derivative = ((der%2==1)?+1.0:-1.0)*(vn_dir_this_side_p[face_neighbors->neighbor_face_idx[der]] - vn_dir_this_side_p[face_idx])/dxyz_min[der/2];
    one_sided_derivative.theta = 1.0;
    one_sided_derivative.xgfm  = false;
  }
  return;
}

void my_p4est_two_phase_flows_t::get_velocity_seen_from_face(neighbor_value& neighbor_velocity, const p4est_locidx_t& face_idx, const uniform_face_ngbd* face_neighbors,
                                                             const double *fine_phi_p, const double *fine_phi_xxyyzz_p[P4EST_DIM],
                                                             const unsigned char &der, const unsigned char &dir,
                                                             const double *vn_dir_minus_p, const double *vn_dir_plus_p, const double *jump_mu_grad_vdir_p[P4EST_DIM],
                                                             const double* mass_flux_p, const double* normal_dir_p)
{
  p4est_locidx_t fine_idx_of_other_face = -1;
  p4est_locidx_t fine_idx_of_face=-1;
  const bool face_is_in_omega_minus = is_face_in_omega_minus(face_idx, dir, fine_phi_p, fine_idx_of_face);
  const double* vn_dir_this_side_p  = (face_is_in_omega_minus? vn_dir_minus_p : vn_dir_plus_p );
  const double* vn_dir_other_side_p = (face_is_in_omega_minus? vn_dir_plus_p  : vn_dir_minus_p);
  const bool other_face_is_in_omega_minus = is_face_in_omega_minus(face_neighbors->neighbor_face_idx[der], dir, fine_phi_p, fine_idx_of_other_face);
  if((face_is_in_omega_minus && !other_face_is_in_omega_minus) || (!face_is_in_omega_minus && other_face_is_in_omega_minus))
  {
    P4EST_ASSERT(fine_idx_of_face != -1);
    P4EST_ASSERT(fine_idx_of_other_face!=-1);
    p4est_quadrant_t qm, qp;
    faces_n->find_quads_touching_face(face_idx, dir, qm, qp);
    P4EST_ASSERT((qm.p.piggy3.local_num!=-1) && (qp.p.piggy3.local_num!=-1));
    p4est_locidx_t fine_intermediary_node = -1;
    if(der==2*dir)
    {
#ifdef P4_TO_P8
      get_fine_node_idx_of_logical_vertex(qm.p.piggy3.local_num, qm.p.piggy3.which_tree, 0, 0, 0, fine_intermediary_node, &qm);
#else
      get_fine_node_idx_of_logical_vertex(qm.p.piggy3.local_num, qm.p.piggy3.which_tree, 0, 0, fine_intermediary_node, &qm);
#endif
    }
    else if(der==2*dir+1)
#ifdef P4_TO_P8
      get_fine_node_idx_of_logical_vertex(qp.p.piggy3.local_num, qp.p.piggy3.which_tree, 0, 0, 0, fine_intermediary_node, &qp);
#else
      get_fine_node_idx_of_logical_vertex(qp.p.piggy3.local_num, qp.p.piggy3.which_tree, 0, 0, fine_intermediary_node, &qp);
#endif
    else
      get_fine_node_idx_of_logical_vertex(qp.p.piggy3.local_num, qp.p.piggy3.which_tree,
                                          ((dir==dir::x)?-1:((der/2==dir::x)? (2*(der%2)-1):0)),
                                          ((dir==dir::y)?-1:((der/2==dir::y)? (2*(der%2)-1):0)),
                                    #ifdef P4_TO_P8
                                          ((dir==dir::z)?-1:((der/2==dir::z)? (2*(der%2)-1):0)),
                                    #endif
                                          fine_intermediary_node, &qp);
    P4EST_ASSERT(fine_intermediary_node != -1);
    P4EST_ASSERT((mass_flux_p == NULL) || ((mass_flux_p != NULL) && (normal_dir_p != NULL)));

    const double dsub = dxyz_min[der/2]*0.5;
    double theta;
    double jump_flux_comp_across;
    double jump_v_dir;
    const double phi_intermediary = fine_phi_p[fine_intermediary_node];
    const double phi_face         = fine_phi_p[fine_idx_of_face];
    const double phi_other_face   = fine_phi_p[fine_idx_of_other_face];
    if(signs_of_phi_are_different(phi_face, phi_intermediary))
    {
      P4EST_ASSERT(!signs_of_phi_are_different(phi_intermediary, phi_other_face));
      if(fine_phi_xxyyzz_p[der/2]!=NULL)
        theta = fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(phi_face, phi_intermediary, fine_phi_xxyyzz_p[der/2][fine_idx_of_face], fine_phi_xxyyzz_p[der/2][fine_intermediary_node], dsub);
      else
        theta = fraction_Interval_Covered_By_Irregular_Domain(phi_face, phi_intermediary, dsub, dsub);
      theta = ((phi_face>0.0)? (1.0-theta):theta);
      theta = ((theta < EPS)?0.0:MIN(theta, 1.0));
      jump_flux_comp_across = (1.0-theta)*jump_mu_grad_vdir_p[der/2][fine_idx_of_face] + theta*jump_mu_grad_vdir_p[der/2][fine_intermediary_node];
      jump_v_dir            = ((mass_flux_p == NULL)? 0.0 : (((1.0-theta)*mass_flux_p[fine_idx_of_face]*normal_dir_p[fine_idx_of_face] + theta*mass_flux_p[fine_intermediary_node]*normal_dir_p[fine_intermediary_node])*inverse_mass_density_jump()));
      theta *= 0.5;
    }
    else
    {
      P4EST_ASSERT(signs_of_phi_are_different(phi_intermediary, phi_other_face));
      if(fine_phi_xxyyzz_p[der/2]!=NULL)
        theta = fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(phi_intermediary, phi_other_face, fine_phi_xxyyzz_p[der/2][fine_intermediary_node], fine_phi_xxyyzz_p[der/2][fine_idx_of_other_face], dsub);
      else
        theta = fraction_Interval_Covered_By_Irregular_Domain(phi_intermediary, phi_other_face, dsub, dsub);
      theta = ((phi_intermediary>0.0)? (1.0-theta):theta);
      theta = ((theta < EPS)?0.0:MIN(theta, 1.0));
      jump_flux_comp_across = (1.0-theta)*jump_mu_grad_vdir_p[der/2][fine_intermediary_node] + theta*jump_mu_grad_vdir_p[der/2][fine_idx_of_other_face];
      jump_v_dir            = ((mass_flux_p == NULL)? 0.0 : (((1.0-theta)*mass_flux_p[fine_intermediary_node]*normal_dir_p[fine_intermediary_node] + theta*mass_flux_p[fine_idx_of_other_face]*normal_dir_p[fine_idx_of_other_face])*inverse_mass_density_jump()));
      theta = 0.5*(1.0+theta);
    }
    P4EST_ASSERT((theta >=0.0) && (theta <=1.0));

    double mu_tilde     = (1-theta)*(face_is_in_omega_minus? mu_minus : mu_plus) + theta*(face_is_in_omega_minus? mu_plus : mu_minus);
    double mu_across    = (face_is_in_omega_minus? mu_plus  : mu_minus);
    double mu_this_side = (face_is_in_omega_minus? mu_minus : mu_plus);

    P4EST_ASSERT((fabs(vn_dir_this_side_p[face_idx]) < threshold_dbl_max) && (fabs(vn_dir_other_side_p[face_neighbors->neighbor_face_idx[der]]) < threshold_dbl_max));
    neighbor_velocity.value     = ((1.0-theta)*mu_this_side*vn_dir_this_side_p[face_idx] +
                                   theta*(mu_across*(vn_dir_other_side_p[face_neighbors->neighbor_face_idx[der]] + (face_is_in_omega_minus? -1.0: +1.0)*jump_v_dir)
                                   + (((der%2==1)? +1.0:-1.0)*(face_is_in_omega_minus? -1.0:+1.0))*jump_flux_comp_across))/mu_tilde;
    neighbor_velocity.distance  = theta*dxyz_min[der/2];
  }
  else
  {
    neighbor_velocity.value     = vn_dir_this_side_p[face_neighbors->neighbor_face_idx[der]];
    neighbor_velocity.distance  = dxyz_min[der/2];
  }
  return;
}


//void my_p4est_two_phase_flows_t::set_sign_of_phi_faces()
//{
//  PetscErrorCode ierr;
//  double* sign_of_phi_face_p;
//  const double *fine_phi_p;
//  ierr = VecGetArrayRead(fine_phi, &fine_phi_p); CHKERRXX(ierr);
//  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
//    ierr = VecGetArray(sign_of_phi_face[dir], &sign_of_phi_face_p); CHKERRXX(ierr);
//    for (p4est_locidx_t face_idx = 0; face_idx < faces_n->num_local[dir]; ++face_idx) {
//      sign_of_phi_face_p[face_idx] = (is_face_in_omega_minus(face_idx, dir, fine_phi_p)? -1.0:+1.0);
//    }
//    ierr = VecRestoreArray(sign_of_phi_face[dir], &sign_of_phi_face_p); CHKERRXX(ierr);
//    ierr = VecGhostUpdateBegin(sign_of_phi_face[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  }
//  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
//    ierr = VecGhostUpdateEnd(sign_of_phi_face[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  }
//  ierr = VecRestoreArrayRead(fine_phi, &fine_phi_p); CHKERRXX(ierr);
//}


void my_p4est_two_phase_flows_t::compute_velocity_at_nodes()
{
  PetscErrorCode ierr;
  const double *fine_phi_p;
  const double *vnp1_plus_p[P4EST_DIM], *vnp1_minus_p[P4EST_DIM]; //, *sign_of_phi_face_p[P4EST_DIM];
  double *v_nodes_omega_plus_p[P4EST_DIM], *v_nodes_omega_minus_p[P4EST_DIM];


  ierr = VecGetArrayRead(fine_phi, &fine_phi_p); CHKERRXX(ierr);
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = create_p4est_n_node_vector_if_needed(vnp1_nodes_omega_minus[dir]);          CHKERRXX(ierr);
    ierr = create_p4est_n_node_vector_if_needed(vnp1_nodes_omega_plus[dir]);           CHKERRXX(ierr);
    ierr = VecGetArray(vnp1_nodes_omega_minus[dir], &v_nodes_omega_minus_p[dir]);   CHKERRXX(ierr);
    ierr = VecGetArray(vnp1_nodes_omega_plus[dir], &v_nodes_omega_plus_p[dir]);     CHKERRXX(ierr);

    ierr = VecGetArrayRead(vnp1_plus[dir], &vnp1_plus_p[dir]); CHKERRXX(ierr);
    ierr = VecGetArrayRead(vnp1_minus[dir], &vnp1_minus_p[dir]); CHKERRXX(ierr);
    //    ierr = VecGetArrayRead(sign_of_phi_face[dir], &sign_of_phi_face_p[dir]); CHKERRXX(ierr);
  }
  // loop through layer nodes first
  for(size_t i=0; i<ngbd_n->get_layer_size(); ++i)
  {
    p4est_locidx_t node_idx = ngbd_n->get_layer_node(i);
    interpolate_velocity_at_node(node_idx, v_nodes_omega_minus_p, v_nodes_omega_plus_p,
                                 fine_phi_p, vnp1_plus_p, vnp1_minus_p/*, sign_of_phi_face_p*/);
  }
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecGhostUpdateBegin(vnp1_nodes_omega_minus[dir], INSERT_VALUES, SCATTER_FORWARD);  CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(vnp1_nodes_omega_plus[dir], INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);
  }
  /* interpolate vnp1 from faces to nodes */
  for(size_t i=0; i<ngbd_n->get_local_size(); ++i)
  {
    p4est_locidx_t node_idx = ngbd_n->get_local_node(i);
    interpolate_velocity_at_node(node_idx, v_nodes_omega_minus_p, v_nodes_omega_plus_p,
                                 fine_phi_p, vnp1_plus_p, vnp1_minus_p/*, sign_of_phi_face_p*/);
  }
  ierr = VecRestoreArrayRead(fine_phi, &fine_phi_p); CHKERRXX(ierr);
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecGhostUpdateEnd(vnp1_nodes_omega_minus[dir], INSERT_VALUES, SCATTER_FORWARD);  CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(vnp1_nodes_omega_plus[dir], INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);
    ierr = VecRestoreArray(vnp1_nodes_omega_minus[dir], &v_nodes_omega_minus_p[dir]);   CHKERRXX(ierr);
    ierr = VecRestoreArray(vnp1_nodes_omega_plus[dir], &v_nodes_omega_plus_p[dir]);     CHKERRXX(ierr);

    ierr = VecRestoreArrayRead(vnp1_plus[dir], &vnp1_plus_p[dir]); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(vnp1_minus[dir], &vnp1_minus_p[dir]); CHKERRXX(ierr);
    //    ierr = VecRestoreArrayRead(sign_of_phi_face[dir], &sign_of_phi_face_p[dir]); CHKERRXX(ierr);
  }

  //  compute_vorticity();
  //  compute_max_L2_norm_u();
}

void my_p4est_two_phase_flows_t::interpolate_velocity_at_node(const p4est_locidx_t& node_idx, double* v_nodes_omega_plus_p[P4EST_DIM], double* v_nodes_omega_minus_p[P4EST_DIM],
                                                              const double *fine_phi_p, const double *vnp1_plus_p[P4EST_DIM], const double *vnp1_minus_p[P4EST_DIM])
{
  PetscErrorCode ierr;

  double xyz[P4EST_DIM];
  node_xyz_fr_n(node_idx, p4est_n, nodes_n, xyz);
  p4est_indep_t *node = (p4est_indep_t*) sc_array_index(&nodes_n->indep_nodes, node_idx);
#ifdef P4_TO_P8
  const double min_tree_dim = MIN(convert_to_xyz[0], convert_to_xyz[1], convert_to_xyz[2]);
#else
  const double min_tree_dim = MIN(convert_to_xyz[0], convert_to_xyz[1]);
#endif

  vector<bool> velocity_component_is_set(2*P4EST_DIM, false); // P4EST_DIM for minus components + P4EST_DIM for plus components

  if(bc_v!=NULL && is_node_Wall(p4est_n, node))
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      if(bc_v[dir].wallType(xyz)==DIRICHLET)
      {
        v_nodes_omega_minus_p[dir][node_idx]      = bc_v[dir].wallValue(xyz);
        velocity_component_is_set[dir]            = true;
        v_nodes_omega_plus_p[dir][node_idx]       = bc_v[dir].wallValue(xyz);
        velocity_component_is_set[P4EST_DIM+dir]  = true;
      }

  /* gather the neighborhood */
  vector<p4est_quadrant_t> ngbd_tmp;
  set<p4est_locidx_t> set_of_faces[P4EST_DIM];
  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;
  double scaling = DBL_MAX;
#ifdef CASL_THROWS
  bool is_local = false;
#endif

  for (char ii = -1; ii < 2; ii+=2)
    for (char jj = -1; jj < 2; jj+=2)
#ifdef P4_TO_P8
      for (char kk = -1; kk < 2; kk+=2)
      {
        ngbd_n->find_neighbor_cell_of_node(node_idx, ii, jj, kk, quad_idx, tree_idx);
#else
    {
      ngbd_n->find_neighbor_cell_of_node(node_idx, ii, jj, quad_idx, tree_idx);
#endif
      if(quad_idx!=NOT_A_VALID_QUADRANT)
      {
        p4est_quadrant_t quad;
        if(quad_idx<p4est_n->local_num_quadrants)
        {
          p4est_tree_t* tree = p4est_tree_array_index(p4est_n->trees, tree_idx);
          quad = *p4est_quadrant_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);
        }
        else
          quad = *p4est_quadrant_array_index(&ghost_n->ghosts, quad_idx-p4est_n->local_num_quadrants);

        quad.p.piggy3.local_num   = quad_idx;
#ifdef CASL_THROWS
        is_local = is_local || (quad_idx<p4est_n->local_num_quadrants);
#endif

        ngbd_tmp.push_back(quad);
        scaling = MIN(scaling, .5*min_tree_dim*(double)P4EST_QUADRANT_LEN(quad.level)/(double)P4EST_ROOT_LEN);
        add_all_faces_to_sets_and_clear_vector_of_quad(set_of_faces, ngbd_tmp);

        for (char i = 0; abs(i) <= abs(ii); i+=ii)
          for (char j = 0; abs(j) <= abs(jj); j+=jj)
#ifdef P4_TO_P8
            for (char k = 0; abs(k) <= abs(kk); k+=kk)
            {
              if((i == 0) && (j ==0) && (k == 0))
                continue;
              ngbd_c->find_neighbor_cells_of_cell(ngbd_tmp, quad_idx, tree_idx, i, j, k);   add_all_faces_to_sets_and_clear_vector_of_quad(set_of_faces, ngbd_tmp);
            }
#else
          {
            if((i == 0) && (j ==0))
              continue;
            ngbd_c->find_neighbor_cells_of_cell(ngbd_tmp, quad_idx, tree_idx, i, j);   add_all_faces_to_sets_and_clear_vector_of_quad(set_of_faces, ngbd_tmp);
          }
#endif
      }
    };

#ifdef CASL_THROWS
  if(!is_local)
  {
    ierr = PetscPrintf(p4est_n->mpicomm, "Warning !! interpolate_velocity_at_node: the node is not local."); CHKERRXX(ierr);
  }
#endif


  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    for (char sign = 0; sign < 2; ++sign) {
      if(velocity_component_is_set[P4EST_DIM*sign+dir])
        continue;

      const double *face_field_to_interpolate = ((sign == 0)? vnp1_minus_p[dir]: vnp1_plus_p[dir]);
      double *result_field = ((sign == 0)? v_nodes_omega_minus_p[dir]: v_nodes_omega_plus_p[dir]);

      matrix_t        A_lsqr;
      vector<double>  rhs_lsqr;
      std::set<p4est_qcoord_t>  nb[P4EST_DIM];
      unsigned int nb_neumann_walls = 0;
      char neumann_wall[P4EST_DIM];
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
        neumann_wall[dir] = ((bc_v!=NULL && bc_v[dir].wallType(xyz)==NEUMANN) ? (is_node_Wall(p4est_n, node, 2*dir+1)? +1 : (is_node_Wall(p4est_n, node, 2*dir)? -1 : 0)) : 0);
        nb_neumann_walls += abs(neumann_wall[dir]);
      }

      double min_w = 1e-6;
      double inv_max_w = 1e-6;

      int row_idx = 0;
      int col_idx;
      for (std::set<p4est_locidx_t>::const_iterator got_it= set_of_faces[dir].begin(); got_it != set_of_faces[dir].end(); ++got_it)
      {
        p4est_locidx_t neighbor_face = *got_it;
        P4EST_ASSERT((neighbor_face>=0) && (neighbor_face < (faces_n->num_local[dir] + faces_n->num_ghost[dir])));
        row_idx += ((fabs(face_field_to_interpolate[*got_it]) < threshold_dbl_max)? 1 : 0);
      }
      if(row_idx == 0)
      {
        result_field[node_idx] = threshold_dbl_max; // absurd value for undefined interpolated field (e.g. vn_minus far in omega plus --> no well-defined neighbor to be found so move on)
        continue;
      }

      A_lsqr.resize(row_idx, (1+P4EST_DIM+0.5*P4EST_DIM*(P4EST_DIM+1)-nb_neumann_walls));
      rhs_lsqr.resize(row_idx);

      row_idx = 0;
      for (std::set<p4est_locidx_t>::const_iterator got_it= set_of_faces[dir].begin(); got_it != set_of_faces[dir].end(); ++got_it) {
        p4est_locidx_t neighbor_face = *got_it;
        if(fabs(face_field_to_interpolate[*got_it]) > threshold_dbl_max)
          continue;

        double xyz_t[P4EST_DIM];
        faces_n->xyz_fr_f(neighbor_face, dir, xyz_t);

        for(unsigned char i=0; i<P4EST_DIM; ++i)
        {
          double rel_dist = (xyz[i] - xyz_t[i]);
          if(is_periodic(p4est_n, i))
            for (char cc = -1; cc < 2; cc+=2)
              if(fabs(xyz[i] - xyz_t[i] + ((double) cc)*(xyz_max[i]-xyz_min[i])) < fabs(rel_dist))
                rel_dist = (xyz[i] - xyz_t[i] + ((double) cc)*(xyz_max[i]-xyz_min[i]));
          xyz_t[i] = rel_dist / scaling;
        }

#ifdef P4_TO_P8
        double w = MAX(min_w,1./MAX(inv_max_w, sqrt(SQR(xyz_t[0]) + SQR(xyz_t[1]) + SQR(xyz_t[2]))));
#else
        double w = MAX(min_w,1./MAX(inv_max_w, sqrt(SQR(xyz_t[0]) + SQR(xyz_t[1]))));
#endif


        rhs_lsqr[row_idx] = 0.0;
        col_idx = 0;
        A_lsqr.set_value(row_idx,    col_idx++,                                             1                           * w);
        for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
          if(neumann_wall[dir] == 0)
            A_lsqr.set_value(row_idx,  col_idx++,                                           xyz_t[dir]                  * w);
          else
          {
            P4EST_ASSERT(abs(neumann_wall[dir]) == 1);
            rhs_lsqr[row_idx] += ((double) neumann_wall[dir])*bc_v[dir].wallValue(xyz)*     xyz_t[dir]*scaling;
          }
          if(std::find(nb[dir].begin(), nb[dir].end(), xyz_t[dir]) == nb[dir].end())
            nb[dir].insert(xyz_t[dir]);
        }
        for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
          for (unsigned char cross_dir = dir; cross_dir < P4EST_DIM; ++cross_dir)
            A_lsqr.set_value(row_idx, col_idx++,                                            xyz_t[dir]*xyz_t[cross_dir] *w);
        P4EST_ASSERT(col_idx == (1+P4EST_DIM+0.5*P4EST_DIM*(P4EST_DIM+1)-nb_neumann_walls));
        rhs_lsqr[row_idx]   += face_field_to_interpolate[neighbor_face];
        rhs_lsqr[row_idx]   *= w;
        row_idx++;
      }
      P4EST_ASSERT(row_idx == A_lsqr.num_rows());
      A_lsqr.scale_by_maxabs(rhs_lsqr);

#ifdef P4_TO_P8
      result_field[node_idx] = solve_lsqr_system(A_lsqr, rhs_lsqr, nb[0].size(), nb[1].size(), nb[2].size(), 2);
#else
      result_field[node_idx] = solve_lsqr_system(A_lsqr, rhs_lsqr, nb[0].size(), nb[1].size(), 2);
#endif
    }
  }
}

void my_p4est_two_phase_flows_t::initialize_normal_derivative_of_velocity_on_faces(Vec normal_derivative_of_vnp1_minus[P4EST_DIM], Vec normal_derivative_of_vnp1_plus[P4EST_DIM], Vec vnp1_minus [P4EST_DIM], Vec vnp1_plus [P4EST_DIM])
{
  PetscErrorCode ierr;
  my_p4est_interpolation_nodes_t interp_normal(fine_ngbd_n);
  interp_normal.set_input(normal, linear, P4EST_DIM);
  double local_normal[P4EST_DIM], xyz_face[P4EST_DIM];
  sharp_derivative sharp_derivative_p, sharp_derivative_m;
  const uniform_face_ngbd* face_ngbd;
  const double *fine_phi_p, *fine_phi_xxyyzz_p[P4EST_DIM], *vnp1_minus_p[P4EST_DIM], *vnp1_plus_p[P4EST_DIM], *jump_mu_grad_v_p[P4EST_DIM][P4EST_DIM];
  double *normal_derivative_of_vnp1_minus_p[P4EST_DIM], *normal_derivative_of_vnp1_plus_p[P4EST_DIM];
  // initialize data pointers
  ierr = VecGetArrayRead(fine_phi, &fine_phi_p); CHKERRXX(ierr);
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecGetArrayRead(vnp1_minus[dir], &vnp1_minus_p[dir]); CHKERRXX(ierr);
    ierr = VecGetArrayRead(vnp1_plus[dir], &vnp1_plus_p[dir]); CHKERRXX(ierr);
    ierr = VecGetArray(normal_derivative_of_vnp1_minus[dir], &normal_derivative_of_vnp1_minus_p[dir]); CHKERRXX(ierr);
    ierr = VecGetArray(normal_derivative_of_vnp1_plus[dir], &normal_derivative_of_vnp1_plus_p[dir]); CHKERRXX(ierr);
    if(fine_phi_xxyyzz[dir]!=NULL){
      ierr = VecGetArrayRead(fine_phi_xxyyzz[dir], &fine_phi_xxyyzz_p[dir]); CHKERRXX(ierr);
    }
    else
      fine_phi_xxyyzz_p[dir] = NULL;
    for (unsigned char der = 0; der < P4EST_DIM; ++der) {
      ierr = VecGetArrayRead(jump_mu_grad_v[dir][der], &jump_mu_grad_v_p[dir][der]); CHKERRXX(ierr);
    }
  }

  // do the calculations for the layer faces
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    for (size_t k = 0; k < faces_n->get_layer_size(dir); ++k) {
      p4est_locidx_t local_face_idx = faces_n->get_layer_face(dir, k);
      if(faces_n->found_uniform_face_neighborhood(local_face_idx, dir, face_ngbd))
      {
        p4est_quadrant_t qm, qp;
        faces_n->find_quads_touching_face(local_face_idx, dir, qm, qp);
        p4est_locidx_t fine_idx_of_face;
        const bool face_is_in_omega_minus = is_face_in_omega_minus(local_face_idx, dir, fine_phi_p, fine_idx_of_face);
        faces_n->xyz_fr_f(local_face_idx, dir, xyz_face);
        interp_normal(xyz_face, local_normal);
        double value = 0.0;
        for (unsigned char der = 0; der < P4EST_DIM; ++der) {
          sharp_derivative_of_face_field(sharp_derivative_p, local_face_idx, face_is_in_omega_minus, fine_idx_of_face, face_ngbd,
                                         fine_phi_p, fine_phi_xxyyzz_p, 2*der+1, dir,
                                         vnp1_minus_p[dir], vnp1_plus_p[dir], jump_mu_grad_v_p[dir], qm, qp, NULL, NULL);
          sharp_derivative_of_face_field(sharp_derivative_m, local_face_idx, face_is_in_omega_minus, fine_idx_of_face, face_ngbd,
                                         fine_phi_p, fine_phi_xxyyzz_p, 2*der, dir,
                                         vnp1_minus_p[dir], vnp1_plus_p[dir], jump_mu_grad_v_p[dir], qm, qp, NULL, NULL);
          value += local_normal[der]*(sharp_derivative_m.theta*sharp_derivative_p.derivative + sharp_derivative_p.theta*sharp_derivative_m.derivative)/(sharp_derivative_m.theta+sharp_derivative_p.theta);
        }
        if(face_is_in_omega_minus)
        {
          normal_derivative_of_vnp1_minus_p[dir][local_face_idx] = value;
          normal_derivative_of_vnp1_plus_p[dir][local_face_idx] = DBL_MAX;
        }
        else
        {
          normal_derivative_of_vnp1_minus_p[dir][local_face_idx] = DBL_MAX;
          normal_derivative_of_vnp1_plus_p[dir][local_face_idx] = -value; // minus because reverse normal for that field
        }
      }
    }
    // start updating ghost values
    ierr = VecGhostUpdateBegin(normal_derivative_of_vnp1_minus[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(normal_derivative_of_vnp1_plus[dir], INSERT_VALUES, SCATTER_FORWARD);  CHKERRXX(ierr);
  }

  // do the calculations for the local (inner) faces
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    for (size_t k = 0; k < faces_n->get_local_size(dir); ++k) {
      p4est_locidx_t local_face_idx = faces_n->get_local_face(dir, k);
      if(faces_n->found_uniform_face_neighborhood(local_face_idx, dir, face_ngbd))
      {
        p4est_quadrant_t qm, qp;
        faces_n->find_quads_touching_face(local_face_idx, dir, qm, qp);
        p4est_locidx_t fine_idx_of_face;
        const bool face_is_in_omega_minus = is_face_in_omega_minus(local_face_idx, dir, fine_phi_p, fine_idx_of_face);
        faces_n->xyz_fr_f(local_face_idx, dir, xyz_face);
        interp_normal(xyz_face, local_normal);
        double value = 0.0;
        for (unsigned char der = 0; der < P4EST_DIM; ++der) {
          sharp_derivative_of_face_field(sharp_derivative_p, local_face_idx, face_is_in_omega_minus, fine_idx_of_face, face_ngbd,
                                         fine_phi_p, fine_phi_xxyyzz_p, 2*der+1, dir,
                                         vnp1_minus_p[dir], vnp1_plus_p[dir], jump_mu_grad_v_p[dir], qm, qp, NULL, NULL);
          sharp_derivative_of_face_field(sharp_derivative_m, local_face_idx, face_is_in_omega_minus, fine_idx_of_face, face_ngbd,
                                         fine_phi_p, fine_phi_xxyyzz_p, 2*der, dir,
                                         vnp1_minus_p[dir], vnp1_plus_p[dir], jump_mu_grad_v_p[dir], qm, qp, NULL, NULL);
          value += local_normal[der]*(sharp_derivative_m.theta*sharp_derivative_p.derivative + sharp_derivative_p.theta*sharp_derivative_m.derivative)/(sharp_derivative_m.theta+sharp_derivative_p.theta);
        }
        if(face_is_in_omega_minus)
        {
          normal_derivative_of_vnp1_minus_p[dir][local_face_idx] = value;
          normal_derivative_of_vnp1_plus_p[dir][local_face_idx] = DBL_MAX;
        }
        else
        {
          normal_derivative_of_vnp1_minus_p[dir][local_face_idx] = DBL_MAX;
          normal_derivative_of_vnp1_plus_p[dir][local_face_idx] = -value; // minus because reverse normal for that field
        }
      }
    }
    // finish updating ghost values
    ierr = VecGhostUpdateEnd(normal_derivative_of_vnp1_minus[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(normal_derivative_of_vnp1_plus[dir], INSERT_VALUES, SCATTER_FORWARD);  CHKERRXX(ierr);
  }

  // restore data pointers
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecRestoreArrayRead(vnp1_minus[dir], &vnp1_minus_p[dir]); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(vnp1_plus[dir], &vnp1_plus_p[dir]); CHKERRXX(ierr);
    ierr = VecRestoreArray(normal_derivative_of_vnp1_minus[dir], &normal_derivative_of_vnp1_minus_p[dir]); CHKERRXX(ierr);
    ierr = VecRestoreArray(normal_derivative_of_vnp1_plus[dir], &normal_derivative_of_vnp1_plus_p[dir]); CHKERRXX(ierr);
    if(fine_phi_xxyyzz_p[dir]!=NULL){
      ierr = VecRestoreArrayRead(fine_phi_xxyyzz[dir], &fine_phi_xxyyzz_p[dir]); CHKERRXX(ierr);
    }
    for (unsigned char der = 0; der < P4EST_DIM; ++der) {
      ierr = VecRestoreArrayRead(jump_mu_grad_v[dir][der], &jump_mu_grad_v_p[dir][der]); CHKERRXX(ierr);
    }
  }
  ierr = VecRestoreArrayRead(fine_phi, &fine_phi_p); CHKERRXX(ierr);
}

void my_p4est_two_phase_flows_t::extrapolate_velocities_across_interface_in_finest_computational_cells_Aslam_PDE(const unsigned int& n_iterations)
{
  P4EST_ASSERT(n_iterations>0);
  PetscErrorCode ierr;
  my_p4est_interpolation_nodes_t interp_normal(fine_ngbd_n);
  interp_normal.set_input(normal, linear, P4EST_DIM);
  Vec normal_derivative_of_vnp1_minus[P4EST_DIM], normal_derivative_of_vnp1_plus[P4EST_DIM];
  double *normal_derivative_of_vnp1_minus_p[P4EST_DIM], *normal_derivative_of_vnp1_plus_p[P4EST_DIM], *vnp1_minus_p[P4EST_DIM], *vnp1_plus_p[P4EST_DIM];
  const double *fine_phi_p, *fine_phi_xxyyzz_p[P4EST_DIM], *jump_mu_grad_v_p[P4EST_DIM][P4EST_DIM];
  // initialize normal derivatives of the P4EST_DIM face-sampled fields
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecCreateGhostFaces(p4est_n, faces_n, &normal_derivative_of_vnp1_minus[dir], dir); CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est_n, faces_n, &normal_derivative_of_vnp1_plus[dir], dir); CHKERRXX(ierr);
  }
  initialize_normal_derivative_of_velocity_on_faces(normal_derivative_of_vnp1_minus, normal_derivative_of_vnp1_plus, vnp1_minus, vnp1_plus);
  // get data pointers
  ierr = VecGetArrayRead(fine_phi, &fine_phi_p); CHKERRXX(ierr);
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecGetArray(vnp1_minus[dir], &vnp1_minus_p[dir]); CHKERRXX(ierr);
    ierr = VecGetArray(vnp1_plus[dir], &vnp1_plus_p[dir]); CHKERRXX(ierr);
    ierr = VecGetArray(normal_derivative_of_vnp1_minus[dir], &normal_derivative_of_vnp1_minus_p[dir]); CHKERRXX(ierr);
    ierr = VecGetArray(normal_derivative_of_vnp1_plus[dir], &normal_derivative_of_vnp1_plus_p[dir]); CHKERRXX(ierr);
    if(fine_phi_xxyyzz[dir]!=NULL){
      ierr = VecGetArrayRead(fine_phi_xxyyzz[dir], &fine_phi_xxyyzz_p[dir]); CHKERRXX(ierr);
    }
    else
      fine_phi_xxyyzz_p[dir] = NULL;
    for (unsigned char der = 0; der < P4EST_DIM; ++der) {
      ierr = VecGetArrayRead(jump_mu_grad_v[dir][der], &jump_mu_grad_v_p[dir][der]); CHKERRXX(ierr);
    }
  }

  for (unsigned int iter = 0; iter < n_iterations; ++iter) {
    // local layer faces first
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      for (size_t k = 0; k < faces_n->get_layer_size(dir); ++k) {
        p4est_locidx_t local_face_idx = faces_n->get_layer_face(dir, k);
        solve_velocity_extrapolation_local(local_face_idx, dir, interp_normal,
                                           jump_mu_grad_v_p[dir], fine_phi_p, fine_phi_xxyyzz_p,
                                           vnp1_minus_p, vnp1_plus_p, normal_derivative_of_vnp1_minus_p, normal_derivative_of_vnp1_plus_p);
      }
      ierr = VecGhostUpdateBegin(normal_derivative_of_vnp1_minus[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(normal_derivative_of_vnp1_plus[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(vnp1_minus[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(vnp1_plus[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }

    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      for (size_t k = 0; k < faces_n->get_local_size(dir); ++k) {
        p4est_locidx_t local_face_idx = faces_n->get_local_face(dir, k);
        solve_velocity_extrapolation_local(local_face_idx, dir, interp_normal,
                                           jump_mu_grad_v_p[dir], fine_phi_p, fine_phi_xxyyzz_p,
                                           vnp1_minus_p, vnp1_plus_p, normal_derivative_of_vnp1_minus_p, normal_derivative_of_vnp1_plus_p);
      }
      ierr = VecGhostUpdateEnd(normal_derivative_of_vnp1_minus[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(normal_derivative_of_vnp1_plus[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(vnp1_minus[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(vnp1_plus[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }
  }

  // restore data pointers
  ierr = VecRestoreArrayRead(fine_phi, &fine_phi_p); CHKERRXX(ierr);
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecRestoreArray(vnp1_minus[dir], &vnp1_minus_p[dir]); CHKERRXX(ierr);
    ierr = VecRestoreArray(vnp1_plus[dir], &vnp1_plus_p[dir]); CHKERRXX(ierr);
    ierr = VecRestoreArray(normal_derivative_of_vnp1_minus[dir], &normal_derivative_of_vnp1_minus_p[dir]); CHKERRXX(ierr);
    ierr = VecRestoreArray(normal_derivative_of_vnp1_plus[dir], &normal_derivative_of_vnp1_plus_p[dir]); CHKERRXX(ierr);
    if(fine_phi_xxyyzz_p[dir]!=NULL){
      ierr = VecRestoreArrayRead(fine_phi_xxyyzz[dir], &fine_phi_xxyyzz_p[dir]); CHKERRXX(ierr);
    }
    for (unsigned char der = 0; der < P4EST_DIM; ++der) {
      ierr = VecRestoreArrayRead(jump_mu_grad_v[dir][der], &jump_mu_grad_v_p[dir][der]); CHKERRXX(ierr);
    }
  }
}

void my_p4est_two_phase_flows_t::solve_velocity_extrapolation_local(const p4est_locidx_t& local_face_idx, const unsigned char& dir, const my_p4est_interpolation_nodes_t& interp_normal,
                                                                    const double* jump_mu_grad_vdir_p[P4EST_DIM], const double* fine_phi_p, const double *fine_phi_xxyyzz_p[P4EST_DIM],
                                                                    double* vnp1_minus_p[P4EST_DIM], double* vnp1_plus_p[P4EST_DIM],
                                                                    double* normal_derivative_of_vnp1_minus_p[P4EST_DIM], double* normal_derivative_of_vnp1_plus_p[P4EST_DIM])
{
  const uniform_face_ngbd* face_ngbd;
  if(faces_n->found_uniform_face_neighborhood(local_face_idx, dir, face_ngbd)) // we do it only in the finest locally uniform faces (i.e. close to the interface)
  {
    neighbor_value neighbor_velocity;
    double xyz_face[P4EST_DIM], local_normal[P4EST_DIM];
    faces_n->xyz_fr_f(local_face_idx, dir, xyz_face);
    interp_normal(xyz_face, local_normal);
    double rhs_normal_derivative = 0.0;
    double lhs_coeff_normal_derivative = 0.0;
    double rhs_field = 0.0;
    double lhs_coeff_field = 0.0;
    bool solve_m = ((fabs(vnp1_minus_p[dir][local_face_idx]) > threshold_dbl_max) && (fabs(normal_derivative_of_vnp1_minus_p[dir][local_face_idx]) > threshold_dbl_max));
    bool solve_p = ((fabs(vnp1_plus_p[dir][local_face_idx]) > threshold_dbl_max) && (fabs(normal_derivative_of_vnp1_plus_p[dir][local_face_idx]) > threshold_dbl_max));
    P4EST_ASSERT(!(solve_m && solve_p)); // can't technically solve for both: MUST be exclusive if not both already set!
    bool solve = solve_m || solve_p;
    if(!solve) // already done here --> return!
      return;
    P4EST_ASSERT((solve_m && (fabs(vnp1_minus_p[dir][local_face_idx]) > threshold_dbl_max) && (fabs(normal_derivative_of_vnp1_minus_p[dir][local_face_idx]) > threshold_dbl_max)) ||
                 (solve_p && (fabs(vnp1_plus_p[dir][local_face_idx]) > threshold_dbl_max) && (fabs(normal_derivative_of_vnp1_plus_p[dir][local_face_idx]) > threshold_dbl_max)));
    double* field_p                       = (solve_m? vnp1_minus_p[dir]                       : vnp1_plus_p[dir]);
    double* normal_derivative_of_field_p  = (solve_m? normal_derivative_of_vnp1_minus_p[dir]  : normal_derivative_of_vnp1_plus_p[dir]);
    for (unsigned char der = 0; solve && (der < P4EST_DIM); ++der) {
      double signed_normal_component          = (solve_m?+1.0:-1.0)*local_normal[der];
      const unsigned char local_oriented_der  = ((signed_normal_component > 0.0)? 2*der:2*der+1);
      p4est_locidx_t neighbor_face_idx        = face_ngbd->neighbor_face_idx[local_oriented_der];
      // we solve only if the neighbors are all already set!
      solve = solve && (fabs(field_p[neighbor_face_idx]) < threshold_dbl_max) && (fabs(normal_derivative_of_field_p[neighbor_face_idx]));
      // normal derivatives: always on the grid!
      lhs_coeff_normal_derivative += fabs(signed_normal_component)/dxyz_min[der];
      rhs_normal_derivative       += fabs(signed_normal_component)*normal_derivative_of_field_p[neighbor_face_idx]/dxyz_min[der];
      // field: may be subresolved
      get_velocity_seen_from_face(neighbor_velocity, local_face_idx, face_ngbd, fine_phi_p, fine_phi_xxyyzz_p, local_oriented_der, dir, vnp1_minus_p[dir], vnp1_plus_p[dir], jump_mu_grad_vdir_p, NULL, NULL);
      P4EST_ASSERT(neighbor_velocity.distance >= 0.0);
      if((neighbor_velocity.distance < dxyz_min[der]*(dxyz_min[der]/convert_to_xyz[der])) && (fabs(field_p[local_face_idx]) > threshold_dbl_max)) // "too close" and not set yet --> avoid ill-defined terms due to inverses of small amounts, set the value equal to the direct neighbor
        field_p[local_face_idx] = neighbor_velocity.value;
      else
      {
        lhs_coeff_field             += fabs(signed_normal_component)/neighbor_velocity.distance;
        rhs_field                   += fabs(signed_normal_component)*neighbor_velocity.value/neighbor_velocity.distance;
      }
    }
    if(solve)
    {
      P4EST_ASSERT(fabs(normal_derivative_of_field_p[local_face_idx]) > threshold_dbl_max); // make sure it is not set yet
      normal_derivative_of_field_p[local_face_idx]  = rhs_normal_derivative/lhs_coeff_normal_derivative;
      if(fabs(field_p[local_face_idx]) > threshold_dbl_max) // not set yet, could have been overwritten
        field_p[local_face_idx]                     = (normal_derivative_of_field_p[local_face_idx] + rhs_field)/lhs_coeff_field;
    }
  }
}

