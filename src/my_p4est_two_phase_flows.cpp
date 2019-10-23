#ifdef P4_TO_P8
#include "my_p8est_level_set.h"
#include "my_p8est_two_phase_flows.h"
#include "my_p8est_trajectory_of_point.h"
#include "my_p8est_vtk.h"
#else
#include "my_p4est_level_set.h"
#include "my_p4est_two_phase_flows.h"
#include "my_p4est_trajectory_of_point.h"
#include "my_p4est_vtk.h"
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
extern PetscLogEvent log_my_p4est_two_phase_trajectory_of_all_faces;
extern PetscLogEvent log_my_p4est_two_phase_advect_interface;
extern PetscLogEvent log_my_p4est_two_phase_advect_update;
#endif

void my_p4est_two_phase_flows_t::splitting_criteria_computational_grid_two_phase_t::
tag_quadrant(p4est_t *p4est_np1, p4est_locidx_t quad_idx, p4est_topidx_t tree_idx, p4est_nodes_t* nodes_np1,
             const double *phi_np1_p, const double *vorticities_np1_p)
{
  p4est_tree_t *tree = p4est_tree_array_index(p4est_np1->trees, tree_idx);
  p4est_quadrant_t *quad = p4est_quadrant_array_index(&tree->quadrants, quad_idx-tree->quadrants_offset);

  if (quad->level < min_lvl)
    quad->p.user_int = REFINE_QUADRANT;
  else if (quad->level > max_lvl)
    quad->p.user_int = COARSEN_QUADRANT;
  else
  {
    double inv_quad_scal = 1.0/((double) (1<<quad->level));
    const double quad_diag          = owner->tree_diag*inv_quad_scal;
#ifdef P4_TO_P8
    const double quad_dxyz_max      = MAX(owner->convert_to_xyz[0], owner->convert_to_xyz[1], owner->convert_to_xyz[2])*inv_quad_scal;
    const double smallest_dxyz_max  = MAX(owner->dxyz_min[0], owner->dxyz_min[1], owner->dxyz_min[2]);
#else
    const double quad_dxyz_max      = MAX(owner->convert_to_xyz[0], owner->convert_to_xyz[1])*inv_quad_scal;
    const double smallest_dxyz_max  = MAX(owner->dxyz_min[0], owner->dxyz_min[1]);
#endif

    bool coarsen = (quad->level > min_lvl);
    if(coarsen)
    {
      bool cor_vort = true;
      bool cor_band = true;
      bool cor_intf = true;
      p4est_locidx_t node_idx;

      for (unsigned short k = 0; coarsen && (k < P4EST_CHILDREN); ++k) {
        node_idx    = nodes_np1->local_nodes[P4EST_CHILDREN*quad_idx+k];
        if(phi_np1_p[node_idx] > 0.0){
          cor_vort  = cor_vort && (fabs(vorticities_np1_p[2*node_idx+1]) *2.0*quad_dxyz_max/owner->max_L2_norm_u[1] <= owner->threshold_split_cell);
          cor_band  = cor_band && (fabs(phi_np1_p[node_idx]) >= owner->uniform_band_plus*smallest_dxyz_max);
        }
        else{
          cor_vort  = cor_vort && (fabs(vorticities_np1_p[2*node_idx])   *2.0*quad_dxyz_max/owner->max_L2_norm_u[0] <= owner->threshold_split_cell);
          cor_band  = cor_band && (fabs(phi_np1_p[node_idx]) >= owner->uniform_band_minus*smallest_dxyz_max);
        }
        cor_intf    = cor_intf && (fabs(phi_np1_p[node_idx]) >= lip*2.0*quad_diag);
        coarsen     = (cor_vort && cor_band && cor_intf);
      }
    }


    bool refine = (quad->level < max_lvl);
    if(refine)
    {
      bool ref_vort     = false;
      bool ref_band     = false;
      bool ref_intf     = false;
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
            if ((k==1)&&(j==1)&&(i==1))
#else
            if ((j==1)&&(i==1))
#endif
              continue;
#ifdef P4_TO_P8
            if (((k==0)||(k==2)) && ((j==0)||(j==2)) && ((i==0)||(i==2)))
#else
            if (((j==0)||(j==2)) && ((i==0)||(i==2)))
#endif
            {
              node_found=true;
#ifdef P4_TO_P8
              node_idx = nodes_np1->local_nodes[P4EST_CHILDREN*quad_idx+4*(k/2)+2*(j/2)+(i/2)];
#else
              node_idx = nodes_np1->local_nodes[P4EST_CHILDREN*quad_idx+2*(j/2)+(i/2)];
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
              p4est_node_canonicalize(p4est_np1, tree_idx, &r, &c);
              node_found = index_of_node(&c, nodes_np1, node_idx);
            }
            if(node_found)
            {
              P4EST_ASSERT(node_idx < ((p4est_locidx_t) nodes_np1->indep_nodes.elem_count));
              if(phi_np1_p[node_idx] > 0.0) {
                ref_vort  = ref_vort || (fabs(vorticities_np1_p[2*node_idx+1])*quad_dxyz_max/owner->max_L2_norm_u[1] > owner->threshold_split_cell);
                ref_band  = ref_band || (fabs(phi_np1_p[node_idx]) < owner->uniform_band_plus*smallest_dxyz_max);
              } else {
                ref_vort  = ref_vort || (fabs(vorticities_np1_p[2*node_idx])  *quad_dxyz_max/owner->max_L2_norm_u[0] > owner->threshold_split_cell);
                ref_band  = ref_band || (fabs(phi_np1_p[node_idx]) < owner->uniform_band_minus*smallest_dxyz_max);
              }
              ref_intf    = ref_intf || fabs(phi_np1_p[node_idx]) < lip*quad_diag;
              refine      = (ref_vort || ref_band || ref_intf);
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


bool my_p4est_two_phase_flows_t::splitting_criteria_computational_grid_two_phase_t::
refine_and_coarsen(p4est_t* p4est_np1, p4est_nodes_t* nodes_np1,
                   Vec phi_np1, Vec vorticities_np1)
{
  const double *phi_np1_p, *vorticities_np1_p;
  PetscErrorCode ierr;
  ierr = VecGetArrayRead(phi_np1, &phi_np1_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(vorticities_np1, &vorticities_np1_p); CHKERRXX(ierr);

  /* tag the quadrants that need to be refined or coarsened */
  for (p4est_topidx_t tree_idx = p4est_np1->first_local_tree; tree_idx <= p4est_np1->last_local_tree; ++tree_idx) {
    p4est_tree_t* tree = p4est_tree_array_index(p4est_np1->trees, tree_idx);
    for (size_t q = 0; q < tree->quadrants.elem_count; ++q) {
      p4est_locidx_t quad_idx  = q + tree->quadrants_offset;
      tag_quadrant(p4est_np1, quad_idx, tree_idx, nodes_np1, phi_np1_p, vorticities_np1_p);
    }
  }

  my_p4est_coarsen(p4est_np1,       P4EST_FALSE, splitting_criteria_computational_grid_two_phase_t::coarsen_fn, splitting_criteria_computational_grid_two_phase_t::init_fn);
  my_p4est_refine (p4est_np1,       P4EST_FALSE, splitting_criteria_computational_grid_two_phase_t::refine_fn,  splitting_criteria_computational_grid_two_phase_t::init_fn);

  int is_grid_changed = false;
  for (p4est_topidx_t tree_idx = p4est_np1->first_local_tree; tree_idx <= p4est_np1->last_local_tree; ++tree_idx) {
    p4est_tree_t* tree = p4est_tree_array_index(p4est_np1->trees, tree_idx);
    for (size_t q = 0; q < tree->quadrants.elem_count; ++q) {
      p4est_quadrant_t *quad = p4est_quadrant_array_index(&tree->quadrants, q);
      if (quad->p.user_int == NEW_QUADRANT) {
        is_grid_changed = true;
        goto function_end;
      }
    }
  }

function_end:
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &is_grid_changed, 1, MPI_INT, MPI_LOR, p4est_np1->mpicomm); SC_CHECK_MPI(mpiret);

  ierr = VecRestoreArrayRead(phi_np1, &phi_np1_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(vorticities_np1, &vorticities_np1_p); CHKERRXX(ierr);

  return is_grid_changed;
}




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
  cfl = 0.1; //1.0;
  dt_updated = false;
  max_L2_norm_u[0] = max_L2_norm_u[1] = 0.0;

  sl_order = 2;

  double p = 1.0;
  while (DBL_MAX-p == DBL_MAX) p*=2.0;
  threshold_dbl_max = DBL_MAX -p;

  double *v2c = p4est_n->connectivity->vertices;
  p4est_topidx_t *t2v = p4est_n->connectivity->tree_to_vertex;
  p4est_topidx_t first_tree = 0, last_tree = p4est_n->trees->elem_count-1;
  p4est_topidx_t first_vertex = 0, last_vertex = P4EST_CHILDREN - 1;

  splitting_criteria_t *data = (splitting_criteria_t*)p4est_n->user_pointer;

  tree_diag = 0.0;
  for (int dir=0; dir<P4EST_DIM; dir++)
  {
    xyz_min[dir]  = v2c[3*t2v[P4EST_CHILDREN*first_tree + first_vertex] + dir];
    xyz_max[dir]  = v2c[3*t2v[P4EST_CHILDREN*last_tree  + last_vertex ] + dir];
    periodic[dir] = is_periodic(p4est_n, dir);

    double xyz_tmp = v2c[3*t2v[P4EST_CHILDREN*first_tree + last_vertex] + dir];
    dxyz_min[dir] = (xyz_tmp-xyz_min[dir]) / (1<<data->max_lvl);
    convert_to_xyz[dir] = xyz_tmp-xyz_min[dir];
    tree_diag += SQR(convert_to_xyz[dir]);
  }
  tree_diag = sqrt(tree_diag);

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
  fine_phi                      = NULL;
  fine_curvature                = NULL;
  fine_jump_hodge               = NULL;
  fine_jump_normal_flux_hodge   = NULL;
  fine_mass_flux                = NULL;
  fine_variable_surface_tension = NULL;
  // vector fields
  fine_normal                   = NULL;
  fine_phi_xxyyzz               = NULL;
  // tensor/matrix fields
  fine_jump_mu_grad_v           = NULL;
  // -----------------------------------------------------------------------
  // ----- FIELDS SAMPLED AT NODES OF THE COMPUTATIONAL GRID AT TIME N -----
  // -----------------------------------------------------------------------
  // scalar fields
  vorticities                   = NULL;
  // vector fields
  vnp1_nodes_omega_minus        = NULL;
  vnp1_nodes_omega_plus         = NULL;
  interface_velocity_np1        = NULL;
  vn_nodes_omega_minus          = NULL;
  vn_nodes_omega_plus           = NULL;
  // tensor/matrix fields
  vn_nodes_omega_minus_xxyyzz   = NULL;
  vn_nodes_omega_plus_xxyyzz    = NULL;
  interface_velocity_np1_xxyyzz = NULL;
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
  }
  // -------------------------------------------------------------------------
  // ----- FIELDS SAMPLED AT NODES OF THE COMPUTATIONAL GRID AT TIME NM1 -----
  // -------------------------------------------------------------------------
  // vector fields
  vnm1_nodes_omega_minus        = NULL;
  vnm1_nodes_omega_plus         = NULL;
  interface_velocity_n          = NULL;
  // tensor/matrix fields
  vnm1_nodes_omega_minus_xxyyzz = NULL;
  vnm1_nodes_omega_plus_xxyyzz  = NULL;
  interface_velocity_n_xxyyzz   = NULL;

  // clear backtracing points and computational-to-fine-grid maps
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    xyz_n[dir].clear();
    xyz_nm1[dir].clear();
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
  interp_phi = NULL;
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
  return _parent->bc_pressure->wallValue(x,y,z) * _parent->dt_n / (_parent->BDF_alpha());
#else
  return _parent->bc_pressure->wallValue(x,y) * _parent->dt_n / (_parent->BDF_alpha());
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

void my_p4est_two_phase_flows_t::set_bc(BoundaryConditionsDIM *bc_v, BoundaryConditionsDIM *bc_p)
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
  if(fine_phi!=NULL)                          { ierr = delete_and_nullify_vector(fine_phi);                        CHKERRXX(ierr); }
  if(fine_curvature!=NULL)                    { ierr = delete_and_nullify_vector(fine_curvature);                  CHKERRXX(ierr); }
  if(fine_jump_hodge!=NULL)                   { ierr = delete_and_nullify_vector(fine_jump_hodge);                 CHKERRXX(ierr); }
  if(fine_jump_normal_flux_hodge!=NULL)       { ierr = delete_and_nullify_vector(fine_jump_normal_flux_hodge);     CHKERRXX(ierr); }
  if(fine_mass_flux!=NULL)                    { ierr = delete_and_nullify_vector(fine_mass_flux);                  CHKERRXX(ierr); }
  if(fine_variable_surface_tension!=NULL)     { ierr = delete_and_nullify_vector(fine_variable_surface_tension);   CHKERRXX(ierr); }
  if(fine_normal!=NULL)                       { ierr = delete_and_nullify_vector(fine_normal);                     CHKERRXX(ierr); }
  if(fine_phi_xxyyzz != NULL)                 { ierr = delete_and_nullify_vector(fine_phi_xxyyzz);                 CHKERRXX(ierr); }
  if(fine_jump_mu_grad_v!=NULL)               { ierr = delete_and_nullify_vector(fine_jump_mu_grad_v);             CHKERRXX(ierr); }
  // node-sampled fields on the computational grids n and nm1
  if(vorticities!=NULL)                       { ierr = delete_and_nullify_vector(vorticities);                     CHKERRXX(ierr); }
  if(vnm1_nodes_omega_minus!=NULL)            { ierr = delete_and_nullify_vector(vnm1_nodes_omega_minus);          CHKERRXX(ierr); }
  if(vnm1_nodes_omega_plus!=NULL)             { ierr = delete_and_nullify_vector(vnm1_nodes_omega_plus);           CHKERRXX(ierr); }
  if(vn_nodes_omega_minus!=NULL)              { ierr = delete_and_nullify_vector(vn_nodes_omega_minus);            CHKERRXX(ierr); }
  if(vn_nodes_omega_plus!=NULL)               { ierr = delete_and_nullify_vector(vn_nodes_omega_plus);             CHKERRXX(ierr); }
  if(vnp1_nodes_omega_minus!=NULL)            { ierr = delete_and_nullify_vector(vnp1_nodes_omega_minus);          CHKERRXX(ierr); }
  if(vnp1_nodes_omega_plus!=NULL)             { ierr = delete_and_nullify_vector(vnp1_nodes_omega_plus);           CHKERRXX(ierr); }
  if(interface_velocity_np1!=NULL)            { ierr = delete_and_nullify_vector(interface_velocity_np1);          CHKERRXX(ierr); }
  if(interface_velocity_np1_xxyyzz!=NULL)     { ierr = delete_and_nullify_vector(interface_velocity_np1_xxyyzz);   CHKERRXX(ierr); }
  if(interface_velocity_n!=NULL)              { ierr = delete_and_nullify_vector(interface_velocity_n);            CHKERRXX(ierr); }
  if(interface_velocity_n_xxyyzz!=NULL)       { ierr = delete_and_nullify_vector(interface_velocity_n_xxyyzz);     CHKERRXX(ierr); }
  if(vnm1_nodes_omega_minus_xxyyzz!=NULL)     { ierr = delete_and_nullify_vector(vnm1_nodes_omega_minus_xxyyzz);   CHKERRXX(ierr); }
  if(vnm1_nodes_omega_plus_xxyyzz!=NULL)      { ierr = delete_and_nullify_vector(vnm1_nodes_omega_plus_xxyyzz);    CHKERRXX(ierr); }
  if(vn_nodes_omega_minus_xxyyzz!=NULL)       { ierr = delete_and_nullify_vector(vn_nodes_omega_minus_xxyyzz);     CHKERRXX(ierr); }
  if(vn_nodes_omega_plus_xxyyzz!=NULL)        { ierr = delete_and_nullify_vector(vn_nodes_omega_plus_xxyyzz);      CHKERRXX(ierr); }
  // cell-sampled fields, computational grid n
  if(hodge!=NULL)                             { ierr = delete_and_nullify_vector(hodge);                           CHKERRXX(ierr); }
  // face-sampled fields, computational grid n
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    if(dxyz_hodge[dir]!=NULL)                 { ierr = delete_and_nullify_vector(dxyz_hodge[dir]);                 CHKERRXX(ierr); }
    if(vstar[dir]!=NULL)                      { ierr = delete_and_nullify_vector(vstar[dir]);                      CHKERRXX(ierr); }
    if(vnp1_minus[dir]!=NULL)                 { ierr = delete_and_nullify_vector(vnp1_minus[dir]);                 CHKERRXX(ierr); }
    if(vnp1_plus[dir]!=NULL)                  { ierr = delete_and_nullify_vector(vnp1_plus[dir]);                  CHKERRXX(ierr); }
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

void my_p4est_two_phase_flows_t::compute_dt(const double &min_value_for_umax)
{
  dt_nm1 = dt_n;
  double max_L2_norm_u_overall = MAX(max_L2_norm_u[0], max_L2_norm_u[1], min_value_for_umax);
  dt_n = MIN(1/min_value_for_umax, 1/max_L2_norm_u_overall) * cfl * MIN(DIM(dxyz_min[0], dxyz_min[1], dxyz_min[2]));

  dt_n = MIN(dt_n, 1.0/(MAX(mu_minus/rho_minus, mu_plus/rho_plus)*2.0*(1.0/SQR(dxyz_min[0]) + 1.0/SQR(dxyz_min[1]) ONLY3D(+ 1.0/SQR(dxyz_min[2])))));

  dt_updated = true;
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
  if(fine_phi!=NULL){   ierr = delete_and_nullify_vector(fine_phi);  CHKERRXX(ierr); }
  fine_phi = fine_phi_;
  compute_normals_curvature_and_second_derivatives(set_second_derivatives);
  if(interp_phi!=NULL)
    delete  interp_phi;
  interp_phi = new my_p4est_interpolation_nodes_t(fine_ngbd_n);
  if(set_second_derivatives)
    interp_phi->set_input(fine_phi, fine_phi_xxyyzz, quadratic_non_oscillatory);
  else
    interp_phi->set_input(fine_phi, linear);
}

void my_p4est_two_phase_flows_t::set_node_velocities(CF_DIM* vnm1_omega_minus[P4EST_DIM], CF_DIM* vn_omega_minus[P4EST_DIM], CF_DIM* vnm1_omega_plus[P4EST_DIM], CF_DIM* vn_omega_plus[P4EST_DIM])
{
  PetscErrorCode ierr;
  double xyz_node_p4est_n[P4EST_DIM], xyz_node_p4est_nm1[P4EST_DIM];
  double *vnm1_nodes_omega_minus_p, *vnm1_nodes_omega_plus_p, *vn_nodes_omega_minus_p, *vn_nodes_omega_plus_p;
  ierr = create_node_vector_if_needed(vnm1_nodes_omega_minus, p4est_nm1, nodes_nm1, P4EST_DIM);   CHKERRXX(ierr);
  ierr = create_node_vector_if_needed(vnm1_nodes_omega_plus, p4est_nm1, nodes_nm1, P4EST_DIM);    CHKERRXX(ierr);
  ierr = create_node_vector_if_needed(vn_nodes_omega_minus, p4est_n, nodes_n, P4EST_DIM);         CHKERRXX(ierr);
  ierr = create_node_vector_if_needed(vn_nodes_omega_plus, p4est_n, nodes_n, P4EST_DIM);          CHKERRXX(ierr);
  ierr = VecGetArray(vnm1_nodes_omega_minus, &vnm1_nodes_omega_minus_p);                          CHKERRXX(ierr);
  ierr = VecGetArray(vnm1_nodes_omega_plus, &vnm1_nodes_omega_plus_p);                            CHKERRXX(ierr);
  ierr = VecGetArray(vn_nodes_omega_minus, &vn_nodes_omega_minus_p);                              CHKERRXX(ierr);
  ierr = VecGetArray(vn_nodes_omega_plus, &vn_nodes_omega_plus_p);                                CHKERRXX(ierr);
  for (size_t n = 0; n < MAX(nodes_n->indep_nodes.elem_count, nodes_nm1->indep_nodes.elem_count); ++n) {
    if(n < nodes_n->indep_nodes.elem_count)
      node_xyz_fr_n(n, p4est_n, nodes_n, xyz_node_p4est_n);
    if(n < nodes_nm1->indep_nodes.elem_count)
      node_xyz_fr_n(n, p4est_nm1, nodes_nm1, xyz_node_p4est_nm1);
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      if(n < nodes_n->indep_nodes.elem_count)
      {
        vn_nodes_omega_minus_p[P4EST_DIM*n+dir]     = (*vn_omega_minus[dir])(xyz_node_p4est_n);
        vn_nodes_omega_plus_p[P4EST_DIM*n+dir]      = (*vn_omega_plus[dir])(xyz_node_p4est_n);
      }
      if(n < nodes_nm1->indep_nodes.elem_count)
      {
        vnm1_nodes_omega_minus_p[P4EST_DIM*n+dir]   = (*vnm1_omega_minus[dir])(xyz_node_p4est_nm1);
        vnm1_nodes_omega_plus_p[P4EST_DIM*n+dir]    = (*vnm1_omega_plus[dir])(xyz_node_p4est_nm1);
      }
    }
  }
  ierr = VecRestoreArray(vnm1_nodes_omega_minus, &vnm1_nodes_omega_minus_p);                      CHKERRXX(ierr);
  ierr = VecRestoreArray(vnm1_nodes_omega_plus, &vnm1_nodes_omega_plus_p);                        CHKERRXX(ierr);
  ierr = VecRestoreArray(vn_nodes_omega_minus, &vn_nodes_omega_minus_p);                          CHKERRXX(ierr);
  ierr = VecRestoreArray(vn_nodes_omega_plus, &vn_nodes_omega_plus_p);                            CHKERRXX(ierr);
  compute_second_derivatives_of_n_and_nm1_velocities();
}

//void my_p4est_two_phase_flows_t::set_node_vorticities(CF_DIM* vorticity_minus, CF_DIM* vorticity_plus)
//{
//  PetscErrorCode ierr;
//  ierr = create_node_vector_if_needed(vorticities, p4est_n, nodes_n, 2);  CHKERRXX(ierr);
//  double *vorticities_p;
//  ierr = VecGetArray(vorticities, &vorticities_p);                        CHKERRXX(ierr);
//  double xyz_node[P4EST_DIM];
//  for (size_t n = 0; n < nodes_n->indep_nodes.elem_count; ++n)
//  {
//    node_xyz_fr_n(n, p4est_n, nodes_n, xyz_node);
//    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
//      vorticities_p[2*n]    = (*vorticity_minus)(xyz_node);
//      vorticities_p[2*n+1]  = (*vorticity_plus)(xyz_node);
//    }
//  }
//  ierr = VecRestoreArray(vorticities, &vorticities_p);                    CHKERRXX(ierr);
//}

void my_p4est_two_phase_flows_t::set_face_velocities_np1(CF_DIM* vnp1_omega_minus[P4EST_DIM], CF_DIM* vnp1_omega_plus[P4EST_DIM])
{
  PetscErrorCode ierr;
  double *vnp1_minus_p[P4EST_DIM], *vnp1_plus_p[P4EST_DIM];
  double xyz_face[P4EST_DIM];
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecGetArray(vnp1_minus[dir], &vnp1_minus_p[dir]);  CHKERRXX(ierr);
    ierr = VecGetArray(vnp1_plus[dir],  &vnp1_plus_p[dir]);   CHKERRXX(ierr);
    for (size_t k = 0; k < faces_n->get_layer_size(dir); ++k) {
      p4est_locidx_t face_idx = faces_n->get_layer_face(dir, k);
      faces_n->xyz_fr_f(face_idx, dir, xyz_face);
      if((*interp_phi)(xyz_face) <=0.0)
      {
        vnp1_minus_p[dir][face_idx] = (*vnp1_omega_minus[dir])(xyz_face);
        vnp1_plus_p[dir][face_idx]  = DBL_MAX;
      }
      else
      {
        vnp1_plus_p[dir][face_idx]  = (*vnp1_omega_plus[dir])(xyz_face);
        vnp1_minus_p[dir][face_idx] = DBL_MAX;
      }
    }
  }
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecGhostUpdateBegin(vnp1_minus[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(vnp1_plus[dir],  INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    for (size_t k = 0; k < faces_n->get_local_size(dir); ++k) {
      p4est_locidx_t face_idx = faces_n->get_local_face(dir, k);
      faces_n->xyz_fr_f(face_idx, dir, xyz_face);
      if((*interp_phi)(xyz_face) <=0.0)
      {
        vnp1_minus_p[dir][face_idx] = (*vnp1_omega_minus[dir])(xyz_face);
        vnp1_plus_p[dir][face_idx]  = DBL_MAX;
      }
      else
      {
        vnp1_plus_p[dir][face_idx]  = (*vnp1_omega_plus[dir])(xyz_face);
        vnp1_minus_p[dir][face_idx]  = DBL_MAX;
      }
    }
  }
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecGhostUpdateEnd(vnp1_minus[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(vnp1_plus[dir],  INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecRestoreArray(vnp1_minus[dir], &vnp1_minus_p[dir]);  CHKERRXX(ierr);
    ierr = VecRestoreArray(vnp1_plus[dir],  &vnp1_plus_p[dir]);   CHKERRXX(ierr);
  }
}

void my_p4est_two_phase_flows_t::set_jump_mu_grad_v(CF_DIM* jump_mu_grad_v_op[P4EST_DIM][P4EST_DIM])
{
  PetscErrorCode ierr;
  ierr = create_node_vector_if_needed(fine_jump_mu_grad_v, fine_p4est_n, fine_nodes_n, P4EST_DIM*P4EST_DIM); CHKERRXX(ierr);
  double *fine_jump_mu_grad_v_p;
  ierr = VecGetArray(fine_jump_mu_grad_v, &fine_jump_mu_grad_v_p); CHKERRXX(ierr);
  double xyz_node[P4EST_DIM];
  for (size_t k = 0; k < fine_nodes_n->indep_nodes.elem_count; ++k) {
    node_xyz_fr_n(k, fine_p4est_n, fine_nodes_n, xyz_node);
    for (unsigned char ii = 0; ii < P4EST_DIM; ++ii)
      for (unsigned char jj = 0; jj < P4EST_DIM; ++jj)
        fine_jump_mu_grad_v_p[P4EST_DIM*P4EST_DIM*k+P4EST_DIM*ii+jj] = (*jump_mu_grad_v_op[ii][jj])(xyz_node);
  }
  ierr = VecRestoreArray(fine_jump_mu_grad_v, &fine_jump_mu_grad_v_p); CHKERRXX(ierr);
}


void my_p4est_two_phase_flows_t::compute_second_derivatives_of_n_velocities()
{
  PetscErrorCode ierr;
  ierr = create_node_vector_if_needed(vn_nodes_omega_minus_xxyyzz, p4est_n, nodes_n, P4EST_DIM*P4EST_DIM);        CHKERRXX(ierr);
  ierr = create_node_vector_if_needed(vn_nodes_omega_plus_xxyyzz, p4est_n, nodes_n, P4EST_DIM*P4EST_DIM);         CHKERRXX(ierr);

  Vec fields_to_differentiate[2] = {vn_nodes_omega_minus, vn_nodes_omega_plus};
  Vec second_derivatives[2] = {vn_nodes_omega_minus_xxyyzz, vn_nodes_omega_plus_xxyyzz};
  ngbd_n->second_derivatives_central(fields_to_differentiate, second_derivatives, 2, P4EST_DIM);
}
void my_p4est_two_phase_flows_t::compute_second_derivatives_of_nm1_velocities()
{
  PetscErrorCode ierr;
  ierr = create_node_vector_if_needed(vnm1_nodes_omega_minus_xxyyzz, p4est_nm1, nodes_nm1, P4EST_DIM*P4EST_DIM);  CHKERRXX(ierr);
  ierr = create_node_vector_if_needed(vnm1_nodes_omega_plus_xxyyzz, p4est_nm1, nodes_nm1, P4EST_DIM*P4EST_DIM);   CHKERRXX(ierr);

  Vec fields_to_differentiate[2] = {vnm1_nodes_omega_minus, vnm1_nodes_omega_plus};
  Vec second_derivatives[2] = {vnm1_nodes_omega_minus_xxyyzz, vnm1_nodes_omega_plus_xxyyzz};
  ngbd_nm1->second_derivatives_central(fields_to_differentiate, second_derivatives, 2, P4EST_DIM);
}

void my_p4est_two_phase_flows_t::compute_normals_curvature_and_second_derivatives(const bool& set_second_derivatives)
{
  PetscErrorCode ierr;
  // make sure normals are properly allocated
  // same for second derivatives if desired
  create_node_vector_if_needed(fine_normal, fine_p4est_n, fine_nodes_n, P4EST_DIM);
  if(set_second_derivatives){
    ierr = create_node_vector_if_needed(fine_phi_xxyyzz, fine_p4est_n, fine_nodes_n, P4EST_DIM); CHKERRXX(ierr); }
  else{
    ierr = delete_and_nullify_vector(fine_phi_xxyyzz); CHKERRXX(ierr); }
  // get the normals now, and the second derivatives, if desired
  // get the pointers
  double *fine_normal_p, *fine_phi_xxyyzz_p;
  const double* fine_phi_read_p;
  ierr = VecGetArrayRead(fine_phi, &fine_phi_read_p); CHKERRXX(ierr);
  ierr = VecGetArray(fine_normal, &fine_normal_p); CHKERRXX(ierr);
  if(set_second_derivatives){
    ierr = VecGetArray(fine_phi_xxyyzz, &fine_phi_xxyyzz_p); CHKERRXX(ierr); }
  else
    fine_phi_xxyyzz_p = NULL;

  p4est_locidx_t fine_node_idx;
  const quad_neighbor_nodes_of_node_t* qnnn;
  // get the values for the layer nodes
  for (size_t k = 0; k < fine_ngbd_n->layer_nodes.size(); ++k) {
    fine_node_idx = fine_ngbd_n->get_layer_node(k);
    fine_ngbd_n->get_neighbors(fine_node_idx, qnnn);
    compute_gradient_and_second_derivatives(fine_node_idx, qnnn, fine_phi_read_p, fine_normal_p, fine_phi_xxyyzz_p);
  }
  ierr = VecGhostUpdateBegin(fine_normal, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  if(set_second_derivatives){
    ierr = VecGhostUpdateBegin(fine_phi_xxyyzz, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr); }
  for (size_t k = 0; k < fine_ngbd_n->local_nodes.size(); ++k) {
    fine_node_idx = fine_ngbd_n->get_local_node(k);
    fine_ngbd_n->get_neighbors(fine_node_idx, qnnn);
    compute_gradient_and_second_derivatives(fine_node_idx, qnnn, fine_phi_read_p, fine_normal_p, fine_phi_xxyyzz_p);
  }
  ierr = VecGhostUpdateEnd(fine_normal, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  if(set_second_derivatives){
    ierr = VecGhostUpdateEnd(fine_phi_xxyyzz, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr); }
  ierr = VecRestoreArrayRead(fine_phi, &fine_phi_read_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(fine_normal, &fine_normal_p); CHKERRXX(ierr);
  if(set_second_derivatives){
    ierr = VecRestoreArray(fine_phi_xxyyzz, &fine_phi_xxyyzz_p); CHKERRXX(ierr); }
  compute_curvature();
  normalize_normals();
}

void my_p4est_two_phase_flows_t::compute_curvature()
{
  PetscErrorCode ierr;
  // make sure fine curvature is properly allocated
  ierr = create_node_vector_if_needed(fine_curvature, fine_p4est_n, fine_nodes_n); CHKERRXX(ierr);
  double *fine_curvature_p;
  const double *fine_phi_read_p, *fine_normal_p, *fine_phi_xxyyzz_p;
  ierr = VecGetArray(fine_curvature, &fine_curvature_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(fine_phi, &fine_phi_read_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(fine_normal, &fine_normal_p); CHKERRXX(ierr);
  if(fine_phi_xxyyzz!=NULL){
    ierr = VecGetArrayRead(fine_phi_xxyyzz, &fine_phi_xxyyzz_p); CHKERRXX(ierr);}
  else
    fine_phi_xxyyzz_p = NULL;
  p4est_locidx_t fine_node_idx;
  const quad_neighbor_nodes_of_node_t* qnnn;
  // get the values for the layer nodes
  for (size_t k = 0; k < fine_ngbd_n->layer_nodes.size(); ++k) {
    fine_node_idx = fine_ngbd_n->get_layer_node(k);
    fine_ngbd_n->get_neighbors(fine_node_idx, qnnn);
    compute_local_curvature(fine_node_idx, qnnn, fine_phi_read_p, fine_phi_xxyyzz_p, fine_normal_p, fine_curvature_p);
  }
  ierr = VecGhostUpdateBegin(fine_curvature, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for (size_t k = 0; k < fine_ngbd_n->local_nodes.size(); ++k) {
    fine_node_idx = fine_ngbd_n->get_local_node(k);
    fine_ngbd_n->get_neighbors(fine_node_idx, qnnn);
    compute_local_curvature(fine_node_idx, qnnn, fine_phi_read_p, fine_phi_xxyyzz_p, fine_normal_p, fine_curvature_p);
  }
  ierr = VecGhostUpdateEnd(fine_curvature, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecRestoreArray(fine_curvature, &fine_curvature_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(fine_phi, &fine_phi_read_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(fine_normal, &fine_normal_p); CHKERRXX(ierr);
  if(fine_phi_xxyyzz!=NULL){
    ierr = VecRestoreArrayRead(fine_phi_xxyyzz, &fine_phi_xxyyzz_p); CHKERRXX(ierr);
  }
}

void my_p4est_two_phase_flows_t::normalize_normals()
{
  PetscErrorCode ierr;
  double norm_of_grad;
  double *fine_normal_p;
  ierr = VecGetArray(fine_normal, &fine_normal_p); CHKERRXX(ierr);
  for (unsigned int n = 0; n < fine_nodes_n->indep_nodes.elem_count; ++n)
  {
    norm_of_grad = 0.0;
    for (unsigned int der = 0; der < P4EST_DIM; ++der)
      norm_of_grad += SQR(fine_normal_p[P4EST_DIM*n+der]);
    norm_of_grad = sqrt(norm_of_grad);
    for (unsigned int der = 0; der < P4EST_DIM; ++der)
      fine_normal_p[P4EST_DIM*n+der] = ((norm_of_grad > threshold_norm_of_n)? (fine_normal_p[P4EST_DIM*n+der]/norm_of_grad): 0.0);
  }
  ierr = VecRestoreArray(fine_normal, &fine_normal_p); CHKERRXX(ierr);
}

void my_p4est_two_phase_flows_t::set_dt(double dt_nm1_, double dt_n_)
{
  dt_nm1  = dt_nm1_;
  dt_n    = dt_n_;
}

void my_p4est_two_phase_flows_t::get_velocity_seen_from_cell(neighbor_value& neighbor_velocity, const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const int& face_dir,
                                                             const double *vstar_p, const double *fine_phi_p, const double *fine_phi_xxyyzz_p, const double *fine_jump_mu_grad_v_p)
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
      if(fine_phi_xxyyzz_p!=NULL)
        theta = fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(phi_center, fine_phi_p[fine_face_idx],
                                                                                          fine_phi_xxyyzz_p[P4EST_DIM*fine_center_idx+(face_dir/2)], fine_phi_xxyyzz_p[P4EST_DIM*fine_face_idx+(face_dir/2)], neighbor_velocity.distance);
      else
        theta = fraction_Interval_Covered_By_Irregular_Domain(phi_center, fine_phi_p[fine_face_idx], neighbor_velocity.distance, neighbor_velocity.distance);
      theta = ((phi_center>0.0)? (1.0-theta):theta);
      theta = ((theta < EPS)?0.0:MIN(theta, 1.0));
      neighbor_velocity.distance *= theta;
      p4est_locidx_t opposite_fine_face_idx;
      if(face_is_across(quad_idx, tree_idx, ((face_dir%2==0)? (face_dir+1): (face_dir-1)), fine_phi_p, fine_center_idx, opposite_fine_face_idx, quad))
        throw std::runtime_error("get_velocity_seen_from_cell:: subrefined cell, not implemented yet...");
      double jump_flux_comp_across = (1.0-theta)*fine_jump_mu_grad_v_p[P4EST_DIM*P4EST_DIM*fine_center_idx+P4EST_DIM*(face_dir/2)+(face_dir/2)] + theta*fine_jump_mu_grad_v_p[P4EST_DIM*P4EST_DIM*fine_face_idx+P4EST_DIM*(face_dir/2)+(face_dir/2)];
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
                                                      const double *fine_phi_xxyyzz_p, const double *fine_jump_mu_grad_v_p)
{
  double val = 0;
  neighbor_value nb_p, nb_m;
  for(unsigned char dir=0; dir<P4EST_DIM; ++dir)
  {
    get_velocity_seen_from_cell(nb_p, quad_idx, tree_idx, 2*dir+1,  vstar_p[dir], fine_phi_p, fine_phi_xxyyzz_p, fine_jump_mu_grad_v_p);
    get_velocity_seen_from_cell(nb_m, quad_idx, tree_idx, 2*dir,    vstar_p[dir], fine_phi_p, fine_phi_xxyyzz_p, fine_jump_mu_grad_v_p);
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
  const double *vstar_p[P4EST_DIM];
  const double *fine_phi_p, *fine_jump_mu_grad_v_p, *fine_phi_xxyyzz_p;
  ierr = VecGetArrayRead(fine_jump_mu_grad_v, &fine_jump_mu_grad_v_p); CHKERRXX(ierr);
  if(fine_phi_xxyyzz != NULL){
    ierr = VecGetArrayRead(fine_phi_xxyyzz, &fine_phi_xxyyzz_p); CHKERRXX(ierr); }
  else
    fine_phi_xxyyzz_p = NULL;
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecGetArrayRead(vstar[dir], &vstar_p[dir]); CHKERRXX(ierr);
  }
  ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(fine_phi, &fine_phi_p); CHKERRXX(ierr);
  /* compute the right-hand-side */
  for(p4est_topidx_t tree_idx=p4est_n->first_local_tree; tree_idx<=p4est_n->last_local_tree; ++tree_idx)
  {
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est_n->trees, tree_idx);
    for(size_t q_idx=0; q_idx<tree->quadrants.elem_count; ++q_idx)
    {
      p4est_locidx_t quad_idx = q_idx+tree->quadrants_offset;
      rhs_p[quad_idx] = -compute_divergence(quad_idx, tree_idx, vstar_p, fine_phi_p, fine_phi_xxyyzz_p, fine_jump_mu_grad_v_p);
      P4EST_ASSERT(!ISNAN(rhs_p[quad_idx]));
    }
  }
  ierr = VecRestoreArrayRead(fine_jump_mu_grad_v, &fine_jump_mu_grad_v_p); CHKERRXX(ierr);
  if(fine_phi_xxyyzz != NULL){
    ierr = VecRestoreArrayRead(fine_phi_xxyyzz, &fine_phi_xxyyzz_p ); CHKERRXX(ierr); }
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecRestoreArrayRead(vstar[dir], &vstar_p[dir]); CHKERRXX(ierr); }
  ierr = VecRestoreArrayRead(fine_phi, &fine_phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);

  /* solve the linear system */
  if(cell_poisson_jump_solver == NULL)
  {
    cell_poisson_jump_solver = new my_p4est_xgfm_cells_t(ngbd_c, ngbd_n, fine_ngbd_n, activate_xgfm);

//    const double *fine_phi_p, *fine_phi_xxyyzz_p, *fine_normal_p, *fine_jump_hodge_p, *fine_jump_normal_flux_hodge_p;
//    ierr = VecGetArrayRead(fine_phi, &fine_phi_p); CHKERRXX(ierr);
//    ierr = VecGetArrayRead(fine_phi_xxyyzz, &fine_phi_xxyyzz_p); CHKERRXX(ierr);
//    ierr = VecGetArrayRead(fine_normal, &fine_normal_p); CHKERRXX(ierr);
//    ierr = VecGetArrayRead(fine_jump_hodge, &fine_jump_hodge_p); CHKERRXX(ierr);
//    ierr = VecGetArrayRead(fine_jump_normal_flux_hodge, &fine_jump_normal_flux_hodge_p); CHKERRXX(ierr);
//    for (size_t k = 0; k < fine_nodes_n->indep_nodes.elem_count; ++k) {
//      P4EST_ASSERT(!ISNAN(fine_phi_p[k]));
//      P4EST_ASSERT(!ISNAN(fine_jump_hodge_p[k]));
//      P4EST_ASSERT(!ISNAN(fine_jump_normal_flux_hodge_p[k]));
//      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
//      {
//        P4EST_ASSERT(!ISNAN(fine_phi_xxyyzz_p[P4EST_DIM*k+dir]));
//        P4EST_ASSERT(!ISNAN(fine_normal_p[P4EST_DIM*k+dir]));
//      }
//    }
//    ierr = VecRestoreArrayRead(fine_phi, &fine_phi_p); CHKERRXX(ierr);
//    ierr = VecRestoreArrayRead(fine_phi_xxyyzz, &fine_phi_xxyyzz_p); CHKERRXX(ierr);
//    ierr = VecRestoreArrayRead(fine_normal, &fine_normal_p); CHKERRXX(ierr);
//    ierr = VecRestoreArrayRead(fine_jump_hodge, &fine_jump_hodge_p); CHKERRXX(ierr);
//    ierr = VecRestoreArrayRead(fine_jump_normal_flux_hodge, &fine_jump_normal_flux_hodge_p); CHKERRXX(ierr);

    cell_poisson_jump_solver->set_phi(fine_phi, fine_phi_xxyyzz);
    cell_poisson_jump_solver->set_normals(fine_normal);
    cell_poisson_jump_solver->set_diagonals(0.0, 0.0);
    cell_poisson_jump_solver->set_mus(1.0/rho_minus, 1.0/rho_plus);
    cell_poisson_jump_solver->set_bc(bc_hodge);
    cell_poisson_jump_solver->set_jumps(fine_jump_hodge, fine_jump_normal_flux_hodge);
  }
  cell_poisson_jump_solver->set_rhs(rhs);
  cell_poisson_jump_solver->solve(ksp, pc);

  cell_poisson_jump_solver->get_flux_components_and_subtract_them_from_velocities(dxyz_hodge, faces_n, vstar, vnp1_minus, vnp1_plus);
  if(hodge!=NULL){
    ierr = delete_and_nullify_vector(hodge); CHKERRXX(ierr); }
  hodge = cell_poisson_jump_solver->get_solution();
  ierr = delete_and_nullify_vector(rhs); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_two_phase_flows_projection, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_two_phase_flows_t::compute_jump_mu_grad_v()
{
  PetscErrorCode ierr;
  Vec fine_mass_flux_times_normal = NULL;
  if(fine_mass_flux!=NULL)
  {
    double* fine_mass_flux_times_normal_p;
    const double *fine_normal_p, *fine_mass_flux_p;
    ierr = VecGetArrayRead(fine_mass_flux, &fine_mass_flux_p); CHKERRXX(ierr);
    ierr = create_node_vector_if_needed(fine_mass_flux_times_normal, fine_p4est_n, fine_nodes_n, P4EST_DIM); CHKERRXX(ierr);
    ierr = VecGetArrayRead(fine_normal, &fine_normal_p); CHKERRXX(ierr);
    ierr = VecGetArray(fine_mass_flux_times_normal, &fine_mass_flux_times_normal_p); CHKERRXX(ierr);
    for (size_t k = 0; k < fine_nodes_n->indep_nodes.elem_count; ++k)
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
        fine_mass_flux_times_normal_p[P4EST_DIM*k+dir] = fine_mass_flux_p[k]*fine_normal_p[P4EST_DIM*k+dir];
    ierr = VecRestoreArrayRead(fine_normal, &fine_normal_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(fine_mass_flux_times_normal, &fine_mass_flux_times_normal_p); CHKERRXX(ierr);
  }
  Vec grad_underlined_vn_nodes = NULL;
  ierr = create_node_vector_if_needed(grad_underlined_vn_nodes, p4est_n, nodes_n, P4EST_DIM*P4EST_DIM);
  ngbd_n->first_derivatives_central(((underlined_side(velocity_field) == OMEGA_PLUS)? vn_nodes_omega_plus : vn_nodes_omega_minus), grad_underlined_vn_nodes, P4EST_DIM);
  my_p4est_interpolation_nodes_t interp_grad_underlined_vn_nodes(ngbd_n);
  interp_grad_underlined_vn_nodes.set_input(grad_underlined_vn_nodes, linear, P4EST_DIM*P4EST_DIM);

  const double *fine_curvature_p;
  const double *fine_normal_p;
  const double *fine_mass_flux_p = NULL, *fine_variable_surface_tension_p = NULL;
  const double *fine_mass_flux_times_normal_p = NULL;
  double* fine_jump_mu_grad_v_p;
  const quad_neighbor_nodes_of_node_t* qnnn;

  if(fine_mass_flux!=NULL){
    ierr = VecGetArrayRead(fine_mass_flux, &fine_mass_flux_p); CHKERRXX(ierr); }
  if(fine_variable_surface_tension!=NULL){
    ierr = VecGetArrayRead(fine_variable_surface_tension, &fine_variable_surface_tension_p); CHKERRXX(ierr); }
  ierr = VecGetArrayRead(fine_curvature, &fine_curvature_p); CHKERRXX(ierr);
  ierr = create_node_vector_if_needed(fine_jump_mu_grad_v, fine_p4est_n, fine_nodes_n, P4EST_DIM*P4EST_DIM); CHKERRXX(ierr);
  ierr = VecGetArray(fine_jump_mu_grad_v, &fine_jump_mu_grad_v_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(fine_normal, &fine_normal_p); CHKERRXX(ierr);
  if(fine_mass_flux!=NULL){
    ierr = VecGetArrayRead(fine_mass_flux_times_normal, &fine_mass_flux_times_normal_p); CHKERRXX(ierr); }

  for (unsigned int kk = 0; kk < fine_ngbd_n->get_layer_size(); ++kk) {
    p4est_locidx_t fine_node_idx = fine_ngbd_n->get_layer_node(kk);
    fine_ngbd_n->get_neighbors(fine_node_idx, qnnn);
    compute_local_jump_mu_grad_v_elements(fine_node_idx, qnnn,
                                          interp_grad_underlined_vn_nodes, fine_normal_p,
                                          fine_mass_flux_p, fine_mass_flux_times_normal_p,
                                          fine_variable_surface_tension_p, fine_curvature_p,
                                          fine_jump_mu_grad_v_p);
  }
  ierr = VecGhostUpdateBegin(fine_jump_mu_grad_v, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for (unsigned int kk = 0; kk < fine_ngbd_n->get_local_size(); ++kk) {
    p4est_locidx_t fine_node_idx = fine_ngbd_n->get_local_node(kk);
    fine_ngbd_n->get_neighbors(fine_node_idx, qnnn);
    compute_local_jump_mu_grad_v_elements(fine_node_idx, qnnn,
                                          interp_grad_underlined_vn_nodes, fine_normal_p,
                                          fine_mass_flux_p, fine_mass_flux_times_normal_p,
                                          fine_variable_surface_tension_p, fine_curvature_p,
                                          fine_jump_mu_grad_v_p);
  }
  ierr = VecGhostUpdateEnd(fine_jump_mu_grad_v, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  if(fine_mass_flux!=NULL)
  {
    ierr = VecRestoreArrayRead(fine_mass_flux_times_normal, &fine_mass_flux_times_normal_p); CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(fine_mass_flux_times_normal); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(fine_mass_flux, &fine_mass_flux_p); CHKERRXX(ierr);
  }
  if(fine_variable_surface_tension!=NULL){
    ierr = VecRestoreArrayRead(fine_variable_surface_tension, &fine_variable_surface_tension_p); CHKERRXX(ierr); }

  ierr = VecRestoreArrayRead(fine_curvature, &fine_curvature_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(fine_jump_mu_grad_v, &fine_jump_mu_grad_v_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(fine_normal, &fine_normal_p); CHKERRXX(ierr);

  ierr = delete_and_nullify_vector(grad_underlined_vn_nodes); CHKERRXX(ierr);
}

void my_p4est_two_phase_flows_t::compute_jumps_hodge()
{
  PetscErrorCode ierr;
  ierr = create_node_vector_if_needed(fine_jump_hodge, fine_p4est_n, fine_nodes_n);                   CHKERRXX(ierr);
  ierr = create_node_vector_if_needed(fine_jump_normal_flux_hodge, fine_p4est_n, fine_nodes_n);       CHKERRXX(ierr);


  Vec grad_vn_nodes_minus = NULL;
  Vec grad_vn_nodes_plus  = NULL;
  ierr = create_node_vector_if_needed(grad_vn_nodes_minus,  p4est_n, nodes_n, P4EST_DIM*P4EST_DIM);
  ierr = create_node_vector_if_needed(grad_vn_nodes_plus,   p4est_n, nodes_n, P4EST_DIM*P4EST_DIM);
  Vec inputs[2] = {vn_nodes_omega_minus, vn_nodes_omega_plus};
  Vec outputs[2] = {grad_vn_nodes_minus, grad_vn_nodes_plus};
  ngbd_n->first_derivatives_central(inputs, outputs, 2, P4EST_DIM);
  my_p4est_interpolation_nodes_t interp_grad_underlined_vn_nodes(ngbd_n);
  interp_grad_underlined_vn_nodes.set_input(outputs, linear, 2, P4EST_DIM*P4EST_DIM);
  double local_grad_v[2*P4EST_DIM*P4EST_DIM];

  double xyz_node[P4EST_DIM];
  const double *fine_curvature_p;
  const double *fine_normal_p;
  const double *fine_mass_flux_p = NULL;
  double *fine_jump_hodge_p, *fine_jump_normal_flux_hodge_p;
  const double *fine_phi_p;
  ierr = VecGetArrayRead(fine_phi, &fine_phi_p);                              CHKERRXX(ierr);
  ierr = VecGetArray(fine_jump_hodge, &fine_jump_hodge_p);                              CHKERRXX(ierr);
  ierr = VecGetArray(fine_jump_normal_flux_hodge, &fine_jump_normal_flux_hodge_p);      CHKERRXX(ierr);
  ierr = VecGetArrayRead(fine_curvature, &fine_curvature_p);                            CHKERRXX(ierr);
  ierr = VecGetArrayRead(fine_normal, &fine_normal_p);                                  CHKERRXX(ierr);
  if(fine_mass_flux!=NULL){
    ierr = VecGetArrayRead(fine_mass_flux, &fine_mass_flux_p);                          CHKERRXX(ierr); }
  for (unsigned int k = 0; k < fine_nodes_n->indep_nodes.elem_count; ++k) {
    node_xyz_fr_n(k, fine_p4est_n, fine_nodes_n, xyz_node);
    interp_grad_underlined_vn_nodes(xyz_node, local_grad_v);
    double jump_two_mu_nEn = 0.0;
    size_t position = P4EST_DIM*k;
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      for (unsigned char der = 0; der < P4EST_DIM; ++der)
        jump_two_mu_nEn += 2.0*fine_normal_p[position+der]*(mu_plus*local_grad_v[P4EST_DIM*P4EST_DIM+P4EST_DIM*dir+der]-mu_minus*local_grad_v[P4EST_DIM*dir+der])*fine_normal_p[position+dir];
    fine_jump_hodge_p[k] = (-surface_tension*fine_curvature_p[k] + jump_two_mu_nEn - ((fine_mass_flux_p!=NULL)? (SQR(fine_mass_flux_p[k])*jump_inverse_mass_density()): 0.0))*dt_n/(BDF_alpha());

    if(fabs(fine_phi_p[k]) > 2.0*tree_diag/((double) (1<<(((splitting_criteria_t*) fine_p4est_n->user_pointer)->max_lvl))))
      fine_jump_hodge_p[k] = 0.0;

    if(fabs(fine_phi_p[k]) < 2.0*tree_diag/((double) (1<<(((splitting_criteria_t*) fine_p4est_n->user_pointer)->max_lvl))))
      P4EST_ASSERT(!ISNAN(fine_jump_hodge_p[k]));

    fine_jump_normal_flux_hodge_p[k] = 0.0;
  }

  ierr = VecRestoreArrayRead(fine_phi, &fine_phi_p);                              CHKERRXX(ierr);
  if(fine_mass_flux_p!=NULL){
    ierr = VecRestoreArrayRead(fine_mass_flux, &fine_mass_flux_p);                      CHKERRXX(ierr); }
  ierr = VecRestoreArrayRead(fine_normal, &fine_normal_p);                              CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(fine_curvature, &fine_curvature_p);                        CHKERRXX(ierr);
  ierr = VecRestoreArray(fine_jump_hodge, &fine_jump_hodge_p);                          CHKERRXX(ierr);
  ierr = VecRestoreArray(fine_jump_normal_flux_hodge, &fine_jump_normal_flux_hodge_p);  CHKERRXX(ierr);
}

void my_p4est_two_phase_flows_t::solve_viscosity_explicit()
{
  PetscErrorCode ierr;

  ierr = PetscLogEventBegin(log_my_p4est_two_phase_flows_viscosity, 0, 0, 0, 0); CHKERRXX(ierr);

  double alpha  = BDF_alpha();
  double beta   = BDF_beta();

  const double *fine_phi_p;
  ierr = VecGetArrayRead(fine_phi, &fine_phi_p); CHKERRXX(ierr);

  /* construct the right hand side */

  if(!semi_lagrangian_backtrace_is_done)
  {
    trajectory_from_all_faces_two_phases(p4est_n, faces_n, ngbd_nm1, ngbd_n, fine_phi_p,
                                         vnm1_nodes_omega_minus, vnm1_nodes_omega_minus_xxyyzz,
                                         vnm1_nodes_omega_plus, vnm1_nodes_omega_plus_xxyyzz,
                                         vn_nodes_omega_minus, vn_nodes_omega_minus_xxyyzz,
                                         vn_nodes_omega_plus, vn_nodes_omega_plus_xxyyzz,
                                         dt_nm1, dt_n, xyz_n, ((sl_order ==2)? xyz_nm1 : NULL));
    semi_lagrangian_backtrace_is_done = true;
  }


  const double *fine_phi_xxyyzz_p = NULL;
  const double *fine_jump_mu_grad_v_p = NULL;
  ierr = VecGetArrayRead(fine_jump_mu_grad_v, &fine_jump_mu_grad_v_p); CHKERRXX(ierr);
  if(fine_phi_xxyyzz != NULL){
    ierr = VecGetArrayRead(fine_phi_xxyyzz, &fine_phi_xxyyzz_p); CHKERRXX(ierr); }
  for(unsigned char dir=0; dir<P4EST_DIM; ++dir)
  {
    /* find the velocity at the backtraced points */
    my_p4est_interpolation_nodes_t interp_nm1_omega_minus(ngbd_nm1), interp_nm1_omega_plus(ngbd_nm1);
    my_p4est_interpolation_nodes_t interp_n_omega_minus  (ngbd_n),   interp_n_omega_plus  (ngbd_n);
    for(p4est_locidx_t f_idx=0; f_idx<faces_n->num_local[dir]; ++f_idx)
    {
      const bool face_is_in_omega_minus = is_face_in_omega_minus(f_idx, dir, fine_phi_p);
      if(sl_order==2)
      {
        if(face_is_in_omega_minus)
          interp_nm1_omega_minus.add_point(f_idx, &xyz_nm1[dir][P4EST_DIM*f_idx]);
        else
          interp_nm1_omega_plus.add_point(f_idx, &xyz_nm1[dir][P4EST_DIM*f_idx]);
      }
      if(face_is_in_omega_minus)
        interp_n_omega_minus.add_point(f_idx, &xyz_n[dir][P4EST_DIM*f_idx]);
      else
        interp_n_omega_plus.add_point(f_idx, &xyz_n[dir][P4EST_DIM*f_idx]);
    }

    std::vector<double> vnm1_faces(faces_n->num_local[dir]);
    if(sl_order==2)
    {
      vnm1_faces.resize(faces_n->num_local[dir]);
      interp_nm1_omega_minus.set_input(vnm1_nodes_omega_minus, vnm1_nodes_omega_minus_xxyyzz, quadratic, P4EST_DIM);
      interp_nm1_omega_plus.set_input(vnm1_nodes_omega_plus, vnm1_nodes_omega_plus_xxyyzz, quadratic, P4EST_DIM);
      interp_nm1_omega_minus.interpolate(vnm1_faces.data(), dir);
      interp_nm1_omega_plus.interpolate(vnm1_faces.data(), dir);
    }
    Vec vn_faces;
    double *vn_faces_p;
    ierr = VecCreateGhostFaces(p4est_n, faces_n, &vn_faces, dir); CHKERRXX(ierr);
    ierr = VecGetArray(vn_faces, &vn_faces_p); CHKERRXX(ierr);
    interp_n_omega_minus.set_input(vn_nodes_omega_minus, vn_nodes_omega_minus_xxyyzz, quadratic, P4EST_DIM);
    interp_n_omega_plus.set_input(vn_nodes_omega_plus, vn_nodes_omega_plus_xxyyzz, quadratic, P4EST_DIM);
    interp_n_omega_minus.interpolate(vn_faces_p, dir);
    interp_n_omega_plus.interpolate(vn_faces_p, dir);

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
        vstar_p[f_idx] = + (dt_n/(alpha*(face_is_in_omega_minus?rho_minus:rho_plus)))*div_mu_grad_u_dir(f_idx, dir, face_is_in_omega_minus, fine_idx_of_face, vn_faces_p, fine_jump_mu_grad_v_p, fine_phi_p, fine_phi_xxyyzz_p) + (beta*dt_n/(alpha*dt_nm1))*vnm1_faces[f_idx] + (1.0-(beta*dt_n/(alpha*dt_nm1)))*vn_faces_p[f_idx];
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
    ierr = VecRestoreArray(vstar[dir], &vstar_p);   CHKERRXX(ierr);
    ierr = VecRestoreArray(vn_faces, &vn_faces_p);  CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(vn_faces);     CHKERRXX(ierr);
  }
  ierr = VecRestoreArrayRead(fine_jump_mu_grad_v, &fine_jump_mu_grad_v_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(fine_phi, &fine_phi_p); CHKERRXX(ierr);
  if(fine_phi_xxyyzz != NULL){
    ierr = VecRestoreArrayRead(fine_phi_xxyyzz, &fine_phi_xxyyzz_p); CHKERRXX(ierr); }

  ierr = PetscLogEventEnd(log_my_p4est_two_phase_flows_viscosity, 0, 0, 0, 0); CHKERRXX(ierr);
}

double my_p4est_two_phase_flows_t::div_mu_grad_u_dir(const p4est_locidx_t &face_idx, const unsigned char &dir,
                                                     const bool &face_is_in_omega_minus, const p4est_locidx_t &fine_idx_of_face,
                                                     const double *vn_dir_p, const double *fine_jump_mu_grad_v_p,
                                                     const double *fine_phi_p, const double *fine_phi_xxyyzz_p)
{
  bool xgfm_treatment_required;
  double xgfm_fluxes[P4EST_FACES];
#ifdef P4_TO_P8
  Voronoi3D voro_cell = compute_voronoi_cell(face_idx, dir, face_is_in_omega_minus, fine_idx_of_face, fine_phi_p, fine_phi_xxyyzz_p, vn_dir_p, fine_jump_mu_grad_v_p, xgfm_treatment_required, xgfm_fluxes);
  const vector<ngbd3Dseed> *points;
#else
  Voronoi2D voro_cell = compute_voronoi_cell(face_idx, dir, face_is_in_omega_minus, fine_idx_of_face, fine_phi_p, fine_phi_xxyyzz_p, vn_dir_p, fine_jump_mu_grad_v_p, xgfm_treatment_required, xgfm_fluxes);
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
Voronoi3D my_p4est_two_phase_flows_t::compute_voronoi_cell(const p4est_locidx_t &face_idx, const unsigned char &dir,
                                                           const bool &face_is_in_omega_minus, const p4est_locidx_t &fine_idx_of_face,
                                                           const double *fine_phi_p, const double *fine_phi_xxyyzz_p,
                                                           const double *vn_dir_p, const double *fine_jump_mu_grad_v_p,
                                                           bool &xgfm_treatment_required, double xgfm_fluxes[P4EST_DIM])
#else
Voronoi2D my_p4est_two_phase_flows_t::compute_voronoi_cell(const p4est_locidx_t &face_idx, const unsigned char &dir,
                                                           const bool &face_is_in_omega_minus, const p4est_locidx_t &fine_idx_of_face,
                                                           const double *fine_phi_p, const double *fine_phi_xxyyzz_p,
                                                           const double *vn_dir_p, const double *fine_jump_mu_grad_v_p,
                                                           bool &xgfm_treatment_required, double xgfm_fluxes[P4EST_DIM])
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
  if(faces_n->found_uniform_face_neighborhood(face_idx, dir, face_neighbors) && no_wall_in_face_neighborhood(face_neighbors)) // is it in the finest uniform tesselation? (possible presence of interface, no need to search for neighbors already done)
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
                                     vn_dir_p, vn_dir_p, fine_jump_mu_grad_v_p,
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

      /* add the walls in 2d, note that they are dealt with by voro++ in 3D
       * This needs to be done AFTER assemble_from_set_of_faces because the latter starts by clearing the neighbor seeds
       * */
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


void my_p4est_two_phase_flows_t::sharp_derivative_of_face_field(sharp_derivative &one_sided_derivative,
                                                                const p4est_locidx_t &face_idx, const bool &face_is_in_omega_minus, const p4est_locidx_t &fine_idx_of_face, const uniform_face_ngbd *face_neighbors,
                                                                const double *fine_phi_p, const double *fine_phi_xxyyzz_p,
                                                                const unsigned char &der, const unsigned char &dir,
                                                                const double *vn_dir_minus_p, const double *vn_dir_plus_p, const double *fine_jump_mu_grad_vdir_p,
                                                                const p4est_quadrant &qm, const p4est_quadrant &qp,
                                                                const double *fine_mass_flux_p, const double *fine_normal_dir_p)
{
  p4est_locidx_t fine_idx_of_other_face = -1;
  const double* vn_dir_this_side_p  = (face_is_in_omega_minus? vn_dir_minus_p : vn_dir_plus_p );
  const double* vn_dir_other_side_p = (face_is_in_omega_minus? vn_dir_plus_p  : vn_dir_minus_p);
  bool other_face_is_in_omega_minus ;
  if(face_neighbors->neighbor_face_idx[der] >= 0)
    other_face_is_in_omega_minus = is_face_in_omega_minus(face_neighbors->neighbor_face_idx[der], dir, fine_phi_p, fine_idx_of_other_face);
  else // neighbor "face" is wall
  {
    double xyz_wall[P4EST_DIM]; faces_n->xyz_fr_f(face_idx, dir, xyz_wall);
    xyz_wall[der/2] = (der%2==1)? xyz_max[der/2] : xyz_min[der/2];
    other_face_is_in_omega_minus = ((*interp_phi)(xyz_wall) <= 0.0);
    if(bc_v[dir].wallType(xyz_wall) != DIRICHLET){
      int mpiret = MPI_Abort(p4est_n->mpicomm, 475869); SC_CHECK_MPI(mpiret); }
  }
  if((face_is_in_omega_minus && !other_face_is_in_omega_minus) || (!face_is_in_omega_minus && other_face_is_in_omega_minus))
  {
    P4EST_ASSERT(fine_idx_of_face != -1);
    if(fine_idx_of_other_face==-1)
    {
      double xyz_face[P4EST_DIM];
      faces_n->xyz_fr_f(face_idx, dir, xyz_face);
      std::cout << "face located at " << xyz_face[0] << ", " << xyz_face[1] << std::endl;
      std::cout << "oriented der = " << (int) der << std::endl;
      std::cout << "face_is_in_omega_minus = " << (face_is_in_omega_minus? "true" : "false") << std::endl;
      std::cout << "other_face_is_in_omega_minus = " << (other_face_is_in_omega_minus? "true" : "false") << std::endl;
    }
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
    P4EST_ASSERT((fine_mass_flux_p == NULL) || ((fine_mass_flux_p != NULL) && (fine_normal_dir_p != NULL)));

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
      if(fine_phi_xxyyzz_p!=NULL)
        theta = fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(phi_face, phi_intermediary, fine_phi_xxyyzz_p[P4EST_DIM*fine_idx_of_face+(der/2)], fine_phi_xxyyzz_p[P4EST_DIM*fine_intermediary_node+(der/2)], dsub);
      else
        theta = fraction_Interval_Covered_By_Irregular_Domain(phi_face, phi_intermediary, dsub, dsub);
      theta = ((phi_face>0.0)? (1.0-theta):theta);
      theta = ((theta < EPS)?0.0:MIN(theta, 1.0));
      jump_flux_comp_across = (1.0-theta)*fine_jump_mu_grad_vdir_p[P4EST_DIM*P4EST_DIM*fine_idx_of_face+P4EST_DIM*dir+(der/2)] + theta*fine_jump_mu_grad_vdir_p[P4EST_DIM*P4EST_DIM*fine_intermediary_node+P4EST_DIM*dir+(der/2)];
      jump_v_dir            = ((fine_mass_flux_p == NULL)? 0.0 : (((1.0-theta)*fine_mass_flux_p[fine_idx_of_face]*fine_normal_dir_p[fine_idx_of_face] + theta*fine_mass_flux_p[fine_intermediary_node]*fine_normal_dir_p[fine_intermediary_node])*jump_inverse_mass_density()));
      theta *= 0.5;
    }
    else
    {
      P4EST_ASSERT(signs_of_phi_are_different(phi_intermediary, phi_other_face));
      if(fine_phi_xxyyzz_p!=NULL)
        theta = fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(phi_intermediary, phi_other_face, fine_phi_xxyyzz_p[P4EST_DIM*fine_intermediary_node+(der/2)], fine_phi_xxyyzz_p[P4EST_DIM*fine_idx_of_other_face+(der/2)], dsub);
      else
        theta = fraction_Interval_Covered_By_Irregular_Domain(phi_intermediary, phi_other_face, dsub, dsub);
      theta = ((phi_intermediary>0.0)? (1.0-theta):theta);
      theta = ((theta < EPS)?0.0:MIN(theta, 1.0));
      jump_flux_comp_across = (1.0-theta)*fine_jump_mu_grad_vdir_p[P4EST_DIM*P4EST_DIM*fine_intermediary_node+P4EST_DIM*dir+(der/2)] + theta*fine_jump_mu_grad_vdir_p[P4EST_DIM*P4EST_DIM*fine_idx_of_other_face+P4EST_DIM*dir+(der/2)];
      jump_v_dir            = ((fine_mass_flux_p == NULL)? 0.0 : (((1.0-theta)*fine_mass_flux_p[fine_intermediary_node]*fine_normal_dir_p[fine_intermediary_node] + theta*fine_mass_flux_p[fine_idx_of_other_face]*fine_normal_dir_p[fine_idx_of_other_face])*jump_inverse_mass_density()));
      theta = 0.5*(1.0+theta);
    }
    P4EST_ASSERT((theta >=0.0) && (theta <=1.0));

    double mu_tilde   = (1-theta)*(face_is_in_omega_minus? mu_minus:mu_plus) + theta*(face_is_in_omega_minus? mu_plus:mu_minus);
    double mu_across  = (face_is_in_omega_minus? mu_plus : mu_minus);

    if(face_neighbors->neighbor_face_idx[der] >=0)
    {
      P4EST_ASSERT((fabs(vn_dir_this_side_p[face_idx]) < threshold_dbl_max) && (fabs(vn_dir_other_side_p[face_neighbors->neighbor_face_idx[der]]) < threshold_dbl_max));
      one_sided_derivative.derivative =
          ((der%2==1)?+1.0:-1.0)*(mu_across/mu_tilde)*(vn_dir_other_side_p[face_neighbors->neighbor_face_idx[der]] - vn_dir_this_side_p[face_idx] + (face_is_in_omega_minus? -1.0: +1.0)*jump_v_dir)/dxyz_min[der/2]
          + (face_is_in_omega_minus? -1.0: +1.0)*(1.0-theta)*jump_flux_comp_across/mu_tilde;
    }
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

void my_p4est_two_phase_flows_t::get_velocity_seen_from_face(neighbor_value &neighbor_velocity, const p4est_locidx_t &face_idx, const p4est_locidx_t &neighbor_face_idx,
                                                             const double *fine_phi_p, const double *fine_phi_xxyyzz_p,
                                                             const unsigned char &der, const unsigned char &dir,
                                                             const double *vn_dir_minus_p, const double *vn_dir_plus_p, const double *fine_jump_mu_grad_vdir_p,
                                                             const double *fine_mass_flux_p, const double *fine_normal_dir_p)
{
  p4est_locidx_t fine_idx_of_other_face = -1;
  p4est_locidx_t fine_idx_of_face=-1;
  const bool face_is_in_omega_minus = is_face_in_omega_minus(face_idx, dir, fine_phi_p, fine_idx_of_face);
  const double* vn_dir_this_side_p  = (face_is_in_omega_minus? vn_dir_minus_p : vn_dir_plus_p );
  const double* vn_dir_other_side_p = (face_is_in_omega_minus? vn_dir_plus_p  : vn_dir_minus_p);
  const bool other_face_is_in_omega_minus = is_face_in_omega_minus(neighbor_face_idx, dir, fine_phi_p, fine_idx_of_other_face);
  if((face_is_in_omega_minus && !other_face_is_in_omega_minus) || (!face_is_in_omega_minus && other_face_is_in_omega_minus))
  {
    P4EST_ASSERT(fine_idx_of_face != -1);
    P4EST_ASSERT(fine_idx_of_other_face!=-1);
    p4est_quadrant_t qm, qp;
    faces_n->find_quads_touching_face(face_idx, dir, qm, qp);
    P4EST_ASSERT((qm.p.piggy3.local_num!=-1) && (qp.p.piggy3.local_num!=-1));
    p4est_locidx_t fine_intermediary_node = -1;
    if(der==2*dir)
      get_fine_node_idx_of_logical_vertex(qm.p.piggy3.local_num, qm.p.piggy3.which_tree, DIM(0, 0, 0), fine_intermediary_node, &qm);
    else if(der==2*dir+1)
      get_fine_node_idx_of_logical_vertex(qp.p.piggy3.local_num, qp.p.piggy3.which_tree, DIM(0, 0, 0), fine_intermediary_node, &qp);
    else
      get_fine_node_idx_of_logical_vertex(qp.p.piggy3.local_num, qp.p.piggy3.which_tree,
                                          DIM(((dir==dir::x)?-1:((der/2==dir::x)? (2*(der%2)-1):0)), ((dir==dir::y)?-1:((der/2==dir::y)? (2*(der%2)-1):0)), ((dir==dir::z)?-1:((der/2==dir::z)? (2*(der%2)-1):0))), fine_intermediary_node, &qp);
    P4EST_ASSERT(fine_intermediary_node != -1);
    P4EST_ASSERT((fine_mass_flux_p == NULL) || ((fine_mass_flux_p != NULL) && (fine_normal_dir_p != NULL)));

#ifdef DEBUG
    double xyz_fine_face[P4EST_DIM], xyz_fine_other_face[P4EST_DIM], xyz_fine_intermediary[P4EST_DIM];
    node_xyz_fr_n(fine_idx_of_face, fine_p4est_n, fine_nodes_n, xyz_fine_face);
    node_xyz_fr_n(fine_idx_of_other_face, fine_p4est_n, fine_nodes_n, xyz_fine_other_face);
    node_xyz_fr_n(fine_intermediary_node, fine_p4est_n, fine_nodes_n, xyz_fine_intermediary);
    for (unsigned char ddir = 0; ddir < P4EST_DIM; ++ddir) {
      if(ddir==der/2)
        P4EST_ASSERT(fabs(0.5*(xyz_fine_face[ddir]+xyz_fine_other_face[ddir]) - xyz_fine_intermediary[ddir]) < 0.1*0.5*dxyz_min[ddir]);
      else
        P4EST_ASSERT(fabs(xyz_fine_face[ddir]-xyz_fine_other_face[ddir])<0.1*dxyz_min[ddir] && fabs(xyz_fine_intermediary[ddir]-xyz_fine_other_face[ddir])<0.1*dxyz_min[ddir]);
    }
#endif

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
      if(fine_phi_xxyyzz_p!=NULL)
        theta = fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(phi_face, phi_intermediary, fine_phi_xxyyzz_p[P4EST_DIM*fine_idx_of_face+(der/2)], fine_phi_xxyyzz_p[P4EST_DIM*fine_intermediary_node+(der/2)], dsub);
      else
        theta = fraction_Interval_Covered_By_Irregular_Domain(phi_face, phi_intermediary, dsub, dsub);
      theta = ((phi_face>0.0)? (1.0-theta):theta);
      theta = ((theta < EPS)?0.0:MIN(theta, 1.0));
      jump_flux_comp_across = (1.0-theta)*fine_jump_mu_grad_vdir_p[P4EST_DIM*P4EST_DIM*fine_idx_of_face+P4EST_DIM*dir+(der/2)] + theta*fine_jump_mu_grad_vdir_p[P4EST_DIM*P4EST_DIM*fine_intermediary_node+P4EST_DIM*dir+(der/2)];
      jump_v_dir            = ((fine_mass_flux_p == NULL)? 0.0 : (((1.0-theta)*fine_mass_flux_p[fine_idx_of_face]*fine_normal_dir_p[fine_idx_of_face] + theta*fine_mass_flux_p[fine_intermediary_node]*fine_normal_dir_p[fine_intermediary_node])*jump_inverse_mass_density()));
      theta *= 0.5;
    }
    else
    {
      P4EST_ASSERT(signs_of_phi_are_different(phi_intermediary, phi_other_face));
      if(fine_phi_xxyyzz_p!=NULL)
        theta = fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(phi_intermediary, phi_other_face, fine_phi_xxyyzz_p[P4EST_DIM*fine_intermediary_node+(der/2)], fine_phi_xxyyzz_p[P4EST_DIM*fine_idx_of_other_face+(der/2)], dsub);
      else
        theta = fraction_Interval_Covered_By_Irregular_Domain(phi_intermediary, phi_other_face, dsub, dsub);
      theta = ((phi_intermediary>0.0)? (1.0-theta):theta);
      theta = ((theta < EPS)?0.0:MIN(theta, 1.0));
      jump_flux_comp_across = (1.0-theta)*fine_jump_mu_grad_vdir_p[P4EST_DIM*P4EST_DIM*fine_intermediary_node+P4EST_DIM*dir+(der/2)] + theta*fine_jump_mu_grad_vdir_p[P4EST_DIM*P4EST_DIM*fine_idx_of_face+P4EST_DIM*dir+(der/2)];
      jump_v_dir            = ((fine_mass_flux_p == NULL)? 0.0 : (((1.0-theta)*fine_mass_flux_p[fine_intermediary_node]*fine_normal_dir_p[fine_intermediary_node] + theta*fine_mass_flux_p[fine_idx_of_other_face]*fine_normal_dir_p[fine_idx_of_other_face])*jump_inverse_mass_density()));
      theta = 0.5*(1.0+theta);
    }
    P4EST_ASSERT((theta >=0.0) && (theta <=1.0));

    double mu_tilde     = (1-theta)*(face_is_in_omega_minus? mu_minus : mu_plus) + theta*(face_is_in_omega_minus? mu_plus : mu_minus);
    double mu_across    = (face_is_in_omega_minus? mu_plus  : mu_minus);
    double mu_this_side = (face_is_in_omega_minus? mu_minus : mu_plus);

    P4EST_ASSERT((fabs(vn_dir_this_side_p[face_idx]) < threshold_dbl_max) && (fabs(vn_dir_other_side_p[neighbor_face_idx]) < threshold_dbl_max));
    neighbor_velocity.value     = ((1.0-theta)*mu_this_side*vn_dir_this_side_p[face_idx] +
                                   theta*(mu_across*(vn_dir_other_side_p[neighbor_face_idx] + (face_is_in_omega_minus? -1.0: +1.0)*jump_v_dir)
                                          + (((der%2==1)? +1.0:-1.0)*(face_is_in_omega_minus? -1.0:+1.0))*(1.0-theta)*dxyz_min[der/2]*jump_flux_comp_across))/mu_tilde;
    neighbor_velocity.distance  = theta*dxyz_min[der/2];
  }
  else
  {
    neighbor_velocity.value     = vn_dir_this_side_p[neighbor_face_idx];
    neighbor_velocity.distance  = dxyz_min[der/2];
  }
  return;
}

void my_p4est_two_phase_flows_t::compute_velocity_at_nodes()
{
  PetscErrorCode ierr;
  const double *fine_phi_p;
  const double *vnp1_plus_p[P4EST_DIM], *vnp1_minus_p[P4EST_DIM];
  double *v_nodes_omega_plus_p, *v_nodes_omega_minus_p;


  ierr = VecGetArrayRead(fine_phi, &fine_phi_p); CHKERRXX(ierr);
  ierr = create_node_vector_if_needed(vnp1_nodes_omega_minus, p4est_n, nodes_n, P4EST_DIM); CHKERRXX(ierr);
  ierr = create_node_vector_if_needed(vnp1_nodes_omega_plus, p4est_n, nodes_n, P4EST_DIM);  CHKERRXX(ierr);
  ierr = VecGetArray(vnp1_nodes_omega_minus, &v_nodes_omega_minus_p);   CHKERRXX(ierr);
  ierr = VecGetArray(vnp1_nodes_omega_plus, &v_nodes_omega_plus_p);     CHKERRXX(ierr);

  max_L2_norm_u[0] = max_L2_norm_u[1] = 0.0;

  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecGetArrayRead(vnp1_plus[dir], &vnp1_plus_p[dir]); CHKERRXX(ierr);
    ierr = VecGetArrayRead(vnp1_minus[dir], &vnp1_minus_p[dir]); CHKERRXX(ierr);
  }
  // loop through layer nodes first
  for(size_t i=0; i<ngbd_n->get_layer_size(); ++i)
  {
    p4est_locidx_t node_idx = ngbd_n->get_layer_node(i);
    interpolate_velocity_at_node(node_idx, v_nodes_omega_plus_p, v_nodes_omega_minus_p,
                                 vnp1_plus_p, vnp1_minus_p);
  }
  ierr = VecGhostUpdateBegin(vnp1_nodes_omega_minus, INSERT_VALUES, SCATTER_FORWARD);  CHKERRXX(ierr);
  ierr = VecGhostUpdateBegin(vnp1_nodes_omega_plus, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);
  /* interpolate vnp1 from faces to nodes */
  for(size_t i=0; i<ngbd_n->get_local_size(); ++i)
  {
    p4est_locidx_t node_idx = ngbd_n->get_local_node(i);
    interpolate_velocity_at_node(node_idx, v_nodes_omega_plus_p, v_nodes_omega_minus_p,
                                 vnp1_plus_p, vnp1_minus_p);
  }
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, max_L2_norm_u, 2, MPI_DOUBLE, MPI_MAX, p4est_n->mpicomm); SC_CHECK_MPI(mpiret);
  ierr = VecRestoreArrayRead(fine_phi, &fine_phi_p); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(vnp1_nodes_omega_minus, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(vnp1_nodes_omega_plus, INSERT_VALUES, SCATTER_FORWARD);  CHKERRXX(ierr);
  ierr = VecRestoreArray(vnp1_nodes_omega_minus, &v_nodes_omega_minus_p);           CHKERRXX(ierr);
  ierr = VecRestoreArray(vnp1_nodes_omega_plus, &v_nodes_omega_plus_p);             CHKERRXX(ierr);
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecRestoreArrayRead(vnp1_plus[dir], &vnp1_plus_p[dir]);    CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(vnp1_minus[dir], &vnp1_minus_p[dir]);  CHKERRXX(ierr);
  }
  create_node_vector_if_needed(vorticities, p4est_n, nodes_n, 2);
  compute_vorticities();
}

void my_p4est_two_phase_flows_t::interpolate_velocity_at_node(const p4est_locidx_t &node_idx, double *v_nodes_omega_plus_p, double *v_nodes_omega_minus_p,
                                                              const double *vnp1_plus_p[P4EST_DIM], const double *vnp1_minus_p[P4EST_DIM])
{
#ifdef CASL_THROWS
  PetscErrorCode ierr;
#endif

  double xyz[P4EST_DIM];
  node_xyz_fr_n(node_idx, p4est_n, nodes_n, xyz);
  p4est_indep_t *node = (p4est_indep_t*) sc_array_index(&nodes_n->indep_nodes, node_idx);
#ifdef P4_TO_P8
  const double min_tree_dim = MIN(convert_to_xyz[0], convert_to_xyz[1], convert_to_xyz[2]);
#else
  const double min_tree_dim = MIN(convert_to_xyz[0], convert_to_xyz[1]);
#endif

  vector<bool> velocity_component_is_set(2*P4EST_DIM, false); // P4EST_DIM for minus components + P4EST_DIM for plus components

  double mag_v_minus = 0.0;
  double mag_v_plus = 0.0;

  if(bc_v!=NULL && is_node_Wall(p4est_n, node))
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      if(bc_v[dir].wallType(xyz)==DIRICHLET)
      {
        v_nodes_omega_minus_p[P4EST_DIM*node_idx+dir] = bc_v[dir].wallValue(xyz); mag_v_minus += SQR(v_nodes_omega_minus_p[P4EST_DIM*node_idx+dir]);
        velocity_component_is_set[dir]                = true;
        v_nodes_omega_plus_p[P4EST_DIM*node_idx+dir]  = bc_v[dir].wallValue(xyz); mag_v_plus += SQR(v_nodes_omega_plus_p[P4EST_DIM*node_idx+dir]);
        velocity_component_is_set[P4EST_DIM+dir]      = true;
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
#endif
      {
        ngbd_n->find_neighbor_cell_of_node(node_idx, DIM(ii, jj, kk), quad_idx, tree_idx);
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
      double *result_field = ((sign == 0)? v_nodes_omega_minus_p: v_nodes_omega_plus_p);
      double &mag_v = ((sign == 0)? mag_v_minus: mag_v_plus);

      matrix_t        A_lsqr;
      vector<double>  rhs_lsqr;
      std::set<__ino64_t>  nb[P4EST_DIM];
      unsigned int nb_neumann_walls = 0;
      char neumann_wall[P4EST_DIM];
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
        neumann_wall[dir] = ((bc_v!=NULL && bc_v[dir].wallType(xyz)==NEUMANN) ? (is_node_Wall(p4est_n, node, 2*dir+1)? +1 : (is_node_Wall(p4est_n, node, 2*dir)? -1 : 0)) : 0);
        nb_neumann_walls += abs(neumann_wall[dir]);
        nb[dir].clear(); //initialize this
      }

      double min_w = 1e-6;
      double inv_max_w = 1e-6;

      int row_idx = 0;
      int col_idx;
      for (std::set<p4est_locidx_t>::const_iterator got_it= set_of_faces[dir].begin(); got_it != set_of_faces[dir].end(); ++got_it)
      {
        P4EST_ASSERT((*got_it>=0) && (*got_it < (faces_n->num_local[dir] + faces_n->num_ghost[dir])));
        row_idx += ((fabs(face_field_to_interpolate[*got_it]) < threshold_dbl_max)? 1 : 0);
        P4EST_ASSERT(!isnan(face_field_to_interpolate[*got_it]));
      }
      if(row_idx == 0)
      {
        result_field[P4EST_DIM*node_idx+dir] = NAN; // absurd value for undefined interpolated field (e.g. vn_minus far in omega plus --> no well-defined neighbor to be found so move on)
        mag_v = NAN;
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
        int64_t logical_qcoord_diff[P4EST_DIM];
        faces_n->rel_xyz_face_fr_node(neighbor_face, dir, xyz_t, xyz, node, brick, logical_qcoord_diff);

        for(unsigned char i=0; i<P4EST_DIM; ++i)
        {
          xyz_t[i] /= scaling;
          nb[i].insert(logical_qcoord_diff[i]);
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
            rhs_lsqr[row_idx] -= ((double) neumann_wall[dir])*bc_v[dir].wallValue(xyz)*     xyz_t[dir]*scaling; // multiplication by w at the end
          }
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
      result_field[P4EST_DIM*node_idx+dir] = solve_lsqr_system(A_lsqr, rhs_lsqr, nb[0].size(), nb[1].size(), nb[2].size(), 2);
#else
      result_field[P4EST_DIM*node_idx+dir] = solve_lsqr_system(A_lsqr, rhs_lsqr, nb[0].size(), nb[1].size(), 2);
#endif
      if (!ISNAN(mag_v))
        mag_v += SQR(result_field[P4EST_DIM*node_idx+dir]);
    }
  }
  if (!ISNAN(mag_v_minus))
    max_L2_norm_u[0] = MAX(max_L2_norm_u[0], sqrt(mag_v_minus));
  if (!ISNAN(mag_v_plus))
    max_L2_norm_u[1] = MAX(max_L2_norm_u[1], sqrt(mag_v_plus));
}

void my_p4est_two_phase_flows_t::compute_vorticities()
{
  PetscErrorCode ierr;

  const quad_neighbor_nodes_of_node_t *qnnn;

  const double *vnp1_minus_p, *vnp1_plus_p;
  ierr = VecGetArrayRead(vnp1_nodes_omega_minus,  &vnp1_minus_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(vnp1_nodes_omega_plus,   &vnp1_plus_p); CHKERRXX(ierr);
  const double *inputs[2] = {vnp1_minus_p, vnp1_plus_p};

  double *vorticities_p;
  ierr = VecGetArray(vorticities, &vorticities_p); CHKERRXX(ierr);
  double vel_grad_minus[P4EST_DIM*P4EST_DIM], vel_grad_plus[P4EST_DIM*P4EST_DIM];
  double *outputs[2] = {vel_grad_minus, vel_grad_plus};

  for(size_t i=0; i<ngbd_n->get_layer_size(); ++i)
  {
    p4est_locidx_t n    = ngbd_n->get_layer_node(i);
    p4est_locidx_t twon = 2*n;
    ngbd_n->get_neighbors(n, qnnn);
    qnnn->gradient_all_components(inputs, outputs, 2, P4EST_DIM);
#ifdef P4_TO_P8
    double vx_plus        = vel_grad_plus[P4EST_DIM*dir::z+dir::y]  - vel_grad_plus[P4EST_DIM*dir::y+dir::z];
    double vx_minus       = vel_grad_minus[P4EST_DIM*dir::z+dir::y] - vel_grad_minus[P4EST_DIM*dir::y+dir::z];
    double vy_plus        = vel_grad_plus[P4EST_DIM*dir::x+dir::z]  - vel_grad_plus[P4EST_DIM*dir::z+dir::x];
    double vy_minus       = vel_grad_minus[P4EST_DIM*dir::x+dir::z] - vel_grad_minus[P4EST_DIM*dir::z+dir::x];
    double vz_plus        = vel_grad_plus[P4EST_DIM*dir::y+dir::x]  - vel_grad_plus[P4EST_DIM*dir::x+dir::y];
    double vz_minus       = vel_grad_minus[P4EST_DIM*dir::y+dir::x] - vel_grad_minus[P4EST_DIM*dir::x+dir::y];
    vorticities_p[twon]   = sqrt(SQR(vx_minus) + SQR(vy_minus) + SQR(vz_minus));
    vorticities_p[twon+1] = sqrt(SQR(vx_plus) + SQR(vy_plus) + SQR(vz_plus));
#else
    vorticities_p[twon]   = vel_grad_minus[P4EST_DIM*dir::y+dir::x] - vel_grad_minus[P4EST_DIM*dir::x+dir::y];
    vorticities_p[twon+1] = vel_grad_plus[P4EST_DIM*dir::y+dir::x]  - vel_grad_plus[P4EST_DIM*dir::x+dir::y];
#endif
  }
  ierr = VecGhostUpdateBegin(vorticities, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for(size_t i=0; i<ngbd_n->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n->get_local_node(i);
    p4est_locidx_t twon = 2*n;
    ngbd_n->get_neighbors(n, qnnn);
    qnnn->gradient_all_components(inputs, outputs, 2, P4EST_DIM);
#ifdef P4_TO_P8
    double vx_plus        = vel_grad_plus[P4EST_DIM*dir::z+dir::y]  - vel_grad_plus[P4EST_DIM*dir::y+dir::z];
    double vx_minus       = vel_grad_minus[P4EST_DIM*dir::z+dir::y] - vel_grad_minus[P4EST_DIM*dir::y+dir::z];
    double vy_plus        = vel_grad_plus[P4EST_DIM*dir::x+dir::z]  - vel_grad_plus[P4EST_DIM*dir::z+dir::x];
    double vy_minus       = vel_grad_minus[P4EST_DIM*dir::x+dir::z] - vel_grad_minus[P4EST_DIM*dir::z+dir::x];
    double vz_plus        = vel_grad_plus[P4EST_DIM*dir::y+dir::x]  - vel_grad_plus[P4EST_DIM*dir::x+dir::y];
    double vz_minus       = vel_grad_minus[P4EST_DIM*dir::y+dir::x] - vel_grad_minus[P4EST_DIM*dir::x+dir::y];
    vorticities_p[twon]   = sqrt(SQR(vx_minus) + SQR(vy_minus) + SQR(vz_minus));
    vorticities_p[twon+1] = sqrt(SQR(vx_plus) + SQR(vy_plus) + SQR(vz_plus));
#else
    vorticities_p[twon]   = vel_grad_minus[P4EST_DIM*dir::y+dir::x] - vel_grad_minus[P4EST_DIM*dir::x+dir::y];
    vorticities_p[twon+1] = vel_grad_plus[P4EST_DIM*dir::y+dir::x]  - vel_grad_plus[P4EST_DIM*dir::x+dir::y];
#endif
  }
  ierr = VecGhostUpdateEnd(vorticities, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecRestoreArray(vorticities, &vorticities_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(vnp1_nodes_omega_plus,   &vnp1_plus_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(vnp1_nodes_omega_minus,  &vnp1_minus_p); CHKERRXX(ierr);
}

void my_p4est_two_phase_flows_t::extrapolate_velocities_across_interface_in_finest_computational_cells_Aslam_PDE(const extrapolation_technique& extrapolation_method, const unsigned int& n_iterations)
{
  P4EST_ASSERT(n_iterations>0);
  PetscErrorCode ierr;
  my_p4est_interpolation_nodes_t interp_normal(fine_ngbd_n);
  interp_normal.set_input(fine_normal, linear, P4EST_DIM);
  Vec normal_derivative_of_vnp1_minus[P4EST_DIM], normal_derivative_of_vnp1_plus[P4EST_DIM];
  double *normal_derivative_of_vnp1_minus_p[P4EST_DIM], *normal_derivative_of_vnp1_plus_p[P4EST_DIM], *vnp1_minus_p[P4EST_DIM], *vnp1_plus_p[P4EST_DIM];
  const double *fine_phi_p, *fine_phi_xxyyzz_p, *fine_jump_mu_grad_v_p;
  // initialize normal derivatives of the P4EST_DIM face-sampled fields
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecCreateGhostFaces(p4est_n, faces_n, &normal_derivative_of_vnp1_minus[dir], dir); CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est_n, faces_n, &normal_derivative_of_vnp1_plus[dir], dir); CHKERRXX(ierr);
  }
  // get data pointers
  ierr = VecGetArrayRead(fine_phi, &fine_phi_p); CHKERRXX(ierr);
  if(fine_phi_xxyyzz!=NULL){
    ierr = VecGetArrayRead(fine_phi_xxyyzz, &fine_phi_xxyyzz_p); CHKERRXX(ierr); }
  else
    fine_phi_xxyyzz_p = NULL;
  ierr = VecGetArrayRead(fine_jump_mu_grad_v, &fine_jump_mu_grad_v_p); CHKERRXX(ierr);
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecGetArray(vnp1_minus[dir], &vnp1_minus_p[dir]); CHKERRXX(ierr);
    ierr = VecGetArray(vnp1_plus[dir], &vnp1_plus_p[dir]); CHKERRXX(ierr);
    ierr = VecGetArray(normal_derivative_of_vnp1_minus[dir], &normal_derivative_of_vnp1_minus_p[dir]); CHKERRXX(ierr);
    ierr = VecGetArray(normal_derivative_of_vnp1_plus[dir], &normal_derivative_of_vnp1_plus_p[dir]); CHKERRXX(ierr);
  }

  /* initialize the normal derivatives */
  // local layer faces first
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    for (size_t k = 0; k < faces_n->get_layer_size(dir); ++k) {
      p4est_locidx_t local_face_idx = faces_n->get_layer_face(dir, k);
      initialize_normal_derivative_of_velocity_on_faces_local(local_face_idx, dir, interp_normal, fine_jump_mu_grad_v_p, fine_phi_p, fine_phi_xxyyzz_p,
                                                              vnp1_minus_p, vnp1_plus_p, normal_derivative_of_vnp1_minus_p, normal_derivative_of_vnp1_plus_p);
    }
  }
  // start updates
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecGhostUpdateBegin(normal_derivative_of_vnp1_minus[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(normal_derivative_of_vnp1_plus[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }
  // local inner faces
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    for (size_t k = 0; k < faces_n->get_local_size(dir); ++k) {
      p4est_locidx_t local_face_idx = faces_n->get_local_face(dir, k);
      initialize_normal_derivative_of_velocity_on_faces_local(local_face_idx, dir, interp_normal, fine_jump_mu_grad_v_p, fine_phi_p, fine_phi_xxyyzz_p,
                                                              vnp1_minus_p, vnp1_plus_p, normal_derivative_of_vnp1_minus_p, normal_derivative_of_vnp1_plus_p);
    }
  }
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecGhostUpdateEnd(normal_derivative_of_vnp1_minus[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(normal_derivative_of_vnp1_plus[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }
  /* extrapolate the normal derivatives */
  for (unsigned int iter = 0; iter < n_iterations; ++iter) {
    // local layer faces first
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      for (size_t k = 0; k < faces_n->get_layer_size(dir); ++k) {
        p4est_locidx_t local_face_idx = faces_n->get_layer_face(dir, k);
        switch (extrapolation_method) {
        case PSEUDO_TIME:
          extrapolate_normal_derivatives_of_face_velocity_local_pseudo_time(local_face_idx, dir, interp_normal, fine_phi_p,
                                                                            normal_derivative_of_vnp1_minus_p, normal_derivative_of_vnp1_plus_p);
          break;
        case EXPLICIT_ITERATIVE:
          extrapolate_normal_derivatives_of_face_velocity_local_explicit_iterative(local_face_idx, dir, interp_normal, fine_phi_p,
                                                                                   normal_derivative_of_vnp1_minus_p, normal_derivative_of_vnp1_plus_p);
          break;
        default:
          throw std::invalid_argument("my_p4est_two_phase_flows_t::extrapolate_velocities_across_interface_in_finest_computational_cells_Aslam_PDE(): unknown extrapolation_method");
          break;
        }
      }
    }
    // start updates
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      ierr = VecGhostUpdateBegin(normal_derivative_of_vnp1_minus[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(normal_derivative_of_vnp1_plus[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }
    // local inner faces
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      for (size_t k = 0; k < faces_n->get_local_size(dir); ++k) {
        p4est_locidx_t local_face_idx = faces_n->get_local_face(dir, k);
        switch (extrapolation_method) {
        case PSEUDO_TIME:
          extrapolate_normal_derivatives_of_face_velocity_local_pseudo_time(local_face_idx, dir, interp_normal, fine_phi_p,
                                                                            normal_derivative_of_vnp1_minus_p, normal_derivative_of_vnp1_plus_p);
          break;
        case EXPLICIT_ITERATIVE:
          extrapolate_normal_derivatives_of_face_velocity_local_explicit_iterative(local_face_idx, dir, interp_normal, fine_phi_p,
                                                                                   normal_derivative_of_vnp1_minus_p, normal_derivative_of_vnp1_plus_p);
          break;
        default:
          throw std::invalid_argument("my_p4est_two_phase_flows_t::extrapolate_velocities_across_interface_in_finest_computational_cells_Aslam_PDE(): unknown extrapolation_method");
          break;
        }
      }
    }
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      ierr = VecGhostUpdateEnd(normal_derivative_of_vnp1_minus[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(normal_derivative_of_vnp1_plus[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }
  }

  const double *normal_derivative_of_vnp1_minus_read_p[P4EST_DIM], *normal_derivative_of_vnp1_plus_read_p[P4EST_DIM];
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecRestoreArray(normal_derivative_of_vnp1_minus[dir], &normal_derivative_of_vnp1_minus_p[dir]); CHKERRXX(ierr);
    ierr = VecRestoreArray(normal_derivative_of_vnp1_plus[dir], &normal_derivative_of_vnp1_plus_p[dir]); CHKERRXX(ierr);
    ierr = VecGetArrayRead(normal_derivative_of_vnp1_minus[dir], &normal_derivative_of_vnp1_minus_read_p[dir]); CHKERRXX(ierr);
    ierr = VecGetArrayRead(normal_derivative_of_vnp1_plus[dir], &normal_derivative_of_vnp1_plus_read_p[dir]); CHKERRXX(ierr);
  }


  /* extrapolate the velocities */
  for (unsigned int iter = 0; iter < n_iterations; ++iter) {
    // local layer faces first
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      for (size_t k = 0; k < faces_n->get_layer_size(dir); ++k) {
        p4est_locidx_t local_face_idx = faces_n->get_layer_face(dir, k);
        switch (extrapolation_method) {
        case PSEUDO_TIME:
          solve_velocity_extrapolation_local_pseudo_time(local_face_idx, dir, interp_normal,
                                                         fine_jump_mu_grad_v_p, fine_phi_p, fine_phi_xxyyzz_p,
                                                         normal_derivative_of_vnp1_minus_read_p, normal_derivative_of_vnp1_plus_read_p, vnp1_minus_p, vnp1_plus_p);
          break;
        case EXPLICIT_ITERATIVE:
//          solve_velocity_extrapolation_local_pseudo_time(local_face_idx, dir, interp_normal,
//                                                         fine_jump_mu_grad_v_p, fine_phi_p, fine_phi_xxyyzz_p,
//                                                         normal_derivative_of_vnp1_minus_read_p, normal_derivative_of_vnp1_plus_read_p, vnp1_minus_p, vnp1_plus_p);
          solve_velocity_extrapolation_local_explicit_iterative(local_face_idx, dir, interp_normal,
                                                                fine_jump_mu_grad_v_p, fine_phi_p, fine_phi_xxyyzz_p,
                                                                normal_derivative_of_vnp1_minus_read_p, normal_derivative_of_vnp1_plus_read_p, vnp1_minus_p, vnp1_plus_p);
          break;
        default:
          throw std::invalid_argument("my_p4est_two_phase_flows_t::extrapolate_velocities_across_interface_in_finest_computational_cells_Aslam_PDE(): unknown extrapolation_method");
          break;
        }
      }
    }
    // start updates
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      ierr = VecGhostUpdateBegin(vnp1_minus[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(vnp1_plus[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }
    // local inner faces
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      for (size_t k = 0; k < faces_n->get_local_size(dir); ++k) {
        p4est_locidx_t local_face_idx = faces_n->get_local_face(dir, k);
        switch (extrapolation_method) {
        case PSEUDO_TIME:
          solve_velocity_extrapolation_local_pseudo_time(local_face_idx, dir, interp_normal,
                                                         fine_jump_mu_grad_v_p, fine_phi_p, fine_phi_xxyyzz_p,
                                                         normal_derivative_of_vnp1_minus_read_p, normal_derivative_of_vnp1_plus_read_p, vnp1_minus_p, vnp1_plus_p);
          break;
        case EXPLICIT_ITERATIVE:
//          solve_velocity_extrapolation_local_pseudo_time(local_face_idx, dir, interp_normal,
//                                                         fine_jump_mu_grad_v_p, fine_phi_p, fine_phi_xxyyzz_p,
//                                                         normal_derivative_of_vnp1_minus_read_p, normal_derivative_of_vnp1_plus_read_p, vnp1_minus_p, vnp1_plus_p);
          solve_velocity_extrapolation_local_explicit_iterative(local_face_idx, dir, interp_normal,
                                                                fine_jump_mu_grad_v_p, fine_phi_p, fine_phi_xxyyzz_p,
                                                                normal_derivative_of_vnp1_minus_read_p, normal_derivative_of_vnp1_plus_read_p, vnp1_minus_p, vnp1_plus_p);
          break;
        default:
          throw std::invalid_argument("my_p4est_two_phase_flows_t::extrapolate_velocities_across_interface_in_finest_computational_cells_Aslam_PDE(): unknown extrapolation_method");
          break;
        }
      }
    }
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      ierr = VecGhostUpdateEnd(vnp1_minus[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(vnp1_plus[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }
  }


  // restore data pointers
  ierr = VecRestoreArrayRead(fine_phi, &fine_phi_p); CHKERRXX(ierr);
  if(fine_phi_xxyyzz_p!=NULL){
    ierr = VecRestoreArrayRead(fine_phi_xxyyzz, &fine_phi_xxyyzz_p); CHKERRXX(ierr); }
  ierr = VecRestoreArrayRead(fine_jump_mu_grad_v, &fine_jump_mu_grad_v_p); CHKERRXX(ierr);
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecRestoreArray(vnp1_minus[dir], &vnp1_minus_p[dir]); CHKERRXX(ierr);
    ierr = VecRestoreArray(vnp1_plus[dir], &vnp1_plus_p[dir]); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(normal_derivative_of_vnp1_minus[dir], &normal_derivative_of_vnp1_minus_read_p[dir]); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(normal_derivative_of_vnp1_plus[dir], &normal_derivative_of_vnp1_plus_read_p[dir]); CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(normal_derivative_of_vnp1_minus[dir]); CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(normal_derivative_of_vnp1_plus[dir]); CHKERRXX(ierr);
  }
}

void my_p4est_two_phase_flows_t::initialize_normal_derivative_of_velocity_on_faces_local(const p4est_locidx_t &local_face_idx, const unsigned char &dir, const my_p4est_interpolation_nodes_t &interp_normal,
                                                                                         const double *fine_jump_mu_grad_vdir_p, const double *fine_phi_p, const double *fine_phi_xxyyzz_p,
                                                                                         double *vnp1_minus_p[P4EST_DIM], double *vnp1_plus_p[P4EST_DIM],
                                                                                         double *normal_derivative_of_vnp1_minus_p[P4EST_DIM], double *normal_derivative_of_vnp1_plus_p[P4EST_DIM])
{
  double local_normal[P4EST_DIM], xyz_face[P4EST_DIM];
  p4est_locidx_t fine_idx_of_face;
  const bool face_is_in_omega_minus = is_face_in_omega_minus(local_face_idx, dir, fine_phi_p, fine_idx_of_face);
  faces_n->xyz_fr_f(local_face_idx, dir, xyz_face);
  interp_normal(xyz_face, local_normal);
  const uniform_face_ngbd* face_ngbd;
  if(faces_n->found_uniform_face_neighborhood(local_face_idx, dir, face_ngbd))
  {
    sharp_derivative sharp_derivative_p, sharp_derivative_m;
    p4est_quadrant_t qm, qp;
    faces_n->find_quads_touching_face(local_face_idx, dir, qm, qp);
    double value = 0.0;
    for (unsigned char der = 0; der < P4EST_DIM; ++der) {
      sharp_derivative_of_face_field(sharp_derivative_p, local_face_idx, face_is_in_omega_minus, fine_idx_of_face, face_ngbd,
                                     fine_phi_p, fine_phi_xxyyzz_p, 2*der+1, dir,
                                     vnp1_minus_p[dir], vnp1_plus_p[dir], fine_jump_mu_grad_vdir_p, qm, qp, NULL, NULL);
      sharp_derivative_of_face_field(sharp_derivative_m, local_face_idx, face_is_in_omega_minus, fine_idx_of_face, face_ngbd,
                                     fine_phi_p, fine_phi_xxyyzz_p, 2*der, dir,
                                     vnp1_minus_p[dir], vnp1_plus_p[dir], fine_jump_mu_grad_vdir_p, qm, qp, NULL, NULL);
      value += local_normal[der]*(sharp_derivative_m.theta*sharp_derivative_p.derivative + sharp_derivative_p.theta*sharp_derivative_m.derivative)/(sharp_derivative_m.theta+sharp_derivative_p.theta);
    }
    if(face_is_in_omega_minus)
    {
      normal_derivative_of_vnp1_minus_p[dir][local_face_idx] = value;
      normal_derivative_of_vnp1_plus_p[dir][local_face_idx] = 0.0; // to be calculated in actual extrapolation
      P4EST_ASSERT(fabs(vnp1_plus_p[dir][local_face_idx]) > threshold_dbl_max);
      vnp1_plus_p[dir][local_face_idx] = vnp1_minus_p[dir][local_face_idx] +0.1 /* + jump */;
    }
    else
    {
      normal_derivative_of_vnp1_minus_p[dir][local_face_idx] = 0.0; // to be calculated in actual extrapolation
      normal_derivative_of_vnp1_plus_p[dir][local_face_idx] = -value; // minus because reverse normal for that field
      P4EST_ASSERT(fabs(vnp1_minus_p[dir][local_face_idx]) > threshold_dbl_max);
      vnp1_minus_p[dir][local_face_idx] = vnp1_plus_p[dir][local_face_idx] +0.1 /* - jump */;
    }
  }
  else
  {
    for (unsigned char der = 0; der < P4EST_DIM; ++der)
    {
      double signed_normal_component          = (face_is_in_omega_minus?-1.0:+1.0)*local_normal[der];
      const unsigned char local_oriented_der  = ((signed_normal_component > 0.0)? 2*der:2*der+1);
      p4est_locidx_t neighbor_face_idx;
      if(!faces_n->found_finest_face_neighbor(local_face_idx, dir, local_oriented_der, neighbor_face_idx))
      {
        P4EST_ASSERT((face_is_in_omega_minus? fabs(vnp1_plus_p[dir][local_face_idx]) : fabs(vnp1_minus_p[dir][local_face_idx])) > threshold_dbl_max);
        return; // no initialization to be done here, no extrapolation will be calculated and the points will be rejected from interpolation stencils
      }
      P4EST_ASSERT((neighbor_face_idx >=0) && (neighbor_face_idx < faces_n->num_local[dir] + faces_n->num_ghost[dir]));
    }

    if(face_is_in_omega_minus)
    {
      normal_derivative_of_vnp1_plus_p[dir][local_face_idx] = 0.0; // to be calculated in extrapolation
      P4EST_ASSERT(fabs(vnp1_plus_p[dir][local_face_idx]) > threshold_dbl_max);
      vnp1_plus_p[dir][local_face_idx] = vnp1_minus_p[dir][local_face_idx] +0.1 /* + jump */;
    }
    else
    {
      normal_derivative_of_vnp1_minus_p[dir][local_face_idx] = 0.0; // to be calculated in extrapolation
      P4EST_ASSERT(fabs(vnp1_minus_p[dir][local_face_idx]) > threshold_dbl_max);
      vnp1_minus_p[dir][local_face_idx] = vnp1_plus_p[dir][local_face_idx] +0.1 /* - jump */;
    }
  }
}

void my_p4est_two_phase_flows_t::extrapolate_normal_derivatives_of_face_velocity_local_explicit_iterative(const p4est_locidx_t &local_face_idx, const unsigned char &dir, const my_p4est_interpolation_nodes_t &interp_normal, const double *fine_phi_p,
                                                                                                          double *normal_derivative_of_vnp1_minus_p[P4EST_DIM], double *normal_derivative_of_vnp1_plus_p[P4EST_DIM])
{
  double xyz_face[P4EST_DIM], local_normal[P4EST_DIM];
  faces_n->xyz_fr_f(local_face_idx, dir, xyz_face);
  interp_normal(xyz_face, local_normal);
  double lhs_normal_derivative_field    = 0.0;
  double rhs_normal_derivative_field    = 0.0;
  // if the face is in omega minus, we extend the + field and the normal is reverted
  const bool face_is_in_omega_minus     = is_face_in_omega_minus(local_face_idx, dir, fine_phi_p);
  double* normal_derivative_of_field_p  = (face_is_in_omega_minus? normal_derivative_of_vnp1_plus_p[dir]  : normal_derivative_of_vnp1_minus_p[dir]);
  for (unsigned char der = 0; der < P4EST_DIM; ++der)
  {
    double signed_normal_component          = (face_is_in_omega_minus?-1.0:+1.0)*local_normal[der];
    const unsigned char local_oriented_der  = ((signed_normal_component > 0.0)? 2*der:2*der+1);
    p4est_locidx_t neighbor_face_idx;
    if(!faces_n->found_finest_face_neighbor(local_face_idx, dir, local_oriented_der, neighbor_face_idx))
      return; // one of the neighbor is not a uniform fine neigbor, forget about this one and return
    P4EST_ASSERT((neighbor_face_idx >=0) && (neighbor_face_idx < faces_n->num_local[dir] + faces_n->num_ghost[dir]));
    // normal derivatives: always on the grid!
    P4EST_ASSERT(fabs(normal_derivative_of_field_p[neighbor_face_idx]) < threshold_dbl_max);
    lhs_normal_derivative_field  += fabs(signed_normal_component)/dxyz_min[der];
    rhs_normal_derivative_field  += fabs(signed_normal_component)*normal_derivative_of_field_p[neighbor_face_idx]/dxyz_min[der];
  }
  normal_derivative_of_field_p[local_face_idx] = rhs_normal_derivative_field/lhs_normal_derivative_field;
}

void my_p4est_two_phase_flows_t::solve_velocity_extrapolation_local_explicit_iterative(const p4est_locidx_t &local_face_idx, const unsigned char &dir, const my_p4est_interpolation_nodes_t &interp_normal,
                                                                                       const double *fine_jump_mu_grad_vdir_p, const double *fine_phi_p, const double *fine_phi_xxyyzz_p,
                                                                                       const double *normal_derivative_of_vnp1_minus_p[P4EST_DIM], const double *normal_derivative_of_vnp1_plus_p[P4EST_DIM],
                                                                                       double *vnp1_minus_p[P4EST_DIM], double *vnp1_plus_p[P4EST_DIM])
{
  neighbor_value neighbor_velocity;
  double xyz_face[P4EST_DIM], local_normal[P4EST_DIM];
  faces_n->xyz_fr_f(local_face_idx, dir, xyz_face);
  interp_normal(xyz_face, local_normal);
  double lhs_field                      = 0.0;
  double rhs_field                      = 0.0;
  bool too_close_flag                   = false;
  // if the face is in omega minus, we extend the + field and the normal is reverted
  const bool face_is_in_omega_minus     = is_face_in_omega_minus(local_face_idx, dir, fine_phi_p);
  double* field_p                       = (face_is_in_omega_minus? vnp1_plus_p[dir]                       : vnp1_minus_p[dir]);
  const double* normal_derivative_of_field_p  = (face_is_in_omega_minus? normal_derivative_of_vnp1_plus_p[dir]  : normal_derivative_of_vnp1_minus_p[dir]);
  for (unsigned char der = 0; der < P4EST_DIM; ++der)
  {
    double signed_normal_component          = (face_is_in_omega_minus?-1.0:+1.0)*local_normal[der];
    const unsigned char local_oriented_der  = ((signed_normal_component > 0.0)? 2*der:2*der+1);
    p4est_locidx_t neighbor_face_idx;
    if(!faces_n->found_finest_face_neighbor(local_face_idx, dir, local_oriented_der, neighbor_face_idx))
    {
      P4EST_ASSERT(fabs(field_p[local_face_idx]) > threshold_dbl_max);
      return; // one of the neighbor is not a uniform fine neigbor, forget about this one and return
    }
    P4EST_ASSERT((neighbor_face_idx >=0) && (neighbor_face_idx < faces_n->num_local[dir] + faces_n->num_ghost[dir]));
    // field: may be subresolved
    get_velocity_seen_from_face(neighbor_velocity, local_face_idx, neighbor_face_idx, fine_phi_p, fine_phi_xxyyzz_p, local_oriented_der, dir, vnp1_minus_p[dir], vnp1_plus_p[dir], fine_jump_mu_grad_vdir_p, NULL, NULL);
    P4EST_ASSERT(neighbor_velocity.distance >= 0.0);
    if(neighbor_velocity.distance < dxyz_min[der]*(dxyz_min[der]/convert_to_xyz[der])) // "too close" --> avoid ill-defined terms due to inverses of very small distances, set the value equal to the direct neighbor
    {
      field_p[local_face_idx]     = neighbor_velocity.value;
      too_close_flag              = true;
    }
    else
    {
      P4EST_ASSERT(fabs(neighbor_velocity.value) < threshold_dbl_max);
      lhs_field                  += fabs(signed_normal_component)/neighbor_velocity.distance;
      rhs_field                  += fabs(signed_normal_component)*neighbor_velocity.value/neighbor_velocity.distance;
    }
  }
  if(!too_close_flag)
  {
    P4EST_ASSERT(fabs(field_p[local_face_idx]) < threshold_dbl_max);
    rhs_field                   += normal_derivative_of_field_p[local_face_idx];
    field_p[local_face_idx]      = rhs_field/lhs_field;
    P4EST_ASSERT(!isnan(field_p[local_face_idx]) && (fabs(field_p[local_face_idx]) < threshold_dbl_max));
  }
}


void my_p4est_two_phase_flows_t::extrapolate_normal_derivatives_of_face_velocity_local_pseudo_time(const p4est_locidx_t &local_face_idx, const unsigned char &dir, const my_p4est_interpolation_nodes_t &interp_normal, const double *fine_phi_p,
                                                                                                   double *normal_derivative_of_vnp1_minus_p[P4EST_DIM], double *normal_derivative_of_vnp1_plus_p[P4EST_DIM])
{
  double xyz_face[P4EST_DIM], local_normal[P4EST_DIM];
  faces_n->xyz_fr_f(local_face_idx, dir, xyz_face);
  interp_normal(xyz_face, local_normal);
  double increment_normal_derivative    = 0.0;
  // if the face is in omega minus, we extend the + field and the normal is reverted
  const bool face_is_in_omega_minus     = is_face_in_omega_minus(local_face_idx, dir, fine_phi_p);
  double* normal_derivative_of_field_p  = (face_is_in_omega_minus? normal_derivative_of_vnp1_plus_p[dir]  : normal_derivative_of_vnp1_minus_p[dir]);
  for (unsigned char der = 0; der < P4EST_DIM; ++der)
  {
    double signed_normal_component          = (face_is_in_omega_minus?-1.0:+1.0)*local_normal[der];
    const unsigned char local_oriented_der  = ((signed_normal_component > 0.0)? 2*der:2*der+1);
    p4est_locidx_t neighbor_face_idx;
    if(!faces_n->found_finest_face_neighbor(local_face_idx, dir, local_oriented_der, neighbor_face_idx))
      return; // one of the neighbor is not a uniform fine neigbor, forget about this one and return
    P4EST_ASSERT((neighbor_face_idx >=0) && (neighbor_face_idx < faces_n->num_local[dir] + faces_n->num_ghost[dir]));
    // normal derivatives: always on the grid!
    P4EST_ASSERT(fabs(normal_derivative_of_field_p[neighbor_face_idx]) < threshold_dbl_max);
    increment_normal_derivative  -= fabs(signed_normal_component)*(normal_derivative_of_field_p[local_face_idx] - normal_derivative_of_field_p[neighbor_face_idx])/dxyz_min[der];
  }
  double dt_normal_derivative           = MIN(DIM(dxyz_min[0], dxyz_min[1], dxyz_min[2]))/((double) P4EST_DIM); // as in other extensions (at nodes)
  normal_derivative_of_field_p[local_face_idx] += dt_normal_derivative*increment_normal_derivative;
}


void my_p4est_two_phase_flows_t::solve_velocity_extrapolation_local_pseudo_time(const p4est_locidx_t &local_face_idx, const unsigned char &dir, const my_p4est_interpolation_nodes_t &interp_normal,
                                                                                const double *fine_jump_mu_grad_vdir_p, const double *fine_phi_p, const double *fine_phi_xxyyzz_p,
                                                                                const double *normal_derivative_of_vnp1_minus_p[P4EST_DIM], const double *normal_derivative_of_vnp1_plus_p[P4EST_DIM],
                                                                                double *vnp1_minus_p[P4EST_DIM], double *vnp1_plus_p[P4EST_DIM])
{
  neighbor_value neighbor_velocity;
  double xyz_face[P4EST_DIM], local_normal[P4EST_DIM];
  faces_n->xyz_fr_f(local_face_idx, dir, xyz_face);
  interp_normal(xyz_face, local_normal);
  double dt_field                       = MIN(DIM(dxyz_min[0], dxyz_min[1], dxyz_min[2]));
  double increment_field                = 0.0;
  bool too_close_flag                   = false;
  // if the face is in omega minus, we extend the + field and the normal is reverted
  const bool face_is_in_omega_minus     = is_face_in_omega_minus(local_face_idx, dir, fine_phi_p);
  double* field_p                       = (face_is_in_omega_minus? vnp1_plus_p[dir]                       : vnp1_minus_p[dir]);
  const double* normal_derivative_of_field_p  = (face_is_in_omega_minus? normal_derivative_of_vnp1_plus_p[dir]  : normal_derivative_of_vnp1_minus_p[dir]);
  for (unsigned char der = 0; der < P4EST_DIM; ++der)
  {
    double signed_normal_component          = (face_is_in_omega_minus?-1.0:+1.0)*local_normal[der];
    const unsigned char local_oriented_der  = ((signed_normal_component > 0.0)? 2*der:2*der+1);
    p4est_locidx_t neighbor_face_idx;
    if(!faces_n->found_finest_face_neighbor(local_face_idx, dir, local_oriented_der, neighbor_face_idx))
    {
      P4EST_ASSERT(fabs(field_p[local_face_idx]) > threshold_dbl_max);
      return; // one of the neighbor is not a uniform fine neigbor, forget about this one and return
    }
    P4EST_ASSERT((neighbor_face_idx >=0) && (neighbor_face_idx < faces_n->num_local[dir] + faces_n->num_ghost[dir]));
    // field: may be subresolved
    get_velocity_seen_from_face(neighbor_velocity, local_face_idx, neighbor_face_idx, fine_phi_p, fine_phi_xxyyzz_p, local_oriented_der, dir, vnp1_minus_p[dir], vnp1_plus_p[dir], fine_jump_mu_grad_vdir_p, NULL, NULL);
    P4EST_ASSERT(neighbor_velocity.distance >= 0.0);
    if(neighbor_velocity.distance < dxyz_min[der]*(dxyz_min[der]/convert_to_xyz[der])) // "too close" --> avoid ill-defined terms due to inverses of very small distances, set the value equal to the direct neighbor
    {
      field_p[local_face_idx]     = neighbor_velocity.value;
      too_close_flag              = true;
    }
    else
    {
      P4EST_ASSERT(fabs(neighbor_velocity.value) < threshold_dbl_max);
      increment_field            -= fabs(signed_normal_component)*(field_p[local_face_idx]-neighbor_velocity.value)/neighbor_velocity.distance;
      dt_field                    = MIN(dt_field, neighbor_velocity.distance);
    }
  }
  if(!too_close_flag)
  {
    dt_field /= ((double) P4EST_DIM);
    field_p[local_face_idx] += dt_field*(normal_derivative_of_field_p[local_face_idx] + increment_field);
    P4EST_ASSERT(!isnan(field_p[local_face_idx]) && (fabs(field_p[local_face_idx]) < threshold_dbl_max));
  }
}

void my_p4est_two_phase_flows_t::interpolate_linearly_from_fine_nodes_to_coarse_nodes(const Vec& vv_fine, Vec& vv_coarse)
{
  PetscErrorCode ierr;
  ierr = create_node_vector_if_needed(vv_coarse, p4est_n, nodes_n); CHKERRXX(ierr);
  double *vv_coarse_p, xyz_coarse_node[P4EST_DIM];
  ierr = VecGetArray(vv_coarse, &vv_coarse_p); CHKERRXX(ierr);
  my_p4est_interpolation_nodes_t fine_node_interpolator(fine_ngbd_n);
  fine_node_interpolator.set_input(vv_fine, linear);
  for (size_t k = 0; k < ngbd_n->get_layer_size(); ++k) {
    p4est_locidx_t coarse_node_idx = ngbd_n->get_layer_node(k);
    node_xyz_fr_n(coarse_node_idx, p4est_n, nodes_n, xyz_coarse_node);
    vv_coarse_p[coarse_node_idx] = fine_node_interpolator(xyz_coarse_node);
  }
  ierr = VecGhostUpdateBegin(vv_coarse, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for (size_t k = 0; k < ngbd_n->get_local_size(); ++k) {
    p4est_locidx_t coarse_node_idx = ngbd_n->get_local_node(k);
    node_xyz_fr_n(coarse_node_idx, p4est_n, nodes_n, xyz_coarse_node);
    vv_coarse_p[coarse_node_idx] = fine_node_interpolator(xyz_coarse_node);
  }
  ierr = VecGhostUpdateEnd(vv_coarse, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecRestoreArray(vv_coarse, &vv_coarse_p); CHKERRXX(ierr);
}

void my_p4est_two_phase_flows_t::save_vtk(const char* name, const bool& export_fine_grid, const char* name_fine)
{
  PetscErrorCode ierr;
  Vec phi_coarse = NULL;
  interpolate_linearly_from_fine_nodes_to_coarse_nodes(fine_phi, phi_coarse);
  const double* phi_coarse_p = NULL;
  const double* hodge_p, *v_nodes_omega_plus_p, *v_nodes_omega_minus_p;
  ierr = VecGetArrayRead(hodge, &hodge_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(phi_coarse, &phi_coarse_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(vnp1_nodes_omega_minus, &v_nodes_omega_minus_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(vnp1_nodes_omega_plus, &v_nodes_omega_plus_p); CHKERRXX(ierr);

  my_p4est_vtk_write_all_general(p4est_n, nodes_n, ghost_n,
                                 P4EST_TRUE, P4EST_TRUE,
                                 1, /* number of VTK_POINT_DATA */
                                 0, /* number of VTK_POINT_DATA_VECTOR_BY_COMPONENTS */
                                 2, /* number of VTK_POINT_DATA_VECTOR_BLOCK */
                                 1, /* number of VTK_CELL_DATA */
                                 0, /* number of VTK_CELL_DATA_VECTOR_BY_COMPONENTS */
                                 0, /* number of VTK_CELL_DATA_VECTOR_BLOCK */
                                 name,
                                 VTK_NODE_SCALAR, "phi", phi_coarse_p,
                                 VTK_NODE_VECTOR_BLOCK, "vn_plus" , v_nodes_omega_plus_p,
                                 VTK_NODE_VECTOR_BLOCK, "vn_minus", v_nodes_omega_minus_p,
                                 VTK_CELL_DATA, "hodge", hodge_p);
  ierr = VecRestoreArrayRead(hodge, &hodge_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(phi_coarse, &phi_coarse_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(vnp1_nodes_omega_minus, &v_nodes_omega_minus_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(vnp1_nodes_omega_plus, &v_nodes_omega_plus_p); CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(phi_coarse); CHKERRXX(ierr);

  if(export_fine_grid)
  {
    P4EST_ASSERT(name_fine!=NULL);
    Vec fine_jump_mu_grad_v_comp[P4EST_DIM];
    double *fine_jump_mu_grad_v_comp_p[P4EST_DIM];
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      fine_jump_mu_grad_v_comp[dir] = NULL;
      ierr = create_node_vector_if_needed(fine_jump_mu_grad_v_comp[dir], fine_p4est_n, fine_nodes_n, P4EST_DIM); CHKERRXX(ierr);
      ierr = VecGetArray(fine_jump_mu_grad_v_comp[dir], &fine_jump_mu_grad_v_comp_p[dir]); CHKERRXX(ierr);
    }
    const double *fine_jump_mu_grad_v_p;
    ierr = VecGetArrayRead(fine_jump_mu_grad_v, &fine_jump_mu_grad_v_p); CHKERRXX(ierr);
    for (size_t k = 0; k < fine_nodes_n->indep_nodes.elem_count; ++k)
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
        for (unsigned char der = 0; der < P4EST_DIM; ++der)
          fine_jump_mu_grad_v_comp_p[dir][P4EST_DIM*k+der] = fine_jump_mu_grad_v_p[P4EST_DIM*P4EST_DIM*k+P4EST_DIM*dir+der];

    ierr = VecRestoreArrayRead(fine_jump_mu_grad_v, &fine_jump_mu_grad_v_p); CHKERRXX(ierr);

    const double* fine_phi_p, *fine_curvature_p, *fine_normal_p;
    ierr = VecGetArrayRead(fine_normal, &fine_normal_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(fine_curvature, &fine_curvature_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(fine_phi, &fine_phi_p); CHKERRXX(ierr);
    my_p4est_vtk_write_all_general(fine_p4est_n, fine_nodes_n, fine_ghost_n,
                                   P4EST_FALSE, P4EST_FALSE,
                                   2, /* number of VTK_POINT_DATA */
                                   0, /* number of VTK_POINT_DATA_VECTOR_BY_COMPONENTS */
                                   1+P4EST_DIM, /* number of VTK_POINT_DATA_VECTOR_BLOCK */
                                   0, /* number of VTK_CELL_DATA */
                                   0, /* number of VTK_CELL_DATA_VECTOR_BY_COMPONENTS */
                                   0, /* number of VTK_CELL_DATA_VECTOR_BLOCK */
                                   name_fine,
                                   VTK_NODE_SCALAR, "phi", fine_phi_p,
                                   VTK_NODE_SCALAR, "curvature", fine_curvature_p,
                                   VTK_NODE_VECTOR_BLOCK, "normal", fine_normal_p,
                                   VTK_NODE_VECTOR_BLOCK, "jump mu grad u", fine_jump_mu_grad_v_comp_p[0],
                                   VTK_NODE_VECTOR_BLOCK, "jump mu grad v", fine_jump_mu_grad_v_comp_p[1]
    #ifdef P4_TO_P8
        , VTK_NODE_VECTOR_BLOCK, "jump mu grad w", fine_jump_mu_grad_v_comp_p[2]
    #endif
        );
    ierr = VecRestoreArrayRead(fine_phi, &fine_phi_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(fine_curvature, &fine_curvature_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(fine_normal, &fine_normal_p); CHKERRXX(ierr);
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      ierr = VecRestoreArray(fine_jump_mu_grad_v_comp[dir], &fine_jump_mu_grad_v_comp_p[dir]); CHKERRXX(ierr);
      ierr = delete_and_nullify_vector(fine_jump_mu_grad_v_comp[dir]); CHKERRXX(ierr);
      fine_jump_mu_grad_v_comp[dir] = NULL;
    }
  }
}

void my_p4est_two_phase_flows_t::trajectory_from_all_faces_two_phases(p4est_t *p4est_n, my_p4est_faces_t *faces_n, my_p4est_node_neighbors_t *ngbd_nm1, my_p4est_node_neighbors_t *ngbd_n, const double *fine_phi_p,
                                                                      Vec vnm1_nodes_omega_minus, Vec vnm1_nodes_omega_minus_xxyyzz,
                                                                      Vec vnm1_nodes_omega_plus, Vec vnm1_nodes_omega_plus_xxyyzz,
                                                                      Vec vn_nodes_omega_minus, Vec vn_nodes_omega_minus_xxyyzz,
                                                                      Vec vn_nodes_omega_plus, Vec vn_nodes_omega_plus_xxyyzz,
                                                                      double dt_nm1, double dt_n,
                                                                      std::vector<double> xyz_n[P4EST_DIM],
                                                                      std::vector<double> xyz_nm1[P4EST_DIM])
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_two_phase_trajectory_of_all_faces, 0, 0, 0, 0); CHKERRXX(ierr);

  P4EST_ASSERT(((vnm1_nodes_omega_minus_xxyyzz==NULL) && (vnm1_nodes_omega_plus_xxyyzz==NULL) && (vn_nodes_omega_minus_xxyyzz==NULL) && (vn_nodes_omega_plus_xxyyzz==NULL)) ||
               ((vnm1_nodes_omega_minus_xxyyzz!=NULL) && (vnm1_nodes_omega_plus_xxyyzz!=NULL) && (vn_nodes_omega_minus_xxyyzz!=NULL) && (vn_nodes_omega_plus_xxyyzz!=NULL)));

  /* first find the velocity at the np1 points */
  my_p4est_interpolation_nodes_t interp_np1_omega_plus(ngbd_n), interp_np1_omega_minus(ngbd_n);
  bool use_second_derivatives = ((vnm1_nodes_omega_minus_xxyyzz!=NULL) && (vnm1_nodes_omega_plus_xxyyzz!=NULL) && (vn_nodes_omega_minus_xxyyzz!=NULL) && (vn_nodes_omega_plus_xxyyzz!=NULL));

  p4est_locidx_t serialized_offset=0;
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    serialized_offset += faces_n->num_local[dir];
  vector<double> xyz_np1;         xyz_np1.resize(P4EST_DIM*serialized_offset);
  vector<double> vnp1;            vnp1.resize(P4EST_DIM*serialized_offset);
  vector<bool> from_omega_minus;  from_omega_minus.resize(serialized_offset);
  serialized_offset=0;
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    for (p4est_locidx_t f_idx = 0; f_idx < faces_n->num_local[dir]; ++f_idx) {
      faces_n->xyz_fr_f(f_idx, dir, &xyz_np1[P4EST_DIM*serialized_offset+P4EST_DIM*f_idx]);
      from_omega_minus[serialized_offset+f_idx] = is_face_in_omega_minus(f_idx, dir, fine_phi_p);
      if(from_omega_minus[serialized_offset+f_idx])
        interp_np1_omega_minus.add_point(serialized_offset+f_idx, &xyz_np1[P4EST_DIM*serialized_offset+P4EST_DIM*f_idx]);
      else
        interp_np1_omega_plus.add_point(serialized_offset+f_idx, &xyz_np1[P4EST_DIM*serialized_offset+P4EST_DIM*f_idx]);
    }
    serialized_offset += faces_n->num_local[dir];
  }
  if (use_second_derivatives)
  {
    interp_np1_omega_minus.set_input(vn_nodes_omega_minus, vn_nodes_omega_minus_xxyyzz, quadratic, P4EST_DIM);
    interp_np1_omega_plus.set_input(vn_nodes_omega_plus, vn_nodes_omega_plus_xxyyzz, quadratic, P4EST_DIM);
  }
  else
  {
    interp_np1_omega_minus.set_input(vn_nodes_omega_minus, quadratic, P4EST_DIM);
    interp_np1_omega_plus.set_input(vn_nodes_omega_plus, quadratic, P4EST_DIM);
  }
  interp_np1_omega_minus.interpolate(vnp1.data());
  interp_np1_omega_plus.interpolate(vnp1.data());
  /* find xyz_star */
  my_p4est_interpolation_nodes_t interp_nm1_omega_minus(ngbd_nm1), interp_nm1_omega_plus(ngbd_nm1);
  my_p4est_interpolation_nodes_t interp_n_omega_minus  (ngbd_n  ), interp_n_omega_plus  (ngbd_n  );
  serialized_offset=0;
  double xyz_star[P4EST_DIM];
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    for (p4est_locidx_t f_idx = 0; f_idx < faces_n->num_local[dir]; ++f_idx) {
      for (unsigned char comp = 0; comp < P4EST_DIM; ++comp)
        xyz_star[comp] = xyz_np1[P4EST_DIM*serialized_offset+P4EST_DIM*f_idx+comp] - 0.5*dt_n*vnp1[P4EST_DIM*serialized_offset+P4EST_DIM*f_idx+comp];
      clip_in_domain(xyz_star, xyz_min, xyz_max, periodic);
      if(from_omega_minus[serialized_offset+f_idx])
      {
        interp_nm1_omega_minus.add_point(serialized_offset+f_idx, xyz_star);
        interp_n_omega_minus.add_point(serialized_offset+f_idx, xyz_star);
      }
      else
      {
        interp_nm1_omega_plus.add_point(serialized_offset+f_idx, xyz_star);
        interp_n_omega_plus.add_point(serialized_offset+f_idx, xyz_star);
      }
    }
    serialized_offset += faces_n->num_local[dir];
  }

  /* compute the velocities at the intermediate point */
  std::vector<double> vn_star;    vn_star.resize(P4EST_DIM*serialized_offset);
  std::vector<double> vnm1_star;  vnm1_star.resize(P4EST_DIM*serialized_offset);
  if(use_second_derivatives)
  {
    interp_nm1_omega_minus.set_input(vnm1_nodes_omega_minus, vnm1_nodes_omega_minus_xxyyzz, quadratic, P4EST_DIM);
    interp_nm1_omega_plus.set_input(vnm1_nodes_omega_plus, vnm1_nodes_omega_plus_xxyyzz, quadratic, P4EST_DIM);
    interp_n_omega_minus.set_input(vn_nodes_omega_minus, vn_nodes_omega_minus_xxyyzz, quadratic, P4EST_DIM);
    interp_n_omega_plus.set_input(vn_nodes_omega_plus, vn_nodes_omega_plus_xxyyzz, quadratic, P4EST_DIM);
  }
  else
  {
    interp_nm1_omega_minus.set_input(vnm1_nodes_omega_minus, quadratic, P4EST_DIM);
    interp_nm1_omega_plus.set_input(vnm1_nodes_omega_plus, quadratic, P4EST_DIM);
    interp_n_omega_minus.set_input(vn_nodes_omega_minus, quadratic, P4EST_DIM);
    interp_n_omega_plus.set_input(vn_nodes_omega_plus, quadratic, P4EST_DIM);
  }
  interp_nm1_omega_minus.interpolate(vnm1_star.data());
  interp_nm1_omega_plus.interpolate(vnm1_star.data());
  interp_n_omega_minus.interpolate(vn_star.data());
  interp_n_omega_plus.interpolate(vn_star.data());
  interp_nm1_omega_minus.clear();
  interp_nm1_omega_plus.clear();
  interp_n_omega_minus.clear();
  interp_n_omega_plus.clear();

  /* now find the departure point at time n */
  serialized_offset=0;
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    xyz_n[dir].resize(P4EST_DIM*faces_n->num_local[dir]);
    for (p4est_locidx_t f_idx = 0; f_idx < faces_n->num_local[dir]; ++f_idx) {
      for (unsigned char comp = 0; comp < P4EST_DIM; ++comp)
        xyz_n[dir][P4EST_DIM*f_idx+comp] = xyz_np1[P4EST_DIM*serialized_offset+P4EST_DIM*f_idx+comp] - dt_n*((1.0+.5*dt_n/dt_nm1)*vn_star[P4EST_DIM*serialized_offset+P4EST_DIM*f_idx+comp] - .5*(dt_n/dt_nm1)*vnm1_star[P4EST_DIM*serialized_offset+P4EST_DIM*f_idx+comp]);
      clip_in_domain(&xyz_n[dir][P4EST_DIM*f_idx], xyz_min, xyz_max, periodic);
    }
    serialized_offset += faces_n->num_local[dir];
  }

  // EXTRA STUFF FOR FINDING xyz_nm1 ONLY (for second-order bdf advection terms, for instance)
  if(xyz_nm1 != NULL)
  {
    /* proceed similarly for the departure point at time nm1 */
    serialized_offset = 0;
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      for (p4est_locidx_t f_idx = 0; f_idx < faces_n->num_local[dir]; ++f_idx) {
        for (unsigned char comp = 0; comp < P4EST_DIM; ++comp)
        {
          xyz_star[comp] = xyz_np1[P4EST_DIM*serialized_offset+P4EST_DIM*f_idx+comp] - 0.5*(dt_n + dt_nm1)*vnp1[P4EST_DIM*serialized_offset+P4EST_DIM*f_idx+comp];
          if      (is_periodic(p4est_n,comp) && xyz_star[comp]<xyz_min[comp]) xyz_star[comp] += xyz_max[comp]-xyz_min[comp];
          else if (is_periodic(p4est_n,comp) && xyz_star[comp]>xyz_max[comp]) xyz_star[comp] -= xyz_max[comp]-xyz_min[comp];
          else                                                                xyz_star[comp] = MAX(xyz_min[comp], MIN(xyz_max[comp], xyz_star[comp]));
        }
        if(from_omega_minus[serialized_offset+f_idx])
        {
          interp_nm1_omega_minus.add_point(serialized_offset+f_idx, xyz_star);
          interp_n_omega_minus.add_point(serialized_offset+f_idx, xyz_star);
        }
        else
        {
          interp_nm1_omega_plus.add_point(serialized_offset+f_idx, xyz_star);
          interp_n_omega_plus.add_point(serialized_offset+f_idx, xyz_star);
        }
      }
      serialized_offset += faces_n->num_local[dir];
    }

    /* compute the velocities at the intermediate point */
    if(use_second_derivatives)
    {
      interp_nm1_omega_minus.set_input(vnm1_nodes_omega_minus, vnm1_nodes_omega_minus_xxyyzz, quadratic, P4EST_DIM);
      interp_nm1_omega_plus.set_input(vnm1_nodes_omega_plus, vnm1_nodes_omega_plus_xxyyzz, quadratic, P4EST_DIM);
    }
    else
    {
      interp_nm1_omega_minus.set_input(vnm1_nodes_omega_minus, quadratic, P4EST_DIM);
      interp_nm1_omega_plus.set_input(vnm1_nodes_omega_plus, quadratic, P4EST_DIM);
    }
    interp_nm1_omega_minus.interpolate(vnm1_star.data());
    interp_nm1_omega_plus.interpolate(vnm1_star.data());
    if(use_second_derivatives)
    {
      interp_n_omega_minus.set_input(vn_nodes_omega_minus, vn_nodes_omega_minus_xxyyzz, quadratic, P4EST_DIM);
      interp_n_omega_plus.set_input(vn_nodes_omega_plus, vn_nodes_omega_plus_xxyyzz, quadratic, P4EST_DIM);
    }
    else
    {
      interp_n_omega_minus.set_input(vn_nodes_omega_minus, quadratic, P4EST_DIM);
      interp_n_omega_plus.set_input(vn_nodes_omega_plus, quadratic, P4EST_DIM);
    }
    interp_n_omega_minus.interpolate(vn_star.data());
    interp_n_omega_plus.interpolate(vn_star.data());
    interp_nm1_omega_minus.clear();
    interp_nm1_omega_plus.clear();
    interp_n_omega_minus.clear();
    interp_n_omega_plus.clear();

    /* now find the departure point at time nm1 */
    serialized_offset=0;
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
      xyz_nm1[dir].resize(P4EST_DIM*faces_n->num_local[dir]);
      for (p4est_locidx_t f_idx = 0; f_idx < faces_n->num_local[dir]; ++f_idx) {
        for (unsigned char comp = 0; comp < P4EST_DIM; ++comp)
        {
          xyz_nm1[dir][P4EST_DIM*f_idx+comp] = xyz_np1[P4EST_DIM*serialized_offset+P4EST_DIM*f_idx+comp] - (dt_n+dt_nm1)*((1.0+.5*dt_n/dt_nm1)*vn_star[P4EST_DIM*serialized_offset+P4EST_DIM*f_idx+comp] - .5*(dt_n/dt_nm1)*vnm1_star[P4EST_DIM*serialized_offset+P4EST_DIM*f_idx+comp]);
          if      (is_periodic(p4est_n,comp) && xyz_nm1[dir][P4EST_DIM*f_idx+comp]<xyz_min[comp]) xyz_nm1[dir][P4EST_DIM*f_idx+comp] += xyz_max[comp]-xyz_min[comp];
          else if (is_periodic(p4est_n,comp) && xyz_nm1[dir][P4EST_DIM*f_idx+comp]>xyz_max[comp]) xyz_nm1[dir][P4EST_DIM*f_idx+comp] -= xyz_max[comp]-xyz_min[comp];
          else                                                                                    xyz_nm1[dir][P4EST_DIM*f_idx+comp] = MAX(xyz_min[comp], MIN(xyz_max[comp], xyz_nm1[dir][P4EST_DIM*f_idx+comp]));
        }
      }
      serialized_offset += faces_n->num_local[dir];
    }
  }

  ierr = PetscLogEventEnd(log_my_p4est_two_phase_trajectory_of_all_faces, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_two_phase_flows_t::set_interface_velocity()
{
  PetscErrorCode ierr;
  ierr = create_node_vector_if_needed(interface_velocity_np1, p4est_n, nodes_n, P4EST_DIM); CHKERRXX(ierr);
  double *interface_velocity_np1_p            = NULL;
  const double *fluid_velocity_omega_minus_p  = NULL;
  const double *fluid_velocity_omega_plus_p   = NULL;
  const double *fine_mass_flux_p              = NULL;
  const double *fine_normal_p                 = NULL;
  ierr = VecGetArray(interface_velocity_np1, &interface_velocity_np1_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(vnp1_nodes_omega_plus, &fluid_velocity_omega_plus_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(vnp1_nodes_omega_minus, &fluid_velocity_omega_minus_p); CHKERRXX(ierr);
  if(fine_mass_flux!=NULL){
    ierr = VecGetArrayRead(fine_mass_flux, &fine_mass_flux_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(fine_normal, &fine_normal_p); CHKERRXX(ierr);
  }
  double xyz_node[P4EST_DIM];
  for (size_t k = 0; k < ngbd_n->get_layer_size(); ++k) {
    p4est_locidx_t n = ngbd_n->get_layer_node(k);
    node_xyz_fr_n(n, p4est_n, nodes_n, xyz_node);
    domain_side side_to_choose = ((*interp_phi)(xyz_node) <= 0.0)? OMEGA_MINUS : OMEGA_PLUS;
    p4est_locidx_t fine_node_idx;
    size_t position = P4EST_DIM*n;
//    bool u_side_is_good = true;
//    for (unsigned char dir = 0; u_side_is_good && (dir < P4EST_DIM); ++dir)
//      u_side_is_good = !ISNAN(underlined_fluid_velocity_p[position+dir]);
//    domain_side side_to_choose = u_side_is_good? u_side : o_side;
//#ifdef DEBUG
//    bool side_to_choose_is_good = true;
//    for (unsigned char dir = 0; side_to_choose_is_good && (dir < P4EST_DIM); ++dir)
//      side_to_choose_is_good = !ISNAN((side_to_choose==u_side)? underlined_fluid_velocity_p[position+dir] : overlined_fluid_velocity_p[position+dir]);
//    P4EST_ASSERT(side_to_choose_is_good);
//#endif
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      interface_velocity_np1_p[position+dir] = side_to_choose==OMEGA_MINUS ? fluid_velocity_omega_minus_p[position+dir] : fluid_velocity_omega_plus_p[position+dir];
    if(fine_mass_flux!=NULL && get_fine_node_idx_node(n, fine_node_idx))
    {
      size_t fine_position = P4EST_DIM*fine_node_idx;
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
        interface_velocity_np1_p[position+dir] -= (fine_mass_flux_p[fine_position+dir]*fine_normal_p[fine_position+dir]/(side_to_choose==OMEGA_MINUS?rho_minus:rho_plus));
    }
  }
  ierr = VecGhostUpdateBegin(interface_velocity_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for (size_t k = 0; k < ngbd_n->get_local_size(); ++k) {
    p4est_locidx_t n = ngbd_n->get_local_node(k);
    node_xyz_fr_n(n, p4est_n, nodes_n, xyz_node);
    domain_side side_to_choose = ((*interp_phi)(xyz_node) <= 0.0)? OMEGA_MINUS : OMEGA_PLUS;
    p4est_locidx_t fine_node_idx;
    size_t position = P4EST_DIM*n;
//    bool u_side_is_good = true;
//    for (unsigned char dir = 0; u_side_is_good && (dir < P4EST_DIM); ++dir)
//      u_side_is_good = !ISNAN(underlined_fluid_velocity_p[position+dir]);
//    domain_side side_to_choose = u_side_is_good? u_side : o_side;
//#ifdef DEBUG
//    bool side_to_choose_is_good = true;
//    for (unsigned char dir = 0; side_to_choose_is_good && (dir < P4EST_DIM); ++dir)
//      side_to_choose_is_good = !ISNAN((side_to_choose==u_side)? underlined_fluid_velocity_p[position+dir] : overlined_fluid_velocity_p[position+dir]);
//    P4EST_ASSERT(side_to_choose_is_good);
//#endif
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
      interface_velocity_np1_p[position+dir] = (side_to_choose==OMEGA_MINUS? fluid_velocity_omega_minus_p[position+dir] : fluid_velocity_omega_plus_p[position+dir]);
    if(fine_mass_flux!=NULL && get_fine_node_idx_node(n, fine_node_idx))
    {
      size_t fine_position = P4EST_DIM*fine_node_idx;
      for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
        interface_velocity_np1_p[position+dir] -= (fine_mass_flux_p[fine_position+dir]*fine_normal_p[fine_position+dir]/(side_to_choose==OMEGA_MINUS?rho_minus:rho_plus));
    }
  }
  ierr = VecGhostUpdateEnd(interface_velocity_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = create_node_vector_if_needed(interface_velocity_np1_xxyyzz, p4est_n, nodes_n, P4EST_DIM*P4EST_DIM); CHKERRXX(ierr);
  ngbd_n->second_derivatives_central(interface_velocity_np1, interface_velocity_np1_xxyyzz, P4EST_DIM);

  if(fine_mass_flux!=NULL){
    ierr = VecRestoreArrayRead(fine_normal, &fine_normal_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(fine_mass_flux, &fine_mass_flux_p); CHKERRXX(ierr);
  }
  ierr = VecRestoreArrayRead(vnp1_nodes_omega_plus, &fluid_velocity_omega_plus_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(vnp1_nodes_omega_minus, &fluid_velocity_omega_minus_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(interface_velocity_np1, &interface_velocity_np1_p); CHKERRXX(ierr);
}

void my_p4est_two_phase_flows_t::advect_interface(p4est_t *fine_p4est_np1, p4est_nodes_t *fine_nodes_np1, Vec fine_phi_np1,
                                                  p4est_nodes_t *known_nodes, Vec known_phi_np1)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_two_phase_advect_interface, 0, 0, 0, 0); CHKERRXX(ierr);
  P4EST_ASSERT(known_nodes!=NULL);

  my_p4est_interpolation_nodes_t interp_n(ngbd_nm1); // yes, it's normal: we are in the process of advancing time...
  my_p4est_interpolation_nodes_t interp_np1(ngbd_n); // yes, it's normal: we are in the process of advancing time...

  std::vector<double> v_tmp_n;
  std::vector<double> v_tmp_np1;
  std::vector<size_t> already_known(0);
  p4est_locidx_t origin_local_idx;
  const p4est_quadrant_t *node = NULL;
  const double *known_phi_np1_p = NULL;
  double *fine_phi_np1_p = NULL;
  if(known_phi_np1!=NULL)
  {
    ierr = VecGetArrayRead(known_phi_np1, &known_phi_np1_p); CHKERRXX(ierr);
    ierr = VecGetArray(fine_phi_np1, &fine_phi_np1_p); CHKERRXX(ierr);
  }

//  size_t n_to_show = -1;

  /* find the velocity field at time np1 */
  size_t to_compute = 0;
  for (size_t n=0; n < fine_nodes_np1->indep_nodes.elem_count; ++n)
  {
    if(known_phi_np1!=NULL)
    {
      node = ((const p4est_quadrant_t*) sc_array_index(&fine_nodes_np1->indep_nodes, n));
      P4EST_ASSERT (p4est_quadrant_is_node (node, 1));
    }
    if((node != NULL) && index_of_node(node, known_nodes, origin_local_idx))
    {
      fine_phi_np1_p[n] = known_phi_np1_p[origin_local_idx];
      already_known.push_back(n);
    }
    else
    {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, fine_p4est_np1, fine_nodes_np1, xyz);
//      n_to_show = (fabs(xyz[0]-0.6660156407137385) < 0.0001 && fabs(xyz[1]-0.2617187329960994) < 0.0001)? to_compute: n_to_show;
      interp_np1.add_point(to_compute++, xyz);
    }
  }
  P4EST_ASSERT(to_compute+already_known.size() == fine_nodes_np1->indep_nodes.elem_count);

  v_tmp_np1.resize(P4EST_DIM*to_compute);
  interp_np1.set_input(interface_velocity_np1, interface_velocity_np1_xxyyzz, quadratic, P4EST_DIM);
  interp_np1.interpolate(v_tmp_np1.data());
  interp_np1.clear();

  /* now find x_star */
  size_t known_idx = 0;
  to_compute = 0;
  for (size_t n=0; n < fine_nodes_np1->indep_nodes.elem_count; ++n)
  {
    if((known_phi_np1!=NULL) && (n == already_known[known_idx]))
    {
      known_idx++;
      continue;
    }
    double xyz_star[P4EST_DIM];
    node_xyz_fr_n(n, fine_p4est_np1, fine_nodes_np1, xyz_star);
    size_t position = P4EST_DIM*to_compute;
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir)
    {
      xyz_star[dir] -= .5*dt_n*v_tmp_np1[position+dir];
//      if(to_compute==n_to_show)
//      {
//        std::cout << "vstar[" << (int) dir << "] = " << v_tmp_np1[position+dir] << std::endl;
//        std::cout << "xyzstar[" << (int) dir << "] = " << xyz_star[dir] << std::endl;
//      }
    }
    clip_in_domain(xyz_star, xyz_min, xyz_max, periodic);

    interp_np1. add_point(to_compute, xyz_star);
    interp_n.   add_point(to_compute, xyz_star);
    to_compute++;
  }
  P4EST_ASSERT(to_compute+known_idx == fine_nodes_np1->indep_nodes.elem_count);

  /* interpolate vnm1 */
  v_tmp_n.resize(P4EST_DIM*to_compute);
  interp_n.set_input(interface_velocity_n, interface_velocity_n_xxyyzz, quadratic, P4EST_DIM);
  interp_n.interpolate(v_tmp_n.data());
  interp_n.clear();
  interp_np1.set_input(interface_velocity_np1, interface_velocity_np1_xxyyzz, quadratic, P4EST_DIM);
  interp_np1.interpolate(v_tmp_np1.data());
  interp_np1.clear();

  /* finally, find the backtracing value */
  /* find the departure node via backtracing */
  double v_star[P4EST_DIM];
  known_idx = 0;
  to_compute = 0;
  for (size_t n=0; n < fine_nodes_np1->indep_nodes.elem_count; ++n)
  {
    if((known_phi_np1!=NULL) && (n == already_known[known_idx]))
    {
      known_idx++;
      continue;
    }
    size_t position = P4EST_DIM*to_compute;
    double xyz_d[P4EST_DIM]; node_xyz_fr_n(n, fine_p4est_np1, fine_nodes_np1, xyz_d);
    for (unsigned char dir = 0; dir < P4EST_DIM; ++dir){
      v_star[dir] = (1.0 + 0.5*dt_n/dt_nm1)*v_tmp_np1[position+dir] - 0.5*dt_n/dt_nm1*v_tmp_n[position+dir];
      xyz_d[dir] -= dt_n*v_star[dir];
//      if(to_compute==n_to_show)
//      {
//        std::cout << "vd[" << (int) dir << "] = " << v_star[position+dir] << std::endl;
//        std::cout << "xyz_d[" << (int) dir << "] = " << xyz_d[dir] << std::endl;
//      }
    }

    clip_in_domain(xyz_d, xyz_min, xyz_max, periodic);

    interp_phi->add_point(n, xyz_d);
    to_compute++;
  }
  P4EST_ASSERT(to_compute+known_idx == fine_nodes_np1->indep_nodes.elem_count);

  interp_phi->interpolate(fine_phi_np1);
  interp_phi->clear(); // to clear the buffers
  // reset inputs
  if(fine_phi_xxyyzz!=NULL)
    interp_phi->set_input(fine_phi, fine_phi_xxyyzz, quadratic_non_oscillatory);
  else
    interp_phi->set_input(fine_phi, linear);

  ierr = PetscLogEventEnd(log_my_p4est_two_phase_advect_interface, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_two_phase_flows_t::update_from_tn_to_tnp1(const unsigned int &nnn)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_two_phase_advect_update, 0, 0, 0, 0); CHKERRXX(ierr);

  if(!dt_updated)
    compute_dt();
  dt_updated = false;

  set_interface_velocity();
  if(interface_velocity_n == NULL && interface_velocity_n_xxyyzz == NULL)
  {
    ierr = create_node_vector_if_needed(interface_velocity_n, p4est_nm1, nodes_nm1, P4EST_DIM);
    ierr = create_node_vector_if_needed(interface_velocity_n_xxyyzz, p4est_nm1, nodes_nm1, P4EST_DIM*P4EST_DIM);
    Vec loc_interface_velocity_n, loc_interface_velocity_n_xxyyzz;
    ierr = VecGhostGetLocalForm(interface_velocity_n, &loc_interface_velocity_n); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(interface_velocity_n_xxyyzz, &loc_interface_velocity_n_xxyyzz); CHKERRXX(ierr);
    ierr = VecSet(loc_interface_velocity_n, 0.0); CHKERRXX(ierr);
    ierr = VecSet(loc_interface_velocity_n_xxyyzz, 0.0); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(interface_velocity_n_xxyyzz, &loc_interface_velocity_n_xxyyzz); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(interface_velocity_n, &loc_interface_velocity_n); CHKERRXX(ierr);
  }

//  {
//    PetscErrorCode ierr;
//    const double *interface_velocity_p;
//    ierr = VecGetArrayRead(interface_velocity_np1, &interface_velocity_p); CHKERRXX(ierr);

//    my_p4est_vtk_write_all_general(p4est_n, nodes_n, ghost_n,
//                                   P4EST_TRUE, P4EST_TRUE,
//                                   0, /* number of VTK_POINT_DATA */
//                                   0, /* number of VTK_POINT_DATA_VECTOR_BY_COMPONENTS */
//                                   1, /* number of VTK_POINT_DATA_VECTOR_BLOCK */
//                                   0, /* number of VTK_CELL_DATA */
//                                   0, /* number of VTK_CELL_DATA_VECTOR_BY_COMPONENTS */
//                                   0, /* number of VTK_CELL_DATA_VECTOR_BLOCK */
//                                   ("/home/regan/workspace/projects/two_phase_flow/sharp_advection/interface_velocity_" + std::to_string(nnn)).c_str(),
//                                   VTK_NODE_VECTOR_BLOCK, "interface_velocity" , interface_velocity_p);
//    ierr = VecRestoreArrayRead(interface_velocity_np1, &interface_velocity_p); CHKERRXX(ierr);
//  }


  // find the np1 computational grid
  splitting_criteria_computational_grid_two_phase_t criterion_computational_grid(this);
  // if max_lvl==min_lvl, the computational grid is uniform and we don't have to worry about adapting it, so define
  bool iterative_grid_update_converged = (criterion_computational_grid.max_lvl == criterion_computational_grid.min_lvl);
  /* initialize the new forest */
  p4est_t *p4est_np1                  = iterative_grid_update_converged? p4est_n : p4est_copy(p4est_n, P4EST_FALSE); // no need to copy if no need to refine/coarsen, otherwise very efficient copy
  p4est_np1->connectivity             = p4est_n->connectivity; // connectivity is not duplicated by p4est_copy, the pointer (i.e. the memory-address) of connectivity seems to be copied from my understanding of the source file of p4est_copy, but I feel this is a bit safer [Raphael Egan]
  p4est_nodes_t *nodes_np1            = nodes_n;
  p4est_ghost_t *ghost_np1            = ghost_n;
  my_p4est_hierarchy_t *hierarchy_np1 = hierarchy_n;
  my_p4est_node_neighbors_t *ngbd_np1 = ngbd_n;
  Vec vorticities_np1                 = vorticities;
  Vec phi_np1                         = NULL; // we don't know it yet, we'll need to advect it
  my_p4est_interpolation_nodes_t interp_nodes(ngbd_n);
  unsigned int iter=0;
  while(!iterative_grid_update_converged)
  {
    p4est_nodes_t *previous_nodes_np1 = nodes_np1;
    Vec previous_phi_np1              = phi_np1;
    Vec previous_vorticities_np1      = vorticities_np1;
    /* ---   FIND THE NEXT ADAPTIVE GRID   --- */
    /* For the very first iteration of the grid-update procedure, p4est_np1 is a
     * simple pure copy of p4est_n, so no node creation nor data interpolation is
     * required. Hence the "if(iter>0)..." statements */

    if(iter > 0)
    {
      // maybe no longer properly balanced so partition...
      my_p4est_partition(p4est_np1, P4EST_FALSE, NULL);
      // get nodes_np1
      nodes_np1 = my_p4est_nodes_new(p4est_np1, NULL);
    }
    // reset phi_np1
    phi_np1 = NULL;
    ierr = create_node_vector_if_needed(phi_np1, p4est_np1, nodes_np1); CHKERRXX(ierr);
    P4EST_ASSERT(((iter > 0) && (previous_phi_np1 != NULL))  || ((iter == 0) && (previous_phi_np1 == NULL)));

    // advect phi
    advect_interface(p4est_np1, nodes_np1, phi_np1,
                     previous_nodes_np1, previous_phi_np1); // limit the workload: use what you already know if you know it!

    // get vorticity_np1
    if(iter > 0){
      // reset vorticity_np1 if needed
      vorticities_np1 = NULL;
      ierr = create_node_vector_if_needed(vorticities_np1, p4est_np1, nodes_np1, 2); CHKERRXX(ierr);
      const double *previous_vorticities_np1_p;
      double *vorticities_np1_p;
      ierr = VecGetArrayRead(previous_vorticities_np1, &previous_vorticities_np1_p); CHKERRXX(ierr);
      ierr = VecGetArray(vorticities_np1, &vorticities_np1_p); CHKERRXX(ierr);

      interp_nodes.clear();
      for(size_t n=0; n < nodes_np1->indep_nodes.elem_count; ++n)
      {
        const p4est_quadrant_t *node = ((const p4est_quadrant_t*) sc_array_index(&nodes_np1->indep_nodes, n));
        P4EST_ASSERT (p4est_quadrant_is_node (node, 1));
        p4est_locidx_t origin_idx;
        if(index_of_node(node, previous_nodes_np1, origin_idx))
          for (unsigned char k = 0; k < 2; ++k)
            vorticities_np1_p[2*n+k] = previous_vorticities_np1_p[2*origin_idx+k];
        else
        {
          double xyz[P4EST_DIM];
          node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);
          interp_nodes.add_point(n, xyz);
        }
      }
      interp_nodes.set_input(vorticities, linear, 2);
      interp_nodes.interpolate(vorticities_np1);

      ierr = VecRestoreArrayRead(previous_vorticities_np1, &previous_vorticities_np1_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(vorticities_np1, &vorticities_np1_p); CHKERRXX(ierr);
    }


//    {
//      PetscErrorCode ierr;
//      const double *phi_coarse_p = NULL;
//      const double *vorticities_p;
//      ierr = VecGetArrayRead(phi_np1, &phi_coarse_p); CHKERRXX(ierr);
//      ierr = VecGetArrayRead(vorticities_np1, &vorticities_p); CHKERRXX(ierr);

//      my_p4est_vtk_write_all_general(p4est_np1, nodes_np1, NULL,
//                                     P4EST_FALSE, P4EST_FALSE,
//                                     1, /* number of VTK_POINT_DATA */
//                                     0, /* number of VTK_POINT_DATA_VECTOR_BY_COMPONENTS */
//                                     1, /* number of VTK_POINT_DATA_VECTOR_BLOCK */
//                                     0, /* number of VTK_CELL_DATA */
//                                     0, /* number of VTK_CELL_DATA_VECTOR_BY_COMPONENTS */
//                                     0, /* number of VTK_CELL_DATA_VECTOR_BLOCK */
//                                     ("/home/regan/workspace/projects/two_phase_flow/sharp_advection/before_update_" + std::to_string(iter)).c_str(),
//                                     VTK_NODE_SCALAR, "phi_np1" , phi_coarse_p,
//                                     VTK_NODE_VECTOR_BLOCK, "vorticities" , vorticities_p);
//      ierr = VecRestoreArrayRead(phi_np1, &phi_coarse_p); CHKERRXX(ierr);
//      ierr = VecRestoreArrayRead(vorticities_np1, &vorticities_p); CHKERRXX(ierr);
//    }

    // update the grid
    iterative_grid_update_converged = !criterion_computational_grid.refine_and_coarsen(p4est_np1, nodes_np1, phi_np1, vorticities_np1);

    iter++;

    if(previous_phi_np1!=NULL){
      ierr = delete_and_nullify_vector(previous_phi_np1); CHKERRXX(ierr); }
    if(previous_nodes_np1!=nodes_n)
      p4est_nodes_destroy(previous_nodes_np1);
    if(previous_vorticities_np1!=vorticities){
      ierr = delete_and_nullify_vector(previous_vorticities_np1); CHKERRXX(ierr); }

    if(iter > ((unsigned int) 2+criterion_computational_grid.max_lvl-criterion_computational_grid.min_lvl)) // increase the rhs by one to account for the very first step that used to be out of the loop, [Raphael]
    {
      ierr = PetscPrintf(p4est_n->mpicomm, "ooops ... the grid update did not converge\n"); CHKERRXX(ierr);
      break;
    }
  }

  if(vorticities_np1!=vorticities){
    ierr = delete_and_nullify_vector(vorticities_np1); CHKERRXX(ierr); }
  // we do not need the vorticities anymore
  ierr = delete_and_nullify_vector(vorticities); CHKERRXX(ierr);

  // Save what we already know from the advection of the levelset (in case we know anything) in
  // order to limit workload later on, regarding the adaptation for the interface-capturing grid
  p4est_nodes_t *fine_nodes_np1 = nodes_np1;
  Vec fine_phi_np1              = phi_np1; phi_np1 = NULL;

  // Finalize the computational grid np1 (if needed):
  if(p4est_np1!=p4est_n)
  {
    // Balance the grid and repartition
    p4est_balance(p4est_np1, P4EST_CONNECT_FULL, NULL);
    my_p4est_partition(p4est_np1, P4EST_FALSE, NULL);
    /* Get the ghost cells at time np1, */
    ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
    my_p4est_ghost_expand(p4est_np1, ghost_np1);
    nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);
    hierarchy_np1 = new my_p4est_hierarchy_t(p4est_np1, ghost_np1, brick);
    ngbd_np1 = new my_p4est_node_neighbors_t(hierarchy_np1, nodes_np1);
    ngbd_np1->init_neighbors();
  }
  // we are done with the computational grid

  // now, find the np1 interface-capturing grid
  splitting_criteria_t* data_fine = (splitting_criteria_t*) fine_p4est_n->user_pointer;
  splitting_criteria_tag_t criterion_fine_grid(data_fine);
  P4EST_ASSERT(criterion_fine_grid.max_lvl > criterion_computational_grid.max_lvl);
  P4EST_ASSERT(criterion_fine_grid.max_lvl > criterion_fine_grid.min_lvl);
  /* initialize the new forest */
  // if p4est_np1 has changed, we start the procedure from p4est_np1, we'll coarsen as much as possible and refine as needed but we keep always things local
  // if not, fine_p4est_n is still valid and is a better initial guess
  p4est_t *fine_p4est_np1                   = (p4est_np1!=p4est_n)? p4est_copy(p4est_np1, P4EST_FALSE) : p4est_copy(fine_p4est_n, P4EST_FALSE);
  fine_p4est_np1->connectivity              = fine_p4est_n->connectivity; // connectivity is not duplicated by p4est_copy, the pointer (i.e. the memory-address) of connectivity seems to be copied from my understanding of the source file of p4est_copy, but I feel this is a bit safer [Raphael Egan]
  fine_p4est_np1->user_pointer              = data_fine;
  // fine_nodes_np1 and fine_phi_np1 have been initialized here above (to shortcut workload in advection in the following loop, if possible)
  iter=0;
  iterative_grid_update_converged = false;
  while(!iterative_grid_update_converged)
  {
    p4est_nodes_t *previous_fine_nodes_np1  = fine_nodes_np1;
    Vec previous_fine_phi_np1               = fine_phi_np1;
    /* ---   FIND THE NEXT ADAPTIVE GRID   --- */
    /* For the very first iteration of the grid-update procedure, fine_p4est_np1 is a
     * simple pure copy of p4est_n_np1, so no node creation nor data interpolation is
     * required. Hence the "if(iter>0)..." statements */
    // We never partition this one, to make ensure locality of interface-capturing data...
    // get fine_nodes_np1
    // reset fine_nodes_np1 if needed
    fine_nodes_np1 = my_p4est_nodes_new(fine_p4est_np1, NULL);
    // reset fine_phi_np1
    fine_phi_np1 = NULL;
    create_node_vector_if_needed(fine_phi_np1, fine_p4est_np1, fine_nodes_np1);
    advect_interface(fine_p4est_np1, fine_nodes_np1, fine_phi_np1,
                     previous_fine_nodes_np1, previous_fine_phi_np1); // limit the workload: use what you already know!
    const double *fine_phi_np1_read_p=NULL;

    ierr = VecGetArrayRead(fine_phi_np1, &fine_phi_np1_read_p); CHKERRXX(ierr);
//    if(iter==293)
//    {
//      my_p4est_vtk_write_all_general(fine_p4est_np1, fine_nodes_np1, NULL,
//                                     P4EST_FALSE, P4EST_FALSE,
//                                     1, /* number of VTK_POINT_DATA */
//                                     0, /* number of VTK_POINT_DATA_VECTOR_BY_COMPONENTS */
//                                     0, /* number of VTK_POINT_DATA_VECTOR_BLOCK */
//                                     0, /* number of VTK_CELL_DATA */
//                                     0, /* number of VTK_CELL_DATA_VECTOR_BY_COMPONENTS */
//                                     0, /* number of VTK_CELL_DATA_VECTOR_BLOCK */
//                                     ("/home/regan/workspace/projects/two_phase_flow/before_fine_update" + std::to_string(iter)).c_str(),
//                                     VTK_NODE_SCALAR, "phi_np1" , fine_phi_np1_read_p);
//    }
    // update the grid
    iterative_grid_update_converged = !criterion_fine_grid.refine_and_coarsen(fine_p4est_np1, fine_nodes_np1, fine_phi_np1_read_p);
    ierr = VecRestoreArrayRead(fine_phi_np1, &fine_phi_np1_read_p); CHKERRXX(ierr);
    iter++;

    ierr = delete_and_nullify_vector(previous_fine_phi_np1); CHKERRXX(ierr);
    if(previous_fine_nodes_np1!=nodes_n)
      p4est_nodes_destroy(previous_fine_nodes_np1);

    if(iter > ((unsigned int) 2+criterion_fine_grid.max_lvl-criterion_fine_grid.min_lvl))
    {
      ierr = PetscPrintf(p4est_n->mpicomm, "ooops ... the grid update did not converge for the fine grid\n"); CHKERRXX(ierr);
      break;
    }
  }

  p4est_nodes_t *previous_fine_nodes_np1  = fine_nodes_np1;
  Vec previous_fine_phi_np1               = fine_phi_np1;
  p4est_ghost_t *fine_ghost_np1 = my_p4est_ghost_new(fine_p4est_np1, P4EST_CONNECT_FULL);
  my_p4est_ghost_expand(fine_p4est_np1, fine_ghost_np1);
  fine_nodes_np1 = my_p4est_nodes_new(fine_p4est_np1, fine_ghost_np1);
  // reset fine_phi_np1
  fine_phi_np1 = NULL;
  create_node_vector_if_needed(fine_phi_np1, fine_p4est_np1, fine_nodes_np1);
  my_p4est_hierarchy_t* fine_hierarchy_np1 = new my_p4est_hierarchy_t(fine_p4est_np1, fine_ghost_np1, brick);
  my_p4est_node_neighbors_t *fine_ngbd_np1 = new my_p4est_node_neighbors_t(fine_hierarchy_np1, fine_nodes_np1);
  fine_ngbd_np1->init_neighbors();

  if(!iterative_grid_update_converged)
  {
    // this should never happen, but in case it does, we need to advect once more
    // to ensure that every node's levelset value is well-defined...
    // that every final local node is known already, so here we are
    advect_interface(fine_p4est_np1, fine_nodes_np1, fine_phi_np1,
                     previous_fine_nodes_np1, previous_fine_phi_np1); // limit the workload: use what you already know!
  }
  else
  {
    const double *previous_fine_phi_np1_p;
    double *fine_phi_np1_p;
    ierr = VecGetArrayRead(previous_fine_phi_np1, &previous_fine_phi_np1_p); CHKERRXX(ierr);
    ierr = VecGetArray(fine_phi_np1, &fine_phi_np1_p); CHKERRXX(ierr);
    for (size_t k = 0; k < fine_ngbd_np1->get_layer_size(); ++k) {
      p4est_locidx_t n = fine_ngbd_np1->get_layer_node(k);
  #ifdef DEBUG
      const p4est_quadrant_t *node = ((const p4est_quadrant_t*) sc_array_index(&fine_nodes_np1->indep_nodes, n));
      P4EST_ASSERT (p4est_quadrant_is_node (node, 1));
      p4est_locidx_t origin_idx;
      P4EST_ASSERT(index_of_node(node, previous_fine_nodes_np1, origin_idx)); // force the abort, since the implementation assumption is wrong --> the developer has more work to do here
      P4EST_ASSERT(origin_idx == n);
  #endif
      fine_phi_np1_p[n] = previous_fine_phi_np1_p[n];
    }
    ierr = VecGhostUpdateBegin(fine_phi_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    for (size_t k = 0; k < fine_ngbd_np1->get_local_size(); ++k) {
      p4est_locidx_t n = fine_ngbd_np1->get_local_node(k);
  #ifdef DEBUG
      const p4est_quadrant_t *node = ((const p4est_quadrant_t*) sc_array_index(&fine_nodes_np1->indep_nodes, n));
      P4EST_ASSERT (p4est_quadrant_is_node (node, 1));
      p4est_locidx_t origin_idx;
      P4EST_ASSERT(index_of_node(node, previous_fine_nodes_np1, origin_idx)); // force the abort, since the implementation assumption is wrong --> the developer has more work to do here
      P4EST_ASSERT(origin_idx == n);
  #endif
      fine_phi_np1_p[n] = previous_fine_phi_np1_p[n];
    }
    ierr = VecGhostUpdateEnd(fine_phi_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecRestoreArray(fine_phi_np1, &fine_phi_np1_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(previous_fine_phi_np1, &previous_fine_phi_np1_p); CHKERRXX(ierr);
  }
  my_p4est_level_set_t ls(fine_ngbd_np1);
  ls.reinitialize_2nd_order(fine_phi_np1, 20);
  // Now, you for sure no longer need the "previous_*"
  if(previous_fine_nodes_np1!=nodes_n)
    p4est_nodes_destroy(previous_fine_nodes_np1);
  ierr = delete_and_nullify_vector(previous_fine_phi_np1); CHKERRXX(ierr);
  // you have your new grids, now transfer your data!


  /* slide relevant fiels and grids in time: nm1 data are disregarded, n data becomes nm1 data and np1 data become n data...
   * In particular, if grid_is_unchanged is false, the np1 grid is different than grid at time n, we need to
   * re-construct its faces and cell-neighbors and the solvers we have used will need to be destroyed... */
  // we can finally slide phi, current phi is no longer needed (IF different than phi_np1)...
  p4est_destroy(fine_p4est_n);        fine_p4est_n      = fine_p4est_np1;
  p4est_ghost_destroy(fine_ghost_n);  fine_ghost_n      = fine_ghost_np1;
  p4est_nodes_destroy(fine_nodes_n);  fine_nodes_n      = fine_nodes_np1;
  delete  fine_hierarchy_n;           fine_hierarchy_n  = fine_hierarchy_np1;
  delete  fine_ngbd_n;                fine_ngbd_n       = fine_ngbd_np1;
  ierr = delete_and_nullify_vector(fine_phi);                         CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(fine_curvature);                   CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(fine_normal);                      CHKERRXX(ierr);
  const bool set_second_derivatives = (fine_phi_xxyyzz != NULL);
  if(set_second_derivatives){
    ierr = delete_and_nullify_vector(fine_phi_xxyyzz);                CHKERRXX(ierr); }
  set_phi(fine_phi_np1, set_second_derivatives);
  ierr = delete_and_nullify_vector(fine_jump_hodge);                  CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(fine_jump_normal_flux_hodge);      CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(fine_jump_mu_grad_v);              CHKERRXX(ierr);
  if(fine_mass_flux!=NULL){
    ierr = delete_and_nullify_vector(fine_mass_flux);                 CHKERRXX(ierr); }
  if(fine_variable_surface_tension!=NULL){
    ierr = delete_and_nullify_vector(fine_variable_surface_tension);  CHKERRXX(ierr); }
  // on computational grid at time nm1, just "slide" fields and grids in discrete time
  if(p4est_nm1!=p4est_n)
    p4est_destroy(p4est_nm1);
  if(ghost_nm1!=ghost_n)
    p4est_ghost_destroy(ghost_nm1);
  if(nodes_nm1!=nodes_n)
    p4est_nodes_destroy(nodes_nm1);
  if(hierarchy_nm1!=hierarchy_n)
    delete hierarchy_nm1;
  if(ngbd_nm1!=ngbd_n)
    delete ngbd_nm1;
  p4est_nm1     = p4est_n;
  ghost_nm1     = ghost_n;
  nodes_nm1     = nodes_n;
  hierarchy_nm1 = hierarchy_n;
  ngbd_nm1      = ngbd_n;
  ierr = delete_and_nullify_vector(vnm1_nodes_omega_minus);           CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(vnm1_nodes_omega_plus);            CHKERRXX(ierr);
  if(vnm1_nodes_omega_minus_xxyyzz!=NULL){
    ierr = delete_and_nullify_vector(vnm1_nodes_omega_minus_xxyyzz);  CHKERRXX(ierr); }
  if(vnm1_nodes_omega_plus_xxyyzz!=NULL){
    ierr = delete_and_nullify_vector(vnm1_nodes_omega_plus_xxyyzz);   CHKERRXX(ierr); }
  if(interface_velocity_n!=NULL){
    ierr = delete_and_nullify_vector(interface_velocity_n);           CHKERRXX(ierr); }
  if(interface_velocity_n_xxyyzz!=NULL){
    ierr = delete_and_nullify_vector(interface_velocity_n_xxyyzz);    CHKERRXX(ierr); }
  vnm1_nodes_omega_minus  = vn_nodes_omega_minus;
  vnm1_nodes_omega_plus   = vn_nodes_omega_plus;
  vnm1_nodes_omega_minus_xxyyzz = vn_nodes_omega_minus_xxyyzz;
  vnm1_nodes_omega_plus_xxyyzz = vn_nodes_omega_plus_xxyyzz;
  interface_velocity_n        = interface_velocity_np1;
  interface_velocity_n_xxyyzz = interface_velocity_np1_xxyyzz;
  // on computational grid at time n, we will need to interpolate things...
  // we will need to interpolate for vnp1 to vn (and then we'll take their derivatives)
  vn_nodes_omega_minus        = NULL; vn_nodes_omega_plus         = NULL;
  vn_nodes_omega_minus_xxyyzz = NULL; vn_nodes_omega_plus_xxyyzz  = NULL;
  ierr = create_node_vector_if_needed(vn_nodes_omega_minus, p4est_np1, nodes_np1, P4EST_DIM);
  ierr = create_node_vector_if_needed(vn_nodes_omega_plus, p4est_np1, nodes_np1, P4EST_DIM);
  ierr = create_node_vector_if_needed(vn_nodes_omega_minus_xxyyzz, p4est_np1, nodes_np1, P4EST_DIM*P4EST_DIM);
  ierr = create_node_vector_if_needed(vn_nodes_omega_plus_xxyyzz, p4est_np1, nodes_np1, P4EST_DIM*P4EST_DIM);
  interp_nodes.clear();
  for (size_t k = 0; k < nodes_np1->indep_nodes.elem_count; ++k) {
    double xyz[P4EST_DIM];
    node_xyz_fr_n(k, p4est_np1,nodes_np1, xyz);
    interp_nodes.add_point(k, xyz);
  }
  Vec inputs[2] = {vnp1_nodes_omega_minus, vnp1_nodes_omega_plus};
  Vec outputs[2] = {vn_nodes_omega_minus, vn_nodes_omega_plus};
  Vec outputs_xxyyzz[2] = {vn_nodes_omega_minus_xxyyzz, vn_nodes_omega_plus_xxyyzz};
  interp_nodes.set_input(inputs, linear, 2, P4EST_DIM);
  interp_nodes.interpolate(outputs);
  ngbd_np1->second_derivatives_central(outputs, outputs_xxyyzz, 2, P4EST_DIM);
  ierr = delete_and_nullify_vector(vnp1_nodes_omega_minus);           CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(vnp1_nodes_omega_plus);            CHKERRXX(ierr);
  interface_velocity_np1        = NULL;
  interface_velocity_np1_xxyyzz = NULL;

  ierr = delete_and_nullify_vector(hodge);                            CHKERRXX(ierr);
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = delete_and_nullify_vector(dxyz_hodge[dir]);                CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(vstar[dir]);                     CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(vnp1_minus[dir]);                CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(vnp1_plus[dir]);                 CHKERRXX(ierr);
  }
  if(p4est_np1!=p4est_n || ghost_np1!=ghost_n || hierarchy_np1!=hierarchy_n){
    if(hierarchy_np1!=hierarchy_n)
    {
      delete  ngbd_c;
      ngbd_c = new my_p4est_cell_neighbors_t(hierarchy_np1);
    }
    delete  faces_n;
    faces_n = new my_p4est_faces_t(p4est_np1, ghost_np1, brick, ngbd_c, true);
  }
  p4est_n     = p4est_np1;
  ghost_n     = ghost_np1;
  nodes_n     = nodes_np1;
  hierarchy_n = hierarchy_np1;
  ngbd_n      = ngbd_np1;

  // clear grid-related buffers, flags and backtrace semi-lagrangian points
  semi_lagrangian_backtrace_is_done = false;
  for (unsigned char dir = 0; dir < P4EST_DIM; ++dir) {
    xyz_n[dir].clear();
    xyz_nm1[dir].clear();
    face_to_fine_node[dir].clear();
    face_to_fine_node_maps_are_set[dir] = false;
    ierr = VecCreateGhostFaces(p4est_n, faces_n, &vstar[dir],       dir); CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est_n, faces_n, &dxyz_hodge[dir],  dir); CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est_n, faces_n, &vnp1_minus[dir],  dir); CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est_n, faces_n, &vnp1_plus[dir],   dir); CHKERRXX(ierr);
  }
  ierr = VecCreateGhostCells(p4est_n, ghost_n, &hodge); CHKERRXX(ierr);
  cell_to_fine_node.clear(); cell_to_fine_node_map_is_set = false;
  node_to_fine_node.clear(); node_to_fine_node_map_is_set = false;


  semi_lagrangian_backtrace_is_done = false;
  ierr = PetscLogEventEnd(log_my_p4est_two_phase_advect_update, 0, 0, 0, 0); CHKERRXX(ierr);


//  {
//    PetscErrorCode ierr;
//    Vec phi_coarse = NULL;
//    interpolate_linearly_from_fine_nodes_to_coarse_nodes(fine_phi, phi_coarse);
//    const double *phi_coarse_p = NULL;
//    const double *v_nodes_omega_plus_p, *v_nodes_omega_minus_p;
//    ierr = VecGetArrayRead(phi_coarse, &phi_coarse_p); CHKERRXX(ierr);
//    ierr = VecGetArrayRead(vn_nodes_omega_minus, &v_nodes_omega_minus_p); CHKERRXX(ierr);
//    ierr = VecGetArrayRead(vn_nodes_omega_plus, &v_nodes_omega_plus_p); CHKERRXX(ierr);

//    my_p4est_vtk_write_all_general(p4est_n, nodes_n, ghost_n,
//                                   P4EST_TRUE, P4EST_TRUE,
//                                   1, /* number of VTK_POINT_DATA */
//                                   0, /* number of VTK_POINT_DATA_VECTOR_BY_COMPONENTS */
//                                   2, /* number of VTK_POINT_DATA_VECTOR_BLOCK */
//                                   0, /* number of VTK_CELL_DATA */
//                                   0, /* number of VTK_CELL_DATA_VECTOR_BY_COMPONENTS */
//                                   0, /* number of VTK_CELL_DATA_VECTOR_BLOCK */
//                                   "/home/regan/workspace/projects/two_phase_flow/after_update",
//                                   VTK_NODE_VECTOR_BLOCK, "vn_plus" , v_nodes_omega_plus_p,
//                                   VTK_NODE_VECTOR_BLOCK, "vn_minus", v_nodes_omega_minus_p,
//                                   VTK_POINT_DATA, "phi", phi_coarse_p);
//    ierr = VecRestoreArrayRead(phi_coarse, &phi_coarse_p); CHKERRXX(ierr);
//    ierr = VecRestoreArrayRead(vn_nodes_omega_minus, &v_nodes_omega_minus_p); CHKERRXX(ierr);
//    ierr = VecRestoreArrayRead(vn_nodes_omega_plus, &v_nodes_omega_plus_p); CHKERRXX(ierr);
//    ierr = delete_and_nullify_vector(phi_coarse); CHKERRXX(ierr);
//  }


  return;
}

