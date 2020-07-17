#ifdef P4_TO_P8
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_trajectory_of_point.h>
#include <src/my_p8est_vtk.h>
#include <src/my_p8est_two_phase_flows.h>
#include <src/my_p8est_solve_lsqr.h>
#else
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_trajectory_of_point.h>
#include <src/my_p4est_vtk.h>
#include <src/my_p4est_two_phase_flows.h>
#include <src/my_p4est_solve_lsqr.h>
#endif

void my_p4est_two_phase_flows_t::splitting_criteria_computational_grid_two_phase_t::
tag_quadrant(p4est_t *p4est_np1, const p4est_locidx_t& quad_idx, const p4est_topidx_t& tree_idx, const p4est_nodes_t* nodes_np1,
             const double *phi_np1_on_computational_nodes_p,
             const double *vorticity_magnitude_np1_on_computational_nodes_minus_p,
             const double *vorticity_magnitude_np1_on_computational_nodes_plus_p)
{
  p4est_tree_t *tree = p4est_tree_array_index(p4est_np1->trees, tree_idx);
  p4est_quadrant_t *quad = p4est_quadrant_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);

  if (quad->level < min_lvl)
    quad->p.user_int = REFINE_QUADRANT;
  else if (quad->level > max_lvl)
    quad->p.user_int = COARSEN_QUADRANT;
  else
  {
    double inv_quad_scal            = 1.0/((double) (1 << quad->level));
    const double quad_diag          = owner->tree_diagonal*inv_quad_scal;
    const double quad_dxyz_max      = MAX(DIM(owner->tree_dimension[0], owner->tree_dimension[1], owner->tree_dimension[2]))*inv_quad_scal;
    const double smallest_dxyz_max  = MAX(DIM(owner->dxyz_smallest_quad[0], owner->dxyz_smallest_quad[1], owner->dxyz_smallest_quad[2]));

    bool coarsen = (quad->level > min_lvl);
    if(coarsen)
    {
      bool cor_vort = true;
      bool cor_band = true;
      bool cor_intf = true;
      p4est_locidx_t node_idx;

      for (u_char k = 0; coarsen && (k < P4EST_CHILDREN); ++k) {
        node_idx    = nodes_np1->local_nodes[P4EST_CHILDREN*quad_idx + k];
        const double &phi_node      = phi_np1_on_computational_nodes_p[node_idx];
        // fetch the appropriate values relevant to criteria
        const double &vorticity     = (phi_node <= 0.0 ? vorticity_magnitude_np1_on_computational_nodes_minus_p[node_idx] : vorticity_magnitude_np1_on_computational_nodes_plus_p[node_idx]);
        const double &max_velocity  = (phi_node <= 0.0 ? owner->max_L2_norm_velocity_minus  : owner->max_L2_norm_velocity_plus);
        const double &uniform_band  = (phi_node <= 0.0 ? owner->uniform_band_minus          : owner->uniform_band_plus);
        // evaluate criteria
        cor_vort  = cor_vort && fabs(vorticity)*2.0*quad_dxyz_max/max_velocity <= owner->threshold_split_cell;
        cor_band  = cor_band && fabs(phi_node) >= uniform_band*smallest_dxyz_max;
        cor_intf  = cor_intf && fabs(phi_node) >= lip*2.0*quad_diag;
        coarsen   = cor_vort && cor_band && cor_intf;
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
      const p4est_qcoord_t mid_qh = P4EST_QUADRANT_LEN (quad->level + 1);
#ifdef P4_TO_P8
      for(u_char k = 0; k < 3; ++k)
#endif
        for(u_char j = 0; j < 3; ++j)
          for(u_char i = 0; i < 3; ++i)
          {
            if (ANDD(i == 1, j == 1, k == 1)) // for sure no node at the center
              continue;
            if (ANDD(i == 0 || i == 2, j == 0 || j == 2, k == 0 || k == 2)) // standard cell-vertex
            {
              node_found = true;
              node_idx = nodes_np1->local_nodes[P4EST_CHILDREN*quad_idx + SUMD(i/2, j, 2*k)];
            }
            else // could be a node associated with a T-junction
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
              const double &phi_node      = phi_np1_on_computational_nodes_p[node_idx];
              // fetch the appropriate values relevant to criteria
              const double &vorticity     = (phi_node <= 0.0 ? vorticity_magnitude_np1_on_computational_nodes_minus_p[node_idx] : vorticity_magnitude_np1_on_computational_nodes_plus_p[node_idx]);
              const double &max_velocity  = (phi_node <= 0.0 ? owner->max_L2_norm_velocity_minus  : owner->max_L2_norm_velocity_plus);
              const double &uniform_band  = (phi_node <= 0.0 ? owner->uniform_band_minus          : owner->uniform_band_plus);
              // evaluate criteria
              ref_vort  = ref_vort  || fabs(vorticity)*quad_dxyz_max/max_velocity > owner->threshold_split_cell;
              ref_band  = ref_band  || fabs(phi_node) < uniform_band*smallest_dxyz_max;
              ref_intf  = ref_intf  || fabs(phi_node) < lip*quad_diag;
              refine    = ref_vort || ref_band || ref_intf;

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
refine_and_coarsen(p4est_t* p4est_np1, const p4est_nodes_t* nodes_np1,
                   Vec phi_np1_on_computational_nodes,
                   Vec vorticity_magnitude_np1_on_computational_nodes_minus,
                   Vec vorticity_magnitude_np1_on_computational_nodes_plus)
{
  const double *phi_np1_on_computational_nodes_p, *vorticity_magnitude_np1_on_computational_nodes_minus_p, *vorticity_magnitude_np1_on_computational_nodes_plus_p;
  PetscErrorCode ierr;
  ierr = VecGetArrayRead(phi_np1_on_computational_nodes,              &phi_np1_on_computational_nodes_p);             CHKERRXX(ierr);
  ierr = VecGetArrayRead(vorticity_magnitude_np1_on_computational_nodes_minus,  &vorticity_magnitude_np1_on_computational_nodes_minus_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(vorticity_magnitude_np1_on_computational_nodes_plus,   &vorticity_magnitude_np1_on_computational_nodes_plus_p);  CHKERRXX(ierr);

  /* tag the quadrants that need to be refined or coarsened */
  for (p4est_topidx_t tree_idx = p4est_np1->first_local_tree; tree_idx <= p4est_np1->last_local_tree; ++tree_idx) {
    p4est_tree_t* tree = p4est_tree_array_index(p4est_np1->trees, tree_idx);
    for (size_t q = 0; q < tree->quadrants.elem_count; ++q) {
      p4est_locidx_t quad_idx  = q + tree->quadrants_offset;
      tag_quadrant(p4est_np1, quad_idx, tree_idx, nodes_np1, phi_np1_on_computational_nodes_p, vorticity_magnitude_np1_on_computational_nodes_minus_p, vorticity_magnitude_np1_on_computational_nodes_plus_p);
    }
  }

  my_p4est_coarsen(p4est_np1, P4EST_FALSE, coarsen_fn, init_fn);
  my_p4est_refine (p4est_np1, P4EST_FALSE, refine_fn,  init_fn);

  int is_grid_changed = false; // "int" cause of MPI_Allreduce thereafter
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

  ierr = VecRestoreArrayRead(vorticity_magnitude_np1_on_computational_nodes_plus,   &vorticity_magnitude_np1_on_computational_nodes_plus_p);  CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(vorticity_magnitude_np1_on_computational_nodes_minus,  &vorticity_magnitude_np1_on_computational_nodes_minus_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(phi_np1_on_computational_nodes,              &phi_np1_on_computational_nodes_p);             CHKERRXX(ierr);

  return is_grid_changed;
}

my_p4est_two_phase_flows_t::my_p4est_two_phase_flows_t(my_p4est_node_neighbors_t *ngbd_nm1_, my_p4est_node_neighbors_t *ngbd_n_, my_p4est_faces_t *faces_n_,
                                                       my_p4est_node_neighbors_t *fine_ngbd_n_)
  : brick(ngbd_n_->myb), conn(ngbd_n_->p4est->connectivity),
    p4est_nm1(ngbd_nm1_->p4est), ghost_nm1(ngbd_nm1_->ghost), nodes_nm1(ngbd_nm1_->nodes), hierarchy_nm1(ngbd_nm1_->hierarchy), ngbd_nm1(ngbd_nm1_),
    p4est_n(ngbd_n_->p4est), fine_p4est_n((fine_ngbd_n_ == NULL ? NULL : fine_ngbd_n_->p4est)),
    ghost_n(ngbd_n_->ghost), fine_ghost_n((fine_ngbd_n_ == NULL ? NULL : fine_ngbd_n_->ghost)),
    nodes_n(ngbd_n_->nodes), fine_nodes_n((fine_ngbd_n_ == NULL ? NULL : fine_ngbd_n_->nodes)),
    hierarchy_n(ngbd_n_->hierarchy), fine_hierarchy_n((fine_ngbd_n_ == NULL ? NULL : fine_ngbd_n_->hierarchy)),
    ngbd_n(ngbd_n_), fine_ngbd_n(fine_ngbd_n_),
    ngbd_c(faces_n_->get_ngbd_c()), faces_n(faces_n_),
    threshold_dbl_max(get_largest_dbl_smaller_than_dbl_max())
{
  surface_tension       =  0.0;
  mu_minus  = mu_plus   =  1.0;
  rho_minus = rho_plus  =  1.0;
  uniform_band_minus = uniform_band_plus = 0.0;
  threshold_split_cell = 0.04;
  cfl = 1.0;
  dt_updated = false;
  max_L2_norm_velocity_minus = max_L2_norm_velocity_plus = 0.0;

  sl_order = 2;

  interface_manager = new my_p4est_interface_manager_t(faces_n, nodes_n, (fine_ngbd_n != NULL ? fine_ngbd_n : ngbd_n));
  xyz_min = p4est_n->connectivity->vertices + 3*p4est_n->connectivity->tree_to_vertex[P4EST_CHILDREN*0                                + 0];
  xyz_max = p4est_n->connectivity->vertices + 3*p4est_n->connectivity->tree_to_vertex[P4EST_CHILDREN*(p4est_n->trees->elem_count - 1) + P4EST_CHILDREN - 1];
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    tree_dimension[dim]     = p4est_n->connectivity->vertices[3*p4est_n->connectivity->tree_to_vertex[P4EST_CHILDREN*0 + P4EST_CHILDREN - 1] + dim] - xyz_min[dim];
    dxyz_smallest_quad[dim] = tree_dimension[dim]/((double) (1 << interface_manager->get_max_level_computational_grid()));
    periodicity[dim]        = is_periodic(p4est_n, dim);
  }
  tree_diagonal = sqrt(SUMD(SQR(tree_dimension[0]), SQR(tree_dimension[1]), SQR(tree_dimension[2])));

  dt_nm1 = dt_n = 0.5*MIN(DIM(dxyz_smallest_quad[0], dxyz_smallest_quad[1], dxyz_smallest_quad[2]));

  bc_velocity = NULL;
  bc_pressure = NULL;
  for(u_char dim = 0; dim < P4EST_DIM; ++dim)
    force_per_unit_mass[dim]  = NULL;

  // -------------------------------------------------------------------
  // ----- FIELDS SAMPLED AT NODES OF THE INTERFACE-CAPTURING GRID -----
  // -------------------------------------------------------------------
  // scalar fields
  phi           = NULL;
  pressure_jump = NULL;
  mass_flux     = NULL;
  // vector fields and/or other P4EST_DIM-block-structured
  phi_xxyyzz  = NULL;
  interface_stress = NULL;
  // -----------------------------------------------------------------------
  // ----- FIELDS SAMPLED AT NODES OF THE COMPUTATIONAL GRID AT TIME N -----
  // -----------------------------------------------------------------------
  // scalar fields
  phi_on_computational_nodes    = NULL;
  vorticity_magnitude_minus     = NULL;
  vorticity_magnitude_plus      = NULL;
  // vector fields
  vnp1_nodes_minus              = NULL;
  vnp1_nodes_plus               = NULL;
  vn_nodes_minus                = NULL;
  vn_nodes_plus                 = NULL;
  interface_velocity_np1        = NULL;
  // tensor/matrix fields
  vn_nodes_minus_xxyyzz         = NULL;
  vn_nodes_plus_xxyyzz          = NULL;
  interface_velocity_np1_xxyyzz = NULL;
  // ------------------------------------------------------------------------------
  // ----- FIELDS SAMPLED AT FACE CENTERS OF THE COMPUTATIONAL GRID AT TIME N -----
  // ------------------------------------------------------------------------------
  // vector fields
  for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
    vnp1_face_minus[dir]  = NULL;
    vnp1_face_plus[dir]   = NULL;
  }
  // -------------------------------------------------------------------------
  // ----- FIELDS SAMPLED AT NODES OF THE COMPUTATIONAL GRID AT TIME NM1 -----
  // -------------------------------------------------------------------------
  // vector fields
  vnm1_nodes_minus        = NULL;
  vnm1_nodes_plus         = NULL;
  // tensor/matrix fields
  vnm1_nodes_minus_xxyyzz = NULL;
  vnm1_nodes_plus_xxyyzz  = NULL;

  // clear backtraced values of velocity components and computational-to-fine-grid maps
  for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
    backtraced_vn_faces_minus[dir].clear();   backtraced_vn_faces_plus[dir].clear();
    backtraced_vnm1_faces_minus[dir].clear(); backtraced_vnm1_faces_plus[dir].clear();
    voro_cell[dir].resize(faces_n->num_local[dir]);
  }
  semi_lagrangian_backtrace_is_done = false;
  voronoi_on_the_fly = false;

  ngbd_n->init_neighbors();
  ngbd_nm1->init_neighbors();
  if(!faces_n->finest_faces_neighborhoods_have_been_set())
    faces_n->set_finest_face_neighborhoods();

  viscosity_solver.set_environment(this);
}

my_p4est_two_phase_flows_t::~my_p4est_two_phase_flows_t()
{
  PetscErrorCode ierr;
  // node-sampled fields on the interface-capturing grid
  if(phi != NULL)                           { ierr = delete_and_nullify_vector(phi);                            CHKERRXX(ierr); }
  if(pressure_jump != NULL)                 { ierr = delete_and_nullify_vector(pressure_jump);                  CHKERRXX(ierr); }
  if(mass_flux != NULL)                     { ierr = delete_and_nullify_vector(mass_flux);                      CHKERRXX(ierr); }
  if(phi_xxyyzz != NULL)                    { ierr = delete_and_nullify_vector(phi_xxyyzz);                     CHKERRXX(ierr); }
  if(interface_stress != NULL)              { ierr = delete_and_nullify_vector(interface_stress);               CHKERRXX(ierr); }
  // node-sampled fields on the computational grids n
  if(phi_on_computational_nodes != NULL)    { ierr = delete_and_nullify_vector(phi_on_computational_nodes);     CHKERRXX(ierr); }
  if(vorticity_magnitude_minus != NULL)     { ierr = delete_and_nullify_vector(vorticity_magnitude_minus);      CHKERRXX(ierr); }
  if(vorticity_magnitude_plus != NULL)      { ierr = delete_and_nullify_vector(vorticity_magnitude_plus);       CHKERRXX(ierr); }
  if(vnp1_nodes_minus !=  NULL)             { ierr = delete_and_nullify_vector(vnp1_nodes_minus);               CHKERRXX(ierr); }
  if(vnp1_nodes_plus != NULL)               { ierr = delete_and_nullify_vector(vnp1_nodes_plus);                CHKERRXX(ierr); }
  if(vn_nodes_minus != NULL)                { ierr = delete_and_nullify_vector(vn_nodes_minus);                 CHKERRXX(ierr); }
  if(vn_nodes_plus != NULL)                 { ierr = delete_and_nullify_vector(vn_nodes_plus);                  CHKERRXX(ierr); }
  if(interface_velocity_np1 != NULL)        { ierr = delete_and_nullify_vector(interface_velocity_np1);         CHKERRXX(ierr); }
  if(vn_nodes_minus_xxyyzz != NULL)         { ierr = delete_and_nullify_vector(vn_nodes_minus_xxyyzz);          CHKERRXX(ierr); }
  if(vn_nodes_plus_xxyyzz != NULL)          { ierr = delete_and_nullify_vector(vn_nodes_plus_xxyyzz);           CHKERRXX(ierr); }
  if(interface_velocity_np1_xxyyzz != NULL) { ierr = delete_and_nullify_vector(interface_velocity_np1_xxyyzz);  CHKERRXX(ierr); }
  // node-sampled fields on the computational grids nm1
  if(vnm1_nodes_minus != NULL)              { ierr = delete_and_nullify_vector(vnm1_nodes_minus);               CHKERRXX(ierr); }
  if(vnm1_nodes_plus != NULL)               { ierr = delete_and_nullify_vector(vnm1_nodes_plus);                CHKERRXX(ierr); }
  if(vnm1_nodes_minus_xxyyzz != NULL)       { ierr = delete_and_nullify_vector(vnm1_nodes_minus_xxyyzz);        CHKERRXX(ierr); }
  if(vnm1_nodes_plus_xxyyzz != NULL)        { ierr = delete_and_nullify_vector(vnm1_nodes_plus_xxyyzz);         CHKERRXX(ierr); }
  // face-sampled fields, computational grid n
  for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
    if(vnp1_face_minus[dir] != NULL)        { ierr = delete_and_nullify_vector(vnp1_face_minus[dir]);           CHKERRXX(ierr); }
    if(vnp1_face_plus[dir] != NULL)         { ierr = delete_and_nullify_vector(vnp1_face_plus[dir]);            CHKERRXX(ierr); }
  }

  if(interface_manager != NULL)
    delete interface_manager;

  if(ngbd_nm1 != ngbd_n && ngbd_nm1 != NULL)
    delete ngbd_nm1;
  if(ngbd_n != NULL)
    delete ngbd_n;
  if(fine_ngbd_n != NULL)
    delete fine_ngbd_n;
  if(hierarchy_nm1 != hierarchy_n && hierarchy_nm1 != NULL)
    delete hierarchy_nm1;
  if(hierarchy_n != NULL)
    delete hierarchy_n;
  if(fine_hierarchy_n != NULL)
    delete fine_hierarchy_n;
  if(nodes_nm1 != nodes_n && nodes_nm1 != NULL)
    p4est_nodes_destroy(nodes_nm1);
  if(nodes_n != NULL)
    p4est_nodes_destroy(nodes_n);
  if(fine_nodes_n != NULL)
    p4est_nodes_destroy(fine_nodes_n);
  if(p4est_nm1 != p4est_n && p4est_nm1 != NULL)
    p4est_destroy(p4est_nm1);
  if(p4est_n != NULL)
    p4est_destroy(p4est_n);
  if(fine_p4est_n != NULL)
    p4est_destroy(fine_p4est_n);
  if(ghost_nm1 != ghost_n && ghost_nm1 != NULL)
    p4est_ghost_destroy(ghost_nm1);
  if(ghost_n != NULL)
    p4est_ghost_destroy(ghost_n);
  if(fine_ghost_n != NULL)
    p4est_ghost_destroy(fine_ghost_n);

  if(faces_n != NULL)
    delete faces_n;
  if(ngbd_c != NULL)
    delete ngbd_c;
}

void my_p4est_two_phase_flows_t::set_phi(Vec phi_on_interface_capturing_nodes, const interpolation_method& method)
{
  PetscErrorCode ierr;
  P4EST_ASSERT(phi_on_interface_capturing_nodes != NULL);
  if(phi != phi_on_interface_capturing_nodes)
  {
    if(phi != NULL){
      ierr = delete_and_nullify_vector(phi); CHKERRXX(ierr); }
    phi = phi_on_interface_capturing_nodes;
    if(phi_xxyyzz != NULL){
      ierr = delete_and_nullify_vector(phi_xxyyzz); CHKERRXX(ierr); }// no longer valid
  }
  else
  {
    if(method == linear && phi_xxyyzz != NULL){
      ierr = delete_and_nullify_vector(phi_xxyyzz); CHKERRXX(ierr);
    }
  }

  if(method != linear && phi_xxyyzz == NULL){
    ierr = interface_manager->create_vector_on_interface_capturing_nodes(phi_xxyyzz, P4EST_DIM); CHKERRXX(ierr);
    interface_manager->get_interface_capturing_ngbd_n().second_derivatives_central(phi, phi_xxyyzz);
  }
  interface_manager->set_levelset(phi, method, phi_xxyyzz, true, true);
  return;
}

void my_p4est_two_phase_flows_t::set_node_velocities(CF_DIM* vnm1_minus_functor[P4EST_DIM], CF_DIM* vn_minus_functor[P4EST_DIM],
                                                     CF_DIM* vnm1_plus_functor[P4EST_DIM],  CF_DIM* vn_plus_functor[P4EST_DIM])
{
  PetscErrorCode ierr;
  double xyz_node[P4EST_DIM];
  if(vnm1_nodes_minus == NULL){
    ierr = VecCreateGhostNodesBlock(p4est_nm1, nodes_nm1, P4EST_DIM, &vnm1_nodes_minus); CHKERRXX(ierr); }
  if(vnm1_nodes_plus == NULL){
    ierr = VecCreateGhostNodesBlock(p4est_nm1, nodes_nm1, P4EST_DIM, &vnm1_nodes_plus); CHKERRXX(ierr); }
  if(vn_nodes_minus == NULL){
    ierr = VecCreateGhostNodesBlock(p4est_n, nodes_n, P4EST_DIM, &vn_nodes_minus); CHKERRXX(ierr); }
  if(vn_nodes_plus == NULL){
    ierr = VecCreateGhostNodesBlock(p4est_n, nodes_n, P4EST_DIM, &vn_nodes_plus); CHKERRXX(ierr); }
  double *vnm1_nodes_minus_p, *vnm1_nodes_plus_p, *vn_nodes_minus_p, *vn_nodes_plus_p;

  ierr = VecGetArray(vnm1_nodes_minus,  &vnm1_nodes_minus_p); CHKERRXX(ierr);
  ierr = VecGetArray(vnm1_nodes_plus,   &vnm1_nodes_plus_p);  CHKERRXX(ierr);
  ierr = VecGetArray(vn_nodes_minus,    &vn_nodes_minus_p);   CHKERRXX(ierr);
  ierr = VecGetArray(vn_nodes_plus,     &vn_nodes_plus_p);    CHKERRXX(ierr);
  for (size_t n = 0; n < MAX(nodes_n->indep_nodes.elem_count, nodes_nm1->indep_nodes.elem_count); ++n) {
    if(n < nodes_n->indep_nodes.elem_count)
    {
      node_xyz_fr_n(n, p4est_n, nodes_n, xyz_node);
      for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
        vn_nodes_minus_p[P4EST_DIM*n + dir] = (*vn_minus_functor[dir])(xyz_node);
        vn_nodes_plus_p[P4EST_DIM*n + dir]  = (*vn_plus_functor[dir])(xyz_node);
      }
    }
    if(n < nodes_nm1->indep_nodes.elem_count)
    {
      node_xyz_fr_n(n, p4est_nm1, nodes_nm1, xyz_node);
      for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
        vnm1_nodes_minus_p[P4EST_DIM*n + dir] = (*vnm1_minus_functor[dir])(xyz_node);
        vnm1_nodes_plus_p[P4EST_DIM*n + dir]  = (*vnm1_plus_functor[dir])(xyz_node);
      }
    }
  }
  ierr = VecRestoreArray(vnm1_nodes_minus,  &vnm1_nodes_minus_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(vnm1_nodes_plus,   &vnm1_nodes_plus_p);  CHKERRXX(ierr);
  ierr = VecRestoreArray(vn_nodes_minus,    &vn_nodes_minus_p);   CHKERRXX(ierr);
  ierr = VecRestoreArray(vn_nodes_plus,     &vn_nodes_plus_p);    CHKERRXX(ierr);
  compute_second_derivatives_of_nm1_velocities();
  compute_second_derivatives_of_n_velocities();
  return;
}

void my_p4est_two_phase_flows_t::set_face_velocities_np1(CF_DIM* vnp1_minus_functor[P4EST_DIM], CF_DIM* vnp1_plus_functor[P4EST_DIM])
{
  PetscErrorCode ierr;
  double *vnp1_face_minus_p[P4EST_DIM], *vnp1_face_plus_p[P4EST_DIM];
  double xyz_face[P4EST_DIM];
  for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
    if(vnp1_face_minus[dir] == NULL){
      ierr = VecCreateGhostFaces(p4est_n, faces_n, &vnp1_face_minus[dir], dir); CHKERRXX(ierr); }
    if(vnp1_face_plus[dir] == NULL){
      ierr = VecCreateGhostFaces(p4est_n, faces_n, &vnp1_face_plus[dir], dir); CHKERRXX(ierr); }
    ierr = VecGetArray(vnp1_face_minus[dir], &vnp1_face_minus_p[dir]);  CHKERRXX(ierr);
    ierr = VecGetArray(vnp1_face_plus[dir],  &vnp1_face_plus_p[dir]);   CHKERRXX(ierr);
    for (size_t k = 0; k < faces_n->get_layer_size(dir); ++k) {
      p4est_locidx_t face_idx = faces_n->get_layer_face(dir, k);
      faces_n->xyz_fr_f(face_idx, dir, xyz_face);
      vnp1_face_minus_p[dir][face_idx]  = (*vnp1_minus_functor[dir])(xyz_face);
      vnp1_face_plus_p[dir][face_idx]   = (*vnp1_plus_functor[dir])(xyz_face);
    }
    ierr = VecGhostUpdateBegin(vnp1_face_minus[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(vnp1_face_plus[dir],  INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
    for (size_t k = 0; k < faces_n->get_local_size(dir); ++k) {
      p4est_locidx_t face_idx = faces_n->get_local_face(dir, k);
      faces_n->xyz_fr_f(face_idx, dir, xyz_face);
      vnp1_face_minus_p[dir][face_idx]  = (*vnp1_minus_functor[dir])(xyz_face);
      vnp1_face_plus_p[dir][face_idx]   = (*vnp1_plus_functor[dir])(xyz_face);
    }
    ierr = VecGhostUpdateEnd(vnp1_face_minus[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(vnp1_face_plus[dir],  INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecRestoreArray(vnp1_face_minus[dir], &vnp1_face_minus_p[dir]);  CHKERRXX(ierr);
    ierr = VecRestoreArray(vnp1_face_plus[dir],  &vnp1_face_plus_p[dir]);   CHKERRXX(ierr);
  }
  return;
}

void my_p4est_two_phase_flows_t::compute_second_derivatives_of_n_velocities()
{
  PetscErrorCode ierr;
  if(vn_nodes_minus_xxyyzz == NULL){
    ierr = VecCreateGhostNodesBlock(p4est_n, nodes_n, SQR_P4EST_DIM, &vn_nodes_minus_xxyyzz); CHKERRXX(ierr);
  }
  if(vn_nodes_plus_xxyyzz == NULL){
    ierr = VecCreateGhostNodesBlock(p4est_n, nodes_n, SQR_P4EST_DIM, &vn_nodes_plus_xxyyzz); CHKERRXX(ierr);
  }

  Vec fields_to_differentiate[2]  = {vn_nodes_minus,        vn_nodes_plus};
  Vec second_derivatives[2]       = {vn_nodes_minus_xxyyzz, vn_nodes_plus_xxyyzz};
  ngbd_n->second_derivatives_central(fields_to_differentiate, second_derivatives, 2, P4EST_DIM);
  return;
}

void my_p4est_two_phase_flows_t::compute_second_derivatives_of_nm1_velocities()
{
  PetscErrorCode ierr;
  if(vnm1_nodes_minus_xxyyzz == NULL){
    ierr = VecCreateGhostNodesBlock(p4est_nm1, nodes_nm1, SQR_P4EST_DIM, &vnm1_nodes_minus_xxyyzz); CHKERRXX(ierr);
  }
  if(vnm1_nodes_plus_xxyyzz == NULL){
    ierr = VecCreateGhostNodesBlock(p4est_nm1, nodes_nm1, SQR_P4EST_DIM, &vnm1_nodes_plus_xxyyzz); CHKERRXX(ierr);
  }

  Vec fields_to_differentiate[2]  = {vnm1_nodes_minus,        vnm1_nodes_plus};
  Vec second_derivatives[2]       = {vnm1_nodes_minus_xxyyzz, vnm1_nodes_plus_xxyyzz};
  ngbd_nm1->second_derivatives_central(fields_to_differentiate, second_derivatives, 2, P4EST_DIM);
  return;
}

void my_p4est_two_phase_flows_t::compute_pressure_jump()
{
  PetscErrorCode ierr;
  if(pressure_jump == NULL){
    interface_manager->create_vector_on_interface_capturing_nodes(pressure_jump); CHKERRXX(ierr); }

  // [p] = -surface_tension*curvature - SQR(mass_flux)*jump_of_inverse_mass_density + [2\mu n \cdot E \cdot n]
  // and we use [2\mu n\cdot E\cdot n] = 2*[\mu] n_cdot_grad_u_minus_or_plus_cdot_n + 2*mu_plus_or_minus*(-curvature*mass_flux*jump_of_inverse_mass_density)
  // (proof in our summary of equations)

  // compute n_cdot_grad_u_minus_cdot_n and n_cdot_grad_u_minus_cdot_n if needed
  Vec n_cdot_grad_u_minus_cdot_n = NULL, n_cdot_grad_u_plus_cdot_n = NULL;
  if(fabs(mu_plus - mu_minus) > EPS*MAX(fabs(mu_plus), fabs(mu_minus)))
  {
    if(vnp1_nodes_minus == NULL || vnp1_nodes_plus == NULL)
      throw std::runtime_error("my_p4est_two_phase_flows_t::compute_pressure_jump() : the (n + 1) node velocities would be required to evaluate some terms in the pressure jump (have you interpolated at the nodes after the viscosity step?)");
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &n_cdot_grad_u_minus_cdot_n);  CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &n_cdot_grad_u_plus_cdot_n);   CHKERRXX(ierr);
    double *n_cdot_grad_u_minus_cdot_n_p, *n_cdot_grad_u_plus_cdot_n_p;
    ierr = VecGetArray(n_cdot_grad_u_minus_cdot_n,  &n_cdot_grad_u_minus_cdot_n_p); CHKERRXX(ierr);
    ierr = VecGetArray(n_cdot_grad_u_plus_cdot_n,   &n_cdot_grad_u_plus_cdot_n_p);  CHKERRXX(ierr);

    const double *vnp1_nodes_minus_p, *vnp1_nodes_plus_p;
    ierr = VecGetArrayRead(vnp1_nodes_minus,  &vnp1_nodes_minus_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(vnp1_nodes_plus,   &vnp1_nodes_plus_p);  CHKERRXX(ierr);
    double xyz_node[P4EST_DIM], normal_vector[P4EST_DIM];
    double grad_u_minus[SQR_P4EST_DIM], grad_u_plus[SQR_P4EST_DIM];
    const double *inputs[2] = {vnp1_nodes_minus_p, vnp1_nodes_plus_p};
    double* outputs[2] = {grad_u_minus, grad_u_plus};
    quad_neighbor_nodes_of_node_t qnnn_buf;
    const quad_neighbor_nodes_of_node_t* qnnn_p = (ngbd_n->neighbors_are_initialized() ? NULL : &qnnn_buf);
    for (size_t k = 0; k < ngbd_n->get_layer_size(); ++k) {
      const p4est_locidx_t node_idx = ngbd_n->get_layer_node(k);
      if(ngbd_n->neighbors_are_initialized())
        ngbd_n->get_neighbors(node_idx, qnnn_p);
      else
        ngbd_n->get_neighbors(node_idx, qnnn_buf);

      node_xyz_fr_n(node_idx, p4est_n, nodes_n, xyz_node);
      interface_manager->normal_vector_at_point(xyz_node, normal_vector);
      qnnn_p->gradient_all_components(inputs, outputs, 2, P4EST_DIM);
      n_cdot_grad_u_minus_cdot_n_p[node_idx]  = 0.0;
      n_cdot_grad_u_plus_cdot_n_p[node_idx]   = 0.0;
      for (u_char uu = 0; uu < P4EST_DIM; ++uu)
        for (u_char vv = 0; vv < P4EST_DIM; ++vv)
        {
          n_cdot_grad_u_minus_cdot_n_p[node_idx]  += normal_vector[uu]*normal_vector[vv]*grad_u_minus[P4EST_DIM*uu + vv];
          n_cdot_grad_u_plus_cdot_n_p[node_idx]   += normal_vector[uu]*normal_vector[vv]*grad_u_plus[P4EST_DIM*uu + vv];
        }
    }
    ierr = VecGhostUpdateBegin(n_cdot_grad_u_minus_cdot_n,  INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(n_cdot_grad_u_plus_cdot_n,   INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    for (size_t k = 0; k < ngbd_n->get_local_size(); ++k) {
      const p4est_locidx_t node_idx = ngbd_n->get_local_node(k);
      if(ngbd_n->neighbors_are_initialized())
        ngbd_n->get_neighbors(node_idx, qnnn_p);
      else
        ngbd_n->get_neighbors(node_idx, qnnn_buf);

      node_xyz_fr_n(node_idx, p4est_n, nodes_n, xyz_node);
      interface_manager->normal_vector_at_point(xyz_node, normal_vector);
      qnnn_p->gradient_all_components(inputs, outputs, 2, P4EST_DIM);
      n_cdot_grad_u_minus_cdot_n_p[node_idx]  = 0.0;
      n_cdot_grad_u_plus_cdot_n_p[node_idx]   = 0.0;
      for (u_char uu = 0; uu < P4EST_DIM; ++uu)
        for (u_char vv = 0; vv < P4EST_DIM; ++vv)
        {
          n_cdot_grad_u_minus_cdot_n_p[node_idx]  += normal_vector[uu]*normal_vector[vv]*grad_u_minus[P4EST_DIM*uu + vv];
          n_cdot_grad_u_plus_cdot_n_p[node_idx]   += normal_vector[uu]*normal_vector[vv]*grad_u_plus[P4EST_DIM*uu + vv];
        }
    }
    ierr = VecGhostUpdateBegin(n_cdot_grad_u_minus_cdot_n,  INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(n_cdot_grad_u_plus_cdot_n,   INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecRestoreArrayRead(vnp1_nodes_minus,  &vnp1_nodes_minus_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(vnp1_nodes_plus,   &vnp1_nodes_plus_p);  CHKERRXX(ierr);
    ierr = VecRestoreArray(n_cdot_grad_u_minus_cdot_n,  &n_cdot_grad_u_minus_cdot_n_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(n_cdot_grad_u_plus_cdot_n,   &n_cdot_grad_u_plus_cdot_n_p);  CHKERRXX(ierr);
  }
  // let's use (linear) interpolators since we may need to do that at subresolved nodes
  my_p4est_interpolation_nodes_t *interp_n_cdot_grad_u_minus_cdot_n = NULL, *interp_n_cdot_grad_u_plus_cdot_n = NULL;
  if(n_cdot_grad_u_minus_cdot_n != NULL && n_cdot_grad_u_plus_cdot_n != NULL)
  {
    interp_n_cdot_grad_u_minus_cdot_n = new my_p4est_interpolation_nodes_t(ngbd_n); interp_n_cdot_grad_u_minus_cdot_n->set_input(n_cdot_grad_u_minus_cdot_n, linear);
    interp_n_cdot_grad_u_plus_cdot_n  = new my_p4est_interpolation_nodes_t(ngbd_n); interp_n_cdot_grad_u_plus_cdot_n->set_input(n_cdot_grad_u_plus_cdot_n, linear);
  }
  const double* curvature_p = NULL;
  const double* mass_flux_p = NULL;

  if(mass_flux != NULL && fabs(rho_minus - rho_plus) > EPS*MAX(fabs(rho_minus), fabs(rho_plus))){
    ierr = VecGetArrayRead(mass_flux, &mass_flux_p); CHKERRXX(ierr); }
  if(fabs(surface_tension) > EPS || mass_flux_p != NULL)
  {
    if(!interface_manager->is_curvature_set())
      interface_manager->set_curvature(); // Maybe we'd want to flatten it... -> to be tested!

    ierr = VecGetArrayRead(interface_manager->get_curvature(), &curvature_p); CHKERRXX(ierr);
  }

  double* pressure_jump_p;
  const double *phi_p;
  ierr = VecGetArray(pressure_jump, &pressure_jump_p);          CHKERRXX(ierr);
  ierr = VecGetArrayRead(interface_manager->get_phi(), &phi_p); CHKERRXX(ierr);
  double xyz_node[P4EST_DIM];
  const my_p4est_node_neighbors_t& interface_capturing_ngbd_n = interface_manager->get_interface_capturing_ngbd_n();
  for (size_t k = 0; k < interface_capturing_ngbd_n.get_layer_size(); ++k) {
    const p4est_locidx_t node_idx = interface_capturing_ngbd_n.get_layer_node(k);
    pressure_jump_p[node_idx] = 0.0;
    if(curvature_p != NULL)
      pressure_jump_p[node_idx] -= surface_tension*curvature_p[node_idx];
    if(mass_flux_p != NULL)
      pressure_jump_p[node_idx] -= SQR(mass_flux_p[node_idx])*jump_inverse_mass_density();
    if(curvature_p != NULL && mass_flux_p != NULL)
      pressure_jump_p[node_idx] -= 2.0*(phi_p[node_idx] <= 0.0 ? mu_plus : mu_minus)*curvature_p[node_idx]*mass_flux_p[node_idx]*jump_inverse_mass_density();
    if(interp_n_cdot_grad_u_minus_cdot_n != NULL && interp_n_cdot_grad_u_plus_cdot_n != NULL)
    {
      node_xyz_fr_n(node_idx, interface_capturing_ngbd_n.get_p4est(), interface_capturing_ngbd_n.get_nodes(), xyz_node);
      pressure_jump_p[node_idx] += 2.0*jump_viscosity()*(phi_p[node_idx] <= 0.0 ? (*interp_n_cdot_grad_u_minus_cdot_n)(xyz_node) : (*interp_n_cdot_grad_u_plus_cdot_n)(xyz_node));
    }
  }
  ierr = VecGhostUpdateBegin(pressure_jump, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for (size_t k = 0; k < interface_capturing_ngbd_n.get_local_size(); ++k) {
    const p4est_locidx_t node_idx = interface_capturing_ngbd_n.get_local_node(k);
    pressure_jump_p[node_idx] = 0.0;
    if(curvature_p != NULL)
      pressure_jump_p[node_idx] -= surface_tension*curvature_p[node_idx];
    if(mass_flux_p != NULL)
      pressure_jump_p[node_idx] -= SQR(mass_flux_p[node_idx])*jump_inverse_mass_density();
    if(curvature_p != NULL && mass_flux_p != NULL)
      pressure_jump_p[node_idx] -= 2.0*(phi_p[node_idx] <= 0.0 ? mu_plus : mu_minus)*curvature_p[node_idx]*mass_flux_p[node_idx]*jump_inverse_mass_density();
    if(interp_n_cdot_grad_u_minus_cdot_n != NULL && interp_n_cdot_grad_u_plus_cdot_n != NULL)
    {
      node_xyz_fr_n(node_idx, interface_capturing_ngbd_n.get_p4est(), interface_capturing_ngbd_n.get_nodes(), xyz_node);
      pressure_jump_p[node_idx] += 2.0*jump_viscosity()*(phi_p[node_idx] <= 0.0 ? (*interp_n_cdot_grad_u_minus_cdot_n)(xyz_node) : (*interp_n_cdot_grad_u_plus_cdot_n)(xyz_node));
    }
  }
  ierr = VecGhostUpdateEnd(pressure_jump, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecRestoreArrayRead(interface_manager->get_phi(), &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(pressure_jump, &pressure_jump_p); CHKERRXX(ierr);

  if(mass_flux_p != NULL){
    ierr = VecRestoreArrayRead(mass_flux, &mass_flux_p); CHKERRXX(ierr); }
  if(curvature_p != NULL){
    ierr = VecRestoreArrayRead(interface_manager->get_curvature(), &curvature_p); CHKERRXX(ierr); }

  if(n_cdot_grad_u_minus_cdot_n != NULL){
    ierr = VecDestroy(n_cdot_grad_u_minus_cdot_n); CHKERRXX(ierr); }
  if(n_cdot_grad_u_plus_cdot_n != NULL){
    ierr = VecDestroy(n_cdot_grad_u_plus_cdot_n); CHKERRXX(ierr); }
  if(interp_n_cdot_grad_u_minus_cdot_n != NULL)
    delete interp_n_cdot_grad_u_minus_cdot_n;
  if(interp_n_cdot_grad_u_plus_cdot_n != NULL)
    delete interp_n_cdot_grad_u_plus_cdot_n;
  return;
}

/* solve the projection step, we consider (dt_n/(alpha*rho))*p to be the HODGE variable, and we solve for p right away
 * -div((dt_n/(alpha*rho))*grad(p)) = -div(vstar)
 * jump_p = - surface_tension*kappa  - SQR(mass_flux)*jump_inverse_mass_density() + [2*mu*(<n | E | n>)]
 * jump_normal_flux_hodge = 0.0! --> because we assume the normal jump in u_star has been correctly captured so far...
 */
void my_p4est_two_phase_flows_t::solve_projection(my_p4est_poisson_jump_cells_fv_t* &cell_poisson_jump_solver, const KSPType ksp, const PCType pc)
{
  /* Make the two-phase velocity field divergence free : */
  compute_pressure_jump();
  if(cell_poisson_jump_solver == NULL)
    cell_poisson_jump_solver = new my_p4est_poisson_jump_cells_fv_t(ngbd_c, nodes_n);

  cell_poisson_jump_solver->set_interface(interface_manager);
  cell_poisson_jump_solver->set_diagonals(0.0, 0.0);
  cell_poisson_jump_solver->set_mus(dt_n/(BDF_alpha()*rho_minus), dt_n/(BDF_alpha()*rho_plus));
  cell_poisson_jump_solver->set_bc(*bc_pressure);
  cell_poisson_jump_solver->set_jumps(pressure_jump, NULL);

  if(mass_flux != NULL)
    throw std::runtime_error("my_p4est_two_phase_flows_t::solve_projection : you have more work to do here when considering a nonzero mass flux");
  cell_poisson_jump_solver->set_velocity_on_faces(vnp1_face_minus, vnp1_face_plus, NULL);
  cell_poisson_jump_solver->solve(ksp, pc);

  cell_poisson_jump_solver->project_face_velocities(faces_n);

  return;
}

void my_p4est_two_phase_flows_t::solve_viscosity_explicit()
{
  PetscErrorCode ierr;

  /* construct the right hand side */
  compute_backtraced_velocities();
  my_p4est_interpolation_nodes_t interp_n_minus(ngbd_n), interp_n_plus(ngbd_n);
  interp_n_minus.set_input(vn_nodes_minus, vn_nodes_minus_xxyyzz, quadratic, P4EST_DIM);
  interp_n_plus.set_input(vn_nodes_plus, vn_nodes_plus_xxyyzz, quadratic, P4EST_DIM);
  const double alpha = BDF_alpha();
  const double beta = BDF_beta();
  for(u_char dir = 0; dir < P4EST_DIM; ++dir)
  {
    for(p4est_locidx_t f_idx = 0; f_idx < faces_n->num_local[dir]; ++f_idx)
    {
      double xyz_face[P4EST_DIM]; faces_n->xyz_fr_f(f_idx, dir, xyz_face);
      const char sgn_face = (interface_manager->phi_at_point(xyz_face) <= 0.0 ? -1 : +1);
      if(sgn_face < 0)
        interp_n_minus.add_point(f_idx, xyz_face);
      else
        interp_n_plus.add_point(f_idx, xyz_face);
    }
    Vec vn_faces;
    double *vn_faces_p;
    ierr = VecCreateGhostFaces(p4est_n, faces_n, &vn_faces, dir); CHKERRXX(ierr);
    ierr = VecGetArray(vn_faces, &vn_faces_p);                    CHKERRXX(ierr);
    interp_n_minus.interpolate(vn_faces_p, dir);  interp_n_minus.clear();
    interp_n_plus.interpolate(vn_faces_p, dir);   interp_n_plus.clear();

    ierr = VecGhostUpdateBegin(vn_faces, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(vn_faces, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);

    double *vstar_minus_p, *vstar_plus_p;
    ierr = VecGetArray(vnp1_face_minus[dir], &vstar_minus_p); CHKERRXX(ierr);
    ierr = VecGetArray(vnp1_face_plus[dir], &vstar_plus_p);   CHKERRXX(ierr);
    for(p4est_locidx_t f_idx = 0; f_idx < faces_n->num_local[dir]; ++f_idx)
    {
      double xyz_face[P4EST_DIM]; faces_n->xyz_fr_f(f_idx, dir, xyz_face);
      const char sgn_face = (interface_manager->phi_at_point(xyz_face) <= 0.0 ? -1 : +1);
      double *vstar_p = (sgn_face < 0.0 ? vstar_minus_p : vstar_plus_p);
      const double& rho_face = (sgn_face < 0.0 ? rho_minus : rho_plus);
      const std::vector<double>& backtraced_vn    = (sgn_face < 0 ? backtraced_vn_faces_minus[dir] : backtraced_vn_faces_plus[dir]);
      const std::vector<double>* backtraced_vnm1  = (sl_order == 2 ?  (sgn_face < 0 ? &backtraced_vnm1_faces_minus[dir] : &backtraced_vnm1_faces_plus[dir]) : NULL);
      if(face_is_dirichlet_wall(f_idx, dir))
      {
        vstar_p[f_idx] = bc_velocity[dir].wallValue(xyz_face);
        continue;
      }

      const double viscous_term = div_mu_grad_u_dir(f_idx, dir, vn_faces_p);
      vstar_p[f_idx] = + (dt_n/(alpha*rho_face))*viscous_term
          + (1.0 - (beta*dt_n/(alpha*dt_nm1)))*backtraced_vn[f_idx]
          + (sl_order == 2 ? (beta*dt_n/(alpha*dt_nm1))*(*backtraced_vnm1)[f_idx] : 0.0);
      if (force_per_unit_mass[dir] != NULL)
        vstar_p[f_idx] += (dt_n/alpha)*(*force_per_unit_mass[dir])(xyz_face);
    }
    ierr = VecRestoreArray(vnp1_face_plus[dir], &vstar_plus_p);   CHKERRXX(ierr);
    ierr = VecRestoreArray(vnp1_face_minus[dir], &vstar_minus_p); CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(vn_faces); CHKERRXX(ierr);
  }
}

double my_p4est_two_phase_flows_t::div_mu_grad_u_dir(const p4est_locidx_t &face_idx, const u_char &dir, const double *vn_dir_p)
{
  const augmented_voronoi_cell my_cell = get_augmented_voronoi_cell(face_idx, dir);
  double xyz_face[P4EST_DIM]; my_cell.voro.get_center_point(xyz_face);
  const char sgn_face = (interface_manager->phi_at_point(xyz_face) <= 0.0 ? -1 : +1);
  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;
  faces_n->f2q(face_idx, dir, quad_idx, tree_idx);
  const p4est_quadrant_t *quad = fetch_quad(quad_idx, tree_idx, p4est_n, ghost_n);

  const u_char face_touch = (faces_n->q2f(quad_idx, 2*dir) == face_idx ? 2*dir : 2*dir + 1);
  P4EST_ASSERT(faces_n->q2f(quad_idx, face_touch) == face_idx);
  p4est_quadrant_t qm, qp;
  faces_n->find_quads_touching_face(face_idx, dir, qm, qp);

  bool wall[P4EST_FACES];
  for(u_char d = 0; d < P4EST_DIM; ++d)
  {
    if(d == dir)
    {
      wall[2*d]     = qm.p.piggy3.local_num == -1;
      wall[2*d + 1] = qp.p.piggy3.local_num == -1;
    }
    else
    {
      wall[2*d]     = (qm.p.piggy3.local_num == -1 || is_quad_Wall(p4est_n, qm.p.piggy3.which_tree, &qm, 2*d    )) && (qp.p.piggy3.local_num == -1 || is_quad_Wall(p4est_n, qp.p.piggy3.which_tree, &qp, 2*d));
      wall[2*d + 1] = (qm.p.piggy3.local_num == -1 || is_quad_Wall(p4est_n, qm.p.piggy3.which_tree, &qm, 2*d + 1)) && (qp.p.piggy3.local_num == -1 || is_quad_Wall(p4est_n, qp.p.piggy3.which_tree, &qp, 2*d + 1));
    }
  }


  const vector<ngbdDIMseed> *points;
#ifndef P4_TO_P8
  const vector<Point2> *partition;
  my_cell.voro.get_partition(partition);
#endif
  my_cell.voro.get_neighbor_seeds(points);
  const double volume = my_cell.voro.get_volume();

  double to_return = 0.0;
  if(!my_cell.has_neighbor_across || (my_cell.cell_type != parallelepiped_no_wall && my_cell.cell_type != parallelepiped_with_wall))
  {
    for (size_t m = 0; m < points->size(); ++m) {
#ifdef P4_TO_P8
      const double surface = (*points)[m].s;
#else
      size_t k = mod(m - 1, points->size());
      const double surface = ((*partition)[m] - (*partition)[k]).norm_L2();
#endif
      const double distance_to_neighbor = sqrt(SUMD(SQR((*points)[m].p.x - xyz_face[0]), SQR((*points)[m].p.y - xyz_face[1]), SQR((*points)[m].p.z - xyz_face[2])));
      const double& mu_this_side = (sgn_face < 0 ? mu_minus : mu_plus);
      switch ((*points)[m].n) {
      case WALL_m00:
      case WALL_p00:
      case WALL_0m0:
      case WALL_0p0:
#ifdef P4_TO_P8
      case WALL_00m:
      case WALL_00p:
#endif
      {
        char wall_orientation = -1 - (*points)[m].n;
        P4EST_ASSERT(wall_orientation >= 0 && wall_orientation < P4EST_FACES);
        double wall_eval[P4EST_DIM];
        const double lambda = ((wall_orientation%2 == 1 ? xyz_max[wall_orientation/2] : xyz_min[wall_orientation/2]) - xyz_face[wall_orientation/2])/((*points)[m].p.xyz(wall_orientation/2) - xyz_face[wall_orientation/2]);
        for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
          if(dim == wall_orientation/2)
            wall_eval[dim] = (wall_orientation%2 == 1 ? xyz_max[wall_orientation/2] : xyz_min[wall_orientation/2]); // on the wall of interest
          else
            wall_eval[dim] = MIN(MAX(xyz_face[dim] + lambda*((*points)[m].p.xyz(dim) - xyz_face[dim]), xyz_min[dim] + 2.0*EPS*(xyz_max[dim] - xyz_min[dim])), xyz_max[dim] - 2.0*EPS*(xyz_max[dim] - xyz_min[dim])); // make sure it's indeed inside, just to be safe in case the bc object needs that
        }
        switch(bc_velocity[dir].wallType(wall_eval))
        {
        case DIRICHLET:
        {
          if(dir == wall_orientation/2)
            throw std::runtime_error("my_p4est_two_phase_flows_t::div_mu_grad_u_dir: cannot be called on Dirichlet wall faces...");

          const bool across = my_cell.has_neighbor_across && (sgn_of_wall_neighbor_of_face(face_idx, dir, wall_orientation, wall_eval) != sgn_face);
          // WARNING distance_to_neighbor is actually *twice* what we would need here, hence the "0.5*" factors here under!
          if(!across)
            to_return += mu_this_side*surface*(bc_velocity[dir].wallValue(wall_eval) /*+ interp_dxyz_hodge(wall_eval)*/ - vn_dir_p[face_idx])/(0.5*distance_to_neighbor);
          else
          {
            const FD_interface_neighbor& face_interface_neighbor = interface_manager->get_face_FD_interface_neighbor_for(face_idx, (*points)[m].n, dir, wall_orientation);
            const double& mu_across = (sgn_face > 0 ? mu_minus : mu_plus);
            const bool is_in_positive_domain = (sgn_face > 0);
            to_return += (wall_orientation%2 == 1 ? +1.0 : -1.0)*surface*face_interface_neighbor.GFM_flux_component(mu_this_side, mu_across, wall_orientation, is_in_positive_domain, vn_dir_p[face_idx], bc_velocity[dir].wallValue(wall_eval), 0.0, 0.0, 0.5*distance_to_neighbor);
          }
          break;
        }
        case NEUMANN:
          if(sgn_face != sgn_of_wall_neighbor_of_face(face_idx, dir, wall_orientation, wall_eval))
            throw std::runtime_error("my_p4est_two_phase_flows_t::div_mu_grad_u_dir: Neumann boundary condition to be imposed on a tranverse wall that lies across the interface, but the face has non-uniform neighbors : this is not implemented yet, sorry...");
          to_return += mu_this_side*surface*(bc_velocity[dir].wallValue(wall_eval) /*+ (apply_hodge_second_derivative_if_neumann ? 0.0 : 0.0)*/);
          break;
        default:
          throw std::invalid_argument("my_p4est_two_phase_flows_t::div_mu_grad_u_dir: unknown wall type for a cell that is either one-sided or that has a neighbor across the interface but is not a parallelepiped regular --> not handled yet, TO BE DONE IF NEEDED.");
        }
        break;
      }
      case INTERFACE:
        throw std::logic_error("my_p4est_two_phase_flows_t::div_mu_grad_u_dir: a Voronoi seed neighbor was marked INTERFACE in a cell that is either one-sided or that has a neighbor across the interface but is not a parallelepiped regular. This must not happen in this solver, have you constructed your Voronoi cell using clip_interface()?");
        break;
      default:
        // this is a regular face so
        double xyz_face_neighbor[P4EST_DIM]; faces_n->xyz_fr_f((*points)[m].n, dir, xyz_face_neighbor);
        const bool across = (my_cell.has_neighbor_across && sgn_face != (interface_manager->phi_at_point(xyz_face_neighbor) <= 0.0 ? -1 : +1));
        if(!across)
          to_return += mu_this_side*surface*(vn_dir_p[(*points)[m].n] - vn_dir_p[face_idx])/distance_to_neighbor;
        else
        {
          // std::cerr << "This is bad: your grid is messed up here but, hey, I don't want to crash either..." << std::endl;
          char neighbor_orientation = -1;
          for (u_char dim = 0; dim < P4EST_DIM; ++dim)
            if(fabs(xyz_face_neighbor[dim] - xyz_face[dim]) > 0.1*dxyz_smallest_quad[dim])
              neighbor_orientation = 2*dim + (xyz_face_neighbor[dim] - xyz_face[dim] > 0.0 ? 1 : 0);
          P4EST_ASSERT(fabs(distance_to_neighbor - dxyz_smallest_quad[neighbor_orientation/2]) < 0.001*dxyz_smallest_quad[neighbor_orientation/2]);
          const FD_interface_neighbor& face_interface_neighbor = interface_manager->get_face_FD_interface_neighbor_for(face_idx, (*points)[m].n, dir, neighbor_orientation);
          const double& mu_across = (sgn_face > 0 ? mu_minus : mu_plus);
          const bool is_in_positive_domain = (sgn_face > 0);
          to_return += (neighbor_orientation%2 == 1 ? +1.0 : -1.0)*surface*face_interface_neighbor.GFM_flux_component(mu_this_side, mu_across, neighbor_orientation, is_in_positive_domain, vn_dir_p[face_idx], vn_dir_p[(*points)[m].n], 0.0, 0.0, distance_to_neighbor);
        }
      }
    }
  }
  else
  {
    P4EST_ASSERT(my_cell.cell_type == parallelepiped_no_wall || my_cell.cell_type == parallelepiped_with_wall);
    if(points->size() != P4EST_FACES)
      throw std::runtime_error("my_p4est_two_phase_flows_t::div_mu_grad_u_dir: not the expected number of face neighbors for face with an interface neighbor...");
    P4EST_ASSERT((qm.p.piggy3.local_num == -1 || qm.level == ((const splitting_criteria_t *) p4est_n->user_pointer)->max_lvl)
                 && (qp.p.piggy3.local_num == -1 || qp.level == ((const splitting_criteria_t *) p4est_n->user_pointer)->max_lvl));

    P4EST_ASSERT(fabs(volume - (wall[2*dir] || wall[2*dir + 1] ? 0.5 : 1.0)*MULTD(dxyz_smallest_quad[0], dxyz_smallest_quad[1], dxyz_smallest_quad[2])) < 0.001*(wall[2*dir] || wall[2*dir + 1] ? 0.5 : 1.0)*MULTD(dxyz_smallest_quad[0], dxyz_smallest_quad[1], dxyz_smallest_quad[2])); // half the "regular" volume if the face is a NEUMANN wall face

    for (u_char ff = 0; ff < P4EST_FACES; ++ff)
    {
      // get the face index of the direct face/wall neighbor
      p4est_locidx_t neighbor_face_idx;
      if(my_cell.cell_type == parallelepiped_no_wall)
      {
#ifndef P4_TO_P8
        neighbor_face_idx = (*points)[face_order_to_counterclock_cycle_order[ff]].n;
#else
        neighbor_face_idx = (*points)[ff].n; // already ordered like this, in 3D
#endif
      }
      else // the cell most probably has a wall neighbor, but it's a parallelepiped and it was build by built-in routines (not by hand)
      {
        // the voronoi cell was actually constructed by built-in routines, gathering neighbors etc.
        // but it should still be locally uniform, maybe with wall(s). Let's re-order the neighbors as we need them
        if(wall[ff])
          neighbor_face_idx = WALL_idx(ff);
        else if(ff/2 == dir)
        {
          p4est_locidx_t tmp_quad_idx = (ff%2 == 1 ? qp.p.piggy3.local_num : qm.p.piggy3.local_num);
          P4EST_ASSERT(tmp_quad_idx != -1); // can't be, otherwise wall[ff] would be true...
          neighbor_face_idx = faces_n->q2f(tmp_quad_idx, ff);
        }
        else
        {
          set_of_neighboring_quadrants ngbd;
          ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, ff);
          P4EST_ASSERT(ngbd.size() == 1 && ngbd.begin()->level == quad->level && ngbd.begin()->level == ((const splitting_criteria_t *) p4est_n->user_pointer)->max_lvl);
          neighbor_face_idx = faces_n->q2f(ngbd.begin()->p.piggy3.local_num, face_touch);
        }
#ifdef P4EST_DEBUG
        // check if it was indeed in there for consistency!
        bool found = false;
        for (size_t k = 0; k < points->size() && !found; ++k)
          found = ((*points)[k].n != neighbor_face_idx);
        P4EST_ASSERT(found);
#endif
      }


      // area between the current face and the direct neighbor
      const double neighbor_area = ((wall[2*dir] || wall[2*dir + 1]) && ff/2 != dir ? 0.5 : 1.0)*dxyz_smallest_quad[(ff/2 + 1)%P4EST_DIM] ONLY3D(*dxyz_smallest_quad[(ff/2 + 2)%P4EST_DIM]);
      // get the contribution of the direct neighbor to the discretization of the negative laplacian and add it to the matrix
      if(neighbor_face_idx >= 0)
      {
        double xyz_neighbor_face[P4EST_DIM]; faces_n->xyz_fr_f(neighbor_face_idx, dir, xyz_neighbor_face);
        const char sgn_neighbor_face = (interface_manager->phi_at_point(xyz_neighbor_face) <= 0.0 ? -1 : 1);
        const bool across = (sgn_face != sgn_neighbor_face);
        if(across)
        {
          const FD_interface_neighbor& face_interface_neighbor = interface_manager->get_face_FD_interface_neighbor_for(face_idx, neighbor_face_idx, dir, ff);
          const double& mu_this_side  = (sgn_face < 0 ? mu_minus  : mu_plus);
          const double& mu_across     = (sgn_face < 0 ? mu_plus   : mu_minus);
          const bool is_in_positive_domain = (sgn_face > 0);
          to_return += (ff%2 == 1 ? +1.0 : -1.0)*neighbor_area*face_interface_neighbor.GFM_flux_component(mu_this_side, mu_across, ff, is_in_positive_domain, vn_dir_p[face_idx], vn_dir_p[neighbor_face_idx], 0.0, 0.0, dxyz_smallest_quad[ff/2]);
        }
        else
          to_return += ((sgn_face < 0 ? mu_minus : mu_plus)*neighbor_area/dxyz_smallest_quad[ff/2])*(vn_dir_p[neighbor_face_idx] - vn_dir_p[face_idx]);
      }
      else
      {
        P4EST_ASSERT(-1 - neighbor_face_idx == ff);
        if(ff/2 == dir) // parallel wall --> the face itself is the wall --> it *cannot* be DIRICHLET
        {
          P4EST_ASSERT(wall[ff] && bc_velocity[dir].wallType(xyz_face) == NEUMANN); // the face is a wall face so it MUST be NEUMANN (non-DIRICHLET) boundary condition in that case
          to_return += neighbor_area*bc_velocity[dir].wallValue(xyz_face);
        }
        else // it is a tranverse wall
        {
          double xyz_wall[P4EST_DIM] = {DIM(xyz_face[0], xyz_face[1], xyz_face[2])};
          xyz_wall[ff/2] = (ff%2 == 1 ? xyz_max[ff/2] : xyz_min[ff/2]);
          const bool across = (sgn_of_wall_neighbor_of_face(face_idx, dir, ff, xyz_wall) != sgn_face);
          if(across) // the tranverse wall is across the interface
          {
            const FD_interface_neighbor& face_interface_neighbor = interface_manager->get_face_FD_interface_neighbor_for(face_idx, neighbor_face_idx, dir, ff);
            // /!\ WARNING /!\ : theta is relative to 0.5*dxyz_min[ff/2] in this case!
            const double& mu_this_side  = (sgn_face < 0 ? mu_minus  : mu_plus);
            const double& mu_across     = (sgn_face < 0 ? mu_plus   : mu_minus);
            const bool is_in_positive_domain = (sgn_face > 0);

            switch (bc_velocity[dir].wallType(xyz_wall)) {
            case DIRICHLET:
            {
              to_return += (ff%2 == 1 ? 1.0 : -1.0)*neighbor_area*face_interface_neighbor.GFM_flux_component(mu_this_side, mu_across, ff, is_in_positive_domain, vn_dir_p[face_idx], bc_velocity[dir].wallValue(xyz_wall), 0.0, 0.0, 0.5*dxyz_smallest_quad[ff/2]);
              break;
            }
            case NEUMANN:
            {
              to_return += neighbor_area*(bc_velocity[dir].wallValue(xyz_wall) + (sgn_face < 0 ? -1.0 : +1.0)*(ff%2 == 1 ? +1.0 : -1.0)*0.0);
              break;
            }
            default:
              throw std::invalid_argument("my_p4est_two_phase_flows_t::div_mu_grad_u_dir: unknown wall type for a tranverse wall neighbor of a face, across the interface --> not handled yet, TO BE DONE...");
              break;
            }
          }
          else // the tranverse wall is on the same side of the interface
          {
            switch (bc_velocity[dir].wallType(xyz_wall)) {
            case DIRICHLET:
              to_return += 2.0*((sgn_face < 0 ? mu_minus : mu_plus)*neighbor_area/dxyz_smallest_quad[ff/2])*(bc_velocity[dir].wallValue(xyz_wall) - vn_dir_p[face_idx]);
              break;
            case NEUMANN:
              to_return += neighbor_area*bc_velocity[dir].wallValue(xyz_wall);
              break;
            default:
              throw std::invalid_argument("my_p4est_two_phase_flows_t::div_mu_grad_u_dir: unknown wall type for a tranverse wall neighbor of a face, on the same side of the interface --> not handled yet, TO BE DONE...");
              break;
            }
          }
        }
      }
    }
  }
  return to_return/volume;
}

void my_p4est_two_phase_flows_t::solve_viscosity()
{
  compute_backtraced_velocities();
  viscosity_solver.set_diagonals(BDF_alpha()*rho_minus/dt_n, BDF_alpha()*rho_plus/dt_n);
  viscosity_solver.solve();
  viscosity_solver.get_vstar_velocities(vnp1_face_minus, vnp1_face_plus);
}

void my_p4est_two_phase_flows_t::jump_face_solver::solve(const PetscBool &use_nonzero_initial_guess, const KSPType &ksp_type, const PCType &pc_type)
{
  PetscErrorCode ierr;

  for(u_char dir = 0; dir < P4EST_DIM; ++dir)
    if(solution[dir] == NULL){
      ierr = VecCreateGhostFaces(env->p4est_n, env->faces_n, &solution[dir], dir); CHKERRXX(ierr);
    }

#ifdef CASL_THROWS
  if(env->bc_velocity == NULL && !ANDD(env->periodicity[0], env->periodicity[1], env->periodicity[2]))
    throw std::domain_error("my_p4est_two_phase_flows_t::jump_face_solver::solve(): the boundary conditions on velocity components have not been set.");
#endif

  for(u_char dir = 0; dir < P4EST_DIM; ++dir)
  {
    /* assemble the linear system if required, and initialize the Krylov solver and its preconditioner based on that*/
    setup_linear_system(dir);
    setup_linear_solver(dir, use_nonzero_initial_guess, ksp_type, pc_type);

    /* solve the system */
    ierr = KSPSolve(ksp[dir], rhs[dir], solution[dir]); CHKERRXX(ierr);

    /* start ghost update but don't block the process, keep working... */
    ierr = VecGhostUpdateBegin(solution[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  /* finish the ghost update, now... */
  for(u_char dir = 0; dir < P4EST_DIM; ++dir){
    ierr = VecGhostUpdateEnd  (solution[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }
}

void my_p4est_two_phase_flows_t::jump_face_solver::setup_linear_solver(const u_char &dir, const PetscBool &use_nonzero_initial_guess, const KSPType &ksp_type, const PCType &pc_type)
{
  PetscErrorCode ierr;

  P4EST_ASSERT(ksp[dir] != NULL);
  /* set ksp type */
  KSPType ksp_type_as_such;
  ierr = KSPGetType(ksp[dir], &ksp_type_as_such); CHKERRXX(ierr);
  if(ksp_type != ksp_type_as_such){
    ierr = KSPSetType(ksp[dir], ksp_type); CHKERRXX(ierr); }

  PetscBool ksp_initial_guess;
  ierr = KSPGetInitialGuessNonzero(ksp[dir], &ksp_initial_guess); CHKERRXX(ierr);
  if (ksp_initial_guess != use_nonzero_initial_guess){
    ierr = KSPSetInitialGuessNonzero(ksp[dir], use_nonzero_initial_guess); CHKERRXX(ierr); }
  if(!ksp_is_set_from_options[dir])
  {
    ierr = KSPSetFromOptions(ksp[dir]); CHKERRXX(ierr);
    ksp_is_set_from_options[dir] = true;
  }

  /* set pc type */
  PC pc;
  ierr = KSPGetPC(ksp[dir], &pc); CHKERRXX(ierr);
  P4EST_ASSERT(pc != NULL);
  PCType pc_type_as_such;
  ierr = PCGetType(pc, &pc_type_as_such); CHKERRXX(ierr);
  if(pc_type_as_such != pc_type)
  {
    ierr = PCSetType(pc, pc_type); CHKERRXX(ierr);

    /* If using hypre, we can make some adjustments here. The most important parameters to be set are:
     * 1- Strong Threshold
     * 2- Coarsennig Type
     * 3- Truncation Factor
     *
     * Please refer to HYPRE manual for more information on the actual importance or check Mohammad Mirzadeh's
     * summary of HYPRE papers! Also for a complete list of all the options that can be set from PETSc, one can
     * consult the 'src/ksp/pc/impls/hypre.c' in the PETSc home directory.
     */
    if (!strcmp(pc_type, PCHYPRE)){
      /* 1- Strong threshold:
       * Between 0 to 1
       * "0 "gives better convergence rate (in 3D).
       * Suggested values (By Hypre manual): 0.25 for 2D, 0.5 for 3D
      */
      ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_strong_threshold", "0.5"); CHKERRXX(ierr);

      /* 2- Coarsening type
       * Available Options:
       * "CLJP","Ruge-Stueben","modifiedRuge-Stueben","Falgout", "PMIS", "HMIS". Falgout is usually the best.
       */
      ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_coarsen_type", "Falgout"); CHKERRXX(ierr);

      /* 3- Truncation factor
       * Greater than zero.
       * Use zero for the best convergence. However, if you have memory problems, use greater than zero to save some memory.
       */
      ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_truncfactor", "0.1"); CHKERRXX(ierr);
    }
  }
  if(!pc_is_set_from_options[dir])
  {
    ierr = PCSetFromOptions(pc); CHKERRXX(ierr);
    pc_is_set_from_options[dir] = true;
  }
}

void my_p4est_two_phase_flows_t::jump_face_solver::preallocate_matrix(const u_char &dir)
{
  if(matrix_is_preallocated[dir])
    return;

  PetscErrorCode ierr;
  PetscInt num_owned_local  = (PetscInt) env->faces_n->num_local[dir];
  PetscInt num_owned_global = (PetscInt) env->faces_n->proc_offset[dir][env->p4est_n->mpisize];

  if(matrix[dir] != NULL){
    ierr = MatDestroy(matrix[dir]); CHKERRXX(ierr); }

  /* set up the matrix */
  ierr = MatCreate(env->p4est_n->mpicomm, &matrix[dir]); CHKERRXX(ierr);
  ierr = MatSetType(matrix[dir], MATAIJ); CHKERRXX(ierr);
  ierr = MatSetSizes(matrix[dir], num_owned_local , num_owned_local,
                     num_owned_global, num_owned_global); CHKERRXX(ierr);
  ierr = MatSetFromOptions(matrix[dir]); CHKERRXX(ierr);

  vector<PetscInt> d_nnz(num_owned_local, 1), o_nnz(num_owned_local, 0);

  for(p4est_locidx_t f_idx = 0; f_idx < env->faces_n->num_local[dir]; ++f_idx)
  {
    if(!face_is_dirichlet_wall(f_idx, dir))
    {
      const augmented_voronoi_cell my_cell = env->get_augmented_voronoi_cell(f_idx, dir);
      const vector<ngbdDIMseed> *points;
      my_cell.voro.get_neighbor_seeds(points);

      for(size_t n = 0; n < points->size(); ++n)
        if((*points)[n].n >= 0)
        {
          if((*points)[n].n < num_owned_local)  d_nnz[f_idx]++;
          else                                  o_nnz[f_idx]++;
        }
    }
  }

  ierr = MatSeqAIJSetPreallocation(matrix[dir], 0, (const PetscInt*)&d_nnz[0]); CHKERRXX(ierr);
  ierr = MatMPIAIJSetPreallocation(matrix[dir], 0, (const PetscInt*)&d_nnz[0], 0, (const PetscInt*)&o_nnz[0]); CHKERRXX(ierr);

  matrix_is_preallocated[dir] = true;
}

void my_p4est_two_phase_flows_t::jump_face_solver::setup_linear_system(const u_char &dir)
{
  PetscErrorCode ierr;

  // check that the current "diagonal" is as desired if the matrix is ready to go...
  P4EST_ASSERT(!matrix_is_ready[dir] || current_diags_are_as_desired(dir));
  if(!only_diags_are_modified[dir] && !matrix_is_ready[dir])
  {
    reset_current_diagonals(dir);
    /* preallocate the matrix and compute the voronoi partition */
    preallocate_matrix(dir);
  }

  matrix_has_nullspace[dir] = true;

  double *rhs_p;
  if(rhs[dir] == NULL){
    ierr = VecCreateNoGhostFaces(env->p4est_n, env->faces_n, &rhs[dir], dir); CHKERRXX(ierr);
  }
  ierr = VecGetArray(rhs[dir], &rhs_p); CHKERRXX(ierr);

  for(p4est_locidx_t f_idx = 0; f_idx < env->faces_n->num_local[dir]; ++f_idx)
  {
    p4est_gloidx_t f_idx_g = env->faces_n->global_index(f_idx, dir);

    p4est_locidx_t quad_idx;
    p4est_topidx_t tree_idx;
    env->faces_n->f2q(f_idx, dir, quad_idx, tree_idx);
    const p4est_quadrant_t *quad = fetch_quad(quad_idx, tree_idx, env->p4est_n, env->ghost_n);

    const u_char face_touch = (env->faces_n->q2f(quad_idx, 2*dir) == f_idx ? 2*dir : 2*dir + 1);
    P4EST_ASSERT(env->faces_n->q2f(quad_idx, face_touch) == f_idx);

    double xyz_face[P4EST_DIM]; env->faces_n->xyz_fr_f(f_idx, dir, xyz_face);

    p4est_quadrant_t qm, qp;
    env->faces_n->find_quads_touching_face(f_idx, dir, qm, qp);
    /* check for walls */
    if(qm.p.piggy3.local_num == -1 && env->bc_velocity[dir].wallType(xyz_face) == DIRICHLET)
    {
      matrix_has_nullspace[dir] = false;
      if(!only_diags_are_modified[dir] && !matrix_is_ready[dir]) {
        ierr = MatSetValue(matrix[dir], f_idx_g, f_idx_g, 1.0, ADD_VALUES); CHKERRXX(ierr); } // needs to be done only if fully reset
      rhs_p[f_idx] = env->bc_velocity[dir].wallValue(xyz_face); //  + interp_dxyz_hodge(xyz);
      continue;
    }
    if(qp.p.piggy3.local_num == -1 && env->bc_velocity[dir].wallType(xyz_face) == DIRICHLET)
    {
      matrix_has_nullspace[dir] = false;
      if(!only_diags_are_modified[dir] && !matrix_is_ready[dir]) {
        ierr = MatSetValue(matrix[dir], f_idx_g, f_idx_g, 1.0, ADD_VALUES); CHKERRXX(ierr); }// needs to be done only if fully reset
      rhs_p[f_idx] = env->bc_velocity[dir].wallValue(xyz_face); //  + interp_dxyz_hodge(xyz);
      continue;
    }

    bool wall[P4EST_FACES];
    for(u_char d = 0; d < P4EST_DIM; ++d)
    {
      if(d == dir)
      {
        wall[2*d]     = qm.p.piggy3.local_num == -1;
        wall[2*d + 1] = qp.p.piggy3.local_num == -1;
      }
      else
      {
        wall[2*d]     = (qm.p.piggy3.local_num == -1 || is_quad_Wall(env->p4est_n, qm.p.piggy3.which_tree, &qm, 2*d    )) && (qp.p.piggy3.local_num == -1 || is_quad_Wall(env->p4est_n, qp.p.piggy3.which_tree, &qp, 2*d));
        wall[2*d + 1] = (qm.p.piggy3.local_num == -1 || is_quad_Wall(env->p4est_n, qm.p.piggy3.which_tree, &qm, 2*d + 1)) && (qp.p.piggy3.local_num == -1 || is_quad_Wall(env->p4est_n, qp.p.piggy3.which_tree, &qp, 2*d + 1));
      }
    }

    const augmented_voronoi_cell my_cell = env->get_augmented_voronoi_cell(f_idx, dir);

    const vector<ngbdDIMseed> *points;
#ifndef P4_TO_P8
    const vector<Point2> *partition;
    my_cell.voro.get_partition(partition);
#endif
    my_cell.voro.get_neighbor_seeds(points);

    const double volume = my_cell.voro.get_volume();
    const char sgn_face = (env->interface_manager->phi_at_point(xyz_face) <= 0.0 ? -1 : +1);

    if((sgn_face < 0 ? desired_diag_minus[dir] : desired_diag_plus[dir]) > 0.0)
      matrix_has_nullspace[dir] = false;
    if(!matrix_is_ready[dir]) {
      ierr = MatSetValue(matrix[dir], f_idx_g, f_idx_g, (sgn_face < 0 ? desired_diag_minus[dir] - current_diag_minus[dir] : desired_diag_plus[dir] - current_diag_plus[dir])*volume, ADD_VALUES); CHKERRXX(ierr);
      // (note: current_diag_m and current_diag_p are both initially set to be 0.0 --> no conflict here)
    }

    // get the rhs from the solver's data, or for validation purposes:
    P4EST_ASSERT(env->semi_lagrangian_backtrace_is_done);
    const double& rho                           = (sgn_face < 0 ? env->rho_minus                  : env->rho_plus);
    const std::vector<double>* backtraced_vn    = (sgn_face < 0 ? env->backtraced_vn_faces_minus  : env->backtraced_vn_faces_plus);
    const std::vector<double>* backtraced_vnm1  = (env->sl_order == 2 ? (sgn_face < 0 ? env->backtraced_vnm1_faces_minus : env->backtraced_vnm1_faces_plus) : NULL);

    rhs_p[f_idx]  = rho*(env->BDF_alpha()/env->dt_n - env->BDF_beta()/env->dt_nm1)*backtraced_vn[dir][f_idx]
        + (env->sl_order == 2 ? (rho*env->BDF_beta()/env->dt_nm1)*backtraced_vnm1[dir][f_idx] : 0.0); // alpha and beta are 1 and 0 if sl_order = 1...

    if (env->force_per_unit_mass[dir] != NULL)
      rhs_p[f_idx] += rho*(*env->force_per_unit_mass[dir])(xyz_face);
    // multiply by volume and here starts the fun, now!
    rhs_p[f_idx] *= volume;

    if(my_cell.has_neighbor_across && (my_cell.cell_type == parallelepiped_no_wall || my_cell.cell_type == parallelepiped_with_wall))
    {
      // some (x)GFM discretization, multiplied by the volume of the parallelipiped cell (*not* cut/clipped by the interface)
      P4EST_ASSERT((qm.p.piggy3.local_num == -1 || qm.level == ((const splitting_criteria_t *) env->p4est_n->user_pointer)->max_lvl)
                   && (qp.p.piggy3.local_num == -1 || qp.level == ((const splitting_criteria_t *) env->p4est_n->user_pointer)->max_lvl));

      // Although intrinsically finite difference in nature for this case, we multiply every "flux contribution" to the local
      // discretized negative laplacian by the area of the face of the associated FV cell, clipped in domain (i.e. regular
      // expected tranverse grid cell area, except in presence of NEUMANN wall faces). The "flux contribution" are evaluated
      // using standard or (x)GFM-like approximations or by wall value in case of NEUMANN wall boundary condition.
      // - We do NOT clip those FV cells by the interface in any way; in other words, the presence of the interface is irrelevant
      // to the definition of those areas.
      // - The right hand side and diagonal contributions are consistently multiplied by the volume of the FV cell (see above).
      // --> This is done in order to ensure symmetry and consistency between off-diagonal weights by comparison with other
      // discretized equations involving nearby faces that do not have any neighbor across the interface
      // (reminder: if no neigbor across, FV on Voronoi cells)
      P4EST_ASSERT(fabs(volume - (wall[2*dir] || wall[2*dir + 1] ? 0.5 : 1.0)*MULTD(env->dxyz_smallest_quad[0], env->dxyz_smallest_quad[1], env->dxyz_smallest_quad[2])) < 0.001*(wall[2*dir] || wall[2*dir + 1] ? 0.5 : 1.0)*MULTD(env->dxyz_smallest_quad[0], env->dxyz_smallest_quad[1], env->dxyz_smallest_quad[2])); // half the "regular" volume if the face is a NEUMANN wall face

      for (u_char ff = 0; ff < P4EST_FACES; ++ff)
      {
        // get the face index of the direct face/wall neighbor
        p4est_locidx_t neighbor_face_idx;
        if(my_cell.cell_type == parallelepiped_no_wall)
        {
#ifndef P4_TO_P8
          neighbor_face_idx = (*points)[face_order_to_counterclock_cycle_order[ff]].n;
#else
          neighbor_face_idx = (*points)[ff].n; // already ordered like this, in 3D
#endif
        }
        else // the cell most probably has a wall neighbor, but it's a parallelepiped and it was build by built-in routines (not by hand)
        {
          // the voronoi cell was actually constructed by built-in routines, gathering neighbors etc.
          // but it should still be locally uniform, maybe with wall(s). Let's re-order the neighbors as we need them
          if(wall[ff])
            neighbor_face_idx = WALL_idx(ff);
          else if(ff/2 == dir)
          {
            p4est_locidx_t tmp_quad_idx = (ff%2 == 1 ? qp.p.piggy3.local_num : qm.p.piggy3.local_num);
            P4EST_ASSERT(tmp_quad_idx != -1); // can't be, otherwise wall[ff] would be true...
            neighbor_face_idx = env->faces_n->q2f(tmp_quad_idx, ff);
          }
          else
          {
            set_of_neighboring_quadrants ngbd;
            env->ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, ff);
            P4EST_ASSERT(ngbd.size() == 1 && ngbd.begin()->level == quad->level && ngbd.begin()->level == ((const splitting_criteria_t *) env->p4est_n->user_pointer)->max_lvl);
            neighbor_face_idx = env->faces_n->q2f(ngbd.begin()->p.piggy3.local_num, face_touch);
          }
#ifdef P4EST_DEBUG
          // check if it was indeed in there for consistency!
          bool found = false;
          for (size_t k = 0; k < points->size() && !found; ++k)
            found = ((*points)[k].n != neighbor_face_idx);
          P4EST_ASSERT(found);
#endif
        }

        // area between the current face and the direct neighbor
        // this area should be the standard, regular finest uniform-grid area EXCEPT if the considered face is a wall face and we are looking in a tranverse direction
        const double neighbor_area = ((wall[2*dir] || wall[2*dir + 1]) && ff/2 != dir ? 0.5 : 1.0)*env->dxyz_smallest_quad[(ff/2 + 1)%P4EST_DIM] ONLY3D(*env->dxyz_smallest_quad[(ff/2 + 2)%P4EST_DIM]);
        double offdiag_coeff;
        // get the contribution of the direct neighbor to the discretization of the negative laplacian and add it to the matrix
        if(neighbor_face_idx >= 0)
        {
          double xyz_neighbor_face[P4EST_DIM]; env->faces_n->xyz_fr_f(neighbor_face_idx, dir, xyz_neighbor_face);
          const char sgn_neighbor_face = (env->interface_manager->phi_at_point(xyz_neighbor_face) <= 0.0 ? -1 : +1);
          if(sgn_neighbor_face != sgn_face)
          {
            const FD_interface_neighbor& face_interface_neighbor = env->interface_manager->get_face_FD_interface_neighbor_for(f_idx, neighbor_face_idx, dir, ff);
            const double& mu_this_side  = (sgn_face < 0 ? env->mu_minus  : env->mu_plus);
            const double& mu_across     = (sgn_face < 0 ? env->mu_plus   : env->mu_minus);
            const bool is_in_positive_domain = (sgn_face > 0);
            offdiag_coeff = -face_interface_neighbor.GFM_mu_jump(mu_this_side, mu_across)*neighbor_area/env->dxyz_smallest_quad[ff/2];
            rhs_p[f_idx] += neighbor_area*(ff%2 == 1 ? +1.0 : -1.0)*face_interface_neighbor.GFM_jump_terms_for_flux_component(mu_this_side, mu_across, ff, is_in_positive_domain, 0.0, 0.0, env->dxyz_smallest_quad[ff/2]);
          }
          else
            offdiag_coeff   = -(sgn_face ? env->mu_minus : env->mu_plus)*neighbor_area/env->dxyz_smallest_quad[ff/2];

          // check if the neighbor face is a wall (to subtract it from the RHS right away if it is a Dirichlet BC and keep a symmetric matrix for CG)
          // the neigbor face is a wall, if
          // - either we are currently at a non-wall face and we are fetching a neighbor in a face-normal direction that happens to lie on a wall
          // - or we are currently at a wall face (which could happen if the current face is Neumann face) and we are fetching any face neighbor in a tranverse direction
          const bool neighbor_face_is_wall = (ff/2 == dir && (ff%2 == 1 ? is_quad_Wall(env->p4est_n, qp.p.piggy3.which_tree, &qp, ff) : is_quad_Wall(env->p4est_n, qm.p.piggy3.which_tree, &qm, ff))) || ((wall[2*dir] || wall[2*dir + 1]) && ff/2 != dir);
          double xyz_face_neighbor[P4EST_DIM];
          if(neighbor_face_is_wall)
            env->faces_n->xyz_fr_f(neighbor_face_idx, dir, xyz_face_neighbor); // we need it only if it's wall, avoid useless steps...
          // this is a regular face so
          if(!only_diags_are_modified[dir] && !matrix_is_ready[dir]){
            // we do not add the off-diagonal element to the matrix if the neighbor is a Dirichlet wall face
            // but we modify the rhs correspondingly, right away instead (see if statement here below)
            // --> ensures full symmetry of the matrix, one can use CG solver, safely!
            if(!neighbor_face_is_wall || env->bc_velocity[dir].wallType(xyz_face_neighbor) != DIRICHLET) {
              ierr = MatSetValue(matrix[dir], f_idx_g, env->faces_n->global_index(neighbor_face_idx, dir),  offdiag_coeff, ADD_VALUES); CHKERRXX(ierr); }
            ierr = MatSetValue(matrix[dir], f_idx_g, f_idx_g, -offdiag_coeff, ADD_VALUES); CHKERRXX(ierr);
          }
          if(neighbor_face_is_wall && env->bc_velocity[dir].wallType(xyz_face_neighbor) == DIRICHLET)
            rhs_p[f_idx] -= offdiag_coeff*(env->bc_velocity[dir].wallValue(xyz_face_neighbor) /* + interp_dxyz_hodge(xyz_face_neighbor) <- maybe this requires global interpolation /!\ */);
        }
        else
        {
          P4EST_ASSERT(-1 - neighbor_face_idx == ff);
          if(ff/2 == dir) // parallel wall --> the face itself is the wall --> it *cannot* be DIRICHLET
          {
            P4EST_ASSERT(wall[ff] && env->bc_velocity[dir].wallType(xyz_face) == NEUMANN); // the face is a wall face so it MUST be NEUMANN (non-DIRICHLET) boundary condition in that case
            // the contribution of the the local term in the discretization (to the negative laplacian) is
            // -(wall value of NEUMANN boundary condition)*neighbor_area --> goes ot RHS
            rhs_p[f_idx] += neighbor_area*env->bc_velocity[dir].wallValue(xyz_face);
            // no subtraction/addition to diagonal in this, and a nullspace still exists
          }
          else // it is a tranverse wall
          {
            double xyz_wall[P4EST_DIM] = {DIM(xyz_face[0], xyz_face[1], xyz_face[2])};
            xyz_wall[ff/2] = (ff%2 == 1 ? env->xyz_max[ff/2] : env->xyz_min[ff/2]);
            const bool across = (env->sgn_of_wall_neighbor_of_face(f_idx, dir, ff, xyz_wall) != sgn_face);
            if(across) // the tranverse wall is across the interface
            {
              const FD_interface_neighbor& face_interface_neighbor = env->interface_manager->get_face_FD_interface_neighbor_for(f_idx, neighbor_face_idx, dir, ff);
              // /!\ WARNING /!\ : theta is relative to 0.5*dxyz_min[ff/2] in this case!
              const double& mu_this_side  = (sgn_face < 0 ? env->mu_minus  : env->mu_plus);
              const double& mu_across     = (sgn_face < 0 ? env->mu_plus   : env->mu_minus);
              const bool is_in_positive_domain = (sgn_face > 0);

              switch (env->bc_velocity[dir].wallType(xyz_wall)) {
              case DIRICHLET:
              {
                offdiag_coeff = -face_interface_neighbor.GFM_mu_jump(mu_this_side, mu_across)*neighbor_area/(0.5*env->dxyz_smallest_quad[ff/2]);
                rhs_p[f_idx] += neighbor_area*(ff%2 == 1 ? +1.0 : -1.0)*face_interface_neighbor.GFM_jump_terms_for_flux_component(mu_this_side, mu_across, ff, is_in_positive_domain, 0.0, 0.0, 0.5*env->dxyz_smallest_quad[ff/2]);
                if(!only_diags_are_modified[dir] && !matrix_is_ready[dir])
                {
                  ierr = MatSetValue(matrix[dir], f_idx_g, f_idx_g, -offdiag_coeff, ADD_VALUES); CHKERRXX(ierr);
                }
                matrix_has_nullspace[dir] = false; // no nullspace in this case...
                break;
              }
              case NEUMANN:
              {
                // The tranverse wall has a NEUMANN boundary condition but it is across the interface
                // We will make that an equivalent NEUMANN boundary condition on this_sided field, subtracting/adding the known flux in jump component
                // We take a crude estimation: (wall_value +/- jump_in_flux_component) since jump_in_flux_component is 1st order only (and if we do
                // things right), anyways...
                // Therefore, the contribution of the local term in the discretization (to the negative laplacian) is
                //
                // mu_this_side*(u[f_idx] - "ghost_across_wall")*neighbor_area/dxyz_min[ff/2]
                // = -(wall value of NEUMANN boundary condition + (my_cell.is_in_negative_domain ? -1.0 : +1.0)*(ff%2 == 1 ? +1.0 : -1.0)*jump_flux_component)*neighbor_area
                //
                // therefore we have
                rhs_p[f_idx] += neighbor_area*(env->bc_velocity[dir].wallValue(xyz_wall) + (sgn_face < 0 ? -1.0 : +1.0)*(ff%2 == 1 ? +1.0 : -1.0)*0.0);
                // no subtraction/addition to diagonal in this, and a nullspace still exists
                break;
              }
              default:
                throw std::invalid_argument("my_p4est_two_phase_flows_t::jump_face_solver::setup_linear_solver: unknown wall type for a tranverse wall neighbor of a face, across the interface --> not handled yet, TO BE DONE...");
                break;
              }
            }
            else // the tranverse wall is on the same side of the interface
            {
              switch (env->bc_velocity[dir].wallType(xyz_wall)) {
              case DIRICHLET:
                // ghost value = 2*wall_value - center value, so
                //
                // mu_this_side*(u[f_idx] - ghost_value)*neighbor_area/dxyz_min[ff/2]
                // = 2.0*mu_this_side*(u[f_idx] - wall_value)*neighbor_area/dxyz_min[ff/2]
                // --> no nullspace anymore
                offdiag_coeff =  -2.0*(sgn_face < 0 ? env->mu_minus : env->mu_plus)*neighbor_area/env->dxyz_smallest_quad[ff/2];
                rhs_p[f_idx] -= offdiag_coeff*env->bc_velocity[dir].wallValue(xyz_wall);
                if(!only_diags_are_modified[dir] && !matrix_is_ready[dir]) {
                  ierr = MatSetValue(matrix[dir], f_idx_g, f_idx_g, -offdiag_coeff, ADD_VALUES); CHKERRXX(ierr); }
                matrix_has_nullspace[dir] = false; // no nullspace in this case...
                break;
              case NEUMANN:
                // The contribution of the the local term in the discretization (to the negative laplacian) is
                //
                // mu_this_side*(u[f_idx] - "ghost_across_wall")*neighbor_area/dxyz_min[ff/2]
                // = -(wall value of NEUMANN boundary condition)*neighbor_area
                rhs_p[f_idx] += neighbor_area*env->bc_velocity[dir].wallValue(xyz_wall);
                // no subtraction/addition to diagonal in this, and a nullspace still exists
                break;
              default:
                throw std::invalid_argument("my_p4est_two_phase_flows_t::jump_face_solver::setup_linear_solver: unknown wall type for a tranverse wall neighbor of a face, on the same side of the interface --> not handled yet, TO BE DONE...");
                break;
              }
            }
          }
        }
      } /* end of going through uniform neighbors in face order, taking care of jump conditions, where needed */
      continue;
    }
    else // "regular" stuff --> FV on voronoi cell, no neighbor across so let's roll!
    {
      for(size_t m = 0; m < points->size(); ++m)
      {
#ifdef P4_TO_P8
        const double surface = (*points)[m].s;
#else
        size_t k = mod(m - 1, points->size());
        const double surface = ((*partition)[m] - (*partition)[k]).norm_L2();
#endif
        const double distance_to_neighbor = sqrt(SUMD(SQR((*points)[m].p.x - xyz_face[0]), SQR((*points)[m].p.y - xyz_face[1]), SQR((*points)[m].p.z - xyz_face[2])));
        const double mu_this_side = (sgn_face < 0 ? env->mu_minus : env->mu_plus);

        switch((*points)[m].n)
        {
        case WALL_m00:
        case WALL_p00:
        case WALL_0m0:
        case WALL_0p0:
#ifdef P4_TO_P8
        case WALL_00m:
        case WALL_00p:
#endif
        {
          char wall_orientation = -1 - (*points)[m].n;
          P4EST_ASSERT(wall_orientation >= 0 && wall_orientation < P4EST_FACES);
          double wall_eval[P4EST_DIM];
          const double lambda = ((wall_orientation%2 == 1 ? env->xyz_max[wall_orientation/2] : env->xyz_min[wall_orientation/2]) - xyz_face[wall_orientation/2])/((*points)[m].p.xyz(wall_orientation/2) - xyz_face[wall_orientation/2]);
          for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
            if(dim == wall_orientation/2)
              wall_eval[dim] = (wall_orientation%2 == 1 ? env->xyz_max[wall_orientation/2] : env->xyz_min[wall_orientation/2]); // on the wall of interest
            else
              wall_eval[dim] = MIN(MAX(xyz_face[dim] + lambda*((*points)[m].p.xyz(dim) - xyz_face[dim]), env->xyz_min[dim] + 2.0*EPS*(env->xyz_max[dim] - env->xyz_min[dim])), env->xyz_max[dim] - 2.0*EPS*(env->xyz_max[dim] - env->xyz_min[dim])); // make sure it's indeed inside, just to be safe in case the bc object needs that
          }
          switch(env->bc_velocity[dir].wallType(wall_eval))
          {
          case DIRICHLET:
          {
            if(dir == wall_orientation/2)
              throw std::runtime_error("my_p4est_two_phase_flows_t::jump_face_solver::setup_linear_solver: Dirichlet boundary conditions on walls parallel to faces should have been done before... You might be using an unconventional aspect ratio for your cells: if yes, it is not taken care yet, sorry!");
            const bool across = my_cell.has_neighbor_across && (env->sgn_of_wall_neighbor_of_face(f_idx, dir, wall_orientation, wall_eval) != sgn_face);
            // WARNING distance_to_neighbor is actually *twice* what we would need here, hence the "0.5*" factors here under!
            double offdiag_coeff;
            // WARNING distance_to_neighbor is actually *twice* what we would need here, hence the "0.5*" factors here under!
            if(!across)
              offdiag_coeff = -mu_this_side*surface/(0.5*distance_to_neighbor);
            else
            {
              const FD_interface_neighbor& face_interface_neighbor = env->interface_manager->get_face_FD_interface_neighbor_for(f_idx, (*points)[m].n, dir, wall_orientation);
              const double& mu_across = (sgn_face > 0 ? env->mu_minus : env->mu_plus);
              const bool is_in_positive_domain = (sgn_face > 0);
              offdiag_coeff = -surface*face_interface_neighbor.GFM_mu_jump(mu_this_side, mu_across)/(0.5*distance_to_neighbor);
              rhs_p[f_idx] += surface*(wall_orientation%2 == 1 ? +1.0 : -1.0)*face_interface_neighbor.GFM_jump_terms_for_flux_component(mu_this_side, mu_across, wall_orientation, is_in_positive_domain, 0.0, 0.0, 0.5*distance_to_neighbor);
            }
            matrix_has_nullspace[dir] = false;
            if(!only_diags_are_modified[dir] && !matrix_is_ready[dir]) {
              ierr = MatSetValue(matrix[dir], f_idx_g, f_idx_g, -offdiag_coeff, ADD_VALUES); CHKERRXX(ierr); } // needs only to be done if fully reset
            rhs_p[f_idx] -= offdiag_coeff*(env->bc_velocity[dir].wallValue(wall_eval) /*+ interp_dxyz_hodge(wall_eval)*/);
            break;
          }
          case NEUMANN:
            if(sgn_face != env->sgn_of_wall_neighbor_of_face(f_idx, dir, wall_orientation, wall_eval))
              throw std::runtime_error("my_p4est_two_phase_flows_t::jump_face_solver::setup_linear_solver: Neumann boundary condition to be imposed on a tranverse wall that lies across the interface, but the face has non-uniform neighbors : this is not implemented yet, sorry...");
            rhs_p[f_idx] += mu_this_side*surface*(env->bc_velocity[dir].wallValue(wall_eval) /*+ (apply_hodge_second_derivative_if_neumann ? 0.0 : 0.0)*/); // apply_hodge_second_derivative_if_neumann: would need to be fixed later --> good luck!
            break;
          default:
            throw std::invalid_argument("my_p4est_two_phase_flows_t::jump_face_solver::setup_linear_solver: unknown wall type for a cell that is either one-sided or that has a neighbor across the interface but is not a parallelepiped regular --> not handled yet, TO BE DONE IF NEEDED.");
          }
          break;
        }
        case INTERFACE:
          throw std::logic_error("my_p4est_two_phase_flows_t::jump_face_solver::setup_linear_solver: a Voronoi seed neighbor was marked INTERFACE in a cell that is either one-sided or that has a neighbor across the interface but is not a parallelepiped regular. This must not happen in this solver, have you constructed your Voronoi cell using clip_interface()?");
          break;
        default:
          // this is a regular face so
          double xyz_face_neighbor[P4EST_DIM]; env->faces_n->xyz_fr_f((*points)[m].n, dir, xyz_face_neighbor);
          const bool across = (my_cell.has_neighbor_across && sgn_face != (env->interface_manager->phi_at_point(xyz_face_neighbor) <= 0.0 ? -1 : +1));
          const bool neighbor_face_is_wall = fabs(xyz_face_neighbor[dir] - env->xyz_max[dir]) < 0.1*env->dxyz_smallest_quad[dir] || fabs(xyz_face_neighbor[dir] - env->xyz_min[dir]) < 0.1*env->dxyz_smallest_quad[dir];
          double offdiag_coeff;
          if(!across)
            offdiag_coeff = -mu_this_side*surface/distance_to_neighbor;
          else
          {
            // std::cerr << "This is bad: your grid is messed up here but, hey, I don't want to crash either..." << std::endl;
            char neighbor_orientation = -1;
            for (u_char dim = 0; dim < P4EST_DIM; ++dim)
              if(fabs(xyz_face_neighbor[dim] - xyz_face[dim]) > 0.1*env->dxyz_smallest_quad[dim])
                neighbor_orientation = 2*dim + (xyz_face_neighbor[dim] - xyz_face[dim] > 0.0 ? 1 : 0);
            P4EST_ASSERT(fabs(distance_to_neighbor - env->dxyz_smallest_quad[neighbor_orientation/2]) < 0.001*env->dxyz_smallest_quad[neighbor_orientation/2]);
            const FD_interface_neighbor& face_interface_neighbor = env->interface_manager->get_face_FD_interface_neighbor_for(f_idx, (*points)[m].n, dir, neighbor_orientation);
            const double& mu_across = (sgn_face > 0 ? env->mu_minus : env->mu_plus);
            const bool is_in_positive_domain = (sgn_face > 0);
            offdiag_coeff = surface*face_interface_neighbor.GFM_mu_jump(mu_this_side, mu_across)/distance_to_neighbor;
            rhs_p[f_idx] += surface*(neighbor_orientation%2 == 1 ? +1.0 : -1.0)*face_interface_neighbor.GFM_jump_terms_for_flux_component(mu_this_side, mu_across, neighbor_orientation, is_in_positive_domain, 0.0, 0.0, distance_to_neighbor);
          }
          if(!only_diags_are_modified[dir] && !matrix_is_ready[dir]){
            // we do not add the off-diagonal element to the matrix if the neighbor is a Dirichlet wall face
            // but we modify the rhs correspondingly, right away instead (see if statement here below)
            // --> ensures full symmetry of the matrix, one can use CG solver, safely!
            if(!neighbor_face_is_wall || env->bc_velocity[dir].wallType(xyz_face_neighbor) != DIRICHLET) {
              ierr = MatSetValue(matrix[dir], f_idx_g, env->faces_n->global_index((*points)[m].n, dir),  offdiag_coeff, ADD_VALUES); CHKERRXX(ierr); }
            ierr = MatSetValue(matrix[dir], f_idx_g, f_idx_g, -offdiag_coeff, ADD_VALUES); CHKERRXX(ierr);
          }
          if(neighbor_face_is_wall && env->bc_velocity[dir].wallType(xyz_face_neighbor) == DIRICHLET)
            rhs_p[f_idx] -= offdiag_coeff*(env->bc_velocity[dir].wallValue(xyz_face_neighbor) /* + interp_dxyz_hodge(xyz_face_neighbor) <- maybe this requires global interpolation /!\ */);
        }
      }
    }
  }

  ierr = VecRestoreArray(rhs[dir], &rhs_p); CHKERRXX(ierr);

  if(!matrix_is_ready[dir])
  {
    /* Assemble the matrix */
    ierr = MatAssemblyBegin(matrix[dir], MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
    ierr = MatAssemblyEnd  (matrix[dir], MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
  }

  /* take care of the nullspace if needed */
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &matrix_has_nullspace[dir], 1, MPI_INT, MPI_LAND, env->p4est_n->mpicomm); SC_CHECK_MPI(mpiret);
  if(matrix_has_nullspace[dir])
  {
    MatNullSpace matrix_nullspace;
    ierr = MatNullSpaceCreate(env->p4est_n->mpicomm, PETSC_TRUE, 0, NULL, &matrix_nullspace); CHKERRXX(ierr);
    ierr = MatSetNullSpace(matrix[dir], matrix_nullspace); CHKERRXX(ierr);
    ierr = MatNullSpaceRemove(matrix_nullspace, rhs[dir], NULL); CHKERRXX(ierr);
    ierr = MatNullSpaceDestroy(matrix_nullspace); CHKERRXX(ierr);
  }

  ierr = KSPSetOperators(ksp[dir], matrix[dir], matrix[dir], SAME_NONZERO_PATTERN); CHKERRXX(ierr);
  /* [Raphael Egan:] Starting from version 3.5, the last argument in KSPSetOperators became
   * irrelevant and is now simply disregarded in the above call. The matrices now keep track
   * of changes to their values and/or to their nonzero pattern by themselves. If no
   * modification was made to the matrix, the ksp environment can figure it out and knows
   * that the current preconditioner is still valid, thus it won't be recomputed.
   * If one desires to force reusing the current preconditioner EVEN IF a modification was
   * made to the matrix, one needs to call
   * ierr = KSPSetReusePreconditioner(ksp, PETSC_TRUE); CHKERRXX(ierr);
   * before the subsequent call to KSPSolve().
   * I have decided not to enforce that...
   */
  matrix_is_ready[dir]  = true;
  current_diag_minus[dir] = desired_diag_minus[dir];
  current_diag_plus[dir]  = desired_diag_plus[dir];
  P4EST_ASSERT(current_diags_are_as_desired(dir));
}

//sharp_derivative my_p4est_two_phase_flows_t::sharp_derivative_of_face_field(const p4est_locidx_t &face_idx, const uniform_face_ngbd *face_neighbors, const u_char &der, const u_char &dir,
//                                                                            const double *vn_dir_m_p, const double *vn_dir_p_p, const double *fine_jump_u_p, const double *fine_jump_mu_grad_v_p,
//                                                                            const p4est_quadrant_t &qm, const p4est_quadrant_t &qp)
//{
//  p4est_locidx_t fine_idx_of_other_face = -1;
//  const double* vn_dir_this_side_p  = (face_is_in_negative_domain ? vn_dir_m_p : vn_dir_p_p);
//  const double* vn_dir_other_side_p = (face_is_in_negative_domain ? vn_dir_p_p : vn_dir_m_p);
//  bool other_face_is_in_negative_domain;
//  if(face_neighbors->neighbor_face_idx[der] >= 0)
//    other_face_is_in_negative_domain = is_face_in_negative_domain(face_neighbors->neighbor_face_idx[der], dir, fine_phi_p, fine_idx_of_other_face);
//  else // neighbor "face" is wall
//  {
//    double xyz_wall[P4EST_DIM]; faces_n->xyz_fr_f(face_idx, dir, xyz_wall);
//    xyz_wall[der/2] = (der%2 == 1 ? xyz_max[der/2] : xyz_min[der/2]);
//    other_face_is_in_negative_domain = ((*interp_phi)(xyz_wall) <= 0.0);
//    if(bc_v[dir].wallType(xyz_wall) != DIRICHLET){
//      throw std::runtime_error("my_p4est_two_phase_flows_t::sharp_derivative_of_face_field: you're calculating the sharp derivative of a face-sampled field close to a non-Dirichlet wall here, it's not implemented yet...");
//      int mpiret = MPI_Abort(p4est_n->mpicomm, 475869); SC_CHECK_MPI(mpiret); }
//  }
//  sharp_derivative to_return;
//  if(face_is_in_negative_domain != other_face_is_in_negative_domain)
//  {
//    interface_data jump_info = interface_data_between_faces(fine_phi_p, fine_phi_xxyyzz_p, fine_jump_u_p, fine_jump_mu_grad_v_p, fine_idx_of_face, fine_idx_of_other_face, qm, qp, dir, der);
//    const double mu_tilde   = (face_is_in_negative_domain ? (1.0 - jump_info.theta)*mu_m + jump_info.theta*mu_p : (1.0 - jump_info.theta)*mu_p + jump_info.theta*mu_m);
//    const double mu_across  = (face_is_in_negative_domain ? mu_p                                                : mu_m);

//    if(face_neighbors->neighbor_face_idx[der] >= 0)
//    {
//      P4EST_ASSERT(fabs(vn_dir_this_side_p[face_idx]) < threshold_dbl_max && fabs(vn_dir_other_side_p[face_neighbors->neighbor_face_idx[der]]) < threshold_dbl_max);
//      to_return.derivative =
//          (der%2 == 1 ? +1.0 : -1.0)*mu_across*(vn_dir_other_side_p[face_neighbors->neighbor_face_idx[der]] + (face_is_in_negative_domain ? -1.0 : +1.0)*jump_info.jump_field - vn_dir_this_side_p[face_idx])/(mu_tilde*dxyz_min[der/2])
//          + (face_is_in_negative_domain ? -1.0 : +1.0)*(1.0 - jump_info.theta)*jump_info.jump_flux_component/mu_tilde;
//    }
//    to_return.theta = jump_info.theta;
//    to_return.xgfm  = true;
//  }
//  else
//  {
//    to_return.derivative = (der%2 == 1 ? +1.0 : -1.0)*(vn_dir_this_side_p[face_neighbors->neighbor_face_idx[der]] - vn_dir_this_side_p[face_idx])/dxyz_min[der/2];
//    to_return.theta      = 1.0;
//    to_return.xgfm       = false;
//  }
//  return to_return;
//}

//void my_p4est_two_phase_flows_t::get_velocity_from_other_domain_seen_from_face(neighbor_value &neighbor_velocity, const p4est_locidx_t &face_idx, const p4est_locidx_t &neighbor_face_idx,
//                                                                               const double *fine_phi_p, const double *fine_phi_xxyyzz_p,
//                                                                               const double *vn_dir_m_p, const double *vn_dir_p_p, const double *fine_jump_u_p, const double *fine_jump_mu_grad_v_p,
//                                                                               const u_char &der, const u_char &dir)
//{
//  p4est_locidx_t fine_idx_of_other_face = -1;
//  p4est_locidx_t fine_idx_of_face       = -1;
//  const bool face_is_in_negative_domain = is_face_in_negative_domain(face_idx, dir, fine_phi_p, fine_idx_of_face);
//  const double* vn_dir_this_side_p  = (face_is_in_negative_domain ? vn_dir_m_p : vn_dir_p_p );
//  const double* vn_dir_other_side_p = (face_is_in_negative_domain ? vn_dir_p_p : vn_dir_m_p);
//  const bool other_face_is_in_negative_domain = is_face_in_negative_domain(neighbor_face_idx, dir, fine_phi_p, fine_idx_of_other_face);
//  if(face_is_in_negative_domain != other_face_is_in_negative_domain)
//  {
//    p4est_quadrant_t qm, qp;
//    faces_n->find_quads_touching_face(face_idx, dir, qm, qp);
//    interface_data jump_info = interface_data_between_faces(fine_phi_p, fine_phi_xxyyzz_p, fine_jump_u_p, fine_jump_mu_grad_v_p, fine_idx_of_face, fine_idx_of_other_face, qm, qp, dir, der);
//    double mu_tilde     = (face_is_in_negative_domain ? (1.0 - jump_info.theta)*mu_m + jump_info.theta*mu_p : (1.0 - jump_info.theta)*mu_p + jump_info.theta*mu_m);
//    double mu_across    = (face_is_in_negative_domain ? mu_p                                                : mu_m);
//    double mu_this_side = (face_is_in_negative_domain ? mu_m                                                : mu_p);

//    P4EST_ASSERT((fabs(vn_dir_this_side_p[face_idx]) < threshold_dbl_max) && (fabs(vn_dir_other_side_p[neighbor_face_idx]) < threshold_dbl_max));
//    neighbor_velocity.value     = ((1.0 - jump_info.theta)*(mu_this_side*(vn_dir_this_side_p[face_idx] + (face_is_in_negative_domain ? +1.0 : -1.0)*jump_info.jump_field)) +
//                                   jump_info.theta*(mu_across*vn_dir_other_side_p[neighbor_face_idx]
//                                                    + (der%2 == 1 ? +1.0 : -1.0)*(face_is_in_negative_domain ? -1.0 : +1.0)*(1.0 - jump_info.theta)*dxyz_min[der/2]*jump_info.jump_flux_component))/mu_tilde;
//    neighbor_velocity.distance  = jump_info.theta*dxyz_min[der/2];
//  }
//  else
//  {
//    neighbor_velocity.value     = vn_dir_other_side_p[neighbor_face_idx];
//    neighbor_velocity.distance  = dxyz_min[der/2];
//  }
//  return;
//}

void my_p4est_two_phase_flows_t::compute_velocity_at_nodes()
{
  PetscErrorCode ierr;

  if(vnp1_nodes_minus == NULL){
    ierr = VecCreateGhostNodesBlock(p4est_n, nodes_n, P4EST_DIM, &vnp1_nodes_minus); CHKERRXX(ierr); }
  if(vnp1_nodes_plus == NULL){
    ierr = VecCreateGhostNodesBlock(p4est_n, nodes_n, P4EST_DIM, &vnp1_nodes_plus); CHKERRXX(ierr); }

  const double *vnp1_face_plus_p[P4EST_DIM], *vnp1_face_minus_p[P4EST_DIM];
  double *vnp1_nodes_plus_p, *vnp1_nodes_minus_p;
  ierr = VecGetArray(vnp1_nodes_minus,  &vnp1_nodes_minus_p); CHKERRXX(ierr);
  ierr = VecGetArray(vnp1_nodes_plus,   &vnp1_nodes_plus_p);  CHKERRXX(ierr);
  max_L2_norm_velocity_minus = max_L2_norm_velocity_plus = 0.0;

  for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecGetArrayRead(vnp1_face_minus[dir],  &vnp1_face_minus_p[dir]); CHKERRXX(ierr);
    ierr = VecGetArrayRead(vnp1_face_plus[dir],   &vnp1_face_plus_p[dir]);  CHKERRXX(ierr);
  }
  // loop through layer nodes first
  for(size_t i = 0; i < ngbd_n->get_layer_size(); ++i)
    interpolate_velocity_at_node(ngbd_n->get_layer_node(i), vnp1_nodes_minus_p, vnp1_nodes_plus_p, vnp1_face_minus_p, vnp1_face_plus_p);
  ierr = VecGhostUpdateBegin(vnp1_nodes_minus,  INSERT_VALUES, SCATTER_FORWARD);  CHKERRXX(ierr);
  ierr = VecGhostUpdateBegin(vnp1_nodes_plus,   INSERT_VALUES, SCATTER_FORWARD);  CHKERRXX(ierr);
  /* interpolate vnp1 from faces to nodes */
  for(size_t i = 0; i < ngbd_n->get_local_size(); ++i)
    interpolate_velocity_at_node(ngbd_n->get_local_node(i), vnp1_nodes_minus_p, vnp1_nodes_plus_p, vnp1_face_minus_p, vnp1_face_plus_p);

  int mpiret;
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &max_L2_norm_velocity_minus, 1, MPI_DOUBLE, MPI_MAX, p4est_n->mpicomm); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &max_L2_norm_velocity_plus,  1, MPI_DOUBLE, MPI_MAX, p4est_n->mpicomm); SC_CHECK_MPI(mpiret);
  ierr = VecGhostUpdateEnd(vnp1_nodes_minus, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(vnp1_nodes_plus, INSERT_VALUES, SCATTER_FORWARD);  CHKERRXX(ierr);
  ierr = VecRestoreArray(vnp1_nodes_minus,  &vnp1_nodes_minus_p);             CHKERRXX(ierr);
  ierr = VecRestoreArray(vnp1_nodes_plus,   &vnp1_nodes_plus_p);              CHKERRXX(ierr);
  for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecRestoreArrayRead(vnp1_face_minus[dir],  &vnp1_face_minus_p[dir]); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(vnp1_face_plus[dir],   &vnp1_face_plus_p[dir]);  CHKERRXX(ierr);
  }
  compute_vorticities();
}

//interface_data my_p4est_two_phase_flows_t::interface_data_between_faces(const double *fine_phi_p, const double *fine_phi_xxyyzz_p,
//                                                                        const double *fine_jump_u_p, const double *fine_jump_mu_grad_v_p,
//                                                                        const p4est_locidx_t &fine_idx_of_face, const p4est_locidx_t &fine_idx_of_other_face,
//                                                                        const p4est_quadrant_t &qm, const p4est_quadrant_t &qp,
//                                                                        const u_char &dir, const u_char der)
//{
//  P4EST_ASSERT(fine_idx_of_face >= 0 || fine_idx_of_other_face >= 0);
//  interface_data to_return;
//  to_return.fine_intermediary_idx = -1;
//  if(der == 2*dir)
//    get_fine_node_idx_of_logical_vertex(qm.p.piggy3.local_num, qm.p.piggy3.which_tree, DIM(0, 0, 0), to_return.fine_intermediary_idx, &qm);
//  else if(der == 2*dir + 1)
//    get_fine_node_idx_of_logical_vertex(qp.p.piggy3.local_num, qp.p.piggy3.which_tree, DIM(0, 0, 0), to_return.fine_intermediary_idx, &qp);
//  else
//  {
//    char search[P4EST_DIM] = {DIM(0,0,0)}; search[dir] = -1; search[der/2] = (der%2 == 0 ? -1 : 1);
//    get_fine_node_idx_of_logical_vertex(qp.p.piggy3.local_num, qp.p.piggy3.which_tree, DIM(search[0], search[1], search[2]), to_return.fine_intermediary_idx, &qp);
//  }
//  P4EST_ASSERT(to_return.fine_intermediary_idx != -1);

//  const double &phi_face                    = (fine_idx_of_face >= 0 ? fine_phi_p[fine_idx_of_face] : value_not_needed);
//  const double &phi_intermediary            = fine_phi_p[to_return.fine_intermediary_idx];
//  const double &phi_other_face              = (fine_idx_of_other_face >= 0  ? fine_phi_p[fine_idx_of_other_face] : value_not_needed);
//  const double dsub                         = dxyz_min[der/2]*0.5;
//  const bool no_past_mid_point              = (fine_idx_of_face >= 0 ? signs_of_phi_are_different(phi_face, phi_intermediary) : !signs_of_phi_are_different(phi_intermediary, phi_other_face));
//  P4EST_ASSERT((fine_idx_of_other_face >= 0 || no_past_mid_point) && (fine_idx_of_face >= 0 || !no_past_mid_point));
//  const double &phi_this_side               = (no_past_mid_point ? phi_face                         : phi_intermediary);
//  const double &phi_across                  = (no_past_mid_point ? phi_intermediary                 : phi_other_face);
//  const p4est_locidx_t& fine_idx_this_side  = (no_past_mid_point ? fine_idx_of_face                 : to_return.fine_intermediary_idx);
//  const p4est_locidx_t& fine_idx_across     = (no_past_mid_point ? to_return.fine_intermediary_idx  : fine_idx_of_other_face);
//  if(!((no_past_mid_point && !signs_of_phi_are_different(phi_intermediary, phi_other_face)) || (!no_past_mid_point && signs_of_phi_are_different(phi_intermediary, phi_other_face))))
//  {
//    std::cout << "Problem found on proc " << p4est_n->mpirank << " phi_face = " << phi_face << ", phi_intermediary = " << phi_intermediary << ", phi_other_face = " << phi_other_face << std::endl;
//    std::cout << "fine_idx_of_face = " << fine_idx_of_face << ", fine_idx_of_other_face= " << fine_idx_of_other_face << std::endl;
//  }
//  P4EST_ASSERT((no_past_mid_point && !signs_of_phi_are_different(phi_intermediary, phi_other_face)) || (!no_past_mid_point && signs_of_phi_are_different(phi_intermediary, phi_other_face)));
//  if(fine_phi_xxyyzz_p != NULL)
//    to_return.theta                         = fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(phi_this_side, phi_across, fine_phi_xxyyzz_p[P4EST_DIM*fine_idx_this_side + (der/2)], fine_phi_xxyyzz_p[P4EST_DIM*fine_idx_across + (der/2)], dsub);
//  else
//    to_return.theta                         = fraction_Interval_Covered_By_Irregular_Domain(phi_this_side, phi_across, dsub, dsub);
//  to_return.theta                           = (phi_this_side > 0.0 ? 1.0 - to_return.theta : to_return.theta);
//  to_return.theta                           = MAX(0.0, MIN(to_return.theta, 1.0));
//  to_return.jump_flux_component             = (1.0 - to_return.theta)*fine_jump_mu_grad_v_p[SQR_P4EST_DIM*fine_idx_this_side + P4EST_DIM*dir + (der/2)] + to_return.theta*fine_jump_mu_grad_v_p[SQR_P4EST_DIM*fine_idx_across + P4EST_DIM*dir + (der/2)];
//  to_return.jump_field                      = (fine_jump_u_p != NULL ? (1.0 - to_return.theta)*fine_jump_u_p[P4EST_DIM*fine_idx_this_side + dir] + to_return.theta*fine_jump_u_p[P4EST_DIM*fine_idx_across + dir] : 0.0);

//  to_return.theta                           = 0.5*(to_return.theta + (no_past_mid_point ? 0.0 : 1.0));
//  P4EST_ASSERT(to_return.theta >= 0.0 && to_return.theta <= 1.0);

//  return to_return;
//}


//interface_data my_p4est_two_phase_flows_t::interface_data_between_face_and_tranverse_wall(const double *fine_phi_p, const double *fine_phi_xxyyzz_p,
//                                                                                          const double *fine_jump_u_p, const double *fine_jump_mu_grad_v_p,
//                                                                                          const p4est_locidx_t &fine_idx_of_face, const p4est_locidx_t &fine_idx_of_wall_node,
//                                                                                          const u_char &dir, const u_char der)
//{
//  P4EST_ASSERT(fine_idx_of_face >= 0);
//  P4EST_ASSERT(fine_idx_of_wall_node >= 0);
//  interface_data to_return;
//  to_return.fine_intermediary_idx = -1;
//  P4EST_ASSERT(der/2 != dir);

//  const double phi_face                     = fine_phi_p[fine_idx_of_face];
//  const double phi_wall                     = fine_phi_p[fine_idx_of_wall_node];
//  const double dsub                         = dxyz_min[der/2]*0.5;
//  P4EST_ASSERT(signs_of_phi_are_different(phi_face, phi_wall));
//  if(fine_phi_xxyyzz_p != NULL)
//    to_return.theta                         = fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(phi_face, phi_wall, fine_phi_xxyyzz_p[P4EST_DIM*fine_idx_of_face + (der/2)], fine_phi_xxyyzz_p[P4EST_DIM*fine_idx_of_wall_node + (der/2)], dsub);
//  else
//    to_return.theta                         = fraction_Interval_Covered_By_Irregular_Domain(phi_face, phi_wall, dsub, dsub);
//  to_return.theta                           = (phi_face > 0.0 ? 1.0 - to_return.theta : to_return.theta);
//  to_return.theta                           = MAX(0.0, MIN(to_return.theta, 1.0));
//  to_return.jump_flux_component             = (1.0 - to_return.theta)*fine_jump_mu_grad_v_p[SQR_P4EST_DIM*fine_idx_of_face + P4EST_DIM*dir + (der/2)] + to_return.theta*fine_jump_mu_grad_v_p[SQR_P4EST_DIM*fine_idx_of_wall_node + P4EST_DIM*dir + (der/2)];
//  to_return.jump_field                      = (fine_jump_u_p != NULL ? (1.0 - to_return.theta)*fine_jump_u_p[P4EST_DIM*fine_idx_of_face + dir] + to_return.theta*fine_jump_u_p[P4EST_DIM*fine_idx_of_wall_node + dir] : 0.0);
//  // [IMPORTANT NOTE : ]
//  // in this case,
//  // to_return.theta is relative to the subresolved scale, i.e. relative to 0.5*dxyz_min[der/2]!

//  P4EST_ASSERT(to_return.theta >= 0.0 && to_return.theta <= 1.0);

//  return to_return;
//}

void my_p4est_two_phase_flows_t::interpolate_velocity_at_node(const p4est_locidx_t &node_idx, double *vnp1_nodes_minus_p, double *vnp1_nodes_plus_p,
                                                              const double *vnp1_face_minus_p[P4EST_DIM],  const double *vnp1_face_plus_p[P4EST_DIM])
{
  double xyz_node[P4EST_DIM]; node_xyz_fr_n(node_idx, p4est_n, nodes_n, xyz_node);
  p4est_indep_t *node = (p4est_indep_t*) sc_array_index(&nodes_n->indep_nodes, node_idx);

  bool velocity_component_minus_is_set[P4EST_DIM] = {DIM(false, false, false)};
  bool velocity_component_plus_is_set[P4EST_DIM]  = {DIM(false, false, false)};
  double magnitude_velocity_minus = 0.0;
  double magnitude_velocity_plus  = 0.0;

  if(bc_velocity != NULL && is_node_Wall(p4est_n, node))
    for (u_char dir = 0; dir < P4EST_DIM; ++dir)
      if(bc_velocity[dir].wallType(xyz_node) == DIRICHLET)
      {
        vnp1_nodes_minus_p[P4EST_DIM*node_idx + dir] = bc_velocity[dir].wallValue(xyz_node); magnitude_velocity_minus += SQR(vnp1_nodes_minus_p[P4EST_DIM*node_idx + dir]);
        velocity_component_minus_is_set[dir] = true;
        vnp1_nodes_plus_p[P4EST_DIM*node_idx + dir] = bc_velocity[dir].wallValue(xyz_node); magnitude_velocity_plus += SQR(vnp1_nodes_plus_p[P4EST_DIM*node_idx + dir]);
        velocity_component_plus_is_set[dir] = true;
      }
  if(ANDD(velocity_component_minus_is_set[0], velocity_component_minus_is_set[1], velocity_component_minus_is_set[2]) &&
     ANDD(velocity_component_plus_is_set[0], velocity_component_plus_is_set[1], velocity_component_plus_is_set[2]))
    return;

  /* gather the neighborhood */
  set_of_neighboring_quadrants neighbor_cells;
  const p4est_qcoord_t logical_size_of_smallest_first_degree_neighbor = ngbd_c->gather_neighbor_cells_of_node(node_idx, nodes_n, neighbor_cells, true);
  const double lsqr_scaling = ((double) logical_size_of_smallest_first_degree_neighbor/(double) P4EST_ROOT_LEN)*0.5*MIN(DIM(tree_dimension[0], tree_dimension[1], tree_dimension[2]));
  set<p4est_locidx_t> set_of_faces[P4EST_DIM];
  add_all_faces_to_sets_and_clear_set_of_quad(faces_n, set_of_faces, neighbor_cells);

  for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
    for (char sign = -1; sign <= 1; sign += 2) {
      const double *vnp1_face_p = (sign < 0 ? vnp1_face_minus_p[dir] : vnp1_face_plus_p[dir]);
      double *result_field = (sign < 0 ? vnp1_nodes_minus_p : vnp1_nodes_plus_p);
      double &mag_v = (sign < 0 ? magnitude_velocity_minus : magnitude_velocity_plus);

      if((sign < 0 ? velocity_component_minus_is_set[dir] : velocity_component_plus_is_set[dir]))
        continue;

      matrix_t        A_lsqr;
      vector<double>  rhs_lsqr;
      std::set<int64_t> nb[P4EST_DIM];
      bool is_wall[P4EST_FACES];
      char neumann_wall[P4EST_DIM] = {DIM(0, 0, 0)};
      u_char nb_neumann_walls = 0;
      if(is_node_Wall(p4est_n, node, is_wall) && bc_velocity[dir].wallType(xyz_node) == NEUMANN)
        for (u_char dd = 0; dd < P4EST_DIM; ++dd)
        {
          neumann_wall[dir] = (is_wall[2*dd] ? -1 : (is_wall[2*dd + 1] ? +1 : 0));
          nb_neumann_walls += abs(neumann_wall[dir]);
        }
      const double min_w = 1e-6;
      const double inv_max_w = 1e-6;

      int row_idx = 0;
      int col_idx;
      for (std::set<p4est_locidx_t>::const_iterator it = set_of_faces[dir].begin(); it != set_of_faces[dir].end(); ++it)
      {
        P4EST_ASSERT(*it >= 0 && *it < faces_n->num_local[dir] + faces_n->num_ghost[dir]);
        row_idx += (fabs(vnp1_face_p[*it]) < threshold_dbl_max ? 1 : 0);
        P4EST_ASSERT(!ISNAN(vnp1_face_p[*it]));
      }
      if(row_idx == 0)
      {
        result_field[P4EST_DIM*node_idx + dir] = NAN; // absurd value for undefined interpolated field (e.g. vn_minus far in negative domain --> no well-defined neighbor to be found so move on)
        mag_v = NAN;
        continue;
      }

      A_lsqr.resize(row_idx, (1 + P4EST_DIM + P4EST_DIM*(P4EST_DIM + 1)/2 - nb_neumann_walls));
      rhs_lsqr.resize(row_idx);
      row_idx = 0;
      for (std::set<p4est_locidx_t>::const_iterator it = set_of_faces[dir].begin(); it != set_of_faces[dir].end(); ++it) {
        const p4est_locidx_t& neighbor_face_idx = *it;
        if(fabs(vnp1_face_p[*it]) > threshold_dbl_max)
          continue;

        double xyz_t[P4EST_DIM];
        int64_t logical_qcoord_diff[P4EST_DIM];
        faces_n->rel_qxyz_face_fr_node(neighbor_face_idx, dir, xyz_t, xyz_node, node, logical_qcoord_diff);

        for(u_char i = 0; i < P4EST_DIM; ++i)
        {
          xyz_t[i] /= lsqr_scaling;
          nb[i].insert(logical_qcoord_diff[i]);
        }

        const double w = MAX(min_w, 1./MAX(inv_max_w, sqrt(SUMD(SQR(xyz_t[0]), SQR(xyz_t[1]), SQR(xyz_t[2])))));

        rhs_lsqr[row_idx] = 0.0;
        col_idx = 0;
        A_lsqr.set_value(row_idx, col_idx++, w);
        for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
          if(neumann_wall[dim] == 0)
            A_lsqr.set_value(row_idx, col_idx++, xyz_t[dim]*w);
          else
          {
            P4EST_ASSERT(abs(neumann_wall[dim]) == 1);
            rhs_lsqr[row_idx] -= neumann_wall[dim]*bc_velocity[dim].wallValue(xyz_node)*xyz_t[dim]*lsqr_scaling; // multiplication by w at the end
          }
        }
        for (u_char dim = 0; dim < P4EST_DIM; ++dim)
          for (u_char cross_dim = dim; cross_dim < P4EST_DIM; ++cross_dim)
            A_lsqr.set_value(row_idx, col_idx++, xyz_t[dim]*xyz_t[cross_dim] *w);
        P4EST_ASSERT(col_idx == 1 + P4EST_DIM + P4EST_DIM*(P4EST_DIM + 1)/2 - nb_neumann_walls);
        rhs_lsqr[row_idx] += vnp1_face_p[neighbor_face_idx];
        rhs_lsqr[row_idx] *= w;
        row_idx++;
      }
      P4EST_ASSERT(row_idx == A_lsqr.num_rows());
      A_lsqr.scale_by_maxabs(rhs_lsqr);

      result_field[P4EST_DIM*node_idx + dir] = solve_lsqr_system(A_lsqr, rhs_lsqr, DIM(nb[0].size(), nb[1].size(), nb[2].size()), 2, nb_neumann_walls);

      if (!ISNAN(mag_v))
        mag_v += SQR(result_field[P4EST_DIM*node_idx + dir]);
    }
  }
  if (!ISNAN(magnitude_velocity_minus) && interface_manager->phi_at_point(xyz_node) < 2.0*cfl*MAX(DIM(dxyz_smallest_quad[0], dxyz_smallest_quad[1], dxyz_smallest_quad[2])))
    max_L2_norm_velocity_minus = MAX(max_L2_norm_velocity_minus, sqrt(magnitude_velocity_minus));
  if (!ISNAN(magnitude_velocity_plus) && interface_manager->phi_at_point(xyz_node) > -2.0*cfl*MAX(DIM(dxyz_smallest_quad[0], dxyz_smallest_quad[1], dxyz_smallest_quad[2])))
    max_L2_norm_velocity_plus = MAX(max_L2_norm_velocity_plus, sqrt(magnitude_velocity_plus));
  return;
}

void my_p4est_two_phase_flows_t::compute_vorticities()
{
  PetscErrorCode ierr;

  if(vorticity_magnitude_minus == NULL){
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &vorticity_magnitude_minus); CHKERRXX(ierr);
  }
  if(vorticity_magnitude_plus == NULL){
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &vorticity_magnitude_plus); CHKERRXX(ierr);
  }

  const double *vnp1_nodes_minus_p, *vnp1_nodes_plus_p;
  double *vorticity_magnitude_minus_p, *vorticity_magnitude_plus_p;
  ierr = VecGetArrayRead(vnp1_nodes_minus,  &vnp1_nodes_minus_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(vnp1_nodes_plus,   &vnp1_nodes_plus_p);  CHKERRXX(ierr);
  ierr = VecGetArray(vorticity_magnitude_minus, &vorticity_magnitude_minus_p);  CHKERRXX(ierr);
  ierr = VecGetArray(vorticity_magnitude_plus,  &vorticity_magnitude_plus_p);   CHKERRXX(ierr);
  double velocity_gradient_minus[SQR_P4EST_DIM], velocity_gradient_plus[SQR_P4EST_DIM];
  const double *inputs[2] = {vnp1_nodes_minus_p, vnp1_nodes_plus_p};
  double *outputs[2] = {velocity_gradient_minus, velocity_gradient_plus};

  quad_neighbor_nodes_of_node_t qnnn_buf;
  const bool neighbors_are_initialized = ngbd_n->neighbors_are_initialized();
  const quad_neighbor_nodes_of_node_t *qnnn_p = (neighbors_are_initialized ? NULL : &qnnn_buf);

  for(size_t i = 0; i < ngbd_n->get_layer_size(); ++i)
  {
    const p4est_locidx_t node_idx = ngbd_n->get_layer_node(i);
    if(neighbors_are_initialized)
      ngbd_n->get_neighbors(node_idx, qnnn_p);
    else
      ngbd_n->get_neighbors(node_idx, qnnn_buf);

    qnnn_p->gradient_all_components(inputs, outputs, 2, P4EST_DIM);
    for (char sgn = -1; sgn <= 1; sgn += 2) {
      const double* velocity_gradient = (sgn < 0 ? velocity_gradient_minus      : velocity_gradient_plus);
      double *vorticity_magnitude_p   = (sgn < 0 ? vorticity_magnitude_minus_p  : vorticity_magnitude_plus_p);
#ifdef P4_TO_P8
      vorticity_magnitude_p[node_idx] = 0.0;
      vorticity_magnitude_p[node_idx] = sqrt(SQR(velocity_gradient[P4EST_DIM*dir::y + dir::x] - velocity_gradient[P4EST_DIM*dir::x + dir::y])
          + SQR(velocity_gradient[P4EST_DIM*dir::x + dir::z] - velocity_gradient[P4EST_DIM*dir::z + dir::x])
          + SQR(velocity_gradient[P4EST_DIM*dir::z + dir::y] - velocity_gradient[P4EST_DIM*dir::y + dir::z]))
#else
      vorticity_magnitude_p[node_idx] = velocity_gradient[P4EST_DIM*dir::y + dir::x] - velocity_gradient[P4EST_DIM*dir::x + dir::y];
#endif
    }
  }
  ierr = VecGhostUpdateBegin(vorticity_magnitude_minus, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateBegin(vorticity_magnitude_plus,  INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for(size_t i = 0; i < ngbd_n->get_local_size(); ++i)
  {
    const p4est_locidx_t node_idx = ngbd_n->get_local_node(i);
    if(neighbors_are_initialized)
      ngbd_n->get_neighbors(node_idx, qnnn_p);
    else
      ngbd_n->get_neighbors(node_idx, qnnn_buf);

    qnnn_p->gradient_all_components(inputs, outputs, 2, P4EST_DIM);
    for (char sgn = -1; sgn <= 1; sgn += 2) {
      const double* velocity_gradient = (sgn < 0 ? velocity_gradient_minus      : velocity_gradient_plus);
      double *vorticity_magnitude_p   = (sgn < 0 ? vorticity_magnitude_minus_p  : vorticity_magnitude_plus_p);
#ifdef P4_TO_P8
      vorticity_magnitude_p[node_idx] = 0.0;
      vorticity_magnitude_p[node_idx] = sqrt(SQR(velocity_gradient[P4EST_DIM*dir::y + dir::x] - velocity_gradient[P4EST_DIM*dir::x + dir::y])
          + SQR(velocity_gradient[P4EST_DIM*dir::x + dir::z] - velocity_gradient[P4EST_DIM*dir::z + dir::x])
          + SQR(velocity_gradient[P4EST_DIM*dir::z + dir::y] - velocity_gradient[P4EST_DIM*dir::y + dir::z]))
#else
      vorticity_magnitude_p[node_idx] = velocity_gradient[P4EST_DIM*dir::y + dir::x] - velocity_gradient[P4EST_DIM*dir::x + dir::y];
#endif
    }
  }
  ierr = VecGhostUpdateEnd(vorticity_magnitude_minus, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(vorticity_magnitude_plus,  INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecRestoreArray(vorticity_magnitude_plus,  &vorticity_magnitude_plus_p);   CHKERRXX(ierr);
  ierr = VecRestoreArray(vorticity_magnitude_minus, &vorticity_magnitude_minus_p);  CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(vnp1_nodes_plus,   &vnp1_nodes_plus_p);  CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(vnp1_nodes_minus,  &vnp1_nodes_minus_p); CHKERRXX(ierr);
}

//void my_p4est_two_phase_flows_t::extrapolate_velocities_across_interface_in_finest_computational_cells_Aslam_PDE(const extrapolation_technique& extrapolation_method, const unsigned int& n_iterations, const u_char& extrapolation_degree)
//{
//  P4EST_ASSERT(n_iterations > 0);
//  P4EST_ASSERT(extrapolation_degree <= 1);
//  PetscErrorCode ierr;
//  my_p4est_interpolation_nodes_t interp_normal(fine_ngbd_n);
//  interp_normal.set_input(fine_normal, linear, P4EST_DIM);
//  // optional normal derivatives of components (needed only if extrapolation_degree > 0)--> initialized to NULL
//  Vec normal_derivative_of_vnp1_m[P4EST_DIM]                  = {DIM(NULL, NULL, NULL)};
//  Vec normal_derivative_of_vnp1_p[P4EST_DIM]                  = {DIM(NULL, NULL, NULL)};
//  double *normal_derivative_of_vnp1_m_p[P4EST_DIM]            = {DIM(NULL, NULL, NULL)};
//  double *normal_derivative_of_vnp1_p_p[P4EST_DIM]            = {DIM(NULL, NULL, NULL)};
//  const double *normal_derivative_of_vnp1_m_read_p[P4EST_DIM] = {DIM(NULL, NULL, NULL)};
//  const double *normal_derivative_of_vnp1_p_read_p[P4EST_DIM] = {DIM(NULL, NULL, NULL)};
//  // optional extra data (initialize to NULL as well)
//  const double *fine_phi_xxyyzz_p   = NULL; // for second-order interface localization
//  const double *fine_jump_u_p       = NULL; // required only if playing with mass flux or for testing only

//  double *vnp1_m_p[P4EST_DIM], *vnp1_p_p[P4EST_DIM];
//  const double *fine_phi_p, *fine_jump_mu_grad_v_p;
//  // get pointers, initialize normal derivatives of the P4EST_DIM face-sampled fields, etc.
//  for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
//    ierr = VecGetArray(vnp1_m[dir], &vnp1_m_p[dir]);                                            CHKERRXX(ierr);
//    ierr = VecGetArray(vnp1_p[dir], &vnp1_p_p[dir]);                                            CHKERRXX(ierr);

//    if(extrapolation_degree > 0)
//    {
//    ierr = VecCreateGhostFaces(p4est_n, faces_n, &normal_derivative_of_vnp1_m[dir], dir);       CHKERRXX(ierr);
//    ierr = VecCreateGhostFaces(p4est_n, faces_n, &normal_derivative_of_vnp1_p[dir], dir);       CHKERRXX(ierr);
//    ierr = VecGetArray(normal_derivative_of_vnp1_m[dir], &normal_derivative_of_vnp1_m_p[dir]);  CHKERRXX(ierr);
//    ierr = VecGetArray(normal_derivative_of_vnp1_p[dir], &normal_derivative_of_vnp1_p_p[dir]);  CHKERRXX(ierr);
//    }
//  }
//  // get data pointers
//  ierr = VecGetArrayRead(fine_phi, &fine_phi_p);                                                CHKERRXX(ierr);
//  if(fine_phi_xxyyzz != NULL){
//    ierr = VecGetArrayRead(fine_phi_xxyyzz, &fine_phi_xxyyzz_p);                                CHKERRXX(ierr); }
//  ierr = VecGetArrayRead(fine_jump_mu_grad_v, &fine_jump_mu_grad_v_p);                          CHKERRXX(ierr);
//  if(fine_jump_u != NULL){
//    ierr = VecGetArrayRead(fine_jump_u, &fine_jump_u_p);                                        CHKERRXX(ierr); }
//  // INITIALIZE extrapolation
//  // local layer faces first
//  for (u_char dir = 0; dir < P4EST_DIM; ++dir)
//  {
//    for (size_t k = 0; k < faces_n->get_layer_size(dir); ++k)
//      initialize_face_extrapolation(faces_n->get_layer_face(dir, k), dir, interp_normal, fine_phi_p, fine_phi_xxyyzz_p, extrapolation_degree,
//                                    fine_jump_u_p, fine_jump_mu_grad_v_p, vnp1_m_p, vnp1_p_p, normal_derivative_of_vnp1_m_p, normal_derivative_of_vnp1_p_p);
//    // start updates
//    if(extrapolation_degree > 0)
//    {
//      ierr = VecGhostUpdateBegin(normal_derivative_of_vnp1_m[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//      ierr = VecGhostUpdateBegin(normal_derivative_of_vnp1_p[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//    }
//    ierr = VecGhostUpdateBegin(vnp1_m[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//    ierr = VecGhostUpdateBegin(vnp1_p[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  }
//  // local inner faces
//  for (u_char dir = 0; dir < P4EST_DIM; ++dir)
//  {
//    for (size_t k = 0; k < faces_n->get_local_size(dir); ++k)
//      initialize_face_extrapolation(faces_n->get_local_face(dir, k), dir, interp_normal, fine_phi_p, fine_phi_xxyyzz_p, extrapolation_degree,
//                                    fine_jump_u_p, fine_jump_mu_grad_v_p, vnp1_m_p, vnp1_p_p, normal_derivative_of_vnp1_m_p, normal_derivative_of_vnp1_p_p);
//    // complete updates
//    if(extrapolation_degree > 0)
//    {
//      ierr = VecGhostUpdateEnd(normal_derivative_of_vnp1_m[dir], INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);
//      ierr = VecGhostUpdateEnd(normal_derivative_of_vnp1_p[dir], INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);
//    }
//    ierr = VecGhostUpdateEnd(vnp1_m[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//    ierr = VecGhostUpdateEnd(vnp1_p[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  }
//  if(extrapolation_degree > 0)
//  {
//    /* EXTRAPOLATE normal derivatives of velocity components */
//    for (unsigned int iter = 0; iter < n_iterations; ++iter) {
//      // local layer faces first
//      for (u_char dir = 0; dir < P4EST_DIM; ++dir)
//      {
//        for (size_t k = 0; k < faces_n->get_layer_size(dir); ++k)
//        {
//          p4est_locidx_t local_face_idx = faces_n->get_layer_face(dir, k);
//          switch (extrapolation_method) {
//          case PSEUDO_TIME:
//            extrapolate_normal_derivatives_of_face_velocity_local_pseudo_time(local_face_idx, dir, interp_normal, fine_phi_p,
//                                                                              normal_derivative_of_vnp1_m_p, normal_derivative_of_vnp1_p_p);
//            break;
//          case EXPLICIT_ITERATIVE:
//            extrapolate_normal_derivatives_of_face_velocity_local_explicit_iterative(local_face_idx, dir, interp_normal, fine_phi_p,
//                                                                                     normal_derivative_of_vnp1_m_p, normal_derivative_of_vnp1_p_p);
//            break;
//          default:
//            throw std::invalid_argument("my_p4est_two_phase_flows_t::extrapolate_velocities_across_interface_in_finest_computational_cells_Aslam_PDE(): unknown extrapolation_method");
//            break;
//          }
//        }
//        // start updates
//        ierr = VecGhostUpdateBegin(normal_derivative_of_vnp1_p[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//        ierr = VecGhostUpdateBegin(normal_derivative_of_vnp1_m[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//      }
//      // local inner faces
//      for (u_char dir = 0; dir < P4EST_DIM; ++dir)
//      {
//        for (size_t k = 0; k < faces_n->get_local_size(dir); ++k)
//        {
//          p4est_locidx_t local_face_idx = faces_n->get_local_face(dir, k);
//          switch (extrapolation_method) {
//          case PSEUDO_TIME:
//            extrapolate_normal_derivatives_of_face_velocity_local_pseudo_time(local_face_idx, dir, interp_normal, fine_phi_p,
//                                                                              normal_derivative_of_vnp1_m_p, normal_derivative_of_vnp1_p_p);
//            break;
//          case EXPLICIT_ITERATIVE:
//            extrapolate_normal_derivatives_of_face_velocity_local_explicit_iterative(local_face_idx, dir, interp_normal, fine_phi_p,
//                                                                                     normal_derivative_of_vnp1_m_p, normal_derivative_of_vnp1_p_p);
//            break;
//          default:
//            throw std::invalid_argument("my_p4est_two_phase_flows_t::extrapolate_velocities_across_interface_in_finest_computational_cells_Aslam_PDE(): unknown extrapolation_method");
//            break;
//          }
//        }
//        // complete updates
//        ierr = VecGhostUpdateEnd(normal_derivative_of_vnp1_m[dir], INSERT_VALUES, SCATTER_FORWARD);       CHKERRXX(ierr);
//        ierr = VecGhostUpdateEnd(normal_derivative_of_vnp1_p[dir], INSERT_VALUES, SCATTER_FORWARD);       CHKERRXX(ierr);
//      }
//    }
//    for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
//      ierr = VecRestoreArray(normal_derivative_of_vnp1_m[dir], &normal_derivative_of_vnp1_m_p[dir]);      CHKERRXX(ierr);
//      ierr = VecRestoreArray(normal_derivative_of_vnp1_p[dir], &normal_derivative_of_vnp1_p_p[dir]);      CHKERRXX(ierr);
//      ierr = VecGetArrayRead(normal_derivative_of_vnp1_m[dir], &normal_derivative_of_vnp1_m_read_p[dir]); CHKERRXX(ierr);
//      ierr = VecGetArrayRead(normal_derivative_of_vnp1_p[dir], &normal_derivative_of_vnp1_p_read_p[dir]); CHKERRXX(ierr);
//    }
//  }
//  /* EXTRAPOLATE velocity components */
//  for (unsigned int iter = 0; iter < n_iterations; ++iter) {
//    // local layer faces first
//    for (u_char dir = 0; dir < P4EST_DIM; ++dir)
//    {
//      for (size_t k = 0; k < faces_n->get_layer_size(dir); ++k) {
//        p4est_locidx_t local_face_idx = faces_n->get_layer_face(dir, k);
//        switch (extrapolation_method) {
//        case PSEUDO_TIME:
//          solve_velocity_extrapolation_local_pseudo_time(local_face_idx, dir, interp_normal, fine_phi_p, fine_phi_xxyyzz_p, extrapolation_degree,
//                                                         fine_jump_u_p, fine_jump_mu_grad_v_p, normal_derivative_of_vnp1_m_read_p, normal_derivative_of_vnp1_p_read_p, vnp1_m_p, vnp1_p_p);
//          break;
//        case EXPLICIT_ITERATIVE:
//          solve_velocity_extrapolation_local_explicit_iterative(local_face_idx, dir, interp_normal, fine_phi_p, fine_phi_xxyyzz_p, extrapolation_degree,
//                                                                fine_jump_u_p, fine_jump_mu_grad_v_p, normal_derivative_of_vnp1_m_read_p, normal_derivative_of_vnp1_p_read_p, vnp1_m_p, vnp1_p_p);
//          break;
//        default:
//          throw std::invalid_argument("my_p4est_two_phase_flows_t::extrapolate_velocities_across_interface_in_finest_computational_cells_Aslam_PDE(): unknown extrapolation_method");
//          break;
//        }
//      }
//      // start updates
//      ierr = VecGhostUpdateBegin(vnp1_m[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//      ierr = VecGhostUpdateBegin(vnp1_p[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//    }
//    // local inner faces
//    for (u_char dir = 0; dir < P4EST_DIM; ++dir)
//    {
//      for (size_t k = 0; k < faces_n->get_local_size(dir); ++k) {
//        p4est_locidx_t local_face_idx = faces_n->get_local_face(dir, k);
//        switch (extrapolation_method) {
//        case PSEUDO_TIME:
//          solve_velocity_extrapolation_local_pseudo_time(local_face_idx, dir, interp_normal, fine_phi_p, fine_phi_xxyyzz_p, extrapolation_degree,
//                                                         fine_jump_u_p, fine_jump_mu_grad_v_p, normal_derivative_of_vnp1_m_read_p, normal_derivative_of_vnp1_p_read_p, vnp1_m_p, vnp1_p_p);
//          break;
//        case EXPLICIT_ITERATIVE:
//          solve_velocity_extrapolation_local_explicit_iterative(local_face_idx, dir, interp_normal, fine_phi_p, fine_phi_xxyyzz_p, extrapolation_degree,
//                                                                fine_jump_u_p, fine_jump_mu_grad_v_p, normal_derivative_of_vnp1_m_read_p, normal_derivative_of_vnp1_p_read_p, vnp1_m_p, vnp1_p_p);
//          break;
//        default:
//          throw std::invalid_argument("my_p4est_two_phase_flows_t::extrapolate_velocities_across_interface_in_finest_computational_cells_Aslam_PDE(): unknown extrapolation_method");
//          break;
//        }
//      }
//      // complete updates
//      ierr = VecGhostUpdateEnd(vnp1_m[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//      ierr = VecGhostUpdateEnd(vnp1_p[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//    }
//  }

//  // restore data pointers
//  ierr = VecRestoreArrayRead(fine_phi, &fine_phi_p);                                                          CHKERRXX(ierr);
//  if(fine_phi_xxyyzz_p != NULL){
//    ierr = VecRestoreArrayRead(fine_phi_xxyyzz, &fine_phi_xxyyzz_p);                                          CHKERRXX(ierr); }
//  ierr = VecRestoreArrayRead(fine_jump_mu_grad_v, &fine_jump_mu_grad_v_p);                                    CHKERRXX(ierr);
//  if(fine_jump_u != NULL){
//    ierr = VecRestoreArrayRead(fine_jump_u, &fine_jump_u_p);                                                  CHKERRXX(ierr); }
//  for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
//    ierr = VecRestoreArray(vnp1_m[dir], &vnp1_m_p[dir]);                                                      CHKERRXX(ierr);
//    ierr = VecRestoreArray(vnp1_p[dir], &vnp1_p_p[dir]);                                                      CHKERRXX(ierr);
//    if(extrapolation_degree > 0)
//    {
//      ierr = VecRestoreArrayRead(normal_derivative_of_vnp1_m[dir], &normal_derivative_of_vnp1_m_read_p[dir]); CHKERRXX(ierr);
//      ierr = VecRestoreArrayRead(normal_derivative_of_vnp1_p[dir], &normal_derivative_of_vnp1_p_read_p[dir]); CHKERRXX(ierr);
//      ierr = delete_and_nullify_vector(normal_derivative_of_vnp1_m[dir]);                                     CHKERRXX(ierr);
//      ierr = delete_and_nullify_vector(normal_derivative_of_vnp1_p[dir]);                                     CHKERRXX(ierr);
//    }
//  }
//}

//void my_p4est_two_phase_flows_t::initialize_face_extrapolation(const p4est_locidx_t &local_face_idx, const u_char &dir, const my_p4est_interpolation_nodes_t &interp_normal,
//                                                               const double *fine_phi_p, const double *fine_phi_xxyyzz_p, const u_char &extrapolation_degree,
//                                                               const double *fine_jump_u_p, const double *fine_jump_mu_grad_v_p,
//                                                               double *vnp1_m_p[P4EST_DIM], double *vnp1_p_p[P4EST_DIM], double *normal_derivative_of_vnp1_m_p[P4EST_DIM], double *normal_derivative_of_vnp1_p_p[P4EST_DIM])
//{
//  p4est_locidx_t fine_idx_of_face = -1;
//  const bool face_is_in_negative_domain = is_face_in_negative_domain(local_face_idx, dir, fine_phi_p, fine_idx_of_face);
//  double *normal_derivative_of_vnp1_dir_this_side_p   = (extrapolation_degree > 0 ? (face_is_in_negative_domain ? normal_derivative_of_vnp1_m_p[dir] : normal_derivative_of_vnp1_p_p[dir]) : NULL);
//  double *normal_derivative_of_vnp1_dir_other_side_p  = (extrapolation_degree > 0 ? (face_is_in_negative_domain ? normal_derivative_of_vnp1_p_p[dir] : normal_derivative_of_vnp1_m_p[dir]) : NULL);
//  const double *vnp1_dir_this_side_p                  = (face_is_in_negative_domain ? vnp1_m_p[dir] : vnp1_p_p[dir]);
//  double *vnp1_dir_other_side_p                       = (face_is_in_negative_domain ? vnp1_p_p[dir] : vnp1_m_p[dir]);
//  double local_normal[P4EST_DIM];
//  if(extrapolation_degree > 0)
//  {
//    double xyz_face[P4EST_DIM]; faces_n->xyz_fr_f(local_face_idx, dir, xyz_face);
//    interp_normal(xyz_face, local_normal);
//    const double mag = sqrt(SUMD(SQR(local_normal[0]), SQR(local_normal[1]), SQR(local_normal[2])));
//    for (u_char dir = 0; dir < P4EST_DIM; ++dir)
//    {
//      if(mag > EPS)
//        local_normal[dir] /= mag;
//      else
//        local_normal[dir] = 0.0;
//    }
//  }
//  const uniform_face_ngbd* face_ngbd;
//  if(faces_n->found_uniform_face_neighborhood(local_face_idx, dir, face_ngbd))
//  {
//    double n_dot_grad_u_dir = 0.0;
//    if(extrapolation_degree > 0)
//    {
//      p4est_quadrant_t qm, qp;
//      faces_n->find_quads_touching_face(local_face_idx, dir, qm, qp);
//      for (u_char der = 0; der < P4EST_DIM; ++der) {
//        sharp_derivative sd_p = sharp_derivative_of_face_field(local_face_idx, face_is_in_negative_domain, fine_idx_of_face, face_ngbd,
//                                                               fine_phi_p, fine_phi_xxyyzz_p, 2*der + 1, dir,
//                                                               vnp1_m_p[dir], vnp1_p_p[dir], fine_jump_u_p, fine_jump_mu_grad_v_p, qm, qp);
//        sharp_derivative sd_m = sharp_derivative_of_face_field(local_face_idx, face_is_in_negative_domain, fine_idx_of_face, face_ngbd,
//                                                               fine_phi_p, fine_phi_xxyyzz_p, 2*der, dir,
//                                                               vnp1_m_p[dir], vnp1_p_p[dir], fine_jump_u_p, fine_jump_mu_grad_v_p, qm, qp);
//        n_dot_grad_u_dir += (face_is_in_negative_domain ? +1.0 : -1.0)*local_normal[der]*(sd_p.theta*sd_p.derivative + sd_m.theta*sd_m.derivative)/(sd_m.theta + sd_p.theta);
//      }
//    }

//    if(extrapolation_degree > 0)
//    {
//      normal_derivative_of_vnp1_dir_this_side_p[local_face_idx]   = n_dot_grad_u_dir;
//      normal_derivative_of_vnp1_dir_other_side_p[local_face_idx]  = 0.0; // to be calculated later on in actual extrapolation
//    }
//    P4EST_ASSERT(fabs(vnp1_dir_other_side_p[local_face_idx]) > threshold_dbl_max);
//    vnp1_dir_other_side_p[local_face_idx] = vnp1_dir_this_side_p[local_face_idx] /* +/- jump */;
//  }
//  else
//  {
//    normal_derivative_of_vnp1_dir_other_side_p[local_face_idx] = 0.0; // to be calculated later on in extrapolation (if possible)
//    P4EST_ASSERT(fabs(vnp1_dir_other_side_p[local_face_idx]) > threshold_dbl_max);
//    vnp1_dir_other_side_p[local_face_idx] = vnp1_dir_this_side_p[local_face_idx] /* +/- jump */;
//  }
//}

//void my_p4est_two_phase_flows_t::extrapolate_normal_derivatives_of_face_velocity_local_explicit_iterative(const p4est_locidx_t &local_face_idx, const u_char &dir, const my_p4est_interpolation_nodes_t &interp_normal, const double *fine_phi_p,
//                                                                                                          double *normal_derivative_of_vnp1_m_p[P4EST_DIM], double *normal_derivative_of_vnp1_p_p[P4EST_DIM])
//{
//  double xyz_face[P4EST_DIM], local_normal[P4EST_DIM];
//  faces_n->xyz_fr_f(local_face_idx, dir, xyz_face);
//  interp_normal(xyz_face, local_normal);
//  const double mag_normal = sqrt(SUMD(SQR(local_normal[0]), SQR(local_normal[1]), SQR(local_normal[2])));
//  for (u_char dir = 0; dir < P4EST_DIM; ++dir)
//  {
//    if(mag_normal > EPS)
//      local_normal[dir] /= mag_normal;
//    else
//      local_normal[dir] = 0.0;
//  }
//  double lhs_normal_derivative_field    = 0.0;
//  double rhs_normal_derivative_field    = 0.0;
//  // if the face is in negative domain, we extend the + field and the normal is reverted
//  const bool face_is_in_negative_domain = is_face_in_negative_domain(local_face_idx, dir, fine_phi_p);
//  double* normal_derivative_of_field_p  = (face_is_in_negative_domain ? normal_derivative_of_vnp1_p_p[dir]  : normal_derivative_of_vnp1_m_p[dir]);
//  for (u_char der = 0; der < P4EST_DIM; ++der)
//  {
//    double signed_normal_component          = (face_is_in_negative_domain     ? -1.0  : +1.0)*local_normal[der];
//    const u_char local_oriented_der  = (signed_normal_component > 0.0  ? 2*der : 2*der + 1);
//    p4est_locidx_t neighbor_face_idx;
//    if(!faces_n->found_finest_face_neighbor(local_face_idx, dir, local_oriented_der, neighbor_face_idx))
//      return; // one of the neighbor is not a uniform fine neigbor, forget about this one and return
//    P4EST_ASSERT((neighbor_face_idx >=0) && (neighbor_face_idx < faces_n->num_local[dir] + faces_n->num_ghost[dir]));
//    // normal derivatives: always on the grid!
//    P4EST_ASSERT(fabs(normal_derivative_of_field_p[neighbor_face_idx]) < threshold_dbl_max);
//    lhs_normal_derivative_field  += fabs(signed_normal_component)/dxyz_min[der];
//    rhs_normal_derivative_field  += fabs(signed_normal_component)*normal_derivative_of_field_p[neighbor_face_idx]/dxyz_min[der];
//  }
//  if(mag_normal > 0.1)
//    normal_derivative_of_field_p[local_face_idx] = rhs_normal_derivative_field/lhs_normal_derivative_field;
//  else
//    normal_derivative_of_field_p[local_face_idx] = 0.0;
//}

//void my_p4est_two_phase_flows_t::solve_velocity_extrapolation_local_explicit_iterative(const p4est_locidx_t &local_face_idx, const u_char &dir, const my_p4est_interpolation_nodes_t &interp_normal,
//                                                                                       const double *fine_phi_p, const double *fine_phi_xxyyzz_p, const u_char &extrapolation_degree,
//                                                                                       const double *fine_jump_u_p, const double *fine_jump_mu_grad_v_p,
//                                                                                       const double *normal_derivative_of_vnp1_m_p[P4EST_DIM], const double *normal_derivative_of_vnp1_p_p[P4EST_DIM],
//                                                                                       double *vnp1_m_p[P4EST_DIM], double *vnp1_p_p[P4EST_DIM])
//{
//  neighbor_value neighbor_velocity;
//  double xyz_face[P4EST_DIM], local_normal[P4EST_DIM];
//  faces_n->xyz_fr_f(local_face_idx, dir, xyz_face);
//  interp_normal(xyz_face, local_normal);
//  double mag_normal = 0.0;
//  double avg_neighbors = 0.0;
//  double lhs_field                            = 0.0;
//  double rhs_field                            = 0.0;
//  bool too_close_flag                         = false;
//  // if the face is in negative domain, we extend the + field and the normal is reverted
//  const bool face_is_in_negative_domain       = is_face_in_negative_domain(local_face_idx, dir, fine_phi_p);
//  double* field_p                             = (face_is_in_negative_domain ? vnp1_p_p[dir]                      : vnp1_m_p[dir]);
//  const double* normal_derivative_of_field_p  = (face_is_in_negative_domain ? normal_derivative_of_vnp1_p_p[dir] : normal_derivative_of_vnp1_m_p[dir]);
//  for (u_char der = 0; der < P4EST_DIM; ++der)
//  {
//    double signed_normal_component  = (face_is_in_negative_domain     ? -1.0  : +1.0)*local_normal[der];
//    const u_char local_oriented_der = (signed_normal_component > 0.0  ? 2*der : 2*der+1);
//    mag_normal += SQR(signed_normal_component);
//    p4est_locidx_t neighbor_face_idx;
//    if(!faces_n->found_finest_face_neighbor(local_face_idx, dir, local_oriented_der, neighbor_face_idx))
//    {
//      P4EST_ASSERT(fabs(field_p[local_face_idx]) > threshold_dbl_max);
//      return; // one of the neighbor is not a uniform fine neigbor, forget about this one and return
//    }
//    P4EST_ASSERT(neighbor_face_idx >= 0 && neighbor_face_idx < faces_n->num_local[dir] + faces_n->num_ghost[dir]);
//    // field: may be subresolved
//    get_velocity_from_other_domain_seen_from_face(neighbor_velocity, local_face_idx, neighbor_face_idx, fine_phi_p, fine_phi_xxyyzz_p, vnp1_m_p[dir], vnp1_p_p[dir], fine_jump_u_p, fine_jump_mu_grad_v_p, local_oriented_der, dir);
//    P4EST_ASSERT(neighbor_velocity.distance >= 0.0);
//    if(neighbor_velocity.distance < dxyz_min[der]*(dxyz_min[der]/convert_to_xyz[der])) // "too close" --> avoid ill-defined terms due to inverses of very small distances, set the value equal to the direct neighbor
//    {
//      field_p[local_face_idx]     = neighbor_velocity.value;
//      too_close_flag              = true;
//    }
//    else
//    {
//      P4EST_ASSERT(fabs(neighbor_velocity.value) < threshold_dbl_max);
//      lhs_field                  += fabs(signed_normal_component)/neighbor_velocity.distance;
//      rhs_field                  += fabs(signed_normal_component)*neighbor_velocity.value/neighbor_velocity.distance;
//      avg_neighbors              =+ neighbor_velocity.value/((double) P4EST_DIM);
//    }
//  }
//  if(!too_close_flag)
//  {
//    mag_normal = sqrt(mag_normal);
//    if(mag_normal > EPS)
//    {
//      lhs_field /= mag_normal;
//      rhs_field /= mag_normal;
//    }
//    else
//    {
//      lhs_field = 0.0;
//      rhs_field = 0.0;
//    }

//    P4EST_ASSERT(fabs(field_p[local_face_idx]) < threshold_dbl_max);
//    if(extrapolation_degree > 0)
//      rhs_field              += normal_derivative_of_field_p[local_face_idx];
//    if(mag_normal > 0.1)
//      field_p[local_face_idx] = rhs_field/lhs_field;
//    else
//      field_p[local_face_idx] = avg_neighbors;
//    P4EST_ASSERT(!isnan(field_p[local_face_idx]) && fabs(field_p[local_face_idx]) < threshold_dbl_max);
//  }
//}

//void my_p4est_two_phase_flows_t::extrapolate_normal_derivatives_of_face_velocity_local_pseudo_time(const p4est_locidx_t &local_face_idx, const u_char &dir, const my_p4est_interpolation_nodes_t &interp_normal, const double *fine_phi_p,
//                                                                                                   double *normal_derivative_of_vnp1_m_p[P4EST_DIM], double *normal_derivative_of_vnp1_p_p[P4EST_DIM])
//{
//  double xyz_face[P4EST_DIM], local_normal[P4EST_DIM];
//  faces_n->xyz_fr_f(local_face_idx, dir, xyz_face);
//  interp_normal(xyz_face, local_normal);
//  const double mag_normal = sqrt(SUMD(SQR(local_normal[0]), SQR(local_normal[1]), SQR(local_normal[2])));
//  for (u_char dir = 0; dir < P4EST_DIM; ++dir)
//  {
//    if(mag_normal > EPS)
//      local_normal[dir] /= mag_normal;
//    else
//      local_normal[dir] = 0.0;
//  }
//  double increment_normal_derivative        = 0.0;
//  // if the face is in negative domain, we extend the + field and the normal is reverted
//  const bool face_is_in_negative_domain     = is_face_in_negative_domain(local_face_idx, dir, fine_phi_p);
//  double* normal_derivative_of_field_p      = (face_is_in_negative_domain ? normal_derivative_of_vnp1_p_p[dir] : normal_derivative_of_vnp1_m_p[dir]);
//  for (u_char der = 0; der < P4EST_DIM; ++der)
//  {
//    const double signed_normal_component    = (face_is_in_negative_domain     ? -1.0  : +1.0)*local_normal[der];
//    const u_char local_oriented_der  = (signed_normal_component > 0.0  ? 2*der : 2*der + 1);
//    p4est_locidx_t neighbor_face_idx;
//    if(!faces_n->found_finest_face_neighbor(local_face_idx, dir, local_oriented_der, neighbor_face_idx))
//      return; // one of the neighbor is not a uniform fine neigbor, forget about this one and return
//    P4EST_ASSERT(neighbor_face_idx >= 0 && neighbor_face_idx < faces_n->num_local[dir] + faces_n->num_ghost[dir]);
//    // normal derivatives: always on the grid!
//    P4EST_ASSERT(fabs(normal_derivative_of_field_p[neighbor_face_idx]) < threshold_dbl_max);
//    increment_normal_derivative  -= fabs(signed_normal_component)*(normal_derivative_of_field_p[local_face_idx] - normal_derivative_of_field_p[neighbor_face_idx])/dxyz_min[der];
//  }
//  const double dt_normal_derivative = MIN(DIM(dxyz_min[0], dxyz_min[1], dxyz_min[2]))/((double) P4EST_DIM); // as in other extensions (at nodes)
//  normal_derivative_of_field_p[local_face_idx] += dt_normal_derivative*increment_normal_derivative;
//}

//void my_p4est_two_phase_flows_t::solve_velocity_extrapolation_local_pseudo_time(const p4est_locidx_t &local_face_idx, const u_char &dir, const my_p4est_interpolation_nodes_t &interp_normal,
//                                                                                const double *fine_phi_p, const double *fine_phi_xxyyzz_p, const u_char &extrapolation_degree,
//                                                                                const double *fine_jump_u_p, const double *fine_jump_mu_grad_v_p,
//                                                                                const double *normal_derivative_of_vnp1_m_p[P4EST_DIM], const double *normal_derivative_of_vnp1_p_p[P4EST_DIM],
//                                                                                double *vnp1_m_p[P4EST_DIM], double *vnp1_p_p[P4EST_DIM])
//{
//  neighbor_value neighbor_velocity;
//  double xyz_face[P4EST_DIM], local_normal[P4EST_DIM]; faces_n->xyz_fr_f(local_face_idx, dir, xyz_face);
//  interp_normal(xyz_face, local_normal);
//  const double mag_normal = sqrt(SUMD(SQR(local_normal[0]), SQR(local_normal[1]), SQR(local_normal[2])));
//  for (u_char dir = 0; dir < P4EST_DIM; ++dir)
//  {
//    if(mag_normal > EPS)
//      local_normal[dir] /= mag_normal;
//    else
//      local_normal[dir] = 0.0;
//  }
//  double dt_field                             = MIN(DIM(dxyz_min[0], dxyz_min[1], dxyz_min[2]));
//  double increment_field                      = 0.0;
//  bool too_close_flag                         = false;
//  // if the face is in negative domain, we extend the + field and the normal is reverted
//  const bool face_is_in_negative_domain       = is_face_in_negative_domain(local_face_idx, dir, fine_phi_p);
//  double* field_p                             = (face_is_in_negative_domain ? vnp1_p_p[dir]                      : vnp1_m_p[dir]);
//  const double* normal_derivative_of_field_p  = (face_is_in_negative_domain ? normal_derivative_of_vnp1_p_p[dir] : normal_derivative_of_vnp1_m_p[dir]);
//  for (u_char der = 0; der < P4EST_DIM; ++der)
//  {
//    double signed_normal_component  = (face_is_in_negative_domain     ? -1.0  : +1.0)*local_normal[der];
//    const u_char local_oriented_der = (signed_normal_component > 0.0  ? 2*der : 2*der + 1);
//    p4est_locidx_t neighbor_face_idx;
//    if(!faces_n->found_finest_face_neighbor(local_face_idx, dir, local_oriented_der, neighbor_face_idx))
//    {
//      P4EST_ASSERT(fabs(field_p[local_face_idx]) > threshold_dbl_max);
//      return; // one of the neighbor is not a uniform fine neigbor, forget about this one and return
//    }
//    P4EST_ASSERT(neighbor_face_idx >= 0 && neighbor_face_idx < faces_n->num_local[dir] + faces_n->num_ghost[dir]);
//    // field: may be subresolved
//    get_velocity_from_other_domain_seen_from_face(neighbor_velocity, local_face_idx, neighbor_face_idx, fine_phi_p, fine_phi_xxyyzz_p, vnp1_m_p[dir], vnp1_p_p[dir], fine_jump_u_p, fine_jump_mu_grad_v_p, local_oriented_der, dir);
//    P4EST_ASSERT(neighbor_velocity.distance >= 0.0);
//    if(neighbor_velocity.distance < dxyz_min[der]*(dxyz_min[der]/convert_to_xyz[der])) // "too close" --> avoid ill-defined terms due to inverses of very small distances, set the value equal to the direct neighbor
//    {
//      field_p[local_face_idx]     = neighbor_velocity.value;
//      too_close_flag              = true;
//    }
//    else
//    {
//      P4EST_ASSERT(fabs(neighbor_velocity.value) < threshold_dbl_max);
//      increment_field            -= fabs(signed_normal_component)*(field_p[local_face_idx] - neighbor_velocity.value)/neighbor_velocity.distance;
//      dt_field                    = MIN(dt_field, neighbor_velocity.distance);
//    }
//  }
//  if(!too_close_flag)
//  {
//    dt_field /= ((double) P4EST_DIM);
//    field_p[local_face_idx] += dt_field*((extrapolation_degree > 0 ? normal_derivative_of_field_p[local_face_idx] : 0.0) + increment_field);
//    P4EST_ASSERT(!isnan(field_p[local_face_idx]) && (fabs(field_p[local_face_idx]) < threshold_dbl_max));
//  }
//}

void my_p4est_two_phase_flows_t::save_vtk(const char* name, const bool& export_fine_grid, const char* name_fine)
{
  PetscErrorCode ierr;
  Vec phi_coarse = NULL;
  interpolate_linearly_from_fine_nodes_to_coarse_nodes(fine_phi, phi_coarse);
  const double* phi_coarse_p = NULL;
  const double* hodge_p, *v_nodes_p_p, *v_nodes_m_p;
  ierr = VecGetArrayRead(hodge, &hodge_p);                  CHKERRXX(ierr);
  ierr = VecGetArrayRead(phi_coarse, &phi_coarse_p);        CHKERRXX(ierr);
  ierr = VecGetArrayRead(vnp1_nodes_m, &v_nodes_m_p);       CHKERRXX(ierr);
  ierr = VecGetArrayRead(vnp1_nodes_p, &v_nodes_p_p);       CHKERRXX(ierr);

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
                                 VTK_NODE_VECTOR_BLOCK, "vn_plus" , v_nodes_p_p,
                                 VTK_NODE_VECTOR_BLOCK, "vn_minus", v_nodes_m_p,
                                 VTK_CELL_DATA, "hodge", hodge_p);
  ierr = VecRestoreArrayRead(hodge, &hodge_p);              CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(phi_coarse, &phi_coarse_p);    CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(vnp1_nodes_m, &v_nodes_m_p);   CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(vnp1_nodes_p, &v_nodes_p_p);   CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(phi_coarse);             CHKERRXX(ierr);

  if(export_fine_grid)
  {
    P4EST_ASSERT(name_fine != NULL);
    Vec fine_jump_mu_grad_v_comp[P4EST_DIM];
    double *fine_jump_mu_grad_v_comp_p[P4EST_DIM];
    for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
      fine_jump_mu_grad_v_comp[dir] = NULL;
      ierr = create_node_vector_if_needed(fine_jump_mu_grad_v_comp[dir], fine_p4est_n, fine_nodes_n, P4EST_DIM);  CHKERRXX(ierr);
      ierr = VecGetArray(fine_jump_mu_grad_v_comp[dir], &fine_jump_mu_grad_v_comp_p[dir]);                        CHKERRXX(ierr);
    }
    const double *fine_jump_mu_grad_v_p;
    ierr = VecGetArrayRead(fine_jump_mu_grad_v, &fine_jump_mu_grad_v_p);  CHKERRXX(ierr);
    for (size_t k = 0; k < fine_nodes_n->indep_nodes.elem_count; ++k)
      for (u_char dir = 0; dir < P4EST_DIM; ++dir)
        for (u_char der = 0; der < P4EST_DIM; ++der)
          fine_jump_mu_grad_v_comp_p[dir][P4EST_DIM*k + der] = fine_jump_mu_grad_v_p[SQR_P4EST_DIM*k + P4EST_DIM*dir + der];

    ierr = VecRestoreArrayRead(fine_jump_mu_grad_v, &fine_jump_mu_grad_v_p); CHKERRXX(ierr);

    const double* fine_phi_p, *fine_curvature_p, *fine_normal_p;
    ierr = VecGetArrayRead(fine_normal, &fine_normal_p);        CHKERRXX(ierr);
    ierr = VecGetArrayRead(fine_curvature, &fine_curvature_p);  CHKERRXX(ierr);
    ierr = VecGetArrayRead(fine_phi, &fine_phi_p);              CHKERRXX(ierr);
    my_p4est_vtk_write_all_general(fine_p4est_n, fine_nodes_n, fine_ghost_n,
                                   P4EST_TRUE, P4EST_FALSE,
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
    ierr = VecRestoreArrayRead(fine_phi, &fine_phi_p);              CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(fine_curvature, &fine_curvature_p);  CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(fine_normal, &fine_normal_p);        CHKERRXX(ierr);
    for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
      ierr = VecRestoreArray(fine_jump_mu_grad_v_comp[dir], &fine_jump_mu_grad_v_comp_p[dir]);  CHKERRXX(ierr);
      ierr = delete_and_nullify_vector(fine_jump_mu_grad_v_comp[dir]);                          CHKERRXX(ierr);
      fine_jump_mu_grad_v_comp[dir] = NULL;
    }
  }
  ierr = PetscPrintf(p4est_n->mpicomm, "Saved visual data in ... %s\n", name); CHKERRXX(ierr);
}

void my_p4est_two_phase_flows_t::compute_backtraced_velocities()
{
  if(semi_lagrangian_backtrace_is_done)
    return;
  P4EST_ASSERT((vnm1_nodes_minus_xxyyzz == NULL && vnm1_nodes_plus_xxyyzz == NULL && vn_nodes_minus_xxyyzz == NULL && vn_nodes_plus_xxyyzz == NULL) ||
               (vnm1_nodes_minus_xxyyzz != NULL && vnm1_nodes_plus_xxyyzz != NULL && vn_nodes_minus_xxyyzz != NULL && vn_nodes_plus_xxyyzz != NULL));

  /* first find the velocity at the np1 points */
  my_p4est_interpolation_nodes_t interp_np1_plus(ngbd_n), interp_np1_minus(ngbd_n);
  bool use_second_derivatives = (vnm1_nodes_minus_xxyyzz != NULL && vnm1_nodes_plus_xxyyzz != NULL && vn_nodes_minus_xxyyzz != NULL && vn_nodes_plus_xxyyzz != NULL);
  p4est_locidx_t serialized_offset = 0;
  for (u_char dir = 0; dir < P4EST_DIM; ++dir)
    serialized_offset += faces_n->num_local[dir];
  vector<double> xyz_np1;     xyz_np1.resize(P4EST_DIM*serialized_offset);    // coordinates of the origin point, i.e. face centers --> same for minus and plus...
  vector<double> vnp1_minus;  vnp1_minus.resize(P4EST_DIM*serialized_offset);
  vector<double> vnp1_plus;   vnp1_plus.resize(P4EST_DIM*serialized_offset);
  serialized_offset = 0;
  for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
    for (p4est_locidx_t f_idx = 0; f_idx < faces_n->num_local[dir]; ++f_idx) {
      double xyz_face[P4EST_DIM];
      faces_n->xyz_fr_f(f_idx, dir, xyz_face);
      for (u_char comp = 0; comp < P4EST_DIM; ++comp)
        xyz_np1[P4EST_DIM*(serialized_offset + f_idx) + comp] = xyz_face[comp];

      interp_np1_minus.add_point(serialized_offset + f_idx, xyz_face);
      interp_np1_plus.add_point(serialized_offset + f_idx, xyz_face);;
    }
    serialized_offset += faces_n->num_local[dir];
  }

  if (use_second_derivatives)
  {
    interp_np1_minus.set_input(vn_nodes_minus,  vn_nodes_minus_xxyyzz,  quadratic, P4EST_DIM);
    interp_np1_plus.set_input(vn_nodes_plus,    vn_nodes_plus_xxyyzz,   quadratic, P4EST_DIM);
  }
  else
  {
    interp_np1_minus.set_input(vn_nodes_minus,  quadratic, P4EST_DIM);
    interp_np1_plus.set_input(vn_nodes_plus,    quadratic, P4EST_DIM);
  }
  interp_np1_minus.interpolate(vnp1_minus.data());
  interp_np1_plus.interpolate(vnp1_plus.data());

  /* find xyz_star */
  my_p4est_interpolation_nodes_t interp_nm1_minus(ngbd_nm1), interp_nm1_plus(ngbd_nm1);
  my_p4est_interpolation_nodes_t interp_n_minus  (ngbd_n  ), interp_n_plus  (ngbd_n  );
  serialized_offset = 0;
  for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
    for (p4est_locidx_t f_idx = 0; f_idx < faces_n->num_local[dir]; ++f_idx) {
      double xyz_star_minus[P4EST_DIM], xyz_star_plus[P4EST_DIM];
      for (u_char comp = 0; comp < P4EST_DIM; ++comp)
      {
        xyz_star_minus[comp]  = xyz_np1[P4EST_DIM*(serialized_offset + f_idx) + comp] - 0.5*dt_n*vnp1_minus[P4EST_DIM*(serialized_offset + f_idx) + comp];
        xyz_star_plus[comp]   = xyz_np1[P4EST_DIM*(serialized_offset + f_idx) + comp] - 0.5*dt_n*vnp1_plus[P4EST_DIM*(serialized_offset + f_idx) + comp];
      }
      clip_in_domain(xyz_star_minus, xyz_min, xyz_max, periodicity);
      clip_in_domain(xyz_star_plus, xyz_min, xyz_max, periodicity);

      interp_nm1_minus.add_point(serialized_offset + f_idx, xyz_star_minus);
      interp_nm1_plus.add_point(serialized_offset + f_idx, xyz_star_plus);

      interp_n_minus.add_point(serialized_offset + f_idx, xyz_star_minus);
      interp_n_plus.add_point(serialized_offset + f_idx, xyz_star_plus);
    }
    serialized_offset += faces_n->num_local[dir];
  }

  /* compute the velocities at the intermediate point */
  std::vector<double> vn_star_minus;    vn_star_minus.resize(P4EST_DIM*serialized_offset);
  std::vector<double> vn_star_plus;     vn_star_plus.resize(P4EST_DIM*serialized_offset);
  std::vector<double> vnm1_star_minus;  vnm1_star_minus.resize(P4EST_DIM*serialized_offset);
  std::vector<double> vnm1_star_plus;   vnm1_star_plus.resize(P4EST_DIM*serialized_offset);
  if(use_second_derivatives)
  {
    interp_nm1_minus.set_input(vnm1_nodes_minus,  vnm1_nodes_minus_xxyyzz,  quadratic, P4EST_DIM);
    interp_nm1_plus.set_input (vnm1_nodes_plus,   vnm1_nodes_plus_xxyyzz,   quadratic, P4EST_DIM);
    interp_n_minus.set_input  (vn_nodes_minus,    vn_nodes_minus_xxyyzz,    quadratic, P4EST_DIM);
    interp_n_plus.set_input   (vn_nodes_plus,     vn_nodes_plus_xxyyzz,     quadratic, P4EST_DIM);
  }
  else
  {
    interp_nm1_minus.set_input(vnm1_nodes_minus,  quadratic, P4EST_DIM);
    interp_nm1_plus.set_input (vnm1_nodes_plus,   quadratic, P4EST_DIM);
    interp_n_minus.set_input  (vn_nodes_minus,    quadratic, P4EST_DIM);
    interp_n_plus.set_input   (vn_nodes_plus,     quadratic, P4EST_DIM);
  }
  interp_nm1_minus.interpolate(vnm1_star_minus.data()); interp_nm1_minus.clear();
  interp_nm1_plus.interpolate(vnm1_star_plus.data());   interp_nm1_plus.clear();
  interp_n_minus.interpolate(vn_star_minus.data());     interp_n_minus.clear();
  interp_n_plus.interpolate(vn_star_plus.data());       interp_n_plus.clear();

  /* now find the departure point at time n and interpolate the appropriate velocity component there */
  serialized_offset = 0;
  for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
    backtraced_vn_faces_minus[dir].resize(faces_n->num_local[dir]);
    backtraced_vn_faces_plus[dir].resize(faces_n->num_local[dir]);
    for (p4est_locidx_t f_idx = 0; f_idx < faces_n->num_local[dir]; ++f_idx) {
      double xyz_backtraced_n_minus[P4EST_DIM], xyz_backtraced_n_plus[P4EST_DIM];
      for (u_char comp = 0; comp < P4EST_DIM; ++comp)
      {
        xyz_backtraced_n_minus[comp] = xyz_np1[P4EST_DIM*(serialized_offset + f_idx) + comp]
            - dt_n*((1.0 + .5*dt_n/dt_nm1)*vn_star_minus[P4EST_DIM*(serialized_offset + f_idx) + comp] - .5*(dt_n/dt_nm1)*vnm1_star_minus[P4EST_DIM*(serialized_offset + f_idx) + comp]);
        xyz_backtraced_n_plus[comp] = xyz_np1[P4EST_DIM*(serialized_offset + f_idx) + comp]
            - dt_n*((1.0 + .5*dt_n/dt_nm1)*vn_star_plus[P4EST_DIM*(serialized_offset + f_idx) + comp] - .5*(dt_n/dt_nm1)*vnm1_star_plus[P4EST_DIM*(serialized_offset + f_idx) + comp]);
      }
      clip_in_domain(xyz_backtraced_n_minus, xyz_min, xyz_max, periodicity);
      clip_in_domain(xyz_backtraced_n_plus, xyz_min, xyz_max, periodicity);
      interp_n_minus.add_point(f_idx, xyz_backtraced_n_minus);
      interp_n_plus.add_point(f_idx, xyz_backtraced_n_plus);
    }
    interp_n_minus.interpolate(backtraced_vn_faces_minus[dir].data(), dir); interp_n_minus.clear();
    interp_n_plus.interpolate(backtraced_vn_faces_plus[dir].data(),  dir);  interp_n_plus.clear();
    serialized_offset += faces_n->num_local[dir];
  }

  // EXTRA STUFF FOR FINDING xyz_nm1 ONLY (for second-order bdf advection terms, for instance)
  if(sl_order == 2)
  {
    /* proceed similarly for the departure point at time nm1 */
    serialized_offset = 0;
    for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
      for (p4est_locidx_t f_idx = 0; f_idx < faces_n->num_local[dir]; ++f_idx) {
        double xyz_star_minus[P4EST_DIM], xyz_star_plus[P4EST_DIM];
        for (u_char comp = 0; comp < P4EST_DIM; ++comp)
        {
          xyz_star_minus[comp]  = xyz_np1[P4EST_DIM*(serialized_offset + f_idx) + comp] - 0.5*(dt_n + dt_nm1)*vnp1_minus[P4EST_DIM*(serialized_offset + f_idx) + comp];
          xyz_star_plus[comp]   = xyz_np1[P4EST_DIM*(serialized_offset + f_idx) + comp] - 0.5*(dt_n + dt_nm1)*vnp1_plus[P4EST_DIM*(serialized_offset + f_idx) + comp];
        }
        clip_in_domain(xyz_star_minus, xyz_min, xyz_max, periodicity);
        clip_in_domain(xyz_star_plus, xyz_min, xyz_max, periodicity);

        interp_nm1_minus.add_point(serialized_offset + f_idx, xyz_star_minus);
        interp_nm1_plus.add_point(serialized_offset + f_idx, xyz_star_plus);
        interp_n_minus.add_point(serialized_offset + f_idx, xyz_star_minus);
        interp_n_plus.add_point(serialized_offset + f_idx, xyz_star_plus);
      }
      serialized_offset += faces_n->num_local[dir];
    }

    /* compute the velocities at the intermediate point */
    interp_nm1_minus.interpolate(vnm1_star_minus.data()); interp_nm1_minus.clear();
    interp_nm1_plus.interpolate(vnm1_star_plus.data());   interp_nm1_plus.clear();
    interp_n_minus.interpolate(vn_star_minus.data());     interp_n_minus.clear();
    interp_n_plus.interpolate(vn_star_plus.data());       interp_n_plus.clear();

    /* now find the departure point at time nm1 and interpolate the appropriate velocity component there */
    serialized_offset = 0;
    for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
      backtraced_vnm1_faces_minus[dir].resize(faces_n->num_local[dir]);
      backtraced_vnm1_faces_plus[dir].resize(faces_n->num_local[dir]);
      for (p4est_locidx_t f_idx = 0; f_idx < faces_n->num_local[dir]; ++f_idx) {
        double xyz_backtraced_nm1_minus[P4EST_DIM], xyz_backtraced_nm1_plus[P4EST_DIM];
        for (u_char comp = 0; comp < P4EST_DIM; ++comp)
        {
          xyz_backtraced_nm1_minus[comp] = xyz_np1[P4EST_DIM*(serialized_offset + f_idx) + comp]
              - (dt_n + dt_nm1)*((1.0 + .5*(dt_n - dt_nm1)/dt_nm1)*vn_star_minus[P4EST_DIM*(serialized_offset + f_idx) + comp] - .5*((dt_n - dt_nm1)/dt_nm1)*vnm1_star_minus[P4EST_DIM*(serialized_offset + f_idx) + comp]);
          xyz_backtraced_nm1_plus[comp] = xyz_np1[P4EST_DIM*(serialized_offset + f_idx) + comp]
              - (dt_n + dt_nm1)*((1.0 + .5*(dt_n - dt_nm1)/dt_nm1)*vn_star_plus[P4EST_DIM*(serialized_offset + f_idx) + comp] - .5*((dt_n - dt_nm1)/dt_nm1)*vnm1_star_plus[P4EST_DIM*(serialized_offset + f_idx) + comp]);
        }
        clip_in_domain(xyz_backtraced_nm1_minus, xyz_min, xyz_max, periodicity);
        clip_in_domain(xyz_backtraced_nm1_plus, xyz_min, xyz_max, periodicity);

        interp_nm1_minus.add_point(f_idx, xyz_backtraced_nm1_minus);
        interp_nm1_plus.add_point(f_idx, xyz_backtraced_nm1_plus);
      }
      interp_nm1_minus.interpolate(backtraced_vnm1_faces_minus[dir].data(), dir); interp_nm1_minus.clear();
      interp_nm1_plus.interpolate(backtraced_vnm1_faces_plus[dir].data(), dir);   interp_nm1_plus.clear();
      serialized_offset += faces_n->num_local[dir];
    }
  }
  semi_lagrangian_backtrace_is_done = true;
  return;
}

void my_p4est_two_phase_flows_t::set_interface_velocity()
{
  PetscErrorCode ierr;
  ierr = create_node_vector_if_needed(interface_velocity_np1, p4est_n, nodes_n, P4EST_DIM); CHKERRXX(ierr);
  double *interface_velocity_np1_p  = NULL;
  const double *fluid_velocity_m_p  = NULL;
  const double *fluid_velocity_p_p  = NULL;
  const double *fine_mass_flux_p    = NULL;
  const double *fine_normal_p       = NULL;
  ierr = VecGetArray(interface_velocity_np1, &interface_velocity_np1_p);  CHKERRXX(ierr);
  ierr = VecGetArrayRead(vnp1_nodes_p, &fluid_velocity_p_p);              CHKERRXX(ierr);
  ierr = VecGetArrayRead(vnp1_nodes_m, &fluid_velocity_m_p);              CHKERRXX(ierr);
  if(fine_mass_flux != NULL){
    ierr = VecGetArrayRead(fine_mass_flux, &fine_mass_flux_p);            CHKERRXX(ierr);
    ierr = VecGetArrayRead(fine_normal, &fine_normal_p);                  CHKERRXX(ierr);
  }
  double xyz_node[P4EST_DIM];
  for (size_t k = 0; k < ngbd_n->get_layer_size(); ++k) {
    p4est_locidx_t n = ngbd_n->get_layer_node(k);
    node_xyz_fr_n(n, p4est_n, nodes_n, xyz_node);
    domain_side side_to_choose = ((*interp_phi)(xyz_node) <= 0.0 ? OMEGA_MINUS : OMEGA_PLUS);
    p4est_locidx_t fine_node_idx;
    size_t position = P4EST_DIM*n;
    for (u_char dir = 0; dir < P4EST_DIM; ++dir)
      interface_velocity_np1_p[position + dir] = (side_to_choose == OMEGA_MINUS ? fluid_velocity_m_p[position + dir] : fluid_velocity_p_p[position + dir]);
    if(fine_mass_flux != NULL && get_fine_node_idx_node(n, fine_node_idx))
    {
      size_t fine_position = P4EST_DIM*fine_node_idx;
      for (u_char dir = 0; dir < P4EST_DIM; ++dir)
        interface_velocity_np1_p[position + dir] -= (fine_mass_flux_p[fine_position + dir]*fine_normal_p[fine_position + dir]/(side_to_choose == OMEGA_MINUS ? rho_m : rho_p));
    }
  }
  ierr = VecGhostUpdateBegin(interface_velocity_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for (size_t k = 0; k < ngbd_n->get_local_size(); ++k) {
    p4est_locidx_t n = ngbd_n->get_local_node(k);
    node_xyz_fr_n(n, p4est_n, nodes_n, xyz_node);
    domain_side side_to_choose = ((*interp_phi)(xyz_node) <= 0.0 ? OMEGA_MINUS : OMEGA_PLUS);
    p4est_locidx_t fine_node_idx;
    size_t position = P4EST_DIM*n;
    for (u_char dir = 0; dir < P4EST_DIM; ++dir)
      interface_velocity_np1_p[position + dir] = (side_to_choose == OMEGA_MINUS ? fluid_velocity_m_p[position + dir] : fluid_velocity_p_p[position + dir]);
    if(fine_mass_flux != NULL && get_fine_node_idx_node(n, fine_node_idx))
    {
      size_t fine_position = P4EST_DIM*fine_node_idx;
      for (u_char dir = 0; dir < P4EST_DIM; ++dir)
        interface_velocity_np1_p[position + dir] -= (fine_mass_flux_p[fine_position + dir]*fine_normal_p[fine_position + dir]/(side_to_choose == OMEGA_MINUS ? rho_m : rho_p));
    }
  }
//  if(!node_to_fine_node_map_is_set)
//    node_to_fine_node_map_is_set = true;
  ierr = VecGhostUpdateEnd(interface_velocity_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = create_node_vector_if_needed(interface_velocity_np1_xxyyzz, p4est_n, nodes_n, SQR_P4EST_DIM); CHKERRXX(ierr);
  ngbd_n->second_derivatives_central(interface_velocity_np1, interface_velocity_np1_xxyyzz, P4EST_DIM);

  if(fine_mass_flux != NULL){
    ierr = VecRestoreArrayRead(fine_normal, &fine_normal_p);                  CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(fine_mass_flux, &fine_mass_flux_p);            CHKERRXX(ierr);
  }
  ierr = VecRestoreArrayRead(vnp1_nodes_p, &fluid_velocity_p_p);              CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(vnp1_nodes_m, &fluid_velocity_m_p);              CHKERRXX(ierr);
  ierr = VecRestoreArray(interface_velocity_np1, &interface_velocity_np1_p);  CHKERRXX(ierr);
}

void my_p4est_two_phase_flows_t::advect_interface(p4est_t *fine_p4est_np1, p4est_nodes_t *fine_nodes_np1, Vec fine_phi_np1,
                                                  p4est_nodes_t *known_nodes, Vec known_phi_np1)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_two_phase_advect_interface, 0, 0, 0, 0); CHKERRXX(ierr);
  P4EST_ASSERT(known_nodes != NULL);

  my_p4est_interpolation_nodes_t interp_n(ngbd_nm1); // yes, it's normal: we are in the process of advancing time...
  my_p4est_interpolation_nodes_t interp_np1(ngbd_n); // yes, it's normal: we are in the process of advancing time...

  std::vector<double> v_tmp_n;
  std::vector<double> v_tmp_np1;
  std::vector<size_t> already_known(0);
  p4est_locidx_t origin_local_idx;
  const p4est_quadrant_t *node = NULL;
  const double *known_phi_np1_p = NULL;
  double *fine_phi_np1_p = NULL;
  if(known_phi_np1 != NULL)
  {
    ierr = VecGetArrayRead(known_phi_np1, &known_phi_np1_p);  CHKERRXX(ierr);
    ierr = VecGetArray(fine_phi_np1, &fine_phi_np1_p);        CHKERRXX(ierr);
  }

  //  size_t n_to_show = -1;

  /* find the velocity field at time np1 */
  size_t to_compute = 0;
  for (size_t n=0; n < fine_nodes_np1->indep_nodes.elem_count; ++n)
  {
    if(known_phi_np1 != NULL)
    {
      node = (const p4est_quadrant_t*) sc_array_index(&fine_nodes_np1->indep_nodes, n);
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
    if((known_phi_np1 != NULL) && (n == already_known[known_idx]))
    {
      known_idx++;
      continue;
    }
    double xyz_star[P4EST_DIM];
    node_xyz_fr_n(n, fine_p4est_np1, fine_nodes_np1, xyz_star);
    size_t position = P4EST_DIM*to_compute;
    for (u_char dir = 0; dir < P4EST_DIM; ++dir)
    {
      xyz_star[dir] -= .5*dt_n*v_tmp_np1[position + dir];
      //      if(to_compute==n_to_show)
      //      {
      //        std::cout << "vstar[" << (int) dir << "] = " << v_tmp_np1[position + dir] << std::endl;
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
    if((known_phi_np1 != NULL) && (n == already_known[known_idx]))
    {
      known_idx++;
      continue;
    }
    size_t position = P4EST_DIM*to_compute;
    double xyz_d[P4EST_DIM]; node_xyz_fr_n(n, fine_p4est_np1, fine_nodes_np1, xyz_d);
    for (u_char dir = 0; dir < P4EST_DIM; ++dir){
      v_star[dir] = (1.0 + 0.5*dt_n/dt_nm1)*v_tmp_np1[position + dir] - 0.5*dt_n/dt_nm1*v_tmp_n[position + dir];
      xyz_d[dir] -= dt_n*v_star[dir];
      //      if(to_compute==n_to_show)
      //      {
      //        std::cout << "vd[" << (int) dir << "] = " << v_star[position + dir] << std::endl;
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
  if(fine_phi_xxyyzz != NULL)
    interp_phi->set_input(fine_phi, fine_phi_xxyyzz, quadratic_non_oscillatory);
  else
    interp_phi->set_input(fine_phi, linear);

  ierr = PetscLogEventEnd(log_my_p4est_two_phase_advect_interface, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_two_phase_flows_t::update_from_tn_to_tnp1(/*const unsigned int &nnn*/)
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
    ierr = create_node_vector_if_needed(interface_velocity_n_xxyyzz, p4est_nm1, nodes_nm1, SQR_P4EST_DIM);
    Vec loc_interface_velocity_n, loc_interface_velocity_n_xxyyzz;
    ierr = VecGhostGetLocalForm(interface_velocity_n, &loc_interface_velocity_n);                   CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(interface_velocity_n_xxyyzz, &loc_interface_velocity_n_xxyyzz);     CHKERRXX(ierr);
    ierr = VecSet(loc_interface_velocity_n, 0.0);                                                   CHKERRXX(ierr);
    ierr = VecSet(loc_interface_velocity_n_xxyyzz, 0.0);                                            CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(interface_velocity_n_xxyyzz, &loc_interface_velocity_n_xxyyzz); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(interface_velocity_n, &loc_interface_velocity_n);               CHKERRXX(ierr);
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
  p4est_t *p4est_np1                  = (iterative_grid_update_converged ? p4est_n : p4est_copy(p4est_n, P4EST_FALSE)); // no need to copy if no need to refine/coarsen, otherwise very efficient copy
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
    P4EST_ASSERT((iter > 0 && previous_phi_np1 != NULL)  || (iter == 0 && previous_phi_np1 == NULL));

    // advect phi
    advect_interface(p4est_np1, nodes_np1, phi_np1,
                     previous_nodes_np1, previous_phi_np1); // limit the workload: use what you already know if you know it!

    // get vorticity_np1
    if(iter > 0){
      // reset vorticity_np1 if needed
      vorticities_np1 = NULL;
      ierr = create_node_vector_if_needed(vorticities_np1, p4est_np1, nodes_np1, 2);  CHKERRXX(ierr);
      const double *previous_vorticities_np1_p;
      double *vorticities_np1_p;
      ierr = VecGetArrayRead(previous_vorticities_np1, &previous_vorticities_np1_p);  CHKERRXX(ierr);
      ierr = VecGetArray(vorticities_np1, &vorticities_np1_p);                        CHKERRXX(ierr);

      interp_nodes.clear();
      for(size_t n=0; n < nodes_np1->indep_nodes.elem_count; ++n)
      {
        const p4est_quadrant_t *node = (const p4est_quadrant_t*) sc_array_index(&nodes_np1->indep_nodes, n);
        P4EST_ASSERT (p4est_quadrant_is_node (node, 1));
        p4est_locidx_t origin_idx;
        if(index_of_node(node, previous_nodes_np1, origin_idx))
          for (u_char k = 0; k < 2; ++k)
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

      ierr = VecRestoreArrayRead(previous_vorticities_np1, &previous_vorticities_np1_p);  CHKERRXX(ierr);
      ierr = VecRestoreArray(vorticities_np1, &vorticities_np1_p);                        CHKERRXX(ierr);
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

    if(previous_phi_np1 != NULL){
      ierr = delete_and_nullify_vector(previous_phi_np1);         CHKERRXX(ierr); }
    if(previous_nodes_np1 != nodes_n)
      p4est_nodes_destroy(previous_nodes_np1);
    if(previous_vorticities_np1 != vorticities){
      ierr = delete_and_nullify_vector(previous_vorticities_np1); CHKERRXX(ierr); }

    if(iter > ((unsigned int) 2+criterion_computational_grid.max_lvl-criterion_computational_grid.min_lvl)) // increase the rhs by one to account for the very first step that used to be out of the loop, [Raphael]
    {
      ierr = PetscPrintf(p4est_n->mpicomm, "ooops ... the grid update did not converge\n"); CHKERRXX(ierr);
      break;
    }
  }

  if(vorticities_np1 != vorticities){
    ierr = delete_and_nullify_vector(vorticities_np1); CHKERRXX(ierr); }
  // we do not need the vorticities anymore
  ierr = delete_and_nullify_vector(vorticities); CHKERRXX(ierr);

  // Save what we already know from the advection of the levelset (in case we know anything) in
  // order to limit workload later on, regarding the adaptation for the interface-capturing grid
  p4est_nodes_t *fine_nodes_np1 = nodes_np1;
  Vec fine_phi_np1              = phi_np1; phi_np1 = NULL;

  // Finalize the computational grid np1 (if needed):
  if(p4est_np1 != p4est_n)
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
  p4est_t *fine_p4est_np1                   = (p4est_np1 != p4est_n ? p4est_copy(p4est_np1, P4EST_FALSE) : p4est_copy(fine_p4est_n, P4EST_FALSE));
  fine_p4est_np1->connectivity              = fine_p4est_n->connectivity; // connectivity is not duplicated by p4est_copy, the pointer (i.e. the memory-address) of connectivity seems to be copied from my understanding of the source file of p4est_copy, but I feel this is a bit safer [Raphael Egan]
  fine_p4est_np1->user_pointer              = data_fine;
  // fine_nodes_np1 and fine_phi_np1 have been initialized here above (to shortcut workload in advection in the following loop, if possible)
  iter = 0;
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
    if(previous_fine_nodes_np1 != nodes_n)
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
      const p4est_quadrant_t *node = (const p4est_quadrant_t*) sc_array_index(&fine_nodes_np1->indep_nodes, n);
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
      const p4est_quadrant_t *node = (const p4est_quadrant_t*) sc_array_index(&fine_nodes_np1->indep_nodes, n);
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
  if(previous_fine_nodes_np1 != nodes_n)
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
  if(fine_mass_flux != NULL){
    ierr = delete_and_nullify_vector(fine_mass_flux);                 CHKERRXX(ierr); }
  if(fine_variable_surface_tension != NULL){
    ierr = delete_and_nullify_vector(fine_variable_surface_tension);  CHKERRXX(ierr); }
  // on computational grid at time nm1, just "slide" fields and grids in discrete time
  if(p4est_nm1 != p4est_n)
    p4est_destroy(p4est_nm1);
  if(ghost_nm1 != ghost_n)
    p4est_ghost_destroy(ghost_nm1);
  if(nodes_nm1 != nodes_n)
    p4est_nodes_destroy(nodes_nm1);
  if(hierarchy_nm1 != hierarchy_n)
    delete hierarchy_nm1;
  if(ngbd_nm1 != ngbd_n)
    delete ngbd_nm1;
  p4est_nm1     = p4est_n;
  ghost_nm1     = ghost_n;
  nodes_nm1     = nodes_n;
  hierarchy_nm1 = hierarchy_n;
  ngbd_nm1      = ngbd_n;
  ierr = delete_and_nullify_vector(vnm1_nodes_m);                     CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(vnm1_nodes_p);                     CHKERRXX(ierr);
  if(vnm1_nodes_m_xxyyzz != NULL){
    ierr = delete_and_nullify_vector(vnm1_nodes_m_xxyyzz);            CHKERRXX(ierr); }
  if(vnm1_nodes_p_xxyyzz != NULL){
    ierr = delete_and_nullify_vector(vnm1_nodes_p_xxyyzz);            CHKERRXX(ierr); }
  if(interface_velocity_n != NULL){
    ierr = delete_and_nullify_vector(interface_velocity_n);           CHKERRXX(ierr); }
  if(interface_velocity_n_xxyyzz != NULL){
    ierr = delete_and_nullify_vector(interface_velocity_n_xxyyzz);    CHKERRXX(ierr); }
  vnm1_nodes_m        = vn_nodes_m;
  vnm1_nodes_p        = vn_nodes_p;
  vnm1_nodes_m_xxyyzz = vn_nodes_m_xxyyzz;
  vnm1_nodes_p_xxyyzz = vn_nodes_p_xxyyzz;
  interface_velocity_n        = interface_velocity_np1;
  interface_velocity_n_xxyyzz = interface_velocity_np1_xxyyzz;
  // on computational grid at time n, we will need to interpolate things...
  // we will need to interpolate for vnp1 to vn (and then we'll take their derivatives)
  vn_nodes_m        = NULL; vn_nodes_p         = NULL;
  vn_nodes_m_xxyyzz = NULL; vn_nodes_p_xxyyzz  = NULL;
  ierr = create_node_vector_if_needed(vn_nodes_m, p4est_np1, nodes_np1, P4EST_DIM);
  ierr = create_node_vector_if_needed(vn_nodes_p, p4est_np1, nodes_np1, P4EST_DIM);
  ierr = create_node_vector_if_needed(vn_nodes_m_xxyyzz, p4est_np1, nodes_np1, SQR_P4EST_DIM);
  ierr = create_node_vector_if_needed(vn_nodes_p_xxyyzz, p4est_np1, nodes_np1, SQR_P4EST_DIM);
  interp_nodes.clear();
  for (size_t k = 0; k < nodes_np1->indep_nodes.elem_count; ++k) {
    double xyz[P4EST_DIM];
    node_xyz_fr_n(k, p4est_np1,nodes_np1, xyz);
    interp_nodes.add_point(k, xyz);
  }
  Vec inputs[2] = {vnp1_nodes_m, vnp1_nodes_p};
  Vec outputs[2] = {vn_nodes_m, vn_nodes_p};
  Vec outputs_xxyyzz[2] = {vn_nodes_m_xxyyzz, vn_nodes_p_xxyyzz};
  interp_nodes.set_input(inputs, linear, 2, P4EST_DIM);
  interp_nodes.interpolate(outputs);
  ngbd_np1->second_derivatives_central(outputs, outputs_xxyyzz, 2, P4EST_DIM);
  ierr = delete_and_nullify_vector(vnp1_nodes_m);                     CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(vnp1_nodes_p);                     CHKERRXX(ierr);
  interface_velocity_np1        = NULL;
  interface_velocity_np1_xxyyzz = NULL;

  ierr = delete_and_nullify_vector(hodge);                            CHKERRXX(ierr);
  for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = delete_and_nullify_vector(dxyz_hodge[dir]);                CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(vstar[dir]);                     CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(vnp1_m[dir]);                    CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(vnp1_p[dir]);                    CHKERRXX(ierr);
  }
  if(p4est_np1 != p4est_n || ghost_np1 != ghost_n || hierarchy_np1 != hierarchy_n){
    if(hierarchy_np1 != hierarchy_n)
    {
      delete  ngbd_c;
      ngbd_c = new my_p4est_cell_neighbors_t(hierarchy_np1);
      cell_to_fine_node.clear(); cell_to_fine_node_map_is_set = false;
    }
    delete  faces_n;
    faces_n = new my_p4est_faces_t(p4est_np1, ghost_np1, brick, ngbd_c, true);
  }
  p4est_n     = p4est_np1;
  ghost_n     = ghost_np1;
  nodes_n     = nodes_np1;
  hierarchy_n = hierarchy_np1;
  ngbd_n      = ngbd_np1;
  viscosity_solver.reset();

  // clear grid-related buffers, flags and backtrace semi-lagrangian points
  semi_lagrangian_backtrace_is_done = false;
  for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
    backtraced_vn_faces[dir].clear();
    backtraced_vnm1_faces[dir].clear();
    face_to_fine_node[dir].clear();
    face_to_fine_node_maps_are_set[dir] = false;
    ierr = VecCreateGhostFaces(p4est_n, faces_n, &vstar[dir],       dir); CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est_n, faces_n, &dxyz_hodge[dir],  dir); CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est_n, faces_n, &vnp1_m[dir],  dir);     CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est_n, faces_n, &vnp1_p[dir],   dir);    CHKERRXX(ierr);
    if(!voronoi_on_the_fly)
    {
      voro_cell[dir].clear();
      voro_cell[dir].resize(faces_n->num_local[dir]);
    }
  }
  ierr = VecCreateGhostCells(p4est_n, ghost_n, &hodge);                   CHKERRXX(ierr);
  cell_to_fine_node.clear(); cell_to_fine_node_map_is_set = false;
  node_to_fine_node.clear(); node_to_fine_node_map_is_set = false;

  ierr = PetscLogEventEnd(log_my_p4est_two_phase_advect_update, 0, 0, 0, 0); CHKERRXX(ierr);


  //  {
  //    PetscErrorCode ierr;
  //    Vec phi_coarse = NULL;
  //    interpolate_linearly_from_fine_nodes_to_coarse_nodes(fine_phi, phi_coarse);
  //    const double *phi_coarse_p = NULL;
  //    const double *v_nodes_p_p, *v_nodes_m_p;
  //    ierr = VecGetArrayRead(phi_coarse, &phi_coarse_p); CHKERRXX(ierr);
  //    ierr = VecGetArrayRead(vn_nodes_m, &v_nodes_m_p); CHKERRXX(ierr);
  //    ierr = VecGetArrayRead(vn_nodes_p, &v_nodes_p_p); CHKERRXX(ierr);

  //    my_p4est_vtk_write_all_general(p4est_n, nodes_n, ghost_n,
  //                                   P4EST_TRUE, P4EST_TRUE,
  //                                   1, /* number of VTK_POINT_DATA */
  //                                   0, /* number of VTK_POINT_DATA_VECTOR_BY_COMPONENTS */
  //                                   2, /* number of VTK_POINT_DATA_VECTOR_BLOCK */
  //                                   0, /* number of VTK_CELL_DATA */
  //                                   0, /* number of VTK_CELL_DATA_VECTOR_BY_COMPONENTS */
  //                                   0, /* number of VTK_CELL_DATA_VECTOR_BLOCK */
  //                                   "/home/regan/workspace/projects/two_phase_flow/after_update",
  //                                   VTK_NODE_VECTOR_BLOCK, "vn_plus" , v_nodes_p_p,
  //                                   VTK_NODE_VECTOR_BLOCK, "vn_minus", v_nodes_m_p,
  //                                   VTK_POINT_DATA, "phi", phi_coarse_p);
  //    ierr = VecRestoreArrayRead(phi_coarse, &phi_coarse_p); CHKERRXX(ierr);
  //    ierr = VecRestoreArrayRead(vn_nodes_m, &v_nodes_m_p); CHKERRXX(ierr);
  //    ierr = VecRestoreArrayRead(vn_nodes_p, &v_nodes_p_p); CHKERRXX(ierr);
  //    ierr = delete_and_nullify_vector(phi_coarse); CHKERRXX(ierr);
  //  }

  return;
}

