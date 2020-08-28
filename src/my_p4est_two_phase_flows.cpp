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
    const double inv_quad_scal      = 1.0/((double) (1 << quad->level));
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
  ierr = VecGetArrayRead(phi_np1_on_computational_nodes,                        &phi_np1_on_computational_nodes_p);                       CHKERRXX(ierr);
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

  bool is_grid_changed = false;
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
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &is_grid_changed, 1, MPI_CXX_BOOL, MPI_LOR, p4est_np1->mpicomm); SC_CHECK_MPI(mpiret);

  ierr = VecRestoreArrayRead(vorticity_magnitude_np1_on_computational_nodes_plus,   &vorticity_magnitude_np1_on_computational_nodes_plus_p);  CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(vorticity_magnitude_np1_on_computational_nodes_minus,  &vorticity_magnitude_np1_on_computational_nodes_minus_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(phi_np1_on_computational_nodes,                        &phi_np1_on_computational_nodes_p);                       CHKERRXX(ierr);

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
  cfl_advection = 1.0;
  cfl_surface_tension = 0.5;
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
  tree_diagonal = ABSD(tree_dimension[0], tree_dimension[1], tree_dimension[2]);
  smallest_diagonal = ABSD(dxyz_smallest_quad[0], dxyz_smallest_quad[1], dxyz_smallest_quad[2]);

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
    grad_p_guess_over_rho_minus[dir]  = NULL;
    grad_p_guess_over_rho_plus[dir]   = NULL;
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
  pressure_guess_solver = NULL;
  pressure_guess_is_set = false;
  divergence_free_projector = NULL;
  cell_jump_solver_to_use = FV; // default is finite-volume solver for projection step
  fetch_interface_FD_neighbors_with_second_order_accuracy = false;
}

my_p4est_two_phase_flows_t::~my_p4est_two_phase_flows_t()
{
  PetscErrorCode ierr;
  // if using subcell resolution, you'll want to take care of this separately
  if(phi_on_computational_nodes != phi) {
    ierr = delete_and_nullify_vector(phi_on_computational_nodes);     CHKERRXX(ierr); }
  // node-sampled fields on the interface-capturing grid
  ierr = delete_and_nullify_vector(phi);                            CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(pressure_jump);                  CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(mass_flux);                      CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(phi_xxyyzz);                     CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(interface_stress);               CHKERRXX(ierr);
  // node-sampled fields on the computational grids n
  ierr = delete_and_nullify_vector(vorticity_magnitude_minus);      CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(vorticity_magnitude_plus);       CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(vnp1_nodes_minus);               CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(vnp1_nodes_plus);                CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(vn_nodes_minus);                 CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(vn_nodes_plus);                  CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(interface_velocity_np1);         CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(vn_nodes_minus_xxyyzz);          CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(vn_nodes_plus_xxyyzz);           CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(interface_velocity_np1_xxyyzz);  CHKERRXX(ierr);
  // node-sampled fields on the computational grids nm1
  ierr = delete_and_nullify_vector(vnm1_nodes_minus);               CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(vnm1_nodes_plus);                CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(vnm1_nodes_minus_xxyyzz);        CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(vnm1_nodes_plus_xxyyzz);         CHKERRXX(ierr);
  // face-sampled fields, computational grid n
  for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = delete_and_nullify_vector(grad_p_guess_over_rho_minus[dir]); CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(grad_p_guess_over_rho_plus[dir]);  CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(vnp1_face_minus[dir]);             CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(vnp1_face_plus[dir]);              CHKERRXX(ierr);
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
  if(pressure_guess_solver != NULL)
    delete pressure_guess_solver;
  if(divergence_free_projector != NULL)
    delete divergence_free_projector;
}

void my_p4est_two_phase_flows_t::set_phi(Vec phi_on_interface_capturing_nodes, const interpolation_method& method, Vec phi_on_computational_nodes_)
{
  PetscErrorCode ierr;
  P4EST_ASSERT(phi_on_interface_capturing_nodes != NULL);
  if(phi != phi_on_interface_capturing_nodes)
  {
    ierr = delete_and_nullify_vector(phi); CHKERRXX(ierr);
    phi = phi_on_interface_capturing_nodes;
    ierr = delete_and_nullify_vector(phi_xxyyzz); CHKERRXX(ierr);// no longer valid
  }
  else
  {
    if(method == linear){
      ierr = delete_and_nullify_vector(phi_xxyyzz); CHKERRXX(ierr);
    }
  }

  levelset_interpolation_method = method;

  if(levelset_interpolation_method != linear && phi_xxyyzz == NULL){
    ierr = interface_manager->create_vector_on_interface_capturing_nodes(phi_xxyyzz, P4EST_DIM); CHKERRXX(ierr);
    interface_manager->get_interface_capturing_ngbd_n().second_derivatives_central(phi, phi_xxyyzz);
  }
  interface_manager->set_levelset(phi, levelset_interpolation_method, phi_xxyyzz, true, true);

#ifdef CASL_THROWS
  if(interface_manager->subcell_resolution() == 0 && phi_on_computational_nodes_ != NULL && phi_on_computational_nodes_ != phi_on_interface_capturing_nodes)
    throw std::invalid_argument("my_p4est_two_phase_flows_t::set_phi() : if not using subcell-resolution for your levelset, this object requires phi_on_interface_capturing_nodes == phi_on_computational_nodes_ or phi_on_computational_nodes_ == NULL");
#endif

  if(interface_manager->subcell_resolution() == 0)
    phi_on_computational_nodes = phi_on_interface_capturing_nodes; // the interface-manager figures it out for itself, like a big boy!
  else if(phi_on_computational_nodes_ != NULL)
  {
    if(phi_on_computational_nodes != phi_on_computational_nodes_){
      ierr =  delete_and_nullify_vector(phi_on_computational_nodes); CHKERRXX(ierr); }
    phi_on_computational_nodes = phi_on_computational_nodes_;
    interface_manager->set_under_resolved_levelset(phi_on_computational_nodes);
  }

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
  if(pressure_jump == NULL)
    interface_manager->create_vector_on_interface_capturing_nodes(pressure_jump);


  /* HOW TO MAKE SURE WE TAKE [2\mu n \cdot E \cdot n] into account, in general?
   * irrelevant so long as mu_minus and mu_plus are equal, but we'll need to address that, eventually
   */

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
      interface_manager->set_curvature(); // Maybe we'd want to flatten it...

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

  ierr = delete_and_nullify_vector(n_cdot_grad_u_minus_cdot_n); CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(n_cdot_grad_u_plus_cdot_n); CHKERRXX(ierr);
  if(interp_n_cdot_grad_u_minus_cdot_n != NULL)
    delete interp_n_cdot_grad_u_minus_cdot_n;
  if(interp_n_cdot_grad_u_plus_cdot_n != NULL)
    delete interp_n_cdot_grad_u_plus_cdot_n;
  return;
}

/* solve the pressure guess equation:
 * -div((1.0/rho)*grad(p_guess)) = 0.0
 * jump_p = -surface_tension*kappa  - SQR(mass_flux)*jump_inverse_mass_density()
 * jump_normal_flux = 0.0
 */
void my_p4est_two_phase_flows_t::solve_for_pressure_guess(const KSPType ksp, const PCType pc)
{
  if(pressure_guess_is_set)
    return;
  /* Make the two-phase velocity field divergence free : */
  compute_pressure_jump();

  if(pressure_guess_solver == NULL)
  {
    if(cell_jump_solver_to_use == GFM || cell_jump_solver_to_use == xGFM)
    {
      pressure_guess_solver = new my_p4est_poisson_jump_cells_xgfm_t(ngbd_c, nodes_n);
      dynamic_cast<my_p4est_poisson_jump_cells_xgfm_t*>(pressure_guess_solver)->activate_xGFM_corrections(cell_jump_solver_to_use == xGFM);
    }
    else
      pressure_guess_solver = new my_p4est_poisson_jump_cells_fv_t(ngbd_c, nodes_n);
  }

  pressure_guess_solver->set_interface(interface_manager);
  pressure_guess_solver->set_diagonals(0.0, 0.0);
  pressure_guess_solver->set_mus(1.0/rho_minus, 1.0/rho_plus);
  if(bc_pressure != NULL)
    pressure_guess_solver->set_bc(*bc_pressure);
  pressure_guess_solver->set_jumps(pressure_jump, NULL);
  pressure_guess_solver->solve(ksp, pc);

  PetscErrorCode ierr;
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    if(grad_p_guess_over_rho_minus[dim] == NULL){
      ierr = VecCreateGhostFaces(p4est_n, faces_n, &grad_p_guess_over_rho_minus[dim], dim); CHKERRXX(ierr); }
    if(grad_p_guess_over_rho_plus[dim] == NULL){
      ierr = VecCreateGhostFaces(p4est_n, faces_n, &grad_p_guess_over_rho_plus[dim], dim); CHKERRXX(ierr); }
  }

  pressure_guess_solver->get_flux_components(grad_p_guess_over_rho_minus, grad_p_guess_over_rho_plus, faces_n);

  pressure_guess_is_set = true;
  return;
}


/* solve the projection step, we consider (PHI/rho) to be the HODGE variable, define the divergence-free projection as
 * v^{n + 1} = v^{\star} - (1.0/rho) grad PHI, and we solve for PHI as the solution of
 * -div((1.0/rho)*grad(PHI)) = -div(vstar)
 * jump_PHI = 0.0
 * jump_normal_flux = 0.0! --> because we assume the normal jump in u_star has been correctly captured earlier on...
 */
void my_p4est_two_phase_flows_t::solve_projection(const KSPType ksp, const PCType pc)
{
  /* Make the two-phase velocity field divergence free : */
  if(divergence_free_projector == NULL)
  {
    if(cell_jump_solver_to_use == GFM || cell_jump_solver_to_use == xGFM)
    {
      divergence_free_projector = new my_p4est_poisson_jump_cells_xgfm_t(ngbd_c, nodes_n);
      dynamic_cast<my_p4est_poisson_jump_cells_xgfm_t*>(divergence_free_projector)->activate_xGFM_corrections(cell_jump_solver_to_use == xGFM);
    }
    else
      divergence_free_projector = new my_p4est_poisson_jump_cells_fv_t(ngbd_c, nodes_n);
  }

  divergence_free_projector->set_interface(interface_manager);
  divergence_free_projector->set_diagonals(0.0, 0.0);
  divergence_free_projector->set_mus(1.0/rho_minus, 1.0/rho_plus);
  struct wall_type_hodge_t : WallBCDIM{
    BoundaryConditionType operator()(DIM(double, double, double)) const { return NEUMANN; }
  } wall_type_hodge;
  BoundaryConditionsDIM bc_hodge;
  bc_hodge.setWallTypes(wall_type_hodge); bc_hodge.setWallValues(zero_cf);
  divergence_free_projector->set_bc(bc_hodge);
  divergence_free_projector->set_jumps(NULL, NULL);

  if(mass_flux != NULL)
    throw std::runtime_error("my_p4est_two_phase_flows_t::solve_projection : you have more work to do here when considering a nonzero mass flux");
  divergence_free_projector->set_velocity_on_faces(vnp1_face_minus, vnp1_face_plus, NULL /* mass_flux times jump of inverse mass density */);
  divergence_free_projector->solve(ksp, pc);

  // extrapolate the solutions from either side so that the projection can be done for ghost-values velocities, as well
  const int niter = 10*MAX(3, (int)ceil((sl_order + 1)*cfl_advection)); // in case someone has the brilliant idea of using a stupidly large advection cfl ("+1" for safety)
  divergence_free_projector->extrapolate_solution_from_either_side_to_the_other(niter);
  divergence_free_projector->project_face_velocities(faces_n);

  return;
}

//void my_p4est_two_phase_flows_t::solve_viscosity_explicit()
//{
//  PetscErrorCode ierr;

//  /* construct the right hand side */
//  compute_backtraced_velocities();
//  my_p4est_interpolation_nodes_t interp_n_minus(ngbd_n), interp_n_plus(ngbd_n);
//  interp_n_minus.set_input(vn_nodes_minus, vn_nodes_minus_xxyyzz, quadratic, P4EST_DIM);
//  interp_n_plus.set_input(vn_nodes_plus, vn_nodes_plus_xxyyzz, quadratic, P4EST_DIM);
//  const double alpha = BDF_alpha();
//  const double beta = BDF_beta();
//  for(u_char dir = 0; dir < P4EST_DIM; ++dir)
//  {
//    for(p4est_locidx_t f_idx = 0; f_idx < faces_n->num_local[dir]; ++f_idx)
//    {
//      double xyz_face[P4EST_DIM]; faces_n->xyz_fr_f(f_idx, dir, xyz_face);
//      const char sgn_face = (interface_manager->phi_at_point(xyz_face) <= 0.0 ? -1 : +1);
//      if(sgn_face < 0)
//        interp_n_minus.add_point(f_idx, xyz_face);
//      else
//        interp_n_plus.add_point(f_idx, xyz_face);
//    }
//    Vec vn_faces;
//    double *vn_faces_p;
//    ierr = VecCreateGhostFaces(p4est_n, faces_n, &vn_faces, dir); CHKERRXX(ierr);
//    ierr = VecGetArray(vn_faces, &vn_faces_p);                    CHKERRXX(ierr);
//    interp_n_minus.interpolate(vn_faces_p, dir);  interp_n_minus.clear();
//    interp_n_plus.interpolate(vn_faces_p, dir);   interp_n_plus.clear();

//    ierr = VecGhostUpdateBegin(vn_faces, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//    ierr = VecGhostUpdateEnd(vn_faces, INSERT_VALUES, SCATTER_FORWARD);   CHKERRXX(ierr);

//    create_vnp1_face_vectors_if_needed();
//    double *vstar_minus_p, *vstar_plus_p;
//    ierr = VecGetArray(vnp1_face_minus[dir], &vstar_minus_p); CHKERRXX(ierr);
//    ierr = VecGetArray(vnp1_face_plus[dir], &vstar_plus_p);   CHKERRXX(ierr);
//    for(p4est_locidx_t f_idx = 0; f_idx < faces_n->num_local[dir]; ++f_idx)
//    {
//      double xyz_face[P4EST_DIM]; faces_n->xyz_fr_f(f_idx, dir, xyz_face);
//      const char sgn_face = (interface_manager->phi_at_point(xyz_face) <= 0.0 ? -1 : +1);
//      double *vstar_p = (sgn_face < 0.0 ? vstar_minus_p : vstar_plus_p);
//      const double& rho_face = (sgn_face < 0.0 ? rho_minus : rho_plus);
//      const std::vector<double>& backtraced_vn    = (sgn_face < 0 ? backtraced_vn_faces_minus[dir] : backtraced_vn_faces_plus[dir]);
//      const std::vector<double>* backtraced_vnm1  = (sl_order == 2 ?  (sgn_face < 0 ? &backtraced_vnm1_faces_minus[dir] : &backtraced_vnm1_faces_plus[dir]) : NULL);
//      if(face_is_dirichlet_wall(f_idx, dir))
//      {
//        vstar_p[f_idx] = bc_velocity[dir].wallValue(xyz_face);
//        continue;
//      }

//      const double viscous_term = div_mu_grad_u_dir(f_idx, dir, vn_faces_p);
//      vstar_p[f_idx] = + (dt_n/(alpha*rho_face))*viscous_term
//          + (1.0 - (beta*dt_n/(alpha*dt_nm1)))*backtraced_vn[f_idx]
//          + (sl_order == 2 ? (beta*dt_n/(alpha*dt_nm1))*(*backtraced_vnm1)[f_idx] : 0.0);
//      if (force_per_unit_mass[dir] != NULL)
//        vstar_p[f_idx] += (dt_n/alpha)*(*force_per_unit_mass[dir])(xyz_face);
//    }
//    ierr = VecRestoreArray(vnp1_face_plus[dir], &vstar_plus_p);   CHKERRXX(ierr);
//    ierr = VecRestoreArray(vnp1_face_minus[dir], &vstar_minus_p); CHKERRXX(ierr);
//    ierr = delete_and_nullify_vector(vn_faces); CHKERRXX(ierr);
//  }
//  return;
//}

//double my_p4est_two_phase_flows_t::div_mu_grad_u_dir(const p4est_locidx_t &face_idx, const u_char &dir, const double *vn_dir_p)
//{
//  const augmented_voronoi_cell my_cell = get_augmented_voronoi_cell(face_idx, dir);
//  double xyz_face[P4EST_DIM]; my_cell.voro.get_center_point(xyz_face);
//  const char sgn_face = (interface_manager->phi_at_point(xyz_face) <= 0.0 ? -1 : +1);
//  p4est_locidx_t quad_idx;
//  p4est_topidx_t tree_idx;
//  faces_n->f2q(face_idx, dir, quad_idx, tree_idx);
//#ifdef P4EST_DEBUG
//  const p4est_quadrant_t *quad = fetch_quad(quad_idx, tree_idx, p4est_n, ghost_n);
//#endif

//  const u_char face_touch = (faces_n->q2f(quad_idx, 2*dir) == face_idx ? 2*dir : 2*dir + 1);
//  P4EST_ASSERT(faces_n->q2f(quad_idx, face_touch) == face_idx);
//  p4est_quadrant_t qm, qp;
//  faces_n->find_quads_touching_face(face_idx, dir, qm, qp);

//  bool wall[P4EST_FACES];
//  for(u_char d = 0; d < P4EST_DIM; ++d)
//  {
//    if(d == dir)
//    {
//      wall[2*d]     = qm.p.piggy3.local_num == -1;
//      wall[2*d + 1] = qp.p.piggy3.local_num == -1;
//    }
//    else
//    {
//      wall[2*d]     = (qm.p.piggy3.local_num == -1 || is_quad_Wall(p4est_n, qm.p.piggy3.which_tree, &qm, 2*d    )) && (qp.p.piggy3.local_num == -1 || is_quad_Wall(p4est_n, qp.p.piggy3.which_tree, &qp, 2*d));
//      wall[2*d + 1] = (qm.p.piggy3.local_num == -1 || is_quad_Wall(p4est_n, qm.p.piggy3.which_tree, &qm, 2*d + 1)) && (qp.p.piggy3.local_num == -1 || is_quad_Wall(p4est_n, qp.p.piggy3.which_tree, &qp, 2*d + 1));
//    }
//  }


//  const vector<ngbdDIMseed> *points;
//#ifndef P4_TO_P8
//  const vector<Point2> *partition;
//  my_cell.voro.get_partition(partition);
//#endif
//  my_cell.voro.get_neighbor_seeds(points);
//  const double volume = my_cell.voro.get_volume();

//  double to_return = 0.0;
//  if(!my_cell.has_neighbor_across || (my_cell.cell_type != parallelepiped_no_wall && my_cell.cell_type != parallelepiped_with_wall))
//  {
//    for (size_t m = 0; m < points->size(); ++m) {
//#ifdef P4_TO_P8
//      const double surface = (*points)[m].s;
//#else
//      size_t k = mod(m - 1, points->size());
//      const double surface = ((*partition)[m] - (*partition)[k]).norm_L2();
//#endif
//      const double distance_to_neighbor = ABSD((*points)[m].p.x - xyz_face[0], (*points)[m].p.y - xyz_face[1], (*points)[m].p.z - xyz_face[2]);
//      const double& mu_this_side = (sgn_face < 0 ? mu_minus : mu_plus);
//      switch ((*points)[m].n) {
//      case WALL_m00:
//      case WALL_p00:
//      case WALL_0m0:
//      case WALL_0p0:
//#ifdef P4_TO_P8
//      case WALL_00m:
//      case WALL_00p:
//#endif
//      {
//        char wall_orientation = -1 - (*points)[m].n;
//        P4EST_ASSERT(wall_orientation >= 0 && wall_orientation < P4EST_FACES);
//        double wall_eval[P4EST_DIM];
//        const double lambda = ((wall_orientation%2 == 1 ? xyz_max[wall_orientation/2] : xyz_min[wall_orientation/2]) - xyz_face[wall_orientation/2])/((*points)[m].p.xyz(wall_orientation/2) - xyz_face[wall_orientation/2]);
//        for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
//          if(dim == wall_orientation/2)
//            wall_eval[dim] = (wall_orientation%2 == 1 ? xyz_max[wall_orientation/2] : xyz_min[wall_orientation/2]); // on the wall of interest
//          else
//            wall_eval[dim] = MIN(MAX(xyz_face[dim] + lambda*((*points)[m].p.xyz(dim) - xyz_face[dim]), xyz_min[dim] + 2.0*EPS*(xyz_max[dim] - xyz_min[dim])), xyz_max[dim] - 2.0*EPS*(xyz_max[dim] - xyz_min[dim])); // make sure it's indeed inside, just to be safe in case the bc object needs that
//        }
//        switch(bc_velocity[dir].wallType(wall_eval))
//        {
//        case DIRICHLET:
//        {
//          if(dir == wall_orientation/2)
//            throw std::runtime_error("my_p4est_two_phase_flows_t::div_mu_grad_u_dir: cannot be called on Dirichlet wall faces...");

//          const bool across = my_cell.has_neighbor_across && (sgn_of_wall_neighbor_of_face(face_idx, dir, wall_orientation, wall_eval) != sgn_face);
//          // WARNING distance_to_neighbor is actually *twice* what we would need here, hence the "0.5*" factors here under!
//          if(!across)
//            to_return += mu_this_side*surface*(bc_velocity[dir].wallValue(wall_eval) /*+ interp_dxyz_hodge(wall_eval)*/ - vn_dir_p[face_idx])/(0.5*distance_to_neighbor);
//          else
//          {
//            const FD_interface_neighbor& face_interface_neighbor = interface_manager->get_face_FD_interface_neighbor_for(face_idx, (*points)[m].n, dir, wall_orientation);
//            const double& mu_across = (sgn_face > 0 ? mu_minus : mu_plus);
//            const bool is_in_positive_domain = (sgn_face > 0);
//            to_return += (wall_orientation%2 == 1 ? +1.0 : -1.0)*surface*face_interface_neighbor.GFM_flux_component(mu_this_side, mu_across, wall_orientation, is_in_positive_domain, vn_dir_p[face_idx], bc_velocity[dir].wallValue(wall_eval), 0.0, 0.0, 0.5*distance_to_neighbor);
//          }
//          break;
//        }
//        case NEUMANN:
//          if(sgn_face != sgn_of_wall_neighbor_of_face(face_idx, dir, wall_orientation, wall_eval))
//            throw std::runtime_error("my_p4est_two_phase_flows_t::div_mu_grad_u_dir: Neumann boundary condition to be imposed on a tranverse wall that lies across the interface, but the face has non-uniform neighbors : this is not implemented yet, sorry...");
//          to_return += mu_this_side*surface*(bc_velocity[dir].wallValue(wall_eval) /*+ (apply_hodge_second_derivative_if_neumann ? 0.0 : 0.0)*/);
//          break;
//        default:
//          throw std::invalid_argument("my_p4est_two_phase_flows_t::div_mu_grad_u_dir: unknown wall type for a cell that is either one-sided or that has a neighbor across the interface but is not a parallelepiped regular --> not handled yet, TO BE DONE IF NEEDED.");
//        }
//        break;
//      }
//      case INTERFACE:
//        throw std::logic_error("my_p4est_two_phase_flows_t::div_mu_grad_u_dir: a Voronoi seed neighbor was marked INTERFACE in a cell that is either one-sided or that has a neighbor across the interface but is not a parallelepiped regular. This must not happen in this solver, have you constructed your Voronoi cell using clip_interface()?");
//        break;
//      default:
//        // this is a regular face so
//        double xyz_face_neighbor[P4EST_DIM]; faces_n->xyz_fr_f((*points)[m].n, dir, xyz_face_neighbor);
//        const bool across = (my_cell.has_neighbor_across && sgn_face != (interface_manager->phi_at_point(xyz_face_neighbor) <= 0.0 ? -1 : +1));
//        if(!across)
//          to_return += mu_this_side*surface*(vn_dir_p[(*points)[m].n] - vn_dir_p[face_idx])/distance_to_neighbor;
//        else
//        {
//          // std::cerr << "This is bad: your grid is messed up here but, hey, I don't want to crash either..." << std::endl;
//          char neighbor_orientation = -1;
//          for (u_char dim = 0; dim < P4EST_DIM; ++dim)
//            if(fabs(xyz_face_neighbor[dim] - xyz_face[dim]) > 0.1*dxyz_smallest_quad[dim])
//              neighbor_orientation = 2*dim + (xyz_face_neighbor[dim] - xyz_face[dim] > 0.0 ? 1 : 0);
//          P4EST_ASSERT(fabs(distance_to_neighbor - dxyz_smallest_quad[neighbor_orientation/2]) < 0.001*dxyz_smallest_quad[neighbor_orientation/2]);
//          const FD_interface_neighbor& face_interface_neighbor = interface_manager->get_face_FD_interface_neighbor_for(face_idx, (*points)[m].n, dir, neighbor_orientation);
//          const double& mu_across = (sgn_face > 0 ? mu_minus : mu_plus);
//          const bool is_in_positive_domain = (sgn_face > 0);
//          to_return += (neighbor_orientation%2 == 1 ? +1.0 : -1.0)*surface*face_interface_neighbor.GFM_flux_component(mu_this_side, mu_across, neighbor_orientation, is_in_positive_domain, vn_dir_p[face_idx], vn_dir_p[(*points)[m].n], 0.0, 0.0, distance_to_neighbor);
//        }
//      }
//    }
//  }
//  else
//  {
//    P4EST_ASSERT(my_cell.cell_type == parallelepiped_no_wall || my_cell.cell_type == parallelepiped_with_wall);
//    if(points->size() != P4EST_FACES)
//      throw std::runtime_error("my_p4est_two_phase_flows_t::div_mu_grad_u_dir: not the expected number of face neighbors for face with an interface neighbor...");
//    P4EST_ASSERT((qm.p.piggy3.local_num == -1 || qm.level == ((const splitting_criteria_t *) p4est_n->user_pointer)->max_lvl)
//                 && (qp.p.piggy3.local_num == -1 || qp.level == ((const splitting_criteria_t *) p4est_n->user_pointer)->max_lvl));

//    P4EST_ASSERT(fabs(volume - (wall[2*dir] || wall[2*dir + 1] ? 0.5 : 1.0)*MULTD(dxyz_smallest_quad[0], dxyz_smallest_quad[1], dxyz_smallest_quad[2])) < 0.001*(wall[2*dir] || wall[2*dir + 1] ? 0.5 : 1.0)*MULTD(dxyz_smallest_quad[0], dxyz_smallest_quad[1], dxyz_smallest_quad[2])); // half the "regular" volume if the face is a NEUMANN wall face

//    for (u_char ff = 0; ff < P4EST_FACES; ++ff)
//    {
//      // get the face index of the direct face/wall neighbor
//      p4est_locidx_t neighbor_face_idx;
//      if(my_cell.cell_type == parallelepiped_no_wall)
//      {
//#ifndef P4_TO_P8
//        neighbor_face_idx = (*points)[face_order_to_counterclock_cycle_order[ff]].n;
//#else
//        neighbor_face_idx = (*points)[ff].n; // already ordered like this, in 3D
//#endif
//      }
//      else // the cell most probably has a wall neighbor, but it's a parallelepiped and it was build by built-in routines (not by hand)
//      {
//        // the voronoi cell was actually constructed by built-in routines, gathering neighbors etc.
//        // but it should still be locally uniform, maybe with wall(s). Let's re-order the neighbors as we need them
//        if(wall[ff])
//          neighbor_face_idx = WALL_idx(ff);
//        else if(ff/2 == dir)
//        {
//          p4est_locidx_t tmp_quad_idx = (ff%2 == 1 ? qp.p.piggy3.local_num : qm.p.piggy3.local_num);
//          P4EST_ASSERT(tmp_quad_idx != -1); // can't be, otherwise wall[ff] would be true...
//          neighbor_face_idx = faces_n->q2f(tmp_quad_idx, ff);
//        }
//        else
//        {
//          set_of_neighboring_quadrants ngbd;
//          ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, ff);
//          P4EST_ASSERT(ngbd.size() == 1 && ngbd.begin()->level == quad->level && ngbd.begin()->level == ((const splitting_criteria_t *) p4est_n->user_pointer)->max_lvl);
//          neighbor_face_idx = faces_n->q2f(ngbd.begin()->p.piggy3.local_num, face_touch);
//        }
//#ifdef P4EST_DEBUG
//        // check if it was indeed in there for consistency!
//        bool found = false;
//        for (size_t k = 0; k < points->size() && !found; ++k)
//          found = ((*points)[k].n != neighbor_face_idx);
//        P4EST_ASSERT(found);
//#endif
//      }


//      // area between the current face and the direct neighbor
//      const double neighbor_area = ((wall[2*dir] || wall[2*dir + 1]) && ff/2 != dir ? 0.5 : 1.0)*dxyz_smallest_quad[(ff/2 + 1)%P4EST_DIM] ONLY3D(*dxyz_smallest_quad[(ff/2 + 2)%P4EST_DIM]);
//      // get the contribution of the direct neighbor to the discretization of the negative laplacian and add it to the matrix
//      if(neighbor_face_idx >= 0)
//      {
//        double xyz_neighbor_face[P4EST_DIM]; faces_n->xyz_fr_f(neighbor_face_idx, dir, xyz_neighbor_face);
//        const char sgn_neighbor_face = (interface_manager->phi_at_point(xyz_neighbor_face) <= 0.0 ? -1 : 1);
//        const bool across = (sgn_face != sgn_neighbor_face);
//        if(across)
//        {
//          const FD_interface_neighbor& face_interface_neighbor = interface_manager->get_face_FD_interface_neighbor_for(face_idx, neighbor_face_idx, dir, ff);
//          const double& mu_this_side  = (sgn_face < 0 ? mu_minus  : mu_plus);
//          const double& mu_across     = (sgn_face < 0 ? mu_plus   : mu_minus);
//          const bool is_in_positive_domain = (sgn_face > 0);
//          to_return += (ff%2 == 1 ? +1.0 : -1.0)*neighbor_area*face_interface_neighbor.GFM_flux_component(mu_this_side, mu_across, ff, is_in_positive_domain, vn_dir_p[face_idx], vn_dir_p[neighbor_face_idx], 0.0, 0.0, dxyz_smallest_quad[ff/2]);
//        }
//        else
//          to_return += ((sgn_face < 0 ? mu_minus : mu_plus)*neighbor_area/dxyz_smallest_quad[ff/2])*(vn_dir_p[neighbor_face_idx] - vn_dir_p[face_idx]);
//      }
//      else
//      {
//        P4EST_ASSERT(-1 - neighbor_face_idx == ff);
//        if(ff/2 == dir) // parallel wall --> the face itself is the wall --> it *cannot* be DIRICHLET
//        {
//          P4EST_ASSERT(wall[ff] && bc_velocity[dir].wallType(xyz_face) == NEUMANN); // the face is a wall face so it MUST be NEUMANN (non-DIRICHLET) boundary condition in that case
//          to_return += neighbor_area*bc_velocity[dir].wallValue(xyz_face);
//        }
//        else // it is a tranverse wall
//        {
//          double xyz_wall[P4EST_DIM] = {DIM(xyz_face[0], xyz_face[1], xyz_face[2])};
//          xyz_wall[ff/2] = (ff%2 == 1 ? xyz_max[ff/2] : xyz_min[ff/2]);
//          const bool across = (sgn_of_wall_neighbor_of_face(face_idx, dir, ff, xyz_wall) != sgn_face);
//          if(across) // the tranverse wall is across the interface
//          {
//            const FD_interface_neighbor& face_interface_neighbor = interface_manager->get_face_FD_interface_neighbor_for(face_idx, neighbor_face_idx, dir, ff);
//            // /!\ WARNING /!\ : theta is relative to 0.5*dxyz_min[ff/2] in this case!
//            const double& mu_this_side  = (sgn_face < 0 ? mu_minus  : mu_plus);
//            const double& mu_across     = (sgn_face < 0 ? mu_plus   : mu_minus);
//            const bool is_in_positive_domain = (sgn_face > 0);

//            switch (bc_velocity[dir].wallType(xyz_wall)) {
//            case DIRICHLET:
//            {
//              to_return += (ff%2 == 1 ? 1.0 : -1.0)*neighbor_area*face_interface_neighbor.GFM_flux_component(mu_this_side, mu_across, ff, is_in_positive_domain, vn_dir_p[face_idx], bc_velocity[dir].wallValue(xyz_wall), 0.0, 0.0, 0.5*dxyz_smallest_quad[ff/2]);
//              break;
//            }
//            case NEUMANN:
//            {
//              to_return += neighbor_area*(bc_velocity[dir].wallValue(xyz_wall) + ((double) sgn_face)*(ff%2 == 1 ? +1.0 : -1.0)*0.0);
//              break;
//            }
//            default:
//              throw std::invalid_argument("my_p4est_two_phase_flows_t::div_mu_grad_u_dir: unknown wall type for a tranverse wall neighbor of a face, across the interface --> not handled yet, TO BE DONE...");
//              break;
//            }
//          }
//          else // the tranverse wall is on the same side of the interface
//          {
//            switch (bc_velocity[dir].wallType(xyz_wall)) {
//            case DIRICHLET:
//              to_return += 2.0*((sgn_face < 0 ? mu_minus : mu_plus)*neighbor_area/dxyz_smallest_quad[ff/2])*(bc_velocity[dir].wallValue(xyz_wall) - vn_dir_p[face_idx]);
//              break;
//            case NEUMANN:
//              to_return += neighbor_area*bc_velocity[dir].wallValue(xyz_wall);
//              break;
//            default:
//              throw std::invalid_argument("my_p4est_two_phase_flows_t::div_mu_grad_u_dir: unknown wall type for a tranverse wall neighbor of a face, on the same side of the interface --> not handled yet, TO BE DONE...");
//              break;
//            }
//          }
//        }
//      }
//    }
//  }
//  return to_return/volume;
//}

void my_p4est_two_phase_flows_t::solve_viscosity()
{
  if(!pressure_guess_is_set)
    solve_for_pressure_guess((cell_jump_solver_to_use == FV ? KSPBCGS : KSPCG), PCHYPRE);
  create_vnp1_face_vectors_if_needed();
  compute_backtraced_velocities();
  viscosity_solver.set_diagonals(BDF_alpha()*rho_minus/dt_n, BDF_alpha()*rho_plus/dt_n);
  viscosity_solver.solve();
  viscosity_solver.extrapolate_face_velocities_across_interface(vnp1_face_minus, vnp1_face_plus);
  return;
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
#ifdef P4_TO_P8
      ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_strong_threshold", "0.5"); CHKERRXX(ierr);
#else
      ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_strong_threshold", "0.25"); CHKERRXX(ierr);
#endif

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
    if(!env->face_is_dirichlet_wall(f_idx, dir))
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
  const double *grad_p_guess_over_rho_minus_dir_p, *grad_p_guess_over_rho_plus_dir_p;
  if(rhs[dir] == NULL){
    ierr = VecCreateNoGhostFaces(env->p4est_n, env->faces_n, &rhs[dir], dir); CHKERRXX(ierr);
  }
  ierr = VecGetArray(rhs[dir], &rhs_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(env->grad_p_guess_over_rho_minus[dir], &grad_p_guess_over_rho_minus_dir_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(env->grad_p_guess_over_rho_plus[dir], &grad_p_guess_over_rho_plus_dir_p); CHKERRXX(ierr);

  for(p4est_locidx_t f_idx = 0; f_idx < env->faces_n->num_local[dir]; ++f_idx)
  {
    p4est_gloidx_t f_idx_g = env->faces_n->global_index(f_idx, dir);

    p4est_locidx_t quad_idx;
    p4est_topidx_t tree_idx;
    env->faces_n->f2q(f_idx, dir, quad_idx, tree_idx);
#ifdef P4EST_DEBUG
    const p4est_quadrant_t *quad = fetch_quad(quad_idx, tree_idx, env->p4est_n, env->ghost_n);
#endif

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

    rhs_p[f_idx] -= rho*(sgn_face < 0 ? grad_p_guess_over_rho_minus_dir_p[f_idx] : grad_p_guess_over_rho_plus_dir_p[f_idx]);
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
                rhs_p[f_idx] += neighbor_area*(env->bc_velocity[dir].wallValue(xyz_wall) + ((double) sgn_face)*(ff%2 == 1 ? +1.0 : -1.0)*0.0);
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
        const double distance_to_neighbor = ABSD((*points)[m].p.x - xyz_face[0], (*points)[m].p.y - xyz_face[1], (*points)[m].p.z - xyz_face[2]);
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
  ierr = VecRestoreArrayRead(env->grad_p_guess_over_rho_minus[dir], &grad_p_guess_over_rho_minus_dir_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(env->grad_p_guess_over_rho_plus[dir], &grad_p_guess_over_rho_plus_dir_p); CHKERRXX(ierr);

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

void my_p4est_two_phase_flows_t::jump_face_solver::extrapolate_face_velocities_across_interface(Vec vnp1_face_minus[P4EST_DIM], Vec vnp1_face_plus[P4EST_DIM], const u_int& n_iterations, const u_char& degree)
{
//  PetscErrorCode ierr;
//  for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
//    ierr = VecCopyGhost(solution[dir], vnp1_face_minus[dir]); CHKERRXX(ierr);
//    ierr = VecCopyGhost(solution[dir], vnp1_face_plus[dir]); CHKERRXX(ierr);
//  }
  P4EST_ASSERT(n_iterations > 0);
  PetscErrorCode ierr;
  // normal derivatives of velocity components
  Vec normal_derivative_of_vnp1_minus[P4EST_DIM] = {DIM(NULL, NULL, NULL)};
  Vec normal_derivative_of_vnp1_plus[P4EST_DIM] = {DIM(NULL, NULL, NULL)};
  double *normal_derivative_of_vnp1_minus_p[P4EST_DIM] = {DIM(NULL, NULL, NULL)};
  double *normal_derivative_of_vnp1_plus_p[P4EST_DIM] = {DIM(NULL, NULL, NULL)};
  const double *normal_derivative_of_vnp1_minus_read_p[P4EST_DIM] = {DIM(NULL, NULL, NULL)};
  const double *normal_derivative_of_vnp1_plus_read_p[P4EST_DIM] = {DIM(NULL, NULL, NULL)};
  double *vnp1_face_minus_p[P4EST_DIM], *vnp1_face_plus_p[P4EST_DIM];
  const double *sharp_solution_p[P4EST_DIM];

  // get pointers, initialize normal derivatives of the P4EST_DIM face-sampled fields, etc.
  for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = VecGetArrayRead(solution[dir],     &sharp_solution_p[dir]);  CHKERRXX(ierr);
    ierr = VecGetArray(vnp1_face_minus[dir],  &vnp1_face_minus_p[dir]); CHKERRXX(ierr);
    ierr = VecGetArray(vnp1_face_plus[dir],   &vnp1_face_plus_p[dir]);  CHKERRXX(ierr);
    if(degree > 0)
    {
      ierr = VecCreateGhostFaces(env->p4est_n, env->faces_n, &normal_derivative_of_vnp1_minus[dir], dir); CHKERRXX(ierr);
      ierr = VecCreateGhostFaces(env->p4est_n, env->faces_n, &normal_derivative_of_vnp1_plus[dir], dir);  CHKERRXX(ierr);
      ierr = VecGetArray(normal_derivative_of_vnp1_minus[dir],  &normal_derivative_of_vnp1_minus_p[dir]); CHKERRXX(ierr);
      ierr = VecGetArray(normal_derivative_of_vnp1_plus[dir],   &normal_derivative_of_vnp1_plus_p[dir]);  CHKERRXX(ierr);
    }
  }
  // INITIALIZE extrapolation
  for (u_char dir = 0; dir < P4EST_DIM; ++dir)
  {
    // local layer faces first
    for (size_t k = 0; k < env->faces_n->get_layer_size(dir); ++k)
      initialize_face_extrapolation(env->faces_n->get_layer_face(dir, k), dir, sharp_solution_p, vnp1_face_minus_p, vnp1_face_plus_p, normal_derivative_of_vnp1_minus_p, normal_derivative_of_vnp1_plus_p, degree);
    // start updates
    if(degree > 0)
    {
      ierr = VecGhostUpdateBegin(normal_derivative_of_vnp1_minus[dir],  INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(normal_derivative_of_vnp1_plus[dir],   INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }
    ierr = VecGhostUpdateBegin(vnp1_face_minus[dir],  INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(vnp1_face_plus[dir],   INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }
  for (u_char dir = 0; dir < P4EST_DIM; ++dir)
  {
    // local inner faces
    for (size_t k = 0; k < env->faces_n->get_local_size(dir); ++k)
      initialize_face_extrapolation(env->faces_n->get_local_face(dir, k), dir, sharp_solution_p, vnp1_face_minus_p, vnp1_face_plus_p, normal_derivative_of_vnp1_minus_p, normal_derivative_of_vnp1_plus_p, degree);
    // complete updates
    if(degree > 0)
    {
      ierr = VecGhostUpdateEnd(normal_derivative_of_vnp1_minus[dir],  INSERT_VALUES, SCATTER_FORWARD);  CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(normal_derivative_of_vnp1_plus[dir],   INSERT_VALUES, SCATTER_FORWARD);  CHKERRXX(ierr);
    }
    ierr = VecGhostUpdateEnd(vnp1_face_minus[dir],  INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(vnp1_face_plus[dir],   INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  /* EXTRAPOLATE normal derivatives of velocity components */
  if(degree > 0)
  {
    for (u_int iter = 0; iter < n_iterations; ++iter) {
      for (u_char dir = 0; dir < P4EST_DIM; ++dir)
      {
        // local layer faces first
        for (size_t k = 0; k < env->faces_n->get_layer_size(dir); ++k)
          extrapolate_normal_derivatives_of_face_velocity_local(env->faces_n->get_layer_face(dir, k), dir, normal_derivative_of_vnp1_minus_p, normal_derivative_of_vnp1_plus_p);
        // start updates
        ierr = VecGhostUpdateBegin(normal_derivative_of_vnp1_minus[dir],  INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        ierr = VecGhostUpdateBegin(normal_derivative_of_vnp1_plus[dir],   INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      }
      for (u_char dir = 0; dir < P4EST_DIM; ++dir)
      {
        // local inner faces
        for (size_t k = 0; k < env->faces_n->get_local_size(dir); ++k)
          extrapolate_normal_derivatives_of_face_velocity_local(env->faces_n->get_local_face(dir, k), dir, normal_derivative_of_vnp1_minus_p, normal_derivative_of_vnp1_plus_p);
        // complete updates
        ierr = VecGhostUpdateEnd(normal_derivative_of_vnp1_minus[dir],  INSERT_VALUES, SCATTER_FORWARD);  CHKERRXX(ierr);
        ierr = VecGhostUpdateEnd(normal_derivative_of_vnp1_plus[dir],   INSERT_VALUES, SCATTER_FORWARD);  CHKERRXX(ierr);
      }
    }
    // restore read-write pointer, get read-only pointers
    for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
      ierr = VecRestoreArray(normal_derivative_of_vnp1_minus[dir],  &normal_derivative_of_vnp1_minus_p[dir]); CHKERRXX(ierr);
      ierr = VecRestoreArray(normal_derivative_of_vnp1_plus[dir],   &normal_derivative_of_vnp1_plus_p[dir]);  CHKERRXX(ierr);
      ierr = VecGetArrayRead(normal_derivative_of_vnp1_minus[dir],  &normal_derivative_of_vnp1_minus_read_p[dir]); CHKERRXX(ierr);
      ierr = VecGetArrayRead(normal_derivative_of_vnp1_plus[dir],   &normal_derivative_of_vnp1_plus_read_p[dir]);  CHKERRXX(ierr);
    }
  }

  /* EXTRAPOLATE velocity components */
  for (u_int iter = 0; iter < n_iterations; ++iter) {
    for (u_char dir = 0; dir < P4EST_DIM; ++dir)
    {
      // local layer faces first
      for (size_t k = 0; k < env->faces_n->get_layer_size(dir); ++k)
        face_velocity_extrapolation_local(env->faces_n->get_layer_face(dir, k), dir, sharp_solution_p, normal_derivative_of_vnp1_minus_read_p, normal_derivative_of_vnp1_plus_read_p, vnp1_face_minus_p, vnp1_face_plus_p);
      // start updates
      ierr = VecGhostUpdateBegin(vnp1_face_minus[dir],  INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(vnp1_face_plus[dir],   INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }
    for (u_char dir = 0; dir < P4EST_DIM; ++dir)
    {
      // local inner faces
      for (size_t k = 0; k < env->faces_n->get_local_size(dir); ++k)
        face_velocity_extrapolation_local(env->faces_n->get_local_face(dir, k), dir, sharp_solution_p, normal_derivative_of_vnp1_minus_read_p, normal_derivative_of_vnp1_plus_read_p, vnp1_face_minus_p, vnp1_face_plus_p);
      // complete updates
      ierr = VecGhostUpdateEnd(vnp1_face_minus[dir],  INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(vnp1_face_plus[dir],   INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }
  }

  // restore data pointers and delete locally created vectors
  for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
    if(degree > 0)
    {
      ierr = VecRestoreArrayRead(normal_derivative_of_vnp1_plus[dir],   &normal_derivative_of_vnp1_plus_read_p[dir]);  CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(normal_derivative_of_vnp1_minus[dir],  &normal_derivative_of_vnp1_minus_read_p[dir]); CHKERRXX(ierr);
      ierr = delete_and_nullify_vector(normal_derivative_of_vnp1_plus[dir]);  CHKERRXX(ierr);
      ierr = delete_and_nullify_vector(normal_derivative_of_vnp1_minus[dir]); CHKERRXX(ierr);
    }
    ierr = VecRestoreArray(vnp1_face_plus[dir],   &vnp1_face_plus_p[dir]);  CHKERRXX(ierr);
    ierr = VecRestoreArray(vnp1_face_minus[dir],  &vnp1_face_minus_p[dir]); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(solution[dir],     &sharp_solution_p[dir]);  CHKERRXX(ierr);
  }
  return;
}

void my_p4est_two_phase_flows_t::jump_face_solver::initialize_face_extrapolation(const p4est_locidx_t &f_idx, const u_char &dir, const double* sharp_solution_p[P4EST_DIM],
                                                                                 double *vnp1_face_minus_p[P4EST_DIM], double *vnp1_face_plus_p[P4EST_DIM],
                                                                                 double *normal_derivative_of_vnp1_face_minus_p[P4EST_DIM], double *normal_derivative_of_vnp1_face_plus_p[P4EST_DIM], const u_char& degree)
{
  double xyz_face[P4EST_DIM]; env->faces_n->xyz_fr_f(f_idx, dir, xyz_face);
  const char sgn_face = env->sgn_of_face(f_idx, dir, xyz_face);
  double *normal_derivative_of_vnp1_dir_this_side_p   = (sgn_face < 0 ? normal_derivative_of_vnp1_face_minus_p[dir] : normal_derivative_of_vnp1_face_plus_p[dir]);
  double *normal_derivative_of_vnp1_dir_other_side_p  = (sgn_face < 0 ? normal_derivative_of_vnp1_face_plus_p[dir]  : normal_derivative_of_vnp1_face_minus_p[dir]);
  double *vnp1_dir_this_side_p                        = (sgn_face < 0 ? vnp1_face_minus_p[dir]  : vnp1_face_plus_p[dir]);
  double *vnp1_dir_other_side_p                       = (sgn_face < 0 ? vnp1_face_plus_p[dir]   : vnp1_face_minus_p[dir]);
  double oriented_normal[P4EST_DIM];
  if(degree > 0)
    env->interface_manager->normal_vector_at_point(xyz_face, oriented_normal, (sgn_face < 0 ? +1.0 : -1.0));
  const uniform_face_ngbd* face_ngbd;
  if(env->faces_n->found_uniform_face_neighborhood(f_idx, dir, face_ngbd))
  {
    if(degree > 0)
    {
      double n_dot_grad_u_dir = 0.0;
      for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
        const p4est_locidx_t p_neighbor_face = face_ngbd->neighbor_face_idx[2*dim + 1];
        const p4est_locidx_t m_neighbor_face = face_ngbd->neighbor_face_idx[2*dim];
        P4EST_ASSERT((p_neighbor_face >= 0 || -1 - p_neighbor_face == 2*dim + 1) && (m_neighbor_face >= 0 || -1 - m_neighbor_face == 2*dim));
        double xyz_p_wall[P4EST_DIM] = {DIM(xyz_face[0], xyz_face[1], xyz_face[2])}; xyz_p_wall[dim] = env->xyz_max[dim];
        double xyz_m_wall[P4EST_DIM] = {DIM(xyz_face[0], xyz_face[1], xyz_face[2])}; xyz_m_wall[dim] = env->xyz_min[dim];
        const char sgn_p_neighbor = (p_neighbor_face >= 0 ? env->sgn_of_face(p_neighbor_face, dir) : env->sgn_of_wall_neighbor_of_face(f_idx, dir, 2*dim + 1, xyz_p_wall));
        const char sgn_m_neighbor = (m_neighbor_face >= 0 ? env->sgn_of_face(m_neighbor_face, dir) : env->sgn_of_wall_neighbor_of_face(f_idx, dir, 2*dim,     xyz_m_wall));
        if((m_neighbor_face < 0 && env->bc_velocity[dir].wallType(xyz_m_wall) == NEUMANN) || (p_neighbor_face < 0 && env->bc_velocity[dir].wallType(xyz_p_wall) == NEUMANN))
          throw std::runtime_error("my_p4est_two_phase_flows_t::jump_face_solver::initialize_face_extrapolation : not handling neumann velocity walls, yet");
        const double& mu_this_side  = (sgn_face < 0 ? env->mu_minus : env->mu_plus);
        const double& mu_across     = (sgn_face < 0 ? env->mu_plus  : env->mu_minus);
        const bool& is_in_positive_domain = (sgn_face > 0);
        double finite_difference = 0.0;
        double differentiation_arm = 0.0;
        if(sgn_p_neighbor == sgn_face)
        {
          finite_difference   += (p_neighbor_face >= 0 ? sharp_solution_p[dir][p_neighbor_face] : env->bc_velocity[dir].wallValue(xyz_p_wall));
          differentiation_arm += (p_neighbor_face >= 0 ? 1.0 : 0.5)*env->dxyz_smallest_quad[dim];
        }
        else
        {
          const double solution_across = (p_neighbor_face >= 0 ? sharp_solution_p[dir][p_neighbor_face] : env->bc_velocity[dir].wallValue(xyz_p_wall));
          const FD_interface_neighbor& face_interface_neighbor = env->interface_manager->get_face_FD_interface_neighbor_for(f_idx, p_neighbor_face, dir, 2*dim + 1);
          // fetch the interface-defined value (on the same side)
          finite_difference   += face_interface_neighbor.GFM_interface_value(mu_this_side, mu_across, 2*dim + 1, is_in_positive_domain, is_in_positive_domain, sharp_solution_p[dir][f_idx], solution_across, 0.0, 0.0, (p_neighbor_face >= 0 ? 1.0 : 0.5)*env->dxyz_smallest_quad[dim]);
          differentiation_arm += face_interface_neighbor.theta*(p_neighbor_face >= 0 ? 1.0 : 0.5)*env->dxyz_smallest_quad[dim];
        }

        if(sgn_m_neighbor == sgn_face)
        {
          finite_difference   -= (m_neighbor_face >= 0 ? sharp_solution_p[dir][m_neighbor_face] : env->bc_velocity[dir].wallValue(xyz_m_wall));
          differentiation_arm += (m_neighbor_face >= 0 ? 1.0 : 0.5)*env->dxyz_smallest_quad[dim];
        }
        else
        {
          const double solution_across = (m_neighbor_face >= 0 ? sharp_solution_p[dir][m_neighbor_face] : env->bc_velocity[dir].wallValue(xyz_m_wall));
          // fetch the interface-defined value (on the same side)
          const FD_interface_neighbor& face_interface_neighbor = env->interface_manager->get_face_FD_interface_neighbor_for(f_idx, m_neighbor_face, dir, 2*dim);
          finite_difference   -= face_interface_neighbor.GFM_interface_value(mu_this_side, mu_across, 2*dim, is_in_positive_domain, is_in_positive_domain, sharp_solution_p[dir][f_idx], solution_across, 0.0, 0.0, (m_neighbor_face >= 0 ? 1.0 : 0.5)*env->dxyz_smallest_quad[dim]);
          differentiation_arm += face_interface_neighbor.theta*(m_neighbor_face >= 0 ? 1.0 : 0.5)*env->dxyz_smallest_quad[dim];
        }

        if(differentiation_arm > pow(2.0, -env->interface_manager->get_max_level_computational_grid())*env->dxyz_smallest_quad[dim])
          n_dot_grad_u_dir += oriented_normal[dim]*finite_difference/differentiation_arm;
        else // very small differentation arm --> safer to use values across and the jump in flux component
        {
          if(p_neighbor_face < 0 || m_neighbor_face < 0)
            throw std::runtime_error("my_p4est_two_phase_flows_t::jump_face_solver::initialize_face_extrapolation : this is a nightmare, please kill me now...");
          const double derivative = (mu_across*(sharp_solution_p[p_neighbor_face] - sharp_solution_p[m_neighbor_face])/(2.0*env->dxyz_smallest_quad[dim]) + sgn_face*0.0)/mu_this_side;
          n_dot_grad_u_dir += oriented_normal[dim]*derivative;
        }
      }

      normal_derivative_of_vnp1_dir_this_side_p[f_idx] = n_dot_grad_u_dir;
      normal_derivative_of_vnp1_dir_other_side_p[f_idx] = 0.0; // to be calculated later on in actual extrapolation
    }
    vnp1_dir_this_side_p[f_idx]   = sharp_solution_p[dir][f_idx];
    vnp1_dir_other_side_p[f_idx]  = sharp_solution_p[dir][f_idx] + (sgn_face < 0 ? 1.0 : -1.0)*0.0; // rough initialization
  }
  else
  {
    if(degree > 0)
    {
      normal_derivative_of_vnp1_dir_this_side_p[f_idx] = 0.0; // ---> I need HELP, please...
      normal_derivative_of_vnp1_dir_other_side_p[f_idx] = 0.0; // to be calculated later on in actual extrapolation
    }
    vnp1_dir_this_side_p[f_idx]   = sharp_solution_p[dir][f_idx];
    vnp1_dir_other_side_p[f_idx]  = sharp_solution_p[dir][f_idx] + (sgn_face < 0 ? 1.0 : -1.0)*0.0; // rough initialization
  }
  return;
}

void my_p4est_two_phase_flows_t::jump_face_solver::extrapolate_normal_derivatives_of_face_velocity_local(const p4est_locidx_t &f_idx, const u_char &dir,
                                                                                                         double *normal_derivative_of_vnp1_minus_p[P4EST_DIM], double *normal_derivative_of_vnp1_plus_p[P4EST_DIM])
{
  double xyz_face[P4EST_DIM], oriented_normal[P4EST_DIM];
  env->faces_n->xyz_fr_f(f_idx, dir, xyz_face);
  // if the face is in negative domain, we extend the + field and the normal is reverted
  const char sgn_face = env->sgn_of_face(f_idx, dir, xyz_face);
  env->interface_manager->normal_vector_at_point(xyz_face, oriented_normal, (double) sgn_face);
  double *normal_derivative_of_field_p = (sgn_face < 0 ? normal_derivative_of_vnp1_plus_p[dir] : normal_derivative_of_vnp1_minus_p[dir]);
  double n_dot_grad_normal_derivative_of_field = 0.0;
  for (u_char der = 0; der < P4EST_DIM; ++der)
  {
    const u_char local_oriented_der  = (oriented_normal[der] > 0.0  ? 2*der : 2*der + 1);
    p4est_locidx_t neighbor_face_idx;
    if(!env->faces_n->found_finest_face_neighbor(f_idx, dir, local_oriented_der, neighbor_face_idx))
      return; // one of the neighbor is not a uniform fine neigbor, forget about this one for now and return
    P4EST_ASSERT((0 <= neighbor_face_idx || local_oriented_der == -1 - neighbor_face_idx) && neighbor_face_idx < env->faces_n->num_local[dir] + env->faces_n->num_ghost[dir]);
    // normal derivatives: always on the grid!
    if(neighbor_face_idx >= 0.0) //
      n_dot_grad_normal_derivative_of_field += fabs(oriented_normal[der])*(normal_derivative_of_field_p[f_idx] - normal_derivative_of_field_p[neighbor_face_idx])/env->dxyz_smallest_quad[der];
    else
    {
      std::ostringstream oss;
      oss << "The face is located at (" << xyz_face[0] << ", " << xyz_face[1] ONLY3D(<< ", " << xyz_face[2] ) << "), ";
      oss << "the oriented normal is (" << oriented_normal[0] << ", " << oriented_normal[1] ONLY3D(<< ", " << oriented_normal[2] ) << ", ";
      oss << "and the problem occurs for component " << (int) der << std::endl;
      throw std::runtime_error("my_p4est_two_phase_flows_t::jump_face_solver::extrapolate_normal_derivatives_of_face_velocity_local : wall neighbor of face, not handled yet.\n" + oss.str());
    }
  }
  const double dtau = MIN(DIM(env->dxyz_smallest_quad[0], env->dxyz_smallest_quad[1], env->dxyz_smallest_quad[2]))/((double) P4EST_DIM); // as in other extensions (at nodes)
  normal_derivative_of_field_p[f_idx] -= dtau*n_dot_grad_normal_derivative_of_field;
  return;
}

void my_p4est_two_phase_flows_t::jump_face_solver::face_velocity_extrapolation_local(const p4est_locidx_t &f_idx, const u_char &dir, const double* sharp_solution_p[P4EST_DIM],
                                                                                     const double *normal_derivative_of_vnp1_face_minus_p[P4EST_DIM], const double *normal_derivative_of_vnp1_face_plus_p[P4EST_DIM],
                                                                                     double *vnp1_face_minus_p[P4EST_DIM], double *vnp1_face_plus_p[P4EST_DIM])
{
  double xyz_face[P4EST_DIM], oriented_normal[P4EST_DIM];
  env->faces_n->xyz_fr_f(f_idx, dir, xyz_face);
  // if the face is in negative domain, we extend the + field and the normal is reverted
  const char sgn_face = env->sgn_of_face(f_idx, dir, xyz_face);
  env->interface_manager->normal_vector_at_point(xyz_face, oriented_normal, (double) sgn_face);
  const double& mu_this_side  = (sgn_face < 0 ? env->mu_minus : env->mu_plus);
  const double& mu_across     = (sgn_face < 0 ? env->mu_plus  : env->mu_minus);
  const bool& is_in_positive_domain = (sgn_face > 0);
  double dtau = MIN(DIM(env->dxyz_smallest_quad[0], env->dxyz_smallest_quad[1], env->dxyz_smallest_quad[2]));
  bool too_close_flag = false;
  // if the face is in negative domain, we extend the + field and the normal is reverted
  double* field_p  = (sgn_face < 0 ? vnp1_face_plus_p[dir] : vnp1_face_minus_p[dir]);
  const double* normal_derivative_of_field_p  = (sgn_face < 0 ? normal_derivative_of_vnp1_face_plus_p[dir] : normal_derivative_of_vnp1_face_minus_p[dir]);
  double n_dot_grad_field = 0.0;
  for (u_char der = 0; !too_close_flag && der < P4EST_DIM; ++der)
  {
    const u_char local_oriented_der = (oriented_normal[der] > 0.0  ? 2*der : 2*der + 1);
    p4est_locidx_t neighbor_face_idx;
    if(!env->faces_n->found_finest_face_neighbor(f_idx, dir, local_oriented_der, neighbor_face_idx))
      return; // one of the neighbor is not a uniform fine neigbor, forget about this one for now and return
    P4EST_ASSERT((0 <= neighbor_face_idx || local_oriented_der == -1 - neighbor_face_idx)  && neighbor_face_idx < env->faces_n->num_local[dir] + env->faces_n->num_ghost[dir]);
    // The field may be subresolved
    double xyz_wall_neighbor[P4EST_DIM] = {DIM(xyz_face[0], xyz_face[1], xyz_face[2])}; xyz_wall_neighbor[local_oriented_der/2] = (local_oriented_der%2 == 1 ? env->xyz_max[local_oriented_der/2] : env->xyz_min[local_oriented_der/2]);
    const char sgn_neighbor = (neighbor_face_idx >= 0 ? env->sgn_of_face(neighbor_face_idx, dir) : env->sgn_of_wall_neighbor_of_face(f_idx, dir, local_oriented_der, xyz_wall_neighbor));
    if(neighbor_face_idx < 0 && env->bc_velocity[dir].wallType(xyz_wall_neighbor) == NEUMANN)
      throw std::runtime_error("my_p4est_two_phase_flows_t::jump_face_solver::initialize_face_extrapolation : not handling neumann velocity walls, yet");

    if(sgn_neighbor != sgn_face)
    {
      const FD_interface_neighbor face_interface_neighbor = env->interface_manager->get_face_FD_interface_neighbor_for(f_idx, neighbor_face_idx, dir, local_oriented_der);
      const double solution_across = (neighbor_face_idx >= 0 ? sharp_solution_p[dir][neighbor_face_idx] : env->bc_velocity[dir].wallValue(xyz_wall_neighbor));
      // fetch the interface-defined value (!!! on the OTHER side --> you are extrapolation, here !!!!)
      const double interface_value = face_interface_neighbor.GFM_interface_value(mu_this_side, mu_across, local_oriented_der, is_in_positive_domain, !is_in_positive_domain, sharp_solution_p[dir][f_idx], solution_across, 0.0, 0.0, (neighbor_face_idx >= 0 ? 1.0 : 0.5)*env->dxyz_smallest_quad[der]);
      if(face_interface_neighbor.theta < 0.1*pow(2.0, -env->interface_manager->get_max_level_computational_grid()))
      {
        field_p[f_idx] = interface_value;
        too_close_flag = true;
      }
      else
      {
        n_dot_grad_field += fabs(oriented_normal[der])*(field_p[f_idx] - interface_value)/(face_interface_neighbor.theta*(neighbor_face_idx >= 0 ? 1.0 : 0.5)*env->dxyz_smallest_quad[der]);
        dtau = MIN(dtau, face_interface_neighbor.theta*(neighbor_face_idx >= 0 ? 1.0 : 0.5)*env->dxyz_smallest_quad[der]);
      }
    }
    else
    {
      const double neighbor_value = (neighbor_face_idx >= 0 ? field_p[neighbor_face_idx] : env->bc_velocity[dir].wallValue(xyz_wall_neighbor));
      n_dot_grad_field += fabs(oriented_normal[der])*(field_p[f_idx] - neighbor_value)/(env->dxyz_smallest_quad[der]);
    }
  }
  if(!too_close_flag)
  {
    dtau /= ((double) P4EST_DIM);
    field_p[f_idx] -= dtau*(n_dot_grad_field - (normal_derivative_of_field_p != NULL ? normal_derivative_of_field_p[f_idx] : 0.0));
  }
  return;
}

void my_p4est_two_phase_flows_t::compute_velocities_at_nodes()
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
    interpolate_velocities_at_node(ngbd_n->get_layer_node(i), vnp1_nodes_minus_p, vnp1_nodes_plus_p, vnp1_face_minus_p, vnp1_face_plus_p);
  ierr = VecGhostUpdateBegin(vnp1_nodes_minus,  INSERT_VALUES, SCATTER_FORWARD);  CHKERRXX(ierr);
  ierr = VecGhostUpdateBegin(vnp1_nodes_plus,   INSERT_VALUES, SCATTER_FORWARD);  CHKERRXX(ierr);
  /* interpolate vnp1 from faces to nodes */
  for(size_t i = 0; i < ngbd_n->get_local_size(); ++i)
    interpolate_velocities_at_node(ngbd_n->get_local_node(i), vnp1_nodes_minus_p, vnp1_nodes_plus_p, vnp1_face_minus_p, vnp1_face_plus_p);

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

//  TVD_extrapolation_of_np1_node_velocities();
  compute_vorticities();
}

void my_p4est_two_phase_flows_t::interpolate_velocities_at_node(const p4est_locidx_t &node_idx, double *vnp1_nodes_minus_p, double *vnp1_nodes_plus_p,
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

        const double w = MAX(min_w, 1./MAX(inv_max_w, ABSD(xyz_t[0], xyz_t[1], xyz_t[2])));

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

  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    if(ISNAN(vnp1_nodes_minus_p[P4EST_DIM*node_idx + dim]))
    {
      P4EST_ASSERT(!ISNAN(vnp1_nodes_plus_p[P4EST_DIM*node_idx + dim]));
      vnp1_nodes_minus_p[P4EST_DIM*node_idx + dim] = vnp1_nodes_plus_p[P4EST_DIM*node_idx + dim] - 0.0;
    }
    else if(ISNAN(vnp1_nodes_plus_p[P4EST_DIM*node_idx + dim]))
    {
      P4EST_ASSERT(!ISNAN(vnp1_nodes_minus_p[P4EST_DIM*node_idx + dim]));
      vnp1_nodes_plus_p[P4EST_DIM*node_idx + dim] = vnp1_nodes_minus_p[P4EST_DIM*node_idx + dim] + 0.0;
    }
  }


  if (!ISNAN(magnitude_velocity_minus) && interface_manager->phi_at_point(xyz_node) <= (sl_order + 1)*cfl_advection*smallest_diagonal) // "+ 1" for safety
    max_L2_norm_velocity_minus = MAX(max_L2_norm_velocity_minus, sqrt(magnitude_velocity_minus));
  if (!ISNAN(magnitude_velocity_plus) && interface_manager->phi_at_point(xyz_node) >= -(sl_order + 1)*cfl_advection*smallest_diagonal) // "+ 1" for safety
    max_L2_norm_velocity_plus = MAX(max_L2_norm_velocity_plus, sqrt(magnitude_velocity_plus));
  return;
}

void my_p4est_two_phase_flows_t::TVD_extrapolation_of_np1_node_velocities(const u_int& niterations, const u_char& order)
{
  PetscErrorCode ierr;
  Vec vnp1_nodes_minus_no_block[P4EST_DIM], vnp1_nodes_plus_no_block[P4EST_DIM], normal_vector[P4EST_DIM];
  double *vnp1_nodes_minus_no_block_p[P4EST_DIM], *vnp1_nodes_plus_no_block_p[P4EST_DIM], *normal_vector_p[P4EST_DIM];
  double *phi_on_computational_nodes_p = NULL;
  bool phi_on_computational_nodes_locally_created = false;
  P4EST_ASSERT(interface_manager->subcell_resolution() != 0 || phi_on_computational_nodes != NULL);
  if(phi_on_computational_nodes == NULL)
  {
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &phi_on_computational_nodes); CHKERRXX(ierr);
    ierr = VecGetArray(phi_on_computational_nodes, &phi_on_computational_nodes_p); CHKERRXX(ierr);
    phi_on_computational_nodes_locally_created = true;
  }
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &vnp1_nodes_minus_no_block[dim]);  CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &vnp1_nodes_plus_no_block[dim]);   CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &normal_vector[dim]);              CHKERRXX(ierr);
    ierr = VecGetArray(vnp1_nodes_minus_no_block[dim], &vnp1_nodes_minus_no_block_p[dim]);  CHKERRXX(ierr);
    ierr = VecGetArray(vnp1_nodes_plus_no_block[dim], &vnp1_nodes_plus_no_block_p[dim]);    CHKERRXX(ierr);
    ierr = VecGetArray(normal_vector[dim], &normal_vector_p[dim]); CHKERRXX(ierr);
  }
  double *vnp1_nodes_minus_p, *vnp1_nodes_plus_p;
  const double *grad_phi_p = NULL;
  if(interface_manager->subcell_resolution() == 0){
    ierr= VecGetArrayRead(interface_manager->get_grad_phi(), &grad_phi_p); CHKERRXX(ierr); }
  double xyz_node[P4EST_DIM];
  ierr = VecGetArray(vnp1_nodes_minus, &vnp1_nodes_minus_p); CHKERRXX(ierr);
  ierr = VecGetArray(vnp1_nodes_plus, &vnp1_nodes_plus_p); CHKERRXX(ierr);
  for (size_t node_idx = 0; node_idx < nodes_n->indep_nodes.elem_count; ++node_idx) {
    for (u_char dim = 0; dim < P4EST_DIM; ++dim)
    {
      vnp1_nodes_minus_no_block_p[dim][node_idx] = vnp1_nodes_minus_p[P4EST_DIM*node_idx + dim];
      vnp1_nodes_plus_no_block_p[dim][node_idx] = vnp1_nodes_plus_p[P4EST_DIM*node_idx + dim];
    }
    if(node_idx < (size_t) nodes_n->num_owned_indeps)
    {
      if(interface_manager->subcell_resolution() > 0)
      {
        P4EST_ASSERT(grad_phi_p == NULL);
        node_xyz_fr_n(node_idx, p4est_n, nodes_n, xyz_node);
        if(phi_on_computational_nodes_p != NULL)
          phi_on_computational_nodes_p[node_idx] = interface_manager->phi_at_point(xyz_node);
        double local_normal[P4EST_DIM];
        interface_manager->normal_vector_at_point(xyz_node, local_normal);
        for (u_char dim = 0; dim < P4EST_DIM; ++dim)
          normal_vector_p[dim][node_idx] = local_normal[dim];
      }
      else
      {
        P4EST_ASSERT(phi_on_computational_nodes_p == NULL && grad_phi_p != NULL);
        const double magnitude = ABSD(grad_phi_p[P4EST_DIM*node_idx], grad_phi_p[P4EST_DIM*node_idx + 1], grad_phi_p[P4EST_DIM*node_idx + 2]);
        for (u_char dim = 0; dim < P4EST_DIM; ++dim)
          normal_vector_p[dim][node_idx] = (magnitude > EPS ? grad_phi_p[P4EST_DIM*node_idx + dim]/magnitude : 0.0);
      }
    }
  }
  if(interface_manager->subcell_resolution() > 0){
    ierr = VecGhostUpdateBegin(phi_on_computational_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(phi_on_computational_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecGhostUpdateBegin(normal_vector[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(normal_vector[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  if(interface_manager->subcell_resolution() == 0){
    ierr = VecRestoreArrayRead(interface_manager->get_grad_phi(), &grad_phi_p); CHKERRXX(ierr); }
  ierr = VecRestoreArray(vnp1_nodes_minus, &vnp1_nodes_minus_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(vnp1_nodes_plus, &vnp1_nodes_plus_p); CHKERRXX(ierr);
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecRestoreArray(normal_vector[dim], &normal_vector_p[dim]); CHKERRXX(ierr);
    ierr = VecRestoreArray(vnp1_nodes_plus_no_block[dim], &vnp1_nodes_plus_no_block_p[dim]); CHKERRXX(ierr);
    ierr = VecRestoreArray(vnp1_nodes_minus_no_block[dim], &vnp1_nodes_minus_no_block_p[dim]); CHKERRXX(ierr);
  }
  if(phi_on_computational_nodes_p != NULL){
    ierr = VecRestoreArray(phi_on_computational_nodes, &phi_on_computational_nodes_p); CHKERRXX(ierr); }

  my_p4est_level_set_t ls_nodes(ngbd_n);

  for (char sgn = -1; sgn <= 1; sgn += 2) {
    if(sgn > 0)
    {
      ierr = VecScaleGhost(phi_on_computational_nodes, -1.0); CHKERRXX(ierr);
      for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
        ierr = VecScaleGhost(normal_vector[dim], -1.0); CHKERRXX(ierr); }
    }
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      Vec node_velocity_component_to_extrapolate = (sgn < 0 ? vnp1_nodes_minus_no_block[dim] : vnp1_nodes_plus_no_block[dim]);
      ls_nodes.extend_Over_Interface_TVD(phi_on_computational_nodes, node_velocity_component_to_extrapolate, niterations, order, 0.0, -DBL_MAX, +DBL_MAX, DBL_MAX, normal_vector);
    }
  }

  ierr = VecScaleGhost(phi_on_computational_nodes, -1.0); CHKERRXX(ierr);
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecScaleGhost(normal_vector[dim], -1.0); CHKERRXX(ierr); }

  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecGetArray(vnp1_nodes_minus_no_block[dim], &vnp1_nodes_minus_no_block_p[dim]); CHKERRXX(ierr);
    ierr = VecGetArray(vnp1_nodes_plus_no_block[dim], &vnp1_nodes_plus_no_block_p[dim]); CHKERRXX(ierr);
  }
  ierr = VecGetArray(vnp1_nodes_minus, &vnp1_nodes_minus_p); CHKERRXX(ierr);
  ierr = VecGetArray(vnp1_nodes_plus, &vnp1_nodes_plus_p); CHKERRXX(ierr);
  for (size_t node_idx = 0; node_idx < nodes_n->indep_nodes.elem_count; ++node_idx) {
    for (u_char dim = 0; dim < P4EST_DIM; ++dim)
    {
      vnp1_nodes_minus_p[P4EST_DIM*node_idx + dim] = vnp1_nodes_minus_no_block_p[dim][node_idx];
      vnp1_nodes_plus_p[P4EST_DIM*node_idx + dim] = vnp1_nodes_plus_no_block_p[dim][node_idx];
    }
  }
  ierr = VecRestoreArray(vnp1_nodes_minus, &vnp1_nodes_minus_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(vnp1_nodes_plus, &vnp1_nodes_plus_p); CHKERRXX(ierr);
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecRestoreArray(vnp1_nodes_minus_no_block[dim], &vnp1_nodes_minus_no_block_p[dim]); CHKERRXX(ierr);
    ierr = VecRestoreArray(vnp1_nodes_plus_no_block[dim], &vnp1_nodes_plus_no_block_p[dim]); CHKERRXX(ierr);
    ierr = VecRestoreArray(normal_vector[dim], &normal_vector_p[dim]); CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(vnp1_nodes_minus_no_block[dim]); CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(vnp1_nodes_plus_no_block[dim]); CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(normal_vector[dim]); CHKERRXX(ierr);
  }
  if(phi_on_computational_nodes_locally_created){
    ierr = delete_and_nullify_vector(phi_on_computational_nodes); CHKERRXX(ierr); }

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
          + SQR(velocity_gradient[P4EST_DIM*dir::z + dir::y] - velocity_gradient[P4EST_DIM*dir::y + dir::z]));
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
      vorticity_magnitude_p[node_idx] = sqrt(SQR(velocity_gradient[P4EST_DIM*dir::y + dir::x] - velocity_gradient[P4EST_DIM*dir::x + dir::y])
          + SQR(velocity_gradient[P4EST_DIM*dir::x + dir::z] - velocity_gradient[P4EST_DIM*dir::z + dir::x])
          + SQR(velocity_gradient[P4EST_DIM*dir::z + dir::y] - velocity_gradient[P4EST_DIM*dir::y + dir::z]));
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

void my_p4est_two_phase_flows_t::save_vtk(const std::string& vtk_directory, const int& index) const
{
  PetscErrorCode ierr;
  std::vector<const double*> node_scalar_data;        std::vector<std::string> node_scalar_names;
  std::vector<const double*> node_vector_block_data;  std::vector<std::string> node_vector_block_names;
  std::vector<const double*> cell_scalar_data;        std::vector<std::string> cell_scalar_names;
  const double *phi_on_computational_nodes_p, *projection_variable_p, *vnp1_nodes_plus_p, *vnp1_nodes_minus_p;
  const double *phi_on_fine_nodes_p, *curvature_p, *grad_phi_p;
  Vec projection_variable = divergence_free_projector->get_solution();
  if(phi_on_computational_nodes != NULL){
    ierr = VecGetArrayRead(phi_on_computational_nodes, &phi_on_computational_nodes_p); CHKERRXX(ierr);
    node_scalar_data.push_back(phi_on_computational_nodes_p);
    node_scalar_names.push_back("phi");
  }
  if(projection_variable != NULL){
    ierr = VecGetArrayRead(projection_variable, &projection_variable_p); CHKERRXX(ierr);
    cell_scalar_data.push_back(projection_variable_p);
    cell_scalar_names.push_back("projection variable");
  }
  if(vnp1_nodes_minus != NULL){
    ierr = VecGetArrayRead(vnp1_nodes_minus, &vnp1_nodes_minus_p); CHKERRXX(ierr);
    node_vector_block_data.push_back(vnp1_nodes_minus_p);
    node_vector_block_names.push_back("vnp1 minus");
  }
  if(vnp1_nodes_plus != NULL){
    ierr = VecGetArrayRead(vnp1_nodes_plus, &vnp1_nodes_plus_p); CHKERRXX(ierr);
    node_vector_block_data.push_back(vnp1_nodes_plus_p);
    node_vector_block_names.push_back("vnp1 plus");
  }
  if(interface_manager->subcell_resolution() == 0)
  {
    ierr = VecGetArrayRead(interface_manager->get_curvature(), &curvature_p); CHKERRXX(ierr);
    node_scalar_data.push_back(curvature_p);
    node_scalar_names.push_back("curvature");
    ierr = VecGetArrayRead(interface_manager->get_grad_phi(), &grad_phi_p); CHKERRXX(ierr);
    node_vector_block_data.push_back(grad_phi_p);
    node_vector_block_names.push_back("grad_phi");
  }
  my_p4est_vtk_write_all_general_lists(p4est_n, nodes_n, ghost_n,
                                       P4EST_TRUE, P4EST_TRUE,
                                       (vtk_directory + "/snapshot_" + std::to_string(index)).c_str(),
                                       &node_scalar_data, &node_scalar_names,
                                       NULL, NULL,
                                       &node_vector_block_data, &node_vector_block_names,
                                       &cell_scalar_data, &cell_scalar_names,
                                       NULL, NULL,
                                       NULL, NULL);
  if(vnp1_nodes_plus != NULL){
    ierr = VecRestoreArrayRead(vnp1_nodes_plus, &vnp1_nodes_plus_p); CHKERRXX(ierr); }
  if(vnp1_nodes_minus != NULL){
    ierr = VecRestoreArrayRead(vnp1_nodes_minus, &vnp1_nodes_minus_p); CHKERRXX(ierr); }
  if(projection_variable != NULL){
    ierr = VecRestoreArrayRead(projection_variable, &projection_variable_p); CHKERRXX(ierr); }
  if(phi_on_computational_nodes != NULL){
    ierr = VecRestoreArrayRead(phi_on_computational_nodes, &phi_on_computational_nodes_p); CHKERRXX(ierr); }

  if(interface_manager->subcell_resolution() == 0)
  {
    ierr = VecRestoreArrayRead(interface_manager->get_curvature(), &curvature_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(interface_manager->get_grad_phi(), &grad_phi_p); CHKERRXX(ierr);
  }

  if(interface_manager->subcell_resolution() > 0)
  {
    node_scalar_data.clear(); node_scalar_names.clear();
    node_vector_block_data.clear(); node_vector_block_names.clear();
    ierr = VecGetArrayRead(phi, &phi_on_fine_nodes_p); CHKERRXX(ierr);
    node_scalar_data.push_back(phi_on_fine_nodes_p);
    node_scalar_names.push_back("phi");
    ierr = VecGetArrayRead(interface_manager->get_curvature(), &curvature_p); CHKERRXX(ierr);
    node_scalar_data.push_back(curvature_p);
    node_scalar_names.push_back("curvature");
    ierr = VecGetArrayRead(interface_manager->get_grad_phi(), &grad_phi_p); CHKERRXX(ierr);
    node_vector_block_data.push_back(grad_phi_p);
    node_vector_block_names.push_back("grad_phi");
    my_p4est_vtk_write_all_general_lists(fine_p4est_n, fine_nodes_n, fine_ghost_n,
                                         P4EST_TRUE, P4EST_TRUE,
                                         (vtk_directory + "/subresolved_snapshot_" + std::to_string(index)).c_str(),
                                         &node_scalar_data, &node_scalar_names, NULL, NULL, &node_vector_block_data, &node_vector_block_names,
                                         NULL, NULL, NULL, NULL, NULL, NULL);
    ierr = VecRestoreArrayRead(phi, &phi_on_fine_nodes_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(interface_manager->get_curvature(), &curvature_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(interface_manager->get_grad_phi(), &grad_phi_p); CHKERRXX(ierr);
  }

  ierr = PetscPrintf(p4est_n->mpicomm, "Saved visual data in ... %s (snapshot %d)\n", vtk_directory.c_str(), index); CHKERRXX(ierr);
  return;
}

void my_p4est_two_phase_flows_t::compute_backtraced_velocities()
{
  if(semi_lagrangian_backtrace_is_done)
    return;
  P4EST_ASSERT((vnm1_nodes_minus_xxyyzz == NULL && vnm1_nodes_plus_xxyyzz == NULL && vn_nodes_minus_xxyyzz == NULL && vn_nodes_plus_xxyyzz == NULL) ||
               (vnm1_nodes_minus_xxyyzz != NULL && vnm1_nodes_plus_xxyyzz != NULL && vn_nodes_minus_xxyyzz != NULL && vn_nodes_plus_xxyyzz != NULL));

  /* first find the velocity at the np1 points */
  my_p4est_interpolation_nodes_t interp_np1_plus(ngbd_n), interp_np1_minus(ngbd_n);
  const bool use_second_derivatives = (vnm1_nodes_minus_xxyyzz != NULL && vnm1_nodes_plus_xxyyzz != NULL && vn_nodes_minus_xxyyzz != NULL && vn_nodes_plus_xxyyzz != NULL);
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
  if(interface_velocity_np1 == NULL){
    ierr = VecCreateGhostNodesBlock(p4est_n, nodes_n, P4EST_DIM, &interface_velocity_np1); CHKERRXX(ierr);
  }

  double *interface_velocity_np1_p  = NULL;
  my_p4est_interpolation_nodes_t *interp_mass_flux = NULL;
  if(mass_flux != NULL)
  {
    interp_mass_flux = new my_p4est_interpolation_nodes_t(&interface_manager->get_interface_capturing_ngbd_n());
    interp_mass_flux->set_input(mass_flux, linear);
  }
  const double *vnp1_nodes_minus_p, *vnp1_nodes_plus_p;
  ierr = VecGetArray(interface_velocity_np1, &interface_velocity_np1_p);  CHKERRXX(ierr);
  ierr = VecGetArrayRead(vnp1_nodes_plus,   &vnp1_nodes_plus_p);  CHKERRXX(ierr);
  ierr = VecGetArrayRead(vnp1_nodes_minus,  &vnp1_nodes_minus_p); CHKERRXX(ierr);
  double xyz_node[P4EST_DIM];
  double local_normal_vector[P4EST_DIM] = {DIM(0.0, 0.0, 0.0)}; // initialize it so that there is no weird shit going on in case mass_flux == NULL
  for (size_t k = 0; k < ngbd_n->get_layer_size(); ++k) {
    const p4est_locidx_t node_idx = ngbd_n->get_layer_node(k);
    node_xyz_fr_n(node_idx, p4est_n, nodes_n, xyz_node);
    const double local_mass_flux = (interp_mass_flux != NULL ? (*interp_mass_flux)(xyz_node) : 0.0);
    if(interp_mass_flux != NULL)
      interface_manager->normal_vector_at_point(xyz_node, local_normal_vector);
    for (u_char dir = 0; dir < P4EST_DIM; ++dir)
      interface_velocity_np1_p[P4EST_DIM*node_idx + dir] = ((vnp1_nodes_minus_p[P4EST_DIM*node_idx + dir] - local_mass_flux*local_normal_vector[dir]/rho_minus)*mu_minus + (vnp1_nodes_plus_p[P4EST_DIM*node_idx + dir] - local_mass_flux*local_normal_vector[dir]/rho_plus)*mu_plus)/(mu_minus + mu_plus);
  }
  ierr = VecGhostUpdateBegin(interface_velocity_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for (size_t k = 0; k < ngbd_n->get_local_size(); ++k) {
    const p4est_locidx_t node_idx = ngbd_n->get_local_node(k);
    node_xyz_fr_n(node_idx, p4est_n, nodes_n, xyz_node);
    const double local_mass_flux = (interp_mass_flux != NULL ? (*interp_mass_flux)(xyz_node) : 0.0);
    if(interp_mass_flux != NULL)
      interface_manager->normal_vector_at_point(xyz_node, local_normal_vector);
    for (u_char dir = 0; dir < P4EST_DIM; ++dir)
      interface_velocity_np1_p[P4EST_DIM*node_idx + dir] = ((vnp1_nodes_minus_p[P4EST_DIM*node_idx + dir] - local_mass_flux*local_normal_vector[dir]/rho_minus)*mu_minus + (vnp1_nodes_plus_p[P4EST_DIM*node_idx + dir] - local_mass_flux*local_normal_vector[dir]/rho_plus)*mu_plus)/(mu_minus + mu_plus);
  }
  ierr = VecGhostUpdateEnd(interface_velocity_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  if(interface_velocity_np1_xxyyzz == NULL){
    ierr = VecCreateGhostNodesBlock(p4est_n, nodes_n, SQR_P4EST_DIM, &interface_velocity_np1_xxyyzz); CHKERRXX(ierr);
  }
  ngbd_n->second_derivatives_central(interface_velocity_np1, interface_velocity_np1_xxyyzz, P4EST_DIM);

  ierr = VecRestoreArrayRead(vnp1_nodes_minus,  &vnp1_nodes_minus_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(vnp1_nodes_plus,   &vnp1_nodes_plus_p);  CHKERRXX(ierr);
  ierr = VecRestoreArray(interface_velocity_np1, &interface_velocity_np1_p);  CHKERRXX(ierr);
  if(interp_mass_flux != NULL)
    delete interp_mass_flux;

  // now extend from interface to flatten that?

  return;
}

void my_p4est_two_phase_flows_t::advect_interface(const p4est_t *p4est_np1, const p4est_nodes_t *nodes_np1, Vec phi_np1,
                                                  const p4est_nodes_t *known_nodes, Vec known_phi_np1)
{
  PetscErrorCode ierr;

  my_p4est_interpolation_nodes_t interp_np1(ngbd_n); // yes, it's normal: we are in the process of advancing time...
  interp_np1.set_input(interface_velocity_np1, interface_velocity_np1_xxyyzz, quadratic, P4EST_DIM);

  std::vector<size_t> already_known;
  p4est_locidx_t origin_local_idx;
  const p4est_quadrant_t *node = NULL;
  const double *known_phi_np1_p = NULL;
  double *phi_np1_p = NULL;
  if(known_phi_np1 != NULL)
  {
    ierr = VecGetArrayRead(known_phi_np1, &known_phi_np1_p);  CHKERRXX(ierr);
    ierr = VecGetArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);
  }

  /* find the velocity field at time np1 */
  size_t to_compute = 0;
  for (size_t node_idx = 0; node_idx < (size_t) nodes_np1->num_owned_indeps; ++node_idx)
  {
    if(known_phi_np1_p != NULL)
    {
      node = (const p4est_quadrant_t*) sc_const_array_index(&nodes_np1->indep_nodes, node_idx);
      if(index_of_node(node, known_nodes, origin_local_idx))
      {
        phi_np1_p[node_idx] = known_phi_np1_p[origin_local_idx];
        already_known.push_back(node_idx);
        continue;
      }
    }
    double xyz[P4EST_DIM];
    node_xyz_fr_n(node_idx, p4est_np1, nodes_np1, xyz);
    interp_np1.add_point(to_compute++, xyz);
  }
  P4EST_ASSERT(to_compute + already_known.size() == (size_t) nodes_np1->num_owned_indeps);

  std::vector<double> interface_velocity_np1_buffer(P4EST_DIM*to_compute);
  interp_np1.interpolate(interface_velocity_np1_buffer.data()); interp_np1.clear();

  /* now backtrace the points to backtrace and interpolate phi there to define phi_np1*/
  my_p4est_interpolation_nodes_t& interp_phi_n = interface_manager->get_interp_phi(); interp_phi_n.clear(); // clear the buffers, if not empty
  size_t known_idx = 0;
  to_compute = 0;
  for (size_t node_idx = 0; node_idx < (size_t) nodes_np1->num_owned_indeps; ++node_idx)
  {
    if(known_phi_np1_p != NULL && known_idx < already_known.size() && node_idx == already_known[known_idx])
    {
      known_idx++;
      continue;
    }
    double xyz_d[P4EST_DIM];
    node_xyz_fr_n(node_idx, p4est_np1, nodes_np1, xyz_d);
    for (u_char dir = 0; dir < P4EST_DIM; ++dir)
      xyz_d[dir] -= dt_n*interface_velocity_np1_buffer[P4EST_DIM*to_compute + dir];
    clip_in_domain(xyz_d, xyz_min, xyz_max, periodicity);

    interp_phi_n.add_point(node_idx, xyz_d);
    to_compute++;
  }
  P4EST_ASSERT(to_compute + known_idx == (size_t) nodes_np1->num_owned_indeps);

  if(known_phi_np1 != NULL)
  {
    ierr = VecRestoreArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(known_phi_np1, &known_phi_np1_p);  CHKERRXX(ierr);
  }

  interp_phi_n.interpolate(phi_np1); interp_phi_n.clear();
  ierr = VecGhostUpdateBegin(phi_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(phi_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  return;
}

void my_p4est_two_phase_flows_t::sample_static_levelset_on_nodes(const p4est_t *p4est_np1, const p4est_nodes_t *nodes_np1, Vec phi_np1)
{
  PetscErrorCode ierr;

  /* simply interpolate phi to the new nodes to define phi_np1*/
  my_p4est_interpolation_nodes_t& interp_phi_n = interface_manager->get_interp_phi(); interp_phi_n.clear(); // clear the buffers, if not empty
  for (size_t node_idx = 0; node_idx < (size_t) nodes_np1->num_owned_indeps; ++node_idx)
  {
    double xyz_node[P4EST_DIM];
    node_xyz_fr_n(node_idx, p4est_np1, nodes_np1, xyz_node);
    interp_phi_n.add_point(node_idx, xyz_node);
  }

  interp_phi_n.interpolate(phi_np1); interp_phi_n.clear();
  ierr = VecGhostUpdateBegin(phi_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(phi_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  return;
}


void my_p4est_two_phase_flows_t::update_from_tn_to_tnp1(const bool& reinitialize_levelset, const bool& static_interface)
{
  PetscErrorCode ierr;

  if(!dt_updated)
    compute_dt();
  dt_updated = false;

  if(!static_interface)
    set_interface_velocity();

  // find the np1 computational grid
  splitting_criteria_computational_grid_two_phase_t criterion_computational_grid(this);
  // if max_lvl == min_lvl, the computational grid is uniform and we don't have to worry about adapting it, so define
  const bool uniform_grid = criterion_computational_grid.max_lvl == criterion_computational_grid.min_lvl;
  /* initialize the new forest */
  p4est_t *p4est_np1 = (uniform_grid ? p4est_n : p4est_copy(p4est_n, P4EST_FALSE)); // no need to copy if no need to refine/coarsen, otherwise very efficient copy
  p4est_nodes_t *nodes_np1 = nodes_n; // no change, yet
  Vec phi_on_computational_nodes_np1 = NULL; // unknown yet, we need to advect it!
  Vec vorticity_magnitude_np1_minus = vorticity_magnitude_minus;
  Vec vorticity_magnitude_np1_plus  = vorticity_magnitude_plus;
  my_p4est_interpolation_nodes_t interp_nodes(ngbd_n);
  u_int iter = 0;
  bool iterative_grid_update_converged = false;
  while(!iterative_grid_update_converged)
  {
    p4est_nodes_t *previous_nodes_np1 = nodes_np1;
    Vec previous_phi_on_computational_nodes_np1 = phi_on_computational_nodes_np1;
    Vec previous_vorticity_magnitude_np1_minus = vorticity_magnitude_np1_minus;
    Vec previous_vorticity_magnitude_np1_plus  = vorticity_magnitude_np1_plus;
    /* ---   FIND THE NEXT ADAPTIVE GRID   --- */
    /* For the very first iteration of the grid-update procedure, p4est_np1 is a
     * simple pure copy of p4est_n, so no node creation nor data interpolation is
     * required. Hence the "if(iter > 0)..." statements */

    if(iter > 0)
    {
      // maybe no longer properly balanced so partition...
      my_p4est_partition(p4est_np1, P4EST_FALSE, NULL);
      // get nodes_np1
      nodes_np1 = my_p4est_nodes_new(p4est_np1, NULL);
    }
    // reset phi_on_computational_nodes_np1
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_on_computational_nodes_np1); CHKERRXX(ierr);
    P4EST_ASSERT((iter > 0 && previous_phi_on_computational_nodes_np1 != NULL)  || (iter == 0 && previous_phi_on_computational_nodes_np1 == NULL));
    if(!static_interface)
      advect_interface(p4est_np1, nodes_np1, phi_on_computational_nodes_np1,
                       previous_nodes_np1, previous_phi_on_computational_nodes_np1); // limit the workload: use what you already know if you know it!
    else
      sample_static_levelset_on_nodes(p4est_np1, nodes_np1, phi_on_computational_nodes_np1);


    // get vorticity_np1
    if(iter > 0){
      // reset vorticity_np1 if needed
      ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &vorticity_magnitude_np1_minus);  CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &vorticity_magnitude_np1_plus);   CHKERRXX(ierr);
      const double *previous_vorticity_magnitude_np1_minus_p, *previous_vorticity_magnitude_np1_plus_p;
      double *vorticity_magnitude_np1_minus_p, *vorticity_magnitude_np1_plus_p;
      ierr = VecGetArrayRead(previous_vorticity_magnitude_np1_minus, &previous_vorticity_magnitude_np1_minus_p);  CHKERRXX(ierr);
      ierr = VecGetArrayRead(previous_vorticity_magnitude_np1_plus,  &previous_vorticity_magnitude_np1_plus_p);   CHKERRXX(ierr);
      ierr = VecGetArray(vorticity_magnitude_np1_minus, &vorticity_magnitude_np1_minus_p);  CHKERRXX(ierr);
      ierr = VecGetArray(vorticity_magnitude_np1_plus,  &vorticity_magnitude_np1_plus_p);   CHKERRXX(ierr);

      interp_nodes.clear();
      for(size_t node_idx = 0; node_idx < nodes_np1->indep_nodes.elem_count; ++node_idx)
      {
        const p4est_quadrant_t *node = (const p4est_quadrant_t*) sc_const_array_index(&nodes_np1->indep_nodes, node_idx);
        P4EST_ASSERT (p4est_quadrant_is_node (node, 1));
        p4est_locidx_t origin_idx;
        if(index_of_node(node, previous_nodes_np1, origin_idx))
        {
          vorticity_magnitude_np1_minus_p[node_idx] = previous_vorticity_magnitude_np1_minus_p[origin_idx];
          vorticity_magnitude_np1_plus_p[node_idx]  = previous_vorticity_magnitude_np1_plus_p[origin_idx];
        }
        else
        {
          double xyz[P4EST_DIM];
          node_xyz_fr_n(node_idx, p4est_np1, nodes_np1, xyz);
          interp_nodes.add_point(node_idx, xyz);
        }
      }
      Vec inputs[2]   = {vorticity_magnitude_minus, vorticity_magnitude_plus};
      Vec outputs[2]  = {vorticity_magnitude_np1_minus, vorticity_magnitude_np1_plus};
      interp_nodes.set_input(inputs, linear, 2);
      interp_nodes.interpolate(outputs);

      ierr = VecRestoreArray(vorticity_magnitude_np1_plus,  &vorticity_magnitude_np1_plus_p);   CHKERRXX(ierr);
      ierr = VecRestoreArray(vorticity_magnitude_np1_minus, &vorticity_magnitude_np1_minus_p);  CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(previous_vorticity_magnitude_np1_plus,  &previous_vorticity_magnitude_np1_plus_p);   CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(previous_vorticity_magnitude_np1_minus, &previous_vorticity_magnitude_np1_minus_p);  CHKERRXX(ierr);
    }

    // update the grid
    if(!uniform_grid)
      iterative_grid_update_converged = !criterion_computational_grid.refine_and_coarsen(p4est_np1, nodes_np1, phi_on_computational_nodes_np1, vorticity_magnitude_np1_minus, vorticity_magnitude_np1_plus);
    else
      iterative_grid_update_converged = true;

    iter++;

    ierr = delete_and_nullify_vector(previous_phi_on_computational_nodes_np1); CHKERRXX(ierr);
    if(previous_nodes_np1 != nodes_n)
      p4est_nodes_destroy(previous_nodes_np1);
    if(previous_vorticity_magnitude_np1_minus != vorticity_magnitude_minus){
      ierr = delete_and_nullify_vector(previous_vorticity_magnitude_np1_minus); CHKERRXX(ierr); }
    if(previous_vorticity_magnitude_np1_plus != vorticity_magnitude_plus){
      ierr = delete_and_nullify_vector(previous_vorticity_magnitude_np1_plus); CHKERRXX(ierr); }

    if(iter > ((unsigned int) 2 + criterion_computational_grid.max_lvl - criterion_computational_grid.min_lvl)) // increase the rhs by one to account for the very first step that used to be out of the loop, [Raphael]
    {
      ierr = PetscPrintf(p4est_n->mpicomm, "ooops ... the grid update did not converge\n"); CHKERRXX(ierr);
      break;
    }
  }

  if(vorticity_magnitude_np1_minus != vorticity_magnitude_minus){
    ierr = delete_and_nullify_vector(vorticity_magnitude_np1_minus); CHKERRXX(ierr); }
  if(vorticity_magnitude_np1_plus != vorticity_magnitude_plus){
    ierr = delete_and_nullify_vector(vorticity_magnitude_np1_plus); CHKERRXX(ierr); }
  // we do not need the vorticities anymore
  ierr = delete_and_nullify_vector(vorticity_magnitude_minus); CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(vorticity_magnitude_plus); CHKERRXX(ierr);

  // Finalize the computational grid np1 (if needed):
  p4est_ghost_t *ghost_np1 = ghost_n;
  my_p4est_hierarchy_t *hierarchy_np1 = hierarchy_n;
  my_p4est_node_neighbors_t *ngbd_np1 = ngbd_n;
  if(p4est_np1 != p4est_n)
  {
    // Balance the grid and repartition
    p4est_balance(p4est_np1, P4EST_CONNECT_FULL, NULL);
    my_p4est_partition(p4est_np1, P4EST_FALSE, NULL);
    /* Get the ghost cells at time np1, */
    ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
    my_p4est_ghost_expand(p4est_np1, ghost_np1);
    // save the currently known nodes and phi_np1 to alleviate workload of the advection
    p4est_nodes_t *known_nodes_np1 = nodes_np1;
    Vec known_phi_on_computational_nodes_np1 = phi_on_computational_nodes_np1;
    // get the final computational nodes np1
    nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);
    hierarchy_np1 = new my_p4est_hierarchy_t(p4est_np1, ghost_np1, brick);
    ngbd_np1 = new my_p4est_node_neighbors_t(hierarchy_np1, nodes_np1);
    ngbd_np1->init_neighbors();

    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_on_computational_nodes_np1); CHKERRXX(ierr);
    if(!static_interface)
      advect_interface(p4est_np1, nodes_np1, phi_on_computational_nodes_np1, known_nodes_np1, known_phi_on_computational_nodes_np1);
    else
      sample_static_levelset_on_nodes(p4est_np1, nodes_np1, phi_on_computational_nodes_np1);
    // destroy what you saved
    p4est_nodes_destroy(known_nodes_np1);
    ierr = delete_and_nullify_vector(known_phi_on_computational_nodes_np1); CHKERRXX(ierr);
  }

  // we are done with the computational grid np1
  p4est_t *fine_p4est_np1 = NULL;
  p4est_ghost_t *fine_ghost_np1 = NULL;
  p4est_nodes_t *fine_nodes_np1 = NULL;
  my_p4est_hierarchy_t *fine_hierarchy_np1 = NULL;
  my_p4est_node_neighbors_t *fine_ngbd_np1 = NULL;
  Vec phi_on_fine_nodes_np1 = NULL;
  if(interface_manager->subcell_resolution() > 0)
  {
    // now, find the np1 interface-capturing grid
    splitting_criteria_t* data_fine = (splitting_criteria_t*) fine_p4est_n->user_pointer;
    splitting_criteria_tag_t criterion_fine_grid(data_fine);
    P4EST_ASSERT(criterion_fine_grid.max_lvl > criterion_computational_grid.max_lvl);
    P4EST_ASSERT(criterion_fine_grid.max_lvl > criterion_fine_grid.min_lvl);
    /* initialize the new sub-resolving forest */
    fine_p4est_np1 = p4est_copy(p4est_np1, P4EST_FALSE);
    fine_p4est_np1->user_pointer = data_fine;

    // initialization
    fine_nodes_np1 = nodes_np1;
    phi_on_fine_nodes_np1 = phi_on_computational_nodes_np1;

    iter = 0;
    iterative_grid_update_converged = !criterion_fine_grid.refine(fine_p4est_np1, fine_nodes_np1, phi_on_fine_nodes_np1);
    /* ---   FIND THE NEXT ADAPTIVE GRID   --- */
    // We never partition this one, to ensure locality of interface-capturing data...
    // We also do not coarsen the grid in any ways (not really parallel-friendly...)
    while(!iterative_grid_update_converged)
    {
      iter++;
      p4est_nodes_t *previous_fine_nodes_np1 = fine_nodes_np1;
      Vec previous_phi_on_fine_nodes_np1 = phi_on_fine_nodes_np1;
      // advect more fine_phi_np1
      fine_nodes_np1 = my_p4est_nodes_new(fine_p4est_np1, NULL);
      ierr = VecCreateGhostNodes(fine_p4est_np1, fine_nodes_np1, &phi_on_fine_nodes_np1); CHKERRXX(ierr);
      if(!static_interface)
        advect_interface(fine_p4est_np1, fine_nodes_np1, phi_on_fine_nodes_np1,
                         previous_fine_nodes_np1, previous_phi_on_fine_nodes_np1); // limit the workload: use what you already know!
      else
        sample_static_levelset_on_nodes(fine_p4est_np1, fine_nodes_np1, phi_on_fine_nodes_np1);

      if(previous_phi_on_fine_nodes_np1 != phi_on_computational_nodes_np1){
        ierr = delete_and_nullify_vector(previous_phi_on_fine_nodes_np1); CHKERRXX(ierr); }
      if(previous_fine_nodes_np1 != nodes_np1)
        p4est_nodes_destroy(previous_fine_nodes_np1);

      iterative_grid_update_converged = !criterion_fine_grid.refine(fine_p4est_np1, fine_nodes_np1, phi_on_fine_nodes_np1);

      if(iter > (u_int) 1 + interface_manager->subcell_resolution())
      {
        ierr = PetscPrintf(p4est_n->mpicomm, "ooops ... the grid update did not converge for the subresolving grid\n"); CHKERRXX(ierr);
        break;
      }
    }
    P4EST_ASSERT(iter > 0); // must have gone through the loop at least once!

    fine_ghost_np1 = my_p4est_ghost_new(fine_p4est_np1, P4EST_CONNECT_FULL);
    my_p4est_ghost_expand(fine_p4est_np1, fine_ghost_np1);
    // finalize fine_phi_np1
    // save what you know (without the ghost)
    p4est_nodes_t *previous_fine_nodes_np1 = fine_nodes_np1;
    Vec previous_phi_on_fine_nodes_np1 = phi_on_fine_nodes_np1;
    // create the final guys
    fine_nodes_np1 = my_p4est_nodes_new(fine_p4est_np1, fine_ghost_np1);
    ierr = VecCreateGhostNodes(fine_p4est_np1, fine_nodes_np1, &phi_on_fine_nodes_np1); CHKERRXX(ierr);

    fine_hierarchy_np1 = new my_p4est_hierarchy_t(fine_p4est_np1, fine_ghost_np1, brick);
    fine_ngbd_np1 = new my_p4est_node_neighbors_t(fine_hierarchy_np1, fine_nodes_np1);
    fine_ngbd_np1->init_neighbors();

    if(!iterative_grid_update_converged)
    {
      // this should never happen, but in case it does, we need to advect once more
      // to ensure that every node's levelset value is well-defined...
      // that every final local node is known already, so here we are
      if(!static_interface)
        advect_interface(fine_p4est_np1, fine_nodes_np1, phi_on_fine_nodes_np1,
                         previous_fine_nodes_np1, previous_phi_on_fine_nodes_np1); // limit the workload: use what you already know!
      else
        sample_static_levelset_on_nodes(fine_p4est_np1, fine_nodes_np1, phi_on_fine_nodes_np1); CHKERRXX(ierr);
    }
    else
    {
      const double *previous_phi_on_fine_nodes_np1_p;
      double *phi_on_fine_nodes_np1_p;
      ierr = VecGetArrayRead(previous_phi_on_fine_nodes_np1, &previous_phi_on_fine_nodes_np1_p); CHKERRXX(ierr);
      ierr = VecGetArray(phi_on_fine_nodes_np1, &phi_on_fine_nodes_np1_p); CHKERRXX(ierr);
      for (size_t k = 0; k < fine_ngbd_np1->get_layer_size(); ++k) {
        const p4est_locidx_t fine_node_idx = fine_ngbd_np1->get_layer_node(k);
#ifdef P4EST_DEBUG
        // hard check your assumption
        const p4est_quadrant_t *fine_node = (const p4est_quadrant_t*) sc_array_index(&fine_nodes_np1->indep_nodes, fine_node_idx);
        P4EST_ASSERT (p4est_quadrant_is_node (fine_node, 1));
        p4est_locidx_t origin_idx;
        P4EST_ASSERT(index_of_node(fine_node, previous_fine_nodes_np1, origin_idx)); // force the abort, since the implementation assumption is wrong --> the developer has more work to do here
        P4EST_ASSERT(origin_idx == fine_node_idx);
#endif
        phi_on_fine_nodes_np1_p[fine_node_idx] = previous_phi_on_fine_nodes_np1_p[fine_node_idx];
      }
      ierr = VecGhostUpdateBegin(phi_on_fine_nodes_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      for (size_t k = 0; k < fine_ngbd_np1->get_local_size(); ++k) {
        const p4est_locidx_t fine_node_idx = fine_ngbd_np1->get_local_node(k);
#ifdef P4EST_DEBUG
        // hard check your assumption
        const p4est_quadrant_t *fine_node = (const p4est_quadrant_t*) sc_array_index(&fine_nodes_np1->indep_nodes, fine_node_idx);
        P4EST_ASSERT (p4est_quadrant_is_node (fine_node, 1));
        p4est_locidx_t origin_idx;
        P4EST_ASSERT(index_of_node(fine_node, previous_fine_nodes_np1, origin_idx)); // force the abort, since the implementation assumption is wrong --> the developer has more work to do here
        P4EST_ASSERT(origin_idx == fine_node_idx);
#endif
        phi_on_fine_nodes_np1_p[fine_node_idx] = previous_phi_on_fine_nodes_np1_p[fine_node_idx];
      }
      ierr = VecGhostUpdateEnd(phi_on_fine_nodes_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecRestoreArray(phi_on_fine_nodes_np1, &phi_on_fine_nodes_np1_p); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(previous_phi_on_fine_nodes_np1, &previous_phi_on_fine_nodes_np1_p); CHKERRXX(ierr);
    }
    if(previous_fine_nodes_np1 != nodes_np1)
      p4est_nodes_destroy(previous_fine_nodes_np1);

    if(previous_phi_on_fine_nodes_np1 != phi_on_computational_nodes_np1){
      ierr = delete_and_nullify_vector(previous_phi_on_fine_nodes_np1); CHKERRXX(ierr); }

    P4EST_ASSERT(fine_p4est_np1 != NULL && fine_ghost_np1 != NULL && fine_nodes_np1 != NULL && fine_nodes_np1 != nodes_np1 && fine_hierarchy_np1 != NULL && fine_ngbd_np1 != NULL && phi_on_fine_nodes_np1 != NULL && phi_on_fine_nodes_np1 != phi_on_computational_nodes_np1);
  }

  const my_p4est_node_neighbors_t* interface_resolving_ngbd_np1 = (fine_ngbd_np1 != NULL ? fine_ngbd_np1 : ngbd_np1);
  if(reinitialize_levelset)
  {
    Vec& interface_resolving_phi_np1 = (fine_ngbd_np1 != NULL ? phi_on_fine_nodes_np1 : phi_on_computational_nodes_np1);
    my_p4est_level_set_t ls(interface_resolving_ngbd_np1);
    ls.reinitialize_2nd_order(interface_resolving_phi_np1);
  }

  if(interface_manager->subcell_resolution() > 0)
  {
    // transfer reinitialized levelset values from the fine grid data to the computational grid
    double *phi_on_computational_nodes_np1_p;
    const double *phi_on_fine_nodes_np1_p;
    ierr = VecGetArrayRead(phi_on_fine_nodes_np1, &phi_on_fine_nodes_np1_p); CHKERRXX(ierr);
    ierr = VecGetArray(phi_on_computational_nodes_np1, &phi_on_computational_nodes_np1_p); CHKERRXX(ierr);
    for (size_t k = 0; k < ngbd_np1->get_layer_size(); ++k) {
      const p4est_locidx_t node_idx = ngbd_np1->get_layer_node(k);
      // hard check your assumption
      const p4est_quadrant_t *node = (const p4est_quadrant_t*) sc_array_index(&nodes_np1->indep_nodes, node_idx);
      P4EST_ASSERT (p4est_quadrant_is_node (node, 1));
      p4est_locidx_t corresponding_fine_node_idx;
      const bool node_found = index_of_node(node, fine_nodes_np1, corresponding_fine_node_idx); // force the abort, since the implementation assumption is wrong --> the developer has more work to do here
      P4EST_ASSERT(node_found); (void) node_found;
      phi_on_computational_nodes_np1_p[node_idx] = phi_on_fine_nodes_np1_p[corresponding_fine_node_idx];
    }
    ierr = VecGhostUpdateBegin(phi_on_computational_nodes_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    for (size_t k = 0; k < ngbd_np1->get_local_size(); ++k) {
      const p4est_locidx_t node_idx = ngbd_np1->get_local_node(k);
      // hard check your assumption
      const p4est_quadrant_t *node = (const p4est_quadrant_t*) sc_array_index(&nodes_np1->indep_nodes, node_idx);
      P4EST_ASSERT (p4est_quadrant_is_node (node, 1));
      p4est_locidx_t corresponding_fine_node_idx;
      const bool node_found = index_of_node(node, fine_nodes_np1, corresponding_fine_node_idx); // force the abort, since the implementation assumption is wrong --> the developer has more work to do here
      P4EST_ASSERT(node_found); (void) node_found;
      phi_on_computational_nodes_np1_p[node_idx] = phi_on_fine_nodes_np1_p[corresponding_fine_node_idx];
    }
    ierr = VecGhostUpdateEnd(phi_on_computational_nodes_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_on_computational_nodes_np1, &phi_on_computational_nodes_np1_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(phi_on_fine_nodes_np1, &phi_on_fine_nodes_np1_p); CHKERRXX(ierr);
  }
  // you have your new grids and the new levelset function, now transfer your data!

  /* slide relevant fiels and grids in time: nm1 data are disregarded, n data becomes nm1 data and np1 data become n data...
   * In particular, if grid_is_unchanged is false, the np1 grid is different than grid at time n, we need to
   * re-construct its faces and cell-neighbors and the solvers we have used will need to be destroyed... */

  ierr = delete_and_nullify_vector(mass_flux); CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(pressure_jump);  CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(interface_stress); CHKERRXX(ierr);
  // on computational grid at time nm1, just "slide" fields and grids in discrete time
  ierr = delete_and_nullify_vector(vnm1_nodes_minus); CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(vnm1_nodes_plus);  CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(vnm1_nodes_minus_xxyyzz);  CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(vnm1_nodes_plus_xxyyzz);   CHKERRXX(ierr);
  vnm1_nodes_minus  = vn_nodes_minus;
  vnm1_nodes_plus   = vn_nodes_plus;
  vnm1_nodes_minus_xxyyzz = vn_nodes_minus_xxyyzz;
  vnm1_nodes_plus_xxyyzz  = vn_nodes_plus_xxyyzz;
  // no longer need the interface velocity
  ierr = delete_and_nullify_vector(interface_velocity_np1); CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(interface_velocity_np1_xxyyzz); CHKERRXX(ierr);
  // on computational grid at time n, we will need to interpolate things...
  // we will need to interpolate for vnp1 to vn (and then we'll take their derivatives)
  if(p4est_np1 != p4est_n)
  {
    ierr = VecCreateGhostNodesBlock(p4est_np1, nodes_np1, P4EST_DIM, &vn_nodes_minus); CHKERRXX(ierr);
    ierr = VecCreateGhostNodesBlock(p4est_np1, nodes_np1, P4EST_DIM, &vn_nodes_plus); CHKERRXX(ierr);
    interp_nodes.clear();
    for (size_t k = 0; k < nodes_np1->indep_nodes.elem_count; ++k) {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(k, p4est_np1, nodes_np1, xyz);
      interp_nodes.add_point(k, xyz);
    }
    Vec inputs[2]   = {vnp1_nodes_minus, vnp1_nodes_plus};
    Vec outputs[2]  = {vn_nodes_minus, vn_nodes_plus};
    interp_nodes.set_input(inputs, quadratic, 2, P4EST_DIM);
    interp_nodes.interpolate(outputs);
    // clear those
    ierr = delete_and_nullify_vector(vnp1_nodes_minus); CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(vnp1_nodes_plus);  CHKERRXX(ierr);
  }
  else{
    vn_nodes_minus  = vnp1_nodes_minus;
    vn_nodes_plus   = vnp1_nodes_plus;
    vnp1_nodes_minus  = NULL; // to avoid overwriting thereafter
    vnp1_nodes_plus   = NULL; // to avoid overwriting thereafter
  }
  ierr = VecCreateGhostNodesBlock(p4est_np1, nodes_np1, SQR_P4EST_DIM, &vn_nodes_minus_xxyyzz); CHKERRXX(ierr);
  ierr = VecCreateGhostNodesBlock(p4est_np1, nodes_np1, SQR_P4EST_DIM, &vn_nodes_plus_xxyyzz);  CHKERRXX(ierr);
  Vec inputs[2]         = {vn_nodes_minus, vn_nodes_plus};
  Vec outputs_xxyyzz[2] = {vn_nodes_minus_xxyyzz, vn_nodes_plus_xxyyzz};
  ngbd_np1->second_derivatives_central(inputs, outputs_xxyyzz, 2, P4EST_DIM);

  if(interface_manager->subcell_resolution() > 0)
  {
    p4est_destroy(fine_p4est_n);        fine_p4est_n      = fine_p4est_np1;
    p4est_ghost_destroy(fine_ghost_n);  fine_ghost_n      = fine_ghost_np1;
    p4est_nodes_destroy(fine_nodes_n);  fine_nodes_n      = fine_nodes_np1;
    delete  fine_hierarchy_n;           fine_hierarchy_n  = fine_hierarchy_np1;
    delete  fine_ngbd_n;                fine_ngbd_n       = fine_ngbd_np1;
  }
  else
    P4EST_ASSERT(fine_p4est_n == NULL && fine_p4est_np1 == NULL && fine_ghost_n == NULL && fine_ghost_np1 == NULL &&  fine_nodes_n == NULL && fine_nodes_np1 == NULL &&
                 fine_hierarchy_n == NULL && fine_hierarchy_np1 == NULL && fine_ngbd_n == NULL && fine_ngbd_np1 == NULL);
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
  if(p4est_np1 != p4est_n || ghost_np1 != ghost_n || hierarchy_np1 != hierarchy_n){
    if(hierarchy_np1 != hierarchy_n)
    {
      delete ngbd_c;
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

  delete interface_manager;
  interface_manager = new my_p4est_interface_manager_t(faces_n, nodes_n, interface_resolving_ngbd_np1);
  interface_manager->evaluate_FD_theta_with_quadratics(fetch_interface_FD_neighbors_with_second_order_accuracy);
  set_phi((fine_ngbd_n != NULL ? phi_on_fine_nodes_np1 : phi_on_computational_nodes_np1), levelset_interpolation_method, phi_on_computational_nodes_np1); // memory handled therein!

  for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = delete_and_nullify_vector(grad_p_guess_over_rho_minus[dir]); CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(grad_p_guess_over_rho_plus[dir]);  CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(vnp1_face_minus[dir]); CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(vnp1_face_plus[dir]);  CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est_n, faces_n, &vnp1_face_minus[dir], dir); CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est_n, faces_n, &vnp1_face_plus[dir],  dir); CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est_n, faces_n, &grad_p_guess_over_rho_minus[dir], dir); CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est_n, faces_n, &grad_p_guess_over_rho_plus[dir],  dir); CHKERRXX(ierr);
  }

  viscosity_solver.reset();
  if(divergence_free_projector != NULL)
  {
    delete divergence_free_projector;

    if(cell_jump_solver_to_use == GFM || cell_jump_solver_to_use == xGFM)
    {
      divergence_free_projector = new my_p4est_poisson_jump_cells_xgfm_t(ngbd_c, nodes_n);
      dynamic_cast<my_p4est_poisson_jump_cells_xgfm_t*>(divergence_free_projector)->activate_xGFM_corrections(cell_jump_solver_to_use == xGFM);
    }
    else
      divergence_free_projector = new my_p4est_poisson_jump_cells_fv_t(ngbd_c, nodes_n);
  }

  // clear grid-related buffers, flags and backtrace semi-lagrangian points
  semi_lagrangian_backtrace_is_done = false;
  for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
    backtraced_vn_faces_minus[dir].resize(faces_n->num_local[dir]);
    backtraced_vn_faces_plus[dir].resize(faces_n->num_local[dir]);
    if(sl_order == 2)
    {
      backtraced_vnm1_faces_minus[dir].resize(faces_n->num_local[dir]);
      backtraced_vnm1_faces_plus[dir].resize(faces_n->num_local[dir]);
    }
    else
    {
      backtraced_vnm1_faces_minus[dir].clear();
      backtraced_vnm1_faces_plus[dir].clear();
    }
    if(!voronoi_on_the_fly)
    {
      voro_cell[dir].clear();
      voro_cell[dir].resize(faces_n->num_local[dir]);
    }
  }

  return;
}

