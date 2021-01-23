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

#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_my_p4est_two_phase_flows_update;
extern PetscLogEvent log_my_p4est_two_phase_flows_solve_pressure_guess;
extern PetscLogEvent log_my_p4est_two_phase_flows_solve_projection;
extern PetscLogEvent log_my_p4est_two_phase_flows_compute_backtracing;
extern PetscLogEvent log_my_p4est_two_phase_flows_solve_viscosity;
extern PetscLogEvent log_my_p4est_two_phase_flows_interpolate_velocity_at_nodes;
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
  cfl_visco_capillary = 0.95;
  cfl_capillary = 0.95;
  dt_updated = false;
  max_L2_norm_velocity_minus = max_L2_norm_velocity_plus = 0.0;

  sl_order = 2;
  sl_order_interface = 2;
  n_viscous_subiterations = INT_MAX;

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
  phi                       = NULL;
  non_viscous_pressure_jump = NULL;
  jump_normal_velocity      = NULL;
  // vector fields and/or other P4EST_DIM-block-structured
  phi_xxyyzz  = NULL;
  interface_tangential_stress = NULL;
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
  interface_velocity_n          = NULL;
  // tensor/matrix fields
  vn_nodes_minus_xxyyzz         = NULL;
  vn_nodes_plus_xxyyzz          = NULL;
  interface_velocity_np1_xxyyzz = NULL;
  interface_velocity_n_xxyyzz   = NULL;
  // ------------------------------------------------------------------------------
  // ----- FIELDS SAMPLED AT CELL CENTERS OF THE COMPUTATIONAL GRID AT TIME N -----
  // ------------------------------------------------------------------------------
  // scalar fields
  pressure_minus                = NULL;
  pressure_plus                 = NULL;
  // ------------------------------------------------------------------------------
  // ----- FIELDS SAMPLED AT FACE CENTERS OF THE COMPUTATIONAL GRID AT TIME N -----
  // ------------------------------------------------------------------------------
  // vector fields
  for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
    vnp1_face_minus_km1[dir]  = NULL;
    vnp1_face_plus_km1[dir]   = NULL;
    vnp1_face_minus[dir]      = NULL;
    vnp1_face_plus[dir]       = NULL;
    viscosity_rhs_minus[dir]    = NULL;
    viscosity_rhs_plus[dir]     = NULL;
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
  }
  semi_lagrangian_backtrace_is_done = false;

  ngbd_n->init_neighbors();
  ngbd_nm1->init_neighbors();
  if(!faces_n->finest_faces_neighborhoods_have_been_set())
    faces_n->set_finest_face_neighborhoods();

  cell_jump_solver = NULL;
  pressure_guess_is_set = false;
  face_jump_solver = NULL;
  voronoi_on_the_fly = false;

  set_cell_jump_solver(FV);   // default is finite-volume solver for projection step
  set_face_jump_solvers(xGFM); // default is finite-difference xGFM solver for viscosity step
  fetch_interface_FD_neighbors_with_second_order_accuracy = false;
}

my_p4est_two_phase_flows_t::my_p4est_two_phase_flows_t(const mpi_environment_t& mpi, const char* path_to_saved_state, double& simulation_time)
  : threshold_dbl_max(get_largest_dbl_smaller_than_dbl_max())
{
  // we need to initialize those to NULL, otherwise the loader will freak out
  // no risk of memory leak here, this is a CONSTRUCTOR
  brick = NULL; conn = NULL;
  p4est_nm1 = NULL; ghost_nm1 = NULL; nodes_nm1 = NULL;
  hierarchy_nm1 = NULL; ngbd_nm1 = NULL;
  p4est_n = NULL; ghost_n = NULL; nodes_n = NULL;
  hierarchy_n = NULL; ngbd_n = NULL;
  ngbd_c = NULL; faces_n = NULL;
  fine_p4est_n = NULL; fine_ghost_n = NULL; fine_nodes_n = NULL;
  fine_hierarchy_n = NULL; fine_ngbd_n = NULL;
  interface_manager = NULL;

  bc_velocity = NULL;
  bc_pressure = NULL;
  for(u_char dim = 0; dim < P4EST_DIM; ++dim)
    force_per_unit_mass[dim]  = NULL;

  // -------------------------------------------------------------------
  // ----- FIELDS SAMPLED AT NODES OF THE INTERFACE-CAPTURING GRID -----
  // -------------------------------------------------------------------
  // scalar fields
  phi                       = NULL;
  non_viscous_pressure_jump = NULL;
  jump_normal_velocity      = NULL;
  // vector fields and/or other P4EST_DIM-block-structured
  phi_xxyyzz  = NULL;
  interface_tangential_stress = NULL;
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
  interface_velocity_n          = NULL;
  // tensor/matrix fields
  vn_nodes_minus_xxyyzz         = NULL;
  vn_nodes_plus_xxyyzz          = NULL;
  interface_velocity_np1_xxyyzz = NULL;
  interface_velocity_n_xxyyzz   = NULL;
  // ------------------------------------------------------------------------------
  // ----- FIELDS SAMPLED AT CELL CENTERS OF THE COMPUTATIONAL GRID AT TIME N -----
  // ------------------------------------------------------------------------------
  // scalar fields
  pressure_minus                = NULL;
  pressure_plus                 = NULL;
  // ------------------------------------------------------------------------------
  // ----- FIELDS SAMPLED AT FACE CENTERS OF THE COMPUTATIONAL GRID AT TIME N -----
  // ------------------------------------------------------------------------------
  // vector fields
  for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
    vnp1_face_minus_km1[dir]    = NULL;
    vnp1_face_plus_km1[dir]     = NULL;
    vnp1_face_minus[dir]        = NULL;
    vnp1_face_plus[dir]         = NULL;
    viscosity_rhs_minus[dir]    = NULL;
    viscosity_rhs_plus[dir]     = NULL;
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
  dt_updated = false;

  // load the solver state from disk
  load_state(mpi, path_to_saved_state, simulation_time);

  // clear backtraced values of velocity components
  for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
    backtraced_vn_faces_minus[dir].clear();   backtraced_vn_faces_plus[dir].clear();
    backtraced_vnm1_faces_minus[dir].clear(); backtraced_vnm1_faces_plus[dir].clear();
  }
  semi_lagrangian_backtrace_is_done = false;

  ngbd_n->init_neighbors();
  ngbd_nm1->init_neighbors();
  if(!faces_n->finest_faces_neighborhoods_have_been_set())
    faces_n->set_finest_face_neighborhoods();

  pressure_guess_is_set = false;
  interface_manager = new my_p4est_interface_manager_t(faces_n, nodes_n, (fine_ngbd_n != NULL ? fine_ngbd_n : ngbd_n));
  xyz_min = p4est_n->connectivity->vertices + 3*p4est_n->connectivity->tree_to_vertex[P4EST_CHILDREN*0                                + 0];
  xyz_max = p4est_n->connectivity->vertices + 3*p4est_n->connectivity->tree_to_vertex[P4EST_CHILDREN*(p4est_n->trees->elem_count - 1) + P4EST_CHILDREN - 1];
  for (u_char dim = 0; dim < P4EST_DIM; ++dim)
    periodicity[dim] = is_periodic(p4est_n, dim);
  tree_diagonal = ABSD(tree_dimension[0], tree_dimension[1], tree_dimension[2]);
  smallest_diagonal = ABSD(dxyz_smallest_quad[0], dxyz_smallest_quad[1], dxyz_smallest_quad[2]);

  set_phi((fine_ngbd_n != NULL ? phi : phi_on_computational_nodes), levelset_interpolation_method, phi_on_computational_nodes);

  compute_second_derivatives_of_n_velocities();
  compute_second_derivatives_of_nm1_velocities();
}

void my_p4est_two_phase_flows_t::load_state(const mpi_environment_t& mpi, const char* path_to_folder, double& tn)
{
  PetscErrorCode ierr;
  char filename[PATH_MAX];

  if(!is_folder(path_to_folder))
    throw std::invalid_argument("my_p4est_two_phase_flows_t::load_state: path_to_folder is invalid.");

  // load general solver parameters first
  splitting_criteria_t* data = new splitting_criteria_t;
  splitting_criteria_t* fine_data = new splitting_criteria_t;
  sprintf(filename, "%s/solver_parameters", path_to_folder);
  save_or_load_parameters(filename, data, fine_data, LOAD, tn, &mpi);

  // load p4est_n and the corresponding objects
  my_p4est_load_forest_and_data(mpi.comm(), path_to_folder, p4est_n, conn, P4EST_TRUE, ghost_n, nodes_n,
                                P4EST_TRUE, brick, P4EST_TRUE, faces_n, hierarchy_n, ngbd_c,
                                "p4est_n", 3,
                                "phi_comp", NODE_DATA, 1, &phi_on_computational_nodes,
                                "vn_nodes_minus", NODE_BLOCK_VECTOR_DATA, 1, &vn_nodes_minus,
                                "vn_nodes_plus", NODE_BLOCK_VECTOR_DATA, 1, &vn_nodes_plus);

  P4EST_ASSERT(find_max_level(p4est_n) == data->max_lvl);

  if(ngbd_n != NULL)
    delete ngbd_n;
  ngbd_n = new my_p4est_node_neighbors_t(hierarchy_n, nodes_n);
  ngbd_n->init_neighbors();
  // load p4est_nm1 and the corresponding objects
  p4est_connectivity_t* conn_nm1 = NULL;
  my_p4est_load_forest_and_data(mpi.comm(), path_to_folder, p4est_nm1, conn_nm1, P4EST_TRUE, ghost_nm1, nodes_nm1,
                                "p4est_nm1", 2,
                                "vnm1_nodes_minus", NODE_BLOCK_VECTOR_DATA, 1, &vnm1_nodes_minus,
                                "vnm1_nodes_plus", NODE_BLOCK_VECTOR_DATA, 1, &vnm1_nodes_plus);
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

void my_p4est_two_phase_flows_t::save_state(const char* path_to_root_directory, double& tn, const int& n_saved)
{
  if(!is_folder(path_to_root_directory))
  {
    if(!create_directory(path_to_root_directory, p4est_n->mpirank, p4est_n->mpicomm))
    {
      char error_msg[BUFSIZ];
      sprintf(error_msg, "my_p4est_two_phase_flows_t::save_state: the path %s is invalid and the directory could not be created", path_to_root_directory);
      throw std::invalid_argument(error_msg);
    }
  }

  int backup_idx = 0;

  if(p4est_n->mpirank == 0)
  {
    int n_backup_subfolders = 0;
    // get the current number of backups already present
    // delete the extra ones that may exist for whatever reason
    std::vector<std::string> subfolders; subfolders.resize(0);
    get_subdirectories_in(path_to_root_directory, subfolders);
    char temp_backup_folder_to_delete[PATH_MAX]; int to_delete_idx = 0;
    for (size_t idx = 0; idx < subfolders.size(); ++idx) {
      if(!subfolders[idx].compare(0, 7, "backup_"))
      {
        int read_idx;
        sscanf(subfolders[idx].c_str(), "backup_%d", &read_idx);
        if(read_idx >= n_saved)
        {
          // delete extra backups existing for whatever reasons (renamed to temporary folders beforehand to avoid issues)
          char full_path[PATH_MAX];
          sprintf(full_path, "%s/%s", path_to_root_directory, subfolders[idx].c_str());
          sprintf(temp_backup_folder_to_delete, "%s/temp_backup_folder_to_delete_%d", path_to_root_directory, to_delete_idx++);
          rename(full_path, temp_backup_folder_to_delete);
          delete_directory(temp_backup_folder_to_delete, p4est_n->mpirank, p4est_n->mpicomm, true);
        }
        else
          n_backup_subfolders++;
      }
    }

    // check that they are successively indexed if less than the max number
    if(n_backup_subfolders < n_saved)
    {
      backup_idx = 0;
      for (int idx = 0; idx < n_backup_subfolders; ++idx) {
        char expected_dir[PATH_MAX];
        sprintf(expected_dir, "%s/backup_%d", path_to_root_directory, idx);
        if(!is_folder(expected_dir))
          break; // well, it's a mess in there, but I can't really do any better...
        backup_idx++;
      }
    }
    if (n_saved > 1 && n_backup_subfolders == n_saved)
    {
      // delete the 0th (renamed to a temporary folder beforehand to avoid issues)
      char full_path_zeroth_index[PATH_MAX];
      sprintf(full_path_zeroth_index, "%s/backup_0", path_to_root_directory);
      sprintf(temp_backup_folder_to_delete, "%s/temp_backup_folder_to_delete_%d", path_to_root_directory, to_delete_idx++);
      rename(full_path_zeroth_index, temp_backup_folder_to_delete);
      delete_directory(temp_backup_folder_to_delete, p4est_n->mpirank, p4est_n->mpicomm, true);
      // shift the others
      for (int idx = 1; idx < n_saved; ++idx) {
        char old_name[PATH_MAX], new_name[PATH_MAX];
        sprintf(old_name, "%s/backup_%d", path_to_root_directory, (int) idx);
        sprintf(new_name, "%s/backup_%d", path_to_root_directory, (int) (idx - 1));
        rename(old_name, new_name);
      }
      backup_idx = n_saved - 1;
    }
  }
  int mpiret = MPI_Bcast(&backup_idx, 1, MPI_INT, 0, p4est_n->mpicomm); SC_CHECK_MPI(mpiret);// acts as an MPI_Barrier, too

  char path_to_folder[PATH_MAX];
  sprintf(path_to_folder, "%s/backup_%d", path_to_root_directory, (int) backup_idx);
  create_directory(path_to_folder, p4est_n->mpirank, p4est_n->mpicomm);


  char filename[PATH_MAX];
  // save the solver parameters
  sprintf(filename, "%s/solver_parameters", path_to_folder);
  save_or_load_parameters(filename, (splitting_criteria_t*) p4est_n->user_pointer, (fine_p4est_n != NULL ? (splitting_criteria_t*) fine_p4est_n->user_pointer : NULL), SAVE, tn);
  // save p4est_n and all corresponding data
  if(fine_p4est_n != NULL || phi_on_computational_nodes == NULL)
    throw std::runtime_error("my_p4est_two_phase_flows_t::save_state: cannot handle the use of subrefining grids yet...");

  my_p4est_save_forest_and_data(path_to_folder, p4est_n, nodes_n, faces_n,
                                "p4est_n", 3,
                                "phi_comp", 1, &phi_on_computational_nodes,
                                "vn_nodes_minus", 1, &vn_nodes_minus,
                                "vn_nodes_plus", 1, &vn_nodes_plus);

  // save p4est_nm1
  my_p4est_save_forest_and_data(path_to_folder, p4est_nm1, nodes_nm1,
                                "p4est_nm1", 2,
                                "vnm1_nodes_minus", 1, &vnm1_nodes_minus,
                                "vnm1_nodes_plus", 1, &vnm1_nodes_plus);
  PetscErrorCode ierr = PetscPrintf(p4est_n->mpicomm, "Saved solver state in ... %s\n", path_to_folder); CHKERRXX(ierr);
}

void my_p4est_two_phase_flows_t::fill_or_load_double_parameters(save_or_load flag, std::vector<PetscReal>& data, splitting_criteria_t *splitting_criterion, splitting_criteria_t* fine_splitting_criterion, double &tn)
{
  size_t idx = 0;
  for (u_char dim = 0; dim < P4EST_DIM; ++dim)
  {
    switch (flag) {
    case SAVE:
      data[idx++] = tree_dimension[dim];
      break;
    case LOAD:
      tree_dimension[dim] = data[idx++];
      break;
    default:
      throw std::runtime_error("my_p4est_two_phase_flows_t::fill_or_load_double_data: unknown flag value");
      break;
    }
  }
  for (u_char dim = 0; dim < P4EST_DIM; ++dim)
  {
    switch (flag) {
    case SAVE:
      data[idx++] = dxyz_smallest_quad[dim];
      break;
    case LOAD:
      dxyz_smallest_quad[dim] = data[idx++];
      break;
    default:
      throw std::runtime_error("my_p4est_two_phase_flows_t::fill_or_load_double_data: unknown flag value");
      break;
    }
  }
  {
    switch (flag) {
    case SAVE:
    {
      data[idx++] = surface_tension;
      data[idx++] = mu_minus;
      data[idx++] = mu_plus;
      data[idx++] = rho_minus;
      data[idx++] = rho_plus;
      data[idx++] = tn;
      data[idx++] = dt_n;
      data[idx++] = dt_nm1;
      data[idx++] = max_L2_norm_velocity_minus;
      data[idx++] = max_L2_norm_velocity_plus;
      data[idx++] = uniform_band_minus;
      data[idx++] = uniform_band_plus;
      data[idx++] = threshold_split_cell;
      data[idx++] = cfl_advection;
      data[idx++] = cfl_visco_capillary;
      data[idx++] = cfl_capillary;
      data[idx++] = splitting_criterion->lip;
      data[idx++] = (fine_splitting_criterion != NULL ? fine_splitting_criterion->lip : splitting_criterion->lip);
      break;
    }
    case LOAD:
    {
      surface_tension = data[idx++];
      mu_minus = data[idx++];
      mu_plus = data[idx++];
      rho_minus = data[idx++];
      rho_plus = data[idx++];
      tn = data[idx++];
      dt_n = data[idx++];
      dt_nm1 = data[idx++];
      max_L2_norm_velocity_minus = data[idx++];
      max_L2_norm_velocity_plus = data[idx++];
      uniform_band_minus = data[idx++];
      uniform_band_plus = data[idx++];
      threshold_split_cell = data[idx++];
      cfl_advection = data[idx++];
      cfl_visco_capillary = data[idx++];
      cfl_capillary = data[idx++];
      splitting_criterion->lip = data[idx++];
      (fine_splitting_criterion != NULL ? fine_splitting_criterion->lip : splitting_criterion->lip) = data[idx++];
      break;
    }
    default:
      throw std::runtime_error("my_p4est_two_phase_flows_t::fill_or_load_double_data: unknown flag value");
      break;
    }
  }
  P4EST_ASSERT(idx == data.size());
}

void my_p4est_two_phase_flows_t::fill_or_load_integer_parameters(save_or_load flag, std::vector<PetscInt>& data, splitting_criteria_t* splitting_criterion, splitting_criteria_t* fine_splitting_criterion)
{
  size_t idx = 0;
  switch (flag) {
  case SAVE:
  {
    data[idx++] = P4EST_DIM;
    data[idx++] = (PetscInt) cell_jump_solver_to_use;
    data[idx++] = (PetscInt) fetch_interface_FD_neighbors_with_second_order_accuracy;
    data[idx++] = splitting_criterion->min_lvl;
    data[idx++] = splitting_criterion->max_lvl;
    data[idx++] = (fine_splitting_criterion != NULL ? fine_splitting_criterion->min_lvl : splitting_criterion->min_lvl);
    data[idx++] = (fine_splitting_criterion != NULL ? fine_splitting_criterion->max_lvl : splitting_criterion->max_lvl);
    data[idx++] = (PetscInt) levelset_interpolation_method;
    data[idx++] = sl_order;
    data[idx++] = sl_order_interface;
    data[idx++] = n_viscous_subiterations;
    data[idx++] = (PetscInt) voronoi_on_the_fly;
    break;
  }
  case LOAD:
  {
    PetscInt P4EST_DIM_COPY       = data[idx++];
    if(P4EST_DIM_COPY != P4EST_DIM)
      throw std::runtime_error("my_p4est_two_phase_flows_t::fill_or_load_integer_parameters(...): you're trying to load 2D (resp. 3D) data with a 3D (resp. 2D) program...");
    cell_jump_solver_to_use = (jump_solver_tag) data[idx++];
    fetch_interface_FD_neighbors_with_second_order_accuracy = (bool) data[idx++];
    splitting_criterion->min_lvl  = data[idx++];
    splitting_criterion->max_lvl  = data[idx++];
    (fine_splitting_criterion != NULL ? fine_splitting_criterion->min_lvl : splitting_criterion->min_lvl) = data[idx++];
    (fine_splitting_criterion != NULL ? fine_splitting_criterion->max_lvl : splitting_criterion->max_lvl) = data[idx++];
    levelset_interpolation_method = (interpolation_method) data[idx++];
    sl_order = data[idx++];
    sl_order_interface = data[idx++];
    n_viscous_subiterations = data[idx++];
    voronoi_on_the_fly = (bool) data[idx++];
    break;
  }
  default:
    throw std::runtime_error("my_p4est_two_phase_flows_t::fill_or_load_integer_data: unknown flag value");
    break;
  }
  P4EST_ASSERT(idx == data.size());
}

void my_p4est_two_phase_flows_t::save_or_load_parameters(const char* filename, splitting_criteria_t* splitting_criterion, splitting_criteria_t* fine_splitting_criterion, save_or_load flag, double &tn, const mpi_environment_t* mpi)
{
  PetscErrorCode ierr;
  // tree_dimension, dxyz_smallest_quad, surface_tension, mu_minus, mu_plus, rho_minus, rho_plus,
  // tn, dt_n, dt_nm1, max_L2_norm_velocity_minus, max_L2_norm_velocity_plus, uniform_band_minus, uniform_band_plus,
  // threshold_split_cell, cfl_advection, cfl_visco_capillary, cfl_capillary, splitting_criterion->lip, fine_splitting_criterion->lip
  // that makes 2*P4EST_DIM + 18 doubles to save
  const size_t ndouble_values =  2*P4EST_DIM + 18;
  std::vector<PetscReal> double_parameters(ndouble_values);
  // P4EST_DIM, cell_jump_solver_to_use, fetch_interface_FD_neighbors_with_second_order_accuracy, data->min_lvl, data->max_lvl,
  // fine_data->min_lvl, fine_data->max_lvl, levelset_interpolation_method, sl_order, sl_order_interface, n_viscous_subiterations
  // voronoi_on_the_fly
  // that makes 12 integers
  const size_t ninteger_values = 12;
  std::vector<PetscInt> integer_parameters(ninteger_values);
  int fd;
  char diskfilename[PATH_MAX];
  switch (flag) {
  case SAVE:
  {
    if(p4est_n->mpirank == 0)
    {
      sprintf(diskfilename, "%s_integers", filename);
      fill_or_load_integer_parameters(flag, integer_parameters, splitting_criterion, fine_splitting_criterion);
      ierr = PetscBinaryOpen(diskfilename, FILE_MODE_WRITE, &fd); CHKERRXX(ierr);
      ierr = PetscBinaryWrite(fd, integer_parameters.data(), integer_parameters.size(), PETSC_INT, PETSC_TRUE); CHKERRXX(ierr);
      ierr = PetscBinaryClose(fd); CHKERRXX(ierr);
      // Then we save the double parameters
      sprintf(diskfilename, "%s_doubles", filename);
      fill_or_load_double_parameters(flag, double_parameters, splitting_criterion, fine_splitting_criterion, tn);
      ierr = PetscBinaryOpen(diskfilename, FILE_MODE_WRITE, &fd); CHKERRXX(ierr);
      ierr = PetscBinaryWrite(fd, double_parameters.data(), double_parameters.size(), PETSC_DOUBLE, PETSC_TRUE); CHKERRXX(ierr);
      ierr = PetscBinaryClose(fd); CHKERRXX(ierr);
    }
    break;
  }
  case LOAD:
  {
    sprintf(diskfilename, "%s_integers", filename);
    if(!file_exists(diskfilename))
      throw std::invalid_argument("my_p4est_two_phase_flows_t::save_or_load_parameters: the file storing the solver's integer parameters could not be found");
    if(mpi->rank() == 0)
    {
      ierr = PetscBinaryOpen(diskfilename, FILE_MODE_READ, &fd); CHKERRXX(ierr);
      ierr = PetscBinaryRead(fd, integer_parameters.data(), integer_parameters.size(), PETSC_INT); CHKERRXX(ierr);
      ierr = PetscBinaryClose(fd); CHKERRXX(ierr);
    }
    int mpiret = MPI_Bcast(integer_parameters.data(), ninteger_values, MPIU_INT, 0, mpi->comm()); SC_CHECK_MPI(mpiret); // "MPIU_INT" so that it still works if PetSc uses 64-bit integers (correct MPI type defined in Petscsys.h for you!)
    fill_or_load_integer_parameters(flag, integer_parameters, splitting_criterion, fine_splitting_criterion);
    // Then we save the double parameters
    sprintf(diskfilename, "%s_doubles", filename);
    if(!file_exists(diskfilename))
      throw std::invalid_argument("my_p4est_two_phase_flows_t::save_or_load_parameters: the file storing the solver's double parameters could not be found");
    if(mpi->rank() == 0)
    {
      ierr = PetscBinaryOpen(diskfilename, FILE_MODE_READ, &fd); CHKERRXX(ierr);
      ierr = PetscBinaryRead(fd, double_parameters.data(), double_parameters.size(), PETSC_DOUBLE); CHKERRXX(ierr);
      ierr = PetscBinaryClose(fd); CHKERRXX(ierr);
    }
    mpiret = MPI_Bcast(double_parameters.data(), ndouble_values, MPI_DOUBLE, 0, mpi->comm()); SC_CHECK_MPI(mpiret);
    fill_or_load_double_parameters(flag, double_parameters, splitting_criterion, fine_splitting_criterion, tn);
    break;
  }
  default:
    throw std::runtime_error("my_p4est_two_phase_flows_t::save_or_load_parameters: unknown flag value");
    break;
    break;
  }
}

my_p4est_two_phase_flows_t::~my_p4est_two_phase_flows_t()
{
  PetscErrorCode ierr;
  // if using subcell resolution, you'll want to take care of this separately
  if(phi_on_computational_nodes != phi) {
    ierr = delete_and_nullify_vector(phi_on_computational_nodes);       CHKERRXX(ierr); }
  // node-sampled fields on the interface-capturing grid
  ierr = delete_and_nullify_vector(phi);                                CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(jump_normal_velocity);               CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(non_viscous_pressure_jump);          CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(phi_xxyyzz);                         CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(interface_tangential_stress);        CHKERRXX(ierr);
  // node-sampled fields on the computational grids n
  ierr = delete_and_nullify_vector(vorticity_magnitude_minus);          CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(vorticity_magnitude_plus);           CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(vnp1_nodes_minus);                   CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(vnp1_nodes_plus);                    CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(vn_nodes_minus);                     CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(vn_nodes_plus);                      CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(interface_velocity_np1);             CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(vn_nodes_minus_xxyyzz);              CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(vn_nodes_plus_xxyyzz);               CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(interface_velocity_np1_xxyyzz);      CHKERRXX(ierr);
  // cell-sampled fields on the computational grids n
  ierr = delete_and_nullify_vector(pressure_minus);                     CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(pressure_plus);                      CHKERRXX(ierr);
  // node-sampled fields on the computational grids nm1
  ierr = delete_and_nullify_vector(interface_velocity_n);               CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(interface_velocity_n_xxyyzz);        CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(vnm1_nodes_minus);                   CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(vnm1_nodes_plus);                    CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(vnm1_nodes_minus_xxyyzz);            CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(vnm1_nodes_plus_xxyyzz);             CHKERRXX(ierr);
  // face-sampled fields, computational grid n
  for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = delete_and_nullify_vector(vnp1_face_minus_km1[dir]);         CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(vnp1_face_plus_km1[dir]);          CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(vnp1_face_minus[dir]);             CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(vnp1_face_plus[dir]);              CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(viscosity_rhs_minus[dir]);         CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(viscosity_rhs_plus[dir]);          CHKERRXX(ierr);
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
  if(cell_jump_solver != NULL)
    delete cell_jump_solver;
  if(face_jump_solver != NULL)
    delete face_jump_solver;
}

void my_p4est_two_phase_flows_t::build_face_jump_solver()
{
  if(face_jump_solver != NULL)
    return;

  switch (face_jump_solver_to_use) {
  case xGFM:
  case GFM:
    face_jump_solver = new my_p4est_poisson_jump_faces_xgfm_t(faces_n, nodes_n);
    dynamic_cast<my_p4est_poisson_jump_faces_xgfm_t*>(face_jump_solver)->activate_xGFM_corrections(face_jump_solver_to_use == xGFM);
    break;
  default:
    throw std::runtime_error("my_p4est_two_phase_flows_t::build_face_jump_solver(): unkown face jump solver, only (x)GFM solvers are implemented for face problems as of now");
    break;
  }
  return;
}

void my_p4est_two_phase_flows_t::build_cell_jump_solver()
{
  if(cell_jump_solver != NULL)
    return;

  switch (cell_jump_solver_to_use) {
  case xGFM:
  case GFM:
    cell_jump_solver = new my_p4est_poisson_jump_cells_xgfm_t(ngbd_c, nodes_n);
    dynamic_cast<my_p4est_poisson_jump_cells_xgfm_t*>(cell_jump_solver)->activate_xGFM_corrections(cell_jump_solver_to_use == xGFM);
    break;
  case FV:
    cell_jump_solver = new my_p4est_poisson_jump_cells_fv_t(ngbd_c, nodes_n);
    break;
  default:
    throw std::runtime_error("my_p4est_two_phase_flows_t::build_cell_jump_solver(): unkown cell jump solver, only (x)GFM and FV solvers are implemented for face problems as of now");
    break;
  }
  return;
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

void my_p4est_two_phase_flows_t::compute_non_viscous_pressure_jump()
{
  PetscErrorCode ierr;
  if(non_viscous_pressure_jump == NULL){
    ierr = interface_manager->create_vector_on_interface_capturing_nodes(non_viscous_pressure_jump); CHKERRXX(ierr);
  }

  // [p] = -surface_tension*curvature - SQR(mass_flux)*jump_of_inverse_mass_density + [2\mu n \cdot E \cdot n]
  // we calculate the two first terms in here (known before any time step)

  const double* curvature_p = NULL;
  const double* jump_normal_velocity_p = NULL;

  if(jump_normal_velocity != NULL && !mass_densities_are_equal()){
    ierr = VecGetArrayRead(jump_normal_velocity, &jump_normal_velocity_p); CHKERRXX(ierr); }
  if(fabs(surface_tension) > EPS || jump_normal_velocity_p != NULL)
  {
    if(!interface_manager->is_curvature_set())
      interface_manager->set_curvature(); // Maybe we'd want to flatten it...

    ierr = VecGetArrayRead(interface_manager->get_curvature(), &curvature_p); CHKERRXX(ierr);
  }

  double* non_viscous_pressure_jump_p;
  ierr = VecGetArray(non_viscous_pressure_jump, &non_viscous_pressure_jump_p); CHKERRXX(ierr);
  const my_p4est_node_neighbors_t& interface_capturing_ngbd_n = interface_manager->get_interface_capturing_ngbd_n();
  for (size_t k = 0; k < interface_capturing_ngbd_n.get_layer_size(); ++k) {
    const p4est_locidx_t node_idx = interface_capturing_ngbd_n.get_layer_node(k);
    non_viscous_pressure_jump_p[node_idx] = 0.0;
    if(fabs(surface_tension) > EPS)
      non_viscous_pressure_jump_p[node_idx] -= surface_tension*curvature_p[node_idx];
    if(jump_normal_velocity_p != NULL)
      non_viscous_pressure_jump_p[node_idx] -= SQR(jump_normal_velocity_p[node_idx])/jump_inverse_mass_density();
  }
  ierr = VecGhostUpdateBegin(non_viscous_pressure_jump, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for (size_t k = 0; k < interface_capturing_ngbd_n.get_local_size(); ++k) {
    const p4est_locidx_t node_idx = interface_capturing_ngbd_n.get_local_node(k);
    non_viscous_pressure_jump_p[node_idx] = 0.0;
    if(fabs(surface_tension) > EPS)
      non_viscous_pressure_jump_p[node_idx] -= surface_tension*curvature_p[node_idx];
    if(jump_normal_velocity_p != NULL)
      non_viscous_pressure_jump_p[node_idx] -= SQR(jump_normal_velocity_p[node_idx])/jump_inverse_mass_density();
  }
  ierr = VecGhostUpdateEnd(non_viscous_pressure_jump, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecRestoreArray(non_viscous_pressure_jump, &non_viscous_pressure_jump_p); CHKERRXX(ierr);

  if(jump_normal_velocity_p != NULL){
    ierr = VecRestoreArrayRead(jump_normal_velocity, &jump_normal_velocity_p); CHKERRXX(ierr); }
  if(curvature_p != NULL){
    ierr = VecRestoreArrayRead(interface_manager->get_curvature(), &curvature_p); CHKERRXX(ierr); }

  return;
}

/* solve the pressure guess equation:
 * -div((1.0/rho)*grad(p_guess)) = 0.0
 * jump in pressure guess = -surface_tension*kappa  - SQR(mass_flux)*jump_inverse_mass_density()
 *                           + jump in (mu n*E*n) <-- this one is estimated with vnp1_face_*_km1
 * jump_normal_flux = 0.0
 */
void my_p4est_two_phase_flows_t::solve_for_pressure_guess()
{
  if(pressure_guess_is_set)
    return;
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_two_phase_flows_solve_pressure_guess, 0, 0, 0, 0); CHKERRXX(ierr);

  /* Solve for the pressure guess: */
  if(cell_jump_solver == NULL)
  {
    build_cell_jump_solver();
    cell_jump_solver->set_interface(interface_manager);
    cell_jump_solver->set_diagonals(0.0, 0.0);
    cell_jump_solver->set_mus(1.0/rho_minus, 1.0/rho_plus);
    cell_jump_solver->set_shear_viscosities(mu_minus, mu_plus); // required for the last terms in pressure jumps
    if(bc_pressure != NULL)
      cell_jump_solver->set_bc(*bc_pressure);
  }

  compute_non_viscous_pressure_jump();
  cell_jump_solver->set_jumps(non_viscous_pressure_jump, NULL); /* NULL for the jump in (1/rho normal derivative of p) because we don't know't do any better */
  if(ANDD(vnp1_face_minus_km1[0] != NULL, vnp1_face_minus_km1[1] != NULL, vnp1_face_minus_km1[2] != NULL) &&
     ANDD(vnp1_face_plus_km1[0] != NULL,  vnp1_face_plus_km1[1] != NULL,  vnp1_face_plus_km1[2] != NULL))
    cell_jump_solver->set_face_velocities_km1(vnp1_face_minus_km1, vnp1_face_plus_km1);
  else
    cell_jump_solver->set_face_velocities_km1(NULL, NULL);

  // no "set_rhs" here, we consider the homogeneous problem
  cell_jump_solver->solve(Krylov_solver_for_cell_problems, preconditioner_for_cell_problems);
  cell_jump_solver->extrapolate_solution_from_either_side_to_the_other(npseudo_time_steps());

  if(pressure_minus != NULL){
    ierr = delete_and_nullify_vector(pressure_minus); CHKERRXX(ierr); }
  if(pressure_plus != NULL){
    ierr = delete_and_nullify_vector(pressure_plus); CHKERRXX(ierr); }

  pressure_minus  = cell_jump_solver->return_extrapolated_solution_minus();
  pressure_plus   = cell_jump_solver->return_extrapolated_solution_plus();

  pressure_guess_is_set = true;

  ierr = PetscLogEventEnd(log_my_p4est_two_phase_flows_solve_pressure_guess, 0, 0, 0, 0); CHKERRXX(ierr);
  return;
}

/* solve the projection step, we consider (PHI/rho) to be the HODGE variable, define the divergence-free projection as
 * v^{n + 1} = v^{\star} - (1.0/rho) grad PHI, and we solve for PHI as the solution of
 * -div((1.0/rho)*grad(PHI)) = -div(vstar)
 * jump_PHI = (dt/alpha)*(jump in (mu*n*E*n) - jump in (mu*n*E_km1*n) + jump in (mu div v))
 * jump_normal_flux = 0.0! --> because we assume the normal jump in u_star has been correctly captured earlier on...
 */
void my_p4est_two_phase_flows_t::solve_projection()
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_two_phase_flows_solve_projection, 0, 0, 0, 0); CHKERRXX(ierr);
  /* Make the two-phase velocity field divergence free (and correct pressure): */
  if(cell_jump_solver == NULL)
    throw std::runtime_error("my_p4est_two_phase_flows_t::solve_projection(): the cell jump solver has *NOT* been constructed yet, have you call solve_for_pressure_guess beforehand? ");

  cell_jump_solver->set_for_projection_steps(dt_n/BDF_advection_alpha());
  // If not previously done, the above does
  // - clear the node-sampled jumps (surface tension + recoil), internally to the solver
  // - makes the memorized jump-related data homoegeneous
  // - raise a flag that
  //  1) activates the definition of jumps thereafter as
  //              (dt/alpha)*(jump in (mu n_dot_E_dot_n) - jump in (mu n_dot_Ekm1_dot_n) + jump in (div u))
  //  2) keeps the boundary condition types as originally given (i.e. as for pressure guess) makes them homogeneous

  my_p4est_interpolation_nodes_t* interp_jump_normal_velocity = NULL;
  if(jump_normal_velocity != NULL)
  {
    interp_jump_normal_velocity = new my_p4est_interpolation_nodes_t(ngbd_n);
    interp_jump_normal_velocity->set_input(jump_normal_velocity, linear);
  }
  if(ANDD(vnp1_face_minus_km1[0] != NULL, vnp1_face_minus_km1[1] != NULL, vnp1_face_minus_km1[2] != NULL) &&
     ANDD(vnp1_face_plus_km1[0] != NULL, vnp1_face_plus_km1[1] != NULL, vnp1_face_plus_km1[2] != NULL))
    cell_jump_solver->set_face_velocities_km1(vnp1_face_minus_km1, vnp1_face_plus_km1);
  else
    cell_jump_solver->set_face_velocities_km1(NULL, NULL);

  if(ORD(vnp1_face_minus[0] == NULL,  vnp1_face_minus[1] == NULL, vnp1_face_minus[2] == NULL) ||
     ORD(vnp1_face_plus[0] == NULL,   vnp1_face_plus[1] == NULL,  vnp1_face_plus[2] == NULL))
    throw std::runtime_error("my_p4est_two_phase_flows_t::solve_projection(): current face-sampled velocities are not defined, what are you making divergence-free?");

  cell_jump_solver->set_velocity_on_faces(vnp1_face_minus, vnp1_face_plus, interp_jump_normal_velocity);
  cell_jump_solver->solve(Krylov_solver_for_cell_problems, preconditioner_for_cell_problems);

  // extrapolate the solutions from either side so that the projection can be done for ghost-values velocities, as well
  cell_jump_solver->extrapolate_solution_from_either_side_to_the_other(npseudo_time_steps());

  // --------------------------------------
  // --- start updating pressure fields ---
  // (done BEFORE we project the velocity field, because the correction term "mu div vnp1_face_*" requires the not-yet-updated face-velocity)
  if(pressure_minus == NULL || pressure_plus == NULL)
    throw  std::runtime_error("my_p4est_two_phase_flows_t::solve_projection(): the pressure field is not defined yet...");
  const double *projection_variable_minus_p, *projection_variable_plus_p;
  const double *vnp1_face_plus_p[P4EST_DIM], *vnp1_face_minus_p[P4EST_DIM];
  double *pressure_minus_p, *pressure_plus_p;
  ierr = VecGetArrayRead(cell_jump_solver->get_extrapolated_solution_minus(), &projection_variable_minus_p);  CHKERRXX(ierr);
  ierr = VecGetArrayRead(cell_jump_solver->get_extrapolated_solution_plus(),  &projection_variable_plus_p);   CHKERRXX(ierr);
  ierr = VecGetArray(pressure_minus,  &pressure_minus_p); CHKERRXX(ierr);
  ierr = VecGetArray(pressure_plus,   &pressure_plus_p);  CHKERRXX(ierr);
  for(u_char dim = 0; dim < P4EST_DIM; dim++)
  {
    ierr = VecGetArrayRead(vnp1_face_minus[dim],  &vnp1_face_minus_p[dim]); CHKERRXX(ierr);
    ierr = VecGetArrayRead(vnp1_face_plus[dim],   &vnp1_face_plus_p[dim]);  CHKERRXX(ierr);
  }

  const double alpha_over_dt = BDF_advection_alpha()/dt_n;
  linear_combination_of_dof_t cell_divergence[P4EST_DIM];
  for(size_t k = 0; k < hierarchy_n->get_layer_size(); k++)
  {
    p4est_locidx_t quad_idx = hierarchy_n->get_local_index_of_layer_quadrant(k);
    p4est_topidx_t tree_idx = hierarchy_n->get_tree_index_of_layer_quadrant(k);
    cell_jump_solver->get_divergence_operator_on_cell(quad_idx, tree_idx, cell_divergence);
    pressure_plus_p[quad_idx]   += alpha_over_dt*projection_variable_plus_p[quad_idx]   - mu_plus *SUMD(cell_divergence[0](vnp1_face_plus_p[0]),  cell_divergence[1](vnp1_face_plus_p[1]),  cell_divergence[2](vnp1_face_plus_p[2]));
    pressure_minus_p[quad_idx]  += alpha_over_dt*projection_variable_minus_p[quad_idx]  - mu_minus*SUMD(cell_divergence[0](vnp1_face_minus_p[0]), cell_divergence[1](vnp1_face_minus_p[1]), cell_divergence[2](vnp1_face_minus_p[2]));
  }
  ierr = VecGhostUpdateBegin(pressure_minus,  INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateBegin(pressure_plus,   INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for(size_t k = 0; k < hierarchy_n->get_inner_size(); k++)
  {
    p4est_locidx_t quad_idx = hierarchy_n->get_local_index_of_inner_quadrant(k);
    p4est_topidx_t tree_idx = hierarchy_n->get_tree_index_of_inner_quadrant(k);
    cell_jump_solver->get_divergence_operator_on_cell(quad_idx, tree_idx, cell_divergence);
    pressure_plus_p[quad_idx]   += alpha_over_dt*projection_variable_plus_p[quad_idx]   - mu_plus *SUMD(cell_divergence[0](vnp1_face_plus_p[0]),  cell_divergence[1](vnp1_face_plus_p[1]),  cell_divergence[2](vnp1_face_plus_p[2]));
    pressure_minus_p[quad_idx]  += alpha_over_dt*projection_variable_minus_p[quad_idx]  - mu_minus*SUMD(cell_divergence[0](vnp1_face_minus_p[0]), cell_divergence[1](vnp1_face_minus_p[1]), cell_divergence[2](vnp1_face_minus_p[2]));
  }
  ierr = VecGhostUpdateEnd(pressure_minus,  INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(pressure_plus,   INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecRestoreArrayRead(cell_jump_solver->get_extrapolated_solution_minus(), &projection_variable_minus_p);  CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(cell_jump_solver->get_extrapolated_solution_plus(),  &projection_variable_plus_p);   CHKERRXX(ierr);
  ierr = VecRestoreArray(pressure_minus,  &pressure_minus_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(pressure_plus,   &pressure_plus_p);  CHKERRXX(ierr);
  for(u_char dim = 0; dim < P4EST_DIM; dim++)
  {
    ierr = VecRestoreArrayRead(vnp1_face_minus[dim],  &vnp1_face_minus_p[dim]); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(vnp1_face_plus[dim],   &vnp1_face_plus_p[dim]);  CHKERRXX(ierr);
  }
  // ------- done updating pressure -------
  // --------------------------------------

  cell_jump_solver->project_face_velocities(faces_n, NULL, NULL, max_velocity_correction_in_projection);

  if(interp_jump_normal_velocity != NULL)
    delete interp_jump_normal_velocity;

  ierr = PetscLogEventEnd(log_my_p4est_two_phase_flows_solve_projection, 0, 0, 0, 0); CHKERRXX(ierr);
  return;
}

void my_p4est_two_phase_flows_t::solve_time_step(const double& velocity_relative_threshold, const int& max_niter)
{
  PetscErrorCode ierr;
  max_velocity_correction_in_projection[0] = max_velocity_correction_in_projection[1] = DBL_MAX;
  int iter = 0;
  while((max_velocity_correction_in_projection[0] > velocity_relative_threshold*max_velocity_component_before_projection[0] ||
         max_velocity_correction_in_projection[1] > velocity_relative_threshold*max_velocity_component_before_projection[1])
        && iter < max_niter)
  {
    solve_viscosity(); // will do the pressure guess calculation in the very first pass, if needed
    solve_projection();

    ierr = PetscPrintf(p4est_n->mpicomm, " -------- Internal iteration #%02d -------- \n"); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est_n->mpicomm, " --- In negative domain --- \n"); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est_n->mpicomm, " \t\t\t max velocity component before projection: %.5e \n", max_velocity_component_before_projection[0]); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est_n->mpicomm, " \t\t\t max velocity correction in projection: %.5e \n", max_velocity_correction_in_projection[0]); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est_n->mpicomm, " --- In positive domain --- \n"); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est_n->mpicomm, " \t\t\t max velocity component before projection: %.5e \n", max_velocity_component_before_projection[1]); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est_n->mpicomm, " \t\t\t max velocity correction in projection: %.5e \n", max_velocity_correction_in_projection[1]); CHKERRXX(ierr);
  }
  return;
}

void my_p4est_two_phase_flows_t::set_cell_jump_solver(const jump_solver_tag& solver_to_use, const KSPType& KSP_, const PCType& PC_)
{
  cell_jump_solver_to_use = solver_to_use;
  if(strcmp(KSP_, "default") != 0)
    Krylov_solver_for_cell_problems = KSP_;
  else
  {
    switch (cell_jump_solver_to_use) {
    case GFM:
    case xGFM:
      Krylov_solver_for_cell_problems = KSPCG; // the matrix is always SPD in those case, CG is the best
      break;
    case FV:
      Krylov_solver_for_cell_problems = (mass_densities_are_equal() ? KSPCG : KSPBCGS); // matrix is SPD iff rho_minus == rho_plus;
    default:
      throw std::runtime_error("my_p4est_two_phase_flows_t::set_cell_jump_solver(): unknown solver type for cell-based jump problems");
      break;
    }
  }

  if(strcmp(PC_, "default") != 0)
    preconditioner_for_cell_problems = PC_;
  else
    preconditioner_for_cell_problems = PCHYPRE; // we might need all the help we can get

  return;
}


void my_p4est_two_phase_flows_t::set_face_jump_solvers(const jump_solver_tag& solver_to_use,  const KSPType& KSP_, const PCType& PC_)
{
  face_jump_solver_to_use = solver_to_use;
  if(strcmp(KSP_, "default") != 0)
    Krylov_solver_for_face_problems = KSP_;
  else
  {
    switch (face_jump_solver_to_use) {
    case GFM:
    case xGFM:
      Krylov_solver_for_face_problems = KSPCG; // the matrix is always SPD in those case, CG is the best
      break;
    case FV:
      Krylov_solver_for_face_problems = (viscosities_are_equal() ? KSPCG : KSPBCGS); // matrix is SPD iff mu_minus == mu_plus;
    default:
      throw std::runtime_error("my_p4est_two_phase_flows_t::set_face_jump_solvers(): unknown solver type for face-based jump problems");
      break;
    }
  }

  if(strcmp(PC_, "default") != 0)
    preconditioner_for_face_problems = PC_;
  else
    preconditioner_for_face_problems = PCSOR; // we have nonzero diagonal terms in this case, no need to be anal about it

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
  compute_backtraced_velocities();
  compute_viscosity_rhs();

  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_two_phase_flows_solve_viscosity, 0, 0, 0, 0); CHKERRXX(ierr);
  if(face_jump_solver == NULL)
  {
    build_face_jump_solver();
    face_jump_solver->set_interface(interface_manager);
    face_jump_solver->set_mus(mu_minus, mu_plus);
    face_jump_solver->set_bc(bc_velocity);
    face_jump_solver->set_jumps(jump_normal_velocity, interface_tangential_stress);
    face_jump_solver->set_compute_partition_on_the_fly(voronoi_on_the_fly);
  }

  face_jump_solver->set_rhs(viscosity_rhs_minus, viscosity_rhs_plus);
  face_jump_solver->set_max_number_of_iter(n_viscous_subiterations);
  face_jump_solver->set_diagonals(BDF_advection_alpha()*rho_minus/dt_n, BDF_advection_alpha()*rho_plus/dt_n);
  face_jump_solver->solve(Krylov_solver_for_face_problems, preconditioner_for_face_problems);
  face_jump_solver->extrapolate_solution_from_either_side_to_the_other(npseudo_time_steps(), 1, max_velocity_component_before_projection);

  P4EST_ASSERT((ANDD(vnp1_face_minus[0] == NULL, vnp1_face_minus[1] == NULL, vnp1_face_minus[2] == NULL) &&
      ANDD(vnp1_face_plus[0] == NULL, vnp1_face_plus[1] == NULL, vnp1_face_plus[2] == NULL)) ||
      (ANDD(vnp1_face_minus[0] != NULL, vnp1_face_minus[1] != NULL, vnp1_face_minus[2] != NULL) &&
      ANDD(vnp1_face_plus[0] != NULL, vnp1_face_plus[1] != NULL, vnp1_face_plus[2] != NULL))); // either all undefined or all defined...
  if(ANDD(vnp1_face_minus[0] != NULL, vnp1_face_minus[1] != NULL, vnp1_face_minus[2] != NULL) &&
     ANDD(vnp1_face_plus[0] != NULL,  vnp1_face_plus[1] != NULL,  vnp1_face_plus[2] != NULL))
  {
    for(u_char dim = 0; dim < P4EST_DIM; dim++)
    {
      // no longer need km1 (if we had it)
      ierr = delete_and_nullify_vector(vnp1_face_minus_km1[dim]); CHKERRXX(ierr);
      ierr = delete_and_nullify_vector(vnp1_face_plus_km1[dim]);  CHKERRXX(ierr);
      // swap current and km1 (current is NULL until the face jump solver returns its knowledge herebelow)
      std::swap(vnp1_face_minus_km1[dim], vnp1_face_minus[dim]);
      std::swap(vnp1_face_plus_km1[dim], vnp1_face_plus[dim]);
    }
  }
  face_jump_solver->return_ownership_of_extrapolations(vnp1_face_minus, vnp1_face_plus);
  ierr = PetscLogEventEnd(log_my_p4est_two_phase_flows_solve_viscosity, 0, 0, 0, 0); CHKERRXX(ierr);
  return;
}

void my_p4est_two_phase_flows_t::compute_velocities_at_nodes()
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_two_phase_flows_interpolate_velocity_at_nodes, 0, 0, 0, 0); CHKERRXX(ierr);

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

  ierr = PetscLogEventEnd(log_my_p4est_two_phase_flows_interpolate_velocity_at_nodes, 0, 0, 0, 0); CHKERRXX(ierr);

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
          neumann_wall[dd] = (is_wall[2*dd] ? -1 : (is_wall[2*dd + 1] ? +1 : 0));
          nb_neumann_walls += abs(neumann_wall[dd]);
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

#ifdef P4EST_DEBUG
  for (u_char dim = 0; dim < P4EST_DIM; ++dim)
    P4EST_ASSERT(!ISNAN(vnp1_nodes_minus_p[P4EST_DIM*node_idx + dim]) && !ISNAN(vnp1_nodes_plus_p[P4EST_DIM*node_idx + dim]));
#endif

  if (interface_manager->phi_at_point(xyz_node) <= 0.0)
    max_L2_norm_velocity_minus = MAX(max_L2_norm_velocity_minus, sqrt(magnitude_velocity_minus));
  if (interface_manager->phi_at_point(xyz_node) >= 0.0)
    max_L2_norm_velocity_plus = MAX(max_L2_norm_velocity_plus, sqrt(magnitude_velocity_plus));

  return;
}

//void my_p4est_two_phase_flows_t::TVD_extrapolation_of_np1_node_velocities(const u_int& niterations, const u_char& order)
//{
//  PetscErrorCode ierr;
//  Vec vnp1_nodes_minus_no_block[P4EST_DIM], vnp1_nodes_plus_no_block[P4EST_DIM], normal_vector[P4EST_DIM];
//  double *vnp1_nodes_minus_no_block_p[P4EST_DIM], *vnp1_nodes_plus_no_block_p[P4EST_DIM], *normal_vector_p[P4EST_DIM];
//  double *phi_on_computational_nodes_p = NULL;
//  bool phi_on_computational_nodes_locally_created = false;
//  P4EST_ASSERT(interface_manager->subcell_resolution() != 0 || phi_on_computational_nodes != NULL);
//  if(phi_on_computational_nodes == NULL)
//  {
//    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &phi_on_computational_nodes); CHKERRXX(ierr);
//    ierr = VecGetArray(phi_on_computational_nodes, &phi_on_computational_nodes_p); CHKERRXX(ierr);
//    phi_on_computational_nodes_locally_created = true;
//  }
//  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
//    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &vnp1_nodes_minus_no_block[dim]);  CHKERRXX(ierr);
//    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &vnp1_nodes_plus_no_block[dim]);   CHKERRXX(ierr);
//    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &normal_vector[dim]);              CHKERRXX(ierr);
//    ierr = VecGetArray(vnp1_nodes_minus_no_block[dim], &vnp1_nodes_minus_no_block_p[dim]);  CHKERRXX(ierr);
//    ierr = VecGetArray(vnp1_nodes_plus_no_block[dim], &vnp1_nodes_plus_no_block_p[dim]);    CHKERRXX(ierr);
//    ierr = VecGetArray(normal_vector[dim], &normal_vector_p[dim]); CHKERRXX(ierr);
//  }
//  double *vnp1_nodes_minus_p, *vnp1_nodes_plus_p;
//  const double *grad_phi_p = NULL;
//  if(interface_manager->subcell_resolution() == 0){
//    ierr= VecGetArrayRead(interface_manager->get_grad_phi(), &grad_phi_p); CHKERRXX(ierr); }
//  double xyz_node[P4EST_DIM];
//  ierr = VecGetArray(vnp1_nodes_minus, &vnp1_nodes_minus_p); CHKERRXX(ierr);
//  ierr = VecGetArray(vnp1_nodes_plus, &vnp1_nodes_plus_p); CHKERRXX(ierr);
//  for (size_t node_idx = 0; node_idx < nodes_n->indep_nodes.elem_count; ++node_idx) {
//    for (u_char dim = 0; dim < P4EST_DIM; ++dim)
//    {
//      vnp1_nodes_minus_no_block_p[dim][node_idx] = vnp1_nodes_minus_p[P4EST_DIM*node_idx + dim];
//      vnp1_nodes_plus_no_block_p[dim][node_idx] = vnp1_nodes_plus_p[P4EST_DIM*node_idx + dim];
//    }
//    if(node_idx < (size_t) nodes_n->num_owned_indeps)
//    {
//      if(interface_manager->subcell_resolution() > 0)
//      {
//        P4EST_ASSERT(grad_phi_p == NULL);
//        node_xyz_fr_n(node_idx, p4est_n, nodes_n, xyz_node);
//        if(phi_on_computational_nodes_p != NULL)
//          phi_on_computational_nodes_p[node_idx] = interface_manager->phi_at_point(xyz_node);
//        double local_normal[P4EST_DIM];
//        interface_manager->normal_vector_at_point(xyz_node, local_normal);
//        for (u_char dim = 0; dim < P4EST_DIM; ++dim)
//          normal_vector_p[dim][node_idx] = local_normal[dim];
//      }
//      else
//      {
//        P4EST_ASSERT(phi_on_computational_nodes_p == NULL && grad_phi_p != NULL);
//        const double magnitude = ABSD(grad_phi_p[P4EST_DIM*node_idx], grad_phi_p[P4EST_DIM*node_idx + 1], grad_phi_p[P4EST_DIM*node_idx + 2]);
//        for (u_char dim = 0; dim < P4EST_DIM; ++dim)
//          normal_vector_p[dim][node_idx] = (magnitude > EPS ? grad_phi_p[P4EST_DIM*node_idx + dim]/magnitude : 0.0);
//      }
//    }
//  }
//  if(interface_manager->subcell_resolution() > 0){
//    ierr = VecGhostUpdateBegin(phi_on_computational_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//    ierr = VecGhostUpdateEnd(phi_on_computational_nodes, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  }
//  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
//    ierr = VecGhostUpdateBegin(normal_vector[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//    ierr = VecGhostUpdateEnd(normal_vector[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  }

//  if(interface_manager->subcell_resolution() == 0){
//    ierr = VecRestoreArrayRead(interface_manager->get_grad_phi(), &grad_phi_p); CHKERRXX(ierr); }
//  ierr = VecRestoreArray(vnp1_nodes_minus, &vnp1_nodes_minus_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(vnp1_nodes_plus, &vnp1_nodes_plus_p); CHKERRXX(ierr);
//  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
//    ierr = VecRestoreArray(normal_vector[dim], &normal_vector_p[dim]); CHKERRXX(ierr);
//    ierr = VecRestoreArray(vnp1_nodes_plus_no_block[dim], &vnp1_nodes_plus_no_block_p[dim]); CHKERRXX(ierr);
//    ierr = VecRestoreArray(vnp1_nodes_minus_no_block[dim], &vnp1_nodes_minus_no_block_p[dim]); CHKERRXX(ierr);
//  }
//  if(phi_on_computational_nodes_p != NULL){
//    ierr = VecRestoreArray(phi_on_computational_nodes, &phi_on_computational_nodes_p); CHKERRXX(ierr); }

//  my_p4est_level_set_t ls_nodes(ngbd_n);

//  for (char sgn = -1; sgn <= 1; sgn += 2) {
//    if(sgn > 0)
//    {
//      ierr = VecScaleGhost(phi_on_computational_nodes, -1.0); CHKERRXX(ierr);
//      for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
//        ierr = VecScaleGhost(normal_vector[dim], -1.0); CHKERRXX(ierr); }
//    }
//    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
//      Vec node_velocity_component_to_extrapolate = (sgn < 0 ? vnp1_nodes_minus_no_block[dim] : vnp1_nodes_plus_no_block[dim]);
//      ls_nodes.extend_Over_Interface_TVD(phi_on_computational_nodes, node_velocity_component_to_extrapolate, niterations, order, 0.0, -DBL_MAX, +DBL_MAX, DBL_MAX, normal_vector);
//    }
//  }

//  ierr = VecScaleGhost(phi_on_computational_nodes, -1.0); CHKERRXX(ierr);
//  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
//    ierr = VecScaleGhost(normal_vector[dim], -1.0); CHKERRXX(ierr); }

//  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
//    ierr = VecGetArray(vnp1_nodes_minus_no_block[dim], &vnp1_nodes_minus_no_block_p[dim]); CHKERRXX(ierr);
//    ierr = VecGetArray(vnp1_nodes_plus_no_block[dim], &vnp1_nodes_plus_no_block_p[dim]); CHKERRXX(ierr);
//  }
//  ierr = VecGetArray(vnp1_nodes_minus, &vnp1_nodes_minus_p); CHKERRXX(ierr);
//  ierr = VecGetArray(vnp1_nodes_plus, &vnp1_nodes_plus_p); CHKERRXX(ierr);
//  for (size_t node_idx = 0; node_idx < nodes_n->indep_nodes.elem_count; ++node_idx) {
//    for (u_char dim = 0; dim < P4EST_DIM; ++dim)
//    {
//      vnp1_nodes_minus_p[P4EST_DIM*node_idx + dim] = vnp1_nodes_minus_no_block_p[dim][node_idx];
//      vnp1_nodes_plus_p[P4EST_DIM*node_idx + dim] = vnp1_nodes_plus_no_block_p[dim][node_idx];
//    }
//  }
//  ierr = VecRestoreArray(vnp1_nodes_minus, &vnp1_nodes_minus_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(vnp1_nodes_plus, &vnp1_nodes_plus_p); CHKERRXX(ierr);
//  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
//    ierr = VecRestoreArray(vnp1_nodes_minus_no_block[dim], &vnp1_nodes_minus_no_block_p[dim]); CHKERRXX(ierr);
//    ierr = VecRestoreArray(vnp1_nodes_plus_no_block[dim], &vnp1_nodes_plus_no_block_p[dim]); CHKERRXX(ierr);
//    ierr = VecRestoreArray(normal_vector[dim], &normal_vector_p[dim]); CHKERRXX(ierr);
//    ierr = delete_and_nullify_vector(vnp1_nodes_minus_no_block[dim]); CHKERRXX(ierr);
//    ierr = delete_and_nullify_vector(vnp1_nodes_plus_no_block[dim]); CHKERRXX(ierr);
//    ierr = delete_and_nullify_vector(normal_vector[dim]); CHKERRXX(ierr);
//  }
//  if(phi_on_computational_nodes_locally_created){
//    ierr = delete_and_nullify_vector(phi_on_computational_nodes); CHKERRXX(ierr); }

//  return;
//}

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

void my_p4est_two_phase_flows_t::transfer_face_sampled_vnp1_to_cells(Vec vnp1_minus_on_cells, Vec vnp1_plus_on_cells) const
{
  PetscErrorCode ierr;
  const double *vnp1_face_minus_p[P4EST_DIM]= {DIM(NULL, NULL, NULL)}; double *vnp1_minus_on_cells_p  = NULL;
  const double *vnp1_face_plus_p[P4EST_DIM] = {DIM(NULL, NULL, NULL)}; double *vnp1_plus_on_cells_p   = NULL;
  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecGetArrayRead(vnp1_face_minus[dim], &vnp1_face_minus_p[dim]); CHKERRXX(ierr);
    ierr = VecGetArrayRead(vnp1_face_plus[dim], &vnp1_face_plus_p[dim]); CHKERRXX(ierr);
  }
  ierr = VecGetArray(vnp1_minus_on_cells, &vnp1_minus_on_cells_p); CHKERRXX(ierr);
  ierr = VecGetArray(vnp1_plus_on_cells, &vnp1_plus_on_cells_p); CHKERRXX(ierr);

  for (size_t k = 0; k < ngbd_c->get_hierarchy()->get_layer_size(); ++k) {
    p4est_locidx_t quad_idx = ngbd_c->get_hierarchy()->get_local_index_of_layer_quadrant(k);
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      p4est_locidx_t face_idx = faces_n->q2f(quad_idx, 2*dim);
      if(face_idx != NO_VELOCITY){
        vnp1_minus_on_cells_p[P4EST_DIM*quad_idx + dim] = vnp1_face_minus_p[dim][face_idx];
        vnp1_plus_on_cells_p[P4EST_DIM*quad_idx + dim] = vnp1_face_plus_p[dim][face_idx];
      }
      else
      {
        set_of_neighboring_quadrants nb_quad;
        ngbd_c->find_neighbor_cells_of_cell(nb_quad, quad_idx,tree_index_of_quad(quad_idx, p4est_n, ghost_n), 2*dim);
        vnp1_minus_on_cells_p[P4EST_DIM*quad_idx + dim] = vnp1_plus_on_cells_p[P4EST_DIM*quad_idx + dim] = 0.0;
        double sum_w = 0.0;
        for (set_of_neighboring_quadrants::const_iterator it = nb_quad.begin(); it != nb_quad.end(); ++it) {
          double w = 1.0/((double) (1 << (P4EST_DIM - 1)*it->level));
          vnp1_minus_on_cells_p[P4EST_DIM*quad_idx + dim] += vnp1_face_minus_p[dim][faces_n->q2f(it->p.piggy3.local_num, 2*dim + 1)]*w;
          vnp1_plus_on_cells_p[P4EST_DIM*quad_idx + dim] += vnp1_face_plus_p[dim][faces_n->q2f(it->p.piggy3.local_num, 2*dim + 1)]*w;
          sum_w += w;
        }
        vnp1_minus_on_cells_p[P4EST_DIM*quad_idx + dim] /= sum_w;
        vnp1_plus_on_cells_p[P4EST_DIM*quad_idx + dim] /= sum_w;
      }
    }
  }

  ierr = VecGhostUpdateBegin(vnp1_minus_on_cells, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateBegin(vnp1_plus_on_cells, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  for (size_t k = 0; k < ngbd_c->get_hierarchy()->get_inner_size(); ++k) {
    p4est_locidx_t quad_idx = ngbd_c->get_hierarchy()->get_local_index_of_inner_quadrant(k);
    for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
      p4est_locidx_t face_idx = faces_n->q2f(quad_idx, 2*dim);
      if(face_idx != NO_VELOCITY){
        vnp1_minus_on_cells_p[P4EST_DIM*quad_idx + dim] = vnp1_face_minus_p[dim][face_idx];
        vnp1_plus_on_cells_p[P4EST_DIM*quad_idx + dim] = vnp1_face_plus_p[dim][face_idx];
      }
      else
      {
        set_of_neighboring_quadrants nb_quad;
        ngbd_c->find_neighbor_cells_of_cell(nb_quad, quad_idx,tree_index_of_quad(quad_idx, p4est_n, ghost_n), 2*dim);
        vnp1_minus_on_cells_p[P4EST_DIM*quad_idx + dim] = vnp1_plus_on_cells_p[P4EST_DIM*quad_idx + dim] = 0.0;
        double sum_w = 0.0;
        for (set_of_neighboring_quadrants::const_iterator it = nb_quad.begin(); it != nb_quad.end(); ++it) {
          double w = 1.0/((double) (1 << (P4EST_DIM - 1)*it->level));
          vnp1_minus_on_cells_p[P4EST_DIM*quad_idx + dim] += vnp1_face_minus_p[dim][faces_n->q2f(it->p.piggy3.local_num, 2*dim + 1)]*w;
          vnp1_plus_on_cells_p[P4EST_DIM*quad_idx + dim] += vnp1_face_plus_p[dim][faces_n->q2f(it->p.piggy3.local_num, 2*dim + 1)]*w;
          sum_w += w;
        }
        vnp1_minus_on_cells_p[P4EST_DIM*quad_idx + dim] /= sum_w;
        vnp1_plus_on_cells_p[P4EST_DIM*quad_idx + dim] /= sum_w;
      }
    }
  }

  ierr = VecGhostUpdateEnd(vnp1_minus_on_cells, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(vnp1_plus_on_cells, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  for (u_char dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecRestoreArrayRead(vnp1_face_minus[dim], &vnp1_face_minus_p[dim]); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(vnp1_face_plus[dim], &vnp1_face_plus_p[dim]); CHKERRXX(ierr);
  }
  ierr = VecRestoreArray(vnp1_minus_on_cells, &vnp1_minus_on_cells_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(vnp1_plus_on_cells, &vnp1_plus_on_cells_p); CHKERRXX(ierr);
  return;
}

void my_p4est_two_phase_flows_t::save_vtk(const std::string& vtk_directory, const int& index, const bool& exhaustive) const
{
  PetscErrorCode ierr;
  std::vector<Vec_for_vtk_export_t> node_scalar_fields;
  std::vector<Vec_for_vtk_export_t> node_vector_fields;
  std::vector<Vec_for_vtk_export_t> cell_scalar_fields;
  std::vector<Vec_for_vtk_export_t> cell_vector_fields;

  // those are the primary variables of interest, we should always export them:
  if(phi_on_computational_nodes != NULL)
    node_scalar_fields.push_back(Vec_for_vtk_export_t(phi_on_computational_nodes, "phi"));
  if(vnp1_nodes_minus != NULL)
    node_vector_fields.push_back(Vec_for_vtk_export_t(vnp1_nodes_minus, "vnp1_minus"));
  if(vnp1_nodes_plus != NULL)
    node_vector_fields.push_back(Vec_for_vtk_export_t(vnp1_nodes_plus, "vnp1_plus"));
  if(interface_velocity_np1 != NULL)
    node_vector_fields.push_back(Vec_for_vtk_export_t(interface_velocity_np1, "itfc_vnp1"));
  if(pressure_minus != NULL)
    cell_scalar_fields.push_back(Vec_for_vtk_export_t(pressure_minus, "pressure_minus"));
  if(pressure_plus != NULL)
    cell_scalar_fields.push_back(Vec_for_vtk_export_t(pressure_plus, "pressure_plus"));

  if(interface_manager->subcell_resolution() == 0)
  {
    node_scalar_fields.push_back(Vec_for_vtk_export_t(interface_manager->get_curvature(), "curvature"));
    node_vector_fields.push_back(Vec_for_vtk_export_t(interface_manager->get_grad_phi(), "grad_phi"));
    if(exhaustive && non_viscous_pressure_jump != NULL)
      node_scalar_fields.push_back(Vec_for_vtk_export_t(non_viscous_pressure_jump, "non_viscous_pressure_jump"));
    if(exhaustive && jump_normal_velocity != NULL)
      node_scalar_fields.push_back(Vec_for_vtk_export_t(jump_normal_velocity, "jump_normal_velocity"));
    if(exhaustive && interface_tangential_stress != NULL)
      node_vector_fields.push_back(Vec_for_vtk_export_t(interface_tangential_stress, "interface_tangential_stress"));
  }

  Vec vnp1_minus_on_cells = NULL;
  Vec vnp1_plus_on_cells = NULL;
  Vec discretized_div_u_star = NULL;
  if(exhaustive)
  {
    if(cell_jump_solver->is_set_for_projection_step)
    {
      if(cell_jump_solver->get_solution() != NULL)
        cell_scalar_fields.push_back(Vec_for_vtk_export_t(cell_jump_solver->get_solution(), "latest_sharp_projection_variable"));
      if(cell_jump_solver->get_extrapolated_solution_minus() != NULL)
        cell_scalar_fields.push_back(Vec_for_vtk_export_t(cell_jump_solver->get_extrapolated_solution_minus(), "latest_projection_variable_minus"));
      if(cell_jump_solver->get_extrapolated_solution_plus() != NULL)
        cell_scalar_fields.push_back(Vec_for_vtk_export_t(cell_jump_solver->get_extrapolated_solution_plus(), "latest_projection_variable_plus"));

      if(cell_jump_solver->get_rhs() != NULL)
      {
        ierr = VecCreateGhostCells(p4est_n, ghost_n, &discretized_div_u_star); CHKERRXX(ierr);
        ierr = VecCopy(cell_jump_solver->get_rhs(), discretized_div_u_star); CHKERRXX(ierr);
        ierr = VecGhostUpdateBegin(discretized_div_u_star, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        ierr = VecGhostUpdateEnd(discretized_div_u_star, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        cell_scalar_fields.push_back(Vec_for_vtk_export_t(discretized_div_u_star, "discrete_div_u_star"));
      }
    }

    if(ANDD(vnp1_face_minus[0] != NULL, vnp1_face_minus[1] != NULL, vnp1_face_minus[2] != NULL) &&
       ANDD(vnp1_face_plus[0] != NULL, vnp1_face_plus[1] != NULL, vnp1_face_plus[2] != NULL))
    {
      ierr = VecCreateGhostCellsBlock(p4est_n, ghost_n, P4EST_DIM, &vnp1_minus_on_cells); CHKERRXX(ierr);
      ierr = VecCreateGhostCellsBlock(p4est_n, ghost_n, P4EST_DIM, &vnp1_plus_on_cells); CHKERRXX(ierr);
      transfer_face_sampled_vnp1_to_cells(vnp1_minus_on_cells, vnp1_plus_on_cells);
    }

    cell_vector_fields.push_back(Vec_for_vtk_export_t(vnp1_minus_on_cells,  "vnp1_minus"));
    cell_vector_fields.push_back(Vec_for_vtk_export_t(vnp1_plus_on_cells,   "vnp1_plus"));
  }

  my_p4est_vtk_write_all_general_lists(p4est_n, nodes_n, ghost_n,
                                       P4EST_TRUE, P4EST_TRUE,
                                       (vtk_directory + "/snapshot_" + std::to_string(index)).c_str(),
                                       &node_scalar_fields, &node_vector_fields, &cell_scalar_fields, &cell_vector_fields);

  ierr = delete_and_nullify_vector(discretized_div_u_star); CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(vnp1_minus_on_cells); CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(vnp1_plus_on_cells); CHKERRXX(ierr);

  if(interface_manager->subcell_resolution() > 0)
  {
    node_scalar_fields.clear();
    node_vector_fields.clear();
    node_scalar_fields.push_back(Vec_for_vtk_export_t(phi, "phi"));
    node_scalar_fields.push_back(Vec_for_vtk_export_t(interface_manager->get_curvature(), "curvature"));
    node_vector_fields.push_back(Vec_for_vtk_export_t(interface_manager->get_grad_phi(), "grad_phi"));
    if(exhaustive && non_viscous_pressure_jump != NULL)
      node_scalar_fields.push_back(Vec_for_vtk_export_t(non_viscous_pressure_jump, "non_viscous_pressure_jump"));
    if(exhaustive && jump_normal_velocity != NULL)
      node_scalar_fields.push_back(Vec_for_vtk_export_t(jump_normal_velocity, "jump_normal_velocity"));
    if(exhaustive && interface_tangential_stress != NULL)
      node_vector_fields.push_back(Vec_for_vtk_export_t(interface_tangential_stress, "interface_tangential_stress"));

    my_p4est_vtk_write_all_general_lists(fine_p4est_n, fine_nodes_n, fine_ghost_n,
                                         P4EST_TRUE, P4EST_TRUE,
                                         (vtk_directory + "/subresolved_snapshot_" + std::to_string(index)).c_str(),
                                         &node_scalar_fields, &node_vector_fields, NULL, NULL);
  }
  // restore the pointers (in destructors of Vec_for_vtk_export_t)
  node_scalar_fields.clear();
  node_vector_fields.clear();
  cell_scalar_fields.clear();
  cell_vector_fields.clear();

  ierr = PetscPrintf(p4est_n->mpicomm, "Saved visual data in ... %s (snapshot %d)\n", vtk_directory.c_str(), index); CHKERRXX(ierr);
  return;
}

void my_p4est_two_phase_flows_t::compute_backtraced_velocities()
{
  if(semi_lagrangian_backtrace_is_done)
    return;
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_two_phase_flows_compute_backtracing, 0, 0, 0, 0); CHKERRXX(ierr);

  P4EST_ASSERT((vnm1_nodes_minus_xxyyzz == NULL && vnm1_nodes_plus_xxyyzz == NULL && vn_nodes_minus_xxyyzz == NULL && vn_nodes_plus_xxyyzz == NULL) ||
               (vnm1_nodes_minus_xxyyzz != NULL && vnm1_nodes_plus_xxyyzz != NULL && vn_nodes_minus_xxyyzz != NULL && vn_nodes_plus_xxyyzz != NULL));

  /* first find the velocity n at the np1 points */
  my_p4est_interpolation_nodes_t interp_np1(ngbd_n);
  const bool use_second_derivatives_n   = (vn_nodes_minus_xxyyzz    != NULL && vn_nodes_plus_xxyyzz   != NULL);
  const bool use_second_derivatives_nm1 = (vnm1_nodes_minus_xxyyzz  != NULL && vnm1_nodes_plus_xxyyzz != NULL);
  const p4est_locidx_t serialized_offset[P4EST_DIM] = {DIM(0, faces_n->num_local[0], faces_n->num_local[0] + faces_n->num_local[1])};
  const size_t total_number_of_faces = SUMD(faces_n->num_local[0], faces_n->num_local[1], faces_n->num_local[2]);

  vector<double> xyz_np1(P4EST_DIM*total_number_of_faces);    // coordinates of the origin point, i.e. face centers --> same for minus and plus...
  vector<double> vnp1_minus(P4EST_DIM*total_number_of_faces);
  vector<double> vnp1_plus(P4EST_DIM*total_number_of_faces);

  for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
    for (p4est_locidx_t f_idx = 0; f_idx < faces_n->num_local[dir]; ++f_idx) {
      double xyz_face[P4EST_DIM];
      faces_n->xyz_fr_f(f_idx, dir, xyz_face);
      for (u_char comp = 0; comp < P4EST_DIM; ++comp)
        xyz_np1[P4EST_DIM*(serialized_offset[dir] + f_idx) + comp] = xyz_face[comp];

      interp_np1.add_point(serialized_offset[dir] + f_idx, xyz_face);
    }
  }

  Vec input_fields[2]         = {vn_nodes_minus,        vn_nodes_plus};
  Vec input_fields_xxyyzz[2]  = {vn_nodes_minus_xxyyzz, vn_nodes_plus_xxyyzz};
  double *outputs[2]          = {vnp1_minus.data(),     vnp1_plus.data()};

  if (use_second_derivatives_n)
    interp_np1.set_input(input_fields, input_fields_xxyyzz, quadratic, 2, P4EST_DIM);
  else
    interp_np1.set_input(input_fields, quadratic, 2, P4EST_DIM);
  interp_np1.interpolate(outputs);

  /* find xyz_star */
  my_p4est_interpolation_nodes_t interp_nm1_minus(ngbd_nm1), interp_nm1_plus(ngbd_nm1);
  my_p4est_interpolation_nodes_t interp_n_minus  (ngbd_n  ), interp_n_plus  (ngbd_n  );
  for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
    for (p4est_locidx_t f_idx = 0; f_idx < faces_n->num_local[dir]; ++f_idx) {
      double xyz_star_minus[P4EST_DIM], xyz_star_plus[P4EST_DIM];
      for (u_char comp = 0; comp < P4EST_DIM; ++comp)
      {
        xyz_star_minus[comp]  = xyz_np1[P4EST_DIM*(serialized_offset[dir] + f_idx) + comp] - 0.5*dt_n*vnp1_minus[P4EST_DIM*(serialized_offset[dir] + f_idx) + comp];
        xyz_star_plus[comp]   = xyz_np1[P4EST_DIM*(serialized_offset[dir] + f_idx) + comp] - 0.5*dt_n*vnp1_plus[P4EST_DIM*(serialized_offset[dir] + f_idx) + comp];
      }
      clip_in_domain(xyz_star_minus, xyz_min, xyz_max, periodicity);
      clip_in_domain(xyz_star_plus, xyz_min, xyz_max, periodicity);

      interp_nm1_minus.add_point(serialized_offset[dir] + f_idx, xyz_star_minus);
      interp_nm1_plus.add_point(serialized_offset[dir] + f_idx, xyz_star_plus);

      interp_n_minus.add_point(serialized_offset[dir] + f_idx, xyz_star_minus);
      interp_n_plus.add_point(serialized_offset[dir] + f_idx, xyz_star_plus);
    }
  }

  /* compute the velocities at the intermediate point */
  std::vector<double> vn_star_minus(P4EST_DIM*total_number_of_faces);
  std::vector<double> vn_star_plus(P4EST_DIM*total_number_of_faces);
  std::vector<double> vnm1_star_minus(P4EST_DIM*total_number_of_faces);
  std::vector<double> vnm1_star_plus(P4EST_DIM*total_number_of_faces);
  if(use_second_derivatives_n)
  {
    interp_n_minus.set_input  (vn_nodes_minus,    vn_nodes_minus_xxyyzz,    quadratic, P4EST_DIM);
    interp_n_plus.set_input   (vn_nodes_plus,     vn_nodes_plus_xxyyzz,     quadratic, P4EST_DIM);
  }
  else
  {
    interp_n_minus.set_input  (vn_nodes_minus,    quadratic, P4EST_DIM);
    interp_n_plus.set_input   (vn_nodes_plus,     quadratic, P4EST_DIM);
  }
  if(use_second_derivatives_nm1)
  {
    interp_nm1_minus.set_input(vnm1_nodes_minus,  vnm1_nodes_minus_xxyyzz,  quadratic, P4EST_DIM);
    interp_nm1_plus.set_input (vnm1_nodes_plus,   vnm1_nodes_plus_xxyyzz,   quadratic, P4EST_DIM);
  }
  else
  {
    interp_nm1_minus.set_input(vnm1_nodes_minus,  quadratic, P4EST_DIM);
    interp_nm1_plus.set_input (vnm1_nodes_plus,   quadratic, P4EST_DIM);
  }
  interp_nm1_minus.interpolate(vnm1_star_minus.data()); interp_nm1_minus.clear();
  interp_nm1_plus.interpolate(vnm1_star_plus.data());   interp_nm1_plus.clear();
  interp_n_minus.interpolate(vn_star_minus.data());     interp_n_minus.clear();
  interp_n_plus.interpolate(vn_star_plus.data());       interp_n_plus.clear();

  /* now find the departure point at time n and interpolate the appropriate velocity component there */
  for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
    backtraced_vn_faces_minus[dir].resize(faces_n->num_local[dir]);
    backtraced_vn_faces_plus[dir].resize(faces_n->num_local[dir]);
    for (p4est_locidx_t f_idx = 0; f_idx < faces_n->num_local[dir]; ++f_idx) {
      double xyz_backtraced_n_minus[P4EST_DIM], xyz_backtraced_n_plus[P4EST_DIM];
      for (u_char comp = 0; comp < P4EST_DIM; ++comp)
      {
        xyz_backtraced_n_minus[comp] = xyz_np1[P4EST_DIM*(serialized_offset[dir] + f_idx) + comp]
            - dt_n*((1.0 + .5*dt_n/dt_nm1)*vn_star_minus[P4EST_DIM*(serialized_offset[dir] + f_idx) + comp] - .5*(dt_n/dt_nm1)*vnm1_star_minus[P4EST_DIM*(serialized_offset[dir] + f_idx) + comp]);
        xyz_backtraced_n_plus[comp] = xyz_np1[P4EST_DIM*(serialized_offset[dir] + f_idx) + comp]
            - dt_n*((1.0 + .5*dt_n/dt_nm1)*vn_star_plus[P4EST_DIM*(serialized_offset[dir] + f_idx) + comp] - .5*(dt_n/dt_nm1)*vnm1_star_plus[P4EST_DIM*(serialized_offset[dir] + f_idx) + comp]);
      }
      clip_in_domain(xyz_backtraced_n_minus, xyz_min, xyz_max, periodicity);
      clip_in_domain(xyz_backtraced_n_plus, xyz_min, xyz_max, periodicity);
      interp_n_minus.add_point(f_idx, xyz_backtraced_n_minus);
      interp_n_plus.add_point(f_idx, xyz_backtraced_n_plus);
    }
    interp_n_minus.interpolate(backtraced_vn_faces_minus[dir].data(), dir); interp_n_minus.clear();
    interp_n_plus.interpolate(backtraced_vn_faces_plus[dir].data(),  dir);  interp_n_plus.clear();
  }

  // EXTRA STUFF FOR FINDING xyz_nm1 ONLY (for second-order bdf advection terms, for instance)
  if(sl_order == 2)
  {
    /* proceed similarly for the departure point at time nm1 */
    for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
      for (p4est_locidx_t f_idx = 0; f_idx < faces_n->num_local[dir]; ++f_idx) {
        double xyz_star_minus[P4EST_DIM], xyz_star_plus[P4EST_DIM];
        for (u_char comp = 0; comp < P4EST_DIM; ++comp)
        {
          xyz_star_minus[comp]  = xyz_np1[P4EST_DIM*(serialized_offset[dir] + f_idx) + comp] - 0.5*(dt_n + dt_nm1)*vnp1_minus[P4EST_DIM*(serialized_offset[dir] + f_idx) + comp];
          xyz_star_plus[comp]   = xyz_np1[P4EST_DIM*(serialized_offset[dir] + f_idx) + comp] - 0.5*(dt_n + dt_nm1)*vnp1_plus[P4EST_DIM*(serialized_offset[dir] + f_idx) + comp];
        }
        clip_in_domain(xyz_star_minus, xyz_min, xyz_max, periodicity);
        clip_in_domain(xyz_star_plus, xyz_min, xyz_max, periodicity);

        interp_nm1_minus.add_point(serialized_offset[dir] + f_idx, xyz_star_minus);
        interp_nm1_plus.add_point(serialized_offset[dir] + f_idx, xyz_star_plus);
        interp_n_minus.add_point(serialized_offset[dir] + f_idx, xyz_star_minus);
        interp_n_plus.add_point(serialized_offset[dir] + f_idx, xyz_star_plus);
      }
    }

    /* compute the velocities at the intermediate point */
    interp_nm1_minus.interpolate(vnm1_star_minus.data()); interp_nm1_minus.clear();
    interp_nm1_plus.interpolate(vnm1_star_plus.data());   interp_nm1_plus.clear();
    interp_n_minus.interpolate(vn_star_minus.data());     interp_n_minus.clear();
    interp_n_plus.interpolate(vn_star_plus.data());       interp_n_plus.clear();

    /* now find the departure point at time nm1 and interpolate the appropriate velocity component there */
    for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
      backtraced_vnm1_faces_minus[dir].resize(faces_n->num_local[dir]);
      backtraced_vnm1_faces_plus[dir].resize(faces_n->num_local[dir]);
      for (p4est_locidx_t f_idx = 0; f_idx < faces_n->num_local[dir]; ++f_idx) {
        double xyz_backtraced_nm1_minus[P4EST_DIM], xyz_backtraced_nm1_plus[P4EST_DIM];
        for (u_char comp = 0; comp < P4EST_DIM; ++comp)
        {
          xyz_backtraced_nm1_minus[comp] = xyz_np1[P4EST_DIM*(serialized_offset[dir] + f_idx) + comp]
              - (dt_n + dt_nm1)*((1.0 + .5*(dt_n - dt_nm1)/dt_nm1)*vn_star_minus[P4EST_DIM*(serialized_offset[dir] + f_idx) + comp] - .5*((dt_n - dt_nm1)/dt_nm1)*vnm1_star_minus[P4EST_DIM*(serialized_offset[dir] + f_idx) + comp]);
          xyz_backtraced_nm1_plus[comp] = xyz_np1[P4EST_DIM*(serialized_offset[dir] + f_idx) + comp]
              - (dt_n + dt_nm1)*((1.0 + .5*(dt_n - dt_nm1)/dt_nm1)*vn_star_plus[P4EST_DIM*(serialized_offset[dir] + f_idx) + comp] - .5*((dt_n - dt_nm1)/dt_nm1)*vnm1_star_plus[P4EST_DIM*(serialized_offset[dir] + f_idx) + comp]);
        }
        clip_in_domain(xyz_backtraced_nm1_minus, xyz_min, xyz_max, periodicity);
        clip_in_domain(xyz_backtraced_nm1_plus, xyz_min, xyz_max, periodicity);

        interp_nm1_minus.add_point(f_idx, xyz_backtraced_nm1_minus);
        interp_nm1_plus.add_point(f_idx, xyz_backtraced_nm1_plus);
      }
      interp_nm1_minus.interpolate(backtraced_vnm1_faces_minus[dir].data(), dir); interp_nm1_minus.clear();
      interp_nm1_plus.interpolate(backtraced_vnm1_faces_plus[dir].data(), dir);   interp_nm1_plus.clear();
    }
  }
  semi_lagrangian_backtrace_is_done = true;
  ierr = PetscLogEventEnd(log_my_p4est_two_phase_flows_compute_backtracing, 0, 0, 0, 0); CHKERRXX(ierr);
  return;
}

void my_p4est_two_phase_flows_t::compute_viscosity_rhs()
{
  if(pressure_minus == NULL || pressure_plus == NULL)
    solve_for_pressure_guess();
  compute_backtraced_velocities();

  P4EST_ASSERT(pressure_minus != NULL && pressure_plus != NULL);

  PetscErrorCode ierr;
  const double *pressure_minus_p, *pressure_plus_p;
  ierr = VecGetArrayRead(pressure_minus, &pressure_minus_p);  CHKERRXX(ierr);
  ierr = VecGetArrayRead(pressure_plus, &pressure_plus_p);    CHKERRXX(ierr);

  for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
    double *viscosity_rhs_minus_dir_p, *viscosity_rhs_plus_dir_p;
    if(viscosity_rhs_minus[dir] == NULL){
      ierr = VecCreateNoGhostFaces(p4est_n, faces_n, &viscosity_rhs_minus[dir], dir); CHKERRXX(ierr);
    }
    if(viscosity_rhs_plus[dir] == NULL){
      ierr = VecCreateNoGhostFaces(p4est_n, faces_n, &viscosity_rhs_plus[dir], dir); CHKERRXX(ierr);
    }
    ierr = VecGetArray(viscosity_rhs_minus[dir],  &viscosity_rhs_minus_dir_p);  CHKERRXX(ierr);
    ierr = VecGetArray(viscosity_rhs_plus[dir],   &viscosity_rhs_plus_dir_p);   CHKERRXX(ierr);

    for(p4est_locidx_t f_idx = 0; f_idx < faces_n->num_local[dir]; ++f_idx)
    {
      if(sl_order == 1)
      {
        viscosity_rhs_minus_dir_p[f_idx]  = (rho_minus/dt_n)*backtraced_vn_faces_minus[dir][f_idx];
        viscosity_rhs_plus_dir_p[f_idx]   = (rho_plus/dt_n)*backtraced_vn_faces_plus[dir][f_idx];
      }
      else
      {
        viscosity_rhs_minus_dir_p[f_idx] = rho_minus*(BDF_advection_alpha()/dt_n - BDF_advection_beta()/dt_nm1)*backtraced_vn_faces_minus[dir][f_idx]
            + rho_minus*(BDF_advection_beta()/dt_nm1)*backtraced_vnm1_faces_minus[dir][f_idx];
        viscosity_rhs_plus_dir_p[f_idx]   = rho_plus*(BDF_advection_alpha()/dt_n - BDF_advection_beta()/dt_nm1)*backtraced_vn_faces_plus[dir][f_idx]
            + rho_plus*(BDF_advection_beta()/dt_nm1)*backtraced_vnm1_faces_plus[dir][f_idx];
      }

      if(force_per_unit_mass[dir] != NULL)
      {
        double xyz_face[P4EST_DIM];
        if(dynamic_cast<const cf_const_t*>(force_per_unit_mass[dir]) == NULL){
          faces_n->xyz_fr_f(f_idx, dir, xyz_face); // get coordinates only if not a stupid constant function...
        }
        viscosity_rhs_minus_dir_p[f_idx]  += rho_minus*(*force_per_unit_mass[dir])(xyz_face);
        viscosity_rhs_plus_dir_p[f_idx]   += rho_plus*(*force_per_unit_mass[dir])(xyz_face);
      }

      // pressure gradient!
      p4est_quadrant_t qm, qp;
      faces_n->find_quads_touching_face(f_idx, dir, qm, qp);
      if(qm.p.piggy3.local_num == -1 || qp.p.piggy3.local_num == -1) // i.e. wall face
      {
        const p4est_quadrant_t& quad  = (qm.p.piggy3.local_num != -1 ? qm : qp);
        const double face_orientation = (qm.p.piggy3.local_num != -1 ? +1.0 : -1.0);
        const double cell_dx = tree_dimension[dir]*((double) P4EST_QUADRANT_LEN(quad.level))/((double) P4EST_ROOT_LEN);
        double xyz_face[P4EST_DIM];
        faces_n->xyz_fr_f(f_idx, dir, xyz_face);
        switch (bc_pressure->wallType(xyz_face)) {
        case DIRICHLET:
          viscosity_rhs_minus_dir_p[f_idx]  -= face_orientation*2.0*(bc_pressure->wallType(xyz_face) - pressure_minus_p[quad.p.piggy3.local_num])/cell_dx;
          viscosity_rhs_plus_dir_p[f_idx]   -= face_orientation*2.0*(bc_pressure->wallType(xyz_face) - pressure_plus_p[quad.p.piggy3.local_num])/cell_dx;
          break;
        case NEUMANN:
          viscosity_rhs_minus_dir_p[f_idx]  -= face_orientation*bc_pressure->wallType(xyz_face);
          viscosity_rhs_plus_dir_p[f_idx]   -= face_orientation*bc_pressure->wallType(xyz_face);
          break;
        default:
          throw std::runtime_error("my_p4est_two_phase_flows_t::compute_viscosity_rhs(): unknown pressure wall boundary type");
          break;
        }
      }
      else
      {
        const p4est_quadrant_t& quad  = (qm.level < qp.level || qm.level == qp.level ? qm : qp);        // we pick the "major" quad to avoid yet another search for neighbors in "stable_projection_..."
        const u_char face_dir         = 2*dir + (qm.level < qp.level || qm.level == qp.level ? 1 : 0);
        set_of_neighboring_quadrants dummy1;
        bool dummy2;
        linear_combination_of_dof_t derivative_operator = cell_jump_solver->stable_projection_derivative_operator_at_face(quad.p.piggy3.local_num, quad.p.piggy3.which_tree, face_dir, dummy1, dummy2);
        viscosity_rhs_minus_dir_p[f_idx]  -= derivative_operator(pressure_minus_p);
        viscosity_rhs_plus_dir_p[f_idx]   -= derivative_operator(pressure_plus_p);
      }
    }

    ierr = VecRestoreArray(viscosity_rhs_minus[dir],  &viscosity_rhs_minus_dir_p);  CHKERRXX(ierr);
    ierr = VecRestoreArray(viscosity_rhs_plus[dir],   &viscosity_rhs_plus_dir_p);   CHKERRXX(ierr);
  }

  ierr = VecRestoreArrayRead(pressure_minus, &pressure_minus_p);  CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(pressure_plus, &pressure_plus_p);    CHKERRXX(ierr);

  return;
}

void my_p4est_two_phase_flows_t::set_interface_velocity_np1()
{
  PetscErrorCode ierr;
  if(interface_velocity_np1 == NULL){
    ierr = VecCreateGhostNodesBlock(p4est_n, nodes_n, P4EST_DIM, &interface_velocity_np1); CHKERRXX(ierr);
  }

  double *interface_velocity_np1_p  = NULL;
  my_p4est_interpolation_nodes_t *interp_jump_normal_velocity = NULL;
  if(jump_normal_velocity != NULL)
  {
    interp_jump_normal_velocity = new my_p4est_interpolation_nodes_t(&interface_manager->get_interface_capturing_ngbd_n());
    interp_jump_normal_velocity->set_input(jump_normal_velocity, linear);
  }
  const double *vnp1_nodes_minus_p, *vnp1_nodes_plus_p;
  ierr = VecGetArray(interface_velocity_np1, &interface_velocity_np1_p);  CHKERRXX(ierr);
  ierr = VecGetArrayRead(vnp1_nodes_plus,   &vnp1_nodes_plus_p);  CHKERRXX(ierr);
  ierr = VecGetArrayRead(vnp1_nodes_minus,  &vnp1_nodes_minus_p); CHKERRXX(ierr);
  double xyz_node[P4EST_DIM];
  double local_normal_vector[P4EST_DIM] = {DIM(0.0, 0.0, 0.0)}; // initialize it so that there is no weird shit going on in case mass_flux == NULL
  for (size_t k = 0; k < ngbd_n->get_layer_size(); ++k) {
    const p4est_locidx_t node_idx = ngbd_n->get_layer_node(k);
    double local_mass_flux = 0.0;
    if(interp_jump_normal_velocity != NULL)
    {
      node_xyz_fr_n(node_idx, p4est_n, nodes_n, xyz_node);
      local_mass_flux = (*interp_jump_normal_velocity)(xyz_node)/jump_inverse_mass_density();
      interface_manager->normal_vector_at_point(xyz_node, local_normal_vector);
    }
    for (u_char dir = 0; dir < P4EST_DIM; ++dir)
      interface_velocity_np1_p[P4EST_DIM*node_idx + dir] = 0.5*((vnp1_nodes_minus_p[P4EST_DIM*node_idx + dir] - local_mass_flux*local_normal_vector[dir]/rho_minus) + (vnp1_nodes_plus_p[P4EST_DIM*node_idx + dir] - local_mass_flux*local_normal_vector[dir]/rho_plus));
  }
  ierr = VecGhostUpdateBegin(interface_velocity_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for (size_t k = 0; k < ngbd_n->get_local_size(); ++k) {
    const p4est_locidx_t node_idx = ngbd_n->get_local_node(k);
    double local_mass_flux = 0.0;
    if(interp_jump_normal_velocity != NULL)
    {
      node_xyz_fr_n(node_idx, p4est_n, nodes_n, xyz_node);
      local_mass_flux = (*interp_jump_normal_velocity)(xyz_node)/jump_inverse_mass_density();
      interface_manager->normal_vector_at_point(xyz_node, local_normal_vector);
    }
    for (u_char dir = 0; dir < P4EST_DIM; ++dir)
      interface_velocity_np1_p[P4EST_DIM*node_idx + dir] = 0.5*((vnp1_nodes_minus_p[P4EST_DIM*node_idx + dir] - local_mass_flux*local_normal_vector[dir]/rho_minus) + (vnp1_nodes_plus_p[P4EST_DIM*node_idx + dir] - local_mass_flux*local_normal_vector[dir]/rho_plus));
  }
  ierr = VecGhostUpdateEnd(interface_velocity_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecRestoreArray(interface_velocity_np1, &interface_velocity_np1_p);  CHKERRXX(ierr);

  // Maybe better to flatten the interface velocity here, before finalization?
  my_p4est_level_set_t ls(ngbd_n);
  Vec flat_interface_velocity_np1;
  ierr = VecCreateGhostNodesBlock(p4est_n, nodes_n, P4EST_DIM, &flat_interface_velocity_np1); CHKERRXX(ierr);
  ls.extend_from_interface_to_whole_domain_TVD(phi_on_computational_nodes, interface_velocity_np1, flat_interface_velocity_np1, 20, NULL, 2, 10, NULL, interface_manager->get_grad_phi(), P4EST_DIM);
  ierr = delete_and_nullify_vector(interface_velocity_np1); CHKERRXX(ierr);
  interface_velocity_np1 = flat_interface_velocity_np1;

  if(interface_velocity_np1_xxyyzz == NULL){
    ierr = VecCreateGhostNodesBlock(p4est_n, nodes_n, SQR_P4EST_DIM, &interface_velocity_np1_xxyyzz); CHKERRXX(ierr);
  }
  ngbd_n->second_derivatives_central(interface_velocity_np1, interface_velocity_np1_xxyyzz, P4EST_DIM);

  ierr = VecRestoreArrayRead(vnp1_nodes_minus,  &vnp1_nodes_minus_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(vnp1_nodes_plus,   &vnp1_nodes_plus_p);  CHKERRXX(ierr);
  if(interp_jump_normal_velocity != NULL)
    delete interp_jump_normal_velocity;

  return;
}

void my_p4est_two_phase_flows_t::advect_interface(const p4est_t *p4est_np1, const p4est_nodes_t *nodes_np1, Vec phi_np1,
                                                  const p4est_nodes_t *known_nodes, Vec known_phi_np1)
{
  PetscErrorCode ierr;

  my_p4est_interpolation_nodes_t interp_np1(ngbd_n); // yes, it's normal: we are in the process of advancing time and we actually are determining phi_np2 on p4est_np1 (since "phi" is actually phi_np1 at all times...)
  interp_np1.set_input(interface_velocity_np1, interface_velocity_np1_xxyyzz, quadratic, P4EST_DIM);
  const bool second_order = (sl_order_interface == 2 && interface_velocity_n != NULL);
  my_p4est_interpolation_nodes_t* interp_n = NULL;
  if(second_order){
    interp_n = new my_p4est_interpolation_nodes_t(ngbd_nm1);
    P4EST_ASSERT(VecIsSetForNodes(interface_velocity_n, ngbd_nm1->get_nodes(), ngbd_nm1->get_p4est()->mpicomm, P4EST_DIM));
    interp_n->set_input(interface_velocity_n, interface_velocity_n_xxyyzz, quadratic, P4EST_DIM);
  }
  P4EST_ASSERT(!second_order || interp_n != NULL);

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
    double xyz_star[P4EST_DIM];
    node_xyz_fr_n(node_idx, p4est_np1, nodes_np1, xyz_star);
    for (u_char dir = 0; dir < P4EST_DIM; ++dir)
      xyz_star[dir] -= (second_order ? 0.5 : 1.0)*dt_n*interface_velocity_np1_buffer[P4EST_DIM*to_compute + dir];
    clip_in_domain(xyz_star, xyz_min, xyz_max, periodicity);

    if(second_order){
      interp_np1.add_point(to_compute, xyz_star);
      interp_n->add_point(to_compute, xyz_star);
    }
    else
      interp_phi_n.add_point(node_idx, xyz_star);
    to_compute++;
  }
  P4EST_ASSERT(to_compute + known_idx == (size_t) nodes_np1->num_owned_indeps);

  if(second_order)
  {
    interp_np1.interpolate(interface_velocity_np1_buffer.data()); interp_np1.clear();
    std::vector<double> interface_velocity_n_buffer(P4EST_DIM*to_compute);
    interp_n->interpolate(interface_velocity_n_buffer.data()); interp_n->clear();
    known_idx = 0;
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
        xyz_d[dir] -= dt_n*((1.0 + 0.5*dt_n/dt_nm1)*interface_velocity_np1_buffer[P4EST_DIM*to_compute + dir] - 0.5*(dt_n/dt_nm1)*interface_velocity_n_buffer[P4EST_DIM*to_compute + dir]);
      clip_in_domain(xyz_d, xyz_min, xyz_max, periodicity);
      interp_phi_n.add_point(node_idx, xyz_d);
      to_compute++;
    }
    P4EST_ASSERT(to_compute + known_idx == (size_t) nodes_np1->num_owned_indeps);
  }

  if(known_phi_np1 != NULL)
  {
    ierr = VecRestoreArray(phi_np1, &phi_np1_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(known_phi_np1, &known_phi_np1_p);  CHKERRXX(ierr);
  }

  interp_phi_n.interpolate(phi_np1); interp_phi_n.clear();
  ierr = VecGhostUpdateBegin(phi_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(phi_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  if(interp_n != NULL)
    delete interp_n;

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

void my_p4est_two_phase_flows_t::get_average_interface_velocity(double avg_itfc_velocity[P4EST_DIM])
{
  if(interface_velocity_np1 != NULL)
    set_interface_velocity_np1();

  my_p4est_interpolation_nodes_t interp_itfc_velocity(ngbd_n);
  interp_itfc_velocity.set_input(interface_velocity_np1, interface_velocity_np1_xxyyzz, quadratic, P4EST_DIM);

  double integral_velocity_and_surface[P4EST_DIM + 1] = {0.0, DIM(0.0, 0.0, 0.0)};

  for (p4est_topidx_t tree_idx = p4est_n->first_local_tree; tree_idx <= p4est_n->last_local_tree; ++tree_idx) {
    const p4est_tree_t* tree = p4est_tree_array_index(p4est_n->trees, tree_idx);
    for (size_t q = 0; q < tree->quadrants.elem_count; ++q) {
      const p4est_locidx_t quad_idx = q + tree->quadrants_offset;

      if((cell_jump_solver != NULL && dynamic_cast<my_p4est_poisson_jump_cells_fv_t*>(cell_jump_solver) != NULL
          && dynamic_cast<my_p4est_poisson_jump_cells_fv_t*>(cell_jump_solver)->are_required_finite_volumes_and_correction_functions_known
          && dynamic_cast<my_p4est_poisson_jump_cells_fv_t*>(cell_jump_solver)->finite_volume_data_for_quad.find(quad_idx) != dynamic_cast<my_p4est_poisson_jump_cells_fv_t*>(cell_jump_solver)->finite_volume_data_for_quad.end())
         || interface_manager->is_quad_crossed_by_interface(quad_idx, tree_idx, NULL))
      {
        double xyz_quad[P4EST_DIM];
        const double *tree_xyz_min, *tree_xyz_max;
        const p4est_quadrant_t* quad;
        fetch_quad_and_tree_coordinates(quad, tree_xyz_min, tree_xyz_max, quad_idx, tree_idx, p4est_n, ghost_n);
        xyz_of_quad_center(quad, tree_xyz_min, tree_xyz_max, xyz_quad);
        my_p4est_finite_volume_t fv;
        if(cell_jump_solver != NULL && dynamic_cast<my_p4est_poisson_jump_cells_fv_t*>(cell_jump_solver) != NULL
           && dynamic_cast<my_p4est_poisson_jump_cells_fv_t*>(cell_jump_solver)->are_required_finite_volumes_and_correction_functions_known
           && dynamic_cast<my_p4est_poisson_jump_cells_fv_t*>(cell_jump_solver)->finite_volume_data_for_quad.find(quad_idx) != dynamic_cast<my_p4est_poisson_jump_cells_fv_t*>(cell_jump_solver)->finite_volume_data_for_quad.end())
          fv = dynamic_cast<my_p4est_poisson_jump_cells_fv_t*>(cell_jump_solver)->finite_volume_data_for_quad[quad_idx];
        else
          fv = interface_manager->get_finite_volume_for_quad(quad_idx, tree_idx);

        P4EST_ASSERT(fv.interfaces.size() <= 1);
        for (size_t k = 0; k < fv.interfaces.size(); ++k) {
          const double xyz_interface_quadrature[P4EST_DIM] = {DIM(xyz_quad[0] + fv.interfaces[k].centroid[0], xyz_quad[1] + fv.interfaces[k].centroid[1], xyz_quad[2] + fv.interfaces[k].centroid[2])};
          double interface_velocity_at_quadrature_point[P4EST_DIM];
          interp_itfc_velocity(xyz_interface_quadrature, interface_velocity_at_quadrature_point);
          for (u_char dim = 0; dim < P4EST_DIM; ++dim)
            integral_velocity_and_surface[dim] += interface_velocity_at_quadrature_point[dim]*fv.interfaces[k].area;
          integral_velocity_and_surface[P4EST_DIM] += fv.interfaces[k].area;
        }
      }
    }
  }

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, integral_velocity_and_surface, 1 + P4EST_DIM, MPI_DOUBLE, MPI_SUM, p4est_n->mpicomm); SC_CHECK_MPI(mpiret);

  for (u_char dim = 0; dim < P4EST_DIM; ++dim)
    avg_itfc_velocity[dim] = integral_velocity_and_surface[dim]/integral_velocity_and_surface[P4EST_DIM];

  return;
}

void my_p4est_two_phase_flows_t::update_from_tn_to_tnp1(const bool& reinitialize_levelset, const bool& static_interface)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_two_phase_flows_update, 0, 0, 0, 0); CHKERRXX(ierr);

  if(!dt_updated)
    compute_dt();
  dt_updated = false;

  if(!static_interface && interface_velocity_np1 == NULL)
    set_interface_velocity_np1();

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
  // we do not need the vorticities anymore (only need them for grid construction)
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
    // destroy what you had saved
    if(known_nodes_np1 != nodes_n)
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

  ierr = delete_and_nullify_vector(jump_normal_velocity); CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(non_viscous_pressure_jump);  CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(interface_tangential_stress); CHKERRXX(ierr);
  // on computational grid at time nm1, just "slide" fields and grids in discrete time
  ierr = delete_and_nullify_vector(vnm1_nodes_minus); CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(vnm1_nodes_plus);  CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(vnm1_nodes_minus_xxyyzz);  CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(vnm1_nodes_plus_xxyyzz);   CHKERRXX(ierr);
  vnm1_nodes_minus  = vn_nodes_minus;
  vnm1_nodes_plus   = vn_nodes_plus;
  vnm1_nodes_minus_xxyyzz = vn_nodes_minus_xxyyzz;
  vnm1_nodes_plus_xxyyzz  = vn_nodes_plus_xxyyzz;
  // no longer need the n interface velocity
  ierr = delete_and_nullify_vector(interface_velocity_n); CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(interface_velocity_n_xxyyzz); CHKERRXX(ierr);
  interface_velocity_n        = interface_velocity_np1;
  interface_velocity_n_xxyyzz = interface_velocity_np1_xxyyzz;
  interface_velocity_np1 = NULL; // will need to be determined after completion of the next time step
  interface_velocity_np1_xxyyzz = NULL; // will need to be determined after completion of the next time step
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
    Vec inputs[2]   = {vnp1_nodes_minus,  vnp1_nodes_plus};
    Vec outputs[2]  = {vn_nodes_minus,    vn_nodes_plus};
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
  set_phi((fine_ngbd_n != NULL ? phi_on_fine_nodes_np1 : phi_on_computational_nodes_np1), levelset_interpolation_method, phi_on_computational_nodes_np1); // memory handled therein!
  interface_manager->evaluate_FD_theta_with_quadratics(fetch_interface_FD_neighbors_with_second_order_accuracy);

  ierr = delete_and_nullify_vector(pressure_minus); CHKERRXX(ierr);
  ierr = delete_and_nullify_vector(pressure_plus);  CHKERRXX(ierr);
  for (u_char dir = 0; dir < P4EST_DIM; ++dir) {
    ierr = delete_and_nullify_vector(vnp1_face_minus[dir]); CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(vnp1_face_plus[dir]);  CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(viscosity_rhs_minus[dir]); CHKERRXX(ierr);
    ierr = delete_and_nullify_vector(viscosity_rhs_plus[dir]);  CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est_n, faces_n, &vnp1_face_minus[dir], dir); CHKERRXX(ierr);
    ierr = VecCreateGhostFaces(p4est_n, faces_n, &vnp1_face_plus[dir],  dir); CHKERRXX(ierr);
    ierr = VecCreateNoGhostFaces(p4est_n, faces_n, &viscosity_rhs_minus[dir], dir); CHKERRXX(ierr);
    ierr = VecCreateNoGhostFaces(p4est_n, faces_n, &viscosity_rhs_plus[dir],  dir); CHKERRXX(ierr);
  }

  if(cell_jump_solver != NULL)
    delete cell_jump_solver;

  if(face_jump_solver != NULL)
    delete face_jump_solver;
  pressure_guess_is_set = false;

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
  }

  ierr = PetscLogEventEnd(log_my_p4est_two_phase_flows_update, 0, 0, 0, 0); CHKERRXX(ierr);

  return;
}

