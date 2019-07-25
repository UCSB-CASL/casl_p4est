#ifdef P4_TO_P8
#include "my_p8est_xgfm_cells.h"
#include <p8est_communication.h>
#else
#include "my_p4est_xgfm_cells.h"
#include <p4est_communication.h>
#endif

#ifdef CASL_THROWS
#ifdef P4_TO_P8
#include <p8est_algorithms.h>
#else
#include <p4est_algorithms.h>
#endif
#endif

#include <src/petsc_compatibility.h>
#include <src/casl_math.h>
#include <algorithm>

// logging variables -- defined in src/petsc_logging.cpp
#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_my_p4est_xgfm_cells_matrix_preallocation;
extern PetscLogEvent log_my_p4est_xgfm_cells_matrix_setup;
extern PetscLogEvent log_my_p4est_xgfm_cells_rhsvec_setup;
extern PetscLogEvent log_my_p4est_xgfm_cells_solve;
extern PetscLogEvent log_my_p4est_xgfm_cells_KSPSolve;
extern PetscLogEvent log_my_p4est_xgfm_cells_extend_field;
extern PetscLogEvent log_my_p4est_xgfm_cells_interpolate_coarse_cell_field_to_fine_nodes;
extern PetscLogEvent log_my_p4est_xgfm_cells_get_corrected_rhs;
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif
#define bcstrength 1.0

my_p4est_xgfm_cells_t::my_p4est_xgfm_cells_t(const my_p4est_cell_neighbors_t *ngbd_c,
                                             const my_p4est_node_neighbors_t *ngbd_n,
                                             const my_p4est_node_neighbors_t *fine_ngbd_n,
                                             const bool activate_x_)
  : cell_ngbd(ngbd_c), node_ngbd(ngbd_n),
    p4est(ngbd_c->p4est), nodes(ngbd_n->nodes), ghost(ngbd_c->ghost), brick(ngbd_c->myb),
    mu_m(1.0), mu_p(1.0), add_diag_m(0.0), add_diag_p(0.0),
    rhs(NULL), solution(NULL),
    fine_p4est(fine_ngbd_n->p4est), fine_nodes(fine_ngbd_n->nodes), fine_ghost(fine_ngbd_n->ghost),
    fine_node_ngbd(fine_ngbd_n),
    phi(NULL), jump_u(NULL),
    phi_has_been_set(false), normals_have_been_set(false), mus_have_been_set(false), jumps_have_been_set(false), second_derivatives_of_phi_are_set(false),
    corrected_rhs(NULL), extension_cell_values(NULL), extension_on_fine_nodes(NULL),
    bc(NULL),
    nullspace_use_fixed_point(false),
    A(NULL), A_null_space(NULL),
    null_space(NULL),
    is_matrix_built(false), matrix_has_nullspace(false),
    activate_x(activate_x_)
{
  // set up the KSP solver
  ierr = KSPCreate(p4est->mpicomm, &ksp); CHKERRXX(ierr);

  splitting_criteria_t *data        = (splitting_criteria_t*) p4est->user_pointer;
  splitting_criteria_t *data_fine   = (splitting_criteria_t*) fine_p4est->user_pointer;

  // compute grid parameters
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    xyz_min[dir]          = p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*0                              +                0]  + dir];
    tree_dimensions[dir]  = p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*0                              + P4EST_CHILDREN-1]  + dir] - xyz_min[dir];
    xyz_max[dir]          = p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*(p4est->trees->elem_count - 1) + P4EST_CHILDREN-1]  + dir];
    dxyz_min[dir]         = tree_dimensions[dir] / pow(2.,(double) data->max_lvl);
    dxyz_min_fine[dir]    = (tree_dimensions[dir]) / pow(2.,(double) data_fine->max_lvl);
    periodicity[dir]      = is_periodic(p4est, dir);
  }

#ifdef P4_TO_P8
  d_min                   = MIN(dxyz_min[0], dxyz_min[1], dxyz_min[2]);
  diag_min                = sqrt(SQR(dxyz_min[0]) + SQR(dxyz_min[1]) + SQR(dxyz_min[2]));
  cell_volume_min         = dxyz_min[0]*dxyz_min[1]*dxyz_min[2];
#else
  d_min                   = MIN(dxyz_min[0], dxyz_min[1]);
  diag_min                = sqrt(SQR(dxyz_min[0]) + SQR(dxyz_min[1]));
  cell_volume_min         = dxyz_min[0]*dxyz_min[1];
#endif

  for (short dim = 0; dim < P4EST_DIM; ++dim) {
    normals[dim]          = NULL;
    phi_second_der[dim]   = NULL;
    jump_mu_grad_u[dim]   = NULL;
  }
  map_of_interface_neighbors.clear();
  map_of_neighbors_is_initialized = false;
  extension_entries.clear(); extension_entries.resize(p4est->local_num_quadrants);
  extension_entries_are_set       = false;
  local_interpolator.clear(); local_interpolator.resize(fine_nodes->num_owned_indeps);
  local_interpolator_is_set       = false;
  interface_values_are_set        = false;
  solution_is_set                 = false;
  use_initial_guess               = false;
}

my_p4est_xgfm_cells_t::~my_p4est_xgfm_cells_t()
{
  if (A             != NULL)          { ierr = MatDestroy(A);                       CHKERRXX(ierr); }
  if (A_null_space  != NULL)          { ierr = MatNullSpaceDestroy (A_null_space);  CHKERRXX(ierr); }
  if (null_space    != NULL)          { ierr = VecDestroy(null_space);              CHKERRXX(ierr); }
  if (ksp           != NULL)          { ierr = KSPDestroy(ksp);                     CHKERRXX(ierr); }
  if (extension_cell_values  != NULL) { ierr = VecDestroy(extension_cell_values);   CHKERRXX(ierr); }
  if (extension_on_fine_nodes != NULL){ ierr = VecDestroy(extension_on_fine_nodes); CHKERRXX(ierr); }
  if (corrected_rhs != NULL)          { ierr = VecDestroy(corrected_rhs);           CHKERRXX(ierr); }
  if (solution      != NULL)          { ierr = VecDestroy(solution);                CHKERRXX(ierr); }
  for (short dim = 0; dim < P4EST_DIM; ++dim) {
    if(jump_mu_grad_u[dim] != NULL)   { ierr = VecDestroy(jump_mu_grad_u[dim]);     CHKERRXX(ierr); }
  }
}

#ifdef P4_TO_P8
void my_p4est_xgfm_cells_t::set_phi(Vec phi_on_fine_mesh, Vec phi_xx_on_fine_mesh, Vec phi_yy_on_fine_mesh, Vec phi_zz_on_fine_mesh)
#else
void my_p4est_xgfm_cells_t::set_phi(Vec phi_on_fine_mesh, Vec phi_xx_on_fine_mesh, Vec phi_yy_on_fine_mesh)
#endif
{
#ifdef P4_TO_P8
  second_derivatives_of_phi_are_set = ((phi_xx_on_fine_mesh != NULL) && (phi_yy_on_fine_mesh != NULL) && (phi_zz_on_fine_mesh != NULL));
#else
  second_derivatives_of_phi_are_set = ((phi_xx_on_fine_mesh != NULL) && (phi_yy_on_fine_mesh != NULL));
#endif
#ifdef CASL_THROWS
  // compare local size
  PetscInt local_size;
  ierr = VecGetLocalSize(phi_on_fine_mesh, &local_size); CHKERRXX(ierr);
  int my_error = ( ((PetscInt) fine_nodes->num_owned_indeps) != local_size);

  // compare global size
  PetscInt global_size;
  ierr = VecGetSize(phi_on_fine_mesh, &global_size); CHKERRXX(ierr);
  PetscInt global_nb_fine_nodes = 0;
  for (int r = 0; r<p4est->mpisize; ++r)
    global_nb_fine_nodes += (PetscInt)fine_nodes->global_owned_indeps[r];
  my_error = my_error || (global_size != global_nb_fine_nodes);

  // compare ghost layers
  PetscInt ghost_layer_size;
  Vec phi_fine_loc;
  ierr = VecGhostGetLocalForm(phi_on_fine_mesh, &phi_fine_loc); CHKERRXX(ierr);
  ierr = VecGetSize(phi_fine_loc, &ghost_layer_size); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(phi_on_fine_mesh, &phi_fine_loc); CHKERRXX(ierr);
  ghost_layer_size -= local_size;
  my_error = my_error || (ghost_layer_size != ((PetscInt) (fine_nodes->indep_nodes.elem_count - fine_nodes->num_owned_indeps)));

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &my_error, 1, MPI_INT, MPI_LOR, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  if(my_error)
    throw std::invalid_argument("my_p4est_xgfm_cells_t::set_phi(Vec) : the phi vector must be of the same size as the number of nodes in the fine mesh");

  // check the second derivatives if needed
#ifdef P4_TO_P8
  my_error = !(((phi_xx_on_fine_mesh == NULL) && (phi_yy_on_fine_mesh == NULL) && (phi_zz_on_fine_mesh == NULL)) ||
               ((phi_xx_on_fine_mesh != NULL) && (phi_yy_on_fine_mesh != NULL) && (phi_zz_on_fine_mesh != NULL)));
#else
  my_error = !(((phi_xx_on_fine_mesh == NULL) && (phi_yy_on_fine_mesh == NULL)) ||
               ((phi_xx_on_fine_mesh != NULL) && (phi_yy_on_fine_mesh != NULL)));
#endif
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &my_error, 1, MPI_INT, MPI_LOR, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  if(my_error)
#ifdef P4_TO_P8
    throw std::invalid_argument("my_p4est_xgfm_cells_t::set_phi(Vec, Vec, Vec, Vec) : the second derivatives of phi should either be all set or all disregarded");
#else
    throw std::invalid_argument("my_p4est_xgfm_cells_t::set_phi(Vec, Vec, Vec) : the second derivatives of phi should either be all set or all disregarded");
#endif

  if (second_derivatives_of_phi_are_set)
  {
    // check second derivatives of phi
    // compare local, global and ghost layer sizes

    // xx
    ierr = VecGetLocalSize(phi_xx_on_fine_mesh, &local_size); CHKERRXX(ierr);
    my_error = ( ((PetscInt) fine_nodes->num_owned_indeps) != local_size);
    ierr = VecGetSize(phi_xx_on_fine_mesh, &global_size); CHKERRXX(ierr);
    my_error = my_error || (global_size != global_nb_fine_nodes);
    Vec phi_xx_on_fine_mesh_loc;
    ierr = VecGhostGetLocalForm(phi_xx_on_fine_mesh, &phi_xx_on_fine_mesh_loc); CHKERRXX(ierr);
    ierr = VecGetSize(phi_xx_on_fine_mesh_loc, &ghost_layer_size); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(phi_xx_on_fine_mesh, &phi_xx_on_fine_mesh_loc); CHKERRXX(ierr);
    ghost_layer_size -= local_size;
    my_error = my_error || (ghost_layer_size != ((PetscInt) (fine_nodes->indep_nodes.elem_count - fine_nodes->num_owned_indeps)));
    // yy
    ierr = VecGetLocalSize(phi_yy_on_fine_mesh, &local_size); CHKERRXX(ierr);
    my_error = ( ((PetscInt) fine_nodes->num_owned_indeps) != local_size);
    ierr = VecGetSize(phi_yy_on_fine_mesh, &global_size); CHKERRXX(ierr);
    my_error = my_error || (global_size != global_nb_fine_nodes);
    Vec phi_yy_on_fine_mesh_loc;
    ierr = VecGhostGetLocalForm(phi_yy_on_fine_mesh, &phi_yy_on_fine_mesh_loc); CHKERRXX(ierr);
    ierr = VecGetSize(phi_yy_on_fine_mesh_loc, &ghost_layer_size); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(phi_yy_on_fine_mesh, &phi_yy_on_fine_mesh_loc); CHKERRXX(ierr);
    ghost_layer_size -= local_size;
    my_error = my_error || (ghost_layer_size != ((PetscInt) (fine_nodes->indep_nodes.elem_count - fine_nodes->num_owned_indeps)));
#ifdef P4_TO_P8
    // zz
    ierr = VecGetLocalSize(phi_zz_on_fine_mesh, &local_size); CHKERRXX(ierr);
    my_error = ( ((PetscInt) fine_nodes->num_owned_indeps) != local_size);
    ierr = VecGetSize(phi_zz_on_fine_mesh, &global_size); CHKERRXX(ierr);
    my_error = my_error || (global_size != global_nb_fine_nodes);
    Vec phi_zz_on_fine_mesh_loc;
    ierr = VecGhostGetLocalForm(phi_zz_on_fine_mesh, &phi_zz_on_fine_mesh_loc); CHKERRXX(ierr);
    ierr = VecGetSize(phi_zz_on_fine_mesh_loc, &ghost_layer_size); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(phi_zz_on_fine_mesh, &phi_zz_on_fine_mesh_loc); CHKERRXX(ierr);
    ghost_layer_size -= local_size;
    my_error = my_error || (ghost_layer_size != ((PetscInt) (fine_nodes->indep_nodes.elem_count - fine_nodes->num_owned_indeps)));
#endif

    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &my_error, 1, MPI_INT, MPI_LOR, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    if(my_error)
#ifdef P4_TO_P8
      throw std::invalid_argument("my_p4est_xgfm_cells_t::set_phi(Vec, Vec, Vec, Vec) : the second derivatives of phi must be of the same size as the number of nodes in the fine mesh");
#else
      throw std::invalid_argument("my_p4est_xgfm_cells_t::set_phi(Vec, Vec, Vec) : the second derivatives of phi must be of the same size as the number of nodes in the fine mesh");
#endif


  }
#endif
  if(second_derivatives_of_phi_are_set)
  {
    phi_second_der[0] = phi_xx_on_fine_mesh;
    phi_second_der[1] = phi_yy_on_fine_mesh;
#ifdef P4_TO_P8
    phi_second_der[2] = phi_zz_on_fine_mesh;
#endif
  }

  phi = phi_on_fine_mesh;

  phi_has_been_set = true;
}

void my_p4est_xgfm_cells_t::set_normals(Vec normals_[])
{
#ifdef CASL_THROWS
  // compare local size
  PetscInt local_size[P4EST_DIM];
  for (short dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecGetLocalSize(normals_[dim], &local_size[dim]); CHKERRXX(ierr);}

  int my_error = ( ((PetscInt) fine_nodes->num_owned_indeps) != local_size[0]);
  for (short dim = 1; dim < P4EST_DIM; ++dim)
    my_error = my_error || (local_size[dim] != local_size[dim-1]);

  // compare global size
  PetscInt global_size[P4EST_DIM];
  for (short dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecGetSize(normals_[dim], &global_size[dim]); CHKERRXX(ierr);}

  PetscInt global_nb_fine_nodes = 0;
  for (int r = 0; r<p4est->mpisize; ++r)
    global_nb_fine_nodes += (PetscInt)fine_nodes->global_owned_indeps[r];
  my_error = my_error || (global_size[0] != global_nb_fine_nodes);
  for (short dim = 1; dim < P4EST_DIM; ++dim)
    my_error = my_error || (global_size[dim] != global_size[dim-1]);

  // compare ghost layers
  PetscInt ghost_layer_size[P4EST_DIM];
  Vec normals_loc[P4EST_DIM];
  for (short dim = 0; dim < P4EST_DIM; ++dim)
  {
    ierr = VecGhostGetLocalForm(normals_[dim], &normals_loc[dim]); CHKERRXX(ierr);
    ierr = VecGetSize(normals_loc[dim], &ghost_layer_size[dim]); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(normals_[dim], &normals_loc[dim]); CHKERRXX(ierr);
    ghost_layer_size[dim] -= local_size[dim];
  }

  my_error = my_error || (ghost_layer_size[0] != ((PetscInt) (fine_nodes->indep_nodes.elem_count - fine_nodes->num_owned_indeps)));
  for (short dim = 1; dim < P4EST_DIM; ++dim)
    my_error = my_error || (ghost_layer_size[dim] != ghost_layer_size[dim-1]);

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &my_error, 1, MPI_INT, MPI_LOR, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  if(my_error)
    throw std::invalid_argument("my_p4est_xgfm_cells_t::set_normals(Vec[]) : the normal vectors must be of the same size as the number of nodes in the fine mesh");
#endif
  for (short dim = 0; dim < P4EST_DIM; ++dim)
    normals[dim] = normals_[dim];

  normals_have_been_set = true;
}

void my_p4est_xgfm_cells_t::set_jumps(Vec jump_sol, Vec jump_normal_flux)
{
#ifdef CASL_THROWS
  // compare local size
  PetscInt local_size[2];
  ierr = VecGetLocalSize(jump_sol, &local_size[0]); CHKERRXX(ierr);
  ierr = VecGetLocalSize(jump_normal_flux, &local_size[1]); CHKERRXX(ierr);
  int my_error = (((PetscInt) fine_nodes->num_owned_indeps) != local_size[0]) || (local_size[0] != local_size[1]);

  // compare global size
  PetscInt global_size[2];
  ierr = VecGetSize(jump_sol, &global_size[0]); CHKERRXX(ierr);
  ierr = VecGetSize(jump_normal_flux, &global_size[1]); CHKERRXX(ierr);

  PetscInt global_nb_fine_nodes = 0;
  for (int r = 0; r<p4est->mpisize; ++r)
    global_nb_fine_nodes += (PetscInt)fine_nodes->global_owned_indeps[r];
  my_error = my_error || (global_size[0] != global_nb_fine_nodes) || (global_size[0] != global_size[1]);

  // compare ghost layers
  PetscInt ghost_layer_size[2];
  Vec jump_sol_loc, jump_normal_flux_loc;
  ierr = VecGhostGetLocalForm(jump_sol, &jump_sol_loc); CHKERRXX(ierr);
  ierr = VecGhostGetLocalForm(jump_normal_flux, &jump_normal_flux_loc); CHKERRXX(ierr);
  ierr = VecGetSize(jump_sol_loc, &ghost_layer_size[0]); CHKERRXX(ierr);
  ghost_layer_size[0] -= local_size[0];
  ierr = VecGetSize(jump_normal_flux_loc, &ghost_layer_size[1]); CHKERRXX(ierr);
  ghost_layer_size[1] -= local_size[1];
  ierr = VecGhostRestoreLocalForm(jump_normal_flux, &jump_normal_flux_loc); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(jump_sol, &jump_sol_loc); CHKERRXX(ierr);
  my_error = my_error || (ghost_layer_size[0] != ((PetscInt) (fine_nodes->indep_nodes.elem_count - fine_nodes->num_owned_indeps))) || (ghost_layer_size[0] != ghost_layer_size[1]);

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &my_error, 1, MPI_INT, MPI_LOR, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  if(my_error)
    throw std::invalid_argument("my_p4est_xgfm_cells_t::set_jumps(Vec, Vec): the vectors of jump values must be of the same size as the number of nodes in the fine mesh");

  if(!normals_have_been_set)
    throw std::invalid_argument("my_p4est_xgfm_cells_t::set_jumps(Vec, Vec): the normals must be set first!");
  if(!mus_have_been_set)
    throw std::invalid_argument("my_p4est_xgfm_cells_t::set_jumps(Vec, Vec): the values of the coefficients must be set first!");
#endif
  jump_u            = jump_sol;

  const double *jump_u_read_p, *jump_normal_flux_read_p, *normals_read_p[P4EST_DIM];
  ierr = VecGetArrayRead(jump_u, &jump_u_read_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(jump_normal_flux, &jump_normal_flux_read_p); CHKERRXX(ierr);
  double *jump_mu_grad_u_p[P4EST_DIM];
  for (short dim = 0; dim < P4EST_DIM; ++dim) {
    if(jump_mu_grad_u[dim] != NULL){
      ierr = VecDestroy(jump_mu_grad_u[dim]); CHKERRXX(ierr);}
    ierr = VecCreateGhostNodes(fine_p4est, fine_nodes, &jump_mu_grad_u[dim]); CHKERRXX(ierr);
    ierr = VecGetArray(jump_mu_grad_u[dim], &jump_mu_grad_u_p[dim]); CHKERRXX(ierr);
    ierr = VecGetArrayRead(normals[dim], &normals_read_p[dim]); CHKERRXX(ierr);
  }
  set_jump_mu_grad_u_for_nodes(fine_node_ngbd->layer_nodes, jump_mu_grad_u_p, jump_normal_flux_read_p, normals_read_p, jump_u_read_p);
  for (short dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecGhostUpdateBegin(jump_mu_grad_u[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }
  set_jump_mu_grad_u_for_nodes(fine_node_ngbd->local_nodes, jump_mu_grad_u_p, jump_normal_flux_read_p, normals_read_p, jump_u_read_p);
  for (short dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecGhostUpdateEnd(jump_mu_grad_u[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  for (short dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecRestoreArrayRead(normals[dim], &normals_read_p[dim]); CHKERRXX(ierr);
    ierr = VecRestoreArray(jump_mu_grad_u[dim], &jump_mu_grad_u_p[dim]); CHKERRXX(ierr);
  }
  ierr = VecRestoreArrayRead(jump_normal_flux, &jump_normal_flux_read_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(jump_u, &jump_u_read_p); CHKERRXX(ierr);
  jumps_have_been_set = true;
}

void my_p4est_xgfm_cells_t::set_jump_mu_grad_u_for_nodes(const std::vector<p4est_locidx_t> &list_of_node_indices, double *jump_mu_grad_u_p[], const double *jump_normal_flux_read_p, const double *normals_read_p[], const double *jump_u_read_p)
{
  p4est_locidx_t node_idx;
  double grad_jump_u_cdot_normal = 0.0, norm_n;
  double grad_jump_u[P4EST_DIM], xyz_node[P4EST_DIM], n_comp[P4EST_DIM];
  const quad_neighbor_nodes_of_node_t *qnnn;
  for (size_t ii = 0; ii < list_of_node_indices.size(); ++ii) {
    node_idx = list_of_node_indices.at(ii);
    if(activate_x){
      node_xyz_fr_n(node_idx, fine_p4est, fine_nodes, xyz_node);
      qnnn = &fine_node_ngbd->neighbors[node_idx];
      grad_jump_u_cdot_normal = 0.0;
    }
    norm_n = 0.0;
    for (short dim = 0; dim < P4EST_DIM; ++dim)
    {
      n_comp[dim] = normals_read_p[dim][node_idx];
      if(activate_x)
      {
        switch (dim) {
        case 0:
          grad_jump_u[dim] = qnnn->dx_central(jump_u_read_p);
          break;
        case 1:
          grad_jump_u[dim] = qnnn->dy_central(jump_u_read_p);
          break;
#ifdef P4_TO_P8
        case 2:
          grad_jump_u[dim] = qnnn->dz_central(jump_u_read_p);
          break;
#endif
        }
        grad_jump_u_cdot_normal += n_comp[dim]*grad_jump_u[dim];
      }
      norm_n += SQR(n_comp[dim]);
    }
    norm_n = sqrt(norm_n);
    if(norm_n > EPS)
    {
      grad_jump_u_cdot_normal /= norm_n;
      for (short dim = 0; dim < P4EST_DIM; ++dim)
        n_comp[dim] /= norm_n;
    }
    else
    {
      grad_jump_u_cdot_normal = 0.0;
      for (short dim = 0; dim < P4EST_DIM; ++dim)
        n_comp[dim] = 0.0;
    }
    for (short dim = 0; dim < P4EST_DIM; ++dim) {
      jump_mu_grad_u_p[dim][node_idx] = jump_normal_flux_read_p[node_idx]*n_comp[dim] +
          ((activate_x) ? ((mu_m_is_larger)? mu_p(xyz_node) : mu_m(xyz_node))*(grad_jump_u[dim] - grad_jump_u_cdot_normal*n_comp[dim]) : 0.0);
    }
  }
}

void my_p4est_xgfm_cells_t::preallocate_matrix()
{
  // enable logging for the preallocation
  ierr = PetscLogEventBegin(log_my_p4est_xgfm_cells_matrix_preallocation, A, 0, 0, 0); CHKERRXX(ierr);

  PetscInt num_owned_global = p4est->global_num_quadrants;
  PetscInt num_owned_local  = p4est->local_num_quadrants;

  std::vector<p4est_quadrant_t> ngbd;

  if (A != NULL){
    ierr = MatDestroy(A); CHKERRXX(ierr);}

  // set up the matrix
  ierr = MatCreate(p4est->mpicomm, &A); CHKERRXX(ierr);
  ierr = MatSetType(A, MATAIJ); CHKERRXX(ierr);
  ierr = MatSetSizes(A, num_owned_local , num_owned_local,
                     num_owned_global, num_owned_global); CHKERRXX(ierr);
  ierr = MatSetFromOptions(A); CHKERRXX(ierr);

  std::vector<PetscInt> d_nnz(num_owned_local, 1), o_nnz(num_owned_local, 0);

  std::vector<p4est_locidx_t> indices;

  for (p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx){
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for (size_t q=0; q<tree->quadrants.elem_count; q++)
    {
      const p4est_quadrant_t *quad  = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
      p4est_locidx_t quad_idx = q + tree->quadrants_offset;

      indices.resize(0);

      /*
     * Check for neighboring cells:
     * 1) If they exist and are local quads, increment d_nnz[n]
     * 2) If they exist but are not local quads, increment o_nnz[n]
     * 3) If they do not exist, simply skip
     */
      for(int dir=0; dir<P4EST_FACES; ++dir)
      {
        ngbd.resize(0);
        cell_ngbd->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, dir);

        if(ngbd.size()==1 && ngbd[0].level<quad->level)
        {
          p4est_locidx_t q_tmp = ngbd[0].p.piggy3.local_num;
          p4est_topidx_t t_tmp = ngbd[0].p.piggy3.which_tree;

          /* no need to add this one to "indices" since it can't be found with a search in another direction */
          // [Raphael:] --> I think that's wrong.... Consider the following for the X-quad, the top right cell
          // would be counted twice... So, I've decided to add it to indices
          //
          //   _________________________
          //   |     |     |            |
          //   |     |     |            |
          //   |_____|_____|            |
          //   |     |     |            |
          //   |     |  X  |            |
          //   |_____|_____|____________|
          //   |                        |
          //   |                        |
          //   |                        |
          //   |                        |
          //   |                        |
          //   |                        |
          //   |                        |
          //   |________________________|
          //

          if(std::find(indices.begin(), indices.end(), ngbd[0].p.piggy3.local_num)==indices.end())
          {
            indices.push_back(ngbd[0].p.piggy3.local_num);
            if(q_tmp<num_owned_local) d_nnz[quad_idx]++;
            else                      o_nnz[quad_idx]++;
          }

          ngbd.resize(0);
          cell_ngbd->find_neighbor_cells_of_cell(ngbd, q_tmp, t_tmp, dir%2==0 ? dir+1 : dir-1);
          for(unsigned int m=0; m<ngbd.size(); ++m)
          {
            if(ngbd[m].p.piggy3.local_num!=quad_idx && std::find(indices.begin(), indices.end(), ngbd[m].p.piggy3.local_num)==indices.end())
            {
              indices.push_back(ngbd[m].p.piggy3.local_num);
              if(ngbd[m].p.piggy3.local_num<num_owned_local) d_nnz[quad_idx]++;
              else                                           o_nnz[quad_idx]++;
            }
          }
        }
        else
        {
          for(unsigned int m=0; m<ngbd.size(); ++m)
          {
            if(std::find(indices.begin(), indices.end(), ngbd[m].p.piggy3.local_num)==indices.end())
            {
              indices.push_back(ngbd[m].p.piggy3.local_num);
              if(ngbd[m].p.piggy3.local_num<num_owned_local) d_nnz[quad_idx]++;
              else                                           o_nnz[quad_idx]++;
            }
          }
        }
      }
    }
  }

  ierr = MatSeqAIJSetPreallocation(A, 0, (const PetscInt*)&d_nnz[0]); CHKERRXX(ierr);
  ierr = MatMPIAIJSetPreallocation(A, 0, (const PetscInt*)&d_nnz[0], 0, (const PetscInt*)&o_nnz[0]); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_xgfm_cells_matrix_preallocation, A, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_xgfm_cells_t::set_initial_guess(Vec& initial_guess)
{
#ifdef CASL_THROWS
  PetscInt local_size, global_size, ghost_layer_size;
  ierr = VecGetSize(initial_guess, &global_size); CHKERRXX(ierr);
  ierr = VecGetLocalSize(initial_guess, &local_size); CHKERRXX(ierr);
  Vec loc_initial_guess;
  ierr = VecGhostGetLocalForm(initial_guess, &loc_initial_guess); CHKERRXX(ierr);
  ierr = VecGetSize(loc_initial_guess, &ghost_layer_size); CHKERRXX(ierr);
  ierr = VecGhostRestoreLocalForm(initial_guess, &loc_initial_guess); CHKERRXX(ierr);
  ghost_layer_size -= local_size;
  int my_error = (local_size != p4est->local_num_quadrants) || (ghost_layer_size != (PetscInt) ghost->ghosts.elem_count) || (global_size != p4est->global_num_quadrants);
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &my_error, 1, MPI_INT, MPI_LOR, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  if (my_error)
    throw std::invalid_argument("my_p4est_xgfm_cells_t::set_initial_guess(Vec): the vector must have the same layour as if constructed with VecCreateGhostCells on the computational (coarse) p4est");
#endif
  if(solution != NULL){
    ierr = VecDestroy(solution); CHKERRXX(ierr);} // make sure we don't have a memory leak here
  solution          = initial_guess;              // --> the solver gets the ownership of the object
  initial_guess     = NULL;                       // --> the user loses the ownership of the object
  use_initial_guess = true;                       // --> the solver will use this object as its initial guess!
}

void my_p4est_xgfm_cells_t::solve(KSPType ksp_type, PCType pc_type,
                                  double absolute_accuracy_threshold, double tolerance_on_rel_residual)
{
  ierr = PetscLogEventBegin(log_my_p4est_xgfm_cells_solve, A, rhs, ksp, 0); CHKERRXX(ierr);
  int mpiret;

#ifdef CASL_THROWS
  int my_error = (bc == NULL);
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &my_error, 1, MPI_INT, MPI_LOR, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  if(my_error) throw std::domain_error("my_p4est_xgfm_cells_t::solve(): the boundary conditions have not been set.");

  my_error = (absolute_accuracy_threshold < EPS);
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &my_error, 1, MPI_INT, MPI_LOR, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  if(my_error)
    throw std::invalid_argument("my_p4est_xgfm_cells_t::solve(): the absolute tolerance for the solution must be strictly positive (at least EPS)...");

  my_error = (tolerance_on_rel_residual < EPS);
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &my_error, 1, MPI_INT, MPI_LOR, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  if(my_error)
    throw std::invalid_argument("my_p4est_xgfm_cells_t::solve(): the relative tolerance on the residual for the solution must be strictly positive (at least EPS)...");

  my_error = !phi_has_been_set;
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &my_error, 1, MPI_INT, MPI_LOR, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  if(my_error)
    throw std::invalid_argument("my_p4est_xgfm_cells_t::solve(): the levelset must be set beforehand...");
  my_error = !normals_have_been_set;
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &my_error, 1, MPI_INT, MPI_LOR, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  if(my_error)
    throw std::invalid_argument("my_p4est_xgfm_cells_t::solve(): the normal vectors must be set beforehand...");
  my_error = !jumps_have_been_set;
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &my_error, 1, MPI_INT, MPI_LOR, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  if(my_error)
    throw std::invalid_argument("my_p4est_xgfm_cells_t::solve(): the jump values must be set beforehand...");
#endif

  P4EST_ASSERT(((solution == NULL) && !use_initial_guess) || ((solution != NULL) && use_initial_guess));
  if(solution == NULL){
    ierr = VecCreateGhostCells(p4est, ghost, &solution); CHKERRXX(ierr);}

  /* reinitialize the counters and montioring vectors */
  numbers_of_ksp_iterations.resize(0);
  max_corrections.resize(0);
  relative_residuals.resize(0);

  /*
   * Here we set the matrix, ksp, and pc. If the matrix is not changed during
   * successive solves, we will reuse the same preconditioner, otherwise we
   * have to recompute the preconditioner
   */
  if (!is_matrix_built)
  {
    matrix_has_nullspace = true;
    setup_negative_laplace_matrix_with_jumps();

    ierr = KSPSetOperators(ksp, A, A, SAME_NONZERO_PATTERN); CHKERRXX(ierr);
  } else {
    ierr = KSPSetOperators(ksp, A, A, SAME_PRECONDITIONER);  CHKERRXX(ierr);
  }

  // setup rhs
  setup_negative_laplace_rhsvec_with_jumps();

  // set ksp type
  ierr = KSPSetType(ksp, ksp_type); CHKERRXX(ierr);
  ierr = KSPSetTolerances(ksp, tolerance_on_rel_residual, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRXX(ierr);
  if (use_initial_guess)
  {
    ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE); CHKERRXX(ierr);
  }
  ierr = KSPSetFromOptions(ksp); CHKERRXX(ierr);

  // set pc type
  PC pc;
  ierr = KSPGetPC(ksp, &pc); CHKERRXX(ierr);
  ierr = PCSetType(pc, pc_type); CHKERRXX(ierr);

  /* If using hypre, we can make some adjustments here. The most important parameters to be set are:
   * 1- Strong Threshold
   * 2- Coarsennig Type
   * 3- Truncation Factor
   *
   * Plerase refer to HYPRE manual for more information on the actual importance or check Mohammad Mirzadeh's
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

    /* 3- Trancation factor
     * Greater than zero.
     * Use zero for the best convergence. However, if you have memory problems, use greate than zero to save some memory.
     */
    ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_truncfactor", "0.1"); CHKERRXX(ierr);

    // Finally, if matrix has a nullspace, one should _NOT_ use Gaussian-Elimination as the smoother for the coarsest grid
    if (matrix_has_nullspace && !nullspace_use_fixed_point){
      ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_relax_type_coarse", "symmetric-SOR/Jacobi"); CHKERRXX(ierr);
    }
  }
  ierr = PCSetFromOptions(pc); CHKERRXX(ierr);

  // Solve the system
  ierr = PetscLogEventBegin(log_my_p4est_xgfm_cells_KSPSolve, solution, rhs, ksp, 0); CHKERRXX(ierr);
  ierr = KSPSolve(ksp, rhs, solution); CHKERRXX(ierr);
  ierr = PetscLogEventEnd(log_my_p4est_xgfm_cells_KSPSolve, solution, rhs, ksp, 0); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(solution, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  PetscInt nb_iterations;
  ierr = KSPGetIterationNumber(ksp, &nb_iterations); CHKERRXX(ierr);
  numbers_of_ksp_iterations.push_back(nb_iterations);
  max_corrections.push_back(0.0);
  relative_residuals.push_back(0.0);
  ierr = VecGhostUpdateEnd  (solution, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  if(activate_x && !mu_m_and_mu_p_equal)// no correction if mu_m = mu_p
  {
    // clear the workspace before we start...
    if(extension_cell_values != NULL){
      ierr = VecDestroy(extension_cell_values); CHKERRXX(ierr);}
    if(extension_on_fine_nodes != NULL){
      ierr = VecDestroy(extension_on_fine_nodes); extension_on_fine_nodes = NULL; CHKERRXX(ierr);}
    if(corrected_rhs != NULL){
      ierr = VecDestroy(corrected_rhs); corrected_rhs = NULL; CHKERRXX(ierr);}

    // extend interface values
    ierr = VecCreateGhostCells(p4est, ghost, &extension_cell_values); CHKERRXX(ierr);
    double *solution_p;
    ierr = VecGetArray(solution, &solution_p); CHKERRXX(ierr);
    extend_interface_values(solution_p, extension_cell_values, NULL);
    // interpolate it on the fine grid
    ierr = VecCreateGhostNodes(fine_p4est, fine_nodes, &extension_on_fine_nodes); CHKERRXX(ierr);
    double *extension_cell_values_p;
    ierr = VecGetArray(extension_cell_values, &extension_cell_values_p); CHKERRXX(ierr);
    interpolate_coarse_cell_field_to_fine_nodes(extension_cell_values_p, extension_on_fine_nodes);
    // calculate the correction for the rhs
    ierr = VecCreateCellsNoGhost(p4est, &corrected_rhs); CHKERRXX(ierr);
    double *extension_on_fine_nodes_p;
    ierr = VecGetArray(extension_on_fine_nodes, &extension_on_fine_nodes_p); CHKERRXX(ierr);
    get_corrected_rhs(corrected_rhs, extension_on_fine_nodes_p);
    // calculate the residual of the current solution
    Vec residual;
    ierr = VecCreateCellsNoGhost(p4est, &residual); CHKERRXX(ierr);
    ierr = VecRestoreArray(solution, &solution_p); CHKERRXX(ierr);
    ierr = MatMult(A, solution, residual); CHKERRXX(ierr);
    ierr = VecGetArray(solution, &solution_p); CHKERRXX(ierr);
    // finish the operation as residual -= corrected_rhs
    // --> can be done in the same loop as the calculation of relevant two-norms
    // done in the following for optimization
    // (knowingly avoiding separate Petsc operations that would multiply the number of such loops)
    double *residual_p, *corrected_rhs_p, norm_residual_and_corrected_rhs[2];
    norm_residual_and_corrected_rhs[0] = norm_residual_and_corrected_rhs[1] = 0.0;
    ierr = VecGetArray(residual, &residual_p); CHKERRXX(ierr);
    ierr = VecGetArray(corrected_rhs, &corrected_rhs_p); CHKERRXX(ierr);
    for (p4est_locidx_t qq = 0; qq < p4est->local_num_quadrants; ++qq) {
      residual_p[qq] -= corrected_rhs_p[qq];
      norm_residual_and_corrected_rhs[0] += SQR(residual_p[qq]);
      norm_residual_and_corrected_rhs[1] += SQR(corrected_rhs_p[qq]);
    }
    mpiret = MPI_Allreduce(MPI_IN_PLACE, norm_residual_and_corrected_rhs, 2, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    norm_residual_and_corrected_rhs[0] = sqrt(norm_residual_and_corrected_rhs[0]);
    norm_residual_and_corrected_rhs[1] = sqrt(norm_residual_and_corrected_rhs[1]);

    relative_residuals[0] = norm_residual_and_corrected_rhs[0]/norm_residual_and_corrected_rhs[1];

    double inner_products[2]; // inner_products[0] = <residual, residual - residual_tilde>; inner_products[1] = <residual - residual_tilde, residual - residual_tilde>;
    double step_size;
    Vec solution_tilde = NULL; double *solution_tilde_p = NULL;
    ierr = VecCreateGhostCells(p4est, ghost, &solution_tilde); CHKERRXX(ierr);
    ierr = VecGetArray(solution_tilde, &solution_tilde_p); CHKERRXX(ierr);
    // update solution_tilde <= solution to have a good initial guess for next KSP solve
    // the original vector's ghost values are already up-to-date, no need to GhostUpdate...
    // [knowingly avoiding separate Petsc operations, because of simplicity (no VecGhostGetLocalForm, etc.)]
    for (p4est_locidx_t qq = 0; qq < (p4est_locidx_t) (p4est->local_num_quadrants + ghost->ghosts.elem_count); ++qq)
      solution_tilde_p[qq]  = solution_p[qq];
    // other tilde vectors
    Vec extension_cell_values_tilde = NULL; double *extension_cell_values_tilde_p = NULL;
    Vec extension_on_fine_nodes_tilde = NULL; double *extension_on_fine_nodes_tilde_p = NULL;
    Vec corrected_rhs_tilde = NULL; double *corrected_rhs_tilde_p = NULL;
    Vec residual_tilde = NULL; double *residual_tilde_p = NULL;
    bool other_tilde_vectors_have_been_created = false;
    // convergence flag
    bool solver_has_converged = false;
    ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE); CHKERRXX(ierr);

    while (!solver_has_converged) {

      ierr = PetscLogEventBegin(log_my_p4est_xgfm_cells_KSPSolve, solution_tilde, corrected_rhs, ksp, 0); CHKERRXX(ierr);
      if (matrix_has_nullspace)
      {
        if(!nullspace_use_fixed_point){
          ierr = MatNullSpaceRemove(A_null_space, corrected_rhs, NULL); CHKERRXX(ierr);}
        else if(fixed_value_idx_l >= 0)
          corrected_rhs_p[fixed_value_idx_l] = 0;
      }
      ierr = KSPSolve(ksp, corrected_rhs, solution_tilde); CHKERRXX(ierr);
      ierr = PetscLogEventEnd(log_my_p4est_xgfm_cells_KSPSolve, solution_tilde, corrected_rhs, ksp, 0); CHKERRXX(ierr);
      ierr = KSPGetIterationNumber(ksp, &nb_iterations); CHKERRXX(ierr);
      if(nb_iterations == 0)
      {
        // the solver converged if the initial guess was so close that ksp did not run a single iteration --> fixed-point reached, no further correction to expect, break...
        numbers_of_ksp_iterations.push_back(nb_iterations);
        max_corrections.push_back(0.0);
        relative_residuals.push_back(relative_residuals[relative_residuals.size()-1]);
        break;
      }
      if(!other_tilde_vectors_have_been_created)
      {
        P4EST_ASSERT(extension_cell_values_tilde == NULL);
        ierr = VecCreateGhostCells(p4est, ghost, &extension_cell_values_tilde); CHKERRXX(ierr);
        ierr = VecGetArray(extension_cell_values_tilde, &extension_cell_values_tilde_p); CHKERRXX(ierr);
        P4EST_ASSERT(extension_on_fine_nodes_tilde == NULL);
        ierr = VecCreateGhostNodes(fine_p4est, fine_nodes, &extension_on_fine_nodes_tilde); CHKERRXX(ierr);
        ierr = VecGetArray(extension_on_fine_nodes_tilde, &extension_on_fine_nodes_tilde_p); CHKERRXX(ierr);
        P4EST_ASSERT(corrected_rhs_tilde == NULL);
        ierr = VecCreateCellsNoGhost(p4est, &corrected_rhs_tilde); CHKERRXX(ierr);
        ierr = VecGetArray(corrected_rhs_tilde, &corrected_rhs_tilde_p); CHKERRXX(ierr);
        P4EST_ASSERT(residual_tilde == NULL);
        ierr = VecCreateCellsNoGhost(p4est, &residual_tilde); CHKERRXX(ierr);
        ierr = VecGetArray(residual_tilde, &residual_tilde_p); CHKERRXX(ierr);
        // update extension_cell_values_tilde <= extension_cell_values as well for good initial guess
        // the original vector's ghost values are already up-to-date, no need to GhostUpdate...
        // [knowingly avoiding separate Petsc operations, because of simplicity (no VecGhostGetLocalForm, etc.)]
        for (p4est_locidx_t qq = 0; qq < (p4est_locidx_t) (p4est->local_num_quadrants + ghost->ghosts.elem_count); ++qq)
          extension_cell_values_tilde_p[qq] = extension_cell_values_p[qq];

        other_tilde_vectors_have_been_created = true;
      }


      ierr = VecGhostUpdateBegin(solution_tilde, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(solution_tilde, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      extend_interface_values(solution_tilde_p, extension_cell_values_tilde, extension_on_fine_nodes_p);
      interpolate_coarse_cell_field_to_fine_nodes(extension_cell_values_tilde_p, extension_on_fine_nodes_tilde);
      get_corrected_rhs(corrected_rhs_tilde, extension_on_fine_nodes_tilde_p);

      ierr = MatMult(A, solution_tilde, residual_tilde); CHKERRXX(ierr);
      // finish the operation as residual -= corrected_rhs
      // --> can be done in the same loop as the calculation of relevant inner_product for min-res step
      // done in the following for optimization
      // (knowingly avoiding separate Petsc operations that would multiply the number of such loops)
      inner_products[0] = inner_products[1] = 0;
      for (p4est_locidx_t qq = 0; qq < p4est->local_num_quadrants; ++qq) {
        residual_tilde_p[qq]  -= corrected_rhs_tilde_p[qq];
        inner_products[0]     += residual_p[qq]*(residual_p[qq] - residual_tilde_p[qq]);
        inner_products[1]     += (residual_p[qq] - residual_tilde_p[qq])*(residual_p[qq] - residual_tilde_p[qq]);
      }
      mpiret = MPI_Allreduce(MPI_IN_PLACE, inner_products, 2, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); SC_CHECK_MPI(mpiret);
      step_size = inner_products[0]/inner_products[1];

      double max_correction = 0.0;
      norm_residual_and_corrected_rhs[0] = norm_residual_and_corrected_rhs[1] = 0.0;
      for (p4est_locidx_t qq = 0; qq < p4est->local_num_quadrants; ++qq) {
        max_correction                      = MAX(max_correction, fabs(step_size*(solution_tilde_p[qq] - solution_p[qq])));
        residual_p[qq]                      = (1.0-step_size)*residual_p[qq] + step_size*residual_tilde_p[qq];
        corrected_rhs_p[qq]                 = (1.0-step_size)*corrected_rhs_p[qq] + step_size*corrected_rhs_tilde_p[qq];
        norm_residual_and_corrected_rhs[0] += SQR(residual_p[qq]);
        norm_residual_and_corrected_rhs[1] += SQR(corrected_rhs_p[qq]);
        solution_p[qq]                      = (1.0-step_size)*solution_p[qq] + step_size*solution_tilde_p[qq];
        solution_tilde_p[qq]                = solution_p[qq]; // update initial guess for next KSPSolve
        extension_cell_values_p[qq]         = (1.0-step_size)*extension_cell_values_p[qq] + step_size*extension_cell_values_tilde_p[qq];
        extension_cell_values_tilde_p[qq]   = extension_cell_values_p[qq]; // update next extension_cell_values_tilde for initial guess for extension
        if(qq < (p4est_locidx_t) fine_nodes->indep_nodes.elem_count)
          extension_on_fine_nodes_p[qq]     = (1.0-step_size)*extension_on_fine_nodes_p[qq] + step_size*extension_on_fine_nodes_tilde_p[qq];
      }
      mpiret = MPI_Allreduce(MPI_IN_PLACE, &max_correction, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
      mpiret = MPI_Allreduce(MPI_IN_PLACE, norm_residual_and_corrected_rhs, 2, MPI_DOUBLE, MPI_SUM, p4est->mpicomm); SC_CHECK_MPI(mpiret);
      norm_residual_and_corrected_rhs[0] = sqrt(norm_residual_and_corrected_rhs[0]);
      norm_residual_and_corrected_rhs[1] = sqrt(norm_residual_and_corrected_rhs[1]);
      // update the ghost elements of ghosted cell-vectors
      for (p4est_locidx_t qq = p4est->local_num_quadrants; qq < (p4est_locidx_t) (p4est->local_num_quadrants + ghost->ghosts.elem_count); ++qq) {
        solution_p[qq]                      = (1.0-step_size)*solution_p[qq] + step_size*solution_tilde_p[qq];
        solution_tilde_p[qq]                = solution_p[qq]; // update initial guess for next KSPSolve
        extension_cell_values_p[qq]         = (1.0-step_size)*extension_cell_values_p[qq] + step_size*extension_cell_values_tilde_p[qq];
        extension_cell_values_tilde_p[qq]   = extension_cell_values_p[qq]; // update next extension_cell_values_tilde for initial guess for extension
        if(qq < (p4est_locidx_t) fine_nodes->indep_nodes.elem_count)
          extension_on_fine_nodes_p[qq]     = (1.0-step_size)*extension_on_fine_nodes_p[qq] + step_size*extension_on_fine_nodes_tilde_p[qq];
      }
      // update the last element of extension_on_fine_nodes if needed
      if((p4est_locidx_t) fine_nodes->indep_nodes.elem_count > (p4est_locidx_t) (p4est->local_num_quadrants + ghost->ghosts.elem_count))
        for (p4est_locidx_t qq = (p4est_locidx_t) (p4est->local_num_quadrants + ghost->ghosts.elem_count); qq < (p4est_locidx_t) fine_nodes->indep_nodes.elem_count; ++qq)
          extension_on_fine_nodes_p[qq]     = (1.0-step_size)*extension_on_fine_nodes_p[qq] + step_size*extension_on_fine_nodes_tilde_p[qq];



      numbers_of_ksp_iterations.push_back(nb_iterations);
      max_corrections.push_back(max_correction);
      relative_residuals.push_back(norm_residual_and_corrected_rhs[0]/norm_residual_and_corrected_rhs[1]);

      size_t iter = relative_residuals.size() -1;
      solver_has_converged =
          (max_correction < absolute_accuracy_threshold) && // if_nb_iterations != 0, max_correction must be below the accuracy requirement AND
          ((norm_residual_and_corrected_rhs[0]/norm_residual_and_corrected_rhs[1] < tolerance_on_rel_residual) || // either the relative residual is below the desired threshold as well
          (fabs(relative_residuals[iter] - relative_residuals[iter-1]) < 1.0e-6*MAX(relative_residuals[iter], relative_residuals[iter-1]))); // or we have reached a fixed-point for which the relative residual is above the threshold but can't really be made any smaller apparently
    }

    ierr = VecRestoreArray(corrected_rhs, &corrected_rhs_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(residual, &residual_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(solution, &solution_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(extension_cell_values, &extension_cell_values_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(extension_on_fine_nodes, &extension_on_fine_nodes_p); CHKERRXX(ierr);

    correct_jump_mu_grad_u();

    P4EST_ASSERT(solution_tilde_p != NULL);
    ierr = VecRestoreArray(solution_tilde, &solution_tilde_p); CHKERRXX(ierr);
    P4EST_ASSERT(solution_tilde != NULL);
    ierr = VecDestroy(solution_tilde); CHKERRXX(ierr);
    if(other_tilde_vectors_have_been_created)
    {
      P4EST_ASSERT(extension_cell_values_tilde_p != NULL);
      ierr = VecRestoreArray(extension_cell_values_tilde, &extension_cell_values_tilde_p); CHKERRXX(ierr);
      P4EST_ASSERT(extension_cell_values_tilde != NULL);
      ierr = VecDestroy(extension_cell_values_tilde); CHKERRXX(ierr);
      P4EST_ASSERT(extension_on_fine_nodes_tilde_p != NULL);
      ierr = VecRestoreArray(extension_on_fine_nodes_tilde, &extension_on_fine_nodes_tilde_p); CHKERRXX(ierr);
      P4EST_ASSERT(extension_on_fine_nodes_tilde != NULL);
      ierr = VecDestroy(extension_on_fine_nodes_tilde); CHKERRXX(ierr);
      P4EST_ASSERT(corrected_rhs_tilde_p != NULL);
      ierr = VecRestoreArray(corrected_rhs_tilde, &corrected_rhs_tilde_p); CHKERRXX(ierr);
      P4EST_ASSERT(corrected_rhs_tilde != NULL);
      ierr = VecDestroy(corrected_rhs_tilde); CHKERRXX(ierr);
      P4EST_ASSERT(residual_tilde_p != NULL);
      ierr = VecRestoreArray(residual_tilde, &residual_tilde_p); CHKERRXX(ierr);
      P4EST_ASSERT(residual_tilde != NULL);
      ierr = VecDestroy(residual_tilde); CHKERRXX(ierr);
    }

    ierr  = VecDestroy(residual); CHKERRXX(ierr);
  }
  solution_is_set = true;
  ierr = PetscLogEventEnd(log_my_p4est_xgfm_cells_solve, A, rhs, ksp, 0); CHKERRXX(ierr);
}

bool my_p4est_xgfm_cells_t::interface_neighbor_is_found(const p4est_locidx_t &quad_idx, const int &dir, interface_neighbor& int_nb)
{
  P4EST_ASSERT((quad_idx>=0) && (quad_idx < p4est->local_num_quadrants) && (dir >=0) && (dir < P4EST_FACES));
  std::map<p4est_locidx_t, std::map<int, interface_neighbor> >::const_iterator it = map_of_interface_neighbors.find(quad_idx);
  if(it != map_of_interface_neighbors.end())
  {
    std::map<int, interface_neighbor>::const_iterator itt = (it->second).find(dir);
    if(itt != (it->second).end())
    {
      int_nb = itt->second;
      return true;
    }
  }
  // not found in map!
  return false;
}

interface_neighbor my_p4est_xgfm_cells_t::get_interface_neighbor(const p4est_locidx_t &quad_idx, const int &dir,
                                                                 const p4est_locidx_t &tmp_quad_idx, const p4est_locidx_t &quad_fine_node_idx, const p4est_locidx_t &tmp_fine_node_idx,
                                                                 const double *phi_read_p, const double *phi_dd_read_p[])
{
  P4EST_ASSERT((quad_idx>=0) && (quad_idx < p4est->local_num_quadrants) && (dir >=0) && (dir < P4EST_FACES));
#ifdef DEBUG
  std::vector<p4est_quadrant_t> ngbd; ngbd.resize(0);
  p4est_topidx_t tree_idx = p4est->first_local_tree;
  while ((tree_idx < p4est->last_local_tree) && (quad_idx >= ((p4est_tree_t*) sc_array_index(p4est->trees, tree_idx+1))->quadrants_offset))
    tree_idx++;
  cell_ngbd->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, dir);
  P4EST_ASSERT((ngbd.size() == 1) && (ngbd[0].p.piggy3.local_num == tmp_quad_idx));
  splitting_criteria_t *data = (splitting_criteria_t*) p4est->user_pointer;
  p4est_tree_t* tree = (p4est_tree_t*) sc_array_index(p4est->trees, tree_idx);
  const p4est_quadrant_t* quadrant = (const p4est_quadrant_t*) sc_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);
  const p4est_quadrant_t* neighbor;
  if(tmp_quad_idx < p4est->local_num_quadrants)
  {
    p4est_topidx_t tree_idx_tmp = p4est->first_local_tree;
    while ((tree_idx_tmp < p4est->last_local_tree) && (tmp_quad_idx >= ((p4est_tree_t*) sc_array_index(p4est->trees, tree_idx_tmp + 1))->quadrants_offset))
      tree_idx_tmp++;
    p4est_tree_t* tree_tmp = (p4est_tree_t*) sc_array_index(p4est->trees, tree_idx_tmp);
    neighbor = (const p4est_quadrant_t*) sc_array_index(&tree_tmp->quadrants, tmp_quad_idx - tree_tmp->quadrants_offset);
  }
  else
    neighbor = (const p4est_quadrant_t*) sc_array_index(&ghost->ghosts, tmp_quad_idx - p4est->local_num_quadrants);
  P4EST_ASSERT((quadrant->level == neighbor->level) && (quadrant->level == ((int8_t) data->max_lvl)));
#endif

  interface_neighbor interface_nb;
  if(interface_neighbor_is_found(quad_idx, dir, interface_nb))
    return interface_nb;

  P4EST_ASSERT((quad_fine_node_idx >=0) && (quad_fine_node_idx < (p4est_locidx_t)(fine_nodes->indep_nodes.elem_count)));
  P4EST_ASSERT((tmp_fine_node_idx >=0) && (tmp_fine_node_idx < (p4est_locidx_t)(fine_nodes->indep_nodes.elem_count)));
  interface_nb.phi_q    = phi_read_p[quad_fine_node_idx];
  interface_nb.phi_tmp  = phi_read_p[tmp_fine_node_idx];
  P4EST_ASSERT(((interface_nb.phi_q > 0.0) && (interface_nb.phi_tmp <= 0.0)) || ((interface_nb.phi_q <= 0.0) && (interface_nb.phi_tmp > 0.0)));
  interface_nb.quad_tmp_idx             = tmp_quad_idx;
  interface_nb.quad_fine_node_idx       = quad_fine_node_idx;
  interface_nb.tmp_fine_node_idx        = tmp_fine_node_idx;
  interface_nb.mid_point_fine_node_idx  = fine_idx_of_direct_neighbor(fine_node_ngbd->neighbors[quad_fine_node_idx], dir);
  P4EST_ASSERT((interface_nb.mid_point_fine_node_idx >= 0) && (interface_nb.mid_point_fine_node_idx == fine_idx_of_direct_neighbor(fine_node_ngbd->neighbors[tmp_fine_node_idx], dir + ((dir%2==0)? +1:-1))));


  double mid_point_phi = phi_read_p[interface_nb.mid_point_fine_node_idx];
  double dd_phi_q, dd_phi_mid_point, dd_phi_tmp;
  dd_phi_q = dd_phi_mid_point = dd_phi_tmp = 0.0;
  if(second_derivatives_of_phi_are_set)
  {
    dd_phi_q          = phi_dd_read_p[dir/2][quad_fine_node_idx];
    dd_phi_tmp        = phi_dd_read_p[dir/2][tmp_fine_node_idx];
    dd_phi_mid_point  = phi_dd_read_p[dir/2][interface_nb.mid_point_fine_node_idx];
  }

  double theta;
  if ((( mid_point_phi > 0.0) && (interface_nb.phi_q > 0.0)) || ((mid_point_phi <= 0.0) && (interface_nb.phi_q <= 0.0)))
  {
    // mid_point on same side
    if(second_derivatives_of_phi_are_set)
      theta = (interface_nb.phi_q <= 0.0) ?
            0.5*(1.0+fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(mid_point_phi, interface_nb.phi_tmp, dd_phi_mid_point, dd_phi_tmp, .5*dxyz_min[dir/2])):
        0.5*(2.0-fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(mid_point_phi, interface_nb.phi_tmp, dd_phi_mid_point, dd_phi_tmp, .5*dxyz_min[dir/2]));
    else
      theta = (interface_nb.phi_q <= 0.0) ?
            0.5*(1.0+fraction_Interval_Covered_By_Irregular_Domain(mid_point_phi, interface_nb.phi_tmp, .5*dxyz_min[dir/2], .5*dxyz_min[dir/2])):
        0.5*(2.0-fraction_Interval_Covered_By_Irregular_Domain(mid_point_phi, interface_nb.phi_tmp, .5*dxyz_min[dir/2], .5*dxyz_min[dir/2]));
  }
  else
  {
    // mid_point on the other side
    if(second_derivatives_of_phi_are_set)
      theta = (interface_nb.phi_q <= 0.0) ?
            0.5*(fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(interface_nb.phi_q, mid_point_phi, dd_phi_q, dd_phi_mid_point, .5*dxyz_min[dir/2])):
        0.5*(1.0-fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(interface_nb.phi_q, mid_point_phi, dd_phi_q, dd_phi_mid_point, .5*dxyz_min[dir/2]));
    else
      theta = (interface_nb.phi_q <= 0.0) ?
            0.5*(fraction_Interval_Covered_By_Irregular_Domain(interface_nb.phi_q, mid_point_phi, .5*dxyz_min[dir/2], .5*dxyz_min[dir/2])):
        0.5*(1.0-fraction_Interval_Covered_By_Irregular_Domain(interface_nb.phi_q, mid_point_phi, .5*dxyz_min[dir/2], .5*dxyz_min[dir/2]));
  }
  if(theta<EPS) theta = 0.0;
  if(theta>1  ) theta = 1;

  interface_nb.theta = theta;

  interface_nb.mu_this_side   = (interface_nb.phi_q <= 0.0)? ((1.0 - .5*theta)*mu_m(quad_idx) + .5*theta*mu_m(tmp_quad_idx))          : ((1.0 - .5*theta)*mu_p(quad_idx) + .5*theta*mu_p(tmp_quad_idx));
  interface_nb.mu_other_side  = (interface_nb.phi_q <= 0.0)? (.5*(1.0 - theta)*mu_p(quad_idx) + .5*(1.0 + theta)*mu_p(tmp_quad_idx))  : (.5*(1.0 - theta)*mu_m(quad_idx) + .5*(1.0 + theta)*mu_m(tmp_quad_idx));

  map_of_interface_neighbors[quad_idx][dir] = interface_nb; // add it to the map so that future access is read in memory;
  return interface_nb;
}


void my_p4est_xgfm_cells_t::setup_negative_laplace_matrix_with_jumps()
{
  preallocate_matrix();

  double *null_space_p;
  if(!nullspace_use_fixed_point)
  {
    ierr = VecDuplicate(rhs, &null_space); CHKERRXX(ierr);
    ierr = VecGetArray(null_space, &null_space_p); CHKERRXX(ierr);
  }
  const double *phi_read_p, *phi_dd_read_p[P4EST_DIM];
  ierr = VecGetArrayRead(phi, &phi_read_p); CHKERRXX(ierr);
  if(second_derivatives_of_phi_are_set)
    for (short dim = 0; dim < P4EST_DIM; ++dim){
      ierr = VecGetArrayRead(phi_second_der[dim], &phi_dd_read_p[dim]); CHKERRXX(ierr);}
  bool is_quad_a_fine_node, is_tmp_a_fine_node;

  fixed_value_idx_g = p4est->global_num_quadrants;

  // register for logging purpose
  ierr = PetscLogEventBegin(log_my_p4est_xgfm_cells_matrix_setup, A, 0, 0, 0); CHKERRXX(ierr);

  std::vector<p4est_quadrant_t> ngbd;
  double cell_dxyz[P4EST_DIM];
  p4est_qcoord_t qxyz_quad[P4EST_DIM], qxyz_tmp[P4EST_DIM];
  p4est_topidx_t nxyz_tree_quad[P4EST_DIM], nxyz_tree_tmp[P4EST_DIM];
  p4est_locidx_t local_fine_indices_for_quad_interp[P4EST_CHILDREN], local_fine_indices_for_tmp_interp[P4EST_CHILDREN];
  double interp_weights_for_quad_interp[P4EST_CHILDREN], interp_weights_for_tmp_interp[P4EST_CHILDREN];
#ifdef CASL_THROWS
  bool is_face_a_fine_node;
  p4est_locidx_t local_fine_indices_for_face_interp[P4EST_CHILDREN];
  double interp_weights_for_face_interp[P4EST_CHILDREN];
#endif

  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx){
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for (short dim = 0; dim < P4EST_DIM; ++dim) {
      double rel_top_idx = (p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0] + dim] - xyz_min[dim])/tree_dimensions[dim];
      P4EST_ASSERT(fabs(rel_top_idx - floor(rel_top_idx)) < EPS);
      nxyz_tree_quad[dim]   = ((p4est_topidx_t) floor(rel_top_idx));
    }
    for (size_t q=0; q<tree->quadrants.elem_count; ++q){
      const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
      p4est_locidx_t quad_idx = q + tree->quadrants_offset;

      PetscInt quad_gloidx = quad_idx + p4est->global_first_quadrant[p4est->mpirank];

      qxyz_quad[0] = quad->x + P4EST_QUADRANT_LEN(quad->level+1);
      qxyz_quad[1] = quad->y + P4EST_QUADRANT_LEN(quad->level+1);
#ifdef P4_TO_P8
      qxyz_quad[2] = quad->z + P4EST_QUADRANT_LEN(quad->level+1);
#endif
      is_quad_a_fine_node = multilinear_interpolation_weights(nxyz_tree_quad, qxyz_quad, local_fine_indices_for_quad_interp, interp_weights_for_quad_interp);

      double phi_q = phi_read_p[local_fine_indices_for_quad_interp[0]]*interp_weights_for_quad_interp[0];
      for (short ccc = 1; ccc < (is_quad_a_fine_node ? 0 : P4EST_CHILDREN); ++ccc)
        phi_q += phi_read_p[local_fine_indices_for_quad_interp[ccc]]*interp_weights_for_quad_interp[ccc];


      double dtmp = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
      double cell_volume;
      cell_dxyz[0]  = tree_dimensions[0] * dtmp; cell_volume  = cell_dxyz[0];
      cell_dxyz[1]  = tree_dimensions[1] * dtmp; cell_volume *= cell_dxyz[1];
#ifdef P4_TO_P8
      cell_dxyz[2]  = tree_dimensions[2] * dtmp; cell_volume *= cell_dxyz[2];
#endif

      if(!nullspace_use_fixed_point) null_space_p[quad_idx] = 1;

      /* First add the diagonal term */
      double add_diag = (phi_q > 0.0)? add_diag_p.get_value(): add_diag_m.get_value();
      if(fabs(add_diag) > EPS)
      {
        ierr = MatSetValue(A, quad_gloidx, quad_gloidx, cell_volume*add_diag, ADD_VALUES); CHKERRXX(ierr);
        matrix_has_nullspace = false;
      }

      for(int dir=0; dir<P4EST_FACES; ++dir)
      {
        double face_area = cell_volume/cell_dxyz[dir/2];

        /* first check if the cell is a wall
         * We will assume that walls are not crossed by the interface, in a first attempt! */
        if(is_quad_Wall(p4est, tree_idx, quad, dir))
        {
          double wall_face_center[P4EST_DIM];
#ifdef CASL_THROWS
          p4est_qcoord_t qwall_face_center[P4EST_DIM];
#endif
          for (short dim = 0; dim < P4EST_DIM; ++dim) {
            wall_face_center[dim] = xyz_min[dim] + (nxyz_tree_quad[dim] + (((double) qxyz_quad[dim])/((double) P4EST_ROOT_LEN)))*tree_dimensions[dim] + ((dim == dir/2)? ((dir%2 == 0) ? -.5*cell_dxyz[dir/2] : .5*cell_dxyz[dir/2]): 0.0);
#ifdef CASL_THROWS
            qwall_face_center[dim]= qxyz_quad[dim]+ ((dim == dir/2)? ((dir%2 == 0) ? -P4EST_QUADRANT_LEN(quad->level+1) : P4EST_QUADRANT_LEN(quad->level+1)): 0.0);
#endif
          }
          if(bc->wallType(wall_face_center) == DIRICHLET)
          {
#ifdef CASL_THROWS
            is_face_a_fine_node = multilinear_interpolation_weights(nxyz_tree_quad, qwall_face_center, local_fine_indices_for_face_interp, interp_weights_for_face_interp);
            double phi_face = phi_read_p[local_fine_indices_for_face_interp[0]]*interp_weights_for_face_interp[0];
            for (short ccc = 1; ccc < ((is_face_a_fine_node)? 0 : P4EST_CHILDREN) ; ++ccc)
              phi_face += phi_read_p[local_fine_indices_for_face_interp[ccc]]*interp_weights_for_face_interp[ccc];
            int is_wall_cell_crossed = ((phi_q > 0.0)  && (phi_face <= 0.0)) || ((phi_q <= 0.0) && (phi_face >  0.0));
            if(is_wall_cell_crossed)
              throw std::invalid_argument("my_p4est_xgfm_cells_t::setup_negative_laplace_matrix_with_jumps() : a wall-cell is crossed by an interface with Dirichlet boundary condition...");
#endif
            matrix_has_nullspace = false;
            double mu = (phi_q > 0.0)? mu_p(wall_face_center): mu_m(wall_face_center);
            ierr = MatSetValue(A, quad_gloidx, quad_gloidx, 2*mu*face_area/cell_dxyz[dir/2], ADD_VALUES); CHKERRXX(ierr);
          }
          continue;
        }

        if(nullspace_use_fixed_point && quad_gloidx<fixed_value_idx_g)
        {
          fixed_value_idx_l = quad_idx;
          fixed_value_idx_g = quad_gloidx;
        }

        /* now get the neighbors */
        ngbd.resize(0);
        cell_ngbd->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, dir);

        if(ngbd.size()==1)
        {
          int8_t level_tmp = ngbd[0].level;
          p4est_locidx_t quad_tmp_idx = ngbd[0].p.piggy3.local_num;
          p4est_topidx_t tree_tmp_idx = ngbd[0].p.piggy3.which_tree;

          for (short dim = 0; dim < P4EST_DIM; ++dim) {
            double rel_top_idx = (p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_tmp_idx + 0] + dim] - xyz_min[dim])/tree_dimensions[dim];
            P4EST_ASSERT(fabs(rel_top_idx - floor(rel_top_idx)) < EPS);
            nxyz_tree_tmp[dim] = ((p4est_topidx_t) floor(rel_top_idx));
          }
          qxyz_tmp[0] = ngbd[0].x + P4EST_QUADRANT_LEN(ngbd[0].level+1);
          qxyz_tmp[1] = ngbd[0].y + P4EST_QUADRANT_LEN(ngbd[0].level+1);
#ifdef P4_TO_P8
          qxyz_tmp[2] = ngbd[0].z + P4EST_QUADRANT_LEN(ngbd[0].level+1);
#endif
          is_tmp_a_fine_node = multilinear_interpolation_weights(nxyz_tree_tmp, qxyz_tmp, local_fine_indices_for_tmp_interp, interp_weights_for_tmp_interp);
          double phi_tmp = phi_read_p[local_fine_indices_for_tmp_interp[0]]*interp_weights_for_tmp_interp[0];
          for (short ccc = 1; ccc < ((is_tmp_a_fine_node) ? 0: P4EST_CHILDREN); ++ccc)
            phi_tmp += phi_read_p[local_fine_indices_for_tmp_interp[ccc]]*interp_weights_for_tmp_interp[ccc];

          /* If interface across the two cells.
           * We assume that the interface is tesselated with uniform finest grid level */
          if(((phi_q > 0.0) && (phi_tmp <= 0.0)) || ((phi_q <= 0.0) && (phi_tmp > 0.0)))
          {
            P4EST_ASSERT(is_quad_a_fine_node && is_tmp_a_fine_node);
            interface_neighbor int_nb = get_interface_neighbor(quad_idx, dir, quad_tmp_idx, local_fine_indices_for_quad_interp[0], local_fine_indices_for_tmp_interp[0], phi_read_p, phi_dd_read_p);
            double mu_across = int_nb.mu_this_side*int_nb.mu_other_side/((1.0-int_nb.theta)*int_nb.mu_this_side + int_nb.theta*int_nb.mu_other_side);

            ierr = MatSetValue(A, quad_gloidx, quad_gloidx,                         mu_across * face_area/dxyz_min[dir/2], ADD_VALUES); CHKERRXX(ierr);
            ierr = MatSetValue(A, quad_gloidx, compute_global_index(quad_tmp_idx), -mu_across * face_area/dxyz_min[dir/2], ADD_VALUES); CHKERRXX(ierr);
          }
          /* no interface - regular discretization */
          else
          {
            double s_tmp = pow((double)P4EST_QUADRANT_LEN(ngbd[0].level)/(double)P4EST_ROOT_LEN, (double)P4EST_DIM-1);

            ngbd.resize(0);
            cell_ngbd->find_neighbor_cells_of_cell(ngbd, quad_tmp_idx, tree_tmp_idx, dir%2==0 ? dir+1 : dir-1);

            std::vector<double> s_ng(ngbd.size());
            for(unsigned int i=0; i<ngbd.size(); ++i)
              s_ng[i] = pow((double)P4EST_QUADRANT_LEN(ngbd[i].level)/(double)P4EST_ROOT_LEN, (double)P4EST_DIM-1);

            double d = 0;
            for(unsigned int i=0; i<ngbd.size(); ++i)
              d += s_ng[i]/s_tmp * 0.5 * (double)(P4EST_QUADRANT_LEN(level_tmp)+P4EST_QUADRANT_LEN(ngbd[i].level))/(double)P4EST_ROOT_LEN;

            d *= tree_dimensions[dir/2];

            double mu_tmp = 0.0;
            double mu_nb;
            double xyz_face[P4EST_DIM];
            for(unsigned int i=0; i<ngbd.size(); ++i)
            {
              xyz_face[0] = quad_x_fr_q(ngbd[i].p.piggy3.local_num, ngbd[i].p.piggy3.which_tree, p4est, ghost);
              xyz_face[1] = quad_y_fr_q(ngbd[i].p.piggy3.local_num, ngbd[i].p.piggy3.which_tree, p4est, ghost);
#ifdef P4_TO_P8
              xyz_face[2] = quad_z_fr_q(ngbd[i].p.piggy3.local_num, ngbd[i].p.piggy3.which_tree, p4est, ghost);
#endif
              xyz_face[dir/2] += ((dir%2 == 0)? -1.0: +1.0)*(.5*((double)(P4EST_QUADRANT_LEN(ngbd[i].level))/(double)P4EST_ROOT_LEN))*tree_dimensions[dir/2];

              mu_nb = (phi_q <= 0.0)? mu_m(xyz_face) : mu_p(xyz_face);
              ierr = MatSetValue(A, quad_gloidx, compute_global_index(ngbd[i].p.piggy3.local_num), mu_nb*face_area * s_ng[i]/s_tmp/d, ADD_VALUES); CHKERRXX(ierr);
              mu_tmp += mu_nb*s_ng[i]/s_tmp;
            }

            ierr = MatSetValue(A, quad_gloidx, compute_global_index(quad_tmp_idx), -mu_tmp*face_area/d, ADD_VALUES); CHKERRXX(ierr);
          }
        }
        /* there is more than one neighbor, regular bulk case. This assumes uniform on interface ! */
        else if(ngbd.size()>1)
        {
          double s_tmp = pow((double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN, (double)P4EST_DIM-1);

          std::vector<double> s_ng(ngbd.size());
          for(unsigned int i=0; i<ngbd.size(); ++i)
            s_ng[i] = pow((double)P4EST_QUADRANT_LEN(ngbd[i].level)/(double)P4EST_ROOT_LEN, (double)P4EST_DIM-1);

          double d = 0;
          for(unsigned int i=0; i<ngbd.size(); ++i)
            d += s_ng[i]/s_tmp * 0.5 * (double)(P4EST_QUADRANT_LEN(quad->level)+P4EST_QUADRANT_LEN(ngbd[i].level))/(double)P4EST_ROOT_LEN;

          d *= tree_dimensions[dir/2];

          double mu_tmp = 0.0;
          double mu_nb;
          double xyz_face[P4EST_DIM];
          for(unsigned int i=0; i<ngbd.size(); ++i)
          {
            xyz_face[0] = quad_x_fr_q(ngbd[i].p.piggy3.local_num, ngbd[i].p.piggy3.which_tree, p4est, ghost);
            xyz_face[1] = quad_y_fr_q(ngbd[i].p.piggy3.local_num, ngbd[i].p.piggy3.which_tree, p4est, ghost);
#ifdef P4_TO_P8
            xyz_face[2] = quad_z_fr_q(ngbd[i].p.piggy3.local_num, ngbd[i].p.piggy3.which_tree, p4est, ghost);
#endif
            xyz_face[dir/2] += ((dir%2 == 0)? +1.0: -1.0)*(.5*((double)(P4EST_QUADRANT_LEN(ngbd[i].level))/(double)P4EST_ROOT_LEN))*tree_dimensions[dir/2];

            mu_nb = (phi_q <= 0.0)? mu_m(xyz_face) : mu_p(xyz_face);
            ierr = MatSetValue(A, quad_gloidx, compute_global_index(ngbd[i].p.piggy3.local_num), -mu_nb*face_area * s_ng[i]/s_tmp/d, ADD_VALUES); CHKERRXX(ierr);
            mu_tmp += mu_nb*s_ng[i]/s_tmp;
          }

          ierr = MatSetValue(A, quad_gloidx, quad_gloidx, mu_tmp*face_area/d, ADD_VALUES); CHKERRXX(ierr);

        }
      }
    }
  }

  if(!nullspace_use_fixed_point)
  {
    ierr = VecRestoreArray(null_space, &null_space_p); CHKERRXX(ierr);
  }


  // Assemble the matrix
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
  ierr = MatAssemblyEnd  (A, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);

  // check for null space
  ierr = MPI_Allreduce(MPI_IN_PLACE, &matrix_has_nullspace, 1, MPI_INT, MPI_LAND, p4est->mpicomm); CHKERRXX(ierr);
  if (matrix_has_nullspace)
  {
    if(!nullspace_use_fixed_point)
    {
      if(A_null_space != NULL){
        ierr = MatNullSpaceDestroy(A_null_space); CHKERRXX(ierr);}

      double norm;
      ierr = VecNormalize(null_space, &norm); CHKERRXX(ierr);
      ierr = MatNullSpaceCreate(p4est->mpicomm, PETSC_FALSE, 1, &null_space, &A_null_space); CHKERRXX(ierr);

      ierr = MatSetNullSpace(A, A_null_space); CHKERRXX(ierr);
      ierr = MatSetTransposeNullSpace(A, A_null_space); CHKERRXX(ierr);
    }
    else
    {
      ierr = MatSetOption(A, MAT_NO_OFF_PROC_ZERO_ROWS, PETSC_TRUE); CHKERRXX(ierr);
      p4est_gloidx_t fixed_value_idx;
      MPI_Allreduce(&fixed_value_idx_g, &fixed_value_idx, 1, MPI_LONG_LONG_INT, MPI_MIN, p4est->mpicomm);
      if(fixed_value_idx_g>=p4est->global_num_quadrants)
        throw std::invalid_argument("my_p4est_poisson_cells_t->setup_negative_laplace_matrix: could not fix value for all neumann problem. Maybe there is no point inside the domain and away from the interface?");
      if (fixed_value_idx_g != fixed_value_idx){ // we are not setting the fixed value
        fixed_value_idx_l = -1;
        fixed_value_idx_g = fixed_value_idx;
        ierr = MatZeroRows(A, 0, (PetscInt*)(&fixed_value_idx_g), 1.0, NULL, NULL); CHKERRXX(ierr);
      }
      else { // reset the value
        ierr = MatZeroRows(A, 1, (PetscInt*)(&fixed_value_idx_g), 1.0, NULL, NULL); CHKERRXX(ierr);}
    }
  }

  if(!nullspace_use_fixed_point)
  {
    ierr = VecDestroy(null_space); CHKERRXX(ierr);
  }

  is_matrix_built = true;

  ierr = VecRestoreArrayRead(phi, &phi_read_p); CHKERRXX(ierr);
  if(second_derivatives_of_phi_are_set)
    for (short dim = 0; dim < P4EST_DIM; ++dim)
    {
      ierr = VecRestoreArrayRead(phi_second_der[dim], &phi_dd_read_p[dim]); CHKERRXX(ierr);
    }

  ierr = PetscLogEventEnd(log_my_p4est_xgfm_cells_matrix_setup, A, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_xgfm_cells_t::setup_negative_laplace_rhsvec_with_jumps()
{
  // register for logging purpose
  ierr = PetscLogEventBegin(log_my_p4est_xgfm_cells_rhsvec_setup, 0, 0, 0, 0); CHKERRXX(ierr);

  double *rhs_p;
  const double *phi_read_p, *phi_dd_read_p[P4EST_DIM], *jump_u_read_p, *jump_mu_grad_u_read_p[P4EST_DIM];
  ierr    = VecGetArray(rhs,    &rhs_p   ); CHKERRXX(ierr);
  ierr    = VecGetArrayRead(phi, &phi_read_p); CHKERRXX(ierr);
  ierr    = VecGetArrayRead(jump_u, &jump_u_read_p); CHKERRXX(ierr);
  for (short dim = 0; dim < P4EST_DIM; ++dim) {
    ierr  = VecGetArrayRead(jump_mu_grad_u[dim], &jump_mu_grad_u_read_p[dim]); CHKERRXX(ierr);
    if(second_derivatives_of_phi_are_set){
      ierr  = VecGetArrayRead(phi_second_der[dim], &phi_dd_read_p[dim]); CHKERRXX(ierr);}
  }
  bool is_quad_a_fine_node, is_tmp_a_fine_node;

  std::vector<p4est_quadrant_t> ngbd;
  double cell_dxyz[P4EST_DIM];
  p4est_qcoord_t qxyz_quad[P4EST_DIM], qxyz_tmp[P4EST_DIM];
  p4est_topidx_t nxyz_tree_quad[P4EST_DIM], nxyz_tree_tmp[P4EST_DIM];
  p4est_locidx_t local_fine_indices_for_quad_interp[P4EST_CHILDREN], local_fine_indices_for_tmp_interp[P4EST_CHILDREN];
  double interp_weights_for_quad_interp[P4EST_CHILDREN], interp_weights_for_tmp_interp[P4EST_CHILDREN];
#ifdef CASL_THROWS
  bool is_face_a_fine_node;
  p4est_locidx_t local_fine_indices_for_face_interp[P4EST_CHILDREN];
  double interp_weights_for_face_interp[P4EST_CHILDREN];
#endif

  /* Main loop over all local quadrant */
  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx){
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for (short dim = 0; dim < P4EST_DIM; ++dim) {
      double rel_top_idx = (p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0] + dim] - xyz_min[dim])/tree_dimensions[dim];
      P4EST_ASSERT(fabs(rel_top_idx - floor(rel_top_idx)) < EPS);
      nxyz_tree_quad[dim]   = ((p4est_topidx_t) floor(rel_top_idx));
    }

    for (size_t q=0; q<tree->quadrants.elem_count; ++q){
      const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
      p4est_locidx_t quad_idx = q + tree->quadrants_offset;

      qxyz_quad[0] = quad->x + P4EST_QUADRANT_LEN(quad->level+1);
      qxyz_quad[1] = quad->y + P4EST_QUADRANT_LEN(quad->level+1);
#ifdef P4_TO_P8
      qxyz_quad[2] = quad->z + P4EST_QUADRANT_LEN(quad->level+1);
#endif

      is_quad_a_fine_node = multilinear_interpolation_weights(nxyz_tree_quad, qxyz_quad, local_fine_indices_for_quad_interp, interp_weights_for_quad_interp);
      double phi_q = phi_read_p[local_fine_indices_for_quad_interp[0]]*interp_weights_for_quad_interp[0];
      for (short ccc = 1; ccc < (is_quad_a_fine_node ? 0 : P4EST_CHILDREN); ++ccc)
        phi_q += phi_read_p[local_fine_indices_for_quad_interp[ccc]]*interp_weights_for_quad_interp[ccc];

      double dtmp = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
      double cell_volume;
      cell_dxyz[0]  = tree_dimensions[0] * dtmp; cell_volume  = cell_dxyz[0];
      cell_dxyz[1]  = tree_dimensions[1] * dtmp; cell_volume *= cell_dxyz[1];
#ifdef P4_TO_P8
      cell_dxyz[2]  = tree_dimensions[2] * dtmp; cell_volume *= cell_dxyz[2];
#endif
      rhs_p[quad_idx] *= cell_volume;

      for(int dir=0; dir<P4EST_FACES; ++dir)
      {
        double face_area = cell_volume/cell_dxyz[dir/2];

        /* first check if the cell is a wall
         * We will assume that walls are not crossed by the interface, in a first attempt! */
        if(is_quad_Wall(p4est, tree_idx, quad, dir))
        {
          double wall_face_center[P4EST_DIM];
#ifdef CASL_THROWS
          p4est_qcoord_t qwall_face_center[P4EST_DIM];
#endif
          for (short dim = 0; dim < P4EST_DIM; ++dim) {
            wall_face_center[dim] = xyz_min[dim] + (nxyz_tree_quad[dim] + (((double) qxyz_quad[dim])/((double) P4EST_ROOT_LEN)))*tree_dimensions[dim] + ((dim == dir/2)? ((dir%2 == 0) ? -.5*cell_dxyz[dir/2] : .5*cell_dxyz[dir/2]): 0.0);
#ifdef CASL_THROWS
            qwall_face_center[dim]= qxyz_quad[dim]+ ((dim == dir/2)? ((dir%2 == 0) ? -P4EST_QUADRANT_LEN(quad->level+1) : P4EST_QUADRANT_LEN(quad->level+1)): 0.0);
#endif
          }
#ifdef CASL_THROWS
          is_face_a_fine_node = multilinear_interpolation_weights(nxyz_tree_quad, qwall_face_center, local_fine_indices_for_face_interp, interp_weights_for_face_interp);
          double phi_face = phi_read_p[local_fine_indices_for_face_interp[0]]*interp_weights_for_face_interp[0];
          for (short ccc = 1; ccc < ((is_face_a_fine_node)? 0 : P4EST_CHILDREN) ; ++ccc)
            phi_face += phi_read_p[local_fine_indices_for_face_interp[ccc]]*interp_weights_for_face_interp[ccc];
          int is_wall_cell_crossed = ((phi_q > 0.0)  && (phi_face <= 0.0)) || ((phi_q <= 0.0) && (phi_face >  0.0));
          if(is_wall_cell_crossed)
            throw std::invalid_argument("my_p4est_xgfm_cells_t::setup_negative_laplace_rhsvec_with_jumps(): a wall-cell is crossed by the interface...");
#endif
          double mu           = (phi_q > 0.0)? mu_p(wall_face_center): mu_m(wall_face_center);
          double val_wall     = bc->wallValue(wall_face_center);
          switch(bc->wallType(wall_face_center))
          {
          case DIRICHLET:
            rhs_p[quad_idx]  += 2.0*mu*face_area*val_wall/cell_dxyz[dir/2];
            break;
          case NEUMANN:
            rhs_p[quad_idx]  += mu*face_area*val_wall;
            break;
          default:
            throw std::invalid_argument("my_p4est_xgfm_cells_t::setup_negative_laplace_rhsvec_with_jumps(): unknown boundary condition on a wall.");
          }
          continue;
        }

        /* now get the neighbors */
        ngbd.resize(0);
        cell_ngbd->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, dir);

        if(ngbd.size()==1)
        {
          p4est_locidx_t quad_tmp_idx = ngbd[0].p.piggy3.local_num;
          p4est_topidx_t tree_tmp_idx = ngbd[0].p.piggy3.which_tree;

          for (short dim = 0; dim < P4EST_DIM; ++dim) {
            double rel_top_idx = (p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_tmp_idx + 0] + dim] - xyz_min[dim])/tree_dimensions[dim];
            P4EST_ASSERT(fabs(rel_top_idx - floor(rel_top_idx)) < EPS);
            nxyz_tree_tmp[dim] = ((p4est_topidx_t) floor(rel_top_idx));
          }
          qxyz_tmp[0] = ngbd[0].x + P4EST_QUADRANT_LEN(ngbd[0].level+1);
          qxyz_tmp[1] = ngbd[0].y + P4EST_QUADRANT_LEN(ngbd[0].level+1);
#ifdef P4_TO_P8
          qxyz_tmp[2] = ngbd[0].z + P4EST_QUADRANT_LEN(ngbd[0].level+1);
#endif
          is_tmp_a_fine_node = multilinear_interpolation_weights(nxyz_tree_tmp, qxyz_tmp, local_fine_indices_for_tmp_interp, interp_weights_for_tmp_interp);
          double phi_tmp = phi_read_p[local_fine_indices_for_tmp_interp[0]]*interp_weights_for_tmp_interp[0];
          for (short ccc = 1; ccc < ((is_tmp_a_fine_node) ? 0: P4EST_CHILDREN); ++ccc)
            phi_tmp += phi_read_p[local_fine_indices_for_tmp_interp[ccc]]*interp_weights_for_tmp_interp[ccc];
          /* If interface across the two cells.
           * We assume that the interface is tesselated with uniform finest grid level */
          if(((phi_q > 0.0) && (phi_tmp <= 0.0)) || ((phi_q <= 0.0) && (phi_tmp > 0.0)))
          {
            P4EST_ASSERT(is_quad_a_fine_node && is_tmp_a_fine_node);
            interface_neighbor int_nb = get_interface_neighbor(quad_idx, dir, quad_tmp_idx, local_fine_indices_for_quad_interp[0], local_fine_indices_for_tmp_interp[0], phi_read_p, phi_dd_read_p);

            double mu_across = int_nb.mu_this_side*int_nb.mu_other_side/((1.0-int_nb.theta)*int_nb.mu_this_side + int_nb.theta*int_nb.mu_other_side);
            double jump_sol_across, jump_flux_comp_across;
            if (int_nb.theta >=0.5)
            {
              // mid_point on same side
              P4EST_ASSERT(int_nb.theta <= 1.0);
              jump_sol_across         = (2.0 - 2.0*int_nb.theta)*jump_u_read_p[int_nb.mid_point_fine_node_idx]                + (2.0*int_nb.theta - 1.0)*jump_u_read_p[int_nb.tmp_fine_node_idx];
              jump_flux_comp_across   = (2.0 - 2.0*int_nb.theta)*jump_mu_grad_u_read_p[dir/2][int_nb.mid_point_fine_node_idx] + (2.0*int_nb.theta - 1.0)*jump_mu_grad_u_read_p[dir/2][int_nb.tmp_fine_node_idx];
            }
            else
            {
              // mid_point on the other side
              P4EST_ASSERT(int_nb.theta >= 0.0);
              jump_sol_across         = 2.0*int_nb.theta*jump_u_read_p[int_nb.mid_point_fine_node_idx]                + (1.0 - 2.0*int_nb.theta)*jump_u_read_p[int_nb.quad_fine_node_idx];
              jump_flux_comp_across   = 2.0*int_nb.theta*jump_mu_grad_u_read_p[dir/2][int_nb.mid_point_fine_node_idx] + (1.0 - 2.0*int_nb.theta)*jump_mu_grad_u_read_p[dir/2][int_nb.quad_fine_node_idx];
            }
            double sign_jump = (phi_q > 0.0)?+1.0:-1.0;
            double sign_jump_in_flux_factor = (dir%2==0)? -1.0:+1.0;

            rhs_p[quad_idx] += sign_jump*mu_across*( sign_jump_in_flux_factor*jump_flux_comp_across*(1-int_nb.theta)/int_nb.mu_other_side + jump_sol_across/dxyz_min[dir/2])*face_area;
          }
        }
      }
    }
  }

  if (matrix_has_nullspace)
  {
    if(!nullspace_use_fixed_point){
      ierr = MatNullSpaceRemove(A_null_space, rhs, NULL); CHKERRXX(ierr);}
    else if(fixed_value_idx_l >= 0)
      rhs_p[fixed_value_idx_l] = 0;
  }

  // restore the pointers
  for (short dim = 0; dim < P4EST_DIM; ++dim) {
    ierr  = VecRestoreArrayRead(jump_mu_grad_u[dim], &jump_mu_grad_u_read_p[dim]); CHKERRXX(ierr);
    if(second_derivatives_of_phi_are_set)
    {
      ierr  = VecRestoreArrayRead(phi_second_der[dim], &phi_dd_read_p[dim]); CHKERRXX(ierr);
    }
  }
  ierr    = VecRestoreArrayRead(jump_u, &jump_u_read_p); CHKERRXX(ierr);
  ierr    = VecRestoreArrayRead(phi, &phi_read_p); CHKERRXX(ierr);
  ierr    = VecRestoreArray(rhs,    &rhs_p   ); CHKERRXX(ierr);

  ierr    = PetscLogEventEnd(log_my_p4est_xgfm_cells_rhsvec_setup, rhs, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_xgfm_cells_t::update_interface_values(Vec new_cell_extension, const double *solution_read_p, const double *extension_on_fine_nodes_read_p)
{
#ifdef CASL_THROWS
  // compare local size, global size and ghost layers
  PetscInt local_size, global_size, ghost_layer_size;

  ierr = VecGetLocalSize(new_cell_extension, &local_size); CHKERRXX(ierr);
  int my_error = ( ((PetscInt) p4est->local_num_quadrants) != local_size);

  ierr = VecGetSize(new_cell_extension, &global_size); CHKERRXX(ierr);
  my_error = my_error || (global_size != ((PetscInt) p4est->global_num_quadrants));

  Vec new_cell_extension_loc;
  ierr = VecGhostGetLocalForm(new_cell_extension, &new_cell_extension_loc); CHKERRXX(ierr);
  ierr = VecGetSize(new_cell_extension_loc, &ghost_layer_size); CHKERRXX(ierr);
  ghost_layer_size -= local_size;
  my_error = my_error || (ghost_layer_size != ((PetscInt) (ghost->ghosts.elem_count)));
  ierr = VecGhostRestoreLocalForm(new_cell_extension, &new_cell_extension_loc); CHKERRXX(ierr);

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &my_error, 1, MPI_INT, MPI_LOR, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  if(my_error)
    throw std::invalid_argument("my_p4est_xgfm_cells_t::extend_interface_values(const double *, Vec , const double* , bool, double, uint) : the second vector argument must be preallocated and have the same layout as if constructed with VecCreateGhostCells on the coarse grid...");

  my_error = !phi_has_been_set;
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &my_error, 1, MPI_INT, MPI_LOR, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  if(my_error)
    throw std::invalid_argument("my_p4est_xgfm_cells_t::extend_interface_values(const double *, Vec , const double* , bool, double, uint) : the levelset must be set beforehand...");
  my_error = !jumps_have_been_set;
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &my_error, 1, MPI_INT, MPI_LOR, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  if(my_error)
    throw std::invalid_argument("my_p4est_xgfm_cells_t::extend_interface_values(const double *, Vec , const double* , bool, double, uint) : the jump values must be set beforehand...");
#endif

#ifdef DEBUG
  bool fine_extension_is_defined = (extension_on_fine_nodes_read_p != NULL);
#endif
  P4EST_ASSERT((!interface_values_are_set? !fine_extension_is_defined : fine_extension_is_defined));

  const double *phi_read_p, *normals_read_p[P4EST_DIM], *phi_dd_read_p[P4EST_DIM], *jump_u_read_p, *jump_mu_grad_u_read_p[P4EST_DIM];
  ierr = VecGetArrayRead(phi, &phi_read_p); CHKERRXX(ierr);
  for (short dim = 0; dim < P4EST_DIM; ++dim)
  {
    ierr = VecGetArrayRead(normals[dim], &normals_read_p[dim]); CHKERRXX(ierr);
    ierr = VecGetArrayRead(jump_mu_grad_u[dim], &jump_mu_grad_u_read_p[dim]); CHKERRXX(ierr);
    if(second_derivatives_of_phi_are_set){
      ierr = VecGetArrayRead(phi_second_der[dim], &phi_dd_read_p[dim]); CHKERRXX(ierr);}
  }
  ierr = VecGetArrayRead(jump_u, &jump_u_read_p); CHKERRXX(ierr);

  if(interface_values_are_set)
  {
    for (std::map<p4est_locidx_t, std::map<int, interface_neighbor> >::const_iterator it = map_of_interface_neighbors.begin();
         it != map_of_interface_neighbors.end(); ++it)
    {
      p4est_locidx_t quad_idx = it->first;
      for (std::map<int, interface_neighbor>::const_iterator itt = map_of_interface_neighbors[quad_idx].begin();
           itt != map_of_interface_neighbors[quad_idx].end(); ++itt)
      {
        int dir                     = itt->first;
        interface_neighbor& int_nb  = map_of_interface_neighbors[quad_idx][dir];
        P4EST_ASSERT(((int_nb.phi_q > 0.0) && (int_nb.phi_tmp <= 0.0)) || ((int_nb.phi_q <= 0.0) && (int_nb.phi_tmp > 0.0)));
        P4EST_ASSERT(int_nb.mid_point_fine_node_idx >= 0);

        double jump_sol_across, jump_flux_comp_across;
        double jump_mu_mid_point = .5*(mu_p(quad_idx) + mu_p(int_nb.quad_tmp_idx) - mu_m(quad_idx) - mu_m(int_nb.quad_tmp_idx));
        double jump_mu_other_fine_node;
        p4est_locidx_t other_fine_node_idx;
        if (int_nb.theta >= 0.5)
        {
          P4EST_ASSERT(int_nb.theta <= 1.0);
          // mid_point and point across are fine nodes!
          jump_sol_across         = (2.0 - 2.0*int_nb.theta)*jump_u_read_p[int_nb.mid_point_fine_node_idx]                + (2.0*int_nb.theta - 1.0)*jump_u_read_p[int_nb.tmp_fine_node_idx];
          jump_flux_comp_across   = (2.0 - 2.0*int_nb.theta)*jump_mu_grad_u_read_p[dir/2][int_nb.mid_point_fine_node_idx] + (2.0*int_nb.theta - 1.0)*jump_mu_grad_u_read_p[dir/2][int_nb.tmp_fine_node_idx];
          jump_mu_other_fine_node = mu_p(int_nb.quad_tmp_idx) - mu_m(int_nb.quad_tmp_idx);
          other_fine_node_idx     = int_nb.tmp_fine_node_idx;
        }
        else
        {
          P4EST_ASSERT(int_nb.theta>=0.0);
          // quad and mid-point are fine nodes!
          jump_sol_across         = 2.0*int_nb.theta*jump_u_read_p[int_nb.mid_point_fine_node_idx]                + (1.0 - 2.0*int_nb.theta)*jump_u_read_p[int_nb.quad_fine_node_idx];
          jump_flux_comp_across   = 2.0*int_nb.theta*jump_mu_grad_u_read_p[dir/2][int_nb.mid_point_fine_node_idx] + (1.0 - 2.0*int_nb.theta)*jump_mu_grad_u_read_p[dir/2][int_nb.quad_fine_node_idx];
          jump_mu_other_fine_node = mu_p(quad_idx) - mu_m(quad_idx);
          other_fine_node_idx     = int_nb.quad_fine_node_idx;
        }

        const quad_neighbor_nodes_of_node_t *qnnn_mid_point_fine_node, *qnnn_other_fine_node;
        qnnn_mid_point_fine_node  = &fine_node_ngbd->neighbors[int_nb.mid_point_fine_node_idx];
        qnnn_other_fine_node      = &fine_node_ngbd->neighbors[other_fine_node_idx];
        double jump_correction_mid_point[] = {
          jump_mu_mid_point*qnnn_mid_point_fine_node->dx_central(extension_on_fine_nodes_read_p)
          , jump_mu_mid_point*qnnn_mid_point_fine_node->dy_central(extension_on_fine_nodes_read_p)
  #ifdef P4_TO_P8
          , jump_mu_mid_point*qnnn_mid_point_fine_node->dz_central(extension_on_fine_nodes_read_p)
  #endif
        };
        double jump_correction_other_fine_node[] = {
          jump_mu_other_fine_node*qnnn_other_fine_node->dx_central(extension_on_fine_nodes_read_p)
          , jump_mu_other_fine_node*qnnn_other_fine_node->dy_central(extension_on_fine_nodes_read_p)
  #ifdef P4_TO_P8
          , jump_mu_other_fine_node*qnnn_other_fine_node->dz_central(extension_on_fine_nodes_read_p)
  #endif
        };
        double normal_vector_mid_point[P4EST_DIM], normal_vector_other_fine_node[P4EST_DIM], norm_normal_vector_mid_point, norm_normal_vector_other_fine_node, inner_product_mid_point, inner_product_other_fine_node;
        norm_normal_vector_mid_point          = norm_normal_vector_other_fine_node = inner_product_mid_point = inner_product_other_fine_node = 0.0;
        for (short dim = 0; dim < P4EST_DIM; ++dim)
        {
          normal_vector_mid_point[dim]        = normals_read_p[dim][int_nb.mid_point_fine_node_idx];
          normal_vector_other_fine_node[dim]  = normals_read_p[dim][other_fine_node_idx];
          norm_normal_vector_mid_point       += SQR(normal_vector_mid_point[dim]);
          norm_normal_vector_other_fine_node += SQR(normal_vector_other_fine_node[dim]);
          inner_product_mid_point            += jump_correction_mid_point[dim]*normal_vector_mid_point[dim];
          inner_product_other_fine_node      += jump_correction_other_fine_node[dim]*normal_vector_other_fine_node[dim];
        }
        norm_normal_vector_mid_point          = sqrt(norm_normal_vector_mid_point);
        norm_normal_vector_other_fine_node    = sqrt(norm_normal_vector_other_fine_node);
        P4EST_ASSERT((norm_normal_vector_mid_point > EPS) && (norm_normal_vector_other_fine_node > EPS));
        inner_product_mid_point              /= norm_normal_vector_mid_point;
        inner_product_other_fine_node        /= norm_normal_vector_other_fine_node;

        double jump_correction[P4EST_DIM];
        for (short dim = 0; dim < P4EST_DIM; ++dim)
        {
          normal_vector_mid_point[dim]         /= norm_normal_vector_mid_point;
          normal_vector_other_fine_node[dim]   /= norm_normal_vector_other_fine_node;
          jump_correction_mid_point[dim]       -= normal_vector_mid_point[dim]*inner_product_mid_point;
          jump_correction_other_fine_node[dim] -= normal_vector_other_fine_node[dim]*inner_product_other_fine_node;
          if (int_nb.theta >= 0.5) // mid_point and point across are fine nodes!
            jump_correction[dim]                = (2.0 - 2.0*int_nb.theta)*jump_correction_mid_point[dim] + (2.0*int_nb.theta - 1.0)*jump_correction_other_fine_node[dim];
          else // quad and mid-point are fine nodes!
            jump_correction[dim]                = 2.0*int_nb.theta*jump_correction_mid_point[dim] + (1.0 - 2.0*int_nb.theta)*jump_correction_other_fine_node[dim];
        }
        jump_flux_comp_across  += jump_correction[dir/2];

        int_nb.int_value =
            ((1.0-int_nb.theta)*int_nb.mu_this_side*solution_read_p[quad_idx]
             + int_nb.theta*int_nb.mu_other_side*solution_read_p[int_nb.quad_tmp_idx]
             + ((dir%2 == 0) ? ((int_nb.phi_q <= 0) ? +1.0 : -1.0) :
                               ((int_nb.phi_q <= 0) ? -1.0 : +1.0))*int_nb.theta*(1.0 - int_nb.theta)*dxyz_min[dir/2]*jump_flux_comp_across
            + ((mu_m_is_larger) ?  ((int_nb.phi_q <= 0.0)? -int_nb.mu_other_side*int_nb.theta : -int_nb.mu_this_side*(1.0 - int_nb.theta)) :
                                        ((int_nb.phi_q <= 0.0)? +int_nb.mu_this_side*(1.0 - int_nb.theta) : +int_nb.mu_other_side*int_nb.theta))*jump_sol_across
            )/((1.0 - int_nb.theta)*int_nb.mu_this_side + int_nb.theta*int_nb.mu_other_side);
      }
    }
  }
  else
  {
    // The interface values are set but initial guesses for the extended values are built as well
    double *new_cell_extension_p;
    ierr = VecGetArray(new_cell_extension, &new_cell_extension_p); CHKERRXX(ierr);
    splitting_criteria_t *data = (splitting_criteria_t*) p4est->user_pointer;
    double rel_top_idx;
    p4est_topidx_t nxyz_tree_quad[P4EST_DIM], nxyz_tree_tmp[P4EST_DIM];
    p4est_qcoord_t qxyz_quad[P4EST_DIM], qxyz_tmp[P4EST_DIM];
    p4est_locidx_t local_fine_indices_for_quad_interp[P4EST_CHILDREN], local_fine_indices_for_tmp_interp[P4EST_CHILDREN];
    double interp_weights_for_quad_interp[P4EST_CHILDREN], interp_weights_for_tmp_interp[P4EST_CHILDREN];
    bool is_quad_a_fine_node, is_tmp_a_fine_node;
    std::vector<p4est_quadrant_t> neighbor_cells;
    for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx){
      p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
      for (short dim = 0; dim < P4EST_DIM; ++dim)
      {
        rel_top_idx = (p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0] + dim] - xyz_min[dim])/tree_dimensions[dim];
        P4EST_ASSERT(fabs(rel_top_idx - floor(rel_top_idx)) < EPS);
        nxyz_tree_quad[dim]   = ((p4est_topidx_t) floor(rel_top_idx));
      }
      for (size_t q=0; q<tree->quadrants.elem_count; ++q)
      {
        const p4est_quadrant_t *quad  = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
        p4est_locidx_t quad_idx = q + tree->quadrants_offset;

        qxyz_quad[0] = quad->x + P4EST_QUADRANT_LEN(quad->level+1);
        qxyz_quad[1] = quad->y + P4EST_QUADRANT_LEN(quad->level+1);
  #ifdef P4_TO_P8
        qxyz_quad[2] = quad->z + P4EST_QUADRANT_LEN(quad->level+1);
  #endif
        is_quad_a_fine_node = multilinear_interpolation_weights(nxyz_tree_quad, qxyz_quad, local_fine_indices_for_quad_interp, interp_weights_for_quad_interp);
        double phi_q = phi_read_p[local_fine_indices_for_quad_interp[0]]*interp_weights_for_quad_interp[0];
        for (short ccc = 1; ccc < (is_quad_a_fine_node ? 0 : P4EST_CHILDREN); ++ccc)
          phi_q += phi_read_p[local_fine_indices_for_quad_interp[ccc]]*interp_weights_for_quad_interp[ccc];
        // build the educated guess
        new_cell_extension_p[quad_idx] = solution_read_p[quad_idx];
        double jump_u_at_quad = jump_u_read_p[local_fine_indices_for_quad_interp[0]]*interp_weights_for_quad_interp[0];
        for (short ccc = 1; ccc < (is_quad_a_fine_node ? 0 : P4EST_CHILDREN); ++ccc)
          jump_u_at_quad += jump_u_read_p[local_fine_indices_for_quad_interp[ccc]]*interp_weights_for_quad_interp[ccc];
        if(mu_m_is_larger && phi_q > 0.0)
          new_cell_extension_p[quad_idx] -= jump_u_at_quad;
        else if(!mu_m_is_larger && phi_q <= 0.0)
          new_cell_extension_p[quad_idx] += jump_u_at_quad;
        // set the interface values
        if(quad->level == data->max_lvl)
        {
          // finest quadrant, check if neighboring the interface and fill relevant data in std::vector structures
          for(int dir=0; dir<P4EST_FACES; ++dir)
          {
            neighbor_cells.resize(0);
            cell_ngbd->find_neighbor_cells_of_cell(neighbor_cells, quad_idx, tree_idx, dir);
            if(neighbor_cells.size()==1 && neighbor_cells[0].level == quad->level)
            {
              p4est_locidx_t quad_tmp_idx = neighbor_cells[0].p.piggy3.local_num;
              p4est_topidx_t tree_tmp_idx = neighbor_cells[0].p.piggy3.which_tree;

              for (short dim = 0; dim < P4EST_DIM; ++dim) {
                rel_top_idx = (p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_tmp_idx + 0] + dim] - xyz_min[dim])/tree_dimensions[dim];
                P4EST_ASSERT(fabs(rel_top_idx - floor(rel_top_idx)) < EPS);
                nxyz_tree_tmp[dim] = ((p4est_topidx_t) floor(rel_top_idx));
              }
              qxyz_tmp[0] = neighbor_cells[0].x + P4EST_QUADRANT_LEN(neighbor_cells[0].level+1);
              qxyz_tmp[1] = neighbor_cells[0].y + P4EST_QUADRANT_LEN(neighbor_cells[0].level+1);
    #ifdef P4_TO_P8
              qxyz_tmp[2] = neighbor_cells[0].z + P4EST_QUADRANT_LEN(neighbor_cells[0].level+1);
    #endif

              is_tmp_a_fine_node = multilinear_interpolation_weights(nxyz_tree_tmp, qxyz_tmp, local_fine_indices_for_tmp_interp, interp_weights_for_tmp_interp);
              double phi_tmp = phi_read_p[local_fine_indices_for_tmp_interp[0]]*interp_weights_for_tmp_interp[0];
              for (short ccc = 1; ccc < ((is_tmp_a_fine_node) ? 0: P4EST_CHILDREN); ++ccc)
                phi_tmp += phi_read_p[local_fine_indices_for_tmp_interp[ccc]]*interp_weights_for_tmp_interp[ccc];
              if(((phi_q > 0.0) && (phi_tmp <= 0.0)) || ((phi_q <= 0.0) && (phi_tmp > 0.0)))
              {
                P4EST_ASSERT(is_quad_a_fine_node && is_tmp_a_fine_node);
                interface_neighbor int_nb = get_interface_neighbor(quad_idx, dir, quad_tmp_idx, local_fine_indices_for_quad_interp[0], local_fine_indices_for_tmp_interp[0], phi_read_p, phi_dd_read_p);
                P4EST_ASSERT(interface_neighbor_is_found(quad_idx, dir, int_nb));

                double jump_sol_across, jump_flux_comp_across;
                if (int_nb.theta >=0.5)
                {
                  // mid_point on same side
                  P4EST_ASSERT(int_nb.theta <= 1.0);
                  // mid_point and point across are fine nodes!
                  jump_sol_across         = (2.0 - 2.0*int_nb.theta)*jump_u_read_p[int_nb.mid_point_fine_node_idx]                + (2.0*int_nb.theta - 1.0)*jump_u_read_p[int_nb.tmp_fine_node_idx];
                  jump_flux_comp_across   = (2.0 - 2.0*int_nb.theta)*jump_mu_grad_u_read_p[dir/2][int_nb.mid_point_fine_node_idx] + (2.0*int_nb.theta - 1.0)*jump_mu_grad_u_read_p[dir/2][int_nb.tmp_fine_node_idx];
                }
                else
                {
                  // mid_point on the other side
                  P4EST_ASSERT(int_nb.theta >=0.0);
                  // quad and mid-point are fine nodes!
                  jump_sol_across         = 2.0*int_nb.theta*jump_u_read_p[int_nb.mid_point_fine_node_idx]                + (1.0 - 2.0*int_nb.theta)*jump_u_read_p[int_nb.quad_fine_node_idx];
                  jump_flux_comp_across   = 2.0*int_nb.theta*jump_mu_grad_u_read_p[dir/2][int_nb.mid_point_fine_node_idx] + (1.0 - 2.0*int_nb.theta)*jump_mu_grad_u_read_p[dir/2][int_nb.quad_fine_node_idx];
                }

                map_of_interface_neighbors[quad_idx][dir].int_value =
                    ((1.0-int_nb.theta)*int_nb.mu_this_side*solution_read_p[quad_idx]
                     + int_nb.theta*int_nb.mu_other_side*solution_read_p[quad_tmp_idx]
                     + ((dir%2 == 0) ? ((phi_q <= 0) ? +1.0 : -1.0) :
                                       ((phi_q <= 0) ? -1.0 : +1.0))*int_nb.theta*(1.0 - int_nb.theta)*dxyz_min[dir/2]*jump_flux_comp_across
                    + ((mu_m_is_larger) ?  ((phi_q <= 0.0)? -int_nb.mu_other_side*int_nb.theta : -int_nb.mu_this_side*(1.0 - int_nb.theta)) :
                                                ((phi_q <= 0.0)? +int_nb.mu_this_side*(1.0 - int_nb.theta) : +int_nb.mu_other_side*int_nb.theta))*jump_sol_across
                    )/((1.0 - int_nb.theta)*int_nb.mu_this_side + int_nb.theta*int_nb.mu_other_side);
              }
            }
          }
        }
      }
    }
    ierr = VecGhostUpdateBegin(new_cell_extension, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#ifdef CASL_THROWS
    P4EST_ASSERT(is_map_consistent());
#endif
    ierr = VecGhostUpdateEnd(new_cell_extension, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecRestoreArray(new_cell_extension, &new_cell_extension_p); CHKERRXX(ierr);
  }
  ierr = VecRestoreArrayRead(jump_u, &jump_u_read_p); CHKERRXX(ierr);
  for (short dim = 0; dim < P4EST_DIM; ++dim)
  {
    ierr = VecRestoreArrayRead(jump_mu_grad_u[dim], &jump_mu_grad_u_read_p[dim]); CHKERRXX(ierr);
    if(second_derivatives_of_phi_are_set){
      ierr = VecRestoreArrayRead(phi_second_der[dim], &phi_dd_read_p[dim]); CHKERRXX(ierr);}
  }
  interface_values_are_set = true;
}

void my_p4est_xgfm_cells_t::cell_TVD_extension_of_interface_values(Vec new_cell_extension, const double& threshold, const uint& niter_max)
{
#ifdef CASL_THROWS
  // compare local size, global size and ghost layers
  PetscInt local_size, global_size, ghost_layer_size;

  ierr = VecGetLocalSize(new_cell_extension, &local_size); CHKERRXX(ierr);
  int my_error = ( ((PetscInt) p4est->local_num_quadrants) != local_size);

  ierr = VecGetSize(new_cell_extension, &global_size); CHKERRXX(ierr);
  my_error = my_error || (global_size != ((PetscInt) p4est->global_num_quadrants));

  Vec new_cell_extension_loc;
  ierr = VecGhostGetLocalForm(new_cell_extension, &new_cell_extension_loc); CHKERRXX(ierr);
  ierr = VecGetSize(new_cell_extension_loc, &ghost_layer_size); CHKERRXX(ierr);
  ghost_layer_size -= local_size;
  my_error = my_error || (ghost_layer_size != ((PetscInt) (ghost->ghosts.elem_count)));
  ierr = VecGhostRestoreLocalForm(new_cell_extension, &new_cell_extension_loc); CHKERRXX(ierr);

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &my_error, 1, MPI_INT, MPI_LOR, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  if(my_error)
    throw std::invalid_argument("my_p4est_xgfm_cells_t::extend_interface_values(const double *, Vec , const double* , bool, double, uint) : the second vector argument must be preallocated and have the same layout as if constructed with VecCreateGhostCells on the coarse grid...");

  my_error = !phi_has_been_set;
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &my_error, 1, MPI_INT, MPI_LOR, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  if(my_error)
    throw std::invalid_argument("my_p4est_xgfm_cells_t::extend_interface_values(const double *, Vec , const double* , bool, double, uint) : the levelset must be set beforehand...");
  my_error = threshold < EPS;
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &my_error, 1, MPI_INT, MPI_LOR, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  if(my_error)
    throw std::invalid_argument("my_p4est_xgfm_cells_t::extend_interface_values(const double *, Vec , const double* , bool, double, uint) : the threshold value must be strictly positive...");
  my_error = niter_max==0;
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &my_error, 1, MPI_INT, MPI_LOR, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  if(my_error)
    throw std::invalid_argument("my_p4est_xgfm_cells_t::extend_interface_values(const double *, Vec , const double* , bool, double, uint) : the max number of iterations must be strictly positive...");
#endif

  const double *phi_read_p, *normals_read_p[P4EST_DIM];
  ierr = VecGetArrayRead(phi, &phi_read_p); CHKERRXX(ierr);
  for (short dim = 0; dim < P4EST_DIM; ++dim){
    ierr = VecGetArrayRead(normals[dim], &normals_read_p[dim]); CHKERRXX(ierr);}
  double *new_cell_extension_p;
  ierr = VecGetArray(new_cell_extension, &new_cell_extension_p); CHKERRXX(ierr);

  double max_corr = 10.0*threshold;
  uint iter = 0;

  bool reverse_flag = true; // we reverse the z-ordering between two iterations to alleviate the slow convergence along the characteristic direction perpendicular to the main domain diagonal (z-order)
  while ((max_corr > threshold) && (iter < niter_max))
  {
    max_corr = 0.0; // reset the measure;
    /* Main loop over all local quadrant */
    for (p4est_topidx_t tree_idx = ((reverse_flag) ? p4est->last_local_tree : p4est->first_local_tree) ;
         ((reverse_flag) ?  tree_idx >= p4est->first_local_tree: tree_idx <= p4est->last_local_tree) ;
         ((reverse_flag) ?  --tree_idx: ++tree_idx))
    {
      p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
      p4est_topidx_t nxyz_tree_quad[P4EST_DIM];
      for (short dim = 0; dim < P4EST_DIM; ++dim)
      {
        double rel_top_idx = (p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0] + dim] - xyz_min[dim])/tree_dimensions[dim];
        P4EST_ASSERT(fabs(rel_top_idx - floor(rel_top_idx)) < EPS);
        nxyz_tree_quad[dim]   = ((p4est_topidx_t) floor(rel_top_idx));
      }
      for (size_t q = 0 ; q < tree->quadrants.elem_count ; q++)
      {
        double phi_q;
        double increment = 0.0;
        p4est_locidx_t quad_idx = ((reverse_flag) ? tree->quadrants.elem_count-q-1: q) + tree->quadrants_offset;
        if(extension_entries_are_set)
        {
          if(extension_entries[quad_idx].too_close)
          {
            P4EST_ASSERT(fabs(extension_entries[quad_idx].diag_entry) < EPS);
            new_cell_extension_p[quad_idx]  = map_of_interface_neighbors[quad_idx][extension_entries[quad_idx].forced_interface_value_dir].int_value;
          }
          else
          {
            increment += (extension_entries[quad_idx].diag_entry)*new_cell_extension_p[quad_idx];
            for (size_t i = 0; i < extension_entries[quad_idx].interface_entries.size(); ++i)
              increment += (extension_entries[quad_idx].interface_entries[i].coeff)*(map_of_interface_neighbors[quad_idx][extension_entries[quad_idx].interface_entries[i].dir].int_value);
            for (size_t i = 0; i < extension_entries[quad_idx].quad_entries.size(); ++i)
              increment += (extension_entries[quad_idx].quad_entries[i].coeff)*(new_cell_extension_p[extension_entries[quad_idx].quad_entries[i].loc_idx]);
            increment *= extension_entries[quad_idx].dtau;
          }
          phi_q = extension_entries[quad_idx].phi_q;
        }
        else
        {
          // we'll build the extension matrix entries!
          // the extension procedure, can be formally written as
          // increment = A*cell_values + B*interface_values
          // where A and B are appropriate (sparse) matrices
          // here below, we build the entries of A and B.
          // For the local quadrant of index quad_idx,
          // - extension_entries[quad_idx].quad_entries = vector of structures encompassing i) the local number of a relevant neighbor ii) the coefficient by which this neighbor-value must be multiplied to contribute to -sgn_n*grad_u
          // - extension_entries[quad_idx].interface_entries = vector of structures encompassing i) the relevant direction in which the interface neighbor can be found, ii) the coefficient by which this neighbor-value must be multiplied to contribute to -sgn_n*grad_u
          // - extension_entries[quad_idx].diag_entry = value of the coefficient multiplying the local value of the current quadrant when contributing to -sgn_n*grad_u
          // - extension_entries[quad_idx].dtau = is the fictititious time step for the considered cell, satisfying the cfl condition;
          // - extension_entries[quad_idx].too_close is set to true when the cell value is an interface point --> enforcing the relevant interface value there;
          extension_entries[quad_idx].diag_entry = 0.0;
          const p4est_quadrant_t *quad  = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, ((reverse_flag) ? tree->quadrants.elem_count-1-q: q));
          p4est_qcoord_t qxyz_quad[P4EST_DIM];
          qxyz_quad[0] = quad->x + P4EST_QUADRANT_LEN(quad->level+1);
          qxyz_quad[1] = quad->y + P4EST_QUADRANT_LEN(quad->level+1);
    #ifdef P4_TO_P8
          qxyz_quad[2] = quad->z + P4EST_QUADRANT_LEN(quad->level+1);
    #endif

          p4est_locidx_t local_fine_indices_for_quad_interp[P4EST_CHILDREN];
          double interp_weights_for_quad_interp[P4EST_CHILDREN];
          bool is_quad_a_fine_node = multilinear_interpolation_weights(nxyz_tree_quad, qxyz_quad, local_fine_indices_for_quad_interp, interp_weights_for_quad_interp);
          double sgn_n[P4EST_DIM], norm;
          norm = 0.0;
          phi_q = phi_read_p[local_fine_indices_for_quad_interp[0]]*interp_weights_for_quad_interp[0];
          for (short dim = 0; dim < P4EST_DIM; ++dim)
            sgn_n[dim] = normals_read_p[dim][local_fine_indices_for_quad_interp[0]]*interp_weights_for_quad_interp[0];
          for (short ccc = 1; ccc < (is_quad_a_fine_node ? 0 : P4EST_CHILDREN); ++ccc)
          {
            phi_q += phi_read_p[local_fine_indices_for_quad_interp[ccc]]*interp_weights_for_quad_interp[ccc];
            for (short dim = 0; dim < P4EST_DIM; ++dim)
              sgn_n[dim] += normals_read_p[dim][local_fine_indices_for_quad_interp[ccc]]*interp_weights_for_quad_interp[ccc];
          }

          for (short dim = 0; dim < P4EST_DIM; ++dim)
            norm       += SQR(sgn_n[dim]);
          norm = sqrt(norm);

          if (norm < EPS)
            for (short dim = 0; dim < P4EST_DIM; ++dim)
              sgn_n[dim] = 0.0;
          else
            for (short dim = 0; dim < P4EST_DIM; ++dim)
              sgn_n[dim] = ((phi_q <= 0.0)? -sgn_n[dim]/norm: sgn_n[dim]/norm);

          double dtau = DBL_MAX;
          for (short dim = 0; dim < P4EST_DIM; ++dim)
          {
            int dir = 2*dim + ((sgn_n[dim] > 0.0)? 0 : 1);
            if(is_quad_Wall(p4est, tree_idx, quad, dir))
              continue; // homogeneous Neumann boundary condition

            std::vector<p4est_quadrant_t> neighbor_cells;
            neighbor_cells.resize(0);
            cell_ngbd->find_neighbor_cells_of_cell(neighbor_cells, quad_idx, tree_idx, dir);

            if(neighbor_cells.size() == 1)
            {
              int8_t level_tmp = neighbor_cells[0].level;
              p4est_locidx_t quad_tmp_idx = neighbor_cells[0].p.piggy3.local_num;
              p4est_topidx_t tree_tmp_idx = neighbor_cells[0].p.piggy3.which_tree;

              p4est_topidx_t nxyz_tree_tmp[P4EST_DIM];
              for (short ddim = 0; ddim < P4EST_DIM; ++ddim)
              {
                double rel_top_idx = (p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_tmp_idx + 0] + ddim] - xyz_min[ddim])/tree_dimensions[ddim];
                P4EST_ASSERT(fabs(rel_top_idx - floor(rel_top_idx)) < EPS);
                nxyz_tree_tmp[ddim] = ((p4est_topidx_t) floor(rel_top_idx));
              }
              p4est_qcoord_t qxyz_tmp[P4EST_DIM];
              qxyz_tmp[0] = neighbor_cells[0].x + P4EST_QUADRANT_LEN(neighbor_cells[0].level+1);
              qxyz_tmp[1] = neighbor_cells[0].y + P4EST_QUADRANT_LEN(neighbor_cells[0].level+1);
    #ifdef P4_TO_P8
              qxyz_tmp[2] = neighbor_cells[0].z + P4EST_QUADRANT_LEN(neighbor_cells[0].level+1);
    #endif

              p4est_locidx_t local_fine_indices_for_tmp_interp[P4EST_CHILDREN];
              double interp_weights_for_tmp_interp[P4EST_CHILDREN];
              bool is_tmp_a_fine_node = multilinear_interpolation_weights(nxyz_tree_tmp, qxyz_tmp, local_fine_indices_for_tmp_interp, interp_weights_for_tmp_interp);
              double phi_tmp = phi_read_p[local_fine_indices_for_tmp_interp[0]]*interp_weights_for_tmp_interp[0];
              for (short ccc = 1; ccc < ((is_tmp_a_fine_node) ? 0: P4EST_CHILDREN); ++ccc)
                phi_tmp += phi_read_p[local_fine_indices_for_tmp_interp[ccc]]*interp_weights_for_tmp_interp[ccc];
              /* If interface across the two cells.
               * We assume that the interface is tesselated with uniform finest grid level */
              if(((phi_q > 0.0) && (phi_tmp <= 0.0)) || ((phi_q <= 0.0) && (phi_tmp > 0.0)))
              {
                double theta = map_of_interface_neighbors[quad_idx][dir].theta;
                double interface_value = map_of_interface_neighbors[quad_idx][dir].int_value;
                if(theta > EPS)
                {
                  increment      -= sgn_n[dir/2]*((dir%2 == 0)? +1.0 : -1.0)*(new_cell_extension_p[quad_idx] - interface_value)/(theta*dxyz_min[dim]);
                  dtau            = MIN(dtau, theta*dxyz_min[dim]/((double) P4EST_DIM));
                  extension_interface_value_entry int_entry; int_entry.dir = dir; int_entry.coeff = +sgn_n[dir/2]*((dir%2 == 0)? +1.0 : -1.0)/(theta*dxyz_min[dim]);
                  extension_entries[quad_idx].interface_entries.push_back(int_entry);
                  extension_entries[quad_idx].diag_entry -= sgn_n[dir/2]*((dir%2 == 0)? +1.0 : -1.0)/(theta*dxyz_min[dim]);
                }
                else
                {
                  new_cell_extension_p[quad_idx]  = interface_value;
                  dtau                            = 0.0;
                  extension_entries[quad_idx].interface_entries.resize(0);
                  extension_entries[quad_idx].quad_entries.resize(0);
                  extension_entries[quad_idx].too_close = true;
                  extension_entries[quad_idx].diag_entry = 0.0;
                  extension_entries[quad_idx].forced_interface_value_dir = dir;
                }
              }
              /* no interface - regular discretization */
              else
              {
                double s_tmp = pow((double)P4EST_QUADRANT_LEN(neighbor_cells[0].level)/(double)P4EST_ROOT_LEN, (double)P4EST_DIM-1);

                neighbor_cells.resize(0);
                cell_ngbd->find_neighbor_cells_of_cell(neighbor_cells, quad_tmp_idx, tree_tmp_idx, ((dir%2==0)? dir+1: dir-1));

                std::vector<double> s_ng(neighbor_cells.size());
                double d = 0.0;
                for(unsigned int i=0; i<neighbor_cells.size(); ++i)
                {
                  s_ng[i] = pow((double)P4EST_QUADRANT_LEN(neighbor_cells[i].level)/(double)P4EST_ROOT_LEN, (double)P4EST_DIM-1);
                  d += s_ng[i]/s_tmp * 0.5 * (double)(P4EST_QUADRANT_LEN(level_tmp)+P4EST_QUADRANT_LEN(neighbor_cells[i].level))/(double)P4EST_ROOT_LEN;
                }
                d *= tree_dimensions[dir/2];
                extension_matrix_entry tmp_mat_entry; tmp_mat_entry.loc_idx = quad_tmp_idx; tmp_mat_entry.coeff = 0.0;
                for(unsigned int i=0; i<neighbor_cells.size(); ++i)
                {
                  increment -= sgn_n[dir/2]*((dir%2 == 0)? +1.0 : -1.0)*(new_cell_extension_p[neighbor_cells[i].p.piggy3.local_num] - new_cell_extension_p[quad_tmp_idx])*s_ng[i]/s_tmp/d;
                  extension_matrix_entry mat_entry; mat_entry.loc_idx = neighbor_cells[i].p.piggy3.local_num; mat_entry.coeff = -sgn_n[dir/2]*((dir%2 == 0)? +1.0 : -1.0)*s_ng[i]/s_tmp/d;
                  extension_entries[quad_idx].quad_entries.push_back(mat_entry);
                  tmp_mat_entry.coeff += sgn_n[dir/2]*((dir%2 == 0)? +1.0 : -1.0)*s_ng[i]/s_tmp/d;
                }
                extension_entries[quad_idx].quad_entries.push_back(tmp_mat_entry);
                dtau = MIN(dtau, d/((double) P4EST_DIM));
              }
            }
            /* there is more than one neighbor, regular bulk case. This assumes uniform on interface ! */
            else
            {
              double s_tmp = pow((double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN, (double)P4EST_DIM-1);
              std::vector<double> s_ng(neighbor_cells.size());
              double d = 0.0;
              for(unsigned int i=0; i<neighbor_cells.size(); ++i)
              {
                s_ng[i] = pow((double)P4EST_QUADRANT_LEN(neighbor_cells[i].level)/(double)P4EST_ROOT_LEN, (double)P4EST_DIM-1);
                d += s_ng[i]/s_tmp * 0.5 * (double)(P4EST_QUADRANT_LEN(quad->level)+P4EST_QUADRANT_LEN(neighbor_cells[i].level))/(double)P4EST_ROOT_LEN;
              }
              d *= tree_dimensions[dir/2];
              for(unsigned int i=0; i<neighbor_cells.size(); ++i)
              {
                increment -= sgn_n[dir/2]*((dir%2 == 0)? -1.0 : +1.0)*(new_cell_extension_p[neighbor_cells[i].p.piggy3.local_num] - new_cell_extension_p[quad_idx])*s_ng[i]/s_tmp/d;
                extension_matrix_entry mat_entry; mat_entry.loc_idx = neighbor_cells[i].p.piggy3.local_num; mat_entry.coeff = -sgn_n[dir/2]*((dir%2 == 0)? -1.0 : +1.0)*s_ng[i]/s_tmp/d;
                extension_entries[quad_idx].quad_entries.push_back(mat_entry);
                extension_entries[quad_idx].diag_entry += sgn_n[dir/2]*((dir%2 == 0)? -1.0 : +1.0)*s_ng[i]/s_tmp/d;
              }
              dtau = MIN(dtau, d/((double) P4EST_DIM));
            }
          }
          extension_entries[quad_idx].dtau = dtau;
          extension_entries[quad_idx].phi_q = phi_q;
          increment *= dtau;
        }
        new_cell_extension_p[quad_idx] += increment;

        if(fabs(phi_q) < 3.0*diag_min)
          max_corr = MAX(max_corr, fabs(increment));
      }
    }

    extension_entries_are_set = true;
    ierr = VecGhostUpdateBegin(new_cell_extension, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &max_corr, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
    ierr = VecGhostUpdateEnd(new_cell_extension, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    reverse_flag = !reverse_flag;
    iter++;
  }

  ierr = VecRestoreArrayRead(phi, &phi_read_p); CHKERRXX(ierr);
  for (short dim = 0; dim < P4EST_DIM; ++dim)
  {
    ierr = VecRestoreArrayRead(normals[dim], &normals_read_p[dim]); CHKERRXX(ierr);
  }
  ierr = VecRestoreArray(new_cell_extension, &new_cell_extension_p); CHKERRXX(ierr);
}

void my_p4est_xgfm_cells_t::extend_interface_values(const double *solution_read_p, Vec new_cell_extension, const double* extension_on_fine_nodes_read_p, double threshold, uint niter_max)
{
  ierr = PetscLogEventBegin(log_my_p4est_xgfm_cells_extend_field, 0, 0, 0, 0); CHKERRXX(ierr);
#ifdef CASL_THROWS
  // compare local size, global size and ghost layers
  PetscInt local_size, global_size, ghost_layer_size;

  ierr = VecGetLocalSize(new_cell_extension, &local_size); CHKERRXX(ierr);
  int my_error = ( ((PetscInt) p4est->local_num_quadrants) != local_size);

  ierr = VecGetSize(new_cell_extension, &global_size); CHKERRXX(ierr);
  my_error = my_error || (global_size != ((PetscInt) p4est->global_num_quadrants));

  Vec new_cell_extension_loc;
  ierr = VecGhostGetLocalForm(new_cell_extension, &new_cell_extension_loc); CHKERRXX(ierr);
  ierr = VecGetSize(new_cell_extension_loc, &ghost_layer_size); CHKERRXX(ierr);
  ghost_layer_size -= local_size;
  my_error = my_error || (ghost_layer_size != ((PetscInt) (ghost->ghosts.elem_count)));
  ierr = VecGhostRestoreLocalForm(new_cell_extension, &new_cell_extension_loc); CHKERRXX(ierr);

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &my_error, 1, MPI_INT, MPI_LOR, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  if(my_error)
    throw std::invalid_argument("my_p4est_xgfm_cells_t::extend_interface_values(const double *, Vec , const double* , bool, double, uint) : the second vector argument must be preallocated and have the same layout as if constructed with VecCreateGhostCells on the coarse grid...");

  my_error = !phi_has_been_set;
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &my_error, 1, MPI_INT, MPI_LOR, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  if(my_error)
    throw std::invalid_argument("my_p4est_xgfm_cells_t::extend_interface_values(const double *, Vec , const double* , bool, double, uint) : the levelset must be set beforehand...");
  my_error = !jumps_have_been_set;
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &my_error, 1, MPI_INT, MPI_LOR, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  if(my_error)
    throw std::invalid_argument("my_p4est_xgfm_cells_t::extend_interface_values(const double *, Vec , const double* , bool, double, uint) : the jump values must be set beforehand...");
  my_error = threshold < EPS;
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &my_error, 1, MPI_INT, MPI_LOR, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  if(my_error)
    throw std::invalid_argument("my_p4est_xgfm_cells_t::extend_interface_values(const double *, Vec , const double* , bool, double, uint) : the threshold value must be strictly positive...");
  my_error = niter_max==0;
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &my_error, 1, MPI_INT, MPI_LOR, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  if(my_error)
    throw std::invalid_argument("my_p4est_xgfm_cells_t::extend_interface_values(const double *, Vec , const double* , bool, double, uint) : the max number of iterations must be strictly positive...");
#endif

  P4EST_ASSERT(new_cell_extension != NULL);

  // Update (or set) the map of interface information before iterative procedure
  // and build the educated guess for cell_extension_p if extension_not_defined_yet
  update_interface_values(new_cell_extension, solution_read_p, extension_on_fine_nodes_read_p);
  // TVD cell-centered extension
  cell_TVD_extension_of_interface_values(new_cell_extension, threshold, niter_max);

  ierr = PetscLogEventEnd(log_my_p4est_xgfm_cells_extend_field, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_xgfm_cells_t::interpolate_coarse_cell_field_to_fine_nodes(const double *cell_field_read_p, Vec fine_node_field)
{
  ierr = PetscLogEventBegin(log_my_p4est_xgfm_cells_interpolate_coarse_cell_field_to_fine_nodes, 0, 0, 0, 0); CHKERRXX(ierr);
#ifdef CASL_THROWS
  // compare local size, global size and ghost layers
  PetscInt local_size, global_size, ghost_layer_size;

  ierr = VecGetLocalSize(fine_node_field, &local_size); CHKERRXX(ierr);
  int my_error = ( ((PetscInt) fine_nodes->num_owned_indeps) != local_size);

  p4est_gloidx_t total_nb_fine_nodes = 0;
  for (int r = 0; r < p4est->mpisize; ++r)
    total_nb_fine_nodes += fine_nodes->global_owned_indeps[r];
  ierr = VecGetSize(fine_node_field, &global_size); CHKERRXX(ierr);
  my_error = my_error || (global_size != total_nb_fine_nodes);
  Vec fine_node_field_loc;
  ierr = VecGhostGetLocalForm(fine_node_field, &fine_node_field_loc); CHKERRXX(ierr);
  ierr = VecGetSize(fine_node_field_loc, &ghost_layer_size); CHKERRXX(ierr);
  ghost_layer_size -= local_size;
  my_error = my_error || (ghost_layer_size != ((PetscInt) (fine_nodes->indep_nodes.elem_count - fine_nodes->num_owned_indeps)));
  ierr = VecGhostRestoreLocalForm(fine_node_field, &fine_node_field_loc); CHKERRXX(ierr);

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &my_error, 1, MPI_INT, MPI_LOR, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  if(my_error)
    throw std::invalid_argument("my_p4est_xgfm_cells_t::interpolate_coarse_cell_field_to_fine_nodes(const double*, Vec) : the second vector argument must be preallocated and have the same layout as if constructed with VecCreateGhostNodes on the fine grid...");
#endif

  double *fine_node_field_p;
  ierr = VecGetArray(fine_node_field, &fine_node_field_p); CHKERRXX(ierr);

  if(local_interpolator_is_set)
  {
    for (size_t nnn = 0; nnn < fine_node_ngbd->layer_nodes.size(); ++nnn) {
      p4est_locidx_t fine_node_idx = fine_node_ngbd->layer_nodes[nnn];
      double value = 0.0;
      for (size_t k = 0; k < local_interpolator[fine_node_idx].size(); ++k)
        value += cell_field_read_p[local_interpolator[fine_node_idx][k].quad_idx]*local_interpolator[fine_node_idx][k].weight;
      fine_node_field_p[fine_node_idx] = value;
    }
    // start updating the layer node values
    ierr = VecGhostUpdateBegin(fine_node_field, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    for (size_t nnn = 0; nnn < fine_node_ngbd->local_nodes.size(); ++nnn) {
      p4est_locidx_t fine_node_idx = fine_node_ngbd->local_nodes[nnn];
      double value = 0.0;
      for (size_t k = 0; k < local_interpolator[fine_node_idx].size(); ++k)
        value += cell_field_read_p[local_interpolator[fine_node_idx][k].quad_idx]*local_interpolator[fine_node_idx][k].weight;
      fine_node_field_p[fine_node_idx] = value;
    }
    ierr = VecGhostUpdateEnd(fine_node_field, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }
  else
  {

    splitting_criteria_t *data_fine   = (splitting_criteria_t*) fine_p4est->user_pointer;
    std::vector<bool> fine_node_visited(fine_nodes->num_owned_indeps, false);
    double xyz_coarse_quad[P4EST_DIM], dxyz_coarse_quad[P4EST_DIM], xyz_fine_quad[P4EST_DIM], xyz_fine_node[P4EST_DIM];
    std::vector<p4est_quadrant_t> ngbd_of_coarse_cells; ngbd_of_coarse_cells.resize(0);
    std::vector<p4est_quadrant_t> remote_matches; remote_matches.resize(0);
  #ifdef DEBUG
    p4est_locidx_t num_visited = 0;
  #endif

    for (size_t nnn = 0; nnn < fine_node_ngbd->layer_nodes.size(); ++nnn) {
      p4est_locidx_t fine_node_idx = fine_node_ngbd->layer_nodes[nnn];
      const p4est_indep_t* ni = (const p4est_indep_t*) sc_array_index(&fine_nodes->indep_nodes, fine_node_idx);
      fine_node_visited[fine_node_idx] = true;
#ifdef DEBUG
      num_visited++;
#endif
      node_xyz_fr_n(fine_node_idx, fine_p4est, fine_nodes, xyz_fine_node);
      p4est_quadrant_t best_fine_match;
      int rank = fine_node_ngbd->hierarchy->find_smallest_quadrant_containing_point(xyz_fine_node, best_fine_match, remote_matches);
      P4EST_ASSERT(rank != -1);
      p4est_quadrant_t best_coarse_match; remote_matches.resize(0);
      rank = cell_ngbd->hierarchy->find_smallest_quadrant_containing_point(xyz_fine_node, best_coarse_match, remote_matches);
      p4est_tree_t* tree = (p4est_tree_t*) sc_array_index(p4est->trees, best_coarse_match.p.piggy3.which_tree);
      P4EST_ASSERT(rank != -1);
      p4est_locidx_t best_coarse_match_local_idx = (rank == p4est->mpirank) ? best_coarse_match.p.piggy3.local_num + tree->quadrants_offset : best_coarse_match.p.piggy3.local_num + p4est->local_num_quadrants;
      interpolate_cell_field_at_fine_node(fine_node_idx, ni, cell_field_read_p, fine_node_field_p, (best_fine_match.level == data_fine->max_lvl),
                                          &best_coarse_match, best_coarse_match_local_idx, best_coarse_match.p.piggy3.which_tree, ngbd_of_coarse_cells);
    }

    // start updating the layer node values
    ierr = VecGhostUpdateBegin(fine_node_field, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    // evaluate the values for the local nodes
    // we pass through all quadrants of the fine grid and update the corresponding coarse quadrant and its neighborhood
    // as we progress to avoid calculating the same neighborhood several times
    for (p4est_topidx_t tree_idx = fine_p4est->first_local_tree; tree_idx <= fine_p4est->last_local_tree; ++tree_idx) {
      p4est_tree_t *fine_tree             = (p4est_tree_t*) sc_array_index(fine_p4est->trees, tree_idx);
      p4est_tree_t *tree                  = (p4est_tree_t*) sc_array_index(p4est->trees, tree_idx);
      const p4est_quadrant_t *coarse_quad = (const p4est_quadrant_t*) sc_array_index(&tree->quadrants, 0);
      p4est_locidx_t coarse_quad_idx      = tree->quadrants_offset;
      quad_xyz_fr_q(coarse_quad_idx, tree_idx, p4est, ghost, xyz_coarse_quad);
      for (short dim = 0; dim < P4EST_DIM; ++dim)
        dxyz_coarse_quad[dim] = tree_dimensions[dim]*((double) P4EST_QUADRANT_LEN(coarse_quad->level))/((double) P4EST_ROOT_LEN);
      ngbd_of_coarse_cells.resize(0);
      for(int ii=-1; ii<2; ++ii)
        for(int jj=-1; jj<2; ++jj)
#ifdef P4_TO_P8
          for(int kk=-1; kk<2; ++kk)
            cell_ngbd->find_neighbor_cells_of_cell(ngbd_of_coarse_cells, coarse_quad_idx, tree_idx, ii, jj, kk);
#else
          cell_ngbd->find_neighbor_cells_of_cell(ngbd_of_coarse_cells, coarse_quad_idx, tree_idx, ii, jj);
#endif

      for (size_t q = 0; q < fine_tree->quadrants.elem_count; ++q) {
        const p4est_quadrant_t *fine_quad = (const p4est_quadrant_t*) sc_array_index(&fine_tree->quadrants, q);
        p4est_locidx_t fine_quad_idx      = q + fine_tree->quadrants_offset;
        quad_xyz_fr_q(fine_quad_idx, tree_idx, fine_p4est, fine_ghost, xyz_fine_quad);
        if(!is_quad_in_quad(xyz_coarse_quad, dxyz_coarse_quad, coarse_quad->level, xyz_fine_quad, fine_quad->level))
        {
          coarse_quad = (const p4est_quadrant_t*) sc_array_index(&tree->quadrants, ++coarse_quad_idx - tree->quadrants_offset);
          quad_xyz_fr_q(coarse_quad_idx, tree_idx, p4est, ghost, xyz_coarse_quad);
          for (short dim = 0; dim < P4EST_DIM; ++dim)
            dxyz_coarse_quad[dim] = tree_dimensions[dim]*((double) P4EST_QUADRANT_LEN(coarse_quad->level))/((double) P4EST_ROOT_LEN);
          ngbd_of_coarse_cells.resize(0);
          for(int ii=-1; ii<2; ++ii)
            for(int jj=-1; jj<2; ++jj)
#ifdef P4_TO_P8
              for(int kk=-1; kk<2; ++kk)
                cell_ngbd->find_neighbor_cells_of_cell(ngbd_of_coarse_cells, coarse_quad_idx, tree_idx, ii, jj, kk);
#else
              cell_ngbd->find_neighbor_cells_of_cell(ngbd_of_coarse_cells, coarse_quad_idx, tree_idx, ii, jj);
#endif
          P4EST_ASSERT(is_quad_in_quad(xyz_coarse_quad, dxyz_coarse_quad, coarse_quad->level, xyz_fine_quad, fine_quad->level));
        }
        for (short ccc = 0; ccc < P4EST_CHILDREN; ++ccc) {
          p4est_locidx_t fine_node_idx = fine_nodes->local_nodes[P4EST_CHILDREN*fine_quad_idx + ccc];
          p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&fine_nodes->indep_nodes, fine_node_idx);
          if(fine_node_idx >= fine_nodes->num_owned_indeps || fine_node_visited[fine_node_idx] || ni->pad8 != 0)
            continue;
          fine_node_visited[fine_node_idx] = true;
#ifdef DEBUG
          num_visited++;
#endif
          interpolate_cell_field_at_fine_node(fine_node_idx, ni, cell_field_read_p, fine_node_field_p, (fine_quad->level == data_fine->max_lvl),
                                              coarse_quad, coarse_quad_idx, tree_idx, ngbd_of_coarse_cells);
        }
      }
    }
    ierr = VecGhostUpdateEnd(fine_node_field, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    P4EST_ASSERT(num_visited == fine_nodes->num_owned_indeps);
    local_interpolator_is_set = true;
  }

  ierr = VecRestoreArray(fine_node_field, &fine_node_field_p); CHKERRXX(ierr);
  ierr = PetscLogEventEnd(log_my_p4est_xgfm_cells_interpolate_coarse_cell_field_to_fine_nodes, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_xgfm_cells_t::interpolate_cell_field_at_fine_node(const p4est_locidx_t &fine_node_idx, const p4est_indep_t* ni,
                                                                const double *cell_field_read_p, double *fine_node_field_p,
                                                                const bool& super_fine_node, const p4est_quadrant_t* coarse_quad, const p4est_locidx_t& coarse_quad_idx,
                                                                const p4est_topidx_t& tree_idx_for_coarse_quad, const std::vector<p4est_quadrant_t>& ngbd_of_coarse_cells)
{

  P4EST_ASSERT(!local_interpolator_is_set);
  double xyz_fine_node[P4EST_DIM];
  node_xyz_fr_n(fine_node_idx, fine_p4est, fine_nodes, xyz_fine_node);
  std::vector<p4est_quadrant_t> lsqr_neighbors;

  if(super_fine_node)
  {
    // On the coarse grid, the fine node is either
    // - a cell-center;
    // - a mid-edge point; (in 3D)
    // - a face-center;
    // - a vertex;
    // in the uniform level region of the coarse grid; --> we don't want lsqr interpolation for those
    // - or not entirely surrounded by uniform grids (T-junction) --> lsqr is fine there...
    p4est_qcoord_t qxyz_quad[P4EST_DIM], qxyz_fine_node[P4EST_DIM];
    p4est_topidx_t nxyz_tree_quad[P4EST_DIM], nxyz_tree_node[P4EST_DIM];
    double rel_top_idx;
    for (short dim = 0; dim < P4EST_DIM; ++dim) {
      rel_top_idx = (fine_p4est->connectivity->vertices[3*fine_p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx_for_coarse_quad + 0] + dim] - xyz_min[dim])/tree_dimensions[dim];
      P4EST_ASSERT(fabs(rel_top_idx - floor(rel_top_idx)) < EPS);
      nxyz_tree_quad[dim] = ((p4est_topidx_t) floor(rel_top_idx));

      rel_top_idx = (fine_p4est->connectivity->vertices[3*fine_p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*ni->p.piggy3.which_tree + 0] + dim] - xyz_min[dim])/tree_dimensions[dim];
      P4EST_ASSERT(fabs(rel_top_idx - floor(rel_top_idx)) < EPS);
      nxyz_tree_node[dim] = ((p4est_topidx_t) floor(rel_top_idx));
      P4EST_ASSERT((nxyz_tree_quad[dim] == nxyz_tree_node[dim]) || (nxyz_tree_quad[dim] == (nxyz_tree_node[dim] - 1)) || (nxyz_tree_quad[dim] == (nxyz_tree_node[dim] + 1)));
    }
    qxyz_quad[0]      = coarse_quad->x + P4EST_QUADRANT_LEN(coarse_quad->level+1);
    qxyz_fine_node[0] = ((ni->x == P4EST_ROOT_LEN-1)? P4EST_ROOT_LEN : ni->x);
    qxyz_quad[1]      = coarse_quad->y + P4EST_QUADRANT_LEN(coarse_quad->level+1);
    qxyz_fine_node[1] = ((ni->y == P4EST_ROOT_LEN-1)? P4EST_ROOT_LEN : ni->y);
#ifdef P4_TO_P8
    qxyz_quad[2]      = coarse_quad->z + P4EST_QUADRANT_LEN(coarse_quad->level+1);
    qxyz_fine_node[2] = ((ni->z == P4EST_ROOT_LEN-1)? P4EST_ROOT_LEN : ni->z);
#endif
    for (short dim = 0; dim < P4EST_DIM; ++dim){
      if((nxyz_tree_quad[dim] != nxyz_tree_node[dim]) || (periodicity[dim] && (abs(nxyz_tree_node[dim]-nxyz_tree_quad[dim]) == (brick->nxyztrees[dim]-1))))
      {
        P4EST_ASSERT(((nxyz_tree_node[dim] == nxyz_tree_quad[dim] - 1) && (qxyz_fine_node[dim] == P4EST_ROOT_LEN)) ||
                     ((nxyz_tree_node[dim] == nxyz_tree_quad[dim] + 1) && (qxyz_fine_node[dim] == 0)) ||
                     (periodicity[dim] && (abs(nxyz_tree_node[dim]-nxyz_tree_quad[dim]) == (brick->nxyztrees[dim]-1))));
        if(periodicity[dim] && (abs(nxyz_tree_node[dim]-nxyz_tree_quad[dim]) == (brick->nxyztrees[dim]-1)) &&
           (((qxyz_fine_node[dim] == 0) && (qxyz_quad[dim] == P4EST_ROOT_LEN - P4EST_QUADRANT_LEN(coarse_quad->level+1))) ||
            ((qxyz_fine_node[dim] == P4EST_ROOT_LEN) && (qxyz_quad[dim] == P4EST_QUADRANT_LEN(coarse_quad->level+1))) ))
        {
          if(qxyz_fine_node[dim] == 0)
          {
            P4EST_ASSERT((nxyz_tree_node[dim] == 0) && (nxyz_tree_quad[dim] == brick->nxyztrees[dim] - 1) && (qxyz_quad[dim] = P4EST_ROOT_LEN - P4EST_QUADRANT_LEN(coarse_quad->level+1)));
            nxyz_tree_node[dim] = brick->nxyztrees[dim]-1;
            qxyz_fine_node[dim] = P4EST_ROOT_LEN;
          }
          else
          {
            P4EST_ASSERT((nxyz_tree_node[dim] == brick->nxyztrees[dim] - 1) && (nxyz_tree_quad[dim] == 0) && (qxyz_quad[dim] = P4EST_QUADRANT_LEN(coarse_quad->level+1)));
            nxyz_tree_node[dim] = 0;
            qxyz_fine_node[dim] = 0;
          }
        }
        else
        {
          qxyz_fine_node[dim] += (nxyz_tree_node[dim] - nxyz_tree_quad[dim])*P4EST_ROOT_LEN;
          nxyz_tree_node[dim] += (nxyz_tree_quad[dim] - nxyz_tree_node[dim]);
        }
      }
    }
#ifdef DEBUG
    for (short dim = 0; dim < P4EST_DIM; ++dim)
      P4EST_ASSERT((nxyz_tree_node[dim] == nxyz_tree_quad[dim]) && (qxyz_fine_node[dim] >=0) && (qxyz_fine_node[dim] <= P4EST_ROOT_LEN));
#endif

    int logical_dir_in_quad[P4EST_DIM];
    for (short dim = 0; dim < P4EST_DIM; ++dim) {
      logical_dir_in_quad[dim] = (qxyz_fine_node[dim] - qxyz_quad[dim])/P4EST_QUADRANT_LEN(coarse_quad->level+1);
      P4EST_ASSERT((logical_dir_in_quad[dim] == -1) || (logical_dir_in_quad[dim] == 0) || (logical_dir_in_quad[dim] == 1));
    }

    bool lsqr_is_needed = false;
    std::vector<p4est_quadrant_t> direct_neighbor;
    lsqr_neighbors.resize(0);
    double count = 1.0;

    local_interpolator[fine_node_idx].resize(0);
    interpolation_factor interp_factor; interp_factor.quad_idx = coarse_quad_idx;
    fine_node_field_p[fine_node_idx] = cell_field_read_p[coarse_quad_idx]; local_interpolator[fine_node_idx].push_back(interp_factor);
    for(int ii=-1; ii<2; ++ii){
      for(int jj=-1; jj<2; ++jj){
#ifdef P4_TO_P8
        for(int kk=-1; kk<2; ++kk){
          direct_neighbor.resize(0);
          cell_ngbd->find_neighbor_cells_of_cell(direct_neighbor, coarse_quad_idx, tree_idx_for_coarse_quad, ii, jj, kk);
          if(direct_neighbor.size() == 0)
          {
            // If the coarse quad is a wall-quad, we won't fine the expected neighbor.
            // In that case, we disregard the expected neighbor: equivalent to considering homoegenous Neumann boundary condition for the extension on the Wall
            P4EST_ASSERT((ii != 0) || (jj != 0) || (kk != 0));
            P4EST_ASSERT(((ii == -1)? is_quad_xmWall(p4est, tree_idx_for_coarse_quad, coarse_quad) : ((ii == 1)? is_quad_xpWall(p4est, tree_idx_for_coarse_quad, coarse_quad) :false)) ||
                         ((jj == -1)? is_quad_ymWall(p4est, tree_idx_for_coarse_quad, coarse_quad) : ((jj == 1)? is_quad_ypWall(p4est, tree_idx_for_coarse_quad, coarse_quad) :false)) ||
                         ((kk == -1)? is_quad_zmWall(p4est, tree_idx_for_coarse_quad, coarse_quad) : ((kk == 1)? is_quad_zpWall(p4est, tree_idx_for_coarse_quad, coarse_quad) :false)));
            continue;
          }
          P4EST_ASSERT(direct_neighbor.size() == 1);
          bool add_it = true; size_t nn = 0;
          while (add_it && nn < lsqr_neighbors.size())
            add_it = (lsqr_neighbors[nn++].p.piggy3.local_num!=direct_neighbor[0].p.piggy3.local_num);
          if(add_it) lsqr_neighbors.push_back(direct_neighbor[0]);
          if(((ii == logical_dir_in_quad[0] || ii == 0)
              && (jj == logical_dir_in_quad[1] || jj == 0)
              && (kk == logical_dir_in_quad[2] || kk == 0)) && (ii != 0 || jj !=0 || kk !=0))
          {
            lsqr_is_needed = lsqr_is_needed || (direct_neighbor[0].level < coarse_quad->level);
            fine_node_field_p[fine_node_idx] += cell_field_read_p[direct_neighbor[0].p.piggy3.local_num]; interp_factor.quad_idx = direct_neighbor[0].p.piggy3.local_num; local_interpolator[fine_node_idx].push_back(interp_factor);
            count += 1.0;
          }
        }
#else
        direct_neighbor.resize(0);
        cell_ngbd->find_neighbor_cells_of_cell(direct_neighbor, coarse_quad_idx, tree_idx_for_coarse_quad, ii, jj);
        if(direct_neighbor.size() == 0)
        {
          // If the coarse quad is a wall-quad, we won't fine the expected neighbor.
          // In that case, we disregard the expected neighbor: equivalent to considering homoegenous Neumann boundary condition for the extension on the Wall
          P4EST_ASSERT((ii != 0) || (jj != 0));
          P4EST_ASSERT(((ii == -1)? is_quad_xmWall(p4est, tree_idx_for_coarse_quad, coarse_quad) : ((ii == 1)? is_quad_xpWall(p4est, tree_idx_for_coarse_quad, coarse_quad) :false)) ||
                       ((jj == -1)? is_quad_ymWall(p4est, tree_idx_for_coarse_quad, coarse_quad) : ((jj == 1)? is_quad_ypWall(p4est, tree_idx_for_coarse_quad, coarse_quad) :false)));
          continue;
        }
        P4EST_ASSERT(direct_neighbor.size() == 1);
        bool add_it = true; size_t nn = 0;
        while (add_it && nn < lsqr_neighbors.size())
          add_it = (lsqr_neighbors[nn++].p.piggy3.local_num!=direct_neighbor[0].p.piggy3.local_num);
        if(add_it) lsqr_neighbors.push_back(direct_neighbor[0]);
        if(((ii == logical_dir_in_quad[0] || ii == 0)
            && (jj == logical_dir_in_quad[1] || jj == 0)) && (ii != 0 || jj !=0))
        {
          lsqr_is_needed = lsqr_is_needed || (direct_neighbor[0].level < coarse_quad->level);
          fine_node_field_p[fine_node_idx] += cell_field_read_p[direct_neighbor[0].p.piggy3.local_num]; interp_factor.quad_idx = direct_neighbor[0].p.piggy3.local_num; local_interpolator[fine_node_idx].push_back(interp_factor);
          count += 1.0;
        }
#endif
      }
    }
    fine_node_field_p[fine_node_idx] /= count;
    if(lsqr_is_needed)
      fine_node_field_p[fine_node_idx] = get_lsqr_interpolation_at(xyz_fine_node, lsqr_neighbors, cell_field_read_p, local_interpolator[fine_node_idx]);
    else
      for (size_t i = 0; i < local_interpolator[fine_node_idx].size(); ++i)
        local_interpolator[fine_node_idx][i].weight = 1.0/count;
  }
  else
  {
    if(ngbd_of_coarse_cells.size() > 0)
      fine_node_field_p[fine_node_idx] = get_lsqr_interpolation_at(xyz_fine_node, ngbd_of_coarse_cells, cell_field_read_p, local_interpolator[fine_node_idx]);
    else
    {
      lsqr_neighbors.resize(0);
      for(int ii=-1; ii<2; ++ii)
        for(int jj=-1; jj<2; ++jj)
#ifdef P4_TO_P8
          for(int kk=-1; kk<2; ++kk)
            cell_ngbd->find_neighbor_cells_of_cell(lsqr_neighbors, coarse_quad_idx, tree_idx_for_coarse_quad, ii, jj, kk);
#else
          cell_ngbd->find_neighbor_cells_of_cell(lsqr_neighbors, coarse_quad_idx, tree_idx_for_coarse_quad, ii, jj);
#endif
      fine_node_field_p[fine_node_idx] = get_lsqr_interpolation_at(xyz_fine_node, lsqr_neighbors, cell_field_read_p, local_interpolator[fine_node_idx]);
    }
  }

#ifdef DEBUG
  double value_check = 0.0;
  for (size_t i = 0; i < local_interpolator[fine_node_idx].size(); ++i)
    value_check += cell_field_read_p[local_interpolator[fine_node_idx][i].quad_idx]*local_interpolator[fine_node_idx][i].weight;
  P4EST_ASSERT(fabs(value_check - fine_node_field_p[fine_node_idx]) < MAX(EPS, 0.000001*MAX(fabs(value_check), fabs(fine_node_field_p[fine_node_idx]))));
#endif
}

void my_p4est_xgfm_cells_t::get_corrected_rhs(Vec corrected_rhs, const double *fine_extension_interface_values_read_p)
{
  ierr = PetscLogEventBegin(log_my_p4est_xgfm_cells_get_corrected_rhs, 0, 0, 0, 0); CHKERRXX(ierr);
#ifdef CASL_THROWS
  // compare local size, global size and ghost layers
  PetscInt local_size, global_size;
  ierr = VecGetLocalSize(corrected_rhs, &local_size); CHKERRXX(ierr);
  int my_error = ( ((PetscInt) p4est->local_num_quadrants) != local_size);

  ierr = VecGetSize(corrected_rhs, &global_size); CHKERRXX(ierr);
  my_error = my_error || (global_size != ((PetscInt) p4est->global_num_quadrants));

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &my_error, 1, MPI_INT, MPI_LOR, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  if(my_error)
    throw std::invalid_argument("my_p4est_xgfm_cells_t::get_corrected_rhs(Vec, const double*): the first vector argument must be preallocated and have the same layout as if constructed with VecCreateCellsNoGhost(p4est, &Vec) on the coarse p4est...");

  my_error = !interface_values_are_set;
  if(my_error)
    throw std::invalid_argument("my_p4est_xgfm_cells_t::get_corrected_rhs(Vec, const double*): the corrected_rhs cannot be calculated before the first pass in the extension (the map of neighbor interface values must be set first)...");
#endif

  // reset corrected_rhs!
  ierr = VecCopy(rhs, corrected_rhs); CHKERRXX(ierr);

  // correct it!
  double *corrected_rhs_p;
  ierr = VecGetArray(corrected_rhs, &corrected_rhs_p); CHKERRXX(ierr);
  const double *phi_read_p, *normals_read_p[P4EST_DIM];
  ierr = VecGetArrayRead(phi, &phi_read_p); CHKERRXX(ierr);
  for (short dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecGetArrayRead(normals[dim], &normals_read_p[dim]); CHKERRXX(ierr);}
#ifdef DEBUG
  splitting_criteria_t* data = (splitting_criteria_t*) p4est->user_pointer;
#endif

  for (std::map<p4est_locidx_t, std::map<int, interface_neighbor> >::const_iterator it = map_of_interface_neighbors.begin();
       it != map_of_interface_neighbors.end(); ++it)
  {
    p4est_locidx_t quad_idx  = it->first;
    // find tree_idx
    p4est_topidx_t tree_idx  = p4est->first_local_tree;
    while(tree_idx < p4est->last_local_tree && quad_idx >= ((p4est_tree_t*) sc_array_index(p4est->trees, tree_idx+1))->quadrants_offset){tree_idx++;}
    // get the quadrant
    p4est_tree_t* tree  = (p4est_tree_t*) sc_array_index(p4est->trees, tree_idx);
    P4EST_ASSERT(quad_idx - tree->quadrants_offset >=0);
    const p4est_quadrant_t *quad  = (const p4est_quadrant_t*) sc_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);
    P4EST_ASSERT(quad->level == data->max_lvl);

    double logical_size_of_quad  = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
    double cell_dxyz[P4EST_DIM], cell_volume;
    cell_dxyz[0]          = tree_dimensions[0] * logical_size_of_quad; cell_volume  = cell_dxyz[0];
    cell_dxyz[1]          = tree_dimensions[1] * logical_size_of_quad; cell_volume *= cell_dxyz[1];
#ifdef P4_TO_P8
    cell_dxyz[2]          = tree_dimensions[2] * logical_size_of_quad; cell_volume *= cell_dxyz[2];
#endif

    for (std::map<int, interface_neighbor>::const_iterator itt = map_of_interface_neighbors[quad_idx].begin();
         itt != map_of_interface_neighbors[quad_idx].end(); ++itt)
    {
      int dir                           = itt->first;
      const interface_neighbor& int_nb  = map_of_interface_neighbors[quad_idx][dir];
      P4EST_ASSERT(((int_nb.phi_q > 0.0) && (int_nb.phi_tmp <= 0.0)) || ((int_nb.phi_q <= 0.0) && (int_nb.phi_tmp > 0.0)));

      P4EST_ASSERT(int_nb.mid_point_fine_node_idx >= 0);

      double jump_correction[P4EST_DIM];
      double jump_mu_mid_point = .5*(mu_p(quad_idx) + mu_p(int_nb.quad_tmp_idx) - mu_m(quad_idx) - mu_m(int_nb.quad_tmp_idx));
      double jump_mu_other_fine_node;
      p4est_locidx_t other_fine_node_idx;
      if (int_nb.theta >= 0.5)
      {
        P4EST_ASSERT(int_nb.theta<=1.0);
        // mid_point and point across are fine nodes!
        jump_mu_other_fine_node = mu_p(int_nb.quad_tmp_idx) - mu_m(int_nb.quad_tmp_idx);
        other_fine_node_idx     = int_nb.tmp_fine_node_idx;
      }
      else
      {
        P4EST_ASSERT(int_nb.theta>=0.0);
        // quad and mid-point are fine nodes!
        jump_mu_other_fine_node = mu_p(quad_idx) - mu_m(quad_idx);
        other_fine_node_idx     = int_nb.quad_fine_node_idx;
      }

      const quad_neighbor_nodes_of_node_t *qnnn_mid_point_fine_node, *qnnn_other_fine_node;
      qnnn_mid_point_fine_node  = &fine_node_ngbd->neighbors[int_nb.mid_point_fine_node_idx];
      qnnn_other_fine_node      = &fine_node_ngbd->neighbors[other_fine_node_idx];
      double jump_correction_mid_point[] = {
        jump_mu_mid_point*qnnn_mid_point_fine_node->dx_central(fine_extension_interface_values_read_p)
        , jump_mu_mid_point*qnnn_mid_point_fine_node->dy_central(fine_extension_interface_values_read_p)
#ifdef P4_TO_P8
        , jump_mu_mid_point*qnnn_mid_point_fine_node->dz_central(fine_extension_interface_values_read_p)
#endif
      };
      double jump_correction_other_fine_node[] = {
        jump_mu_other_fine_node*qnnn_other_fine_node->dx_central(fine_extension_interface_values_read_p)
        , jump_mu_other_fine_node*qnnn_other_fine_node->dy_central(fine_extension_interface_values_read_p)
#ifdef P4_TO_P8
        , jump_mu_other_fine_node*qnnn_other_fine_node->dz_central(fine_extension_interface_values_read_p)
#endif
      };
      double normal_vector_mid_point[P4EST_DIM], normal_vector_other_fine_node[P4EST_DIM], norm_normal_vector_mid_point, norm_normal_vector_other_fine_node, inner_product_mid_point, inner_product_other_fine_node;
      norm_normal_vector_mid_point          = norm_normal_vector_other_fine_node = inner_product_mid_point = inner_product_other_fine_node = 0.0;
      for (short dim = 0; dim < P4EST_DIM; ++dim) {
        normal_vector_mid_point[dim]        = normals_read_p[dim][int_nb.mid_point_fine_node_idx];
        normal_vector_other_fine_node[dim]  = normals_read_p[dim][other_fine_node_idx];
        norm_normal_vector_mid_point       += SQR(normal_vector_mid_point[dim]);
        norm_normal_vector_other_fine_node += SQR(normal_vector_other_fine_node[dim]);
        inner_product_mid_point            += jump_correction_mid_point[dim]*normal_vector_mid_point[dim];
        inner_product_other_fine_node      += jump_correction_other_fine_node[dim]*normal_vector_other_fine_node[dim];
      }
      norm_normal_vector_mid_point          = sqrt(norm_normal_vector_mid_point);
      norm_normal_vector_other_fine_node    = sqrt(norm_normal_vector_other_fine_node);
      P4EST_ASSERT((norm_normal_vector_mid_point > EPS) && (norm_normal_vector_other_fine_node > EPS));
      inner_product_mid_point              /= norm_normal_vector_mid_point;
      inner_product_other_fine_node        /= norm_normal_vector_other_fine_node;

      for (short dim = 0; dim < P4EST_DIM; ++dim) {
        normal_vector_mid_point[dim]         /= norm_normal_vector_mid_point;
        normal_vector_other_fine_node[dim]   /= norm_normal_vector_other_fine_node;
        jump_correction_mid_point[dim]       -= normal_vector_mid_point[dim]*inner_product_mid_point;
        jump_correction_other_fine_node[dim] -= normal_vector_other_fine_node[dim]*inner_product_other_fine_node;
        if (int_nb.theta >= 0.5 )
        {
          P4EST_ASSERT(int_nb.theta<=1.0);
          // mid_point and point across are fine nodes!
          jump_correction[dim]                = (2.0 - 2.0*int_nb.theta)*jump_correction_mid_point[dim] + (2.0*int_nb.theta - 1.0)*jump_correction_other_fine_node[dim];
        }
        else
        {
          P4EST_ASSERT(int_nb.theta>=0.0);
          // quad and mid-point are fine nodes!
          jump_correction[dim]                = 2.0*int_nb.theta*jump_correction_mid_point[dim] + (1.0 - 2.0*int_nb.theta)*jump_correction_other_fine_node[dim];
        }
      }

      double sign_jump                = (int_nb.phi_q > 0.0)?+1.0:-1.0;
      double sign_jump_in_flux_factor = (dir%2==0)? -1.0:+1.0;
      double mu_across                = int_nb.mu_this_side*int_nb.mu_other_side/((1.0-int_nb.theta)*int_nb.mu_this_side + int_nb.theta*int_nb.mu_other_side);
      double face_area                = cell_volume/cell_dxyz[dir/2];

      corrected_rhs_p[quad_idx]  += sign_jump*mu_across*sign_jump_in_flux_factor*(jump_correction[dir/2]*(1-int_nb.theta)/int_nb.mu_other_side)*face_area;

    }
  }

  for (short dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecRestoreArrayRead(normals[dim], &normals_read_p[dim]); CHKERRXX(ierr);}
  ierr = VecRestoreArrayRead(phi, &phi_read_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(corrected_rhs, &corrected_rhs_p); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_xgfm_cells_get_corrected_rhs, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_xgfm_cells_t::correct_jump_mu_grad_u()
{
  double *jump_mu_grad_u_p[P4EST_DIM];
  const double *extension_on_fine_nodes_read_p, *normals_read_p[P4EST_DIM];
  for (short dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecGetArray(jump_mu_grad_u[dim], &jump_mu_grad_u_p[dim]); CHKERRXX(ierr);
    ierr = VecGetArrayRead(normals[dim], &normals_read_p[dim]); CHKERRXX(ierr);
  }
  ierr = VecGetArrayRead(extension_on_fine_nodes, &extension_on_fine_nodes_read_p); CHKERRXX(ierr);

  double normal[P4EST_DIM], jump_correction[P4EST_DIM], norm_normal, inner_product;
  const quad_neighbor_nodes_of_node_t *qnnn;

  for (size_t nn = 0; nn < fine_node_ngbd->layer_nodes.size(); ++nn) {
    p4est_locidx_t node_idx = fine_node_ngbd->layer_nodes[nn];
    qnnn = &fine_node_ngbd->neighbors[node_idx];
    jump_correction[0] = (mu_p.get_value() - mu_m.get_value())*qnnn->dx_central(extension_on_fine_nodes_read_p);
    jump_correction[1] = (mu_p.get_value() - mu_m.get_value())*qnnn->dy_central(extension_on_fine_nodes_read_p);
#ifdef P4_TO_P8
    jump_correction[2] = (mu_p.get_value() - mu_m.get_value())*qnnn->dz_central(extension_on_fine_nodes_read_p);
#endif
    norm_normal = inner_product = 0.0;
    for (short dim = 0; dim < P4EST_DIM; ++dim)
    {
      normal[dim]   = normals_read_p[dim][node_idx];
      inner_product += jump_correction[dim]*normal[dim];
      norm_normal   += SQR(normal[dim]);
    }
    norm_normal = sqrt(norm_normal);
    if(norm_normal > EPS)
      inner_product /= norm_normal;
    else
      inner_product = 0.0;

    for (short dim = 0; dim < P4EST_DIM; ++dim)
    {
      if(norm_normal > EPS)
        normal[dim] /= norm_normal;
      else
        normal[dim]         = 0.0;
      jump_correction[dim] -= normal[dim]*inner_product;
      jump_mu_grad_u_p[dim][node_idx] +=  jump_correction[dim];
    }
  }
  for (short dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecGhostUpdateBegin(jump_mu_grad_u[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);}

  for (size_t nn = 0; nn < fine_node_ngbd->local_nodes.size(); ++nn) {
    p4est_locidx_t node_idx = fine_node_ngbd->local_nodes[nn];
    qnnn = &fine_node_ngbd->neighbors[node_idx];
    jump_correction[0] = (mu_p.get_value() - mu_m.get_value())*qnnn->dx_central(extension_on_fine_nodes_read_p);
    jump_correction[1] = (mu_p.get_value() - mu_m.get_value())*qnnn->dy_central(extension_on_fine_nodes_read_p);
#ifdef P4_TO_P8
    jump_correction[2] = (mu_p.get_value() - mu_m.get_value())*qnnn->dz_central(extension_on_fine_nodes_read_p);
#endif
    norm_normal = inner_product = 0.0;
    for (short dim = 0; dim < P4EST_DIM; ++dim)
    {
      normal[dim]   = normals_read_p[dim][node_idx];
      inner_product += jump_correction[dim]*normal[dim];
      norm_normal   += SQR(normal[dim]);
    }
    norm_normal = sqrt(norm_normal);
    if(norm_normal > EPS)
      inner_product /= norm_normal;
    else
      inner_product = 0.0;

    for (short dim = 0; dim < P4EST_DIM; ++dim)
    {
      if(norm_normal > EPS)
        normal[dim] /= norm_normal;
      else
        normal[dim]         = 0.0;
      jump_correction[dim] -= normal[dim]*inner_product;
      jump_mu_grad_u_p[dim][node_idx] +=  jump_correction[dim];
    }
  }
  for (short dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecGhostUpdateEnd(jump_mu_grad_u[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);}

  ierr = VecRestoreArrayRead(extension_on_fine_nodes, &extension_on_fine_nodes_read_p); CHKERRXX(ierr);
  for (short dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecRestoreArray(jump_mu_grad_u[dim], &jump_mu_grad_u_p[dim]); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(normals[dim], &normals_read_p[dim]); CHKERRXX(ierr);
  }
}

void my_p4est_xgfm_cells_t::get_flux_components_and_subtract_them_from_velocities(Vec flux[], my_p4est_faces_t *faces, Vec vstar[], Vec vnp1_minus[], Vec vnp1_plus[])
{
#ifdef CASL_THROWS
  // make sure the faces are properly defined (from the computational grid)
  int my_error = !p4est_is_equal(p4est, faces->get_p4est(), P4EST_FALSE);
  if(my_error)
    throw std::invalid_argument("my_p4est_xgfm_cells_t::get_flux_components_and_subtract_them_from_velocities(Vec [], my_p4est_faces_t*, Vec [], Vec []): the faces must be built from the computational (i.e. coarse) p4est.");

  // compare local sizes and ghost layer sizes
  PetscInt local_size, ghost_layer_size, global_size, global_nb_faces[P4EST_DIM];
  Vec loc_ghost_form;
  for (short dim = 0; dim < P4EST_DIM; ++dim) {
    global_nb_faces[dim] = 0;
    for (int rank = 0; rank < p4est->mpisize; ++rank)
      global_nb_faces[dim] += faces->global_owned_indeps[dim][rank];
    ierr = VecGetSize(flux[dim], &global_size); CHKERRXX(ierr);
    ierr = VecGetLocalSize(flux[dim], &local_size); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(flux[dim], &loc_ghost_form); CHKERRXX(ierr);
    ierr = VecGetSize(loc_ghost_form, &ghost_layer_size); CHKERRXX(ierr);
    ghost_layer_size -= local_size;
    my_error = my_error || (local_size != faces->num_local[dim]) || (ghost_layer_size != faces->num_ghost[dim]) || (global_size != global_nb_faces[dim]);
    ierr = VecGhostRestoreLocalForm(flux[dim], &loc_ghost_form); CHKERRXX(ierr);
  }
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &my_error, 1, MPI_INT, MPI_LOR, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  if(my_error)
    throw std::invalid_argument("my_p4est_xgfm_cells_t::get_flux_components_and_subtract_them_from_velocities(Vec [], my_p4est_faces_t*, Vec [], Vec[]): the flux vectors must have the same layout as if constructed with VecCreateGhostFaces(...) on the computational (coarse) grid.");

  for (short dim = 0; dim < P4EST_DIM; ++dim)
    my_error = my_error || (jump_mu_grad_u[dim] == NULL);
  if(my_error)
    throw std::invalid_argument("my_p4est_xgfm_cells_t::get_flux_components_and_subtract_them_from_velocities(Vec [], my_p4est_faces_t*, Vec [], Vec[]): the flux vectors cannot be calculated if the jumps in mu*grad_u have been returned to the used beforehand.");

  my_error = my_error || (solution_is_set && solution == NULL);
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &my_error, 1, MPI_INT, MPI_LOR, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  if(my_error)
    throw std::invalid_argument("my_p4est_xgfm_cells_t::get_flux_components_and_subtract_them_from_velocities(Vec [], my_p4est_faces_t*, Vec [], Vec[]): the flux vectors cannot be calculated if the solution has been returned to the user beforehand.");

  my_error = my_error || (((vstar != NULL) || (vnp1_minus != NULL) || (vnp1_plus!= NULL)) && ((vstar == NULL) || (vnp1_minus == NULL) || (vnp1_plus == NULL)));
  mpiret = MPI_Allreduce(MPI_IN_PLACE, &my_error, 1, MPI_INT, MPI_LOR, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  if(my_error)
    throw std::invalid_argument("my_p4est_xgfm_cells_t::get_flux_components(Vec [], my_p4est_faces_t*, Vec [], Vec[]): the velocities vstart and vnp1 vectors vectors must either be all defined or be all NULL.");

  if(vstar != NULL && vnp1_minus != NULL && vnp1_plus != NULL)
  {
    Vec* vel;
    for (short flag = 0; flag < 3; ++flag) {
      vel = ((flag == 0)? vstar: ((flag == 1)? vnp1_minus : vnp1_plus));
      for (short dim = 0; dim < P4EST_DIM; ++dim) {
        ierr = VecGetSize(vel[dim], &global_size); CHKERRXX(ierr);
        ierr = VecGetLocalSize(vel[dim], &local_size); CHKERRXX(ierr);
        ierr = VecGhostGetLocalForm(vel[dim], &loc_ghost_form); CHKERRXX(ierr);
        ierr = VecGetSize(loc_ghost_form, &ghost_layer_size); CHKERRXX(ierr);
        ghost_layer_size -= local_size;
        my_error = my_error || (local_size != faces->num_local[dim]) || (ghost_layer_size != faces->num_ghost[dim]) || (global_size != global_nb_faces[dim]);
        ierr = VecGhostRestoreLocalForm(vel[dim], &loc_ghost_form); CHKERRXX(ierr);
      }
    }
    if(my_error)
      throw std::invalid_argument("my_p4est_xgfm_cells_t::get_flux_components_and_subtract_them_from_velocities(Vec [], my_p4est_faces_t*, Vec [], Vec []): the velocities vectors must have the same layout as if constructed with VecCreateGhostFaces(...) on the computational (coarse) grid.");
  }
#endif
  if(!solution_is_set)
    solve();

  P4EST_ASSERT(solution_is_set && (solution != NULL));
  P4EST_ASSERT((((vstar == NULL) && (vnp1_minus == NULL) && (vnp1_plus == NULL)) || ((vstar != NULL) && (vnp1_minus != NULL) && (vnp1_plus != NULL))));
  bool velocities_are_defined = ((vstar != NULL) && (vnp1_minus != NULL) && (vnp1_plus != NULL));

  const double *vstar_read_p[P4EST_DIM], *phi_read_p, *solution_read_p, *jump_u_read_p, *jump_mu_grad_u_read_p[P4EST_DIM], *phi_dd_read_p[P4EST_DIM];
  double *vnp1_plus_p[P4EST_DIM], *vnp1_minus_p[P4EST_DIM], *flux_p[P4EST_DIM];
  std::vector<bool> visited_faces[P4EST_DIM];
  p4est_locidx_t num_visited_faces[P4EST_DIM];
  ierr = VecGetArrayRead(phi, &phi_read_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(solution, &solution_read_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(jump_u, &jump_u_read_p); CHKERRXX(ierr);
  for (short dim = 0; dim < P4EST_DIM; ++dim) {
    visited_faces[dim].resize(faces->num_local[dim], false);
    num_visited_faces[dim] = 0;
    ierr = VecGetArray(flux[dim], &flux_p[dim]); CHKERRXX(ierr);
    ierr = VecGetArrayRead(jump_mu_grad_u[dim], &jump_mu_grad_u_read_p[dim]); CHKERRXX(ierr);
    if(velocities_are_defined){
      ierr = VecGetArrayRead(vstar[dim], &vstar_read_p[dim]); CHKERRXX(ierr);
      ierr = VecGetArray(vnp1_plus[dim], &vnp1_plus_p[dim]); CHKERRXX(ierr);
      ierr = VecGetArray(vnp1_minus[dim], &vnp1_minus_p[dim]); CHKERRXX(ierr);
    }
    if(second_derivatives_of_phi_are_set){
      ierr = VecGetArrayRead(phi_second_der[dim], &phi_dd_read_p[dim]); CHKERRXX(ierr);}
  }

  p4est_locidx_t local_fine_indices_for_quad_interp[P4EST_CHILDREN], local_fine_indices_for_tmp_interp[P4EST_CHILDREN], local_fine_indices_for_face_interp[P4EST_CHILDREN];
  double quad_interp_weights[P4EST_CHILDREN], tmp_interp_weights[P4EST_CHILDREN], face_interp_weights[P4EST_CHILDREN];
  double cell_volume, cell_dxyz[P4EST_DIM], face_xyz[P4EST_DIM];
  p4est_topidx_t nxyz_tree_quad[P4EST_DIM], nxyz_tree_tmp[P4EST_DIM];
  p4est_qcoord_t qxyz_quad[P4EST_DIM], qxyz_tmp[P4EST_DIM], qxyz_face[P4EST_DIM];
  std::vector<p4est_quadrant_t> ngbd;

  for (p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx) {
    p4est_tree_t* tree = (p4est_tree_t*) sc_array_index(p4est->trees, tree_idx);
    for (short dim = 0; dim < P4EST_DIM; ++dim) {
      double rel_top_idx = (p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0] + dim] - xyz_min[dim])/tree_dimensions[dim];
      P4EST_ASSERT(fabs(rel_top_idx - floor(rel_top_idx)) < EPS);
      nxyz_tree_quad[dim]   = ((p4est_topidx_t) floor(rel_top_idx));
    }
    for (size_t q = 0; q < tree->quadrants.elem_count; ++q) {
      p4est_quadrant_t* quad  = (p4est_quadrant_t*) sc_array_index(&tree->quadrants, q);
      p4est_locidx_t quad_idx = q + tree->quadrants_offset;
      qxyz_quad[0] = quad->x + P4EST_QUADRANT_LEN(quad->level+1); cell_dxyz[0] = tree_dimensions[0]*((double) P4EST_QUADRANT_LEN(quad->level)/((double) P4EST_ROOT_LEN)); cell_volume  = cell_dxyz[0];
      qxyz_quad[1] = quad->y + P4EST_QUADRANT_LEN(quad->level+1); cell_dxyz[1] = tree_dimensions[1]*((double) P4EST_QUADRANT_LEN(quad->level)/((double) P4EST_ROOT_LEN)); cell_volume *= cell_dxyz[1];
#ifdef P4_TO_P8
      qxyz_quad[2] = quad->z + P4EST_QUADRANT_LEN(quad->level+1); cell_dxyz[2] = tree_dimensions[2]*((double) P4EST_QUADRANT_LEN(quad->level)/((double) P4EST_ROOT_LEN)); cell_volume *= cell_dxyz[2];
#endif
      bool is_quad_a_fine_node = multilinear_interpolation_weights(nxyz_tree_quad, qxyz_quad, local_fine_indices_for_quad_interp, quad_interp_weights);
      double phi_q = phi_read_p[local_fine_indices_for_quad_interp[0]]*quad_interp_weights[0];
      for (short ccc = 1; ccc < (is_quad_a_fine_node ? 0 : P4EST_CHILDREN); ++ccc)
        phi_q += phi_read_p[local_fine_indices_for_quad_interp[ccc]]*quad_interp_weights[ccc];

      for (short dir = 0; dir < P4EST_FACES; ++dir) {
        p4est_locidx_t face_idx = faces->q2f(quad_idx, dir);
        if ((face_idx >= faces->num_local[dir/2]) || ((face_idx != NO_VELOCITY) && (visited_faces[dir/2][face_idx])))
          continue;

        for (short dim = 0; dim < P4EST_DIM; ++dim)
          qxyz_face[dim] = qxyz_quad[dim] + ((dim == dir/2)? (((dir%2 ==1)? 1:-1)*P4EST_QUADRANT_LEN(quad->level+1)): 0);
        bool is_face_a_fine_node = multilinear_interpolation_weights(nxyz_tree_quad, qxyz_face, local_fine_indices_for_face_interp, face_interp_weights);
        double phi_face = phi_read_p[local_fine_indices_for_face_interp[0]]*face_interp_weights[0];
        for (short ccc = 1; ccc < (is_face_a_fine_node? 0 : P4EST_CHILDREN); ++ccc)
          phi_face += phi_read_p[local_fine_indices_for_face_interp[ccc]]*face_interp_weights[ccc];
        if(is_quad_Wall(p4est, tree_idx, quad, dir))
        {
          P4EST_ASSERT(face_idx != NO_VELOCITY);
          for (short dim = 0; dim < P4EST_DIM; ++dim)
            face_xyz[dim] = xyz_min[dim] + tree_dimensions[dim]*((double) nxyz_tree_quad[dim]+ ((double) qxyz_face[dim])/((double) P4EST_ROOT_LEN));
#ifdef CASL_THROWS
          int is_wall_cell_crossed = ((phi_q > 0.0)  && (phi_face <= 0.0)) || ((phi_q <= 0.0) && (phi_face >  0.0));
          if(is_wall_cell_crossed)
            throw std::invalid_argument("my_p4est_xgfm_cells_t::get_flux_components_and_subtract_them_from_velocities(): a wall-cell is crossed by the interface...");
#endif
          double mu_face   = (phi_face > 0.0)? mu_p(quad_idx): mu_m(quad_idx);
          double val_wall  = bc->wallValue(face_xyz);
          switch(bc->wallType(face_xyz))
          {
          case DIRICHLET:
            flux_p[dir/2][face_idx] = ((dir%2 == 1)? +1.0 : -1.0)*(2.0*mu_face*(val_wall - solution_read_p[quad_idx])/cell_dxyz[dir/2]);
            break;
          case NEUMANN:
            flux_p[dir/2][face_idx] = ((dir%2 == 1)? +1.0 : -1.0)*mu_face*val_wall;
            break;
          default:
            throw std::invalid_argument("my_p4est_xgfm_cells_t::get_flux_components_and_subtract_them_from_velocities(): unknown boundary condition on a wall.");
          }
          if(velocities_are_defined)
          {
            if(phi_face > 0.0)
            {
              vnp1_plus_p[dir/2][face_idx]  = vstar_read_p[dir/2][face_idx] - flux_p[dir/2][face_idx];
              vnp1_minus_p[dir/2][face_idx]  = DBL_MAX;
            }
            else
            {
              vnp1_plus_p[dir/2][face_idx]  = DBL_MAX;
              vnp1_minus_p[dir/2][face_idx] = vstar_read_p[dir/2][face_idx] - flux_p[dir/2][face_idx];
            }
          }
          visited_faces[dir/2][face_idx] = true;
          num_visited_faces[dir/2]++;
          continue;
        }
        /* now get the neighbors */
        ngbd.resize(0);
        cell_ngbd->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, dir);
        if(ngbd.size() == 1)
        {
          P4EST_ASSERT(face_idx != NO_VELOCITY);
          int8_t level_tmp            = ngbd[0].level;
          p4est_locidx_t quad_tmp_idx = ngbd[0].p.piggy3.local_num;
          p4est_topidx_t tree_tmp_idx = ngbd[0].p.piggy3.which_tree;
          for (short dim = 0; dim < P4EST_DIM; ++dim) {
            double rel_top_idx = (p4est->connectivity->vertices[3*p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_tmp_idx + 0] + dim] - xyz_min[dim])/tree_dimensions[dim];
            P4EST_ASSERT(fabs(rel_top_idx - floor(rel_top_idx)) < EPS);
            nxyz_tree_tmp[dim] = ((p4est_topidx_t) floor(rel_top_idx));
          }
          qxyz_tmp[0] = ngbd[0].x + P4EST_QUADRANT_LEN(ngbd[0].level+1);
          qxyz_tmp[1] = ngbd[0].y + P4EST_QUADRANT_LEN(ngbd[0].level+1);
#ifdef P4_TO_P8
          qxyz_tmp[2] = ngbd[0].z + P4EST_QUADRANT_LEN(ngbd[0].level+1);
#endif
          bool is_tmp_a_fine_node = multilinear_interpolation_weights(nxyz_tree_tmp, qxyz_tmp, local_fine_indices_for_tmp_interp, tmp_interp_weights);
          double phi_tmp = phi_read_p[local_fine_indices_for_tmp_interp[0]]*tmp_interp_weights[0];
          for (short ccc = 1; ccc < ((is_tmp_a_fine_node) ? 0: P4EST_CHILDREN); ++ccc)
            phi_tmp += phi_read_p[local_fine_indices_for_tmp_interp[ccc]]*tmp_interp_weights[ccc];

          /* If interface across the two cells.
           * We assume that the interface is tesselated with uniform finest grid level */
          if(((phi_q > 0.0) && (phi_face <= 0.0)) || ((phi_q <= 0.0) && (phi_face > 0.0)) ||
             ((phi_tmp > 0.0) && (phi_face <= 0.0)) || ((phi_tmp <= 0.0) && (phi_face > 0.0)))
          {
            P4EST_ASSERT((quad->level == level_tmp) && (quad->level == ((splitting_criteria_t*)(p4est->user_pointer))->max_lvl));
            P4EST_ASSERT(!visited_faces[dir/2][face_idx]);
            P4EST_ASSERT(is_quad_a_fine_node && is_tmp_a_fine_node && is_face_a_fine_node);

            if(!((phi_face > 0.0)? ((phi_q <= 0.0) && (phi_tmp <= 0.0)):((phi_q > 0.0) && (phi_tmp > 0.0)))) // not under-resolved
            {
              const interface_neighbor int_nb = get_interface_neighbor(quad_idx, dir, quad_tmp_idx, local_fine_indices_for_quad_interp[0], local_fine_indices_for_tmp_interp[0], phi_read_p, phi_dd_read_p);
              P4EST_ASSERT(int_nb.mid_point_fine_node_idx == local_fine_indices_for_face_interp[0]);

              if(int_nb.theta >= 0.5)
              {
                P4EST_ASSERT((phi_face > 0.0)? (int_nb.phi_tmp <= 0.0): (int_nb.phi_tmp > 0.0));
                P4EST_ASSERT((phi_face > 0.0)? (int_nb.phi_q > 0.0)   : (int_nb.phi_q <= 0.0));
                P4EST_ASSERT(int_nb.theta <= 1.0);
                double mu_tilde       = (1.0 - int_nb.theta)*int_nb.mu_this_side + int_nb.theta*int_nb.mu_other_side;
                flux_p[dir/2][face_idx] =
                    ((phi_face <=0.0)? (.5*mu_m(quad_idx) + .5*mu_m(int_nb.quad_tmp_idx)): (.5*mu_p(quad_idx) + .5*mu_p(int_nb.quad_tmp_idx)))*
                    (((dir%2 == 1)?  1.0: -1.0)*(int_nb.mu_other_side/mu_tilde)*(solution_read_p[int_nb.quad_tmp_idx] + ((int_nb.phi_q <=0.0)?-1.0:1.0)*((2.0*int_nb.theta - 1.0)*jump_u_read_p[int_nb.tmp_fine_node_idx] + (2.0 - 2.0*int_nb.theta)*jump_u_read_p[int_nb.mid_point_fine_node_idx]) - solution_read_p[quad_idx])/cell_dxyz[dir/2] +
                    ((int_nb.phi_q <= 0.0)? -1.0: +1.0)*((2.0*int_nb.theta - 1.0)*jump_mu_grad_u_read_p[dir/2][int_nb.tmp_fine_node_idx] + (2.0 - 2.0*int_nb.theta)*jump_mu_grad_u_read_p[dir/2][int_nb.mid_point_fine_node_idx])*(1.0 - int_nb.theta)/mu_tilde);
                visited_faces[dir/2][face_idx] = true;
                num_visited_faces[dir/2]++;
              }
              else
              {
                P4EST_ASSERT((phi_face > 0.0)? (int_nb.phi_tmp > 0.0): (int_nb.phi_tmp <= 0.0));
                P4EST_ASSERT((phi_face > 0.0)? (int_nb.phi_q <= 0.0) : (int_nb.phi_q > 0.0));
                P4EST_ASSERT(int_nb.theta >= 0.0);
                double mu_tilde       = (1.0 - int_nb.theta)*int_nb.mu_this_side + int_nb.theta*int_nb.mu_other_side;
                flux_p[dir/2][face_idx] =
                    ((phi_face <=0)? (.5*mu_m(quad_idx) + .5*mu_m(int_nb.quad_tmp_idx)): (.5*mu_p(quad_idx) + .5*mu_p(int_nb.quad_tmp_idx)))*
                    (((dir%2 == 1)? 1.0: -1.0)*(int_nb.mu_this_side/mu_tilde)*(solution_read_p[int_nb.quad_tmp_idx] + ((int_nb.phi_q <=0.0)?-1.0:1.0)*(2.0*int_nb.theta*jump_u_read_p[int_nb.mid_point_fine_node_idx] + (1.0 - 2.0*int_nb.theta)*jump_u_read_p[int_nb.quad_fine_node_idx]) - solution_read_p[quad_idx])/cell_dxyz[dir/2] +
                    ((int_nb.phi_q <=0.0)?+1.0:-1.0)*(2.0*int_nb.theta*jump_mu_grad_u_read_p[dir/2][int_nb.mid_point_fine_node_idx] + (1.0 - 2.0*int_nb.theta)*jump_mu_grad_u_read_p[dir/2][int_nb.quad_fine_node_idx])*int_nb.theta/mu_tilde);
                visited_faces[dir/2][face_idx] = true;
                num_visited_faces[dir/2]++;
              }
            }

            else
            {
              P4EST_ASSERT((phi_face > 0.0)? ((phi_q <= 0.0) && (phi_tmp <= 0.0)):((phi_q > 0.0) && (phi_tmp > 0.0)));
              flux_p[dir/2][face_idx] = ((phi_face > 0.0)? (.5*mu_m(quad_idx) + .5*mu_m(quad_tmp_idx)): (.5*mu_p(quad_idx) + .5*mu_p(quad_tmp_idx)))*(solution_read_p[quad_tmp_idx] - solution_read_p[quad_idx])/cell_dxyz[dir/2] + ((phi_face > 0.0) ? +1.0 : -1.0)*jump_mu_grad_u_read_p[dir/2][local_fine_indices_for_face_interp[0]];
              visited_faces[dir/2][face_idx] = true;
              num_visited_faces[dir/2]++;
            }
            if(velocities_are_defined)
            {
              if(phi_face > 0.0)
              {
                vnp1_plus_p[dir/2][face_idx]  = vstar_read_p[dir/2][face_idx] - flux_p[dir/2][face_idx];
                vnp1_minus_p[dir/2][face_idx] = DBL_MAX;
              }
              else
              {
                vnp1_plus_p[dir/2][face_idx]  = DBL_MAX;
                vnp1_minus_p[dir/2][face_idx] = vstar_read_p[dir/2][face_idx] - flux_p[dir/2][face_idx];
              }
            }
          }
          /* no interface - regular discretization */
          else
          {
            double s_tmp = pow((double)P4EST_QUADRANT_LEN(ngbd[0].level)/(double)P4EST_ROOT_LEN, (double)P4EST_DIM-1);
            ngbd.resize(0);
            cell_ngbd->find_neighbor_cells_of_cell(ngbd, quad_tmp_idx, tree_tmp_idx, dir%2==0 ? dir+1 : dir-1);
  #ifdef DEBUG
            double check_sum = 0.0;
  #endif
            double weighted_distance = 0.0;
            double local_flux = 0.0;
            for(unsigned int i=0; i<ngbd.size(); ++i)
            {
              quad_xyz_fr_q(ngbd[i].p.piggy3.local_num, ngbd[i].p.piggy3.which_tree, p4est, ghost, face_xyz);
              face_xyz[dir/2] += ((dir%2 == 0)? -1.0: +1.0)*(((double)(P4EST_QUADRANT_LEN(ngbd[i].level +1))/(double)P4EST_ROOT_LEN))*tree_dimensions[dir/2];
              double mu_face = (phi_face <= 0.0)? mu_m(face_xyz) : mu_p(face_xyz);
              double s_nb = pow((double)P4EST_QUADRANT_LEN(ngbd[i].level)/(double)P4EST_ROOT_LEN, (double)P4EST_DIM-1);
              weighted_distance += (s_nb/s_tmp) * 0.5 * (double)(P4EST_QUADRANT_LEN(level_tmp)+P4EST_QUADRANT_LEN(ngbd[i].level))/(double)P4EST_ROOT_LEN;
              local_flux += mu_face*(solution_read_p[quad_tmp_idx] - solution_read_p[ngbd[i].p.piggy3.local_num])*s_nb/s_tmp;
#ifdef DEBUG
              check_sum += s_nb/s_tmp;
#endif
            }
            P4EST_ASSERT(fabs(check_sum - 1.0) < EPS);
            weighted_distance *= tree_dimensions[dir/2];
            local_flux *= ((dir%2 == 1)? +1.0:-1.0)/weighted_distance;
            for (unsigned int i = 0; i < ngbd.size(); ++i) {
              face_idx = faces->q2f(ngbd[i].p.piggy3.local_num, dir);
              P4EST_ASSERT(face_idx != NO_VELOCITY);
              if(face_idx < faces->num_local[dir/2] && !visited_faces[dir/2][face_idx])
              {
                flux_p[dir/2][face_idx]         = local_flux;
                visited_faces[dir/2][face_idx]  = true;
                num_visited_faces[dir/2]++;
                if(velocities_are_defined)
                {
                  if(phi_face > 0.0)
                  {
                    vnp1_plus_p[dir/2][face_idx]  = vstar_read_p[dir/2][face_idx] - flux_p[dir/2][face_idx];
                    vnp1_minus_p[dir/2][face_idx] = DBL_MAX;
                  }
                  else
                  {
                    vnp1_plus_p[dir/2][face_idx]  = DBL_MAX;
                    vnp1_minus_p[dir/2][face_idx] = vstar_read_p[dir/2][face_idx] - flux_p[dir/2][face_idx];
                  }
                }
              }
            }
          }
        }
        /* There is more than one neighbor, regular bulk case.
         * The current face_idx is (must be) NO_VELOCITY, but all neighbors have a well-defined velocity dof.
         * The flux component is identical for all those neighbor's well-defined face dofs, by construction.
         * --> Update all of them simultaneously
         * We assumes a uniform tesselation on the interface ! */
        else if(ngbd.size() > 1)
        {
          P4EST_ASSERT(face_idx == NO_VELOCITY);
          double s_quad = pow((double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN, (double)P4EST_DIM-1);
#ifdef DEBUG
          double check_sum = 0.0;
#endif
          double weighted_distance = 0.0;
          double local_flux = 0.0;
          for(unsigned int i=0; i<ngbd.size(); ++i)
          {
            quad_xyz_fr_q(ngbd[i].p.piggy3.local_num, ngbd[i].p.piggy3.which_tree, p4est, ghost, face_xyz);
            face_xyz[dir/2] += ((dir%2 == 0)? +1.0: -1.0)*(((double)(P4EST_QUADRANT_LEN(ngbd[i].level +1))/(double)P4EST_ROOT_LEN))*tree_dimensions[dir/2];
            double mu_face = (phi_face <= 0.0)? mu_m(face_xyz) : mu_p(face_xyz);
            double s_nb = pow((double)P4EST_QUADRANT_LEN(ngbd[i].level)/(double)P4EST_ROOT_LEN, (double)P4EST_DIM-1);
            weighted_distance += (s_nb/s_quad) * 0.5 * (double)(P4EST_QUADRANT_LEN(quad->level)+P4EST_QUADRANT_LEN(ngbd[i].level))/(double)P4EST_ROOT_LEN;
            local_flux += mu_face*(solution_read_p[ngbd[i].p.piggy3.local_num] - solution_read_p[quad_idx])*s_nb/s_quad;
#ifdef DEBUG
            check_sum += s_nb/s_quad;
#endif
          }
          P4EST_ASSERT(fabs(check_sum - 1.0) < EPS);
          weighted_distance *= tree_dimensions[dir/2];
          local_flux *= ((dir%2 == 1)? +1.0:-1.0)/weighted_distance;
          for (unsigned int i = 0; i < ngbd.size(); ++i) {
            face_idx = faces->q2f(ngbd[i].p.piggy3.local_num, dir + ((dir%2 == 1)? -1: +1));
            P4EST_ASSERT(face_idx != NO_VELOCITY);
            if(face_idx < faces->num_local[dir/2] && !visited_faces[dir/2][face_idx])
            {
              flux_p[dir/2][face_idx]         = local_flux;
              visited_faces[dir/2][face_idx]  = true;
              num_visited_faces[dir/2]++;
              if(velocities_are_defined)
              {
                if(phi_face > 0.0)
                {
                  vnp1_plus_p[dir/2][face_idx]  = vstar_read_p[dir/2][face_idx] - flux_p[dir/2][face_idx];
                  vnp1_minus_p[dir/2][face_idx] = DBL_MAX;
                }
                else
                {
                  vnp1_plus_p[dir/2][face_idx]  = DBL_MAX;
                  vnp1_minus_p[dir/2][face_idx] = vstar_read_p[dir/2][face_idx] - flux_p[dir/2][face_idx];
                }
              }
            }
          }
        }
      }
    }
  }

  ierr = VecRestoreArrayRead(phi, &phi_read_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(solution, &solution_read_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(jump_u, &jump_u_read_p); CHKERRXX(ierr);
  for (short dim = 0; dim < P4EST_DIM; ++dim) {
    P4EST_ASSERT(num_visited_faces[dim] == faces->num_local[dim]);
    ierr = VecRestoreArray(flux[dim], &flux_p[dim]); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(jump_mu_grad_u[dim], &jump_mu_grad_u_read_p[dim]); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(flux[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    if(velocities_are_defined){
      ierr = VecRestoreArrayRead(vstar[dim], &vstar_read_p[dim]); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(vnp1_plus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(vnp1_minus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd  (vnp1_plus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd  (vnp1_minus[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecRestoreArray(vnp1_plus[dim], &vnp1_plus_p[dim]); CHKERRXX(ierr);
      ierr = VecRestoreArray(vnp1_minus[dim], &vnp1_minus_p[dim]); CHKERRXX(ierr);
    }
    if(second_derivatives_of_phi_are_set){
      ierr = VecRestoreArrayRead(phi_second_der[dim], &phi_dd_read_p[dim]); CHKERRXX(ierr);}
    ierr = VecGhostUpdateEnd  (flux[dim], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }
}

#ifdef DEBUG
int my_p4est_xgfm_cells_t::is_map_consistent() const
{

  int mpiret;

  std::vector<int> senders(p4est->mpisize, 0);
  int num_expected_replies = 0;

  int it_is_alright = true;
  std::map<int, std::vector<which_interface_nb> > map_of_query_interface_neighbors; map_of_query_interface_neighbors.clear();
  std::map<int, std::vector<which_interface_nb> > map_of_mirrors; map_of_mirrors.clear();
  for (std::map<p4est_locidx_t, std::map<int, interface_neighbor> >::const_iterator it = map_of_interface_neighbors.begin();
       it != map_of_interface_neighbors.end(); ++it)
  {
    p4est_locidx_t quad_idx = it->first;
    for (std::map<int, interface_neighbor>::const_iterator itt = (map_of_interface_neighbors.at(quad_idx)).begin();
         itt != (map_of_interface_neighbors.at(quad_idx)).end(); ++itt)
    {
      int dir = itt->first;
      // the neighbor is a local quad
      if(((map_of_interface_neighbors.at(quad_idx)).at(dir)).quad_tmp_idx < p4est->local_num_quadrants)
      {
        p4est_locidx_t loc_quad_idx_tmp = ((map_of_interface_neighbors.at(quad_idx)).at(dir)).quad_tmp_idx;
        it_is_alright = it_is_alright && ((map_of_interface_neighbors.at(quad_idx)).at(dir)).is_consistent_with_neighbor_across(((map_of_interface_neighbors.at(loc_quad_idx_tmp)).at(dir + ((dir%2 == 0)?1:-1))));
        if(!it_is_alright && !((map_of_interface_neighbors.at(quad_idx)).at(dir)).is_consistent_with_neighbor_across(((map_of_interface_neighbors.at(loc_quad_idx_tmp)).at(dir + ((dir%2 == 0)?1:-1)))))
          std::cerr << "quad " << quad_idx << " on proc " << p4est->mpirank << " has quad " << loc_quad_idx_tmp << " as a neighbor on proc " << p4est->mpirank << " across and the interface value is " << ((map_of_interface_neighbors.at(quad_idx)).at(dir)).int_value << " while the value from the other is " << ((map_of_interface_neighbors.at(loc_quad_idx_tmp)).at(dir + ((dir%2 == 0)?1:-1))).int_value <<  std::endl;
      }
      else
      {
        const p4est_quadrant_t* ghost_nb_quad = (const p4est_quadrant_t*) sc_array_index(&ghost->ghosts, ((map_of_interface_neighbors.at(quad_idx)).at(dir)).quad_tmp_idx - p4est->local_num_quadrants);
        int rank_owner = p4est_comm_find_owner(p4est, ghost_nb_quad->p.piggy3.which_tree, ghost_nb_quad, p4est->mpirank);
        P4EST_ASSERT(rank_owner != p4est->mpirank);
        num_expected_replies += ((senders[rank_owner] == 1)? 0: 1);
        senders[rank_owner] = 1;
        which_interface_nb new_query;

        new_query.loc_idx = ghost_nb_quad->p.piggy3.local_num;
        new_query.dir = dir + ((dir%2 == 0)?1:-1);
        map_of_query_interface_neighbors[rank_owner].push_back(new_query);
        which_interface_nb this_one;
        this_one.loc_idx  = quad_idx;
        this_one.dir      = dir;
        map_of_mirrors[rank_owner].push_back(this_one);
      }
    }
  }

  std::vector<int> recvcount(p4est->mpisize, 1);
  int num_remaining_queries = 0;
  mpiret = MPI_Reduce_scatter(&senders[0], &num_remaining_queries, &recvcount[0], MPI_INT, MPI_SUM, p4est->mpicomm); SC_CHECK_MPI(mpiret);

  std::vector<MPI_Request> mpi_query_requests; mpi_query_requests.resize(0);
  std::vector<MPI_Request> mpi_reply_requests; mpi_reply_requests.resize(0);

  // send the requests...
  for (std::map<int, std::vector<which_interface_nb> >::const_iterator it = map_of_query_interface_neighbors.begin();
       it != map_of_query_interface_neighbors.end(); ++it) {
    if (it->first == p4est->mpirank)
      continue;

    int rank = it->first;
    P4EST_ASSERT(senders[rank] == 1);

    MPI_Request req;
    mpiret = MPI_Isend((void*)&map_of_query_interface_neighbors[rank][0], sizeof(which_interface_nb)*(map_of_query_interface_neighbors[rank].size()), MPI_BYTE, it->first, 15351, p4est->mpicomm, &req); SC_CHECK_MPI(mpiret);
    mpi_query_requests.push_back(req);
  }

  std::map<int, std::vector<interface_neighbor> > map_of_responses; map_of_responses.clear();

  MPI_Status status;
  bool done = (num_expected_replies == 0 && num_remaining_queries == 0);
  while (!done) {
    if(num_remaining_queries > 0)
    {
      int is_msg_pending;
      mpiret = MPI_Iprobe(MPI_ANY_SOURCE, 15351, p4est->mpicomm, &is_msg_pending, &status); SC_CHECK_MPI(mpiret);
      if (is_msg_pending)
      {
        int byte_count;
        mpiret = MPI_Get_count(&status, MPI_BYTE, &byte_count); SC_CHECK_MPI(mpiret);
        P4EST_ASSERT(byte_count%sizeof(which_interface_nb) == 0);
        std::vector<which_interface_nb> queried_indices(byte_count/sizeof(which_interface_nb));

        mpiret = MPI_Recv((void*) &queried_indices[0], byte_count, MPI_BYTE, status.MPI_SOURCE, 15351, p4est->mpicomm, MPI_STATUS_IGNORE); SC_CHECK_MPI(mpiret);

        std::vector<interface_neighbor>& response = map_of_responses[status.MPI_SOURCE];
        response.resize(byte_count/sizeof(which_interface_nb));

        for (size_t kk = 0; kk < response.size(); ++kk)
          response[kk] = (map_of_interface_neighbors.at(queried_indices[kk].loc_idx)).at(queried_indices[kk].dir);

        // we are done, lets send the buffer back
        MPI_Request req;
        mpiret = MPI_Isend((void*)&response[0], (response.size())*sizeof(interface_neighbor), MPI_BYTE, status.MPI_SOURCE, 42624, p4est->mpicomm, &req); SC_CHECK_MPI(mpiret);
        mpi_reply_requests.push_back(req);
        num_remaining_queries--;
      }
    }

    if(num_expected_replies > 0)
    {
      int is_msg_pending;
      mpiret = MPI_Iprobe(MPI_ANY_SOURCE, 42624, p4est->mpicomm, &is_msg_pending, &status); SC_CHECK_MPI(mpiret);
      if (is_msg_pending)
      {

        int byte_count;
        mpiret = MPI_Get_count(&status, MPI_BYTE, &byte_count); SC_CHECK_MPI(mpiret);
        P4EST_ASSERT(byte_count%sizeof(interface_neighbor) == 0);
        std::vector<interface_neighbor> reply_buffer (byte_count / sizeof(interface_neighbor));

        mpiret = MPI_Recv((void*)&reply_buffer[0], byte_count, MPI_BYTE, status.MPI_SOURCE, 42624, p4est->mpicomm, MPI_STATUS_IGNORE);  SC_CHECK_MPI(mpiret);
        for (size_t kk = 0; kk < reply_buffer.size(); ++kk) {
          which_interface_nb mirror   = map_of_mirrors[status.MPI_SOURCE][kk];
          which_interface_nb queried  = map_of_query_interface_neighbors[status.MPI_SOURCE][kk];
          it_is_alright = it_is_alright && ((map_of_interface_neighbors.at(mirror.loc_idx)).at(mirror.dir)).is_consistent_with_neighbor_across(reply_buffer[kk]);
          if(!it_is_alright && !((map_of_interface_neighbors.at(mirror.loc_idx)).at(mirror.dir)).is_consistent_with_neighbor_across(reply_buffer[kk]))
            std::cerr << "quad " << mirror.loc_idx << " on proc " << p4est->mpirank << " has quad " << queried.loc_idx << " as a neighbor on proc " << status.MPI_SOURCE << " across and the interface value is " << ((map_of_interface_neighbors.at(mirror.loc_idx)).at(mirror.dir)).int_value << " while the value from the other is " << reply_buffer[kk].int_value <<  std::endl;
        }

        num_expected_replies--;
      }
    }
    done = (num_expected_replies == 0 && num_remaining_queries == 0);
  }

  mpiret = MPI_Waitall(mpi_query_requests.size(), &mpi_query_requests[0], MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Waitall(mpi_reply_requests.size(), &mpi_reply_requests[0], MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret);
  mpi_query_requests.clear();
  mpi_reply_requests.clear();

  mpiret = MPI_Allreduce(MPI_IN_PLACE, &it_is_alright, 1, MPI_INT, MPI_LAND, p4est->mpicomm); SC_CHECK_MPI(mpiret);

  return it_is_alright;
}
#endif
