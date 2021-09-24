#include "my_p4est_surfactant.h"

#ifdef P4_TO_P8
#include <src/my_p8est_poisson_cells.h>
#include <src/my_p8est_poisson_faces.h>
#include <src/my_p8est_level_set.h>
#include <src/my_p8est_level_set_cells.h>
#include <src/my_p8est_level_set_faces.h>
#include <src/my_p8est_trajectory_of_point.h>
#include <src/my_p8est_semi_lagrangian.h>
#include <src/my_p8est_vtk.h>
#include <p8est_extended.h>
#include <p8est_algorithms.h>
#else
#include <src/my_p4est_semi_lagrangian.h>
#include <src/my_p4est_poisson_cells.h>
#include <src/my_p4est_poisson_faces.h>
#include <src/my_p4est_level_set.h>
#include <src/my_p4est_level_set_cells.h>
#include <src/my_p4est_level_set_faces.h>
#include <src/my_p4est_trajectory_of_point.h>
#include <src/my_p4est_vtk.h>
#include <p4est_extended.h>
#include <p4est_algorithms.h>
#endif

#include <algorithm>

//#ifndef CASL_LOG_EVENTS
//#undef PetscLogEventBegin
//#undef PetscLogEventEnd
//#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
//#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
//#else
//extern PetscLogEvent log_my_p4est_navier_stokes_viscosity;
//extern PetscLogEvent log_my_p4est_navier_stokes_projection;
//extern PetscLogEvent log_my_p4est_navier_stokes_update;
//#endif


void my_p4est_surfactant_t::splitting_criteria_surfactant_t::tag_quadrant(p4est_t *p4est, p4est_locidx_t quad_idx, p4est_topidx_t tree_idx,
                                                                          p4est_nodes_t *nodes, const double *phi_p)
{
  p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
  p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index( &tree->quadrants, quad_idx-(tree->quadrants_offset) );

  if (quad->level < min_lvl)
    quad->p.user_int = REFINE_QUADRANT; // if coarser than min_lvl, mark for refinement

  else if (quad->level > max_lvl)
    quad->p.user_int = COARSEN_QUADRANT; // if finer than max_lvl, mark for coarsening

  else
  {
    const double quad_diag = (_prnt->dxyz_diag)*(1<<( max_lvl-(quad->level) ));

    bool coarsen = (quad->level > min_lvl);
    if(coarsen)
    {
      bool coar_intf = true;
      bool coar_band = true;
      p4est_locidx_t node_idx;

      // check the P4EST_CHILDREN vertices of the quadrant and enforce all conditions in each
      for(unsigned short v=0; v < P4EST_CHILDREN; ++v)
      {
        node_idx  = nodes->local_nodes[P4EST_CHILDREN*quad_idx+v];

        coar_intf = coar_intf && ( fabs(phi_p[node_idx]) >= lip*2.0*quad_diag );
        coar_band = coar_band && ( _prnt->phi_band_gen.band_fn(phi_p[node_idx]) > (_prnt->sp_crit->uniform_padding)*(_prnt->dxyz_diag) );

        coarsen = coar_intf && coar_band; // need ALL of the coarsening conditions satisfied to coarsen the quadrant
        if(!coarsen)
          break;
      }
    }

    bool refine = (quad->level < max_lvl);
    if(refine)
    {
      bool ref_intf = false;
      bool ref_band = false;
      p4est_locidx_t node_idx;

      // check the P4EST_CHILDREN vertices of the quadrant and enforce all conditions in each
      for(unsigned short v=0; v < P4EST_CHILDREN; ++v)
      {
        node_idx  = nodes->local_nodes[P4EST_CHILDREN*quad_idx+v];

        ref_intf = ref_intf || ( fabs(phi_p[node_idx]) <= lip*quad_diag );
        ref_band = ref_band || ( _prnt->phi_band_gen.band_fn(phi_p[node_idx]) < (_prnt->sp_crit->uniform_padding)*(_prnt->dxyz_diag) ); // WARNING: Review how to optimize (if possible) the 2.0 value

        refine = ref_intf || ref_band; // need AT LEAST ONE of the refining conditions satisfied to refine the quadrant
        if(refine)
          break;
      }
      // [FERNANDO:] The refinement conditions are currently NOT ENFORCED on hanging nodes, TO_BE_IMPLEMENTED here.
    }

    if (refine)
      quad->p.user_int = REFINE_QUADRANT;
    else if (coarsen)
      quad->p.user_int = COARSEN_QUADRANT;
    else
      quad->p.user_int = SKIP_QUADRANT;
  }
}


bool my_p4est_surfactant_t::splitting_criteria_surfactant_t::refine_and_coarsen(p4est_t* p4est, p4est_nodes_t* nodes, Vec phi)
{
  const double *phi_p;
  PetscErrorCode ierr;
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
  // tag the quadrants that need to be refined or coarsened
  for (p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx)
  {
    p4est_tree_t* tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for (size_t q = 0; q <tree->quadrants.elem_count; ++q)
    {
      p4est_locidx_t quad_idx  = q + tree->quadrants_offset;
      tag_quadrant(p4est, quad_idx, tree_idx, nodes, phi_p);
    }
  }
  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);

  my_p4est_coarsen(p4est, P4EST_FALSE, splitting_criteria_surfactant_t::coarsen_fn, splitting_criteria_surfactant_t::init_fn);
  my_p4est_refine (p4est, P4EST_FALSE, splitting_criteria_surfactant_t::refine_fn,  splitting_criteria_surfactant_t::init_fn);

  int is_grid_changed = false;
  for (p4est_topidx_t it = p4est->first_local_tree; it <= p4est->last_local_tree; ++it)
  {
    p4est_tree_t* tree = (p4est_tree_t*)sc_array_index(p4est->trees, it);
    for (size_t q = 0; q <tree->quadrants.elem_count; ++q)
    {
      p4est_quadrant_t *quad = (p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
      if (quad->p.user_int == NEW_QUADRANT)
      {
        is_grid_changed = true;
        goto function_end;
      }
    }
  }

function_end:
  MPI_Allreduce(MPI_IN_PLACE, &is_grid_changed, 1, MPI_INT, MPI_LOR, p4est->mpicomm);
  return is_grid_changed;
}


void my_p4est_surfactant_t::update_phi_band_vector()
{
  PetscErrorCode ierr;
  if(phi_band!=NULL) { ierr = VecDestroy(phi_band); CHKERRXX(ierr); }
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &phi_band); CHKERRXX(ierr);
  const double *phi_p;
  double *phi_band_p;
  VecGetArrayRead(phi, &phi_p);
  VecGetArray(phi_band, &phi_band_p);
  for (size_t n=0; n<nodes_n->indep_nodes.elem_count; ++n)
  {
    phi_band_p[n] = phi_band_gen.band_fn(phi_p[n]);
  }
  VecRestoreArrayRead(phi, &phi_p);
  VecGetArray(phi_band, &phi_band_p);
}


void my_p4est_surfactant_t::enforce_refinement_p4est_n(CF_DIM* ls)
{
  // Create PETSc error flag
  PetscErrorCode ierr;

  // Create 'final' forest, for now it's a copy of the input forest
  p4est_t *p4est_f = NULL;
  p4est_f = p4est_copy(p4est_n, P4EST_FALSE);
  p4est_f->connectivity = p4est_n->connectivity;
  p4est_f->user_pointer = (void*)(sp_crit);

  // Create objects associated to the new forest
  p4est_ghost_t *ghost_f = NULL; ghost_f = my_p4est_ghost_new(p4est_f, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes_f = NULL; nodes_f = my_p4est_nodes_new(p4est_f, ghost_f);
  my_p4est_hierarchy_t *hierarchy_f = NULL;
  my_p4est_node_neighbors_t *ngbd_f = NULL;

  // Create 'final' level-set function, for now it's a copy of the input level-set function
  Vec phi_f = NULL;
  ierr = VecCreateGhostNodes(p4est_f, nodes_f, &phi_f); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est_f, nodes_f, *ls, phi_f);

  // Impose refinement + coarsening criteria and iterate if the grid changes
  bool grid_is_changing = true;
  int iter = 0;
  while(grid_is_changing)
  {
    grid_is_changing = sp_crit->refine_and_coarsen(p4est_f, nodes_f, phi_f);

    if(grid_is_changing)
    {
      my_p4est_partition(p4est_f, P4EST_FALSE, NULL);
      if(ghost_f!=NULL)     { p4est_ghost_destroy(ghost_f); } ghost_f     = my_p4est_ghost_new(p4est_f, P4EST_CONNECT_FULL);
      if(nodes_f!=NULL)     { p4est_nodes_destroy(nodes_f); } nodes_f     = my_p4est_nodes_new(p4est_f, ghost_f);
      if(hierarchy_f!=NULL) { delete hierarchy_f; }           hierarchy_f = new my_p4est_hierarchy_t(p4est_f, ghost_f, &brick);
      if(ngbd_f!=NULL)      { delete ngbd_f; }                ngbd_f      = new my_p4est_node_neighbors_t(hierarchy_f, nodes_f);

      ierr = VecDestroy(phi_f); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est_f, nodes_f, &phi_f); CHKERRXX(ierr);
      sample_cf_on_nodes(p4est_f, nodes_f, *ls, phi_f);

      // Reinitialize final level-set function
      my_p4est_level_set_t lsn(ngbd_f);
      lsn.reinitialize_2nd_order(phi_f);
      lsn.perturb_level_set_function(phi_f, EPS*dxyz_min);

      iter++;
      if(iter>1+sp_crit->max_lvl-sp_crit->min_lvl)
      {
        ierr = PetscPrintf(p4est_f->mpicomm, "The grid update did not converge\n"); CHKERRXX(ierr);
        break;
      }
    }
  }

  // Update input forest and level-set function with the 'final' ones
  if(iter>0)
  {
    p4est_destroy(p4est_n);
    p4est_n = p4est_copy(p4est_f, P4EST_FALSE);
    p4est_n->connectivity = p4est_f->connectivity;
    p4est_n->user_pointer = (void*)(sp_crit);
    p4est_ghost_destroy(ghost_n); ghost_n = my_p4est_ghost_new(p4est_n, P4EST_CONNECT_FULL);
    p4est_nodes_destroy(nodes_n); nodes_n = my_p4est_nodes_new(p4est_n, ghost_n);
    delete hierarchy_n; hierarchy_n = new my_p4est_hierarchy_t(p4est_n, ghost_n, &brick);
    delete ngbd_n; ngbd_n = new my_p4est_node_neighbors_t(hierarchy_n, nodes_n); ngbd_n->init_neighbors();
  }

  // Delete objects that involve memory allocation
  p4est_destroy(p4est_f);
  p4est_ghost_destroy(ghost_f);
  p4est_nodes_destroy(nodes_f);
  if(hierarchy_f!=NULL) { delete hierarchy_f; }
  if(ngbd_f!=NULL) { delete ngbd_f; }
  ierr = VecDestroy(phi_f); CHKERRXX(ierr);
}


my_p4est_surfactant_t::my_p4est_surfactant_t(const mpi_environment_t& mpi,
                                             const double xyz_min_[], const double xyz_max_[], const int n_xyz_[], const int periodicity_[],
                                             const int& lmin, const int& lmax,
                                             CF_DIM *ls_n,
                                             const double& CFL_input,
                                             const double& uniform_padding_input,
                                             const double& band_width)
  : STATIC_GRID(false),
    NO_SURFACE_DIFFUSION(false),
    NO_SURFACE_ADVECTION(false),
    ITER_EXTENSION(50),
    TOL_BAND_NODES(1.5),
    CFL(CFL_input),
    phi_band_gen(this, band_width),
    num_band_nodes(0),
    sl_dep_pts_computed(false),
    sl_dep_pts_s_computed(false)
{
  // Create brick and connectivity
  conn = my_p4est_brick_new(n_xyz_, xyz_min_, xyz_max_, &brick, periodicity_);

  // Set all physical variables to 1.0 for now
  D_s = 1.0;

  // Initialize the refinement criteria that will be used throughout
  sp_crit = new splitting_criteria_surfactant_t(this, lmin, lmax, ls_n->lip, uniform_padding_input);

  // Create trees and associated structures at time n (initial time t0)
  splitting_criteria_t *sp_crit_temp = NULL;
  sp_crit_temp = new splitting_criteria_cf_and_uniform_band_t(lmin, lmax, ls_n, sp_crit->uniform_padding+0.5*(phi_band_gen.band_width), ls_n->lip);
  p4est_n = my_p4est_new(mpi.comm(), conn, 0, NULL, NULL);
  p4est_n->user_pointer = (void*)(sp_crit_temp);
  for(int l=0; l<lmax; ++l)
  {
    my_p4est_refine(p4est_n, P4EST_FALSE, refine_levelset_cf_and_uniform_band, NULL);
    my_p4est_partition(p4est_n, P4EST_FALSE, NULL);
  }
  ghost_n = my_p4est_ghost_new(p4est_n, P4EST_CONNECT_FULL);
  nodes_n = my_p4est_nodes_new(p4est_n, ghost_n);
  hierarchy_n = new my_p4est_hierarchy_t(p4est_n, ghost_n, &brick);
  ngbd_n = new my_p4est_node_neighbors_t(hierarchy_n, nodes_n);
  ngbd_n->init_neighbors();

  // Delete temporary splitting criteria and assign the internal one
  delete sp_crit_temp;
  p4est_n->user_pointer = (void*)(sp_crit);

  // Compute smallest grid sizes
  get_dxyz_min(p4est_n, dxyz, dxyz_min, dxyz_diag);

  // Enforce refinement criteria (not fully enforced above). Doubly helpful if the prescribed 'ls_n' is not a level-set function
  enforce_refinement_p4est_n(ls_n);

  // Sample level-set function into 'phi' and reinitialize
  set_phi(ls_n);
  compute_num_band_nodes();

  // Create trees and associated structures at time nm1 (previous time step t0-dt_nm1)
  p4est_nm1 = my_p4est_copy(p4est_n, P4EST_FALSE);
  p4est_nm1->connectivity = conn;
  p4est_nm1->user_pointer = (void*)(sp_crit);
  ghost_nm1 = my_p4est_ghost_new(p4est_nm1, P4EST_CONNECT_FULL);
  nodes_nm1 = my_p4est_nodes_new(p4est_nm1, ghost_nm1);
  hierarchy_nm1 = new my_p4est_hierarchy_t(p4est_nm1, ghost_nm1, &brick);
  ngbd_nm1 = new my_p4est_node_neighbors_t(hierarchy_nm1, nodes_nm1);
  ngbd_nm1->init_neighbors();
  p4est_nm1->user_pointer = (void*)(sp_crit);

  // Set time steps, for now we take dt = dx everywhere, we will update when (and if) we setup the velocities
  dt_nm1 = dt_n = CFL * dxyz_min;

  // Initialize variables to NULL for now
  Gamma_np1 = NULL;
  Gamma_n   = NULL;
  Gamma_nm1 = NULL;

  str_n = NULL;
  str_nm1 = NULL;

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    vn_nodes[dir]     = NULL;
    vnm1_nodes[dir]   = NULL;
    vn_s_nodes[dir]   = NULL;
    vnm1_s_nodes[dir] = NULL;

    for (unsigned short dd = 0; dd < P4EST_DIM; ++dd)
    {
      dd_vn_nodes[dd][dir]      = NULL;
      dd_vnm1_nodes[dd][dir]    = NULL;
      dd_vn_s_nodes[dd][dir]    = NULL;
      dd_vnm1_s_nodes[dd][dir]  = NULL;
    }
  }

  // Compute normals, curvature, and sample the band l-s function
  for(unsigned short dir=0; dir<P4EST_DIM; ++dir) { normal[dir] = NULL; }
  kappa = NULL;
  phi_band = NULL;
  compute_normal();
  compute_curvature();
  update_phi_band_vector();
}


my_p4est_surfactant_t::~my_p4est_surfactant_t()
{
  // Create PETSc error flag
  PetscErrorCode ierr;

  // Destroy dynamically allocated objects
  if(phi!=NULL)      { ierr = VecDestroy(phi)      ; CHKERRXX(ierr); }
  if(phi_band!=NULL) { ierr = VecDestroy(phi_band) ; CHKERRXX(ierr); }
  if(kappa!=NULL)    { ierr = VecDestroy(kappa)    ; CHKERRXX(ierr); }

  if(Gamma_nm1!=NULL){ ierr = VecDestroy(Gamma_nm1); CHKERRXX(ierr); }
  if(Gamma_n  !=NULL){ ierr = VecDestroy(Gamma_n  ); CHKERRXX(ierr); }
  if(Gamma_np1!=NULL){ ierr = VecDestroy(Gamma_np1); CHKERRXX(ierr); }

  if(str_nm1!=NULL){ ierr = VecDestroy(str_nm1); CHKERRXX(ierr); }
  if(str_n  !=NULL){ ierr = VecDestroy(str_n  ); CHKERRXX(ierr); }

  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    if(vn_nodes[dir]  !=NULL)       { ierr = VecDestroy(vn_nodes[dir]  ); CHKERRXX(ierr); }
    if(vnm1_nodes[dir]!=NULL)       { ierr = VecDestroy(vnm1_nodes[dir]); CHKERRXX(ierr); }
    if(vn_s_nodes[dir]  !=NULL)     { ierr = VecDestroy(vn_nodes[dir]  ); CHKERRXX(ierr); }
    if(vnm1_s_nodes[dir]!=NULL)     { ierr = VecDestroy(vnm1_nodes[dir]); CHKERRXX(ierr); }
    if(normal[dir]!=NULL)           { ierr = VecDestroy(vnm1_nodes[dir]); CHKERRXX(ierr); }

    for (unsigned short dd = 0; dd < P4EST_DIM; ++dd)
    {
      if(dd_vn_nodes[dd][dir]!= NULL) { ierr = VecDestroy(dd_vn_nodes[dd][dir]) ; CHKERRXX(ierr); };
      if(dd_vnm1_nodes[dd][dir] != NULL) { ierr = VecDestroy(dd_vnm1_nodes[dd][dir]) ; CHKERRXX(ierr); };
      if(dd_vn_s_nodes[dd][dir]!= NULL) { ierr = VecDestroy(dd_vn_s_nodes[dd][dir]) ; CHKERRXX(ierr); };
      if(dd_vnm1_s_nodes[dd][dir] != NULL) { ierr = VecDestroy(dd_vnm1_s_nodes[dd][dir]) ; CHKERRXX(ierr); };
    }
  }

  if(ngbd_nm1!=NULL) { delete ngbd_nm1; }
  if(hierarchy_nm1!=NULL) { delete hierarchy_nm1; }
  if(nodes_nm1!=NULL) { p4est_nodes_destroy(nodes_nm1); }
  if(ghost_nm1!=NULL) { p4est_ghost_destroy(ghost_nm1); }
  if(p4est_nm1!=NULL) { p4est_destroy(p4est_nm1); }

  if(ngbd_n!=NULL) { delete ngbd_n; }
  if(hierarchy_n!=NULL) { delete hierarchy_n; }
  if(nodes_n!=NULL) { p4est_nodes_destroy(nodes_n); }
  if(ghost_n!=NULL) { p4est_ghost_destroy(ghost_n); }
  if(p4est_n!=NULL) { p4est_destroy(p4est_n); }

  if(sp_crit!=NULL) delete sp_crit;

  if(conn!=NULL) { my_p4est_brick_destroy(conn, &brick); }
}


void my_p4est_surfactant_t::set_no_surface_diffusion(const bool &flag)
{
  NO_SURFACE_DIFFUSION = flag;
}

void my_p4est_surfactant_t::set_no_surface_advection(const bool &flag)
{
  NO_SURFACE_ADVECTION = flag;
}


void my_p4est_surfactant_t::set_phi(Vec level_set, const bool& do_reinit)
{
  // Create PETSc error flag
  PetscErrorCode ierr;

  if(this->phi!=NULL) { ierr = VecDestroy(this->phi); CHKERRXX(ierr); }
  this->phi = level_set;

  my_p4est_level_set_t lsn(ngbd_n);
  if(do_reinit) { lsn.reinitialize_2nd_order(phi); }
  lsn.perturb_level_set_function(phi, EPS*dxyz_min);
}


void my_p4est_surfactant_t::set_phi(CF_DIM *level_set, const bool& do_reinit)
{
  // Create PETSc error flag
  PetscErrorCode ierr;

  if(phi!=NULL) { ierr = VecDestroy(this->phi); CHKERRXX(ierr); }
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &phi); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est_n, nodes_n, *level_set, phi);

  my_p4est_level_set_t lsn(ngbd_n);
  if(do_reinit) { lsn.reinitialize_2nd_order(phi); }
  lsn.perturb_level_set_function(phi, EPS*dxyz_min);
}


void my_p4est_surfactant_t::compute_num_band_nodes()
{
  // Create PETSc error flag
  PetscErrorCode ierr;

  const double *phi_p;
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);

  num_band_nodes = 0;
  for(p4est_locidx_t n=0; n<nodes_n->num_owned_indeps; ++n)
  {
    if(phi_band_gen.band_fn(phi_p[n])<=TOL_BAND_NODES*dxyz_diag)
    {
      ++num_band_nodes;
    }
  }
}


void my_p4est_surfactant_t::compute_normal()
{
  // Create PETSc error flag
  PetscErrorCode ierr;

  double *normal_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    if(normal[dir]!=NULL) { ierr = VecDestroy(normal[dir]); CHKERRXX(ierr); }
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &normal[dir]); CHKERRXX(ierr);
    ierr = VecGetArray(normal[dir], &normal_p[dir]); CHKERRXX(ierr);
  }

  const double *phi_p;
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);

  quad_neighbor_nodes_of_node_t qnnn;
  for(size_t i=0; i<ngbd_n->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n->get_layer_node(i);
    ngbd_n->get_neighbors(n, qnnn);
    normal_p[0][n] = qnnn.dx_central(phi_p);
    normal_p[1][n] = qnnn.dy_central(phi_p);
    ONLY3D(normal_p[2][n] = qnnn.dz_central(phi_p);)
    double norm = sqrt( SUMD( SQR(normal_p[0][n]), SQR(normal_p[1][n]), SQR(normal_p[2][n]) ) );

    normal_p[0][n] = norm<EPS ? 0.0 : normal_p[0][n]/norm;
    normal_p[1][n] = norm<EPS ? 0.0 : normal_p[1][n]/norm;
    ONLY3D(normal_p[2][n] = norm<EPS ? 0.0 : normal_p[2][n]/norm;)
  }

  for(int dir=0; dir<P4EST_DIM; ++dir) { ierr = VecGhostUpdateBegin(normal[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr); }

  for(size_t i=0; i<ngbd_n->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n->get_local_node(i);
    ngbd_n->get_neighbors(n, qnnn);
    normal_p[0][n] = qnnn.dx_central(phi_p);
    normal_p[1][n] = qnnn.dy_central(phi_p);
    ONLY3D(normal_p[2][n] = qnnn.dz_central(phi_p);)
    double norm = sqrt( SUMD( SQR(normal_p[0][n]), SQR(normal_p[1][n]), SQR(normal_p[2][n]) ) );

    normal_p[0][n] = norm<EPS ? 0.0 : normal_p[0][n]/norm;
    normal_p[1][n] = norm<EPS ? 0.0 : normal_p[1][n]/norm;
    ONLY3D(normal_p[2][n] = norm<EPS ? 0.0 : normal_p[2][n]/norm;)
  }
  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);

  for(int dir=0; dir<P4EST_DIM; ++dir) { ierr = VecGhostUpdateEnd(normal[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr); }

  for(int dir=0; dir<P4EST_DIM; ++dir) { ierr = VecRestoreArray(normal[dir], &normal_p[dir]); CHKERRXX(ierr); }
}


void my_p4est_surfactant_t::compute_curvature()
{
  // Create PETSc error flag
  PetscErrorCode ierr;

  // Compute divergence of the normal vector field
  Vec kappa_temp = NULL;
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &kappa_temp); CHKERRXX(ierr);
  double *kappa_p;
  ierr = VecGetArray(kappa_temp, &kappa_p); CHKERRXX(ierr);

  if(normal==NULL) compute_normal();
  const double *phi_p, *normal_p[P4EST_DIM];
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
  for(int dir=0; dir<P4EST_DIM; ++dir){ ierr = VecGetArrayRead(normal[dir], &normal_p[dir]); CHKERRXX(ierr); }

  quad_neighbor_nodes_of_node_t qnnn;
  for(size_t i=0; i<ngbd_n->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n->get_layer_node(i);
    kappa_p[n] = 0.0;
    if(phi_band_gen.band_fn(phi_p[n])<phi_band_gen.band_width*dxyz_diag)
    {
      ngbd_n->get_neighbors(n, qnnn);
      kappa_p[n] = SUMD( qnnn.dx_central(normal_p[0]), qnnn.dy_central(normal_p[1]), qnnn.dz_central(normal_p[2]) );
    }
  }

  ierr = VecGhostUpdateBegin(kappa_temp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  for(size_t i=0; i<ngbd_n->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n->get_local_node(i);
    kappa_p[n] = 0.0;
    if(phi_band_gen.band_fn(phi_p[n])<phi_band_gen.band_width*dxyz_diag)
    {
      ngbd_n->get_neighbors(n, qnnn);
      kappa_p[n] = SUMD( qnnn.dx_central(normal_p[0]), qnnn.dy_central(normal_p[1]), qnnn.dz_central(normal_p[2]) );
    }
  }

  ierr = VecGhostUpdateEnd(kappa_temp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
  for(int dir=0; dir<P4EST_DIM; ++dir) { ierr = VecRestoreArrayRead(normal[dir], &normal_p[dir]); CHKERRXX(ierr); }
  ierr = VecRestoreArray(kappa_temp, &kappa_p); CHKERRXX(ierr);

  // Constant extension (curvature is defined only on the interface)
  if(kappa!=NULL) { ierr = VecDestroy(kappa); CHKERRXX(ierr); }
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &kappa); CHKERRXX(ierr);
  my_p4est_level_set_t ls(ngbd_n);
  ls.extend_from_interface_to_whole_domain_TVD(phi, kappa_temp, kappa, ITER_EXTENSION);

  // Compute maximum absolute curvature
  max_abs_kappa = 0.0;
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
  const double *kappa_p_r;
  ierr = VecGetArrayRead(kappa, &kappa_p_r); CHKERRXX(ierr);
  for(p4est_locidx_t n=0; n<nodes_n->num_owned_indeps; ++n)
  {
    if(phi_band_gen.band_fn(phi_p[n])<dxyz_diag)
    {
      max_abs_kappa = MAX(max_abs_kappa, fabs(kappa_p_r[n]));
    }
  }
  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(kappa, &kappa_p_r); CHKERRXX(ierr);
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &max_abs_kappa, 1, MPI_DOUBLE, MPI_MAX, p4est_n->mpicomm); SC_CHECK_MPI(mpiret);

  // Check if the grid can resolve the maximum curvature, throw a warning if not
  if( 1.0/MAX(max_abs_kappa,EPS/dxyz_min) < 2.0*dxyz_min )
    ierr = PetscPrintf(p4est_n->mpicomm, "[WARNING] my_p4est_surfactant::compute_curvature: "
                                         "The current grid cannot resolve the max. absolute curvature.\n"); CHKERRXX(ierr);
}


void my_p4est_surfactant_t::set_Gamma(CF_DIM *Gamma_nm1_input, CF_DIM *Gamma_n_input)
{
  // Create PETSc error flag
  PetscErrorCode ierr;

  if(Gamma_nm1!=NULL) { ierr = VecDestroy(Gamma_nm1); CHKERRXX(ierr); }
  ierr = VecCreateGhostNodes(p4est_nm1, nodes_nm1, &Gamma_nm1); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est_nm1, nodes_nm1, *Gamma_nm1_input, Gamma_nm1);

  if(Gamma_n!=NULL) { ierr = VecDestroy(Gamma_n); CHKERRXX(ierr); }
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &Gamma_n); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est_n, nodes_n, *Gamma_n_input, Gamma_n);
}


void my_p4est_surfactant_t::set_velocities(CF_DIM **vnm1, CF_DIM **vn)
{
  // Create PETSc error flag
  PetscErrorCode ierr;

  // Set the velocity fields in the solver from the input continuous functions
  double *v_p;
  for(unsigned short dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecCreateGhostNodes(p4est_nm1, nodes_nm1, &vnm1_nodes[dir]); CHKERRXX(ierr);
    ierr = VecGetArray(vnm1_nodes[dir], &v_p); CHKERRXX(ierr);
    for(size_t n=0; n<nodes_nm1->indep_nodes.elem_count; ++n)
    {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, p4est_nm1, nodes_nm1, xyz);
      v_p[n] = (*vnm1[dir])(DIM(xyz[0], xyz[1], xyz[2]));
    }
    ierr = VecRestoreArray(vnm1_nodes[dir], &v_p); CHKERRXX(ierr);

    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &vn_nodes[dir]); CHKERRXX(ierr);
    ierr = VecGetArray(vn_nodes[dir], &v_p); CHKERRXX(ierr);
    for(size_t n=0; n<nodes_n->indep_nodes.elem_count; ++n)
    {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, p4est_n, nodes_n, xyz);
      v_p[n] = (*vn[dir])(DIM(xyz[0], xyz[1], xyz[2]));
    }
    ierr = VecRestoreArray(vn_nodes[dir], &v_p); CHKERRXX(ierr);
  }

  // Compute (and store) the second derivatives of the velocity field (for second-order interpolation later)
  compute_dd_vnm1();
  compute_dd_vn();
//  for(unsigned short dir = 0; dir < P4EST_DIM; ++dir)
//    for(unsigned short dd = 0; dd < P4EST_DIM; ++dd)
//    {
//      if(dd_vnm1_nodes[dd][dir] != NULL)
//      {
//          ierr = VecDestroy(dd_vnm1_nodes[dd][dir]); CHKERRXX(ierr);
//      }
//      if(dd_vn_nodes[dd][dir] != NULL)
//      {
//          ierr = VecDestroy(dd_vn_nodes[dd][dir]); CHKERRXX(ierr);
//      }

//      ierr = VecCreateGhostNodes(p4est_nm1, nodes_nm1, &dd_vnm1_nodes[dd][dir]); CHKERRXX(ierr);
//      ierr = VecCreateGhostNodes(p4est_n  , nodes_n  , &dd_vn_nodes  [dd][dir]); CHKERRXX(ierr);
//    }
//  ngbd_nm1->second_derivatives_central(vnm1_nodes, DIM(dd_vnm1_nodes[0], dd_vnm1_nodes[1], dd_vnm1_nodes[2]), P4EST_DIM);
//  ngbd_n  ->second_derivatives_central(vn_nodes,   DIM(dd_vn_nodes[0],   dd_vn_nodes[1],   dd_vn_nodes[2]  ), P4EST_DIM);

  // Adapt time stepping after update of velocity field
  dt_n = compute_adapted_dt_n();
}


void my_p4est_surfactant_t::compute_dd_vn()
{
  // Create PETSc error flag
  PetscErrorCode ierr;

  // Destroy previous data if it exists and compute second derivatives
  for(unsigned short dir = 0; dir < P4EST_DIM; ++dir)
    for(unsigned short dd = 0; dd < P4EST_DIM; ++dd)
    {
      if(dd_vn_nodes[dd][dir] != NULL)
      {
          ierr = VecDestroy(dd_vn_nodes[dd][dir]); CHKERRXX(ierr);
      }
      ierr = VecCreateGhostNodes(p4est_n, nodes_n, &dd_vn_nodes[dd][dir]); CHKERRXX(ierr);
    }
  ngbd_n->second_derivatives_central(vn_nodes, DIM(dd_vn_nodes[0], dd_vn_nodes[1], dd_vn_nodes[2]), P4EST_DIM);
}


void my_p4est_surfactant_t::compute_dd_vnm1()
{
  // Create PETSc error flag
  PetscErrorCode ierr;

  // Destroy previous data if it exists and compute second derivatives
  for(unsigned short dir = 0; dir < P4EST_DIM; ++dir)
    for(unsigned short dd = 0; dd < P4EST_DIM; ++dd)
    {
      if(dd_vnm1_nodes[dd][dir] != NULL)
      {
          ierr = VecDestroy(dd_vnm1_nodes[dd][dir]); CHKERRXX(ierr);
      }
      ierr = VecCreateGhostNodes(p4est_nm1, nodes_nm1, &dd_vnm1_nodes[dd][dir]); CHKERRXX(ierr);
    }
  ngbd_nm1->second_derivatives_central(vnm1_nodes, DIM(dd_vnm1_nodes[0], dd_vnm1_nodes[1], dd_vnm1_nodes[2]), P4EST_DIM);
}


void my_p4est_surfactant_t::compute_dd_vn_s()
{
  // Create PETSc error flag
  PetscErrorCode ierr;

  // Destroy previous data if it exists and compute second derivatives
  for(unsigned short dir = 0; dir < P4EST_DIM; ++dir)
    for(unsigned short dd = 0; dd < P4EST_DIM; ++dd)
    {
      if(dd_vn_s_nodes[dd][dir] != NULL)
      {
          ierr = VecDestroy(dd_vn_s_nodes[dd][dir]); CHKERRXX(ierr);
      }
      ierr = VecCreateGhostNodes(p4est_n, nodes_n, &dd_vn_s_nodes[dd][dir]); CHKERRXX(ierr);
    }
  ngbd_n->second_derivatives_central(vn_s_nodes, DIM(dd_vn_s_nodes[0], dd_vn_s_nodes[1], dd_vn_s_nodes[2]), P4EST_DIM);
}


void my_p4est_surfactant_t::compute_dd_vnm1_s()
{
  // Create PETSc error flag
  PetscErrorCode ierr;

  // Destroy previous data if it exists and compute second derivatives
  for(unsigned short dir = 0; dir < P4EST_DIM; ++dir)
    for(unsigned short dd = 0; dd < P4EST_DIM; ++dd)
    {
      if(dd_vnm1_s_nodes[dd][dir] != NULL)
      {
          ierr = VecDestroy(dd_vnm1_s_nodes[dd][dir]); CHKERRXX(ierr);
      }
      ierr = VecCreateGhostNodes(p4est_nm1, nodes_nm1, &dd_vnm1_s_nodes[dd][dir]); CHKERRXX(ierr);
    }
  ngbd_nm1->second_derivatives_central(vnm1_s_nodes, DIM(dd_vnm1_s_nodes[0], dd_vnm1_s_nodes[1], dd_vnm1_s_nodes[2]), P4EST_DIM);
}


void my_p4est_surfactant_t::compute_vnm1_s_from_phi_nm1(CF_DIM *ls_nm1, const bool& do_reinit)
{
  // Create PETSc error flag
  PetscErrorCode ierr;

  //  Create extension object and vnm1_s vector
  my_p4est_level_set_t ls_extend_nm1(ngbd_nm1);
  for(unsigned short dir=0; dir<P4EST_DIM; ++dir)
  {
    if(vnm1_s_nodes[dir]!=NULL) { ierr = VecDestroy(vnm1_s_nodes[dir]); CHKERRXX(ierr); }
    ierr = VecCreateGhostNodes(p4est_nm1, nodes_nm1, &vnm1_s_nodes[dir]); CHKERRXX(ierr);
  }

  // Sample phi_nm1, reinitialize and extend vnm1 from the interface defined by phi_nm1
  Vec phi_nm1_temp = NULL;
  ierr = VecCreateGhostNodes(p4est_nm1, nodes_nm1, &phi_nm1_temp); CHKERRXX(ierr);
  sample_cf_on_nodes(p4est_nm1, nodes_nm1, *ls_nm1, phi_nm1_temp);
  if(do_reinit)
  {
    ls_extend_nm1.reinitialize_2nd_order(phi_nm1_temp);
    ls_extend_nm1.perturb_level_set_function(phi_nm1_temp, EPS*dxyz_min);
  }
  for(unsigned short dir=0; dir<P4EST_DIM; ++dir)
    ls_extend_nm1.extend_from_interface_to_whole_domain_TVD(phi_nm1_temp, vnm1_nodes[dir], vnm1_s_nodes[dir], ITER_EXTENSION);
  ierr = VecDestroy(phi_nm1_temp); CHKERRXX(ierr);

  // Compute second derivatives of vnm1_s
  compute_dd_vnm1_s();
//  for(unsigned short dir = 0; dir < P4EST_DIM; ++dir)
//    for(unsigned short dd = 0; dd < P4EST_DIM; ++dd)
//    {
//      if(dd_vnm1_s_nodes[dd][dir] != NULL)
//      {
//          ierr = VecDestroy(dd_vnm1_s_nodes[dd][dir]); CHKERRXX(ierr);
//      }

//      ierr = VecCreateGhostNodes(p4est_nm1, nodes_nm1, &dd_vnm1_s_nodes[dd][dir]); CHKERRXX(ierr);
//    }
//  ngbd_nm1->second_derivatives_central(vnm1_s_nodes, DIM(dd_vnm1_s_nodes[0], dd_vnm1_s_nodes[1], dd_vnm1_s_nodes[2]), P4EST_DIM);

  // Compute str_nm1
  compute_stretching_term_nm1();
}


void my_p4est_surfactant_t::compute_vn_s()
{
  // Create PETSc error flag
  PetscErrorCode ierr;

  //  Create extension object and vn_s vector
  my_p4est_level_set_t ls_extend_n(ngbd_n);
  for(unsigned short dir=0; dir<P4EST_DIM; ++dir)
  {
    if(vn_s_nodes[dir]!=NULL) { ierr = VecDestroy(vn_s_nodes[dir]); CHKERRXX(ierr); }
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &vn_s_nodes[dir]); CHKERRXX(ierr);
  }

  // Extend vn from the interface defined by phi
  P4EST_ASSERT(phi!=NULL);
  for(unsigned short dir=0; dir<P4EST_DIM; ++dir)
    ls_extend_n.extend_from_interface_to_whole_domain_TVD(phi, vn_nodes[dir], vn_s_nodes[dir], ITER_EXTENSION);

  // Compute second derivatives of vn_s
  compute_dd_vn_s();
//  for(unsigned short dir = 0; dir < P4EST_DIM; ++dir)
//    for(unsigned short dd = 0; dd < P4EST_DIM; ++dd)
//    {
//      if(dd_vn_s_nodes[dd][dir] != NULL)
//      {
//          ierr = VecDestroy(dd_vn_s_nodes[dd][dir]); CHKERRXX(ierr);
//      }

//      ierr = VecCreateGhostNodes(p4est_n, nodes_n, &dd_vn_s_nodes[dd][dir]); CHKERRXX(ierr);
//    }
//  ngbd_n->second_derivatives_central(vn_s_nodes, DIM(dd_vn_s_nodes[0], dd_vn_s_nodes[1], dd_vn_s_nodes[2]), P4EST_DIM);

  // Compute str_n
  compute_stretching_term_n();
}


void my_p4est_surfactant_t::compute_initial_extended_velocities(CF_DIM *ls_nm1, const bool& do_reinit_nm1)
{
  compute_vnm1_s_from_phi_nm1(ls_nm1, do_reinit_nm1);
  compute_vn_s();
}

//void my_p4est_surfactant_t::compute_extended_velocities(CF_DIM *ls_nm1, CF_DIM *ls_n, const bool& do_reinit_nm1, const bool& do_reinit_n)
//{
//  // Create PETSc error flag
//  PetscErrorCode ierr;

//  my_p4est_level_set_t ls_extend_nm1(ngbd_nm1);
//  for(unsigned short dir=0; dir<P4EST_DIM; ++dir){
//    ierr = VecCreateGhostNodes(p4est_nm1, nodes_nm1, &vnm1_s_nodes[dir]); CHKERRXX(ierr); }
//  Vec phi_nm1_temp = NULL;
//  ierr = VecCreateGhostNodes(p4est_nm1, nodes_nm1, &phi_nm1_temp); CHKERRXX(ierr);
//  sample_cf_on_nodes(p4est_nm1, nodes_nm1, *ls_nm1, phi_nm1_temp);
//  if(do_reinit_nm1)
//  {//#else
//  void set_velocities(CF_2 **vnm1, CF_2 **vn);
//#endif
//    ls_extend_nm1.reinitialize_2nd_order(phi_nm1_temp);
//    ls_extend_nm1.perturb_level_set_function(phi_nm1_temp, EPS*dxyz_min);
//  }
//  for(unsigned short dir=0; dir<P4EST_DIM; ++dir)
//    ls_extend_nm1.extend_from_interface_to_whole_domain_TVD(phi_nm1_temp, vnm1_nodes[dir], vnm1_s_nodes[dir], ITER_EXTENSION);

//  my_p4est_level_set_t ls_extend_n(ngbd_n);
//  for(unsigned short dir=0; dir<P4EST_DIM; ++dir){
//    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &vn_s_nodes[dir]); CHKERRXX(ierr); }
//  if(ls_n==NULL)
//  {
//    for(unsigned short dir=0; dir<P4EST_DIM; ++dir)
//    {
//      ls_extend_n.extend_from_interface_to_whole_domain_TVD(phi, vn_nodes[dir], vn_s_nodes[dir], ITER_EXTENSION);
//    }
//  }
//  else
//  {
//    Vec phi_n_temp = NULL;
//    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &phi_n_temp); CHKERRXX(ierr);
//    sample_cf_on_nodes(p4est_n, nodes_n, *ls_n, phi_n_temp);
//    if(do_reinit_n)
//    {
//      ls_extend_n.reinitialize_2nd_order(phi_n_temp);
//      ls_extend_n.perturb_level_set_function(phi_n_temp, EPS*dxyz_min);
//    }
//    for(unsigned short dir=0; dir<P4EST_DIM; ++dir)
//    {
//      ls_extend_n.extend_from_interface_to_whole_domain_TVD(phi_n_temp, vn_nodes[dir], vn_s_nodes[dir], ITER_EXTENSION);
//    }
//  }

//  for(unsigned short dir = 0; dir < P4EST_DIM; ++dir)
//    for(unsigned short dd = 0; dd < P4EST_DIM; ++dd)
//    {
//      if(dd_vnm1_s_nodes[dd][dir] != NULL)
//      {
//          ierr = VecDestroy(dd_vnm1_s_nodes[dd][dir]); CHKERRXX(ierr);
//      }
//      if(dd_vn_s_nodes[dd][dir] != NULL)
//      {
//          ierr = VecDestroy(dd_vn_s_nodes[dd][dir]); CHKERRXX(ierr);
//      }

//      ierr = VecCreateGhostNodes(p4est_nm1, nodes_nm1, &dd_vnm1_s_nodes[dd][dir]); CHKERRXX(ierr);
//      ierr = VecCreateGhostNodes(p4est_n  , nodes_n  , &dd_vn_s_nodes  [dd][dir]); CHKERRXX(ierr);
//    }
//#ifdef P4_TO_P8
//  ngbd_nm1->second_derivatives_central(vnm1_nodes, dd_vnm1_s_nodes[0], dd_vnm1_s_nodes[1], dd_vnm1_s_nodes[2], P4EST_DIM);
//  ngbd_n  ->second_derivatives_central(vn_nodes,   dd_vn_s_nodes[0],   dd_vn_s_nodes[1],   dd_vn_s_nodes[2],   P4EST_DIM);
//#else
//  ngbd_nm1->second_derivatives_central(vnm1_nodes, dd_vnm1_s_nodes[0], dd_vnm1_s_nodes[1], P4EST_DIM);
//  ngbd_n  ->second_derivatives_central(vn_nodes,   dd_vn_s_nodes[0],   dd_vn_s_nodes[1],   P4EST_DIM);
//#endif

//  compute_stretching_term_nm1(ls_nm1);
//  compute_stretching_term_n();
//}


void my_p4est_surfactant_t::compute_stretching_term_nm1()
{
  // Create PETSc error flag
  PetscErrorCode ierr;

  // Create str_nm1 vector
  if(str_nm1!=NULL) { ierr = VecDestroy(str_nm1); CHKERRXX(ierr); }
  ierr = VecCreateGhostNodes(p4est_nm1, nodes_nm1, &str_nm1); CHKERRXX(ierr);
  double *str_nm1_p;
  ierr = VecGetArray(str_nm1, &str_nm1_p); CHKERRXX(ierr);

  // Read from vnm1_s
  const double *vnm1_s_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir){ ierr = VecGetArrayRead(vnm1_s_nodes[dir], &vnm1_s_p[dir]); CHKERRXX(ierr); }

  // Compute div(vnm1_s) for ghost layer nodes and store
  quad_neighbor_nodes_of_node_t qnnn;
  for(size_t i=0; i<ngbd_nm1->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd_nm1->get_layer_node(i);
    ngbd_nm1->get_neighbors(n, qnnn);
    str_nm1_p[n] = SUMD( qnnn.dx_central(vnm1_s_p[0]), qnnn.dy_central(vnm1_s_p[1]), qnnn.dz_central(vnm1_s_p[2]) );
  }

  // Start update of values in ghost layer nodes
  ierr = VecGhostUpdateBegin(str_nm1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  // Compute div(vnm1_s) for local nodes and store
  for(size_t i=0; i<ngbd_nm1->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd_nm1->get_local_node(i);
    ngbd_nm1->get_neighbors(n, qnnn);
    str_nm1_p[n] = SUMD( qnnn.dx_central(vnm1_s_p[0]), qnnn.dy_central(vnm1_s_p[1]), qnnn.dz_central(vnm1_s_p[2]) );
  }

  // Finish update of values in ghost layer nodes
  ierr = VecGhostUpdateEnd(str_nm1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  // Restore vnm1_s
  for(int dir=0; dir<P4EST_DIM; ++dir) { ierr = VecRestoreArrayRead(vnm1_s_nodes[dir], &vnm1_s_p[dir]); CHKERRXX(ierr); }
  ierr = VecRestoreArray(str_nm1, &str_nm1_p); CHKERRXX(ierr);
}


void my_p4est_surfactant_t::compute_stretching_term_n()
{
  // Create PETSc error flag
  PetscErrorCode ierr;

  // Create str_n vector
  if(str_n!=NULL) { ierr = VecDestroy(str_n); CHKERRXX(ierr); }
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &str_n); CHKERRXX(ierr);
  double *str_n_p;
  ierr = VecGetArray(str_n, &str_n_p); CHKERRXX(ierr);

  // Read from vn_s
  const double *vn_s_p[P4EST_DIM];
  for(int dir=0; dir<P4EST_DIM; ++dir){ ierr = VecGetArrayRead(vn_s_nodes[dir], &vn_s_p[dir]); CHKERRXX(ierr); }

  // Compute div(vn_s) for ghost layer nodes and store
  quad_neighbor_nodes_of_node_t qnnn;
  for(size_t i=0; i<ngbd_n->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n->get_layer_node(i);
    ngbd_n->get_neighbors(n, qnnn);
    str_n_p[n] = SUMD( qnnn.dx_central(vn_s_p[0]), qnnn.dy_central(vn_s_p[1]), qnnn.dz_central(vn_s_p[2]) );
  }

  // Start update of values in ghost layer nodes
  ierr = VecGhostUpdateBegin(str_n, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  // Compute div(vn_s) for local nodes and store
  for(size_t i=0; i<ngbd_n->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n->get_local_node(i);
    ngbd_n->get_neighbors(n, qnnn);
    str_n_p[n] = SUMD( qnnn.dx_central(vn_s_p[0]), qnnn.dy_central(vn_s_p[1]), qnnn.dz_central(vn_s_p[2]) );
  }

  // Finish update of values in ghost layer nodes
  ierr = VecGhostUpdateEnd(str_n, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  // Restore vnm1_s
  for(int dir=0; dir<P4EST_DIM; ++dir) { ierr = VecRestoreArrayRead(vn_s_nodes[dir], &vn_s_p[dir]); CHKERRXX(ierr); }
  ierr = VecRestoreArray(str_n, &str_n_p); CHKERRXX(ierr);
}


//#ifdef P4_TO_P8
//void my_p4est_surfactant_t::compute_stretching_term_n(CF_3 *ls_n)
//#else
//void my_p4est_surfactant_t::compute_stretching_term_n(CF_2 *ls_n)
//#endif
//{
//  // Create PETSc error flag
//  PetscErrorCode ierr;

//  Vec str_n_temp = NULL;
//  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &str_n_temp); CHKERRXX(ierr);
//  double *str_n_p;
//  ierr = VecGetArray(str_n_temp, &str_n_p); CHKERRXX(ierr);

//  const double *vn_s_p[P4EST_DIM];
//  for(int dir=0; dir<P4EST_DIM; ++dir){ ierr = VecGetArrayRead(vn_s_nodes[dir], &vn_s_p[dir]); CHKERRXX(ierr); }

//  quad_neighbor_nodes_of_node_t qnnn;
//  for(size_t i=0; i<ngbd_n->get_layer_size(); ++i)
//  {
//    p4est_locidx_t n = ngbd_n->get_layer_node(i);
//    ngbd_n->get_neighbors(n, qnnn);
//#ifdef P4_TO_P8
//    str_n_p[n] = qnnn.dx_central(vn_s_p[0]) + qnnn.dy_central(vn_s_p[1]) + qnnn.dz_central(vn_s_p[2]);
//#else
//    str_n_p[n] = qnnn.dx_central(vn_s_p[0]) + qnnn.dy_central(vn_s_p[1]);
//#endif
//  }

//  ierr = VecGhostUpdateBegin(str_n_temp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

//  for(size_t i=0; i<ngbd_n->get_local_size(); ++i)
//  {
//    p4est_locidx_t n = ngbd_n->get_local_node(i);
//    ngbd_n->get_neighbors(n, qnnn);
//#ifdef P4_TO_P8
//    str_n_p[n] = qnnn.dx_central(vn_s_p[0]) + qnnn.dy_central(vn_s_p[1]) + qnnn.dz_central(vn_s_p[2]);
//#else
//    str_n_p[n] = qnnn.dx_central(vn_s_p[0]) + qnnn.dy_central(vn_s_p[1]);
//#endif
//  }

//  ierr = VecGhostUpdateEnd(str_n_temp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

//  for(int dir=0; dir<P4EST_DIM; ++dir) { ierr = VecRestoreArrayRead(vn_s_nodes[dir], &vn_s_p[dir]); CHKERRXX(ierr); }
//  ierr = VecRestoreArray(str_n_temp, &str_n_p); CHKERRXX(ierr);

//  if(str_n!=NULL) { ierr = VecDestroy(str_n); CHKERRXX(ierr); }
//  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &str_n); CHKERRXX(ierr);

//  Vec from_loc, to_loc;
//  ierr = VecGhostGetLocalForm(str_n_temp, &from_loc);     CHKERRXX(ierr);
//  ierr = VecGhostGetLocalForm(str_n, &to_loc);       CHKERRXX(ierr);
//  ierr = VecCopy(from_loc, to_loc);
//  ierr = VecGhostRestoreLocalForm(str_n_temp, &from_loc); CHKERRXX(ierr);
//  ierr = VecGhostRestoreLocalForm(str_n, &to_loc);   CHKERRXX(ierr);
////  my_p4est_level_set_t ls(ngbd_n);

////  if(ls_n==NULL)
////  {
////    ls.extend_from_interface_to_whole_domain_TVD(phi, str_n_temp, str_n, 50);
////  }
////  else
////  {
////    Vec phi_n_temp = NULL;
////    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &phi_n_temp); CHKERRXX(ierr);
////    sample_cf_on_nodes(p4est_n, nodes_n, *ls_n, phi_n_temp);
////    ls.extend_from_interface_to_whole_domain_TVD(phi_n_temp, str_n_temp, str_n, ITER_EXTENSION);
////  }
//}


void my_p4est_surfactant_t::advect_interface_one_step()
{
  // Create PETSc error flag
  PetscErrorCode ierr;

  // Backtrack characteristics using the velocities and obtain departure points:
  trajectory_from_np1_to_nm1( p4est_n, nodes_n, ngbd_nm1, ngbd_n,
                              vnm1_nodes, dd_vnm1_nodes,
                              vn_nodes, dd_vn_nodes,
                              dt_nm1, dt_n,
                              xyz_dep_nm1, xyz_dep_n );
  sl_dep_pts_computed = true;

  // Compute second derivatives of phi for upcoming interpolation
  Vec dd_phi[P4EST_DIM];
  for(unsigned short dd = 0; dd < P4EST_DIM; ++dd) { ierr = VecCreateGhostNodes(p4est_n, nodes_n, &dd_phi[dd]); CHKERRXX(ierr); }
  ngbd_n->second_derivatives_central(phi, dd_phi);

  // Obtain phi_np1 at departure points from interpolation
  my_p4est_interpolation_nodes_t interp_n(ngbd_n);
  for(p4est_locidx_t n=0; n<nodes_n->num_owned_indeps; ++n)
  {
    double xyz_tmp[P4EST_DIM];

    xyz_tmp[0] = xyz_dep_n[0][n];
    xyz_tmp[1] = xyz_dep_n[1][n];
    ONLY3D(xyz_tmp[2] = xyz_dep_n[2][n];)
    interp_n.add_point(n, xyz_tmp);
  }

  Vec phi_np1 = NULL;
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &phi_np1); CHKERRXX(ierr);
  interp_n.set_input(phi, DIM(dd_phi[0], dd_phi[1], dd_phi[2]), quadratic_non_oscillatory);
  interp_n.interpolate(phi_np1);
  interp_n.clear();

  // Interpolation happened only for local nodes, so we need to update the ghost layer
  ierr = VecGhostUpdateBegin(phi_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (phi_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  // Reinitialize level-set function (will be needed to build cut-cells solving diffusion implicitly)
  my_p4est_level_set_t lsn(ngbd_n);
  lsn.reinitialize_2nd_order(phi_np1);
  lsn.perturb_level_set_function(phi_np1, EPS*dxyz_min);

  // Update phi <- phi_np1
  if(phi!=NULL) ierr = VecDestroy(phi); CHKERRXX(ierr);
  VecCreateGhostNodes(p4est_n, nodes_n, &phi);
  copy_ghosted_vec(phi_np1, phi);

  // Compute umber of band nodes for new phi
  compute_num_band_nodes();

  // Compute new normals, curvature and band
  compute_normal();
  compute_curvature();
  update_phi_band_vector();
}


double my_p4est_surfactant_t::compute_adapted_dt_n()
{
  // If there is no surface advection, just return the minimum grid size times CFL
  if(NO_SURFACE_ADVECTION)
    return CFL*dxyz_min;

  // Create PETSc error flag
  PetscErrorCode ierr;

  double new_dt_n = DBL_MAX;
  const double *v_p[P4EST_DIM], *phi_p;
  for (short dir = 0; dir < P4EST_DIM; ++dir) { ierr = VecGetArrayRead(vn_nodes[dir], &v_p[dir]); CHKERRXX(ierr); }
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);

  for (p4est_topidx_t tree_idx = p4est_n->first_local_tree; tree_idx <= p4est_n->last_local_tree; ++tree_idx)
  {
    p4est_tree_t* tree = p4est_tree_array_index(p4est_n->trees, tree_idx);
    for (size_t qq = 0; qq < tree->quadrants.elem_count; ++qq)
    {
      p4est_locidx_t quad_idx = tree->quadrants_offset + qq;
      double max_local_velocity_magnitude = -1.0;
      p4est_quadrant_t* quad = p4est_quadrant_array_index(&tree->quadrants, qq);
      bool is_quad_in_neg_domain = true;
      for (unsigned short child_idx = 0; child_idx < P4EST_CHILDREN; ++child_idx)
      {
        p4est_locidx_t node_idx = nodes_n->local_nodes[P4EST_CHILDREN*quad_idx + child_idx];

        if(phi_p[node_idx]>2.0*dxyz_diag)
        {
          is_quad_in_neg_domain = false;
          break;
        }

        double node_vel_mag = 0.0;
        for (unsigned short dir = 0; dir < P4EST_DIM; ++dir)
        {
          node_vel_mag += SQR(v_p[dir][node_idx]);
        }
        node_vel_mag = sqrt(node_vel_mag);
        max_local_velocity_magnitude = MAX(max_local_velocity_magnitude, node_vel_mag);
      }
      if(is_quad_in_neg_domain)
        new_dt_n = MIN(new_dt_n, (1.0/max_local_velocity_magnitude)*CFL*dxyz_min*((double) (1<<(sp_crit->max_lvl - quad->level))));
    }
  }

  for (short dir = 0; dir < P4EST_DIM; ++dir) { ierr = VecRestoreArrayRead(vn_nodes[dir], &v_p[dir]); CHKERRXX(ierr); }
  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);

  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &new_dt_n, 1, MPI_DOUBLE, MPI_MIN, p4est_n->mpicomm); SC_CHECK_MPI(mpiret);
  return new_dt_n;
}


void my_p4est_surfactant_t::compute_one_step_Gamma()
{
  // Create PETSc error flag
  PetscErrorCode ierr;

  // Define ratio of time steps for later use
  double alpha =   ( 2.0*dt_n + dt_nm1 ) / ( dt_n + dt_nm1 ) ;
  double beta  = - dt_n / ( dt_n + dt_nm1 ) ;
  double eta   =   ( dt_n + dt_nm1 ) / dt_nm1 ;
  double zeta  = - dt_n / dt_nm1 ;

  Vec Gamma_n_dep = NULL;
  Vec Gamma_nm1_dep = NULL;
  const double *Gamma_n_dep_p, *Gamma_nm1_dep_p;

  Vec str_n_dep = NULL;
  Vec str_nm1_dep = NULL;
  const double *str_n_dep_p, *str_nm1_dep_p;

  // NEED TO CREATE VEC WITH GAMMA DEP POINTS FOR LATER USE IN ITERATION

  if(NO_SURFACE_ADVECTION) // If no surface advection ...
  {
    // ... read departure values directly from Gamma_n ...
    ierr = VecGetArrayRead(Gamma_n, &Gamma_n_dep_p); CHKERRXX(ierr);

    if(STATIC_GRID)
    {
      // ... and also read departure values directly from Gamma_nm1 if grid has not changed ...
      P4EST_ASSERT(ghosts_are_equal(ghost_n, ghost_nm1));
      P4EST_ASSERT(nodes_are_equal(p4est_n->mpisize, nodes_n, nodes_nm1));
      ierr = VecGetArrayRead(Gamma_nm1, &Gamma_nm1_dep_p); CHKERRXX(ierr);
    }
    else
    {
      // ... interpolate values from nm1-grid to n-grid otherwise
      ierr = VecCreateGhostNodes(p4est_n, nodes_n, &Gamma_nm1_dep); CHKERRXX(ierr);
      my_p4est_interpolation_nodes_t interp_nm1(ngbd_nm1);
      interp_nm1.set_input(Gamma_nm1, linear); // TO-DO: SECOND DERIVATIVES AND QUADRATIC (?)
      for(p4est_locidx_t n=0; n<nodes_n->num_owned_indeps; ++n)
      {
        double xyz[P4EST_DIM];
        node_xyz_fr_n(n, p4est_n, nodes_n, xyz);
        interp_nm1.add_point(n, xyz);
      }
      interp_nm1.interpolate(Gamma_nm1_dep);
      interp_nm1.clear();
      ierr = VecGetArrayRead(Gamma_nm1_dep, &Gamma_nm1_dep_p); CHKERRXX(ierr);
    }
  }
  else
  {
    // Backtrack characteristics using the *extended* velocities and obtain departure points:
    trajectory_from_np1_to_nm1( p4est_n, nodes_n, ngbd_nm1, ngbd_n,
                                vnm1_s_nodes, dd_vnm1_s_nodes,
                                vn_s_nodes, dd_vn_s_nodes,
                                dt_nm1, dt_n,
                                xyz_dep_s_nm1, xyz_dep_s_n );
    sl_dep_pts_s_computed = true; // CHECK IF NECESSSARY, MIGHT BE ABLE TO DELETE THIS VARIABLE

    my_p4est_interpolation_nodes_t interp_n(ngbd_n);
    my_p4est_interpolation_nodes_t interp_nm1(ngbd_nm1);

    // TO-DO: Do it only for close nodes, then extrapolate
    std::vector<p4est_locidx_t> node_map;
    node_map.resize(num_band_nodes);
    const double *phi_p;
    ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
    int k=0;
    for(p4est_locidx_t n=0; n<nodes_n->num_owned_indeps; ++n)
    {
      if(phi_band_gen.band_fn(phi_p[n])<=TOL_BAND_NODES*dxyz_diag)
      {
        node_map[k] = n;
        k++;

        double xyz_tmp[P4EST_DIM];
        xyz_tmp[0] = xyz_dep_s_n[0][n];
        xyz_tmp[1] = xyz_dep_s_n[1][n];
        CODE3D(xyz_tmp[2] = xyz_dep_s_n[2][n];)
        interp_n.add_point(k, xyz_tmp);

        xyz_tmp[0] = xyz_dep_s_nm1[0][n];
        xyz_tmp[1] = xyz_dep_s_nm1[1][n];
        CODE3D(xyz_tmp[2] = xyz_dep_s_nm1[2][n];)
        interp_nm1.add_point(k, xyz_tmp);
      }
    }
    P4EST_ASSERT(k==num_band_nodes);
    ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);

    std::vector<double> Gamma_n_dep_band_nodes, str_n_dep_band_nodes;
    Gamma_n_dep_band_nodes.resize(num_band_nodes);
    str_n_dep_band_nodes.resize(num_band_nodes);

    // TRY PREVIOUS compute_one_step_Gamma() function but computing number of band nodes and printing for advection test

    std::vector<Vec> interp_inputs_n; interp_inputs_n.resize(0);
    interp_inputs_n.push_back(Gamma_n);
    interp_inputs_n.push_back(str_n);
    std::vector<Vec> interp_outputs_n; interp_outputs_n.resize(0);
    interp_outputs_n.push_back(Gamma_n_dep); // CHANGE OUTPUT TO PUT VALUES IN VEC
    interp_outputs_n.push_back(str_n_dep);
    interp_n.set_input(interp_inputs_n,linear); // TO-DO: SECOND DERIVATIVES AND QUADRATIC (?)
    interp_n.interpolate(interp_outputs_n);
    interp_inputs_n.resize(0);
    interp_outputs_n.resize(0);
    interp_n.clear();

    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &Gamma_n_dep); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &str_n_dep); CHKERRXX(ierr);


//    Vec Gamma_dep_nm1 = NULL;
//    Vec str_dep_nm1 = NULL;
//    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &Gamma_dep_nm1); CHKERRXX(ierr);
//    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &str_dep_nm1); CHKERRXX(ierr);
//    std::vector<Vec> interp_inputs_nm1; interp_inputs_nm1.resize(0);
//    interp_inputs_nm1.push_back(Gamma_nm1);
//    interp_inputs_nm1.push_back(str_nm1);
//    std::vector<Vec> interp_outputs_nm1; interp_outputs_nm1.resize(0);
//    interp_outputs_nm1.push_back(Gamma_dep_nm1);
//    interp_outputs_nm1.push_back(str_dep_nm1);
//    interp_nm1.set_input(interp_inputs_nm1,linear); // TO-DO: SECOND DERIVATIVES AND QUADRATIC (?)
//    interp_nm1.interpolate(interp_outputs_nm1);
//    interp_inputs_nm1.resize(0);
//    interp_outputs_nm1.resize(0);
//    interp_nm1.clear();



  }





//  my_p4est_interpolation_nodes_t Gamma_nm1_interp(ngbd_nm1);
//  Gamma_nm1_interp.set_input(Gamma_nm1, linear);
//  Vec Gamma_nm1_on_ngbd_n;
//  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &Gamma_nm1_on_ngbd_n); CHKERRXX(ierr);
//  for(p4est_locidx_t n=0; n<nodes_n->num_owned_indeps; ++n)
//  {
//    double xyz[P4EST_DIM];
//    node_xyz_fr_n(n, p4est_n, nodes_n, xyz);
//    Gamma_nm1_interp.add_point(n, xyz);
//  }
//  Gamma_nm1_interp.interpolate(Gamma_nm1_on_ngbd_n);
//  Gamma_nm1_interp.clear();

  // CHECK THIS, PROBABLY NOT NECESSARY
  //ierr = VecGhostUpdateBegin(Gamma_nm1_on_ngbd_n, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  //ierr = VecGhostUpdateEnd  (Gamma_nm1_on_ngbd_n, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  // Create the rhs Vec and fill it
  Vec rhs_Gamma_temp = NULL;
  double *rhs_Gamma_temp_p;
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &rhs_Gamma_temp); CHKERRXX(ierr);
  ierr = VecGetArray(rhs_Gamma_temp, &rhs_Gamma_temp_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(Gamma_n, &Gamma_n_dep_p  ); CHKERRXX(ierr);
  ierr = VecGetArrayRead(Gamma_nm1_on_ngbd_n, &Gamma_nm1_dep_p); CHKERRXX(ierr);
  for(size_t n = 0; n < nodes_n->indep_nodes.elem_count; ++n)
  {
    rhs_Gamma_temp_p[n] = ( alpha/dt_n - beta/dt_nm1 )*Gamma_n_dep_p[n] + ( beta/dt_nm1 )*Gamma_nm1_dep_p[n];
  }
  ierr = VecRestoreArray(rhs_Gamma_temp, &rhs_Gamma_temp_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(Gamma_n, &Gamma_n_dep_p  ); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(Gamma_nm1_on_ngbd_n, &Gamma_nm1_dep_p); CHKERRXX(ierr);

  // Extend the RHS data into definitive Vec
  Vec rhs_Gamma = NULL;
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &rhs_Gamma); CHKERRXX(ierr);
  my_p4est_level_set_t ls(ngbd_n);
  ls.extend_from_interface_to_whole_domain_TVD(phi, rhs_Gamma_temp, rhs_Gamma, ITER_EXTENSION);

  // Allocate Vec for Gamma_np1
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &Gamma_np1); CHKERRXX(ierr);

  if(NO_SURFACE_DIFFUSION) // Without diffusion, just multiply rhs
  {
    // NOTE: Need to update ghost nodes since there is no elliptic solver doing it without diffusion
    ierr = VecGhostUpdateBegin(rhs_Gamma, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (rhs_Gamma, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    scale_ghosted_vec(rhs_Gamma, dt_n/alpha);
    copy_ghosted_vec(rhs_Gamma, Gamma_np1);
  }
  else // With diffusion, solve elliptic problem
  {
    my_p4est_poisson_nodes_mls_t* solver_Gamma;
    solver_Gamma = new my_p4est_poisson_nodes_mls_t(ngbd_n);
    solver_Gamma->set_diag(alpha/dt_n);
    solver_Gamma->set_rhs(rhs_Gamma);
    solver_Gamma->set_mu(D_s);
    solver_Gamma->add_boundary(MLS_INTERSECTION, phi_band, DIM(NULL, NULL, NULL), NEUMANN, zero_cf, zero_cf); // TO-DO: Add second derivatives of phi_band
    solver_Gamma->set_wc(NEUMANN, zero_cf);
    solver_Gamma->set_integration_order(1);
    solver_Gamma->set_use_sc_scheme(0);
    solver_Gamma->preassemble_linear_system();
    solver_Gamma->solve(Gamma_np1, false, true, KSPBCGS, PCHYPRE);
    delete solver_Gamma;
  }

  // Extrapolate solution outside of the band
  my_p4est_level_set_t extrap(ngbd_n);
  extrap.extend_Over_Interface_TVD(phi_band, Gamma_np1); // TO-DO: add band of extrapolation

  // Destroy Vecs
  if(rhs_Gamma_temp != NULL) { ierr = VecDestroy(rhs_Gamma_temp); CHKERRXX(ierr); }
  if(rhs_Gamma      != NULL) { ierr = VecDestroy(rhs_Gamma); CHKERRXX(ierr); }
  if(Gamma_nm1_dep  != NULL) { ierr = VecDestroy(Gamma_nm1_dep); CHKERRXX(ierr); }
  if(Gamma_n_dep    != NULL) { ierr = VecDestroy(Gamma_n_dep);   CHKERRXX(ierr); }
}


//void my_p4est_surfactant_t::compute_one_step_Gamma()
//{
//  // Create PETSc error flag
//  PetscErrorCode ierr;

//  // Define ratio of time steps for later use
//  double alpha =   ( 2.0*dt_n + dt_nm1 ) / ( dt_n + dt_nm1 ) ;
//  double beta  = - dt_n / ( dt_n + dt_nm1 ) ;
//  double eta   =   ( dt_n + dt_nm1 ) / dt_nm1 ;
//  double zeta  = - dt_n / dt_nm1 ;

//  // Create the rhs Vec
//  Vec rhs_Gamma_temp = NULL;
//  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &rhs_Gamma_temp); CHKERRXX(ierr);

//  my_p4est_interpolation_nodes_t Gamma_nm1_interp(ngbd_nm1);
//  Gamma_nm1_interp.set_input(Gamma_nm1, linear);
//  Vec Gamma_nm1_on_ngbd_n;
//  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &Gamma_nm1_on_ngbd_n); CHKERRXX(ierr);
//  for(p4est_locidx_t n=0; n<nodes_n->num_owned_indeps; ++n)
//  {
//    double xyz[P4EST_DIM];
//    node_xyz_fr_n(n, p4est_n, nodes_n, xyz);
//    Gamma_nm1_interp.add_point(n, xyz);
//  }
//  Gamma_nm1_interp.interpolate(Gamma_nm1_on_ngbd_n);
//  Gamma_nm1_interp.clear();

//  // CHECK THIS, PROBABLY NOT NECESSARY
//  ierr = VecGhostUpdateBegin(Gamma_nm1_on_ngbd_n, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//  ierr = VecGhostUpdateEnd  (Gamma_nm1_on_ngbd_n, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

//  double *rhs_Gamma_temp_p;
//  const double *Gamma_n_p, *Gamma_nm1_on_ngbd_n_p;
//  ierr = VecGetArray(rhs_Gamma_temp, &rhs_Gamma_temp_p); CHKERRXX(ierr);
//  ierr = VecGetArrayRead(Gamma_n, &Gamma_n_p  ); CHKERRXX(ierr);
//  ierr = VecGetArrayRead(Gamma_nm1_on_ngbd_n, &Gamma_nm1_on_ngbd_n_p); CHKERRXX(ierr);
//  for(size_t n = 0; n < nodes_n->indep_nodes.elem_count; ++n)
//  {
//    rhs_Gamma_temp_p[n] = ( alpha/dt_n - beta/dt_nm1 )*Gamma_n_p[n] + ( beta/dt_nm1 )*Gamma_nm1_on_ngbd_n_p[n];
//  }
//  ierr = VecRestoreArray(rhs_Gamma_temp, &rhs_Gamma_temp_p); CHKERRXX(ierr);
//  ierr = VecRestoreArrayRead(Gamma_n, &Gamma_n_p  ); CHKERRXX(ierr);
//  ierr = VecRestoreArrayRead(Gamma_nm1_on_ngbd_n, &Gamma_nm1_on_ngbd_n_p); CHKERRXX(ierr);

//  // Extend the RHS data into definitive Vec
//  Vec rhs_Gamma = NULL;
//  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &rhs_Gamma); CHKERRXX(ierr);
//  my_p4est_level_set_t ls(ngbd_n);
//  ls.extend_from_interface_to_whole_domain_TVD(phi, rhs_Gamma_temp, rhs_Gamma, ITER_EXTENSION);

//// aaaaaaaaaaaaaaaaaaaaaaaaaaaa

////  const double *Gamma_nm1_on_n_p, *rhs_temp_p, *rhs_p, *phi_p, *phi_band_p;
////  ierr = VecGetArrayRead(Gamma_nm1_on_ngbd_n, &Gamma_nm1_on_n_p); CHKERRXX(ierr);
////  ierr = VecGetArrayRead(rhs_Gamma_temp, &rhs_temp_p); CHKERRXX(ierr);
////  ierr = VecGetArrayRead(rhs_Gamma, &rhs_p); CHKERRXX(ierr);
////  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
////  ierr = VecGetArrayRead(phi_band, &phi_band_p); CHKERRXX(ierr);
////  my_p4est_vtk_write_all_general(p4est_n, nodes_n, ghost_n,
////                                 P4EST_TRUE, P4EST_TRUE,
////                                 5, // number of VTK_NODE_SCALAR
////                                 0, // number of VTK_NODE_VECTOR_BY_COMPONENTS
////                                 0,                  // number of VTK_NODE_VECTOR_BLOCK
////                                 0,                  // number of VTK_CELL_SCALAR
////                                 0,                  // number of VTK_CELL_VECTOR_BY_COMPONENTS
////                                 0,                  // number of VTK_CELL_VECTOR_BLOCK
////                                 "/home/temprano/Output/p4est_surfactant/tests/2d/surface_diffusion_interface/SBDF2-FV/03_07/vtu/test",
////                                 VTK_NODE_SCALAR, "Gamma_nm1_on_ngbd_n", Gamma_nm1_on_n_p,
////                                 VTK_NODE_SCALAR, "rhs_temp", rhs_temp_p,
////                                 VTK_NODE_SCALAR, "rhs", rhs_p,
////                                 VTK_NODE_SCALAR, "phi", phi_p,
////                                 VTK_NODE_SCALAR, "phi_band", phi_band_p);
////  ierr = VecRestoreArrayRead(Gamma_nm1_on_ngbd_n, &Gamma_nm1_on_n_p); CHKERRXX(ierr);
////  ierr = VecRestoreArrayRead(rhs_Gamma_temp, &rhs_temp_p); CHKERRXX(ierr);
////  ierr = VecRestoreArrayRead(rhs_Gamma, &rhs_p); CHKERRXX(ierr);
////  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
////  ierr = VecRestoreArrayRead(phi_band, &phi_band_p); CHKERRXX(ierr);

//// aaaaaaaaaaaaaaaaaaaaaaaaaaaa

//  //---

//  // Allocate Vec for Gamma_np1
//  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &Gamma_np1); CHKERRXX(ierr);

//  if(NO_SURFACE_DIFFUSION) // Without diffusion, just multiply rhs
//  {
//    ierr = VecGhostUpdateBegin(rhs_Gamma, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//    ierr = VecGhostUpdateEnd  (rhs_Gamma, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
//    scale_ghosted_vec(rhs_Gamma, dt_n/alpha);
//    copy_ghosted_vec(rhs_Gamma, Gamma_np1);
//  }
//  else // With diffusion, solve elliptic problem
//  {
//    my_p4est_poisson_nodes_mls_t* solver_Gamma;
//    solver_Gamma = new my_p4est_poisson_nodes_mls_t(ngbd_n);
//    solver_Gamma->set_diag(alpha/dt_n);
//    solver_Gamma->set_rhs(rhs_Gamma);
//    solver_Gamma->set_mu(D_s);
//    solver_Gamma->add_boundary(MLS_INTERSECTION, phi_band, DIM(NULL, NULL, NULL), NEUMANN, zero_cf, zero_cf); // TO-DO: Add second derivatives of phi_band
//    solver_Gamma->set_wc(NEUMANN, zero_cf);
//    solver_Gamma->set_integration_order(1);
//    solver_Gamma->set_use_sc_scheme(0);
//    solver_Gamma->preassemble_linear_system();
//    solver_Gamma->solve(Gamma_np1, false, true, KSPBCGS, PCHYPRE);
//  }

//  // Extrapolate solution outside of the band
//  my_p4est_level_set_t extrap(ngbd_n);
//  extrap.extend_Over_Interface_TVD(phi_band, Gamma_np1); // TO-DO: add band of extrapolation

//  // Destroy objects
//  delete solver_Gamma;
//  ierr = VecDestroy(rhs_Gamma); CHKERRXX(ierr);
//  ierr = VecDestroy(rhs_Gamma_temp); CHKERRXX(ierr);
//  ierr = VecDestroy(Gamma_nm1_on_ngbd_n); CHKERRXX(ierr);
//}

/*void my_p4est_surfactant_t::compute_one_step_Gamma(time_integrator integ, bool use_SL, bool extending_rhs)
{
  // Create PETSc error flag
  PetscErrorCode ierr;

  // Define ratio of time steps for later use
  double omega = dt_n/dt_nm1;

  // Define the RHS vectors
  Vec rhs_Gamma      = NULL;
  Vec rhs_Gamma_temp = NULL;
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &rhs_Gamma_temp); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &rhs_Gamma);      CHKERRXX(ierr);

  if(NO_SURFACE_ADVECTION)
  {
    my_p4est_interpolation_nodes_t Gamma_nm1_interp(ngbd_nm1); Gamma_nm1_interp.set_input(Gamma_nm1, linear);

    Vec Gamma_nm1_on_ngbd_n;
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &Gamma_nm1_on_ngbd_n); CHKERRXX(ierr);

    for(p4est_locidx_t n=0; n<nodes_n->num_owned_indeps; ++n)
    {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, p4est_n, nodes_n, xyz);
      Gamma_nm1_interp.add_point(n, xyz);
    }
    Gamma_nm1_interp.interpolate(Gamma_nm1_on_ngbd_n);
    Gamma_nm1_interp.clear();

    ierr = VecGhostUpdateBegin(Gamma_nm1_on_ngbd_n, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (Gamma_nm1_on_ngbd_n, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    const double *Gamma_n_p, *Gamma_nm1_on_ngbd_n_p;
    ierr = VecGetArrayRead(Gamma_n, &Gamma_n_p); CHKERRXX(ierr);
    ierr = VecGetArrayRead(Gamma_nm1_on_ngbd_n, &Gamma_nm1_on_ngbd_n_p); CHKERRXX(ierr);
    double *rhs_temp_p;
    ierr = VecGetArray(rhs_Gamma_temp, &rhs_temp_p); CHKERRXX(ierr);

    for(size_t n=0; n<nodes_n->indep_nodes.elem_count; ++n)
    {
      switch(integ)
      {
        case IEEU1: {
          rhs_temp_p[n] = Gamma_n_p[n];
          break;
        }
        case SBDF2: {
          rhs_temp_p[n] = (1.0/(1.0+2.0*omega))*(   Gamma_n_p[n]*SQR(1.0+omega)   -
                                                  - Gamma_nm1_on_ngbd_n_p[n]*SQR(omega) );
          break;
        }
        case CNLF2: {
          throw std::runtime_error("Need to add difussion terms in previous instants");
          rhs_temp_p[n] = (dt_n*(1.0+omega))*( Gamma_n_p[n]  *( - (omega-1.0)/ dt_n             ) +
                                               Gamma_nm1_on_ngbd_n_p[n]*( SQR(omega)/(dt_n*(1.0+omega)) ) );
          break;
        }
        case MCNAB2: {
          throw std::runtime_error("Need to add difussion terms in previous instants");
          rhs_temp_p[n] = dt_n*( Gamma_n_p[n]  *( 1.0/ dt_n   ) +
                                 Gamma_nm1_on_ngbd_n_p[n]   );
          break;
        }
        default: throw std::invalid_argument("my_p4est_surfactant_t::compute_one_step_Gamma: please choose a valid time integrator");
      }
    }


    ierr = VecRestoreArrayRead(Gamma_n, &Gamma_n_p); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(Gamma_nm1_on_ngbd_n, &Gamma_nm1_on_ngbd_n_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(rhs_Gamma_temp, &rhs_temp_p); CHKERRXX(ierr);
  }
  else
  {
    if(use_SL)
    {
      // Backtrack characteristics using the EXTENDED velocity and obtain departure points:
      trajectory_from_np1_to_nm1(p4est_n, nodes_n, ngbd_nm1, ngbd_n,
                                 vnm1_s_nodes, dd_vnm1_s_nodes,
                                 vn_s_nodes, dd_vn_s_nodes,
                                 dt_nm1, dt_n,
                                 xyz_dep_s_nm1, xyz_dep_s_n);

      // Obtain Gamma and the stretching term at departure points from interpolation:
      my_p4est_interpolation_nodes_t interp_n(ngbd_n);
      my_p4est_interpolation_nodes_t interp_nm1(ngbd_nm1);

      // TO-DO: Do it only for close nodes, then extrapolate
      for(p4est_locidx_t n=0; n<nodes_n->num_owned_indeps; ++n)
      {
        double xyz_tmp[P4EST_DIM];

        xyz_tmp[0] = xyz_dep_s_n[0][n];
        xyz_tmp[1] = xyz_dep_s_n[1][n];
    #ifdef P4_TO_P8
        xyz_tmp[2] = xyz_dep_s_n[2][n];
    #endif
        interp_n.add_point(n, xyz_tmp);

        xyz_tmp[0] = xyz_dep_s_nm1[0][n];
        xyz_tmp[1] = xyz_dep_s_nm1[1][n];
    #ifdef P4_TO_P8
        xyz_tmp[2] = xyz_dep_s_nm1[2][n];
    #endif
        interp_nm1.add_point(n, xyz_tmp);
      }

      Vec Gamma_dep_n = NULL;
      Vec str_dep_n = NULL;
      ierr = VecCreateGhostNodes(p4est_n, nodes_n, &Gamma_dep_n); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est_n, nodes_n, &str_dep_n); CHKERRXX(ierr);
      std::vector<Vec> interp_inputs_n; interp_inputs_n.resize(0);
      interp_inputs_n.push_back(Gamma_n);
      interp_inputs_n.push_back(str_n);
      std::vector<Vec> interp_outputs_n; interp_outputs_n.resize(0);
      interp_outputs_n.push_back(Gamma_dep_n);
      interp_outputs_n.push_back(str_dep_n);
      interp_n.set_input(interp_inputs_n,linear);
      interp_n.interpolate(interp_outputs_n);
      interp_inputs_n.resize(0);
      interp_outputs_n.resize(0);
      interp_n.clear();

      Vec Gamma_dep_nm1 = NULL;
      Vec str_dep_nm1 = NULL;
      ierr = VecCreateGhostNodes(p4est_n, nodes_n, &Gamma_dep_nm1); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est_n, nodes_n, &str_dep_nm1); CHKERRXX(ierr);
      std::vector<Vec> interp_inputs_nm1; interp_inputs_nm1.resize(0);
      interp_inputs_nm1.push_back(Gamma_nm1);
      interp_inputs_nm1.push_back(str_nm1);
      std::vector<Vec> interp_outputs_nm1; interp_outputs_nm1.resize(0);
      interp_outputs_nm1.push_back(Gamma_dep_nm1);
      interp_outputs_nm1.push_back(str_dep_nm1);
      interp_nm1.set_input(interp_inputs_nm1,linear);
      interp_nm1.interpolate(interp_outputs_nm1);
      interp_inputs_nm1.resize(0);
      interp_outputs_nm1.resize(0);
      interp_nm1.clear();

      // Interpolation happened only for local nodes, so we need to update the ghost layer:
      ierr = VecGhostUpdateBegin(Gamma_dep_n,   INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(Gamma_dep_nm1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(str_dep_n,     INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateBegin(str_dep_nm1,   INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(Gamma_dep_n,   INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(Gamma_dep_nm1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(str_dep_n,     INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd(str_dep_nm1,   INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      // Compute the right-hand side of the linear system of equations from data at departure points:
      const double *Gamma_dep_nm1_p, *Gamma_dep_n_p, *str_dep_nm1_p, *str_dep_n_p;
      ierr = VecGetArrayRead(Gamma_dep_nm1 , &Gamma_dep_nm1_p  ); CHKERRXX(ierr);
      ierr = VecGetArrayRead(Gamma_dep_n   , &Gamma_dep_n_p  );   CHKERRXX(ierr);
      ierr = VecGetArrayRead(str_dep_nm1   , &str_dep_nm1_p  );   CHKERRXX(ierr);
      ierr = VecGetArrayRead(str_dep_n     , &str_dep_n_p  );     CHKERRXX(ierr);

      double *rhs_temp_p;
      ierr = VecGetArray(rhs_Gamma_temp, &rhs_temp_p); CHKERRXX(ierr);

      for(size_t n=0; n<nodes_n->indep_nodes.elem_count; ++n)
      {
        switch(integ)
        {
          case IEEU1: {
            rhs_temp_p[n] = dt_n*( Gamma_dep_n_p[n]  *( 1.0/dt_n - str_dep_n_p[n] ) );
            break;
          }
          case SBDF2: {
            rhs_temp_p[n] = (dt_n*(1.0+omega)/(1.0+2.0*omega))*( Gamma_dep_n_p[n]  *(  (1.0+omega)/dt_n              - (1.0+omega)*str_dep_n_p[n] ) +
                                                                 Gamma_dep_nm1_p[n]*( -(SQR(omega)/(1.0+omega))/dt_n + omega*str_dep_nm1_p[n]     )   );
            break;
          }
          case CNLF2: {
            rhs_temp_p[n] = (dt_n*(1.0+omega))*( Gamma_dep_n_p[n]  *( - str_dep_n_p[n] - (omega-1.0)/ dt_n             ) +
                                                 Gamma_dep_nm1_p[n]*(                     SQR(omega)/(dt_n*(1.0+omega)))   );
            break;
          }
          case MCNAB2: {
            rhs_temp_p[n] = dt_n*( Gamma_dep_n_p[n]  *( 1.0/ dt_n  - (1.0+0.5*omega)*str_dep_n_p[n]   ) +
                                   Gamma_dep_nm1_p[n]*(                   0.5*omega *str_dep_nm1_p[n] )   );
            break;
          }
          default: throw std::invalid_argument("my_p4est_surfactant_t::compute_one_step_Gamma: please choose a valid time integrator");
        }
      }

      ierr = VecRestoreArray(rhs_Gamma_temp, &rhs_temp_p); CHKERRXX(ierr);

      ierr = VecRestoreArrayRead(Gamma_dep_nm1 , &Gamma_dep_nm1_p ); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(Gamma_dep_n   , &Gamma_dep_n_p   ); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(str_dep_nm1   , &str_dep_nm1_p   ); CHKERRXX(ierr);
      ierr = VecRestoreArrayRead(str_dep_n     , &str_dep_n_p     ); CHKERRXX(ierr);
    }
    else
    {
      throw std::runtime_error("Still need to sort out some interpolator calls, the solver is not ready for FV spatial discretization yet. Please use a Semi-Lagrangian approach.");
      // TO-DO: Adapt to account for wall nodes with half-cells, incorporate 3D
  #ifdef P4_TO_P8
        throw std::runtime_error("The solver is not setup for 3D FV advection yet");
  #endif

  //    double *rhs_temp_p;
  //    const double *phi_p, *Gamma_n_p,*Gamma_nm1_p;
  //    ierr = VecGetArray(rhs_Gamma_temp, &rhs_temp_p); CHKERRXX(ierr);
  //    ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
  //    ierr = VecGetArrayRead(Gamma_n, &Gamma_n_p); CHKERRXX(ierr);
  //    ierr = VecGetArrayRead(Gamma_nm1, &Gamma_nm1_p); CHKERRXX(ierr);

  //    //-----------------------------------------------

  //    my_p4est_interpolation_nodes_t Gamma_nm1_interp(ngbd_nm1); Gamma_nm1_interp.set_input(Gamma_nm1, linear);
  //    my_p4est_interpolation_nodes_t v_nm1_s_interp(ngbd_nm1);   v_nm1_s_interp.set_input(vnm1_s_nodes, linear, P4EST_DIM);

  //    Vec Gamma_nm1_on_ngbd_n;
  //    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &Gamma_nm1_on_ngbd_n); CHKERRXX(ierr);
  //    Vec v_nm1_s_on_ngbd_n[P4EST_DIM];
  //    for(unsigned short dir = 0; dir < P4EST_DIM; ++dir) ierr = VecCreateGhostNodes(p4est_n, nodes_n, &v_nm1_s_on_ngbd_n[dir]); CHKERRXX(ierr);

  //    for(p4est_locidx_t n=0; n<nodes_n->num_owned_indeps; ++n)
  //    {
  //      double xyz[P4EST_DIM];
  //      node_xyz_fr_n(n, p4est_n, nodes_n, xyz);
  //      Gamma_nm1_interp.add_point(n, xyz);
  //      v_nm1_s_interp.add_point(n, xyz);
  //    }
  //    Gamma_nm1_interp.interpolate(Gamma_nm1_on_ngbd_n);
  //    v_nm1_s_interp.interpolate(v_nm1_s_on_ngbd_n);
  //    Gamma_nm1_interp.clear();
  //    v_nm1_s_interp.clear();

  //    ierr = VecGhostUpdateBegin(Gamma_nm1_on_ngbd_n, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  //    for(int dir=0; dir<P4EST_DIM; ++dir) { ierr = VecGhostUpdateBegin(v_nm1_s_on_ngbd_n[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr); }
  //    ierr = VecGhostUpdateEnd  (Gamma_nm1_on_ngbd_n, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  //    for(int dir=0; dir<P4EST_DIM; ++dir) { ierr = VecGhostUpdateEnd(v_nm1_s_on_ngbd_n[dir], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr); }

  //    const double *Gamma_nm1_on_ngbd_n_p;

  ////    char out_name[1024];
  ////    sprintf(out_name, "/home/temprano/Output/p4est_surfactant/tests/2d/advection_expansion/CNLF2-FV/%02d_%02d/vtu/test_%04d", sp_crit->min_lvl, sp_crit->max_lvl, iter);
  ////    const double *phi_band_p;
  ////    const double *vnm1_p[P4EST_DIM];
  ////    ierr = VecGetArrayRead(phi_band, &phi_band_p); CHKERRXX(ierr);
  ////    ierr = VecGetArrayRead(Gamma_nm1_on_ngbd_n, &Gamma_nm1_on_ngbd_n_p); CHKERRXX(ierr);
  ////    for(int dir=0; dir<P4EST_DIM; ++dir)
  ////      ierr = VecGetArrayRead(v_nm1_s_on_ngbd_n[dir], &vnm1_p[dir]); CHKERRXX(ierr);
  ////    my_p4est_vtk_write_all_general(p4est_n, nodes_n, ghost_n,
  ////                                   P4EST_TRUE, P4EST_TRUE,
  ////                                   3, // number of VTK_NODE_SCALAR
  ////                                   1, // number of VTK_NODE_VECTOR_BY_COMPONENTS
  ////                                   0,                  // number of VTK_NODE_VECTOR_BLOCK
  ////                                   0,                  // number of VTK_CELL_SCALAR
  ////                                   0,                  // number of VTK_CELL_VECTOR_BY_COMPONENTS
  ////                                   0,                  // number of VTK_CELL_VECTOR_BLOCK
  ////                                   out_name,
  ////                                   VTK_NODE_SCALAR, "Gamma_nm1", Gamma_nm1_on_ngbd_n_p,
  ////                                   VTK_NODE_VECTOR_BY_COMPONENTS, "v_nm1", vnm1_p[0],
  ////                                                                           vnm1_p[1],
  ////                                   VTK_NODE_SCALAR, "phi", phi_p,
  ////                                   VTK_NODE_SCALAR, "phi_band", phi_band_p);
  ////    ierr = VecRestoreArrayRead(Gamma_nm1_on_ngbd_n, &Gamma_nm1_on_ngbd_n_p); CHKERRXX(ierr);
  ////    ierr = VecRestoreArrayRead(phi_band, &phi_band_p); CHKERRXX(ierr);
  ////    for(int dir=0; dir<P4EST_DIM; ++dir)
  ////      ierr = VecRestoreArrayRead(v_nm1_s_on_ngbd_n[dir], &vnm1_p[dir]); CHKERRXX(ierr);

  //    //-----------------------------------------------

  //    my_p4est_interpolation_nodes_t Gamma_n_interp(ngbd_n);     Gamma_n_interp.set_input(Gamma_n, linear);
  //    my_p4est_interpolation_nodes_t v_n_s_interp(ngbd_n);       v_n_s_interp.set_input(vn_s_nodes, linear, P4EST_DIM);

  //    my_p4est_interpolation_nodes_t Gamma_nm1_on_ngbd_n_interp(ngbd_n); Gamma_nm1_on_ngbd_n_interp.set_input(Gamma_nm1_on_ngbd_n, linear);
  //    my_p4est_interpolation_nodes_t v_nm1_s_on_ngbd_n_interp(ngbd_n);   v_nm1_s_on_ngbd_n_interp.set_input(v_nm1_s_on_ngbd_n, linear, P4EST_DIM);

  //    ierr = VecGetArrayRead(Gamma_nm1_on_ngbd_n, &Gamma_nm1_on_ngbd_n_p); CHKERRXX(ierr);
  //    for(p4est_locidx_t n=0; n<nodes_n->num_owned_indeps; ++n)
  //    {
  //      if(phi_band_gen.band_fn(phi_p[n])<=2.5*dxyz_diag)
  //      {
  //        double F_n, F_nm1;

  //        double x_n  = node_x_fr_n(n, p4est_n, nodes_n);
  //        double y_n  = node_y_fr_n(n, p4est_n, nodes_n);
  //        double v_m0[P4EST_DIM], v_p0[P4EST_DIM], v_0m[P4EST_DIM], v_0p[P4EST_DIM];

  //        double x_vec[P4EST_DIM];
  //        v_n_s_interp(x_n-0.5*dxyz[0], y_n            , v_m0);
  //        v_n_s_interp(x_n+0.5*dxyz[0], y_n            , v_p0);
  //        v_n_s_interp(x_n            , y_n-0.5*dxyz[1], v_0m);
  //        v_n_s_interp(x_n            , y_n+0.5*dxyz[1], v_0p);

  //        F_n = -( - v_m0[0]*Gamma_n_interp(x_n-0.5*dxyz[0], y_n            )*dxyz[1] +
  //                   v_p0[0]*Gamma_n_interp(x_n+0.5*dxyz[0], y_n            )*dxyz[1] -
  //                   v_0m[1]*Gamma_n_interp(x_n            , y_n-0.5*dxyz[1])*dxyz[0] +
  //                   v_0p[1]*Gamma_n_interp(x_n            , y_n+0.5*dxyz[1])*dxyz[0]   )/(dxyz[0]*dxyz[1]);

  //        v_nm1_s_on_ngbd_n_interp(x_n-0.5*dxyz[0], y_n            , v_m0);
  //        v_nm1_s_on_ngbd_n_interp(x_n+0.5*dxyz[0], y_n            , v_p0);
  //        v_nm1_s_on_ngbd_n_interp(x_n            , y_n-0.5*dxyz[1], v_0m);
  //        v_nm1_s_on_ngbd_n_interp(x_n            , y_n+0.5*dxyz[1], v_0p);

  //        F_nm1 = -( - v_m0[0]*Gamma_nm1_on_ngbd_n_interp(x_n-0.5*dxyz[0], y_n            )*dxyz[1] +
  //                     v_p0[0]*Gamma_nm1_on_ngbd_n_interp(x_n+0.5*dxyz[0], y_n            )*dxyz[1] -
  //                     v_0m[1]*Gamma_nm1_on_ngbd_n_interp(x_n            , y_n-0.5*dxyz[1])*dxyz[0] +
  //                     v_0p[1]*Gamma_nm1_on_ngbd_n_interp(x_n            , y_n+0.5*dxyz[1])*dxyz[0]   )/(dxyz[0]*dxyz[1]);

  //        switch(integ)
  //        {
  //          case IEEU1: {
  //            rhs_temp_p[n] = Gamma_n_p[n] + F_n*dt_n;
  //            break;
  //          }
  //          case SBDF2: {
  //            rhs_temp_p[n] = (SQR(1.0+omega)/(1.0+2.0*omega))*Gamma_n_p[n] - (SQR(omega)/(1.0+2.0*omega))*Gamma_nm1_on_ngbd_n_p[n] +
  //                               dt_n*((1.0+omega)/(1.0+2.0*omega))*((1.0+omega)*F_n - omega*F_nm1);
  //            break;
  //          }
  //          case CNLF2: {
  //            rhs_temp_p[n] = (1.0-SQR(omega))*Gamma_n_p[n] + SQR(omega)*Gamma_nm1_on_ngbd_n_p[n] + dt_n*(1+omega)*F_n;
  //            break;
  //          }
  //          case MCNAB2: {
  //            rhs_temp_p[n] = Gamma_n_p[n] + dt_n*((1.0+0.5*omega)*F_n - 0.5*omega*F_nm1);
  //            break;
  //          }
  //          default: throw std::invalid_argument("my_p4est_surfactant_t::compute_one_step_Gamma: please choose a valid time integrator");
  //        }
  //      }
  //    }
  //    ierr = VecRestoreArrayRead(Gamma_nm1_on_ngbd_n, &Gamma_nm1_on_ngbd_n_p); CHKERRXX(ierr);

  //    ierr = VecGhostUpdateBegin(rhs_Gamma_temp,   INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  //    ierr = VecGhostUpdateEnd(rhs_Gamma_temp,   INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ////    my_p4est_vtk_write_all_general(p4est_n, nodes_n, ghost_n,
  ////                                   P4EST_TRUE, P4EST_TRUE,
  ////                                   2, // number of VTK_NODE_SCALAR
  ////                                   0, // number of VTK_NODE_VECTOR_BY_COMPONENTS
  ////                                   0,                  // number of VTK_NODE_VECTOR_BLOCK
  ////                                   0,                  // number of VTK_CELL_SCALAR
  ////                                   0,                  // number of VTK_CELL_VECTOR_BY_COMPONENTS
  ////                                   0,                  // number of VTK_CELL_VECTOR_BLOCK
  ////                                   "/home/temprano/Output/p4est_surfactant/tests/2d/advection_expansion/test",
  ////                                   VTK_NODE_SCALAR, "rhs_FV", rhs_FV_temp_p,
  ////                                   VTK_NODE_SCALAR, "phi", phi_p);

  //    ierr = VecRestoreArray(rhs_Gamma_temp, &rhs_temp_p); CHKERRXX(ierr);
  //    ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
  //    ierr = VecRestoreArrayRead(Gamma_n, &Gamma_n_p); CHKERRXX(ierr);
  //    ierr = VecRestoreArrayRead(Gamma_nm1, &Gamma_nm1_p); CHKERRXX(ierr);

  ////    // Extrapolation of the solution outside of the band (outwards)
  ////    my_p4est_level_set_t ls(ngbd_n);
  ////    ls.extend_Over_Interface_TVD(phi_band, q, 20, 0);

  //    ierr = VecDestroy(Gamma_nm1_on_ngbd_n); CHKERRXX(ierr);
  //    for(unsigned short dir = 0; dir<P4EST_DIM; ++dir) ierr = VecDestroy(v_nm1_s_on_ngbd_n[dir]); CHKERRXX(ierr);
    }
  }

  // Constant extension of the right-hand side // TO-DO: modify function to meet a tolerance within a band then stop
  if(extending_rhs) {
    my_p4est_level_set_t ls(ngbd_n);
    ls.extend_from_interface_to_whole_domain_TVD(phi, rhs_Gamma_temp, rhs_Gamma, ITER_EXTENSION);
  }
  else {
    Vec from_loc, to_loc;
    ierr = VecGhostGetLocalForm(rhs_Gamma_temp, &from_loc);     CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(rhs_Gamma, &to_loc);            CHKERRXX(ierr);
    ierr = VecCopy(from_loc, to_loc);
    ierr = VecGhostRestoreLocalForm(rhs_Gamma_temp, &from_loc); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(rhs_Gamma, &to_loc);        CHKERRXX(ierr);
  }

  // Solve for Gamma at t_np1 (still at the t_n grid!)
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &Gamma_np1); CHKERRXX(ierr);
  if(NO_SURFACE_DIFFUSION)
  {
    // With no surface diffusion, just copy the rhs into Gamma_np1
    Vec from_loc, to_loc;
    ierr = VecGhostGetLocalForm(rhs_Gamma, &from_loc);     CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(Gamma_np1, &to_loc);       CHKERRXX(ierr);
    ierr = VecCopy(from_loc, to_loc);
    ierr = VecGhostRestoreLocalForm(rhs_Gamma, &from_loc); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(Gamma_np1, &to_loc);   CHKERRXX(ierr);
  }
  else
  {
//    Vec dd_phi_band[P4EST_DIM];
//    for(unsigned short dd = 0; dd < P4EST_DIM; ++dd) { ierr = VecCreateGhostNodes(p4est_n, nodes_n, &dd_phi_band[dd]); CHKERRXX(ierr); }
//    ngbd_n->second_derivatives_central(phi_band, dd_phi_band);

    my_p4est_poisson_nodes_mls_t *solver_Gamma;
    solver_Gamma = new my_p4est_poisson_nodes_mls_t(ngbd_n);
    solver_Gamma->add_boundary(MLS_INTERSECTION, phi_band, DIM(NULL, NULL, NULL), NEUMANN, zero_cf, zero_cf);
    solver_Gamma->set_diag(1.0);
    solver_Gamma->set_rhs(rhs_Gamma);

    switch(integ)
    {
      case IEEU1: {
        solver_Gamma->set_mu(1.0);
        break;
      }
      case SBDF2: {
        solver_Gamma->set_mu((1.0+omega)*dt_n*D_s/(1.0+2.0*omega));
        break;
      }
      case CNLF2: {
        throw std::runtime_error("Need to add diagonal term");
        break;
      }
      case MCNAB2: {
        throw std::runtime_error("Need to add diagonal term");
        break;
      }
      default: throw std::invalid_argument("my_p4est_surfactant_t::compute_one_step_Gamma: please choose a valid time integrator");
    }

    solver_Gamma->set_integration_order(1);
    solver_Gamma->set_use_sc_scheme(0);
    solver_Gamma->set_wc(DIRICHLET, zero_cf);

    solver_Gamma->preassemble_linear_system();

    Vec from_loc, to_loc;
    ierr = VecGhostGetLocalForm(Gamma_n, &from_loc);     CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(Gamma_np1, &to_loc);       CHKERRXX(ierr);
    ierr = VecCopy(from_loc, to_loc);
    ierr = VecGhostRestoreLocalForm(Gamma_n, &from_loc); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(Gamma_np1, &to_loc);   CHKERRXX(ierr);

    solver_Gamma->solve(Gamma_np1, false, true, KSPBCGS, PCHYPRE);
    delete solver_Gamma;
  }
}*/


#ifdef P4_TO_P8
void my_p4est_surfactant_t::update_from_tn_to_tnp1()
#else
void my_p4est_surfactant_t::update_from_tn_to_tnp1(CF_DIM **vnp1)
#endif
{
  // Create PETSc error flag
  PetscErrorCode ierr;

  // Create forest for time tnp1, for now it's a copy of the forest at time tn
  p4est_t *p4est_np1 = NULL;
  p4est_np1 = p4est_copy(p4est_n, P4EST_FALSE);
  p4est_np1->connectivity = p4est_n->connectivity;
  p4est_np1->user_pointer = (void*)(sp_crit);

  // Create objects associated to the new forest
  p4est_ghost_t *ghost_np1 = NULL; ghost_np1 = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
  p4est_nodes_t *nodes_np1 = NULL; nodes_np1 = my_p4est_nodes_new(p4est_np1, ghost_np1);
  my_p4est_hierarchy_t *hierarchy_np1 = NULL; hierarchy_np1 = new my_p4est_hierarchy_t(p4est_np1, ghost_np1, &brick);
  my_p4est_node_neighbors_t *ngbd_np1 = NULL; ngbd_np1      = new my_p4est_node_neighbors_t(hierarchy_np1, nodes_np1);

  // Create a vector to store the level-set function at time tnp1 in the new grid
  Vec phi_np1 = NULL;
  ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);
  copy_ghosted_vec(phi, phi_np1);

  // Impose refinement + coarsening criteria and iterate if the grid changes
  bool grid_is_changing = true;
  int iter = 0;
  while(grid_is_changing)
  {
    grid_is_changing = sp_crit->refine_and_coarsen(p4est_np1, nodes_np1, phi_np1);

    if(grid_is_changing)
    {
      my_p4est_partition(p4est_np1, P4EST_FALSE, NULL);
      if(ghost_np1!=NULL)     { p4est_ghost_destroy(ghost_np1); } ghost_np1     = my_p4est_ghost_new(p4est_np1, P4EST_CONNECT_FULL);
      if(nodes_np1!=NULL)     { p4est_nodes_destroy(nodes_np1); } nodes_np1     = my_p4est_nodes_new(p4est_np1, ghost_np1);
      if(hierarchy_np1!=NULL) { delete hierarchy_np1; }           hierarchy_np1 = new my_p4est_hierarchy_t(p4est_np1, ghost_np1, &brick);
      if(ngbd_np1!=NULL)      { delete ngbd_np1; }                ngbd_np1      = new my_p4est_node_neighbors_t(hierarchy_np1, nodes_np1);

      my_p4est_interpolation_nodes_t interp_n(ngbd_n);
      for(p4est_locidx_t n=0; n<nodes_np1->num_owned_indeps; ++n)
      {
        double xyz[P4EST_DIM];
        node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);
        interp_n.add_point(n, xyz);
      }

      ierr = VecDestroy(phi_np1); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &phi_np1); CHKERRXX(ierr);
      interp_n.set_input(phi, linear);
      interp_n.interpolate(phi_np1);
      interp_n.clear();

      // Interpolation happened only for local nodes, so we need to update the ghost layer
      ierr = VecGhostUpdateBegin(phi_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd  (phi_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      iter++;
      if(iter>1+sp_crit->max_lvl-sp_crit->min_lvl)
      {
        ierr = PetscPrintf(p4est_np1->mpicomm, "[WARNING] my_p4est_surfactant_t::update_from_tn_to_tnp1: "
                                               "The grid update did not converge\n"); CHKERRXX(ierr);
        break;
      }
    }
  }

  // Reinitialize final level-set function
  my_p4est_level_set_t lsn(ngbd_np1);
  lsn.reinitialize_2nd_order(phi_np1);
  lsn.perturb_level_set_function(phi_np1, EPS*dxyz_min);

  // Slide Gamma_nm1 <- Gamma_n
  ierr = VecDestroy(Gamma_nm1); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &Gamma_nm1); CHKERRXX(ierr);
  copy_ghosted_vec(Gamma_n, Gamma_nm1);

  // Slide Gamma_n <- Gamma_np1, need to interpolate since Gamma_np1 is still sampled at Grid_n
  Vec dd_Gamma_np1[P4EST_DIM];
  for(unsigned short dir = 0; dir < P4EST_DIM; ++dir) { ierr = VecCreateGhostNodes(p4est_n, nodes_n, &dd_Gamma_np1[dir]); CHKERRXX(ierr); }
  ngbd_n->second_derivatives_central(Gamma_np1, dd_Gamma_np1);
  my_p4est_interpolation_nodes_t interp_n(ngbd_n);
  for(p4est_locidx_t n=0; n<nodes_np1->num_owned_indeps; ++n)
  {
    double xyz[P4EST_DIM];
    node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);
    interp_n.add_point(n, xyz);
  }
  ierr = VecDestroy(Gamma_n); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &Gamma_n); CHKERRXX(ierr);
  interp_n.set_input(Gamma_np1, DIM(dd_Gamma_np1[0], dd_Gamma_np1[1], dd_Gamma_np1[2]), quadratic);
  interp_n.interpolate(Gamma_n);
  interp_n.clear();
  ierr = VecGhostUpdateBegin(Gamma_n, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (Gamma_n, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for(unsigned short dir = 0; dir < P4EST_DIM; ++dir) { ierr = VecDestroy(dd_Gamma_np1[dir]); CHKERRXX(ierr); }

  // Destroy Gamma_np1
  ierr = VecDestroy(Gamma_np1); CHKERRXX(ierr);

  for (unsigned short dir=0; dir<P4EST_DIM; ++dir)
  {
    // Slide vnm1 <- vn
    ierr = VecDestroy(vnm1_nodes[dir]); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &vnm1_nodes[dir]); CHKERRXX(ierr);
    copy_ghosted_vec(vn_nodes[dir], vnm1_nodes[dir]);

    // Slide dd_vnm1 <- dd_vn
    for(unsigned short dd = 0; dd < P4EST_DIM; ++dd)
    {
      if(dd_vnm1_nodes[dd][dir] != NULL){ ierr = VecDestroy(dd_vnm1_nodes[dd][dir]); CHKERRXX(ierr); }
      ierr = VecCreateGhostNodes(p4est_n, nodes_n, &dd_vnm1_nodes[dd][dir]); CHKERRXX(ierr);
      copy_ghosted_vec(dd_vn_nodes[dd][dir], dd_vnm1_nodes[dd][dir]);
    }

    // Slide vnm1_s <- vn_s
    ierr = VecDestroy(vnm1_s_nodes[dir]); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_n, nodes_n, &vnm1_s_nodes[dir]); CHKERRXX(ierr);
    copy_ghosted_vec(vn_s_nodes[dir], vnm1_s_nodes[dir]);

    // Slide dd_vnm1_s <- dd_vn_s
    for(unsigned short dd = 0; dd < P4EST_DIM; ++dd)
    {
      if(dd_vnm1_s_nodes[dd][dir] != NULL) { ierr = VecDestroy(dd_vnm1_s_nodes[dd][dir]); CHKERRXX(ierr); }
      ierr = VecCreateGhostNodes(p4est_n, nodes_n, &dd_vnm1_s_nodes[dd][dir]); CHKERRXX(ierr);
      copy_ghosted_vec(dd_vn_s_nodes[dd][dir], dd_vnm1_s_nodes[dd][dir]);
    }

    // Sample vnp1 in the new tnp1 grid from prescribed data, put it in vn
    ierr = VecDestroy(vn_nodes[dir]); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &vn_nodes[dir]); CHKERRXX(ierr);
    double *v_p;
    ierr = VecGetArray(vn_nodes[dir], &v_p); CHKERRXX(ierr);
    for(size_t n=0; n<nodes_np1->indep_nodes.elem_count; ++n)
    {
      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, p4est_np1, nodes_np1, xyz);
      v_p[n] = (*vnp1[dir])(DIM(xyz[0], xyz[1], xyz[2]));
    }
    ierr = VecRestoreArray(vn_nodes[dir], &v_p); CHKERRXX(ierr);

    // Extend v_np1 in the new tnp1 grid, put it in vn_s
//    ierr = VecDestroy(vn_s_nodes[dir]); CHKERRXX(ierr);
//    ierr = VecCreateGhostNodes(p4est_np1, nodes_np1, &vn_s_nodes[dir]); CHKERRXX(ierr);
//    my_p4est_level_set_t ls_extend_np1(ngbd_np1);
//    ls_extend_np1.extend_from_interface_to_whole_domain_TVD(phi_np1, vn_nodes[dir], vn_s_nodes[dir], ITER_EXTENSION);
  }

  // Slide str_nm1 <- str_n
  ierr = VecDestroy(str_nm1); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est_n, nodes_n, &str_nm1); CHKERRXX(ierr);
  copy_ghosted_vec(str_n, str_nm1);

  // Slide phi <- phi_np1
  ierr = VecDestroy(phi); CHKERRXX(ierr);
  phi = phi_np1;

  // Slide Grid_nm1 <- Grid_n
  p4est_destroy(p4est_nm1);
  p4est_nm1 = p4est_n;
  p4est_nm1->connectivity = p4est_n->connectivity;
  p4est_nm1->user_pointer = (void*)(sp_crit);
  p4est_ghost_destroy(ghost_nm1); ghost_nm1 = ghost_n;
  p4est_nodes_destroy(nodes_nm1); nodes_nm1 = nodes_n;
  delete hierarchy_nm1; hierarchy_nm1 = hierarchy_n;
  delete ngbd_nm1; ngbd_nm1 = ngbd_n;

  // Slide Grid_n <- Grid_np1
  p4est_n = p4est_np1;
  ghost_n = ghost_np1;
  nodes_n = nodes_np1;
  hierarchy_n = hierarchy_np1;
  ngbd_n = ngbd_np1;

  // Compute number of band nodes
  compute_num_band_nodes();

  // Compute normals, curvature, and band
  compute_normal();
  compute_curvature();
  update_phi_band_vector();

  // Compute new dd_vn
  compute_dd_vn();

  // Compute new vn_s (also computes dd_vn_s and str_n)
  compute_vn_s();

  // SLIDE (NOT COMPUTE) SECOND DERIVATIVES FOR TNM1

//  for(unsigned short dir = 0; dir < P4EST_DIM; ++dir)
//    for(unsigned short dd = 0; dd < P4EST_DIM; ++dd)
//    {
//      ierr = VecDestroy(dd_vnm1_nodes[dd][dir]);   CHKERRXX(ierr);
//      ierr = VecDestroy(dd_vn_nodes[dd][dir]);     CHKERRXX(ierr);
//      ierr = VecDestroy(dd_vnm1_s_nodes[dd][dir]); CHKERRXX(ierr);
//      ierr = VecDestroy(dd_vn_s_nodes[dd][dir]);   CHKERRXX(ierr);

//      ierr = VecCreateGhostNodes(p4est_nm1, nodes_nm1, &dd_vnm1_nodes[dd][dir]); CHKERRXX(ierr);
//      ierr = VecCreateGhostNodes(p4est_n  , nodes_n  , &dd_vn_nodes  [dd][dir]); CHKERRXX(ierr);
//      ierr = VecCreateGhostNodes(p4est_nm1, nodes_nm1, &dd_vnm1_s_nodes[dd][dir]); CHKERRXX(ierr);
//      ierr = VecCreateGhostNodes(p4est_n  , nodes_n  , &dd_vn_s_nodes  [dd][dir]); CHKERRXX(ierr);
//    }
//#ifdef P4_TO_P8
//  ngbd_nm1->second_derivatives_central(vnm1_nodes, dd_vnm1_nodes[0], dd_vnm1_nodes[1], dd_vnm1_nodes[2], P4EST_DIM);
//  ngbd_n  ->second_derivatives_central(vn_nodes,   dd_vn_nodes[0],   dd_vn_nodes[1],   dd_vn_nodes[2],   P4EST_DIM);
//  ngbd_nm1->second_derivatives_central(vnm1_s_nodes, dd_vnm1_s_nodes[0], dd_vnm1_s_nodes[1], dd_vnm1_s_nodes[2], P4EST_DIM);
//  ngbd_n  ->second_derivatives_central(vn_s_nodes,   dd_vn_s_nodes[0],   dd_vn_s_nodes[1],   dd_vn_s_nodes[2],   P4EST_DIM);
//#else
//  ngbd_nm1->second_derivatives_central(vnm1_nodes, dd_vnm1_nodes[0], dd_vnm1_nodes[1], P4EST_DIM);
//  ngbd_n  ->second_derivatives_central(vn_nodes,   dd_vn_nodes[0],   dd_vn_nodes[1],   P4EST_DIM);
//  ngbd_nm1->second_derivatives_central(vnm1_s_nodes, dd_vnm1_s_nodes[0], dd_vnm1_s_nodes[1], P4EST_DIM);
//  ngbd_n  ->second_derivatives_central(vn_s_nodes,   dd_vn_s_nodes[0],   dd_vn_s_nodes[1],   P4EST_DIM);
//#endif

  // Update time steps
  dt_nm1 = dt_n;
  dt_n = compute_adapted_dt_n(); //dt_n = CFL*dxyz_min;// NEED TO ADD THIS WHEN ADVECTION IS ADDED: compute_adapted_dt_n();

  // Reset semi-lagrangian backtrace flag
  sl_dep_pts_computed = false;
  sl_dep_pts_s_computed = false;
}

void my_p4est_surfactant_t::set_D_s(const double& D_s)
{
  this->D_s = D_s;
}

void my_p4est_surfactant_t::set_dt_nm1(const double& dt_nm1)
{
  this->dt_nm1 = dt_nm1;
}


void my_p4est_surfactant_t::set_dt_n(const double& dt_n)
{
  this->dt_n = dt_n;
}


void my_p4est_surfactant_t::save_vtk(const char* name)
{
  // Create PETSc error flag
  PetscErrorCode ierr;

  // Keep count of the number of fields (scalars and vectors) that will be exported
  unsigned short count_node_scalars = 0;
  unsigned short count_node_vectors = 0;

  // Declare pointers to scalar fields
  const double *phi_p;      ++count_node_scalars;
  const double *phi_band_p; ++count_node_scalars;
  const double *kappa_p;    ++count_node_scalars;
  const double *Gamma_n_p;  ++count_node_scalars;
  const double *str_n_p;    ++count_node_scalars;
  ierr = VecGetArrayRead(phi,      &phi_p  );    CHKERRXX(ierr);
  ierr = VecGetArrayRead(phi_band, &phi_band_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(kappa,    &kappa_p);    CHKERRXX(ierr);
  ierr = VecGetArrayRead(Gamma_n,  &Gamma_n_p);  CHKERRXX(ierr);
  ierr = VecGetArrayRead(str_n,    &str_n_p);    CHKERRXX(ierr);

  // Declare pointers to vector fields
  const double *vn_p[P4EST_DIM];     ++count_node_vectors;
  const double *vn_s_p[P4EST_DIM];   ++count_node_vectors;
  const double *normal_p[P4EST_DIM]; ++count_node_vectors;
  for(unsigned short dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecGetArrayRead(vn_nodes[dir],   &vn_p[dir]);     CHKERRXX(ierr);
    ierr = VecGetArrayRead(vn_s_nodes[dir], &vn_s_p[dir]);   CHKERRXX(ierr);
    ierr = VecGetArrayRead(normal[dir],     &normal_p[dir]); CHKERRXX(ierr);
  }

  // Export in vtk format
  my_p4est_vtk_write_all_general(p4est_n, nodes_n, ghost_n,
                                 P4EST_TRUE, P4EST_TRUE,
                                 count_node_scalars, // number of VTK_NODE_SCALAR
                                 count_node_vectors, // number of VTK_NODE_VECTOR_BY_COMPONENTS
                                 0,                  // number of VTK_NODE_VECTOR_BLOCK
                                 0,                  // number of VTK_CELL_SCALAR
                                 0,                  // number of VTK_CELL_VECTOR_BY_COMPONENTS
                                 0,                  // number of VTK_CELL_VECTOR_BLOCK
                                 name,
                                 VTK_NODE_SCALAR, "phi",      phi_p,
                                 VTK_NODE_SCALAR, "phi_band", phi_band_p,
                                 VTK_NODE_SCALAR, "Gamma"   , Gamma_n_p,
                                 VTK_NODE_SCALAR, "kappa"   , kappa_p,
                                 VTK_NODE_VECTOR_BY_COMPONENTS, "n",   DIM(normal_p[0], normal_p[1], normal_p[2]),
                                 VTK_NODE_VECTOR_BY_COMPONENTS, "v",   DIM(vn_p[0],     vn_p[1],     vn_p[2]    ),
                                 VTK_NODE_VECTOR_BY_COMPONENTS, "v_s", DIM(vn_s_p[0],   vn_s_p[1],   vn_s_p[2]  ),
                                 VTK_NODE_SCALAR, "str", str_n_p
                                 );

  // Restore pointers to scalar fields
  ierr = VecRestoreArrayRead(phi,      &phi_p  );    CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(phi_band, &phi_band_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(kappa,    &kappa_p);    CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(Gamma_n,  &Gamma_n_p);  CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(str_n,    &str_n_p);    CHKERRXX(ierr);

  // Restore pointers to vector fields
  for(unsigned short dir=0; dir<P4EST_DIM; ++dir)
  {
    ierr = VecRestoreArrayRead(vn_nodes[dir],   &vn_p[dir]);     CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(vn_s_nodes[dir], &vn_s_p[dir]);   CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(normal[dir],     &normal_p[dir]); CHKERRXX(ierr);
  }
}
